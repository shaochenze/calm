# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration_calm import CALMConfig
from .modeling_calm import CALM, CustomCausalLMOutput
from .configuration_autoencoder import AutoencoderConfig
from .modeling_autoencoder import Autoencoder
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel,LlamaModel,LlamaRMSNorm
import random

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

class MLPBlock(nn.Module):
    """
    A single residual block for the MLP-based generative head.

    This block refines an input representation 'x' (e.g., a noise vector embedding)
    by conditioning it on a context vector 'y' (e.g., the Transformer's hidden state).
    It uses a gated MLP structure for effective feature fusion and transformation.
    """
    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(2 * channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True)
        )
        self.gate_act = nn.SiLU()
        self.down_proj = nn.Linear(channels, channels, bias=True)


    def forward(self, x, y):
        h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
        gate_proj, up_proj = torch.chunk(h, 2, dim = -1)
        gate_proj = self.gate_act(gate_proj)
        step = self.down_proj(gate_proj * up_proj)

        return x + step

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(model_channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(model_channels, model_channels, bias=True),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels, bias=True)
        )

    def forward(self, x):
        h = self.linears(self.in_ln(x))
        return h


class MLPGenerator(nn.Module):
    """
    MLP-based generative head. 
    This module takes a Transformer hidden state and a random noise vector as input
    and generates a continuous latent vector prediction. 
    It consists of a stack of MLPBlocks that iteratively refine the noise conditioned on the hidden state.    
    """
    def __init__(self, config):
        super().__init__()
        self.noise_size = config.noise_size
        self.noise_embd = nn.Linear(config.noise_size, config.hidden_size)
        self.hidden_embd = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_hidden = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm_noise = nn.LayerNorm(config.hidden_size, eps=1e-6)

        mlp_blocks = []
        for i in range(config.num_mlp_layers):
            mlp_blocks.append(MLPBlock(
                config.hidden_size,
            ))
        self.mlp_blocks = nn.ModuleList(mlp_blocks)
        self.final_layer = FinalLayer(config.hidden_size, config.latent_size)

    def initialize_weights(self):
        nn.init.constant_(self.final_layer.linears[-1].weight, 0)
        nn.init.constant_(self.final_layer.linears[-1].bias, 0)
        return
    
    def sample(self, hidden_states):

        # Prepare noise for sampling
        noise = torch.rand((*hidden_states.shape[:-1], self.noise_size),
                   dtype=hidden_states.dtype, device=hidden_states.device) - 0.5

        # Embed and normalize inputs
        noise_embds = self.norm_noise(self.noise_embd(noise))
        hidden_states = self.norm_hidden(self.hidden_embd(hidden_states))

        # Iteratively refine the noise embedding through the MLP blocks
        for block in self.mlp_blocks:
            noise_embds = block(noise_embds, hidden_states)

        # Final projection to get the latent vector prediction
        latent_prediction = self.final_layer(noise_embds)
        return latent_prediction

class EnergyTransformer(CALM):
    """
    The main Continuous Autoregressive Language Model (CALM).
    This model integrates a standard Transformer backbone with a continuous generative head.
    It operates by predicting continuous vectors, each representing a chunk of K tokens.
    This model is trained with a likelihood-free Energy Score objective.
    """
    config_class = CALMConfig 

    def __init__(self, config):
        super().__init__(config)
        self.ae_config = AutoencoderConfig.from_pretrained(config.ae_path)
        self.ae_model = Autoencoder.from_pretrained(
            config.ae_path,
            config=self.ae_config,
        )
        # Freeze the autoencoder weights during CALM training
        for param in self.ae_model.parameters():
            param.requires_grad = False
        self.ae_model.eval()

        self.transformer = LlamaModel(config)
        self.mlp_generator = MLPGenerator(config)
        self.generative_head = self.mlp_generator
        self.padding_idx = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.patch_size = config.patch_size

        # Input compression module: maps K token embeddings to a single vector
        self.embed_proj = nn.Sequential(
            nn.Linear(self.patch_size * config.hidden_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6)
        )
        # Initialize weights and apply final processing
        self.post_init()
        self.mlp_generator.initialize_weights()
        self.noise_size = config.noise_size
        self.beta = config.beta
        self.num_samples = config.num_samples

    def distance(self, x_1, x_2):
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1), self.beta)
    
    def energy_score(self, x, mean, log_std):
        n_x = x.shape[0]
        x_i = x.unsqueeze(1)  # (n_x, 1, batch_size, ...)
        x_j = x.unsqueeze(0)  # (1, n_x, batch_size, ...)
        distance_matrix = self.distance(x_i, x_j)
        distance_x = distance_matrix.sum(dim=(0,1)) / (n_x * (n_x - 1))

        std = torch.exp(log_std)
        n_y = 100
        eps = torch.randn((n_y, *mean.shape), device=mean.device)
        y = mean + eps * std  # (n_y, batch_size, ...)
        y = y

        x_ = x.reshape(n_x, 1, *x.shape[1:])  # (n_x, 1, batch_size, ...)
        y_ = y.reshape(1, n_y, *y.shape[1:])  # (1, n_y, batch_size, ...)
        distance_y = self.distance(x_, y_).mean(dim=(0, 1))
        
        score = distance_x - distance_y * 2
        return score

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        batch_size, seq_length = input_ids.size()
        patch_size = self.patch_size
        latent_length = seq_length // patch_size

        labels = labels[:, patch_size:]
        mask = labels.ne(-100)
        labels = labels[mask].unsqueeze(0)

        # Get ground-truth latent vector from the frozen Autoencoder 
        latent_states = self.ae_model.encoder(input_ids=labels)
        latent_states = latent_states.squeeze(0)
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)

        # Prepare Transformer input
        inputs_embeds = self.transformer.embed_tokens(input_ids).reshape(batch_size, latent_length, -1)[:, :-1, :]
        inputs_embeds = self.embed_proj(inputs_embeds)

        # Get hidden states from the Transformer backbone
        outputs = self.transformer(inputs_embeds = inputs_embeds)
        hidden_states = outputs[0]
        patch_mask = mask.reshape(batch_size, latent_length-1, patch_size)[:, :, 0]
        hidden_states = hidden_states[patch_mask]

        # Generate predictions with the MLP Generator
        hidden_states_repeated = hidden_states.unsqueeze(0).repeat(self.num_samples, 1, 1)
        latent_predictions = self.generative_head.sample(hidden_states_repeated)

        # Compute the energy loss
        loss = - self.energy_score(latent_predictions, mean, log_std)
        loss = loss.mean()

        # Brier score is only calculated during evaluation
        if not self.training:
            return self.eval_brier(latent_predictions, input_ids[:, patch_size:], outputs, loss)

        return CustomCausalLMOutput(
            loss=loss,
        )
