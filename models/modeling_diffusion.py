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
from .diffusion import create_diffusion
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, config):
        super(DiffLoss, self).__init__()
        target_channels = config.latent_size
        z_channels = config.hidden_size
        width = config.hidden_size
        depth = config.num_mlp_layers
        num_sampling_steps = '100'
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z):
        target = target.reshape(-1, target.size(-1))
        z = z.reshape(-1, z.size(-1))
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        return loss

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        input_shape = z.shape
        z = z.reshape(-1, z.size(-1))
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )
        sampled_token_latent = sampled_token_latent.reshape(*input_shape[:-1], sampled_token_latent.size(-1))
        return sampled_token_latent

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c
        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)

class DiffusionTransformer(CALM):
    config_class = CALMConfig 

    def __init__(self, config):
        super().__init__(config)
        self.ae_config = AutoencoderConfig.from_pretrained(config.ae_path)
        self.ae_model = Autoencoder.from_pretrained(
            config.ae_path,
            config=self.ae_config,
        )
        for param in self.ae_model.parameters():
            param.requires_grad = False
        self.ae_model.eval()

        self.transformer = LlamaModel(config)
        self.generative_head = DiffLoss(config)
        self.padding_idx = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.patch_size = config.patch_size
        self.embed_proj = nn.Sequential(
            nn.Linear(self.patch_size * config.hidden_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6)
        )
        # Initialize weights and apply final processing
        self.post_init()
        self.generative_head.net.initialize_weights()
        self.noise_size = config.noise_size

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

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

        hidden_states_repeated = hidden_states.unsqueeze(0).repeat(self.num_samples, 1, 1)
        eps = torch.randn((self.num_samples, *mean.shape), device=mean.device)
        std = torch.exp(log_std)
        latent_states_repeated = mean + eps * std
        loss = self.generative_head(z=hidden_states_repeated, target=latent_states_repeated)

        loss = loss.mean()

        # Brier score is only calculated during evaluation
        if not self.training:
            latent_predictions = self.generative_head.sample(hidden_states_repeated[:2])
            return self.eval_brier(latent_predictions, input_ids[:, patch_size:], outputs, loss)

        return CustomCausalLMOutput(
            loss=loss,
        )

