# coding=utf-8
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
from .configuration_autoencoder import AutoencoderConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel,LlamaRMSNorm,LlamaMLP

logger = logging.get_logger(__name__)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

class AELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = LlamaMLP(config)
        self.layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Encoder(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.latent_size = config.latent_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_layers = nn.ModuleList([AELayer(config) for _ in range(config.num_encoder_layers)])
        self.num_stage_layers = config.num_encoder_layers // 2
        self.hidden_to_latent = nn.Linear(config.hidden_size, config.latent_size * 2)
        self.squeeze_layer = nn.Linear(self.patch_size * config.hidden_size, config.hidden_size)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        batch_size, seq_length = input_ids.shape
        num_patches = seq_length // self.patch_size
        input_ids = input_ids.reshape(batch_size * num_patches, self.patch_size)

        inputs_embeds = self.embed_tokens(input_ids)
        if self.training:
            inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)

        hidden_states = inputs_embeds

        for stage in range(2):
            for layer_idx in range(self.num_stage_layers):
                encoder_idx = stage * self.num_stage_layers + layer_idx
                encoder_layer = self.encoder_layers[encoder_idx]
                hidden_states = encoder_layer(hidden_states)

            if stage == 0:
                hidden_states = hidden_states.view(batch_size * num_patches, 1, -1)
                hidden_states = self.squeeze_layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        latent_states = self.hidden_to_latent(hidden_states)
        latent_states = latent_states.reshape(batch_size, num_patches, self.latent_size * 2)

        return latent_states


class Decoder(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.num_stage_layers = config.num_decoder_layers // 2
        
        self.latent_to_hidden = nn.Linear(config.latent_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList([AELayer(config) for _ in range(config.num_decoder_layers)])
        self.expand_layer = nn.Linear(config.hidden_size, self.patch_size * config.hidden_size)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        latent_states,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        batch_size, seq_length, latent_size = latent_states.shape
        hidden_states = self.latent_to_hidden(latent_states)

        for stage in range(2):
            for layer_idx in range(self.num_stage_layers):
                decoder_idx = stage * self.num_stage_layers + layer_idx
                decoder_layer = self.decoder_layers[decoder_idx]
                hidden_states = decoder_layer(hidden_states)

            if stage == 0:
                hidden_states = self.expand_layer(hidden_states)
                hidden_states = hidden_states.reshape(batch_size, seq_length * self.patch_size, -1)

        hidden_states = self.norm(hidden_states)
        logits = F.linear(hidden_states, self.lm_head_weight)
        return logits


class Autoencoder(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.patch_size = config.patch_size
        self.decoder.lm_head_weight = self.encoder.embed_tokens.weight
        self.ae_dropout = config.ae_dropout
        self.kl_clamp = config.kl_clamp
        self.kl_weight = config.kl_weight

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        input_ids = input_ids.reshape(-1, self.patch_size)
        if self.training:
            mask = torch.rand_like(input_ids.float()) > self.ae_dropout
            input_ids = input_ids * mask.long()

        latent_states = self.encoder(input_ids=input_ids)
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        latent_states = mean + eps * std
        latent_states = torch.nn.functional.dropout(latent_states, p=self.ae_dropout, training=self.training)

        kl_loss = 0.5 * (torch.pow(mean, 2) + torch.pow(std, 2) - 1 - log_std * 2)
        kl_loss = torch.clamp(kl_loss, min = self.kl_clamp)
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        logits = self.decoder(latent_states=latent_states).float()
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, self.config.vocab_size)
        labels = labels.view(-1).to(logits.device)
        loss = loss_fct(logits, labels) 
        if self.training:
            loss = loss * self.patch_size + kl_loss * self.kl_weight

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )

