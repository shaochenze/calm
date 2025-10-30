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
from .configuration_autoencoder import AutoencoderConfig
from .modeling_autoencoder import Autoencoder
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel,LlamaModel,LlamaRMSNorm
import random

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

@dataclass
class CustomCausalLMOutput(CausalLMOutputWithPast):
    brier1: torch.FloatTensor = None
    brier2: torch.FloatTensor = None
    brier3: torch.FloatTensor = None
    brier4: torch.FloatTensor = None
    
class CALM(LlamaPreTrainedModel):
    """
    The main Continuous Autoregressive Language Model (CALM).
    This model integrates a standard Transformer backbone with a continuous generative head.
    It operates by predicting continuous vectors, each representing a chunk of K tokens.
    """
    config_class = CALMConfig 

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @torch.no_grad()
    def eval_brier(self, latent_predictions, targets, outputs, loss):
        """
        Calculates a likelihood-free estimate of the Brier score.
        The Brier score is estimated using the formula: E[1{x1=y} + 1{x2=y} - 1{x1=x2}],
        where x1 and x2 are two independent samples from the model, and y is the target.

        Args:
            latent_predictions (torch.Tensor): Two sets of latent predictions from the model.
                                               Shape: [n>2, batch_size, latent_length, latent_dim].
            targets (torch.Tensor): The ground truth token ids. Shape: [batch_size, seq_length].
            outputs: The output object from the transformer, containing past_key_values.
            loss: The energy loss for latent vectors.
        """
        max_eval_length = 4
        patch_size = self.patch_size
        batch_size = targets.shape[0]
        seq_length = targets.shape[1] // patch_size
        targets = targets.reshape(batch_size, seq_length, patch_size)

        # Use the first two samples for evaluation
        latent_predictions = latent_predictions[:2].reshape(2, batch_size, seq_length, latent_predictions.size(-1))
        logits_1 = self.ae_model.decoder(latent_states=latent_predictions[0])
        logits_2 = self.ae_model.decoder(latent_states=latent_predictions[1])
        predictions_1 = torch.argmax(logits_1, dim=-1).reshape(batch_size, seq_length, patch_size)
        predictions_2 = torch.argmax(logits_2, dim=-1).reshape(batch_size, seq_length, patch_size)

        # CASE 1: The model's patch size is 4 or more.
        # In this case, each generation step produces enough tokens to calculate brier-4 directly.
        if patch_size >= max_eval_length:
            acc_1 = torch.cumprod((predictions_1 == targets).float(), dim = -1)
            acc_2 = torch.cumprod((predictions_2 == targets).float(), dim = -1)
            var = torch.cumprod((predictions_1 == predictions_2).float(), dim = -1)
            brier_estimations = (acc_1 + acc_2 - var).mean(dim=(0,1))
            
        # CASE 2: The model's patch size is less than 4.
        # We need to auto-regressively generate multiple patches to get a 4-token sequence.
        else:
            # how many steps are needed to produce 4 tokens.
            num_steps_to_cover = math.ceil(max_eval_length / patch_size)

            # --- Calculate the accuracy part of the Brier score (1{x=y}) ---
            predictions_1_cat = torch.cat([predictions_1[:, i:-(num_steps_to_cover-i), :] for i in range(num_steps_to_cover)], dim = -1)
            predictions_2_cat = torch.cat([predictions_2[:, i:-(num_steps_to_cover-i), :] for i in range(num_steps_to_cover)], dim = -1)
            targets_cat = torch.cat([targets[:, i:-(num_steps_to_cover-i), :] for i in range(num_steps_to_cover)], dim = -1)
            acc_1 = torch.cumprod((predictions_1_cat == targets_cat).float(), dim = -1)[:, :, :max_eval_length]
            acc_2 = torch.cumprod((predictions_2_cat == targets_cat).float(), dim = -1)[:, :, :max_eval_length]

            global_cache = outputs.past_key_values
            brier_estimations = []

            # --- Autoregressive Estimation of Uncertainty Term (1{x1=x2}) ---
            # Loop over every possible starting position in the sequence.
            for i in range(seq_length - num_steps_to_cover):
                prefix_same = torch.ones(batch_size, dtype=torch.bool, device=latent_predictions.device)
                token_same = torch.empty(batch_size, max_eval_length, dtype=torch.bool, device=latent_predictions.device)
                for j in range(num_steps_to_cover):
                    if j == 0:
                        next_tokens = torch.stack((predictions_1[:, i, :], predictions_2[:, i, :]), dim = 0)
                        current_cache = tuple(tuple(x[:, :, :i+1, :] for x in layer_cache) for layer_cache in global_cache)
                    else:
                        inputs_embeds = self.transformer.embed_tokens(current_input).reshape(batch_size, -1)
                        inputs_embeds = self.embed_proj(inputs_embeds)
                        outputs = self.transformer(
                                inputs_embeds=inputs_embeds.unsqueeze(dim = 1),
                                past_key_values=current_cache,
                                use_cache=True
                            )
                        # Prepare hidden state for the MLP generator (repeat for 2 samples).
                        hidden_states = outputs[0].unsqueeze(0).repeat(2, 1, 1, 1)
                        latent_prediction_step = self.generative_head.sample(hidden_states).reshape(2 * batch_size, 1, -1)
                        logits = self.ae_model.decoder(latent_states=latent_prediction_step)
                        next_tokens = torch.argmax(logits, dim=-1).reshape(2, batch_size, patch_size)
                        current_cache = outputs.past_key_values

                    # Optimization: The KV cache is only updated along the path of the first sample (x1).
                    current_input = next_tokens[0]

                    if j != num_steps_to_cover - 1:
                        window_size = patch_size
                    else:
                        # The final patch might be shorter than a full patch_size
                        window_size = 4 - (patch_size * (num_steps_to_cover - 1))

                    start_idx = j * patch_size
                    end_idx = start_idx + window_size
                    this_patch_token_same = (next_tokens[0, :, :window_size] == next_tokens[1, :, :window_size])
                    
                    # Store the result
                    token_same[:, start_idx:end_idx] = this_patch_token_same

                    # Update the prefix tracker
                    prefix_same &= torch.all(this_patch_token_same, dim=-1)

                    if not torch.any(prefix_same):
                        # Fill the rest of the token sameness tensor with False
                        if end_idx < max_eval_length:
                            token_same[:, end_idx:] = False
                        break

                # two_samples_are_same needs to be reshaped to align with the token-level accuracies.
                collision_term = torch.cumprod(token_same.float(), dim=-1)
                brier_estimation = acc_1[:, i, :] + acc_2[:, i, :] - collision_term
                brier_estimations.append(brier_estimation)

            brier_estimations = torch.stack(brier_estimations, dim = 0).mean(dim=(0,1))

        return CustomCausalLMOutput(
            loss=loss,
            brier1=brier_estimations[0],
            brier2=brier_estimations[1],
            brier3=brier_estimations[2],
            brier4=brier_estimations[3],
        )
        

    @torch.no_grad()
    def temperature_sampling(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 0.5,
        num_samples: int = 200,
    ):
        """
        Generates a patch of tokens using the approximate temperature sampling algorithm.

        Args:
            hidden_states (torch.Tensor): The context hidden states from the transformer.
                                          Shape: (..., hidden_size).
            temperature (float): The sampling temperature, T. Must be the reciprocal of an integer (e.g., 1/2, 1/3, 1/4).
            num_samples (int): The number of samples (N) to generate for the pool.

        Returns:
            torch.LongTensor: The generated patch of token IDs.
                              Shape: (..., patch_size).
        """

        # Validate temperature and calculate n.
        if not (0 < temperature <= 1.0):
            raise ValueError("Temperature must be in the range (0, 1].")
        
        inv_temp = 1.0 / temperature
        # Check if 1/T is very close to an integer.
        if not math.isclose(inv_temp, round(inv_temp), rel_tol=1e-9, abs_tol=0.0):
             raise ValueError(f"Temperature must be the reciprocal of an integer. "
                              f"Got T={temperature}, which corresponds to n={inv_temp:.4f}, not an integer.")
        
        n_initial = int(round(inv_temp))

        hidden_shape = hidden_states.shape
        hidden_size = hidden_shape[-1]
        hidden_states = hidden_states.reshape(-1, hidden_size)
        batch_size = hidden_states.shape[0]

        if temperature == 1.0:
            # For T=1, we just need one sample from the base generator.
            latent_predictions = self.generative_head.sample(hidden_states)
            logits = self.ae_model.decoder(latent_predictions.unsqueeze(1))
            outputs = torch.argmax(logits, dim=-1)
            
            # Reshape to the original input shape with the last dimension replaced by patch_size.
            output_shape = hidden_shape[:-1] + (self.patch_size,)
            return outputs.reshape(output_shape)


        # Generate N candidate latent vectors from the generator.
        hidden_states_repeated = hidden_states.unsqueeze(1).repeat(1, num_samples, 1)
        latent_predictions = self.generative_head.sample(hidden_states_repeated)

        # Decode latent vectors into token IDs
        logits = self.ae_model.decoder(latent_predictions)
        generated_tokens = torch.argmax(logits, dim = -1).reshape(batch_size, num_samples, self.patch_size)
        
        selected_patches = []
        for i in range(batch_size):
            # sample_pool shape: (num_samples, patch_size)
            sample_pool = generated_tokens[i]

            # Convert to a hashable tuple
            sample_pool_tuples = [tuple(p.tolist()) for p in sample_pool]
            counts = Counter(sample_pool_tuples)
            
            selected_patch = None
            
            # Implement the cascading selection logic (try n, then n-1, n-2, ...).
            for n in range(n_initial, 0, -1):
                # Find all candidates that appeared at least n times.
                candidates = {patch: count for patch, count in counts.items() if count >= n}
                
                if candidates:                    
                    # Weighted sampling, the weight is the number of size-n combinations, C(k, n).
                    weights = [math.comb(count, n) for count in candidates.values()] 

                    patches = list(candidates.keys())
                    selected_tuple = random.choices(patches, weights=weights, k=1)[0]
                    selected_patch = torch.tensor(selected_tuple, dtype=torch.long, device=self.device)
                    break

            selected_patches.append(selected_patch)

        selected_patches = torch.stack(selected_patches, dim=0)

        output_shape = hidden_shape[:-1] + (self.patch_size,)
        final_output = selected_patches.reshape(output_shape)
            
        return final_output


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: Optional[int] = 2048,
        temperature: Optional[float] = 0.5,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Autoregressively generates sequences of tokens for the CALM model.

        This custom `generate` method is specifically designed for a model that operates on
        continuous vector representations of token, rather than discrete tokens.

        It replaces the standard Hugging Face `GenerationMixin.generate()` method, which
        is not compatible with this model's architecture.
        """
        self.eval()
        patch_size = self.patch_size
        batch_size = input_ids.shape[0]
        use_cache = True
        
        # Ensure the input prompt length is a multiple of patch_size. If not, pad it.
        prompt_len = input_ids.shape[1]
        if prompt_len % patch_size != 0:
            padding_size = patch_size - (prompt_len % patch_size)
            pad_tensor = torch.full((batch_size, padding_size), self.padding_idx, device=input_ids.device, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_tensor], dim=1)

        past_key_values = None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # --- Generation Loop ---
        while True:
            # Prepare model inputs
            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = input_ids[:, -patch_size:]
            current_seq_len = current_input_ids.shape[1]
            num_patches = current_seq_len // patch_size
            
            # Convert discrete tokens to patch embeddings
            inputs_embeds = self.transformer.embed_tokens(current_input_ids).reshape(batch_size, num_patches, -1)
            inputs_embeds = self.embed_proj(inputs_embeds)

            # Get the hidden_state for the next position from the Transformer
            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_states = outputs[0]
            past_key_values = outputs.past_key_values if use_cache else None

            # Use the MLPGenerator and autoencoder to generate the next K tokens
            last_hidden_state = hidden_states[:, -1, :] # Shape: [batch_size, hidden_size]
            next_tokens = self.temperature_sampling(last_hidden_state, temperature=temperature)

            # If a sequence is finished, fill with pad_token
            next_tokens = next_tokens * unfinished_sequences[:, None] + self.padding_idx * (1 - unfinished_sequences[:, None])

            input_ids = torch.cat([input_ids, next_tokens], dim=1)

            # Check if the newly generated tokens contain the eos_token_id
            produced_eos = (next_tokens == self.eos_token_id).any(dim=-1)
            unfinished_sequences = unfinished_sequences.mul((~produced_eos).long())

            # Check if the maximum length is reached or all sequences are done
            if input_ids.shape[1] >= max_length or unfinished_sequences.max() == 0:
                break

        # The `input_ids` tensor now looks like: [prompt, initial_padding, generated_content, final_padding]
        # We need to re-arrange it to: [prompt, generated_content, all_padding]

        clean_outputs = []
        seq_length = input_ids.shape[1]
        for i in range(batch_size):
            sequence = input_ids[i]
            
            # Find the indices of all non-padding tokens, put them on the 
            valid_token_indices = (sequence != self.padding_idx).nonzero(as_tuple=True)[0]
            if len(valid_token_indices) > 0:
                # Extract only the valid, non-padding tokens
                clean_sequence = sequence[valid_token_indices]

                # Create a new tensor of the original length, filled with padding
                new_padded_sequence = torch.full((seq_length,), self.padding_idx, device=input_ids.device, dtype=torch.long)

                # Copy the clean sequence to the start of the new tensor
                copy_len = len(clean_sequence)
                new_padded_sequence[:copy_len] = clean_sequence[:copy_len]
                clean_outputs.append(new_padded_sequence)
            else:
                clean_outputs.append(torch.full((seq_length,), self.padding_idx, device=input_ids.device, dtype=torch.long))

        input_ids = torch.stack(clean_outputs, dim=0)
                
        return input_ids


