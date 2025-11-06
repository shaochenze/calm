# CALM: Continuous Autoregressive Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2510.27688-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.27688)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/shaochenze/calm)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Model-blue.svg)](https://huggingface.co/collections/cccczshao/calm)
[![Blog Post](https://img.shields.io/badge/Blog%20Post-Read%20Here-blue)](https://shaochenze.github.io/blog/2025/CALM)

## Overview
<p align="center">
  <img src="overview.png" width="700">
  <br>
  <em></em>
</p>

Modern Large Language Models (LLMs) are constrained by a fundamental bottleneck: they generate text one token at a time. **CALM (Continuous Autoregressive Language Models)** confronts this challenge by introducing a paradigm shift in language modeling. Instead of predicting one discrete token at a time, CALM learns to predict a single **continuous vector** that represents an entire chunk of **K** tokens. 

This is achieved through a two-stage process:
1.  **A high-fidelity autoencoder** learns to compress K tokens into a single vector and reconstruct them with near-perfect accuracy.
2.  **A continuous-domain language model** then performs autoregressive prediction in this vector space.

An in-depth explanation of CALM is available in [this blog](https://shaochenze.github.io/blog/2025/CALM).

### Key Features

*   🚀 **Ultra-Efficient by Design:** Dramatically improves training and inference efficiency by reducing the number of autoregressive steps by a factor of K. 

*   💡 **A New Scaling Axis:** Introduces a new scaling dimension for LLMs—**semantic bandwidth (K)**. Instead of just scaling parameters and data, you can now scale the amount of information processed in a single step.

*   🛠️ **A Comprehensive Likelihood-Free Toolkit:** Operating in a continuous domain requires new tools. This repository provides the full suite of algorithms that make CALM possible:
    *   **A Robust Autoencoder** to learn high-fidelity continuous representations of token chunks.
    *   **Energy-Based Training**, a principled and likelihood-free method for generative modeling.
    *   **BrierLM**, a new metric for calibrated, likelihood-free evaluation of language models.
    *   **Temperature Sampling** for controlled, high-quality text generation using only a black-box sampler.

## Getting Started

1.  Clone the Repository
```bash
git clone https://github.com/shaochenze/calm.git
cd calm
```

2.  Install Dependencies
```bash
pip install -r requirements.txt
```

3.  Prepare the Training Data

Run the following script to download and process the [pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) dataset for training.
```bash
bash data/get_data.sh
```
The dataset is large. Please ensure you have at least **2.5TB** of free disk space.

## Training

To replicate the results for CALM with `K=4`, follow these steps. The training process is divided into two main stages: train the autoencoder and then train the CALM language model.

### 1. Train the Autoencoder

First, train the autoencoder on approximately 15B tokens of data. This model learns the mapping between token chunks and their continuous vector representations.

```bash
bash train/train_autoencoder.sh
```

<details>
<summary>Click to see the full training script</summary>

```bash
#!/bin/bash

WORK_PATH=/path/to/the/code
CHECKPOINT_PATH=${WORK_PATH}/checkpoints/autoencoder
TOKENIZER_PATH=${WORK_PATH}/llama3_tokenizer
DATASET_TRAIN=${WORK_PATH}/pile-uncopyrighted/train/00.text.jsonl,${WORK_PATH}/pile-uncopyrighted/train/01.text.jsonl
DATASET_VALID=${WORK_PATH}/data/wikitext_document_level-test.json

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
    -m train.train_autoencoder \
    --tokenizer_name $TOKENIZER_PATH \
    --config_overrides "latent_size=128,num_encoder_layers=2,num_decoder_layers=2,patch_size=4" \
    --train_file $DATASET_TRAIN \
    --validation_file $DATASET_VALID \
    --keep_linebreaks True \
    --weight_decay 0.1 \
    --warmup_steps 1000 \
    --block_size 2048 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --streaming \
    --seed 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --max_steps 30000 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --learning_rate 3e-4 \
    --lr_scheduler_type "constant" \
    --logging_steps 100 \
    --do_train \
    --do_eval \
    --save_safetensors False \
    --output_dir $CHECKPOINT_PATH \
    --overwrite_output_dir \
    --bf16 True
```
</details>

### 2. Train the CALM Language Model

Once the autoencoder is trained, you can train the CALM model on the remaining data using our proposed **energy loss**. During evaluation steps, the **BrierLM score** is computed to track performance. This model should achieve a final BrierLM score of approximately 5.72 on the validation set.


```bash
bash train/train_energy.sh
```

<details>
<summary>Click to see the full training script</summary>
  
```bash
#!/bin/bash

WORK_PATH=/path/to/the/code
CHECKPOINT_PATH=${WORK_PATH}/checkpoints/calm_energy
TOKENIZER_PATH=${WORK_PATH}/llama3_tokenizer
AE_PATH=${WORK_PATH}/checkpoints/autoencoder
DATASET_VALID=${WORK_PATH}/data/wikitext_document_level-test.json

for i in $(seq -w 2 29); do
    if [[ $i -eq 2 ]]; then
        DATASET_TRAIN=${WORK_PATH}/pile-uncopyrighted/train/02.text.jsonl
    else
        DATASET_TRAIN=${DATASET_TRAIN},${WORK_PATH}/pile-uncopyrighted/train/${i}.text.jsonl
    fi
done

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
    -m train.train_calm \
    --ae_name_or_path $AE_PATH \
    --tokenizer_name $TOKENIZER_PATH \
    --train_file $DATASET_TRAIN \
    --validation_file $DATASET_VALID \
    --config_overrides "latent_size=128,num_mlp_layers=4,patch_size=4,hidden_size=1024,intermediate_size=2752,num_hidden_layers=16,num_attention_heads=16,num_key_value_heads=16" \
    --keep_linebreaks True \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --block_size 8192 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --streaming \
    --seed 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --max_steps 250000 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --learning_rate 3e-4 \
    --lr_scheduler_type "constant" \
    --logging_steps 100 \
    --do_train \
    --do_eval \
    --save_safetensors False \
    --output_dir $CHECKPOINT_PATH \
    --overwrite_output_dir \
    --bf16 True
```
</details>

We also provide alternative training scripts for generative heads based on **Diffusion** and **Flow Matching**, available at `train/train_diffusion.sh` and `train/train_flow.sh`. However, we found their performance to be slightly below that of the Energy-based head in our experiments.


### 3. (Optional) Train a Baseline Autoregressive Model

For comparison, you can also train a standard autoregressive Transformer baseline. This model is evaluated by the same BrierLM score, allowing for a direct comparison with CALM. The baseline model is expected to reach a BrierLM score of around 6.05. 

```bash
bash train/train_ar.sh
```

<details>
<summary>Click to see the full training script</summary>
  
```bash
#!/bin/bash

WORK_PATH=/path/to/the/code
CHECKPOINT_PATH=${WORK_PATH}/checkpoints/ar
TOKENIZER_PATH=${WORK_PATH}/llama3_tokenizer
DATASET_VALID=${WORK_PATH}/data/wikitext_document_level-test.json

for i in $(seq -w 0 29); do
    if [[ $i -eq 0 ]]; then
        DATASET_TRAIN=${WORK_PATH}/pile-uncopyrighted/train/00.text.jsonl
    else
        DATASET_TRAIN=${DATASET_TRAIN},${WORK_PATH}/pile-uncopyrighted/train/${i}.text.jsonl
    fi
done

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
    -m train.train_ar \
    --model_type llama \
    --tokenizer_name $TOKENIZER_PATH \
    --config_overrides "hidden_size=768,intermediate_size=2048,num_hidden_layers=12,num_attention_heads=16,num_key_value_heads=16" \
    --train_file $DATASET_TRAIN \
    --validation_file $DATASET_VALID \
    --keep_linebreaks True \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --block_size 2048 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --streaming \
    --seed 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 1 \
    --max_steps 250000 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --learning_rate 3e-4 \
    --lr_scheduler_type "constant" \
    --logging_steps 100 \
    --do_train \
    --do_eval \
    --output_dir $CHECKPOINT_PATH \
    --save_safetensors False \
    --overwrite_output_dir \
    --bf16 True
```
</details>

## Evaluation
For convenience, our pre-trained autoencoder and CALM checkpoints can be downloaded directly here as well:
| Model                                                              | Parameters | BrierLM |
| ------------------------------------------------------------------ | :--------: | :-----: |
| [Autoencoder](https://huggingface.co/cccczshao/CALM-Autoencoder) | 75M     | --   |
| [CALM-M](https://huggingface.co/cccczshao/CALM-M)          | 371M    | 5.72   |
| [CALM-L](https://huggingface.co/cccczshao/CALM-L)          | 735M    | 6.58   |
| [CALM-XL](https://huggingface.co/cccczshao/CALM-XL)        | 1.82B      | 8.53   |

Run the following script to evaluate these pre-trained checkpoints:
```bash
bash train/eval_energy.sh
```
<details>
<summary>Click to see the full evaluation script</summary>
  
```bash
#!/bin/bash

WORK_PATH=/path/to/the/code
CHECKPOINT_PATH=/path/to/the/calm/
AE_PATH=/path/to/the/autoencoder
DATASET_VALID=${WORK_PATH}/data/wikitext_document_level-test.json

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
    -m train.train_calm \
    --ae_name_or_path $AE_PATH \
    --model_name_or_path $CHECKPOINT_PATH \
    --validation_file $DATASET_VALID \
    --seed 1 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --output_dir $CHECKPOINT_PATH \
    --bf16 True
```
</details>

## Contact

If you have any questions, feel free to submit an issue or contact `chenzeshao@tencent.com`.
