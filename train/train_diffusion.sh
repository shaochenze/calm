#!/bin/bash

WORK_PATH=/path/to/the/code
CHECKPOINT_PATH=${WORK_PATH}/checkpoints/calm_diffusion
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
    --config_overrides "model_type=diffusion,latent_size=128,num_mlp_layers=4,patch_size=4,hidden_size=1024,intermediate_size=2752,num_hidden_layers=16,num_attention_heads=16,num_key_value_heads=16" \
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
