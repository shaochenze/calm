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
