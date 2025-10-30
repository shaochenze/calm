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
