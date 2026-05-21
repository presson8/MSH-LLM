#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

${PYTHON_BIN:-conda run -n mshllm python} src/main.py \
    --output_dir experiments \
    --comment "classification with MSHLLM tiny overfit diagnostic" \
    --name EthanolConcentration_MSHLLM_tiny_overfit \
    --records_file Classification_records.xls \
    --data_dir ./datasets/EthanolConcentration \
    --data_class tsra \
    --max_seq_len 128 \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --val_interval 10 \
    --lr 0.001 \
    --patch_size 8 \
    --stride 8 \
    --optimizer RAdam \
    --model MSHLLM \
    --d_model 128 \
    --dim_feedforward 128 \
    --d_ff 128 \
    --num_heads 4 \
    --num_layers 1 \
    --dropout 0.1 \
    --batch_size 4 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --llm_model GPT2 \
    --llm_dim 128 \
    --llm_layers 2 \
    --num_token 128 \
    --window_size 4,2 \
    --hyper_num 16,8,4 \
    --learn_prompt 4,4,4 \
    --train_limit_per_class 8 \
    --content "EthanolConcentration tiny overfit diagnostic with 8 samples per class."
