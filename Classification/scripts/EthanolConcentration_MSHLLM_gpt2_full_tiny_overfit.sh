#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

${PYTHON_BIN:-conda run -n mshllm python} src/main.py \
    --output_dir experiments \
    --comment "classification with MSHLLM pretrained GPT-2 full config tiny overfit diagnostic" \
    --name EthanolConcentration_MSHLLM_gpt2_full_tiny_overfit \
    --records_file Classification_records.xls \
    --data_dir ./datasets/EthanolConcentration \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 100 \
    --val_interval 20 \
    --lr 0.0005 \
    --patch_size 8 \
    --stride 8 \
    --optimizer RAdam \
    --model MSHLLM \
    --d_model 768 \
    --dim_feedforward 768 \
    --d_ff 768 \
    --num_heads 4 \
    --num_layers 1 \
    --dropout 0.1 \
    --batch_size 1 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --llm_model GPT2 \
    --llm_dim 768 \
    --llm_layers 6 \
    --llm_pretrain \
    --llm_model_path ./pretrained_models/gpt2-small \
    --num_token 1000 \
    --window_size 4,2 \
    --hyper_num 50,20,10 \
    --learn_prompt 4,4,4 \
    --train_limit_per_class 8 \
    --content "EthanolConcentration full-configuration tiny overfit diagnostic with 8 samples per class."
