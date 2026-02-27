#!/bin/bash
# S69 @ 400K ckpt: multi-dim RMSE vs L scan, max dim=100, L up to 200

export CUDA_VISIBLE_DEVICES=0
cd /gemini/code/moe-icl/src

RUN_DIR="/gemini/code/moe-icl/results/S69_gpt2_w512_d12_nlr201x100/b3807a92-6921-41a0-8f57-6ca413e28a39"

python eval_s68_icl_effect.py \
  --run-dir "$RUN_DIR" \
  --dims "5,10,20,30,50,100" \
  --L-list "1,5,10,20,40,80,120,160,200" \
  --task-name noisy_linear_regression \
  --prefer-model-step 400000 \
  --num-episodes 832 \
  --batch-size 64 \
  --device cuda
