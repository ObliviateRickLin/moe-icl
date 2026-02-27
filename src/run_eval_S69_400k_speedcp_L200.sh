#!/bin/bash
# S69 @ 400K ckpt: split conformal at L=200, dense dimension sweep 5..100

export CUDA_VISIBLE_DEVICES=0
cd /gemini/code/moe-icl/src

RUN_DIR="/gemini/code/moe-icl/results/S69_gpt2_w512_d12_nlr201x100/b3807a92-6921-41a0-8f57-6ca413e28a39"

# Dense dims: 5,10,15,...,95,100 (20 points)
python eval_s68_dims.py \
  --run-dir "$RUN_DIR" \
  --dims "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100" \
  --icl-L 200 \
  --task noisy_linear_regression \
  --prefer-model-step 400000 \
  --num-eval-examples 1280 \
  --output-suffix "S69_400k_L200_dense" \
  --device cuda
