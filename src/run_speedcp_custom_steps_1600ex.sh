#!/bin/bash
# run_speedcp_custom_steps_1600ex.sh

export CUDA_VISIBLE_DEVICES=2
cd /gemini/code/moe-icl/src

# Configs
GAMMA=0.0012
LAMBDA=4.5
NUM_EX=1600
ICL_LEN=200
SUFFIX="maxdim_L200_1600ex"

# S69 400K
echo "=== S69 400K SpeedCP (num_eval_examples=$NUM_EX) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "/gemini/code/moe-icl/results/S69_gpt2_w512_d12_nlr201x100/b3807a92-6921-41a0-8f57-6ca413e28a39" \
  --prefer-model-step 400000 \
  --max-n-points 201 \
  --icl-lens "$ICL_LEN" \
  --num-eval-examples "$NUM_EX" \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma $GAMMA --lambda $LAMBDA \
  --output-suffix "$SUFFIX" \
  --device cuda

# S73 300K
echo "=== S73 300K SpeedCP (num_eval_examples=$NUM_EX) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "/gemini/code/moe-icl/results/S73_gpt2_w512_d12_nlr201x10/db64da3f-ae31-4264-88e9-6f416cbde8c9" \
  --prefer-model-step 300000 \
  --max-n-points 201 \
  --icl-lens "$ICL_LEN" \
  --num-eval-examples "$NUM_EX" \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma $GAMMA --lambda $LAMBDA \
  --output-suffix "$SUFFIX" \
  --device cuda

# S74 300K
echo "=== S74 300K SpeedCP (num_eval_examples=$NUM_EX) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "/gemini/code/moe-icl/results/S74_gpt2_w512_d12_nlr201x20/e77bf731-fd02-4203-87c8-3809c2da0ac3" \
  --prefer-model-step 300000 \
  --max-n-points 201 \
  --icl-lens "$ICL_LEN" \
  --num-eval-examples "$NUM_EX" \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma $GAMMA --lambda $LAMBDA \
  --output-suffix "$SUFFIX" \
  --device cuda

# S75 300K
echo "=== S75 300K SpeedCP (num_eval_examples=$NUM_EX) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "/gemini/code/moe-icl/results/S75_gpt2_w512_d12_nlr201x40/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089" \
  --prefer-model-step 300000 \
  --max-n-points 201 \
  --icl-lens "$ICL_LEN" \
  --num-eval-examples "$NUM_EX" \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma $GAMMA --lambda $LAMBDA \
  --output-suffix "$SUFFIX" \
  --device cuda
