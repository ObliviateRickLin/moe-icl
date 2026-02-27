#!/bin/bash
# SpeedCP (not split) at max dim only, L=200: S69 400K, S73/S74/S75 state.pt

set -e
cd /gemini/code/moe-icl/src
export CUDA_VISIBLE_DEVICES=0

S69_RUN="/gemini/code/moe-icl/results/S69_gpt2_w512_d12_nlr201x100/b3807a92-6921-41a0-8f57-6ca413e28a39"
S73_RUN="/gemini/code/moe-icl/results/S73_gpt2_w512_d12_nlr201x10/db64da3f-ae31-4264-88e9-6f416cbde8c9"
S74_RUN="/gemini/code/moe-icl/results/S74_gpt2_w512_d12_nlr201x20/e77bf731-fd02-4203-87c8-3809c2da0ac3"
S75_RUN="/gemini/code/moe-icl/results/S75_gpt2_w512_d12_nlr201x40/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089"

echo "=== S69 400K SpeedCP max dim (d=100) L=200 (gamma=0.0012 lambda=4.5) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "$S69_RUN" \
  --prefer-model-step 400000 \
  --max-n-points 201 \
  --icl-lens "200" \
  --num-eval-examples 1280 \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma 0.0012 --lambda 4.5 \
  --output-suffix "maxdim_L200" \
  --device cuda

echo "=== S73 SpeedCP max dim (d=10) L=200 (state.pt, gamma=0.0012 lambda=4.5) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "$S73_RUN" \
  --use-state-pt \
  --max-n-points 201 \
  --icl-lens "200" \
  --num-eval-examples 1280 \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma 0.0012 --lambda 4.5 \
  --output-suffix "maxdim_L200" \
  --device cuda

echo "=== S74 SpeedCP max dim (d=20) L=200 (state.pt, gamma=0.0012 lambda=4.5) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "$S74_RUN" \
  --use-state-pt \
  --max-n-points 201 \
  --icl-lens "200" \
  --num-eval-examples 1280 \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma 0.0012 --lambda 4.5 \
  --output-suffix "maxdim_L200" \
  --device cuda

echo "=== S75 SpeedCP max dim (d=40) L=200 (state.pt, gamma=0.0012 lambda=4.5) ==="
python eval_icl_lr2x_speedcp.py \
  --run-dir "$S75_RUN" \
  --use-state-pt \
  --max-n-points 201 \
  --icl-lens "200" \
  --num-eval-examples 1280 \
  --calib-frac 0.5 \
  --alpha 0.05 \
  --gamma 0.0012 --lambda 4.5 \
  --output-suffix "maxdim_L200" \
  --device cuda

echo "Done. CSVs in results/_icl_curves/compare_lr2x_speedcp_summary_*_maxdim_L200_alpha0p05.csv"
