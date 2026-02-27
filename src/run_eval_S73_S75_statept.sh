#!/bin/bash
# S73/S74/S75: RMSE vs L + Split CP using state.pt (current checkpoint), same settings as before

set -e
cd /gemini/code/moe-icl/src
export CUDA_VISIBLE_DEVICES=0

S73_RUN="/gemini/code/moe-icl/results/S73_gpt2_w512_d12_nlr201x10/db64da3f-ae31-4264-88e9-6f416cbde8c9"
S74_RUN="/gemini/code/moe-icl/results/S74_gpt2_w512_d12_nlr201x20/e77bf731-fd02-4203-87c8-3809c2da0ac3"
S75_RUN="/gemini/code/moe-icl/results/S75_gpt2_w512_d12_nlr201x40/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089"

# ---- RMSE vs L (ICL effect) ----
echo "=== S73 RMSE vs L (state.pt) ==="
python eval_s68_icl_effect.py \
  --run-dir "$S73_RUN" \
  --dims "5,10" \
  --L-list "1,5,10,20,40,80,120,160,200" \
  --task-name noisy_linear_regression \
  --use-state-pt \
  --num-episodes 832 \
  --batch-size 64 \
  --device cuda

echo "=== S74 RMSE vs L (state.pt) ==="
python eval_s68_icl_effect.py \
  --run-dir "$S74_RUN" \
  --dims "5,10,20" \
  --L-list "1,5,10,20,40,80,120,160,200" \
  --task-name noisy_linear_regression \
  --use-state-pt \
  --num-episodes 832 \
  --batch-size 64 \
  --device cuda

echo "=== S75 RMSE vs L (state.pt) ==="
python eval_s68_icl_effect.py \
  --run-dir "$S75_RUN" \
  --dims "5,10,20,30,40" \
  --L-list "1,5,10,20,40,80,120,160,200" \
  --task-name noisy_linear_regression \
  --use-state-pt \
  --num-episodes 832 \
  --batch-size 64 \
  --device cuda

# ---- Split conformal @ L=200 ----
echo "=== S73 Split CP L=200 (state.pt) ==="
python eval_s68_dims.py \
  --run-dir "$S73_RUN" \
  --dims "5,6,7,8,9,10" \
  --icl-L 200 \
  --task noisy_linear_regression \
  --use-state-pt \
  --num-eval-examples 1280 \
  --output-suffix "statept_L200" \
  --device cuda

echo "=== S74 Split CP L=200 (state.pt) ==="
python eval_s68_dims.py \
  --run-dir "$S74_RUN" \
  --dims "5,10,15,20" \
  --icl-L 200 \
  --task noisy_linear_regression \
  --use-state-pt \
  --num-eval-examples 1280 \
  --output-suffix "statept_L200" \
  --device cuda

echo "=== S75 Split CP L=200 (state.pt) ==="
python eval_s68_dims.py \
  --run-dir "$S75_RUN" \
  --dims "5,10,15,20,25,30,35,40" \
  --icl-L 200 \
  --task noisy_linear_regression \
  --use-state-pt \
  --num-eval-examples 1280 \
  --output-suffix "statept_L200" \
  --device cuda

echo "Done. RMSE CSVs/plots: results/_icl_curves/*.csv, *.png. Split CP CSVs: results/S7*_gpt2_*/s68_dims_eval_*_statept_L200.csv"
