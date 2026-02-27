#!/bin/bash
# Multi-seed eval for S73/S74/S75 using state.pt
# Runs RMSE vs L and Split CP @ L=200 for several seeds,
# and saves per-seed CSV copies so results can be compared.

set -e

cd /gemini/code/moe-icl/src
export CUDA_VISIBLE_DEVICES=0

S73_RUN="/gemini/code/moe-icl/results/S73_gpt2_w512_d12_nlr201x10/db64da3f-ae31-4264-88e9-6f416cbde8c9"
S74_RUN="/gemini/code/moe-icl/results/S74_gpt2_w512_d12_nlr201x20/e77bf731-fd02-4203-87c8-3809c2da0ac3"
S75_RUN="/gemini/code/moe-icl/results/S75_gpt2_w512_d12_nlr201x40/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089"

OUT_DIR="/gemini/code/moe-icl/results/_icl_curves"

SEEDS=("1" "2")

for SEED in "${SEEDS[@]}"; do
  echo "=== SEED ${SEED} : S73/S74/S75 RMSE vs L (state.pt) ==="

  # --- RMSE vs L ---
  python eval_s68_icl_effect.py \
    --run-dir "$S73_RUN" \
    --dims "5,10" \
    --L-list "1,5,10,20,40,80,120,160,200" \
    --task-name noisy_linear_regression \
    --use-state-pt \
    --num-episodes 832 \
    --batch-size 64 \
    --device cuda \
    --seed "${SEED}"

  cp "${OUT_DIR}/db64da3f-ae31-4264-88e9-6f416cbde8c9_icl_effect_loss_vs_L.csv" \
     "${OUT_DIR}/db64da3f-ae31-4264-88e9-6f416cbde8c9_icl_effect_loss_vs_L_seed${SEED}.csv"

  python eval_s68_icl_effect.py \
    --run-dir "$S74_RUN" \
    --dims "5,10,20" \
    --L-list "1,5,10,20,40,80,120,160,200" \
    --task-name noisy_linear_regression \
    --use-state-pt \
    --num-episodes 832 \
    --batch-size 64 \
    --device cuda \
    --seed "${SEED}"

  cp "${OUT_DIR}/e77bf731-fd02-4203-87c8-3809c2da0ac3_icl_effect_loss_vs_L.csv" \
     "${OUT_DIR}/e77bf731-fd02-4203-87c8-3809c2da0ac3_icl_effect_loss_vs_L_seed${SEED}.csv"

  python eval_s68_icl_effect.py \
    --run-dir "$S75_RUN" \
    --dims "5,10,20,30,40" \
    --L-list "1,5,10,20,40,80,120,160,200" \
    --task-name noisy_linear_regression \
    --use-state-pt \
    --num-episodes 832 \
    --batch-size 64 \
    --device cuda \
    --seed "${SEED}"

  cp "${OUT_DIR}/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089_icl_effect_loss_vs_L.csv" \
     "${OUT_DIR}/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089_icl_effect_loss_vs_L_seed${SEED}.csv"

  echo "=== SEED ${SEED} : S73/S74/S75 Split CP L=200 (state.pt) ==="

  # --- Split CP @ L=200 ---
  python eval_s68_dims.py \
    --run-dir "$S73_RUN" \
    --dims "5,6,7,8,9,10" \
    --icl-L 200 \
    --task noisy_linear_regression \
    --use-state-pt \
    --num-eval-examples 1280 \
    --output-suffix "statept_L200" \
    --device cuda \
    --seed "${SEED}"

  cp "${OUT_DIR}/s68_dims_eval_S73_gpt2_w512_d12_nlr201x10_L200_statept_L200.csv" \
     "${OUT_DIR}/s68_dims_eval_S73_gpt2_w512_d12_nlr201x10_L200_seed${SEED}_statept_L200.csv"

  python eval_s68_dims.py \
    --run-dir "$S74_RUN" \
    --dims "5,10,15,20" \
    --icl-L 200 \
    --task noisy_linear_regression \
    --use-state-pt \
    --num-eval-examples 1280 \
    --output-suffix "statept_L200" \
    --device cuda \
    --seed "${SEED}"

  cp "${OUT_DIR}/s68_dims_eval_S74_gpt2_w512_d12_nlr201x20_L200_statept_L200.csv" \
     "${OUT_DIR}/s68_dims_eval_S74_gpt2_w512_d12_nlr201x20_L200_seed${SEED}_statept_L200.csv"

  python eval_s68_dims.py \
    --run-dir "$S75_RUN" \
    --dims "5,10,15,20,25,30,35,40" \
    --icl-L 200 \
    --task noisy_linear_regression \
    --use-state-pt \
    --num-eval-examples 1280 \
    --output-suffix "statept_L200" \
    --device cuda \
    --seed "${SEED}"

  cp "${OUT_DIR}/s68_dims_eval_S75_gpt2_w512_d12_nlr201x40_L200_statept_L200.csv" \
     "${OUT_DIR}/s68_dims_eval_S75_gpt2_w512_d12_nlr201x40_L200_seed${SEED}_statept_L200.csv"

done

echo "Done multi-seed eval for seeds: ${SEEDS[*]}"

