#!/bin/bash
# S73/S74/S75: multi-seed eval at MAX DIM only (S73 d=10, S74 d=20, S75 d=40)
# Output: RMSE @ L=200 + Split CP @ L=200, one row per seed per model

set -e
cd /gemini/code/moe-icl/src
export CUDA_VISIBLE_DEVICES=0

S73_RUN="/gemini/code/moe-icl/results/S73_gpt2_w512_d12_nlr201x10/db64da3f-ae31-4264-88e9-6f416cbde8c9"
S74_RUN="/gemini/code/moe-icl/results/S74_gpt2_w512_d12_nlr201x20/e77bf731-fd02-4203-87c8-3809c2da0ac3"
S75_RUN="/gemini/code/moe-icl/results/S75_gpt2_w512_d12_nlr201x40/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089"

OUT="../results/_icl_curves"
SEEDS=(42 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  echo "========== SEED $SEED =========="

  # S73 max dim=10: RMSE at L=200 only
  python eval_s68_icl_effect.py --run-dir "$S73_RUN" --dims "10" --L-list "200" \
    --task-name noisy_linear_regression --use-state-pt --num-episodes 832 --batch-size 64 \
    --device cuda --seed "$SEED"
  cp "${OUT}/db64da3f-ae31-4264-88e9-6f416cbde8c9_icl_effect_loss_vs_L.csv" \
     "${OUT}/S73_rmse_maxdim_seed${SEED}.csv"

  # S73 Split CP at dim=10
  python eval_s68_dims.py --run-dir "$S73_RUN" --dims "10" --icl-L 200 \
    --task noisy_linear_regression --use-state-pt --num-eval-examples 1280 \
    --output-suffix "maxdim" --device cuda --seed "$SEED"
  cp "${OUT}/s68_dims_eval_S73_gpt2_w512_d12_nlr201x10_L200_maxdim.csv" \
     "${OUT}/S73_cp_maxdim_seed${SEED}.csv" 2>/dev/null || true
  # eval_s68_dims writes to out_dir; csv name uses run_dir.parent.name
  cp "${OUT}/s68_dims_eval_S73_gpt2_w512_d12_nlr201x10_L200_maxdim.csv" \
     "${OUT}/S73_cp_maxdim_seed${SEED}.csv"

  # S74 max dim=20
  python eval_s68_icl_effect.py --run-dir "$S74_RUN" --dims "20" --L-list "200" \
    --task-name noisy_linear_regression --use-state-pt --num-episodes 832 --batch-size 64 \
    --device cuda --seed "$SEED"
  cp "${OUT}/e77bf731-fd02-4203-87c8-3809c2da0ac3_icl_effect_loss_vs_L.csv" \
     "${OUT}/S74_rmse_maxdim_seed${SEED}.csv"

  python eval_s68_dims.py --run-dir "$S74_RUN" --dims "20" --icl-L 200 \
    --task noisy_linear_regression --use-state-pt --num-eval-examples 1280 \
    --output-suffix "maxdim" --device cuda --seed "$SEED"
  cp "${OUT}/s68_dims_eval_S74_gpt2_w512_d12_nlr201x20_L200_maxdim.csv" \
     "${OUT}/S74_cp_maxdim_seed${SEED}.csv"

  # S75 max dim=40
  python eval_s68_icl_effect.py --run-dir "$S75_RUN" --dims "40" --L-list "200" \
    --task-name noisy_linear_regression --use-state-pt --num-episodes 832 --batch-size 64 \
    --device cuda --seed "$SEED"
  cp "${OUT}/26dd20cb-2228-42b8-9bc3-6f7ae5fd4089_icl_effect_loss_vs_L.csv" \
     "${OUT}/S75_rmse_maxdim_seed${SEED}.csv"

  python eval_s68_dims.py --run-dir "$S75_RUN" --dims "40" --icl-L 200 \
    --task noisy_linear_regression --use-state-pt --num-eval-examples 1280 \
    --output-suffix "maxdim" --device cuda --seed "$SEED"
  cp "${OUT}/s68_dims_eval_S75_gpt2_w512_d12_nlr201x40_L200_maxdim.csv" \
     "${OUT}/S75_cp_maxdim_seed${SEED}.csv"
done

echo "Done. Results: ${OUT}/S7*_rmse_maxdim_seed*.csv, S7*_cp_maxdim_seed*.csv"
