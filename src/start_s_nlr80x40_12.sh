#!/bin/bash

set -euo pipefail

mkdir -p ../results

CONFS=(
  conf/gpt/S13_gpt2_w32_d6_nlr80x40.yaml
  conf/gpt/S14_gpt2_w64_d6_nlr80x40.yaml
  conf/gpt/S15_gpt2_w128_d6_nlr80x40.yaml
  conf/gpt/S16_gpt2_w256_d6_nlr80x40.yaml
  conf/gpt/S17_gpt2_w64_d2_nlr80x40.yaml
  conf/gpt/S18_gpt2_w64_d4_nlr80x40.yaml
  conf/gpt/S19_gpt2_w64_d8_nlr80x40.yaml
  conf/gpt/S20_gpt2_w64_d12_nlr80x40.yaml
  conf/gpt/S21_gpt2_tiny_nlr80x40.yaml
  conf/gpt/S22_gpt2_small_nlr80x40.yaml
  conf/gpt/S23_gpt2_medium_nlr80x40.yaml
  conf/gpt/S24_gpt2_large_nlr80x40.yaml
)

for i in "${!CONFS[@]}"; do
  gpu=$((i % 8))
  conf="${CONFS[$i]}"
  base="$(basename "$conf" .yaml)"
  log="../results/log_${base}.txt"
  echo "[LAUNCH] GPU=${gpu} ${conf} -> ${log}"
  CUDA_VISIBLE_DEVICES=${gpu} nohup python train.py --config "${conf}" --wandb.entity jinruilin-aijobtech > "${log}" 2>&1 &
done

echo "Launched 12 runs (S13-S24 nlr80x40)."
