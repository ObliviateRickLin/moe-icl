#!/bin/bash

set -euo pipefail

mkdir -p ../results

CONFS=(
  conf/gpt/S25_gpt2_w32_d6_mixnoise4_80x40.yaml
  conf/gpt/S26_gpt2_w64_d6_mixnoise4_80x40.yaml
  conf/gpt/S27_gpt2_w128_d6_mixnoise4_80x40.yaml
  conf/gpt/S28_gpt2_w256_d6_mixnoise4_80x40.yaml
  conf/gpt/S29_gpt2_w64_d2_mixnoise4_80x40.yaml
  conf/gpt/S30_gpt2_w64_d4_mixnoise4_80x40.yaml
  conf/gpt/S31_gpt2_w64_d8_mixnoise4_80x40.yaml
  conf/gpt/S32_gpt2_w64_d12_mixnoise4_80x40.yaml
  conf/gpt/S33_gpt2_tiny_mixnoise4_80x40.yaml
  conf/gpt/S34_gpt2_small_mixnoise4_80x40.yaml
  conf/gpt/S35_gpt2_medium_mixnoise4_80x40.yaml
  conf/gpt/S36_gpt2_large_mixnoise4_80x40.yaml
)

for i in "${!CONFS[@]}"; do
  gpu=$((i % 8))
  conf="${CONFS[$i]}"
  base="$(basename "$conf" .yaml)"
  log="../results/log_${base}.txt"
  echo "[LAUNCH] GPU=${gpu} ${conf} -> ${log}"
  CUDA_VISIBLE_DEVICES=${gpu} nohup python train.py --config "${conf}" --wandb.entity jinruilin-aijobtech > "${log}" 2>&1 &
done

echo "Launched 12 runs (S25-S36 mixnoise4_80x40)."
