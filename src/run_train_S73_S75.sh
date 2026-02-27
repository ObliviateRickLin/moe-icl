#!/bin/bash
# Train S73 (10d), S74 (20d), S75 (40d): same arch as S69, L max 200, dims 5->10/20/40

# Proxy for wandb and network (override via env if needed)
export http_proxy="${http_proxy:-http://10.127.12.17:3128}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

# Wandb online logging (set WANDB_API_KEY in env to override)
export WANDB_MODE=online
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

cd /gemini/code/moe-icl/src

# Use 3 GPUs (S69 may be on GPU 0); adjust CUDA_VISIBLE_DEVICES if needed
CUDA_VISIBLE_DEVICES=1 python train.py --config conf/gpt/S73_gpt2_w512_d12_nlr201x10.yaml --wandb.entity jinruilin-aijobtech &
CUDA_VISIBLE_DEVICES=2 python train.py --config conf/gpt/S74_gpt2_w512_d12_nlr201x20.yaml --wandb.entity jinruilin-aijobtech &
CUDA_VISIBLE_DEVICES=3 python train.py --config conf/gpt/S75_gpt2_w512_d12_nlr201x40.yaml --wandb.entity jinruilin-aijobtech &

echo "Launched S73 (GPU 1), S74 (GPU 2), S75 (GPU 3)."
