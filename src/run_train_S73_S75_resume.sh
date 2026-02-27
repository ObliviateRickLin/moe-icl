#!/bin/bash
# Resume S73/S74/S75 with nohup + log files so training survives terminal disconnect

export http_proxy="${http_proxy:-http://10.127.12.17:3128}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export WANDB_MODE=online
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

cd /gemini/code/moe-icl/src

nohup env CUDA_VISIBLE_DEVICES=1 python train.py --config conf/gpt/S73_gpt2_w512_d12_nlr201x10.yaml --training.resume_id db64da3f-ae31-4264-88e9-6f416cbde8c9 --wandb.entity jinruilin-aijobtech >> ../results/log_S73_gpt2_w512_d12_nlr201x10_resume.txt 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=2 python train.py --config conf/gpt/S74_gpt2_w512_d12_nlr201x20.yaml --training.resume_id e77bf731-fd02-4203-87c8-3809c2da0ac3 --wandb.entity jinruilin-aijobtech >> ../results/log_S74_gpt2_w512_d12_nlr201x20_resume.txt 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=3 python train.py --config conf/gpt/S75_gpt2_w512_d12_nlr201x40.yaml --training.resume_id 26dd20cb-2228-42b8-9bc3-6f7ae5fd4089 --wandb.entity jinruilin-aijobtech >> ../results/log_S75_gpt2_w512_d12_nlr201x40_resume.txt 2>&1 &

echo "Resumed S73 (GPU1), S74 (GPU2), S75 (GPU3) with nohup. Logs: results/log_S7*_resume.txt"
