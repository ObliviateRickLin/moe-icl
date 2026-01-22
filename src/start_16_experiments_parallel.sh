#!/bin/bash

# Set Environment Variables
export proxy="http://10.127.12.17:3128"
export https_proxy="http://10.127.12.17:3128"
export http_proxy="http://10.127.12.17:3128"
export WANDB_API_KEY="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"
export WANDB_MODE=online

# Ensure log directory exists
mkdir -p ../results

echo "Starting 16 experiments in PARALLEL..."

# GPU 0: TaS Replication + Basic MoE (E01-E04)
echo "Launching GPU 0 experiments..."
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E01_dense_2noise.yaml > ../results/log_E01.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E02_dense_4noise.yaml > ../results/log_E02.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E03_moe2_4noise.yaml > ../results/log_E03.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E04_moe4_4noise.yaml > ../results/log_E04.txt 2>&1 &

# GPU 1: Expert Scaling + Routing Variants (E05-E08)
echo "Launching GPU 1 experiments..."
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E05_moe8_4noise.yaml > ../results/log_E05.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E06_moe16_4noise.yaml > ../results/log_E06.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E07_moe4_seq_routing.yaml > ../results/log_E07.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E08_moe4_top2.yaml > ../results/log_E08.txt 2>&1 &

# GPU 2: Component Ablation + 8-Noise (E09-E12)
echo "Launching GPU 2 experiments..."
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E09_moe4_no_aux.yaml > ../results/log_E09.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E10_moe4_no_noise.yaml > ../results/log_E10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E11_moe4_8noise.yaml > ../results/log_E11.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E12_moe8_8noise.yaml > ../results/log_E12.txt 2>&1 &

# GPU 3: Curriculum + Large Models (E13-E16)
echo "Launching GPU 3 experiments..."
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E13_dense_curriculum.yaml > ../results/log_E13.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E14_moe4_curriculum.yaml > ../results/log_E14.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E15_moe4_large.yaml > ../results/log_E15.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E16_dense_large.yaml > ../results/log_E16.txt 2>&1 &

echo "All 16 experiments launched in parallel."
