#!/bin/bash

# Set Environment Variables
export proxy="http://10.127.12.17:3128"
export https_proxy="http://10.127.12.17:3128"
export http_proxy="http://10.127.12.17:3128"
export WANDB_API_KEY="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"
export WANDB_MODE=online

mkdir -p ../results

echo "Starting experiments in start_original_experiments.sh..."

mkdir -p ../results/original
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/dense_4noise.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_dense_4noise.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/dense_baseline.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_dense_baseline.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/exp_mixed_noise_dense.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_exp_mixed_noise_dense.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/exp_mixed_noise_moe4.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_exp_mixed_noise_moe4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/exp_mixed_noise_moe4_seq.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_exp_mixed_noise_moe4_seq.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/exp_mixed_noise_moe8.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_exp_mixed_noise_moe8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/lasso_guided_opt_0.1_5_-2.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_lasso_guided_opt_0.1_5_-2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/linear_and_logistic.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_linear_and_logistic.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/linear_classification.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_linear_classification.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/linear_regression.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_linear_regression.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/moe4_4noise.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_moe4_4noise.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/moe4_noisy.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_moe4_noisy.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/moe8_8noise.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_moe8_8noise.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/moe_noisy_linear.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_moe_noisy_linear.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/ridge_0.1.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_ridge_0.1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/ridge_0.25.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_ridge_0.25.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/ridge_0.5.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_ridge_0.5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/ridge_1.0.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_ridge_1.0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/ridge_mtl.yaml --wandb.entity jinruilin-aijobtech > ../results/original/log_ridge_mtl.txt 2>&1 &

echo "All experiments in start_original_experiments.sh launched."
