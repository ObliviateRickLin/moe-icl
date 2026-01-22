#!/bin/bash

# Set Environment Variables
export proxy="http://10.127.12.17:3128"
export https_proxy="http://10.127.12.17:3128"
export http_proxy="http://10.127.12.17:3128"
export WANDB_API_KEY="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"
export WANDB_MODE=online

mkdir -p ../results

echo "Starting experiments in start_gpt_experiments.sh..."

mkdir -p ../results/gpt
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/gpt/linear_and_logistic.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_linear_and_logistic.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/gpt/linear_classification.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_linear_classification.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/gpt/linear_regression.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_linear_regression.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/gpt/ridge_0.1.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_ridge_0.1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/gpt/ridge_0.25.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_ridge_0.25.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/gpt/ridge_0.5.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_ridge_0.5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/gpt/ridge_1.0.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_ridge_1.0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/gpt/ridge_mtl.yaml --wandb.entity jinruilin-aijobtech > ../results/gpt/log_ridge_mtl.txt 2>&1 &

echo "All experiments in start_gpt_experiments.sh launched."
