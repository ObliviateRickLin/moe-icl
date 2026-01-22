#!/bin/bash

# Set Environment Variables
export proxy="http://10.127.12.17:3128"
export https_proxy="http://10.127.12.17:3128"
export http_proxy="http://10.127.12.17:3128"
export WANDB_API_KEY="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"
export WANDB_MODE=online

mkdir -p ../results

echo "Starting experiments in start_64_experiments.sh..."

CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E01_dense_2noise.yaml > ../results/log_E01.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E02_dense_4noise.yaml > ../results/log_E02.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E03_moe2_4noise.yaml > ../results/log_E03.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E04_moe4_4noise.yaml > ../results/log_E04.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E05_moe8_4noise.yaml > ../results/log_E05.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E06_moe16_4noise.yaml > ../results/log_E06.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E07_moe4_seq_routing.yaml > ../results/log_E07.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E08_moe4_top2.yaml > ../results/log_E08.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E09_moe4_no_aux.yaml > ../results/log_E09.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E10_moe4_no_noise.yaml > ../results/log_E10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E11_moe4_8noise.yaml > ../results/log_E11.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E12_moe8_8noise.yaml > ../results/log_E12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E13_dense_curriculum.yaml > ../results/log_E13.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E14_moe4_curriculum.yaml > ../results/log_E14.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E15_moe4_large.yaml > ../results/log_E15.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config conf/encoder/E16_dense_large.yaml > ../results/log_E16.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E17_dense_genmix.yaml > ../results/log_E17.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E18_moe4_genmix.yaml > ../results/log_E18.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E19_moe8_genmix.yaml > ../results/log_E19.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E20_moe16_genmix.yaml > ../results/log_E20.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E21_moe8_genseq.yaml > ../results/log_E21.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E22_moe16_genseq.yaml > ../results/log_E22.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E23_dense_nonlin.yaml > ../results/log_E23.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E24_moe4_nonlin.yaml > ../results/log_E24.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E25_moe8_nonlin.yaml > ../results/log_E25.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E26_moe4_logic.yaml > ../results/log_E26.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E27_moe8_logic.yaml > ../results/log_E27.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E28_moe8_top2.yaml > ../results/log_E28.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E29_dense_multisp.yaml > ../results/log_E29.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E30_moe4_multisp.yaml > ../results/log_E30.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E31_moe8_multisp.yaml > ../results/log_E31.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config conf/encoder/E32_moe16_multisp.yaml > ../results/log_E32.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E33_moe8_seqsp.yaml > ../results/log_E33.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E34_moe16_seqsp.yaml > ../results/log_E34.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E35_dense_highsp.yaml > ../results/log_E35.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E36_moe8_highsp.yaml > ../results/log_E36.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E37_moe4_lasso.yaml > ../results/log_E37.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E38_moe8_lasso.yaml > ../results/log_E38.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E39_moe4_ood.yaml > ../results/log_E39.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E40_moe8_ood.yaml > ../results/log_E40.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E41_dense_10k.yaml > ../results/log_E41.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E42_moe8_10k.yaml > ../results/log_E42.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E43_moe64_10k.yaml > ../results/log_E43.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E44_dense_100k.yaml > ../results/log_E44.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E45_moe8_100k.yaml > ../results/log_E45.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E46_moe64_100k.yaml > ../results/log_E46.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E47_dense_1m.yaml > ../results/log_E47.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config conf/encoder/E48_moe8_1m.yaml > ../results/log_E48.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E49_moe64_1m.yaml > ../results/log_E49.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E50_moe4_fast.yaml > ../results/log_E50.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E51_moe16_fast.yaml > ../results/log_E51.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E52_moe64_fast.yaml > ../results/log_E52.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E53_moe8_aux0.yaml > ../results/log_E53.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E54_moe8_aux10.yaml > ../results/log_E54.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E55_moe8_fix.yaml > ../results/log_E55.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E56_moe8_rand.yaml > ../results/log_E56.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E57_moe8_top4.yaml > ../results/log_E57.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E58_moe8_top8.yaml > ../results/log_E58.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E59_moe4_seqaux0.yaml > ../results/log_E59.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E60_moe16_seqaux0.yaml > ../results/log_E60.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E61_dense_b256.yaml > ../results/log_E61.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E62_moe4_b256.yaml > ../results/log_E62.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E63_moe8_nonorm.yaml > ../results/log_E63.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config conf/encoder/E64_moe32.yaml > ../results/log_E64.txt 2>&1 &

echo "All experiments in start_64_experiments.sh launched."
