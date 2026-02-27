#!/bin/bash
# Launch S69-S72 training: 4 tasks, w512 d24, 1.5M steps, 400 examples, 200 dims
# One experiment per GPU (GPU 0-3)

export WANDB_MODE=offline
export http_proxy=http://192.168.50.105:7890
export https_proxy=http://192.168.50.105:7890

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SRC_DIR/../results"

declare -A EXPS
EXPS[0]="S69_gpt2_w512_d24_nlr401x200"
EXPS[1]="S70_gpt2_w512_d24_nqr401x200"
EXPS[2]="S71_gpt2_w512_d24_n2nn401x200"
EXPS[3]="S72_gpt2_w512_d24_ndt401x200"

for GPU in 0 1 2 3; do
    EXP="${EXPS[$GPU]}"
    LOG_FILE="$LOG_DIR/log_${EXP}.txt"
    echo "Launching $EXP on GPU $GPU -> $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU nohup python "$SRC_DIR/train.py" \
        --config "conf/gpt/${EXP}.yaml" \
        > "$LOG_FILE" 2>&1 &
    echo "  PID=$!"
done

echo "All 4 jobs launched."
wait
