
import os
import glob

# Get all yaml files in the directory
config_dir = "/root/moe-icl/src/conf/encoder"
all_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".yaml")])

# Filter out the E* files we just created
original_files = [f for f in all_files if not f.startswith("E")]

print(f"Found {len(original_files)} original configuration files.")

# Distribute across 4 GPUs
gpus = 4
files_per_gpu = [[] for _ in range(gpus)]
for i, file in enumerate(original_files):
    files_per_gpu[i % gpus].append(file)

# Generate the shell script content
script_content = """#!/bin/bash

# Set Environment Variables
export proxy="http://10.127.12.17:3128"
export https_proxy="http://10.127.12.17:3128"
export http_proxy="http://10.127.12.17:3128"
export WANDB_API_KEY="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"
export WANDB_MODE=online

mkdir -p ../results/original

echo "Starting original experiments..."
"""

for gpu_id in range(gpus):
    script_content += f"\n# GPU {gpu_id}\n"
    for file in files_per_gpu[gpu_id]:
        log_name = file.replace('.yaml', '.txt')
        # We use --wandb.entity to override the config file setting
        # We also redirect output to a separate folder to avoid clutter
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python train.py --config conf/encoder/{file} --wandb.entity jinruilin-aijobtech > ../results/original/log_{log_name} 2>&1 &"
        script_content += f"{cmd}\n"

script_content += '\necho "All original experiments launched."\n'

with open("/root/moe-icl/src/start_original_experiments.sh", "w") as f:
    f.write(script_content)

print("Script generated at /root/moe-icl/src/start_original_experiments.sh")
