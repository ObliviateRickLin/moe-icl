import os
import glob

def create_script(filename, commands):
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Set Environment Variables\n")
        f.write('export proxy="http://10.127.12.17:3128"\n')
        f.write('export https_proxy="http://10.127.12.17:3128"\n')
        f.write('export http_proxy="http://10.127.12.17:3128"\n')
        f.write('export WANDB_API_KEY="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"\n')
        f.write('export WANDB_MODE=online\n\n')
        f.write('mkdir -p ../results\n\n')
        f.write(f'echo "Starting experiments in {filename}..."\n\n')
        
        for cmd in commands:
            f.write(cmd + "\n")
            
        f.write(f'\necho "All experiments in {filename} launched."\n')
    
    os.chmod(filename, 0o755)
    print(f"Generated {filename}")

# --- 1. 64 New Experiments (E01-E64) ---
# E01-E16 -> GPU 0
# E17-E32 -> GPU 1
# E33-E48 -> GPU 2
# E49-E64 -> GPU 3

commands_64 = []
for i in range(1, 65):
    exp_id = f"E{i:02d}"
    # Find the config file
    config_files = glob.glob(f"conf/encoder/{exp_id}_*.yaml")
    if not config_files:
        print(f"Warning: Config for {exp_id} not found.")
        continue
    config_file = config_files[0]
    
    # Assign GPU
    if 1 <= i <= 16:
        gpu = 0
    elif 17 <= i <= 32:
        gpu = 1
    elif 33 <= i <= 48:
        gpu = 2
    else:
        gpu = 3
        
    log_file = f"../results/log_{exp_id}.txt"
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} nohup python train.py --config {config_file} > {log_file} 2>&1 &"
    commands_64.append(cmd)

create_script("start_64_experiments.sh", commands_64)

# --- 2. Original Experiments ---
# Distribute evenly across 4 GPUs
commands_orig = []
original_configs = [f for f in glob.glob("conf/encoder/*.yaml") if not os.path.basename(f).startswith("E")]
original_configs.sort()

for i, config_file in enumerate(original_configs):
    gpu = i % 4
    name = os.path.basename(config_file).replace(".yaml", "")
    log_file = f"../results/original/log_{name}.txt"
    # Ensure directory exists
    cmd_mkdir = "mkdir -p ../results/original"
    if cmd_mkdir not in commands_orig: # Add once usually, but here we add to list. 
        # Actually better to put mkdir in script header, but for now:
        pass 
        
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} nohup python train.py --config {config_file} --wandb.entity jinruilin-aijobtech > {log_file} 2>&1 &"
    commands_orig.append(cmd)

# Prepend mkdir
commands_orig.insert(0, "mkdir -p ../results/original")
create_script("start_original_experiments.sh", commands_orig)

# --- 3. GPT Experiments ---
# Distribute evenly across 4 GPUs
commands_gpt = []
gpt_configs = glob.glob("conf/gpt/*.yaml")
gpt_configs.sort()

for i, config_file in enumerate(gpt_configs):
    gpu = i % 4
    name = os.path.basename(config_file).replace(".yaml", "")
    log_file = f"../results/gpt/log_{name}.txt"
    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} nohup python train.py --config {config_file} --wandb.entity jinruilin-aijobtech > {log_file} 2>&1 &"
    commands_gpt.append(cmd)

commands_gpt.insert(0, "mkdir -p ../results/gpt")
create_script("start_gpt_experiments.sh", commands_gpt)
