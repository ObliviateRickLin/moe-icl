
import os
import yaml

# Common configuration
COMMON_CONFIG = {
    "inherit": ["../base_encoder.yaml"],
    "training": {
        "train_steps": 300001,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "save_every_steps": 10000,
        "keep_every_steps": 100000,
        "curriculum": {
            "dims": {"start": 20, "end": 20, "inc": 1, "interval": 2000},
            "points": {"start": 21, "end": 21, "inc": 2, "interval": 2000}
        }
    },
    "model": {
        "family": "EncoderTF",
        "n_dims": 20,
        "n_layer": 12,
        "aux_loss_coef": 0.01,
        "router_noise": True,
        "noise_scale": 1.0,
        "normalize_attn": True,
        "encoder_activation": "relu"
    }
}

# Task definitions
NOISE_2 = [0.1, 0.5]
NOISE_4 = [0.1, 0.25, 0.5, 1.0]
NOISE_8 = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]

def get_tasks(noise_levels):
    return [{"name": "noisy_linear_regression", "kwargs": {"noise_std": n, "normalize_w": True}} for n in noise_levels]

# Experiment Definitions
experiments = [
    # GPU 0
    {"id": "E01", "name": "dense_2noise", "model": {"use_moe": False, "n_embd": 64, "n_head": 8}, "tasks": NOISE_2},
    {"id": "E02", "name": "dense_4noise", "model": {"use_moe": False, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    {"id": "E03", "name": "moe2_4noise", "model": {"use_moe": True, "num_experts": 2, "top_k": 1, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    {"id": "E04", "name": "moe4_4noise", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    
    # GPU 1
    {"id": "E05", "name": "moe8_4noise", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    {"id": "E06", "name": "moe16_4noise", "model": {"use_moe": True, "num_experts": 16, "top_k": 1, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    {"id": "E07", "name": "moe4_seq_routing", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "seq_level_routing": True, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    {"id": "E08", "name": "moe4_top2", "model": {"use_moe": True, "num_experts": 4, "top_k": 2, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    
    # GPU 2
    {"id": "E09", "name": "moe4_no_aux", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "aux_loss_coef": 0.0, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    {"id": "E10", "name": "moe4_no_noise", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "router_noise": False, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4},
    {"id": "E11", "name": "moe4_8noise", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "n_embd": 64, "n_head": 8}, "tasks": NOISE_8},
    {"id": "E12", "name": "moe8_8noise", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "n_embd": 64, "n_head": 8}, "tasks": NOISE_8},
    
    # GPU 3
    {"id": "E13", "name": "dense_curriculum", "model": {"use_moe": False, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4, 
     "curriculum": {"dims": {"start": 5, "end": 20, "inc": 1, "interval": 10000}, "points": {"start": 11, "end": 21, "inc": 1, "interval": 10000}}},
    {"id": "E14", "name": "moe4_curriculum", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "n_embd": 64, "n_head": 8}, "tasks": NOISE_4,
     "curriculum": {"dims": {"start": 5, "end": 20, "inc": 1, "interval": 10000}, "points": {"start": 11, "end": 21, "inc": 1, "interval": 10000}}},
    {"id": "E15", "name": "moe4_large", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "n_embd": 128, "n_head": 4}, "tasks": NOISE_4},
    {"id": "E16", "name": "dense_large", "model": {"use_moe": False, "n_embd": 128, "n_head": 4}, "tasks": NOISE_4},
]

output_dir = "/root/moe-icl/src/conf/encoder"

for exp in experiments:
    config = COMMON_CONFIG.copy()
    # Deep copy nested dicts to avoid reference issues
    config["training"] = COMMON_CONFIG["training"].copy()
    config["training"]["curriculum"] = COMMON_CONFIG["training"]["curriculum"].copy()
    config["model"] = COMMON_CONFIG["model"].copy()
    
    # Set specific fields
    filename = f"{exp['id']}_{exp['name']}.yaml"
    config["out_dir"] = f"../results/{exp['id']}_{exp['name']}"
    config["wandb"] = {
        "project": "moe-icl",
        "name": f"{exp['id']}_{exp['name']}",
        "entity": "jinruilin-aijobtech"
    }
    
    # Update Model Params
    config["model"].update(exp["model"])
    
    # Update Training Params
    config["training"]["tasks"] = get_tasks(exp["tasks"])
    if "curriculum" in exp:
        config["training"]["curriculum"] = exp["curriculum"]
        
    # Write file
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Generated {filepath}")

print("All configurations generated successfully.")
