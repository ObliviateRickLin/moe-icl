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
        "n_embd": 64,
        "n_head": 8,
        "aux_loss_coef": 0.01,
        "router_noise": True,
        "noise_scale": 1.0,
        "normalize_attn": True,
        "encoder_activation": "relu"
    }
}

# Task Definitions
STANDARD_TASKS = [
    {"name": "noisy_linear_regression", "kwargs": {"noise_std": 0.1, "normalize_w": True}},
    {"name": "noisy_linear_regression", "kwargs": {"noise_std": 0.25, "normalize_w": True}},
    {"name": "noisy_linear_regression", "kwargs": {"noise_std": 0.5, "normalize_w": True}},
    {"name": "noisy_linear_regression", "kwargs": {"noise_std": 1.0, "normalize_w": True}},
]

GEN_TASKS = [
    {"name": "linear_regression", "kwargs": {"normalize_w": True}},
    {"name": "sparse_linear_regression", "kwargs": {"sparsity": 3, "normalize_w": True}},
    {"name": "linear_classification", "kwargs": {"normalize_w": True}},
    {"name": "quadratic_regression", "kwargs": {"normalize_w": True}},
    {"name": "relu_2nn_regression", "kwargs": {"hidden_layer_size": 4}},
]

NON_LIN_TASKS = [
    {"name": "quadratic_regression", "kwargs": {"normalize_w": True}},
    {"name": "relu_2nn_regression", "kwargs": {"hidden_layer_size": 4}},
    {"name": "decision_tree", "kwargs": {"depth": 4}},
]

LOGIC_TASKS = [
    {"name": "linear_classification", "kwargs": {"normalize_w": True}},
    {"name": "decision_tree", "kwargs": {"depth": 4}},
]

MULTI_SPARSE_TASKS = [
    {"name": "sparse_linear_regression", "kwargs": {"sparsity": k, "normalize_w": True}}
    for k in [1, 3, 5, 10, 20]
]

# Helper to get specific noise tasks
def get_noise_tasks(noise_levels):
    return [{"name": "noisy_linear_regression", "kwargs": {"noise_std": n, "normalize_w": True}} for n in noise_levels]

# Helper to get sparse tasks
def get_sparse_tasks(k_list):
    return [{"name": "sparse_linear_regression", "kwargs": {"sparsity": k, "normalize_w": True}} for k in k_list]

# Experiment Definitions
experiments = []

# --- Phase 1: E01-E16 ---

# GPU 0
experiments.extend([
    {"id": "E01", "name": "dense_2noise", "model": {"use_moe": False}, "tasks": get_noise_tasks([0.1, 0.5])},
    {"id": "E02", "name": "dense_4noise", "model": {"use_moe": False}, "tasks": STANDARD_TASKS},
    {"id": "E03", "name": "moe2_4noise", "model": {"use_moe": True, "num_experts": 2, "top_k": 1}, "tasks": STANDARD_TASKS},
    {"id": "E04", "name": "moe4_4noise", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": STANDARD_TASKS},
])

# GPU 1
experiments.extend([
    {"id": "E05", "name": "moe8_4noise", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": STANDARD_TASKS},
    {"id": "E06", "name": "moe16_4noise", "model": {"use_moe": True, "num_experts": 16, "top_k": 1}, "tasks": STANDARD_TASKS},
    {"id": "E07", "name": "moe4_seq_routing", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "seq_level_routing": True}, "tasks": STANDARD_TASKS},
    {"id": "E08", "name": "moe4_top2", "model": {"use_moe": True, "num_experts": 4, "top_k": 2}, "tasks": STANDARD_TASKS},
])

# GPU 2
experiments.extend([
    {"id": "E09", "name": "moe4_no_aux", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "aux_loss_coef": 0.0}, "tasks": STANDARD_TASKS},
    {"id": "E10", "name": "moe4_no_noise", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "router_noise": False}, "tasks": STANDARD_TASKS},
    {"id": "E11", "name": "moe4_8noise", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": get_noise_tasks([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5])},
    {"id": "E12", "name": "moe8_8noise", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": get_noise_tasks([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5])},
])

# GPU 3
experiments.extend([
    {"id": "E13", "name": "dense_curriculum", "model": {"use_moe": False}, "tasks": STANDARD_TASKS,
     "curriculum": {"dims": {"start": 5, "end": 20, "inc": 1, "interval": 10000}, "points": {"start": 11, "end": 21, "inc": 1, "interval": 10000}}},
    {"id": "E14", "name": "moe4_curriculum", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": STANDARD_TASKS,
     "curriculum": {"dims": {"start": 5, "end": 20, "inc": 1, "interval": 10000}, "points": {"start": 11, "end": 21, "inc": 1, "interval": 10000}}},
    {"id": "E15", "name": "moe4_large", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "n_embd": 128, "n_head": 4}, "tasks": STANDARD_TASKS},
    {"id": "E16", "name": "dense_large", "model": {"use_moe": False, "n_embd": 128, "n_head": 4}, "tasks": STANDARD_TASKS},
])

# --- Phase 2: E17-E64 ---

# GPU 0: The Generalist
experiments.extend([
    {"id": "E17", "name": "dense_genmix", "model": {"use_moe": False}, "tasks": GEN_TASKS},
    {"id": "E18", "name": "moe4_genmix", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": GEN_TASKS},
    {"id": "E19", "name": "moe8_genmix", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": GEN_TASKS},
    {"id": "E20", "name": "moe16_genmix", "model": {"use_moe": True, "num_experts": 16, "top_k": 1}, "tasks": GEN_TASKS},
    {"id": "E21", "name": "moe8_genseq", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "seq_level_routing": True}, "tasks": GEN_TASKS},
    {"id": "E22", "name": "moe16_genseq", "model": {"use_moe": True, "num_experts": 16, "top_k": 1, "seq_level_routing": True}, "tasks": GEN_TASKS},
    {"id": "E23", "name": "dense_nonlin", "model": {"use_moe": False}, "tasks": NON_LIN_TASKS},
    {"id": "E24", "name": "moe4_nonlin", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": NON_LIN_TASKS},
    {"id": "E25", "name": "moe8_nonlin", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": NON_LIN_TASKS},
    {"id": "E26", "name": "moe4_logic", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": LOGIC_TASKS},
    {"id": "E27", "name": "moe8_logic", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": LOGIC_TASKS},
    {"id": "E28", "name": "moe8_top2", "model": {"use_moe": True, "num_experts": 8, "top_k": 2}, "tasks": GEN_TASKS},
])

# GPU 1: The Sparsity
experiments.extend([
    {"id": "E29", "name": "dense_multisp", "model": {"use_moe": False}, "tasks": MULTI_SPARSE_TASKS},
    {"id": "E30", "name": "moe4_multisp", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": MULTI_SPARSE_TASKS},
    {"id": "E31", "name": "moe8_multisp", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": MULTI_SPARSE_TASKS},
    {"id": "E32", "name": "moe16_multisp", "model": {"use_moe": True, "num_experts": 16, "top_k": 1}, "tasks": MULTI_SPARSE_TASKS},
    {"id": "E33", "name": "moe8_seqsp", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "seq_level_routing": True}, "tasks": MULTI_SPARSE_TASKS},
    {"id": "E34", "name": "moe16_seqsp", "model": {"use_moe": True, "num_experts": 16, "top_k": 1, "seq_level_routing": True}, "tasks": MULTI_SPARSE_TASKS},
    {"id": "E35", "name": "dense_highsp", "model": {"use_moe": False}, "tasks": get_sparse_tasks([1])},
    {"id": "E36", "name": "moe8_highsp", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": get_sparse_tasks([1])},
    {"id": "E37", "name": "moe4_lasso", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": MULTI_SPARSE_TASKS, "training": {"lasso_guided_opt": True, "lasso_guided_opt_lam": 0.1}},
    {"id": "E38", "name": "moe8_lasso", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": MULTI_SPARSE_TASKS, "training": {"lasso_guided_opt": True, "lasso_guided_opt_lam": 0.1}},
    {"id": "E39", "name": "moe4_ood", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": get_sparse_tasks([1, 20])},
    {"id": "E40", "name": "moe8_ood", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": get_sparse_tasks([1, 20])},
])

# GPU 2: Data Efficiency
experiments.extend([
    {"id": "E41", "name": "dense_10k", "model": {"use_moe": False}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 10000}},
    {"id": "E42", "name": "moe8_10k", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 10000}},
    {"id": "E43", "name": "moe64_10k", "model": {"use_moe": True, "num_experts": 64, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 10000}},
    {"id": "E44", "name": "dense_100k", "model": {"use_moe": False}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 100000}},
    {"id": "E45", "name": "moe8_100k", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 100000}},
    {"id": "E46", "name": "moe64_100k", "model": {"use_moe": True, "num_experts": 64, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 100000}},
    {"id": "E47", "name": "dense_1m", "model": {"use_moe": False}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 1000000}},
    {"id": "E48", "name": "moe8_1m", "model": {"use_moe": True, "num_experts": 8, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 1000000}},
    {"id": "E49", "name": "moe64_1m", "model": {"use_moe": True, "num_experts": 64, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"num_training_examples": 1000000}},
    {"id": "E50", "name": "moe4_fast", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"train_steps": 50000}},
    {"id": "E51", "name": "moe16_fast", "model": {"use_moe": True, "num_experts": 16, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"train_steps": 50000}},
    {"id": "E52", "name": "moe64_fast", "model": {"use_moe": True, "num_experts": 64, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"train_steps": 50000}},
])

# GPU 3: Deep Ablations
experiments.extend([
    {"id": "E53", "name": "moe8_aux0", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "aux_loss_coef": 0.0}, "tasks": STANDARD_TASKS},
    {"id": "E54", "name": "moe8_aux10", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "aux_loss_coef": 10.0}, "tasks": STANDARD_TASKS},
    {"id": "E55", "name": "moe8_fix", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "router_noise": False}, "tasks": STANDARD_TASKS},
    {"id": "E56", "name": "moe8_rand", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "noise_scale": 5.0}, "tasks": STANDARD_TASKS},
    {"id": "E57", "name": "moe8_top4", "model": {"use_moe": True, "num_experts": 8, "top_k": 4}, "tasks": STANDARD_TASKS},
    {"id": "E58", "name": "moe8_top8", "model": {"use_moe": True, "num_experts": 8, "top_k": 8}, "tasks": STANDARD_TASKS},
    {"id": "E59", "name": "moe4_seqaux0", "model": {"use_moe": True, "num_experts": 4, "top_k": 1, "seq_level_routing": True, "aux_loss_coef": 0.0}, "tasks": STANDARD_TASKS},
    {"id": "E60", "name": "moe16_seqaux0", "model": {"use_moe": True, "num_experts": 16, "top_k": 1, "seq_level_routing": True, "aux_loss_coef": 0.0}, "tasks": STANDARD_TASKS},
    {"id": "E61", "name": "dense_b256", "model": {"use_moe": False}, "tasks": STANDARD_TASKS, "training": {"batch_size": 256}},
    {"id": "E62", "name": "moe4_b256", "model": {"use_moe": True, "num_experts": 4, "top_k": 1}, "tasks": STANDARD_TASKS, "training": {"batch_size": 256}},
    {"id": "E63", "name": "moe8_nonorm", "model": {"use_moe": True, "num_experts": 8, "top_k": 1, "normalize_attn": False}, "tasks": STANDARD_TASKS},
    {"id": "E64", "name": "moe32", "model": {"use_moe": True, "num_experts": 32, "top_k": 1}, "tasks": STANDARD_TASKS},
])

output_dir = "/root/moe-icl/src/conf/encoder"

for exp in experiments:
    config = COMMON_CONFIG.copy()
    # Deep copy nested dicts
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
    config["model"].update(exp.get("model", {}))
    
    # Update Training Params
    if "training" in exp:
        config["training"].update(exp["training"])
        
    config["training"]["tasks"] = exp["tasks"]
    if "curriculum" in exp:
        config["training"]["curriculum"] = exp["curriculum"]
        
    # Write file
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Generated {filepath}")

print("All 64 configurations generated successfully.")
