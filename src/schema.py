from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
    stlist,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm", "EncoderTF", "llama_hf", "qwen_hf", "gemma_hf"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    # Decoder-specific options (HF LLaMA/Qwen/Gemma)
    "n_kv_head": merge(tinteger, nullable, default(None)),
    "mlp_hidden_mult": merge(tinteger, default(4)),
    "rmsnorm_eps": merge(tfloat, default(1e-6)),
    "use_rope": merge(tboolean, default(False)),
    "rope_theta": merge(tfloat, default(10000.0)),
    "qkv_bias": merge(tboolean, default(False)),
    "mlp_bias": merge(tboolean, default(False)),
    "max_seq_len": merge(tinteger, nullable, default(None)),
    "encoder_activation": merge(tstring, default("relu")),
    "normalize_attn": merge(tboolean, default(True)),
    # MoE configuration
    "use_moe": merge(tboolean, default(False)),
    "num_experts": merge(tinteger, default(4)),
    "top_k": merge(tinteger, default(1)),
    "seq_level_routing": merge(tboolean, default(False)),
    "moe_layers": merge(stlist(tinteger), nullable, default(None)),
    "aux_loss_coef": merge(tfloat, default(0.01)),
    "router_noise": merge(tboolean, default(True)),  # ST-MoE noisy routing
    "noise_scale": merge(tfloat, default(1.0)),  # Uniform(0, noise_scale)
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "linear_regression",
    "noisy_linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "quadratic_regression",
    "relu_2nn_regression",
    "decision_tree",
]

task_schema = {
    "name": merge(tstring, allowed(TASK_LIST)),
    "kwargs": merge(tdict, default({})),
}

training_schema = {
    # "task": merge(tstring, allowed(TASK_LIST), default("linear_regression")),
    # "task_kwargs": merge(tdict, default({})),
    "tasks": merge(stlist(task_schema), default([])),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian"])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "lasso_guided_opt": merge(tboolean, default(False)),
    "lasso_guided_opt_lam": merge(tfloat, default(1.0)),
    "lasso_guided_opt_layer": merge(tinteger, default(-2)),
    "lasso_guided_opt_token": merge(tinteger, default(-1)),
    "optimizer_reset": merge(tboolean, default(False)),
    "learning_rate_override": merge(tboolean, default(False)),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
