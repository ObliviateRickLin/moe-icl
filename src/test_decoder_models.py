"""
CPU sanity checks for new decoder model families (llama/qwen/deepseek).
Runs a tiny forward pass with and without MoE to verify shapes and aux_loss.
"""

import torch
from types import SimpleNamespace
from models import build_model


def _make_conf(family, use_moe=False, qkv_bias=False, n_kv_head=None):
    conf = SimpleNamespace(
        family=family,
        n_dims=8,
        n_positions=5,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_kv_head=n_kv_head,
        mlp_hidden_mult=4,
        rmsnorm_eps=1e-6,
        use_rope=False,
        rope_theta=10000.0,
        qkv_bias=qkv_bias,
        mlp_bias=False,
        use_moe=use_moe,
        num_experts=4,
        top_k=1,
        seq_level_routing=False,
        aux_loss_coef=0.01,
        router_noise=True,
        noise_scale=1.0,
    )
    conf.keys = lambda: conf.__dict__.keys()
    return conf


def _run_one(family, qkv_bias=False, n_kv_head=None):
    xs = torch.randn(2, 4, 8)
    ys = torch.randn(2, 4)

    conf_dense = _make_conf(family, use_moe=False, qkv_bias=qkv_bias, n_kv_head=n_kv_head)
    model_dense = build_model(conf_dense)
    with torch.no_grad():
        out = model_dense(xs, ys)
    print(f"{family} dense out shape:", out.shape)

    conf_moe = _make_conf(family, use_moe=True, qkv_bias=qkv_bias, n_kv_head=n_kv_head)
    model_moe = build_model(conf_moe)
    with torch.no_grad():
        out, aux = model_moe(xs, ys, return_aux_loss=True)
    print(f"{family} moe out shape:", out.shape, "aux:", float(aux))


if __name__ == "__main__":
    _run_one("llama", qkv_bias=False, n_kv_head=None)
    _run_one("qwen", qkv_bias=True, n_kv_head=None)
    _run_one("deepseek", qkv_bias=False, n_kv_head=2)
