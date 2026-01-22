"""
LLaMA/Qwen/DeepSeek-style decoder-only transformer (from-scratch).

This module provides three model classes with identical core structure:
  - LlamaDecoderModel
  - QwenDecoderModel
  - DeepSeekDecoderModel

Key features:
  - Decoder-only causal self-attention
  - RMSNorm (pre-norm)
  - SwiGLU MLP
  - Optional RoPE (off by default to match existing code style)
  - Optional MoE replacement for MLP (SwiGLU experts)

Designed to integrate with existing training loop (xs, ys inputs).
"""

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        qkv_bias: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if self.use_rope:
            cos, sin = _build_rope_cache(T, self.head_dim, self.rope_theta, x.device, x.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.o_proj(y)
        return y


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_mult: int = 4, bias: bool = False):
        super().__init__()
        hidden_dim = n_embd * hidden_mult
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SwiGLUMoE(nn.Module):
    def __init__(
        self,
        n_embd: int,
        num_experts: int = 4,
        top_k: int = 1,
        seq_level_routing: bool = False,
        aux_loss_coef: float = 0.01,
        hidden_mult: int = 4,
        router_noise: bool = True,
        noise_scale: float = 1.0,
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.seq_level_routing = seq_level_routing
        self.aux_loss_coef = aux_loss_coef
        self.router_noise = router_noise
        self.noise_scale = noise_scale
        self.hidden_dim = n_embd * hidden_mult

        self.router = nn.Linear(n_embd, num_experts, bias=False)

        self.w1 = nn.Parameter(torch.empty(num_experts, n_embd, self.hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, n_embd, self.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, self.hidden_dim, n_embd))

        if mlp_bias:
            self.b1 = nn.Parameter(torch.zeros(num_experts, self.hidden_dim))
            self.b2 = nn.Parameter(torch.zeros(num_experts, self.hidden_dim))
            self.b3 = nn.Parameter(torch.zeros(num_experts, n_embd))
        else:
            self.b1 = None
            self.b2 = None
            self.b3 = None

        # Init to match nn.Linear default
        for i in range(num_experts):
            bound1 = 1 / math.sqrt(n_embd)
            nn.init.uniform_(self.w1[i], -bound1, bound1)
            nn.init.uniform_(self.w2[i], -bound1, bound1)
            bound3 = 1 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(self.w3[i], -bound3, bound3)

    def forward(self, x: torch.Tensor, return_routing_info: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        B, T, C = x.shape

        if self.seq_level_routing:
            x_pooled = x.mean(dim=1)
            router_logits = self.router(x_pooled).unsqueeze(1).expand(-1, T, -1)
        else:
            router_logits = self.router(x)

        router_probs = F.softmax(router_logits, dim=-1)

        if self.training and self.router_noise:
            noise = torch.rand_like(router_logits) * self.noise_scale
            noisy_logits = router_logits + noise
            noisy_probs = F.softmax(noisy_logits, dim=-1)
            _, top_k_indices = noisy_probs.topk(self.top_k, dim=-1)
        else:
            _, top_k_indices = router_probs.topk(self.top_k, dim=-1)

        top_k_probs = torch.gather(router_probs, dim=-1, index=top_k_indices)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        x_flat = x.view(-1, C)
        hidden1 = torch.einsum("bc,ech->beh", x_flat, self.w1)
        hidden2 = torch.einsum("bc,ech->beh", x_flat, self.w2)
        if self.b1 is not None:
            hidden1 = hidden1 + self.b1.unsqueeze(0)
            hidden2 = hidden2 + self.b2.unsqueeze(0)
        hidden = F.silu(hidden1) * hidden2
        out_all = torch.einsum("beh,ehc->bec", hidden, self.w3)
        if self.b3 is not None:
            out_all = out_all + self.b3.unsqueeze(0)

        out_all = out_all.view(B, T, self.num_experts, C)
        gather_idx = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, C)
        selected = torch.gather(out_all, dim=2, index=gather_idx)
        out = (selected * top_k_probs.unsqueeze(-1)).sum(dim=2)

        aux_loss = self._compute_aux_loss(router_probs)

        routing_info = None
        if return_routing_info:
            routing_info = {
                "router_probs": router_probs.detach(),
                "top_k_indices": top_k_indices.detach(),
            }

        return out, aux_loss, routing_info

    def _compute_aux_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        probs_flat = router_probs.view(-1, self.num_experts)
        top_expert = router_probs.argmax(dim=-1)
        expert_counts = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            expert_counts[i] = (top_expert == i).float().sum()
        f = expert_counts / top_expert.numel()
        p = probs_flat.mean(dim=0)
        aux_loss = self.aux_loss_coef * self.num_experts * (f * p).sum()
        return aux_loss


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: Optional[int],
        use_rope: bool,
        rope_theta: float,
        rmsnorm_eps: float,
        mlp_hidden_mult: int,
        qkv_bias: bool,
        mlp_bias: bool,
        use_moe: bool,
        num_experts: int,
        top_k: int,
        seq_level_routing: bool,
        aux_loss_coef: float,
        router_noise: bool,
        noise_scale: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd, eps=rmsnorm_eps)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        self.mlp_norm = RMSNorm(n_embd, eps=rmsnorm_eps)
        self.use_moe = use_moe
        if use_moe:
            self.mlp = SwiGLUMoE(
                n_embd=n_embd,
                num_experts=num_experts,
                top_k=top_k,
                seq_level_routing=seq_level_routing,
                aux_loss_coef=aux_loss_coef,
                hidden_mult=mlp_hidden_mult,
                router_noise=router_noise,
                noise_scale=noise_scale,
                mlp_bias=mlp_bias,
            )
        else:
            self.mlp = SwiGLU(n_embd=n_embd, hidden_mult=mlp_hidden_mult, bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x + self.attn(self.attn_norm(x))
        if self.use_moe:
            mlp_out, aux_loss, _ = self.mlp(self.mlp_norm(h))
            h = h + mlp_out
            return h, aux_loss
        h = h + self.mlp(self.mlp_norm(h))
        return h, torch.tensor(0.0, device=x.device)


class BaseDecoderModel(nn.Module):
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        n_kv_head: Optional[int],
        mlp_hidden_mult: int,
        rmsnorm_eps: float,
        use_rope: bool,
        rope_theta: float,
        qkv_bias: bool,
        mlp_bias: bool,
        use_moe: bool,
        num_experts: int,
        top_k: int,
        seq_level_routing: bool,
        aux_loss_coef: float,
        router_noise: bool,
        noise_scale: float,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.use_moe = use_moe

        max_seq_len = max_seq_len or (2 * n_positions)

        self._read_in = nn.Linear(n_dims, n_embd, bias=False)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                    rmsnorm_eps=rmsnorm_eps,
                    mlp_hidden_mult=mlp_hidden_mult,
                    qkv_bias=qkv_bias,
                    mlp_bias=mlp_bias,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                    seq_level_routing=seq_level_routing,
                    aux_loss_coef=aux_loss_coef,
                    router_noise=router_noise,
                    noise_scale=noise_scale,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layer)
            ]
        )
        self.final_norm = RMSNorm(n_embd, eps=rmsnorm_eps)
        self._read_out = nn.Linear(n_embd, 1, bias=False)

    @staticmethod
    def _combine(xs_b, ys_b):
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None, return_aux_loss: bool = False):
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.tensor(inds, device=ys.device)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        h = self._read_in(zs)

        total_aux = torch.tensor(0.0, device=h.device)
        for blk in self.blocks:
            h, aux = blk(h)
            total_aux = total_aux + aux

        h = self.final_norm(h)
        pred = self._read_out(h)
        # pick x-token positions
        output = pred[:, ::2, 0][:, inds]

        if return_aux_loss:
            return output, total_aux
        return output


class LlamaDecoderModel(BaseDecoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"llama_decoder_{self.n_layer}L_{self.n_embd}d_{self.n_head}h"


class QwenDecoderModel(BaseDecoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"qwen_decoder_{self.n_layer}L_{self.n_embd}d_{self.n_head}h"


class DeepSeekDecoderModel(BaseDecoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"deepseek_decoder_{self.n_layer}L_{self.n_embd}d_{self.n_head}h"
