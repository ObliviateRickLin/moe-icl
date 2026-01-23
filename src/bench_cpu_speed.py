import argparse
import time

import torch

from llama_models import LlamaDecoderModel
from models import TransformerModel


def _make_batches(device, batch_size, n_points, n_dims, num_batches):
    batches = []
    for _ in range(num_batches):
        xs = torch.randn(batch_size, n_points, n_dims, device=device)
        ys = torch.randn(batch_size, n_points, device=device)
        batches.append((xs, ys))
    return batches


def _bench_model(model, batches, steps, warmup, tasks_per_step):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for i in range(warmup * max(tasks_per_step, 1)):
        xs, ys = batches[i % len(batches)]
        opt.zero_grad(set_to_none=True)
        out = model(xs, ys)
        loss = (out - ys).pow(2).mean()
        loss.backward()
        opt.step()

    # Timed
    start = time.perf_counter()
    total_iters = steps * max(tasks_per_step, 1)
    for i in range(total_iters):
        xs, ys = batches[i % len(batches)]
        opt.zero_grad(set_to_none=True)
        out = model(xs, ys)
        loss = (out - ys).pow(2).mean()
        loss.backward()
        opt.step()
    end = time.perf_counter()

    # Return seconds per "training step" (which may include multiple tasks)
    return (end - start) / max(steps, 1)


def _heads_for_embd(n_embd):
    # Keep head_dim around 32 when possible
    return max(1, n_embd // 32)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-points", type=int, default=21)
    parser.add_argument("--n-dims", type=int, default=20)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tasks-per-step", type=int, default=1)
    parser.add_argument("--widths", type=str, default="64,128,256")
    args = parser.parse_args()

    device = torch.device("cpu")
    widths = [int(x) for x in args.widths.split(",") if x.strip()]

    print("CPU benchmark (forward+backward+step)")
    print(f"batch={args.batch_size} points={args.n_points} dims={args.n_dims} layers={args.n_layer} "
          f"steps={args.steps} warmup={args.warmup} tasks_per_step={args.tasks_per_step}")
    print("-" * 90)
    print(f"{'model':<28} {'embd':>5} {'heads':>5} {'sec/step':>10}")

    # Prepare batches to simulate multiple tasks per step
    batches = _make_batches(
        device,
        args.batch_size,
        args.n_points,
        args.n_dims,
        num_batches=max(args.tasks_per_step, args.steps + args.warmup),
    )

    for n_embd in widths:
        n_head = _heads_for_embd(n_embd)

        # HF GPT2 (TransformerModel wrapper)
        gpt = TransformerModel(
            n_dims=args.n_dims,
            n_positions=args.n_points,
            n_embd=n_embd,
            n_layer=args.n_layer,
            n_head=n_head,
            use_moe=False,
        ).to(device)
        sec = _bench_model(gpt, batches, args.steps, args.warmup, args.tasks_per_step)
        print(f"{'gpt2_hf':<28} {n_embd:>5} {n_head:>5} {sec:>10.4f}")

        # LLaMA-style decoder
        llama = LlamaDecoderModel(
            n_dims=args.n_dims,
            n_positions=args.n_points,
            n_embd=n_embd,
            n_layer=args.n_layer,
            n_head=n_head,
            n_kv_head=None,
            mlp_hidden_mult=4,
            rmsnorm_eps=1e-6,
            use_rope=False,
            rope_theta=10000.0,
            qkv_bias=False,
            mlp_bias=False,
            use_moe=False,
            num_experts=4,
            top_k=1,
            seq_level_routing=False,
            aux_loss_coef=0.0,
            router_noise=False,
            noise_scale=1.0,
            max_seq_len=None,
        ).to(device)
        sec = _bench_model(llama, batches, args.steps, args.warmup, args.tasks_per_step)
        print(f"{'llama_swiglu':<28} {n_embd:>5} {n_head:>5} {sec:>10.4f}")


if __name__ == "__main__":
    run()
