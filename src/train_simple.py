"""
Training script for MoE experiments.
Supports: dense, moe4, moe8, moe4_seq
"""
import os
import sys
import torch
import argparse
from tqdm import tqdm

from tasks import get_task_sampler
from samplers import get_data_sampler
from models import build_model
from types import SimpleNamespace

torch.backends.cudnn.benchmark = True

def train_step(model, xs, ys, optimizer, loss_func, use_moe=False):
    optimizer.zero_grad()
    if use_moe:
        output, aux_loss = model(xs, ys, return_aux_loss=True)
    else:
        output = model(xs, ys)
        aux_loss = torch.tensor(0.0)
    
    loss = loss_func(output[:, -1:], ys[:, -1:])
    total_loss = loss + aux_loss
    
    total_loss.backward()
    optimizer.step()
    return loss.item(), aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0, output.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, choices=['dense', 'moe4', 'moe8', 'moe4_seq'])
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Common config
    n_dims = 20
    n_positions = 21
    n_embd = 128
    n_layer = 12
    n_head = 4
    batch_size = 64
    lr = 1e-4
    
    # Experiment-specific MoE config
    moe_configs = {
        'dense': {'use_moe': False},
        'moe4': {'use_moe': True, 'num_experts': 4, 'top_k': 1, 'seq_level_routing': False, 'aux_loss_coef': 0.01},
        'moe8': {'use_moe': True, 'num_experts': 8, 'top_k': 1, 'seq_level_routing': False, 'aux_loss_coef': 0.01},
        'moe4_seq': {'use_moe': True, 'num_experts': 4, 'top_k': 1, 'seq_level_routing': True, 'aux_loss_coef': 0.01},
    }
    moe_cfg = moe_configs[args.exp]
    
    # Create model config
    conf = SimpleNamespace(
        family='EncoderTF',
        n_dims=n_dims, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        encoder_activation='relu', normalize_attn=True, **moe_cfg
    )
    conf.keys = lambda: conf.__dict__.keys()
    
    model = build_model(conf)
    model.cuda()
    model.train()
    print(f'Model: {model.name}, Params: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Data samplers
    data_sampler = get_data_sampler('gaussian', n_dims=n_dims)
    task_sampler_01 = get_task_sampler('noisy_linear_regression', n_dims, batch_size, noise_std=0.1, normalize_w=True)
    task_sampler_05 = get_task_sampler('noisy_linear_regression', n_dims, batch_size, noise_std=0.5, normalize_w=True)
    
    use_moe = moe_cfg.get('use_moe', False)
    
    pbar = tqdm(range(args.steps))
    for step in pbar:
        for task_sampler in [task_sampler_01, task_sampler_05]:
            xs = data_sampler.sample_xs(n_positions, batch_size, n_dims)
            task = task_sampler()
            ys = task.evaluate(xs)
            loss_func = task.get_training_metric()
            
            loss, aux, _ = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, use_moe=use_moe)
            
        pbar.set_description(f'loss={loss:.4f} aux={aux:.4f}')
        
        if step % 10000 == 0 and step > 0:
            save_path = f'/root/moe-icl/results/{args.exp}_step{step}.pt'
            torch.save(model.state_dict(), save_path)
            print(f'Saved to {save_path}')
    
    # Final save
    torch.save(model.state_dict(), f'/root/moe-icl/results/{args.exp}_final.pt')
    print(f'Training complete for {args.exp}!')

if __name__ == '__main__':
    main()
