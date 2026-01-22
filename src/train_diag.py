"""
Diagnostic training script for MoE hyperparameter tuning.
Allows testing different aux_loss coefficients.
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
    return loss.item(), aux_loss.item(), output.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aux_coef', type=float, default=0.01, help='aux loss coefficient')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str, default='diag')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Model config
    n_dims = 20
    n_positions = 21
    n_embd = 128
    n_layer = 12
    n_head = 4
    batch_size = 64
    
    use_moe = args.num_experts > 0
    
    if use_moe:
        conf = SimpleNamespace(
            family='EncoderTF',
            n_dims=n_dims, n_positions=n_positions, n_embd=n_embd, 
            n_layer=n_layer, n_head=n_head,
            encoder_activation='relu', normalize_attn=True,
            use_moe=True, num_experts=args.num_experts, top_k=1,
            seq_level_routing=False, aux_loss_coef=args.aux_coef
        )
    else:
        conf = SimpleNamespace(
            family='EncoderTF',
            n_dims=n_dims, n_positions=n_positions, n_embd=n_embd,
            n_layer=n_layer, n_head=n_head,
            encoder_activation='relu', normalize_attn=True,
            use_moe=False
        )
    conf.keys = lambda: conf.__dict__.keys()
    
    model = build_model(conf)
    model.cuda()
    model.train()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model: {model.name}, Params: {param_count:,}')
    print(f'Config: num_experts={args.num_experts}, aux_coef={args.aux_coef}, lr={args.lr}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data samplers
    data_sampler = get_data_sampler('gaussian', n_dims=n_dims)
    task_sampler_01 = get_task_sampler('noisy_linear_regression', n_dims, batch_size, 
                                        noise_std=0.1, normalize_w=True)
    task_sampler_05 = get_task_sampler('noisy_linear_regression', n_dims, batch_size,
                                        noise_std=0.5, normalize_w=True)
    
    losses = []
    pbar = tqdm(range(args.steps))
    for step in pbar:
        for task_sampler in [task_sampler_01, task_sampler_05]:
            xs = data_sampler.sample_xs(n_positions, batch_size, n_dims)
            task = task_sampler()
            ys = task.evaluate(xs)
            loss_func = task.get_training_metric()
            
            loss, aux, _ = train_step(model, xs.cuda(), ys.cuda(), optimizer, 
                                       loss_func, use_moe=use_moe)
        
        losses.append(loss)
        pbar.set_description(f'loss={loss:.4f} aux={aux:.4f}')
        
        if step % 1000 == 0 and step > 0:
            avg_loss = sum(losses[-1000:]) / 1000
            print(f'Step {step}: avg_loss={avg_loss:.4f}')
    
    # Final stats
    print(f'\nFinal loss: {loss:.4f}')
    print(f'Training complete for {args.name}!')

if __name__ == '__main__':
    main()
