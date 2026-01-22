"""
Standalone training script that supports YAML configs and WandB logging.
Eliminates the 'quinine' dependency while preserving the YAML workflow.
"""
import os
import sys
import torch
import yaml
import argparse
import wandb
from tqdm import tqdm
from types import SimpleNamespace
from tasks import get_task_sampler
from samplers import get_data_sampler
from models import build_model

def merge_dicts(dict1, dict2):
    """Recursively merge two dictionaries."""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def load_config(config_path):
    """Load and merge YAML configs based on 'inherit'."""
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    
    if 'inherit' in conf and conf['inherit']:
        base_dir = os.path.dirname(config_path)
        for inherit_path in conf['inherit']:
            # Handle relative paths
            full_inherit_path = os.path.join(base_dir, inherit_path)
            if os.path.exists(full_inherit_path):
                base_conf = load_config(full_inherit_path)
                conf = merge_dicts(base_conf, conf)
    
    return conf

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    if isinstance(d, dict):
        # Add keys() method to mimic Quinine behavior
        ns = SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        ns.keys = lambda: ns.__dict__.keys()
        return ns
    return d

def train_step(model, xs, ys, optimizer, loss_func, use_moe=False):
    optimizer.zero_grad()
    if use_moe:
        output, aux_loss = model(xs, ys, return_aux_loss=True)
    else:
        output = model(xs, ys)
        aux_loss = torch.tensor(0.0)
    
    # Encoder mode: predict on final token
    loss = loss_func(output[:, -1:], ys[:, -1:])
    total_loss = loss + aux_loss
    
    total_loss.backward()
    optimizer.step()
    return loss.item(), aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load and prepare config
    raw_conf = load_config(args.config)
    conf = dict_to_namespace(raw_conf)
    
    # 2. Init WandB
    wandb_cfg = raw_conf.get('wandb', {})
    wandb.init(
        project=wandb_cfg.get('project', 'moe-icl'),
        name=wandb_cfg.get('name', os.path.basename(args.config).replace('.yaml', '')),
        config=raw_conf
    )
    
    # 3. Build Model
    model = build_model(conf.model)
    model.to(device)
    model.train()
    print(f'Model: {model.name}, Params: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.training.learning_rate)
    
    # 4. Setup Samplers
    data_sampler = get_data_sampler(conf.training.data, n_dims=conf.model.n_dims)
    task_samplers = []
    for task_conf in conf.training.tasks:
        ts = get_task_sampler(
            task_conf['name'], 
            conf.model.n_dims, 
            conf.training.batch_size, 
            **task_conf['kwargs']
        )
        task_samplers.append(ts)
    
    use_moe = getattr(conf.model, 'use_moe', False)
    train_steps = conf.training.train_steps
    batch_size = conf.training.batch_size
    n_positions = conf.model.n_positions
    n_dims = conf.model.n_dims
    
    # 5. Training Loop
    pbar = tqdm(range(train_steps))
    for step in pbar:
        # Sample task
        task_sampler = task_samplers[step % len(task_samplers)]
        xs = data_sampler.sample_xs(n_positions, batch_size, n_dims)
        task = task_sampler()
        ys = task.evaluate(xs)
        loss_func = task.get_training_metric()
        
        loss, aux = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func, use_moe=use_moe)
        
        if step % 100 == 0:
            wandb.log({
                "loss": loss,
                "aux_loss": aux,
                "total_loss": loss + aux,
                "step": step
            })
            pbar.set_description(f'step={step} loss={loss:.4f} aux={aux:.4f}')
        
        if step % 20000 == 0 and step > 0:
            os.makedirs('../results', exist_ok=True)
            save_path = f'../results/{wandb.run.name}_step{step}.pt'
            torch.save(model.state_dict(), save_path)
    
    # Final save
    torch.save(model.state_dict(), f'../results/{wandb.run.name}_final.pt')
    wandb.finish()
    print('Training complete!')

if __name__ == '__main__':
    main()
