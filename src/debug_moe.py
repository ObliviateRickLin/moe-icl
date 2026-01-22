"""Debug script to analyze MoE gradient flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import MoELayer

print('=== Testing MoE Gradient Flow ===')
n_embd = 128
batch, seq = 4, 21

moe = MoELayer(n_embd=n_embd, num_experts=4, top_k=1, aux_loss_coef=0.01)
moe.train()

x = torch.randn(batch, seq, n_embd, requires_grad=True)
target = torch.randn(batch, seq, n_embd)

output, aux_loss, info = moe(x, return_routing_info=True)

loss = F.mse_loss(output, target) + aux_loss
loss.backward()

print('w1 grad norm:', moe.w1.grad.norm().item())
print('w2 grad norm:', moe.w2.grad.norm().item())
print('router grad norm:', moe.router.gate.weight.grad.norm().item())
print('Expert usage:', info['expert_usage'])
print('aux_loss:', aux_loss.item())

print('\nPer-expert w1 grad norms:')
for i in range(4):
    grad_norm = moe.w1.grad[i].norm().item()
    print(f'  Expert {i}: {grad_norm:.6f}')

# Check if output is reasonable
print(f'\nOutput stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}')
print(f'Target stats: mean={target.mean().item():.4f}, std={target.std().item():.4f}')
