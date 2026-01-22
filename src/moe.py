"""
Mixture of Experts (MoE) Layer Implementation for Transformers as Statisticians

This module implements a sparse MoE layer that can replace the FFN in EncoderTransformer.
Key features:
- Top-k sparse routing (default k=1 for Switch Transformer style)
- Token-level or sequence-level routing
- Load-balancing auxiliary loss (Switch Transformer Eq 4-6)

Reference: 
- Switch Transformers: https://jmlr.org/papers/volume23/21-0998/21-0998.pdf
- MoE Generalization: https://openreview.net/pdf?id=pdHYsuezlU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Expert(nn.Module):
    """Single expert: 2-layer ReLU FFN, same structure as original TaS FFN."""
    
    def __init__(self, n_embd: int, hidden_mult: int = 1):
        """
        Args:
            n_embd: Input/output embedding dimension
            hidden_mult: Hidden dimension multiplier (hidden = n_embd * hidden_mult)
        """
        super().__init__()
        hidden_dim = n_embd * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embd),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """Gating network that produces routing logits over experts."""
    
    def __init__(self, n_embd: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(n_embd, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, seq, n_embd) or (batch, n_embd) for sequence-level
        Returns:
            logits: Shape (batch, seq, num_experts) or (batch, num_experts)
        """
        return self.gate(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with sparse top-k routing.
    
    Implements: MoE(h) = sum_{t in TopK(g(h))} p_t(h) * E_t(h)
    
    Key features:
    - Top-k sparse routing with softmax weights
    - Optional sequence-level routing (all tokens share same expert)
    - Load-balancing auxiliary loss to prevent expert collapse
    - OPTIMIZED: Batched expert computation for GPU efficiency
    """
    
    def __init__(
        self, 
        n_embd: int, 
        num_experts: int = 4, 
        top_k: int = 1,
        seq_level_routing: bool = False,
        aux_loss_coef: float = 0.01,
        hidden_mult: int = 1,
        router_noise: bool = True,  # Enable noisy routing for smooth top-k
        noise_scale: float = 1.0,   # Scale for Uniform(0, noise_scale) noise
    ):
        """
        Args:
            n_embd: Embedding dimension
            num_experts: Number of expert FFNs (T in theory)
            top_k: Number of experts to route to per token
            seq_level_routing: If True, entire sequence uses same expert(s)
            aux_loss_coef: Coefficient for load-balancing loss
            hidden_mult: Hidden dimension multiplier for experts
        """
        super().__init__()
        
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.seq_level_routing = seq_level_routing
        self.aux_loss_coef = aux_loss_coef
        self.router_noise = router_noise
        self.noise_scale = noise_scale
        hidden_dim = n_embd * hidden_mult
        self.hidden_dim = hidden_dim
        
        # Router network
        self.router = Router(n_embd, num_experts)
        
        # OPTIMIZED: Batched expert weights instead of ModuleList
        # Shape: (num_experts, n_embd, hidden_dim) and (num_experts, hidden_dim, n_embd)
        self.w1 = nn.Parameter(torch.empty(num_experts, n_embd, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, n_embd))
        
        # Initialize weights - MUST match nn.Linear default initialization
        # nn.Linear uses kaiming_uniform_ with a=sqrt(5), which is equivalent to:
        # bound = 1 / sqrt(fan_in), uniform(-bound, bound)
        import math
        for i in range(num_experts):
            # w1: fan_in = n_embd
            bound1 = 1 / math.sqrt(n_embd)
            nn.init.uniform_(self.w1[i], -bound1, bound1)
            # w2: fan_in = hidden_dim  
            bound2 = 1 / math.sqrt(hidden_dim)
            nn.init.uniform_(self.w2[i], -bound2, bound2)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_routing_info: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        OPTIMIZED Forward pass through MoE layer using batched computation.
        
        Args:
            x: Input tensor, shape (batch, seq, n_embd)
            return_routing_info: If True, return additional routing statistics
            
        Returns:
            output: Same shape as input (batch, seq, n_embd)
            aux_loss: Load-balancing auxiliary loss (scalar tensor)
            routing_info: Optional dict with routing statistics
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Get routing logits
        if self.seq_level_routing:
            x_pooled = x.mean(dim=1)  # (batch, n_embd)
            router_logits = self.router(x_pooled)  # (batch, num_experts)
            router_logits = router_logits.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            router_logits = self.router(x)  # (batch, seq, num_experts)
        
        # Compute routing weights via softmax (for weighted output)
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq, num_experts)
        
        # Noisy Top-k selection: k* = argmax(g(x) + ξ) where ξ ~ Unif(0, noise_scale)
        # This smooths the discrete selection for better gradient flow (per paper)
        if self.training and self.router_noise:
            # Add Uniform(0, noise_scale) noise to logits before top-k
            noise = torch.rand_like(router_logits) * self.noise_scale
            noisy_logits = router_logits + noise
            noisy_probs = F.softmax(noisy_logits, dim=-1)
            _, top_k_indices = noisy_probs.topk(self.top_k, dim=-1)
        else:
            _, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        
        # Gather the actual probabilities for selected experts (use clean probs for weighting)
        top_k_probs = torch.gather(router_probs, dim=-1, index=top_k_indices)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # OPTIMIZED: Batched expert computation
        # Flatten input: (batch * seq, n_embd)
        x_flat = x.view(-1, n_embd)
        
        # Compute ALL expert outputs in parallel: (batch*seq, num_experts, n_embd)
        # Step 1: x @ w1 -> hidden for all experts
        # x_flat: (B*S, E) -> (B*S, 1, E) for broadcasting
        # w1: (T, E, H) -> all_hidden: (B*S, T, H)
        all_hidden = torch.einsum('be,teh->bth', x_flat, self.w1)
        all_hidden = F.relu(all_hidden)
        
        # Step 2: hidden @ w2 -> output for all experts
        # all_hidden: (B*S, T, H), w2: (T, H, E) -> all_expert_out: (B*S, T, E)
        all_expert_out = torch.einsum('bth,the->bte', all_hidden, self.w2)
        
        # Reshape for gathering: (batch, seq, num_experts, n_embd)
        all_expert_out = all_expert_out.view(batch_size, seq_len, self.num_experts, n_embd)
        
        # Gather selected experts and weight
        # top_k_indices: (batch, seq, top_k) -> expand to (batch, seq, top_k, n_embd)
        gather_idx = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, n_embd)
        selected_out = torch.gather(all_expert_out, dim=2, index=gather_idx)  # (batch, seq, top_k, n_embd)
        
        # Weight by routing probabilities
        # top_k_probs: (batch, seq, top_k) -> (batch, seq, top_k, 1)
        output = (selected_out * top_k_probs.unsqueeze(-1)).sum(dim=2)  # (batch, seq, n_embd)
        
        # Compute load-balancing auxiliary loss
        aux_loss = self._compute_aux_loss(router_probs)
        
        # Prepare routing info if requested
        routing_info = None
        if return_routing_info:
            routing_info = {
                'router_probs': router_probs.detach(),
                'top_k_indices': top_k_indices.detach(),
                'expert_usage': self._compute_expert_usage(top_k_indices),
            }
        
        return output, aux_loss, routing_info
    
    def _compute_aux_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load-balancing auxiliary loss (Switch Transformer style).
        
        Loss = alpha * num_experts * sum_i(f_i * P_i)
        
        Where:
        - f_i = fraction of tokens routed to expert i
        - P_i = average routing probability for expert i
        
        Args:
            router_probs: Shape (batch, seq, num_experts)
        
        Returns:
            aux_loss: Scalar tensor
        """
        # Flatten batch and seq dimensions
        probs_flat = router_probs.view(-1, self.num_experts)  # (batch*seq, num_experts)
        
        # f_i: fraction of tokens where expert i is the top choice
        top_expert = router_probs.argmax(dim=-1)  # (batch, seq)
        expert_counts = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            expert_counts[i] = (top_expert == i).float().sum()
        f = expert_counts / top_expert.numel()  # (num_experts,)
        
        # P_i: average probability for expert i
        P = probs_flat.mean(dim=0)  # (num_experts,)
        
        # Load-balancing loss
        aux_loss = self.aux_loss_coef * self.num_experts * (f * P).sum()
        
        return aux_loss
    
    def _compute_expert_usage(self, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Compute usage count for each expert."""
        usage = torch.zeros(self.num_experts, device=top_k_indices.device)
        for i in range(self.num_experts):
            usage[i] = (top_k_indices == i).sum().float()
        return usage / top_k_indices.numel()


# Convenience function for testing
def test_moe_layer():
    """Simple test to verify MoE layer functionality."""
    print("Testing MoE layer...")
    
    # Config
    batch_size, seq_len, n_embd = 2, 10, 128
    num_experts = 4
    top_k = 1
    
    # Create layer
    moe = MoELayer(n_embd=n_embd, num_experts=num_experts, top_k=top_k)
    
    # Test input
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Forward pass
    output, aux_loss, routing_info = moe(x, return_routing_info=True)
    
    # Assertions
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert aux_loss.ndim == 0, f"aux_loss should be scalar, got shape {aux_loss.shape}"
    assert aux_loss >= 0, f"aux_loss should be non-negative"
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Aux loss: {aux_loss.item():.6f}")
    print(f"✓ Expert usage: {routing_info['expert_usage']}")
    
    # Test sequence-level routing
    moe_seq = MoELayer(n_embd=n_embd, num_experts=num_experts, top_k=top_k, 
                       seq_level_routing=True)
    output_seq, aux_loss_seq, _ = moe_seq(x)
    assert output_seq.shape == x.shape
    print(f"✓ Sequence-level routing works")
    
    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    test_moe_layer()
