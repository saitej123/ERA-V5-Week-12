"""
Demo Transformer Model and Analytical FLOP/Activation Estimators.
Implements modular transformer architecture with support for mixed precision (BF16, FP16, MXFP8, FP32).
"""

import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention."""
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(out)


class FeedForwardMLP(nn.Module):
    """SwiGLU / Gated Feed-Forward Network."""
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single Transformer Decoder Layer."""
    def __init__(self, layer_id: int, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.mlp = FeedForwardMLP(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        h = x + self.attn(self.attn_norm(x), mask=mask)
        # Pre-LN MLP
        out = h + self.mlp(self.mlp_norm(h))
        return out


class DemoTransformerModel(nn.Module):
    """Complete Transformer Language Model for simulation."""
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                layer_id=i,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                intermediate_dim=config.intermediate_dim
            )
            for i in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        self.to(dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        
        # Causal attention mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
        
        for layer in self.layers:
            h = layer(h, mask=mask)
            
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def get_layer_parameter_groups(self) -> List[Tuple[str, List[str]]]:
        """Group parameter *names* layer by layer for ZeRO-2/3 bucketing.

        Must match `named_parameters()` prefixes:
        embed_tokens.*, layers.{idx}.*, norm.*, lm_head.*
        """
        groups: List[Tuple[str, List[str]]] = []
        groups.append(("embed_tokens", [n for n, _ in self.named_parameters() if n.startswith("embed_tokens")]))
        for idx in range(len(self.layers)):
            prefix = f"layers.{idx}."
            groups.append((f"layers.{idx}", [n for n, _ in self.named_parameters() if n.startswith(prefix)]))
        groups.append((
            "final_norm_and_head",
            [n for n, _ in self.named_parameters() if n.startswith("norm") or n.startswith("lm_head")],
        ))
        return groups


def calculate_model_memory_breakdown(
    model_cfg: ModelConfig,
    zero_stage: int,
    world_size: int,
    precision: str = "bf16",
    activation_checkpointing: bool = True,
    partition_activations: bool = False
) -> Dict[str, float]:
    """
    Analytically compute exact memory breakdown (in GB) for a given architecture, world size, and ZeRO stage.
    
    Precision bytes:
    - BF16 / FP16: 2 bytes per param/grad
    - FP32: 4 bytes per param/grad
    - MXFP8: 1 byte per param, 2 bytes master grad
    
    Optimizer state (Adam):
    - FP32 master weights: 4 bytes
    - FP32 1st momentum (m): 4 bytes
    - FP32 2nd momentum (v): 4 bytes
    - Total Adam OS = 12 bytes per parameter (in mixed precision)
    """
    total_params = model_cfg.total_parameters
    
    # Bytes per element
    if precision in ["bf16", "fp16"]:
        p_bytes = 2
        g_bytes = 2
        os_bytes = 12
    elif precision == "mxfp8":
        p_bytes = 1
        g_bytes = 1
        os_bytes = 12
    elif precision == "fp32":
        p_bytes = 4
        g_bytes = 4
        os_bytes = 8 # 2 x 4 bytes (m and v in fp32, master weights already fp32)
    else:
        p_bytes = 2
        g_bytes = 2
        os_bytes = 12

    # Model State Memory
    if zero_stage == 0:
        # Standard Data Parallelism (DP): Full replication on all ranks
        param_mem = total_params * p_bytes
        grad_mem = total_params * g_bytes
        opt_mem = total_params * os_bytes
    elif zero_stage == 1:
        # ZeRO-1: Partitioned Optimizer States (Pos)
        param_mem = total_params * p_bytes
        grad_mem = total_params * g_bytes
        opt_mem = (total_params * os_bytes) / world_size
    elif zero_stage == 2:
        # ZeRO-2: Partitioned Optimizer States + Gradients (Pos+g)
        param_mem = total_params * p_bytes
        grad_mem = (total_params * g_bytes) / world_size
        opt_mem = (total_params * os_bytes) / world_size
    elif zero_stage == 3:
        # ZeRO-3: Partitioned Parameters + Gradients + Optimizer States (Pos+g+p)
        param_mem = (total_params * p_bytes) / world_size
        grad_mem = (total_params * g_bytes) / world_size
        opt_mem = (total_params * os_bytes) / world_size
    else:
        raise ValueError(f"Invalid ZeRO stage {zero_stage}")

    # Activation Memory
    # Hidden size h, sequence length s, layers L, heads a, microbatch b
    h = model_cfg.hidden_dim
    s = model_cfg.seq_len
    b = model_cfg.micro_batch_size
    L = model_cfg.num_layers
    a = model_cfg.num_heads
    
    act_element_bytes = 2 if precision in ["bf16", "fp16", "mxfp8"] else 4
    
    if not activation_checkpointing:
        # Standard activations for all layers:
        # Attention activations: Q, K, V (3sbh) + Attention matrix (a * s^2 * b) + Softmax (a * s^2 * b) + Dropout (a * s^2 * b) + Attention out (sbh)
        # MLP activations: Gate (sb * inter) + Up (sb * inter) + Act (sb * inter) + Down (sbh)
        # LayerNorms: 2 * sbh
        # Approximately sbh * (34 + 5 * a * s / h) bytes per layer
        act_mem_per_layer = s * b * h * (34 + (5 * a * s / h)) * act_element_bytes
        act_mem = L * act_mem_per_layer
    else:
        # Activation Checkpointing (recomputation):
        # Store only the input activations to each transformer block
        act_mem = 2 * L * s * b * h * act_element_bytes

    if partition_activations:
        act_mem = act_mem / world_size

    # Temporary / Communication Buffers (e.g. 50MB gradient bucket + working buffers)
    buffer_mem = 50 * (1024 ** 2) if zero_stage > 0 else 100 * (1024 ** 2)
    if zero_stage == 3:
        # ZeRO-3 needs a working buffer to hold one layer's full parameters during forward/backward
        max_layer_params = max(
            model_cfg.vocab_size * h,
            4 * (h ** 2) + 3 * h * model_cfg.intermediate_dim
        )
        buffer_mem += max_layer_params * p_bytes

    # Persistent CUDA / NCCL workspace (allocator slack, collective scratch)
    workspace_mem = 2.0 * (1024 ** 3)
    total_mem = param_mem + grad_mem + opt_mem + act_mem + buffer_mem + workspace_mem

    return {
        "params_gb": param_mem / (1024 ** 3),
        "grads_gb": grad_mem / (1024 ** 3),
        "optimizer_gb": opt_mem / (1024 ** 3),
        "activation_gb": act_mem / (1024 ** 3),
        "buffer_gb": (buffer_mem + workspace_mem) / (1024 ** 3),
        "total_gb": total_mem / (1024 ** 3),
        "params_bytes": param_mem,
        "grads_bytes": grad_mem,
        "optimizer_bytes": opt_mem,
        "activation_bytes": act_mem,
        "total_bytes": total_mem
    }


def calculate_step_compute_flops(model_cfg: ModelConfig, activation_checkpointing: bool = True) -> float:
    """Calculate total theoretical FLOPs per training step for the model."""
    tokens_per_step = model_cfg.micro_batch_size * model_cfg.seq_len
    total_params = model_cfg.total_parameters
    
    # 2 FLOPs per parameter forward, 4 FLOPs per parameter backward
    fwd_flops = 2 * total_params * tokens_per_step
    bwd_flops = 4 * total_params * tokens_per_step
    
    if activation_checkpointing:
        # Recomputation adds 1 extra forward pass during backward (2 * total_params * tokens)
        bwd_flops += fwd_flops
        
    return fwd_flops + bwd_flops
