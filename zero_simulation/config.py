"""
Configuration definitions for ZeRO Parallelism Simulation.
Includes hardware profiles, cluster topologies, model architectures, and ZeRO configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class HardwareProfile:
    """Hardware specifications for compute and interconnect bandwidth."""
    name: str = "NVIDIA H100 SXM5"
    hbm_capacity_gb: float = 80.0             # GPU Memory per device (GB)
    bf16_tflops: float = 989.0                 # Dense BF16 Tensor Core Peak TFLOPS
    fp8_tflops: float = 1978.0                 # Dense FP8 / MXFP8 Peak TFLOPS
    intra_node_bandwidth_gbps: float = 900.0   # NVLink 4 bidirectional bandwidth (GB/s)
    inter_node_bandwidth_gbps: float = 50.0    # InfiniBand NDR (400 Gbps = 50 GB/s)
    pcie_bandwidth_gbps: float = 64.0          # PCIe Gen5 x16 host-to-device (GB/s)
    intra_node_latency_us: float = 1.0         # NVLink latency (microseconds)
    inter_node_latency_us: float = 5.0         # Network latency (microseconds)
    cpu_dram_bandwidth_gbps: float = 300.0     # Host DDR5 RAM bandwidth (GB/s)


# Chronological order matters: Section 5 plots A100 → H100 → B200.
HARDWARE_GENERATION_ORDER = ["A100_SXM4", "H100_SXM5", "B200_NVL72"]

HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "A100_SXM4": HardwareProfile(
        name="NVIDIA Ampere A100 SXM4",
        hbm_capacity_gb=80.0,
        bf16_tflops=312.0,
        fp8_tflops=312.0,                  # No native FP8 tensor cores
        intra_node_bandwidth_gbps=600.0,   # NVLink 3 (600 GB/s)
        inter_node_bandwidth_gbps=25.0,    # 200 Gbps HDR IB (25 GB/s)
        pcie_bandwidth_gbps=32.0,
        intra_node_latency_us=1.5,
        inter_node_latency_us=8.0,
        cpu_dram_bandwidth_gbps=200.0
    ),
    "H100_SXM5": HardwareProfile(
        name="NVIDIA Hopper H100 SXM5",
        hbm_capacity_gb=80.0,
        bf16_tflops=989.0,
        fp8_tflops=1978.0,
        intra_node_bandwidth_gbps=900.0,   # NVLink 4 (900 GB/s)
        inter_node_bandwidth_gbps=50.0,    # 400 Gbps IB (50 GB/s)
        pcie_bandwidth_gbps=64.0,
        intra_node_latency_us=1.0,
        inter_node_latency_us=5.0,
        cpu_dram_bandwidth_gbps=300.0
    ),
    "B200_NVL72": HardwareProfile(
        name="NVIDIA Blackwell B200 (NVLink Rack)",
        hbm_capacity_gb=192.0,
        bf16_tflops=2250.0,
        fp8_tflops=4500.0,
        intra_node_bandwidth_gbps=1800.0,  # NVLink 5 (1.8 TB/s)
        inter_node_bandwidth_gbps=100.0,   # Quantum-X800 InfiniBand (800 Gbps = 100 GB/s)
        pcie_bandwidth_gbps=128.0,
        intra_node_latency_us=0.5,
        inter_node_latency_us=3.0,
        cpu_dram_bandwidth_gbps=400.0
    ),
    "VIRTUAL_CPU": HardwareProfile(
        name="Virtual 32-GPU Host Simulation",
        hbm_capacity_gb=80.0,
        bf16_tflops=100.0,
        fp8_tflops=200.0,
        intra_node_bandwidth_gbps=900.0,
        inter_node_bandwidth_gbps=50.0,
        pcie_bandwidth_gbps=64.0,
        intra_node_latency_us=1.0,
        inter_node_latency_us=5.0,
        cpu_dram_bandwidth_gbps=300.0
    )
}


@dataclass
class ClusterConfig:
    """Cluster topology configuration."""
    world_size: int = 32              # Total number of GPUs
    gpus_per_node: int = 8            # GPUs per physical server / NVLink domain
    num_nodes: int = 4                # Number of nodes (world_size // gpus_per_node)
    
    def __post_init__(self):
        if self.world_size % self.gpus_per_node != 0:
            self.num_nodes = (self.world_size + self.gpus_per_node - 1) // self.gpus_per_node
        else:
            self.num_nodes = self.world_size // self.gpus_per_node


@dataclass
class ModelConfig:
    """Transformer / Deep Network Architecture parameters."""
    name: str = "Demo-Transformer-32VGPU"
    vocab_size: int = 4096
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    intermediate_dim: int = 4096
    seq_len: int = 512
    micro_batch_size: int = 2
    
    @property
    def total_parameters(self) -> int:
        """Calculate total parameter count for the transformer model."""
        # Embeddings: vocab_size * hidden_dim
        emb_params = self.vocab_size * self.hidden_dim
        
        # Per layer:
        # Self Attention: Q, K, V (3 * h * h) + Output Projection (h * h) + 2 LayerNorms (2 * h)
        attn_params = 4 * (self.hidden_dim ** 2) + 2 * self.hidden_dim
        
        # MLP: Gate & Up Projections (2 * h * intermediate_dim) + Down (intermediate_dim * h) + LayerNorm (h)
        mlp_params = 3 * self.hidden_dim * self.intermediate_dim + self.hidden_dim
        
        layer_params = attn_params + mlp_params
        
        # Final Norm + LM Head (if tied, 0 extra, if untied vocab * hidden)
        head_params = self.hidden_dim + self.vocab_size * self.hidden_dim
        
        return emb_params + (self.num_layers * layer_params) + head_params


# Standard model scale presets
MODEL_PRESETS = {
    "demo_small": ModelConfig(
        name="Demo-Transformer-Small",
        vocab_size=2048,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        intermediate_dim=1024,
        seq_len=128,
        micro_batch_size=2
    ),
    "demo_medium": ModelConfig(
        name="Demo-Transformer-Medium",
        vocab_size=4096,
        hidden_dim=1024,
        num_layers=12,
        num_heads=16,
        intermediate_dim=4096,
        seq_len=512,
        micro_batch_size=2
    ),
    "llm_13b": ModelConfig(
        name="LLM-13B",
        vocab_size=32000,
        hidden_dim=5120,
        num_layers=40,
        num_heads=40,
        intermediate_dim=13824,
        seq_len=2048,
        micro_batch_size=2
    ),
    "llm_20b": ModelConfig(
        name="Target-LLM-20B",
        vocab_size=32000,
        hidden_dim=6144,
        num_layers=44,
        num_heads=48,
        intermediate_dim=16384,
        seq_len=2048,
        micro_batch_size=2
    ),
    "llm_70b": ModelConfig(
        name="LLM-70B",
        vocab_size=32000,
        hidden_dim=8192,
        num_layers=80,
        num_heads=64,
        intermediate_dim=28672,
        seq_len=4096,
        micro_batch_size=1
    )
}


@dataclass
class ZeROConfig:
    """ZeRO Stage and Optimization Options."""
    stage: int = 2                       # 0: Standard DP, 1: Pos, 2: Pos+g, 3: Pos+g+p
    precision: str = "bf16"              # "bf16", "fp16", "mxfp8", "fp32"
    bucket_size_mb: float = 25.0         # Gradient bucket size for overlap (MB)
    overlap_comm: bool = True            # Overlap communication with backward computation
    cpu_offload: bool = False            # ZeRO-Offload optimizer states to host RAM
    activation_checkpointing: bool = True# Recompute activations during backward pass
    partition_activations: bool = False  # ZeRO-R activation partitioning across ranks
