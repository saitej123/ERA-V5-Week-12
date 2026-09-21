"""
Virtual GPU and Cluster Management for 32 Virtual GPUs.
Simulates per-rank memory allocation, device buffers, computation, and communication tracking.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np

from .config import HardwareProfile, ClusterConfig, ZeROConfig


class VirtualGPU:
    """Represents an individual Virtual GPU rank."""
    
    def __init__(self, rank: int, cluster_cfg: ClusterConfig, hw_profile: HardwareProfile):
        self.rank = rank
        self.cluster_cfg = cluster_cfg
        self.hw_profile = hw_profile
        
        # Node and topology mapping
        self.gpus_per_node = cluster_cfg.gpus_per_node
        self.node_id = rank // self.gpus_per_node
        self.local_rank = rank % self.gpus_per_node
        
        # Memory tracking in bytes
        self.param_bytes: int = 0
        self.grad_bytes: int = 0
        self.optimizer_bytes: int = 0
        self.activation_bytes: int = 0
        self.buffer_bytes: int = 0
        self.peak_memory_bytes: int = 0
        self.cpu_offloaded_bytes: int = 0
        
        # Real tensor storage for rank
        self.local_param_shards: Dict[str, torch.Tensor] = {}
        self.local_grad_shards: Dict[str, torch.Tensor] = {}
        self.reconstructed_params: Dict[str, torch.Tensor] = {}
        self.master_weights: Dict[str, torch.Tensor] = {}
        self.exp_avg_m: Dict[str, torch.Tensor] = {}
        self.exp_avg_v: Dict[str, torch.Tensor] = {}
        
        # Communication & computation accounting for step logging
        self.step_comm_intra_bytes: int = 0
        self.step_comm_inter_bytes: int = 0
        self.step_comm_time_ms: float = 0.0
        self.step_compute_time_ms: float = 0.0
        self.step_flops: float = 0.0
        self.history: List[Dict[str, Any]] = []

    def allocate_tensor_memory(self, category: str, size_bytes: int):
        """Allocate and track memory for a specific category."""
        if category == "param":
            self.param_bytes += size_bytes
        elif category == "grad":
            self.grad_bytes += size_bytes
        elif category == "optimizer":
            self.optimizer_bytes += size_bytes
        elif category == "activation":
            self.activation_bytes += size_bytes
        elif category == "buffer":
            self.buffer_bytes += size_bytes
        elif category == "cpu_offload":
            self.cpu_offloaded_bytes += size_bytes
            
        current_gpu_total = self.total_gpu_memory_bytes
        if current_gpu_total > self.peak_memory_bytes:
            self.peak_memory_bytes = current_gpu_total

    def free_tensor_memory(self, category: str, size_bytes: int):
        """Free memory for a category."""
        if category == "param":
            self.param_bytes = max(0, self.param_bytes - size_bytes)
        elif category == "grad":
            self.grad_bytes = max(0, self.grad_bytes - size_bytes)
        elif category == "optimizer":
            self.optimizer_bytes = max(0, self.optimizer_bytes - size_bytes)
        elif category == "activation":
            self.activation_bytes = max(0, self.activation_bytes - size_bytes)
        elif category == "buffer":
            self.buffer_bytes = max(0, self.buffer_bytes - size_bytes)
        elif category == "cpu_offload":
            self.cpu_offloaded_bytes = max(0, self.cpu_offloaded_bytes - size_bytes)

    @property
    def total_gpu_memory_bytes(self) -> int:
        """Total memory consumed on GPU device."""
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes + self.activation_bytes + self.buffer_bytes

    @property
    def total_gpu_memory_gb(self) -> float:
        """Total GPU memory in Gigabytes (10^9 or 2^30)."""
        return self.total_gpu_memory_bytes / (1024 ** 3)

    @property
    def is_oom(self) -> bool:
        """Check if memory exceeded physical HBM capacity."""
        return self.total_gpu_memory_gb > self.hw_profile.hbm_capacity_gb

    def reset_step_metrics(self):
        """Reset per-step metrics."""
        self.step_comm_intra_bytes = 0
        self.step_comm_inter_bytes = 0
        self.step_comm_time_ms = 0.0
        self.step_compute_time_ms = 0.0
        self.step_flops = 0.0

    def log_step(self, step_idx: int, loss: float, total_step_time_ms: float, comm_fraction: float):
        """Log full step snapshot."""
        record = {
            "step": step_idx,
            "rank": self.rank,
            "loss": loss,
            "param_gb": self.param_bytes / (1024 ** 3),
            "grad_gb": self.grad_bytes / (1024 ** 3),
            "optimizer_gb": self.optimizer_bytes / (1024 ** 3),
            "activation_gb": self.activation_bytes / (1024 ** 3),
            "buffer_gb": self.buffer_bytes / (1024 ** 3),
            "total_gb": self.total_gpu_memory_gb,
            "peak_gb": self.peak_memory_bytes / (1024 ** 3),
            "comm_time_ms": self.step_comm_time_ms,
            "compute_time_ms": self.step_compute_time_ms,
            "step_time_ms": total_step_time_ms,
            "comm_fraction": comm_fraction,
            "comm_intra_mb": self.step_comm_intra_bytes / (1024 ** 2),
            "comm_inter_mb": self.step_comm_inter_bytes / (1024 ** 2),
        }
        self.history.append(record)


class VirtualCluster:
    """Simulates a cluster of N virtual GPUs with hierarchical topology."""
    
    def __init__(self, cluster_cfg: Optional[ClusterConfig] = None, hw_profile: Optional[HardwareProfile] = None):
        self.cluster_cfg = cluster_cfg or ClusterConfig(world_size=32, gpus_per_node=8)
        self.hw_profile = hw_profile or HardwareProfile()
        
        self.world_size = self.cluster_cfg.world_size
        self.gpus: List[VirtualGPU] = [
            VirtualGPU(rank=i, cluster_cfg=self.cluster_cfg, hw_profile=self.hw_profile)
            for i in range(self.world_size)
        ]

    def get_gpu(self, rank: int) -> VirtualGPU:
        return self.gpus[rank]

    def is_same_node(self, rank_a: int, rank_b: int) -> bool:
        """Check if two ranks share the same physical server/node (NVLink domain)."""
        return self.gpus[rank_a].node_id == self.gpus[rank_b].node_id

    def simulate_transfer_latency(self, size_bytes: int, collective_type: str) -> Tuple[float, int, int]:
        """
        Calculate communication latency and byte breakdown across intra-node and inter-node networks.
        Collective algorithm modeling (Ring/Tree):
        - Ring ReduceScatter: ((N - 1) / N) * Size per GPU
        - Ring AllGather: ((N - 1) / N) * Size per GPU
        - Ring AllReduce: 2 * ((N - 1) / N) * Size per GPU
        """
        N = self.world_size
        gpus_per_node = self.cluster_cfg.gpus_per_node
        num_nodes = self.cluster_cfg.num_nodes
        
        if collective_type in ["reduce_scatter", "all_gather"]:
            volume_per_gpu = int(size_bytes * (N - 1) / N)
        elif collective_type == "all_reduce":
            volume_per_gpu = int(2 * size_bytes * (N - 1) / N)
        elif collective_type == "broadcast":
            volume_per_gpu = size_bytes
        else:
            volume_per_gpu = size_bytes

        # For single node (num_nodes == 1), all traffic is intra-node over NVLink
        if num_nodes == 1:
            intra_bytes = volume_per_gpu
            inter_bytes = 0
            # Bandwidth in bytes/sec
            bw = self.hw_profile.intra_node_bandwidth_gbps * 1e9
            latency_sec = (self.hw_profile.intra_node_latency_us * 1e-6) * (2 * (N - 1))
            transfer_time_sec = latency_sec + (volume_per_gpu / bw)
        else:
            # Multi-node hierarchical or flat ring over combined network
            # Hierarchical collective: Intra-node ReduceScatter/AllGather over NVLink, Inter-node over InfiniBand
            # Section 5 of ZeRO paper highlights the ~9x bandwidth gap between NVLink and InfiniBand
            intra_ratio = (gpus_per_node - 1) / (N - 1)
            inter_ratio = (N - gpus_per_node) / (N - 1)
            
            intra_bytes = int(volume_per_gpu * intra_ratio)
            inter_bytes = int(volume_per_gpu * inter_ratio)
            
            # Transfer time is bounded by the slowest inter-node link + intra-node pipe
            bw_intra = self.hw_profile.intra_node_bandwidth_gbps * 1e9
            bw_inter = self.hw_profile.inter_node_bandwidth_gbps * 1e9
            
            time_intra = (self.hw_profile.intra_node_latency_us * 1e-6) * (2 * (gpus_per_node - 1)) + (intra_bytes / bw_intra)
            time_inter = (self.hw_profile.inter_node_latency_us * 1e-6) * (2 * (num_nodes - 1)) + (inter_bytes / bw_inter)
            
            transfer_time_sec = time_intra + time_inter

        time_ms = transfer_time_sec * 1000.0
        return time_ms, intra_bytes, inter_bytes
