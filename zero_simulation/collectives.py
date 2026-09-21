"""
Collective communication operations for the Virtual GPU cluster.
Performs real PyTorch tensor aggregations/sharding while recording communication metrics.
"""

from typing import List, Tuple, Dict, Optional
import torch
import numpy as np

from .virtual_gpu import VirtualCluster, VirtualGPU


class Communicator:
    """Manages collective operations across virtual GPU ranks."""
    
    def __init__(self, cluster: VirtualCluster):
        self.cluster = cluster
        self.world_size = cluster.world_size

    def all_reduce(self, tensors_per_rank: List[torch.Tensor], op: str = "sum") -> Tuple[List[torch.Tensor], float]:
        """
        AllReduce across all ranks: Sums tensors from all ranks and returns identical copies on each rank.
        Communication volume per rank: 2 * ((N - 1) / N) * tensor_size_bytes.
        """
        assert len(tensors_per_rank) == self.world_size
        
        # Real tensor computation
        stacked = torch.stack(tensors_per_rank, dim=0)
        if op == "sum":
            reduced = torch.sum(stacked, dim=0)
        elif op == "mean":
            reduced = torch.mean(stacked, dim=0)
        else:
            raise ValueError(f"Unsupported op {op}")
            
        result = [reduced.clone() for _ in range(self.world_size)]
        
        # Calculate communication metrics
        tensor_bytes = tensors_per_rank[0].numel() * tensors_per_rank[0].element_size()
        time_ms, intra_b, inter_b = self.cluster.simulate_transfer_latency(tensor_bytes, "all_reduce")
        
        # Log to ranks
        for gpu in self.cluster.gpus:
            gpu.step_comm_intra_bytes += intra_b
            gpu.step_comm_inter_bytes += inter_b
            gpu.step_comm_time_ms += time_ms
            
        return result, time_ms

    def reduce_scatter(self, tensors_per_rank: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        """
        ReduceScatter across all ranks: Sums the input tensors and scatters equal disjoint shards to each rank.
        Input: Each rank provides a full tensor of size S.
        Output: Rank i receives a reduced shard of size S / N.
        Communication volume per rank: ((N - 1) / N) * tensor_size_bytes.
        """
        assert len(tensors_per_rank) == self.world_size
        full_tensor = torch.stack(tensors_per_rank, dim=0).sum(dim=0)
        
        # Split into N shards
        shards = list(torch.chunk(full_tensor, self.world_size, dim=0))
        # Ensure padding if chunking uneven
        output_shards = []
        for i in range(self.world_size):
            output_shards.append(shards[i].clone())
            
        tensor_bytes = tensors_per_rank[0].numel() * tensors_per_rank[0].element_size()
        time_ms, intra_b, inter_b = self.cluster.simulate_transfer_latency(tensor_bytes, "reduce_scatter")
        
        for gpu in self.cluster.gpus:
            gpu.step_comm_intra_bytes += intra_b
            gpu.step_comm_inter_bytes += inter_b
            gpu.step_comm_time_ms += time_ms
            
        return output_shards, time_ms

    def all_gather(self, shard_per_rank: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        """
        AllGather across all ranks: Gathers shards of size S / N from all ranks to produce full tensor S on each rank.
        Communication volume per rank: ((N - 1) / N) * total_tensor_size_bytes.
        """
        assert len(shard_per_rank) == self.world_size
        full_tensor = torch.cat(shard_per_rank, dim=0)
        
        result = [full_tensor.clone() for _ in range(self.world_size)]
        
        total_bytes = full_tensor.numel() * full_tensor.element_size()
        time_ms, intra_b, inter_b = self.cluster.simulate_transfer_latency(total_bytes, "all_gather")
        
        for gpu in self.cluster.gpus:
            gpu.step_comm_intra_bytes += intra_b
            gpu.step_comm_inter_bytes += inter_b
            gpu.step_comm_time_ms += time_ms
            
        return result, time_ms

    def broadcast(self, root_tensor: torch.Tensor, root_rank: int = 0) -> Tuple[List[torch.Tensor], float]:
        """Broadcast tensor from root_rank to all ranks."""
        result = [root_tensor.clone() for _ in range(self.world_size)]
        
        tensor_bytes = root_tensor.numel() * root_tensor.element_size()
        time_ms, intra_b, inter_b = self.cluster.simulate_transfer_latency(tensor_bytes, "broadcast")
        
        for gpu in self.cluster.gpus:
            gpu.step_comm_intra_bytes += intra_b
            gpu.step_comm_inter_bytes += inter_b
            gpu.step_comm_time_ms += time_ms
            
        return result, time_ms
