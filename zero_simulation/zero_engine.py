"""
ZeRO Engines: Concrete implementations of ZeRO-0 (Standard DP), ZeRO-1 (Pos), ZeRO-2 (Pos+g), and ZeRO-3 (Pos+g+p).
Supports 32 Virtual GPUs with tensor execution, gradient bucketing, communication overlap, and per-step logging.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ClusterConfig, HardwareProfile, ModelConfig, ZeROConfig
from .virtual_gpu import VirtualCluster, VirtualGPU
from .collectives import Communicator
from .model import DemoTransformerModel, calculate_model_memory_breakdown, calculate_step_compute_flops


class BaseZeROEngine:
    """Base class for all ZeRO simulation engines."""
    
    def __init__(
        self,
        model_cfg: ModelConfig,
        cluster_cfg: ClusterConfig,
        hw_profile: HardwareProfile,
        zero_cfg: ZeROConfig
    ):
        self.model_cfg = model_cfg
        self.cluster_cfg = cluster_cfg
        self.hw_profile = hw_profile
        self.zero_cfg = zero_cfg
        self.world_size = cluster_cfg.world_size
        
        self.cluster = VirtualCluster(cluster_cfg, hw_profile)
        self.comm = Communicator(self.cluster)
        
        # Determine PyTorch dtype
        if zero_cfg.precision in ["bf16", "fp16"]:
            self.dtype = torch.float32 # Simulation runs on float32 CPU, tracks exact byte representations
        elif zero_cfg.precision == "mxfp8":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float32
            
        self.models: List[DemoTransformerModel] = [
            DemoTransformerModel(model_cfg, dtype=self.dtype)
            for _ in range(self.world_size)
        ]
        
        # Synchronize initial weights across all models
        self._synchronize_initial_weights()
        self._setup_optimizer_and_memory()

    def _synchronize_initial_weights(self):
        root_state = self.models[0].state_dict()
        for i in range(1, self.world_size):
            self.models[i].load_state_dict(root_state)

    def _setup_optimizer_and_memory(self):
        raise NotImplementedError

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        raise NotImplementedError


class ZeRO0_Engine(BaseZeROEngine):
    """ZeRO-0: Standard Data Parallelism (Replicated Parameters, Gradients, and Optimizer States)."""
    
    def _setup_optimizer_and_memory(self):
        mem_info = calculate_model_memory_breakdown(
            self.model_cfg, zero_stage=0, world_size=self.world_size,
            precision=self.zero_cfg.precision, activation_checkpointing=self.zero_cfg.activation_checkpointing
        )
        for gpu in self.cluster.gpus:
            gpu.allocate_tensor_memory("param", int(mem_info["params_bytes"]))
            gpu.allocate_tensor_memory("grad", int(mem_info["grads_bytes"]))
            gpu.allocate_tensor_memory("optimizer", int(mem_info["optimizer_bytes"]))
            gpu.allocate_tensor_memory("activation", int(mem_info["activation_bytes"]))
            gpu.allocate_tensor_memory("buffer", int(mem_info["buffer_gb"] * (1024 ** 3)))

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
            
        losses = []
        t0 = time.time()
        
        # 1. Forward Pass on each rank with microbatch
        for rank in range(self.world_size):
            model = self.models[rank]
            model.train()
            model.zero_grad()
            logits = model(batch_data)
            loss = nn.functional.cross_entropy(logits.view(-1, self.model_cfg.vocab_size), targets.view(-1))
            loss.backward()
            losses.append(loss.item())

        compute_flops = calculate_step_compute_flops(self.model_cfg, self.zero_cfg.activation_checkpointing)
        # Compute time in ms based on hardware TFLOPS
        tflops = self.hw_profile.bf16_tflops if self.zero_cfg.precision != "mxfp8" else self.hw_profile.fp8_tflops
        compute_time_ms = (compute_flops / (tflops * 1e12)) * 1000.0
        
        for gpu in self.cluster.gpus:
            gpu.step_compute_time_ms = compute_time_ms

        # 2. AllReduce Gradients across all ranks
        total_comm_ms = 0.0
        param_names = [name for name, _ in self.models[0].named_parameters()]
        for name in param_names:
            rank_grads = [self.models[r].get_parameter(name).grad for r in range(self.world_size)]
            reduced_grads, comm_ms = self.comm.all_reduce(rank_grads, op="mean")
            total_comm_ms += comm_ms
            for r in range(self.world_size):
                self.models[r].get_parameter(name).grad = reduced_grads[r]

        # 3. Optimizer update (SGD / Adam step)
        for r in range(self.world_size):
            with torch.no_grad():
                for p in self.models[r].parameters():
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-0.001)

        # Communication overlap: in ZeRO-0 without overlap, step time = compute + comm
        # If overlap is enabled, gradient all-reduce overlaps with backward pass
        if self.zero_cfg.overlap_comm:
            # Overlap fraction ~ 80% of backward pass (which is ~66% of total compute)
            overlapped_time = 0.6 * compute_time_ms
            exposed_comm_ms = max(0.0, total_comm_ms - overlapped_time)
            total_step_time_ms = compute_time_ms + exposed_comm_ms
        else:
            total_step_time_ms = compute_time_ms + total_comm_ms

        comm_fraction = total_comm_ms / (compute_time_ms + total_comm_ms) if (compute_time_ms + total_comm_ms) > 0 else 0
        avg_loss = sum(losses) / len(losses)

        for gpu in self.cluster.gpus:
            gpu.log_step(step_idx, avg_loss, total_step_time_ms, comm_fraction)

        return {
            "stage": 0,
            "step": step_idx,
            "loss": avg_loss,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": total_comm_ms,
            "total_step_time_ms": total_step_time_ms,
            "comm_fraction": comm_fraction,
            "peak_memory_gb": self.cluster.gpus[0].peak_memory_bytes / (1024 ** 3)
        }


class ZeRO1_Engine(BaseZeROEngine):
    """ZeRO-1: Partitioned Optimizer States (Pos). Parameters and Gradients remain replicated."""
    
    def _setup_optimizer_and_memory(self):
        mem_info = calculate_model_memory_breakdown(
            self.model_cfg, zero_stage=1, world_size=self.world_size,
            precision=self.zero_cfg.precision, activation_checkpointing=self.zero_cfg.activation_checkpointing
        )
        for gpu in self.cluster.gpus:
            gpu.allocate_tensor_memory("param", int(mem_info["params_bytes"]))
            gpu.allocate_tensor_memory("grad", int(mem_info["grads_bytes"]))
            gpu.allocate_tensor_memory("optimizer", int(mem_info["optimizer_bytes"]))
            gpu.allocate_tensor_memory("activation", int(mem_info["activation_bytes"]))
            gpu.allocate_tensor_memory("buffer", int(mem_info["buffer_gb"] * (1024 ** 3)))

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
            
        losses = []
        # 1. Forward Pass
        for rank in range(self.world_size):
            model = self.models[rank]
            model.train()
            model.zero_grad()
            logits = model(batch_data)
            loss = nn.functional.cross_entropy(logits.view(-1, self.model_cfg.vocab_size), targets.view(-1))
            loss.backward()
            losses.append(loss.item())

        compute_flops = calculate_step_compute_flops(self.model_cfg, self.zero_cfg.activation_checkpointing)
        tflops = self.hw_profile.bf16_tflops if self.zero_cfg.precision != "mxfp8" else self.hw_profile.fp8_tflops
        compute_time_ms = (compute_flops / (tflops * 1e12)) * 1000.0
        
        for gpu in self.cluster.gpus:
            gpu.step_compute_time_ms = compute_time_ms

        total_comm_ms = 0.0
        param_names = [name for name, _ in self.models[0].named_parameters()]
        
        # 2. ReduceScatter gradients to partition owners + Optimizer Step + AllGather updated parameters
        for name in param_names:
            rank_grads = [self.models[r].get_parameter(name).grad for r in range(self.world_size)]
            
            # ReduceScatter: Each rank gets its 1/N partition of gradients
            grad_shards, rs_ms = self.comm.reduce_scatter(rank_grads)
            total_comm_ms += rs_ms
            
            # Local optimizer step on the partition shard
            updated_shards = []
            for r in range(self.world_size):
                shard = grad_shards[r] / self.world_size
                # Get the matching parameter slice
                param_shards = list(torch.chunk(self.models[r].get_parameter(name).data, self.world_size, dim=0))
                param_shard = param_shards[r]
                param_shard.add_(shard, alpha=-0.001)
                updated_shards.append(param_shard)
                
            # AllGather updated parameter shards across all ranks
            gathered_params, ag_ms = self.comm.all_gather(updated_shards)
            total_comm_ms += ag_ms
            
            for r in range(self.world_size):
                self.models[r].get_parameter(name).data.copy_(gathered_params[r])

        if self.zero_cfg.overlap_comm:
            # ReduceScatter is overlapped with backward pass, AllGather is serialized after optimizer step
            rs_fraction = 0.5 * total_comm_ms
            ag_fraction = 0.5 * total_comm_ms
            overlapped_rs = max(0.0, rs_fraction - (0.6 * compute_time_ms))
            total_step_time_ms = compute_time_ms + overlapped_rs + ag_fraction
        else:
            total_step_time_ms = compute_time_ms + total_comm_ms

        comm_fraction = total_comm_ms / (compute_time_ms + total_comm_ms) if (compute_time_ms + total_comm_ms) > 0 else 0
        avg_loss = sum(losses) / len(losses)

        for gpu in self.cluster.gpus:
            gpu.log_step(step_idx, avg_loss, total_step_time_ms, comm_fraction)

        return {
            "stage": 1,
            "step": step_idx,
            "loss": avg_loss,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": total_comm_ms,
            "total_step_time_ms": total_step_time_ms,
            "comm_fraction": comm_fraction,
            "peak_memory_gb": self.cluster.gpus[0].peak_memory_bytes / (1024 ** 3)
        }


class ZeRO2_Engine(BaseZeROEngine):
    """ZeRO-2: Partitioned Optimizer States + Gradients (Pos+g). Gradient Bucketing and Overlap enabled."""
    
    def _setup_optimizer_and_memory(self):
        mem_info = calculate_model_memory_breakdown(
            self.model_cfg, zero_stage=2, world_size=self.world_size,
            precision=self.zero_cfg.precision, activation_checkpointing=self.zero_cfg.activation_checkpointing
        )
        for gpu in self.cluster.gpus:
            gpu.allocate_tensor_memory("param", int(mem_info["params_bytes"]))
            gpu.allocate_tensor_memory("grad", int(mem_info["grads_bytes"]))
            gpu.allocate_tensor_memory("optimizer", int(mem_info["optimizer_bytes"]))
            gpu.allocate_tensor_memory("activation", int(mem_info["activation_bytes"]))
            gpu.allocate_tensor_memory("buffer", int(mem_info["buffer_gb"] * (1024 ** 3)))

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
            
        losses = []
        # 1. Forward Pass
        for rank in range(self.world_size):
            model = self.models[rank]
            model.train()
            model.zero_grad()
            logits = model(batch_data)
            loss = nn.functional.cross_entropy(logits.view(-1, self.model_cfg.vocab_size), targets.view(-1))
            loss.backward()
            losses.append(loss.item())

        compute_flops = calculate_step_compute_flops(self.model_cfg, self.zero_cfg.activation_checkpointing)
        tflops = self.hw_profile.bf16_tflops if self.zero_cfg.precision != "mxfp8" else self.hw_profile.fp8_tflops
        compute_time_ms = (compute_flops / (tflops * 1e12)) * 1000.0
        
        for gpu in self.cluster.gpus:
            gpu.step_compute_time_ms = compute_time_ms

        total_comm_ms = 0.0
        rs_comm_ms = 0.0
        ag_comm_ms = 0.0
        
        # 2. Gradient Bucketing & ReduceScatter
        param_groups = self.models[0].get_layer_parameter_groups()
        for group_name, _ in param_groups:
            # For each layer, ReduceScatter gradients immediately upon backward computation
            for p_name, _ in self.models[0].named_parameters():
                if group_name in p_name:
                    rank_grads = [self.models[r].get_parameter(p_name).grad for r in range(self.world_size)]
                    grad_shards, rs_ms = self.comm.reduce_scatter(rank_grads)
                    rs_comm_ms += rs_ms
                    
                    # Local update on partition
                    updated_shards = []
                    for r in range(self.world_size):
                        shard = grad_shards[r] / self.world_size
                        param_shards = list(torch.chunk(self.models[r].get_parameter(p_name).data, self.world_size, dim=0))
                        param_shard = param_shards[r]
                        param_shard.add_(shard, alpha=-0.001)
                        updated_shards.append(param_shard)
                        
                    # AllGather updated parameter shards
                    gathered_params, ag_ms = self.comm.all_gather(updated_shards)
                    ag_comm_ms += ag_ms
                    
                    for r in range(self.world_size):
                        self.models[r].get_parameter(p_name).data.copy_(gathered_params[r])

        total_comm_ms = rs_comm_ms + ag_comm_ms

        if self.zero_cfg.overlap_comm:
            # High overlap efficiency for ZeRO-2: Bucketed ReduceScatter perfectly overlaps with backward computation
            overlapped_rs = max(0.0, rs_comm_ms - (0.8 * compute_time_ms))
            total_step_time_ms = compute_time_ms + overlapped_rs + ag_comm_ms
        else:
            total_step_time_ms = compute_time_ms + total_comm_ms

        comm_fraction = total_comm_ms / (compute_time_ms + total_comm_ms) if (compute_time_ms + total_comm_ms) > 0 else 0
        avg_loss = sum(losses) / len(losses)

        for gpu in self.cluster.gpus:
            gpu.log_step(step_idx, avg_loss, total_step_time_ms, comm_fraction)

        return {
            "stage": 2,
            "step": step_idx,
            "loss": avg_loss,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": total_comm_ms,
            "total_step_time_ms": total_step_time_ms,
            "comm_fraction": comm_fraction,
            "peak_memory_gb": self.cluster.gpus[0].peak_memory_bytes / (1024 ** 3)
        }


class ZeRO3_Engine(BaseZeROEngine):
    """
    ZeRO-3: Partitioned Parameters + Gradients + Optimizer States (Pos+g+p).
    Layer-wise AllGather in Forward, Layer-wise AllGather in Backward, Layer-wise ReduceScatter in Backward.
    """
    
    def _setup_optimizer_and_memory(self):
        mem_info = calculate_model_memory_breakdown(
            self.model_cfg, zero_stage=3, world_size=self.world_size,
            precision=self.zero_cfg.precision, activation_checkpointing=self.zero_cfg.activation_checkpointing
        )
        for gpu in self.cluster.gpus:
            gpu.allocate_tensor_memory("param", int(mem_info["params_bytes"]))
            gpu.allocate_tensor_memory("grad", int(mem_info["grads_bytes"]))
            gpu.allocate_tensor_memory("optimizer", int(mem_info["optimizer_bytes"]))
            gpu.allocate_tensor_memory("activation", int(mem_info["activation_bytes"]))
            gpu.allocate_tensor_memory("buffer", int(mem_info["buffer_gb"] * (1024 ** 3)))

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
            
        losses = []
        total_comm_ms = 0.0
        fwd_comm_ms = 0.0
        bwd_comm_ms = 0.0
        
        # 1. Forward Pass with Layer-wise AllGather (Volume = 1 * Psi)
        param_groups = self.models[0].get_layer_parameter_groups()
        for group_name, _ in param_groups:
            for p_name, _ in self.models[0].named_parameters():
                if group_name in p_name:
                    # Partition shards across ranks
                    full_p = self.models[0].get_parameter(p_name).data
                    shards = list(torch.chunk(full_p, self.world_size, dim=0))
                    # AllGather layer parameters before forward execution
                    gathered_p, ag_ms = self.comm.all_gather(shards)
                    fwd_comm_ms += ag_ms

        for rank in range(self.world_size):
            model = self.models[rank]
            model.train()
            model.zero_grad()
            logits = model(batch_data)
            loss = nn.functional.cross_entropy(logits.view(-1, self.model_cfg.vocab_size), targets.view(-1))
            loss.backward()
            losses.append(loss.item())

        compute_flops = calculate_step_compute_flops(self.model_cfg, self.zero_cfg.activation_checkpointing)
        tflops = self.hw_profile.bf16_tflops if self.zero_cfg.precision != "mxfp8" else self.hw_profile.fp8_tflops
        compute_time_ms = (compute_flops / (tflops * 1e12)) * 1000.0
        
        for gpu in self.cluster.gpus:
            gpu.step_compute_time_ms = compute_time_ms

        # 2. Backward Pass with Layer-wise AllGather (Volume = 1 * Psi) + ReduceScatter (Volume = 1 * Psi)
        for group_name, _ in reversed(param_groups):
            for p_name, _ in self.models[0].named_parameters():
                if group_name in p_name:
                    # Backward AllGather weights for gradient computation
                    full_p = self.models[0].get_parameter(p_name).data
                    shards = list(torch.chunk(full_p, self.world_size, dim=0))
                    _, ag_ms = self.comm.all_gather(shards)
                    bwd_comm_ms += ag_ms
                    
                    # ReduceScatter layer gradients to owners
                    rank_grads = [self.models[r].get_parameter(p_name).grad for r in range(self.world_size)]
                    grad_shards, rs_ms = self.comm.reduce_scatter(rank_grads)
                    bwd_comm_ms += rs_ms
                    
                    # Local partition update
                    for r in range(self.world_size):
                        shard = grad_shards[r] / self.world_size
                        param_shards = list(torch.chunk(self.models[r].get_parameter(p_name).data, self.world_size, dim=0))
                        param_shard = param_shards[r]
                        param_shard.add_(shard, alpha=-0.001)

        total_comm_ms = fwd_comm_ms + bwd_comm_ms

        if self.zero_cfg.overlap_comm:
            # Prefetching and pipelining overlaps layer parameters during forward and backward
            # Overlap covers up to 70% of forward compute and 70% of backward compute
            overlapped_comm = max(0.0, total_comm_ms - (0.7 * compute_time_ms))
            total_step_time_ms = compute_time_ms + overlapped_comm
        else:
            total_step_time_ms = compute_time_ms + total_comm_ms

        comm_fraction = total_comm_ms / (compute_time_ms + total_comm_ms) if (compute_time_ms + total_comm_ms) > 0 else 0
        avg_loss = sum(losses) / len(losses)

        for gpu in self.cluster.gpus:
            gpu.log_step(step_idx, avg_loss, total_step_time_ms, comm_fraction)

        return {
            "stage": 3,
            "step": step_idx,
            "loss": avg_loss,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": total_comm_ms,
            "total_step_time_ms": total_step_time_ms,
            "comm_fraction": comm_fraction,
            "peak_memory_gb": self.cluster.gpus[0].peak_memory_bytes / (1024 ** 3)
        }
