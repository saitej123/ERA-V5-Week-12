"""
ZeRO Engines: ZeRO-0 (DP), ZeRO-1 (Pos), ZeRO-2 (Pos+g), ZeRO-3 (Pos+g+p).

One PyTorch model runs the real forward/backward. 32 virtual GPU ranks track
memory, ring collectives, gradient buckets, and overlapped step time.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn

from .config import ClusterConfig, HardwareProfile, ModelConfig, ZeROConfig
from .virtual_gpu import VirtualCluster
from .collectives import Communicator
from .model import DemoTransformerModel, calculate_model_memory_breakdown, calculate_step_compute_flops


def _param_in_group(p_name: str, group_name: str) -> bool:
    """Match named_parameters() keys to bucket groups (layers.0.* not layer_0)."""
    if group_name == "embed_tokens":
        return p_name.startswith("embed_tokens")
    if group_name.startswith("layers."):
        return p_name.startswith(group_name + ".") or p_name.startswith(group_name)
    if group_name == "final_norm_and_head":
        return p_name.startswith("norm") or p_name.startswith("lm_head")
    return False


class BaseZeROEngine:
    """Shared 32-rank virtual cluster + a single demo Transformer."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        cluster_cfg: ClusterConfig,
        hw_profile: HardwareProfile,
        zero_cfg: ZeROConfig,
    ):
        self.model_cfg = model_cfg
        self.cluster_cfg = cluster_cfg
        self.hw_profile = hw_profile
        self.zero_cfg = zero_cfg
        self.world_size = cluster_cfg.world_size

        self.cluster = VirtualCluster(cluster_cfg, hw_profile)
        self.comm = Communicator(self.cluster)
        self.dtype = torch.float32
        self.model = DemoTransformerModel(model_cfg, dtype=self.dtype)
        self.models = [self.model]  # compatibility alias
        self._setup_optimizer_and_memory()

    def _setup_optimizer_and_memory(self):
        raise NotImplementedError

    def _allocate_from_breakdown(self, mem_info: Dict[str, float]):
        for gpu in self.cluster.gpus:
            gpu.allocate_tensor_memory("param", int(mem_info["params_bytes"]))
            gpu.allocate_tensor_memory("grad", int(mem_info["grads_bytes"]))
            gpu.allocate_tensor_memory("optimizer", int(mem_info["optimizer_bytes"]))
            gpu.allocate_tensor_memory("activation", int(mem_info["activation_bytes"]))
            gpu.allocate_tensor_memory("buffer", int(mem_info["buffer_gb"] * (1024 ** 3)))

    def _compute_time_ms(self) -> float:
        compute_flops = calculate_step_compute_flops(
            self.model_cfg, self.zero_cfg.activation_checkpointing
        )
        tflops = (
            self.hw_profile.fp8_tflops
            if self.zero_cfg.precision == "mxfp8"
            else self.hw_profile.bf16_tflops
        )
        return (compute_flops / (tflops * 1e12)) * 1000.0

    def _forward_backward(self, batch_data: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.train()
        self.model.zero_grad()
        logits = self.model(batch_data)
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.model_cfg.vocab_size), targets.view(-1)
        )
        loss.backward()
        return loss.item()

    def _overlap_step_time(self, compute_time_ms: float, comm_time_ms: float, stage: int) -> float:
        if not self.zero_cfg.overlap_comm:
            return compute_time_ms + comm_time_ms
        # Bucketing hides most ReduceScatter behind backward compute.
        overlap_window = 0.66 * compute_time_ms
        efficiency = {0: 0.65, 1: 0.70, 2: 0.85, 3: 0.75}.get(stage, 0.70)
        exposed = max(0.0, comm_time_ms - efficiency * overlap_window)
        return compute_time_ms + exposed

    def _finalize_step(
        self, step_idx: int, stage: int, loss: float, compute_time_ms: float, comm_time_ms: float
    ) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.step_compute_time_ms = compute_time_ms
        total_step_time_ms = self._overlap_step_time(compute_time_ms, comm_time_ms, stage)
        denom = compute_time_ms + comm_time_ms
        comm_fraction = comm_time_ms / denom if denom > 0 else 0.0
        for gpu in self.cluster.gpus:
            gpu.log_step(step_idx, loss, total_step_time_ms, comm_fraction)
        return {
            "stage": stage,
            "step": step_idx,
            "loss": loss,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": comm_time_ms,
            "total_step_time_ms": total_step_time_ms,
            "comm_fraction": comm_fraction,
            "peak_memory_gb": self.cluster.gpus[0].peak_memory_bytes / (1024 ** 3),
        }

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        raise NotImplementedError


class ZeRO0_Engine(BaseZeROEngine):
    """Standard data parallelism: replicated P, g, and optimizer states. AllReduce grads."""

    def _setup_optimizer_and_memory(self):
        self._allocate_from_breakdown(
            calculate_model_memory_breakdown(
                self.model_cfg, 0, self.world_size, self.zero_cfg.precision,
                self.zero_cfg.activation_checkpointing,
            )
        )

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
        loss = self._forward_backward(batch_data, targets)
        compute_time_ms = self._compute_time_ms()
        total_comm_ms = 0.0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                rank_grads = [p.grad.clone() for _ in range(self.world_size)]
                reduced, comm_ms = self.comm.all_reduce(rank_grads, op="mean")
                total_comm_ms += comm_ms
                p.grad.copy_(reduced[0])
                p.data.add_(p.grad, alpha=-0.001)
        return self._finalize_step(step_idx, 0, loss, compute_time_ms, total_comm_ms)


class ZeRO1_Engine(BaseZeROEngine):
    """ZeRO-1: partition optimizer states. ReduceScatter grads, Adam on shard, AllGather params."""

    def _setup_optimizer_and_memory(self):
        self._allocate_from_breakdown(
            calculate_model_memory_breakdown(
                self.model_cfg, 1, self.world_size, self.zero_cfg.precision,
                self.zero_cfg.activation_checkpointing,
            )
        )

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
        loss = self._forward_backward(batch_data, targets)
        compute_time_ms = self._compute_time_ms()
        total_comm_ms = 0.0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                rank_grads = [p.grad.clone() for _ in range(self.world_size)]
                grad_shards, rs_ms = self.comm.reduce_scatter(rank_grads)
                total_comm_ms += rs_ms
                updated = []
                param_shards = list(torch.chunk(p.data, self.world_size, dim=0))
                for r in range(self.world_size):
                    shard = param_shards[r].clone()
                    shard.add_(grad_shards[r] / self.world_size, alpha=-0.001)
                    updated.append(shard)
                gathered, ag_ms = self.comm.all_gather(updated)
                total_comm_ms += ag_ms
                p.data.copy_(gathered[0])
        return self._finalize_step(step_idx, 1, loss, compute_time_ms, total_comm_ms)


class ZeRO2_Engine(BaseZeROEngine):
    """ZeRO-2: partition optimizer + gradients. Bucketed ReduceScatter overlapped with backward."""

    def _setup_optimizer_and_memory(self):
        self._allocate_from_breakdown(
            calculate_model_memory_breakdown(
                self.model_cfg, 2, self.world_size, self.zero_cfg.precision,
                self.zero_cfg.activation_checkpointing,
            )
        )

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
        loss = self._forward_backward(batch_data, targets)
        compute_time_ms = self._compute_time_ms()
        total_comm_ms = 0.0
        param_groups = self.model.get_layer_parameter_groups()
        with torch.no_grad():
            for group_name, names in param_groups:
                for p_name, p in self.model.named_parameters():
                    if not _param_in_group(p_name, group_name) or p.grad is None:
                        continue
                    rank_grads = [p.grad.clone() for _ in range(self.world_size)]
                    grad_shards, rs_ms = self.comm.reduce_scatter(rank_grads)
                    total_comm_ms += rs_ms
                    updated = []
                    param_shards = list(torch.chunk(p.data, self.world_size, dim=0))
                    for r in range(self.world_size):
                        shard = param_shards[r].clone()
                        shard.add_(grad_shards[r] / self.world_size, alpha=-0.001)
                        updated.append(shard)
                    gathered, ag_ms = self.comm.all_gather(updated)
                    total_comm_ms += ag_ms
                    p.data.copy_(gathered[0])
        return self._finalize_step(step_idx, 2, loss, compute_time_ms, total_comm_ms)


class ZeRO3_Engine(BaseZeROEngine):
    """ZeRO-3: partition parameters, grads, optimizer. Layer AllGather + ReduceScatter (3Ψ)."""

    def _setup_optimizer_and_memory(self):
        self._allocate_from_breakdown(
            calculate_model_memory_breakdown(
                self.model_cfg, 3, self.world_size, self.zero_cfg.precision,
                self.zero_cfg.activation_checkpointing,
            )
        )

    def train_step(self, step_idx: int, batch_data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        for gpu in self.cluster.gpus:
            gpu.reset_step_metrics()
        param_groups = self.model.get_layer_parameter_groups()
        fwd_comm_ms = 0.0
        # Forward: AllGather each layer's parameters (volume Ψ)
        with torch.no_grad():
            for group_name, names in param_groups:
                for p_name, p in self.model.named_parameters():
                    if not _param_in_group(p_name, group_name):
                        continue
                    shards = list(torch.chunk(p.data, self.world_size, dim=0))
                    _, ag_ms = self.comm.all_gather(shards)
                    fwd_comm_ms += ag_ms

        loss = self._forward_backward(batch_data, targets)
        compute_time_ms = self._compute_time_ms()

        bwd_comm_ms = 0.0
        with torch.no_grad():
            for group_name, names in reversed(param_groups):
                for p_name, p in self.model.named_parameters():
                    if not _param_in_group(p_name, group_name) or p.grad is None:
                        continue
                    shards = list(torch.chunk(p.data, self.world_size, dim=0))
                    _, ag_ms = self.comm.all_gather(shards)
                    bwd_comm_ms += ag_ms
                    rank_grads = [p.grad.clone() for _ in range(self.world_size)]
                    grad_shards, rs_ms = self.comm.reduce_scatter(rank_grads)
                    bwd_comm_ms += rs_ms
                    param_shards = list(torch.chunk(p.data, self.world_size, dim=0))
                    param_shards[0].add_(grad_shards[0] / self.world_size, alpha=-0.001)
        return self._finalize_step(step_idx, 3, loss, compute_time_ms, fwd_comm_ms + bwd_comm_ms)
