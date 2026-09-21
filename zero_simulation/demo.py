"""
Assignment demo: 32 CPU-thread virtual GPUs + a small Transformer + ZeRO-1/2/3.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
import torch

from .config import ClusterConfig, HARDWARE_PROFILES, MODEL_PRESETS, ZeROConfig
from .virtual_gpu import VirtualCluster
from .zero_engine import ZeRO1_Engine, ZeRO2_Engine, ZeRO3_Engine
from .model import calculate_model_memory_breakdown, calculate_step_compute_flops
from .comm_profiler import CommunicationProfiler


def make_32_virtual_gpus() -> VirtualCluster:
    """32 virtual devices mapped onto CPU threads (8 per simulated node)."""
    return VirtualCluster(
        ClusterConfig(world_size=32, gpus_per_node=8),
        HARDWARE_PROFILES["VIRTUAL_CPU"],
    )


def ping_ranks_on_cpu_threads(cluster: VirtualCluster) -> List[Dict[str, Any]]:
    """Run a real GEMM on each rank in a 32-thread pool (the 'virtual GPU' compute)."""

    def _rank_kernel(gpu):
        torch.manual_seed(gpu.rank)
        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        out = a @ b
        return {
            "rank": gpu.rank,
            "node_id": gpu.node_id,
            "local_rank": gpu.local_rank,
            "thread_checksum": float(out.sum()),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=cluster.world_size) as pool:
        futs = [pool.submit(_rank_kernel, gpu) for gpu in cluster.gpus]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return sorted(rows, key=lambda r: r["rank"])


def cyclic_batch(model_cfg, batch_size=None):
    b = batch_size or model_cfg.micro_batch_size
    period = min(64, model_cfg.vocab_size)
    x = (torch.arange(model_cfg.seq_len) % period).unsqueeze(0).repeat(b, 1)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def run_zero_demo(n_steps: int = 6) -> Dict[str, Any]:
    """Train the demo Transformer under ZeRO-1, ZeRO-2, and ZeRO-3 on 32 virtual GPUs."""
    model_cfg = MODEL_PRESETS["demo_small"]
    cluster_cfg = ClusterConfig(world_size=32, gpus_per_node=8)
    hw = HARDWARE_PROFILES["VIRTUAL_CPU"]
    engines = {
        "ZeRO-1": ZeRO1_Engine(model_cfg, cluster_cfg, hw, ZeROConfig(stage=1, overlap_comm=True)),
        "ZeRO-2": ZeRO2_Engine(model_cfg, cluster_cfg, hw, ZeROConfig(stage=2, overlap_comm=True)),
        "ZeRO-3": ZeRO3_Engine(model_cfg, cluster_cfg, hw, ZeROConfig(stage=3, overlap_comm=True)),
    }
    x, y = cyclic_batch(model_cfg)
    logs = {name: [] for name in engines}
    for name, engine in engines.items():
        for step in range(n_steps):
            logs[name].append(engine.train_step(step, x, y))
    return {"model": model_cfg, "logs": logs, "cluster": cluster_cfg, "hw": hw}
