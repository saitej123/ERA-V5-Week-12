"""
Memory tracking and scaling analysis across ZeRO stages and cluster world sizes.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .config import ModelConfig, HardwareProfile, ZeROConfig
from .model import calculate_model_memory_breakdown


class MemoryProfiler:
    """Computes and analyzes memory footprints across configurations."""
    
    @staticmethod
    def evaluate_stages_for_model(
        model_cfg: ModelConfig,
        world_size: int = 32,
        precision: str = "bf16",
        activation_checkpointing: bool = True,
        hbm_capacity_gb: float = 80.0
    ) -> pd.DataFrame:
        """Evaluate memory footprints for ZeRO-0, ZeRO-1, ZeRO-2, ZeRO-3 at fixed world size."""
        records = []
        for stage in [0, 1, 2, 3]:
            breakdown = calculate_model_memory_breakdown(
                model_cfg=model_cfg,
                zero_stage=stage,
                world_size=world_size,
                precision=precision,
                activation_checkpointing=activation_checkpointing
            )
            records.append({
                "Stage": f"ZeRO-{stage}",
                "World Size": world_size,
                "Params (GB)": breakdown["params_gb"],
                "Gradients (GB)": breakdown["grads_gb"],
                "Optimizer States (GB)": breakdown["optimizer_gb"],
                "Activations (GB)": breakdown["activation_gb"],
                "Buffers (GB)": breakdown["buffer_gb"],
                "Total GPU Memory (GB)": breakdown["total_gb"],
                "Fits in HBM": "YES" if breakdown["total_gb"] <= hbm_capacity_gb else "OOM",
                "Memory Reduction vs DP": 1.0 if stage == 0 else (
                    calculate_model_memory_breakdown(model_cfg, 0, world_size, precision, activation_checkpointing)["total_gb"] / breakdown["total_gb"]
                )
            })
        return pd.DataFrame(records)

    @staticmethod
    def evaluate_scaling_sweep(
        model_cfg: ModelConfig,
        world_sizes: Optional[List[int]] = None,
        precision: str = "bf16",
        activation_checkpointing: bool = True
    ) -> pd.DataFrame:
        """Sweep world sizes from 1 to 128 GPUs across all ZeRO stages."""
        if world_sizes is None:
            world_sizes = [1, 2, 4, 8, 16, 32, 64]
            
        records = []
        for n in world_sizes:
            for stage in [0, 1, 2, 3]:
                breakdown = calculate_model_memory_breakdown(
                    model_cfg=model_cfg,
                    zero_stage=stage,
                    world_size=n,
                    precision=precision,
                    activation_checkpointing=activation_checkpointing
                )
                records.append({
                    "World Size": n,
                    "Stage": f"ZeRO-{stage}",
                    "Stage_Int": stage,
                    "Params_GB": breakdown["params_gb"],
                    "Grads_GB": breakdown["grads_gb"],
                    "Optimizer_GB": breakdown["optimizer_gb"],
                    "Activations_GB": breakdown["activation_gb"],
                    "Total_GB": breakdown["total_gb"]
                })
        return pd.DataFrame(records)
