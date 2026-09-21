"""
Communication profiler and Section 5 analysis.
Measures communication volume, transfer latency, communication fraction over steps, and hardware scaling.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .config import HardwareProfile, HARDWARE_PROFILES, HARDWARE_GENERATION_ORDER, ModelConfig, ClusterConfig, ZeROConfig
from .model import calculate_step_compute_flops


class CommunicationProfiler:
    """Simulates communication dynamics, hardware scaling, and topology effects."""
    
    @staticmethod
    def calculate_comm_volume_bytes(model_cfg: ModelConfig, zero_stage: int, precision: str = "bf16") -> Dict[str, float]:
        """
        Calculate total communication volume (bytes) per training step per GPU.
        
        Volumes:
        - ZeRO-0 (DP): 2 * Psi (AllReduce gradients)
        - ZeRO-1 (Pos): 1 * Psi (ReduceScatter gradients) + 1 * Psi (AllGather params) = 2 * Psi
        - ZeRO-2 (Pos+g): 1 * Psi (ReduceScatter gradients) + 1 * Psi (AllGather params) = 2 * Psi
        - ZeRO-3 (Pos+g+p): 1 * Psi (fwd AllGather) + 1 * Psi (bwd AllGather) + 1 * Psi (bwd ReduceScatter) = 3 * Psi
        """
        total_params = model_cfg.total_parameters
        elem_bytes = 1 if precision == "mxfp8" else 2
        psi_bytes = total_params * elem_bytes
        
        if zero_stage in [0, 1, 2]:
            comm_volume = 2.0 * psi_bytes
            fwd_volume = 0.0
            bwd_volume = psi_bytes
            step_update_volume = psi_bytes if zero_stage in [1, 2] else 0.0
        elif zero_stage == 3:
            comm_volume = 3.0 * psi_bytes
            fwd_volume = psi_bytes
            bwd_volume = 2.0 * psi_bytes
            step_update_volume = 0.0
        else:
            raise ValueError(f"Unknown stage {zero_stage}")
            
        return {
            "total_bytes": comm_volume,
            "total_gb": comm_volume / (1024 ** 3),
            "fwd_bytes": fwd_volume,
            "bwd_bytes": bwd_volume,
            "step_update_bytes": step_update_volume
        }

    @staticmethod
    def profile_step_time(
        model_cfg: ModelConfig,
        hw_profile: HardwareProfile,
        cluster_cfg: ClusterConfig,
        zero_stage: int,
        precision: str = "bf16",
        bucket_size_mb: float = 25.0,
        overlap: bool = True,
        activation_checkpointing: bool = True
    ) -> Dict[str, Any]:
        """
        Computes accurate step time breakdown:
        - T_compute: FLOPs / TFLOPS
        - T_comm: (Volume / Bandwidth) + Ring Latency
        - T_comm_exposed: Overlapped communication
        - Communication fraction: T_comm / (T_compute + T_comm)
        """
        # 1. Compute FLOPs & Time
        compute_flops = calculate_step_compute_flops(model_cfg, activation_checkpointing)
        tflops_peak = hw_profile.fp8_tflops if precision == "mxfp8" else hw_profile.bf16_tflops
        
        # Real-world MFU (Model FLOPs Utilization) efficiency factor ~ 45-55%
        mfu = 0.50
        effective_tflops = tflops_peak * mfu
        compute_time_sec = compute_flops / (effective_tflops * 1e12)
        compute_time_ms = compute_time_sec * 1000.0

        # 2. Communication Volume & Time
        comm_vol = CommunicationProfiler.calculate_comm_volume_bytes(model_cfg, zero_stage, precision)
        total_vol_bytes = comm_vol["total_bytes"]
        
        # Topology breakdown
        N = cluster_cfg.world_size
        gpus_per_node = cluster_cfg.gpus_per_node
        num_nodes = cluster_cfg.num_nodes
        
        if num_nodes == 1:
            intra_vol = total_vol_bytes
            inter_vol = 0
            bw = hw_profile.intra_node_bandwidth_gbps * 1e9
            latency = (hw_profile.intra_node_latency_us * 1e-6) * (2 * (N - 1))
            comm_time_sec = latency + (intra_vol / bw)
        else:
            intra_ratio = (gpus_per_node - 1) / (N - 1)
            inter_ratio = (N - gpus_per_node) / (N - 1)
            intra_vol = total_vol_bytes * intra_ratio
            inter_vol = total_vol_bytes * inter_ratio
            
            bw_intra = hw_profile.intra_node_bandwidth_gbps * 1e9
            bw_inter = hw_profile.inter_node_bandwidth_gbps * 1e9
            
            lat_intra = (hw_profile.intra_node_latency_us * 1e-6) * (2 * (gpus_per_node - 1))
            lat_inter = (hw_profile.inter_node_latency_us * 1e-6) * (2 * (num_nodes - 1))
            
            comm_time_sec = lat_intra + lat_inter + (intra_vol / bw_intra) + (inter_vol / bw_inter)

        comm_time_ms = comm_time_sec * 1000.0

        # 3. Overlap and Bucketing Efficiency
        if overlap:
            # Gradient bucketing allows pipelining communication with backward compute
            # ZeRO-2 achieves ~85% overlap of ReduceScatter; ZeRO-3 achieves ~75% overlap of layer AG + RS
            overlap_efficiency = 0.85 if zero_stage == 2 else (0.75 if zero_stage == 3 else 0.65)
            # Available backward/forward compute time window
            overlap_window_ms = 0.66 * compute_time_ms
            exposed_comm_ms = max(0.0, comm_time_ms - (overlap_efficiency * overlap_window_ms))
            step_time_ms = compute_time_ms + exposed_comm_ms
        else:
            exposed_comm_ms = comm_time_ms
            step_time_ms = compute_time_ms + comm_time_ms

        comm_fraction = comm_time_ms / (compute_time_ms + comm_time_ms) if (compute_time_ms + comm_time_ms) > 0 else 0
        exposed_comm_fraction = exposed_comm_ms / step_time_ms if step_time_ms > 0 else 0
        
        tokens_per_step = model_cfg.micro_batch_size * model_cfg.seq_len * cluster_cfg.world_size
        throughput_tokens_per_sec = (tokens_per_step / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0

        return {
            "stage": f"ZeRO-{zero_stage}",
            "stage_num": zero_stage,
            "hardware": hw_profile.name,
            "world_size": N,
            "precision": precision,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": comm_time_ms,
            "exposed_comm_ms": exposed_comm_ms,
            "step_time_ms": step_time_ms,
            "comm_fraction": comm_fraction,
            "exposed_comm_fraction": exposed_comm_fraction,
            "comm_volume_gb": total_vol_bytes / (1024 ** 3),
            "throughput_tokens_sec": throughput_tokens_per_sec
        }

    @staticmethod
    def hardware_evolution_sweep(model_cfg: ModelConfig, zero_stage: int = 2) -> pd.DataFrame:
        """
        Demonstrates the Section 5 phenomenon:
        As hardware generations advance (A100 -> H100 -> B200), compute gets drastically faster,
        causing Communication Fraction of step time to rise from ~20% to >55% if network doesn't match compute growth.
        """
        records = []
        frozen_ib = HARDWARE_PROFILES["A100_SXM4"].inter_node_bandwidth_gbps
        frozen_nvlink = HARDWARE_PROFILES["A100_SXM4"].intra_node_bandwidth_gbps
        for hw_key in HARDWARE_GENERATION_ORDER:
            hw = HARDWARE_PROFILES[hw_key]
            native = CommunicationProfiler.profile_step_time(
                model_cfg=model_cfg,
                hw_profile=hw,
                cluster_cfg=ClusterConfig(world_size=32, gpus_per_node=8),
                zero_stage=zero_stage,
                precision="bf16",
                overlap=True,
            )
            # Section 5: if the network does not keep up with Tensor Core growth,
            # hold interconnect at A100 HDR levels while compute scales.
            frozen_hw = HardwareProfile(
                name=hw.name + " (frozen network)",
                hbm_capacity_gb=hw.hbm_capacity_gb,
                bf16_tflops=hw.bf16_tflops,
                fp8_tflops=hw.fp8_tflops,
                intra_node_bandwidth_gbps=frozen_nvlink,
                inter_node_bandwidth_gbps=frozen_ib,
                pcie_bandwidth_gbps=hw.pcie_bandwidth_gbps,
                intra_node_latency_us=hw.intra_node_latency_us,
                inter_node_latency_us=hw.inter_node_latency_us,
                cpu_dram_bandwidth_gbps=hw.cpu_dram_bandwidth_gbps,
            )
            frozen = CommunicationProfiler.profile_step_time(
                model_cfg=model_cfg,
                hw_profile=frozen_hw,
                cluster_cfg=ClusterConfig(world_size=32, gpus_per_node=8),
                zero_stage=zero_stage,
                precision="bf16",
                overlap=True,
            )
            short = {
                "A100_SXM4": "Ampere A100",
                "H100_SXM5": "Hopper H100",
                "B200_NVL72": "Blackwell B200",
            }[hw_key]
            records.append({
                "Hardware Key": hw_key,
                "Hardware Generation": short,
                "Compute Peak TFLOPS": hw.bf16_tflops,
                "Intra-Node NVLink (GB/s)": hw.intra_node_bandwidth_gbps,
                "Inter-Node IB (GB/s)": hw.inter_node_bandwidth_gbps,
                "Compute Time (ms)": native["compute_time_ms"],
                "Comm Time (ms)": native["comm_time_ms"],
                "Exposed Comm (ms)": native["exposed_comm_ms"],
                "Total Step Time (ms)": native["step_time_ms"],
                "Raw Comm Fraction (%)": native["comm_fraction"] * 100,
                "Exposed Comm Fraction (%)": native["exposed_comm_fraction"] * 100,
                "Frozen-Net Comm Fraction (%)": frozen["comm_fraction"] * 100,
                "Tokens / sec": native["throughput_tokens_sec"],
            })
        return pd.DataFrame(records)
