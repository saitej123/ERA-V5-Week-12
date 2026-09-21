"""
CLI Script to execute full ZeRO simulation and export publication-quality plots.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from zero_simulation import (
    ClusterConfig,
    HardwareProfile,
    HARDWARE_PROFILES,
    ModelConfig,
    MODEL_PRESETS,
    ZeROConfig,
    ZeRO0_Engine,
    ZeRO1_Engine,
    ZeRO2_Engine,
    ZeRO3_Engine,
    MemoryProfiler,
    CommunicationProfiler,
    FourQuestionsSettler,
    calculate_model_memory_breakdown,
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'


def plot_stage_memory_breakdown(model_cfg: ModelConfig, output_dir: str = "plots"):
    """Plot memory breakdown across ZeRO stages."""
    df = MemoryProfiler.evaluate_stages_for_model(model_cfg, world_size=32, precision="bf16")
    
    stages = df["Stage"]
    params = df["Params (GB)"]
    grads = df["Gradients (GB)"]
    opt = df["Optimizer States (GB)"]
    acts = df["Activations (GB)"]
    bufs = df["Buffers (GB)"]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(stages))
    width = 0.55
    
    p1 = ax.bar(x, params, width, label="Parameters ($P$)", color="#1f77b4", edgecolor="black")
    p2 = ax.bar(x, grads, width, bottom=params, label="Gradients ($g$)", color="#ff7f0e", edgecolor="black")
    p3 = ax.bar(x, opt, width, bottom=params + grads, label="Optimizer States ($OS$)", color="#2ca02c", edgecolor="black")
    p4 = ax.bar(x, acts, width, bottom=params + grads + opt, label="Activations ($M_{act}$)", color="#d62728", edgecolor="black")
    p5 = ax.bar(x, bufs, width, bottom=params + grads + opt + acts, label="Buffers & Temp", color="#9467bd", edgecolor="black")
    
    # Reference line for 80GB HBM
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80GB HBM Capacity (H100/A100)')
    
    ax.set_ylabel("Memory Footprint per GPU (GB)", fontsize=12, fontweight='bold')
    ax.set_title(f"Per-GPU Memory Breakdown across ZeRO Stages (World Size = 32, {model_cfg.name})", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    
    # Add value annotations
    for i, total in enumerate(df["Total GPU Memory (GB)"]):
        ax.text(i, total + 3, f"{total:.1f} GB\n({df['Fits in HBM'].iloc[i]})", ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "zero_memory_breakdown_stages.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_memory_scaling_sweep(model_cfg: ModelConfig, output_dir: str = "plots"):
    """Plot memory scaling curves across cluster world sizes (1 to 64 GPUs)."""
    df = MemoryProfiler.evaluate_scaling_sweep(model_cfg, world_sizes=[1, 2, 4, 8, 16, 32, 64])
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {"ZeRO-0": "#d62728", "ZeRO-1": "#ff7f0e", "ZeRO-2": "#1f77b4", "ZeRO-3": "#2ca02c"}
    markers = {"ZeRO-0": "o", "ZeRO-1": "s", "ZeRO-2": "^", "ZeRO-3": "D"}
    
    for stage in ["ZeRO-0", "ZeRO-1", "ZeRO-2", "ZeRO-3"]:
        sub = df[df["Stage"] == stage]
        ax.plot(sub["World Size"], sub["Total_GB"], marker=markers[stage], linewidth=2.5, markersize=8,
                label=f"{stage}", color=colors[stage])
        
    ax.axhline(y=80, color='black', linestyle='--', linewidth=1.5, label='80GB HBM Threshold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel("World Size (Total Virtual GPUs)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Per-GPU Memory Footprint (GB, Log Scale)", fontsize=12, fontweight='bold')
    ax.set_title(f"ZeRO Memory Scaling vs World Size (1 to 64 GPUs, {model_cfg.name})", fontsize=13, fontweight='bold')
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "zero_memory_scaling_world_sizes.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_training_steps_logged(output_dir: str = "plots"):
    """Simulate 32 virtual GPUs running 10 training steps and log communication fraction from Step 1."""
    model_cfg = MODEL_PRESETS["demo_small"]
    cluster_cfg = ClusterConfig(world_size=32, gpus_per_node=8)
    hw = HARDWARE_PROFILES["H100_SXM5"]
    
    # Run 5 steps of ZeRO-2 and ZeRO-3 on 32 virtual GPUs
    engine2 = ZeRO2_Engine(model_cfg, cluster_cfg, hw, ZeROConfig(stage=2, precision="bf16", overlap_comm=True))
    engine3 = ZeRO3_Engine(model_cfg, cluster_cfg, hw, ZeROConfig(stage=3, precision="bf16", overlap_comm=True))
    
    step_records_z2 = []
    step_records_z3 = []
    
    for step in range(8):
        x = torch.randint(0, model_cfg.vocab_size, (model_cfg.micro_batch_size, model_cfg.seq_len))
        y = torch.randint(0, model_cfg.vocab_size, (model_cfg.micro_batch_size, model_cfg.seq_len))
        r2 = engine2.train_step(step, x, y)
        r3 = engine3.train_step(step, x, y)
        step_records_z2.append(r2)
        step_records_z3.append(r3)
        
    df2 = pd.DataFrame(step_records_z2)
    df3 = pd.DataFrame(step_records_z3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    steps = df2["step"] + 1
    # Plot 1: Step Time Breakdown
    ax1.plot(steps, df2["total_step_time_ms"], 'o-', color='#1f77b4', linewidth=2.5, label='ZeRO-2 Total Step Time')
    ax1.plot(steps, df3["total_step_time_ms"], 's-', color='#2ca02c', linewidth=2.5, label='ZeRO-3 Total Step Time')
    ax1.plot(steps, df2["compute_time_ms"], '--', color='gray', linewidth=2, label='Compute Time')
    ax1.set_xlabel("Training Step", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Time (ms)", fontsize=11, fontweight='bold')
    ax1.set_title("Step Time Logging (Bucketing & Overlap Active)", fontsize=12, fontweight='bold')
    ax1.legend(loc="best")
    
    # Plot 2: Communication Fraction of Step Time
    ax2.plot(steps, df2["comm_fraction"] * 100, 'o-', color='#1f77b4', linewidth=2.5, label='ZeRO-2 Comm Fraction (%)')
    ax2.plot(steps, df3["comm_fraction"] * 100, 's-', color='#2ca02c', linewidth=2.5, label='ZeRO-3 Comm Fraction (%)')
    ax2.set_xlabel("Training Step", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Communication Fraction (%)", fontsize=11, fontweight='bold')
    ax2.set_title("Logged Communication Fraction ($T_{comm} / T_{step}$)", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc="best")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "comm_fraction_step_log.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_section5_evolution(output_dir: str = "plots"):
    """Plot Section 5 effect: Rising communication fraction as compute accelerates from A100 to H100 to B200."""
    target_model = MODEL_PRESETS["llm_20b"]
    df = CommunicationProfiler.hardware_evolution_sweep(target_model, zero_stage=2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    hw_names = ["Ampere A100", "Hopper H100", "Blackwell B200"]
    x = np.arange(len(hw_names))
    width = 0.35
    
    # Subplot 1: Compute Time vs Comm Time
    ax1.bar(x - width/2, df["Compute Time (ms)"], width, label="Compute Time ($T_{compute}$)", color="#1f77b4", edgecolor="black")
    ax1.bar(x + width/2, df["Comm Time (ms)"], width, label="Comm Time ($T_{comm}$)", color="#d62728", edgecolor="black")
    ax1.set_ylabel("Time per Step (ms)", fontsize=11, fontweight='bold')
    ax1.set_title("Compute Time Drop vs Comm Time", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(hw_names, fontweight='bold')
    ax1.legend()
    
    # Subplot 2: Rising Communication Fraction (%)
    ax2.plot(x, df["Raw Comm Fraction (%)"], 'o-', color='#d62728', linewidth=3, markersize=10, label='Raw Comm Fraction')
    ax2.plot(x, df["Exposed Comm Fraction (%)"], 's--', color='#2ca02c', linewidth=2.5, markersize=8, label='Exposed Comm Fraction (with Overlap)')
    ax2.set_ylabel("Communication Fraction (%)", fontsize=11, fontweight='bold')
    ax2.set_title("Section 5 Phenomenon: Rising Communication Fraction", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(hw_names, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    for i, val in enumerate(df["Raw Comm Fraction (%)"]):
        ax2.annotate(f"{val:.1f}%", (x[i], val + 3), ha='center', fontweight='bold')
        
    ax2.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "section5_hardware_evolution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_four_questions_artifacts(output_dir: str = "plots"):
    """Generate plots settling the four open questions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Question 1: ZeRO-2 @ 32 vs ZeRO-3 @ 8
    q1 = FourQuestionsSettler.settle_question_1()
    df_q1 = q1["table"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    cfgs = ["ZeRO-2 (32 GPUs)", "ZeRO-3 (8 GPUs)"]
    
    # Throughput
    ax1.bar(cfgs, df_q1["Throughput (Tokens/s)"], color=["#1f77b4", "#ff7f0e"], width=0.45, edgecolor="black")
    ax1.set_ylabel("Throughput (Tokens / Sec)", fontsize=11, fontweight='bold')
    ax1.set_title("Question 1: Training Throughput", fontsize=12, fontweight='bold')
    for i, v in enumerate(df_q1["Throughput (Tokens/s)"]):
        ax1.text(i, v + 200, f"{v:.0f} tok/s\n({df_q1['Speedup vs ZeRO-3 (8 GPUs)'].iloc[i]:.2f}x)", ha='center', fontweight='bold')
        
    # Memory Breakdown
    mem_labels = ["Params", "Grads", "Optimizer", "Activations"]
    z2_mem = [df_q1["Per-GPU Param Mem (GB)"].iloc[0], df_q1["Per-GPU Grad Mem (GB)"].iloc[0], df_q1["Per-GPU Opt Mem (GB)"].iloc[0], df_q1["Per-GPU Act Mem (GB)"].iloc[0]]
    z3_mem = [df_q1["Per-GPU Param Mem (GB)"].iloc[1], df_q1["Per-GPU Grad Mem (GB)"].iloc[1], df_q1["Per-GPU Opt Mem (GB)"].iloc[1], df_q1["Per-GPU Act Mem (GB)"].iloc[1]]
    
    x = np.arange(len(cfgs))
    p1 = ax2.bar(x, [z2_mem[0], z3_mem[0]], 0.45, label="Params", color="#1f77b4", edgecolor="black")
    p2 = ax2.bar(x, [z2_mem[1], z3_mem[1]], 0.45, bottom=[z2_mem[0], z3_mem[0]], label="Grads", color="#ff7f0e", edgecolor="black")
    p3 = ax2.bar(x, [z2_mem[2], z3_mem[2]], 0.45, bottom=[z2_mem[0]+z2_mem[1], z3_mem[0]+z3_mem[1]], label="Optimizer", color="#2ca02c", edgecolor="black")
    p4 = ax2.bar(x, [z2_mem[3], z3_mem[3]], 0.45, bottom=[z2_mem[0]+z2_mem[1]+z2_mem[2], z3_mem[0]+z3_mem[1]+z3_mem[2]], label="Activations", color="#d62728", edgecolor="black")
    ax2.axhline(y=80, color='black', linestyle='--', label='80GB HBM')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfgs, fontweight='bold')
    ax2.set_ylabel("Memory per GPU (GB)", fontsize=11, fontweight='bold')
    ax2.set_title("Question 1: Per-GPU Memory Footprint", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question1_zero2_vs_zero3.png"))
    plt.close()
    
    # Question 2: Topologies (Intra vs Inter Node)
    q2 = FourQuestionsSettler.settle_question_2()
    df_q2 = q2["table"]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    topos = ["4 Nodes x 8 GPUs\n(HGX Standard)", "2 Nodes x 16 GPUs\n(Expanded)", "1 Node x 32 GPUs\n(NVLink Rack)"]
    x = np.arange(len(topos))
    ax.bar(x, df_q2["Intra-Node Traffic (GB)"], 0.45, label="Intra-Node (NVLink @ 900 GB/s)", color="#2ca02c", edgecolor="black")
    ax.bar(x, df_q2["Inter-Node Traffic (GB)"], 0.45, bottom=df_q2["Intra-Node Traffic (GB)"], label="Inter-Node (InfiniBand @ 50 GB/s)", color="#d62728", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(topos, fontweight='bold')
    ax.set_ylabel("Communication Volume per Step (GB)", fontsize=11, fontweight='bold')
    ax.set_title("Question 2: Traffic Distribution across Topologies (Section 5 18x Bandwidth Gap)", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question2_topologies_nvlink_ib.png"))
    plt.close()
    
    # Question 3: BF16 vs MXFP8
    q3 = FourQuestionsSettler.settle_question_3(num_steps=10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    steps = range(1, 11)
    ax1.plot(steps, q3["bf16_losses"], 'o-', color='#1f77b4', linewidth=2.5, label='BF16 Loss')
    ax1.plot(steps, q3["mxfp8_losses"], 's--', color='#2ca02c', linewidth=2.5, label='MXFP8 Loss')
    ax1.set_xlabel("Step", fontweight='bold')
    ax1.set_ylabel("Cross Entropy Loss", fontweight='bold')
    ax1.set_title("Question 3: Convergence Fidelity (BF16 vs MXFP8)", fontweight='bold')
    ax1.legend()
    
    precisions = ["BF16 (16-bit)", "MXFP8 (8-bit)"]
    step_times = [q3["table"]["Total Step Time (ms)"].iloc[0], q3["table"]["Total Step Time (ms)"].iloc[1]]
    ax2.bar(precisions, step_times, color=["#1f77b4", "#2ca02c"], width=0.45, edgecolor="black")
    ax2.set_ylabel("Step Time (ms)", fontweight='bold')
    ax2.set_title("Question 3: Step Time on Blackwell Hardware", fontweight='bold')
    for i, st in enumerate(step_times):
        ax2.text(i, st + 2, f"{st:.1f} ms", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question3_bf16_vs_mxfp8.png"))
    plt.close()
    
    # Question 4: ZeRO-Offload vs GPU HBM
    q4 = FourQuestionsSettler.settle_question_4()
    df_q4 = q4["table"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    modes = ["Pure GPU HBM\n(No Offload)", "ZeRO-Offload\n(CPU DRAM over PCIe)"]
    tputs = df_q4["Throughput (Tokens/s)"]
    ax.bar(modes, tputs, color=["#2ca02c", "#d62728"], width=0.45, edgecolor="black")
    ax.set_ylabel("Throughput (Tokens / Sec)", fontweight='bold')
    ax.set_title("Question 4: Pure GPU HBM vs ZeRO-Offload PCIe Penalty", fontweight='bold')
    for i, tp in enumerate(tputs):
        ax.text(i, tp + 500, f"{tp:.0f} tok/s\n({df_q4['Slowdown Penalty'].iloc[i]})", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question4_zero_offload_penalty.png"))
    plt.close()
    print("[Saved] All Question artifact plots.")


def main():
    print("================================================================================")
    print("        RUNNING COMPLETE ZERO PARALLELISM SIMULATION SUITE                     ")
    print("================================================================================")
    
    target_model = MODEL_PRESETS["llm_20b"]
    print(f"Target Architecture: {target_model.name} (~{target_model.total_parameters / 1e9:.2f}B Parameters)")
    
    plot_stage_memory_breakdown(target_model)
    plot_memory_scaling_sweep(target_model)
    plot_training_steps_logged()
    plot_section5_evolution()
    plot_four_questions_artifacts()
    
    print("\nAll simulations and plots generated successfully in plots/ directory.")


if __name__ == "__main__":
    main()
