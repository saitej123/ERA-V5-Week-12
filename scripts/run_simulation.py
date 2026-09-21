"""
CLI: run the ZeRO simulation and export README screenshots.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from zero_simulation import (
    ClusterConfig,
    HARDWARE_PROFILES,
    MODEL_PRESETS,
    ZeROConfig,
    ZeRO2_Engine,
    ZeRO3_Engine,
    MemoryProfiler,
    CommunicationProfiler,
    FourQuestionsSettler,
)

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

PALETTE = {
    "params": "#3B6FB6",
    "grads": "#E07A3D",
    "opt": "#3A9E6F",
    "act": "#C44B4B",
    "buf": "#7D6BB0",
    "zero0": "#C44B4B",
    "zero1": "#E07A3D",
    "zero2": "#3B6FB6",
    "zero3": "#3A9E6F",
}


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[Saved] {path}")


def _label_bars(ax, values, fmt="{:.1f}", ypad=0.03):
    ymax = max(values) if len(values) else 1.0
    ax.set_ylim(0, ymax * (1.0 + ypad + 0.18))
    for i, v in enumerate(values):
        ax.text(i, v + ymax * ypad, fmt.format(v), ha="center", va="bottom", fontweight="bold", fontsize=9)


def plot_stage_memory_breakdown(model_cfg, output_dir="plots"):
    df = MemoryProfiler.evaluate_stages_for_model(model_cfg, world_size=32, precision="bf16")
    stages = df["Stage"]
    params = df["Params (GB)"]
    grads = df["Gradients (GB)"]
    opt = df["Optimizer States (GB)"]
    acts = df["Activations (GB)"]
    bufs = df["Buffers (GB)"]
    totals = df["Total GPU Memory (GB)"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(stages))
    w = 0.58
    ax.bar(x, params, w, label="Parameters (P)", color=PALETTE["params"], edgecolor="black", linewidth=0.6)
    ax.bar(x, grads, w, bottom=params, label="Gradients (g)", color=PALETTE["grads"], edgecolor="black", linewidth=0.6)
    ax.bar(x, opt, w, bottom=params + grads, label="Optimizer states (OS)", color=PALETTE["opt"], edgecolor="black", linewidth=0.6)
    ax.bar(x, acts, w, bottom=params + grads + opt, label="Activations", color=PALETTE["act"], edgecolor="black", linewidth=0.6)
    ax.bar(x, bufs, w, bottom=params + grads + opt + acts, label="Buffers", color=PALETTE["buf"], edgecolor="black", linewidth=0.6)
    ax.axhline(80, color="#B00020", linestyle="--", linewidth=1.8, label="80 GB HBM")
    ax.set_ylabel("Memory per GPU (GB)")
    ax.set_title(f"Per-GPU memory by ZeRO stage  ·  32 GPUs  ·  {model_cfg.name}")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, max(totals.max() * 1.18, 95))
    for i, total in enumerate(totals):
        tag = df["Fits in HBM"].iloc[i]
        ax.text(i, total + max(totals) * 0.02, f"{total:.1f} GB\n({tag})", ha="center", va="bottom", fontweight="bold", fontsize=9)
    _save(fig, os.path.join(output_dir, "zero_memory_breakdown_stages.png"))


def plot_memory_scaling_sweep(model_cfg, output_dir="plots"):
    df = MemoryProfiler.evaluate_scaling_sweep(model_cfg, world_sizes=[1, 2, 4, 8, 16, 32, 64])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    style = {
        "ZeRO-0": (PALETTE["zero0"], "o"),
        "ZeRO-1": (PALETTE["zero1"], "s"),
        "ZeRO-2": (PALETTE["zero2"], "^"),
        "ZeRO-3": (PALETTE["zero3"], "D"),
    }
    for stage, (color, marker) in style.items():
        sub = df[df["Stage"] == stage]
        ax.plot(sub["World Size"], sub["Total_GB"], marker=marker, linewidth=2.4, markersize=8, label=stage, color=color)
    ax.axhline(80, color="black", linestyle="--", linewidth=1.4, label="80 GB HBM")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("World size (GPUs)")
    ax.set_ylabel("Per-GPU memory (GB, log)")
    ax.set_title(f"Does it fit as we add GPUs?  ·  {model_cfg.name}")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax.legend(loc="upper right")
    _save(fig, os.path.join(output_dir, "zero_memory_scaling_world_sizes.png"))


def plot_memory_and_compute_changes(model_cfg, output_dir="plots"):
    """Show how memory AND computation/communication change across ZeRO stages."""
    hw = HARDWARE_PROFILES["H100_SXM5"]
    cluster = ClusterConfig(world_size=32, gpus_per_node=8)
    mem = MemoryProfiler.evaluate_stages_for_model(model_cfg, world_size=32)
    rows = []
    for stage in [0, 1, 2, 3]:
        step = CommunicationProfiler.profile_step_time(
            model_cfg, hw, cluster, zero_stage=stage, precision="bf16", overlap=True
        )
        rows.append(step)
    step_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    stages = ["ZeRO-0", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    colors = [PALETTE["zero0"], PALETTE["zero1"], PALETTE["zero2"], PALETTE["zero3"]]

    axes[0].bar(stages, mem["Total GPU Memory (GB)"], color=colors, edgecolor="black", width=0.65)
    axes[0].axhline(80, color="#B00020", linestyle="--", linewidth=1.5, label="80 GB")
    axes[0].set_title("Memory / GPU")
    axes[0].set_ylabel("GB")
    axes[0].legend(loc="upper right")
    _label_bars(axes[0], list(mem["Total GPU Memory (GB)"]), "{:.0f}")

    axes[1].bar(stages, step_df["comm_volume_gb"], color=colors, edgecolor="black", width=0.65)
    axes[1].set_title("Comm volume / GPU / step")
    axes[1].set_ylabel("GB")
    _label_bars(axes[1], list(step_df["comm_volume_gb"]), "{:.1f}")

    axes[2].bar(stages, step_df["step_time_ms"], color=colors, edgecolor="black", width=0.65)
    axes[2].set_title("Step time (H100, overlap on)")
    axes[2].set_ylabel("ms")
    _label_bars(axes[2], list(step_df["step_time_ms"]), "{:.0f}")

    fig.suptitle(f"How ZeRO changes memory and step cost  ·  32 GPUs  ·  {model_cfg.name}", fontweight="bold", y=1.02)
    _save(fig, os.path.join(output_dir, "memory_and_compute_changes.png"))


def plot_training_steps_logged(output_dir="plots"):
    """Log comm fraction from step 1 on the 20B target, plus a live 32-rank demo."""
    target = MODEL_PRESETS["llm_20b"]
    hw = HARDWARE_PROFILES["H100_SXM5"]
    records = []
    for step in range(1, 9):
        for stage, gpus, gpn in [(2, 32, 8), (3, 8, 8)]:
            r = CommunicationProfiler.profile_step_time(
                target, hw, ClusterConfig(world_size=gpus, gpus_per_node=gpn),
                zero_stage=stage, precision="bf16", overlap=True,
            )
            records.append({
                "step": step,
                "stage": f"ZeRO-{stage} @ {gpus} GPUs",
                "step_time_ms": r["step_time_ms"],
                "compute_time_ms": r["compute_time_ms"],
                "comm_fraction": r["comm_fraction"] * 100,
                "exposed": r["exposed_comm_fraction"] * 100,
            })
    df = pd.DataFrame(records)

    demo_cfg = MODEL_PRESETS["demo_small"]
    engine = ZeRO2_Engine(
        demo_cfg, ClusterConfig(world_size=32, gpus_per_node=8), hw,
        ZeROConfig(stage=2, precision="bf16", overlap_comm=True),
    )
    live_loss = []
    period = min(64, demo_cfg.vocab_size)
    x = (torch.arange(demo_cfg.seq_len) % period).unsqueeze(0).repeat(demo_cfg.micro_batch_size, 1)
    y = torch.roll(x, shifts=-1, dims=1)
    for step in range(8):
        live_loss.append(engine.train_step(step, x, y)["loss"])

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.7))
    for label, sub in df.groupby("stage"):
        axes[0].plot(sub["step"], sub["step_time_ms"], "o-", linewidth=2.3, markersize=7, label=label)
    axes[0].plot(df[df["stage"] == "ZeRO-2 @ 32 GPUs"]["step"], df[df["stage"] == "ZeRO-2 @ 32 GPUs"]["compute_time_ms"],
                 "--", color="gray", label="Compute only (ZeRO-2)")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("ms")
    axes[0].set_title("20B step time (logged from step 1)")
    axes[0].set_ylim(0, None)
    axes[0].legend(fontsize=8)

    for label, sub in df.groupby("stage"):
        axes[1].plot(sub["step"], sub["comm_fraction"], "o-", linewidth=2.3, markersize=7, label=label + " raw")
    axes[1].plot(
        df[df["stage"] == "ZeRO-2 @ 32 GPUs"]["step"],
        df[df["stage"] == "ZeRO-2 @ 32 GPUs"]["exposed"],
        "s--", color=PALETTE["zero3"], label="ZeRO-2 exposed (overlap)",
    )
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Comm fraction (%)")
    axes[1].set_title("T_comm / (T_compute + T_comm)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)

    axes[2].plot(range(1, 9), live_loss, "o-", color=PALETTE["zero2"], linewidth=2.3, markersize=7)
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Cross-entropy")
    axes[2].set_title("Live demo on 32 virtual GPUs")
    _save(fig, os.path.join(output_dir, "comm_fraction_step_log.png"))


def plot_section5_evolution(output_dir="plots"):
    df = CommunicationProfiler.hardware_evolution_sweep(MODEL_PRESETS["llm_20b"], zero_stage=2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    names = list(df["Hardware Generation"])
    x = np.arange(len(names))
    w = 0.36
    ax1.bar(x - w / 2, df["Compute Time (ms)"], w, label="Compute", color=PALETTE["zero2"], edgecolor="black")
    ax1.bar(x + w / 2, df["Comm Time (ms)"], w, label="Communication", color=PALETTE["zero0"], edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontweight="bold")
    ax1.set_ylabel("Time per step (ms)")
    ax1.set_title("Compute drops faster than the network")
    ax1.legend()

    ax2.plot(x, df["Raw Comm Fraction (%)"], "o-", color=PALETTE["zero0"], linewidth=2.6, markersize=9, label="Native network")
    ax2.plot(x, df["Frozen-Net Comm Fraction (%)"], "s--", color=PALETTE["zero1"], linewidth=2.3, markersize=8, label="Network frozen at A100 HDR")
    ax2.plot(x, df["Exposed Comm Fraction (%)"], "^:", color=PALETTE["zero3"], linewidth=2.2, markersize=8, label="Exposed after overlap")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontweight="bold")
    ax2.set_ylabel("Communication fraction (%)")
    ax2.set_title("Section 5: fraction rises as GPUs get faster")
    ax2.set_ylim(0, 100)
    for i, val in enumerate(df["Raw Comm Fraction (%)"]):
        ax2.annotate(f"{val:.1f}%", (x[i], val + 3), ha="center", fontsize=9, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)
    _save(fig, os.path.join(output_dir, "section5_hardware_evolution.png"))


def plot_four_questions_artifacts(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    q1 = FourQuestionsSettler.settle_question_1()
    df_q1 = q1["table"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.1))
    cfgs = ["ZeRO-2\n32 GPUs", "ZeRO-3\n8 GPUs"]
    tputs = list(df_q1["Throughput (Tokens/s)"])
    ax1.bar(cfgs, tputs, color=[PALETTE["zero2"], PALETTE["zero1"]], width=0.5, edgecolor="black")
    ax1.set_ylabel("Tokens / sec")
    ax1.set_title("Q1 · Throughput (activation memory included)")
    ymax = max(tputs)
    ax1.set_ylim(0, ymax * 1.22)
    for i, v in enumerate(tputs):
        ax1.text(i, v + ymax * 0.03, f"{v:,.0f} tok/s\n({df_q1['Speedup vs ZeRO-3 (8 GPUs)'].iloc[i]:.2f}x)",
                 ha="center", fontweight="bold", fontsize=9)

    z2 = [df_q1["Per-GPU Param Mem (GB)"].iloc[0], df_q1["Per-GPU Grad Mem (GB)"].iloc[0],
          df_q1["Per-GPU Opt Mem (GB)"].iloc[0], df_q1["Per-GPU Act Mem (GB)"].iloc[0]]
    z3 = [df_q1["Per-GPU Param Mem (GB)"].iloc[1], df_q1["Per-GPU Grad Mem (GB)"].iloc[1],
          df_q1["Per-GPU Opt Mem (GB)"].iloc[1], df_q1["Per-GPU Act Mem (GB)"].iloc[1]]
    x = np.arange(2)
    ax2.bar(x, [z2[0], z3[0]], 0.5, label="Params", color=PALETTE["params"], edgecolor="black")
    ax2.bar(x, [z2[1], z3[1]], 0.5, bottom=[z2[0], z3[0]], label="Grads", color=PALETTE["grads"], edgecolor="black")
    ax2.bar(x, [z2[2], z3[2]], 0.5, bottom=[z2[0] + z2[1], z3[0] + z3[1]], label="Optimizer", color=PALETTE["opt"], edgecolor="black")
    ax2.bar(x, [z2[3], z3[3]], 0.5, bottom=[z2[0] + z2[1] + z2[2], z3[0] + z3[1] + z3[2]],
            label="Activations", color=PALETTE["act"], edgecolor="black")
    ax2.axhline(80, color="black", linestyle="--", label="80 GB HBM")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfgs, fontweight="bold")
    ax2.set_ylabel("GB / GPU")
    ax2.set_title("Q1 · Memory (fits both ways)")
    ax2.set_ylim(0, 95)
    ax2.legend(loc="upper right", fontsize=8)
    _save(fig, os.path.join(output_dir, "question1_zero2_vs_zero3.png"))

    q2 = FourQuestionsSettler.settle_question_2()
    df_q2 = q2["table"]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.8, 5.1))
    topos = ["4×8 GPUs\n4 nodes", "2×16 GPUs\n2 nodes", "1×32 GPUs\n1 NVLink domain"]
    x = np.arange(3)
    ax.bar(x, df_q2["Intra-Node Traffic (GB)"], 0.55, label="NVLink (900 GB/s)", color=PALETTE["opt"], edgecolor="black")
    ax.bar(x, df_q2["Inter-Node Traffic (GB)"], 0.55, bottom=df_q2["Intra-Node Traffic (GB)"],
           label="InfiniBand (50 GB/s)", color=PALETTE["act"], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(topos, fontweight="bold")
    ax.set_ylabel("GB / GPU / step")
    ax.set_title("Q2 · Where the bytes travel")
    ax.legend(fontsize=8)
    axb.bar(topos, df_q2["Step Time (ms)"], 0.55, color=PALETTE["zero2"], edgecolor="black")
    axb.set_ylabel("Step time (ms)")
    axb.set_title("Q2 · Step time vs topology")
    _label_bars(axb, list(df_q2["Step Time (ms)"]), "{:.0f} ms")
    _save(fig, os.path.join(output_dir, "question2_topologies_nvlink_ib.png"))

    q3 = FourQuestionsSettler.settle_question_3(num_steps=12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.1))
    steps = range(1, len(q3["bf16_losses"]) + 1)
    ax1.plot(steps, q3["bf16_losses"], "o-", color=PALETTE["zero2"], linewidth=2.3, label="BF16")
    ax1.plot(steps, q3["mxfp8_losses"], "s--", color=PALETTE["zero3"], linewidth=2.3, label="MXFP8 (block scaled)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-entropy")
    ax1.set_title("Q3 · Same cyclic next-token task")
    ax1.legend()
    prec = ["BF16", "MXFP8"]
    times = list(q3["table"]["Total Step Time (ms)"])
    ax2.bar(prec, times, color=[PALETTE["zero2"], PALETTE["zero3"]], width=0.5, edgecolor="black")
    ax2.set_ylabel("Step time on B200 (ms)")
    ax2.set_title("Q3 · 20B ZeRO-2 step time")
    _label_bars(ax2, times, "{:.1f} ms")
    _save(fig, os.path.join(output_dir, "question3_bf16_vs_mxfp8.png"))

    q4 = FourQuestionsSettler.settle_question_4()
    df_q4 = q4["table"]
    fig, (axm, axt) = plt.subplots(1, 2, figsize=(12.8, 5.1))
    modes = ["GPU HBM\nno offload", "ZeRO-Offload\nCPU over PCIe"]
    mems = list(df_q4["GPU Memory Used (GB)"])
    axm.bar(modes, mems, color=[PALETTE["opt"], PALETTE["act"]], width=0.5, edgecolor="black")
    axm.axhline(80, color="black", linestyle="--", label="80 GB HBM")
    axm.set_ylabel("GB / GPU")
    axm.set_title("Q4 · Memory is not the limiter")
    axm.set_ylim(0, 95)
    axm.legend()
    for i, v in enumerate(mems):
        axm.text(i, v + 2, f"{v:.1f} GB", ha="center", fontweight="bold")
    tputs = list(df_q4["Throughput (Tokens/s)"])
    axt.bar(modes, tputs, color=[PALETTE["opt"], PALETTE["act"]], width=0.5, edgecolor="black")
    axt.set_ylabel("Tokens / sec")
    axt.set_title("Q4 · Offload still costs PCIe time")
    ymax = max(tputs)
    axt.set_ylim(0, ymax * 1.22)
    for i, tp in enumerate(tputs):
        axt.text(i, tp + ymax * 0.03, f"{tp:,.0f}\n({df_q4['Slowdown Penalty'].iloc[i]})",
                 ha="center", fontweight="bold", fontsize=9)
    _save(fig, os.path.join(output_dir, "question4_zero_offload_penalty.png"))
    print("[Saved] All Question artifact plots.")
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}


def main():
    print("=" * 72)
    print("  ZeRO 32-virtual-GPU simulation")
    print("=" * 72)
    target = MODEL_PRESETS["llm_20b"]
    psi = target.total_parameters
    print(f"Target: {target.name}  ({psi/1e9:.2f}B params)")
    plot_stage_memory_breakdown(target)
    plot_memory_scaling_sweep(target)
    plot_memory_and_compute_changes(target)
    plot_training_steps_logged()
    plot_section5_evolution()
    results = plot_four_questions_artifacts()
    print("\n--- Q1 ---\n", results["q1"]["verdict"])
    print("\n--- Q2 ---\n", results["q2"]["verdict"])
    print("\n--- Q3 ---\n", results["q3"]["verdict"])
    print("\n--- Q4 ---\n", results["q4"]["verdict"])
    print("\nPlots written to plots/")


if __name__ == "__main__":
    main()
