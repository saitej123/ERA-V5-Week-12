"""
Unit tests for ZeRO Parallelism Simulation.
Tests 32 Virtual GPUs, collective communication, memory calculations, ZeRO stages, and four questions settlement.
"""

import pytest
import torch
from zero_simulation import (
    ClusterConfig,
    HardwareProfile,
    HARDWARE_PROFILES,
    ModelConfig,
    MODEL_PRESETS,
    ZeROConfig,
    VirtualCluster,
    Communicator,
    DemoTransformerModel,
    calculate_model_memory_breakdown,
    calculate_step_compute_flops,
    ZeRO0_Engine,
    ZeRO1_Engine,
    ZeRO2_Engine,
    ZeRO3_Engine,
    MemoryProfiler,
    CommunicationProfiler,
    FourQuestionsSettler,
)


def test_virtual_gpu_cluster_initialization():
    cluster_cfg = ClusterConfig(world_size=32, gpus_per_node=8)
    hw = HARDWARE_PROFILES["H100_SXM5"]
    cluster = VirtualCluster(cluster_cfg, hw)
    
    assert cluster.world_size == 32
    assert len(cluster.gpus) == 32
    assert cluster.cluster_cfg.num_nodes == 4
    
    # Check node assignments
    assert cluster.gpus[0].node_id == 0
    assert cluster.gpus[7].node_id == 0
    assert cluster.gpus[8].node_id == 1
    assert cluster.gpus[31].node_id == 3


def test_collectives_accuracy():
    cluster = VirtualCluster(ClusterConfig(world_size=4, gpus_per_node=4))
    comm = Communicator(cluster)
    
    # 1. AllReduce
    tensors = [torch.tensor([float(i)]) for i in range(4)]
    reduced, time_ms = comm.all_reduce(tensors, op="sum")
    expected_sum = sum([float(i) for i in range(4)])
    for r in reduced:
        assert torch.allclose(r, torch.tensor([expected_sum]))
        
    # 2. ReduceScatter & AllGather
    full_tensors = [torch.tensor([float(i), float(i*2), float(i*3), float(i*4)]) for i in range(4)]
    shards, rs_ms = comm.reduce_scatter(full_tensors)
    assert len(shards) == 4
    
    gathered, ag_ms = comm.all_gather(shards)
    assert len(gathered) == 4
    assert torch.allclose(gathered[0], full_tensors[0] + full_tensors[1] + full_tensors[2] + full_tensors[3])


def test_model_memory_breakdown_theoretical():
    model = MODEL_PRESETS["demo_medium"]
    
    # ZeRO-0 vs ZeRO-1 vs ZeRO-2 vs ZeRO-3 for 32 GPUs
    m0 = calculate_model_memory_breakdown(model, zero_stage=0, world_size=32, precision="bf16")
    m1 = calculate_model_memory_breakdown(model, zero_stage=1, world_size=32, precision="bf16")
    m2 = calculate_model_memory_breakdown(model, zero_stage=2, world_size=32, precision="bf16")
    m3 = calculate_model_memory_breakdown(model, zero_stage=3, world_size=32, precision="bf16")
    
    # Parameters and Gradients
    assert m0["params_bytes"] == m1["params_bytes"] == m2["params_bytes"]
    assert m3["params_bytes"] == m0["params_bytes"] / 32
    
    # Optimizer state: partitioned in ZeRO-1, 2, 3
    assert m1["optimizer_bytes"] == m0["optimizer_bytes"] / 32
    assert m2["optimizer_bytes"] == m0["optimizer_bytes"] / 32
    assert m3["optimizer_bytes"] == m0["optimizer_bytes"] / 32
    
    # Gradients: partitioned in ZeRO-2, 3
    assert m2["grads_bytes"] == m0["grads_bytes"] / 32
    assert m3["grads_bytes"] == m0["grads_bytes"] / 32
    
    # Total memory should strictly decrease from stage 0 to stage 3
    assert m0["total_bytes"] > m1["total_bytes"] > m2["total_bytes"] > m3["total_bytes"]


def test_zero_engines_train_step():
    model_cfg = MODEL_PRESETS["demo_small"]
    cluster_cfg = ClusterConfig(world_size=4, gpus_per_node=4)
    hw = HARDWARE_PROFILES["VIRTUAL_CPU"]
    zero_cfg = ZeROConfig(stage=0, precision="bf16", overlap_comm=True)
    
    batch_data = torch.randint(0, model_cfg.vocab_size, (model_cfg.micro_batch_size, model_cfg.seq_len))
    targets = torch.randint(0, model_cfg.vocab_size, (model_cfg.micro_batch_size, model_cfg.seq_len))
    
    # Test ZeRO-0
    engine0 = ZeRO0_Engine(model_cfg, cluster_cfg, hw, zero_cfg)
    res0 = engine0.train_step(0, batch_data, targets)
    assert res0["loss"] > 0
    assert res0["compute_time_ms"] > 0
    assert res0["comm_time_ms"] > 0
    
    # Test ZeRO-1
    zero_cfg1 = ZeROConfig(stage=1, precision="bf16", overlap_comm=True)
    engine1 = ZeRO1_Engine(model_cfg, cluster_cfg, hw, zero_cfg1)
    res1 = engine1.train_step(0, batch_data, targets)
    assert res1["loss"] > 0
    
    # Test ZeRO-2
    zero_cfg2 = ZeROConfig(stage=2, precision="bf16", overlap_comm=True)
    engine2 = ZeRO2_Engine(model_cfg, cluster_cfg, hw, zero_cfg2)
    res2 = engine2.train_step(0, batch_data, targets)
    assert res2["loss"] > 0
    
    # Test ZeRO-3
    zero_cfg3 = ZeROConfig(stage=3, precision="bf16", overlap_comm=True)
    engine3 = ZeRO3_Engine(model_cfg, cluster_cfg, hw, zero_cfg3)
    res3 = engine3.train_step(0, batch_data, targets)
    assert res3["loss"] > 0


def test_four_questions_settlement():
    # Question 1
    q1_res = FourQuestionsSettler.settle_question_1()
    assert "ZeRO-2 on 32 GPUs" in q1_res["verdict"]
    assert q1_res["table"]["Fits 80GB HBM"].iloc[0] == "YES"
    assert q1_res["table"]["Fits 80GB HBM"].iloc[1] == "YES"
    
    # Question 2
    q2_res = FourQuestionsSettler.settle_question_2()
    assert len(q2_res["table"]) == 3
    assert "Maximize the number of GPUs per NVLink node" in q2_res["verdict"]
    
    # Question 3
    q3_res = FourQuestionsSettler.settle_question_3(num_steps=3)
    assert len(q3_res["bf16_losses"]) == 3
    assert len(q3_res["mxfp8_losses"]) == 3
    assert "Commit to MXFP8" in q3_res["verdict"]
    
    # Question 4
    q4_res = FourQuestionsSettler.settle_question_4()
    assert "NO state should go to system memory" in q4_res["verdict"]
