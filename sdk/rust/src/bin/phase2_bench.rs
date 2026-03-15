use ndarray::{Array1, Array2};
use nqpu_metal::adjoint_diff::{AdjointCircuit, AdjointOp, Observable};
use nqpu_metal::auto_simulator::{AutoSimulator, SimBackend};
use nqpu_metal::circuit_cutting::execute_cut_circuit_z;
use nqpu_metal::cv_quantum::{CvGaussianState, GaussianBosonSampler};
use nqpu_metal::distributed_adjoint::{
    distributed_gradient, distributed_parameter_shift_gradient, DistributedAdjointConfig,
    DistributedGradientMethod,
};
use nqpu_metal::dynamic_surface_code::{DynamicSurfaceCode, RlDecoder};
use nqpu_metal::enhanced_zne::{EnhancedZne, ExtrapolationModel, FoldingStrategy};
use nqpu_metal::f32_fusion::F32FusionExecutor;
use nqpu_metal::gates::Gate;
use nqpu_metal::heisenberg_qpe::{
    estimate_phase_heisenberg, HeisenbergQpeConfig, IdealPhaseOracle,
};
use nqpu_metal::mid_circuit::{
    ClassicalCondition, Operation, QuantumStateWithMeasurements, ShotBranchingConfig,
};
use nqpu_metal::pulse_level::{
    state_fidelity, GrapeConfig, Pulse, PulseHamiltonian, PulseShape, PulseSimulator,
};
use nqpu_metal::topological_quantum::{FibonacciAnyonState, StringNetPlaquette};
use nqpu_metal::{
    CommunicationCostModel, DistributedMetalConfig, DistributedMetalWorldExecutor,
    ShardRemoteExecutionMode,
};
use num_complex::Complex64;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Debug)]
struct BenchStats {
    label: String,
    runs: usize,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench<F>(label: &str, runs: usize, mut f: F) -> BenchStats
where
    F: FnMut(),
{
    // Warmup
    f();

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let mean_ms = samples.iter().sum::<f64>() / runs.max(1) as f64;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    BenchStats {
        label: label.to_string(),
        runs,
        mean_ms,
        p50_ms: percentile(&samples, 0.50),
        p95_ms: percentile(&samples, 0.95),
    }
}

fn print_stats(stats: &BenchStats, baseline_ms: Option<f64>) {
    if let Some(base) = baseline_ms {
        let speedup = if stats.mean_ms > 0.0 {
            base / stats.mean_ms
        } else {
            0.0
        };
        println!(
            "{:<34} mean={:>9.3} ms  p50={:>9.3}  p95={:>9.3}  runs={}  speedup={:>6.2}x",
            stats.label, stats.mean_ms, stats.p50_ms, stats.p95_ms, stats.runs, speedup
        );
    } else {
        println!(
            "{:<34} mean={:>9.3} ms  p50={:>9.3}  p95={:>9.3}  runs={}",
            stats.label, stats.mean_ms, stats.p50_ms, stats.p95_ms, stats.runs
        );
    }
}

fn build_mixed_circuit(num_qubits: usize, depth: usize) -> Vec<Gate> {
    let mut gates = Vec::new();

    for d in 0..depth {
        for q in 0..num_qubits {
            match (d + q) % 4 {
                0 => gates.push(Gate::h(q)),
                1 => gates.push(Gate::rx(q, 0.2 + 0.01 * d as f64)),
                2 => gates.push(Gate::rz(q, -0.15 + 0.02 * q as f64)),
                _ => gates.push(Gate::x(q)),
            }
        }

        for q in (0..num_qubits.saturating_sub(1)).step_by(2) {
            gates.push(Gate::cnot(q, q + 1));
        }
        for q in (1..num_qubits.saturating_sub(1)).step_by(2) {
            gates.push(Gate::cnot(q, q + 1));
        }
    }

    gates
}

fn max_abs_prob_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn benchmark_distributed_world(
    global_num_qubits: usize,
    world_size: usize,
    depth: usize,
    runs: usize,
) {
    let mut gates = Vec::new();
    let shard_qubit_lo = global_num_qubits.saturating_sub(2);
    let shard_qubit_hi = global_num_qubits.saturating_sub(1);
    for d in 0..depth {
        let q = d % (global_num_qubits.saturating_sub(2)).max(1);
        gates.push(Gate::h(q));
        gates.push(Gate::cnot(
            q,
            (q + 1) % (global_num_qubits.saturating_sub(2)).max(1),
        ));
        gates.push(Gate::h(shard_qubit_lo));
        gates.push(Gate::cz(q, shard_qubit_hi));
        gates.push(Gate::cnot(shard_qubit_hi, q));
    }

    println!(
        "\n=== Distributed World Benchmark ({}q, world_size={}, depth={}, gates={}) ===",
        global_num_qubits,
        world_size,
        depth,
        gates.len()
    );

    let ref_sim = AutoSimulator::with_backend(SimBackend::StateVectorFused, global_num_qubits);
    let ref_once = ref_sim.execute(&gates);
    let ref_stats = bench("StateVector+Fusion (reference)", runs, || {
        let p = ref_sim.execute(&gates);
        black_box(p[0]);
    });
    print_stats(&ref_stats, None);

    let mut world = DistributedMetalWorldExecutor::new(
        global_num_qubits,
        world_size,
        DistributedMetalConfig::default(),
    )
    .expect("distributed world executor");
    let world_once = world
        .execute_partitioned(&gates)
        .expect("distributed world run");
    println!(
        "{:<34} max|dP| vs ref = {:.3e}",
        "DistributedWorld (accuracy)",
        max_abs_prob_diff(&ref_once, &world_once.global_probabilities)
    );
    println!(
        "{:<34} remote={}, exch_required={}, pairwise_fast={}, global_fallback={}, fallback_batches={}, comm_events={}, sched_local_batch={}, sched_fallback_batch={}, sched_cost={:.3}",
        "DistributedWorld (metrics)",
        world_once.metrics.remote_gates,
        world_once.metrics.remote_gates_exchange_required,
        world_once.metrics.remote_gates_pairwise_fast_path,
        world_once.metrics.remote_gates_global_fallback,
        world_once.metrics.remote_gates_global_fallback_batches,
        world_once.metrics.communication_events,
        world_once.metrics.adaptive_local_batch_limit,
        world_once.metrics.adaptive_global_fallback_batch_limit,
        world_once.metrics.scheduler_estimated_comm_cost
    );

    let world_stats = bench("DistributedWorld (mixed CPU/GPU)", runs, || {
        let out = world
            .execute_partitioned(&gates)
            .expect("distributed world run");
        black_box(out.global_probabilities[0]);
    });
    print_stats(&world_stats, Some(ref_stats.mean_ms));

    #[cfg(target_os = "macos")]
    {
        let mut strict_world = match DistributedMetalWorldExecutor::new(
            global_num_qubits,
            world_size,
            DistributedMetalConfig {
                strict_gpu_only: true,
                fail_on_remote_gates: false,
                remote_execution_mode: ShardRemoteExecutionMode::Skip,
                max_local_batch: 256,
            },
        ) {
            Ok(exec) => exec,
            Err(err) => {
                println!(
                    "{:<34} unavailable: {}",
                    "DistributedWorld (strict GPU)", err
                );
                return;
            }
        };

        let strict_once = strict_world
            .execute_partitioned(&gates)
            .expect("strict distributed world run");
        println!(
            "{:<34} max|dP| vs ref = {:.3e}",
            "DistributedWorld strict (accuracy)",
            max_abs_prob_diff(&ref_once, &strict_once.global_probabilities)
        );
        let strict_stats = bench("DistributedWorld (strict GPU)", runs, || {
            let out = strict_world
                .execute_partitioned(&gates)
                .expect("strict distributed world run");
            black_box(out.global_probabilities[0]);
        });
        print_stats(&strict_stats, Some(ref_stats.mean_ms));
    }
}

fn benchmark_distributed_fallback_batching(
    global_num_qubits: usize,
    world_size: usize,
    runs: usize,
) {
    let shard_qubit_hi = global_num_qubits.saturating_sub(1);
    let local_qubit = 0usize;

    let mut batched_gates = Vec::new();
    for _ in 0..12 {
        batched_gates.push(Gate::swap(local_qubit, shard_qubit_hi));
    }

    let mut split_gates = Vec::new();
    for i in 0..12 {
        split_gates.push(Gate::swap(local_qubit, shard_qubit_hi));
        split_gates.push(Gate::phase(0, 0.001 * (i as f64 + 1.0)));
    }

    println!(
        "\n=== Distributed Fallback Batching ({}q, world_size={}) ===",
        global_num_qubits, world_size
    );

    let ref_sim = AutoSimulator::with_backend(SimBackend::StateVectorFused, global_num_qubits);

    let mut world_batched = DistributedMetalWorldExecutor::new(
        global_num_qubits,
        world_size,
        DistributedMetalConfig::default(),
    )
    .expect("distributed world executor (batched)");
    let batched_once = world_batched
        .execute_partitioned(&batched_gates)
        .expect("batched distributed run");
    let batched_ref = ref_sim.execute(&batched_gates);
    println!(
        "{:<34} max|dP| vs ref = {:.3e}",
        "FallbackBatched (accuracy)",
        max_abs_prob_diff(&batched_ref, &batched_once.global_probabilities)
    );
    println!(
        "{:<34} remote={}, fallback={}, fallback_batches={}, comm_events={}, sched_fallback_batch={}",
        "FallbackBatched (metrics)",
        batched_once.metrics.remote_gates,
        batched_once.metrics.remote_gates_global_fallback,
        batched_once.metrics.remote_gates_global_fallback_batches,
        batched_once.metrics.communication_events,
        batched_once.metrics.adaptive_global_fallback_batch_limit
    );
    let batched_stats = bench("FallbackBatched (world)", runs, || {
        let out = world_batched
            .execute_partitioned(&batched_gates)
            .expect("batched distributed run");
        black_box(out.global_probabilities[0]);
    });
    print_stats(&batched_stats, None);

    let mut world_split = DistributedMetalWorldExecutor::new(
        global_num_qubits,
        world_size,
        DistributedMetalConfig::default(),
    )
    .expect("distributed world executor (split)");
    let split_once = world_split
        .execute_partitioned(&split_gates)
        .expect("split distributed run");
    let split_ref = ref_sim.execute(&split_gates);
    println!(
        "{:<34} max|dP| vs ref = {:.3e}",
        "FallbackSplit (accuracy)",
        max_abs_prob_diff(&split_ref, &split_once.global_probabilities)
    );
    println!(
        "{:<34} remote={}, fallback={}, fallback_batches={}, comm_events={}, sched_fallback_batch={}",
        "FallbackSplit (metrics)",
        split_once.metrics.remote_gates,
        split_once.metrics.remote_gates_global_fallback,
        split_once.metrics.remote_gates_global_fallback_batches,
        split_once.metrics.communication_events,
        split_once.metrics.adaptive_global_fallback_batch_limit
    );
    let split_stats = bench("FallbackSplit (world)", runs, || {
        let out = world_split
            .execute_partitioned(&split_gates)
            .expect("split distributed run");
        black_box(out.global_probabilities[0]);
    });
    print_stats(&split_stats, Some(batched_stats.mean_ms));

    #[cfg(target_os = "macos")]
    {
        let mut strict_world = match DistributedMetalWorldExecutor::new(
            global_num_qubits,
            world_size,
            DistributedMetalConfig {
                strict_gpu_only: true,
                fail_on_remote_gates: false,
                remote_execution_mode: ShardRemoteExecutionMode::Skip,
                max_local_batch: 256,
            },
        ) {
            Ok(exec) => exec,
            Err(err) => {
                println!("{:<34} unavailable: {}", "FallbackStrictGuard", err);
                return;
            }
        };

        let strict_once = strict_world
            .execute_partitioned(&batched_gates)
            .expect("strict fallback distributed run");
        println!(
            "{:<34} max|dP| vs ref = {:.3e}",
            "FallbackStrictGPU (accuracy)",
            max_abs_prob_diff(&batched_ref, &strict_once.global_probabilities)
        );
        println!(
            "{:<34} remote={}, fallback={}, fallback_batches={}, comm_events={}, sched_fallback_batch={}",
            "FallbackStrictGPU (metrics)",
            strict_once.metrics.remote_gates,
            strict_once.metrics.remote_gates_global_fallback,
            strict_once.metrics.remote_gates_global_fallback_batches,
            strict_once.metrics.communication_events,
            strict_once.metrics.adaptive_global_fallback_batch_limit
        );
        let strict_stats = bench("FallbackStrictGPU (world)", runs, || {
            let out = strict_world
                .execute_partitioned(&batched_gates)
                .expect("strict fallback distributed run");
            black_box(out.global_probabilities[0]);
        });
        print_stats(&strict_stats, Some(batched_stats.mean_ms));
    }
}

fn benchmark_distributed_pairwise_shard_shard(
    global_num_qubits: usize,
    world_size: usize,
    runs: usize,
) {
    let shard_qubit_lo = global_num_qubits.saturating_sub(2);
    let shard_qubit_hi = global_num_qubits.saturating_sub(1);
    let gates = vec![
        Gate::h(shard_qubit_lo),
        Gate::h(shard_qubit_hi),
        Gate::swap(shard_qubit_lo, shard_qubit_hi),
        Gate::iswap(shard_qubit_lo, shard_qubit_hi),
        Gate::h(shard_qubit_lo),
        Gate::h(shard_qubit_hi),
    ];

    println!(
        "\n=== Distributed Pairwise Shard-Shard ({}q, world_size={}) ===",
        global_num_qubits, world_size
    );

    let ref_sim = AutoSimulator::with_backend(SimBackend::StateVectorFused, global_num_qubits);
    let ref_once = ref_sim.execute(&gates);
    let ref_stats = bench("PairwiseShardShard (reference)", runs, || {
        let p = ref_sim.execute(&gates);
        black_box(p[0]);
    });
    print_stats(&ref_stats, None);

    let mut world = DistributedMetalWorldExecutor::new(
        global_num_qubits,
        world_size,
        DistributedMetalConfig::default(),
    )
    .expect("distributed world executor");
    let world_once = world.execute_partitioned(&gates).expect("distributed run");
    println!(
        "{:<34} max|dP| vs ref = {:.3e}",
        "PairwiseShardShard (accuracy)",
        max_abs_prob_diff(&ref_once, &world_once.global_probabilities)
    );
    println!(
        "{:<34} remote={}, pairwise_fast={}, global_fallback={}, fallback_batches={}, comm_events={}",
        "PairwiseShardShard (metrics)",
        world_once.metrics.remote_gates,
        world_once.metrics.remote_gates_pairwise_fast_path,
        world_once.metrics.remote_gates_global_fallback,
        world_once.metrics.remote_gates_global_fallback_batches,
        world_once.metrics.communication_events
    );

    let world_stats = bench("PairwiseShardShard (world)", runs, || {
        let out = world.execute_partitioned(&gates).expect("distributed run");
        black_box(out.global_probabilities[0]);
    });
    print_stats(&world_stats, Some(ref_stats.mean_ms));
}

fn benchmark_backends(num_qubits: usize, depth: usize, runs: usize) {
    let gates = build_mixed_circuit(num_qubits, depth);
    println!(
        "\n=== Backend Benchmark ({} qubits, depth {}, {} gates) ===",
        num_qubits,
        depth,
        gates.len()
    );

    let cpu_sim = AutoSimulator::with_backend(SimBackend::StateVector, num_qubits);
    let cpu_ref = cpu_sim.execute(&gates);
    let cpu_stats = bench("StateVector (CPU)", runs, || {
        let probs = cpu_sim.execute(&gates);
        black_box(probs[0]);
    });
    print_stats(&cpu_stats, None);

    let fused_sim = AutoSimulator::with_backend(SimBackend::StateVectorFused, num_qubits);
    let fused_once = fused_sim.execute(&gates);
    println!(
        "{:<34} max|dP| vs CPU = {:.3e}",
        "StateVector+Fusion (accuracy)",
        max_abs_prob_diff(&cpu_ref, &fused_once)
    );
    let fused_stats = bench("StateVector+Fusion (CPU)", runs, || {
        let probs = fused_sim.execute(&gates);
        black_box(probs[0]);
    });
    print_stats(&fused_stats, Some(cpu_stats.mean_ms));

    let f32fused_sim = AutoSimulator::with_backend(SimBackend::StateVectorF32Fused, num_qubits);
    let f32_once = f32fused_sim.execute(&gates);
    println!(
        "{:<34} max|dP| vs CPU = {:.3e}",
        "StateVector+F32+Fusion (accuracy)",
        max_abs_prob_diff(&cpu_ref, &f32_once)
    );
    let f32fused_stats = bench("StateVector+F32+Fusion", runs, || {
        let probs = f32fused_sim.execute(&gates);
        black_box(probs[0]);
    });
    print_stats(&f32fused_stats, Some(cpu_stats.mean_ms));

    let auto_sim = AutoSimulator::new(&gates, num_qubits, false);
    println!(
        "{:<34} {:?}",
        "AutoSimulator selected backend:",
        auto_sim.backend()
    );
    let auto_stats = bench("AutoSimulator", runs, || {
        let probs = auto_sim.execute(&gates);
        black_box(probs[0]);
    });
    print_stats(&auto_stats, Some(cpu_stats.mean_ms));

    #[cfg(target_os = "macos")]
    {
        let gpu_sim = AutoSimulator::with_backend(SimBackend::MetalGPU, num_qubits);
        let gpu_once = gpu_sim.execute(&gates);
        let norm_err = (gpu_once.iter().sum::<f64>() - 1.0).abs();
        println!(
            "{:<34} max|dP| vs CPU = {:.3e}, norm_err={:.3e}",
            "MetalGPU (accuracy)",
            max_abs_prob_diff(&cpu_ref, &gpu_once),
            norm_err
        );
        let gpu_stats = bench("MetalGPU", runs, || {
            let probs = gpu_sim.execute(&gates);
            black_box(probs[0]);
        });
        print_stats(&gpu_stats, Some(cpu_stats.mean_ms));

        let gpu_only_sim = AutoSimulator::with_gpu_only(num_qubits);
        match gpu_only_sim.execute_result(&gates) {
            Ok(gpu_only_once) => {
                let norm_err = (gpu_only_once.iter().sum::<f64>() - 1.0).abs();
                println!(
                    "{:<34} max|dP| vs CPU = {:.3e}, norm_err={:.3e}",
                    "MetalGPUOnly (accuracy)",
                    max_abs_prob_diff(&cpu_ref, &gpu_only_once),
                    norm_err
                );
                let gpu_only_stats = bench("MetalGPUOnly (strict)", runs, || {
                    let probs = gpu_only_sim
                        .execute_result(&gates)
                        .expect("strict GPU-only execution");
                    black_box(probs[0]);
                });
                print_stats(&gpu_only_stats, Some(cpu_stats.mean_ms));
            }
            Err(err) => {
                println!("{:<34} unavailable: {}", "MetalGPUOnly (strict)", err);
            }
        }
    }
}

fn benchmark_new_modules() {
    println!("\n=== New Module Microbenchmarks ===");

    let base_gates = build_mixed_circuit(12, 12);

    let f32_exec = F32FusionExecutor::new();
    let f32_stats = bench("f32_fusion::execute", 5, || {
        let (_state, metrics) = f32_exec
            .execute(12, &base_gates)
            .expect("f32 fusion execute");
        black_box(metrics.estimated_speedup);
    });
    print_stats(&f32_stats, None);

    let oracle = IdealPhaseOracle {
        phase: 0.318_309_886,
        readout_error: 0.01,
    };
    let qpe_cfg = HeisenbergQpeConfig {
        rounds: 10,
        shots_per_round: 128,
        posterior_grid_size: 2048,
    };
    let qpe_stats = bench("heisenberg_qpe::estimate", 7, || {
        let r = estimate_phase_heisenberg(&oracle, &qpe_cfg);
        black_box(r.phase_estimate);
    });
    print_stats(&qpe_stats, None);

    let zne = EnhancedZne {
        odd_scales: vec![1, 3, 5],
        folding: FoldingStrategy::Global,
        model: ExtrapolationModel::Richardson,
    };
    let zne_stats = bench("enhanced_zne::run", 5, || {
        let (mit, _pts) = zne
            .run(&base_gates, |folded| {
                let sim = AutoSimulator::with_backend(SimBackend::StateVectorFused, 12);
                let probs = sim.execute(folded);
                Ok(probs[0])
            })
            .expect("zne run");
        black_box(mit);
    });
    print_stats(&zne_stats, None);

    let cv_stats = bench("cv_quantum + GBS", 6, || {
        let mut s = CvGaussianState::vacuum(8);
        for m in 0..8 {
            s.displace(m, 0.1 * (m as f64 + 1.0), -0.05 * m as f64);
            s.squeeze(m, 0.15 + 0.01 * m as f64, 0.1);
        }
        for m in 0..7 {
            s.beamsplitter(m, m + 1, std::f64::consts::FRAC_PI_6, 0.0);
        }
        let gbs = GaussianBosonSampler::new(s);
        let samples = gbs.sample_click_patterns(256);
        black_box(samples.len());
    });
    print_stats(&cv_stats, None);

    let cut_stats = bench("circuit_cutting::execute", 6, || {
        let v = execute_cut_circuit_z(&base_gates, 6).expect("cut execute");
        black_box(v);
    });
    print_stats(&cut_stats, None);

    let mut adj = AdjointCircuit::new(6);
    adj.add_op(AdjointOp::Ry { qubit: 0, param: 0 });
    adj.add_op(AdjointOp::Rx { qubit: 1, param: 1 });
    adj.add_op(AdjointOp::Fixed(Gate::cnot(0, 1)));
    adj.add_op(AdjointOp::Rz { qubit: 1, param: 2 });
    adj.add_op(AdjointOp::Fixed(Gate::cnot(1, 2)));
    adj.add_op(AdjointOp::Ry { qubit: 2, param: 3 });
    let params = [0.21, -0.34, 0.44, -0.18];
    let adj_stats = bench("adjoint_diff::gradient", 12, || {
        let g = adj
            .gradient(&params, Observable::PauliZ(2))
            .expect("adjoint gradient");
        black_box(g[0]);
    });
    print_stats(&adj_stats, None);

    let dist_adj_cfg = DistributedAdjointConfig {
        world_size: 4,
        cost_model: CommunicationCostModel::default(),
        method: DistributedGradientMethod::ParameterShift,
        ..DistributedAdjointConfig::default()
    };
    let dist_adj_stats = bench("distributed_adjoint::param_shift", 8, || {
        let out = distributed_parameter_shift_gradient(
            &adj,
            &params,
            Observable::PauliZ(2),
            &dist_adj_cfg,
        )
        .expect("distributed adjoint gradient");
        black_box((
            out.expectation,
            out.gradients[0],
            out.last_metrics.remote_gates,
        ));
    });
    print_stats(&dist_adj_stats, None);

    let dist_adj_auto_cfg = DistributedAdjointConfig {
        world_size: 1,
        cost_model: CommunicationCostModel::default(),
        method: DistributedGradientMethod::Auto,
        ..DistributedAdjointConfig::default()
    };
    let dist_adj_auto_stats = bench("distributed_adjoint::adjoint_auto", 8, || {
        let out = distributed_gradient(&adj, &params, Observable::PauliZ(2), &dist_adj_auto_cfg)
            .expect("distributed adjoint auto gradient");
        black_box((out.expectation, out.gradients[0], out.num_evaluations));
    });
    print_stats(&dist_adj_auto_stats, None);

    let dist_adj_mirror_cfg = DistributedAdjointConfig {
        world_size: 4,
        cost_model: CommunicationCostModel::default(),
        method: DistributedGradientMethod::AdjointMirror,
        ..DistributedAdjointConfig::default()
    };
    let dist_adj_mirror_stats = bench("distributed_adjoint::adjoint_mirror_w4", 8, || {
        let out = distributed_gradient(&adj, &params, Observable::PauliZ(2), &dist_adj_mirror_cfg)
            .expect("distributed adjoint mirror gradient");
        black_box((
            out.expectation,
            out.gradients[0],
            out.last_metrics.remote_gates,
        ));
    });
    print_stats(&dist_adj_mirror_stats, None);

    let topo_stats = bench("topological_quantum", 20, || {
        let mut s = FibonacciAnyonState::basis_zero();
        for _ in 0..200 {
            s.braid_word(&[1, 2, -1, -2]).expect("valid braid");
        }
        s.normalize();

        let mut p = StringNetPlaquette::new(12);
        for _ in 0..64 {
            p.apply_projector();
        }

        black_box((s.probabilities()[0], p.string_count()));
    });
    print_stats(&topo_stats, None);

    let dyn_stats = bench("dynamic_surface_code", 10, || {
        let mut code = DynamicSurfaceCode::new(7).with_error_rate(0.01);
        let mut decoder = RlDecoder::new();
        for _ in 0..128 {
            let r = code.run_cycle(&mut decoder);
            black_box(r.logical_error_rate);
        }
    });
    print_stats(&dyn_stats, None);

    let mid_ops = vec![
        Operation::h(0),
        Operation::measure(0, 0),
        Operation::conditional(Operation::x(1), ClassicalCondition::BitSet(0)),
        Operation::measure(1, 1),
        Operation::conditional(Operation::x(2), ClassicalCondition::all_bits_set(&[0, 1])),
        Operation::measure(2, 2),
    ];
    let mid_cfg = ShotBranchingConfig {
        max_branches: 512,
        prune_probability: 1e-12,
    };
    let mid_stats = bench("mid_circuit::shot_branching", 8, || {
        let sim = QuantumStateWithMeasurements::new(3);
        let out = sim
            .execute_shots_branching(&mid_ops, 2048, &mid_cfg)
            .expect("branching execute");
        black_box((out.counts.len(), out.pruned_branches));
    });
    print_stats(&mid_stats, None);

    let pulse_stats = bench("pulse_level::simulate+grape", 5, || {
        let mut xh = Array2::zeros((2, 2));
        xh[[0, 1]] = Complex64::new(0.5, 0.0);
        xh[[1, 0]] = Complex64::new(0.5, 0.0);

        let ham = PulseHamiltonian {
            drift: Array2::zeros((2, 2)),
            controls: vec![xh],
        };
        let mut sim = PulseSimulator::new(ham, 2e-3, 1.0);
        sim.add_pulse(Pulse {
            channel: 0,
            t0: 0.0,
            duration: 1.0,
            amplitude: std::f64::consts::PI,
            phase: 0.0,
            detuning: 0.0,
            shape: PulseShape::Square,
        });

        let mut psi0 = Array1::zeros(2);
        psi0[0] = Complex64::new(1.0, 0.0);

        let mut target = Array1::zeros(2);
        target[1] = Complex64::new(1.0, 0.0);

        let before = sim.simulate_state(&psi0);
        let _h_eff = sim.floquet_effective_hamiltonian(0.25, 32);
        sim.optimize_grape(
            &[0],
            &psi0,
            &target,
            &GrapeConfig {
                iterations: 4,
                learning_rate: 0.2,
                epsilon: 1e-3,
            },
        );
        let after = sim.simulate_state(&psi0);

        black_box((
            state_fidelity(&before, &target),
            state_fidelity(&after, &target),
        ));
    });
    print_stats(&pulse_stats, None);
}

fn main() {
    println!("=== nQPU-Metal Phase 2 Rust Benchmark ===");
    println!("All paths are pure Rust; GPU path uses Metal kernels on macOS.\n");

    benchmark_backends(10, 24, 7);
    benchmark_backends(14, 20, 5);
    benchmark_backends(18, 16, 3);
    benchmark_distributed_world(16, 4, 12, 4);
    benchmark_distributed_fallback_batching(16, 4, 4);
    benchmark_distributed_pairwise_shard_shard(16, 4, 4);

    benchmark_new_modules();

    println!("\nDone.");
}
