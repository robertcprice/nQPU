use nqpu_metal::ascii_viz::apply_gate_to_state;
use nqpu_metal::concurrent_uma::{
    ConcurrentUmaExecutor, GpuExecutionBackend, PartitionStrategy,
};
use nqpu_metal::gates::Gate;
use nqpu_metal::QuantumState;
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn build_circuit(num_qubits: usize, depth: usize) -> Vec<Gate> {
    // Leave one qubit untouched so chunk-parallel mode can legally split state
    // along that axis and execute halves concurrently.
    let active_qubits = if num_qubits > 2 {
        num_qubits - 1
    } else {
        num_qubits
    };
    let split = active_qubits / 2;
    let mut gates = Vec::new();
    for layer in 0..depth {
        for q in 0..active_qubits {
            gates.push(Gate::h(q));
            gates.push(Gate::rz(q, 0.01 * (1 + layer) as f64));
        }

        // Use two independent interaction blocks to stress disjoint-subset
        // concurrency in the executor benchmark.
        for (start, end) in [(0usize, split), (split, active_qubits)] {
            if end.saturating_sub(start) < 2 {
                continue;
            }
            for q in (start..end.saturating_sub(1)).step_by(2) {
                gates.push(Gate::cnot(q, q + 1));
            }
            for q in (start + 1..end.saturating_sub(1)).step_by(2) {
                gates.push(Gate::cnot(q, q + 1));
            }
        }
    }
    gates
}

fn arg_value(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find_map(|w| if w[0] == key { Some(w[1].clone()) } else { None })
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = arg_value(&args, "--mode", "concurrent");
    let num_qubits: usize = arg_value(&args, "--qubits", "14")
        .parse()
        .expect("invalid --qubits");
    let depth: usize = arg_value(&args, "--depth", "96")
        .parse()
        .expect("invalid --depth");
    let iters: usize = arg_value(&args, "--iters", "8")
        .parse()
        .expect("invalid --iters");

    let gates = build_circuit(num_qubits, depth);
    let executor_concurrent = ConcurrentUmaExecutor::new()
        .with_strategy(PartitionStrategy::Greedy)
        .with_min_concurrent_gates(2)
        .with_gpu_backend(GpuExecutionBackend::CpuEmulation);
    let executor_sequential = ConcurrentUmaExecutor::new()
        .with_strategy(PartitionStrategy::Greedy)
        .with_min_concurrent_gates(usize::MAX)
        .with_gpu_backend(GpuExecutionBackend::CpuEmulation);

    let t0 = Instant::now();
    let mut checksum = 0.0f64;
    let mut total_wall_ns = 0.0f64;
    let mut total_cpu_ns = 0.0f64;
    let mut total_gpu_ns = 0.0f64;
    let mut total_overlap_ns = 0.0f64;
    let mut total_speedup = 0.0f64;
    for _ in 0..iters {
        let mut state = QuantumState::new(num_qubits);
        if mode == "concurrent" {
            let stats = executor_concurrent
                .execute(&gates, &mut state)
                .expect("concurrent execution failed");
            checksum += stats.total_wall_time_ns;
            total_wall_ns += stats.total_wall_time_ns;
            total_cpu_ns += stats.total_cpu_time_ns;
            total_gpu_ns += stats.total_gpu_time_ns;
            total_overlap_ns += stats.total_overlap_ns;
            total_speedup += stats.concurrency_speedup;
        } else if mode == "executor_sequential" {
            let stats = executor_sequential
                .execute(&gates, &mut state)
                .expect("executor sequential execution failed");
            checksum += stats.total_wall_time_ns;
            total_wall_ns += stats.total_wall_time_ns;
            total_cpu_ns += stats.total_cpu_time_ns;
            total_gpu_ns += stats.total_gpu_time_ns;
            total_overlap_ns += stats.total_overlap_ns;
            total_speedup += stats.concurrency_speedup;
        } else if mode == "sequential" {
            for gate in &gates {
                apply_gate_to_state(&mut state, gate);
            }
            checksum += state.probabilities()[0];
        } else {
            panic!(
                "unsupported --mode: {} (expected concurrent|executor_sequential|sequential)",
                mode
            );
        }
        checksum += state.probabilities()[0];
    }
    black_box(checksum);

    let elapsed = t0.elapsed().as_secs_f64();
    let overlap_ratio = if total_cpu_ns + total_gpu_ns > 0.0 {
        (total_overlap_ns / (total_cpu_ns + total_gpu_ns)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mean_speedup = if iters > 0 {
        total_speedup / iters as f64
    } else {
        0.0
    };
    println!(
        "evidence_b2_speed mode={} qubits={} depth={} iters={} elapsed_seconds={:.6} \
         total_wall_ns={:.0} total_cpu_ns={:.0} total_gpu_ns={:.0} total_overlap_ns={:.0} \
         overlap_ratio={:.4} mean_speedup={:.4}",
        mode,
        num_qubits,
        depth,
        iters,
        elapsed,
        total_wall_ns,
        total_cpu_ns,
        total_gpu_ns,
        total_overlap_ns,
        overlap_ratio,
        mean_speedup,
    );
}
