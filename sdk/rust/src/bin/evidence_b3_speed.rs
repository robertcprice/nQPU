use nqpu_metal::gates::Gate;
use nqpu_metal::mixed_precision::{
    F32ExecutionBackend, GateContext, MixedPrecisionSimulator, PrecisionPolicy,
};
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn build_circuit(num_qubits: usize, depth: usize) -> Vec<Gate> {
    let mut gates = Vec::new();
    for layer in 0..depth {
        for q in 0..num_qubits {
            gates.push(Gate::h(q));
            gates.push(Gate::rz(q, 0.03 * (1 + layer) as f64));
        }
        for q in (0..num_qubits.saturating_sub(1)).step_by(2) {
            gates.push(Gate::cnot(q, q + 1));
        }
    }
    gates
}

fn arg_value(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find_map(|w| {
            if w[0] == key {
                Some(w[1].clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = arg_value(&args, "--mode", "adaptive");
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
    let mut contexts = vec![GateContext::ForwardPass; gates.len()];
    if !contexts.is_empty() {
        let last = contexts.len() - 1;
        contexts[last] = GateContext::PreMeasurement;
    }

    let t0 = Instant::now();
    let mut checksum = 0.0f64;
    let mut total_promotions = 0usize;
    let mut total_demotions = 0usize;
    let mut total_promotion_time_ns = 0.0f64;
    let mut total_demotion_time_ns = 0.0f64;
    let mut total_f32_gate_time_ns = 0.0f64;
    let mut total_f64_gate_time_ns = 0.0f64;
    let mut total_exec_time_ns = 0.0f64;
    for _ in 0..iters {
        let policy = match mode.as_str() {
            "adaptive" => PrecisionPolicy::Adaptive,
            "f64" => PrecisionPolicy::AlwaysF64,
            _ => panic!("unsupported --mode: {} (expected adaptive|f64)", mode),
        };

        let mut sim = MixedPrecisionSimulator::new(num_qubits, policy)
            .with_f32_backend(F32ExecutionBackend::CpuFallback)
            .with_error_threshold(1e-5)
            .with_gate_contexts(contexts.clone());

        let stats = sim.execute_circuit(&gates);
        if mode == "adaptive" {
            // Ensure adaptive path actually exercised mixed precision.
            assert!(stats.f32_gates > 0, "adaptive mode produced zero f32 gates");
        }
        total_promotions += stats.promotions;
        total_demotions += stats.demotions;
        total_promotion_time_ns += stats.promotion_time_ns;
        total_demotion_time_ns += stats.demotion_time_ns;
        total_f32_gate_time_ns += stats.f32_gate_time_ns;
        total_f64_gate_time_ns += stats.f64_gate_time_ns;
        total_exec_time_ns += stats.total_exec_time_ns;
        checksum += stats.peak_error;
        checksum += sim.probabilities()[0];
    }
    black_box(checksum);

    let elapsed = t0.elapsed().as_secs_f64();
    let transition_overhead_ratio = if total_exec_time_ns > 0.0 {
        ((total_promotion_time_ns + total_demotion_time_ns) / total_exec_time_ns).clamp(0.0, 1.0)
    } else {
        0.0
    };
    println!(
        "evidence_b3_speed mode={} qubits={} depth={} iters={} elapsed_seconds={:.6} \
         promotions={} demotions={} promotion_time_ns={:.0} demotion_time_ns={:.0} \
         f32_gate_time_ns={:.0} f64_gate_time_ns={:.0} transition_overhead_ratio={:.4}",
        mode,
        num_qubits,
        depth,
        iters,
        elapsed,
        total_promotions,
        total_demotions,
        total_promotion_time_ns,
        total_demotion_time_ns,
        total_f32_gate_time_ns,
        total_f64_gate_time_ns,
        transition_overhead_ratio,
    );
}
