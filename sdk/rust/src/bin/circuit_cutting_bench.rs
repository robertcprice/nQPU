use nqpu_metal::ascii_viz::apply_gate_to_state;
use nqpu_metal::circuit_cutting::{
    count_cut_kinds, execute_cut_plan_z_product, execute_wire_cut_plan_z_quasiprobability,
    plan_cuts_auto, plan_wire_cuts_auto, AutoCutConfig,
};
use nqpu_metal::gates::Gate;
use nqpu_metal::QuantumState;
use std::env;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CircuitPattern {
    MirrorPairs,
    ChainRing,
}

fn arg_value(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find_map(|w| if w[0] == key { Some(w[1].clone()) } else { None })
        .unwrap_or_else(|| default.to_string())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_pattern(name: &str) -> CircuitPattern {
    match name {
        "mirror_pairs" => CircuitPattern::MirrorPairs,
        "chain_ring" => CircuitPattern::ChainRing,
        _ => CircuitPattern::MirrorPairs,
    }
}

fn build_circuit(num_qubits: usize, depth: usize, pattern: CircuitPattern) -> Vec<Gate> {
    let mut gates = Vec::new();
    for layer in 0..depth {
        for q in 0..num_qubits {
            gates.push(Gate::h(q));
            gates.push(Gate::rz(q, 0.01 * (1 + layer) as f64));
        }

        match pattern {
            CircuitPattern::MirrorPairs => {
                // Near-neighbor entanglers.
                for q in (0..num_qubits.saturating_sub(1)).step_by(2) {
                    gates.push(Gate::cnot(q, q + 1));
                }
                // Long-range entanglers to stress cutting.
                for q in 0..(num_qubits / 2) {
                    let t = num_qubits - 1 - q;
                    if q != t {
                        gates.push(Gate::cnot(q, t));
                    }
                }
            }
            CircuitPattern::ChainRing => {
                // Chain + ring closure strongly couples the full register.
                for q in 0..num_qubits.saturating_sub(1) {
                    gates.push(Gate::cnot(q, q + 1));
                }
                if num_qubits >= 3 {
                    gates.push(Gate::cnot(0, num_qubits - 1));
                } else if num_qubits == 2 {
                    gates.push(Gate::cnot(0, 1));
                }
            }
        }
    }
    gates
}

fn execute_full_z(gates: &[Gate], num_qubits: usize) -> f64 {
    let mut state = QuantumState::new(num_qubits);
    for g in gates {
        apply_gate_to_state(&mut state, g);
    }
    state.expectation_z(0)
}

fn max_fragment_width(plan_fragments: &[nqpu_metal::circuit_cutting::CircuitFragment]) -> usize {
    plan_fragments
        .iter()
        .map(|f| f.local_qubits.len())
        .max()
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_qubits: usize = arg_value(&args, "--qubits", "16")
        .parse()
        .expect("invalid --qubits");
    let depth: usize = arg_value(&args, "--depth", "48")
        .parse()
        .expect("invalid --depth");
    let iters: usize = arg_value(&args, "--iters", "4")
        .parse()
        .expect("invalid --iters");
    let max_fragment_qubits: usize = arg_value(&args, "--max-frag", "8")
        .parse()
        .expect("invalid --max-frag");
    let lookahead: usize = arg_value(&args, "--lookahead", "16")
        .parse()
        .expect("invalid --lookahead");
    let pattern_name = arg_value(&args, "--pattern", "mirror_pairs");
    let pattern = parse_pattern(&pattern_name);
    let skip_full = has_flag(&args, "--skip-full");

    let gates = build_circuit(num_qubits, depth, pattern);

    let gate_cfg = AutoCutConfig {
        max_fragment_qubits,
        lookahead_gates: lookahead.max(1),
        prefer_wire_cuts: false,
        wire_cut_min_future_uses: usize::MAX,
    };

    let t_plan_gate = Instant::now();
    let gate_plan = plan_cuts_auto(&gates, &gate_cfg);
    let plan_gate_s = t_plan_gate.elapsed().as_secs_f64();

    let t_plan_wire = Instant::now();
    let wire_plan = plan_wire_cuts_auto(&gates, max_fragment_qubits, lookahead);
    let plan_wire_s = t_plan_wire.elapsed().as_secs_f64();

    let (gate_gate_cuts, gate_wire_cuts) = count_cut_kinds(&gate_plan);
    let (wire_gate_cuts, wire_wire_cuts) = count_cut_kinds(&wire_plan);

    let mut checksum = 0.0f64;
    let mut full_time_s = 0.0f64;
    let mut gate_time_s = 0.0f64;
    let mut wire_time_s = 0.0f64;

    for _ in 0..iters {
        if !skip_full {
            let t0 = Instant::now();
            let z = execute_full_z(&gates, num_qubits);
            full_time_s += t0.elapsed().as_secs_f64();
            checksum += z;
        }

        let t1 = Instant::now();
        let z_gate = execute_cut_plan_z_product(&gate_plan).expect("gate-cut exec failed");
        gate_time_s += t1.elapsed().as_secs_f64();
        checksum += z_gate;

        let t2 = Instant::now();
        let z_wire =
            execute_wire_cut_plan_z_quasiprobability(&wire_plan).expect("wire-cut exec failed");
        wire_time_s += t2.elapsed().as_secs_f64();
        checksum += z_wire;
    }

    black_box(checksum);

    let full_avg = if skip_full || iters == 0 {
        0.0
    } else {
        full_time_s / iters as f64
    };
    let gate_avg = if iters == 0 { 0.0 } else { gate_time_s / iters as f64 };
    let wire_avg = if iters == 0 { 0.0 } else { wire_time_s / iters as f64 };
    let gate_vs_wire = if wire_avg > 0.0 { gate_avg / wire_avg } else { 0.0 };
    let full_vs_wire = if !skip_full && wire_avg > 0.0 {
        full_avg / wire_avg
    } else {
        0.0
    };

    println!(
        "circuit_cutting_bench qubits={} depth={} iters={} max_frag={} lookahead={} \
         pattern={} \
         plan_gate_s={:.6} plan_wire_s={:.6} \
         gate_fragments={} gate_gate_cuts={} gate_wire_cuts={} gate_max_frag_width={} \
         wire_fragments={} wire_gate_cuts={} wire_wire_cuts={} wire_max_frag_width={} \
         avg_full_s={:.6} avg_gatecut_s={:.6} avg_wireqp_s={:.6} \
         speedup_gate_over_wire={:.4} speedup_full_over_wire={:.4}",
        num_qubits,
        depth,
        iters,
        max_fragment_qubits,
        lookahead,
        pattern_name,
        plan_gate_s,
        plan_wire_s,
        gate_plan.fragments.len(),
        gate_gate_cuts,
        gate_wire_cuts,
        max_fragment_width(&gate_plan.fragments),
        wire_plan.fragments.len(),
        wire_gate_cuts,
        wire_wire_cuts,
        max_fragment_width(&wire_plan.fragments),
        full_avg,
        gate_avg,
        wire_avg,
        gate_vs_wire,
        full_vs_wire,
    );
}
