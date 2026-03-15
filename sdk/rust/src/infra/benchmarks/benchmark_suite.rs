//! Benchmark Suite
//!
//! Reproducible benchmarks with circuit generators and timing harness.
//! Validates fidelity between fused and unfused execution.
//!
//! # Circuit Generators
//! - `qft_circuit(n)` — Quantum Fourier Transform
//! - `random_circuit(n, depth, seed)` — Random single + two-qubit gates
//! - `grover_circuit(n, target)` — Grover's search
//! - `bell_circuit(n)` — Bell pair chain

use crate::ascii_viz::apply_gate_to_state;
use crate::gate_fusion::{execute_fused_circuit, fuse_gates};
use crate::gates::{Gate, GateType};
#[cfg(target_os = "macos")]
use crate::metal_backend::MetalSimulator;
use crate::QuantumState;
use std::time::Instant;

// ===================================================================
// CIRCUIT GENERATORS
// ===================================================================

/// Generate a QFT circuit on `n` qubits.
pub fn qft_circuit(n: usize) -> Vec<Gate> {
    let mut gates = Vec::new();
    for i in 0..n {
        gates.push(Gate::h(i));
        for j in (i + 1)..n {
            let k = j - i;
            let angle = std::f64::consts::PI / (1 << k) as f64;
            gates.push(Gate::new(GateType::CR(angle), vec![j], vec![i]));
        }
    }
    // Swap qubits to reverse order (standard QFT convention)
    for i in 0..n / 2 {
        gates.push(Gate::swap(i, n - 1 - i));
    }
    gates
}

/// Generate a random circuit with `n` qubits and given `depth`.
/// Uses a simple deterministic PRNG seeded by `seed`.
pub fn random_circuit(n: usize, depth: usize, seed: u64) -> Vec<Gate> {
    let mut gates = Vec::new();
    let mut rng_state = seed;

    let single_gates = [
        GateType::H,
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::S,
        GateType::T,
    ];

    for _ in 0..depth {
        // Apply random single-qubit gates to all qubits
        for q in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = ((rng_state >> 33) as usize) % single_gates.len();
            gates.push(Gate::single(single_gates[idx].clone(), q));
        }

        // Apply random CNOT layer
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let offset = ((rng_state >> 33) as usize) % 2;
        let mut q = offset;
        while q + 1 < n {
            gates.push(Gate::cnot(q, q + 1));
            q += 2;
        }
    }

    gates
}

/// Generate a Grover circuit for `n` qubits searching for `target`.
pub fn grover_circuit(n: usize, target: usize) -> Vec<Gate> {
    let mut gates = Vec::new();
    let num_iterations = ((std::f64::consts::PI / 4.0) * ((1 << n) as f64).sqrt()) as usize;
    let num_iterations = num_iterations.max(1);

    // Initial superposition
    for q in 0..n {
        gates.push(Gate::h(q));
    }

    for _ in 0..num_iterations {
        // Oracle: flip phase of target state
        // Decompose target into X gates to flip relevant qubits
        for q in 0..n {
            if target & (1 << q) == 0 {
                gates.push(Gate::x(q));
            }
        }
        // Multi-controlled Z (simplified as CZ chain for benchmarking purposes)
        if n >= 2 {
            for q in 0..n - 1 {
                gates.push(Gate::cz(q, q + 1));
            }
        }
        for q in 0..n {
            if target & (1 << q) == 0 {
                gates.push(Gate::x(q));
            }
        }

        // Diffusion operator: H^n * (2|0><0| - I) * H^n
        for q in 0..n {
            gates.push(Gate::h(q));
        }
        for q in 0..n {
            gates.push(Gate::x(q));
        }
        if n >= 2 {
            for q in 0..n - 1 {
                gates.push(Gate::cz(q, q + 1));
            }
        }
        for q in 0..n {
            gates.push(Gate::x(q));
        }
        for q in 0..n {
            gates.push(Gate::h(q));
        }
    }

    gates
}

/// Generate a Bell pair chain: H(0), CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1).
pub fn bell_circuit(n: usize) -> Vec<Gate> {
    let mut gates = Vec::new();
    gates.push(Gate::h(0));
    for q in 0..n.saturating_sub(1) {
        gates.push(Gate::cnot(q, q + 1));
    }
    gates
}

// ===================================================================
// BENCHMARK HARNESS
// ===================================================================

/// Result of a single benchmark run.
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub num_qubits: usize,
    pub num_gates: usize,
    pub sequential_time_ms: f64,
    pub fused_time_ms: f64,
    pub fusion_speedup: f64,
    pub gpu_time_ms: f64,
    pub gpu_speedup: f64,
    pub gpu_fidelity: f64,
    pub gates_eliminated: usize,
    pub fidelity: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.gpu_time_ms > 0.0 {
            write!(
                f,
                "{:<15} n={:<3} gates={:<5} cpu={:>8.2}ms fused={:>8.2}ms gpu={:>8.2}ms gpu_speedup={:>6.1}x fid={:.8}",
                self.name, self.num_qubits, self.num_gates,
                self.sequential_time_ms, self.fused_time_ms, self.gpu_time_ms,
                self.gpu_speedup, self.gpu_fidelity
            )
        } else {
            write!(
                f,
                "{:<15} n={:<3} gates={:<5} cpu={:>8.2}ms fused={:>8.2}ms gpu=N/A",
                self.name,
                self.num_qubits,
                self.num_gates,
                self.sequential_time_ms,
                self.fused_time_ms
            )
        }
    }
}

/// Run a benchmark on a circuit, comparing CPU sequential, CPU fused, and Metal GPU.
pub fn run_benchmark(name: &str, gates: &[Gate], num_qubits: usize) -> BenchmarkResult {
    // CPU sequential execution
    let start = Instant::now();
    let mut state_seq = QuantumState::new(num_qubits);
    for gate in gates {
        apply_gate_to_state(&mut state_seq, gate);
    }
    let sequential_time = start.elapsed().as_secs_f64() * 1000.0;

    // CPU fused execution
    let fusion = fuse_gates(gates);
    let start = Instant::now();
    let mut state_fused = QuantumState::new(num_qubits);
    execute_fused_circuit(&mut state_fused, &fusion);
    let fused_time = start.elapsed().as_secs_f64() * 1000.0;

    let fidelity = state_fused.fidelity(&state_seq);
    let fusion_speedup = if fused_time > 0.0 {
        sequential_time / fused_time
    } else {
        1.0
    };

    // Metal GPU execution
    let (gpu_time, gpu_speedup, gpu_fidelity) =
        run_gpu_timed(gates, num_qubits, &state_seq, sequential_time);

    BenchmarkResult {
        name: name.to_string(),
        num_qubits,
        num_gates: gates.len(),
        sequential_time_ms: sequential_time,
        fused_time_ms: fused_time,
        fusion_speedup,
        gpu_time_ms: gpu_time,
        gpu_speedup,
        gpu_fidelity,
        gates_eliminated: fusion.gates_eliminated,
        fidelity,
    }
}

/// Run GPU benchmark, returning (time_ms, speedup_vs_cpu, fidelity).
/// Returns (0, 0, 0) if GPU is unavailable.
///
/// S-Tier Optimization: Uses batched execution for circuits with > 5 gates
/// to reduce GPU dispatch overhead. Small circuits use direct execution to
/// avoid fusion overhead.
fn run_gpu_timed(
    gates: &[Gate],
    num_qubits: usize,
    cpu_state: &QuantumState,
    cpu_time_ms: f64,
) -> (f64, f64, f64) {
    #[cfg(target_os = "macos")]
    {
        match MetalSimulator::new(num_qubits) {
            Ok(mut sim) => {
                // Use batched execution for larger circuits (reduces GPU dispatch overhead)
                let use_batched = gates.len() > 5;

                // Warmup
                if use_batched {
                    sim.run_circuit_batched(gates);
                } else {
                    sim.run_circuit(gates);
                }
                sim.reset();

                // Timed run
                let start = Instant::now();
                if use_batched {
                    sim.run_circuit_batched(gates);
                } else {
                    sim.run_circuit(gates);
                }
                let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

                let fid = sim.fidelity_vs_cpu(cpu_state);
                let speedup = if gpu_time > 0.0 {
                    cpu_time_ms / gpu_time
                } else {
                    f64::INFINITY
                };
                return (gpu_time, speedup, fid);
            }
            Err(_) => {}
        }
    }
    (0.0, 0.0, 0.0)
}

/// Run the full benchmark suite.
pub fn run_full_suite() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // QFT benchmarks
    for &n in &[10, 15, 20] {
        let gates = qft_circuit(n);
        results.push(run_benchmark(&format!("QFT-{}", n), &gates, n));
    }

    // Random circuit benchmarks
    for &n in &[10, 15, 20] {
        let gates = random_circuit(n, 20, 42);
        results.push(run_benchmark(&format!("Random-{}", n), &gates, n));
    }

    // Grover benchmarks
    for &n in &[10, 14] {
        let target = (1 << n) / 2;
        let gates = grover_circuit(n, target);
        results.push(run_benchmark(&format!("Grover-{}", n), &gates, n));
    }

    // Bell chain benchmarks
    for &n in &[10, 20, 30] {
        let gates = bell_circuit(n);
        results.push(run_benchmark(&format!("Bell-{}", n), &gates, n));
    }

    results
}

/// Print formatted results table.
pub fn print_results(results: &[BenchmarkResult]) {
    println!("{}", "=".repeat(120));
    println!("nQPU-Metal BENCHMARK SUITE — CPU vs GPU");
    #[cfg(target_os = "macos")]
    {
        if let Ok(sim) = MetalSimulator::new(2) {
            println!("GPU Device: {}", sim.device_name());
        }
    }
    println!("{}", "=".repeat(120));
    println!(
        "{:<15} {:<5} {:<7} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "Benchmark", "n", "Gates", "CPU(ms)", "Fused(ms)", "GPU(ms)", "GPU Speedup", "GPU Fid"
    );
    println!("{}", "-".repeat(120));

    for r in results {
        println!("{}", r);
    }

    println!("{}", "-".repeat(120));

    // Summary
    let gpu_results: Vec<_> = results.iter().filter(|r| r.gpu_time_ms > 0.0).collect();
    if !gpu_results.is_empty() {
        let avg_gpu_speedup: f64 =
            gpu_results.iter().map(|r| r.gpu_speedup).sum::<f64>() / gpu_results.len() as f64;
        let min_gpu_fidelity = gpu_results
            .iter()
            .map(|r| r.gpu_fidelity)
            .fold(f64::INFINITY, f64::min);
        let gpu_faster: usize = gpu_results.iter().filter(|r| r.gpu_speedup > 1.0).count();

        println!(
            "GPU avg speedup:  {:.2}x over CPU sequential",
            avg_gpu_speedup
        );
        println!(
            "GPU faster in:    {}/{} benchmarks",
            gpu_faster,
            gpu_results.len()
        );
        println!(
            "Min GPU fidelity: {:.8} {}",
            min_gpu_fidelity,
            if min_gpu_fidelity > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
    } else {
        println!("GPU: not available");
    }

    let avg_fusion: f64 =
        results.iter().map(|r| r.fusion_speedup).sum::<f64>() / results.len() as f64;
    println!(
        "Fusion speedup:   {:.2}x (CPU fused vs CPU sequential)",
        avg_fusion
    );
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qft_circuit_gate_count() {
        let gates = qft_circuit(4);
        // QFT-4: 4 H gates + (3+2+1) = 6 CR gates + 2 SWAPs = 12
        assert!(
            gates.len() >= 10,
            "QFT-4 should have at least 10 gates, got {}",
            gates.len()
        );
    }

    #[test]
    fn test_random_circuit_deterministic() {
        let g1 = random_circuit(4, 5, 42);
        let g2 = random_circuit(4, 5, 42);
        assert_eq!(g1.len(), g2.len());
        for (a, b) in g1.iter().zip(g2.iter()) {
            assert_eq!(a.gate_type, b.gate_type);
            assert_eq!(a.targets, b.targets);
        }
    }

    #[test]
    fn test_bell_circuit() {
        let gates = bell_circuit(4);
        // H(0) + CNOT(0,1) + CNOT(1,2) + CNOT(2,3) = 4 gates
        assert_eq!(gates.len(), 4);
    }

    #[test]
    fn test_grover_circuit_not_empty() {
        let gates = grover_circuit(3, 4);
        assert!(!gates.is_empty());
    }

    #[test]
    fn test_benchmark_fidelity_qft10() {
        let gates = qft_circuit(10);
        let result = run_benchmark("QFT-10", &gates, 10);
        assert!(
            result.fidelity > 1.0 - 1e-10,
            "QFT-10 fidelity too low: {}",
            result.fidelity
        );
    }

    #[test]
    fn test_benchmark_fidelity_random() {
        let gates = random_circuit(8, 10, 123);
        let result = run_benchmark("Random-8", &gates, 8);
        assert!(
            result.fidelity > 1.0 - 1e-10,
            "Random-8 fidelity too low: {}",
            result.fidelity
        );
    }

    #[test]
    fn test_benchmark_fidelity_bell() {
        let gates = bell_circuit(10);
        let result = run_benchmark("Bell-10", &gates, 10);
        assert!(
            result.fidelity > 1.0 - 1e-10,
            "Bell-10 fidelity too low: {}",
            result.fidelity
        );
    }

    #[test]
    fn test_qft_fusion_fidelity() {
        // QFT has no consecutive single-qubit gates on the same qubit
        // (each H is immediately followed by CR gates on that qubit),
        // so elimination count is 0. But fused execution must be correct.
        let gates = qft_circuit(10);
        let result = run_benchmark("QFT-10-fidelity", &gates, 10);
        assert!(
            result.fidelity > 1.0 - 1e-10,
            "QFT-10 fused fidelity too low: {}",
            result.fidelity
        );
    }

    #[test]
    fn test_full_suite_runs() {
        // Just verify it doesn't panic — use small circuits
        let gates = qft_circuit(4);
        let result = run_benchmark("QFT-4", &gates, 4);
        assert!(result.fidelity > 0.99);
    }
}
