//! Qubit Scaling Benchmark: Maximum Achievable Qubits and Memory Usage
//!
//! Allocates QuantumState at increasing qubit counts and applies random gate
//! layers (Hadamard + CNOT), measuring wall time per gate, memory footprint,
//! and throughput (gates/sec, GFlop/s).
//!
//! Usage:
//!     cargo run --release --bin qubit_scaling_bench

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use nqpu_metal::{GateOperations, QuantumState};

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

const WARMUP_RUNS: usize = 3;
const MEASURE_RUNS: usize = 5;

struct BenchResult {
    mean_us: f64,
    min_us: f64,
    max_us: f64,
}

fn measure<F: FnMut()>(mut f: F) -> BenchResult {
    for _ in 0..WARMUP_RUNS {
        f();
    }

    let mut samples = Vec::with_capacity(MEASURE_RUNS);
    for _ in 0..MEASURE_RUNS {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let mean_us = samples.iter().sum::<f64>() / samples.len() as f64;
    let min_us = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    BenchResult {
        mean_us,
        min_us,
        max_us,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn memory_bytes(n_qubits: usize) -> usize {
    // Complex64 = 16 bytes per amplitude, 2^n amplitudes
    (1usize << n_qubits) * 16
}

fn format_memory(bytes: usize) -> String {
    let kb = bytes as f64 / 1024.0;
    let mb = kb / 1024.0;
    let gb = mb / 1024.0;
    if gb >= 1.0 {
        format!("{:.2} GB", gb)
    } else if mb >= 1.0 {
        format!("{:.1} MB", mb)
    } else {
        format!("{:.0} KB", kb)
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Hadamard gate throughput at each qubit count
// ---------------------------------------------------------------------------

fn bench_hadamard(n_qubits: usize) -> BenchResult {
    let mut state = QuantumState::new(n_qubits);
    let iters = 100;

    measure(|| {
        for _ in 0..iters {
            GateOperations::h(&mut state, 0);
        }
        black_box(&state);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: CNOT layer (chain) throughput
// ---------------------------------------------------------------------------

fn bench_cnot_layer(n_qubits: usize) -> BenchResult {
    let mut state = QuantumState::new(n_qubits);

    measure(|| {
        for q in 0..n_qubits - 1 {
            GateOperations::cnot(&mut state, q, q + 1);
        }
        black_box(&state);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Random circuit layer (H + CNOT + T + Rz)
// ---------------------------------------------------------------------------

fn bench_random_circuit(n_qubits: usize, rng: &mut StdRng) -> BenchResult {
    let mut state = QuantumState::new(n_qubits);
    let layer_gates = n_qubits * 2; // ~2 gates per qubit per layer

    // Pre-generate gate schedule
    let mut schedule: Vec<(u8, usize, usize, f64)> = Vec::with_capacity(layer_gates);
    for _ in 0..layer_gates {
        let kind = rng.gen_range(0u8..4);
        let q1 = rng.gen_range(0..n_qubits);
        let mut q2 = rng.gen_range(0..n_qubits);
        while q2 == q1 && n_qubits > 1 {
            q2 = rng.gen_range(0..n_qubits);
        }
        let angle: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        schedule.push((kind, q1, q2, angle));
    }

    measure(|| {
        for &(kind, q1, q2, angle) in &schedule {
            match kind {
                0 => GateOperations::h(&mut state, q1),
                1 => {
                    if n_qubits > 1 {
                        GateOperations::cnot(&mut state, q1, q2);
                    }
                }
                2 => GateOperations::t(&mut state, q1),
                _ => GateOperations::rz(&mut state, q1, angle),
            }
        }
        black_box(&state);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: State allocation time
// ---------------------------------------------------------------------------

fn bench_allocation(n_qubits: usize) -> BenchResult {
    measure(|| {
        let state = QuantumState::new(n_qubits);
        black_box(&state);
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=============================================================");
    println!("  nQPU-Metal: Qubit Scaling Benchmark");
    println!("=============================================================\n");

    let mut rng = StdRng::seed_from_u64(42);

    let qubit_counts: Vec<usize> = vec![10, 15, 20, 22, 24, 26, 28];

    // --- Section 1: State allocation ---
    println!("--- State Allocation ---");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12}",
        "qubits", "memory", "mean_us", "min_us", "max_us");
    println!("{}", "-".repeat(60));

    for &n in &qubit_counts {
        let mem = memory_bytes(n);
        // Skip if estimated memory exceeds 8 GB
        if mem > 8 * 1024 * 1024 * 1024 {
            println!("{:>8} {:>12}  (skipped -- exceeds 8 GB limit)", n, format_memory(mem));
            continue;
        }
        let r = bench_allocation(n);
        println!("{:>8} {:>12} {:>12.1} {:>12.1} {:>12.1}",
            n, format_memory(mem), r.mean_us, r.min_us, r.max_us);
    }

    // --- Section 2: Hadamard gate throughput ---
    println!("\n--- Hadamard Gate Throughput (100 H gates on qubit 0) ---");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "qubits", "memory", "mean_us", "min_us", "max_us", "GFlop/s");
    println!("{}", "-".repeat(74));

    let h_iters = 100;
    for &n in &qubit_counts {
        let mem = memory_bytes(n);
        if mem > 8 * 1024 * 1024 * 1024 {
            println!("{:>8} {:>12}  (skipped)", n, format_memory(mem));
            continue;
        }
        let r = bench_hadamard(n);
        let dim = 1usize << n;
        // Each Hadamard touches dim/2 pairs, ~6 flops per pair
        let gflops = (dim as f64 * 6.0 * h_iters as f64) / (r.mean_us * 1e-6) / 1e9;
        println!("{:>8} {:>12} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
            n, format_memory(mem), r.mean_us, r.min_us, r.max_us, gflops);
    }

    // --- Section 3: CNOT chain layer ---
    println!("\n--- CNOT Chain Layer (n-1 CNOT gates) ---");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12} {:>14}",
        "qubits", "gates", "mean_us", "min_us", "max_us", "us/gate");
    println!("{}", "-".repeat(76));

    for &n in &qubit_counts {
        let mem = memory_bytes(n);
        if mem > 8 * 1024 * 1024 * 1024 {
            println!("{:>8} {:>12}  (skipped)", n, n - 1);
            continue;
        }
        let r = bench_cnot_layer(n);
        let us_per_gate = r.mean_us / (n - 1) as f64;
        println!("{:>8} {:>12} {:>12.1} {:>12.1} {:>12.1} {:>14.2}",
            n, n - 1, r.mean_us, r.min_us, r.max_us, us_per_gate);
    }

    // --- Section 4: Random circuit layer ---
    println!("\n--- Random Circuit Layer (H + CNOT + T + Rz, ~2n gates) ---");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12} {:>14}",
        "qubits", "gates", "mean_us", "min_us", "max_us", "us/gate");
    println!("{}", "-".repeat(76));

    for &n in &qubit_counts {
        let mem = memory_bytes(n);
        if mem > 8 * 1024 * 1024 * 1024 {
            println!("{:>8} {:>12}  (skipped)", n, n * 2);
            continue;
        }
        let layer_gates = n * 2;
        let r = bench_random_circuit(n, &mut rng);
        let us_per_gate = r.mean_us / layer_gates as f64;
        println!("{:>8} {:>12} {:>12.1} {:>12.1} {:>12.1} {:>14.2}",
            n, layer_gates, r.mean_us, r.min_us, r.max_us, us_per_gate);
    }

    // --- Section 5: Summary table ---
    println!("\n--- Summary: Estimated Capacity ---");
    println!("{:>8} {:>14} {:>20}",
        "qubits", "memory", "state_dim");
    println!("{}", "-".repeat(44));

    for n in (10..=33).step_by(1) {
        let mem = memory_bytes(n);
        let dim = 1u64 << n;
        let feasible = if mem <= 512 * 1024 * 1024 {
            "easy"
        } else if mem <= 4 * 1024 * 1024 * 1024 {
            "moderate"
        } else if mem <= 32 * 1024usize.pow(3) {
            "hard"
        } else {
            "OOM"
        };
        println!("{:>8} {:>14} {:>14}   {}", n, format_memory(mem), dim, feasible);
    }

    println!("\n[done]");
}
