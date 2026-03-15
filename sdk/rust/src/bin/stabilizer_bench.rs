//! SIMD Stabilizer Throughput Benchmark
//!
//! Measures Clifford gate throughput of the packed-bitstring stabilizer simulator
//! at various qubit counts (50 to 1000). Reports gates/second and compares
//! against published Stim numbers (~100B Pauli multiplications/sec on AVX-256).
//!
//! The SIMD stabilizer uses NEON 128-bit intrinsics on AArch64 (Apple Silicon),
//! processing 128 qubits per instruction for commutation and row multiplication.
//!
//! Usage:
//!     cargo run --release --bin stabilizer_bench

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use nqpu_metal::simd_stabilizer::{
    CircuitSimulator, SimdStabilizerConfig, StabilizerInstruction, StabilizerTableau,
};

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

const WARMUP_RUNS: usize = 2;
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
// Circuit generation
// ---------------------------------------------------------------------------

/// Generate a random Clifford circuit as a sequence of StabilizerInstructions.
fn random_clifford_circuit(
    n_qubits: usize,
    n_gates: usize,
    rng: &mut StdRng,
) -> Vec<StabilizerInstruction> {
    let mut circuit = Vec::with_capacity(n_gates);
    for _ in 0..n_gates {
        let kind = rng.gen_range(0u8..4);
        match kind {
            0 => {
                let q = rng.gen_range(0..n_qubits);
                circuit.push(StabilizerInstruction::H(q));
            }
            1 => {
                let q = rng.gen_range(0..n_qubits);
                circuit.push(StabilizerInstruction::S(q));
            }
            2 => {
                let c = rng.gen_range(0..n_qubits);
                let mut t = rng.gen_range(0..n_qubits);
                while t == c {
                    t = rng.gen_range(0..n_qubits);
                }
                circuit.push(StabilizerInstruction::CX(c, t));
            }
            _ => {
                let a = rng.gen_range(0..n_qubits);
                let mut b = rng.gen_range(0..n_qubits);
                while b == a {
                    b = rng.gen_range(0..n_qubits);
                }
                circuit.push(StabilizerInstruction::CZ(a, b));
            }
        }
    }
    circuit
}

// ---------------------------------------------------------------------------
// Benchmark: Raw tableau gate throughput
// ---------------------------------------------------------------------------

fn bench_tableau_gates(n_qubits: usize, n_gates: usize, rng: &mut StdRng) -> BenchResult {
    let circuit = random_clifford_circuit(n_qubits, n_gates, rng);

    measure(|| {
        let mut tab = StabilizerTableau::new(n_qubits);
        for instr in &circuit {
            match instr {
                StabilizerInstruction::H(q) => {
                    tab.h(*q).ok();
                }
                StabilizerInstruction::S(q) => {
                    tab.s(*q).ok();
                }
                StabilizerInstruction::CX(c, t) => {
                    tab.cx(*c, *t).ok();
                }
                StabilizerInstruction::CZ(a, b) => {
                    tab.cz(*a, *b).ok();
                }
                _ => {}
            }
        }
        black_box(&tab);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: CircuitSimulator (with instruction dispatch overhead)
// ---------------------------------------------------------------------------

fn bench_circuit_simulator(n_qubits: usize, n_gates: usize, rng: &mut StdRng) -> BenchResult {
    let circuit = random_clifford_circuit(n_qubits, n_gates, rng);
    let config = SimdStabilizerConfig::new(n_qubits);

    measure(|| {
        let mut sim = CircuitSimulator::new(config.clone()).expect("simulator creation failed");
        for instr in &circuit {
            sim.execute(instr).ok();
        }
        black_box(&sim);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Single gate type (H only, CX only) for isolation
// ---------------------------------------------------------------------------

fn bench_single_h(n_qubits: usize, n_gates: usize, rng: &mut StdRng) -> BenchResult {
    let targets: Vec<usize> = (0..n_gates).map(|_| rng.gen_range(0..n_qubits)).collect();

    measure(|| {
        let mut tab = StabilizerTableau::new(n_qubits);
        for &q in &targets {
            tab.h(q).ok();
        }
        black_box(&tab);
    })
}

fn bench_single_cx(n_qubits: usize, n_gates: usize, rng: &mut StdRng) -> BenchResult {
    let pairs: Vec<(usize, usize)> = (0..n_gates)
        .map(|_| {
            let c = rng.gen_range(0..n_qubits);
            let mut t = rng.gen_range(0..n_qubits);
            while t == c {
                t = rng.gen_range(0..n_qubits);
            }
            (c, t)
        })
        .collect();

    measure(|| {
        let mut tab = StabilizerTableau::new(n_qubits);
        for &(c, t) in &pairs {
            tab.cx(c, t).ok();
        }
        black_box(&tab);
    })
}

// ---------------------------------------------------------------------------
// Pauli multiplication throughput estimate
// ---------------------------------------------------------------------------

/// Each Clifford gate on an n-qubit tableau touches 2n rows.
/// A single-qubit gate (H, S) reads/writes 1 word position per row.
/// A two-qubit gate (CX, CZ) reads/writes 2 word positions per row.
/// We report "Pauli row operations per second" for comparison with Stim.
fn pauli_row_ops(n_qubits: usize, n_gates: usize, elapsed_us: f64) -> f64 {
    // Each gate operates on 2n rows
    let total_row_ops = n_gates as f64 * (2 * n_qubits) as f64;
    total_row_ops / (elapsed_us * 1e-6)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=============================================================");
    println!("  nQPU-Metal: SIMD Stabilizer Throughput Benchmark");
    println!("=============================================================\n");

    let mut rng = StdRng::seed_from_u64(42);

    let qubit_counts: Vec<usize> = vec![50, 100, 200, 500, 1000];
    let n_gates = 1_000_000;

    // --- Section 1: Raw tableau throughput (mixed Clifford) ---
    println!("--- Raw Tableau: Mixed Clifford ({} gates) ---", n_gates);
    println!(
        "{:>8} {:>14} {:>14} {:>14} {:>16} {:>16}",
        "qubits", "mean_ms", "min_ms", "max_ms", "gates/sec", "row_ops/sec"
    );
    println!("{}", "-".repeat(86));

    for &n in &qubit_counts {
        let r = bench_tableau_gates(n, n_gates, &mut rng);
        let gates_sec = (n_gates as f64) / (r.mean_us * 1e-6);
        let row_ops = pauli_row_ops(n, n_gates, r.mean_us);
        println!(
            "{:>8} {:>14.2} {:>14.2} {:>14.2} {:>16.0} {:>16.2e}",
            n,
            r.mean_us / 1000.0,
            r.min_us / 1000.0,
            r.max_us / 1000.0,
            gates_sec,
            row_ops
        );
    }

    // --- Section 2: CircuitSimulator (instruction dispatch) ---
    let sim_gates = 100_000; // Fewer gates since CircuitSimulator has more overhead
    println!(
        "\n--- CircuitSimulator: Mixed Clifford ({} gates) ---",
        sim_gates
    );
    println!(
        "{:>8} {:>14} {:>14} {:>14} {:>16}",
        "qubits", "mean_ms", "min_ms", "max_ms", "gates/sec"
    );
    println!("{}", "-".repeat(70));

    for &n in &qubit_counts {
        let r = bench_circuit_simulator(n, sim_gates, &mut rng);
        let gates_sec = (sim_gates as f64) / (r.mean_us * 1e-6);
        println!(
            "{:>8} {:>14.2} {:>14.2} {:>14.2} {:>16.0}",
            n,
            r.mean_us / 1000.0,
            r.min_us / 1000.0,
            r.max_us / 1000.0,
            gates_sec
        );
    }

    // --- Section 3: Isolated gate types ---
    println!("\n--- Isolated Gate: Hadamard Only ({} gates) ---", n_gates);
    println!(
        "{:>8} {:>14} {:>16} {:>14}",
        "qubits", "mean_ms", "gates/sec", "ns/gate"
    );
    println!("{}", "-".repeat(56));

    for &n in &qubit_counts {
        let r = bench_single_h(n, n_gates, &mut rng);
        let gates_sec = (n_gates as f64) / (r.mean_us * 1e-6);
        let ns_per_gate = r.mean_us * 1000.0 / n_gates as f64;
        println!(
            "{:>8} {:>14.2} {:>16.0} {:>14.1}",
            n,
            r.mean_us / 1000.0,
            gates_sec,
            ns_per_gate
        );
    }

    println!("\n--- Isolated Gate: CNOT Only ({} gates) ---", n_gates);
    println!(
        "{:>8} {:>14} {:>16} {:>14}",
        "qubits", "mean_ms", "gates/sec", "ns/gate"
    );
    println!("{}", "-".repeat(56));

    for &n in &qubit_counts {
        let r = bench_single_cx(n, n_gates, &mut rng);
        let gates_sec = (n_gates as f64) / (r.mean_us * 1e-6);
        let ns_per_gate = r.mean_us * 1000.0 / n_gates as f64;
        println!(
            "{:>8} {:>14.2} {:>16.0} {:>14.1}",
            n,
            r.mean_us / 1000.0,
            gates_sec,
            ns_per_gate
        );
    }

    // --- Section 4: Comparison with Stim ---
    println!("\n--- Comparison with Stim Reference ---");
    println!("Stim reports ~100B Pauli mult/sec on AVX-256 (x86_64).");
    println!("nQPU-Metal uses NEON 128-bit (AArch64, Apple Silicon).\n");

    // Pick the 1000-qubit result for comparison
    let n_cmp = 1000;
    let r_cmp = bench_tableau_gates(n_cmp, n_gates, &mut rng);
    let our_row_ops = pauli_row_ops(n_cmp, n_gates, r_cmp.mean_us);
    let stim_ref = 100e9;
    let ratio = our_row_ops / stim_ref;

    println!("  nQPU-Metal ({} qubits, {} gates):", n_cmp, n_gates);
    println!("    Row operations/sec: {:.2e}", our_row_ops);
    println!("    vs Stim reference:  {:.1}%", ratio * 100.0);
    if ratio >= 1.0 {
        println!("    Status: EXCEEDS Stim reference");
    } else if ratio >= 0.5 {
        println!("    Status: Competitive (within 2x of Stim)");
    } else {
        println!(
            "    Status: Room for improvement ({:.1}x slower)",
            1.0 / ratio
        );
    }

    println!("\n[done]");
}
