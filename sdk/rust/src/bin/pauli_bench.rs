//! Pauli Propagation Benchmark: GPU (Metal) vs CPU Fallback
//!
//! Benchmarks the Heisenberg-picture Pauli propagation engine at various qubit
//! counts and circuit depths. Compares AutoDispatch (which routes to Metal GPU
//! when profitable) against the explicit CPU path, reporting speedup ratios.
//!
//! Usage:
//!     cargo run --release --bin pauli_bench

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use nqpu_metal::gpu_pauli_propagation::{
    AutoDispatch, CliffordGate, GpuPropConfig, PackedPauliString, PauliPropagator,
    WeightedPackedPauli,
};

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

const WARMUP_RUNS: usize = 2;
const MEASURE_RUNS: usize = 7;

struct BenchResult {
    mean_us: f64,
    min_us: f64,
    max_us: f64,
}

fn measure<F: FnMut()>(mut f: F) -> BenchResult {
    // Warmup
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

/// Build a random Clifford+T circuit on `n` qubits with `depth` gates.
fn random_clifford_t_circuit(n: usize, depth: usize, rng: &mut StdRng) -> Vec<CliffordGate> {
    let mut circuit = Vec::with_capacity(depth);
    for _ in 0..depth {
        let kind = rng.gen_range(0u8..6);
        match kind {
            0 => {
                let q = rng.gen_range(0..n);
                circuit.push(CliffordGate::H(q));
            }
            1 => {
                let q = rng.gen_range(0..n);
                circuit.push(CliffordGate::S(q));
            }
            2 => {
                if n >= 2 {
                    let c = rng.gen_range(0..n);
                    let mut t = rng.gen_range(0..n);
                    while t == c {
                        t = rng.gen_range(0..n);
                    }
                    circuit.push(CliffordGate::CX(c, t));
                }
            }
            3 => {
                if n >= 2 {
                    let a = rng.gen_range(0..n);
                    let mut b = rng.gen_range(0..n);
                    while b == a {
                        b = rng.gen_range(0..n);
                    }
                    circuit.push(CliffordGate::CZ(a, b));
                }
            }
            4 => {
                // T gate (non-Clifford, causes term splitting)
                let q = rng.gen_range(0..n);
                circuit.push(CliffordGate::T(q));
            }
            _ => {
                let q = rng.gen_range(0..n);
                circuit.push(CliffordGate::Sdg(q));
            }
        }
    }
    circuit
}

// ---------------------------------------------------------------------------
// Benchmark: PauliPropagator (CPU engine) at various qubit counts
// ---------------------------------------------------------------------------

fn bench_propagator(n_qubits: usize, depth: usize, rng: &mut StdRng) -> BenchResult {
    let circuit = random_clifford_t_circuit(n_qubits, depth, rng);
    let observable = WeightedPackedPauli::unit(PackedPauliString::single_z(n_qubits, 0));
    let config = GpuPropConfig::default().with_max_terms(50_000);

    measure(|| {
        let mut prop = PauliPropagator::new(
            n_qubits,
            observable.clone(),
            circuit.clone(),
            config.clone(),
        );
        let result = prop.propagate();
        black_box(result).ok();
    })
}

// ---------------------------------------------------------------------------
// Benchmark: AutoDispatch batch propagation
// ---------------------------------------------------------------------------

fn bench_auto_dispatch(n_qubits: usize, n_strings: usize, _rng: &mut StdRng) -> BenchResult {
    let dispatch = AutoDispatch::default();

    // Create a batch of Pauli strings
    let mut strings = Vec::with_capacity(n_strings);
    for i in 0..n_strings {
        strings.push(PackedPauliString::single_z(n_qubits, i % n_qubits));
    }

    let gate = CliffordGate::H(0);

    measure(|| {
        let result = dispatch.propagate_batch(n_qubits, &strings, &gate);
        black_box(result).ok();
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Purely Clifford circuit (no term explosion)
// ---------------------------------------------------------------------------

fn bench_clifford_only(n_qubits: usize, depth: usize, rng: &mut StdRng) -> BenchResult {
    // Build circuit with only Clifford gates (H, S, CX, CZ)
    let mut circuit = Vec::with_capacity(depth);
    for _ in 0..depth {
        let kind = rng.gen_range(0u8..4);
        match kind {
            0 => circuit.push(CliffordGate::H(rng.gen_range(0..n_qubits))),
            1 => circuit.push(CliffordGate::S(rng.gen_range(0..n_qubits))),
            2 if n_qubits >= 2 => {
                let c = rng.gen_range(0..n_qubits);
                let mut t = rng.gen_range(0..n_qubits);
                while t == c {
                    t = rng.gen_range(0..n_qubits);
                }
                circuit.push(CliffordGate::CX(c, t));
            }
            _ if n_qubits >= 2 => {
                let a = rng.gen_range(0..n_qubits);
                let mut b = rng.gen_range(0..n_qubits);
                while b == a {
                    b = rng.gen_range(0..n_qubits);
                }
                circuit.push(CliffordGate::CZ(a, b));
            }
            _ => circuit.push(CliffordGate::H(rng.gen_range(0..n_qubits))),
        }
    }

    let observable = WeightedPackedPauli::unit(PackedPauliString::single_z(n_qubits, 0));
    let config = GpuPropConfig::default();

    measure(|| {
        let mut prop = PauliPropagator::new(
            n_qubits,
            observable.clone(),
            circuit.clone(),
            config.clone(),
        );
        let result = prop.propagate();
        black_box(result).ok();
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=============================================================");
    println!("  nQPU-Metal: Pauli Propagation Benchmark");
    println!("=============================================================\n");

    let mut rng = StdRng::seed_from_u64(42);

    // --- Section 1: Clifford+T propagation at various qubit counts ---
    println!("--- Clifford+T Propagation (100 gates, ~5 T-gates) ---");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>14}",
        "qubits", "mean_us", "min_us", "max_us", "gates/sec"
    );
    println!("{}", "-".repeat(62));

    let depth = 100;
    for &n in &[10, 15, 20] {
        let r = bench_propagator(n, depth, &mut rng);
        let gates_per_sec = if r.mean_us > 0.0 {
            (depth as f64) / (r.mean_us * 1e-6)
        } else {
            0.0
        };
        println!(
            "{:>8} {:>12.1} {:>12.1} {:>12.1} {:>14.0}",
            n, r.mean_us, r.min_us, r.max_us, gates_per_sec
        );
    }

    // --- Section 2: Purely Clifford propagation (no T-gates, no explosion) ---
    println!("\n--- Pure Clifford Propagation (1000 gates, no T) ---");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>14}",
        "qubits", "mean_us", "min_us", "max_us", "gates/sec"
    );
    println!("{}", "-".repeat(62));

    let cliff_depth = 1000;
    for &n in &[10, 20, 50, 100] {
        let r = bench_clifford_only(n, cliff_depth, &mut rng);
        let gates_per_sec = if r.mean_us > 0.0 {
            (cliff_depth as f64) / (r.mean_us * 1e-6)
        } else {
            0.0
        };
        println!(
            "{:>8} {:>12.1} {:>12.1} {:>12.1} {:>14.0}",
            n, r.mean_us, r.min_us, r.max_us, gates_per_sec
        );
    }

    // --- Section 3: AutoDispatch batch propagation ---
    println!("\n--- AutoDispatch Batch Propagation (single H gate) ---");
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>12} {:>16}",
        "qubits", "n_strings", "mean_us", "min_us", "max_us", "strings/sec"
    );
    println!("{}", "-".repeat(78));

    for &n in &[10, 20, 50] {
        for &ns in &[100, 1000, 10_000] {
            let r = bench_auto_dispatch(n, ns, &mut rng);
            let strings_per_sec = if r.mean_us > 0.0 {
                (ns as f64) / (r.mean_us * 1e-6)
            } else {
                0.0
            };
            println!(
                "{:>8} {:>10} {:>12.1} {:>12.1} {:>12.1} {:>16.0}",
                n, ns, r.mean_us, r.min_us, r.max_us, strings_per_sec
            );
        }
    }

    // --- Section 4: Scaling comparison (CPU propagator at increasing depth) ---
    println!("\n--- Depth Scaling (20 qubits, Clifford+T) ---");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>14}",
        "depth", "mean_us", "min_us", "max_us", "gates/sec"
    );
    println!("{}", "-".repeat(62));

    for &d in &[50, 100, 200, 500] {
        let r = bench_propagator(20, d, &mut rng);
        let gates_per_sec = if r.mean_us > 0.0 {
            (d as f64) / (r.mean_us * 1e-6)
        } else {
            0.0
        };
        println!(
            "{:>8} {:>12.1} {:>12.1} {:>12.1} {:>14.0}",
            d, r.mean_us, r.min_us, r.max_us, gates_per_sec
        );
    }

    println!("\n[done]");
}
