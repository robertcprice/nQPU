//! Tensor Network Benchmark: DMRG and VUMPS Performance
//!
//! Measures convergence time, final energy, and sweep throughput for:
//!   - GPU-DMRG (finite Ising/Heisenberg chains at various lengths and bond dims)
//!   - VUMPS (infinite Ising model at various bond dimensions)
//!
//! ## Numerical Stability Note
//!
//! DMRG can encounter "dimension is zero" panics when SVD truncation removes
//! all singular values at large bond dimensions or near-degenerate states.
//! The `svd_truncate` guard in `gpu_dmrg.rs` and `dmrg_tdvp.rs` prevents
//! this by preserving at least rank 1. If you see zero-dim errors here,
//! verify that fix is in place.
//!
//! ## Expected Scaling
//!
//! - DMRG per-sweep cost: O(L * D^3 * d^2) where L = sites, D = bond dim, d = phys dim
//! - VUMPS per-iteration cost: O(chi^3 * d^2) where chi = bond dim
//! - Doubling D should increase time by ~8x (D^3 scaling)
//!
//! Usage:
//!     cargo run --release --bin tn_bench

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use nqpu_metal::gpu_dmrg::{GpuDmrgConfig, GpuDmrgEngine, GpuMps, MpoHamiltonian};
use nqpu_metal::imps_ipeps::{ising_two_site_gate, vumps, VumpsConfig};

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

const WARMUP_RUNS: usize = 1;
const MEASURE_RUNS: usize = 3;

struct BenchResult {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    /// Individual timing samples in milliseconds.
    samples: Vec<f64>,
}

fn measure<F: FnMut()>(mut f: F) -> BenchResult {
    for _ in 0..WARMUP_RUNS {
        f();
    }

    let mut samples = Vec::with_capacity(MEASURE_RUNS);
    for _ in 0..MEASURE_RUNS {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    let min_ms = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    BenchResult {
        mean_ms,
        min_ms,
        max_ms,
        samples,
    }
}

// ---------------------------------------------------------------------------
// DMRG benchmarks
// ---------------------------------------------------------------------------

struct DmrgBenchResult {
    timing: BenchResult,
    energy: f64,
    sweeps: usize,
    converged: bool,
    entropy: f64,
    /// Per-sweep energy convergence from the last measured run.
    energy_history: Vec<f64>,
    /// Average milliseconds per sweep (total / sweeps) from the last run.
    ms_per_sweep: f64,
}

/// Benchmark DMRG on the transverse-field Ising model H = -J sum(ZZ) - h sum(X).
///
/// The exact ground state energy for the 1D TFIM can be computed via Jordan-Wigner
/// free-fermion mapping. At h/J = 0.5 (ordered phase), DMRG converges quickly and
/// the entanglement entropy is low, so moderate bond dimensions (D=16-32) suffice.
fn bench_dmrg_ising(
    num_sites: usize,
    bond_dim: usize,
    rng: &mut StdRng,
) -> DmrgBenchResult {
    let j = 1.0;
    let h_field = 0.5;
    let mpo = MpoHamiltonian::ising(num_sites, j, h_field);

    let config = GpuDmrgConfig::new()
        .with_max_bond_dim(bond_dim)
        .with_max_sweeps(20)
        .with_energy_tolerance(1e-8)
        .with_two_site(true);

    let engine = GpuDmrgEngine::new(config);

    let mut last_energy = 0.0;
    let mut last_sweeps = 0;
    let mut last_converged = false;
    let mut last_entropy = 0.0;
    let mut last_energy_history: Vec<f64> = Vec::new();
    let mut last_run_ms = 0.0;

    let timing = measure(|| {
        let run_start = Instant::now();
        let mut mps = GpuMps::random(num_sites, 2, bond_dim.min(4), rng);
        match engine.run(&mut mps, &mpo) {
            Ok(result) => {
                last_energy = result.energy;
                last_sweeps = result.sweeps;
                last_converged = result.converged;
                last_entropy = result.entanglement_entropy;
                last_energy_history = result.energy_history.clone();
                last_run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            }
            Err(e) => {
                eprintln!("  DMRG error (L={}, D={}): {}", num_sites, bond_dim, e);
            }
        }
        black_box(&mps);
    });

    let ms_per_sweep = if last_sweeps > 0 {
        last_run_ms / last_sweeps as f64
    } else {
        0.0
    };

    DmrgBenchResult {
        timing,
        energy: last_energy,
        sweeps: last_sweeps,
        converged: last_converged,
        entropy: last_entropy,
        energy_history: last_energy_history,
        ms_per_sweep,
    }
}

/// Benchmark DMRG on the Heisenberg XXX model H = J sum(XX + YY + ZZ).
///
/// The Heisenberg chain has higher entanglement than Ising (logarithmic scaling
/// with L at criticality), so it requires larger bond dimensions for convergence.
/// This makes it a better stress test for SVD truncation numerics.
fn bench_dmrg_heisenberg(
    num_sites: usize,
    bond_dim: usize,
    rng: &mut StdRng,
) -> DmrgBenchResult {
    let j = 1.0;
    let mpo = MpoHamiltonian::heisenberg(num_sites, j);

    let config = GpuDmrgConfig::new()
        .with_max_bond_dim(bond_dim)
        .with_max_sweeps(30)
        .with_energy_tolerance(1e-8)
        .with_two_site(true);

    let engine = GpuDmrgEngine::new(config);

    let mut last_energy = 0.0;
    let mut last_sweeps = 0;
    let mut last_converged = false;
    let mut last_entropy = 0.0;
    let mut last_energy_history: Vec<f64> = Vec::new();
    let mut last_run_ms = 0.0;

    let timing = measure(|| {
        let run_start = Instant::now();
        let mut mps = GpuMps::random(num_sites, 2, bond_dim.min(4), rng);
        match engine.run(&mut mps, &mpo) {
            Ok(result) => {
                last_energy = result.energy;
                last_sweeps = result.sweeps;
                last_converged = result.converged;
                last_entropy = result.entanglement_entropy;
                last_energy_history = result.energy_history.clone();
                last_run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            }
            Err(e) => {
                eprintln!("  DMRG error (L={}, D={}): {}", num_sites, bond_dim, e);
            }
        }
        black_box(&mps);
    });

    let ms_per_sweep = if last_sweeps > 0 {
        last_run_ms / last_sweeps as f64
    } else {
        0.0
    };

    DmrgBenchResult {
        timing,
        energy: last_energy,
        sweeps: last_sweeps,
        converged: last_converged,
        entropy: last_entropy,
        energy_history: last_energy_history,
        ms_per_sweep,
    }
}

// ---------------------------------------------------------------------------
// VUMPS benchmarks
// ---------------------------------------------------------------------------

struct VumpsBenchResult {
    timing: BenchResult,
    energy_per_site: f64,
    iterations: usize,
    converged: bool,
    gradient_norm: f64,
}

/// Benchmark VUMPS on the infinite transverse-field Ising model.
///
/// VUMPS (Variational Uniform MPS) works directly in the thermodynamic limit
/// using a uniform MPS ansatz. Cost per iteration scales as O(chi^3 * d^2).
/// The exact energy per site at h/J=0.5 is approximately E/site ~ -1.0639.
fn bench_vumps_ising(
    bond_dim: usize,
    rng: &mut StdRng,
) -> VumpsBenchResult {
    let j = 1.0;
    let h_field = 0.5;
    let h_gate = ising_two_site_gate(j, h_field);

    let config = VumpsConfig::new()
        .with_bond_dim(bond_dim)
        .with_max_iterations(100)
        .with_tolerance(1e-8);

    let mut last_energy = 0.0;
    let mut last_iters = 0;
    let mut last_converged = false;
    let mut last_grad = 0.0;

    let timing = measure(|| {
        match vumps(&h_gate, &config, rng) {
            Ok(result) => {
                last_energy = result.energy_per_site;
                last_iters = result.iterations;
                last_converged = result.converged;
                last_grad = result.gradient_norm;
            }
            Err(e) => {
                eprintln!("  VUMPS error (chi={}): {}", bond_dim, e);
            }
        }
    });

    VumpsBenchResult {
        timing,
        energy_per_site: last_energy,
        iterations: last_iters,
        converged: last_converged,
        gradient_norm: last_grad,
    }
}

/// Print per-sweep energy convergence for a DMRG run.
fn print_sweep_convergence(label: &str, result: &DmrgBenchResult) {
    if result.energy_history.len() <= 1 {
        return;
    }
    println!("  {} per-sweep convergence ({} sweeps, {:.2} ms/sweep):",
        label, result.sweeps, result.ms_per_sweep);
    for (i, e) in result.energy_history.iter().enumerate() {
        let delta = if i > 0 {
            format!("dE={:+.2e}", e - result.energy_history[i - 1])
        } else {
            "".to_string()
        };
        println!("    sweep {:>2}: E = {:>14.8}  {}", i, e, delta);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=============================================================");
    println!("  nQPU-Metal: Tensor Network Benchmark (DMRG + VUMPS)");
    println!("=============================================================\n");

    let mut rng = StdRng::seed_from_u64(42);

    // -----------------------------------------------------------------------
    // Section 1: DMRG Ising -- site-count scaling at fixed bond dimension.
    // Measures how wall-clock time grows with chain length L.
    // Expected: roughly linear in L (each sweep visits L sites).
    // -----------------------------------------------------------------------
    println!("--- DMRG: Transverse-Field Ising Model (J=1.0, h=0.5) ---");
    println!("{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8} {:>5} {:>8}",
        "sites", "D", "mean_ms", "min_ms", "max_ms", "ms/sweep", "energy", "sweeps", "conv", "S_ent");
    println!("{}", "-".repeat(100));

    let site_counts = [8, 16, 32];
    let bond_dim = 16;
    for &l in &site_counts {
        let r = bench_dmrg_ising(l, bond_dim, &mut rng);
        let conv_str = if r.converged { "yes" } else { "no" };
        println!("{:>6} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.6} {:>8} {:>5} {:>8.4}",
            l, bond_dim, r.timing.mean_ms, r.timing.min_ms, r.timing.max_ms,
            r.ms_per_sweep, r.energy, r.sweeps, conv_str, r.entropy);
    }

    // Show per-sweep convergence for the largest chain
    let r_detail = bench_dmrg_ising(32, bond_dim, &mut rng);
    print_sweep_convergence("Ising L=32 D=16", &r_detail);

    // -----------------------------------------------------------------------
    // Section 2: DMRG Ising -- bond dimension scaling at fixed chain length.
    // Measures how wall-clock time grows with bond dimension D.
    // Expected: cubic in D (O(D^3) per site per sweep).
    // -----------------------------------------------------------------------
    println!("\n--- DMRG: Ising Bond Dimension Scaling (L=16) ---");
    println!("{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8} {:>5}",
        "sites", "D", "mean_ms", "min_ms", "max_ms", "ms/sweep", "energy", "sweeps", "conv");
    println!("{}", "-".repeat(82));

    let l_fixed = 16;
    for &d in &[4, 8, 16, 32] {
        let r = bench_dmrg_ising(l_fixed, d, &mut rng);
        let conv_str = if r.converged { "yes" } else { "no" };
        println!("{:>6} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.6} {:>8} {:>5}",
            l_fixed, d, r.timing.mean_ms, r.timing.min_ms, r.timing.max_ms,
            r.ms_per_sweep, r.energy, r.sweeps, conv_str);
    }

    // -----------------------------------------------------------------------
    // Section 3: DMRG Heisenberg -- higher entanglement stress test.
    // The Heisenberg model is gapless (critical), producing logarithmically
    // divergent entanglement entropy. This stresses SVD truncation more than
    // the gapped Ising model and is the primary test for the zero-dim guard.
    // -----------------------------------------------------------------------
    // Note: DMRG can encounter "dimension is zero" errors at large bond
    // dimensions when SVD truncation removes all singular values. The
    // svd_truncate guard (min rank 1) in gpu_dmrg.rs and dmrg_tdvp.rs
    // prevents this cascade. If you see errors here, verify that fix.
    println!("\n--- DMRG: Heisenberg XXX Model (J=1.0, D=16) ---");
    println!("{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8} {:>5} {:>8}",
        "sites", "D", "mean_ms", "min_ms", "max_ms", "ms/sweep", "energy", "sweeps", "conv", "S_ent");
    println!("{}", "-".repeat(100));

    for &l in &site_counts {
        let r = bench_dmrg_heisenberg(l, bond_dim, &mut rng);
        let conv_str = if r.converged { "yes" } else { "no" };
        println!("{:>6} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.6} {:>8} {:>5} {:>8.4}",
            l, bond_dim, r.timing.mean_ms, r.timing.min_ms, r.timing.max_ms,
            r.ms_per_sweep, r.energy, r.sweeps, conv_str, r.entropy);
    }

    // Show per-sweep convergence for Heisenberg
    let r_heis_detail = bench_dmrg_heisenberg(16, bond_dim, &mut rng);
    print_sweep_convergence("Heisenberg L=16 D=16", &r_heis_detail);

    // -----------------------------------------------------------------------
    // Section 4: VUMPS -- infinite system in the thermodynamic limit.
    // VUMPS optimizes a translationally-invariant uniform MPS directly,
    // avoiding finite-size effects. Cost is O(chi^3 * d^2) per iteration.
    // -----------------------------------------------------------------------
    println!("\n--- VUMPS: Infinite Ising Model (J=1.0, h=0.5) ---");
    println!("{:>6} {:>10} {:>10} {:>10} {:>14} {:>8} {:>5} {:>12}",
        "chi", "mean_ms", "min_ms", "max_ms", "E/site", "iters", "conv", "grad_norm");
    println!("{}", "-".repeat(84));

    // Exact energy per site for TF Ising at h=0.5 is approximately:
    // E/site ~ -1.0639 (from Jordan-Wigner free-fermion exact solution)
    for &chi in &[4, 8, 16, 32] {
        let r = bench_vumps_ising(chi, &mut rng);
        let conv_str = if r.converged { "yes" } else { "no" };
        println!("{:>6} {:>10.2} {:>10.2} {:>10.2} {:>14.8} {:>8} {:>5} {:>12.2e}",
            chi, r.timing.mean_ms, r.timing.min_ms, r.timing.max_ms,
            r.energy_per_site, r.iterations, conv_str, r.gradient_norm);
    }

    // -----------------------------------------------------------------------
    // Section 5: Empirical scaling analysis.
    // Fits the observed time(D) relationship to t ~ D^alpha and reports
    // the exponent. Theory predicts alpha ~ 3 for DMRG.
    // -----------------------------------------------------------------------
    println!("\n--- Scaling Summary ---");
    println!("DMRG complexity: O(L * D^3 * d^2) per sweep");
    println!("VUMPS complexity: O(chi^3 * d^2) per iteration");
    println!("  where L = sites, D/chi = bond dimension, d = physical dimension (2)");

    // Compute empirical scaling exponent for DMRG
    let r_d4 = bench_dmrg_ising(16, 4, &mut rng);
    let r_d8 = bench_dmrg_ising(16, 8, &mut rng);
    let r_d16 = bench_dmrg_ising(16, 16, &mut rng);

    if r_d4.timing.mean_ms > 0.0 && r_d8.timing.mean_ms > 0.0 && r_d16.timing.mean_ms > 0.0 {
        let exp_1 = (r_d8.timing.mean_ms / r_d4.timing.mean_ms).ln() / (8.0f64 / 4.0).ln();
        let exp_2 = (r_d16.timing.mean_ms / r_d8.timing.mean_ms).ln() / (16.0f64 / 8.0).ln();
        println!("\nEmpirical DMRG bond-dim scaling exponent (L=16):");
        println!("  D=4->8:  t ~ D^{:.2}  ({:.2}ms -> {:.2}ms)", exp_1, r_d4.timing.mean_ms, r_d8.timing.mean_ms);
        println!("  D=8->16: t ~ D^{:.2}  ({:.2}ms -> {:.2}ms)", exp_2, r_d8.timing.mean_ms, r_d16.timing.mean_ms);
        println!("  Expected: t ~ D^3");
    }

    // Per-sweep throughput summary
    if r_d4.sweeps > 0 && r_d8.sweeps > 0 && r_d16.sweeps > 0 {
        println!("\nPer-sweep throughput (L=16):");
        println!("  D=4:  {:.2} ms/sweep ({} sweeps)", r_d4.ms_per_sweep, r_d4.sweeps);
        println!("  D=8:  {:.2} ms/sweep ({} sweeps)", r_d8.ms_per_sweep, r_d8.sweeps);
        println!("  D=16: {:.2} ms/sweep ({} sweeps)", r_d16.ms_per_sweep, r_d16.sweeps);
    }

    // -----------------------------------------------------------------------
    // Section 6: GPU vs CPU comparison.
    // Runs DMRG with GPU dispatch enabled vs forced-CPU (gpu_threshold = MAX)
    // to measure Metal acceleration benefit. GPU wins at larger D where the
    // O(D^3) matvec parallelism outweighs dispatch overhead.
    // -----------------------------------------------------------------------
    println!("\n--- GPU vs CPU Comparison (L=12, varying D) ---");
    println!("{:>6} {:>10} {:>10} {:>10} {:>10}", "D", "GPU_ms", "CPU_ms", "speedup", "backend");
    println!("{}", "-".repeat(50));

    for &d in &[16, 32, 64] {
        // GPU mode (default threshold -- Metal dispatch for D >= gpu_threshold)
        let config_gpu = GpuDmrgConfig::new()
            .with_max_bond_dim(d)
            .with_max_sweeps(3)
            .with_energy_tolerance(1e-6)
            .with_two_site(true);
        let engine_gpu = GpuDmrgEngine::new(config_gpu);

        let t0 = std::time::Instant::now();
        let mut mps_gpu = GpuMps::random(12, 2, d.min(4), &mut rng);
        let result_gpu = engine_gpu.run(&mut mps_gpu, &MpoHamiltonian::ising(12, 1.0, 0.5));
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let backend = result_gpu.map(|r| format!("{}", r.backend)).unwrap_or_else(|_| "N/A".to_string());

        // CPU mode (force by setting threshold very high so Metal is never triggered)
        let config_cpu = GpuDmrgConfig::new()
            .with_max_bond_dim(d)
            .with_max_sweeps(3)
            .with_energy_tolerance(1e-6)
            .with_two_site(true)
            .with_gpu_threshold(usize::MAX); // Force CPU
        let engine_cpu = GpuDmrgEngine::new(config_cpu);

        let t0 = std::time::Instant::now();
        let mut mps_cpu = GpuMps::random(12, 2, d.min(4), &mut rng);
        let _ = engine_cpu.run(&mut mps_cpu, &MpoHamiltonian::ising(12, 1.0, 0.5));
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };
        println!("{:>6} {:>10.1} {:>10.1} {:>10.2}x {:>10}", d, gpu_ms, cpu_ms, speedup, backend);
    }

    // -----------------------------------------------------------------------
    // Section 7: Timing breakdown per sample run.
    // Shows variance across the MEASURE_RUNS samples to assess stability.
    // High variance may indicate memory allocation jitter or GC pressure.
    // -----------------------------------------------------------------------
    println!("\n--- Timing Variance (L=16, D=16, {} samples) ---", MEASURE_RUNS);
    let r_var = bench_dmrg_ising(16, 16, &mut rng);
    for (i, &t) in r_var.timing.samples.iter().enumerate() {
        println!("  Sample {}: {:.3} ms", i, t);
    }
    println!("  Mean:    {:.3} ms", r_var.timing.mean_ms);
    println!("  Min:     {:.3} ms", r_var.timing.min_ms);
    println!("  Max:     {:.3} ms", r_var.timing.max_ms);
    let spread = if r_var.timing.mean_ms > 0.0 {
        (r_var.timing.max_ms - r_var.timing.min_ms) / r_var.timing.mean_ms * 100.0
    } else {
        0.0
    };
    println!("  Spread:  {:.1}% of mean", spread);

    println!("\n[done]");
}
