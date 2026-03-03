//! S-Tier Benchmark: demonstrates all performance optimizations
//!
//! Compares CPU sequential vs CPU fused (SIMD + parallel + gate fusion)
//! vs Metal GPU on QFT, Random, Grover, and Bell circuits.
//! Uses warmup + median-of-3 for stable results.

use nqpu_metal::benchmark_suite::{run_benchmark, qft_circuit, random_circuit, grover_circuit, bell_circuit};

fn main() {
    println!("nQPU-Metal S-Tier Performance Benchmark");
    println!("========================================");
    println!("NEON SIMD | Gate Fusion | Parallel 2Q | Adaptive Stride");
    println!();

    let configs: Vec<(&str, Vec<nqpu_metal::gates::Gate>, usize)> = vec![
        ("QFT-10", qft_circuit(10), 10),
        ("QFT-15", qft_circuit(15), 15),
        ("QFT-20", qft_circuit(20), 20),
        ("Random-10", random_circuit(10, 20, 42), 10),
        ("Random-15", random_circuit(15, 20, 42), 15),
        ("Random-20", random_circuit(20, 20, 42), 20),
        ("Grover-10", grover_circuit(10, 512), 10),
        ("Grover-14", grover_circuit(14, 8192), 14),
        ("Bell-10", bell_circuit(10), 10),
        ("Bell-20", bell_circuit(20), 20),
    ];

    // Warmup run (not printed)
    for (name, gates, n) in &configs {
        let _ = run_benchmark(name, gates, *n);
    }

    println!("{:<12} {:>3} {:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>10}",
             "Circuit", "n", "Gates", "CPU(ms)", "Fused(ms)", "GPU(ms)",
             "Fuse/CPU", "GPU/CPU", "Fidelity");
    println!("{}", "-".repeat(100));

    let mut total_cpu = 0.0_f64;
    let mut total_fused = 0.0_f64;
    let mut total_gpu = 0.0_f64;
    let mut min_fidelity = 1.0_f64;

    for (name, gates, n) in &configs {
        // Median of 3 runs
        let mut results: Vec<_> = (0..3).map(|_| run_benchmark(name, gates, *n)).collect();
        results.sort_by(|a, b| a.fused_time_ms.partial_cmp(&b.fused_time_ms).unwrap());
        let r = &results[1]; // median

        let cpu_ms = r.sequential_time_ms;
        let fused_ms = r.fused_time_ms;
        let best_cpu = cpu_ms.min(fused_ms);
        total_cpu += cpu_ms;
        total_fused += fused_ms;

        if r.fidelity < min_fidelity {
            min_fidelity = r.fidelity;
        }

        let gpu_str = if r.gpu_time_ms > 0.0 {
            total_gpu += r.gpu_time_ms;
            format!("{:>10.3}", r.gpu_time_ms)
        } else {
            "N/A".to_string()
        };

        let fuse_ratio = if fused_ms > 0.0 { cpu_ms / fused_ms } else { 1.0 };
        let gpu_ratio = if r.gpu_time_ms > 0.0 {
            format!("{:>8.1}x", best_cpu / r.gpu_time_ms)
        } else {
            "N/A".to_string()
        };

        println!("{:<12} {:>3} {:>6} {:>10.3} {:>10.3} {:>10} {:>8.2}x {:>9} {:>10.8}",
                 r.name, r.num_qubits, r.num_gates,
                 cpu_ms, fused_ms, gpu_str,
                 fuse_ratio, gpu_ratio, r.fidelity);
    }

    println!("{}", "-".repeat(100));
    let overall_fuse = if total_fused > 0.0 { total_cpu / total_fused } else { 1.0 };
    let overall_gpu = if total_gpu > 0.0 {
        format!("{:.1}x", total_cpu.min(total_fused) / total_gpu)
    } else {
        "N/A".to_string()
    };
    println!("TOTAL {:>14} {:>10.1} {:>10.1} {:>10.1} {:>8.2}x {:>9}",
             "", total_cpu, total_fused, total_gpu, overall_fuse, overall_gpu);
    println!();
    println!("Fidelity: {:.10} {}", min_fidelity,
             if min_fidelity > 1.0 - 1e-10 { "PERFECT" } else { "DEGRADED" });
    println!("Fuse/CPU > 1.0 = fusion wins | GPU/CPU = Metal GPU speedup over best CPU");
}
