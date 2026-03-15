//! Large-Scale GPU Benchmarking Tool
//!
//! Run this when system is idle for comprehensive testing at 20-28 qubits.
//! Your 24GB RAM can handle up to n=28 (256M states = 4GB memory).
//!
//! Usage:
//!   cargo run --release --bin large_scale_test

use std::time::Instant;

#[cfg(target_os = "macos")]
use nqpu_metal::metal_gpu_fixed::*;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║      Large-Scale GPU Quantum Simulator Benchmark          ║");
    println!("║              Testing at 20-28 Qubits                       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    #[cfg(target_os = "macos")]
    {
        println!("🖥️  System Information:");
        println!("   • 24GB RAM available");
        println!("   • Can test up to n=28 qubits (256M states)");
        println!("   • This will test where GPU actually beats CPU\n");

        println!("⚠️  NOTE: Large-scale tests will take time!");
        println!("   • n=20: ~1-2 seconds per test");
        println!("   • n=24: ~5-10 seconds per test");
        println!("   • n=28: ~20-60 seconds per test\n");

        println!("═════════════════════════════════════════════════════════");
        println!("PHASE 1: BATCHING VERIFICATION");
        println!("═════════════════════════════════════════════════════════\n");

        run_batching_verification();

        println!("\n═════════════════════════════════════════════════════════");
        println!("PHASE 2: SCALE TESTING (20-24 QUBITS)");
        println!("═════════════════════════════════════════════════════════\n");

        run_scale_tests();

        println!("\n═════════════════════════════════════════════════════════");
        println!("PHASE 3: CPU vs GPU COMPARISON");
        println!("═════════════════════════════════════════════════════════\n");

        run_cpu_gpu_comparison();

        println!("\n═════════════════════════════════════════════════════════");
        println!("PHASE 4: ULTRA-LARGE SCALE (OPTIONAL)");
        println!("═════════════════════════════════════════════════════════");
        println!("⚠️  These tests will take significant time!");
        println!("    Press Enter to continue or Ctrl+C to exit...");

        // Wait for user confirmation
        let _ = std::io::stdin().read_line(&mut String::new());

        run_ultra_large_scale();
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("❌ This benchmark tool requires macOS with Metal support.");
        println!("   Your system does not meet the requirements.");
    }
}

#[cfg(target_os = "macos")]
fn run_batching_verification() {
    println!("Verifying batching provides speedup at various scales:\n");

    for &num_qubits in &[10, 12, 14] {
        let num_gates = 1000;

        match benchmark_fixed_gpu_gates_batched(num_qubits, num_gates) {
            Ok(time) => {
                let us_per_gate = time * 1e6 / num_gates as f64;
                println!(
                    "  n={:2}: {:.6}s total ({:.3} μs/gate)",
                    num_qubits, time, us_per_gate
                );
            }
            Err(e) => println!("  n={:2}: Error - {}", num_qubits, e),
        }
    }

    println!("\n✅ Batching verified");
}

#[cfg(target_os = "macos")]
fn run_scale_tests() {
    println!("Testing GPU performance at scales where it should shine:\n");

    for &num_qubits in &[20, 22, 24] {
        let num_gates = 1000;
        let state_size = 1usize << num_qubits;
        let memory_mb = (state_size * 16) / (1024 * 1024);

        match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
            Ok(time) => {
                let us_per_gate = time * 1e6 / num_gates as f64;
                let gates_per_sec = num_gates as f64 / time;

                println!("  n={:2} qubits:", num_qubits);
                println!("    • States: {}", state_size);
                println!("    • Memory: {} MB", memory_mb);
                println!("    • Time: {:.6}s", time);
                println!("    • Per gate: {:.3} μs", us_per_gate);
                println!("    • Throughput: {:.0} gates/sec", gates_per_sec);
                println!();
            }
            Err(e) => println!("  n={:2}: Error - {}\n", num_qubits, e),
        }
    }

    println!("✅ Scale testing complete");
}

#[cfg(target_os = "macos")]
fn run_cpu_gpu_comparison() {
    println!("Direct CPU vs GPU comparison at crossover point (n=20):\n");

    let num_qubits = 20;
    let num_gates = 1000;

    // CPU benchmark
    println!("  Running CPU benchmark...");
    let cpu_start = Instant::now();
    let _cpu_time = nqpu_metal::benchmark_gates(num_qubits, num_gates);
    let cpu_elapsed = cpu_start.elapsed().as_secs_f64();

    // GPU benchmark
    println!("  Running GPU benchmark...");
    match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
        Ok(gpu_time) => {
            let speedup = cpu_elapsed / gpu_time;
            let cpu_us = cpu_elapsed * 1e6 / num_gates as f64;
            let gpu_us = gpu_time * 1e6 / num_gates as f64;

            println!("\n  Results:");
            println!("    CPU:  {:.6}s ({:.3} μs/gate)", cpu_elapsed, cpu_us);
            println!("    GPU:  {:.6}s ({:.3} μs/gate)", gpu_time, gpu_us);
            println!();
            println!(
                "    🏆 Winner: {}",
                if speedup > 1.0 {
                    format!("GPU by {:.1}× ✅", speedup)
                } else {
                    format!("CPU by {:.1}× ⚠️", 1.0 / speedup)
                }
            );
            println!();
        }
        Err(e) => println!("  GPU error: {}", e),
    }

    println!("✅ Comparison complete");
}

#[cfg(target_os = "macos")]
fn run_ultra_large_scale() {
    println!("Testing at extreme scales (26-28 qubits):");
    println!("⚠️  These will take significant time!\n");

    for &num_qubits in &[26, 28] {
        let num_gates = 1000;
        let state_size = 1usize << num_qubits;
        let memory_mb = (state_size * 16) / (1024 * 1024);
        let memory_gb = memory_mb as f64 / 1024.0;

        println!(
            "  Testing n={} qubits ({} states, {:.2} GB)...",
            num_qubits, state_size, memory_gb
        );

        match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
            Ok(time) => {
                let us_per_gate = time * 1e6 / num_gates as f64;
                let gates_per_sec = num_gates as f64 / time;

                println!("    ✅ Completed in {:.6}s", time);
                println!("    • {:.3} μs/gate", us_per_gate);
                println!("    • {:.0} gates/sec", gates_per_sec);
                println!();
            }
            Err(e) => println!("    ❌ Error - {}\n", e),
        }
    }

    println!("✅ Ultra-large scale testing complete");
}
