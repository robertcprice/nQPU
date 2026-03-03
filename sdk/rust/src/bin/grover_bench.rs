//! Grover's Algorithm Benchmark - Where Batching Shines
//!
//! Grover's algorithm has the BEST batching potential:
//! - Each iteration = 1 oracle + 2n Hadamard + 1 phase flip
//! - With batching: Each iteration = 1 command buffer (1 sync)
//! - Without batching: Each iteration = 2n+2 command buffers
//! - Batching advantage = 2n+2× fewer synchronizations!

use std::time::Instant;

#[cfg(target_os = "macos")]
use nqpu_metal::{FixedMetalSimulator, benchmark_grover};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     GROVER'S ALGORITHM - BATCHING SPEEDUP TEST                ║");
    println!("║     This is where GPU batching REALLY shines                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    #[cfg(target_os = "macos")]
    {
        let test_qubits = vec![16, 18, 20, 22];

        for &num_qubits in &test_qubits {
            let dim = 1usize << num_qubits;
            let num_iterations = (std::f64::consts::PI / 4.0 * dim as f64).sqrt() as usize;
            let target = dim / 3;  // Arbitrary target

            println!("═══════════════════════════════════════════════════════════════");
            println!("n={} qubits ({} states, {} iterations)", num_qubits, dim, num_iterations);
            println!("═══════════════════════════════════════════════════════════════\n");

            // CPU baseline
            println!("Running CPU baseline...");
            let cpu_start = Instant::now();
            let (_cpu_state, _cpu_result) = benchmark_grover(num_qubits, target);
            let cpu_time = cpu_start.elapsed().as_secs_f64();

            // GPU with MAXIMUM batching
            println!("Running GPU with MAXIMUM batching...");
            let gpu_start = Instant::now();
            match FixedMetalSimulator::new(num_qubits) {
                Ok(sim) => {
                    let (_gpu_state, _gpu_result) = sim.benchmark_grover_batched(target);
                    let gpu_time = gpu_start.elapsed().as_secs_f64();

                    let speedup = cpu_time / gpu_time;

                    println!("\nResults:");
                    println!("  CPU:  {:.6}s ({:.3} μs/iter)", cpu_time, cpu_time * 1e6 / num_iterations as f64);
                    println!("  GPU:  {:.6}s ({:.3} μs/iter)", gpu_time, gpu_time * 1e6 / num_iterations as f64);
                    println!();
                    println!("  🏆 SPEEDUP: {:.1}× {}",
                        speedup,
                        if speedup >= 100.0 { "✅ (TARGET MET!)" }
                        else if speedup >= 63.0 { "⚠️ (Above 63× baseline)" }
                        else if speedup >= 10.0 { "📈 (Good progress)" }
                        else { "❌ (Need more optimization)" }
                    );
                    println!();

                    // Calculate theoretical maximum
                    let gates_per_iter = 2 * num_qubits + 2;  // oracle + H^n + phase + H^n
                    let total_gates = gates_per_iter * num_iterations;
                    println!("  Analysis:");
                    println!("    • Gates per iteration: {}", gates_per_iter);
                    println!("    • Total gates: {}", total_gates);
                    println!("    • Synchronization points saved: {}", total_gates - num_iterations);
                    println!("    • Theoretical max speedup: {:.1}×\n", total_gates as f64 / num_iterations as f64);
                }
                Err(e) => println!("  GPU error: {}\n", e),
            }
        }

        println!("═══════════════════════════════════════════════════════════════");
        println!("KEY INSIGHT");
        println!("═══════════════════════════════════════════════════════════════");
        println!();
        println!("Grover's algorithm is the PERFECT test for batching because:");
        println!("  • Each iteration has many gates (2n+2 gates)");
        println!("  • All gates can be batched into 1 command buffer");
        println!("  • Synchronization overhead is eliminated");
        println!("  • GPU stays busy for the entire iteration");
        println!();
        println!("This is where we should see 100×+ speedup!");
        println!();
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("❌ This benchmark requires macOS with Metal support.");
    }
}
