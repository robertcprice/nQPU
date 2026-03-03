//! Fast Stabilizer Benchmark Binary
//!
//! Measures gates/sec throughput for cache-optimized stabilizer simulation.
//! Target: 50M gates/sec (Stim-competitive on Apple Silicon)

fn main() {
    println!("{}", "=".repeat(80));
    println!("nQPU-Metal Fast Stabilizer Benchmark");
    println!("Target: 50M gates/sec (Stim-competitive)");
    println!("{}", "=".repeat(80));
    println!();

    nqpu_metal::fast_stabilizer::print_full_benchmark();
}
