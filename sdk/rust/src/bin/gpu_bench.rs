//! GPU vs CPU Benchmark
//!
//! Runs the full benchmark suite comparing Metal GPU to CPU execution.

fn main() {
    #[cfg(target_os = "macos")]
    {
        let results = nqpu_metal::metal_backend::run_gpu_suite();
        nqpu_metal::metal_backend::print_gpu_results(&results);
    }

    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("Metal GPU benchmarks require macOS");
    }
}
