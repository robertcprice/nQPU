//! QRNG Source Comparison Runner
//!
//! Usage: cargo run --release --bin qrng_source_compare [--samples N]

use nqpu_metal::qrng_source_comparison::{run_source_comparison, SourceConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse sample count from args
    let n_samples = if args.len() > 2 && args[1] == "--samples" {
        args[2].parse().unwrap_or(10_000)
    } else {
        10_000 // Default
    };

    let config = SourceConfig {
        n_samples,
        debias: true,
        lsb_count: 8,
        output_dir: PathBuf::from("experiments"),
    };

    println!(
        "Running QRNG source comparison with {} samples per source...",
        n_samples
    );
    println!();

    match run_source_comparison(config) {
        Ok(results) => {
            println!(
                "\n✅ Comparison complete with {} sources tested",
                results.len()
            );
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    }
}
