//! QRNG Extraction Methods Runner
//!
//! Usage: cargo run --release --bin qrng_extraction_test [--samples N]

use nqpu_metal::qrng_extraction_methods::{run_extraction_experiments, ExtractionConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let n_samples = if args.len() > 2 && args[1] == "--samples" {
        args[2].parse().unwrap_or(5_000)
    } else {
        5_000
    };

    let config = ExtractionConfig {
        n_samples,
        lsb_count: 8,
        output_dir: PathBuf::from("experiments"),
    };

    println!(
        "Testing {} extraction methods with {} samples...\n",
        12, n_samples
    );

    match run_extraction_experiments(config) {
        Ok(results) => {
            // Find best method
            let best = results
                .iter()
                .filter_map(|r| {
                    r.nist_result
                        .as_ref()
                        .map(|n| (r.method_name.clone(), n.passed_count, n.min_entropy))
                })
                .max_by_key(|(_, pass, _)| *pass);

            if let Some((name, pass, entropy)) = best {
                println!(
                    "\n🏆 BEST METHOD: {} with {}/15 NIST passes ({:.4} entropy)",
                    name, pass, entropy
                );
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    }
}
