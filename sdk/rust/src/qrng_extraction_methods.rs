//! QRNG Extraction Method Experiments
//!
//! Test different bit extraction and combination methods to improve NIST pass rates:
//! 1. XOR folding (combine high and low bits)
//! 2. Differential extraction (timing differences)
//! 3. Hash-based conditioning (SHA256, xxHash)
//! 4. Multi-source XOR combination
//! 5. Toeplitz matrix whitening
//! 6. Source-as-selector (one source controls which bits to take from another)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::nist_tests::{NistSuiteResult, NistTestSuite};

/// Configuration for extraction experiments
#[derive(Clone, Debug)]
pub struct ExtractionConfig {
    pub output_dir: PathBuf,
    pub n_samples: usize,
    pub lsb_count: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("experiments"),
            n_samples: 5_000,
            lsb_count: 8,
        }
    }
}

/// Result for a single extraction method
#[derive(Clone, Debug)]
pub struct ExtractionResult {
    pub method_name: String,
    pub bits: Vec<u8>,
    pub nist_result: Option<NistSuiteResult>,
    pub description: String,
}

// ============================================================================
// EXTRACTION METHODS
// ============================================================================

/// Method 1: Simple LSB extraction (baseline)
pub fn extract_lsb(values: &[u64], lsb_count: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(values.len() * lsb_count);
    for &v in values {
        for bit_idx in 0..lsb_count {
            bits.push(((v >> bit_idx) & 1) as u8);
        }
    }
    bits
}

/// Method 2: XOR folding - combine high and low bits
/// This can reduce bias by XORing bits that might be correlated
pub fn extract_xor_fold(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(values.len() * 4);

    for &v in values {
        // XOR upper 32 bits with lower 32 bits, take 4 bits
        let folded = (v ^ (v >> 32)) as u32;
        for bit_idx in 0..4 {
            bits.push(((folded >> bit_idx) & 1) as u8);
        }
    }
    bits
}

/// Method 3: Differential extraction - use differences between consecutive values
/// This removes slow drift and common-mode noise
pub fn extract_differential(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::with_capacity((values.len() - 1) * 4);

    for i in 1..values.len() {
        let diff = values[i].wrapping_sub(values[i - 1]);
        // Take middle bits of difference (where entropy is highest)
        for bit_idx in 4..8 {
            bits.push(((diff >> bit_idx) & 1) as u8);
        }
    }
    bits
}

/// Method 4: Hash-based conditioning using simple hasher
/// Cryptographic hash removes any remaining structure
pub fn extract_hash_conditioned(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::new();

    // Hash each value and extract bits from hash
    for &v in values {
        let mut hasher = DefaultHasher::new();
        v.hash(&mut hasher);
        let hash = hasher.finish();

        // Extract 8 bits from hash
        for bit_idx in 0..8 {
            bits.push(((hash >> bit_idx) & 1) as u8);
        }
    }
    bits
}

/// Method 5: Von Neumann debiasing (pairs: 01→0, 10→1, discard 00/11)
pub fn von_neumann_debias(bits: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bits.len() / 4);
    for chunk in bits.chunks(2) {
        if chunk.len() == 2 {
            match (chunk[0], chunk[1]) {
                (0, 1) => output.push(0),
                (1, 0) => output.push(1),
                _ => {}
            }
        }
    }
    output
}

/// Method 6: Toeplitz matrix whitening
/// Uses a random binary matrix to whiten the input
pub fn toeplitz_whiten(bits: &[u8], output_len: usize) -> Vec<u8> {
    if bits.len() < output_len {
        return bits.to_vec();
    }

    let mut output = Vec::with_capacity(output_len);

    // Simple Toeplitz: XOR bits[i] with bits[(i+k) % n] for various k
    for i in 0..output_len {
        let mut sum = 0u8;
        // XOR 8 bits together
        for k in 0..8 {
            let idx = (i + k * 13) % bits.len(); // Prime stride for better mixing
            sum ^= bits[idx];
        }
        output.push(sum);
    }
    output
}

/// Method 7: Multi-source XOR - combine bits from multiple sources
pub fn xor_combine(sources: &[&[u8]]) -> Vec<u8> {
    if sources.is_empty() {
        return Vec::new();
    }

    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    let mut result = Vec::with_capacity(min_len);

    for i in 0..min_len {
        let mut bit = 0u8;
        for source in sources {
            bit ^= source[i];
        }
        result.push(bit);
    }
    result
}

/// Method 8: Source-as-selector - use one source to select bits from another
/// If selector[i] == 1, take bit from source, otherwise skip
pub fn selector_extraction(selector: &[u8], source: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let min_len = selector.len().min(source.len());

    for i in 0..min_len {
        if selector[i] == 1 {
            output.push(source[i]);
        }
    }
    output
}

/// Method 9: Timing threshold - only take bits when timing crosses median
pub fn threshold_extraction(values: &[u64], lsb_count: usize) -> Vec<u8> {
    // Calculate median
    let mut sorted = values.to_vec();
    sorted.sort();
    let median = sorted[sorted.len() / 2];

    let mut bits = Vec::new();
    for &v in values {
        if v > median {
            // Only extract bits when above median
            for bit_idx in 0..lsb_count {
                bits.push(((v >> bit_idx) & 1) as u8);
            }
        }
    }
    bits
}

/// Method 10: Parity extraction - use parity of timing value
pub fn parity_extraction(values: &[u64]) -> Vec<u8> {
    values.iter().map(|&v| (v.count_ones() % 2) as u8).collect()
}

/// Method 11: Concatenate and hash - combine multiple timings, then hash
pub fn concat_hash_extraction(values: &[u64], chunk_size: usize) -> Vec<u8> {
    let mut bits = Vec::new();

    for chunk in values.chunks(chunk_size) {
        let mut hasher = DefaultHasher::new();
        for &v in chunk {
            v.hash(&mut hasher);
        }
        let hash = hasher.finish();

        // Extract all 64 bits from hash
        for bit_idx in 0..64 {
            bits.push(((hash >> bit_idx) & 1) as u8);
        }
    }
    bits
}

// ============================================================================
// DATA COLLECTION
// ============================================================================

/// Collect SSD timing data
pub fn collect_ssd_timings(n_samples: usize) -> Result<Vec<u64>, String> {
    use std::fs::OpenOptions;
    use std::io::Write as IoWrite;

    let test_file = "/tmp/ssd_extraction_test.bin";
    let data = vec![0x55u8; 4096];
    let mut timings = Vec::with_capacity(n_samples);

    // Warm-up
    for _ in 0..20 {
        if let Ok(mut file) = OpenOptions::new()
            .write(true).create(true).truncate(true).open(test_file)
        {
            let _ = file.write_all(&data);
        }
    }

    println!("  Collecting {} SSD samples...", n_samples);

    for _ in 0..n_samples {
        let start = Instant::now();
        if let Ok(mut file) = OpenOptions::new()
            .write(true).create(true).truncate(false).open(test_file)
        {
            let _ = file.write_all(&data);
            #[cfg(unix)]
            {
                
                let _ = file.sync_all();
            }
        }
        timings.push(start.elapsed().as_nanos() as u64);
    }

    let _ = std::fs::remove_file(test_file);
    Ok(timings)
}

/// Collect CPU jitter data
pub fn collect_cpu_jitter(n_samples: usize) -> Result<Vec<u64>, String> {
    let mut timings = Vec::with_capacity(n_samples);

    println!("  Collecting {} CPU jitter samples...", n_samples);

    for _ in 0..n_samples {
        let start = Instant::now();
        let mut sum: u64 = 0;
        for j in 0..100 {
            sum = sum.wrapping_add(j as u64);
        }
        std::hint::black_box(sum);
        timings.push(start.elapsed().as_nanos() as u64);
    }

    Ok(timings)
}

// ============================================================================
// EXPERIMENT RUNNER
// ============================================================================

/// Run all extraction method experiments
pub fn run_extraction_experiments(config: ExtractionConfig) -> Result<Vec<ExtractionResult>, String> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     QRNG EXTRACTION METHOD EXPERIMENTS                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Testing different bit extraction methods to improve NIST     ║");
    println!("║  pass rates for SSD timing QRNG                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Collect raw data
    println!("▶ COLLECTING RAW DATA...");
    let ssd_timings = collect_ssd_timings(config.n_samples)?;
    let cpu_timings = collect_cpu_jitter(config.n_samples)?;
    println!("  ✅ SSD: {} samples, CPU: {} samples", ssd_timings.len(), cpu_timings.len());
    println!();

    let mut results = Vec::new();
    let suite = NistTestSuite::new();

    // Method 1: Simple LSB (baseline)
    println!("▶ METHOD 1: Simple LSB Extraction (baseline)");
    let bits = extract_lsb(&ssd_timings, config.lsb_count);
    let debiased = von_neumann_debias(&bits);
    println!("  Raw: {} bits → Debiased: {} bits", bits.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "LSB + Von Neumann".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "Standard LSB extraction with Von Neumann debiasing".to_string(),
    });

    // Method 2: XOR Folding
    println!("\n▶ METHOD 2: XOR Folding");
    let bits = extract_xor_fold(&ssd_timings);
    let debiased = von_neumann_debias(&bits);
    println!("  Raw: {} bits → Debiased: {} bits", bits.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "XOR Folding + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "XOR upper/lower 32 bits, then Von Neumann".to_string(),
    });

    // Method 3: Differential
    println!("\n▶ METHOD 3: Differential Extraction");
    let bits = extract_differential(&ssd_timings);
    let debiased = von_neumann_debias(&bits);
    println!("  Raw: {} bits → Debiased: {} bits", bits.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Differential + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "Use timing differences between consecutive samples".to_string(),
    });

    // Method 4: Hash Conditioned
    println!("\n▶ METHOD 4: Hash Conditioning");
    let bits = extract_hash_conditioned(&ssd_timings);
    println!("  Hash-conditioned: {} bits", bits.len());

    let nist = if bits.len() >= 1000 {
        let result = suite.run_all_tests(&bits);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Hash Conditioned".to_string(),
        bits: bits.clone(),
        nist_result: nist,
        description: "Hash each timing value, extract bits from hash".to_string(),
    });

    // Method 5: Hash + VN
    println!("\n▶ METHOD 5: Hash + Von Neumann");
    let bits = extract_hash_conditioned(&ssd_timings);
    let debiased = von_neumann_debias(&bits);
    println!("  Raw: {} bits → Debiased: {} bits", bits.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Hash + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "Hash conditioning followed by Von Neumann".to_string(),
    });

    // Method 6: Toeplitz Whitening
    println!("\n▶ METHOD 6: Toeplitz Matrix Whitening");
    let raw_bits = extract_lsb(&ssd_timings, config.lsb_count);
    let bits = toeplitz_whiten(&raw_bits, 10000);
    println!("  Whitened: {} bits", bits.len());

    let nist = if bits.len() >= 1000 {
        let result = suite.run_all_tests(&bits);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Toeplitz Whitening".to_string(),
        bits,
        nist_result: nist,
        description: "XOR bits with prime-stride offsets for mixing".to_string(),
    });

    // Method 7: Multi-source XOR (SSD + CPU)
    println!("\n▶ METHOD 7: Multi-Source XOR (SSD ⊕ CPU)");
    let ssd_bits = extract_lsb(&ssd_timings, 4);
    let cpu_bits = extract_lsb(&cpu_timings, 4);
    let combined = xor_combine(&[&ssd_bits, &cpu_bits]);
    let debiased = von_neumann_debias(&combined);
    println!("  Combined: {} bits → Debiased: {} bits", combined.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "SSD ⊕ CPU XOR + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "XOR combine SSD and CPU timing bits".to_string(),
    });

    // Method 8: Source-as-Selector
    println!("\n▶ METHOD 8: CPU-as-Selector for SSD bits");
    let ssd_bits = extract_lsb(&ssd_timings, config.lsb_count);
    let cpu_bits = extract_lsb(&cpu_timings, config.lsb_count);
    let selected = selector_extraction(&cpu_bits, &ssd_bits);
    let debiased = von_neumann_debias(&selected);
    println!("  Selected: {} bits → Debiased: {} bits", selected.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "CPU→SSD Selector + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "CPU bits select which SSD bits to keep".to_string(),
    });

    // Method 9: Threshold Extraction
    println!("\n▶ METHOD 9: Threshold (Above-Median) Extraction");
    let bits = threshold_extraction(&ssd_timings, 4);
    let debiased = von_neumann_debias(&bits);
    println!("  Threshold bits: {} → Debiased: {} bits", bits.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Threshold + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "Only extract bits when timing > median".to_string(),
    });

    // Method 10: Parity Extraction
    println!("\n▶ METHOD 10: Parity Extraction");
    let bits = parity_extraction(&ssd_timings);
    let debiased = von_neumann_debias(&bits);
    println!("  Parity bits: {} → Debiased: {} bits", bits.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Parity + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "Use parity (XOR all bits) of timing value".to_string(),
    });

    // Method 11: Concat Hash (4 values → 1 hash → 64 bits)
    println!("\n▶ METHOD 11: Concatenated Hash (4 timings → 64 bits)");
    let bits = concat_hash_extraction(&ssd_timings, 4);
    println!("  Concat-hashed: {} bits", bits.len());

    let nist = if bits.len() >= 1000 {
        let result = suite.run_all_tests(&bits);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Concat Hash (4→64)".to_string(),
        bits,
        nist_result: nist,
        description: "Hash 4 timings together, extract 64 bits".to_string(),
    });

    // Method 12: Triple combination (SSD + CPU + Differential)
    println!("\n▶ METHOD 12: Triple Combination (SSD ⊕ CPU ⊕ Diff)");
    let ssd_bits = extract_lsb(&ssd_timings, 4);
    let cpu_bits = extract_lsb(&cpu_timings, 4);
    let diff_bits = extract_differential(&ssd_timings);
    let min_len = ssd_bits.len().min(cpu_bits.len()).min(diff_bits.len());
    let combined = xor_combine(&[
        &ssd_bits[..min_len],
        &cpu_bits[..min_len],
        &diff_bits[..min_len],
    ]);
    let debiased = von_neumann_debias(&combined);
    println!("  Triple-XOR: {} bits → Debiased: {} bits", combined.len(), debiased.len());

    let nist = if debiased.len() >= 1000 {
        let result = suite.run_all_tests(&debiased);
        println!("  NIST: {}/15 passed", result.passed_count);
        Some(result)
    } else { None };

    results.push(ExtractionResult {
        method_name: "Triple XOR + VN".to_string(),
        bits: debiased,
        nist_result: nist,
        description: "XOR SSD + CPU + Differential bits".to_string(),
    });

    // Print summary
    print_extraction_summary(&results);

    // Save results
    save_extraction_results(&results, &config)?;

    Ok(results)
}

fn print_extraction_summary(results: &[ExtractionResult]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    EXTRACTION METHOD COMPARISON                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Method                    │ Bits    │ NIST Pass │ Min-Ent  │ Rating ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let mut best_pass = 0;
    let mut best_entropy = 0.0;

    for result in results {
        let (pass_str, ent_str, rating) = if let Some(ref nist) = result.nist_result {
            if nist.passed_count > best_pass {
                best_pass = nist.passed_count;
            }
            if nist.min_entropy > best_entropy {
                best_entropy = nist.min_entropy;
            }
            let rating = if nist.passed_count >= 14 {
                "🏆 BEST"
            } else if nist.passed_count >= 12 {
                "✅ GOOD"
            } else if nist.passed_count >= 10 {
                "⚠️ OK"
            } else {
                "❌ POOR"
            };
            (
                format!("{}/15 ({:.0}%)", nist.passed_count, nist.passed_count as f64 / 15.0 * 100.0),
                format!("{:.4}", nist.min_entropy),
                rating,
            )
        } else {
            ("N/A".to_string(), "N/A".to_string(), "⏭ SKIP")
        };

        println!(
            "║ {:<25} │ {:>7} │ {:>9} │ {:>8} │ {:>6} ║",
            result.method_name,
            result.bits.len(),
            pass_str,
            ent_str,
            rating
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Best NIST Pass Rate: {}/15", best_pass);
    println!("Best Min-Entropy: {:.4} bits/bit", best_entropy);
}

fn save_extraction_results(results: &[ExtractionResult], config: &ExtractionConfig) -> Result<(), String> {
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("Cannot create output dir: {}", e))?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let report_path = config.output_dir.join(format!("EXTRACTION_METHODS_{}.md", timestamp));
    let mut report = String::new();

    report.push_str("# QRNG Extraction Method Experiments\n\n");
    report.push_str(&format!("**Date**: {}\n\n", timestamp));
    report.push_str("## Methods Tested\n\n");
    report.push_str("| Method | Bits | NIST Pass | Min-Entropy | Rating |\n");
    report.push_str("|--------|------|-----------|-------------|--------|\n");

    for result in results {
        let (pass_str, ent_str, rating) = if let Some(ref nist) = result.nist_result {
            let rating = if nist.passed_count >= 14 { "🏆 BEST" }
            else if nist.passed_count >= 12 { "✅ GOOD" }
            else if nist.passed_count >= 10 { "⚠️ OK" }
            else { "❌ POOR" };
            (
                format!("{}/15", nist.passed_count),
                format!("{:.4}", nist.min_entropy),
                rating,
            )
        } else {
            ("N/A".to_string(), "N/A".to_string(), "⏭ SKIP")
        };

        report.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            result.method_name, result.bits.len(), pass_str, ent_str, rating
        ));
    }

    report.push_str("\n## Descriptions\n\n");
    for result in results {
        report.push_str(&format!("**{}**: {}\n", result.method_name, result.description));
    }

    std::fs::write(&report_path, &report)
        .map_err(|e| format!("Cannot write report: {}", e))?;

    println!("\n📄 Report saved to: {}", report_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_methods() {
        let test_values: Vec<u64> = (0..100).map(|i| i * 1000 + (i % 7) * 100).collect();

        // Test each method produces output
        assert!(!extract_lsb(&test_values, 8).is_empty());
        assert!(!extract_xor_fold(&test_values).is_empty());
        assert!(!extract_differential(&test_values).is_empty());
        assert!(!extract_hash_conditioned(&test_values).is_empty());
        assert!(!parity_extraction(&test_values).is_empty());
    }
}
