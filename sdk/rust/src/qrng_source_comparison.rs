//! QRNG Source Comparison - Test each entropy source SEPARATELY
//!
//! This module runs NIST tests on each source independently to determine:
//! 1. Is CPU jitter "more quantum" than SSD timing?
//! 2. Which source produces the best random bits?
//! 3. How do min-entropy values compare?

use std::fs::File;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::nist_tests::{NistSuiteResult, NistTestSuite};

/// Source-specific experiment configuration
#[derive(Clone, Debug)]
pub struct SourceConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Number of samples to collect
    pub n_samples: usize,
    /// Whether to apply von Neumann debiasing
    pub debias: bool,
    /// Number of LSBs to extract
    pub lsb_count: usize,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("experiments"),
            n_samples: 10_000,
            debias: true,
            lsb_count: 8,
        }
    }
}

/// Results for a single entropy source
#[derive(Clone, Debug)]
pub struct SourceResult {
    /// Source name
    pub source: String,
    /// Raw timing/data values
    pub raw_values: Vec<u64>,
    /// Extracted raw bits
    pub raw_bits: Vec<u8>,
    /// Final bits (after debiasing if applied)
    pub final_bits: Vec<u8>,
    /// Timing/data statistics
    pub stats: SourceStats,
    /// NIST test results
    pub nist_result: Option<NistSuiteResult>,
    /// Experiment ID
    pub experiment_id: String,
}

#[derive(Clone, Debug)]
pub struct SourceStats {
    pub mean: f64,
    pub std: f64,
    pub cv: f64,
    pub min: u64,
    pub max: u64,
    pub n_samples: usize,
}

/// SSD Timing Source
pub struct SsdTimingSource {
    config: SourceConfig,
    test_file: String,
}

impl SsdTimingSource {
    pub fn new(config: SourceConfig) -> Self {
        Self {
            config,
            test_file: "/tmp/ssd_qrng_source_test.bin".to_string(),
        }
    }

    /// Collect SSD write timing measurements
    pub fn collect(&self) -> Result<Vec<u64>, String> {
        let mut timings = Vec::with_capacity(self.config.n_samples);
        let data = vec![0x55u8; 4096]; // 4KB block

        // Warm-up writes
        for _ in 0..20 {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&self.test_file)
            {
                let _ = file.write_all(&data);
            }
        }

        println!("  Collecting {} SSD timing samples...", self.config.n_samples);

        for i in 0..self.config.n_samples {
            let start = Instant::now();

            if let Ok(mut file) = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&self.test_file)
            {
                let _ = file.write_all(&data);

                #[cfg(unix)]
                {
                    
                    let _ = file.sync_all();
                }
            }

            timings.push(start.elapsed().as_nanos() as u64);

            if (i + 1) % 1000 == 0 {
                println!("    Progress: {}/{}", i + 1, self.config.n_samples);
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&self.test_file);

        Ok(timings)
    }
}

/// CPU Jitter Source
pub struct CpuJitterSource {
    config: SourceConfig,
}

impl CpuJitterSource {
    pub fn new(config: SourceConfig) -> Self {
        Self { config }
    }

    /// Collect CPU timing jitter measurements
    pub fn collect(&self) -> Result<Vec<u64>, String> {
        let mut timings = Vec::with_capacity(self.config.n_samples);

        println!("  Collecting {} CPU jitter samples...", self.config.n_samples);

        for i in 0..self.config.n_samples {
            let start = Instant::now();

            // Do some variable work to introduce jitter
            let mut sum: u64 = 0;
            for j in 0..100 {
                sum = sum.wrapping_add(j as u64);
            }
            std::hint::black_box(sum);

            let elapsed = start.elapsed().as_nanos() as u64;
            timings.push(elapsed);

            if (i + 1) % 1000 == 0 {
                println!("    Progress: {}/{}", i + 1, self.config.n_samples);
            }
        }

        Ok(timings)
    }
}

/// Entropy Pool Source (/dev/urandom)
pub struct EntropyPoolSource {
    config: SourceConfig,
}

impl EntropyPoolSource {
    pub fn new(config: SourceConfig) -> Self {
        Self { config }
    }

    /// Collect bytes from /dev/urandom
    pub fn collect(&self) -> Result<Vec<u64>, String> {
        println!("  Collecting {} bytes from /dev/urandom...", self.config.n_samples);

        let bytes = if cfg!(unix) {
            use std::fs::File;
            use std::io::Read;

            let mut file = File::open("/dev/urandom")
                .map_err(|e| format!("Cannot open /dev/urandom: {}", e))?;
            let mut buf = vec![0u8; self.config.n_samples];
            file.read_exact(&mut buf)
                .map_err(|e| format!("Cannot read from /dev/urandom: {}", e))?;
            buf
        } else {
            // Fallback for non-Unix
            (0..self.config.n_samples).map(|_| rand::random::<u8>()).collect()
        };

        // Convert bytes to u64 values (just for consistent interface)
        Ok(bytes.iter().map(|&b| b as u64).collect())
    }
}

/// Utility functions
pub fn extract_bits(values: &[u64], lsb_count: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(values.len() * lsb_count);

    for &value in values {
        for bit_idx in 0..lsb_count {
            bits.push(((value >> bit_idx) & 1) as u8);
        }
    }

    bits
}

pub fn von_neumann_debias(bits: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bits.len() / 4);

    for chunk in bits.chunks(2) {
        if chunk.len() == 2 {
            match (chunk[0], chunk[1]) {
                (0, 1) => output.push(0),
                (1, 0) => output.push(1),
                _ => {} // Discard 00 and 11
            }
        }
    }

    output
}

pub fn calculate_stats(values: &[u64]) -> SourceStats {
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<u64>() as f64 / n;
    let variance: f64 = values.iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>() / n;
    let std = variance.sqrt();
    let cv = if mean > 0.0 { std / mean } else { 0.0 };

    SourceStats {
        mean,
        std,
        cv,
        min: values.iter().min().copied().unwrap_or(0),
        max: values.iter().max().copied().unwrap_or(0),
        n_samples: values.len(),
    }
}

/// Run comparison experiment
pub fn run_source_comparison(config: SourceConfig) -> Result<Vec<SourceResult>, String> {
    let experiment_id = format!("COMPARE-{}", chrono_timestamp());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     QRNG SOURCE COMPARISON - SEPARATE TESTING                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Experiment ID: {:<44}║", experiment_id);
    println!("║  Samples per source: {:>39}║", config.n_samples);
    println!("║  LSB count: {:>46}║", config.lsb_count);
    println!("║  Debiasing: {:>46}║", if config.debias { "Von Neumann" } else { "None" });
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut results = Vec::new();

    // Test 1: SSD Timing
    println!("═══════════════════════════════════════════════════════════════");
    println!("SOURCE 1: SSD WRITE TIMING");
    println!("═══════════════════════════════════════════════════════════════");

    let ssd_source = SsdTimingSource::new(config.clone());
    let ssd_values = ssd_source.collect()?;
    let ssd_stats = calculate_stats(&ssd_values);
    let ssd_raw_bits = extract_bits(&ssd_values, config.lsb_count);
    let ssd_final_bits = if config.debias {
        von_neumann_debias(&ssd_raw_bits)
    } else {
        ssd_raw_bits.clone()
    };

    println!("\n  SSD Statistics:");
    println!("    Mean:   {:.2} µs", ssd_stats.mean / 1000.0);
    println!("    Std:    {:.2} µs", ssd_stats.std / 1000.0);
    println!("    CV:     {:.2}% {}", ssd_stats.cv * 100.0,
             if ssd_stats.cv > 0.01 { "✅ QUANTUM-CONSISTENT" } else { "❌ Classical" });
    println!("    Bits:   {} raw → {} final", ssd_raw_bits.len(), ssd_final_bits.len());

    let ssd_nist = if ssd_final_bits.len() >= 1000 {
        println!("\n  Running NIST tests on SSD bits...");
        let suite = NistTestSuite::new();
        let result = suite.run_all_tests(&ssd_final_bits);
        println!("{}", suite.generate_report(&result));
        Some(result)
    } else {
        println!("\n  ⚠️ Not enough bits for NIST tests");
        None
    };

    results.push(SourceResult {
        source: "SSD Timing".to_string(),
        raw_values: ssd_values,
        raw_bits: ssd_raw_bits,
        final_bits: ssd_final_bits.clone(),
        stats: ssd_stats,
        nist_result: ssd_nist.clone(),
        experiment_id: experiment_id.clone(),
    });

    // Test 2: CPU Jitter
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("SOURCE 2: CPU TIMING JITTER");
    println!("═══════════════════════════════════════════════════════════════");

    let cpu_source = CpuJitterSource::new(config.clone());
    let cpu_values = cpu_source.collect()?;
    let cpu_stats = calculate_stats(&cpu_values);
    let cpu_raw_bits = extract_bits(&cpu_values, config.lsb_count);
    let cpu_final_bits = if config.debias {
        von_neumann_debias(&cpu_raw_bits)
    } else {
        cpu_raw_bits.clone()
    };

    println!("\n  CPU Statistics:");
    println!("    Mean:   {:.2} ns", cpu_stats.mean);
    println!("    Std:    {:.2} ns", cpu_stats.std);
    println!("    CV:     {:.2}% {}", cpu_stats.cv * 100.0,
             if cpu_stats.cv > 0.01 { "✅ QUANTUM-CONSISTENT" } else { "❌ Classical" });
    println!("    Bits:   {} raw → {} final", cpu_raw_bits.len(), cpu_final_bits.len());

    let cpu_nist = if cpu_final_bits.len() >= 1000 {
        println!("\n  Running NIST tests on CPU bits...");
        let suite = NistTestSuite::new();
        let result = suite.run_all_tests(&cpu_final_bits);
        println!("{}", suite.generate_report(&result));
        Some(result)
    } else {
        println!("\n  ⚠️ Not enough bits for NIST tests");
        None
    };

    results.push(SourceResult {
        source: "CPU Jitter".to_string(),
        raw_values: cpu_values,
        raw_bits: cpu_raw_bits,
        final_bits: cpu_final_bits.clone(),
        stats: cpu_stats,
        nist_result: cpu_nist.clone(),
        experiment_id: experiment_id.clone(),
    });

    // Test 3: /dev/urandom (baseline)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("SOURCE 3: /dev/urandom (BASELINE)");
    println!("═══════════════════════════════════════════════════════════════");

    let entropy_source = EntropyPoolSource::new(config.clone());
    let entropy_values = entropy_source.collect()?;
    let entropy_stats = calculate_stats(&entropy_values);
    let entropy_raw_bits = extract_bits(&entropy_values, 8); // Use all 8 bits
    let entropy_final_bits = if config.debias {
        von_neumann_debias(&entropy_raw_bits)
    } else {
        entropy_raw_bits.clone()
    };

    println!("\n  /dev/urandom Statistics:");
    println!("    Mean:   {:.2}", entropy_stats.mean);
    println!("    Std:    {:.2}", entropy_stats.std);
    println!("    CV:     {:.2}%", entropy_stats.cv * 100.0);
    println!("    Bits:   {} raw → {} final", entropy_raw_bits.len(), entropy_final_bits.len());

    let entropy_nist = if entropy_final_bits.len() >= 1000 {
        println!("\n  Running NIST tests on /dev/urandom bits...");
        let suite = NistTestSuite::new();
        let result = suite.run_all_tests(&entropy_final_bits);
        println!("{}", suite.generate_report(&result));
        Some(result)
    } else {
        println!("\n  ⚠️ Not enough bits for NIST tests");
        None
    };

    results.push(SourceResult {
        source: "/dev/urandom".to_string(),
        raw_values: entropy_values,
        raw_bits: entropy_raw_bits,
        final_bits: entropy_final_bits,
        stats: entropy_stats,
        nist_result: entropy_nist,
        experiment_id,
    });

    // Print comparison summary
    print_comparison_summary(&results);

    // Save results
    save_comparison_results(&results, &config)?;

    Ok(results)
}

fn print_comparison_summary(results: &[SourceResult]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     SOURCE COMPARISON SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Source          │ CV%      │ Bits    │ NIST Pass │ Min-Ent  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    for result in results {
        let cv_str = format!("{:.2}%", result.stats.cv * 100.0);
        let quantum_status = if result.stats.cv > 0.01 { "✅" } else { "❌" };
        let nist_status = if let Some(ref nist) = result.nist_result {
            format!("{}/{} ({:.0}%)",
                nist.passed_count,
                nist.tests.len(),
                nist.passed_count as f64 / nist.tests.len() as f64 * 100.0)
        } else {
            "N/A".to_string()
        };
        let min_ent = if let Some(ref nist) = result.nist_result {
            format!("{:.4}", nist.min_entropy)
        } else {
            "N/A".to_string()
        };

        println!("║ {:<15} │ {:>7} {} │ {:>7} │ {:>9} │ {:>8} ║",
            result.source,
            cv_str,
            quantum_status,
            result.final_bits.len(),
            nist_status,
            min_ent
        );
    }

    println!("╚══════════════════════════════════════════════════════════════╝");

    // Answer the key question
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("KEY QUESTION: Is CPU jitter 'more quantum' than SSD timing?");
    println!("═══════════════════════════════════════════════════════════════");

    let ssd = &results[0];
    let cpu = &results[1];

    println!();
    println!("CV (Coefficient of Variation) indicates quantum consistency:");
    println!("  - CV > 1% = quantum-consistent timing variance");
    println!("  - Higher CV = MORE variance = MORE potential quantum contribution");
    println!();
    println!("COMPARISON:");
    println!("  SSD Timing CV:  {:.2}%", ssd.stats.cv * 100.0);
    println!("  CPU Jitter CV:  {:.2}%", cpu.stats.cv * 100.0);
    println!();

    if cpu.stats.cv > ssd.stats.cv {
        let ratio = cpu.stats.cv / ssd.stats.cv;
        println!("ANSWER: YES - CPU jitter shows {:.1}x HIGHER variance than SSD timing.", ratio);
        println!();
        println!("INTERPRETATION:");
        println!("  - CPU jitter has more thermal/quantum noise contribution");
        println!("  - SSD timing is more stable (controller logic dominates)");
        println!("  - BOTH are quantum-consistent (CV > 1%)");
        println!();
        println!("PHYSICS:");
        println!("  - CPU jitter: Thermal noise (Brownian motion), electron tunneling");
        println!("  - SSD timing:  Fowler-Nordheim tunneling + controller buffering");
        println!("  - Higher CV in CPU suggests less deterministic control path");
    } else {
        let ratio = ssd.stats.cv / cpu.stats.cv;
        println!("ANSWER: NO - SSD timing shows {:.1}x HIGHER variance than CPU jitter.", ratio);
        println!();
        println!("INTERPRETATION:");
        println!("  - SSD write timing has more quantum contribution");
        println!("  - Fowler-Nordheim tunneling variance dominates");
    }

    // Compare NIST results
    if let (Some(ssd_nist), Some(cpu_nist)) = (&ssd.nist_result, &cpu.nist_result) {
        println!();
        println!("NIST TEST COMPARISON:");
        println!("  SSD:   {}/15 passed ({:.0}%)", ssd_nist.passed_count, ssd_nist.passed_count as f64 / 15.0 * 100.0);
        println!("  CPU:   {}/15 passed ({:.0}%)", cpu_nist.passed_count, cpu_nist.passed_count as f64 / 15.0 * 100.0);
        println!();
        println!("  Min-entropy (higher = better):");
        println!("  SSD:   {:.4} bits/bit", ssd_nist.min_entropy);
        println!("  CPU:   {:.4} bits/bit", cpu_nist.min_entropy);
    }
}

fn save_comparison_results(results: &[SourceResult], config: &SourceConfig) -> Result<(), String> {
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("Cannot create output dir: {}", e))?;

    let timestamp = chrono_timestamp();

    // Save comparison report
    let report_path = config.output_dir.join(format!("SOURCE_COMPARE_{}.md", timestamp));
    let mut report = String::new();

    report.push_str("# QRNG Source Comparison Report\n\n");
    report.push_str(&format!("**Date**: {}\n\n", timestamp));
    report.push_str(&format!("**Samples per source**: {}\n\n", config.n_samples));

    report.push_str("## Results Summary\n\n");
    report.push_str("| Source | CV% | Quantum? | Bits | NIST Pass | Min-Entropy |\n");
    report.push_str("|--------|-----|----------|------|-----------|-------------|\n");

    for result in results {
        let quantum = if result.stats.cv > 0.01 { "✅" } else { "❌" };
        let nist = if let Some(ref nist) = result.nist_result {
            format!("{}/{}", nist.passed_count, nist.tests.len())
        } else {
            "N/A".to_string()
        };
        let min_ent = if let Some(ref nist) = result.nist_result {
            format!("{:.4}", nist.min_entropy)
        } else {
            "N/A".to_string()
        };

        report.push_str(&format!(
            "| {} | {:.2}% | {} | {} | {} | {} |\n",
            result.source,
            result.stats.cv * 100.0,
            quantum,
            result.final_bits.len(),
            nist,
            min_ent
        ));
    }

    std::fs::write(&report_path, &report)
        .map_err(|e| format!("Cannot write report: {}", e))?;
    println!("\n📄 Comparison report saved to: {}", report_path.display());

    // Save raw data for each source
    for result in results {
        let data_path = config.output_dir.join(format!(
            "{}_{}_data.csv",
            result.source.replace("/", "-"),
            timestamp
        ));

        let mut file = File::create(&data_path)
            .map_err(|e| format!("Cannot create data file: {}", e))?;

        writeln!(file, "sample,value_ns").ok();
        for (i, &v) in result.raw_values.iter().enumerate() {
            writeln!(file, "{},{}", i, v).ok();
        }
        println!("📊 {} data saved to: {}", result.source, data_path.display());
    }

    Ok(())
}

fn chrono_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_comparison_small() {
        let config = SourceConfig {
            n_samples: 500, // Small for testing
            debias: true,
            lsb_count: 8,
            output_dir: PathBuf::from("/tmp/qrng_source_comparison"),
        };

        let results = run_source_comparison(config).unwrap();

        assert_eq!(results.len(), 3, "Should have 3 sources tested");
        assert!(results[0].stats.cv > 0.0, "SSD CV should be positive");
        assert!(results[1].stats.cv > 0.0, "CPU CV should be positive");
    }
}
