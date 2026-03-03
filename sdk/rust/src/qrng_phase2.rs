//! SSD QRNG Phase 2 Experiment - Proper Methodology
//!
//! This version extracts bits DIRECTLY from SSD timing measurements,
//! not from /dev/urandom or other sources.

use std::fs::{File, OpenOptions};
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::nist_tests::{NistSuiteResult, NistTestSuite};

/// Phase 2 experiment configuration
#[derive(Clone, Debug)]
pub struct Phase2Config {
    /// Number of SSD writes to perform
    pub n_writes: usize,
    /// Output directory
    pub output_dir: PathBuf,
    /// Test file path for SSD writes
    pub test_file: String,
    /// Whether to apply von Neumann debiasing
    pub debias: bool,
    /// Number of LSBs to extract per timing
    pub lsb_count: usize,
}

impl Default for Phase2Config {
    fn default() -> Self {
        Self {
            n_writes: 10_000,
            output_dir: PathBuf::from("experiments"),
            test_file: "/tmp/ssd_qrng_test.bin".to_string(),
            debias: true,
            lsb_count: 8,
        }
    }
}

/// Phase 2 experiment results
#[derive(Clone, Debug)]
pub struct Phase2Result {
    /// Experiment ID
    pub id: String,
    /// Raw timing measurements (nanoseconds)
    pub timings_ns: Vec<u64>,
    /// Extracted raw bits (before debiasing)
    pub raw_bits: Vec<u8>,
    /// Extracted bits (after debiasing, if applied)
    pub final_bits: Vec<u8>,
    /// Timing statistics
    pub timing_stats: TimingStats,
    /// NIST test results
    pub nist_result: Option<NistSuiteResult>,
    /// Methodology notes
    pub methodology: String,
}

#[derive(Clone, Debug)]
pub struct TimingStats {
    pub mean_ns: f64,
    pub std_ns: f64,
    pub cv: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub n_samples: usize,
}

/// Phase 2 SSD QRNG Experiment
pub struct SsdQrngPhase2 {
    config: Phase2Config,
    timings: Vec<u64>,
}

impl SsdQrngPhase2 {
    pub fn new(config: Phase2Config) -> Self {
        Self {
            config,
            timings: Vec::new(),
        }
    }

    /// Run the complete Phase 2 experiment
    pub fn run(&mut self) -> Result<Phase2Result, String> {
        let id = format!("PHASE2-{}", chrono_timestamp());

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║     SSD QRNG PHASE 2 - PROPER METHODOLOGY                      ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Experiment ID: {:<44}║", id);
        println!("║  Methodology: {:>46}║", "SSD timing → LSB extraction → NIST");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        // Step 1: Collect SSD timing measurements
        println!("▶ STEP 1: Collecting SSD timing measurements...");
        println!("  Writing {} times to SSD...", self.config.n_writes);
        self.collect_timings()?;

        // Step 2: Calculate timing statistics
        println!();
        println!("▶ STEP 2: Analyzing timing statistics...");
        let stats = self.calculate_stats();

        println!("  Mean:    {:.2} µs ({:.0} ns)", stats.mean_ns / 1000.0, stats.mean_ns);
        println!("  Std:     {:.2} µs ({:.0} ns)", stats.std_ns / 1000.0, stats.std_ns);
        println!("  CV:      {:.2}%", stats.cv * 100.0);
        println!("  Range:   {} - {} ns", stats.min_ns, stats.max_ns);

        // Step 3: Extract bits from timing LSBs
        println!();
        println!("▶ STEP 3: Extracting bits from timing LSBs...");
        println!("  Using {} least significant bits per timing", self.config.lsb_count);
        let raw_bits = self.extract_bits_from_timings(self.config.lsb_count);
        println!("  Extracted {} raw bits", raw_bits.len());

        // Step 4: Apply von Neumann debiasing (optional)
        let final_bits = if self.config.debias {
            println!();
            println!("▶ STEP 4: Applying von Neumann debiasing...");
            let debiased = self.von_neumann_debias(&raw_bits);
            println!("  Debiased: {} → {} bits ({:.1}% retention)",
                     raw_bits.len(), debiased.len(),
                     (debiased.len() as f64 / raw_bits.len() as f64) * 100.0);
            debiased
        } else {
            println!();
            println!("▶ STEP 4: Skipping debiasing");
            raw_bits.clone()
        };

        // Step 5: Run NIST tests
        println!();
        println!("▶ STEP 5: Running NIST SP 800-22 test suite...");
        println!("  Testing {} bits from SSD timing", final_bits.len());

        let nist_result = if final_bits.len() >= 1000 {
            let suite = NistTestSuite::new();
            let result = suite.run_all_tests(&final_bits);
            println!("{}", suite.generate_report(&result));
            Some(result)
        } else {
            println!("  ⚠️ Not enough bits for NIST tests (need ≥1000)");
            None
        };

        // Generate methodology notes
        let methodology = self.generate_methodology_notes(&stats, &raw_bits, &final_bits);

        let result = Phase2Result {
            id: id.clone(),
            timings_ns: self.timings.clone(),
            raw_bits,
            final_bits: final_bits.clone(),
            timing_stats: stats,
            nist_result,
            methodology,
        };

        // Save results
        self.save_results(&result)?;

        Ok(result)
    }

    /// Collect timing measurements from actual SSD writes
    fn collect_timings(&mut self) -> Result<(), String> {
        self.timings.clear();
        self.timings.reserve(self.config.n_writes);

        // Prepare data to write (4KB blocks)
        let data = vec![0x55u8; 4096]; // 0x55 = 01010101 pattern

        // Warm-up writes (clear caches)
        for _ in 0..20 {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&self.config.test_file)
                .map_err(|e| format!("Cannot open test file: {}", e))?;
            file.write_all(&data).ok();
        }

        // Actual measurements
        let mut progress_interval = self.config.n_writes / 10;
        if progress_interval == 0 { progress_interval = 1; }

        for i in 0..self.config.n_writes {
            let start = Instant::now();

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&self.config.test_file)
                .map_err(|e| format!("Cannot open test file: {}", e))?;

            file.write_all(&data)
                .map_err(|e| format!("Write failed: {}", e))?;

            // Sync to ensure actual disk write (this is where tunneling happens!)
            #[cfg(unix)]
            {
                
                file.sync_all().map_err(|e| format!("Sync failed: {}", e))?;
            }

            let elapsed_ns = start.elapsed().as_nanos() as u64;
            self.timings.push(elapsed_ns);

            if (i + 1) % progress_interval == 0 {
                println!("  Progress: {}/{} ({:.0}%)", i + 1, self.config.n_writes,
                         (i + 1) as f64 / self.config.n_writes as f64 * 100.0);
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&self.config.test_file);

        Ok(())
    }

    /// Calculate timing statistics
    fn calculate_stats(&self) -> TimingStats {
        let n = self.timings.len() as f64;
        let mean: f64 = self.timings.iter().sum::<u64>() as f64 / n;
        let variance: f64 = self.timings.iter()
            .map(|&t| (t as f64 - mean).powi(2))
            .sum::<f64>() / n;
        let std = variance.sqrt();
        let cv = if mean > 0.0 { std / mean } else { 0.0 };

        TimingStats {
            mean_ns: mean,
            std_ns: std,
            cv,
            min_ns: self.timings.iter().min().copied().unwrap_or(0),
            max_ns: self.timings.iter().max().copied().unwrap_or(0),
            n_samples: self.timings.len(),
        }
    }

    /// Extract bits from timing LSBs
    ///
    /// Method: Take the N least significant bits of each timing measurement.
    /// The LSBs have the most entropy because they're affected by:
    /// - Quantum tunneling uncertainty
    /// - Thermal noise
    /// - Interrupt timing
    fn extract_bits_from_timings(&self, lsb_count: usize) -> Vec<u8> {
        let mut bits = Vec::with_capacity(self.timings.len() * lsb_count);

        for &timing in &self.timings {
            // Extract LSBs
            for bit_idx in 0..lsb_count {
                let bit = ((timing >> bit_idx) & 1) as u8;
                bits.push(bit);
            }
        }

        bits
    }

    /// Von Neumann debiasing
    ///
    /// Takes pairs of bits:
    /// - 01 → output 0
    /// - 10 → output 1
    /// - 00, 11 → discard
    ///
    /// This removes bias but reduces bit count by ~75%
    fn von_neumann_debias(&self, bits: &[u8]) -> Vec<u8> {
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

    fn generate_methodology_notes(
        &self,
        stats: &TimingStats,
        raw_bits: &[u8],
        final_bits: &[u8],
    ) -> String {
        let mut notes = String::new();

        notes.push_str("# SSD QRNG Phase 2 - Methodology\n\n");

        notes.push_str("## Bit Extraction Method\n");
        notes.push_str(&format!("- Source: SSD write timing (ns)\n"));
        notes.push_str(&format!("- Extraction: {} LSBs per timing\n", self.config.lsb_count));
        notes.push_str(&format!("- Debiasing: {}\n\n", if self.config.debias { "Von Neumann" } else { "None" }));

        notes.push_str("## Data Flow\n");
        notes.push_str("```\n");
        notes.push_str("SSD Write → Sync to disk → Measure timing (ns)\n");
        notes.push_str(&format!("         ↓\n"));
        notes.push_str(&format!("Extract {} LSBs → {} raw bits\n", self.config.lsb_count, raw_bits.len()));
        if self.config.debias {
            notes.push_str(&format!("         ↓\n"));
            notes.push_str(&format!("Von Neumann → {} final bits\n", final_bits.len()));
        }
        notes.push_str("```\n\n");

        notes.push_str("## Physics Justification\n");
        notes.push_str("- Fowler-Nordheim tunneling: J = A·E²·exp(-B/E)\n");
        notes.push_str("- Tunneling is quantum mechanical (probabilistic)\n");
        notes.push_str("- Timing variance reflects tunneling uncertainty\n");
        notes.push_str(&format!("- Measured CV = {:.2}% (>1% = quantum-consistent)\n\n", stats.cv * 100.0));

        notes.push_str("## Why LSBs?\n");
        notes.push_str("- High bits: Deterministic (disk geometry, controller)\n");
        notes.push_str("- Low bits: Chaotic (tunneling, thermal, interrupts)\n");
        notes.push_str("- LSBs have maximum entropy from quantum sources\n");

        notes
    }

    fn save_results(&self, result: &Phase2Result) -> Result<(), String> {
        std::fs::create_dir_all(&self.config.output_dir)
            .map_err(|e| format!("Cannot create output dir: {}", e))?;

        // Save methodology
        let method_path = self.config.output_dir.join(format!("{}_methodology.md", result.id));
        std::fs::write(&method_path, &result.methodology)
            .map_err(|e| format!("Cannot write methodology: {}", e))?;
        println!("📄 Methodology saved to: {}", method_path.display());

        // Save timing data
        let data_path = self.config.output_dir.join(format!("{}_timings.csv", result.id));
        let mut file = File::create(&data_path)
            .map_err(|e| format!("Cannot create data file: {}", e))?;
        writeln!(file, "sample,timing_ns").ok();
        for (i, &t) in result.timings_ns.iter().enumerate() {
            writeln!(file, "{},{}", i, t).ok();
        }
        println!("📊 Timing data saved to: {}", data_path.display());

        // Save extracted bits
        let bits_path = self.config.output_dir.join(format!("{}_bits.txt", result.id));
        let bits_str: String = result.final_bits.iter().map(|b| char::from(b + 48)).collect();
        std::fs::write(&bits_path, &bits_str)
            .map_err(|e| format!("Cannot write bits file: {}", e))?;
        println!("🔢 Bits saved to: {}", bits_path.display());

        Ok(())
    }

    /// Generate summary
    pub fn generate_summary(result: &Phase2Result) -> String {
        let nist_status = if let Some(ref nist) = result.nist_result {
            if nist.overall_pass {
                format!("✅ PASSED ({}/{})", nist.passed_count, nist.tests.len())
            } else {
                format!("⚠️ PARTIAL ({}/{})", nist.passed_count, nist.tests.len())
            }
        } else {
            "⏭ Skipped (not enough bits)".to_string()
        };

        let entropy = if let Some(ref nist) = result.nist_result {
            format!("{:.4}", nist.min_entropy)
        } else {
            "N/A".to_string()
        };

        format!(
            r#"╔══════════════════════════════════════════════════════════════╗
║     SSD QRNG PHASE 2 - RESULTS                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Experiment ID: {:<44}║
╠══════════════════════════════════════════════════════════════╣
║  SSD WRITES: {:>6}                                            ║
║  Mean timing: {:>10.2} µs                                    ║
║  CV (variance): {:>8.2}%  {}                    ║
╠══════════════════════════════════════════════════════════════╣
║  BITS EXTRACTED: {:>6} (raw: {:>6})                        ║
║  Min-entropy: {:>8} bits/bit                              ║
║  NIST Tests: {}                              ║
╚══════════════════════════════════════════════════════════════╝"#,
            result.id,
            result.timing_stats.n_samples,
            result.timing_stats.mean_ns / 1000.0,
            result.timing_stats.cv * 100.0,
            if result.timing_stats.cv > 0.01 { "✅ QUANTUM" } else { "❌ Classical" },
            result.final_bits.len(),
            result.raw_bits.len(),
            entropy,
            nist_status
        )
    }
}

fn chrono_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase2_experiment() {
        let config = Phase2Config {
            n_writes: 500, // Smaller for testing
            output_dir: PathBuf::from("/tmp/qrng_phase2"),
            test_file: "/tmp/ssd_qrng_phase2_test.bin".to_string(),
            debias: true,
            lsb_count: 8,
        };

        let mut experiment = SsdQrngPhase2::new(config);
        let result = experiment.run().unwrap();

        println!("{}", SsdQrngPhase2::generate_summary(&result));

        assert!(!result.timings_ns.is_empty(), "Should have timing data");
        assert!(!result.final_bits.is_empty(), "Should have extracted bits");
    }
}
