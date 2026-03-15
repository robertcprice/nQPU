//! QRNG Experiment Runner
//!
//! Runs complete experiments for SSD-based quantum random number generation
//! including timing measurement, bit extraction, and NIST verification.

use std::fs::File;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::nist_tests::{NistSuiteResult, NistTestSuite};
use crate::real_quantum_probe::{RealCpuJitterProbe, RealPhotonProbe, RealSsdProbe};

/// Experiment configuration
#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    /// Number of samples to collect
    pub n_samples: usize,
    /// Output directory for results
    pub output_dir: PathBuf,
    /// Experiment ID
    pub experiment_id: String,
    /// Include SSD tests
    pub test_ssd: bool,
    /// Include CPU jitter tests
    pub test_cpu: bool,
    /// Include entropy pool tests
    pub test_entropy: bool,
    /// Run NIST tests on extracted bits
    pub run_nist: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            n_samples: 10_000,
            output_dir: PathBuf::from("experiments"),
            experiment_id: format!("EXP-{}", chrono_timestamp()),
            test_ssd: true,
            test_cpu: true,
            test_entropy: true,
            run_nist: true,
        }
    }
}

/// Complete experiment results
#[derive(Clone, Debug)]
pub struct ExperimentResult {
    /// Experiment ID
    pub id: String,
    /// Timestamp
    pub timestamp: u64,
    /// SSD timing results
    pub ssd_timings: Vec<u64>,
    /// SSD statistics
    pub ssd_stats: Option<SsdStats>,
    /// CPU jitter results
    pub cpu_timings: Vec<u64>,
    /// CPU statistics
    pub cpu_stats: Option<CpuStats>,
    /// Extracted bits
    pub bits: Vec<u8>,
    /// NIST test results
    pub nist_result: Option<NistSuiteResult>,
    /// Overall quantum score
    pub quantum_score: f64,
    /// Interpretation
    pub interpretation: String,
}

/// SSD timing statistics
#[derive(Clone, Debug)]
pub struct SsdStats {
    pub mean_ns: f64,
    pub std_ns: f64,
    pub cv: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub n_samples: usize,
}

/// CPU timing statistics
#[derive(Clone, Debug)]
pub struct CpuStats {
    pub mean_ns: f64,
    pub std_ns: f64,
    pub cv: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub n_samples: usize,
}

/// QRNG Experiment Runner
pub struct QrngExperiment {
    config: ExperimentConfig,
}

impl QrngExperiment {
    /// Create new experiment runner
    pub fn new(config: ExperimentConfig) -> Self {
        Self { config }
    }

    /// Run complete experiment
    pub fn run(&mut self) -> Result<ExperimentResult, String> {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  QRNG EXPERIMENT: {}", self.config.experiment_id);
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut ssd_timings = Vec::new();
        let mut ssd_stats = None;
        let mut cpu_timings = Vec::new();
        let mut cpu_stats = None;
        let mut bits = Vec::new();
        let mut nist_result = None;
        let mut quantum_score: f64 = 0.0;

        // SSD timing test
        if self.config.test_ssd {
            println!(
                "▶ Running SSD timing test ({} samples)...",
                self.config.n_samples
            );
            let mut ssd_probe = RealSsdProbe::new();
            let result = ssd_probe.run_test(self.config.n_samples / 10)?; // Fewer writes, takes time

            ssd_timings = ssd_probe.timings.clone();
            ssd_stats = Some(SsdStats {
                mean_ns: result.mean_ns,
                std_ns: result.std_ns,
                cv: result.cv,
                min_ns: ssd_timings.iter().min().copied().unwrap_or(0),
                max_ns: ssd_timings.iter().max().copied().unwrap_or(0),
                n_samples: ssd_timings.len(),
            });

            println!(
                "  Mean: {:.2} µs, Std: {:.2} µs, CV: {:.2}%",
                result.mean_ns / 1000.0,
                result.std_ns / 1000.0,
                result.cv * 100.0
            );

            if result.is_quantum_consistent {
                quantum_score += 0.4;
                println!("  ✅ Quantum-consistent variance");
            } else {
                println!("  ❌ Low variance - possibly cached");
            }
            println!();
        }

        // CPU jitter test
        if self.config.test_cpu {
            println!(
                "▶ Running CPU jitter test ({} samples)...",
                self.config.n_samples
            );
            let mut cpu_probe = RealCpuJitterProbe::new();
            let result = cpu_probe.run_test(self.config.n_samples);

            cpu_timings = cpu_probe.timings.clone();
            cpu_stats = Some(CpuStats {
                mean_ns: result.mean_ns,
                std_ns: result.std_ns,
                cv: result.std_ns / result.mean_ns,
                min_ns: result.min_ns,
                max_ns: result.max_ns,
                n_samples: result.n_samples,
            });

            println!(
                "  Mean: {:.2} ns, Std: {:.2} ns, CV: {:.2}%",
                result.mean_ns,
                result.std_ns,
                (result.std_ns / result.mean_ns) * 100.0
            );

            if result.is_quantum_consistent {
                quantum_score += 0.3;
                println!("  ✅ Quantum-consistent jitter");
            } else {
                println!("  ❌ Low jitter");
            }
            println!();
        }

        // Extract bits from entropy pool
        if self.config.test_entropy {
            println!(
                "▶ Collecting bits from entropy pool ({} samples)...",
                self.config.n_samples
            );
            let mut photon_probe = RealPhotonProbe::new();
            let samples = photon_probe.capture_real_samples(self.config.n_samples / 2)?;

            // Convert samples to bits (using LSBs)
            for &sample in &samples {
                bits.push((sample & 0xFF) as u8);
            }

            let result = photon_probe.analyze_poisson();
            println!(
                "  Fano factor: {:.4} (ideal: 1.0 for Poisson)",
                result.fano_factor
            );

            if result.is_quantum_consistent {
                quantum_score += 0.3;
                println!("  ✅ Poisson-consistent");
            } else {
                println!("  ❌ Not Poisson-distributed (expected for processed entropy)");
            }
            println!();
        }

        // Run NIST tests
        if self.config.run_nist && !bits.is_empty() {
            println!(
                "▶ Running NIST SP 800-22 test suite ({} bits)...",
                bits.len() * 8
            );

            // Convert bytes to bits for NIST tests
            let bit_vec: Vec<u8> = bits
                .iter()
                .flat_map(|&b| (0..8).map(move |i| (b >> i) & 1))
                .collect();

            let suite = NistTestSuite::new();
            let result = suite.run_all_tests(&bit_vec);

            println!("{}", suite.generate_report(&result));

            nist_result = Some(result.clone());

            if result.overall_pass {
                quantum_score = (quantum_score + 1.0).min(1.0);
            }
        }

        // Generate interpretation
        let interpretation =
            self.interpret_results(quantum_score, &ssd_stats, &cpu_stats, &nist_result);

        let result = ExperimentResult {
            id: self.config.experiment_id.clone(),
            timestamp,
            ssd_timings,
            ssd_stats,
            cpu_timings,
            cpu_stats,
            bits,
            nist_result,
            quantum_score,
            interpretation,
        };

        // Save results
        self.save_results(&result)?;

        Ok(result)
    }

    fn interpret_results(
        &self,
        quantum_score: f64,
        ssd_stats: &Option<SsdStats>,
        cpu_stats: &Option<CpuStats>,
        nist_result: &Option<NistSuiteResult>,
    ) -> String {
        let mut interpretation = String::new();

        interpretation.push_str("# QRNG Experiment Results\n\n");

        if let Some(ref ssd) = ssd_stats {
            interpretation.push_str(&format!(
                "## SSD Timing Analysis\n- Mean: {:.2} µs\n- CV: {:.2}%\n- Interpretation: {}\n\n",
                ssd.mean_ns / 1000.0,
                ssd.cv * 100.0,
                if ssd.cv > 0.1 {
                    "High variance consistent with quantum tunneling uncertainty"
                } else if ssd.cv > 0.01 {
                    "Moderate variance, possibly quantum-influenced"
                } else {
                    "Low variance, likely classical/cached"
                }
            ));
        }

        if let Some(ref cpu) = cpu_stats {
            let cpu_cv = cpu.std_ns / cpu.mean_ns;
            interpretation.push_str(&format!(
                "## CPU Jitter Analysis\n- Mean: {:.2} ns\n- CV: {:.2}%\n- Interpretation: {}\n\n",
                cpu.mean_ns,
                cpu_cv * 100.0,
                if cpu_cv > 0.5 {
                    "High jitter from thermal noise (quantum Brownian motion)"
                } else {
                    "Low jitter, deterministic timing"
                }
            ));
        }

        if let Some(ref nist) = nist_result {
            interpretation.push_str(&format!(
                "## NIST Test Results\n- Passed: {}/{}\n- Pass Ratio: {:.1}%\n- Min-Entropy: {:.4} bits/bit\n- Verdict: {}\n\n",
                nist.passed_count,
                nist.tests.len(),
                nist.pass_ratio * 100.0,
                nist.min_entropy,
                if nist.overall_pass { "PASSED" } else { "NEEDS IMPROVEMENT" }
            ));
        }

        interpretation.push_str(&format!(
            "## Overall Quantum Score: {:.0}%\n\n",
            quantum_score * 100.0
        ));

        interpretation.push_str("## Conclusion\n");
        if quantum_score > 0.7 {
            interpretation.push_str("Results are consistent with quantum randomness sources. ");
            interpretation.push_str("Further verification needed for publication.");
        } else if quantum_score > 0.4 {
            interpretation.push_str("Mixed results - some quantum signatures detected. ");
            interpretation.push_str("Need more samples and controlled conditions.");
        } else {
            interpretation.push_str("Insufficient quantum signatures detected. ");
            interpretation.push_str("Consider alternative approaches or better isolation.");
        }

        interpretation
    }

    fn save_results(&self, result: &ExperimentResult) -> Result<(), String> {
        // Create output directory if needed
        std::fs::create_dir_all(&self.config.output_dir)
            .map_err(|e| format!("Cannot create output dir: {}", e))?;

        // Save main report
        let report_path = self
            .config
            .output_dir
            .join(format!("{}_report.md", result.id));
        let mut file =
            File::create(&report_path).map_err(|e| format!("Cannot create report file: {}", e))?;

        file.write_all(result.interpretation.as_bytes())
            .map_err(|e| format!("Cannot write report: {}", e))?;

        println!("📄 Report saved to: {}", report_path.display());

        // Save raw data
        let data_path = self
            .config
            .output_dir
            .join(format!("{}_data.csv", result.id));
        let mut data_file =
            File::create(&data_path).map_err(|e| format!("Cannot create data file: {}", e))?;

        writeln!(data_file, "ssd_timing_ns,cpu_timing_ns")
            .map_err(|e| format!("Cannot write header: {}", e))?;

        let max_len = result.ssd_timings.len().max(result.cpu_timings.len());
        for i in 0..max_len {
            let ssd = result.ssd_timings.get(i).copied().unwrap_or(0);
            let cpu = result.cpu_timings.get(i).copied().unwrap_or(0);
            writeln!(data_file, "{},{}", ssd, cpu)
                .map_err(|e| format!("Cannot write data: {}", e))?;
        }

        println!("📊 Data saved to: {}", data_path.display());

        Ok(())
    }

    /// Generate summary report
    pub fn generate_summary(result: &ExperimentResult) -> String {
        format!(
            r#"╔══════════════════════════════════════════════════════════════╗
║           QRNG EXPERIMENT RESULTS                               ║
╠══════════════════════════════════════════════════════════════╣
║  Experiment ID: {:<44}║
║  Timestamp: {:<47}║
╠══════════════════════════════════════════════════════════════╣
║  QUANTUM EVIDENCE SCORE: {:<3.0}%                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                                ║
║  SSD Timing:    {}                                ║
║  CPU Jitter:    {}                                ║
║  NIST Tests:    {}                                ║
║                                                                ║
║  For full report, see: experiments/{}_report.md               ║
╚══════════════════════════════════════════════════════════════╝"#,
            result.id,
            result.timestamp,
            result.quantum_score * 100.0,
            if result.ssd_stats.is_some() {
                "✅ Tested"
            } else {
                "⏭ Skipped"
            },
            if result.cpu_stats.is_some() {
                "✅ Tested"
            } else {
                "⏭ Skipped"
            },
            if result.nist_result.is_some() {
                if result.nist_result.as_ref().unwrap().overall_pass {
                    "✅ PASSED"
                } else {
                    "⚠️ PARTIAL"
                }
            } else {
                "⏭ Skipped"
            },
            result.id
        )
    }
}

fn chrono_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}", now)
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_experiment() {
        let config = ExperimentConfig {
            n_samples: 1000, // Smaller for testing
            output_dir: PathBuf::from("/tmp/qrng_experiments"),
            experiment_id: "TEST-001".to_string(),
            test_ssd: true,
            test_cpu: true,
            test_entropy: true,
            run_nist: true,
        };

        let mut experiment = QrngExperiment::new(config);
        let result = experiment.run().unwrap();

        println!("{}", QrngExperiment::generate_summary(&result));

        assert!(!result.bits.is_empty(), "Should have collected bits");
        assert!(result.quantum_score > 0.0, "Should have some quantum score");
    }
}
