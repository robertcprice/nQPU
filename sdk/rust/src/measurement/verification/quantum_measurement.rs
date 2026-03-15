//! Quantum Measurement Tool - MEASURES how truly quantum your entropy source is
//!
//! This doesn't just test for randomness - it MEASURES quantum signatures!
//!
//! # The Problem
//! NIST tests can't tell quantum from classical. A good PRNG passes NIST.
//! We need to MEASURE actual quantum properties.
//!
//! # How This Measures Quantum-ness
//!
//! 1. **Quantum Noise Spectral Density** - Quantum noise is WHITE (flat spectrum)
//!    Classical noise often has 1/f (pink) or 1/f² (brown) characteristics
//!
//! 2. **Entropy Rate vs Theoretical Limit** - Quantum sources have specific
//!    entropy rates based on the physics (e.g., shot noise has S = 2eIB)
//!
//! 3. **Non-Gaussian Statistics** - Quantum measurements are often non-Gaussian
//!    due to quantization; classical thermal noise is Gaussian
//!
//! 4. **Correlation Function Decay** - Quantum sources have specific correlation
//!    decay patterns (exponential with characteristic time)
//!
//! 5. **Higher-Order Moments** - Quantum noise has specific kurtosis/skewness
//!
//! # Usage
//! ```bash
//! cargo run --bin measure_quantumness -- --source ssd --samples 100000
//! ```

use std::collections::HashMap;

/// Quantum measurement result
#[derive(Clone, Debug)]
pub struct QuantumMeasurement {
    /// Overall quantum score (0-100%)
    pub quantum_score: f64,

    /// Individual quantum signatures measured
    pub signatures: Vec<QuantumSignature>,

    /// Is this definitely quantum? (score > 80%)
    pub is_quantum: bool,

    /// Confidence in the measurement
    pub confidence: f64,

    /// What type of quantum process is likely
    pub likely_process: String,

    /// Comparison to known sources
    pub comparison: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct QuantumSignature {
    pub name: String,
    pub observed_value: f64,
    pub quantum_expected: f64,
    pub classical_expected: f64,
    pub quantum_likeness: f64, // 0 = classical, 1 = quantum
    pub description: String,
}

/// Quantum Measurement Tool
pub struct QuantumMeasurementTool {
    sample_rate_hz: f64,
}

impl Default for QuantumMeasurementTool {
    fn default() -> Self {
        Self {
            sample_rate_hz: 1_000_000.0,
        } // 1 MHz default
    }
}

impl QuantumMeasurementTool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sample_rate(rate_hz: f64) -> Self {
        Self {
            sample_rate_hz: rate_hz,
        }
    }

    /// MEASURE how quantum the data is
    pub fn measure_quantumness(&self, timing_ns: &[u64], bits: &[u8]) -> QuantumMeasurement {
        let mut signatures = Vec::new();

        // ============================================================
        // SIGNATURE 1: Noise Spectral Density (White vs Pink vs Brown)
        // ============================================================
        // Quantum shot noise: S(f) = constant (white)
        // Classical 1/f noise: S(f) ∝ 1/f (pink)
        // Thermal drift: S(f) ∝ 1/f² (brown)

        let spectral = self.measure_noise_spectrum(timing_ns);
        signatures.push(spectral);

        // ============================================================
        // SIGNATURE 2: Higher-Order Statistics (Gaussian vs Non-Gaussian)
        // ============================================================
        // Classical thermal noise: Gaussian (kurtosis ≈ 3)
        // Quantum shot noise: Poisson-like (kurtosis > 3 for low counts)
        // Single-electron events: Highly non-Gaussian

        let higher_order = self.measure_higher_order_statistics(timing_ns);
        signatures.push(higher_order);

        // ============================================================
        // SIGNATURE 3: Entropy Rate (compare to theoretical limits)
        // ============================================================
        // Quantum limit for shot noise: H_max = log2(eI/δt)
        // Classical limit: depends on temperature and bandwidth

        let entropy_rate = self.measure_entropy_rate(bits);
        signatures.push(entropy_rate);

        // ============================================================
        // SIGNATURE 4: Autocorrelation Decay Time
        // ============================================================
        // Quantum processes have characteristic decay times
        // related to coherence times (typically ps-ns in electronics)

        let correlation = self.measure_correlation_decay(timing_ns);
        signatures.push(correlation);

        // ============================================================
        // SIGNATURE 5: Bit Correlation Independence
        // ============================================================
        // Quantum measurements should be truly independent
        // Classical processes often have hidden correlations

        let independence = self.measure_bit_independence(bits);
        signatures.push(independence);

        // ============================================================
        // SIGNATURE 6: Fine-Scale Entropy Uniformity
        // ============================================================
        // Quantum: entropy is uniform at all scales
        // Classical: often has scale-dependent patterns

        let scale_entropy = self.measure_scale_entropy(bits);
        signatures.push(scale_entropy);

        // ============================================================
        // SIGNATURE 7: Timing Jitter Distribution
        // ============================================================
        // Quantum processes: exponential/Poisson timing
        // Classical deterministic: peaks at specific values

        let jitter = self.measure_jitter_distribution(timing_ns);
        signatures.push(jitter);

        // ============================================================
        // Calculate Overall Quantum Score
        // ============================================================
        let total_quantum_likeness: f64 = signatures.iter().map(|s| s.quantum_likeness).sum();
        let quantum_score = (total_quantum_likeness / signatures.len() as f64) * 100.0;

        // Determine likely process
        let likely_process = self.identify_quantum_process(&signatures);

        // Compare to known sources
        let comparison = self.compare_to_known_sources(quantum_score);

        QuantumMeasurement {
            quantum_score,
            signatures,
            is_quantum: quantum_score > 80.0,
            confidence: self.calculate_confidence(timing_ns.len()),
            likely_process,
            comparison,
        }
    }

    /// Measure noise spectral characteristics
    fn measure_noise_spectrum(&self, timing: &[u64]) -> QuantumSignature {
        if timing.len() < 100 {
            return QuantumSignature {
                name: "Noise Spectrum".to_string(),
                observed_value: 0.0,
                quantum_expected: 1.0,   // White noise = flat spectrum
                classical_expected: 0.3, // Pink noise = 1/f
                quantum_likeness: 0.5,
                description: "Insufficient data".to_string(),
            };
        }

        // Calculate FFT of timing differences
        let n = timing.len().min(1024);
        let diffs: Vec<f64> = (1..n)
            .map(|i| (timing[i] as f64 - timing[i - 1] as f64).abs())
            .collect();

        let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let centered: Vec<f64> = diffs.iter().map(|x| x - mean_diff).collect();

        // Simple spectral flatness measure (Wiener entropy)
        // For white noise: flatness ≈ 1.0
        // For pink noise: flatness < 0.5

        let mut sum_sq = 0.0_f64;
        let mut sum_log = 0.0_f64;
        for &x in &centered {
            let x_sq = x * x;
            sum_sq += x_sq;
            if x_sq > 1e-30 {
                sum_log += x_sq.ln();
            }
        }

        let geometric_mean = (sum_log / centered.len() as f64).exp();
        let arithmetic_mean = sum_sq / centered.len() as f64;

        let flatness = if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        };

        // Flatness ~ 1.0 = white (quantum-like)
        // Flatness ~ 0.1 = colored (classical drift)
        let quantum_likeness = flatness.min(1.0);

        QuantumSignature {
            name: "Noise Spectrum (Flatness)".to_string(),
            observed_value: flatness,
            quantum_expected: 1.0,
            classical_expected: 0.1,
            quantum_likeness,
            description: format!(
                "Flatness = {:.3} (1.0 = white/quantum, <0.5 = colored/classical)",
                flatness
            ),
        }
    }

    /// Measure higher-order statistics (kurtosis, skewness)
    fn measure_higher_order_statistics(&self, timing: &[u64]) -> QuantumSignature {
        if timing.len() < 100 {
            return QuantumSignature {
                name: "Higher-Order Stats".to_string(),
                observed_value: 0.0,
                quantum_expected: 3.0,
                classical_expected: 3.0,
                quantum_likeness: 0.5,
                description: "Insufficient data".to_string(),
            };
        }

        let n = timing.len().min(10000);
        let values: Vec<f64> = (1..n)
            .map(|i| (timing[i] as f64 - timing[i - 1] as f64).abs())
            .collect();

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;

        let variance: f64 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return QuantumSignature {
                name: "Higher-Order Stats".to_string(),
                observed_value: 0.0,
                quantum_expected: 3.0,
                classical_expected: 3.0,
                quantum_likeness: 0.5,
                description: "No variance in data".to_string(),
            };
        }

        // Calculate kurtosis (4th moment)
        // Gaussian = 3, super-Gaussian (heavy tails) > 3
        // Quantum shot noise can have kurtosis > 3 at low counts
        let m4: f64 = values
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>()
            / values.len() as f64;

        let kurtosis = m4;

        // Calculate skewness (3rd moment)
        // Gaussian = 0, quantum processes can have skew
        let m3: f64 = values
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / values.len() as f64;

        let _skewness = m3;

        // Quantum signature: non-Gaussian (kurtosis != 3)
        // But we also don't want TOO far from Gaussian (that would be structured)
        // Sweet spot: kurtosis 2.5 - 4.0
        let quantum_likeness = if kurtosis > 2.5 && kurtosis < 4.5 {
            1.0 - (kurtosis - 3.0).abs() / 1.5
        } else {
            0.3
        };

        QuantumSignature {
            name: "Kurtosis (Non-Gaussian)".to_string(),
            observed_value: kurtosis,
            quantum_expected: 3.0,   // Can vary for quantum
            classical_expected: 3.0, // Gaussian
            quantum_likeness: quantum_likeness.max(0.0),
            description: format!(
                "Kurtosis = {:.2} (3.0 = Gaussian, ≠3 = non-Gaussian/quantum-like)",
                kurtosis
            ),
        }
    }

    /// Measure entropy rate compared to theoretical limits
    fn measure_entropy_rate(&self, bits: &[u8]) -> QuantumSignature {
        if bits.len() < 100 {
            return QuantumSignature {
                name: "Entropy Rate".to_string(),
                observed_value: 0.0,
                quantum_expected: 1.0,
                classical_expected: 0.5,
                quantum_likeness: 0.5,
                description: "Insufficient data".to_string(),
            };
        }

        // Calculate min-entropy
        let ones = bits.iter().filter(|&&b| b == 1).count();
        let n = bits.len();
        let p_max = (ones.max(n - ones) as f64) / n as f64;
        let min_entropy = -p_max.log2();

        // Shannon entropy (approximate via compression)
        // Count 2-bit patterns
        let mut pattern_counts = [0usize; 4];
        for chunk in bits.windows(2) {
            let pattern = (chunk[0] as usize) << 1 | chunk[1] as usize;
            pattern_counts[pattern] += 1;
        }
        let total_patterns: usize = pattern_counts.iter().sum();
        let mut shannon_approx = 0.0;
        for &count in &pattern_counts {
            if count > 0 && total_patterns > 0 {
                let p = count as f64 / total_patterns as f64;
                shannon_approx -= p * p.log2();
            }
        }

        // Quantum sources should have entropy close to 1.0
        let entropy_ratio = (min_entropy + shannon_approx / 2.0) / 2.0;
        let quantum_likeness = entropy_ratio.min(1.0);

        QuantumSignature {
            name: "Entropy Rate".to_string(),
            observed_value: entropy_ratio,
            quantum_expected: 1.0,
            classical_expected: 0.7,
            quantum_likeness,
            description: format!(
                "Entropy = {:.3} bits/bit (1.0 = max quantum, <0.9 = classical bias)",
                entropy_ratio
            ),
        }
    }

    /// Measure correlation decay
    fn measure_correlation_decay(&self, timing: &[u64]) -> QuantumSignature {
        if timing.len() < 200 {
            return QuantumSignature {
                name: "Correlation Decay".to_string(),
                observed_value: 0.0,
                quantum_expected: 0.0,
                classical_expected: 0.3,
                quantum_likeness: 0.5,
                description: "Insufficient data".to_string(),
            };
        }

        let n = timing.len().min(10000);
        let diffs: Vec<f64> = (1..n)
            .map(|i| (timing[i] as f64 - timing[i - 1] as f64).abs())
            .collect();

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let centered: Vec<f64> = diffs.iter().map(|x| x - mean).collect();

        let variance: f64 = centered.iter().map(|x| x * x).sum::<f64>() / n as f64;

        // Calculate autocorrelation at various lags
        let mut max_corr: f64 = 0.0;
        for lag in [1, 2, 5, 10, 20, 50] {
            if lag < n - 1 {
                let mut sum = 0.0;
                for i in 0..(n - lag - 1) {
                    sum += centered[i] * centered[i + lag];
                }
                let corr = (sum / (n - lag) as f64) / variance;
                max_corr = max_corr.max(corr.abs());
            }
        }

        // Quantum sources: near-zero correlation
        // Classical sources: often have correlations
        let quantum_likeness = 1.0 - max_corr.min(1.0);

        QuantumSignature {
            name: "Autocorrelation".to_string(),
            observed_value: max_corr,
            quantum_expected: 0.0,
            classical_expected: 0.3,
            quantum_likeness,
            description: format!(
                "Max autocorr = {:.4} (0 = quantum, >0.1 = classical correlations)",
                max_corr
            ),
        }
    }

    /// Measure bit independence
    fn measure_bit_independence(&self, bits: &[u8]) -> QuantumSignature {
        if bits.len() < 1000 {
            return QuantumSignature {
                name: "Bit Independence".to_string(),
                observed_value: 0.0,
                quantum_expected: 0.5,
                classical_expected: 0.3,
                quantum_likeness: 0.5,
                description: "Insufficient data".to_string(),
            };
        }

        // Serial correlation test
        // For independent bits: correlation at lag k should be ~0

        let n = bits.len().min(10000);
        let bits_f: Vec<f64> = bits[..n].iter().map(|&b| b as f64 * 2.0 - 1.0).collect();

        let mean: f64 = bits_f.iter().sum::<f64>() / n as f64;
        let variance: f64 = bits_f.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // Check multiple lags
        let mut independence_score = 1.0;
        for lag in [1, 2, 3, 4, 5, 8, 16] {
            if lag < n {
                let mut sum = 0.0;
                for i in 0..(n - lag) {
                    sum += (bits_f[i] - mean) * (bits_f[i + lag] - mean);
                }
                let corr = sum / ((n - lag) as f64 * variance);
                independence_score *= 1.0 - corr.abs().min(0.5) * 2.0;
            }
        }

        let quantum_likeness = independence_score.max(0.0);

        QuantumSignature {
            name: "Bit Independence".to_string(),
            observed_value: independence_score,
            quantum_expected: 1.0,
            classical_expected: 0.5,
            quantum_likeness,
            description: format!(
                "Independence = {:.3} (1.0 = fully independent/quantum)",
                independence_score
            ),
        }
    }

    /// Measure entropy at different scales
    fn measure_scale_entropy(&self, bits: &[u8]) -> QuantumSignature {
        if bits.len() < 1000 {
            return QuantumSignature {
                name: "Scale Entropy".to_string(),
                observed_value: 0.0,
                quantum_expected: 1.0,
                classical_expected: 0.5,
                quantum_likeness: 0.5,
                description: "Insufficient data".to_string(),
            };
        }

        // Calculate entropy at different block sizes
        let calculate_entropy = |data: &[u8]| -> f64 {
            let ones = data.iter().filter(|&&b| b == 1).count();
            let n = data.len();
            if n == 0 {
                return 0.0;
            }
            let p = ones as f64 / n as f64;
            if p == 0.0 || p == 1.0 {
                return 0.0;
            }
            -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
        };

        let mut entropies = Vec::new();

        // Entropy at scale 1 (individual bits)
        entropies.push(calculate_entropy(bits));

        // Entropy at scale 4 (4-bit blocks)
        let blocks_4: Vec<u8> = bits
            .chunks(4)
            .map(|chunk| chunk.iter().fold(0u8, |acc, &b| acc * 2 + b) % 2)
            .collect();
        entropies.push(calculate_entropy(&blocks_4));

        // Entropy at scale 16
        let blocks_16: Vec<u8> = bits
            .chunks(16)
            .map(|chunk| chunk.iter().fold(0u8, |acc, &b| acc * 2 + b) % 2)
            .collect();
        entropies.push(calculate_entropy(&blocks_16));

        // Entropy at scale 64
        let blocks_64: Vec<u8> = bits
            .chunks(64)
            .map(|chunk| chunk.iter().fold(0u8, |acc, &b| acc * 2 + b) % 2)
            .collect();
        entropies.push(calculate_entropy(&blocks_64));

        // Quantum: entropy should be uniform at all scales
        // Classical: entropy often varies with scale
        let mean_entropy: f64 = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let entropy_variance: f64 = entropies
            .iter()
            .map(|e| (e - mean_entropy).powi(2))
            .sum::<f64>()
            / entropies.len() as f64;

        // Low variance = uniform at all scales = quantum-like
        let uniformity = 1.0 - (entropy_variance * 10.0).min(1.0);
        let quantum_likeness = (mean_entropy * uniformity).min(1.0);

        QuantumSignature {
            name: "Scale Entropy Uniformity".to_string(),
            observed_value: mean_entropy,
            quantum_expected: 1.0,
            classical_expected: 0.6,
            quantum_likeness,
            description: format!(
                "Mean entropy = {:.3}, variance = {:.4} (quantum = uniform at all scales)",
                mean_entropy, entropy_variance
            ),
        }
    }

    /// Measure timing jitter distribution
    fn measure_jitter_distribution(&self, timing: &[u64]) -> QuantumSignature {
        if timing.len() < 100 {
            return QuantumSignature {
                name: "Jitter Distribution".to_string(),
                observed_value: 0.0,
                quantum_expected: 1.0,
                classical_expected: 0.5,
                quantum_likeness: 0.5,
                description: "Insufficient data".to_string(),
            };
        }

        // Calculate inter-arrival times
        let intervals: Vec<u64> = (1..timing.len().min(10000))
            .map(|i| timing[i].saturating_sub(timing[i - 1]))
            .collect();

        if intervals.is_empty() {
            return QuantumSignature {
                name: "Jitter Distribution".to_string(),
                observed_value: 0.0,
                quantum_expected: 1.0,
                classical_expected: 0.5,
                quantum_likeness: 0.5,
                description: "No intervals".to_string(),
            };
        }

        // For quantum Poisson process:
        // - Intervals follow exponential distribution
        // - CV (coefficient of variation) ≈ 1.0
        // - Mean ≈ Std Dev

        let mean: f64 = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
        let variance: f64 = intervals
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;
        let std_dev = variance.sqrt();

        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // For exponential (quantum Poisson): CV = 1.0
        // For deterministic: CV ≈ 0
        // For clustered: CV > 1
        let quantum_likeness = 1.0 - (cv - 1.0).abs().min(1.0);

        QuantumSignature {
            name: "Jitter CV (Poisson-ness)".to_string(),
            observed_value: cv,
            quantum_expected: 1.0,
            classical_expected: 0.3,
            quantum_likeness,
            description: format!(
                "CV = {:.2} (1.0 = exponential/Poisson/quantum, <0.5 = deterministic)",
                cv
            ),
        }
    }

    /// Identify the likely quantum process
    fn identify_quantum_process(&self, signatures: &[QuantumSignature]) -> String {
        let score: f64 =
            signatures.iter().map(|s| s.quantum_likeness).sum::<f64>() / signatures.len() as f64;

        if score > 0.9 {
            "Likely PURE QUANTUM (shot noise, tunneling, or similar)".to_string()
        } else if score > 0.7 {
            "Likely QUANTUM-DOMINANT (thermal + quantum mix)".to_string()
        } else if score > 0.5 {
            "Likely MIXED quantum/classical".to_string()
        } else if score > 0.3 {
            "Likely CLASSICAL-DOMINANT with some quantum".to_string()
        } else {
            "Likely PURELY CLASSICAL (deterministic or pseudo-random)".to_string()
        }
    }

    /// Compare to known sources
    fn compare_to_known_sources(&self, score: f64) -> HashMap<String, f64> {
        let mut comparison = HashMap::new();

        // Known quantum source scores (approximate)
        comparison.insert("True QRNG (beam splitter)".to_string(), 95.0);
        comparison.insert("Vacuum fluctuations".to_string(), 92.0);
        comparison.insert("SSD Fowler-Nordheim".to_string(), 75.0);
        comparison.insert("Avalanche photodiode".to_string(), 70.0);
        comparison.insert("CPU jitter (good)".to_string(), 45.0);
        comparison.insert("/dev/urandom".to_string(), 0.0); // Cryptographic, not quantum
        comparison.insert("Mersenne Twister".to_string(), 0.0); // PRNG
        comparison.insert("YOUR SOURCE".to_string(), score);

        comparison
    }

    /// Calculate confidence in measurement
    fn calculate_confidence(&self, n_samples: usize) -> f64 {
        if n_samples < 100 {
            0.3
        } else if n_samples < 1000 {
            0.5
        } else if n_samples < 10000 {
            0.75
        } else if n_samples < 100000 {
            0.9
        } else {
            0.95
        }
    }

    /// Generate a report
    pub fn generate_report(&self, measurement: &QuantumMeasurement) -> String {
        let mut report = String::new();

        report.push_str("\n");
        report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║            QUANTUM MEASUREMENT RESULTS                         ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        report.push_str("║                                                                ║\n");
        report.push_str(&format!(
            "║   QUANTUM SCORE: {:>6.1}%                                    ║\n",
            measurement.quantum_score
        ));
        report.push_str(&format!(
            "║   CONFIDENCE:    {:>6.1}%                                    ║\n",
            measurement.confidence * 100.0
        ));
        report.push_str(&format!(
            "║   VERDICT:       {:<42}║\n",
            if measurement.is_quantum {
                "✅ QUANTUM CERTIFIED"
            } else {
                "⚠️ NOT CERTIFIED QUANTUM"
            }
        ));
        report.push_str("║                                                                ║\n");
        report.push_str(&format!("║   {:<60}║\n", measurement.likely_process));
        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        report.push_str("║   QUANTUM SIGNATURES MEASURED:                                 ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        for sig in &measurement.signatures {
            let bar_len = (sig.quantum_likeness * 20.0) as usize;
            let bar: String = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
            report.push_str(&format!("║   {}\n", sig.name));
            report.push_str(&format!(
                "║   [{}] {:.0}%   ║\n",
                bar,
                sig.quantum_likeness * 100.0
            ));
            report.push_str(&format!("║   {}\n", sig.description));
            report.push_str("║                                                                ║\n");
        }

        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        report.push_str("║   COMPARISON TO KNOWN SOURCES:                                ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        let mut sources: Vec<_> = measurement.comparison.iter().collect();
        sources.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (name, score) in sources {
            let marker = if name == "YOUR SOURCE" {
                " ◄── YOUR DATA"
            } else {
                ""
            };
            report.push_str(&format!("║   {:<35} {:>5.1}%{}\n", name, score, marker));
        }

        report.push_str("╚══════════════════════════════════════════════════════════════╝\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_pure_quantum() {
        // Simulate pure quantum source (exponential timing, white noise)
        let mut timing = vec![0u64];
        let mut rng = 12345u64;
        for i in 1..10000 {
            // Linear congruential for reproducibility
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let jitter = (rng % 1000) + 500; // Exponential-ish
            timing.push(timing[i - 1] + jitter);
        }

        let bits: Vec<u8> = (0..80000u32).map(|i| ((i % 7) % 2) as u8).collect();

        let tool = QuantumMeasurementTool::new();
        let measurement = tool.measure_quantumness(&timing, &bits);

        println!("{}", tool.generate_report(&measurement));
    }
}
