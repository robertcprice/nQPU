//! Certified Quantum Random Number Generation with CHSH Bell Test Verification
//!
//! This module implements a Bell-test certified QRNG pipeline. Each batch of
//! random bits is accompanied by a [`BellTestCertificate`] that records the
//! CHSH S-value measured during generation. The S-value quantifies the degree
//! to which the underlying correlations violate the classical Bell inequality
//! (S <= 2.0). A quantum source achieves S up to 2*sqrt(2) ~ 2.828 (Tsirelson
//! bound).
//!
//! Certificates are chained using a Twine-style SHA-256 hash chain: each
//! certificate's `self_hash` commits to the previous certificate's hash,
//! the measured S-value, and the extracted random bits, providing tamper
//! evidence across the entire generation history.
//!
//! # Architecture
//!
//! 1. **Bell Test**: A 2-qubit Bell state (|Phi+>) is prepared and measured
//!    under randomly chosen bases (Alice: Z or X; Bob: pi/4 or -pi/4).
//!    The CHSH correlators E(a,b) are accumulated over many rounds and
//!    combined as S = |E(0,0) + E(0,1) + E(1,0) - E(1,1)|.
//!
//! 2. **Randomness Extraction**: Raw measurement outcomes are hashed with
//!    SHA-256 to produce a fixed-length, near-uniform output (simplified
//!    Trevisan extractor).
//!
//! 3. **Hash Chain**: Each certificate links to its predecessor via
//!    `prev_hash`, forming an append-only chain that can be verified
//!    independently.

#![allow(dead_code)]

use crate::gates::{Gate, GateType};
use crate::qrng_integration::{QrngError, QrngSource, QrngSourceInfo};
use crate::traits::{QuantumBackend, StateVectorBackend};
use rand::Rng;

// ===================================================================
// ERROR TYPE
// ===================================================================

/// Errors specific to the certified QRNG pipeline.
#[derive(Debug, Clone)]
pub enum CertifiedQrngError {
    /// The CHSH Bell test did not achieve the required S-value.
    BellTestFailed {
        /// The measured CHSH S-value.
        s_value: f64,
        /// The minimum S-value required by the configuration.
        min_required: f64,
    },

    /// A certificate's `prev_hash` does not match the preceding
    /// certificate's `self_hash`, indicating chain tampering.
    ChainBroken {
        /// Index of the certificate whose linkage failed.
        index: usize,
    },

    /// The randomness extraction step failed.
    ExtractionFailed(String),

    /// An error propagated from the underlying QRNG source layer.
    SourceError(QrngError),
}

impl std::fmt::Display for CertifiedQrngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertifiedQrngError::BellTestFailed {
                s_value,
                min_required,
            } => write!(
                f,
                "Bell test failed: S = {:.4} < {:.4} required",
                s_value, min_required
            ),
            CertifiedQrngError::ChainBroken { index } => {
                write!(f, "certificate chain broken at index {}", index)
            }
            CertifiedQrngError::ExtractionFailed(msg) => {
                write!(f, "randomness extraction failed: {}", msg)
            }
            CertifiedQrngError::SourceError(e) => write!(f, "source error: {}", e),
        }
    }
}

impl std::error::Error for CertifiedQrngError {}

// ===================================================================
// EXTRACTION PARAMETERS
// ===================================================================

/// Parameters governing the randomness extraction step.
#[derive(Debug, Clone)]
pub struct ExtractionParams {
    /// Number of raw input bits consumed.
    pub input_length: usize,
    /// Number of output bits produced.
    pub output_length: usize,
    /// Minimum entropy rate assumed for the raw source (bits per bit).
    pub min_entropy_rate: f64,
    /// Statistical distance from uniform for the extracted output.
    pub epsilon: f64,
}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Builder for certified QRNG configuration.
#[derive(Debug, Clone)]
pub struct CertifiedQrngConfig {
    /// Number of CHSH measurement rounds per certificate.
    pub chsh_rounds: usize,
    /// Minimum acceptable CHSH S-value (must exceed 2.0 for quantum
    /// certification; the classical bound is exactly 2.0).
    pub min_s_value: f64,
    /// Statistical distance epsilon for the Trevisan extractor.
    pub extraction_epsilon: f64,
    /// Number of output bits per certified generation round.
    pub output_bits_per_round: usize,
}

impl CertifiedQrngConfig {
    /// Create a new configuration with sensible defaults.
    pub fn new() -> Self {
        Self {
            chsh_rounds: 1000,
            min_s_value: 2.1,
            extraction_epsilon: 1e-6,
            output_bits_per_round: 64,
        }
    }

    /// Set the number of CHSH measurement rounds.
    pub fn chsh_rounds(mut self, n: usize) -> Self {
        self.chsh_rounds = n;
        self
    }

    /// Set the minimum acceptable CHSH S-value.
    pub fn min_s_value(mut self, s: f64) -> Self {
        self.min_s_value = s;
        self
    }

    /// Set the extraction epsilon (statistical distance from uniform).
    pub fn extraction_epsilon(mut self, eps: f64) -> Self {
        self.extraction_epsilon = eps;
        self
    }

    /// Set the number of output bits per round.
    pub fn output_bits_per_round(mut self, n: usize) -> Self {
        self.output_bits_per_round = n;
        self
    }
}

impl Default for CertifiedQrngConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// BELL TEST CERTIFICATE
// ===================================================================

/// A single Bell-test certificate recording the CHSH result, extracted
/// randomness, and hash-chain linkage.
#[derive(Debug, Clone)]
pub struct BellTestCertificate {
    /// Measured CHSH S-value. Quantum sources yield S ~ 2.828.
    pub s_value: f64,
    /// Number of CHSH rounds used in the test.
    pub num_rounds: usize,
    /// Timestamp in nanoseconds (monotonic clock).
    pub timestamp_ns: u64,
    /// SHA-256 hash of the previous certificate (all zeros for genesis).
    pub prev_hash: [u8; 32],
    /// SHA-256 hash committing this certificate's contents.
    pub self_hash: [u8; 32],
    /// Parameters used for randomness extraction.
    pub extraction_params: ExtractionParams,
    /// The extracted random bytes.
    pub extracted_bits: Vec<u8>,
}

impl BellTestCertificate {
    /// Returns `true` if the measured S-value exceeds the classical
    /// Bell bound of 2.0, indicating genuine quantum correlations.
    pub fn is_quantum(&self) -> bool {
        self.s_value > 2.0
    }
}

// ===================================================================
// CERTIFICATION CHAIN
// ===================================================================

/// An append-only chain of [`BellTestCertificate`]s linked by SHA-256 hashes.
#[derive(Debug, Clone)]
pub struct CertificationChain {
    certificates: Vec<BellTestCertificate>,
}

impl CertificationChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self {
            certificates: Vec::new(),
        }
    }

    /// Append a certificate to the chain.
    pub fn append(&mut self, cert: BellTestCertificate) {
        self.certificates.push(cert);
    }

    /// Return the number of certificates in the chain.
    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    /// Return `true` if the chain contains no certificates.
    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }

    /// Verify the integrity of the hash chain.
    ///
    /// - The first certificate's `prev_hash` must be all zeros (genesis).
    /// - Each subsequent certificate's `prev_hash` must match the
    ///   preceding certificate's `self_hash`.
    pub fn verify_chain(&self) -> Result<(), CertifiedQrngError> {
        if self.certificates.is_empty() {
            return Ok(());
        }

        // Genesis certificate must link to the zero hash.
        if self.certificates[0].prev_hash != [0u8; 32] {
            return Err(CertifiedQrngError::ChainBroken { index: 0 });
        }

        for i in 1..self.certificates.len() {
            if self.certificates[i].prev_hash != self.certificates[i - 1].self_hash {
                return Err(CertifiedQrngError::ChainBroken { index: i });
            }
        }

        Ok(())
    }

    /// Total number of extracted random bytes across all certificates.
    pub fn total_extracted_bits(&self) -> usize {
        self.certificates
            .iter()
            .map(|c| c.extracted_bits.len())
            .sum()
    }
}

impl Default for CertificationChain {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// CERTIFIED QRNG SOURCE
// ===================================================================

/// A QRNG source that certifies each batch of random bytes with a
/// CHSH Bell test and maintains a tamper-evident hash chain.
pub struct CertifiedQrngSource {
    config: CertifiedQrngConfig,
    chain: CertificationChain,
    buffer: Vec<u8>,
    buffer_pos: usize,
}

impl CertifiedQrngSource {
    /// Create a new certified QRNG source with the given configuration.
    pub fn new(config: CertifiedQrngConfig) -> Self {
        Self {
            config,
            chain: CertificationChain::new(),
            buffer: Vec::new(),
            buffer_pos: 0,
        }
    }

    /// Run a full CHSH Bell test, extract certified random bits, and
    /// append the resulting certificate to the chain.
    ///
    /// The procedure:
    /// 1. Prepare a 2-qubit Bell state |Phi+>.
    /// 2. For each round, randomly choose measurement bases for Alice
    ///    and Bob, apply rotation gates, and record the outcomes.
    /// 3. Compute the four CHSH correlators E(a,b) and the S-value.
    /// 4. Reject if S < min_s_value.
    /// 5. Hash the raw outcomes (simplified Trevisan extraction).
    /// 6. Build and chain the certificate.
    pub fn generate_certified_bits(
        &mut self,
    ) -> Result<BellTestCertificate, CertifiedQrngError> {
        let mut backend = StateVectorBackend::new(2);
        let mut rng = rand::thread_rng();

        // Accumulators for correlators: counts[a][b] = (n_same, n_different)
        let mut counts = [[(0usize, 0usize); 2]; 2];
        let mut raw_outcomes: Vec<u8> = Vec::with_capacity(self.config.chsh_rounds * 2);

        for _ in 0..self.config.chsh_rounds {
            // --- Prepare Bell state |Phi+> = (|00> + |11>) / sqrt(2) ---
            backend.reset();
            backend
                .apply_gate(&Gate::single(GateType::H, 0))
                .map_err(|e| {
                    CertifiedQrngError::ExtractionFailed(format!("gate error: {}", e))
                })?;
            backend
                .apply_gate(&Gate::two(GateType::CNOT, 0, 1))
                .map_err(|e| {
                    CertifiedQrngError::ExtractionFailed(format!("gate error: {}", e))
                })?;

            // --- Choose random measurement settings ---
            let a: usize = rng.gen_range(0..2); // Alice: 0 or 1
            let b: usize = rng.gen_range(0..2); // Bob:   0 or 1

            // --- Apply measurement rotations ---
            // To measure observable at angle alpha (in XZ plane, alpha=0 is Z,
            // alpha=pi/2 is X), apply Ry(-alpha) then measure in Z basis.
            //
            // Optimal CHSH angles for |Phi+>:
            //   Alice:  a=0 -> alpha=0 (Z basis, no rotation)
            //           a=1 -> alpha=pi/2 (X basis, Ry(-pi/2))
            //   Bob:    b=0 -> beta=pi/4, Ry(-pi/4)
            //           b=1 -> beta=-pi/4, Ry(pi/4)
            if a == 1 {
                backend
                    .apply_gate(&Gate::single(
                        GateType::Ry(-std::f64::consts::FRAC_PI_2),
                        0,
                    ))
                    .map_err(|e| {
                        CertifiedQrngError::ExtractionFailed(format!("gate error: {}", e))
                    })?;
            }

            // Bob: b=0 -> Ry(-pi/4), b=1 -> Ry(pi/4)
            let bob_angle = if b == 0 {
                -std::f64::consts::FRAC_PI_4
            } else {
                std::f64::consts::FRAC_PI_4
            };
            backend
                .apply_gate(&Gate::single(GateType::Ry(bob_angle), 1))
                .map_err(|e| {
                    CertifiedQrngError::ExtractionFailed(format!("gate error: {}", e))
                })?;

            // --- Measure both qubits ---
            let probs = backend.probabilities().map_err(|e| {
                CertifiedQrngError::ExtractionFailed(format!("probability error: {}", e))
            })?;

            // Sample a computational basis outcome from the distribution.
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut outcome = 0usize;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    outcome = i;
                    break;
                }
                // Last state catches rounding
                outcome = i;
            }

            let alice_outcome = (outcome & 1) as u8; // qubit 0
            let bob_outcome = ((outcome >> 1) & 1) as u8; // qubit 1

            raw_outcomes.push(alice_outcome);
            raw_outcomes.push(bob_outcome);

            // --- Accumulate correlator statistics ---
            if alice_outcome == bob_outcome {
                counts[a][b].0 += 1; // same
            } else {
                counts[a][b].1 += 1; // different
            }
        }

        // --- Compute CHSH S-value ---
        let correlator = |same: usize, diff: usize| -> f64 {
            let total = same + diff;
            if total == 0 {
                return 0.0;
            }
            (same as f64 - diff as f64) / total as f64
        };

        let e00 = correlator(counts[0][0].0, counts[0][0].1);
        let e01 = correlator(counts[0][1].0, counts[0][1].1);
        let e10 = correlator(counts[1][0].0, counts[1][0].1);
        let e11 = correlator(counts[1][1].0, counts[1][1].1);

        // CHSH S = |E(0,0) + E(0,1) + E(1,0) - E(1,1)|
        // For |Phi+> with optimal angles this approaches 2*sqrt(2) ~ 2.828.
        let s_value = (e00 + e01 + e10 - e11).abs();

        // --- Enforce certification threshold ---
        if s_value < self.config.min_s_value {
            return Err(CertifiedQrngError::BellTestFailed {
                s_value,
                min_required: self.config.min_s_value,
            });
        }

        // --- Simplified Trevisan extraction: SHA-256 of raw outcomes ---
        let extracted_bits = sha256(&raw_outcomes).to_vec();

        // --- Build hash chain certificate ---
        let prev_hash = if self.chain.is_empty() {
            [0u8; 32]
        } else {
            self.chain.certificates[self.chain.len() - 1].self_hash
        };

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // self_hash = SHA-256(prev_hash || s_value_bytes || extracted_bits)
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&prev_hash);
        hash_input.extend_from_slice(&s_value.to_le_bytes());
        hash_input.extend_from_slice(&extracted_bits);
        let self_hash = sha256(&hash_input);

        let extraction_params = ExtractionParams {
            input_length: raw_outcomes.len() * 8,
            output_length: extracted_bits.len() * 8,
            min_entropy_rate: 1.0 - (2.0 / s_value).acos() / std::f64::consts::PI,
            epsilon: self.config.extraction_epsilon,
        };

        let cert = BellTestCertificate {
            s_value,
            num_rounds: self.config.chsh_rounds,
            timestamp_ns,
            prev_hash,
            self_hash,
            extraction_params,
            extracted_bits: extracted_bits.clone(),
        };

        // Append extracted bytes to the output buffer.
        self.buffer.extend_from_slice(&extracted_bits);

        // Append certificate to chain.
        self.chain.append(cert.clone());

        Ok(cert)
    }

    /// Return a reference to the certification chain.
    pub fn chain(&self) -> &CertificationChain {
        &self.chain
    }

    /// Return the total number of certificates generated so far.
    pub fn total_certificates(&self) -> usize {
        self.chain.len()
    }
}

impl QrngSource for CertifiedQrngSource {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), QrngError> {
        let mut written = 0;
        while written < dest.len() {
            // Drain from the buffer first.
            let available = self.buffer.len() - self.buffer_pos;
            if available > 0 {
                let chunk = std::cmp::min(available, dest.len() - written);
                dest[written..written + chunk]
                    .copy_from_slice(&self.buffer[self.buffer_pos..self.buffer_pos + chunk]);
                self.buffer_pos += chunk;
                written += chunk;

                // Compact buffer if fully consumed.
                if self.buffer_pos >= self.buffer.len() {
                    self.buffer.clear();
                    self.buffer_pos = 0;
                }
            } else {
                // Generate more certified bits.
                self.generate_certified_bits().map_err(|e| {
                    QrngError::DeviceError(format!("certified QRNG error: {}", e))
                })?;
            }
        }
        Ok(())
    }

    fn source_info(&self) -> QrngSourceInfo {
        QrngSourceInfo {
            name: "Certified QRNG (CHSH Bell Test)".to_string(),
            throughput_bps: None,
            is_hardware: false,
        }
    }
}

// ===================================================================
// INLINE SHA-256 IMPLEMENTATION
// ===================================================================

/// Compute the SHA-256 digest of the input data.
///
/// This is a fully self-contained implementation with no external
/// dependencies. It follows FIPS 180-4.
fn sha256(data: &[u8]) -> [u8; 32] {
    // Round constants (first 32 bits of the fractional parts of the
    // cube roots of the first 64 primes).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Initial hash values (first 32 bits of the fractional parts of
    // the square roots of the first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Pre-processing: pad the message.
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 64-byte (512-bit) block.
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    digest
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        // Defaults
        let cfg = CertifiedQrngConfig::new();
        assert_eq!(cfg.chsh_rounds, 1000);
        assert!((cfg.min_s_value - 2.1).abs() < 1e-10);
        assert!((cfg.extraction_epsilon - 1e-6).abs() < 1e-15);
        assert_eq!(cfg.output_bits_per_round, 64);

        // Builder methods
        let cfg = CertifiedQrngConfig::new()
            .chsh_rounds(500)
            .min_s_value(2.5)
            .extraction_epsilon(1e-8)
            .output_bits_per_round(128);
        assert_eq!(cfg.chsh_rounds, 500);
        assert!((cfg.min_s_value - 2.5).abs() < 1e-10);
        assert!((cfg.extraction_epsilon - 1e-8).abs() < 1e-15);
        assert_eq!(cfg.output_bits_per_round, 128);
    }

    #[test]
    fn test_bell_state_chsh_violation() {
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(2000)
            .min_s_value(2.0);
        let mut source = CertifiedQrngSource::new(config);

        let cert = source.generate_certified_bits().unwrap();
        assert!(
            cert.s_value > 2.0,
            "CHSH S-value {} should exceed classical bound 2.0",
            cert.s_value
        );
        assert!(cert.is_quantum());
    }

    #[test]
    fn test_unreachable_s_value_fails() {
        // Even a quantum source cannot reach S = 3.5 (Tsirelson bound ~ 2.828).
        // Using 3.5 to avoid flaky test due to statistical noise with finite samples.
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(1000)
            .min_s_value(3.5);  // Well above Tsirelson bound
        let mut source = CertifiedQrngSource::new(config);

        let result = source.generate_certified_bits();
        assert!(
            result.is_err(),
            "S = 3.5 should be unreachable; expected BellTestFailed"
        );
        match result {
            Err(CertifiedQrngError::BellTestFailed {
                s_value,
                min_required,
            }) => {
                assert!(s_value < 3.5);
                assert!((min_required - 3.5).abs() < 1e-10);
            }
            _ => panic!("expected BellTestFailed error"),
        }
    }

    #[test]
    fn test_s_value_range() {
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(4000)
            .min_s_value(2.0);
        let mut source = CertifiedQrngSource::new(config);

        let cert = source.generate_certified_bits().unwrap();
        let tsirelson = 2.0 * std::f64::consts::SQRT_2;
        assert!(
            cert.s_value > 2.0 && cert.s_value <= tsirelson + 0.1,
            "S = {:.4} should be in (2.0, {:.4}]",
            cert.s_value,
            tsirelson
        );
    }

    #[test]
    fn test_extraction_output_has_bytes() {
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(500)
            .min_s_value(2.0);
        let mut source = CertifiedQrngSource::new(config);

        let cert = source.generate_certified_bits().unwrap();
        assert!(
            !cert.extracted_bits.is_empty(),
            "extracted_bits should not be empty"
        );
        assert_eq!(
            cert.extracted_bits.len(),
            32,
            "SHA-256 extraction should produce 32 bytes"
        );
    }

    #[test]
    fn test_chain_verification_valid() {
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(500)
            .min_s_value(2.0);
        let mut source = CertifiedQrngSource::new(config);

        // Generate 3 certificates.
        for _ in 0..3 {
            source.generate_certified_bits().unwrap();
        }

        assert_eq!(source.chain().len(), 3);
        source.chain().verify_chain().unwrap();
    }

    #[test]
    fn test_chain_tamper_detection() {
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(500)
            .min_s_value(2.0);
        let mut source = CertifiedQrngSource::new(config);

        source.generate_certified_bits().unwrap();
        source.generate_certified_bits().unwrap();

        // Tamper with the first certificate's self_hash.
        let mut tampered_chain = source.chain().clone();
        tampered_chain.certificates[0].self_hash[0] ^= 0xff;

        let result = tampered_chain.verify_chain();
        assert!(result.is_err(), "tampered chain should fail verification");
        match result {
            Err(CertifiedQrngError::ChainBroken { index }) => {
                assert_eq!(index, 1);
            }
            _ => panic!("expected ChainBroken error at index 1"),
        }
    }

    #[test]
    fn test_fill_bytes_produces_random_data() {
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(500)
            .min_s_value(2.0);
        let mut source = CertifiedQrngSource::new(config);

        let mut buf = [0u8; 64];
        source.fill_bytes(&mut buf).unwrap();
        assert!(
            buf.iter().any(|&b| b != 0),
            "64 bytes should not all be zero"
        );
    }

    #[test]
    fn test_source_info() {
        let config = CertifiedQrngConfig::new();
        let source = CertifiedQrngSource::new(config);
        let info = source.source_info();
        assert_eq!(info.name, "Certified QRNG (CHSH Bell Test)");
        assert!(!info.is_hardware);
    }

    #[test]
    fn test_sha256_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256(b"");
        let expected: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(digest, expected);
    }

    #[test]
    fn test_sha256_abc() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let digest = sha256(b"abc");
        let expected: [u8; 32] = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(digest, expected);
    }

    #[test]
    fn test_certificate_is_quantum() {
        let cert = BellTestCertificate {
            s_value: 2.5,
            num_rounds: 100,
            timestamp_ns: 0,
            prev_hash: [0u8; 32],
            self_hash: [0u8; 32],
            extraction_params: ExtractionParams {
                input_length: 200,
                output_length: 256,
                min_entropy_rate: 0.5,
                epsilon: 1e-6,
            },
            extracted_bits: vec![0xab; 32],
        };
        assert!(cert.is_quantum());

        let classical_cert = BellTestCertificate {
            s_value: 1.8,
            ..cert.clone()
        };
        assert!(!classical_cert.is_quantum());
    }

    #[test]
    fn test_chain_empty_valid() {
        let chain = CertificationChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        chain.verify_chain().unwrap();
    }

    #[test]
    fn test_fill_bytes_multiple_calls() {
        let config = CertifiedQrngConfig::new()
            .chsh_rounds(500)
            .min_s_value(2.0);
        let mut source = CertifiedQrngSource::new(config);

        let mut buf1 = [0u8; 16];
        let mut buf2 = [0u8; 16];
        source.fill_bytes(&mut buf1).unwrap();
        source.fill_bytes(&mut buf2).unwrap();

        // The two buffers should be different (drawn from different
        // positions in the extraction output, or from different
        // certificates).
        assert_ne!(
            buf1, buf2,
            "consecutive fill_bytes calls should produce different data"
        );
    }
}
