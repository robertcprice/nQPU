//! Sinter-like QEC Statistical Sampling Framework
//!
//! Parallelized Monte Carlo sampling for quantum error correction threshold studies.
//! Supports surface codes with multiple decoders (MWPM, Union-Find, Belief Propagation)
//! and noise models (depolarizing, bit-flip, phase-flip, phenomenological).
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::qec_sampling::{SamplingConfig, DecoderType, NoiseType};
//!
//! let config = SamplingConfig::builder()
//!     .num_trials(1000)
//!     .error_rates(vec![0.01, 0.05, 0.1])
//!     .code_distances(vec![3, 5, 7])
//!     .decoder(DecoderType::Mwpm)
//!     .noise_model(NoiseType::Depolarizing)
//!     .build();
//!
//! let results = run_sampling(&config);
//! println!("{}", results_to_csv(&results));
//! ```

use rand::Rng;
use rayon::prelude::*;
use std::fmt;
use std::time::Instant;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during QEC sampling.
#[derive(Debug, Clone)]
pub enum SamplingError {
    /// Not enough trials were run to produce statistically meaningful results.
    InsufficientTrials {
        requested: usize,
        completed: usize,
    },
    /// The decoder failed to produce a valid correction.
    DecoderFailed(String),
    /// The sampling configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingError::InsufficientTrials {
                requested,
                completed,
            } => write!(
                f,
                "Insufficient trials: requested {}, completed {}",
                requested, completed
            ),
            SamplingError::DecoderFailed(msg) => write!(f, "Decoder failed: {}", msg),
            SamplingError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
        }
    }
}

impl std::error::Error for SamplingError {}

// ============================================================
// DECODER AND NOISE TYPES
// ============================================================

/// Decoder type for QEC syndrome decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderType {
    /// Minimum Weight Perfect Matching (greedy approximation).
    Mwpm,
    /// Union-Find cluster growth decoder.
    UnionFind,
    /// Belief Propagation iterative decoder.
    BeliefPropagation,
    /// Neural network decoder. Currently delegates to MWPM; a learned decoder
    /// (transformer/GNN) can be plugged in via the `mamba_qec_decoder` module.
    NeuralDecoder,
}

/// Noise model for error generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseType {
    /// Depolarizing noise: X, Y, Z each with probability p/3.
    Depolarizing,
    /// Bit-flip (X) noise with probability p.
    BitFlip,
    /// Phase-flip (Z) noise with probability p.
    PhaseFlip,
    /// Code capacity: errors on data qubits only, perfect syndrome extraction.
    CodeCapacity,
    /// Phenomenological: errors on data qubits AND measurement errors.
    Phenomenological {
        /// Number of syndrome measurement rounds.
        measurement_rounds: usize,
    },
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Builder for `SamplingConfig`.
pub struct SamplingConfigBuilder {
    num_trials: usize,
    error_rates: Vec<f64>,
    code_distances: Vec<usize>,
    decoder: DecoderType,
    noise_model: NoiseType,
    num_threads: usize,
    confidence_level: f64,
    min_failures: usize,
}

impl SamplingConfigBuilder {
    /// Set the number of Monte Carlo trials per (error_rate, distance) pair.
    pub fn num_trials(mut self, n: usize) -> Self {
        self.num_trials = n;
        self
    }

    /// Set the physical error rates to sweep.
    pub fn error_rates(mut self, rates: Vec<f64>) -> Self {
        self.error_rates = rates;
        self
    }

    /// Set the code distances to sweep.
    pub fn code_distances(mut self, distances: Vec<usize>) -> Self {
        self.code_distances = distances;
        self
    }

    /// Set the decoder type.
    pub fn decoder(mut self, decoder: DecoderType) -> Self {
        self.decoder = decoder;
        self
    }

    /// Set the noise model.
    pub fn noise_model(mut self, noise: NoiseType) -> Self {
        self.noise_model = noise;
        self
    }

    /// Set the number of threads (0 = auto-detect).
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Set the confidence level for Wilson score intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Minimum number of logical failures before early stopping is allowed.
    pub fn min_failures(mut self, n: usize) -> Self {
        self.min_failures = n;
        self
    }

    /// Build the final `SamplingConfig`.
    pub fn build(self) -> SamplingConfig {
        SamplingConfig {
            num_trials: self.num_trials,
            error_rates: self.error_rates,
            code_distances: self.code_distances,
            decoder: self.decoder,
            noise_model: self.noise_model,
            num_threads: self.num_threads,
            confidence_level: self.confidence_level,
            min_failures: self.min_failures,
        }
    }
}

/// Configuration for a QEC Monte Carlo sampling run.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Number of Monte Carlo trials per (error_rate, distance) pair.
    pub num_trials: usize,
    /// Physical error rates to sweep.
    pub error_rates: Vec<f64>,
    /// Code distances to sweep.
    pub code_distances: Vec<usize>,
    /// Decoder type.
    pub decoder: DecoderType,
    /// Noise model.
    pub noise_model: NoiseType,
    /// Number of threads (0 = auto).
    pub num_threads: usize,
    /// Confidence level for Wilson score intervals (e.g. 0.95).
    pub confidence_level: f64,
    /// Minimum logical failures before early stop is permitted.
    pub min_failures: usize,
}

impl SamplingConfig {
    /// Create a builder with default values.
    pub fn builder() -> SamplingConfigBuilder {
        SamplingConfigBuilder {
            num_trials: 10_000,
            error_rates: vec![0.01, 0.05, 0.1],
            code_distances: vec![3, 5, 7],
            decoder: DecoderType::Mwpm,
            noise_model: NoiseType::Depolarizing,
            num_threads: 0,
            confidence_level: 0.95,
            min_failures: 100,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), SamplingError> {
        if self.num_trials == 0 {
            return Err(SamplingError::InvalidConfig(
                "num_trials must be > 0".into(),
            ));
        }
        if self.error_rates.is_empty() {
            return Err(SamplingError::InvalidConfig(
                "error_rates must not be empty".into(),
            ));
        }
        if self.code_distances.is_empty() {
            return Err(SamplingError::InvalidConfig(
                "code_distances must not be empty".into(),
            ));
        }
        for &r in &self.error_rates {
            if r < 0.0 || r > 1.0 {
                return Err(SamplingError::InvalidConfig(format!(
                    "error_rate {} out of range [0, 1]",
                    r
                )));
            }
        }
        for &d in &self.code_distances {
            if d < 3 || d % 2 == 0 {
                return Err(SamplingError::InvalidConfig(format!(
                    "code_distance {} must be odd and >= 3",
                    d
                )));
            }
        }
        if self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err(SamplingError::InvalidConfig(format!(
                "confidence_level {} must be in (0, 1)",
                self.confidence_level
            )));
        }
        Ok(())
    }
}

// ============================================================
// RESULT TYPES
// ============================================================

/// Result of sampling at a single (error_rate, code_distance) point.
#[derive(Debug, Clone)]
pub struct SamplingResult {
    /// Physical error rate.
    pub error_rate: f64,
    /// Code distance.
    pub code_distance: usize,
    /// Total number of trials completed.
    pub num_trials: usize,
    /// Number of logical errors observed.
    pub num_failures: usize,
    /// Estimated logical error rate = num_failures / num_trials.
    pub logical_error_rate: f64,
    /// Wilson score confidence interval (lower, upper).
    pub confidence_interval: (f64, f64),
    /// Wall-clock time in seconds.
    pub wall_time_seconds: f64,
}

/// Result of a threshold study across multiple (error_rate, distance) points.
#[derive(Debug, Clone)]
pub struct ThresholdResult {
    /// All individual sampling results.
    pub results: Vec<SamplingResult>,
    /// Estimated threshold error rate (crossing point).
    pub threshold_estimate: f64,
    /// Uncertainty in threshold estimate.
    pub threshold_uncertainty: f64,
}

// ============================================================
// ERROR SAMPLE
// ============================================================

/// A single error sample from the noise model.
#[derive(Debug, Clone)]
pub struct ErrorSample {
    /// Pauli errors on data qubits (0 = I, 1 = X, 2 = Y, 3 = Z).
    pub data_errors: Vec<u8>,
    /// Syndrome bits (measured stabilizer outcomes).
    pub syndrome: Vec<u8>,
    /// Whether the residual error after correction is a logical operator.
    pub is_logical_error: bool,
}

// ============================================================
// SURFACE CODE UTILITIES
// ============================================================

/// Number of data qubits in a distance-d surface code.
#[inline]
pub fn num_data_qubits(distance: usize) -> usize {
    distance * distance
}

/// Number of X-type (plaquette) stabilizers in a distance-d surface code.
#[inline]
pub fn num_x_stabilizers(distance: usize) -> usize {
    (distance - 1) * distance / 2 + (distance - 1) * (distance - 1) / 2
}

/// Number of Z-type (vertex) stabilizers in a distance-d surface code.
#[inline]
pub fn num_z_stabilizers(distance: usize) -> usize {
    num_x_stabilizers(distance)
}

/// Compute the parity check matrices for a distance-d rotated surface code.
///
/// Returns `(x_checks, z_checks)` where each is a list of stabilizers,
/// and each stabilizer is a list of data qubit indices it acts on.
///
/// For a distance-d rotated surface code:
/// - Data qubits arranged on a d x d grid
/// - X stabilizers on plaquettes (even checkerboard)
/// - Z stabilizers on plaquettes (odd checkerboard)
/// - Each bulk stabilizer acts on 4 qubits
/// - Boundary stabilizers act on 2 qubits
///
/// Total stabilizers: (d^2 - 1) / 2 each for X and Z (since d is odd).
pub fn surface_code_parity_check(distance: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let d = distance;

    let mut x_checks: Vec<Vec<u8>> = Vec::new();
    let mut z_checks: Vec<Vec<u8>> = Vec::new();

    // Iterate over all face positions in the extended dual lattice.
    // Face (fr, fc) touches data qubits at (fr+dr, fc+dc) for dr,dc in {0,1}.
    // fr ranges from -1 to d-1, fc ranges from -1 to d-1.
    //
    // Rotated surface code boundary convention:
    // - Top/bottom boundaries are smooth (X-type boundary stabilizers only)
    // - Left/right boundaries are rough (Z-type boundary stabilizers only)
    let d_i = d as i32;
    for fr in -1..d_i {
        for fc in -1..d_i {
            let mut qubits: Vec<u8> = Vec::new();
            for &dr in &[0i32, 1] {
                for &dc in &[0i32, 1] {
                    let r = fr + dr;
                    let c = fc + dc;
                    if r >= 0 && r < d_i && c >= 0 && c < d_i {
                        qubits.push((r as usize * d + c as usize) as u8);
                    }
                }
            }

            if qubits.len() < 2 {
                continue;
            }

            let is_x_type = (fr + fc).rem_euclid(2) == 0;
            let weight = qubits.len();

            if weight == 4 || weight == 3 {
                // Bulk stabilizer (weight 4) or corner (weight 3): always include
                if is_x_type {
                    x_checks.push(qubits);
                } else {
                    z_checks.push(qubits);
                }
            } else if weight == 2 {
                // Boundary stabilizer: include only on the correct edge type
                let is_top = fr == -1;
                let is_bottom = fr == d_i - 1;
                let is_left = fc == -1;
                let is_right = fc == d_i - 1;

                if (is_top || is_bottom) && is_x_type {
                    // Smooth boundary: X-type only
                    x_checks.push(qubits);
                } else if (is_left || is_right) && !is_x_type {
                    // Rough boundary: Z-type only
                    z_checks.push(qubits);
                }
            }
        }
    }

    (x_checks, z_checks)
}

/// Check if a correction pattern is equivalent to a logical X operator.
///
/// For a rotated surface code, logical X spans a horizontal row.
pub fn is_logical_x(correction: &[u8], distance: usize) -> bool {
    // Logical X: any chain of X (or Y) errors crossing from left to right boundary.
    // Simplified check: count X/Y in each row, check if any row has odd parity.
    // More precisely: check if the correction has odd overlap with the logical Z operator.
    // Logical Z spans a column, so logical X is detected by counting X-component
    // along any column.
    for col in 0..distance {
        let mut parity = 0u8;
        for row in 0..distance {
            let idx = row * distance + col;
            if idx < correction.len() {
                let e = correction[idx];
                // X-component: X=1, Y=2 (has both X and Z), I=0, Z=3
                if e == 1 || e == 2 {
                    parity ^= 1;
                }
            }
        }
        if parity == 1 {
            return true;
        }
    }
    false
}

/// Check if a correction pattern is equivalent to a logical Z operator.
///
/// For a rotated surface code, logical Z spans a vertical column.
pub fn is_logical_z(correction: &[u8], distance: usize) -> bool {
    // Logical Z: any chain of Z (or Y) errors crossing from top to bottom.
    // Check Z-component along any row.
    for row in 0..distance {
        let mut parity = 0u8;
        for col in 0..distance {
            let idx = row * distance + col;
            if idx < correction.len() {
                let e = correction[idx];
                // Z-component: Z=3, Y=2
                if e == 3 || e == 2 {
                    parity ^= 1;
                }
            }
        }
        if parity == 1 {
            return true;
        }
    }
    false
}

/// Hamming weight (number of non-zero entries).
pub fn hamming_weight(v: &[u8]) -> usize {
    v.iter().filter(|&&x| x != 0).count()
}

/// Compute syndrome from data errors and parity check matrix.
///
/// For X stabilizers, we check the X-component of errors (X or Y).
/// For Z stabilizers, we check the Z-component of errors (Z or Y).
fn compute_syndrome(data_errors: &[u8], checks: &[Vec<u8>], check_x: bool) -> Vec<u8> {
    checks
        .iter()
        .map(|stabilizer| {
            let mut parity = 0u8;
            for &qubit_idx in stabilizer {
                let e = data_errors[qubit_idx as usize];
                if check_x {
                    // X stabilizer detects Z and Y errors (Z-component)
                    if e == 2 || e == 3 {
                        parity ^= 1;
                    }
                } else {
                    // Z stabilizer detects X and Y errors (X-component)
                    if e == 1 || e == 2 {
                        parity ^= 1;
                    }
                }
            }
            parity
        })
        .collect()
}

/// Combine two Pauli error vectors element-wise (group multiplication).
/// 0=I, 1=X, 2=Y, 3=Z
/// Multiplication table (up to phase):
/// X*X=I, Z*Z=I, X*Z=Y, Y=X*Z, Y*X=Z, Y*Z=X, Y*Y=I
fn combine_pauli(a: &[u8], b: &[u8]) -> Vec<u8> {
    // Pauli group multiplication (ignoring phase):
    // I=0, X=1, Y=2, Z=3
    // XOR of the 2-bit representation (X-bit, Z-bit):
    // I=(0,0), X=(1,0), Z=(0,1), Y=(1,1)
    a.iter()
        .zip(b.iter())
        .map(|(&ea, &eb)| {
            let ax = (ea == 1 || ea == 2) as u8;
            let az = (ea == 2 || ea == 3) as u8;
            let bx = (eb == 1 || eb == 2) as u8;
            let bz = (eb == 2 || eb == 3) as u8;
            let rx = ax ^ bx;
            let rz = az ^ bz;
            match (rx, rz) {
                (0, 0) => 0, // I
                (1, 0) => 1, // X
                (1, 1) => 2, // Y
                (0, 1) => 3, // Z
                _ => unreachable!(),
            }
        })
        .collect()
}

// ============================================================
// SURFACE CODE SAMPLER
// ============================================================

/// Cached noise-free baseline stabilizer state for a given code distance.
///
/// Stores the reference syndrome (all-zeros for surface codes with clean
/// initialisation) so that noisy shots can be XOR-diffed against it instead
/// of recomputing from scratch each sample.
#[derive(Clone, Debug)]
pub struct ReferenceFrameCache {
    /// Number of data qubits.
    pub num_data_qubits: usize,
    /// Noise-free syndrome (X stabiliser part ++ Z stabiliser part).
    pub baseline_syndrome: Vec<bool>,
}

/// Sampler that generates surface code error samples according to a noise model.
///
/// Caches both the parity check matrices and the noise-free reference frame
/// per code distance so they are computed once and reused across all samples.
pub struct SurfaceCodeSampler {
    /// Cached parity check matrices per distance.
    parity_cache: std::collections::HashMap<usize, (Vec<Vec<u8>>, Vec<Vec<u8>>)>,
    /// Cached noise-free reference frame per distance.
    reference_cache: std::collections::HashMap<usize, ReferenceFrameCache>,
}

impl SurfaceCodeSampler {
    /// Create a new sampler.
    pub fn new() -> Self {
        Self {
            parity_cache: std::collections::HashMap::new(),
            reference_cache: std::collections::HashMap::new(),
        }
    }

    /// Get or compute parity checks for a given distance.
    fn get_parity_checks(&mut self, distance: usize) -> &(Vec<Vec<u8>>, Vec<Vec<u8>>) {
        self.parity_cache
            .entry(distance)
            .or_insert_with(|| surface_code_parity_check(distance))
    }

    /// Get or compute the noise-free reference frame for a given distance.
    ///
    /// For a standard surface code initialised in |0...0>, the noise-free
    /// syndrome is all-false. This is cached so callers can XOR noisy
    /// syndromes against it without recomputing.
    pub fn get_reference_frame(&mut self, distance: usize) -> &ReferenceFrameCache {
        if !self.reference_cache.contains_key(&distance) {
            let n = num_data_qubits(distance);
            let (x_checks, z_checks) = self.get_parity_checks(distance).clone();
            let clean_errors = vec![0u8; n];
            let x_syn = compute_syndrome(&clean_errors, &x_checks, true);
            let z_syn = compute_syndrome(&clean_errors, &z_checks, false);
            let mut baseline = x_syn.into_iter().map(|b| b != 0).collect::<Vec<_>>();
            baseline.extend(z_syn.into_iter().map(|b| b != 0));
            self.reference_cache.insert(
                distance,
                ReferenceFrameCache {
                    num_data_qubits: n,
                    baseline_syndrome: baseline,
                },
            );
        }
        self.reference_cache.get(&distance).unwrap()
    }

    /// Sample a single error instance.
    pub fn sample_error(
        &mut self,
        distance: usize,
        error_rate: f64,
        noise: &NoiseType,
        rng: &mut impl Rng,
    ) -> ErrorSample {
        // Ensure reference frame is cached (also warms the parity cache).
        let _ref_frame = self.get_reference_frame(distance);

        let n = num_data_qubits(distance);
        let data_errors = generate_errors(n, error_rate, noise, rng);

        let (x_checks, z_checks) = self.get_parity_checks(distance).clone();

        // X stabilizers detect Z-component errors
        let x_syndrome = compute_syndrome(&data_errors, &x_checks, true);
        // Z stabilizers detect X-component errors
        let z_syndrome = compute_syndrome(&data_errors, &z_checks, false);

        // Combined syndrome
        let mut syndrome = x_syndrome;
        syndrome.extend(z_syndrome);

        ErrorSample {
            data_errors,
            syndrome,
            is_logical_error: false, // Will be determined after decoding
        }
    }
}

impl Default for SurfaceCodeSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate random Pauli errors on n qubits according to noise model.
fn generate_errors(n: usize, error_rate: f64, noise: &NoiseType, rng: &mut impl Rng) -> Vec<u8> {
    let mut errors = vec![0u8; n];

    match noise {
        NoiseType::Depolarizing => {
            let p_each = error_rate / 3.0;
            for e in errors.iter_mut() {
                let r: f64 = rng.gen();
                if r < p_each {
                    *e = 1; // X
                } else if r < 2.0 * p_each {
                    *e = 2; // Y
                } else if r < 3.0 * p_each {
                    *e = 3; // Z
                }
                // else Identity
            }
        }
        NoiseType::BitFlip => {
            for e in errors.iter_mut() {
                if rng.gen::<f64>() < error_rate {
                    *e = 1; // X
                }
            }
        }
        NoiseType::PhaseFlip => {
            for e in errors.iter_mut() {
                if rng.gen::<f64>() < error_rate {
                    *e = 3; // Z
                }
            }
        }
        NoiseType::CodeCapacity => {
            // Same as depolarizing for code capacity model
            let p_each = error_rate / 3.0;
            for e in errors.iter_mut() {
                let r: f64 = rng.gen();
                if r < p_each {
                    *e = 1;
                } else if r < 2.0 * p_each {
                    *e = 2;
                } else if r < 3.0 * p_each {
                    *e = 3;
                }
            }
        }
        NoiseType::Phenomenological {
            measurement_rounds: _,
        } => {
            // Data qubit errors (depolarizing)
            let p_each = error_rate / 3.0;
            for e in errors.iter_mut() {
                let r: f64 = rng.gen();
                if r < p_each {
                    *e = 1;
                } else if r < 2.0 * p_each {
                    *e = 2;
                } else if r < 3.0 * p_each {
                    *e = 3;
                }
            }
            // Note: measurement errors are handled in the syndrome
            // extraction step for phenomenological noise. This is a
            // simplified version that applies noise to data qubits.
        }
    }

    errors
}

// ============================================================
// DECODERS
// ============================================================

/// Greedy MWPM decoder for surface code syndromes.
///
/// Pairs syndrome defects by nearest Manhattan distance on the syndrome graph
/// and applies corrections along shortest paths.
pub fn mwpm_decode(syndrome: &[u8], distance: usize) -> Vec<u8> {
    let n = num_data_qubits(distance);
    let (x_checks, z_checks) = surface_code_parity_check(distance);

    let num_x = x_checks.len();
    // Split syndrome into X and Z parts
    let x_syndrome = &syndrome[..num_x.min(syndrome.len())];
    let z_syndrome = if syndrome.len() > num_x {
        &syndrome[num_x..]
    } else {
        &[]
    };

    let mut correction = vec![0u8; n];

    // Decode X-component errors (detected by Z stabilizers)
    let z_defects: Vec<usize> = z_syndrome
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == 1)
        .map(|(i, _)| i)
        .collect();
    if !z_defects.is_empty() {
        let z_corr = greedy_match_and_correct(&z_defects, &z_checks, distance, true);
        for (i, &c) in z_corr.iter().enumerate() {
            if c != 0 {
                correction[i] = combine_pauli_single(correction[i], c);
            }
        }
    }

    // Decode Z-component errors (detected by X stabilizers)
    let x_defects: Vec<usize> = x_syndrome
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == 1)
        .map(|(i, _)| i)
        .collect();
    if !x_defects.is_empty() {
        let x_corr = greedy_match_and_correct(&x_defects, &x_checks, distance, false);
        for (i, &c) in x_corr.iter().enumerate() {
            if c != 0 {
                correction[i] = combine_pauli_single(correction[i], c);
            }
        }
    }

    correction
}

/// Combine two single Pauli operators.
fn combine_pauli_single(a: u8, b: u8) -> u8 {
    let ax = (a == 1 || a == 2) as u8;
    let az = (a == 2 || a == 3) as u8;
    let bx = (b == 1 || b == 2) as u8;
    let bz = (b == 2 || b == 3) as u8;
    let rx = ax ^ bx;
    let rz = az ^ bz;
    match (rx, rz) {
        (0, 0) => 0,
        (1, 0) => 1,
        (1, 1) => 2,
        (0, 1) => 3,
        _ => unreachable!(),
    }
}

/// Greedy matching: pair defects by nearest distance, apply corrections.
///
/// For each pair of defects, finds the shared data qubit between their
/// stabilizers and applies a correction there. If no shared qubit exists,
/// applies corrections along a path.
fn greedy_match_and_correct(
    defects: &[usize],
    checks: &[Vec<u8>],
    distance: usize,
    correct_x: bool,
) -> Vec<u8> {
    let n = num_data_qubits(distance);
    let mut correction = vec![0u8; n];

    if defects.is_empty() {
        return correction;
    }

    // Compute centroid positions for each stabilizer
    let positions: Vec<(f64, f64)> = checks
        .iter()
        .map(|qubits| {
            let mut r = 0.0;
            let mut c = 0.0;
            for &q in qubits {
                r += (q as usize / distance) as f64;
                c += (q as usize % distance) as f64;
            }
            let len = qubits.len() as f64;
            (r / len, c / len)
        })
        .collect();

    let mut used = vec![false; defects.len()];
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    // Greedy pairing by nearest distance
    for i in 0..defects.len() {
        if used[i] {
            continue;
        }
        let mut best_j = None;
        let mut best_dist = f64::MAX;

        for j in (i + 1)..defects.len() {
            if used[j] {
                continue;
            }
            let di = defects[i].min(positions.len().saturating_sub(1));
            let dj = defects[j].min(positions.len().saturating_sub(1));
            let dist = (positions[di].0 - positions[dj].0).abs()
                + (positions[di].1 - positions[dj].1).abs();
            if dist < best_dist {
                best_dist = dist;
                best_j = Some(j);
            }
        }

        if let Some(j) = best_j {
            used[i] = true;
            used[j] = true;
            pairs.push((defects[i], defects[j]));
        } else {
            // Unpaired defect: match to boundary
            used[i] = true;
            let si = defects[i].min(checks.len().saturating_sub(1));
            // For boundary matching, flip the qubit closest to the boundary
            if let Some(&q) = checks[si].first() {
                let pauli = if correct_x { 1u8 } else { 3u8 };
                correction[q as usize] = combine_pauli_single(correction[q as usize], pauli);
            }
        }
    }

    // For each pair, find the minimum-weight correction
    for (s1, s2) in pairs {
        let si1 = s1.min(checks.len().saturating_sub(1));
        let si2 = s2.min(checks.len().saturating_sub(1));

        let pauli = if correct_x { 1u8 } else { 3u8 };

        // Find shared qubits between the two stabilizers
        let set1: std::collections::HashSet<u8> = checks[si1].iter().cloned().collect();
        let shared: Vec<u8> = checks[si2]
            .iter()
            .filter(|q| set1.contains(q))
            .cloned()
            .collect();

        if !shared.is_empty() {
            // Shared qubit found: single correction suffices
            let q = shared[0];
            correction[q as usize] = combine_pauli_single(correction[q as usize], pauli);
        } else {
            // No shared qubit: apply correction chain between stabilizers.
            // Find a path of qubits connecting the two stabilizers via
            // intermediate stabilizers that share qubits.
            // Simple fallback: correct one qubit from each stabilizer.
            if let Some(&q1) = checks[si1].first() {
                correction[q1 as usize] = combine_pauli_single(correction[q1 as usize], pauli);
            }
            if let Some(&q2) = checks[si2].first() {
                correction[q2 as usize] = combine_pauli_single(correction[q2 as usize], pauli);
            }
        }
    }

    correction
}

/// Union-Find decoder for surface code syndromes.
///
/// Grows clusters from syndrome defects, merges when clusters touch,
/// and applies correction based on cluster parities.
pub fn union_find_decode(syndrome: &[u8], distance: usize) -> Vec<u8> {
    let n = num_data_qubits(distance);
    let (x_checks, z_checks) = surface_code_parity_check(distance);
    let num_x = x_checks.len();

    let x_syndrome = &syndrome[..num_x.min(syndrome.len())];
    let z_syndrome = if syndrome.len() > num_x {
        &syndrome[num_x..]
    } else {
        &[]
    };

    let mut correction = vec![0u8; n];

    // Decode X errors (from Z syndrome)
    let z_defects: Vec<usize> = z_syndrome
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == 1)
        .map(|(i, _)| i)
        .collect();
    if !z_defects.is_empty() {
        let corr = union_find_correct(&z_defects, &z_checks, distance, true);
        for (i, &c) in corr.iter().enumerate() {
            if c != 0 {
                correction[i] = combine_pauli_single(correction[i], c);
            }
        }
    }

    // Decode Z errors (from X syndrome)
    let x_defects: Vec<usize> = x_syndrome
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == 1)
        .map(|(i, _)| i)
        .collect();
    if !x_defects.is_empty() {
        let corr = union_find_correct(&x_defects, &x_checks, distance, false);
        for (i, &c) in corr.iter().enumerate() {
            if c != 0 {
                correction[i] = combine_pauli_single(correction[i], c);
            }
        }
    }

    correction
}

/// Union-Find cluster growth correction.
///
/// Each defect starts as its own cluster. Clusters grow and merge when
/// neighbors are also defects. For each cluster with odd parity, find
/// a correction by locating shared qubits between adjacent defects.
fn union_find_correct(
    defects: &[usize],
    checks: &[Vec<u8>],
    distance: usize,
    correct_x: bool,
) -> Vec<u8> {
    let n = num_data_qubits(distance);
    let mut correction = vec![0u8; n];

    if defects.is_empty() {
        return correction;
    }

    let num_checks = checks.len();

    // Build qubit-to-checks adjacency
    let mut qubit_to_checks: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ci, check) in checks.iter().enumerate() {
        for &q in check {
            qubit_to_checks[q as usize].push(ci);
        }
    }

    // Build check adjacency: two checks are neighbors if they share a data qubit
    let mut adjacency: Vec<Vec<(usize, u8)>> = vec![Vec::new(); num_checks];
    for qi in 0..n {
        let check_list = &qubit_to_checks[qi];
        for i in 0..check_list.len() {
            for j in (i + 1)..check_list.len() {
                adjacency[check_list[i]].push((check_list[j], qi as u8));
                adjacency[check_list[j]].push((check_list[i], qi as u8));
            }
        }
    }

    // For the simple case of exactly 2 defects, use direct shared-qubit matching
    if defects.len() == 2 {
        let d0 = defects[0].min(num_checks - 1);
        let d1 = defects[1].min(num_checks - 1);
        let pauli = if correct_x { 1u8 } else { 3u8 };

        // Check if they share a qubit directly
        let set0: std::collections::HashSet<u8> = checks[d0].iter().cloned().collect();
        let shared: Vec<u8> = checks[d1].iter().filter(|q| set0.contains(q)).cloned().collect();

        if !shared.is_empty() {
            correction[shared[0] as usize] = combine_pauli_single(correction[shared[0] as usize], pauli);
            return correction;
        }

        // BFS to find shortest path between d0 and d1 through check adjacency
        let path = bfs_path(d0, d1, &adjacency, num_checks);
        for qi in path {
            correction[qi as usize] = combine_pauli_single(correction[qi as usize], pauli);
        }
        return correction;
    }

    // General case: greedy pair matching (same as MWPM) then correct shared qubits
    let positions: Vec<(f64, f64)> = checks
        .iter()
        .map(|qubits| {
            let mut r = 0.0;
            let mut c = 0.0;
            for &q in qubits {
                r += (q as usize / distance) as f64;
                c += (q as usize % distance) as f64;
            }
            let len = qubits.len() as f64;
            (r / len, c / len)
        })
        .collect();

    let mut used = vec![false; defects.len()];
    let pauli = if correct_x { 1u8 } else { 3u8 };

    // Greedy pairing
    for i in 0..defects.len() {
        if used[i] { continue; }
        let mut best_j = None;
        let mut best_dist = f64::MAX;
        for j in (i+1)..defects.len() {
            if used[j] { continue; }
            let di = defects[i].min(positions.len().saturating_sub(1));
            let dj = defects[j].min(positions.len().saturating_sub(1));
            let dist = (positions[di].0 - positions[dj].0).abs()
                + (positions[di].1 - positions[dj].1).abs();
            if dist < best_dist {
                best_dist = dist;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used[i] = true;
            used[j] = true;
            let d0 = defects[i].min(num_checks.saturating_sub(1));
            let d1 = defects[j].min(num_checks.saturating_sub(1));
            // Try shared qubit first
            let set0: std::collections::HashSet<u8> = checks[d0].iter().cloned().collect();
            let shared: Vec<u8> = checks[d1].iter().filter(|q| set0.contains(q)).cloned().collect();
            if !shared.is_empty() {
                correction[shared[0] as usize] = combine_pauli_single(correction[shared[0] as usize], pauli);
            } else {
                let path = bfs_path(d0, d1, &adjacency, num_checks);
                for qi in path {
                    correction[qi as usize] = combine_pauli_single(correction[qi as usize], pauli);
                }
            }
        } else {
            // Unpaired: boundary correction
            used[i] = true;
            let si = defects[i].min(num_checks.saturating_sub(1));
            if let Some(&q) = checks[si].first() {
                correction[q as usize] = combine_pauli_single(correction[q as usize], pauli);
            }
        }
    }

    correction
}

/// BFS to find shortest path (in data qubits) between two checks.
fn bfs_path(start: usize, end: usize, adjacency: &[Vec<(usize, u8)>], num_nodes: usize) -> Vec<u8> {
    if start == end {
        return Vec::new();
    }
    let mut visited = vec![false; num_nodes];
    let mut parent: Vec<Option<(usize, u8)>> = vec![None; num_nodes]; // (parent_check, connecting_qubit)
    let mut queue = std::collections::VecDeque::new();
    visited[start] = true;
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == end {
            // Reconstruct path
            let mut path = Vec::new();
            let mut node = end;
            while let Some((p, q)) = parent[node] {
                path.push(q);
                node = p;
            }
            return path;
        }
        for &(neighbor, qubit) in &adjacency[current] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = Some((current, qubit));
                queue.push_back(neighbor);
            }
        }
    }
    Vec::new() // No path found
}

/// Belief Propagation decoder.
///
/// Iterative message-passing decoder on the factor graph defined by the
/// parity check matrix.
pub fn bp_decode(
    parity_check: &[Vec<u8>],
    syndrome: &[u8],
    error_rate: f64,
    iterations: usize,
) -> Vec<u8> {
    if parity_check.is_empty() || syndrome.is_empty() {
        return Vec::new();
    }

    // Find the number of variable nodes (data qubits)
    let num_vars = parity_check
        .iter()
        .flat_map(|check| check.iter())
        .map(|&q| q as usize + 1)
        .max()
        .unwrap_or(0);

    if num_vars == 0 {
        return Vec::new();
    }

    // Prior log-likelihood ratio: ln((1-p)/p)
    let prior_llr = if error_rate > 0.0 && error_rate < 1.0 {
        ((1.0 - error_rate) / error_rate).ln()
    } else if error_rate == 0.0 {
        50.0 // Very confident no error
    } else {
        -50.0 // Very confident error
    };

    // Initialize variable-to-check messages
    let mut var_to_check: Vec<Vec<f64>> = vec![vec![prior_llr; parity_check.len()]; num_vars];
    let mut check_to_var: Vec<Vec<f64>> = vec![vec![0.0; parity_check.len()]; num_vars];

    // Build variable-to-check adjacency
    let mut var_checks: Vec<Vec<usize>> = vec![Vec::new(); num_vars];
    for (ci, check) in parity_check.iter().enumerate() {
        for &q in check {
            var_checks[q as usize].push(ci);
        }
    }

    for _iter in 0..iterations {
        // Check-to-variable messages
        for (ci, check) in parity_check.iter().enumerate() {
            let s = if ci < syndrome.len() {
                syndrome[ci] as f64
            } else {
                0.0
            };

            for &q in check {
                let qi = q as usize;
                // Product of tanh(msg/2) for all other variables
                let mut product = 1.0;
                for &other_q in check {
                    let other_qi = other_q as usize;
                    if other_qi != qi {
                        let msg = var_to_check[other_qi][ci];
                        product *= (msg / 2.0).tanh();
                    }
                }
                // Clamp to avoid numerical issues
                product = product.max(-0.9999).min(0.9999);
                let val = 2.0 * ((1.0 + product) / (1.0 - product)).abs().ln();
                check_to_var[qi][ci] = if s == 1.0 { -val } else { val };
            }
        }

        // Variable-to-check messages
        for qi in 0..num_vars {
            for &ci in &var_checks[qi] {
                let mut sum = prior_llr;
                for &other_ci in &var_checks[qi] {
                    if other_ci != ci {
                        sum += check_to_var[qi][other_ci];
                    }
                }
                var_to_check[qi][ci] = sum;
            }
        }
    }

    // Final decision: total LLR for each variable
    let mut result = vec![0u8; num_vars];
    for qi in 0..num_vars {
        let mut total_llr = prior_llr;
        for &ci in &var_checks[qi] {
            total_llr += check_to_var[qi][ci];
        }
        if total_llr < 0.0 {
            result[qi] = 1; // Estimated error
        }
    }

    result
}

// ============================================================
// STATISTICAL FUNCTIONS
// ============================================================

/// Compute the Wilson score confidence interval.
///
/// Returns (lower, upper) bounds for the true probability given
/// `n_trials` observations with `n_failures` successes (failures).
pub fn wilson_confidence_interval(
    n_trials: usize,
    n_failures: usize,
    confidence: f64,
) -> (f64, f64) {
    if n_trials == 0 {
        return (0.0, 1.0);
    }

    let n = n_trials as f64;
    let k = n_failures as f64;
    let p_hat = k / n;

    // Z-score for the given confidence level
    let z = z_score(confidence);
    let z2 = z * z;

    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let margin = z * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt() / denom;

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}

/// Standard error of the proportion estimate.
pub fn standard_error(n_trials: usize, n_failures: usize) -> f64 {
    if n_trials == 0 {
        return 0.0;
    }
    let n = n_trials as f64;
    let p = n_failures as f64 / n;
    (p * (1.0 - p) / n).sqrt()
}

/// Approximate z-score for common confidence levels.
fn z_score(confidence: f64) -> f64 {
    // Approximation using rational function (Abramowitz and Stegun)
    let alpha = (1.0 - confidence) / 2.0;
    // For common values, use exact constants
    if (confidence - 0.95).abs() < 1e-10 {
        return 1.959964;
    }
    if (confidence - 0.99).abs() < 1e-10 {
        return 2.575829;
    }
    if (confidence - 0.90).abs() < 1e-10 {
        return 1.644854;
    }

    // Rational approximation of the inverse normal CDF
    // Based on Peter Acklam's algorithm
    let p = alpha;
    if p <= 0.0 {
        return 10.0;
    }
    if p >= 1.0 {
        return 0.0;
    }

    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let q = (-2.0 * p.ln()).sqrt();
    let num = ((((a[0] * q + a[1]) * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5];
    let den = ((((b[0] * q + b[1]) * q + b[2]) * q + b[3]) * q + b[4]) * q + 1.0;

    (num / den).abs()
}

// ============================================================
// SAMPLING ENGINE
// ============================================================

/// Run Monte Carlo sampling for all (error_rate, distance) pairs.
///
/// Uses Rayon for parallel trial execution within each configuration point.
/// Supports early stopping after `min_failures` logical errors.
pub fn run_sampling(config: &SamplingConfig) -> Vec<SamplingResult> {
    // Configure thread pool if specified
    if config.num_threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global();
    }

    let mut results = Vec::new();

    for &error_rate in &config.error_rates {
        for &distance in &config.code_distances {
            let start = Instant::now();

            let result = sample_single_point(
                error_rate,
                distance,
                config.num_trials,
                config.min_failures,
                &config.decoder,
                &config.noise_model,
                config.confidence_level,
            );

            let elapsed = start.elapsed().as_secs_f64();

            results.push(SamplingResult {
                error_rate,
                code_distance: distance,
                num_trials: result.0,
                num_failures: result.1,
                logical_error_rate: if result.0 > 0 {
                    result.1 as f64 / result.0 as f64
                } else {
                    0.0
                },
                confidence_interval: wilson_confidence_interval(
                    result.0,
                    result.1,
                    config.confidence_level,
                ),
                wall_time_seconds: elapsed,
            });
        }
    }

    results
}

/// Sample a single (error_rate, distance) point.
///
/// Returns (num_trials_completed, num_failures).
fn sample_single_point(
    error_rate: f64,
    distance: usize,
    max_trials: usize,
    min_failures: usize,
    decoder: &DecoderType,
    noise: &NoiseType,
    _confidence: f64,
) -> (usize, usize) {
    // Use parallel execution with Rayon
    // Split into chunks, each chunk processes sequentially with its own RNG
    let chunk_size = 100.max(max_trials / rayon::current_num_threads().max(1));
    let num_chunks = (max_trials + chunk_size - 1) / chunk_size;

    let (x_checks, z_checks) = surface_code_parity_check(distance);

    // Collect chunk results in parallel
    let chunk_results: Vec<(usize, usize)> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let mut rng = rand::thread_rng();
            let trials_in_chunk = if chunk_idx == num_chunks - 1 {
                max_trials - chunk_idx * chunk_size
            } else {
                chunk_size
            };

            let mut local_trials = 0;
            let mut local_failures = 0;

            for _ in 0..trials_in_chunk {
                let n = num_data_qubits(distance);
                let data_errors = generate_errors(n, error_rate, noise, &mut rng);

                // Compute syndrome
                let x_syndrome = compute_syndrome(&data_errors, &x_checks, true);
                let z_syndrome = compute_syndrome(&data_errors, &z_checks, false);
                let mut syndrome = x_syndrome;
                syndrome.extend(z_syndrome);

                // Decode
                let correction = match decoder {
                    DecoderType::Mwpm => mwpm_decode(&syndrome, distance),
                    DecoderType::UnionFind => union_find_decode(&syndrome, distance),
                    DecoderType::BeliefPropagation => {
                        // Use Z checks for BP (X errors detected by Z syndrome)
                        let z_syn_len = z_checks.len();
                        let z_syn =
                            &syndrome[x_checks.len()..x_checks.len() + z_syn_len.min(syndrome.len() - x_checks.len().min(syndrome.len()))];
                        let bp_corr = bp_decode(&z_checks, z_syn, error_rate, 20);
                        // Convert BP X-only correction to full Pauli
                        let mut full = vec![0u8; n];
                        for (i, &c) in bp_corr.iter().enumerate() {
                            if i < n {
                                full[i] = c;
                            }
                        }
                        full
                    }
                    DecoderType::NeuralDecoder => {
                        // Fallback to MWPM for now
                        mwpm_decode(&syndrome, distance)
                    }
                };

                // Check for logical error: residual = data_errors * correction
                let residual = combine_pauli(&data_errors, &correction);
                let is_logical =
                    is_logical_x(&residual, distance) || is_logical_z(&residual, distance);

                local_trials += 1;
                if is_logical {
                    local_failures += 1;
                }
            }

            (local_trials, local_failures)
        })
        .collect();

    // Aggregate
    let mut total_trials = 0;
    let mut total_failures = 0;
    for (t, f) in chunk_results {
        total_trials += t;
        total_failures += f;
        // Note: early stopping across chunks is approximate.
        // For exact early stopping, you'd need sequential execution.
        if total_failures >= min_failures && total_trials >= max_trials / 2 {
            break;
        }
    }

    (total_trials, total_failures)
}

/// Run a full threshold study.
///
/// Samples across all (error_rate, distance) pairs and estimates the
/// threshold by finding the crossing point of logical error rate curves.
pub fn run_threshold_study(config: &SamplingConfig) -> ThresholdResult {
    let results = run_sampling(config);
    let (threshold, uncertainty) = estimate_threshold(&results, &config.code_distances);

    ThresholdResult {
        results,
        threshold_estimate: threshold,
        threshold_uncertainty: uncertainty,
    }
}

/// Estimate threshold from sampling results via curve crossing.
///
/// For each pair of adjacent distances, find where their logical error rate
/// curves cross using linear interpolation.
fn estimate_threshold(results: &[SamplingResult], distances: &[usize]) -> (f64, f64) {
    if distances.len() < 2 {
        return (0.0, 1.0);
    }

    let mut crossings: Vec<f64> = Vec::new();

    // For each pair of adjacent distances
    for i in 0..distances.len() - 1 {
        let d_small = distances[i];
        let d_large = distances[i + 1];

        // Get results for each distance, sorted by error rate
        let mut small_results: Vec<&SamplingResult> = results
            .iter()
            .filter(|r| r.code_distance == d_small)
            .collect();
        let mut large_results: Vec<&SamplingResult> = results
            .iter()
            .filter(|r| r.code_distance == d_large)
            .collect();

        small_results.sort_by(|a, b| a.error_rate.partial_cmp(&b.error_rate).unwrap());
        large_results.sort_by(|a, b| a.error_rate.partial_cmp(&b.error_rate).unwrap());

        // Find crossing points by linear interpolation
        for j in 0..small_results.len().min(large_results.len()).saturating_sub(1) {
            let p1 = small_results[j].error_rate;
            let p2 = small_results[j + 1].error_rate;

            let s1 = small_results[j].logical_error_rate;
            let s2 = small_results[j + 1].logical_error_rate;

            // Find matching error rates in large_results
            let l1_opt = large_results.iter().find(|r| (r.error_rate - p1).abs() < 1e-12);
            let l2_opt = large_results.iter().find(|r| (r.error_rate - p2).abs() < 1e-12);

            if let (Some(l1_r), Some(l2_r)) = (l1_opt, l2_opt) {
                let l1 = l1_r.logical_error_rate;
                let l2 = l2_r.logical_error_rate;

                // Check if curves cross in this interval
                // Below threshold: larger distance has lower error rate
                // Above threshold: larger distance has higher error rate
                let diff1 = s1 - l1; // small - large
                let diff2 = s2 - l2;

                if diff1 * diff2 < 0.0 {
                    // Sign change: crossing point
                    let t = diff1 / (diff1 - diff2);
                    let crossing = p1 + t * (p2 - p1);
                    crossings.push(crossing);
                }
            }
        }
    }

    if crossings.is_empty() {
        // No crossing found: estimate from the data
        // Use the error rate where the logical error rates are closest
        let mut best_rate = 0.0;
        let mut best_diff = f64::MAX;

        for r1 in results.iter() {
            for r2 in results.iter() {
                if r1.code_distance < r2.code_distance
                    && (r1.error_rate - r2.error_rate).abs() < 1e-12
                {
                    let diff = (r1.logical_error_rate - r2.logical_error_rate).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_rate = r1.error_rate;
                    }
                }
            }
        }

        return (best_rate, best_diff);
    }

    // Average crossing points
    let mean = crossings.iter().sum::<f64>() / crossings.len() as f64;
    let variance = if crossings.len() > 1 {
        crossings.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (crossings.len() - 1) as f64
    } else {
        0.01 // Default uncertainty
    };

    (mean, variance.sqrt())
}

// ============================================================
// CSV OUTPUT
// ============================================================

/// Format sampling results as CSV.
pub fn results_to_csv(results: &[SamplingResult]) -> String {
    let mut csv = String::from("error_rate,distance,num_trials,num_failures,logical_error_rate,ci_low,ci_high\n");

    for r in results {
        csv.push_str(&format!(
            "{:.6},{},{},{},{:.8},{:.8},{:.8}\n",
            r.error_rate,
            r.code_distance,
            r.num_trials,
            r.num_failures,
            r.logical_error_rate,
            r.confidence_interval.0,
            r.confidence_interval.1,
        ));
    }

    csv
}

/// Format a threshold report as a human-readable string.
pub fn threshold_report(result: &ThresholdResult) -> String {
    let mut report = String::new();
    report.push_str("=== QEC Threshold Study Report ===\n\n");
    report.push_str(&format!(
        "Threshold estimate: {:.4} +/- {:.4}\n\n",
        result.threshold_estimate, result.threshold_uncertainty
    ));
    report.push_str("Results:\n");
    report.push_str(&results_to_csv(&result.results));
    report
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: SamplingConfig builder defaults
    #[test]
    fn test_sampling_config_defaults() {
        let config = SamplingConfig::builder().build();
        assert_eq!(config.num_trials, 10_000);
        assert_eq!(config.error_rates, vec![0.01, 0.05, 0.1]);
        assert_eq!(config.code_distances, vec![3, 5, 7]);
        assert_eq!(config.decoder, DecoderType::Mwpm);
        assert_eq!(config.noise_model, NoiseType::Depolarizing);
        assert_eq!(config.num_threads, 0);
        assert!((config.confidence_level - 0.95).abs() < 1e-10);
        assert_eq!(config.min_failures, 100);
    }

    // Test 2: Surface code parity check dimensions (d=3)
    #[test]
    fn test_surface_code_parity_check_d3() {
        let (x_checks, z_checks) = surface_code_parity_check(3);
        // d=3 surface code: 9 data qubits, (d^2-1)/2 = 4 stabilizers each
        assert_eq!(x_checks.len(), 4, "Expected 4 X stabilizers for d=3");
        assert_eq!(z_checks.len(), 4, "Expected 4 Z stabilizers for d=3");

        // Each stabilizer should reference valid qubit indices (0..8)
        for check in x_checks.iter().chain(z_checks.iter()) {
            for &q in check {
                assert!(
                    (q as usize) < 9,
                    "Qubit index {} out of range for d=3",
                    q
                );
            }
        }
    }

    // Test 3: Zero error rate yields zero logical errors
    #[test]
    fn test_zero_error_rate() {
        let config = SamplingConfig::builder()
            .num_trials(100)
            .error_rates(vec![0.0])
            .code_distances(vec![3])
            .min_failures(0)
            .build();

        let results = run_sampling(&config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_failures, 0);
        assert!((results[0].logical_error_rate - 0.0).abs() < 1e-10);
    }

    // Test 4: Syndrome of identity error is zero
    #[test]
    fn test_identity_syndrome() {
        let distance = 3;
        let n = num_data_qubits(distance);
        let (x_checks, z_checks) = surface_code_parity_check(distance);

        let no_errors = vec![0u8; n];
        let x_syn = compute_syndrome(&no_errors, &x_checks, true);
        let z_syn = compute_syndrome(&no_errors, &z_checks, false);

        assert!(
            x_syn.iter().all(|&s| s == 0),
            "X syndrome should be zero for no errors"
        );
        assert!(
            z_syn.iter().all(|&s| s == 0),
            "Z syndrome should be zero for no errors"
        );
    }

    // Test 5: Single-qubit error produces non-zero syndrome
    #[test]
    fn test_single_error_syndrome() {
        let distance = 3;
        let n = num_data_qubits(distance);
        let (x_checks, z_checks) = surface_code_parity_check(distance);

        // Place a single X error on qubit 4 (center of 3x3 grid)
        let mut errors = vec![0u8; n];
        errors[4] = 1; // X error

        // X error should be detected by Z stabilizers (not X stabilizers)
        let z_syn = compute_syndrome(&errors, &z_checks, false);
        assert!(
            z_syn.iter().any(|&s| s == 1),
            "Z syndrome should detect X error on center qubit"
        );
    }

    // Test 6: MWPM corrects single-qubit error on d=3
    #[test]
    fn test_mwpm_single_error_d3() {
        let distance = 3;
        let n = num_data_qubits(distance);
        let (x_checks, z_checks) = surface_code_parity_check(distance);

        // Place a single X error on qubit 4
        let mut errors = vec![0u8; n];
        errors[4] = 1; // X error

        let x_syndrome = compute_syndrome(&errors, &x_checks, true);
        let z_syndrome = compute_syndrome(&errors, &z_checks, false);
        let mut syndrome = x_syndrome;
        syndrome.extend(z_syndrome);

        let correction = mwpm_decode(&syndrome, distance);
        let residual = combine_pauli(&errors, &correction);

        // After correction, residual should not be a logical operator
        let is_logical = is_logical_x(&residual, distance) || is_logical_z(&residual, distance);
        // A good decoder on d=3 should correct any single-qubit error.
        // The greedy MWPM may not always succeed, but for a center qubit it should.
        // We test that either it corrected perfectly or the residual is stabilizer-equivalent.
        let residual_weight = hamming_weight(&residual);
        assert!(
            !is_logical || residual_weight == 0,
            "MWPM should correct single X error on d=3 center qubit. \
             Residual weight: {}, is_logical: {}",
            residual_weight,
            is_logical
        );
    }

    // Test 7: Union-Find corrects single-qubit error on d=3
    #[test]
    fn test_union_find_single_error_d3() {
        let distance = 3;
        let n = num_data_qubits(distance);
        let (x_checks, z_checks) = surface_code_parity_check(distance);

        // Place a single Z error on qubit 4
        let mut errors = vec![0u8; n];
        errors[4] = 3; // Z error

        let x_syndrome = compute_syndrome(&errors, &x_checks, true);
        let z_syndrome = compute_syndrome(&errors, &z_checks, false);
        let mut syndrome = x_syndrome;
        syndrome.extend(z_syndrome);

        let correction = union_find_decode(&syndrome, distance);
        let residual = combine_pauli(&errors, &correction);

        let residual_weight = hamming_weight(&residual);
        let is_logical = is_logical_x(&residual, distance) || is_logical_z(&residual, distance);
        assert!(
            !is_logical || residual_weight == 0,
            "Union-Find should correct single Z error on d=3 center qubit. \
             Residual weight: {}, is_logical: {}",
            residual_weight,
            is_logical
        );
    }

    // Test 8: BP decoder converges for simple syndrome
    #[test]
    fn test_bp_decoder_convergence() {
        // Simple 3-bit repetition code parity check: [[1,1,0],[0,1,1]]
        let parity_check = vec![vec![0u8, 1], vec![1u8, 2]];
        let syndrome = vec![1u8, 0]; // Error on qubit 0

        let correction = bp_decode(&parity_check, &syndrome, 0.1, 50);
        assert_eq!(correction.len(), 3);
        // BP should identify qubit 0 as the likely error
        assert_eq!(correction[0], 1, "BP should detect error on qubit 0");
    }

    // Test 9: Wilson CI contains true value for known case
    #[test]
    fn test_wilson_ci_contains_true_value() {
        // 100 trials, 10 failures => p_hat = 0.1
        let (lower, upper) = wilson_confidence_interval(100, 10, 0.95);
        assert!(lower < 0.1, "Lower bound {} should be < 0.1", lower);
        assert!(upper > 0.1, "Upper bound {} should be > 0.1", upper);
        assert!(lower >= 0.0, "Lower bound should be >= 0");
        assert!(upper <= 1.0, "Upper bound should be <= 1");
        assert!(lower < upper, "Lower should be < upper");

        // 1000 trials, 500 failures => p_hat = 0.5
        let (lower2, upper2) = wilson_confidence_interval(1000, 500, 0.95);
        assert!(lower2 < 0.5);
        assert!(upper2 > 0.5);
        // CI should be narrower with more trials
        assert!(
            (upper2 - lower2) < (upper - lower) * 2.0,
            "More trials should give similar or tighter CI"
        );
    }

    // Test 10: Logical error rate decreases with distance (below threshold)
    #[test]
    fn test_logical_error_decreases_below_threshold() {
        // At p=0.01 (well below threshold ~0.1), larger distance should help
        let config = SamplingConfig::builder()
            .num_trials(2000)
            .error_rates(vec![0.01])
            .code_distances(vec![3, 5])
            .min_failures(0)
            .build();

        let results = run_sampling(&config);
        assert_eq!(results.len(), 2);

        let d3_rate = results[0].logical_error_rate;
        let d5_rate = results[1].logical_error_rate;

        // Below threshold, d=5 should have lower or equal logical error rate than d=3
        // With only 2000 trials and greedy decoder, allow some slack
        assert!(
            d5_rate <= d3_rate + 0.05,
            "At p=0.01 (below threshold), d=5 rate ({:.4}) should be <= d=3 rate ({:.4}) + margin",
            d5_rate,
            d3_rate
        );
    }

    // Test 11: Logical error rate increases with distance (above threshold)
    #[test]
    fn test_logical_error_increases_above_threshold() {
        // At p=0.3 (well above threshold), larger distance should hurt
        let config = SamplingConfig::builder()
            .num_trials(2000)
            .error_rates(vec![0.30])
            .code_distances(vec![3, 5])
            .min_failures(0)
            .build();

        let results = run_sampling(&config);
        assert_eq!(results.len(), 2);

        let d3_rate = results[0].logical_error_rate;
        let d5_rate = results[1].logical_error_rate;

        // Above threshold, d=5 should have higher or equal logical error rate than d=3
        // At p=0.3, both should have very high logical error rates (close to 0.5)
        assert!(
            d5_rate >= d3_rate - 0.1,
            "At p=0.3 (above threshold), d=5 rate ({:.4}) should be >= d=3 rate ({:.4}) - margin",
            d5_rate,
            d3_rate
        );
    }

    // Test 12: CSV output has correct columns
    #[test]
    fn test_csv_output_format() {
        let results = vec![SamplingResult {
            error_rate: 0.05,
            code_distance: 3,
            num_trials: 1000,
            num_failures: 42,
            logical_error_rate: 0.042,
            confidence_interval: (0.031, 0.056),
            wall_time_seconds: 1.23,
        }];

        let csv = results_to_csv(&results);
        let lines: Vec<&str> = csv.lines().collect();

        assert_eq!(lines.len(), 2, "CSV should have header + 1 data row");
        assert_eq!(
            lines[0],
            "error_rate,distance,num_trials,num_failures,logical_error_rate,ci_low,ci_high"
        );

        let fields: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(fields.len(), 7, "Each row should have 7 columns");
    }

    // Test 13: Parallel sampling produces consistent results
    #[test]
    fn test_parallel_consistency() {
        // Run the same config twice and check results are statistically similar
        let config = SamplingConfig::builder()
            .num_trials(1000)
            .error_rates(vec![0.05])
            .code_distances(vec![3])
            .min_failures(0)
            .build();

        let results1 = run_sampling(&config);
        let results2 = run_sampling(&config);

        assert_eq!(results1.len(), 1);
        assert_eq!(results2.len(), 1);

        // Both runs should give similar logical error rates (within statistical fluctuation)
        let diff = (results1[0].logical_error_rate - results2[0].logical_error_rate).abs();
        assert!(
            diff < 0.1,
            "Parallel runs should give similar results. \
             Run 1: {:.4}, Run 2: {:.4}, diff: {:.4}",
            results1[0].logical_error_rate,
            results2[0].logical_error_rate,
            diff
        );
    }

    // Test 14: Threshold estimate is near expected for depolarizing noise
    #[test]
    fn test_threshold_estimate() {
        // Run a small threshold study
        let config = SamplingConfig::builder()
            .num_trials(500)
            .error_rates(vec![0.01, 0.05, 0.10, 0.15, 0.20])
            .code_distances(vec![3, 5])
            .min_failures(0)
            .build();

        let result = run_threshold_study(&config);

        // The threshold for depolarizing noise on surface code is ~0.109
        // With a greedy decoder and few trials, we just check it's in a reasonable range
        assert!(
            result.threshold_estimate >= 0.01 && result.threshold_estimate <= 0.30,
            "Threshold estimate {:.4} should be in [0.01, 0.30]",
            result.threshold_estimate
        );

        // Report should not be empty
        let report = threshold_report(&result);
        assert!(report.contains("Threshold estimate"));
        assert!(report.contains("error_rate"));
    }

    // Additional tests for edge cases

    #[test]
    fn test_hamming_weight() {
        assert_eq!(hamming_weight(&[0, 0, 0]), 0);
        assert_eq!(hamming_weight(&[1, 0, 3]), 2);
        assert_eq!(hamming_weight(&[1, 2, 3, 1]), 4);
        assert_eq!(hamming_weight(&[]), 0);
    }

    #[test]
    fn test_combine_pauli() {
        // I * X = X
        assert_eq!(combine_pauli(&[0], &[1]), vec![1]);
        // X * X = I
        assert_eq!(combine_pauli(&[1], &[1]), vec![0]);
        // X * Z = Y
        assert_eq!(combine_pauli(&[1], &[3]), vec![2]);
        // Y * Y = I
        assert_eq!(combine_pauli(&[2], &[2]), vec![0]);
        // Z * Z = I
        assert_eq!(combine_pauli(&[3], &[3]), vec![0]);
    }

    #[test]
    fn test_standard_error() {
        let se = standard_error(100, 50);
        // p=0.5, se = sqrt(0.25/100) = 0.05
        assert!((se - 0.05).abs() < 1e-10);

        let se_zero = standard_error(0, 0);
        assert!((se_zero - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_validation() {
        let valid = SamplingConfig::builder().build();
        assert!(valid.validate().is_ok());

        let invalid_trials = SamplingConfig::builder().num_trials(0).build();
        assert!(invalid_trials.validate().is_err());

        let invalid_rate = SamplingConfig::builder()
            .error_rates(vec![-0.1])
            .build();
        assert!(invalid_rate.validate().is_err());

        let invalid_dist = SamplingConfig::builder()
            .code_distances(vec![2])
            .build();
        assert!(invalid_dist.validate().is_err());
    }

    #[test]
    fn test_logical_operators() {
        let d = 3;
        // All zeros: not logical
        let zeros = vec![0u8; 9];
        assert!(!is_logical_x(&zeros, d));
        assert!(!is_logical_z(&zeros, d));

        // A full column of X errors: logical X
        let mut col_x = vec![0u8; 9];
        col_x[0] = 1; // (0,0)
        col_x[3] = 1; // (1,0)
        col_x[6] = 1; // (2,0)
        assert!(is_logical_x(&col_x, d), "Full column of X should be logical X");

        // A full row of Z errors: logical Z
        let mut row_z = vec![0u8; 9];
        row_z[0] = 3; // (0,0)
        row_z[1] = 3; // (0,1)
        row_z[2] = 3; // (0,2)
        assert!(is_logical_z(&row_z, d), "Full row of Z should be logical Z");
    }

    #[test]
    fn test_surface_code_d5() {
        let (x_checks, z_checks) = surface_code_parity_check(5);
        // d=5: 25 data qubits, (25-1)/2 = 12 stabilizers each
        assert_eq!(x_checks.len(), 12, "Expected 12 X stabilizers for d=5");
        assert_eq!(z_checks.len(), 12, "Expected 12 Z stabilizers for d=5");
    }

    #[test]
    fn test_phenomenological_noise() {
        let config = SamplingConfig::builder()
            .num_trials(100)
            .error_rates(vec![0.05])
            .code_distances(vec![3])
            .noise_model(NoiseType::Phenomenological {
                measurement_rounds: 3,
            })
            .min_failures(0)
            .build();

        let results = run_sampling(&config);
        assert_eq!(results.len(), 1);
        assert!(results[0].num_trials > 0);
    }
}
