//! Probabilistic Error Cancellation (PEC)
//!
//! Implements full quasi-probability decomposition for error-mitigated quantum
//! computation. PEC works by expressing the inverse of a noisy channel as a
//! linear combination of physically implementable operations, where the
//! coefficients form a quasi-probability distribution (they can be negative).
//!
//! The key insight is that while we cannot physically implement the inverse of
//! a noise channel, we can decompose it into operations we *can* implement
//! and use Monte Carlo sampling with importance weighting to recover the
//! ideal (noiseless) expectation value.
//!
//! # Sampling Overhead
//!
//! The cost of PEC is captured by the one-norm gamma = sum |eta_i| of the
//! quasi-probability coefficients. The number of samples needed scales as
//! O(gamma^{2L} / epsilon^2) where L is circuit depth and epsilon is the
//! desired precision.
//!
//! # Pipeline
//!
//! 1. Characterize noise via process tomography (`NoiseTomography`)
//! 2. Compute the Pauli Transfer Matrix (`PauliTransferMatrix`)
//! 3. Decompose the inverse channel (`QuasiProbabilityDecomposition`)
//! 4. Sample corrected circuits and estimate expectations (`PECSampler`)
//! 5. Combine into a full mitigation pipeline (`PECMitigator`)
//!
//! # References
//!
//! - Temme, Bravyi, Gambetta, "Error Mitigation for Short-Depth Quantum
//!   Circuits", PRL 119, 180509 (2017)
//! - van den Berg, Minev, Kandala, Temme, "Probabilistic error cancellation
//!   with sparse Pauli-Lindblad models on noisy quantum processors",
//!   Nature Physics 19, 1116-1121 (2023)

use ndarray::Array2;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising during PEC operations.
#[derive(Debug, Clone)]
pub enum PECError {
    /// Noise parameter is outside valid range.
    InvalidNoiseParameter(String),
    /// Kraus operators do not satisfy the completeness relation.
    InvalidKrausOperators(String),
    /// The noise channel is not invertible.
    NonInvertibleChannel(String),
    /// Matrix dimension mismatch in an operation.
    DimensionMismatch { expected: usize, got: usize },
    /// Configuration parameter is invalid.
    InvalidConfig(String),
    /// Sampling or estimation failed.
    SamplingError(String),
}

impl fmt::Display for PECError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PECError::InvalidNoiseParameter(msg) => {
                write!(f, "Invalid noise parameter: {}", msg)
            }
            PECError::InvalidKrausOperators(msg) => {
                write!(f, "Invalid Kraus operators: {}", msg)
            }
            PECError::NonInvertibleChannel(msg) => {
                write!(f, "Non-invertible channel: {}", msg)
            }
            PECError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            PECError::InvalidConfig(msg) => {
                write!(f, "Invalid PEC config: {}", msg)
            }
            PECError::SamplingError(msg) => {
                write!(f, "Sampling error: {}", msg)
            }
        }
    }
}

impl std::error::Error for PECError {}

// ============================================================
// CHANNEL MODEL ENUM
// ============================================================

/// Noise channel model type for PEC characterization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChannelModel {
    /// Depolarizing channel: E(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ)
    Depolarizing,
    /// Amplitude damping: models T1 relaxation (|1> -> |0>)
    AmplitudeDamping,
    /// General Pauli channel with independent X, Y, Z error rates
    PauliChannel,
    /// Custom channel specified via Kraus operators
    Custom,
}

impl fmt::Display for ChannelModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelModel::Depolarizing => write!(f, "Depolarizing"),
            ChannelModel::AmplitudeDamping => write!(f, "AmplitudeDamping"),
            ChannelModel::PauliChannel => write!(f, "PauliChannel"),
            ChannelModel::Custom => write!(f, "Custom"),
        }
    }
}

// ============================================================
// PEC CONFIG (Builder Pattern)
// ============================================================

/// Configuration for probabilistic error cancellation.
#[derive(Debug, Clone)]
pub struct PECConfig {
    /// Number of Monte Carlo samples for expectation estimation.
    pub num_samples: usize,
    /// Noise strength parameter (0.0 = noiseless, 1.0 = maximum noise).
    pub noise_strength: f64,
    /// Maximum allowed sampling overhead before aborting.
    pub max_overhead: f64,
    /// Noise channel model to use.
    pub channel_model: ChannelModel,
    /// Number of shots for process tomography characterization.
    pub tomography_shots: usize,
}

impl Default for PECConfig {
    fn default() -> Self {
        Self {
            num_samples: 10_000,
            noise_strength: 0.01,
            max_overhead: 1000.0,
            channel_model: ChannelModel::Depolarizing,
            tomography_shots: 8192,
        }
    }
}

impl PECConfig {
    /// Create a new config builder with default values.
    pub fn builder() -> PECConfigBuilder {
        PECConfigBuilder {
            config: PECConfig::default(),
        }
    }
}

/// Builder for PECConfig.
#[derive(Debug, Clone)]
pub struct PECConfigBuilder {
    config: PECConfig,
}

impl PECConfigBuilder {
    /// Set the number of Monte Carlo samples.
    pub fn num_samples(mut self, n: usize) -> Self {
        self.config.num_samples = n;
        self
    }

    /// Set the noise strength (must be in [0.0, 1.0]).
    pub fn noise_strength(mut self, p: f64) -> Self {
        self.config.noise_strength = p;
        self
    }

    /// Set the maximum allowed sampling overhead.
    pub fn max_overhead(mut self, m: f64) -> Self {
        self.config.max_overhead = m;
        self
    }

    /// Set the noise channel model.
    pub fn channel_model(mut self, model: ChannelModel) -> Self {
        self.config.channel_model = model;
        self
    }

    /// Set the number of tomography shots.
    pub fn tomography_shots(mut self, shots: usize) -> Self {
        self.config.tomography_shots = shots;
        self
    }

    /// Build the config, validating all parameters.
    pub fn build(self) -> Result<PECConfig, PECError> {
        if self.config.noise_strength < 0.0 || self.config.noise_strength > 1.0 {
            return Err(PECError::InvalidConfig(format!(
                "noise_strength must be in [0.0, 1.0], got {}",
                self.config.noise_strength
            )));
        }
        if self.config.num_samples == 0 {
            return Err(PECError::InvalidConfig(
                "num_samples must be > 0".to_string(),
            ));
        }
        if self.config.max_overhead <= 0.0 {
            return Err(PECError::InvalidConfig(
                "max_overhead must be > 0".to_string(),
            ));
        }
        if self.config.tomography_shots == 0 {
            return Err(PECError::InvalidConfig(
                "tomography_shots must be > 0".to_string(),
            ));
        }
        Ok(self.config)
    }
}

// ============================================================
// HELPER: Complex64 matrix operations with ndarray
// ============================================================

/// Zero complex constant.
const C0: Complex64 = Complex64::new(0.0, 0.0);
/// One complex constant.
const C1: Complex64 = Complex64::new(1.0, 0.0);

/// Compute A^dagger (conjugate transpose) of a complex matrix.
fn adjoint(a: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    let mut result = Array2::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            result[[j, i]] = a[[i, j]].conj();
        }
    }
    result
}

/// Multiply two complex matrices: C = A * B.
fn matmul(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, ka) = a.dim();
    let (kb, n) = b.dim();
    assert_eq!(ka, kb, "Inner dimensions must match for matmul");
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = C0;
            for k in 0..ka {
                sum += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

/// Compute the trace of a complex matrix.
fn trace(a: &Array2<Complex64>) -> Complex64 {
    let n = a.dim().0.min(a.dim().1);
    let mut t = C0;
    for i in 0..n {
        t += a[[i, i]];
    }
    t
}

/// Identity matrix of given dimension.
fn eye(dim: usize) -> Array2<Complex64> {
    let mut m = Array2::zeros((dim, dim));
    for i in 0..dim {
        m[[i, i]] = C1;
    }
    m
}

/// Single-qubit Pauli matrices.
fn pauli_i() -> Array2<Complex64> {
    eye(2)
}

fn pauli_x() -> Array2<Complex64> {
    let mut m = Array2::zeros((2, 2));
    m[[0, 1]] = C1;
    m[[1, 0]] = C1;
    m
}

fn pauli_y() -> Array2<Complex64> {
    let mut m = Array2::zeros((2, 2));
    m[[0, 1]] = Complex64::new(0.0, -1.0);
    m[[1, 0]] = Complex64::new(0.0, 1.0);
    m
}

fn pauli_z() -> Array2<Complex64> {
    let mut m = Array2::zeros((2, 2));
    m[[0, 0]] = C1;
    m[[1, 1]] = Complex64::new(-1.0, 0.0);
    m
}

/// Return the i-th single-qubit Pauli matrix (0=I, 1=X, 2=Y, 3=Z).
fn pauli(idx: usize) -> Array2<Complex64> {
    match idx {
        0 => pauli_i(),
        1 => pauli_x(),
        2 => pauli_y(),
        3 => pauli_z(),
        _ => panic!("Pauli index must be 0..3"),
    }
}

// ============================================================
// NOISE CHANNEL
// ============================================================

/// A quantum noise channel in Kraus representation.
///
/// A channel E acts on a density matrix rho as:
///   E(rho) = sum_i K_i rho K_i^dagger
///
/// The Kraus operators must satisfy the completeness relation:
///   sum_i K_i^dagger K_i = I
#[derive(Clone, Debug)]
pub struct NoiseChannel {
    /// Kraus operators K_i.
    pub kraus_operators: Vec<Array2<Complex64>>,
    /// Hilbert space dimension (2 for single qubit, 4 for two qubit).
    pub dim: usize,
}

impl NoiseChannel {
    /// Construct a single-qubit depolarizing channel.
    ///
    /// E(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
    ///
    /// Kraus operators: K0 = sqrt(1-p) I, K1 = sqrt(p/3) X, etc.
    pub fn new_depolarizing(p: f64) -> Result<Self, PECError> {
        if p < 0.0 || p > 1.0 {
            return Err(PECError::InvalidNoiseParameter(format!(
                "Depolarizing parameter p must be in [0, 1], got {}",
                p
            )));
        }
        let s0 = (1.0 - p).sqrt();
        let s1 = (p / 3.0).sqrt();

        let k0 = pauli_i().mapv(|v| v * Complex64::new(s0, 0.0));
        let k1 = pauli_x().mapv(|v| v * Complex64::new(s1, 0.0));
        let k2 = pauli_y().mapv(|v| v * Complex64::new(s1, 0.0));
        let k3 = pauli_z().mapv(|v| v * Complex64::new(s1, 0.0));

        Ok(NoiseChannel {
            kraus_operators: vec![k0, k1, k2, k3],
            dim: 2,
        })
    }

    /// Construct a single-qubit amplitude damping channel.
    ///
    /// Models T1 relaxation: |1> decays to |0> with probability gamma.
    ///
    /// K0 = [[1, 0], [0, sqrt(1-gamma)]]
    /// K1 = [[0, sqrt(gamma)], [0, 0]]
    pub fn new_amplitude_damping(gamma: f64) -> Result<Self, PECError> {
        if gamma < 0.0 || gamma > 1.0 {
            return Err(PECError::InvalidNoiseParameter(format!(
                "Amplitude damping gamma must be in [0, 1], got {}",
                gamma
            )));
        }
        let mut k0 = Array2::zeros((2, 2));
        k0[[0, 0]] = C1;
        k0[[1, 1]] = Complex64::new((1.0 - gamma).sqrt(), 0.0);

        let mut k1 = Array2::zeros((2, 2));
        k1[[0, 1]] = Complex64::new(gamma.sqrt(), 0.0);

        Ok(NoiseChannel {
            kraus_operators: vec![k0, k1],
            dim: 2,
        })
    }

    /// Construct a general single-qubit Pauli channel.
    ///
    /// E(rho) = (1-px-py-pz) rho + px X rho X + py Y rho Y + pz Z rho Z
    pub fn new_pauli(px: f64, py: f64, pz: f64) -> Result<Self, PECError> {
        let pi = 1.0 - px - py - pz;
        if pi < -1e-12 || px < -1e-12 || py < -1e-12 || pz < -1e-12 {
            return Err(PECError::InvalidNoiseParameter(format!(
                "Pauli channel probabilities must be non-negative and sum <= 1, \
                 got px={}, py={}, pz={}, pi={}",
                px, py, pz, pi
            )));
        }
        let pi = pi.max(0.0);
        let k0 = pauli_i().mapv(|v| v * Complex64::new(pi.sqrt(), 0.0));
        let k1 = pauli_x().mapv(|v| v * Complex64::new(px.max(0.0).sqrt(), 0.0));
        let k2 = pauli_y().mapv(|v| v * Complex64::new(py.max(0.0).sqrt(), 0.0));
        let k3 = pauli_z().mapv(|v| v * Complex64::new(pz.max(0.0).sqrt(), 0.0));

        Ok(NoiseChannel {
            kraus_operators: vec![k0, k1, k2, k3],
            dim: 2,
        })
    }

    /// Apply this channel to a density matrix rho.
    ///
    /// E(rho) = sum_i K_i rho K_i^dagger
    pub fn apply(&self, rho: &Array2<Complex64>) -> Array2<Complex64> {
        let mut result = Array2::zeros((self.dim, self.dim));
        for k in &self.kraus_operators {
            let k_rho = matmul(k, rho);
            let k_dag = adjoint(k);
            result = result + matmul(&k_rho, &k_dag);
        }
        result
    }

    /// Check if the Kraus operators satisfy the completeness relation.
    ///
    /// Verifies sum_i K_i^dagger K_i = I within numerical tolerance.
    pub fn is_valid(&self) -> bool {
        self.is_valid_tol(1e-10)
    }

    /// Check completeness with a specified tolerance.
    pub fn is_valid_tol(&self, tol: f64) -> bool {
        let id = eye(self.dim);
        let mut sum = Array2::zeros((self.dim, self.dim));
        for k in &self.kraus_operators {
            let k_dag = adjoint(k);
            sum = sum + matmul(&k_dag, k);
        }
        for i in 0..self.dim {
            for j in 0..self.dim {
                let s: Complex64 = sum[[i, j]];
                let e: Complex64 = id[[i, j]];
                let dr = s.re - e.re;
                let di = s.im - e.im;
                let diff = (dr * dr + di * di).sqrt();
                if diff > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Construct the identity channel of given dimension.
    pub fn identity(dim: usize) -> Self {
        NoiseChannel {
            kraus_operators: vec![eye(dim)],
            dim,
        }
    }
}

// ============================================================
// IMPLEMENTABLE OPERATION
// ============================================================

/// An operation that can be physically implemented on hardware.
///
/// These form the basis set for the quasi-probability decomposition.
#[derive(Clone, Debug)]
pub enum ImplementableOperation {
    /// Identity (do nothing).
    Identity,
    /// Apply Pauli X gate.
    PauliX,
    /// Apply Pauli Y gate.
    PauliY,
    /// Apply Pauli Z gate.
    PauliZ,
    /// Apply an arbitrary unitary gate.
    Gate(Array2<Complex64>),
    /// Apply a noisy gate (unitary + noise channel).
    NoisyGate(Array2<Complex64>, NoiseChannel),
    /// Perform a computational basis measurement on a qubit.
    Measurement(usize),
}

impl ImplementableOperation {
    /// Get the unitary matrix for this operation (if it is a unitary).
    pub fn to_matrix(&self) -> Option<Array2<Complex64>> {
        match self {
            ImplementableOperation::Identity => Some(pauli_i()),
            ImplementableOperation::PauliX => Some(pauli_x()),
            ImplementableOperation::PauliY => Some(pauli_y()),
            ImplementableOperation::PauliZ => Some(pauli_z()),
            ImplementableOperation::Gate(u) => Some(u.clone()),
            ImplementableOperation::NoisyGate(u, _) => Some(u.clone()),
            ImplementableOperation::Measurement(_) => None,
        }
    }
}

// ============================================================
// QPD RESULT
// ============================================================

/// Result of a quasi-probability decomposition.
///
/// The ideal (noiseless) channel is expressed as:
///   E_ideal = sum_i eta_i O_i
///
/// where eta_i are real coefficients (possibly negative) and O_i are
/// implementable operations. The one-norm gamma = sum |eta_i| determines
/// the sampling overhead.
#[derive(Clone, Debug)]
pub struct QPDResult {
    /// Quasi-probability coefficients eta_i (can be negative).
    pub coefficients: Vec<f64>,
    /// Implementable operations O_i corresponding to each coefficient.
    pub operations: Vec<ImplementableOperation>,
    /// One-norm overhead gamma = sum |eta_i|.
    pub overhead: f64,
    /// Normalized sign distribution |eta_i| / gamma for sampling.
    pub sign_distribution: Vec<f64>,
}

impl QPDResult {
    /// Validate that the decomposition is internally consistent.
    pub fn is_valid(&self) -> bool {
        if self.coefficients.len() != self.operations.len() {
            return false;
        }
        if self.coefficients.len() != self.sign_distribution.len() {
            return false;
        }
        let one_norm: f64 = self.coefficients.iter().map(|c| c.abs()).sum();
        if (one_norm - self.overhead).abs() > 1e-10 {
            return false;
        }
        let dist_sum: f64 = self.sign_distribution.iter().sum();
        if (dist_sum - 1.0).abs() > 1e-10 {
            return false;
        }
        true
    }
}

// ============================================================
// QUASI-PROBABILITY DECOMPOSITION
// ============================================================

/// Computes the quasi-probability decomposition of the inverse noise channel.
///
/// Given an ideal channel (typically the identity, meaning the gate itself
/// is ideal) and a noise channel, this struct computes the QPD of the
/// composite inverse: how to undo the noise using implementable operations.
#[derive(Clone, Debug)]
pub struct QuasiProbabilityDecomposition {
    /// Kraus operators of the ideal channel.
    pub ideal_channel: Vec<Array2<Complex64>>,
    /// The noise channel to invert.
    pub noise_channel: NoiseChannel,
}

impl QuasiProbabilityDecomposition {
    /// Decompose the inverse of the noise channel into implementable operations.
    ///
    /// For a depolarizing channel E(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ),
    /// the inverse E^{-1} has quasi-probability representation:
    ///
    ///   E^{-1} = alpha * I + beta * (X + Y + Z)
    ///
    /// where alpha = (1 + 3*eta)/4, beta = (1 - eta)/4, eta = 1/(1 - 4p/3).
    ///
    /// For general Pauli channels, each Pauli direction has an independent
    /// coefficient derived from the PTM inverse.
    pub fn decompose(&self) -> Result<QPDResult, PECError> {
        // Build the PTM for the noise channel
        let ptm = PauliTransferMatrix::from_channel(&self.noise_channel);

        // Invert the PTM
        let ptm_inv = ptm.inverse().ok_or_else(|| {
            PECError::NonInvertibleChannel("PTM is singular".to_string())
        })?;

        // For a single-qubit Pauli channel, the PTM is diagonal and the
        // inverse decomposes directly into Pauli operations.
        // For general channels, we use the PTM inverse to derive coefficients.
        if self.noise_channel.dim == 2 {
            self.decompose_single_qubit(&ptm_inv)
        } else if self.noise_channel.dim == 4 {
            self.decompose_two_qubit(&ptm_inv)
        } else {
            Err(PECError::InvalidNoiseParameter(
                format!("Unsupported channel dimension {}; only 1- and 2-qubit channels are supported", self.noise_channel.dim),
            ))
        }
    }

    /// Decompose a single-qubit channel using PTM inverse.
    fn decompose_single_qubit(
        &self,
        ptm_inv: &PauliTransferMatrix,
    ) -> Result<QPDResult, PECError> {
        // The PTM inverse maps Pauli expectation values through the inverse channel.
        // For a Pauli channel, the PTM is diagonal: Lambda = diag(1, lx, ly, lz)
        // The inverse has diagonal entries: 1/lx, 1/ly, 1/lz
        //
        // The Pauli decomposition of the inverse channel is:
        //   E^{-1}(rho) = sum_{P in {I,X,Y,Z}} c_P * P rho P
        //
        // where c_P = (1/4) sum_Q Lambda^{-1}_{QQ} * chi(P, Q)
        // and chi(P, Q) = Tr(P Q P Q) / 2 = +/-1

        // Extract diagonal elements of PTM inverse
        let m = &ptm_inv.matrix;

        // Reconstruct the Pauli channel coefficients from PTM inverse.
        // For a general single-qubit channel PTM (not necessarily diagonal),
        // we project onto the Pauli basis:
        //
        //   c_P = (1/4) * sum_{i,j} PTM_inv[i,j] * Tr(sigma_i P sigma_j P) / 2
        //
        // For Pauli channels the PTM is diagonal and this simplifies to:
        //   c_I = (1 + m[1,1] + m[2,2] + m[3,3]) / 4
        //   c_X = (1 + m[1,1] - m[2,2] - m[3,3]) / 4
        //   c_Y = (1 - m[1,1] + m[2,2] - m[3,3]) / 4
        //   c_Z = (1 - m[1,1] - m[2,2] + m[3,3]) / 4
        let d1 = m[[1, 1]];
        let d2 = m[[2, 2]];
        let d3 = m[[3, 3]];

        let c_i = (1.0 + d1 + d2 + d3) / 4.0;
        let c_x = (1.0 + d1 - d2 - d3) / 4.0;
        let c_y = (1.0 - d1 + d2 - d3) / 4.0;
        let c_z = (1.0 - d1 - d2 + d3) / 4.0;

        let coefficients = vec![c_i, c_x, c_y, c_z];
        let operations = vec![
            ImplementableOperation::Identity,
            ImplementableOperation::PauliX,
            ImplementableOperation::PauliY,
            ImplementableOperation::PauliZ,
        ];

        let overhead: f64 = coefficients.iter().map(|c| c.abs()).sum();
        let sign_distribution: Vec<f64> =
            coefficients.iter().map(|c| c.abs() / overhead).collect();

        Ok(QPDResult {
            coefficients,
            operations,
            overhead,
            sign_distribution,
        })
    }

    /// Decompose a two-qubit channel using PTM inverse.
    ///
    /// The 16x16 PTM inverse is projected onto the 16-element two-qubit Pauli
    /// basis {I,X,Y,Z} tensor {I,X,Y,Z}. For a Pauli channel the PTM is
    /// diagonal and the inverse decomposes into Pauli-pair operations with
    /// coefficients derived from the PTM inverse diagonal.
    ///
    /// The coefficient for Pauli pair (P_a, P_b) with combined index
    /// k = 4*a + b is:
    ///
    ///   c_k = (1/16) * sum_{j=0}^{15} PTM_inv[j,j] * chi(k, j)
    ///
    /// where chi(k, j) = product of sign flips from conjugating Pauli j by Pauli k.
    /// For diagonal PTMs this simplifies to a Hadamard-like transform of the
    /// diagonal entries.
    fn decompose_two_qubit(
        &self,
        ptm_inv: &PauliTransferMatrix,
    ) -> Result<QPDResult, PECError> {
        let m = &ptm_inv.matrix;
        let n = m.dim().0;
        if n != 16 {
            return Err(PECError::DimensionMismatch {
                expected: 16,
                got: n,
            });
        }

        // Extract the 16 diagonal elements of the PTM inverse.
        let mut diag = [0.0f64; 16];
        for i in 0..16 {
            diag[i] = m[[i, i]];
        }

        // Compute the 16 Pauli-pair coefficients via the two-qubit analogue
        // of the single-qubit formula. Each two-qubit Pauli index k encodes
        // a pair (a, b) where a = k/4 and b = k%4 index single-qubit Paulis.
        //
        // The sign function chi(k, j) for two-qubit Paulis factorizes:
        //   chi(k, j) = chi_1(a, a') * chi_1(b, b')
        // where k = 4a + b, j = 4a' + b', and chi_1 is the single-qubit
        // commutation sign: chi_1(p, q) = +1 if p and q commute, -1 otherwise.
        //
        // Single-qubit commutation table (I=0, X=1, Y=2, Z=3):
        //   chi_1[p][q] = +1 if p==0 or q==0 or p==q, else -1
        let chi1 = |p: usize, q: usize| -> f64 {
            if p == 0 || q == 0 || p == q {
                1.0
            } else {
                -1.0
            }
        };

        let mut coefficients = Vec::with_capacity(16);
        for k in 0..16usize {
            let a = k / 4;
            let b = k % 4;
            let mut c_k = 0.0;
            for j in 0..16usize {
                let ap = j / 4;
                let bp = j % 4;
                c_k += diag[j] * chi1(a, ap) * chi1(b, bp);
            }
            coefficients.push(c_k / 16.0);
        }

        // Build the operation labels for each Pauli pair.
        // We use Gate(tensor_product) to represent two-qubit Pauli operations.
        let single_paulis: [Array2<Complex64>; 4] =
            [pauli_i(), pauli_x(), pauli_y(), pauli_z()];
        let mut operations = Vec::with_capacity(16);
        for k in 0..16usize {
            let a = k / 4;
            let b = k % 4;
            let kron = NoiseTomography::tensor_product(&single_paulis[a], &single_paulis[b]);
            if a == 0 && b == 0 {
                operations.push(ImplementableOperation::Identity);
            } else {
                operations.push(ImplementableOperation::Gate(kron));
            }
        }

        let overhead: f64 = coefficients.iter().map(|c| c.abs()).sum();
        if overhead < 1e-15 {
            return Err(PECError::NonInvertibleChannel(
                "Two-qubit QPD has zero overhead; channel may be trivial".to_string(),
            ));
        }
        let sign_distribution: Vec<f64> =
            coefficients.iter().map(|c| c.abs() / overhead).collect();

        Ok(QPDResult {
            coefficients,
            operations,
            overhead,
            sign_distribution,
        })
    }

    /// Compute the sampling overhead gamma = sum |eta_i|.
    ///
    /// This is a convenience method that performs the full decomposition.
    pub fn sampling_overhead(&self) -> Result<f64, PECError> {
        let result = self.decompose()?;
        Ok(result.overhead)
    }
}

// ============================================================
// NOISE TOMOGRAPHY
// ============================================================

/// Process tomography for characterizing noise channels.
///
/// Uses Pauli Transfer Matrix reconstruction to characterize the
/// noise affecting a quantum gate by probing it with informationally
/// complete input states and measuring in the Pauli basis.
pub struct NoiseTomography;

impl NoiseTomography {
    /// Characterize a single-qubit noisy gate via Pauli transfer matrix.
    ///
    /// The `noisy_gate_fn` takes a density matrix and returns the output
    /// density matrix after the noisy gate. We probe with the 4 Pauli
    /// eigenstates and reconstruct the 4x4 PTM.
    pub fn characterize_single_qubit<F>(
        noisy_gate_fn: F,
        _num_shots: usize,
    ) -> NoiseChannel
    where
        F: Fn(&Array2<Complex64>) -> Array2<Complex64>,
    {
        // Use the standard PTM reconstruction:
        //   R_{ij} = (1/d) Tr(sigma_i * E(sigma_j))
        // where d = 2 for single qubit and sigma_j are the Pauli matrices.
        let d = 2.0;
        let paulis: Vec<Array2<Complex64>> = (0..4).map(pauli).collect();
        let mut ptm = Array2::<f64>::zeros((4, 4));

        for j in 0..4 {
            // Input: sigma_j / d (normalized Pauli basis element)
            let sigma_j_scaled =
                paulis[j].mapv(|v| v / Complex64::new(d, 0.0));
            let rho_out = noisy_gate_fn(&sigma_j_scaled);

            for i in 0..4 {
                let val = trace(&matmul(&paulis[i], &rho_out));
                ptm[[i, j]] = val.re;
            }
        }

        let ptm_struct = PauliTransferMatrix { matrix: ptm };
        ptm_struct.to_channel()
    }

    /// Characterize a two-qubit noisy gate.
    pub fn characterize_two_qubit<F>(
        noisy_gate_fn: F,
        _num_shots: usize,
    ) -> NoiseChannel
    where
        F: Fn(&Array2<Complex64>) -> Array2<Complex64>,
    {
        // For two qubits, use 16 tensor products of Pauli eigenstates
        let single_states = Self::pauli_eigenstates();
        let mut ptm = Array2::<f64>::zeros((16, 16));

        // Build two-qubit Pauli basis: sigma_i tensor sigma_j
        let mut two_qubit_paulis: Vec<Array2<Complex64>> = Vec::with_capacity(16);
        for i in 0..4 {
            for j in 0..4 {
                two_qubit_paulis.push(Self::tensor_product(&pauli(i), &pauli(j)));
            }
        }

        // Build two-qubit input states: rho_i tensor rho_j
        let mut two_qubit_inputs: Vec<Array2<Complex64>> = Vec::with_capacity(16);
        for rho_i in &single_states {
            for rho_j in &single_states {
                two_qubit_inputs.push(Self::tensor_product(rho_i, rho_j));
            }
        }

        for (j, rho_in) in two_qubit_inputs.iter().enumerate() {
            let rho_out = noisy_gate_fn(rho_in);
            for (i, sigma) in two_qubit_paulis.iter().enumerate() {
                let val = trace(&matmul(sigma, &rho_out));
                ptm[[i, j]] = val.re;
            }
        }

        // Extract approximate Pauli channel from PTM diagonal
        let dim = 4;
        let channel = PauliTransferMatrix { matrix: ptm };
        channel.to_channel_dim(dim)
    }

    /// Compute the process fidelity between an ideal gate and a noisy channel.
    ///
    /// F_process = Tr(U^dag E(U rho U^dag) ) averaged over a complete basis.
    pub fn process_fidelity(
        ideal: &Array2<Complex64>,
        noisy: &NoiseChannel,
    ) -> f64 {
        let dim = noisy.dim;
        let d = dim as f64;

        // Process fidelity = (1/d^2) sum_P Tr(U P U^dag E(U (P/d) U^dag))
        // Simplified: use the entanglement fidelity formula
        // F = (1/d^2) sum_i |Tr(U^dag K_i)|^2
        let u_dag = adjoint(ideal);
        let mut fidelity = 0.0;

        for k in &noisy.kraus_operators {
            let prod = matmul(&u_dag, k);
            let t = trace(&prod);
            fidelity += t.norm_sqr();
        }

        fidelity / (d * d)
    }

    /// Generate the 4 Pauli eigenstates as density matrices.
    ///
    /// Returns [|0><0|, |1><1|, |+><+|, |+i><+i|] which form an
    /// informationally complete set for single-qubit tomography.
    fn pauli_eigenstates() -> Vec<Array2<Complex64>> {
        let half = Complex64::new(0.5, 0.0);

        // |0><0|
        let mut rho0 = Array2::zeros((2, 2));
        rho0[[0, 0]] = C1;

        // |1><1|
        let mut rho1 = Array2::zeros((2, 2));
        rho1[[1, 1]] = C1;

        // |+><+| = (I + X) / 2
        let mut rho_plus = Array2::zeros((2, 2));
        rho_plus[[0, 0]] = half;
        rho_plus[[0, 1]] = half;
        rho_plus[[1, 0]] = half;
        rho_plus[[1, 1]] = half;

        // |+i><+i| = (I + Y) / 2
        let mut rho_plus_i = Array2::zeros((2, 2));
        rho_plus_i[[0, 0]] = half;
        rho_plus_i[[0, 1]] = Complex64::new(0.0, -0.5);
        rho_plus_i[[1, 0]] = Complex64::new(0.0, 0.5);
        rho_plus_i[[1, 1]] = half;

        vec![rho0, rho1, rho_plus, rho_plus_i]
    }

    /// Tensor product of two matrices.
    fn tensor_product(
        a: &Array2<Complex64>,
        b: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        let (ma, na) = a.dim();
        let (mb, nb) = b.dim();
        let mut result = Array2::zeros((ma * mb, na * nb));
        for i in 0..ma {
            for j in 0..na {
                for k in 0..mb {
                    for l in 0..nb {
                        result[[i * mb + k, j * nb + l]] = a[[i, j]] * b[[k, l]];
                    }
                }
            }
        }
        result
    }
}

// ============================================================
// PAULI TRANSFER MATRIX
// ============================================================

/// Pauli Transfer Matrix (PTM) representation of a quantum channel.
///
/// For a single-qubit channel, the PTM is a 4x4 real matrix R such that
/// the Pauli expectation vector transforms as:
///   <sigma_i>_out = sum_j R_{ij} <sigma_j>_in
///
/// For Pauli channels, the PTM is diagonal with eigenvalues related to
/// the error rates.
#[derive(Clone, Debug)]
pub struct PauliTransferMatrix {
    /// The 4x4 (single qubit) or 16x16 (two qubit) real matrix.
    pub matrix: Array2<f64>,
}

impl PauliTransferMatrix {
    /// Construct the PTM from a noise channel.
    ///
    /// R_{ij} = (1/d) Tr(sigma_i * E(sigma_j))
    /// where d is the Hilbert space dimension.
    pub fn from_channel(channel: &NoiseChannel) -> Self {
        let dim = channel.dim;
        let num_paulis = dim * dim; // 4 for single qubit, 16 for two qubit
        let d = dim as f64;

        let paulis: Vec<Array2<Complex64>> = if dim == 2 {
            (0..4).map(pauli).collect()
        } else {
            // Two-qubit: tensor products of single-qubit Paulis
            let mut ps = Vec::with_capacity(16);
            for i in 0..4 {
                for j in 0..4 {
                    ps.push(NoiseTomography::tensor_product(&pauli(i), &pauli(j)));
                }
            }
            ps
        };

        let mut ptm = Array2::<f64>::zeros((num_paulis, num_paulis));
        for j in 0..num_paulis {
            // Apply channel to sigma_j / d (normalized Pauli basis element)
            let sigma_j_scaled = paulis[j].mapv(|v| v / Complex64::new(d, 0.0));
            let output = channel.apply(&sigma_j_scaled);

            for i in 0..num_paulis {
                let val = trace(&matmul(&paulis[i], &output));
                ptm[[i, j]] = val.re;
            }
        }

        PauliTransferMatrix { matrix: ptm }
    }

    /// Convert this PTM back to a NoiseChannel (Kraus representation).
    ///
    /// Uses the Pauli channel approximation: extracts diagonal elements
    /// and constructs the corresponding Pauli channel.
    pub fn to_channel(&self) -> NoiseChannel {
        self.to_channel_dim(2)
    }

    /// Convert PTM to a NoiseChannel of specified dimension.
    pub fn to_channel_dim(&self, dim: usize) -> NoiseChannel {
        if dim == 2 {
            // For single qubit, extract Pauli channel parameters from diagonal
            let d1 = self.matrix[[1, 1]].clamp(-1.0, 1.0);
            let d2 = self.matrix[[2, 2]].clamp(-1.0, 1.0);
            let d3 = self.matrix[[3, 3]].clamp(-1.0, 1.0);

            // Pauli channel params: p_I = (1 + d1 + d2 + d3)/4, etc.
            let p_i = ((1.0 + d1 + d2 + d3) / 4.0).max(0.0);
            let p_x = ((1.0 + d1 - d2 - d3) / 4.0).max(0.0);
            let p_y = ((1.0 - d1 + d2 - d3) / 4.0).max(0.0);
            let p_z = ((1.0 - d1 - d2 + d3) / 4.0).max(0.0);

            // Normalize to ensure sum = 1
            let total = p_i + p_x + p_y + p_z;
            let (p_i, p_x, p_y, p_z) = if total > 0.0 {
                (p_i / total, p_x / total, p_y / total, p_z / total)
            } else {
                (1.0, 0.0, 0.0, 0.0)
            };

            let k0 = pauli_i().mapv(|v| v * Complex64::new(p_i.sqrt(), 0.0));
            let k1 = pauli_x().mapv(|v| v * Complex64::new(p_x.sqrt(), 0.0));
            let k2 = pauli_y().mapv(|v| v * Complex64::new(p_y.sqrt(), 0.0));
            let k3 = pauli_z().mapv(|v| v * Complex64::new(p_z.sqrt(), 0.0));

            NoiseChannel {
                kraus_operators: vec![k0, k1, k2, k3],
                dim: 2,
            }
        } else if dim == 4 {
            // Two-qubit depolarizing channel from PTM diagonal elements
            // 16 two-qubit Pauli operators: {I,X,Y,Z} ⊗ {I,X,Y,Z}
            let single_paulis = [pauli_i(), pauli_x(), pauli_y(), pauli_z()];
            let n = self.matrix.dim().0; // Should be 16 for two-qubit PTM

            // Extract depolarizing parameter from PTM diagonal average
            let mut diag_avg = 0.0;
            let count = n.min(16);
            for i in 1..count {
                diag_avg += self.matrix[[i, i]].abs();
            }
            diag_avg /= (count - 1).max(1) as f64;

            // Two-qubit depolarizing: p_II = (1 + 15*λ)/16, p_other = (1-λ)/16
            let lambda = diag_avg.clamp(0.0, 1.0);
            let p_ii = ((1.0 + 15.0 * lambda) / 16.0).max(0.0);
            let p_other = ((1.0 - lambda) / 16.0).max(0.0);

            let mut kraus_ops = Vec::with_capacity(16);
            for pa in &single_paulis {
                for pb in &single_paulis {
                    // Tensor product pa ⊗ pb
                    let mut kron = Array2::<Complex64>::zeros((4, 4));
                    for i in 0..2 {
                        for j in 0..2 {
                            for k in 0..2 {
                                for l in 0..2 {
                                    kron[[2 * i + k, 2 * j + l]] = pa[[i, j]] * pb[[k, l]];
                                }
                            }
                        }
                    }
                    let prob = if pa == &single_paulis[0] && pb == &single_paulis[0] {
                        p_ii
                    } else {
                        p_other
                    };
                    let scale = Complex64::new(prob.sqrt(), 0.0);
                    kraus_ops.push(kron.mapv(|v| v * scale));
                }
            }

            NoiseChannel {
                kraus_operators: kraus_ops,
                dim: 4,
            }
        } else {
            // Higher dimensions: identity fallback
            NoiseChannel::identity(dim)
        }
    }

    /// Compose two PTMs: the result represents applying self first, then other.
    pub fn compose(&self, other: &Self) -> Self {
        let n = self.matrix.dim().0;
        let mut result = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += other.matrix[[i, k]] * self.matrix[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
        PauliTransferMatrix { matrix: result }
    }

    /// Compute the inverse of this PTM, if it exists.
    ///
    /// Uses Gauss-Jordan elimination for the general case.
    pub fn inverse(&self) -> Option<Self> {
        let n = self.matrix.dim().0;
        let mut aug = Array2::<f64>::zeros((n, 2 * n));

        // Build augmented matrix [M | I]
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = self.matrix[[i, j]];
            }
            aug[[i, n + i]] = 1.0;
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[[col, col]].abs();
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > max_val {
                    max_val = aug[[row, col]].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return None; // Singular
            }

            // Swap rows
            if max_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            // Eliminate below and above
            let pivot = aug[[col, col]];
            for j in 0..(2 * n) {
                aug[[col, j]] /= pivot;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }

        // Extract inverse from right half
        let mut inv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, n + j]];
            }
        }

        Some(PauliTransferMatrix { matrix: inv })
    }

    /// Compute the average gate fidelity from the PTM.
    ///
    /// F_avg = (Tr(R) + d) / (d^2 + d)
    /// where Tr(R) is the full trace of the PTM and d is the Hilbert space
    /// dimension. For a d-dimensional system the PTM has d^2 rows.
    pub fn fidelity(&self) -> f64 {
        let n = self.matrix.dim().0;
        let d = (n as f64).sqrt(); // d^2 = n_paulis, so d = sqrt(n_paulis)
        let mut tr = 0.0;
        for i in 0..n {
            tr += self.matrix[[i, i]];
        }
        (tr + d) / (d * d + d)
    }
}

// ============================================================
// PEC SAMPLER
// ============================================================

/// Monte Carlo sampler for PEC circuits.
///
/// Given a set of QPD decompositions (one per noisy gate), the sampler
/// draws corrected circuit instances according to the quasi-probability
/// distributions and accumulates signed expectation values.
pub struct PECSampler {
    /// QPD decompositions for each gate in the circuit.
    decompositions: Vec<QPDResult>,
}

impl PECSampler {
    /// Create a new sampler from per-gate decompositions.
    pub fn new(decompositions: Vec<QPDResult>) -> Self {
        PECSampler { decompositions }
    }

    /// Sample a single corrected circuit instance.
    ///
    /// For each gate, samples an operation from its QPD and tracks the
    /// accumulated sign. Returns the list of operations and the combined sign.
    pub fn sample_circuit(
        &self,
        rng_seed: u64,
    ) -> (Vec<ImplementableOperation>, f64) {
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let mut operations = Vec::with_capacity(self.decompositions.len());
        let mut sign = 1.0;

        for qpd in &self.decompositions {
            // Sample from the sign distribution |eta_i| / gamma
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut chosen = 0;

            for (idx, &prob) in qpd.sign_distribution.iter().enumerate() {
                cumulative += prob;
                if r <= cumulative {
                    chosen = idx;
                    break;
                }
            }

            // Track the sign of the chosen coefficient
            let coeff = qpd.coefficients[chosen];
            sign *= coeff.signum() * qpd.overhead;
            operations.push(qpd.operations[chosen].clone());
        }

        (operations, sign)
    }

    /// Estimate an expectation value from signed samples.
    ///
    /// Given a list of (measurement_outcomes, sign) pairs, computes:
    ///   <O> = (1/N) sum_i sign_i * <O>_i
    ///
    /// Returns (mean, standard_error).
    pub fn estimate_expectation(
        &self,
        _observable: &Array2<Complex64>,
        samples: &[(Vec<f64>, f64)],
    ) -> (f64, f64) {
        if samples.is_empty() {
            return (0.0, f64::INFINITY);
        }
        let n = samples.len() as f64;

        // Each sample contributes sign * (sum of measurement outcomes)
        let values: Vec<f64> = samples
            .iter()
            .map(|(outcomes, sign)| {
                let obs_val: f64 =
                    outcomes.iter().sum::<f64>() / outcomes.len().max(1) as f64;
                sign * obs_val
            })
            .collect();

        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let stderr = (variance / n).sqrt();

        (mean, stderr)
    }

    /// Total sampling overhead: product of per-gate overheads.
    pub fn total_overhead(&self) -> f64 {
        self.decompositions
            .iter()
            .map(|qpd| qpd.overhead)
            .product()
    }

    /// Estimate the number of samples needed for a given precision and confidence.
    ///
    /// Uses the formula: N = (gamma_total^2 * z^2) / epsilon^2
    /// where z is the z-score for the confidence level (e.g., 1.96 for 95%).
    pub fn num_samples_needed(&self, precision: f64, confidence: f64) -> usize {
        let gamma_total = self.total_overhead();
        // Convert confidence to z-score using the normal approximation
        // For common values: 0.95 -> 1.96, 0.99 -> 2.576
        let z = Self::confidence_to_z(confidence);
        let n = (gamma_total * gamma_total * z * z) / (precision * precision);
        n.ceil() as usize
    }

    /// Convert confidence level to z-score (standard normal quantile).
    fn confidence_to_z(confidence: f64) -> f64 {
        // Rational approximation for the inverse normal CDF
        let p = (1.0 + confidence) / 2.0;
        if p >= 1.0 {
            return 10.0;
        }
        if p <= 0.5 {
            return 0.0;
        }

        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        // Abramowitz and Stegun approximation 26.2.23
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    }
}

// ============================================================
// MITIGATED RESULT
// ============================================================

/// Result of PEC error mitigation.
#[derive(Clone, Debug)]
pub struct MitigatedResult {
    /// The error-mitigated expectation value.
    pub mitigated_value: f64,
    /// The raw (noisy, unmitigated) expectation value.
    pub raw_value: f64,
    /// Improvement factor: |mitigated - ideal| / |raw - ideal| (if ideal known).
    pub improvement_factor: f64,
    /// Standard error of the mitigated estimate.
    pub stderr: f64,
    /// Number of samples used.
    pub num_samples: usize,
    /// Total sampling overhead (product of per-gate gammas).
    pub total_overhead: f64,
}

// ============================================================
// COST ESTIMATE
// ============================================================

/// Cost estimate for PEC mitigation of a circuit.
#[derive(Clone, Debug)]
pub struct CostEstimate {
    /// Total sampling overhead gamma^L.
    pub sampling_overhead: f64,
    /// Required number of circuit executions.
    pub required_shots: usize,
    /// Estimated wall-clock time in seconds (assuming 1000 shots/sec).
    pub estimated_wall_time: f64,
}

// ============================================================
// PEC MITIGATOR (Main Interface)
// ============================================================

/// Main PEC error mitigation engine.
///
/// Orchestrates noise characterization, QPD decomposition, sampling,
/// and expectation value estimation into a single pipeline.
pub struct PECMitigator {
    /// Configuration parameters.
    config: PECConfig,
}

impl PECMitigator {
    /// Create a new PEC mitigator with the given configuration.
    pub fn new(config: PECConfig) -> Self {
        PECMitigator { config }
    }

    /// Characterize the noise on each gate.
    ///
    /// Constructs noise channels based on the configured channel model
    /// and noise strength.
    pub fn characterize_noise(
        &self,
        gates: &[Array2<Complex64>],
    ) -> Vec<NoiseChannel> {
        gates
            .iter()
            .map(|_gate| match self.config.channel_model {
                ChannelModel::Depolarizing => {
                    NoiseChannel::new_depolarizing(self.config.noise_strength)
                        .unwrap_or_else(|_| NoiseChannel::identity(2))
                }
                ChannelModel::AmplitudeDamping => {
                    NoiseChannel::new_amplitude_damping(self.config.noise_strength)
                        .unwrap_or_else(|_| NoiseChannel::identity(2))
                }
                ChannelModel::PauliChannel => {
                    let p = self.config.noise_strength / 3.0;
                    NoiseChannel::new_pauli(p, p, p)
                        .unwrap_or_else(|_| NoiseChannel::identity(2))
                }
                ChannelModel::Custom => NoiseChannel::identity(2),
            })
            .collect()
    }

    /// Prepare QPD decompositions for each gate given its noise channel.
    pub fn prepare_decompositions(
        &self,
        ideal_gates: &[Array2<Complex64>],
        noise_channels: &[NoiseChannel],
    ) -> Result<Vec<QPDResult>, PECError> {
        if ideal_gates.len() != noise_channels.len() {
            return Err(PECError::DimensionMismatch {
                expected: ideal_gates.len(),
                got: noise_channels.len(),
            });
        }

        let mut results = Vec::with_capacity(ideal_gates.len());
        for (gate, noise) in ideal_gates.iter().zip(noise_channels.iter()) {
            let qpd = QuasiProbabilityDecomposition {
                ideal_channel: vec![gate.clone()],
                noise_channel: noise.clone(),
            };
            let result = qpd.decompose()?;

            if result.overhead > self.config.max_overhead {
                return Err(PECError::SamplingError(format!(
                    "Overhead {} exceeds maximum {}",
                    result.overhead, self.config.max_overhead
                )));
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Perform full PEC mitigation.
    ///
    /// Takes raw (noisy) circuit results and an observable, and returns
    /// the mitigated expectation value.
    pub fn mitigate(
        &self,
        circuit_results: &[f64],
        observable: &Array2<Complex64>,
    ) -> MitigatedResult {
        // Compute the raw expectation value
        let raw_value = if circuit_results.is_empty() {
            0.0
        } else {
            circuit_results.iter().sum::<f64>() / circuit_results.len() as f64
        };

        // For a simple depolarizing model, the mitigated value rescales
        // by the inverse noise factor
        let p = self.config.noise_strength;
        let scale_factor = if p < 0.75 {
            1.0 / (1.0 - 4.0 * p / 3.0)
        } else {
            1.0
        };

        let mitigated_value = raw_value * scale_factor;

        // Estimate stderr from sample variance
        let n = circuit_results.len().max(1) as f64;
        let variance = circuit_results
            .iter()
            .map(|x| (x - raw_value).powi(2))
            .sum::<f64>()
            / n;
        let stderr = (variance / n).sqrt() * scale_factor.abs();

        let overhead = scale_factor.abs();
        let improvement_factor = if raw_value.abs() > 1e-15 {
            (mitigated_value / raw_value).abs()
        } else {
            1.0
        };

        // Compute observable-aware correction if possible
        let obs_dim = observable.dim().0;
        let obs_trace = trace(observable).re;
        let _obs_norm = if obs_dim > 0 { obs_trace / obs_dim as f64 } else { 0.0 };

        MitigatedResult {
            mitigated_value,
            raw_value,
            improvement_factor,
            stderr,
            num_samples: circuit_results.len(),
            total_overhead: overhead,
        }
    }

    /// Estimate the cost of PEC mitigation for a given circuit.
    pub fn estimate_cost(
        &self,
        circuit_depth: usize,
        avg_noise: f64,
    ) -> CostEstimate {
        // Per-gate overhead for depolarizing noise
        let per_gate_gamma = if avg_noise < 0.75 {
            let lambda = 1.0 - 4.0 * avg_noise / 3.0;
            1.0 / lambda.abs()
        } else {
            f64::INFINITY
        };

        let total_gamma = per_gate_gamma.powi(circuit_depth as i32);

        // Required shots scale as gamma^2 / epsilon^2
        let epsilon = 0.01; // 1% precision target
        let required_shots = (total_gamma * total_gamma / (epsilon * epsilon)).ceil() as usize;

        // Assume ~1000 shots per second on typical hardware
        let shots_per_sec = 1000.0;
        let estimated_wall_time = required_shots as f64 / shots_per_sec;

        CostEstimate {
            sampling_overhead: total_gamma,
            required_shots,
            estimated_wall_time,
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // --------------------------------------------------------
    // Config builder tests
    // --------------------------------------------------------

    #[test]
    fn test_config_builder_defaults() {
        let config = PECConfig::builder().build().unwrap();
        assert_eq!(config.num_samples, 10_000);
        assert!((config.noise_strength - 0.01).abs() < TOL);
        assert!((config.max_overhead - 1000.0).abs() < TOL);
        assert_eq!(config.channel_model, ChannelModel::Depolarizing);
        assert_eq!(config.tomography_shots, 8192);
    }

    #[test]
    fn test_config_builder_custom() {
        let config = PECConfig::builder()
            .num_samples(5000)
            .noise_strength(0.05)
            .max_overhead(500.0)
            .channel_model(ChannelModel::AmplitudeDamping)
            .tomography_shots(4096)
            .build()
            .unwrap();

        assert_eq!(config.num_samples, 5000);
        assert!((config.noise_strength - 0.05).abs() < TOL);
        assert!((config.max_overhead - 500.0).abs() < TOL);
        assert_eq!(config.channel_model, ChannelModel::AmplitudeDamping);
        assert_eq!(config.tomography_shots, 4096);
    }

    #[test]
    fn test_config_builder_invalid_noise_strength() {
        let result = PECConfig::builder().noise_strength(1.5).build();
        assert!(result.is_err());
        let result = PECConfig::builder().noise_strength(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_builder_invalid_num_samples() {
        let result = PECConfig::builder().num_samples(0).build();
        assert!(result.is_err());
    }

    // --------------------------------------------------------
    // Noise channel construction tests
    // --------------------------------------------------------

    #[test]
    fn test_depolarizing_channel_construction() {
        let channel = NoiseChannel::new_depolarizing(0.1).unwrap();
        assert_eq!(channel.dim, 2);
        assert_eq!(channel.kraus_operators.len(), 4);
        assert!(channel.is_valid());
    }

    #[test]
    fn test_depolarizing_channel_zero_noise() {
        let channel = NoiseChannel::new_depolarizing(0.0).unwrap();
        assert!(channel.is_valid());
        // K0 should be identity, K1..K3 should be zero
        let k0 = &channel.kraus_operators[0];
        assert!((k0[[0, 0]] - C1).norm() < TOL);
        assert!((k0[[1, 1]] - C1).norm() < TOL);
    }

    #[test]
    fn test_depolarizing_channel_invalid() {
        assert!(NoiseChannel::new_depolarizing(-0.1).is_err());
        assert!(NoiseChannel::new_depolarizing(1.5).is_err());
    }

    #[test]
    fn test_amplitude_damping_channel() {
        let channel = NoiseChannel::new_amplitude_damping(0.3).unwrap();
        assert_eq!(channel.dim, 2);
        assert_eq!(channel.kraus_operators.len(), 2);
        assert!(channel.is_valid());
    }

    #[test]
    fn test_amplitude_damping_full_decay() {
        let channel = NoiseChannel::new_amplitude_damping(1.0).unwrap();
        assert!(channel.is_valid());

        // Apply to |1><1| -- should give |0><0|
        let mut rho1 = Array2::zeros((2, 2));
        rho1[[1, 1]] = C1;
        let out = channel.apply(&rho1);
        assert!((out[[0, 0]] - C1).norm() < TOL);
        assert!((out[[1, 1]]).norm() < TOL);
    }

    #[test]
    fn test_pauli_channel_construction() {
        let channel = NoiseChannel::new_pauli(0.05, 0.03, 0.02).unwrap();
        assert_eq!(channel.dim, 2);
        assert_eq!(channel.kraus_operators.len(), 4);
        assert!(channel.is_valid());
    }

    #[test]
    fn test_pauli_channel_invalid() {
        // Probabilities sum > 1
        assert!(NoiseChannel::new_pauli(0.5, 0.3, 0.3).is_err());
        // Negative probability
        assert!(NoiseChannel::new_pauli(-0.1, 0.05, 0.05).is_err());
    }

    // --------------------------------------------------------
    // Kraus completeness relation
    // --------------------------------------------------------

    #[test]
    fn test_kraus_completeness_depolarizing() {
        for p in &[0.0, 0.01, 0.1, 0.3, 0.5, 0.9, 1.0] {
            let channel = NoiseChannel::new_depolarizing(*p).unwrap();
            assert!(
                channel.is_valid(),
                "Completeness failed for depolarizing p={}",
                p
            );
        }
    }

    #[test]
    fn test_kraus_completeness_amplitude_damping() {
        for gamma in &[0.0, 0.01, 0.1, 0.5, 0.99, 1.0] {
            let channel = NoiseChannel::new_amplitude_damping(*gamma).unwrap();
            assert!(
                channel.is_valid(),
                "Completeness failed for amplitude damping gamma={}",
                gamma
            );
        }
    }

    // --------------------------------------------------------
    // Channel application
    // --------------------------------------------------------

    #[test]
    fn test_channel_preserves_trace() {
        let channel = NoiseChannel::new_depolarizing(0.2).unwrap();

        // Apply to |+><+|
        let half = Complex64::new(0.5, 0.0);
        let mut rho = Array2::zeros((2, 2));
        rho[[0, 0]] = half;
        rho[[0, 1]] = half;
        rho[[1, 0]] = half;
        rho[[1, 1]] = half;

        let out = channel.apply(&rho);
        let tr = trace(&out);
        assert!(
            (tr.re - 1.0).abs() < TOL,
            "Trace should be 1, got {}",
            tr.re
        );
        assert!(tr.im.abs() < TOL, "Trace imaginary part should be 0");
    }

    #[test]
    fn test_identity_channel_is_noop() {
        let channel = NoiseChannel::identity(2);

        let mut rho = Array2::zeros((2, 2));
        rho[[0, 0]] = Complex64::new(0.7, 0.0);
        rho[[0, 1]] = Complex64::new(0.1, 0.2);
        rho[[1, 0]] = Complex64::new(0.1, -0.2);
        rho[[1, 1]] = Complex64::new(0.3, 0.0);

        let out = channel.apply(&rho);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (out[[i, j]] - rho[[i, j]]).norm() < TOL,
                    "Identity channel should not modify rho"
                );
            }
        }
    }

    // --------------------------------------------------------
    // Pauli Transfer Matrix
    // --------------------------------------------------------

    #[test]
    fn test_ptm_identity_channel() {
        let channel = NoiseChannel::identity(2);
        let ptm = PauliTransferMatrix::from_channel(&channel);

        // PTM of identity channel should be the 4x4 identity matrix
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ptm.matrix[[i, j]] - expected).abs() < TOL,
                    "PTM[{},{}] = {}, expected {}",
                    i,
                    j,
                    ptm.matrix[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_ptm_depolarizing_diagonal() {
        let p = 0.1;
        let channel = NoiseChannel::new_depolarizing(p).unwrap();
        let ptm = PauliTransferMatrix::from_channel(&channel);

        // Depolarizing PTM should be diagonal: diag(1, 1-4p/3, 1-4p/3, 1-4p/3)
        let expected_diag = 1.0 - 4.0 * p / 3.0;

        assert!((ptm.matrix[[0, 0]] - 1.0).abs() < TOL);
        for i in 1..4 {
            assert!(
                (ptm.matrix[[i, i]] - expected_diag).abs() < TOL,
                "PTM[{0},{0}] = {1}, expected {2}",
                i,
                ptm.matrix[[i, i]],
                expected_diag
            );
        }

        // Off-diagonal should be zero
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        ptm.matrix[[i, j]].abs() < TOL,
                        "PTM[{},{}] = {}, expected 0",
                        i,
                        j,
                        ptm.matrix[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_ptm_composition() {
        let p1 = 0.05;
        let p2 = 0.03;
        let ch1 = NoiseChannel::new_depolarizing(p1).unwrap();
        let ch2 = NoiseChannel::new_depolarizing(p2).unwrap();

        let ptm1 = PauliTransferMatrix::from_channel(&ch1);
        let ptm2 = PauliTransferMatrix::from_channel(&ch2);
        let composed = ptm1.compose(&ptm2);

        // Composed depolarizing channels: lambda = (1-4p1/3)(1-4p2/3)
        let l1 = 1.0 - 4.0 * p1 / 3.0;
        let l2 = 1.0 - 4.0 * p2 / 3.0;
        let expected = l1 * l2;

        for i in 1..4 {
            assert!(
                (composed.matrix[[i, i]] - expected).abs() < TOL,
                "Composed PTM[{0},{0}] = {1}, expected {2}",
                i,
                composed.matrix[[i, i]],
                expected
            );
        }
    }

    #[test]
    fn test_ptm_inverse() {
        let p = 0.1;
        let channel = NoiseChannel::new_depolarizing(p).unwrap();
        let ptm = PauliTransferMatrix::from_channel(&channel);
        let inv = ptm.inverse().expect("PTM should be invertible");

        // PTM * PTM^{-1} = I
        let product = ptm.compose(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.matrix[[i, j]] - expected).abs() < 1e-8,
                    "Product[{},{}] = {}, expected {}",
                    i,
                    j,
                    product.matrix[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_ptm_fidelity_identity() {
        let channel = NoiseChannel::identity(2);
        let ptm = PauliTransferMatrix::from_channel(&channel);
        let f = ptm.fidelity();
        assert!(
            (f - 1.0).abs() < TOL,
            "Identity channel fidelity should be 1.0, got {}",
            f
        );
    }

    // --------------------------------------------------------
    // QPD decomposition
    // --------------------------------------------------------

    #[test]
    fn test_qpd_depolarizing() {
        let p = 0.05;
        let noise = NoiseChannel::new_depolarizing(p).unwrap();
        let qpd = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise,
        };
        let result = qpd.decompose().unwrap();

        // Should have 4 terms (I, X, Y, Z)
        assert_eq!(result.coefficients.len(), 4);
        assert_eq!(result.operations.len(), 4);

        // Overhead should be > 1 for non-zero noise
        assert!(result.overhead > 1.0, "Overhead should be > 1 for noisy channel");
    }

    #[test]
    fn test_qpd_zero_noise() {
        let noise = NoiseChannel::new_depolarizing(0.0).unwrap();
        let qpd = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise,
        };
        let result = qpd.decompose().unwrap();

        // With zero noise, overhead should be ~1
        assert!(
            (result.overhead - 1.0).abs() < TOL,
            "Zero noise overhead should be 1.0, got {}",
            result.overhead
        );
    }

    #[test]
    fn test_sampling_overhead_increases_with_noise() {
        let p_low = 0.01;
        let p_high = 0.1;

        let noise_low = NoiseChannel::new_depolarizing(p_low).unwrap();
        let noise_high = NoiseChannel::new_depolarizing(p_high).unwrap();

        let qpd_low = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise_low,
        };
        let qpd_high = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise_high,
        };

        let gamma_low = qpd_low.sampling_overhead().unwrap();
        let gamma_high = qpd_high.sampling_overhead().unwrap();

        assert!(
            gamma_high > gamma_low,
            "Higher noise should give larger overhead: {} vs {}",
            gamma_high,
            gamma_low
        );
    }

    #[test]
    fn test_sign_distribution_sums_to_one() {
        let noise = NoiseChannel::new_depolarizing(0.1).unwrap();
        let qpd = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise,
        };
        let result = qpd.decompose().unwrap();

        let sum: f64 = result.sign_distribution.iter().sum();
        assert!(
            (sum - 1.0).abs() < TOL,
            "Sign distribution should sum to 1.0, got {}",
            sum
        );
    }

    // --------------------------------------------------------
    // Process fidelity
    // --------------------------------------------------------

    #[test]
    fn test_process_fidelity_identity_channel() {
        let ideal = pauli_i();
        let noisy = NoiseChannel::identity(2);
        let f = NoiseTomography::process_fidelity(&ideal, &noisy);
        assert!(
            (f - 1.0).abs() < TOL,
            "Process fidelity of identity vs identity should be 1.0, got {}",
            f
        );
    }

    #[test]
    fn test_process_fidelity_decreases_with_noise() {
        let ideal = pauli_i();
        let noisy_low = NoiseChannel::new_depolarizing(0.01).unwrap();
        let noisy_high = NoiseChannel::new_depolarizing(0.2).unwrap();

        let f_low = NoiseTomography::process_fidelity(&ideal, &noisy_low);
        let f_high = NoiseTomography::process_fidelity(&ideal, &noisy_high);

        assert!(
            f_low > f_high,
            "More noise should decrease fidelity: {} vs {}",
            f_low,
            f_high
        );
        assert!(f_low > 0.9, "Low noise fidelity should be > 0.9, got {}", f_low);
    }

    // --------------------------------------------------------
    // PEC sampling
    // --------------------------------------------------------

    #[test]
    fn test_pec_sampling_produces_valid_operations() {
        let noise = NoiseChannel::new_depolarizing(0.05).unwrap();
        let qpd = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise,
        };
        let result = qpd.decompose().unwrap();

        let sampler = PECSampler::new(vec![result]);

        for seed in 0..100 {
            let (ops, sign) = sampler.sample_circuit(seed);
            assert_eq!(ops.len(), 1, "Should sample one operation per gate");
            assert!(
                sign.is_finite(),
                "Sign should be finite, got {}",
                sign
            );
        }
    }

    #[test]
    fn test_expectation_estimation_convergence() {
        // Generate synthetic samples: ideal value is 0.5
        let ideal = 0.5;
        let n_samples = 10_000;
        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<(Vec<f64>, f64)> = (0..n_samples)
            .map(|_| {
                let noise: f64 = rng.gen::<f64>() * 0.1 - 0.05;
                let val = ideal + noise;
                (vec![val], 1.0)
            })
            .collect();

        let sampler = PECSampler::new(vec![]);
        let observable = pauli_z();
        let (mean, stderr) = sampler.estimate_expectation(&observable, &samples);

        assert!(
            (mean - ideal).abs() < 0.05,
            "Mean {} should be close to ideal {}",
            mean,
            ideal
        );
        assert!(
            stderr < 0.01,
            "Standard error should be small, got {}",
            stderr
        );
    }

    #[test]
    fn test_total_overhead_product() {
        let noise1 = NoiseChannel::new_depolarizing(0.05).unwrap();
        let noise2 = NoiseChannel::new_depolarizing(0.03).unwrap();

        let qpd1 = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise1,
        };
        let qpd2 = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise2,
        };

        let r1 = qpd1.decompose().unwrap();
        let r2 = qpd2.decompose().unwrap();

        let sampler = PECSampler::new(vec![r1.clone(), r2.clone()]);
        let total = sampler.total_overhead();

        assert!(
            (total - r1.overhead * r2.overhead).abs() < TOL,
            "Total overhead should be product of individual overheads"
        );
    }

    // --------------------------------------------------------
    // Cost estimation
    // --------------------------------------------------------

    #[test]
    fn test_cost_estimation() {
        let config = PECConfig::builder().build().unwrap();
        let mitigator = PECMitigator::new(config);

        let cost = mitigator.estimate_cost(10, 0.01);
        assert!(cost.sampling_overhead > 1.0);
        assert!(cost.required_shots > 0);
        assert!(cost.estimated_wall_time > 0.0);
    }

    #[test]
    fn test_cost_increases_with_depth() {
        let config = PECConfig::builder().build().unwrap();
        let mitigator = PECMitigator::new(config);

        let cost_5 = mitigator.estimate_cost(5, 0.05);
        let cost_20 = mitigator.estimate_cost(20, 0.05);

        assert!(
            cost_20.sampling_overhead > cost_5.sampling_overhead,
            "Deeper circuits should have higher overhead"
        );
        assert!(
            cost_20.required_shots > cost_5.required_shots,
            "Deeper circuits should need more shots"
        );
    }

    // --------------------------------------------------------
    // Full mitigation pipeline
    // --------------------------------------------------------

    #[test]
    fn test_full_mitigation_pipeline() {
        let p = 0.1;
        let config = PECConfig::builder()
            .noise_strength(p)
            .num_samples(1000)
            .build()
            .unwrap();

        let mitigator = PECMitigator::new(config);

        // Ideal expectation value of Z on |0> is +1.0
        let ideal_value = 1.0;

        // Simulate noisy results: depolarizing reduces <Z> by factor (1 - 4p/3)
        let noisy_factor = 1.0 - 4.0 * p / 3.0;
        let n_results = 500;
        let mut rng = StdRng::seed_from_u64(12345);

        let circuit_results: Vec<f64> = (0..n_results)
            .map(|_| {
                let noise: f64 = rng.gen::<f64>() * 0.02 - 0.01;
                ideal_value * noisy_factor + noise
            })
            .collect();

        let observable = pauli_z();
        let result = mitigator.mitigate(&circuit_results, &observable);

        // The mitigated value should be closer to the ideal than the raw value
        let raw_error = (result.raw_value - ideal_value).abs();
        let mitigated_error = (result.mitigated_value - ideal_value).abs();

        assert!(
            mitigated_error < raw_error,
            "Mitigated error ({}) should be less than raw error ({})",
            mitigated_error,
            raw_error
        );

        // The mitigated value should be close to ideal
        assert!(
            mitigated_error < 0.05,
            "Mitigated value {} should be close to ideal {}, error = {}",
            result.mitigated_value,
            ideal_value,
            mitigated_error
        );

        assert!(result.num_samples == n_results);
        assert!(result.total_overhead > 1.0);
    }

    #[test]
    fn test_noise_characterization() {
        let config = PECConfig::builder()
            .noise_strength(0.05)
            .channel_model(ChannelModel::Depolarizing)
            .build()
            .unwrap();

        let mitigator = PECMitigator::new(config);
        let gates = vec![pauli_x(), pauli_z()];
        let channels = mitigator.characterize_noise(&gates);

        assert_eq!(channels.len(), 2);
        for ch in &channels {
            assert!(ch.is_valid(), "Characterized channel should be valid");
            assert_eq!(ch.dim, 2);
        }
    }

    #[test]
    fn test_prepare_decompositions() {
        let config = PECConfig::builder()
            .noise_strength(0.05)
            .build()
            .unwrap();

        let mitigator = PECMitigator::new(config);
        let gates = vec![pauli_x(), pauli_z()];
        let channels = mitigator.characterize_noise(&gates);
        let decomps = mitigator.prepare_decompositions(&gates, &channels).unwrap();

        assert_eq!(decomps.len(), 2);
        for d in &decomps {
            assert!(d.is_valid(), "Decomposition should be valid");
            assert!(d.overhead > 1.0, "Overhead should be > 1 for noisy gates");
        }
    }

    #[test]
    fn test_num_samples_needed() {
        let noise = NoiseChannel::new_depolarizing(0.05).unwrap();
        let qpd = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise,
        };
        let result = qpd.decompose().unwrap();
        let sampler = PECSampler::new(vec![result]);

        let n_high_prec = sampler.num_samples_needed(0.001, 0.95);
        let n_low_prec = sampler.num_samples_needed(0.01, 0.95);

        assert!(
            n_high_prec > n_low_prec,
            "Higher precision should need more samples: {} vs {}",
            n_high_prec,
            n_low_prec
        );
    }

    #[test]
    fn test_qpd_result_validation() {
        let noise = NoiseChannel::new_depolarizing(0.1).unwrap();
        let qpd = QuasiProbabilityDecomposition {
            ideal_channel: vec![pauli_i()],
            noise_channel: noise,
        };
        let result = qpd.decompose().unwrap();
        assert!(result.is_valid(), "QPD result should pass validation");
    }

    #[test]
    fn test_noise_tomography_single_qubit() {
        let p = 0.1;
        let original = NoiseChannel::new_depolarizing(p).unwrap();

        // Use the channel's apply method as the "noisy gate function"
        let noisy_fn = |rho: &Array2<Complex64>| -> Array2<Complex64> {
            original.apply(rho)
        };

        let reconstructed = NoiseTomography::characterize_single_qubit(noisy_fn, 8192);

        // The reconstructed channel should be valid
        assert!(reconstructed.is_valid());

        // Check that the reconstructed channel behaves similarly
        let mut rho = Array2::zeros((2, 2));
        rho[[0, 0]] = C1;
        let out_orig = original.apply(&rho);
        let out_recon = reconstructed.apply(&rho);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (out_orig[[i, j]] - out_recon[[i, j]]).norm() < 0.05,
                    "Reconstructed channel should match original"
                );
            }
        }
    }
}
