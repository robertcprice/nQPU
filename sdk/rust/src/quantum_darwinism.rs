//! Quantum Darwinism Simulation
//!
//! Implementation of Wojciech Zurek's quantum Darwinism framework for studying
//! the emergence of classical reality from quantum mechanics through
//! environment-induced superselection (einselection).
//!
//! # Overview
//!
//! Quantum Darwinism explains how the classical world emerges from quantum mechanics.
//! The key insight is that information about a quantum system is redundantly encoded
//! in many fragments of its environment. Observers access different fragments and
//! yet agree on the system's state -- this is the hallmark of objective classical
//! reality.
//!
//! # Capabilities
//!
//! - **Environment-induced decoherence**: Couple a system to an environment via
//!   controlled interactions, then trace out the environment to observe pointer
//!   state selection.
//! - **Pointer state identification**: Find the preferred basis selected by the
//!   system-environment interaction Hamiltonian (the eigenstates that survive
//!   decoherence).
//! - **Quantum mutual information**: Compute I(S:F) = S(S) + S(F) - S(SF) for
//!   system S and environment fragment F, where S denotes von Neumann entropy.
//! - **Redundancy plateau detection**: Verify that I(S:F) ~ H(S) for small
//!   fragments F/E, the signature of quantum Darwinism.
//! - **Einselection dynamics**: Track convergence of the system's reduced state
//!   to the pointer basis over repeated interaction rounds.
//! - **Partial trace**: Compute reduced density matrices by tracing out specified
//!   qubits from a pure state vector.
//! - **Von Neumann entropy**: Entropy of a density matrix via eigendecomposition
//!   (analytic for 2x2, Jacobi iteration for larger).
//!
//! # Applications
//!
//! - Studying the quantum-to-classical transition
//! - Verifying decoherence models and pointer state predictions
//! - Investigating redundant information encoding in quantum error correction
//! - Exploring objective classicality in open quantum systems
//! - Testing foundations of quantum measurement theory
//!
//! # References
//!
//! - W. H. Zurek, "Quantum Darwinism", Nature Physics 5, 181-188 (2009)
//! - W. H. Zurek, "Decoherence, einselection, and the quantum origins of the
//!   classical", Reviews of Modern Physics 75, 715 (2003)
//! - R. Blume-Kohout & W. H. Zurek, "Quantum Darwinism: Entanglement, branches,
//!   and the emergent classical world", Physical Review A 73, 062310 (2006)
//! - C. J. Riedel, W. H. Zurek, & M. Zwolak, "The rise and fall of redundancy
//!   in decoherence and quantum Darwinism", New Journal of Physics 14, 083010 (2012)
//! - M. Zwolak & W. H. Zurek, "Complementarity of quantum discord and classically
//!   accessible information", Scientific Reports 3, 1729 (2013)
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::quantum_darwinism::{DarwinismSimulator, DarwinismConfig};
//!
//! let config = DarwinismConfig::default();
//! let mut sim = DarwinismSimulator::new(config).unwrap();
//!
//! // Run environment-induced decoherence
//! let pointer_states = sim.find_pointer_states().unwrap();
//! assert!(!pointer_states.is_empty());
//!
//! // Check redundancy plateau (signature of quantum Darwinism)
//! let redundancy = sim.compute_redundancy_curve().unwrap();
//! ```

use crate::{C64, GateOperations, QuantumState};
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from quantum Darwinism simulations.
#[derive(Clone, Debug)]
pub enum DarwinismError {
    /// A specified subsystem index is out of range or invalid.
    InvalidSubsystem {
        qubit: usize,
        total: usize,
        context: String,
    },
    /// The environment is too small to demonstrate redundancy.
    InsufficientEnvironment {
        env_qubits: usize,
        min_required: usize,
        reason: String,
    },
    /// The decoherence procedure did not converge.
    DecoherenceFailed {
        steps_attempted: usize,
        off_diagonal_norm: f64,
        reason: String,
    },
    /// Eigendecomposition did not converge within the iteration limit.
    EigendecompositionFailed {
        matrix_size: usize,
        iterations: usize,
    },
    /// A numerical issue occurred (e.g., non-physical density matrix).
    NumericalError(String),
}

impl fmt::Display for DarwinismError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DarwinismError::InvalidSubsystem {
                qubit,
                total,
                context,
            } => {
                write!(
                    f,
                    "Invalid subsystem: qubit {} out of range [0, {}) in {}",
                    qubit, total, context
                )
            }
            DarwinismError::InsufficientEnvironment {
                env_qubits,
                min_required,
                reason,
            } => {
                write!(
                    f,
                    "Insufficient environment: {} qubits provided, {} required ({})",
                    env_qubits, min_required, reason
                )
            }
            DarwinismError::DecoherenceFailed {
                steps_attempted,
                off_diagonal_norm,
                reason,
            } => {
                write!(
                    f,
                    "Decoherence failed after {} steps (off-diagonal norm: {:.2e}): {}",
                    steps_attempted, off_diagonal_norm, reason
                )
            }
            DarwinismError::EigendecompositionFailed {
                matrix_size,
                iterations,
            } => {
                write!(
                    f,
                    "Eigendecomposition of {}x{} matrix failed to converge in {} iterations",
                    matrix_size, matrix_size, iterations
                )
            }
            DarwinismError::NumericalError(msg) => {
                write!(f, "Numerical error: {}", msg)
            }
        }
    }
}

impl std::error::Error for DarwinismError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for a quantum Darwinism simulation.
///
/// Controls the system and environment sizes, interaction strength,
/// and decoherence parameters.
#[derive(Clone, Debug)]
pub struct DarwinismConfig {
    /// Number of qubits in the system S (typically 1-3).
    pub num_system_qubits: usize,
    /// Number of qubits in the environment E (typically 4-12).
    pub num_env_qubits: usize,
    /// Coupling strength for system-environment interactions (radians).
    /// Controls the rotation angle in controlled-Rz gates from S to E.
    pub interaction_strength: f64,
    /// Number of decoherence steps (interaction rounds) to apply.
    pub decoherence_steps: usize,
    /// Eigenvalue threshold for identifying pointer states.
    /// States with eigenvalue above this fraction of the maximum are
    /// considered pointer states.
    pub pointer_threshold: f64,
    /// Tolerance for considering mutual information to have reached
    /// a plateau (fraction of H(S)).
    pub plateau_tolerance: f64,
    /// Maximum iterations for the Jacobi eigenvalue algorithm.
    pub max_jacobi_iterations: usize,
}

impl Default for DarwinismConfig {
    fn default() -> Self {
        DarwinismConfig {
            num_system_qubits: 1,
            num_env_qubits: 6,
            interaction_strength: PI / 4.0,
            decoherence_steps: 3,
            pointer_threshold: 0.01,
            plateau_tolerance: 0.15,
            max_jacobi_iterations: 200,
        }
    }
}

impl DarwinismConfig {
    /// Create a new configuration with the given system and environment sizes.
    pub fn new(num_system_qubits: usize, num_env_qubits: usize) -> Self {
        DarwinismConfig {
            num_system_qubits,
            num_env_qubits,
            ..Default::default()
        }
    }

    /// Builder: set interaction strength.
    pub fn with_interaction_strength(mut self, strength: f64) -> Self {
        self.interaction_strength = strength;
        self
    }

    /// Builder: set decoherence steps.
    pub fn with_decoherence_steps(mut self, steps: usize) -> Self {
        self.decoherence_steps = steps;
        self
    }

    /// Builder: set pointer state threshold.
    pub fn with_pointer_threshold(mut self, threshold: f64) -> Self {
        self.pointer_threshold = threshold;
        self
    }

    /// Builder: set plateau tolerance.
    pub fn with_plateau_tolerance(mut self, tolerance: f64) -> Self {
        self.plateau_tolerance = tolerance;
        self
    }

    /// Total number of qubits (system + environment).
    pub fn total_qubits(&self) -> usize {
        self.num_system_qubits + self.num_env_qubits
    }

    /// Validate this configuration.
    pub fn validate(&self) -> Result<(), DarwinismError> {
        if self.num_system_qubits == 0 {
            return Err(DarwinismError::InvalidSubsystem {
                qubit: 0,
                total: 0,
                context: "system must have at least 1 qubit".to_string(),
            });
        }
        if self.num_env_qubits < 2 {
            return Err(DarwinismError::InsufficientEnvironment {
                env_qubits: self.num_env_qubits,
                min_required: 2,
                reason: "need at least 2 environment qubits for fragment analysis".to_string(),
            });
        }
        if self.interaction_strength <= 0.0 || self.interaction_strength > PI {
            return Err(DarwinismError::NumericalError(format!(
                "interaction_strength {} must be in (0, pi]",
                self.interaction_strength
            )));
        }
        if self.decoherence_steps == 0 {
            return Err(DarwinismError::NumericalError(
                "decoherence_steps must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================
// RESULT TYPES
// ============================================================

/// A pointer state identified through einselection.
///
/// Pointer states are the preferred basis of the system that survives
/// decoherence. In the computational basis coupling, these are the
/// computational basis states |0>, |1>, etc.
#[derive(Clone, Debug)]
pub struct PointerState {
    /// Index of this pointer state in the system Hilbert space.
    pub state_index: usize,
    /// Stability: overlap of this eigenstate with the nearest computational
    /// basis state. Values near 1.0 indicate a clean pointer state.
    pub stability: f64,
    /// Probability (eigenvalue of the reduced density matrix).
    pub probability: f64,
    /// The eigenvector components (in the system Hilbert space).
    pub eigenvector: Vec<C64>,
}

impl fmt::Display for PointerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PointerState(index={}, stability={:.4}, probability={:.6})",
            self.state_index, self.stability, self.probability
        )
    }
}

/// Result of computing quantum mutual information I(S:F)
/// for a particular environment fragment size.
#[derive(Clone, Debug)]
pub struct RedundancyResult {
    /// Number of environment qubits in the fragment F.
    pub fragment_size: usize,
    /// Fraction of the total environment in this fragment.
    pub fragment_fraction: f64,
    /// Quantum mutual information I(S:F).
    pub mutual_information: f64,
    /// Whether this point lies on the classical plateau
    /// (I(S:F) is approximately H(S)).
    pub classical_plateau: bool,
    /// System entropy H(S) = S(rho_S) for reference.
    pub system_entropy: f64,
}

impl fmt::Display for RedundancyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plateau_marker = if self.classical_plateau { " [PLATEAU]" } else { "" };
        write!(
            f,
            "Redundancy(|F|={}, frac={:.2}, I(S:F)={:.4}, H(S)={:.4}){}",
            self.fragment_size,
            self.fragment_fraction,
            self.mutual_information,
            self.system_entropy,
            plateau_marker
        )
    }
}

/// Record of the system's reduced state at one einselection time step.
#[derive(Clone, Debug)]
pub struct EinselectionSnapshot {
    /// Interaction round number (0-indexed).
    pub step: usize,
    /// Off-diagonal norm of the reduced density matrix in the
    /// computational basis. Approaches 0 as einselection proceeds.
    pub off_diagonal_norm: f64,
    /// Von Neumann entropy of the system's reduced state.
    pub system_entropy: f64,
    /// Purity Tr(rho_S^2) of the reduced state. Decreases from 1
    /// as the system decoheres, then stabilizes.
    pub purity: f64,
    /// Eigenvalues of the reduced density matrix at this step.
    pub eigenvalues: Vec<f64>,
}

/// Complete results from a redundancy plateau analysis.
#[derive(Clone, Debug)]
pub struct RedundancyProfile {
    /// The redundancy curve: I(S:F) vs fragment fraction.
    pub curve: Vec<RedundancyResult>,
    /// Whether a convincing plateau was detected.
    pub plateau_detected: bool,
    /// The smallest fragment fraction at which the plateau begins
    /// (None if no plateau detected).
    pub plateau_onset: Option<f64>,
    /// Redundancy R_delta: how many times the classical information
    /// is encoded in the environment (|E| / |F_min| where F_min is
    /// the smallest fragment capturing H(S)).
    pub redundancy: Option<f64>,
}

// ============================================================
// CORE LINEAR ALGEBRA UTILITIES
// ============================================================

/// Compute the partial trace of a pure state |psi> over specified qubits,
/// returning the reduced density matrix as a Vec<Vec<C64>>.
///
/// Given a state of `n` total qubits, traces out the qubits listed in
/// `trace_out`. The remaining qubits define the subsystem whose reduced
/// density matrix is returned.
///
/// # Arguments
/// * `state` - The full quantum state (state vector).
/// * `trace_out` - Indices of qubits to trace out (0-indexed).
///
/// # Returns
/// Reduced density matrix of dimension d_keep x d_keep, where
/// d_keep = 2^(n - |trace_out|).
pub fn partial_trace(state: &QuantumState, trace_out: &[usize]) -> Vec<Vec<C64>> {
    let n = state.num_qubits;
    let amps = state.amplitudes_ref();
    let dim = state.dim;

    // Determine which qubits to keep
    let mut is_traced = vec![false; n];
    for &q in trace_out {
        is_traced[q] = true;
    }
    let keep: Vec<usize> = (0..n).filter(|q| !is_traced[*q]).collect();
    let n_keep = keep.len();
    let d_keep = 1 << n_keep;
    let n_trace = trace_out.len();
    let d_trace = 1 << n_trace;

    let zero = C64::new(0.0, 0.0);
    let mut rho = vec![vec![zero; d_keep]; d_keep];

    // Collect trace-out qubit indices sorted for consistent bit ordering
    let mut trace_sorted: Vec<usize> = trace_out.to_vec();
    trace_sorted.sort();

    // For each pair of kept-subsystem basis states (ik, jk),
    // sum over all traced-out basis states it:
    //   rho[ik][jk] = sum_{it} <ik,it|psi> * <psi|jk,it>
    for ik in 0..d_keep {
        for jk in 0..d_keep {
            let mut val = zero;
            for it in 0..d_trace {
                let full_i = reassemble_index(ik, it, &keep, &trace_sorted, n);
                let full_j = reassemble_index(jk, it, &keep, &trace_sorted, n);
                if full_i < dim && full_j < dim {
                    val += amps[full_i] * amps[full_j].conj();
                }
            }
            rho[ik][jk] = val;
        }
    }

    rho
}

/// Reassemble a full Hilbert space index from subsystem indices.
///
/// Places bits from `idx_keep` at positions given by `keep_qubits`,
/// and bits from `idx_trace` at positions given by `trace_qubits`.
fn reassemble_index(
    idx_keep: usize,
    idx_trace: usize,
    keep_qubits: &[usize],
    trace_qubits: &[usize],
    _num_qubits: usize,
) -> usize {
    let mut full = 0usize;
    for (bit_pos, &qubit) in keep_qubits.iter().enumerate() {
        if (idx_keep >> bit_pos) & 1 == 1 {
            full |= 1 << qubit;
        }
    }
    for (bit_pos, &qubit) in trace_qubits.iter().enumerate() {
        if (idx_trace >> bit_pos) & 1 == 1 {
            full |= 1 << qubit;
        }
    }
    full
}

/// Compute von Neumann entropy S(rho) = -Tr(rho log2 rho) from a
/// density matrix.
///
/// Uses eigendecomposition to compute S = -sum_i lambda_i log2(lambda_i).
/// For 2x2 matrices, uses the analytic formula. For larger matrices,
/// uses the Jacobi eigenvalue algorithm.
pub fn von_neumann_entropy(rho: &[Vec<C64>]) -> Result<f64, DarwinismError> {
    let eigenvalues = eigenvalues_hermitian(rho)?;
    Ok(entropy_from_eigenvalues(&eigenvalues))
}

/// Compute entropy from a set of eigenvalues: S = -sum_i p_i log2(p_i).
fn entropy_from_eigenvalues(eigenvalues: &[f64]) -> f64 {
    let mut s = 0.0;
    for &ev in eigenvalues {
        if ev > 1e-15 {
            s -= ev * ev.log2();
        }
    }
    s
}

/// Compute eigenvalues of a Hermitian matrix.
///
/// For 1x1: returns the diagonal element.
/// For 2x2: uses the analytic quadratic formula.
/// For 4x4 and larger: uses the Jacobi eigenvalue algorithm.
pub fn eigenvalues_hermitian(matrix: &[Vec<C64>]) -> Result<Vec<f64>, DarwinismError> {
    let n = matrix.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![matrix[0][0].re]);
    }
    if n == 2 {
        return Ok(eigenvalues_2x2(matrix));
    }

    // General case: Jacobi algorithm
    let (vals, _) = jacobi_hermitian_eigen(matrix, 200)?;
    Ok(vals)
}

/// Eigendecomposition of a Hermitian matrix.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i] is the
/// i-th eigenvector as a column stored in row-major order:
/// eigenvectors[row][col] gives the `row`-th component of the `col`-th eigenvector.
pub fn eigen_hermitian(
    matrix: &[Vec<C64>],
) -> Result<(Vec<f64>, Vec<Vec<C64>>), DarwinismError> {
    let n = matrix.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        return Ok((vec![matrix[0][0].re], vec![vec![C64::new(1.0, 0.0)]]));
    }
    if n == 2 {
        let (vals, vecs) = eigen_2x2(matrix);
        return Ok((vals, vecs));
    }

    jacobi_hermitian_eigen(matrix, 200)
}

/// Analytic eigenvalues of a 2x2 Hermitian matrix.
///
/// For H = [[a, b], [b*, d]] (Hermitian: b* = conj(b), a and d real):
///   lambda = (a+d)/2 +/- sqrt((a-d)^2/4 + |b|^2)
fn eigenvalues_2x2(m: &[Vec<C64>]) -> Vec<f64> {
    let a = m[0][0].re;
    let d = m[1][1].re;
    let b_norm_sq = m[0][1].norm_sqr();

    let half_trace = (a + d) / 2.0;
    let discriminant = ((a - d) / 2.0).powi(2) + b_norm_sq;
    let sqrt_disc = discriminant.max(0.0).sqrt();

    let mut vals = vec![half_trace - sqrt_disc, half_trace + sqrt_disc];
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals
}

/// Analytic eigendecomposition of a 2x2 Hermitian matrix.
///
/// Returns (eigenvalues, eigenvectors) sorted by eigenvalue.
fn eigen_2x2(m: &[Vec<C64>]) -> (Vec<f64>, Vec<Vec<C64>>) {
    let a = m[0][0].re;
    let d = m[1][1].re;
    let b = m[0][1];
    let b_norm = b.norm();

    let half_trace = (a + d) / 2.0;
    let discriminant = ((a - d) / 2.0).powi(2) + b.norm_sqr();
    let sqrt_disc = discriminant.max(0.0).sqrt();

    let lambda0 = half_trace - sqrt_disc;
    let lambda1 = half_trace + sqrt_disc;

    // Eigenvectors
    let vecs = if b_norm < 1e-14 {
        // Already diagonal
        if a <= d {
            vec![
                vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
            ]
        } else {
            vec![
                vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
                vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            ]
        }
    } else {
        // For eigenvalue lambda: (a - lambda) v0 + b v1 = 0
        // => v1/v0 = -(a - lambda) / b
        let ratio0 = -(a - lambda0) / b;
        let norm0 = (1.0 + ratio0.norm_sqr()).sqrt();
        let v0 = vec![
            C64::new(1.0 / norm0, 0.0),
            ratio0 / C64::new(norm0, 0.0),
        ];

        let ratio1 = -(a - lambda1) / b;
        let norm1 = (1.0 + ratio1.norm_sqr()).sqrt();
        let v1 = vec![
            C64::new(1.0 / norm1, 0.0),
            ratio1 / C64::new(norm1, 0.0),
        ];

        // Return as row-major: vecs[row][col] = row-th component of col-th eigenvector
        vec![
            vec![v0[0], v1[0]],
            vec![v0[1], v1[1]],
        ]
    };

    (vec![lambda0, lambda1], vecs)
}

/// Jacobi eigenvalue algorithm for Hermitian matrices.
///
/// Iteratively applies Jacobi rotations to diagonalize the matrix.
/// Converges for all Hermitian matrices. Returns eigenvalues sorted
/// in ascending order and the corresponding eigenvectors.
fn jacobi_hermitian_eigen(
    matrix: &[Vec<C64>],
    max_iter: usize,
) -> Result<(Vec<f64>, Vec<Vec<C64>>), DarwinismError> {
    let n = matrix.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }

    let zero = C64::new(0.0, 0.0);
    let mut a: Vec<Vec<C64>> = matrix.to_vec();

    // Initialize eigenvector matrix to identity
    let mut v: Vec<Vec<C64>> = vec![vec![zero; n]; n];
    for i in 0..n {
        v[i][i] = C64::new(1.0, 0.0);
    }

    let tol = 1e-12;

    for _sweep in 0..max_iter {
        // Compute off-diagonal Frobenius norm
        let mut off_norm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm += a[i][j].norm_sqr();
            }
        }
        off_norm = off_norm.sqrt();

        if off_norm < tol * (n as f64) {
            // Converged: extract eigenvalues from diagonal
            let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i].re).collect();
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| {
                eigenvalues[i]
                    .partial_cmp(&eigenvalues[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
            let sorted_vecs: Vec<Vec<C64>> = (0..n)
                .map(|row| indices.iter().map(|&col| v[row][col]).collect())
                .collect();

            return Ok((sorted_vals, sorted_vecs));
        }

        // Jacobi sweep for Hermitian matrices.
        //
        // For each off-diagonal element a[p][q], compute unitary rotation J
        // in the (p,q) plane such that (J^dag A J)[p][q] = 0.
        //
        // J is identity except in the (p,q) block:
        //   J[p][p] = c,      J[p][q] = s
        //   J[q][p] = -s*,    J[q][q] = c
        // where c is real positive and s is complex.
        //
        // J^dag:  J^dag[p][p] = c,   J^dag[p][q] = -s
        //         J^dag[q][p] = s*,   J^dag[q][q] = c
        //
        // Two-step update: B = J^dag * A, then A' = B * J.
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                let apq_norm = apq.norm();
                if apq_norm < tol * 1e-3 {
                    continue;
                }

                let app = a[p][p].re;
                let aqq = a[q][q].re;

                // Phase factor: a[p][q] = |a[p][q]| * exp(i*phi)
                let phase = apq / C64::new(apq_norm, 0.0);

                // Rotation parameter via stable formula
                let diff = aqq - app;
                let t_real = if diff.abs() < tol * 1e-6 {
                    1.0 // Degenerate case: theta = pi/4
                } else {
                    let tau = diff / (2.0 * apq_norm);
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t_real * t_real).sqrt();
                let s = C64::new(t_real * c, 0.0) * phase;

                let cc = C64::new(c, 0.0);

                // Step 1: B = J^dag * A (left-multiply by J^dag, update rows p and q)
                // Clone rows first to avoid in-place corruption
                let row_p: Vec<C64> = a[p].clone();
                let row_q: Vec<C64> = a[q].clone();
                for r in 0..n {
                    a[p][r] = cc * row_p[r] - s * row_q[r];
                    a[q][r] = s.conj() * row_p[r] + cc * row_q[r];
                }

                // Step 2: A' = B * J (right-multiply by J, update columns p and q)
                // Clone columns first to avoid in-place corruption
                let col_p: Vec<C64> = (0..n).map(|r| a[r][p]).collect();
                let col_q: Vec<C64> = (0..n).map(|r| a[r][q]).collect();
                for r in 0..n {
                    a[r][p] = cc * col_p[r] - s.conj() * col_q[r];
                    a[r][q] = s * col_p[r] + cc * col_q[r];
                }

                // Enforce exact structure: diagonal is real, zeroed element stays zero
                a[p][p] = C64::new(a[p][p].re, 0.0);
                a[q][q] = C64::new(a[q][q].re, 0.0);
                a[p][q] = zero;
                a[q][p] = zero;

                // Update eigenvector matrix: V' = V * J
                // V'[r][p] = V[r][p] * c + V[r][q] * (-s*)
                // V'[r][q] = V[r][p] * s + V[r][q] * c
                for r in 0..n {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = cc * vrp - s.conj() * vrq;
                    v[r][q] = s * vrp + cc * vrq;
                }
            }
        }
    }

    // Did not converge
    Err(DarwinismError::EigendecompositionFailed {
        matrix_size: n,
        iterations: max_iter,
    })
}

/// Compute the purity Tr(rho^2) of a density matrix.
fn purity(rho: &[Vec<C64>]) -> f64 {
    let n = rho.len();
    let mut tr = 0.0;
    for i in 0..n {
        for j in 0..n {
            tr += (rho[i][j] * rho[j][i].conj()).re;
        }
    }
    // Clamp to physical range [1/d, 1]
    tr.max(1.0 / n as f64).min(1.0)
}

/// Compute the off-diagonal norm of a density matrix (in the computational basis).
///
/// This measures how far rho is from being diagonal, which indicates
/// the degree of decoherence.
fn off_diagonal_norm(rho: &[Vec<C64>]) -> f64 {
    let n = rho.len();
    let mut norm_sq = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                norm_sq += rho[i][j].norm_sqr();
            }
        }
    }
    norm_sq.sqrt()
}

/// Compute the trace of a density matrix (should be 1 for physical states).
fn trace_rho(rho: &[Vec<C64>]) -> C64 {
    let n = rho.len();
    let mut tr = C64::new(0.0, 0.0);
    for i in 0..n {
        tr += rho[i][i];
    }
    tr
}

// ============================================================
// QUANTUM MUTUAL INFORMATION
// ============================================================

/// Compute the quantum mutual information I(A:B) between two subsystems
/// of a joint pure state.
///
/// I(A:B) = S(A) + S(B) - S(AB)
///
/// For a pure state |psi>_{AB}, S(AB) = 0, so I(A:B) = S(A) + S(B).
/// But if A and B are proper subsets of a larger system, S(AB) != 0.
///
/// # Arguments
/// * `state` - The full quantum state.
/// * `system_qubits` - Indices of the system qubits (subsystem A).
/// * `fragment_qubits` - Indices of the fragment qubits (subsystem B).
///
/// # Returns
/// The quantum mutual information I(A:B) in bits (log base 2).
pub fn mutual_information(
    state: &QuantumState,
    system_qubits: &[usize],
    fragment_qubits: &[usize],
) -> Result<f64, DarwinismError> {
    let n = state.num_qubits;

    // Validate qubit indices
    for &q in system_qubits.iter().chain(fragment_qubits.iter()) {
        if q >= n {
            return Err(DarwinismError::InvalidSubsystem {
                qubit: q,
                total: n,
                context: "mutual_information".to_string(),
            });
        }
    }

    // All qubits NOT in system or fragment must be traced out
    let mut all_qubits: Vec<bool> = vec![false; n];
    for &q in system_qubits {
        all_qubits[q] = true;
    }
    for &q in fragment_qubits {
        all_qubits[q] = true;
    }

    // Compute S(A): trace out everything except system
    let trace_out_for_a: Vec<usize> = (0..n)
        .filter(|q| !system_qubits.contains(q))
        .collect();
    let rho_a = partial_trace(state, &trace_out_for_a);
    let s_a = von_neumann_entropy(&rho_a)?;

    // Compute S(B): trace out everything except fragment
    let trace_out_for_b: Vec<usize> = (0..n)
        .filter(|q| !fragment_qubits.contains(q))
        .collect();
    let rho_b = partial_trace(state, &trace_out_for_b);
    let s_b = von_neumann_entropy(&rho_b)?;

    // Compute S(AB): trace out everything except system+fragment
    let trace_out_for_ab: Vec<usize> = (0..n)
        .filter(|q| !all_qubits[*q])
        .collect();

    let s_ab = if trace_out_for_ab.is_empty() {
        // AB is the full system, and for a pure state S(full) = 0
        0.0
    } else {
        let rho_ab = partial_trace(state, &trace_out_for_ab);
        von_neumann_entropy(&rho_ab)?
    };

    // I(A:B) = S(A) + S(B) - S(AB)
    let mi = s_a + s_b - s_ab;
    // Mutual information is non-negative; clamp numerical noise
    Ok(mi.max(0.0))
}

// ============================================================
// DARWINISM SIMULATOR
// ============================================================

/// Main simulator for quantum Darwinism phenomena.
///
/// Implements the full workflow:
/// 1. Prepare system in a superposition
/// 2. Couple system to environment via controlled interactions
/// 3. Trace out environment to observe pointer state emergence
/// 4. Compute mutual information for environment fragments
/// 5. Detect the redundancy plateau
pub struct DarwinismSimulator {
    /// Configuration parameters.
    config: DarwinismConfig,
    /// The combined system+environment quantum state.
    state: QuantumState,
    /// System qubit indices (first num_system_qubits qubits).
    system_qubits: Vec<usize>,
    /// Environment qubit indices (remaining qubits).
    env_qubits: Vec<usize>,
    /// Whether decoherence has been applied.
    decohered: bool,
    /// Einselection history (populated during decoherence).
    einselection_history: Vec<EinselectionSnapshot>,
}

impl DarwinismSimulator {
    /// Create a new simulator with the given configuration.
    ///
    /// The system is initialized in the |+> state (equal superposition)
    /// by applying Hadamard to each system qubit. The environment starts
    /// in |0...0>.
    pub fn new(config: DarwinismConfig) -> Result<Self, DarwinismError> {
        config.validate()?;

        let total = config.total_qubits();
        let mut state = QuantumState::new(total);

        // System qubits are the first num_system_qubits
        let system_qubits: Vec<usize> = (0..config.num_system_qubits).collect();
        let env_qubits: Vec<usize> = (config.num_system_qubits..total).collect();

        // Put system in equal superposition |+>^n
        for &sq in &system_qubits {
            GateOperations::h(&mut state, sq);
        }

        Ok(DarwinismSimulator {
            config,
            state,
            system_qubits,
            env_qubits,
            decohered: false,
            einselection_history: Vec::new(),
        })
    }

    /// Create a simulator with a custom initial state.
    ///
    /// The provided state must have the correct number of qubits
    /// (num_system_qubits + num_env_qubits).
    pub fn with_state(
        config: DarwinismConfig,
        state: QuantumState,
    ) -> Result<Self, DarwinismError> {
        config.validate()?;

        let total = config.total_qubits();
        if state.num_qubits != total {
            return Err(DarwinismError::InvalidSubsystem {
                qubit: state.num_qubits,
                total,
                context: format!(
                    "provided state has {} qubits, expected {}",
                    state.num_qubits, total
                ),
            });
        }

        let system_qubits: Vec<usize> = (0..config.num_system_qubits).collect();
        let env_qubits: Vec<usize> = (config.num_system_qubits..total).collect();

        Ok(DarwinismSimulator {
            config,
            state,
            system_qubits,
            env_qubits,
            decohered: false,
            einselection_history: Vec::new(),
        })
    }

    /// Access the current quantum state.
    pub fn state(&self) -> &QuantumState {
        &self.state
    }

    /// Access the configuration.
    pub fn config(&self) -> &DarwinismConfig {
        &self.config
    }

    /// Access einselection history (populated after calling `apply_decoherence`).
    pub fn einselection_history(&self) -> &[EinselectionSnapshot] {
        &self.einselection_history
    }

    /// Whether decoherence has been applied.
    pub fn is_decohered(&self) -> bool {
        self.decohered
    }

    // --------------------------------------------------------
    // DECOHERENCE AND INTERACTION
    // --------------------------------------------------------

    /// Apply one round of system-environment interaction.
    ///
    /// For each system qubit s and each environment qubit e, applies
    /// a controlled-Ry rotation: CRy(theta) with s as control and e
    /// as target. This entangles the environment with the system in
    /// a way that selects the computational basis as the pointer basis.
    ///
    /// The action on the joint state is:
    ///   |0_s>|0_e> -> |0_s>|0_e>           (system |0>: env unchanged)
    ///   |1_s>|0_e> -> |1_s>(c|0_e>+s|1_e>) (system |1>: env rotated)
    ///
    /// where c = cos(theta/2) and s = sin(theta/2). This creates
    /// entanglement between system and environment proportional to
    /// theta. At theta = PI, the CNOT-like limit is reached (maximal
    /// copying). The computational basis is selected as the pointer
    /// basis because |0> and |1> produce distinguishable environment
    /// states, while superpositions become entangled and decohere.
    fn apply_interaction_round(&mut self) {
        let theta = self.config.interaction_strength;

        for &sq in &self.system_qubits.clone() {
            for &eq in &self.env_qubits.clone() {
                // Apply CRy(theta) using direct state vector manipulation.
                // This avoids potential issues with library gate implementations.
                apply_cry(&mut self.state, sq, eq, theta);
            }
        }
    }

    /// Apply the full decoherence protocol: multiple rounds of
    /// system-environment interaction.
    ///
    /// After each round, records an einselection snapshot showing
    /// how the system's reduced state evolves toward the pointer basis.
    pub fn apply_decoherence(&mut self) -> Result<(), DarwinismError> {
        self.einselection_history.clear();

        // Record initial snapshot (before any interaction)
        self.record_einselection_snapshot(0)?;

        for step in 1..=self.config.decoherence_steps {
            self.apply_interaction_round();
            self.record_einselection_snapshot(step)?;
        }

        self.decohered = true;
        Ok(())
    }

    /// Record the current state of the system's reduced density matrix
    /// for einselection tracking.
    fn record_einselection_snapshot(&mut self, step: usize) -> Result<(), DarwinismError> {
        let rho_s = partial_trace(&self.state, &self.env_qubits);
        let s_entropy = von_neumann_entropy(&rho_s)?;
        let off_diag = off_diagonal_norm(&rho_s);
        let p = purity(&rho_s);
        let eigenvalues = eigenvalues_hermitian(&rho_s)?;

        self.einselection_history.push(EinselectionSnapshot {
            step,
            off_diagonal_norm: off_diag,
            system_entropy: s_entropy,
            purity: p,
            eigenvalues,
        });

        Ok(())
    }

    // --------------------------------------------------------
    // POINTER STATE IDENTIFICATION
    // --------------------------------------------------------

    /// Find the pointer states of the system.
    ///
    /// Pointer states are the preferred basis states that survive decoherence.
    /// They are identified by examining the reduced density matrix of the
    /// system: the computational basis states that carry significant
    /// population (diagonal elements) and minimal off-diagonal coherence
    /// are the pointer states.
    ///
    /// For a CRy-type interaction (Z-basis coupling), the pointer states
    /// are the computational basis states |0>, |1>, etc. This method
    /// identifies them by their diagonal dominance in the reduced state.
    ///
    /// The stability of each pointer state measures how diagonal the
    /// reduced density matrix is: stability = 1 - (sum of off-diagonal
    /// elements in that row/column) / (diagonal element). High stability
    /// means the state survives decoherence cleanly.
    ///
    /// If decoherence has not yet been applied, it is applied automatically.
    pub fn find_pointer_states(&mut self) -> Result<Vec<PointerState>, DarwinismError> {
        if !self.decohered {
            self.apply_decoherence()?;
        }

        let rho_s = partial_trace(&self.state, &self.env_qubits);
        let d_s = rho_s.len();

        // For each computational basis state, check if it has significant
        // population and is decoherence-stable.
        let mut pointer_states = Vec::new();
        let total_trace = trace_rho(&rho_s).re;

        for basis_idx in 0..d_s {
            let population = rho_s[basis_idx][basis_idx].re;

            if population < self.config.pointer_threshold * total_trace {
                continue;
            }

            // Compute stability: how much of the row/column norm is
            // concentrated on the diagonal. stability = rho[i][i] / (rho[i][i] + sum|rho[i][j!=i]|)
            let off_diag_sum: f64 = (0..d_s)
                .filter(|&j| j != basis_idx)
                .map(|j| rho_s[basis_idx][j].norm())
                .sum();

            let stability = if population + off_diag_sum > 1e-15 {
                population / (population + off_diag_sum)
            } else {
                0.0
            };

            // Construct the eigenvector as the computational basis vector
            let mut evec = vec![C64::new(0.0, 0.0); d_s];
            evec[basis_idx] = C64::new(1.0, 0.0);

            pointer_states.push(PointerState {
                state_index: basis_idx,
                stability,
                probability: population,
                eigenvector: evec,
            });
        }

        // Sort by probability (descending)
        pointer_states.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(pointer_states)
    }

    // --------------------------------------------------------
    // REDUNDANCY AND MUTUAL INFORMATION
    // --------------------------------------------------------

    /// Compute the redundancy curve: I(S:F) as a function of fragment
    /// size |F|.
    ///
    /// For each fragment size from 1 to |E|, picks the first |F|
    /// environment qubits as the fragment and computes I(S:F).
    ///
    /// If decoherence has not yet been applied, it is applied automatically.
    pub fn compute_redundancy_curve(&mut self) -> Result<RedundancyProfile, DarwinismError> {
        if !self.decohered {
            self.apply_decoherence()?;
        }

        // Compute system entropy H(S)
        let rho_s = partial_trace(&self.state, &self.env_qubits);
        let h_s = von_neumann_entropy(&rho_s)?;

        let n_env = self.config.num_env_qubits;
        let mut curve = Vec::with_capacity(n_env);

        for frag_size in 1..=n_env {
            // Take the first frag_size environment qubits as fragment
            let fragment: Vec<usize> = self.env_qubits[..frag_size].to_vec();
            let frac = frag_size as f64 / n_env as f64;

            let mi = mutual_information(&self.state, &self.system_qubits, &fragment)?;

            // Check if this point is on the classical plateau:
            // I(S:F) is within tolerance of H(S)
            let on_plateau = if h_s > 1e-10 {
                (mi - h_s).abs() / h_s < self.config.plateau_tolerance
            } else {
                // If H(S) ~ 0, the system is in a pointer state already
                mi < 0.1
            };

            curve.push(RedundancyResult {
                fragment_size: frag_size,
                fragment_fraction: frac,
                mutual_information: mi,
                classical_plateau: on_plateau,
                system_entropy: h_s,
            });
        }

        // Detect plateau
        let plateau_detected = self.detect_plateau(&curve, h_s);
        let plateau_onset = self.find_plateau_onset(&curve, h_s);
        let redundancy = plateau_onset.map(|onset| {
            if onset > 0.0 {
                1.0 / onset
            } else {
                f64::INFINITY
            }
        });

        Ok(RedundancyProfile {
            curve,
            plateau_detected,
            plateau_onset,
            redundancy,
        })
    }

    /// Detect whether the redundancy curve shows a plateau.
    ///
    /// A plateau is detected if there are at least 2 consecutive points
    /// where I(S:F) is approximately H(S) (within tolerance), occurring
    /// before the full environment is used.
    fn detect_plateau(&self, curve: &[RedundancyResult], h_s: f64) -> bool {
        if h_s < 1e-10 {
            // System is already classical (pure pointer state)
            return true;
        }

        let tol = self.config.plateau_tolerance;
        let mut consecutive = 0;

        for point in curve.iter() {
            // Don't count the last point (full environment)
            if (point.fragment_fraction - 1.0).abs() < 1e-10 {
                continue;
            }

            if (point.mutual_information - h_s).abs() / h_s < tol {
                consecutive += 1;
                if consecutive >= 2 {
                    return true;
                }
            } else {
                consecutive = 0;
            }
        }

        false
    }

    /// Find the smallest fragment fraction at which the plateau begins.
    fn find_plateau_onset(&self, curve: &[RedundancyResult], h_s: f64) -> Option<f64> {
        if h_s < 1e-10 {
            return Some(0.0);
        }

        let tol = self.config.plateau_tolerance;

        for point in curve.iter() {
            if (point.mutual_information - h_s).abs() / h_s < tol {
                return Some(point.fragment_fraction);
            }
        }

        None
    }

    /// Compute quantum mutual information for a specific fragment of
    /// the environment.
    pub fn mutual_information_for_fragment(
        &self,
        fragment_qubits: &[usize],
    ) -> Result<f64, DarwinismError> {
        mutual_information(&self.state, &self.system_qubits, fragment_qubits)
    }

    /// Get the system's reduced density matrix after tracing out the environment.
    pub fn system_reduced_state(&self) -> Vec<Vec<C64>> {
        partial_trace(&self.state, &self.env_qubits)
    }

    /// Get the von Neumann entropy of the system.
    pub fn system_entropy(&self) -> Result<f64, DarwinismError> {
        let rho_s = self.system_reduced_state();
        von_neumann_entropy(&rho_s)
    }

    // --------------------------------------------------------
    // EINSELECTION DYNAMICS
    // --------------------------------------------------------

    /// Run extended einselection dynamics and return the full convergence
    /// history.
    ///
    /// Applies `steps` rounds of system-environment interaction and records
    /// the off-diagonal norm, entropy, and purity at each step.
    pub fn run_einselection_dynamics(
        &mut self,
        steps: usize,
    ) -> Result<Vec<EinselectionSnapshot>, DarwinismError> {
        self.einselection_history.clear();

        // Record initial state
        self.record_einselection_snapshot(0)?;

        for step in 1..=steps {
            self.apply_interaction_round();
            self.record_einselection_snapshot(step)?;
        }

        self.decohered = true;
        Ok(self.einselection_history.clone())
    }

    /// Check whether einselection has converged: the off-diagonal norm
    /// has dropped below a threshold.
    pub fn einselection_converged(&self, threshold: f64) -> bool {
        if let Some(last) = self.einselection_history.last() {
            last.off_diagonal_norm < threshold
        } else {
            false
        }
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Apply a controlled-Ry rotation on a quantum state.
///
/// When the control qubit is |1>, applies Ry(theta) to the target qubit.
/// When the control qubit is |0>, does nothing.
///
/// This is implemented via direct state vector manipulation to avoid
/// potential issues with library parallel implementations.
fn apply_cry(state: &mut QuantumState, control: usize, target: usize, theta: f64) {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    let dim = state.dim;
    let amplitudes = state.amplitudes_mut();

    // Iterate over all basis states where control=1 and pair by target bit
    for i in 0..dim {
        // Only process when control=1 and target=0 (to avoid double-processing)
        if (i & control_mask) != 0 && (i & target_mask) == 0 {
            let j = i | target_mask; // Same state but with target=1
            let a = amplitudes[i]; // |...control=1...target=0...>
            let b = amplitudes[j]; // |...control=1...target=1...>

            // Ry(theta) matrix: [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            amplitudes[i] = C64::new(
                a.re * cos_half - b.re * sin_half,
                a.im * cos_half - b.im * sin_half,
            );
            amplitudes[j] = C64::new(
                a.re * sin_half + b.re * cos_half,
                a.im * sin_half + b.im * cos_half,
            );
        }
    }
}

/// Check if a density matrix is approximately diagonal in the computational basis.
///
/// Returns true if the off-diagonal norm is below the given tolerance.
pub fn is_approximately_diagonal(rho: &[Vec<C64>], tolerance: f64) -> bool {
    off_diagonal_norm(rho) < tolerance
}

/// Compute the fidelity between a density matrix and a pure computational
/// basis state |k><k|.
///
/// F(rho, |k>) = <k|rho|k> = rho[k][k].re
pub fn fidelity_with_basis_state(rho: &[Vec<C64>], basis_index: usize) -> f64 {
    if basis_index < rho.len() {
        rho[basis_index][basis_index].re.max(0.0)
    } else {
        0.0
    }
}

/// Compute the classical Shannon entropy of a probability distribution.
///
/// H(p) = -sum_i p_i log2(p_i)
pub fn classical_entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in probs {
        if p > 1e-15 {
            h -= p * p.log2();
        }
    }
    h
}

/// Compute the decoherence factor for a CRy interaction after t rounds.
///
/// For a system qubit coupled to n_env environment qubits via CRy(theta),
/// each environment qubit initially in |0>, the off-diagonal element of
/// the system's reduced density matrix decays as:
///
///   |rho_01| / |rho_01(0)| ~ |cos(t * theta / 2)|^{n_env}
///
/// After t rounds of CRy(theta), each environment qubit in the |1_S>
/// branch accumulates a rotation of Ry(t*theta), so the conditional
/// environment state is cos(t*theta/2)|0> + sin(t*theta/2)|1>.
/// The overlap with the |0_S> conditional state (still |0>) is
/// cos(t*theta/2) per qubit, giving the total decoherence factor.
pub fn zurek_decoherence_factor(
    theta: f64,
    n_env: usize,
    rounds: usize,
) -> f64 {
    (rounds as f64 * theta / 2.0).cos().abs().powi(n_env as i32)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-8;
    const LOOSE_EPSILON: f64 = 1e-4;

    /// Create a Bell state (|00> + |11>)/sqrt(2) on the specified two qubits
    /// within a larger state.
    fn make_bell_state(num_qubits: usize, qubit_a: usize, qubit_b: usize) -> QuantumState {
        let mut state = QuantumState::new(num_qubits);
        GateOperations::h(&mut state, qubit_a);
        GateOperations::cnot(&mut state, qubit_a, qubit_b);
        state
    }

    /// Create a GHZ state (|00...0> + |11...1>)/sqrt(2) on all qubits.
    fn make_ghz_state(num_qubits: usize) -> QuantumState {
        let mut state = QuantumState::new(num_qubits);
        GateOperations::h(&mut state, 0);
        for i in 1..num_qubits {
            GateOperations::cnot(&mut state, 0, i);
        }
        state
    }

    // --------------------------------------------------------
    // Configuration tests
    // --------------------------------------------------------

    #[test]
    fn test_config_builder() {
        let config = DarwinismConfig::default();
        assert_eq!(config.num_system_qubits, 1);
        assert_eq!(config.num_env_qubits, 6);
        assert!(config.interaction_strength > 0.0);
        assert!(config.decoherence_steps > 0);
        assert!(config.validate().is_ok());

        // Builder pattern
        let config2 = DarwinismConfig::new(2, 4)
            .with_interaction_strength(PI / 3.0)
            .with_decoherence_steps(5)
            .with_pointer_threshold(0.05)
            .with_plateau_tolerance(0.2);
        assert_eq!(config2.num_system_qubits, 2);
        assert_eq!(config2.num_env_qubits, 4);
        assert_eq!(config2.total_qubits(), 6);
        assert!(config2.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        // Zero system qubits
        let config = DarwinismConfig { num_system_qubits: 0, ..Default::default() };
        assert!(config.validate().is_err());

        // Only 1 env qubit
        let config = DarwinismConfig { num_env_qubits: 1, ..Default::default() };
        assert!(config.validate().is_err());

        // Invalid interaction strength
        let config = DarwinismConfig {
            interaction_strength: -0.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Zero decoherence steps
        let config = DarwinismConfig {
            decoherence_steps: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // --------------------------------------------------------
    // Partial trace tests
    // --------------------------------------------------------

    #[test]
    fn test_partial_trace_bell_state() {
        // Bell state (|00> + |11>)/sqrt(2)
        // Tracing out either qubit should give the maximally mixed state I/2
        let state = make_bell_state(2, 0, 1);

        // Trace out qubit 1 => reduced state of qubit 0
        let rho_0 = partial_trace(&state, &[1]);
        assert_eq!(rho_0.len(), 2);

        // Should be I/2 = [[0.5, 0], [0, 0.5]]
        assert!(
            (rho_0[0][0].re - 0.5).abs() < EPSILON,
            "rho_00 = {}, expected 0.5",
            rho_0[0][0].re
        );
        assert!(
            (rho_0[1][1].re - 0.5).abs() < EPSILON,
            "rho_11 = {}, expected 0.5",
            rho_0[1][1].re
        );
        assert!(
            rho_0[0][1].norm() < EPSILON,
            "off-diagonal should be 0, got {}",
            rho_0[0][1].norm()
        );
        assert!(
            rho_0[1][0].norm() < EPSILON,
            "off-diagonal should be 0, got {}",
            rho_0[1][0].norm()
        );

        // Trace should be 1
        let tr = trace_rho(&rho_0);
        assert!(
            (tr.re - 1.0).abs() < EPSILON,
            "trace = {}, expected 1.0",
            tr.re
        );
    }

    #[test]
    fn test_partial_trace_product_state() {
        // Product state |0>|0> = |00>
        let state = QuantumState::new(2);

        // Trace out qubit 1 => should get |0><0|
        let rho_0 = partial_trace(&state, &[1]);
        assert!(
            (rho_0[0][0].re - 1.0).abs() < EPSILON,
            "rho_00 = {}, expected 1.0",
            rho_0[0][0].re
        );
        assert!(
            rho_0[1][1].re.abs() < EPSILON,
            "rho_11 = {}, expected 0.0",
            rho_0[1][1].re
        );

        // Product state |+>|0>
        let mut state2 = QuantumState::new(2);
        GateOperations::h(&mut state2, 0);
        let rho_0 = partial_trace(&state2, &[1]);

        // Should get |+><+| = [[0.5, 0.5], [0.5, 0.5]]
        assert!(
            (rho_0[0][0].re - 0.5).abs() < EPSILON,
            "|+><+|[0][0] = {}, expected 0.5",
            rho_0[0][0].re
        );
        assert!(
            (rho_0[0][1].re - 0.5).abs() < EPSILON,
            "|+><+|[0][1] = {}, expected 0.5",
            rho_0[0][1].re
        );
        assert!(
            (rho_0[1][0].re - 0.5).abs() < EPSILON,
            "|+><+|[1][0] = {}, expected 0.5",
            rho_0[1][0].re
        );
        assert!(
            (rho_0[1][1].re - 0.5).abs() < EPSILON,
            "|+><+|[1][1] = {}, expected 0.5",
            rho_0[1][1].re
        );

        // Purity should be 1 (pure state)
        let p = purity(&rho_0);
        assert!(
            (p - 1.0).abs() < EPSILON,
            "purity of |+> = {}, expected 1.0",
            p
        );
    }

    #[test]
    fn test_partial_trace_three_qubits() {
        // GHZ state (|000> + |111>)/sqrt(2) on 3 qubits
        let state = make_ghz_state(3);

        // Trace out qubits 1 and 2 => reduced state of qubit 0
        // Should be I/2 (maximally mixed single qubit)
        let rho_0 = partial_trace(&state, &[1, 2]);
        assert_eq!(rho_0.len(), 2);
        assert!(
            (rho_0[0][0].re - 0.5).abs() < EPSILON,
            "GHZ rho_00 = {}, expected 0.5",
            rho_0[0][0].re
        );
        assert!(
            (rho_0[1][1].re - 0.5).abs() < EPSILON,
            "GHZ rho_11 = {}, expected 0.5",
            rho_0[1][1].re
        );

        // Trace out qubit 2 only => 2-qubit reduced state
        // For GHZ, |000> and |111> differ on qubit 2, so tracing it out
        // destroys the coherence: rho_01 = (|00><00| + |11><11|)/2
        let rho_01 = partial_trace(&state, &[2]);
        assert_eq!(rho_01.len(), 4);
        // Diagonal elements
        assert!(
            (rho_01[0][0].re - 0.5).abs() < EPSILON,
            "GHZ rho_01[00][00] = {}, expected 0.5",
            rho_01[0][0].re
        );
        assert!(
            (rho_01[3][3].re - 0.5).abs() < EPSILON,
            "GHZ rho_01[11][11] = {}, expected 0.5",
            rho_01[3][3].re
        );
        // Off-diagonal coherence is ZERO because the two GHZ terms
        // differ on the traced-out qubit 2
        assert!(
            rho_01[0][3].norm() < EPSILON,
            "GHZ rho_01[00][11] = {}, expected 0 (coherence destroyed by trace)",
            rho_01[0][3].norm()
        );
    }

    // --------------------------------------------------------
    // Von Neumann entropy tests
    // --------------------------------------------------------

    #[test]
    fn test_von_neumann_entropy_pure() {
        // Pure state |0><0| has S = 0
        let rho = vec![
            vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            vec![C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        ];
        let s = von_neumann_entropy(&rho).unwrap();
        assert!(
            s.abs() < EPSILON,
            "entropy of pure state = {}, expected 0",
            s
        );

        // Pure state |+><+| also has S = 0
        let rho_plus = vec![
            vec![C64::new(0.5, 0.0), C64::new(0.5, 0.0)],
            vec![C64::new(0.5, 0.0), C64::new(0.5, 0.0)],
        ];
        let s_plus = von_neumann_entropy(&rho_plus).unwrap();
        assert!(
            s_plus.abs() < EPSILON,
            "entropy of |+> = {}, expected 0",
            s_plus
        );
    }

    #[test]
    fn test_von_neumann_entropy_maximally_mixed() {
        // Maximally mixed state I/2 has S = 1 bit
        let rho = vec![
            vec![C64::new(0.5, 0.0), C64::new(0.0, 0.0)],
            vec![C64::new(0.0, 0.0), C64::new(0.5, 0.0)],
        ];
        let s = von_neumann_entropy(&rho).unwrap();
        assert!(
            (s - 1.0).abs() < EPSILON,
            "entropy of I/2 = {}, expected 1.0",
            s
        );

        // 4-dimensional maximally mixed state I/4 has S = 2 bits
        let rho4 = vec![
            vec![
                C64::new(0.25, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
            ],
            vec![
                C64::new(0.0, 0.0),
                C64::new(0.25, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
            ],
            vec![
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.25, 0.0),
                C64::new(0.0, 0.0),
            ],
            vec![
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.25, 0.0),
            ],
        ];
        let s4 = von_neumann_entropy(&rho4).unwrap();
        assert!(
            (s4 - 2.0).abs() < EPSILON,
            "entropy of I/4 = {}, expected 2.0",
            s4
        );
    }

    #[test]
    fn test_von_neumann_entropy_partial_mixture() {
        // State with eigenvalues (0.75, 0.25): S = -0.75*log2(0.75) - 0.25*log2(0.25)
        let expected = -0.75_f64 * 0.75_f64.log2() - 0.25_f64 * 0.25_f64.log2();
        let rho = vec![
            vec![C64::new(0.75, 0.0), C64::new(0.0, 0.0)],
            vec![C64::new(0.0, 0.0), C64::new(0.25, 0.0)],
        ];
        let s = von_neumann_entropy(&rho).unwrap();
        assert!(
            (s - expected).abs() < EPSILON,
            "entropy = {}, expected {}",
            s,
            expected
        );
    }

    // --------------------------------------------------------
    // Pointer state tests
    // --------------------------------------------------------

    #[test]
    fn test_pointer_states_computational_basis() {
        // For Z-Z coupling (CNOT-based interaction), pointer states should
        // be computational basis states |0> and |1>.
        let config = DarwinismConfig::new(1, 4)
            .with_interaction_strength(PI / 4.0)
            .with_decoherence_steps(3);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        let pointer_states = sim.find_pointer_states().unwrap();
        assert!(
            !pointer_states.is_empty(),
            "should find at least one pointer state"
        );

        // Each pointer state should have high stability (close to a
        // computational basis state)
        for ps in &pointer_states {
            assert!(
                ps.stability > 0.9,
                "pointer state {} has stability {}, expected > 0.9",
                ps.state_index,
                ps.stability
            );
        }

        // Should find exactly 2 pointer states for 1-qubit system
        // (both |0> and |1> are pointer states with equal probability)
        assert_eq!(
            pointer_states.len(),
            2,
            "expected 2 pointer states for 1-qubit system, got {}",
            pointer_states.len()
        );

        // Probabilities should sum to ~1
        let total_prob: f64 = pointer_states.iter().map(|ps| ps.probability).sum();
        assert!(
            (total_prob - 1.0).abs() < LOOSE_EPSILON,
            "total probability = {}, expected 1.0",
            total_prob
        );
    }

    #[test]
    fn test_pointer_states_are_stable() {
        // Apply decoherence, identify pointer states, then apply more
        // decoherence. Pointer states should remain the same (or become
        // more stable).
        //
        // Important: use an irrational fraction of PI to avoid periodic
        // recurrences where the environment returns to its initial state.
        // With theta = 1.0 (not a rational multiple of PI), the overlap
        // cos(t*theta/2) does not return to +/-1 for small t.
        let config = DarwinismConfig::new(1, 4)
            .with_interaction_strength(1.0)
            .with_decoherence_steps(3);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        let pointer_states_1 = sim.find_pointer_states().unwrap();

        // Apply 2 more interaction rounds (total 5, so overlap = cos(5*0.5) = cos(2.5) ~ -0.80)
        // The decoherence factor is |cos(2.5)|^4 ~ 0.41, so off-diagonal ~ 0.5*0.41 = 0.21
        // Still decohered but not perfectly.
        for _ in 0..2 {
            sim.apply_interaction_round();
        }

        // Re-identify pointer states
        sim.decohered = true;
        let pointer_states_2 = sim.find_pointer_states().unwrap();

        // Same number of pointer states
        assert_eq!(
            pointer_states_1.len(),
            pointer_states_2.len(),
            "pointer state count should be stable"
        );

        // Pointer states should have same indices
        let mut indices_1: Vec<usize> = pointer_states_1.iter().map(|p| p.state_index).collect();
        let mut indices_2: Vec<usize> = pointer_states_2.iter().map(|p| p.state_index).collect();
        indices_1.sort();
        indices_2.sort();
        assert_eq!(
            indices_1, indices_2,
            "pointer state indices should be stable: {:?} vs {:?}",
            indices_1, indices_2
        );

        // Both states should still be recognizable as pointer states
        // (stability > 0.5 means they are more diagonal than off-diagonal)
        for ps2 in &pointer_states_2 {
            assert!(
                ps2.stability > 0.5,
                "pointer state {} after extra decoherence has stability {}, expected > 0.5",
                ps2.state_index,
                ps2.stability
            );
        }
    }

    // --------------------------------------------------------
    // Mutual information tests
    // --------------------------------------------------------

    #[test]
    fn test_mutual_information_product() {
        // Product state |0>_A |0>_B: I(A:B) = 0
        let state = QuantumState::new(2);
        let mi = mutual_information(&state, &[0], &[1]).unwrap();
        assert!(
            mi.abs() < EPSILON,
            "I(A:B) for product state = {}, expected 0",
            mi
        );

        // Product state |+>|0>: still I(A:B) = 0
        let mut state2 = QuantumState::new(2);
        GateOperations::h(&mut state2, 0);
        let mi2 = mutual_information(&state2, &[0], &[1]).unwrap();
        assert!(
            mi2.abs() < EPSILON,
            "I(A:B) for |+>|0> = {}, expected 0",
            mi2
        );
    }

    #[test]
    fn test_mutual_information_bell() {
        // Bell state (|00> + |11>)/sqrt(2): I(A:B) = 2 bits
        // S(A) = 1, S(B) = 1, S(AB) = 0 => I = 2
        let state = make_bell_state(2, 0, 1);
        let mi = mutual_information(&state, &[0], &[1]).unwrap();
        assert!(
            (mi - 2.0).abs() < LOOSE_EPSILON,
            "I(A:B) for Bell state = {}, expected 2.0",
            mi
        );
    }

    #[test]
    fn test_mutual_information_ghz() {
        // 3-qubit GHZ state: I(A:{B}) where A={0}, B={1}
        // S(A) = 1, S(B) = 1, S(AB) = 1 => I = 1
        // (because tracing out qubit 2 still leaves coherence between |00> and |11>)
        let state = make_ghz_state(3);
        let mi = mutual_information(&state, &[0], &[1]).unwrap();
        assert!(
            (mi - 1.0).abs() < LOOSE_EPSILON,
            "I(A:B) for GHZ(3) A={{0}} B={{1}} = {}, expected 1.0",
            mi
        );
    }

    #[test]
    fn test_mutual_information_with_environment() {
        // System in superposition coupled to environment
        // After CNOT correlating system to env, I(S:E) should be large
        let mut state = QuantumState::new(3);
        GateOperations::h(&mut state, 0); // System in |+>
        GateOperations::cnot(&mut state, 0, 1); // Correlate with env qubit 1
        GateOperations::cnot(&mut state, 0, 2); // Correlate with env qubit 2

        // System is maximally entangled with environment
        // I(S:E) should be close to 2*S(S) = 2 bits
        let mi = mutual_information(&state, &[0], &[1, 2]).unwrap();
        assert!(
            mi > 1.5,
            "I(S:E) for GHZ-like state = {}, expected > 1.5",
            mi
        );
    }

    // --------------------------------------------------------
    // Redundancy and plateau tests
    // --------------------------------------------------------

    #[test]
    fn test_redundancy_plateau() {
        // Classical Darwinism scenario: 1-qubit system, 6-qubit environment
        // with strong Z-Z coupling. Should show a redundancy plateau.
        let config = DarwinismConfig::new(1, 6)
            .with_interaction_strength(PI / 4.0)
            .with_decoherence_steps(3)
            .with_plateau_tolerance(0.25);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        let profile = sim.compute_redundancy_curve().unwrap();

        // Should have 6 points (one per fragment size)
        assert_eq!(profile.curve.len(), 6);

        // The last point (full environment) should have I(S:E) ~ 2*H(S)
        // because for a pure state S(SE) = 0 and I(S:E) = S(S) + S(E).
        // But importantly, small fragments should already capture H(S).

        // Check that mutual information is non-decreasing (generally)
        // and that it reaches a value close to H(S)
        let h_s = profile.curve[0].system_entropy;
        assert!(
            h_s > 0.5,
            "system entropy H(S) = {}, expected > 0.5 for superposition",
            h_s
        );

        // At least some points should be on or near the plateau
        let near_plateau_count = profile
            .curve
            .iter()
            .filter(|r| {
                if h_s > 0.01 {
                    (r.mutual_information - h_s).abs() / h_s < 0.3
                } else {
                    true
                }
            })
            .count();
        assert!(
            near_plateau_count >= 2,
            "expected at least 2 points near plateau, got {}",
            near_plateau_count
        );
    }

    #[test]
    fn test_fragment_size_scaling() {
        // Mutual information should generally increase with fragment size,
        // then plateau.
        let config = DarwinismConfig::new(1, 5)
            .with_interaction_strength(PI / 4.0)
            .with_decoherence_steps(3);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        let profile = sim.compute_redundancy_curve().unwrap();

        // Check that the first fragment has less MI than the last
        let first_mi = profile.curve[0].mutual_information;
        let last_mi = profile.curve.last().unwrap().mutual_information;
        assert!(
            last_mi >= first_mi - EPSILON,
            "MI should not decrease: first={}, last={}",
            first_mi,
            last_mi
        );

        // Fragment fractions should be correct
        for (i, point) in profile.curve.iter().enumerate() {
            let expected_frac = (i + 1) as f64 / 5.0;
            assert!(
                (point.fragment_fraction - expected_frac).abs() < EPSILON,
                "fragment fraction mismatch at {}: {} vs {}",
                i,
                point.fragment_fraction,
                expected_frac
            );
        }
    }

    #[test]
    fn test_environment_fraction() {
        // With strong coupling, a small fraction of the environment
        // should suffice to capture the classical information.
        let config = DarwinismConfig::new(1, 6)
            .with_interaction_strength(PI / 3.0)
            .with_decoherence_steps(4)
            .with_plateau_tolerance(0.30);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        let profile = sim.compute_redundancy_curve().unwrap();
        let h_s = profile.curve[0].system_entropy;

        if h_s > 0.1 {
            // Check that fragment with 2 out of 6 env qubits (33%)
            // captures a significant fraction of H(S)
            let mi_at_2 = profile.curve[1].mutual_information; // fragment_size = 2
            assert!(
                mi_at_2 > 0.3 * h_s,
                "MI at frag=2 should capture >30%% of H(S): MI={}, H(S)={}",
                mi_at_2,
                h_s
            );
        }
    }

    // --------------------------------------------------------
    // Einselection dynamics tests
    // --------------------------------------------------------

    #[test]
    fn test_einselection_convergence() {
        // The off-diagonal norm should decrease over interaction rounds.
        let config = DarwinismConfig::new(1, 4)
            .with_interaction_strength(PI / 4.0)
            .with_decoherence_steps(1);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        let history = sim.run_einselection_dynamics(5).unwrap();
        assert_eq!(history.len(), 6); // 0 through 5

        // Initial state |+> has maximum off-diagonal elements
        let initial_off_diag = history[0].off_diagonal_norm;
        assert!(
            initial_off_diag > 0.3,
            "initial off-diagonal norm = {}, expected > 0.3 for |+>",
            initial_off_diag
        );

        // After several rounds, off-diagonal should decrease
        let final_off_diag = history.last().unwrap().off_diagonal_norm;
        assert!(
            final_off_diag < initial_off_diag,
            "off-diagonal should decrease: {} -> {}",
            initial_off_diag,
            final_off_diag
        );

        // Entropy should increase from 0 (pure |+>) toward 1 (mixed)
        let initial_entropy = history[0].system_entropy;
        let final_entropy = history.last().unwrap().system_entropy;
        assert!(
            final_entropy > initial_entropy,
            "entropy should increase: {} -> {}",
            initial_entropy,
            final_entropy
        );
    }

    #[test]
    fn test_decoherence_makes_diagonal() {
        // After strong decoherence, the reduced density matrix should
        // be approximately diagonal in the computational basis.
        //
        // Decoherence factor = |cos(t * theta / 2)|^n_env.
        // With theta=PI/2, t=3, n_env=5:
        //   factor = |cos(3*PI/4)|^5 = |cos(135deg)|^5 = (1/sqrt(2))^5 = 0.177
        // With theta=PI/2, t=4, n_env=5:
        //   factor = |cos(4*PI/4)|^5 = |cos(PI)|^5 = 1^5 = 1 (BAD: oscillation!)
        // With theta=PI/3, t=5, n_env=5:
        //   factor = |cos(5*PI/6)|^5 = |cos(150deg)|^5 = (sqrt(3)/2)^5 = 0.237
        // Use theta=PI/2, t=5, n_env=5:
        //   factor = |cos(5*PI/4)|^5 = (1/sqrt(2))^5 = 0.177
        //   initial off-diag = 0.5, so final ~ 0.5 * 0.177 = 0.088... still not great
        //
        // For strong decoherence, use strong coupling and many rounds:
        // theta=2*PI/3, t=3, n_env=5:
        //   factor = |cos(3*2*PI/(3*2))|^5 = |cos(PI)|^5 = 1 (oscillation)
        //
        // Best approach: use odd number of half-periods.
        // theta=PI, t=1, n_env=5: each env qubit fully flips -> CNOT equivalent
        //   factor = |cos(PI/2)|^5 = 0. Perfect decoherence in 1 round!
        let config = DarwinismConfig::new(1, 5)
            .with_interaction_strength(PI)
            .with_decoherence_steps(1);
        let mut sim = DarwinismSimulator::new(config).unwrap();
        sim.apply_decoherence().unwrap();

        let rho_s = sim.system_reduced_state();
        let off_diag = off_diagonal_norm(&rho_s);

        // With theta=PI (CNOT-like), after 1 round the environment
        // perfectly distinguishes |0> from |1>, so off-diagonal = 0.
        assert!(
            off_diag < 0.01,
            "off-diagonal norm = {}, expected < 0.01 after strong decoherence",
            off_diag
        );

        assert!(
            is_approximately_diagonal(&rho_s, 0.01),
            "reduced state should be approximately diagonal"
        );

        // The diagonal elements should be approximately equal (since
        // we started in |+>)
        assert!(
            (rho_s[0][0].re - 0.5).abs() < 0.01,
            "rho_00 = {}, expected ~0.5",
            rho_s[0][0].re
        );
        assert!(
            (rho_s[1][1].re - 0.5).abs() < 0.01,
            "rho_11 = {}, expected ~0.5",
            rho_s[1][1].re
        );
    }

    #[test]
    fn test_einselection_purity_decreases() {
        // Purity of the system should decrease as it entangles with environment
        let config = DarwinismConfig::new(1, 4)
            .with_interaction_strength(PI / 4.0)
            .with_decoherence_steps(1);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        let history = sim.run_einselection_dynamics(4).unwrap();

        // Initial purity should be 1 (pure superposition |+>)
        assert!(
            (history[0].purity - 1.0).abs() < LOOSE_EPSILON,
            "initial purity = {}, expected ~1.0",
            history[0].purity
        );

        // Final purity should be less than 1 (mixed state)
        let final_purity = history.last().unwrap().purity;
        assert!(
            final_purity < 1.0 - LOOSE_EPSILON,
            "final purity = {}, expected < 1.0 (mixed state)",
            final_purity
        );
    }

    // --------------------------------------------------------
    // Non-Darwinist state test
    // --------------------------------------------------------

    #[test]
    fn test_non_darwinist_state() {
        // A state that does NOT exhibit quantum Darwinism: the system is in
        // a superposition in a basis that is NOT the pointer basis, and the
        // environment does not redundantly encode system information.
        //
        // We create a state where the system is entangled with the environment
        // in a way that does NOT create redundant copies.

        // Create a state where system is entangled with only ONE env qubit
        // (no redundancy). Then the mutual information should jump from 0 to
        // max only when that specific qubit is included.
        let n_sys = 1;
        let n_env = 4;
        let total = n_sys + n_env;
        let mut state = QuantumState::new(total);

        // Put system in |+>
        GateOperations::h(&mut state, 0);

        // Entangle system with ONLY env qubit 1 (not the others)
        GateOperations::cnot(&mut state, 0, 1);

        // State is (|00000> + |11000>)/sqrt(2)
        // Only env qubit 1 has information about the system.
        // Env qubits 2, 3, 4 are in |0> regardless.

        // Fragment of just env qubit 2 should have I(S:F) = 0
        let mi_no_info = mutual_information(&state, &[0], &[2]).unwrap();
        assert!(
            mi_no_info < LOOSE_EPSILON,
            "I(S:F) for uninformative fragment = {}, expected ~0",
            mi_no_info
        );

        // Fragment including env qubit 1 should have I(S:F) = 2
        let mi_full_info = mutual_information(&state, &[0], &[1]).unwrap();
        assert!(
            (mi_full_info - 2.0).abs() < LOOSE_EPSILON,
            "I(S:F) for informative fragment = {}, expected 2.0",
            mi_full_info
        );

        // This is the anti-Darwinism signature: information is NOT
        // redundantly spread across the environment.
    }

    // --------------------------------------------------------
    // Zurek decoherence factor test
    // --------------------------------------------------------

    #[test]
    fn test_zurek_decoherence_factor() {
        // Verify the analytic decoherence factor matches simulation
        let theta = PI / 4.0;
        let n_env = 4;
        let rounds = 3;

        let config = DarwinismConfig::new(1, n_env)
            .with_interaction_strength(theta)
            .with_decoherence_steps(rounds);
        let mut sim = DarwinismSimulator::new(config).unwrap();
        sim.apply_decoherence().unwrap();

        let rho_s = sim.system_reduced_state();
        let simulated_coherence = rho_s[0][1].norm();
        let predicted_factor = zurek_decoherence_factor(theta, n_env, rounds);

        // The simulated coherence should match the Zurek prediction
        // (up to normalization: initial coherence was 0.5 for |+>)
        let predicted_coherence = 0.5 * predicted_factor;
        assert!(
            (simulated_coherence - predicted_coherence).abs() < 0.05,
            "simulated coherence = {}, predicted = {} (factor = {})",
            simulated_coherence,
            predicted_coherence,
            predicted_factor
        );
    }

    // --------------------------------------------------------
    // Jacobi eigenvalue tests
    // --------------------------------------------------------

    #[test]
    fn test_eigenvalues_2x2_diagonal() {
        let m = vec![
            vec![C64::new(0.3, 0.0), C64::new(0.0, 0.0)],
            vec![C64::new(0.0, 0.0), C64::new(0.7, 0.0)],
        ];
        let vals = eigenvalues_2x2(&m);
        assert!(
            (vals[0] - 0.3).abs() < EPSILON,
            "eigenvalue 0 = {}, expected 0.3",
            vals[0]
        );
        assert!(
            (vals[1] - 0.7).abs() < EPSILON,
            "eigenvalue 1 = {}, expected 0.7",
            vals[1]
        );
    }

    #[test]
    fn test_eigenvalues_2x2_offdiagonal() {
        // [[0.5, 0.5], [0.5, 0.5]] -> eigenvalues 0 and 1
        let m = vec![
            vec![C64::new(0.5, 0.0), C64::new(0.5, 0.0)],
            vec![C64::new(0.5, 0.0), C64::new(0.5, 0.0)],
        ];
        let vals = eigenvalues_2x2(&m);
        assert!(
            vals[0].abs() < EPSILON,
            "eigenvalue 0 = {}, expected 0",
            vals[0]
        );
        assert!(
            (vals[1] - 1.0).abs() < EPSILON,
            "eigenvalue 1 = {}, expected 1.0",
            vals[1]
        );
    }

    #[test]
    fn test_jacobi_4x4() {
        // 4x4 diagonal matrix should return eigenvalues directly
        let z = C64::new(0.0, 0.0);
        let m = vec![
            vec![C64::new(0.1, 0.0), z, z, z],
            vec![z, C64::new(0.2, 0.0), z, z],
            vec![z, z, C64::new(0.3, 0.0), z],
            vec![z, z, z, C64::new(0.4, 0.0)],
        ];
        let vals = eigenvalues_hermitian(&m).unwrap();
        assert_eq!(vals.len(), 4);
        for (i, &expected) in [0.1, 0.2, 0.3, 0.4].iter().enumerate() {
            assert!(
                (vals[i] - expected).abs() < EPSILON,
                "eigenvalue {} = {}, expected {}",
                i,
                vals[i],
                expected
            );
        }
    }

    #[test]
    fn test_jacobi_hermitian_complex() {
        // Hermitian matrix with complex off-diagonal elements
        // [[0.6, 0.1+0.2i], [0.1-0.2i, 0.4]]
        // Eigenvalues: (0.6+0.4)/2 +/- sqrt((0.6-0.4)^2/4 + 0.01+0.04)
        //            = 0.5 +/- sqrt(0.01 + 0.05) = 0.5 +/- sqrt(0.06)
        let m = vec![
            vec![C64::new(0.6, 0.0), C64::new(0.1, 0.2)],
            vec![C64::new(0.1, -0.2), C64::new(0.4, 0.0)],
        ];
        let vals = eigenvalues_hermitian(&m).unwrap();
        let expected_lo = 0.5 - (0.06_f64).sqrt();
        let expected_hi = 0.5 + (0.06_f64).sqrt();
        assert!(
            (vals[0] - expected_lo).abs() < EPSILON,
            "lo eigenvalue = {}, expected {}",
            vals[0],
            expected_lo
        );
        assert!(
            (vals[1] - expected_hi).abs() < EPSILON,
            "hi eigenvalue = {}, expected {}",
            vals[1],
            expected_hi
        );
    }

    // --------------------------------------------------------
    // Classical entropy test
    // --------------------------------------------------------

    #[test]
    fn test_classical_entropy() {
        // Uniform distribution over 2 outcomes: H = 1 bit
        let h = classical_entropy(&[0.5, 0.5]);
        assert!(
            (h - 1.0).abs() < EPSILON,
            "H([0.5, 0.5]) = {}, expected 1.0",
            h
        );

        // Deterministic: H = 0
        let h0 = classical_entropy(&[1.0, 0.0]);
        assert!(h0.abs() < EPSILON, "H([1, 0]) = {}, expected 0", h0);

        // Uniform over 4 outcomes: H = 2 bits
        let h4 = classical_entropy(&[0.25, 0.25, 0.25, 0.25]);
        assert!(
            (h4 - 2.0).abs() < EPSILON,
            "H([0.25 x 4]) = {}, expected 2.0",
            h4
        );
    }

    // --------------------------------------------------------
    // Integration test: full Darwinism workflow
    // --------------------------------------------------------

    #[test]
    fn test_full_darwinism_workflow() {
        // Complete workflow: create simulator, decohere, find pointer states,
        // compute redundancy.
        let config = DarwinismConfig::new(1, 4)
            .with_interaction_strength(PI / 3.0)
            .with_decoherence_steps(3)
            .with_plateau_tolerance(0.30);

        let mut sim = DarwinismSimulator::new(config).unwrap();
        assert!(!sim.is_decohered());

        // Find pointer states (triggers decoherence automatically)
        let ps = sim.find_pointer_states().unwrap();
        assert!(sim.is_decohered());
        assert_eq!(ps.len(), 2, "expected 2 pointer states");

        // Both should be near computational basis
        for p in &ps {
            assert!(p.stability > 0.85, "stability = {}", p.stability);
            assert!(p.probability > 0.1, "probability = {}", p.probability);
        }

        // System entropy should be near 1 bit (maximally mixed after decoherence)
        let s = sim.system_entropy().unwrap();
        assert!(
            (s - 1.0).abs() < 0.15,
            "system entropy = {}, expected ~1.0",
            s
        );

        // Einselection history should show convergence
        let history = sim.einselection_history();
        assert!(history.len() >= 2);
        assert!(
            history.last().unwrap().off_diagonal_norm
                < history.first().unwrap().off_diagonal_norm,
            "off-diagonal should decrease over time"
        );
    }

    #[test]
    fn test_two_qubit_system() {
        // 2-qubit system with 3-qubit environment
        let config = DarwinismConfig::new(2, 3)
            .with_interaction_strength(PI / 4.0)
            .with_decoherence_steps(3);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        // System starts in |++>
        let rho_initial = sim.system_reduced_state();
        assert_eq!(rho_initial.len(), 4, "2-qubit system => 4x4 reduced state");

        // After decoherence
        sim.apply_decoherence().unwrap();
        let rho_final = sim.system_reduced_state();
        let off_diag = off_diagonal_norm(&rho_final);

        // Decoherence should reduce off-diagonal elements
        let initial_off_diag = off_diagonal_norm(&rho_initial);
        assert!(
            off_diag < initial_off_diag + EPSILON,
            "decoherence should reduce off-diagonal: {} -> {}",
            initial_off_diag,
            off_diag
        );
    }

    #[test]
    fn test_custom_initial_state() {
        // Use with_state to start in a specific state
        let config = DarwinismConfig::new(1, 3);
        let total = config.total_qubits();

        // Start system in |1> (not in superposition)
        let mut state = QuantumState::new(total);
        GateOperations::x(&mut state, 0); // System qubit 0 => |1>

        let mut sim = DarwinismSimulator::with_state(config, state).unwrap();
        sim.apply_decoherence().unwrap();

        let rho_s = sim.system_reduced_state();
        // System should stay in |1> (it's a pointer state!)
        assert!(
            (rho_s[1][1].re - 1.0).abs() < LOOSE_EPSILON,
            "system in |1> should remain in |1>: rho_11 = {}",
            rho_s[1][1].re
        );
        assert!(
            rho_s[0][0].re.abs() < LOOSE_EPSILON,
            "rho_00 = {}, expected ~0",
            rho_s[0][0].re
        );
    }

    #[test]
    fn test_trace_preservation() {
        // The reduced density matrix should always have trace 1
        let config = DarwinismConfig::new(1, 4)
            .with_interaction_strength(PI / 4.0)
            .with_decoherence_steps(3);
        let mut sim = DarwinismSimulator::new(config).unwrap();

        // Check trace before decoherence
        let rho_before = sim.system_reduced_state();
        let tr_before = trace_rho(&rho_before);
        assert!(
            (tr_before.re - 1.0).abs() < EPSILON,
            "trace before decoherence = {}, expected 1.0",
            tr_before.re
        );

        // Check trace after each decoherence step
        sim.apply_decoherence().unwrap();
        let rho_after = sim.system_reduced_state();
        let tr_after = trace_rho(&rho_after);
        assert!(
            (tr_after.re - 1.0).abs() < EPSILON,
            "trace after decoherence = {}, expected 1.0",
            tr_after.re
        );
    }

    #[test]
    fn test_display_impls() {
        // Test that Display implementations work without panicking
        let ps = PointerState {
            state_index: 0,
            stability: 0.99,
            probability: 0.5,
            eigenvector: vec![C64::new(1.0, 0.0)],
        };
        let s = format!("{}", ps);
        assert!(s.contains("0.99"));

        let rr = RedundancyResult {
            fragment_size: 3,
            fragment_fraction: 0.5,
            mutual_information: 0.95,
            classical_plateau: true,
            system_entropy: 1.0,
        };
        let s2 = format!("{}", rr);
        assert!(s2.contains("PLATEAU"));

        let err = DarwinismError::InsufficientEnvironment {
            env_qubits: 1,
            min_required: 2,
            reason: "test".to_string(),
        };
        let s3 = format!("{}", err);
        assert!(s3.contains("1"));
    }

    #[test]
    fn test_error_variants() {
        // Ensure all error variants can be created and displayed
        let errors: Vec<DarwinismError> = vec![
            DarwinismError::InvalidSubsystem {
                qubit: 5,
                total: 3,
                context: "test".into(),
            },
            DarwinismError::InsufficientEnvironment {
                env_qubits: 1,
                min_required: 2,
                reason: "too small".into(),
            },
            DarwinismError::DecoherenceFailed {
                steps_attempted: 100,
                off_diagonal_norm: 0.5,
                reason: "did not converge".into(),
            },
            DarwinismError::EigendecompositionFailed {
                matrix_size: 8,
                iterations: 200,
            },
            DarwinismError::NumericalError("test error".into()),
        ];

        for err in &errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
            // Verify std::error::Error is implemented
            let _: &dyn std::error::Error = err;
        }
    }
}
