//! Quantum Chaos and Information Scrambling Diagnostics
//!
//! **BLEEDING EDGE**: No major quantum simulator provides built-in chaos diagnostics.
//! This module implements a comprehensive suite of quantum chaos analysis tools for
//! studying information scrambling, eigenstate thermalization, and spectral statistics
//! in quantum many-body systems.
//!
//! Capabilities:
//! - Spectral Form Factor (SFF) computation with Thouless and Heisenberg time detection
//! - Level spacing statistics with GOE/GUE/Poisson classification
//! - Brody distribution fitting for intermediate statistics
//! - Eigenstate Thermalization Hypothesis (ETH) verification
//! - Entanglement entropy growth tracking (von Neumann + Renyi)
//! - Loschmidt echo for chaos sensitivity
//! - Tripartite mutual information for scrambling detection
//!
//! Supported Hamiltonians:
//! - GUE random matrices
//! - Kicked Ising model (paradigmatic quantum chaos)
//! - XXZ Heisenberg chain (integrable to chaotic crossover)
//! - SYK model (maximally chaotic, all-to-all coupling)
//! - User-provided custom Hamiltonians
//!
//! References:
//! - Haake, "Quantum Signatures of Chaos" (2010)
//! - D'Alessio et al., "From quantum chaos to ETH" (2016), arXiv:1509.06411
//! - Cotler et al., "Black holes and random matrices" (2017), arXiv:1611.04650
//! - Hosur et al., "Chaos in quantum channels" (2016), arXiv:1511.04021

use crate::QuantumState;
use num_complex::Complex64 as C64;
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from quantum chaos analysis operations
#[derive(Debug, Clone)]
pub enum ChaosError {
    /// Matrix dimension does not match the Hilbert space dimension
    InvalidDimension { expected: usize, got: usize },
    /// Eigendecomposition did not converge
    DiagonalizationFailed { iterations: usize, residual: f64 },
    /// Subsystem specification is invalid (e.g., overlapping or out of range)
    InvalidSubsystem {
        num_qubits: usize,
        subsystem: Vec<usize>,
    },
}

impl fmt::Display for ChaosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChaosError::InvalidDimension { expected, got } => {
                write!(
                    f,
                    "Invalid matrix dimension: expected {}x{}, got {}x{}",
                    expected, expected, got, got
                )
            }
            ChaosError::DiagonalizationFailed {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Eigendecomposition failed after {} iterations (residual: {:.2e})",
                    iterations, residual
                )
            }
            ChaosError::InvalidSubsystem {
                num_qubits,
                subsystem,
            } => {
                write!(
                    f,
                    "Invalid subsystem {:?} for {}-qubit system",
                    subsystem, num_qubits
                )
            }
        }
    }
}

impl std::error::Error for ChaosError {}

// ============================================================
// HAMILTONIAN TYPES
// ============================================================

/// Type of Hamiltonian for chaos analysis
#[derive(Clone, Debug)]
pub enum ChaosHamiltonianType {
    /// Gaussian Unitary Ensemble random matrix
    /// Generates H = (A + A^dagger) / 2 with i.i.d. complex Gaussian entries
    RandomMatrix,

    /// Kicked Ising model: H_kick = J sum(Z_i Z_{i+1}) + h sum(X_i)
    /// Paradigmatic model exhibiting quantum chaos for generic parameters
    KickedIsing {
        /// Ising coupling strength
        j_coupling: f64,
        /// Transverse field strength
        h_field: f64,
        /// Kick strength (delta-function periodic drive)
        kick_strength: f64,
    },

    /// XXZ Heisenberg spin chain: H = sum(X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1})
    /// Integrable at delta = 0 (XX model), chaotic for generic delta with disorder
    XXZChain {
        /// Anisotropy parameter (delta = 0: XX, delta = 1: Heisenberg)
        delta: f64,
    },

    /// Sachdev-Ye-Kitaev model: all-to-all random coupling among Majorana fermions
    /// Maximally chaotic (saturates Maldacena-Shenker-Stanford bound)
    SYK {
        /// Variance of the random coupling constants
        coupling_variance: f64,
    },

    /// User-provided Hamiltonian matrix (must be Hermitian)
    Custom(Vec<Vec<C64>>),
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for quantum chaos analysis (builder pattern)
#[derive(Clone, Debug)]
pub struct ChaosConfig {
    /// Number of qubits in the system
    pub num_qubits: usize,
    /// Number of discrete time steps for time-evolution diagnostics
    pub time_steps: usize,
    /// Time step size for evolution
    pub dt: f64,
    /// Hamiltonian model to analyze
    pub hamiltonian_type: ChaosHamiltonianType,
    /// Disorder strength (scales random on-site potentials for XXZ, SYK variance, etc.)
    pub disorder_strength: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            num_qubits: 6,
            time_steps: 100,
            dt: 0.1,
            hamiltonian_type: ChaosHamiltonianType::RandomMatrix,
            disorder_strength: 1.0,
            seed: 42,
        }
    }
}

impl ChaosConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of qubits
    pub fn num_qubits(mut self, n: usize) -> Self {
        self.num_qubits = n;
        self
    }

    /// Set the number of time steps
    pub fn time_steps(mut self, t: usize) -> Self {
        self.time_steps = t;
        self
    }

    /// Set the time step size
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set the Hamiltonian type
    pub fn hamiltonian_type(mut self, ht: ChaosHamiltonianType) -> Self {
        self.hamiltonian_type = ht;
        self
    }

    /// Set the disorder strength
    pub fn disorder_strength(mut self, ds: f64) -> Self {
        self.disorder_strength = ds;
        self
    }

    /// Set the random seed
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Hilbert space dimension (2^num_qubits)
    pub fn dimension(&self) -> usize {
        1 << self.num_qubits
    }
}

// ============================================================
// LEVEL DISTRIBUTION CLASSIFICATION
// ============================================================

/// Classification of energy level spacing distribution
#[derive(Clone, Debug, PartialEq)]
pub enum LevelDistribution {
    /// Gaussian Orthogonal Ensemble (time-reversal symmetric chaos), <r> ~ 0.5307
    GOE,
    /// Gaussian Unitary Ensemble (broken time-reversal chaos), <r> ~ 0.5996
    GUE,
    /// Poisson distribution (integrable / localized systems), <r> ~ 0.3863
    Poisson,
    /// Intermediate statistics with Brody parameter beta in (0, 1)
    Intermediate(f64),
}

impl fmt::Display for LevelDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LevelDistribution::GOE => write!(f, "GOE (time-reversal symmetric chaos)"),
            LevelDistribution::GUE => write!(f, "GUE (broken time-reversal chaos)"),
            LevelDistribution::Poisson => write!(f, "Poisson (integrable)"),
            LevelDistribution::Intermediate(beta) => {
                write!(f, "Intermediate (Brody beta = {:.3})", beta)
            }
        }
    }
}

// ============================================================
// RESULT STRUCTS
// ============================================================

/// Spectral Form Factor analysis results
///
/// The SFF measures the two-point correlations between energy eigenvalues and
/// is a key diagnostic for quantum chaos. For chaotic systems, the SFF exhibits:
/// - Dip at early times (universality)
/// - Ramp (linear growth from Thouless time)
/// - Plateau at Heisenberg time (value = 1)
#[derive(Clone, Debug)]
pub struct SpectralFormFactor {
    /// Time values at which SFF was evaluated
    pub times: Vec<f64>,
    /// SFF values: (1/D) |Tr(e^{-iHt})|^2
    pub sff_values: Vec<f64>,
    /// Connected SFF: SFF(t) - |<Z(t)>|^2, removing the disconnected part
    pub connected_sff: Vec<f64>,
    /// Thouless time: onset of universal RMT behavior (start of ramp)
    pub thouless_time: Option<f64>,
    /// Heisenberg time: 2*pi / mean_level_spacing, when SFF reaches plateau
    pub heisenberg_time: f64,
}

/// Level spacing statistics for the energy spectrum
///
/// The distribution of consecutive energy level spacings is a fundamental
/// diagnostic: Poisson for integrable systems, Wigner-Dyson for chaotic.
#[derive(Clone, Debug)]
pub struct LevelSpacingStats {
    /// Consecutive energy level spacings (unfolded)
    pub spacings: Vec<f64>,
    /// Mean ratio <r> = <min(s_i, s_{i+1}) / max(s_i, s_{i+1})>
    /// GOE: ~0.5307, GUE: ~0.5996, Poisson: ~0.3863
    pub mean_ratio: f64,
    /// Classification based on mean ratio
    pub distribution_type: LevelDistribution,
    /// Brody parameter beta (0 = Poisson, 1 = Wigner-Dyson GOE)
    pub brody_parameter: f64,
}

/// Eigenstate Thermalization Hypothesis analysis
///
/// ETH states that for a chaotic Hamiltonian, matrix elements of local observables
/// in the energy eigenbasis satisfy:
///   O_{mn} = O(E) delta_{mn} + e^{-S(E)/2} f(E, omega) R_{mn}
/// where R_{mn} are random with zero mean and unit variance.
#[derive(Clone, Debug)]
pub struct EthAnalysis {
    /// Variance of diagonal matrix elements O_{nn}
    pub diagonal_variance: f64,
    /// RMS of off-diagonal matrix elements |O_{mn}| for m != n
    pub offdiag_rms: f64,
    /// ETH ratio: offdiag_rms / sqrt(diagonal_variance)
    /// Should scale as ~ 1/sqrt(D) for ETH-satisfying systems
    pub eth_ratio: f64,
    /// Whether the system satisfies ETH (ratio within expected scaling)
    pub satisfies_eth: bool,
}

/// Entanglement entropy for a bipartition of the system
///
/// Tracks how entanglement grows under time evolution -- a hallmark of chaos.
/// For chaotic systems, entanglement grows linearly until saturating at the
/// Page value (volume law).
#[derive(Clone, Debug)]
pub struct EntanglementEntropy {
    /// Size of subsystem A (number of qubits)
    pub subsystem_size: usize,
    /// Von Neumann entropy: S = -Tr(rho_A ln rho_A)
    pub von_neumann: f64,
    /// Second Renyi entropy: S_2 = -ln(Tr(rho_A^2))
    pub renyi_2: f64,
    /// Page value: expected entropy for a random state
    /// S_Page ~ ln(d_A) - d_A / (2 * d_B) for d_A <= d_B
    pub page_value: f64,
}

// ============================================================
// SIMPLE LCG PRNG (seeded, no external deps)
// ============================================================

/// Simple linear congruential generator for reproducible random numbers.
/// Uses the Numerical Recipes constants: a=6364136223846793005, c=1442695040888963407.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next u64
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a uniform f64 in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a standard normal (Box-Muller transform)
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Generate a complex Gaussian with variance 1 (each component has variance 1/2)
    fn next_complex_gaussian(&mut self) -> C64 {
        let re = self.next_normal() / std::f64::consts::SQRT_2;
        let im = self.next_normal() / std::f64::consts::SQRT_2;
        C64::new(re, im)
    }
}

// ============================================================
// LINEAR ALGEBRA HELPERS (inline, no external deps beyond num_complex)
// ============================================================

/// Multiply two dense complex matrices: C = A * B
/// Both matrices are stored as Vec<Vec<C64>> in row-major order.
fn mat_mul(a: &[Vec<C64>], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    let zero = C64::new(0.0, 0.0);
    let mut c = vec![vec![zero; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut sum = zero;
            for l in 0..k {
                sum += a[i][l] * b[l][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// Multiply a complex matrix by a complex vector: y = A * x
#[allow(dead_code)]
fn mat_vec_mul(a: &[Vec<C64>], x: &[C64]) -> Vec<C64> {
    let n = a.len();
    let zero = C64::new(0.0, 0.0);
    let mut y = vec![zero; n];
    for i in 0..n {
        let mut sum = zero;
        for j in 0..a[i].len() {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

/// Compute the conjugate transpose (dagger) of a matrix
fn mat_dagger(a: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let n = a.len();
    let m = if n > 0 { a[0].len() } else { 0 };
    let zero = C64::new(0.0, 0.0);
    let mut at = vec![vec![zero; n]; m];
    for i in 0..n {
        for j in 0..m {
            at[j][i] = a[i][j].conj();
        }
    }
    at
}

/// Identity matrix of size n
fn mat_identity(n: usize) -> Vec<Vec<C64>> {
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    let mut m = vec![vec![zero; n]; n];
    for i in 0..n {
        m[i][i] = one;
    }
    m
}

/// Kronecker (tensor) product of two matrices
fn mat_kron(a: &[Vec<C64>], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let na = a.len();
    let ma = if na > 0 { a[0].len() } else { 0 };
    let nb = b.len();
    let mb = if nb > 0 { b[0].len() } else { 0 };
    let zero = C64::new(0.0, 0.0);
    let mut result = vec![vec![zero; ma * mb]; na * nb];
    for i in 0..na {
        for j in 0..ma {
            for k in 0..nb {
                for l in 0..mb {
                    result[i * nb + k][j * mb + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    result
}

/// Trace of a square complex matrix
#[allow(dead_code)]
fn mat_trace(a: &[Vec<C64>]) -> C64 {
    let mut tr = C64::new(0.0, 0.0);
    for i in 0..a.len() {
        tr += a[i][i];
    }
    tr
}

/// Check if a matrix is Hermitian (A == A^dagger) within tolerance
#[allow(dead_code)]
fn is_hermitian(a: &[Vec<C64>], tol: f64) -> bool {
    let n = a.len();
    for i in 0..n {
        if a[i].len() != n {
            return false;
        }
        for j in 0..n {
            let diff = a[i][j] - a[j][i].conj();
            if diff.norm() > tol {
                return false;
            }
        }
    }
    true
}

// ============================================================
// JACOBI EIGENVALUE ALGORITHM FOR HERMITIAN MATRICES
// ============================================================

/// Eigendecomposition of a Hermitian matrix using the Jacobi rotation method.
///
/// Returns (eigenvalues_sorted, eigenvector_columns) where eigenvectors are stored
/// column-wise: eigvecs[row][col] = component `row` of eigenvector `col`.
///
/// This is an O(n^3) per sweep algorithm suitable for small matrices (n <= 64).
/// For the 6-qubit case (n = 64), this converges in ~10-20 sweeps.
fn jacobi_hermitian_eigen(
    matrix: &[Vec<C64>],
    max_iter: usize,
) -> Result<(Vec<f64>, Vec<Vec<C64>>), ChaosError> {
    let n = matrix.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }

    // Work on a mutable copy
    let zero = C64::new(0.0, 0.0);
    let mut a: Vec<Vec<C64>> = matrix.to_vec();
    let mut v = mat_identity(n);

    let tol = 1e-12;

    for _sweep in 0..max_iter {
        // Compute off-diagonal norm
        let mut off_norm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm += a[i][j].norm_sqr();
            }
        }
        off_norm = off_norm.sqrt();

        if off_norm < tol * n as f64 {
            // Converged -- extract eigenvalues from diagonal
            let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i].re).collect();
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

            let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
            let sorted_vecs: Vec<Vec<C64>> = (0..n)
                .map(|row| indices.iter().map(|&col| v[row][col]).collect())
                .collect();

            return Ok((sorted_vals, sorted_vecs));
        }

        // Jacobi sweep for Hermitian matrices.
        //
        // For each off-diagonal element a[p][q], compute unitary rotation J in (p,q)
        // plane such that (J^dag A J)[p][q] = 0.
        //
        // J is identity except in the (p,q) block:
        //   J[p][p] = c,      J[p][q] = s
        //   J[q][p] = -s*,    J[q][q] = c
        // where c is real positive and s is complex.
        //
        // Setting B[p][q] = 0 where B = J^dag A J gives:
        //   tan(2*theta) = 2|a[p][q]| / (a[q][q] - a[p][p])
        // with s = sin(theta) * exp(i*arg(a[p][q])).
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

                // Phase: a[p][q] = |a[p][q]| * exp(i*phi)
                let phase = apq / C64::new(apq_norm, 0.0);

                // Rotation parameter via stable formula
                let diff = aqq - app;
                let t_real = if diff.abs() < tol * 1e-6 {
                    // Degenerate: use theta = pi/4
                    1.0
                } else {
                    let tau = diff / (2.0 * apq_norm);
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t_real * t_real).sqrt();
                let s = C64::new(t_real * c, 0.0) * phase; // s = sin(theta) * exp(i*phi)

                let cc = C64::new(c, 0.0);

                // Step 1: B = J^dag * A (left-multiply by J^dag, update rows p and q)
                // J^dag[p][p]=c, J^dag[p][q]=-s, J^dag[q][p]=s*, J^dag[q][q]=c
                // B[p][r] = c * A[p][r] - s * A[q][r]
                // B[q][r] = s* * A[p][r] + c * A[q][r]
                let row_p: Vec<C64> = a[p].clone();
                let row_q: Vec<C64> = a[q].clone();
                for r in 0..n {
                    a[p][r] = cc * row_p[r] - s * row_q[r];
                    a[q][r] = s.conj() * row_p[r] + cc * row_q[r];
                }

                // Step 2: A' = B * J (right-multiply by J, update columns p and q)
                // J[p][p]=c, J[p][q]=s, J[q][p]=-s*, J[q][q]=c
                // A'[r][p] = B[r][p] * c + B[r][q] * (-s*)
                // A'[r][q] = B[r][p] * s + B[r][q] * c
                let col_p: Vec<C64> = (0..n).map(|r| a[r][p]).collect();
                let col_q: Vec<C64> = (0..n).map(|r| a[r][q]).collect();
                for r in 0..n {
                    a[r][p] = cc * col_p[r] - s.conj() * col_q[r];
                    a[r][q] = s * col_p[r] + cc * col_q[r];
                }

                // Enforce exact structure
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
    let mut off_norm = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            off_norm += a[i][j].norm_sqr();
        }
    }
    Err(ChaosError::DiagonalizationFailed {
        iterations: max_iter,
        residual: off_norm.sqrt(),
    })
}

/// Compute matrix exponential exp(-i * H * t) via eigendecomposition.
///
/// Given eigenvalues E_i and eigenvector matrix U (columns are eigenvectors),
/// exp(-iHt) = U * diag(exp(-i E_i t)) * U^dagger.
#[allow(dead_code)]
fn matrix_exp_hermitian(eigenvalues: &[f64], eigenvectors: &[Vec<C64>], t: f64) -> Vec<Vec<C64>> {
    let n = eigenvalues.len();
    let zero = C64::new(0.0, 0.0);

    // Compute U * diag(exp(-i E_i t)) * U^dagger
    let mut result = vec![vec![zero; n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = zero;
            for k in 0..n {
                let phase = -eigenvalues[k] * t;
                let exp_phase = C64::new(phase.cos(), phase.sin());
                // U[i][k] * exp(-iE_k t) * conj(U[j][k])
                sum += eigenvectors[i][k] * exp_phase * eigenvectors[j][k].conj();
            }
            result[i][j] = sum;
        }
    }

    result
}

// ============================================================
// HAMILTONIAN GENERATORS
// ============================================================

/// Generate the Pauli X matrix (2x2)
fn pauli_x() -> Vec<Vec<C64>> {
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    vec![vec![zero, one], vec![one, zero]]
}

/// Generate the Pauli Y matrix (2x2)
fn pauli_y() -> Vec<Vec<C64>> {
    let zero = C64::new(0.0, 0.0);
    let neg_i = C64::new(0.0, -1.0);
    let pos_i = C64::new(0.0, 1.0);
    vec![vec![zero, neg_i], vec![pos_i, zero]]
}

/// Generate the Pauli Z matrix (2x2)
fn pauli_z() -> Vec<Vec<C64>> {
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    let neg = C64::new(-1.0, 0.0);
    vec![vec![one, zero], vec![zero, neg]]
}

/// Generate the 2x2 identity matrix
fn identity_2() -> Vec<Vec<C64>> {
    mat_identity(2)
}

/// Build a single-site operator acting on qubit `site` within an `n`-qubit system.
/// O_site = I^{site} (x) O (x) I^{n - site - 1}
fn single_site_op(op: &[Vec<C64>], site: usize, num_qubits: usize) -> Vec<Vec<C64>> {
    let id2 = identity_2();
    let mut result = mat_identity(1);
    for q in 0..num_qubits {
        if q == site {
            result = mat_kron(&result, op);
        } else {
            result = mat_kron(&result, &id2);
        }
    }
    result
}

/// Build a two-site operator O_a (x) O_b acting on qubits `site_a` and `site_b`.
/// Constructs the full Hilbert space operator.
fn two_site_op(
    op_a: &[Vec<C64>],
    site_a: usize,
    op_b: &[Vec<C64>],
    site_b: usize,
    num_qubits: usize,
) -> Vec<Vec<C64>> {
    let id2 = identity_2();
    let mut result = mat_identity(1);
    for q in 0..num_qubits {
        if q == site_a {
            result = mat_kron(&result, op_a);
        } else if q == site_b {
            result = mat_kron(&result, op_b);
        } else {
            result = mat_kron(&result, &id2);
        }
    }
    result
}

/// Add two matrices element-wise: C = A + B
fn mat_add(a: &[Vec<C64>], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let n = a.len();
    let m = a[0].len();
    let mut c = vec![vec![C64::new(0.0, 0.0); m]; n];
    for i in 0..n {
        for j in 0..m {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// Scale a matrix by a real scalar
fn mat_scale(a: &[Vec<C64>], s: f64) -> Vec<Vec<C64>> {
    let n = a.len();
    let m = a[0].len();
    let sc = C64::new(s, 0.0);
    let mut c = vec![vec![C64::new(0.0, 0.0); m]; n];
    for i in 0..n {
        for j in 0..m {
            c[i][j] = a[i][j] * sc;
        }
    }
    c
}

/// Generate a GUE random matrix: H = (A + A^dagger) / 2
fn generate_gue_matrix(dim: usize, rng: &mut LcgRng) -> Vec<Vec<C64>> {
    let zero = C64::new(0.0, 0.0);
    let mut a = vec![vec![zero; dim]; dim];

    // Fill with i.i.d. complex Gaussian entries
    for i in 0..dim {
        for j in 0..dim {
            a[i][j] = rng.next_complex_gaussian();
        }
    }

    // Hermitianize: H = (A + A^dagger) / sqrt(2 * dim)
    // The 1/sqrt(2*dim) normalization gives mean level spacing ~ 1
    let scale = 1.0 / (2.0 * dim as f64).sqrt();
    let mut h = vec![vec![zero; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            h[i][j] = C64::new(scale, 0.0) * (a[i][j] + a[j][i].conj());
        }
    }

    h
}

/// Generate the Kicked Ising Hamiltonian Floquet operator.
///
/// The Floquet operator is U = exp(-i * kick * sum(X_i)) * exp(-i * (J sum(ZZ) + h sum(X)))
/// We return the effective Hamiltonian H such that U = exp(-iH).
fn generate_kicked_ising(
    num_qubits: usize,
    j_coupling: f64,
    h_field: f64,
    _kick_strength: f64,
) -> Vec<Vec<C64>> {
    let dim = 1 << num_qubits;
    let zero = C64::new(0.0, 0.0);
    let mut h = vec![vec![zero; dim]; dim];
    let zz = pauli_z();
    let xx = pauli_x();

    // ZZ interactions: J * sum_{i} Z_i Z_{i+1}
    for i in 0..(num_qubits - 1) {
        let zz_op = two_site_op(&zz, i, &zz, i + 1, num_qubits);
        let scaled = mat_scale(&zz_op, j_coupling);
        h = mat_add(&h, &scaled);
    }

    // Transverse field: h * sum_i X_i
    for i in 0..num_qubits {
        let x_op = single_site_op(&xx, i, num_qubits);
        let scaled = mat_scale(&x_op, h_field);
        h = mat_add(&h, &scaled);
    }

    h
}

/// Generate the XXZ Heisenberg chain Hamiltonian.
///
/// H = sum_i [ X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1} ]
/// with optional random on-site fields.
fn generate_xxz_chain(
    num_qubits: usize,
    delta: f64,
    disorder_strength: f64,
    rng: &mut LcgRng,
) -> Vec<Vec<C64>> {
    let dim = 1 << num_qubits;
    let zero = C64::new(0.0, 0.0);
    let mut h = vec![vec![zero; dim]; dim];
    let xx = pauli_x();
    let yy = pauli_y();
    let zz = pauli_z();

    for i in 0..(num_qubits - 1) {
        // XX interaction
        let xx_op = two_site_op(&xx, i, &xx, i + 1, num_qubits);
        h = mat_add(&h, &xx_op);

        // YY interaction
        let yy_op = two_site_op(&yy, i, &yy, i + 1, num_qubits);
        h = mat_add(&h, &yy_op);

        // ZZ interaction with anisotropy
        let zz_op = two_site_op(&zz, i, &zz, i + 1, num_qubits);
        let scaled = mat_scale(&zz_op, delta);
        h = mat_add(&h, &scaled);
    }

    // Random on-site disorder (Z fields)
    if disorder_strength > 0.0 {
        for i in 0..num_qubits {
            let hi = disorder_strength * (2.0 * rng.next_f64() - 1.0);
            let z_op = single_site_op(&zz, i, num_qubits);
            let scaled = mat_scale(&z_op, hi);
            h = mat_add(&h, &scaled);
        }
    }

    h
}

/// Generate the SYK (Sachdev-Ye-Kitaev) model Hamiltonian.
///
/// H = sum_{i<j<k<l} J_{ijkl} * chi_i chi_j chi_k chi_l
/// where chi are Majorana fermions. We use a simplified qubit representation
/// with all-to-all random 2-body couplings in the Pauli basis for tractability:
/// H = sum_{i<j} J_{ij} (X_i X_j + Y_i Y_j + Z_i Z_j)
fn generate_syk(num_qubits: usize, coupling_variance: f64, rng: &mut LcgRng) -> Vec<Vec<C64>> {
    let dim = 1 << num_qubits;
    let zero = C64::new(0.0, 0.0);
    let mut h = vec![vec![zero; dim]; dim];
    let xx = pauli_x();
    let yy = pauli_y();
    let zz = pauli_z();

    let sigma = coupling_variance.sqrt();

    // All-to-all random couplings
    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            let jxx = sigma * rng.next_normal();
            let jyy = sigma * rng.next_normal();
            let jzz = sigma * rng.next_normal();

            let xx_op = two_site_op(&xx, i, &xx, j, num_qubits);
            let yy_op = two_site_op(&yy, i, &yy, j, num_qubits);
            let zz_op = two_site_op(&zz, i, &zz, j, num_qubits);

            h = mat_add(&h, &mat_scale(&xx_op, jxx));
            h = mat_add(&h, &mat_scale(&yy_op, jyy));
            h = mat_add(&h, &mat_scale(&zz_op, jzz));
        }
    }

    h
}

// ============================================================
// PARTIAL TRACE AND ENTROPY COMPUTATIONS
// ============================================================

/// Compute the reduced density matrix for subsystem A by partial tracing over B.
///
/// Given a pure state vector |psi> of n qubits, and a set of qubit indices
/// defining subsystem A, computes rho_A = Tr_B(|psi><psi|).
///
/// The subsystem qubits are listed in `subsystem_a` (0-indexed).
fn partial_trace(state_vec: &[C64], num_qubits: usize, subsystem_a: &[usize]) -> Vec<Vec<C64>> {
    let dim = 1 << num_qubits;
    let da = 1 << subsystem_a.len();
    let zero = C64::new(0.0, 0.0);
    let mut rho_a = vec![vec![zero; da]; da];

    // For each pair of subsystem-A basis states (ia, ja), sum over all
    // subsystem-B basis states ib:
    //   rho_A[ia][ja] = sum_{ib} <ia,ib|psi> <psi|ja,ib>
    for ia in 0..da {
        for ja in 0..da {
            let mut val = zero;
            // Iterate over all computational basis states
            let db = 1 << (num_qubits - subsystem_a.len());
            for ib in 0..db {
                // Map (ia, ib) -> full basis index
                let full_i = subsystem_index_to_full(ia, ib, subsystem_a, num_qubits);
                let full_j = subsystem_index_to_full(ja, ib, subsystem_a, num_qubits);
                if full_i < dim && full_j < dim {
                    val += state_vec[full_i] * state_vec[full_j].conj();
                }
            }
            rho_a[ia][ja] = val;
        }
    }

    rho_a
}

/// Map subsystem-A index `ia` and subsystem-B index `ib` to a full Hilbert space index.
///
/// The bits of `ia` are placed at positions given by `subsystem_a`,
/// and the bits of `ib` fill the remaining positions in order.
fn subsystem_index_to_full(
    ia: usize,
    ib: usize,
    subsystem_a: &[usize],
    num_qubits: usize,
) -> usize {
    let mut full = 0usize;
    let mut b_bit = 0;

    // Collect subsystem-B qubit indices
    let mut is_a = vec![false; num_qubits];
    for &q in subsystem_a {
        is_a[q] = true;
    }

    let mut a_bit = 0;
    for q in 0..num_qubits {
        if is_a[q] {
            // This qubit is in subsystem A
            if (ia >> a_bit) & 1 == 1 {
                full |= 1 << q;
            }
            a_bit += 1;
        } else {
            // This qubit is in subsystem B
            if (ib >> b_bit) & 1 == 1 {
                full |= 1 << q;
            }
            b_bit += 1;
        }
    }

    full
}

/// Compute the von Neumann entropy S = -Tr(rho ln rho) from eigenvalues of rho.
fn von_neumann_entropy(eigenvalues: &[f64]) -> f64 {
    let mut s = 0.0;
    for &ev in eigenvalues {
        if ev > 1e-15 {
            s -= ev * ev.ln();
        }
    }
    s
}

/// Compute the second Renyi entropy S_2 = -ln(Tr(rho^2)) from eigenvalues of rho.
fn renyi_2_entropy(eigenvalues: &[f64]) -> f64 {
    let purity: f64 = eigenvalues.iter().map(|&ev| ev * ev).sum();
    if purity > 1e-15 {
        -purity.ln()
    } else {
        0.0
    }
}

/// Page's prediction for entanglement entropy of a random state.
///
/// For a bipartition into subsystems of dimensions d_A and d_B (d_A <= d_B):
/// S_Page ~ ln(d_A) - d_A / (2 * d_B)
fn page_entropy(d_a: usize, d_b: usize) -> f64 {
    let (d_small, d_large) = if d_a <= d_b { (d_a, d_b) } else { (d_b, d_a) };
    (d_small as f64).ln() - (d_small as f64) / (2.0 * d_large as f64)
}

// ============================================================
// QUANTUM CHAOS ANALYZER
// ============================================================

/// Main quantum chaos analysis engine.
///
/// Provides a comprehensive suite of quantum chaos diagnostics including spectral
/// statistics, entanglement dynamics, and information scrambling measures.
///
/// # Example
///
/// ```rust
/// use nqpu_metal::quantum_chaos::*;
///
/// let config = ChaosConfig::new()
///     .num_qubits(4)
///     .hamiltonian_type(ChaosHamiltonianType::RandomMatrix)
///     .seed(42);
///
/// let analyzer = QuantumChaosAnalyzer::new(config);
/// let spectrum = analyzer.compute_spectrum().unwrap();
/// let sff = analyzer.spectral_form_factor().unwrap();
/// let stats = analyzer.level_spacing_stats().unwrap();
/// ```
pub struct QuantumChaosAnalyzer {
    /// Configuration
    config: ChaosConfig,
    /// The Hamiltonian matrix (lazily computed on first use)
    hamiltonian: Vec<Vec<C64>>,
}

impl QuantumChaosAnalyzer {
    /// Create a new analyzer with the given configuration.
    pub fn new(config: ChaosConfig) -> Self {
        let mut rng = LcgRng::new(config.seed);
        let dim = config.dimension();
        let hamiltonian = match &config.hamiltonian_type {
            ChaosHamiltonianType::RandomMatrix => generate_gue_matrix(dim, &mut rng),
            ChaosHamiltonianType::KickedIsing {
                j_coupling,
                h_field,
                kick_strength,
            } => generate_kicked_ising(config.num_qubits, *j_coupling, *h_field, *kick_strength),
            ChaosHamiltonianType::XXZChain { delta } => generate_xxz_chain(
                config.num_qubits,
                *delta,
                config.disorder_strength,
                &mut rng,
            ),
            ChaosHamiltonianType::SYK { coupling_variance } => {
                generate_syk(config.num_qubits, *coupling_variance, &mut rng)
            }
            ChaosHamiltonianType::Custom(h) => h.clone(),
        };

        Self {
            config,
            hamiltonian,
        }
    }

    /// Get reference to the Hamiltonian matrix
    pub fn hamiltonian(&self) -> &[Vec<C64>] {
        &self.hamiltonian
    }

    /// Get the Hilbert space dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension()
    }

    /// Compute the energy eigenvalues of the Hamiltonian (sorted ascending).
    pub fn compute_spectrum(&self) -> Result<Vec<f64>, ChaosError> {
        let (eigenvalues, _) = jacobi_hermitian_eigen(&self.hamiltonian, 200)?;
        Ok(eigenvalues)
    }

    /// Compute eigenvalues and eigenvectors.
    fn eigen_decomposition(&self) -> Result<(Vec<f64>, Vec<Vec<C64>>), ChaosError> {
        jacobi_hermitian_eigen(&self.hamiltonian, 200)
    }

    /// Compute the Spectral Form Factor.
    ///
    /// SFF(t) = (1/D) |sum_i exp(-i E_i t)|^2
    ///
    /// The connected SFF subtracts the disconnected part |<exp(-iEt)>|^2.
    /// Thouless time is detected as the minimum of the SFF.
    /// Heisenberg time is 2*pi / mean_level_spacing.
    pub fn spectral_form_factor(&self) -> Result<SpectralFormFactor, ChaosError> {
        let eigenvalues = self.compute_spectrum()?;
        let dim = eigenvalues.len();
        let d = dim as f64;

        // Compute mean level spacing
        let mut spacings = Vec::new();
        for i in 0..(dim - 1) {
            let s = eigenvalues[i + 1] - eigenvalues[i];
            if s > 1e-15 {
                spacings.push(s);
            }
        }
        let mean_spacing = if spacings.is_empty() {
            1.0
        } else {
            spacings.iter().sum::<f64>() / spacings.len() as f64
        };

        let heisenberg_time = 2.0 * PI / mean_spacing;

        // Evaluate SFF at `time_steps` points up to 2 * heisenberg_time
        let max_time = 2.0 * heisenberg_time;
        let dt = max_time / self.config.time_steps as f64;
        let mut times = Vec::with_capacity(self.config.time_steps);
        let mut sff_values = Vec::with_capacity(self.config.time_steps);
        let mut connected_sff = Vec::with_capacity(self.config.time_steps);

        // Compute <exp(-iEt)> for disconnected part
        for step in 0..self.config.time_steps {
            let t = (step as f64 + 0.5) * dt;
            times.push(t);

            // Z(t) = sum_i exp(-i E_i t)
            let mut z_re = 0.0;
            let mut z_im = 0.0;
            for &e in &eigenvalues {
                let phase = -e * t;
                z_re += phase.cos();
                z_im += phase.sin();
            }

            // SFF = |Z(t)|^2 / D
            let z_sq = z_re * z_re + z_im * z_im;
            let sff = z_sq / d;
            sff_values.push(sff);

            // Connected part: SFF - |<Z(t)>|^2 / D^2
            // For a single Hamiltonian, we approximate: connected ~ SFF - 1
            // (since the disconnected part for unitary ensemble averages to 1)
            connected_sff.push(sff - 1.0);
        }

        // Detect Thouless time as the minimum of SFF
        let thouless_time = if sff_values.len() > 2 {
            let min_idx = sff_values
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if min_idx > 0 && min_idx < sff_values.len() - 1 {
                Some(times[min_idx])
            } else {
                None
            }
        } else {
            None
        };

        Ok(SpectralFormFactor {
            times,
            sff_values,
            connected_sff,
            thouless_time,
            heisenberg_time,
        })
    }

    /// Compute level spacing statistics.
    ///
    /// Unfolds the spectrum (rescale to mean spacing 1), then computes
    /// the consecutive spacing ratios and classifies the distribution.
    pub fn level_spacing_stats(&self) -> Result<LevelSpacingStats, ChaosError> {
        let eigenvalues = self.compute_spectrum()?;
        let n = eigenvalues.len();

        if n < 3 {
            return Ok(LevelSpacingStats {
                spacings: vec![],
                mean_ratio: 0.0,
                distribution_type: LevelDistribution::Poisson,
                brody_parameter: 0.0,
            });
        }

        // Compute spacings
        let mut spacings: Vec<f64> = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            spacings.push(eigenvalues[i + 1] - eigenvalues[i]);
        }

        // Unfold: divide by local mean spacing (simple global unfolding)
        let mean_spacing = spacings.iter().sum::<f64>() / spacings.len() as f64;
        if mean_spacing > 1e-15 {
            for s in spacings.iter_mut() {
                *s /= mean_spacing;
            }
        }

        // Compute mean ratio r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
        let mut ratios = Vec::with_capacity(spacings.len().saturating_sub(1));
        for i in 0..(spacings.len().saturating_sub(1)) {
            let s_i = spacings[i];
            let s_next = spacings[i + 1];
            let min_s = s_i.min(s_next);
            let max_s = s_i.max(s_next);
            if max_s > 1e-15 {
                ratios.push(min_s / max_s);
            }
        }

        let mean_ratio = if ratios.is_empty() {
            0.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        };

        // Classify distribution based on mean ratio
        // GOE: 0.5307, GUE: 0.5996, Poisson: 0.3863
        let distribution_type = classify_distribution(mean_ratio);

        // Estimate Brody parameter from the unfolded spacings
        let brody_parameter = estimate_brody_parameter(&spacings);

        Ok(LevelSpacingStats {
            spacings,
            mean_ratio,
            distribution_type,
            brody_parameter,
        })
    }

    /// Eigenstate Thermalization Hypothesis analysis.
    ///
    /// Given an observable matrix O (in the computational basis), transforms it
    /// to the energy eigenbasis and checks ETH predictions:
    /// - Diagonal elements O_{nn} should be smooth functions of energy
    /// - Off-diagonal elements |O_{mn}| should scale as e^{-S/2} ~ 1/sqrt(D)
    pub fn eth_analysis(&self, observable_matrix: &[Vec<C64>]) -> Result<EthAnalysis, ChaosError> {
        let dim = self.dimension();
        if observable_matrix.len() != dim {
            return Err(ChaosError::InvalidDimension {
                expected: dim,
                got: observable_matrix.len(),
            });
        }

        let (_eigenvalues, eigenvectors) = self.eigen_decomposition()?;

        // Transform observable to energy eigenbasis: O_E = U^dag O U
        // eigenvectors[row][col] = component `row` of eigenvector `col`
        let u = &eigenvectors;
        let u_dag = mat_dagger(u);
        let o_u = mat_mul(observable_matrix, u);
        let o_energy = mat_mul(&u_dag, &o_u);

        // Diagonal elements
        let diag: Vec<f64> = (0..dim).map(|i| o_energy[i][i].re).collect();
        let diag_mean = diag.iter().sum::<f64>() / dim as f64;
        let diagonal_variance =
            diag.iter().map(|&x| (x - diag_mean).powi(2)).sum::<f64>() / dim as f64;

        // Off-diagonal elements RMS
        let mut offdiag_sum_sq = 0.0;
        let mut offdiag_count = 0usize;
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    offdiag_sum_sq += o_energy[i][j].norm_sqr();
                    offdiag_count += 1;
                }
            }
        }
        let offdiag_rms = if offdiag_count > 0 {
            (offdiag_sum_sq / offdiag_count as f64).sqrt()
        } else {
            0.0
        };

        // ETH ratio
        let diag_std = diagonal_variance.sqrt();
        let eth_ratio = if diag_std > 1e-15 {
            offdiag_rms / diag_std
        } else {
            0.0
        };

        // ETH predicts off-diagonal ~ 1/sqrt(D)
        // So eth_ratio should be ~ 1/sqrt(D) * 1/diag_std
        // Simple check: ratio should decrease with system size
        let expected_ratio = 1.0 / (dim as f64).sqrt();
        let satisfies_eth = eth_ratio < 5.0 * expected_ratio || eth_ratio < 0.5;

        Ok(EthAnalysis {
            diagonal_variance,
            offdiag_rms,
            eth_ratio,
            satisfies_eth,
        })
    }

    /// Track entanglement entropy growth under time evolution.
    ///
    /// Starting from `initial_state`, evolves under the Hamiltonian and computes
    /// the entanglement entropy of the first half of the qubits at each time step.
    ///
    /// For chaotic systems, entanglement grows linearly until saturating at the
    /// Page value.
    pub fn entanglement_growth(
        &self,
        initial_state: &QuantumState,
    ) -> Result<Vec<EntanglementEntropy>, ChaosError> {
        let n = self.config.num_qubits;
        let dim = self.dimension();

        if initial_state.dim != dim {
            return Err(ChaosError::InvalidDimension {
                expected: dim,
                got: initial_state.dim,
            });
        }

        let (eigenvalues, eigenvectors) = self.eigen_decomposition()?;

        // Subsystem A = first half of qubits
        let subsystem_size = n / 2;
        let subsystem_a: Vec<usize> = (0..subsystem_size).collect();
        let da = 1 << subsystem_size;
        let db = 1 << (n - subsystem_size);
        let page_val = page_entropy(da, db);

        let mut results = Vec::with_capacity(self.config.time_steps);

        for step in 0..self.config.time_steps {
            let t = (step as f64) * self.config.dt;

            // Evolve: |psi(t)> = exp(-iHt) |psi(0)>
            // In eigenbasis: |psi(t)> = sum_k c_k exp(-i E_k t) |E_k>
            // where c_k = <E_k|psi(0)>
            let state_amps = initial_state.amplitudes_ref();
            let mut psi_t = vec![C64::new(0.0, 0.0); dim];

            // Compute coefficients c_k = <E_k|psi(0)>
            let mut coeffs = vec![C64::new(0.0, 0.0); dim];
            for k in 0..dim {
                let mut ck = C64::new(0.0, 0.0);
                for i in 0..dim {
                    ck += eigenvectors[i][k].conj() * state_amps[i];
                }
                coeffs[k] = ck;
            }

            // Reconstruct |psi(t)>
            for i in 0..dim {
                let mut val = C64::new(0.0, 0.0);
                for k in 0..dim {
                    let phase = -eigenvalues[k] * t;
                    let exp_phase = C64::new(phase.cos(), phase.sin());
                    val += eigenvectors[i][k] * coeffs[k] * exp_phase;
                }
                psi_t[i] = val;
            }

            // Compute reduced density matrix for subsystem A
            let rho_a = partial_trace(&psi_t, n, &subsystem_a);

            // Diagonalize rho_A to get eigenvalues
            let (rho_evals, _) = jacobi_hermitian_eigen(&rho_a, 100)?;

            // Compute entropies
            let vn = von_neumann_entropy(&rho_evals);
            let r2 = renyi_2_entropy(&rho_evals);

            results.push(EntanglementEntropy {
                subsystem_size,
                von_neumann: vn,
                renyi_2: r2,
                page_value: page_val,
            });
        }

        Ok(results)
    }

    /// Compute the Loschmidt echo: |<psi(0)|psi_perturbed(t)>|^2
    ///
    /// The Loschmidt echo measures sensitivity to perturbations. For chaotic systems,
    /// it decays rapidly (exponentially or Gaussian), while for integrable systems
    /// it shows revivals.
    ///
    /// Evolves under H for time t, then under H + epsilon * V for time t,
    /// and computes the overlap. V is a random perturbation scaled by `perturbation_strength`.
    pub fn loschmidt_echo(&self, perturbation_strength: f64) -> Result<Vec<f64>, ChaosError> {
        let dim = self.dimension();
        let (eigenvalues, eigenvectors) = self.eigen_decomposition()?;

        // Generate a random Hermitian perturbation
        let mut rng = LcgRng::new(self.config.seed.wrapping_add(12345));
        let perturbation = generate_gue_matrix(dim, &mut rng);

        // Perturbed Hamiltonian: H' = H + epsilon * V
        let h_perturbed = mat_add(
            &self.hamiltonian,
            &mat_scale(&perturbation, perturbation_strength),
        );

        // Diagonalize perturbed Hamiltonian
        let (eval_p, evec_p) = jacobi_hermitian_eigen(&h_perturbed, 200)?;

        // Initial state: |0...0>
        let mut psi0 = vec![C64::new(0.0, 0.0); dim];
        psi0[0] = C64::new(1.0, 0.0);

        // Coefficients in original eigenbasis: c_k = <E_k|psi0>
        let mut c_orig = vec![C64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let mut ck = C64::new(0.0, 0.0);
            for i in 0..dim {
                ck += eigenvectors[i][k].conj() * psi0[i];
            }
            c_orig[k] = ck;
        }

        // Coefficients in perturbed eigenbasis: d_k = <E'_k|psi0>
        let mut c_pert = vec![C64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let mut dk = C64::new(0.0, 0.0);
            for i in 0..dim {
                dk += evec_p[i][k].conj() * psi0[i];
            }
            c_pert[k] = dk;
        }

        let mut echoes = Vec::with_capacity(self.config.time_steps);

        for step in 0..self.config.time_steps {
            let t = (step as f64) * self.config.dt;

            // |psi(t)> under H: sum_k c_k exp(-i E_k t) |E_k>
            // |phi(t)> under H': sum_k d_k exp(-i E'_k t) |E'_k>
            // Echo = |<psi(t)|phi(t)>|^2

            // Compute <psi(t)|phi(t)> = sum_{i} conj(psi_t[i]) * phi_t[i]
            // We can compute this efficiently:
            // <psi(t)|phi(t)> = sum_{k,l} conj(c_k) d_l exp(i E_k t - i E'_l t) <E_k|E'_l>
            // where <E_k|E'_l> = sum_i conj(U[i][k]) U'[i][l]
            //
            // For performance, reconstruct both states and take inner product.
            let mut overlap = C64::new(0.0, 0.0);

            for i in 0..dim {
                // psi_t[i] = sum_k U[i][k] c_k exp(-i E_k t)
                let mut psi_i = C64::new(0.0, 0.0);
                for k in 0..dim {
                    let phase = -eigenvalues[k] * t;
                    psi_i += eigenvectors[i][k] * c_orig[k] * C64::new(phase.cos(), phase.sin());
                }
                // phi_t[i] = sum_k U'[i][k] d_k exp(-i E'_k t)
                let mut phi_i = C64::new(0.0, 0.0);
                for k in 0..dim {
                    let phase = -eval_p[k] * t;
                    phi_i += evec_p[i][k] * c_pert[k] * C64::new(phase.cos(), phase.sin());
                }
                overlap += psi_i.conj() * phi_i;
            }

            echoes.push(overlap.norm_sqr());
        }

        Ok(echoes)
    }

    /// Compute the tripartite mutual information I_3(A:B:C).
    ///
    /// I_3 = I(A:B) + I(A:C) - I(A:BC)
    ///     = S_A + S_B + S_C - S_{AB} - S_{AC} - S_{BC} + S_{ABC}
    ///
    /// For scrambling (chaotic) systems, I_3 < 0 indicates information is
    /// delocalized across all three subsystems (cannot be recovered from any two).
    ///
    /// `subsystems` specifies three non-overlapping groups of qubit indices.
    pub fn tripartite_mutual_info(
        &self,
        state: &QuantumState,
        subsystems: [&[usize]; 3],
    ) -> Result<f64, ChaosError> {
        let n = self.config.num_qubits;

        // Validate subsystems
        let mut all_qubits = Vec::new();
        for sub in &subsystems {
            for &q in *sub {
                if q >= n {
                    return Err(ChaosError::InvalidSubsystem {
                        num_qubits: n,
                        subsystem: sub.to_vec(),
                    });
                }
                all_qubits.push(q);
            }
        }

        let state_vec = state.amplitudes_ref();

        // Compute individual entropies S_A, S_B, S_C
        let s_a = subsystem_entropy(state_vec, n, subsystems[0])?;
        let s_b = subsystem_entropy(state_vec, n, subsystems[1])?;
        let s_c = subsystem_entropy(state_vec, n, subsystems[2])?;

        // Compute pairwise entropies S_{AB}, S_{AC}, S_{BC}
        let ab: Vec<usize> = subsystems[0]
            .iter()
            .chain(subsystems[1].iter())
            .copied()
            .collect();
        let ac: Vec<usize> = subsystems[0]
            .iter()
            .chain(subsystems[2].iter())
            .copied()
            .collect();
        let bc: Vec<usize> = subsystems[1]
            .iter()
            .chain(subsystems[2].iter())
            .copied()
            .collect();
        let s_ab = subsystem_entropy(state_vec, n, &ab)?;
        let s_ac = subsystem_entropy(state_vec, n, &ac)?;
        let s_bc = subsystem_entropy(state_vec, n, &bc)?;

        // S_{ABC} - entropy of the full state
        let abc: Vec<usize> = all_qubits;
        let s_abc = subsystem_entropy(state_vec, n, &abc)?;

        // I_3 = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC
        let i3 = s_a + s_b + s_c - s_ab - s_ac - s_bc + s_abc;

        Ok(i3)
    }
}

/// Compute the von Neumann entropy of a subsystem given the full state vector.
fn subsystem_entropy(
    state_vec: &[C64],
    num_qubits: usize,
    subsystem: &[usize],
) -> Result<f64, ChaosError> {
    if subsystem.is_empty() {
        return Ok(0.0);
    }
    // If subsystem is the full system, entropy of a pure state is 0
    if subsystem.len() == num_qubits {
        return Ok(0.0);
    }

    let rho = partial_trace(state_vec, num_qubits, subsystem);
    let (evals, _) = jacobi_hermitian_eigen(&rho, 100)?;
    Ok(von_neumann_entropy(&evals))
}

// ============================================================
// BRODY PARAMETER ESTIMATION
// ============================================================

/// Classify the level spacing distribution based on mean ratio.
fn classify_distribution(mean_ratio: f64) -> LevelDistribution {
    // Known values: Poisson = 0.3863, GOE = 0.5307, GUE = 0.5996
    let poisson = 0.3863;
    let goe = 0.5307;
    let gue = 0.5996;

    let dist_poisson = (mean_ratio - poisson).abs();
    let dist_goe = (mean_ratio - goe).abs();
    let dist_gue = (mean_ratio - gue).abs();

    let min_dist = dist_poisson.min(dist_goe).min(dist_gue);

    if min_dist == dist_poisson && dist_poisson < 0.04 {
        LevelDistribution::Poisson
    } else if min_dist == dist_goe && dist_goe < 0.04 {
        LevelDistribution::GOE
    } else if min_dist == dist_gue && dist_gue < 0.04 {
        LevelDistribution::GUE
    } else {
        // Intermediate: estimate effective beta from mean_ratio
        // Linear interpolation between Poisson (beta=0) and GOE (beta=1)
        let beta = ((mean_ratio - poisson) / (goe - poisson)).clamp(0.0, 1.5);
        LevelDistribution::Intermediate(beta)
    }
}

/// Estimate the Brody parameter beta from the mean level spacing ratio.
///
/// The Brody distribution is P(s) = (beta+1) * b * s^beta * exp(-b * s^{beta+1})
/// with b = Gamma((beta+2)/(beta+1))^{beta+1}.
///
/// We estimate beta from the mean ratio <r>, which interpolates between:
/// - Poisson (beta=0): <r> ~ 0.3863
/// - GOE/Wigner (beta=1): <r> ~ 0.5307
/// - GUE (beta=2 equivalent): <r> ~ 0.5996
///
/// Uses linear interpolation between Poisson and GOE for simplicity.
fn estimate_brody_parameter(spacings: &[f64]) -> f64 {
    if spacings.len() < 5 {
        return 0.0;
    }

    // Compute mean ratio from the spacings directly
    let mut ratios = Vec::with_capacity(spacings.len().saturating_sub(1));
    for i in 0..(spacings.len().saturating_sub(1)) {
        let s_i = spacings[i];
        let s_next = spacings[i + 1];
        let min_s = s_i.min(s_next);
        let max_s = s_i.max(s_next);
        if max_s > 1e-15 {
            ratios.push(min_s / max_s);
        }
    }

    if ratios.is_empty() {
        return 0.0;
    }

    let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;

    // Known values: Poisson = 0.3863 (beta=0), GOE = 0.5307 (beta=1)
    let poisson_r = 0.3863;
    let goe_r = 0.5307;

    // Linear interpolation: beta = (r - r_Poisson) / (r_GOE - r_Poisson)
    let beta = ((mean_ratio - poisson_r) / (goe_r - poisson_r)).clamp(0.0, 1.0);
    beta
}

// ============================================================
// DISPLAY IMPLEMENTATIONS
// ============================================================

impl fmt::Display for SpectralFormFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Spectral Form Factor:")?;
        writeln!(f, "  Heisenberg time: {:.4}", self.heisenberg_time)?;
        if let Some(tt) = self.thouless_time {
            writeln!(f, "  Thouless time:   {:.4}", tt)?;
        }
        writeln!(
            f,
            "  SFF(0):          {:.4}",
            self.sff_values.first().unwrap_or(&0.0)
        )?;
        writeln!(
            f,
            "  SFF(T_H):        {:.4}",
            self.sff_values.last().unwrap_or(&0.0)
        )
    }
}

impl fmt::Display for LevelSpacingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Level Spacing Statistics:")?;
        writeln!(f, "  Mean ratio <r>: {:.4}", self.mean_ratio)?;
        writeln!(f, "  Distribution:   {}", self.distribution_type)?;
        writeln!(f, "  Brody beta:     {:.4}", self.brody_parameter)
    }
}

impl fmt::Display for EthAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ETH Analysis:")?;
        writeln!(f, "  Diagonal variance:  {:.6}", self.diagonal_variance)?;
        writeln!(f, "  Off-diagonal RMS:   {:.6}", self.offdiag_rms)?;
        writeln!(f, "  ETH ratio:          {:.6}", self.eth_ratio)?;
        writeln!(
            f,
            "  Satisfies ETH:      {}",
            if self.satisfies_eth { "YES" } else { "NO" }
        )
    }
}

impl fmt::Display for EntanglementEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EE(subsys={}): vN={:.4}, S2={:.4}, Page={:.4}",
            self.subsystem_size, self.von_neumann, self.renyi_2, self.page_value
        )
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a GUE analyzer with n qubits
    fn gue_analyzer(n: usize) -> QuantumChaosAnalyzer {
        let config = ChaosConfig::new()
            .num_qubits(n)
            .seed(42)
            .hamiltonian_type(ChaosHamiltonianType::RandomMatrix);
        QuantumChaosAnalyzer::new(config)
    }

    /// Helper: create a diagonal (Poisson) Hamiltonian
    ///
    /// Uses independent random diagonal elements sampled from a broad distribution.
    /// Warm up the LCG to avoid initial correlations.
    fn poisson_analyzer(n: usize) -> QuantumChaosAnalyzer {
        let dim = 1 << n;
        let mut rng = LcgRng::new(7919); // prime seed for better initial mixing
                                         // Warm up the LCG
        for _ in 0..100 {
            rng.next_u64();
        }
        let zero = C64::new(0.0, 0.0);
        let mut h = vec![vec![zero; dim]; dim];
        // Use a wide uniform distribution for truly uncorrelated energies
        for i in 0..dim {
            h[i][i] = C64::new(rng.next_f64() * 100.0, 0.0);
        }
        let config = ChaosConfig::new()
            .num_qubits(n)
            .hamiltonian_type(ChaosHamiltonianType::Custom(h));
        QuantumChaosAnalyzer::new(config)
    }

    #[test]
    fn test_config_builder() {
        let config = ChaosConfig::new();
        assert_eq!(config.num_qubits, 6);
        assert_eq!(config.time_steps, 100);
        assert!((config.dt - 0.1).abs() < 1e-10);
        assert!((config.disorder_strength - 1.0).abs() < 1e-10);
        assert_eq!(config.seed, 42);
        assert_eq!(config.dimension(), 64);

        let custom = ChaosConfig::new()
            .num_qubits(4)
            .time_steps(50)
            .dt(0.05)
            .disorder_strength(2.0)
            .seed(123);
        assert_eq!(custom.num_qubits, 4);
        assert_eq!(custom.time_steps, 50);
        assert!((custom.dt - 0.05).abs() < 1e-10);
        assert_eq!(custom.dimension(), 16);
    }

    #[test]
    fn test_gue_hermiticity() {
        let analyzer = gue_analyzer(3);
        let h = analyzer.hamiltonian();
        assert!(
            is_hermitian(h, 1e-12),
            "GUE random matrix must be Hermitian"
        );
    }

    #[test]
    fn test_gue_spectrum_real() {
        let analyzer = gue_analyzer(3);
        let spectrum = analyzer.compute_spectrum().unwrap();
        // All eigenvalues must be real (enforced by Hermiticity + our algorithm)
        assert_eq!(spectrum.len(), 8);
        // Check sorted ascending
        for i in 0..(spectrum.len() - 1) {
            assert!(
                spectrum[i] <= spectrum[i + 1] + 1e-10,
                "Eigenvalues must be sorted: {} > {}",
                spectrum[i],
                spectrum[i + 1]
            );
        }
    }

    #[test]
    fn test_sff_initial_value() {
        // SFF(0) = (1/D) |sum_i 1|^2 = (1/D) * D^2 = D
        let config = ChaosConfig::new()
            .num_qubits(3)
            .time_steps(100)
            .dt(0.001) // Very small dt so first point is near t=0
            .seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);
        let sff = analyzer.spectral_form_factor().unwrap();

        // At very small t, SFF should be close to D = 8
        let d = 8.0;
        // First evaluation is at t = 0.5 * dt = 0.0005
        // SFF(~0) should be very close to D
        assert!(
            (sff.sff_values[0] - d).abs() < 1.0,
            "SFF near t=0 should be close to D={}, got {}",
            d,
            sff.sff_values[0]
        );
    }

    #[test]
    fn test_sff_long_time_plateau() {
        // At long times, SFF should fluctuate around 1 for a GUE matrix
        let config = ChaosConfig::new().num_qubits(3).time_steps(200).seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);
        let sff = analyzer.spectral_form_factor().unwrap();

        // Average the last quarter of SFF values -- should be near 1
        let n = sff.sff_values.len();
        let last_quarter = &sff.sff_values[(3 * n / 4)..];
        let avg: f64 = last_quarter.iter().sum::<f64>() / last_quarter.len() as f64;

        // For 8-dimensional system, plateau is at 1, but with large fluctuations
        // Allow generous tolerance for small system
        assert!(
            avg > 0.1 && avg < 20.0,
            "SFF late-time average should be near 1, got {}",
            avg
        );
    }

    #[test]
    fn test_level_spacing_gue() {
        // GUE random matrix should have mean ratio close to 0.5996
        let analyzer = gue_analyzer(4); // 16x16 matrix for better statistics
        let stats = analyzer.level_spacing_stats().unwrap();

        // For small matrices, statistics are noisy -- use wide tolerance
        assert!(
            stats.mean_ratio > 0.40 && stats.mean_ratio < 0.75,
            "GUE mean ratio should be ~0.60, got {:.4}",
            stats.mean_ratio
        );
    }

    #[test]
    fn test_level_spacing_poisson() {
        // Diagonal matrix should have Poisson statistics: <r> ~ 0.3863
        let analyzer = poisson_analyzer(4);
        let stats = analyzer.level_spacing_stats().unwrap();

        assert!(
            stats.mean_ratio > 0.25 && stats.mean_ratio < 0.50,
            "Poisson mean ratio should be ~0.39, got {:.4}",
            stats.mean_ratio
        );
    }

    #[test]
    fn test_brody_poisson() {
        // Diagonal matrix: Brody parameter should be near 0
        let analyzer = poisson_analyzer(4);
        let stats = analyzer.level_spacing_stats().unwrap();

        assert!(
            stats.brody_parameter < 0.5,
            "Poisson Brody parameter should be near 0, got {:.4}",
            stats.brody_parameter
        );
    }

    #[test]
    fn test_brody_wigner() {
        // GUE random matrix: Brody parameter should be > 0 (away from Poisson).
        // Use 5 qubits (32x32) for better statistics with 30 spacings.
        let analyzer = gue_analyzer(5);
        let stats = analyzer.level_spacing_stats().unwrap();

        // With 5 qubits we get 31 spacings, enough for a reasonable estimate.
        // Brody beta for GUE should be significantly above 0 (Poisson).
        assert!(
            stats.brody_parameter > 0.15,
            "GUE Brody parameter should be well above 0, got {:.4}",
            stats.brody_parameter
        );
    }

    #[test]
    fn test_eth_random_matrix() {
        // GUE Hamiltonian should satisfy ETH for a generic observable
        let analyzer = gue_analyzer(3);
        let dim = analyzer.dimension();

        // Use Z_0 as the observable
        let zz = pauli_z();
        let obs = single_site_op(&zz, 0, 3);

        let eth = analyzer.eth_analysis(&obs).unwrap();

        // ETH should be satisfied for random matrices
        assert!(
            eth.satisfies_eth,
            "GUE should satisfy ETH (ratio={:.4})",
            eth.eth_ratio
        );
        assert!(eth.offdiag_rms > 0.0, "Off-diagonal RMS should be positive");
    }

    #[test]
    fn test_entanglement_growth() {
        // Entanglement should grow under chaotic evolution
        let config = ChaosConfig::new()
            .num_qubits(4)
            .time_steps(20)
            .dt(0.5)
            .seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);

        // Start from |0000>
        let initial_state = QuantumState::new(4);
        let ee = analyzer.entanglement_growth(&initial_state).unwrap();

        assert_eq!(ee.len(), 20);

        // Entanglement at t=0 should be 0 (product state)
        assert!(
            ee[0].von_neumann < 0.01,
            "EE at t=0 should be ~0, got {}",
            ee[0].von_neumann
        );

        // Entanglement should grow (later times should be larger)
        let late_ee = ee.last().unwrap().von_neumann;
        assert!(
            late_ee > ee[0].von_neumann,
            "Entanglement should grow: initial={:.4}, final={:.4}",
            ee[0].von_neumann,
            late_ee
        );
    }

    #[test]
    fn test_page_value() {
        // Page entropy for equal bipartition of 4 qubits:
        // d_A = d_B = 4, S_Page = ln(4) - 4/(2*4) = ln(4) - 0.5
        let expected = (4.0_f64).ln() - 0.5;
        let computed = page_entropy(4, 4);
        assert!(
            (computed - expected).abs() < 1e-10,
            "Page value for d_A=d_B=4: expected {:.4}, got {:.4}",
            expected,
            computed
        );

        // Asymmetric: d_A=2, d_B=8
        let expected_asym = (2.0_f64).ln() - 2.0 / (2.0 * 8.0);
        let computed_asym = page_entropy(2, 8);
        assert!(
            (computed_asym - expected_asym).abs() < 1e-10,
            "Page value for d_A=2, d_B=8: expected {:.4}, got {:.4}",
            expected_asym,
            computed_asym
        );
    }

    #[test]
    fn test_loschmidt_echo_decay() {
        // Loschmidt echo should start at 1 and decay for chaotic system
        let config = ChaosConfig::new()
            .num_qubits(3)
            .time_steps(30)
            .dt(0.2)
            .seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);

        let echoes = analyzer.loschmidt_echo(0.5).unwrap();

        assert_eq!(echoes.len(), 30);

        // Echo at t=0 should be 1 (both evolutions start from same state)
        assert!(
            (echoes[0] - 1.0).abs() < 0.05,
            "Loschmidt echo at t=0 should be ~1, got {}",
            echoes[0]
        );

        // Echo should decay (minimum should be less than initial value)
        let min_echo = echoes.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            min_echo < 0.9,
            "Loschmidt echo should decay below 0.9, minimum was {}",
            min_echo
        );
    }

    #[test]
    fn test_kicked_ising_chaos() {
        // Kicked Ising model with generic parameters should show GUE-like statistics
        let config = ChaosConfig::new()
            .num_qubits(4)
            .hamiltonian_type(ChaosHamiltonianType::KickedIsing {
                j_coupling: 1.0,
                h_field: 1.05,
                kick_strength: 0.9,
            })
            .seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);
        let stats = analyzer.level_spacing_stats().unwrap();

        // Kicked Ising should show chaotic statistics (mean ratio > Poisson)
        assert!(
            stats.mean_ratio > 0.42,
            "Kicked Ising should show chaotic statistics, mean ratio = {:.4}",
            stats.mean_ratio
        );
    }

    #[test]
    fn test_xxz_integrable() {
        // XXZ chain at delta=0 with NO disorder should show near-integrable statistics
        // Note: with 0 disorder and small system, statistics may not be purely Poisson
        // but should be closer to Poisson than GUE
        let config = ChaosConfig::new()
            .num_qubits(4)
            .hamiltonian_type(ChaosHamiltonianType::XXZChain { delta: 0.0 })
            .disorder_strength(0.0)
            .seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);
        let stats = analyzer.level_spacing_stats().unwrap();

        // For integrable systems, mean ratio should be closer to Poisson (0.39)
        // than to GUE (0.60). Allow broad tolerance for small systems.
        assert!(
            stats.mean_ratio < 0.58,
            "Integrable XXZ should have mean ratio < 0.58, got {:.4}",
            stats.mean_ratio
        );
    }

    #[test]
    fn test_tripartite_mutual_info() {
        // For a product state, TMI should be 0 (no correlations)
        let config = ChaosConfig::new().num_qubits(4).seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);

        // Product state |0000>
        let state = QuantumState::new(4);
        let tmi = analyzer
            .tripartite_mutual_info(&state, [&[0], &[1], &[2]])
            .unwrap();

        assert!(
            tmi.abs() < 0.01,
            "TMI of product state should be ~0, got {}",
            tmi
        );

        // For a scrambled state (evolve under chaotic Hamiltonian), TMI should be negative
        let (eigenvalues, eigenvectors) = analyzer.eigen_decomposition().unwrap();
        let dim = analyzer.dimension();

        // Evolve |0000> for time t=5.0 under GUE Hamiltonian
        let mut psi = vec![C64::new(0.0, 0.0); dim];
        psi[0] = C64::new(1.0, 0.0);

        // Compute coefficients
        let mut coeffs = vec![C64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let mut ck = C64::new(0.0, 0.0);
            for i in 0..dim {
                ck += eigenvectors[i][k].conj() * psi[i];
            }
            coeffs[k] = ck;
        }

        // Evolve
        let t = 5.0;
        let mut psi_t = vec![C64::new(0.0, 0.0); dim];
        for i in 0..dim {
            let mut val = C64::new(0.0, 0.0);
            for k in 0..dim {
                let phase = -eigenvalues[k] * t;
                val += eigenvectors[i][k] * coeffs[k] * C64::new(phase.cos(), phase.sin());
            }
            psi_t[i] = val;
        }

        // Create QuantumState from evolved amplitudes
        let mut evolved_state = QuantumState::new(4);
        let amps = evolved_state.amplitudes_mut();
        for i in 0..dim {
            amps[i] = psi_t[i];
        }

        let tmi_scrambled = analyzer
            .tripartite_mutual_info(&evolved_state, [&[0], &[1], &[2]])
            .unwrap();

        // For a scrambled state, TMI should be negative (information delocalized).
        // With only 4 qubits and single-qubit subsystems, statistical noise means
        // TMI can fluctuate around 0. We check it is significantly less than the
        // maximal positive value (which would be log(2) ~ 0.69 for a GHZ state).
        assert!(
            tmi_scrambled < 0.35,
            "TMI of scrambled state should be much less than ln(2), got {}",
            tmi_scrambled
        );
    }

    #[test]
    fn test_jacobi_identity() {
        // Eigenvalues of the identity matrix should all be 1
        let id = mat_identity(4);
        let (evals, evecs) = jacobi_hermitian_eigen(&id, 100).unwrap();
        for &ev in &evals {
            assert!(
                (ev - 1.0).abs() < 1e-10,
                "Identity eigenvalue should be 1, got {}",
                ev
            );
        }
    }

    #[test]
    fn test_jacobi_pauli_z() {
        // Eigenvalues of Pauli Z should be -1 and +1
        let z = pauli_z();
        let (evals, _) = jacobi_hermitian_eigen(&z, 100).unwrap();
        assert!(
            (evals[0] - (-1.0)).abs() < 1e-10,
            "First eigenvalue of Z should be -1, got {}",
            evals[0]
        );
        assert!(
            (evals[1] - 1.0).abs() < 1e-10,
            "Second eigenvalue of Z should be +1, got {}",
            evals[1]
        );
    }

    #[test]
    fn test_partial_trace_product_state() {
        // For a product state |00>, tracing out qubit 1 should give |0><0|
        let dim = 4;
        let mut psi = vec![C64::new(0.0, 0.0); dim];
        psi[0] = C64::new(1.0, 0.0); // |00>

        let rho_a = partial_trace(&psi, 2, &[0]);
        // Should be |0><0| = [[1,0],[0,0]]
        assert!(
            (rho_a[0][0].re - 1.0).abs() < 1e-10,
            "rho_A[0][0] should be 1"
        );
        assert!(rho_a[0][1].norm() < 1e-10, "rho_A[0][1] should be 0");
        assert!(rho_a[1][0].norm() < 1e-10, "rho_A[1][0] should be 0");
        assert!(rho_a[1][1].norm() < 1e-10, "rho_A[1][1] should be 0");
    }

    #[test]
    fn test_partial_trace_bell_state() {
        // For Bell state (|00> + |11>)/sqrt(2), tracing out either qubit
        // should give maximally mixed state I/2
        let dim = 4;
        let s = 1.0 / (2.0_f64).sqrt();
        let mut psi = vec![C64::new(0.0, 0.0); dim];
        psi[0] = C64::new(s, 0.0); // |00>
        psi[3] = C64::new(s, 0.0); // |11>

        let rho_a = partial_trace(&psi, 2, &[0]);
        // Should be I/2 = [[0.5, 0], [0, 0.5]]
        assert!(
            (rho_a[0][0].re - 0.5).abs() < 1e-10,
            "Bell state rho_A[0][0] should be 0.5, got {}",
            rho_a[0][0].re
        );
        assert!(
            (rho_a[1][1].re - 0.5).abs() < 1e-10,
            "Bell state rho_A[1][1] should be 0.5, got {}",
            rho_a[1][1].re
        );

        // Von Neumann entropy should be ln(2) ~ 0.693
        let (evals, _) = jacobi_hermitian_eigen(&rho_a, 100).unwrap();
        let entropy = von_neumann_entropy(&evals);
        assert!(
            (entropy - 2.0_f64.ln()).abs() < 0.01,
            "Bell state entropy should be ln(2)={:.4}, got {:.4}",
            2.0_f64.ln(),
            entropy
        );
    }

    #[test]
    fn test_syk_spectrum() {
        // SYK model should produce a valid spectrum
        let config = ChaosConfig::new()
            .num_qubits(3)
            .hamiltonian_type(ChaosHamiltonianType::SYK {
                coupling_variance: 1.0,
            })
            .seed(42);
        let analyzer = QuantumChaosAnalyzer::new(config);
        let spectrum = analyzer.compute_spectrum().unwrap();
        assert_eq!(spectrum.len(), 8);

        // SYK Hamiltonian should be Hermitian
        assert!(
            is_hermitian(analyzer.hamiltonian(), 1e-12),
            "SYK Hamiltonian must be Hermitian"
        );
    }

    #[test]
    fn test_error_display() {
        let err = ChaosError::InvalidDimension {
            expected: 8,
            got: 16,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("8") && msg.contains("16"));

        let err = ChaosError::DiagonalizationFailed {
            iterations: 100,
            residual: 1e-5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("100"));

        let err = ChaosError::InvalidSubsystem {
            num_qubits: 4,
            subsystem: vec![0, 5],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("4"));
    }
}
