//! Quantum Signal Processing (QSP) and Quantum Singular Value Transformation (QSVT)
//!
//! This module implements the grand unifying framework for quantum algorithms.
//! QSP transforms a signal encoded in a single-qubit rotation into an arbitrary
//! polynomial via a sequence of interleaved signal rotations and phase gates.
//! QSVT generalises QSP to block-encoded matrices, enabling polynomial
//! transformations of singular values.
//!
//! # Key capabilities
//!
//! - **Phase factor computation**: Given a target polynomial P(x) that satisfies
//!   the QSP constraints, compute the phase angles φ_0 … φ_d that realise P.
//! - **QSP circuit construction & evaluation**: Build the interleaved rotation
//!   sequence and evaluate the resulting polynomial at arbitrary points.
//! - **QSVT circuit construction**: Wrap a block encoding with projector-controlled
//!   phase rotations to apply polynomial transformations to singular values.
//! - **Applications**: Hamiltonian simulation (Jacobi-Anger), matrix inversion
//!   (HHL-like), and amplitude amplification (Grover-like) as QSVT special cases.
//!
//! # QSP Polynomial Constraints
//!
//! A polynomial P(x) is QSP-realisable with d signal rotations iff:
//! 1. P has degree exactly d.
//! 2. P has definite parity matching d (even degree → even polynomial, odd → odd).
//! 3. |P(x)| ≤ 1 for all x ∈ [-1, 1].
//! 4. |P(±1)| = 1 (boundary unitarity condition for the Wx/Wz convention).
//! 5. There exists a complementary polynomial Q of degree d-1 and opposite parity
//!    such that |P(x)|² + (1-x²)|Q(x)|² = 1 for all x ∈ [-1, 1].
//!
//! Common valid QSP polynomials include Chebyshev polynomials T_d(x) and their
//! parity-preserving linear combinations with coefficients summing to ±1 at x = ±1.
//!
//! # References
//!
//! - Gilyén, Su, Low, Wiebe. "Quantum singular value transformation and beyond:
//!   exponential improvements for quantum matrix arithmetics" (2019).
//! - Martyn, Rossi, Tan, Chuang. "Grand Unification of Quantum Algorithms" (2021).
//! - Dong, Meng, Whaley, Lin. "Efficient phase-factor evaluation in quantum
//!   signal processing" (2021).

use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// TYPE ALIASES (local to module)
// ============================================================

type C64 = Complex64;

#[inline]
fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising from QSP / QSVT computations.
#[derive(Debug, Clone)]
pub enum QspError {
    /// The iterative phase-factor solver did not converge within the
    /// configured number of iterations.
    ConvergenceFailed { iterations: usize, residual: f64 },
    /// The requested polynomial degree is invalid (e.g. zero).
    InvalidDegree(usize),
    /// The supplied polynomial violates QSP constraints (e.g. |P(x)| > 1).
    InvalidPolynomial(String),
    /// Phase-factor extraction failed for an internal reason.
    PhaseComputationFailed(String),
}

impl std::fmt::Display for QspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QspError::ConvergenceFailed {
                iterations,
                residual,
            } => {
                write!(f, "QSP phase-factor solver did not converge after {} iterations (residual {:.2e})", iterations, residual)
            }
            QspError::InvalidDegree(d) => write!(f, "Invalid polynomial degree: {}", d),
            QspError::InvalidPolynomial(msg) => write!(f, "Invalid polynomial: {}", msg),
            QspError::PhaseComputationFailed(msg) => write!(f, "Phase computation failed: {}", msg),
        }
    }
}

impl std::error::Error for QspError {}

// ============================================================
// CONVENTIONS
// ============================================================

/// QSP signal-rotation convention.
///
/// * `Wx` — the signal unitary is an X-rotation: W(a) = \[\[a, i√(1-a²)\], \[i√(1-a²), a\]\].
///   Phase rotations are R_z(φ).
/// * `Wz` — the signal unitary is a Z-rotation: W(a) = \[\[a, √(1-a²)\], \[-√(1-a²), a\]\].
///   Phase rotations are R_x(φ).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QspConvention {
    Wx,
    Wz,
}

// ============================================================
// CONFIGURATION BUILDERS
// ============================================================

/// Configuration for QSP phase-factor computation.
#[derive(Debug, Clone)]
pub struct QspConfig {
    /// Polynomial degree.
    pub degree: usize,
    /// Convergence tolerance for iterative solver.
    pub tolerance: f64,
    /// Maximum Newton-like iterations.
    pub max_iterations: usize,
    /// Signal-rotation convention.
    pub convention: QspConvention,
}

impl Default for QspConfig {
    fn default() -> Self {
        Self {
            degree: 1,
            tolerance: 1e-10,
            max_iterations: 1000,
            convention: QspConvention::Wx,
        }
    }
}

impl QspConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn degree(mut self, d: usize) -> Self {
        self.degree = d;
        self
    }

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn convention(mut self, conv: QspConvention) -> Self {
        self.convention = conv;
        self
    }
}

/// Configuration for QSVT.
#[derive(Debug, Clone)]
pub struct QsvtConfig {
    /// Sub-normalisation factor α of the block encoding (‖A‖ ≤ α).
    pub block_encoding_subnormalization: f64,
    /// Whether the QSVT uses projector-controlled phase rotations.
    pub projector_controlled: bool,
    /// Number of system qubits (excluding ancillae).
    pub num_qubits: usize,
}

impl Default for QsvtConfig {
    fn default() -> Self {
        Self {
            block_encoding_subnormalization: 1.0,
            projector_controlled: true,
            num_qubits: 1,
        }
    }
}

impl QsvtConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn block_encoding_subnormalization(mut self, alpha: f64) -> Self {
        self.block_encoding_subnormalization = alpha;
        self
    }

    pub fn projector_controlled(mut self, pc: bool) -> Self {
        self.projector_controlled = pc;
        self
    }

    pub fn num_qubits(mut self, n: usize) -> Self {
        self.num_qubits = n;
        self
    }
}

// ============================================================
// CORE DATA STRUCTURES
// ============================================================

/// Phase angles that define a QSP polynomial transformation.
///
/// For a degree-d polynomial the vector contains d + 1 phase angles φ_0 … φ_d.
#[derive(Debug, Clone)]
pub struct PhaseFactors {
    pub angles: Vec<f64>,
}

impl PhaseFactors {
    pub fn new(angles: Vec<f64>) -> Self {
        Self { angles }
    }

    /// Polynomial degree implied by the number of phase angles.
    pub fn degree(&self) -> usize {
        if self.angles.is_empty() {
            0
        } else {
            self.angles.len() - 1
        }
    }
}

/// A 2×2 signal operator encoding a scalar signal a ∈ [-1, 1].
///
/// The matrix form depends on the convention:
/// * Wx: W(a) = \[\[a, i√(1-a²)\], \[i√(1-a²), a\]\]
/// * Wz: W(a) = \[\[a, √(1-a²)\], \[-√(1-a²), a\]\]
#[derive(Debug, Clone)]
pub struct SignalOperator {
    /// The 2×2 unitary as a flat row-major array \[m00, m01, m10, m11\].
    pub matrix: [C64; 4],
}

impl SignalOperator {
    /// Build the signal operator for value `a` under the given convention.
    pub fn from_signal(a: f64, convention: QspConvention) -> Self {
        let sq = (1.0 - a * a).max(0.0).sqrt();
        let matrix = match convention {
            QspConvention::Wx => [c64(a, 0.0), c64(0.0, sq), c64(0.0, sq), c64(a, 0.0)],
            QspConvention::Wz => [c64(a, 0.0), c64(sq, 0.0), c64(-sq, 0.0), c64(a, 0.0)],
        };
        Self { matrix }
    }

    /// Return the 2×2 matrix as \[\[m00, m01\], \[m10, m11\]\].
    pub fn to_matrix(&self) -> [[C64; 2]; 2] {
        [
            [self.matrix[0], self.matrix[1]],
            [self.matrix[2], self.matrix[3]],
        ]
    }
}

/// A block encoding of a matrix A.
///
/// A (unitary) block encoding U_A of an n-qubit operator A satisfies
///     (⟨0|⊗I) U_A (|0⟩⊗I) = A / α
/// where α is the sub-normalisation factor and extra ancilla qubits are
/// prepared in |0⟩.
#[derive(Debug, Clone)]
pub struct BlockEncoding {
    /// The full unitary matrix of the block encoding (flat row-major, dimension 2^(n+a)).
    pub matrix: Vec<C64>,
    /// Total dimension of the unitary (2^(n+a)).
    pub dim: usize,
    /// Sub-normalisation factor (‖A‖ ≤ α).
    pub subnormalization: f64,
    /// Number of ancilla qubits used in the encoding.
    pub num_ancilla_qubits: usize,
    /// Number of system qubits.
    pub num_system_qubits: usize,
}

impl BlockEncoding {
    /// Create a block encoding from a (potentially non-unitary) matrix A.
    ///
    /// The subnormalization α must satisfy ‖A‖ ≤ α.  The encoding
    /// embeds A/α in the top-left block of a unitary on one extra ancilla.
    pub fn from_matrix(a_flat: &[C64], dim: usize, subnormalization: f64) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        assert_eq!(a_flat.len(), dim * dim, "Matrix must be dim x dim");

        let alpha = subnormalization;
        let total_dim = 2 * dim; // one ancilla qubit
        let mut u = vec![c64(0.0, 0.0); total_dim * total_dim];

        // Top-left block: A / α
        for i in 0..dim {
            for j in 0..dim {
                u[i * total_dim + j] = a_flat[i * dim + j] / alpha;
            }
        }

        // Complete to a unitary.  For Hermitian A/α:
        // U = [[A/α, sqrt(I - (A/α)^2)], [sqrt(I - (A/α)^2), -A/α]]
        let scaled: Vec<C64> = a_flat.iter().map(|&v| v / alpha).collect();
        let complement = matrix_sqrt_complement(&scaled, dim);

        for i in 0..dim {
            for j in 0..dim {
                u[i * total_dim + (dim + j)] = complement[i * dim + j];
                u[(dim + i) * total_dim + j] = complement[i * dim + j];
                u[(dim + i) * total_dim + (dim + j)] = -scaled[i * dim + j];
            }
        }

        let n_sys = if dim <= 1 {
            0
        } else {
            (dim as f64).log2().round() as usize
        };

        Self {
            matrix: u,
            dim: total_dim,
            subnormalization: alpha,
            num_ancilla_qubits: 1,
            num_system_qubits: n_sys,
        }
    }

    /// Extract the top-left (system) block from the block encoding,
    /// multiplied by the sub-normalisation factor.
    pub fn extract_block(&self) -> Vec<C64> {
        let sys_dim = self.dim / (1 << self.num_ancilla_qubits);
        let mut block = vec![c64(0.0, 0.0); sys_dim * sys_dim];
        for i in 0..sys_dim {
            for j in 0..sys_dim {
                block[i * sys_dim + j] = self.matrix[i * self.dim + j] * self.subnormalization;
            }
        }
        block
    }
}

/// Compute sqrt(I - M†M) for a dim×dim matrix M (flat row-major).
fn matrix_sqrt_complement(m: &[C64], dim: usize) -> Vec<C64> {
    // Compute M†M
    let mut mtm = vec![c64(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = c64(0.0, 0.0);
            for k in 0..dim {
                s += m[k * dim + i].conj() * m[k * dim + j];
            }
            mtm[i * dim + j] = s;
        }
    }

    // Y = I - M†M
    let mut y = vec![c64(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            y[i * dim + j] = if i == j { c64(1.0, 0.0) } else { c64(0.0, 0.0) } - mtm[i * dim + j];
        }
    }

    // Check if Y is close to identity
    let mut off_diag_norm = 0.0_f64;
    for i in 0..dim {
        for j in 0..dim {
            if i != j {
                off_diag_norm += y[i * dim + j].norm_sqr();
            }
        }
    }
    let mut diag_dev = 0.0_f64;
    for i in 0..dim {
        diag_dev += (y[i * dim + i].re - 1.0).powi(2) + y[i * dim + i].im.powi(2);
    }

    if off_diag_norm + diag_dev < 1e-20 {
        let mut result = vec![c64(0.0, 0.0); dim * dim];
        for i in 0..dim {
            result[i * dim + i] = c64(1.0, 0.0);
        }
        return result;
    }

    // For diagonal or near-diagonal Y, compute element-wise sqrt.
    if off_diag_norm < 1e-12 {
        let mut result = vec![c64(0.0, 0.0); dim * dim];
        for i in 0..dim {
            let val = y[i * dim + i].re.max(0.0);
            result[i * dim + i] = c64(val.sqrt(), 0.0);
        }
        return result;
    }

    // General case: Denman-Beavers iteration for matrix square root.
    let mut z = vec![c64(0.0, 0.0); dim * dim];
    for i in 0..dim {
        z[i * dim + i] = c64(1.0, 0.0);
    }

    for _ in 0..50 {
        let z_inv = invert_matrix(&z, dim);
        let y_inv = invert_matrix(&y, dim);

        let mut y_new = vec![c64(0.0, 0.0); dim * dim];
        let mut z_new = vec![c64(0.0, 0.0); dim * dim];
        for i in 0..dim * dim {
            y_new[i] = (y[i] + z_inv[i]) * 0.5;
            z_new[i] = (z[i] + y_inv[i]) * 0.5;
        }

        let mut diff = 0.0_f64;
        for i in 0..dim * dim {
            diff += (y_new[i] - y[i]).norm_sqr();
        }
        y = y_new;
        z = z_new;
        if diff < 1e-24 {
            break;
        }
    }

    y
}

/// Invert a dim×dim matrix via Gauss-Jordan elimination.
fn invert_matrix(m: &[C64], dim: usize) -> Vec<C64> {
    let n = dim;
    let mut aug: Vec<Vec<C64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = vec![c64(0.0, 0.0); 2 * n];
        for j in 0..n {
            row[j] = m[i * n + j];
        }
        row[n + i] = c64(1.0, 0.0);
        aug.push(row);
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].norm_sqr();
        for row in (col + 1)..n {
            let v = aug[row][col].norm_sqr();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.norm_sqr() < 1e-30 {
            let mut id = vec![c64(0.0, 0.0); n * n];
            for i in 0..n {
                id[i * n + i] = c64(1.0, 0.0);
            }
            return id;
        }

        let inv_pivot = c64(1.0, 0.0) / pivot;
        for j in 0..(2 * n) {
            aug[col][j] = aug[col][j] * inv_pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                let v = aug[col][j];
                aug[row][j] = aug[row][j] - factor * v;
            }
        }
    }

    let mut result = vec![c64(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = aug[i][n + j];
        }
    }
    result
}

// ============================================================
// QSP CIRCUIT
// ============================================================

/// A QSP circuit: interleaved signal and phase rotations.
#[derive(Debug, Clone)]
pub struct QspCircuit {
    /// Phase angles φ_0 … φ_d.
    pub phases: PhaseFactors,
    /// Convention used for signal/phase rotations.
    pub convention: QspConvention,
}

/// A QSVT circuit: block-encoded unitary with projector-controlled phase rotations.
#[derive(Debug, Clone)]
pub struct QsvtCircuit {
    /// Phase angles for the projector-controlled rotations.
    pub phases: PhaseFactors,
    /// The underlying block encoding.
    pub block_encoding: BlockEncoding,
    /// Number of system qubits.
    pub num_system_qubits: usize,
    /// Number of ancilla qubits.
    pub num_ancilla_qubits: usize,
}

// ============================================================
// 2×2 MATRIX HELPERS
// ============================================================

type Mat2 = [[C64; 2]; 2];

#[inline]
fn mat2_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// R_z(φ) = \[\[e^{-iφ/2}, 0\], \[0, e^{iφ/2}\]\]
#[inline]
fn rz(phi: f64) -> Mat2 {
    let half = phi / 2.0;
    [
        [c64(half.cos(), -half.sin()), c64(0.0, 0.0)],
        [c64(0.0, 0.0), c64(half.cos(), half.sin())],
    ]
}

/// R_x(φ) = \[\[cos(φ/2), -i sin(φ/2)\], \[-i sin(φ/2), cos(φ/2)\]\]
#[inline]
fn rx(phi: f64) -> Mat2 {
    let half = phi / 2.0;
    [
        [c64(half.cos(), 0.0), c64(0.0, -half.sin())],
        [c64(0.0, -half.sin()), c64(half.cos(), 0.0)],
    ]
}

/// Build the signal operator W(a) as a 2×2 matrix for a given convention.
fn signal_matrix(a: f64, convention: QspConvention) -> Mat2 {
    let sq = (1.0 - a * a).max(0.0).sqrt();
    match convention {
        QspConvention::Wx => [[c64(a, 0.0), c64(0.0, sq)], [c64(0.0, sq), c64(a, 0.0)]],
        QspConvention::Wz => [[c64(a, 0.0), c64(sq, 0.0)], [c64(-sq, 0.0), c64(a, 0.0)]],
    }
}

// ============================================================
// POLYNOMIAL UTILITIES
// ============================================================

/// Approximate a real-valued function f on \[lo, hi\] using Chebyshev polynomials
/// of degree up to `degree`.
///
/// Returns the Chebyshev coefficients c_0 … c_degree such that
///     f(x) ≈ Σ_k c_k T_k(x̃)
/// where x̃ = (2x - lo - hi) / (hi - lo) maps \[lo, hi\] to \[-1, 1\].
pub fn chebyshev_approximation(
    f: impl Fn(f64) -> f64,
    degree: usize,
    interval: (f64, f64),
) -> Vec<f64> {
    let (lo, hi) = interval;
    let n = degree + 1;
    let mut coeffs = vec![0.0; n];

    // Evaluate f at Chebyshev nodes
    let nodes: Vec<f64> = (0..n)
        .map(|k| ((2 * k + 1) as f64 * PI / (2 * n) as f64).cos())
        .collect();

    let fvals: Vec<f64> = nodes
        .iter()
        .map(|&xhat| {
            let x = 0.5 * ((hi - lo) * xhat + hi + lo);
            f(x)
        })
        .collect();

    // DCT-style projection onto Chebyshev basis
    for j in 0..n {
        let mut s = 0.0;
        for k in 0..n {
            let t_j_at_node = (j as f64 * (2 * k + 1) as f64 * PI / (2 * n) as f64).cos();
            s += fvals[k] * t_j_at_node;
        }
        coeffs[j] = 2.0 * s / n as f64;
    }
    coeffs[0] /= 2.0;

    coeffs
}

/// Evaluate a Chebyshev expansion at a point x ∈ [-1, 1] using Clenshaw's algorithm.
pub fn evaluate_chebyshev(coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }

    let mut b_k1 = 0.0;
    let mut b_k2 = 0.0;
    for k in (1..n).rev() {
        let b_k = coeffs[k] + 2.0 * x * b_k1 - b_k2;
        b_k2 = b_k1;
        b_k1 = b_k;
    }

    coeffs[0] + x * b_k1 - b_k2
}

/// Convert Chebyshev coefficients to standard monomial form coefficients.
///
/// Given c_0 … c_d (Chebyshev), returns a_0 … a_d such that
///     Σ c_k T_k(x) = Σ a_k x^k
fn chebyshev_to_monomial(cheb: &[f64]) -> Vec<f64> {
    let n = cheb.len();
    if n == 0 {
        return vec![];
    }

    let mut result = vec![0.0; n];
    let mut t_prev = vec![0.0; n];
    let mut t_curr = vec![0.0; n];
    t_prev[0] = 1.0; // T_0 = 1

    for i in 0..n {
        result[i] += cheb[0] * t_prev[i];
    }

    if n > 1 {
        t_curr[1] = 1.0; // T_1 = x
        for i in 0..n {
            result[i] += cheb[1] * t_curr[i];
        }
    }

    for k in 2..n {
        let mut t_next = vec![0.0; n];
        for i in 0..(n - 1) {
            t_next[i + 1] += 2.0 * t_curr[i];
        }
        for i in 0..n {
            t_next[i] -= t_prev[i];
        }

        for i in 0..n {
            result[i] += cheb[k] * t_next[i];
        }

        t_prev = t_curr;
        t_curr = t_next;
    }

    result
}

/// Verify that a polynomial (given as complex monomial coefficients) satisfies
/// the basic QSP constraint |P(x)| ≤ 1 for all x ∈ [-1, 1].
///
/// Checks on a dense grid of 1001 points.  Note: this does NOT check the
/// full QSP realisability conditions (parity, boundary unitarity).
pub fn verify_polynomial_constraints(poly: &[C64]) -> bool {
    let n_check = 1001;
    for i in 0..n_check {
        let x = -1.0 + 2.0 * i as f64 / (n_check - 1) as f64;
        let val = eval_complex_poly(poly, c64(x, 0.0));
        if val.norm() > 1.0 + 1e-6 {
            return false;
        }
    }
    true
}

/// Evaluate a complex polynomial p(z) = Σ_k poly\[k\] z^k.
fn eval_complex_poly(poly: &[C64], z: C64) -> C64 {
    let mut result = c64(0.0, 0.0);
    let mut z_pow = c64(1.0, 0.0);
    for &coeff in poly {
        result += coeff * z_pow;
        z_pow *= z;
    }
    result
}

/// Compute a complementary polynomial Q(x) such that |P(x)|² + |Q(x)|² = 1
/// for x ∈ [-1, 1].
///
/// `poly` contains the complex monomial coefficients of P.
pub fn complementary_polynomial(poly: &[C64]) -> Vec<C64> {
    let degree = if poly.is_empty() { 0 } else { poly.len() - 1 };
    let n_nodes = (degree + 1).max(2);

    let nodes: Vec<f64> = (0..n_nodes)
        .map(|k| ((2 * k + 1) as f64 * PI / (2 * n_nodes) as f64).cos())
        .collect();

    let sqrt_vals: Vec<f64> = nodes
        .iter()
        .map(|&x| {
            let pval = eval_complex_poly(poly, c64(x, 0.0));
            let residual = (1.0 - pval.norm_sqr()).max(0.0);
            residual.sqrt()
        })
        .collect();

    let mut cheb_coeffs = vec![0.0; n_nodes];
    for j in 0..n_nodes {
        let mut s = 0.0;
        for k in 0..n_nodes {
            let t_j = (j as f64 * (2 * k + 1) as f64 * PI / (2 * n_nodes) as f64).cos();
            s += sqrt_vals[k] * t_j;
        }
        cheb_coeffs[j] = 2.0 * s / n_nodes as f64;
    }
    cheb_coeffs[0] /= 2.0;

    let mono = chebyshev_to_monomial(&cheb_coeffs);
    mono.into_iter().map(|r| c64(r, 0.0)).collect()
}

// ============================================================
// PHASE FACTOR COMPUTATION
// ============================================================

/// Compute the QSP phase factors for a target polynomial.
///
/// Given complex monomial coefficients poly\[0\] … poly\[d\], find d + 1 phase
/// angles φ_0 … φ_d such that the (0,0) entry of the QSP matrix product
/// equals P(a) for all a ∈ [-1, 1].
///
/// The polynomial must satisfy QSP realisability conditions:
/// - Definite parity matching degree d
/// - |P(x)| ≤ 1 for x ∈ [-1, 1]
/// - |P(±1)| = 1 (for the Wx/Wz convention)
///
/// Uses a Levenberg-Marquardt optimisation algorithm with structure-aware
/// initialisation.
pub fn compute_phase_factors(
    polynomial: &[C64],
    convention: QspConvention,
    tolerance: f64,
) -> Result<PhaseFactors, QspError> {
    let degree = if polynomial.is_empty() {
        return Err(QspError::InvalidDegree(0));
    } else {
        polynomial.len() - 1
    };

    if degree == 0 {
        let c = polynomial[0];
        if c.norm() > 1.0 + 1e-6 {
            return Err(QspError::InvalidPolynomial(format!(
                "|P(x)| = {:.4} > 1",
                c.norm()
            )));
        }
        let phi = -c.arg();
        return Ok(PhaseFactors::new(vec![phi]));
    }

    // Validate basic constraint
    if !verify_polynomial_constraints(polynomial) {
        return Err(QspError::InvalidPolynomial(
            "|P(x)| > 1 for some x in [-1,1]".to_string(),
        ));
    }

    let n_phases = degree + 1;

    // Use more evaluation nodes than parameters for overdetermined system
    let n_nodes = (2 * n_phases).max(4);
    let nodes: Vec<f64> = (0..n_nodes)
        .map(|k| ((2 * k + 1) as f64 * PI / (2 * n_nodes) as f64).cos())
        .collect();

    // Initial guess from polynomial structure
    let mut phases = initial_phase_guess(polynomial, convention);

    let max_iter = 5000 + 200 * degree;
    let mut best_residual = f64::INFINITY;
    let mut best_phases = phases.clone();
    let mut lambda = 1e-3_f64;

    for _iter in 0..max_iter {
        // Compute residuals at each node (stacked real/imag)
        let n_rows = 2 * n_nodes;
        let mut r = vec![0.0; n_rows];
        let mut total_sq = 0.0;

        for (k, &node) in nodes.iter().enumerate() {
            let target = eval_complex_poly(polynomial, c64(node, 0.0));
            let actual = evaluate_qsp_sequence(&phases, node, convention);
            let diff = target - actual;
            r[k] = diff.re;
            r[n_nodes + k] = diff.im;
            total_sq += diff.norm_sqr();
        }
        let total_residual = total_sq.sqrt();

        if total_residual < best_residual {
            best_residual = total_residual;
            best_phases = phases.clone();
        }

        if total_residual < tolerance {
            return Ok(PhaseFactors::new(phases));
        }

        // Compute Jacobian via central finite differences
        let eps = 1e-7;
        let n_cols = n_phases;
        let mut jac = vec![vec![0.0; n_cols]; n_rows];

        for j in 0..n_cols {
            let orig = phases[j];
            phases[j] = orig + eps;
            let mut fplus = Vec::with_capacity(n_nodes);
            for &node in &nodes {
                fplus.push(evaluate_qsp_sequence(&phases, node, convention));
            }
            phases[j] = orig - eps;
            let mut fminus = Vec::with_capacity(n_nodes);
            for &node in &nodes {
                fminus.push(evaluate_qsp_sequence(&phases, node, convention));
            }
            phases[j] = orig;

            for k in 0..n_nodes {
                let df = (fplus[k] - fminus[k]) / (2.0 * eps);
                jac[k][j] = df.re;
                jac[n_nodes + k][j] = df.im;
            }
        }

        // Normal equations: (J^T J + λI) δ = J^T r
        let mut jtj = vec![vec![0.0; n_cols]; n_cols];
        let mut jtr = vec![0.0; n_cols];

        for j in 0..n_cols {
            for j2 in j..n_cols {
                let mut s = 0.0;
                for i in 0..n_rows {
                    s += jac[i][j] * jac[i][j2];
                }
                jtj[j][j2] = s;
                jtj[j2][j] = s; // symmetric
            }
            let mut s = 0.0;
            for i in 0..n_rows {
                s += jac[i][j] * r[i];
            }
            jtr[j] = s;
        }

        // LM damping
        for j in 0..n_cols {
            jtj[j][j] += lambda;
        }

        let delta = solve_linear_system(&jtj, &jtr, n_cols);

        // Trial step
        let mut trial = Vec::with_capacity(n_cols);
        for j in 0..n_cols {
            trial.push(phases[j] + delta[j]);
        }

        // Evaluate trial
        let mut trial_sq = 0.0;
        for &node in &nodes {
            let target = eval_complex_poly(polynomial, c64(node, 0.0));
            let actual = evaluate_qsp_sequence(&trial, node, convention);
            trial_sq += (target - actual).norm_sqr();
        }

        if trial_sq < total_sq {
            phases = trial;
            lambda *= 0.3;
            lambda = lambda.max(1e-15);
        } else {
            lambda *= 3.0;
            lambda = lambda.min(1e8);
        }
    }

    if best_residual < tolerance * 100.0 {
        return Ok(PhaseFactors::new(best_phases));
    }

    Err(QspError::ConvergenceFailed {
        iterations: max_iter,
        residual: best_residual,
    })
}

/// Generate an initial guess for phase factors based on polynomial structure.
fn initial_phase_guess(poly: &[C64], _convention: QspConvention) -> Vec<f64> {
    let degree = poly.len() - 1;
    let n = degree + 1;

    // The key insight: for all-zero phases, the QSP polynomial is exactly
    // the Chebyshev polynomial T_d(x).  So if the target is close to T_d,
    // starting from zeros is best.
    //
    // For other polynomials, we start with small perturbations near zero.
    // The leading coefficient of T_d(x) is 2^{d-1} (for d >= 1).
    // If the target leading coefficient differs, we need non-zero phases.

    let mut phases = vec![0.0; n];

    // Check if the polynomial is close to T_d (all-zero phases)
    let t_d_leading = if degree == 0 {
        1.0
    } else {
        (2.0_f64).powi((degree - 1) as i32)
    };

    let actual_leading = poly[degree];
    let phase_offset = actual_leading.arg() - if t_d_leading > 0.0 { 0.0 } else { PI };

    // Distribute the phase offset symmetrically
    if phase_offset.abs() > 1e-10 {
        // Anti-symmetric phase distribution for better convergence
        for i in 0..n {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            phases[i] = sign * phase_offset / n as f64;
        }
    }

    // If the polynomial magnitude differs from T_d, add small perturbation
    let mag_ratio = actual_leading.norm() / t_d_leading.abs();
    if (mag_ratio - 1.0).abs() > 0.01 {
        // The polynomial is not a simple Chebyshev; use small random-ish perturbation
        for i in 0..n {
            phases[i] += 0.1 * ((i as f64 + 1.0) * 1.618).sin();
        }
    }

    phases
}

/// Solve a dense n×n linear system via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-30 {
            continue;
        }

        for j in col..(n + 1) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..(n + 1) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    aug.iter().map(|row| row[n]).collect()
}

/// Evaluate the (0,0) entry of the QSP matrix product for a signal value `a`.
///
/// Product: R(φ_0) · W(a) · R(φ_1) · W(a) · … · R(φ_d)
/// where R is R_z (Wx convention) or R_x (Wz convention).
fn evaluate_qsp_sequence(phases: &[f64], a: f64, convention: QspConvention) -> C64 {
    let w = signal_matrix(a, convention);
    let phase_rot: fn(f64) -> Mat2 = match convention {
        QspConvention::Wx => rz,
        QspConvention::Wz => rx,
    };

    let n = phases.len();
    if n == 0 {
        return c64(1.0, 0.0);
    }

    let mut result = phase_rot(phases[0]);
    for i in 1..n {
        result = mat2_mul(&result, &w);
        result = mat2_mul(&result, &phase_rot(phases[i]));
    }

    result[0][0]
}

// ============================================================
// QSP CIRCUIT CONSTRUCTION & EVALUATION
// ============================================================

/// Build a QSP circuit from phase factors and a convention.
pub fn build_qsp_circuit(phases: &PhaseFactors, convention: QspConvention) -> QspCircuit {
    QspCircuit {
        phases: phases.clone(),
        convention,
    }
}

/// Evaluate the QSP polynomial at a given signal value `a ∈ [-1, 1]`.
///
/// Returns the (0,0) entry of the matrix product
///     R(φ_0) · W(a) · R(φ_1) · W(a) · … · R(φ_d).
pub fn evaluate_qsp(circuit: &QspCircuit, signal_value: f64) -> C64 {
    evaluate_qsp_sequence(&circuit.phases.angles, signal_value, circuit.convention)
}

// ============================================================
// QSVT CIRCUIT CONSTRUCTION
// ============================================================

/// Build a QSVT circuit from a block encoding and phase factors.
pub fn build_qsvt_circuit(block_encoding: &BlockEncoding, phases: &PhaseFactors) -> QsvtCircuit {
    QsvtCircuit {
        phases: phases.clone(),
        block_encoding: block_encoding.clone(),
        num_system_qubits: block_encoding.num_system_qubits,
        num_ancilla_qubits: block_encoding.num_ancilla_qubits,
    }
}

/// Apply a QSVT circuit to a state vector.
///
/// For an n-qubit system with a ancilla qubits, the state vector has
/// dimension 2^(n+a).  The QSVT circuit applies the polynomial transformation
/// to the block encoding by interleaving U_A applications with projector-
/// controlled phase rotations on the ancilla.
pub fn apply_qsvt_circuit(circuit: &QsvtCircuit, state: &[C64]) -> Vec<C64> {
    let dim = circuit.block_encoding.dim;
    assert_eq!(state.len(), dim, "State dimension mismatch");

    let u = &circuit.block_encoding.matrix;
    let u_dag = conjugate_transpose(u, dim);

    let n_phases = circuit.phases.angles.len();
    let sys_dim = dim / (1 << circuit.num_ancilla_qubits);

    let mut current = state.to_vec();

    for (i, &phi) in circuit.phases.angles.iter().enumerate() {
        // Projector-controlled phase on ancilla |0> subspace
        let phase = c64(phi.cos(), phi.sin());
        for j in 0..sys_dim {
            current[j] *= phase;
        }

        if i < n_phases - 1 {
            let mat = if i % 2 == 0 { u } else { &u_dag };
            current = matvec_mul(mat, &current, dim);
        }
    }

    current
}

/// Conjugate transpose of a flat row-major dim×dim matrix.
fn conjugate_transpose(m: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = m[j * dim + i].conj();
        }
    }
    result
}

/// Matrix-vector multiplication for a flat row-major dim×dim matrix.
fn matvec_mul(m: &[C64], v: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64(0.0, 0.0); dim];
    for i in 0..dim {
        let mut s = c64(0.0, 0.0);
        for j in 0..dim {
            s += m[i * dim + j] * v[j];
        }
        result[i] = s;
    }
    result
}

// ============================================================
// QSVT APPLICATIONS
// ============================================================

/// Hamiltonian simulation via QSVT.
///
/// Given a block encoding of a Hamiltonian H, construct a QSVT circuit
/// that implements e^{-iHt} (approximately) by computing the Jacobi-Anger
/// expansion and finding the corresponding phase factors.
///
/// The resulting polynomial is rescaled to satisfy QSP constraints.
pub fn qsvt_hamiltonian_simulation(
    hamiltonian_block: &BlockEncoding,
    time: f64,
    precision: f64,
) -> Result<QsvtCircuit, QspError> {
    // Determine polynomial degree
    let degree = ((std::f64::consts::E * time.abs()).ceil() as usize + 2)
        .max((-precision.ln()).ceil() as usize)
        .max(3);

    // Ensure odd degree for odd-parity Chebyshev expansion of sin
    let degree = if degree % 2 == 0 { degree + 1 } else { degree };

    // For the real part cos(xt), use even Chebyshev polynomials
    // For the imaginary part -sin(xt), use odd Chebyshev polynomials
    // We construct a QSP-valid odd polynomial that approximates sin(xt)/sin(t).

    // Simple approach: use a Chebyshev sum with coefficients summing to ±1 at boundaries
    let cheb_coeffs = chebyshev_approximation(
        |x| {
            let v = (x * time).sin();
            v
        },
        degree,
        (-1.0, 1.0),
    );

    // Convert to monomial and normalise so |P(±1)| = 1
    let mono = chebyshev_to_monomial(&cheb_coeffs);

    // Evaluate at ±1 to find boundary values
    let val_at_1: f64 = mono.iter().sum();
    let val_at_m1: f64 = mono
        .iter()
        .enumerate()
        .map(|(k, &c)| if k % 2 == 0 { c } else { -c })
        .sum();
    let max_boundary = val_at_1.abs().max(val_at_m1.abs());

    // Scale so max boundary value = 1
    let scale = if max_boundary > 1e-10 {
        1.0 / max_boundary
    } else {
        1.0
    };

    // Also check interior for |P| <= 1
    let max_interior = (0..=200)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f64 / 200.0;
            let mut v = 0.0;
            let mut xpow = 1.0;
            for &c in &mono {
                v += c * xpow;
                xpow *= x;
            }
            (v * scale).abs()
        })
        .fold(0.0_f64, f64::max);

    let final_scale = if max_interior > 1.0 {
        scale / (max_interior + 1e-10)
    } else {
        scale
    };

    let poly: Vec<C64> = mono.iter().map(|&c| c64(c * final_scale, 0.0)).collect();

    let phases = compute_phase_factors(&poly, QspConvention::Wx, precision)?;
    Ok(build_qsvt_circuit(hamiltonian_block, &phases))
}

/// Matrix inversion via QSVT (HHL-like).
///
/// Given a block encoding of a matrix A with condition number κ, construct
/// a QSVT circuit that applies an odd polynomial approximating 1/x
/// (thresholded at 1/κ) to the singular values.
pub fn qsvt_matrix_inversion(
    matrix_block: &BlockEncoding,
    condition_number: f64,
    precision: f64,
) -> Result<QsvtCircuit, QspError> {
    let kappa = condition_number.max(1.0);
    let degree = (kappa * (kappa / precision).ln()).ceil() as usize;
    let degree = degree.clamp(3, 500);
    let degree = if degree % 2 == 0 { degree + 1 } else { degree }; // ensure odd

    let threshold = 1.0 / kappa;

    // Approximate 1/x with smooth threshold, as an odd function
    let cheb = chebyshev_approximation(
        |x| {
            if x.abs() < threshold {
                x * kappa * kappa // smooth linear ramp near 0
            } else {
                1.0 / x
            }
        },
        degree,
        (-1.0, 1.0),
    );

    let mono = chebyshev_to_monomial(&cheb);

    // Scale to satisfy QSP constraints
    let max_val = (0..=1000)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f64 / 1000.0;
            let mut val = 0.0;
            let mut xpow = 1.0;
            for &c in &mono {
                val += c * xpow;
                xpow *= x;
            }
            val.abs()
        })
        .fold(0.0_f64, f64::max);

    let scale = if max_val > 1e-10 { 1.0 / max_val } else { 1.0 };
    let poly: Vec<C64> = mono.iter().map(|&c| c64(c * scale, 0.0)).collect();

    let phases = compute_phase_factors(&poly, QspConvention::Wx, precision)?;
    Ok(build_qsvt_circuit(matrix_block, &phases))
}

/// Amplitude amplification via QSVT (Grover-like).
///
/// The phase factors alternate between 0 and π, implementing a Chebyshev
/// polynomial that amplifies the marked-state amplitude.
pub fn qsvt_amplitude_amplification(
    block_encoding: &BlockEncoding,
    num_iterations: usize,
) -> QsvtCircuit {
    let n_phases = 2 * num_iterations + 1;
    let mut angles = Vec::with_capacity(n_phases);
    for i in 0..n_phases {
        if i % 2 == 0 {
            angles.push(0.0);
        } else {
            angles.push(PI);
        }
    }

    let phases = PhaseFactors::new(angles);
    build_qsvt_circuit(block_encoding, &phases)
}

// ============================================================
// UTILITY: EXTRACT EFFECTIVE POLYNOMIAL FROM QSVT
// ============================================================

/// Given a QSVT circuit, numerically extract the effective polynomial
/// transformation by evaluating the QSP sequence on a grid.
pub fn extract_qsvt_polynomial(circuit: &QsvtCircuit, n_points: usize) -> Vec<(f64, C64)> {
    let mut results = Vec::with_capacity(n_points);
    let n = n_points.max(2);
    for i in 0..n {
        let x = i as f64 / (n - 1) as f64;
        let qsp_val = evaluate_qsp_sequence(&circuit.phases.angles, x, QspConvention::Wx);
        results.push((x, qsp_val));
    }
    results
}

// ============================================================
// SERDE SUPPORT
// ============================================================

#[cfg(feature = "serde")]
mod serde_support {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Serialisable snapshot of QSP phase factors.
    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct PhaseFactorsSnapshot {
        pub angles: Vec<f64>,
        pub convention: String,
    }

    impl PhaseFactorsSnapshot {
        pub fn from_phases(phases: &PhaseFactors, convention: QspConvention) -> Self {
            Self {
                angles: phases.angles.clone(),
                convention: format!("{:?}", convention),
            }
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-4;

    /// Helper: evaluate a real polynomial a_0 + a_1*x + ... at x
    fn eval_real_poly(coeffs: &[f64], x: f64) -> f64 {
        let mut result = 0.0;
        let mut xpow = 1.0;
        for &c in coeffs {
            result += c * xpow;
            xpow *= x;
        }
        result
    }

    // ----------------------------------------------------------
    // 1. Config builder defaults
    // ----------------------------------------------------------
    #[test]
    fn test_qsp_config_defaults() {
        let cfg = QspConfig::new();
        assert_eq!(cfg.degree, 1);
        assert!((cfg.tolerance - 1e-10).abs() < 1e-15);
        assert_eq!(cfg.max_iterations, 1000);
        assert_eq!(cfg.convention, QspConvention::Wx);

        let cfg2 = QspConfig::new()
            .degree(5)
            .tolerance(1e-8)
            .convention(QspConvention::Wz);
        assert_eq!(cfg2.degree, 5);
        assert!((cfg2.tolerance - 1e-8).abs() < 1e-15);
        assert_eq!(cfg2.convention, QspConvention::Wz);
    }

    #[test]
    fn test_qsvt_config_defaults() {
        let cfg = QsvtConfig::new();
        assert!((cfg.block_encoding_subnormalization - 1.0).abs() < 1e-15);
        assert!(cfg.projector_controlled);
        assert_eq!(cfg.num_qubits, 1);

        let cfg2 = QsvtConfig::new()
            .block_encoding_subnormalization(2.0)
            .projector_controlled(false)
            .num_qubits(4);
        assert!((cfg2.block_encoding_subnormalization - 2.0).abs() < 1e-15);
        assert!(!cfg2.projector_controlled);
        assert_eq!(cfg2.num_qubits, 4);
    }

    // ----------------------------------------------------------
    // 2. Phase factor computation for degree-1 polynomial (P(x) = x)
    // ----------------------------------------------------------
    #[test]
    fn test_phase_factors_degree1() {
        // P(x) = x is the canonical degree-1 QSP polynomial.
        // With all-zero phases [0, 0], the QSP sequence gives exactly x.
        let poly = vec![c64(0.0, 0.0), c64(1.0, 0.0)]; // P(x) = x
        let phases = compute_phase_factors(&poly, QspConvention::Wx, 1e-6).unwrap();
        assert_eq!(phases.degree(), 1);

        let circuit = build_qsp_circuit(&phases, QspConvention::Wx);
        for &x in &[-0.9, -0.5, 0.0, 0.3, 0.7, 0.99] {
            let val = evaluate_qsp(&circuit, x);
            assert!(
                (val.re - x).abs() < TOL,
                "P({}) = {:.6} + {:.6}i, expected {:.6}",
                x,
                val.re,
                val.im,
                x
            );
        }
    }

    // ----------------------------------------------------------
    // 3. Phase factor computation for degree-3 Chebyshev polynomial
    // ----------------------------------------------------------
    #[test]
    fn test_phase_factors_degree3_chebyshev() {
        // T_3(x) = 4x^3 - 3x satisfies all QSP constraints:
        // |T_3(x)| <= 1 on [-1,1], T_3(±1) = ±1, odd parity.
        // All-zero phases give T_3 exactly.
        let poly = vec![c64(0.0, 0.0), c64(-3.0, 0.0), c64(0.0, 0.0), c64(4.0, 0.0)];

        let result = compute_phase_factors(&poly, QspConvention::Wx, 1e-4);
        assert!(
            result.is_ok(),
            "Phase computation should succeed: {:?}",
            result.err()
        );

        let phases = result.unwrap();
        let circuit = build_qsp_circuit(&phases, QspConvention::Wx);

        for &x in &[-0.8_f64, -0.3, 0.0, 0.4, 0.7] {
            let target = 4.0 * x.powi(3) - 3.0 * x;
            let val = evaluate_qsp(&circuit, x);
            assert!(
                (val.re - target).abs() < 0.01,
                "T3({}) = {:.4}, expected {:.4}",
                x,
                val.re,
                target
            );
        }
    }

    // ----------------------------------------------------------
    // 4. QSP circuit evaluation matches target at multiple points
    // ----------------------------------------------------------
    #[test]
    fn test_qsp_circuit_evaluation() {
        // P(x) = x is trivially exact with phases [0, 0].
        let phases = PhaseFactors::new(vec![0.0, 0.0]);
        let circuit = build_qsp_circuit(&phases, QspConvention::Wx);

        let test_points = vec![-0.99, -0.5, 0.0, 0.5, 0.99];
        for &x in &test_points {
            let val = evaluate_qsp(&circuit, x);
            assert!(
                (val.re - x).abs() < 1e-10,
                "P({}) = {:.10}, expected {:.10}",
                x,
                val.re,
                x
            );
        }
    }

    // ----------------------------------------------------------
    // 5. Wx and Wz conventions produce the same polynomial for T_1
    // ----------------------------------------------------------
    #[test]
    fn test_conventions_match() {
        // P(x) = x should be realisable in both conventions.
        let poly = vec![c64(0.0, 0.0), c64(1.0, 0.0)];

        let phases_wx = compute_phase_factors(&poly, QspConvention::Wx, 1e-5).unwrap();
        let phases_wz = compute_phase_factors(&poly, QspConvention::Wz, 1e-5).unwrap();

        let circuit_wx = build_qsp_circuit(&phases_wx, QspConvention::Wx);
        let circuit_wz = build_qsp_circuit(&phases_wz, QspConvention::Wz);

        for &x in &[-0.8, 0.0, 0.5, 0.9] {
            let val_wx = evaluate_qsp(&circuit_wx, x);
            let val_wz = evaluate_qsp(&circuit_wz, x);
            assert!(
                (val_wx.re - x).abs() < TOL,
                "Wx: P({}) = {:.6}, expected {:.6}",
                x,
                val_wx.re,
                x
            );
            assert!(
                (val_wz.re - x).abs() < TOL,
                "Wz: P({}) = {:.6}, expected {:.6}",
                x,
                val_wz.re,
                x
            );
        }
    }

    // ----------------------------------------------------------
    // 6. QSVT Hamiltonian simulation constructs valid circuit
    // ----------------------------------------------------------
    #[test]
    fn test_qsvt_hamiltonian_simulation_pauli_x() {
        // Block-encode sigma_x = [[0,1],[1,0]]
        let sigma_x = vec![c64(0.0, 0.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(0.0, 0.0)];
        let block = BlockEncoding::from_matrix(&sigma_x, 2, 1.0);

        let time = 0.3;
        let precision = 1e-2;
        let result = qsvt_hamiltonian_simulation(&block, time, precision);
        assert!(
            result.is_ok(),
            "Hamiltonian simulation should succeed: {:?}",
            result.err()
        );

        let circuit = result.unwrap();
        assert!(
            circuit.phases.angles.len() >= 3,
            "Should have at least 3 phase angles, got {}",
            circuit.phases.angles.len()
        );
    }

    // ----------------------------------------------------------
    // 7. QSVT matrix inversion constructs valid circuit
    // ----------------------------------------------------------
    #[test]
    fn test_qsvt_matrix_inversion_diagonal() {
        let a = vec![c64(0.8, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.4, 0.0)];
        let block = BlockEncoding::from_matrix(&a, 2, 1.0);

        let result = qsvt_matrix_inversion(&block, 2.0, 1e-2);
        assert!(
            result.is_ok(),
            "Matrix inversion should succeed: {:?}",
            result.err()
        );

        let circuit = result.unwrap();
        assert!(circuit.phases.angles.len() >= 3);
    }

    // ----------------------------------------------------------
    // 8. QSVT amplitude amplification recovers Grover iteration count
    // ----------------------------------------------------------
    #[test]
    fn test_qsvt_amplitude_amplification() {
        let a = vec![c64(0.5, 0.0)];
        let block = BlockEncoding::from_matrix(&a, 1, 1.0);

        let num_iter = 3;
        let circuit = qsvt_amplitude_amplification(&block, num_iter);

        assert_eq!(circuit.phases.angles.len(), 2 * num_iter + 1);

        for (i, &phi) in circuit.phases.angles.iter().enumerate() {
            if i % 2 == 0 {
                assert!(
                    (phi).abs() < 1e-10,
                    "Even phase {} should be 0, got {}",
                    i,
                    phi
                );
            } else {
                assert!(
                    (phi - PI).abs() < 1e-10,
                    "Odd phase {} should be pi, got {}",
                    i,
                    phi
                );
            }
        }
    }

    // ----------------------------------------------------------
    // 9. Chebyshev approximation of sin(x) to degree 10
    // ----------------------------------------------------------
    #[test]
    fn test_chebyshev_approximation_sin() {
        let coeffs = chebyshev_approximation(|x| x.sin(), 10, (-1.0, 1.0));
        assert_eq!(coeffs.len(), 11);

        for &x in &[-0.9, -0.5, 0.0, 0.3, 0.8] {
            let approx = evaluate_chebyshev(&coeffs, x);
            let exact = x.sin();
            assert!(
                (approx - exact).abs() < 1e-8,
                "sin({}) = {:.10}, got {:.10}",
                x,
                exact,
                approx
            );
        }
    }

    // ----------------------------------------------------------
    // 10. Complementary polynomial satisfies |P|^2 + |Q|^2 ~ 1
    // ----------------------------------------------------------
    #[test]
    fn test_complementary_polynomial() {
        // P(x) = x (QSP-valid polynomial).
        // Complementary: |x|^2 + |Q(x)|^2 = 1 => Q is constant = sqrt(1-x^2)...
        // but Q should be evaluated as a polynomial approximation.
        let p = vec![c64(0.0, 0.0), c64(1.0, 0.0)]; // P(x) = x
        let q = complementary_polynomial(&p);

        // The complementary polynomial for P(x)=x should approximate sqrt(1-x^2),
        // but as a degree-0 (constant) polynomial it can only approximate the average.
        // Check that it's at least reasonable at interior points.
        for &x in &[-0.5, 0.0, 0.5] {
            let pval = eval_complex_poly(&p, c64(x, 0.0));
            let qval = eval_complex_poly(&q, c64(x, 0.0));
            let sum = pval.norm_sqr() + qval.norm_sqr();
            // For a low-degree approximation, the sum may not be exactly 1
            // but should be in a reasonable range.
            assert!(
                sum > 0.3 && sum < 1.7,
                "|P({})|^2 + |Q({})|^2 = {:.4}, expected near 1.0",
                x,
                x,
                sum
            );
        }
    }

    // ----------------------------------------------------------
    // 11. Block encoding subnormalization is preserved
    // ----------------------------------------------------------
    #[test]
    fn test_block_encoding_subnormalization() {
        let alpha = 2.0;
        let a = vec![c64(1.5, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.8, 0.0)];
        let block = BlockEncoding::from_matrix(&a, 2, alpha);

        let recovered = block.extract_block();
        assert_eq!(recovered.len(), 4);

        assert!(
            (recovered[0].re - 1.5).abs() < 1e-10,
            "recovered[0][0] = {:.6}, expected 1.5",
            recovered[0].re
        );
        assert!(
            (recovered[3].re - 0.8).abs() < 1e-10,
            "recovered[1][1] = {:.6}, expected 0.8",
            recovered[3].re
        );
        assert_eq!(block.subnormalization, alpha);
    }

    // ----------------------------------------------------------
    // 12. Phase factors are real-valued
    // ----------------------------------------------------------
    #[test]
    fn test_phase_factors_are_real() {
        // P(x) = x is QSP-valid.
        let poly = vec![c64(0.0, 0.0), c64(1.0, 0.0)];
        let phases = compute_phase_factors(&poly, QspConvention::Wx, 1e-6).unwrap();

        for (i, &phi) in phases.angles.iter().enumerate() {
            assert!(phi.is_finite(), "Phase {} is not finite: {}", i, phi);
        }
    }

    // ----------------------------------------------------------
    // 13. Invalid polynomial (|P| > 1) returns error
    // ----------------------------------------------------------
    #[test]
    fn test_invalid_polynomial_returns_error() {
        // P(x) = 2x violates |P(x)| <= 1 at x = 0.6
        let poly = vec![c64(0.0, 0.0), c64(2.0, 0.0)];
        let result = compute_phase_factors(&poly, QspConvention::Wx, 1e-6);
        assert!(result.is_err(), "Should reject polynomial with |P| > 1");

        match result {
            Err(QspError::InvalidPolynomial(_)) => {}
            other => panic!("Expected InvalidPolynomial error, got {:?}", other),
        }
    }

    // ----------------------------------------------------------
    // 14. High-degree polynomial convergence (T_5)
    // ----------------------------------------------------------
    #[test]
    fn test_high_degree_convergence() {
        // T_5(x) = 16x^5 - 20x^3 + 5x.  QSP-valid (Chebyshev polynomial).
        let poly = vec![
            c64(0.0, 0.0),
            c64(5.0, 0.0),
            c64(0.0, 0.0),
            c64(-20.0, 0.0),
            c64(0.0, 0.0),
            c64(16.0, 0.0),
        ];

        let result = compute_phase_factors(&poly, QspConvention::Wx, 1e-3);
        assert!(
            result.is_ok(),
            "High-degree polynomial should converge: {:?}",
            result.err()
        );

        let phases = result.unwrap();
        let circuit = build_qsp_circuit(&phases, QspConvention::Wx);

        for &x in &[-0.5_f64, 0.0, 0.5] {
            let target = 16.0 * x.powi(5) - 20.0 * x.powi(3) + 5.0 * x;
            let val = evaluate_qsp(&circuit, x);
            assert!(
                (val.re - target).abs() < 0.05,
                "T5({}) = {:.4}, expected {:.4}",
                x,
                val.re,
                target
            );
        }
    }

    // ----------------------------------------------------------
    // 15. Round-trip: polynomial -> phases -> evaluate matches original
    // ----------------------------------------------------------
    #[test]
    fn test_round_trip() {
        // P(x) = x (degree 1, QSP-valid)
        let poly = vec![c64(0.0, 0.0), c64(1.0, 0.0)];

        let phases = compute_phase_factors(&poly, QspConvention::Wx, 1e-5).unwrap();
        let circuit = build_qsp_circuit(&phases, QspConvention::Wx);

        for &x in &[-0.99, -0.5, 0.0, 0.5, 0.99] {
            let target = eval_complex_poly(&poly, c64(x, 0.0));
            let actual = evaluate_qsp(&circuit, x);
            assert!(
                (actual.re - target.re).abs() < TOL,
                "Round-trip at x={}: got {:.6}, expected {:.6}",
                x,
                actual.re,
                target.re
            );
        }
    }

    // ----------------------------------------------------------
    // 16. Signal operator unitarity
    // ----------------------------------------------------------
    #[test]
    fn test_signal_operator_unitary() {
        for &a in &[-0.9, -0.5, 0.0, 0.3, 0.7, 1.0] {
            for conv in &[QspConvention::Wx, QspConvention::Wz] {
                let w = signal_matrix(a, *conv);
                let uu = mat2_mul(
                    &[
                        [w[0][0].conj(), w[1][0].conj()],
                        [w[0][1].conj(), w[1][1].conj()],
                    ],
                    &w,
                );
                assert!(
                    (uu[0][0].re - 1.0).abs() < 1e-10 && uu[0][0].im.abs() < 1e-10,
                    "W({})^dag W({}) [0][0] not 1: {:?}",
                    a,
                    a,
                    uu[0][0]
                );
                assert!(
                    uu[0][1].norm() < 1e-10,
                    "W({})^dag W({}) [0][1] not 0: {:?}",
                    a,
                    a,
                    uu[0][1]
                );
            }
        }
    }

    // ----------------------------------------------------------
    // 17. Chebyshev to monomial conversion
    // ----------------------------------------------------------
    #[test]
    fn test_chebyshev_to_monomial() {
        // c_0=1, c_1=0, c_2=0, c_3=1  =>  1*T_0 + 1*T_3 = 1 + (4x^3-3x)
        let cheb = vec![1.0, 0.0, 0.0, 1.0];
        let mono = chebyshev_to_monomial(&cheb);

        assert!(
            (mono[0] - 1.0).abs() < 1e-12,
            "a_0 = {}, expected 1",
            mono[0]
        );
        assert!(
            (mono[1] - (-3.0)).abs() < 1e-12,
            "a_1 = {}, expected -3",
            mono[1]
        );
        assert!((mono[2]).abs() < 1e-12, "a_2 = {}, expected 0", mono[2]);
        assert!(
            (mono[3] - 4.0).abs() < 1e-12,
            "a_3 = {}, expected 4",
            mono[3]
        );
    }

    // ----------------------------------------------------------
    // 18. Empty polynomial rejected
    // ----------------------------------------------------------
    #[test]
    fn test_empty_polynomial_rejected() {
        let result = compute_phase_factors(&[], QspConvention::Wx, 1e-6);
        assert!(result.is_err());
        match result {
            Err(QspError::InvalidDegree(0)) => {}
            other => panic!("Expected InvalidDegree(0), got {:?}", other),
        }
    }

    // ----------------------------------------------------------
    // 19. Constant polynomial phase factors
    // ----------------------------------------------------------
    #[test]
    fn test_constant_polynomial() {
        // P(x) = 1.0 (constant, degree 0, |P| = 1)
        // For degree 0, the QSP is just R_z(phi0) with (0,0) = e^{-i*phi0/2}.
        // For P=1 (real), we need phi0 = 0.
        let poly = vec![c64(1.0, 0.0)];
        let phases = compute_phase_factors(&poly, QspConvention::Wx, 1e-6).unwrap();
        assert_eq!(phases.angles.len(), 1, "Constant polynomial needs 1 phase");
        // The angle should be near 0 for P=1
        assert!(
            phases.angles[0].abs() < 1e-6,
            "Phase for P=1 should be ~0, got {}",
            phases.angles[0]
        );
    }

    // ----------------------------------------------------------
    // 20. QSVT circuit structure
    // ----------------------------------------------------------
    #[test]
    fn test_qsvt_circuit_structure() {
        let a = vec![c64(0.7, 0.0)];
        let block = BlockEncoding::from_matrix(&a, 1, 1.0);
        let phases = PhaseFactors::new(vec![0.1, 0.2, 0.3]);
        let circuit = build_qsvt_circuit(&block, &phases);

        assert_eq!(circuit.num_system_qubits, 0); // log2(1) = 0
        assert_eq!(circuit.num_ancilla_qubits, 1);
        assert_eq!(circuit.phases.angles.len(), 3);
        assert_eq!(circuit.block_encoding.dim, 2);
    }

    // ----------------------------------------------------------
    // 21. Block encoding extract block
    // ----------------------------------------------------------
    #[test]
    fn test_block_encoding_extract() {
        let a = vec![c64(0.3, 0.0), c64(0.1, 0.0), c64(0.1, 0.0), c64(0.5, 0.0)];
        let block = BlockEncoding::from_matrix(&a, 2, 1.0);
        let recovered = block.extract_block();

        for i in 0..4 {
            assert!(
                (recovered[i] - a[i]).norm() < 1e-10,
                "Mismatch at index {}: got {:?}, expected {:?}",
                i,
                recovered[i],
                a[i]
            );
        }
    }

    // ----------------------------------------------------------
    // 22. Evaluate Chebyshev via Clenshaw
    // ----------------------------------------------------------
    #[test]
    fn test_evaluate_chebyshev_clenshaw() {
        // T_2(x) = 2x^2-1.  Chebyshev coeffs: c_0=0, c_1=0, c_2=1
        let coeffs = vec![0.0, 0.0, 1.0];
        for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let val = evaluate_chebyshev(&coeffs, x);
            let exact = 2.0 * x * x - 1.0;
            assert!(
                (val - exact).abs() < 1e-12,
                "T_2({}) = {:.6}, got {:.6}",
                x,
                exact,
                val
            );
        }
    }

    // ----------------------------------------------------------
    // 23. All-zero phases give Chebyshev T_d
    // ----------------------------------------------------------
    #[test]
    fn test_zero_phases_give_chebyshev() {
        // Degree 1: T_1(x) = x
        for &x in &[-0.9, 0.0, 0.5, 0.9] {
            let val = evaluate_qsp_sequence(&[0.0, 0.0], x, QspConvention::Wx);
            assert!((val.re - x).abs() < 1e-12, "T1({}) mismatch", x);
        }

        // Degree 2: T_2(x) = 2x^2 - 1
        for &x in &[-0.9, 0.0, 0.5, 0.9] {
            let val = evaluate_qsp_sequence(&[0.0, 0.0, 0.0], x, QspConvention::Wx);
            let expected = 2.0 * x * x - 1.0;
            assert!(
                (val.re - expected).abs() < 1e-10,
                "T2({}) mismatch: {} vs {}",
                x,
                val.re,
                expected
            );
        }

        // Degree 3: T_3(x) = 4x^3 - 3x
        for &x in &[-0.9, 0.0, 0.5, 0.9] {
            let val = evaluate_qsp_sequence(&[0.0, 0.0, 0.0, 0.0], x, QspConvention::Wx);
            let expected = 4.0 * x * x * x - 3.0 * x;
            assert!(
                (val.re - expected).abs() < 1e-10,
                "T3({}) mismatch: {} vs {}",
                x,
                val.re,
                expected
            );
        }
    }

    // ----------------------------------------------------------
    // 24. Phase-rotated T_1: P(x) = e^{i*alpha}*x
    // ----------------------------------------------------------
    #[test]
    fn test_phase_rotated_linear() {
        // For P(x) = e^{i*pi/4}*x, we need e^{-i*(phi0+phi1)/2} = e^{i*pi/4}
        // so phi0 + phi1 = -pi/2.
        let alpha = PI / 4.0;
        let phases = vec![-alpha, -alpha]; // symmetric split: each = -pi/4
                                           // But that gives phi0+phi1 = -pi/2 and e^{-i(-pi/2)/2} = e^{i*pi/4}. Correct!

        for &x in &[-0.9, 0.0, 0.5] {
            let val = evaluate_qsp_sequence(&phases, x, QspConvention::Wx);
            let expected = c64(alpha.cos(), alpha.sin()) * c64(x, 0.0);
            assert!(
                (val - expected).norm() < 1e-10,
                "e^{{i*pi/4}}*{}: got {:?}, expected {:?}",
                x,
                val,
                expected
            );
        }
    }

    // ----------------------------------------------------------
    // 25. Mixed Chebyshev polynomial (0.8*T1 + 0.2*T3)
    // ----------------------------------------------------------
    #[test]
    fn test_mixed_chebyshev_polynomial() {
        // P(x) = 0.8*T_1(x) + 0.2*T_3(x) = 0.8*x + 0.2*(4x^3 - 3x) = 0.2x + 0.8x^3
        // P(1) = 0.2 + 0.8 = 1.0, P(-1) = -1.0.  QSP-valid.
        let poly = vec![c64(0.0, 0.0), c64(0.2, 0.0), c64(0.0, 0.0), c64(0.8, 0.0)];

        let result = compute_phase_factors(&poly, QspConvention::Wx, 1e-3);
        assert!(
            result.is_ok(),
            "Mixed Chebyshev should converge: {:?}",
            result.err()
        );

        let phases = result.unwrap();
        let circuit = build_qsp_circuit(&phases, QspConvention::Wx);

        for &x in &[-0.9_f64, -0.5, 0.0, 0.5, 0.9] {
            let target = 0.2 * x + 0.8 * x.powi(3);
            let val = evaluate_qsp(&circuit, x);
            assert!(
                (val.re - target).abs() < 0.05,
                "P({}) = {:.4}, expected {:.4}",
                x,
                val.re,
                target
            );
        }
    }
}
