//! Harrow-Hassidim-Lloyd (HHL) Algorithm for Quantum Linear Systems
//!
//! Solves the linear system Ax = b, producing a quantum state |x> proportional
//! to A^{-1}|b>. The algorithm uses:
//!
//! 1. **QPE** to decompose |b> into the eigenbasis of A
//! 2. **Controlled rotation** to encode 1/lambda into ancilla amplitudes
//! 3. **Inverse QPE** to uncompute the eigenvalue register
//! 4. **Ancilla measurement** to select successful outcomes
//!
//! Since this is a classical simulation of HHL, we compute the exact solution
//! via eigendecomposition of the Hermitian matrix A rather than constructing
//! the full QPE circuit. This gives identical mathematical results to the
//! quantum algorithm while being efficient to simulate.
//!
//! # References
//!
//! - Harrow, Hassidim, Lloyd. "Quantum Algorithm for Linear Systems of
//!   Equations" (2009). Physical Review Letters 103(15):150502.
//! - Childs, Kothari, Somma. "Quantum Algorithm for Systems of Linear
//!   Equations with Exponentially Improved Dependence on Precision" (2017).

use num_complex::Complex64;

// ============================================================
// LOCAL HELPERS
// ============================================================

type C64 = Complex64;

#[inline]
fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

#[inline]
fn c64_zero() -> C64 {
    c64(0.0, 0.0)
}

#[inline]
fn c64_one() -> C64 {
    c64(1.0, 0.0)
}

/// Matrix-vector multiply for flat row-major NxN matrix and length-N vector.
fn matvec(matrix: &[C64], vec: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim];
    for i in 0..dim {
        let mut s = c64_zero();
        for j in 0..dim {
            s += matrix[i * dim + j] * vec[j];
        }
        result[i] = s;
    }
    result
}

/// Matrix-matrix multiply for flat row-major NxN matrices.
fn matmul(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = c64_zero();
            for k in 0..dim {
                s += a[i * dim + k] * b[k * dim + j];
            }
            result[i * dim + j] = s;
        }
    }
    result
}

/// Conjugate transpose of a flat row-major NxN matrix.
fn adjoint(matrix: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[j * dim + i] = matrix[i * dim + j].conj();
        }
    }
    result
}

/// Convert Vec<Vec<C64>> to flat row-major representation.
fn vv_to_flat(matrix: &[Vec<C64>], dim: usize) -> Vec<C64> {
    let mut flat = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim.min(matrix[i].len()) {
            flat[i * dim + j] = matrix[i][j];
        }
    }
    flat
}

// ============================================================
// RESULT TYPE
// ============================================================

/// Result of the HHL quantum linear systems algorithm.
///
/// Contains the normalized quantum state |x> proportional to A^{-1}|b>,
/// along with metadata about the solution quality and algorithm performance.
#[derive(Debug, Clone)]
pub struct HHLResult {
    /// The solution vector x as a normalized quantum state.
    /// This is |x> = A^{-1}|b> / ||A^{-1}|b>||.
    pub solution: Vec<C64>,

    /// The norm ||A^{-1}b|| needed to recover the actual (unnormalized)
    /// solution from the quantum state: x_actual = solution_norm * solution.
    pub solution_norm: f64,

    /// Probability that the ancilla measurement succeeds, selecting the
    /// |1> outcome that signals a valid solution. Higher is better.
    /// For well-conditioned systems this is close to 1; for ill-conditioned
    /// systems it can be very small, requiring many repetitions.
    pub success_probability: f64,

    /// Eigenvalues of A discovered during the QPE phase.
    pub eigenvalues: Vec<f64>,

    /// Condition number of A: ratio of largest to smallest eigenvalue
    /// magnitude. Large condition numbers indicate ill-conditioned systems
    /// where the HHL success probability will be low.
    pub condition_number: f64,

    /// Number of clock qubits used in the QPE phase.
    pub num_clock_qubits: usize,
}

// ============================================================
// HHL SOLVER
// ============================================================

/// Harrow-Hassidim-Lloyd solver for quantum linear systems.
///
/// Simulates the HHL algorithm classically via eigendecomposition.
/// The `num_clock_qubits` parameter controls eigenvalue resolution
/// (more qubits = finer resolution of the eigenvalue spectrum).
/// The `scaling` parameter sets the constant C in the controlled
/// rotation step (ancilla amplitude = C / lambda_j).
///
/// # Example
///
/// ```
/// use num_complex::Complex64 as C64;
/// use nqpu_metal::algorithms::hhl::{HHLSolver, HHLResult};
///
/// let a = vec![
///     vec![C64::new(2.0, 0.0), C64::new(0.0, 0.0)],
///     vec![C64::new(0.0, 0.0), C64::new(4.0, 0.0)],
/// ];
/// let b = vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
///
/// let solver = HHLSolver::new(4);
/// let result = solver.solve(&a, &b);
///
/// // Solution should be proportional to [1/2, 1/4]
/// assert!(result.success_probability > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct HHLSolver {
    /// Number of clock (evaluation) qubits for the QPE phase.
    /// More qubits give finer eigenvalue resolution.
    pub num_clock_qubits: usize,

    /// Scale factor C for the controlled rotation step.
    /// The ancilla rotation amplitude is C / lambda_j.
    /// If <= 0.0, auto-scaling is used (C = min |lambda_j|).
    pub scaling: f64,
}

impl HHLSolver {
    /// Create a new HHL solver with automatic scaling.
    ///
    /// The scaling factor C is set to the smallest eigenvalue magnitude,
    /// which maximises the success probability while ensuring all
    /// rotation amplitudes remain <= 1.
    ///
    /// # Arguments
    ///
    /// * `num_clock_qubits` - Number of QPE evaluation qubits.
    pub fn new(num_clock_qubits: usize) -> Self {
        Self {
            num_clock_qubits,
            scaling: 0.0, // 0 signals auto-scale
        }
    }

    /// Create a new HHL solver with explicit scaling.
    ///
    /// # Arguments
    ///
    /// * `num_clock_qubits` - Number of QPE evaluation qubits.
    /// * `scaling` - Explicit scale factor C. Must satisfy C <= min |lambda_j|
    ///   for the algorithm to produce valid results.
    pub fn with_scaling(num_clock_qubits: usize, scaling: f64) -> Self {
        Self {
            num_clock_qubits,
            scaling,
        }
    }

    /// Solve the linear system Ax = b using the HHL algorithm.
    ///
    /// Produces a quantum state |x> proportional to A^{-1}|b>.
    ///
    /// # Arguments
    ///
    /// * `a_matrix` - The matrix A as Vec<Vec<Complex64>>. Must be square
    ///   and Hermitian (A[i][j] == conj(A[j][i])).
    /// * `b_vector` - The right-hand side vector b. Length must equal
    ///   the dimension of A.
    ///
    /// # Panics
    ///
    /// Panics if A is not square, dimensions don't match b, or A has
    /// a zero eigenvalue (singular matrix).
    pub fn solve(&self, a_matrix: &[Vec<C64>], b_vector: &[C64]) -> HHLResult {
        let dim = a_matrix.len();

        // Validate dimensions
        assert!(dim > 0, "Matrix A must be non-empty");
        for (i, row) in a_matrix.iter().enumerate() {
            assert_eq!(
                row.len(),
                dim,
                "Matrix A must be square: row {} has length {}, expected {}",
                i,
                row.len(),
                dim
            );
        }
        assert_eq!(
            b_vector.len(),
            dim,
            "Vector b length {} must match matrix dimension {}",
            b_vector.len(),
            dim
        );

        // Convert to flat representation
        let a_flat = vv_to_flat(a_matrix, dim);

        // Verify Hermitian (warning-level: we proceed even if not perfectly Hermitian)
        debug_assert!(
            is_hermitian(&a_flat, dim, 1e-8),
            "Matrix A should be Hermitian for HHL"
        );

        // Step 1: Eigendecomposition A = V D V^dag
        let (eigenvalues, eigenvectors) = eigendecompose_hermitian(&a_flat, dim);

        // Validate no zero eigenvalues (A must be invertible)
        let min_eigenval_abs = eigenvalues
            .iter()
            .map(|&e| e.abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_eigenval_abs > 1e-14,
            "Matrix A is singular: smallest eigenvalue magnitude is {:.2e}",
            min_eigenval_abs
        );

        // Condition number
        let max_eigenval_abs = eigenvalues
            .iter()
            .map(|&e| e.abs())
            .fold(0.0_f64, f64::max);
        let condition_number = max_eigenval_abs / min_eigenval_abs;

        // Step 2: Express |b> in eigenbasis
        // beta_j = <v_j | b> where v_j is the j-th eigenvector (column of V)
        let mut beta = vec![c64_zero(); dim];
        for j in 0..dim {
            let mut s = c64_zero();
            for i in 0..dim {
                // eigenvectors[j] is column j, element i
                s += eigenvectors[j * dim + i].conj() * b_vector[i];
            }
            beta[j] = s;
        }

        // Step 3: Determine scaling factor C
        let scaling_c = if self.scaling > 0.0 {
            self.scaling
        } else {
            // Auto-scale: C = min |lambda_j| so that max |C/lambda_j| = 1
            min_eigenval_abs
        };

        // Step 4: Compute A^{-1}b in the eigenbasis.
        // In the eigenbasis, (A^{-1}b)_j = beta_j / lambda_j.
        // Separately track the HHL success probability using the scaling C:
        // the controlled rotation amplitude is C/lambda_j, so
        // P(success) = sum_j |C/lambda_j|^2 |beta_j|^2 / ||b||^2.
        let mut x_eigen = vec![c64_zero(); dim];
        let mut success_prob_sum = 0.0_f64;

        for j in 0..dim {
            let lambda_j = eigenvalues[j];
            x_eigen[j] = beta[j] / c64(lambda_j, 0.0);
            let ratio = scaling_c / lambda_j;
            success_prob_sum += ratio * ratio * beta[j].norm_sqr();
        }

        // Step 5: Transform back to computational basis: x = V * x_eigen
        // x_i = sum_j V[i][j] * x_eigen[j]
        // where eigenvectors[j * dim + i] = V[i][j] (component i of eigenvector j)
        let mut x_unnorm = vec![c64_zero(); dim];
        for i in 0..dim {
            let mut s = c64_zero();
            for j in 0..dim {
                s += eigenvectors[j * dim + i] * x_eigen[j];
            }
            x_unnorm[i] = s;
        }

        // Step 6: Compute norm ||A^{-1}b|| and normalize to unit quantum state
        let solution_norm = x_unnorm.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

        let solution = if solution_norm > 1e-15 {
            x_unnorm.iter().map(|c| c / solution_norm).collect()
        } else {
            x_unnorm
        };

        // Success probability: P(ancilla = |1>) = sum_j |C/lambda_j|^2 |beta_j|^2 / ||b||^2.
        let b_norm_sq: f64 = b_vector.iter().map(|c| c.norm_sqr()).sum();
        let success_probability = if b_norm_sq > 1e-15 {
            (success_prob_sum / b_norm_sq).min(1.0)
        } else {
            0.0
        };

        HHLResult {
            solution,
            solution_norm,
            success_probability,
            eigenvalues,
            condition_number,
            num_clock_qubits: self.num_clock_qubits,
        }
    }
}

// ============================================================
// EIGENDECOMPOSITION (Jacobi Method)
// ============================================================

/// Compute eigenvalues and eigenvectors of a Hermitian matrix using
/// the Jacobi eigenvalue algorithm.
///
/// The Jacobi method iteratively applies 2x2 rotations (Givens rotations)
/// to drive off-diagonal elements to zero. Each rotation zeroes out one
/// off-diagonal pair, and after sufficient sweeps the matrix converges
/// to a diagonal form. The accumulated rotations form the eigenvector matrix.
///
/// # Arguments
///
/// * `matrix` - Flat row-major Hermitian matrix of dimension dim x dim.
/// * `dim` - Matrix dimension.
///
/// # Returns
///
/// A tuple of (eigenvalues, eigenvector_matrix) where the eigenvector matrix
/// is stored in flat row-major format with eigenvectors as columns. That is,
/// eigenvectors[j * dim + i] is the i-th component of the j-th eigenvector.
pub fn eigendecompose_hermitian(matrix: &[C64], dim: usize) -> (Vec<f64>, Vec<C64>) {
    // Work on a mutable copy of the matrix
    let mut a = matrix.to_vec();

    // Initialize eigenvector matrix V as identity (columns will become eigenvectors).
    // V is stored row-major: V[i * dim + j] = component i of eigenvector j.
    let mut v = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        v[i * dim + i] = c64_one();
    }

    let max_sweeps = 200;
    let tol = 1e-14;

    for _sweep in 0..max_sweeps {
        // Compute off-diagonal Frobenius norm (upper triangle only)
        let mut off_diag_norm = 0.0_f64;
        for i in 0..dim {
            for j in (i + 1)..dim {
                off_diag_norm += a[i * dim + j].norm_sqr();
            }
        }

        if off_diag_norm.sqrt() < tol {
            break;
        }

        // Sweep through all off-diagonal pairs (p < q)
        for p in 0..dim {
            for q in (p + 1)..dim {
                let a_pq = a[p * dim + q];
                if a_pq.norm_sqr() < tol * tol {
                    continue;
                }

                // ── Two-step complex Jacobi rotation ──────────────────────
                //
                // For a Hermitian matrix A with complex off-diagonal A[p][q],
                // we decompose the rotation into:
                //   1. A diagonal phase matrix D that makes A[p][q] real
                //   2. A standard real Jacobi rotation R that zeroes it
                //
                // Phase extraction: A[p][q] = |A[p][q]| * e^{i*alpha}
                // D = diag(1, e^{-i*alpha}) at position q.
                // After D^H A D, element (p,q) becomes real = |A[p][q]|.
                //
                // Standard Jacobi rotation R = [[c, s], [-s, c]]
                // (Golub & Van Loan convention: R^T M R is diagonal for
                // symmetric M when tau = (M_qq - M_pp) / (2 M_pq)).
                //
                // Combined: G = D * R
                //   G[p][p] = c
                //   G[p][q] = s
                //   G[q][p] = -s * e^{-i*alpha}
                //   G[q][q] = c * e^{-i*alpha}

                let a_pp = a[p * dim + p].re;
                let a_qq = a[q * dim + q].re;
                let abs_pq = a_pq.norm();

                if abs_pq < tol {
                    continue;
                }

                let e_ia = a_pq / abs_pq; // e^{i*alpha}
                let e_nia = e_ia.conj(); // e^{-i*alpha}

                // Real Jacobi rotation for [[a_pp, |a_pq|], [|a_pq|, a_qq]]
                let tau = (a_qq - a_pp) / (2.0 * abs_pq);
                let t = if tau.abs() > 1e15 {
                    1.0 / (2.0 * tau)
                } else {
                    let sign_tau = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign_tau / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // G = D * R entries
                let g_pp = c64(c, 0.0);
                let g_pq = c64(s, 0.0);
                let g_qp = c64(-s, 0.0) * e_nia;
                let g_qq = c64(c, 0.0) * e_nia;

                // G^H entries: G^H[i][j] = conj(G[j][i])
                let gh_pp = g_pp.conj(); // c
                let gh_pq = g_qp.conj(); // -s * conj(e^{-ia}) = -s * e^{ia}
                let gh_qp = g_pq.conj(); // s
                let gh_qq = g_qq.conj(); // c * e^{ia}

                // A <- G^H * A * G
                // Step 1: B = A * G (right-multiply, update columns p and q)
                for k in 0..dim {
                    let akp = a[k * dim + p];
                    let akq = a[k * dim + q];
                    a[k * dim + p] = akp * g_pp + akq * g_qp;
                    a[k * dim + q] = akp * g_pq + akq * g_qq;
                }

                // Step 2: A' = G^H * B (left-multiply, update rows p and q)
                for k in 0..dim {
                    let bpk = a[p * dim + k];
                    let bqk = a[q * dim + k];
                    a[p * dim + k] = gh_pp * bpk + gh_pq * bqk;
                    a[q * dim + k] = gh_qp * bpk + gh_qq * bqk;
                }

                // Force exact zeros on the (p,q) and (q,p) entries
                a[p * dim + q] = c64_zero();
                a[q * dim + p] = c64_zero();

                // Accumulate eigenvectors: V <- V * G
                for k in 0..dim {
                    let vkp = v[k * dim + p];
                    let vkq = v[k * dim + q];
                    v[k * dim + p] = vkp * g_pp + vkq * g_qp;
                    v[k * dim + q] = vkp * g_pq + vkq * g_qq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..dim).map(|i| a[i * dim + i].re).collect();

    // V is stored row-major with eigenvectors as columns:
    //   V[i * dim + j] = component i of eigenvector j.
    // The caller convention is eigenvectors[j * dim + i] = component i of eigenvector j.
    // So we transpose V.
    let mut eigvecs = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            eigvecs[j * dim + i] = v[i * dim + j];
        }
    }

    (eigenvalues, eigvecs)
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Check if a matrix is Hermitian: A[i][j] == conj(A[j][i]) within tolerance.
///
/// A Hermitian matrix satisfies A = A^dag, meaning each element equals the
/// complex conjugate of its transpose. The diagonal must be real.
///
/// # Arguments
///
/// * `matrix` - Flat row-major matrix of dimension dim x dim.
/// * `dim` - Matrix dimension.
/// * `tol` - Tolerance for floating-point comparison.
pub fn is_hermitian(matrix: &[C64], dim: usize, tol: f64) -> bool {
    for i in 0..dim {
        // Diagonal elements must be real
        if matrix[i * dim + i].im.abs() > tol {
            return false;
        }
        for j in (i + 1)..dim {
            let a_ij = matrix[i * dim + j];
            let a_ji = matrix[j * dim + i];
            if (a_ij.re - a_ji.re).abs() > tol || (a_ij.im + a_ji.im).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Classical solver for Ax = b using Gaussian elimination with partial pivoting.
///
/// This provides a reference solution for validating the HHL results.
/// The solution is returned as an unnormalized vector (not a quantum state).
///
/// # Arguments
///
/// * `a_matrix` - The matrix A as Vec<Vec<Complex64>>.
/// * `b_vector` - The right-hand side vector b.
///
/// # Panics
///
/// Panics if the matrix is singular (zero pivot encountered).
pub fn classical_solve(a_matrix: &[Vec<C64>], b_vector: &[C64]) -> Vec<C64> {
    let dim = a_matrix.len();

    // Build augmented matrix [A | b]
    let mut aug = vec![vec![c64_zero(); dim + 1]; dim];
    for i in 0..dim {
        for j in 0..dim {
            aug[i][j] = a_matrix[i][j];
        }
        aug[i][dim] = b_vector[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..dim {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].norm();
        for row in (col + 1)..dim {
            let val = aug[row][col].norm();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        assert!(
            max_val > 1e-14,
            "Singular matrix encountered at column {}",
            col
        );

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate below pivot
        let pivot = aug[col][col];
        for row in (col + 1)..dim {
            let factor = aug[row][col] / pivot;
            for j in col..=dim {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![c64_zero(); dim];
    for i in (0..dim).rev() {
        let mut s = aug[i][dim];
        for j in (i + 1)..dim {
            s -= aug[i][j] * x[j];
        }
        x[i] = s / aug[i][i];
    }

    x
}

/// Standard matrix-vector product for Vec<Vec<C64>> format.
///
/// Computes y = A * x where A is dim x dim and x has length dim.
pub fn matrix_vector_product(matrix: &[Vec<C64>], vector: &[C64]) -> Vec<C64> {
    let dim = matrix.len();
    let mut result = vec![c64_zero(); dim];
    for i in 0..dim {
        let mut s = c64_zero();
        for j in 0..dim {
            s += matrix[i][j] * vector[j];
        }
        result[i] = s;
    }
    result
}

/// Compute the cosine similarity between two complex vectors.
///
/// Returns |<u|v>| / (||u|| * ||v||), which equals 1.0 when the
/// vectors are parallel (same direction up to a global phase).
fn cosine_similarity(u: &[C64], v: &[C64]) -> f64 {
    let mut dot = c64_zero();
    for i in 0..u.len() {
        dot += u[i].conj() * v[i];
    }
    let norm_u = u.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let norm_v = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm_u < 1e-15 || norm_v < 1e-15 {
        return 0.0;
    }
    dot.norm() / (norm_u * norm_v)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // 1. Simple 2x2 diagonal system
    // ----------------------------------------------------------
    #[test]
    fn test_simple_2x2_diagonal() {
        // A = [[2, 0], [0, 4]], b = [1, 1]
        // A^{-1}b = [1/2, 1/4], normalized = [2/sqrt(5), 1/sqrt(5)]
        let a = vec![
            vec![c64(2.0, 0.0), c64_zero()],
            vec![c64_zero(), c64(4.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(1.0, 0.0)];

        let solver = HHLSolver::new(4);
        let result = solver.solve(&a, &b);

        // Check solution direction matches [1/2, 1/4]
        let expected_unnorm = vec![c64(0.5, 0.0), c64(0.25, 0.0)];
        let sim = cosine_similarity(&result.solution, &expected_unnorm);
        assert!(
            sim > 0.999,
            "Solution direction should match A^{{-1}}b: cosine similarity = {:.6}",
            sim
        );

        // Check solution norm
        let expected_norm = (0.25_f64 + 0.0625).sqrt(); // sqrt(1/4 + 1/16)
        assert!(
            (result.solution_norm - expected_norm).abs() < 1e-10,
            "Solution norm {:.6} should match {:.6}",
            result.solution_norm,
            expected_norm
        );

        assert!(result.success_probability > 0.0);
        assert!(result.success_probability <= 1.0);
    }

    // ----------------------------------------------------------
    // 2. Identity matrix: A = I, solution = b
    // ----------------------------------------------------------
    #[test]
    fn test_identity_matrix() {
        let dim = 3;
        let a: Vec<Vec<C64>> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| if i == j { c64_one() } else { c64_zero() })
                    .collect()
            })
            .collect();

        let b = vec![c64(1.0, 0.0), c64_zero(), c64_zero()];

        let solver = HHLSolver::new(4);
        let result = solver.solve(&a, &b);

        // A^{-1}b = b, so solution should be [1, 0, 0]
        let sim = cosine_similarity(&result.solution, &b);
        assert!(
            sim > 0.999,
            "Identity solve: solution should equal b, cosine similarity = {:.6}",
            sim
        );

        // Norm should be ||b|| = 1
        assert!(
            (result.solution_norm - 1.0).abs() < 1e-10,
            "Solution norm should be 1.0, got {:.6}",
            result.solution_norm
        );

        // Condition number should be 1
        assert!(
            (result.condition_number - 1.0).abs() < 1e-10,
            "Identity matrix condition number should be 1.0, got {:.6}",
            result.condition_number
        );
    }

    // ----------------------------------------------------------
    // 3. Hermitian check
    // ----------------------------------------------------------
    #[test]
    fn test_is_hermitian_positive() {
        // Hermitian: [[1, 1+i], [1-i, 2]]
        let h = vec![
            c64(1.0, 0.0),
            c64(1.0, 1.0),
            c64(1.0, -1.0),
            c64(2.0, 0.0),
        ];
        assert!(is_hermitian(&h, 2, 1e-10));
    }

    #[test]
    fn test_is_hermitian_negative() {
        // Not Hermitian: [[1, 1+i], [1+i, 2]] (off-diag not conjugate)
        let not_h = vec![
            c64(1.0, 0.0),
            c64(1.0, 1.0),
            c64(1.0, 1.0), // should be (1, -1) for Hermitian
            c64(2.0, 0.0),
        ];
        assert!(!is_hermitian(&not_h, 2, 1e-10));
    }

    #[test]
    fn test_is_hermitian_imaginary_diagonal() {
        // Not Hermitian: diagonal has imaginary part
        let bad = vec![
            c64(1.0, 0.5), // imaginary diagonal
            c64_zero(),
            c64_zero(),
            c64(2.0, 0.0),
        ];
        assert!(!is_hermitian(&bad, 2, 1e-10));
    }

    // ----------------------------------------------------------
    // 4. Eigendecomposition of Pauli Z
    // ----------------------------------------------------------
    #[test]
    fn test_eigendecomposition_pauli_z() {
        // Pauli Z = [[1, 0], [0, -1]]
        let z = vec![c64(1.0, 0.0), c64_zero(), c64_zero(), c64(-1.0, 0.0)];

        let (eigenvalues, eigenvectors) = eigendecompose_hermitian(&z, 2);

        // Eigenvalues should be +1 and -1 (in some order)
        let mut sorted_evals = eigenvalues.clone();
        sorted_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(
            (sorted_evals[0] - (-1.0)).abs() < 1e-10,
            "Smallest eigenvalue should be -1, got {:.6}",
            sorted_evals[0]
        );
        assert!(
            (sorted_evals[1] - 1.0).abs() < 1e-10,
            "Largest eigenvalue should be 1, got {:.6}",
            sorted_evals[1]
        );

        // Verify A * v_j = lambda_j * v_j for each eigenpair
        for j in 0..2 {
            let v_j: Vec<C64> = (0..2).map(|i| eigenvectors[j * 2 + i]).collect();
            let av_j = matvec(&z, &v_j, 2);
            for i in 0..2 {
                let expected = v_j[i] * c64(eigenvalues[j], 0.0);
                assert!(
                    (av_j[i] - expected).norm() < 1e-10,
                    "A*v[{}][{}] = ({:.6}, {:.6}), expected ({:.6}, {:.6})",
                    j,
                    i,
                    av_j[i].re,
                    av_j[i].im,
                    expected.re,
                    expected.im
                );
            }
        }
    }

    // ----------------------------------------------------------
    // 5. Condition number
    // ----------------------------------------------------------
    #[test]
    fn test_condition_number() {
        // Diagonal matrix with eigenvalues [1, 100]
        let a = vec![
            vec![c64(1.0, 0.0), c64_zero()],
            vec![c64_zero(), c64(100.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(1.0, 0.0)];

        let solver = HHLSolver::new(4);
        let result = solver.solve(&a, &b);

        assert!(
            (result.condition_number - 100.0).abs() < 1e-8,
            "Condition number should be 100, got {:.6}",
            result.condition_number
        );
    }

    // ----------------------------------------------------------
    // 6. Solution accuracy vs classical solver
    // ----------------------------------------------------------
    #[test]
    fn test_solution_accuracy_vs_classical() {
        // A = [[3, 1], [1, 2]], b = [1, 0]
        let a = vec![
            vec![c64(3.0, 0.0), c64(1.0, 0.0)],
            vec![c64(1.0, 0.0), c64(2.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64_zero()];

        let solver = HHLSolver::new(6);
        let result = solver.solve(&a, &b);

        let classical = classical_solve(&a, &b);

        // The HHL solution is a normalized quantum state proportional to A^{-1}b.
        // Compare directions using cosine similarity.
        let sim = cosine_similarity(&result.solution, &classical);
        assert!(
            sim > 0.99,
            "HHL solution should match classical solution direction: cosine similarity = {:.6}",
            sim
        );
    }

    // ----------------------------------------------------------
    // 7. Success probability bounds
    // ----------------------------------------------------------
    #[test]
    fn test_success_probability_bounds() {
        // Well-conditioned system: identity
        let a_good = vec![
            vec![c64(1.0, 0.0), c64_zero()],
            vec![c64_zero(), c64(1.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(1.0, 0.0)];

        let solver = HHLSolver::new(4);
        let result_good = solver.solve(&a_good, &b);

        assert!(
            result_good.success_probability > 0.0,
            "Success probability must be positive"
        );
        assert!(
            result_good.success_probability <= 1.0,
            "Success probability must be <= 1.0, got {:.6}",
            result_good.success_probability
        );

        // For identity matrix with auto-scaling, success probability should be 1.0
        assert!(
            (result_good.success_probability - 1.0).abs() < 1e-10,
            "Identity matrix should have success probability 1.0, got {:.6}",
            result_good.success_probability
        );
    }

    // ----------------------------------------------------------
    // 8. 3x3 diagonal system
    // ----------------------------------------------------------
    #[test]
    fn test_3x3_diagonal() {
        // A = diag(1, 2, 3), b = [1, 1, 1]
        // A^{-1}b = [1, 1/2, 1/3]
        let a = vec![
            vec![c64(1.0, 0.0), c64_zero(), c64_zero()],
            vec![c64_zero(), c64(2.0, 0.0), c64_zero()],
            vec![c64_zero(), c64_zero(), c64(3.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(1.0, 0.0), c64(1.0, 0.0)];

        let solver = HHLSolver::new(4);
        let result = solver.solve(&a, &b);

        let expected = vec![c64(1.0, 0.0), c64(0.5, 0.0), c64(1.0 / 3.0, 0.0)];

        let sim = cosine_similarity(&result.solution, &expected);
        assert!(
            sim > 0.999,
            "3x3 diagonal solve: cosine similarity = {:.6}",
            sim
        );

        // Verify eigenvalues
        let mut evals = result.eigenvalues.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (evals[0] - 1.0).abs() < 1e-10,
            "Eigenvalue 0 should be 1, got {:.6}",
            evals[0]
        );
        assert!(
            (evals[1] - 2.0).abs() < 1e-10,
            "Eigenvalue 1 should be 2, got {:.6}",
            evals[1]
        );
        assert!(
            (evals[2] - 3.0).abs() < 1e-10,
            "Eigenvalue 2 should be 3, got {:.6}",
            evals[2]
        );
    }

    // ----------------------------------------------------------
    // 9. Off-diagonal Hermitian system
    // ----------------------------------------------------------
    #[test]
    fn test_off_diagonal_hermitian() {
        // A = [[2, 1+i], [1-i, 3]], b = [1, 0]
        let a = vec![
            vec![c64(2.0, 0.0), c64(1.0, 1.0)],
            vec![c64(1.0, -1.0), c64(3.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64_zero()];

        // Verify A is Hermitian
        let a_flat = vv_to_flat(&a, 2);
        assert!(is_hermitian(&a_flat, 2, 1e-10));

        let solver = HHLSolver::new(6);
        let result = solver.solve(&a, &b);

        // Compare with classical solution
        let classical = classical_solve(&a, &b);
        let sim = cosine_similarity(&result.solution, &classical);
        assert!(
            sim > 0.99,
            "Off-diagonal Hermitian: cosine similarity = {:.6}",
            sim
        );

        // Verify by computing A * x and checking it matches b direction
        let x_actual: Vec<C64> = result
            .solution
            .iter()
            .map(|c| c * result.solution_norm)
            .collect();
        let ax = matrix_vector_product(&a, &x_actual);
        let residual_sim = cosine_similarity(&ax, &b);
        assert!(
            residual_sim > 0.99,
            "A * x should be proportional to b: cosine similarity = {:.6}",
            residual_sim
        );
    }

    // ----------------------------------------------------------
    // 10. Ill-conditioned system
    // ----------------------------------------------------------
    #[test]
    fn test_ill_conditioned_system() {
        // A = diag(0.01, 100) -> condition number = 10000
        let a = vec![
            vec![c64(0.01, 0.0), c64_zero()],
            vec![c64_zero(), c64(100.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(1.0, 0.0)];

        let solver = HHLSolver::new(4);
        let result = solver.solve(&a, &b);

        assert!(
            (result.condition_number - 10000.0).abs() < 1e-4,
            "Condition number should be 10000, got {:.6}",
            result.condition_number
        );

        // Success probability should be low for ill-conditioned systems
        // with auto-scaling (C = min eigenvalue = 0.01)
        assert!(
            result.success_probability < 1.0,
            "Ill-conditioned system should have success_probability < 1.0, got {:.6}",
            result.success_probability
        );

        // Solution should still be correct in direction
        let expected = vec![c64(100.0, 0.0), c64(0.01, 0.0)];
        let sim = cosine_similarity(&result.solution, &expected);
        assert!(
            sim > 0.999,
            "Ill-conditioned solution should still be directionally correct: cosine similarity = {:.6}",
            sim
        );
    }

    // ----------------------------------------------------------
    // 11. Classical solver validation
    // ----------------------------------------------------------
    #[test]
    fn test_classical_solver_simple() {
        // A = [[1, 0], [0, 2]], b = [3, 4] -> x = [3, 2]
        let a = vec![
            vec![c64(1.0, 0.0), c64_zero()],
            vec![c64_zero(), c64(2.0, 0.0)],
        ];
        let b = vec![c64(3.0, 0.0), c64(4.0, 0.0)];

        let x = classical_solve(&a, &b);

        assert!(
            (x[0] - c64(3.0, 0.0)).norm() < 1e-10,
            "x[0] should be 3, got ({:.6}, {:.6})",
            x[0].re,
            x[0].im
        );
        assert!(
            (x[1] - c64(2.0, 0.0)).norm() < 1e-10,
            "x[1] should be 2, got ({:.6}, {:.6})",
            x[1].re,
            x[1].im
        );
    }

    #[test]
    fn test_classical_solver_dense() {
        // A = [[2, 1], [1, 3]], b = [1, 1]
        // Solution: x = [2/5, 1/5] (from 2x+y=1, x+3y=1)
        let a = vec![
            vec![c64(2.0, 0.0), c64(1.0, 0.0)],
            vec![c64(1.0, 0.0), c64(3.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(1.0, 0.0)];

        let x = classical_solve(&a, &b);

        assert!(
            (x[0].re - 0.4).abs() < 1e-10,
            "x[0] should be 0.4, got {:.6}",
            x[0].re
        );
        assert!(
            (x[1].re - 0.2).abs() < 1e-10,
            "x[1] should be 0.2, got {:.6}",
            x[1].re
        );
    }

    #[test]
    fn test_classical_solver_complex() {
        // A = [[1, i], [-i, 1]], b = [1, 0]
        // A is Hermitian. det(A) = 1-(-i)(i) = 1-1 = 0... no.
        // det = 1*1 - i*(-i) = 1 - (-i^2) = 1 - 1 = 0. Singular!
        // Use A = [[2, i], [-i, 2]] instead. det = 4 - 1 = 3.
        // A^{-1} = (1/3)[[2, -i], [i, 2]]
        // A^{-1}b = (1/3)[2, i]
        let a = vec![
            vec![c64(2.0, 0.0), c64(0.0, 1.0)],
            vec![c64(0.0, -1.0), c64(2.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64_zero()];

        let x = classical_solve(&a, &b);

        assert!(
            (x[0] - c64(2.0 / 3.0, 0.0)).norm() < 1e-10,
            "x[0] should be 2/3, got ({:.6}, {:.6})",
            x[0].re,
            x[0].im
        );
        assert!(
            (x[1] - c64(0.0, 1.0 / 3.0)).norm() < 1e-10,
            "x[1] should be i/3, got ({:.6}, {:.6})",
            x[1].re,
            x[1].im
        );
    }

    // ----------------------------------------------------------
    // 12. Matrix-vector product
    // ----------------------------------------------------------
    #[test]
    fn test_matrix_vector_product() {
        let a = vec![
            vec![c64(1.0, 0.0), c64(2.0, 0.0)],
            vec![c64(3.0, 0.0), c64(4.0, 0.0)],
        ];
        let v = vec![c64(1.0, 0.0), c64(1.0, 0.0)];

        let result = matrix_vector_product(&a, &v);

        assert!(
            (result[0].re - 3.0).abs() < 1e-10,
            "Result[0] should be 3, got {:.6}",
            result[0].re
        );
        assert!(
            (result[1].re - 7.0).abs() < 1e-10,
            "Result[1] should be 7, got {:.6}",
            result[1].re
        );
    }

    // ----------------------------------------------------------
    // 13. Eigendecomposition reconstruction
    // ----------------------------------------------------------
    #[test]
    fn test_eigendecomposition_reconstruction() {
        // Verify V * D * V^dag = A for a non-trivial Hermitian matrix
        let a_flat = vec![
            c64(3.0, 0.0),
            c64(1.0, 1.0),
            c64(1.0, -1.0),
            c64(2.0, 0.0),
        ];

        let (eigenvalues, eigenvectors) = eigendecompose_hermitian(&a_flat, 2);

        // Reconstruct A = V * D * V^dag
        // V[i][j] = eigenvectors[j * 2 + i] (i-th component of j-th eigenvector)
        // So V_flat[i * 2 + j] = eigenvectors[j * 2 + i]
        let mut v_flat = vec![c64_zero(); 4];
        for i in 0..2 {
            for j in 0..2 {
                v_flat[i * 2 + j] = eigenvectors[j * 2 + i];
            }
        }

        // D = diag(eigenvalues)
        let mut d_flat = vec![c64_zero(); 4];
        for i in 0..2 {
            d_flat[i * 2 + i] = c64(eigenvalues[i], 0.0);
        }

        let v_dag = adjoint(&v_flat, 2);
        let vd = matmul(&v_flat, &d_flat, 2);
        let reconstructed = matmul(&vd, &v_dag, 2);

        for i in 0..4 {
            assert!(
                (reconstructed[i] - a_flat[i]).norm() < 1e-8,
                "Reconstructed[{}] = ({:.6}, {:.6}), expected ({:.6}, {:.6})",
                i,
                reconstructed[i].re,
                reconstructed[i].im,
                a_flat[i].re,
                a_flat[i].im
            );
        }
    }

    // ----------------------------------------------------------
    // 14. HHL with explicit scaling
    // ----------------------------------------------------------
    #[test]
    fn test_hhl_with_explicit_scaling() {
        let a = vec![
            vec![c64(2.0, 0.0), c64_zero()],
            vec![c64_zero(), c64(4.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(1.0, 0.0)];

        // Use scaling = 1.0 (smaller than min eigenvalue = 2)
        let solver = HHLSolver::with_scaling(4, 1.0);
        let result = solver.solve(&a, &b);

        // Solution direction should still be correct
        let expected = vec![c64(0.5, 0.0), c64(0.25, 0.0)];
        let sim = cosine_similarity(&result.solution, &expected);
        assert!(
            sim > 0.999,
            "Explicit scaling: cosine similarity = {:.6}",
            sim
        );

        // Success probability should be lower with smaller C
        assert!(
            result.success_probability > 0.0,
            "Success probability should be positive"
        );
        assert!(
            result.success_probability <= 1.0,
            "Success probability should be <= 1.0"
        );
    }

    // ----------------------------------------------------------
    // 15. 4x4 system
    // ----------------------------------------------------------
    #[test]
    fn test_4x4_system() {
        // A = diag(1, 2, 3, 4), b = [1, 1, 1, 1]
        // A^{-1}b = [1, 1/2, 1/3, 1/4]
        let dim = 4;
        let a: Vec<Vec<C64>> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        if i == j {
                            c64((i + 1) as f64, 0.0)
                        } else {
                            c64_zero()
                        }
                    })
                    .collect()
            })
            .collect();
        let b: Vec<C64> = vec![c64_one(); dim];

        let solver = HHLSolver::new(6);
        let result = solver.solve(&a, &b);

        let expected: Vec<C64> = (1..=dim)
            .map(|i| c64(1.0 / i as f64, 0.0))
            .collect();

        let sim = cosine_similarity(&result.solution, &expected);
        assert!(
            sim > 0.999,
            "4x4 diagonal solve: cosine similarity = {:.6}",
            sim
        );
    }

    // ----------------------------------------------------------
    // 16. Verify A * x_actual = b for HHL solution
    // ----------------------------------------------------------
    #[test]
    fn test_hhl_residual_check() {
        let a = vec![
            vec![c64(4.0, 0.0), c64(1.0, 0.0)],
            vec![c64(1.0, 0.0), c64(3.0, 0.0)],
        ];
        let b = vec![c64(1.0, 0.0), c64(2.0, 0.0)];

        let solver = HHLSolver::new(6);
        let result = solver.solve(&a, &b);

        // Recover actual (unnormalized) solution
        let x_actual: Vec<C64> = result
            .solution
            .iter()
            .map(|c| c * result.solution_norm)
            .collect();

        // Compute A * x_actual
        let ax = matrix_vector_product(&a, &x_actual);

        // The scaling factor C means A * x_actual = (C / lambda_scaling) * b
        // For auto-scaling, A * x_actual should be proportional to b
        let residual_sim = cosine_similarity(&ax, &b);
        assert!(
            residual_sim > 0.999,
            "A * x should be proportional to b: cosine similarity = {:.6}",
            residual_sim
        );
    }

    // ----------------------------------------------------------
    // 17. Eigendecomposition of 3x3 matrix
    // ----------------------------------------------------------
    #[test]
    fn test_eigendecomposition_3x3() {
        // Diagonal 3x3: eigenvalues should be 1, 5, 9
        let a = vec![
            c64(1.0, 0.0), c64_zero(), c64_zero(),
            c64_zero(), c64(5.0, 0.0), c64_zero(),
            c64_zero(), c64_zero(), c64(9.0, 0.0),
        ];

        let (eigenvalues, _eigenvectors) = eigendecompose_hermitian(&a, 3);

        let mut sorted_evals = eigenvalues.clone();
        sorted_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(
            (sorted_evals[0] - 1.0).abs() < 1e-10,
            "Eigenvalue 0: expected 1, got {:.6}",
            sorted_evals[0]
        );
        assert!(
            (sorted_evals[1] - 5.0).abs() < 1e-10,
            "Eigenvalue 1: expected 5, got {:.6}",
            sorted_evals[1]
        );
        assert!(
            (sorted_evals[2] - 9.0).abs() < 1e-10,
            "Eigenvalue 2: expected 9, got {:.6}",
            sorted_evals[2]
        );
    }
}
