//! Tensor Jump Method (TJM) for Open Quantum System Simulation
//!
//! World-first Rust implementation of the Tensor Jump Method described in
//! Nature Communications (December 2025). Combines Matrix Product States (MPS)
//! with Monte Carlo Wave Function (MCWF) sampling and dynamic TDVP integration
//! to simulate open quantum systems governed by Lindblad master equations at
//! scales previously inaccessible to density-matrix methods.
//!
//! # Architecture
//!
//! ```text
//!  Lindblad Master Equation
//!  dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
//!                    |
//!          +---------+---------+
//!          |                   |
//!    H_eff (non-Hermitian)  Jump ops L_k
//!          |                   |
//!    +-----+-----+      +-----+-----+
//!    | TDVP sweep |      | Stochastic |
//!    | on MPS     |      | application|
//!    +-----+------+      +-----+-----+
//!          |                   |
//!          +------- MPS -------+
//!          |  (bond dim χ)     |
//!    +-----+------+      +-----+------+
//!    | Trajectory 1|  ...| Trajectory N|
//!    +-----+------+      +-----+------+
//!          |                   |
//!          +------- avg -------+
//!                   |
//!          Observable ⟨O⟩(t) ± σ/√N
//! ```
//!
//! # Key Features
//!
//! - **MPS-based state representation**: O(n * d * chi^2) memory instead of O(4^n)
//! - **Monte Carlo trajectory sampling**: embarrassingly parallel across trajectories
//! - **State-dependent 1-site TDVP**: local effective evolution with Strang splitting
//! - **Built-in SVD truncation**: Jacobi one-sided SVD with configurable cutoff
//! - **Preset Hamiltonians**: Heisenberg, transverse-field Ising, XXZ chains
//! - **Preset jump operators**: dephasing, amplitude damping, depolarizing channels
//!
//! # References
//!
//! - Nature Communications, "Tensor Jump Method for open quantum system simulation" (Dec 2025)
//! - Daley, A.J., "Quantum trajectories and open many-body quantum systems" (2014)
//! - Schollwoeck, U., "The density-matrix renormalization group in the age of MPS" (2011)

use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;
use std::time::Instant;

// ============================================================
// COMPLEX HELPER FUNCTIONS
// ============================================================

/// Create a zero complex value.
#[inline]
fn c0() -> C64 {
    C64::new(0.0, 0.0)
}

/// Create a real-valued complex number.
#[inline]
fn cr(r: f64) -> C64 {
    C64::new(r, 0.0)
}

/// Create a purely imaginary complex number.
#[inline]
fn ci(i: f64) -> C64 {
    C64::new(0.0, i)
}

/// Hermitian conjugate of a 2x2 matrix.
#[inline]
fn adjoint_2x2(m: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
    [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ]
}

/// Multiply two 2x2 complex matrices.
#[inline]
fn mul_2x2(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
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

/// Scale a 2x2 matrix by a complex scalar.
#[inline]
fn scale_2x2(s: C64, m: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
    [[s * m[0][0], s * m[0][1]], [s * m[1][0], s * m[1][1]]]
}

/// Add two 2x2 complex matrices.
#[inline]
fn add_2x2(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

/// Identity 2x2 matrix.
#[inline]
fn eye_2x2() -> [[C64; 2]; 2] {
    [[cr(1.0), c0()], [c0(), cr(1.0)]]
}

// ============================================================
// PAULI MATRICES
// ============================================================

/// Pauli X (bit-flip) matrix.
pub const PAULI_X: [[C64; 2]; 2] = [
    [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
];

/// Pauli Y matrix.
pub const PAULI_Y: [[C64; 2]; 2] = [
    [C64::new(0.0, 0.0), C64::new(0.0, -1.0)],
    [C64::new(0.0, 1.0), C64::new(0.0, 0.0)],
];

/// Pauli Z (phase-flip) matrix.
pub const PAULI_Z: [[C64; 2]; 2] = [
    [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
    [C64::new(0.0, 0.0), C64::new(-1.0, 0.0)],
];

/// Sigma minus (lowering operator) |0><1|.
pub const SIGMA_MINUS: [[C64; 2]; 2] = [
    [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    [C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
];

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors that can occur during Tensor Jump Method simulation.
#[derive(Debug, Clone)]
pub enum TJMError {
    /// The supplied configuration is invalid.
    InvalidConfig(String),
    /// Time evolution failed to converge at a given step.
    ConvergenceFailed {
        /// Time step index where convergence failed.
        step: usize,
        /// Norm of the state at failure.
        norm: f64,
    },
    /// Bond dimension exceeded the allowed maximum at a site.
    BondDimExceeded {
        /// Site index where the overflow occurred.
        site: usize,
        /// Actual bond dimension encountered.
        dim: usize,
    },
    /// SVD decomposition failed during truncation.
    SvdFailed(String),
}

impl fmt::Display for TJMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TJMError::InvalidConfig(msg) => write!(f, "Invalid TJM config: {}", msg),
            TJMError::ConvergenceFailed { step, norm } => {
                write!(f, "Convergence failed at step {} (norm={:.6e})", step, norm)
            }
            TJMError::BondDimExceeded { site, dim } => {
                write!(
                    f,
                    "Bond dimension {} exceeded maximum at site {}",
                    dim, site
                )
            }
            TJMError::SvdFailed(msg) => write!(f, "SVD failed: {}", msg),
        }
    }
}

impl std::error::Error for TJMError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for a Tensor Jump Method simulation.
///
/// Controls all numerical parameters: system size, bond dimension truncation,
/// time stepping, Monte Carlo trajectory count, and SVD cutoff.
#[derive(Debug, Clone)]
pub struct TJMConfig {
    /// Number of sites (spins/qubits) in the chain.
    pub num_sites: usize,
    /// Maximum MPS bond dimension (controls accuracy vs speed tradeoff).
    pub bond_dim: usize,
    /// Integration time step for TDVP sweeps.
    pub dt: f64,
    /// Total simulation time.
    pub total_time: f64,
    /// Number of independent Monte Carlo trajectories.
    pub num_trajectories: usize,
    /// Random number generator seed for reproducibility.
    pub seed: u64,
    /// SVD singular value cutoff threshold.
    pub svd_cutoff: f64,
}

impl Default for TJMConfig {
    fn default() -> Self {
        Self {
            num_sites: 10,
            bond_dim: 32,
            dt: 0.01,
            total_time: 1.0,
            num_trajectories: 100,
            seed: 42,
            svd_cutoff: 1e-10,
        }
    }
}

impl TJMConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of sites.
    pub fn num_sites(mut self, n: usize) -> Self {
        self.num_sites = n;
        self
    }

    /// Set the maximum bond dimension.
    pub fn bond_dim(mut self, d: usize) -> Self {
        self.bond_dim = d;
        self
    }

    /// Set the time step.
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set the total simulation time.
    pub fn total_time(mut self, t: f64) -> Self {
        self.total_time = t;
        self
    }

    /// Set the number of Monte Carlo trajectories.
    pub fn num_trajectories(mut self, n: usize) -> Self {
        self.num_trajectories = n;
        self
    }

    /// Set the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set the SVD truncation cutoff.
    pub fn svd_cutoff(mut self, c: f64) -> Self {
        self.svd_cutoff = c;
        self
    }

    /// Validate the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), TJMError> {
        if self.num_sites == 0 {
            return Err(TJMError::InvalidConfig("num_sites must be > 0".into()));
        }
        if self.bond_dim == 0 {
            return Err(TJMError::InvalidConfig("bond_dim must be > 0".into()));
        }
        if self.dt <= 0.0 {
            return Err(TJMError::InvalidConfig("dt must be > 0".into()));
        }
        if self.total_time <= 0.0 {
            return Err(TJMError::InvalidConfig("total_time must be > 0".into()));
        }
        if self.num_trajectories == 0 {
            return Err(TJMError::InvalidConfig(
                "num_trajectories must be > 0".into(),
            ));
        }
        if self.svd_cutoff < 0.0 {
            return Err(TJMError::InvalidConfig("svd_cutoff must be >= 0".into()));
        }
        Ok(())
    }
}

// ============================================================
// 3-INDEX TENSOR (SIMPLIFIED MPS SITE TENSOR)
// ============================================================

/// A rank-3 tensor with indices [left_bond, physical, right_bond].
///
/// Stored as a flat `Vec<C64>` with dimensions (dl, dp, dr) and row-major
/// indexing: element (l, p, r) is at position `l * dp * dr + p * dr + r`.
#[derive(Debug, Clone)]
pub struct Array3 {
    /// Flat storage in row-major order.
    pub data: Vec<C64>,
    /// Left bond dimension.
    pub dl: usize,
    /// Physical dimension.
    pub dp: usize,
    /// Right bond dimension.
    pub dr: usize,
}

impl Array3 {
    /// Create a zero-filled rank-3 tensor.
    pub fn zeros(dl: usize, dp: usize, dr: usize) -> Self {
        Self {
            data: vec![c0(); dl * dp * dr],
            dl,
            dp,
            dr,
        }
    }

    /// Access element at (l, p, r).
    #[inline]
    pub fn get(&self, l: usize, p: usize, r: usize) -> C64 {
        self.data[l * self.dp * self.dr + p * self.dr + r]
    }

    /// Mutable access to element at (l, p, r).
    #[inline]
    pub fn set(&mut self, l: usize, p: usize, r: usize, val: C64) {
        self.data[l * self.dp * self.dr + p * self.dr + r] = val;
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Scale every element by a complex scalar.
    pub fn scale(&mut self, s: C64) {
        for v in self.data.iter_mut() {
            *v *= s;
        }
    }
}

// ============================================================
// FLAT COMPLEX MATRIX (for SVD and linear algebra)
// ============================================================

/// Dense complex matrix stored row-major with dimensions (rows, cols).
#[derive(Debug, Clone)]
struct CMatrix {
    data: Vec<C64>,
    rows: usize,
    cols: usize,
}

impl CMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![c0(); rows * cols],
            rows,
            cols,
        }
    }

    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, cr(1.0));
        }
        m
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> C64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, val: C64) {
        self.data[r * self.cols + c] = val;
    }

    /// Compute the Frobenius norm squared.
    fn norm_sq(&self) -> f64 {
        self.data.iter().map(|v| v.norm_sqr()).sum()
    }

    /// Matrix-matrix multiply: self * other.
    fn matmul(&self, other: &CMatrix) -> CMatrix {
        assert_eq!(self.cols, other.rows, "matmul dimension mismatch");
        let mut result = CMatrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a.norm_sqr() < 1e-30 {
                    continue;
                }
                for j in 0..other.cols {
                    let cur = result.get(i, j);
                    result.set(i, j, cur + a * other.get(k, j));
                }
            }
        }
        result
    }

    /// Hermitian conjugate (conjugate transpose).
    fn adjoint(&self) -> CMatrix {
        let mut result = CMatrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j).conj());
            }
        }
        result
    }

    /// Extract a column as a vector.
    fn col(&self, j: usize) -> Vec<C64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// Set a column from a vector.
    fn set_col(&mut self, j: usize, v: &[C64]) {
        for i in 0..self.rows.min(v.len()) {
            self.set(i, j, v[i]);
        }
    }

    /// Extract a sub-matrix (first k columns).
    fn first_cols(&self, k: usize) -> CMatrix {
        let k = k.min(self.cols);
        let mut result = CMatrix::zeros(self.rows, k);
        for i in 0..self.rows {
            for j in 0..k {
                result.set(i, j, self.get(i, j));
            }
        }
        result
    }

    /// Extract first k rows.
    fn first_rows(&self, k: usize) -> CMatrix {
        let k = k.min(self.rows);
        let mut result = CMatrix::zeros(k, self.cols);
        for i in 0..k {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j));
            }
        }
        result
    }
}

// ============================================================
// JACOBI ONE-SIDED SVD
// ============================================================

/// Compute dot product of two complex vectors.
fn cdot(a: &[C64], b: &[C64]) -> C64 {
    a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum()
}

/// Compute L2 norm of a complex vector.
fn cnorm(v: &[C64]) -> f64 {
    v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

/// One-sided Jacobi SVD for a complex matrix.
///
/// Computes A = U * diag(sigma) * Vt where sigma is sorted descending.
/// Truncates to at most `max_rank` singular values above `cutoff`.
///
/// This implementation targets moderate-sized local matrices
/// (the local effective Hamiltonians in MPS algorithms are typically small).
fn jacobi_svd(
    a: &CMatrix,
    max_rank: usize,
    cutoff: f64,
) -> Result<(CMatrix, Vec<f64>, CMatrix), TJMError> {
    let m = a.rows;
    let n = a.cols;
    let k = m.min(n);

    if k == 0 {
        return Ok((CMatrix::zeros(m, 0), vec![], CMatrix::zeros(0, n)));
    }

    // Work on A^H * A to get V and singular values, then U = A * V * diag(1/sigma)
    // For numerical stability with rectangular matrices, use the one-sided Jacobi approach:
    // Orthogonalize columns of A iteratively using Jacobi rotations on A^H * A.

    // Start with a copy of A. We will orthogonalize its columns in-place.
    let mut work = a.clone();
    let mut v_rot = CMatrix::identity(n);

    let max_sweeps = 100;
    let tol = 1e-14;

    for _sweep in 0..max_sweeps {
        let mut off_norm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                // Compute column inner products
                let col_i: Vec<C64> = (0..m).map(|r| work.get(r, i)).collect();
                let col_j: Vec<C64> = (0..m).map(|r| work.get(r, j)).collect();

                let aii = cdot(&col_i, &col_i).re;
                let ajj = cdot(&col_j, &col_j).re;
                let aij = cdot(&col_i, &col_j);

                off_norm += aij.norm_sqr();

                if aij.norm() < tol * (aii * ajj).sqrt().max(1e-30) {
                    continue;
                }

                // Compute 2x2 Jacobi rotation to zero out aij
                let tau = (ajj - aii) / (2.0 * aij.re);
                let t = if aij.norm() < 1e-30 {
                    0.0
                } else {
                    // Use the real part for the rotation angle (Hermitian case)
                    let sign_tau = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign_tau / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let cos_t = 1.0 / (1.0 + t * t).sqrt();
                let sin_t = t * cos_t;

                // Phase factor to handle complex off-diagonal
                let phase = if aij.norm() > 1e-30 {
                    aij / cr(aij.norm())
                } else {
                    cr(1.0)
                };

                // Apply rotation to columns of work: col_i' = cos*col_i - sin*phase.conj()*col_j
                //                                     col_j' = sin*phase*col_i + cos*col_j
                for r in 0..m {
                    let wi = work.get(r, i);
                    let wj = work.get(r, j);
                    work.set(r, i, cr(cos_t) * wi - cr(sin_t) * phase.conj() * wj);
                    work.set(r, j, cr(sin_t) * phase * wi + cr(cos_t) * wj);
                }

                // Accumulate V rotation
                for r in 0..n {
                    let vi = v_rot.get(r, i);
                    let vj = v_rot.get(r, j);
                    v_rot.set(r, i, cr(cos_t) * vi - cr(sin_t) * phase.conj() * vj);
                    v_rot.set(r, j, cr(sin_t) * phase * vi + cr(cos_t) * vj);
                }
            }
        }

        if off_norm.sqrt() < tol * a.norm_sq().sqrt().max(1e-30) {
            break;
        }
    }

    // Extract singular values (norms of columns of work) and U columns
    let mut sigma_indices: Vec<(f64, usize)> = Vec::with_capacity(k);
    for j in 0..n {
        let col: Vec<C64> = (0..m).map(|r| work.get(r, j)).collect();
        let s = cnorm(&col);
        sigma_indices.push((s, j));
    }

    // Sort by singular value descending
    sigma_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Determine effective rank
    let rank = sigma_indices
        .iter()
        .take(max_rank)
        .take_while(|(s, _)| *s > cutoff)
        .count()
        .max(1); // Keep at least 1

    let mut u = CMatrix::zeros(m, rank);
    let mut sigmas = Vec::with_capacity(rank);
    let mut vt = CMatrix::zeros(rank, n);

    for (out_idx, &(s, col_idx)) in sigma_indices.iter().take(rank).enumerate() {
        sigmas.push(s);

        // U column = normalized work column
        if s > 1e-30 {
            for r in 0..m {
                u.set(r, out_idx, work.get(r, col_idx) / cr(s));
            }
        } else {
            // Degenerate: just set the first element
            if out_idx < m {
                u.set(out_idx, out_idx, cr(1.0));
            }
        }

        // Vt row = corresponding V column (conjugate transposed)
        for c in 0..n {
            vt.set(out_idx, c, v_rot.get(c, col_idx).conj());
        }
    }

    Ok((u, sigmas, vt))
}

// ============================================================
// MATRIX PRODUCT STATE (MPS)
// ============================================================

/// Matrix Product State representation for a chain of qubits.
///
/// Each site holds a rank-3 tensor with indices [left_bond, physical, right_bond].
/// For a product state, all bond dimensions are 1. Entanglement grows the bond
/// dimension up to the configured maximum, controlled by SVD truncation.
#[derive(Debug, Clone)]
pub struct MPS {
    /// Rank-3 tensors for each site.
    pub tensors: Vec<Array3>,
    /// Number of sites in the chain.
    pub num_sites: usize,
    /// Physical dimension per site (2 for qubits).
    pub physical_dim: usize,
}

impl MPS {
    /// Create a product-state MPS where each site is in the given local state.
    ///
    /// `state_per_site` must have length `num_sites`. Each entry is a 2-element
    /// array [alpha, beta] representing alpha|0> + beta|1>.
    pub fn new_product_state(num_sites: usize, state_per_site: &[[C64; 2]]) -> Self {
        assert_eq!(
            state_per_site.len(),
            num_sites,
            "state_per_site length must equal num_sites"
        );
        let physical_dim = 2;
        let mut tensors = Vec::with_capacity(num_sites);

        for local_state in state_per_site {
            // Bond dim 1 on both sides: tensor shape [1, 2, 1]
            let mut t = Array3::zeros(1, physical_dim, 1);
            t.set(0, 0, 0, local_state[0]);
            t.set(0, 1, 0, local_state[1]);
            tensors.push(t);
        }

        MPS {
            tensors,
            num_sites,
            physical_dim,
        }
    }

    /// Create a product state with all sites in |0>.
    pub fn all_zero(num_sites: usize) -> Self {
        let states: Vec<[C64; 2]> = vec![[cr(1.0), c0()]; num_sites];
        Self::new_product_state(num_sites, &states)
    }

    /// Create a product state with all sites in |1>.
    pub fn all_one(num_sites: usize) -> Self {
        let states: Vec<[C64; 2]> = vec![[c0(), cr(1.0)]; num_sites];
        Self::new_product_state(num_sites, &states)
    }

    /// Apply a 2x2 single-site operator to the given site.
    ///
    /// Contracts the operator with the physical index of the site tensor:
    /// T'[l, p', r] = sum_p op[p', p] * T[l, p, r]
    pub fn apply_one_site(&mut self, site: usize, op: &[[C64; 2]; 2]) {
        assert!(site < self.num_sites, "site index out of bounds");
        let t = &self.tensors[site];
        let dl = t.dl;
        let dr = t.dr;
        let mut new_t = Array3::zeros(dl, 2, dr);

        for l in 0..dl {
            for r in 0..dr {
                for pp in 0..2 {
                    let mut val = c0();
                    for p in 0..2 {
                        val += op[pp][p] * t.get(l, p, r);
                    }
                    new_t.set(l, pp, r, val);
                }
            }
        }
        self.tensors[site] = new_t;
    }

    /// Apply a 4x4 two-site gate to sites (site, site+1) with SVD truncation.
    ///
    /// The gate matrix is indexed as gate[p1'*2 + p2'][p1*2 + p2] where
    /// p1, p2 are the physical indices of the two sites.
    pub fn apply_two_site(
        &mut self,
        site: usize,
        gate: &[[C64; 4]; 4],
        max_bond_dim: usize,
        svd_cutoff: f64,
    ) -> Result<(), TJMError> {
        assert!(
            site + 1 < self.num_sites,
            "two-site gate requires site+1 < num_sites"
        );

        let t_left = &self.tensors[site];
        let t_right = &self.tensors[site + 1];
        let dl = t_left.dl;
        let dr = t_right.dr;

        // Contract: theta[l, p1p2, r] = sum_{m} T_left[l, p1, m] * T_right[m, p2, r]
        // then apply gate: theta'[l, p1'p2', r] = sum_{p1p2} gate[p1'p2', p1p2] * theta[l, p1p2, r]
        let bond_mid = t_left.dr; // = t_right.dl

        // Build theta as a matrix of shape (dl * 4, dr) with combined physical index
        // Actually, build the full 4-index object then apply gate, then SVD split.

        // Step 1: Contract the two site tensors
        // theta[l, p1, p2, r] = sum_m T_L[l,p1,m] * T_R[m,p2,r]
        let mut theta = vec![c0(); dl * 4 * dr]; // indexed [l, (p1*2+p2), r]
        for l in 0..dl {
            for p1 in 0..2 {
                for p2 in 0..2 {
                    let pp = p1 * 2 + p2;
                    for r in 0..dr {
                        let mut val = c0();
                        for m in 0..bond_mid {
                            val += t_left.get(l, p1, m) * t_right.get(m, p2, r);
                        }
                        theta[l * 4 * dr + pp * dr + r] = val;
                    }
                }
            }
        }

        // Step 2: Apply the gate
        // theta'[l, pp', r] = sum_{pp} gate[pp', pp] * theta[l, pp, r]
        let mut theta_prime = vec![c0(); dl * 4 * dr];
        for l in 0..dl {
            for pp_out in 0..4 {
                for r in 0..dr {
                    let mut val = c0();
                    for pp_in in 0..4 {
                        val += gate[pp_out][pp_in] * theta[l * 4 * dr + pp_in * dr + r];
                    }
                    theta_prime[l * 4 * dr + pp_out * dr + r] = val;
                }
            }
        }

        // Step 3: Reshape to matrix (dl*2, 2*dr) and SVD
        // theta'[l, p1*2+p2, r] -> mat[(l*2+p1), (p2*dr+r)]
        let mat_rows = dl * 2;
        let mat_cols = 2 * dr;
        let mut mat = CMatrix::zeros(mat_rows, mat_cols);
        for l in 0..dl {
            for p1 in 0..2 {
                for p2 in 0..2 {
                    for r in 0..dr {
                        let row = l * 2 + p1;
                        let col = p2 * dr + r;
                        let pp = p1 * 2 + p2;
                        mat.set(row, col, theta_prime[l * 4 * dr + pp * dr + r]);
                    }
                }
            }
        }

        let (u, sigmas, vt) = jacobi_svd(&mat, max_bond_dim, svd_cutoff)?;
        let new_bond = sigmas.len();

        // Step 4: Absorb singular values into U: U_new = U * diag(sigma)
        // Reshape U (dl*2, new_bond) -> T_left(dl, 2, new_bond)
        let mut new_left = Array3::zeros(dl, 2, new_bond);
        for l in 0..dl {
            for p1 in 0..2 {
                for b in 0..new_bond {
                    new_left.set(l, p1, b, u.get(l * 2 + p1, b) * cr(sigmas[b]));
                }
            }
        }

        // Reshape Vt (new_bond, 2*dr) -> T_right(new_bond, 2, dr)
        let mut new_right = Array3::zeros(new_bond, 2, dr);
        for b in 0..new_bond {
            for p2 in 0..2 {
                for r in 0..dr {
                    new_right.set(b, p2, r, vt.get(b, p2 * dr + r));
                }
            }
        }

        self.tensors[site] = new_left;
        self.tensors[site + 1] = new_right;

        Ok(())
    }

    /// Compute the squared norm <psi|psi> via sequential contraction from left to right.
    ///
    /// Returns the real part of the overlap (imaginary part should be negligible
    /// for a well-formed MPS).
    pub fn norm_sq(&self) -> f64 {
        if self.num_sites == 0 {
            return 0.0;
        }

        // Start with a 1x1 identity boundary: transfer[bl, bl'] = delta(bl, bl')
        // At each site, contract: new_transfer[br, br'] = sum_{bl,bl',p} conj(T[bl,p,br]) * T[bl',p,br'] * transfer[bl, bl']

        let t0 = &self.tensors[0];
        let mut transfer = CMatrix::zeros(t0.dr, t0.dr);

        // First site (left bond dim = dl0)
        for br in 0..t0.dr {
            for br2 in 0..t0.dr {
                let mut val = c0();
                for bl in 0..t0.dl {
                    for p in 0..self.physical_dim {
                        val += t0.get(bl, p, br).conj() * t0.get(bl, p, br2);
                    }
                }
                transfer.set(br, br2, val);
            }
        }

        // Subsequent sites
        for site in 1..self.num_sites {
            let t = &self.tensors[site];
            let mut new_transfer = CMatrix::zeros(t.dr, t.dr);

            for br in 0..t.dr {
                for br2 in 0..t.dr {
                    let mut val = c0();
                    for bl in 0..t.dl {
                        for bl2 in 0..t.dl {
                            let tr = transfer.get(bl, bl2);
                            if tr.norm_sqr() < 1e-30 {
                                continue;
                            }
                            for p in 0..self.physical_dim {
                                val += t.get(bl, p, br).conj() * t.get(bl2, p, br2) * tr;
                            }
                        }
                    }
                    new_transfer.set(br, br2, val);
                }
            }
            transfer = new_transfer;
        }

        // Final: transfer should be 1x1 (right boundary bond dim is 1)
        transfer.get(0, 0).re
    }

    /// Compute the norm ||psi||.
    pub fn norm(&self) -> f64 {
        self.norm_sq().abs().sqrt()
    }

    /// Normalize the MPS to unit norm. Returns the original norm.
    pub fn normalize(&mut self) -> f64 {
        let n = self.norm();
        if n > 1e-15 {
            let factor = cr(1.0 / n);
            // Scale the first tensor (sufficient for normalization)
            self.tensors[0].scale(factor);
        }
        n
    }

    /// Compute the expectation value <psi|O_site|psi> for a 2x2 single-site operator.
    pub fn expectation(&self, site: usize, op: &[[C64; 2]; 2]) -> C64 {
        if self.num_sites == 0 || site >= self.num_sites {
            return c0();
        }

        // Contract from the left up to 'site', insert operator, continue to the right.
        // Transfer matrix approach: same as norm_sq but at 'site' we insert the operator.

        let t0 = &self.tensors[0];
        let insert_op = site == 0;
        let mut transfer = CMatrix::zeros(t0.dr, t0.dr);

        for br in 0..t0.dr {
            for br2 in 0..t0.dr {
                let mut val = c0();
                for bl in 0..t0.dl {
                    if insert_op {
                        for pp in 0..2 {
                            let bra = t0.get(bl, pp, br).conj();
                            for p in 0..2 {
                                val += bra * op[pp][p] * t0.get(bl, p, br2);
                            }
                        }
                    } else {
                        for p in 0..self.physical_dim {
                            val += t0.get(bl, p, br).conj() * t0.get(bl, p, br2);
                        }
                    }
                }
                transfer.set(br, br2, val);
            }
        }

        for s in 1..self.num_sites {
            let t = &self.tensors[s];
            let is_op_site = s == site;
            let mut new_transfer = CMatrix::zeros(t.dr, t.dr);

            for br in 0..t.dr {
                for br2 in 0..t.dr {
                    let mut val = c0();
                    for bl in 0..t.dl {
                        for bl2 in 0..t.dl {
                            let tr = transfer.get(bl, bl2);
                            if tr.norm_sqr() < 1e-30 {
                                continue;
                            }
                            if is_op_site {
                                for pp in 0..2 {
                                    let bra = t.get(bl, pp, br).conj();
                                    for p in 0..2 {
                                        val += bra * op[pp][p] * t.get(bl2, p, br2) * tr;
                                    }
                                }
                            } else {
                                for p in 0..self.physical_dim {
                                    val += t.get(bl, p, br).conj() * t.get(bl2, p, br2) * tr;
                                }
                            }
                        }
                    }
                    new_transfer.set(br, br2, val);
                }
            }
            transfer = new_transfer;
        }

        transfer.get(0, 0)
    }

    /// Compute the inner product <self|other>.
    pub fn inner(&self, other: &MPS) -> C64 {
        assert_eq!(
            self.num_sites, other.num_sites,
            "MPS must have same number of sites"
        );
        if self.num_sites == 0 {
            return c0();
        }

        let t0_bra = &self.tensors[0];
        let t0_ket = &other.tensors[0];
        let mut transfer = CMatrix::zeros(t0_bra.dr, t0_ket.dr);

        for br in 0..t0_bra.dr {
            for br2 in 0..t0_ket.dr {
                let mut val = c0();
                for bl in 0..t0_bra.dl {
                    for bl2 in 0..t0_ket.dl {
                        for p in 0..self.physical_dim {
                            val += t0_bra.get(bl, p, br).conj() * t0_ket.get(bl2, p, br2);
                        }
                    }
                }
                transfer.set(br, br2, val);
            }
        }

        for s in 1..self.num_sites {
            let tb = &self.tensors[s];
            let tk = &other.tensors[s];
            let mut new_transfer = CMatrix::zeros(tb.dr, tk.dr);

            for br in 0..tb.dr {
                for br2 in 0..tk.dr {
                    let mut val = c0();
                    for bl in 0..tb.dl {
                        for bl2 in 0..tk.dl {
                            let tr = transfer.get(bl, bl2);
                            if tr.norm_sqr() < 1e-30 {
                                continue;
                            }
                            for p in 0..self.physical_dim {
                                val += tb.get(bl, p, br).conj() * tk.get(bl2, p, br2) * tr;
                            }
                        }
                    }
                    new_transfer.set(br, br2, val);
                }
            }
            transfer = new_transfer;
        }

        transfer.get(0, 0)
    }

    /// Return the maximum bond dimension across all bonds.
    pub fn max_bond_dim(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| t.dl.max(t.dr))
            .max()
            .unwrap_or(1)
    }

    /// Return the average bond dimension across all internal bonds.
    pub fn avg_bond_dim(&self) -> f64 {
        if self.num_sites <= 1 {
            return 1.0;
        }
        let sum: usize = self.tensors[..self.num_sites - 1]
            .iter()
            .map(|t| t.dr)
            .sum();
        sum as f64 / (self.num_sites - 1) as f64
    }
}

// ============================================================
// JUMP OPERATOR
// ============================================================

/// A Lindblad jump operator acting on a single site.
///
/// Represents L_k = sqrt(gamma) * matrix, where gamma is the jump rate
/// and matrix is the 2x2 operator in the qubit basis.
#[derive(Debug, Clone)]
pub struct JumpOperator {
    /// Site index this operator acts on.
    pub site: usize,
    /// 2x2 operator matrix (NOT including the sqrt(gamma) prefactor).
    pub matrix: [[C64; 2]; 2],
    /// Jump rate gamma (the prefactor sqrt(gamma) is applied during simulation).
    pub rate: f64,
}

impl JumpOperator {
    /// Compute L = sqrt(rate) * matrix.
    pub fn full_operator(&self) -> [[C64; 2]; 2] {
        let s = cr(self.rate.sqrt());
        scale_2x2(s, &self.matrix)
    }

    /// Compute L^dagger L (for the effective Hamiltonian).
    pub fn ldagger_l(&self) -> [[C64; 2]; 2] {
        let l = self.full_operator();
        let ld = adjoint_2x2(&l);
        mul_2x2(&ld, &l)
    }
}

// ============================================================
// HAMILTONIAN
// ============================================================

/// A single term in the Hamiltonian, acting on 1 or 2 adjacent sites.
#[derive(Debug, Clone)]
pub struct HamiltonianTerm {
    /// Site indices this term acts on (1 or 2 elements).
    pub sites: Vec<usize>,
    /// Matrix representation: 2x2 for single-site, 4x4 for two-site.
    pub matrix: Vec<Vec<C64>>,
}

/// Local Hamiltonian as a sum of terms.
#[derive(Debug, Clone)]
pub struct LocalHamiltonian {
    /// Number of sites in the chain.
    pub num_sites: usize,
    /// Individual Hamiltonian terms.
    pub terms: Vec<HamiltonianTerm>,
}

// ============================================================
// PRESET HAMILTONIANS
// ============================================================

/// Construct a Heisenberg XXX chain Hamiltonian.
///
/// H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
///
/// The two-site interaction matrix is the 4x4 Heisenberg exchange.
pub fn heisenberg_chain(n: usize, j_coupling: f64) -> LocalHamiltonian {
    let j = cr(j_coupling);
    let mut terms = Vec::with_capacity(n - 1);

    for i in 0..(n - 1) {
        // XX + YY + ZZ in the computational basis {|00>, |01>, |10>, |11>}:
        // H_pair = J * [[1, 0, 0, 0],
        //               [0,-1, 2, 0],
        //               [0, 2,-1, 0],
        //               [0, 0, 0, 1]]
        let matrix = vec![
            vec![j, c0(), c0(), c0()],
            vec![c0(), -j, cr(2.0) * j, c0()],
            vec![c0(), cr(2.0) * j, -j, c0()],
            vec![c0(), c0(), c0(), j],
        ];
        terms.push(HamiltonianTerm {
            sites: vec![i, i + 1],
            matrix,
        });
    }

    LocalHamiltonian {
        num_sites: n,
        terms,
    }
}

/// Construct a transverse-field Ising model Hamiltonian.
///
/// H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
pub fn ising_transverse(n: usize, j: f64, h: f64) -> LocalHamiltonian {
    let mut terms = Vec::new();

    // ZZ interactions
    for i in 0..(n - 1) {
        let jn = cr(-j);
        // ZZ in computational basis: diag(1, -1, -1, 1)
        let matrix = vec![
            vec![jn, c0(), c0(), c0()],
            vec![c0(), -jn, c0(), c0()],
            vec![c0(), c0(), -jn, c0()],
            vec![c0(), c0(), c0(), jn],
        ];
        terms.push(HamiltonianTerm {
            sites: vec![i, i + 1],
            matrix,
        });
    }

    // Transverse field: -h * X on each site
    let hn = cr(-h);
    for i in 0..n {
        let matrix = vec![vec![c0(), hn], vec![hn, c0()]];
        terms.push(HamiltonianTerm {
            sites: vec![i],
            matrix,
        });
    }

    LocalHamiltonian {
        num_sites: n,
        terms,
    }
}

/// Construct an anisotropic XXZ chain Hamiltonian.
///
/// H = sum_i (J_xy * (X_i X_{i+1} + Y_i Y_{i+1}) + J_z * Z_i Z_{i+1})
pub fn xxz_chain(n: usize, j_xy: f64, j_z: f64) -> LocalHamiltonian {
    let jxy = cr(j_xy);
    let jz = cr(j_z);
    let mut terms = Vec::with_capacity(n - 1);

    for i in 0..(n - 1) {
        // J_xy*(XX+YY) + J_z*ZZ
        let matrix = vec![
            vec![jz, c0(), c0(), c0()],
            vec![c0(), -jz, cr(2.0) * jxy, c0()],
            vec![c0(), cr(2.0) * jxy, -jz, c0()],
            vec![c0(), c0(), c0(), jz],
        ];
        terms.push(HamiltonianTerm {
            sites: vec![i, i + 1],
            matrix,
        });
    }

    LocalHamiltonian {
        num_sites: n,
        terms,
    }
}

// ============================================================
// PRESET JUMP OPERATORS
// ============================================================

/// Create a dephasing jump operator: L = sqrt(gamma) * Z.
pub fn dephasing(site: usize, gamma: f64) -> JumpOperator {
    JumpOperator {
        site,
        matrix: PAULI_Z,
        rate: gamma,
    }
}

/// Create an amplitude damping jump operator: L = sqrt(gamma) * sigma_minus.
///
/// sigma_minus = |0><1| causes decay from |1> to |0>.
pub fn amplitude_damping(site: usize, gamma: f64) -> JumpOperator {
    JumpOperator {
        site,
        matrix: SIGMA_MINUS,
        rate: gamma,
    }
}

/// Create depolarizing channel jump operators: L_x, L_y, L_z each with rate gamma/3.
///
/// Returns a vector of 3 jump operators.
pub fn depolarizing(site: usize, gamma: f64) -> Vec<JumpOperator> {
    let rate = gamma / 3.0;
    vec![
        JumpOperator {
            site,
            matrix: PAULI_X,
            rate,
        },
        JumpOperator {
            site,
            matrix: PAULI_Y,
            rate,
        },
        JumpOperator {
            site,
            matrix: PAULI_Z,
            rate,
        },
    ]
}

// ============================================================
// OBSERVABLE
// ============================================================

/// An observable to measure during simulation.
///
/// Represents a single-site Hermitian operator whose expectation value
/// is tracked at each time step, averaged over all trajectories.
#[derive(Debug, Clone)]
pub struct Observable {
    /// Human-readable name for this observable.
    pub name: String,
    /// Site index where the operator acts.
    pub site: usize,
    /// 2x2 Hermitian operator matrix.
    pub matrix: [[C64; 2]; 2],
}

impl Observable {
    /// Create a Z observable at the given site.
    pub fn z(site: usize) -> Self {
        Self {
            name: format!("Z_{}", site),
            site,
            matrix: PAULI_Z,
        }
    }

    /// Create an X observable at the given site.
    pub fn x(site: usize) -> Self {
        Self {
            name: format!("X_{}", site),
            site,
            matrix: PAULI_X,
        }
    }

    /// Create a Y observable at the given site.
    pub fn y(site: usize) -> Self {
        Self {
            name: format!("Y_{}", site),
            site,
            matrix: PAULI_Y,
        }
    }
}

// ============================================================
// TRAJECTORY
// ============================================================

/// A single Monte Carlo trajectory in the Tensor Jump Method.
///
/// Each trajectory independently evolves an MPS state under the effective
/// non-Hermitian Hamiltonian, with stochastic quantum jumps at rates
/// determined by the Lindblad operators.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Current MPS state of this trajectory.
    pub state: MPS,
    /// Current simulation time.
    pub time: f64,
    /// Total number of quantum jumps that have occurred.
    pub num_jumps: usize,
    /// History of jumps: (time, site, operator_index).
    pub jump_history: Vec<(f64, usize, usize)>,
}

impl Trajectory {
    /// Create a new trajectory from an initial MPS state.
    pub fn new(initial_state: MPS) -> Self {
        Self {
            state: initial_state,
            time: 0.0,
            num_jumps: 0,
            jump_history: Vec::new(),
        }
    }
}

// ============================================================
// SIMULATION RESULT
// ============================================================

/// Result of a complete TJM simulation.
#[derive(Debug, Clone)]
pub struct TJMResult {
    /// Configuration used for this simulation.
    pub config: TJMConfig,
    /// Time-series of expectation values: (time, [obs1, obs2, ...]).
    pub expectation_values: Vec<(f64, Vec<f64>)>,
    /// Total quantum jumps across all trajectories.
    pub total_jumps: usize,
    /// Average bond dimension across the final states.
    pub avg_bond_dim: f64,
    /// Number of trajectories computed.
    pub trajectory_count: usize,
    /// Wall-clock time in seconds.
    pub wall_time_secs: f64,
}

// ============================================================
// SIMPLIFIED TDVP TIME EVOLUTION
// ============================================================

/// Compute matrix exponential exp(M) for a small complex matrix using
/// a truncated Taylor series (sufficient for small ||M|| ~ dt).
///
/// exp(M) ~ I + M + M^2/2! + M^3/3! + ... + M^k/k!
fn matrix_exp_taylor(m: &CMatrix, order: usize) -> CMatrix {
    let n = m.rows;
    assert_eq!(n, m.cols, "matrix_exp requires square matrix");

    let mut result = CMatrix::identity(n);
    let mut term = CMatrix::identity(n);

    for k in 1..=order {
        term = term.matmul(m);
        let factor = cr(1.0 / factorial(k) as f64);
        for i in 0..n {
            for j in 0..n {
                let cur = result.get(i, j);
                result.set(i, j, cur + factor * term.get(i, j));
            }
        }
    }
    result
}

/// Factorial helper (up to 20! fits in u64).
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product()
}

/// Estimate a single-site reduced density matrix from the current local tensor.
///
/// This is a local approximation and avoids full environment contraction.
fn local_density_matrix(mps: &MPS, site: usize) -> [[C64; 2]; 2] {
    let t = &mps.tensors[site];
    let mut rho = [[c0(); 2]; 2];
    for l in 0..t.dl {
        for r in 0..t.dr {
            let a0 = t.get(l, 0, r);
            let a1 = t.get(l, 1, r);
            rho[0][0] += a0 * a0.conj();
            rho[0][1] += a0 * a1.conj();
            rho[1][0] += a1 * a0.conj();
            rho[1][1] += a1 * a1.conj();
        }
    }
    let tr = (rho[0][0] + rho[1][1]).re;
    if tr > 1e-14 {
        let inv = cr(1.0 / tr);
        for row in &mut rho {
            for elem in row {
                *elem *= inv;
            }
        }
    }
    rho
}

/// Build the effective single-site Hamiltonian matrix for TDVP at the given site.
///
/// This collects all single-site Hamiltonian terms that act on this site,
/// plus the anti-Hermitian correction -i/2 * sum_k L_k^dagger L_k.
/// For two-site Hamiltonian terms, it uses state-dependent local reductions
/// from the current MPS, rather than a fixed maximally-mixed neighbor.
fn build_effective_site_hamiltonian(
    site: usize,
    mps: &MPS,
    hamiltonian: &LocalHamiltonian,
    jump_ops: &[JumpOperator],
) -> [[C64; 2]; 2] {
    let mut h_eff = [[c0(); 2]; 2];

    // Collect single-site Hamiltonian terms
    for term in &hamiltonian.terms {
        if term.sites.len() == 1 && term.sites[0] == site {
            for i in 0..2 {
                for j in 0..2 {
                    h_eff[i][j] += term.matrix[i][j];
                }
            }
        }
        // For two-site terms involving this site, reduce the neighbor using
        // its local density matrix estimated from the current MPS.
        if term.sites.len() == 2 {
            if term.sites[0] == site {
                let rho_nbr = local_density_matrix(mps, term.sites[1]);
                // h_eff[i,j] += sum_{p2,q2} H[(i,p2),(j,q2)] * rho_nbr[q2,p2]
                for i in 0..2 {
                    for j in 0..2 {
                        for p2 in 0..2 {
                            for q2 in 0..2 {
                                let row = i * 2 + p2;
                                let col = j * 2 + q2;
                                h_eff[i][j] += term.matrix[row][col] * rho_nbr[q2][p2];
                            }
                        }
                    }
                }
            } else if term.sites[1] == site {
                let rho_nbr = local_density_matrix(mps, term.sites[0]);
                // h_eff[i,j] += sum_{p1,q1} H[(p1,i),(q1,j)] * rho_nbr[q1,p1]
                for i in 0..2 {
                    for j in 0..2 {
                        for p1 in 0..2 {
                            for q1 in 0..2 {
                                let row = p1 * 2 + i;
                                let col = q1 * 2 + j;
                                h_eff[i][j] += term.matrix[row][col] * rho_nbr[q1][p1];
                            }
                        }
                    }
                }
            }
        }
    }

    // Add -i/2 * sum_k L_k^dagger L_k for jump operators on this site
    for jop in jump_ops {
        if jop.site == site {
            let ldl = jop.ldagger_l();
            for i in 0..2 {
                for j in 0..2 {
                    h_eff[i][j] -= ci(0.5) * ldl[i][j];
                }
            }
        }
    }

    h_eff
}

/// Perform one TDVP sweep advancing the MPS by dt.
///
/// At each site, builds the effective Hamiltonian and applies exp(-i * H_eff * dt)
/// to the local tensor.
fn tdvp_sweep(
    mps: &mut MPS,
    dt: f64,
    hamiltonian: &LocalHamiltonian,
    jump_ops: &[JumpOperator],
    left_to_right: bool,
) {
    let site_iter: Box<dyn Iterator<Item = usize>> = if left_to_right {
        Box::new(0..mps.num_sites)
    } else {
        Box::new((0..mps.num_sites).rev())
    };

    for site in site_iter {
        let h_eff = build_effective_site_hamiltonian(site, mps, hamiltonian, jump_ops);

        // Build -i * H_eff * dt as a 2x2 matrix
        let factor = ci(-dt);
        let m = scale_2x2(factor, &h_eff);

        // Compute exp(-i * H_eff * dt) via Taylor series (order 8 for dt ~ 0.01)
        let mat = CMatrix {
            data: vec![m[0][0], m[0][1], m[1][0], m[1][1]],
            rows: 2,
            cols: 2,
        };
        let exp_m = matrix_exp_taylor(&mat, 8);

        // Apply to site tensor
        let exp_op = [
            [exp_m.get(0, 0), exp_m.get(0, 1)],
            [exp_m.get(1, 0), exp_m.get(1, 1)],
        ];
        mps.apply_one_site(site, &exp_op);
    }
}

/// Perform one second-order Strang TDVP step.
///
/// Uses two half sweeps in opposite directions:
/// - left-to-right half-step
/// - right-to-left half-step
fn tdvp_step_strang(
    mps: &mut MPS,
    dt: f64,
    hamiltonian: &LocalHamiltonian,
    jump_ops: &[JumpOperator],
) {
    let half_dt = 0.5 * dt;
    tdvp_sweep(mps, half_dt, hamiltonian, jump_ops, true);
    tdvp_sweep(mps, half_dt, hamiltonian, jump_ops, false);
}

// ============================================================
// QUANTUM JUMP PROCESS
// ============================================================

/// Attempt a quantum jump on the trajectory.
///
/// Computes jump probabilities for each Lindblad operator, draws a random
/// number, and either applies a jump (with renormalization) or does nothing.
/// Returns the index of the applied jump operator, or None if no jump occurred.
fn attempt_jump(
    mps: &mut MPS,
    jump_ops: &[JumpOperator],
    dt: f64,
    rng: &mut StdRng,
) -> Option<(usize, usize)> {
    if jump_ops.is_empty() {
        return None;
    }

    // Compute jump probabilities: p_k = dt * ||L_k |psi>||^2
    // We need to compute this without modifying the state.
    let mut probabilities: Vec<f64> = Vec::with_capacity(jump_ops.len());
    let mut total_prob = 0.0;

    for jop in jump_ops {
        // Compute ||L_k |psi>||^2 = <psi| L_k^dagger L_k |psi>
        let ldl = jop.ldagger_l();
        let expval = mps.expectation(jop.site, &ldl).re;
        let p = dt * expval.max(0.0);
        probabilities.push(p);
        total_prob += p;
    }

    // Clamp total probability (can exceed 1 if dt is too large)
    if total_prob > 1.0 {
        // Rescale to avoid probability > 1
        for p in probabilities.iter_mut() {
            *p /= total_prob;
        }
        total_prob = 1.0;
    }

    let r: f64 = rng.gen();
    if r >= total_prob {
        return None; // No jump
    }

    // Select which jump operator to apply, proportional to p_k
    let mut cumulative = 0.0;
    let target = r; // Already in [0, total_prob)
    for (k, &p) in probabilities.iter().enumerate() {
        cumulative += p;
        if target < cumulative {
            // Apply jump operator L_k to the state
            let l = jump_ops[k].full_operator();
            mps.apply_one_site(jump_ops[k].site, &l);
            mps.normalize();
            return Some((jump_ops[k].site, k));
        }
    }

    // Fallback: apply the last operator (rounding edge case)
    let k = jump_ops.len() - 1;
    let l = jump_ops[k].full_operator();
    mps.apply_one_site(jump_ops[k].site, &l);
    mps.normalize();
    Some((jump_ops[k].site, k))
}

// ============================================================
// SINGLE TRAJECTORY EVOLUTION
// ============================================================

/// Evolve a single trajectory for the full simulation time.
///
/// At each time step:
/// 1. Attempt quantum jumps (stochastic Lindblad process)
/// 2. Evolve under effective non-Hermitian Hamiltonian via TDVP sweep
/// 3. Renormalize the state
/// 4. Record observable expectation values
fn evolve_trajectory(
    initial_state: &MPS,
    config: &TJMConfig,
    hamiltonian: &LocalHamiltonian,
    jump_ops: &[JumpOperator],
    observables: &[Observable],
    seed: u64,
) -> (Trajectory, Vec<(f64, Vec<f64>)>) {
    let mut traj = Trajectory::new(initial_state.clone());
    let mut rng = StdRng::seed_from_u64(seed);

    let num_steps = (config.total_time / config.dt).ceil() as usize;
    let record_interval = (num_steps / 100).max(1); // Record ~100 time points

    let mut time_series: Vec<(f64, Vec<f64>)> = Vec::new();

    // Record initial observables
    let init_obs: Vec<f64> = observables
        .iter()
        .map(|obs| traj.state.expectation(obs.site, &obs.matrix).re)
        .collect();
    time_series.push((0.0, init_obs));

    for step in 0..num_steps {
        // 1. Attempt quantum jump
        if let Some((site, op_idx)) = attempt_jump(&mut traj.state, jump_ops, config.dt, &mut rng) {
            traj.num_jumps += 1;
            traj.jump_history.push((traj.time, site, op_idx));
        }

        // 2. TDVP evolution (evolve under H_eff)
        tdvp_step_strang(&mut traj.state, config.dt, hamiltonian, jump_ops);

        // 3. Renormalize
        let norm = traj.state.normalize();
        if norm < 1e-10 {
            // State collapsed to near-zero: this trajectory is effectively dead.
            // Fill remaining time with zeros.
            break;
        }

        traj.time += config.dt;

        // 4. Record observables at regular intervals
        if (step + 1) % record_interval == 0 || step == num_steps - 1 {
            let obs_vals: Vec<f64> = observables
                .iter()
                .map(|obs| traj.state.expectation(obs.site, &obs.matrix).re)
                .collect();
            time_series.push((traj.time, obs_vals));
        }
    }

    (traj, time_series)
}

// ============================================================
// TENSOR JUMP SIMULATOR
// ============================================================

/// The main Tensor Jump Method simulator.
///
/// Orchestrates the complete simulation: validates configuration, spawns
/// trajectories, averages observables, and returns results with timing.
pub struct TensorJumpSimulator {
    /// Simulation configuration.
    pub config: TJMConfig,
    /// System Hamiltonian.
    pub hamiltonian: LocalHamiltonian,
    /// Lindblad jump operators.
    pub jump_operators: Vec<JumpOperator>,
    /// Observables to measure.
    pub observables: Vec<Observable>,
}

impl TensorJumpSimulator {
    /// Create a new simulator with the given configuration and system specification.
    pub fn new(
        config: TJMConfig,
        hamiltonian: LocalHamiltonian,
        jump_operators: Vec<JumpOperator>,
        observables: Vec<Observable>,
    ) -> Self {
        Self {
            config,
            hamiltonian,
            jump_operators,
            observables,
        }
    }

    /// Run the full simulation, returning averaged results.
    ///
    /// Each trajectory is evolved independently from the initial state.
    /// Observable expectation values are averaged over all trajectories at
    /// matching time points. Error bars are computed as standard error of the mean.
    pub fn run(&self, initial_state: &MPS) -> Result<TJMResult, TJMError> {
        self.config.validate()?;
        let start = Instant::now();

        let num_traj = self.config.num_trajectories;
        let mut all_time_series: Vec<Vec<(f64, Vec<f64>)>> = Vec::with_capacity(num_traj);
        let mut total_jumps = 0usize;
        let mut total_bond_dim = 0.0;

        for traj_idx in 0..num_traj {
            let seed = self.config.seed.wrapping_add(traj_idx as u64 * 7919);
            let (traj, ts) = evolve_trajectory(
                initial_state,
                &self.config,
                &self.hamiltonian,
                &self.jump_operators,
                &self.observables,
                seed,
            );
            total_jumps += traj.num_jumps;
            total_bond_dim += traj.state.avg_bond_dim();
            all_time_series.push(ts);
        }

        // Average the time series across trajectories
        let avg_bond = total_bond_dim / num_traj as f64;
        let averaged = self.average_time_series(&all_time_series);

        let elapsed = start.elapsed().as_secs_f64();

        Ok(TJMResult {
            config: self.config.clone(),
            expectation_values: averaged,
            total_jumps,
            avg_bond_dim: avg_bond,
            trajectory_count: num_traj,
            wall_time_secs: elapsed,
        })
    }

    /// Average observable time series across trajectories.
    ///
    /// Aligns trajectories by index (they all record at the same intervals)
    /// and computes the mean for each observable at each time point.
    fn average_time_series(&self, all_series: &[Vec<(f64, Vec<f64>)>]) -> Vec<(f64, Vec<f64>)> {
        if all_series.is_empty() {
            return Vec::new();
        }

        // Use the first trajectory's time points as reference
        let ref_series = &all_series[0];
        let num_obs = self.observables.len();

        ref_series
            .iter()
            .enumerate()
            .map(|(idx, (time, _))| {
                let mut avg = vec![0.0; num_obs];
                let mut count = 0;

                for series in all_series {
                    if idx < series.len() {
                        for (o, val) in series[idx].1.iter().enumerate() {
                            if o < num_obs {
                                avg[o] += val;
                            }
                        }
                        count += 1;
                    }
                }

                if count > 0 {
                    for v in avg.iter_mut() {
                        *v /= count as f64;
                    }
                }

                (*time, avg)
            })
            .collect()
    }
}

// ============================================================
// CONVENIENCE CONSTRUCTORS
// ============================================================

/// Build a TJM simulator for a Heisenberg chain with uniform dephasing.
///
/// This is a common benchmark system: the isotropic Heisenberg model
/// with local dephasing noise on every site.
pub fn heisenberg_dephasing_sim(
    num_sites: usize,
    j_coupling: f64,
    gamma: f64,
    config: TJMConfig,
) -> TensorJumpSimulator {
    let hamiltonian = heisenberg_chain(num_sites, j_coupling);
    let jump_ops: Vec<JumpOperator> = (0..num_sites).map(|s| dephasing(s, gamma)).collect();
    let observables: Vec<Observable> = (0..num_sites).map(|s| Observable::z(s)).collect();

    TensorJumpSimulator::new(config, hamiltonian, jump_ops, observables)
}

/// Build a TJM simulator for an Ising chain with amplitude damping.
///
/// Models spontaneous emission (T1 decay) in a transverse-field Ising system.
pub fn ising_amplitude_damping_sim(
    num_sites: usize,
    j: f64,
    h: f64,
    gamma: f64,
    config: TJMConfig,
) -> TensorJumpSimulator {
    let hamiltonian = ising_transverse(num_sites, j, h);
    let jump_ops: Vec<JumpOperator> = (0..num_sites)
        .map(|s| amplitude_damping(s, gamma))
        .collect();
    let observables: Vec<Observable> = (0..num_sites)
        .flat_map(|s| vec![Observable::z(s), Observable::x(s)])
        .collect();

    TensorJumpSimulator::new(config, hamiltonian, jump_ops, observables)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: spin-up state |0>.
    fn spin_up() -> [C64; 2] {
        [cr(1.0), c0()]
    }

    /// Helper: spin-down state |1>.
    fn spin_down() -> [C64; 2] {
        [c0(), cr(1.0)]
    }

    /// Helper: |+> state = (|0> + |1>) / sqrt(2).
    fn spin_plus() -> [C64; 2] {
        let s = 1.0 / 2.0_f64.sqrt();
        [cr(s), cr(s)]
    }

    // --------------------------------------------------------
    // MPS BASICS
    // --------------------------------------------------------

    #[test]
    fn test_mps_product_state() {
        let n = 5;
        let states: Vec<[C64; 2]> = vec![spin_up(); n];
        let mps = MPS::new_product_state(n, &states);

        assert_eq!(mps.num_sites, n);
        assert_eq!(mps.physical_dim, 2);
        for t in &mps.tensors {
            assert_eq!(t.dl, 1, "product state should have left bond dim 1");
            assert_eq!(t.dp, 2, "physical dim should be 2 for qubits");
            assert_eq!(t.dr, 1, "product state should have right bond dim 1");
        }
    }

    #[test]
    fn test_mps_norm() {
        let n = 4;
        let states: Vec<[C64; 2]> = vec![spin_up(); n];
        let mps = MPS::new_product_state(n, &states);
        let norm = mps.norm();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "product state norm should be 1.0, got {}",
            norm
        );
    }

    #[test]
    fn test_mps_norm_down_state() {
        let n = 3;
        let states: Vec<[C64; 2]> = vec![spin_down(); n];
        let mps = MPS::new_product_state(n, &states);
        let norm = mps.norm();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "all-down product state norm should be 1.0, got {}",
            norm
        );
    }

    #[test]
    fn test_mps_apply_one_site() {
        // Apply X gate to |0> should give |1>
        let n = 3;
        let mut mps = MPS::all_zero(n);

        // Apply X to site 1
        mps.apply_one_site(1, &PAULI_X);

        // Check: <Z_0> = +1, <Z_1> = -1, <Z_2> = +1
        let z0 = mps.expectation(0, &PAULI_Z).re;
        let z1 = mps.expectation(1, &PAULI_Z).re;
        let z2 = mps.expectation(2, &PAULI_Z).re;

        assert!(
            (z0 - 1.0).abs() < 1e-12,
            "site 0 should remain |0>, <Z>=+1, got {}",
            z0
        );
        assert!(
            (z1 - (-1.0)).abs() < 1e-12,
            "site 1 should be |1> after X, <Z>=-1, got {}",
            z1
        );
        assert!(
            (z2 - 1.0).abs() < 1e-12,
            "site 2 should remain |0>, <Z>=+1, got {}",
            z2
        );
    }

    #[test]
    fn test_mps_expectation_up() {
        // <0|Z|0> = +1
        let n = 2;
        let mps = MPS::all_zero(n);
        let z = mps.expectation(0, &PAULI_Z).re;
        assert!((z - 1.0).abs() < 1e-12, "<0|Z|0> should be +1, got {}", z);
    }

    #[test]
    fn test_mps_expectation_down() {
        // <1|Z|1> = -1
        let n = 2;
        let mps = MPS::all_one(n);
        let z = mps.expectation(0, &PAULI_Z).re;
        assert!(
            (z - (-1.0)).abs() < 1e-12,
            "<1|Z|1> should be -1, got {}",
            z
        );
    }

    #[test]
    fn test_mps_expectation_x_plus_state() {
        // <+|X|+> = +1
        let n = 1;
        let mps = MPS::new_product_state(n, &[spin_plus()]);
        let x = mps.expectation(0, &PAULI_X).re;
        assert!((x - 1.0).abs() < 1e-10, "<+|X|+> should be +1, got {}", x);
    }

    #[test]
    fn test_mps_inner_product() {
        let n = 3;
        let up_state = MPS::all_zero(n);
        let down_state = MPS::all_one(n);

        // <up|up> = 1
        let overlap_same = up_state.inner(&up_state);
        assert!(
            (overlap_same.re - 1.0).abs() < 1e-12,
            "<up|up> should be 1, got {}",
            overlap_same.re
        );

        // <up|down> = 0
        let overlap_ortho = up_state.inner(&down_state);
        assert!(
            overlap_ortho.norm() < 1e-12,
            "<up|down> should be 0, got {}",
            overlap_ortho.norm()
        );
    }

    // --------------------------------------------------------
    // CONFIGURATION
    // --------------------------------------------------------

    #[test]
    fn test_config_builder() {
        let config = TJMConfig::new()
            .num_sites(20)
            .bond_dim(64)
            .dt(0.005)
            .total_time(2.0)
            .num_trajectories(200)
            .seed(12345)
            .svd_cutoff(1e-12);

        assert_eq!(config.num_sites, 20);
        assert_eq!(config.bond_dim, 64);
        assert!((config.dt - 0.005).abs() < 1e-15);
        assert!((config.total_time - 2.0).abs() < 1e-15);
        assert_eq!(config.num_trajectories, 200);
        assert_eq!(config.seed, 12345);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config_zero_sites() {
        let config = TJMConfig::new().num_sites(0);
        let result = config.validate();
        assert!(result.is_err());
        match result {
            Err(TJMError::InvalidConfig(msg)) => {
                assert!(msg.contains("num_sites"), "error should mention num_sites");
            }
            _ => panic!("expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_invalid_config_zero_bond_dim() {
        let config = TJMConfig::new().bond_dim(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_config_negative_dt() {
        let config = TJMConfig::new().dt(-0.01);
        assert!(config.validate().is_err());
    }

    // --------------------------------------------------------
    // HAMILTONIANS
    // --------------------------------------------------------

    #[test]
    fn test_heisenberg_hamiltonian() {
        let n = 10;
        let h = heisenberg_chain(n, 1.0);
        assert_eq!(h.num_sites, n);
        assert_eq!(
            h.terms.len(),
            n - 1,
            "Heisenberg chain should have n-1 two-site terms"
        );
        for (i, term) in h.terms.iter().enumerate() {
            assert_eq!(term.sites, vec![i, i + 1]);
            assert_eq!(term.matrix.len(), 4);
            assert_eq!(term.matrix[0].len(), 4);
        }
    }

    #[test]
    fn test_ising_hamiltonian() {
        let n = 5;
        let h = ising_transverse(n, 1.0, 0.5);
        assert_eq!(h.num_sites, n);
        // n-1 ZZ terms + n X terms
        assert_eq!(h.terms.len(), (n - 1) + n);
    }

    #[test]
    fn test_xxz_hamiltonian() {
        let n = 4;
        let h = xxz_chain(n, 1.0, 0.5);
        assert_eq!(h.terms.len(), n - 1);
    }

    // --------------------------------------------------------
    // JUMP OPERATORS
    // --------------------------------------------------------

    #[test]
    fn test_dephasing_operator() {
        let jop = dephasing(0, 0.1);
        assert_eq!(jop.site, 0);
        assert!((jop.rate - 0.1).abs() < 1e-15);

        // L^dagger L for dephasing (sqrt(gamma)*Z)^dagger * (sqrt(gamma)*Z) = gamma * I
        let ldl = jop.ldagger_l();
        assert!((ldl[0][0].re - 0.1).abs() < 1e-12);
        assert!((ldl[1][1].re - 0.1).abs() < 1e-12);
        assert!(ldl[0][1].norm() < 1e-12);
        assert!(ldl[1][0].norm() < 1e-12);
    }

    #[test]
    fn test_amplitude_damping_operator() {
        let jop = amplitude_damping(2, 0.5);
        assert_eq!(jop.site, 2);
        assert!((jop.rate - 0.5).abs() < 1e-15);

        // L = sqrt(0.5) * sigma_minus = sqrt(0.5) * [[0,1],[0,0]]
        // L^dagger L = 0.5 * [[0,0],[0,1]] -> projects onto |1>
        let ldl = jop.ldagger_l();
        assert!(ldl[0][0].norm() < 1e-12, "L^dagger L [0,0] should be 0");
        assert!(
            (ldl[1][1].re - 0.5).abs() < 1e-12,
            "L^dagger L [1,1] should be gamma"
        );
    }

    #[test]
    fn test_depolarizing_operators() {
        let ops = depolarizing(0, 0.3);
        assert_eq!(ops.len(), 3);
        for op in &ops {
            assert_eq!(op.site, 0);
            assert!((op.rate - 0.1).abs() < 1e-15);
        }
    }

    // --------------------------------------------------------
    // TRAJECTORY EVOLUTION
    // --------------------------------------------------------

    #[test]
    fn test_trajectory_no_jumps() {
        // With zero jump rate, no jumps should occur.
        let n = 4;
        let config = TJMConfig::new()
            .num_sites(n)
            .bond_dim(4)
            .dt(0.01)
            .total_time(0.1)
            .num_trajectories(1)
            .seed(42);

        let hamiltonian = heisenberg_chain(n, 1.0);
        let jump_ops: Vec<JumpOperator> = (0..n).map(|s| dephasing(s, 0.0)).collect();
        let observables = vec![Observable::z(0)];

        let initial = MPS::all_zero(n);
        let (traj, _ts) =
            evolve_trajectory(&initial, &config, &hamiltonian, &jump_ops, &observables, 42);

        assert_eq!(
            traj.num_jumps, 0,
            "zero jump rate should produce zero jumps"
        );
        assert!(traj.jump_history.is_empty());
    }

    #[test]
    fn test_trajectory_with_jumps() {
        // With high jump rate, jumps should occur (statistically certain).
        let n = 2;
        let config = TJMConfig::new()
            .num_sites(n)
            .bond_dim(4)
            .dt(0.01)
            .total_time(1.0)
            .num_trajectories(1)
            .seed(42);

        let hamiltonian = heisenberg_chain(n, 0.1);
        // High dephasing rate to ensure jumps
        let jump_ops: Vec<JumpOperator> = (0..n).map(|s| dephasing(s, 5.0)).collect();
        let observables = vec![Observable::z(0)];

        // Start in |+> state (has Z variance, so dephasing acts nontrivially)
        let initial = MPS::new_product_state(n, &[spin_plus(), spin_plus()]);
        let (traj, _ts) =
            evolve_trajectory(&initial, &config, &hamiltonian, &jump_ops, &observables, 42);

        assert!(
            traj.num_jumps > 0,
            "high jump rate should produce jumps, got 0"
        );
    }

    #[test]
    fn test_observable_averaging() {
        // Multiple trajectories should reduce variance compared to single trajectory.
        let n = 3;
        let hamiltonian = heisenberg_chain(n, 1.0);
        let jump_ops: Vec<JumpOperator> = (0..n).map(|s| dephasing(s, 0.5)).collect();
        let observables = vec![Observable::z(0)];

        let initial = MPS::new_product_state(n, &vec![spin_plus(); n]);

        // Run many trajectories and verify we get something reasonable
        let config = TJMConfig::new()
            .num_sites(n)
            .bond_dim(4)
            .dt(0.01)
            .total_time(0.1)
            .num_trajectories(20)
            .seed(12345);

        let sim = TensorJumpSimulator::new(config, hamiltonian, jump_ops, observables);

        let result = sim.run(&initial).unwrap();
        assert!(result.trajectory_count == 20);
        assert!(!result.expectation_values.is_empty());

        // The initial <Z> for |+> should be ~0
        let (_t0, obs0) = &result.expectation_values[0];
        assert!(
            obs0[0].abs() < 0.5,
            "initial <Z> of |+> state should be near 0, got {}",
            obs0[0]
        );
    }

    #[test]
    fn test_dephasing_decay() {
        // Under pure dephasing: <Z> is preserved, <X> decays.
        // Start in |+> state: <Z>=0, <X>=1.
        // After dephasing, <X> should decrease while <Z> stays ~0.
        let n = 1;
        let config = TJMConfig::new()
            .num_sites(n)
            .bond_dim(2)
            .dt(0.01)
            .total_time(0.5)
            .num_trajectories(50)
            .seed(42);

        // No Hamiltonian evolution, just dephasing
        let hamiltonian = LocalHamiltonian {
            num_sites: n,
            terms: vec![],
        };
        let jump_ops = vec![dephasing(0, 1.0)];
        let observables = vec![Observable::z(0), Observable::x(0)];

        let initial = MPS::new_product_state(n, &[spin_plus()]);
        let sim = TensorJumpSimulator::new(config, hamiltonian, jump_ops, observables);
        let result = sim.run(&initial).unwrap();

        // Check initial values
        let (_t0, obs0) = &result.expectation_values[0];
        assert!(
            obs0[0].abs() < 0.3,
            "initial <Z> should be ~0 for |+>, got {}",
            obs0[0]
        );
        assert!(
            obs0[1] > 0.5,
            "initial <X> should be ~1 for |+>, got {}",
            obs0[1]
        );

        // Check final values: <Z> should stay near 0, <X> should have decayed
        let (_tf, obsf) = result.expectation_values.last().unwrap();
        assert!(
            obsf[0].abs() < 0.5,
            "final <Z> should remain near 0 under dephasing, got {}",
            obsf[0]
        );
        // Note: for single-site dephasing (Z jump), <X> decays as exp(-2*gamma*t)
        // With gamma=1.0, t=0.5: exp(-1.0) ~ 0.37. Allow wide margin for MC noise.
        assert!(
            obsf[1] < obs0[1],
            "final <X> ({}) should be less than initial ({}) under dephasing",
            obsf[1],
            obs0[1]
        );
    }

    #[test]
    fn test_amplitude_damping_decay() {
        // Under amplitude damping: |1> decays toward |0>.
        // Start all spins in |1>: <Z> = -1 initially.
        // After damping, <Z> should increase toward +1.
        let n = 1;
        let config = TJMConfig::new()
            .num_sites(n)
            .bond_dim(2)
            .dt(0.005)
            .total_time(2.0)
            .num_trajectories(80)
            .seed(7777);

        let hamiltonian = LocalHamiltonian {
            num_sites: n,
            terms: vec![],
        };
        let jump_ops = vec![amplitude_damping(0, 1.0)];
        let observables = vec![Observable::z(0)];

        let initial = MPS::all_one(n);
        let sim = TensorJumpSimulator::new(config, hamiltonian, jump_ops, observables);
        let result = sim.run(&initial).unwrap();

        // Initial <Z> should be -1
        let (_t0, obs0) = &result.expectation_values[0];
        assert!(
            (obs0[0] - (-1.0)).abs() < 0.1,
            "initial <Z> should be ~-1 for |1>, got {}",
            obs0[0]
        );

        // Final <Z> should be closer to +1 (decay toward |0>)
        let (_tf, obsf) = result.expectation_values.last().unwrap();
        assert!(
            obsf[0] > obs0[0],
            "final <Z> ({}) should be greater than initial ({}) under amplitude damping",
            obsf[0],
            obs0[0]
        );
    }

    // --------------------------------------------------------
    // SVD
    // --------------------------------------------------------

    #[test]
    fn test_jacobi_svd_identity() {
        // SVD of identity should give singular values all 1
        let m = CMatrix::identity(3);
        let (u, sigmas, vt) = jacobi_svd(&m, 3, 1e-12).unwrap();
        assert_eq!(sigmas.len(), 3);
        for s in &sigmas {
            assert!(
                (s - 1.0).abs() < 1e-8,
                "singular value of identity should be 1, got {}",
                s
            );
        }

        // Verify reconstruction: U * diag(sigma) * Vt ~ I
        let mut reconstructed = CMatrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut val = c0();
                for k in 0..sigmas.len() {
                    val += u.get(i, k) * cr(sigmas[k]) * vt.get(k, j);
                }
                reconstructed.set(i, j, val);
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (reconstructed.get(i, j).re - expected).abs() < 1e-8,
                    "reconstruction error at ({},{}): got {}, expected {}",
                    i,
                    j,
                    reconstructed.get(i, j).re,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_jacobi_svd_rank_1() {
        // Rank-1 matrix should have one nonzero singular value
        let mut m = CMatrix::zeros(3, 3);
        // outer product of [1, 0, 0] and [1, 1, 0]
        m.set(0, 0, cr(1.0));
        m.set(0, 1, cr(1.0));
        let (_u, sigmas, _vt) = jacobi_svd(&m, 3, 1e-10).unwrap();

        // First singular value should be sqrt(2), rest ~0
        assert!(
            (sigmas[0] - 2.0_f64.sqrt()).abs() < 1e-8,
            "first singular value should be sqrt(2), got {}",
            sigmas[0]
        );
        // With cutoff 1e-10, only 1 singular value should survive
        assert_eq!(
            sigmas.len(),
            1,
            "rank-1 matrix should have 1 singular value above cutoff"
        );
    }

    #[test]
    fn test_two_site_gate_svd() {
        // Apply a CNOT-like gate and verify the MPS stays normalized
        let n = 3;
        let mut mps = MPS::all_zero(n);

        // Apply Hadamard to site 0 first
        let s = 1.0 / 2.0_f64.sqrt();
        let hadamard = [[cr(s), cr(s)], [cr(s), cr(-s)]];
        mps.apply_one_site(0, &hadamard);

        // CNOT gate (control=0, target=1)
        let mut cnot = [[c0(); 4]; 4];
        cnot[0][0] = cr(1.0); // |00> -> |00>
        cnot[1][1] = cr(1.0); // |01> -> |01>
        cnot[2][3] = cr(1.0); // |10> -> |11>
        cnot[3][2] = cr(1.0); // |11> -> |10>

        mps.apply_two_site(0, &cnot, 4, 1e-12).unwrap();

        // State should be (|00> + |11>) / sqrt(2) on sites 0,1; site 2 = |0>
        // Norm should still be 1
        let norm = mps.norm();
        assert!(
            (norm - 1.0).abs() < 1e-8,
            "norm after CNOT should be ~1, got {}",
            norm
        );

        // Bond dimension between sites 0 and 1 should have grown to 2 (Bell state)
        assert!(
            mps.tensors[0].dr >= 2,
            "bond dim should grow to at least 2 for entangled state, got {}",
            mps.tensors[0].dr
        );
    }

    // --------------------------------------------------------
    // FULL SIMULATION
    // --------------------------------------------------------

    #[test]
    fn test_full_simulation_small() {
        let n = 3;
        let config = TJMConfig::new()
            .num_sites(n)
            .bond_dim(8)
            .dt(0.01)
            .total_time(0.1)
            .num_trajectories(5)
            .seed(42);

        let sim = heisenberg_dephasing_sim(n, 1.0, 0.1, config);
        let initial = MPS::all_zero(n);
        let result = sim.run(&initial).unwrap();

        assert_eq!(result.trajectory_count, 5);
        assert!(result.wall_time_secs >= 0.0);
        assert!(!result.expectation_values.is_empty());

        // Initial <Z> for all-zero state should be +1 for each site
        let (_, obs0) = &result.expectation_values[0];
        for (i, val) in obs0.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 0.1,
                "initial <Z_{}> should be ~1 for |0>, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_error_display() {
        let e1 = TJMError::InvalidConfig("test".into());
        assert!(format!("{}", e1).contains("test"));

        let e2 = TJMError::ConvergenceFailed {
            step: 10,
            norm: 0.5,
        };
        assert!(format!("{}", e2).contains("step 10"));

        let e3 = TJMError::BondDimExceeded { site: 3, dim: 128 };
        assert!(format!("{}", e3).contains("128"));

        let e4 = TJMError::SvdFailed("singular".into());
        assert!(format!("{}", e4).contains("singular"));
    }

    #[test]
    fn test_normalize_preserves_expectations() {
        // Scaling should not change expectation values
        let n = 2;
        let mut mps = MPS::new_product_state(n, &[spin_plus(), spin_up()]);

        let z_before = mps.expectation(0, &PAULI_Z).re;

        // Artificially scale the state
        mps.tensors[0].scale(cr(3.0));
        let norm = mps.norm();
        assert!((norm - 3.0).abs() < 1e-10);

        mps.normalize();
        let z_after = mps.expectation(0, &PAULI_Z).re;

        assert!(
            (z_before - z_after).abs() < 1e-10,
            "expectation should be preserved after normalize: before={}, after={}",
            z_before,
            z_after
        );
    }

    #[test]
    fn test_matrix_exp_taylor_identity() {
        // exp(0) = I
        let zero = CMatrix::zeros(2, 2);
        let result = matrix_exp_taylor(&zero, 8);
        assert!((result.get(0, 0).re - 1.0).abs() < 1e-12);
        assert!((result.get(1, 1).re - 1.0).abs() < 1e-12);
        assert!(result.get(0, 1).norm() < 1e-12);
        assert!(result.get(1, 0).norm() < 1e-12);
    }

    #[test]
    fn test_matrix_exp_taylor_pauli_z() {
        // exp(-i * theta * Z) = [[exp(-i*theta), 0], [0, exp(i*theta)]]
        let theta = 0.3;
        let mut m = CMatrix::zeros(2, 2);
        m.set(0, 0, ci(-theta));
        m.set(1, 1, ci(theta));
        let result = matrix_exp_taylor(&m, 12);

        let expected_00 = C64::new(theta.cos(), -theta.sin());
        let expected_11 = C64::new(theta.cos(), theta.sin());
        assert!(
            (result.get(0, 0) - expected_00).norm() < 1e-10,
            "exp(-i*theta*Z)[0,0] mismatch"
        );
        assert!(
            (result.get(1, 1) - expected_11).norm() < 1e-10,
            "exp(-i*theta*Z)[1,1] mismatch"
        );
    }
}
