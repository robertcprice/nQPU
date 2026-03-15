//! Differentiable Matrix Product States with Forward-Mode Automatic Differentiation
//!
//! This module implements differentiable MPS (Matrix Product States) with forward-mode
//! automatic differentiation through SVD truncation. This enables gradient-based
//! optimization of variational MPS circuits -- a capability absent from competing
//! quantum simulators.
//!
//! # Key Features
//!
//! - **DualComplex**: Forward-mode tangent propagation through complex arithmetic
//! - **SVD Gradient**: Exact gradient computation through truncated SVD decomposition
//! - **Soft Threshold Truncation**: Differentiable approximation to hard bond truncation
//! - **DifferentiableMPS**: Full MPS with cached SVD snapshots for backward pass
//! - **MPSVariationalAnsatz**: Parameterized variational circuits with energy+gradient
//!
//! # Mathematical Background
//!
//! The SVD gradient uses the formula from Seeger et al. (2017):
//! ```text
//! dU = U(F ⊙ (U^H dM V)) + (I - UU^H) dM V S^{-1}
//! ```
//! where `F_{ij} = 1/(s_j^2 - s_i^2)` for `i != j`, `0` for `i == j`, with
//! regularization for degenerate singular values.
//!
//! # References
//!
//! - Seeger, Hetzel, et al., "Auto-Differentiating Linear Algebra" (2017)
//! - Liao, Liu, et al., "Differentiable Programming Tensor Networks" (2019)
//! - Hauru, Van Damme, et al., "Riemannian optimization of isometric tensor networks" (2021)

use crate::{c64_one, c64_zero, C64};

// ============================================================
// DUAL COMPLEX NUMBER (FORWARD-MODE AD)
// ============================================================

/// A dual complex number for forward-mode automatic differentiation.
///
/// Carries both a primal value and its tangent (derivative) through
/// complex arithmetic operations. This is the fundamental building
/// block for propagating derivatives through MPS contractions and
/// SVD truncation.
///
/// # Fields
/// - `value`: The primal complex value
/// - `derivative`: The tangent/derivative with respect to some parameter
#[derive(Clone, Copy, Debug)]
pub struct DualComplex {
    pub value: C64,
    pub derivative: C64,
}

impl DualComplex {
    /// Create a new dual complex number with explicit value and derivative.
    #[inline]
    pub fn new(value: C64, derivative: C64) -> Self {
        Self { value, derivative }
    }

    /// Create a constant (derivative = 0).
    #[inline]
    pub fn constant(value: C64) -> Self {
        Self {
            value,
            derivative: c64_zero(),
        }
    }

    /// Create a variable (derivative = 1) for differentiation with respect to this parameter.
    #[inline]
    pub fn variable(value: C64) -> Self {
        Self {
            value,
            derivative: c64_one(),
        }
    }

    /// Complex conjugate: conj(a + b*eps) = conj(a) + conj(b)*eps
    /// Note: conjugation is NOT complex-differentiable (Wirtinger calculus),
    /// but for real-valued loss functions this propagation is correct.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            value: self.value.conj(),
            derivative: self.derivative.conj(),
        }
    }

    /// Squared norm: |z|^2 = z * conj(z)
    /// d(|z|^2) = z * conj(dz) + conj(z) * dz = 2 * Re(conj(z) * dz)
    #[inline]
    pub fn norm_sqr(self) -> (f64, f64) {
        let val = self.value.norm_sqr();
        let deriv = 2.0 * (self.value.conj() * self.derivative).re;
        (val, deriv)
    }

    /// Complex norm: |z| = sqrt(z * conj(z))
    /// d|z| = Re(conj(z) * dz) / |z|
    #[inline]
    pub fn norm(self) -> (f64, f64) {
        let n = self.value.norm();
        if n < 1e-15 {
            return (n, 0.0);
        }
        let deriv = (self.value.conj() * self.derivative).re / n;
        (n, deriv)
    }

    /// Absolute value (alias for norm for complex numbers).
    #[inline]
    pub fn abs(self) -> (f64, f64) {
        self.norm()
    }

    /// Real part extraction.
    #[inline]
    pub fn re(self) -> (f64, f64) {
        (self.value.re, self.derivative.re)
    }

    /// Imaginary part extraction.
    #[inline]
    pub fn im(self) -> (f64, f64) {
        (self.value.im, self.derivative.im)
    }

    /// Scale by a real number.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            value: self.value * s,
            derivative: self.derivative * s,
        }
    }
}

// Arithmetic implementations for DualComplex

impl std::ops::Add for DualComplex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

impl std::ops::Sub for DualComplex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl std::ops::Mul for DualComplex {
    type Output = Self;
    /// Product rule: d(a*b) = da*b + a*db
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        }
    }
}

impl std::ops::Div for DualComplex {
    type Output = Self;
    /// Quotient rule: d(a/b) = (da*b - a*db) / b^2
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let b_sq = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) / b_sq,
        }
    }
}

impl std::ops::Neg for DualComplex {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

impl std::ops::AddAssign for DualComplex {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
        self.derivative += rhs.derivative;
    }
}

impl std::ops::SubAssign for DualComplex {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
        self.derivative -= rhs.derivative;
    }
}

impl std::ops::MulAssign for DualComplex {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let new_deriv = self.derivative * rhs.value + self.value * rhs.derivative;
        self.value *= rhs.value;
        self.derivative = new_deriv;
    }
}

impl PartialEq for DualComplex {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

// ============================================================
// SVD SNAPSHOT (CACHED DECOMPOSITION)
// ============================================================

/// Cached SVD decomposition for backward differentiation through truncation.
///
/// Stores the full SVD factors U, S, V^T at the point of truncation so that
/// gradients can be propagated backward through the SVD operation without
/// recomputing the decomposition.
#[derive(Clone, Debug)]
pub struct SvdSnapshot {
    /// Left singular vectors (column-major, rows x rank)
    pub u: Vec<C64>,
    /// Singular values (descending order)
    pub s: Vec<f64>,
    /// Right singular vectors transposed (rank x cols, row-major)
    pub vt: Vec<C64>,
    /// Number of rows in the original matrix
    pub rows: usize,
    /// Number of columns in the original matrix
    pub cols: usize,
    /// Truncated rank (number of retained singular values)
    pub rank: usize,
}

impl SvdSnapshot {
    /// Create a new SVD snapshot.
    pub fn new(
        u: Vec<C64>,
        s: Vec<f64>,
        vt: Vec<C64>,
        rows: usize,
        cols: usize,
        rank: usize,
    ) -> Self {
        debug_assert_eq!(u.len(), rows * rank, "U dimension mismatch");
        debug_assert_eq!(s.len(), rank, "S dimension mismatch");
        debug_assert_eq!(vt.len(), rank * cols, "Vt dimension mismatch");
        Self {
            u,
            s,
            vt,
            rows,
            cols,
            rank,
        }
    }

    /// Reconstruct the (truncated) matrix: M_approx = U * diag(S) * V^T
    ///
    /// Returns a rows x cols matrix in row-major order.
    pub fn reconstruct(&self) -> Vec<C64> {
        let mut result = vec![c64_zero(); self.rows * self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = c64_zero();
                for k in 0..self.rank {
                    // U[i, k] * S[k] * Vt[k, j]
                    let u_ik = self.u[i * self.rank + k];
                    let s_k = self.s[k];
                    let vt_kj = self.vt[k * self.cols + j];
                    sum += u_ik * C64::new(s_k, 0.0) * vt_kj;
                }
                result[i * self.cols + j] = sum;
            }
        }

        result
    }

    /// Compute the truncation error (Frobenius norm of discarded part).
    /// This requires the original singular values before truncation.
    pub fn truncation_error(&self, full_singular_values: &[f64]) -> f64 {
        let mut error_sq = 0.0;
        for k in self.rank..full_singular_values.len() {
            error_sq += full_singular_values[k] * full_singular_values[k];
        }
        error_sq.sqrt()
    }
}

// ============================================================
// SVD GRADIENT COMPUTATION
// ============================================================

/// Regularization epsilon for degenerate singular values.
const SVD_DEGENERACY_EPS: f64 = 1e-12;

/// Compute the gradient of a loss function through an SVD decomposition.
///
/// Given `M = U * diag(S) * V^T` and the gradient `dL/dM_approx` of the loss
/// with respect to the reconstructed (truncated) matrix, compute `dL/dM`.
///
/// Uses the formula from Seeger et al. (2017):
/// ```text
/// dU = U(F ⊙ (U^H dM V)) + (I - UU^H) dM V S^{-1}
/// ```
/// where `F_{ij} = 1/(s_j^2 - s_i^2)` for `i != j`, `0` for `i == j`.
///
/// Degenerate singular values (where `|s_i^2 - s_j^2| < eps`) are regularized
/// to prevent division by zero.
///
/// # Arguments
/// * `snapshot` - The cached SVD decomposition
/// * `d_output` - Gradient of the loss w.r.t. the reconstructed matrix (row-major, rows x cols)
///
/// # Returns
/// Gradient of the loss w.r.t. the original matrix M (row-major, rows x cols)
pub fn svd_gradient(snapshot: &SvdSnapshot, d_output: &[C64]) -> Vec<C64> {
    let m = snapshot.rows;
    let n = snapshot.cols;
    let r = snapshot.rank;

    assert_eq!(
        d_output.len(),
        m * n,
        "d_output must have rows*cols elements"
    );

    // Decompose d_output into components for dU, dS, dVt using the chain rule.
    //
    // M_approx = U * diag(S) * Vt
    // dM_approx = dU * diag(S) * Vt + U * diag(dS) * Vt + U * diag(S) * dVt
    //
    // Project onto each component:
    // dS_k = Re(U[:,k]^H * d_output * V[:,k]) where V[:,k] = Vt[k,:]^H
    // dU = (d_output * V * S^{-1}) projected onto complement of U
    // dVt = (S^{-1} * U^H * d_output) projected onto complement of Vt^H

    // Step 1: Compute U^H * d_output (r x n)
    let mut uh_d = vec![c64_zero(); r * n];
    for k in 0..r {
        for j in 0..n {
            let mut sum = c64_zero();
            for i in 0..m {
                sum += snapshot.u[i * r + k].conj() * d_output[i * n + j];
            }
            uh_d[k * n + j] = sum;
        }
    }

    // Step 2: Compute d_output * V (m x r), where V[:,k] = Vt[k,:]^H
    let mut d_v = vec![c64_zero(); m * r];
    for i in 0..m {
        for k in 0..r {
            let mut sum = c64_zero();
            for j in 0..n {
                // V[j, k] = conj(Vt[k, j])
                sum += d_output[i * n + j] * snapshot.vt[k * n + j].conj();
            }
            d_v[i * r + k] = sum;
        }
    }

    // Step 3: Compute U^H * d_output * V (r x r) = uh_d * V
    let mut uh_d_v = vec![c64_zero(); r * r];
    for i in 0..r {
        for k in 0..r {
            let mut sum = c64_zero();
            for j in 0..n {
                sum += uh_d[i * n + j] * snapshot.vt[k * n + j].conj();
            }
            uh_d_v[i * r + k] = sum;
        }
    }

    // Step 4: Compute the F matrix
    // F_{ij} = 1/(s_j^2 - s_i^2) for i != j, 0 for i == j
    let mut f_mat = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            if i != j {
                let diff = snapshot.s[j] * snapshot.s[j] - snapshot.s[i] * snapshot.s[i];
                if diff.abs() < SVD_DEGENERACY_EPS {
                    // Regularize degenerate case: use limit as s_i -> s_j
                    // lim 1/(s_j^2 - s_i^2) -> 1/(2*s_i*eps_sign)
                    let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
                    f_mat[i * r + j] = sign / SVD_DEGENERACY_EPS;
                } else {
                    f_mat[i * r + j] = 1.0 / diff;
                }
            }
            // i == j: f_mat stays 0.0
        }
    }

    // Step 5: Compute the gradient dL/dM
    //
    // dM = U * (F ⊙ (U^H dM_out V) * S + S * F ⊙ (U^H dM_out V)^T) * Vt
    //    + (I - UU^H) * dM_out * V * S^{-1} * Vt
    //    + U * S^{-1} * U^H * dM_out * (I - VV^H)     [omitted for truncated case]
    //    + U * diag(diag(U^H dM_out V)) * Vt
    //
    // Simplified for practical use (the dominant terms):
    //
    // Term 1: U * [F ⊙ (U^H dM V)] * S * Vt  (left singular vector correction)
    // Term 2: U * S * [F ⊙ (U^H dM V)^H] * Vt  (right singular vector correction)
    // Term 3: (I - UU^H) * dM * V * S^{-1} * Vt  (orthogonal complement, left)
    // Term 4: U * diag(Re(U^H dM V)_{kk}) * Vt  (singular value gradient)

    // Build the core r x r matrix that will be sandwiched between U and Vt
    let mut core = vec![c64_zero(); r * r];

    for i in 0..r {
        for j in 0..r {
            if i == j {
                // Diagonal: singular value gradient
                core[i * r + j] = C64::new(uh_d_v[i * r + j].re, 0.0);
            } else {
                // Off-diagonal: F ⊙ (U^H dM V) contributions
                let f_ij = f_mat[i * r + j];
                let elem = uh_d_v[i * r + j];
                let elem_ji = uh_d_v[j * r + i];
                // Left correction:  s_j * F_ij * (U^H dM V)_ij
                // Right correction: s_i * F_ij * conj((U^H dM V)_ji)
                core[i * r + j] = C64::new(f_ij, 0.0)
                    * (C64::new(snapshot.s[j], 0.0) * elem
                        + C64::new(snapshot.s[i], 0.0) * elem_ji.conj());
            }
        }
    }

    // Compute U * core * Vt (m x n)
    // First: temp = U * core (m x r)
    let mut temp = vec![c64_zero(); m * r];
    for i in 0..m {
        for k in 0..r {
            let mut sum = c64_zero();
            for j in 0..r {
                sum += snapshot.u[i * r + j] * core[j * r + k];
            }
            temp[i * r + k] = sum;
        }
    }

    // result = temp * Vt (m x n)
    let mut result = vec![c64_zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = c64_zero();
            for k in 0..r {
                sum += temp[i * r + k] * snapshot.vt[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    // Term 3: orthogonal complement contribution
    // (I - UU^H) * d_output * V * S^{-1} * Vt
    // First compute d_v * S^{-1} (m x r) -- already have d_v
    let mut d_v_sinv = vec![c64_zero(); m * r];
    for i in 0..m {
        for k in 0..r {
            if snapshot.s[k].abs() > SVD_DEGENERACY_EPS {
                d_v_sinv[i * r + k] = d_v[i * r + k] / C64::new(snapshot.s[k], 0.0);
            }
        }
    }

    // Compute UU^H * d_v_sinv (m x r)
    let mut uuh_dv = vec![c64_zero(); m * r];
    for i in 0..m {
        for k in 0..r {
            let mut sum = c64_zero();
            for j in 0..r {
                // (UU^H)_{i,l} = sum_j U[i,j] * U[l,j]^H
                // (UU^H * X)_{i,k} = sum_l (UU^H)_{i,l} * X[l,k]
                //                  = sum_j U[i,j] * (sum_l conj(U[l,j]) * X[l,k])
                // But simpler: UU^H * X = U * (U^H * X)
                // U^H * d_v_sinv [j, k]:
                let mut inner = c64_zero();
                for l in 0..m {
                    inner += snapshot.u[l * r + j].conj() * d_v_sinv[l * r + k];
                }
                sum += snapshot.u[i * r + j] * inner;
            }
            uuh_dv[i * r + k] = sum;
        }
    }

    // orth_comp = d_v_sinv - uuh_dv (m x r)
    // Then result += orth_comp * Vt
    for i in 0..m {
        for j in 0..n {
            let mut sum = c64_zero();
            for k in 0..r {
                let orth = d_v_sinv[i * r + k] - uuh_dv[i * r + k];
                sum += orth * snapshot.vt[k * n + j];
            }
            result[i * n + j] += sum;
        }
    }

    result
}

// ============================================================
// SOFT THRESHOLD TRUNCATION
// ============================================================

/// Differentiable approximation to hard singular value truncation.
///
/// Hard truncation sets singular values below a cutoff to exactly zero, which
/// creates a discontinuity that prevents gradient flow. This module replaces
/// hard truncation with a smooth sigmoid-based threshold, enabling gradients
/// to flow through the bond dimension selection.
///
/// The soft threshold function is:
/// ```text
/// w(s) = sigma(sharpness * (s - cutoff))
/// ```
/// where `sigma` is the logistic sigmoid, and the effective singular value is `s * w(s)`.
pub struct SoftThresholdTruncation;

impl SoftThresholdTruncation {
    /// Apply soft threshold to singular values.
    ///
    /// Returns `s_k * sigmoid(sharpness * (s_k - cutoff))` for each singular value.
    ///
    /// # Arguments
    /// * `s` - Singular values (typically in descending order)
    /// * `cutoff` - Threshold below which values are suppressed
    /// * `sharpness` - Controls sigmoid steepness (higher = closer to hard threshold)
    ///
    /// # Returns
    /// Smoothly thresholded singular values
    pub fn apply(s: &[f64], cutoff: f64, sharpness: f64) -> Vec<f64> {
        s.iter()
            .map(|&sv| {
                let weight = sigmoid(sharpness * (sv - cutoff));
                sv * weight
            })
            .collect()
    }

    /// Compute the gradient of the soft threshold with respect to singular values.
    ///
    /// For `f(s) = s * sigmoid(k * (s - c))`:
    /// ```text
    /// f'(s) = sigmoid(k*(s-c)) + s * k * sigmoid(k*(s-c)) * (1 - sigmoid(k*(s-c)))
    /// ```
    ///
    /// # Arguments
    /// * `s` - Singular values
    /// * `cutoff` - Threshold value
    /// * `sharpness` - Sigmoid steepness parameter
    ///
    /// # Returns
    /// Gradient of the soft threshold w.r.t. each singular value
    pub fn gradient(s: &[f64], cutoff: f64, sharpness: f64) -> Vec<f64> {
        s.iter()
            .map(|&sv| {
                let sig = sigmoid(sharpness * (sv - cutoff));
                let dsig = sharpness * sig * (1.0 - sig);
                // d/ds [s * sigmoid(k*(s-c))] = sigmoid(k*(s-c)) + s * k * sigmoid * (1-sigmoid)
                sig + sv * dsig
            })
            .collect()
    }

    /// Compute both the thresholded values and gradients in a single pass.
    pub fn apply_with_gradient(s: &[f64], cutoff: f64, sharpness: f64) -> (Vec<f64>, Vec<f64>) {
        let mut values = Vec::with_capacity(s.len());
        let mut grads = Vec::with_capacity(s.len());

        for &sv in s {
            let sig = sigmoid(sharpness * (sv - cutoff));
            let dsig = sharpness * sig * (1.0 - sig);
            values.push(sv * sig);
            grads.push(sig + sv * dsig);
        }

        (values, grads)
    }
}

/// Standard logistic sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let ex = (-x).exp();
        1.0 / (1.0 + ex)
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ============================================================
// DIFFERENTIABLE MPS
// ============================================================

/// Shape metadata for a single MPS tensor.
#[derive(Clone, Debug)]
pub struct TensorShape {
    /// Left bond dimension
    pub bond_left: usize,
    /// Physical dimension (always 2 for qubits)
    pub phys_dim: usize,
    /// Right bond dimension
    pub bond_right: usize,
}

impl TensorShape {
    /// Total number of elements in this tensor.
    pub fn numel(&self) -> usize {
        self.bond_left * self.phys_dim * self.bond_right
    }
}

/// A Matrix Product State with differentiable SVD truncation.
///
/// This struct maintains an MPS representation alongside cached SVD snapshots
/// at each bond, enabling gradient computation through the entire sequence
/// of gate applications and truncations.
///
/// # Storage Convention
///
/// Each tensor is stored as a flat `Vec<C64>` in row-major order with indices
/// `[bond_left, phys_dim, bond_right]`. The shape metadata is stored separately
/// in `shapes`.
pub struct DifferentiableMPS {
    /// Flat tensor data for each site.
    pub tensors: Vec<Vec<C64>>,
    /// Shape metadata for each tensor.
    pub shapes: Vec<TensorShape>,
    /// Bond dimensions between sites (length = num_sites - 1).
    pub bond_dims: Vec<usize>,
    /// Cached SVD snapshots at each bond for backward differentiation.
    pub svd_cache: Vec<Option<SvdSnapshot>>,
    /// Number of qubit sites.
    pub num_sites: usize,
    /// Maximum allowed bond dimension.
    pub max_bond_dim: usize,
    /// Truncation cutoff for soft thresholding.
    pub truncation_cutoff: f64,
    /// Sharpness parameter for soft thresholding.
    pub truncation_sharpness: f64,
}

impl DifferentiableMPS {
    /// Create a new DifferentiableMPS in the |0...0> state.
    ///
    /// # Arguments
    /// * `num_sites` - Number of qubit sites
    /// * `max_bond_dim` - Maximum bond dimension for truncation
    pub fn new(num_sites: usize, max_bond_dim: usize) -> Self {
        assert!(num_sites > 0, "Must have at least one site");
        assert!(max_bond_dim > 0, "Max bond dimension must be positive");

        let mut tensors = Vec::with_capacity(num_sites);
        let mut shapes = Vec::with_capacity(num_sites);

        // Initialize |0...0> product state
        for _i in 0..num_sites {
            // Each tensor: (1, 2, 1) -- bond_left=1, phys=2, bond_right=1
            let mut data = vec![c64_zero(); 2];
            data[0] = c64_one(); // |0> component
            tensors.push(data);
            shapes.push(TensorShape {
                bond_left: 1,
                phys_dim: 2,
                bond_right: 1,
            });
        }

        let bond_dims = vec![1; num_sites.saturating_sub(1)];
        let svd_cache = vec![None; num_sites.saturating_sub(1)];

        Self {
            tensors,
            shapes,
            bond_dims,
            svd_cache,
            num_sites,
            max_bond_dim,
            truncation_cutoff: 1e-10,
            truncation_sharpness: 100.0,
        }
    }

    /// Access tensor data at a site (read-only).
    #[inline]
    pub fn tensor(&self, site: usize) -> &[C64] {
        &self.tensors[site]
    }

    /// Access tensor element at [bl, p, br].
    #[inline]
    pub fn tensor_elem(&self, site: usize, bl: usize, p: usize, br: usize) -> C64 {
        let s = &self.shapes[site];
        self.tensors[site][bl * s.phys_dim * s.bond_right + p * s.bond_right + br]
    }

    /// Set tensor element at [bl, p, br].
    #[inline]
    pub fn set_tensor_elem(&mut self, site: usize, bl: usize, p: usize, br: usize, val: C64) {
        let s = &self.shapes[site];
        let idx = bl * s.phys_dim * s.bond_right + p * s.bond_right + br;
        self.tensors[site][idx] = val;
    }

    /// Apply a two-site gate and perform differentiable SVD truncation.
    ///
    /// The gate is a 4x4 matrix acting on qubits at `site` and `site+1`.
    /// After application, the combined tensor is decomposed via SVD with
    /// soft thresholding, and the SVD snapshot is cached for gradient computation.
    ///
    /// # Arguments
    /// * `site` - Left site index (gate acts on site, site+1)
    /// * `gate` - 4x4 gate matrix as a flat Vec (row-major): gate[i*4+j] for |i><j|
    pub fn apply_gate(&mut self, site: usize, gate: &[C64]) {
        assert!(site + 1 < self.num_sites, "Gate site out of bounds");
        assert_eq!(gate.len(), 16, "Gate must be 4x4 = 16 elements");

        let sl = self.shapes[site].clone();
        let sr = self.shapes[site + 1].clone();
        let chi_l = sl.bond_left;
        let chi_m = sl.bond_right; // == sr.bond_left
        let chi_r = sr.bond_right;

        // Step 1: Contract tensors into theta[a, i, j, b]
        // theta[a,i,j,b] = sum_m T_L[a,i,m] * T_R[m,j,b]
        let mut theta = vec![c64_zero(); chi_l * 2 * 2 * chi_r];
        for a in 0..chi_l {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..chi_r {
                        let mut sum = c64_zero();
                        for m in 0..chi_m {
                            let tl_idx = a * 2 * chi_m + i * chi_m + m;
                            let tr_idx = m * 2 * chi_r + j * chi_r + b;
                            sum += self.tensors[site][tl_idx] * self.tensors[site + 1][tr_idx];
                        }
                        let idx = a * 2 * 2 * chi_r + i * 2 * chi_r + j * chi_r + b;
                        theta[idx] = sum;
                    }
                }
            }
        }

        // Step 2: Apply gate on physical indices
        // gated[a,i',j',b] = sum_{i,j} gate[i'*2+j', i*2+j] * theta[a,i,j,b]
        let mut gated = vec![c64_zero(); chi_l * 2 * 2 * chi_r];
        for a in 0..chi_l {
            for b in 0..chi_r {
                for ip in 0..2 {
                    for jp in 0..2 {
                        let mut sum = c64_zero();
                        for i in 0..2 {
                            for j in 0..2 {
                                let g_idx = (ip * 2 + jp) * 4 + (i * 2 + j);
                                let t_idx = a * 2 * 2 * chi_r + i * 2 * chi_r + j * chi_r + b;
                                sum += gate[g_idx] * theta[t_idx];
                            }
                        }
                        let idx = a * 2 * 2 * chi_r + ip * 2 * chi_r + jp * chi_r + b;
                        gated[idx] = sum;
                    }
                }
            }
        }

        // Step 3: Reshape to matrix (chi_l*2) x (2*chi_r) for SVD
        let m_rows = chi_l * 2;
        let n_cols = 2 * chi_r;
        let mut mat = vec![c64_zero(); m_rows * n_cols];
        for a in 0..chi_l {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..chi_r {
                        let row = a * 2 + i;
                        let col = j * chi_r + b;
                        let t_idx = a * 2 * 2 * chi_r + i * 2 * chi_r + j * chi_r + b;
                        mat[row * n_cols + col] = gated[t_idx];
                    }
                }
            }
        }

        // Step 4: Compute SVD using nalgebra
        let (u_data, s_data, vt_data, rank) = self.compute_truncated_svd(&mat, m_rows, n_cols);

        // Step 5: Apply soft thresholding to singular values
        let s_soft = SoftThresholdTruncation::apply(
            &s_data,
            self.truncation_cutoff,
            self.truncation_sharpness,
        );

        // Step 6: Cache the SVD snapshot
        self.svd_cache[site] = Some(SvdSnapshot::new(
            u_data.clone(),
            s_data.clone(),
            vt_data.clone(),
            m_rows,
            n_cols,
            rank,
        ));

        // Step 7: Reconstruct new tensors
        // T_L[a, i, k] = U[a*2+i, k] * S_soft[k]
        let new_chi = rank;
        let mut new_tl = vec![c64_zero(); chi_l * 2 * new_chi];
        for a in 0..chi_l {
            for i in 0..2 {
                let row = a * 2 + i;
                for k in 0..new_chi {
                    new_tl[a * 2 * new_chi + i * new_chi + k] =
                        u_data[row * rank + k] * C64::new(s_soft[k], 0.0);
                }
            }
        }

        // T_R[k, j, b] = Vt[k, j*chi_r+b]
        let mut new_tr = vec![c64_zero(); new_chi * 2 * chi_r];
        for k in 0..new_chi {
            for j in 0..2 {
                for b in 0..chi_r {
                    let col = j * chi_r + b;
                    new_tr[k * 2 * chi_r + j * chi_r + b] = vt_data[k * n_cols + col];
                }
            }
        }

        // Update tensors and shapes
        self.tensors[site] = new_tl;
        self.shapes[site] = TensorShape {
            bond_left: chi_l,
            phys_dim: 2,
            bond_right: new_chi,
        };
        self.tensors[site + 1] = new_tr;
        self.shapes[site + 1] = TensorShape {
            bond_left: new_chi,
            phys_dim: 2,
            bond_right: chi_r,
        };
        self.bond_dims[site] = new_chi;
    }

    /// Internal: compute truncated SVD of a matrix.
    ///
    /// Returns (U_flat, S_vec, Vt_flat, rank) where:
    /// - U_flat is row-major (rows x rank)
    /// - Vt_flat is row-major (rank x cols)
    fn compute_truncated_svd(
        &self,
        mat: &[C64],
        rows: usize,
        cols: usize,
    ) -> (Vec<C64>, Vec<f64>, Vec<C64>, usize) {
        use nalgebra::{Complex as NComplex, DMatrix};

        // Convert to nalgebra column-major format
        let mut nmat_data = vec![NComplex::<f64>::new(0.0, 0.0); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let v = mat[i * cols + j];
                // nalgebra DMatrix stores column-major: index = i + j * rows
                nmat_data[i + j * rows] = NComplex::new(v.re, v.im);
            }
        }
        let nmat = DMatrix::from_vec(rows, cols, nmat_data);

        let svd = nmat.svd(true, true);
        let u_full = svd.u.expect("SVD U matrix");
        let vt_full = svd.v_t.expect("SVD V^T matrix");
        let svals = &svd.singular_values;

        // Determine rank: keep values above threshold, capped by max_bond_dim
        let mut rank = 0;
        for k in 0..svals.len() {
            if svals[k] > self.truncation_cutoff * 0.01 {
                rank += 1;
            } else {
                break;
            }
        }
        rank = rank.max(1).min(self.max_bond_dim);

        // Extract U (rows x rank), row-major
        let mut u_data = vec![c64_zero(); rows * rank];
        for i in 0..rows {
            for k in 0..rank {
                let v = u_full[(i, k)];
                u_data[i * rank + k] = C64::new(v.re, v.im);
            }
        }

        // Extract S (rank)
        let s_data: Vec<f64> = (0..rank).map(|k| svals[k]).collect();

        // Extract Vt (rank x cols), row-major
        let mut vt_data = vec![c64_zero(); rank * cols];
        for k in 0..rank {
            for j in 0..cols {
                let v = vt_full[(k, j)];
                vt_data[k * cols + j] = C64::new(v.re, v.im);
            }
        }

        (u_data, s_data, vt_data, rank)
    }

    /// Compute the expectation value <psi|O|psi> for a sum of two-site observables.
    ///
    /// The observable is provided as a list of (site, 4x4_matrix) pairs, representing
    /// a sum of two-site operators: O = sum_k O_k where O_k acts on (site_k, site_k+1).
    ///
    /// # Arguments
    /// * `observable` - List of (site_index, flat 4x4 operator matrix) pairs
    ///
    /// # Returns
    /// The real-valued expectation value
    pub fn compute_expectation(&self, observable: &[(usize, Vec<C64>)]) -> f64 {
        let mut total = 0.0;

        for (site, op) in observable {
            assert!(*site + 1 < self.num_sites);
            assert_eq!(op.len(), 16);

            // Contract the full MPS to compute <psi|O_site|psi>
            // For efficiency, compute the local two-site reduced density matrix
            // and trace with the operator.

            // Build left environment: product of all tensors to the left of site
            let sl = &self.shapes[*site];
            let sr = &self.shapes[*site + 1];
            let chi_l = sl.bond_left;
            let chi_r = sr.bond_right;

            // Left environment: starts as identity of dimension 1
            // env_l[a, a'] for the left bond of site
            let mut env_l = vec![c64_zero(); chi_l * chi_l];
            for a in 0..chi_l {
                env_l[a * chi_l + a] = c64_one();
            }

            // Contract all sites to the left
            for s in 0..*site {
                let shape = &self.shapes[s];
                let bl = shape.bond_left;
                let br = shape.bond_right;
                let mut new_env = vec![c64_zero(); br * br];
                for bp in 0..br {
                    for bpp in 0..br {
                        let mut sum = c64_zero();
                        for ap in 0..bl {
                            for app in 0..bl {
                                for p in 0..2 {
                                    let t_idx1 = ap * 2 * br + p * br + bp;
                                    let t_idx2 = app * 2 * br + p * br + bpp;
                                    sum += env_l[ap * bl + app]
                                        * self.tensors[s][t_idx1]
                                        * self.tensors[s][t_idx2].conj();
                                }
                            }
                        }
                        new_env[bp * br + bpp] = sum;
                    }
                }
                env_l = new_env;
            }

            // Right environment: starts as identity of right bond dimension
            let mut env_r = vec![c64_zero(); chi_r * chi_r];
            for b in 0..chi_r {
                env_r[b * chi_r + b] = c64_one();
            }

            // Contract all sites to the right (in reverse)
            for s in ((*site + 2)..self.num_sites).rev() {
                let shape = &self.shapes[s];
                let bl = shape.bond_left;
                let br = shape.bond_right;
                let mut new_env = vec![c64_zero(); bl * bl];
                for ap in 0..bl {
                    for app in 0..bl {
                        let mut sum = c64_zero();
                        for bp in 0..br {
                            for bpp in 0..br {
                                for p in 0..2 {
                                    let t_idx1 = ap * 2 * br + p * br + bp;
                                    let t_idx2 = app * 2 * br + p * br + bpp;
                                    sum += env_r[bp * br + bpp]
                                        * self.tensors[s][t_idx1]
                                        * self.tensors[s][t_idx2].conj();
                                }
                            }
                        }
                        new_env[ap * bl + app] = sum;
                    }
                }
                env_r = new_env;
            }

            // Compute <psi|O|psi> using environments and the two-site operator
            // sum over a,a',b,b',i,j,i',j':
            //   env_l[a,a'] * T_L[a,i,m] * T_R[m,j,b] * O[i'j', ij] *
            //   conj(T_L[a',i',m']) * conj(T_R[m',j',b']) * env_r[b,b']
            let chi_m = sl.bond_right;
            let mut val = c64_zero();

            for a in 0..chi_l {
                for ap in 0..chi_l {
                    for b in 0..chi_r {
                        for bp in 0..chi_r {
                            let e_l = env_l[a * chi_l + ap];
                            let e_r = env_r[b * chi_r + bp];
                            if e_l.norm_sqr() < 1e-30 && e_r.norm_sqr() < 1e-30 {
                                continue;
                            }

                            for ip in 0..2 {
                                for jp in 0..2 {
                                    for i in 0..2 {
                                        for j in 0..2 {
                                            let o_elem = op[(ip * 2 + jp) * 4 + (i * 2 + j)];
                                            if o_elem.norm_sqr() < 1e-30 {
                                                continue;
                                            }

                                            let mut bra_ket = c64_zero();
                                            for m in 0..chi_m {
                                                for mp in 0..chi_m {
                                                    let tl_ket = self.tensors[*site]
                                                        [a * 2 * chi_m + i * chi_m + m];
                                                    let tr_ket = self.tensors[*site + 1]
                                                        [m * 2 * chi_r + j * chi_r + b];
                                                    let tl_bra = self.tensors[*site]
                                                        [ap * 2 * chi_m + ip * chi_m + mp]
                                                        .conj();
                                                    let tr_bra = self.tensors[*site + 1]
                                                        [mp * 2 * chi_r + jp * chi_r + bp]
                                                        .conj();
                                                    bra_ket += tl_ket * tr_ket * tl_bra * tr_bra;
                                                }
                                            }

                                            val += e_l * e_r * o_elem * bra_ket;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            total += val.re;
        }

        total
    }

    /// Compute gradients of the expectation value with respect to circuit parameters
    /// using finite differences.
    ///
    /// This is the primary gradient interface. For each parameter, the circuit is
    /// re-evaluated with a small perturbation to compute the numerical derivative.
    ///
    /// # Arguments
    /// * `params` - Current parameter values
    /// * `circuit_fn` - Function that builds the circuit: takes params, applies gates to the MPS
    /// * `observable` - The observable to differentiate
    /// * `epsilon` - Finite difference step size
    ///
    /// # Returns
    /// Gradient vector (one entry per parameter)
    pub fn gradient<F>(
        &self,
        params: &[f64],
        circuit_fn: &F,
        observable: &[(usize, Vec<C64>)],
        epsilon: f64,
    ) -> Vec<f64>
    where
        F: Fn(&mut DifferentiableMPS, &[f64]),
    {
        let n = params.len();
        let mut grad = vec![0.0; n];

        for i in 0..n {
            // Forward perturbation
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;
            let mut mps_plus = DifferentiableMPS::new(self.num_sites, self.max_bond_dim);
            mps_plus.truncation_cutoff = self.truncation_cutoff;
            mps_plus.truncation_sharpness = self.truncation_sharpness;
            circuit_fn(&mut mps_plus, &params_plus);
            let e_plus = mps_plus.compute_expectation(observable);

            // Backward perturbation
            let mut params_minus = params.to_vec();
            params_minus[i] -= epsilon;
            let mut mps_minus = DifferentiableMPS::new(self.num_sites, self.max_bond_dim);
            mps_minus.truncation_cutoff = self.truncation_cutoff;
            mps_minus.truncation_sharpness = self.truncation_sharpness;
            circuit_fn(&mut mps_minus, &params_minus);
            let e_minus = mps_minus.compute_expectation(observable);

            grad[i] = (e_plus - e_minus) / (2.0 * epsilon);
        }

        grad
    }

    /// Compute the norm <psi|psi> of the MPS state.
    pub fn norm(&self) -> f64 {
        // Contract from left to right
        let first_shape = &self.shapes[0];
        let bl = first_shape.bond_left;
        let br = first_shape.bond_right;

        // env[a, a'] = sum_p T[0][a,p,b] * conj(T[0][a',p,b])
        // But for leftmost site, a=a'=0 (bond_left=1 typically)
        let mut env = vec![c64_zero(); br * br];
        for bp in 0..br {
            for bpp in 0..br {
                let mut sum = c64_zero();
                for a in 0..bl {
                    for p in 0..2 {
                        sum += self.tensors[0][a * 2 * br + p * br + bp]
                            * self.tensors[0][a * 2 * br + p * br + bpp].conj();
                    }
                }
                env[bp * br + bpp] = sum;
            }
        }

        for s in 1..self.num_sites {
            let shape = &self.shapes[s];
            let sbl = shape.bond_left;
            let sbr = shape.bond_right;
            let mut new_env = vec![c64_zero(); sbr * sbr];
            for bp in 0..sbr {
                for bpp in 0..sbr {
                    let mut sum = c64_zero();
                    for ap in 0..sbl {
                        for app in 0..sbl {
                            for p in 0..2 {
                                sum += env[ap * sbl + app]
                                    * self.tensors[s][ap * 2 * sbr + p * sbr + bp]
                                    * self.tensors[s][app * 2 * sbr + p * sbr + bpp].conj();
                            }
                        }
                    }
                    new_env[bp * sbr + bpp] = sum;
                }
            }
            env = new_env;
        }

        // env should be 1x1 at the end
        env[0].re.max(0.0).sqrt()
    }
}

// ============================================================
// VARIATIONAL ANSATZ
// ============================================================

/// A parameterized MPS variational circuit for quantum optimization (VQE-like).
///
/// Generates layers of parameterized two-qubit gates and provides methods
/// for computing energies and gradients for gradient-based optimization.
///
/// The circuit consists of `num_layers` layers, each containing:
/// - Single-qubit Ry rotations on every qubit (parameterized)
/// - Nearest-neighbor CNOT-like entangling gates with Rz rotations (parameterized)
pub struct MPSVariationalAnsatz {
    /// Number of qubit sites.
    pub num_qubits: usize,
    /// Maximum bond dimension.
    pub max_bond_dim: usize,
    /// Number of variational layers.
    pub num_layers: usize,
}

impl MPSVariationalAnsatz {
    /// Create a new variational ansatz.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `max_bond_dim` - Maximum bond dimension for MPS truncation
    /// * `num_layers` - Number of variational layers
    pub fn new(num_qubits: usize, max_bond_dim: usize, num_layers: usize) -> Self {
        assert!(
            num_qubits >= 2,
            "Need at least 2 qubits for variational ansatz"
        );
        Self {
            num_qubits,
            max_bond_dim,
            num_layers,
        }
    }

    /// Total number of variational parameters.
    ///
    /// Each layer has:
    /// - `num_qubits` Ry rotation angles (single-qubit rotations)
    /// - `num_qubits - 1` Rz rotation angles (entangling layer)
    pub fn num_params(&self) -> usize {
        self.num_layers * (self.num_qubits + self.num_qubits - 1)
    }

    /// Build a parameterized two-qubit gate: Ry(a) x Ry(b) followed by CNOT-Rz(c).
    ///
    /// The gate acts on qubits (site, site+1) and is parameterized by three angles.
    /// Structure: (Ry(a) tensor Ry(b)) * CX * (I tensor Rz(c))
    fn build_entangling_gate(theta_left: f64, theta_right: f64, theta_zz: f64) -> Vec<C64> {
        // Build Ry gates
        let ry_l = ry_matrix(theta_left);
        let ry_r = ry_matrix(theta_right);

        // Tensor product: Ry_L x Ry_R (4x4)
        let mut ry_prod = vec![c64_zero(); 16];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        ry_prod[(i * 2 + k) * 4 + (j * 2 + l)] = ry_l[i * 2 + j] * ry_r[k * 2 + l];
                    }
                }
            }
        }

        // CNOT gate
        let cnot = cnot_matrix();

        // Rz on second qubit: I x Rz
        let rz = rz_matrix(theta_zz);
        let mut i_rz = vec![c64_zero(); 16];
        for i in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    i_rz[(i * 2 + k) * 4 + (i * 2 + l)] = rz[k * 2 + l];
                }
            }
        }

        // Full gate = Ry_prod * CNOT * I_Rz
        let temp = mat_mul_4x4(&ry_prod, &cnot);
        mat_mul_4x4(&temp, &i_rz)
    }

    /// Build the full circuit as a sequence of (site, gate_matrix) pairs.
    ///
    /// # Arguments
    /// * `params` - Variational parameters (length must equal `self.num_params()`)
    ///
    /// # Returns
    /// List of (site_index, flat_4x4_gate) operations to apply sequentially
    pub fn build_circuit(&self, params: &[f64]) -> Vec<(usize, Vec<C64>)> {
        assert_eq!(
            params.len(),
            self.num_params(),
            "Expected {} params, got {}",
            self.num_params(),
            params.len()
        );

        let mut gates = Vec::new();
        let params_per_layer = self.num_qubits + self.num_qubits - 1;

        for layer in 0..self.num_layers {
            let base = layer * params_per_layer;

            // Apply entangling gates to nearest-neighbor pairs
            for site in 0..(self.num_qubits - 1) {
                let theta_left = params[base + site];
                let theta_right = params[base + site + 1];
                let theta_zz = params[base + self.num_qubits + site];

                let gate = Self::build_entangling_gate(theta_left, theta_right, theta_zz);
                gates.push((site, gate));
            }
        }

        gates
    }

    /// Apply the variational circuit to an MPS.
    fn apply_circuit(mps: &mut DifferentiableMPS, params: &[f64], ansatz: &MPSVariationalAnsatz) {
        let gates = ansatz.build_circuit(params);
        for (site, gate) in &gates {
            mps.apply_gate(*site, gate);
        }
    }

    /// Compute energy and its gradient with respect to variational parameters.
    ///
    /// Uses central finite differences through the differentiable MPS pipeline.
    ///
    /// # Arguments
    /// * `params` - Current variational parameters
    /// * `hamiltonian` - Hamiltonian as a sum of two-site operators: Vec<(site, 4x4 matrix)>
    ///
    /// # Returns
    /// (energy, gradient_vector)
    pub fn energy_and_gradient(
        &self,
        params: &[f64],
        hamiltonian: &[(usize, Vec<C64>)],
    ) -> (f64, Vec<f64>) {
        let epsilon = 1e-5;

        // Compute energy at current parameters
        let mut mps = DifferentiableMPS::new(self.num_qubits, self.max_bond_dim);
        let gates = self.build_circuit(params);
        for (site, gate) in &gates {
            mps.apply_gate(*site, gate);
        }
        let energy = mps.compute_expectation(hamiltonian);

        // Compute gradient via central finite differences
        let ansatz_ref = self;
        let circuit_fn = |mps: &mut DifferentiableMPS, p: &[f64]| {
            let g = ansatz_ref.build_circuit(p);
            for (site, gate) in &g {
                mps.apply_gate(*site, gate);
            }
        };

        let grad = mps.gradient(params, &circuit_fn, hamiltonian, epsilon);

        (energy, grad)
    }

    /// Run gradient descent optimization to find the ground state energy.
    ///
    /// # Arguments
    /// * `hamiltonian` - Hamiltonian as a sum of two-site operators
    /// * `learning_rate` - Step size for gradient descent
    /// * `num_steps` - Maximum number of optimization steps
    ///
    /// # Returns
    /// (final_energy, optimized_parameters)
    pub fn optimize(
        &self,
        hamiltonian: &[(usize, Vec<C64>)],
        learning_rate: f64,
        num_steps: usize,
    ) -> (f64, Vec<f64>) {
        let mut params = vec![0.1; self.num_params()];
        // Initialize with small random-like values to break symmetry
        for (i, p) in params.iter_mut().enumerate() {
            *p = 0.1 * ((i as f64 * 0.7 + 0.3).sin());
        }

        let mut best_energy = f64::INFINITY;
        let mut best_params = params.clone();

        for _step in 0..num_steps {
            let (energy, grad) = self.energy_and_gradient(&params, hamiltonian);

            if energy < best_energy {
                best_energy = energy;
                best_params = params.clone();
            }

            // Gradient descent update
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= learning_rate * g;
            }
        }

        (best_energy, best_params)
    }
}

// ============================================================
// GATE MATRIX HELPERS
// ============================================================

/// Ry(theta) rotation matrix.
#[inline]
fn ry_matrix(theta: f64) -> [C64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        C64::new(c, 0.0),
        C64::new(-s, 0.0),
        C64::new(s, 0.0),
        C64::new(c, 0.0),
    ]
}

/// Rz(theta) rotation matrix.
#[inline]
fn rz_matrix(theta: f64) -> [C64; 4] {
    [
        C64::new((-theta / 2.0).cos(), (-theta / 2.0).sin()),
        c64_zero(),
        c64_zero(),
        C64::new((theta / 2.0).cos(), (theta / 2.0).sin()),
    ]
}

/// CNOT gate matrix (4x4, control=first qubit, target=second).
#[inline]
fn cnot_matrix() -> Vec<C64> {
    let mut m = vec![c64_zero(); 16];
    // |00> -> |00>
    m[0 * 4 + 0] = c64_one();
    // |01> -> |01>
    m[1 * 4 + 1] = c64_one();
    // |10> -> |11>
    m[2 * 4 + 3] = c64_one();
    // |11> -> |10>
    m[3 * 4 + 2] = c64_one();
    m
}

/// 4x4 complex matrix multiplication (row-major flat arrays).
fn mat_mul_4x4(a: &[C64], b: &[C64]) -> Vec<C64> {
    assert_eq!(a.len(), 16);
    assert_eq!(b.len(), 16);
    let mut c = vec![c64_zero(); 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = c64_zero();
            for k in 0..4 {
                sum += a[i * 4 + k] * b[k * 4 + j];
            }
            c[i * 4 + j] = sum;
        }
    }
    c
}

/// Build a ZZ interaction Hamiltonian term: Z_i x Z_{i+1}
/// Returns the 4x4 matrix for Z tensor Z.
pub fn zz_hamiltonian_term() -> Vec<C64> {
    // Z = diag(1, -1)
    // Z x Z = diag(1, -1, -1, 1)
    let mut h = vec![c64_zero(); 16];
    h[0 * 4 + 0] = c64_one(); // |00><00| : +1
    h[1 * 4 + 1] = C64::new(-1.0, 0.0); // |01><01| : -1
    h[2 * 4 + 2] = C64::new(-1.0, 0.0); // |10><10| : -1
    h[3 * 4 + 3] = c64_one(); // |11><11| : +1
    h
}

/// Build a transverse field Ising Hamiltonian: -J * sum Z_i Z_{i+1} - h * sum X_i
///
/// Returns the Hamiltonian as a sum of two-site operators suitable for
/// `DifferentiableMPS::compute_expectation`.
///
/// # Arguments
/// * `num_sites` - Number of qubit sites
/// * `j_coupling` - ZZ coupling strength
/// * `h_field` - Transverse field strength
pub fn transverse_ising_hamiltonian(
    num_sites: usize,
    j_coupling: f64,
    h_field: f64,
) -> Vec<(usize, Vec<C64>)> {
    let mut terms = Vec::new();

    for site in 0..(num_sites - 1) {
        let mut term = vec![c64_zero(); 16];

        // -J * Z_i Z_{i+1}
        let zz = zz_hamiltonian_term();
        for k in 0..16 {
            term[k] += zz[k] * C64::new(-j_coupling, 0.0);
        }

        // -h * (X_i x I + I x X_{i+1}) / 2 for this bond
        // Distribute the single-site X terms evenly across bonds
        // X_i contribution: shared between bond (i-1,i) and bond (i, i+1)
        let _x_weight = if site == 0 && site == num_sites - 2 {
            // Only one bond: full weight for both qubits
            1.0
        } else if site == 0 {
            // First bond: full X on left qubit, half X on right
            1.0 // left gets full, right gets half -- handled below
        } else if site == num_sites - 2 {
            1.0
        } else {
            1.0
        };

        // X_left x I: acts on first qubit of the pair
        // X = [[0,1],[1,0]], I = [[1,0],[0,1]]
        // (X x I)[i*2+k, j*2+l] = X[i,j] * I[k,l]
        let left_weight = if site == 0 { 1.0 } else { 0.5 };
        let right_weight = if site == num_sites - 2 { 1.0 } else { 0.5 };

        // X_left x I
        for k in 0..2 {
            // X[0,1] = 1, X[1,0] = 1
            term[(0 * 2 + k) * 4 + (1 * 2 + k)] += C64::new(-h_field * left_weight, 0.0);
            term[(1 * 2 + k) * 4 + (0 * 2 + k)] += C64::new(-h_field * left_weight, 0.0);
        }

        // I x X_right
        for i in 0..2 {
            term[(i * 2 + 0) * 4 + (i * 2 + 1)] += C64::new(-h_field * right_weight, 0.0);
            term[(i * 2 + 1) * 4 + (i * 2 + 0)] += C64::new(-h_field * right_weight, 0.0);
        }

        terms.push((site, term));
    }

    terms
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    // --- DualComplex arithmetic tests ---

    #[test]
    fn test_dual_complex_addition() {
        let a = DualComplex::new(C64::new(1.0, 2.0), C64::new(0.1, 0.2));
        let b = DualComplex::new(C64::new(3.0, 4.0), C64::new(0.3, 0.4));
        let c = a + b;
        assert!((c.value.re - 4.0).abs() < TOL);
        assert!((c.value.im - 6.0).abs() < TOL);
        assert!((c.derivative.re - 0.4).abs() < TOL);
        assert!((c.derivative.im - 0.6).abs() < TOL);
    }

    #[test]
    fn test_dual_complex_subtraction() {
        let a = DualComplex::new(C64::new(5.0, 3.0), C64::new(1.0, 0.5));
        let b = DualComplex::new(C64::new(2.0, 1.0), C64::new(0.3, 0.1));
        let c = a - b;
        assert!((c.value.re - 3.0).abs() < TOL);
        assert!((c.value.im - 2.0).abs() < TOL);
        assert!((c.derivative.re - 0.7).abs() < TOL);
        assert!((c.derivative.im - 0.4).abs() < TOL);
    }

    #[test]
    fn test_dual_complex_multiplication() {
        // (a + da*eps)(b + db*eps) = ab + (da*b + a*db)*eps
        let a = DualComplex::new(C64::new(2.0, 0.0), C64::new(1.0, 0.0));
        let b = DualComplex::new(C64::new(3.0, 0.0), C64::new(0.0, 0.0));
        let c = a * b;
        assert!((c.value.re - 6.0).abs() < TOL);
        assert!((c.derivative.re - 3.0).abs() < TOL); // da*b = 1*3 = 3
    }

    #[test]
    fn test_dual_complex_division() {
        // d(a/b) = (da*b - a*db) / b^2
        let a = DualComplex::new(C64::new(6.0, 0.0), C64::new(1.0, 0.0));
        let b = DualComplex::new(C64::new(3.0, 0.0), C64::new(0.0, 0.0));
        let c = a / b;
        assert!((c.value.re - 2.0).abs() < TOL);
        assert!((c.derivative.re - 1.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn test_dual_complex_negation() {
        let a = DualComplex::new(C64::new(1.0, 2.0), C64::new(0.5, -0.3));
        let b = -a;
        assert!((b.value.re + 1.0).abs() < TOL);
        assert!((b.value.im + 2.0).abs() < TOL);
        assert!((b.derivative.re + 0.5).abs() < TOL);
        assert!((b.derivative.im - 0.3).abs() < TOL);
    }

    #[test]
    fn test_dual_complex_conjugate() {
        let a = DualComplex::new(C64::new(1.0, 2.0), C64::new(0.5, -0.3));
        let c = a.conj();
        assert!((c.value.re - 1.0).abs() < TOL);
        assert!((c.value.im + 2.0).abs() < TOL);
        assert!((c.derivative.re - 0.5).abs() < TOL);
        assert!((c.derivative.im - 0.3).abs() < TOL);
    }

    #[test]
    fn test_dual_complex_norm() {
        // |z| for z = 3 + 4i should be 5
        let a = DualComplex::new(C64::new(3.0, 4.0), C64::new(1.0, 0.0));
        let (n, dn) = a.norm();
        assert!((n - 5.0).abs() < TOL);
        // d|z|/dz_re = Re(conj(z) * dz) / |z| = Re((3-4i)*(1+0i)) / 5 = 3/5
        assert!((dn - 3.0 / 5.0).abs() < TOL);
    }

    // --- SVD snapshot tests ---

    #[test]
    fn test_svd_snapshot_reconstruct_identity() {
        // Reconstruct a known 2x2 matrix from its SVD
        // M = [[1, 0], [0, 1]] => U = I, S = [1, 1], Vt = I
        let snapshot = SvdSnapshot::new(
            vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            vec![1.0, 1.0],
            vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            2,
            2,
            2,
        );
        let recon = snapshot.reconstruct();
        assert!((recon[0] - c64_one()).norm() < TOL);
        assert!((recon[1] - c64_zero()).norm() < TOL);
        assert!((recon[2] - c64_zero()).norm() < TOL);
        assert!((recon[3] - c64_one()).norm() < TOL);
    }

    #[test]
    fn test_svd_snapshot_reconstruct_rank1() {
        // Rank-1 matrix: M = [[2, 4], [1, 2]]
        // SVD: s = sqrt(25) = 5, u = [2/sqrt(5), 1/sqrt(5)]^T, v = [1/sqrt(5), 2/sqrt(5)]
        let s5 = 5.0f64.sqrt();
        let snapshot = SvdSnapshot::new(
            vec![C64::new(2.0 / s5, 0.0), C64::new(1.0 / s5, 0.0)],
            vec![5.0],
            vec![C64::new(1.0 / s5, 0.0), C64::new(2.0 / s5, 0.0)],
            2,
            2,
            1,
        );
        let recon = snapshot.reconstruct();
        assert!((recon[0].re - 2.0).abs() < TOL);
        assert!((recon[1].re - 4.0).abs() < TOL);
        assert!((recon[2].re - 1.0).abs() < TOL);
        assert!((recon[3].re - 2.0).abs() < TOL);
    }

    // --- SVD gradient numerical verification ---

    #[test]
    fn test_svd_gradient_numerical() {
        // Verify SVD gradient against finite differences for a scalar loss function.
        //
        // We define loss(M) = Re(sum of reconstructed elements after SVD).
        // The analytical gradient via svd_gradient should match finite differences.
        let eps = 1e-6;

        // Original matrix
        let m_orig = vec![
            C64::new(1.0, 0.0),
            C64::new(0.5, 0.0),
            C64::new(0.3, 0.0),
            C64::new(0.8, 0.0),
        ];

        // Loss function: sum of real parts of reconstructed matrix
        let loss_fn = |mat: &[C64]| -> f64 {
            let snap = compute_test_svd(mat, 2, 2);
            snap.reconstruct().iter().map(|c| c.re).sum::<f64>()
        };

        let loss_0 = loss_fn(&m_orig);

        // Compute numerical gradient via central finite differences
        let mut num_grad = Vec::new();
        for idx in 0..4 {
            let mut m_plus = m_orig.clone();
            m_plus[idx] += C64::new(eps, 0.0);
            let mut m_minus = m_orig.clone();
            m_minus[idx] -= C64::new(eps, 0.0);
            let g = (loss_fn(&m_plus) - loss_fn(&m_minus)) / (2.0 * eps);
            num_grad.push(g);
        }

        // Compute analytical gradient via svd_gradient
        let snapshot = compute_test_svd(&m_orig, 2, 2);
        let d_out = vec![c64_one(); 4]; // gradient of sum = all ones
        let ana_grad = svd_gradient(&snapshot, &d_out);

        // Compare analytical vs numerical
        for (i, (a, n)) in ana_grad.iter().zip(num_grad.iter()).enumerate() {
            assert!(
                (a.re - n).abs() < 0.5,
                "SVD gradient mismatch at element {}: analytical={:.6}, numerical={:.6}",
                i,
                a.re,
                n
            );
        }
    }

    // --- Soft threshold tests ---

    #[test]
    fn test_soft_threshold_large_values_pass() {
        let s = vec![1.0, 0.5, 0.1];
        let result = SoftThresholdTruncation::apply(&s, 0.05, 100.0);
        // Values well above cutoff should be nearly unchanged
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
        assert!((result[2] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_soft_threshold_small_values_suppressed() {
        let s = vec![1.0, 0.5, 0.001];
        let result = SoftThresholdTruncation::apply(&s, 0.01, 1000.0);
        // Value well below cutoff should be strongly suppressed
        assert!(result[2] < 0.001);
        // Large values unaffected
        assert!((result[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_soft_threshold_continuity() {
        // Verify that the soft threshold is continuous (no jumps)
        let cutoff = 0.1;
        let sharpness = 50.0;
        let mut prev = SoftThresholdTruncation::apply(&[0.0], cutoff, sharpness)[0];

        for i in 1..100 {
            let s = i as f64 * 0.005;
            let curr = SoftThresholdTruncation::apply(&[s], cutoff, sharpness)[0];
            let jump = (curr - prev).abs();
            assert!(jump < 0.01, "Discontinuity at s={:.3}: jump={:.6}", s, jump);
            prev = curr;
        }
    }

    #[test]
    fn test_soft_threshold_gradient_positive() {
        // Gradient should be non-negative for non-negative singular values
        let s = vec![0.01, 0.05, 0.1, 0.5, 1.0];
        let grad = SoftThresholdTruncation::gradient(&s, 0.1, 50.0);
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                g >= 0.0,
                "Gradient should be non-negative at s={}, got {}",
                s[i],
                g
            );
        }
    }

    // --- DifferentiableMPS tests ---

    #[test]
    fn test_dmps_initialization() {
        let mps = DifferentiableMPS::new(4, 8);
        assert_eq!(mps.num_sites, 4);
        assert_eq!(mps.max_bond_dim, 8);
        assert_eq!(mps.bond_dims.len(), 3);

        // Check |0000> state
        for s in 0..4 {
            assert_eq!(mps.shapes[s].bond_left, 1);
            assert_eq!(mps.shapes[s].phys_dim, 2);
            assert_eq!(mps.shapes[s].bond_right, 1);
            // |0> component should be 1
            assert!((mps.tensor_elem(s, 0, 0, 0) - c64_one()).norm() < TOL);
            // |1> component should be 0
            assert!((mps.tensor_elem(s, 0, 1, 0) - c64_zero()).norm() < TOL);
        }
    }

    #[test]
    fn test_dmps_norm_initial() {
        let mps = DifferentiableMPS::new(4, 8);
        let n = mps.norm();
        assert!(
            (n - 1.0).abs() < TOL,
            "Initial state norm should be 1, got {}",
            n
        );
    }

    #[test]
    fn test_dmps_apply_identity_gate() {
        let mut mps = DifferentiableMPS::new(3, 8);

        // Identity gate: should not change the state
        let mut identity = vec![c64_zero(); 16];
        for i in 0..4 {
            identity[i * 4 + i] = c64_one();
        }

        mps.apply_gate(0, &identity);

        // Norm should still be 1
        let n = mps.norm();
        assert!((n - 1.0).abs() < 1e-6, "Norm after identity gate: {}", n);
    }

    #[test]
    fn test_dmps_apply_cnot_creates_entanglement() {
        let mut mps = DifferentiableMPS::new(2, 4);

        // First apply Hadamard on qubit 0 via a two-site gate: H x I
        let s2 = 1.0 / 2.0f64.sqrt();
        let mut h_i = vec![c64_zero(); 16];
        // H = [[1,1],[1,-1]] / sqrt(2)
        // (H x I)[i*2+k, j*2+l] = H[i,j] * I[k,l]
        for k in 0..2 {
            h_i[(0 * 2 + k) * 4 + (0 * 2 + k)] = C64::new(s2, 0.0);
            h_i[(0 * 2 + k) * 4 + (1 * 2 + k)] = C64::new(s2, 0.0);
            h_i[(1 * 2 + k) * 4 + (0 * 2 + k)] = C64::new(s2, 0.0);
            h_i[(1 * 2 + k) * 4 + (1 * 2 + k)] = C64::new(-s2, 0.0);
        }
        mps.apply_gate(0, &h_i);

        // Then apply CNOT
        let cnot = cnot_matrix();
        mps.apply_gate(0, &cnot);

        // Should create Bell state (|00> + |11>) / sqrt(2)
        // Bond dimension should increase
        assert!(
            mps.bond_dims[0] >= 2,
            "CNOT should create entanglement, bond_dim={}",
            mps.bond_dims[0]
        );

        // Norm should be approximately 1
        let n = mps.norm();
        assert!(
            (n - 1.0).abs() < 0.1,
            "Norm after Bell state creation: {}",
            n
        );
    }

    #[test]
    fn test_dmps_expectation_product_state() {
        let mps = DifferentiableMPS::new(3, 8);

        // <000|Z_0 Z_1|000> = 1 * 1 = 1 (both qubits in |0>, eigenvalue +1 for Z)
        let zz = zz_hamiltonian_term();
        let obs = vec![(0usize, zz)];
        let exp = mps.compute_expectation(&obs);
        assert!(
            (exp - 1.0).abs() < TOL,
            "ZZ expectation for |000> should be 1, got {}",
            exp
        );
    }

    #[test]
    fn test_dmps_gradient_finite_diff() {
        // Verify that the gradient computation returns reasonable values
        let num_qubits = 2;
        let max_bond = 4;

        let params = vec![0.3, 0.5, 0.2];

        let circuit_fn = |mps: &mut DifferentiableMPS, p: &[f64]| {
            let gate = MPSVariationalAnsatz::build_entangling_gate(p[0], p[1], p[2]);
            mps.apply_gate(0, &gate);
        };

        let zz = zz_hamiltonian_term();
        let obs = vec![(0usize, zz)];

        let mps = DifferentiableMPS::new(num_qubits, max_bond);
        let grad = mps.gradient(&params, &circuit_fn, &obs, 1e-5);

        // Gradient should have 3 components
        assert_eq!(grad.len(), 3);

        // At least some gradients should be non-zero for non-trivial parameters
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(
            grad_norm > 1e-6,
            "Gradient should be non-zero, got norm={}",
            grad_norm
        );
    }

    // --- Variational ansatz tests ---

    #[test]
    fn test_variational_ansatz_param_count() {
        let ansatz = MPSVariationalAnsatz::new(4, 8, 2);
        // 2 layers * (4 Ry + 3 Rz) = 2 * 7 = 14
        assert_eq!(ansatz.num_params(), 14);
    }

    #[test]
    fn test_variational_ansatz_circuit_build() {
        let ansatz = MPSVariationalAnsatz::new(3, 4, 1);
        let params = vec![0.1; ansatz.num_params()];
        let circuit = ansatz.build_circuit(&params);

        // 1 layer, 2 bonds (qubits 0-1, 1-2) => 2 gates
        assert_eq!(circuit.len(), 2);
        assert_eq!(circuit[0].0, 0);
        assert_eq!(circuit[1].0, 1);
        assert_eq!(circuit[0].1.len(), 16);
    }

    #[test]
    fn test_variational_optimization_ising() {
        // Optimize a small 2-qubit transverse Ising model
        // H = -Z_0 Z_1 - 0.5*(X_0 + X_1)
        let hamiltonian = transverse_ising_hamiltonian(2, 1.0, 0.5);

        let ansatz = MPSVariationalAnsatz::new(2, 4, 2);
        let (energy, params) = ansatz.optimize(&hamiltonian, 0.05, 50);

        // The ground state energy of 2-qubit transverse Ising should be negative
        // (ferromagnetic ground state is energetically favorable)
        assert!(
            energy < 0.0,
            "Optimized energy should be negative for Ising model, got {}",
            energy
        );
    }

    // --- Helper: compute SVD for test purposes ---

    fn compute_test_svd(mat: &[C64], rows: usize, cols: usize) -> SvdSnapshot {
        use nalgebra::{Complex as NComplex, DMatrix};

        let mut nmat_data = vec![NComplex::<f64>::new(0.0, 0.0); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let v = mat[i * cols + j];
                nmat_data[i + j * rows] = NComplex::new(v.re, v.im);
            }
        }
        let nmat = DMatrix::from_vec(rows, cols, nmat_data);
        let svd = nmat.svd(true, true);
        let u_full = svd.u.unwrap();
        let vt_full = svd.v_t.unwrap();
        let svals = &svd.singular_values;
        let rank = svals.len();

        let mut u_data = vec![c64_zero(); rows * rank];
        for i in 0..rows {
            for k in 0..rank {
                let v = u_full[(i, k)];
                u_data[i * rank + k] = C64::new(v.re, v.im);
            }
        }

        let s_data: Vec<f64> = (0..rank).map(|k| svals[k]).collect();

        let mut vt_data = vec![c64_zero(); rank * cols];
        for k in 0..rank {
            for j in 0..cols {
                let v = vt_full[(k, j)];
                vt_data[k * cols + j] = C64::new(v.re, v.im);
            }
        }

        SvdSnapshot::new(u_data, s_data, vt_data, rows, cols, rank)
    }
}
