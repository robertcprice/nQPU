//! Tucker State Preparation: Iterative Quantum State Synthesis
//!
//! Based on arXiv:2602.09909 - "Tucker Iterative Quantum State Preparation"
//!
//! # Overview
//!
//! Tucker decomposition factorizes high-dimensional tensors into a core tensor
//! and factor matrices. Applied to quantum states, this enables efficient
//! state preparation with:
//!
//! - Lower circuit depth than direct preparation
//! - Adaptive refinement based on approximation error
//! - Natural compression of multi-qubit states
//!
//! # Mathematical Background
//!
//! A quantum state |psi> can be viewed as a tensor psi_{i1,i2,...,in}.
//! Tucker decomposition factorizes it as:
//!
//! ```text
//! psi_{i1,...,in} = sum_{r1,...,rn} G_{r1,...,rn} * U^{(1)}_{i1,r1} * ... * U^{(n)}_{in,rn}
//! ```
//!
//! Where:
//! - G is the core tensor (compressed representation)
//! - U^{(k)} are factor matrices (unitary transformations per qubit mode)
//! - (r1,...,rn) are the Tucker ranks
//!
//! # Algorithm
//!
//! 1. **HOSVD** (Higher-Order SVD) computes an initial decomposition by
//!    unfolding the tensor along each mode, computing an SVD, and extracting
//!    the leading left singular vectors as factor matrices.
//!
//! 2. **HOOI** (Higher-Order Orthogonal Iteration) refines the decomposition
//!    iteratively by alternating between updating factor matrices and recomputing
//!    the core tensor, converging to a locally optimal Tucker approximation.

use crate::C64;
use nalgebra::{Complex as NComplex, DMatrix};

/// Configuration for Tucker state preparation.
#[derive(Clone, Debug)]
pub struct TuckerConfig {
    /// Maximum Tucker rank per dimension.
    pub max_rank: usize,
    /// Convergence threshold for HOOI algorithm.
    pub convergence_threshold: f64,
    /// Maximum iterations for Tucker decomposition.
    pub max_iterations: usize,
    /// Use randomized SVD for efficiency.
    pub randomized_svd: bool,
    /// Truncation threshold for small singular values.
    pub truncation_threshold: f64,
    /// Use iterative refinement (HOOI after initial HOSVD).
    pub iterative_refinement: bool,
}

impl Default for TuckerConfig {
    fn default() -> Self {
        Self {
            max_rank: 4,
            convergence_threshold: 1e-6,
            max_iterations: 100,
            randomized_svd: true,
            truncation_threshold: 1e-10,
            iterative_refinement: true,
        }
    }
}

/// Tucker decomposition result.
#[derive(Clone, Debug)]
pub struct TuckerDecomposition {
    /// Core tensor stored in row-major order with shape `core_shape`.
    pub core: Vec<C64>,
    /// Core tensor shape (one entry per mode, each entry is the rank for that mode).
    pub core_shape: Vec<usize>,
    /// Factor matrices (one per mode). Factor k has shape (dim_k x rank_k) stored
    /// in row-major order, where dim_k = 2 for qubit modes.
    pub factors: Vec<Vec<C64>>,
    /// Tucker ranks (one per mode).
    pub ranks: Vec<usize>,
    /// Approximation error: ||original - reconstructed||_2.
    pub error: f64,
    /// Fidelity: |<original|reconstructed>|^2 (for normalized states).
    pub fidelity: f64,
}

impl TuckerDecomposition {
    /// Reconstruct the full tensor from the Tucker decomposition.
    ///
    /// Computes: T_{i1,...,in} = sum_{r1,...,rn} G_{r1,...,rn} * U1_{i1,r1} * ... * Un_{in,rn}
    ///
    /// Returns a flat vector of length dim_1 * dim_2 * ... * dim_n in row-major order.
    pub fn reconstruct(&self, dims: &[usize]) -> Vec<C64> {
        let n = dims.len();
        let total: usize = dims.iter().product();
        let mut result = vec![C64::new(0.0, 0.0); total];

        // Iterate over all core tensor entries
        let core_total: usize = self.core_shape.iter().product();
        for core_flat in 0..core_total {
            let g_val = self.core[core_flat];
            if g_val.re * g_val.re + g_val.im * g_val.im < 1e-30 {
                continue;
            }

            // Decode core multi-index (r_0, r_1, ..., r_{n-1}) from flat index
            let mut core_indices = vec![0usize; n];
            let mut rem = core_flat;
            for k in (0..n).rev() {
                core_indices[k] = rem % self.core_shape[k];
                rem /= self.core_shape[k];
            }

            // For each output element, accumulate contribution from this core entry
            for out_flat in 0..total {
                // Decode output multi-index (i_0, i_1, ..., i_{n-1})
                let mut out_indices = vec![0usize; n];
                let mut rem = out_flat;
                for k in (0..n).rev() {
                    out_indices[k] = rem % dims[k];
                    rem /= dims[k];
                }

                // Compute product of factor matrix entries: U0[i0,r0] * U1[i1,r1] * ...
                let mut product = g_val;
                for k in 0..n {
                    let i_k = out_indices[k];
                    let r_k = core_indices[k];
                    // Factor k stored row-major: factors[k][i_k * ranks[k] + r_k]
                    let u_val = self.factors[k][i_k * self.ranks[k] + r_k];
                    product = C64::new(
                        product.re * u_val.re - product.im * u_val.im,
                        product.re * u_val.im + product.im * u_val.re,
                    );
                }

                result[out_flat] += product;
            }
        }

        result
    }
}

/// Circuit operation description.
#[derive(Clone, Debug)]
pub enum CircuitOp {
    /// Single-qubit X gate.
    X { qubit: usize },
    /// Rotation around Y axis.
    RY { qubit: usize, angle: f64 },
    /// Rotation around Z axis.
    RZ { qubit: usize, angle: f64 },
    /// Hadamard gate.
    H { qubit: usize },
}

/// Tucker State Preparation engine.
#[derive(Clone, Debug)]
pub struct TuckerPrep {
    /// Number of qubits.
    n_qubits: usize,
    /// Configuration.
    config: TuckerConfig,
    /// Last decomposition result.
    last_decomp: Option<TuckerDecomposition>,
}

impl TuckerPrep {
    /// Create a new Tucker state preparer.
    pub fn new(n_qubits: usize, config: TuckerConfig) -> Self {
        Self {
            n_qubits,
            config,
            last_decomp: None,
        }
    }

    /// Prepare a quantum circuit to generate the target state.
    pub fn prepare(&mut self, target: &[C64]) -> Result<Vec<CircuitOp>, TuckerError> {
        let expected_len = 1 << self.n_qubits;
        if target.len() != expected_len {
            return Err(TuckerError::InvalidStateLength {
                expected: expected_len,
                got: target.len(),
            });
        }

        // Normalize the state using L2 norm
        let norm_sq: f64 = target.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        let norm = norm_sq.sqrt();
        let normalized: Vec<C64> = if norm > 1e-15 {
            target
                .iter()
                .map(|c| C64::new(c.re / norm, c.im / norm))
                .collect()
        } else {
            return Err(TuckerError::NumericalError {
                message: "Target state has near-zero norm".to_string(),
            });
        };

        // Compute Tucker decomposition on the normalized state
        let decomp = self.compute_tucker(&normalized)?;
        let circuit = self.decomposition_to_circuit(&decomp)?;

        self.last_decomp = Some(decomp);
        Ok(circuit)
    }

    /// Compute the Tucker decomposition of a quantum state vector.
    ///
    /// The state is treated as a tensor of shape [2, 2, ..., 2] with n_qubits modes.
    /// Uses HOSVD (Higher-Order SVD) for initialization, followed by HOOI
    /// (Higher-Order Orthogonal Iteration) for refinement when configured.
    fn compute_tucker(&self, state: &[C64]) -> Result<TuckerDecomposition, TuckerError> {
        let n = self.n_qubits;
        let dims: Vec<usize> = vec![2; n];
        let total: usize = dims.iter().product();

        if state.len() != total {
            return Err(TuckerError::InvalidDimensions {
                message: format!(
                    "State length {} does not match tensor size {}",
                    state.len(),
                    total
                ),
            });
        }

        // Determine Tucker ranks: min(max_rank, dim_k) for each mode
        let ranks: Vec<usize> = dims.iter().map(|&d| self.config.max_rank.min(d)).collect();

        // Phase 1: HOSVD -- compute initial factor matrices via mode-k unfoldings
        let mut factors: Vec<Vec<C64>> = Vec::with_capacity(n);
        for mode in 0..n {
            let factor = self.compute_mode_factor(state, &dims, mode, ranks[mode])?;
            factors.push(factor);
        }

        // Phase 2: Compute core tensor G = state x_1 U1^H x_2 U2^H x ... x_n Un^H
        let mut core = self.compute_core_tensor(state, &dims, &factors, &ranks)?;
        let mut core_shape = ranks.clone();

        // Phase 3: HOOI refinement (if configured)
        if self.config.iterative_refinement && n > 1 {
            let mut prev_norm = core_frobenius_norm(&core);

            for _iter in 0..self.config.max_iterations {
                // Update each factor matrix while holding others fixed
                for mode in 0..n {
                    // Compute Y_k = state x_1 U1^H ... x_{k-1} U_{k-1}^H x_{k+1} U_{k+1}^H ... x_n Un^H
                    // (contract with all factor conjugate-transposes EXCEPT mode k)
                    let y_k = self.partial_core_tensor(state, &dims, &factors, &ranks, mode)?;

                    // Mode-k unfolding of Y_k, then SVD to get updated U_k
                    let y_dims = self.partial_core_dims(&dims, &ranks, mode);
                    let updated_factor =
                        self.svd_of_mode_unfolding(&y_k, &y_dims, mode, ranks[mode])?;
                    factors[mode] = updated_factor;
                }

                // Recompute core
                core = self.compute_core_tensor(state, &dims, &factors, &ranks)?;
                core_shape = ranks.clone();

                let new_norm = core_frobenius_norm(&core);
                let change = (new_norm - prev_norm).abs() / (prev_norm.max(1e-15));
                if change < self.config.convergence_threshold {
                    break;
                }
                prev_norm = new_norm;
            }
        }

        // Compute approximation error and fidelity
        let reconstructed = TuckerDecomposition {
            core: core.clone(),
            core_shape: core_shape.clone(),
            factors: factors.clone(),
            ranks: ranks.clone(),
            error: 0.0,
            fidelity: 0.0,
        }
        .reconstruct(&dims);

        let (error, fidelity) = compute_error_and_fidelity(state, &reconstructed);

        Ok(TuckerDecomposition {
            core,
            core_shape,
            factors,
            ranks,
            error,
            fidelity,
        })
    }

    /// Compute the factor matrix for a given mode via truncated SVD of the mode-k unfolding.
    ///
    /// Mode-k unfolding reshapes the tensor into a matrix where:
    /// - Rows correspond to index i_k (size = dims[mode])
    /// - Columns correspond to all other indices combined
    ///
    /// The leading `rank` left singular vectors form the factor matrix.
    fn compute_mode_factor(
        &self,
        tensor: &[C64],
        dims: &[usize],
        mode: usize,
        rank: usize,
    ) -> Result<Vec<C64>, TuckerError> {
        self.svd_of_mode_unfolding(tensor, dims, mode, rank)
    }

    /// Perform SVD on the mode-k unfolding of a tensor and return the leading
    /// left singular vectors as a factor matrix of shape (dims[mode] x rank) in row-major.
    fn svd_of_mode_unfolding(
        &self,
        tensor: &[C64],
        dims: &[usize],
        mode: usize,
        rank: usize,
    ) -> Result<Vec<C64>, TuckerError> {
        let n = dims.len();
        let rows = dims[mode];
        let cols: usize = dims
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != mode)
            .map(|(_, &d)| d)
            .product::<usize>()
            .max(1);

        // Build the mode-k unfolding matrix
        // For a tensor T with multi-index (i_0, ..., i_{n-1}):
        // Row = i_mode
        // Col = linearized index of all other dimensions
        let total: usize = dims.iter().product();
        let mut mat_data = vec![NComplex::<f64>::new(0.0, 0.0); rows * cols];

        for flat_idx in 0..total {
            // Decode multi-index from flat (row-major) index
            let mut indices = vec![0usize; n];
            let mut rem = flat_idx;
            for k in (0..n).rev() {
                indices[k] = rem % dims[k];
                rem /= dims[k];
            }

            let row = indices[mode];

            // Compute column: linearize all non-mode indices
            let mut col = 0;
            let mut stride = 1;
            for k in (0..n).rev() {
                if k != mode {
                    col += indices[k] * stride;
                    stride *= dims[k];
                }
            }

            let v = tensor[flat_idx];
            // nalgebra stores column-major: index = row + col * rows
            mat_data[row + col * rows] = NComplex::new(v.re, v.im);
        }

        let mat = DMatrix::from_vec(rows, cols, mat_data);
        let svd = mat.svd(true, true);
        let u_full = svd.u.ok_or_else(|| TuckerError::NumericalError {
            message: format!("SVD failed to compute U matrix for mode {}", mode),
        })?;

        // Extract leading `rank` columns of U as the factor matrix
        let actual_rank = rank.min(u_full.ncols()).min(rows);
        let mut factor = vec![C64::new(0.0, 0.0); rows * actual_rank];
        for i in 0..rows {
            for r in 0..actual_rank {
                let u_val = u_full[(i, r)];
                // Store row-major: factor[i * actual_rank + r]
                factor[i * actual_rank + r] = C64::new(u_val.re, u_val.im);
            }
        }

        Ok(factor)
    }

    /// Compute the core tensor by contracting the original tensor with the
    /// conjugate-transpose of all factor matrices:
    ///
    /// G_{r1,...,rn} = sum_{i1,...,in} T_{i1,...,in} * conj(U1_{i1,r1}) * ... * conj(Un_{in,rn})
    fn compute_core_tensor(
        &self,
        tensor: &[C64],
        dims: &[usize],
        factors: &[Vec<C64>],
        ranks: &[usize],
    ) -> Result<Vec<C64>, TuckerError> {
        let n = dims.len();
        let total_input: usize = dims.iter().product();
        let core_total: usize = ranks.iter().product();
        let mut core = vec![C64::new(0.0, 0.0); core_total];

        // For each element of the original tensor, distribute its contribution
        // to all relevant core entries.
        for in_flat in 0..total_input {
            let t_val = tensor[in_flat];
            if t_val.re * t_val.re + t_val.im * t_val.im < 1e-30 {
                continue;
            }

            // Decode input multi-index
            let mut in_indices = vec![0usize; n];
            let mut rem = in_flat;
            for k in (0..n).rev() {
                in_indices[k] = rem % dims[k];
                rem /= dims[k];
            }

            // For each core entry, accumulate T_val * prod_k conj(U_k[i_k, r_k])
            for core_flat in 0..core_total {
                let mut core_indices = vec![0usize; n];
                let mut rem = core_flat;
                for k in (0..n).rev() {
                    core_indices[k] = rem % ranks[k];
                    rem /= ranks[k];
                }

                let mut product = t_val;
                for k in 0..n {
                    let i_k = in_indices[k];
                    let r_k = core_indices[k];
                    let u_val = factors[k][i_k * ranks[k] + r_k];
                    // Multiply by conj(u_val)
                    let conj_u = C64::new(u_val.re, -u_val.im);
                    product = C64::new(
                        product.re * conj_u.re - product.im * conj_u.im,
                        product.re * conj_u.im + product.im * conj_u.re,
                    );
                }

                core[core_flat] += product;
            }
        }

        Ok(core)
    }

    /// Compute the partial core tensor for HOOI: contract the original tensor with
    /// conjugate-transposes of all factor matrices EXCEPT the one at `skip_mode`.
    ///
    /// Result is a tensor with:
    /// - dim `dims[skip_mode]` along the skip_mode axis
    /// - dim `ranks[k]` along all other axes k
    fn partial_core_tensor(
        &self,
        tensor: &[C64],
        dims: &[usize],
        factors: &[Vec<C64>],
        ranks: &[usize],
        skip_mode: usize,
    ) -> Result<Vec<C64>, TuckerError> {
        let n = dims.len();

        // Output shape: for mode k, size is dims[k] if k == skip_mode, else ranks[k]
        let out_dims: Vec<usize> = (0..n)
            .map(|k| if k == skip_mode { dims[k] } else { ranks[k] })
            .collect();
        let out_total: usize = out_dims.iter().product();
        let in_total: usize = dims.iter().product();

        let mut result = vec![C64::new(0.0, 0.0); out_total];

        for in_flat in 0..in_total {
            let t_val = tensor[in_flat];
            if t_val.re * t_val.re + t_val.im * t_val.im < 1e-30 {
                continue;
            }

            // Decode input multi-index
            let mut in_indices = vec![0usize; n];
            let mut rem = in_flat;
            for k in (0..n).rev() {
                in_indices[k] = rem % dims[k];
                rem /= dims[k];
            }

            // Iterate over output entries
            for out_flat in 0..out_total {
                let mut out_indices = vec![0usize; n];
                let mut rem = out_flat;
                for k in (0..n).rev() {
                    out_indices[k] = rem % out_dims[k];
                    rem /= out_dims[k];
                }

                // The skip_mode output index must match the input index
                if out_indices[skip_mode] != in_indices[skip_mode] {
                    continue;
                }

                let mut product = t_val;
                for k in 0..n {
                    if k == skip_mode {
                        continue;
                    }
                    let i_k = in_indices[k];
                    let r_k = out_indices[k];
                    let u_val = factors[k][i_k * ranks[k] + r_k];
                    let conj_u = C64::new(u_val.re, -u_val.im);
                    product = C64::new(
                        product.re * conj_u.re - product.im * conj_u.im,
                        product.re * conj_u.im + product.im * conj_u.re,
                    );
                }

                result[out_flat] += product;
            }
        }

        Ok(result)
    }

    /// Compute the shape of the partial core tensor (for HOOI).
    /// Mode `skip_mode` retains its original dimension; all others use their ranks.
    fn partial_core_dims(&self, dims: &[usize], ranks: &[usize], skip_mode: usize) -> Vec<usize> {
        (0..dims.len())
            .map(|k| if k == skip_mode { dims[k] } else { ranks[k] })
            .collect()
    }

    fn decomposition_to_circuit(
        &self,
        decomp: &TuckerDecomposition,
    ) -> Result<Vec<CircuitOp>, TuckerError> {
        let mut ops = Vec::new();

        // Prepare core tensor state
        let core = &decomp.core;
        let shape = &decomp.core_shape;
        let n_core_qubits = shape.len();

        // Find largest amplitude in core tensor
        let (max_idx, _) = core
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let na = a.re * a.re + a.im * a.im;
                let nb = b.re * b.re + b.im * b.im;
                na.partial_cmp(&nb).unwrap()
            })
            .unwrap_or((0, &C64::new(0.0, 0.0)));

        // Prepare state with X gates for 1 bits of dominant basis state
        for i in 0..n_core_qubits {
            if (max_idx >> i) & 1 == 1 {
                ops.push(CircuitOp::X { qubit: i });
            }
        }

        // Apply amplitude encoding rotations from core tensor
        for (idx, &amp) in core.iter().enumerate() {
            if idx == 0 {
                continue;
            }

            let prob = amp.re * amp.re + amp.im * amp.im;
            if prob > self.config.truncation_threshold {
                let theta = 2.0 * prob.sqrt().acos();
                let diff = idx ^ (idx - 1);
                let target_qubit = diff.trailing_zeros() as usize;
                if target_qubit < n_core_qubits {
                    ops.push(CircuitOp::RY {
                        qubit: target_qubit,
                        angle: theta,
                    });
                }
            }
        }

        // Apply factor matrix rotations (single-qubit unitaries per mode)
        for (mode, factor) in decomp.factors.iter().enumerate() {
            let rank = decomp.ranks[mode];
            if rank == 0 {
                continue;
            }
            // Extract the 2x(rank) factor matrix and decompose the leading 2x2 block
            // as a single-qubit rotation
            if rank >= 1 && factor.len() >= 2 {
                // The factor matrix columns are orthonormal vectors in C^2.
                // The first column defines a unitary rotation from |0> to that vector.
                let u00 = factor[0]; // factor[0 * rank + 0]
                let u10 = if factor.len() > rank {
                    factor[rank]
                } else {
                    C64::new(0.0, 0.0)
                }; // factor[1 * rank + 0]

                // For a 2x2 unitary column [u00, u10], the RY angle is:
                //   theta = 2 * atan2(|u10|, |u00|)
                let mag0 = (u00.re * u00.re + u00.im * u00.im).sqrt();
                let mag1 = (u10.re * u10.re + u10.im * u10.im).sqrt();

                if mag0 + mag1 > 1e-15 {
                    let theta = 2.0 * mag1.atan2(mag0);
                    if theta.abs() > self.config.truncation_threshold {
                        ops.push(CircuitOp::RY {
                            qubit: mode,
                            angle: theta,
                        });
                    }

                    // Phase from complex entries
                    let phase = u10.im.atan2(u10.re) - u00.im.atan2(u00.re);
                    if phase.abs() > self.config.truncation_threshold {
                        ops.push(CircuitOp::RZ {
                            qubit: mode,
                            angle: phase,
                        });
                    }
                }
            }
        }

        Ok(ops)
    }

    /// Get fidelity of last preparation.
    pub fn last_fidelity(&self) -> f64 {
        self.last_decomp.as_ref().map(|d| d.fidelity).unwrap_or(0.0)
    }

    /// Get error of last preparation.
    pub fn last_error(&self) -> f64 {
        self.last_decomp
            .as_ref()
            .map(|d| d.error)
            .unwrap_or(f64::MAX)
    }

    /// Access the last decomposition result.
    pub fn last_decomposition(&self) -> Option<&TuckerDecomposition> {
        self.last_decomp.as_ref()
    }
}

/// Compute the Frobenius norm of a complex tensor stored as a flat vector.
fn core_frobenius_norm(tensor: &[C64]) -> f64 {
    tensor
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum::<f64>()
        .sqrt()
}

/// Compute approximation error (L2 norm of difference) and fidelity (|<a|b>|^2).
fn compute_error_and_fidelity(original: &[C64], reconstructed: &[C64]) -> (f64, f64) {
    let mut diff_norm_sq = 0.0;
    let mut inner = C64::new(0.0, 0.0);
    let mut orig_norm_sq = 0.0;
    let mut recon_norm_sq = 0.0;

    for (a, b) in original.iter().zip(reconstructed.iter()) {
        let d = C64::new(a.re - b.re, a.im - b.im);
        diff_norm_sq += d.re * d.re + d.im * d.im;

        // <a|b> = conj(a) * b
        inner += C64::new(a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re);

        orig_norm_sq += a.re * a.re + a.im * a.im;
        recon_norm_sq += b.re * b.re + b.im * b.im;
    }

    let error = diff_norm_sq.sqrt();

    // Fidelity = |<a|b>|^2 / (||a||^2 * ||b||^2)
    let inner_mag_sq = inner.re * inner.re + inner.im * inner.im;
    let denom = orig_norm_sq * recon_norm_sq;
    let fidelity = if denom > 1e-30 {
        inner_mag_sq / denom
    } else {
        0.0
    };

    (error, fidelity)
}

/// Errors in Tucker state preparation.
#[derive(Clone, Debug)]
pub enum TuckerError {
    InvalidStateLength { expected: usize, got: usize },
    NoConvergence { iterations: usize, error: f64 },
    InvalidDimensions { message: String },
    NumericalError { message: String },
}

impl std::fmt::Display for TuckerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidStateLength { expected, got } => {
                write!(
                    f,
                    "Invalid state length: expected {}, got {}",
                    expected, got
                )
            }
            Self::NoConvergence { iterations, error } => {
                write!(
                    f,
                    "No convergence after {} iterations (error: {})",
                    iterations, error
                )
            }
            Self::InvalidDimensions { message } => write!(f, "Invalid dimensions: {}", message),
            Self::NumericalError { message } => write!(f, "Numerical error: {}", message),
        }
    }
}

impl std::error::Error for TuckerError {}

/// Adaptive Tucker state preparation.
///
/// Progressively increases Tucker ranks until the desired fidelity is reached,
/// or the maximum rank is exhausted.
pub struct AdaptiveTuckerPrep {
    prep: TuckerPrep,
    errors: Vec<f64>,
    fidelity_threshold: f64,
}

impl AdaptiveTuckerPrep {
    pub fn new(n_qubits: usize, config: TuckerConfig) -> Self {
        Self {
            prep: TuckerPrep::new(n_qubits, config),
            errors: Vec::new(),
            fidelity_threshold: 0.99,
        }
    }

    /// Set the fidelity threshold for convergence.
    pub fn set_fidelity_threshold(&mut self, threshold: f64) {
        self.fidelity_threshold = threshold;
    }

    pub fn prepare_adaptive(
        &mut self,
        target: &[C64],
        max_rank: usize,
    ) -> Result<Vec<CircuitOp>, TuckerError> {
        let mut rank = 1;

        loop {
            let mut config = self.prep.config.clone();
            config.max_rank = rank;
            self.prep.config = config;

            let circuit = self.prep.prepare(target)?;
            let fidelity = self.prep.last_fidelity();
            let error = self.prep.last_error();
            self.errors.push(error);

            if fidelity >= self.fidelity_threshold || rank >= max_rank {
                return Ok(circuit);
            }

            rank = (rank * 2).min(max_rank);
            if rank == self.prep.config.max_rank {
                // Already tried this rank, avoid infinite loop
                return Ok(circuit);
            }
        }
    }

    /// Get the convergence history (error at each rank tried).
    pub fn error_history(&self) -> &[f64] {
        &self.errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tucker_config_defaults() {
        let config = TuckerConfig::default();
        assert_eq!(config.max_rank, 4);
        assert!(config.iterative_refinement);
    }

    #[test]
    fn test_tucker_prep_creation() {
        let prep = TuckerPrep::new(4, TuckerConfig::default());
        assert_eq!(prep.n_qubits, 4);
    }

    #[test]
    fn test_prepare_invalid_length() {
        let mut prep = TuckerPrep::new(4, TuckerConfig::default());
        let short_state = vec![C64::new(1.0, 0.0); 8];

        let result = prep.prepare(&short_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_computational_basis_state() {
        // |00> = [1, 0, 0, 0] -- a product state should decompose exactly
        let mut prep = TuckerPrep::new(2, TuckerConfig::default());
        let state = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
        ];

        let result = prep.prepare(&state);
        assert!(result.is_ok());

        let fidelity = prep.last_fidelity();
        assert!(
            fidelity > 0.999,
            "Computational basis state should have near-perfect fidelity, got {}",
            fidelity
        );
        let error = prep.last_error();
        assert!(
            error < 1e-6,
            "Computational basis state should have near-zero error, got {}",
            error
        );
    }

    #[test]
    fn test_prepare_bell_state() {
        // Bell state (|00> + |11>) / sqrt(2)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut prep = TuckerPrep::new(
            2,
            TuckerConfig {
                max_rank: 2,
                iterative_refinement: true,
                ..TuckerConfig::default()
            },
        );

        let state = vec![
            C64::new(inv_sqrt2, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(inv_sqrt2, 0.0),
        ];

        let result = prep.prepare(&state);
        assert!(result.is_ok());

        let fidelity = prep.last_fidelity();
        assert!(
            fidelity > 0.99,
            "Bell state with rank 2 should have high fidelity, got {}",
            fidelity
        );
    }

    #[test]
    fn test_prepare_ghz_3qubit() {
        // GHZ state (|000> + |111>) / sqrt(2) -- 3 qubits
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut prep = TuckerPrep::new(
            3,
            TuckerConfig {
                max_rank: 2,
                iterative_refinement: true,
                ..TuckerConfig::default()
            },
        );

        let mut state = vec![C64::new(0.0, 0.0); 8];
        state[0] = C64::new(inv_sqrt2, 0.0); // |000>
        state[7] = C64::new(inv_sqrt2, 0.0); // |111>

        let result = prep.prepare(&state);
        assert!(result.is_ok());

        let fidelity = prep.last_fidelity();
        assert!(
            fidelity > 0.5,
            "GHZ state Tucker decomposition should have reasonable fidelity, got {}",
            fidelity
        );
    }

    #[test]
    fn test_fidelity_tracking() {
        let mut prep = TuckerPrep::new(2, TuckerConfig::default());

        let state = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
        ];

        prep.prepare(&state).unwrap();
        let fidelity = prep.last_fidelity();
        assert!(
            fidelity >= 0.0 && fidelity <= 1.01,
            "Fidelity should be in [0, 1], got {}",
            fidelity
        );
    }

    #[test]
    fn test_reconstruction_roundtrip() {
        // Verify that reconstructing from the decomposition yields the original state
        let mut prep = TuckerPrep::new(
            2,
            TuckerConfig {
                max_rank: 2, // Full rank for 2-qubit = exact
                iterative_refinement: true,
                ..TuckerConfig::default()
            },
        );

        let state = vec![
            C64::new(0.5, 0.0),
            C64::new(0.5, 0.0),
            C64::new(0.5, 0.0),
            C64::new(0.5, 0.0),
        ];

        prep.prepare(&state).unwrap();
        let decomp = prep.last_decomposition().unwrap();

        let reconstructed = decomp.reconstruct(&[2, 2]);
        let (error, fidelity) = compute_error_and_fidelity(
            // The prepare() normalizes, so compare against normalized
            &state
                .iter()
                .map(|c| {
                    let norm: f64 = state
                        .iter()
                        .map(|v| v.re * v.re + v.im * v.im)
                        .sum::<f64>()
                        .sqrt();
                    C64::new(c.re / norm, c.im / norm)
                })
                .collect::<Vec<_>>(),
            &reconstructed,
        );

        assert!(
            fidelity > 0.99,
            "Full-rank reconstruction should be near-exact, fidelity = {}",
            fidelity
        );
        assert!(
            error < 0.1,
            "Full-rank reconstruction error should be small, error = {}",
            error
        );
    }

    #[test]
    fn test_adaptive_prep() {
        let config = TuckerConfig::default();
        let mut prep = AdaptiveTuckerPrep::new(4, config);

        let state: Vec<C64> = (0..16)
            .map(|i| C64::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let result = prep.prepare_adaptive(&state, 2);
        assert!(result.is_ok());
        assert!(!prep.error_history().is_empty());
    }

    #[test]
    fn test_rank1_product_state() {
        // A product state |+>|+> should decompose exactly at rank 1
        let half = 0.5_f64.sqrt();
        let mut prep = TuckerPrep::new(
            2,
            TuckerConfig {
                max_rank: 1,
                iterative_refinement: false,
                ..TuckerConfig::default()
            },
        );

        // |+>|+> = 0.5 * (|00> + |01> + |10> + |11>)
        let state = vec![
            C64::new(0.5, 0.0),
            C64::new(0.5, 0.0),
            C64::new(0.5, 0.0),
            C64::new(0.5, 0.0),
        ];

        prep.prepare(&state).unwrap();
        let fidelity = prep.last_fidelity();
        assert!(
            fidelity > 0.99,
            "Product state should decompose exactly at rank 1, fidelity = {}",
            fidelity
        );
    }

    #[test]
    fn test_complex_state_decomposition() {
        // Test with a state that has nontrivial complex phases
        let mut prep = TuckerPrep::new(
            2,
            TuckerConfig {
                max_rank: 2,
                iterative_refinement: true,
                ..TuckerConfig::default()
            },
        );

        let state = vec![
            C64::new(0.5, 0.5),
            C64::new(-0.5, 0.0),
            C64::new(0.0, 0.5),
            C64::new(0.5, -0.5),
        ];

        let result = prep.prepare(&state);
        assert!(result.is_ok());

        let fidelity = prep.last_fidelity();
        assert!(
            fidelity > 0.9,
            "Complex state should have reasonable fidelity at rank 2, got {}",
            fidelity
        );
    }

    #[test]
    fn test_zero_state_rejected() {
        let mut prep = TuckerPrep::new(2, TuckerConfig::default());
        let state = vec![C64::new(0.0, 0.0); 4];

        let result = prep.prepare(&state);
        assert!(result.is_err(), "Zero state should be rejected");
    }
}
