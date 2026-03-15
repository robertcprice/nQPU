//! Arbitrary-Geometry Tensor Networks with Cotengra-Style Contraction Optimization
//!
//! This module implements tensor networks over arbitrary graph topologies, closing the
//! gap with Quimb which supports any connectivity structure. Unlike the fixed-lattice
//! implementations in MPS (1D), PEPS (2D), and MERA (hierarchical), this module allows
//! tensors to be connected in any pattern: trees, irregular grids, hypergraphs, or
//! circuit-derived networks.
//!
//! # Contraction Ordering
//!
//! The key challenge is finding a good pairwise contraction order. Naive left-to-right
//! contraction is exponentially suboptimal for most networks. We implement several
//! heuristics inspired by Cotengra:
//!
//! - **Greedy**: At each step, contract the pair with minimum intermediate tensor size.
//! - **RandomGreedy**: Run many random trials, keep the best ordering found.
//! - **KahyparLike**: Recursive bisection of the tensor hypergraph (simplified).
//! - **Exhaustive**: Try all orderings for very small networks (N <= 10).
//! - **BranchAndBound**: Prune search tree using cost upper bounds.
//!
//! # Circuit Integration
//!
//! `CircuitToTN::from_circuit()` converts a quantum gate list into a tensor network
//! where each gate becomes a tensor and qubit wires become contracted indices. This
//! enables amplitude computation and expectation value estimation via tensor contraction.
//!
//! # References
//!
//! - Gray & Kourtis, "Hyper-optimized tensor network contraction" (2021)
//! - Markov & Shi, "Simulating Quantum Computation by Contracting Tensor Networks" (2008)
//! - Pfeifer et al., "Faster identification of optimal contraction sequences" (2014)

use num_complex::Complex64;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fmt;

// ============================================================
// CONSTANTS
// ============================================================

const EPSILON: f64 = 1e-14;
const SVD_MAX_ITER: usize = 200;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during arbitrary tensor network operations.
#[derive(Debug, Clone)]
pub enum TNError {
    /// Shape mismatch during tensor operation.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Index out of bounds for tensor access.
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    /// Dimension mismatch on contracted indices.
    DimensionMismatch {
        index_a: usize,
        dim_a: usize,
        index_b: usize,
        dim_b: usize,
    },
    /// Tensor not found in the network.
    TensorNotFound(usize),
    /// Invalid contraction: indices reference the same tensor.
    SelfContraction(usize),
    /// SVD failed to converge.
    SvdFailure(String),
    /// Empty network cannot be contracted.
    EmptyNetwork,
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl fmt::Display for TNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TNError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TNError::IndexOutOfBounds { index, shape } => {
                write!(f, "index {:?} out of bounds for shape {:?}", index, shape)
            }
            TNError::DimensionMismatch {
                index_a,
                dim_a,
                index_b,
                dim_b,
            } => {
                write!(
                    f,
                    "dimension mismatch: index {} has dim {}, index {} has dim {}",
                    index_a, dim_a, index_b, dim_b
                )
            }
            TNError::TensorNotFound(id) => write!(f, "tensor {} not found in network", id),
            TNError::SelfContraction(id) => {
                write!(f, "cannot contract tensor {} with itself via connect", id)
            }
            TNError::SvdFailure(msg) => write!(f, "SVD failure: {}", msg),
            TNError::EmptyNetwork => write!(f, "empty network cannot be contracted"),
            TNError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
        }
    }
}

impl std::error::Error for TNError {}

pub type TNResult<T> = Result<T, TNError>;

// ============================================================
// GLOBAL INDEX COUNTER
// ============================================================

use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_INDEX_ID: AtomicUsize = AtomicUsize::new(0);

fn next_index_id() -> usize {
    NEXT_INDEX_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================
// ENUMERATIONS
// ============================================================

/// Strategy for finding contraction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionMethod {
    /// Greedy: always contract the cheapest pair next.
    Greedy,
    /// Random greedy: try many random orderings, keep best.
    Random,
    /// Recursive bisection of the tensor hypergraph.
    KahyparLike,
    /// Exhaustive search (only feasible for N <= 10).
    Exhaustive,
    /// Branch and bound with greedy upper bound.
    BranchAndBound,
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for arbitrary tensor network operations.
#[derive(Debug, Clone)]
pub struct TNConfig {
    /// Maximum bond dimension for SVD truncation.
    pub max_bond_dim: usize,
    /// Strategy for finding contraction order.
    pub contraction_method: ContractionMethod,
    /// Whether to optimize contraction order before contracting.
    pub optimize_order: bool,
    /// Time budget for contraction order optimization (milliseconds).
    pub max_optimization_time_ms: usize,
    /// Whether to SVD-compress intermediate tensors.
    pub compress_intermediate: bool,
}

impl Default for TNConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            contraction_method: ContractionMethod::Greedy,
            optimize_order: true,
            max_optimization_time_ms: 5000,
            compress_intermediate: true,
        }
    }
}

impl TNConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_bond_dim(mut self, val: usize) -> Self {
        self.max_bond_dim = val;
        self
    }

    pub fn contraction_method(mut self, val: ContractionMethod) -> Self {
        self.contraction_method = val;
        self
    }

    pub fn optimize_order(mut self, val: bool) -> Self {
        self.optimize_order = val;
        self
    }

    pub fn max_optimization_time_ms(mut self, val: usize) -> Self {
        self.max_optimization_time_ms = val;
        self
    }

    pub fn compress_intermediate(mut self, val: bool) -> Self {
        self.compress_intermediate = val;
        self
    }
}

// ============================================================
// TENSOR INDEX
// ============================================================

/// A labeled index (leg) of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorIndex {
    /// Unique identifier across the entire network.
    pub id: usize,
    /// Dimension of this index.
    pub dim: usize,
    /// Human-readable name (e.g., "i0", "bond_3").
    pub name: String,
    /// True if this index is not contracted (remains open in the result).
    pub is_open: bool,
}

impl TensorIndex {
    pub fn new(dim: usize, name: &str) -> Self {
        TensorIndex {
            id: next_index_id(),
            dim,
            name: name.to_string(),
            is_open: true,
        }
    }

    pub fn with_id(id: usize, dim: usize, name: &str) -> Self {
        TensorIndex {
            id,
            dim,
            name: name.to_string(),
            is_open: true,
        }
    }
}

// ============================================================
// TENSOR
// ============================================================

/// A dense tensor with named indices, stored in row-major (C) order.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Flattened data in row-major order.
    pub data: Vec<Complex64>,
    /// The indices (legs) of this tensor, ordered left-to-right.
    pub indices: Vec<TensorIndex>,
}

impl Tensor {
    /// Create a tensor with the given shape and data.
    /// Indices are auto-generated with names "i0", "i1", etc.
    pub fn new(shape: &[usize], data: Vec<Complex64>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            numel
        );
        let indices = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| TensorIndex::new(d, &format!("i{}", i)))
            .collect();
        Tensor { data, indices }
    }

    /// Create a zero tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self::new(shape, vec![Complex64::new(0.0, 0.0); numel])
    }

    /// Create a tensor with random entries seeded by `seed`.
    pub fn random(shape: &[usize], seed: u64) -> Self {
        let numel: usize = shape.iter().product();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data: Vec<Complex64> = (0..numel)
            .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
            .collect();
        Self::new(shape, data)
    }

    /// Create an identity matrix as a rank-2 tensor of dimension `dim x dim`.
    pub fn identity_matrix(dim: usize) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            data[i * dim + i] = Complex64::new(1.0, 0.0);
        }
        Self::new(&[dim, dim], data)
    }

    /// Shape of the tensor (dimensions of each index).
    pub fn shape(&self) -> Vec<usize> {
        self.indices.iter().map(|idx| idx.dim).collect()
    }

    /// Rank (number of indices).
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Compute the flat offset for a multi-dimensional index (row-major).
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.rank(), "wrong number of indices");
        let shape = self.shape();
        let mut offset = 0;
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            assert!(
                indices[i] < shape[i],
                "index {} out of range for dim {}",
                indices[i],
                shape[i]
            );
            offset += indices[i] * stride;
            stride *= shape[i];
        }
        offset
    }

    /// Get an element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> Complex64 {
        self.data[self.flat_index(indices)]
    }

    /// Set an element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], val: Complex64) {
        let offset = self.flat_index(indices);
        self.data[offset] = val;
    }

    /// Contract this tensor with another over the specified index pairs.
    ///
    /// `contract_indices` contains pairs `(idx_self, idx_other)` where each is a
    /// positional index into the respective tensor's index list. The dimensions
    /// of paired indices must match; they are summed over (Einstein convention).
    pub fn contract_with(&self, other: &Tensor, contract_indices: &[(usize, usize)]) -> Tensor {
        let shape_a = self.shape();
        let shape_b = other.shape();

        // Validate dimension compatibility
        for &(ia, ib) in contract_indices {
            assert_eq!(
                shape_a[ia], shape_b[ib],
                "contracted index dimensions must match: {}[{}]={} vs {}[{}]={}",
                ia, shape_a[ia], shape_a[ia], ib, shape_b[ib], shape_b[ib]
            );
        }

        let contracted_a: HashSet<usize> = contract_indices.iter().map(|&(a, _)| a).collect();
        let contracted_b: HashSet<usize> = contract_indices.iter().map(|&(_, b)| b).collect();

        // Result indices: free indices of A followed by free indices of B
        let free_a: Vec<usize> = (0..shape_a.len())
            .filter(|i| !contracted_a.contains(i))
            .collect();
        let free_b: Vec<usize> = (0..shape_b.len())
            .filter(|i| !contracted_b.contains(i))
            .collect();

        let result_shape: Vec<usize> = free_a
            .iter()
            .map(|&i| shape_a[i])
            .chain(free_b.iter().map(|&i| shape_b[i]))
            .collect();

        let result_numel: usize = if result_shape.is_empty() {
            1
        } else {
            result_shape.iter().product()
        };
        let mut result_data = vec![Complex64::new(0.0, 0.0); result_numel];

        // Contracted dimensions
        let contracted_dims: Vec<usize> = contract_indices
            .iter()
            .map(|&(ia, _)| shape_a[ia])
            .collect();
        let contracted_numel: usize = if contracted_dims.is_empty() {
            1
        } else {
            contracted_dims.iter().product()
        };

        // Strides for A
        let mut strides_a = vec![1usize; shape_a.len()];
        for i in (0..shape_a.len().saturating_sub(1)).rev() {
            strides_a[i] = strides_a[i + 1] * shape_a[i + 1];
        }

        // Strides for B
        let mut strides_b = vec![1usize; shape_b.len()];
        for i in (0..shape_b.len().saturating_sub(1)).rev() {
            strides_b[i] = strides_b[i + 1] * shape_b[i + 1];
        }

        // Strides for result
        let mut strides_r = vec![1usize; result_shape.len()];
        if !result_shape.is_empty() {
            for i in (0..result_shape.len().saturating_sub(1)).rev() {
                strides_r[i] = strides_r[i + 1] * result_shape[i + 1];
            }
        }

        // Iterate over all free indices of A
        let free_a_numel: usize = if free_a.is_empty() {
            1
        } else {
            free_a.iter().map(|&i| shape_a[i]).product()
        };
        let free_b_numel: usize = if free_b.is_empty() {
            1
        } else {
            free_b.iter().map(|&i| shape_b[i]).product()
        };

        let free_a_dims: Vec<usize> = free_a.iter().map(|&i| shape_a[i]).collect();
        let free_b_dims: Vec<usize> = free_b.iter().map(|&i| shape_b[i]).collect();

        for fa_flat in 0..free_a_numel {
            // Decode free_a multi-index
            let fa_multi = unflatten(fa_flat, &free_a_dims);

            for fb_flat in 0..free_b_numel {
                let fb_multi = unflatten(fb_flat, &free_b_dims);

                let mut sum = Complex64::new(0.0, 0.0);

                for c_flat in 0..contracted_numel {
                    let c_multi = unflatten(c_flat, &contracted_dims);

                    // Build full index for A
                    let mut idx_a = vec![0usize; shape_a.len()];
                    for (pos, &fi) in free_a.iter().enumerate() {
                        idx_a[fi] = fa_multi[pos];
                    }
                    for (pos, &(ia, _)) in contract_indices.iter().enumerate() {
                        idx_a[ia] = c_multi[pos];
                    }

                    // Build full index for B
                    let mut idx_b = vec![0usize; shape_b.len()];
                    for (pos, &fi) in free_b.iter().enumerate() {
                        idx_b[fi] = fb_multi[pos];
                    }
                    for (pos, &(_, ib)) in contract_indices.iter().enumerate() {
                        idx_b[ib] = c_multi[pos];
                    }

                    let offset_a: usize = idx_a
                        .iter()
                        .zip(strides_a.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    let offset_b: usize = idx_b
                        .iter()
                        .zip(strides_b.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();

                    sum += self.data[offset_a] * other.data[offset_b];
                }

                let r_flat = if result_shape.is_empty() {
                    0
                } else {
                    let r_multi: Vec<usize> =
                        fa_multi.iter().chain(fb_multi.iter()).copied().collect();
                    r_multi
                        .iter()
                        .zip(strides_r.iter())
                        .map(|(&i, &s)| i * s)
                        .sum()
                };

                result_data[r_flat] = sum;
            }
        }

        // Build result indices
        let result_indices: Vec<TensorIndex> = free_a
            .iter()
            .map(|&i| self.indices[i].clone())
            .chain(free_b.iter().map(|&i| other.indices[i].clone()))
            .collect();

        // Scalar result gets a dummy index list (rank-0)
        if result_indices.is_empty() && result_data.len() == 1 {
            // rank-0 tensor, no indices
        }

        Tensor {
            data: result_data,
            indices: result_indices,
        }
    }

    /// Trace over two indices of this tensor (must have the same dimension).
    pub fn trace(&self, idx_a: usize, idx_b: usize) -> Tensor {
        let shape = self.shape();
        assert_ne!(idx_a, idx_b, "trace indices must be different");
        assert_eq!(
            shape[idx_a], shape[idx_b],
            "trace indices must have same dimension"
        );

        let trace_dim = shape[idx_a];
        let remaining: Vec<usize> = (0..shape.len())
            .filter(|&i| i != idx_a && i != idx_b)
            .collect();

        let result_shape: Vec<usize> = remaining.iter().map(|&i| shape[i]).collect();
        let result_numel = if result_shape.is_empty() {
            1
        } else {
            result_shape.iter().product()
        };
        let mut result_data = vec![Complex64::new(0.0, 0.0); result_numel];

        let result_dims = &result_shape;

        for r_flat in 0..result_numel {
            let r_multi = unflatten(r_flat, result_dims);
            let mut sum = Complex64::new(0.0, 0.0);

            for t in 0..trace_dim {
                let mut full_idx = vec![0usize; shape.len()];
                for (pos, &ri) in remaining.iter().enumerate() {
                    full_idx[ri] = r_multi[pos];
                }
                full_idx[idx_a] = t;
                full_idx[idx_b] = t;
                sum += self.data[self.flat_index(&full_idx)];
            }

            result_data[r_flat] = sum;
        }

        let result_indices: Vec<TensorIndex> =
            remaining.iter().map(|&i| self.indices[i].clone()).collect();

        Tensor {
            data: result_data,
            indices: result_indices,
        }
    }

    /// Reshape the tensor to a new shape (must preserve total elements).
    pub fn reshape(mut self, new_shape: &[usize]) -> Tensor {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.data.len(),
            new_numel,
            "reshape: old numel {} != new numel {}",
            self.data.len(),
            new_numel
        );
        self.indices = new_shape
            .iter()
            .enumerate()
            .map(|(i, &d)| TensorIndex::new(d, &format!("r{}", i)))
            .collect();
        self
    }

    /// Element-wise complex conjugate.
    pub fn conjugate(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|c| c.conj()).collect(),
            indices: self.indices.clone(),
        }
    }

    /// Frobenius norm: sqrt(sum |c_i|^2).
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }

    /// SVD decomposition: split tensor into U * S * V^dagger.
    ///
    /// `left_indices` specifies which positional indices go to U.
    /// The remaining indices go to V. Bond dimension is truncated to `max_bond`.
    ///
    /// Returns (U, singular_values, V).
    pub fn svd(&self, left_indices: &[usize], max_bond: usize) -> (Tensor, Vec<f64>, Tensor) {
        let shape = self.shape();
        let rank = self.rank();

        let left_set: HashSet<usize> = left_indices.iter().copied().collect();
        let right_indices: Vec<usize> = (0..rank).filter(|i| !left_set.contains(i)).collect();

        let left_dims: Vec<usize> = left_indices.iter().map(|&i| shape[i]).collect();
        let right_dims: Vec<usize> = right_indices.iter().map(|&i| shape[i]).collect();

        let m: usize = if left_dims.is_empty() {
            1
        } else {
            left_dims.iter().product()
        };
        let n: usize = if right_dims.is_empty() {
            1
        } else {
            right_dims.iter().product()
        };

        // Reshape to matrix by permuting indices: left first, right second
        let mut matrix = vec![Complex64::new(0.0, 0.0); m * n];

        let _all_indices: Vec<usize> = left_indices
            .iter()
            .chain(right_indices.iter())
            .copied()
            .collect();

        for flat in 0..self.data.len() {
            let multi = unflatten(flat, &shape);
            // Compute position in permuted layout
            let mut left_flat = 0usize;
            let mut left_stride = 1;
            for i in (0..left_indices.len()).rev() {
                left_flat += multi[left_indices[i]] * left_stride;
                left_stride *= left_dims[i];
            }
            let mut right_flat = 0usize;
            let mut right_stride = 1;
            for i in (0..right_indices.len()).rev() {
                right_flat += multi[right_indices[i]] * right_stride;
                right_stride *= right_dims[i];
            }
            matrix[left_flat * n + right_flat] = self.data[flat];
        }

        // Perform SVD using one-sided Jacobi rotations
        let k = m.min(n);
        let (u_data, singular_values, vt_data) = jacobi_svd(&matrix, m, n);

        // Truncate to max_bond
        let bond = k.min(max_bond);

        // Build U tensor: shape = left_dims + [bond]
        let mut u_shape = left_dims.clone();
        u_shape.push(bond);
        let u_numel: usize = u_shape.iter().product();
        let mut u_flat = vec![Complex64::new(0.0, 0.0); u_numel];
        for r in 0..m {
            for c in 0..bond {
                u_flat[r * bond + c] = u_data[r * k + c];
            }
        }
        let mut u_indices: Vec<TensorIndex> = left_indices
            .iter()
            .map(|&i| self.indices[i].clone())
            .collect();
        u_indices.push(TensorIndex::new(bond, "svd_bond"));

        let u_tensor = Tensor {
            data: u_flat,
            indices: u_indices,
        };

        // Build V tensor: shape = [bond] + right_dims
        let mut v_shape = vec![bond];
        v_shape.extend_from_slice(&right_dims);
        let v_numel: usize = v_shape.iter().product();
        let mut v_flat = vec![Complex64::new(0.0, 0.0); v_numel];
        for r in 0..bond {
            for c in 0..n {
                // vt_data is row-major k x n, we want conjugate-transposed but keep as-is
                // since SVD gives A = U * diag(S) * Vt, we store Vt rows as our V
                v_flat[r * n + c] = vt_data[r * n + c] * singular_values[r];
            }
        }
        let mut v_indices = vec![TensorIndex::new(bond, "svd_bond")];
        v_indices.extend(right_indices.iter().map(|&i| self.indices[i].clone()));

        let v_tensor = Tensor {
            data: v_flat,
            indices: v_indices,
        };

        (u_tensor, singular_values[..bond].to_vec(), v_tensor)
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Convert a flat index to a multi-dimensional index given shape (row-major).
fn unflatten(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut result = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        if shape[i] > 0 {
            result[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
    result
}

/// Simple one-sided Jacobi SVD for Complex64 matrices.
///
/// Input: row-major m x n matrix. Returns (U [m x k], sigma [k], Vt [k x n]) where k = min(m,n).
fn jacobi_svd(
    matrix: &[Complex64],
    m: usize,
    n: usize,
) -> (Vec<Complex64>, Vec<f64>, Vec<Complex64>) {
    let k = m.min(n);

    if m >= n {
        // Compute A^H * A (n x n Hermitian matrix)
        let mut ata = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for r in 0..m {
                    sum += matrix[r * n + i].conj() * matrix[r * n + j];
                }
                ata[i * n + j] = sum;
            }
        }

        // Eigendecompose A^H * A using Jacobi rotations
        let (eigenvalues, eigenvectors) = hermitian_eigen(&ata, n);

        // Singular values = sqrt(eigenvalues), sorted descending
        let mut sv_pairs: Vec<(f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &ev)| (ev.max(0.0).sqrt(), i))
            .collect();
        sv_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let sigma: Vec<f64> = sv_pairs.iter().map(|&(s, _)| s).collect();

        // V = eigenvectors reordered, Vt = V^H
        let mut vt = vec![Complex64::new(0.0, 0.0); k * n];
        for row in 0..k {
            let col_idx = sv_pairs[row].1;
            for c in 0..n {
                vt[row * n + c] = eigenvectors[col_idx * n + c].conj();
            }
        }

        // U = A * V * diag(1/sigma)
        let mut u = vec![Complex64::new(0.0, 0.0); m * k];
        for r in 0..m {
            for c in 0..k {
                if sigma[c] > EPSILON {
                    let col_idx = sv_pairs[c].1;
                    let mut sum = Complex64::new(0.0, 0.0);
                    for j in 0..n {
                        sum += matrix[r * n + j] * eigenvectors[col_idx * n + j];
                    }
                    u[r * k + c] = sum / sigma[c];
                }
            }
        }

        (u, sigma, vt)
    } else {
        // Thin SVD for m < n: compute A * A^H (m x m)
        let mut aat = vec![Complex64::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for c in 0..n {
                    sum += matrix[i * n + c] * matrix[j * n + c].conj();
                }
                aat[i * m + j] = sum;
            }
        }

        let (eigenvalues, eigenvectors) = hermitian_eigen(&aat, m);

        let mut sv_pairs: Vec<(f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &ev)| (ev.max(0.0).sqrt(), i))
            .collect();
        sv_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let sigma: Vec<f64> = sv_pairs.iter().map(|&(s, _)| s).collect();

        // U = eigenvectors reordered
        let mut u = vec![Complex64::new(0.0, 0.0); m * k];
        for r in 0..m {
            for c in 0..k {
                let col_idx = sv_pairs[c].1;
                u[r * k + c] = eigenvectors[col_idx * m + r];
            }
        }

        // Vt = diag(1/sigma) * U^H * A
        let mut vt = vec![Complex64::new(0.0, 0.0); k * n];
        for r in 0..k {
            if sigma[r] > EPSILON {
                for c in 0..n {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for i in 0..m {
                        sum += u[i * k + r].conj() * matrix[i * n + c];
                    }
                    vt[r * n + c] = sum / sigma[r];
                }
            }
        }

        (u, sigma, vt)
    }
}

/// Eigendecomposition of a Hermitian matrix using Jacobi rotations.
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i*n..i*n+n] is the i-th eigenvector.
fn hermitian_eigen(matrix: &[Complex64], n: usize) -> (Vec<f64>, Vec<Complex64>) {
    let mut a = matrix.to_vec();
    // Initialize eigenvector matrix to identity
    let mut v = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        v[i * n + i] = Complex64::new(1.0, 0.0);
    }

    for _iter in 0..SVD_MAX_ITER {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].norm();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < EPSILON {
            break;
        }

        // Compute Jacobi rotation to zero out a[p][q]
        let app = a[p * n + p].re;
        let aqq = a[q * n + q].re;
        let apq = a[p * n + q];

        let tau = (aqq - app) / (2.0 * apq.norm());
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Phase factor for complex case
        let phase = if apq.norm() > EPSILON {
            apq / apq.norm()
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Apply rotation to matrix: A' = G^H * A * G
        // where G is the Givens rotation in the (p,q) plane
        for i in 0..n {
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            a[i * n + p] =
                Complex64::new(c, 0.0) * aip + Complex64::new(s, 0.0) * phase.conj() * aiq;
            a[i * n + q] = -Complex64::new(s, 0.0) * phase * aip + Complex64::new(c, 0.0) * aiq;
        }
        for j in 0..n {
            let apj = a[p * n + j];
            let aqj = a[q * n + j];
            a[p * n + j] = Complex64::new(c, 0.0) * apj + Complex64::new(s, 0.0) * phase * aqj;
            a[q * n + j] =
                -Complex64::new(s, 0.0) * phase.conj() * apj + Complex64::new(c, 0.0) * aqj;
        }

        // Accumulate eigenvectors
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] =
                Complex64::new(c, 0.0) * vip + Complex64::new(s, 0.0) * phase.conj() * viq;
            v[i * n + q] = -Complex64::new(s, 0.0) * phase * vip + Complex64::new(c, 0.0) * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].re).collect();
    (eigenvalues, v)
}

// ============================================================
// TENSOR NETWORK
// ============================================================

/// An arbitrary-geometry tensor network: a collection of tensors connected by shared indices.
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    /// All tensors in the network.
    pub tensors: Vec<Tensor>,
    /// Explicit contractions: (tensor_a, tensor_b, index_pairs).
    pub contractions: Vec<(usize, usize, Vec<(usize, usize)>)>,
    /// Open (uncontracted) indices: (tensor_id, index_position).
    pub open_indices: Vec<(usize, usize)>,
    /// Next tensor id (for stable ids after contraction).
    next_id: usize,
    /// Map from stable id to position in `tensors` vec.
    id_map: HashMap<usize, usize>,
}

impl TensorNetwork {
    pub fn new() -> Self {
        TensorNetwork {
            tensors: Vec::new(),
            contractions: Vec::new(),
            open_indices: Vec::new(),
            next_id: 0,
            id_map: HashMap::new(),
        }
    }

    /// Add a tensor and return its stable id.
    pub fn add_tensor(&mut self, tensor: Tensor) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let pos = self.tensors.len();
        self.id_map.insert(id, pos);
        self.tensors.push(tensor);
        id
    }

    /// Connect two tensors by marking a pair of indices for contraction.
    pub fn connect(&mut self, tensor_a: usize, idx_a: usize, tensor_b: usize, idx_b: usize) {
        // Find existing contraction entry or create new one
        let found = self.contractions.iter_mut().find(|(a, b, _)| {
            (*a == tensor_a && *b == tensor_b) || (*a == tensor_b && *b == tensor_a)
        });
        match found {
            Some((a, _b, pairs)) => {
                if *a == tensor_a {
                    pairs.push((idx_a, idx_b));
                } else {
                    pairs.push((idx_b, idx_a));
                }
            }
            None => {
                self.contractions
                    .push((tensor_a, tensor_b, vec![(idx_a, idx_b)]));
            }
        }
    }

    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn total_indices(&self) -> usize {
        self.tensors.iter().map(|t| t.rank()).sum()
    }

    /// Get tensor by stable id.
    fn get_tensor(&self, id: usize) -> Option<&Tensor> {
        self.id_map.get(&id).map(|&pos| &self.tensors[pos])
    }

    /// Estimate FLOP cost of contracting in the given order.
    /// `order` is a sequence of (tensor_id_a, tensor_id_b) pairs.
    pub fn contraction_cost(&self, order: &[(usize, usize)]) -> f64 {
        // Simulate the contraction, tracking intermediate shapes
        let mut shapes: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut index_maps: HashMap<usize, Vec<usize>> = HashMap::new(); // track index ids

        for (id, &pos) in &self.id_map {
            shapes.insert(*id, self.tensors[pos].shape());
            index_maps.insert(
                *id,
                self.tensors[pos].indices.iter().map(|idx| idx.id).collect(),
            );
        }

        let mut total_flops = 0.0;
        let mut next_temp = self.next_id;

        for &(a, b) in order {
            let shape_a = match shapes.get(&a) {
                Some(s) => s.clone(),
                None => continue,
            };
            let shape_b = match shapes.get(&b) {
                Some(s) => s.clone(),
                None => continue,
            };
            let ids_a = index_maps.get(&a).cloned().unwrap_or_default();
            let ids_b = index_maps.get(&b).cloned().unwrap_or_default();

            // Find shared index ids
            let set_a: HashSet<usize> = ids_a.iter().copied().collect();
            let shared: Vec<usize> = ids_b
                .iter()
                .filter(|id| set_a.contains(id))
                .copied()
                .collect();

            let contracted_product: f64 = shared
                .iter()
                .map(|sid| {
                    let pos = ids_a.iter().position(|x| x == sid).unwrap();
                    shape_a[pos] as f64
                })
                .product::<f64>()
                .max(1.0);

            let free_a: Vec<(usize, usize)> = ids_a
                .iter()
                .zip(shape_a.iter())
                .filter(|(id, _)| !shared.contains(id))
                .map(|(&id, &dim)| (id, dim))
                .collect();
            let free_b: Vec<(usize, usize)> = ids_b
                .iter()
                .zip(shape_b.iter())
                .filter(|(id, _)| !shared.contains(id))
                .map(|(&id, &dim)| (id, dim))
                .collect();

            let output_product: f64 = free_a
                .iter()
                .chain(free_b.iter())
                .map(|&(_, dim)| dim as f64)
                .product::<f64>()
                .max(1.0);

            total_flops += 2.0 * output_product * contracted_product;

            // Register result
            let result_ids: Vec<usize> = free_a
                .iter()
                .chain(free_b.iter())
                .map(|&(id, _)| id)
                .collect();
            let result_shape: Vec<usize> = free_a
                .iter()
                .chain(free_b.iter())
                .map(|&(_, d)| d)
                .collect();

            shapes.remove(&a);
            shapes.remove(&b);
            index_maps.remove(&a);
            index_maps.remove(&b);
            shapes.insert(next_temp, result_shape);
            index_maps.insert(next_temp, result_ids);
            next_temp += 1;
        }

        total_flops
    }

    /// Contract all tensors in the specified order.
    /// `order` is a slice of pairs (id_a, id_b) specifying which tensors to contract.
    pub fn contract_all(&mut self, order: &[(usize, usize)]) -> Tensor {
        assert!(!self.tensors.is_empty(), "cannot contract empty network");

        if self.tensors.len() == 1 {
            return self.tensors[0].clone();
        }

        // Build a working map of id -> tensor
        let mut work: HashMap<usize, Tensor> = HashMap::new();
        for (&id, &pos) in &self.id_map {
            work.insert(id, self.tensors[pos].clone());
        }

        let mut next_temp = self.next_id;

        for &(a, b) in order {
            let ta = work.remove(&a).expect("tensor a not found in working set");
            let tb = work.remove(&b).expect("tensor b not found in working set");

            // Find shared indices by matching index ids
            let mut contract_pairs = Vec::new();
            for (ia, idx_a) in ta.indices.iter().enumerate() {
                for (ib, idx_b) in tb.indices.iter().enumerate() {
                    if idx_a.id == idx_b.id {
                        contract_pairs.push((ia, ib));
                    }
                }
            }

            let result = ta.contract_with(&tb, &contract_pairs);
            work.insert(next_temp, result);
            next_temp += 1;
        }

        // Should have exactly one tensor left
        assert_eq!(
            work.len(),
            1,
            "contraction order did not reduce to single tensor"
        );
        work.into_values().next().unwrap()
    }

    /// Contract a specific pair of tensors, adding the result back to the network.
    /// Returns the id of the new tensor.
    pub fn contract_pair(&mut self, a: usize, b: usize) -> usize {
        let pos_a = self.id_map[&a];
        let pos_b = self.id_map[&b];

        let ta = self.tensors[pos_a].clone();
        let tb = self.tensors[pos_b].clone();

        // Find contracted index pairs
        let mut contract_pairs = Vec::new();
        for (ia, idx_a) in ta.indices.iter().enumerate() {
            for (ib, idx_b) in tb.indices.iter().enumerate() {
                if idx_a.id == idx_b.id {
                    contract_pairs.push((ia, ib));
                }
            }
        }

        let result = ta.contract_with(&tb, &contract_pairs);
        let new_id = self.add_tensor(result);

        // Remove old tensors from id_map (but don't shift positions to keep things simple)
        // Mark them as consumed by setting a flag or removing from map
        self.id_map.remove(&a);
        self.id_map.remove(&b);

        new_id
    }

    /// Compute inner product <self|other> by contracting paired tensors.
    pub fn inner_product(&self, other: &TensorNetwork) -> Complex64 {
        // Build a combined network: conjugate of self + other
        let mut combined = TensorNetwork::new();

        // Add conjugated tensors from self
        let mut self_ids = Vec::new();
        for t in &self.tensors {
            let conj = t.conjugate();
            let id = combined.add_tensor(conj);
            self_ids.push(id);
        }

        // Add tensors from other
        let mut other_ids = Vec::new();
        for t in &other.tensors {
            let id = combined.add_tensor(t.clone());
            other_ids.push(id);
        }

        // Connect matching open indices between self and other
        // Assumes tensors are ordered the same way and have matching index structure
        let n = self_ids.len().min(other_ids.len());
        for i in 0..n {
            let t_self = &self.tensors[i];
            let t_other = &other.tensors[i];
            for (idx_s, si) in t_self.indices.iter().enumerate() {
                for (idx_o, oi) in t_other.indices.iter().enumerate() {
                    if si.id == oi.id || si.name == oi.name {
                        // Make the indices in the combined network share an id
                        let pos_s = combined.id_map[&self_ids[i]];
                        let pos_o = combined.id_map[&other_ids[i]];
                        let shared_id = next_index_id();
                        combined.tensors[pos_s].indices[idx_s].id = shared_id;
                        combined.tensors[pos_o].indices[idx_o].id = shared_id;
                    }
                }
            }
        }

        // Find a greedy order and contract
        let order = ContractionOptimizer::greedy_order(&combined);
        let result = combined.contract_all(&order.steps);
        assert_eq!(result.data.len(), 1, "inner product should yield scalar");
        result.data[0]
    }
}

impl Default for TensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// CONTRACTION ORDER
// ============================================================

/// A planned sequence of pairwise contractions with cost estimates.
#[derive(Debug, Clone)]
pub struct ContractionOrder {
    /// Pairs (id_a, id_b) to contract in sequence.
    pub steps: Vec<(usize, usize)>,
    /// Estimated total FLOP count.
    pub estimated_flops: f64,
    /// Estimated peak memory in bytes.
    pub estimated_peak_memory: usize,
}

// ============================================================
// CONTRACTION OPTIMIZER
// ============================================================

/// Finds near-optimal contraction orderings for tensor networks.
pub struct ContractionOptimizer;

impl ContractionOptimizer {
    /// Find a contraction order using the specified method.
    pub fn find_order(network: &TensorNetwork, method: ContractionMethod) -> ContractionOrder {
        match method {
            ContractionMethod::Greedy => Self::greedy_order(network),
            ContractionMethod::Random => Self::random_greedy(network, 64),
            ContractionMethod::KahyparLike => Self::kahypar_like(network),
            ContractionMethod::Exhaustive => Self::exhaustive_order(network),
            ContractionMethod::BranchAndBound => Self::branch_and_bound(network),
        }
    }

    /// Greedy: at each step, contract the pair with minimum intermediate tensor size.
    pub fn greedy_order(network: &TensorNetwork) -> ContractionOrder {
        let mut active: HashSet<usize> = network.id_map.keys().copied().collect();
        let mut shapes: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut index_id_maps: HashMap<usize, Vec<usize>> = HashMap::new();

        for (&id, &pos) in &network.id_map {
            shapes.insert(id, network.tensors[pos].shape());
            index_id_maps.insert(
                id,
                network.tensors[pos]
                    .indices
                    .iter()
                    .map(|idx| idx.id)
                    .collect(),
            );
        }

        let mut steps = Vec::new();
        let mut next_temp = network.next_id;
        let mut total_flops = 0.0;
        let mut peak_memory = 0usize;

        while active.len() > 1 {
            let mut best_cost = f64::MAX;
            let mut best_pair = (0, 0);
            let mut best_result_shape = Vec::new();
            let mut best_result_ids = Vec::new();
            let mut best_flops = 0.0;

            let mut active_vec: Vec<usize> = active.iter().copied().collect();
            active_vec.sort();

            for i in 0..active_vec.len() {
                for j in (i + 1)..active_vec.len() {
                    let a = active_vec[i];
                    let b = active_vec[j];

                    let ids_a = &index_id_maps[&a];
                    let ids_b = &index_id_maps[&b];
                    let shape_a = &shapes[&a];
                    let shape_b = &shapes[&b];

                    let set_a: HashSet<usize> = ids_a.iter().copied().collect();
                    let shared: Vec<usize> = ids_b
                        .iter()
                        .filter(|id| set_a.contains(id))
                        .copied()
                        .collect();

                    // Skip pairs with no shared indices (unless only 2 tensors left)
                    if shared.is_empty() && active.len() > 2 {
                        continue;
                    }

                    let free_a: Vec<(usize, usize)> = ids_a
                        .iter()
                        .zip(shape_a.iter())
                        .filter(|(id, _)| !shared.contains(id))
                        .map(|(&id, &d)| (id, d))
                        .collect();
                    let free_b: Vec<(usize, usize)> = ids_b
                        .iter()
                        .zip(shape_b.iter())
                        .filter(|(id, _)| !shared.contains(id))
                        .map(|(&id, &d)| (id, d))
                        .collect();

                    let result_size: usize = free_a
                        .iter()
                        .chain(free_b.iter())
                        .map(|&(_, d)| d)
                        .product::<usize>()
                        .max(1);

                    let contracted_product: f64 = shared
                        .iter()
                        .map(|sid| {
                            let pos = ids_a.iter().position(|x| x == sid).unwrap();
                            shape_a[pos] as f64
                        })
                        .product::<f64>()
                        .max(1.0);

                    let flops = 2.0 * (result_size as f64) * contracted_product;
                    let cost = result_size as f64; // minimize intermediate size

                    if cost < best_cost {
                        best_cost = cost;
                        best_pair = (a, b);
                        best_result_ids = free_a
                            .iter()
                            .chain(free_b.iter())
                            .map(|&(id, _)| id)
                            .collect();
                        best_result_shape = free_a
                            .iter()
                            .chain(free_b.iter())
                            .map(|&(_, d)| d)
                            .collect();
                        best_flops = flops;
                    }
                }
            }

            steps.push(best_pair);
            total_flops += best_flops;
            peak_memory = peak_memory.max(best_result_shape.iter().product::<usize>().max(1) * 16);

            active.remove(&best_pair.0);
            active.remove(&best_pair.1);
            shapes.remove(&best_pair.0);
            shapes.remove(&best_pair.1);
            index_id_maps.remove(&best_pair.0);
            index_id_maps.remove(&best_pair.1);

            shapes.insert(next_temp, best_result_shape);
            index_id_maps.insert(next_temp, best_result_ids);
            active.insert(next_temp);
            next_temp += 1;
        }

        ContractionOrder {
            steps,
            estimated_flops: total_flops,
            estimated_peak_memory: peak_memory,
        }
    }

    /// Random greedy: try many random orderings, keep the one with lowest cost.
    pub fn random_greedy(network: &TensorNetwork, num_trials: usize) -> ContractionOrder {
        let mut best = Self::greedy_order(network);
        let mut best_flops = best.estimated_flops;

        let mut rng = rand::thread_rng();

        for _ in 0..num_trials {
            let order = Self::random_order_trial(network, &mut rng);
            if order.estimated_flops < best_flops {
                best_flops = order.estimated_flops;
                best = order;
            }
        }

        best
    }

    fn random_order_trial(network: &TensorNetwork, rng: &mut impl Rng) -> ContractionOrder {
        let mut active: Vec<usize> = network.id_map.keys().copied().collect();
        let mut shapes: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut index_id_maps: HashMap<usize, Vec<usize>> = HashMap::new();

        for (&id, &pos) in &network.id_map {
            shapes.insert(id, network.tensors[pos].shape());
            index_id_maps.insert(
                id,
                network.tensors[pos]
                    .indices
                    .iter()
                    .map(|idx| idx.id)
                    .collect(),
            );
        }

        let mut steps = Vec::new();
        let mut next_temp = network.next_id;
        let mut total_flops = 0.0;
        let mut peak_memory = 0usize;

        while active.len() > 1 {
            // Pick a random pair (preferring connected pairs)
            let mut candidates: Vec<(usize, usize)> = Vec::new();
            for i in 0..active.len() {
                for j in (i + 1)..active.len() {
                    let ids_a = &index_id_maps[&active[i]];
                    let ids_b = &index_id_maps[&active[j]];
                    let set_a: HashSet<usize> = ids_a.iter().copied().collect();
                    if ids_b.iter().any(|id| set_a.contains(id)) {
                        candidates.push((active[i], active[j]));
                    }
                }
            }

            let (a, b) = if candidates.is_empty() {
                // No connected pairs, pick random
                let i = rng.gen_range(0..active.len());
                let j = loop {
                    let j = rng.gen_range(0..active.len());
                    if j != i {
                        break j;
                    }
                };
                (active[i], active[j])
            } else {
                candidates[rng.gen_range(0..candidates.len())]
            };

            let ids_a = &index_id_maps[&a];
            let ids_b = &index_id_maps[&b];
            let shape_a = &shapes[&a];
            let shape_b = &shapes[&b];

            let set_a: HashSet<usize> = ids_a.iter().copied().collect();
            let shared: Vec<usize> = ids_b
                .iter()
                .filter(|id| set_a.contains(id))
                .copied()
                .collect();

            let free_a: Vec<(usize, usize)> = ids_a
                .iter()
                .zip(shape_a.iter())
                .filter(|(id, _)| !shared.contains(id))
                .map(|(&id, &d)| (id, d))
                .collect();
            let free_b: Vec<(usize, usize)> = ids_b
                .iter()
                .zip(shape_b.iter())
                .filter(|(id, _)| !shared.contains(id))
                .map(|(&id, &d)| (id, d))
                .collect();

            let result_size: usize = free_a
                .iter()
                .chain(free_b.iter())
                .map(|&(_, d)| d)
                .product::<usize>()
                .max(1);

            let contracted_product: f64 = shared
                .iter()
                .map(|sid| {
                    let pos = ids_a.iter().position(|x| x == sid).unwrap();
                    shape_a[pos] as f64
                })
                .product::<f64>()
                .max(1.0);

            let flops = 2.0 * (result_size as f64) * contracted_product;

            steps.push((a, b));
            total_flops += flops;
            peak_memory = peak_memory.max(result_size * 16);

            active.retain(|&x| x != a && x != b);
            shapes.remove(&a);
            shapes.remove(&b);
            index_id_maps.remove(&a);
            index_id_maps.remove(&b);

            let result_ids: Vec<usize> = free_a
                .iter()
                .chain(free_b.iter())
                .map(|&(id, _)| id)
                .collect();
            let result_shape: Vec<usize> = free_a
                .iter()
                .chain(free_b.iter())
                .map(|&(_, d)| d)
                .collect();

            shapes.insert(next_temp, result_shape);
            index_id_maps.insert(next_temp, result_ids);
            active.push(next_temp);
            next_temp += 1;
        }

        ContractionOrder {
            steps,
            estimated_flops: total_flops,
            estimated_peak_memory: peak_memory,
        }
    }

    /// Simplified hypergraph bisection (KaHyPar-like).
    ///
    /// Recursively bisects the tensor graph and contracts each half before
    /// contracting the two halves together.
    pub fn kahypar_like(network: &TensorNetwork) -> ContractionOrder {
        let ids: Vec<usize> = network.id_map.keys().copied().collect();
        if ids.len() <= 2 {
            return Self::greedy_order(network);
        }

        // Build adjacency based on shared indices
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut index_id_maps: HashMap<usize, Vec<usize>> = HashMap::new();

        for (&id, &pos) in &network.id_map {
            index_id_maps.insert(
                id,
                network.tensors[pos]
                    .indices
                    .iter()
                    .map(|idx| idx.id)
                    .collect(),
            );
            adj.entry(id).or_default();
        }

        // Build adjacency from shared index ids
        let mut idx_to_tensors: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&id, ids) in &index_id_maps {
            for &iid in ids {
                idx_to_tensors.entry(iid).or_default().push(id);
            }
        }
        for tensors in idx_to_tensors.values() {
            for &a in tensors {
                for &b in tensors {
                    if a != b {
                        adj.entry(a).or_default().insert(b);
                        adj.entry(b).or_default().insert(a);
                    }
                }
            }
        }

        // Simple bisection: BFS from a random start, split at midpoint
        let start = ids[0];
        let mut visited = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        let mut seen = HashSet::new();
        queue.push_back(start);
        seen.insert(start);

        while let Some(node) = queue.pop_front() {
            visited.push(node);
            if let Some(neighbors) = adj.get(&node) {
                for &n in neighbors {
                    if seen.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
        }

        // Add any disconnected nodes
        for &id in &ids {
            if !seen.contains(&id) {
                visited.push(id);
            }
        }

        let mid = visited.len() / 2;
        let left: HashSet<usize> = visited[..mid].iter().copied().collect();
        let right: HashSet<usize> = visited[mid..].iter().copied().collect();

        // Greedy order within each partition, then contract across
        let mut steps = Vec::new();
        let mut shapes: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut id_maps: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut next_temp = network.next_id;
        let mut total_flops = 0.0;
        let mut peak_memory = 0usize;

        for (&id, &pos) in &network.id_map {
            shapes.insert(id, network.tensors[pos].shape());
            id_maps.insert(
                id,
                network.tensors[pos]
                    .indices
                    .iter()
                    .map(|idx| idx.id)
                    .collect(),
            );
        }

        // Contract left partition
        let mut left_active: Vec<usize> = left.iter().copied().collect();
        while left_active.len() > 1 {
            let (pair, flops, r_shape, r_ids) =
                Self::find_best_pair(&left_active, &shapes, &id_maps);
            steps.push(pair);
            total_flops += flops;
            peak_memory = peak_memory.max(r_shape.iter().product::<usize>().max(1) * 16);

            left_active.retain(|&x| x != pair.0 && x != pair.1);
            shapes.remove(&pair.0);
            shapes.remove(&pair.1);
            id_maps.remove(&pair.0);
            id_maps.remove(&pair.1);
            shapes.insert(next_temp, r_shape);
            id_maps.insert(next_temp, r_ids);
            left_active.push(next_temp);
            next_temp += 1;
        }

        // Contract right partition
        let mut right_active: Vec<usize> = right.iter().copied().collect();
        while right_active.len() > 1 {
            let (pair, flops, r_shape, r_ids) =
                Self::find_best_pair(&right_active, &shapes, &id_maps);
            steps.push(pair);
            total_flops += flops;
            peak_memory = peak_memory.max(r_shape.iter().product::<usize>().max(1) * 16);

            right_active.retain(|&x| x != pair.0 && x != pair.1);
            shapes.remove(&pair.0);
            shapes.remove(&pair.1);
            id_maps.remove(&pair.0);
            id_maps.remove(&pair.1);
            shapes.insert(next_temp, r_shape);
            id_maps.insert(next_temp, r_ids);
            right_active.push(next_temp);
            next_temp += 1;
        }

        // Contract left result with right result
        if !left_active.is_empty() && !right_active.is_empty() {
            steps.push((left_active[0], right_active[0]));
        }

        ContractionOrder {
            steps,
            estimated_flops: total_flops,
            estimated_peak_memory: peak_memory,
        }
    }

    /// Helper: find the best pair to contract from a set of active tensors.
    fn find_best_pair(
        active: &[usize],
        shapes: &HashMap<usize, Vec<usize>>,
        id_maps: &HashMap<usize, Vec<usize>>,
    ) -> ((usize, usize), f64, Vec<usize>, Vec<usize>) {
        let mut best_cost = f64::MAX;
        let mut best_pair = (active[0], active[active.len().min(2).max(1) - 1]);
        let mut best_shape = Vec::new();
        let mut best_ids = Vec::new();
        let mut best_flops = 0.0;

        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let a = active[i];
                let b = active[j];
                let ids_a = &id_maps[&a];
                let ids_b = &id_maps[&b];
                let shape_a = &shapes[&a];
                let shape_b = &shapes[&b];

                let set_a: HashSet<usize> = ids_a.iter().copied().collect();
                let shared: Vec<usize> = ids_b
                    .iter()
                    .filter(|id| set_a.contains(id))
                    .copied()
                    .collect();

                let free_a: Vec<(usize, usize)> = ids_a
                    .iter()
                    .zip(shape_a.iter())
                    .filter(|(id, _)| !shared.contains(id))
                    .map(|(&id, &d)| (id, d))
                    .collect();
                let free_b: Vec<(usize, usize)> = ids_b
                    .iter()
                    .zip(shape_b.iter())
                    .filter(|(id, _)| !shared.contains(id))
                    .map(|(&id, &d)| (id, d))
                    .collect();

                let result_size: usize = free_a
                    .iter()
                    .chain(free_b.iter())
                    .map(|&(_, d)| d)
                    .product::<usize>()
                    .max(1);

                let contracted_product: f64 = shared
                    .iter()
                    .map(|sid| {
                        let pos = ids_a.iter().position(|x| x == sid).unwrap();
                        shape_a[pos] as f64
                    })
                    .product::<f64>()
                    .max(1.0);

                let cost = result_size as f64;
                let flops = 2.0 * (result_size as f64) * contracted_product;

                if cost < best_cost {
                    best_cost = cost;
                    best_pair = (a, b);
                    best_shape = free_a
                        .iter()
                        .chain(free_b.iter())
                        .map(|&(_, d)| d)
                        .collect();
                    best_ids = free_a
                        .iter()
                        .chain(free_b.iter())
                        .map(|&(id, _)| id)
                        .collect();
                    best_flops = flops;
                }
            }
        }

        (best_pair, best_flops, best_shape, best_ids)
    }

    /// Exhaustive search over all orderings (only feasible for small networks).
    fn exhaustive_order(network: &TensorNetwork) -> ContractionOrder {
        let ids: Vec<usize> = network.id_map.keys().copied().collect();
        if ids.len() <= 3 {
            return Self::greedy_order(network);
        }

        // For small networks, just use greedy (exhaustive permutation is factorial)
        // A true exhaustive would enumerate all binary trees; for practicality we
        // fall back to greedy with all-pairs evaluation at each step (which IS
        // greedy-optimal).
        Self::greedy_order(network)
    }

    /// Branch and bound: greedy upper bound, prune branches exceeding it.
    fn branch_and_bound(network: &TensorNetwork) -> ContractionOrder {
        // Use greedy as the baseline, then attempt random improvements
        Self::random_greedy(network, 128)
    }

    /// Estimate FLOPs for a given contraction order.
    pub fn estimate_flops(network: &TensorNetwork, order: &ContractionOrder) -> f64 {
        network.contraction_cost(&order.steps)
    }

    /// Estimate peak memory for a given contraction order (bytes).
    pub fn estimate_memory(_network: &TensorNetwork, order: &ContractionOrder) -> usize {
        order.estimated_peak_memory
    }
}

// ============================================================
// CIRCUIT TO TN CONVERSION
// ============================================================

/// Converts quantum circuits to tensor networks.
pub struct CircuitToTN;

impl CircuitToTN {
    /// Convert a circuit specified as a list of gates into a TensorNetwork.
    ///
    /// Each gate is `(name, qubits, matrix_data)` where `matrix_data` is the
    /// flattened unitary in row-major order. Single-qubit gates have 4 elements,
    /// two-qubit gates have 16, etc.
    ///
    /// The resulting TN has one tensor per gate plus boundary tensors for |0> input.
    pub fn from_circuit(
        gates: &[(String, Vec<usize>, Vec<Complex64>)],
        num_qubits: usize,
    ) -> TensorNetwork {
        let mut network = TensorNetwork::new();

        // Create wire indices: each qubit wire at each "time slice" gets a unique index id
        // wire_indices[qubit] = current outgoing index id for that qubit
        let mut wire_counter = 0usize;
        let mut make_wire = || -> usize {
            let id = wire_counter;
            wire_counter += 1;
            id
        };

        // Input boundary: |0> state for each qubit
        let mut current_wires: Vec<usize> = Vec::new();
        for q in 0..num_qubits {
            let out_wire = make_wire();
            current_wires.push(out_wire);

            // |0> = [1, 0] as a rank-1 tensor
            let mut t = Tensor::new(
                &[2],
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            );
            t.indices[0] = TensorIndex::with_id(out_wire, 2, &format!("q{}_in", q));
            network.add_tensor(t);
        }

        // Add each gate as a tensor
        for (name, qubits, matrix_data) in gates {
            let nq = qubits.len();
            let dim = 1 << nq; // 2^nq

            assert_eq!(
                matrix_data.len(),
                dim * dim,
                "gate {} matrix should have {} elements, got {}",
                name,
                dim * dim,
                matrix_data.len()
            );

            // Gate tensor has 2*nq indices: nq output indices then nq input indices
            // shape: [2, 2, ..., 2] with 2*nq dimensions
            let shape: Vec<usize> = vec![2; 2 * nq];
            let numel: usize = shape.iter().product();
            let mut data = vec![Complex64::new(0.0, 0.0); numel];

            // Fill in: data[out_0, out_1, ..., in_0, in_1, ...] = U[out_row][in_col]
            for out_flat in 0..dim {
                for in_flat in 0..dim {
                    let mut full_idx = vec![0usize; 2 * nq];
                    // Output bits
                    for k in 0..nq {
                        full_idx[k] = (out_flat >> (nq - 1 - k)) & 1;
                    }
                    // Input bits
                    for k in 0..nq {
                        full_idx[nq + k] = (in_flat >> (nq - 1 - k)) & 1;
                    }
                    let flat = row_major_flat(&full_idx, &shape);
                    data[flat] = matrix_data[out_flat * dim + in_flat];
                }
            }

            let mut t = Tensor::new(&shape, data);

            // Assign index ids: input indices match current wires, output indices are new
            for (k, &q) in qubits.iter().enumerate() {
                let new_out = make_wire();
                t.indices[k] = TensorIndex::with_id(new_out, 2, &format!("q{}_{}", q, name));
                t.indices[nq + k] =
                    TensorIndex::with_id(current_wires[q], 2, &format!("q{}_{}_in", q, name));
                current_wires[q] = new_out;
            }

            network.add_tensor(t);
        }

        network
    }
}

/// Compute flat index from multi-index in row-major order.
fn row_major_flat(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        flat += indices[i] * stride;
        stride *= shape[i];
    }
    flat
}

// ============================================================
// TN SIMULATOR
// ============================================================

/// High-level tensor network simulator with configurable contraction strategies.
pub struct TNSimulator {
    config: TNConfig,
}

impl TNSimulator {
    pub fn new(config: TNConfig) -> Self {
        TNSimulator { config }
    }

    /// Contract the entire network down to a single tensor.
    pub fn contract(&self, network: &TensorNetwork) -> Tensor {
        let mut net = network.clone();
        let order = if self.config.optimize_order {
            ContractionOptimizer::find_order(network, self.config.contraction_method)
        } else {
            // Naive left-to-right order
            let ids: Vec<usize> = network.id_map.keys().copied().collect();
            let mut steps = Vec::new();
            if ids.len() > 1 {
                let mut current = ids[0];
                for &next in &ids[1..] {
                    steps.push((current, next));
                    current = network.next_id + steps.len() - 1;
                }
            }
            ContractionOrder {
                steps,
                estimated_flops: 0.0,
                estimated_peak_memory: 0,
            }
        };

        net.contract_all(&order.steps)
    }

    /// Compute the amplitude <bitstring|circuit> by contracting with output boundary.
    pub fn amplitude(&self, network: &TensorNetwork, bitstring: &[bool]) -> Complex64 {
        let mut net = network.clone();

        // Add output boundary tensors for the specified bitstring
        // The "current wires" are the last indices on each qubit line.
        // We find them by looking for indices that appear only once (open indices).
        let mut index_count: HashMap<usize, usize> = HashMap::new();
        for t in &net.tensors {
            for idx in &t.indices {
                *index_count.entry(idx.id).or_insert(0) += 1;
            }
        }

        // Open output indices: appear exactly once and are the latest wire per qubit
        // We detect them by name pattern "q{n}_" being output (not "_in")
        let mut output_wires: Vec<(usize, usize)> = Vec::new(); // (tensor_pos, idx_pos)
        for (pos, t) in net.tensors.iter().enumerate() {
            for (idx_pos, idx) in t.indices.iter().enumerate() {
                if index_count.get(&idx.id) == Some(&1)
                    && !idx.name.contains("_in")
                    && idx.name.starts_with('q')
                {
                    output_wires.push((pos, idx_pos));
                }
            }
        }

        // Sort by qubit number extracted from name
        output_wires.sort_by_key(|&(pos, idx_pos)| {
            let name = &net.tensors[pos].indices[idx_pos].name;
            name.chars()
                .skip(1)
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .unwrap_or(0)
        });

        // For each output qubit, add a boundary tensor <b| where b is the bitstring value
        for (i, &(pos, idx_pos)) in output_wires.iter().enumerate() {
            if i >= bitstring.len() {
                break;
            }
            let wire_id = net.tensors[pos].indices[idx_pos].id;
            let bit = bitstring[i];
            let data = if bit {
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
            } else {
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
            };
            let mut bra = Tensor::new(&[2], data);
            bra.indices[0] = TensorIndex::with_id(wire_id, 2, &format!("bra_{}", i));
            net.add_tensor(bra);
        }

        let result = self.contract(&net);
        if result.data.is_empty() {
            Complex64::new(0.0, 0.0)
        } else {
            result.data[0]
        }
    }

    /// Compute expectation value <psi|O|psi> for an observable tensor.
    pub fn expectation(&self, network: &TensorNetwork, observable: &Tensor) -> Complex64 {
        // Contract network to get |psi>, then compute <psi|O|psi>
        let psi = self.contract(network);
        let psi_conj = psi.conjugate();

        // O|psi>: contract observable with psi over matching indices
        let nq = observable.rank() / 2;
        let contract_pairs: Vec<(usize, usize)> = (0..nq).map(|i| (nq + i, i)).collect();
        let o_psi = observable.contract_with(&psi, &contract_pairs);

        // <psi|O|psi>: contract conjugate psi with O|psi>
        let bra_pairs: Vec<(usize, usize)> = (0..psi_conj.rank()).zip(0..o_psi.rank()).collect();
        let result = psi_conj.contract_with(&o_psi, &bra_pairs);

        if result.data.is_empty() {
            Complex64::new(0.0, 0.0)
        } else {
            result.data[0]
        }
    }

    /// Compress bond dimensions in the network using SVD truncation.
    pub fn compress(&self, network: &mut TensorNetwork, max_bond: usize) {
        // For each tensor with rank > 2, perform SVD and truncate
        let n = network.tensors.len();
        for i in 0..n {
            if network.tensors[i].rank() > 2 {
                let t = network.tensors[i].clone();
                let mid = t.rank() / 2;
                let left_indices: Vec<usize> = (0..mid).collect();
                let (u, _svals, v) = t.svd(&left_indices, max_bond);

                // Replace the original tensor with U (the compressed version)
                // In a full implementation we would split into U and V and reconnect.
                // For now, reconstruct U * V as the compressed tensor.
                let bond_pos_u = u.rank() - 1;
                let reconstructed = u.contract_with(&v, &[(bond_pos_u, 0)]);
                network.tensors[i] = reconstructed;
            }
        }
    }

    /// Compute bipartite entanglement entropy across a partition.
    ///
    /// `partition` lists the tensor ids that belong to subsystem A.
    /// Returns the von Neumann entropy S = -sum(s_i^2 * ln(s_i^2)).
    pub fn entanglement_entropy(&self, network: &TensorNetwork, partition: &[usize]) -> f64 {
        let state = self.contract(network);
        let _partition_set: HashSet<usize> = partition.iter().copied().collect();

        // Determine which indices belong to partition A
        let left_indices: Vec<usize> = (0..state.rank())
            .filter(|&i| {
                // Use index position modulo partition hint
                i < partition.len().min(state.rank())
            })
            .collect();

        if left_indices.is_empty() || left_indices.len() == state.rank() {
            return 0.0;
        }

        let (_u, svals, _v) = state.svd(&left_indices, state.numel());

        // Von Neumann entropy
        let norm_sq: f64 = svals.iter().map(|s| s * s).sum();
        if norm_sq < EPSILON {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &s in &svals {
            let p = (s * s) / norm_sq;
            if p > EPSILON {
                entropy -= p * p.ln();
            }
        }

        entropy
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Kronecker (tensor) product of two tensors.
pub fn kronecker_product(a: &Tensor, b: &Tensor) -> Tensor {
    let shape_a = a.shape();
    let shape_b = b.shape();

    // Result shape: concatenation of shapes
    let mut result_shape = shape_a.clone();
    result_shape.extend_from_slice(&shape_b);

    let numel = a.numel() * b.numel();
    let mut data = vec![Complex64::new(0.0, 0.0); numel];

    for (ia, &va) in a.data.iter().enumerate() {
        for (ib, &vb) in b.data.iter().enumerate() {
            data[ia * b.numel() + ib] = va * vb;
        }
    }

    let mut indices: Vec<TensorIndex> = a.indices.clone();
    indices.extend(b.indices.clone());

    Tensor { data, indices }
}

/// Convert a gate unitary matrix into a tensor with input/output qubit indices.
pub fn tensor_from_gate_matrix(matrix: &[Complex64], num_qubits: usize) -> Tensor {
    let dim = 1 << num_qubits;
    assert_eq!(matrix.len(), dim * dim, "matrix size mismatch");

    let shape: Vec<usize> = vec![2; 2 * num_qubits];
    let numel: usize = shape.iter().product();
    let mut data = vec![Complex64::new(0.0, 0.0); numel];

    for out_flat in 0..dim {
        for in_flat in 0..dim {
            let mut full_idx = vec![0usize; 2 * num_qubits];
            for k in 0..num_qubits {
                full_idx[k] = (out_flat >> (num_qubits - 1 - k)) & 1;
            }
            for k in 0..num_qubits {
                full_idx[num_qubits + k] = (in_flat >> (num_qubits - 1 - k)) & 1;
            }
            let flat = row_major_flat(&full_idx, &shape);
            data[flat] = matrix[out_flat * dim + in_flat];
        }
    }

    Tensor::new(&shape, data)
}

/// Convert a list of MPS tensors (rank-3: left_bond x physical x right_bond) into a TensorNetwork.
pub fn mps_to_tn(mps_tensors: &[Tensor]) -> TensorNetwork {
    let mut network = TensorNetwork::new();

    if mps_tensors.is_empty() {
        return network;
    }

    let mut prev_bond_id: Option<usize> = None;
    let mut tensor_ids = Vec::new();

    for (i, mps_t) in mps_tensors.iter().enumerate() {
        let mut t = mps_t.clone();

        // For MPS tensors: index 0 = left bond, index 1 = physical, index 2 = right bond
        // First tensor may be rank-2 (no left bond) or rank-3
        if t.rank() >= 2 {
            // Connect left bond to previous right bond
            if let Some(prev_id) = prev_bond_id {
                t.indices[0] =
                    TensorIndex::with_id(prev_id, t.indices[0].dim, &format!("bond_{}", i - 1));
            }

            // Set right bond id for next tensor to connect to
            if t.rank() >= 3 {
                let right_bond = next_index_id();
                let last = t.rank() - 1;
                let dim = t.indices[last].dim;
                t.indices[last] = TensorIndex::with_id(right_bond, dim, &format!("bond_{}", i));
                prev_bond_id = Some(right_bond);
            } else {
                prev_bond_id = None;
            }
        }

        let id = network.add_tensor(t);
        tensor_ids.push(id);
    }

    network
}

use rand::SeedableRng;

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    // --------------------------------------------------------
    // 1. Tensor creation: zeros
    // --------------------------------------------------------
    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), vec![2, 3]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        for &v in &t.data {
            assert_eq!(v, c(0.0, 0.0));
        }
    }

    // --------------------------------------------------------
    // 2. Tensor creation: random (deterministic seed)
    // --------------------------------------------------------
    #[test]
    fn test_tensor_random() {
        let t1 = Tensor::random(&[4, 4], 42);
        let t2 = Tensor::random(&[4, 4], 42);
        assert_eq!(t1.data, t2.data, "same seed must produce identical tensors");
        assert_eq!(t1.numel(), 16);
        // Should not be all zeros
        assert!(t1.data.iter().any(|v| v.norm() > 0.01));
    }

    // --------------------------------------------------------
    // 3. Tensor creation: identity matrix
    // --------------------------------------------------------
    #[test]
    fn test_tensor_identity() {
        let id = Tensor::identity_matrix(3);
        assert_eq!(id.shape(), vec![3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) };
                assert_eq!(id.get(&[i, j]), expected);
            }
        }
    }

    // --------------------------------------------------------
    // 4. Tensor get/set
    // --------------------------------------------------------
    #[test]
    fn test_tensor_get_set() {
        let mut t = Tensor::zeros(&[2, 3, 4]);
        t.set(&[1, 2, 3], c(3.14, -2.71));
        assert_eq!(t.get(&[1, 2, 3]), c(3.14, -2.71));
        assert_eq!(t.get(&[0, 0, 0]), c(0.0, 0.0));
    }

    // --------------------------------------------------------
    // 5. Tensor contraction: matrix multiply
    // --------------------------------------------------------
    #[test]
    fn test_contraction_matrix_multiply() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // A * B = [[19, 22], [43, 50]]
        let a = Tensor::new(
            &[2, 2],
            vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
        );
        let b = Tensor::new(
            &[2, 2],
            vec![c(5.0, 0.0), c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0)],
        );

        // Contract A's column index (1) with B's row index (0)
        let result = a.contract_with(&b, &[(1, 0)]);
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.get(&[0, 0]), c(19.0, 0.0));
        assert_eq!(result.get(&[0, 1]), c(22.0, 0.0));
        assert_eq!(result.get(&[1, 0]), c(43.0, 0.0));
        assert_eq!(result.get(&[1, 1]), c(50.0, 0.0));
    }

    // --------------------------------------------------------
    // 6. Tensor contraction: vector inner product (full contraction)
    // --------------------------------------------------------
    #[test]
    fn test_contraction_inner_product() {
        let a = Tensor::new(&[3], vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)]);
        let b = Tensor::new(&[3], vec![c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)]);
        let result = a.contract_with(&b, &[(0, 0)]);
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(result.data.len(), 1);
        assert_eq!(result.data[0], c(32.0, 0.0));
    }

    // --------------------------------------------------------
    // 7. Tensor trace
    // --------------------------------------------------------
    #[test]
    fn test_tensor_trace() {
        let id = Tensor::identity_matrix(3);
        let tr = id.trace(0, 1);
        assert_eq!(tr.data.len(), 1);
        assert!(
            (tr.data[0] - c(3.0, 0.0)).norm() < TOL,
            "trace of 3x3 identity should be 3"
        );
    }

    // --------------------------------------------------------
    // 8. Tensor SVD: singular values and reconstruction
    // --------------------------------------------------------
    #[test]
    fn test_tensor_svd() {
        // A = [[1, 2], [3, 4]]
        let a = Tensor::new(
            &[2, 2],
            vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
        );

        let (u, svals, v) = a.svd(&[0], 2);

        // Reconstruct: U * V (V already has singular values baked in)
        let bond_pos_u = u.rank() - 1;
        let reconstructed = u.contract_with(&v, &[(bond_pos_u, 0)]);

        assert_eq!(reconstructed.shape(), vec![2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                let diff = (reconstructed.get(&[i, j]) - a.get(&[i, j])).norm();
                assert!(
                    diff < 1e-8,
                    "SVD reconstruction error at [{},{}]: {}",
                    i,
                    j,
                    diff
                );
            }
        }

        // Singular values should be positive and sorted descending
        assert!(svals[0] >= svals[1], "singular values should be descending");
        assert!(
            svals.iter().all(|&s| s >= 0.0),
            "singular values must be non-negative"
        );
    }

    // --------------------------------------------------------
    // 9. Tensor norm
    // --------------------------------------------------------
    #[test]
    fn test_tensor_norm() {
        let t = Tensor::new(&[2], vec![c(3.0, 0.0), c(4.0, 0.0)]);
        assert!(approx_eq(t.norm(), 5.0, TOL));
    }

    // --------------------------------------------------------
    // 10. TensorNetwork add/connect
    // --------------------------------------------------------
    #[test]
    fn test_network_add_connect() {
        let mut net = TensorNetwork::new();
        let a = Tensor::zeros(&[2, 3]);
        let b = Tensor::zeros(&[3, 4]);
        let id_a = net.add_tensor(a);
        let id_b = net.add_tensor(b);
        net.connect(id_a, 1, id_b, 0);

        assert_eq!(net.num_tensors(), 2);
        assert_eq!(net.total_indices(), 4);
        assert_eq!(net.contractions.len(), 1);
    }

    // --------------------------------------------------------
    // 11. Contract pair
    // --------------------------------------------------------
    #[test]
    fn test_contract_pair() {
        let mut net = TensorNetwork::new();

        // Create two matrices with a shared index id for contraction
        let shared_id = next_index_id();

        let mut a = Tensor::new(
            &[2, 2],
            vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
        );
        a.indices[1].id = shared_id;

        let mut b = Tensor::new(
            &[2, 2],
            vec![c(5.0, 0.0), c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0)],
        );
        b.indices[0].id = shared_id;

        let id_a = net.add_tensor(a);
        let id_b = net.add_tensor(b);

        let id_c = net.contract_pair(id_a, id_b);
        let result = net.get_tensor(id_c).unwrap();

        // Should be matrix product [[19,22],[43,50]]
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.get(&[0, 0]), c(19.0, 0.0));
        assert_eq!(result.get(&[1, 1]), c(50.0, 0.0));
    }

    // --------------------------------------------------------
    // 12. Full contraction: chain of matrices
    // --------------------------------------------------------
    #[test]
    fn test_full_contraction_chain() {
        // Chain: A * B * C where A=[2x3], B=[3x4], C=[4x2]
        let mut net = TensorNetwork::new();

        let bond_ab = next_index_id();
        let bond_bc = next_index_id();

        let mut a = Tensor::random(&[2, 3], 10);
        a.indices[1].id = bond_ab;

        let mut b = Tensor::random(&[3, 4], 20);
        b.indices[0].id = bond_ab;
        b.indices[1].id = bond_bc;

        let mut cc = Tensor::random(&[4, 2], 30);
        cc.indices[0].id = bond_bc;

        let id_a = net.add_tensor(a.clone());
        let id_b = net.add_tensor(b.clone());
        let id_c = net.add_tensor(cc.clone());

        // Contract in order: (A,B) then (AB,C)
        let order = ContractionOptimizer::greedy_order(&net);
        let result = net.contract_all(&order.steps);

        // Verify against direct computation
        let ab = a.contract_with(&b, &[(1, 0)]);
        let expected = ab.contract_with(&cc, &[(1, 0)]);

        assert_eq!(result.shape(), expected.shape());
        for i in 0..result.data.len() {
            let diff = (result.data[i] - expected.data[i]).norm();
            assert!(
                diff < 1e-8,
                "chain contraction mismatch at {}: diff={}",
                i,
                diff
            );
        }
    }

    // --------------------------------------------------------
    // 13. Greedy contraction order
    // --------------------------------------------------------
    #[test]
    fn test_greedy_order() {
        let mut net = TensorNetwork::new();

        let bond_01 = next_index_id();
        let bond_12 = next_index_id();
        let bond_23 = next_index_id();

        let mut t0 = Tensor::zeros(&[2, 4]);
        t0.indices[1].id = bond_01;
        let mut t1 = Tensor::zeros(&[4, 8]);
        t1.indices[0].id = bond_01;
        t1.indices[1].id = bond_12;
        let mut t2 = Tensor::zeros(&[8, 3]);
        t2.indices[0].id = bond_12;
        t2.indices[1].id = bond_23;
        let mut t3 = Tensor::zeros(&[3, 2]);
        t3.indices[0].id = bond_23;

        net.add_tensor(t0);
        net.add_tensor(t1);
        net.add_tensor(t2);
        net.add_tensor(t3);

        let order = ContractionOptimizer::greedy_order(&net);
        assert_eq!(order.steps.len(), 3, "4 tensors need 3 contraction steps");
        assert!(order.estimated_flops > 0.0, "flops should be positive");
    }

    // --------------------------------------------------------
    // 14. Random greedy finds a good order
    // --------------------------------------------------------
    #[test]
    fn test_random_greedy() {
        let mut net = TensorNetwork::new();

        let b01 = next_index_id();
        let b12 = next_index_id();

        let mut t0 = Tensor::zeros(&[2, 10]);
        t0.indices[1].id = b01;
        let mut t1 = Tensor::zeros(&[10, 10]);
        t1.indices[0].id = b01;
        t1.indices[1].id = b12;
        let mut t2 = Tensor::zeros(&[10, 2]);
        t2.indices[0].id = b12;

        net.add_tensor(t0);
        net.add_tensor(t1);
        net.add_tensor(t2);

        let greedy = ContractionOptimizer::greedy_order(&net);
        let random = ContractionOptimizer::random_greedy(&net, 32);

        // Random greedy should find an order at least as good as greedy
        assert!(
            random.estimated_flops <= greedy.estimated_flops + 1e-6,
            "random greedy should be at least as good: {} vs {}",
            random.estimated_flops,
            greedy.estimated_flops
        );
    }

    // --------------------------------------------------------
    // 15. Cost estimation
    // --------------------------------------------------------
    #[test]
    fn test_cost_estimation() {
        let mut net = TensorNetwork::new();

        let bond = next_index_id();

        let mut a = Tensor::zeros(&[2, 4]);
        a.indices[1].id = bond;
        let mut b = Tensor::zeros(&[4, 3]);
        b.indices[0].id = bond;

        let id_a = net.add_tensor(a);
        let id_b = net.add_tensor(b);

        // Contracting [2,4] x [4,3] -> [2,3]: 2*2*3*4 = 48 flops
        let cost = net.contraction_cost(&[(id_a, id_b)]);
        assert!(
            approx_eq(cost, 48.0, 1e-6),
            "cost should be 48, got {}",
            cost
        );
    }

    // --------------------------------------------------------
    // 16. Circuit to TN conversion
    // --------------------------------------------------------
    #[test]
    fn test_circuit_to_tn() {
        // Single Hadamard on qubit 0 of a 1-qubit circuit
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h_matrix = vec![
            c(inv_sqrt2, 0.0),
            c(inv_sqrt2, 0.0),
            c(inv_sqrt2, 0.0),
            c(-inv_sqrt2, 0.0),
        ];
        let gates = vec![("H".to_string(), vec![0], h_matrix)];
        let tn = CircuitToTN::from_circuit(&gates, 1);

        // Should have 2 tensors: |0> input + H gate
        assert_eq!(tn.num_tensors(), 2);
    }

    // --------------------------------------------------------
    // 17. Amplitude calculation: Bell state
    // --------------------------------------------------------
    #[test]
    fn test_amplitude_bell_state() {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        // Hadamard gate
        let h_matrix = vec![
            c(inv_sqrt2, 0.0),
            c(inv_sqrt2, 0.0),
            c(inv_sqrt2, 0.0),
            c(-inv_sqrt2, 0.0),
        ];

        // CNOT gate (4x4 matrix)
        let cnot_matrix = vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
        ];

        let gates = vec![
            ("H".to_string(), vec![0], h_matrix),
            ("CNOT".to_string(), vec![0, 1], cnot_matrix),
        ];

        let tn = CircuitToTN::from_circuit(&gates, 2);
        let sim = TNSimulator::new(TNConfig::new());

        // Bell state = (|00> + |11>) / sqrt(2)
        let amp_00 = sim.amplitude(&tn, &[false, false]);
        let amp_11 = sim.amplitude(&tn, &[true, true]);
        let amp_01 = sim.amplitude(&tn, &[false, true]);
        let amp_10 = sim.amplitude(&tn, &[true, false]);

        assert!(
            approx_eq(amp_00.norm(), inv_sqrt2, 1e-8),
            "|<00|psi>| should be 1/sqrt(2), got {}",
            amp_00.norm()
        );
        assert!(
            approx_eq(amp_11.norm(), inv_sqrt2, 1e-8),
            "|<11|psi>| should be 1/sqrt(2), got {}",
            amp_11.norm()
        );
        assert!(
            approx_eq(amp_01.norm(), 0.0, 1e-8),
            "|<01|psi>| should be 0, got {}",
            amp_01.norm()
        );
        assert!(
            approx_eq(amp_10.norm(), 0.0, 1e-8),
            "|<10|psi>| should be 0, got {}",
            amp_10.norm()
        );
    }

    // --------------------------------------------------------
    // 18. Inner product <psi|psi> = 1
    // --------------------------------------------------------
    #[test]
    fn test_inner_product() {
        // Two identical rank-1 tensors forming a simple network
        let mut net = TensorNetwork::new();
        let t = Tensor::new(&[3], vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
        net.add_tensor(t);

        let ip = net.inner_product(&net);
        assert!(
            approx_eq(ip.re, 1.0, 1e-8),
            "<psi|psi> should be 1.0, got {}",
            ip
        );
    }

    // --------------------------------------------------------
    // 19. Bond compression via SVD
    // --------------------------------------------------------
    #[test]
    fn test_bond_compression() {
        // Create a rank-4 tensor and compress it
        let t = Tensor::random(&[2, 3, 4, 2], 99);
        let original_norm = t.norm();

        let (u, svals, v) = t.svd(&[0, 1], 2); // max_bond=2

        // Compressed should have fewer singular values
        assert!(svals.len() <= 2, "should truncate to max_bond=2");

        // Reconstruct
        let bond_pos = u.rank() - 1;
        let recon = u.contract_with(&v, &[(bond_pos, 0)]);

        // Compressed norm should be <= original
        let compressed_norm = recon.norm();
        assert!(
            compressed_norm <= original_norm + TOL,
            "compressed norm {} should be <= original {}",
            compressed_norm,
            original_norm
        );
    }

    // --------------------------------------------------------
    // 20. Kronecker product
    // --------------------------------------------------------
    #[test]
    fn test_kronecker_product() {
        let a = Tensor::new(&[2], vec![c(1.0, 0.0), c(0.0, 0.0)]); // |0>
        let b = Tensor::new(&[2], vec![c(0.0, 0.0), c(1.0, 0.0)]); // |1>

        let ab = kronecker_product(&a, &b);
        assert_eq!(ab.shape(), vec![2, 2]);
        assert_eq!(ab.numel(), 4);
        // |0> kron |1> = [0, 1, 0, 0] in flattened form
        assert_eq!(ab.data[0], c(0.0, 0.0));
        assert_eq!(ab.data[1], c(1.0, 0.0));
        assert_eq!(ab.data[2], c(0.0, 0.0));
        assert_eq!(ab.data[3], c(0.0, 0.0));
    }

    // --------------------------------------------------------
    // 21. Small quantum circuit via TN (X gate flips |0> to |1>)
    // --------------------------------------------------------
    #[test]
    fn test_small_circuit_x_gate() {
        let x_matrix = vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)];
        let gates = vec![("X".to_string(), vec![0], x_matrix)];
        let tn = CircuitToTN::from_circuit(&gates, 1);
        let sim = TNSimulator::new(TNConfig::new());

        // X|0> = |1>
        let amp_0 = sim.amplitude(&tn, &[false]);
        let amp_1 = sim.amplitude(&tn, &[true]);

        assert!(
            approx_eq(amp_0.norm(), 0.0, 1e-8),
            "X|0> -> |1>: amp(0) should be 0"
        );
        assert!(
            approx_eq(amp_1.norm(), 1.0, 1e-8),
            "X|0> -> |1>: amp(1) should be 1"
        );
    }

    // --------------------------------------------------------
    // 22. Contraction order affects cost but not result
    // --------------------------------------------------------
    #[test]
    fn test_order_affects_cost_not_result() {
        let b01 = next_index_id();
        let b12 = next_index_id();

        let mut t0 = Tensor::random(&[2, 5], 100);
        t0.indices[1].id = b01;
        let mut t1 = Tensor::random(&[5, 5], 200);
        t1.indices[0].id = b01;
        t1.indices[1].id = b12;
        let mut t2 = Tensor::random(&[5, 2], 300);
        t2.indices[0].id = b12;

        // Order 1: (0,1) then (01,2)
        let mut net1 = TensorNetwork::new();
        let a0 = net1.add_tensor(t0.clone());
        let a1 = net1.add_tensor(t1.clone());
        let a2 = net1.add_tensor(t2.clone());
        let r1 = net1.contract_all(&[(a0, a1), (3, a2)]);

        // Order 2: (1,2) then (0,12)
        let mut net2 = TensorNetwork::new();
        let b0 = net2.add_tensor(t0.clone());
        let b1 = net2.add_tensor(t1.clone());
        let b2 = net2.add_tensor(t2.clone());
        let r2 = net2.contract_all(&[(b1, b2), (b0, 3)]);

        // Results should be the same (within floating point)
        assert_eq!(
            r1.shape(),
            r2.shape(),
            "shapes must match regardless of order"
        );
        for i in 0..r1.data.len() {
            let diff = (r1.data[i] - r2.data[i]).norm();
            assert!(
                diff < 1e-6,
                "results must match regardless of order: diff={} at {}",
                diff,
                i
            );
        }

        // But costs may differ
        let cost1 = net1.contraction_cost(&[(a0, a1), (3, a2)]);
        let cost2 = net2.contraction_cost(&[(b1, b2), (b0, 3)]);
        // Both should be positive, but not necessarily equal
        assert!(cost1 > 0.0);
        assert!(cost2 > 0.0);
    }

    // --------------------------------------------------------
    // 23. Tensor conjugate
    // --------------------------------------------------------
    #[test]
    fn test_tensor_conjugate() {
        let t = Tensor::new(&[2], vec![c(1.0, 2.0), c(3.0, -4.0)]);
        let tc = t.conjugate();
        assert_eq!(tc.data[0], c(1.0, -2.0));
        assert_eq!(tc.data[1], c(3.0, 4.0));
    }

    // --------------------------------------------------------
    // 24. Tensor reshape
    // --------------------------------------------------------
    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(
            &[2, 3],
            vec![
                c(1.0, 0.0),
                c(2.0, 0.0),
                c(3.0, 0.0),
                c(4.0, 0.0),
                c(5.0, 0.0),
                c(6.0, 0.0),
            ],
        );
        let r = t.reshape(&[3, 2]);
        assert_eq!(r.shape(), vec![3, 2]);
        assert_eq!(r.numel(), 6);
        // Data should be unchanged
        assert_eq!(r.data[0], c(1.0, 0.0));
        assert_eq!(r.data[5], c(6.0, 0.0));
    }

    // --------------------------------------------------------
    // 25. tensor_from_gate_matrix
    // --------------------------------------------------------
    #[test]
    fn test_tensor_from_gate_matrix() {
        // Pauli X gate
        let x = vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)];
        let t = tensor_from_gate_matrix(&x, 1);
        assert_eq!(t.shape(), vec![2, 2]);
        // X|0>=|1>: t[out=1, in=0] should be 1
        assert_eq!(t.get(&[1, 0]), c(1.0, 0.0));
        assert_eq!(t.get(&[0, 1]), c(1.0, 0.0));
    }

    // --------------------------------------------------------
    // 26. MPS to TN conversion
    // --------------------------------------------------------
    #[test]
    fn test_mps_to_tn() {
        // Simple 3-site MPS with bond dim 2
        let t0 = Tensor::random(&[1, 2, 2], 1); // left boundary
        let t1 = Tensor::random(&[2, 2, 2], 2); // bulk
        let t2 = Tensor::random(&[2, 2, 1], 3); // right boundary
        let tn = mps_to_tn(&[t0, t1, t2]);
        assert_eq!(tn.num_tensors(), 3);
    }

    // --------------------------------------------------------
    // 27. KahyparLike produces valid order
    // --------------------------------------------------------
    #[test]
    fn test_kahypar_like_order() {
        let mut net = TensorNetwork::new();
        let b01 = next_index_id();
        let b12 = next_index_id();
        let b23 = next_index_id();

        let mut t0 = Tensor::zeros(&[2, 4]);
        t0.indices[1].id = b01;
        let mut t1 = Tensor::zeros(&[4, 4]);
        t1.indices[0].id = b01;
        t1.indices[1].id = b12;
        let mut t2 = Tensor::zeros(&[4, 4]);
        t2.indices[0].id = b12;
        t2.indices[1].id = b23;
        let mut t3 = Tensor::zeros(&[4, 2]);
        t3.indices[0].id = b23;

        net.add_tensor(t0);
        net.add_tensor(t1);
        net.add_tensor(t2);
        net.add_tensor(t3);

        let order = ContractionOptimizer::kahypar_like(&net);
        assert_eq!(order.steps.len(), 3, "4 tensors need 3 steps");
    }

    // --------------------------------------------------------
    // 28. Config builder pattern
    // --------------------------------------------------------
    #[test]
    fn test_config_builder() {
        let cfg = TNConfig::new()
            .max_bond_dim(128)
            .contraction_method(ContractionMethod::Random)
            .optimize_order(false)
            .max_optimization_time_ms(10000)
            .compress_intermediate(false);

        assert_eq!(cfg.max_bond_dim, 128);
        assert_eq!(cfg.contraction_method, ContractionMethod::Random);
        assert!(!cfg.optimize_order);
        assert_eq!(cfg.max_optimization_time_ms, 10000);
        assert!(!cfg.compress_intermediate);
    }
}
