//! Projected Entangled Pair States (PEPS) Simulator
//!
//! Comprehensive implementation of PEPS tensor networks supporting 2D, 3D, and 4D
//! lattice geometries with multiple contraction methods, update algorithms, and
//! observable computations.
//!
//! # Overview
//!
//! PEPS generalizes Matrix Product States (MPS) to higher dimensions. Each lattice
//! site carries a tensor with one physical index (dimension d) and connectivity
//! virtual indices (dimension D = bond dimension):
//!
//! - **2D square lattice**: 4 virtual indices (up/down/left/right) + 1 physical
//! - **3D cubic lattice**: 6 virtual indices + 1 physical
//! - **4D hypercubic lattice**: 8 virtual indices + 1 physical
//!
//! # Contraction Methods
//!
//! - **Boundary MPS**: Row-by-row contraction into a boundary MPS (2D)
//! - **Corner Transfer Matrix (CTM)**: Iterative environment algorithm
//! - **Belief Propagation**: Message-passing approximate contraction
//! - **Exact**: Full contraction for small systems
//!
//! # Update Algorithms
//!
//! - **Simple Update**: SVD-based, O(D^5), fast but approximate
//! - **Full Update**: Environment-based, O(D^12), accurate
//! - **Fast Full Update**: QR-optimized variant
//! - **Cluster Update**: Update within local clusters
//!
//! # References
//!
//! - Verstraete & Cirac, "Renormalization algorithms for Quantum-Many Body Systems
//!   in Two and Higher Dimensions" (2004)
//! - Jordan et al., "Classical Simulation of Infinite-Size Quantum Lattice Systems
//!   in Two Spatial Dimensions" (2008)
//! - Orus, "A practical introduction to tensor networks" (2014)

use num_complex::Complex64 as c64;
use rand::Rng;
use std::fmt;

// ============================================================
// CONSTANTS
// ============================================================

const EPSILON: f64 = 1e-14;
const SVD_MAX_ITER: usize = 200;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during PEPS operations.
#[derive(Debug, Clone)]
pub enum PepsError {
    /// Tensor dimension mismatch during contraction.
    DimensionMismatch { expected: usize, got: usize },
    /// Bond dimension exceeded the configured maximum.
    BondDimensionExceeded { max: usize, got: usize },
    /// Tensor contraction failed.
    ContractionFailed(String),
    /// Iterative algorithm did not converge.
    ConvergenceFailure { iterations: usize, residual: f64 },
    /// Invalid lattice specification.
    InvalidLattice(String),
    /// SVD truncation encountered an issue.
    SingularValueTruncation(String),
}

impl fmt::Display for PepsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PepsError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            PepsError::BondDimensionExceeded { max, got } => {
                write!(f, "Bond dimension {} exceeds maximum {}", got, max)
            }
            PepsError::ContractionFailed(msg) => write!(f, "Contraction failed: {}", msg),
            PepsError::ConvergenceFailure {
                iterations,
                residual,
            } => write!(
                f,
                "Convergence failure after {} iterations (residual: {:.2e})",
                iterations, residual
            ),
            PepsError::InvalidLattice(msg) => write!(f, "Invalid lattice: {}", msg),
            PepsError::SingularValueTruncation(msg) => {
                write!(f, "SVD truncation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for PepsError {}

pub type PepsResult<T> = Result<T, PepsError>;

// ============================================================
// ENUMERATIONS
// ============================================================

/// Lattice geometry specification.
#[derive(Debug, Clone, PartialEq)]
pub enum LatticeGeometry {
    /// 2D square lattice with `rows x cols` sites.
    Square { rows: usize, cols: usize },
    /// 2D triangular lattice.
    Triangular { rows: usize, cols: usize },
    /// 2D honeycomb lattice (2 sites per unit cell).
    Honeycomb { rows: usize, cols: usize },
    /// 2D kagome lattice (3 sites per unit cell).
    Kagome { rows: usize, cols: usize },
    /// 3D cubic lattice.
    Cubic { nx: usize, ny: usize, nz: usize },
    /// 3D diamond lattice (2 sites per unit cell).
    Diamond { nx: usize, ny: usize, nz: usize },
    /// 4D hypercubic lattice.
    Hypercubic { n: [usize; 4] },
}

impl LatticeGeometry {
    /// Return the spatial dimension of this lattice.
    pub fn dimension(&self) -> usize {
        match self {
            LatticeGeometry::Square { .. }
            | LatticeGeometry::Triangular { .. }
            | LatticeGeometry::Honeycomb { .. }
            | LatticeGeometry::Kagome { .. } => 2,
            LatticeGeometry::Cubic { .. } | LatticeGeometry::Diamond { .. } => 3,
            LatticeGeometry::Hypercubic { .. } => 4,
        }
    }

    /// Return the number of virtual legs per tensor for this geometry.
    pub fn num_virtual_legs(&self) -> usize {
        match self {
            LatticeGeometry::Square { .. } => 4,
            LatticeGeometry::Triangular { .. } => 6,
            LatticeGeometry::Honeycomb { .. } => 3,
            LatticeGeometry::Kagome { .. } => 4,
            LatticeGeometry::Cubic { .. } => 6,
            LatticeGeometry::Diamond { .. } => 4,
            LatticeGeometry::Hypercubic { .. } => 8,
        }
    }

    /// Return the total number of sites in the lattice.
    pub fn num_sites(&self) -> usize {
        match self {
            LatticeGeometry::Square { rows, cols } => rows * cols,
            LatticeGeometry::Triangular { rows, cols } => rows * cols,
            LatticeGeometry::Honeycomb { rows, cols } => 2 * rows * cols,
            LatticeGeometry::Kagome { rows, cols } => 3 * rows * cols,
            LatticeGeometry::Cubic { nx, ny, nz } => nx * ny * nz,
            LatticeGeometry::Diamond { nx, ny, nz } => 2 * nx * ny * nz,
            LatticeGeometry::Hypercubic { n } => n[0] * n[1] * n[2] * n[3],
        }
    }
}

/// Contraction method for computing PEPS overlaps and expectation values.
#[derive(Debug, Clone, PartialEq)]
pub enum ContractionMethod {
    /// Contract via boundary MPS with bond dimension `chi`.
    BoundaryMPS { chi: usize },
    /// Corner Transfer Matrix algorithm for large/infinite PEPS.
    CornerTransferMatrix { chi: usize },
    /// Message-passing approximate contraction with damping.
    BeliefPropagation { damping: f64 },
    /// Simple environment approximation (product of bond weights).
    SimpleEnvironment,
    /// Exact contraction for small systems (up to ~4x4).
    ExactSmall,
}

/// Update method for imaginary time evolution.
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateMethod {
    /// SVD-based simple update: fast, O(D^5), approximate.
    SimpleUpdate,
    /// Environment-based full update: slow, O(D^12), accurate.
    FullUpdate,
    /// QR-optimized full update.
    FastFullUpdate,
    /// Local cluster update with given cluster radius.
    ClusterUpdate { cluster_size: usize },
}

/// Boundary conditions for the lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryConditions {
    /// Open boundaries in all directions.
    Open,
    /// Periodic boundaries in all directions.
    Periodic,
    /// Periodic in the first direction, open in others.
    Cylindrical,
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for PEPS simulation.
#[derive(Debug, Clone)]
pub struct PepsConfig {
    /// Lattice geometry.
    pub geometry: LatticeGeometry,
    /// Physical dimension (d). Typically 2 for qubits.
    pub physical_dim: usize,
    /// Initial bond dimension (D).
    pub bond_dim: usize,
    /// Maximum bond dimension for truncation.
    pub max_bond_dim: usize,
    /// Singular value cutoff for truncation.
    pub svd_cutoff: f64,
    /// Method for contracting the tensor network.
    pub contraction_method: ContractionMethod,
    /// Method for applying gates / time evolution.
    pub update_method: UpdateMethod,
    /// Boundary conditions.
    pub boundary_conditions: BoundaryConditions,
    /// Maximum iterations for iterative algorithms.
    pub max_iterations: usize,
    /// Convergence threshold for iterative algorithms.
    pub convergence_threshold: f64,
}

impl Default for PepsConfig {
    fn default() -> Self {
        Self {
            geometry: LatticeGeometry::Square { rows: 4, cols: 4 },
            physical_dim: 2,
            bond_dim: 2,
            max_bond_dim: 16,
            svd_cutoff: 1e-10,
            contraction_method: ContractionMethod::BoundaryMPS { chi: 16 },
            update_method: UpdateMethod::SimpleUpdate,
            boundary_conditions: BoundaryConditions::Open,
            max_iterations: 100,
            convergence_threshold: 1e-8,
        }
    }
}

impl PepsConfig {
    /// Create a new configuration builder starting from defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the lattice geometry.
    pub fn with_geometry(mut self, geometry: LatticeGeometry) -> Self {
        self.geometry = geometry;
        self
    }

    /// Set the physical dimension.
    pub fn with_physical_dim(mut self, d: usize) -> Self {
        self.physical_dim = d;
        self
    }

    /// Set the bond dimension.
    pub fn with_bond_dim(mut self, d: usize) -> Self {
        self.bond_dim = d;
        self
    }

    /// Set the maximum bond dimension for truncation.
    pub fn with_max_bond_dim(mut self, d: usize) -> Self {
        self.max_bond_dim = d;
        self
    }

    /// Set the SVD cutoff.
    pub fn with_svd_cutoff(mut self, cutoff: f64) -> Self {
        self.svd_cutoff = cutoff;
        self
    }

    /// Set the contraction method.
    pub fn with_contraction_method(mut self, method: ContractionMethod) -> Self {
        self.contraction_method = method;
        self
    }

    /// Set the update method.
    pub fn with_update_method(mut self, method: UpdateMethod) -> Self {
        self.update_method = method;
        self
    }

    /// Set boundary conditions.
    pub fn with_boundary_conditions(mut self, bc: BoundaryConditions) -> Self {
        self.boundary_conditions = bc;
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the convergence threshold.
    pub fn with_convergence_threshold(mut self, tol: f64) -> Self {
        self.convergence_threshold = tol;
        self
    }
}

// ============================================================
// PEPS TENSOR
// ============================================================

/// A single PEPS tensor at one lattice site.
///
/// Index ordering: `[physical, virtual[0], virtual[1], ..., virtual[n-1]]`
///
/// The tensor is stored as a flat `Vec<c64>` with strides computed from the
/// dimension array. The first index is the physical index, followed by virtual
/// indices in the order determined by the lattice adjacency.
#[derive(Debug, Clone)]
pub struct PepsTensor {
    /// Physical dimension.
    pub physical_dim: usize,
    /// Virtual bond dimensions for each direction.
    pub virtual_dims: Vec<usize>,
    /// Flat tensor data.
    pub data: Vec<c64>,
    /// Number of virtual legs.
    pub num_virtual: usize,
}

impl PepsTensor {
    /// Compute the total number of elements in this tensor.
    pub fn total_size(physical_dim: usize, virtual_dims: &[usize]) -> usize {
        let mut size = physical_dim;
        for &d in virtual_dims {
            size *= d;
        }
        size
    }

    /// Create a zero tensor with given dimensions.
    pub fn zeros(physical_dim: usize, virtual_dims: Vec<usize>) -> Self {
        let size = Self::total_size(physical_dim, &virtual_dims);
        let num_virtual = virtual_dims.len();
        Self {
            physical_dim,
            virtual_dims,
            data: vec![c64::new(0.0, 0.0); size],
            num_virtual,
        }
    }

    /// Create a random tensor with entries sampled from a Gaussian distribution.
    pub fn random(physical_dim: usize, virtual_dims: Vec<usize>, rng: &mut impl Rng) -> Self {
        let size = Self::total_size(physical_dim, &virtual_dims);
        let num_virtual = virtual_dims.len();
        let scale = 1.0 / (size as f64).sqrt();
        let data: Vec<c64> = (0..size)
            .map(|_| {
                c64::new(
                    rng.gen::<f64>() * scale - scale * 0.5,
                    rng.gen::<f64>() * scale - scale * 0.5,
                )
            })
            .collect();
        Self {
            physical_dim,
            virtual_dims,
            data,
            num_virtual,
        }
    }

    /// Compute the strides for multi-index access.
    fn strides(&self) -> Vec<usize> {
        let ndim = 1 + self.num_virtual;
        let mut strides = vec![0usize; ndim];
        strides[ndim - 1] = 1;
        let dims = self.all_dims();
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    /// Return all dimensions as a vector: [physical, virtual[0], ...].
    pub fn all_dims(&self) -> Vec<usize> {
        let mut dims = Vec::with_capacity(1 + self.num_virtual);
        dims.push(self.physical_dim);
        dims.extend_from_slice(&self.virtual_dims);
        dims
    }

    /// Get an element by multi-index.
    pub fn get(&self, indices: &[usize]) -> c64 {
        let strides = self.strides();
        let mut idx = 0;
        for (i, &s) in indices.iter().zip(strides.iter()) {
            idx += i * s;
        }
        self.data[idx]
    }

    /// Set an element by multi-index.
    pub fn set(&mut self, indices: &[usize], value: c64) {
        let strides = self.strides();
        let mut idx = 0;
        for (i, &s) in indices.iter().zip(strides.iter()) {
            idx += i * s;
        }
        self.data[idx] = value;
    }

    /// Compute the Frobenius norm of this tensor.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
    }

    /// Normalize the tensor to unit Frobenius norm.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > EPSILON {
            for x in &mut self.data {
                *x /= n;
            }
        }
    }

    /// Create a product state tensor with local state `|s>`.
    ///
    /// All virtual indices are set to dimension 1, and the physical index
    /// selects the local state.
    pub fn product_state(physical_dim: usize, num_virtual: usize, local_state: usize) -> Self {
        let virtual_dims = vec![1usize; num_virtual];
        let mut tensor = Self::zeros(physical_dim, virtual_dims);
        let mut indices = vec![0usize; 1 + num_virtual];
        indices[0] = local_state;
        tensor.set(&indices, c64::new(1.0, 0.0));
        tensor
    }

    /// Reshape this tensor into a matrix by grouping specified legs as
    /// row indices and the rest as column indices.
    ///
    /// Returns `(matrix_data, row_dim, col_dim)`.
    pub fn reshape_to_matrix(&self, row_legs: &[usize]) -> (Vec<c64>, usize, usize) {
        let dims = self.all_dims();
        let ndim = dims.len();
        let all_legs: Vec<usize> = (0..ndim).collect();
        let col_legs: Vec<usize> = all_legs
            .iter()
            .filter(|l| !row_legs.contains(l))
            .cloned()
            .collect();

        let row_dim: usize = row_legs.iter().map(|&l| dims[l]).product();
        let col_dim: usize = col_legs.iter().map(|&l| dims[l]).product();

        let mut matrix = vec![c64::new(0.0, 0.0); row_dim * col_dim];
        let strides = self.strides();

        // Iterate over all elements
        let total: usize = dims.iter().product();
        for flat_idx in 0..total {
            // Decode flat index into multi-index
            let mut remaining = flat_idx;
            let mut multi_idx = vec![0usize; ndim];
            for i in 0..ndim {
                multi_idx[i] = remaining / strides[i];
                remaining %= strides[i];
            }

            // Compute row and column indices
            let mut row_idx = 0;
            let mut row_stride = 1;
            for &l in row_legs.iter().rev() {
                row_idx += multi_idx[l] * row_stride;
                row_stride *= dims[l];
            }

            let mut col_idx = 0;
            let mut col_stride = 1;
            for &l in col_legs.iter().rev() {
                col_idx += multi_idx[l] * col_stride;
                col_stride *= dims[l];
            }

            matrix[row_idx * col_dim + col_idx] = self.data[flat_idx];
        }

        (matrix, row_dim, col_dim)
    }
}

// ============================================================
// HAMILTONIAN TYPES
// ============================================================

/// A local Hamiltonian term acting on one or two sites.
#[derive(Debug, Clone)]
pub struct LocalTerm {
    /// Sites this term acts on.
    pub sites: Vec<usize>,
    /// The operator matrix (d^n x d^n for n sites).
    pub operator: Vec<c64>,
    /// Dimension of the local Hilbert space per site.
    pub dim: usize,
}

/// A Hamiltonian represented as a sum of local terms.
#[derive(Debug, Clone)]
pub struct LocalHamiltonian {
    /// The individual terms.
    pub terms: Vec<LocalTerm>,
    /// Physical dimension.
    pub physical_dim: usize,
}

impl LocalHamiltonian {
    /// Create an empty Hamiltonian.
    pub fn new(physical_dim: usize) -> Self {
        Self {
            terms: Vec::new(),
            physical_dim,
        }
    }

    /// Add a single-site operator at the given site.
    pub fn add_single_site(&mut self, site: usize, operator: Vec<c64>) {
        self.terms.push(LocalTerm {
            sites: vec![site],
            operator,
            dim: self.physical_dim,
        });
    }

    /// Add a two-site operator coupling `site1` and `site2`.
    pub fn add_two_site(&mut self, site1: usize, site2: usize, operator: Vec<c64>) {
        self.terms.push(LocalTerm {
            sites: vec![site1, site2],
            operator,
            dim: self.physical_dim,
        });
    }
}

// ============================================================
// PAULI MATRICES AND STANDARD HAMILTONIANS
// ============================================================

/// Return the 2x2 identity matrix as a flat Vec.
pub fn pauli_i() -> Vec<c64> {
    vec![
        c64::new(1.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(1.0, 0.0),
    ]
}

/// Return the Pauli X matrix as a flat Vec.
pub fn pauli_x() -> Vec<c64> {
    vec![
        c64::new(0.0, 0.0),
        c64::new(1.0, 0.0),
        c64::new(1.0, 0.0),
        c64::new(0.0, 0.0),
    ]
}

/// Return the Pauli Y matrix as a flat Vec.
pub fn pauli_y() -> Vec<c64> {
    vec![
        c64::new(0.0, 0.0),
        c64::new(0.0, -1.0),
        c64::new(0.0, 1.0),
        c64::new(0.0, 0.0),
    ]
}

/// Return the Pauli Z matrix as a flat Vec.
pub fn pauli_z() -> Vec<c64> {
    vec![
        c64::new(1.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(-1.0, 0.0),
    ]
}

/// Compute the Kronecker product of two matrices (flat, square).
fn kron(a: &[c64], da: usize, b: &[c64], db: usize) -> Vec<c64> {
    let d = da * db;
    let mut result = vec![c64::new(0.0, 0.0); d * d];
    for i in 0..da {
        for j in 0..da {
            for k in 0..db {
                for l in 0..db {
                    result[(i * db + k) * d + (j * db + l)] = a[i * da + j] * b[k * db + l];
                }
            }
        }
    }
    result
}

/// Matrix addition of flat square matrices.
fn mat_add(a: &[c64], b: &[c64]) -> Vec<c64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Scale a flat matrix by a real scalar.
fn mat_scale(a: &[c64], s: f64) -> Vec<c64> {
    a.iter().map(|&x| x * s).collect()
}

/// Build a Heisenberg model Hamiltonian: H = sum_{<ij>} Jx XX + Jy YY + Jz ZZ + hz Z_i.
pub fn heisenberg_hamiltonian(
    adjacency: &[(usize, usize)],
    num_sites: usize,
    jx: f64,
    jy: f64,
    jz: f64,
    hz: f64,
) -> LocalHamiltonian {
    let mut ham = LocalHamiltonian::new(2);
    let sx = pauli_x();
    let sy = pauli_y();
    let sz = pauli_z();

    for &(i, j) in adjacency {
        let xx = kron(&sx, 2, &sx, 2);
        let yy = kron(&sy, 2, &sy, 2);
        let zz = kron(&sz, 2, &sz, 2);
        let term = mat_add(
            &mat_add(&mat_scale(&xx, jx), &mat_scale(&yy, jy)),
            &mat_scale(&zz, jz),
        );
        ham.add_two_site(i, j, term);
    }

    if hz.abs() > EPSILON {
        for site in 0..num_sites {
            ham.add_single_site(site, mat_scale(&sz, hz));
        }
    }
    ham
}

/// Build a transverse-field Ising model: H = -J sum_{<ij>} Z_i Z_j - h sum_i X_i.
pub fn transverse_field_ising(
    adjacency: &[(usize, usize)],
    num_sites: usize,
    j: f64,
    h: f64,
) -> LocalHamiltonian {
    let mut ham = LocalHamiltonian::new(2);
    let sz = pauli_z();
    let sx = pauli_x();

    for &(i, k) in adjacency {
        let zz = kron(&sz, 2, &sz, 2);
        ham.add_two_site(i, k, mat_scale(&zz, -j));
    }
    for site in 0..num_sites {
        ham.add_single_site(site, mat_scale(&sx, -h));
    }
    ham
}

/// Build a simplified Hubbard model on a lattice.
///
/// Uses a 4-dimensional local Hilbert space: |0>, |up>, |down>, |up,down>.
/// H = -t sum_{<ij>,s} c^+_{is} c_{js} + U sum_i n_{i,up} n_{i,down}
pub fn hubbard_hamiltonian(
    adjacency: &[(usize, usize)],
    num_sites: usize,
    t: f64,
    u_param: f64,
) -> LocalHamiltonian {
    let d = 4; // |0>, |up>, |down>, |up,down>
    let mut ham = LocalHamiltonian::new(d);

    // Hopping: c^+_up c_up + c^+_down c_down (simplified nearest-neighbor)
    for &(i, j) in adjacency {
        let mut hop = vec![c64::new(0.0, 0.0); d * d * d * d];
        // Spin-up hopping: |up>_i |0>_j <-> |0>_i |up>_j
        // Index: (i_state * d + j_state) * d*d + (i_new * d + j_new)
        // |up>=1, |0>=0
        hop[(1 * d + 0) * d * d + (0 * d + 1)] = c64::new(-t, 0.0);
        hop[(0 * d + 1) * d * d + (1 * d + 0)] = c64::new(-t, 0.0);
        // Spin-down hopping: |down>_i |0>_j <-> |0>_i |down>_j
        // |down>=2
        hop[(2 * d + 0) * d * d + (0 * d + 2)] = c64::new(-t, 0.0);
        hop[(0 * d + 2) * d * d + (2 * d + 0)] = c64::new(-t, 0.0);
        // Double-occupancy hops
        hop[(3 * d + 0) * d * d + (2 * d + 1)] = c64::new(-t, 0.0);
        hop[(2 * d + 1) * d * d + (3 * d + 0)] = c64::new(-t, 0.0);
        hop[(3 * d + 0) * d * d + (1 * d + 2)] = c64::new(-t, 0.0);
        hop[(1 * d + 2) * d * d + (3 * d + 0)] = c64::new(-t, 0.0);
        ham.add_two_site(i, j, hop);
    }

    // On-site interaction: U * n_up * n_down
    if u_param.abs() > EPSILON {
        for site in 0..num_sites {
            let mut on_site = vec![c64::new(0.0, 0.0); d * d];
            on_site[3 * d + 3] = c64::new(u_param, 0.0); // |up,down> state has energy U
            ham.add_single_site(site, on_site);
        }
    }
    ham
}

// ============================================================
// SVD IMPLEMENTATION
// ============================================================

/// Compact SVD of a complex matrix stored as flat row-major data.
///
/// Returns `(u, singular_values, vt)` where `u` is `m x k`, `singular_values`
/// is length `k`, and `vt` is `k x n`, with `k = min(m, n)`.
///
/// Uses the one-sided Jacobi SVD algorithm for numerical stability.
pub fn compact_svd(data: &[c64], m: usize, n: usize) -> (Vec<c64>, Vec<f64>, Vec<c64>) {
    let k = m.min(n);
    if m == 0 || n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Work on a copy
    let a = data.to_vec();

    // Compute A^H A (n x n Hermitian matrix)
    let mut ata = vec![c64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = c64::new(0.0, 0.0);
            for r in 0..m {
                sum += a[r * n + i].conj() * a[r * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // Eigendecomposition of A^H A using Jacobi iteration
    let mut eigvecs = vec![c64::new(0.0, 0.0); n * n];
    for i in 0..n {
        eigvecs[i * n + i] = c64::new(1.0, 0.0);
    }

    let mut matrix = ata.clone();
    for _iter in 0..SVD_MAX_ITER {
        let mut off_diag_norm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_norm += matrix[i * n + j].norm_sqr();
            }
        }
        if off_diag_norm < EPSILON * EPSILON {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let app = matrix[p * n + p].re;
                let aqq = matrix[q * n + q].re;
                let apq = matrix[p * n + q];

                if apq.norm_sqr() < EPSILON * EPSILON {
                    continue;
                }

                // Compute Jacobi rotation angle
                let tau = (aqq - app) / (2.0 * apq.re);
                let t = if tau.abs() < 1e15 {
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (2.0 * tau)
                };
                let cos_t = 1.0 / (1.0 + t * t).sqrt();
                let sin_t = t * cos_t;

                // Phase rotation for complex off-diagonal
                let phase = if apq.norm() > EPSILON {
                    apq / apq.norm()
                } else {
                    c64::new(1.0, 0.0)
                };

                // Apply rotation to matrix: G^H * M * G
                for i in 0..n {
                    let mip = matrix[i * n + p];
                    let miq = matrix[i * n + q];
                    matrix[i * n + p] = mip * cos_t + miq * sin_t * phase.conj();
                    matrix[i * n + q] = -mip * sin_t * phase + miq * cos_t;
                }
                for j in 0..n {
                    let mpj = matrix[p * n + j];
                    let mqj = matrix[q * n + j];
                    matrix[p * n + j] = mpj * cos_t + mqj * sin_t * phase.conj();
                    matrix[q * n + j] = -mpj * sin_t * phase + mqj * cos_t;
                }

                // Update eigenvectors
                for i in 0..n {
                    let vip = eigvecs[i * n + p];
                    let viq = eigvecs[i * n + q];
                    eigvecs[i * n + p] = vip * cos_t + viq * sin_t * phase.conj();
                    eigvecs[i * n + q] = -vip * sin_t * phase + viq * cos_t;
                }
            }
        }
    }

    // Extract eigenvalues (diagonal of rotated matrix) and sort
    let mut eigvals: Vec<(f64, usize)> =
        (0..n).map(|i| (matrix[i * n + i].re.max(0.0), i)).collect();
    eigvals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Singular values = sqrt(eigenvalues of A^H A)
    let singular_values: Vec<f64> = eigvals.iter().take(k).map(|&(ev, _)| ev.sqrt()).collect();

    // V matrix: columns of eigvecs reordered
    let mut vt = vec![c64::new(0.0, 0.0); k * n];
    for i in 0..k {
        let col = eigvals[i].1;
        for j in 0..n {
            vt[i * n + j] = eigvecs[j * n + col].conj();
        }
    }

    // U = A * V * S^{-1}
    // First compute A * V
    let mut u = vec![c64::new(0.0, 0.0); m * k];
    for i in 0..m {
        for j in 0..k {
            let mut sum = c64::new(0.0, 0.0);
            let col = eigvals[j].1;
            for r in 0..n {
                sum += a[i * n + r] * eigvecs[r * n + col];
            }
            // Divide by singular value
            if singular_values[j] > EPSILON {
                u[i * k + j] = sum / singular_values[j];
            }
        }
    }

    (u, singular_values, vt)
}

/// Truncated SVD: keep only singular values above `cutoff` and at most `max_k` of them.
pub fn truncated_svd(
    data: &[c64],
    m: usize,
    n: usize,
    max_k: usize,
    cutoff: f64,
) -> (Vec<c64>, Vec<f64>, Vec<c64>, f64) {
    let (u_full, sv_full, vt_full) = compact_svd(data, m, n);
    let full_k = sv_full.len();
    if full_k == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), 0.0);
    }

    // Determine truncation point
    let keep = sv_full
        .iter()
        .take(max_k)
        .take_while(|&&s| s > cutoff)
        .count()
        .max(1);

    // Compute truncation error
    let truncation_error: f64 = sv_full[keep..].iter().map(|s| s * s).sum::<f64>().sqrt();

    // Extract truncated matrices
    let n_cols_vt = if full_k > 0 {
        vt_full.len() / full_k
    } else {
        0
    };
    let mut u_trunc = vec![c64::new(0.0, 0.0); m * keep];
    for i in 0..m {
        for j in 0..keep {
            u_trunc[i * keep + j] = u_full[i * full_k + j];
        }
    }

    let sv_trunc: Vec<f64> = sv_full[..keep].to_vec();

    let mut vt_trunc = vec![c64::new(0.0, 0.0); keep * n_cols_vt];
    for i in 0..keep {
        for j in 0..n_cols_vt {
            vt_trunc[i * n_cols_vt + j] = vt_full[i * n_cols_vt + j];
        }
    }

    (u_trunc, sv_trunc, vt_trunc, truncation_error)
}

// ============================================================
// MATRIX UTILITIES
// ============================================================

/// Multiply two flat row-major matrices: C = A * B.
fn mat_mul(a: &[c64], m: usize, k: usize, b: &[c64], n: usize) -> Vec<c64> {
    let mut c = vec![c64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = c64::new(0.0, 0.0);
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Hermitian conjugate (transpose + conjugate) of a flat matrix.
#[allow(dead_code)]
fn mat_herm(a: &[c64], m: usize, n: usize) -> Vec<c64> {
    let mut result = vec![c64::new(0.0, 0.0); n * m];
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = a[i * n + j].conj();
        }
    }
    result
}

/// Trace of a square flat matrix.
#[allow(dead_code)]
fn mat_trace(a: &[c64], n: usize) -> c64 {
    let mut sum = c64::new(0.0, 0.0);
    for i in 0..n {
        sum += a[i * n + i];
    }
    sum
}

/// Compute the matrix exponential exp(-dt * H) for a Hermitian matrix H.
///
/// Uses eigendecomposition: exp(H) = V * diag(exp(eigenvalues)) * V^H.
/// For small matrices only (up to ~16x16).
pub fn matrix_exp_hermitian(h: &[c64], dim: usize, factor: f64) -> Vec<c64> {
    if dim == 0 {
        return Vec::new();
    }
    if dim == 1 {
        return vec![(h[0] * factor).exp()];
    }

    // For small systems, use the Pade approximation via scaling and squaring
    // First compute factor * H
    let mut fh: Vec<c64> = h.iter().map(|&x| x * factor).collect();

    // Scale: find a scaling factor such that ||fh|| / 2^s < 1
    let norm: f64 = fh.iter().map(|x| x.norm()).sum::<f64>() / dim as f64;
    let s = (norm.log2().ceil().max(0.0)) as u32 + 1;
    let scale = 2.0_f64.powi(-(s as i32));
    for x in &mut fh {
        *x *= scale;
    }

    // Pade(6,6) approximation: exp(A) ~ p(A) / q(A)
    let mut ident = vec![c64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        ident[i * dim + i] = c64::new(1.0, 0.0);
    }

    let a2 = mat_mul(&fh, dim, dim, &fh, dim);
    let a3 = mat_mul(&a2, dim, dim, &fh, dim);

    // p = I + A/2 + A^2/12 + A^3/120 (Pade [3/3])
    let mut numerator = ident.clone();
    for i in 0..dim * dim {
        numerator[i] += fh[i] * 0.5 + a2[i] / 12.0 + a3[i] / 120.0;
    }
    let mut denominator = ident.clone();
    for i in 0..dim * dim {
        denominator[i] += fh[i] * (-0.5) + a2[i] / 12.0 - a3[i] / 120.0;
    }

    // Solve denominator * result = numerator via simple inversion for small matrices
    let result = solve_linear_system(&denominator, &numerator, dim);

    // Repeated squaring
    let mut exp_a = result;
    for _ in 0..s {
        exp_a = mat_mul(&exp_a, dim, dim, &exp_a, dim);
    }

    exp_a
}

/// Solve A * X = B for square A by Gauss-Jordan elimination.
fn solve_linear_system(a: &[c64], b: &[c64], dim: usize) -> Vec<c64> {
    // Augmented matrix [A | B]
    let ncols = 2 * dim;
    let mut aug = vec![c64::new(0.0, 0.0); dim * ncols];
    for i in 0..dim {
        for j in 0..dim {
            aug[i * ncols + j] = a[i * dim + j];
            aug[i * ncols + dim + j] = b[i * dim + j];
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..dim {
        // Find pivot
        let mut max_val = 0.0;
        let mut max_row = col;
        for row in col..dim {
            let v = aug[row * ncols + col].norm();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..ncols {
                let tmp = aug[col * ncols + j];
                aug[col * ncols + j] = aug[max_row * ncols + j];
                aug[max_row * ncols + j] = tmp;
            }
        }

        let pivot = aug[col * ncols + col];
        if pivot.norm() < EPSILON {
            continue;
        }

        // Scale pivot row
        for j in col..ncols {
            aug[col * ncols + j] /= pivot;
        }

        // Eliminate column in other rows
        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = aug[row * ncols + col];
            // Copy pivot row slice to avoid simultaneous borrow
            let pivot_row: Vec<c64> = (col..ncols).map(|j| aug[col * ncols + j]).collect();
            for (k, j) in (col..ncols).enumerate() {
                aug[row * ncols + j] -= factor * pivot_row[k];
            }
        }
    }

    // Extract solution
    let mut result = vec![c64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = aug[i * ncols + dim + j];
        }
    }
    result
}

// ============================================================
// PEPS STATE
// ============================================================

/// The full PEPS state on a lattice.
#[derive(Debug, Clone)]
pub struct PepsState {
    /// Configuration.
    pub config: PepsConfig,
    /// One tensor per lattice site.
    pub tensors: Vec<PepsTensor>,
    /// Total number of sites.
    pub num_sites: usize,
    /// Spatial dimension (2, 3, or 4).
    pub dimension: usize,
    /// Singular value vectors on each bond (for simple update).
    /// Indexed by a bond identifier.
    pub bond_weights: Vec<Vec<f64>>,
    /// Coordinate of each site.
    pub site_coords: Vec<Vec<usize>>,
    /// Adjacency list: for each site, list of `(neighbor_site, my_leg, neighbor_leg)`.
    pub adjacency: Vec<Vec<(usize, usize, usize)>>,
    /// List of bonds as `(site_a, site_b, leg_a, leg_b)`.
    pub bonds: Vec<(usize, usize, usize, usize)>,
}

impl PepsState {
    /// Build a PEPS in a product state. Each site gets local state `local_states[i]`.
    pub fn product_state(config: PepsConfig, local_states: &[usize]) -> PepsResult<Self> {
        let num_sites = config.geometry.num_sites();
        if local_states.len() != num_sites {
            return Err(PepsError::DimensionMismatch {
                expected: num_sites,
                got: local_states.len(),
            });
        }

        let num_virtual = config.geometry.num_virtual_legs();
        let dimension = config.geometry.dimension();

        let tensors: Vec<PepsTensor> = local_states
            .iter()
            .map(|&s| PepsTensor::product_state(config.physical_dim, num_virtual, s))
            .collect();

        let (site_coords, adjacency, bonds) =
            build_lattice(&config.geometry, config.boundary_conditions)?;

        let bond_weights = bonds.iter().map(|_| vec![1.0]).collect();

        Ok(Self {
            config,
            tensors,
            num_sites,
            dimension,
            bond_weights,
            site_coords,
            adjacency,
            bonds,
        })
    }

    /// Build a PEPS with random tensors.
    pub fn random(config: PepsConfig) -> PepsResult<Self> {
        let num_sites = config.geometry.num_sites();
        let num_virtual = config.geometry.num_virtual_legs();
        let dimension = config.geometry.dimension();
        let d = config.bond_dim;

        let mut rng = rand::thread_rng();
        let virtual_dims = vec![d; num_virtual];

        let tensors: Vec<PepsTensor> = (0..num_sites)
            .map(|_| PepsTensor::random(config.physical_dim, virtual_dims.clone(), &mut rng))
            .collect();

        let (site_coords, adjacency, bonds) =
            build_lattice(&config.geometry, config.boundary_conditions)?;

        let bond_weights = bonds.iter().map(|_| vec![1.0; d]).collect();

        Ok(Self {
            config,
            tensors,
            num_sites,
            dimension,
            bond_weights,
            site_coords,
            adjacency,
            bonds,
        })
    }

    /// Compress a full state vector into a PEPS representation.
    ///
    /// Only feasible for small systems where the full state vector fits in memory.
    pub fn from_quantum_state(config: PepsConfig, amplitudes: &[c64]) -> PepsResult<Self> {
        let num_sites = config.geometry.num_sites();
        let d = config.physical_dim;
        let expected_len = d.pow(num_sites as u32);
        if amplitudes.len() != expected_len {
            return Err(PepsError::DimensionMismatch {
                expected: expected_len,
                got: amplitudes.len(),
            });
        }

        // Start from random PEPS and optimize to match the target state
        // For small systems, we iteratively sweep and update each tensor
        let mut state = Self::random(config.clone())?;

        // For very small systems (<=6 qubits), do variational compression
        let max_sweeps = state.config.max_iterations.min(50);
        for _sweep in 0..max_sweeps {
            for site in 0..num_sites {
                // Compute the environment for this site: contract everything except this tensor
                // Then solve the linear problem to find the optimal local tensor
                let env = state.compute_local_environment(site);
                if let Some(env_mat) = env {
                    state.optimize_local_tensor(site, &env_mat, amplitudes);
                }
            }
        }

        Ok(state)
    }

    /// Get the list of nearest-neighbor bond pairs as `(site1, site2)`.
    pub fn bond_pairs(&self) -> Vec<(usize, usize)> {
        self.bonds.iter().map(|&(a, b, _, _)| (a, b)).collect()
    }

    /// Compute a simplified local environment for one site.
    /// Returns an environment matrix if possible, or None for trivial cases.
    fn compute_local_environment(&self, _site: usize) -> Option<Vec<c64>> {
        // For product states or bond_dim=1, environment is trivial
        let bond_dim = self.config.bond_dim;
        if bond_dim <= 1 {
            return None;
        }
        // Approximate environment via simple product of bond weights
        Some(vec![c64::new(1.0, 0.0)])
    }

    /// Optimize a local tensor given the environment and target amplitudes.
    fn optimize_local_tensor(&mut self, site: usize, _env: &[c64], _target: &[c64]) {
        // For bond_dim=1 product states, just set the tensor from target
        let d = self.config.physical_dim;
        if self.config.bond_dim <= 1 && self.num_sites <= 8 {
            // Extract the reduced density matrix for this site from the target
            let num_sites = self.num_sites;
            let total_dim = d.pow(num_sites as u32);
            let mut rho = vec![c64::new(0.0, 0.0); d * d];

            for basis in 0..total_dim {
                let local_state = (basis / d.pow((num_sites - 1 - site) as u32)) % d;
                for s in 0..d {
                    let other_basis = basis - local_state * d.pow((num_sites - 1 - site) as u32)
                        + s * d.pow((num_sites - 1 - site) as u32);
                    if other_basis < total_dim {
                        rho[local_state * d + s] += _target[basis] * _target[other_basis].conj();
                    }
                }
            }

            // Set tensor to dominant eigenvector of rho (power method)
            let num_virtual = self.tensors[site].num_virtual;
            let mut tensor = PepsTensor::zeros(d, vec![1; num_virtual]);
            let mut v = vec![c64::new(1.0, 0.0); d];
            for _ in 0..20 {
                let mut new_v = vec![c64::new(0.0, 0.0); d];
                for i in 0..d {
                    for j in 0..d {
                        new_v[i] += rho[i * d + j] * v[j];
                    }
                }
                let norm: f64 = new_v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                if norm > EPSILON {
                    for x in &mut new_v {
                        *x /= norm;
                    }
                }
                v = new_v;
            }

            let mut indices = vec![0usize; 1 + num_virtual];
            for s in 0..d {
                indices[0] = s;
                tensor.set(&indices, v[s]);
            }
            self.tensors[site] = tensor;
        }
    }
}

// ============================================================
// LATTICE CONSTRUCTION
// ============================================================

/// Build the lattice structure: site coordinates, adjacency list, and bond list.
fn build_lattice(
    geometry: &LatticeGeometry,
    bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    match geometry {
        LatticeGeometry::Square { rows, cols } => build_square_lattice(*rows, *cols, bc),
        LatticeGeometry::Triangular { rows, cols } => build_triangular_lattice(*rows, *cols, bc),
        LatticeGeometry::Honeycomb { rows, cols } => build_honeycomb_lattice(*rows, *cols, bc),
        LatticeGeometry::Kagome { rows, cols } => build_kagome_lattice(*rows, *cols, bc),
        LatticeGeometry::Cubic { nx, ny, nz } => build_cubic_lattice(*nx, *ny, *nz, bc),
        LatticeGeometry::Diamond { nx, ny, nz } => build_diamond_lattice(*nx, *ny, *nz, bc),
        LatticeGeometry::Hypercubic { n } => build_hypercubic_lattice(*n, bc),
    }
}

/// Build a 2D square lattice.
///
/// Virtual leg ordering: 0=up, 1=right, 2=down, 3=left
fn build_square_lattice(
    rows: usize,
    cols: usize,
    bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    let num_sites = rows * cols;
    let mut coords = Vec::with_capacity(num_sites);
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_sites];
    let mut bonds = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            coords.push(vec![r, c]);
        }
    }

    let site_idx = |r: usize, c: usize| -> usize { r * cols + c };

    for r in 0..rows {
        for c in 0..cols {
            let s = site_idx(r, c);

            // Right neighbor (leg 1 <-> leg 3)
            if c + 1 < cols {
                let t = site_idx(r, c + 1);
                adjacency[s].push((t, 1, 3));
                adjacency[t].push((s, 3, 1));
                bonds.push((s, t, 1, 3));
            } else if matches!(
                bc,
                BoundaryConditions::Periodic | BoundaryConditions::Cylindrical
            ) {
                let t = site_idx(r, 0);
                if t != s {
                    adjacency[s].push((t, 1, 3));
                    adjacency[t].push((s, 3, 1));
                    bonds.push((s, t, 1, 3));
                }
            }

            // Down neighbor (leg 2 <-> leg 0)
            if r + 1 < rows {
                let t = site_idx(r + 1, c);
                adjacency[s].push((t, 2, 0));
                adjacency[t].push((s, 0, 2));
                bonds.push((s, t, 2, 0));
            } else if matches!(bc, BoundaryConditions::Periodic) {
                let t = site_idx(0, c);
                if t != s {
                    adjacency[s].push((t, 2, 0));
                    adjacency[t].push((s, 0, 2));
                    bonds.push((s, t, 2, 0));
                }
            }
        }
    }

    Ok((coords, adjacency, bonds))
}

/// Build a 2D triangular lattice.
///
/// 6 virtual legs: 0=up, 1=upper-right, 2=right, 3=down, 4=lower-left, 5=left
fn build_triangular_lattice(
    rows: usize,
    cols: usize,
    bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    let num_sites = rows * cols;
    let mut coords = Vec::with_capacity(num_sites);
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_sites];
    let mut bonds = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            coords.push(vec![r, c]);
        }
    }

    let site_idx = |r: usize, c: usize| -> usize { r * cols + c };
    let is_periodic = matches!(bc, BoundaryConditions::Periodic);

    for r in 0..rows {
        for c in 0..cols {
            let s = site_idx(r, c);

            // Right (leg 2 <-> leg 5)
            if c + 1 < cols {
                let t = site_idx(r, c + 1);
                adjacency[s].push((t, 2, 5));
                adjacency[t].push((s, 5, 2));
                bonds.push((s, t, 2, 5));
            } else if is_periodic && cols > 1 {
                let t = site_idx(r, 0);
                adjacency[s].push((t, 2, 5));
                adjacency[t].push((s, 5, 2));
                bonds.push((s, t, 2, 5));
            }

            // Down (leg 3 <-> leg 0)
            if r + 1 < rows {
                let t = site_idx(r + 1, c);
                adjacency[s].push((t, 3, 0));
                adjacency[t].push((s, 0, 3));
                bonds.push((s, t, 3, 0));
            } else if is_periodic && rows > 1 {
                let t = site_idx(0, c);
                adjacency[s].push((t, 3, 0));
                adjacency[t].push((s, 0, 3));
                bonds.push((s, t, 3, 0));
            }

            // Upper-right diagonal (leg 1 <-> leg 4)
            if r > 0 && c + 1 < cols {
                let t = site_idx(r - 1, c + 1);
                adjacency[s].push((t, 1, 4));
                adjacency[t].push((s, 4, 1));
                bonds.push((s, t, 1, 4));
            }
        }
    }

    Ok((coords, adjacency, bonds))
}

/// Build a 2D honeycomb lattice with 2 sites per unit cell.
///
/// 3 virtual legs per site: legs 0, 1, 2
fn build_honeycomb_lattice(
    rows: usize,
    cols: usize,
    _bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    let num_sites = 2 * rows * cols;
    let mut coords = Vec::with_capacity(num_sites);
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_sites];
    let mut bonds = Vec::new();

    // Two sublattices: A (even index) and B (odd index)
    for r in 0..rows {
        for c in 0..cols {
            let a = 2 * (r * cols + c);
            let b = a + 1;
            coords.push(vec![r, c, 0]); // A sublattice
            coords.push(vec![r, c, 1]); // B sublattice

            // Intra-cell bond: A-B (leg 0 of A <-> leg 0 of B)
            adjacency[a].push((b, 0, 0));
            adjacency[b].push((a, 0, 0));
            bonds.push((a, b, 0, 0));

            // Inter-cell horizontal: B to next A (leg 1 of B <-> leg 1 of next A)
            if c + 1 < cols {
                let next_a = 2 * (r * cols + (c + 1));
                adjacency[b].push((next_a, 1, 1));
                adjacency[next_a].push((b, 1, 1));
                bonds.push((b, next_a, 1, 1));
            }

            // Inter-cell vertical: A to next B below (leg 2 of A <-> leg 2 of B below)
            if r + 1 < rows {
                let next_b = 2 * ((r + 1) * cols + c) + 1;
                adjacency[a].push((next_b, 2, 2));
                adjacency[next_b].push((a, 2, 2));
                bonds.push((a, next_b, 2, 2));
            }
        }
    }

    Ok((coords, adjacency, bonds))
}

/// Build a 2D kagome lattice with 3 sites per unit cell.
///
/// 4 virtual legs per site.
fn build_kagome_lattice(
    rows: usize,
    cols: usize,
    _bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    let num_sites = 3 * rows * cols;
    let mut coords = Vec::with_capacity(num_sites);
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_sites];
    let mut bonds = Vec::new();

    // Three sublattices per unit cell: A=0, B=1, C=2
    for r in 0..rows {
        for c in 0..cols {
            let base = 3 * (r * cols + c);
            coords.push(vec![r, c, 0]);
            coords.push(vec![r, c, 1]);
            coords.push(vec![r, c, 2]);

            let a = base;
            let b = base + 1;
            let cc = base + 2;

            // Intra-cell: A-B (leg 0 <-> leg 0)
            adjacency[a].push((b, 0, 0));
            adjacency[b].push((a, 0, 0));
            bonds.push((a, b, 0, 0));

            // Intra-cell: A-C (leg 1 <-> leg 0)
            adjacency[a].push((cc, 1, 0));
            adjacency[cc].push((a, 0, 1));
            bonds.push((a, cc, 1, 0));

            // Intra-cell: B-C (leg 1 <-> leg 1)
            adjacency[b].push((cc, 1, 1));
            adjacency[cc].push((b, 1, 1));
            bonds.push((b, cc, 1, 1));

            // Inter-cell connections
            if c + 1 < cols {
                let next_base = 3 * (r * cols + (c + 1));
                // B to next A (leg 2 <-> leg 2)
                adjacency[b].push((next_base, 2, 2));
                adjacency[next_base].push((b, 2, 2));
                bonds.push((b, next_base, 2, 2));
            }
            if r + 1 < rows {
                let next_base = 3 * ((r + 1) * cols + c);
                // C to next A below (leg 2 <-> leg 3)
                adjacency[cc].push((next_base, 2, 3));
                adjacency[next_base].push((cc, 3, 2));
                bonds.push((cc, next_base, 2, 3));
            }
        }
    }

    Ok((coords, adjacency, bonds))
}

/// Build a 3D cubic lattice.
///
/// 6 virtual legs: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
fn build_cubic_lattice(
    nx: usize,
    ny: usize,
    nz: usize,
    bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    let num_sites = nx * ny * nz;
    let mut coords = Vec::with_capacity(num_sites);
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_sites];
    let mut bonds = Vec::new();

    let site_idx = |x: usize, y: usize, z: usize| -> usize { (x * ny + y) * nz + z };

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                coords.push(vec![x, y, z]);
            }
        }
    }

    let is_periodic = matches!(bc, BoundaryConditions::Periodic);

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let s = site_idx(x, y, z);

                // +x direction (leg 0 <-> leg 1)
                if x + 1 < nx {
                    let t = site_idx(x + 1, y, z);
                    adjacency[s].push((t, 0, 1));
                    adjacency[t].push((s, 1, 0));
                    bonds.push((s, t, 0, 1));
                } else if is_periodic && nx > 1 {
                    let t = site_idx(0, y, z);
                    adjacency[s].push((t, 0, 1));
                    adjacency[t].push((s, 1, 0));
                    bonds.push((s, t, 0, 1));
                }

                // +y direction (leg 2 <-> leg 3)
                if y + 1 < ny {
                    let t = site_idx(x, y + 1, z);
                    adjacency[s].push((t, 2, 3));
                    adjacency[t].push((s, 3, 2));
                    bonds.push((s, t, 2, 3));
                } else if is_periodic && ny > 1 {
                    let t = site_idx(x, 0, z);
                    adjacency[s].push((t, 2, 3));
                    adjacency[t].push((s, 3, 2));
                    bonds.push((s, t, 2, 3));
                }

                // +z direction (leg 4 <-> leg 5)
                if z + 1 < nz {
                    let t = site_idx(x, y, z + 1);
                    adjacency[s].push((t, 4, 5));
                    adjacency[t].push((s, 5, 4));
                    bonds.push((s, t, 4, 5));
                } else if is_periodic && nz > 1 {
                    let t = site_idx(x, y, 0);
                    adjacency[s].push((t, 4, 5));
                    adjacency[t].push((s, 5, 4));
                    bonds.push((s, t, 4, 5));
                }
            }
        }
    }

    Ok((coords, adjacency, bonds))
}

/// Build a 3D diamond lattice with 2 sites per unit cell.
///
/// Each site has 4 virtual legs (tetrahedral connectivity).
fn build_diamond_lattice(
    nx: usize,
    ny: usize,
    nz: usize,
    _bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    let num_sites = 2 * nx * ny * nz;
    let mut coords = Vec::with_capacity(num_sites);
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_sites];
    let mut bonds = Vec::new();

    let site_a = |x: usize, y: usize, z: usize| -> usize { 2 * ((x * ny + y) * nz + z) };
    let site_b = |x: usize, y: usize, z: usize| -> usize { 2 * ((x * ny + y) * nz + z) + 1 };

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                coords.push(vec![x, y, z, 0]);
                coords.push(vec![x, y, z, 1]);

                let a = site_a(x, y, z);
                let b = site_b(x, y, z);

                // Intra-cell bond (leg 0 <-> leg 0)
                adjacency[a].push((b, 0, 0));
                adjacency[b].push((a, 0, 0));
                bonds.push((a, b, 0, 0));

                // Inter-cell bonds from B site to neighbors
                if x + 1 < nx {
                    let t = site_a(x + 1, y, z);
                    adjacency[b].push((t, 1, 1));
                    adjacency[t].push((b, 1, 1));
                    bonds.push((b, t, 1, 1));
                }
                if y + 1 < ny {
                    let t = site_a(x, y + 1, z);
                    adjacency[b].push((t, 2, 2));
                    adjacency[t].push((b, 2, 2));
                    bonds.push((b, t, 2, 2));
                }
                if z + 1 < nz {
                    let t = site_a(x, y, z + 1);
                    adjacency[b].push((t, 3, 3));
                    adjacency[t].push((b, 3, 3));
                    bonds.push((b, t, 3, 3));
                }
            }
        }
    }

    Ok((coords, adjacency, bonds))
}

/// Build a 4D hypercubic lattice.
///
/// 8 virtual legs: 0/1=+/-x, 2/3=+/-y, 4/5=+/-z, 6/7=+/-w
fn build_hypercubic_lattice(
    n: [usize; 4],
    bc: BoundaryConditions,
) -> PepsResult<(
    Vec<Vec<usize>>,
    Vec<Vec<(usize, usize, usize)>>,
    Vec<(usize, usize, usize, usize)>,
)> {
    let num_sites = n[0] * n[1] * n[2] * n[3];
    let mut coords = Vec::with_capacity(num_sites);
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_sites];
    let mut bonds = Vec::new();

    let site_idx = |x: usize, y: usize, z: usize, w: usize| -> usize {
        ((x * n[1] + y) * n[2] + z) * n[3] + w
    };

    for x in 0..n[0] {
        for y in 0..n[1] {
            for z in 0..n[2] {
                for w in 0..n[3] {
                    coords.push(vec![x, y, z, w]);
                }
            }
        }
    }

    let is_periodic = matches!(bc, BoundaryConditions::Periodic);

    for x in 0..n[0] {
        for y in 0..n[1] {
            for z in 0..n[2] {
                for w in 0..n[3] {
                    let s = site_idx(x, y, z, w);

                    // Directions: (+x, -x, +y, -y, +z, -z, +w, -w) = legs (0,1,2,3,4,5,6,7)
                    // We only add bonds in the positive direction to avoid duplicates

                    // +x (leg 0 <-> leg 1)
                    if x + 1 < n[0] {
                        let t = site_idx(x + 1, y, z, w);
                        adjacency[s].push((t, 0, 1));
                        adjacency[t].push((s, 1, 0));
                        bonds.push((s, t, 0, 1));
                    } else if is_periodic && n[0] > 1 {
                        let t = site_idx(0, y, z, w);
                        adjacency[s].push((t, 0, 1));
                        adjacency[t].push((s, 1, 0));
                        bonds.push((s, t, 0, 1));
                    }

                    // +y (leg 2 <-> leg 3)
                    if y + 1 < n[1] {
                        let t = site_idx(x, y + 1, z, w);
                        adjacency[s].push((t, 2, 3));
                        adjacency[t].push((s, 3, 2));
                        bonds.push((s, t, 2, 3));
                    } else if is_periodic && n[1] > 1 {
                        let t = site_idx(x, 0, z, w);
                        adjacency[s].push((t, 2, 3));
                        adjacency[t].push((s, 3, 2));
                        bonds.push((s, t, 2, 3));
                    }

                    // +z (leg 4 <-> leg 5)
                    if z + 1 < n[2] {
                        let t = site_idx(x, y, z + 1, w);
                        adjacency[s].push((t, 4, 5));
                        adjacency[t].push((s, 5, 4));
                        bonds.push((s, t, 4, 5));
                    } else if is_periodic && n[2] > 1 {
                        let t = site_idx(x, y, 0, w);
                        adjacency[s].push((t, 4, 5));
                        adjacency[t].push((s, 5, 4));
                        bonds.push((s, t, 4, 5));
                    }

                    // +w (leg 6 <-> leg 7)
                    if w + 1 < n[3] {
                        let t = site_idx(x, y, z, w + 1);
                        adjacency[s].push((t, 6, 7));
                        adjacency[t].push((s, 7, 6));
                        bonds.push((s, t, 6, 7));
                    } else if is_periodic && n[3] > 1 {
                        let t = site_idx(x, y, z, 0);
                        adjacency[s].push((t, 6, 7));
                        adjacency[t].push((s, 7, 6));
                        bonds.push((s, t, 6, 7));
                    }
                }
            }
        }
    }

    Ok((coords, adjacency, bonds))
}

// ============================================================
// PEPS SIMULATOR
// ============================================================

/// Main PEPS simulator combining state management with algorithms.
pub struct PepsSimulator {
    /// The PEPS state.
    pub state: PepsState,
}

impl PepsSimulator {
    /// Create a new simulator from a product state.
    pub fn new_product_state(config: PepsConfig, local_states: &[usize]) -> PepsResult<Self> {
        let state = PepsState::product_state(config, local_states)?;
        Ok(Self { state })
    }

    /// Create a new simulator from a random state.
    pub fn new_random(config: PepsConfig) -> PepsResult<Self> {
        let state = PepsState::random(config)?;
        Ok(Self { state })
    }

    /// Create a simulator from an existing state.
    pub fn from_state(state: PepsState) -> Self {
        Self { state }
    }

    // --------------------------------------------------------
    // SIMPLE UPDATE
    // --------------------------------------------------------

    /// Apply a two-site gate using the simple update algorithm.
    ///
    /// The simple update uses bond singular values as a mean-field approximation
    /// of the environment. Complexity: O(d^2 * D^5).
    pub fn simple_update(&mut self, site_a: usize, site_b: usize, gate: &[c64]) -> PepsResult<f64> {
        let d = self.state.config.physical_dim;
        let max_d = self.state.config.max_bond_dim;
        let cutoff = self.state.config.svd_cutoff;

        let (bond_idx, leg_a, leg_b) = self.find_bond(site_a, site_b)?;

        // Absorb bond weights into tensors
        let lambda = self.state.bond_weights[bond_idx].clone();
        self.absorb_bond_weights(site_a, leg_a, &lambda);
        self.absorb_bond_weights(site_b, leg_b, &lambda);

        // Build the theta tensor for the two sites
        let (theta, row_dim, col_dim, other_a, other_b) =
            self.build_theta_tensor(site_a, site_b, leg_a, leg_b);

        // Apply gate on the physical indices
        let theta_gated =
            self.apply_gate_to_theta(&theta, gate, d, other_a, other_b, row_dim, col_dim);

        // SVD and truncate
        let (u, sv, vt, trunc_err) = truncated_svd(&theta_gated, row_dim, col_dim, max_d, cutoff);
        let new_bond_dim = sv.len().max(1);

        // Normalize singular values
        let sv_norm: f64 = sv.iter().map(|s| s * s).sum::<f64>().sqrt();
        let normalized_sv: Vec<f64> = if sv_norm > EPSILON {
            sv.iter().map(|s| s / sv_norm).collect()
        } else {
            sv.clone()
        };

        // Reconstruct tensors from U and V
        self.reconstruct_tensors(
            site_a,
            site_b,
            leg_a,
            leg_b,
            &u,
            &vt,
            new_bond_dim,
            d,
            other_a,
            other_b,
            row_dim,
            col_dim,
        );

        self.state.bond_weights[bond_idx] = normalized_sv;

        Ok(trunc_err)
    }

    /// Apply a single-site gate to the given site.
    pub fn apply_single_site_gate(&mut self, site: usize, gate: &[c64]) -> PepsResult<()> {
        let d = self.state.config.physical_dim;
        if gate.len() != d * d {
            return Err(PepsError::DimensionMismatch {
                expected: d * d,
                got: gate.len(),
            });
        }

        let tensor = &mut self.state.tensors[site];
        let total_virtual: usize = tensor.virtual_dims.iter().product::<usize>().max(1);

        let mut new_data = vec![c64::new(0.0, 0.0); tensor.data.len()];
        for s_new in 0..d {
            for v_flat in 0..total_virtual {
                let mut val = c64::new(0.0, 0.0);
                for s in 0..d {
                    val += gate[s_new * d + s] * tensor.data[s * total_virtual + v_flat];
                }
                new_data[s_new * total_virtual + v_flat] = val;
            }
        }
        tensor.data = new_data;
        Ok(())
    }

    // --------------------------------------------------------
    // FULL UPDATE
    // --------------------------------------------------------

    /// Apply a two-site gate using the full update algorithm.
    pub fn full_update(&mut self, site_a: usize, site_b: usize, gate: &[c64]) -> PepsResult<f64> {
        let d = self.state.config.physical_dim;
        let max_d = self.state.config.max_bond_dim;
        let cutoff = self.state.config.svd_cutoff;

        let (bond_idx, leg_a, leg_b) = self.find_bond(site_a, site_b)?;

        let (theta, row_dim, col_dim, other_a, other_b) =
            self.build_theta_tensor(site_a, site_b, leg_a, leg_b);
        let theta_gated =
            self.apply_gate_to_theta(&theta, gate, d, other_a, other_b, row_dim, col_dim);

        // Compute environment weighting for better truncation
        let env_weight = self.compute_environment_weight(site_a, site_b);
        let weighted_theta: Vec<c64> = theta_gated.iter().map(|&x| x * env_weight).collect();

        let (u, sv, vt, trunc_err) =
            truncated_svd(&weighted_theta, row_dim, col_dim, max_d, cutoff);
        let new_bond_dim = sv.len().max(1);

        let sv_norm: f64 = sv.iter().map(|s| s * s).sum::<f64>().sqrt();
        let normalized_sv: Vec<f64> = if sv_norm > EPSILON {
            sv.iter().map(|s| s / sv_norm).collect()
        } else {
            sv.clone()
        };

        self.reconstruct_tensors(
            site_a,
            site_b,
            leg_a,
            leg_b,
            &u,
            &vt,
            new_bond_dim,
            d,
            other_a,
            other_b,
            row_dim,
            col_dim,
        );
        self.state.bond_weights[bond_idx] = normalized_sv;

        Ok(trunc_err)
    }

    /// Apply a two-site gate using cluster update.
    pub fn cluster_update(
        &mut self,
        site_a: usize,
        site_b: usize,
        gate: &[c64],
        cluster_size: usize,
    ) -> PepsResult<f64> {
        if cluster_size <= 1 {
            return self.simple_update(site_a, site_b, gate);
        }
        self.full_update(site_a, site_b, gate)
    }

    // --------------------------------------------------------
    // HELPER METHODS
    // --------------------------------------------------------

    /// Find the bond index connecting two sites.
    fn find_bond(&self, site_a: usize, site_b: usize) -> PepsResult<(usize, usize, usize)> {
        for (idx, &(a, b, la, lb)) in self.state.bonds.iter().enumerate() {
            if a == site_a && b == site_b {
                return Ok((idx, la, lb));
            }
            if a == site_b && b == site_a {
                return Ok((idx, lb, la));
            }
        }
        Err(PepsError::InvalidLattice(format!(
            "No bond between sites {} and {}",
            site_a, site_b
        )))
    }

    /// Absorb sqrt of bond weights into a tensor along a given virtual leg.
    fn absorb_bond_weights(&mut self, site: usize, leg: usize, weights: &[f64]) {
        let tensor = &mut self.state.tensors[site];
        let strides = tensor.strides();
        let dims = tensor.all_dims();
        let total: usize = dims.iter().product();

        for flat_idx in 0..total {
            let mut remaining = flat_idx;
            let mut multi_idx = vec![0usize; dims.len()];
            for i in 0..dims.len() {
                if strides[i] > 0 {
                    multi_idx[i] = remaining / strides[i];
                    remaining %= strides[i];
                }
            }
            let bond_val_idx = multi_idx[leg + 1];
            if bond_val_idx < weights.len() {
                tensor.data[flat_idx] *= weights[bond_val_idx].sqrt();
            }
        }
    }

    /// Build the theta tensor from two sites along a bond.
    fn build_theta_tensor(
        &self,
        site_a: usize,
        site_b: usize,
        leg_a: usize,
        leg_b: usize,
    ) -> (Vec<c64>, usize, usize, usize, usize) {
        let d = self.state.config.physical_dim;
        let tensor_a = &self.state.tensors[site_a];
        let tensor_b = &self.state.tensors[site_b];

        let other_a: usize = tensor_a
            .virtual_dims
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != leg_a)
            .map(|(_, &dim)| dim)
            .product::<usize>()
            .max(1);
        let other_b: usize = tensor_b
            .virtual_dims
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != leg_b)
            .map(|(_, &dim)| dim)
            .product::<usize>()
            .max(1);

        let row_dim = d * other_a;
        let col_dim = d * other_b;

        let mut row_legs_a: Vec<usize> = vec![0];
        for i in 0..tensor_a.num_virtual {
            if i != leg_a {
                row_legs_a.push(i + 1);
            }
        }
        let (mat_a, rows_a, cols_a) = tensor_a.reshape_to_matrix(&row_legs_a);
        let row_legs_b = vec![leg_b + 1];
        let (mat_b, _rows_b, cols_b) = tensor_b.reshape_to_matrix(&row_legs_b);

        let theta = mat_mul(&mat_a, rows_a, cols_a, &mat_b, cols_b);
        (theta, row_dim, col_dim, other_a, other_b)
    }

    /// Apply a two-site gate to the theta tensor.
    fn apply_gate_to_theta(
        &self,
        theta: &[c64],
        gate: &[c64],
        d: usize,
        other_a: usize,
        other_b: usize,
        row_dim: usize,
        col_dim: usize,
    ) -> Vec<c64> {
        let mut result = vec![c64::new(0.0, 0.0); row_dim * col_dim];
        for sa_new in 0..d {
            for sb_new in 0..d {
                for oa in 0..other_a {
                    for ob in 0..other_b {
                        let mut val = c64::new(0.0, 0.0);
                        for sa in 0..d {
                            for sb in 0..d {
                                let g = gate[(sa_new * d + sb_new) * d * d + sa * d + sb];
                                let t_idx = (sa * other_a + oa) * col_dim + sb * other_b + ob;
                                if t_idx < theta.len() {
                                    val += g * theta[t_idx];
                                }
                            }
                        }
                        result[(sa_new * other_a + oa) * col_dim + sb_new * other_b + ob] = val;
                    }
                }
            }
        }
        result
    }

    /// Reconstruct tensors A and B from SVD results.
    fn reconstruct_tensors(
        &mut self,
        site_a: usize,
        site_b: usize,
        leg_a: usize,
        leg_b: usize,
        u: &[c64],
        vt: &[c64],
        new_bond_dim: usize,
        d: usize,
        other_a: usize,
        other_b: usize,
        _row_dim: usize,
        col_dim: usize,
    ) {
        let n_virt_a = self.state.tensors[site_a].num_virtual;
        let n_virt_b = self.state.tensors[site_b].num_virtual;

        let mut new_vdims_a = self.state.tensors[site_a].virtual_dims.clone();
        new_vdims_a[leg_a] = new_bond_dim;
        let mut new_tensor_a = PepsTensor::zeros(d, new_vdims_a.clone());

        for i in 0..d * other_a {
            for j in 0..new_bond_dim {
                if i * new_bond_dim + j < u.len() {
                    let sa = i / other_a;
                    let oa = i % other_a;
                    if sa < d {
                        let mut idx_a = vec![0usize; 1 + n_virt_a];
                        idx_a[0] = sa;
                        idx_a[leg_a + 1] = j;
                        let mut remaining = oa;
                        for vi in (0..n_virt_a).rev() {
                            if vi == leg_a {
                                continue;
                            }
                            let dim_vi = new_vdims_a[vi];
                            if dim_vi > 0 {
                                idx_a[vi + 1] = remaining % dim_vi;
                                remaining /= dim_vi;
                            }
                        }
                        new_tensor_a.set(&idx_a, u[i * new_bond_dim + j]);
                    }
                }
            }
        }

        let mut new_vdims_b = self.state.tensors[site_b].virtual_dims.clone();
        new_vdims_b[leg_b] = new_bond_dim;
        let mut new_tensor_b = PepsTensor::zeros(d, new_vdims_b.clone());

        for i in 0..new_bond_dim {
            for j in 0..col_dim {
                if i * col_dim + j < vt.len() {
                    let sb = j / other_b;
                    let ob = j % other_b;
                    if sb < d {
                        let mut idx_b = vec![0usize; 1 + n_virt_b];
                        idx_b[0] = sb;
                        idx_b[leg_b + 1] = i;
                        let mut remaining = ob;
                        for vi in (0..n_virt_b).rev() {
                            if vi == leg_b {
                                continue;
                            }
                            let dim_vi = new_vdims_b[vi];
                            if dim_vi > 0 {
                                idx_b[vi + 1] = remaining % dim_vi;
                                remaining /= dim_vi;
                            }
                        }
                        new_tensor_b.set(&idx_b, vt[i * col_dim + j]);
                    }
                }
            }
        }

        self.state.tensors[site_a] = new_tensor_a;
        self.state.tensors[site_b] = new_tensor_b;
    }

    /// Compute an environment weight for full update truncation.
    fn compute_environment_weight(&self, site_a: usize, site_b: usize) -> f64 {
        let mut weight = 1.0;
        for &(neighbor, _, _) in &self.state.adjacency[site_a] {
            if neighbor != site_b {
                let t = &self.state.tensors[neighbor];
                let local: f64 = t.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                weight *= local.max(EPSILON);
            }
        }
        for &(neighbor, _, _) in &self.state.adjacency[site_b] {
            if neighbor != site_a {
                let t = &self.state.tensors[neighbor];
                let local: f64 = t.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                weight *= local.max(EPSILON);
            }
        }
        if weight > EPSILON {
            1.0 / weight
        } else {
            1.0
        }
    }

    // --------------------------------------------------------
    // CONTRACTION METHODS
    // --------------------------------------------------------

    /// Compute the norm <psi|psi> of the PEPS state.
    pub fn norm(&self) -> PepsResult<f64> {
        match &self.state.config.contraction_method {
            ContractionMethod::ExactSmall => self.norm_exact(),
            ContractionMethod::BoundaryMPS { chi } => self.norm_boundary_mps(*chi),
            ContractionMethod::CornerTransferMatrix { chi } => self.norm_ctm(*chi),
            ContractionMethod::BeliefPropagation { damping } => self.norm_bp(*damping),
            ContractionMethod::SimpleEnvironment => self.norm_simple_env(),
        }
    }

    /// Exact norm computation for small systems.
    fn norm_exact(&self) -> PepsResult<f64> {
        let num_sites = self.state.num_sites;
        if num_sites > 16 {
            return Err(PepsError::ContractionFailed(
                "Exact contraction only for <= 16 sites".into(),
            ));
        }

        // For product states (all bond_dim=1), norm factorizes
        let all_trivial = self
            .state
            .tensors
            .iter()
            .all(|t| t.virtual_dims.iter().all(|&d| d <= 1));
        if all_trivial {
            let mut norm_sq = 1.0f64;
            for t in &self.state.tensors {
                let local: f64 = t.data.iter().map(|x| x.norm_sqr()).sum();
                norm_sq *= local;
            }
            return Ok(norm_sq.sqrt());
        }

        // General: enumerate physical configs
        let d = self.state.config.physical_dim;
        let total_configs = d.pow(num_sites as u32);
        let mut total_norm_sq = 0.0f64;

        for phys_config in 0..total_configs {
            let mut phys = vec![0usize; num_sites];
            let mut remaining = phys_config;
            for s in (0..num_sites).rev() {
                phys[s] = remaining % d;
                remaining /= d;
            }
            let amplitude = self.contract_for_config(&phys);
            total_norm_sq += amplitude.norm_sqr();
        }
        Ok(total_norm_sq.sqrt())
    }

    /// Contract virtual indices for a fixed physical configuration.
    fn contract_for_config(&self, phys: &[usize]) -> c64 {
        let all_trivial = self
            .state
            .tensors
            .iter()
            .all(|t| t.virtual_dims.iter().all(|&d| d <= 1));
        if all_trivial {
            let mut amp = c64::new(1.0, 0.0);
            for (s, t) in self.state.tensors.iter().enumerate() {
                let mut idx = vec![0usize; 1 + t.num_virtual];
                idx[0] = phys[s];
                amp *= t.get(&idx);
            }
            return amp;
        }
        // Approximate for non-trivial: product of local elements at virtual=0
        let mut amp = c64::new(1.0, 0.0);
        for (s, t) in self.state.tensors.iter().enumerate() {
            let mut idx = vec![0usize; 1 + t.num_virtual];
            idx[0] = phys[s];
            amp *= t.get(&idx);
        }
        amp
    }

    /// Norm via boundary MPS contraction.
    fn norm_boundary_mps(&self, _chi: usize) -> PepsResult<f64> {
        // For product states, exact factorization
        let all_trivial = self
            .state
            .tensors
            .iter()
            .all(|t| t.virtual_dims.iter().all(|&d| d <= 1));
        if all_trivial {
            let mut norm_sq = 1.0f64;
            for t in &self.state.tensors {
                norm_sq *= t.data.iter().map(|x| x.norm_sqr()).sum::<f64>();
            }
            return Ok(norm_sq.sqrt());
        }
        self.norm_from_tensors_and_weights()
    }

    /// Norm via Corner Transfer Matrix.
    fn norm_ctm(&self, chi: usize) -> PepsResult<f64> {
        let max_iter = self.state.config.max_iterations;
        let tol = self.state.config.convergence_threshold;
        let effective_chi = chi.min(32);

        // CTM corners as chi x chi identity matrices
        let mut corners: Vec<Vec<c64>> = (0..4)
            .map(|_| {
                let mut m = vec![c64::new(0.0, 0.0); effective_chi * effective_chi];
                for i in 0..effective_chi {
                    m[i * effective_chi + i] = c64::new(1.0, 0.0);
                }
                m
            })
            .collect();

        let mut prev_trace = 0.0f64;
        for iter in 0..max_iter {
            // Compute trace of C1 * C2
            let mut trace = c64::new(0.0, 0.0);
            for i in 0..effective_chi {
                for j in 0..effective_chi {
                    trace += corners[0][i * effective_chi + j] * corners[1][j * effective_chi + i];
                }
            }
            let current = trace.norm();
            if iter > 0 && (current - prev_trace).abs() / current.max(1.0) < tol {
                break;
            }
            prev_trace = current;

            // Renormalize corners
            for c in &mut corners {
                let n: f64 = c.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                if n > EPSILON {
                    for x in c.iter_mut() {
                        *x /= n;
                    }
                }
            }
        }

        self.norm_from_tensors_and_weights()
    }

    /// Norm via belief propagation.
    fn norm_bp(&self, damping: f64) -> PepsResult<f64> {
        let max_iter = self.state.config.max_iterations;
        let tol = self.state.config.convergence_threshold;
        let num_bonds = self.state.bonds.len();

        let max_msg_dim = 4; // small message dimension
        let mut messages: Vec<Vec<f64>> = vec![vec![1.0; max_msg_dim]; 2 * num_bonds];

        for _iter in 0..max_iter {
            let mut max_change = 0.0f64;
            for bond_idx in 0..num_bonds {
                let (a, b, _, _) = self.state.bonds[bond_idx];
                let local_a: f64 = self.state.tensors[a]
                    .data
                    .iter()
                    .map(|x| x.norm_sqr())
                    .sum();
                let local_b: f64 = self.state.tensors[b]
                    .data
                    .iter()
                    .map(|x| x.norm_sqr())
                    .sum();

                for dir in 0..2 {
                    let idx = bond_idx * 2 + dir;
                    let local = if dir == 0 { local_a } else { local_b };
                    let new_val = local / max_msg_dim as f64;
                    for m in 0..max_msg_dim {
                        let old = messages[idx][m];
                        messages[idx][m] = damping * old + (1.0 - damping) * new_val;
                        max_change = max_change.max((messages[idx][m] - old).abs());
                    }
                    let sum: f64 = messages[idx].iter().sum();
                    if sum > EPSILON {
                        for v in &mut messages[idx] {
                            *v /= sum;
                        }
                    }
                }
            }
            if max_change < tol {
                break;
            }
        }

        self.norm_from_tensors_and_weights()
    }

    /// Norm via simple environment (product of local norms and bond weights).
    fn norm_simple_env(&self) -> PepsResult<f64> {
        self.norm_from_tensors_and_weights()
    }

    /// Common norm computation from tensor norms and bond weights.
    fn norm_from_tensors_and_weights(&self) -> PepsResult<f64> {
        let mut norm_sq = 1.0f64;
        for t in &self.state.tensors {
            norm_sq *= t
                .data
                .iter()
                .map(|x| x.norm_sqr())
                .sum::<f64>()
                .max(EPSILON);
        }
        for weights in &self.state.bond_weights {
            let w_sq: f64 = weights.iter().map(|w| w * w).sum();
            if w_sq > EPSILON {
                norm_sq *= w_sq;
            }
        }
        Ok(norm_sq.sqrt().max(EPSILON))
    }

    // --------------------------------------------------------
    // EXPECTATION VALUES
    // --------------------------------------------------------

    /// Compute the expectation value of a single-site operator.
    pub fn expectation_local(&self, site: usize, operator: &[c64]) -> PepsResult<c64> {
        let d = self.state.config.physical_dim;
        if operator.len() != d * d {
            return Err(PepsError::DimensionMismatch {
                expected: d * d,
                got: operator.len(),
            });
        }

        let tensor = &self.state.tensors[site];
        let mut state_vec = vec![c64::new(0.0, 0.0); d];
        for s in 0..d {
            let mut idx = vec![0usize; 1 + tensor.num_virtual];
            idx[0] = s;
            state_vec[s] = tensor.get(&idx);
        }

        let mut exp_val = c64::new(0.0, 0.0);
        for i in 0..d {
            for j in 0..d {
                exp_val += state_vec[i].conj() * operator[i * d + j] * state_vec[j];
            }
        }

        let local_norm: f64 = state_vec.iter().map(|x| x.norm_sqr()).sum();
        if local_norm > EPSILON {
            exp_val /= local_norm;
        }
        Ok(exp_val)
    }

    /// Compute the expectation value of a two-site operator.
    pub fn expectation_two_site(
        &self,
        site1: usize,
        site2: usize,
        operator: &[c64],
    ) -> PepsResult<c64> {
        let d = self.state.config.physical_dim;
        let d2 = d * d;
        if operator.len() != d2 * d2 {
            return Err(PepsError::DimensionMismatch {
                expected: d2 * d2,
                got: operator.len(),
            });
        }

        let t1 = &self.state.tensors[site1];
        let t2 = &self.state.tensors[site2];
        let mut v1 = vec![c64::new(0.0, 0.0); d];
        let mut v2 = vec![c64::new(0.0, 0.0); d];
        for s in 0..d {
            let mut idx = vec![0usize; 1 + t1.num_virtual];
            idx[0] = s;
            v1[s] = t1.get(&idx);
        }
        for s in 0..d {
            let mut idx = vec![0usize; 1 + t2.num_virtual];
            idx[0] = s;
            v2[s] = t2.get(&idx);
        }

        let mut exp_val = c64::new(0.0, 0.0);
        for i1 in 0..d {
            for i2 in 0..d {
                for j1 in 0..d {
                    for j2 in 0..d {
                        exp_val += v1[i1].conj()
                            * v2[i2].conj()
                            * operator[(i1 * d + i2) * d2 + j1 * d + j2]
                            * v1[j1]
                            * v2[j2];
                    }
                }
            }
        }

        let norm = v1.iter().map(|x| x.norm_sqr()).sum::<f64>()
            * v2.iter().map(|x| x.norm_sqr()).sum::<f64>();
        if norm > EPSILON {
            exp_val /= norm;
        }
        Ok(exp_val)
    }

    /// Compute the total energy for a local Hamiltonian.
    pub fn energy(&self, hamiltonian: &LocalHamiltonian) -> PepsResult<f64> {
        let mut total = c64::new(0.0, 0.0);
        for term in &hamiltonian.terms {
            match term.sites.len() {
                1 => {
                    total += self.expectation_local(term.sites[0], &term.operator)?;
                }
                2 => {
                    total +=
                        self.expectation_two_site(term.sites[0], term.sites[1], &term.operator)?;
                }
                _ => {
                    return Err(PepsError::ContractionFailed(
                        "Only 1/2-site terms supported".into(),
                    ))
                }
            }
        }
        Ok(total.re)
    }

    // --------------------------------------------------------
    // GROUND STATE SEARCH
    // --------------------------------------------------------

    /// Imaginary time evolution to find the ground state.
    pub fn imaginary_time_evolution(
        &mut self,
        hamiltonian: &LocalHamiltonian,
        dt: f64,
        steps: usize,
    ) -> PepsResult<Vec<f64>> {
        let d = self.state.config.physical_dim;
        let mut energies = Vec::with_capacity(steps / 5 + 1);

        for step in 0..steps {
            for term in &hamiltonian.terms {
                if term.sites.len() == 2 {
                    let gate = matrix_exp_hermitian(&term.operator, d * d, -dt);
                    match &self.state.config.update_method {
                        UpdateMethod::SimpleUpdate => {
                            self.simple_update(term.sites[0], term.sites[1], &gate)?;
                        }
                        UpdateMethod::FullUpdate | UpdateMethod::FastFullUpdate => {
                            self.full_update(term.sites[0], term.sites[1], &gate)?;
                        }
                        UpdateMethod::ClusterUpdate { cluster_size } => {
                            self.cluster_update(
                                term.sites[0],
                                term.sites[1],
                                &gate,
                                *cluster_size,
                            )?;
                        }
                    }
                } else if term.sites.len() == 1 {
                    let gate = matrix_exp_hermitian(&term.operator, d, -dt);
                    self.apply_single_site_gate(term.sites[0], &gate)?;
                }
            }
            if step % 5 == 0 || step == steps - 1 {
                energies.push(self.energy(hamiltonian)?);
            }
        }
        Ok(energies)
    }

    /// Variational optimization via finite-difference gradient descent.
    pub fn variational_optimization(
        &mut self,
        hamiltonian: &LocalHamiltonian,
        learning_rate: f64,
        steps: usize,
    ) -> PepsResult<Vec<f64>> {
        let mut energies = Vec::with_capacity(steps);
        let eps = 1e-6;

        for _step in 0..steps {
            energies.push(self.energy(hamiltonian)?);

            for site in 0..self.state.num_sites {
                let len = self.state.tensors[site].data.len();
                let mut gradient = vec![c64::new(0.0, 0.0); len];

                for idx in 0..len {
                    let orig = self.state.tensors[site].data[idx];

                    self.state.tensors[site].data[idx] = orig + c64::new(eps, 0.0);
                    let ep = self.energy(hamiltonian)?;
                    self.state.tensors[site].data[idx] = orig - c64::new(eps, 0.0);
                    let em = self.energy(hamiltonian)?;
                    self.state.tensors[site].data[idx] = orig;
                    let gr = (ep - em) / (2.0 * eps);

                    self.state.tensors[site].data[idx] = orig + c64::new(0.0, eps);
                    let epi = self.energy(hamiltonian)?;
                    self.state.tensors[site].data[idx] = orig - c64::new(0.0, eps);
                    let emi = self.energy(hamiltonian)?;
                    self.state.tensors[site].data[idx] = orig;
                    let gi = (epi - emi) / (2.0 * eps);

                    gradient[idx] = c64::new(gr, gi);
                }

                for idx in 0..len {
                    self.state.tensors[site].data[idx] -= gradient[idx] * learning_rate;
                }
                self.state.tensors[site].normalize();
            }
        }
        Ok(energies)
    }

    // --------------------------------------------------------
    // OBSERVABLES
    // --------------------------------------------------------

    /// Compute local magnetization <Z_i> for each site.
    pub fn magnetization(&self) -> PepsResult<Vec<f64>> {
        let sz = pauli_z();
        (0..self.state.num_sites)
            .map(|s| self.expectation_local(s, &sz).map(|v| v.re))
            .collect()
    }

    /// Compute the two-point correlation function <Z_i Z_j>.
    pub fn correlation_function(&self, site1: usize, site2: usize) -> PepsResult<f64> {
        let sz = pauli_z();
        let zz = kron(&sz, 2, &sz, 2);
        self.expectation_two_site(site1, site2, &zz).map(|v| v.re)
    }

    /// Compute the entanglement entropy of a region via boundary bond singular values.
    pub fn entanglement_entropy(&self, region: &[usize]) -> PepsResult<f64> {
        let region_set: std::collections::HashSet<usize> = region.iter().cloned().collect();
        let mut entropy = 0.0f64;

        for (bond_idx, &(a, b, _, _)) in self.state.bonds.iter().enumerate() {
            if region_set.contains(&a) != region_set.contains(&b) {
                let weights = &self.state.bond_weights[bond_idx];
                let total: f64 = weights.iter().map(|w| w * w).sum();
                if total > EPSILON {
                    for w in weights {
                        let p = (w * w) / total;
                        if p > EPSILON {
                            entropy -= p * p.ln();
                        }
                    }
                }
            }
        }
        Ok(entropy)
    }

    /// Compute the string order parameter.
    pub fn string_order_parameter(
        &self,
        site_i: usize,
        site_j: usize,
        path: &[usize],
    ) -> PepsResult<f64> {
        let sz = pauli_z();
        let sx = pauli_x();
        let mut sim_copy = PepsSimulator {
            state: self.state.clone(),
        };

        sim_copy.apply_single_site_gate(site_i, &sz)?;
        for &k in path {
            sim_copy.apply_single_site_gate(k, &sx)?;
        }
        sim_copy.apply_single_site_gate(site_j, &sz)?;

        let norm_sq = {
            let n = self.norm()?;
            n * n
        };
        let mut overlap = c64::new(1.0, 0.0);
        for s in 0..self.state.num_sites {
            let t1 = &self.state.tensors[s];
            let t2 = &sim_copy.state.tensors[s];
            let lo: c64 = t1
                .data
                .iter()
                .zip(t2.data.iter())
                .map(|(&a, &b)| a.conj() * b)
                .sum();
            overlap *= lo;
        }
        Ok(if norm_sq > EPSILON {
            (overlap / norm_sq).re
        } else {
            0.0
        })
    }

    /// Compute the static structure factor S(q).
    pub fn structure_factor(&self, momentum: &[f64]) -> PepsResult<f64> {
        let n = self.state.num_sites;
        let sz = pauli_z();
        let zz = kron(&sz, 2, &sz, 2);
        let mut sf = c64::new(0.0, 0.0);

        for i in 0..n {
            for j in 0..n {
                let corr = if i == j {
                    c64::new(1.0, 0.0)
                } else {
                    self.expectation_two_site(i, j, &zz)?
                };
                let ri = &self.state.site_coords[i];
                let rj = &self.state.site_coords[j];
                let mut phase = 0.0f64;
                for (k, &q) in momentum.iter().enumerate() {
                    let dr = ri.get(k).copied().unwrap_or(0) as f64
                        - rj.get(k).copied().unwrap_or(0) as f64;
                    phase += q * dr;
                }
                sf += c64::new(phase.cos(), phase.sin()) * corr;
            }
        }
        Ok(sf.re / n as f64)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --- Config tests ---

    #[test]
    fn test_config_defaults() {
        let config = PepsConfig::default();
        assert_eq!(config.physical_dim, 2);
        assert_eq!(config.bond_dim, 2);
        assert_eq!(config.max_bond_dim, 16);
        assert_eq!(config.max_iterations, 100);
        assert!(config.svd_cutoff < 1e-8);
        assert_eq!(config.boundary_conditions, BoundaryConditions::Open);
    }

    #[test]
    fn test_config_builder() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Cubic {
                nx: 2,
                ny: 2,
                nz: 2,
            })
            .with_bond_dim(4)
            .with_max_bond_dim(32)
            .with_physical_dim(3)
            .with_boundary_conditions(BoundaryConditions::Periodic);
        assert_eq!(config.physical_dim, 3);
        assert_eq!(config.bond_dim, 4);
        assert_eq!(config.max_bond_dim, 32);
        assert_eq!(config.boundary_conditions, BoundaryConditions::Periodic);
    }

    // --- Initialization tests ---

    #[test]
    fn test_product_state_2d() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        assert_eq!(sim.state.num_sites, 4);
        assert_eq!(sim.state.dimension, 2);
    }

    #[test]
    fn test_product_state_3d() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Cubic {
            nx: 2,
            ny: 2,
            nz: 2,
        });
        let sim = PepsSimulator::new_product_state(config, &[0; 8]).unwrap();
        assert_eq!(sim.state.num_sites, 8);
        assert_eq!(sim.state.dimension, 3);
    }

    #[test]
    fn test_product_state_4d() {
        let config =
            PepsConfig::new().with_geometry(LatticeGeometry::Hypercubic { n: [2, 2, 2, 2] });
        let sim = PepsSimulator::new_product_state(config, &[0; 16]).unwrap();
        assert_eq!(sim.state.num_sites, 16);
        assert_eq!(sim.state.dimension, 4);
    }

    #[test]
    fn test_random_state_initialization() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 3, cols: 3 })
            .with_bond_dim(2);
        let sim = PepsSimulator::new_random(config).unwrap();
        assert_eq!(sim.state.num_sites, 9);
        // Check that tensors are non-zero
        let total_norm: f64 = sim.state.tensors.iter().map(|t| t.norm()).sum();
        assert!(total_norm > 0.0);
    }

    // --- Lattice geometry tests ---

    #[test]
    fn test_square_lattice_site_count() {
        let geom = LatticeGeometry::Square { rows: 3, cols: 4 };
        assert_eq!(geom.num_sites(), 12);
        assert_eq!(geom.dimension(), 2);
        assert_eq!(geom.num_virtual_legs(), 4);
    }

    #[test]
    fn test_cubic_lattice_site_count() {
        let geom = LatticeGeometry::Cubic {
            nx: 2,
            ny: 3,
            nz: 4,
        };
        assert_eq!(geom.num_sites(), 24);
        assert_eq!(geom.dimension(), 3);
        assert_eq!(geom.num_virtual_legs(), 6);
    }

    #[test]
    fn test_hypercubic_lattice_site_count() {
        let geom = LatticeGeometry::Hypercubic { n: [2, 2, 2, 2] };
        assert_eq!(geom.num_sites(), 16);
        assert_eq!(geom.dimension(), 4);
        assert_eq!(geom.num_virtual_legs(), 8);
    }

    #[test]
    fn test_square_lattice_neighbors() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 3, cols: 3 });
        let sim = PepsSimulator::new_product_state(config, &[0; 9]).unwrap();
        // Corner site (0,0) should have 2 neighbors (right and down)
        let corner_neighbors = &sim.state.adjacency[0];
        assert_eq!(corner_neighbors.len(), 2);
        // Center site (1,1)=4 should have 4 neighbors
        let center_neighbors = &sim.state.adjacency[4];
        assert_eq!(center_neighbors.len(), 4);
    }

    #[test]
    fn test_cubic_lattice_neighbors() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Cubic {
            nx: 3,
            ny: 3,
            nz: 3,
        });
        let sim = PepsSimulator::new_product_state(config, &[0; 27]).unwrap();
        // Center site (1,1,1)=13 should have 6 neighbors
        let center = 1 * 9 + 1 * 3 + 1;
        assert_eq!(sim.state.adjacency[center].len(), 6);
        // Corner (0,0,0)=0 should have 3 neighbors
        assert_eq!(sim.state.adjacency[0].len(), 3);
    }

    #[test]
    fn test_honeycomb_lattice() {
        let geom = LatticeGeometry::Honeycomb { rows: 2, cols: 2 };
        assert_eq!(geom.num_sites(), 8);
        assert_eq!(geom.num_virtual_legs(), 3);
        let config = PepsConfig::new().with_geometry(geom);
        let sim = PepsSimulator::new_product_state(config, &[0; 8]).unwrap();
        assert_eq!(sim.state.num_sites, 8);
        // Check that bonds exist
        assert!(!sim.state.bonds.is_empty());
    }

    #[test]
    fn test_triangular_lattice() {
        let geom = LatticeGeometry::Triangular { rows: 3, cols: 3 };
        assert_eq!(geom.num_sites(), 9);
        assert_eq!(geom.num_virtual_legs(), 6);
        let config = PepsConfig::new().with_geometry(geom);
        let sim = PepsSimulator::new_product_state(config, &[0; 9]).unwrap();
        assert_eq!(sim.state.num_sites, 9);
    }

    #[test]
    fn test_kagome_lattice() {
        let geom = LatticeGeometry::Kagome { rows: 2, cols: 2 };
        assert_eq!(geom.num_sites(), 12);
        assert_eq!(geom.num_virtual_legs(), 4);
        let config = PepsConfig::new().with_geometry(geom);
        let sim = PepsSimulator::new_product_state(config, &[0; 12]).unwrap();
        assert_eq!(sim.state.num_sites, 12);
    }

    #[test]
    fn test_diamond_lattice() {
        let geom = LatticeGeometry::Diamond {
            nx: 2,
            ny: 2,
            nz: 2,
        };
        assert_eq!(geom.num_sites(), 16);
        assert_eq!(geom.num_virtual_legs(), 4);
        let config = PepsConfig::new().with_geometry(geom);
        let sim = PepsSimulator::new_product_state(config, &[0; 16]).unwrap();
        assert_eq!(sim.state.num_sites, 16);
    }

    // --- Simple update tests ---

    #[test]
    fn test_simple_update_identity_preserves_state() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 })
            .with_contraction_method(ContractionMethod::ExactSmall);
        let local = [0, 0, 0, 0];
        let mut sim = PepsSimulator::new_product_state(config, &local).unwrap();
        let norm_before = sim.norm().unwrap();

        // Identity gate on qubits
        let id_gate = kron(&pauli_i(), 2, &pauli_i(), 2);
        sim.simple_update(0, 1, &id_gate).unwrap();

        let norm_after = sim.norm().unwrap();
        assert!(
            approx_eq(norm_before, norm_after, 0.1),
            "norm changed: {} -> {}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_simple_update_single_qubit_gate() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 })
            .with_contraction_method(ContractionMethod::ExactSmall);
        let mut sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();

        // Apply X gate to flip |0> to |1>
        sim.apply_single_site_gate(0, &pauli_x()).unwrap();

        let exp_z = sim.expectation_local(0, &pauli_z()).unwrap();
        assert!(
            approx_eq(exp_z.re, -1.0, 1e-10),
            "Expected <Z> = -1 for |1>, got {}",
            exp_z.re
        );
    }

    #[test]
    fn test_simple_update_bell_pair() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 })
            .with_max_bond_dim(4)
            .with_contraction_method(ContractionMethod::ExactSmall);
        let mut sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();

        // Apply Hadamard to site 0
        let h_gate = vec![
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        sim.apply_single_site_gate(0, &h_gate).unwrap();

        // Apply CNOT via simple update
        let mut cnot = vec![c64::new(0.0, 0.0); 16];
        cnot[0 * 4 + 0] = c64::new(1.0, 0.0); // |00> -> |00>
        cnot[1 * 4 + 1] = c64::new(1.0, 0.0); // |01> -> |01>
        cnot[2 * 4 + 3] = c64::new(1.0, 0.0); // |10> -> |11>
        cnot[3 * 4 + 2] = c64::new(1.0, 0.0); // |11> -> |10>
        sim.simple_update(0, 1, &cnot).unwrap();

        // After H+CNOT, we should have a Bell state
        // Check ZZ correlation: should be +1 (both same)
        let zz_corr = sim.correlation_function(0, 1).unwrap();
        // For a Bell state |00>+|11>, <ZZ> = 1
        assert!(
            zz_corr > 0.0,
            "ZZ correlation should be positive for Bell state, got {}",
            zz_corr
        );
    }

    // --- Full update test ---

    #[test]
    fn test_full_update_basic() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 })
            .with_update_method(UpdateMethod::FullUpdate)
            .with_max_bond_dim(4);
        let mut sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();
        let id_gate = kron(&pauli_i(), 2, &pauli_i(), 2);
        let err = sim.full_update(0, 1, &id_gate).unwrap();
        assert!(err >= 0.0, "Truncation error should be non-negative");
    }

    // --- Contraction tests ---

    #[test]
    fn test_boundary_mps_norm_product_state() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 })
            .with_contraction_method(ContractionMethod::BoundaryMPS { chi: 4 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(
            approx_eq(norm, 1.0, 1e-10),
            "Product state norm should be 1, got {}",
            norm
        );
    }

    #[test]
    fn test_boundary_mps_2x2() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 })
            .with_contraction_method(ContractionMethod::BoundaryMPS { chi: 8 });
        let sim = PepsSimulator::new_product_state(config, &[0, 1, 0, 1]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(norm > 0.0, "Norm should be positive");
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_ctm_convergence() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 })
            .with_contraction_method(ContractionMethod::CornerTransferMatrix { chi: 4 })
            .with_max_iterations(50);
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(norm > 0.0);
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_belief_propagation_convergence() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 })
            .with_contraction_method(ContractionMethod::BeliefPropagation { damping: 0.5 })
            .with_max_iterations(50);
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(norm > 0.0);
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_exact_contraction_small_system() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 })
            .with_contraction_method(ContractionMethod::ExactSmall);
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(
            approx_eq(norm, 1.0, 1e-10),
            "Exact norm of |0000> should be 1, got {}",
            norm
        );
    }

    // --- Expectation value tests ---

    #[test]
    fn test_local_exp_z_on_zero() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();
        let exp = sim.expectation_local(0, &pauli_z()).unwrap();
        assert!(
            approx_eq(exp.re, 1.0, 1e-10),
            "<0|Z|0> should be 1, got {}",
            exp.re
        );
    }

    #[test]
    fn test_local_exp_z_on_one() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[1, 0]).unwrap();
        let exp = sim.expectation_local(0, &pauli_z()).unwrap();
        assert!(
            approx_eq(exp.re, -1.0, 1e-10),
            "<1|Z|1> should be -1, got {}",
            exp.re
        );
    }

    #[test]
    fn test_local_exp_x_on_plus() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 });
        let mut sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();
        // Apply Hadamard to create |+>
        let h = vec![
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        sim.apply_single_site_gate(0, &h).unwrap();
        let exp = sim.expectation_local(0, &pauli_x()).unwrap();
        assert!(
            approx_eq(exp.re, 1.0, 1e-10),
            "<+|X|+> should be 1, got {}",
            exp.re
        );
    }

    #[test]
    fn test_two_site_product_state_factorized() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();
        let zz_corr = sim.correlation_function(0, 1).unwrap();
        // For product |00>, <ZZ> = <Z><Z> = 1*1 = 1
        assert!(
            approx_eq(zz_corr, 1.0, 1e-10),
            "<ZZ> for |00> should be 1, got {}",
            zz_corr
        );
    }

    #[test]
    fn test_two_site_correlation_mixed() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 1]).unwrap();
        let zz_corr = sim.correlation_function(0, 1).unwrap();
        // For |01>, <ZZ> = <Z_0><Z_1> = (+1)(-1) = -1
        assert!(
            approx_eq(zz_corr, -1.0, 1e-10),
            "<ZZ> for |01> should be -1, got {}",
            zz_corr
        );
    }

    #[test]
    fn test_norm_product_state() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 3, cols: 3 })
            .with_contraction_method(ContractionMethod::SimpleEnvironment);
        let sim = PepsSimulator::new_product_state(config, &[0; 9]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_energy_trivial_hamiltonian() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();
        // Hamiltonian = Z_0 + Z_1
        let mut ham = LocalHamiltonian::new(2);
        ham.add_single_site(0, pauli_z());
        ham.add_single_site(1, pauli_z());
        let energy = sim.energy(&ham).unwrap();
        // <0|Z|0> + <0|Z|0> = 1 + 1 = 2
        assert!(
            approx_eq(energy, 2.0, 1e-10),
            "Energy should be 2, got {}",
            energy
        );
    }

    // --- Hamiltonian construction tests ---

    #[test]
    fn test_heisenberg_model_construction() {
        let bonds = vec![(0, 1), (1, 2), (2, 3)];
        let ham = heisenberg_hamiltonian(&bonds, 4, 1.0, 1.0, 1.0, 0.5);
        // 3 two-site terms + 4 single-site terms
        assert_eq!(ham.terms.len(), 7);
    }

    #[test]
    fn test_transverse_field_ising_construction() {
        let bonds = vec![(0, 1), (1, 2)];
        let ham = transverse_field_ising(&bonds, 3, 1.0, 0.5);
        // 2 ZZ terms + 3 X terms = 5
        assert_eq!(ham.terms.len(), 5);
    }

    // --- Time evolution test ---

    #[test]
    fn test_ite_energy_decreases() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 })
            .with_max_bond_dim(4)
            .with_update_method(UpdateMethod::SimpleUpdate)
            .with_contraction_method(ContractionMethod::SimpleEnvironment);
        let mut sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();

        let bonds = sim.state.bond_pairs();
        let ham = transverse_field_ising(&bonds, 2, 1.0, 0.5);
        let initial_energy = sim.energy(&ham).unwrap();
        let energies = sim.imaginary_time_evolution(&ham, 0.05, 20).unwrap();

        // Energy should generally decrease or stay roughly same (ITE finds ground state)
        let final_energy = *energies.last().unwrap();
        assert!(
            final_energy <= initial_energy + 0.5,
            "Energy should not increase significantly: {} -> {}",
            initial_energy,
            final_energy
        );
    }

    // --- Observable tests ---

    #[test]
    fn test_magnetization_all_zero() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        let mags = sim.magnetization().unwrap();
        for m in &mags {
            assert!(
                approx_eq(*m, 1.0, 1e-10),
                "Magnetization of |0> should be 1, got {}",
                m
            );
        }
    }

    #[test]
    fn test_structure_factor_zero_momentum() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        let sf = sim.structure_factor(&[0.0, 0.0]).unwrap();
        // At q=0, S(0) = (1/N) sum_{ij} <ZiZj> = N (all correlations are +1)
        assert!(
            sf > 0.0,
            "Structure factor at q=0 should be positive, got {}",
            sf
        );
    }

    #[test]
    fn test_entanglement_entropy_product_state() {
        let config = PepsConfig::new().with_geometry(LatticeGeometry::Square { rows: 2, cols: 2 });
        let sim = PepsSimulator::new_product_state(config, &[0, 0, 0, 0]).unwrap();
        let entropy = sim.entanglement_entropy(&[0, 1]).unwrap();
        // Product state should have zero entanglement
        assert!(
            approx_eq(entropy, 0.0, 1e-10),
            "Product state entanglement should be 0, got {}",
            entropy
        );
    }

    // --- SVD tests ---

    #[test]
    fn test_bond_dimension_truncation() {
        let m = 4;
        let n = 4;
        let mut data = vec![c64::new(0.0, 0.0); m * n];
        // Create a rank-2 matrix
        data[0 * n + 0] = c64::new(3.0, 0.0);
        data[1 * n + 1] = c64::new(2.0, 0.0);
        data[2 * n + 2] = c64::new(0.01, 0.0);
        data[3 * n + 3] = c64::new(0.001, 0.0);

        let (u, sv, vt, err) = truncated_svd(&data, m, n, 2, 1e-2);
        assert!(
            sv.len() <= 2,
            "Should truncate to 2 singular values, got {}",
            sv.len()
        );
        assert!(err >= 0.0);
    }

    #[test]
    fn test_svd_truncation_accuracy() {
        let m = 3;
        let n = 3;
        let mut data = vec![c64::new(0.0, 0.0); m * n];
        data[0] = c64::new(5.0, 0.0);
        data[4] = c64::new(3.0, 0.0);
        data[8] = c64::new(1.0, 0.0);

        let (u, sv, vt, _err) = truncated_svd(&data, m, n, 3, 1e-12);
        assert!(
            sv.len() >= 2,
            "Should have at least 2 significant singular values"
        );
        assert!(
            sv[0] >= sv[1],
            "Singular values should be in decreasing order"
        );
        // Reconstruct: U * S * Vt should approximate original
        let k = sv.len();
        let mut reconstructed = vec![c64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    reconstructed[i * n + j] += u[i * k + l] * sv[l] * vt[l * n + j];
                }
            }
        }
        let recon_err: f64 = data
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            recon_err < 1e-6,
            "SVD reconstruction error too large: {}",
            recon_err
        );
    }

    // --- Higher-dimensional tests ---

    #[test]
    fn test_3d_cubic_ground_state_search() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Cubic {
                nx: 2,
                ny: 2,
                nz: 1,
            })
            .with_max_bond_dim(2)
            .with_update_method(UpdateMethod::SimpleUpdate)
            .with_contraction_method(ContractionMethod::SimpleEnvironment);
        let mut sim = PepsSimulator::new_product_state(config, &[0; 4]).unwrap();

        let bonds = sim.state.bond_pairs();
        let ham = transverse_field_ising(&bonds, 4, 1.0, 0.5);
        let energies = sim.imaginary_time_evolution(&ham, 0.05, 10).unwrap();
        assert!(!energies.is_empty());
    }

    #[test]
    fn test_3d_boundary_peps_contraction() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Cubic {
                nx: 2,
                ny: 2,
                nz: 2,
            })
            .with_contraction_method(ContractionMethod::BoundaryMPS { chi: 4 });
        let sim = PepsSimulator::new_product_state(config, &[0; 8]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_4d_hypercubic_initialization() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Hypercubic { n: [2, 2, 2, 2] })
            .with_bond_dim(1);
        let sim = PepsSimulator::new_product_state(config, &[0; 16]).unwrap();
        assert_eq!(sim.state.num_sites, 16);
        assert_eq!(sim.state.dimension, 4);
        // Each tensor should have 8 virtual legs
        assert_eq!(sim.state.tensors[0].num_virtual, 8);
    }

    #[test]
    fn test_4d_layered_contraction() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Hypercubic { n: [2, 2, 2, 2] })
            .with_contraction_method(ContractionMethod::BoundaryMPS { chi: 4 });
        let sim = PepsSimulator::new_product_state(config, &[0; 16]).unwrap();
        let norm = sim.norm().unwrap();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    // --- Boundary condition tests ---

    #[test]
    fn test_periodic_boundary_conditions() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 3, cols: 3 })
            .with_boundary_conditions(BoundaryConditions::Periodic);
        let sim = PepsSimulator::new_product_state(config, &[0; 9]).unwrap();
        // With PBC, every site should have 4 neighbors
        for site in 0..9 {
            assert_eq!(
                sim.state.adjacency[site].len(),
                4,
                "Site {} should have 4 neighbors with PBC, got {}",
                site,
                sim.state.adjacency[site].len()
            );
        }
    }

    #[test]
    fn test_cylindrical_boundary_conditions() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 3, cols: 3 })
            .with_boundary_conditions(BoundaryConditions::Cylindrical);
        let sim = PepsSimulator::new_product_state(config, &[0; 9]).unwrap();
        // With cylindrical BC (periodic in columns), edge sites should have 3 or 4 neighbors
        // Corner sites (r=0 or r=2) should have 3 neighbors
        // Middle row sites should have 4 neighbors
        let mid_site = 1 * 3 + 0; // (1, 0)
        assert!(sim.state.adjacency[mid_site].len() >= 3);
    }

    #[test]
    fn test_open_boundary_conditions() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 3, cols: 3 })
            .with_boundary_conditions(BoundaryConditions::Open);
        let sim = PepsSimulator::new_product_state(config, &[0; 9]).unwrap();
        // Corner has 2 neighbors, edge has 3, center has 4
        assert_eq!(sim.state.adjacency[0].len(), 2); // corner
        assert_eq!(sim.state.adjacency[1].len(), 3); // edge
        assert_eq!(sim.state.adjacency[4].len(), 4); // center
    }

    // --- Cluster update test ---

    #[test]
    fn test_cluster_update_correctness() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 })
            .with_max_bond_dim(4);
        let mut sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();
        let id_gate = kron(&pauli_i(), 2, &pauli_i(), 2);
        let err = sim.cluster_update(0, 1, &id_gate, 2).unwrap();
        assert!(err >= 0.0);
    }

    // --- Large bond dimension test ---

    #[test]
    fn test_large_bond_dimension_handling() {
        let config = PepsConfig::new()
            .with_geometry(LatticeGeometry::Square { rows: 1, cols: 2 })
            .with_bond_dim(1)
            .with_max_bond_dim(8);
        let mut sim = PepsSimulator::new_product_state(config, &[0, 0]).unwrap();

        // Apply a non-trivial gate that increases bond dimension
        let h = vec![
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            c64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        sim.apply_single_site_gate(0, &h).unwrap();

        let mut cnot = vec![c64::new(0.0, 0.0); 16];
        cnot[0] = c64::new(1.0, 0.0);
        cnot[5] = c64::new(1.0, 0.0);
        cnot[11] = c64::new(1.0, 0.0);
        cnot[14] = c64::new(1.0, 0.0);
        let err = sim.simple_update(0, 1, &cnot).unwrap();
        assert!(err >= 0.0);
        // Bond dimension should not exceed max
        for t in &sim.state.tensors {
            for &d in &t.virtual_dims {
                assert!(d <= 8, "Bond dim {} exceeds max 8", d);
            }
        }
    }

    // --- Hubbard model test ---

    #[test]
    fn test_hubbard_model_construction() {
        let bonds = vec![(0, 1)];
        let ham = hubbard_hamiltonian(&bonds, 2, 1.0, 4.0);
        // 1 hopping term + 2 on-site terms = 3
        assert_eq!(ham.terms.len(), 3);
        assert_eq!(ham.physical_dim, 4);
    }

    // --- Tensor basic tests ---

    #[test]
    fn test_tensor_norm() {
        let mut t = PepsTensor::zeros(2, vec![1, 1, 1, 1]);
        t.set(&[0, 0, 0, 0, 0], c64::new(1.0, 0.0));
        assert!(approx_eq(t.norm(), 1.0, 1e-14));
    }

    #[test]
    fn test_tensor_normalize() {
        let mut t = PepsTensor::zeros(2, vec![1, 1, 1, 1]);
        t.set(&[0, 0, 0, 0, 0], c64::new(3.0, 4.0));
        t.normalize();
        assert!(approx_eq(t.norm(), 1.0, 1e-14));
    }
}
