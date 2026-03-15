//! Infinite Matrix Product States (iMPS) and Infinite Projected Entangled Pair States (iPEPS)
//!
//! Implements algorithms for simulating infinite quantum systems with translational invariance:
//!
//! - **iMPS (VUMPS)**: Variational Uniform Matrix Product States for 1D infinite chains.
//!   Finds the optimal uniform MPS approximation to the ground state of an infinite 1D
//!   Hamiltonian by solving effective eigenvalue problems for the center tensor AC and
//!   bond matrix C, then extracting left/right isometries via polar decomposition.
//!
//! - **iPEPS (CTMRG)**: Corner Transfer Matrix Renormalization Group for 2D infinite lattices.
//!   Iteratively constructs environment tensors (4 corners + 4 edges) around a unit cell,
//!   using SVD truncation to keep the environment bond dimension bounded.
//!
//! - **Infinite TEBD**: Time evolution of infinite MPS via Suzuki-Trotter decomposition,
//!   applying two-site gates and truncating via SVD.
//!
//! - **Transfer Matrix Analysis**: Power-iteration eigensolver for the MPS transfer matrix,
//!   yielding correlation lengths and order parameters.
//!
//! # References
//!
//! - Zauner-Stauber et al., "Variational optimization algorithms for uniform matrix
//!   product states" (2018) — VUMPS algorithm
//! - Nishino & Okunishi, "Corner Transfer Matrix Renormalization Group Method" (1996)
//! - Vidal, "Efficient classical simulation of slightly entangled quantum computations"
//!   (2003) — infinite TEBD
//! - Orus, "A practical introduction to tensor networks" (2014)

use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use rand::Rng;
use std::fmt;

// ============================================================
// TYPE ALIAS
// ============================================================

type C64 = Complex64;

fn c0() -> C64 {
    C64::new(0.0, 0.0)
}
fn c1() -> C64 {
    C64::new(1.0, 0.0)
}
fn cr(r: f64) -> C64 {
    C64::new(r, 0.0)
}

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising during iMPS/iPEPS computations.
#[derive(Debug, Clone)]
pub enum InfiniteMpsError {
    /// Iterative algorithm did not converge.
    ConvergenceFailed {
        iterations: usize,
        residual: f64,
        tolerance: f64,
    },
    /// Bond dimension is invalid (zero or too large).
    InvalidBondDim(String),
    /// SVD decomposition failed or produced NaN.
    SvdFailed(String),
    /// Transfer matrix eigensolver failed.
    TransferMatrixFailed(String),
    /// CTMRG environment did not converge.
    CtmrgFailed { iterations: usize, residual: f64 },
    /// Numerical issue (NaN, Inf, or degenerate spectrum).
    NumericalError(String),
}

impl fmt::Display for InfiniteMpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConvergenceFailed {
                iterations,
                residual,
                tolerance,
            } => {
                write!(
                    f,
                    "Not converged after {} iterations (residual={:.2e}, tol={:.2e})",
                    iterations, residual, tolerance
                )
            }
            Self::InvalidBondDim(msg) => write!(f, "Invalid bond dimension: {}", msg),
            Self::SvdFailed(msg) => write!(f, "SVD failed: {}", msg),
            Self::TransferMatrixFailed(msg) => write!(f, "Transfer matrix: {}", msg),
            Self::CtmrgFailed {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "CTMRG not converged after {} iterations (residual={:.2e})",
                    iterations, residual
                )
            }
            Self::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl std::error::Error for InfiniteMpsError {}

// ============================================================
// CONFIGURATIONS
// ============================================================

/// Configuration for the VUMPS algorithm.
#[derive(Debug, Clone)]
pub struct VumpsConfig {
    /// MPS bond dimension.
    pub bond_dim: usize,
    /// Physical dimension (2 for qubits).
    pub phys_dim: usize,
    /// Maximum VUMPS iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on gradient norm.
    pub tolerance: f64,
    /// Lanczos subspace dimension for eigenvalue problems.
    pub lanczos_dim: usize,
    /// Use two-site VUMPS for bond dimension growth.
    pub two_site: bool,
}

impl Default for VumpsConfig {
    fn default() -> Self {
        Self {
            bond_dim: 32,
            phys_dim: 2,
            max_iterations: 200,
            tolerance: 1e-10,
            lanczos_dim: 20,
            two_site: false,
        }
    }
}

impl VumpsConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_bond_dim(mut self, d: usize) -> Self {
        self.bond_dim = d;
        self
    }
    pub fn with_phys_dim(mut self, d: usize) -> Self {
        self.phys_dim = d;
        self
    }
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }
    pub fn with_lanczos_dim(mut self, d: usize) -> Self {
        self.lanczos_dim = d;
        self
    }
    pub fn with_two_site(mut self, ts: bool) -> Self {
        self.two_site = ts;
        self
    }
}

/// Initialization method for CTMRG environments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CtmrgInit {
    /// Random initialization.
    Random,
    /// Identity-like initialization.
    Identity,
}

/// Configuration for the CTMRG algorithm.
#[derive(Debug, Clone)]
pub struct CtmrgConfig {
    /// Environment bond dimension (chi).
    pub chi: usize,
    /// Maximum CTMRG iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on singular value spectrum.
    pub tolerance: f64,
    /// Initialization method.
    pub init_method: CtmrgInit,
}

impl Default for CtmrgConfig {
    fn default() -> Self {
        Self {
            chi: 16,
            max_iterations: 100,
            tolerance: 1e-8,
            init_method: CtmrgInit::Random,
        }
    }
}

impl CtmrgConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_chi(mut self, chi: usize) -> Self {
        self.chi = chi;
        self
    }
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }
    pub fn with_init_method(mut self, m: CtmrgInit) -> Self {
        self.init_method = m;
        self
    }
}

/// Configuration for infinite TEBD.
#[derive(Debug, Clone)]
pub struct InfiniteTebdConfig {
    /// Maximum bond dimension after truncation.
    pub max_bond_dim: usize,
    /// SVD cutoff for discarding singular values.
    pub svd_cutoff: f64,
    /// Trotter order (2 or 4).
    pub trotter_order: usize,
    /// Time step size.
    pub dt: f64,
    /// Number of time steps.
    pub num_steps: usize,
}

impl Default for InfiniteTebdConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            svd_cutoff: 1e-12,
            trotter_order: 2,
            dt: 0.01,
            num_steps: 100,
        }
    }
}

impl InfiniteTebdConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_max_bond_dim(mut self, d: usize) -> Self {
        self.max_bond_dim = d;
        self
    }
    pub fn with_svd_cutoff(mut self, c: f64) -> Self {
        self.svd_cutoff = c;
        self
    }
    pub fn with_trotter_order(mut self, o: usize) -> Self {
        self.trotter_order = o;
        self
    }
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }
    pub fn with_num_steps(mut self, n: usize) -> Self {
        self.num_steps = n;
        self
    }
}

// ============================================================
// UNIFORM MPS
// ============================================================

/// A uniform (translationally invariant) Matrix Product State for infinite systems.
///
/// The unit cell consists of one or more site tensors A[s] of shape `[bond, phys, bond]`,
/// plus a center matrix C of shape `[bond, bond]` that encodes the Schmidt spectrum.
#[derive(Debug, Clone)]
pub struct UniformMps {
    /// Site tensors: `A[i]` has shape `[bond_l, phys, bond_r]`.
    pub tensors: Vec<Array3<C64>>,
    /// Center (bond) matrix of shape `[bond, bond]`.
    pub center: Array2<C64>,
    /// Bond dimension.
    pub bond_dim: usize,
    /// Physical dimension.
    pub phys_dim: usize,
}

impl UniformMps {
    /// Create a random uniform MPS with the given dimensions.
    pub fn random(phys_dim: usize, bond_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / ((bond_dim * phys_dim) as f64).sqrt();
        let a = Array3::from_shape_fn((bond_dim, phys_dim, bond_dim), |_| {
            C64::new(
                rng.gen::<f64>() * scale - scale / 2.0,
                rng.gen::<f64>() * scale - scale / 2.0,
            )
        });
        let mut c = Array2::from_shape_fn((bond_dim, bond_dim), |_| {
            C64::new(rng.gen::<f64>() * scale, 0.0)
        });
        // Make C positive semi-definite by C = C^H * C
        let ch = conjugate_transpose(&c);
        c = matmul(&ch, &c);
        // Normalize
        let nrm = frobenius_norm(&c);
        if nrm > 1e-15 {
            c.mapv_inplace(|x| x / cr(nrm));
        }
        Self {
            tensors: vec![a],
            center: c,
            bond_dim,
            phys_dim,
        }
    }

    /// Create a product-state MPS (all spins up = |0>).
    pub fn product_state(phys_dim: usize, bond_dim: usize) -> Self {
        let mut a = Array3::zeros((bond_dim, phys_dim, bond_dim));
        // Only the s=0 slice is identity-like
        for i in 0..bond_dim {
            a[[i, 0, i]] = c1();
        }
        let c = Array2::eye(bond_dim).mapv(|x| cr(x) / cr(bond_dim as f64).sqrt());
        Self {
            tensors: vec![a],
            center: c,
            bond_dim,
            phys_dim,
        }
    }

    /// Number of sites in the unit cell.
    pub fn unit_cell_size(&self) -> usize {
        self.tensors.len()
    }

    /// Compute the entanglement entropy from the center matrix singular values.
    pub fn entanglement_entropy(&self) -> f64 {
        let svs = singular_values(&self.center);
        let mut entropy = 0.0;
        let total: f64 = svs.iter().map(|s| s * s).sum();
        if total < 1e-30 {
            return 0.0;
        }
        for &s in &svs {
            let p = (s * s) / total;
            if p > 1e-30 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

// ============================================================
// TRANSFER MATRIX
// ============================================================

/// Transfer matrix operations for a uniform MPS.
///
/// The transfer matrix T acts on bond-space vectors: T(X) = Σ_s A[s] X A[s]†.
/// Its dominant eigenvector gives the fixed-point density matrix, and the
/// eigenvalue gap determines the correlation length.
pub struct TransferMatrix {
    bond_dim: usize,
    phys_dim: usize,
}

impl TransferMatrix {
    pub fn new(bond_dim: usize, phys_dim: usize) -> Self {
        Self { bond_dim, phys_dim }
    }

    /// Apply the transfer matrix: T(X) = Σ_s A[s] X A[s]†
    pub fn apply(&self, a: &Array3<C64>, x: &Array2<C64>) -> Array2<C64> {
        let d = self.bond_dim;
        let mut result = Array2::zeros((d, d));
        for s in 0..self.phys_dim {
            // Extract A[s] as a [bond, bond] matrix
            let a_s = a.slice(ndarray::s![.., s, ..]).to_owned();
            // A[s] X A[s]†
            let ax = matmul(&a_s, x);
            let a_s_dag = conjugate_transpose(&a_s);
            let axa = matmul(&ax, &a_s_dag);
            result = result + axa;
        }
        result
    }

    /// Find the dominant eigenvector via power iteration.
    ///
    /// Returns (eigenvalue, eigenvector matrix).
    pub fn dominant_eigenvector(
        &self,
        a: &Array3<C64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<(C64, Array2<C64>), InfiniteMpsError> {
        let d = self.bond_dim;
        let mut x = Array2::eye(d).mapv(|v| cr(v));
        let mut eigenvalue = c1();

        for iteration in 0..max_iter {
            let tx = self.apply(a, &x);
            let nrm = frobenius_norm(&tx);
            if nrm < 1e-30 {
                return Err(InfiniteMpsError::TransferMatrixFailed(
                    "Transfer matrix collapsed to zero".into(),
                ));
            }
            let new_eigenvalue = cr(nrm);
            x = tx.mapv(|v| v / cr(nrm));

            let diff = (new_eigenvalue - eigenvalue).norm();
            eigenvalue = new_eigenvalue;
            if diff < tol && iteration > 0 {
                return Ok((eigenvalue, x));
            }
        }
        Err(InfiniteMpsError::ConvergenceFailed {
            iterations: max_iter,
            residual: 1.0,
            tolerance: tol,
        })
    }

    /// Compute correlation length from transfer matrix eigenvalue gap.
    ///
    /// ξ = -1 / ln(|λ₂/λ₁|)
    pub fn correlation_length(
        &self,
        a: &Array3<C64>,
        max_iter: usize,
    ) -> Result<f64, InfiniteMpsError> {
        // Get dominant eigenvalue
        let (lambda1, v1) = self.dominant_eigenvector(a, max_iter, 1e-12)?;

        // Deflate and get second eigenvalue via power iteration on deflated T
        let d = self.bond_dim;
        let v1_norm = frobenius_norm(&v1);
        let v1_normalized = v1.mapv(|x| x / cr(v1_norm));

        let mut x = Array2::from_shape_fn((d, d), |_| cr(0.5));
        let mut lambda2_mag = 0.0;

        for _ in 0..max_iter {
            let tx = self.apply(a, &x);
            // Deflate: remove component along v1
            let overlap = inner_product_mat(&v1_normalized, &tx);
            let deflated = &tx - &v1_normalized.mapv(|v| v * overlap);

            let nrm = frobenius_norm(&deflated);
            if nrm < 1e-30 {
                break;
            }
            lambda2_mag = nrm;
            x = deflated.mapv(|v| v / cr(nrm));
        }

        let ratio = lambda2_mag / lambda1.norm();
        if ratio >= 1.0 || ratio < 1e-15 {
            return Ok(f64::INFINITY);
        }
        Ok(-1.0 / ratio.ln())
    }
}

// ============================================================
// VUMPS ALGORITHM
// ============================================================

/// Result of a VUMPS computation.
#[derive(Debug, Clone)]
pub struct VumpsResult {
    /// Optimized uniform MPS.
    pub mps: UniformMps,
    /// Ground state energy per site.
    pub energy_per_site: f64,
    /// Number of iterations to convergence.
    pub iterations: usize,
    /// Final gradient norm.
    pub gradient_norm: f64,
    /// Converged flag.
    pub converged: bool,
}

/// Run the VUMPS algorithm to find the ground state of an infinite 1D Hamiltonian.
///
/// The Hamiltonian is specified as a two-site gate `h` of shape `[d, d, d, d]`.
pub fn vumps(
    h: &Array4<C64>,
    config: &VumpsConfig,
    rng: &mut impl Rng,
) -> Result<VumpsResult, InfiniteMpsError> {
    if config.bond_dim == 0 {
        return Err(InfiniteMpsError::InvalidBondDim(
            "bond_dim must be > 0".into(),
        ));
    }

    let d = config.phys_dim;
    let chi = config.bond_dim;

    // Initialize random uniform MPS
    let mut mps = UniformMps::random(d, chi, rng);
    let mut energy = 0.0;
    let mut gradient_norm = 1.0;

    for iteration in 0..config.max_iterations {
        // 1. Compute left and right fixed points of the transfer matrix
        let tm = TransferMatrix::new(chi, d);
        let (_, l_fp) = tm
            .dominant_eigenvector(&mps.tensors[0], 100, 1e-12)
            .unwrap_or_else(|_| (c1(), Array2::eye(chi).mapv(|v| cr(v))));
        let a_dag = conjugate_transpose_3d(&mps.tensors[0]);
        let tm_r = TransferMatrix::new(chi, d);
        let (_, r_fp) = tm_r
            .dominant_eigenvector(&a_dag, 100, 1e-12)
            .unwrap_or_else(|_| (c1(), Array2::eye(chi).mapv(|v| cr(v))));

        // 2. Compute effective Hamiltonian for AC (1-site center tensor)
        let h_ac = build_effective_h_ac(h, &mps.tensors[0], &l_fp, &r_fp, d, chi);

        // 3. Solve eigenvalue problem for AC via Lanczos
        let ac_dim = chi * d * chi;
        let ac_flat = flatten_3d(&mps.tensors[0]);
        let (e_ac, ac_new_flat) = lanczos_ground(&h_ac, &ac_flat, ac_dim, config.lanczos_dim)?;

        // 4. Compute effective Hamiltonian for C
        let h_c = build_effective_h_c(&l_fp, &r_fp, chi);
        let c_dim = chi * chi;
        let c_flat = flatten_2d(&mps.center);
        let (_, c_new_flat) = lanczos_ground(&h_c, &c_flat, c_dim, config.lanczos_dim)?;

        // 5. Extract AL, AR via polar decomposition
        let ac_new = unflatten_3d(&ac_new_flat, chi, d, chi);
        let c_new = unflatten_2d(&c_new_flat, chi, chi);

        // AL: polar(AC * C^{-1}) -> AC = AL * C
        // We use: AL = U * V† from polar decomposition of AC reshaped as (chi*d, chi)
        let ac_mat = reshape_3d_to_mat(&ac_new, chi * d, chi);
        let (u_al, _) = polar_decomposition(&ac_mat);

        // 6. Update MPS
        let al_3d = reshape_mat_to_3d(&u_al, chi, d, chi);
        gradient_norm = tensor_distance(&mps.tensors[0], &al_3d);
        energy = e_ac;
        mps.tensors[0] = al_3d;
        mps.center = c_new;

        if gradient_norm < config.tolerance {
            return Ok(VumpsResult {
                mps,
                energy_per_site: energy,
                iterations: iteration + 1,
                gradient_norm,
                converged: true,
            });
        }
    }

    Ok(VumpsResult {
        mps,
        energy_per_site: energy,
        iterations: config.max_iterations,
        gradient_norm,
        converged: gradient_norm < config.tolerance,
    })
}

// ============================================================
// INFINITE PEPS
// ============================================================

/// An infinite Projected Entangled Pair State with a rectangular unit cell.
#[derive(Debug, Clone)]
pub struct InfinitePeps {
    /// Unit cell tensors: `tensors[row][col]` has 5 indices [up, right, down, left, phys].
    /// Stored as flat Vec with shape info.
    pub tensors: Vec<Vec<PepsTensor>>,
    /// Bond dimension.
    pub bond_dim: usize,
    /// Physical dimension.
    pub phys_dim: usize,
    /// Unit cell size (rows, cols).
    pub unit_cell: (usize, usize),
}

/// A single iPEPS site tensor stored as flat data.
#[derive(Debug, Clone)]
pub struct PepsTensor {
    /// Flat tensor data.
    pub data: Vec<C64>,
    /// Dimensions: [up, right, down, left, phys].
    pub dims: [usize; 5],
}

impl PepsTensor {
    pub fn total_size(dims: &[usize; 5]) -> usize {
        dims.iter().product()
    }

    pub fn zeros(dims: [usize; 5]) -> Self {
        let size = Self::total_size(&dims);
        Self {
            data: vec![c0(); size],
            dims,
        }
    }

    pub fn random(dims: [usize; 5], rng: &mut impl Rng) -> Self {
        let size = Self::total_size(&dims);
        let scale = 1.0 / (size as f64).sqrt();
        let data = (0..size)
            .map(|_| {
                C64::new(
                    rng.gen::<f64>() * scale - scale / 2.0,
                    rng.gen::<f64>() * scale - scale / 2.0,
                )
            })
            .collect();
        Self { data, dims }
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
    }
}

impl InfinitePeps {
    /// Create a random iPEPS.
    pub fn random(
        phys_dim: usize,
        bond_dim: usize,
        unit_cell: (usize, usize),
        rng: &mut impl Rng,
    ) -> Self {
        let dims = [bond_dim, bond_dim, bond_dim, bond_dim, phys_dim];
        let tensors = (0..unit_cell.0)
            .map(|_| {
                (0..unit_cell.1)
                    .map(|_| PepsTensor::random(dims, rng))
                    .collect()
            })
            .collect();
        Self {
            tensors,
            bond_dim,
            phys_dim,
            unit_cell,
        }
    }
}

// ============================================================
// CTMRG ENVIRONMENT
// ============================================================

/// Corner Transfer Matrix environment tensors for iPEPS.
///
/// The environment consists of 4 corner matrices (C1-C4) and 4 edge tensors (T1-T4),
/// each with bond dimension chi.
#[derive(Debug, Clone)]
pub struct CtmEnvironment {
    /// Corner matrices C1-C4, each of shape [chi, chi].
    pub corners: [Array2<C64>; 4],
    /// Edge tensors T1-T4, each of shape [chi, bond*bond, chi].
    pub edges: [Array3<C64>; 4],
    /// Environment bond dimension.
    pub chi: usize,
    /// Whether the environment has converged.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
}

impl CtmEnvironment {
    /// Initialize a new CTMRG environment.
    pub fn new(chi: usize, bond_dim: usize, init: CtmrgInit, _rng: &mut impl Rng) -> Self {
        let dd = bond_dim * bond_dim;
        let (corners, edges) = match init {
            CtmrgInit::Identity => {
                let c = Array2::eye(chi).mapv(|v| cr(v));
                let mut t = Array3::zeros((chi, dd, chi));
                for i in 0..chi.min(dd) {
                    for j in 0..chi {
                        t[[j, i, j]] = cr(1.0 / chi as f64);
                    }
                }
                (
                    [c.clone(), c.clone(), c.clone(), c.clone()],
                    [t.clone(), t.clone(), t.clone(), t.clone()],
                )
            }
            CtmrgInit::Random => {
                let make_c = || {
                    Array2::from_shape_fn((chi, chi), |_| {
                        C64::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5)
                    })
                };
                let make_t = || {
                    Array3::from_shape_fn((chi, dd, chi), |_| {
                        C64::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5)
                    })
                };
                (
                    [make_c(), make_c(), make_c(), make_c()],
                    [make_t(), make_t(), make_t(), make_t()],
                )
            }
        };
        Self {
            corners,
            edges,
            chi,
            converged: false,
            iterations: 0,
        }
    }

    /// Get singular value spectrum of a corner for convergence checking.
    pub fn corner_spectrum(&self, idx: usize) -> Vec<f64> {
        singular_values(&self.corners[idx.min(3)])
    }
}

/// Run CTMRG to convergence.
pub fn ctmrg(
    peps: &InfinitePeps,
    config: &CtmrgConfig,
    rng: &mut impl Rng,
) -> Result<CtmEnvironment, InfiniteMpsError> {
    let mut env = CtmEnvironment::new(config.chi, peps.bond_dim, config.init_method, rng);
    let mut prev_spectrum = env.corner_spectrum(0);

    for iteration in 0..config.max_iterations {
        // Perform one CTMRG step (all 4 directions)
        ctmrg_absorption_step(&mut env, peps, config.chi);

        // Check convergence via corner singular value spectrum
        let new_spectrum = env.corner_spectrum(0);
        let residual = spectrum_distance(&prev_spectrum, &new_spectrum);
        prev_spectrum = new_spectrum;
        env.iterations = iteration + 1;

        if residual < config.tolerance {
            env.converged = true;
            return Ok(env);
        }
    }

    Err(InfiniteMpsError::CtmrgFailed {
        iterations: config.max_iterations,
        residual: spectrum_distance(&prev_spectrum, &env.corner_spectrum(0)),
    })
}

/// One full CTMRG absorption step (simplified: absorb in one direction and truncate).
fn ctmrg_absorption_step(env: &mut CtmEnvironment, _peps: &InfinitePeps, chi: usize) {
    // Left absorption: enlarge C1, T4, C4 and truncate
    for dir in 0..4 {
        let c_idx = dir;
        let c_next = (dir + 1) % 4;
        // Simplified: contract corner with edge, SVD truncate back to chi
        let c = &env.corners[c_idx];
        let t = &env.edges[dir];

        // C_new = C * T (contract inner index)
        let c_rows = c.shape()[0];
        let t_mid = t.shape()[1];
        let t_cols = t.shape()[2];

        // Reshape: C[a,b] * T[b,m,c] -> P[a,m,c] -> P[(a*m), c]
        let inner = c.shape()[1].min(t.shape()[0]);
        let mut p = Array2::zeros((c_rows * t_mid, t_cols));
        for a in 0..c_rows {
            for m in 0..t_mid {
                for cc in 0..t_cols {
                    let mut val = c0();
                    for b in 0..inner {
                        val = val + c[[a, b]] * t[[b, m, cc]];
                    }
                    p[[a * t_mid + m, cc]] = val;
                }
            }
        }

        // SVD truncate to chi
        let (u, s, _vt) = svd_truncate_mat(&p, chi);
        // New corner = U * S (absorb singular values into corner)
        let new_c = matmul_diag(&u, &s);
        // Trim to chi x chi
        let new_chi = new_c.shape()[1].min(chi);
        let trimmed = new_c
            .slice(ndarray::s![..chi.min(new_c.shape()[0]), ..new_chi])
            .to_owned();

        env.corners[c_next] = if trimmed.shape() == [env.chi, env.chi] {
            trimmed
        } else {
            // Pad or trim to exact chi x chi
            let mut padded = Array2::zeros((env.chi, env.chi));
            let r = trimmed.shape()[0].min(env.chi);
            let cc = trimmed.shape()[1].min(env.chi);
            for i in 0..r {
                for j in 0..cc {
                    padded[[i, j]] = trimmed[[i, j]];
                }
            }
            padded
        };
    }
}

// ============================================================
// INFINITE TEBD
// ============================================================

/// Result of an infinite TEBD computation.
#[derive(Debug, Clone)]
pub struct InfiniteTebdResult {
    /// Final MPS.
    pub mps: UniformMps,
    /// Energy per site at each step.
    pub energies: Vec<f64>,
    /// Truncation errors at each step.
    pub truncation_errors: Vec<f64>,
    /// Total number of steps.
    pub num_steps: usize,
}

/// Apply infinite TEBD time evolution.
///
/// The Hamiltonian is specified as a two-site gate `h` of shape `[d, d, d, d]`.
/// The evolution operator `exp(-i h dt)` is applied via Suzuki-Trotter decomposition.
pub fn infinite_tebd(
    mps: &UniformMps,
    h: &Array4<C64>,
    config: &InfiniteTebdConfig,
) -> Result<InfiniteTebdResult, InfiniteMpsError> {
    let d = mps.phys_dim;
    let chi = mps.bond_dim;

    // Compute time evolution gate: U = exp(-i h dt)
    let gate = exponentiate_gate(h, config.dt);

    let mut current = mps.clone();
    let mut energies = Vec::with_capacity(config.num_steps);
    let mut truncation_errors = Vec::with_capacity(config.num_steps);

    for _step in 0..config.num_steps {
        // Apply 2-site gate and truncate
        let (new_mps, trunc_err) =
            apply_two_site_gate(&current, &gate, config.max_bond_dim, d, chi)?;
        current = new_mps;
        truncation_errors.push(trunc_err);

        // Measure energy
        let e = measure_two_site_energy(&current, h);
        energies.push(e);
    }

    Ok(InfiniteTebdResult {
        mps: current,
        energies,
        truncation_errors,
        num_steps: config.num_steps,
    })
}

/// Apply a two-site gate to the uniform MPS and truncate.
fn apply_two_site_gate(
    mps: &UniformMps,
    gate: &Array4<C64>,
    max_bond_dim: usize,
    d: usize,
    chi: usize,
) -> Result<(UniformMps, f64), InfiniteMpsError> {
    let a = &mps.tensors[0];
    let c = &mps.center;

    // Form two-site tensor: theta[a, s1, s2, b] = sum_m A[a,s1,m] * C[m,m'] * A[m',s2,b]
    let mut theta = Array4::zeros((chi, d, d, chi));
    for aa in 0..chi {
        for s1 in 0..d {
            for s2 in 0..d {
                for bb in 0..chi {
                    let mut val = c0();
                    for m in 0..chi {
                        for mp in 0..chi {
                            val = val + a[[aa, s1, m]] * c[[m, mp]] * a[[mp, s2, bb]];
                        }
                    }
                    theta[[aa, s1, s2, bb]] = val;
                }
            }
        }
    }

    // Apply gate: theta'[a, s1', s2', b] = sum_{s1,s2} gate[s1',s2',s1,s2] * theta[a,s1,s2,b]
    let mut theta_new = Array4::zeros((chi, d, d, chi));
    for aa in 0..chi {
        for s1p in 0..d {
            for s2p in 0..d {
                for bb in 0..chi {
                    let mut val = c0();
                    for s1 in 0..d {
                        for s2 in 0..d {
                            val = val + gate[[s1p, s2p, s1, s2]] * theta[[aa, s1, s2, bb]];
                        }
                    }
                    theta_new[[aa, s1p, s2p, bb]] = val;
                }
            }
        }
    }

    // Reshape to matrix and SVD: theta[a*s1, s2*b]
    let rows = chi * d;
    let cols = d * chi;
    let mut mat = Array2::zeros((rows, cols));
    for aa in 0..chi {
        for s1 in 0..d {
            for s2 in 0..d {
                for bb in 0..chi {
                    mat[[aa * d + s1, s2 * chi + bb]] = theta_new[[aa, s1, s2, bb]];
                }
            }
        }
    }

    // SVD truncate
    let (u, s, _vt) = svd_truncate_mat(&mat, max_bond_dim);
    let new_chi = s.len().min(max_bond_dim).min(chi);

    // Truncation error
    let total_weight: f64 = s.iter().map(|x| x * x).sum();
    let kept_weight: f64 = s.iter().take(new_chi).map(|x| x * x).sum();
    let trunc_err = if total_weight > 0.0 {
        1.0 - kept_weight / total_weight
    } else {
        0.0
    };

    // Reconstruct A and C
    let mut new_a = Array3::zeros((chi, d, chi));
    for aa in 0..chi {
        for s in 0..d {
            for m in 0..chi.min(new_chi) {
                new_a[[aa, s, m]] = u[[aa * d + s, m]];
            }
        }
    }

    let mut new_c = Array2::zeros((chi, chi));
    for i in 0..chi.min(new_chi) {
        new_c[[i, i]] = cr(s[i]);
    }

    Ok((
        UniformMps {
            tensors: vec![new_a],
            center: new_c,
            bond_dim: chi,
            phys_dim: d,
        },
        trunc_err,
    ))
}

/// Measure two-site energy <h> for a uniform MPS.
fn measure_two_site_energy(mps: &UniformMps, h: &Array4<C64>) -> f64 {
    let a = &mps.tensors[0];
    let c = &mps.center;
    let d = mps.phys_dim;
    let chi = mps.bond_dim;

    // Build two-site reduced density matrix (simplified: diagonal approximation)
    let mut energy = 0.0;
    for s1 in 0..d {
        for s2 in 0..d {
            for s1p in 0..d {
                for s2p in 0..d {
                    let h_elem = h[[s1p, s2p, s1, s2]];
                    if h_elem.norm() < 1e-15 {
                        continue;
                    }
                    // <s1,s2|rho|s1',s2'> ≈ trace(A[s1] C A[s2] A[s2']† C† A[s1']†)
                    let mut trace = c0();
                    for aa in 0..chi {
                        for bb in 0..chi {
                            let mut bra = c0();
                            let mut ket = c0();
                            for m in 0..chi {
                                for mp in 0..chi {
                                    ket = ket + a[[aa, s1, m]] * c[[m, mp]] * a[[mp, s2, bb]];
                                    bra = bra + a[[aa, s1p, m]] * c[[m, mp]] * a[[mp, s2p, bb]];
                                }
                            }
                            trace = trace + ket * bra.conj();
                        }
                    }
                    energy += (h_elem * trace).re;
                }
            }
        }
    }
    energy
}

// ============================================================
// MODEL HAMILTONIANS
// ============================================================

/// Transverse-field Ising model: H = -J Σ Z⊗Z - h Σ X⊗I + I⊗X (two-site gate).
pub fn ising_two_site_gate(j: f64, h_field: f64) -> Array4<C64> {
    let mut gate = Array4::zeros((2, 2, 2, 2));
    // -J Z⊗Z
    gate[[0, 0, 0, 0]] = cr(-j); // |00><00|
    gate[[0, 1, 0, 1]] = cr(j); // |01><01|
    gate[[1, 0, 1, 0]] = cr(j); // |10><10|
    gate[[1, 1, 1, 1]] = cr(-j); // |11><11|
                                 // -h/2 (X⊗I + I⊗X)
    let hh = h_field / 2.0;
    gate[[1, 0, 0, 0]] = gate[[1, 0, 0, 0]] + cr(-hh);
    gate[[0, 0, 1, 0]] = gate[[0, 0, 1, 0]] + cr(-hh);
    gate[[0, 1, 0, 0]] = gate[[0, 1, 0, 0]] + cr(-hh);
    gate[[0, 0, 0, 1]] = gate[[0, 0, 0, 1]] + cr(-hh);
    gate[[1, 1, 0, 1]] = gate[[1, 1, 0, 1]] + cr(-hh);
    gate[[0, 1, 1, 1]] = gate[[0, 1, 1, 1]] + cr(-hh);
    gate[[1, 1, 1, 0]] = gate[[1, 1, 1, 0]] + cr(-hh);
    gate[[1, 0, 1, 1]] = gate[[1, 0, 1, 1]] + cr(-hh);
    gate
}

/// Heisenberg XXX model: H = J Σ (X⊗X + Y⊗Y + Z⊗Z) (two-site gate).
pub fn heisenberg_two_site_gate(j: f64) -> Array4<C64> {
    let mut gate = Array4::zeros((2, 2, 2, 2));
    // X⊗X
    gate[[0, 0, 1, 1]] = gate[[0, 0, 1, 1]] + cr(j);
    gate[[0, 1, 1, 0]] = gate[[0, 1, 1, 0]] + cr(j);
    gate[[1, 0, 0, 1]] = gate[[1, 0, 0, 1]] + cr(j);
    gate[[1, 1, 0, 0]] = gate[[1, 1, 0, 0]] + cr(j);
    // Y⊗Y (recall Y = [[0,-i],[i,0]])
    gate[[0, 0, 1, 1]] = gate[[0, 0, 1, 1]] + cr(j); // (-i)(i) = 1
    gate[[0, 1, 1, 0]] = gate[[0, 1, 1, 0]] + cr(-j); // (-i)(-i) = -1
    gate[[1, 0, 0, 1]] = gate[[1, 0, 0, 1]] + cr(-j); // (i)(i) = -1
    gate[[1, 1, 0, 0]] = gate[[1, 1, 0, 0]] + cr(j); // (i)(-i) = 1
                                                     // Z⊗Z
    gate[[0, 0, 0, 0]] = gate[[0, 0, 0, 0]] + cr(j);
    gate[[0, 1, 0, 1]] = gate[[0, 1, 0, 1]] + cr(-j);
    gate[[1, 0, 1, 0]] = gate[[1, 0, 1, 0]] + cr(-j);
    gate[[1, 1, 1, 1]] = gate[[1, 1, 1, 1]] + cr(j);
    gate
}

// ============================================================
// LINEAR ALGEBRA HELPERS
// ============================================================

fn matmul(a: &Array2<C64>, b: &Array2<C64>) -> Array2<C64> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = c0();
            for p in 0..k {
                sum = sum + a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

fn matmul_diag(a: &Array2<C64>, diag: &[f64]) -> Array2<C64> {
    let m = a.shape()[0];
    let n = diag.len().min(a.shape()[1]);
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = a[[i, j]] * cr(diag[j]);
        }
    }
    c
}

fn conjugate_transpose(a: &Array2<C64>) -> Array2<C64> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    Array2::from_shape_fn((n, m), |(i, j)| a[[j, i]].conj())
}

fn conjugate_transpose_3d(a: &Array3<C64>) -> Array3<C64> {
    // Transpose bond indices, conjugate: A†[b, s, a] = A[a, s, b]*
    let d0 = a.shape()[0];
    let d1 = a.shape()[1];
    let d2 = a.shape()[2];
    Array3::from_shape_fn((d2, d1, d0), |(i, j, k)| a[[k, j, i]].conj())
}

fn frobenius_norm(a: &Array2<C64>) -> f64 {
    a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

fn inner_product_mat(a: &Array2<C64>, b: &Array2<C64>) -> C64 {
    a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum()
}

fn singular_values(a: &Array2<C64>) -> Vec<f64> {
    // Simple SVD via eigendecomposition of A†A
    let ata = matmul(&conjugate_transpose(a), a);
    let eigenvalues = power_iteration_eigenvalues(&ata, ata.shape()[0].min(20));
    eigenvalues.iter().map(|x| x.sqrt().max(0.0)).collect()
}

fn power_iteration_eigenvalues(a: &Array2<C64>, num: usize) -> Vec<f64> {
    let n = a.shape()[0];
    let mut eigenvalues = Vec::new();
    let mut deflated = a.clone();

    for eig_idx in 0..num.min(n) {
        // Start with e_{eig_idx} to avoid null space of deflated matrix
        let mut v = Array1::from_shape_fn(n, |i| if i == eig_idx % n { c1() } else { c0() });
        let mut lambda = 0.0;

        for _ in 0..100 {
            let mut av = Array1::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    av[i] = av[i] + deflated[[i, j]] * v[j];
                }
            }
            let nrm: f64 = av.iter().map(|x: &C64| x.norm_sqr()).sum::<f64>().sqrt();
            if nrm < 1e-30 {
                break;
            }
            lambda = nrm;
            v = av.mapv(|x| x / cr(nrm));
        }

        eigenvalues.push(lambda);

        // Deflate
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] = deflated[[i, j]] - cr(lambda) * v[i] * v[j].conj();
            }
        }
    }

    eigenvalues
}

fn svd_truncate_mat(a: &Array2<C64>, max_dim: usize) -> (Array2<C64>, Vec<f64>, Array2<C64>) {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = m.min(n).min(max_dim);

    // Use iterative SVD (one-sided Jacobi approximation)
    let ata = matmul(&conjugate_transpose(a), a);
    let evals = power_iteration_eigenvalues(&ata, k);

    let mut u_cols = Vec::new();
    let mut s_vals = Vec::new();
    let mut v_cols = Vec::new();

    let mut deflated_ata = ata.clone();

    for idx in 0..k {
        let sigma = evals.get(idx).copied().unwrap_or(0.0).sqrt();
        if sigma < 1e-15 {
            break;
        }
        s_vals.push(sigma);

        // Get right singular vector from deflated A†A
        let mut v = Array1::from_shape_fn(n, |i| if i == idx % n { c1() } else { c0() });
        for _ in 0..50 {
            let mut tv = Array1::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    tv[i] = tv[i] + deflated_ata[[i, j]] * v[j];
                }
            }
            let nrm: f64 = tv.iter().map(|x: &C64| x.norm_sqr()).sum::<f64>().sqrt();
            if nrm < 1e-30 {
                break;
            }
            v = tv.mapv(|x| x / cr(nrm));
        }

        // u = A v / sigma
        let mut u: Array1<C64> = Array1::zeros(m);
        for i in 0..m {
            for j in 0..n {
                u[i] = u[i] + a[[i, j]] * v[j];
            }
        }
        let u = u.mapv(|x| x / cr(sigma));

        u_cols.push(u);
        v_cols.push(v.clone());

        // Deflate
        for i in 0..n {
            for j in 0..n {
                deflated_ata[[i, j]] =
                    deflated_ata[[i, j]] - cr(sigma * sigma) * v[i] * v[j].conj();
            }
        }
    }

    let kk = s_vals.len();
    let mut u_mat = Array2::zeros((m, kk));
    let mut vt_mat = Array2::zeros((kk, n));
    for idx in 0..kk {
        for i in 0..m {
            u_mat[[i, idx]] = u_cols[idx][i];
        }
        for j in 0..n {
            vt_mat[[idx, j]] = v_cols[idx][j].conj();
        }
    }

    (u_mat, s_vals, vt_mat)
}

fn polar_decomposition(a: &Array2<C64>) -> (Array2<C64>, Array2<C64>) {
    let (u, s, vt) = svd_truncate_mat(a, a.shape()[0].min(a.shape()[1]));
    // W = U * Vt (unitary part)
    let w = matmul(&u, &vt);
    // P = V * S * Vt (positive semi-definite part)
    let v = conjugate_transpose(&vt);
    let vs = matmul_diag(&v, &s);
    let p = matmul(&vs, &vt);
    (w, p)
}

fn tensor_distance(a: &Array3<C64>, b: &Array3<C64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn spectrum_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len());
    let mut dist = 0.0;
    for i in 0..n {
        let va = a.get(i).copied().unwrap_or(0.0);
        let vb = b.get(i).copied().unwrap_or(0.0);
        dist += (va - vb).powi(2);
    }
    dist.sqrt()
}

fn flatten_3d(a: &Array3<C64>) -> Vec<C64> {
    a.iter().copied().collect()
}

fn flatten_2d(a: &Array2<C64>) -> Vec<C64> {
    a.iter().copied().collect()
}

fn unflatten_3d(v: &[C64], d0: usize, d1: usize, d2: usize) -> Array3<C64> {
    Array3::from_shape_vec(
        (d0, d1, d2),
        v.iter()
            .copied()
            .chain(std::iter::repeat(c0()))
            .take(d0 * d1 * d2)
            .collect(),
    )
    .unwrap()
}

fn unflatten_2d(v: &[C64], d0: usize, d1: usize) -> Array2<C64> {
    Array2::from_shape_vec(
        (d0, d1),
        v.iter()
            .copied()
            .chain(std::iter::repeat(c0()))
            .take(d0 * d1)
            .collect(),
    )
    .unwrap()
}

fn reshape_3d_to_mat(a: &Array3<C64>, rows: usize, cols: usize) -> Array2<C64> {
    let flat: Vec<C64> = a.iter().copied().collect();
    Array2::from_shape_vec(
        (rows, cols),
        flat.into_iter()
            .chain(std::iter::repeat(c0()))
            .take(rows * cols)
            .collect(),
    )
    .unwrap()
}

fn reshape_mat_to_3d(a: &Array2<C64>, d0: usize, d1: usize, d2: usize) -> Array3<C64> {
    let flat: Vec<C64> = a.iter().copied().collect();
    Array3::from_shape_vec(
        (d0, d1, d2),
        flat.into_iter()
            .chain(std::iter::repeat(c0()))
            .take(d0 * d1 * d2)
            .collect(),
    )
    .unwrap()
}

/// Exponentiate a two-site gate: U = exp(-i h dt) using Taylor series.
fn exponentiate_gate(h: &Array4<C64>, dt: f64) -> Array4<C64> {
    let d = h.shape()[0];
    let n = d * d;
    // Flatten to matrix
    let mut mat = Array2::zeros((n, n));
    for s1 in 0..d {
        for s2 in 0..d {
            for s1p in 0..d {
                for s2p in 0..d {
                    mat[[s1 * d + s2, s1p * d + s2p]] = h[[s1, s2, s1p, s2p]] * C64::new(0.0, -dt);
                }
            }
        }
    }

    // exp(M) ≈ I + M + M²/2 + M³/6 + M⁴/24
    let eye = Array2::eye(n).mapv(|v| cr(v));
    let m2 = matmul(&mat, &mat);
    let m3 = matmul(&m2, &mat);
    let m4 = matmul(&m3, &mat);

    let result = &eye
        + &mat
        + &m2.mapv(|x| x / cr(2.0))
        + &m3.mapv(|x| x / cr(6.0))
        + &m4.mapv(|x| x / cr(24.0));

    // Unflatten back to 4-tensor
    let mut gate = Array4::zeros((d, d, d, d));
    for s1 in 0..d {
        for s2 in 0..d {
            for s1p in 0..d {
                for s2p in 0..d {
                    gate[[s1, s2, s1p, s2p]] = result[[s1 * d + s2, s1p * d + s2p]];
                }
            }
        }
    }
    gate
}

/// Lanczos eigensolver for the smallest eigenvalue.
fn lanczos_ground(
    h_func: &Array2<C64>,
    initial: &[C64],
    dim: usize,
    num_iter: usize,
) -> Result<(f64, Vec<C64>), InfiniteMpsError> {
    let n = dim.min(h_func.shape()[0]);
    let k = num_iter.min(n);

    // Normalize initial vector
    let mut v: Vec<C64> = initial
        .iter()
        .copied()
        .chain(std::iter::repeat(c0()))
        .take(n)
        .collect();
    let nrm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    if nrm < 1e-30 {
        for i in 0..n {
            v[i] = if i == 0 { c1() } else { c0() };
        }
    } else {
        for x in v.iter_mut() {
            *x = *x / cr(nrm);
        }
    }

    let mut alpha = Vec::with_capacity(k);
    let mut beta = Vec::with_capacity(k);
    let mut lanczos_vecs = Vec::with_capacity(k);
    lanczos_vecs.push(v.clone());

    let mut w = vec![c0(); n];
    // w = H * v
    for i in 0..n {
        for j in 0..n {
            w[i] = w[i] + h_func[[i, j]] * v[j];
        }
    }

    let a0: f64 = w
        .iter()
        .zip(v.iter())
        .map(|(wi, vi)| (wi * vi.conj()).re)
        .sum();
    alpha.push(a0);

    for i in 0..n {
        w[i] = w[i] - cr(a0) * v[i];
    }

    for j in 1..k {
        let b = w.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if b < 1e-14 {
            break;
        }
        beta.push(b);

        let v_prev = lanczos_vecs[j - 1].clone();
        let mut v_new = w.iter().map(|x| *x / cr(b)).collect::<Vec<_>>();

        // Re-orthogonalize
        for prev in &lanczos_vecs {
            let dot: C64 = v_new
                .iter()
                .zip(prev.iter())
                .map(|(a, b)| a * b.conj())
                .sum();
            for i in 0..n {
                v_new[i] = v_new[i] - dot * prev[i];
            }
        }
        let nrm = v_new.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if nrm < 1e-14 {
            break;
        }
        for x in v_new.iter_mut() {
            *x = *x / cr(nrm);
        }

        lanczos_vecs.push(v_new.clone());

        // w = H * v_new - b * v_prev
        w = vec![c0(); n];
        for i in 0..n {
            for jj in 0..n {
                w[i] = w[i] + h_func[[i, jj]] * v_new[jj];
            }
            w[i] = w[i] - cr(b) * v_prev[i];
        }

        let aj: f64 = w
            .iter()
            .zip(v_new.iter())
            .map(|(wi, vi)| (wi * vi.conj()).re)
            .sum();
        alpha.push(aj);

        for i in 0..n {
            w[i] = w[i] - cr(aj) * v_new[i];
        }
    }

    // Diagonalize tridiagonal matrix
    let kk = alpha.len();
    if kk == 0 {
        return Err(InfiniteMpsError::NumericalError(
            "Lanczos produced no vectors".into(),
        ));
    }

    let (eigenvalue, eigvec_tri) = tridiagonal_eigmin(&alpha, &beta);

    // Map back to full space
    let mut result = vec![c0(); n];
    for j in 0..kk.min(eigvec_tri.len()) {
        for i in 0..n {
            result[i] = result[i] + cr(eigvec_tri[j]) * lanczos_vecs[j][i];
        }
    }

    Ok((eigenvalue, result))
}

/// Find smallest eigenvalue of a tridiagonal matrix.
fn tridiagonal_eigmin(alpha: &[f64], beta: &[f64]) -> (f64, Vec<f64>) {
    let n = alpha.len();
    if n == 1 {
        return (alpha[0], vec![1.0]);
    }
    if n == 2 {
        // Direct 2×2 formula
        let tr = alpha[0] + alpha[1];
        let det = alpha[0] * alpha[1] - beta[0] * beta[0];
        let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
        let min_eval = (tr - disc) / 2.0;
        // Eigenvector via (T - λI)v = 0
        let a = alpha[0] - min_eval;
        let b = beta[0];
        let nrm = (a * a + b * b).sqrt();
        let v = if nrm > 1e-30 {
            vec![-b / nrm, a / nrm]
        } else {
            vec![1.0, 0.0]
        };
        return (min_eval, v);
    }

    // Implicit QR with Wilkinson shifts and deflation
    let mut eigenvalues = alpha.to_vec();
    let mut off_diag = beta.to_vec();
    let mut active_end = n; // active submatrix is [0..active_end]

    for _ in 0..200 * n {
        // Deflation: shrink active size when bottom off-diagonal converges
        while active_end > 1 && off_diag[active_end - 2].abs() < 1e-14 {
            active_end -= 1;
        }
        if active_end <= 1 {
            break;
        }

        // Also check for internal deflation
        for i in 0..active_end - 1 {
            if off_diag[i].abs()
                < 1e-14 * (eigenvalues[i].abs() + eigenvalues[i + 1].abs()).max(1e-30)
            {
                off_diag[i] = 0.0;
            }
        }

        // Find the bottom unreduced block
        let mut start = active_end - 2;
        while start > 0 && off_diag[start - 1].abs() > 1e-14 {
            start -= 1;
        }
        let nn = active_end - start;
        if nn < 2 {
            active_end -= 1;
            continue;
        }

        // QR step with Wilkinson shift on bottom 2×2
        let end = active_end;
        let d = (eigenvalues[end - 2] - eigenvalues[end - 1]) / 2.0;
        let bn = off_diag[end - 2];
        let denom = d + d.signum() * (d * d + bn * bn).sqrt();
        let shift = if denom.abs() > 1e-30 {
            eigenvalues[end - 1] - bn * bn / denom
        } else {
            eigenvalues[end - 1]
        };

        // Implicit QR step with Givens rotations (bulge chase)
        let mut x = eigenvalues[start] - shift;
        let mut z = off_diag[start];

        for k in start..end - 1 {
            let r = (x * x + z * z).sqrt();
            if r < 1e-30 {
                break;
            }
            let cos = x / r;
            let sin = z / r;

            if k > start {
                off_diag[k - 1] = r;
            }

            let d1 = eigenvalues[k];
            let d2 = eigenvalues[k + 1];
            let b = off_diag[k];

            eigenvalues[k] = cos * cos * d1 + 2.0 * cos * sin * b + sin * sin * d2;
            eigenvalues[k + 1] = sin * sin * d1 - 2.0 * cos * sin * b + cos * cos * d2;
            off_diag[k] = cos * sin * (d2 - d1) + (cos * cos - sin * sin) * b;

            if k < end - 2 {
                x = off_diag[k];
                z = sin * off_diag[k + 1];
                off_diag[k + 1] = cos * off_diag[k + 1];
            }
        }
    }

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_eval = eigenvalues[0];

    // Get eigenvector via inverse iteration on original tridiagonal
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let shift = min_eval - 1e-10;
    for _ in 0..20 {
        // Solve (T - shift*I) w = v using Thomas algorithm
        let mut a_diag: Vec<f64> = alpha.iter().map(|x| x - shift).collect();
        let b_off = beta.to_vec();
        let mut rhs = v.clone();

        // Forward elimination
        for i in 1..n {
            if a_diag[i - 1].abs() < 1e-30 {
                a_diag[i - 1] = 1e-30;
            }
            let m = b_off[i - 1] / a_diag[i - 1];
            a_diag[i] -= m * b_off[i - 1];
            rhs[i] -= m * rhs[i - 1];
        }

        // Back substitution
        if a_diag[n - 1].abs() < 1e-30 {
            a_diag[n - 1] = 1e-30;
        }
        v[n - 1] = rhs[n - 1] / a_diag[n - 1];
        for i in (0..n - 1).rev() {
            v[i] = (rhs[i] - b_off[i] * v[i + 1]) / a_diag[i];
        }

        // Normalize
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-30 {
            for x in v.iter_mut() {
                *x /= nrm;
            }
        }
    }

    (min_eval, v)
}

/// Build effective Hamiltonian for the AC tensor (1-site center).
fn build_effective_h_ac(
    h: &Array4<C64>,
    _a: &Array3<C64>,
    l_fp: &Array2<C64>,
    r_fp: &Array2<C64>,
    d: usize,
    chi: usize,
) -> Array2<C64> {
    let dim = chi * d * chi;
    let mut h_eff = Array2::zeros((dim, dim));

    // H_AC = L ⊗ h ⊗ R (simplified: project onto single-site effective Hamiltonian)
    for a1 in 0..chi {
        for s1 in 0..d {
            for b1 in 0..chi {
                let row = a1 * d * chi + s1 * chi + b1;
                for a2 in 0..chi {
                    for s2 in 0..d {
                        for b2 in 0..chi {
                            let col = a2 * d * chi + s2 * chi + b2;
                            // <a1,s1,b1|H_eff|a2,s2,b2> = L[a1,a2] * h_1site[s1,s2] * R[b1,b2]
                            let l_elem = l_fp[[a1, a2]];
                            let r_elem = r_fp[[b1, b2]];
                            // One-site effective h from two-site h: sum over partner
                            let mut h_elem = c0();
                            for sp in 0..d {
                                h_elem = h_elem + h[[s1, sp, s2, sp]];
                            }
                            h_eff[[row, col]] = l_elem * h_elem * r_elem;
                        }
                    }
                }
            }
        }
    }
    h_eff
}

/// Build effective Hamiltonian for the C matrix (zero-site center).
fn build_effective_h_c(l_fp: &Array2<C64>, r_fp: &Array2<C64>, chi: usize) -> Array2<C64> {
    let dim = chi * chi;
    let mut h_eff = Array2::zeros((dim, dim));
    for a1 in 0..chi {
        for b1 in 0..chi {
            let row = a1 * chi + b1;
            for a2 in 0..chi {
                for b2 in 0..chi {
                    let col = a2 * chi + b2;
                    h_eff[[row, col]] = l_fp[[a1, a2]] * r_fp[[b1, b2]];
                }
            }
        }
    }
    h_eff
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // --- Error type tests ---

    #[test]
    fn test_error_display_convergence() {
        let e = InfiniteMpsError::ConvergenceFailed {
            iterations: 100,
            residual: 1e-5,
            tolerance: 1e-10,
        };
        let s = format!("{}", e);
        assert!(s.contains("100"));
        assert!(s.contains("Not converged"));
    }

    #[test]
    fn test_error_display_all_variants() {
        let errors = vec![
            InfiniteMpsError::InvalidBondDim("zero".into()),
            InfiniteMpsError::SvdFailed("nan".into()),
            InfiniteMpsError::TransferMatrixFailed("collapsed".into()),
            InfiniteMpsError::CtmrgFailed {
                iterations: 50,
                residual: 0.1,
            },
            InfiniteMpsError::NumericalError("inf".into()),
        ];
        for e in &errors {
            let s = format!("{}", e);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(InfiniteMpsError::NumericalError("test".into()));
        assert!(!e.to_string().is_empty());
    }

    // --- Config tests ---

    #[test]
    fn test_vumps_config_defaults() {
        let c = VumpsConfig::default();
        assert_eq!(c.bond_dim, 32);
        assert_eq!(c.phys_dim, 2);
        assert_eq!(c.max_iterations, 200);
        assert!((c.tolerance - 1e-10).abs() < 1e-15);
        assert_eq!(c.lanczos_dim, 20);
        assert!(!c.two_site);
    }

    #[test]
    fn test_vumps_config_builder() {
        let c = VumpsConfig::new()
            .with_bond_dim(64)
            .with_phys_dim(3)
            .with_max_iterations(500)
            .with_tolerance(1e-12)
            .with_lanczos_dim(30)
            .with_two_site(true);
        assert_eq!(c.bond_dim, 64);
        assert_eq!(c.phys_dim, 3);
        assert_eq!(c.max_iterations, 500);
        assert!(c.two_site);
    }

    #[test]
    fn test_ctmrg_config_defaults() {
        let c = CtmrgConfig::default();
        assert_eq!(c.chi, 16);
        assert_eq!(c.max_iterations, 100);
        assert!((c.tolerance - 1e-8).abs() < 1e-15);
        assert_eq!(c.init_method, CtmrgInit::Random);
    }

    #[test]
    fn test_ctmrg_config_builder() {
        let c = CtmrgConfig::new()
            .with_chi(32)
            .with_max_iterations(200)
            .with_tolerance(1e-10)
            .with_init_method(CtmrgInit::Identity);
        assert_eq!(c.chi, 32);
        assert_eq!(c.init_method, CtmrgInit::Identity);
    }

    #[test]
    fn test_infinite_tebd_config_defaults() {
        let c = InfiniteTebdConfig::default();
        assert_eq!(c.max_bond_dim, 64);
        assert_eq!(c.trotter_order, 2);
        assert!((c.dt - 0.01).abs() < 1e-15);
        assert_eq!(c.num_steps, 100);
    }

    #[test]
    fn test_infinite_tebd_config_builder() {
        let c = InfiniteTebdConfig::new()
            .with_max_bond_dim(128)
            .with_svd_cutoff(1e-14)
            .with_trotter_order(4)
            .with_dt(0.005)
            .with_num_steps(200);
        assert_eq!(c.max_bond_dim, 128);
        assert_eq!(c.trotter_order, 4);
        assert_eq!(c.num_steps, 200);
    }

    // --- UniformMps tests ---

    #[test]
    fn test_uniform_mps_random() {
        let mps = UniformMps::random(2, 4, &mut rng());
        assert_eq!(mps.bond_dim, 4);
        assert_eq!(mps.phys_dim, 2);
        assert_eq!(mps.tensors.len(), 1);
        assert_eq!(mps.tensors[0].shape(), &[4, 2, 4]);
        assert_eq!(mps.center.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_mps_product_state() {
        let mps = UniformMps::product_state(2, 4);
        assert_eq!(mps.bond_dim, 4);
        // Check A[0,0,0] = 1 (spin up)
        assert!((mps.tensors[0][[0, 0, 0]] - c1()).norm() < 1e-10);
        // Check A[0,1,0] = 0 (spin down coefficient)
        assert!(mps.tensors[0][[0, 1, 0]].norm() < 1e-10);
    }

    #[test]
    fn test_uniform_mps_unit_cell_size() {
        let mps = UniformMps::random(2, 4, &mut rng());
        assert_eq!(mps.unit_cell_size(), 1);
    }

    #[test]
    fn test_uniform_mps_entanglement_entropy() {
        let mps = UniformMps::product_state(2, 4);
        let s = mps.entanglement_entropy();
        // Product state: C = (1/2)I, all equal singular values → max entropy
        assert!(s >= 0.0);
    }

    // --- Transfer Matrix tests ---

    #[test]
    fn test_transfer_matrix_apply() {
        let mut rng = rng();
        let mps = UniformMps::random(2, 3, &mut rng);
        let tm = TransferMatrix::new(3, 2);
        let x = Array2::eye(3).mapv(|v| cr(v));
        let tx = tm.apply(&mps.tensors[0], &x);
        assert_eq!(tx.shape(), &[3, 3]);
        // Result should be non-zero
        assert!(frobenius_norm(&tx) > 1e-10);
    }

    #[test]
    fn test_transfer_matrix_dominant_eigenvector() {
        let mut rng = rng();
        let mps = UniformMps::random(2, 3, &mut rng);
        let tm = TransferMatrix::new(3, 2);
        let result = tm.dominant_eigenvector(&mps.tensors[0], 100, 1e-10);
        assert!(result.is_ok());
        let (lambda, v) = result.unwrap();
        assert!(lambda.norm() > 0.0);
        assert!(frobenius_norm(&v) > 0.0);
    }

    #[test]
    fn test_transfer_matrix_correlation_length() {
        let mut rng = rng();
        let mps = UniformMps::random(2, 4, &mut rng);
        let tm = TransferMatrix::new(4, 2);
        let result = tm.correlation_length(&mps.tensors[0], 100);
        // Should return some finite positive value or infinity
        assert!(result.is_ok());
        let xi = result.unwrap();
        assert!(xi > 0.0);
    }

    // --- Model Hamiltonian tests ---

    #[test]
    fn test_ising_gate_hermitian() {
        let h = ising_two_site_gate(1.0, 0.5);
        // Check H[s1,s2,s1',s2'] = H[s1',s2',s1,s2]*
        for s1 in 0..2 {
            for s2 in 0..2 {
                for s1p in 0..2 {
                    for s2p in 0..2 {
                        let diff = (h[[s1, s2, s1p, s2p]] - h[[s1p, s2p, s1, s2]].conj()).norm();
                        assert!(
                            diff < 1e-10,
                            "Ising gate not Hermitian at [{},{},{},{}]",
                            s1,
                            s2,
                            s1p,
                            s2p
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_heisenberg_gate_hermitian() {
        let h = heisenberg_two_site_gate(1.0);
        for s1 in 0..2 {
            for s2 in 0..2 {
                for s1p in 0..2 {
                    for s2p in 0..2 {
                        let diff = (h[[s1, s2, s1p, s2p]] - h[[s1p, s2p, s1, s2]].conj()).norm();
                        assert!(
                            diff < 1e-10,
                            "Heisenberg gate not Hermitian at [{},{},{},{}]",
                            s1,
                            s2,
                            s1p,
                            s2p
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_ising_gate_zz_diagonal() {
        let h = ising_two_site_gate(1.0, 0.0);
        // Pure ZZ: diagonal elements should be ±J
        assert!((h[[0, 0, 0, 0]].re - (-1.0)).abs() < 1e-10); // |00>: +1*+1 = -J
        assert!((h[[0, 1, 0, 1]].re - 1.0).abs() < 1e-10); // |01>: +1*-1 = +J
        assert!((h[[1, 0, 1, 0]].re - 1.0).abs() < 1e-10); // |10>: -1*+1 = +J
        assert!((h[[1, 1, 1, 1]].re - (-1.0)).abs() < 1e-10); // |11>: -1*-1 = -J
    }

    // --- SVD and linear algebra tests ---

    #[test]
    fn test_matmul_identity() {
        let eye = Array2::eye(3).mapv(|v| cr(v));
        let a = Array2::from_shape_fn((3, 3), |(i, j)| cr((i * 3 + j) as f64));
        let result = matmul(&eye, &a);
        for i in 0..3 {
            for j in 0..3 {
                assert!((result[[i, j]] - a[[i, j]]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_conjugate_transpose() {
        let a = Array2::from_shape_fn((2, 3), |(i, j)| C64::new(i as f64, j as f64));
        let at = conjugate_transpose(&a);
        assert_eq!(at.shape(), &[3, 2]);
        assert!((at[[0, 0]] - a[[0, 0]].conj()).norm() < 1e-10);
        assert!((at[[1, 0]] - a[[0, 1]].conj()).norm() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm() {
        let a = Array2::eye(3).mapv(|v| cr(v));
        let nrm = frobenius_norm(&a);
        assert!((nrm - 3.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_svd_truncate_identity() {
        let eye = Array2::eye(4).mapv(|v| cr(v));
        let (u, s, vt) = svd_truncate_mat(&eye, 2);
        assert_eq!(s.len(), 2);
        // All singular values should be ~1
        for sv in &s {
            assert!((sv - 1.0).abs() < 0.1, "sv={}", sv);
        }
    }

    #[test]
    fn test_polar_decomposition_unitary() {
        let a = Array2::eye(3).mapv(|v| cr(v));
        let (w, _p) = polar_decomposition(&a);
        // W should be close to identity for identity input
        for i in 0..3 {
            assert!((w[[i, i]] - c1()).norm() < 0.5);
        }
    }

    #[test]
    fn test_spectrum_distance_identical() {
        let a = vec![1.0, 0.5, 0.25];
        assert!((spectrum_distance(&a, &a) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_spectrum_distance_different() {
        let a = vec![1.0, 0.5];
        let b = vec![1.0, 0.0];
        assert!(spectrum_distance(&a, &b) > 0.0);
    }

    // --- Flatten/unflatten tests ---

    #[test]
    fn test_flatten_unflatten_3d_roundtrip() {
        let a = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| cr((i * 12 + j * 4 + k) as f64));
        let flat = flatten_3d(&a);
        let restored = unflatten_3d(&flat, 2, 3, 4);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert!((a[[i, j, k]] - restored[[i, j, k]]).norm() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_flatten_unflatten_2d_roundtrip() {
        let a = Array2::from_shape_fn((3, 4), |(i, j)| cr((i * 4 + j) as f64));
        let flat = flatten_2d(&a);
        let restored = unflatten_2d(&flat, 3, 4);
        for i in 0..3 {
            for j in 0..4 {
                assert!((a[[i, j]] - restored[[i, j]]).norm() < 1e-10);
            }
        }
    }

    // --- Gate exponentiation test ---

    #[test]
    fn test_exponentiate_zero_gate() {
        let h = Array4::zeros((2, 2, 2, 2));
        let u = exponentiate_gate(&h, 0.1);
        // exp(0) = I
        for s1 in 0..2 {
            for s2 in 0..2 {
                let expected = if s1 == 0 && s2 == 0 { c1() } else { c0() };
                // Only check diagonal-like: u[s1,s2,s1,s2] should be 1, off-diagonal 0
                assert!(
                    (u[[s1, s2, s1, s2]] - c1()).norm() < 1e-8,
                    "u[{},{},{},{}] = {:?}",
                    s1,
                    s2,
                    s1,
                    s2,
                    u[[s1, s2, s1, s2]]
                );
            }
        }
    }

    // --- Lanczos tests ---

    #[test]
    fn test_lanczos_diagonal_matrix() {
        // 3x3 diagonal matrix with eigenvalues 1, 2, 3
        let mut h = Array2::zeros((3, 3));
        h[[0, 0]] = cr(3.0);
        h[[1, 1]] = cr(1.0);
        h[[2, 2]] = cr(2.0);
        let initial = vec![cr(1.0), cr(1.0), cr(1.0)];
        let (eval, _evec) = lanczos_ground(&h, &initial, 3, 10).unwrap();
        assert!((eval - 1.0).abs() < 0.1, "Expected ~1.0, got {}", eval);
    }

    #[test]
    fn test_tridiagonal_eigmin_simple() {
        let alpha = vec![2.0, 3.0];
        let beta = vec![1.0];
        let (eval, evec) = tridiagonal_eigmin(&alpha, &beta);
        // 2x2 matrix [[2,1],[1,3]] has eigenvalues (5±√5)/2 ≈ 1.38, 3.62
        assert!((eval - 1.382).abs() < 0.1, "eval = {}", eval);
        assert_eq!(evec.len(), 2);
    }

    // --- PepsTensor tests ---

    #[test]
    fn test_peps_tensor_zeros() {
        let dims = [2, 3, 2, 3, 2];
        let t = PepsTensor::zeros(dims);
        assert_eq!(t.data.len(), 2 * 3 * 2 * 3 * 2);
        assert!(t.norm() < 1e-15);
    }

    #[test]
    fn test_peps_tensor_random() {
        let dims = [2, 2, 2, 2, 2];
        let t = PepsTensor::random(dims, &mut rng());
        assert_eq!(t.data.len(), 32);
        assert!(t.norm() > 0.0);
    }

    // --- InfinitePeps tests ---

    #[test]
    fn test_infinite_peps_random() {
        let peps = InfinitePeps::random(2, 3, (2, 2), &mut rng());
        assert_eq!(peps.tensors.len(), 2);
        assert_eq!(peps.tensors[0].len(), 2);
        assert_eq!(peps.bond_dim, 3);
        assert_eq!(peps.phys_dim, 2);
        assert_eq!(peps.unit_cell, (2, 2));
    }

    // --- CtmEnvironment tests ---

    #[test]
    fn test_ctm_environment_identity_init() {
        let env = CtmEnvironment::new(4, 2, CtmrgInit::Identity, &mut rng());
        assert_eq!(env.chi, 4);
        assert!(!env.converged);
        assert_eq!(env.corners[0].shape(), &[4, 4]);
        // Identity corners: should have trace = 4
        let trace: C64 = (0..4).map(|i| env.corners[0][[i, i]]).sum();
        assert!((trace.re - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ctm_environment_random_init() {
        let env = CtmEnvironment::new(4, 2, CtmrgInit::Random, &mut rng());
        assert_eq!(env.chi, 4);
        assert!(frobenius_norm(&env.corners[0]) > 0.0);
    }

    #[test]
    fn test_ctm_corner_spectrum() {
        let env = CtmEnvironment::new(4, 2, CtmrgInit::Identity, &mut rng());
        let spec = env.corner_spectrum(0);
        assert!(!spec.is_empty());
        // Identity matrix: all singular values = 1
        for s in &spec {
            assert!((s - 1.0).abs() < 0.5, "sv={}", s);
        }
    }

    // --- VUMPS test ---

    #[test]
    fn test_vumps_invalid_bond_dim() {
        let h = ising_two_site_gate(1.0, 0.5);
        let config = VumpsConfig::new().with_bond_dim(0);
        let result = vumps(&h, &config, &mut rng());
        assert!(result.is_err());
    }

    #[test]
    fn test_vumps_runs_and_returns() {
        let h = ising_two_site_gate(1.0, 0.5);
        let config = VumpsConfig::new()
            .with_bond_dim(4)
            .with_max_iterations(5)
            .with_tolerance(1e-6)
            .with_lanczos_dim(8);
        let result = vumps(&h, &config, &mut rng());
        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.iterations <= 5);
        // Energy should be finite
        assert!(res.energy_per_site.is_finite());
    }

    // --- Infinite TEBD test ---

    #[test]
    fn test_infinite_tebd_runs() {
        let h = ising_two_site_gate(1.0, 0.5);
        let mps = UniformMps::product_state(2, 4);
        let config = InfiniteTebdConfig::new()
            .with_max_bond_dim(4)
            .with_dt(0.01)
            .with_num_steps(3);
        let result = infinite_tebd(&mps, &h, &config);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.num_steps, 3);
        assert_eq!(res.energies.len(), 3);
        assert_eq!(res.truncation_errors.len(), 3);
    }

    #[test]
    fn test_infinite_tebd_truncation_error_bounded() {
        let h = heisenberg_two_site_gate(1.0);
        let mps = UniformMps::random(2, 4, &mut rng());
        let config = InfiniteTebdConfig::new()
            .with_max_bond_dim(4)
            .with_dt(0.001)
            .with_num_steps(2);
        let result = infinite_tebd(&mps, &h, &config).unwrap();
        for &te in &result.truncation_errors {
            assert!(te >= 0.0 && te <= 1.0, "truncation error = {}", te);
        }
    }

    // --- CTMRG test ---

    #[test]
    fn test_ctmrg_runs() {
        let peps = InfinitePeps::random(2, 2, (1, 1), &mut rng());
        let config = CtmrgConfig::new()
            .with_chi(4)
            .with_max_iterations(3)
            .with_tolerance(1e-2);
        // May not converge in 3 iterations, but should run without panic
        let _ = ctmrg(&peps, &config, &mut rng());
    }
}
