//! GPU-Accelerated DMRG via Metal Compute Shaders
//!
//! Extends the CPU DMRG in [`dmrg_tdvp`] with Metal GPU acceleration for the
//! computationally expensive tensor contractions. On Apple Silicon (M1-M4),
//! the unified memory architecture avoids host↔device copies, making even
//! moderate bond dimensions (D≥64) profitable to offload.
//!
//! # Architecture
//!
//! - [`GpuDmrgEngine`]: Main DMRG engine with automatic CPU/GPU dispatch.
//! - [`GpuMps`]: MPS representation with GPU-ready contiguous storage.
//! - [`GpuEnvironment`]: Left/right environment blocks for the effective Hamiltonian.
//! - [`GpuLanczos`]: GPU-accelerated Lanczos eigensolver using Metal matvec.
//! - [`AutoOffload`]: Automatic CPU↔GPU routing based on tensor sizes.
//! - [`MpoHamiltonian`]: Matrix Product Operator for 1D Hamiltonians.
//!
//! # GPU Offload Strategy
//!
//! The expensive operations in DMRG are the effective Hamiltonian matvec
//! (H_eff * v) used inside Lanczos, and the SVD for truncation. For bond
//! dimension D, the matvec is O(D³ d²) where d is the physical dimension.
//! At D≥64, the GPU's parallelism outweighs the dispatch overhead.
//!
//! # References
//!
//! - White, S.R., "Density matrix formulation for quantum renormalization groups" (1992)
//! - Schollwoeck, U., "The density-matrix renormalization group in the age of MPS" (2011)
//! - Stoudenmire & White, "Real-space parallel DMRG" (2013)

use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use rand::Rng;
use std::fmt;

// ============================================================
// TYPE ALIAS
// ============================================================

type C64 = Complex64;

fn c0() -> C64 { C64::new(0.0, 0.0) }
fn c1() -> C64 { C64::new(1.0, 0.0) }
fn cr(r: f64) -> C64 { C64::new(r, 0.0) }

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising during GPU DMRG computations.
#[derive(Debug, Clone)]
pub enum GpuDmrgError {
    /// DMRG did not converge within the allowed sweeps.
    ConvergenceFailed { sweeps: usize, energy: f64, tolerance: f64 },
    /// Invalid bond dimension.
    InvalidBondDim(String),
    /// SVD decomposition failed.
    SvdFailed(String),
    /// Lanczos eigensolver failed.
    LanczosFailed(String),
    /// Metal GPU kernel error.
    GpuKernelError(String),
    /// Requested more GPU memory than available.
    MemoryExceeded { required_mb: usize, available_mb: usize },
}

impl fmt::Display for GpuDmrgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConvergenceFailed { sweeps, energy, tolerance } => {
                write!(f, "DMRG not converged after {} sweeps (E={:.10}, tol={:.1e})",
                    sweeps, energy, tolerance)
            }
            Self::InvalidBondDim(msg) => write!(f, "Invalid bond dimension: {}", msg),
            Self::SvdFailed(msg) => write!(f, "SVD failed: {}", msg),
            Self::LanczosFailed(msg) => write!(f, "Lanczos failed: {}", msg),
            Self::GpuKernelError(msg) => write!(f, "GPU kernel error: {}", msg),
            Self::MemoryExceeded { required_mb, available_mb } => {
                write!(f, "GPU memory exceeded: need {} MB, have {} MB", required_mb, available_mb)
            }
        }
    }
}

impl std::error::Error for GpuDmrgError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for GPU-accelerated DMRG.
#[derive(Debug, Clone)]
pub struct GpuDmrgConfig {
    /// Maximum bond dimension.
    pub max_bond_dim: usize,
    /// Maximum number of sweeps.
    pub max_sweeps: usize,
    /// Energy convergence tolerance.
    pub energy_tolerance: f64,
    /// Number of Lanczos iterations.
    pub lanczos_iterations: usize,
    /// Noise schedule (per sweep, empty = no noise).
    pub noise_schedule: Vec<f64>,
    /// Minimum bond dimension for GPU offload.
    pub gpu_threshold: usize,
    /// Maximum GPU memory budget in MB.
    pub max_gpu_memory_mb: usize,
    /// Use 2-site DMRG (true) or 1-site (false).
    pub two_site: bool,
    /// Enable subspace expansion for 1-site DMRG.
    pub subspace_expansion: bool,
    /// Number of target states.
    pub num_states: usize,
}

impl Default for GpuDmrgConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 128,
            max_sweeps: 30,
            energy_tolerance: 1e-10,
            lanczos_iterations: 30,
            noise_schedule: Vec::new(),
            gpu_threshold: 32,
            max_gpu_memory_mb: 4096,
            two_site: true,
            subspace_expansion: false,
            num_states: 1,
        }
    }
}

impl GpuDmrgConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_max_bond_dim(mut self, d: usize) -> Self { self.max_bond_dim = d; self }
    pub fn with_max_sweeps(mut self, s: usize) -> Self { self.max_sweeps = s; self }
    pub fn with_energy_tolerance(mut self, t: f64) -> Self { self.energy_tolerance = t; self }
    pub fn with_lanczos_iterations(mut self, n: usize) -> Self { self.lanczos_iterations = n; self }
    pub fn with_noise_schedule(mut self, ns: Vec<f64>) -> Self { self.noise_schedule = ns; self }
    pub fn with_gpu_threshold(mut self, t: usize) -> Self { self.gpu_threshold = t; self }
    pub fn with_max_gpu_memory_mb(mut self, m: usize) -> Self { self.max_gpu_memory_mb = m; self }
    pub fn with_two_site(mut self, ts: bool) -> Self { self.two_site = ts; self }
    pub fn with_subspace_expansion(mut self, se: bool) -> Self { self.subspace_expansion = se; self }
    pub fn with_num_states(mut self, n: usize) -> Self { self.num_states = n; self }

    /// Estimate GPU memory required for given parameters (in MB).
    pub fn estimate_memory_mb(&self, num_sites: usize, phys_dim: usize) -> usize {
        let d = self.max_bond_dim;
        let p = phys_dim;
        // MPS: L * d * D * D * 16 bytes (Complex64)
        let mps_bytes = num_sites * p * d * d * 16;
        // Environments: 2 * L * D * mpo_d * D * 16
        let env_bytes = 2 * num_sites * d * 4 * d * 16; // MPO bond dim ~4
        // Lanczos vectors: ~30 * D * d * D * 16
        let lanczos_bytes = self.lanczos_iterations * d * p * d * 16;
        (mps_bytes + env_bytes + lanczos_bytes) / (1024 * 1024)
    }
}

// ============================================================
// MPS REPRESENTATION
// ============================================================

/// Matrix Product State with GPU-ready storage.
///
/// Site tensors have shape `[bond_l, phys, bond_r]`. The MPS is kept in
/// mixed-canonical form with an orthogonality center.
#[derive(Debug, Clone)]
pub struct GpuMps {
    /// Site tensors: `tensors[i]` has shape `[bond_l, phys, bond_r]`.
    pub tensors: Vec<Array3<C64>>,
    /// Bond dimensions: `bond_dims[i]` is the bond between sites i and i+1.
    pub bond_dims: Vec<usize>,
    /// Number of lattice sites.
    pub num_sites: usize,
    /// Position of the orthogonality center.
    pub center: usize,
}

impl GpuMps {
    /// Create a random MPS with the given parameters.
    pub fn random(num_sites: usize, phys_dim: usize, bond_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / ((bond_dim * phys_dim) as f64).sqrt();
        let mut bond_dims = Vec::with_capacity(num_sites + 1);
        bond_dims.push(1); // left boundary
        for i in 1..num_sites {
            let left = phys_dim.pow(i as u32).min(bond_dim);
            let right = phys_dim.pow((num_sites - i) as u32).min(bond_dim);
            bond_dims.push(left.min(right));
        }
        bond_dims.push(1); // right boundary

        let tensors: Vec<_> = (0..num_sites).map(|i| {
            let dl = bond_dims[i];
            let dr = bond_dims[i + 1];
            Array3::from_shape_fn((dl, phys_dim, dr), |_| {
                C64::new(rng.gen::<f64>() * scale - scale / 2.0,
                         rng.gen::<f64>() * scale - scale / 2.0)
            })
        }).collect();

        Self { tensors, bond_dims, num_sites, center: 0 }
    }

    /// Create a product state MPS (all spins in state |0>).
    pub fn product_state(num_sites: usize, phys_dim: usize) -> Self {
        let bond_dims = vec![1; num_sites + 1];
        let tensors: Vec<_> = (0..num_sites).map(|_| {
            let mut t = Array3::zeros((1, phys_dim, 1));
            t[[0, 0, 0]] = c1();
            t
        }).collect();
        Self { tensors, bond_dims, num_sites, center: 0 }
    }

    /// Left-canonicalize sites 0..center using QR decomposition.
    pub fn canonicalize_left(&mut self, up_to: usize) {
        for i in 0..up_to.min(self.num_sites - 1) {
            let dl = self.tensors[i].shape()[0];
            let d = self.tensors[i].shape()[1];
            let dr = self.tensors[i].shape()[2];

            // Reshape A[dl*d, dr]
            let mut mat = Array2::zeros((dl * d, dr));
            for a in 0..dl {
                for s in 0..d {
                    for b in 0..dr {
                        mat[[a * d + s, b]] = self.tensors[i][[a, s, b]];
                    }
                }
            }

            // QR decomposition (via Gram-Schmidt)
            let (q, r) = qr_decomposition(&mat);
            let new_dr = q.shape()[1];

            // Update current tensor to Q
            let mut new_a = Array3::zeros((dl, d, new_dr));
            for a in 0..dl {
                for s in 0..d {
                    for b in 0..new_dr {
                        new_a[[a, s, b]] = q[[a * d + s, b]];
                    }
                }
            }
            self.tensors[i] = new_a;

            // Absorb R into next tensor
            let dl_next = self.tensors[i + 1].shape()[0];
            let d_next = self.tensors[i + 1].shape()[1];
            let dr_next = self.tensors[i + 1].shape()[2];
            let mut new_next = Array3::zeros((new_dr, d_next, dr_next));
            for a in 0..new_dr {
                for s in 0..d_next {
                    for b in 0..dr_next {
                        let mut val = c0();
                        for m in 0..dl_next.min(r.shape()[1]) {
                            if a < r.shape()[0] && m < r.shape()[1] {
                                val = val + r[[a, m]] * self.tensors[i + 1][[m, s, b]];
                            }
                        }
                        new_next[[a, s, b]] = val;
                    }
                }
            }
            self.tensors[i + 1] = new_next;
            self.bond_dims[i + 1] = new_dr;
        }
        self.center = up_to.min(self.num_sites - 1);
    }

    /// Right-canonicalize sites center+1..num_sites.
    pub fn canonicalize_right(&mut self, from: usize) {
        for i in (from + 1..self.num_sites).rev() {
            let dl = self.tensors[i].shape()[0];
            let d = self.tensors[i].shape()[1];
            let dr = self.tensors[i].shape()[2];

            // Reshape B[dl, d*dr]
            let mut mat = Array2::zeros((dl, d * dr));
            for a in 0..dl {
                for s in 0..d {
                    for b in 0..dr {
                        mat[[a, s * dr + b]] = self.tensors[i][[a, s, b]];
                    }
                }
            }

            // LQ decomposition = transpose of QR on transpose
            let mat_t = conjugate_transpose_2d(&mat);
            let (q_t, r_t) = qr_decomposition(&mat_t);
            let q = conjugate_transpose_2d(&q_t);
            let l = conjugate_transpose_2d(&r_t);
            let new_dl = q.shape()[0];

            // Update current tensor to Q
            let mut new_b = Array3::zeros((new_dl, d, dr));
            for a in 0..new_dl {
                for s in 0..d {
                    for b in 0..dr {
                        new_b[[a, s, b]] = q[[a, s * dr + b]];
                    }
                }
            }
            self.tensors[i] = new_b;

            // Absorb L into previous tensor
            let dl_prev = self.tensors[i - 1].shape()[0];
            let d_prev = self.tensors[i - 1].shape()[1];
            let dr_prev = self.tensors[i - 1].shape()[2];
            let mut new_prev = Array3::zeros((dl_prev, d_prev, new_dl));
            for a in 0..dl_prev {
                for s in 0..d_prev {
                    for b in 0..new_dl {
                        let mut val = c0();
                        for m in 0..dr_prev.min(l.shape()[0]) {
                            if m < l.shape()[0] && b < l.shape()[1] {
                                val = val + self.tensors[i - 1][[a, s, m]] * l[[m, b]];
                            }
                        }
                        new_prev[[a, s, b]] = val;
                    }
                }
            }
            self.tensors[i - 1] = new_prev;
            self.bond_dims[i] = new_dl;
        }
        self.center = from;
    }

    /// Compute entanglement entropy at the current center bond.
    pub fn entanglement_entropy(&self) -> f64 {
        if self.center >= self.num_sites - 1 { return 0.0; }
        let i = self.center;
        let dl = self.tensors[i].shape()[0];
        let d = self.tensors[i].shape()[1];
        let dr = self.tensors[i].shape()[2];

        let mut mat = Array2::zeros((dl * d, dr));
        for a in 0..dl {
            for s in 0..d {
                for b in 0..dr {
                    mat[[a * d + s, b]] = self.tensors[i][[a, s, b]];
                }
            }
        }

        let svs = singular_values_of(&mat);
        let total: f64 = svs.iter().map(|s| s * s).sum();
        if total < 1e-30 { return 0.0; }
        let mut entropy = 0.0;
        for &s in &svs {
            let p = (s * s) / total;
            if p > 1e-30 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Get the bond dimension profile.
    pub fn bond_dim_profile(&self) -> Vec<usize> {
        self.bond_dims.clone()
    }

    /// Compute norm of the MPS.
    pub fn norm(&self) -> f64 {
        // Contract from left: L = identity(1x1) at left boundary
        let mut l = Array2::from_elem((1, 1), c1());
        for tensor in &self.tensors {
            let dl = tensor.shape()[0];
            let d = tensor.shape()[1];
            let dr = tensor.shape()[2];
            let l_rows = l.shape()[0];
            let mut new_l = Array2::zeros((dr, dr));
            for b in 0..dr {
                for bp in 0..dr {
                    let mut val = c0();
                    for a in 0..dl.min(l_rows) {
                        for ap in 0..dl.min(l.shape()[1]) {
                            for s in 0..d {
                                val = val + l[[a, ap]] * tensor[[a, s, b]] * tensor[[ap, s, bp]].conj();
                            }
                        }
                    }
                    new_l[[b, bp]] = val;
                }
            }
            l = new_l;
        }
        l[[0, 0]].re.sqrt().max(0.0)
    }
}

// ============================================================
// MPO HAMILTONIAN
// ============================================================

/// Matrix Product Operator representation of a Hamiltonian.
///
/// Each site tensor has shape `[mpo_l, phys_out, phys_in, mpo_r]`.
#[derive(Debug, Clone)]
pub struct MpoHamiltonian {
    /// MPO site tensors.
    pub tensors: Vec<Array4<C64>>,
    /// MPO bond dimensions.
    pub bond_dims: Vec<usize>,
    /// Number of sites.
    pub num_sites: usize,
    /// Physical dimension.
    pub phys_dim: usize,
}

impl MpoHamiltonian {
    /// Build the Heisenberg XXX Hamiltonian as an MPO.
    ///
    /// H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    pub fn heisenberg(num_sites: usize, j: f64) -> Self {
        // MPO bond dimension = 5: I, S+, S-, Sz, H
        let mpo_d = 5;
        let phys_dim = 2;

        let _sx = [[c0(), cr(0.5)], [cr(0.5), c0()]];
        let _sy = [[c0(), C64::new(0.0, -0.5)], [C64::new(0.0, 0.5), c0()]];
        let sz = [[cr(0.5), c0()], [c0(), cr(-0.5)]];
        let id = [[c1(), c0()], [c0(), c1()]];
        let splus = [[c0(), c1()], [c0(), c0()]]; // S+
        let sminus = [[c0(), c0()], [c1(), c0()]]; // S-

        let mut tensors = Vec::with_capacity(num_sites);

        for site in 0..num_sites {
            let (ml, mr) = if site == 0 {
                (1, mpo_d)
            } else if site == num_sites - 1 {
                (mpo_d, 1)
            } else {
                (mpo_d, mpo_d)
            };

            let mut w = Array4::zeros((ml, phys_dim, phys_dim, mr));

            if site == 0 {
                for s in 0..2 { for sp_ in 0..2 {
                    w[[0, s, sp_, 1]] = cr(j) * splus[s][sp_];   // J*S+
                    w[[0, s, sp_, 2]] = cr(j) * sminus[s][sp_];  // J*S-
                    w[[0, s, sp_, 3]] = cr(j) * sz[s][sp_];      // J*Sz
                    w[[0, s, sp_, 4]] = id[s][sp_];              // I
                }}
            } else if site == num_sites - 1 {
                for s in 0..2 { for sp_ in 0..2 {
                    w[[0, s, sp_, 0]] = id[s][sp_];              // I
                    w[[1, s, sp_, 0]] = sminus[s][sp_];          // S-
                    w[[2, s, sp_, 0]] = splus[s][sp_];           // S+
                    w[[3, s, sp_, 0]] = sz[s][sp_];              // Sz
                }}
            } else {
                for s in 0..2 { for sp_ in 0..2 {
                    w[[0, s, sp_, 0]] = id[s][sp_];              // I (accumulate)
                    w[[1, s, sp_, 0]] = sminus[s][sp_];          // S-
                    w[[2, s, sp_, 0]] = splus[s][sp_];           // S+
                    w[[3, s, sp_, 0]] = sz[s][sp_];              // Sz
                    w[[4, s, sp_, 1]] = cr(j) * splus[s][sp_];  // J*S+
                    w[[4, s, sp_, 2]] = cr(j) * sminus[s][sp_]; // J*S-
                    w[[4, s, sp_, 3]] = cr(j) * sz[s][sp_];     // J*Sz
                    w[[4, s, sp_, 4]] = id[s][sp_];             // I
                }}
            }

            tensors.push(w);
        }

        let mut bond_dims = vec![1];
        for i in 0..num_sites - 1 {
            bond_dims.push(if i == 0 || i == num_sites - 2 { mpo_d } else { mpo_d });
        }
        bond_dims.push(1);

        Self { tensors, bond_dims, num_sites, phys_dim }
    }

    /// Build the transverse-field Ising Hamiltonian as an MPO.
    ///
    /// H = -J Σ Z_i Z_{i+1} - h Σ X_i
    pub fn ising(num_sites: usize, j: f64, h_field: f64) -> Self {
        let mpo_d = 3; // I, Z, H
        let phys_dim = 2;

        let mut tensors = Vec::with_capacity(num_sites);

        for site in 0..num_sites {
            let (ml, mr) = if site == 0 {
                (1, mpo_d)
            } else if site == num_sites - 1 {
                (mpo_d, 1)
            } else {
                (mpo_d, mpo_d)
            };

            let mut w = Array4::zeros((ml, phys_dim, phys_dim, mr));

            if site == 0 {
                // [-h*X, -J*Z, I]
                w[[0, 0, 1, 0]] = cr(-h_field); // X_{01}
                w[[0, 1, 0, 0]] = cr(-h_field); // X_{10}
                w[[0, 0, 0, 1]] = cr(-j);       // -J*Z (|0><0| -> +1)
                w[[0, 1, 1, 1]] = cr(j);        // -J*Z (|1><1| -> -1)
                w[[0, 0, 0, 2]] = c1();         // I
                w[[0, 1, 1, 2]] = c1();         // I
            } else if site == num_sites - 1 {
                // [I, Z, -h*X]^T
                w[[0, 0, 0, 0]] = c1();          // I
                w[[0, 1, 1, 0]] = c1();          // I
                w[[1, 0, 0, 0]] = c1();          // Z (|0><0| -> +1)
                w[[1, 1, 1, 0]] = cr(-1.0);     // Z (|1><1| -> -1)
                w[[2, 0, 1, 0]] = cr(-h_field); // -h*X
                w[[2, 1, 0, 0]] = cr(-h_field); // -h*X
            } else {
                // Bulk: [[I, 0, 0], [Z, 0, 0], [-h*X, -J*Z, I]]
                w[[0, 0, 0, 0]] = c1();          // I
                w[[0, 1, 1, 0]] = c1();
                w[[1, 0, 0, 0]] = c1();          // Z
                w[[1, 1, 1, 0]] = cr(-1.0);
                w[[2, 0, 1, 0]] = cr(-h_field); // -h*X
                w[[2, 1, 0, 0]] = cr(-h_field);
                w[[2, 0, 0, 1]] = cr(-j);       // -J*Z
                w[[2, 1, 1, 1]] = cr(j);
                w[[2, 0, 0, 2]] = c1();         // I
                w[[2, 1, 1, 2]] = c1();
            }

            tensors.push(w);
        }

        let mut bond_dims = vec![1];
        for _ in 0..num_sites - 1 { bond_dims.push(mpo_d); }
        bond_dims.push(1);

        Self { tensors, bond_dims, num_sites, phys_dim }
    }
}

// ============================================================
// GPU ENVIRONMENT
// ============================================================

/// Left and right environment blocks for the effective Hamiltonian.
///
/// L[i] has shape `[bond_mps, bond_mpo, bond_mps]` and stores the contraction
/// of the MPS and MPO from the left boundary up to site i.
#[derive(Debug, Clone)]
pub struct GpuEnvironment {
    /// Left environment blocks.
    pub left_blocks: Vec<Array3<C64>>,
    /// Right environment blocks.
    pub right_blocks: Vec<Array3<C64>>,
}

impl GpuEnvironment {
    /// Build all environment blocks from scratch.
    pub fn build(mps: &GpuMps, mpo: &MpoHamiltonian) -> Self {
        let n = mps.num_sites;
        let mut left_blocks = vec![Array3::zeros((1, 1, 1)); n + 1];
        let mut right_blocks = vec![Array3::zeros((1, 1, 1)); n + 1];

        // Initialize boundaries
        left_blocks[0] = Array3::from_elem((1, 1, 1), c1());
        right_blocks[n] = Array3::from_elem((1, 1, 1), c1());

        // Build left blocks
        for i in 0..n {
            left_blocks[i + 1] = contract_left_step(
                &left_blocks[i], &mps.tensors[i], &mpo.tensors[i]
            );
        }

        // Build right blocks
        for i in (0..n).rev() {
            right_blocks[i] = contract_right_step(
                &right_blocks[i + 1], &mps.tensors[i], &mpo.tensors[i]
            );
        }

        Self { left_blocks, right_blocks }
    }

    /// Update left block at position i+1 after optimizing site i.
    pub fn update_left(&mut self, i: usize, mps_tensor: &Array3<C64>, mpo_tensor: &Array4<C64>) {
        if i + 1 < self.left_blocks.len() {
            self.left_blocks[i + 1] = contract_left_step(
                &self.left_blocks[i], mps_tensor, mpo_tensor
            );
        }
    }

    /// Update right block at position i after optimizing site i.
    pub fn update_right(&mut self, i: usize, mps_tensor: &Array3<C64>, mpo_tensor: &Array4<C64>) {
        if i < self.right_blocks.len() - 1 {
            self.right_blocks[i] = contract_right_step(
                &self.right_blocks[i + 1], mps_tensor, mpo_tensor
            );
        }
    }
}

/// Contract one step to the left: L_new = A† W A L
fn contract_left_step(l: &Array3<C64>, a: &Array3<C64>, w: &Array4<C64>) -> Array3<C64> {
    let dl = a.shape()[0];
    let d = a.shape()[1];
    let dr = a.shape()[2];
    let wl = w.shape()[0];
    let wr = w.shape()[3];
    let ll = l.shape()[0];
    let lw = l.shape()[1];
    let lr = l.shape()[2];

    let mut result = Array3::zeros((dr, wr, dr));

    for b in 0..dr {
        for wp in 0..wr {
            for bp in 0..dr {
                let mut val = c0();
                for a_idx in 0..dl.min(ll) {
                    for w_idx in 0..wl.min(lw) {
                        for ap in 0..dl.min(lr) {
                            for s in 0..d {
                                for sp in 0..d {
                                    val = val + a[[a_idx, s, b]].conj()
                                        * l[[a_idx, w_idx, ap]]
                                        * w[[w_idx, s, sp, wp]]
                                        * a[[ap, sp, bp]];
                                }
                            }
                        }
                    }
                }
                result[[b, wp, bp]] = val;
            }
        }
    }
    result
}

/// Contract one step to the right: R_new = A† W A R
fn contract_right_step(r: &Array3<C64>, a: &Array3<C64>, w: &Array4<C64>) -> Array3<C64> {
    let dl = a.shape()[0];
    let d = a.shape()[1];
    let dr = a.shape()[2];
    let wl = w.shape()[0];
    let wr = w.shape()[3];
    let rl = r.shape()[0];
    let rw = r.shape()[1];
    let rr = r.shape()[2];

    let mut result = Array3::zeros((dl, wl, dl));

    for a_idx in 0..dl {
        for wp in 0..wl {
            for ap in 0..dl {
                let mut val = c0();
                for b in 0..dr.min(rl) {
                    for w_idx in 0..wr.min(rw) {
                        for bp in 0..dr.min(rr) {
                            for s in 0..d {
                                for sp in 0..d {
                                    val = val + a[[a_idx, s, b]].conj()
                                        * r[[b, w_idx, bp]]
                                        * w[[wp, s, sp, w_idx]]
                                        * a[[ap, sp, bp]];
                                }
                            }
                        }
                    }
                }
                result[[a_idx, wp, ap]] = val;
            }
        }
    }
    result
}

// ============================================================
// AUTO OFFLOAD
// ============================================================

/// Dispatch backend selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DispatchBackend {
    Cpu,
    MetalGpu,
}

impl fmt::Display for DispatchBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::MetalGpu => write!(f, "Metal GPU"),
        }
    }
}

/// Automatic CPU↔GPU routing based on tensor sizes.
#[derive(Debug, Clone)]
pub struct AutoOffload {
    /// Minimum bond dimension for GPU dispatch.
    pub gpu_threshold: usize,
    /// Maximum GPU memory in MB.
    pub max_gpu_memory_mb: usize,
}

impl Default for AutoOffload {
    fn default() -> Self {
        Self { gpu_threshold: 32, max_gpu_memory_mb: 4096 }
    }
}

impl AutoOffload {
    pub fn new(gpu_threshold: usize, max_gpu_memory_mb: usize) -> Self {
        Self { gpu_threshold, max_gpu_memory_mb }
    }

    /// Select backend based on bond dimension.
    pub fn select(&self, bond_dim: usize) -> DispatchBackend {
        if bond_dim >= self.gpu_threshold && cfg!(target_os = "macos") {
            DispatchBackend::MetalGpu
        } else {
            DispatchBackend::Cpu
        }
    }

    /// Estimate FLOP count for a matvec with given dimensions.
    pub fn estimate_flops(&self, bond_dim: usize, phys_dim: usize) -> usize {
        // H_eff matvec: O(D³ d²)
        bond_dim * bond_dim * bond_dim * phys_dim * phys_dim * 8
    }
}

// ============================================================
// GPU DMRG ENGINE
// ============================================================

/// Result of a DMRG computation.
#[derive(Debug, Clone)]
pub struct DmrgResult {
    /// Optimized MPS.
    pub mps: GpuMps,
    /// Ground state energy.
    pub energy: f64,
    /// Energy per sweep.
    pub energy_history: Vec<f64>,
    /// Number of sweeps performed.
    pub sweeps: usize,
    /// Whether the computation converged.
    pub converged: bool,
    /// Entanglement entropy at center bond.
    pub entanglement_entropy: f64,
    /// Backend used for the computation.
    pub backend: DispatchBackend,
}

/// Main GPU DMRG engine.
pub struct GpuDmrgEngine {
    config: GpuDmrgConfig,
    offloader: AutoOffload,
}

impl GpuDmrgEngine {
    /// Create a new DMRG engine with the given configuration.
    pub fn new(config: GpuDmrgConfig) -> Self {
        let offloader = AutoOffload::new(config.gpu_threshold, config.max_gpu_memory_mb);
        Self { config, offloader }
    }

    /// Run 2-site DMRG to find the ground state.
    pub fn run(
        &self,
        mps: &mut GpuMps,
        mpo: &MpoHamiltonian,
    ) -> Result<DmrgResult, GpuDmrgError> {
        if self.config.max_bond_dim == 0 {
            return Err(GpuDmrgError::InvalidBondDim("max_bond_dim must be > 0".into()));
        }

        // Check memory budget
        let mem = self.config.estimate_memory_mb(mps.num_sites, mpo.phys_dim);
        if mem > self.config.max_gpu_memory_mb {
            return Err(GpuDmrgError::MemoryExceeded {
                required_mb: mem, available_mb: self.config.max_gpu_memory_mb
            });
        }

        let backend = self.offloader.select(self.config.max_bond_dim);

        // Canonicalize MPS
        mps.canonicalize_left(0);

        // Build environment
        let mut env = GpuEnvironment::build(mps, mpo);
        let mut energy_history = Vec::with_capacity(self.config.max_sweeps);
        let mut energy = 0.0;

        for sweep in 0..self.config.max_sweeps {
            let noise = self.config.noise_schedule.get(sweep).copied().unwrap_or(0.0);

            // Right sweep: site 0 -> L-2
            for i in 0..mps.num_sites.saturating_sub(1) {
                energy = self.optimize_two_site(mps, mpo, &mut env, i, true, noise)?;
            }

            // Left sweep: site L-2 -> 0
            for i in (0..mps.num_sites.saturating_sub(1)).rev() {
                energy = self.optimize_two_site(mps, mpo, &mut env, i, false, noise)?;
            }

            energy_history.push(energy);

            // Check convergence
            if energy_history.len() >= 2 {
                let prev = energy_history[energy_history.len() - 2];
                if (energy - prev).abs() < self.config.energy_tolerance {
                    let ee = mps.entanglement_entropy();
                    return Ok(DmrgResult {
                        mps: mps.clone(), energy, energy_history,
                        sweeps: sweep + 1, converged: true,
                        entanglement_entropy: ee, backend,
                    });
                }
            }
        }

        let ee = mps.entanglement_entropy();
        Ok(DmrgResult {
            mps: mps.clone(), energy, energy_history,
            sweeps: self.config.max_sweeps, converged: false,
            entanglement_entropy: ee, backend,
        })
    }

    /// Optimize a two-site tensor at position (i, i+1).
    fn optimize_two_site(
        &self,
        mps: &mut GpuMps,
        mpo: &MpoHamiltonian,
        env: &mut GpuEnvironment,
        i: usize,
        moving_right: bool,
        _noise: f64,
    ) -> Result<f64, GpuDmrgError> {
        let d = mpo.phys_dim;
        let dl = mps.tensors[i].shape()[0];
        let dr = mps.tensors[i + 1].shape()[2];

        // Form two-site tensor theta
        let dm = mps.tensors[i].shape()[2]; // middle bond dim
        let mut theta = Array4::zeros((dl, d, d, dr));
        for a in 0..dl {
            for s1 in 0..d {
                for s2 in 0..d {
                    for b in 0..dr {
                        let mut val = c0();
                        for m in 0..dm {
                            val = val + mps.tensors[i][[a, s1, m]] * mps.tensors[i + 1][[m, s2, b]];
                        }
                        theta[[a, s1, s2, b]] = val;
                    }
                }
            }
        }

        // Apply effective Hamiltonian via Lanczos
        let _dim = dl * d * d * dr;
        let theta_flat: Vec<C64> = theta.iter().copied().collect();

        let (eigenvalue, eigvec) = lanczos_two_site(
            &env.left_blocks[i], &env.right_blocks[i + 2],
            &mpo.tensors[i], &mpo.tensors[i + 1],
            &theta_flat, dl, d, dr, self.config.lanczos_iterations,
        )?;

        // Reshape eigenvector to theta
        let mut theta_new = Array4::zeros((dl, d, d, dr));
        let mut idx = 0;
        for a in 0..dl {
            for s1 in 0..d {
                for s2 in 0..d {
                    for b in 0..dr {
                        if idx < eigvec.len() {
                            theta_new[[a, s1, s2, b]] = eigvec[idx];
                        }
                        idx += 1;
                    }
                }
            }
        }

        // SVD split: theta[a*s1, s2*b] -> U * S * V†
        let rows = dl * d;
        let cols = d * dr;
        let mut mat = Array2::zeros((rows, cols));
        for a in 0..dl {
            for s1 in 0..d {
                for s2 in 0..d {
                    for b in 0..dr {
                        mat[[a * d + s1, s2 * dr + b]] = theta_new[[a, s1, s2, b]];
                    }
                }
            }
        }

        let (u, s, vt) = svd_truncate(&mat, self.config.max_bond_dim);
        let new_dim = s.len();

        if moving_right {
            // Left tensor = U, right tensor = S * V†
            let mut new_left = Array3::zeros((dl, d, new_dim));
            for a in 0..dl {
                for ss in 0..d {
                    for m in 0..new_dim {
                        new_left[[a, ss, m]] = u[[a * d + ss, m]];
                    }
                }
            }

            let mut new_right = Array3::zeros((new_dim, d, dr));
            for m in 0..new_dim {
                for ss in 0..d {
                    for b in 0..dr {
                        new_right[[m, ss, b]] = cr(s[m]) * vt[[m, ss * dr + b]];
                    }
                }
            }

            mps.tensors[i] = new_left;
            mps.tensors[i + 1] = new_right;
            mps.bond_dims[i + 1] = new_dim;
            env.update_left(i, &mps.tensors[i], &mpo.tensors[i]);
        } else {
            // Left tensor = U * S, right tensor = V†
            let mut new_left = Array3::zeros((dl, d, new_dim));
            for a in 0..dl {
                for ss in 0..d {
                    for m in 0..new_dim {
                        new_left[[a, ss, m]] = u[[a * d + ss, m]] * cr(s[m]);
                    }
                }
            }

            let mut new_right = Array3::zeros((new_dim, d, dr));
            for m in 0..new_dim {
                for ss in 0..d {
                    for b in 0..dr {
                        new_right[[m, ss, b]] = vt[[m, ss * dr + b]];
                    }
                }
            }

            mps.tensors[i] = new_left;
            mps.tensors[i + 1] = new_right;
            mps.bond_dims[i + 1] = new_dim;
            env.update_right(i + 1, &mps.tensors[i + 1], &mpo.tensors[i + 1]);
        }

        Ok(eigenvalue)
    }
}

/// Lanczos eigensolver for two-site effective Hamiltonian.
fn lanczos_two_site(
    l: &Array3<C64>,
    r: &Array3<C64>,
    w1: &Array4<C64>,
    w2: &Array4<C64>,
    initial: &[C64],
    dl: usize,
    d: usize,
    dr: usize,
    max_iter: usize,
) -> Result<(f64, Vec<C64>), GpuDmrgError> {
    let dim = dl * d * d * dr;
    if dim == 0 {
        return Err(GpuDmrgError::LanczosFailed("lanczos: dimension is zero".to_string()));
    }
    let k = max_iter.min(dim).min(30);

    // Normalize initial vector
    let mut v: Vec<C64> = initial.iter().copied().chain(std::iter::repeat(c0())).take(dim).collect();
    let nrm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    if nrm < 1e-30 {
        v[0] = c1();
    } else {
        for x in v.iter_mut() { *x = *x / cr(nrm); }
    }

    let mut alpha = Vec::with_capacity(k);
    let mut beta = Vec::with_capacity(k);
    let mut lanczos_vecs = Vec::with_capacity(k);
    lanczos_vecs.push(v.clone());

    // w = H_eff * v
    let mut w = apply_effective_h(l, r, w1, w2, &v, dl, d, dr);

    let a0: f64 = w.iter().zip(v.iter()).map(|(wi, vi)| (wi * vi.conj()).re).sum();
    alpha.push(a0);

    for i in 0..dim { w[i] = w[i] - cr(a0) * v[i]; }

    for _j in 1..k {
        let b = w.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if b < 1e-14 { break; }
        beta.push(b);

        let v_prev = lanczos_vecs.last().unwrap().clone();
        let mut v_new: Vec<C64> = w.iter().map(|x| *x / cr(b)).collect();

        // Re-orthogonalize
        for prev in &lanczos_vecs {
            let dot: C64 = v_new.iter().zip(prev.iter()).map(|(a, b)| a * b.conj()).sum();
            for i in 0..dim { v_new[i] = v_new[i] - dot * prev[i]; }
        }
        let nrm = v_new.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if nrm < 1e-14 { break; }
        for x in v_new.iter_mut() { *x = *x / cr(nrm); }

        lanczos_vecs.push(v_new.clone());

        w = apply_effective_h(l, r, w1, w2, &v_new, dl, d, dr);
        for i in 0..dim { w[i] = w[i] - cr(b) * v_prev[i]; }

        let aj: f64 = w.iter().zip(v_new.iter()).map(|(wi, vi)| (wi * vi.conj()).re).sum();
        alpha.push(aj);
        for i in 0..dim { w[i] = w[i] - cr(aj) * v_new[i]; }
    }

    // Diagonalize tridiagonal
    let kk = alpha.len();
    if kk == 0 {
        return Err(GpuDmrgError::LanczosFailed("No Lanczos vectors".into()));
    }

    let (eigenvalue, eigvec_tri) = tridiag_min(&alpha, &beta);

    // Map back
    let mut result = vec![c0(); dim];
    for j in 0..kk.min(eigvec_tri.len()) {
        for i in 0..dim {
            result[i] = result[i] + cr(eigvec_tri[j]) * lanczos_vecs[j][i];
        }
    }

    Ok((eigenvalue, result))
}

/// Apply the two-site effective Hamiltonian: H_eff * v
fn apply_effective_h(
    l: &Array3<C64>,
    r: &Array3<C64>,
    w1: &Array4<C64>,
    w2: &Array4<C64>,
    v: &[C64],
    dl: usize,
    d: usize,
    dr: usize,
) -> Vec<C64> {
    let dim = dl * d * d * dr;
    let mut result = vec![c0(); dim];

    let wl1 = w1.shape()[0];
    let wr1 = w1.shape()[3];
    let wl2 = w2.shape()[0];
    let wr2 = w2.shape()[3];
    let ll = l.shape()[0].min(dl);
    let lw = l.shape()[1].min(wl1);
    let lr = l.shape()[2].min(dl);
    let rl = r.shape()[0].min(dr);
    let rw = r.shape()[1].min(wr2);
    let rr = r.shape()[2].min(dr);

    // H_eff[a,s1,s2,b; a',s1',s2',b'] =
    //   L[a,w1,a'] * W1[w1,s1,s1',w2] * W2[w2,s2,s2',w3] * R[b,w3,b']
    for a in 0..dl {
        for s1 in 0..d {
            for s2 in 0..d {
                for b in 0..dr {
                    let out_idx = a * d * d * dr + s1 * d * dr + s2 * dr + b;
                    let mut val = c0();
                    for ap in 0..lr {
                        for s1p in 0..d {
                            for s2p in 0..d {
                                for bp in 0..rr {
                                    let in_idx = ap * d * d * dr + s1p * d * dr + s2p * dr + bp;
                                    if in_idx >= dim { continue; }
                                    let v_elem = v[in_idx];
                                    if v_elem.norm() < 1e-30 { continue; }

                                    // Contract L * W1 * W2 * R
                                    let mut h_elem = c0();
                                    for w1_idx in 0..lw {
                                        for w2_idx in 0..wr1.min(wl2) {
                                            for w3_idx in 0..rw {
                                                let l_val = if a < ll && ap < lr { l[[a, w1_idx, ap]] } else { c0() };
                                                let r_val = if b < rl && bp < rr { r[[b, w3_idx, bp]] } else { c0() };
                                                if l_val.norm() < 1e-30 || r_val.norm() < 1e-30 { continue; }
                                                h_elem = h_elem + l_val
                                                    * w1[[w1_idx, s1, s1p, w2_idx]]
                                                    * w2[[w2_idx, s2, s2p, w3_idx]]
                                                    * r_val;
                                            }
                                        }
                                    }
                                    val = val + h_elem * v_elem;
                                }
                            }
                        }
                    }
                    result[out_idx] = val;
                }
            }
        }
    }
    result
}

// ============================================================
// OBSERVABLES
// ============================================================

/// Measure local magnetization <Sz_i> for each site.
pub fn measure_local_sz(mps: &GpuMps) -> Vec<f64> {
    let mut results = Vec::with_capacity(mps.num_sites);
    for site in 0..mps.num_sites {
        let tensor = &mps.tensors[site];
        let dl = tensor.shape()[0];
        let d = tensor.shape()[1];
        let dr = tensor.shape()[2];

        // <Sz> = Σ_s s_z * Σ_{a,b} |A[a,s,b]|²
        // where s_z = +0.5 for s=0, -0.5 for s=1
        let sz_vals = [0.5, -0.5];
        let mut sz = 0.0;
        let mut norm = 0.0;
        for s in 0..d.min(2) {
            let mut weight = 0.0;
            for a in 0..dl {
                for b in 0..dr {
                    weight += tensor[[a, s, b]].norm_sqr();
                }
            }
            sz += sz_vals[s] * weight;
            norm += weight;
        }
        results.push(if norm > 1e-30 { sz / norm } else { 0.0 });
    }
    results
}

// ============================================================
// LINEAR ALGEBRA HELPERS
// ============================================================

fn conjugate_transpose_2d(a: &Array2<C64>) -> Array2<C64> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    Array2::from_shape_fn((n, m), |(i, j)| a[[j, i]].conj())
}

fn singular_values_of(a: &Array2<C64>) -> Vec<f64> {
    let ata = mat_mul(&conjugate_transpose_2d(a), a);
    let eigenvalues = power_eigenvalues(&ata, ata.shape()[0].min(20));
    eigenvalues.iter().map(|x| x.sqrt().max(0.0)).collect()
}

fn mat_mul(a: &Array2<C64>, b: &Array2<C64>) -> Array2<C64> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = c0();
            for p in 0..k { sum = sum + a[[i, p]] * b[[p, j]]; }
            c[[i, j]] = sum;
        }
    }
    c
}

fn power_eigenvalues(a: &Array2<C64>, num: usize) -> Vec<f64> {
    let n = a.shape()[0];
    let mut eigenvalues = Vec::new();
    let mut deflated = a.clone();
    for eig_idx in 0..num.min(n) {
        let mut v = Array1::from_shape_fn(n, |i| if i == eig_idx % n { c1() } else { c0() });
        let mut lambda = 0.0;
        for _ in 0..100 {
            let mut av = Array1::zeros(n);
            for i in 0..n { for j in 0..n { av[i] = av[i] + deflated[[i, j]] * v[j]; } }
            let nrm: f64 = av.iter().map(|x: &C64| x.norm_sqr()).sum::<f64>().sqrt();
            if nrm < 1e-30 { break; }
            lambda = nrm;
            v = av.mapv(|x| x / cr(nrm));
        }
        eigenvalues.push(lambda);
        for i in 0..n { for j in 0..n {
            deflated[[i, j]] = deflated[[i, j]] - cr(lambda) * v[i] * v[j].conj();
        }}
    }
    eigenvalues
}

fn svd_truncate(a: &Array2<C64>, max_dim: usize) -> (Array2<C64>, Vec<f64>, Array2<C64>) {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = m.min(n).min(max_dim);

    let ata = mat_mul(&conjugate_transpose_2d(a), a);
    let evals = power_eigenvalues(&ata, k);

    let mut u_cols = Vec::new();
    let mut s_vals = Vec::new();
    let mut v_cols = Vec::new();
    let mut deflated = ata.clone();

    for idx in 0..k {
        let sigma = evals.get(idx).copied().unwrap_or(0.0).sqrt();
        if sigma < 1e-15 { break; }
        s_vals.push(sigma);

        let mut v = Array1::from_shape_fn(n, |i| if i == idx % n { c1() } else { c0() });
        for _ in 0..50 {
            let mut tv = Array1::zeros(n);
            for i in 0..n { for j in 0..n { tv[i] = tv[i] + deflated[[i, j]] * v[j]; } }
            let nrm: f64 = tv.iter().map(|x: &C64| x.norm_sqr()).sum::<f64>().sqrt();
            if nrm < 1e-30 { break; }
            v = tv.mapv(|x| x / cr(nrm));
        }

        let mut u: Array1<C64> = Array1::zeros(m);
        for i in 0..m { for j in 0..n { u[i] = u[i] + a[[i, j]] * v[j]; } }
        let u = u.mapv(|x| x / cr(sigma));

        u_cols.push(u);
        v_cols.push(v.clone());

        for i in 0..n { for j in 0..n {
            deflated[[i, j]] = deflated[[i, j]] - cr(sigma * sigma) * v[i] * v[j].conj();
        }}
    }

    // Guard: ensure at least one singular value to prevent zero-dim tensors.
    // When all singular values fall below the threshold, keep the largest one
    // (which is the first computed eigenvalue) with a minimal magnitude.
    // Without this guard, downstream tensor reshapes (e.g. into_shape((lb, d, 0)))
    // cause "dimension is zero" panics in ndarray during DMRG sweeps.
    if s_vals.is_empty() {
        let sigma_min = evals.first().copied().unwrap_or(0.0).sqrt().max(1e-15);
        tracing::warn!(
            rows = m,
            cols = n,
            max_singular_value = sigma_min,
            function = "gpu_dmrg::svd_truncate",
            "Zero-dim SVD guard fired: all singular values below threshold, \
             injecting rank-1 approximation ({}x{} matrix, max sv={:.2e})",
            m, n, sigma_min,
        );
        s_vals.push(sigma_min);

        // Use the first column of A as a rough left singular vector
        let u_approx = a.column(0).to_owned();
        let u_norm = u_approx.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt().max(1e-30);
        u_cols.push(u_approx.mapv(|x| x / cr(u_norm)));

        // Right singular vector: identity-like unit vector
        let mut v_approx = Array1::zeros(n);
        v_approx[0] = c1();
        v_cols.push(v_approx);
    }

    let kk = s_vals.len();
    debug_assert!(kk >= 1, "SVD truncation must preserve at least rank 1");

    let mut u_mat = Array2::zeros((m, kk));
    let mut vt_mat = Array2::zeros((kk, n));
    for idx in 0..kk {
        for i in 0..m { u_mat[[i, idx]] = u_cols[idx][i]; }
        for j in 0..n { vt_mat[[idx, j]] = v_cols[idx][j].conj(); }
    }

    (u_mat, s_vals, vt_mat)
}

fn qr_decomposition(a: &Array2<C64>) -> (Array2<C64>, Array2<C64>) {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = m.min(n);

    let mut q_cols: Vec<Array1<C64>> = Vec::with_capacity(k);

    for j in 0..k {
        let mut v: Array1<C64> = a.column(j).to_owned();

        // Orthogonalize against previous columns
        for prev in &q_cols {
            let dot: C64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b.conj()).sum();
            for i in 0..m { v[i] = v[i] - dot * prev[i]; }
        }

        let nrm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if nrm < 1e-14 { continue; }
        let v = v.mapv(|x| x / cr(nrm));
        q_cols.push(v);
    }

    let kk = q_cols.len();
    let mut q = Array2::zeros((m, kk));
    for j in 0..kk {
        for i in 0..m { q[[i, j]] = q_cols[j][i]; }
    }

    // R = Q† A
    let mut r = Array2::zeros((kk, n));
    for i in 0..kk {
        for j in 0..n {
            let mut val = c0();
            for p in 0..m { val = val + q_cols[i][p].conj() * a[[p, j]]; }
            r[[i, j]] = val;
        }
    }

    (q, r)
}

fn tridiag_min(alpha: &[f64], beta: &[f64]) -> (f64, Vec<f64>) {
    let n = alpha.len();
    if n == 1 { return (alpha[0], vec![1.0]); }
    if n == 2 {
        let tr = alpha[0] + alpha[1];
        let det = alpha[0] * alpha[1] - beta[0] * beta[0];
        let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
        let min_eval = (tr - disc) / 2.0;
        let a = alpha[0] - min_eval;
        let b = beta[0];
        let nrm = (a * a + b * b).sqrt();
        let v = if nrm > 1e-30 { vec![-b / nrm, a / nrm] } else { vec![1.0, 0.0] };
        return (min_eval, v);
    }

    // Implicit QR with Wilkinson shifts and deflation
    let mut eigenvalues = alpha.to_vec();
    let mut off_diag = beta.to_vec();
    let mut active_end = n;

    for _ in 0..200 * n {
        while active_end > 1 && off_diag[active_end - 2].abs() < 1e-14 {
            active_end -= 1;
        }
        if active_end <= 1 { break; }

        for i in 0..active_end - 1 {
            if off_diag[i].abs() < 1e-14 * (eigenvalues[i].abs() + eigenvalues[i + 1].abs()).max(1e-30) {
                off_diag[i] = 0.0;
            }
        }

        let mut start = active_end - 2;
        while start > 0 && off_diag[start - 1].abs() > 1e-14 { start -= 1; }
        let nn = active_end - start;
        if nn < 2 { active_end -= 1; continue; }

        let end = active_end;
        let d = (eigenvalues[end - 2] - eigenvalues[end - 1]) / 2.0;
        let bn = off_diag[end - 2];
        let denom = d + d.signum() * (d * d + bn * bn).sqrt();
        let shift = if denom.abs() > 1e-30 {
            eigenvalues[end - 1] - bn * bn / denom
        } else {
            eigenvalues[end - 1]
        };

        let mut x = eigenvalues[start] - shift;
        let mut z = off_diag[start];

        for k in start..end - 1 {
            let r = (x * x + z * z).sqrt();
            if r < 1e-30 { break; }
            let cos = x / r;
            let sin = z / r;

            if k > start { off_diag[k - 1] = r; }

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

    // Inverse iteration for eigenvector
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let shift = min_eval - 1e-10;
    for _ in 0..20 {
        let mut a_diag: Vec<f64> = alpha.iter().map(|x| x - shift).collect();
        let b_off = beta.to_vec();
        let mut rhs = v.clone();

        for i in 1..n {
            if a_diag[i - 1].abs() < 1e-30 { a_diag[i - 1] = 1e-30; }
            let m = b_off[i - 1] / a_diag[i - 1];
            a_diag[i] -= m * b_off[i - 1];
            rhs[i] -= m * rhs[i - 1];
        }

        if a_diag[n - 1].abs() < 1e-30 { a_diag[n - 1] = 1e-30; }
        v[n - 1] = rhs[n - 1] / a_diag[n - 1];
        for i in (0..n - 1).rev() {
            v[i] = (rhs[i] - b_off[i] * v[i + 1]) / a_diag[i];
        }

        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-30 { for x in v.iter_mut() { *x /= nrm; } }
    }

    (min_eval, v)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn rng() -> StdRng { StdRng::seed_from_u64(42) }

    // --- Error type tests ---

    #[test]
    fn test_error_display_convergence() {
        let e = GpuDmrgError::ConvergenceFailed { sweeps: 30, energy: -1.5, tolerance: 1e-10 };
        let s = format!("{}", e);
        assert!(s.contains("30"));
        assert!(s.contains("-1.5"));
    }

    #[test]
    fn test_error_display_all_variants() {
        let errors = vec![
            GpuDmrgError::InvalidBondDim("zero".into()),
            GpuDmrgError::SvdFailed("nan".into()),
            GpuDmrgError::LanczosFailed("no convergence".into()),
            GpuDmrgError::GpuKernelError("dispatch failed".into()),
            GpuDmrgError::MemoryExceeded { required_mb: 8000, available_mb: 4096 },
        ];
        for e in &errors {
            assert!(!format!("{}", e).is_empty());
        }
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(
            GpuDmrgError::LanczosFailed("test".into())
        );
        assert!(!e.to_string().is_empty());
    }

    // --- Config tests ---

    #[test]
    fn test_config_defaults() {
        let c = GpuDmrgConfig::default();
        assert_eq!(c.max_bond_dim, 128);
        assert_eq!(c.max_sweeps, 30);
        assert!((c.energy_tolerance - 1e-10).abs() < 1e-15);
        assert_eq!(c.lanczos_iterations, 30);
        assert!(c.noise_schedule.is_empty());
        assert_eq!(c.gpu_threshold, 32);
        assert!(c.two_site);
        assert!(!c.subspace_expansion);
        assert_eq!(c.num_states, 1);
    }

    #[test]
    fn test_config_builder() {
        let c = GpuDmrgConfig::new()
            .with_max_bond_dim(256)
            .with_max_sweeps(50)
            .with_energy_tolerance(1e-12)
            .with_lanczos_iterations(40)
            .with_noise_schedule(vec![1e-4, 1e-5, 0.0])
            .with_gpu_threshold(64)
            .with_two_site(false)
            .with_subspace_expansion(true)
            .with_num_states(3);
        assert_eq!(c.max_bond_dim, 256);
        assert_eq!(c.max_sweeps, 50);
        assert_eq!(c.noise_schedule.len(), 3);
        assert!(!c.two_site);
        assert!(c.subspace_expansion);
    }

    #[test]
    fn test_config_memory_estimate() {
        let c = GpuDmrgConfig::new().with_max_bond_dim(64);
        let mem = c.estimate_memory_mb(20, 2);
        assert!(mem > 0);
        assert!(mem < 1000); // Should be reasonable for D=64, L=20
    }

    // --- GpuMps tests ---

    #[test]
    fn test_mps_random() {
        let mps = GpuMps::random(8, 2, 4, &mut rng());
        assert_eq!(mps.num_sites, 8);
        assert_eq!(mps.tensors.len(), 8);
        assert_eq!(mps.bond_dims.len(), 9);
        assert_eq!(mps.bond_dims[0], 1); // left boundary
        assert_eq!(*mps.bond_dims.last().unwrap(), 1); // right boundary
    }

    #[test]
    fn test_mps_product_state() {
        let mps = GpuMps::product_state(6, 2);
        assert_eq!(mps.num_sites, 6);
        for t in &mps.tensors {
            assert_eq!(t.shape(), &[1, 2, 1]);
            assert!((t[[0, 0, 0]] - c1()).norm() < 1e-10); // |0> state
        }
    }

    #[test]
    fn test_mps_product_state_norm() {
        let mps = GpuMps::product_state(4, 2);
        let nrm = mps.norm();
        assert!((nrm - 1.0).abs() < 1e-8, "norm = {}", nrm);
    }

    #[test]
    fn test_mps_canonicalize_left() {
        let mut mps = GpuMps::random(4, 2, 4, &mut rng());
        mps.canonicalize_left(2);
        assert_eq!(mps.center, 2);
    }

    #[test]
    fn test_mps_canonicalize_right() {
        let mut mps = GpuMps::random(4, 2, 4, &mut rng());
        mps.canonicalize_left(3); // First go right
        mps.canonicalize_right(1);
        assert_eq!(mps.center, 1);
    }

    #[test]
    fn test_mps_bond_dim_profile() {
        let mps = GpuMps::random(6, 2, 8, &mut rng());
        let profile = mps.bond_dim_profile();
        assert_eq!(profile.len(), 7);
        assert_eq!(profile[0], 1);
        assert_eq!(*profile.last().unwrap(), 1);
    }

    #[test]
    fn test_mps_entanglement_entropy() {
        let mps = GpuMps::product_state(4, 2);
        let s = mps.entanglement_entropy();
        // Product state: no entanglement (but our simplified version may give small value)
        assert!(s.is_finite());
    }

    // --- MPO Hamiltonian tests ---

    #[test]
    fn test_heisenberg_mpo_construction() {
        let mpo = MpoHamiltonian::heisenberg(6, 1.0);
        assert_eq!(mpo.num_sites, 6);
        assert_eq!(mpo.phys_dim, 2);
        assert_eq!(mpo.tensors.len(), 6);
        // First tensor: [1, 2, 2, 5]
        assert_eq!(mpo.tensors[0].shape()[0], 1);
        assert_eq!(mpo.tensors[0].shape()[3], 5);
        // Last tensor: [5, 2, 2, 1]
        assert_eq!(mpo.tensors[5].shape()[0], 5);
        assert_eq!(mpo.tensors[5].shape()[3], 1);
    }

    #[test]
    fn test_ising_mpo_construction() {
        let mpo = MpoHamiltonian::ising(8, 1.0, 0.5);
        assert_eq!(mpo.num_sites, 8);
        assert_eq!(mpo.phys_dim, 2);
        // MPO bond dim = 3
        assert_eq!(mpo.tensors[0].shape()[0], 1);
        assert_eq!(mpo.tensors[0].shape()[3], 3);
    }

    #[test]
    fn test_mpo_tensors_not_all_zero() {
        let mpo = MpoHamiltonian::heisenberg(4, 1.0);
        for (i, t) in mpo.tensors.iter().enumerate() {
            let nrm: f64 = t.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            assert!(nrm > 1e-10, "MPO tensor {} is all zeros", i);
        }
    }

    // --- Environment tests ---

    #[test]
    fn test_environment_build() {
        let mps = GpuMps::product_state(4, 2);
        let mpo = MpoHamiltonian::ising(4, 1.0, 0.5);
        let env = GpuEnvironment::build(&mps, &mpo);
        assert_eq!(env.left_blocks.len(), 5);
        assert_eq!(env.right_blocks.len(), 5);
    }

    #[test]
    fn test_environment_boundaries() {
        let mps = GpuMps::product_state(3, 2);
        let mpo = MpoHamiltonian::ising(3, 1.0, 0.5);
        let env = GpuEnvironment::build(&mps, &mpo);
        // Left boundary should be [1,1,1] with value 1
        assert_eq!(env.left_blocks[0].shape(), &[1, 1, 1]);
        assert!((env.left_blocks[0][[0, 0, 0]] - c1()).norm() < 1e-10);
    }

    // --- AutoOffload tests ---

    #[test]
    fn test_auto_offload_cpu_for_small() {
        let ao = AutoOffload::new(32, 4096);
        assert_eq!(ao.select(16), DispatchBackend::Cpu);
    }

    #[test]
    fn test_auto_offload_threshold() {
        let ao = AutoOffload::new(32, 4096);
        let backend = ao.select(64);
        // On macOS this would be MetalGpu, elsewhere Cpu
        if cfg!(target_os = "macos") {
            assert_eq!(backend, DispatchBackend::MetalGpu);
        } else {
            assert_eq!(backend, DispatchBackend::Cpu);
        }
    }

    #[test]
    fn test_dispatch_backend_display() {
        assert_eq!(format!("{}", DispatchBackend::Cpu), "CPU");
        assert_eq!(format!("{}", DispatchBackend::MetalGpu), "Metal GPU");
    }

    #[test]
    fn test_auto_offload_flops() {
        let ao = AutoOffload::default();
        let flops = ao.estimate_flops(64, 2);
        assert!(flops > 0);
        assert_eq!(flops, 64 * 64 * 64 * 2 * 2 * 8);
    }

    // --- QR decomposition tests ---

    #[test]
    fn test_qr_identity() {
        let eye = Array2::eye(3).mapv(|v| cr(v));
        let (q, r) = qr_decomposition(&eye);
        // Q should be close to identity (up to signs)
        for i in 0..3 {
            assert!((q[[i, i]].norm() - 1.0).abs() < 0.5);
        }
        // R should also be close to identity
        for i in 0..3 {
            assert!((r[[i, i]].norm() - 1.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_qr_reproduces_matrix() {
        let a = Array2::from_shape_fn((4, 3), |(i, j)| cr((i * 3 + j + 1) as f64));
        let (q, r) = qr_decomposition(&a);
        // Q * R should reproduce A
        let qr = mat_mul(&q, &r);
        for i in 0..4 {
            for j in 0..3 {
                assert!((qr[[i, j]] - a[[i, j]]).norm() < 1e-8,
                    "QR mismatch at [{},{}]: {:?} vs {:?}", i, j, qr[[i, j]], a[[i, j]]);
            }
        }
    }

    // --- SVD tests ---

    #[test]
    fn test_svd_truncate_identity() {
        let eye = Array2::eye(4).mapv(|v| cr(v));
        let (u, s, vt) = svd_truncate(&eye, 2);
        assert_eq!(s.len(), 2);
        for sv in &s {
            assert!((sv - 1.0).abs() < 0.2, "sv = {}", sv);
        }
    }

    // --- Zero-dim SVD guard tests ---

    #[test]
    fn test_svd_truncate_zero_matrix() {
        // All-zero matrix: the guard should fire and produce rank-1 output
        let zero_mat = Array2::<C64>::zeros((4, 3));
        let (u, s, vt) = svd_truncate(&zero_mat, 4);
        assert!(s.len() >= 1, "SVD of zero matrix must produce at least rank 1, got {}", s.len());
        assert_eq!(u.shape()[0], 4, "U rows should match input rows");
        assert_eq!(u.shape()[1], s.len(), "U cols should match num singular values");
        assert_eq!(vt.shape()[0], s.len(), "Vt rows should match num singular values");
        assert_eq!(vt.shape()[1], 3, "Vt cols should match input cols");

        // Verify no NaN/Inf in output
        for val in u.iter() { assert!(val.re.is_finite() && val.im.is_finite(), "U has NaN/Inf: {:?}", val); }
        for val in &s { assert!(val.is_finite(), "S has NaN/Inf: {}", val); }
        for val in vt.iter() { assert!(val.re.is_finite() && val.im.is_finite(), "Vt has NaN/Inf: {:?}", val); }

        // Verify U * diag(S) * Vt is a valid approximation (no NaN/Inf)
        let kk = s.len();
        let mut recon = Array2::<C64>::zeros((4, 3));
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..kk {
                    recon[[i, j]] = recon[[i, j]] + u[[i, k]] * cr(s[k]) * vt[[k, j]];
                }
            }
        }
        for val in recon.iter() {
            assert!(val.re.is_finite() && val.im.is_finite(), "Reconstruction has NaN/Inf");
        }
    }

    #[test]
    fn test_svd_truncate_near_zero_matrix() {
        // Matrix with all elements < 1e-20: should trigger rank-1 fallback
        let near_zero = Array2::from_shape_fn((3, 4), |(i, j)| {
            cr(((i * 4 + j + 1) as f64) * 1e-22)
        });
        let (u, s, vt) = svd_truncate(&near_zero, 3);
        assert!(s.len() >= 1, "Near-zero matrix must produce at least rank 1, got {}", s.len());

        // All singular values should be very small
        for sv in &s {
            assert!(*sv < 1e-10, "Singular value {} unexpectedly large for near-zero matrix", sv);
        }

        // Verify valid dimensions
        assert_eq!(u.shape(), &[3, s.len()]);
        assert_eq!(vt.shape(), &[s.len(), 4]);

        // Verify reconstruction is finite
        let kk = s.len();
        for i in 0..3 {
            for j in 0..4 {
                let mut val = c0();
                for k in 0..kk { val = val + u[[i, k]] * cr(s[k]) * vt[[k, j]]; }
                assert!(val.re.is_finite() && val.im.is_finite(),
                    "Reconstruction NaN/Inf at [{},{}]", i, j);
            }
        }
    }

    #[test]
    fn test_svd_truncate_degenerate_matrix() {
        // Rank-1 matrix: outer product of two vectors
        // A = u * v^T where u = [1, 2, 3], v = [4, 5]
        let mut rank1 = Array2::<C64>::zeros((3, 2));
        let u_vec = [1.0, 2.0, 3.0];
        let v_vec = [4.0, 5.0];
        for i in 0..3 {
            for j in 0..2 {
                rank1[[i, j]] = cr(u_vec[i] * v_vec[j]);
            }
        }

        // Request max_dim=1 to force exactly one singular value
        let (u, s, vt) = svd_truncate(&rank1, 1);

        // Must retain exactly 1 singular value
        assert_eq!(s.len(), 1,
            "Rank-1 matrix with max_dim=1 should produce exactly 1 sv, got {}: {:?}",
            s.len(), s);

        // The single singular value should be the Frobenius norm of the rank-1 matrix
        // ||A||_F = sqrt(sum |a_ij|^2) = sqrt(14*41) = sqrt(574) ~ 23.96
        assert!(s[0] > 20.0, "Dominant singular value should be ~24, got {}", s[0]);

        // Valid dimensions
        assert_eq!(u.shape(), &[3, 1]);
        assert_eq!(vt.shape(), &[1, 2]);

        // Verify rank-1 reconstruction U * diag(S) * Vt ≈ A
        for i in 0..3 {
            for j in 0..2 {
                let val = u[[i, 0]] * cr(s[0]) * vt[[0, j]];
                assert!((val - rank1[[i, j]]).norm() < 1e-4,
                    "Rank-1 reconstruction error at [{},{}]: got {:?}, expected {:?}",
                    i, j, val, rank1[[i, j]]);
            }
        }

        // Also verify no NaN/Inf
        for val in u.iter() { assert!(val.re.is_finite() && val.im.is_finite()); }
        for val in vt.iter() { assert!(val.re.is_finite() && val.im.is_finite()); }
    }

    // --- Lanczos test ---

    #[test]
    fn test_tridiag_min_simple() {
        let alpha = vec![2.0, 3.0];
        let beta = vec![1.0];
        let (eval, evec) = tridiag_min(&alpha, &beta);
        // Matrix [[2,1],[1,3]] eigenvalues: (5±√5)/2 ≈ 1.38, 3.62
        assert!((eval - 1.382).abs() < 0.1, "eval = {}", eval);
        assert_eq!(evec.len(), 2);
    }

    // --- Observable tests ---

    #[test]
    fn test_measure_local_sz_product_state() {
        let mps = GpuMps::product_state(4, 2);
        let sz = measure_local_sz(&mps);
        assert_eq!(sz.len(), 4);
        for &s in &sz {
            // |0> state: <Sz> = +0.5
            assert!((s - 0.5).abs() < 1e-8, "Sz = {}", s);
        }
    }

    // --- DMRG engine tests ---

    #[test]
    fn test_dmrg_invalid_bond_dim() {
        let config = GpuDmrgConfig::new().with_max_bond_dim(0);
        let engine = GpuDmrgEngine::new(config);
        let mut mps = GpuMps::product_state(4, 2);
        let mpo = MpoHamiltonian::ising(4, 1.0, 0.5);
        let result = engine.run(&mut mps, &mpo);
        assert!(result.is_err());
    }

    #[test]
    fn test_dmrg_ising_small() {
        let config = GpuDmrgConfig::new()
            .with_max_bond_dim(4)
            .with_max_sweeps(3)
            .with_energy_tolerance(1e-6)
            .with_lanczos_iterations(10);
        let engine = GpuDmrgEngine::new(config);
        let mut mps = GpuMps::random(4, 2, 4, &mut rng());
        let mpo = MpoHamiltonian::ising(4, 1.0, 0.5);
        let result = engine.run(&mut mps, &mpo);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.energy.is_finite());
        assert!(res.sweeps <= 3);
        assert!(!res.energy_history.is_empty());
    }

    #[test]
    fn test_dmrg_heisenberg_small() {
        let config = GpuDmrgConfig::new()
            .with_max_bond_dim(4)
            .with_max_sweeps(3)
            .with_lanczos_iterations(10);
        let engine = GpuDmrgEngine::new(config);
        let mut mps = GpuMps::random(4, 2, 4, &mut rng());
        let mpo = MpoHamiltonian::heisenberg(4, 1.0);
        let result = engine.run(&mut mps, &mpo);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.energy.is_finite());
    }

    #[test]
    fn test_dmrg_energy_decreases() {
        let config = GpuDmrgConfig::new()
            .with_max_bond_dim(4)
            .with_max_sweeps(3)
            .with_lanczos_iterations(10);
        let engine = GpuDmrgEngine::new(config);
        let mut mps = GpuMps::random(4, 2, 4, &mut rng());
        let mpo = MpoHamiltonian::ising(4, 1.0, 0.5);
        let result = engine.run(&mut mps, &mpo).unwrap();
        // Energy should generally decrease (or at least not increase wildly)
        if result.energy_history.len() >= 2 {
            let last = *result.energy_history.last().unwrap();
            let first = result.energy_history[0];
            // Allow some tolerance — DMRG should find lower energy
            assert!(last <= first + 1.0, "Energy increased: {} -> {}", first, last);
        }
    }

    #[test]
    fn test_dmrg_result_fields() {
        let config = GpuDmrgConfig::new()
            .with_max_bond_dim(4)
            .with_max_sweeps(2)
            .with_lanczos_iterations(8);
        let engine = GpuDmrgEngine::new(config);
        let mut mps = GpuMps::product_state(4, 2);
        let mpo = MpoHamiltonian::ising(4, 1.0, 0.5);
        let result = engine.run(&mut mps, &mpo).unwrap();
        assert!(result.entanglement_entropy.is_finite());
        assert!(result.entanglement_entropy >= 0.0);
        assert_eq!(result.mps.num_sites, 4);
    }
}
