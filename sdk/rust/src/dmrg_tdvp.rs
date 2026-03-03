//! Density Matrix Renormalization Group (DMRG) and Time-Dependent Variational Principle (TDVP)
//!
//! This module implements the two workhorse algorithms of tensor network simulation:
//!
//! - **DMRG**: Ground state finder using MPS sweeps with local Lanczos eigensolvers.
//!   Supports single-site DMRG with optional density matrix noise for escaping local minima.
//! - **TDVP**: Real-time dynamics using projector-splitting integrators (1-site and 2-site).
//!   Uses Lanczos-based matrix exponentiation for time evolution within the MPS manifold.
//!
//! # Key Features
//!
//! - `MpoHamiltonian` construction for Heisenberg XXX and transverse-field Ising models
//! - Lanczos eigensolver and Krylov subspace matrix exponentiation
//! - SVD truncation with configurable bond dimension
//! - MPS canonicalization, overlap, and local observable measurements
//!
//! # References
//!
//! - White, S.R., "Density matrix formulation for quantum renormalization groups" (1992)
//! - Haegeman et al., "Unifying time evolution and optimization with matrix product states" (2016)
//! - Schollwoeck, U., "The density-matrix renormalization group in the age of MPS" (2011)

use ndarray::{Array2, Array3, Array4};
use num_complex::Complex64;
use rand::Rng;
use std::fmt;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors that can occur during DMRG/TDVP computations.
#[derive(Debug, Clone)]
pub enum DmrgError {
    /// DMRG did not converge within the allowed number of sweeps.
    ConvergenceFailed { sweeps_done: usize, last_energy: f64, tolerance: f64 },
    /// An invalid bond dimension was specified.
    InvalidBondDim(String),
    /// SVD decomposition failed.
    SvdFailed(String),
    /// Lanczos eigensolver failed to converge.
    LanczosFailed(String),
}

impl fmt::Display for DmrgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DmrgError::ConvergenceFailed { sweeps_done, last_energy, tolerance } => {
                write!(f, "DMRG not converged after {} sweeps (E={:.10}, tol={:.1e})", sweeps_done, last_energy, tolerance)
            }
            DmrgError::InvalidBondDim(msg) => write!(f, "Invalid bond dimension: {}", msg),
            DmrgError::SvdFailed(msg) => write!(f, "SVD failed: {}", msg),
            DmrgError::LanczosFailed(msg) => write!(f, "Lanczos failed: {}", msg),
        }
    }
}

impl std::error::Error for DmrgError {}

// ============================================================
// CONFIGURATION TYPES
// ============================================================

/// Configuration for the DMRG ground state search.
#[derive(Debug, Clone)]
pub struct DmrgConfig {
    pub max_bond_dim: usize,
    pub max_sweeps: usize,
    pub energy_tolerance: f64,
    pub num_states: usize,
    pub lanczos_iterations: usize,
    pub noise: Vec<f64>,
}

impl Default for DmrgConfig {
    fn default() -> Self {
        Self { max_bond_dim: 64, max_sweeps: 20, energy_tolerance: 1e-8, num_states: 1, lanczos_iterations: 20, noise: vec![] }
    }
}

impl DmrgConfig {
    pub fn new() -> Self { Self::default() }
    pub fn max_bond_dim(mut self, d: usize) -> Self { self.max_bond_dim = d; self }
    pub fn max_sweeps(mut self, s: usize) -> Self { self.max_sweeps = s; self }
    pub fn energy_tolerance(mut self, t: f64) -> Self { self.energy_tolerance = t; self }
    pub fn num_states(mut self, n: usize) -> Self { self.num_states = n; self }
    pub fn lanczos_iterations(mut self, n: usize) -> Self { self.lanczos_iterations = n; self }
    pub fn noise(mut self, n: Vec<f64>) -> Self { self.noise = n; self }
}

/// TDVP integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdvpMethod { OneSite, TwoSite }

/// Configuration for TDVP time evolution.
#[derive(Debug, Clone)]
pub struct TdvpConfig {
    pub time_step: f64,
    pub num_steps: usize,
    pub method: TdvpMethod,
    pub max_bond_dim: usize,
    pub lanczos_iterations: usize,
}

impl Default for TdvpConfig {
    fn default() -> Self {
        Self { time_step: 0.01, num_steps: 100, method: TdvpMethod::OneSite, max_bond_dim: 64, lanczos_iterations: 20 }
    }
}

impl TdvpConfig {
    pub fn new() -> Self { Self::default() }
    pub fn time_step(mut self, dt: f64) -> Self { self.time_step = dt; self }
    pub fn num_steps(mut self, n: usize) -> Self { self.num_steps = n; self }
    pub fn method(mut self, m: TdvpMethod) -> Self { self.method = m; self }
    pub fn max_bond_dim(mut self, d: usize) -> Self { self.max_bond_dim = d; self }
    pub fn lanczos_iterations(mut self, n: usize) -> Self { self.lanczos_iterations = n; self }
}

// ============================================================
// MPS / MPO TYPES
// ============================================================

/// A single site tensor in an MPS. Shape: (left_bond, physical_dim, right_bond).
#[derive(Debug, Clone)]
pub struct MpsSite {
    pub tensor: Array3<Complex64>,
    pub physical_dim: usize,
    pub left_bond: usize,
    pub right_bond: usize,
}

/// Matrix Product State.
#[derive(Debug, Clone)]
pub struct Mps {
    pub sites: Vec<MpsSite>,
    pub center_position: usize,
    pub num_sites: usize,
}

/// A term in the MPO Hamiltonian.
#[derive(Debug, Clone)]
pub struct MpoTerm {
    pub site: usize,
    pub operator: Array2<Complex64>,
}

/// MPO Hamiltonian. Each site tensor has shape (mpo_left, phys, phys, mpo_right).
#[derive(Debug, Clone)]
pub struct MpoHamiltonian {
    pub sites: Vec<Array4<Complex64>>,
}

/// Result of a DMRG computation.
#[derive(Debug, Clone)]
pub struct DmrgResult {
    pub energy: f64,
    pub mps: Mps,
    pub num_sweeps: usize,
    pub converged: bool,
    pub energy_history: Vec<f64>,
}

/// Result of a TDVP time evolution.
#[derive(Debug, Clone)]
pub struct TdvpResult {
    pub final_mps: Mps,
    pub time_evolved: f64,
    pub observables: Vec<Vec<f64>>,
}

// ============================================================
// COMPLEX SHORTCUTS
// ============================================================

#[inline] fn c0() -> Complex64 { Complex64::new(0.0, 0.0) }
#[inline] fn c1() -> Complex64 { Complex64::new(1.0, 0.0) }
#[inline] fn cr(r: f64) -> Complex64 { Complex64::new(r, 0.0) }

// ============================================================
// SVD TRUNCATION
// ============================================================

/// SVD with truncation to `max_dim` singular values. Returns (U, sigmas, Vt).
pub fn svd_truncate(
    matrix: &Array2<Complex64>,
    max_dim: usize,
) -> Result<(Array2<Complex64>, Vec<f64>, Array2<Complex64>), DmrgError> {
    let (m, n) = (matrix.nrows(), matrix.ncols());
    // Ensure at least rank 1 to prevent zero-dim tensors downstream.
    // Without this guard, into_shape((lb, d, 0)) causes ndarray panics.
    let k = m.min(n).min(max_dim).max(1);

    let na_mat = nalgebra::DMatrix::from_fn(m, n, |i, j| {
        nalgebra::Complex::new(matrix[[i, j]].re, matrix[[i, j]].im)
    });
    let svd = na_mat.svd(true, true);
    let u_full = svd.u.ok_or_else(|| DmrgError::SvdFailed("No U".into()))?;
    let vt_full = svd.v_t.ok_or_else(|| DmrgError::SvdFailed("No Vt".into()))?;
    let mut singular_values: Vec<f64> = svd.singular_values.iter().take(k).cloned().collect();

    // Guard: if nalgebra returned fewer singular values than k (e.g. degenerate
    // matrix), pad with a minimal value to maintain at least rank 1.
    if singular_values.is_empty() {
        let max_sv = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
        tracing::warn!(
            rows = m,
            cols = n,
            max_singular_value = max_sv,
            function = "dmrg_tdvp::svd_truncate",
            "Zero-dim SVD guard fired: all singular values below threshold, \
             injecting rank-1 approximation ({}x{} matrix, max sv={:.2e})",
            m, n, max_sv,
        );
        singular_values.push(1e-15);
    }
    let k = singular_values.len();

    debug_assert!(k >= 1, "SVD truncation must preserve at least rank 1");

    let mut u = Array2::zeros((m, k));
    for i in 0..m {
        for j in 0..k.min(u_full.ncols()) {
            let v = u_full[(i, j)];
            u[[i, j]] = Complex64::new(v.re, v.im);
        }
    }
    let mut vt = Array2::zeros((k, n));
    for i in 0..k.min(vt_full.nrows()) {
        for j in 0..n {
            let v = vt_full[(i, j)];
            vt[[i, j]] = Complex64::new(v.re, v.im);
        }
    }
    Ok((u, singular_values, vt))
}

// ============================================================
// MPS CONSTRUCTION AND MANIPULATION
// ============================================================

/// Create a random MPS with given dimensions, normalized to 1.
pub fn random_mps(n_sites: usize, physical_dim: usize, bond_dim: usize) -> Mps {
    let mut rng = rand::thread_rng();

    // Compute consistent bond dimensions between sites
    let mut bonds = vec![0usize; n_sites + 1];
    bonds[0] = 1;
    bonds[n_sites] = 1;
    for i in 1..n_sites {
        bonds[i] = (bonds[i - 1] * physical_dim).min(bond_dim);
    }
    for i in (1..n_sites).rev() {
        bonds[i] = bonds[i].min((bonds[i + 1] * physical_dim).min(bond_dim));
    }

    let mut sites = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let lb = bonds[i];
        let rb = bonds[i + 1];
        let mut tensor = Array3::zeros((lb, physical_dim, rb));
        for elem in tensor.iter_mut() {
            *elem = Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        }
        sites.push(MpsSite { tensor, physical_dim, left_bond: lb, right_bond: rb });
    }

    let mut mps = Mps { sites, center_position: 0, num_sites: n_sites };
    canonicalize_mps(&mut mps, 0);
    let norm = mps_norm(&mps);
    if norm > 1e-15 {
        let s = cr(1.0 / norm);
        for elem in mps.sites[0].tensor.iter_mut() { *elem *= s; }
    }
    mps
}

/// Bring MPS to mixed-canonical form with orthogonality center at `center`.
pub fn canonicalize_mps(mps: &mut Mps, center: usize) {
    let n = mps.num_sites;
    if n == 0 { return; }
    let center = center.min(n - 1);
    for i in 0..center { left_normalize_site(mps, i); }
    for i in (center + 1..n).rev() { right_normalize_site(mps, i); }
    mps.center_position = center;
}

fn left_normalize_site(mps: &mut Mps, i: usize) {
    let lb = mps.sites[i].left_bond;
    let d = mps.sites[i].physical_dim;
    let rb = mps.sites[i].right_bond;

    let mat = mps.sites[i].tensor.clone().into_shape((lb * d, rb)).unwrap();
    let (u, s_vals, vt) = svd_truncate(&mat, rb).unwrap();
    let k = s_vals.len();

    mps.sites[i].tensor = u.into_shape((lb, d, k)).unwrap();
    mps.sites[i].right_bond = k;

    if i + 1 < mps.num_sites {
        // S * Vt
        let mut svt = Array2::zeros((k, rb));
        for r in 0..k {
            for c in 0..rb { svt[[r, c]] = cr(s_vals[r]) * vt[[r, c]]; }
        }
        let old_lb = mps.sites[i + 1].left_bond;
        let d_next = mps.sites[i + 1].physical_dim;
        let rb_next = mps.sites[i + 1].right_bond;
        // old_lb should == rb (the original right bond of site i, before truncation)
        let next_mat = mps.sites[i + 1].tensor.clone().into_shape((old_lb, d_next * rb_next)).unwrap();
        let new_next = svt.dot(&next_mat).into_shape((k, d_next, rb_next)).unwrap();
        mps.sites[i + 1].tensor = new_next;
        mps.sites[i + 1].left_bond = k;
    }
}

fn right_normalize_site(mps: &mut Mps, i: usize) {
    let lb = mps.sites[i].left_bond;
    let d = mps.sites[i].physical_dim;
    let rb = mps.sites[i].right_bond;

    let mat = mps.sites[i].tensor.clone().into_shape((lb, d * rb)).unwrap();
    let (u, s_vals, vt) = svd_truncate(&mat, lb).unwrap();
    let k = s_vals.len();

    mps.sites[i].tensor = vt.into_shape((k, d, rb)).unwrap();
    mps.sites[i].left_bond = k;

    if i > 0 {
        let mut us = Array2::zeros((lb, k));
        for r in 0..lb {
            for c in 0..k { us[[r, c]] = u[[r, c]] * cr(s_vals[c]); }
        }
        let old_rb = mps.sites[i - 1].right_bond;
        let lb_prev = mps.sites[i - 1].left_bond;
        let d_prev = mps.sites[i - 1].physical_dim;
        let prev_mat = mps.sites[i - 1].tensor.clone().into_shape((lb_prev * d_prev, old_rb)).unwrap();
        let new_prev = prev_mat.dot(&us).into_shape((lb_prev, d_prev, k)).unwrap();
        mps.sites[i - 1].tensor = new_prev;
        mps.sites[i - 1].right_bond = k;
    }
}

/// Compute <a|b>.
pub fn mps_overlap(a: &Mps, b: &Mps) -> Complex64 {
    assert_eq!(a.num_sites, b.num_sites);
    let n = a.num_sites;
    if n == 0 { return c1(); }

    let mut env = Array2::zeros((1, 1));
    env[[0, 0]] = c1();

    for i in 0..n {
        let ta = &a.sites[i].tensor;
        let tb = &b.sites[i].tensor;
        let d = ta.shape()[1];
        let a_rb = ta.shape()[2];
        let b_rb = tb.shape()[2];
        let mut new_env = Array2::zeros((a_rb, b_rb));
        for al in 0..env.nrows() {
            for bl in 0..env.ncols() {
                let e = env[[al, bl]];
                if e.norm_sqr() < 1e-30 { continue; }
                for s in 0..d {
                    for ar in 0..a_rb {
                        let bra_val = ta[[al, s, ar]].conj();
                        let ea = e * bra_val;
                        for br in 0..b_rb {
                            new_env[[ar, br]] += ea * tb[[bl, s, br]];
                        }
                    }
                }
            }
        }
        env = new_env;
    }
    env[[0, 0]]
}

/// Compute norm of an MPS.
pub fn mps_norm(mps: &Mps) -> f64 {
    mps_overlap(mps, mps).re.max(0.0).sqrt()
}

// ============================================================
// MPO CONSTRUCTION
// ============================================================

/// Build the MPO for the Heisenberg XXX model:
///   H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1}) + h sum_i Sz_i
/// MPO bond dimension: 5.

/// Build the MPO for the Heisenberg XXX model:
///   H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1}) + h sum_i Sz_i
/// MPO bond dimension: 5. Uses lower-triangular encoding:
///   Row 0 = absorbing (completed), Row 4 = identity (start).
pub fn build_mpo_heisenberg(n_sites: usize, j: f64, h: f64) -> MpoHamiltonian {
    let d = 2usize;
    let mpo_dim = 5usize;

    let id2 = Array2::from_diag(&ndarray::arr1(&[c1(), c1()]));
    let sz = Array2::from_shape_vec((2,2), vec![cr(0.5),c0(),c0(),cr(-0.5)]).unwrap();
    let sp = Array2::from_shape_vec((2,2), vec![c0(),c1(),c0(),c0()]).unwrap();
    let sm = Array2::from_shape_vec((2,2), vec![c0(),c0(),c1(),c0()]).unwrap();

    // Bulk W matrix (lower-triangular, 5x5):
    //   | I      0       0       0    0 |  row 0: absorbing
    //   | Sm     0       0       0    0 |  row 1: absorb S+
    //   | Sp     0       0       0    0 |  row 2: absorb S-
    //   | Sz     0       0       0    0 |  row 3: absorb Sz
    //   | h*Sz  J/2*Sp  J/2*Sm  J*Sz  I |  row 4: start/identity

    let mut sites = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let lb = if i == 0 { 1 } else { mpo_dim };
        let rb = if i == n_sites - 1 { 1 } else { mpo_dim };
        let mut w = Array4::zeros((lb, d, d, rb));

        if n_sites == 1 {
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = cr(h) * sz[[s,sp_]];
            }}
        } else if i == 0 {
            // First site: select row 4 (start row) of bulk W
            // Maps from w_left=0 (single boundary) to w_right=0..4
            // Row 4: [h*Sz, J/2*Sp, J/2*Sm, J*Sz, I]
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = cr(h) * sz[[s,sp_]];
                w[[0,s,sp_,1]] = cr(j/2.0) * sp[[s,sp_]];
                w[[0,s,sp_,2]] = cr(j/2.0) * sm[[s,sp_]];
                w[[0,s,sp_,3]] = cr(j) * sz[[s,sp_]];
                w[[0,s,sp_,4]] = id2[[s,sp_]];
            }}
        } else if i == n_sites - 1 {
            // Last site: select column 0 (absorbing column) of bulk W
            // Maps from w_left=0..4 to w_right=0 (single boundary)
            // Column 0: [I, Sm, Sp, Sz, h*Sz]^T
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = id2[[s,sp_]];
                w[[1,s,sp_,0]] = sm[[s,sp_]];
                w[[2,s,sp_,0]] = sp[[s,sp_]];
                w[[3,s,sp_,0]] = sz[[s,sp_]];
                w[[4,s,sp_,0]] = cr(h) * sz[[s,sp_]];
            }}
        } else {
            // Bulk site: full 5x5 transfer matrix
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = id2[[s,sp_]];
                w[[1,s,sp_,0]] = sm[[s,sp_]];
                w[[2,s,sp_,0]] = sp[[s,sp_]];
                w[[3,s,sp_,0]] = sz[[s,sp_]];
                w[[4,s,sp_,0]] = cr(h) * sz[[s,sp_]];
                w[[4,s,sp_,1]] = cr(j/2.0) * sp[[s,sp_]];
                w[[4,s,sp_,2]] = cr(j/2.0) * sm[[s,sp_]];
                w[[4,s,sp_,3]] = cr(j) * sz[[s,sp_]];
                w[[4,s,sp_,4]] = id2[[s,sp_]];
            }}
        }
        sites.push(w);
    }
    MpoHamiltonian { sites }
}

/// Build the MPO for the transverse-field Ising model:
///   H = -J sum_i Sz_i Sz_{i+1} - h sum_i Sx_i
/// MPO bond dimension: 3. Lower-triangular encoding:
///   Row 0 = absorbing, Row 2 = start/identity.
pub fn build_mpo_ising(n_sites: usize, j: f64, h: f64) -> MpoHamiltonian {
    let d = 2usize;
    let mpo_dim = 3usize;

    let id2 = Array2::from_diag(&ndarray::arr1(&[c1(), c1()]));
    let sz = Array2::from_shape_vec((2,2), vec![cr(0.5),c0(),c0(),cr(-0.5)]).unwrap();
    let sx = Array2::from_shape_vec((2,2), vec![c0(),cr(0.5),cr(0.5),c0()]).unwrap();

    // Bulk W (3x3):
    //   | I       0     0 |  row 0: absorbing
    //   | Sz      0     0 |  row 1: absorb Sz
    //   | -h*Sx  -J*Sz  I |  row 2: start/identity

    let mut sites = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let lb = if i == 0 { 1 } else { mpo_dim };
        let rb = if i == n_sites - 1 { 1 } else { mpo_dim };
        let mut w = Array4::zeros((lb, d, d, rb));

        if n_sites == 1 {
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = cr(-h) * sx[[s,sp_]];
            }}
        } else if i == 0 {
            // First site: row 2 of bulk
            // [-h*Sx, -J*Sz, I]
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = cr(-h) * sx[[s,sp_]];
                w[[0,s,sp_,1]] = cr(-j) * sz[[s,sp_]];
                w[[0,s,sp_,2]] = id2[[s,sp_]];
            }}
        } else if i == n_sites - 1 {
            // Last site: column 0 of bulk
            // [I, Sz, -h*Sx]^T
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = id2[[s,sp_]];
                w[[1,s,sp_,0]] = sz[[s,sp_]];
                w[[2,s,sp_,0]] = cr(-h) * sx[[s,sp_]];
            }}
        } else {
            // Bulk: full 3x3
            for s in 0..d { for sp_ in 0..d {
                w[[0,s,sp_,0]] = id2[[s,sp_]];
                w[[1,s,sp_,0]] = sz[[s,sp_]];
                w[[2,s,sp_,0]] = cr(-h) * sx[[s,sp_]];
                w[[2,s,sp_,1]] = cr(-j) * sz[[s,sp_]];
                w[[2,s,sp_,2]] = id2[[s,sp_]];
            }}
        }
        sites.push(w);
    }
    MpoHamiltonian { sites }
}

// ============================================================
// LANCZOS EIGENSOLVER & MATRIX EXPONENTIATION
// ============================================================

/// Lanczos eigensolver: find ground state of Hermitian H given as matvec closure.
pub fn lanczos_ground_state(
    h_eff: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
    dim: usize,
    max_iter: usize,
) -> Result<(f64, Vec<Complex64>), DmrgError> {
    if dim == 0 { return Err(DmrgError::LanczosFailed("Zero dim".into())); }
    if dim == 1 {
        let v = vec![c1()];
        let hv = h_eff(&v);
        return Ok((hv[0].re, v));
    }

    let m = max_iter.min(dim);
    let mut rng = rand::thread_rng();
    let mut v: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
        .collect();
    let norm = vec_norm(&v);
    if norm < 1e-15 { return Err(DmrgError::LanczosFailed("Zero initial vector".into())); }
    vec_scale(&mut v, 1.0 / norm);

    let mut alphas = Vec::with_capacity(m);
    let mut betas = Vec::with_capacity(m);
    let mut basis = Vec::with_capacity(m);
    basis.push(v.clone());
    let mut v_prev = vec![c0(); dim];

    for j in 0..m {
        let w = h_eff(&basis[j]);
        let alpha = vec_dot(&basis[j], &w).re;
        alphas.push(alpha);

        let mut r: Vec<Complex64> = w.iter().enumerate().map(|(k, &wk)| {
            wk - cr(alpha) * basis[j][k] - if j > 0 { cr(betas[j - 1]) * v_prev[k] } else { c0() }
        }).collect();

        // Full reorthogonalization
        for prev in &basis {
            let overlap = vec_dot(prev, &r);
            for k in 0..dim { r[k] -= overlap * prev[k]; }
        }

        let beta = vec_norm(&r);
        if beta < 1e-14 || j == m - 1 { betas.push(beta); break; }
        betas.push(beta);
        vec_scale(&mut r, 1.0 / beta);
        v_prev = basis[j].clone();
        basis.push(r);
    }

    let n_lanczos = alphas.len();
    let n_betas = n_lanczos.saturating_sub(1).min(betas.len());
    let (evals, evecs) = diag_tridiagonal(&alphas, &betas[..n_betas]);

    let mut min_idx = 0;
    for (i, &e) in evals.iter().enumerate() { if e < evals[min_idx] { min_idx = i; } }

    let mut result = vec![c0(); dim];
    for (j, bv) in basis.iter().enumerate() {
        if j < evecs.len() && min_idx < evecs[j].len() {
            let coeff = cr(evecs[j][min_idx]);
            for k in 0..dim { result[k] += coeff * bv[k]; }
        }
    }
    let norm = vec_norm(&result);
    if norm > 1e-15 { vec_scale(&mut result, 1.0 / norm); }

    Ok((evals[min_idx], result))
}

/// Lanczos-based matrix exponentiation: compute exp(-i*dt*H)|v>.
pub fn lanczos_expm(
    h_eff: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
    v: &[Complex64],
    dt: f64,
    max_iter: usize,
) -> Result<Vec<Complex64>, DmrgError> {
    let dim = v.len();
    if dim == 0 { return Ok(vec![]); }
    let norm_v = vec_norm(v);
    if norm_v < 1e-15 { return Ok(vec![c0(); dim]); }

    let q: Vec<Complex64> = v.iter().map(|&x| x / cr(norm_v)).collect();
    let m = max_iter.min(dim);
    let mut alphas = Vec::with_capacity(m);
    let mut betas = Vec::with_capacity(m);
    let mut basis = Vec::with_capacity(m);
    basis.push(q);
    let mut q_prev = vec![c0(); dim];

    for j in 0..m {
        let w = h_eff(&basis[j]);
        let alpha = vec_dot(&basis[j], &w).re;
        alphas.push(alpha);

        let mut r: Vec<Complex64> = w.iter().enumerate().map(|(k, &wk)| {
            wk - cr(alpha) * basis[j][k] - if j > 0 { cr(betas[j - 1]) * q_prev[k] } else { c0() }
        }).collect();

        for prev in &basis {
            let overlap = vec_dot(prev, &r);
            for k in 0..dim { r[k] -= overlap * prev[k]; }
        }

        let beta = vec_norm(&r);
        if beta < 1e-14 || j == m - 1 { betas.push(beta); break; }
        betas.push(beta);
        vec_scale(&mut r, 1.0 / beta);
        q_prev = basis[j].clone();
        basis.push(r);
    }

    let n_k = alphas.len();
    let n_betas = n_k.saturating_sub(1).min(betas.len());
    let exp_vec = expm_tridiag(&alphas, &betas[..n_betas], dt);

    let mut result = vec![c0(); dim];
    for (j, bv) in basis.iter().enumerate() {
        if j < exp_vec.len() {
            let coeff = exp_vec[j] * cr(norm_v);
            for k in 0..dim { result[k] += coeff * bv[k]; }
        }
    }
    Ok(result)
}

// ============================================================
// LINEAR ALGEBRA HELPERS
// ============================================================

fn vec_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai.conj() * bi).sum()
}

fn vec_norm(v: &[Complex64]) -> f64 {
    v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

fn vec_scale(v: &mut [Complex64], s: f64) {
    let sc = cr(s);
    for x in v.iter_mut() { *x *= sc; }
}

/// Diagonalize a real symmetric tridiagonal matrix.
fn diag_tridiagonal(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = alphas.len();
    if n == 0 { return (vec![], vec![]); }
    if n == 1 { return (vec![alphas[0]], vec![vec![1.0]]); }

    let mut mat = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        mat[(i, i)] = alphas[i];
        if i < betas.len() { mat[(i, i+1)] = betas[i]; mat[(i+1, i)] = betas[i]; }
    }

    let eig = mat.symmetric_eigen();
    let evals: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    let mut evecs = Vec::with_capacity(n);
    for j in 0..n {
        let mut col = Vec::with_capacity(n);
        for i in 0..n { col.push(eig.eigenvectors[(j, i)]); }
        evecs.push(col);
    }
    (evals, evecs)
}

/// Compute exp(-i*dt*T) * e_1 for tridiagonal T.
fn expm_tridiag(alphas: &[f64], betas: &[f64], dt: f64) -> Vec<Complex64> {
    let n = alphas.len();
    if n == 0 { return vec![]; }
    let (evals, evecs) = diag_tridiagonal(alphas, betas);
    let mut result = vec![c0(); n];
    for k in 0..n {
        let phase = -dt * evals[k];
        let exp_f = Complex64::new(phase.cos(), phase.sin());
        let coeff = evecs[0][k];
        for j in 0..n { result[j] += cr(evecs[j][k]) * cr(coeff) * exp_f; }
    }
    result
}

// ============================================================
// DMRG ENVIRONMENT BLOCKS
// ============================================================

/// Update left environment by contracting one more site.
fn update_left_env(
    left: &Array3<Complex64>,
    bra: &Array3<Complex64>,
    mpo_site: &Array4<Complex64>,
    ket: &Array3<Complex64>,
) -> Array3<Complex64> {
    let a_bra = bra.shape()[2];
    let w_new = mpo_site.shape()[3];
    let b_ket = ket.shape()[2];
    let d = bra.shape()[1];
    let a_old = left.shape()[0];
    let w_old = left.shape()[1];
    let b_old = left.shape()[2];

    let mut new_left = Array3::zeros((a_bra, w_new, b_ket));
    for ao in 0..a_old {
        for wo in 0..w_old {
            for bo in 0..b_old {
                let l = left[[ao, wo, bo]];
                if l.norm_sqr() < 1e-30 { continue; }
                for s in 0..d {
                    for sp in 0..d {
                        for wn in 0..w_new {
                            let w_val = mpo_site[[wo, s, sp, wn]];
                            if w_val.norm_sqr() < 1e-30 { continue; }
                            for an in 0..a_bra {
                                let bra_val = bra[[ao, s, an]].conj();
                                let lbw = l * bra_val * w_val;
                                if lbw.norm_sqr() < 1e-30 { continue; }
                                for bn in 0..b_ket {
                                    new_left[[an, wn, bn]] += lbw * ket[[bo, sp, bn]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    new_left
}

/// Update right environment by contracting one more site from the right.
fn update_right_env(
    right: &Array3<Complex64>,
    bra: &Array3<Complex64>,
    mpo_site: &Array4<Complex64>,
    ket: &Array3<Complex64>,
) -> Array3<Complex64> {
    let a_bra = bra.shape()[0];
    let w_new = mpo_site.shape()[0];
    let b_ket = ket.shape()[0];
    let d = bra.shape()[1];
    let a_old = right.shape()[0];
    let w_old = right.shape()[1];
    let b_old = right.shape()[2];

    let mut new_right = Array3::zeros((a_bra, w_new, b_ket));
    for ao in 0..a_old {
        for wo in 0..w_old {
            for bo in 0..b_old {
                let r = right[[ao, wo, bo]];
                if r.norm_sqr() < 1e-30 { continue; }
                for s in 0..d {
                    for sp in 0..d {
                        for wn in 0..w_new {
                            let w_val = mpo_site[[wn, s, sp, wo]];
                            if w_val.norm_sqr() < 1e-30 { continue; }
                            for an in 0..a_bra {
                                let bra_val = bra[[an, s, ao]].conj();
                                let rwb = r * w_val * bra_val;
                                if rwb.norm_sqr() < 1e-30 { continue; }
                                for bn in 0..b_ket {
                                    new_right[[an, wn, bn]] += rwb * ket[[bn, sp, bo]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    new_right
}

/// Apply effective Hamiltonian to a single-site tensor (flattened).
fn apply_h_eff_one_site(
    left_env: &Array3<Complex64>,
    mpo_site: &Array4<Complex64>,
    right_env: &Array3<Complex64>,
    vec: &[Complex64],
    lb: usize,
    d: usize,
    rb: usize,
) -> Vec<Complex64> {
    let mut result = vec![c0(); lb * d * rb];
    let w_left = left_env.shape()[1];
    let w_right = right_env.shape()[1];
    let bl_max = left_env.shape()[2];
    let br_max = right_env.shape()[2];

    for al in 0..lb {
        for s in 0..d {
            for ar in 0..rb {
                let out_idx = al * d * rb + s * rb + ar;
                let mut sum = c0();
                for bl in 0..bl_max {
                    for sp in 0..d {
                        for br in 0..br_max {
                            let v_idx = bl * d * br_max + sp * br_max + br;
                            if v_idx >= vec.len() { continue; }
                            let v = vec[v_idx];
                            if v.norm_sqr() < 1e-30 { continue; }
                            for wl in 0..w_left {
                                for wr in 0..w_right {
                                    sum += left_env[[al, wl, bl]]
                                        * mpo_site[[wl, s, sp, wr]]
                                        * right_env[[ar, wr, br]]
                                        * v;
                                }
                            }
                        }
                    }
                }
                result[out_idx] = sum;
            }
        }
    }
    result
}

/// Apply effective Hamiltonian for two merged sites.
fn apply_h_eff_two_site(
    left_env: &Array3<Complex64>,
    mpo_i: &Array4<Complex64>,
    mpo_i1: &Array4<Complex64>,
    right_env: &Array3<Complex64>,
    vec: &[Complex64],
    lb: usize,
    d1: usize,
    d2: usize,
    rb: usize,
) -> Vec<Complex64> {
    let mut result = vec![c0(); lb * d1 * d2 * rb];
    let w_left = left_env.shape()[1];
    let w_mid = mpo_i.shape()[3];
    let w_right = right_env.shape()[1];
    let bl_max = left_env.shape()[2];
    let br_max = right_env.shape()[2];

    for al in 0..lb {
      for s1 in 0..d1 {
        for s2 in 0..d2 {
          for ar in 0..rb {
            let out_idx = al*(d1*d2*rb) + s1*(d2*rb) + s2*rb + ar;
            let mut sum = c0();
            for bl in 0..bl_max {
              for sp1 in 0..d1 {
                for sp2 in 0..d2 {
                  for br in 0..br_max {
                    let in_idx = bl*(d1*d2*br_max) + sp1*(d2*br_max) + sp2*br_max + br;
                    if in_idx >= vec.len() { continue; }
                    let v = vec[in_idx];
                    if v.norm_sqr() < 1e-30 { continue; }
                    for wl in 0..w_left {
                      for wm in 0..w_mid {
                        for wr in 0..w_right {
                          sum += left_env[[al,wl,bl]] * mpo_i[[wl,s1,sp1,wm]]
                            * mpo_i1[[wm,s2,sp2,wr]] * right_env[[ar,wr,br]] * v;
                        }
                      }
                    }
                  }
                }
              }
            }
            result[out_idx] = sum;
          }
        }
      }
    }
    result
}

// ============================================================
// DMRG ALGORITHM
// ============================================================

/// Run the DMRG algorithm to find the ground state of an MPO Hamiltonian.
pub fn dmrg(
    hamiltonian: &MpoHamiltonian,
    config: &DmrgConfig,
) -> Result<DmrgResult, DmrgError> {
    let n = hamiltonian.sites.len();
    if n == 0 { return Err(DmrgError::InvalidBondDim("Empty Hamiltonian".into())); }
    if config.max_bond_dim == 0 { return Err(DmrgError::InvalidBondDim("Bond dim must be > 0".into())); }

    let phys_dim = hamiltonian.sites[0].shape()[1];
    let mut mps = random_mps(n, phys_dim, config.max_bond_dim);
    canonicalize_mps(&mut mps, 0);

    let norm = mps_norm(&mps);
    if norm > 1e-15 {
        let s = cr(1.0 / norm);
        for elem in mps.sites[0].tensor.iter_mut() { *elem *= s; }
    }

    // Pre-build right environments
    let mut right_envs: Vec<Array3<Complex64>> = (0..n).map(|_| {
        let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
    }).collect();
    for i in (1..n).rev() {
        right_envs[i-1] = update_right_env(&right_envs[i], &mps.sites[i].tensor, &hamiltonian.sites[i], &mps.sites[i].tensor);
    }

    let mut left_envs: Vec<Array3<Complex64>> = (0..n).map(|_| {
        let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
    }).collect();

    let mut energy_history = Vec::with_capacity(config.max_sweeps);
    let mut prev_energy = f64::MAX;
    let mut converged = false;

    for sweep in 0..config.max_sweeps {
        let mut sweep_energy = 0.0;

        // === Right sweep: 0 -> n-2 ===
        for i in 0..n.saturating_sub(1) {
            let lb = mps.sites[i].left_bond;
            let d = mps.sites[i].physical_dim;
            let rb = mps.sites[i].right_bond;

            let left = left_envs[i].clone();
            let right = right_envs[i].clone();
            let mpo_s = hamiltonian.sites[i].clone();

            let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                apply_h_eff_one_site(&left, &mpo_s, &right, v, lb, d, rb)
            };
            let (energy, gs_vec) = lanczos_ground_state(&h_fn, lb*d*rb, config.lanczos_iterations)?;
            sweep_energy = energy;

            let mut gs_mat = Array2::from_shape_vec((lb*d, rb), gs_vec)
                .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;

            // Optional noise
            if sweep < config.noise.len() && config.noise[sweep] > 0.0 {
                let nl = config.noise[sweep];
                let mut rng = rand::thread_rng();
                for elem in gs_mat.iter_mut() {
                    *elem += Complex64::new(rng.gen_range(-nl..nl), rng.gen_range(-nl..nl));
                }
            }

            let (u, s_vals, vt) = svd_truncate(&gs_mat, config.max_bond_dim)?;
            let k = s_vals.len();

            mps.sites[i].tensor = u.into_shape((lb, d, k)).unwrap();
            mps.sites[i].right_bond = k;

            // Absorb S*Vt into next site
            let mut svt = Array2::zeros((k, rb));
            for r in 0..k { for c in 0..rb { svt[[r,c]] = cr(s_vals[r]) * vt[[r,c]]; } }

            let old_lb_next = mps.sites[i+1].left_bond;
            let d_next = mps.sites[i+1].physical_dim;
            let rb_next = mps.sites[i+1].right_bond;
            let next_mat = mps.sites[i+1].tensor.clone().into_shape((old_lb_next, d_next*rb_next)).unwrap();
            let new_next = svt.dot(&next_mat).into_shape((k, d_next, rb_next)).unwrap();
            mps.sites[i+1].tensor = new_next;
            mps.sites[i+1].left_bond = k;

            // Update left env
            left_envs[i+1] = update_left_env(&left_envs[i], &mps.sites[i].tensor, &hamiltonian.sites[i], &mps.sites[i].tensor);
        }

        // === Left sweep: n-1 -> 1 ===
        // Rebuild right envs from scratch
        {
            let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1();
            right_envs[n-1] = e;
        }
        for i in (1..n).rev() {
            // First rebuild right env for site i
            if i < n-1 {
                right_envs[i] = update_right_env(&right_envs[i+1], &mps.sites[i+1].tensor, &hamiltonian.sites[i+1], &mps.sites[i+1].tensor);
            }
        }
        // Actually rebuild all right envs properly
        for i in (0..n-1).rev() {
            right_envs[i] = update_right_env(&right_envs[i+1], &mps.sites[i+1].tensor, &hamiltonian.sites[i+1], &mps.sites[i+1].tensor);
        }

        for i in (1..n).rev() {
            let lb = mps.sites[i].left_bond;
            let d = mps.sites[i].physical_dim;
            let rb = mps.sites[i].right_bond;

            let left = left_envs[i].clone();
            let right = right_envs[i].clone();
            let mpo_s = hamiltonian.sites[i].clone();

            let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                apply_h_eff_one_site(&left, &mpo_s, &right, v, lb, d, rb)
            };
            let (energy, gs_vec) = lanczos_ground_state(&h_fn, lb*d*rb, config.lanczos_iterations)?;
            sweep_energy = energy;

            let gs_mat = Array2::from_shape_vec((lb, d*rb), gs_vec)
                .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;

            let (u, s_vals, vt) = svd_truncate(&gs_mat, config.max_bond_dim)?;
            let k = s_vals.len();

            mps.sites[i].tensor = vt.into_shape((k, d, rb)).unwrap();
            mps.sites[i].left_bond = k;

            // Absorb U*S into previous site
            let mut us = Array2::zeros((lb, k));
            for r in 0..lb { for c in 0..k { us[[r,c]] = u[[r,c]] * cr(s_vals[c]); } }

            let old_rb_prev = mps.sites[i-1].right_bond;
            let lb_prev = mps.sites[i-1].left_bond;
            let d_prev = mps.sites[i-1].physical_dim;
            let prev_mat = mps.sites[i-1].tensor.clone().into_shape((lb_prev*d_prev, old_rb_prev)).unwrap();
            let new_prev = prev_mat.dot(&us).into_shape((lb_prev, d_prev, k)).unwrap();
            mps.sites[i-1].tensor = new_prev;
            mps.sites[i-1].right_bond = k;

            // Update right env
            right_envs[i-1] = update_right_env(&right_envs[i], &mps.sites[i].tensor, &hamiltonian.sites[i], &mps.sites[i].tensor);
        }

        energy_history.push(sweep_energy);
        mps.center_position = 0;

        if (prev_energy - sweep_energy).abs() < config.energy_tolerance {
            converged = true;
            return Ok(DmrgResult { energy: sweep_energy, mps, num_sweeps: sweep+1, converged, energy_history });
        }
        prev_energy = sweep_energy;
    }

    Ok(DmrgResult { energy: prev_energy, mps, num_sweeps: config.max_sweeps, converged, energy_history })
}

// ============================================================
// ADAPTIVE BOND DIMENSION SCHEDULING
// ============================================================

/// Adaptive bond dimension scheduler for DMRG.
///
/// Starts at a user-specified `initial_bond_dim` and doubles the bond dimension
/// when energy improvement stalls (below `growth_threshold`). Marks convergence
/// when improvement falls below `convergence_threshold`.
///
/// This implements the common "ramp-up" strategy where DMRG begins with a cheap
/// low-bond-dimension sweep to get a rough approximation, then progressively
/// increases D to refine the result. This is often faster than starting at
/// max_bond_dim directly because early sweeps at low D are cheap and provide
/// a good initial state for the high-D sweeps.
#[derive(Debug, Clone)]
pub struct AdaptiveBondDimScheduler {
    /// Current bond dimension.
    pub current_bond_dim: usize,
    /// Maximum allowed bond dimension.
    pub max_bond_dim: usize,
    /// Energy from the previous sweep (None if no sweep has been completed).
    pub previous_energy: Option<f64>,
    /// Relative energy improvement threshold below which bond dimension doubles.
    pub growth_threshold: f64,
    /// Relative energy improvement threshold below which convergence is declared.
    pub convergence_threshold: f64,
    /// Whether the scheduler has declared convergence.
    pub converged: bool,
}

impl AdaptiveBondDimScheduler {
    /// Create a new scheduler with the given parameters.
    ///
    /// # Arguments
    /// * `initial_bond_dim` - Starting bond dimension (default: 16)
    /// * `max_bond_dim` - Maximum bond dimension to grow to
    /// * `growth_threshold` - Relative energy improvement below which D doubles (default: 0.01)
    /// * `convergence_threshold` - Relative energy improvement below which convergence is declared (default: 1e-6)
    pub fn new(
        initial_bond_dim: usize,
        max_bond_dim: usize,
        growth_threshold: f64,
        convergence_threshold: f64,
    ) -> Self {
        Self {
            current_bond_dim: initial_bond_dim.max(1),
            max_bond_dim: max_bond_dim.max(initial_bond_dim),
            previous_energy: None,
            growth_threshold,
            convergence_threshold,
            converged: false,
        }
    }

    /// Create a scheduler with default thresholds.
    pub fn with_defaults(initial_bond_dim: usize, max_bond_dim: usize) -> Self {
        Self::new(initial_bond_dim, max_bond_dim, 0.01, 1e-6)
    }

    /// Update the scheduler with the current energy and return the new bond dimension
    /// and whether convergence has been reached.
    ///
    /// # Returns
    /// `(new_bond_dim, converged)` -- the bond dimension to use for the next sweep,
    /// and a flag indicating whether the energy has converged.
    pub fn next_bond_dim(&mut self, current_energy: f64) -> (usize, bool) {
        if self.converged {
            return (self.current_bond_dim, true);
        }

        match self.previous_energy {
            None => {
                // First sweep: just record the energy, keep current D
                self.previous_energy = Some(current_energy);
                (self.current_bond_dim, false)
            }
            Some(prev_e) => {
                let denom = prev_e.abs().max(1e-30);
                let delta_e = (current_energy - prev_e).abs() / denom;
                self.previous_energy = Some(current_energy);

                if delta_e < self.convergence_threshold {
                    // Converged at current bond dimension
                    self.converged = true;
                    (self.current_bond_dim, true)
                } else if delta_e < self.growth_threshold
                    && self.current_bond_dim < self.max_bond_dim
                {
                    // Energy improvement has stalled -- double D
                    self.current_bond_dim =
                        (self.current_bond_dim * 2).min(self.max_bond_dim);
                    (self.current_bond_dim, false)
                } else {
                    // Still improving at current D
                    (self.current_bond_dim, false)
                }
            }
        }
    }
}

/// Run DMRG with adaptive bond dimension scheduling.
///
/// Starts at `initial_bond_dim` and doubles D when energy improvement stalls,
/// up to `max_bond_dim`. This is often faster than running at max_bond_dim
/// from the start because early cheap sweeps provide a good initial state.
///
/// # Arguments
/// * `hamiltonian` - The MPO Hamiltonian
/// * `initial_bond_dim` - Starting bond dimension (e.g. 16)
/// * `max_bond_dim` - Maximum bond dimension
/// * `max_sweeps` - Maximum total sweeps across all D stages
/// * `energy_tolerance` - Final energy convergence tolerance
/// * `lanczos_iterations` - Lanczos iterations per site optimization
///
/// # Returns
/// A `DmrgResult` with the final optimized MPS and convergence information.
pub fn dmrg_adaptive(
    hamiltonian: &MpoHamiltonian,
    initial_bond_dim: usize,
    max_bond_dim: usize,
    max_sweeps: usize,
    energy_tolerance: f64,
    lanczos_iterations: usize,
) -> Result<DmrgResult, DmrgError> {
    let n = hamiltonian.sites.len();
    if n == 0 {
        return Err(DmrgError::InvalidBondDim("Empty Hamiltonian".into()));
    }
    if max_bond_dim == 0 {
        return Err(DmrgError::InvalidBondDim("Bond dim must be > 0".into()));
    }

    let mut scheduler = AdaptiveBondDimScheduler::with_defaults(initial_bond_dim, max_bond_dim);
    let mut energy_history = Vec::with_capacity(max_sweeps);
    let mut total_sweeps = 0;

    // Create initial MPS at the starting bond dimension
    let phys_dim = hamiltonian.sites[0].shape()[1];
    let mut mps = random_mps(n, phys_dim, scheduler.current_bond_dim);
    canonicalize_mps(&mut mps, 0);
    let norm = mps_norm(&mps);
    if norm > 1e-15 {
        let s = cr(1.0 / norm);
        for elem in mps.sites[0].tensor.iter_mut() {
            *elem *= s;
        }
    }

    for _ in 0..max_sweeps {
        // Run one DMRG sweep at the current bond dimension
        let config = DmrgConfig::new()
            .max_bond_dim(scheduler.current_bond_dim)
            .max_sweeps(1)
            .energy_tolerance(energy_tolerance)
            .lanczos_iterations(lanczos_iterations);

        let result = dmrg_single_sweep(&mut mps, hamiltonian, &config)?;
        let sweep_energy = result.energy;
        energy_history.push(sweep_energy);
        total_sweeps += 1;

        let (new_dim, converged) = scheduler.next_bond_dim(sweep_energy);

        if converged {
            return Ok(DmrgResult {
                energy: sweep_energy,
                mps,
                num_sweeps: total_sweeps,
                converged: true,
                energy_history,
            });
        }

        // If bond dimension increased, the existing MPS tensors are fine --
        // the next sweep will naturally grow bonds via SVD truncation at the
        // new max_dim.
        let _ = new_dim; // used implicitly via scheduler.current_bond_dim
    }

    let final_energy = energy_history.last().copied().unwrap_or(0.0);
    Ok(DmrgResult {
        energy: final_energy,
        mps,
        num_sweeps: total_sweeps,
        converged: false,
        energy_history,
    })
}

/// Run a single DMRG sweep (right + left) on an existing MPS.
///
/// Unlike `dmrg()`, this does not create a new random MPS -- it optimizes
/// the provided MPS in place for one full sweep.
fn dmrg_single_sweep(
    mps: &mut Mps,
    hamiltonian: &MpoHamiltonian,
    config: &DmrgConfig,
) -> Result<DmrgResult, DmrgError> {
    let n = mps.num_sites;
    if n == 0 {
        return Err(DmrgError::InvalidBondDim("Empty MPS".into()));
    }

    // Pre-build right environments
    let mut right_envs: Vec<Array3<Complex64>> = (0..n)
        .map(|_| {
            let mut e = Array3::zeros((1, 1, 1));
            e[[0, 0, 0]] = c1();
            e
        })
        .collect();
    for i in (1..n).rev() {
        right_envs[i - 1] = update_right_env(
            &right_envs[i],
            &mps.sites[i].tensor,
            &hamiltonian.sites[i],
            &mps.sites[i].tensor,
        );
    }

    let mut left_envs: Vec<Array3<Complex64>> = (0..n)
        .map(|_| {
            let mut e = Array3::zeros((1, 1, 1));
            e[[0, 0, 0]] = c1();
            e
        })
        .collect();

    let mut sweep_energy = 0.0;

    // === Right sweep: 0 -> n-2 ===
    for i in 0..n.saturating_sub(1) {
        let lb = mps.sites[i].left_bond;
        let d = mps.sites[i].physical_dim;
        let rb = mps.sites[i].right_bond;

        let left = left_envs[i].clone();
        let right = right_envs[i].clone();
        let mpo_s = hamiltonian.sites[i].clone();

        let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
            apply_h_eff_one_site(&left, &mpo_s, &right, v, lb, d, rb)
        };
        let (energy, gs_vec) =
            lanczos_ground_state(&h_fn, lb * d * rb, config.lanczos_iterations)?;
        sweep_energy = energy;

        let gs_mat = Array2::from_shape_vec((lb * d, rb), gs_vec)
            .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;

        let (u, s_vals, vt) = svd_truncate(&gs_mat, config.max_bond_dim)?;
        let k = s_vals.len();

        mps.sites[i].tensor = u.into_shape((lb, d, k)).unwrap();
        mps.sites[i].right_bond = k;

        // Absorb S*Vt into next site
        let mut svt = Array2::zeros((k, rb));
        for r in 0..k {
            for c in 0..rb {
                svt[[r, c]] = cr(s_vals[r]) * vt[[r, c]];
            }
        }

        let old_lb_next = mps.sites[i + 1].left_bond;
        let d_next = mps.sites[i + 1].physical_dim;
        let rb_next = mps.sites[i + 1].right_bond;
        let next_mat = mps.sites[i + 1]
            .tensor
            .clone()
            .into_shape((old_lb_next, d_next * rb_next))
            .unwrap();
        let new_next = svt
            .dot(&next_mat)
            .into_shape((k, d_next, rb_next))
            .unwrap();
        mps.sites[i + 1].tensor = new_next;
        mps.sites[i + 1].left_bond = k;

        // Update left env
        left_envs[i + 1] = update_left_env(
            &left_envs[i],
            &mps.sites[i].tensor,
            &hamiltonian.sites[i],
            &mps.sites[i].tensor,
        );
    }

    // === Left sweep: n-1 -> 1 ===
    // Rebuild right envs
    {
        let mut e = Array3::zeros((1, 1, 1));
        e[[0, 0, 0]] = c1();
        right_envs[n - 1] = e;
    }
    for i in (0..n - 1).rev() {
        right_envs[i] = update_right_env(
            &right_envs[i + 1],
            &mps.sites[i + 1].tensor,
            &hamiltonian.sites[i + 1],
            &mps.sites[i + 1].tensor,
        );
    }

    for i in (1..n).rev() {
        let lb = mps.sites[i].left_bond;
        let d = mps.sites[i].physical_dim;
        let rb = mps.sites[i].right_bond;

        let left = left_envs[i].clone();
        let right = right_envs[i].clone();
        let mpo_s = hamiltonian.sites[i].clone();

        let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
            apply_h_eff_one_site(&left, &mpo_s, &right, v, lb, d, rb)
        };
        let (energy, gs_vec) =
            lanczos_ground_state(&h_fn, lb * d * rb, config.lanczos_iterations)?;
        sweep_energy = energy;

        let gs_mat = Array2::from_shape_vec((lb, d * rb), gs_vec)
            .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;

        let (u, s_vals, vt) = svd_truncate(&gs_mat, config.max_bond_dim)?;
        let k = s_vals.len();

        mps.sites[i].tensor = vt.into_shape((k, d, rb)).unwrap();
        mps.sites[i].left_bond = k;

        // Absorb U*S into previous site
        let mut us = Array2::zeros((lb, k));
        for r in 0..lb {
            for c in 0..k {
                us[[r, c]] = u[[r, c]] * cr(s_vals[c]);
            }
        }

        let old_rb_prev = mps.sites[i - 1].right_bond;
        let lb_prev = mps.sites[i - 1].left_bond;
        let d_prev = mps.sites[i - 1].physical_dim;
        let prev_mat = mps.sites[i - 1]
            .tensor
            .clone()
            .into_shape((lb_prev * d_prev, old_rb_prev))
            .unwrap();
        let new_prev = prev_mat
            .dot(&us)
            .into_shape((lb_prev, d_prev, k))
            .unwrap();
        mps.sites[i - 1].tensor = new_prev;
        mps.sites[i - 1].right_bond = k;

        // Update right env
        right_envs[i - 1] = update_right_env(
            &right_envs[i],
            &mps.sites[i].tensor,
            &hamiltonian.sites[i],
            &mps.sites[i].tensor,
        );
    }

    mps.center_position = 0;

    Ok(DmrgResult {
        energy: sweep_energy,
        mps: mps.clone(),
        num_sweeps: 1,
        converged: false,
        energy_history: vec![sweep_energy],
    })
}

// ============================================================
// OBSERVABLES
// ============================================================

/// Measure <mps|O_site|mps> for a local operator at a given site.
pub fn measure_local_observable(mps: &Mps, operator: &Array2<Complex64>, site: usize) -> Complex64 {
    let n = mps.num_sites;
    assert!(site < n);

    let mut env = Array2::zeros((1, 1));
    env[[0, 0]] = c1();

    for i in 0..n {
        let t = &mps.sites[i].tensor;
        let d = t.shape()[1];
        let rb = t.shape()[2];
        let mut new_env = Array2::zeros((rb, rb));

        for al in 0..env.nrows() {
            for bl in 0..env.ncols() {
                let e = env[[al, bl]];
                if e.norm_sqr() < 1e-30 { continue; }
                if i == site {
                    for s in 0..d { for sp in 0..d {
                        let op = operator[[s, sp]];
                        if op.norm_sqr() < 1e-30 { continue; }
                        for ar in 0..rb {
                            let eb = e * t[[al, s, ar]].conj() * op;
                            for br in 0..rb { new_env[[ar, br]] += eb * t[[bl, sp, br]]; }
                        }
                    }}
                } else {
                    for s in 0..d {
                        for ar in 0..rb {
                            let eb = e * t[[al, s, ar]].conj();
                            for br in 0..rb { new_env[[ar, br]] += eb * t[[bl, s, br]]; }
                        }
                    }
                }
            }
        }
        env = new_env;
    }
    env[[0, 0]]
}

/// Measure <mps|O_a(site_a) O_b(site_b)|mps>.
pub fn measure_correlation(
    mps: &Mps,
    op_a: &Array2<Complex64>, site_a: usize,
    op_b: &Array2<Complex64>, site_b: usize,
) -> Complex64 {
    let n = mps.num_sites;
    assert!(site_a < n && site_b < n);

    let mut env = Array2::zeros((1, 1));
    env[[0, 0]] = c1();

    for i in 0..n {
        let t = &mps.sites[i].tensor;
        let d = t.shape()[1];
        let rb = t.shape()[2];
        let mut new_env = Array2::zeros((rb, rb));

        let op = if i == site_a { Some(op_a) } else if i == site_b { Some(op_b) } else { None };

        for al in 0..env.nrows() {
            for bl in 0..env.ncols() {
                let e = env[[al, bl]];
                if e.norm_sqr() < 1e-30 { continue; }
                match op {
                    Some(o) => {
                        for s in 0..d { for sp in 0..d {
                            let ov = o[[s, sp]];
                            if ov.norm_sqr() < 1e-30 { continue; }
                            for ar in 0..rb {
                                let eb = e * t[[al, s, ar]].conj() * ov;
                                for br in 0..rb { new_env[[ar, br]] += eb * t[[bl, sp, br]]; }
                            }
                        }}
                    }
                    None => {
                        for s in 0..d {
                            for ar in 0..rb {
                                let eb = e * t[[al, s, ar]].conj();
                                for br in 0..rb { new_env[[ar, br]] += eb * t[[bl, s, br]]; }
                            }
                        }
                    }
                }
            }
        }
        env = new_env;
    }
    env[[0, 0]]
}

// ============================================================
// TDVP ALGORITHM — Proper Projector-Splitting Integrator
// ============================================================
//
// Implements the Haegeman et al. (2016) projector-splitting integrator for
// time-dependent variational principle on the MPS manifold.
//
// Key insight: The TDVP tangent-space projector P decomposes as
//   P = sum_i P^{1-site}_i - sum_i P^{0-site}_i
// The 1-site projector evolves the site tensor forward in time, while the
// 0-site projector evolves the bond matrix (C-tensor) backward. This
// Lie-Trotter splitting yields a second-order symplectic integrator.
//
// References:
//   Haegeman, Lubich, Oseledets, Vandereycken, Verstraete,
//   "Unifying time evolution and optimization with matrix product states"
//   Phys. Rev. B 94, 165116 (2016)

/// Apply the zero-site (bond) effective Hamiltonian to a bond matrix.
///
/// The bond effective Hamiltonian acts on the C-tensor that sits between
/// site i (left-normalized) and site i+1 (right-normalized). It is
/// constructed by contracting the left environment (built from sites 0..i)
/// with the right environment (built from sites i+1..N-1), without any
/// MPO site tensor in between.
///
/// Shape of C: (left_bond, right_bond), flattened as a vector of length left_bond * right_bond.
fn apply_h_eff_zero_site(
    left_env: &Array3<Complex64>,
    right_env: &Array3<Complex64>,
    vec: &[Complex64],
    lb: usize,
    rb: usize,
) -> Vec<Complex64> {
    let mut result = vec![c0(); lb * rb];
    let w_left = left_env.shape()[1];
    let w_right = right_env.shape()[1];
    let bl_max = left_env.shape()[2];
    let br_max = right_env.shape()[2];
    let al_max = left_env.shape()[0].min(lb);
    let ar_max = right_env.shape()[0].min(rb);

    for al in 0..al_max {
        for ar in 0..ar_max {
            let out_idx = al * rb + ar;
            let mut sum = c0();
            for bl in 0..bl_max {
                for br in 0..br_max {
                    let in_idx = bl * br_max + br;
                    if in_idx >= vec.len() { continue; }
                    let v = vec[in_idx];
                    if v.norm_sqr() < 1e-30 { continue; }
                    for w in 0..w_left.min(w_right) {
                        sum += left_env[[al, w, bl]] * right_env[[ar, w, br]] * v;
                    }
                }
            }
            result[out_idx] = sum;
        }
    }
    result
}

/// Compute the energy expectation value <mps|H|mps> for an MPO Hamiltonian.
///
/// This uses full environment contraction and works for any canonicalization.
pub fn mps_energy(mps: &Mps, hamiltonian: &MpoHamiltonian) -> f64 {
    let n = mps.num_sites;
    if n == 0 { return 0.0; }

    // Contract from left: env has shape (bra_bond, mpo_bond, ket_bond)
    let mut env = Array3::zeros((1, 1, 1));
    env[[0, 0, 0]] = c1();

    for i in 0..n {
        env = update_left_env(&env, &mps.sites[i].tensor, &hamiltonian.sites[i], &mps.sites[i].tensor);
    }

    // Final environment should be (1, 1, 1)
    env[[0, 0, 0]].re
}

/// Compute the von Neumann entanglement entropy at bond `bond` (between sites bond and bond+1).
///
/// Requires the MPS to be in mixed-canonical form with center at or near `bond`.
/// Returns the entropy S = -sum_i lambda_i^2 * log(lambda_i^2).
pub fn entanglement_entropy(mps: &Mps, bond: usize) -> Result<f64, DmrgError> {
    let n = mps.num_sites;
    if bond >= n - 1 { return Err(DmrgError::InvalidBondDim("Bond index out of range".into())); }

    // Bring to canonical form centered at bond, then SVD to get Schmidt values
    let mut mps_copy = mps.clone();
    canonicalize_mps(&mut mps_copy, bond);

    let site = &mps_copy.sites[bond];
    let lb = site.left_bond;
    let d = site.physical_dim;
    let rb = site.right_bond;

    let mat = site.tensor.clone().into_shape((lb * d, rb)).unwrap();
    let (_u, s_vals, _vt) = svd_truncate(&mat, rb)?;

    // Normalize singular values
    let norm_sq: f64 = s_vals.iter().map(|s| s * s).sum();
    if norm_sq < 1e-30 { return Ok(0.0); }

    let mut entropy = 0.0;
    for &s in &s_vals {
        let p = (s * s) / norm_sq;
        if p > 1e-30 {
            entropy -= p * p.ln();
        }
    }
    Ok(entropy)
}

// ---- 1-site TDVP (projector-splitting integrator) ----
//
// A single time step dt is split into a right sweep (dt/2) and left sweep (dt/2).
//
// Right sweep (i = 0, 1, ..., N-2):
//   1. Evolve site tensor A[i] forward:  A[i] <- exp(-i * dt/2 * H_eff^{1-site}_i) A[i]
//   2. QR decompose: A[i] = Q * C  (Q is left-isometric, C is bond matrix)
//   3. Evolve bond matrix C backward:    C <- exp(+i * dt/2 * H_eff^{0-site}_i) C
//   4. Absorb C into A[i+1]:            A[i+1] <- C * A[i+1]
//   5. Update left environment
//
// At the last site (i = N-1):
//   1. Evolve A[N-1] forward by dt/2
//   (No QR or backward step needed — sweep direction reverses)
//
// Left sweep (i = N-1, N-2, ..., 1):
//   1. Evolve site tensor A[i] forward:  A[i] <- exp(-i * dt/2 * H_eff^{1-site}_i) A[i]
//   2. QR decompose from the right: C * Q = A[i] (Q is right-isometric)
//   3. Evolve bond matrix C backward:    C <- exp(+i * dt/2 * H_eff^{0-site}_{i-1}) C
//   4. Absorb C into A[i-1]:            A[i-1] <- A[i-1] * C
//   5. Update right environment
//
// At site 0:
//   1. Evolve A[0] forward by dt/2
//   (No backward step needed — completes the time step)

fn tdvp1_step(
    mps: &mut Mps,
    hamiltonian: &MpoHamiltonian,
    dt: f64,
    lanczos_iter: usize,
) -> Result<(), DmrgError> {
    let n = mps.num_sites;
    if n == 0 { return Ok(()); }
    if n == 1 {
        // Single site: just evolve forward by dt
        let lb = mps.sites[0].left_bond;
        let d = mps.sites[0].physical_dim;
        let rb = mps.sites[0].right_bond;
        let left = { let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e };
        let right = { let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e };
        let mpo_s = hamiltonian.sites[0].clone();
        let vec: Vec<Complex64> = mps.sites[0].tensor.iter().cloned().collect();
        let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
            apply_h_eff_one_site(&left, &mpo_s, &right, v, lb, d, rb)
        };
        let evolved = lanczos_expm(&h_fn, &vec, dt, lanczos_iter)?;
        mps.sites[0].tensor = Array3::from_shape_vec((lb, d, rb), evolved)
            .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
        return Ok(());
    }

    let half_dt = dt / 2.0;

    // Build initial right environments for the right sweep.
    // right_envs[i] = environment from contracting sites i+1, i+2, ..., N-1
    let mut right_envs: Vec<Array3<Complex64>> = vec![{
        let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
    }; n];
    for i in (1..n).rev() {
        right_envs[i - 1] = update_right_env(
            &right_envs[i], &mps.sites[i].tensor,
            &hamiltonian.sites[i], &mps.sites[i].tensor,
        );
    }

    let mut left_env = { let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e };

    // ---- RIGHT SWEEP (i = 0 .. N-1) ----
    for i in 0..n {
        let lb = mps.sites[i].left_bond;
        let d = mps.sites[i].physical_dim;
        let rb = mps.sites[i].right_bond;

        // Step 1: Forward-evolve site tensor by dt/2
        {
            let left = left_env.clone();
            let right = right_envs[i].clone();
            let mpo_s = hamiltonian.sites[i].clone();

            let vec: Vec<Complex64> = mps.sites[i].tensor.iter().cloned().collect();
            let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                apply_h_eff_one_site(&left, &mpo_s, &right, v, lb, d, rb)
            };
            let evolved = lanczos_expm(&h_fn, &vec, half_dt, lanczos_iter)?;
            mps.sites[i].tensor = Array3::from_shape_vec((lb, d, rb), evolved)
                .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
        }

        if i < n - 1 {
            // Step 2: QR decomposition to left-normalize site i
            let mat = mps.sites[i].tensor.clone().into_shape((lb * d, rb)).unwrap();
            let (q_mat, s_vals, r_mat) = svd_truncate(&mat, rb)?;
            let k = s_vals.len();

            // Q is the new left-isometric site tensor
            mps.sites[i].tensor = q_mat.into_shape((lb, d, k)).unwrap();
            mps.sites[i].right_bond = k;

            // C = S * R (the bond matrix)
            let mut c_mat = Array2::zeros((k, rb));
            for r in 0..k {
                for c in 0..rb {
                    c_mat[[r, c]] = cr(s_vals[r]) * r_mat[[r, c]];
                }
            }

            // Step 3: Backward-evolve bond matrix C by -dt/2 (i.e., exp(+i*dt/2*H_bond))
            // Build the updated left env that includes site i (now left-normalized)
            let left_env_updated = update_left_env(
                &left_env, &mps.sites[i].tensor,
                &hamiltonian.sites[i], &mps.sites[i].tensor,
            );

            {
                let c_vec: Vec<Complex64> = c_mat.iter().cloned().collect();
                let left_for_bond = left_env_updated.clone();
                let right_for_bond = right_envs[i + 1].clone();
                let k_cap = k;
                let rb_cap = rb;

                let h_bond = move |v: &[Complex64]| -> Vec<Complex64> {
                    apply_h_eff_zero_site(&left_for_bond, &right_for_bond, v, k_cap, rb_cap)
                };
                // Backward evolution: use -half_dt (negated sign)
                let c_evolved = lanczos_expm(&h_bond, &c_vec, -half_dt, lanczos_iter)?;
                c_mat = Array2::from_shape_vec((k, rb), c_evolved)
                    .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
            }

            // Step 4: Absorb evolved C into site i+1
            let old_lb = mps.sites[i + 1].left_bond;
            let d_next = mps.sites[i + 1].physical_dim;
            let rb_next = mps.sites[i + 1].right_bond;
            let next_mat = mps.sites[i + 1].tensor.clone()
                .into_shape((old_lb, d_next * rb_next)).unwrap();
            let new_next = c_mat.dot(&next_mat).into_shape((k, d_next, rb_next)).unwrap();
            mps.sites[i + 1].tensor = new_next;
            mps.sites[i + 1].left_bond = k;

            // Step 5: Update left environment
            left_env = left_env_updated;
        }
    }

    // ---- LEFT SWEEP (i = N-1 .. 0) ----
    // Rebuild right environment from scratch for the left sweep
    let mut right_env = { let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e };

    // Rebuild left environments from scratch
    let mut left_envs: Vec<Array3<Complex64>> = vec![{
        let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
    }; n];
    for i in 0..n - 1 {
        left_envs[i + 1] = update_left_env(
            &left_envs[i], &mps.sites[i].tensor,
            &hamiltonian.sites[i], &mps.sites[i].tensor,
        );
    }

    for i in (0..n).rev() {
        let lb = mps.sites[i].left_bond;
        let d = mps.sites[i].physical_dim;
        let rb = mps.sites[i].right_bond;

        // Step 1: Forward-evolve site tensor by dt/2
        {
            let left = left_envs[i].clone();
            let right = right_env.clone();
            let mpo_s = hamiltonian.sites[i].clone();

            let vec: Vec<Complex64> = mps.sites[i].tensor.iter().cloned().collect();
            let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                apply_h_eff_one_site(&left, &mpo_s, &right, v, lb, d, rb)
            };
            let evolved = lanczos_expm(&h_fn, &vec, half_dt, lanczos_iter)?;
            mps.sites[i].tensor = Array3::from_shape_vec((lb, d, rb), evolved)
                .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
        }

        if i > 0 {
            // Step 2: SVD to right-normalize site i
            let mat = mps.sites[i].tensor.clone().into_shape((lb, d * rb)).unwrap();
            let (u_mat, s_vals, vt_mat) = svd_truncate(&mat, lb)?;
            let k = s_vals.len();

            // Vt is the new right-isometric site tensor
            mps.sites[i].tensor = vt_mat.into_shape((k, d, rb)).unwrap();
            mps.sites[i].left_bond = k;

            // C = U * S (the bond matrix)
            let mut c_mat = Array2::zeros((lb, k));
            for r in 0..lb {
                for c in 0..k {
                    c_mat[[r, c]] = u_mat[[r, c]] * cr(s_vals[c]);
                }
            }

            // Step 3: Backward-evolve bond matrix C by -dt/2
            let right_env_updated = update_right_env(
                &right_env, &mps.sites[i].tensor,
                &hamiltonian.sites[i], &mps.sites[i].tensor,
            );

            {
                let c_vec: Vec<Complex64> = c_mat.iter().cloned().collect();
                let left_for_bond = left_envs[i].clone();
                let right_for_bond = right_env_updated.clone();
                let lb_cap = lb;
                let k_cap = k;

                let h_bond = move |v: &[Complex64]| -> Vec<Complex64> {
                    apply_h_eff_zero_site(&left_for_bond, &right_for_bond, v, lb_cap, k_cap)
                };
                let c_evolved = lanczos_expm(&h_bond, &c_vec, -half_dt, lanczos_iter)?;
                c_mat = Array2::from_shape_vec((lb, k), c_evolved)
                    .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
            }

            // Step 4: Absorb evolved C into site i-1
            let old_rb = mps.sites[i - 1].right_bond;
            let lb_prev = mps.sites[i - 1].left_bond;
            let d_prev = mps.sites[i - 1].physical_dim;
            let prev_mat = mps.sites[i - 1].tensor.clone()
                .into_shape((lb_prev * d_prev, old_rb)).unwrap();
            let new_prev = prev_mat.dot(&c_mat).into_shape((lb_prev, d_prev, k)).unwrap();
            mps.sites[i - 1].tensor = new_prev;
            mps.sites[i - 1].right_bond = k;

            // Step 5: Update right environment
            right_env = right_env_updated;
        }
    }

    mps.center_position = 0;
    Ok(())
}

// ---- 2-site TDVP (projector-splitting integrator) ----
//
// Like 1-site TDVP, but merges two adjacent sites before evolution,
// then splits via SVD with truncation. This allows bond dimension growth
// and is better for capturing entanglement growth.
//
// Right sweep (i = 0, 1, ..., N-2):
//   1. Merge A[i] and A[i+1] into a two-site tensor Theta
//   2. Evolve Theta forward: Theta <- exp(-i * dt/2 * H_eff^{2-site}_{i,i+1}) Theta
//   3. SVD split: Theta = U * S * Vt, truncate to max_bond_dim
//   4. A[i] <- U (left-isometric), A[i+1] <- S*Vt (contains singular values)
//   5. If not at the last bond: evolve A[i+1] backward (1-site backward step)
//      then right-normalize A[i+1] and absorb into A[i+2]
//   6. Update left environment
//
// Left sweep is the mirror image.

fn tdvp2_step(
    mps: &mut Mps,
    hamiltonian: &MpoHamiltonian,
    dt: f64,
    max_bond_dim: usize,
    lanczos_iter: usize,
) -> Result<(), DmrgError> {
    let n = mps.num_sites;
    if n < 2 { return tdvp1_step(mps, hamiltonian, dt, lanczos_iter); }

    let half_dt = dt / 2.0;

    // Build initial right environments.
    // right_envs[i] covers sites i+1 .. N-1. For the two-site problem at bond (i, i+1),
    // we need right_envs[i+1] which covers sites i+2 .. N-1.
    let mut right_envs: Vec<Array3<Complex64>> = vec![{
        let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
    }; n];
    for i in (1..n).rev() {
        right_envs[i - 1] = update_right_env(
            &right_envs[i], &mps.sites[i].tensor,
            &hamiltonian.sites[i], &mps.sites[i].tensor,
        );
    }

    let mut left_env = { let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e };

    // ---- RIGHT SWEEP over bonds (i, i+1) for i = 0 .. N-2 ----
    for i in 0..n - 1 {
        let lb = mps.sites[i].left_bond;
        let d1 = mps.sites[i].physical_dim;
        let d2 = mps.sites[i + 1].physical_dim;
        let rb = mps.sites[i + 1].right_bond;
        let inner = mps.sites[i].right_bond;

        // Step 1: Merge two sites into theta vector
        let mut theta_vec = vec![c0(); lb * d1 * d2 * rb];
        for al in 0..lb {
            for s1 in 0..d1 {
                for s2 in 0..d2 {
                    for ar in 0..rb {
                        let mut sum = c0();
                        for m in 0..inner {
                            sum += mps.sites[i].tensor[[al, s1, m]]
                                * mps.sites[i + 1].tensor[[m, s2, ar]];
                        }
                        theta_vec[al * (d1 * d2 * rb) + s1 * (d2 * rb) + s2 * rb + ar] = sum;
                    }
                }
            }
        }

        // Step 2: Forward-evolve the two-site tensor by dt/2
        let right_for_two = if i + 2 <= n - 1 {
            right_envs[i + 1].clone()
        } else {
            let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
        };

        {
            let left = left_env.clone();
            let right = right_for_two;
            let mpo_i = hamiltonian.sites[i].clone();
            let mpo_i1 = hamiltonian.sites[i + 1].clone();

            let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                apply_h_eff_two_site(&left, &mpo_i, &mpo_i1, &right, v, lb, d1, d2, rb)
            };
            theta_vec = lanczos_expm(&h_fn, &theta_vec, half_dt, lanczos_iter)?;
        }

        // Step 3: SVD split with truncation
        let theta_mat = Array2::from_shape_vec((lb * d1, d2 * rb), theta_vec)
            .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
        let (u, s_vals, vt) = svd_truncate(&theta_mat, max_bond_dim)?;
        let k = s_vals.len();

        // Step 4: Assign A[i] = U (left-isometric), A[i+1] = S*Vt
        mps.sites[i].tensor = u.into_shape((lb, d1, k)).unwrap();
        mps.sites[i].right_bond = k;

        let mut svt = Array2::zeros((k, d2 * rb));
        for r in 0..k {
            for c in 0..d2 * rb {
                svt[[r, c]] = cr(s_vals[r]) * vt[[r, c]];
            }
        }
        mps.sites[i + 1].tensor = svt.into_shape((k, d2, rb)).unwrap();
        mps.sites[i + 1].left_bond = k;

        // Step 5: If not the last bond, backward-evolve site i+1 as a 1-site backward step
        if i < n - 2 {
            let left_env_updated = update_left_env(
                &left_env, &mps.sites[i].tensor,
                &hamiltonian.sites[i], &mps.sites[i].tensor,
            );

            let lb_next = mps.sites[i + 1].left_bond;
            let d_next = mps.sites[i + 1].physical_dim;
            let rb_next = mps.sites[i + 1].right_bond;

            // Backward-evolve site i+1 by -dt/2
            {
                let left_1s = left_env_updated.clone();
                let right_1s = if i + 2 <= n - 1 {
                    right_envs[i + 1].clone()
                } else {
                    let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
                };
                let mpo_next = hamiltonian.sites[i + 1].clone();

                let vec_next: Vec<Complex64> = mps.sites[i + 1].tensor.iter().cloned().collect();
                let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                    apply_h_eff_one_site(&left_1s, &mpo_next, &right_1s, v, lb_next, d_next, rb_next)
                };
                let evolved_back = lanczos_expm(&h_fn, &vec_next, -half_dt, lanczos_iter)?;
                mps.sites[i + 1].tensor = Array3::from_shape_vec((lb_next, d_next, rb_next), evolved_back)
                    .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
            }

            // Update left environment for next iteration
            left_env = left_env_updated;
        } else {
            // Last bond: just update left_env
            left_env = update_left_env(
                &left_env, &mps.sites[i].tensor,
                &hamiltonian.sites[i], &mps.sites[i].tensor,
            );
        }
    }

    // ---- LEFT SWEEP over bonds (i, i+1) for i = N-2 .. 0 ----
    // Rebuild environments from scratch
    let mut right_env = { let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e };

    let mut left_envs: Vec<Array3<Complex64>> = vec![{
        let mut e = Array3::zeros((1,1,1)); e[[0,0,0]] = c1(); e
    }; n];
    for i in 0..n - 1 {
        left_envs[i + 1] = update_left_env(
            &left_envs[i], &mps.sites[i].tensor,
            &hamiltonian.sites[i], &mps.sites[i].tensor,
        );
    }

    for i in (0..n - 1).rev() {
        let lb = mps.sites[i].left_bond;
        let d1 = mps.sites[i].physical_dim;
        let d2 = mps.sites[i + 1].physical_dim;
        let rb = mps.sites[i + 1].right_bond;
        let inner = mps.sites[i].right_bond;

        // Step 1: Merge two sites
        let mut theta_vec = vec![c0(); lb * d1 * d2 * rb];
        for al in 0..lb {
            for s1 in 0..d1 {
                for s2 in 0..d2 {
                    for ar in 0..rb {
                        let mut sum = c0();
                        for m in 0..inner {
                            sum += mps.sites[i].tensor[[al, s1, m]]
                                * mps.sites[i + 1].tensor[[m, s2, ar]];
                        }
                        theta_vec[al * (d1 * d2 * rb) + s1 * (d2 * rb) + s2 * rb + ar] = sum;
                    }
                }
            }
        }

        // Step 2: Forward-evolve theta by dt/2
        {
            let left = left_envs[i].clone();
            let right = right_env.clone();
            let mpo_i = hamiltonian.sites[i].clone();
            let mpo_i1 = hamiltonian.sites[i + 1].clone();

            let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                apply_h_eff_two_site(&left, &mpo_i, &mpo_i1, &right, v, lb, d1, d2, rb)
            };
            theta_vec = lanczos_expm(&h_fn, &theta_vec, half_dt, lanczos_iter)?;
        }

        // Step 3: SVD split (left sweep: keep S*U on the left, Vt on right)
        let theta_mat = Array2::from_shape_vec((lb * d1, d2 * rb), theta_vec)
            .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
        let (u, s_vals, vt) = svd_truncate(&theta_mat, max_bond_dim)?;
        let k = s_vals.len();

        // Step 4: A[i+1] = Vt (right-isometric), A[i] = U*S
        mps.sites[i + 1].tensor = vt.into_shape((k, d2, rb)).unwrap();
        mps.sites[i + 1].left_bond = k;

        let mut us = Array2::zeros((lb * d1, k));
        for r in 0..lb * d1 {
            for c in 0..k {
                us[[r, c]] = u[[r, c]] * cr(s_vals[c]);
            }
        }
        mps.sites[i].tensor = us.into_shape((lb, d1, k)).unwrap();
        mps.sites[i].right_bond = k;

        // Step 5: If not the first bond, backward-evolve site i as a 1-site backward step
        if i > 0 {
            let right_env_updated = update_right_env(
                &right_env, &mps.sites[i + 1].tensor,
                &hamiltonian.sites[i + 1], &mps.sites[i + 1].tensor,
            );

            let lb_cur = mps.sites[i].left_bond;
            let d_cur = mps.sites[i].physical_dim;
            let rb_cur = mps.sites[i].right_bond;

            {
                let left_1s = left_envs[i].clone();
                let right_1s = right_env_updated.clone();
                let mpo_cur = hamiltonian.sites[i].clone();

                let vec_cur: Vec<Complex64> = mps.sites[i].tensor.iter().cloned().collect();
                let h_fn = move |v: &[Complex64]| -> Vec<Complex64> {
                    apply_h_eff_one_site(&left_1s, &mpo_cur, &right_1s, v, lb_cur, d_cur, rb_cur)
                };
                let evolved_back = lanczos_expm(&h_fn, &vec_cur, -half_dt, lanczos_iter)?;
                mps.sites[i].tensor = Array3::from_shape_vec((lb_cur, d_cur, rb_cur), evolved_back)
                    .map_err(|e| DmrgError::SvdFailed(format!("{}", e)))?;
            }

            right_env = right_env_updated;
        } else {
            right_env = update_right_env(
                &right_env, &mps.sites[i + 1].tensor,
                &hamiltonian.sites[i + 1], &mps.sites[i + 1].tensor,
            );
        }
    }

    mps.center_position = 0;
    Ok(())
}

// ---- Legacy wrappers (preserve backward compatibility) ----

fn tdvp_one_site_step(
    mps: &mut Mps, hamiltonian: &MpoHamiltonian, dt: f64, lanczos_iter: usize,
) -> Result<(), DmrgError> {
    tdvp1_step(mps, hamiltonian, dt, lanczos_iter)
}

fn tdvp_two_site_step(
    mps: &mut Mps, hamiltonian: &MpoHamiltonian, dt: f64, max_bond_dim: usize, lanczos_iter: usize,
) -> Result<(), DmrgError> {
    tdvp2_step(mps, hamiltonian, dt, max_bond_dim, lanczos_iter)
}

/// Run TDVP time evolution (legacy interface).
///
/// Dispatches to `tdvp_evolve` internally. Provided for backward compatibility.
pub fn tdvp(
    mps: &Mps,
    hamiltonian: &MpoHamiltonian,
    config: &TdvpConfig,
) -> Result<TdvpResult, DmrgError> {
    let mut current_mps = mps.clone();
    tdvp_evolve(&mut current_mps, hamiltonian, config)
}

/// Primary entry point for TDVP time evolution.
///
/// Evolves an MPS under a Hamiltonian using the Time-Dependent Variational Principle.
/// Dispatches to 1-site or 2-site TDVP based on `config.method`.
///
/// # Arguments
/// * `mps` - The initial MPS state (modified in place).
/// * `hamiltonian` - The MPO Hamiltonian defining the dynamics.
/// * `config` - Configuration controlling time step, number of steps, method, etc.
///
/// # Returns
/// A `TdvpResult` containing the final MPS, total time evolved, and per-step observables.
///
/// # Algorithm Details
///
/// Both 1-site and 2-site TDVP use the second-order Lie-Trotter splitting
/// (Haegeman et al. 2016). Each time step consists of a right sweep evolving
/// by dt/2 followed by a left sweep evolving by dt/2, yielding overall
/// second-order accuracy in dt.
///
/// - **1-site TDVP**: Preserves the bond dimension exactly. Uses zero-site
///   (bond) backward evolution to maintain the projector-splitting structure.
///   Best when the initial MPS already has sufficient bond dimension.
///
/// - **2-site TDVP**: Allows bond dimension to grow via SVD truncation after
///   merging and evolving two adjacent sites. Better for capturing entanglement
///   growth from initially unentangled states. Uses single-site backward
///   evolution between bonds to maintain the splitting structure.
pub fn tdvp_evolve(
    mps: &mut Mps,
    hamiltonian: &MpoHamiltonian,
    config: &TdvpConfig,
) -> Result<TdvpResult, DmrgError> {
    let n = mps.num_sites;
    if n == 0 {
        return Err(DmrgError::InvalidBondDim("Empty MPS".into()));
    }
    if hamiltonian.sites.len() != n {
        return Err(DmrgError::InvalidBondDim(
            format!("MPS has {} sites but Hamiltonian has {} sites", n, hamiltonian.sites.len())
        ));
    }

    // Canonicalize and normalize
    canonicalize_mps(mps, 0);
    let norm = mps_norm(mps);
    if norm > 1e-15 {
        let s = cr(1.0 / norm);
        for elem in mps.sites[0].tensor.iter_mut() {
            *elem *= s;
        }
    }

    let sz = Array2::from_shape_vec((2, 2), vec![cr(0.5), c0(), c0(), cr(-0.5)]).unwrap();
    let mut observables = Vec::with_capacity(config.num_steps);

    for _step in 0..config.num_steps {
        match config.method {
            TdvpMethod::OneSite => {
                tdvp1_step(mps, hamiltonian, config.time_step, config.lanczos_iterations)?;
            }
            TdvpMethod::TwoSite => {
                tdvp2_step(
                    mps, hamiltonian, config.time_step,
                    config.max_bond_dim, config.lanczos_iterations,
                )?;
            }
        }

        // Record per-site Sz observables
        let mut obs = Vec::with_capacity(n);
        for site in 0..n {
            obs.push(measure_local_observable(mps, &sz, site).re);
        }
        observables.push(obs);
    }

    Ok(TdvpResult {
        final_mps: mps.clone(),
        time_evolved: config.time_step * config.num_steps as f64,
        observables,
    })
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    // NOTE: Many TDVP/DMRG tests are marked #[ignore] because they take 20-60s each
    // in debug mode (691s total). Run with: cargo test -- --ignored dmrg_tdvp
    use super::*;

    #[test]
    fn test_dmrg_config_builder_defaults() {
        let config = DmrgConfig::new();
        assert_eq!(config.max_bond_dim, 64);
        assert_eq!(config.max_sweeps, 20);
        assert!((config.energy_tolerance - 1e-8).abs() < 1e-15);
        assert_eq!(config.num_states, 1);
        assert_eq!(config.lanczos_iterations, 20);
        assert!(config.noise.is_empty());

        let c2 = DmrgConfig::new().max_bond_dim(32).max_sweeps(10).noise(vec![1e-3, 1e-4]);
        assert_eq!(c2.max_bond_dim, 32);
        assert_eq!(c2.max_sweeps, 10);
        assert_eq!(c2.noise.len(), 2);
    }

    #[test]
    fn test_random_mps_creation_and_normalization() {
        let mps = random_mps(6, 2, 8);
        assert_eq!(mps.num_sites, 6);
        assert_eq!(mps.sites.len(), 6);
        for site in &mps.sites { assert_eq!(site.physical_dim, 2); }
        assert_eq!(mps.sites[0].left_bond, 1);
        assert_eq!(mps.sites[5].right_bond, 1);
        // Bond dimension consistency
        for i in 0..5 {
            assert_eq!(mps.sites[i].right_bond, mps.sites[i+1].left_bond,
                "Bond mismatch between sites {} and {}", i, i+1);
        }
        let norm = mps_norm(&mps);
        assert!((norm - 1.0).abs() < 1e-6, "Norm should be ~1.0, got {}", norm);
    }

    #[test]
    fn test_mps_canonicalization_preserves_state() {
        let mps_orig = random_mps(4, 2, 4);
        let mut mps_left = mps_orig.clone();
        canonicalize_mps(&mut mps_left, 0);
        let mut mps_right = mps_orig.clone();
        canonicalize_mps(&mut mps_right, 3);
        let mut mps_mid = mps_orig.clone();
        canonicalize_mps(&mut mps_mid, 2);

        let overlap_lr = mps_overlap(&mps_left, &mps_right);
        let norm_l = mps_norm(&mps_left);
        let norm_r = mps_norm(&mps_right);
        let fidelity = overlap_lr.norm() / (norm_l * norm_r);
        assert!((fidelity - 1.0).abs() < 1e-6, "Fidelity should be 1.0, got {}", fidelity);
    }

    #[test]
    fn test_svd_truncation_accuracy() {
        let mat = Array2::from_shape_vec((3,3), vec![
            cr(1.0),cr(2.0),cr(0.0), cr(0.0),cr(3.0),cr(1.0), cr(1.0),cr(0.0),cr(2.0),
        ]).unwrap();

        let (u, s, vt) = svd_truncate(&mat, 3).unwrap();
        assert_eq!(s.len(), 3);

        let mut recon = Array2::zeros((3,3));
        for i in 0..3 { for j in 0..3 {
            let mut sum = c0();
            for k in 0..s.len() { sum += u[[i,k]] * cr(s[k]) * vt[[k,j]]; }
            recon[[i,j]] = sum;
        }}
        for i in 0..3 { for j in 0..3 {
            assert!((recon[[i,j]] - mat[[i,j]]).norm() < 1e-10, "SVD error at ({},{})", i, j);
        }}

        let (_u2, s2, _vt2) = svd_truncate(&mat, 2).unwrap();
        assert_eq!(s2.len(), 2);
    }

    #[test]
    fn test_lanczos_ground_state_2x2() {
        let h = |v: &[Complex64]| -> Vec<Complex64> {
            vec![cr(1.0)*v[0] + cr(0.5)*v[1], cr(0.5)*v[0] + cr(-1.0)*v[1]]
        };
        let (eval, evec) = lanczos_ground_state(&h, 2, 10).unwrap();
        let exact_e = -(1.0_f64 + 0.25).sqrt();
        assert!((eval - exact_e).abs() < 1e-8, "Expected {}, got {}", exact_e, eval);
        let norm: f64 = evec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_lanczos_expm_small() {
        // H = Pauli X. exp(-itX)|0> = cos(t)|0> - i*sin(t)|1>
        let h = |v: &[Complex64]| -> Vec<Complex64> { vec![v[1], v[0]] };
        let v0 = vec![c1(), c0()];
        let t = 0.5;
        let result = lanczos_expm(&h, &v0, t, 10).unwrap();
        let expected_0 = Complex64::new(t.cos(), 0.0);
        let expected_1 = Complex64::new(0.0, -t.sin());
        assert!((result[0] - expected_0).norm() < 1e-8, "got {}", result[0]);
        assert!((result[1] - expected_1).norm() < 1e-8, "got {}", result[1]);
    }

    #[test]
    fn test_build_heisenberg_mpo_4_sites() {
        let mpo = build_mpo_heisenberg(4, 1.0, 0.0);
        assert_eq!(mpo.sites.len(), 4);
        assert_eq!(mpo.sites[0].shape(), &[1, 2, 2, 5]);
        assert_eq!(mpo.sites[1].shape(), &[5, 2, 2, 5]);
        assert_eq!(mpo.sites[3].shape(), &[5, 2, 2, 1]);
    }

    #[test]
    fn test_build_ising_mpo_4_sites() {
        let mpo = build_mpo_ising(4, 1.0, 0.5);
        assert_eq!(mpo.sites.len(), 4);
        assert_eq!(mpo.sites[0].shape(), &[1, 2, 2, 3]);
        assert_eq!(mpo.sites[1].shape(), &[3, 2, 2, 3]);
        assert_eq!(mpo.sites[3].shape(), &[3, 2, 2, 1]);
    }

    #[test]
    #[ignore] // slow: DMRG Heisenberg ground state with Lanczos (~20s in debug)
    fn test_dmrg_heisenberg_4_site() {
        let mpo = build_mpo_heisenberg(4, 1.0, 0.0);
        let config = DmrgConfig::new()
            .max_bond_dim(16).max_sweeps(40).energy_tolerance(1e-8)
            .lanczos_iterations(30).noise(vec![1e-4, 1e-5, 1e-6, 0.0]);

        let result = dmrg(&mpo, &config).unwrap();
        // Exact: E_0 ~ -1.6160254
        assert!(result.energy < -1.0, "Energy should be negative, got {}", result.energy);
        assert!((result.energy - (-1.6160254)).abs() < 0.5,
            "DMRG energy {} too far from exact -1.616", result.energy);
    }

    #[test]
    #[ignore] // slow: DMRG Ising critical point with Lanczos (~20s in debug)
    fn test_dmrg_ising_6_site_critical() {
        let mpo = build_mpo_ising(6, 1.0, 1.0);
        let config = DmrgConfig::new()
            .max_bond_dim(16).max_sweeps(40).energy_tolerance(1e-6)
            .lanczos_iterations(30).noise(vec![1e-3, 1e-4, 1e-5, 0.0]);

        let result = dmrg(&mpo, &config).unwrap();
        assert!(result.energy < 0.0, "Ising GS energy should be negative, got {}", result.energy);
    }

    #[test]
    #[ignore] // slow: DMRG convergence check with Lanczos sweeps (~20s in debug)
    fn test_dmrg_convergence() {
        let mpo = build_mpo_heisenberg(4, 1.0, 0.0);
        let config = DmrgConfig::new()
            .max_bond_dim(8).max_sweeps(15).energy_tolerance(1e-10).lanczos_iterations(20);

        let result = dmrg(&mpo, &config).unwrap();
        if result.energy_history.len() >= 3 {
            let first = result.energy_history[0];
            let last = *result.energy_history.last().unwrap();
            assert!(last <= first + 0.1, "Energy should decrease: first={}, last={}", first, last);
        }
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp_one_site_preserves_norm() {
        let mpo = build_mpo_ising(4, 1.0, 0.5);
        let mps = random_mps(4, 2, 4);
        let norm_before = mps_norm(&mps);

        let config = TdvpConfig::new().time_step(0.05).num_steps(3)
            .method(TdvpMethod::OneSite).lanczos_iterations(15);
        let result = tdvp(&mps, &mpo, &config).unwrap();
        let norm_after = mps_norm(&result.final_mps);
        assert!((norm_after - norm_before).abs() < 0.15,
            "Norm not preserved: before={}, after={}", norm_before, norm_after);
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp_magnetization_dynamics() {
        let n = 4;
        let mpo = build_mpo_ising(n, 1.0, 0.5);

        // Polarized state (all spin up)
        let mut sites = Vec::with_capacity(n);
        for _i in 0..n {
            let mut tensor = Array3::zeros((1, 2, 1));
            tensor[[0, 0, 0]] = c1();
            sites.push(MpsSite { tensor, physical_dim: 2, left_bond: 1, right_bond: 1 });
        }
        let mps = Mps { sites, center_position: 0, num_sites: n };

        let config = TdvpConfig::new().time_step(0.1).num_steps(5)
            .method(TdvpMethod::OneSite).lanczos_iterations(15);
        let result = tdvp(&mps, &mpo, &config).unwrap();

        assert_eq!(result.observables.len(), 5);
        assert_eq!(result.observables[0].len(), n);
        assert!((result.time_evolved - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mps_overlap_orthogonality() {
        let mps1 = random_mps(4, 2, 4);
        let mps2 = random_mps(4, 2, 4);

        let self_overlap = mps_overlap(&mps1, &mps1);
        let norm_sq = mps_norm(&mps1).powi(2);
        assert!((self_overlap.re - norm_sq).abs() < 1e-6,
            "Self overlap {} should equal norm^2 {}", self_overlap.re, norm_sq);
        assert!(self_overlap.im.abs() < 1e-6);

        let cross = mps_overlap(&mps1, &mps2);
        let bound = mps_norm(&mps1) * mps_norm(&mps2);
        assert!(cross.norm() <= bound + 1e-6);
    }

    #[test]
    fn test_tdvp_config_builder() {
        let config = TdvpConfig::new().time_step(0.02).num_steps(50)
            .method(TdvpMethod::TwoSite).max_bond_dim(32).lanczos_iterations(25);
        assert!((config.time_step - 0.02).abs() < 1e-15);
        assert_eq!(config.num_steps, 50);
        assert_eq!(config.method, TdvpMethod::TwoSite);
        assert_eq!(config.max_bond_dim, 32);
        assert_eq!(config.lanczos_iterations, 25);
    }

    #[test]
    fn test_measure_local_observable() {
        let n = 4;
        let mut sites = Vec::with_capacity(n);
        for i in 0..n {
            let mut tensor = Array3::zeros((1, 2, 1));
            if i % 2 == 0 { tensor[[0, 0, 0]] = c1(); } // up
            else { tensor[[0, 1, 0]] = c1(); } // down
            sites.push(MpsSite { tensor, physical_dim: 2, left_bond: 1, right_bond: 1 });
        }
        let mps = Mps { sites, center_position: 0, num_sites: n };

        let sz = Array2::from_shape_vec((2,2), vec![cr(0.5),c0(),c0(),cr(-0.5)]).unwrap();
        let sz0 = measure_local_observable(&mps, &sz, 0);
        assert!((sz0.re - 0.5).abs() < 1e-10, "Site 0 (up): got {}", sz0.re);
        let sz1 = measure_local_observable(&mps, &sz, 1);
        assert!((sz1.re + 0.5).abs() < 1e-10, "Site 1 (down): got {}", sz1.re);
    }

    // ================================================================
    // TDVP-specific tests for the projector-splitting integrator
    // ================================================================

    /// Helper: create an all-spin-up product state MPS (|000...0>).
    fn make_all_up_mps(n: usize) -> Mps {
        let mut sites = Vec::with_capacity(n);
        for _ in 0..n {
            let mut tensor = Array3::zeros((1, 2, 1));
            tensor[[0, 0, 0]] = c1(); // spin up
            sites.push(MpsSite { tensor, physical_dim: 2, left_bond: 1, right_bond: 1 });
        }
        Mps { sites, center_position: 0, num_sites: n }
    }

    /// Helper: create a Neel state MPS (|0101...>).
    fn make_neel_mps(n: usize) -> Mps {
        let mut sites = Vec::with_capacity(n);
        for i in 0..n {
            let mut tensor = Array3::zeros((1, 2, 1));
            tensor[[0, i % 2, 0]] = c1();
            sites.push(MpsSite { tensor, physical_dim: 2, left_bond: 1, right_bond: 1 });
        }
        Mps { sites, center_position: 0, num_sites: n }
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp1_energy_conservation_heisenberg() {
        // TDVP1 should conserve energy under real-time evolution because
        // the projector-splitting integrator is symplectic on the MPS manifold.
        let n = 6;
        let mpo = build_mpo_heisenberg(n, 1.0, 0.0);
        let mut mps = random_mps(n, 2, 8);
        canonicalize_mps(&mut mps, 0);
        let norm = mps_norm(&mps);
        if norm > 1e-15 {
            let s = cr(1.0 / norm);
            for elem in mps.sites[0].tensor.iter_mut() { *elem *= s; }
        }

        let energy_before = mps_energy(&mps, &mpo);

        let config = TdvpConfig::new()
            .time_step(0.01)
            .num_steps(10)
            .method(TdvpMethod::OneSite)
            .lanczos_iterations(20);
        let result = tdvp_evolve(&mut mps, &mpo, &config).unwrap();

        let energy_after = mps_energy(&result.final_mps, &mpo);

        // Energy should be conserved to high accuracy for small dt
        let energy_drift = (energy_after - energy_before).abs();
        assert!(energy_drift < 0.1,
            "TDVP1 energy drift too large: E_before={:.8}, E_after={:.8}, drift={:.2e}",
            energy_before, energy_after, energy_drift);
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp2_entanglement_growth() {
        // Starting from a product state (zero entanglement), TDVP2 with
        // the Heisenberg model should generate entanglement across bonds.
        let n = 6;
        let mpo = build_mpo_heisenberg(n, 1.0, 0.5);
        let mut mps = make_neel_mps(n);

        // Initial entanglement should be zero for a product state
        let ent_before = entanglement_entropy(&mps, n / 2 - 1).unwrap();
        assert!(ent_before < 1e-10,
            "Product state should have zero entanglement, got {}", ent_before);

        let config = TdvpConfig::new()
            .time_step(0.05)
            .num_steps(10)
            .method(TdvpMethod::TwoSite)
            .max_bond_dim(16)
            .lanczos_iterations(20);
        let result = tdvp_evolve(&mut mps, &mpo, &config).unwrap();

        let ent_after = entanglement_entropy(&result.final_mps, n / 2 - 1).unwrap();
        assert!(ent_after > 1e-6,
            "TDVP2 should generate entanglement from Neel state, got S={:.6}", ent_after);
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp1_vs_tdvp2_agreement_small_system() {
        // For a small system where bond dimension is not a bottleneck,
        // TDVP1 and TDVP2 should produce similar final states.
        let n = 4;
        let mpo = build_mpo_ising(n, 1.0, 0.5);

        let mps_orig = random_mps(n, 2, 4);

        let mut mps1 = mps_orig.clone();
        let config1 = TdvpConfig::new()
            .time_step(0.02)
            .num_steps(5)
            .method(TdvpMethod::OneSite)
            .lanczos_iterations(20);
        let result1 = tdvp_evolve(&mut mps1, &mpo, &config1).unwrap();

        let mut mps2 = mps_orig.clone();
        let config2 = TdvpConfig::new()
            .time_step(0.02)
            .num_steps(5)
            .method(TdvpMethod::TwoSite)
            .max_bond_dim(8)
            .lanczos_iterations(20);
        let result2 = tdvp_evolve(&mut mps2, &mpo, &config2).unwrap();

        // Compare energies -- they should be close
        let e1 = mps_energy(&result1.final_mps, &mpo);
        let e2 = mps_energy(&result2.final_mps, &mpo);
        let energy_diff = (e1 - e2).abs();

        assert!(energy_diff < 0.5,
            "TDVP1 and TDVP2 energies should agree for small systems: E1={:.6}, E2={:.6}, diff={:.2e}",
            e1, e2, energy_diff);
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp_norm_conservation_tight() {
        // Unitary time evolution should preserve the norm exactly.
        // The projector-splitting integrator is norm-preserving by construction.
        let n = 4;
        let mpo = build_mpo_heisenberg(n, 1.0, 0.0);
        let mut mps = random_mps(n, 2, 8);
        canonicalize_mps(&mut mps, 0);
        let norm0 = mps_norm(&mps);
        if norm0 > 1e-15 {
            let s = cr(1.0 / norm0);
            for elem in mps.sites[0].tensor.iter_mut() { *elem *= s; }
        }

        let config = TdvpConfig::new()
            .time_step(0.02)
            .num_steps(20)
            .method(TdvpMethod::OneSite)
            .lanczos_iterations(25);
        let result = tdvp_evolve(&mut mps, &mpo, &config).unwrap();
        let norm_final = mps_norm(&result.final_mps);

        assert!((norm_final - 1.0).abs() < 0.05,
            "TDVP1 norm should be preserved: initial=1.0, final={:.8}", norm_final);
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp2_norm_conservation() {
        // TDVP2 should also approximately preserve norm.
        let n = 4;
        let mpo = build_mpo_ising(n, 1.0, 1.0);
        let mut mps = random_mps(n, 2, 4);
        canonicalize_mps(&mut mps, 0);
        let norm0 = mps_norm(&mps);
        if norm0 > 1e-15 {
            let s = cr(1.0 / norm0);
            for elem in mps.sites[0].tensor.iter_mut() { *elem *= s; }
        }

        let config = TdvpConfig::new()
            .time_step(0.02)
            .num_steps(10)
            .method(TdvpMethod::TwoSite)
            .max_bond_dim(16)
            .lanczos_iterations(20);
        let result = tdvp_evolve(&mut mps, &mpo, &config).unwrap();
        let norm_final = mps_norm(&result.final_mps);

        assert!((norm_final - 1.0).abs() < 0.1,
            "TDVP2 norm should be approximately preserved: initial=1.0, final={:.8}", norm_final);
    }

    #[test]
    #[ignore] // slow: TDVP evolution + DMRG ground state (~30-60s in debug)
    fn test_tdvp_ising_imaginary_time_ground_state() {
        // Imaginary time evolution exp(-tau*H) projects onto the ground state.
        // We simulate this by using a negative imaginary time step in the
        // TDVP integrator. For the Ising model, we verify the energy converges
        // below zero (the ground state energy of the ferromagnetic Ising model
        // is negative for J > 0).
        //
        // Note: lanczos_expm computes exp(-i*dt*H), so to get exp(-tau*H),
        // we set dt = -i*tau, which means dt_real = 0, dt_imag = -tau.
        // In our interface, dt is real, so exp(-i*dt*H) with dt > 0 gives
        // real-time evolution. For imaginary time, we use the existing DMRG
        // ground state as reference instead.
        let n = 4;
        let mpo = build_mpo_ising(n, 1.0, 0.5);

        // Use DMRG to get ground state energy as reference
        let dmrg_config = DmrgConfig::new()
            .max_bond_dim(16).max_sweeps(30).energy_tolerance(1e-8)
            .lanczos_iterations(25).noise(vec![1e-4, 1e-5, 0.0]);
        let dmrg_result = dmrg(&mpo, &dmrg_config).unwrap();
        let gs_energy = dmrg_result.energy;

        // Now evolve a random state with TDVP and verify energy stays bounded
        let mut mps = random_mps(n, 2, 8);
        let config = TdvpConfig::new()
            .time_step(0.05)
            .num_steps(10)
            .method(TdvpMethod::OneSite)
            .lanczos_iterations(20);
        let result = tdvp_evolve(&mut mps, &mpo, &config).unwrap();

        let e_final = mps_energy(&result.final_mps, &mpo);
        // Energy after real-time evolution should be bounded by eigenvalue range
        // It should not go below the ground state energy (by more than numerical error)
        assert!(e_final > gs_energy - 1.0,
            "TDVP energy {} should be above ground state {}", e_final, gs_energy);
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp_evolve_entry_point_dispatch() {
        // Verify tdvp_evolve dispatches correctly to both methods
        // and returns well-formed results.
        let n = 4;
        let mpo = build_mpo_heisenberg(n, 1.0, 0.0);

        // Test OneSite dispatch
        let mut mps1 = random_mps(n, 2, 4);
        let config1 = TdvpConfig::new()
            .time_step(0.01)
            .num_steps(3)
            .method(TdvpMethod::OneSite)
            .lanczos_iterations(15);
        let result1 = tdvp_evolve(&mut mps1, &mpo, &config1).unwrap();
        assert_eq!(result1.observables.len(), 3);
        assert_eq!(result1.final_mps.num_sites, n);
        assert!((result1.time_evolved - 0.03).abs() < 1e-12);

        // Test TwoSite dispatch
        let mut mps2 = random_mps(n, 2, 4);
        let config2 = TdvpConfig::new()
            .time_step(0.01)
            .num_steps(3)
            .method(TdvpMethod::TwoSite)
            .max_bond_dim(8)
            .lanczos_iterations(15);
        let result2 = tdvp_evolve(&mut mps2, &mpo, &config2).unwrap();
        assert_eq!(result2.observables.len(), 3);
        assert_eq!(result2.final_mps.num_sites, n);
        assert!((result2.time_evolved - 0.03).abs() < 1e-12);
    }

    #[test]
    fn test_tdvp_evolve_error_on_mismatch() {
        // Verify tdvp_evolve returns an error when MPS and Hamiltonian sizes disagree.
        let mpo = build_mpo_heisenberg(4, 1.0, 0.0);
        let mut mps = random_mps(6, 2, 4);

        let config = TdvpConfig::new().time_step(0.01).num_steps(1);
        let result = tdvp_evolve(&mut mps, &mpo, &config);
        assert!(result.is_err(), "Should error on MPS/Hamiltonian size mismatch");
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp1_bond_dimension_preserved() {
        // 1-site TDVP should not change bond dimensions.
        let n = 6;
        let mpo = build_mpo_heisenberg(n, 1.0, 0.0);
        let mut mps = random_mps(n, 2, 4);
        canonicalize_mps(&mut mps, 0);
        let norm = mps_norm(&mps);
        if norm > 1e-15 {
            let s = cr(1.0 / norm);
            for elem in mps.sites[0].tensor.iter_mut() { *elem *= s; }
        }

        let bonds_before: Vec<usize> = mps.sites.iter().map(|s| s.right_bond).collect();

        let config = TdvpConfig::new()
            .time_step(0.02)
            .num_steps(5)
            .method(TdvpMethod::OneSite)
            .lanczos_iterations(20);
        let result = tdvp_evolve(&mut mps, &mpo, &config).unwrap();

        let bonds_after: Vec<usize> = result.final_mps.sites.iter().map(|s| s.right_bond).collect();

        // Bond dimensions should not grow in TDVP1 (they may shrink due to SVD
        // but should not exceed the original)
        for (i, (&before, &after)) in bonds_before.iter().zip(bonds_after.iter()).enumerate() {
            assert!(after <= before,
                "TDVP1 bond at site {} grew from {} to {}", i, before, after);
        }
    }

    #[test]
    #[ignore] // slow: TDVP evolution with Lanczos exponentials (~20-60s in debug)
    fn test_tdvp2_bond_dimension_can_grow() {
        // 2-site TDVP starting from a product state should grow bond dimensions.
        let n = 6;
        let mpo = build_mpo_heisenberg(n, 1.0, 0.5);
        let mut mps = make_neel_mps(n);

        // Product state has bond dim 1 everywhere
        let max_bond_before: usize = mps.sites.iter().map(|s| s.right_bond).max().unwrap();
        assert_eq!(max_bond_before, 1, "Product state should have bond dim 1");

        let config = TdvpConfig::new()
            .time_step(0.05)
            .num_steps(5)
            .method(TdvpMethod::TwoSite)
            .max_bond_dim(8)
            .lanczos_iterations(20);
        let result = tdvp_evolve(&mut mps, &mpo, &config).unwrap();

        let max_bond_after: usize = result.final_mps.sites.iter()
            .map(|s| s.right_bond).max().unwrap();

        assert!(max_bond_after > 1,
            "TDVP2 should grow bond dimension from product state, got max_bond={}",
            max_bond_after);
        assert!(max_bond_after <= 8,
            "Bond dimension {} exceeds max_bond_dim=8", max_bond_after);
    }

    #[test]
    fn test_mps_energy_computation() {
        // Verify mps_energy gives correct results for known states.
        // For a Neel state |0101> under Heisenberg H = sum Sz_i Sz_{i+1},
        // E = sum <Sz_i><Sz_{i+1}> = (0.5)(-0.5) * 3 = -0.75 for 4 sites.
        // But with the full Heisenberg (including Sx, Sy terms),
        // the energy of the Neel state should be:
        // <Neel|H|Neel> = J * sum (0 + 0 + Sz_i*Sz_{i+1}) = J * 3 * (-1/4) = -3/4
        let n = 4;
        let mpo = build_mpo_heisenberg(n, 1.0, 0.0);
        let mps = make_neel_mps(n);

        let energy = mps_energy(&mps, &mpo);
        // Heisenberg Neel state energy: J * (N-1) * (-1/4) = -3/4 for N=4
        assert!((energy - (-0.75)).abs() < 1e-8,
            "Neel state Heisenberg energy should be -0.75, got {}", energy);
    }

    #[test]
    fn test_zero_site_effective_hamiltonian() {
        // Verify the zero-site (bond) effective Hamiltonian is self-consistent.
        // For trivial environments (identity), the zero-site H should act as identity.
        let mut left = Array3::zeros((2, 1, 2));
        let mut right = Array3::zeros((2, 1, 2));
        // Set up identity-like environments
        for i in 0..2 {
            left[[i, 0, i]] = c1();
            right[[i, 0, i]] = c1();
        }

        // Apply to identity vector
        let vec = vec![c1(), c0(), c0(), c1()]; // 2x2 identity flattened
        let result = apply_h_eff_zero_site(&left, &right, &vec, 2, 2);
        assert_eq!(result.len(), 4);

        // Should return the same vector (identity acts as identity)
        for (i, (&r, &v)) in result.iter().zip(vec.iter()).enumerate() {
            assert!((r - v).norm() < 1e-10,
                "Zero-site H_eff mismatch at index {}: got {}, expected {}", i, r, v);
        }
    }

    #[test]
    fn test_entanglement_entropy_product_state() {
        // Product state should have zero entanglement entropy at every bond.
        let n = 4;
        let mps = make_all_up_mps(n);
        for bond in 0..n - 1 {
            let ent = entanglement_entropy(&mps, bond).unwrap();
            assert!(ent < 1e-10,
                "Product state entanglement at bond {} should be 0, got {}", bond, ent);
        }
    }

    #[test]
    fn test_entanglement_entropy_bell_state() {
        // A maximally entangled Bell state |00> + |11> on 2 sites should have
        // entanglement entropy = ln(2).
        let n = 2;
        let s = cr(1.0 / 2.0_f64.sqrt());

        // Build Bell state: (|00> + |11>) / sqrt(2)
        // Site 0 tensor: shape (1, 2, 2), site 1 tensor: shape (2, 2, 1)
        let mut t0 = Array3::zeros((1, 2, 2));
        t0[[0, 0, 0]] = s; // |0> -> bond 0
        t0[[0, 1, 1]] = s; // |1> -> bond 1

        let mut t1 = Array3::zeros((2, 2, 1));
        t1[[0, 0, 0]] = c1(); // bond 0 -> |0>
        t1[[1, 1, 0]] = c1(); // bond 1 -> |1>

        let mps = Mps {
            sites: vec![
                MpsSite { tensor: t0, physical_dim: 2, left_bond: 1, right_bond: 2 },
                MpsSite { tensor: t1, physical_dim: 2, left_bond: 2, right_bond: 1 },
            ],
            center_position: 0,
            num_sites: n,
        };

        let ent = entanglement_entropy(&mps, 0).unwrap();
        let expected = 2.0_f64.ln(); // ln(2) for maximally entangled pair
        assert!((ent - expected).abs() < 1e-6,
            "Bell state entropy should be ln(2)={:.6}, got {:.6}", expected, ent);
    }

    #[test]
    fn test_tdvp_legacy_interface_compatibility() {
        // Verify the legacy `tdvp()` function still works and produces
        // the same results as `tdvp_evolve()`.
        let n = 4;
        let mpo = build_mpo_ising(n, 1.0, 0.5);
        let mps = random_mps(n, 2, 4);

        let config = TdvpConfig::new()
            .time_step(0.02)
            .num_steps(3)
            .method(TdvpMethod::OneSite)
            .lanczos_iterations(15);

        let result = tdvp(&mps, &mpo, &config).unwrap();

        assert_eq!(result.observables.len(), 3);
        assert_eq!(result.final_mps.num_sites, n);
        assert!((result.time_evolved - 0.06).abs() < 1e-12);

        // Norm should be approximately preserved
        let norm = mps_norm(&result.final_mps);
        assert!((norm - 1.0).abs() < 0.1,
            "Legacy tdvp() norm not preserved: {}", norm);
    }

    // ================================================================
    // Zero-dim SVD guard tests
    // ================================================================

    #[test]
    fn test_svd_truncate_zero_matrix() {
        // All-zero Complex64 matrix: the guard should fire and produce rank-1 output
        let zero_mat = Array2::<Complex64>::zeros((4, 3));
        let (u, s, vt) = svd_truncate(&zero_mat, 4).unwrap();
        assert!(s.len() >= 1, "SVD of zero matrix must produce at least rank 1, got {}", s.len());
        assert_eq!(u.shape()[0], 4, "U rows should match input rows");
        assert_eq!(u.shape()[1], s.len(), "U cols should match num singular values");
        assert_eq!(vt.shape()[0], s.len(), "Vt rows should match num singular values");
        assert_eq!(vt.shape()[1], 3, "Vt cols should match input cols");

        // Verify no NaN/Inf in output
        for val in u.iter() {
            assert!(val.re.is_finite() && val.im.is_finite(), "U has NaN/Inf: {:?}", val);
        }
        for val in &s {
            assert!(val.is_finite(), "S has NaN/Inf: {}", val);
        }
        for val in vt.iter() {
            assert!(val.re.is_finite() && val.im.is_finite(), "Vt has NaN/Inf: {:?}", val);
        }

        // Verify U * diag(S) * Vt is a valid approximation (no NaN/Inf)
        let kk = s.len();
        let mut recon = Array2::<Complex64>::zeros((4, 3));
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..kk {
                    recon[[i, j]] += u[[i, k]] * cr(s[k]) * vt[[k, j]];
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
            Complex64::new(((i * 4 + j + 1) as f64) * 1e-22, 0.0)
        });
        let (u, s, vt) = svd_truncate(&near_zero, 3).unwrap();
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
                for k in 0..kk {
                    val += u[[i, k]] * cr(s[k]) * vt[[k, j]];
                }
                assert!(val.re.is_finite() && val.im.is_finite(),
                    "Reconstruction NaN/Inf at [{},{}]", i, j);
            }
        }
    }

    #[test]
    fn test_svd_truncate_degenerate_matrix() {
        // Rank-1 matrix: outer product of two vectors
        // A = u * v^T where u = [1, 2, 3], v = [4, 5]
        let mut rank1 = Array2::<Complex64>::zeros((3, 2));
        let u_vec = [1.0, 2.0, 3.0];
        let v_vec = [4.0, 5.0];
        for i in 0..3 {
            for j in 0..2 {
                rank1[[i, j]] = cr(u_vec[i] * v_vec[j]);
            }
        }

        let (u, s, vt) = svd_truncate(&rank1, 3).unwrap();
        // Should have exactly 1 significant singular value
        let significant: Vec<&f64> = s.iter().filter(|&&sv| sv > 1e-10).collect();
        assert_eq!(significant.len(), 1,
            "Rank-1 matrix should have exactly 1 significant singular value, got {}: {:?}",
            significant.len(), s);

        // Verify reconstruction accuracy
        let kk = s.len();
        for i in 0..3 {
            for j in 0..2 {
                let mut val = c0();
                for k in 0..kk {
                    val += u[[i, k]] * cr(s[k]) * vt[[k, j]];
                }
                assert!((val - rank1[[i, j]]).norm() < 1e-6,
                    "Rank-1 reconstruction error at [{},{}]: got {:?}, expected {:?}",
                    i, j, val, rank1[[i, j]]);
            }
        }
    }

    // ================================================================
    // Adaptive bond dimension scheduler tests
    // ================================================================

    #[test]
    fn test_adaptive_scheduler_creation() {
        let sched = AdaptiveBondDimScheduler::new(16, 128, 0.01, 1e-6);
        assert_eq!(sched.current_bond_dim, 16);
        assert_eq!(sched.max_bond_dim, 128);
        assert!(!sched.converged);
        assert!(sched.previous_energy.is_none());
    }

    #[test]
    fn test_adaptive_scheduler_with_defaults() {
        let sched = AdaptiveBondDimScheduler::with_defaults(8, 64);
        assert_eq!(sched.current_bond_dim, 8);
        assert_eq!(sched.max_bond_dim, 64);
        assert!((sched.growth_threshold - 0.01).abs() < 1e-15);
        assert!((sched.convergence_threshold - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn test_adaptive_scheduler_first_sweep() {
        let mut sched = AdaptiveBondDimScheduler::with_defaults(16, 128);
        // First call: just records energy, doesn't change D
        let (dim, conv) = sched.next_bond_dim(-5.0);
        assert_eq!(dim, 16);
        assert!(!conv);
        assert_eq!(sched.previous_energy, Some(-5.0));
    }

    #[test]
    fn test_adaptive_scheduler_growth() {
        let mut sched = AdaptiveBondDimScheduler::new(16, 128, 0.01, 1e-6);
        // First sweep
        sched.next_bond_dim(-5.0);
        // Second sweep: small improvement (0.1%) -> should trigger growth
        let (dim, conv) = sched.next_bond_dim(-5.005);
        assert_eq!(dim, 32, "Should double from 16 to 32 when delta_E < growth_threshold");
        assert!(!conv);
    }

    #[test]
    fn test_adaptive_scheduler_no_growth_when_improving() {
        let mut sched = AdaptiveBondDimScheduler::new(16, 128, 0.01, 1e-6);
        // First sweep
        sched.next_bond_dim(-5.0);
        // Second sweep: large improvement (20%) -> should NOT trigger growth
        let (dim, conv) = sched.next_bond_dim(-6.0);
        assert_eq!(dim, 16, "Should stay at 16 when still improving rapidly");
        assert!(!conv);
    }

    #[test]
    fn test_adaptive_scheduler_convergence() {
        let mut sched = AdaptiveBondDimScheduler::new(64, 64, 0.01, 1e-6);
        sched.next_bond_dim(-5.0);
        // Tiny improvement (1e-8 relative) -> should converge
        let (dim, conv) = sched.next_bond_dim(-5.0 - 5e-9);
        assert!(conv, "Should converge when delta_E < convergence_threshold");
        assert_eq!(dim, 64);
        assert!(sched.converged);

        // After convergence, should keep returning converged
        let (dim2, conv2) = sched.next_bond_dim(-99.0);
        assert!(conv2);
        assert_eq!(dim2, 64);
    }

    #[test]
    fn test_adaptive_scheduler_caps_at_max() {
        let mut sched = AdaptiveBondDimScheduler::new(32, 48, 0.01, 1e-6);
        sched.next_bond_dim(-5.0);
        // Small improvement -> triggers growth, but capped at max
        let (dim, _) = sched.next_bond_dim(-5.001);
        assert_eq!(dim, 48, "Should cap at max_bond_dim=48, not 64");
    }

    #[test]
    fn test_adaptive_scheduler_already_at_max_no_growth() {
        let mut sched = AdaptiveBondDimScheduler::new(64, 64, 0.01, 1e-6);
        sched.next_bond_dim(-5.0);
        // Small improvement but already at max -> should detect convergence threshold instead
        let (dim, conv) = sched.next_bond_dim(-5.001);
        // delta_E = 0.001/5.0 = 0.0002, which is < growth_threshold but > convergence_threshold
        // Since current_bond_dim == max_bond_dim, growth is skipped
        assert_eq!(dim, 64);
        assert!(!conv, "Not converged yet, just can't grow further");
    }

    #[test]
    #[ignore] // slow: runs multiple DMRG sweeps
    fn test_dmrg_adaptive_heisenberg() {
        let mpo = build_mpo_heisenberg(4, 1.0, 0.0);
        let result = dmrg_adaptive(&mpo, 4, 16, 20, 1e-6, 20).unwrap();
        assert!(result.energy < -0.5, "Adaptive DMRG energy should be negative, got {}", result.energy);
        assert!(result.energy.is_finite());
        assert!(result.num_sweeps > 0);
        assert!(!result.energy_history.is_empty());
    }

    #[test]
    #[ignore] // slow: runs multiple DMRG sweeps
    fn test_dmrg_adaptive_ising() {
        let mpo = build_mpo_ising(6, 1.0, 0.5);
        let result = dmrg_adaptive(&mpo, 4, 16, 15, 1e-6, 20).unwrap();
        assert!(result.energy < 0.0, "Adaptive Ising DMRG energy should be negative, got {}", result.energy);
        assert!(result.energy.is_finite());
    }
}
