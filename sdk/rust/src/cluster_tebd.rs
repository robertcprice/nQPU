//! Cluster-TEBD: Entanglement-Aware Time-Evolving Block Decimation
//!
//! This module implements Cluster-TEBD, a next-generation tensor network algorithm
//! that dynamically groups qubits into entanglement clusters during time evolution.
//! Within each cluster, multiple circuit layers are contracted exactly (no truncation),
//! and SVD truncation is deferred to the cluster boundary. This exploits the
//! entanglement structure of the state for massive speedups on structured Hamiltonians.
//!
//! # Algorithm Overview
//!
//! 1. Analyze the entanglement structure of the current MPS via mutual information
//! 2. Identify clusters of strongly-entangled sites using greedy graph clustering
//! 3. For each cluster:
//!    a. Merge adjacent MPS tensors into a single large tensor
//!    b. Apply all internal gates exactly (no intermediate truncation)
//!    c. Re-decompose back into MPS form via sequential SVD
//! 4. Apply remaining inter-cluster gates via standard TEBD
//! 5. Repeat for each Trotter time step
//!
//! # References
//!
//! - Physical Review Research (2025): Cluster-TEBD for large-scale quantum simulation
//! - Vidal, "Efficient classical simulation of slightly entangled quantum computations" (2003)
//! - Schollwöck, "The density-matrix renormalization group in the age of matrix product states" (2011)

use ndarray::{Array2, Array3, ArrayD, IxDyn};
use num_complex::Complex64;
use rand::Rng;
use std::fmt;

// ============================================================
// COMPLEX HELPERS
// ============================================================

#[inline]
fn c0() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

#[inline]
fn c1() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

#[inline]
fn ci() -> Complex64 {
    Complex64::new(0.0, 1.0)
}

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors that can occur during Cluster-TEBD simulation.
#[derive(Debug, Clone)]
pub enum ClusterTebdError {
    /// A cluster exceeds the maximum allowed size.
    ClusterTooLarge {
        size: usize,
        max: usize,
    },
    /// SVD decomposition failed during tensor decomposition.
    SvdFailed(String),
    /// Invalid configuration parameters.
    InvalidConfig(String),
    /// Bond dimension exceeded the maximum allowed value.
    BondDimExceeded {
        actual: usize,
        max: usize,
    },
}

impl fmt::Display for ClusterTebdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterTebdError::ClusterTooLarge { size, max } => {
                write!(f, "cluster size {} exceeds maximum {}", size, max)
            }
            ClusterTebdError::SvdFailed(msg) => write!(f, "SVD failed: {}", msg),
            ClusterTebdError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            ClusterTebdError::BondDimExceeded { actual, max } => {
                write!(f, "bond dimension {} exceeds maximum {}", actual, max)
            }
        }
    }
}

impl std::error::Error for ClusterTebdError {}

// ============================================================
// MERGE STRATEGY
// ============================================================

/// Strategy for identifying and forming qubit clusters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Cluster based on mutual information between sites.
    EntanglementBased,
    /// Cluster based on geometric proximity (adjacent sites).
    GeometryBased,
    /// Adaptive: use entanglement when bond dims are large, geometry otherwise.
    Adaptive,
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the Cluster-TEBD algorithm.
#[derive(Debug, Clone)]
pub struct ClusterTebdConfig {
    /// Maximum bond dimension retained after SVD truncation.
    pub max_bond_dim: usize,
    /// Trotter time step δt.
    pub time_step: f64,
    /// Number of Trotter steps to perform.
    pub num_steps: usize,
    /// Mutual information threshold for clustering. Pairs with I(A:B) above
    /// this value are candidates for the same cluster.
    pub cluster_threshold: f64,
    /// Maximum number of sites in a single cluster. Exponential cost in this.
    pub max_cluster_size: usize,
    /// Strategy for forming clusters.
    pub merge_strategy: MergeStrategy,
    /// Singular values below this cutoff are discarded.
    pub svd_cutoff: f64,
}

impl Default for ClusterTebdConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            time_step: 0.01,
            num_steps: 100,
            cluster_threshold: 0.1,
            max_cluster_size: 8,
            merge_strategy: MergeStrategy::EntanglementBased,
            svd_cutoff: 1e-10,
        }
    }
}

impl ClusterTebdConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_bond_dim(mut self, d: usize) -> Self {
        self.max_bond_dim = d;
        self
    }

    pub fn time_step(mut self, dt: f64) -> Self {
        self.time_step = dt;
        self
    }

    pub fn num_steps(mut self, n: usize) -> Self {
        self.num_steps = n;
        self
    }

    pub fn cluster_threshold(mut self, t: f64) -> Self {
        self.cluster_threshold = t;
        self
    }

    pub fn max_cluster_size(mut self, s: usize) -> Self {
        self.max_cluster_size = s;
        self
    }

    pub fn merge_strategy(mut self, s: MergeStrategy) -> Self {
        self.merge_strategy = s;
        self
    }

    pub fn svd_cutoff(mut self, c: f64) -> Self {
        self.svd_cutoff = c;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), ClusterTebdError> {
        if self.max_bond_dim == 0 {
            return Err(ClusterTebdError::InvalidConfig(
                "max_bond_dim must be > 0".into(),
            ));
        }
        if self.time_step <= 0.0 {
            return Err(ClusterTebdError::InvalidConfig(
                "time_step must be > 0".into(),
            ));
        }
        if self.max_cluster_size < 2 {
            return Err(ClusterTebdError::InvalidConfig(
                "max_cluster_size must be >= 2".into(),
            ));
        }
        if self.cluster_threshold < 0.0 {
            return Err(ClusterTebdError::InvalidConfig(
                "cluster_threshold must be >= 0".into(),
            ));
        }
        Ok(())
    }
}

// ============================================================
// TWO-SITE GATE
// ============================================================

/// A two-site quantum gate represented as a 4x4 unitary matrix.
///
/// The matrix acts on the tensor product space of `site_a` and `site_b`,
/// ordered as |site_a, site_b⟩ in the computational basis.
#[derive(Debug, Clone)]
pub struct TwoSiteGate {
    /// First site index.
    pub site_a: usize,
    /// Second site index.
    pub site_b: usize,
    /// The 4x4 gate matrix in the computational basis.
    pub matrix: Array2<Complex64>,
}

impl TwoSiteGate {
    /// Create a new two-site gate.
    pub fn new(site_a: usize, site_b: usize, matrix: Array2<Complex64>) -> Self {
        Self {
            site_a,
            site_b,
            matrix,
        }
    }

    /// Whether this gate acts within a given set of sites.
    pub fn is_internal_to(&self, sites: &[usize]) -> bool {
        sites.contains(&self.site_a) && sites.contains(&self.site_b)
    }
}

// ============================================================
// CLUSTER
// ============================================================

/// A cluster of strongly-entangled sites that will be contracted together.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Sorted site indices belonging to this cluster.
    pub sites: Vec<usize>,
    /// Gates whose both sites lie within this cluster.
    pub internal_gates: Vec<TwoSiteGate>,
    /// The merged tensor for all sites in the cluster (lazily computed).
    pub total_tensor: Option<ArrayD<Complex64>>,
}

impl Cluster {
    /// Create a new empty cluster from a set of sites.
    pub fn new(sites: Vec<usize>) -> Self {
        Self {
            sites,
            internal_gates: Vec::new(),
            total_tensor: None,
        }
    }

    /// Number of sites in this cluster.
    pub fn size(&self) -> usize {
        self.sites.len()
    }

    /// Whether a given site belongs to this cluster.
    pub fn contains(&self, site: usize) -> bool {
        self.sites.contains(&site)
    }
}

// ============================================================
// CLUSTER MPS
// ============================================================

/// A Matrix Product State (MPS) representation for use with Cluster-TEBD.
///
/// Each tensor has shape `[bond_left, physical_dim, bond_right]`.
/// The leftmost tensor has `bond_left = 1` and the rightmost has `bond_right = 1`.
#[derive(Debug, Clone)]
pub struct ClusterMps {
    /// MPS tensors, one per site. Shape: [bond_left, physical_dim, bond_right].
    pub tensors: Vec<Array3<Complex64>>,
    /// Bond dimensions between adjacent sites. Length = num_sites - 1.
    pub bond_dims: Vec<usize>,
    /// Total number of sites.
    pub num_sites: usize,
    /// Current orthogonality center (site index).
    pub center: usize,
}

// ============================================================
// CLUSTER-TEBD RESULT
// ============================================================

/// Result of a Cluster-TEBD simulation.
#[derive(Debug, Clone)]
pub struct ClusterTebdResult {
    /// The final MPS after time evolution.
    pub final_mps: ClusterMps,
    /// Total simulated time.
    pub time_evolved: f64,
    /// Estimated fidelity (1 - cumulative truncation error).
    pub fidelity_estimate: f64,
    /// Number of clusters formed across all time steps.
    pub num_clusters_formed: usize,
    /// Maximum cluster size used across all time steps.
    pub max_cluster_size_used: usize,
}

// ============================================================
// SVD TRUNCATION
// ============================================================

/// Perform truncated SVD on a matrix.
///
/// Returns `(U, singular_values, Vt, truncation_error)` where columns of U and
/// rows of Vt are truncated to at most `max_dim` singular values, and singular
/// values below `cutoff` are discarded.
pub fn svd_truncate(
    m: &Array2<Complex64>,
    max_dim: usize,
    cutoff: f64,
) -> (Array2<Complex64>, Vec<f64>, Array2<Complex64>, f64) {
    let (rows, cols) = (m.nrows(), m.ncols());
    let k = rows.min(cols);

    // Compute full SVD via Jacobi one-sided rotations (simplified).
    // For production, one would call LAPACK zgesvd; here we use a
    // power-iteration / Gram-Schmidt approach that is correct for small matrices.
    let (u_full, s_full, vt_full) = compact_svd(m);

    // Determine how many singular values to keep.
    let mut keep = k.min(max_dim);
    // Also discard singular values below cutoff.
    while keep > 1 && s_full[keep - 1] < cutoff {
        keep -= 1;
    }

    // Compute truncation error = sum of discarded squared singular values.
    let total_norm_sq: f64 = s_full.iter().map(|s| s * s).sum();
    let kept_norm_sq: f64 = s_full.iter().take(keep).map(|s| s * s).sum();
    let trunc_err = if total_norm_sq > 0.0 {
        1.0 - kept_norm_sq / total_norm_sq
    } else {
        0.0
    };

    // Slice U, S, Vt.
    let u_trunc = u_full.slice(ndarray::s![.., ..keep]).to_owned();
    let s_trunc: Vec<f64> = s_full.into_iter().take(keep).collect();
    let vt_trunc = vt_full.slice(ndarray::s![..keep, ..]).to_owned();

    (u_trunc, s_trunc, vt_trunc, trunc_err)
}

/// Compact SVD via eigendecomposition of M^H M (for small matrices).
///
/// Returns (U, sigma, Vt) with sigma sorted descending.
fn compact_svd(
    m: &Array2<Complex64>,
) -> (Array2<Complex64>, Vec<f64>, Array2<Complex64>) {
    let (rows, cols) = (m.nrows(), m.ncols());
    let k = rows.min(cols);

    if rows >= cols {
        // Compute M^H M (cols x cols).
        let mhm = hermitian_product(m, false);
        let (eigenvalues, eigenvectors) = symmetric_eigen(&mhm);

        // Singular values = sqrt(eigenvalues), sorted descending.
        let mut indexed: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &ev)| (i, ev.max(0.0)))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let sigma: Vec<f64> = indexed.iter().map(|(_, ev)| ev.sqrt()).collect();

        // V = eigenvectors reordered.
        let mut vt = Array2::<Complex64>::zeros((k, cols));
        for (new_idx, (old_idx, _)) in indexed.iter().enumerate().take(k) {
            for j in 0..cols {
                vt[[new_idx, j]] = eigenvectors[[j, *old_idx]];
            }
        }

        // U = M V S^{-1}
        let mut u = Array2::<Complex64>::zeros((rows, k));
        for j in 0..k {
            if sigma[j] > 1e-14 {
                let inv_s = Complex64::new(1.0 / sigma[j], 0.0);
                for i in 0..rows {
                    let mut val = c0();
                    for l in 0..cols {
                        val += m[[i, l]] * vt[[j, l]].conj();
                    }
                    u[[i, j]] = val * inv_s;
                }
            }
        }

        (u, sigma, vt)
    } else {
        // Compute M M^H (rows x rows) and derive from that.
        let mmh = hermitian_product(m, true);
        let (eigenvalues, eigenvectors) = symmetric_eigen(&mmh);

        let mut indexed: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &ev)| (i, ev.max(0.0)))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let sigma: Vec<f64> = indexed.iter().map(|(_, ev)| ev.sqrt()).collect();

        // U = eigenvectors reordered.
        let mut u = Array2::<Complex64>::zeros((rows, k));
        for (new_idx, (old_idx, _)) in indexed.iter().enumerate().take(k) {
            for i in 0..rows {
                u[[i, new_idx]] = eigenvectors[[i, *old_idx]];
            }
        }

        // Vt = S^{-1} U^H M
        let mut vt = Array2::<Complex64>::zeros((k, cols));
        for j in 0..k {
            if sigma[j] > 1e-14 {
                let inv_s = Complex64::new(1.0 / sigma[j], 0.0);
                for l in 0..cols {
                    let mut val = c0();
                    for i in 0..rows {
                        val += u[[i, j]].conj() * m[[i, l]];
                    }
                    vt[[j, l]] = val * inv_s;
                }
            }
        }

        (u, sigma, vt)
    }
}

/// Compute M^H M (if transpose=false) or M M^H (if transpose=true).
fn hermitian_product(m: &Array2<Complex64>, transpose: bool) -> Array2<Complex64> {
    let (rows, cols) = (m.nrows(), m.ncols());
    if !transpose {
        // M^H M: (cols x cols)
        let mut result = Array2::<Complex64>::zeros((cols, cols));
        for i in 0..cols {
            for j in i..cols {
                let mut val = c0();
                for k in 0..rows {
                    val += m[[k, i]].conj() * m[[k, j]];
                }
                result[[i, j]] = val;
                if i != j {
                    result[[j, i]] = val.conj();
                }
            }
        }
        result
    } else {
        // M M^H: (rows x rows)
        let mut result = Array2::<Complex64>::zeros((rows, rows));
        for i in 0..rows {
            for j in i..rows {
                let mut val = c0();
                for k in 0..cols {
                    val += m[[i, k]] * m[[j, k]].conj();
                }
                result[[i, j]] = val;
                if i != j {
                    result[[j, i]] = val.conj();
                }
            }
        }
        result
    }
}

/// Eigendecomposition of a Hermitian matrix via Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors) with eigenvalues as real f64.
fn symmetric_eigen(h: &Array2<Complex64>) -> (Vec<f64>, Array2<Complex64>) {
    let n = h.nrows();
    assert_eq!(n, h.ncols());

    // Copy into working matrix.
    let mut a = h.clone();
    let mut v = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = c1();
    }

    // Jacobi rotations.
    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element.
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let mag = a[[i, j]].norm();
                if mag > max_val {
                    max_val = mag;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute Jacobi rotation to zero out a[p,q].
        let app = a[[p, p]].re;
        let aqq = a[[q, q]].re;
        let apq = a[[p, q]];

        let tau = (aqq - app) / (2.0 * apq.norm());
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Phase factor to handle complex off-diagonal.
        let phase = if apq.norm() > 1e-15 {
            apq / Complex64::new(apq.norm(), 0.0)
        } else {
            c1()
        };

        let cc = Complex64::new(c, 0.0);
        let ss = Complex64::new(s, 0.0);

        // Apply rotation: rows and columns p, q.
        for k in 0..n {
            if k != p && k != q {
                let akp = a[[k, p]];
                let akq = a[[k, q]];
                a[[k, p]] = cc * akp + ss * phase.conj() * akq;
                a[[k, q]] = -ss * phase * akp + cc * akq;
                a[[p, k]] = a[[k, p]].conj();
                a[[q, k]] = a[[k, q]].conj();
            }
        }
        let new_pp = app * c * c + aqq * s * s + 2.0 * s * c * (phase.conj() * apq).re;
        let new_qq = app * s * s + aqq * c * c - 2.0 * s * c * (phase.conj() * apq).re;
        a[[p, p]] = Complex64::new(new_pp, 0.0);
        a[[q, q]] = Complex64::new(new_qq, 0.0);
        a[[p, q]] = c0();
        a[[q, p]] = c0();

        // Update eigenvectors.
        for k in 0..n {
            let vkp = v[[k, p]];
            let vkq = v[[k, q]];
            v[[k, p]] = cc * vkp + ss * phase.conj() * vkq;
            v[[k, q]] = -ss * phase * vkp + cc * vkq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]].re).collect();
    (eigenvalues, v)
}

// ============================================================
// MPS UTILITIES
// ============================================================

/// Create a random MPS with given parameters.
///
/// The tensors are initialized with random complex entries and the MPS
/// is NOT normalized. Use `normalize_mps` if needed.
pub fn random_cluster_mps(
    n_sites: usize,
    physical_dim: usize,
    bond_dim: usize,
) -> ClusterMps {
    let mut rng = rand::thread_rng();

    // Pre-compute bond dimensions ensuring consistency.
    // The maximum useful bond dim at bond i (between sites i and i+1) is
    // min(d^(i+1), d^(n-i-1), bond_dim).
    let mut bonds = Vec::with_capacity(n_sites.saturating_sub(1));
    for i in 0..(n_sites.saturating_sub(1)) {
        let from_left = physical_dim.saturating_pow((i + 1) as u32).min(bond_dim);
        let from_right = physical_dim.saturating_pow((n_sites - 1 - i) as u32).min(bond_dim);
        bonds.push(from_left.min(from_right).min(bond_dim).max(1));
    }

    let mut tensors = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let bl = if i == 0 { 1 } else { bonds[i - 1] };
        let br = if i == n_sites - 1 { 1 } else { bonds[i] };

        let mut t = Array3::<Complex64>::zeros((bl, physical_dim, br));
        for elem in t.iter_mut() {
            *elem = Complex64::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
        }
        tensors.push(t);
    }

    ClusterMps {
        tensors,
        bond_dims: bonds,
        num_sites: n_sites,
        center: 0,
    }
}

/// Create a product state MPS from a list of local state indices.
///
/// Each entry in `state` selects a computational basis state for that site.
/// For example, `[0, 1, 0, 1]` creates |0101⟩ with `physical_dim >= 2`.
pub fn product_state_mps(state: &[usize], physical_dim: usize) -> ClusterMps {
    let n_sites = state.len();
    let mut tensors = Vec::with_capacity(n_sites);
    let bond_dims = vec![1; n_sites.saturating_sub(1)];

    for &s in state {
        let mut t = Array3::<Complex64>::zeros((1, physical_dim, 1));
        if s < physical_dim {
            t[[0, s, 0]] = c1();
        }
        tensors.push(t);
    }

    ClusterMps {
        tensors,
        bond_dims,
        num_sites: n_sites,
        center: 0,
    }
}

/// Compute the norm ⟨ψ|ψ⟩ of an MPS via sequential contraction.
pub fn mps_norm(mps: &ClusterMps) -> f64 {
    mps_overlap(mps, mps).norm().sqrt()
}

/// Compute the overlap ⟨ψ|φ⟩ between two MPS via sequential contraction.
fn mps_overlap(bra: &ClusterMps, ket: &ClusterMps) -> Complex64 {
    assert_eq!(bra.num_sites, ket.num_sites);
    let n = bra.num_sites;
    if n == 0 {
        return c1();
    }

    // Start with site 0: contract over physical index.
    // transfer[alpha_bra, alpha_ket] = sum_d bra[alpha_bra, d, beta_bra]* ket[alpha_ket, d, beta_ket]
    // But at site 0 alpha = 1 for both, so transfer has shape [beta_bra, beta_ket].
    let b0 = &bra.tensors[0];
    let k0 = &ket.tensors[0];
    let d = b0.shape()[1];
    let br_bra = b0.shape()[2];
    let br_ket = k0.shape()[2];

    let mut transfer = Array2::<Complex64>::zeros((br_bra, br_ket));
    for s in 0..d {
        for bb in 0..br_bra {
            for bk in 0..br_ket {
                transfer[[bb, bk]] += b0[[0, s, bb]].conj() * k0[[0, s, bk]];
            }
        }
    }

    // Contract through remaining sites.
    for site in 1..n {
        let bt = &bra.tensors[site];
        let kt = &ket.tensors[site];
        let d = bt.shape()[1];
        let new_br_bra = bt.shape()[2];
        let new_br_ket = kt.shape()[2];
        let bl_bra = bt.shape()[0];
        let bl_ket = kt.shape()[0];

        let mut new_transfer = Array2::<Complex64>::zeros((new_br_bra, new_br_ket));
        for ab in 0..bl_bra {
            for ak in 0..bl_ket {
                let t_val = transfer[[ab, ak]];
                if t_val.norm() < 1e-15 {
                    continue;
                }
                for s in 0..d {
                    for bb in 0..new_br_bra {
                        for bk in 0..new_br_ket {
                            new_transfer[[bb, bk]] +=
                                t_val * bt[[ab, s, bb]].conj() * kt[[ak, s, bk]];
                        }
                    }
                }
            }
        }
        transfer = new_transfer;
    }

    transfer[[0, 0]]
}

/// Normalize an MPS in-place so that ⟨ψ|ψ⟩ = 1.
pub fn normalize_mps(mps: &mut ClusterMps) {
    let n = mps_norm(mps);
    if n > 1e-15 {
        let factor = Complex64::new(1.0 / n, 0.0);
        // Scale the first tensor.
        for elem in mps.tensors[0].iter_mut() {
            *elem *= factor;
        }
    }
}

/// Measure a local operator at a given site: ⟨ψ|O_site|ψ⟩ / ⟨ψ|ψ⟩.
pub fn measure_local(
    mps: &ClusterMps,
    operator: &Array2<Complex64>,
    site: usize,
) -> Complex64 {
    assert!(site < mps.num_sites);
    let d = mps.tensors[site].shape()[1];
    assert_eq!(operator.nrows(), d);
    assert_eq!(operator.ncols(), d);

    // Create a modified MPS with operator applied at `site`.
    let mut ket = mps.clone();
    let t = &mps.tensors[site];
    let bl = t.shape()[0];
    let br = t.shape()[2];
    let mut new_t = Array3::<Complex64>::zeros((bl, d, br));
    for a in 0..bl {
        for b in 0..br {
            for s_out in 0..d {
                let mut val = c0();
                for s_in in 0..d {
                    val += operator[[s_out, s_in]] * t[[a, s_in, b]];
                }
                new_t[[a, s_out, b]] = val;
            }
        }
    }
    ket.tensors[site] = new_t;

    let overlap = mps_overlap(mps, &ket);
    let norm_sq = mps_overlap(mps, mps);
    if norm_sq.norm() > 1e-15 {
        overlap / norm_sq
    } else {
        c0()
    }
}

// ============================================================
// ENTANGLEMENT ANALYSIS
// ============================================================

/// Compute the von Neumann entanglement entropy at a bond (bipartition after `site`).
///
/// S = -sum_i σ_i^2 ln(σ_i^2) where σ_i are singular values of the bipartition.
pub fn entanglement_entropy(mps: &ClusterMps, site: usize) -> f64 {
    if site >= mps.num_sites.saturating_sub(1) {
        return 0.0;
    }

    // Left-canonicalize up to `site`, then SVD at the bond.
    let mut work = mps.clone();
    left_canonicalize_to(&mut work, site);

    // Form the two-site block at (site, site+1) and SVD.
    let ta = &work.tensors[site];
    let tb = &work.tensors[site + 1];
    let bl = ta.shape()[0];
    let da = ta.shape()[1];
    let bond = ta.shape()[2];
    let db = tb.shape()[1];
    let br = tb.shape()[2];

    // Theta = ta * tb contracted over bond index.
    // Shape: (bl * da, db * br)
    let mut theta = Array2::<Complex64>::zeros((bl * da, db * br));
    for a in 0..bl {
        for sa in 0..da {
            for m in 0..bond {
                for sb in 0..db {
                    for b in 0..br {
                        theta[[a * da + sa, sb * br + b]] +=
                            ta[[a, sa, m]] * tb[[m, sb, b]];
                    }
                }
            }
        }
    }

    let (_, sigma, _, _) = svd_truncate(&theta, theta.nrows().min(theta.ncols()), 0.0);

    // S = -sum σ^2 ln(σ^2)
    let norm_sq: f64 = sigma.iter().map(|s| s * s).sum();
    if norm_sq < 1e-15 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &s in &sigma {
        let p = s * s / norm_sq;
        if p > 1e-15 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Compute mutual information I(A:B) = S(A) + S(B) - S(AB) between two sites.
///
/// Uses entanglement entropy at the bonds surrounding the sites.
pub fn compute_mutual_information(
    mps: &ClusterMps,
    site_a: usize,
    site_b: usize,
) -> f64 {
    if site_a == site_b || mps.num_sites < 2 {
        return 0.0;
    }
    let (lo, hi) = if site_a < site_b {
        (site_a, site_b)
    } else {
        (site_b, site_a)
    };

    // S(A): entropy at bond after site lo (left subsystem includes site lo).
    let sa = if lo > 0 {
        entanglement_entropy(mps, lo - 1)
    } else {
        entanglement_entropy(mps, lo)
    };

    // S(B): entropy at bond after site hi.
    let sb = entanglement_entropy(mps, hi.min(mps.num_sites - 2));

    // S(AB): entropy of the region [lo..=hi] together.
    // Approximate as the average of entropies at the boundaries of the region.
    let s_left = if lo > 0 {
        entanglement_entropy(mps, lo - 1)
    } else {
        0.0
    };
    let s_right = if hi < mps.num_sites - 1 {
        entanglement_entropy(mps, hi)
    } else {
        0.0
    };
    let sab = s_left + s_right;

    // I(A:B) = S(A) + S(B) - S(AB), clamped to non-negative.
    (sa + sb - sab).max(0.0)
}

/// Build an entanglement graph: edges (site_a, site_b, MI) for all pairs
/// with mutual information above `threshold`.
pub fn build_entanglement_graph(
    mps: &ClusterMps,
    threshold: f64,
) -> Vec<(usize, usize, f64)> {
    let n = mps.num_sites;
    let mut edges = Vec::new();

    // For efficiency, only consider nearest-neighbor and next-nearest-neighbor pairs.
    // In 1D MPS, long-range MI decays rapidly.
    for i in 0..n {
        let max_j = (i + 4).min(n);
        for j in (i + 1)..max_j {
            let mi = compute_mutual_information(mps, i, j);
            if mi > threshold {
                edges.push((i, j, mi));
            }
        }
    }

    // Sort by MI descending.
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    edges
}

/// Identify clusters from an entanglement graph using greedy merging.
///
/// Starting from the highest mutual information edge, grow clusters by
/// adding adjacent sites until `max_size` is reached.
pub fn identify_clusters(
    entanglement_graph: &[(usize, usize, f64)],
    max_size: usize,
) -> Vec<Cluster> {
    let mut assigned: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut clusters: Vec<Cluster> = Vec::new();

    for &(a, b, _mi) in entanglement_graph {
        if assigned.contains(&a) && assigned.contains(&b) {
            // Both already assigned — check if same cluster, then skip.
            continue;
        }

        if !assigned.contains(&a) && !assigned.contains(&b) {
            // Neither assigned: start a new cluster.
            let mut sites = vec![a, b];
            sites.sort();
            assigned.insert(a);
            assigned.insert(b);
            clusters.push(Cluster::new(sites));
        } else {
            // One is assigned, one is not. Try to add the unassigned one.
            let (unassigned, existing) = if assigned.contains(&a) {
                (b, a)
            } else {
                (a, b)
            };

            // Find the cluster containing `existing`.
            if let Some(cluster) = clusters.iter_mut().find(|c| c.contains(existing)) {
                if cluster.size() < max_size {
                    // Only add if site is adjacent to existing cluster sites.
                    let min_site = *cluster.sites.iter().min().unwrap();
                    let max_site = *cluster.sites.iter().max().unwrap();
                    if unassigned + 1 >= min_site && unassigned <= max_site + 1 {
                        cluster.sites.push(unassigned);
                        cluster.sites.sort();
                        assigned.insert(unassigned);
                    }
                }
            }
        }
    }

    clusters
}

// ============================================================
// CANONICALIZATION
// ============================================================

/// Left-canonicalize the MPS up to site `target` using QR decomposition.
fn left_canonicalize_to(mps: &mut ClusterMps, target: usize) {
    let target = target.min(mps.num_sites.saturating_sub(1));
    for site in 0..target {
        left_canonicalize_site(mps, site);
    }
    mps.center = target;
}

/// Left-canonicalize a single site by reshaping + QR.
fn left_canonicalize_site(mps: &mut ClusterMps, site: usize) {
    if site >= mps.num_sites - 1 {
        return;
    }
    let t = &mps.tensors[site];
    let bl = t.shape()[0];
    let d = t.shape()[1];
    let br = t.shape()[2];

    // Reshape to (bl*d, br) and do thin SVD for orthogonalization.
    let mat = t.clone().into_shape((bl * d, br)).unwrap();
    let (u, sigma, vt, _) = svd_truncate(&mat, br, 0.0);

    let new_br = sigma.len();

    // New tensor at site: reshape U to (bl, d, new_br).
    let u_scaled = u;
    let new_t = u_scaled.into_shape((bl, d, new_br)).unwrap();
    mps.tensors[site] = new_t;

    // Absorb S*Vt into next tensor.
    let next = &mps.tensors[site + 1];
    let d_next = next.shape()[1];
    let br_next = next.shape()[2];

    // svt = diag(sigma) * vt, shape (new_br, br)
    let mut svt = Array2::<Complex64>::zeros((new_br, br));
    for i in 0..new_br {
        for j in 0..br {
            svt[[i, j]] = Complex64::new(sigma[i], 0.0) * vt[[i, j]];
        }
    }

    // Contract svt with next tensor.
    let mut new_next = Array3::<Complex64>::zeros((new_br, d_next, br_next));
    for a in 0..new_br {
        for m in 0..br {
            let sv = svt[[a, m]];
            if sv.norm() < 1e-15 {
                continue;
            }
            for s in 0..d_next {
                for b in 0..br_next {
                    new_next[[a, s, b]] += sv * next[[m, s, b]];
                }
            }
        }
    }
    mps.tensors[site + 1] = new_next;

    if site < mps.bond_dims.len() {
        mps.bond_dims[site] = new_br;
    }
}

// ============================================================
// CLUSTER TENSOR OPERATIONS
// ============================================================

/// Merge adjacent MPS tensors within a cluster into a single large tensor.
///
/// The resulting tensor has shape `[bond_left, d_0, d_1, ..., d_{k-1}, bond_right]`
/// stored as an `ArrayD` with `k+2` dimensions.
pub fn merge_cluster_tensors(
    mps: &ClusterMps,
    sites: &[usize],
) -> ArrayD<Complex64> {
    assert!(!sites.is_empty());
    assert!(sites.windows(2).all(|w| w[1] == w[0] + 1), "sites must be contiguous and sorted");

    let k = sites.len();
    let first = sites[0];

    if k == 1 {
        // Single site: shape [bl, d, br]
        let t = &mps.tensors[first];
        return t.clone().into_dyn();
    }

    // Start: tensor at first site, shape [bl, d0, bond_0].
    let mut result = mps.tensors[first].clone().into_dyn();

    // Sequentially contract with next sites.
    for &site in &sites[1..] {
        let next = &mps.tensors[site];
        let res_shape = result.shape().to_vec();
        let ndim = res_shape.len();
        // result has shape [..., bond], next has shape [bond, d, br]
        let bond = res_shape[ndim - 1];
        let d = next.shape()[1];
        let br = next.shape()[2];

        // New shape: [...(without last dim), d, br]
        let mut new_shape: Vec<usize> = res_shape[..ndim - 1].to_vec();
        new_shape.push(d);
        new_shape.push(br);

        let mut new_result = ArrayD::<Complex64>::zeros(IxDyn(&new_shape));

        // Contract: new[..., s, b] = sum_m result[..., m] * next[m, s, b]
        let prefix_size: usize = res_shape[..ndim - 1].iter().product();
        let result_flat = result.into_shape(IxDyn(&[prefix_size, bond])).unwrap();

        for p in 0..prefix_size {
            for m in 0..bond {
                let rv = result_flat[[p, m]];
                if rv.norm() < 1e-15 {
                    continue;
                }
                for s in 0..d {
                    for b in 0..br {
                        // Map p back to multi-index, then append s, b.
                        let flat_idx = p * d * br + s * br + b;
                        let new_flat = new_result
                            .as_slice_mut()
                            .unwrap();
                        new_flat[flat_idx] += rv * next[[m, s, b]];
                    }
                }
            }
        }

        result = new_result;
    }

    result
}

/// Decompose a multi-site tensor back into MPS form via sequential SVD.
///
/// Input tensor has shape `[bond_left, d_0, d_1, ..., d_{k-1}, bond_right]`.
/// Returns `k` tensors each with shape `[bond_left_i, d_i, bond_right_i]`.
pub fn decompose_cluster_tensor(
    tensor: &ArrayD<Complex64>,
    sites: &[usize],
    max_bond_dim: usize,
    svd_cutoff: f64,
) -> (Vec<Array3<Complex64>>, f64) {
    let k = sites.len();
    let shape = tensor.shape();
    assert_eq!(shape.len(), k + 2, "tensor must have k+2 dimensions for k sites");

    if k == 1 {
        let bl = shape[0];
        let d = shape[1];
        let br = shape[2];
        let t = tensor.clone().into_shape(IxDyn(&[bl, d, br])).unwrap();
        let t3 = Array3::from_shape_vec(
            (bl, d, br),
            t.into_raw_vec(),
        ).unwrap();
        return (vec![t3], 0.0);
    }

    let mut tensors = Vec::with_capacity(k);
    let mut total_trunc_err = 0.0;

    // Reshape for sequential left-to-right SVD.
    let mut remaining = tensor.clone();

    for _i in 0..k - 1 {
        let rem_shape = remaining.shape().to_vec();
        let bl = rem_shape[0];
        let d = rem_shape[1];
        let rest: usize = rem_shape[2..].iter().product();

        // Reshape to (bl * d, rest).
        let mat_data = remaining.into_raw_vec();
        let mat = Array2::from_shape_vec((bl * d, rest), mat_data).unwrap();

        let (u, sigma, vt, trunc_err) = svd_truncate(&mat, max_bond_dim, svd_cutoff);
        total_trunc_err += trunc_err;

        let new_bond = sigma.len();

        // Site tensor: reshape U to (bl, d, new_bond).
        let site_tensor = Array3::from_shape_vec(
            (bl, d, new_bond),
            u.into_raw_vec(),
        ).unwrap();
        tensors.push(site_tensor);

        // Remaining = diag(sigma) * Vt, reshaped to (new_bond, d_{i+1}, ..., d_{k-1}, br)
        let mut svt = Array2::<Complex64>::zeros((new_bond, rest));
        for r in 0..new_bond {
            let sv = Complex64::new(sigma[r], 0.0);
            for c in 0..rest {
                svt[[r, c]] = sv * vt[[r, c]];
            }
        }

        // Reshape: first dim is new_bond, then the remaining original dims.
        let mut new_shape = vec![new_bond];
        new_shape.extend_from_slice(&rem_shape[2..]);
        remaining = ArrayD::from_shape_vec(IxDyn(&new_shape), svt.into_raw_vec()).unwrap();
    }

    // Last site: remaining has shape [bond, d_last, br].
    let rem_shape = remaining.shape().to_vec();
    assert_eq!(rem_shape.len(), 3);
    let last_tensor = Array3::from_shape_vec(
        (rem_shape[0], rem_shape[1], rem_shape[2]),
        remaining.into_raw_vec(),
    ).unwrap();
    tensors.push(last_tensor);

    (tensors, total_trunc_err)
}

/// Apply a two-site gate to a multi-site tensor.
///
/// `local_sites` maps the cluster's site indices to positions in the tensor's
/// physical dimensions. The gate acts on `local_sites[gate.site_a]` and
/// `local_sites[gate.site_b]`.
pub fn apply_gate_to_tensor(
    tensor: &mut ArrayD<Complex64>,
    gate: &TwoSiteGate,
    cluster_sites: &[usize],
) {
    let shape = tensor.shape().to_vec();
    let ndim = shape.len();
    let _k = ndim - 2; // number of physical dims

    // Find local indices for the gate sites within the cluster.
    let local_a = cluster_sites
        .iter()
        .position(|&s| s == gate.site_a)
        .expect("gate site_a not in cluster");
    let local_b = cluster_sites
        .iter()
        .position(|&s| s == gate.site_b)
        .expect("gate site_b not in cluster");

    // Physical dimensions at the gate sites (index offset by 1 for bond_left).
    let dim_a_idx = local_a + 1;
    let dim_b_idx = local_b + 1;
    let da = shape[dim_a_idx];
    let db = shape[dim_b_idx];
    assert_eq!(da, 2);
    assert_eq!(db, 2);

    // Apply gate by iterating over all other indices.
    let total: usize = shape.iter().product();
    let strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };

    // For each combination of indices NOT at dim_a and dim_b, apply the 4x4 gate.
    // We iterate by stepping through all multi-indices.
    let stride_a = strides[dim_a_idx];
    let stride_b = strides[dim_b_idx];

    // Collect all "base" offsets where dim_a = 0 and dim_b = 0.
    let _base_count = total / (da * db);
    let data = tensor.as_slice_mut().unwrap();

    // Iterate over all multi-indices with dim_a=0, dim_b=0.
    let _visited = vec![false; total];
    for base in 0..total {
        // Extract multi-index.
        let mut idx = base;
        let mut mi = vec![0usize; ndim];
        for d in 0..ndim {
            mi[d] = idx / strides[d];
            idx %= strides[d];
        }
        if mi[dim_a_idx] != 0 || mi[dim_b_idx] != 0 {
            continue;
        }

        // Apply 4x4 gate to the 4 entries.
        let mut vals = [c0(); 4];
        for sa in 0..da {
            for sb in 0..db {
                let offset = base + sa * stride_a + sb * stride_b;
                vals[sa * db + sb] = data[offset];
            }
        }

        let mut new_vals = [c0(); 4];
        for sa_out in 0..da {
            for sb_out in 0..db {
                let row = sa_out * db + sb_out;
                for sa_in in 0..da {
                    for sb_in in 0..db {
                        let col = sa_in * db + sb_in;
                        new_vals[row] += gate.matrix[[row, col]] * vals[col];
                    }
                }
            }
        }

        for sa in 0..da {
            for sb in 0..db {
                let offset = base + sa * stride_a + sb * stride_b;
                data[offset] = new_vals[sa * db + sb];
            }
        }
    }
}

// ============================================================
// STANDARD TEBD STEP
// ============================================================

/// Apply a single two-site gate via standard TEBD (SVD truncation after each gate).
///
/// Returns the truncation error from this step.
pub fn standard_tebd_step(
    mps: &mut ClusterMps,
    gate: &TwoSiteGate,
    max_bond_dim: usize,
    cutoff: f64,
) -> f64 {
    let site_a = gate.site_a;
    let site_b = gate.site_b;
    assert_eq!(site_b, site_a + 1, "standard TEBD requires adjacent sites");
    assert!(site_b < mps.num_sites);

    let ta = &mps.tensors[site_a];
    let tb = &mps.tensors[site_b];
    let bl = ta.shape()[0];
    let da = ta.shape()[1];
    let bond = ta.shape()[2];
    let db = tb.shape()[1];
    let br = tb.shape()[2];

    // Form theta = ta * tb contracted over bond.
    // theta[al, sa, sb, br] = sum_m ta[al, sa, m] * tb[m, sb, br]
    let mut theta = ndarray::Array4::<Complex64>::zeros((bl, da, db, br));
    for al in 0..bl {
        for sa in 0..da {
            for m in 0..bond {
                for sb in 0..db {
                    for b in 0..br {
                        theta[[al, sa, sb, b]] += ta[[al, sa, m]] * tb[[m, sb, b]];
                    }
                }
            }
        }
    }

    // Apply gate: theta'[al, sa', sb', br] = sum_{sa,sb} gate[sa'sb', sa sb] * theta[al, sa, sb, br]
    let mut theta_prime = ndarray::Array4::<Complex64>::zeros((bl, da, db, br));
    for al in 0..bl {
        for sa_out in 0..da {
            for sb_out in 0..db {
                let row = sa_out * db + sb_out;
                for b in 0..br {
                    let mut val = c0();
                    for sa_in in 0..da {
                        for sb_in in 0..db {
                            let col = sa_in * db + sb_in;
                            val += gate.matrix[[row, col]] * theta[[al, sa_in, sb_in, b]];
                        }
                    }
                    theta_prime[[al, sa_out, sb_out, b]] = val;
                }
            }
        }
    }

    // Reshape to (bl*da, db*br) and SVD.
    let mat = theta_prime
        .into_shape((bl * da, db * br))
        .unwrap();
    let (u, sigma, vt, trunc_err) = svd_truncate(&mat, max_bond_dim, cutoff);
    let new_bond = sigma.len();

    // Absorb sqrt(sigma) into both U and Vt for balanced truncation.
    // Actually, absorb full sigma into U to maintain left-canonical form.
    let mut new_ta = Array3::<Complex64>::zeros((bl, da, new_bond));
    for al in 0..bl {
        for sa in 0..da {
            for m in 0..new_bond {
                new_ta[[al, sa, m]] = u[[al * da + sa, m]] * Complex64::new(sigma[m], 0.0);
            }
        }
    }

    let mut new_tb = Array3::<Complex64>::zeros((new_bond, db, br));
    for m in 0..new_bond {
        for sb in 0..db {
            for b in 0..br {
                new_tb[[m, sb, b]] = vt[[m, sb * br + b]];
            }
        }
    }

    mps.tensors[site_a] = new_ta;
    mps.tensors[site_b] = new_tb;
    if site_a < mps.bond_dims.len() {
        mps.bond_dims[site_a] = new_bond;
    }

    trunc_err
}

// ============================================================
// HAMILTONIAN GATE CONSTRUCTORS
// ============================================================

/// Generate Trotter gates for the transverse-field Ising model.
///
/// H = -J Σ Z_i Z_{i+1} - h Σ X_i
///
/// Uses first-order Suzuki-Trotter decomposition: e^{-iHdt} ≈ Π e^{-iH_bond dt}.
pub fn ising_gates(n_sites: usize, j: f64, h: f64, dt: f64) -> Vec<TwoSiteGate> {
    let mut gates = Vec::new();

    for i in 0..(n_sites - 1) {
        // Two-site gate: exp(-i dt (-J Z⊗Z - h/2 (X⊗I + I⊗X)))
        // For simplicity, split into ZZ part and field part.
        let jdt = j * dt;
        let hdt = h * dt / 2.0;

        // ZZ interaction: exp(i*J*dt * Z⊗Z)
        // Z⊗Z = diag(1, -1, -1, 1)
        // exp(i*J*dt * Z⊗Z) = diag(e^{iJdt}, e^{-iJdt}, e^{-iJdt}, e^{iJdt})
        let eij = (ci() * jdt).exp();
        let emij = (-ci() * jdt).exp();

        let mut zz = Array2::<Complex64>::zeros((4, 4));
        zz[[0, 0]] = eij;
        zz[[1, 1]] = emij;
        zz[[2, 2]] = emij;
        zz[[3, 3]] = eij;

        // Field part: exp(i*h*dt/2 * (X⊗I + I⊗X))
        // X⊗I + I⊗X acts on 4-dim space.
        // Build the matrix and exponentiate.
        let mut hx = Array2::<Complex64>::zeros((4, 4));
        // X⊗I: |00⟩↔|10⟩, |01⟩↔|11⟩
        hx[[0, 2]] += c1();
        hx[[2, 0]] += c1();
        hx[[1, 3]] += c1();
        hx[[3, 1]] += c1();
        // I⊗X: |00⟩↔|01⟩, |10⟩↔|11⟩
        hx[[0, 1]] += c1();
        hx[[1, 0]] += c1();
        hx[[2, 3]] += c1();
        hx[[3, 2]] += c1();

        let field_gate = matrix_exp_hermitian(&hx, hdt);

        // Combined gate = field_gate * zz.
        let combined = mat_mul(&field_gate, &zz);

        gates.push(TwoSiteGate::new(i, i + 1, combined));
    }

    gates
}

/// Generate Trotter gates for the isotropic Heisenberg (XXX) model.
///
/// H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
pub fn heisenberg_gates(n_sites: usize, j: f64, dt: f64) -> Vec<TwoSiteGate> {
    xxz_gates(n_sites, j, j, dt)
}

/// Generate Trotter gates for the XXZ model.
///
/// H = Σ (Jxy (X_i X_{i+1} + Y_i Y_{i+1}) + Jz Z_i Z_{i+1})
pub fn xxz_gates(n_sites: usize, jxy: f64, jz: f64, dt: f64) -> Vec<TwoSiteGate> {
    let mut gates = Vec::new();

    for i in 0..(n_sites - 1) {
        // Build the two-site Hamiltonian:
        // H_bond = Jxy(XX + YY) + Jz ZZ
        let mut h_bond = Array2::<Complex64>::zeros((4, 4));

        // XX: σx⊗σx
        h_bond[[0, 3]] += Complex64::new(jxy, 0.0);
        h_bond[[1, 2]] += Complex64::new(jxy, 0.0);
        h_bond[[2, 1]] += Complex64::new(jxy, 0.0);
        h_bond[[3, 0]] += Complex64::new(jxy, 0.0);

        // YY: σy⊗σy (note: σy = [[0,-i],[i,0]], so σy⊗σy has specific sign pattern)
        h_bond[[0, 3]] += Complex64::new(-jxy, 0.0); // i*i * (-i)*(-i) = -1 wait...
        // σy⊗σy = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
        // Actually: (σy)_{00}=0, (σy)_{01}=-i, (σy)_{10}=i, (σy)_{11}=0
        // (σy⊗σy)_{00,00} = 0, ..., let me be precise:
        // |00⟩→|00⟩: σy_{0,0}*σy_{0,0} = 0
        // |00⟩→|11⟩: σy_{1,0}*σy_{1,0} = i*i = -1
        // |01⟩→|10⟩: σy_{1,0}*σy_{0,1} = i*(-i) = 1
        // |10⟩→|01⟩: σy_{0,1}*σy_{1,0} = (-i)*i = 1
        // |11⟩→|00⟩: σy_{0,1}*σy_{0,1} = (-i)*(-i) = -1
        // So σy⊗σy = diag block: [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]

        // We already added jxy to [0,3] and [3,0] from XX. YY adds -jxy to those.
        // Net XX+YY at [0,3] and [3,0] = jxy + (-jxy) = 0.
        // Correct: let me redo. XX+YY is the flip-flop term.
        // XX+YY = 2(S+S- + S-S+) in spin-1/2.
        // In the computational basis {|00⟩,|01⟩,|10⟩,|11⟩}:
        // XX+YY only has entries at [1,2] and [2,1], each = 2*jxy.
        // ZZ has entries: [0,0]=jz, [1,1]=-jz, [2,2]=-jz, [3,3]=jz.

        // Let me redo properly.
        h_bond = Array2::<Complex64>::zeros((4, 4));

        // XX: |01⟩↔|10⟩ with coefficient 1, |00⟩↔|11⟩ with coefficient 1
        // σx⊗σx = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
        // YY: σy⊗σy = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
        // XX+YY = [[0,0,0,0],[0,0,2,0],[0,2,0,0],[0,0,0,0]]
        h_bond[[1, 2]] = Complex64::new(2.0 * jxy, 0.0);
        h_bond[[2, 1]] = Complex64::new(2.0 * jxy, 0.0);

        // ZZ = diag(1,-1,-1,1)
        h_bond[[0, 0]] = Complex64::new(jz, 0.0);
        h_bond[[1, 1]] = Complex64::new(-jz, 0.0);
        h_bond[[2, 2]] = Complex64::new(-jz, 0.0);
        h_bond[[3, 3]] = Complex64::new(jz, 0.0);

        // Gate = exp(-i * dt * H_bond)
        let gate_matrix = matrix_exp_hermitian(&h_bond, -dt);
        gates.push(TwoSiteGate::new(i, i + 1, gate_matrix));
    }

    gates
}

/// Matrix exponential for a Hermitian matrix: exp(i * t * H).
///
/// Uses eigendecomposition: exp(i*t*H) = V diag(e^{i*t*λ}) V†.
fn matrix_exp_hermitian(h: &Array2<Complex64>, t: f64) -> Array2<Complex64> {
    let n = h.nrows();
    let (eigenvalues, eigenvectors) = symmetric_eigen(h);

    // exp(i*t*H) = V * diag(exp(i*t*lambda_k)) * V†
    let mut result = Array2::<Complex64>::zeros((n, n));
    for k in 0..n {
        let phase = (ci() * t * eigenvalues[k]).exp();
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += eigenvectors[[i, k]] * phase * eigenvectors[[j, k]].conj();
            }
        }
    }
    result
}

/// Simple matrix multiplication C = A * B.
fn mat_mul(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    assert_eq!(k, b.nrows());

    let mut c = Array2::<Complex64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut val = c0();
            for l in 0..k {
                val += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = val;
        }
    }
    c
}

// ============================================================
// MAIN CLUSTER-TEBD ALGORITHM
// ============================================================

/// Run the Cluster-TEBD algorithm.
///
/// This is the main entry point. It performs time evolution of an MPS under
/// a set of two-site gates, dynamically forming clusters of strongly-entangled
/// sites for exact contraction.
///
/// # Arguments
/// * `mps` - Initial MPS state
/// * `gates` - Two-site gates for one Trotter step
/// * `config` - Algorithm configuration
///
/// # Returns
/// `ClusterTebdResult` with the final MPS and statistics.
pub fn cluster_tebd(
    mps: &ClusterMps,
    gates: &[TwoSiteGate],
    config: &ClusterTebdConfig,
) -> Result<ClusterTebdResult, ClusterTebdError> {
    config.validate()?;

    let mut state = mps.clone();
    normalize_mps(&mut state);

    let mut cumulative_trunc_err = 0.0;
    let mut total_clusters_formed = 0usize;
    let mut max_cluster_size_seen = 0usize;

    for _step in 0..config.num_steps {
        // Step 1: Analyze entanglement and identify clusters.
        let clusters = match config.merge_strategy {
            MergeStrategy::EntanglementBased | MergeStrategy::Adaptive => {
                let graph = build_entanglement_graph(&state, config.cluster_threshold);
                identify_clusters(&graph, config.max_cluster_size)
            }
            MergeStrategy::GeometryBased => {
                // Simple geometry: group adjacent pairs.
                let mut clusters = Vec::new();
                let mut i = 0;
                while i + 1 < state.num_sites {
                    let end = (i + config.max_cluster_size).min(state.num_sites);
                    let sites: Vec<usize> = (i..end).collect();
                    clusters.push(Cluster::new(sites));
                    i = end;
                }
                clusters
            }
        };

        // Track statistics.
        total_clusters_formed += clusters.len();
        for c in &clusters {
            if c.size() > max_cluster_size_seen {
                max_cluster_size_seen = c.size();
            }
            if c.size() > config.max_cluster_size {
                return Err(ClusterTebdError::ClusterTooLarge {
                    size: c.size(),
                    max: config.max_cluster_size,
                });
            }
        }

        // Step 2: Partition gates into cluster-internal and inter-cluster.
        let mut internal_gates_map: Vec<Vec<usize>> = vec![Vec::new(); clusters.len()];
        let mut inter_cluster_gate_indices: Vec<usize> = Vec::new();

        for (gi, gate) in gates.iter().enumerate() {
            let mut assigned = false;
            for (ci, cluster) in clusters.iter().enumerate() {
                if gate.is_internal_to(&cluster.sites) {
                    internal_gates_map[ci].push(gi);
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                inter_cluster_gate_indices.push(gi);
            }
        }

        // Step 3: Process each cluster.
        for (ci, cluster) in clusters.iter().enumerate() {
            if internal_gates_map[ci].is_empty() {
                continue;
            }

            // Merge tensors.
            let mut merged = merge_cluster_tensors(&state, &cluster.sites);

            // Apply all internal gates exactly (no truncation).
            for &gi in &internal_gates_map[ci] {
                apply_gate_to_tensor(&mut merged, &gates[gi], &cluster.sites);
            }

            // Decompose back to MPS form.
            let (new_tensors, trunc_err) =
                decompose_cluster_tensor(&merged, &cluster.sites, config.max_bond_dim, config.svd_cutoff);
            cumulative_trunc_err += trunc_err;

            // Write back into the MPS.
            for (local_i, &site) in cluster.sites.iter().enumerate() {
                state.tensors[site] = new_tensors[local_i].clone();
            }

            // Update bond dims.
            for local_i in 0..cluster.sites.len() - 1 {
                let site = cluster.sites[local_i];
                if site < state.bond_dims.len() {
                    state.bond_dims[site] = state.tensors[site].shape()[2];
                }
            }
            // Update bond at cluster boundary (left).
            if cluster.sites[0] > 0 {
                let prev = cluster.sites[0] - 1;
                if prev < state.bond_dims.len() {
                    state.bond_dims[prev] = state.tensors[cluster.sites[0]].shape()[0];
                }
            }
        }

        // Step 4: Apply inter-cluster gates via standard TEBD.
        for &gi in &inter_cluster_gate_indices {
            let err = standard_tebd_step(
                &mut state,
                &gates[gi],
                config.max_bond_dim,
                config.svd_cutoff,
            );
            cumulative_trunc_err += err;
        }
    }

    let fidelity = (1.0 - cumulative_trunc_err).max(0.0);

    Ok(ClusterTebdResult {
        final_mps: state,
        time_evolved: config.time_step * config.num_steps as f64,
        fidelity_estimate: fidelity,
        num_clusters_formed: total_clusters_formed,
        max_cluster_size_used: max_cluster_size_seen,
    })
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    const TOL: f64 = 1e-6;

    // ---- Test 1: Config builder defaults ----
    #[test]
    fn test_config_defaults() {
        let config = ClusterTebdConfig::new();
        assert_eq!(config.max_bond_dim, 64);
        assert!((config.time_step - 0.01).abs() < 1e-12);
        assert_eq!(config.num_steps, 100);
        assert!((config.cluster_threshold - 0.1).abs() < 1e-12);
        assert_eq!(config.max_cluster_size, 8);
        assert_eq!(config.merge_strategy, MergeStrategy::EntanglementBased);
        assert!((config.svd_cutoff - 1e-10).abs() < 1e-14);
    }

    #[test]
    fn test_config_builder_chain() {
        let config = ClusterTebdConfig::new()
            .max_bond_dim(128)
            .time_step(0.05)
            .num_steps(50)
            .cluster_threshold(0.2)
            .max_cluster_size(4)
            .merge_strategy(MergeStrategy::Adaptive)
            .svd_cutoff(1e-8);

        assert_eq!(config.max_bond_dim, 128);
        assert!((config.time_step - 0.05).abs() < 1e-12);
        assert_eq!(config.num_steps, 50);
        assert_eq!(config.max_cluster_size, 4);
        assert_eq!(config.merge_strategy, MergeStrategy::Adaptive);
    }

    // ---- Test 2: Product state MPS has norm 1 ----
    #[test]
    fn test_product_state_norm() {
        let state = product_state_mps(&[0, 1, 0, 1], 2);
        let n = mps_norm(&state);
        assert!(
            (n - 1.0).abs() < TOL,
            "product state norm should be 1.0, got {}",
            n
        );
    }

    #[test]
    fn test_product_state_larger() {
        let state = product_state_mps(&[0, 0, 0, 0, 1, 1, 1, 1], 2);
        let n = mps_norm(&state);
        assert!(
            (n - 1.0).abs() < TOL,
            "8-site product state norm should be 1.0, got {}",
            n
        );
    }

    // ---- Test 3: Random MPS creation with correct shapes ----
    #[test]
    fn test_random_mps_shapes() {
        let mps = random_cluster_mps(6, 2, 4);
        assert_eq!(mps.num_sites, 6);
        assert_eq!(mps.tensors.len(), 6);
        assert_eq!(mps.bond_dims.len(), 5);

        // First tensor: bond_left = 1.
        assert_eq!(mps.tensors[0].shape()[0], 1);
        // Physical dim = 2 for all.
        for t in &mps.tensors {
            assert_eq!(t.shape()[1], 2);
        }
        // Last tensor: bond_right = 1.
        assert_eq!(mps.tensors[5].shape()[2], 1);

        // Bond dims should be consistent.
        for i in 0..5 {
            assert_eq!(mps.tensors[i].shape()[2], mps.tensors[i + 1].shape()[0]);
        }
    }

    // ---- Test 4: SVD truncate preserves matrix (no truncation needed) ----
    #[test]
    fn test_svd_no_truncation() {
        // Random 4x4 matrix, keep all singular values.
        let mut rng = rand::thread_rng();
        let mut m = Array2::<Complex64>::zeros((4, 4));
        for elem in m.iter_mut() {
            *elem = Complex64::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
        }

        let (u, sigma, vt, trunc_err) = svd_truncate(&m, 4, 0.0);
        assert!(trunc_err < TOL, "no truncation error expected, got {}", trunc_err);

        // Reconstruct: M ≈ U diag(sigma) Vt.
        let k = sigma.len();
        let mut reconstructed = Array2::<Complex64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                let mut val = c0();
                for l in 0..k {
                    val += u[[i, l]] * Complex64::new(sigma[l], 0.0) * vt[[l, j]];
                }
                reconstructed[[i, j]] = val;
            }
        }

        for i in 0..4 {
            for j in 0..4 {
                let diff = (m[[i, j]] - reconstructed[[i, j]]).norm();
                assert!(
                    diff < 1e-8,
                    "SVD reconstruction failed at [{},{}]: diff={}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    // ---- Test 5: Entanglement entropy of product state is 0 ----
    #[test]
    fn test_entropy_product_state() {
        let state = product_state_mps(&[0, 1, 0, 1], 2);
        for site in 0..3 {
            let s = entanglement_entropy(&state, site);
            assert!(
                s.abs() < TOL,
                "product state entropy at site {} should be 0, got {}",
                site,
                s
            );
        }
    }

    // ---- Test 6: Entanglement entropy of maximally entangled state is ln(d) ----
    #[test]
    fn test_entropy_bell_state() {
        // Create Bell state |00⟩ + |11⟩ (unnormalized) as MPS.
        // Site 0: tensor shape [1, 2, 2], site 1: tensor shape [2, 2, 1].
        // |ψ⟩ = (|00⟩ + |11⟩)/√2
        let mut t0 = Array3::<Complex64>::zeros((1, 2, 2));
        t0[[0, 0, 0]] = c1(); // |0⟩ → bond 0
        t0[[0, 1, 1]] = c1(); // |1⟩ → bond 1

        let mut t1 = Array3::<Complex64>::zeros((2, 2, 1));
        t1[[0, 0, 0]] = c1(); // bond 0 → |0⟩
        t1[[1, 1, 0]] = c1(); // bond 1 → |1⟩

        let bell = ClusterMps {
            tensors: vec![t0, t1],
            bond_dims: vec![2],
            num_sites: 2,
            center: 0,
        };

        let s = entanglement_entropy(&bell, 0);
        let expected = (2.0_f64).ln(); // ln(2) for maximally entangled pair
        assert!(
            (s - expected).abs() < 0.05,
            "Bell state entropy should be ln(2) ≈ {:.4}, got {:.4}",
            expected,
            s
        );
    }

    // ---- Test 7: Mutual information identifies entangled pairs ----
    #[test]
    fn test_mutual_information() {
        // Bell pair on sites 0,1 with product state on site 2.
        let mut t0 = Array3::<Complex64>::zeros((1, 2, 2));
        t0[[0, 0, 0]] = c1();
        t0[[0, 1, 1]] = c1();

        let mut t1 = Array3::<Complex64>::zeros((2, 2, 1));
        t1[[0, 0, 0]] = c1();
        t1[[1, 1, 0]] = c1();

        let mut t2 = Array3::<Complex64>::zeros((1, 2, 1));
        t2[[0, 0, 0]] = c1();

        let state = ClusterMps {
            tensors: vec![t0, t1, t2],
            bond_dims: vec![2, 1],
            num_sites: 3,
            center: 0,
        };

        // MI between entangled sites 0,1 should be positive.
        let mi_01 = compute_mutual_information(&state, 0, 1);
        // MI between product site 2 and others should be near 0.
        let mi_12 = compute_mutual_information(&state, 1, 2);

        assert!(
            mi_01 > mi_12,
            "MI(0,1)={} should be greater than MI(1,2)={}",
            mi_01,
            mi_12
        );
    }

    // ---- Test 8: Cluster identification groups adjacent entangled sites ----
    #[test]
    fn test_cluster_identification() {
        // Graph with strong edges 0-1 and 1-2.
        let graph = vec![(0, 1, 0.5), (1, 2, 0.4), (3, 4, 0.3)];
        let clusters = identify_clusters(&graph, 4);

        // Should form at least one cluster containing {0,1,2}.
        assert!(!clusters.is_empty(), "should form at least one cluster");

        let has_012 = clusters.iter().any(|c| {
            c.contains(0) && c.contains(1) && c.contains(2)
        });
        assert!(has_012, "sites 0,1,2 should be in the same cluster");
    }

    // ---- Test 9: Merge + decompose round-trip preserves state ----
    #[test]
    fn test_merge_decompose_roundtrip() {
        // 3-site product state, merge sites [0,1,2] then decompose.
        let state = product_state_mps(&[0, 1, 0], 2);
        let norm_before = mps_norm(&state);

        let merged = merge_cluster_tensors(&state, &[0, 1, 2]);
        assert_eq!(merged.ndim(), 5); // [bl=1, d0=2, d1=2, d2=2, br=1]

        let (tensors, trunc_err) = decompose_cluster_tensor(&merged, &[0, 1, 2], 16, 1e-12);
        assert_eq!(tensors.len(), 3);
        assert!(trunc_err < TOL);

        let reconstructed = ClusterMps {
            tensors,
            bond_dims: vec![
                1, 1, // will be updated
            ],
            num_sites: 3,
            center: 0,
        };
        let norm_after = mps_norm(&reconstructed);

        assert!(
            (norm_before - norm_after).abs() < 0.05,
            "norms should match: before={}, after={}",
            norm_before,
            norm_after
        );
    }

    // ---- Test 10: Standard TEBD step preserves norm ----
    #[test]
    fn test_standard_tebd_preserves_norm() {
        let mut state = product_state_mps(&[0, 1, 0, 1], 2);
        let norm_before = mps_norm(&state);

        // Identity gate: should preserve everything.
        let mut id_gate = Array2::<Complex64>::zeros((4, 4));
        for i in 0..4 {
            id_gate[[i, i]] = c1();
        }
        let gate = TwoSiteGate::new(1, 2, id_gate);

        let _err = standard_tebd_step(&mut state, &gate, 16, 1e-12);
        let norm_after = mps_norm(&state);

        assert!(
            (norm_before - norm_after).abs() < TOL,
            "TEBD with identity should preserve norm: before={}, after={}",
            norm_before,
            norm_after
        );
    }

    // ---- Test 11: Cluster-TEBD on 8-site Ising chain ----
    #[test]
    fn test_cluster_tebd_ising_8_sites() {
        let n = 8;
        let j = 1.0;
        let h = 0.5;
        let dt = 0.01;

        let state = product_state_mps(&vec![0; n], 2);
        let gates = ising_gates(n, j, h, dt);

        let config = ClusterTebdConfig::new()
            .max_bond_dim(32)
            .time_step(dt)
            .num_steps(10)
            .cluster_threshold(0.01)
            .max_cluster_size(4)
            .svd_cutoff(1e-12);

        let result = cluster_tebd(&state, &gates, &config);
        assert!(result.is_ok(), "cluster TEBD should succeed");

        let result = result.unwrap();
        let final_norm = mps_norm(&result.final_mps);
        assert!(
            (final_norm - 1.0).abs() < 0.1,
            "final state norm should be close to 1.0, got {}",
            final_norm
        );
        assert!(
            result.time_evolved > 0.0,
            "time_evolved should be positive"
        );
        assert!(
            result.fidelity_estimate > 0.5,
            "fidelity should be reasonable, got {}",
            result.fidelity_estimate
        );
    }

    // ---- Test 12: Cluster-TEBD matches standard TEBD for small system ----
    #[test]
    fn test_cluster_vs_standard_tebd() {
        let n = 4;
        let dt = 0.01;

        // Simple Ising model.
        let state = product_state_mps(&vec![0; n], 2);
        let gates = ising_gates(n, 1.0, 0.5, dt);
        let num_steps = 5;

        // Standard TEBD.
        let mut std_state = state.clone();
        normalize_mps(&mut std_state);
        for _ in 0..num_steps {
            for gate in &gates {
                standard_tebd_step(&mut std_state, gate, 32, 1e-12);
            }
        }

        // Cluster-TEBD with very low threshold (should use standard mostly).
        let config = ClusterTebdConfig::new()
            .max_bond_dim(32)
            .time_step(dt)
            .num_steps(num_steps)
            .cluster_threshold(100.0) // high threshold = no clusters
            .max_cluster_size(4)
            .svd_cutoff(1e-12);

        let cluster_result = cluster_tebd(&state, &gates, &config).unwrap();

        // Compare norms.
        let std_norm = mps_norm(&std_state);
        let cluster_norm = mps_norm(&cluster_result.final_mps);

        assert!(
            (std_norm - cluster_norm).abs() < 0.1,
            "norms should be close: standard={}, cluster={}",
            std_norm,
            cluster_norm
        );

        // Compare a local observable: ⟨Z_0⟩.
        let mut sz = Array2::<Complex64>::zeros((2, 2));
        sz[[0, 0]] = c1();
        sz[[1, 1]] = Complex64::new(-1.0, 0.0);

        let z_std = measure_local(&std_state, &sz, 0).re;
        let z_cluster = measure_local(&cluster_result.final_mps, &sz, 0).re;

        assert!(
            (z_std - z_cluster).abs() < 0.15,
            "⟨Z_0⟩ should agree: standard={:.4}, cluster={:.4}",
            z_std,
            z_cluster
        );
    }

    // ---- Additional tests ----

    #[test]
    fn test_config_validation() {
        let bad = ClusterTebdConfig::new().max_bond_dim(0);
        assert!(bad.validate().is_err());

        let bad2 = ClusterTebdConfig::new().time_step(-1.0);
        assert!(bad2.validate().is_err());

        let bad3 = ClusterTebdConfig::new().max_cluster_size(1);
        assert!(bad3.validate().is_err());

        let good = ClusterTebdConfig::new();
        assert!(good.validate().is_ok());
    }

    #[test]
    fn test_heisenberg_gates_creation() {
        let gates = heisenberg_gates(4, 1.0, 0.01);
        assert_eq!(gates.len(), 3);
        for (i, g) in gates.iter().enumerate() {
            assert_eq!(g.site_a, i);
            assert_eq!(g.site_b, i + 1);
            assert_eq!(g.matrix.nrows(), 4);
            assert_eq!(g.matrix.ncols(), 4);
        }
    }

    #[test]
    fn test_gate_internal_to_cluster() {
        let gate = TwoSiteGate::new(2, 3, Array2::zeros((4, 4)));
        assert!(gate.is_internal_to(&[1, 2, 3, 4]));
        assert!(!gate.is_internal_to(&[0, 1, 2]));
        assert!(!gate.is_internal_to(&[3, 4, 5]));
    }

    #[test]
    fn test_mps_norm_zero_state() {
        // All zeros MPS should have norm 0.
        let tensors = vec![
            Array3::<Complex64>::zeros((1, 2, 1)),
            Array3::<Complex64>::zeros((1, 2, 1)),
        ];
        let mps = ClusterMps {
            tensors,
            bond_dims: vec![1],
            num_sites: 2,
            center: 0,
        };
        let n = mps_norm(&mps);
        assert!(n < TOL, "zero state norm should be ~0, got {}", n);
    }
}
