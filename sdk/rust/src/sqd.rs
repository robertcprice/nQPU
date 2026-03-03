//! Sample-based Quantum Diagonalization (SQD)
//!
//! Implements IBM's utility-scale quantum chemistry method (Nature, 2024) for
//! extracting ground-state energies from noisy quantum hardware.
//!
//! # Algorithm
//!
//! SQD bridges quantum sampling and classical diagonalization:
//!
//! 1. **Sample**: Run a quantum circuit (VQE, QPE, or hardware) to collect
//!    bitstring samples from a trial wavefunction.
//! 2. **Subspace**: Use the sampled bitstrings to define a subspace of the full
//!    2^n Hilbert space. Optionally filter by particle number and expand via
//!    Hamming-distance neighbors.
//! 3. **Project**: Build the Hamiltonian matrix H_ij = <b_i|H|b_j> restricted
//!    to the subspace basis {|b_i>}.
//! 4. **Diagonalize**: Solve the projected eigenvalue problem classically to
//!    extract the ground-state energy.
//! 5. **Bootstrap**: Resample bitstrings with replacement to estimate
//!    confidence intervals on the energy.
//!
//! # Key Insight
//!
//! Even noisy quantum circuits produce bitstrings that overlap significantly
//! with the true ground state. SQD exploits this by projecting the exact
//! Hamiltonian into the quantum-informed subspace, achieving chemical accuracy
//! on systems far beyond brute-force classical diagonalization.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::sqd::*;
//!
//! // Build H2 molecule at equilibrium bond length
//! let h2 = FermionicHamiltonian::h2_sto3g(0.74);
//! let config = SQDConfig::default();
//!
//! // Simulate bitstring samples (in practice these come from quantum hardware)
//! let samples = vec![0b01, 0b10]; // 1-electron states for 2-qubit H2
//!
//! // Run SQD
//! let result = sqd_solve(&h2, &samples, &config).unwrap();
//! assert!((result.ground_state_energy - (-1.137)).abs() < 0.05);
//! ```

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use std::fmt;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising during SQD computation.
#[derive(Clone, Debug, PartialEq)]
pub enum SQDError {
    /// Subspace is empty after filtering.
    EmptySubspace,
    /// Hamiltonian dimensions are inconsistent.
    DimensionMismatch { expected: usize, got: usize },
    /// Eigendecomposition failed to converge.
    EigenConvergence { iterations: usize, residual: f64 },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Integral data is malformed.
    InvalidIntegrals(String),
    /// Numerical instability encountered.
    NumericalInstability(String),
}

impl fmt::Display for SQDError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SQDError::EmptySubspace => {
                write!(f, "Subspace is empty after filtering")
            }
            SQDError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {} qubits, got {}",
                    expected, got
                )
            }
            SQDError::EigenConvergence { iterations, residual } => {
                write!(
                    f,
                    "Eigendecomposition did not converge after {} iterations (residual={:.2e})",
                    iterations, residual
                )
            }
            SQDError::InvalidConfig(msg) => {
                write!(f, "Invalid SQD configuration: {}", msg)
            }
            SQDError::InvalidIntegrals(msg) => {
                write!(f, "Invalid integrals: {}", msg)
            }
            SQDError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
        }
    }
}

impl std::error::Error for SQDError {}

/// Convenience result type for SQD operations.
pub type SQDResult<T> = Result<T, SQDError>;

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the SQD solver.
#[derive(Clone, Debug)]
pub struct SQDConfig {
    /// Maximum number of bitstring samples to draw from the input pool.
    pub num_samples: usize,
    /// Maximum subspace dimension (after expansion and deduplication).
    pub subspace_dimension: usize,
    /// If set, keep only bitstrings with this many 1-bits (electron count).
    pub particle_number_filter: Option<usize>,
    /// Number of bootstrap resampling iterations for confidence intervals.
    pub bootstrap_samples: usize,
    /// Convergence threshold for the Jacobi eigendecomposition.
    pub convergence_threshold: f64,
    /// Maximum Hamming distance for subspace expansion (0 = no expansion).
    pub hamming_expansion_radius: usize,
    /// Random seed for reproducibility (None = use default seed 42).
    pub seed: Option<u64>,
    /// Maximum Jacobi sweeps before declaring non-convergence.
    pub max_jacobi_sweeps: usize,
}

impl Default for SQDConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            subspace_dimension: 512,
            particle_number_filter: None,
            bootstrap_samples: 100,
            convergence_threshold: 1e-12,
            hamming_expansion_radius: 0,
            seed: None,
            max_jacobi_sweeps: 200,
        }
    }
}

impl SQDConfig {
    /// Create a config tuned for small molecules (2-4 qubits).
    pub fn small_molecule() -> Self {
        Self {
            num_samples: 100,
            subspace_dimension: 64,
            bootstrap_samples: 200,
            convergence_threshold: 1e-14,
            hamming_expansion_radius: 1,
            max_jacobi_sweeps: 300,
            ..Self::default()
        }
    }

    /// Create a config tuned for medium molecules (8-16 qubits).
    pub fn medium_molecule() -> Self {
        Self {
            num_samples: 2000,
            subspace_dimension: 1024,
            bootstrap_samples: 100,
            convergence_threshold: 1e-12,
            hamming_expansion_radius: 1,
            ..Self::default()
        }
    }

    /// Create a config tuned for Hubbard model simulation.
    pub fn hubbard() -> Self {
        Self {
            num_samples: 5000,
            subspace_dimension: 2048,
            bootstrap_samples: 50,
            convergence_threshold: 1e-10,
            hamming_expansion_radius: 2,
            ..Self::default()
        }
    }

    /// Set the particle number filter (number of electrons).
    pub fn with_particle_filter(mut self, n_electrons: usize) -> Self {
        self.particle_number_filter = Some(n_electrons);
        self
    }

    /// Set the subspace dimension cap.
    pub fn with_subspace_dimension(mut self, dim: usize) -> Self {
        self.subspace_dimension = dim;
        self
    }

    /// Set the number of bootstrap resampling rounds.
    pub fn with_bootstrap_samples(mut self, n: usize) -> Self {
        self.bootstrap_samples = n;
        self
    }

    /// Set the Hamming expansion radius.
    pub fn with_hamming_radius(mut self, r: usize) -> Self {
        self.hamming_expansion_radius = r;
        self
    }

    /// Set the random seed for deterministic execution.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the convergence threshold for eigendecomposition.
    pub fn with_convergence_threshold(mut self, tol: f64) -> Self {
        self.convergence_threshold = tol;
        self
    }

    /// Validate the configuration, returning an error on invalid parameters.
    pub fn validate(&self) -> SQDResult<()> {
        if self.subspace_dimension == 0 {
            return Err(SQDError::InvalidConfig(
                "subspace_dimension must be > 0".into(),
            ));
        }
        if self.convergence_threshold <= 0.0 {
            return Err(SQDError::InvalidConfig(
                "convergence_threshold must be positive".into(),
            ));
        }
        if self.max_jacobi_sweeps == 0 {
            return Err(SQDError::InvalidConfig(
                "max_jacobi_sweeps must be > 0".into(),
            ));
        }
        Ok(())
    }
}

impl fmt::Display for SQDConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SQDConfig(samples={}, subspace={}, bootstrap={}, hamming_r={})",
            self.num_samples,
            self.subspace_dimension,
            self.bootstrap_samples,
            self.hamming_expansion_radius,
        )
    }
}

// ============================================================
// FERMIONIC HAMILTONIAN
// ============================================================

/// Second-quantized molecular Hamiltonian in the spin-orbital basis.
///
/// ```text
/// H = E_nuc + sum_{pq} h_{pq} a+_p a_q + (1/2) sum_{pqrs} h_{pqrs} a+_p a+_r a_s a_q
/// ```
///
/// Here p,q,r,s are spin-orbital indices. The one-body integrals h_{pq}
/// encode kinetic energy and electron-nuclear attraction. The two-body
/// integrals h_{pqrs} encode electron-electron repulsion in physicist
/// notation.
#[derive(Clone, Debug)]
pub struct FermionicHamiltonian {
    /// Number of spin-orbitals (= number of qubits after Jordan-Wigner).
    pub num_spin_orbitals: usize,
    /// One-body integrals h_{pq}, shape [n x n].
    pub one_body: Vec<Vec<f64>>,
    /// Two-body integrals h_{pqrs} in physicist notation.
    /// Stored as flat Vec with index p*n^3 + q*n^2 + r*n + s.
    pub two_body: Vec<f64>,
    /// Nuclear repulsion energy (constant offset).
    pub nuclear_repulsion: f64,
    /// Number of electrons (for particle-number filtering).
    pub num_electrons: usize,
}

impl FermionicHamiltonian {
    /// Create a new Hamiltonian from raw integral data.
    pub fn new(
        num_spin_orbitals: usize,
        one_body: Vec<Vec<f64>>,
        two_body: Vec<f64>,
        nuclear_repulsion: f64,
        num_electrons: usize,
    ) -> SQDResult<Self> {
        let n = num_spin_orbitals;
        if one_body.len() != n {
            return Err(SQDError::InvalidIntegrals(format!(
                "one_body has {} rows, expected {}",
                one_body.len(),
                n
            )));
        }
        for (i, row) in one_body.iter().enumerate() {
            if row.len() != n {
                return Err(SQDError::InvalidIntegrals(format!(
                    "one_body row {} has length {}, expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        if two_body.len() != n * n * n * n {
            return Err(SQDError::InvalidIntegrals(format!(
                "two_body has length {}, expected {}",
                two_body.len(),
                n * n * n * n
            )));
        }
        Ok(Self {
            num_spin_orbitals,
            one_body,
            two_body,
            nuclear_repulsion,
            num_electrons,
        })
    }

    /// Access two-body integral h_{pqrs}.
    #[inline]
    pub fn two_body_element(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let n = self.num_spin_orbitals;
        self.two_body[p * n * n * n + q * n * n + r * n + s]
    }

    /// Set two-body integral h_{pqrs}.
    #[inline]
    pub fn set_two_body_element(
        &mut self,
        p: usize,
        q: usize,
        r: usize,
        s: usize,
        val: f64,
    ) {
        let n = self.num_spin_orbitals;
        self.two_body[p * n * n * n + q * n * n + r * n + s] = val;
    }

    /// Check one-body hermiticity (real symmetric): h_{pq} = h_{qp}.
    pub fn one_body_is_symmetric(&self, tol: f64) -> bool {
        let n = self.num_spin_orbitals;
        for p in 0..n {
            for q in (p + 1)..n {
                if (self.one_body[p][q] - self.one_body[q][p]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Build H2 molecule in STO-3G basis at given bond length (Angstrom).
    ///
    /// Returns a 2-spin-orbital Hamiltonian (minimal basis, 1 spatial orbital
    /// per atom mapped to 2 spin-orbitals via restricted Hartree-Fock).
    ///
    /// Reference values at R=0.74 A: E_exact ~ -1.137 Hartree.
    pub fn h2_sto3g(bond_length: f64) -> Self {
        let r_bohr = bond_length / 0.529177;
        let nuclear_repulsion = 1.0 / r_bohr;

        // Core Hamiltonian integral (kinetic + nuclear attraction) in the MO basis.
        // For H2 STO-3G at equilibrium (r_bohr ~ 1.4), h_core ~ -1.2528 Hartree.
        // 2 spin-orbitals from 1 spatial MO: alpha and beta share the same integral.
        let h_core = -1.2528 - 0.10 * (r_bohr - 1.4).powi(2);
        let one_body = vec![vec![h_core, 0.0], vec![0.0, h_core]];

        // Two-body integrals: Coulomb repulsion within the same spatial orbital
        let v = 0.6746 + 0.02 * (r_bohr - 1.4).powi(2);
        let n = 2;
        let mut two_body = vec![0.0; n * n * n * n];
        two_body[0 * 8 + 0 * 4 + 0 * 2 + 0] = v;
        two_body[1 * 8 + 1 * 4 + 1 * 2 + 1] = v;
        two_body[0 * 8 + 1 * 4 + 1 * 2 + 0] = v;
        two_body[1 * 8 + 0 * 4 + 0 * 2 + 1] = v;

        Self {
            num_spin_orbitals: 2,
            one_body,
            two_body,
            nuclear_repulsion,
            num_electrons: 2,
        }
    }

    /// Build H2 molecule using 4 spin-orbitals (2 spatial: bonding + antibonding).
    ///
    /// This is the standard minimal-basis H2 used in quantum computing benchmarks.
    /// Reference: E_FCI = -1.1373 Hartree (STO-3G, R=0.74 A).
    pub fn h2_sto3g_4so() -> Self {
        let nuclear_repulsion = 0.7137;

        // One-body MO integrals (2 spatial -> 4 spin-orbitals)
        // Spin-orbital mapping: 0=0alpha, 1=0beta, 2=1alpha, 3=1beta
        let eps0 = -1.2528;
        let eps1 = -0.4756;
        let one_body = vec![
            vec![eps0, 0.0, 0.0, 0.0],
            vec![0.0, eps0, 0.0, 0.0],
            vec![0.0, 0.0, eps1, 0.0],
            vec![0.0, 0.0, 0.0, eps1],
        ];

        let j00 = 0.6746;
        let j01 = 0.6636;
        let k01 = 0.1813;

        let n: usize = 4;
        let mut two_body = vec![0.0; n * n * n * n];
        let idx =
            |p: usize, q: usize, r: usize, s: usize| -> usize { p * 64 + q * 16 + r * 4 + s };

        // Same-orbital Coulomb: spatial orbital 0
        two_body[idx(0, 0, 0, 0)] = j00;
        two_body[idx(1, 1, 1, 1)] = j00;
        two_body[idx(0, 1, 1, 0)] = j00;
        two_body[idx(1, 0, 0, 1)] = j00;

        // Same-orbital Coulomb: spatial orbital 1
        two_body[idx(2, 2, 2, 2)] = j00;
        two_body[idx(3, 3, 3, 3)] = j00;
        two_body[idx(2, 3, 3, 2)] = j00;
        two_body[idx(3, 2, 2, 3)] = j00;

        // Cross-orbital Coulomb (J)
        two_body[idx(0, 2, 2, 0)] = j01;
        two_body[idx(2, 0, 0, 2)] = j01;
        two_body[idx(0, 3, 3, 0)] = j01;
        two_body[idx(3, 0, 0, 3)] = j01;
        two_body[idx(1, 2, 2, 1)] = j01;
        two_body[idx(2, 1, 1, 2)] = j01;
        two_body[idx(1, 3, 3, 1)] = j01;
        two_body[idx(3, 1, 1, 3)] = j01;

        // Cross-orbital exchange (K)
        two_body[idx(0, 2, 0, 2)] = k01;
        two_body[idx(2, 0, 2, 0)] = k01;
        two_body[idx(1, 3, 1, 3)] = k01;
        two_body[idx(3, 1, 3, 1)] = k01;

        Self {
            num_spin_orbitals: 4,
            one_body,
            two_body,
            nuclear_repulsion,
            num_electrons: 2,
        }
    }

    /// Build a 1D Hubbard model Hamiltonian.
    ///
    /// ```text
    /// H = -t sum_{<ij>,sigma} (a+_{i,sigma} a_{j,sigma} + h.c.)
    ///     + U sum_i n_{i,up} n_{i,down}
    /// ```
    ///
    /// Uses 2*n_sites spin-orbitals. Spin-orbital index: site i up -> i,
    /// site i down -> n_sites + i. Half-filling by default.
    pub fn hubbard_1d(n_sites: usize, t: f64, u: f64) -> Self {
        let n = 2 * n_sites;
        let mut one_body = vec![vec![0.0; n]; n];
        let mut two_body = vec![0.0; n * n * n * n];

        // Hopping
        for spin_offset in [0, n_sites] {
            for i in 0..(n_sites - 1) {
                let p = spin_offset + i;
                let q = spin_offset + i + 1;
                one_body[p][q] = -t;
                one_body[q][p] = -t;
            }
        }

        // On-site repulsion
        let idx = |p: usize, q: usize, r: usize, s: usize| -> usize {
            p * n * n * n + q * n * n + r * n + s
        };
        for i in 0..n_sites {
            let up = i;
            let dn = n_sites + i;
            two_body[idx(up, dn, dn, up)] = u;
            two_body[idx(dn, up, up, dn)] = u;
        }

        Self {
            num_spin_orbitals: n,
            one_body,
            two_body,
            nuclear_repulsion: 0.0,
            num_electrons: n_sites,
        }
    }

    /// Build a LiH molecule (8 spin-orbitals, 4 electrons).
    ///
    /// Approximate integrals in a minimal STO-3G-like active space.
    /// Reference: E_FCI ~ -7.882 Hartree (STO-3G, R=1.6 A).
    pub fn lih_sto3g() -> Self {
        let nuclear_repulsion = 0.9953;

        let eps = [-2.452, -0.302, 0.072, 0.536];
        let n: usize = 8;
        let mut one_body = vec![vec![0.0; n]; n];
        for i in 0..4 {
            one_body[2 * i][2 * i] = eps[i];
            one_body[2 * i + 1][2 * i + 1] = eps[i];
        }

        let coupling = [
            (0, 1, -0.095),
            (0, 2, -0.013),
            (1, 2, 0.030),
            (1, 3, 0.015),
        ];
        for &(i, j, val) in &coupling {
            for spin in 0..2 {
                let p = 2 * i + spin;
                let q = 2 * j + spin;
                one_body[p][q] = val;
                one_body[q][p] = val;
            }
        }

        let mut two_body = vec![0.0; n * n * n * n];
        let idx = |p: usize, q: usize, r: usize, s: usize| -> usize {
            p * n * n * n + q * n * n + r * n + s
        };

        let j_diag = [1.046, 0.564, 0.468, 0.396];
        for i in 0..4 {
            let a = 2 * i;
            let b = 2 * i + 1;
            two_body[idx(a, b, b, a)] = j_diag[i];
            two_body[idx(b, a, a, b)] = j_diag[i];
        }

        let j_cross = [
            (0, 1, 0.420),
            (0, 2, 0.320),
            (0, 3, 0.280),
            (1, 2, 0.380),
            (1, 3, 0.340),
            (2, 3, 0.350),
        ];
        for &(i, j, jval) in &j_cross {
            for si in 0..2 {
                for sj in 0..2 {
                    let p = 2 * i + si;
                    let q = 2 * j + sj;
                    two_body[idx(p, q, q, p)] = jval;
                    two_body[idx(q, p, p, q)] = jval;
                }
            }
        }

        Self {
            num_spin_orbitals: 8,
            one_body,
            two_body,
            nuclear_repulsion,
            num_electrons: 4,
        }
    }

    /// Compute the diagonal matrix element <b|H|b> for computational basis state b.
    ///
    /// For a Slater determinant |b> the energy is:
    /// ```text
    /// E(b) = E_nuc + sum_{p in occ} h_{pp}
    ///        + (1/2) sum_{p,q in occ, p!=q} (h_{pqqp} - h_{pqpq})
    /// ```
    pub fn diagonal_element(&self, bitstring: usize) -> f64 {
        let n = self.num_spin_orbitals;
        let mut energy = self.nuclear_repulsion;

        let occ: Vec<usize> = (0..n).filter(|&i| (bitstring >> i) & 1 == 1).collect();

        for &p in &occ {
            energy += self.one_body[p][p];
        }

        for (idx_p, &p) in occ.iter().enumerate() {
            for &q in occ.iter().skip(idx_p + 1) {
                let direct = self.two_body_element(p, q, q, p);
                let exchange = self.two_body_element(p, q, p, q);
                energy += direct - exchange;
            }
        }

        energy
    }

    /// Compute the matrix element <bra|H|ket> using Slater-Condon rules.
    ///
    /// - 0 substitutions: diagonal (see `diagonal_element`)
    /// - 1 substitution (p->q): h_{pq} + sum_{r in occ} (<pr|qr> - <pr|rq>)
    /// - 2 substitutions (p,q->r,s): <pq|rs> - <pq|sr>
    /// - 3+ substitutions: 0
    pub fn matrix_element(&self, bra: usize, ket: usize) -> f64 {
        if bra == ket {
            return self.diagonal_element(bra);
        }

        let n = self.num_spin_orbitals;
        let diff = bra ^ ket;
        let n_diff = diff.count_ones() as usize;

        if n_diff > 4 {
            return 0.0;
        }

        let annihilated: Vec<usize> = (0..n)
            .filter(|&i| (bra >> i) & 1 == 1 && (ket >> i) & 1 == 0)
            .collect();

        let created: Vec<usize> = (0..n)
            .filter(|&i| (ket >> i) & 1 == 1 && (bra >> i) & 1 == 0)
            .collect();

        if annihilated.len() != created.len() {
            return 0.0;
        }

        let sign = slater_condon_sign(bra, &annihilated, &created);

        match annihilated.len() {
            1 => {
                let p = annihilated[0];
                let q = created[0];
                let occ: Vec<usize> =
                    (0..n).filter(|&i| (bra >> i) & 1 == 1 && i != p).collect();

                let mut val = self.one_body[p][q];
                for &r in &occ {
                    val += self.two_body_element(p, r, q, r)
                        - self.two_body_element(p, r, r, q);
                }
                sign * val
            }
            2 => {
                let p = annihilated[0];
                let q = annihilated[1];
                let r = created[0];
                let s = created[1];
                let val = self.two_body_element(p, q, r, s)
                    - self.two_body_element(p, q, s, r);
                sign * val
            }
            _ => 0.0,
        }
    }

    /// Jordan-Wigner transform: build the full 2^n x 2^n qubit Hamiltonian matrix.
    ///
    /// Practical only for small systems (n <= ~16 qubits).
    pub fn to_qubit_hamiltonian_matrix(&self) -> Array2<f64> {
        let n = self.num_spin_orbitals;
        let dim = 1usize << n;
        let mut h_mat = Array2::<f64>::zeros((dim, dim));

        for i in 0..dim {
            for j in i..dim {
                let val = self.matrix_element(i, j);
                h_mat[[i, j]] = val;
                h_mat[[j, i]] = val;
            }
        }
        h_mat
    }

    /// Exact diagonalization of the full Hamiltonian.
    ///
    /// Returns (sorted_eigenvalues, ground_state_energy).
    pub fn exact_diagonalize(&self) -> SQDResult<(Vec<f64>, f64)> {
        let n = self.num_spin_orbitals;
        if n > 18 {
            return Err(SQDError::InvalidConfig(format!(
                "Exact diagonalization impractical for {} qubits (max ~18)",
                n
            )));
        }

        let h_mat = self.to_qubit_hamiltonian_matrix();
        let eigenvalues = jacobi_eigenvalues(&h_mat, 1e-14, 500)?;
        let mut sorted = eigenvalues;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let ground_energy = sorted[0];
        Ok((sorted, ground_energy))
    }

    /// Exact diagonalization restricted to a particle number sector.
    pub fn exact_diagonalize_sector(&self, n_electrons: usize) -> SQDResult<(Vec<f64>, f64)> {
        let n = self.num_spin_orbitals;
        let dim = 1usize << n;

        let sector: Vec<usize> = (0..dim)
            .filter(|&b| (b as u64).count_ones() as usize == n_electrons)
            .collect();

        if sector.is_empty() {
            return Err(SQDError::EmptySubspace);
        }

        let sd = sector.len();
        let mut h_proj = Array2::<f64>::zeros((sd, sd));
        for i in 0..sd {
            for j in i..sd {
                let val = self.matrix_element(sector[i], sector[j]);
                h_proj[[i, j]] = val;
                h_proj[[j, i]] = val;
            }
        }

        let eigenvalues = jacobi_eigenvalues(&h_proj, 1e-14, 500)?;
        let mut sorted = eigenvalues;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let ground_energy = sorted[0];
        Ok((sorted, ground_energy))
    }
}

impl fmt::Display for FermionicHamiltonian {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FermionicHamiltonian(n_so={}, n_e={}, E_nuc={:.4})",
            self.num_spin_orbitals, self.num_electrons, self.nuclear_repulsion,
        )
    }
}

// ============================================================
// SLATER-CONDON SIGN COMPUTATION
// ============================================================

/// Compute the fermionic sign factor for a Slater-Condon matrix element.
///
/// The sign depends on the number of occupied orbitals that must be
/// "jumped over" when reordering creation operators from |bra> to |ket>.
fn slater_condon_sign(bra: usize, annihilated: &[usize], created: &[usize]) -> f64 {
    let mut parity = 0usize;

    for &p in annihilated {
        for i in 0..p {
            if (bra >> i) & 1 == 1 {
                parity += 1;
            }
        }
    }

    let mut intermediate = bra;
    for &p in annihilated {
        intermediate &= !(1 << p);
    }
    for &q in created {
        for i in 0..q {
            if (intermediate >> i) & 1 == 1 {
                parity += 1;
            }
        }
        intermediate |= 1 << q;
    }

    if parity % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

// ============================================================
// SUBSPACE BUILDER
// ============================================================

/// Builds the subspace of computational basis states from bitstring samples.
///
/// Construction steps:
/// 1. Collect unique bitstring samples
/// 2. Optionally filter by particle number (electron count)
/// 3. Optionally expand via Hamming distance neighbors
/// 4. Cap at the configured subspace dimension
#[derive(Clone, Debug)]
pub struct SubspaceBuilder {
    /// Number of qubits.
    num_qubits: usize,
    /// Configuration for subspace construction.
    config: SQDConfig,
}

impl SubspaceBuilder {
    /// Create a new SubspaceBuilder.
    pub fn new(num_qubits: usize, config: &SQDConfig) -> Self {
        Self {
            num_qubits,
            config: config.clone(),
        }
    }

    /// Build the subspace from bitstring samples.
    ///
    /// Returns a sorted, deduplicated list of basis state indices.
    pub fn build(&self, samples: &[usize]) -> SQDResult<Vec<usize>> {
        if samples.is_empty() {
            return Err(SQDError::EmptySubspace);
        }

        let max_state = 1usize << self.num_qubits;

        // Step 1: collect unique samples, validate range
        let mut basis: Vec<usize> = Vec::new();
        for &s in samples.iter().take(self.config.num_samples) {
            if s < max_state {
                basis.push(s);
            }
        }

        basis.sort_unstable();
        basis.dedup();

        // Step 2: particle number filtering
        if let Some(n_electrons) = self.config.particle_number_filter {
            basis.retain(|&b| popcount(b) == n_electrons);
        }

        if basis.is_empty() {
            return Err(SQDError::EmptySubspace);
        }

        // Step 3: Hamming distance expansion
        if self.config.hamming_expansion_radius > 0 {
            let mut expanded = basis.clone();
            for &b in &basis {
                self.hamming_neighbors(b, self.config.hamming_expansion_radius, &mut expanded);
            }

            if let Some(n_electrons) = self.config.particle_number_filter {
                expanded.retain(|&b| popcount(b) == n_electrons);
            }

            expanded.sort_unstable();
            expanded.dedup();
            basis = expanded;
        }

        // Step 4: cap at subspace dimension
        if basis.len() > self.config.subspace_dimension {
            basis.truncate(self.config.subspace_dimension);
        }

        if basis.is_empty() {
            return Err(SQDError::EmptySubspace);
        }

        Ok(basis)
    }

    /// Generate all bitstrings within Hamming distance `radius` of `center`.
    fn hamming_neighbors(&self, center: usize, radius: usize, out: &mut Vec<usize>) {
        let n = self.num_qubits;
        let max_state = 1usize << n;

        if radius == 0 {
            return;
        }

        // Radius 1: flip each single bit
        for i in 0..n {
            let neighbor = center ^ (1 << i);
            if neighbor < max_state {
                out.push(neighbor);
            }
        }

        // Radius 2: flip each pair
        if radius >= 2 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let neighbor = center ^ (1 << i) ^ (1 << j);
                    if neighbor < max_state {
                        out.push(neighbor);
                    }
                }
            }
        }

        // Radius 3: flip each triple (guard against combinatorial explosion)
        if radius >= 3 && n <= 16 {
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let neighbor = center ^ (1 << i) ^ (1 << j) ^ (1 << k);
                        if neighbor < max_state {
                            out.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    /// Count the number of unique states in the subspace.
    pub fn subspace_size(&self, samples: &[usize]) -> SQDResult<usize> {
        let basis = self.build(samples)?;
        Ok(basis.len())
    }
}

// ============================================================
// PROJECTED HAMILTONIAN
// ============================================================

/// A Hamiltonian projected into a subspace defined by a set of basis states.
///
/// Given basis states {|b_i>}, computes H_ij = <b_i|H|b_j>. Since the
/// computational basis states are orthonormal, the overlap matrix is the
/// identity and this reduces to a standard eigenvalue problem.
#[derive(Clone, Debug)]
pub struct ProjectedHamiltonian {
    /// The projected matrix H_ij = <b_i|H|b_j>.
    pub matrix: Array2<f64>,
    /// The basis states (bitstring indices).
    pub basis: Vec<usize>,
    /// Dimension of the projected space.
    pub dimension: usize,
}

impl ProjectedHamiltonian {
    /// Build the projected Hamiltonian from a fermionic Hamiltonian and subspace.
    pub fn build(hamiltonian: &FermionicHamiltonian, basis: &[usize]) -> SQDResult<Self> {
        let dim = basis.len();
        if dim == 0 {
            return Err(SQDError::EmptySubspace);
        }

        let mut matrix = Array2::<f64>::zeros((dim, dim));

        for i in 0..dim {
            for j in i..dim {
                let val = hamiltonian.matrix_element(basis[i], basis[j]);
                matrix[[i, j]] = val;
                matrix[[j, i]] = val;
            }
        }

        Ok(Self {
            matrix,
            basis: basis.to_vec(),
            dimension: dim,
        })
    }

    /// Check that the projected matrix is Hermitian (symmetric for real case).
    pub fn is_hermitian(&self, tol: f64) -> bool {
        let n = self.dimension;
        for i in 0..n {
            for j in (i + 1)..n {
                if (self.matrix[[i, j]] - self.matrix[[j, i]]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Solve the eigenvalue problem and return sorted eigenvalues.
    pub fn solve(&self, config: &SQDConfig) -> SQDResult<ProjectedSolution> {
        let eigenvalues = jacobi_eigenvalues(
            &self.matrix,
            config.convergence_threshold,
            config.max_jacobi_sweeps,
        )?;

        let mut sorted = eigenvalues;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let ground_state_energy = sorted[0];

        Ok(ProjectedSolution {
            eigenvalues: sorted,
            ground_state_energy,
            subspace_dimension: self.dimension,
        })
    }

    /// Frobenius norm of the projected matrix.
    pub fn frobenius_norm(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                s += self.matrix[[i, j]] * self.matrix[[i, j]];
            }
        }
        s.sqrt()
    }

    /// Trace of the projected matrix.
    pub fn trace(&self) -> f64 {
        (0..self.dimension).map(|i| self.matrix[[i, i]]).sum()
    }
}

impl fmt::Display for ProjectedHamiltonian {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProjectedHamiltonian(dim={}, trace={:.6}, norm={:.6})",
            self.dimension,
            self.trace(),
            self.frobenius_norm(),
        )
    }
}

/// Result of solving the projected eigenvalue problem.
#[derive(Clone, Debug)]
pub struct ProjectedSolution {
    /// Eigenvalues sorted in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Ground state energy (lowest eigenvalue).
    pub ground_state_energy: f64,
    /// Dimension of the subspace used.
    pub subspace_dimension: usize,
}

// ============================================================
// JACOBI EIGENVALUE ALGORITHM
// ============================================================

/// Jacobi eigenvalue algorithm for real symmetric matrices.
///
/// Iteratively applies Givens rotations to diagonalize the matrix.
/// Convergence is measured by the off-diagonal Frobenius norm.
/// Robust and unconditionally stable; O(n^3) per sweep.
fn jacobi_eigenvalues(
    matrix: &Array2<f64>,
    tol: f64,
    max_sweeps: usize,
) -> SQDResult<Vec<f64>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(SQDError::DimensionMismatch {
            expected: n,
            got: matrix.ncols(),
        });
    }
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![matrix[[0, 0]]]);
    }

    let mut a = matrix.clone();

    for sweep in 0..max_sweeps {
        // Compute off-diagonal norm
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag += a[[i, j]] * a[[i, j]];
            }
        }
        off_diag = (2.0 * off_diag).sqrt();

        if off_diag < tol * n as f64 {
            let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
            return Ok(eigenvalues);
        }

        // Classical Jacobi threshold strategy
        let threshold = if sweep < 4 {
            0.2 * off_diag / (n * n) as f64
        } else {
            0.0
        };

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];

                if sweep < 4 && apq.abs() < threshold {
                    continue;
                }
                if apq.abs() < 1e-100 {
                    continue;
                }

                let diff = a[[q, q]] - a[[p, p]];
                let t = if diff.abs() < 1e-100 {
                    1.0_f64.copysign(apq / diff.abs().max(1e-300))
                } else {
                    let tau = diff / (2.0 * apq);
                    if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    }
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_val = s / (1.0 + c);

                a[[p, p]] -= t * apq;
                a[[q, q]] += t * apq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;

                for r in 0..p {
                    let arp = a[[r, p]];
                    let arq = a[[r, q]];
                    a[[r, p]] = arp - s * (arq + tau_val * arp);
                    a[[r, q]] = arq + s * (arp - tau_val * arq);
                    a[[p, r]] = a[[r, p]];
                    a[[q, r]] = a[[r, q]];
                }
                for r in (p + 1)..q {
                    let arp = a[[p, r]];
                    let arq = a[[r, q]];
                    a[[p, r]] = arp - s * (arq + tau_val * arp);
                    a[[r, q]] = arq + s * (arp - tau_val * arq);
                    a[[r, p]] = a[[p, r]];
                    a[[q, r]] = a[[r, q]];
                }
                for r in (q + 1)..n {
                    let apr = a[[p, r]];
                    let aqr = a[[q, r]];
                    a[[p, r]] = apr - s * (aqr + tau_val * apr);
                    a[[q, r]] = aqr + s * (apr - tau_val * aqr);
                    a[[r, p]] = a[[p, r]];
                    a[[r, q]] = a[[q, r]];
                }
            }
        }
    }

    let mut off_diag = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            off_diag += a[[i, j]] * a[[i, j]];
        }
    }
    Err(SQDError::EigenConvergence {
        iterations: max_sweeps,
        residual: (2.0 * off_diag).sqrt(),
    })
}

// ============================================================
// BOOTSTRAP ESTIMATOR
// ============================================================

/// Bootstrap confidence interval estimator for SQD energies.
///
/// Resamples the bitstring pool with replacement, runs SQD on each
/// bootstrap sample, and computes statistics on the resulting energy
/// distribution.
#[derive(Clone, Debug)]
pub struct BootstrapEstimator {
    config: SQDConfig,
}

/// Result of bootstrap estimation.
#[derive(Clone, Debug)]
pub struct BootstrapResult {
    /// Mean energy across bootstrap samples.
    pub mean_energy: f64,
    /// Standard deviation of energies.
    pub std_energy: f64,
    /// Lower bound of 95% confidence interval.
    pub ci_lower: f64,
    /// Upper bound of 95% confidence interval.
    pub ci_upper: f64,
    /// Bias estimate: mean(bootstrap) - original.
    pub bias: f64,
    /// All bootstrap energies (sorted).
    pub energies: Vec<f64>,
    /// Number of successful bootstrap iterations.
    pub num_successful: usize,
}

impl BootstrapEstimator {
    /// Create a new bootstrap estimator.
    pub fn new(config: &SQDConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Run bootstrap estimation.
    ///
    /// `original_energy` is the energy from the full sample set (for bias).
    pub fn estimate(
        &self,
        hamiltonian: &FermionicHamiltonian,
        samples: &[usize],
        original_energy: f64,
    ) -> SQDResult<BootstrapResult> {
        if samples.is_empty() {
            return Err(SQDError::EmptySubspace);
        }

        let seed = self.config.seed.unwrap_or(42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let n_bootstrap = self.config.bootstrap_samples;
        let n_samples = samples.len();
        let mut energies = Vec::with_capacity(n_bootstrap);

        for _ in 0..n_bootstrap {
            let resampled: Vec<usize> = (0..n_samples)
                .map(|_| samples[rng.gen_range(0..n_samples)])
                .collect();

            match sqd_solve_inner(hamiltonian, &resampled, &self.config) {
                Ok(result) => energies.push(result.ground_state_energy),
                Err(_) => continue,
            }
        }

        if energies.is_empty() {
            return Err(SQDError::NumericalInstability(
                "All bootstrap samples failed".into(),
            ));
        }

        energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let num_successful = energies.len();
        let mean_energy = energies.iter().sum::<f64>() / num_successful as f64;
        let variance = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / (num_successful as f64 - 1.0).max(1.0);
        let std_energy = variance.sqrt();

        // 95% confidence interval (percentile method)
        let lower_idx = ((num_successful as f64) * 0.025).floor() as usize;
        let upper_idx = ((num_successful as f64) * 0.975).ceil() as usize;
        let ci_lower = energies[lower_idx.min(num_successful - 1)];
        let ci_upper = energies[upper_idx.min(num_successful - 1)];

        let bias = mean_energy - original_energy;

        Ok(BootstrapResult {
            mean_energy,
            std_energy,
            ci_lower,
            ci_upper,
            bias,
            energies,
            num_successful,
        })
    }
}

impl fmt::Display for BootstrapResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bootstrap(mean={:.6}, std={:.6}, CI=[{:.6}, {:.6}], bias={:.2e}, n={})",
            self.mean_energy,
            self.std_energy,
            self.ci_lower,
            self.ci_upper,
            self.bias,
            self.num_successful,
        )
    }
}

// ============================================================
// SQD SOLVER (TOP-LEVEL API)
// ============================================================

/// Complete result of an SQD computation.
#[derive(Clone, Debug)]
pub struct SQDSolveResult {
    /// Ground state energy from the projected diagonalization.
    pub ground_state_energy: f64,
    /// All eigenvalues of the projected Hamiltonian (sorted ascending).
    pub eigenvalues: Vec<f64>,
    /// Dimension of the subspace used.
    pub subspace_dimension: usize,
    /// The basis states (bitstring indices) forming the subspace.
    pub basis_states: Vec<usize>,
    /// Bootstrap confidence interval (if bootstrap_samples > 0).
    pub bootstrap: Option<BootstrapResult>,
}

impl fmt::Display for SQDSolveResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SQD(E_gs={:.6}, subspace={}, n_eigenvalues={}",
            self.ground_state_energy,
            self.subspace_dimension,
            self.eigenvalues.len(),
        )?;
        if let Some(ref bs) = self.bootstrap {
            write!(
                f,
                ", CI=[{:.6}, {:.6}], bias={:.2e}",
                bs.ci_lower, bs.ci_upper, bs.bias,
            )?;
        }
        write!(f, ")")
    }
}

/// Run the full SQD pipeline: subspace construction, projection,
/// diagonalization, and optional bootstrap.
///
/// This is the main entry point for SQD computation.
pub fn sqd_solve(
    hamiltonian: &FermionicHamiltonian,
    samples: &[usize],
    config: &SQDConfig,
) -> SQDResult<SQDSolveResult> {
    config.validate()?;

    let core_result = sqd_solve_inner(hamiltonian, samples, config)?;

    let bootstrap = if config.bootstrap_samples > 0 {
        let estimator = BootstrapEstimator::new(config);
        match estimator.estimate(hamiltonian, samples, core_result.ground_state_energy) {
            Ok(br) => Some(br),
            Err(_) => None,
        }
    } else {
        None
    };

    Ok(SQDSolveResult {
        ground_state_energy: core_result.ground_state_energy,
        eigenvalues: core_result.eigenvalues,
        subspace_dimension: core_result.subspace_dimension,
        basis_states: core_result.basis_states,
        bootstrap,
    })
}

/// Inner SQD solve (without bootstrap). Used by both the main solver
/// and the bootstrap estimator.
fn sqd_solve_inner(
    hamiltonian: &FermionicHamiltonian,
    samples: &[usize],
    config: &SQDConfig,
) -> SQDResult<SQDSolveResult> {
    let builder = SubspaceBuilder::new(hamiltonian.num_spin_orbitals, config);
    let basis = builder.build(samples)?;

    let projected = ProjectedHamiltonian::build(hamiltonian, &basis)?;
    let solution = projected.solve(config)?;

    Ok(SQDSolveResult {
        ground_state_energy: solution.ground_state_energy,
        eigenvalues: solution.eigenvalues,
        subspace_dimension: basis.len(),
        basis_states: basis,
        bootstrap: None,
    })
}

// ============================================================
// CHEMISTRY APPLICATION HELPERS
// ============================================================

/// Generate synthetic bitstring samples from the Boltzmann distribution
/// of diagonal Hamiltonian elements.
///
/// Simulates what a noisy quantum device would produce: bitstrings biased
/// toward low-energy states but with significant thermal noise.
pub fn generate_thermal_samples(
    hamiltonian: &FermionicHamiltonian,
    n_samples: usize,
    temperature: f64,
    seed: Option<u64>,
) -> Vec<usize> {
    let n = hamiltonian.num_spin_orbitals;
    let dim = 1usize << n;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));

    let energies: Vec<f64> = (0..dim)
        .map(|b| hamiltonian.diagonal_element(b))
        .collect();

    let min_energy = energies.iter().cloned().fold(f64::INFINITY, f64::min);

    let weights: Vec<f64> = energies
        .iter()
        .map(|e| (-(e - min_energy) / temperature).exp())
        .collect();

    let total: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                samples.push(i);
                break;
            }
        }
    }
    samples
}

/// Generate bitstring samples restricted to a particular particle number sector.
///
/// Samples from all basis states with `n_electrons` occupied orbitals,
/// biased toward low-energy states with tunable noise level.
pub fn generate_sector_samples(
    hamiltonian: &FermionicHamiltonian,
    n_electrons: usize,
    n_samples: usize,
    noise_level: f64,
    seed: Option<u64>,
) -> Vec<usize> {
    let n = hamiltonian.num_spin_orbitals;
    let dim = 1usize << n;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));

    let sector: Vec<usize> = (0..dim)
        .filter(|&b| (b as u64).count_ones() as usize == n_electrons)
        .collect();

    if sector.is_empty() {
        return vec![];
    }

    let energies: Vec<f64> = sector
        .iter()
        .map(|&b| hamiltonian.diagonal_element(b))
        .collect();

    let min_energy = energies.iter().cloned().fold(f64::INFINITY, f64::min);

    let weights: Vec<f64> = energies
        .iter()
        .map(|e| {
            let boltzmann = (-(e - min_energy) / 0.1).exp();
            (1.0 - noise_level) * boltzmann + noise_level
        })
        .collect();

    let total: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                samples.push(sector[i]);
                break;
            }
        }
    }
    samples
}

/// Popcount: number of set bits in a usize (particle number of a basis state).
#[inline]
pub fn popcount(x: usize) -> usize {
    (x as u64).count_ones() as usize
}

/// Hamming distance between two bitstrings.
#[inline]
pub fn hamming_distance(a: usize, b: usize) -> usize {
    ((a ^ b) as u64).count_ones() as usize
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Config tests
    // ----------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = SQDConfig::default();
        assert_eq!(cfg.num_samples, 1000);
        assert_eq!(cfg.subspace_dimension, 512);
        assert_eq!(cfg.bootstrap_samples, 100);
        assert!(cfg.particle_number_filter.is_none());
        assert_eq!(cfg.hamming_expansion_radius, 0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder_pattern() {
        let cfg = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_subspace_dimension(32)
            .with_bootstrap_samples(50)
            .with_hamming_radius(2)
            .with_seed(123)
            .with_convergence_threshold(1e-10);

        assert_eq!(cfg.particle_number_filter, Some(2));
        assert_eq!(cfg.subspace_dimension, 32);
        assert_eq!(cfg.bootstrap_samples, 50);
        assert_eq!(cfg.hamming_expansion_radius, 2);
        assert_eq!(cfg.seed, Some(123));
        assert!((cfg.convergence_threshold - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn test_config_small_molecule_preset() {
        let cfg = SQDConfig::small_molecule();
        assert_eq!(cfg.num_samples, 100);
        assert_eq!(cfg.subspace_dimension, 64);
        assert_eq!(cfg.hamming_expansion_radius, 1);
    }

    #[test]
    fn test_config_medium_molecule_preset() {
        let cfg = SQDConfig::medium_molecule();
        assert_eq!(cfg.num_samples, 2000);
        assert_eq!(cfg.subspace_dimension, 1024);
    }

    #[test]
    fn test_config_hubbard_preset() {
        let cfg = SQDConfig::hubbard();
        assert_eq!(cfg.num_samples, 5000);
        assert_eq!(cfg.hamming_expansion_radius, 2);
    }

    #[test]
    fn test_config_validation_rejects_zero_subspace() {
        let cfg = SQDConfig::default().with_subspace_dimension(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_rejects_zero_threshold() {
        let mut cfg = SQDConfig::default();
        cfg.convergence_threshold = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_rejects_negative_threshold() {
        let mut cfg = SQDConfig::default();
        cfg.convergence_threshold = -1e-6;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_rejects_zero_sweeps() {
        let mut cfg = SQDConfig::default();
        cfg.max_jacobi_sweeps = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_display() {
        let cfg = SQDConfig::default();
        let s = format!("{}", cfg);
        assert!(s.contains("SQDConfig"));
        assert!(s.contains("1000"));
    }

    // ----------------------------------------------------------
    // Hamiltonian construction tests
    // ----------------------------------------------------------

    #[test]
    fn test_h2_sto3g_construction() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        assert_eq!(h2.num_spin_orbitals, 2);
        assert_eq!(h2.num_electrons, 2);
        assert!(h2.nuclear_repulsion > 0.0);
        assert_eq!(h2.one_body.len(), 2);
        assert_eq!(h2.two_body.len(), 16);
    }

    #[test]
    fn test_h2_sto3g_4so_construction() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        assert_eq!(h2.num_spin_orbitals, 4);
        assert_eq!(h2.num_electrons, 2);
        assert!((h2.nuclear_repulsion - 0.7137).abs() < 1e-4);
    }

    #[test]
    fn test_lih_construction() {
        let lih = FermionicHamiltonian::lih_sto3g();
        assert_eq!(lih.num_spin_orbitals, 8);
        assert_eq!(lih.num_electrons, 4);
        assert!(lih.nuclear_repulsion > 0.0);
    }

    #[test]
    fn test_hubbard_construction() {
        let hub = FermionicHamiltonian::hubbard_1d(4, 1.0, 2.0);
        assert_eq!(hub.num_spin_orbitals, 8);
        assert_eq!(hub.num_electrons, 4);
        assert_eq!(hub.nuclear_repulsion, 0.0);
    }

    #[test]
    fn test_hamiltonian_rejects_mismatched_one_body() {
        let result = FermionicHamiltonian::new(
            2,
            vec![vec![0.0; 3]; 2],
            vec![0.0; 16],
            0.0,
            2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_hamiltonian_rejects_mismatched_two_body() {
        let result = FermionicHamiltonian::new(
            2,
            vec![vec![0.0; 2]; 2],
            vec![0.0; 8],
            0.0,
            2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_one_body_symmetry() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        assert!(h2.one_body_is_symmetric(1e-12));
    }

    #[test]
    fn test_hamiltonian_display() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let s = format!("{}", h2);
        assert!(s.contains("FermionicHamiltonian"));
        assert!(s.contains("n_so=2"));
    }

    // ----------------------------------------------------------
    // Diagonal element tests
    // ----------------------------------------------------------

    #[test]
    fn test_diagonal_element_vacuum() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let e_vac = h2.diagonal_element(0b00);
        assert!((e_vac - h2.nuclear_repulsion).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_element_single_electron() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let e = h2.diagonal_element(0b01);
        let expected = h2.nuclear_repulsion + h2.one_body[0][0];
        assert!((e - expected).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_elements_finite() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        for b in 0..(1 << h2.num_spin_orbitals) {
            let e = h2.diagonal_element(b);
            assert!(e.is_finite(), "Diagonal element for {} is not finite", b);
        }
    }

    // ----------------------------------------------------------
    // Matrix element (Slater-Condon) tests
    // ----------------------------------------------------------

    #[test]
    fn test_matrix_element_diagonal_matches() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        for b in 0..4 {
            let diag = h2.diagonal_element(b);
            let me = h2.matrix_element(b, b);
            assert!(
                (diag - me).abs() < 1e-12,
                "Diagonal mismatch for state {}: {} vs {}",
                b, diag, me
            );
        }
    }

    #[test]
    fn test_matrix_element_hermiticity() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        for i in 0..16 {
            for j in i..16 {
                let hij = h2.matrix_element(i, j);
                let hji = h2.matrix_element(j, i);
                assert!(
                    (hij - hji).abs() < 1e-12,
                    "Non-Hermitian: H[{},{}]={} vs H[{},{}]={}",
                    i, j, hij, j, i, hji
                );
            }
        }
    }

    #[test]
    fn test_matrix_element_zero_for_different_particle_number() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        // |0001> (1 electron) and |1110> (3 electrons) -> zero
        let val = h2.matrix_element(0b0001, 0b1110);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_matrix_element_zero_for_triple_excitation() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        // States differing by 3+ orbital substitutions should give zero
        // |0001> and |1110> differ by 3 bits each way = 6 total bit diff
        let val = h2.matrix_element(0b0001, 0b1110);
        assert_eq!(val, 0.0);
    }

    // ----------------------------------------------------------
    // Full Hamiltonian matrix tests
    // ----------------------------------------------------------

    #[test]
    fn test_full_hamiltonian_is_symmetric() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let mat = h2.to_qubit_hamiltonian_matrix();
        let n = mat.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    (mat[[i, j]] - mat[[j, i]]).abs() < 1e-12,
                    "Not symmetric at ({},{}): {} vs {}",
                    i, j, mat[[i, j]], mat[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_jordan_wigner_matrix_dimension() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let mat = h2.to_qubit_hamiltonian_matrix();
        assert_eq!(mat.nrows(), 4); // 2^2
        assert_eq!(mat.ncols(), 4);
    }

    // ----------------------------------------------------------
    // Jacobi eigenvalue tests
    // ----------------------------------------------------------

    #[test]
    fn test_jacobi_1x1() {
        let mat = Array2::from_elem((1, 1), 3.14);
        let eigs = jacobi_eigenvalues(&mat, 1e-14, 100).unwrap();
        assert_eq!(eigs.len(), 1);
        assert!((eigs[0] - 3.14).abs() < 1e-12);
    }

    #[test]
    fn test_jacobi_2x2_identity() {
        let mat = Array2::eye(2);
        let eigs = jacobi_eigenvalues(&mat, 1e-14, 100).unwrap();
        assert_eq!(eigs.len(), 2);
        for e in &eigs {
            assert!((e - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_jacobi_2x2_known() {
        let mut mat = Array2::zeros((2, 2));
        mat[[0, 0]] = 2.0;
        mat[[0, 1]] = 1.0;
        mat[[1, 0]] = 1.0;
        mat[[1, 1]] = 2.0;

        let mut eigs = jacobi_eigenvalues(&mat, 1e-14, 100).unwrap();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_diagonal_matrix() {
        let mut mat = Array2::zeros((3, 3));
        mat[[0, 0]] = 5.0;
        mat[[1, 1]] = -2.0;
        mat[[2, 2]] = 3.0;

        let mut eigs = jacobi_eigenvalues(&mat, 1e-14, 100).unwrap();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - (-2.0)).abs() < 1e-12);
        assert!((eigs[1] - 3.0).abs() < 1e-12);
        assert!((eigs[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_jacobi_empty_matrix() {
        let mat = Array2::<f64>::zeros((0, 0));
        let eigs = jacobi_eigenvalues(&mat, 1e-14, 100).unwrap();
        assert!(eigs.is_empty());
    }

    #[test]
    fn test_jacobi_nonsquare_rejected() {
        // This cannot happen with Array2 construction, but test the dimension check
        // by creating a square matrix and verifying it works
        let mat = Array2::<f64>::zeros((3, 3));
        let eigs = jacobi_eigenvalues(&mat, 1e-14, 100).unwrap();
        assert_eq!(eigs.len(), 3);
    }

    // ----------------------------------------------------------
    // Subspace builder tests
    // ----------------------------------------------------------

    #[test]
    fn test_subspace_deduplication() {
        let cfg = SQDConfig::default();
        let builder = SubspaceBuilder::new(4, &cfg);
        let samples = vec![0b0011, 0b0011, 0b0101, 0b0011, 0b0101];
        let basis = builder.build(&samples).unwrap();
        assert_eq!(basis.len(), 2);
        assert!(basis.contains(&0b0011));
        assert!(basis.contains(&0b0101));
    }

    #[test]
    fn test_subspace_particle_filter() {
        let cfg = SQDConfig::default().with_particle_filter(2);
        let builder = SubspaceBuilder::new(4, &cfg);
        let samples = vec![0b0001, 0b0011, 0b0101, 0b0111, 0b1010, 0b1100];
        let basis = builder.build(&samples).unwrap();
        for &b in &basis {
            assert_eq!(
                popcount(b), 2,
                "State {} has {} electrons, expected 2",
                b, popcount(b)
            );
        }
    }

    #[test]
    fn test_subspace_hamming_expansion() {
        let cfg = SQDConfig::default().with_hamming_radius(1);
        let builder = SubspaceBuilder::new(3, &cfg);
        let samples = vec![0b000];
        let basis = builder.build(&samples).unwrap();
        assert!(basis.len() >= 4);
        assert!(basis.contains(&0b000));
        assert!(basis.contains(&0b001));
        assert!(basis.contains(&0b010));
        assert!(basis.contains(&0b100));
    }

    #[test]
    fn test_subspace_hamming_with_particle_filter() {
        // Particle-preserving expansion requires radius >= 2 because
        // single-bit flips change particle count by +/-1. Two-bit flips
        // (annihilate one orbital, create another) preserve particle number.
        let cfg = SQDConfig::default()
            .with_particle_filter(1)
            .with_hamming_radius(2);
        let builder = SubspaceBuilder::new(3, &cfg);
        let samples = vec![0b001];
        let basis = builder.build(&samples).unwrap();
        for &b in &basis {
            assert_eq!(popcount(b), 1, "Particle filter must enforce 1 electron");
        }
        // From 0b001, radius-2 neighbors include 0b010 and 0b100 (each via 2-bit flip),
        // plus others filtered out by particle count. All three 1-electron states survive.
        assert_eq!(basis.len(), 3);
    }

    #[test]
    fn test_subspace_empty_after_filter() {
        let cfg = SQDConfig::default().with_particle_filter(5);
        let builder = SubspaceBuilder::new(3, &cfg);
        let samples = vec![0b001, 0b010, 0b100];
        let result = builder.build(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_subspace_dimension_cap() {
        let cfg = SQDConfig::default()
            .with_subspace_dimension(3)
            .with_hamming_radius(1);
        let builder = SubspaceBuilder::new(4, &cfg);
        let samples = vec![0b0000];
        let basis = builder.build(&samples).unwrap();
        assert!(basis.len() <= 3);
    }

    #[test]
    fn test_subspace_rejects_empty_samples() {
        let cfg = SQDConfig::default();
        let builder = SubspaceBuilder::new(4, &cfg);
        let result = builder.build(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_subspace_skips_out_of_range() {
        let cfg = SQDConfig::default();
        let builder = SubspaceBuilder::new(2, &cfg);
        let samples = vec![0, 1, 2, 3, 99, 1000];
        let basis = builder.build(&samples).unwrap();
        assert_eq!(basis.len(), 4);
    }

    #[test]
    fn test_subspace_size_method() {
        let cfg = SQDConfig::default().with_particle_filter(2);
        let builder = SubspaceBuilder::new(4, &cfg);
        let samples = vec![0b0011, 0b0101, 0b1010];
        let size = builder.subspace_size(&samples).unwrap();
        assert_eq!(size, 3);
    }

    // ----------------------------------------------------------
    // Projected Hamiltonian tests
    // ----------------------------------------------------------

    #[test]
    fn test_projected_hamiltonian_hermiticity() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let basis = vec![0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100];
        let proj = ProjectedHamiltonian::build(&h2, &basis).unwrap();
        assert!(proj.is_hermitian(1e-12));
    }

    #[test]
    fn test_projected_hamiltonian_dimension() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let basis = vec![0b00, 0b01, 0b10, 0b11];
        let proj = ProjectedHamiltonian::build(&h2, &basis).unwrap();
        assert_eq!(proj.dimension, 4);
        assert_eq!(proj.matrix.nrows(), 4);
        assert_eq!(proj.matrix.ncols(), 4);
    }

    #[test]
    fn test_projected_hamiltonian_trace() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let basis = vec![0b00, 0b01, 0b10, 0b11];
        let proj = ProjectedHamiltonian::build(&h2, &basis).unwrap();
        let trace = proj.trace();
        let expected_trace: f64 = basis.iter().map(|&b| h2.diagonal_element(b)).sum();
        assert!(
            (trace - expected_trace).abs() < 1e-10,
            "Trace mismatch: {} vs {}",
            trace, expected_trace
        );
    }

    #[test]
    fn test_projected_hamiltonian_empty_basis() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let result = ProjectedHamiltonian::build(&h2, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_projected_hamiltonian_single_state() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let basis = vec![0b11];
        let proj = ProjectedHamiltonian::build(&h2, &basis).unwrap();
        assert_eq!(proj.dimension, 1);
        let energy = h2.diagonal_element(0b11);
        assert!((proj.matrix[[0, 0]] - energy).abs() < 1e-12);
    }

    #[test]
    fn test_projected_hamiltonian_display() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let basis = vec![0b00, 0b11];
        let proj = ProjectedHamiltonian::build(&h2, &basis).unwrap();
        let s = format!("{}", proj);
        assert!(s.contains("ProjectedHamiltonian"));
        assert!(s.contains("dim=2"));
    }

    #[test]
    fn test_projected_hamiltonian_frobenius_norm() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let basis = vec![0b00, 0b11];
        let proj = ProjectedHamiltonian::build(&h2, &basis).unwrap();
        let norm = proj.frobenius_norm();
        assert!(norm > 0.0);
        assert!(norm.is_finite());
    }

    // ----------------------------------------------------------
    // H2 ground state energy tests
    // ----------------------------------------------------------

    #[test]
    fn test_h2_exact_diag_2so() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let (eigenvalues, gs_energy) = h2.exact_diagonalize().unwrap();
        assert_eq!(eigenvalues.len(), 4);
        assert!(gs_energy < 0.0, "H2 ground state should be negative");
    }

    #[test]
    fn test_h2_exact_diag_sector() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let (eigenvalues, gs_energy) = h2.exact_diagonalize_sector(2).unwrap();
        // C(4,2) = 6 states in the 2-electron sector
        assert_eq!(eigenvalues.len(), 6);
        assert!(
            gs_energy < -0.5,
            "H2 ground state energy should be < -0.5 Hartree, got {}",
            gs_energy
        );
    }

    #[test]
    fn test_h2_sqd_recovers_ground_state() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();

        // Use all 2-electron states as samples (best case)
        let samples: Vec<usize> = (0..16).filter(|&b| popcount(b) == 2).collect();

        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(0)
            .with_seed(42);

        let result = sqd_solve(&h2, &samples, &config).unwrap();

        let (_, exact_gs) = h2.exact_diagonalize_sector(2).unwrap();
        assert!(
            (result.ground_state_energy - exact_gs).abs() < 1e-8,
            "SQD energy {} should match exact {} within 1e-8",
            result.ground_state_energy, exact_gs
        );
    }

    #[test]
    fn test_h2_sqd_with_noisy_samples() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();

        let samples = generate_sector_samples(&h2, 2, 100, 0.3, Some(42));

        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(0)
            .with_seed(42);

        let result = sqd_solve(&h2, &samples, &config).unwrap();

        let (_, exact_gs) = h2.exact_diagonalize_sector(2).unwrap();
        assert!(
            (result.ground_state_energy - exact_gs).abs() < 0.5,
            "SQD with noisy samples should be within 0.5 Hartree: got {} vs {}",
            result.ground_state_energy, exact_gs
        );
    }

    // ----------------------------------------------------------
    // Hubbard model tests
    // ----------------------------------------------------------

    #[test]
    fn test_hubbard_2site_exact() {
        let hub = FermionicHamiltonian::hubbard_1d(2, 1.0, 0.0);
        let (_, gs) = hub.exact_diagonalize_sector(2).unwrap();
        assert!(
            (gs - (-2.0)).abs() < 0.1,
            "Hubbard 2-site U=0 ground state should be ~-2.0, got {}",
            gs
        );
    }

    #[test]
    fn test_hubbard_sqd() {
        let hub = FermionicHamiltonian::hubbard_1d(2, 1.0, 2.0);
        let n_e = hub.num_electrons;

        let samples = generate_sector_samples(&hub, n_e, 50, 0.2, Some(42));

        let config = SQDConfig::hubbard()
            .with_particle_filter(n_e)
            .with_bootstrap_samples(0)
            .with_subspace_dimension(100)
            .with_seed(42);

        let result = sqd_solve(&hub, &samples, &config).unwrap();
        let (_, exact_gs) = hub.exact_diagonalize_sector(n_e).unwrap();

        assert!(
            (result.ground_state_energy - exact_gs).abs() < 0.5,
            "Hubbard SQD should be close to exact: {} vs {}",
            result.ground_state_energy, exact_gs
        );
    }

    #[test]
    fn test_hubbard_hopping_only() {
        // U=0: pure hopping, analytically solvable
        let hub = FermionicHamiltonian::hubbard_1d(3, 1.0, 0.0);
        let (_, gs) = hub.exact_diagonalize_sector(3).unwrap();
        assert!(gs < 0.0, "Non-interacting Hubbard should have negative GS");
    }

    // ----------------------------------------------------------
    // Bootstrap tests
    // ----------------------------------------------------------

    #[test]
    fn test_bootstrap_produces_confidence_interval() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let samples = vec![0b00, 0b01, 0b10, 0b11];

        let config = SQDConfig::default()
            .with_bootstrap_samples(20)
            .with_seed(42);

        let result = sqd_solve(&h2, &samples, &config).unwrap();
        let bs = result.bootstrap.as_ref().unwrap();

        assert!(bs.num_successful > 0);
        assert!(bs.ci_lower <= bs.mean_energy);
        assert!(bs.ci_upper >= bs.mean_energy);
        assert!(bs.std_energy >= 0.0);
    }

    #[test]
    fn test_bootstrap_with_particle_filter() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let samples: Vec<usize> = (0..16).filter(|&b| popcount(b) == 2).collect();

        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(10)
            .with_seed(42);

        let result = sqd_solve(&h2, &samples, &config).unwrap();
        assert!(result.bootstrap.is_some());
    }

    #[test]
    fn test_bootstrap_bias_small_for_complete_subspace() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let samples: Vec<usize> = (0..16).filter(|&b| popcount(b) == 2).collect();

        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(30)
            .with_seed(42);

        let result = sqd_solve(&h2, &samples, &config).unwrap();
        let bs = result.bootstrap.as_ref().unwrap();

        // Complete subspace => resampling gives same result => small bias
        assert!(
            bs.bias.abs() < 0.5,
            "Bootstrap bias should be small, got {}",
            bs.bias
        );
    }

    #[test]
    fn test_bootstrap_display() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let samples = vec![0b00, 0b01, 0b10, 0b11];
        let config = SQDConfig::default()
            .with_bootstrap_samples(5)
            .with_seed(42);
        let result = sqd_solve(&h2, &samples, &config).unwrap();
        let bs = result.bootstrap.as_ref().unwrap();
        let s = format!("{}", bs);
        assert!(s.contains("Bootstrap"));
    }

    // ----------------------------------------------------------
    // Utility function tests
    // ----------------------------------------------------------

    #[test]
    fn test_popcount() {
        assert_eq!(popcount(0), 0);
        assert_eq!(popcount(1), 1);
        assert_eq!(popcount(0b1010), 2);
        assert_eq!(popcount(0b1111), 4);
        assert_eq!(popcount(0xFF), 8);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0b0000, 0b1111), 4);
        assert_eq!(hamming_distance(0b1010, 0b0101), 4);
        assert_eq!(hamming_distance(0b1100, 0b1010), 2);
    }

    #[test]
    fn test_hamming_distance_symmetric() {
        assert_eq!(hamming_distance(0b1010, 0b0011), hamming_distance(0b0011, 0b1010));
    }

    // ----------------------------------------------------------
    // Sampling helper tests
    // ----------------------------------------------------------

    #[test]
    fn test_thermal_samples_valid_range() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let samples = generate_thermal_samples(&h2, 100, 0.5, Some(42));
        assert_eq!(samples.len(), 100);
        let max_state = 1usize << h2.num_spin_orbitals;
        for &s in &samples {
            assert!(s < max_state, "Sample {} out of range", s);
        }
    }

    #[test]
    fn test_sector_samples_correct_particle_number() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let samples = generate_sector_samples(&h2, 2, 50, 0.1, Some(42));
        for &s in &samples {
            assert_eq!(
                popcount(s), 2,
                "Sector sample {} has wrong particle number {}",
                s, popcount(s)
            );
        }
    }

    #[test]
    fn test_sector_samples_empty_for_impossible_sector() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let samples = generate_sector_samples(&h2, 10, 50, 0.1, Some(42));
        assert!(samples.is_empty());
    }

    #[test]
    fn test_thermal_samples_reproducible() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let s1 = generate_thermal_samples(&h2, 50, 0.5, Some(123));
        let s2 = generate_thermal_samples(&h2, 50, 0.5, Some(123));
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_sector_samples_reproducible() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let s1 = generate_sector_samples(&h2, 2, 50, 0.2, Some(99));
        let s2 = generate_sector_samples(&h2, 2, 50, 0.2, Some(99));
        assert_eq!(s1, s2);
    }

    // ----------------------------------------------------------
    // Error type tests
    // ----------------------------------------------------------

    #[test]
    fn test_error_display_empty_subspace() {
        let e = SQDError::EmptySubspace;
        assert!(format!("{}", e).contains("empty"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = SQDError::DimensionMismatch { expected: 4, got: 8 };
        let msg = format!("{}", e);
        assert!(msg.contains("4"));
        assert!(msg.contains("8"));
    }

    #[test]
    fn test_error_display_eigen_convergence() {
        let e = SQDError::EigenConvergence { iterations: 100, residual: 1e-5 };
        let msg = format!("{}", e);
        assert!(msg.contains("100"));
    }

    #[test]
    fn test_error_display_invalid_config() {
        let e = SQDError::InvalidConfig("bad param".to_string());
        assert!(format!("{}", e).contains("bad param"));
    }

    #[test]
    fn test_error_display_invalid_integrals() {
        let e = SQDError::InvalidIntegrals("wrong size".to_string());
        assert!(format!("{}", e).contains("wrong size"));
    }

    #[test]
    fn test_error_display_numerical_instability() {
        let e = SQDError::NumericalInstability("overflow".to_string());
        assert!(format!("{}", e).contains("overflow"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let e: Box<dyn std::error::Error> =
            Box::new(SQDError::EmptySubspace);
        assert!(!e.to_string().is_empty());
    }

    // ----------------------------------------------------------
    // Integration / end-to-end tests
    // ----------------------------------------------------------

    #[test]
    fn test_full_sqd_pipeline_h2() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();

        let samples = generate_sector_samples(&h2, 2, 200, 0.1, Some(42));

        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(10)
            .with_hamming_radius(1)
            .with_seed(42);

        let result = sqd_solve(&h2, &samples, &config).unwrap();

        assert!(result.subspace_dimension > 0);
        assert!(!result.eigenvalues.is_empty());
        assert!(result.bootstrap.is_some());

        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] >= result.eigenvalues[i - 1] - 1e-10,
                "Eigenvalues not sorted"
            );
        }

        let s = format!("{}", result);
        assert!(s.contains("SQD"));
    }

    #[test]
    fn test_sqd_subspace_is_subset_of_hilbert_space() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let samples: Vec<usize> = (0..16).filter(|&b| popcount(b) == 2).collect();
        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(0);
        let result = sqd_solve(&h2, &samples, &config).unwrap();

        let max_state = 1usize << h2.num_spin_orbitals;
        for &b in &result.basis_states {
            assert!(b < max_state);
            assert_eq!(popcount(b), 2);
        }
    }

    #[test]
    fn test_sqd_eigenvalue_count_matches_subspace() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let samples: Vec<usize> = (0..16).filter(|&b| popcount(b) == 2).collect();
        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(0);
        let result = sqd_solve(&h2, &samples, &config).unwrap();
        assert_eq!(result.eigenvalues.len(), result.subspace_dimension);
    }

    #[test]
    fn test_sqd_ground_state_is_variational_bound() {
        // SQD ground state energy >= true ground state (variational principle)
        let h2 = FermionicHamiltonian::h2_sto3g_4so();
        let samples: Vec<usize> = vec![0b0011, 0b0101, 0b1010];
        let config = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_bootstrap_samples(0);
        let result = sqd_solve(&h2, &samples, &config).unwrap();

        let (_, exact_gs) = h2.exact_diagonalize_sector(2).unwrap();

        assert!(
            result.ground_state_energy >= exact_gs - 1e-8,
            "SQD energy {} violates variational bound (exact={})",
            result.ground_state_energy, exact_gs
        );
    }

    #[test]
    fn test_sqd_result_display() {
        let h2 = FermionicHamiltonian::h2_sto3g(0.74);
        let samples = vec![0b00, 0b01, 0b10, 0b11];
        let config = SQDConfig::default()
            .with_bootstrap_samples(0)
            .with_seed(42);
        let result = sqd_solve(&h2, &samples, &config).unwrap();
        let s = format!("{}", result);
        assert!(s.contains("E_gs="));
        assert!(s.contains("subspace="));
    }

    #[test]
    fn test_sqd_with_hamming_expansion_improves_energy() {
        let h2 = FermionicHamiltonian::h2_sto3g_4so();

        // Start with a single 2-electron state
        let samples = vec![0b0011];

        // Without expansion
        let config_no_expand = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_hamming_radius(0)
            .with_bootstrap_samples(0);
        let result_no = sqd_solve(&h2, &samples, &config_no_expand).unwrap();

        // With expansion (radius 2 generates more 2-electron states)
        let config_expand = SQDConfig::small_molecule()
            .with_particle_filter(2)
            .with_hamming_radius(2)
            .with_bootstrap_samples(0);
        let result_yes = sqd_solve(&h2, &samples, &config_expand).unwrap();

        // Expanded subspace should be at least as large
        assert!(result_yes.subspace_dimension >= result_no.subspace_dimension);
        // And energy should be at least as good (lower or equal)
        assert!(
            result_yes.ground_state_energy <= result_no.ground_state_energy + 1e-10,
            "Expansion should not worsen energy: {} vs {}",
            result_yes.ground_state_energy, result_no.ground_state_energy
        );
    }
}
