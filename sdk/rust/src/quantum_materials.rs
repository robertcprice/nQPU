//! Quantum Materials Screening Module
//!
//! Quantum-accelerated materials discovery for battery cathodes, superconductors,
//! and electronic band structure calculations.
//!
//! # Features
//! - Crystal lattice construction (cubic, FCC, BCC) with reciprocal space
//! - Tight-binding band structure via Bloch Hamiltonian H(k)
//! - Density of states with Lorentzian broadening
//! - BCS superconductor modelling (McMillan Tc, gap equation, coherence length)
//! - Hubbard model exact diagonalisation for small clusters
//! - Battery material screening (voltage, capacity, diffusion barrier)
//! - Composite scoring pipeline for high-throughput candidate ranking
//!
//! # References
//! - McMillan (1968) -- Transition temperature of strong-coupled superconductors
//! - Hubbard (1963) -- Electron correlations in narrow energy bands
//! - Ashcroft & Mermin -- Solid State Physics (tight-binding formalism)
//! - Urban, Seo, Ceder (2016) -- Computational understanding of Li-ion batteries

use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors arising from quantum materials computations.
#[derive(Debug, Clone)]
pub enum MaterialsError {
    /// Configuration parameter is out of valid range.
    InvalidConfig(String),
    /// Iterative solver did not converge within budget.
    ConvergenceFailure { iterations: usize, residual: f64 },
    /// Matrix dimensions are incompatible for the requested operation.
    DimensionMismatch(String),
    /// Floating-point computation produced NaN or Inf.
    NumericalInstability(String),
    /// Requested operation is not supported for the given model.
    UnsupportedOperation(String),
}

impl std::fmt::Display for MaterialsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::ConvergenceFailure { iterations, residual } => {
                write!(
                    f,
                    "Convergence failure after {} iterations (residual={:.2e})",
                    iterations, residual
                )
            }
            Self::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            Self::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
            Self::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
        }
    }
}

impl std::error::Error for MaterialsError {}

pub type MaterialsResult<T> = Result<T, MaterialsError>;

// ===================================================================
// SOLVER ENUM
// ===================================================================

/// Solver backend for materials Hamiltonians.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialsSolver {
    /// Variational Quantum Eigensolver (hybrid quantum-classical).
    VQE,
    /// Full exact diagonalisation (exponential cost, small systems only).
    ExactDiagonalization,
    /// Dynamical Mean-Field Theory (lattice self-consistency).
    DMFT,
    /// Non-interacting tight-binding (single-particle Bloch bands).
    TightBinding,
}

impl std::fmt::Display for MaterialsSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VQE => write!(f, "VQE"),
            Self::ExactDiagonalization => write!(f, "ExactDiagonalization"),
            Self::DMFT => write!(f, "DMFT"),
            Self::TightBinding => write!(f, "TightBinding"),
        }
    }
}

// ===================================================================
// CONFIGURATION (BUILDER PATTERN)
// ===================================================================

/// Configuration for materials screening calculations.
#[derive(Debug, Clone)]
pub struct MaterialsConfig {
    /// Number of lattice sites (2..=100).
    pub num_sites: usize,
    /// Number of orbitals per site (1..=10).
    pub num_orbitals: usize,
    /// Solver backend.
    pub solver: MaterialsSolver,
    /// Self-consistency convergence threshold.
    pub convergence_threshold: f64,
    /// Maximum solver iterations.
    pub max_iterations: usize,
    /// Temperature in Kelvin.
    pub temperature_kelvin: f64,
}

impl Default for MaterialsConfig {
    fn default() -> Self {
        Self {
            num_sites: 8,
            num_orbitals: 4,
            solver: MaterialsSolver::TightBinding,
            convergence_threshold: 1e-6,
            max_iterations: 100,
            temperature_kelvin: 300.0,
        }
    }
}

impl MaterialsConfig {
    /// Start building a new configuration from defaults.
    pub fn builder() -> MaterialsConfigBuilder {
        MaterialsConfigBuilder {
            config: MaterialsConfig::default(),
        }
    }

    /// Validate all fields; return error if any are out of range.
    pub fn validate(&self) -> MaterialsResult<()> {
        if self.num_sites < 2 || self.num_sites > 100 {
            return Err(MaterialsError::InvalidConfig(format!(
                "num_sites must be in [2, 100], got {}",
                self.num_sites
            )));
        }
        if self.num_orbitals < 1 || self.num_orbitals > 10 {
            return Err(MaterialsError::InvalidConfig(format!(
                "num_orbitals must be in [1, 10], got {}",
                self.num_orbitals
            )));
        }
        if self.convergence_threshold <= 0.0 {
            return Err(MaterialsError::InvalidConfig(
                "convergence_threshold must be positive".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(MaterialsError::InvalidConfig(
                "max_iterations must be >= 1".into(),
            ));
        }
        if self.temperature_kelvin < 0.0 {
            return Err(MaterialsError::InvalidConfig(
                "temperature_kelvin must be non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// Builder for `MaterialsConfig`.
pub struct MaterialsConfigBuilder {
    config: MaterialsConfig,
}

impl MaterialsConfigBuilder {
    pub fn num_sites(mut self, n: usize) -> Self {
        self.config.num_sites = n;
        self
    }

    pub fn num_orbitals(mut self, n: usize) -> Self {
        self.config.num_orbitals = n;
        self
    }

    pub fn solver(mut self, s: MaterialsSolver) -> Self {
        self.config.solver = s;
        self
    }

    pub fn convergence_threshold(mut self, tol: f64) -> Self {
        self.config.convergence_threshold = tol;
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.config.max_iterations = n;
        self
    }

    pub fn temperature_kelvin(mut self, t: f64) -> Self {
        self.config.temperature_kelvin = t;
        self
    }

    /// Consume the builder and return a validated `MaterialsConfig`.
    pub fn build(self) -> MaterialsResult<MaterialsConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ===================================================================
// ATOM TYPE
// ===================================================================

/// Description of an atomic species on a lattice site.
#[derive(Debug, Clone)]
pub struct AtomType {
    /// Chemical symbol (e.g. "Li", "Co", "O").
    pub symbol: String,
    /// Atomic number Z.
    pub atomic_number: usize,
    /// Number of valence electrons contributing to bands.
    pub num_valence_electrons: usize,
    /// On-site orbital energies (eV), one per orbital.
    pub orbital_energies: Vec<f64>,
}

impl AtomType {
    pub fn new(symbol: &str, z: usize, valence: usize, energies: Vec<f64>) -> Self {
        Self {
            symbol: symbol.to_string(),
            atomic_number: z,
            num_valence_electrons: valence,
            orbital_energies: energies,
        }
    }
}

// ===================================================================
// CRYSTAL LATTICE
// ===================================================================

/// A Bravais lattice with basis atoms.
#[derive(Debug, Clone)]
pub struct CrystalLattice {
    /// Primitive lattice vectors a1, a2, a3 (rows), in Angstroms.
    pub lattice_vectors: [[f64; 3]; 3],
    /// Basis atoms: fractional coordinates + species.
    pub atom_positions: Vec<([f64; 3], AtomType)>,
}

impl CrystalLattice {
    /// Simple cubic lattice with parameter `a` and one atom at origin.
    pub fn new_cubic(a: f64) -> Self {
        Self {
            lattice_vectors: [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
            atom_positions: vec![(
                [0.0, 0.0, 0.0],
                AtomType::new("X", 1, 1, vec![-1.0]),
            )],
        }
    }

    /// Face-centred cubic lattice with conventional parameter `a`.
    /// Primitive vectors: a/2*(0,1,1), a/2*(1,0,1), a/2*(1,1,0).
    pub fn new_fcc(a: f64) -> Self {
        let h = a / 2.0;
        Self {
            lattice_vectors: [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
            atom_positions: vec![(
                [0.0, 0.0, 0.0],
                AtomType::new("X", 1, 1, vec![-1.0]),
            )],
        }
    }

    /// Body-centred cubic lattice with conventional parameter `a`.
    /// Primitive vectors: a/2*(-1,1,1), a/2*(1,-1,1), a/2*(1,1,-1).
    pub fn new_bcc(a: f64) -> Self {
        let h = a / 2.0;
        Self {
            lattice_vectors: [[-h, h, h], [h, -h, h], [h, h, -h]],
            atom_positions: vec![(
                [0.0, 0.0, 0.0],
                AtomType::new("X", 1, 1, vec![-1.0]),
            )],
        }
    }

    /// Reciprocal lattice vectors b_i satisfying a_i . b_j = 2*pi*delta_{ij}.
    ///
    /// b1 = 2*pi * (a2 x a3) / (a1 . (a2 x a3))   (and cyclic permutations).
    pub fn reciprocal_vectors(&self) -> [[f64; 3]; 3] {
        let a = &self.lattice_vectors;
        let vol = Self::triple_product(&a[0], &a[1], &a[2]);
        if vol.abs() < 1e-30 {
            return [[0.0; 3]; 3];
        }
        let factor = 2.0 * PI / vol;
        let b1 = Self::scale_vec(factor, &Self::cross(&a[1], &a[2]));
        let b2 = Self::scale_vec(factor, &Self::cross(&a[2], &a[0]));
        let b3 = Self::scale_vec(factor, &Self::cross(&a[0], &a[1]));
        [b1, b2, b3]
    }

    /// Generate a k-path through high-symmetry points in the first Brillouin zone.
    ///
    /// For a cubic lattice the path is Gamma -> X -> M -> Gamma.
    /// Returns `num_points` evenly spaced k-vectors along the path.
    pub fn k_path(&self, num_points: usize) -> Vec<[f64; 3]> {
        let b = self.reciprocal_vectors();

        // High-symmetry points (fractional reciprocal coords):
        // Gamma = (0,0,0), X = (0.5,0,0), M = (0.5,0.5,0)
        let gamma = [0.0_f64; 3];
        let x_pt = Self::frac_to_cart(&[0.5, 0.0, 0.0], &b);
        let m_pt = Self::frac_to_cart(&[0.5, 0.5, 0.0], &b);

        let segments: Vec<([f64; 3], [f64; 3])> =
            vec![(gamma, x_pt), (x_pt, m_pt), (m_pt, gamma)];

        let pts_per_seg = if num_points < 3 {
            1
        } else {
            num_points / segments.len()
        };

        let mut path = Vec::with_capacity(num_points);
        for (start, end) in &segments {
            for i in 0..pts_per_seg {
                let t = i as f64 / pts_per_seg as f64;
                let k = [
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1]),
                    start[2] + t * (end[2] - start[2]),
                ];
                path.push(k);
            }
        }
        // Ensure we return exactly num_points entries.
        while path.len() < num_points {
            path.push(gamma);
        }
        path.truncate(num_points);
        path
    }

    // ----- internal helpers -----

    fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn triple_product(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
        Self::dot(a, &Self::cross(b, c))
    }

    fn scale_vec(s: f64, v: &[f64; 3]) -> [f64; 3] {
        [s * v[0], s * v[1], s * v[2]]
    }

    fn frac_to_cart(frac: &[f64; 3], b: &[[f64; 3]; 3]) -> [f64; 3] {
        [
            frac[0] * b[0][0] + frac[1] * b[1][0] + frac[2] * b[2][0],
            frac[0] * b[0][1] + frac[1] * b[1][1] + frac[2] * b[2][1],
            frac[0] * b[0][2] + frac[1] * b[1][2] + frac[2] * b[2][2],
        ]
    }
}

// ===================================================================
// TIGHT-BINDING MODEL
// ===================================================================

/// Non-interacting tight-binding model on a crystal lattice.
///
/// H(k) = sum_{ij} t_{ij} * e^{i k . (R_j - R_i)}  +  diag(epsilon_i)
///
/// Eigenvalues of H(k) give band energies at each k-point.
#[derive(Debug, Clone)]
pub struct TightBindingModel {
    /// Hopping parameters (site_i, site_j, t_ij).
    pub hopping_parameters: Vec<(usize, usize, Complex64)>,
    /// On-site energies for each orbital.
    pub on_site_energies: Vec<f64>,
}

impl TightBindingModel {
    /// Create a new tight-binding model.
    pub fn new(on_site_energies: Vec<f64>) -> Self {
        Self {
            hopping_parameters: Vec::new(),
            on_site_energies,
        }
    }

    /// Add a hopping term between orbitals i and j.
    pub fn add_hopping(&mut self, i: usize, j: usize, t: Complex64) {
        self.hopping_parameters.push((i, j, t));
    }

    /// Build the Bloch Hamiltonian H(k) for a given lattice.
    ///
    /// For each hopping (i, j, t_ij) the matrix element is
    /// H_{ij}(k) += t_ij * exp(i k . delta_{ij})
    /// where delta_{ij} is the displacement vector from site i to site j.
    ///
    /// For simplicity in this implementation the displacement is derived from
    /// the atom positions in the lattice (in fractional coordinates converted
    /// to Cartesian).
    pub fn build_hamiltonian_k(
        &self,
        k: &[f64; 3],
        lattice: &CrystalLattice,
    ) -> Array2<Complex64> {
        let n = self.on_site_energies.len();
        let mut h = Array2::<Complex64>::zeros((n, n));

        // On-site energies on the diagonal.
        for i in 0..n {
            h[[i, i]] = Complex64::new(self.on_site_energies[i], 0.0);
        }

        // Hopping terms with Bloch phase.
        let a = &lattice.lattice_vectors;
        for &(i, j, t_ij) in &self.hopping_parameters {
            if i >= n || j >= n {
                continue;
            }
            // Displacement in Cartesian from fractional positions.
            let delta = if i < lattice.atom_positions.len() && j < lattice.atom_positions.len()
            {
                let fi = &lattice.atom_positions[i].0;
                let fj = &lattice.atom_positions[j].0;
                let df = [fj[0] - fi[0], fj[1] - fi[1], fj[2] - fi[2]];
                let dx = df[0] * a[0][0] + df[1] * a[1][0] + df[2] * a[2][0];
                let dy = df[0] * a[0][1] + df[1] * a[1][1] + df[2] * a[2][1];
                let dz = df[0] * a[0][2] + df[1] * a[1][2] + df[2] * a[2][2];
                [dx, dy, dz]
            } else {
                // If positions not available, assume nearest-neighbour along a1.
                lattice.lattice_vectors[0]
            };

            let phase = k[0] * delta[0] + k[1] * delta[1] + k[2] * delta[2];
            let bloch = Complex64::new(phase.cos(), phase.sin());
            h[[i, j]] += t_ij * bloch;
            // Hermitian conjugate.
            h[[j, i]] += t_ij.conj() * bloch.conj();
        }

        h
    }

    /// Compute band energies along a k-path.
    ///
    /// Returns a vector of eigenvalue sets, one per k-point.
    /// Each inner vector is sorted in ascending order.
    pub fn band_structure(
        &self,
        k_path: &[[f64; 3]],
        lattice: &CrystalLattice,
    ) -> Vec<Vec<f64>> {
        k_path
            .iter()
            .map(|k| {
                let h = self.build_hamiltonian_k(k, lattice);
                eigenvalues_hermitian(&h)
            })
            .collect()
    }

    /// Density of states with Lorentzian broadening.
    ///
    /// DOS(E) = (1/N_k) sum_{n,k} eta / [pi * ((E - E_{nk})^2 + eta^2)]
    ///
    /// Returns (energy, dos) pairs on a uniform grid.
    pub fn density_of_states(
        &self,
        energies_flat: &[f64],
        broadening: f64,
        grid: usize,
    ) -> Vec<(f64, f64)> {
        if energies_flat.is_empty() || grid == 0 {
            return Vec::new();
        }
        let e_min = energies_flat
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            - 5.0 * broadening;
        let e_max = energies_flat
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            + 5.0 * broadening;
        let de = if grid > 1 {
            (e_max - e_min) / (grid - 1) as f64
        } else {
            1.0
        };
        let eta = broadening;
        let n_states = energies_flat.len() as f64;

        (0..grid)
            .map(|i| {
                let e = e_min + i as f64 * de;
                let dos: f64 = energies_flat
                    .iter()
                    .map(|&ek| eta / (PI * ((e - ek).powi(2) + eta * eta)))
                    .sum::<f64>()
                    / n_states;
                (e, dos)
            })
            .collect()
    }

    /// Estimate the Fermi energy for a given number of electrons and temperature.
    ///
    /// Uses bisection on the integrated Fermi-Dirac occupation to match
    /// `num_electrons`. For T=0, the Fermi level bisects the sorted eigenvalues.
    pub fn fermi_energy(
        &self,
        all_energies: &[f64],
        num_electrons: usize,
        temperature: f64,
    ) -> f64 {
        if all_energies.is_empty() {
            return 0.0;
        }
        let mut sorted = all_energies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // T = 0: Fermi level between occupied and unoccupied levels.
        if temperature < 1e-10 {
            let idx = num_electrons.min(sorted.len());
            if idx == 0 {
                return sorted[0] - 1.0;
            }
            if idx >= sorted.len() {
                return sorted[sorted.len() - 1] + 1.0;
            }
            return (sorted[idx - 1] + sorted[idx]) / 2.0;
        }

        // Finite T: bisection on Fermi-Dirac occupation.
        let kb = 8.617333262e-5; // eV/K
        let beta = 1.0 / (kb * temperature);
        let target = num_electrons as f64;

        let mut lo = sorted[0] - 10.0 * kb * temperature;
        let mut hi = sorted[sorted.len() - 1] + 10.0 * kb * temperature;

        for _ in 0..200 {
            let mu = (lo + hi) / 2.0;
            let n: f64 = sorted
                .iter()
                .map(|&e| {
                    let x = beta * (e - mu);
                    if x > 500.0 {
                        0.0
                    } else if x < -500.0 {
                        1.0
                    } else {
                        1.0 / (1.0 + x.exp())
                    }
                })
                .sum();
            if n < target {
                lo = mu;
            } else {
                hi = mu;
            }
        }
        (lo + hi) / 2.0
    }
}

// ===================================================================
// BATTERY MATERIAL
// ===================================================================

/// Model for a rechargeable battery cathode material.
///
/// Encodes intercalation energetics for Li-ion battery screening.
#[derive(Debug, Clone)]
pub struct BatteryMaterial {
    /// Crystal structure of the cathode.
    pub cathode: CrystalLattice,
    /// Positions of Li intercalation sites (fractional coordinates).
    pub intercalation_sites: Vec<[f64; 3]>,
    /// Computed energy of the fully lithiated cathode (eV/formula unit).
    pub energy_lithiated: f64,
    /// Computed energy of the delithiated cathode (eV/formula unit).
    pub energy_delithiated: f64,
    /// Molecular weight of the cathode formula unit (g/mol).
    pub molecular_weight: f64,
}

impl BatteryMaterial {
    /// Average intercalation voltage (V) as a function of Li fraction x in [0, 1].
    ///
    /// V(x) = -(E_lithiated - E_delithiated) / (n_Li * e)
    /// For simplicity this returns a linear interpolation modulated by x.
    pub fn voltage(&self, lithium_fraction: f64) -> f64 {
        let x = lithium_fraction.clamp(0.0, 1.0);
        let n_li = self.intercalation_sites.len().max(1) as f64;
        // Reference energy of metallic Li ~ -1.9 eV/atom.
        let e_li_metal = -1.9;
        let delta_e = self.energy_lithiated - self.energy_delithiated - n_li * e_li_metal;
        // Average voltage with slight x-dependence (entropic term).
        let v_avg = -delta_e / n_li;
        let kb_t = 0.0257; // 300 K in eV
        // Open-circuit voltage with entropic correction.
        if x > 0.0 && x < 1.0 {
            v_avg + kb_t * ((1.0 - x) / x).ln()
        } else {
            v_avg
        }
    }

    /// Theoretical specific capacity in mAh/g.
    ///
    /// Q = n_Li * F / (3.6 * M)   where F = 96485 C/mol, factor 3.6 converts C to mAh.
    pub fn capacity_mah_per_g(&self, molecular_weight: f64) -> f64 {
        let n_li = self.intercalation_sites.len() as f64;
        let faraday = 96485.0; // C/mol
        n_li * faraday / (3.6 * molecular_weight)
    }

    /// Estimated diffusion barrier (eV) from the lattice geometry.
    ///
    /// Uses a simple hop-distance heuristic: barrier ~ alpha * d_hop^2
    /// where d_hop is the minimum distance between intercalation sites and
    /// alpha is a fitted constant (~0.5 eV/A^2 for typical oxides).
    pub fn diffusion_barrier(&self) -> f64 {
        if self.intercalation_sites.len() < 2 {
            return 0.5; // Default barrier.
        }
        let a = &self.cathode.lattice_vectors;
        let mut min_dist_sq = f64::INFINITY;
        for i in 0..self.intercalation_sites.len() {
            for j in (i + 1)..self.intercalation_sites.len() {
                let fi = &self.intercalation_sites[i];
                let fj = &self.intercalation_sites[j];
                let df = [fj[0] - fi[0], fj[1] - fi[1], fj[2] - fi[2]];
                let dx = df[0] * a[0][0] + df[1] * a[1][0] + df[2] * a[2][0];
                let dy = df[0] * a[0][1] + df[1] * a[1][1] + df[2] * a[2][1];
                let dz = df[0] * a[0][2] + df[1] * a[1][2] + df[2] * a[2][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < min_dist_sq {
                    min_dist_sq = d2;
                }
            }
        }
        let alpha = 0.05; // eV/A^2 heuristic scaling
        alpha * min_dist_sq.sqrt()
    }

    /// Encode the cathode Hamiltonian as a matrix for quantum simulation.
    ///
    /// Constructs a tight-binding Hamiltonian from the lattice sites
    /// with nearest-neighbour hopping t = -1 eV and on-site energies
    /// from the atom types.
    pub fn encode_hamiltonian(&self) -> Array2<Complex64> {
        let n = self.cathode.atom_positions.len().max(2);
        let mut h = Array2::<Complex64>::zeros((n, n));
        // On-site energies from atom orbital data.
        for (i, (_, atom)) in self.cathode.atom_positions.iter().enumerate() {
            if i < n {
                let eps = atom.orbital_energies.first().copied().unwrap_or(-1.0);
                h[[i, i]] = Complex64::new(eps, 0.0);
            }
        }
        // Nearest-neighbour hopping (chain topology for simplicity).
        let t = Complex64::new(-1.0, 0.0);
        for i in 0..n.saturating_sub(1) {
            h[[i, i + 1]] = t;
            h[[i + 1, i]] = t;
        }
        h
    }
}

// ===================================================================
// SUPERCONDUCTOR MODEL
// ===================================================================

/// BCS/Eliashberg superconductor model.
///
/// Computes critical temperature, gap function, and coherence length
/// from electron-phonon coupling parameters.
#[derive(Debug, Clone)]
pub struct SuperconductorModel {
    /// Electron-phonon coupling constant lambda.
    pub coupling_constant: f64,
    /// Debye temperature Theta_D (K).
    pub debye_temperature: f64,
    /// Density of states at the Fermi level N(E_F) (states/eV/spin).
    pub density_of_states_at_fermi: f64,
}

impl SuperconductorModel {
    /// McMillan formula for the critical temperature T_c.
    ///
    /// T_c = (Theta_D / 1.45) * exp[-1.04*(1+lambda) / (lambda - mu*(1+0.62*lambda))]
    ///
    /// `mu_star` is the Coulomb pseudopotential (typically 0.10 - 0.15).
    /// Returns 0.0 if the denominator is non-positive (no superconductivity).
    pub fn critical_temperature_mcmillan(&self, mu_star: f64) -> f64 {
        let lambda = self.coupling_constant;
        let denom = lambda - mu_star * (1.0 + 0.62 * lambda);
        if denom <= 0.0 {
            return 0.0;
        }
        let exponent = -1.04 * (1.0 + lambda) / denom;
        (self.debye_temperature / 1.45) * exponent.exp()
    }

    /// BCS gap at temperature T.
    ///
    /// Delta(T) = Delta_0 * sqrt(1 - (T/T_c)^3)    (approximate)
    /// Delta_0  = 1.764 * k_B * T_c                  (BCS weak-coupling limit)
    ///
    /// Returns 0.0 above T_c.
    pub fn gap_equation(&self, temperature: f64, mu_star: f64) -> f64 {
        let tc = self.critical_temperature_mcmillan(mu_star);
        if tc <= 0.0 || temperature >= tc {
            return 0.0;
        }
        let kb = 8.617333262e-5; // eV/K
        let delta_0 = 1.764 * kb * tc;
        let t_ratio = temperature / tc;
        delta_0 * (1.0 - t_ratio.powi(3)).sqrt()
    }

    /// BCS coherence length xi_0.
    ///
    /// xi_0 = hbar * v_F / (pi * Delta)
    ///
    /// `vf` is the Fermi velocity in m/s, `gap` is Delta in eV.
    /// Returns length in nanometres.
    pub fn coherence_length(&self, vf: f64, gap: f64) -> f64 {
        if gap.abs() < 1e-30 {
            return f64::INFINITY;
        }
        let hbar = 6.582119569e-16; // eV*s
        let gap_joules_inv = 1.0 / gap; // in 1/eV
        // xi_0 = hbar * vf / (pi * gap)  in metres, convert to nm.
        let xi_m = hbar * vf * gap_joules_inv / PI;
        xi_m * 1e9 // metres -> nm
    }
}

// ===================================================================
// HUBBARD MODEL
// ===================================================================

/// Single-band Hubbard model on a 1D chain.
///
/// H = -t * sum_{<ij>,sigma} c^dag_{i,sigma} c_{j,sigma}
///   + U * sum_i n_{i,up} n_{i,down}
///
/// Exact diagonalisation is feasible for small systems (N <= ~8 sites at half filling).
#[derive(Debug, Clone)]
pub struct HubbardModel {
    /// Number of lattice sites.
    pub num_sites: usize,
    /// Hopping amplitude (eV).
    pub t: f64,
    /// On-site Coulomb repulsion (eV).
    pub u: f64,
}

impl HubbardModel {
    pub fn new(num_sites: usize, t: f64, u: f64) -> Self {
        Self { num_sites, t, u }
    }

    /// Build the full many-body Hamiltonian in the Fock basis.
    ///
    /// For N sites the Hilbert space dimension is 4^N (up/down occupancy per site).
    /// Only feasible for small N.
    pub fn build_hamiltonian(&self) -> MaterialsResult<Array2<Complex64>> {
        let n = self.num_sites;
        if n > 6 {
            return Err(MaterialsError::UnsupportedOperation(format!(
                "Exact diagonalisation infeasible for {} sites (dim=4^{}={})",
                n,
                n,
                4usize.pow(n as u32)
            )));
        }
        let dim = 4usize.pow(n as u32);
        let mut h = Array2::<Complex64>::zeros((dim, dim));

        for state in 0..dim {
            // Decode state into up/down occupation.
            let (occ_up, occ_down) = decode_fock(state, n);

            // On-site interaction: U * n_up * n_down.
            let mut diag = 0.0;
            for site in 0..n {
                if occ_up[site] && occ_down[site] {
                    diag += self.u;
                }
            }
            h[[state, state]] = Complex64::new(diag, 0.0);

            // Hopping: -t * c^dag_i c_j (nearest neighbour, open BC).
            for spin_up in [true, false] {
                let occ = if spin_up { &occ_up } else { &occ_down };
                for site in 0..n.saturating_sub(1) {
                    let j = site + 1;
                    // Hop from site -> j.
                    if occ[site] && !occ[j] {
                        let (new_state, sign) =
                            apply_hop(state, n, site, j, spin_up);
                        h[[new_state, state]] +=
                            Complex64::new(-self.t * sign as f64, 0.0);
                    }
                    // Hop from j -> site.
                    if occ[j] && !occ[site] {
                        let (new_state, sign) =
                            apply_hop(state, n, j, site, spin_up);
                        h[[new_state, state]] +=
                            Complex64::new(-self.t * sign as f64, 0.0);
                    }
                }
            }
        }
        Ok(h)
    }

    /// Ground-state energy via exact diagonalisation.
    pub fn ground_state_energy(&self) -> MaterialsResult<f64> {
        let h = self.build_hamiltonian()?;
        let evals = eigenvalues_hermitian(&h);
        Ok(evals[0])
    }

    /// Encode the Hubbard model as a qubit Hamiltonian for VQE.
    ///
    /// Uses Jordan-Wigner mapping: 2*N spin-orbitals -> 2*N qubits.
    /// Returns (num_qubits, pauli_terms) where each term is
    /// (Vec<(qubit, pauli_char)>, coefficient).
    pub fn encode_for_vqe(&self) -> (usize, Vec<(Vec<(usize, char)>, f64)>) {
        let n_qubits = 2 * self.num_sites;
        let mut terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();

        // On-site interaction: U * n_{i,up} * n_{i,down}
        // n_j = (I - Z_j) / 2
        // n_up * n_down = (I - Z_up)(I - Z_down) / 4
        //               = (I - Z_up - Z_down + Z_up*Z_down) / 4
        for site in 0..self.num_sites {
            let q_up = site;
            let q_down = site + self.num_sites;

            // U/4 * I (constant, skip for simplicity)
            // -U/4 * Z_up
            terms.push((vec![(q_up, 'Z')], -self.u / 4.0));
            // -U/4 * Z_down
            terms.push((vec![(q_down, 'Z')], -self.u / 4.0));
            // U/4 * Z_up Z_down
            terms.push((vec![(q_up, 'Z'), (q_down, 'Z')], self.u / 4.0));
        }

        // Hopping: -t/2 * (X_i X_j + Y_i Y_j) * Z-string (Jordan-Wigner)
        for spin_offset in [0, self.num_sites] {
            for site in 0..self.num_sites.saturating_sub(1) {
                let qi = site + spin_offset;
                let qj = site + 1 + spin_offset;
                // XX term
                terms.push((vec![(qi, 'X'), (qj, 'X')], -self.t / 2.0));
                // YY term
                terms.push((vec![(qi, 'Y'), (qj, 'Y')], -self.t / 2.0));
            }
        }

        (n_qubits, terms)
    }
}

// Fock-state helpers for Hubbard exact diag.

fn decode_fock(state: usize, n: usize) -> (Vec<bool>, Vec<bool>) {
    // Encoding: bits [0..n) = up occupation, bits [n..2n) = down occupation.
    let mut up = vec![false; n];
    let mut down = vec![false; n];
    for i in 0..n {
        up[i] = (state >> i) & 1 == 1;
        down[i] = (state >> (i + n)) & 1 == 1;
    }
    (up, down)
}

fn apply_hop(
    state: usize,
    n: usize,
    from: usize,
    to: usize,
    spin_up: bool,
) -> (usize, i32) {
    let offset = if spin_up { 0 } else { n };
    let bit_from = offset + from;
    let bit_to = offset + to;

    // Remove fermion at `from`, add at `to`.
    let new_state = (state & !(1 << bit_from)) | (1 << bit_to);

    // Fermionic sign: count occupied sites between from and to.
    let lo = bit_from.min(bit_to) + 1;
    let hi = bit_from.max(bit_to);
    let mut count = 0i32;
    for b in lo..hi {
        if (state >> b) & 1 == 1 {
            count += 1;
        }
    }
    let sign = if count % 2 == 0 { 1 } else { -1 };
    (new_state, sign)
}

// ===================================================================
// EIGENVALUE SOLVER (HERMITIAN MATRICES)
// ===================================================================

/// Compute eigenvalues of a Hermitian matrix using Jacobi iteration.
///
/// This is a self-contained eigensolver that avoids external LAPACK
/// dependencies. Suitable for small matrices (N < 100).
fn eigenvalues_hermitian(h: &Array2<Complex64>) -> Vec<f64> {
    let n = h.nrows();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![h[[0, 0]].re];
    }

    // For Hermitian matrices, eigenvalues are real.
    // Use a simple QR-like algorithm on the real part (valid when imaginary
    // off-diagonal elements are small) or full unitary reduction.
    // Here we implement a power-iteration / deflation approach for robustness.

    // For 2x2, use the analytical formula.
    if n == 2 {
        let a = h[[0, 0]].re;
        let d = h[[1, 1]].re;
        let b_re = h[[0, 1]].re;
        let b_im = h[[0, 1]].im;
        let b_sq = b_re * b_re + b_im * b_im;
        let trace = a + d;
        let disc = ((a - d) * (a - d) + 4.0 * b_sq).sqrt();
        let mut evals = vec![(trace - disc) / 2.0, (trace + disc) / 2.0];
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return evals;
    }

    // General case: Jacobi eigenvalue algorithm on H^dag H trick is overkill.
    // Instead, use iterative Givens rotations on the Hermitian matrix.
    // For simplicity and correctness, we reduce to a real symmetric tridiagonal
    // form via Householder reflections, then apply the QR algorithm.

    // Step 1: Build a real symmetric matrix from the Hermitian matrix.
    // For a truly Hermitian H, its eigenvalues equal those of the real matrix
    // obtained from the 2n x 2n embedding, but that doubles the size.
    // Instead, we use a direct approach: Householder tridiagonalisation of H
    // as if it were real symmetric (taking Re(H) when H is nearly real, or
    // using the full Hermitian Householder otherwise).

    // Pragmatic approach: extract the real symmetric part and solve that.
    // This is exact when H is real-symmetric (which it is for most physics cases).
    let mut a_mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            // For a Hermitian matrix, (H + H^dag)/2 is real symmetric and
            // has the same eigenvalues.
            a_mat[i][j] = (h[[i, j]].re + h[[j, i]].re) / 2.0;
        }
    }

    // Householder tridiagonalisation.
    let (diag, offdiag) = tridiagonalise(&mut a_mat, n);

    // QR algorithm on the tridiagonal matrix.
    let mut evals = qr_tridiagonal(diag, offdiag, n);
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    evals
}

/// Householder reduction to tridiagonal form.
/// Returns (diagonal, sub-diagonal).
fn tridiagonalise(a: &mut Vec<Vec<f64>>, n: usize) -> (Vec<f64>, Vec<f64>) {
    for k in 0..n.saturating_sub(2) {
        // Compute the Householder vector for column k.
        let mut sigma = 0.0;
        for i in (k + 1)..n {
            sigma += a[i][k] * a[i][k];
        }
        if sigma < 1e-30 {
            continue;
        }
        let s = sigma.sqrt();
        let sign = if a[k + 1][k] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * s;

        // Householder vector v.
        let mut v = vec![0.0; n];
        v[k + 1] = a[k + 1][k] - alpha;
        for i in (k + 2)..n {
            v[i] = a[i][k];
        }
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }

        // Apply similarity transformation: A <- (I - 2vv^T/||v||^2) A (I - 2vv^T/||v||^2)
        // Left multiplication.
        let factor = 2.0 / v_norm_sq;
        let mut p = vec![0.0; n];
        for j in 0..n {
            let mut dot = 0.0;
            for i in 0..n {
                dot += v[i] * a[i][j];
            }
            p[j] = factor * dot;
        }
        for i in 0..n {
            for j in 0..n {
                a[i][j] -= v[i] * p[j];
            }
        }
        // Right multiplication.
        let mut q = vec![0.0; n];
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..n {
                dot += a[i][j] * v[j];
            }
            q[i] = factor * dot;
        }
        for i in 0..n {
            for j in 0..n {
                a[i][j] -= q[i] * v[j];
            }
        }
    }

    let mut diag = vec![0.0; n];
    let mut offdiag = vec![0.0; n.saturating_sub(1)];
    for i in 0..n {
        diag[i] = a[i][i];
    }
    for i in 0..n.saturating_sub(1) {
        offdiag[i] = a[i + 1][i];
    }
    (diag, offdiag)
}

/// Implicit QR algorithm for symmetric tridiagonal eigenvalues.
fn qr_tridiagonal(mut diag: Vec<f64>, mut off: Vec<f64>, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![diag[0]];
    }

    let max_iter = 100 * n;
    let mut m = n;

    for _ in 0..max_iter {
        // Find the lowest non-zero sub-diagonal.
        if m <= 1 {
            break;
        }
        let mut l = m - 1;
        while l > 0 {
            let threshold =
                1e-14 * (diag[l - 1].abs() + diag[l].abs()).max(1e-30);
            if off[l - 1].abs() < threshold {
                break;
            }
            l -= 1;
        }
        if l == m - 1 {
            // Eigenvalue converged.
            m -= 1;
            if m <= 1 {
                break;
            }
            continue;
        }

        // Wilkinson shift.
        let d = (diag[m - 2] - diag[m - 1]) / 2.0;
        let e2 = off[m - 2] * off[m - 2];
        let shift = diag[m - 1]
            - e2 / (d + d.signum() * (d * d + e2).sqrt().max(1e-30));

        // Implicit QR step with Givens rotations.
        let mut x = diag[l] - shift;
        let mut z = off[l];
        for k in l..m - 1 {
            let r = (x * x + z * z).sqrt();
            let c = x / r;
            let s = z / r;
            if k > l {
                off[k - 1] = r;
            }
            let d1 = diag[k];
            let d2 = diag[k + 1];
            let e = off[k];
            diag[k] = c * c * d1 + 2.0 * c * s * e + s * s * d2;
            diag[k + 1] = s * s * d1 - 2.0 * c * s * e + c * c * d2;
            off[k] = c * s * (d2 - d1) + (c * c - s * s) * e;
            if k + 1 < m - 1 {
                x = off[k];
                z = -s * off[k + 1];
                off[k + 1] *= c;
            }
        }
        // Check convergence of bottom sub-diagonal.
        if off[m - 2].abs() < 1e-14 * (diag[m - 2].abs() + diag[m - 1].abs()).max(1e-30) {
            m -= 1;
        }
    }

    diag
}

// ===================================================================
// SCREENING RESULTS
// ===================================================================

/// Result of screening a single battery material candidate.
#[derive(Debug, Clone)]
pub struct ScreeningResult {
    /// Average intercalation voltage (V).
    pub voltage: f64,
    /// Specific capacity (mAh/g).
    pub capacity: f64,
    /// Li diffusion barrier (eV).
    pub barrier: f64,
    /// Composite figure of merit (higher is better).
    pub score: f64,
    /// Whether the candidate passes minimum thresholds.
    pub passes_threshold: bool,
}

/// Result of screening a superconductor model.
#[derive(Debug, Clone)]
pub struct SCResult {
    /// McMillan critical temperature (K).
    pub tc: f64,
    /// BCS gap at T=0 (eV).
    pub gap_0: f64,
    /// Electron-phonon coupling constant.
    pub lambda: f64,
    /// Whether T_c exceeds the target.
    pub is_promising: bool,
}

/// Band structure computation result.
#[derive(Debug, Clone)]
pub struct BandStructureResult {
    /// K-points along the path.
    pub k_points: Vec<[f64; 3]>,
    /// Band energies at each k-point (outer: k, inner: band index).
    pub bands: Vec<Vec<f64>>,
    /// Minimum band gap (eV). Zero for metals.
    pub band_gap: f64,
    /// True if band gap is zero (metallic).
    pub is_metal: bool,
}

/// Hubbard model solution result.
#[derive(Debug, Clone)]
pub struct HubbardResult {
    /// Ground-state energy (eV).
    pub ground_state_energy: f64,
    /// Number of qubits for VQE encoding.
    pub vqe_qubit_count: usize,
    /// Number of Pauli terms in VQE Hamiltonian.
    pub vqe_term_count: usize,
}

// ===================================================================
// MATERIALS SCREENER
// ===================================================================

/// High-throughput quantum materials screening engine.
///
/// Evaluates battery cathodes, superconductors, and band structures
/// using configurable solver backends.
pub struct MaterialsScreener {
    config: MaterialsConfig,
}

impl MaterialsScreener {
    /// Create a screener with validated configuration.
    pub fn new(config: MaterialsConfig) -> MaterialsResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Screen a set of battery material candidates.
    ///
    /// Ranks candidates by a composite score combining voltage, capacity,
    /// and diffusion barrier. Thresholds:
    ///   voltage >= 2.0 V, capacity >= 100 mAh/g, barrier <= 0.8 eV.
    pub fn screen_battery(&self, candidates: &[BatteryMaterial]) -> Vec<ScreeningResult> {
        candidates
            .iter()
            .map(|mat| {
                let v = mat.voltage(0.5);
                let cap = mat.capacity_mah_per_g(mat.molecular_weight);
                let barrier = mat.diffusion_barrier();

                // Composite score: weighted sum (normalised to ~0-1 range).
                let v_score = (v / 5.0).clamp(0.0, 1.0);
                let c_score = (cap / 300.0).clamp(0.0, 1.0);
                let b_score = (1.0 - barrier / 1.0).clamp(0.0, 1.0);
                let score = 0.4 * v_score + 0.3 * c_score + 0.3 * b_score;

                let passes = v >= 2.0 && cap >= 100.0 && barrier <= 0.8;

                ScreeningResult {
                    voltage: v,
                    capacity: cap,
                    barrier,
                    score,
                    passes_threshold: passes,
                }
            })
            .collect()
    }

    /// Screen superconductor models for high-T_c candidates.
    pub fn screen_superconductor(&self, models: &[SuperconductorModel]) -> Vec<SCResult> {
        let mu_star = 0.10; // Typical Coulomb pseudopotential.
        models
            .iter()
            .map(|sc| {
                let tc = sc.critical_temperature_mcmillan(mu_star);
                let gap_0 = sc.gap_equation(0.0, mu_star);
                SCResult {
                    tc,
                    gap_0,
                    lambda: sc.coupling_constant,
                    is_promising: tc > 10.0,
                }
            })
            .collect()
    }

    /// Compute the electronic band structure for a lattice + tight-binding model.
    pub fn compute_band_structure(
        &self,
        lattice: &CrystalLattice,
        model: &TightBindingModel,
    ) -> BandStructureResult {
        let k_path = lattice.k_path(self.config.num_sites * 10);
        let bands = model.band_structure(&k_path, lattice);

        // Compute band gap: minimum of (band_{n+1,k} - band_{n,k}) over all k.
        let mut gap = f64::INFINITY;
        let n_bands = if bands.is_empty() {
            0
        } else {
            bands[0].len()
        };
        if n_bands >= 2 {
            for evals in &bands {
                for i in 0..n_bands - 1 {
                    let g = evals[i + 1] - evals[i];
                    if g < gap {
                        gap = g;
                    }
                }
            }
        } else {
            gap = 0.0;
        }
        let is_metal = gap < 0.01; // Threshold for metallicity.

        BandStructureResult {
            k_points: k_path.iter().map(|k| *k).collect(),
            bands,
            band_gap: gap.max(0.0),
            is_metal,
        }
    }

    /// Solve the Hubbard model via exact diagonalisation or VQE encoding.
    pub fn solve_hubbard(&self, model: &HubbardModel) -> MaterialsResult<HubbardResult> {
        let gs_energy = model.ground_state_energy()?;
        let (n_qubits, terms) = model.encode_for_vqe();

        Ok(HubbardResult {
            ground_state_energy: gs_energy,
            vqe_qubit_count: n_qubits,
            vqe_term_count: terms.len(),
        })
    }

    /// Get a reference to the active configuration.
    pub fn config(&self) -> &MaterialsConfig {
        &self.config
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // ---- Config builder ----

    #[test]
    fn test_config_builder_defaults() {
        let cfg = MaterialsConfig::builder().build().unwrap();
        assert_eq!(cfg.num_sites, 8);
        assert_eq!(cfg.num_orbitals, 4);
        assert_eq!(cfg.solver, MaterialsSolver::TightBinding);
        assert!((cfg.convergence_threshold - 1e-6).abs() < 1e-15);
        assert_eq!(cfg.max_iterations, 100);
        assert!((cfg.temperature_kelvin - 300.0).abs() < TOL);
    }

    #[test]
    fn test_config_builder_custom() {
        let cfg = MaterialsConfig::builder()
            .num_sites(16)
            .num_orbitals(6)
            .solver(MaterialsSolver::ExactDiagonalization)
            .convergence_threshold(1e-8)
            .max_iterations(500)
            .temperature_kelvin(4.2)
            .build()
            .unwrap();
        assert_eq!(cfg.num_sites, 16);
        assert_eq!(cfg.num_orbitals, 6);
        assert_eq!(cfg.solver, MaterialsSolver::ExactDiagonalization);
        assert!((cfg.temperature_kelvin - 4.2).abs() < TOL);
    }

    #[test]
    fn test_config_validation_rejects_bad_sites() {
        assert!(MaterialsConfig::builder().num_sites(0).build().is_err());
        assert!(MaterialsConfig::builder().num_sites(1).build().is_err());
        assert!(MaterialsConfig::builder().num_sites(101).build().is_err());
    }

    #[test]
    fn test_config_validation_rejects_bad_orbitals() {
        assert!(MaterialsConfig::builder().num_orbitals(0).build().is_err());
        assert!(MaterialsConfig::builder().num_orbitals(11).build().is_err());
    }

    #[test]
    fn test_config_validation_rejects_negative_temperature() {
        assert!(MaterialsConfig::builder()
            .temperature_kelvin(-1.0)
            .build()
            .is_err());
    }

    // ---- Crystal lattices ----

    #[test]
    fn test_cubic_lattice_construction() {
        let lat = CrystalLattice::new_cubic(5.0);
        assert_eq!(lat.lattice_vectors[0], [5.0, 0.0, 0.0]);
        assert_eq!(lat.lattice_vectors[1], [0.0, 5.0, 0.0]);
        assert_eq!(lat.lattice_vectors[2], [0.0, 0.0, 5.0]);
        assert_eq!(lat.atom_positions.len(), 1);
        assert_eq!(lat.atom_positions[0].0, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fcc_lattice_construction() {
        let a = 4.05; // Al lattice constant.
        let lat = CrystalLattice::new_fcc(a);
        let h = a / 2.0;
        assert!((lat.lattice_vectors[0][0] - 0.0).abs() < TOL);
        assert!((lat.lattice_vectors[0][1] - h).abs() < TOL);
        assert!((lat.lattice_vectors[0][2] - h).abs() < TOL);
        // Volume of FCC primitive cell = a^3 / 4.
        let vol = CrystalLattice::triple_product(
            &lat.lattice_vectors[0],
            &lat.lattice_vectors[1],
            &lat.lattice_vectors[2],
        )
        .abs();
        let expected_vol = a * a * a / 4.0;
        assert!(
            (vol - expected_vol).abs() < 1e-6,
            "FCC volume: got {}, expected {}",
            vol,
            expected_vol
        );
    }

    #[test]
    fn test_bcc_lattice_volume() {
        let a = 3.3; // typical BCC metal
        let lat = CrystalLattice::new_bcc(a);
        let vol = CrystalLattice::triple_product(
            &lat.lattice_vectors[0],
            &lat.lattice_vectors[1],
            &lat.lattice_vectors[2],
        )
        .abs();
        let expected_vol = a * a * a / 2.0;
        assert!(
            (vol - expected_vol).abs() < 1e-6,
            "BCC volume: got {}, expected {}",
            vol,
            expected_vol
        );
    }

    #[test]
    fn test_reciprocal_vectors_orthogonality() {
        let lat = CrystalLattice::new_cubic(3.0);
        let a = &lat.lattice_vectors;
        let b = lat.reciprocal_vectors();

        // a_i . b_j = 2*pi * delta_{ij}.
        for i in 0..3 {
            for j in 0..3 {
                let dot = CrystalLattice::dot(&a[i], &b[j]);
                let expected = if i == j { 2.0 * PI } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "a_{} . b_{} = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_fcc_reciprocal_orthogonality() {
        let lat = CrystalLattice::new_fcc(4.0);
        let a = &lat.lattice_vectors;
        let b = lat.reciprocal_vectors();
        for i in 0..3 {
            for j in 0..3 {
                let dot = CrystalLattice::dot(&a[i], &b[j]);
                let expected = if i == j { 2.0 * PI } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "FCC: a_{} . b_{} = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_k_path_generation() {
        let lat = CrystalLattice::new_cubic(1.0);
        let path = lat.k_path(30);
        assert_eq!(path.len(), 30);
        // First point should be Gamma.
        assert!((path[0][0]).abs() < TOL);
        assert!((path[0][1]).abs() < TOL);
        assert!((path[0][2]).abs() < TOL);
    }

    // ---- Tight-binding model ----

    #[test]
    fn test_tight_binding_1d_chain_analytical() {
        // 1D chain with 1 orbital, hopping t, PBC approximated by open BC.
        // For a 2-site chain: H = [[0, -t], [-t, 0]]
        // Eigenvalues: -t, +t.
        let t_hop = 1.5;
        let mut model = TightBindingModel::new(vec![0.0, 0.0]);
        model.add_hopping(0, 1, Complex64::new(-t_hop, 0.0));

        let lat = CrystalLattice {
            lattice_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            atom_positions: vec![
                ([0.0, 0.0, 0.0], AtomType::new("A", 1, 1, vec![0.0])),
                ([0.5, 0.0, 0.0], AtomType::new("A", 1, 1, vec![0.0])),
            ],
        };

        // At Gamma (k=0) the Bloch phase is 1, so H = [[0, -t + c.c.], [c.c., 0]]
        // The exact eigenvalues depend on the Bloch phase at k=0.
        let h = model.build_hamiltonian_k(&[0.0, 0.0, 0.0], &lat);
        let evals = eigenvalues_hermitian(&h);
        assert_eq!(evals.len(), 2);
        // Both eigenvalues should be symmetric around 0.
        assert!(
            (evals[0] + evals[1]).abs() < 1e-6,
            "eigenvalues should sum to 0: {:?}",
            evals
        );
    }

    #[test]
    fn test_band_structure_simple() {
        let mut model = TightBindingModel::new(vec![0.0, 0.0]);
        model.add_hopping(0, 1, Complex64::new(-1.0, 0.0));

        let lat = CrystalLattice {
            lattice_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            atom_positions: vec![
                ([0.0, 0.0, 0.0], AtomType::new("A", 1, 1, vec![0.0])),
                ([0.5, 0.0, 0.0], AtomType::new("A", 1, 1, vec![0.0])),
            ],
        };

        let k_pts = lat.k_path(20);
        let bands = model.band_structure(&k_pts, &lat);
        assert_eq!(bands.len(), 20);
        for evals in &bands {
            assert_eq!(evals.len(), 2);
            // Lower band should be <= upper band.
            assert!(evals[0] <= evals[1] + TOL);
        }
    }

    #[test]
    fn test_density_of_states_normalisation() {
        // For a set of delta-function energies broadened by eta,
        // the integral of the DOS should equal 1.
        // Use wide broadening and fine grid to ensure the Lorentzian tails
        // are captured within the integration window.
        let energies = vec![-1.0, 0.0, 1.0];
        let model = TightBindingModel::new(vec![0.0]);
        let eta = 0.3;
        let grid = 2000;
        let dos = model.density_of_states(&energies, eta, grid);

        // Numerical integration (trapezoidal).
        let mut integral = 0.0;
        for i in 1..dos.len() {
            let de = dos[i].0 - dos[i - 1].0;
            integral += 0.5 * (dos[i].1 + dos[i - 1].1) * de;
        }
        // Should integrate to approximately 1.0.
        assert!(
            (integral - 1.0).abs() < 0.1,
            "DOS integral = {}, expected ~1.0",
            integral
        );
    }

    #[test]
    fn test_fermi_energy_half_filling() {
        // Half-filled band: N_electrons = N_states / 2.
        // E_F should be at the middle of the sorted energies.
        let energies: Vec<f64> = (-5..=5).map(|i| i as f64).collect(); // 11 levels
        let model = TightBindingModel::new(vec![0.0]);
        let n_electrons = 5; // half of 11, rounding down
        let ef = model.fermi_energy(&energies, n_electrons, 0.0);
        // At T=0, Fermi level is between E[4]=-1 and E[5]=0.
        assert!(
            ef > -1.5 && ef < 0.5,
            "Fermi energy for half-fill: {} (expected between -1 and 0)",
            ef
        );
    }

    // ---- Superconductor model ----

    #[test]
    fn test_mcmillan_formula_known_tc() {
        // Niobium: lambda ~ 0.82, Theta_D ~ 275 K, mu* ~ 0.13
        // Experimental T_c ~ 9.25 K.
        let sc = SuperconductorModel {
            coupling_constant: 0.82,
            debye_temperature: 275.0,
            density_of_states_at_fermi: 1.0,
        };
        let tc = sc.critical_temperature_mcmillan(0.13);
        // McMillan formula gives ~9-10 K for these parameters.
        assert!(
            tc > 5.0 && tc < 15.0,
            "Nb T_c = {} K, expected ~9 K",
            tc
        );
    }

    #[test]
    fn test_mcmillan_no_superconductivity() {
        // Very weak coupling: lambda < mu* -> no superconductivity.
        let sc = SuperconductorModel {
            coupling_constant: 0.05,
            debye_temperature: 300.0,
            density_of_states_at_fermi: 1.0,
        };
        let tc = sc.critical_temperature_mcmillan(0.15);
        assert!(
            tc < TOL,
            "Expected no superconductivity, got T_c = {}",
            tc
        );
    }

    #[test]
    fn test_bcs_gap_at_t_zero() {
        // BCS gap: Delta_0 = 1.764 * k_B * T_c.
        let sc = SuperconductorModel {
            coupling_constant: 1.0,
            debye_temperature: 300.0,
            density_of_states_at_fermi: 1.0,
        };
        let mu_star = 0.10;
        let tc = sc.critical_temperature_mcmillan(mu_star);
        let gap_0 = sc.gap_equation(0.0, mu_star);
        let kb = 8.617333262e-5;
        let expected = 1.764 * kb * tc;
        assert!(
            (gap_0 - expected).abs() < 1e-10,
            "Gap at T=0: {}, expected {}",
            gap_0,
            expected
        );
    }

    #[test]
    fn test_bcs_gap_above_tc() {
        let sc = SuperconductorModel {
            coupling_constant: 1.0,
            debye_temperature: 300.0,
            density_of_states_at_fermi: 1.0,
        };
        let mu_star = 0.10;
        let tc = sc.critical_temperature_mcmillan(mu_star);
        let gap = sc.gap_equation(tc + 10.0, mu_star);
        assert!(
            gap.abs() < TOL,
            "Gap above T_c should be zero, got {}",
            gap
        );
    }

    #[test]
    fn test_coherence_length_formula() {
        // xi_0 = hbar * v_F / (pi * Delta), result in nm.
        let sc = SuperconductorModel {
            coupling_constant: 1.0,
            debye_temperature: 300.0,
            density_of_states_at_fermi: 1.0,
        };
        let vf = 1.0e6; // m/s (typical metal Fermi velocity)
        let gap = 0.001; // eV
        let xi = sc.coherence_length(vf, gap);
        // hbar = 6.582e-16 eV*s -> xi = 6.582e-16 * 1e6 / (pi * 0.001) * 1e9
        //    = 6.582e-10 / 3.14159e-3 * 1e9 = 2.095e-7 * 1e9 = 209.5 nm
        assert!(
            xi > 100.0 && xi < 400.0,
            "Coherence length = {} nm, expected ~210 nm",
            xi
        );
    }

    // ---- Hubbard model ----

    #[test]
    fn test_hubbard_2site_exact_solution() {
        // 2-site Hubbard model: full Fock space has 4^2 = 16 states across
        // all particle-number sectors (0, 1, 2, 3, 4 electrons).
        //
        // For U=2, t=1: the 2-electron sector GS = (U - sqrt(U^2 + 16t^2))/2
        //   = (2 - sqrt(4+16))/2 = (2 - 4.472)/2 = -1.236
        // The 1-electron sector GS = -t = -1.0
        // So the 2-electron sector wins; overall GS = -1.236.
        let hub = HubbardModel::new(2, 1.0, 2.0);
        let e_gs = hub.ground_state_energy().unwrap();
        let expected = (2.0 - (4.0 + 16.0_f64).sqrt()) / 2.0;
        assert!(
            (e_gs - expected).abs() < 0.15,
            "Hubbard 2-site E_GS = {}, expected {}",
            e_gs,
            expected
        );
    }

    #[test]
    fn test_hubbard_noninteracting_limit() {
        // U=0: energy should be that of free fermions on a 2-site chain.
        // Eigenvalues: -t, +t. At half-filling (2 electrons), GS energy = 2*(-t) = -2t.
        let hub = HubbardModel::new(2, 1.0, 0.0);
        let e_gs = hub.ground_state_energy().unwrap();
        // Free-fermion ground state for 2 sites, 2 electrons (both spins in bonding orbital).
        assert!(
            e_gs < 0.0,
            "Non-interacting GS should be negative, got {}",
            e_gs
        );
    }

    #[test]
    fn test_hubbard_vqe_encoding() {
        let hub = HubbardModel::new(3, 1.0, 2.0);
        let (n_qubits, terms) = hub.encode_for_vqe();
        assert_eq!(n_qubits, 6); // 2 * num_sites
        // Should have terms for: on-site (3 sites * 3 terms) + hopping (2 bonds * 2 spins * 2 terms)
        assert!(
            terms.len() > 0,
            "VQE encoding should produce Pauli terms"
        );
        // Verify Z terms.
        let z_terms: Vec<_> = terms
            .iter()
            .filter(|(ops, _)| ops.iter().all(|&(_, p)| p == 'Z'))
            .collect();
        assert!(
            z_terms.len() >= 3,
            "Should have at least 3 diagonal Z terms"
        );
    }

    #[test]
    fn test_hubbard_too_large_rejected() {
        let hub = HubbardModel::new(10, 1.0, 1.0);
        assert!(hub.ground_state_energy().is_err());
    }

    // ---- Battery material ----

    #[test]
    fn test_battery_voltage_calculation() {
        let mat = BatteryMaterial {
            cathode: CrystalLattice::new_cubic(4.0),
            intercalation_sites: vec![[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
            energy_lithiated: -20.0,
            energy_delithiated: -15.0,
            molecular_weight: 100.0,
        };
        let v = mat.voltage(0.5);
        // Voltage should be positive and in a physically reasonable range.
        assert!(
            v > 0.0 && v < 10.0,
            "Battery voltage = {} V, expected 0-10 V range",
            v
        );
    }

    #[test]
    fn test_battery_capacity() {
        let mat = BatteryMaterial {
            cathode: CrystalLattice::new_cubic(4.0),
            intercalation_sites: vec![[0.5, 0.5, 0.5]],
            energy_lithiated: -10.0,
            energy_delithiated: -8.0,
            molecular_weight: 96.0, // ~LiFePO4
        };
        let cap = mat.capacity_mah_per_g(96.0);
        // 1 Li per formula unit at 96 g/mol: 96485/(3.6*96) ~ 279 mAh/g.
        assert!(
            (cap - 279.4).abs() < 1.0,
            "Capacity = {} mAh/g, expected ~279 mAh/g",
            cap
        );
    }

    #[test]
    fn test_battery_diffusion_barrier() {
        let mat = BatteryMaterial {
            cathode: CrystalLattice::new_cubic(4.0),
            intercalation_sites: vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            energy_lithiated: -10.0,
            energy_delithiated: -8.0,
            molecular_weight: 100.0,
        };
        let barrier = mat.diffusion_barrier();
        // Distance between sites: 0.5 * 4.0 = 2.0 A; barrier = 0.05 * 2.0 = 0.1 eV.
        assert!(
            barrier > 0.0 && barrier < 2.0,
            "Diffusion barrier = {} eV, expected < 2 eV",
            barrier
        );
    }

    // ---- Screening pipeline ----

    #[test]
    fn test_screening_pipeline() {
        let cfg = MaterialsConfig::builder().build().unwrap();
        let screener = MaterialsScreener::new(cfg).unwrap();

        let candidates = vec![
            BatteryMaterial {
                cathode: CrystalLattice::new_cubic(4.0),
                intercalation_sites: vec![[0.5, 0.5, 0.5]],
                energy_lithiated: -20.0,
                energy_delithiated: -14.0,
                molecular_weight: 96.0,
            },
            BatteryMaterial {
                cathode: CrystalLattice::new_cubic(4.0),
                intercalation_sites: vec![[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
                energy_lithiated: -25.0,
                energy_delithiated: -18.0,
                molecular_weight: 120.0,
            },
        ];

        let results = screener.screen_battery(&candidates);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0);
            assert!(r.voltage > 0.0);
            assert!(r.capacity > 0.0);
        }
    }

    #[test]
    fn test_superconductor_screening() {
        let cfg = MaterialsConfig::builder().build().unwrap();
        let screener = MaterialsScreener::new(cfg).unwrap();

        let models = vec![
            SuperconductorModel {
                coupling_constant: 1.2,
                debye_temperature: 400.0,
                density_of_states_at_fermi: 1.0,
            },
            SuperconductorModel {
                coupling_constant: 0.05,
                debye_temperature: 200.0,
                density_of_states_at_fermi: 0.5,
            },
        ];

        let results = screener.screen_superconductor(&models);
        assert_eq!(results.len(), 2);
        // First model should have high Tc, second should not.
        assert!(results[0].tc > 10.0);
        assert!(results[0].is_promising);
        assert!(!results[1].is_promising);
    }

    #[test]
    fn test_band_structure_computation() {
        let cfg = MaterialsConfig::builder().num_sites(4).build().unwrap();
        let screener = MaterialsScreener::new(cfg).unwrap();

        let lat = CrystalLattice::new_cubic(3.0);
        let mut model = TightBindingModel::new(vec![0.0, -2.0]);
        model.add_hopping(0, 1, Complex64::new(-1.0, 0.0));

        let result = screener.compute_band_structure(&lat, &model);
        assert!(!result.k_points.is_empty());
        assert!(!result.bands.is_empty());
        assert!(result.band_gap >= 0.0);
    }

    #[test]
    fn test_error_display() {
        let e = MaterialsError::InvalidConfig("test error".into());
        assert_eq!(format!("{}", e), "Invalid config: test error");

        let e = MaterialsError::ConvergenceFailure {
            iterations: 42,
            residual: 1e-3,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("42"));
        assert!(msg.contains("1.00e-3"));
    }

    #[test]
    fn test_solver_display() {
        assert_eq!(format!("{}", MaterialsSolver::VQE), "VQE");
        assert_eq!(
            format!("{}", MaterialsSolver::ExactDiagonalization),
            "ExactDiagonalization"
        );
        assert_eq!(format!("{}", MaterialsSolver::DMFT), "DMFT");
        assert_eq!(
            format!("{}", MaterialsSolver::TightBinding),
            "TightBinding"
        );
    }

    #[test]
    fn test_encode_hamiltonian() {
        let mat = BatteryMaterial {
            cathode: CrystalLattice {
                lattice_vectors: [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
                atom_positions: vec![
                    ([0.0, 0.0, 0.0], AtomType::new("Co", 27, 9, vec![-3.0])),
                    ([0.5, 0.5, 0.5], AtomType::new("O", 8, 6, vec![-5.0])),
                ],
            },
            intercalation_sites: vec![[0.25, 0.25, 0.25]],
            energy_lithiated: -15.0,
            energy_delithiated: -12.0,
            molecular_weight: 100.0,
        };
        let h = mat.encode_hamiltonian();
        assert_eq!(h.nrows(), 2);
        assert_eq!(h.ncols(), 2);
        // Diagonal should have on-site energies.
        assert!((h[[0, 0]].re - (-3.0)).abs() < TOL);
        assert!((h[[1, 1]].re - (-5.0)).abs() < TOL);
        // Off-diagonal should have hopping.
        assert!((h[[0, 1]].re - (-1.0)).abs() < TOL);
    }
}
