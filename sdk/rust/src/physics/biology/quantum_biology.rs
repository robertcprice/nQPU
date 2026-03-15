//! Quantum Biology Module
//!
//! Simulates quantum mechanical effects in biological systems -- a world first
//! for any quantum simulator. Five major phenomena are modelled:
//!
//! 1. **FMO Photosynthesis** -- Coherent energy transfer through the 7-site
//!    Fenna-Matthews-Olson complex in green sulfur bacteria, solved via the
//!    Lindblad master equation at 300 K.
//!
//! 2. **Enzyme Quantum Tunneling** -- WKB tunneling of protons through
//!    enzyme active-site barriers, with kinetic isotope effect (KIE) and
//!    temperature-dependent rate calculations.
//!
//! 3. **Avian Magnetoreception** -- Radical-pair mechanism in cryptochrome,
//!    modelling singlet-triplet interconversion under the geomagnetic field
//!    (~50 uT) with hyperfine interactions.
//!
//! 4. **DNA Quantum Mutations** -- Spontaneous tautomeric shifts via proton
//!    tunneling along hydrogen bonds in A-T and G-C base pairs, modelled as
//!    asymmetric double-well potentials.
//!
//! 5. **Quantum Olfaction** -- Turin's vibrational theory of smell using
//!    inelastic electron tunneling spectroscopy (IETS), demonstrating
//!    resonance-enhanced tunneling at matching odorant frequencies.
//!
//! # References
//!
//! - Engel et al., Nature 446, 782 (2007) -- FMO coherence
//! - Adolphs & Renger, Biophys. J. 91, 2778 (2006) -- FMO Hamiltonian
//! - Hay & Scrutton, Nat. Chem. 4, 161 (2012) -- enzyme tunneling
//! - Hore & Mouritsen, Annu. Rev. Biophys. 45, 299 (2016) -- radical pair
//! - Lowdin, Rev. Mod. Phys. 35, 724 (1963) -- DNA proton tunneling
//! - Turin, Chem. Senses 21, 773 (1996) -- vibration theory of olfaction

use num_complex::Complex64;
use std::f64::consts::PI;
use std::fmt;

/// Double-precision complex number alias.
type C64 = Complex64;

// ===================================================================
// PHYSICAL CONSTANTS
// ===================================================================

/// Reduced Planck constant (J s).
const HBAR: f64 = 1.054_571_817e-34;

/// Boltzmann constant (J / K).
const K_B: f64 = 1.380_649e-23;

/// Proton mass (kg).
const M_PROTON: f64 = 1.672_621_92e-27;

/// Deuterium mass (kg) -- approximately 2x proton.
const M_DEUTERIUM: f64 = 3.343_583_72e-27;

/// Electron mass (kg).
const M_ELECTRON: f64 = 9.109_383_7e-31;

/// Electron-volt to Joule conversion.
const EV_TO_J: f64 = 1.602_176_634e-19;

/// Wavenumber (cm^-1) to Joule conversion.
const CM_INV_TO_J: f64 = 1.986_445_68e-23;

/// Bohr magneton (J / T).
const BOHR_MAGNETON: f64 = 9.274_010_078_3e-24;

/// Free-electron g-factor.
const G_ELECTRON: f64 = 2.002_319_304;

/// Femtosecond to second conversion.
const FS_TO_S: f64 = 1.0e-15;

/// Nanometre to metre conversion.
const NM_TO_M: f64 = 1.0e-9;

/// Micro-Tesla to Tesla conversion.
const UT_TO_T: f64 = 1.0e-6;

// ===================================================================
// ERROR TYPE
// ===================================================================

/// Errors arising from quantum biology simulations.
#[derive(Debug, Clone)]
pub enum QuantumBiologyError {
    /// Invalid number of chromophore sites.
    InvalidSiteCount(usize),
    /// Negative or unphysical barrier parameter.
    InvalidBarrier(String),
    /// Invalid temperature (must be > 0 K).
    InvalidTemperature(f64),
    /// Invalid field angle (must be in [0, 2*pi]).
    InvalidFieldAngle(f64),
    /// Generic numerical failure.
    NumericalError(String),
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl fmt::Display for QuantumBiologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSiteCount(n) => write!(f, "Invalid site count: {}", n),
            Self::InvalidBarrier(msg) => write!(f, "Invalid barrier: {}", msg),
            Self::InvalidTemperature(t) => write!(f, "Invalid temperature: {} K", t),
            Self::InvalidFieldAngle(a) => write!(f, "Invalid field angle: {} rad", a),
            Self::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
        }
    }
}

impl std::error::Error for QuantumBiologyError {}

// ===================================================================
// TOP-LEVEL CONFIG (BUILDER PATTERN)
// ===================================================================

/// Master configuration for quantum biology simulations.
#[derive(Debug, Clone)]
pub struct QuantumBiologyConfig {
    /// Temperature in Kelvin (default 300 K for room temperature).
    pub temperature_k: f64,
    /// Whether to include decoherence / dissipation.
    pub include_decoherence: bool,
    /// Number of time steps for dynamical simulations.
    pub default_steps: usize,
}

impl Default for QuantumBiologyConfig {
    fn default() -> Self {
        Self {
            temperature_k: 300.0,
            include_decoherence: true,
            default_steps: 1000,
        }
    }
}

impl QuantumBiologyConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the temperature.
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature_k = t;
        self
    }

    /// Enable or disable decoherence.
    pub fn decoherence(mut self, enable: bool) -> Self {
        self.include_decoherence = enable;
        self
    }

    /// Set the default number of time steps.
    pub fn steps(mut self, n: usize) -> Self {
        self.default_steps = n;
        self
    }
}

// ===================================================================
// DENSITY MATRIX HELPERS
// ===================================================================

/// Inline helpers for row-major NxN density matrices stored as Vec<C64>.

/// Zero complex.
#[inline]
fn c0() -> C64 {
    C64::new(0.0, 0.0)
}

/// Real complex.
#[inline]
fn cr(x: f64) -> C64 {
    C64::new(x, 0.0)
}

/// Imaginary complex.
#[inline]
fn ci(x: f64) -> C64 {
    C64::new(0.0, x)
}

/// Index into a row-major NxN matrix.
#[inline]
fn idx(row: usize, col: usize, n: usize) -> usize {
    row * n + col
}

/// Trace of an NxN density matrix.
fn dm_trace(rho: &[C64], n: usize) -> f64 {
    (0..n).map(|i| rho[idx(i, i, n)].re).sum()
}

/// Population on site `k` (diagonal element).
#[inline]
fn population(rho: &[C64], k: usize, n: usize) -> f64 {
    rho[idx(k, k, n)].re
}

/// Matrix multiply: C = A * B (all NxN row-major).
fn mat_mul(a: &[C64], b: &[C64], n: usize) -> Vec<C64> {
    let mut c = vec![c0(); n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[idx(i, k, n)];
            if a_ik.norm_sqr() < 1e-30 {
                continue;
            }
            for j in 0..n {
                c[idx(i, j, n)] += a_ik * b[idx(k, j, n)];
            }
        }
    }
    c
}

/// Conjugate transpose (dagger) of an NxN matrix.
fn mat_dagger(a: &[C64], n: usize) -> Vec<C64> {
    let mut d = vec![c0(); n * n];
    for i in 0..n {
        for j in 0..n {
            d[idx(i, j, n)] = a[idx(j, i, n)].conj();
        }
    }
    d
}

/// Commutator [A, B] = AB - BA.
fn commutator(a: &[C64], b: &[C64], n: usize) -> Vec<C64> {
    let ab = mat_mul(a, b, n);
    let ba = mat_mul(b, a, n);
    let mut result = vec![c0(); n * n];
    for i in 0..n * n {
        result[i] = ab[i] - ba[i];
    }
    result
}

/// Anticommutator {A, B} = AB + BA.
fn anticommutator(a: &[C64], b: &[C64], n: usize) -> Vec<C64> {
    let ab = mat_mul(a, b, n);
    let ba = mat_mul(b, a, n);
    let mut result = vec![c0(); n * n];
    for i in 0..n * n {
        result[i] = ab[i] + ba[i];
    }
    result
}

/// Matrix exponential via Pade approximation (order 6) for small matrices.
/// Computes exp(A) where A is NxN. Uses scaling and squaring for stability.
fn mat_exp(a: &[C64], n: usize) -> Vec<C64> {
    // Estimate norm for scaling
    let norm: f64 = a.iter().map(|x| x.norm()).sum::<f64>() / (n as f64);
    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32 + 1
    } else {
        0
    };
    let scale = 2.0_f64.powi(-(s as i32));

    // Scale the matrix
    let mut scaled = vec![c0(); n * n];
    for i in 0..n * n {
        scaled[i] = a[i] * scale;
    }

    // Pade [6/6] approximant: exp(A) ~ (D + N) * inv(D - N)
    // where N = sum_{k=1}^{6} c_k A^k, D similarly
    // For simplicity use Taylor series with enough terms
    let mut result = vec![c0(); n * n];
    // Identity
    for i in 0..n {
        result[idx(i, i, n)] = cr(1.0);
    }

    let mut power = result.clone(); // A^0 = I
    let mut factorial = 1.0_f64;

    for k in 1..=13 {
        factorial *= k as f64;
        power = mat_mul(&power, &scaled, n);
        let coeff = 1.0 / factorial;
        for i in 0..n * n {
            result[i] += power[i] * coeff;
        }
        // Early exit if converged
        if k >= 6 {
            let term_norm: f64 = power.iter().map(|x| x.norm() * coeff).sum::<f64>();
            if term_norm < 1e-15 * (n as f64) {
                break;
            }
        }
    }

    // Squaring phase
    for _ in 0..s {
        result = mat_mul(&result, &result, n);
    }

    result
}

// ===================================================================
// 1. FMO COMPLEX PHOTOSYNTHESIS
// ===================================================================

/// Result of FMO time evolution.
#[derive(Debug, Clone)]
pub struct FmoEvolution {
    /// Site populations at each time step. `site_populations[t][k]` is the
    /// population on chromophore `k` at time step `t`.
    pub site_populations: Vec<Vec<f64>>,
    /// Time points in femtoseconds.
    pub times_fs: Vec<f64>,
    /// Coherence magnitudes |rho_{ij}| at each time step (flattened upper triangle).
    pub coherences: Vec<Vec<f64>>,
    /// Transfer efficiency to the reaction center (site 3) at each step.
    pub transfer_efficiency: Vec<f64>,
}

/// 7-site Fenna-Matthews-Olson photosynthetic complex.
///
/// Implements the Lindblad master equation for open quantum system dynamics
/// of the FMO complex in *Chlorobaculum tepidum*.  Site energies and
/// inter-chromophore couplings are taken from Adolphs & Renger (2006).
///
/// The master equation is:
///   d rho / dt = -i/hbar [H, rho] + sum_k gamma_k D[L_k](rho)
///
/// where D[L](rho) = L rho L^dag - 0.5 {L^dag L, rho}.
#[derive(Debug, Clone)]
pub struct FmoComplex {
    /// Number of chromophore sites (7 for standard FMO).
    pub sites: usize,
    /// System Hamiltonian (sites x sites), in units of cm^-1.
    pub hamiltonian: Vec<Vec<f64>>,
    /// Site-to-bath dephasing rates (cm^-1).
    pub dephasing_rates: Vec<f64>,
    /// Relaxation rates between adjacent sites (cm^-1).
    pub relaxation_rate: f64,
    /// Temperature in Kelvin.
    pub temperature_k: f64,
    /// Trapping rate at the reaction center, site 3 (cm^-1).
    pub trapping_rate: f64,
    /// Initial excitation site index (0-based, typically 0 or 5).
    pub initial_site: usize,
}

impl FmoComplex {
    /// Construct the standard 7-site FMO complex with experimentally-derived
    /// parameters from Adolphs & Renger, Biophys. J. 91, 2778 (2006).
    ///
    /// Site energies are relative to 12210 cm^-1.
    pub fn standard() -> Self {
        // Site energies (cm^-1) relative to 12210 cm^-1 baseline
        // From Adolphs & Renger (2006), Table 1
        let site_energies = [
            200.0, // BChl 1
            320.0, // BChl 2
            0.0,   // BChl 3 (lowest energy -- reaction center trap)
            110.0, // BChl 4
            270.0, // BChl 5
            420.0, // BChl 6
            230.0, // BChl 7
        ];

        // Inter-chromophore couplings (cm^-1)
        // From Adolphs & Renger (2006), Table 2
        // Only upper triangle; matrix is symmetric.
        let couplings: [(usize, usize, f64); 21] = [
            (0, 1, -87.7), // 1-2
            (0, 2, 5.5),   // 1-3
            (0, 3, -5.9),  // 1-4
            (0, 4, 6.7),   // 1-5
            (0, 5, -13.7), // 1-6
            (0, 6, -9.9),  // 1-7
            (1, 2, 30.8),  // 2-3
            (1, 3, 8.2),   // 2-4
            (1, 4, 0.7),   // 2-5
            (1, 5, 11.8),  // 2-6
            (1, 6, 4.3),   // 2-7
            (2, 3, -53.5), // 3-4
            (2, 4, -2.2),  // 3-5
            (2, 5, -9.6),  // 3-6
            (2, 6, 6.0),   // 3-7
            (3, 4, -70.7), // 4-5
            (3, 5, -17.0), // 4-6
            (3, 6, -63.3), // 4-7
            (4, 5, 81.1),  // 5-6
            (4, 6, -1.3),  // 5-7
            (5, 6, 39.7),  // 6-7
        ];

        let n = 7;
        let mut hamiltonian = vec![vec![0.0; n]; n];
        for i in 0..n {
            hamiltonian[i][i] = site_energies[i];
        }
        for &(i, j, v) in &couplings {
            hamiltonian[i][j] = v;
            hamiltonian[j][i] = v;
        }

        // Dephasing rates -- environment-induced pure dephasing at 300 K
        // Typical values ~100 cm^-1 (Ishizaki & Fleming, PNAS 2009)
        let dephasing_rates = vec![100.0; n];

        FmoComplex {
            sites: n,
            hamiltonian,
            dephasing_rates,
            relaxation_rate: 1.0, // inter-site relaxation (cm^-1)
            temperature_k: 300.0,
            trapping_rate: 1.0, // trapping at reaction center (cm^-1)
            initial_site: 0,    // excitation enters at BChl 1
        }
    }

    /// Set which site receives the initial excitation (0-based).
    pub fn initial_site(mut self, site: usize) -> Self {
        self.initial_site = site;
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature_k = t;
        self
    }

    /// Set the trapping rate at the reaction center.
    pub fn trapping_rate(mut self, rate: f64) -> Self {
        self.trapping_rate = rate;
        self
    }

    /// Set the dephasing rates for all sites uniformly.
    pub fn dephasing(mut self, rate: f64) -> Self {
        self.dephasing_rates = vec![rate; self.sites];
        self
    }

    /// Set the relaxation rate.
    pub fn relaxation(mut self, rate: f64) -> Self {
        self.relaxation_rate = rate;
        self
    }

    /// Build the complex Hamiltonian matrix (in Joules) as a flat row-major
    /// Vec<C64> for density-matrix evolution.
    fn hamiltonian_matrix_j(&self) -> Vec<C64> {
        let n = self.sites;
        let mut h = vec![c0(); n * n];
        for i in 0..n {
            for j in 0..n {
                h[idx(i, j, n)] = cr(self.hamiltonian[i][j] * CM_INV_TO_J);
            }
        }
        h
    }

    /// Evolve the FMO density matrix using the Lindblad master equation.
    ///
    /// # Arguments
    /// * `duration_fs` -- total evolution time in femtoseconds.
    /// * `steps` -- number of discrete time steps.
    ///
    /// # Returns
    /// An `FmoEvolution` struct with populations, coherences, and
    /// transfer efficiency at each time step.
    pub fn evolve(
        &self,
        duration_fs: f64,
        steps: usize,
    ) -> Result<FmoEvolution, QuantumBiologyError> {
        if self.sites == 0 {
            return Err(QuantumBiologyError::InvalidSiteCount(0));
        }
        if duration_fs <= 0.0 {
            return Err(QuantumBiologyError::InvalidConfig(
                "Duration must be positive".into(),
            ));
        }
        if steps == 0 {
            return Err(QuantumBiologyError::InvalidConfig(
                "Steps must be > 0".into(),
            ));
        }

        let n = self.sites;
        let dt_s = (duration_fs * FS_TO_S) / steps as f64;

        // Convert Hamiltonian to Joules
        let h_j = self.hamiltonian_matrix_j();

        // Initial density matrix: pure state on initial_site
        let mut rho = vec![c0(); n * n];
        let s0 = self.initial_site.min(n - 1);
        rho[idx(s0, s0, n)] = cr(1.0);

        let mut site_populations = Vec::with_capacity(steps + 1);
        let mut times_fs = Vec::with_capacity(steps + 1);
        let mut coherences = Vec::with_capacity(steps + 1);
        let mut transfer_efficiency = Vec::with_capacity(steps + 1);

        // Record initial state
        let record = |rho: &[C64],
                      pops: &mut Vec<Vec<f64>>,
                      coh: &mut Vec<Vec<f64>>,
                      eff: &mut Vec<f64>,
                      n: usize| {
            let mut p = Vec::with_capacity(n);
            for k in 0..n {
                p.push(population(rho, k, n).max(0.0));
            }
            // Coherences: upper triangle |rho_{ij}|
            let mut c_vec = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    c_vec.push(rho[idx(i, j, n)].norm());
                }
            }
            let rc_site = 2; // site 3 (0-indexed = 2) is the reaction center
            eff.push(p[rc_site]);
            pops.push(p);
            coh.push(c_vec);
        };

        record(
            &rho,
            &mut site_populations,
            &mut coherences,
            &mut transfer_efficiency,
            n,
        );
        times_fs.push(0.0);

        // Lindblad evolution: Euler method with small dt
        // d rho/dt = -i/hbar [H, rho]
        //          + sum_k gamma_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
        //          + trapping dissipator at site 3

        for step in 1..=steps {
            // Coherent part: -i/hbar [H, rho]
            let comm = commutator(&h_j, &rho, n);
            let mut drho = vec![c0(); n * n];
            let prefactor = C64::new(0.0, -1.0 / HBAR);
            for i in 0..n * n {
                drho[i] = comm[i] * prefactor;
            }

            // Dephasing Lindblad operators: L_k = |k><k| (pure dephasing)
            // D[L_k](rho) = L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho}
            // For projector L_k = |k><k|:
            //   L_k rho L_k = rho_{kk} |k><k|
            //   L_k^dag L_k = |k><k|
            //   {|k><k|, rho}_{ij} = delta_{ik} rho_{kj} + rho_{ik} delta_{kj}
            for k in 0..n {
                let gamma = self.dephasing_rates[k] * CM_INV_TO_J / HBAR;
                // Off-diagonal decay
                for i in 0..n {
                    for j in 0..n {
                        if i == k && j == k {
                            // Diagonal: L rho L^dag = rho_{kk} |k><k|
                            // minus 0.5 * (rho_{kk} + rho_{kk}) = -rho_{kk}
                            // Net: rho_{kk} - rho_{kk} = 0  (pure dephasing preserves diag)
                        } else if i == k || j == k {
                            // One index matches: decay of coherence
                            drho[idx(i, j, n)] -= cr(0.5 * gamma) * rho[idx(i, j, n)];
                        }
                        // Both indices differ from k: no contribution from this L_k
                    }
                }
            }

            // Inter-site relaxation: Lindblad operators L_{k->k+1} = sqrt(gamma) |k+1><k|
            // Models downhill energy transfer
            for k in 0..n.saturating_sub(1) {
                let gamma_rel = self.relaxation_rate * CM_INV_TO_J / HBAR;
                // L = |k+1><k|, so L rho L^dag has element (k+1,k+1) = rho_{kk}
                // L^dag L = |k><k|
                // D[L](rho) = gamma * (rho_{kk}|k+1><k+1| - 0.5 {|k><k|, rho})
                let rho_kk = rho[idx(k, k, n)];
                drho[idx(k + 1, k + 1, n)] += cr(gamma_rel) * rho_kk;
                drho[idx(k, k, n)] -= cr(gamma_rel) * rho_kk;
                // Off-diagonal decay from {|k><k|, rho}
                for j in 0..n {
                    if j != k {
                        drho[idx(k, j, n)] -= cr(0.5 * gamma_rel) * rho[idx(k, j, n)];
                        drho[idx(j, k, n)] -= cr(0.5 * gamma_rel) * rho[idx(j, k, n)];
                    }
                }
            }

            // Trapping at reaction center (site 2, 0-indexed)
            let rc = 2;
            let gamma_trap = self.trapping_rate * CM_INV_TO_J / HBAR;
            let rho_rc = rho[idx(rc, rc, n)];
            drho[idx(rc, rc, n)] -= cr(gamma_trap) * rho_rc;
            for j in 0..n {
                if j != rc {
                    drho[idx(rc, j, n)] -= cr(0.5 * gamma_trap) * rho[idx(rc, j, n)];
                    drho[idx(j, rc, n)] -= cr(0.5 * gamma_trap) * rho[idx(j, rc, n)];
                }
            }

            // Euler step
            for i in 0..n * n {
                rho[i] += drho[i] * dt_s;
            }

            // Enforce hermiticity
            for i in 0..n {
                rho[idx(i, i, n)] = cr(rho[idx(i, i, n)].re.max(0.0));
                for j in (i + 1)..n {
                    let avg = (rho[idx(i, j, n)] + rho[idx(j, i, n)].conj()) * 0.5;
                    rho[idx(i, j, n)] = avg;
                    rho[idx(j, i, n)] = avg.conj();
                }
            }

            let t_fs = step as f64 * duration_fs / steps as f64;
            times_fs.push(t_fs);
            record(
                &rho,
                &mut site_populations,
                &mut coherences,
                &mut transfer_efficiency,
                n,
            );
        }

        Ok(FmoEvolution {
            site_populations,
            times_fs,
            coherences,
            transfer_efficiency,
        })
    }

    /// Return the total population (trace of density matrix) from evolution
    /// results. Should remain <= 1.0 due to trapping loss.
    pub fn total_population(evolution: &FmoEvolution, step: usize) -> f64 {
        evolution.site_populations[step].iter().sum()
    }

    /// Compute average coherence at a given time step.
    pub fn average_coherence(evolution: &FmoEvolution, step: usize) -> f64 {
        let coh = &evolution.coherences[step];
        if coh.is_empty() {
            return 0.0;
        }
        coh.iter().sum::<f64>() / coh.len() as f64
    }
}

// ===================================================================
// 2. ENZYME QUANTUM TUNNELING
// ===================================================================

/// Enzyme quantum tunneling through active-site barriers.
///
/// Models proton (or hydrogen) transfer across a rectangular-ish barrier
/// using the WKB (Wentzel-Kramers-Brillouin) approximation.  Captures
/// the kinetic isotope effect (KIE) by comparing H and D tunneling rates,
/// and demonstrates quantum dominance at low temperature.
#[derive(Debug, Clone)]
pub struct EnzymeTunneling {
    /// Barrier height in eV.
    pub barrier_height_ev: f64,
    /// Barrier width in nm.
    pub barrier_width_nm: f64,
    /// Particle mass in kg (default: proton).
    pub particle_mass_kg: f64,
    /// Asymmetry of the double-well potential in eV.
    /// Positive means product well is lower than reactant.
    pub asymmetry_ev: f64,
    /// Attempt frequency in Hz (vibrational frequency at well bottom).
    pub attempt_frequency_hz: f64,
}

impl EnzymeTunneling {
    /// Create a tunneling model with typical enzyme parameters.
    ///
    /// Default: 0.3 eV barrier, 0.5 A width, proton mass, 10^13 Hz attempt freq.
    pub fn new(barrier_height_ev: f64, barrier_width_nm: f64) -> Self {
        Self {
            barrier_height_ev,
            barrier_width_nm,
            particle_mass_kg: M_PROTON,
            asymmetry_ev: 0.0,
            attempt_frequency_hz: 1.0e13,
        }
    }

    /// Set the particle mass (for KIE calculations).
    pub fn mass(mut self, mass_kg: f64) -> Self {
        self.particle_mass_kg = mass_kg;
        self
    }

    /// Configure for deuterium instead of hydrogen.
    pub fn deuterium(mut self) -> Self {
        self.particle_mass_kg = M_DEUTERIUM;
        self
    }

    /// Set the double-well asymmetry.
    pub fn asymmetry(mut self, asym_ev: f64) -> Self {
        self.asymmetry_ev = asym_ev;
        self
    }

    /// Set the attempt frequency.
    pub fn attempt_frequency(mut self, freq_hz: f64) -> Self {
        self.attempt_frequency_hz = freq_hz;
        self
    }

    /// WKB tunneling probability through a rectangular barrier.
    ///
    /// T = exp(-2 * integral sqrt(2m(V-E)) / hbar dx)
    ///
    /// For a rectangular barrier of height V and width a with particle at
    /// energy E = 0 (ground state of reactant well):
    ///   T = exp(-2a * sqrt(2mV) / hbar)
    pub fn tunneling_probability(&self) -> Result<f64, QuantumBiologyError> {
        if self.barrier_height_ev < 0.0 {
            return Err(QuantumBiologyError::InvalidBarrier(
                "Barrier height must be non-negative".into(),
            ));
        }
        if self.barrier_width_nm < 0.0 {
            return Err(QuantumBiologyError::InvalidBarrier(
                "Barrier width must be non-negative".into(),
            ));
        }
        if self.barrier_height_ev == 0.0 || self.barrier_width_nm == 0.0 {
            return Ok(1.0);
        }

        let v = self.barrier_height_ev * EV_TO_J;
        let a = self.barrier_width_nm * NM_TO_M;
        let m = self.particle_mass_kg;

        let kappa = (2.0 * m * v).sqrt() / HBAR;
        let exponent = -2.0 * kappa * a;

        // Clamp to avoid underflow
        if exponent < -700.0 {
            return Ok(0.0);
        }

        Ok(exponent.exp())
    }

    /// Tunneling rate: attempt_frequency * tunneling_probability.
    pub fn tunneling_rate(&self) -> Result<f64, QuantumBiologyError> {
        let prob = self.tunneling_probability()?;
        Ok(self.attempt_frequency_hz * prob)
    }

    /// Kinetic Isotope Effect: ratio of H tunneling rate to D tunneling rate.
    ///
    /// KIE = k_H / k_D.  For pure tunneling through a rectangular barrier:
    ///   KIE = exp(2a/hbar * (sqrt(2*m_D*V) - sqrt(2*m_H*V)))
    ///
    /// KIE > 1 indicates quantum tunneling contributes significantly.
    /// Classical KIE (Arrhenius) is typically < 7; quantum KIE can be >> 10.
    pub fn kie_ratio(&self) -> Result<f64, QuantumBiologyError> {
        let h_tunnel = EnzymeTunneling {
            particle_mass_kg: M_PROTON,
            ..self.clone()
        };
        let d_tunnel = EnzymeTunneling {
            particle_mass_kg: M_DEUTERIUM,
            ..self.clone()
        };
        let rate_h = h_tunnel.tunneling_rate()?;
        let rate_d = d_tunnel.tunneling_rate()?;
        if rate_d < 1e-300 {
            return Ok(f64::INFINITY);
        }
        Ok(rate_h / rate_d)
    }

    /// Classical (Arrhenius) thermal rate at temperature T.
    ///
    /// k_classical = nu * exp(-V / k_B T)
    pub fn classical_rate(&self, temperature_k: f64) -> Result<f64, QuantumBiologyError> {
        if temperature_k <= 0.0 {
            return Err(QuantumBiologyError::InvalidTemperature(temperature_k));
        }
        let v = self.barrier_height_ev * EV_TO_J;
        let exponent = -v / (K_B * temperature_k);
        if exponent < -700.0 {
            return Ok(0.0);
        }
        Ok(self.attempt_frequency_hz * exponent.exp())
    }

    /// Total rate (tunneling + thermal) at a given temperature.
    ///
    /// k_total = nu * [T_wkb + exp(-V / k_B T)]
    ///
    /// At low T, tunneling dominates.  At high T, thermal activation dominates.
    pub fn total_rate(&self, temperature_k: f64) -> Result<f64, QuantumBiologyError> {
        let tunnel = self.tunneling_probability()?;
        let thermal = if temperature_k > 0.0 {
            let v = self.barrier_height_ev * EV_TO_J;
            let exponent = -v / (K_B * temperature_k);
            if exponent < -700.0 {
                0.0
            } else {
                exponent.exp()
            }
        } else {
            0.0
        };
        Ok(self.attempt_frequency_hz * (tunnel + thermal))
    }

    /// Compute the rate vs temperature curve.
    ///
    /// Returns `(temperatures, total_rates, classical_rates, tunnel_rates)`.
    pub fn rate_vs_temperature(
        &self,
        t_min: f64,
        t_max: f64,
        steps: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), QuantumBiologyError> {
        if t_min <= 0.0 {
            return Err(QuantumBiologyError::InvalidTemperature(t_min));
        }
        if t_max <= t_min {
            return Err(QuantumBiologyError::InvalidConfig(
                "t_max must exceed t_min".into(),
            ));
        }

        let tunnel_rate = self.tunneling_rate()?;
        let mut temps = Vec::with_capacity(steps);
        let mut total = Vec::with_capacity(steps);
        let mut classical = Vec::with_capacity(steps);
        let mut tunnel = Vec::with_capacity(steps);

        for i in 0..steps {
            let t = t_min + (t_max - t_min) * i as f64 / (steps.max(1) - 1).max(1) as f64;
            let t_rate = self.total_rate(t)?;
            let c_rate = self.classical_rate(t)?;
            temps.push(t);
            total.push(t_rate);
            classical.push(c_rate);
            tunnel.push(tunnel_rate);
        }

        Ok((temps, total, classical, tunnel))
    }

    /// Double-well energy splitting due to tunneling.
    ///
    /// For a symmetric double well, the tunnel splitting is approximately:
    ///   Delta = hbar * omega * exp(-S/hbar)
    /// where S is the WKB action through the barrier, and omega is the
    /// attempt frequency.
    pub fn tunnel_splitting_ev(&self) -> Result<f64, QuantumBiologyError> {
        let prob = self.tunneling_probability()?;
        // Delta ~ hbar * omega * sqrt(T)  for thin barriers
        let omega = 2.0 * PI * self.attempt_frequency_hz;
        let delta_j = HBAR * omega * prob.sqrt();
        Ok(delta_j / EV_TO_J)
    }
}

// ===================================================================
// 3. AVIAN MAGNETORECEPTION (RADICAL PAIR MECHANISM)
// ===================================================================

/// Radical pair mechanism for avian magnetoreception.
///
/// Models the cryptochrome-based compass in migratory birds.  A radical
/// pair (e.g., FAD^.- / Trp^.+) is created in a singlet state and
/// undergoes singlet-triplet interconversion driven by the Zeeman and
/// hyperfine interactions.  The singlet yield depends on the angle of
/// the geomagnetic field relative to the radical pair axis, providing
/// directional information.
///
/// The spin Hamiltonian for two electrons and one nuclear spin (I=1/2):
///   H = omega_1 S1z + omega_2 S2z + A (S1 . I)
/// where omega = g * mu_B * B / hbar is the Larmor frequency.
#[derive(Debug, Clone)]
pub struct RadicalPair {
    /// Isotropic hyperfine coupling constant (MHz, angular frequency units).
    pub hyperfine_coupling_mhz: f64,
    /// Hyperfine anisotropy: A_parallel = A*(1+aniso), A_perp = A*(1-aniso/2).
    /// Zero gives isotropic coupling; ~0.3 is typical for nitrogen in FAD.
    pub hyperfine_anisotropy: f64,
    /// Exchange coupling between the two electrons (MHz).
    pub exchange_coupling_mhz: f64,
    /// External magnetic field strength (micro-Tesla).
    pub field_strength_ut: f64,
    /// Singlet recombination rate (MHz).
    pub k_s: f64,
    /// Triplet recombination rate (MHz).
    pub k_t: f64,
    /// Radical pair lifetime (microseconds).
    pub lifetime_us: f64,
}

impl RadicalPair {
    /// Construct a radical pair with parameters typical of cryptochrome.
    ///
    /// Default: hyperfine ~ 1.0 mT equivalent, Earth's field 50 uT,
    /// lifetime ~ 1 us, recombination rates k_s = k_t = 1 MHz.
    pub fn cryptochrome() -> Self {
        Self {
            // ~1 mT hyperfine corresponds to ~28 MHz for electron
            hyperfine_coupling_mhz: 28.0,
            // ~30% anisotropy typical for nitrogen nuclei in FAD cofactor
            hyperfine_anisotropy: 0.3,
            exchange_coupling_mhz: 0.0,
            field_strength_ut: 50.0, // Earth's field
            k_s: 1.0,
            k_t: 1.0,
            lifetime_us: 1.0,
        }
    }

    /// Set external magnetic field strength.
    pub fn field_strength(mut self, ut: f64) -> Self {
        self.field_strength_ut = ut;
        self
    }

    /// Set hyperfine coupling.
    pub fn hyperfine(mut self, mhz: f64) -> Self {
        self.hyperfine_coupling_mhz = mhz;
        self
    }

    /// Set hyperfine anisotropy (0.0 = isotropic, 0.3 = typical nitrogen).
    pub fn anisotropy(mut self, delta: f64) -> Self {
        self.hyperfine_anisotropy = delta;
        self
    }

    /// Set exchange coupling.
    pub fn exchange(mut self, mhz: f64) -> Self {
        self.exchange_coupling_mhz = mhz;
        self
    }

    /// Set recombination rates.
    pub fn recombination_rates(mut self, k_s: f64, k_t: f64) -> Self {
        self.k_s = k_s;
        self.k_t = k_t;
        self
    }

    /// Set radical pair lifetime.
    pub fn lifetime(mut self, us: f64) -> Self {
        self.lifetime_us = us;
        self
    }

    /// Compute the singlet yield as a function of field angle.
    ///
    /// Uses a simplified model with one electron having a hyperfine-coupled
    /// nucleus (I = 1/2) and one "free" electron.  The Hilbert space is
    /// 2 (electron 1) x 2 (electron 2) x 2 (nucleus) = 8-dimensional.
    ///
    /// The field angle theta is measured from the radical pair symmetry axis.
    ///
    /// # Returns
    /// Singlet yield Phi_S in [0, 1].
    pub fn singlet_yield(&self, field_angle_rad: f64) -> Result<f64, QuantumBiologyError> {
        // Larmor frequency (angular, in MHz)
        let b = self.field_strength_ut * UT_TO_T;
        let omega = G_ELECTRON * BOHR_MAGNETON * b / (HBAR * 2.0 * PI * 1.0e6);
        // omega is now in MHz

        let a = self.hyperfine_coupling_mhz;
        let j = self.exchange_coupling_mhz;
        let theta = field_angle_rad;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Build the 8x8 spin Hamiltonian in the basis
        // |e1, e2, n> where each is |up>, |down>
        // Ordering: e1 x e2 x n
        //   0: |uuu>, 1: |uud>, 2: |udu>, 3: |udd>,
        //   4: |duu>, 5: |dud>, 6: |ddu>, 7: |ddd>
        let dim = 8;
        let mut h = vec![c0(); dim * dim];

        // Zeeman: omega * (S1z + S2z) where field is along theta
        // S1z in rotated frame: S1z*cos(theta) + S1x*sin(theta)
        // For simplicity in the analytical model, take field along z
        // and rotate the hyperfine tensor.  But for the standard RPM
        // model we use the z-projection:
        //   H_Z = omega * cos(theta) * (S1z + S2z)
        //       + omega * sin(theta) * (S1x + S2x)

        // Pauli matrices for each particle
        // S1z: +1/2 for e1=u, -1/2 for e1=d
        // S2z: +1/2 for e2=u, -1/2 for e2=d
        // Nz:  +1/2 for n=u, -1/2 for n=d

        // Helper: action of Sz on qubit i in 3-qubit system
        let sz_diag = |qubit: usize, state: usize| -> f64 {
            let bit = (state >> (2 - qubit)) & 1;
            if bit == 0 {
                0.5
            } else {
                -0.5
            }
        };

        // Diagonal Zeeman terms (Sz components)
        for s in 0..dim {
            let s1z = sz_diag(0, s);
            let s2z = sz_diag(1, s);
            h[idx(s, s, dim)] += cr(omega * cos_t * (s1z + s2z));
        }

        // Off-diagonal Zeeman (Sx components)
        // S1x flips qubit 0: |0xx> <-> |1xx>
        for s in 0..dim {
            let flipped = s ^ 4; // flip bit 2 (qubit 0)
            h[idx(s, flipped, dim)] += cr(0.5 * omega * sin_t);
        }
        // S2x flips qubit 1: |x0x> <-> |x1x>
        for s in 0..dim {
            let flipped = s ^ 2; // flip bit 1 (qubit 1)
            h[idx(s, flipped, dim)] += cr(0.5 * omega * sin_t);
        }

        // Anisotropic hyperfine: A_z * S1z*Iz + A_xy * 0.5*(S1+I- + S1-I+)
        // A_z = A * (1 + anisotropy), A_xy = A * (1 - anisotropy/2)
        let a_z = a * (1.0 + self.hyperfine_anisotropy);
        let a_xy = a * (1.0 - self.hyperfine_anisotropy / 2.0);

        // S1z * Iz (diagonal)
        for s in 0..dim {
            let s1z = sz_diag(0, s);
            let nz = sz_diag(2, s);
            h[idx(s, s, dim)] += cr(a_z * s1z * nz);
        }
        // S1+ I- + S1- I+  (flip-flop terms)
        // S1+ I-: e1 goes 1->0, n goes 0->1: flip bits 2 and 0
        for s in 0..dim {
            let e1_bit = (s >> 2) & 1;
            let n_bit = s & 1;
            if e1_bit == 1 && n_bit == 0 {
                // S1+ I- applies
                let target = (s ^ 4) ^ 1; // flip both
                h[idx(target, s, dim)] += cr(0.5 * a_xy);
                h[idx(s, target, dim)] += cr(0.5 * a_xy);
            }
        }

        // Exchange coupling: J * (S1 . S2)
        // S1.S2 = S1z*S2z + 0.5*(S1+S2- + S1-S2+)
        if j.abs() > 1e-15 {
            for s in 0..dim {
                let s1z = sz_diag(0, s);
                let s2z = sz_diag(1, s);
                h[idx(s, s, dim)] += cr(j * s1z * s2z);
            }
            // S1+ S2-: e1: 1->0, e2: 0->1 => flip bits 2 and 1
            for s in 0..dim {
                let e1_bit = (s >> 2) & 1;
                let e2_bit = (s >> 1) & 1;
                if e1_bit == 1 && e2_bit == 0 {
                    let target = (s ^ 4) ^ 2;
                    h[idx(target, s, dim)] += cr(0.5 * j);
                    h[idx(s, target, dim)] += cr(0.5 * j);
                }
            }
        }

        // Singlet projection operator on electrons 1 and 2:
        // P_S = |S><S| where |S> = (|ud> - |du>) / sqrt(2)
        // In our 8-dim basis, the singlet subspace for electrons is:
        // |S, n_up> = (|udu> - |duu>) / sqrt(2) = (|2> - |4>) / sqrt(2)
        // |S, n_down> = (|udd> - |dud>) / sqrt(2) = (|3> - |5>) / sqrt(2)
        let mut p_s = vec![c0(); dim * dim];
        // |S, n_up><S, n_up| = 0.5 (|2><2| - |2><4| - |4><2| + |4><4|)
        p_s[idx(2, 2, dim)] += cr(0.5);
        p_s[idx(2, 4, dim)] += cr(-0.5);
        p_s[idx(4, 2, dim)] += cr(-0.5);
        p_s[idx(4, 4, dim)] += cr(0.5);
        // |S, n_down><S, n_down|
        p_s[idx(3, 3, dim)] += cr(0.5);
        p_s[idx(3, 5, dim)] += cr(-0.5);
        p_s[idx(5, 3, dim)] += cr(-0.5);
        p_s[idx(5, 5, dim)] += cr(0.5);

        // Initial state: singlet with nuclear spin in mixed state
        // rho(0) = P_S / Tr(P_S) = P_S / 2
        let mut rho = vec![c0(); dim * dim];
        for i in 0..dim * dim {
            rho[i] = p_s[i] * 0.5;
        }

        // Time evolution with recombination using Heun's method (2nd-order)
        // for stability with oscillatory dynamics (Forward Euler is unstable).
        // d rho/dt = -i*2pi*[H, rho] - k_avg*rho - selective correction
        // Singlet yield = k_s * integral_0^inf Tr(P_S rho(t)) dt
        let n_steps = 5000;
        let dt_us = 5.0 * self.lifetime_us / n_steps as f64;
        let dt_mhz = dt_us; // MHz * us = dimensionless

        let k_avg = 0.5 * (self.k_s + self.k_t);
        let dk = 0.5 * (self.k_s - self.k_t);
        let has_selective = dk.abs() > 1e-15;

        let mut singlet_yield_integral = 0.0;

        for _ in 0..n_steps {
            // Tr(P_S * rho)
            let mut tr_ps_rho = 0.0;
            for i in 0..dim {
                for j in 0..dim {
                    tr_ps_rho += (p_s[idx(i, j, dim)] * rho[idx(j, i, dim)]).re;
                }
            }

            singlet_yield_integral += self.k_s * tr_ps_rho * dt_mhz;

            // Heun's method: k1 = f(rho), predictor, k2 = f(predictor), average
            let rp_rhs = |rho_in: &[C64]| -> Vec<C64> {
                let comm = commutator(&h, rho_in, dim);
                let mut d = vec![c0(); dim * dim];
                for i in 0..dim * dim {
                    d[i] = comm[i] * C64::new(0.0, -2.0 * PI);
                    d[i] -= rho_in[i] * cr(k_avg);
                }
                if has_selective {
                    let anti = anticommutator(&p_s, rho_in, dim);
                    for i in 0..dim * dim {
                        d[i] -= anti[i] * cr(0.5 * dk);
                    }
                }
                d
            };

            let k1 = rp_rhs(&rho);

            // Predictor: rho + dt * k1
            let mut rho_pred = vec![c0(); dim * dim];
            for i in 0..dim * dim {
                rho_pred[i] = rho[i] + k1[i] * cr(dt_mhz);
            }

            let k2 = rp_rhs(&rho_pred);

            // Corrector: rho + dt/2 * (k1 + k2)
            for i in 0..dim * dim {
                rho[i] += (k1[i] + k2[i]) * cr(dt_mhz * 0.5);
            }
        }

        // Clamp to physical range
        Ok(singlet_yield_integral.clamp(0.0, 1.0))
    }

    /// Compute singlet yield across a range of field angles.
    ///
    /// Returns (angles_rad, yields) for angles in [0, pi].
    pub fn angular_response(
        &self,
        n_angles: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), QuantumBiologyError> {
        let mut angles = Vec::with_capacity(n_angles);
        let mut yields = Vec::with_capacity(n_angles);
        for i in 0..n_angles {
            let theta = PI * i as f64 / (n_angles - 1).max(1) as f64;
            let sy = self.singlet_yield(theta)?;
            angles.push(theta);
            yields.push(sy);
        }
        Ok((angles, yields))
    }

    /// Compass sensitivity: max - min singlet yield across angles.
    /// A larger anisotropy indicates a better compass.
    pub fn compass_anisotropy(&self, n_angles: usize) -> Result<f64, QuantumBiologyError> {
        let (_, yields) = self.angular_response(n_angles)?;
        let max_y = yields.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_y = yields.iter().cloned().fold(f64::INFINITY, f64::min);
        Ok(max_y - min_y)
    }
}

// ===================================================================
// 4. DNA QUANTUM TUNNELING MUTATIONS
// ===================================================================

/// Type of DNA base pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BasePairType {
    /// Adenine-Thymine base pair (2 hydrogen bonds).
    AT,
    /// Guanine-Cytosine base pair (3 hydrogen bonds).
    GC,
}

impl fmt::Display for BasePairType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AT => write!(f, "A-T"),
            Self::GC => write!(f, "G-C"),
        }
    }
}

/// DNA proton tunneling along hydrogen bonds in base pairs.
///
/// Models spontaneous tautomeric shifts via double proton transfer,
/// following Lowdin's hypothesis (1963).  Each hydrogen bond is modelled
/// as an asymmetric double-well potential.  The tunneling probability
/// gives the rate of spontaneous point mutations.
///
/// A-T has 2 hydrogen bonds with a relatively shallow barrier (~0.4 eV).
/// G-C has 3 hydrogen bonds with a higher barrier (~0.5 eV) but also
/// a narrower width due to stronger coupling.
#[derive(Debug, Clone)]
pub struct DnaTunneling {
    /// Base pair type.
    pub base_pair: BasePairType,
    /// Barrier height (eV) for proton transfer along H-bond.
    pub barrier_height_ev: f64,
    /// Barrier width (nm) -- distance between wells.
    pub barrier_width_nm: f64,
    /// Energy asymmetry between tautomeric forms (eV).
    /// Positive means the rare tautomer is higher in energy.
    pub asymmetry_ev: f64,
    /// Number of hydrogen bonds (2 for AT, 3 for GC).
    pub n_bonds: usize,
    /// Attempt frequency for proton vibration (Hz).
    pub attempt_frequency_hz: f64,
}

impl DnaTunneling {
    /// Create a DNA tunneling model for the specified base pair type.
    ///
    /// Parameters from Lowdin (1963) and Florian et al., JACS 118, 3010 (1996).
    pub fn new(bp: BasePairType) -> Self {
        match bp {
            BasePairType::AT => Self {
                base_pair: bp,
                barrier_height_ev: 0.40, // ~0.4 eV for A-T
                barrier_width_nm: 0.070, // ~0.7 A (N-H...N distance)
                asymmetry_ev: 0.05,      // rare tautomer slightly higher
                n_bonds: 2,
                attempt_frequency_hz: 1.0e13,
            },
            BasePairType::GC => Self {
                base_pair: bp,
                barrier_height_ev: 0.50, // ~0.5 eV for G-C
                barrier_width_nm: 0.060, // slightly narrower
                asymmetry_ev: 0.10,      // larger asymmetry
                n_bonds: 3,
                attempt_frequency_hz: 1.0e13,
            },
        }
    }

    /// Set custom barrier parameters.
    pub fn barrier(mut self, height_ev: f64, width_nm: f64) -> Self {
        self.barrier_height_ev = height_ev;
        self.barrier_width_nm = width_nm;
        self
    }

    /// Set asymmetry.
    pub fn asymmetry(mut self, asym_ev: f64) -> Self {
        self.asymmetry_ev = asym_ev;
        self
    }

    /// Probability of finding the proton in the tautomeric (rare) form
    /// for a single hydrogen bond, at thermal equilibrium.
    ///
    /// This combines tunneling probability with Boltzmann weighting of
    /// the energy asymmetry.
    pub fn tautomer_probability(&self, temperature_k: f64) -> Result<f64, QuantumBiologyError> {
        if temperature_k <= 0.0 {
            return Err(QuantumBiologyError::InvalidTemperature(temperature_k));
        }

        // WKB tunneling probability for a single bond
        let tunneler = EnzymeTunneling::new(self.barrier_height_ev, self.barrier_width_nm);
        let t_prob = tunneler.tunneling_probability()?;

        // Boltzmann factor for the energy asymmetry
        let delta_e = self.asymmetry_ev * EV_TO_J;
        let boltzmann = (-delta_e / (K_B * temperature_k)).exp();

        // Effective population of rare tautomer
        // In a two-state system with tunneling splitting Delta and asymmetry:
        //   P_rare = T * boltzmann / (1 + T * boltzmann)
        // where T is the tunneling probability
        let p_single = t_prob * boltzmann / (1.0 + t_prob * boltzmann);

        Ok(p_single)
    }

    /// Probability that ALL hydrogen bonds simultaneously tunnel to the
    /// tautomeric form (concerted double proton transfer).
    ///
    /// For independent bonds: P_total = P_single ^ n_bonds.
    /// This is an upper bound; correlated tunneling may differ.
    pub fn concerted_tautomer_probability(
        &self,
        temperature_k: f64,
    ) -> Result<f64, QuantumBiologyError> {
        let p_single = self.tautomer_probability(temperature_k)?;
        Ok(p_single.powi(self.n_bonds as i32))
    }

    /// Mutation rate (per second) from quantum tunneling.
    ///
    /// Rate = attempt_frequency * P_tautomer * P_misincorporation
    ///
    /// P_misincorporation ~ 10^-4 to 10^-5 (probability that the rare
    /// tautomer leads to mispairing during replication).
    pub fn mutation_rate(&self, temperature_k: f64) -> Result<f64, QuantumBiologyError> {
        let p_taut = self.tautomer_probability(temperature_k)?;
        let p_misincorporation = 1.0e-4;
        Ok(self.attempt_frequency_hz * p_taut * p_misincorporation)
    }

    /// Compare mutation rates across a temperature range.
    ///
    /// Returns (temperatures, quantum_rates, classical_rates).
    pub fn mutation_rate_vs_temperature(
        &self,
        t_min: f64,
        t_max: f64,
        steps: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), QuantumBiologyError> {
        if t_min <= 0.0 {
            return Err(QuantumBiologyError::InvalidTemperature(t_min));
        }

        let mut temps = Vec::with_capacity(steps);
        let mut quantum = Vec::with_capacity(steps);
        let mut classical = Vec::with_capacity(steps);
        let p_misincorporation = 1.0e-4;

        for i in 0..steps {
            let t = t_min + (t_max - t_min) * i as f64 / (steps - 1).max(1) as f64;
            let q_rate = self.mutation_rate(t)?;

            // Classical rate: pure Arrhenius (no tunneling)
            let v = self.barrier_height_ev * EV_TO_J;
            let c_rate = self.attempt_frequency_hz * (-v / (K_B * t)).exp() * p_misincorporation;

            temps.push(t);
            quantum.push(q_rate);
            classical.push(c_rate);
        }

        Ok((temps, quantum, classical))
    }

    /// Tunnel splitting (energy, in eV) for the double-well proton potential.
    pub fn tunnel_splitting_ev(&self) -> Result<f64, QuantumBiologyError> {
        let tunneler = EnzymeTunneling::new(self.barrier_height_ev, self.barrier_width_nm);
        tunneler.tunnel_splitting_ev()
    }
}

// ===================================================================
// 5. QUANTUM OLFACTION (TURIN'S VIBRATION THEORY)
// ===================================================================

/// Quantum olfaction based on Turin's vibrational theory of smell.
///
/// Models inelastic electron tunneling spectroscopy (IETS) in olfactory
/// receptors.  An electron tunnels from a donor site to an acceptor site,
/// assisted by coupling to the vibrational mode of an odorant molecule.
///
/// When the odorant vibrational frequency matches the donor-acceptor
/// energy gap, tunneling is resonantly enhanced -- this is the proposed
/// mechanism for molecular recognition by smell.
///
/// Two-site model:
///   H = E_D |D><D| + E_A |A><A| + V (|D><A| + |A><D|)
///       + hbar*w_v (a^dag a + 1/2) + lambda (a + a^dag) |A><A|
///
/// where lambda is the electron-phonon coupling.
#[derive(Debug, Clone)]
pub struct QuantumNose {
    /// Donor energy (eV).
    pub donor_energy_ev: f64,
    /// Acceptor energy (eV).
    pub acceptor_energy_ev: f64,
    /// Electronic coupling between donor and acceptor (eV).
    pub coupling_ev: f64,
    /// Electron-phonon (odorant) coupling strength (eV).
    pub phonon_coupling_ev: f64,
    /// Dissipation rate for the electron on the acceptor (eV, as rate/hbar).
    pub dissipation_rate_ev: f64,
    /// Temperature (K).
    pub temperature_k: f64,
    /// Number of Fock states for the vibrational mode (truncation).
    pub n_fock: usize,
}

impl QuantumNose {
    /// Create a quantum nose model with default parameters.
    ///
    /// Default: 0.2 eV gap, moderate coupling, 300 K.
    pub fn new() -> Self {
        Self {
            donor_energy_ev: 0.0,
            acceptor_energy_ev: -0.2,
            coupling_ev: 0.01,
            phonon_coupling_ev: 0.02,
            dissipation_rate_ev: 0.005,
            temperature_k: 300.0,
            n_fock: 10,
        }
    }

    /// Set donor energy.
    pub fn donor_energy(mut self, ev: f64) -> Self {
        self.donor_energy_ev = ev;
        self
    }

    /// Set acceptor energy.
    pub fn acceptor_energy(mut self, ev: f64) -> Self {
        self.acceptor_energy_ev = ev;
        self
    }

    /// Set electronic coupling.
    pub fn coupling(mut self, ev: f64) -> Self {
        self.coupling_ev = ev;
        self
    }

    /// Set electron-phonon coupling.
    pub fn phonon_coupling(mut self, ev: f64) -> Self {
        self.phonon_coupling_ev = ev;
        self
    }

    /// Set dissipation rate.
    pub fn dissipation(mut self, ev: f64) -> Self {
        self.dissipation_rate_ev = ev;
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, k: f64) -> Self {
        self.temperature_k = k;
        self
    }

    /// Set Fock space truncation.
    pub fn fock_states(mut self, n: usize) -> Self {
        self.n_fock = n;
        self
    }

    /// Energy gap between donor and acceptor (eV).
    pub fn energy_gap(&self) -> f64 {
        (self.donor_energy_ev - self.acceptor_energy_ev).abs()
    }

    /// Compute the IETS tunneling rate for a given odorant vibrational
    /// frequency (in cm^-1).
    ///
    /// Uses Fermi's golden rule with phonon-assisted tunneling:
    ///
    ///   Gamma(w) = (2*pi/hbar) * |V|^2 * |<n+1|a^dag|n>|^2 * lambda^2
    ///              * delta(E_D - E_A - hbar*w)
    ///
    /// The delta function is broadened by the dissipation rate into a
    /// Lorentzian, giving a resonance peak when hbar*w matches the gap.
    ///
    /// # Arguments
    /// * `odorant_freq_cm_inv` -- vibrational frequency of the odorant (cm^-1).
    ///
    /// # Returns
    /// Tunneling rate in s^-1.
    pub fn tunneling_rate(&self, odorant_freq_cm_inv: f64) -> Result<f64, QuantumBiologyError> {
        if odorant_freq_cm_inv < 0.0 {
            return Err(QuantumBiologyError::InvalidConfig(
                "Vibrational frequency must be non-negative".into(),
            ));
        }

        let gap_j = self.energy_gap() * EV_TO_J;
        let hw_j = odorant_freq_cm_inv * CM_INV_TO_J;
        let v_j = self.coupling_ev * EV_TO_J;
        let lambda_j = self.phonon_coupling_ev * EV_TO_J;
        let gamma_j = self.dissipation_rate_ev * EV_TO_J;

        // Thermal occupation number of the vibrational mode
        let n_thermal = if self.temperature_k > 0.0 && hw_j > 0.0 {
            1.0 / ((hw_j / (K_B * self.temperature_k)).exp() - 1.0).max(1e-30)
        } else {
            0.0
        };

        // Phonon-assisted matrix element squared
        // For emission (energy loss): |<n+1|a^dag|n>|^2 = n + 1
        let matrix_element_sq = n_thermal + 1.0;

        // Lorentzian spectral density (broadened delta function)
        let detuning = gap_j - hw_j;
        let lorentzian = (gamma_j / PI) / (detuning * detuning + gamma_j * gamma_j);

        // Fermi's golden rule rate
        let rate =
            (2.0 * PI / HBAR) * v_j * v_j * lambda_j * lambda_j * matrix_element_sq * lorentzian;

        Ok(rate.max(0.0))
    }

    /// Compute the tunneling rate spectrum across a range of odorant
    /// frequencies.
    ///
    /// Returns (frequencies_cm_inv, rates_per_s).
    pub fn spectrum(
        &self,
        freq_min: f64,
        freq_max: f64,
        steps: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), QuantumBiologyError> {
        let mut freqs = Vec::with_capacity(steps);
        let mut rates = Vec::with_capacity(steps);

        for i in 0..steps {
            let f = freq_min + (freq_max - freq_min) * i as f64 / (steps - 1).max(1) as f64;
            let r = self.tunneling_rate(f)?;
            freqs.push(f);
            rates.push(r);
        }

        Ok((freqs, rates))
    }

    /// Find the resonant frequency (cm^-1) that maximizes tunneling.
    ///
    /// This is the frequency where the odorant vibration matches the
    /// donor-acceptor energy gap.
    pub fn resonant_frequency_cm_inv(&self) -> f64 {
        self.energy_gap() * EV_TO_J / CM_INV_TO_J
    }

    /// Compute selectivity: ratio of on-resonance to off-resonance rate.
    ///
    /// Higher selectivity means the nose can better discriminate molecules
    /// by their vibrational frequencies.
    pub fn selectivity(&self, detuning_cm_inv: f64) -> Result<f64, QuantumBiologyError> {
        let f_res = self.resonant_frequency_cm_inv();
        let rate_on = self.tunneling_rate(f_res)?;
        let rate_off = self.tunneling_rate(f_res + detuning_cm_inv)?;
        if rate_off < 1e-300 {
            return Ok(f64::INFINITY);
        }
        Ok(rate_on / rate_off)
    }

    /// Check whether a given odorant frequency is "detected" (rate above threshold).
    pub fn is_detected(
        &self,
        odorant_freq_cm_inv: f64,
        threshold_rate: f64,
    ) -> Result<bool, QuantumBiologyError> {
        let rate = self.tunneling_rate(odorant_freq_cm_inv)?;
        Ok(rate > threshold_rate)
    }
}

impl Default for QuantumNose {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// CONVENIENCE: RUN ALL DEMOS
// ===================================================================

/// Summary results from running all quantum biology demonstrations.
#[derive(Debug, Clone)]
pub struct QuantumBiologySummary {
    /// FMO: maximum transfer efficiency to reaction center.
    pub fmo_max_efficiency: f64,
    /// FMO: coherence lifetime estimate (fs) -- time for average coherence
    /// to drop below 50% of its initial value.
    pub fmo_coherence_lifetime_fs: f64,
    /// Enzyme: H tunneling probability.
    pub enzyme_h_tunneling_prob: f64,
    /// Enzyme: KIE ratio.
    pub enzyme_kie: f64,
    /// Radical pair: compass anisotropy (max - min singlet yield).
    pub rp_anisotropy: f64,
    /// DNA: A-T tautomer probability at 300 K.
    pub dna_at_tautomer_prob: f64,
    /// DNA: G-C tautomer probability at 300 K.
    pub dna_gc_tautomer_prob: f64,
    /// Olfaction: selectivity (on/off resonance ratio).
    pub olfaction_selectivity: f64,
}

/// Run all five quantum biology demonstrations with default parameters.
///
/// Returns a summary of key results.
pub fn run_all_demos() -> Result<QuantumBiologySummary, QuantumBiologyError> {
    // 1. FMO
    let fmo = FmoComplex::standard();
    let evo = fmo.evolve(1000.0, 2000)?;
    let max_eff = evo
        .transfer_efficiency
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);

    // Coherence lifetime: find when avg coherence < 50% of initial
    let c0_avg = FmoComplex::average_coherence(&evo, 0);
    let threshold = c0_avg * 0.5;
    let coh_lifetime = evo
        .times_fs
        .iter()
        .zip(0..)
        .find(|(_, i)| FmoComplex::average_coherence(&evo, *i) < threshold)
        .map(|(t, _)| *t)
        .unwrap_or(1000.0);

    // 2. Enzyme tunneling
    let enzyme = EnzymeTunneling::new(0.3, 0.05);
    let h_prob = enzyme.tunneling_probability()?;
    let kie = enzyme.kie_ratio()?;

    // 3. Radical pair
    let rp = RadicalPair::cryptochrome();
    let aniso = rp.compass_anisotropy(36)?;

    // 4. DNA tunneling
    let dna_at = DnaTunneling::new(BasePairType::AT);
    let dna_gc = DnaTunneling::new(BasePairType::GC);
    let at_prob = dna_at.tautomer_probability(300.0)?;
    let gc_prob = dna_gc.tautomer_probability(300.0)?;

    // 5. Olfaction
    let nose = QuantumNose::new();
    let sel = nose.selectivity(500.0)?;

    Ok(QuantumBiologySummary {
        fmo_max_efficiency: max_eff,
        fmo_coherence_lifetime_fs: coh_lifetime,
        enzyme_h_tunneling_prob: h_prob,
        enzyme_kie: kie,
        rp_anisotropy: aniso,
        dna_at_tautomer_prob: at_prob,
        dna_gc_tautomer_prob: gc_prob,
        olfaction_selectivity: sel,
    })
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    // ---------------------------------------------------------------
    // Helper
    // ---------------------------------------------------------------

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---------------------------------------------------------------
    // FMO tests
    // ---------------------------------------------------------------

    #[test]
    fn fmo_standard_has_7_sites() {
        let fmo = FmoComplex::standard();
        assert_eq!(fmo.sites, 7);
        assert_eq!(fmo.hamiltonian.len(), 7);
        assert_eq!(fmo.hamiltonian[0].len(), 7);
    }

    #[test]
    fn fmo_hamiltonian_is_symmetric() {
        let fmo = FmoComplex::standard();
        for i in 0..7 {
            for j in 0..7 {
                assert!(
                    approx_eq(fmo.hamiltonian[i][j], fmo.hamiltonian[j][i], 1e-10),
                    "H[{}][{}] = {} != H[{}][{}] = {}",
                    i,
                    j,
                    fmo.hamiltonian[i][j],
                    j,
                    i,
                    fmo.hamiltonian[j][i]
                );
            }
        }
    }

    #[test]
    fn fmo_site_energies_loaded() {
        let fmo = FmoComplex::standard();
        // BChl 3 (index 2) should have the lowest energy (0.0)
        assert!(approx_eq(fmo.hamiltonian[2][2], 0.0, 1e-10));
        // BChl 6 (index 5) should have the highest energy (420.0)
        assert!(approx_eq(fmo.hamiltonian[5][5], 420.0, 1e-10));
    }

    #[test]
    fn fmo_couplings_loaded() {
        let fmo = FmoComplex::standard();
        // Check coupling between BChl 1 and BChl 2 (strongest, -87.7 cm^-1)
        assert!(approx_eq(fmo.hamiltonian[0][1], -87.7, 1e-10));
        assert!(approx_eq(fmo.hamiltonian[1][0], -87.7, 1e-10));
    }

    #[test]
    fn fmo_evolution_conserves_population_initially() {
        let fmo = FmoComplex::standard().trapping_rate(0.0);
        let evo = fmo.evolve(100.0, 200).unwrap();
        // Without trapping, total population should be approximately 1
        // (dephasing alone doesn't change diagonal elements much for
        // pure dephasing Lindblad operators)
        let total = FmoComplex::total_population(&evo, 0);
        assert!(
            approx_eq(total, 1.0, 1e-10),
            "Initial population = {}",
            total
        );
    }

    #[test]
    fn fmo_coherent_transfer_to_site3() {
        // With moderate evolution time, population should transfer to
        // the low-energy site (BChl 3, index 2)
        let fmo = FmoComplex::standard()
            .dephasing(50.0)
            .relaxation(5.0)
            .trapping_rate(0.1);
        let evo = fmo.evolve(500.0, 1000).unwrap();
        let last = evo.site_populations.last().unwrap();
        // At least some population should have reached site 3
        let site3_pop = last[2];
        assert!(
            site3_pop > 0.0,
            "Site 3 population should be > 0, got {}",
            site3_pop
        );
    }

    #[test]
    fn fmo_decoherence_reduces_coherence() {
        let fmo = FmoComplex::standard().dephasing(200.0);
        let evo = fmo.evolve(500.0, 500).unwrap();
        // Coherence builds from zero (localized initial state), peaks, then decays.
        // Compare peak coherence in the first half to the final value.
        let peak_early = (1..250)
            .map(|s| FmoComplex::average_coherence(&evo, s))
            .fold(0.0f64, f64::max);
        let c_final = FmoComplex::average_coherence(&evo, 499);
        // Final coherence should be less than the peak
        assert!(
            c_final < peak_early || peak_early < 1e-10,
            "Coherence should decay from peak: peak_early={}, final={}",
            peak_early,
            c_final
        );
    }

    #[test]
    fn fmo_returns_correct_time_points() {
        let fmo = FmoComplex::standard();
        let evo = fmo.evolve(100.0, 50).unwrap();
        assert_eq!(evo.times_fs.len(), 51); // 0..=50
        assert!(approx_eq(evo.times_fs[0], 0.0, 1e-10));
        assert!(approx_eq(*evo.times_fs.last().unwrap(), 100.0, 1e-10));
    }

    #[test]
    fn fmo_invalid_inputs() {
        let fmo = FmoComplex::standard();
        assert!(fmo.evolve(-1.0, 100).is_err());
        assert!(fmo.evolve(100.0, 0).is_err());
    }

    // ---------------------------------------------------------------
    // Enzyme tunneling tests
    // ---------------------------------------------------------------

    #[test]
    fn enzyme_tunneling_positive_for_thin_barrier() {
        let e = EnzymeTunneling::new(0.3, 0.05); // 0.3 eV, 0.5 A
        let prob = e.tunneling_probability().unwrap();
        assert!(prob > 0.0, "Tunneling probability = {}", prob);
        assert!(prob < 1.0, "Tunneling probability = {}", prob);
    }

    #[test]
    fn enzyme_tunneling_negligible_for_thick_barrier() {
        let e = EnzymeTunneling::new(1.0, 1.0); // 1 eV, 10 A -- very thick
        let prob = e.tunneling_probability().unwrap();
        assert!(
            prob < 1e-10,
            "Thick barrier should suppress tunneling, got {}",
            prob
        );
    }

    #[test]
    fn enzyme_tunneling_unity_for_zero_barrier() {
        let e = EnzymeTunneling::new(0.0, 0.5);
        assert!(approx_eq(e.tunneling_probability().unwrap(), 1.0, 1e-10));

        let e2 = EnzymeTunneling::new(0.3, 0.0);
        assert!(approx_eq(e2.tunneling_probability().unwrap(), 1.0, 1e-10));
    }

    #[test]
    fn enzyme_kie_greater_than_one() {
        let e = EnzymeTunneling::new(0.3, 0.05);
        let kie = e.kie_ratio().unwrap();
        assert!(
            kie > 1.0,
            "KIE should be > 1 (H tunnels faster), got {}",
            kie
        );
    }

    #[test]
    fn enzyme_kie_increases_with_barrier() {
        let e1 = EnzymeTunneling::new(0.2, 0.05);
        let e2 = EnzymeTunneling::new(0.5, 0.05);
        let kie1 = e1.kie_ratio().unwrap();
        let kie2 = e2.kie_ratio().unwrap();
        assert!(
            kie2 > kie1,
            "Higher barrier should give larger KIE: {} vs {}",
            kie1,
            kie2
        );
    }

    #[test]
    fn enzyme_tunneling_dominates_at_low_temperature() {
        let e = EnzymeTunneling::new(0.3, 0.05);
        let rate_10k = e.total_rate(10.0).unwrap();
        let classical_10k = e.classical_rate(10.0).unwrap();
        let tunnel_rate = e.tunneling_rate().unwrap();

        // At 10 K, tunneling should dominate over thermal activation
        assert!(
            tunnel_rate > classical_10k,
            "At 10 K, tunneling ({}) should dominate over classical ({})",
            tunnel_rate,
            classical_10k
        );
        // Total rate should be approximately the tunneling rate
        assert!(
            rate_10k > classical_10k,
            "Total rate should exceed classical at low T"
        );
    }

    #[test]
    fn enzyme_classical_rate_increases_with_temperature() {
        let e = EnzymeTunneling::new(0.3, 0.05);
        let r100 = e.classical_rate(100.0).unwrap();
        let r300 = e.classical_rate(300.0).unwrap();
        assert!(r300 > r100, "Classical rate should increase with T");
    }

    #[test]
    fn enzyme_rate_vs_temperature_returns_correct_length() {
        let e = EnzymeTunneling::new(0.3, 0.05);
        let (temps, total, classical, tunnel) = e.rate_vs_temperature(10.0, 500.0, 50).unwrap();
        assert_eq!(temps.len(), 50);
        assert_eq!(total.len(), 50);
        assert_eq!(classical.len(), 50);
        assert_eq!(tunnel.len(), 50);
    }

    #[test]
    fn enzyme_invalid_temperature() {
        let e = EnzymeTunneling::new(0.3, 0.05);
        assert!(e.classical_rate(0.0).is_err());
        assert!(e.classical_rate(-10.0).is_err());
    }

    #[test]
    fn enzyme_negative_barrier_rejected() {
        let e = EnzymeTunneling::new(-0.3, 0.05);
        assert!(e.tunneling_probability().is_err());
    }

    #[test]
    fn enzyme_tunnel_splitting_positive() {
        let e = EnzymeTunneling::new(0.3, 0.05);
        let split = e.tunnel_splitting_ev().unwrap();
        assert!(split > 0.0, "Tunnel splitting should be > 0, got {}", split);
    }

    // ---------------------------------------------------------------
    // Radical pair / magnetoreception tests
    // ---------------------------------------------------------------

    #[test]
    fn rp_singlet_yield_in_valid_range() {
        let rp = RadicalPair::cryptochrome();
        let sy = rp.singlet_yield(0.0).unwrap();
        assert!(
            sy >= 0.0 && sy <= 1.0,
            "Singlet yield should be in [0,1], got {}",
            sy
        );
    }

    #[test]
    fn rp_singlet_yield_depends_on_field_angle() {
        let rp = RadicalPair::cryptochrome();
        let sy_0 = rp.singlet_yield(0.0).unwrap();
        let sy_90 = rp.singlet_yield(PI / 2.0).unwrap();
        // The yield should differ between parallel and perpendicular field
        assert!(
            (sy_0 - sy_90).abs() > 1e-6,
            "Singlet yield should vary with angle: parallel={}, perpendicular={}",
            sy_0,
            sy_90
        );
    }

    #[test]
    fn rp_zero_field_gives_roughly_half_singlet() {
        // At zero field WITH hyperfine coupling, the singlet-triplet mixing
        // (driven by hyperfine) averages the singlet character to ~0.5.
        // With no hyperfine there's no mixing and yield stays ~1.0.
        let rp = RadicalPair::cryptochrome()
            .field_strength(0.0)
            .recombination_rates(1.0, 1.0);
        let sy = rp.singlet_yield(0.0).unwrap();
        assert!(
            (sy - 0.5).abs() < 0.3,
            "Zero-field singlet yield should be ~0.5, got {}",
            sy
        );
    }

    #[test]
    fn rp_anisotropy_detectable_at_earth_field() {
        let rp = RadicalPair::cryptochrome();
        let aniso = rp.compass_anisotropy(36).unwrap();
        assert!(
            aniso > 1e-4,
            "Compass anisotropy at 50 uT should be detectable, got {}",
            aniso
        );
    }

    #[test]
    fn rp_angular_response_correct_length() {
        let rp = RadicalPair::cryptochrome();
        let (angles, yields) = rp.angular_response(10).unwrap();
        assert_eq!(angles.len(), 10);
        assert_eq!(yields.len(), 10);
        assert!(approx_eq(angles[0], 0.0, 1e-10));
        assert!(approx_eq(*angles.last().unwrap(), PI, 1e-10));
    }

    #[test]
    fn rp_stronger_field_changes_anisotropy() {
        let rp_weak = RadicalPair::cryptochrome().field_strength(10.0);
        let rp_strong = RadicalPair::cryptochrome().field_strength(500.0);
        let a_weak = rp_weak.compass_anisotropy(18).unwrap();
        let a_strong = rp_strong.compass_anisotropy(18).unwrap();
        // The anisotropy pattern should differ at different field strengths
        assert!(
            (a_weak - a_strong).abs() > 1e-6,
            "Different field strengths should give different anisotropy"
        );
    }

    // ---------------------------------------------------------------
    // DNA tunneling tests
    // ---------------------------------------------------------------

    #[test]
    fn dna_at_base_pair_parameters() {
        let dna = DnaTunneling::new(BasePairType::AT);
        assert_eq!(dna.base_pair, BasePairType::AT);
        assert_eq!(dna.n_bonds, 2);
        assert!(dna.barrier_height_ev > 0.0);
    }

    #[test]
    fn dna_gc_base_pair_parameters() {
        let dna = DnaTunneling::new(BasePairType::GC);
        assert_eq!(dna.base_pair, BasePairType::GC);
        assert_eq!(dna.n_bonds, 3);
    }

    #[test]
    fn dna_tautomer_probability_positive_but_small() {
        let dna = DnaTunneling::new(BasePairType::AT);
        let p = dna.tautomer_probability(300.0).unwrap();
        assert!(p > 0.0, "Tautomer probability should be > 0, got {}", p);
        assert!(p < 0.5, "Tautomer probability should be small, got {}", p);
    }

    #[test]
    fn dna_gc_differs_from_at() {
        let at = DnaTunneling::new(BasePairType::AT);
        let gc = DnaTunneling::new(BasePairType::GC);
        let p_at = at.tautomer_probability(300.0).unwrap();
        let p_gc = gc.tautomer_probability(300.0).unwrap();
        assert!(
            (p_at - p_gc).abs() > 1e-20,
            "AT and GC should have different tunneling rates: {} vs {}",
            p_at,
            p_gc
        );
    }

    #[test]
    fn dna_mutation_rate_positive() {
        let dna = DnaTunneling::new(BasePairType::AT);
        let rate = dna.mutation_rate(300.0).unwrap();
        assert!(rate > 0.0, "Mutation rate should be > 0, got {}", rate);
    }

    #[test]
    fn dna_mutation_rate_increases_with_temperature() {
        let dna = DnaTunneling::new(BasePairType::AT);
        let r_low = dna.mutation_rate(200.0).unwrap();
        let r_high = dna.mutation_rate(400.0).unwrap();
        assert!(
            r_high > r_low,
            "Mutation rate should increase with T: {} vs {}",
            r_low,
            r_high
        );
    }

    #[test]
    fn dna_concerted_probability_smaller_than_single() {
        let dna = DnaTunneling::new(BasePairType::AT);
        let p_single = dna.tautomer_probability(300.0).unwrap();
        let p_concerted = dna.concerted_tautomer_probability(300.0).unwrap();
        assert!(
            p_concerted <= p_single,
            "Concerted probability should be <= single: {} vs {}",
            p_concerted,
            p_single
        );
    }

    #[test]
    fn dna_invalid_temperature() {
        let dna = DnaTunneling::new(BasePairType::AT);
        assert!(dna.tautomer_probability(0.0).is_err());
        assert!(dna.tautomer_probability(-100.0).is_err());
    }

    #[test]
    fn dna_tunnel_splitting_positive() {
        let dna = DnaTunneling::new(BasePairType::GC);
        let split = dna.tunnel_splitting_ev().unwrap();
        assert!(split > 0.0, "Tunnel splitting should be > 0");
    }

    #[test]
    fn dna_mutation_rate_vs_temperature() {
        let dna = DnaTunneling::new(BasePairType::AT);
        let (temps, quantum, classical) =
            dna.mutation_rate_vs_temperature(100.0, 500.0, 20).unwrap();
        assert_eq!(temps.len(), 20);
        assert_eq!(quantum.len(), 20);
        assert_eq!(classical.len(), 20);
        // All rates should be non-negative
        for r in &quantum {
            assert!(*r >= 0.0);
        }
    }

    // ---------------------------------------------------------------
    // Quantum olfaction tests
    // ---------------------------------------------------------------

    #[test]
    fn olfaction_resonance_at_matching_frequency() {
        let nose = QuantumNose::new();
        let f_res = nose.resonant_frequency_cm_inv();
        let rate_on = nose.tunneling_rate(f_res).unwrap();
        let rate_off = nose.tunneling_rate(f_res + 1000.0).unwrap();
        assert!(
            rate_on > rate_off,
            "On-resonance rate ({}) should exceed off-resonance ({})",
            rate_on,
            rate_off
        );
    }

    #[test]
    fn olfaction_off_resonance_suppressed() {
        let nose = QuantumNose::new();
        let f_res = nose.resonant_frequency_cm_inv();
        let rate_on = nose.tunneling_rate(f_res).unwrap();
        let rate_far = nose.tunneling_rate(f_res + 5000.0).unwrap();
        // Far off-resonance should be much smaller
        assert!(
            rate_on > 10.0 * rate_far,
            "Far off-resonance should be strongly suppressed: on={}, far={}",
            rate_on,
            rate_far
        );
    }

    #[test]
    fn olfaction_selectivity_greater_than_one() {
        let nose = QuantumNose::new();
        let sel = nose.selectivity(500.0).unwrap();
        assert!(sel > 1.0, "Selectivity should be > 1, got {}", sel);
    }

    #[test]
    fn olfaction_resonant_frequency_matches_gap() {
        let nose = QuantumNose::new();
        let f_res = nose.resonant_frequency_cm_inv();
        let gap_cm = nose.energy_gap() * EV_TO_J / CM_INV_TO_J;
        assert!(
            approx_eq(f_res, gap_cm, 1e-5),
            "Resonant frequency should match gap: {} vs {}",
            f_res,
            gap_cm
        );
    }

    #[test]
    fn olfaction_spectrum_correct_length() {
        let nose = QuantumNose::new();
        let (freqs, rates) = nose.spectrum(500.0, 3000.0, 100).unwrap();
        assert_eq!(freqs.len(), 100);
        assert_eq!(rates.len(), 100);
    }

    #[test]
    fn olfaction_spectrum_has_peak() {
        let nose = QuantumNose::new();
        let f_res = nose.resonant_frequency_cm_inv();
        let (freqs, rates) = nose.spectrum(f_res - 500.0, f_res + 500.0, 100).unwrap();
        // Find the maximum rate
        let max_rate = rates.iter().cloned().fold(0.0_f64, f64::max);
        let max_idx = rates.iter().position(|r| *r == max_rate).unwrap();
        let peak_freq = freqs[max_idx];
        // Peak should be near the resonant frequency
        assert!(
            (peak_freq - f_res).abs() < 20.0,
            "Peak at {} should be near resonance at {}",
            peak_freq,
            f_res
        );
    }

    #[test]
    fn olfaction_detection_works() {
        let nose = QuantumNose::new();
        let f_res = nose.resonant_frequency_cm_inv();
        let rate_on = nose.tunneling_rate(f_res).unwrap();
        // Should be detected at a threshold well below the on-resonance rate
        let detected = nose.is_detected(f_res, rate_on * 0.5).unwrap();
        assert!(detected, "On-resonance odorant should be detected");
        // Far off-resonance should not be detected at that threshold
        let not_detected = nose.is_detected(f_res + 5000.0, rate_on * 0.5).unwrap();
        assert!(!not_detected, "Far off-resonance should not be detected");
    }

    #[test]
    fn olfaction_negative_frequency_rejected() {
        let nose = QuantumNose::new();
        assert!(nose.tunneling_rate(-100.0).is_err());
    }

    #[test]
    fn olfaction_default_trait() {
        let n1 = QuantumNose::new();
        let n2 = QuantumNose::default();
        assert!(approx_eq(n1.donor_energy_ev, n2.donor_energy_ev, 1e-15));
    }

    // ---------------------------------------------------------------
    // Integration / summary tests
    // ---------------------------------------------------------------

    #[test]
    fn run_all_demos_succeeds() {
        let summary = run_all_demos().unwrap();
        assert!(summary.fmo_max_efficiency > 0.0);
        assert!(summary.enzyme_h_tunneling_prob > 0.0);
        assert!(summary.enzyme_kie > 1.0);
        assert!(summary.rp_anisotropy > 0.0);
        assert!(summary.dna_at_tautomer_prob > 0.0);
        assert!(summary.dna_gc_tautomer_prob > 0.0);
        assert!(summary.olfaction_selectivity > 1.0);
    }

    #[test]
    fn config_builder_works() {
        let cfg = QuantumBiologyConfig::new()
            .temperature(77.0)
            .decoherence(false)
            .steps(500);
        assert!(approx_eq(cfg.temperature_k, 77.0, 1e-10));
        assert!(!cfg.include_decoherence);
        assert_eq!(cfg.default_steps, 500);
    }

    #[test]
    fn error_display_works() {
        let e = QuantumBiologyError::InvalidTemperature(-5.0);
        let s = format!("{}", e);
        assert!(s.contains("-5"));
    }

    // ---------------------------------------------------------------
    // Matrix helper tests
    // ---------------------------------------------------------------

    #[test]
    fn mat_exp_identity_is_identity() {
        let n = 3;
        let zero = vec![c0(); n * n];
        let result = mat_exp(&zero, n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(result[idx(i, j, n)].re, expected, 1e-10),
                    "exp(0)[{},{}] = {:?}, expected {}",
                    i,
                    j,
                    result[idx(i, j, n)],
                    expected
                );
                assert!(
                    result[idx(i, j, n)].im.abs() < 1e-10,
                    "exp(0)[{},{}] imaginary part = {}",
                    i,
                    j,
                    result[idx(i, j, n)].im
                );
            }
        }
    }

    #[test]
    fn commutator_of_same_matrix_is_zero() {
        let n = 2;
        let a = vec![cr(1.0), cr(2.0), cr(3.0), cr(4.0)];
        let result = commutator(&a, &a, n);
        for val in &result {
            assert!(val.norm() < 1e-10, "Commutator [A,A] should be 0");
        }
    }

    #[test]
    fn mat_dagger_of_real_symmetric_is_self() {
        let n = 2;
        let a = vec![cr(1.0), cr(2.0), cr(2.0), cr(3.0)];
        let d = mat_dagger(&a, n);
        for i in 0..n * n {
            assert!(approx_eq(a[i].re, d[i].re, 1e-10));
        }
    }

    #[test]
    fn base_pair_display() {
        assert_eq!(format!("{}", BasePairType::AT), "A-T");
        assert_eq!(format!("{}", BasePairType::GC), "G-C");
    }
}
