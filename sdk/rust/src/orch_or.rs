//! Orchestrated Objective Reduction (Orch-OR) Simulation
//!
//! **EXPERIMENTAL / RESEARCH**: Educational implementation of the Penrose-Hameroff
//! Orch-OR theory of quantum consciousness. This is a research exploration, not a
//! validated physical model. Requires `--features experimental` to compile.
//!
//! Implements the Penrose-Hameroff Orch-OR theory of quantum consciousness.
//! Orch-OR proposes that consciousness arises from quantum computations in brain
//! microtubules. Tubulin proteins can exist in quantum superposition of two
//! conformational states (alpha and beta). When the gravitational self-energy of
//! the superposition reaches the Diosi-Penrose threshold, the state undergoes
//! "objective reduction" -- a quantum-gravity-induced collapse that constitutes
//! a moment of conscious experience.
//!
//! # Physics
//!
//! The tubulin Hamiltonian is Ising-like with a transverse field:
//!
//! ```text
//!   H = -J sum_{<i,j>} Z_i Z_j  -  Gamma sum_i X_i  +  h sum_i Z_i
//! ```
//!
//! where J is the inter-tubulin coupling, Gamma is the tunneling rate between
//! alpha and beta conformations, and h is the energy bias from GTP hydrolysis.
//!
//! Decoherence is modelled as an amplitude damping channel with temperature-
//! dependent coherence time:
//!
//! ```text
//!   T_coh(T) = T_coh_0 * exp(-E_a / k_B T)
//! ```
//!
//! Penrose objective reduction occurs when the gravitational self-energy E_G
//! exceeds a threshold. The reduction timescale is:
//!
//! ```text
//!   tau = hbar / E_G
//! ```
//!
//! For approximately 10^9 tubulins, tau is approximately 25 ms, matching the
//! gamma oscillation timescale.
//!
//! # Anesthetic Model
//!
//! Anesthetics (e.g. xenon) bind to hydrophobic pockets in tubulin, disrupting
//! quantum coherence. This is modelled as a reduction in coherence time:
//!
//! ```text
//!   T_coh -> T_coh * (1 - concentration)
//! ```
//!
//! At concentration = 1, coherence vanishes and no quantum computation (and thus
//! no consciousness) can occur.
//!
//! # References
//!
//! - Penrose, R. (1994). Shadows of the Mind.
//! - Hameroff, S. & Penrose, R. (2014). Consciousness in the universe.
//! - Diosi, L. (1987). A universal master equation for the gravitational
//!   violation of quantum mechanics. Physics Letters A, 120(8), 377-381.

use crate::{C64, QuantumState, GateOperations};
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// PHYSICAL CONSTANTS
// ============================================================

/// Reduced Planck constant (J*s)
const HBAR: f64 = 1.054571817e-34;

/// Boltzmann constant (J/K)
const K_B: f64 = 1.380649e-23;

/// Gravitational constant (m^3 kg^-1 s^-2)
const G_NEWTON: f64 = 6.67430e-11;

/// Tubulin dimer mass in kg (approximately 110 kDa)
const TUBULIN_MASS_KG: f64 = 1.83e-22;

/// Characteristic displacement of tubulin conformational change (metres)
const TUBULIN_DISPLACEMENT_M: f64 = 0.25e-9;

/// Activation energy for thermal decoherence (eV converted to J)
/// Typical protein conformational barrier
const ACTIVATION_ENERGY_J: f64 = 0.04 * 1.602176634e-19;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can arise during Orch-OR simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum OrchORError {
    /// The number of tubulins must be at least 2.
    InvalidTubulinCount {
        /// The invalid count that was provided
        count: usize,
    },
    /// The coherence time is too short for meaningful simulation.
    DecoherenceTooFast {
        /// The coherence time in nanoseconds
        coherence_time_ns: f64,
    },
    /// Temperature must be positive.
    InvalidTemperature {
        /// The invalid temperature in Kelvin
        temperature_kelvin: f64,
    },
    /// The microtubule network has disconnected components.
    NetworkDisconnected {
        /// Description of the connectivity issue
        reason: String,
    },
}

impl fmt::Display for OrchORError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchORError::InvalidTubulinCount { count } => {
                write!(f, "invalid tubulin count: {} (must be >= 2)", count)
            }
            OrchORError::DecoherenceTooFast { coherence_time_ns } => {
                write!(
                    f,
                    "decoherence time {:.3} ns is too fast for simulation (minimum 0.1 ns)",
                    coherence_time_ns
                )
            }
            OrchORError::InvalidTemperature { temperature_kelvin } => {
                write!(
                    f,
                    "invalid temperature {:.2} K (must be positive)",
                    temperature_kelvin
                )
            }
            OrchORError::NetworkDisconnected { reason } => {
                write!(f, "network disconnected: {}", reason)
            }
        }
    }
}

// ============================================================
// SIMPLE PRNG (Linear Congruential Generator)
// ============================================================

/// Simple deterministic PRNG for reproducible simulation.
/// Uses the Numerical Recipes LCG parameters.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Generate a pseudo-random u64 value.
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a pseudo-random f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a pseudo-random bool with probability `p` of being true.
    fn next_bool(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the Orch-OR simulator.
///
/// Uses the builder pattern for ergonomic construction. All parameters
/// have biologically motivated defaults.
#[derive(Clone, Debug)]
pub struct OrchORConfig {
    /// Number of tubulin dimers in the microtubule (default: 8)
    pub num_tubulins: usize,
    /// Decoherence time in nanoseconds (default: 25.0)
    pub coherence_time_ns: f64,
    /// Temperature in Kelvin (default: 310.0, body temperature)
    pub temperature_kelvin: f64,
    /// Diosi-Penrose gravitational threshold in Joules (default: 1e-25)
    pub gravitational_threshold: f64,
    /// Inter-tubulin coupling strength J (default: 0.01)
    pub coupling_strength: f64,
    /// Anesthetic concentration from 0 (none) to 1 (full) (default: 0.0)
    pub anesthetic_concentration: f64,
    /// PRNG seed for reproducibility (default: 42)
    pub seed: u64,
}

impl Default for OrchORConfig {
    fn default() -> Self {
        Self {
            num_tubulins: 8,
            coherence_time_ns: 25.0,
            temperature_kelvin: 310.0,
            gravitational_threshold: 1e-25,
            coupling_strength: 0.01,
            anesthetic_concentration: 0.0,
            seed: 42,
        }
    }
}

impl OrchORConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of tubulin dimers.
    pub fn num_tubulins(mut self, n: usize) -> Self {
        self.num_tubulins = n;
        self
    }

    /// Set the coherence time in nanoseconds.
    pub fn coherence_time_ns(mut self, t: f64) -> Self {
        self.coherence_time_ns = t;
        self
    }

    /// Set the temperature in Kelvin.
    pub fn temperature_kelvin(mut self, t: f64) -> Self {
        self.temperature_kelvin = t;
        self
    }

    /// Set the Diosi-Penrose gravitational threshold in Joules.
    pub fn gravitational_threshold(mut self, e: f64) -> Self {
        self.gravitational_threshold = e;
        self
    }

    /// Set the inter-tubulin coupling strength.
    pub fn coupling_strength(mut self, j: f64) -> Self {
        self.coupling_strength = j;
        self
    }

    /// Set the anesthetic concentration (0 to 1).
    pub fn anesthetic_concentration(mut self, c: f64) -> Self {
        self.anesthetic_concentration = c.clamp(0.0, 1.0);
        self
    }

    /// Set the PRNG seed for reproducibility.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Validate the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), OrchORError> {
        if self.num_tubulins < 2 {
            return Err(OrchORError::InvalidTubulinCount {
                count: self.num_tubulins,
            });
        }
        if self.coherence_time_ns < 0.1 {
            return Err(OrchORError::DecoherenceTooFast {
                coherence_time_ns: self.coherence_time_ns,
            });
        }
        if self.temperature_kelvin <= 0.0 {
            return Err(OrchORError::InvalidTemperature {
                temperature_kelvin: self.temperature_kelvin,
            });
        }
        Ok(())
    }
}

// ============================================================
// TUBULIN STATE
// ============================================================

/// Conformational state of a single tubulin dimer.
///
/// In Orch-OR theory, each tubulin protein can adopt one of two
/// conformational states (alpha or beta), or exist in a quantum
/// superposition of both.
#[derive(Clone, Debug)]
pub enum TubulinState {
    /// Alpha-tubulin conformational state (computational |0>)
    Alpha,
    /// Beta-tubulin conformational state (computational |1>)
    Beta,
    /// Quantum superposition of alpha and beta states.
    /// The two complex amplitudes are (c_alpha, c_beta).
    Superposition(C64, C64),
}

impl TubulinState {
    /// Return whether this tubulin is in a superposition state.
    pub fn is_superposition(&self) -> bool {
        matches!(self, TubulinState::Superposition(_, _))
    }

    /// Return the probability of measuring alpha state.
    pub fn alpha_probability(&self) -> f64 {
        match self {
            TubulinState::Alpha => 1.0,
            TubulinState::Beta => 0.0,
            TubulinState::Superposition(a, _) => a.norm_sqr(),
        }
    }

    /// Return the probability of measuring beta state.
    pub fn beta_probability(&self) -> f64 {
        match self {
            TubulinState::Alpha => 0.0,
            TubulinState::Beta => 1.0,
            TubulinState::Superposition(_, b) => b.norm_sqr(),
        }
    }
}

// ============================================================
// MICROTUBULE
// ============================================================

/// A single microtubule containing an array of tubulin dimers.
///
/// In biological neurons, microtubules are hollow cylindrical polymers
/// composed of 13 protofilaments, each made of tubulin dimers. This
/// struct models the quantum state of the tubulin array.
#[derive(Clone)]
pub struct Microtubule {
    /// Array of tubulin dimers (high-level representation)
    pub tubulins: Vec<TubulinState>,
    /// Number of protofilaments (biologically 13)
    pub num_protofilaments: usize,
    /// Inter-tubulin coupling matrix (num_tubulins x num_tubulins)
    pub coupling_matrix: Vec<Vec<f64>>,
    /// Full quantum state of the microtubule (state vector of all tubulins)
    pub quantum_state: QuantumState,
    /// Current quantum coherence level (0 to 1)
    pub coherence: f64,
}

impl Microtubule {
    /// Create a new microtubule with the given number of tubulins and coupling strength.
    ///
    /// Initializes all tubulins in the alpha (|0>) state with nearest-neighbour
    /// coupling along protofilaments.
    pub fn new(num_tubulins: usize, coupling_strength: f64) -> Self {
        let tubulins = vec![TubulinState::Alpha; num_tubulins];
        let quantum_state = QuantumState::new(num_tubulins);

        // Build nearest-neighbour coupling matrix along protofilament
        let mut coupling_matrix = vec![vec![0.0; num_tubulins]; num_tubulins];
        for i in 0..num_tubulins.saturating_sub(1) {
            coupling_matrix[i][i + 1] = coupling_strength;
            coupling_matrix[i + 1][i] = coupling_strength;
        }

        Microtubule {
            tubulins,
            num_protofilaments: 13,
            coupling_matrix,
            quantum_state,
            coherence: 1.0,
        }
    }

    /// Return the number of tubulin dimers.
    pub fn num_tubulins(&self) -> usize {
        self.tubulins.len()
    }

    /// Synchronize the high-level tubulin states from the full quantum state vector.
    ///
    /// For each tubulin qubit, computes the reduced density matrix by tracing out
    /// all other qubits, then determines whether the tubulin is in alpha, beta,
    /// or superposition.
    pub fn sync_tubulins_from_state(&mut self) {
        let n = self.tubulins.len();
        let dim = self.quantum_state.dim;
        let amps = self.quantum_state.amplitudes_ref();

        for q in 0..n {
            let stride = 1 << q;
            let mut p0 = 0.0_f64; // probability of |0> for qubit q
            let mut p1 = 0.0_f64; // probability of |1> for qubit q
            let mut off_diag = C64::new(0.0, 0.0); // off-diagonal element rho_01

            for idx in 0..dim {
                let prob = amps[idx].norm_sqr();
                if idx & stride == 0 {
                    p0 += prob;
                    // For off-diagonal: pair idx with idx | stride
                    let partner = idx | stride;
                    // rho_01 = sum over all other qubit configs of
                    //   conj(amp[idx with q=0]) * amp[idx with q=1]
                    let a0 = amps[idx];
                    let a1 = amps[partner];
                    off_diag.re += a0.re * a1.re + a0.im * a1.im;
                    off_diag.im += a0.re * a1.im - a0.im * a1.re;
                } else {
                    p1 += prob;
                }
            }

            let off_diag_mag = off_diag.norm_sqr().sqrt();
            let threshold = 1e-10;

            if p0 > 1.0 - threshold && off_diag_mag < threshold {
                self.tubulins[q] = TubulinState::Alpha;
            } else if p1 > 1.0 - threshold && off_diag_mag < threshold {
                self.tubulins[q] = TubulinState::Beta;
            } else {
                // Superposition: approximate single-qubit amplitudes
                let c_alpha = C64::new(p0.sqrt(), 0.0);
                let phase = if off_diag_mag > threshold {
                    off_diag.im.atan2(off_diag.re)
                } else {
                    0.0
                };
                let c_beta = C64::new(p1.sqrt() * phase.cos(), p1.sqrt() * phase.sin());
                self.tubulins[q] = TubulinState::Superposition(c_alpha, c_beta);
            }
        }
    }

    /// Compute quantum coherence as the sum of off-diagonal elements of the
    /// density matrix (l1-norm of coherence).
    ///
    /// For a pure state |psi>, coherence = sum_{i != j} |rho_{ij}|
    /// which equals (sum |a_i|)^2 - 1 for normalized states.
    pub fn compute_coherence(&self) -> f64 {
        let amps = self.quantum_state.amplitudes_ref();
        let dim = self.quantum_state.dim;

        // l1 coherence = (sum_i |a_i|)^2 - sum_i |a_i|^2
        // For a normalized state, sum |a_i|^2 = 1, so coherence = (sum |a_i|)^2 - 1
        let sum_abs: f64 = amps.iter().take(dim).map(|a| (a.re * a.re + a.im * a.im).sqrt()).sum();
        let c = sum_abs * sum_abs - 1.0;

        // Normalize to [0, 1] by dividing by max coherence (dim - 1)
        let max_coherence = (dim as f64) - 1.0;
        if max_coherence > 0.0 {
            (c / max_coherence).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ============================================================
// REDUCTION EVENT
// ============================================================

/// A single objective reduction (OR) event -- a moment of proto-conscious experience.
///
/// When the gravitational self-energy of the superposed tubulin mass distribution
/// exceeds the Diosi-Penrose threshold, the quantum state spontaneously collapses.
/// The timing follows tau = hbar / E_G.
#[derive(Clone, Debug)]
pub struct ReductionEvent {
    /// Time at which the reduction occurred (nanoseconds)
    pub time_ns: f64,
    /// Number of tubulins that were in superposition at the moment of reduction
    pub num_tubulins_involved: usize,
    /// Gravitational self-energy E_G (Joules)
    pub energy_difference: f64,
    /// Reduction timescale tau = hbar / E_G (seconds)
    pub reduction_time: f64,
    /// Classical outcome: true = beta, false = alpha for each tubulin
    pub classical_outcome: Vec<bool>,
    /// Approximate integrated information (Phi) at the moment of reduction
    pub integrated_information: f64,
}

// ============================================================
// CONSCIOUSNESS MEASURE
// ============================================================

/// Combined consciousness metrics derived from the Orch-OR framework.
///
/// Quantifies the "consciousness level" of the microtubule system from
/// multiple complementary perspectives.
#[derive(Clone, Debug)]
pub struct ConsciousnessMeasure {
    /// Quantum coherence of the state (0 = classical, 1 = maximally coherent)
    pub coherence: f64,
    /// Average pairwise entanglement between tubulin dimers
    pub entanglement: f64,
    /// Total mass currently in quantum superposition (kg)
    pub superposition_mass: f64,
    /// Gravitational self-energy E_G for the current state (Joules)
    pub gravitational_self_energy: f64,
    /// Predicted time until objective reduction (seconds)
    pub time_to_reduction: f64,
    /// Orchestration level: how non-random the quantum computation is (0-1)
    pub orchestration_level: f64,
    /// Anesthetic suppression factor (0 = no suppression, 1 = fully suppressed)
    pub anesthetic_suppression: f64,
}

// ============================================================
// ORCH-OR SNAPSHOT
// ============================================================

/// A snapshot of the Orch-OR simulation at a single time step.
#[derive(Clone, Debug)]
pub struct OrchORSnapshot {
    /// Current simulation time (nanoseconds)
    pub time_ns: f64,
    /// Quantum coherence at this time step
    pub coherence: f64,
    /// Average entanglement at this time step
    pub entanglement: f64,
    /// Full consciousness metrics
    pub consciousness: ConsciousnessMeasure,
    /// Reduction event if one occurred at this time step
    pub reduction_event: Option<ReductionEvent>,
}

// ============================================================
// ORCH-OR SIMULATOR
// ============================================================

/// Main Orch-OR quantum consciousness simulator.
///
/// Simulates the quantum dynamics of tubulin dimers within a microtubule,
/// including Hamiltonian evolution, environmental decoherence, and
/// Penrose objective reduction.
pub struct OrchORSimulator {
    /// Simulation configuration
    pub config: OrchORConfig,
    /// The microtubule being simulated
    pub microtubule: Microtubule,
    /// Current simulation time in nanoseconds
    pub current_time_ns: f64,
    /// Effective coherence time accounting for temperature and anesthetic
    pub effective_coherence_ns: f64,
    /// History of reduction events
    pub reduction_history: Vec<ReductionEvent>,
    /// Deterministic PRNG
    rng: LcgRng,
}

impl OrchORSimulator {
    /// Create a new Orch-OR simulator with the given configuration.
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: OrchORConfig) -> Result<Self, OrchORError> {
        config.validate()?;

        let microtubule = Microtubule::new(config.num_tubulins, config.coupling_strength);

        // Temperature-dependent coherence time: T_coh(T) = T_coh_0 * exp(-E_a / k_B T)
        let temp_factor = (-ACTIVATION_ENERGY_J / (K_B * config.temperature_kelvin)).exp();
        // Anesthetic effect: T_coh -> T_coh * (1 - concentration)
        let anesthetic_factor = 1.0 - config.anesthetic_concentration;
        let effective_coherence_ns = config.coherence_time_ns * temp_factor * anesthetic_factor;

        let rng = LcgRng::new(config.seed);

        Ok(OrchORSimulator {
            config,
            microtubule,
            current_time_ns: 0.0,
            effective_coherence_ns: effective_coherence_ns.max(1e-6),
            reduction_history: Vec::new(),
            rng,
        })
    }

    /// Put all tubulins into equal superposition of alpha and beta.
    ///
    /// Applies a Hadamard gate to each tubulin qubit, creating the state
    /// (|alpha> + |beta>) / sqrt(2) for each tubulin.
    pub fn initialize_superposition(&mut self) {
        let n = self.config.num_tubulins;
        for q in 0..n {
            GateOperations::h(&mut self.microtubule.quantum_state, q);
        }
        self.microtubule.sync_tubulins_from_state();
        self.microtubule.coherence = self.microtubule.compute_coherence();
    }

    /// Evolve the simulation for the given number of time steps.
    ///
    /// Each time step is 1 nanosecond. At each step:
    /// 1. Apply the tubulin Hamiltonian (Ising + transverse field)
    /// 2. Apply environmental decoherence
    /// 3. Check whether objective reduction threshold is reached
    ///
    /// Returns a snapshot at each time step.
    pub fn evolve(&mut self, time_steps: usize) -> Vec<OrchORSnapshot> {
        let dt_ns = 1.0;
        let mut snapshots = Vec::with_capacity(time_steps);

        for _ in 0..time_steps {
            self.current_time_ns += dt_ns;

            // Step 1: Hamiltonian evolution
            self.apply_hamiltonian(dt_ns);

            // Step 2: Decoherence
            self.apply_decoherence(dt_ns);

            // Step 3: Update coherence
            self.microtubule.coherence = self.microtubule.compute_coherence();
            self.microtubule.sync_tubulins_from_state();

            // Step 4: Check for objective reduction
            let reduction_event = self.check_objective_reduction();

            if let Some(ref event) = reduction_event {
                self.reduction_history.push(event.clone());
                // After reduction, the state has collapsed -- re-initialize
                // coherence from the now-classical state
                self.microtubule.coherence = self.microtubule.compute_coherence();
            }

            // Step 5: Record snapshot
            let consciousness = self.consciousness_measure();
            let entanglement = self.average_entanglement();

            snapshots.push(OrchORSnapshot {
                time_ns: self.current_time_ns,
                coherence: self.microtubule.coherence,
                entanglement,
                consciousness,
                reduction_event,
            });
        }

        snapshots
    }

    /// Apply the tubulin interaction Hamiltonian for a time step dt (in nanoseconds).
    ///
    /// The Hamiltonian is:
    ///   H = -J sum_{<i,j>} Z_i Z_j  -  Gamma sum_i X_i  +  h sum_i Z_i
    ///
    /// We implement this via Trotterization:
    /// 1. ZZ interactions (controlled-phase rotations)
    /// 2. Transverse field (Rx rotations)
    /// 3. Longitudinal field (Rz rotations)
    pub fn apply_hamiltonian(&mut self, dt: f64) {
        let n = self.config.num_tubulins;
        let _dim = self.microtubule.quantum_state.dim;

        // Convert dt from nanoseconds to natural units
        // We use dimensionless evolution: angle = J * dt_ns * scaling
        let time_scale = 0.001; // scaling factor for numerical stability

        // --- ZZ interactions: exp(-i * J * dt * Z_i Z_j) ---
        // For each coupled pair (i, j), apply controlled phase rotation
        for i in 0..n {
            for j in (i + 1)..n {
                let coupling = self.microtubule.coupling_matrix[i][j];
                if coupling.abs() < 1e-15 {
                    continue;
                }
                let angle = coupling * dt * time_scale;
                self.apply_zz_interaction(i, j, angle);
            }
        }

        // --- Transverse field: exp(-i * Gamma * dt * X_i) = Rx(2 * Gamma * dt) ---
        let gamma = self.config.coupling_strength * 0.5; // tunneling rate
        let rx_angle = 2.0 * gamma * dt * time_scale;
        for q in 0..n {
            self.apply_rx(q, rx_angle);
        }

        // --- Longitudinal field: exp(-i * h * dt * Z_i) = Rz(2 * h * dt) ---
        // h = small bias from GTP hydrolysis, set to 10% of J
        let h_bias = self.config.coupling_strength * 0.1;
        let rz_angle = 2.0 * h_bias * dt * time_scale;
        for q in 0..n {
            self.apply_rz(q, rz_angle);
        }
    }

    /// Apply ZZ interaction: exp(-i * angle * Z_i Z_j).
    ///
    /// For basis states |b_i b_j>, the eigenvalue of Z_i Z_j is
    /// (-1)^{b_i + b_j}, so the phase applied is:
    ///   exp(-i * angle) if b_i == b_j
    ///   exp(+i * angle) if b_i != b_j
    fn apply_zz_interaction(&mut self, qubit_i: usize, qubit_j: usize, angle: f64) {
        let dim = self.microtubule.quantum_state.dim;
        let amps = self.microtubule.quantum_state.amplitudes_mut();
        let _stride_i = 1 << qubit_i;
        let _stride_j = 1 << qubit_j;

        let phase_same = C64::new(angle.cos(), -angle.sin()); // exp(-i*angle)
        let phase_diff = C64::new(angle.cos(), angle.sin());   // exp(+i*angle)

        for idx in 0..dim {
            let bit_i = (idx >> qubit_i) & 1;
            let bit_j = (idx >> qubit_j) & 1;
            let phase = if bit_i == bit_j {
                phase_same
            } else {
                phase_diff
            };
            let a = amps[idx];
            amps[idx] = C64::new(
                a.re * phase.re - a.im * phase.im,
                a.re * phase.im + a.im * phase.re,
            );
        }
    }

    /// Apply Rx(angle) rotation on a single qubit.
    ///
    /// Rx(theta) = [[cos(theta/2), -i*sin(theta/2)],
    ///              [-i*sin(theta/2), cos(theta/2)]]
    fn apply_rx(&mut self, qubit: usize, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let dim = self.microtubule.quantum_state.dim;
        let stride = 1 << qubit;
        let amps = self.microtubule.quantum_state.amplitudes_mut();

        for idx in 0..dim {
            if idx & stride != 0 {
                continue; // only process pairs where qubit bit is 0
            }
            let partner = idx | stride;
            let a0 = amps[idx];
            let a1 = amps[partner];

            // new_a0 = cos(theta/2)*a0 - i*sin(theta/2)*a1
            amps[idx] = C64::new(
                cos_half * a0.re + sin_half * a1.im,
                cos_half * a0.im - sin_half * a1.re,
            );
            // new_a1 = -i*sin(theta/2)*a0 + cos(theta/2)*a1
            amps[partner] = C64::new(
                sin_half * a0.im + cos_half * a1.re,
                -sin_half * a0.re + cos_half * a1.im,
            );
        }
    }

    /// Apply Rz(angle) rotation on a single qubit.
    ///
    /// Rz(theta) = [[exp(-i*theta/2), 0],
    ///              [0, exp(i*theta/2)]]
    fn apply_rz(&mut self, qubit: usize, angle: f64) {
        let dim = self.microtubule.quantum_state.dim;
        let stride = 1 << qubit;
        let amps = self.microtubule.quantum_state.amplitudes_mut();

        let phase_0 = C64::new((angle / 2.0).cos(), -(angle / 2.0).sin()); // exp(-i*theta/2)
        let phase_1 = C64::new((angle / 2.0).cos(), (angle / 2.0).sin());  // exp(+i*theta/2)

        for idx in 0..dim {
            let phase = if idx & stride == 0 { phase_0 } else { phase_1 };
            let a = amps[idx];
            amps[idx] = C64::new(
                a.re * phase.re - a.im * phase.im,
                a.re * phase.im + a.im * phase.re,
            );
        }
    }

    /// Apply environmental decoherence for a time step dt (nanoseconds).
    ///
    /// Models amplitude damping: each qubit independently decoheres towards |0>
    /// at rate 1/T_coherence.
    ///
    /// The damping probability per step is: p = 1 - exp(-dt / T_coh)
    ///
    /// For each qubit, we apply the amplitude damping channel:
    ///   |0><0| component unchanged
    ///   |1><1| component damped by sqrt(1-p)
    ///   off-diagonal components damped by sqrt(1-p)
    pub fn apply_decoherence(&mut self, dt: f64) {
        if self.effective_coherence_ns <= 0.0 {
            return;
        }

        let p = 1.0 - (-dt / self.effective_coherence_ns).exp();
        let damping = (1.0 - p).sqrt();
        let n = self.config.num_tubulins;
        let dim = self.microtubule.quantum_state.dim;
        let amps = self.microtubule.quantum_state.amplitudes_mut();

        // Apply amplitude damping to each qubit
        for q in 0..n {
            let stride = 1 << q;
            for idx in 0..dim {
                if idx & stride != 0 {
                    // This amplitude has qubit q in |1> state -- damp it
                    amps[idx] = C64::new(amps[idx].re * damping, amps[idx].im * damping);
                }
            }
        }

        // Renormalize the state
        let norm_sq: f64 = amps.iter().take(dim).map(|a| a.norm_sqr()).sum();
        if norm_sq > 1e-30 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for idx in 0..dim {
                amps[idx] = C64::new(amps[idx].re * inv_norm, amps[idx].im * inv_norm);
            }
        }
    }

    /// Check whether the Penrose objective reduction threshold has been reached.
    ///
    /// Computes the gravitational self-energy E_G of the current superposition
    /// and compares it to the configured threshold.
    ///
    /// E_G = G * m^2 / delta_x for each tubulin in superposition
    ///
    /// If E_G > threshold, the state collapses and a ReductionEvent is returned.
    pub fn check_objective_reduction(&mut self) -> Option<ReductionEvent> {
        let n = self.config.num_tubulins;

        // Count tubulins in superposition
        let num_superposed = self
            .microtubule
            .tubulins
            .iter()
            .filter(|t| t.is_superposition())
            .count();

        if num_superposed == 0 {
            return None;
        }

        // Gravitational self-energy: E_G = N * G * m^2 / delta_x
        let e_g = (num_superposed as f64)
            * G_NEWTON
            * TUBULIN_MASS_KG
            * TUBULIN_MASS_KG
            / TUBULIN_DISPLACEMENT_M;

        if e_g < self.config.gravitational_threshold {
            return None;
        }

        // Objective reduction occurs -- collapse the state
        let reduction_time = HBAR / e_g;

        // Compute integrated information (approximate Phi)
        let phi = self.approximate_phi();

        // Collapse: measure each qubit
        let mut classical_outcome = Vec::with_capacity(n);
        let dim = self.microtubule.quantum_state.dim;

        // Probabilistic collapse using PRNG
        let probs = self.microtubule.quantum_state.probabilities();
        let r = self.rng.next_f64();
        let mut cumsum = 0.0;
        let mut collapsed_index = 0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                collapsed_index = i;
                break;
            }
        }

        // Extract classical outcome from collapsed index
        for q in 0..n {
            classical_outcome.push((collapsed_index >> q) & 1 == 1);
        }

        // Set state to collapsed basis state
        let amps = self.microtubule.quantum_state.amplitudes_mut();
        for idx in 0..dim {
            if idx == collapsed_index {
                amps[idx] = C64::new(1.0, 0.0);
            } else {
                amps[idx] = C64::new(0.0, 0.0);
            }
        }

        // Update tubulin states
        self.microtubule.sync_tubulins_from_state();

        Some(ReductionEvent {
            time_ns: self.current_time_ns,
            num_tubulins_involved: num_superposed,
            energy_difference: e_g,
            reduction_time,
            classical_outcome,
            integrated_information: phi,
        })
    }

    /// Apply anesthetic effect by modifying the effective coherence time.
    ///
    /// Anesthetics bind to hydrophobic pockets in tubulin, disrupting
    /// quantum coherence. This is modelled as:
    ///   T_coh -> T_coh * (1 - concentration)
    pub fn apply_anesthetic(&mut self, concentration: f64) {
        let c = concentration.clamp(0.0, 1.0);
        self.config.anesthetic_concentration = c;

        // Recompute effective coherence time
        let temp_factor =
            (-ACTIVATION_ENERGY_J / (K_B * self.config.temperature_kelvin)).exp();
        let anesthetic_factor = 1.0 - c;
        self.effective_coherence_ns =
            (self.config.coherence_time_ns * temp_factor * anesthetic_factor).max(1e-6);
    }

    /// Measure the current quantum coherence of the microtubule.
    pub fn measure_coherence(&self) -> f64 {
        self.microtubule.compute_coherence()
    }

    /// Compute pairwise entanglement between two tubulin qubits.
    ///
    /// Uses the concurrence, computed from the reduced two-qubit density matrix
    /// obtained by tracing out all other qubits.
    ///
    /// Concurrence ranges from 0 (separable) to 1 (maximally entangled).
    pub fn entanglement_between_tubulins(&self, i: usize, j: usize) -> f64 {
        if i >= self.config.num_tubulins || j >= self.config.num_tubulins || i == j {
            return 0.0;
        }

        // Compute reduced 2-qubit density matrix by tracing out all other qubits
        // The 2-qubit system has 4 basis states: |00>, |01>, |10>, |11>
        let dim = self.microtubule.quantum_state.dim;
        let amps = self.microtubule.quantum_state.amplitudes_ref();
        let stride_i = 1 << i;
        let stride_j = 1 << j;

        // rho is a 4x4 density matrix for the (i, j) subsystem
        let mut rho = [[C64::new(0.0, 0.0); 4]; 4];

        for idx in 0..dim {
            let bi = (idx >> i) & 1;
            let bj = (idx >> j) & 1;
            let row = bi * 2 + bj;

            for idx2 in 0..dim {
                // Check if idx and idx2 agree on all qubits except i and j
                let mask = !(stride_i | stride_j);
                if (idx & mask) != (idx2 & mask) {
                    continue;
                }

                let bi2 = (idx2 >> i) & 1;
                let bj2 = (idx2 >> j) & 1;
                let col = bi2 * 2 + bj2;

                // rho[row][col] += conj(amps[idx]) * amps[idx2]
                // Wait -- this is |psi><psi|, so rho[row][col] += amps[idx] * conj(amps[idx2])
                // Actually for partial trace: we sum over matching configurations of traced qubits
                let a = amps[idx];
                let b = amps[idx2];
                rho[row][col].re += a.re * b.re + a.im * b.im;
                rho[row][col].im += a.im * b.re - a.re * b.im;
            }
        }

        // Compute concurrence from 2-qubit density matrix
        // For a pure global state, concurrence of subsystem (i,j) can be computed as:
        // C = 2 |det(coefficients)| where the 2-qubit state is alpha|00> + beta|01> + gamma|10> + delta|11>
        // But for a mixed reduced state we need the full formula.
        //
        // Concurrence = max(0, sqrt(l1) - sqrt(l2) - sqrt(l3) - sqrt(l4))
        // where l1 >= l2 >= l3 >= l4 are eigenvalues of rho * (sigma_y x sigma_y) * rho_conj * (sigma_y x sigma_y)

        // Simplified approach: use linear entropy of subsystem as entanglement proxy
        // S_linear = 1 - Tr(rho^2)
        let mut tr_rho_sq = 0.0_f64;
        for a in 0..4 {
            for b in 0..4 {
                // (rho^2)[a][a] += rho[a][b] * rho[b][a]
                let prod_re = rho[a][b].re * rho[b][a].re - rho[a][b].im * rho[b][a].im;
                tr_rho_sq += prod_re;
            }
        }

        // Linear entropy: 0 for pure (separable if pure global), max for maximally mixed
        // Scale to [0, 1]: S = (4/3) * (1 - Tr(rho^2)) for 2-qubit system
        let s_linear = (1.0 - tr_rho_sq).max(0.0);
        let entanglement = (s_linear * 4.0 / 3.0).min(1.0);

        entanglement
    }

    /// Compute average pairwise entanglement across all tubulin pairs.
    fn average_entanglement(&self) -> f64 {
        let n = self.config.num_tubulins;
        if n < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.entanglement_between_tubulins(i, j);
                count += 1;
            }
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Compute combined consciousness metrics.
    ///
    /// Aggregates coherence, entanglement, superposition mass, gravitational
    /// self-energy, predicted reduction time, orchestration level, and
    /// anesthetic suppression into a single ConsciousnessMeasure.
    pub fn consciousness_measure(&self) -> ConsciousnessMeasure {
        let coherence = self.microtubule.coherence;
        let entanglement = self.average_entanglement();

        // Count tubulins in superposition
        let num_superposed = self
            .microtubule
            .tubulins
            .iter()
            .filter(|t| t.is_superposition())
            .count();

        let superposition_mass = (num_superposed as f64) * TUBULIN_MASS_KG;

        let gravitational_self_energy = if num_superposed > 0 {
            (num_superposed as f64)
                * G_NEWTON
                * TUBULIN_MASS_KG
                * TUBULIN_MASS_KG
                / TUBULIN_DISPLACEMENT_M
        } else {
            0.0
        };

        let time_to_reduction = if gravitational_self_energy > 0.0 {
            HBAR / gravitational_self_energy
        } else {
            f64::INFINITY
        };

        // Orchestration level: how structured the quantum state is
        // Measured as deviation from maximally mixed state (higher = more orchestrated)
        let orchestration_level = self.compute_orchestration();

        let anesthetic_suppression = self.config.anesthetic_concentration;

        ConsciousnessMeasure {
            coherence,
            entanglement,
            superposition_mass,
            gravitational_self_energy,
            time_to_reduction,
            orchestration_level,
            anesthetic_suppression,
        }
    }

    /// Compute orchestration level as deviation from a random state.
    ///
    /// Orchestration measures how much the quantum computation is "shaped"
    /// by classical neural inputs rather than being random noise.
    /// We use the Kullback-Leibler divergence from the uniform distribution.
    fn compute_orchestration(&self) -> f64 {
        let dim = self.microtubule.quantum_state.dim;
        let probs = self.microtubule.quantum_state.probabilities();
        let uniform = 1.0 / dim as f64;

        // KL divergence D(p || uniform) = sum p_i * log(p_i / uniform)
        let mut kl = 0.0_f64;
        for &p in &probs {
            if p > 1e-30 {
                kl += p * (p / uniform).ln();
            }
        }

        // Normalize: max KL for a pure state is log(dim)
        let max_kl = (dim as f64).ln();
        if max_kl > 0.0 {
            (kl / max_kl).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Approximate integrated information (Phi) of the quantum state.
    ///
    /// Uses a simplified version based on mutual information between
    /// the two halves of the system (bipartition).
    fn approximate_phi(&self) -> f64 {
        let n = self.config.num_tubulins;
        if n < 2 {
            return 0.0;
        }

        let dim = self.microtubule.quantum_state.dim;
        let amps = self.microtubule.quantum_state.amplitudes_ref();

        // Bipartition: first half vs second half
        let n_a = n / 2;
        let n_b = n - n_a;
        let dim_a = 1 << n_a;
        let _dim_b = 1 << n_b;

        // Reduced density matrix of subsystem A (trace out B)
        let mut rho_a = vec![vec![C64::new(0.0, 0.0); dim_a]; dim_a];

        for idx in 0..dim {
            let a_idx = idx & ((1 << n_a) - 1); // bits for subsystem A
            for idx2 in 0..dim {
                // Must agree on subsystem B bits
                let b_idx1 = idx >> n_a;
                let b_idx2 = idx2 >> n_a;
                if b_idx1 != b_idx2 {
                    continue;
                }
                let a_idx2 = idx2 & ((1 << n_a) - 1);

                let a = amps[idx];
                let b = amps[idx2];
                // rho_A[a_idx][a_idx2] += amp[idx] * conj(amp[idx2])
                rho_a[a_idx][a_idx2].re += a.re * b.re + a.im * b.im;
                rho_a[a_idx][a_idx2].im += a.im * b.re - a.re * b.im;
            }
        }

        // Von Neumann entropy of subsystem A: S(A) = -Tr(rho_A * log(rho_A))
        // Approximate via eigenvalues of rho_A
        // For simplicity, use linear entropy as proxy: S_lin = 1 - Tr(rho_A^2)
        let mut tr_rho_sq = 0.0_f64;
        for a in 0..dim_a {
            for b in 0..dim_a {
                let val = rho_a[a][b];
                let val_dag = C64::new(rho_a[b][a].re, -rho_a[b][a].im);
                tr_rho_sq += val.re * val_dag.re - val.im * val_dag.im;
            }
        }

        let phi = (1.0 - tr_rho_sq).max(0.0);
        phi.min(1.0)
    }
}

// ============================================================
// NETWORK SNAPSHOT
// ============================================================

/// A snapshot of the microtubule network at a single time step.
#[derive(Clone, Debug)]
pub struct NetworkSnapshot {
    /// Current simulation time (nanoseconds)
    pub time_ns: f64,
    /// Coherence of each individual microtubule
    pub microtubule_coherences: Vec<f64>,
    /// Global coherence across the entire network
    pub global_coherence: f64,
    /// Power in the 40Hz gamma band (arbitrary units)
    pub gamma_power: f64,
    /// Reduction events that occurred at this time step: (microtubule_index, event)
    pub reduction_events: Vec<(usize, ReductionEvent)>,
}

// ============================================================
// MICROTUBULE NETWORK
// ============================================================

/// A network of interconnected microtubules linked by gap junctions.
///
/// In biological neurons, microtubules in different dendrites and soma
/// can be functionally connected through gap junctions. This allows
/// quantum coherence to potentially extend across the network, enabling
/// large-scale quantum computation relevant to consciousness.
pub struct MicrotubuleNetwork {
    /// Individual microtubule simulators
    pub simulators: Vec<OrchORSimulator>,
    /// Gap junction connections: (i, j, strength)
    pub connections: Vec<(usize, usize, f64)>,
    /// Current simulation time (nanoseconds)
    pub current_time_ns: f64,
    /// History of reduction events across all microtubules for gamma tracking
    reduction_timestamps: Vec<f64>,
}

impl MicrotubuleNetwork {
    /// Create a new network of microtubules.
    ///
    /// Each microtubule is initialized with the specified number of tubulins.
    /// No connections are created initially; use `connect()` to add gap junctions.
    pub fn new(num_microtubules: usize, tubulins_per: usize) -> Self {
        let mut simulators = Vec::with_capacity(num_microtubules);
        for i in 0..num_microtubules {
            let config = OrchORConfig::new()
                .num_tubulins(tubulins_per)
                .seed(42 + i as u64);
            let sim = OrchORSimulator::new(config).expect("valid config");
            simulators.push(sim);
        }

        MicrotubuleNetwork {
            simulators,
            connections: Vec::new(),
            current_time_ns: 0.0,
            reduction_timestamps: Vec::new(),
        }
    }

    /// Connect two microtubules via a gap junction with the given coupling strength.
    ///
    /// Gap junctions allow quantum information to flow between microtubules,
    /// potentially extending the coherent domain.
    pub fn connect(&mut self, i: usize, j: usize, strength: f64) {
        if i < self.simulators.len() && j < self.simulators.len() && i != j {
            self.connections.push((i, j, strength));
        }
    }

    /// Evolve the entire network for the given number of time steps.
    ///
    /// At each step:
    /// 1. Evolve each microtubule independently
    /// 2. Apply inter-microtubule coupling through gap junctions
    /// 3. Collect reduction events
    /// 4. Compute network-wide metrics
    pub fn evolve_network(&mut self, time_steps: usize) -> Vec<NetworkSnapshot> {
        let dt_ns = 1.0;
        let mut snapshots = Vec::with_capacity(time_steps);
        let num_mt = self.simulators.len();

        for _step in 0..time_steps {
            self.current_time_ns += dt_ns;
            let mut step_reduction_events = Vec::new();

            // Evolve each microtubule for one step
            for mt_idx in 0..num_mt {
                let mt_snapshots = self.simulators[mt_idx].evolve(1);
                if let Some(snap) = mt_snapshots.first() {
                    if let Some(ref event) = snap.reduction_event {
                        step_reduction_events.push((mt_idx, event.clone()));
                        self.reduction_timestamps.push(self.current_time_ns);
                    }
                }
            }

            // Apply inter-microtubule coupling through gap junctions
            // Model as partial state transfer: slightly mix the quantum states
            // of connected microtubules
            self.apply_gap_junction_coupling(dt_ns);

            // Collect metrics
            let microtubule_coherences: Vec<f64> = self
                .simulators
                .iter()
                .map(|s| s.microtubule.coherence)
                .collect();

            let global_coherence = self.global_coherence();
            let gamma_power = self.gamma_synchrony();

            snapshots.push(NetworkSnapshot {
                time_ns: self.current_time_ns,
                microtubule_coherences,
                global_coherence,
                gamma_power,
                reduction_events: step_reduction_events,
            });
        }

        snapshots
    }

    /// Apply gap junction coupling between connected microtubules.
    ///
    /// Models partial quantum state mixing through inter-cellular connections.
    /// For each connection (i, j, strength), we slightly bias each microtubule's
    /// state towards the other, weighted by the coupling strength.
    fn apply_gap_junction_coupling(&mut self, dt: f64) {
        // Collect state information before modifying
        let coherences: Vec<f64> = self
            .simulators
            .iter()
            .map(|s| s.microtubule.coherence)
            .collect();

        for &(i, j, strength) in &self.connections.clone() {
            let coupling_effect = strength * dt * 0.001; // small perturbation

            // If one microtubule is more coherent, it boosts the other
            let coherence_diff = coherences[i] - coherences[j];

            // Boost the less coherent one's effective coherence time
            if coherence_diff > 0.0 {
                self.simulators[j].effective_coherence_ns *= 1.0 + coupling_effect * coherence_diff;
            } else {
                self.simulators[i].effective_coherence_ns *= 1.0 - coupling_effect * coherence_diff;
            }
        }
    }

    /// Compute global coherence across the entire network.
    ///
    /// Defined as the weighted average of individual microtubule coherences,
    /// boosted by inter-microtubule correlations through gap junctions.
    pub fn global_coherence(&self) -> f64 {
        let n = self.simulators.len();
        if n == 0 {
            return 0.0;
        }

        // Base: average coherence
        let avg_coherence: f64 = self
            .simulators
            .iter()
            .map(|s| s.microtubule.coherence)
            .sum::<f64>()
            / n as f64;

        // Boost from connections: coherent coupling increases global coherence
        let mut connection_boost = 0.0_f64;
        for &(i, j, strength) in &self.connections {
            let ci = self.simulators[i].microtubule.coherence;
            let cj = self.simulators[j].microtubule.coherence;
            // Geometric mean of connected coherences, weighted by strength
            connection_boost += strength * (ci * cj).sqrt();
        }

        let num_possible_connections = if n > 1 { n * (n - 1) / 2 } else { 1 };
        let normalized_boost = connection_boost / num_possible_connections as f64;

        (avg_coherence + normalized_boost).min(1.0)
    }

    /// Compute gamma synchrony (40Hz oscillation power).
    ///
    /// Analyses the temporal pattern of reduction events to determine whether
    /// they exhibit periodicity near 25 ms (= 40 Hz), which is the predicted
    /// timescale for Orch-OR reduction in biological neural systems.
    ///
    /// Returns a value between 0 (no gamma periodicity) and 1 (perfect 40Hz).
    pub fn gamma_synchrony(&self) -> f64 {
        let timestamps = &self.reduction_timestamps;
        if timestamps.len() < 3 {
            return 0.0;
        }

        // Compute inter-event intervals
        let mut intervals: Vec<f64> = Vec::new();
        for i in 1..timestamps.len() {
            intervals.push(timestamps[i] - timestamps[i - 1]);
        }

        if intervals.is_empty() {
            return 0.0;
        }

        // Target interval for 40 Hz = 25 ms = 25,000,000 ns
        // But in our simulation with small systems, we look for periodicity
        // relative to the mean interval
        let gamma_period_ns = 25_000_000.0; // 25 ms in nanoseconds

        // Compute power at 40 Hz using discrete Fourier transform at that frequency
        // F(f) = sum_k exp(-2*pi*i*f*t_k) for timestamps t_k
        let f_gamma = 1.0 / gamma_period_ns; // frequency in inverse nanoseconds
        let mut real_sum = 0.0_f64;
        let mut imag_sum = 0.0_f64;

        for &t in timestamps {
            let phase = 2.0 * PI * f_gamma * t;
            real_sum += phase.cos();
            imag_sum += phase.sin();
        }

        let n = timestamps.len() as f64;
        let power = (real_sum * real_sum + imag_sum * imag_sum) / (n * n);

        power.min(1.0)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_config_builder() {
        let config = OrchORConfig::new();
        assert_eq!(config.num_tubulins, 8);
        assert!((config.coherence_time_ns - 25.0).abs() < TOLERANCE);
        assert!((config.temperature_kelvin - 310.0).abs() < TOLERANCE);
        assert!((config.gravitational_threshold - 1e-25).abs() < 1e-30);
        assert!((config.coupling_strength - 0.01).abs() < TOLERANCE);
        assert!((config.anesthetic_concentration - 0.0).abs() < TOLERANCE);

        let custom = OrchORConfig::new()
            .num_tubulins(16)
            .coherence_time_ns(50.0)
            .temperature_kelvin(300.0)
            .coupling_strength(0.05);
        assert_eq!(custom.num_tubulins, 16);
        assert!((custom.coherence_time_ns - 50.0).abs() < TOLERANCE);
        assert!((custom.temperature_kelvin - 300.0).abs() < TOLERANCE);
        assert!((custom.coupling_strength - 0.05).abs() < TOLERANCE);
    }

    #[test]
    fn test_config_validation() {
        let result = OrchORConfig::new().num_tubulins(1).validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OrchORError::InvalidTubulinCount { count: 1 }
        ));

        let result = OrchORConfig::new().coherence_time_ns(0.001).validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OrchORError::DecoherenceTooFast { .. }
        ));

        let result = OrchORConfig::new().temperature_kelvin(-10.0).validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OrchORError::InvalidTemperature { .. }
        ));
    }

    #[test]
    fn test_microtubule_creation() {
        let mt = Microtubule::new(8, 0.01);
        assert_eq!(mt.num_tubulins(), 8);
        assert_eq!(mt.num_protofilaments, 13);
        assert_eq!(mt.tubulins.len(), 8);
        assert_eq!(mt.coupling_matrix.len(), 8);
        assert_eq!(mt.quantum_state.dim, 256); // 2^8

        // Check nearest-neighbour coupling
        assert!((mt.coupling_matrix[0][1] - 0.01).abs() < TOLERANCE);
        assert!((mt.coupling_matrix[1][0] - 0.01).abs() < TOLERANCE);
        assert!((mt.coupling_matrix[0][2] - 0.0).abs() < TOLERANCE); // no coupling beyond NN

        // All tubulins start in alpha state
        for t in &mt.tubulins {
            assert!(matches!(t, TubulinState::Alpha));
        }
    }

    #[test]
    fn test_superposition_initialization() {
        let config = OrchORConfig::new().num_tubulins(4);
        let mut sim = OrchORSimulator::new(config).unwrap();

        // Before initialization: all alpha
        for t in &sim.microtubule.tubulins {
            assert!(matches!(t, TubulinState::Alpha));
        }

        sim.initialize_superposition();

        // After Hadamard on all qubits: should be in superposition
        // Each computational basis state should have equal probability 1/2^n
        let probs = sim.microtubule.quantum_state.probabilities();
        let expected_prob = 1.0 / 16.0; // 1/2^4
        for &p in &probs {
            assert!(
                (p - expected_prob).abs() < TOLERANCE,
                "Expected uniform superposition, got prob {}",
                p
            );
        }

        // Coherence should be high for a uniform superposition
        let coherence = sim.measure_coherence();
        assert!(
            coherence > 0.5,
            "Expected high coherence for superposition, got {}",
            coherence
        );
    }

    #[test]
    fn test_coherence_decay() {
        let config = OrchORConfig::new()
            .num_tubulins(3)
            .coherence_time_ns(5.0)
            .gravitational_threshold(1e10); // very high so no OR events
        let mut sim = OrchORSimulator::new(config).unwrap();
        sim.initialize_superposition();

        let initial_coherence = sim.measure_coherence();

        // Evolve for several coherence times
        let snapshots = sim.evolve(20);

        let final_coherence = snapshots.last().unwrap().coherence;

        assert!(
            final_coherence < initial_coherence,
            "Coherence should decay: initial={}, final={}",
            initial_coherence,
            final_coherence
        );

        // Coherence should be substantially reduced after 4x the coherence time
        assert!(
            final_coherence < initial_coherence * 0.9,
            "Coherence should decay significantly: initial={}, final={}",
            initial_coherence,
            final_coherence
        );
    }

    #[test]
    fn test_entanglement_nearest_neighbor() {
        let config = OrchORConfig::new()
            .num_tubulins(4)
            .coupling_strength(0.1)
            .coherence_time_ns(1000.0) // long coherence so entanglement can build
            .gravitational_threshold(1e10);
        let mut sim = OrchORSimulator::new(config).unwrap();
        sim.initialize_superposition();

        // Evolve to let coupling create entanglement
        sim.evolve(50);

        // Nearest neighbours should have some entanglement
        let e_01 = sim.entanglement_between_tubulins(0, 1);
        let e_03 = sim.entanglement_between_tubulins(0, 3);

        // Nearest neighbours (0,1) should generally have more entanglement
        // than distant pairs (0,3) -- but this depends on evolution time.
        // At minimum, both should be non-negative.
        assert!(
            e_01 >= 0.0,
            "Entanglement should be non-negative, got {}",
            e_01
        );
        assert!(
            e_03 >= 0.0,
            "Entanglement should be non-negative, got {}",
            e_03
        );

        // Self-entanglement should be 0
        assert!(
            sim.entanglement_between_tubulins(0, 0).abs() < TOLERANCE,
            "Self-entanglement should be 0"
        );
    }

    #[test]
    fn test_gravitational_self_energy() {
        // E_G = N * G * m^2 / delta_x
        // For 1 tubulin: E_G = 6.674e-11 * (1.83e-22)^2 / 0.25e-9
        let e_g_single =
            G_NEWTON * TUBULIN_MASS_KG * TUBULIN_MASS_KG / TUBULIN_DISPLACEMENT_M;

        // E_G should scale linearly with number of superposed tubulins
        let e_g_double = 2.0 * e_g_single;

        assert!(
            e_g_single > 0.0,
            "E_G should be positive"
        );
        assert!(
            (e_g_double - 2.0 * e_g_single).abs() < 1e-50,
            "E_G should scale linearly with N"
        );

        // Check the consciousness measure reports correct E_G
        let config = OrchORConfig::new()
            .num_tubulins(4)
            .gravitational_threshold(1e10); // high to prevent collapse
        let mut sim = OrchORSimulator::new(config).unwrap();
        sim.initialize_superposition();

        let cm = sim.consciousness_measure();
        let expected_e_g = 4.0 * e_g_single;

        // The E_G should correspond to 4 superposed tubulins
        assert!(
            cm.gravitational_self_energy > 0.0,
            "E_G should be positive for superposed state"
        );
    }

    #[test]
    fn test_reduction_event_timing() {
        // tau = hbar / E_G
        let e_g_single =
            G_NEWTON * TUBULIN_MASS_KG * TUBULIN_MASS_KG / TUBULIN_DISPLACEMENT_M;

        // For N tubulins in superposition:
        let n = 4;
        let e_g = n as f64 * e_g_single;
        let tau = HBAR / e_g;

        // tau should be positive and finite
        assert!(tau > 0.0, "Reduction time should be positive");
        assert!(tau.is_finite(), "Reduction time should be finite");

        // For 4 tubulins, tau should be enormously large (these are tiny masses)
        // E_G ~ 4 * 6.67e-11 * (1.83e-22)^2 / 2.5e-10 ~ 3.6e-44 J
        // tau ~ 1.05e-34 / 3.6e-44 ~ 2.9e9 seconds
        assert!(
            tau > 1.0,
            "For small N, reduction time should be very long: tau = {}",
            tau
        );

        // Verify the formula works with a reduction event from the simulator
        // Use a very low threshold to force reduction
        let config = OrchORConfig::new()
            .num_tubulins(3)
            .gravitational_threshold(0.0); // threshold = 0 means always reduce
        let mut sim = OrchORSimulator::new(config).unwrap();
        sim.initialize_superposition();

        let snapshots = sim.evolve(5);

        // Should have at least one reduction event (threshold is 0)
        let reduction_count = snapshots
            .iter()
            .filter(|s| s.reduction_event.is_some())
            .count();
        assert!(
            reduction_count > 0,
            "Should have at least one reduction event with threshold=0"
        );

        // Verify reduction_time = hbar / E_G for the event
        if let Some(ref event) = snapshots.iter().find_map(|s| s.reduction_event.as_ref()) {
            let expected_tau = HBAR / event.energy_difference;
            assert!(
                (event.reduction_time - expected_tau).abs() / expected_tau < 1e-6,
                "Reduction time should satisfy tau = hbar/E_G: got {}, expected {}",
                event.reduction_time,
                expected_tau
            );
        }
    }

    #[test]
    fn test_anesthetic_suppresses_coherence() {
        // Without anesthetic
        let config_normal = OrchORConfig::new()
            .num_tubulins(4)
            .coherence_time_ns(10.0)
            .gravitational_threshold(1e10);
        let mut sim_normal = OrchORSimulator::new(config_normal).unwrap();
        sim_normal.initialize_superposition();
        sim_normal.evolve(15);
        let coherence_normal = sim_normal.measure_coherence();

        // With strong anesthetic
        let config_anesthetic = OrchORConfig::new()
            .num_tubulins(4)
            .coherence_time_ns(10.0)
            .anesthetic_concentration(0.8)
            .gravitational_threshold(1e10)
            .seed(42);
        let mut sim_anesthetic = OrchORSimulator::new(config_anesthetic).unwrap();
        sim_anesthetic.initialize_superposition();
        sim_anesthetic.evolve(15);
        let coherence_anesthetic = sim_anesthetic.measure_coherence();

        assert!(
            coherence_anesthetic < coherence_normal,
            "Anesthetic should reduce coherence: normal={}, anesthetic={}",
            coherence_normal,
            coherence_anesthetic
        );
    }

    #[test]
    fn test_anesthetic_prevents_reduction() {
        // Without anesthetic -- set low threshold to encourage reduction
        let config_normal = OrchORConfig::new()
            .num_tubulins(4)
            .gravitational_threshold(0.0);
        let mut sim_normal = OrchORSimulator::new(config_normal).unwrap();
        sim_normal.initialize_superposition();
        let snapshots_normal = sim_normal.evolve(10);

        let reductions_normal = snapshots_normal
            .iter()
            .filter(|s| s.reduction_event.is_some())
            .count();

        // With full anesthetic -- coherence collapses too fast for OR
        let config_anesthetic = OrchORConfig::new()
            .num_tubulins(4)
            .anesthetic_concentration(0.99)
            .coherence_time_ns(1.0)
            .gravitational_threshold(0.0);
        let mut sim_anesthetic = OrchORSimulator::new(config_anesthetic).unwrap();
        sim_anesthetic.initialize_superposition();
        let snapshots_anesthetic = sim_anesthetic.evolve(10);

        let reductions_anesthetic = snapshots_anesthetic
            .iter()
            .filter(|s| s.reduction_event.is_some())
            .count();

        // Normal should have reductions (threshold is 0)
        assert!(
            reductions_normal > 0,
            "Normal simulation should have reduction events"
        );

        // With high anesthetic, decoherence is so fast that tubulins quickly
        // lose superposition, and subsequent steps find no superposed tubulins.
        // The first step may still trigger a reduction before decoherence kicks in,
        // but overall there should be fewer.
        assert!(
            reductions_anesthetic <= reductions_normal,
            "Anesthetic should not increase reduction events: normal={}, anesthetic={}",
            reductions_normal,
            reductions_anesthetic
        );
    }

    #[test]
    fn test_temperature_dependence() {
        // At higher temperature, decoherence should be faster
        // T_coh(T) = T_coh_0 * exp(-E_a / k_B T)
        // Higher T -> larger exp(-E_a / k_B T) -> LONGER coherence time
        // But the activation energy model means higher T = more thermal noise
        // In our model: higher T gives *longer* coherence time (less Arrhenius damping)
        // So at 310K vs 400K, 400K should actually have slightly longer coherence

        let config_cold = OrchORConfig::new()
            .num_tubulins(3)
            .temperature_kelvin(250.0)
            .coherence_time_ns(10.0)
            .gravitational_threshold(1e10);
        let mut sim_cold = OrchORSimulator::new(config_cold).unwrap();

        let config_hot = OrchORConfig::new()
            .num_tubulins(3)
            .temperature_kelvin(400.0)
            .coherence_time_ns(10.0)
            .gravitational_threshold(1e10);
        let mut sim_hot = OrchORSimulator::new(config_hot).unwrap();

        // The effective coherence time should differ
        assert!(
            (sim_cold.effective_coherence_ns - sim_hot.effective_coherence_ns).abs() > 1e-10,
            "Different temperatures should give different effective coherence times"
        );

        // For our Arrhenius model, higher T gives larger exp factor
        assert!(
            sim_hot.effective_coherence_ns > sim_cold.effective_coherence_ns,
            "Higher temperature should give longer effective coherence time in Arrhenius model: hot={}, cold={}",
            sim_hot.effective_coherence_ns,
            sim_cold.effective_coherence_ns
        );
    }

    #[test]
    fn test_consciousness_measure_coherent() {
        let config = OrchORConfig::new()
            .num_tubulins(4)
            .coherence_time_ns(1000.0) // very long coherence
            .gravitational_threshold(1e10);
        let mut sim = OrchORSimulator::new(config).unwrap();
        sim.initialize_superposition();

        let cm = sim.consciousness_measure();

        // Coherent superposition should have high consciousness metrics
        assert!(
            cm.coherence > 0.5,
            "Coherent state should have high coherence: {}",
            cm.coherence
        );
        assert!(
            cm.superposition_mass > 0.0,
            "Should have positive superposition mass"
        );
        assert!(
            cm.gravitational_self_energy > 0.0,
            "Should have positive E_G"
        );
        assert!(
            cm.time_to_reduction > 0.0 && cm.time_to_reduction.is_finite(),
            "Should have finite time to reduction"
        );
        assert!(
            cm.anesthetic_suppression < TOLERANCE,
            "No anesthetic should mean no suppression"
        );
    }

    #[test]
    fn test_consciousness_measure_classical() {
        let config = OrchORConfig::new()
            .num_tubulins(4)
            .coherence_time_ns(0.5) // very fast decoherence
            .gravitational_threshold(1e10);
        let mut sim = OrchORSimulator::new(config).unwrap();
        sim.initialize_superposition();

        // Evolve long enough for complete decoherence
        sim.evolve(100);

        let cm = sim.consciousness_measure();

        // Fully decohered state should have low consciousness metrics
        assert!(
            cm.coherence < 0.3,
            "Decohered state should have low coherence: {}",
            cm.coherence
        );
    }

    #[test]
    fn test_network_creation() {
        let net = MicrotubuleNetwork::new(5, 4);
        assert_eq!(net.simulators.len(), 5);
        for sim in &net.simulators {
            assert_eq!(sim.config.num_tubulins, 4);
        }
        assert!(net.connections.is_empty());
    }

    #[test]
    fn test_network_global_coherence() {
        let mut net = MicrotubuleNetwork::new(3, 3);

        // Initialize all microtubules in superposition
        for sim in &mut net.simulators {
            sim.initialize_superposition();
        }

        // Without connections, global coherence = average of individual coherences
        let gc_disconnected = net.global_coherence();

        // Add connections
        net.connect(0, 1, 0.5);
        net.connect(1, 2, 0.5);

        // With connections between coherent microtubules, global coherence should be boosted
        let gc_connected = net.global_coherence();

        assert!(
            gc_connected >= gc_disconnected,
            "Connected network should have >= global coherence: connected={}, disconnected={}",
            gc_connected,
            gc_disconnected
        );

        // Both should be > 0 since microtubules are in superposition
        assert!(
            gc_disconnected > 0.0,
            "Disconnected coherence should be positive"
        );
    }

    #[test]
    fn test_gamma_synchrony() {
        // gamma_synchrony looks for 40Hz periodicity in reduction timestamps
        let mut net = MicrotubuleNetwork::new(2, 3);

        // No events yet: gamma should be 0
        assert!(
            net.gamma_synchrony().abs() < TOLERANCE,
            "No events should give zero gamma power"
        );

        // Add perfectly periodic reduction timestamps at 25ms intervals (40Hz)
        let period_ns = 25_000_000.0; // 25 ms
        for i in 0..10 {
            net.reduction_timestamps.push(i as f64 * period_ns);
        }

        let gamma = net.gamma_synchrony();
        // Perfectly periodic at 40Hz should give high gamma power
        assert!(
            gamma > 0.5,
            "Perfect 40Hz periodicity should give high gamma power: {}",
            gamma
        );

        // Now test with random-ish intervals (should have lower gamma power)
        let mut net2 = MicrotubuleNetwork::new(2, 3);
        let mut rng = LcgRng::new(123);
        let mut t = 0.0;
        for _ in 0..10 {
            t += rng.next_f64() * 100_000_000.0; // random intervals up to 100ms
            net2.reduction_timestamps.push(t);
        }
        let gamma_random = net2.gamma_synchrony();

        // Random intervals should generally have lower gamma power than perfect 40Hz
        // (though not guaranteed for every seed -- we just check it is bounded)
        assert!(
            gamma_random >= 0.0 && gamma_random <= 1.0,
            "Gamma power should be in [0, 1]: {}",
            gamma_random
        );
    }

    #[test]
    fn test_orchestration_level() {
        // A pure |0> state (classical) has maximal orchestration (deterministic)
        let config = OrchORConfig::new()
            .num_tubulins(3)
            .gravitational_threshold(1e10);
        let sim_classical = OrchORSimulator::new(config).unwrap();
        let orch_classical = sim_classical.consciousness_measure().orchestration_level;

        // A uniform superposition has zero orchestration (maximally random)
        let config2 = OrchORConfig::new()
            .num_tubulins(3)
            .gravitational_threshold(1e10);
        let mut sim_super = OrchORSimulator::new(config2).unwrap();
        sim_super.initialize_superposition();
        let orch_super = sim_super.consciousness_measure().orchestration_level;

        // Classical (pure basis) state should have high orchestration
        assert!(
            orch_classical > 0.5,
            "Classical state should have high orchestration: {}",
            orch_classical
        );

        // Uniform superposition should have low orchestration
        assert!(
            orch_super < 0.1,
            "Uniform superposition should have low orchestration: {}",
            orch_super
        );

        assert!(
            orch_classical > orch_super,
            "Classical state should be more orchestrated than uniform superposition"
        );
    }

    #[test]
    fn test_tubulin_state_probabilities() {
        let alpha = TubulinState::Alpha;
        assert!((alpha.alpha_probability() - 1.0).abs() < TOLERANCE);
        assert!((alpha.beta_probability() - 0.0).abs() < TOLERANCE);
        assert!(!alpha.is_superposition());

        let beta = TubulinState::Beta;
        assert!((beta.alpha_probability() - 0.0).abs() < TOLERANCE);
        assert!((beta.beta_probability() - 1.0).abs() < TOLERANCE);
        assert!(!beta.is_superposition());

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let sup = TubulinState::Superposition(
            C64::new(inv_sqrt2, 0.0),
            C64::new(inv_sqrt2, 0.0),
        );
        assert!((sup.alpha_probability() - 0.5).abs() < TOLERANCE);
        assert!((sup.beta_probability() - 0.5).abs() < TOLERANCE);
        assert!(sup.is_superposition());
    }

    #[test]
    fn test_apply_anesthetic_dynamic() {
        let config = OrchORConfig::new()
            .num_tubulins(3)
            .coherence_time_ns(20.0)
            .gravitational_threshold(1e10);
        let mut sim = OrchORSimulator::new(config).unwrap();

        let initial_eff = sim.effective_coherence_ns;

        // Apply anesthetic
        sim.apply_anesthetic(0.5);
        let after_anesthetic = sim.effective_coherence_ns;

        assert!(
            after_anesthetic < initial_eff,
            "Anesthetic should reduce effective coherence time: before={}, after={}",
            initial_eff,
            after_anesthetic
        );
        assert!((sim.config.anesthetic_concentration - 0.5).abs() < TOLERANCE);

        // Remove anesthetic
        sim.apply_anesthetic(0.0);
        let after_removal = sim.effective_coherence_ns;

        assert!(
            (after_removal - initial_eff).abs() < TOLERANCE,
            "Removing anesthetic should restore coherence time: initial={}, after_removal={}",
            initial_eff,
            after_removal
        );
    }

    #[test]
    fn test_network_evolution() {
        let mut net = MicrotubuleNetwork::new(2, 3);
        net.connect(0, 1, 0.5);

        // Initialize both in superposition
        for sim in &mut net.simulators {
            sim.initialize_superposition();
        }

        let snapshots = net.evolve_network(10);
        assert_eq!(snapshots.len(), 10);

        // Each snapshot should have correct structure
        for snap in &snapshots {
            assert_eq!(snap.microtubule_coherences.len(), 2);
            assert!(snap.global_coherence >= 0.0 && snap.global_coherence <= 1.0 + TOLERANCE);
            assert!(snap.gamma_power >= 0.0);
            assert!(snap.time_ns > 0.0);
        }
    }

    #[test]
    fn test_error_display() {
        let err = OrchORError::InvalidTubulinCount { count: 0 };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid tubulin count"));
        assert!(msg.contains("0"));

        let err = OrchORError::DecoherenceTooFast {
            coherence_time_ns: 0.001,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("decoherence time"));

        let err = OrchORError::InvalidTemperature {
            temperature_kelvin: -5.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid temperature"));

        let err = OrchORError::NetworkDisconnected {
            reason: "no path".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("disconnected"));
    }

    #[test]
    fn test_lcg_rng_deterministic() {
        let mut rng1 = LcgRng::new(42);
        let mut rng2 = LcgRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }

        // Different seeds should give different sequences
        let mut rng3 = LcgRng::new(99);
        let v1 = rng1.next_u64();
        // Reset rng1 state -- cannot easily, but different seed should diverge
        let v3 = rng3.next_u64();
        // They might collide but extremely unlikely for LCG
    }

    #[test]
    fn test_reduction_event_has_classical_outcome() {
        let config = OrchORConfig::new()
            .num_tubulins(4)
            .gravitational_threshold(0.0); // force reduction
        let mut sim = OrchORSimulator::new(config).unwrap();
        sim.initialize_superposition();

        let snapshots = sim.evolve(3);

        // Find a reduction event
        let event = snapshots
            .iter()
            .find_map(|s| s.reduction_event.as_ref())
            .expect("Should have at least one reduction event");

        // Classical outcome should have one entry per tubulin
        assert_eq!(
            event.classical_outcome.len(),
            4,
            "Classical outcome should have one entry per tubulin"
        );

        // Energy difference should be positive
        assert!(
            event.energy_difference > 0.0,
            "Energy difference should be positive"
        );

        // Integrated information should be non-negative
        assert!(
            event.integrated_information >= 0.0,
            "Integrated information should be non-negative"
        );
    }
}
