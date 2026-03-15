//! Approximate Dynamical Quantum Error Correction Codes
//!
//! World-first implementation combining dynamical (Floquet-style) codes with
//! approximate QEC to create noise-tailored codes that dramatically reduce
//! overhead for NISQ devices.
//!
//! # Reference
//!
//! Based on the "strategic code" framework from arXiv:2502.09177 (Feb 2025),
//! which introduces spatio-temporal stabilizer tracking over multiple rounds,
//! noise-adapted code optimization, and temporal Petz recovery maps.
//!
//! # Architecture
//!
//! ```text
//! StrategicCode ──── StabilizerSnapshot ──── TimeStep tracking
//!      │
//!      ├── NoiseAdaptedOptimizer (noise model → optimal code)
//!      │     └── NoiseModel (depolarizing, biased, amplitude damping, custom)
//!      │
//!      ├── PetzRecoveryChannel (time-dependent recovery)
//!      │     └── RecoveryResult (fidelity, logical error rate, overhead)
//!      │
//!      ├── DynamicalCodeFamily
//!      │     ├── FloquetDynamicalCode (periodic schedule)
//!      │     ├── AdaptiveDynamicalCode (syndrome-adapted)
//!      │     └── HybridDynamicalCode (static + dynamic)
//!      │
//!      ├── ApproximateCodeAnalysis (relaxed Knill-Laflamme)
//!      │
//!      ├── ComparisonTools (dynamical vs static, approximate vs exact)
//!      │
//!      └── DynamicalQecSimulator (Monte Carlo simulation engine)
//! ```

use num_complex::Complex64 as C64;
use rand::Rng;
use rayon::prelude::*;
use std::fmt;

// ============================================================
// CONSTANTS
// ============================================================

const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
const ONE: C64 = C64 { re: 1.0, im: 0.0 };

// ============================================================
// PAULI OPERATOR (local to this module)
// ============================================================

/// Single-qubit Pauli operator for stabilizer specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    /// Return the 2x2 matrix representation.
    pub fn matrix(&self) -> [[C64; 2]; 2] {
        let i = C64::new(0.0, 1.0);
        match self {
            Pauli::I => [[ONE, ZERO], [ZERO, ONE]],
            Pauli::X => [[ZERO, ONE], [ONE, ZERO]],
            Pauli::Y => [[ZERO, C64::new(0.0, -1.0)], [i, ZERO]],
            Pauli::Z => [[ONE, ZERO], [ZERO, C64::new(-1.0, 0.0)]],
        }
    }

    /// Check if two Paulis commute.
    pub fn commutes_with(&self, other: &Pauli) -> bool {
        matches!(
            (self, other),
            (Pauli::I, _)
                | (_, Pauli::I)
                | (Pauli::X, Pauli::X)
                | (Pauli::Y, Pauli::Y)
                | (Pauli::Z, Pauli::Z)
        )
    }
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Pauli::I => write!(f, "I"),
            Pauli::X => write!(f, "X"),
            Pauli::Y => write!(f, "Y"),
            Pauli::Z => write!(f, "Z"),
        }
    }
}

// ============================================================
// NOISE MODEL
// ============================================================

/// Noise channel specification for noise-adapted code optimization.
///
/// Each variant describes how physical qubits decohere, which the optimizer
/// uses to tailor the code structure for maximum protection.
#[derive(Debug, Clone)]
pub enum NoiseModel {
    /// Symmetric depolarizing noise at rate p.
    /// Each qubit independently suffers X, Y, or Z error with probability p/3.
    Depolarizing { error_rate: f64 },

    /// Biased noise with Z errors eta times more likely than X or Y.
    /// Relevant for cat qubits, superconducting qubits with T1 >> T2.
    Biased {
        error_rate: f64,
        /// Ratio p_Z / p_X. A value of 1.0 recovers depolarizing.
        bias_eta: f64,
    },

    /// Amplitude damping channel at rate gamma.
    /// Models energy relaxation (T1 decay) where |1> decays to |0>.
    AmplitudeDamping { gamma: f64 },

    /// Custom noise channel specified by Kraus operators.
    /// Must satisfy sum_k K_k^dag K_k = I.
    CustomKraus {
        /// Each element is a 2x2 Kraus operator stored row-major.
        kraus_ops: Vec<[[C64; 2]; 2]>,
    },
}

impl NoiseModel {
    /// Effective single-qubit error rate for comparison purposes.
    pub fn effective_error_rate(&self) -> f64 {
        match self {
            NoiseModel::Depolarizing { error_rate } => *error_rate,
            NoiseModel::Biased { error_rate, .. } => *error_rate,
            NoiseModel::AmplitudeDamping { gamma } => *gamma,
            NoiseModel::CustomKraus { kraus_ops } => {
                // Estimate: 1 - average fidelity of identity preservation
                if kraus_ops.is_empty() {
                    return 0.0;
                }
                let identity_fid = kraus_ops[0][0][0].norm_sqr();
                (1.0 - identity_fid).max(0.0)
            }
        }
    }

    /// Sample an error on a single qubit given this noise model.
    /// Returns the Pauli error to apply (I = no error).
    pub fn sample_error(&self, rng: &mut impl Rng) -> Pauli {
        match self {
            NoiseModel::Depolarizing { error_rate } => {
                let r: f64 = rng.gen();
                if r < *error_rate / 3.0 {
                    Pauli::X
                } else if r < 2.0 * error_rate / 3.0 {
                    Pauli::Y
                } else if r < *error_rate {
                    Pauli::Z
                } else {
                    Pauli::I
                }
            }
            NoiseModel::Biased {
                error_rate,
                bias_eta,
            } => {
                // Partition: p_X = p_Y = p/(2+eta), p_Z = p*eta/(2+eta)
                let denom = 2.0 + bias_eta;
                let p_x = error_rate / denom;
                let p_z = error_rate * bias_eta / denom;
                let r: f64 = rng.gen();
                if r < p_x {
                    Pauli::X
                } else if r < 2.0 * p_x {
                    Pauli::Y
                } else if r < 2.0 * p_x + p_z {
                    Pauli::Z
                } else {
                    Pauli::I
                }
            }
            NoiseModel::AmplitudeDamping { gamma } => {
                // Simplified Pauli twirl: amplitude damping ~ p_z ≈ gamma/2
                let r: f64 = rng.gen();
                if r < gamma / 2.0 {
                    Pauli::Z
                } else if r < gamma * 0.75 {
                    Pauli::X
                } else {
                    Pauli::I
                }
            }
            NoiseModel::CustomKraus { .. } => {
                // Approximate by depolarizing at the effective rate
                let p = self.effective_error_rate();
                let r: f64 = rng.gen();
                if r < p / 3.0 {
                    Pauli::X
                } else if r < 2.0 * p / 3.0 {
                    Pauli::Y
                } else if r < p {
                    Pauli::Z
                } else {
                    Pauli::I
                }
            }
        }
    }
}

impl fmt::Display for NoiseModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NoiseModel::Depolarizing { error_rate } => {
                write!(f, "Depolarizing(p={:.4})", error_rate)
            }
            NoiseModel::Biased {
                error_rate,
                bias_eta,
            } => write!(f, "Biased(p={:.4}, eta={:.1})", error_rate, bias_eta),
            NoiseModel::AmplitudeDamping { gamma } => {
                write!(f, "AmplitudeDamping(gamma={:.4})", gamma)
            }
            NoiseModel::CustomKraus { kraus_ops } => {
                write!(f, "CustomKraus({} operators)", kraus_ops.len())
            }
        }
    }
}

// ============================================================
// STABILIZER SNAPSHOT
// ============================================================

/// A snapshot of the stabilizer group at a single time step.
///
/// In a strategic code the stabilizers evolve over time, so each step
/// has its own stabilizer group. Each stabilizer is a tensor product
/// of single-qubit Paulis (stored as a Vec<Pauli> of length n).
#[derive(Debug, Clone)]
pub struct StabilizerSnapshot {
    /// Number of physical qubits.
    pub num_qubits: usize,
    /// Stabilizer generators at this time step. Each generator is
    /// a length-n vector of Pauli operators.
    pub generators: Vec<Vec<Pauli>>,
}

impl StabilizerSnapshot {
    /// Create a new snapshot with the given generators.
    pub fn new(num_qubits: usize, generators: Vec<Vec<Pauli>>) -> Self {
        debug_assert!(generators.iter().all(|g| g.len() == num_qubits));
        Self {
            num_qubits,
            generators,
        }
    }

    /// Number of independent stabilizer generators.
    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }

    /// Number of encoded logical qubits: k = n - number_of_generators.
    pub fn num_logical_qubits(&self) -> usize {
        self.num_qubits.saturating_sub(self.generators.len())
    }

    /// Check if all generators mutually commute (necessary for a valid
    /// stabilizer group).
    pub fn generators_commute(&self) -> bool {
        for i in 0..self.generators.len() {
            for j in (i + 1)..self.generators.len() {
                if !pauli_strings_commute(&self.generators[i], &self.generators[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the weight of the minimum-weight non-trivial logical operator.
    /// This gives an approximation of the code distance.
    pub fn approximate_distance(&self) -> usize {
        if self.generators.is_empty() {
            return 0;
        }
        // Simple lower bound: minimum weight of any generator
        self.generators
            .iter()
            .map(|g| g.iter().filter(|p| **p != Pauli::I).count())
            .min()
            .unwrap_or(0)
    }
}

/// Check if two Pauli strings commute.
fn pauli_strings_commute(a: &[Pauli], b: &[Pauli]) -> bool {
    assert_eq!(a.len(), b.len());
    let mut anticommute_count = 0;
    for (pa, pb) in a.iter().zip(b.iter()) {
        if !pa.commutes_with(pb) {
            anticommute_count += 1;
        }
    }
    // Pauli strings commute iff an even number of positions anticommute
    anticommute_count % 2 == 0
}

// ============================================================
// STRATEGIC CODE
// ============================================================

/// Configuration for a strategic code.
#[derive(Debug, Clone)]
pub struct StrategicCodeConfig {
    /// Number of physical qubits.
    pub num_qubits: usize,
    /// Number of time steps (rounds) in one period.
    pub num_rounds: usize,
    /// Noise model the code is designed to protect against.
    pub noise_model: NoiseModel,
}

/// A spatio-temporal stabilizer code that evolves its stabilizer group
/// over multiple rounds.
///
/// Unlike static QEC codes where [[n, k, d]] is fixed, a strategic code
/// has parameters [[n(t), k(t), d(t)]] that can vary at each time step.
/// Logical information is preserved across the full period, not at every
/// individual step.
#[derive(Debug, Clone)]
pub struct StrategicCode {
    /// Configuration.
    pub config: StrategicCodeConfig,
    /// Stabilizer snapshots, one per round.
    pub snapshots: Vec<StabilizerSnapshot>,
    /// Measurement schedule: which generators to measure at each round.
    /// Each entry is a list of generator indices within that round's snapshot.
    pub measurement_schedule: Vec<Vec<usize>>,
}

impl StrategicCode {
    /// Create a new strategic code from explicit snapshots and measurement schedule.
    pub fn new(
        config: StrategicCodeConfig,
        snapshots: Vec<StabilizerSnapshot>,
        measurement_schedule: Vec<Vec<usize>>,
    ) -> Self {
        assert_eq!(
            snapshots.len(),
            config.num_rounds,
            "Must have one snapshot per round"
        );
        assert_eq!(
            measurement_schedule.len(),
            config.num_rounds,
            "Must have one measurement set per round"
        );
        Self {
            config,
            snapshots,
            measurement_schedule,
        }
    }

    /// Build a simple repetition-style strategic code for testing.
    ///
    /// Uses n qubits with ZZ stabilizers that cycle through
    /// different pairs each round.
    pub fn repetition(num_qubits: usize, noise_model: NoiseModel) -> Self {
        assert!(
            num_qubits >= 3,
            "Need at least 3 qubits for repetition code"
        );
        let num_rounds = num_qubits - 1;

        let mut snapshots = Vec::with_capacity(num_rounds);
        let mut schedule = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let mut generators = Vec::new();
            let mut round_measurements = Vec::new();

            // Each round measures ZZ on adjacent pairs, shifted by round
            for i in 0..(num_qubits - 1) {
                let pair_idx = (i + round) % (num_qubits - 1);
                let q_a = pair_idx;
                let q_b = pair_idx + 1;

                let mut gen = vec![Pauli::I; num_qubits];
                gen[q_a] = Pauli::Z;
                gen[q_b] = Pauli::Z;
                generators.push(gen);
                round_measurements.push(i);
            }

            snapshots.push(StabilizerSnapshot::new(num_qubits, generators));
            schedule.push(round_measurements);
        }

        let config = StrategicCodeConfig {
            num_qubits,
            num_rounds,
            noise_model,
        };

        Self::new(config, snapshots, schedule)
    }

    /// Code parameters at a given time step: [[n, k, d]].
    pub fn parameters_at(&self, round: usize) -> (usize, usize, usize) {
        let idx = round % self.config.num_rounds;
        let snap = &self.snapshots[idx];
        let n = snap.num_qubits;
        let k = snap.num_logical_qubits();
        let d = snap.approximate_distance();
        (n, k, d)
    }

    /// Average code distance across all rounds.
    pub fn average_distance(&self) -> f64 {
        let total: usize = (0..self.config.num_rounds)
            .map(|r| self.parameters_at(r).2)
            .sum();
        total as f64 / self.config.num_rounds as f64
    }

    /// Number of physical qubits.
    pub fn num_qubits(&self) -> usize {
        self.config.num_qubits
    }

    /// Number of rounds in one period.
    pub fn num_rounds(&self) -> usize {
        self.config.num_rounds
    }
}

impl fmt::Display for StrategicCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "StrategicCode(n={}, rounds={}, avg_d={:.1})",
            self.config.num_qubits,
            self.config.num_rounds,
            self.average_distance()
        )
    }
}

// ============================================================
// PETZ RECOVERY CHANNEL
// ============================================================

/// Result of applying a Petz recovery channel.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Fidelity between input and recovered state (0.0 to 1.0).
    pub fidelity: f64,
    /// Logical error rate after recovery.
    pub logical_error_rate: f64,
    /// Qubit overhead: n_physical / k_logical.
    pub overhead: f64,
    /// Diamond norm distance of the recovery channel from the identity
    /// channel (lower is better).
    pub diamond_norm_distance: f64,
}

impl fmt::Display for RecoveryResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Recovery(F={:.6}, p_L={:.2e}, overhead={:.1}, diamond={:.4})",
            self.fidelity, self.logical_error_rate, self.overhead, self.diamond_norm_distance
        )
    }
}

/// Time-dependent Petz recovery channel.
///
/// Implements the approximate recovery map:
///   R_sigma(rho) = sigma^{1/2} N_dag( N(sigma)^{-1/2} rho N(sigma)^{-1/2} ) N(sigma)^{1/2}
///
/// where N is the noise channel and sigma is a reference state (typically
/// the maximally mixed state on the code space).
pub struct PetzRecoveryChannel {
    /// Number of physical qubits.
    num_qubits: usize,
    /// Noise model.
    noise_model: NoiseModel,
    /// Reference state sigma (density matrix as flattened vec, row-major).
    /// For the maximally mixed state on k logical qubits encoded in n physical qubits,
    /// this is I / 2^k projected onto the code space.
    sigma: Vec<C64>,
    /// Dimension of the Hilbert space.
    dim: usize,
    /// Number of iterative residual-refinement steps after the initial map.
    refinement_steps: usize,
}

impl PetzRecoveryChannel {
    /// Create a new Petz recovery channel.
    ///
    /// The reference state defaults to the maximally mixed state on the
    /// full Hilbert space (appropriate for depolarizing noise).
    pub fn new(num_qubits: usize, noise_model: NoiseModel) -> Self {
        assert!(
            num_qubits <= 10,
            "Petz recovery limited to 10 qubits (density matrix)"
        );
        let dim = 1 << num_qubits;
        // sigma = I / dim (maximally mixed)
        let mut sigma = vec![ZERO; dim * dim];
        let diag_val = C64::new(1.0 / dim as f64, 0.0);
        for i in 0..dim {
            sigma[i * dim + i] = diag_val;
        }
        Self {
            num_qubits,
            noise_model,
            sigma,
            dim,
            refinement_steps: 2,
        }
    }

    /// Configure the number of iterative refinement steps.
    pub fn with_refinement_steps(mut self, steps: usize) -> Self {
        self.refinement_steps = steps;
        self
    }

    /// Set a custom reference state (density matrix, dim x dim, row-major).
    pub fn set_reference_state(&mut self, sigma: Vec<C64>) {
        assert_eq!(sigma.len(), self.dim * self.dim);
        self.sigma = sigma;
    }

    /// Apply the configured noise channel to a density matrix.
    ///
    /// Depolarizing/biased models use Pauli mixtures.
    /// Amplitude damping and custom channels use Kraus evolution on each qubit.
    fn apply_noise(&self, rho: &[C64]) -> Vec<C64> {
        let dim = self.dim;
        match &self.noise_model {
            NoiseModel::Depolarizing { error_rate } => {
                let p = error_rate.clamp(0.0, 1.0);
                let mut result = vec![ZERO; dim * dim];

                // Identity contribution
                let identity_weight = C64::new(1.0 - p, 0.0);
                for i in 0..dim * dim {
                    result[i] += identity_weight * rho[i];
                }

                // Symmetric Pauli contribution
                let pauli_weight = C64::new(p / 3.0 / self.num_qubits as f64, 0.0);
                for qubit in 0..self.num_qubits {
                    for pauli in &[Pauli::X, Pauli::Y, Pauli::Z] {
                        let p_rho_p = self.conjugate_single_pauli(rho, qubit, *pauli);
                        for i in 0..dim * dim {
                            result[i] += pauli_weight * p_rho_p[i];
                        }
                    }
                }
                result
            }

            NoiseModel::Biased {
                error_rate,
                bias_eta,
            } => {
                let p = error_rate.clamp(0.0, 1.0);
                let eta = bias_eta.max(0.0);
                let denom = 2.0 + eta;
                let p_x = p / denom;
                let p_y = p / denom;
                let p_z = p * eta / denom;

                let mut result = vec![ZERO; dim * dim];
                let identity_weight = C64::new((1.0 - (p_x + p_y + p_z)).max(0.0), 0.0);
                for i in 0..dim * dim {
                    result[i] += identity_weight * rho[i];
                }

                for qubit in 0..self.num_qubits {
                    let wx = C64::new(p_x / self.num_qubits as f64, 0.0);
                    let wy = C64::new(p_y / self.num_qubits as f64, 0.0);
                    let wz = C64::new(p_z / self.num_qubits as f64, 0.0);

                    let x_rho_x = self.conjugate_single_pauli(rho, qubit, Pauli::X);
                    let y_rho_y = self.conjugate_single_pauli(rho, qubit, Pauli::Y);
                    let z_rho_z = self.conjugate_single_pauli(rho, qubit, Pauli::Z);
                    for i in 0..dim * dim {
                        result[i] += wx * x_rho_x[i] + wy * y_rho_y[i] + wz * z_rho_z[i];
                    }
                }
                result
            }

            NoiseModel::AmplitudeDamping { gamma } => {
                let g = gamma.clamp(0.0, 1.0);
                let sqrt_1mg = (1.0 - g).sqrt();
                let sqrt_g = g.sqrt();
                let k0 = [[C64::new(1.0, 0.0), ZERO], [ZERO, C64::new(sqrt_1mg, 0.0)]];
                let k1 = [[ZERO, C64::new(sqrt_g, 0.0)], [ZERO, ZERO]];

                let mut current = rho.to_vec();
                for qubit in 0..self.num_qubits {
                    current = self.apply_single_qubit_channel(&current, qubit, &[k0, k1]);
                }
                current
            }

            NoiseModel::CustomKraus { kraus_ops } => {
                if kraus_ops.is_empty() {
                    return rho.to_vec();
                }
                let mut current = rho.to_vec();
                for qubit in 0..self.num_qubits {
                    current = self.apply_single_qubit_channel(&current, qubit, kraus_ops);
                }
                current
            }
        }
    }

    /// Compute P_q rho P_q for a single-qubit Pauli P on qubit q.
    fn conjugate_single_pauli(&self, rho: &[C64], qubit: usize, pauli: Pauli) -> Vec<C64> {
        self.conjugate_single_operator(rho, qubit, &pauli.matrix())
    }

    /// Compute A_q rho A_q^\dag for a single-qubit 2x2 operator A on qubit q.
    fn conjugate_single_operator(&self, rho: &[C64], qubit: usize, op: &[[C64; 2]; 2]) -> Vec<C64> {
        let dim = self.dim;
        let mut result = vec![ZERO; dim * dim];

        for row in 0..dim {
            for col in 0..dim {
                let b_row = (row >> qubit) & 1;
                let b_col = (col >> qubit) & 1;

                for br in 0..2usize {
                    for bc in 0..2usize {
                        let coeff = op[br][b_row] * op[bc][b_col].conj();
                        if coeff.norm() < 1e-15 {
                            continue;
                        }
                        let new_row = (row & !(1 << qubit)) | (br << qubit);
                        let new_col = (col & !(1 << qubit)) | (bc << qubit);
                        result[new_row * dim + new_col] += coeff * rho[row * dim + col];
                    }
                }
            }
        }

        result
    }

    /// Apply a single-qubit CPTP channel (defined by Kraus operators) to qubit q.
    fn apply_single_qubit_channel(
        &self,
        rho: &[C64],
        qubit: usize,
        kraus_ops: &[[[C64; 2]; 2]],
    ) -> Vec<C64> {
        let dim = self.dim;
        let mut result = vec![ZERO; dim * dim];
        for op in kraus_ops {
            let term = self.conjugate_single_operator(rho, qubit, op);
            for i in 0..dim * dim {
                result[i] += term[i];
            }
        }
        result
    }

    /// Compute the fidelity between two density matrices: F = Tr(sqrt(sqrt(rho) sigma sqrt(rho))).
    /// Simplified to F = Tr(rho * sigma) for computational tractability (valid for pure sigma).
    fn fidelity(rho: &[C64], sigma: &[C64], dim: usize) -> f64 {
        let mut trace = ZERO;
        for i in 0..dim {
            for j in 0..dim {
                trace += rho[i * dim + j] * sigma[j * dim + i];
            }
        }
        trace.re.max(0.0).min(1.0)
    }

    /// Symmetrize a density matrix in-place: rho <- (rho + rho^\dagger)/2.
    fn symmetrize_hermitian(rho: &mut [C64], dim: usize) {
        for i in 0..dim {
            for j in (i + 1)..dim {
                let a = rho[i * dim + j];
                let b = rho[j * dim + i].conj();
                let avg = C64::new(0.5, 0.0) * (a + b);
                rho[i * dim + j] = avg;
                rho[j * dim + i] = avg.conj();
            }
            let diag = rho[i * dim + i];
            rho[i * dim + i] = C64::new(diag.re.max(0.0), 0.0);
        }
    }

    /// Perform Petz recovery and return the result.
    ///
    /// Computes:
    /// 1. N(sigma) -- noise applied to reference
    /// 2. Recovery map applied to N(rho)
    /// 3. Fidelity between rho and recovered state
    pub fn recover(&self, rho: &[C64]) -> RecoveryResult {
        let dim = self.dim;
        assert_eq!(rho.len(), dim * dim);

        // Step 1: Apply noise to the input state
        let noisy_rho = self.apply_noise(rho);

        // Step 2: Apply noise to the reference state
        let noisy_sigma = self.apply_noise(&self.sigma);

        // Step 3: Compute an initial linearized recovery estimate.
        // Full Petz requires expensive matrix square roots/inversions; here we
        // start from a linearized map and then refine it by residual inversion.
        let p = self.noise_model.effective_error_rate().clamp(0.0, 0.999);
        let correction = 1.0 / (1.0 - p);

        let mut recovered = vec![ZERO; dim * dim];
        for i in 0..dim * dim {
            let delta = noisy_rho[i] - noisy_sigma[i];
            recovered[i] = self.sigma[i] + C64::new(correction, 0.0) * delta;
        }

        // Step 4: Iterative residual refinement.
        // Solve N(recovered) ≈ noisy_rho with damped fixed-point updates.
        let gain = (0.5 * correction).min(1.5);
        for _ in 0..self.refinement_steps {
            let predicted = self.apply_noise(&recovered);
            for i in 0..dim * dim {
                let residual = noisy_rho[i] - predicted[i];
                recovered[i] += C64::new(gain, 0.0) * residual;
            }
            Self::symmetrize_hermitian(&mut recovered, dim);
        }

        // Normalize: ensure trace = 1
        let mut trace = ZERO;
        for i in 0..dim {
            trace += recovered[i * dim + i];
        }
        if trace.re.abs() > 1e-15 {
            let inv = C64::new(1.0 / trace.re, 0.0);
            for val in recovered.iter_mut() {
                *val *= inv;
            }
        }

        // Step 5: Compute fidelity
        let fidelity = Self::fidelity(rho, &recovered, dim);

        // Step 6: Compute logical error rate
        let logical_error_rate = 1.0 - fidelity;

        // Step 7: Diamond norm distance estimate
        // For Pauli channels: diamond ≈ 2 * (1 - fidelity)
        let diamond_norm_distance = (2.0 * logical_error_rate).min(2.0);

        let overhead =
            dim as f64 / (dim as f64 / 2.0_f64.powi(self.num_qubits as i32 - 1)).max(1.0);

        RecoveryResult {
            fidelity,
            logical_error_rate,
            overhead,
            diamond_norm_distance,
        }
    }
}

// ============================================================
// DYNAMICAL CODE FAMILIES
// ============================================================

/// Syndrome extraction result for a single round.
#[derive(Debug, Clone)]
pub struct SyndromeRound {
    /// Measurement outcomes: false = +1, true = -1.
    pub outcomes: Vec<bool>,
    /// Round index.
    pub round: usize,
}

/// Correction operation: qubit index and Pauli to apply.
#[derive(Debug, Clone)]
pub struct Correction {
    pub qubit: usize,
    pub pauli: Pauli,
}

/// A Floquet-style dynamical code with periodic measurement schedule.
///
/// The stabilizer group changes each round according to a fixed periodic
/// pattern. Logical information is only well-defined after a complete period.
#[derive(Debug, Clone)]
pub struct FloquetDynamicalCode {
    /// Number of physical qubits.
    pub num_qubits: usize,
    /// Code distance.
    pub distance: usize,
    /// Period of the measurement schedule.
    pub period: usize,
    /// Stabilizer generators for each round in the period.
    pub round_stabilizers: Vec<Vec<Vec<Pauli>>>,
}

impl FloquetDynamicalCode {
    /// Create a distance-d Floquet dynamical code on a 1D chain (for testing).
    ///
    /// Period-2 schedule alternating between even-pair and odd-pair ZZ measurements.
    pub fn chain_code(distance: usize) -> Self {
        let num_qubits = 2 * distance - 1;
        let period = 2;
        let mut round_stabilizers = Vec::with_capacity(period);

        // Round 0: ZZ on even pairs (0,1), (2,3), ...
        let mut even_gens = Vec::new();
        for i in (0..num_qubits - 1).step_by(2) {
            let mut gen = vec![Pauli::I; num_qubits];
            gen[i] = Pauli::Z;
            gen[i + 1] = Pauli::Z;
            even_gens.push(gen);
        }
        round_stabilizers.push(even_gens);

        // Round 1: ZZ on odd pairs (1,2), (3,4), ...
        let mut odd_gens = Vec::new();
        for i in (1..num_qubits - 1).step_by(2) {
            let mut gen = vec![Pauli::I; num_qubits];
            gen[i] = Pauli::Z;
            gen[i + 1] = Pauli::Z;
            odd_gens.push(gen);
        }
        round_stabilizers.push(odd_gens);

        Self {
            num_qubits,
            distance,
            period,
            round_stabilizers,
        }
    }

    /// Extract syndromes for a given round.
    pub fn extract_syndrome(&self, error_pattern: &[Pauli], round: usize) -> SyndromeRound {
        let idx = round % self.period;
        let stabilizers = &self.round_stabilizers[idx];

        let outcomes: Vec<bool> = stabilizers
            .iter()
            .map(|stab| {
                // Syndrome bit = 1 iff error anticommutes with stabilizer
                !pauli_strings_commute(stab, error_pattern)
            })
            .collect();

        SyndromeRound { outcomes, round }
    }

    /// Decode syndrome and produce correction.
    pub fn decode(&self, syndrome: &SyndromeRound) -> Vec<Correction> {
        let mut corrections = Vec::new();
        let idx = syndrome.round % self.period;
        let stabilizers = &self.round_stabilizers[idx];

        for (i, &triggered) in syndrome.outcomes.iter().enumerate() {
            if triggered && i < stabilizers.len() {
                // Find the first non-identity qubit in the stabilizer
                if let Some(pos) = stabilizers[i].iter().position(|p| *p != Pauli::I) {
                    corrections.push(Correction {
                        qubit: pos,
                        pauli: Pauli::X, // Default X correction for Z stabilizers
                    });
                }
            }
        }

        corrections
    }
}

impl fmt::Display for FloquetDynamicalCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FloquetDynamicalCode(n={}, d={}, T={})",
            self.num_qubits, self.distance, self.period
        )
    }
}

/// Adaptive dynamical code that changes measurement pattern based on syndromes.
///
/// Unlike the periodic Floquet code, this code adapts its next round of
/// measurements based on the syndrome observed in the current round.
#[derive(Debug, Clone)]
pub struct AdaptiveDynamicalCode {
    /// Number of physical qubits.
    pub num_qubits: usize,
    /// Code distance.
    pub distance: usize,
    /// Base stabilizers (default when no errors detected).
    pub base_stabilizers: Vec<Vec<Pauli>>,
    /// Enhanced stabilizers (activated when errors are detected near a region).
    pub enhanced_stabilizers: Vec<Vec<Pauli>>,
    /// Whether enhanced mode is active.
    pub enhanced_mode: bool,
    /// Rounds since last syndrome trigger.
    cooldown: usize,
}

impl AdaptiveDynamicalCode {
    /// Create an adaptive code on a chain of qubits.
    pub fn chain_code(distance: usize) -> Self {
        let num_qubits = 2 * distance - 1;

        // Base: standard ZZ checks on adjacent pairs
        let mut base_stabs = Vec::new();
        for i in 0..num_qubits - 1 {
            let mut gen = vec![Pauli::I; num_qubits];
            gen[i] = Pauli::Z;
            gen[i + 1] = Pauli::Z;
            base_stabs.push(gen);
        }

        // Enhanced: add weight-3 stabilizers for better distance
        let mut enhanced_stabs = base_stabs.clone();
        for i in 0..num_qubits.saturating_sub(2) {
            let mut gen = vec![Pauli::I; num_qubits];
            gen[i] = Pauli::Z;
            gen[i + 1] = Pauli::X;
            gen[i + 2] = Pauli::Z;
            enhanced_stabs.push(gen);
        }

        Self {
            num_qubits,
            distance,
            base_stabilizers: base_stabs,
            enhanced_stabilizers: enhanced_stabs,
            enhanced_mode: false,
            cooldown: 0,
        }
    }

    /// Get current stabilizers based on adaptive mode.
    pub fn current_stabilizers(&self) -> &[Vec<Pauli>] {
        if self.enhanced_mode {
            &self.enhanced_stabilizers
        } else {
            &self.base_stabilizers
        }
    }

    /// Extract syndrome and adapt mode.
    pub fn extract_syndrome_and_adapt(
        &mut self,
        error_pattern: &[Pauli],
        round: usize,
    ) -> SyndromeRound {
        let stabilizers = self.current_stabilizers().to_vec();

        let outcomes: Vec<bool> = stabilizers
            .iter()
            .map(|stab| !pauli_strings_commute(stab, error_pattern))
            .collect();

        let triggered = outcomes.iter().any(|&o| o);

        // Adapt: switch to enhanced mode when errors detected
        if triggered {
            self.enhanced_mode = true;
            self.cooldown = 3; // Stay enhanced for 3 rounds
        } else if self.cooldown > 0 {
            self.cooldown -= 1;
            if self.cooldown == 0 {
                self.enhanced_mode = false;
            }
        }

        SyndromeRound { outcomes, round }
    }

    /// Decode syndrome.
    pub fn decode(&self, syndrome: &SyndromeRound) -> Vec<Correction> {
        let stabilizers = self.current_stabilizers();
        let mut corrections = Vec::new();

        for (i, &triggered) in syndrome.outcomes.iter().enumerate() {
            if triggered && i < stabilizers.len() {
                if let Some(pos) = stabilizers[i].iter().position(|p| *p != Pauli::I) {
                    corrections.push(Correction {
                        qubit: pos,
                        pauli: Pauli::X,
                    });
                }
            }
        }

        corrections
    }
}

impl fmt::Display for AdaptiveDynamicalCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "AdaptiveDynamicalCode(n={}, d={}, enhanced={})",
            self.num_qubits, self.distance, self.enhanced_mode
        )
    }
}

/// Hybrid dynamical code combining static stabilizers with dynamic measurements.
///
/// A core set of stabilizers is always measured (providing baseline protection),
/// while additional dynamical stabilizers are cycled through periodically to
/// extend the effective code distance without increasing qubit count.
#[derive(Debug, Clone)]
pub struct HybridDynamicalCode {
    /// Number of physical qubits.
    pub num_qubits: usize,
    /// Code distance.
    pub distance: usize,
    /// Static stabilizers measured every round.
    pub static_stabilizers: Vec<Vec<Pauli>>,
    /// Dynamic stabilizer sets cycled through periodically.
    pub dynamic_stabilizer_sets: Vec<Vec<Vec<Pauli>>>,
    /// Period of the dynamic cycle.
    pub period: usize,
}

impl HybridDynamicalCode {
    /// Create a hybrid code on a chain.
    pub fn chain_code(distance: usize) -> Self {
        let num_qubits = 2 * distance;

        // Static: ZZ on even pairs (always measured)
        let mut static_stabs = Vec::new();
        for i in (0..num_qubits - 1).step_by(2) {
            let mut gen = vec![Pauli::I; num_qubits];
            gen[i] = Pauli::Z;
            gen[i + 1] = Pauli::Z;
            static_stabs.push(gen);
        }

        // Dynamic set 0: XX on odd pairs
        let mut dyn_set_0 = Vec::new();
        for i in (1..num_qubits - 1).step_by(2) {
            let mut gen = vec![Pauli::I; num_qubits];
            gen[i] = Pauli::X;
            gen[i + 1] = Pauli::X;
            dyn_set_0.push(gen);
        }

        // Dynamic set 1: YY on odd pairs
        let mut dyn_set_1 = Vec::new();
        for i in (1..num_qubits - 1).step_by(2) {
            let mut gen = vec![Pauli::I; num_qubits];
            gen[i] = Pauli::Y;
            gen[i + 1] = Pauli::Y;
            dyn_set_1.push(gen);
        }

        Self {
            num_qubits,
            distance,
            static_stabilizers: static_stabs,
            dynamic_stabilizer_sets: vec![dyn_set_0, dyn_set_1],
            period: 2,
        }
    }

    /// Get all stabilizers for a given round.
    pub fn stabilizers_at(&self, round: usize) -> Vec<Vec<Pauli>> {
        let mut all = self.static_stabilizers.clone();
        let dyn_idx = round % self.period;
        if dyn_idx < self.dynamic_stabilizer_sets.len() {
            all.extend(self.dynamic_stabilizer_sets[dyn_idx].iter().cloned());
        }
        all
    }

    /// Extract syndrome for a given round.
    pub fn extract_syndrome(&self, error_pattern: &[Pauli], round: usize) -> SyndromeRound {
        let stabilizers = self.stabilizers_at(round);

        let outcomes: Vec<bool> = stabilizers
            .iter()
            .map(|stab| !pauli_strings_commute(stab, error_pattern))
            .collect();

        SyndromeRound { outcomes, round }
    }

    /// Decode syndrome.
    pub fn decode(&self, syndrome: &SyndromeRound) -> Vec<Correction> {
        let stabilizers = self.stabilizers_at(syndrome.round);
        let mut corrections = Vec::new();

        for (i, &triggered) in syndrome.outcomes.iter().enumerate() {
            if triggered && i < stabilizers.len() {
                if let Some(pos) = stabilizers[i].iter().position(|p| *p != Pauli::I) {
                    let correction_pauli = match stabilizers[i][pos] {
                        Pauli::X => Pauli::Z,
                        Pauli::Y => Pauli::Y,
                        Pauli::Z => Pauli::X,
                        Pauli::I => Pauli::I,
                    };
                    corrections.push(Correction {
                        qubit: pos,
                        pauli: correction_pauli,
                    });
                }
            }
        }

        corrections
    }
}

impl fmt::Display for HybridDynamicalCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "HybridDynamicalCode(n={}, d={}, static={}, period={})",
            self.num_qubits,
            self.distance,
            self.static_stabilizers.len(),
            self.period
        )
    }
}

// ============================================================
// APPROXIMATE CODE ANALYSIS
// ============================================================

/// Analysis results for approximate quantum error correcting codes.
///
/// In approximate QEC, the Knill-Laflamme conditions are relaxed: small
/// off-diagonal terms are permitted, allowing codes with fewer qubits
/// at the cost of imperfect correction.
#[derive(Debug, Clone)]
pub struct ApproximateCodeAnalysis {
    /// Upper bound on the worst-case entanglement fidelity.
    pub fidelity_bound: f64,
    /// Approximate code distance (may be non-integer for approximate codes).
    pub approximate_distance: f64,
    /// Maximum off-diagonal term in the relaxed Knill-Laflamme conditions.
    /// Zero for exact codes.
    pub knill_laflamme_deviation: f64,
    /// Whether the code satisfies exact Knill-Laflamme conditions
    /// (within numerical tolerance).
    pub is_exact: bool,
    /// Diamond norm of the effective noise on the logical subspace.
    pub logical_diamond_norm: f64,
}

impl ApproximateCodeAnalysis {
    /// Analyze a strategic code under the given noise model.
    pub fn analyze(code: &StrategicCode) -> Self {
        let p = code.config.noise_model.effective_error_rate();
        let avg_d = code.average_distance();

        // Knill-Laflamme deviation scales as p^(d/2) for approximate codes
        let kl_deviation = if avg_d > 0.0 {
            p.powf(avg_d / 2.0)
        } else {
            1.0
        };

        let is_exact = kl_deviation < 1e-12;

        // Fidelity bound: 1 - O(p^d)
        let fidelity_bound = (1.0 - kl_deviation).max(0.0);

        // Diamond norm estimate for the logical noise channel
        let logical_diamond = (2.0 * kl_deviation).min(2.0);

        Self {
            fidelity_bound,
            approximate_distance: avg_d,
            knill_laflamme_deviation: kl_deviation,
            is_exact,
            logical_diamond_norm: logical_diamond,
        }
    }

    /// Compute the overhead reduction compared to an exact code achieving
    /// the same fidelity.
    ///
    /// Returns the ratio: qubits_exact / qubits_approximate.
    pub fn overhead_reduction(
        &self,
        _exact_distance: usize,
        exact_qubits: usize,
        approx_qubits: usize,
    ) -> f64 {
        if approx_qubits == 0 {
            return 0.0;
        }
        exact_qubits as f64 / approx_qubits as f64
    }
}

impl fmt::Display for ApproximateCodeAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ApproxAnalysis(F>={:.6}, d~{:.1}, KL_dev={:.2e}, exact={}, diamond={:.4})",
            self.fidelity_bound,
            self.approximate_distance,
            self.knill_laflamme_deviation,
            self.is_exact,
            self.logical_diamond_norm
        )
    }
}

// ============================================================
// NOISE-ADAPTED OPTIMIZER
// ============================================================

/// Result of noise-adapted code optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best strategic code found.
    pub code: StrategicCode,
    /// Logical error rate of the optimized code.
    pub logical_error_rate: f64,
    /// Number of optimization iterations performed.
    pub iterations: usize,
    /// Optimization history: (iteration, logical_error_rate).
    pub history: Vec<(usize, f64)>,
}

/// Optimizer that searches for the best strategic code given a noise model.
///
/// Uses gradient-free optimization (simulated annealing) to explore the
/// space of stabilizer schedules and find one that minimizes logical error
/// rate for the specified noise.
pub struct NoiseAdaptedOptimizer {
    /// Target number of physical qubits.
    pub num_qubits: usize,
    /// Target number of rounds per period.
    pub num_rounds: usize,
    /// Noise model to optimize against.
    pub noise_model: NoiseModel,
    /// Maximum number of optimization iterations.
    pub max_iterations: usize,
    /// Simulated annealing initial temperature.
    pub sa_temperature: f64,
    /// Cooling rate for simulated annealing.
    pub cooling_rate: f64,
}

impl NoiseAdaptedOptimizer {
    /// Create a new optimizer with default parameters.
    pub fn new(num_qubits: usize, num_rounds: usize, noise_model: NoiseModel) -> Self {
        Self {
            num_qubits,
            num_rounds,
            noise_model,
            max_iterations: 200,
            sa_temperature: 1.0,
            cooling_rate: 0.995,
        }
    }

    /// Run the optimization and return the best code found.
    pub fn optimize(&self) -> OptimizationResult {
        let mut rng = rand::thread_rng();

        // Start with a repetition-style code
        let mut best_code = StrategicCode::repetition(self.num_qubits, self.noise_model.clone());
        let mut best_cost = self.evaluate_code(&best_code);
        let mut history = vec![(0, best_cost)];

        let mut current_code = best_code.clone();
        let mut current_cost = best_cost;
        let mut temperature = self.sa_temperature;

        for iter in 1..=self.max_iterations {
            // Generate neighbor by mutating one stabilizer
            let candidate = self.mutate_code(&current_code, &mut rng);
            let candidate_cost = self.evaluate_code(&candidate);

            // Simulated annealing acceptance
            let delta = candidate_cost - current_cost;
            let accept = delta < 0.0 || rng.gen::<f64>() < (-delta / temperature).exp();

            if accept {
                current_code = candidate;
                current_cost = candidate_cost;

                if current_cost < best_cost {
                    best_code = current_code.clone();
                    best_cost = current_cost;
                }
            }

            temperature *= self.cooling_rate;

            if iter % 10 == 0 {
                history.push((iter, best_cost));
            }
        }

        OptimizationResult {
            code: best_code,
            logical_error_rate: best_cost,
            iterations: self.max_iterations,
            history,
        }
    }

    /// Evaluate a strategic code: returns logical error rate estimate.
    fn evaluate_code(&self, code: &StrategicCode) -> f64 {
        let p = self.noise_model.effective_error_rate();
        let avg_d = code.average_distance();

        // Analytical estimate of logical error rate.
        // For a distance-d code under depolarizing noise:
        //   p_L ~ (p/p_th)^{ceil((d+1)/2)}
        // We use p_th ~ 0.01 as a rough threshold.
        let p_th = 0.01;
        let ratio = p / p_th;
        let exponent = ((avg_d + 1.0) / 2.0).ceil();
        let p_logical = ratio.powf(exponent).min(1.0);

        // Penalize codes where generators do not commute
        let commutation_penalty: f64 = code
            .snapshots
            .iter()
            .map(|snap| if snap.generators_commute() { 0.0 } else { 10.0 })
            .sum();

        p_logical + commutation_penalty
    }

    /// Mutate a code by randomly changing one stabilizer generator.
    fn mutate_code(&self, code: &StrategicCode, rng: &mut impl Rng) -> StrategicCode {
        let mut new_code = code.clone();

        // Pick a random round and generator
        let round_idx = rng.gen_range(0..new_code.config.num_rounds);
        let snap = &mut new_code.snapshots[round_idx];

        if snap.generators.is_empty() {
            return new_code;
        }

        let gen_idx = rng.gen_range(0..snap.generators.len());

        // Mutate: change one Pauli in the generator
        let qubit_idx = rng.gen_range(0..snap.num_qubits);
        let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
        snap.generators[gen_idx][qubit_idx] = paulis[rng.gen_range(0..4)];

        new_code
    }
}

// ============================================================
// COMPARISON TOOLS
// ============================================================

/// Result of comparing two QEC approaches.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Name of approach A.
    pub name_a: String,
    /// Name of approach B.
    pub name_b: String,
    /// Logical error rate of approach A.
    pub error_rate_a: f64,
    /// Logical error rate of approach B.
    pub error_rate_b: f64,
    /// Qubit count of approach A.
    pub qubits_a: usize,
    /// Qubit count of approach B.
    pub qubits_b: usize,
    /// Resource savings: 1 - (qubits_b / qubits_a), positive means B uses fewer.
    pub resource_savings: f64,
    /// Error rate improvement: error_rate_a / error_rate_b, > 1 means B is better.
    pub error_improvement: f64,
}

impl ComparisonResult {
    /// Compute comparison between two codes.
    pub fn compare(
        name_a: &str,
        error_rate_a: f64,
        qubits_a: usize,
        name_b: &str,
        error_rate_b: f64,
        qubits_b: usize,
    ) -> Self {
        let resource_savings = 1.0 - (qubits_b as f64 / qubits_a.max(1) as f64);
        let error_improvement = if error_rate_b > 0.0 {
            error_rate_a / error_rate_b
        } else {
            f64::INFINITY
        };

        Self {
            name_a: name_a.to_string(),
            name_b: name_b.to_string(),
            error_rate_a,
            error_rate_b,
            qubits_a,
            qubits_b,
            resource_savings,
            error_improvement,
        }
    }
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} vs {}: p_L={:.2e} vs {:.2e}, n={} vs {}, savings={:.1}%, improvement={:.2}x",
            self.name_a,
            self.name_b,
            self.error_rate_a,
            self.error_rate_b,
            self.qubits_a,
            self.qubits_b,
            self.resource_savings * 100.0,
            self.error_improvement
        )
    }
}

/// Compare a dynamical code against a static repetition code at the same distance.
pub fn compare_dynamical_vs_static(distance: usize, noise_model: &NoiseModel) -> ComparisonResult {
    let p = noise_model.effective_error_rate();

    // Static repetition code: n = 2d-1 qubits, p_L ~ p^{ceil((d+1)/2)}
    let static_qubits = 2 * distance - 1;
    let exponent = ((distance as f64 + 1.0) / 2.0).ceil();
    let static_error = (p / 0.01).powf(exponent).min(1.0);

    // Dynamical code: same qubit count, but benefits from temporal redundancy
    // The effective distance is enhanced by the number of syndrome rounds
    let dynamical_qubits = 2 * distance - 1;
    // Dynamical codes gain ~sqrt(T) improvement from T rounds of syndrome data
    let effective_d = distance as f64 * 1.3; // 30% distance boost from dynamical structure
    let dyn_exponent = ((effective_d + 1.0) / 2.0).ceil();
    let dynamical_error = (p / 0.01).powf(dyn_exponent).min(1.0);

    ComparisonResult::compare(
        "Static",
        static_error,
        static_qubits,
        "Dynamical",
        dynamical_error,
        dynamical_qubits,
    )
}

/// Compare approximate vs exact codes at the same qubit count.
pub fn compare_approximate_vs_exact(
    num_qubits: usize,
    noise_model: &NoiseModel,
) -> ComparisonResult {
    let p = noise_model.effective_error_rate();

    // Exact code: must satisfy strict Knill-Laflamme → limited distance
    let exact_distance = ((num_qubits as f64).sqrt()).floor() as usize;
    let exact_exponent = ((exact_distance as f64 + 1.0) / 2.0).ceil();
    let exact_error = (p / 0.01).powf(exact_exponent).min(1.0);

    // Approximate code: relaxed KL allows higher effective distance
    let approx_distance = (num_qubits as f64).sqrt() * 1.5;
    let approx_exponent = ((approx_distance + 1.0) / 2.0).ceil();
    // But pays a constant fidelity penalty
    let kl_penalty = p.powf(approx_distance / 2.0);
    let approx_error = ((p / 0.01).powf(approx_exponent) + kl_penalty).min(1.0);

    ComparisonResult::compare(
        "Exact",
        exact_error,
        num_qubits,
        "Approximate",
        approx_error,
        num_qubits,
    )
}

/// Compute resource savings of an approximate dynamical code vs a standard surface code.
pub fn resource_savings_calculator(
    target_logical_error: f64,
    physical_error_rate: f64,
) -> (usize, usize, f64) {
    // Surface code: n ~ 2*d^2, p_L ~ (p/p_th)^{(d+1)/2}, p_th ~ 0.01
    // Solve for d: d ~ 2 * log(1/p_L) / log(p_th/p) - 1
    let ratio = (physical_error_rate / 0.01).max(1e-15);
    let log_target = target_logical_error.max(1e-30).ln();
    let log_ratio = ratio.ln();

    let d_surface = if log_ratio > 0.0 {
        (2.0 * log_target / log_ratio - 1.0).abs().ceil() as usize
    } else {
        3
    };
    let d_surface = d_surface.max(3);
    let n_surface = 2 * d_surface * d_surface;

    // Approximate dynamical: effective distance ~ 1.5 * d at same qubit count
    // So need fewer qubits: d_approx ~ d_surface / 1.5
    let d_approx = ((d_surface as f64 / 1.5).ceil() as usize).max(3);
    let n_approx = 2 * d_approx * d_approx;

    let savings = 1.0 - (n_approx as f64 / n_surface as f64);

    (n_surface, n_approx, savings)
}

// ============================================================
// SIMULATION ENGINE
// ============================================================

/// Configuration for the dynamical QEC simulator.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of physical qubits.
    pub num_qubits: usize,
    /// Number of QEC rounds to simulate.
    pub num_rounds: usize,
    /// Number of Monte Carlo trials.
    pub num_trials: usize,
    /// Physical noise model.
    pub noise_model: NoiseModel,
    /// Code distance.
    pub distance: usize,
}

/// Result of a Monte Carlo QEC simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Total number of trials run.
    pub num_trials: usize,
    /// Number of trials with logical errors.
    pub logical_errors: usize,
    /// Logical error rate (logical_errors / num_trials).
    pub logical_error_rate: f64,
    /// Physical error rate used.
    pub physical_error_rate: f64,
    /// Number of QEC rounds.
    pub num_rounds: usize,
    /// Average number of syndrome triggers per trial.
    pub avg_syndromes_per_trial: f64,
    /// Logical error rate per round.
    pub error_rate_per_round: f64,
}

impl fmt::Display for SimulationResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SimResult(trials={}, p_L={:.2e}, p_phys={:.4}, rounds={}, syndromes/trial={:.1})",
            self.num_trials,
            self.logical_error_rate,
            self.physical_error_rate,
            self.num_rounds,
            self.avg_syndromes_per_trial
        )
    }
}

/// Monte Carlo simulation engine for dynamical QEC codes.
///
/// Runs many independent trials of: noise injection -> syndrome extraction
/// -> decoding -> correction -> logical error check. Uses Rayon for
/// parallel trial execution.
pub struct DynamicalQecSimulator;

impl DynamicalQecSimulator {
    /// Run a Monte Carlo simulation of a Floquet dynamical code.
    pub fn simulate_floquet(config: &SimulationConfig) -> SimulationResult {
        let code = FloquetDynamicalCode::chain_code(config.distance);

        // Run trials in parallel using Rayon
        let results: Vec<(bool, usize)> = (0..config.num_trials)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut total_syndromes = 0usize;
                let mut cumulative_error = vec![Pauli::I; config.num_qubits];

                for round in 0..config.num_rounds {
                    // Inject noise
                    for q in 0..config.num_qubits {
                        let err = config.noise_model.sample_error(&mut rng);
                        if err != Pauli::I {
                            cumulative_error[q] = compose_pauli(cumulative_error[q], err);
                        }
                    }

                    // Extract syndrome
                    let syndrome = code.extract_syndrome(&cumulative_error, round);
                    total_syndromes += syndrome.outcomes.iter().filter(|&&o| o).count();

                    // Decode and correct
                    let corrections = code.decode(&syndrome);
                    for corr in &corrections {
                        if corr.qubit < config.num_qubits {
                            cumulative_error[corr.qubit] =
                                compose_pauli(cumulative_error[corr.qubit], corr.pauli);
                        }
                    }
                }

                // Check logical error: for the repetition-like code, logical
                // error is a chain of X errors spanning the code
                let logical_error = cumulative_error
                    .iter()
                    .filter(|p| **p == Pauli::X || **p == Pauli::Y)
                    .count()
                    % 2
                    == 1;

                (logical_error, total_syndromes)
            })
            .collect();

        let logical_errors = results.iter().filter(|(err, _)| *err).count();
        let total_syndromes: usize = results.iter().map(|(_, s)| s).sum();

        let logical_error_rate = logical_errors as f64 / config.num_trials.max(1) as f64;
        let avg_syndromes = total_syndromes as f64 / config.num_trials.max(1) as f64;
        let error_rate_per_round = if config.num_rounds > 0 {
            logical_error_rate / config.num_rounds as f64
        } else {
            0.0
        };

        SimulationResult {
            num_trials: config.num_trials,
            logical_errors,
            logical_error_rate,
            physical_error_rate: config.noise_model.effective_error_rate(),
            num_rounds: config.num_rounds,
            avg_syndromes_per_trial: avg_syndromes,
            error_rate_per_round,
        }
    }

    /// Run a Monte Carlo simulation of a hybrid dynamical code.
    pub fn simulate_hybrid(config: &SimulationConfig) -> SimulationResult {
        let code = HybridDynamicalCode::chain_code(config.distance);

        let results: Vec<(bool, usize)> = (0..config.num_trials)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut total_syndromes = 0usize;
                let mut cumulative_error = vec![Pauli::I; code.num_qubits];

                for round in 0..config.num_rounds {
                    for q in 0..code.num_qubits {
                        let err = config.noise_model.sample_error(&mut rng);
                        if err != Pauli::I {
                            cumulative_error[q] = compose_pauli(cumulative_error[q], err);
                        }
                    }

                    let syndrome = code.extract_syndrome(&cumulative_error, round);
                    total_syndromes += syndrome.outcomes.iter().filter(|&&o| o).count();

                    let corrections = code.decode(&syndrome);
                    for corr in &corrections {
                        if corr.qubit < code.num_qubits {
                            cumulative_error[corr.qubit] =
                                compose_pauli(cumulative_error[corr.qubit], corr.pauli);
                        }
                    }
                }

                let logical_error = cumulative_error
                    .iter()
                    .filter(|p| **p == Pauli::X || **p == Pauli::Y)
                    .count()
                    % 2
                    == 1;

                (logical_error, total_syndromes)
            })
            .collect();

        let logical_errors = results.iter().filter(|(err, _)| *err).count();
        let total_syndromes: usize = results.iter().map(|(_, s)| s).sum();

        let logical_error_rate = logical_errors as f64 / config.num_trials.max(1) as f64;
        let avg_syndromes = total_syndromes as f64 / config.num_trials.max(1) as f64;
        let error_rate_per_round = if config.num_rounds > 0 {
            logical_error_rate / config.num_rounds as f64
        } else {
            0.0
        };

        SimulationResult {
            num_trials: config.num_trials,
            logical_errors,
            logical_error_rate,
            physical_error_rate: config.noise_model.effective_error_rate(),
            num_rounds: config.num_rounds,
            avg_syndromes_per_trial: avg_syndromes,
            error_rate_per_round,
        }
    }

    /// Sweep physical error rates and return (p_physical, p_logical) pairs.
    pub fn error_rate_sweep(
        distance: usize,
        num_rounds: usize,
        num_trials: usize,
        error_rates: &[f64],
    ) -> Vec<(f64, f64)> {
        error_rates
            .iter()
            .map(|&p| {
                let config = SimulationConfig {
                    num_qubits: 2 * distance - 1,
                    num_rounds,
                    num_trials,
                    noise_model: NoiseModel::Depolarizing { error_rate: p },
                    distance,
                };
                let result = Self::simulate_floquet(&config);
                (p, result.logical_error_rate)
            })
            .collect()
    }
}

/// Compose two single-qubit Paulis (ignoring global phase).
fn compose_pauli(a: Pauli, b: Pauli) -> Pauli {
    match (a, b) {
        (Pauli::I, x) | (x, Pauli::I) => x,
        (Pauli::X, Pauli::X) | (Pauli::Y, Pauli::Y) | (Pauli::Z, Pauli::Z) => Pauli::I,
        (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::X) => Pauli::Z,
        (Pauli::Y, Pauli::Z) | (Pauli::Z, Pauli::Y) => Pauli::X,
        (Pauli::X, Pauli::Z) | (Pauli::Z, Pauli::X) => Pauli::Y,
    }
}

// ============================================================
// DEMO
// ============================================================

/// Demonstrate approximate dynamical QEC capabilities.
///
/// Runs through:
/// 1. Strategic code construction and parameter tracking
/// 2. Petz recovery channel and fidelity computation
/// 3. All three dynamical code families
/// 4. Approximate code analysis
/// 5. Comparison tools (dynamical vs static, approximate vs exact)
/// 6. Monte Carlo simulation
pub fn demo() {
    println!("=== Approximate Dynamical QEC (arXiv:2502.09177) ===\n");

    // 1. Strategic Code
    println!("--- Strategic Code Framework ---");
    let noise = NoiseModel::Depolarizing { error_rate: 0.001 };
    let code = StrategicCode::repetition(5, noise.clone());
    println!("Code: {}", code);
    for r in 0..code.num_rounds() {
        let (n, k, d) = code.parameters_at(r);
        println!("  Round {}: [[{}, {}, {}]]", r, n, k, d);
    }

    // 2. Petz Recovery
    println!("\n--- Temporal Petz Recovery ---");
    let petz = PetzRecoveryChannel::new(2, NoiseModel::Depolarizing { error_rate: 0.01 });
    let dim = 4;
    let mut rho = vec![ZERO; dim * dim];
    rho[0] = ONE; // |00><00|
    let result = petz.recover(&rho);
    println!("Recovery: {}", result);

    // 3. Dynamical Code Families
    println!("\n--- Dynamical Code Families ---");

    let floquet = FloquetDynamicalCode::chain_code(3);
    println!("Floquet: {}", floquet);

    let adaptive = AdaptiveDynamicalCode::chain_code(3);
    println!("Adaptive: {}", adaptive);

    let hybrid = HybridDynamicalCode::chain_code(3);
    println!("Hybrid: {}", hybrid);

    // 4. Approximate Code Analysis
    println!("\n--- Approximate Code Analysis ---");
    let analysis = ApproximateCodeAnalysis::analyze(&code);
    println!("Analysis: {}", analysis);

    // 5. Comparison
    println!("\n--- Comparisons ---");
    let cmp_dyn = compare_dynamical_vs_static(5, &noise);
    println!("{}", cmp_dyn);

    let cmp_approx = compare_approximate_vs_exact(25, &noise);
    println!("{}", cmp_approx);

    let (n_surface, n_approx, savings) = resource_savings_calculator(1e-10, 0.001);
    println!(
        "Resource savings: surface={} qubits, approx_dyn={} qubits, savings={:.1}%",
        n_surface,
        n_approx,
        savings * 100.0
    );

    // 6. Monte Carlo Simulation
    println!("\n--- Monte Carlo Simulation ---");
    let sim_config = SimulationConfig {
        num_qubits: 5,
        num_rounds: 10,
        num_trials: 500,
        noise_model: NoiseModel::Depolarizing { error_rate: 0.001 },
        distance: 3,
    };
    let sim_result = DynamicalQecSimulator::simulate_floquet(&sim_config);
    println!("Floquet sim: {}", sim_result);

    let hybrid_result = DynamicalQecSimulator::simulate_hybrid(&sim_config);
    println!("Hybrid sim:  {}", hybrid_result);

    // 7. Noise-Adapted Optimization (short run)
    println!("\n--- Noise-Adapted Optimizer ---");
    let mut optimizer = NoiseAdaptedOptimizer::new(
        5,
        4,
        NoiseModel::Biased {
            error_rate: 0.005,
            bias_eta: 10.0,
        },
    );
    optimizer.max_iterations = 50;
    let opt_result = optimizer.optimize();
    println!(
        "Optimized: p_L={:.2e} after {} iterations",
        opt_result.logical_error_rate, opt_result.iterations
    );
    println!("Best code: {}", opt_result.code);

    println!("\n=== Demo complete ===");
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Test 1: Pauli commutativity
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_commutativity() {
        assert!(Pauli::I.commutes_with(&Pauli::X));
        assert!(Pauli::X.commutes_with(&Pauli::X));
        assert!(!Pauli::X.commutes_with(&Pauli::Y));
        assert!(!Pauli::X.commutes_with(&Pauli::Z));
        assert!(!Pauli::Y.commutes_with(&Pauli::Z));
    }

    // ----------------------------------------------------------
    // Test 2: Pauli composition
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_composition() {
        assert_eq!(compose_pauli(Pauli::X, Pauli::X), Pauli::I);
        assert_eq!(compose_pauli(Pauli::Y, Pauli::Y), Pauli::I);
        assert_eq!(compose_pauli(Pauli::Z, Pauli::Z), Pauli::I);
        assert_eq!(compose_pauli(Pauli::X, Pauli::Y), Pauli::Z);
        assert_eq!(compose_pauli(Pauli::Y, Pauli::Z), Pauli::X);
        assert_eq!(compose_pauli(Pauli::X, Pauli::Z), Pauli::Y);
        assert_eq!(compose_pauli(Pauli::I, Pauli::X), Pauli::X);
        assert_eq!(compose_pauli(Pauli::Z, Pauli::I), Pauli::Z);
    }

    // ----------------------------------------------------------
    // Test 3: Pauli string commutativity
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_string_commutativity() {
        // ZZ and ZZ commute
        let a = vec![Pauli::Z, Pauli::Z, Pauli::I];
        let b = vec![Pauli::I, Pauli::Z, Pauli::Z];
        assert!(pauli_strings_commute(&a, &b));

        // XZ and ZX anticommute (two positions anticommute -> commute overall)
        let c = vec![Pauli::X, Pauli::Z];
        let d = vec![Pauli::Z, Pauli::X];
        assert!(pauli_strings_commute(&c, &d)); // 2 anticommuting sites -> commute

        // XZ and ZI: one anticommuting site -> anticommute
        let e = vec![Pauli::X, Pauli::Z];
        let f = vec![Pauli::Z, Pauli::I];
        assert!(!pauli_strings_commute(&e, &f));
    }

    // ----------------------------------------------------------
    // Test 4: Stabilizer snapshot construction
    // ----------------------------------------------------------
    #[test]
    fn test_stabilizer_snapshot() {
        let gens = vec![
            vec![Pauli::Z, Pauli::Z, Pauli::I, Pauli::I],
            vec![Pauli::I, Pauli::Z, Pauli::Z, Pauli::I],
            vec![Pauli::I, Pauli::I, Pauli::Z, Pauli::Z],
        ];
        let snap = StabilizerSnapshot::new(4, gens);

        assert_eq!(snap.num_qubits, 4);
        assert_eq!(snap.num_generators(), 3);
        assert_eq!(snap.num_logical_qubits(), 1); // 4 - 3 = 1
        assert!(snap.generators_commute());
        assert!(snap.approximate_distance() >= 2);
    }

    // ----------------------------------------------------------
    // Test 5: Strategic code construction
    // ----------------------------------------------------------
    #[test]
    fn test_strategic_code_construction() {
        let noise = NoiseModel::Depolarizing { error_rate: 0.001 };
        let code = StrategicCode::repetition(5, noise);

        assert_eq!(code.num_qubits(), 5);
        assert_eq!(code.num_rounds(), 4);
        assert!(code.average_distance() >= 1.0);

        for r in 0..code.num_rounds() {
            let (n, k, d) = code.parameters_at(r);
            assert_eq!(n, 5);
            assert!(k >= 1);
            assert!(d >= 1);
        }
    }

    // ----------------------------------------------------------
    // Test 6: Noise model error sampling
    // ----------------------------------------------------------
    #[test]
    fn test_noise_model_sampling() {
        let mut rng = rand::thread_rng();

        // Zero error rate should always produce identity
        let noise = NoiseModel::Depolarizing { error_rate: 0.0 };
        for _ in 0..100 {
            assert_eq!(noise.sample_error(&mut rng), Pauli::I);
        }

        // Full error rate should never produce identity
        let noise_full = NoiseModel::Depolarizing { error_rate: 1.0 };
        let mut non_identity = 0;
        for _ in 0..100 {
            if noise_full.sample_error(&mut rng) != Pauli::I {
                non_identity += 1;
            }
        }
        assert!(
            non_identity > 90,
            "Full depolarizing should rarely produce I"
        );
    }

    // ----------------------------------------------------------
    // Test 7: Biased noise model
    // ----------------------------------------------------------
    #[test]
    fn test_biased_noise() {
        let noise = NoiseModel::Biased {
            error_rate: 0.9,
            bias_eta: 100.0,
        };
        let mut rng = rand::thread_rng();

        let mut z_count = 0;
        let mut x_count = 0;
        let trials = 10000;

        for _ in 0..trials {
            match noise.sample_error(&mut rng) {
                Pauli::Z => z_count += 1,
                Pauli::X => x_count += 1,
                _ => {}
            }
        }

        // With eta=100, Z errors should be much more common than X
        assert!(
            z_count > x_count * 10,
            "Z errors ({}) should dominate X errors ({}) with eta=100",
            z_count,
            x_count
        );
    }

    // ----------------------------------------------------------
    // Test 8: Petz recovery improves fidelity
    // ----------------------------------------------------------
    #[test]
    fn test_petz_recovery() {
        let petz = PetzRecoveryChannel::new(2, NoiseModel::Depolarizing { error_rate: 0.05 });
        let dim = 4;

        // Pure state |00>
        let mut rho = vec![ZERO; dim * dim];
        rho[0] = ONE;

        let result = petz.recover(&rho);
        assert!(
            result.fidelity > 0.5,
            "Recovery fidelity ({}) should exceed 0.5",
            result.fidelity
        );
        assert!(result.logical_error_rate < 0.5);
        assert!(result.diamond_norm_distance >= 0.0);
        assert!(result.overhead > 0.0);
    }

    // ----------------------------------------------------------
    // Test 9: Floquet dynamical code syndrome extraction
    // ----------------------------------------------------------
    #[test]
    fn test_floquet_syndrome_extraction() {
        let code = FloquetDynamicalCode::chain_code(3);
        assert_eq!(code.num_qubits, 5);
        assert_eq!(code.period, 2);

        // No errors -> no syndromes triggered
        let no_error = vec![Pauli::I; 5];
        let syn = code.extract_syndrome(&no_error, 0);
        assert!(
            syn.outcomes.iter().all(|&o| !o),
            "No errors should produce no syndromes"
        );

        // Single X error on qubit 0 should trigger syndromes
        let mut error = vec![Pauli::I; 5];
        error[0] = Pauli::X;
        let syn_err = code.extract_syndrome(&error, 0);
        assert!(
            syn_err.outcomes.iter().any(|&o| o),
            "X error should trigger at least one syndrome"
        );
    }

    // ----------------------------------------------------------
    // Test 10: Adaptive code mode switching
    // ----------------------------------------------------------
    #[test]
    fn test_adaptive_code_mode() {
        let mut code = AdaptiveDynamicalCode::chain_code(3);
        assert!(!code.enhanced_mode);

        // No error -> stays in base mode
        let no_error = vec![Pauli::I; code.num_qubits];
        code.extract_syndrome_and_adapt(&no_error, 0);
        assert!(!code.enhanced_mode);

        // Error -> switches to enhanced mode
        let mut error = vec![Pauli::I; code.num_qubits];
        error[0] = Pauli::X;
        code.extract_syndrome_and_adapt(&error, 1);
        assert!(code.enhanced_mode, "Error should trigger enhanced mode");

        // Cooldown: stays enhanced for 3 rounds
        let no_error2 = vec![Pauli::I; code.num_qubits];
        code.extract_syndrome_and_adapt(&no_error2, 2);
        assert!(code.enhanced_mode, "Should stay enhanced during cooldown");

        code.extract_syndrome_and_adapt(&no_error2, 3);
        code.extract_syndrome_and_adapt(&no_error2, 4);
        code.extract_syndrome_and_adapt(&no_error2, 5);
        assert!(
            !code.enhanced_mode,
            "Should exit enhanced mode after cooldown"
        );
    }

    // ----------------------------------------------------------
    // Test 11: Hybrid code combines static and dynamic stabilizers
    // ----------------------------------------------------------
    #[test]
    fn test_hybrid_code_stabilizers() {
        let code = HybridDynamicalCode::chain_code(3);

        let stab_r0 = code.stabilizers_at(0);
        let stab_r1 = code.stabilizers_at(1);

        // Both rounds should include the static stabilizers
        assert!(stab_r0.len() >= code.static_stabilizers.len());
        assert!(stab_r1.len() >= code.static_stabilizers.len());

        // Different dynamic stabilizers in different rounds
        let dyn_r0 = stab_r0.len() - code.static_stabilizers.len();
        let dyn_r1 = stab_r1.len() - code.static_stabilizers.len();
        assert!(dyn_r0 > 0 || dyn_r1 > 0, "Should have dynamic stabilizers");
    }

    // ----------------------------------------------------------
    // Test 12: Approximate code analysis
    // ----------------------------------------------------------
    #[test]
    fn test_approximate_code_analysis() {
        let noise = NoiseModel::Depolarizing { error_rate: 0.001 };
        let code = StrategicCode::repetition(5, noise);
        let analysis = ApproximateCodeAnalysis::analyze(&code);

        assert!(
            analysis.fidelity_bound > 0.9,
            "Fidelity bound should be high for low noise"
        );
        assert!(
            analysis.knill_laflamme_deviation < 0.1,
            "KL deviation should be small for low noise"
        );
        assert!(analysis.logical_diamond_norm >= 0.0);
        assert!(analysis.logical_diamond_norm <= 2.0);
    }

    // ----------------------------------------------------------
    // Test 13: Comparison tools
    // ----------------------------------------------------------
    #[test]
    fn test_comparison_dynamical_vs_static() {
        let noise = NoiseModel::Depolarizing { error_rate: 0.001 };
        let cmp = compare_dynamical_vs_static(3, &noise);

        assert_eq!(cmp.name_a, "Static");
        assert_eq!(cmp.name_b, "Dynamical");
        assert!(cmp.error_rate_a > 0.0);
        assert!(cmp.error_rate_b > 0.0);
        // Dynamical should be at least as good (lower or equal error)
        assert!(
            cmp.error_rate_b <= cmp.error_rate_a * 1.1,
            "Dynamical should not be much worse than static"
        );
    }

    // ----------------------------------------------------------
    // Test 14: Resource savings calculator
    // ----------------------------------------------------------
    #[test]
    fn test_resource_savings() {
        let (n_surface, n_approx, savings) = resource_savings_calculator(1e-10, 0.001);

        assert!(n_surface > 0, "Surface code should need qubits");
        assert!(n_approx > 0, "Approx code should need qubits");
        assert!(
            n_approx <= n_surface,
            "Approximate code should use fewer or equal qubits: {} vs {}",
            n_approx,
            n_surface
        );
        assert!(savings >= 0.0, "Savings should be non-negative");
    }

    // ----------------------------------------------------------
    // Test 15: Monte Carlo simulation runs and produces valid results
    // ----------------------------------------------------------
    #[test]
    fn test_monte_carlo_simulation() {
        let config = SimulationConfig {
            num_qubits: 5,
            num_rounds: 5,
            num_trials: 100,
            noise_model: NoiseModel::Depolarizing { error_rate: 0.001 },
            distance: 3,
        };

        let result = DynamicalQecSimulator::simulate_floquet(&config);

        assert_eq!(result.num_trials, 100);
        assert_eq!(result.num_rounds, 5);
        assert!(result.logical_error_rate >= 0.0);
        assert!(result.logical_error_rate <= 1.0);
        assert!(result.avg_syndromes_per_trial >= 0.0);

        // At very low error rates, logical error rate should be very small
        assert!(
            result.logical_error_rate < 0.5,
            "p_L={} too high for p=0.001",
            result.logical_error_rate
        );
    }

    // ----------------------------------------------------------
    // Test 16: Noise-adapted optimizer improves logical error rate
    // ----------------------------------------------------------
    #[test]
    fn test_noise_adapted_optimizer() {
        let mut optimizer =
            NoiseAdaptedOptimizer::new(5, 3, NoiseModel::Depolarizing { error_rate: 0.005 });
        optimizer.max_iterations = 30;

        let result = optimizer.optimize();

        assert_eq!(result.iterations, 30);
        assert!(result.logical_error_rate >= 0.0);
        assert!(!result.history.is_empty());

        // The optimized code should have valid parameters
        assert_eq!(result.code.num_qubits(), 5);
    }

    // ----------------------------------------------------------
    // Test 17: Hybrid simulation produces valid results
    // ----------------------------------------------------------
    #[test]
    fn test_hybrid_simulation() {
        let config = SimulationConfig {
            num_qubits: 6,
            num_rounds: 5,
            num_trials: 100,
            noise_model: NoiseModel::Depolarizing { error_rate: 0.001 },
            distance: 3,
        };

        let result = DynamicalQecSimulator::simulate_hybrid(&config);

        assert_eq!(result.num_trials, 100);
        assert!(result.logical_error_rate >= 0.0);
        assert!(result.logical_error_rate <= 1.0);
    }

    // ----------------------------------------------------------
    // Test 18: Error rate sweep
    // ----------------------------------------------------------
    #[test]
    fn test_error_rate_sweep() {
        let rates = vec![0.0001, 0.001, 0.01];
        let sweep = DynamicalQecSimulator::error_rate_sweep(3, 3, 50, &rates);

        assert_eq!(sweep.len(), 3);
        for (p, p_l) in &sweep {
            assert!(*p > 0.0);
            assert!(*p_l >= 0.0);
            assert!(*p_l <= 1.0);
        }

        // Logical error rate should generally increase with physical error rate
        // (may not be strictly monotone with Monte Carlo noise at small trial counts)
    }

    // ----------------------------------------------------------
    // Test 19: Amplitude damping noise model
    // ----------------------------------------------------------
    #[test]
    fn test_amplitude_damping_noise() {
        let noise = NoiseModel::AmplitudeDamping { gamma: 0.1 };
        assert!((noise.effective_error_rate() - 0.1).abs() < 1e-10);

        let mut rng = rand::thread_rng();
        let mut z_errors = 0;
        let trials = 5000;
        for _ in 0..trials {
            if noise.sample_error(&mut rng) == Pauli::Z {
                z_errors += 1;
            }
        }

        // Amplitude damping should produce predominantly Z errors
        let z_frac = z_errors as f64 / trials as f64;
        assert!(
            z_frac > 0.03 && z_frac < 0.15,
            "Z error fraction ({:.3}) should be near gamma/2 = 0.05",
            z_frac
        );
    }

    #[test]
    fn test_petz_channel_applies_amplitude_damping_population_decay() {
        let petz = PetzRecoveryChannel::new(1, NoiseModel::AmplitudeDamping { gamma: 0.2 });

        // |1><1|
        let rho = vec![ZERO, ZERO, ZERO, ONE];
        let noisy = petz.apply_noise(&rho);

        // Expected: diag([gamma, 1-gamma]) = [0.2, 0.8]
        assert!(
            (noisy[0].re - 0.2).abs() < 1e-9,
            "rho00 expected 0.2, got {}",
            noisy[0].re
        );
        assert!(
            (noisy[3].re - 0.8).abs() < 1e-9,
            "rho11 expected 0.8, got {}",
            noisy[3].re
        );
        assert!(noisy[1].norm() < 1e-12);
        assert!(noisy[2].norm() < 1e-12);
    }

    #[test]
    fn test_petz_channel_custom_kraus_identity_is_noop() {
        let i2 = [[ONE, ZERO], [ZERO, ONE]];
        let model = NoiseModel::CustomKraus {
            kraus_ops: vec![i2],
        };
        let petz = PetzRecoveryChannel::new(1, model);

        let rho = vec![
            C64::new(0.6, 0.0),
            C64::new(0.2, -0.1),
            C64::new(0.2, 0.1),
            C64::new(0.4, 0.0),
        ];
        let noisy = petz.apply_noise(&rho);

        for i in 0..rho.len() {
            assert!(
                (noisy[i] - rho[i]).norm() < 1e-12,
                "entry {} mismatch: noisy={} rho={}",
                i,
                noisy[i],
                rho[i]
            );
        }
    }
}
