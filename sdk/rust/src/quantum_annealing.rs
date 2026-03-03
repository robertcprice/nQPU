//! Simulated Quantum Annealing (SQA)
//!
//! Models D-Wave-style adiabatic quantum computation via path-integral Monte
//! Carlo.  The system evolves under a time-dependent Hamiltonian
//!
//!   H(s) = (1 - s) H_driver + s H_problem
//!
//! where s varies from 0 to 1 according to an annealing schedule.  The driver
//! is a transverse-field Ising term and the problem Hamiltonian encodes the
//! optimization objective as an Ising model.
//!
//! Three simulation backends are provided:
//!
//! - **SQA** (`anneal_quantum`): Path-integral Monte Carlo with Trotter-Suzuki
//!   decomposition -- replicates the spin configuration across P imaginary-time
//!   slices coupled by a transverse-field-derived inter-slice interaction.
//! - **SA** (`anneal_classical`): Standard simulated annealing baseline.
//! - **Exact** (`anneal_statevector`): Full statevector time evolution via
//!   Trotterised unitary steps -- limited to small systems (<=20 qubits).
//!
//! A library of standard combinatorial optimisation problems is included
//! (`max_cut`, `number_partitioning`, `random_ising`, `sk_model`).
//!
//! # References
//! - Kadowaki & Nishimori (1998) - Quantum annealing in the transverse Ising model
//! - Santoro et al. (2002) - Theory of quantum annealing of an Ising spin glass
//! - Farhi et al. (2001) - A quantum adiabatic evolution algorithm
//! - Morita & Nishimori (2008) - Mathematical foundation of quantum annealing

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors arising from quantum annealing computations.
#[derive(Debug, Clone)]
pub enum AnnealingError {
    /// Problem specification is invalid.
    InvalidProblem(String),
    /// Parameters are out of valid range.
    InvalidParameters(String),
    /// Statevector simulation exceeds qubit limit.
    TooManyQubits { requested: usize, max: usize },
    /// Numerical issue (NaN, Inf, negative norm, etc.).
    NumericalInstability(String),
}

impl std::fmt::Display for AnnealingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProblem(msg) => write!(f, "Invalid problem: {}", msg),
            Self::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            Self::TooManyQubits { requested, max } => {
                write!(f, "Too many qubits: {} requested, max {}", requested, max)
            }
            Self::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
        }
    }
}

impl std::error::Error for AnnealingError {}

pub type AnnealingResult<T> = Result<T, AnnealingError>;

// ===================================================================
// ISING MODEL
// ===================================================================

/// Classical Ising model: H = -sum_i h_i s_i - sum_{i<j} J_{ij} s_i s_j
///
/// Spins are +1 / -1.  In the `energy` method a `bool` configuration is
/// mapped as `true -> +1`, `false -> -1`.
#[derive(Clone, Debug)]
pub struct IsingModel {
    /// Number of spin variables.
    pub n_spins: usize,
    /// Local fields h_i.
    pub h: Vec<f64>,
    /// Couplings (i, j, J_ij).  Each pair should appear only once with i < j.
    pub j: Vec<(usize, usize, f64)>,
}

impl IsingModel {
    /// Construct a new Ising model.  Validates that indices are within bounds
    /// and that h has the correct length.
    pub fn new(n_spins: usize, h: Vec<f64>, j: Vec<(usize, usize, f64)>) -> AnnealingResult<Self> {
        if h.len() != n_spins {
            return Err(AnnealingError::InvalidProblem(format!(
                "h length {} != n_spins {}",
                h.len(),
                n_spins
            )));
        }
        for &(a, b, _) in &j {
            if a >= n_spins || b >= n_spins {
                return Err(AnnealingError::InvalidProblem(format!(
                    "coupling index ({}, {}) out of range for {} spins",
                    a, b, n_spins
                )));
            }
        }
        Ok(Self { n_spins, h, j })
    }

    /// Convert a QUBO (Quadratic Unconstrained Binary Optimisation) matrix to
    /// Ising form.  QUBO variables x_i in {0, 1} are mapped via x_i = (s_i+1)/2.
    ///
    /// The QUBO objective is  sum_{(i,j)} Q_{ij} x_i x_j  (with diagonal terms
    /// Q_{ii} x_i included as (i, i, Q_{ii})).
    pub fn from_qubo(n: usize, q: &[(usize, usize, f64)]) -> Self {
        // x_i = (s_i + 1) / 2
        // Q_{ii} x_i = Q_{ii} (s_i + 1) / 2 = Q_{ii}/2 * s_i + Q_{ii}/2
        // Q_{ij} x_i x_j = Q_{ij}/4 * (s_i s_j + s_i + s_j + 1)  [i != j]
        let mut h_vec = vec![0.0; n];
        let mut j_map: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        let mut _offset = 0.0; // constant energy offset (not tracked in model)

        for &(i, j, qij) in q {
            if i == j {
                // Diagonal: linear term
                h_vec[i] += qij / 2.0;
                _offset += qij / 2.0;
            } else {
                // Off-diagonal: coupling + linear contributions
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                *j_map.entry((lo, hi)).or_insert(0.0) += qij / 4.0;
                h_vec[i] += qij / 4.0;
                h_vec[j] += qij / 4.0;
                _offset += qij / 4.0;
            }
        }

        let j_vec: Vec<(usize, usize, f64)> = j_map
            .into_iter()
            .map(|((a, b), v)| (a, b, v))
            .collect();

        Self {
            n_spins: n,
            h: h_vec,
            j: j_vec,
        }
    }

    /// Compute the classical Ising energy for a spin configuration.
    ///
    /// `config[i] == true` corresponds to s_i = +1, `false` to s_i = -1.
    ///
    /// H = -sum_i h_i s_i - sum_{i<j} J_{ij} s_i s_j
    pub fn energy(&self, config: &[bool]) -> f64 {
        assert_eq!(config.len(), self.n_spins);
        let spin = |b: bool| -> f64 {
            if b {
                1.0
            } else {
                -1.0
            }
        };

        let mut e = 0.0;
        // Local field contribution
        for i in 0..self.n_spins {
            e -= self.h[i] * spin(config[i]);
        }
        // Coupling contribution
        for &(a, b, jab) in &self.j {
            e -= jab * spin(config[a]) * spin(config[b]);
        }
        e
    }

    /// Enumerate all 2^n configurations and return the ground state energy
    /// and configuration.  Only feasible for small n (<=20 or so).
    pub fn brute_force_ground_state(&self) -> (f64, Vec<bool>) {
        assert!(self.n_spins <= 24, "brute force limited to <=24 spins");
        let total = 1u64 << self.n_spins;
        let mut best_energy = f64::INFINITY;
        let mut best_config = vec![false; self.n_spins];

        for bits in 0..total {
            let config: Vec<bool> = (0..self.n_spins).map(|i| (bits >> i) & 1 == 1).collect();
            let e = self.energy(&config);
            if e < best_energy {
                best_energy = e;
                best_config = config;
            }
        }
        (best_energy, best_config)
    }
}

// ===================================================================
// ANNEALING SCHEDULE
// ===================================================================

/// Annealing schedule controlling s(t) in [0, 1].
///
/// s determines the interpolation between driver and problem Hamiltonian.
#[derive(Clone, Debug)]
pub enum QuantumAnnealingSchedule {
    /// Linear ramp: s(t) = t / T.
    Linear,
    /// Quadratic ramp: s(t) = (t / T)^2.  Slower start, faster finish.
    Quadratic,
    /// Custom schedule defined by interpolation points (normalised_time, s).
    /// Points must be sorted by time with time in [0, 1] and s in [0, 1].
    Custom(Vec<(f64, f64)>),
    /// Pause-and-quench: ramp to `s_pause`, hold from `start` to `end`,
    /// then ramp to 1.0.  Times are normalised fractions of total anneal.
    Pause {
        start: f64,
        end: f64,
        s_pause: f64,
    },
}

impl QuantumAnnealingSchedule {
    /// Evaluate the schedule at normalised time `t_norm` in [0, 1].
    pub fn evaluate(&self, t_norm: f64) -> f64 {
        let t = t_norm.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::Quadratic => t * t,
            Self::Custom(points) => {
                if points.is_empty() {
                    return t;
                }
                if t <= points[0].0 {
                    return points[0].1;
                }
                if t >= points[points.len() - 1].0 {
                    return points[points.len() - 1].1;
                }
                // Linear interpolation between bounding points.
                for w in points.windows(2) {
                    if t >= w[0].0 && t <= w[1].0 {
                        let frac = (t - w[0].0) / (w[1].0 - w[0].0);
                        return w[0].1 + frac * (w[1].1 - w[0].1);
                    }
                }
                t // fallback
            }
            Self::Pause { start, end, s_pause } => {
                if t < *start {
                    // Ramp from 0 to s_pause
                    t / start * s_pause
                } else if t <= *end {
                    // Hold at s_pause
                    *s_pause
                } else {
                    // Ramp from s_pause to 1.0
                    let remaining = 1.0 - end;
                    if remaining <= 0.0 {
                        1.0
                    } else {
                        s_pause + (1.0 - s_pause) * (t - end) / remaining
                    }
                }
            }
        }
    }
}

// ===================================================================
// ANNEALING CONFIGURATION
// ===================================================================

/// Configuration for a quantum annealing run.
#[derive(Clone, Debug)]
pub struct QuantumAnnealerConfig {
    /// Annealing schedule.
    pub schedule: QuantumAnnealingSchedule,
    /// Number of discrete time steps in the anneal.
    pub n_steps: usize,
    /// Number of Trotter slices for SQA path-integral Monte Carlo.
    pub n_trotter: usize,
    /// Inverse temperature beta = 1 / (k_B T).
    pub beta: f64,
    /// Transverse field strength at s = 0 (driver scale).
    pub gamma_initial: f64,
    /// Number of independent runs for success probability estimation.
    pub n_runs: usize,
    /// Number of Metropolis sweeps per time step.
    pub n_sweeps_per_step: usize,
}

impl Default for QuantumAnnealerConfig {
    fn default() -> Self {
        Self {
            schedule: QuantumAnnealingSchedule::Linear,
            n_steps: 1000,
            n_trotter: 32,
            beta: 10.0,
            gamma_initial: 3.0,
            n_runs: 10,
            n_sweeps_per_step: 1,
        }
    }
}

// ===================================================================
// QUANTUM ANNEALER
// ===================================================================

/// Simulated quantum annealer.
///
/// Supports three simulation modes:
/// - `anneal_quantum`: SQA via path-integral Monte Carlo
/// - `anneal_classical`: Classical simulated annealing baseline
/// - `anneal_statevector`: Exact unitary time evolution (small systems)
pub struct QuantumAnnealer {
    /// The Ising problem to solve.
    pub problem: IsingModel,
    /// Annealing configuration.
    pub config: QuantumAnnealerConfig,
}

impl QuantumAnnealer {
    /// Create a new quantum annealer.
    pub fn new(problem: IsingModel, config: QuantumAnnealerConfig) -> Self {
        Self { problem, config }
    }

    // ---------------------------------------------------------------
    // SQA: Path-Integral Monte Carlo
    // ---------------------------------------------------------------

    /// Run simulated quantum annealing via path-integral Monte Carlo.
    ///
    /// The classical spin system is replicated across `n_trotter` imaginary-time
    /// slices.  Within each slice the spins interact via the problem Hamiltonian.
    /// Between adjacent slices a ferromagnetic coupling (derived from the
    /// transverse field strength via the Suzuki-Trotter mapping) encourages
    /// alignment.
    ///
    /// Returns the best configuration found across all runs.
    pub fn anneal_quantum(&self) -> SqaResult {
        let n = self.problem.n_spins;
        let p = self.config.n_trotter;
        let beta = self.config.beta;
        let n_steps = self.config.n_steps;
        let n_runs = self.config.n_runs;

        let mut rng = rand::thread_rng();
        let mut best_energy = f64::INFINITY;
        let mut best_config = vec![false; n];
        let mut energy_history = vec![0.0f64; n_steps];
        let mut ground_hits = 0usize;

        // Brute-force ground state for small problems to measure success rate.
        let ground_energy = if n <= 20 {
            Some(self.problem.brute_force_ground_state().0)
        } else {
            None
        };

        for _run in 0..n_runs {
            // Initialise P Trotter slices with random configurations.
            let mut slices: Vec<Vec<bool>> = (0..p)
                .map(|_| (0..n).map(|_| rng.gen::<bool>()).collect())
                .collect();

            let mut run_best_energy = f64::INFINITY;
            let mut run_best_config = vec![false; n];

            for step in 0..n_steps {
                let t_norm = step as f64 / n_steps as f64;
                let s = self.config.schedule.evaluate(t_norm);

                // Transverse field strength at this point.
                let gamma = self.config.gamma_initial * (1.0 - s);

                // Inter-slice coupling from Suzuki-Trotter mapping.
                // J_perp = -(P * T) / 2 * ln(tanh(gamma / (P * T)))
                // where T = 1/beta, so P*T = P/beta.
                let pt = p as f64 / beta;
                let arg = gamma / pt;
                let j_perp = if arg > 1e-10 {
                    -pt / 2.0 * arg.tanh().ln()
                } else {
                    // For very small gamma the coupling diverges; clamp it.
                    // In this regime the transverse field is negligible.
                    20.0 * beta
                };

                // Metropolis sweeps
                for _sweep in 0..self.config.n_sweeps_per_step {
                    for slice_idx in 0..p {
                        for spin_idx in 0..n {
                            // Energy change from flipping spin_idx in this slice.
                            let delta_e = self.flip_energy_change(
                                &slices,
                                slice_idx,
                                spin_idx,
                                s,
                                j_perp,
                                p,
                            );

                            // Metropolis acceptance at inverse temperature beta/P.
                            let accept = if delta_e <= 0.0 {
                                true
                            } else {
                                let prob = (-beta / p as f64 * delta_e).exp();
                                rng.gen::<f64>() < prob
                            };

                            if accept {
                                slices[slice_idx][spin_idx] = !slices[slice_idx][spin_idx];
                            }
                        }
                    }
                }

                // Track best energy across all slices.
                for slice in &slices {
                    let e = self.problem.energy(slice);
                    if e < run_best_energy {
                        run_best_energy = e;
                        run_best_config = slice.clone();
                    }
                }

                // Record energy history (average over runs, updated incrementally).
                energy_history[step] += run_best_energy / n_runs as f64;
            }

            if run_best_energy < best_energy {
                best_energy = run_best_energy;
                best_config = run_best_config;
            }

            if let Some(ge) = ground_energy {
                if (run_best_energy - ge).abs() < 1e-8 {
                    ground_hits += 1;
                }
            }
        }

        let success_probability = if ground_energy.is_some() {
            ground_hits as f64 / n_runs as f64
        } else {
            0.0
        };

        // Time to solution: TTS = t_anneal * ln(1 - 0.99) / ln(1 - p_success)
        let time_to_solution = if success_probability > 0.0 && success_probability < 1.0 {
            let t_anneal = n_steps as f64;
            t_anneal * (1.0 - 0.99_f64).ln() / (1.0 - success_probability).ln()
        } else if success_probability >= 1.0 {
            n_steps as f64
        } else {
            f64::INFINITY
        };

        SqaResult {
            best_config,
            best_energy,
            energy_history,
            success_probability,
            time_to_solution,
        }
    }

    /// Compute the energy change from flipping `spin_idx` in `slice_idx`.
    ///
    /// Accounts for both intra-slice (problem Hamiltonian) and inter-slice
    /// (transverse field / Trotter coupling) contributions.
    fn flip_energy_change(
        &self,
        slices: &[Vec<bool>],
        slice_idx: usize,
        spin_idx: usize,
        s: f64,
        j_perp: f64,
        p: usize,
    ) -> f64 {
        let spin_val = |b: bool| -> f64 {
            if b {
                1.0
            } else {
                -1.0
            }
        };

        let current_spin = spin_val(slices[slice_idx][spin_idx]);

        // Intra-slice contribution (problem Hamiltonian, scaled by s).
        let mut delta_intra = 0.0;

        // Local field: flipping s_i changes -h_i s_i by +2 h_i s_i.
        delta_intra += 2.0 * self.problem.h[spin_idx] * current_spin;

        // Couplings: flipping s_i changes -J_{ij} s_i s_j by +2 J_{ij} s_i s_j.
        for &(a, b, jab) in &self.problem.j {
            if a == spin_idx {
                delta_intra += 2.0 * jab * current_spin * spin_val(slices[slice_idx][b]);
            } else if b == spin_idx {
                delta_intra += 2.0 * jab * spin_val(slices[slice_idx][a]) * current_spin;
            }
        }

        // Scale intra-slice by the problem fraction s.
        delta_intra *= s;

        // Inter-slice contribution (Trotter coupling).
        // Coupling between this slice and its neighbours (periodic boundary).
        let prev = if slice_idx == 0 { p - 1 } else { slice_idx - 1 };
        let next = if slice_idx == p - 1 { 0 } else { slice_idx + 1 };

        let neighbour_sum =
            spin_val(slices[prev][spin_idx]) + spin_val(slices[next][spin_idx]);

        // Flipping changes -J_perp * s_i * (s_{i,prev} + s_{i,next})
        // by +2 * J_perp * current_spin * neighbour_sum.
        let delta_inter = 2.0 * j_perp * current_spin * neighbour_sum;

        delta_intra + delta_inter
    }

    // ---------------------------------------------------------------
    // Classical Simulated Annealing
    // ---------------------------------------------------------------

    /// Run classical simulated annealing (SA) as a baseline comparison.
    ///
    /// Uses Metropolis updates with an exponential temperature schedule derived
    /// from the annealing schedule parameter s.
    pub fn anneal_classical(&self) -> SqaResult {
        let n = self.problem.n_spins;
        let n_steps = self.config.n_steps;
        let n_runs = self.config.n_runs;

        let mut rng = rand::thread_rng();
        let mut best_energy = f64::INFINITY;
        let mut best_config = vec![false; n];
        let mut energy_history = vec![0.0f64; n_steps];
        let mut ground_hits = 0usize;

        let ground_energy = if n <= 20 {
            Some(self.problem.brute_force_ground_state().0)
        } else {
            None
        };

        let t_initial: f64 = 5.0;
        let t_final: f64 = 0.01;

        for _run in 0..n_runs {
            let mut config: Vec<bool> = (0..n).map(|_| rng.gen::<bool>()).collect();
            let mut current_energy = self.problem.energy(&config);
            let mut run_best_energy = current_energy;
            let mut run_best_config = config.clone();

            for step in 0..n_steps {
                let t_norm = step as f64 / n_steps as f64;
                // Exponential temperature schedule.
                let temperature: f64 = t_initial * (t_final / t_initial).powf(t_norm);

                // Flip a random spin.
                let flip_idx = rng.gen_range(0..n);

                // Compute energy change without recomputing full energy.
                let spin_val = |b: bool| -> f64 {
                    if b {
                        1.0
                    } else {
                        -1.0
                    }
                };
                let current_spin = spin_val(config[flip_idx]);
                let mut delta_e = 2.0 * self.problem.h[flip_idx] * current_spin;
                for &(a, b, jab) in &self.problem.j {
                    if a == flip_idx {
                        delta_e += 2.0 * jab * current_spin * spin_val(config[b]);
                    } else if b == flip_idx {
                        delta_e += 2.0 * jab * spin_val(config[a]) * current_spin;
                    }
                }

                let accept = if delta_e <= 0.0 {
                    true
                } else {
                    rng.gen::<f64>() < (-delta_e / temperature).exp()
                };

                if accept {
                    config[flip_idx] = !config[flip_idx];
                    current_energy += delta_e;
                }

                if current_energy < run_best_energy {
                    run_best_energy = current_energy;
                    run_best_config = config.clone();
                }

                energy_history[step] += run_best_energy / n_runs as f64;
            }

            if run_best_energy < best_energy {
                best_energy = run_best_energy;
                best_config = run_best_config;
            }

            if let Some(ge) = ground_energy {
                if (run_best_energy - ge).abs() < 1e-8 {
                    ground_hits += 1;
                }
            }
        }

        let success_probability = if ground_energy.is_some() {
            ground_hits as f64 / n_runs as f64
        } else {
            0.0
        };

        let time_to_solution = if success_probability > 0.0 && success_probability < 1.0 {
            let t_anneal = n_steps as f64;
            t_anneal * (1.0 - 0.99_f64).ln() / (1.0 - success_probability).ln()
        } else if success_probability >= 1.0 {
            n_steps as f64
        } else {
            f64::INFINITY
        };

        SqaResult {
            best_config,
            best_energy,
            energy_history,
            success_probability,
            time_to_solution,
        }
    }

    // ---------------------------------------------------------------
    // Exact Statevector Evolution
    // ---------------------------------------------------------------

    /// Exact time evolution of the annealing Hamiltonian on a statevector.
    ///
    /// Builds H(s) as dense matrices and performs Trotterised unitary steps.
    /// Limited to `n_qubits` <= 20 due to exponential memory.
    pub fn anneal_statevector(&self, n_qubits: usize) -> AnnealingResult<SqaResult> {
        if n_qubits > 20 {
            return Err(AnnealingError::TooManyQubits {
                requested: n_qubits,
                max: 20,
            });
        }
        if n_qubits != self.problem.n_spins {
            return Err(AnnealingError::InvalidParameters(format!(
                "n_qubits {} != problem n_spins {}",
                n_qubits, self.problem.n_spins
            )));
        }

        let dim = 1usize << n_qubits;
        let n_steps = self.config.n_steps;
        let dt = 1.0 / n_steps as f64; // normalised time step

        // Start in uniform superposition (ground state of transverse field).
        let amp = 1.0 / (dim as f64).sqrt();
        let mut state: Vec<Complex64> = vec![Complex64::new(amp, 0.0); dim];

        let mut energy_history = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let t_norm = (step as f64 + 0.5) / n_steps as f64;
            let s = self.config.schedule.evaluate(t_norm);

            // Build diagonal of problem Hamiltonian (Ising is diagonal in Z basis).
            let h_problem_diag = self.build_problem_diagonal(n_qubits);

            // Apply exp(-i * s * H_problem * dt) (diagonal in computational basis).
            for k in 0..dim {
                let phase = -s * h_problem_diag[k] * dt;
                let rot = Complex64::new(phase.cos(), phase.sin());
                state[k] *= rot;
            }

            // Apply exp(-i * (1-s) * H_driver * dt).
            // H_driver = -gamma * sum_i X_i.
            // For each qubit, X_i acts as a rotation in the {|0>, |1>} subspace.
            let gamma = self.config.gamma_initial;
            let theta = (1.0 - s) * gamma * dt;
            for q in 0..n_qubits {
                self.apply_rx(&mut state, q, 2.0 * theta);
            }

            // Measure expectation of problem Hamiltonian for energy history.
            let e = self.statevector_energy(&state, &h_problem_diag);
            energy_history.push(e);
        }

        // Find the most probable configuration.
        let probs: Vec<f64> = state.iter().map(|a| a.norm_sqr()).collect();
        let (best_idx, _) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let best_config: Vec<bool> = (0..n_qubits).map(|i| (best_idx >> i) & 1 == 1).collect();
        let best_energy = self.problem.energy(&best_config);

        // Success probability = probability of being in the ground state.
        let ground_energy = self.problem.brute_force_ground_state().0;
        let success_probability: f64 = probs
            .iter()
            .enumerate()
            .filter(|&(idx, _)| {
                let cfg: Vec<bool> = (0..n_qubits).map(|i| (idx >> i) & 1 == 1).collect();
                (self.problem.energy(&cfg) - ground_energy).abs() < 1e-8
            })
            .map(|(_, p)| p)
            .sum();

        // Verify norm preservation.
        let norm: f64 = probs.iter().sum();
        if (norm - 1.0).abs() > 1e-6 {
            return Err(AnnealingError::NumericalInstability(format!(
                "Statevector norm {} deviates from 1.0",
                norm
            )));
        }

        let time_to_solution = if success_probability > 0.0 && success_probability < 1.0 {
            let t_anneal = n_steps as f64;
            t_anneal * (1.0 - 0.99_f64).ln() / (1.0 - success_probability).ln()
        } else if success_probability >= 1.0 - 1e-10 {
            n_steps as f64
        } else {
            f64::INFINITY
        };

        Ok(SqaResult {
            best_config,
            best_energy,
            energy_history,
            success_probability,
            time_to_solution,
        })
    }

    /// Build the diagonal of the problem Hamiltonian in the computational basis.
    ///
    /// For each basis state |z>, H_problem |z> = E(z) |z> where E(z) is the
    /// classical Ising energy.
    fn build_problem_diagonal(&self, n_qubits: usize) -> Vec<f64> {
        let dim = 1usize << n_qubits;
        let mut diag = vec![0.0; dim];
        for k in 0..dim {
            let config: Vec<bool> = (0..n_qubits).map(|i| (k >> i) & 1 == 1).collect();
            diag[k] = self.problem.energy(&config);
        }
        diag
    }

    /// Apply Rx(theta) gate on qubit `q` of the statevector.
    fn apply_rx(&self, state: &mut [Complex64], q: usize, theta: f64) {
        let dim = state.len();
        let s = 1usize << q;
        let c = (theta / 2.0).cos();
        let sn = (theta / 2.0).sin();

        let mut i = 0;
        while i < dim {
            for k in 0..s {
                let idx0 = i + k;
                let idx1 = i + k + s;
                let a = state[idx0];
                let b = state[idx1];
                // Rx(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
                state[idx0] = Complex64::new(
                    a.re * c + b.im * sn,
                    a.im * c - b.re * sn,
                );
                state[idx1] = Complex64::new(
                    b.re * c + a.im * sn,
                    b.im * c - a.re * sn,
                );
            }
            i += s << 1;
        }
    }

    /// Compute <psi| H_problem |psi> using the diagonal representation.
    fn statevector_energy(&self, state: &[Complex64], diag: &[f64]) -> f64 {
        state
            .iter()
            .zip(diag.iter())
            .map(|(a, &e)| a.norm_sqr() * e)
            .sum()
    }
}

// ===================================================================
// RESULT STRUCT
// ===================================================================

/// Result of a quantum annealing simulation.
#[derive(Clone, Debug)]
pub struct SqaResult {
    /// Best spin configuration found (true = +1, false = -1).
    pub best_config: Vec<bool>,
    /// Energy of the best configuration.
    pub best_energy: f64,
    /// Best energy found at each time step (averaged over runs where applicable).
    pub energy_history: Vec<f64>,
    /// Fraction of runs that found the exact ground state (0 if unknown).
    pub success_probability: f64,
    /// Estimated wall-clock steps to reach 99% success probability.
    pub time_to_solution: f64,
}

// ===================================================================
// PROBLEM LIBRARY
// ===================================================================

/// Standard combinatorial optimisation problems encoded as Ising models.
pub mod problems {
    use super::*;

    /// Max-Cut on an unweighted graph.
    ///
    /// Given edges, find the partition of vertices that maximises the number
    /// of edges crossing the partition.  The Ising encoding is:
    ///   H = -sum_{(i,j) in edges} (1 - s_i s_j) / 2
    /// which is minimised when adjacent spins differ.  We negate the couplings
    /// so that minimising H corresponds to maximising the cut.
    pub fn max_cut(n_vertices: usize, edges: &[(usize, usize)]) -> IsingModel {
        let h = vec![0.0; n_vertices];
        let j: Vec<(usize, usize, f64)> = edges
            .iter()
            .map(|&(a, b)| {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                // Minimise H = -sum J_ij s_i s_j with J_ij = -1/2
                // so that anti-aligned spins are preferred.
                (lo, hi, -0.5)
            })
            .collect();

        IsingModel {
            n_spins: n_vertices,
            h,
            j,
        }
    }

    /// Number partitioning: split `numbers` into two subsets with equal sum.
    ///
    /// Ising encoding: minimise H = (sum_i n_i s_i)^2.
    /// Expanding: H = sum_i n_i^2 + sum_{i<j} 2 n_i n_j s_i s_j.
    /// The constant term does not affect the minimum, so we set:
    ///   J_{ij} = -2 n_i n_j  (negative because H = -sum J s_i s_j,
    ///                          and we want to minimise the square).
    ///
    /// Wait -- to be precise: we want to minimise (sum n_i s_i)^2,
    /// which equals sum_i n_i^2 + 2 sum_{i<j} n_i n_j s_i s_j.
    /// Since our Ising energy is E = -sum h_i s_i - sum J_{ij} s_i s_j,
    /// we need -J_{ij} = 2 n_i n_j, i.e. J_{ij} = -2 n_i n_j.
    pub fn number_partitioning(numbers: &[f64]) -> IsingModel {
        let n = numbers.len();
        let h = vec![0.0; n];
        let mut j = Vec::new();
        for i in 0..n {
            for jj in (i + 1)..n {
                // We want to minimise sum_pairs 2*n_i*n_j*s_i*s_j.
                // Our energy is -sum J s_i s_j, so J = -2*n_i*n_j.
                j.push((i, jj, -2.0 * numbers[i] * numbers[jj]));
            }
        }
        IsingModel { n_spins: n, h, j }
    }

    /// Random sparse Ising model.
    ///
    /// Each coupling is present with probability `density` and drawn from
    /// a standard normal distribution.  Local fields are also standard normal.
    pub fn random_ising(n: usize, density: f64) -> IsingModel {
        let mut rng = rand::thread_rng();
        let h: Vec<f64> = (0..n).map(|_| random_normal(&mut rng)).collect();
        let mut j = Vec::new();
        for i in 0..n {
            for jj in (i + 1)..n {
                if rng.gen::<f64>() < density {
                    j.push((i, jj, random_normal(&mut rng)));
                }
            }
        }
        IsingModel { n_spins: n, h, j }
    }

    /// Sherrington-Kirkpatrick model: fully connected random Ising.
    ///
    /// All-to-all couplings drawn from N(0, 1/sqrt(N)).  No local fields.
    pub fn sk_model(n: usize) -> IsingModel {
        let mut rng = rand::thread_rng();
        let h = vec![0.0; n];
        let scale = 1.0 / (n as f64).sqrt();
        let mut j = Vec::new();
        for i in 0..n {
            for jj in (i + 1)..n {
                j.push((i, jj, random_normal(&mut rng) * scale));
            }
        }
        IsingModel { n_spins: n, h, j }
    }

    /// Box-Muller transform for standard normal random variable.
    fn random_normal(rng: &mut impl Rng) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-15);
        let u2: f64 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::problems::*;

    // ---------------------------------------------------------------
    // 1. QUBO to Ising conversion
    // ---------------------------------------------------------------
    #[test]
    fn test_qubo_to_ising_conversion() {
        // Simple 2-variable QUBO: min x0 + 2*x0*x1
        // Q = [(0,0,1.0), (0,1,2.0)]
        let qubo = vec![(0, 0, 1.0), (0, 1, 2.0)];
        let ising = IsingModel::from_qubo(2, &qubo);

        assert_eq!(ising.n_spins, 2);
        assert_eq!(ising.h.len(), 2);

        // Verify: for x0=0,x1=0 (s0=-1,s1=-1) QUBO value = 0
        // for x0=1,x1=1 (s0=+1,s1=+1) QUBO value = 1 + 2 = 3
        // The Ising energies should differ by 3 (up to offset).
        let e00 = ising.energy(&[false, false]);
        let e11 = ising.energy(&[true, true]);
        let e10 = ising.energy(&[true, false]);
        let e01 = ising.energy(&[false, true]);

        // QUBO values: x=(0,0)->0, x=(1,0)->1, x=(0,1)->0, x=(1,1)->3
        // Ising energies should preserve the ordering (shifted by constant).
        // e00 and e01 should be equal (both map to QUBO 0).
        // e10 should correspond to QUBO 1.
        // e11 should correspond to QUBO 3.
        // Differences should be consistent.
        let diff_11_00 = e11 - e00;
        let diff_10_00 = e10 - e00;

        // The ratio of QUBO differences is (3-0)/(1-0) = 3.
        // Ising encoding preserves this ratio.
        assert!(
            (diff_11_00 / diff_10_00 - 3.0).abs() < 1e-10,
            "QUBO->Ising ratio mismatch: {}/{}",
            diff_11_00,
            diff_10_00
        );
    }

    // ---------------------------------------------------------------
    // 2. Energy calculation correctness
    // ---------------------------------------------------------------
    #[test]
    fn test_energy_calculation() {
        // 3-spin system: H = -h0 s0 - h1 s1 - h2 s2 - J01 s0 s1
        // h = [1.0, -0.5, 0.0], J = [(0,1, 2.0)]
        let model = IsingModel::new(3, vec![1.0, -0.5, 0.0], vec![(0, 1, 2.0)]).unwrap();

        // Config: [true, true, false] -> s = [+1, +1, -1]
        // E = -1*1 - (-0.5)*1 - 0*(-1) - 2*1*1 = -1 + 0.5 - 2 = -2.5
        let e = model.energy(&[true, true, false]);
        assert!(
            (e - (-2.5)).abs() < 1e-10,
            "Energy mismatch: expected -2.5, got {}",
            e
        );

        // Config: [false, true, true] -> s = [-1, +1, +1]
        // E = -1*(-1) - (-0.5)*1 - 0*1 - 2*(-1)*1 = 1 + 0.5 + 2 = 3.5
        let e2 = model.energy(&[false, true, true]);
        assert!(
            (e2 - 3.5).abs() < 1e-10,
            "Energy mismatch: expected 3.5, got {}",
            e2
        );
    }

    // ---------------------------------------------------------------
    // 3. Two-spin system exact solution
    // ---------------------------------------------------------------
    #[test]
    fn test_two_spin_exact_solution() {
        // Ferromagnetic 2-spin: H = -J s0 s1, J = 1.
        // Ground states: (T,T) and (F,F) with energy -1.
        let model = IsingModel::new(2, vec![0.0, 0.0], vec![(0, 1, 1.0)]).unwrap();

        let (gs_energy, gs_config) = model.brute_force_ground_state();
        assert!(
            (gs_energy - (-1.0)).abs() < 1e-10,
            "Ground state energy should be -1, got {}",
            gs_energy
        );
        assert_eq!(
            gs_config[0], gs_config[1],
            "Ground state spins should be aligned"
        );
    }

    // ---------------------------------------------------------------
    // 4. Linear vs quadratic schedule shape
    // ---------------------------------------------------------------
    #[test]
    fn test_schedule_shapes() {
        let linear = QuantumAnnealingSchedule::Linear;
        let quadratic = QuantumAnnealingSchedule::Quadratic;

        // At t=0, both should give s=0.
        assert!((linear.evaluate(0.0)).abs() < 1e-10);
        assert!((quadratic.evaluate(0.0)).abs() < 1e-10);

        // At t=1, both should give s=1.
        assert!((linear.evaluate(1.0) - 1.0).abs() < 1e-10);
        assert!((quadratic.evaluate(1.0) - 1.0).abs() < 1e-10);

        // At t=0.5: linear -> 0.5, quadratic -> 0.25.
        assert!((linear.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((quadratic.evaluate(0.5) - 0.25).abs() < 1e-10);

        // Quadratic should always be <= linear for t in [0,1].
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            assert!(
                quadratic.evaluate(t) <= linear.evaluate(t) + 1e-10,
                "Quadratic should be <= linear at t={}",
                t
            );
        }
    }

    // ---------------------------------------------------------------
    // 5. Classical SA finds ground state of small problem
    // ---------------------------------------------------------------
    #[test]
    fn test_classical_sa_finds_ground_state() {
        // Simple ferromagnetic chain: ground state is all-aligned.
        let n = 6;
        let j: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        let model = IsingModel::new(n, vec![0.0; n], j).unwrap();

        let config = QuantumAnnealerConfig {
            n_steps: 2000,
            n_runs: 5,
            ..Default::default()
        };

        let annealer = QuantumAnnealer::new(model.clone(), config);
        let result = annealer.anneal_classical();

        let (gs_energy, _) = model.brute_force_ground_state();
        assert!(
            (result.best_energy - gs_energy).abs() < 1e-8,
            "SA should find ground state: got {} expected {}",
            result.best_energy,
            gs_energy
        );
    }

    // ---------------------------------------------------------------
    // 6. SQA finds ground state of small problem
    // ---------------------------------------------------------------
    #[test]
    fn test_sqa_finds_ground_state() {
        // Anti-ferromagnetic chain: known ground state alternates.
        let n = 4;
        let j: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, -1.0)).collect();
        let model = IsingModel::new(n, vec![0.0; n], j).unwrap();

        let config = QuantumAnnealerConfig {
            n_steps: 1000,
            n_trotter: 16,
            beta: 5.0,
            gamma_initial: 2.0,
            n_runs: 10,
            n_sweeps_per_step: 1,
            ..Default::default()
        };

        let annealer = QuantumAnnealer::new(model.clone(), config);
        let result = annealer.anneal_quantum();

        let (gs_energy, _) = model.brute_force_ground_state();
        assert!(
            (result.best_energy - gs_energy).abs() < 1e-8,
            "SQA should find ground state: got {} expected {}",
            result.best_energy,
            gs_energy
        );
    }

    // ---------------------------------------------------------------
    // 7. Max-cut on triangle graph
    // ---------------------------------------------------------------
    #[test]
    fn test_max_cut_triangle() {
        // Triangle: 3 vertices, 3 edges.  Max cut = 2 (any 1-vs-2 partition).
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let model = max_cut(3, &edges);

        let (gs_energy, gs_config) = model.brute_force_ground_state();

        // In the max-cut Ising encoding, each satisfied edge contributes -0.5
        // to the energy (s_i != s_j -> -J*(-1) = -(-0.5)*(-1) = -0.5).
        // Max cut of 2 edges -> minimum energy should correspond to 2 anti-aligned pairs.
        // Count anti-aligned edges in ground state.
        let mut cut_size = 0;
        for &(a, b) in &edges {
            if gs_config[a] != gs_config[b] {
                cut_size += 1;
            }
        }
        assert_eq!(cut_size, 2, "Triangle max cut should be 2, got {}", cut_size);
    }

    // ---------------------------------------------------------------
    // 8. Number partitioning small instance
    // ---------------------------------------------------------------
    #[test]
    fn test_number_partitioning() {
        // Numbers: [1, 2, 3] -> best partition: {3} vs {1,2}, diff = 0.
        let model = number_partitioning(&[1.0, 2.0, 3.0]);

        let config = QuantumAnnealerConfig {
            n_steps: 2000,
            n_runs: 10,
            ..Default::default()
        };

        let annealer = QuantumAnnealer::new(model, config);
        let result = annealer.anneal_classical();

        // Compute partition difference.
        let numbers = [1.0, 2.0, 3.0];
        let diff: f64 = result
            .best_config
            .iter()
            .enumerate()
            .map(|(i, &b)| if b { numbers[i] } else { -numbers[i] })
            .sum::<f64>()
            .abs();

        assert!(
            diff < 1e-8,
            "Should find perfect partition, got diff = {}",
            diff
        );
    }

    // ---------------------------------------------------------------
    // 9. Energy monotonically decreases during classical annealing
    // ---------------------------------------------------------------
    #[test]
    fn test_energy_monotonic_decrease_classical() {
        let n = 5;
        let j: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        let model = IsingModel::new(n, vec![0.0; n], j).unwrap();

        let config = QuantumAnnealerConfig {
            n_steps: 500,
            n_runs: 1, // single run for clean history
            ..Default::default()
        };

        let annealer = QuantumAnnealer::new(model, config);
        let result = annealer.anneal_classical();

        // The energy history tracks the best-so-far, which should be
        // non-increasing (since we only record improvements).
        for i in 1..result.energy_history.len() {
            assert!(
                result.energy_history[i] <= result.energy_history[i - 1] + 1e-10,
                "Best energy should be non-increasing at step {}: {} > {}",
                i,
                result.energy_history[i],
                result.energy_history[i - 1]
            );
        }
    }

    // ---------------------------------------------------------------
    // 10. Statevector evolution preserves norm
    // ---------------------------------------------------------------
    #[test]
    fn test_statevector_norm_preservation() {
        // 3-qubit system -- should preserve norm throughout.
        let model = IsingModel::new(
            3,
            vec![0.5, -0.3, 0.1],
            vec![(0, 1, 0.8), (1, 2, -0.5)],
        )
        .unwrap();

        let config = QuantumAnnealerConfig {
            n_steps: 100,
            ..Default::default()
        };

        let annealer = QuantumAnnealer::new(model, config);
        // If norm is not preserved, anneal_statevector returns an error.
        let result = annealer.anneal_statevector(3);
        assert!(
            result.is_ok(),
            "Statevector evolution should preserve norm: {:?}",
            result.err()
        );
    }

    // ---------------------------------------------------------------
    // 11. Random Ising model generation
    // ---------------------------------------------------------------
    #[test]
    fn test_random_ising_generation() {
        let model = random_ising(10, 0.5);
        assert_eq!(model.n_spins, 10);
        assert_eq!(model.h.len(), 10);

        // With density 0.5 and 10 spins, expected couplings ~ 45 * 0.5 = 22.5.
        // Allow wide range for randomness.
        assert!(
            !model.j.is_empty(),
            "Random Ising should have some couplings"
        );
        assert!(
            model.j.len() < 50,
            "Should not exceed total possible pairs"
        );

        // All indices should be in range.
        for &(a, b, _) in &model.j {
            assert!(a < 10 && b < 10);
            assert!(a < b, "Couplings should have i < j");
        }
    }

    // ---------------------------------------------------------------
    // 12. Trotter convergence: more slices -> better
    // ---------------------------------------------------------------
    #[test]
    fn test_trotter_convergence() {
        // A frustrated system where SQA quality should improve with more
        // Trotter slices.  We compare P=4 vs P=32.
        let n = 4;
        let model = IsingModel::new(
            n,
            vec![0.1, -0.2, 0.3, -0.1],
            vec![(0, 1, 1.0), (1, 2, -1.0), (2, 3, 0.7), (0, 3, -0.5)],
        )
        .unwrap();

        let (gs_energy, _) = model.brute_force_ground_state();

        // Run with few Trotter slices.
        let config_few = QuantumAnnealerConfig {
            n_steps: 500,
            n_trotter: 4,
            beta: 5.0,
            gamma_initial: 2.0,
            n_runs: 20,
            n_sweeps_per_step: 1,
            ..Default::default()
        };

        let annealer_few = QuantumAnnealer::new(model.clone(), config_few);
        let result_few = annealer_few.anneal_quantum();

        // Run with many Trotter slices.
        let config_many = QuantumAnnealerConfig {
            n_steps: 500,
            n_trotter: 32,
            beta: 5.0,
            gamma_initial: 2.0,
            n_runs: 20,
            n_sweeps_per_step: 1,
            ..Default::default()
        };

        let annealer_many = QuantumAnnealer::new(model, config_many);
        let result_many = annealer_many.anneal_quantum();

        // More Trotter slices should yield energy at least as good.
        // (Allow tolerance for stochasticity; the many-slice result should
        // not be significantly worse.)
        assert!(
            result_many.best_energy <= result_few.best_energy + 0.5,
            "More Trotter slices should not significantly degrade quality: P=32 got {}, P=4 got {}",
            result_many.best_energy,
            result_few.best_energy
        );
    }

    // ---------------------------------------------------------------
    // 13. Custom schedule interpolation
    // ---------------------------------------------------------------
    #[test]
    fn test_custom_schedule() {
        let schedule = QuantumAnnealingSchedule::Custom(vec![
            (0.0, 0.0),
            (0.3, 0.5),
            (0.7, 0.5),
            (1.0, 1.0),
        ]);

        // At boundaries.
        assert!((schedule.evaluate(0.0)).abs() < 1e-10);
        assert!((schedule.evaluate(1.0) - 1.0).abs() < 1e-10);

        // In the plateau region.
        assert!((schedule.evaluate(0.5) - 0.5).abs() < 1e-10);

        // In the ramp regions.
        let s015 = schedule.evaluate(0.15);
        assert!(s015 > 0.0 && s015 < 0.5);
    }

    // ---------------------------------------------------------------
    // 14. Pause schedule behaviour
    // ---------------------------------------------------------------
    #[test]
    fn test_pause_schedule() {
        let schedule = QuantumAnnealingSchedule::Pause {
            start: 0.3,
            end: 0.7,
            s_pause: 0.4,
        };

        // Before pause: ramp from 0 to s_pause.
        assert!((schedule.evaluate(0.0)).abs() < 1e-10);
        let s_at_start = schedule.evaluate(0.3);
        assert!(
            (s_at_start - 0.4).abs() < 1e-10,
            "At pause start, s should be s_pause"
        );

        // During pause.
        assert!((schedule.evaluate(0.5) - 0.4).abs() < 1e-10);

        // After pause: ramp from s_pause to 1.
        assert!((schedule.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 15. SK model structure
    // ---------------------------------------------------------------
    #[test]
    fn test_sk_model() {
        let model = sk_model(8);
        assert_eq!(model.n_spins, 8);
        assert_eq!(model.h, vec![0.0; 8]);
        // Fully connected: 8*7/2 = 28 couplings.
        assert_eq!(model.j.len(), 28);
    }

    // ---------------------------------------------------------------
    // 16. IsingModel validation
    // ---------------------------------------------------------------
    #[test]
    fn test_ising_model_validation() {
        // Valid model.
        let ok = IsingModel::new(2, vec![1.0, 2.0], vec![(0, 1, 0.5)]);
        assert!(ok.is_ok());

        // Wrong h length.
        let bad_h = IsingModel::new(2, vec![1.0], vec![]);
        assert!(bad_h.is_err());

        // Out of range coupling.
        let bad_j = IsingModel::new(2, vec![1.0, 2.0], vec![(0, 5, 1.0)]);
        assert!(bad_j.is_err());
    }
}
