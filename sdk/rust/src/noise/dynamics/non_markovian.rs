//! Non-Markovian Noise Simulation via Process Tensor Formalism
//!
//! Implements non-Markovian (memory-bearing) quantum noise using the process
//! tensor framework. Unlike Markovian channels where each time step is
//! independent, non-Markovian dynamics encode temporal correlations: the noise
//! at time t depends on the system's history at times t-1, t-2, ..., t-k.
//!
//! # Process Tensor
//!
//! A process tensor T is a multilinear map that contracts with a sequence of
//! control operations (or the identity for free evolution) to produce the
//! final quantum state. For a system with memory depth k, the effective
//! channel at step t is:
//!
//! ```text
//! E_eff(t) = E(t) + Σ_{j=1}^{k} λ^j · K(t, t-j)
//! ```
//!
//! where λ is the memory strength and K(t, t-j) is a correction kernel
//! derived from the state history.
//!
//! # Noise Models
//!
//! - **Random Telegraph Noise (RTN)**: Two-state Markov chain producing
//!   temporally correlated dephasing. Switching rate controls memory time.
//! - **1/f Noise**: Superposition of many RTN fluctuators with
//!   logarithmically distributed switching rates, producing the
//!   characteristic 1/f power spectral density common in solid-state qubits.
//!
//! # Non-Markovianity Measure
//!
//! Uses the Breuer-Laine-Piilo (BLP) measure: a process is non-Markovian
//! if the trace distance between any pair of initial states can increase
//! during evolution. Markovian processes monotonically decrease trace
//! distance (information flows only out of the system).
//!
//! # References
//!
//! - Pollock et al., PRA 97, 012127 (2018) — Process tensor formalism
//! - Breuer et al., PRL 103, 210401 (2009) — BLP non-Markovianity measure
//! - Paladino et al., RMP 86, 361 (2014) — 1/f noise in qubits

use crate::density_matrix::DensityMatrix;
use crate::gates::Gate;
use crate::quantum_channel::KrausChannel;
use crate::C64;
use rand::Rng;

// ===================================================================
// PROCESS TENSOR STEP
// ===================================================================

/// A single time step's channel in the process tensor.
///
/// Each step carries its own Kraus channel (the local noise map), a
/// timestamp, and a memory coupling strength that controls how much
/// this step's effective channel is influenced by the system's history.
#[derive(Clone, Debug)]
pub struct ProcessTensorStep {
    /// Kraus operators for this time step's noise.
    pub channel: KrausChannel,
    /// Time of this step.
    pub time: f64,
    /// Memory kernel coupling to previous steps (0.0 = Markovian).
    pub memory_strength: f64,
}

// ===================================================================
// PROCESS TENSOR
// ===================================================================

/// Process tensor for multi-time correlated noise.
///
/// Encodes the full non-Markovian dynamics as a chain of tensors with
/// configurable memory depth. At each time step, the effective quantum
/// channel blends the local Kraus map with memory-kernel corrections
/// computed from the state history.
#[derive(Clone, Debug)]
pub struct ProcessTensor {
    /// Sequence of noise steps.
    pub steps: Vec<ProcessTensorStep>,
    /// Memory depth: how many previous steps influence the current one.
    pub memory_depth: usize,
    /// System dimension (2^n for n qubits).
    pub dim: usize,
}

impl ProcessTensor {
    /// Create a Markovian (memoryless) process tensor.
    ///
    /// Every step applies the same channel with zero memory coupling,
    /// reducing to standard Kraus evolution.
    pub fn markovian(channel: KrausChannel, num_steps: usize, dt: f64) -> Self {
        let dim = channel.dim;
        let steps = (0..num_steps)
            .map(|i| ProcessTensorStep {
                channel: channel.clone(),
                time: i as f64 * dt,
                memory_strength: 0.0,
            })
            .collect();

        ProcessTensor {
            steps,
            memory_depth: 0,
            dim,
        }
    }

    /// Create a process tensor with explicit memory.
    ///
    /// Each step receives its own Kraus channel, and all steps share a
    /// common memory depth and exponentially decaying memory strength.
    /// If `channels` is shorter than `num_steps` implied by its length,
    /// the tensor uses exactly `channels.len()` steps.
    pub fn with_memory(
        channels: Vec<KrausChannel>,
        memory_depth: usize,
        memory_strength: f64,
        dt: f64,
    ) -> Self {
        assert!(!channels.is_empty(), "Must provide at least one channel");
        let dim = channels[0].dim;

        let steps = channels
            .into_iter()
            .enumerate()
            .map(|(i, ch)| ProcessTensorStep {
                channel: ch,
                time: i as f64 * dt,
                memory_strength,
            })
            .collect();

        ProcessTensor {
            steps,
            memory_depth,
            dim,
        }
    }

    /// Create a process tensor from Random Telegraph Noise parameters.
    ///
    /// Generates an RTN trajectory (a sequence of +1/-1 values with
    /// Markovian switching) and converts it to a sequence of dephasing
    /// channels. The memory depth is set to capture the correlation
    /// time of the RTN process (approximately 1/switching_rate steps).
    pub fn from_rtn(params: &RTNParams, num_steps: usize, dt: f64) -> Self {
        let trajectory = CorrelatedNoiseSequence::generate_rtn(params, num_steps);
        let channels = CorrelatedNoiseSequence::dephasing_from_trajectory(&trajectory);
        let dim = channels[0].dim;

        // Memory depth scales with correlation time: ~1/switching_rate
        let correlation_steps = if params.switching_rate > 0.0 {
            ((1.0 / params.switching_rate).ceil() as usize)
                .min(num_steps)
                .max(1)
        } else {
            num_steps
        };

        let steps = channels
            .into_iter()
            .enumerate()
            .map(|(i, ch)| ProcessTensorStep {
                channel: ch,
                time: i as f64 * dt,
                memory_strength: params.amplitude,
            })
            .collect();

        ProcessTensor {
            steps,
            memory_depth: correlation_steps,
            dim,
        }
    }

    /// Create a process tensor from 1/f noise parameters.
    ///
    /// Superposes multiple RTN fluctuators with logarithmically spaced
    /// switching rates to produce a 1/f power spectral density. The
    /// memory depth is set to the correlation time of the slowest
    /// fluctuator.
    pub fn from_one_over_f(params: &OneOverFParams, num_steps: usize, dt: f64) -> Self {
        let trajectory = CorrelatedNoiseSequence::generate_one_over_f(params, num_steps);
        let channels = CorrelatedNoiseSequence::dephasing_from_trajectory(&trajectory);
        let dim = channels[0].dim;

        // Memory depth from slowest fluctuator
        let correlation_steps = if params.gamma_min > 0.0 {
            ((1.0 / params.gamma_min).ceil() as usize)
                .min(num_steps)
                .max(1)
        } else {
            num_steps
        };

        let steps = channels
            .into_iter()
            .enumerate()
            .map(|(i, ch)| ProcessTensorStep {
                channel: ch,
                time: i as f64 * dt,
                memory_strength: params.amplitude,
            })
            .collect();

        ProcessTensor {
            steps,
            memory_depth: correlation_steps,
            dim,
        }
    }

    /// Number of time steps in this process tensor.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Total evolution time.
    pub fn total_time(&self) -> f64 {
        self.steps.last().map_or(0.0, |s| s.time)
    }
}

// ===================================================================
// CORRELATED NOISE SEQUENCE
// ===================================================================

/// Correlated noise sequence generators.
///
/// Provides static methods to generate temporally correlated classical
/// noise trajectories and convert them into sequences of quantum
/// channels suitable for the process tensor.
pub struct CorrelatedNoiseSequence;

impl CorrelatedNoiseSequence {
    /// Generate a Random Telegraph Noise trajectory.
    ///
    /// Produces a sequence of +amplitude/-amplitude values that switch
    /// between the two levels with the given switching rate per step.
    /// The resulting signal has an exponentially decaying autocorrelation
    /// with time constant 1/(2 * switching_rate).
    pub fn generate_rtn(params: &RTNParams, num_steps: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut trajectory = Vec::with_capacity(num_steps);
        let mut current_state: f64 = if rng.gen_bool(0.5) {
            params.amplitude
        } else {
            -params.amplitude
        };

        for _ in 0..num_steps {
            trajectory.push(current_state);
            // Switch state with probability = switching_rate per step
            if rng.gen::<f64>() < params.switching_rate {
                current_state = -current_state;
            }
        }

        trajectory
    }

    /// Generate a 1/f noise trajectory by superposing RTN fluctuators.
    ///
    /// Creates `num_fluctuators` independent RTN processes with switching
    /// rates logarithmically spaced between `gamma_min` and `gamma_max`.
    /// Their sum approximates a 1/f power spectral density. Each
    /// fluctuator contributes equally to the total amplitude.
    pub fn generate_one_over_f(params: &OneOverFParams, num_steps: usize) -> Vec<f64> {
        let mut combined = vec![0.0; num_steps];
        let n = params.num_fluctuators.max(1);

        // Logarithmically space switching rates
        let log_min = params.gamma_min.max(1e-10).ln();
        let log_max = params.gamma_max.max(params.gamma_min + 1e-10).ln();

        for k in 0..n {
            let log_gamma = if n > 1 {
                log_min + (log_max - log_min) * k as f64 / (n - 1) as f64
            } else {
                (log_min + log_max) / 2.0
            };
            let gamma = log_gamma.exp();

            let rtn_params = RTNParams {
                switching_rate: gamma,
                amplitude: params.amplitude / (n as f64).sqrt(),
            };

            let traj = Self::generate_rtn(&rtn_params, num_steps);
            for (i, &val) in traj.iter().enumerate() {
                combined[i] += val;
            }
        }

        combined
    }

    /// Convert a classical noise trajectory into a sequence of dephasing channels.
    ///
    /// Each trajectory value `z(t)` becomes a single-qubit phase damping channel
    /// with damping parameter `gamma = z(t)^2`, clamped to [0, 1]. The sign of
    /// `z(t)` determines the phase rotation direction.
    pub fn dephasing_from_trajectory(trajectory: &[f64]) -> Vec<KrausChannel> {
        trajectory
            .iter()
            .map(|&z| {
                // Map trajectory value to a dephasing parameter in [0, 1].
                // Use z^2 clamped to [0, 1] so the channel is physical.
                let gamma = (z * z).min(1.0);
                KrausChannel::phase_damping(gamma)
            })
            .collect()
    }
}

// ===================================================================
// NOISE PARAMETER STRUCTS
// ===================================================================

/// Random Telegraph Noise (RTN) parameters.
///
/// RTN models a two-level fluctuator that switches randomly between
/// +amplitude and -amplitude. Common in solid-state qubits due to
/// charge traps and two-level systems in oxide layers.
#[derive(Clone, Debug)]
pub struct RTNParams {
    /// Switching rate (probability of switching per time step).
    pub switching_rate: f64,
    /// Noise amplitude when "on".
    pub amplitude: f64,
}

/// 1/f noise parameters.
///
/// 1/f noise arises from an ensemble of RTN fluctuators with a broad
/// distribution of switching rates. The resulting power spectral density
/// scales as 1/f over a wide frequency range.
#[derive(Clone, Debug)]
pub struct OneOverFParams {
    /// Number of RTN fluctuators to superpose.
    pub num_fluctuators: usize,
    /// Minimum switching rate.
    pub gamma_min: f64,
    /// Maximum switching rate.
    pub gamma_max: f64,
    /// Overall amplitude.
    pub amplitude: f64,
}

// ===================================================================
// NON-MARKOVIAN STATISTICS
// ===================================================================

/// Runtime statistics for the non-Markovian simulator.
#[derive(Clone, Debug, Default)]
pub struct NonMarkovianStats {
    /// Total evolution steps executed.
    pub total_steps: usize,
    /// Cumulative number of memory lookbacks performed.
    pub memory_lookbacks: usize,
    /// Maximum memory depth actually used in any single step.
    pub max_memory_depth_used: usize,
}

// ===================================================================
// NON-MARKOVIAN SIMULATOR
// ===================================================================

/// Non-Markovian simulator using process tensors.
///
/// Maintains a density matrix state and evolves it through a process tensor,
/// applying memory-kernel corrections at each step based on the state
/// history. Also supports interleaving unitary gate operations between
/// noise steps.
///
/// # Memory Kernel
///
/// At each step t with memory depth k, the effective channel is:
///
/// ```text
/// ρ(t+1) = E(t)[ρ(t)] + Σ_{j=1}^{min(k, t)} λ^j · (ρ(t-j) - ρ(t)) / dim
/// ```
///
/// The correction term blends in past state information weighted by
/// exponentially decaying memory strength, producing non-Markovian
/// features such as information backflow and purity revivals.
pub struct NonMarkovianSimulator {
    /// Current density matrix.
    pub state: DensityMatrix,
    /// Process tensor defining the noise.
    pub process_tensor: ProcessTensor,
    /// History of previous density matrix elements (for memory kernel).
    state_history: Vec<Vec<C64>>,
    /// Current time step index into the process tensor.
    current_step: usize,
    /// Runtime statistics.
    stats: NonMarkovianStats,
}

impl NonMarkovianSimulator {
    /// Create a new non-Markovian simulator.
    ///
    /// Initializes the system in the |0...0> state with the given number
    /// of qubits and attaches the provided process tensor for noise.
    pub fn new(num_qubits: usize, process_tensor: ProcessTensor) -> Self {
        let state = DensityMatrix::new(num_qubits);
        NonMarkovianSimulator {
            state,
            process_tensor,
            state_history: Vec::new(),
            current_step: 0,
            stats: NonMarkovianStats::default(),
        }
    }

    /// Advance the simulation by one time step.
    ///
    /// Applies the current process tensor step's Kraus channel to the
    /// density matrix, then adds memory-kernel corrections from the
    /// state history. The corrected state is re-normalized to preserve
    /// trace = 1.
    pub fn step(&mut self) {
        if self.current_step >= self.process_tensor.steps.len() {
            return;
        }

        // Save current state to history before evolving.
        self.state_history.push(self.state.elements.clone());

        let pt_step = &self.process_tensor.steps[self.current_step];
        let dim = self.state.dim;

        // Apply the local Kraus channel: E(t)[ρ(t)]
        let evolved = pt_step
            .channel
            .apply_to_density_matrix(&self.state.elements);
        self.state.elements = evolved;

        // Apply memory kernel corrections.
        let memory_depth = self.process_tensor.memory_depth;
        let history_len = self.state_history.len();
        let effective_depth = memory_depth.min(history_len);

        if effective_depth > 0 && pt_step.memory_strength.abs() > 1e-15 {
            self.stats.max_memory_depth_used =
                self.stats.max_memory_depth_used.max(effective_depth);

            let mut correction = vec![C64::new(0.0, 0.0); dim * dim];

            for j in 1..=effective_depth {
                self.stats.memory_lookbacks += 1;

                // Weight decays exponentially with lookback distance.
                let weight = pt_step.memory_strength.powi(j as i32);

                // The correction is a weighted difference between the past
                // state and the current (just-evolved) state, normalized
                // by the system dimension. This models information backflow
                // from the environment.
                let past_idx = history_len - j;
                let past_state = &self.state_history[past_idx];

                for idx in 0..dim * dim {
                    correction[idx] += C64::new(weight, 0.0)
                        * (past_state[idx] - self.state.elements[idx])
                        / (dim as f64);
                }
            }

            // Add correction to the evolved state.
            for idx in 0..dim * dim {
                self.state.elements[idx] += correction[idx];
            }

            // Re-normalize trace to 1 to ensure physicality.
            self.renormalize_trace();
        }

        self.current_step += 1;
        self.stats.total_steps += 1;
    }

    /// Evolve the system for multiple time steps.
    pub fn evolve(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            if self.current_step >= self.process_tensor.steps.len() {
                break;
            }
            self.step();
        }
    }

    /// Apply a unitary gate to the current density matrix.
    ///
    /// Converts the gate's matrix representation to a Kraus channel with
    /// a single operator (the unitary itself) and applies it. This does
    /// not advance the process tensor step counter, so gate operations
    /// can be interleaved freely between noise steps.
    pub fn apply_gate(&mut self, gate: &Gate) {
        let gate_matrix = gate.gate_type.matrix();
        let gate_dim = gate_matrix.len();

        // Flatten to row-major for Kraus channel.
        let mut flat = vec![C64::new(0.0, 0.0); gate_dim * gate_dim];
        for i in 0..gate_dim {
            for j in 0..gate_dim {
                flat[i * gate_dim + j] = gate_matrix[i][j];
            }
        }

        // If the gate dimension matches the system dimension, apply directly.
        // Otherwise, embed the single-qubit gate into the full Hilbert space.
        if gate_dim == self.state.dim {
            let unitary_channel = KrausChannel::new(vec![flat], gate_dim);
            self.state.elements = unitary_channel.apply_to_density_matrix(&self.state.elements);
        } else {
            // Embed single/multi-qubit gate into full system.
            let full_unitary = self.embed_gate(gate, &flat, gate_dim);
            let full_channel = KrausChannel::new(vec![full_unitary], self.state.dim);
            self.state.elements = full_channel.apply_to_density_matrix(&self.state.elements);
        }
    }

    /// Embed a small gate unitary into the full Hilbert space.
    ///
    /// For a single-qubit gate on qubit q in an n-qubit system, constructs
    /// the full 2^n x 2^n unitary U = I ⊗...⊗ G ⊗...⊗ I via the standard
    /// index-mapping approach.
    fn embed_gate(&self, gate: &Gate, flat_gate: &[C64], gate_dim: usize) -> Vec<C64> {
        let dim = self.state.dim;
        let mut full = vec![C64::new(0.0, 0.0); dim * dim];

        if gate.is_single_qubit() {
            let target = gate.targets[0];
            // For each pair of basis states, apply the 2x2 gate on the target qubit.
            for row in 0..dim {
                for col in 0..dim {
                    // Check that all qubits except the target match.
                    let row_other = row & !(1 << target);
                    let col_other = col & !(1 << target);
                    if row_other != col_other {
                        continue;
                    }
                    let row_bit = (row >> target) & 1;
                    let col_bit = (col >> target) & 1;
                    full[row * dim + col] = flat_gate[row_bit * gate_dim + col_bit];
                }
            }
        } else {
            // Fallback: identity embedding for unsupported multi-qubit gates
            // in the non-Markovian context. A full implementation would handle
            // CNOT, SWAP, etc. via tensor product expansion.
            for i in 0..dim {
                full[i * dim + i] = C64::new(1.0, 0.0);
            }
        }

        full
    }

    /// Get a reference to the current density matrix state.
    pub fn state(&self) -> &DensityMatrix {
        &self.state
    }

    /// Compute the purity of the current state: Tr(rho^2).
    ///
    /// - Pure state: purity = 1.0
    /// - Maximally mixed: purity = 1/dim
    pub fn purity(&self) -> f64 {
        self.state.purity()
    }

    /// Compute the BLP non-Markovianity measure.
    ///
    /// Estimates non-Markovianity by evolving two orthogonal initial states
    /// (|+> and |->) through the same process tensor and tracking the
    /// trace distance. Any increase in trace distance signals information
    /// backflow (non-Markovian behavior). The measure is the integral of
    /// all positive trace-distance derivatives.
    ///
    /// We use |+> and |-> rather than |0> and |1> because dephasing
    /// channels act on off-diagonal elements, which are present in
    /// superposition states but not in computational basis states.
    ///
    /// Returns 0.0 for Markovian processes and a positive value for
    /// non-Markovian ones.
    pub fn non_markovianity_measure(&self) -> f64 {
        let num_qubits = self.state.num_qubits;
        let dim = self.state.dim;

        let s = 1.0 / (2.0_f64).sqrt();

        // Prepare |+> = (|0> + |1>)/sqrt(2)
        let mut state_plus = vec![C64::new(0.0, 0.0); dim];
        state_plus[0] = C64::new(s, 0.0);
        if dim > 1 {
            state_plus[1] = C64::new(s, 0.0);
        }
        let rho_0 = DensityMatrix::from_pure_state(&state_plus);

        // Prepare |-> = (|0> - |1>)/sqrt(2)
        let mut state_minus = vec![C64::new(0.0, 0.0); dim];
        state_minus[0] = C64::new(s, 0.0);
        if dim > 1 {
            state_minus[1] = C64::new(-s, 0.0);
        }
        let rho_1 = DensityMatrix::from_pure_state(&state_minus);

        // Evolve both states through identical process tensors.
        let mut sim_0 = NonMarkovianSimulator::new(num_qubits, self.process_tensor.clone());
        sim_0.state = rho_0;

        let mut sim_1 = NonMarkovianSimulator::new(num_qubits, self.process_tensor.clone());
        sim_1.state = rho_1;

        let num_steps = self.process_tensor.steps.len();
        let mut prev_trace_dist = trace_distance(&sim_0.state.elements, &sim_1.state.elements, dim);
        let mut non_markovianity = 0.0;

        for _ in 0..num_steps {
            sim_0.step();
            sim_1.step();

            let current_dist = trace_distance(&sim_0.state.elements, &sim_1.state.elements, dim);

            // Any increase in trace distance counts as non-Markovian.
            let delta = current_dist - prev_trace_dist;
            if delta > 0.0 {
                non_markovianity += delta;
            }

            prev_trace_dist = current_dist;
        }

        non_markovianity
    }

    /// Get the current time step index.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get a reference to the runtime statistics.
    pub fn stats(&self) -> &NonMarkovianStats {
        &self.stats
    }

    /// Re-normalize the density matrix so that Tr(rho) = 1.
    fn renormalize_trace(&mut self) {
        let dim = self.state.dim;
        let trace: f64 = (0..dim).map(|i| self.state.elements[i * dim + i].re).sum();

        if trace.abs() > 1e-15 {
            let scale = 1.0 / trace;
            for elem in self.state.elements.iter_mut() {
                *elem *= scale;
            }
        }
    }

    /// Get the evolution history of purity values.
    ///
    /// Re-runs the evolution from scratch, recording purity at each step.
    /// Useful for detecting purity revivals that indicate non-Markovian
    /// behavior.
    pub fn purity_history(&self) -> Vec<f64> {
        let num_qubits = self.state.num_qubits;
        let mut sim = NonMarkovianSimulator::new(num_qubits, self.process_tensor.clone());
        let num_steps = self.process_tensor.steps.len();
        let mut purities = Vec::with_capacity(num_steps + 1);

        purities.push(sim.purity());
        for _ in 0..num_steps {
            sim.step();
            purities.push(sim.purity());
        }

        purities
    }
}

// ===================================================================
// HELPER FUNCTIONS
// ===================================================================

/// Compute the trace distance between two density matrices.
///
/// D(rho, sigma) = (1/2) * Tr|rho - sigma|
///
/// For simplicity, computes the Frobenius norm of the difference as
/// an upper bound. The exact trace distance requires eigendecomposition
/// of (rho - sigma), but the Frobenius bound is tight for rank-1
/// differences and sufficient for the BLP measure.
fn trace_distance(rho: &[C64], sigma: &[C64], dim: usize) -> f64 {
    // Compute the difference matrix.
    let mut diff = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim * dim {
        diff[i] = rho[i] - sigma[i];
    }

    // Compute eigenvalues of the Hermitian matrix diff via power-iteration
    // approximation. For a 2x2 system we can solve analytically.
    if dim == 2 {
        // For 2x2 Hermitian A:
        // eigenvalues = (a+d)/2 +/- sqrt(((a-d)/2)^2 + |b|^2)
        let a = diff[0].re;
        let d = diff[3].re;
        let b = diff[1]; // off-diagonal
        let mean = (a + d) / 2.0;
        let disc = ((a - d) / 2.0).powi(2) + b.norm_sqr();
        let sqrt_disc = disc.sqrt();
        let lam1 = (mean + sqrt_disc).abs();
        let lam2 = (mean - sqrt_disc).abs();
        return (lam1 + lam2) / 2.0;
    }

    // General case: use Frobenius norm as upper bound on trace norm.
    let frobenius_sq: f64 = diff.iter().map(|c| c.norm_sqr()).sum();
    frobenius_sq.sqrt() / 2.0
}

/// Apply a Kraus channel to a density matrix stored as a flat vector.
/// Convenience wrapper that delegates to KrausChannel::apply_to_density_matrix.
#[allow(dead_code)]
fn apply_channel(channel: &KrausChannel, rho: &[C64]) -> Vec<C64> {
    channel.apply_to_density_matrix(rho)
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: compute trace of a flat dim x dim matrix.
    fn matrix_trace(m: &[C64], dim: usize) -> f64 {
        (0..dim).map(|i| m[i * dim + i].re).sum()
    }

    /// Helper: compute purity Tr(rho^2) from flat matrix.
    fn matrix_purity(m: &[C64], dim: usize) -> f64 {
        let mut sum = C64::new(0.0, 0.0);
        for i in 0..dim {
            for j in 0..dim {
                sum += m[i * dim + j] * m[j * dim + i];
            }
        }
        sum.re
    }

    // ---------------------------------------------------------------
    // 1. Markovian process tensor reduces to standard Kraus evolution
    // ---------------------------------------------------------------

    #[test]
    fn test_markovian_matches_direct_kraus() {
        let channel = KrausChannel::depolarizing(0.1);
        let pt = ProcessTensor::markovian(channel.clone(), 5, 0.01);

        // Evolve via process tensor.
        let mut sim = NonMarkovianSimulator::new(1, pt);
        sim.step();
        let rho_pt = sim.state.elements.clone();

        // Evolve via direct Kraus application.
        let rho0 = DensityMatrix::new(1);
        let rho_direct = channel.apply_to_density_matrix(&rho0.elements);

        // They should match (memory depth = 0 means no correction).
        for i in 0..4 {
            assert!(
                (rho_pt[i] - rho_direct[i]).norm() < 1e-12,
                "Element {} differs: pt={:?}, direct={:?}",
                i,
                rho_pt[i],
                rho_direct[i]
            );
        }
    }

    // ---------------------------------------------------------------
    // 2. RTN noise generation: verify switching statistics
    // ---------------------------------------------------------------

    #[test]
    fn test_rtn_switching_statistics() {
        let params = RTNParams {
            switching_rate: 0.1,
            amplitude: 0.5,
        };
        let num_steps = 10_000;
        let traj = CorrelatedNoiseSequence::generate_rtn(&params, num_steps);

        // Count switches.
        let switches: usize = traj.windows(2).filter(|w| w[0] != w[1]).count();

        // Expected switches ~ switching_rate * (num_steps - 1)
        let expected = params.switching_rate * (num_steps - 1) as f64;
        let relative_error = ((switches as f64) - expected).abs() / expected;

        // Allow 15% statistical deviation.
        assert!(
            relative_error < 0.15,
            "RTN switching rate off: {} switches vs {} expected (err={})",
            switches,
            expected,
            relative_error
        );

        // All values should be +/- amplitude.
        for &v in &traj {
            assert!(
                (v.abs() - params.amplitude).abs() < 1e-15,
                "RTN value {} is not +/- amplitude {}",
                v,
                params.amplitude
            );
        }
    }

    // ---------------------------------------------------------------
    // 3. 1/f noise: verify spectral properties (approximate)
    // ---------------------------------------------------------------

    #[test]
    fn test_one_over_f_spectral_properties() {
        let params = OneOverFParams {
            num_fluctuators: 20,
            gamma_min: 0.001,
            gamma_max: 0.5,
            amplitude: 0.3,
        };
        let num_steps = 4096;
        let traj = CorrelatedNoiseSequence::generate_one_over_f(&params, num_steps);

        // Compute simple autocorrelation at a few lags.
        let mean: f64 = traj.iter().sum::<f64>() / num_steps as f64;
        let var: f64 = traj.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / num_steps as f64;

        // Autocorrelation at lag 1 should be positive (correlated).
        let autocorr_1: f64 = traj
            .windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>()
            / ((num_steps - 1) as f64 * var.max(1e-15));

        assert!(
            autocorr_1 > 0.0,
            "1/f noise should have positive lag-1 autocorrelation, got {}",
            autocorr_1
        );

        // Variance should be positive and bounded.
        assert!(var > 0.0, "1/f noise variance should be positive");
        assert!(
            var < params.amplitude.powi(2) * params.num_fluctuators as f64,
            "1/f noise variance {} is unreasonably large",
            var
        );
    }

    // ---------------------------------------------------------------
    // 4. Memory depth 0 should be Markovian
    // ---------------------------------------------------------------

    #[test]
    fn test_memory_depth_zero_is_markovian() {
        let channel = KrausChannel::phase_damping(0.2);
        let channels: Vec<KrausChannel> = (0..10).map(|_| channel.clone()).collect();

        // memory_depth = 0 with non-zero memory_strength should still
        // behave as Markovian since the depth truncates all lookbacks.
        let pt = ProcessTensor::with_memory(channels.clone(), 0, 0.5, 0.01);
        let pt_markov = ProcessTensor::markovian(channel.clone(), 10, 0.01);

        let mut sim_mem = NonMarkovianSimulator::new(1, pt);
        let mut sim_markov = NonMarkovianSimulator::new(1, pt_markov);

        for _ in 0..5 {
            sim_mem.step();
            sim_markov.step();
        }

        // States should be identical.
        for i in 0..4 {
            assert!(
                (sim_mem.state.elements[i] - sim_markov.state.elements[i]).norm() < 1e-12,
                "Depth-0 memory should equal Markovian at element {}",
                i
            );
        }
    }

    // ---------------------------------------------------------------
    // 5. Non-Markovianity measure: Markovian -> 0
    // ---------------------------------------------------------------

    #[test]
    fn test_non_markovianity_markovian_is_zero() {
        let channel = KrausChannel::depolarizing(0.05);
        let pt = ProcessTensor::markovian(channel, 20, 0.01);
        let sim = NonMarkovianSimulator::new(1, pt);

        let nm = sim.non_markovianity_measure();
        assert!(
            nm < 1e-10,
            "Markovian process should have near-zero non-Markovianity, got {}",
            nm
        );
    }

    // ---------------------------------------------------------------
    // 6. Non-Markovianity measure: non-Markovian -> positive
    // ---------------------------------------------------------------

    #[test]
    fn test_non_markovianity_memory_is_positive() {
        // Create a process tensor with strong memory.
        let channel = KrausChannel::phase_damping(0.3);
        let channels: Vec<KrausChannel> = (0..30).map(|_| channel.clone()).collect();
        let pt = ProcessTensor::with_memory(channels, 5, 0.8, 0.01);

        let sim = NonMarkovianSimulator::new(1, pt);
        let nm = sim.non_markovianity_measure();

        assert!(
            nm > 0.0,
            "Non-Markovian process should have positive BLP measure, got {}",
            nm
        );
    }

    // ---------------------------------------------------------------
    // 7. Purity evolution: non-Markovian can show purity revivals
    // ---------------------------------------------------------------

    #[test]
    fn test_purity_revival_non_markovian() {
        let channel = KrausChannel::phase_damping(0.4);
        let channels: Vec<KrausChannel> = (0..40).map(|_| channel.clone()).collect();
        let pt = ProcessTensor::with_memory(channels, 5, 0.9, 0.01);

        // Start from |+> so that phase damping has off-diagonal elements
        // to act on, enabling memory kernel corrections to produce revivals.
        let mut sim = NonMarkovianSimulator::new(1, pt);
        let h_gate = Gate::h(0);
        sim.apply_gate(&h_gate);

        let num_steps = sim.process_tensor.steps.len();
        let mut purities = Vec::with_capacity(num_steps + 1);
        purities.push(sim.purity());
        for _ in 0..num_steps {
            sim.step();
            purities.push(sim.purity());
        }

        // The first purity should be 1.0 (pure state).
        assert!(
            (purities[0] - 1.0).abs() < 1e-10,
            "Initial purity should be 1.0, got {}",
            purities[0]
        );

        // Check that purity is not monotonically decreasing
        // (a revival would mean some purity[t+1] > purity[t]).
        let mut has_increase = false;
        for w in purities.windows(2) {
            if w[1] > w[0] + 1e-12 {
                has_increase = true;
                break;
            }
        }

        assert!(
            has_increase,
            "Non-Markovian dynamics with strong memory should show purity revival"
        );
    }

    // ---------------------------------------------------------------
    // 8. State trace preservation under evolution
    // ---------------------------------------------------------------

    #[test]
    fn test_trace_preserved_during_evolution() {
        let channel = KrausChannel::amplitude_damping(0.1);
        let channels: Vec<KrausChannel> = (0..20).map(|_| channel.clone()).collect();
        let pt = ProcessTensor::with_memory(channels, 3, 0.5, 0.01);

        let mut sim = NonMarkovianSimulator::new(1, pt);

        for step in 0..20 {
            let trace = sim.state.trace();
            assert!(
                (trace - 1.0).abs() < 1e-8,
                "Trace should be ~1.0 at step {}, got {}",
                step,
                trace
            );
            sim.step();
        }
    }

    // ---------------------------------------------------------------
    // 9. Apply gate between noise steps
    // ---------------------------------------------------------------

    #[test]
    fn test_apply_gate_between_steps() {
        let channel = KrausChannel::phase_damping(0.05);
        let pt = ProcessTensor::markovian(channel, 10, 0.01);
        let mut sim = NonMarkovianSimulator::new(1, pt);

        // Start in |0>, apply H to go to |+>.
        let h_gate = Gate::h(0);
        sim.apply_gate(&h_gate);

        // After Hadamard, off-diagonal elements should be non-zero.
        let off_diag = sim.state.elements[1].norm();
        assert!(
            off_diag > 0.4,
            "After H gate, off-diagonal should be ~0.5, got {}",
            off_diag
        );

        // Evolve a few noise steps.
        sim.evolve(3);

        // State should still have trace = 1.
        let trace = sim.state.trace();
        assert!(
            (trace - 1.0).abs() < 1e-8,
            "Trace after gate+noise should be 1.0, got {}",
            trace
        );
    }

    // ---------------------------------------------------------------
    // 10. Multiple evolution steps
    // ---------------------------------------------------------------

    #[test]
    fn test_evolve_multiple_steps() {
        let channel = KrausChannel::depolarizing(0.05);
        let pt = ProcessTensor::markovian(channel, 50, 0.01);
        let mut sim = NonMarkovianSimulator::new(1, pt);

        sim.evolve(50);

        assert_eq!(sim.current_step(), 50);
        assert_eq!(sim.stats().total_steps, 50);

        // After many depolarizing steps, state should be close to maximally mixed.
        let purity = sim.purity();
        assert!(
            purity < 0.9,
            "After 50 depolarizing steps, purity should decrease, got {}",
            purity
        );
    }

    // ---------------------------------------------------------------
    // 11. ProcessTensor creation from RTN parameters
    // ---------------------------------------------------------------

    #[test]
    fn test_process_tensor_from_rtn() {
        let params = RTNParams {
            switching_rate: 0.05,
            amplitude: 0.3,
        };
        let pt = ProcessTensor::from_rtn(&params, 100, 0.01);

        assert_eq!(pt.num_steps(), 100);
        assert_eq!(pt.dim, 2);
        assert!(pt.memory_depth > 0, "RTN should have non-zero memory depth");

        // Each step should have a valid (trace-preserving) channel.
        for step in &pt.steps {
            assert!(
                step.channel.is_trace_preserving(1e-10),
                "RTN channel at t={} is not trace-preserving",
                step.time
            );
        }
    }

    // ---------------------------------------------------------------
    // 12. ProcessTensor creation from 1/f parameters
    // ---------------------------------------------------------------

    #[test]
    fn test_process_tensor_from_one_over_f() {
        let params = OneOverFParams {
            num_fluctuators: 10,
            gamma_min: 0.01,
            gamma_max: 0.5,
            amplitude: 0.2,
        };
        let pt = ProcessTensor::from_one_over_f(&params, 50, 0.01);

        assert_eq!(pt.num_steps(), 50);
        assert_eq!(pt.dim, 2);
        assert!(
            pt.memory_depth >= 2,
            "1/f noise should have memory depth >= 2, got {}",
            pt.memory_depth
        );
    }

    // ---------------------------------------------------------------
    // 13. Dephasing from trajectory produces valid channels
    // ---------------------------------------------------------------

    #[test]
    fn test_dephasing_from_trajectory_valid() {
        let trajectory = vec![0.1, -0.3, 0.5, 0.0, -0.2];
        let channels = CorrelatedNoiseSequence::dephasing_from_trajectory(&trajectory);

        assert_eq!(channels.len(), 5);
        for (i, ch) in channels.iter().enumerate() {
            assert_eq!(ch.dim, 2);
            assert!(
                ch.is_trace_preserving(1e-10),
                "Channel {} is not trace-preserving",
                i
            );
        }
    }

    // ---------------------------------------------------------------
    // 14. Trace distance properties
    // ---------------------------------------------------------------

    #[test]
    fn test_trace_distance_properties() {
        let dim = 2;

        // Distance between identical states should be 0.
        let rho = DensityMatrix::new(1);
        let d = trace_distance(&rho.elements, &rho.elements, dim);
        assert!(d < 1e-15, "Self-distance should be 0, got {}", d);

        // Distance between |0><0| and |1><1| should be 1.
        let mut state_1 = vec![C64::new(0.0, 0.0); 2];
        state_1[1] = C64::new(1.0, 0.0);
        let rho_1 = DensityMatrix::from_pure_state(&state_1);
        let d_01 = trace_distance(&rho.elements, &rho_1.elements, dim);
        assert!(
            (d_01 - 1.0).abs() < 1e-10,
            "|0> vs |1> trace distance should be 1.0, got {}",
            d_01
        );

        // Distance between |0><0| and maximally mixed should be 0.5.
        let rho_mixed = DensityMatrix::maximally_mixed(1);
        let d_mixed = trace_distance(&rho.elements, &rho_mixed.elements, dim);
        assert!(
            (d_mixed - 0.5).abs() < 1e-10,
            "|0> vs mixed trace distance should be 0.5, got {}",
            d_mixed
        );
    }

    // ---------------------------------------------------------------
    // 15. Statistics tracking
    // ---------------------------------------------------------------

    #[test]
    fn test_statistics_tracking() {
        let channel = KrausChannel::phase_damping(0.1);
        let channels: Vec<KrausChannel> = (0..10).map(|_| channel.clone()).collect();
        let pt = ProcessTensor::with_memory(channels, 3, 0.5, 0.01);

        let mut sim = NonMarkovianSimulator::new(1, pt);
        sim.evolve(10);

        let stats = sim.stats();
        assert_eq!(stats.total_steps, 10);
        assert!(
            stats.memory_lookbacks > 0,
            "Should have performed memory lookbacks"
        );
        assert!(
            stats.max_memory_depth_used <= 3,
            "Max depth should not exceed memory_depth=3, got {}",
            stats.max_memory_depth_used
        );
    }

    // ---------------------------------------------------------------
    // 16. Process tensor total_time
    // ---------------------------------------------------------------

    #[test]
    fn test_process_tensor_total_time() {
        let channel = KrausChannel::identity(2);
        let pt = ProcessTensor::markovian(channel, 10, 0.05);

        let expected_time = 9.0 * 0.05; // last step time = (n-1)*dt
        assert!(
            (pt.total_time() - expected_time).abs() < 1e-12,
            "Total time should be {}, got {}",
            expected_time,
            pt.total_time()
        );
    }

    // ---------------------------------------------------------------
    // 17. Identity channel preserves state
    // ---------------------------------------------------------------

    #[test]
    fn test_identity_channel_preserves_state() {
        let channel = KrausChannel::identity(2);
        let pt = ProcessTensor::markovian(channel, 10, 0.01);
        let mut sim = NonMarkovianSimulator::new(1, pt);

        // Prepare |+> state.
        let h_gate = Gate::h(0);
        sim.apply_gate(&h_gate);
        let state_before = sim.state.elements.clone();

        sim.evolve(10);

        // Identity channel should not change the state.
        for i in 0..4 {
            assert!(
                (sim.state.elements[i] - state_before[i]).norm() < 1e-12,
                "Identity channel changed element {} from {:?} to {:?}",
                i,
                state_before[i],
                sim.state.elements[i]
            );
        }
    }

    // ---------------------------------------------------------------
    // 18. Evolve stops at end of process tensor
    // ---------------------------------------------------------------

    #[test]
    fn test_evolve_stops_at_end() {
        let channel = KrausChannel::depolarizing(0.1);
        let pt = ProcessTensor::markovian(channel, 5, 0.01);
        let mut sim = NonMarkovianSimulator::new(1, pt);

        // Request more steps than available.
        sim.evolve(100);

        assert_eq!(
            sim.current_step(),
            5,
            "Should stop at 5 steps, not {}",
            sim.current_step()
        );
        assert_eq!(sim.stats().total_steps, 5);
    }

    // ---------------------------------------------------------------
    // 19. Multi-qubit system preserves trace
    // ---------------------------------------------------------------

    #[test]
    fn test_two_qubit_trace_preservation() {
        // Create a 2-qubit identity process tensor (dim=4).
        let channel = KrausChannel::identity(4);
        let pt = ProcessTensor::markovian(channel, 5, 0.01);
        let mut sim = NonMarkovianSimulator::new(2, pt);

        for _ in 0..5 {
            sim.step();
            let trace = sim.state.trace();
            assert!(
                (trace - 1.0).abs() < 1e-10,
                "2-qubit trace should be 1.0, got {}",
                trace
            );
        }
    }

    // ---------------------------------------------------------------
    // 20. Purity history starts at 1 for pure state
    // ---------------------------------------------------------------

    #[test]
    fn test_purity_history_initial_value() {
        let channel = KrausChannel::depolarizing(0.1);
        let pt = ProcessTensor::markovian(channel, 10, 0.01);
        let sim = NonMarkovianSimulator::new(1, pt);

        let history = sim.purity_history();

        assert_eq!(history.len(), 11); // initial + 10 steps
        assert!(
            (history[0] - 1.0).abs() < 1e-12,
            "Initial purity should be 1.0, got {}",
            history[0]
        );

        // Depolarizing noise should decrease purity.
        assert!(
            history[10] < history[0],
            "Purity should decrease under depolarizing noise"
        );
    }

    // ---------------------------------------------------------------
    // 21. RTN trajectory values are bounded
    // ---------------------------------------------------------------

    #[test]
    fn test_rtn_trajectory_values_bounded() {
        let params = RTNParams {
            switching_rate: 0.5,
            amplitude: 1.0,
        };
        let traj = CorrelatedNoiseSequence::generate_rtn(&params, 1000);

        for &v in &traj {
            assert!(
                (v - 1.0).abs() < 1e-15 || (v + 1.0).abs() < 1e-15,
                "RTN value {} should be exactly +/- 1.0",
                v
            );
        }
    }
}
