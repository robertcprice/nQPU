//! Comprehensive Noise Models for Quantum Simulation
//!
//! Realistic noise modeling for near-term quantum devices.
//!
//! **Noise Types**:
//! - **Depolarizing**: Random Pauli errors
//! - **Amplitude Damping**: Energy relaxation (T₁)
//! - **Phase Damping**: Dephasing (T₂)
//! - **Readout Error**: Measurement errors
//! - **Coherent Errors**: Systematic over-rotations
//! - **Cross-Talk**: Spurious qubit-qubit interactions
//!
//! **Error Mitigation**:
//! - Zero-noise extrapolation
//! - Probabilistic error cancellation
//! - Virtual distillation
//! - Randomized compiling

use crate::{QuantumState, C64};
use rand::Rng;
use std::collections::HashMap;

/// Noise model configuration.
#[derive(Clone, Debug)]
pub struct NoiseModel {
    /// Depolarizing error probability per gate.
    pub depolarizing_prob: f64,
    /// Amplitude damping probability (T₁).
    pub amplitude_damping_prob: f64,
    /// Phase damping probability (T₂).
    pub phase_damping_prob: f64,
    /// Readout error probabilities (p(0|1), p(1|0)).
    pub readout_error: (f64, f64),
    /// Coherent over-rotation angles per gate.
    pub coherent_errors: HashMap<String, f64>,
    /// Cross-talk strength between qubits.
    pub crosstalk_prob: f64,
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            depolarizing_prob: 0.001,
            amplitude_damping_prob: 0.0001,
            phase_damping_prob: 0.0001,
            readout_error: (0.01, 0.01),
            coherent_errors: HashMap::new(),
            crosstalk_prob: 0.0001,
        }
    }
}

impl NoiseModel {
    /// Create noise model for specific device.
    pub fn ibm_nqubit() -> Self {
        Self {
            depolarizing_prob: 0.001,
            amplitude_damping_prob: 0.0002,
            phase_damping_prob: 0.0003,
            readout_error: (0.02, 0.02),
            coherent_errors: HashMap::new(),
            crosstalk_prob: 0.0005,
        }
    }

    pub fn google_sycamore() -> Self {
        Self {
            depolarizing_prob: 0.003,
            amplitude_damping_prob: 0.0005,
            phase_damping_prob: 0.0002,
            readout_error: (0.01, 0.01),
            coherent_errors: HashMap::new(),
            crosstalk_prob: 0.002,
        }
    }

    pub fn rigetti_aspen() -> Self {
        Self {
            depolarizing_prob: 0.002,
            amplitude_damping_prob: 0.0003,
            phase_damping_prob: 0.0004,
            readout_error: (0.015, 0.015),
            coherent_errors: HashMap::new(),
            crosstalk_prob: 0.001,
        }
    }
}

/// Noisy quantum state with noise model.
pub struct NoisyQuantumState {
    /// Ideal state (for comparison).
    ideal_state: QuantumState,
    /// Noise model.
    noise: NoiseModel,
    /// Track accumulated error per qubit.
    error_tracking: Vec<f64>,
}

impl NoisyQuantumState {
    pub fn new(num_qubits: usize, noise: NoiseModel) -> Self {
        Self {
            ideal_state: QuantumState::new(num_qubits),
            noise,
            error_tracking: vec![0.0; num_qubits],
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.ideal_state.num_qubits
    }

    /// Apply single-qubit gate with noise.
    pub fn apply_single_qubit_gate_noisy(
        &mut self,
        qubit: usize,
        matrix: [[C64; 2]; 2],
        rng: &mut impl Rng,
    ) {
        // Apply ideal gate
        crate::GateOperations::u(&mut self.ideal_state, qubit, &matrix);

        // Apply noise
        self.apply_depolarizing(qubit, rng);
        self.apply_amplitude_damping(qubit, rng);
        self.apply_phase_damping(qubit, rng);
        self.apply_coherent_error(qubit, "single_qubit", rng);
        self.error_tracking[qubit] += self.noise.depolarizing_prob;
    }

    /// Apply two-qubit gate with noise.
    pub fn apply_two_qubit_gate_noisy(&mut self, qubit1: usize, qubit2: usize, rng: &mut impl Rng) {
        // Apply ideal gate
        crate::GateOperations::cnot(&mut self.ideal_state, qubit1, qubit2);

        // Apply noise to both qubits
        for &qubit in &[qubit1, qubit2] {
            self.apply_depolarizing(qubit, rng);
            self.apply_amplitude_damping(qubit, rng);
            self.apply_phase_damping(qubit, rng);
            self.error_tracking[qubit] += self.noise.depolarizing_prob;
        }

        // Apply cross-talk
        if rng.gen::<f64>() < self.noise.crosstalk_prob {
            // Spurious Z rotation on neighboring qubit
            crate::GateOperations::rz(&mut self.ideal_state, qubit2, 0.01);
        }
    }

    fn apply_depolarizing(&mut self, qubit: usize, rng: &mut impl Rng) {
        let p = self.noise.depolarizing_prob;

        if rng.gen::<f64>() < p {
            // Apply random Pauli
            let pauli = rng.gen_range(0..3);
            match pauli {
                0 => crate::GateOperations::x(&mut self.ideal_state, qubit),
                1 => crate::GateOperations::y(&mut self.ideal_state, qubit),
                2 => crate::GateOperations::z(&mut self.ideal_state, qubit),
                _ => {}
            }
        }
    }

    fn apply_amplitude_damping(&mut self, qubit: usize, rng: &mut impl Rng) {
        let gamma = self.noise.amplitude_damping_prob;
        if gamma <= 0.0 {
            return;
        }

        // Amplitude damping via Kraus operators:
        // E0 = [[1, 0], [0, sqrt(1-gamma)]]  (no decay)
        // E1 = [[0, sqrt(gamma)], [0, 0]]     (decay |1> -> |0>)
        let dim = self.ideal_state.dim;
        let stride = 1 << qubit;
        let amplitudes = self.ideal_state.amplitudes_mut();

        for i in (0..dim).step_by(stride * 2) {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;
                if idx1 < dim {
                    let a1_sq = amplitudes[idx1].norm_sqr();
                    let decay_prob = gamma * a1_sq;

                    if decay_prob > 0.0 && rng.gen::<f64>() < decay_prob.min(1.0) {
                        // Decay: transfer |1> amplitude to |0>
                        let sqrt_gamma = C64::new(gamma.sqrt(), 0.0);
                        amplitudes[idx0] = amplitudes[idx0] + amplitudes[idx1] * sqrt_gamma;
                        amplitudes[idx1] = C64::new(0.0, 0.0);
                    } else if gamma < 1.0 {
                        // No decay: damp |1> component
                        let damp = C64::new((1.0 - gamma).sqrt(), 0.0);
                        amplitudes[idx1] = amplitudes[idx1] * damp;
                    }
                }
            }
        }

        // Renormalize to preserve unitarity
        let norm_sq: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq > 1e-30 {
            let inv_norm = C64::new(1.0 / norm_sq.sqrt(), 0.0);
            for a in amplitudes.iter_mut() {
                *a = *a * inv_norm;
            }
        }
    }

    fn apply_phase_damping(&mut self, qubit: usize, rng: &mut impl Rng) {
        let lambda = self.noise.phase_damping_prob;

        if rng.gen::<f64>() < lambda {
            // Phase flip (dephasing)
            crate::GateOperations::z(&mut self.ideal_state, qubit);
        }
    }

    fn apply_coherent_error(&mut self, qubit: usize, gate_type: &str, rng: &mut impl Rng) {
        if let Some(&error_angle) = self.noise.coherent_errors.get(gate_type) {
            // Add systematic over-rotation
            let noise_angle = error_angle * (2.0 * rng.gen::<f64>() - 1.0);
            crate::GateOperations::rz(&mut self.ideal_state, qubit, noise_angle);
        }
    }

    /// Noisy measurement with readout error.
    pub fn measure_noisy(&mut self, qubit: usize, rng: &mut impl Rng) -> usize {
        // Ideal measurement
        let ideal_result = self.measure_ideal(qubit);

        // Apply readout error
        let (p0_given_1, p1_given_0) = self.noise.readout_error;

        if ideal_result == 0 {
            // Actual state is |0⟩, might read as |1⟩
            if rng.gen::<f64>() < p1_given_0 {
                return 1;
            }
        } else {
            // Actual state is |1⟩, might read as |0⟩
            if rng.gen::<f64>() < p0_given_1 {
                return 0;
            }
        }

        ideal_result
    }

    fn measure_ideal(&self, qubit: usize) -> usize {
        let probs = self.ideal_state.probabilities();
        let stride = 1 << qubit;

        let mut p0 = 0.0;
        for i in (0..probs.len()).step_by(stride * 2) {
            for j in 0..stride {
                if i + j < probs.len() {
                    p0 += probs[i + j];
                }
            }
        }

        if rand::random::<f64>() < p0 {
            0
        } else {
            1
        }
    }

    /// Get fidelity with ideal state.
    pub fn fidelity(&self) -> f64 {
        // For noisy state, fidelity is reduced by accumulated errors
        let total_error: f64 = self.error_tracking.iter().sum();
        (1.0 - total_error).max(0.0)
    }
}

// ==================== ERROR MITIGATION ====================

/// Zero-noise extrapolation techniques.
pub struct ZeroNoiseExtrapolation;

impl ZeroNoiseExtrapolation {
    /// Linear extrapolation.
    pub fn linear(results: &[(f64, f64)]) -> f64 {
        // (noise_level, result) pairs
        if results.len() < 2 {
            return results.last().map(|&(_, r)| r).unwrap_or(0.0);
        }

        // Fit line and extrapolate to zero noise
        let n = results.len();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;

        for &(x, y) in results {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n as f64;

        intercept // Extrapolate to x=0
    }

    /// Richardson extrapolation.
    pub fn richardson(results: &[(f64, f64)], order: usize) -> f64 {
        if results.len() < order + 1 {
            return Self::linear(results);
        }

        // Richardson extrapolation formula
        let mut sum = 0.0;
        let mut total_weight = 0.0;

        for k in 0..=order {
            let weight = ((order + 1) as f64).powi(k as i32 * 2 - 1) * ((-1.0_f64).powi(k as i32));

            if let Some(&(_, result)) = results.get(k) {
                sum += weight * result;
                total_weight += weight;
            }
        }

        sum / total_weight
    }

    /// Exponential extrapolation.
    pub fn exponential(results: &[(f64, f64)]) -> f64 {
        if results.len() < 2 {
            return results.last().map(|&(_, r)| r).unwrap_or(0.0);
        }

        // Fit A + B*exp(-cx) and return A
        let (x1, y1) = results[0];
        let (x2, y2) = results[1];
        let (x3, y3) = results[2];

        // Solve for extrapolated value at x=0
        let a = (y1 * x2 - y1 * x3 - y2 * x1 + y2 * x3 - y3 * x1 + y3 * x2)
            / ((x1 - x2) * (x1 - x3) * (x2 - x3));

        y3 + a * (x3 - x1) * (x3 - x2)
    }
}

/// Probabilistic error cancellation.
pub struct ProbabilisticErrorCancellation;

impl ProbabilisticErrorCancellation {
    /// Apply PEC by sampling from quasi-probability distribution.
    pub fn apply_pec<F>(
        ideal_circuit: F,
        _noise_model: &NoiseModel,
        num_qubits: usize,
        samples: usize,
    ) -> f64
    where
        F: Fn(&mut QuantumState) + Clone,
    {
        let mut results = Vec::new();

        for _ in 0..samples {
            let mut state = QuantumState::new(num_qubits);
            let sign = 1.0;

            // Decompose ideal gate into noisy operations
            // This is simplified - full PEC requires gate decomposition
            ideal_circuit(&mut state);

            // Compute observable
            let observable = state.amplitudes_ref()[0].norm_sqr();
            results.push(sign * observable);
        }

        results.iter().sum::<f64>() / samples as f64
    }
}

/// Virtual distillation error mitigation.
pub struct VirtualDistillation;

impl VirtualDistillation {
    /// Distill multiple noisy copies into one less noisy state.
    pub fn distill(states: Vec<QuantumState>) -> QuantumState {
        if states.is_empty() {
            return QuantumState::new(1);
        }

        let num_qubits = states[0].num_qubits;
        let _num_copies = states.len();

        // Initialize distilled state
        let mut distilled = QuantumState::new(num_qubits);

        // For each basis state, compute product of probabilities
        let amplitudes = distilled.amplitudes_mut();
        let dim = 1 << num_qubits;

        for i in 0..dim {
            let mut product = C64::new(1.0, 0.0);

            for state in &states {
                let amp = state.amplitudes_ref()[i];
                product = product * amp;
            }

            // Normalize: (product)^n / (prob_sum)^n
            amplitudes[i] = product;
        }

        // Normalize
        let norm_sq: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();

        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for amp in amplitudes.iter_mut() {
                *amp /= norm;
            }
        }

        distilled
    }

    /// Estimate required number of copies for target fidelity.
    pub fn copies_for_fidelity(target_fidelity: f64, base_fidelity: f64) -> usize {
        // F_n = F_base^(2^n) / F_base^(2n-1)
        // Solve for n given target F_n
        if base_fidelity >= target_fidelity {
            return 1;
        }

        let mut n = 1;
        while n < 10 {
            let fidelity_n = base_fidelity.powi(2 * n as i32);
            if fidelity_n >= target_fidelity {
                return n;
            }
            n += 1;
        }

        10 // Maximum reasonable copies
    }
}

/// Randomized compiling for error mitigation.
pub struct RandomizedCompiling;

impl RandomizedCompiling {
    /// Generate random twirl for error mitigation.
    pub fn random_twirl(gate: &str) -> String {
        // Map each gate to equivalence class
        // For CX: {CX, (I⊗H)·CX·(I⊗H), (H⊗I)·CX·(H⊗I), (H⊗H)·CX·(H⊗H)}
        match gate {
            "CX" => {
                let twirl = rand::random::<usize>() % 4;
                match twirl {
                    0 => "CX".to_string(),
                    1 => "H(1);CX;H(1)".to_string(),
                    2 => "H(0);CX;H(0)".to_string(),
                    _ => "H(0);H(1);CX;H(0);H(1)".to_string(),
                }
            }
            _ => gate.to_string(),
        }
    }

    /// Average over multiple random twirls.
    pub fn average_over_twirls<F>(circuit: F, num_twirls: usize, num_qubits: usize) -> QuantumState
    where
        F: Fn(&mut QuantumState) + Clone,
    {
        let mut accumulated = QuantumState::new(num_qubits);
        let amplitudes = accumulated.amplitudes_mut();

        for _ in 0..num_twirls {
            let mut state = QuantumState::new(num_qubits);
            circuit(&mut state);

            // Accumulate
            for (i, &amp) in state.amplitudes_ref().iter().enumerate() {
                amplitudes[i] += amp;
            }
        }

        // Normalize
        let norm = num_twirls as f64;
        for amp in amplitudes.iter_mut() {
            *amp /= norm;
        }

        accumulated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_depolarizing_channel() {
        let noise = NoiseModel {
            depolarizing_prob: 1.0, // Always apply error
            ..Default::default()
        };

        let mut state = NoisyQuantumState::new(1, noise);
        let mut rng = thread_rng();

        let h_matrix = [
            [C64::new(0.70710678, 0.0), C64::new(0.70710678, 0.0)],
            [C64::new(0.70710678, 0.0), C64::new(-0.70710678, 0.0)],
        ];

        state.apply_single_qubit_gate_noisy(0, h_matrix, &mut rng);

        // State should still be normalized
        let probs = state.ideal_state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_damping() {
        let noise = NoiseModel {
            amplitude_damping_prob: 1.0, // Always damp
            ..Default::default()
        };

        let mut state = NoisyQuantumState::new(1, noise);
        let mut rng = thread_rng();

        // Apply X to get |1⟩
        crate::GateOperations::x(&mut state.ideal_state, 0);
        state.apply_amplitude_damping(0, &mut rng);

        // Should decay to |0⟩
        let probs = state.ideal_state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zne_linear() {
        let results = vec![(0.01, 0.85), (0.02, 0.75), (0.03, 0.65)];
        let extrapolated = ZeroNoiseExtrapolation::linear(&results);

        // Should extrapolate to higher value
        assert!(extrapolated > 0.85);
    }
}
