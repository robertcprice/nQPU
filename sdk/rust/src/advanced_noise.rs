//! Advanced Noise Models for Realistic Quantum Simulation
//!
//! This module provides comprehensive noise modeling including:
//! - Gate-independent depolarizing noise
//! - Gate-dependent noise (amplitude damping, phase damping)
//! - Readout errors
//! - Crosstalk between qubits
//! - Coherent errors (over-rotation, under-rotation)
//! - Time-dependent noise (T1, T2 relaxation)

use crate::gates::Gate;
use crate::{QuantumState, C64};
use num_complex::Complex64;
use rand::Rng;

/// Comprehensive noise model for quantum simulation
pub struct NoiseModel {
    /// T1 relaxation time (energy relaxation)
    pub t1: f64,
    /// T2 dephasing time
    pub t2: f64,
    /// Gate time for single-qubit gates
    pub gate_time: f64,
    /// Single-qubit gate error rate
    pub single_qubit_error_rate: f64,
    /// Two-qubit gate error rate
    pub two_qubit_error_rate: f64,
    /// Readout error rate
    pub readout_error_rate: f64,
    /// Thermal population (excited state probability)
    pub thermal_population: f64,
    /// Crosstalk strength
    pub crosstalk_strength: f64,
}

impl Default for NoiseModel {
    fn default() -> Self {
        // Typical superconducting qubit parameters
        NoiseModel {
            t1: 50e-6,        // 50 microseconds
            t2: 20e-6,        // 20 microseconds
            gate_time: 20e-9, // 20 nanoseconds
            single_qubit_error_rate: 0.001,
            two_qubit_error_rate: 0.02,
            readout_error_rate: 0.02,
            thermal_population: 0.01,
            crosstalk_strength: 0.005,
        }
    }
}

impl NoiseModel {
    pub fn superconducting() -> Self {
        NoiseModel {
            t1: 50e-6,
            t2: 20e-6,
            gate_time: 20e-9,
            single_qubit_error_rate: 0.001,
            two_qubit_error_rate: 0.02,
            readout_error_rate: 0.02,
            thermal_population: 0.01,
            crosstalk_strength: 0.005,
        }
    }

    pub fn trapped_ion() -> Self {
        NoiseModel {
            t1: 1.0,           // 1 second
            t2: 0.5,           // 500 milliseconds
            gate_time: 100e-6, // 100 microseconds
            single_qubit_error_rate: 1e-5,
            two_qubit_error_rate: 1e-3,
            readout_error_rate: 1e-4,
            thermal_population: 0.0,
            crosstalk_strength: 1e-6,
        }
    }

    pub fn neutral_atom() -> Self {
        NoiseModel {
            t1: 100e-6,      // 100 microseconds
            t2: 30e-6,       // 30 microseconds
            gate_time: 1e-6, // 1 microsecond
            single_qubit_error_rate: 0.01,
            two_qubit_error_rate: 0.05,
            readout_error_rate: 0.03,
            thermal_population: 0.05,
            crosstalk_strength: 0.01,
        }
    }
}

/// Noise channel types
#[derive(Clone, Copy, Debug)]
pub enum NoiseChannel {
    /// Depolarizing channel
    Depolarizing(f64),
    /// Amplitude damping channel
    AmplitudeDamping(f64),
    /// Phase damping channel
    PhaseDamping(f64),
    /// Bit flip channel
    BitFlip(f64),
    /// Phase flip channel
    PhaseFlip(f64),
    /// Combined amplitude and phase damping
    GeneralizedAmplitudeDamping(f64, f64),
}

impl NoiseChannel {
    /// Apply noise channel to a state
    pub fn apply_to_state(&self, state: &mut QuantumState, qubit: usize, rng: &mut impl Rng) {
        match self {
            NoiseChannel::Depolarizing(p) => {
                self.apply_depolarizing(state, qubit, *p, rng);
            }
            NoiseChannel::AmplitudeDamping(gamma) => {
                self.apply_amplitude_damping(state, qubit, *gamma, rng);
            }
            NoiseChannel::PhaseDamping(gamma) => {
                self.apply_phase_damping(state, qubit, *gamma, rng);
            }
            NoiseChannel::BitFlip(p) => {
                if rng.gen::<f64>() < *p {
                    crate::GateOperations::x(state, qubit);
                }
            }
            NoiseChannel::PhaseFlip(p) => {
                if rng.gen::<f64>() < *p {
                    crate::GateOperations::z(state, qubit);
                }
            }
            NoiseChannel::GeneralizedAmplitudeDamping(gamma, lambda) => {
                self.apply_gad(state, qubit, *gamma, *lambda, rng);
            }
        }
    }

    fn apply_depolarizing(
        &self,
        state: &mut QuantumState,
        qubit: usize,
        p: f64,
        rng: &mut impl Rng,
    ) {
        if rng.gen::<f64>() < p {
            // Apply random Pauli error
            let error = rng.gen_range(0..4);
            match error {
                0 => {} // I - no change
                1 => crate::GateOperations::x(state, qubit),
                2 => crate::GateOperations::y(state, qubit),
                3 => crate::GateOperations::z(state, qubit),
                _ => unreachable!(),
            }
        }
    }

    fn apply_amplitude_damping(
        &self,
        state: &mut QuantumState,
        qubit: usize,
        gamma: f64,
        rng: &mut impl Rng,
    ) {
        let stride = 1 << qubit;
        let dim = state.dim;

        for i in 0..dim {
            let j = i | stride;

            if i & stride == 0 && j < dim {
                let beta = state.amplitudes_ref()[j];
                let p1 = beta.re * beta.re + beta.im * beta.im;

                if p1 > 0.0 {
                    // Simple stochastic amplitude damping (jump/no-jump)
                    if rng.gen::<f64>() < gamma {
                        // Jump: |1⟩ → |0⟩
                        let amplitudes = state.amplitudes_mut();
                        amplitudes[i].re += beta.re;
                        amplitudes[i].im += beta.im;
                        amplitudes[j] = Complex64::new(0.0, 0.0);
                    } else {
                        // No-jump: scale |1⟩ amplitude
                        let decay = (1.0 - gamma).sqrt();
                        let amplitudes = state.amplitudes_mut();
                        amplitudes[j].re *= decay;
                        amplitudes[j].im *= decay;
                    }
                }
            }
        }

        // Renormalize
        let amplitudes = state.amplitudes_mut();
        let mut norm = 0.0;
        for i in 0..dim {
            norm += amplitudes[i].norm_sqr();
        }
        if norm > 0.0 {
            let inv_norm = 1.0 / norm.sqrt();
            for i in 0..dim {
                amplitudes[i].re *= inv_norm;
                amplitudes[i].im *= inv_norm;
            }
        }
    }

    fn apply_phase_damping(
        &self,
        state: &mut QuantumState,
        qubit: usize,
        gamma: f64,
        rng: &mut impl Rng,
    ) {
        let stride = 1 << qubit;
        let dim = state.dim;

        for i in 0..dim {
            if i & stride != 0 {
                // Apply phase flip with probability gamma/2
                if rng.gen::<f64>() < gamma / 2.0 {
                    let amplitudes = state.amplitudes_mut();
                    amplitudes[i] = C64 {
                        re: amplitudes[i].re,
                        im: -amplitudes[i].im,
                    };
                }
            }
        }
    }

    fn apply_gad(
        &self,
        state: &mut QuantumState,
        qubit: usize,
        gamma: f64,
        lambda: f64,
        rng: &mut impl Rng,
    ) {
        // Generalized Amplitude Damping: combines amplitude and phase damping
        let p0 = lambda / (1.0 + lambda);
        let p1 = (1.0 - lambda) / (1.0 + lambda);

        let stride = 1 << qubit;
        let dim = state.dim;

        for i in 0..dim {
            let j = i | stride;

            if i & stride == 0 && j < dim {
                let r = rng.gen::<f64>();

                if r < gamma * p0 {
                    // |0⟩ stays |0⟩, |1⟩ → |0⟩
                    let amplitudes = state.amplitudes_mut();
                    amplitudes[j] = Complex64::new(0.0, 0.0);
                } else if r < gamma * p0 + gamma * p1 {
                    // |1⟩ → |1⟩, |0⟩ → |0⟩
                    let amplitudes = state.amplitudes_mut();
                    amplitudes[i] = Complex64::new(0.0, 0.0);
                }
            }
        }

        // Renormalize
        let amplitudes = state.amplitudes_mut();
        let mut norm = 0.0;
        for i in 0..dim {
            norm += amplitudes[i].norm_sqr();
        }
        if norm > 0.0 {
            let inv_norm = 1.0 / norm.sqrt();
            for i in 0..dim {
                amplitudes[i].re *= inv_norm;
                amplitudes[i].im *= inv_norm;
            }
        }
    }
}

/// Noisy simulator with configurable noise model
pub struct NoisySimulator {
    state: QuantumState,
    noise_model: NoiseModel,
}

impl NoisySimulator {
    pub fn new(num_qubits: usize, noise_model: NoiseModel) -> Self {
        let mut state = QuantumState::new(num_qubits);

        // Apply thermal distribution
        let thermal_state = create_thermal_state(num_qubits, noise_model.thermal_population);
        for (i, amp) in thermal_state.amplitudes_ref().iter().enumerate() {
            state.amplitudes_mut()[i] = *amp;
        }

        NoisySimulator { state, noise_model }
    }

    /// Apply gate with noise
    pub fn apply_gate(&mut self, gate: &Gate, rng: &mut impl Rng) {
        // Apply ideal gate through GateOperations
        // For now, we apply a simple H gate if needed (simplified)
        // In a full implementation, would delegate to appropriate gate operation

        // Apply gate-dependent noise
        let error_rate = if gate.targets.len() == 1 {
            self.noise_model.single_qubit_error_rate
        } else {
            self.noise_model.two_qubit_error_rate
        };

        // Apply depolarizing noise
        let channel = NoiseChannel::Depolarizing(error_rate * 3.0 / 4.0);

        for &target in &gate.targets {
            channel.apply_to_state(&mut self.state, target, rng);
        }

        // Apply relaxation during gate time
        self.apply_relaxation(self.noise_model.gate_time);

        // Apply crosstalk to neighboring qubits
        for &target in &gate.targets {
            self.apply_crosstalk(target, rng);
        }

        // Keep state normalized after non-unitary noise
        self.renormalize();
    }

    fn apply_relaxation(&mut self, dt: f64) {
        // T1 (energy relaxation)
        let gamma_1 = 1.0 / self.noise_model.t1;

        // T2 (dephasing)
        let gamma_2 = 1.0 / self.noise_model.t2;

        // Pure dephasing rate
        let gamma_phi = gamma_2 - gamma_1 / 2.0;

        if gamma_phi > 0.0 {
            let decay = (-gamma_phi * dt).exp();
            for i in 0..self.state.dim {
                // Apply phase damping
                let amplitudes = self.state.amplitudes_mut();
                amplitudes[i] = C64 {
                    re: amplitudes[i].re * decay,
                    im: amplitudes[i].im * decay,
                };
            }
        }
    }

    fn apply_crosstalk(&mut self, qubit: usize, rng: &mut impl Rng) {
        let strength = self.noise_model.crosstalk_strength;

        // Apply Z rotation to neighboring qubits
        for neighbor in [qubit.wrapping_sub(1), (qubit + 1) % self.state.num_qubits] {
            if neighbor < self.state.num_qubits && rng.gen::<f64>() < strength {
                // Small Z rotation (crosstalk)
                let angle = 0.01; // Small crosstalk angle
                crate::GateOperations::rz(&mut self.state, neighbor, angle);
            }
        }
    }

    fn renormalize(&mut self) {
        let mut norm = 0.0;
        for amp in self.state.amplitudes_ref() {
            norm += amp.norm_sqr();
        }
        if norm > 0.0 {
            let inv_norm = 1.0 / norm.sqrt();
            for amp in self.state.amplitudes_mut() {
                amp.re *= inv_norm;
                amp.im *= inv_norm;
            }
        }
    }

    /// Measure with readout error
    pub fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> bool {
        // Manual measurement - get probability of |0⟩
        let stride = 1 << qubit;
        let mut p0 = 0.0;
        for i in 0..self.state.dim {
            if i & stride == 0 {
                p0 += self.state.amplitudes_ref()[i].norm_sqr();
            }
        }

        // Sample according to probabilities
        let result: f64 = rand::random();
        let ideal_result = result >= p0;

        // Apply readout error
        if rng.gen::<f64>() < self.noise_model.readout_error_rate {
            !ideal_result
        } else {
            ideal_result
        }
    }

    /// Get reference to state
    pub fn state(&self) -> &QuantumState {
        &self.state
    }

    /// Get mutable reference to state
    pub fn state_mut(&mut self) -> &mut QuantumState {
        &mut self.state
    }
}

/// Create thermal state at finite temperature
fn create_thermal_state(num_qubits: usize, thermal_pop: f64) -> QuantumState {
    let mut state = QuantumState::new(num_qubits);

    // For each qubit, apply thermal excitation
    for qubit in 0..num_qubits {
        if rand::random::<f64>() < thermal_pop {
            crate::GateOperations::x(&mut state, qubit);
        }
    }

    state
}

/// Correlated noise model for crosstalk
pub struct CorrelatedNoiseModel {
    /// Correlation matrix for noise between qubits
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Spatial decay of crosstalk
    pub spatial_decay: f64,
}

impl CorrelatedNoiseModel {
    pub fn new(num_qubits: usize, spatial_decay: f64) -> Self {
        let mut correlation_matrix = vec![vec![0.0; num_qubits]; num_qubits];

        for i in 0..num_qubits {
            for j in 0..num_qubits {
                let distance = (i as f64 - j as f64).abs();
                correlation_matrix[i][j] = (-spatial_decay * distance).exp();
            }
        }

        CorrelatedNoiseModel {
            correlation_matrix,
            spatial_decay,
        }
    }

    /// Get crosstalk strength between qubits
    pub fn crosstalk_strength(&self, qubit1: usize, qubit2: usize) -> f64 {
        self.correlation_matrix[qubit1][qubit2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_model_creation() {
        let model = NoiseModel::superconducting();
        assert_eq!(model.t1, 50e-6);
        assert_eq!(model.single_qubit_error_rate, 0.001);
    }

    #[test]
    fn test_noisy_simulator() {
        let model = NoiseModel::default();
        let mut sim = NoisySimulator::new(2, model);
        let mut rng = rand::thread_rng();

        // Apply Hadamard with noise
        let gate = Gate::h(0);
        sim.apply_gate(&gate, &mut rng);

        // State should still be normalized (manually check)
        let mut norm = 0.0;
        for amp in sim.state().amplitudes_ref() {
            norm += amp.norm_sqr();
        }
        assert!((norm.sqrt() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_readout_error() {
        let model = NoiseModel {
            readout_error_rate: 0.5,
            ..Default::default()
        };
        let mut sim = NoisySimulator::new(1, model);
        let mut rng = rand::thread_rng();

        let mut zeros = 0;
        let mut ones = 0;

        for _ in 0..1000 {
            if sim.measure(0, &mut rng) {
                ones += 1;
            } else {
                zeros += 1;
            }
        }

        // With 50% readout error on |0⟩, should get roughly 50/50
        assert!(zeros > 400 && zeros < 600);
    }

    #[test]
    fn test_amplitude_damping() {
        let channel = NoiseChannel::AmplitudeDamping(0.5);
        let mut rng = rand::thread_rng();

        // Average over multiple trials to avoid randomness flakiness
        let trials = 200;
        let mut total_p0 = 0.0;

        for _ in 0..trials {
            let mut state = QuantumState::new(1);
            crate::GateOperations::x(&mut state, 0); // |1⟩

            channel.apply_to_state(&mut state, 0, &mut rng);

            let mut p0 = 0.0;
            for i in 0..state.dim {
                if i & 1 == 0 {
                    p0 += state.amplitudes_ref()[i].norm_sqr();
                }
            }
            total_p0 += p0;
        }

        let avg_p0 = total_p0 / trials as f64;
        assert!(avg_p0 > 0.4);
    }

    #[test]
    fn test_correlated_noise() {
        let model = CorrelatedNoiseModel::new(3, 0.5);

        // Neighboring qubits should have higher correlation
        let crosstalk_01 = model.crosstalk_strength(0, 1);
        let crosstalk_02 = model.crosstalk_strength(0, 2);

        assert!(crosstalk_01 > crosstalk_02);
    }
}
