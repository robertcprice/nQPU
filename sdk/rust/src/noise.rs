//! Noise Simulation for Quantum Computing
//!
//! Implements various noise models to simulate realistic quantum hardware imperfections:
//! - Bit flip (X) noise: Spontaneous |0⟩ ↔ |1⟩ transitions
//! - Phase flip (Z) noise: Spontaneous phase flips
//! - Depolarizing noise: Random Pauli errors
//! - Amplitude damping (T1): Energy relaxation to |0⟩
//! - Phase damping (T2): Loss of phase coherence (dephasing)
//!
//! # Example
//! ```ignore
//! use nqpu_metal::{QuantumSimulator, noise::{NoiseModel, NoisySimulator, DepolarizingNoise}};
//!
//! let mut sim = NoisySimulator::new(
//!     QuantumSimulator::new(5),
//!     DepolarizingNoise::new(0.01)
//! );
//! sim.h(0);
//! sim.x(1);
//! let result = sim.measure();
//! ```

use crate::{QuantumSimulator, QuantumState};

// ============================================================
// NOISE MODEL TRAIT
// ============================================================

/// Base trait for all noise models
pub trait NoiseModel {
    /// Apply noise to a single-qubit state after a gate operation
    fn apply_single_qubit(&self, state: &mut QuantumState, qubit: usize);

    /// Apply noise to a two-qubit state after a gate operation
    fn apply_two_qubit(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize);

    /// Get the noise parameter (probability or rate)
    fn noise_parameter(&self) -> f64;

    /// Clone the noise model
    fn clone_box(&self) -> Box<dyn NoiseModel>;
}

// ============================================================
// SINGLE-QUBIT NOISE CHANNELS
// ============================================================

/// Bit flip (Pauli-X) noise channel
///
/// With probability p, applies an X gate: |0⟩ → |1⟩, |1⟩ → |0⟩
#[derive(Clone, Copy, Debug)]
pub struct BitFlipNoise {
    /// Probability of bit flip occurring
    pub p: f64,
}

impl BitFlipNoise {
    pub fn new(p: f64) -> Self {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Bit flip probability must be in [0, 1]"
        );
        BitFlipNoise { p }
    }
}

impl NoiseModel for BitFlipNoise {
    fn apply_single_qubit(&self, state: &mut QuantumState, qubit: usize) {
        if rand::random::<f64>() < self.p {
            // Apply bit flip (X gate)
            let stride = 1 << qubit;
            let dim = state.dim;
            let amplitudes = state.amplitudes_mut();

            for i in 0..dim / 2 {
                let j = i | stride;
                if j < dim {
                    amplitudes.swap(i, j);
                }
            }
        }
    }

    fn apply_two_qubit(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        // Apply noise to both qubits independently
        self.apply_single_qubit(state, qubit1);
        self.apply_single_qubit(state, qubit2);
    }

    fn noise_parameter(&self) -> f64 {
        self.p
    }

    fn clone_box(&self) -> Box<dyn NoiseModel> {
        Box::new(*self)
    }
}

/// Phase flip (Pauli-Z) noise channel
///
/// With probability p, applies a Z gate: |0⟩ → |0⟩, |1⟩ → -|1⟩
#[derive(Clone, Copy, Debug)]
pub struct PhaseFlipNoise {
    /// Probability of phase flip occurring
    pub p: f64,
}

impl PhaseFlipNoise {
    pub fn new(p: f64) -> Self {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Phase flip probability must be in [0, 1]"
        );
        PhaseFlipNoise { p }
    }
}

impl NoiseModel for PhaseFlipNoise {
    fn apply_single_qubit(&self, state: &mut QuantumState, qubit: usize) {
        if rand::random::<f64>() < self.p {
            // Apply phase flip (Z gate)
            let mask = 1 << qubit;
            let dim = state.dim;
            let amplitudes = state.amplitudes_mut();

            for i in 0..dim {
                if i & mask != 0 {
                    amplitudes[i].re = -amplitudes[i].re;
                    amplitudes[i].im = -amplitudes[i].im;
                }
            }
        }
    }

    fn apply_two_qubit(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        // Apply noise to both qubits independently
        self.apply_single_qubit(state, qubit1);
        self.apply_single_qubit(state, qubit2);
    }

    fn noise_parameter(&self) -> f64 {
        self.p
    }

    fn clone_box(&self) -> Box<dyn NoiseModel> {
        Box::new(*self)
    }
}

/// Depolarizing noise channel
///
/// With probability p, the state is replaced by the completely mixed state:
/// - Applies I (identity) with probability 1 - 3p/4
/// - Applies X (bit flip) with probability p/4
/// - Applies Y (bit+phase flip) with probability p/4
/// - Applies Z (phase flip) with probability p/4
#[derive(Clone, Copy, Debug)]
pub struct DepolarizingNoise {
    /// Depolarizing parameter (probability of error)
    pub p: f64,
}

impl DepolarizingNoise {
    pub fn new(p: f64) -> Self {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Depolarizing parameter must be in [0, 1]"
        );
        DepolarizingNoise { p }
    }
}

impl NoiseModel for DepolarizingNoise {
    fn apply_single_qubit(&self, state: &mut QuantumState, qubit: usize) {
        let r: f64 = rand::random();

        if r < self.p / 4.0 {
            // Apply Y gate (bit + phase flip)
            let mask = 1 << qubit;
            let dim = state.dim;
            let amplitudes = state.amplitudes_mut();

            for i in 0..dim {
                if i & mask != 0 {
                    let orig = amplitudes[i];
                    // Y|1⟩ = i|0⟩, Y|0⟩ = -i|1⟩, but for state vector:
                    // Effectively swaps with phase: (a, b) → (-bi, ai)
                    amplitudes[i].re = -orig.im;
                    amplitudes[i].im = orig.re;
                } else {
                    let orig = amplitudes[i];
                    amplitudes[i].re = orig.im;
                    amplitudes[i].im = -orig.re;
                }
            }
        } else if r < self.p / 2.0 {
            // Apply X gate (bit flip)
            let stride = 1 << qubit;
            let dim = state.dim;
            let amplitudes = state.amplitudes_mut();

            for i in 0..dim / 2 {
                let j = i | stride;
                if j < dim {
                    amplitudes.swap(i, j);
                }
            }
        } else if r < 3.0 * self.p / 4.0 {
            // Apply Z gate (phase flip)
            let mask = 1 << qubit;
            let dim = state.dim;
            let amplitudes = state.amplitudes_mut();

            for i in 0..dim {
                if i & mask != 0 {
                    amplitudes[i].re = -amplitudes[i].re;
                    amplitudes[i].im = -amplitudes[i].im;
                }
            }
        }
        // Otherwise: Apply I (identity) - do nothing
    }

    fn apply_two_qubit(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        // Apply noise to both qubits independently
        self.apply_single_qubit(state, qubit1);
        self.apply_single_qubit(state, qubit2);
    }

    fn noise_parameter(&self) -> f64 {
        self.p
    }

    fn clone_box(&self) -> Box<dyn NoiseModel> {
        Box::new(*self)
    }
}

/// Amplitude damping (T1 relaxation) noise channel
///
/// Models energy relaxation: the qubit decays from |1⟩ to |0⟩ with probability γ
/// This is the T1 relaxation process in real quantum hardware
#[derive(Clone, Copy, Debug)]
pub struct AmplitudeDamping {
    /// Damping probability (related to T1 time)
    pub gamma: f64,
}

impl AmplitudeDamping {
    pub fn new(gamma: f64) -> Self {
        assert!(
            gamma >= 0.0 && gamma <= 1.0,
            "Damping probability must be in [0, 1]"
        );
        AmplitudeDamping { gamma }
    }

    /// Create from T1 time and gate time
    /// gamma = 1 - exp(-gate_time / T1)
    pub fn from_t1(t1: f64, gate_time: f64) -> Self {
        let gamma = 1.0 - (-gate_time / t1).exp();
        AmplitudeDamping { gamma }
    }
}

impl NoiseModel for AmplitudeDamping {
    fn apply_single_qubit(&self, state: &mut QuantumState, qubit: usize) {
        let sqrt_gamma = self.gamma.sqrt();
        let sqrt_1_minus_gamma = (1.0 - self.gamma).sqrt();

        let stride = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // For each basis state, apply amplitude damping
        // |0⟩ → |0⟩, |1⟩ → sqrt(1-γ)|1⟩ + sqrt(γ)|0⟩
        for i in 0..dim {
            if i & stride != 0 {
                // Qubit is in |1⟩, apply damping
                let orig = amplitudes[i];

                // Decay from |1⟩ to |0⟩ component
                let i_zero = i & !stride; // State with qubit in |0⟩

                // Add the damped component to |0⟩ state
                amplitudes[i_zero].re += sqrt_gamma * orig.re;
                amplitudes[i_zero].im += sqrt_gamma * orig.im;

                // Scale the remaining |1⟩ component
                amplitudes[i].re = sqrt_1_minus_gamma * orig.re;
                amplitudes[i].im = sqrt_1_minus_gamma * orig.im;
            }
        }
    }

    fn apply_two_qubit(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        // Apply noise to both qubits independently
        self.apply_single_qubit(state, qubit1);
        self.apply_single_qubit(state, qubit2);
    }

    fn noise_parameter(&self) -> f64 {
        self.gamma
    }

    fn clone_box(&self) -> Box<dyn NoiseModel> {
        Box::new(*self)
    }
}

/// Phase damping (T2 dephasing) noise channel
///
/// Models loss of quantum coherence without energy loss
/// This is the T2 dephasing process in real quantum hardware
#[derive(Clone, Copy, Debug)]
pub struct PhaseDamping {
    /// Phase damping probability (related to T2 time)
    pub lambda: f64,
}

impl PhaseDamping {
    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda >= 0.0 && lambda <= 1.0,
            "Phase damping probability must be in [0, 1]"
        );
        PhaseDamping { lambda }
    }

    /// Create from T2 time and gate time
    /// lambda = 1 - exp(-2 * gate_time / T2)
    pub fn from_t2(t2: f64, gate_time: f64) -> Self {
        let lambda = 1.0 - (-2.0 * gate_time / t2).exp();
        PhaseDamping { lambda }
    }
}

impl NoiseModel for PhaseDamping {
    fn apply_single_qubit(&self, state: &mut QuantumState, qubit: usize) {
        // With probability lambda, apply a phase flip to |1⟩ state
        if rand::random::<f64>() < self.lambda {
            let mask = 1 << qubit;
            let dim = state.dim;
            let amplitudes = state.amplitudes_mut();

            for i in 0..dim {
                if i & mask != 0 {
                    // Decay off-diagonal elements (coherence terms)
                    // This approximates the phase damping channel
                    amplitudes[i].re *= (1.0 - self.lambda).sqrt();
                    amplitudes[i].im *= (1.0 - self.lambda).sqrt();
                }
            }
        }
    }

    fn apply_two_qubit(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        // Apply noise to both qubits independently
        self.apply_single_qubit(state, qubit1);
        self.apply_single_qubit(state, qubit2);
    }

    fn noise_parameter(&self) -> f64 {
        self.lambda
    }

    fn clone_box(&self) -> Box<dyn NoiseModel> {
        Box::new(*self)
    }
}

// ============================================================
// COMBINED NOISE MODELS
// ============================================================

/// Combined noise model that applies multiple noise channels
pub struct CombinedNoise {
    models: Vec<Box<dyn NoiseModel>>,
}

impl Clone for CombinedNoise {
    fn clone(&self) -> Self {
        CombinedNoise {
            models: self.models.iter().map(|m| m.clone_box()).collect(),
        }
    }
}

impl std::fmt::Debug for CombinedNoise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedNoise")
            .field("num_models", &self.models.len())
            .finish()
    }
}

impl CombinedNoise {
    pub fn new() -> Self {
        CombinedNoise { models: Vec::new() }
    }

    pub fn add<M: NoiseModel + 'static>(mut self, noise: M) -> Self {
        self.models.push(Box::new(noise));
        self
    }
}

impl Default for CombinedNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseModel for CombinedNoise {
    fn apply_single_qubit(&self, state: &mut QuantumState, qubit: usize) {
        for model in &self.models {
            model.apply_single_qubit(state, qubit);
        }
    }

    fn apply_two_qubit(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        for model in &self.models {
            model.apply_two_qubit(state, qubit1, qubit2);
        }
    }

    fn noise_parameter(&self) -> f64 {
        // Return average noise parameter
        if self.models.is_empty() {
            return 0.0;
        }
        self.models.iter().map(|m| m.noise_parameter()).sum::<f64>() / self.models.len() as f64
    }

    fn clone_box(&self) -> Box<dyn NoiseModel> {
        Box::new(self.clone())
    }
}

// ============================================================
// NOISY SIMULATOR WRAPPER
// ============================================================

/// Wrapper around QuantumSimulator that applies noise after each gate operation
pub struct NoisySimulator {
    /// Base simulator (noise-free operations)
    pub simulator: QuantumSimulator,
    /// Noise model to apply after each gate
    pub noise: Box<dyn NoiseModel>,
    /// Track number of gates applied
    gate_count: usize,
}

impl Clone for NoisySimulator {
    fn clone(&self) -> Self {
        NoisySimulator {
            simulator: QuantumSimulator::new(self.simulator.num_qubits()),
            noise: self.noise.clone_box(),
            gate_count: 0, // Reset gate count on clone
        }
    }
}

impl std::fmt::Debug for NoisySimulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoisySimulator")
            .field("num_qubits", &self.simulator.num_qubits())
            .field("gate_count", &self.gate_count)
            .finish()
    }
}

impl NoisySimulator {
    /// Create a new noisy simulator
    ///
    /// # Arguments
    /// * `simulator` - Base quantum simulator
    /// * `noise` - Noise model to apply
    pub fn new<M: NoiseModel + 'static>(simulator: QuantumSimulator, noise: M) -> Self {
        NoisySimulator {
            simulator,
            noise: Box::new(noise),
            gate_count: 0,
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.simulator.num_qubits()
    }

    /// Reset the simulator to |0...0⟩ state
    pub fn reset(&mut self) {
        self.simulator.reset();
        self.gate_count = 0;
    }

    /// Get the gate count
    pub fn gate_count(&self) -> usize {
        self.gate_count
    }

    // Gate wrappers with noise application

    pub fn h(&mut self, qubit: usize) {
        self.simulator.h(qubit);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    pub fn x(&mut self, qubit: usize) {
        self.simulator.x(qubit);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    pub fn z(&mut self, qubit: usize) {
        self.simulator.z(qubit);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    pub fn ry(&mut self, qubit: usize, theta: f64) {
        self.simulator.ry(qubit, theta);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    pub fn cnot(&mut self, control: usize, target: usize) {
        self.simulator.cnot(control, target);
        self.noise
            .apply_two_qubit(&mut self.simulator.state, control, target);
        self.gate_count += 1;
    }

    pub fn cz(&mut self, control: usize, target: usize) {
        self.simulator.cz(control, target);
        self.noise
            .apply_two_qubit(&mut self.simulator.state, control, target);
        self.gate_count += 1;
    }

    pub fn cphase(&mut self, control: usize, target: usize, phi: f64) {
        self.simulator.cphase(control, target, phi);
        self.noise
            .apply_two_qubit(&mut self.simulator.state, control, target);
        self.gate_count += 1;
    }

    // Additional gates with noise

    /// Pauli-Y gate (NOT + phase): |0⟩ → i|1⟩, |1⟩ → -i|0⟩
    pub fn y(&mut self, qubit: usize) {
        self.simulator.y(qubit);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    /// S gate (phase gate): S = [[1, 0], [0, i]]
    pub fn s(&mut self, qubit: usize) {
        self.simulator.s(qubit);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    /// T gate (π/8 gate): T = [[1, 0], [0, exp(iπ/4)]]
    pub fn t(&mut self, qubit: usize) {
        self.simulator.t(qubit);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    /// Rotation around X-axis: Rx(θ) = exp(-iθX/2)
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        self.simulator.rx(qubit, theta);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    /// Rotation around Z-axis: Rz(θ) = exp(-iθZ/2)
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        self.simulator.rz(qubit, theta);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    /// SWAP gate: Swap two qubits
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        self.simulator.swap(qubit1, qubit2);
        self.noise
            .apply_two_qubit(&mut self.simulator.state, qubit1, qubit2);
        self.gate_count += 1;
    }

    /// Toffoli gate (CCX): Three-qubit controlled-X
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        self.simulator.toffoli(control1, control2, target);
        // Apply noise to all three qubits
        self.noise
            .apply_two_qubit(&mut self.simulator.state, control1, control2);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, target);
        self.gate_count += 1;
    }

    /// Mid-circuit measurement: Measure a single qubit without collapsing the entire state
    pub fn measure_qubit(&mut self, qubit: usize) -> (usize, crate::QuantumState) {
        let result = self.simulator.measure_qubit(qubit);
        self.gate_count += 1;
        result
    }

    /// Expectation value of Z operator for a qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        self.simulator.expectation_z(qubit)
    }

    /// Fidelity between this state and another
    pub fn fidelity(&self, other: &crate::QuantumState) -> f64 {
        self.simulator.fidelity(other)
    }

    /// Initialize state from arbitrary amplitudes
    pub fn initialize_from_amplitudes(&mut self, amplitudes: Vec<crate::C64>) -> bool {
        self.simulator.initialize_from_amplitudes(amplitudes)
    }

    /// Measure the quantum state
    pub fn measure(&self) -> usize {
        self.simulator.measure()
    }

    /// Get the probabilities of each basis state
    pub fn probabilities(&self) -> Vec<f64> {
        self.simulator.state.probabilities()
    }

    /// Get a reference to the quantum state
    pub fn state(&self) -> &QuantumState {
        &self.simulator.state
    }

    // Additional gate wrappers with noise

    /// Controlled-RX gate: Apply Rx(θ) to target if control is |1⟩
    pub fn crx(&mut self, control: usize, target: usize, theta: f64) {
        self.simulator.crx(control, target, theta);
        self.noise
            .apply_two_qubit(&mut self.simulator.state, control, target);
        self.gate_count += 1;
    }

    /// Controlled-RY gate: Apply Ry(θ) to target if control is |1⟩
    pub fn cry(&mut self, control: usize, target: usize, theta: f64) {
        self.simulator.cry(control, target, theta);
        self.noise
            .apply_two_qubit(&mut self.simulator.state, control, target);
        self.gate_count += 1;
    }

    /// Controlled-RZ gate: Apply Rz(θ) to target if control is |1⟩
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) {
        self.simulator.crz(control, target, theta);
        self.noise
            .apply_two_qubit(&mut self.simulator.state, control, target);
        self.gate_count += 1;
    }

    /// General single-qubit unitary gate
    pub fn u(&mut self, qubit: usize, matrix: &[[crate::C64; 2]; 2]) {
        self.simulator.u(qubit, matrix);
        self.noise
            .apply_single_qubit(&mut self.simulator.state, qubit);
        self.gate_count += 1;
    }

    /// Reset a qubit to |0⟩
    pub fn reset_qubit(&mut self, qubit: usize) {
        self.simulator.reset_qubit(qubit);
        self.gate_count += 1;
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Create a typical superconducting qubit noise model
/// Combines T1 relaxation, T2 dephasing, and depolarizing errors
pub fn superconducting_noise(t1: f64, t2: f64, gate_time: f64, depol_prob: f64) -> CombinedNoise {
    CombinedNoise::new()
        .add(AmplitudeDamping::from_t1(t1, gate_time))
        .add(PhaseDamping::from_t2(t2, gate_time))
        .add(DepolarizingNoise::new(depol_prob))
}

/// Create a typical trapped ion qubit noise model
/// Trapped ions typically have longer coherence times but different error characteristics
pub fn trapped_ion_noise(t1: f64, t2: f64, gate_time: f64, depol_prob: f64) -> CombinedNoise {
    CombinedNoise::new()
        .add(AmplitudeDamping::from_t1(t1, gate_time))
        .add(PhaseDamping::from_t2(t2, gate_time))
        .add(DepolarizingNoise::new(depol_prob))
}

/// Run Grover's search with noise
/// Returns (result, probability_of_correct_result)
pub fn grover_with_noise<M: NoiseModel + 'static>(
    num_qubits: usize,
    target: usize,
    num_iterations: usize,
    noise: M,
) -> (usize, f64) {
    let mut sim = NoisySimulator::new(QuantumSimulator::new(num_qubits), noise);

    // Initialize uniform superposition
    for i in 0..num_qubits {
        sim.h(i);
    }

    // Grover iterations
    for _ in 0..num_iterations {
        // Oracle
        let amplitudes = sim.simulator.state.amplitudes_mut();
        amplitudes[target].re = -amplitudes[target].re;
        amplitudes[target].im = -amplitudes[target].im;

        // Diffusion
        for i in 0..num_qubits {
            sim.h(i);
        }

        let amplitudes = sim.simulator.state.amplitudes_mut();
        amplitudes[0].re = -amplitudes[0].re;
        amplitudes[0].im = -amplitudes[0].im;

        for i in 0..num_qubits {
            sim.h(i);
        }
    }

    let probs = sim.probabilities();
    let correct_prob = probs[target];

    let result = sim.measure();

    (result, correct_prob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_flip_noise() {
        let mut state = QuantumState::new(2);
        // state is already |00⟩ by default

        let noise = BitFlipNoise::new(1.0); // Always flip
        noise.apply_single_qubit(&mut state, 0);

        // After X on qubit 0: |01⟩ (qubit 0 is LSB)
        let probs = state.probabilities();
        assert!((probs[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_noisy_simulator() {
        let mut sim = NoisySimulator::new(
            QuantumSimulator::new(2),
            BitFlipNoise::new(0.0), // No noise
        );

        sim.h(0);
        let probs = sim.probabilities();
        let total_prob: f64 = probs.iter().sum();
        assert!((total_prob - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_combined_noise() {
        let noise = CombinedNoise::new()
            .add(BitFlipNoise::new(0.1))
            .add(PhaseFlipNoise::new(0.1));

        assert!((noise.noise_parameter() - 0.1).abs() < 0.001);
    }
}
