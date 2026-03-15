//! Digital-Analog Quantum Computation (DAQC) Simulation
//!
//! Implements hybrid gate + Hamiltonian evolution circuits as used in
//! near-term hardware platforms (Google, Qilimanjaro, IQM). DAQC interleaves
//! discrete digital gates with continuous analog evolution under a device
//! Hamiltonian, reducing gate overhead and leveraging always-on couplings.
//!
//! # Overview
//!
//! A [`DAQCCircuit`] is an ordered sequence of [`DAQCSegment`]s, where each
//! segment is either:
//!
//! - **Digital**: a block of standard quantum gates, or
//! - **Analog**: time evolution under a [`LocalHamiltonian1D`] for a specified
//!   duration, compiled via first-order Trotter decomposition into gates.
//!
//! The [`DAQCSimulator`] executes these circuits on a [`StateVectorBackend`]
//! and can optionally apply an O(n) diagonal shortcut for ZZ-only
//! Hamiltonians.
//!
//! # Hardware Presets
//!
//! [`HardwareHamiltonians`] provides factory methods for common hardware
//! Hamiltonians:
//!
//! - **Transmon**: nearest-neighbor ZZ coupling with transverse X field
//! - **Rydberg**: nearest-neighbor ZZ + global X drive + detuning Z
//! - **Trapped ion**: all-to-all ZZ coupling with transverse X field
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::digital_analog::*;
//! use nqpu_metal::gates::{Gate, GateType};
//!
//! let n = 4;
//! let mut circuit = DAQCCircuit::new(n);
//!
//! // Digital layer: Hadamard on all qubits
//! for q in 0..n {
//!     circuit.add_gate(Gate::single(GateType::H, q));
//! }
//!
//! // Analog layer: transmon evolution for 1.0 time units
//! let ham = HardwareHamiltonians::transmon(n, 1.0);
//! circuit.add_analog(AnalogBlock::new(ham, 1.0, 10));
//!
//! // Simulate
//! let sim = DAQCSimulator::new(DAQCConfig::new());
//! let probs = sim.simulate(&circuit).unwrap();
//! assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-6);
//! ```

use crate::gates::{Gate, GateType};
use crate::time_evolution::LocalHamiltonian1D;
use crate::traits::{QuantumBackend, StateVectorBackend};
use std::fmt;

// ===================================================================
// ERROR TYPE
// ===================================================================

/// Errors that can occur during DAQC simulation.
#[derive(Debug, Clone)]
pub enum DAQCError {
    /// A duration or time parameter was invalid (negative, NaN, etc.).
    InvalidDuration(String),
    /// The underlying simulation backend reported a failure.
    SimulationFailed(String),
    /// The circuit contains no segments to simulate.
    EmptyCircuit,
}

impl fmt::Display for DAQCError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DAQCError::InvalidDuration(msg) => write!(f, "invalid duration: {}", msg),
            DAQCError::SimulationFailed(msg) => write!(f, "simulation failed: {}", msg),
            DAQCError::EmptyCircuit => write!(f, "circuit has no segments"),
        }
    }
}

impl std::error::Error for DAQCError {}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Configuration for the DAQC simulator.
///
/// Controls Trotter step density, bond dimension limits, and whether the
/// diagonal ZZ shortcut is enabled.
#[derive(Debug, Clone)]
pub struct DAQCConfig {
    /// Maximum bond dimension for MPS-based backends (unused in state-vector
    /// mode but kept for forward compatibility).
    pub max_bond_dim: usize,
    /// Number of Trotter steps per unit of evolution time. The actual step
    /// count for a block of duration `t` is `ceil(t * trotter_steps_per_unit)`.
    pub trotter_steps_per_unit: usize,
    /// When true, ZZ-only Hamiltonians use an O(n) diagonal gate
    /// decomposition instead of the general Trotter circuit.
    pub optimize_diagonal: bool,
}

impl DAQCConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self {
            max_bond_dim: 64,
            trotter_steps_per_unit: 10,
            optimize_diagonal: true,
        }
    }

    /// Set the maximum bond dimension.
    pub fn max_bond_dim(mut self, d: usize) -> Self {
        self.max_bond_dim = d;
        self
    }

    /// Set the Trotter step density.
    pub fn trotter_steps_per_unit(mut self, s: usize) -> Self {
        self.trotter_steps_per_unit = s;
        self
    }

    /// Enable or disable the diagonal ZZ shortcut.
    pub fn optimize_diagonal(mut self, flag: bool) -> Self {
        self.optimize_diagonal = flag;
        self
    }
}

impl Default for DAQCConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// ANALOG BLOCK
// ===================================================================

/// A continuous-time evolution block under a local Hamiltonian.
///
/// Represents the analog portion of a DAQC circuit: the system evolves
/// under `hamiltonian` for `duration` time units, approximated by
/// `trotter_steps` Trotter layers.
#[derive(Debug, Clone)]
pub struct AnalogBlock {
    /// The Hamiltonian driving the evolution.
    pub hamiltonian: LocalHamiltonian1D,
    /// Total evolution time (must be non-negative).
    pub duration: f64,
    /// Number of first-order Trotter steps.
    pub trotter_steps: usize,
}

impl AnalogBlock {
    /// Create a new analog block.
    pub fn new(hamiltonian: LocalHamiltonian1D, duration: f64, trotter_steps: usize) -> Self {
        Self {
            hamiltonian,
            duration,
            trotter_steps,
        }
    }

    /// Returns `true` when the Hamiltonian contains only ZZ couplings
    /// (no single-qubit X or Z fields). ZZ-only Hamiltonians are diagonal
    /// in the computational basis and admit an efficient O(n) simulation.
    pub fn is_zz_only(&self) -> bool {
        self.hamiltonian.x.is_empty() && self.hamiltonian.z.is_empty()
    }
}

// ===================================================================
// CIRCUIT SEGMENTS AND BUILDER
// ===================================================================

/// A single segment of a DAQC circuit: either digital gates or analog
/// evolution.
#[derive(Debug, Clone)]
pub enum DAQCSegment {
    /// A block of standard quantum gates.
    Digital(Vec<Gate>),
    /// A continuous-time Hamiltonian evolution block.
    Analog(AnalogBlock),
}

/// A hybrid digital-analog quantum circuit.
///
/// Maintains an ordered sequence of [`DAQCSegment`]s. Consecutive
/// `add_gate` calls are merged into a single `Digital` segment;
/// `add_analog` always creates a new `Analog` segment.
#[derive(Debug, Clone)]
pub struct DAQCCircuit {
    /// The ordered segments of this circuit.
    pub segments: Vec<DAQCSegment>,
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
}

impl DAQCCircuit {
    /// Create an empty circuit for `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            segments: Vec::new(),
            num_qubits,
        }
    }

    /// Append a gate. If the last segment is `Digital`, the gate is added
    /// to it; otherwise a new `Digital` segment is created.
    pub fn add_gate(&mut self, gate: Gate) {
        match self.segments.last_mut() {
            Some(DAQCSegment::Digital(gates)) => {
                gates.push(gate);
            }
            _ => {
                self.segments.push(DAQCSegment::Digital(vec![gate]));
            }
        }
    }

    /// Append an analog evolution block. Always creates a new `Analog`
    /// segment (analog blocks are never merged).
    pub fn add_analog(&mut self, block: AnalogBlock) {
        self.segments.push(DAQCSegment::Analog(block));
    }

    /// Return the number of segments in this circuit.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Return the total number of digital gates across all segments.
    pub fn total_gates(&self) -> usize {
        self.segments
            .iter()
            .map(|seg| match seg {
                DAQCSegment::Digital(gates) => gates.len(),
                DAQCSegment::Analog(_) => 0,
            })
            .sum()
    }

    /// Return the total analog evolution time across all segments.
    pub fn total_analog_time(&self) -> f64 {
        self.segments
            .iter()
            .map(|seg| match seg {
                DAQCSegment::Analog(block) => block.duration,
                DAQCSegment::Digital(_) => 0.0,
            })
            .sum()
    }
}

// ===================================================================
// HARDWARE HAMILTONIAN PRESETS
// ===================================================================

/// Factory methods for Hamiltonians modelling common quantum hardware.
pub struct HardwareHamiltonians;

impl HardwareHamiltonians {
    /// Superconducting transmon array: nearest-neighbor ZZ coupling with a
    /// weak transverse X field (hx = 0.1 * j_coupling per qubit).
    pub fn transmon(n: usize, j_coupling: f64) -> LocalHamiltonian1D {
        let mut h = LocalHamiltonian1D::default();
        for i in 0..n.saturating_sub(1) {
            h.zz.push((i, i + 1, j_coupling));
        }
        let hx = 0.1 * j_coupling;
        for i in 0..n {
            h.x.push((i, hx));
        }
        h
    }

    /// Rydberg atom array: nearest-neighbor ZZ interaction at strength
    /// `omega`, global X drive at `omega`, and Z detuning at `-delta`.
    pub fn rydberg(n: usize, omega: f64, delta: f64) -> LocalHamiltonian1D {
        let mut h = LocalHamiltonian1D::default();
        for i in 0..n.saturating_sub(1) {
            h.zz.push((i, i + 1, omega));
        }
        for i in 0..n {
            h.x.push((i, omega));
            h.z.push((i, -delta));
        }
        h
    }

    /// Trapped-ion chain: all-to-all ZZ coupling at `j_coupling` between
    /// every pair, with a transverse X field (hx = 0.1 * j_coupling).
    pub fn trapped_ion(n: usize, j_coupling: f64) -> LocalHamiltonian1D {
        let mut h = LocalHamiltonian1D::default();
        for i in 0..n {
            for j in (i + 1)..n {
                h.zz.push((i, j, j_coupling));
            }
        }
        let hx = 0.1 * j_coupling;
        for i in 0..n {
            h.x.push((i, hx));
        }
        h
    }
}

// ===================================================================
// FIDELITY COMPARISON
// ===================================================================

/// Result of comparing DAQC simulation with a high-accuracy digital
/// reference.
#[derive(Debug, Clone)]
pub struct FidelityComparison {
    /// Bhattacharyya fidelity between the DAQC and reference probability
    /// distributions: F = (sum sqrt(p_i * q_i))^2. Equals 1.0 for
    /// identical distributions.
    pub fidelity: f64,
    /// Total number of digital gates in the DAQC circuit.
    pub digital_gate_count: usize,
    /// Total analog evolution time in the DAQC circuit.
    pub analog_time: f64,
    /// Total number of segments (digital + analog) in the circuit.
    pub total_segments: usize,
}

// ===================================================================
// SIMULATOR
// ===================================================================

/// Executes DAQC circuits on a state-vector backend.
pub struct DAQCSimulator {
    config: DAQCConfig,
}

impl DAQCSimulator {
    /// Create a simulator with the given configuration.
    pub fn new(config: DAQCConfig) -> Self {
        Self { config }
    }

    /// Simulate the circuit and return the probability distribution over
    /// computational basis states (length 2^n).
    pub fn simulate(&self, circuit: &DAQCCircuit) -> Result<Vec<f64>, DAQCError> {
        if circuit.segments.is_empty() {
            return Err(DAQCError::EmptyCircuit);
        }

        let mut backend = StateVectorBackend::new(circuit.num_qubits);

        for segment in &circuit.segments {
            match segment {
                DAQCSegment::Digital(gates) => {
                    backend
                        .apply_circuit(gates)
                        .map_err(|e| DAQCError::SimulationFailed(e.to_string()))?;
                }
                DAQCSegment::Analog(block) => {
                    self.apply_analog_block(&mut backend, block)?;
                }
            }
        }

        backend
            .probabilities()
            .map_err(|e| DAQCError::SimulationFailed(e.to_string()))
    }

    /// Compare the DAQC simulation against a high-accuracy digital
    /// reference. The reference Trotterizes every analog block at 10x
    /// the step count.
    pub fn compare_fidelity(&self, circuit: &DAQCCircuit) -> Result<FidelityComparison, DAQCError> {
        if circuit.segments.is_empty() {
            return Err(DAQCError::EmptyCircuit);
        }

        // Run the circuit at normal resolution.
        let probs = self.simulate(circuit)?;

        // Build a high-accuracy reference config (10x Trotter steps, no
        // diagonal shortcut so the reference always uses the general path).
        let ref_config = DAQCConfig {
            max_bond_dim: self.config.max_bond_dim,
            trotter_steps_per_unit: self.config.trotter_steps_per_unit,
            optimize_diagonal: false,
        };
        let ref_sim = DAQCSimulator::new(ref_config);

        // Build a reference circuit with 10x Trotter steps on every analog
        // block.
        let mut ref_circuit = DAQCCircuit::new(circuit.num_qubits);
        for segment in &circuit.segments {
            match segment {
                DAQCSegment::Digital(gates) => {
                    for gate in gates {
                        ref_circuit.add_gate(gate.clone());
                    }
                }
                DAQCSegment::Analog(block) => {
                    ref_circuit.add_analog(AnalogBlock::new(
                        block.hamiltonian.clone(),
                        block.duration,
                        block.trotter_steps * 10,
                    ));
                }
            }
        }

        let ref_probs = ref_sim.simulate(&ref_circuit)?;

        // Bhattacharyya fidelity: F = (sum sqrt(p_i * q_i))^2
        let bc: f64 = probs
            .iter()
            .zip(ref_probs.iter())
            .map(|(p, q)| (p * q).sqrt())
            .sum();
        let fidelity = bc * bc;

        Ok(FidelityComparison {
            fidelity,
            digital_gate_count: circuit.total_gates(),
            analog_time: circuit.total_analog_time(),
            total_segments: circuit.num_segments(),
        })
    }

    // ---------------------------------------------------------------
    // Internal: apply a single analog block to the backend
    // ---------------------------------------------------------------

    fn apply_analog_block(
        &self,
        backend: &mut StateVectorBackend,
        block: &AnalogBlock,
    ) -> Result<(), DAQCError> {
        if block.duration < 0.0 {
            return Err(DAQCError::InvalidDuration(format!(
                "negative duration: {}",
                block.duration
            )));
        }
        if block.duration == 0.0 {
            return Ok(());
        }
        if block.trotter_steps == 0 {
            return Err(DAQCError::InvalidDuration(
                "trotter_steps must be > 0".to_string(),
            ));
        }

        if self.config.optimize_diagonal && block.is_zz_only() {
            self.apply_zz_diagonal(backend, block)
        } else {
            self.apply_trotter(backend, block)
        }
    }

    /// ZZ-only diagonal shortcut.
    ///
    /// For a Hamiltonian H = sum_{i,j} J_{ij} Z_i Z_j the time evolution
    /// exp(-i H t) is diagonal in the computational basis. We decompose
    /// each ZZ term into CNOT-Rz-CNOT:
    ///
    ///   exp(-i J dt Z_i Z_j) = CNOT(i,j) . Rz(2 J dt, j) . CNOT(i,j)
    ///
    /// This is exact (no Trotter error) so we apply it once for the full
    /// duration rather than per-step.
    fn apply_zz_diagonal(
        &self,
        backend: &mut StateVectorBackend,
        block: &AnalogBlock,
    ) -> Result<(), DAQCError> {
        let dt = block.duration;
        for &(i, j, coeff) in &block.hamiltonian.zz {
            let gates = vec![
                Gate::two(GateType::CNOT, i, j),
                Gate::single(GateType::Rz(2.0 * coeff * dt), j),
                Gate::two(GateType::CNOT, i, j),
            ];
            backend
                .apply_circuit(&gates)
                .map_err(|e| DAQCError::SimulationFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// General first-order Trotter decomposition.
    ///
    /// For each Trotter step with dt = duration / trotter_steps:
    ///   1. Apply X fields:  exp(-i hx dt X_i) = Rx(2 hx dt, i)
    ///   2. Apply Z fields:  exp(-i hz dt Z_i) = Rz(2 hz dt, i)
    ///   3. Apply ZZ terms:  CNOT(i,j) . Rz(2 J dt, j) . CNOT(i,j)
    fn apply_trotter(
        &self,
        backend: &mut StateVectorBackend,
        block: &AnalogBlock,
    ) -> Result<(), DAQCError> {
        let dt = block.duration / block.trotter_steps as f64;

        for _step in 0..block.trotter_steps {
            // Single-qubit X fields
            for &(qubit, coeff) in &block.hamiltonian.x {
                backend
                    .apply_gate(&Gate::single(GateType::Rx(2.0 * coeff * dt), qubit))
                    .map_err(|e| DAQCError::SimulationFailed(e.to_string()))?;
            }

            // Single-qubit Z fields
            for &(qubit, coeff) in &block.hamiltonian.z {
                backend
                    .apply_gate(&Gate::single(GateType::Rz(2.0 * coeff * dt), qubit))
                    .map_err(|e| DAQCError::SimulationFailed(e.to_string()))?;
            }

            // ZZ coupling terms
            for &(i, j, coeff) in &block.hamiltonian.zz {
                let gates = vec![
                    Gate::two(GateType::CNOT, i, j),
                    Gate::single(GateType::Rz(2.0 * coeff * dt), j),
                    Gate::two(GateType::CNOT, i, j),
                ];
                backend
                    .apply_circuit(&gates)
                    .map_err(|e| DAQCError::SimulationFailed(e.to_string()))?;
            }
        }
        Ok(())
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // AnalogBlock
    // ---------------------------------------------------------------

    #[test]
    fn test_analog_block_creation() {
        let mut h = LocalHamiltonian1D::default();
        h.zz.push((0, 1, 1.0));
        let block = AnalogBlock::new(h.clone(), 2.5, 20);
        assert_eq!(block.duration, 2.5);
        assert_eq!(block.trotter_steps, 20);
        assert_eq!(block.hamiltonian.zz.len(), 1);
    }

    #[test]
    fn test_zz_only_detection() {
        // ZZ-only: should be true
        let mut h_zz = LocalHamiltonian1D::default();
        h_zz.zz.push((0, 1, 1.0));
        let block_zz = AnalogBlock::new(h_zz, 1.0, 5);
        assert!(block_zz.is_zz_only());

        // Has X field: should be false
        let mut h_x = LocalHamiltonian1D::default();
        h_x.zz.push((0, 1, 1.0));
        h_x.x.push((0, 0.5));
        let block_x = AnalogBlock::new(h_x, 1.0, 5);
        assert!(!block_x.is_zz_only());

        // Has Z field: should be false
        let mut h_z = LocalHamiltonian1D::default();
        h_z.zz.push((0, 1, 1.0));
        h_z.z.push((0, 0.3));
        let block_z = AnalogBlock::new(h_z, 1.0, 5);
        assert!(!block_z.is_zz_only());
    }

    // ---------------------------------------------------------------
    // DAQCCircuit builder
    // ---------------------------------------------------------------

    #[test]
    fn test_circuit_builder() {
        let mut circuit = DAQCCircuit::new(4);
        circuit.add_gate(Gate::single(GateType::H, 0));
        circuit.add_gate(Gate::single(GateType::X, 1));

        let mut h = LocalHamiltonian1D::default();
        h.zz.push((0, 1, 1.0));
        circuit.add_analog(AnalogBlock::new(h, 1.0, 10));

        circuit.add_gate(Gate::single(GateType::Z, 2));

        assert_eq!(circuit.num_segments(), 3);
        assert_eq!(circuit.total_gates(), 3);
        assert_eq!(circuit.total_analog_time(), 1.0);
    }

    #[test]
    fn test_segment_merging() {
        let mut circuit = DAQCCircuit::new(2);
        circuit.add_gate(Gate::single(GateType::H, 0));
        circuit.add_gate(Gate::single(GateType::X, 1));
        circuit.add_gate(Gate::single(GateType::Y, 0));

        // All three gates should be merged into a single Digital segment.
        assert_eq!(circuit.num_segments(), 1);
        assert_eq!(circuit.total_gates(), 3);

        match &circuit.segments[0] {
            DAQCSegment::Digital(gates) => assert_eq!(gates.len(), 3),
            _ => panic!("expected Digital segment"),
        }
    }

    // ---------------------------------------------------------------
    // Hardware Hamiltonians
    // ---------------------------------------------------------------

    #[test]
    fn test_hardware_transmon() {
        let h = HardwareHamiltonians::transmon(4, 1.0);
        // 3 nearest-neighbor ZZ couplings
        assert_eq!(h.zz.len(), 3);
        // 4 X fields (one per qubit)
        assert_eq!(h.x.len(), 4);
        // No Z fields
        assert!(h.z.is_empty());
        // Check X field strength
        for &(_, hx) in &h.x {
            assert!((hx - 0.1).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hardware_rydberg() {
        let h = HardwareHamiltonians::rydberg(4, 2.0, 0.5);
        // 3 nearest-neighbor ZZ couplings
        assert_eq!(h.zz.len(), 3);
        // 4 X fields and 4 Z fields
        assert_eq!(h.x.len(), 4);
        assert_eq!(h.z.len(), 4);
        // Z fields should be -delta = -0.5
        for &(_, hz) in &h.z {
            assert!((hz - (-0.5)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hardware_trapped_ion() {
        let n = 5;
        let h = HardwareHamiltonians::trapped_ion(n, 1.0);
        // All-to-all: n*(n-1)/2 = 10 ZZ couplings
        assert_eq!(h.zz.len(), n * (n - 1) / 2);
        // X fields on every qubit
        assert_eq!(h.x.len(), n);
    }

    // ---------------------------------------------------------------
    // Simulation: ZZ diagonal path
    // ---------------------------------------------------------------

    #[test]
    fn test_zz_diagonal_identity() {
        // ZZ coupling with coefficient 0 should act as identity.
        let mut h = LocalHamiltonian1D::default();
        h.zz.push((0, 1, 0.0));
        let block = AnalogBlock::new(h, 1.0, 5);

        let mut circuit = DAQCCircuit::new(2);
        circuit.add_analog(block);

        let sim = DAQCSimulator::new(DAQCConfig::new());
        let probs = sim.simulate(&circuit).unwrap();

        // Starting state |00> should remain |00>.
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!(probs[3].abs() < 1e-10);
    }

    #[test]
    fn test_zz_single_coupling() {
        // 2-qubit ZZ with J=pi/4. Starting from |00> the ZZ operator is
        // diagonal so the state remains |00> with an accumulated global
        // phase. Probabilities should be unchanged.
        let mut h = LocalHamiltonian1D::default();
        h.zz.push((0, 1, std::f64::consts::FRAC_PI_4));
        let block = AnalogBlock::new(h, 1.0, 1);

        let mut circuit = DAQCCircuit::new(2);
        circuit.add_analog(block);

        let sim = DAQCSimulator::new(DAQCConfig::new());
        let probs = sim.simulate(&circuit).unwrap();

        // ZZ is diagonal in computational basis, so starting from |00>
        // probabilities stay [1, 0, 0, 0].
        assert!(
            (probs[0] - 1.0).abs() < 1e-6,
            "expected |00> = 1.0, got {}",
            probs[0]
        );
        let rest: f64 = probs[1..].iter().sum();
        assert!(rest.abs() < 1e-6, "expected rest = 0, got {}", rest);
    }

    // ---------------------------------------------------------------
    // Simulation: digital only
    // ---------------------------------------------------------------

    #[test]
    fn test_digital_only_sim() {
        let mut circuit = DAQCCircuit::new(2);
        circuit.add_gate(Gate::single(GateType::H, 0));
        circuit.add_gate(Gate::single(GateType::H, 1));

        let sim = DAQCSimulator::new(DAQCConfig::new());
        let probs = sim.simulate(&circuit).unwrap();

        // H|0> x H|0> = |+> x |+> => equal superposition
        assert_eq!(probs.len(), 4);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-10, "expected 0.25, got {}", p);
        }
    }

    // ---------------------------------------------------------------
    // Simulation: analog only (general Trotter path)
    // ---------------------------------------------------------------

    #[test]
    fn test_analog_only_sim() {
        // Rydberg Hamiltonian with nonzero X drive should produce a
        // non-trivial probability distribution.
        let h = HardwareHamiltonians::rydberg(3, 1.0, 0.5);
        let block = AnalogBlock::new(h, 0.5, 20);

        let mut circuit = DAQCCircuit::new(3);
        circuit.add_analog(block);

        let sim = DAQCSimulator::new(DAQCConfig::new().optimize_diagonal(false));
        let probs = sim.simulate(&circuit).unwrap();

        // Probabilities must sum to 1 and have some population beyond |000>.
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "probabilities must sum to 1, got {}",
            total
        );
        // The X drive should have moved some population out of |000>.
        assert!(
            probs[0] < 1.0 - 1e-6,
            "expected non-trivial evolution, got p(000)={}",
            probs[0]
        );
    }

    // ---------------------------------------------------------------
    // Simulation: mixed digital + analog
    // ---------------------------------------------------------------

    #[test]
    fn test_mixed_daqc_sim() {
        let n = 3;
        let mut circuit = DAQCCircuit::new(n);

        // Digital: H on all qubits
        for q in 0..n {
            circuit.add_gate(Gate::single(GateType::H, q));
        }

        // Analog: short transmon evolution
        let h = HardwareHamiltonians::transmon(n, 0.5);
        circuit.add_analog(AnalogBlock::new(h, 0.3, 15));

        // Digital: X on qubit 0
        circuit.add_gate(Gate::single(GateType::X, 0));

        let sim = DAQCSimulator::new(DAQCConfig::new());
        let probs = sim.simulate(&circuit).unwrap();

        // Basic sanity: probabilities sum to 1, correct length.
        assert_eq!(probs.len(), 1 << n);
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "probabilities must sum to 1, got {}",
            total
        );
        // All probabilities non-negative.
        for (i, &p) in probs.iter().enumerate() {
            assert!(p >= -1e-15, "probability[{}] = {} is negative", i, p);
        }
    }

    // ---------------------------------------------------------------
    // Fidelity comparison
    // ---------------------------------------------------------------

    #[test]
    fn test_fidelity_comparison() {
        let n = 3;
        let mut circuit = DAQCCircuit::new(n);

        for q in 0..n {
            circuit.add_gate(Gate::single(GateType::H, q));
        }

        let h = HardwareHamiltonians::transmon(n, 0.5);
        circuit.add_analog(AnalogBlock::new(h, 0.2, 10));

        let sim = DAQCSimulator::new(DAQCConfig::new());
        let comparison = sim.compare_fidelity(&circuit).unwrap();

        // Fidelity must be in (0, 1].
        assert!(
            comparison.fidelity > 0.0 && comparison.fidelity <= 1.0 + 1e-10,
            "fidelity {} not in (0, 1]",
            comparison.fidelity
        );
        assert_eq!(comparison.digital_gate_count, n);
        assert!((comparison.analog_time - 0.2).abs() < 1e-12);
        assert_eq!(comparison.total_segments, 2);
    }

    // ---------------------------------------------------------------
    // Configuration builder
    // ---------------------------------------------------------------

    #[test]
    fn test_config_builder() {
        // Defaults
        let default = DAQCConfig::new();
        assert_eq!(default.max_bond_dim, 64);
        assert_eq!(default.trotter_steps_per_unit, 10);
        assert!(default.optimize_diagonal);

        // Builder overrides
        let custom = DAQCConfig::new()
            .max_bond_dim(128)
            .trotter_steps_per_unit(50)
            .optimize_diagonal(false);
        assert_eq!(custom.max_bond_dim, 128);
        assert_eq!(custom.trotter_steps_per_unit, 50);
        assert!(!custom.optimize_diagonal);
    }

    // ---------------------------------------------------------------
    // Error cases
    // ---------------------------------------------------------------

    #[test]
    fn test_empty_circuit_error() {
        let circuit = DAQCCircuit::new(2);
        let sim = DAQCSimulator::new(DAQCConfig::new());
        let result = sim.simulate(&circuit);
        assert!(result.is_err());
        match result.unwrap_err() {
            DAQCError::EmptyCircuit => {}
            other => panic!("expected EmptyCircuit, got {:?}", other),
        }
    }

    #[test]
    fn test_negative_duration_error() {
        let mut h = LocalHamiltonian1D::default();
        h.zz.push((0, 1, 1.0));
        let block = AnalogBlock::new(h, -1.0, 10);

        let mut circuit = DAQCCircuit::new(2);
        circuit.add_analog(block);

        let sim = DAQCSimulator::new(DAQCConfig::new());
        let result = sim.simulate(&circuit);
        assert!(result.is_err());
        match result.unwrap_err() {
            DAQCError::InvalidDuration(_) => {}
            other => panic!("expected InvalidDuration, got {:?}", other),
        }
    }

    #[test]
    fn test_zero_duration_identity() {
        let mut h = LocalHamiltonian1D::default();
        h.zz.push((0, 1, 1.0));
        h.x.push((0, 0.5));
        let block = AnalogBlock::new(h, 0.0, 10);

        let mut circuit = DAQCCircuit::new(2);
        // Start with H to create superposition, then zero-duration analog
        circuit.add_gate(Gate::single(GateType::H, 0));
        circuit.add_analog(block);

        let sim = DAQCSimulator::new(DAQCConfig::new());
        let probs = sim.simulate(&circuit).unwrap();

        // Zero-duration evolution is identity. H|0> on qubit 0 gives
        // |+> = (|0> + |1>) / sqrt(2), so for 2 qubits (qubit 0 is LSB):
        // p(|00>) = 0.5, p(|01>) = 0.5, p(|10>) = 0, p(|11>) = 0.
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!(probs[3].abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // Display trait for DAQCError
    // ---------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e1 = DAQCError::InvalidDuration("negative".to_string());
        assert!(format!("{}", e1).contains("negative"));

        let e2 = DAQCError::SimulationFailed("backend crash".to_string());
        assert!(format!("{}", e2).contains("backend crash"));

        let e3 = DAQCError::EmptyCircuit;
        assert!(format!("{}", e3).contains("no segments"));
    }
}
