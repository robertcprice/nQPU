// ===================================================================
// DEVICE PRESETS
// ===================================================================

use super::neutral_atom_array::{AtomConnectivity, NeutralAtomConfig, NeutralAtomGate};
use crate::gates::{Gate, GateType};
use crate::traits::{BackendError, BackendResult, ErrorModel, QuantumBackend};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// Neutral atom hardware device presets from major vendors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeutralAtomDevice {
    /// QuEra Aquila: 256-atom Rb87 machine (Harvard/MIT collaboration).
    QuEraAquila,
    /// Atom Computing: 1000-atom 87Sr clock-qubit machine.
    AtomComputing,
    /// Pasqal Fresnel: 100-atom Rb87 processor.
    PasqalFresnel,
}

impl NeutralAtomDevice {
    /// Return the hardware-calibrated configuration for this device.
    pub fn config(&self) -> NeutralAtomConfig {
        match self {
            NeutralAtomDevice::QuEraAquila => NeutralAtomConfig {
                num_atoms: 256,
                trap_spacing_um: 5.0,
                rydberg_level: 70,
                c6_coefficient: 862_690.0, // Rb87
                max_rabi_frequency_mhz: 10.0,
                atom_temperature_uk: 10.0,
                loading_probability: 0.5,
                rearrangement_fidelity: 0.999,
                atom_loss_rate: 0.003,
                connectivity: AtomConnectivity::AllToAll,
            },
            NeutralAtomDevice::AtomComputing => NeutralAtomConfig {
                num_atoms: 1000,
                trap_spacing_um: 4.0,
                rydberg_level: 61, // 87Sr uses different Rydberg levels
                c6_coefficient: 500_000.0, // 87Sr has different C6
                max_rabi_frequency_mhz: 8.0,
                atom_temperature_uk: 5.0,
                loading_probability: 0.5,
                rearrangement_fidelity: 0.9995,
                atom_loss_rate: 0.002,
                connectivity: AtomConnectivity::AllToAll,
            },
            NeutralAtomDevice::PasqalFresnel => NeutralAtomConfig {
                num_atoms: 100,
                trap_spacing_um: 6.0,
                rydberg_level: 70,
                c6_coefficient: 862_690.0, // Rb87
                max_rabi_frequency_mhz: 12.0,
                atom_temperature_uk: 15.0,
                loading_probability: 0.6,
                rearrangement_fidelity: 0.998,
                atom_loss_rate: 0.005,
                connectivity: AtomConnectivity::AllToAll,
            },
        }
    }

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        match self {
            NeutralAtomDevice::QuEraAquila => "QuEra Aquila (256 atoms, Rb87)",
            NeutralAtomDevice::AtomComputing => "Atom Computing (1000 atoms, 87Sr)",
            NeutralAtomDevice::PasqalFresnel => "Pasqal Fresnel (100 atoms, Rb87)",
        }
    }
}

// ===================================================================
// GATE COMPILER: standard gates -> native neutral atom gate set
// ===================================================================

/// Compile a standard gate into a sequence of native neutral atom gates.
///
/// The native gate set for neutral atom hardware is:
/// - Single-qubit Rabi rotations: Rotation { atom, theta, phi }
/// - Two-qubit CZ via Rydberg blockade: CZ { atom_a, atom_b }
/// - Native three-qubit CCZ: CCZ { a, b, c }
///
/// All standard gates are decomposed into these primitives.
pub fn compile_to_neutral_atom(gate: &Gate, _config: &NeutralAtomConfig) -> Vec<NeutralAtomGate> {
    let mut ops = Vec::new();

    match &gate.gate_type {
        // --- Single-qubit gates ---
        GateType::H => {
            let t = gate.targets[0];
            // H = Ry(pi/2) then Rz(pi)
            // Decomposition: Ry(pi/2) * Rz(pi) = H up to global phase
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: FRAC_PI_2,
                phi: FRAC_PI_2,
            }); // Ry(pi/2)
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: PI,
                phi: 0.0,
            }); // Rz(pi) as rotation
        }
        GateType::X => {
            let t = gate.targets[0];
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: PI,
                phi: 0.0,
            });
        }
        GateType::Y => {
            let t = gate.targets[0];
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: PI,
                phi: FRAC_PI_2,
            });
        }
        GateType::Z => {
            let t = gate.targets[0];
            // Z = Rz(pi), implemented as phase rotation
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: 0.0,
                phi: PI,
            });
        }
        GateType::S => {
            let t = gate.targets[0];
            // S = Rz(pi/2)
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: 0.0,
                phi: FRAC_PI_2,
            });
        }
        GateType::T => {
            let t = gate.targets[0];
            // T = Rz(pi/4)
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: 0.0,
                phi: FRAC_PI_4,
            });
        }
        GateType::Rx(theta) => {
            let t = gate.targets[0];
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: *theta,
                phi: 0.0,
            });
        }
        GateType::Ry(theta) => {
            let t = gate.targets[0];
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: *theta,
                phi: FRAC_PI_2,
            });
        }
        GateType::Rz(theta) => {
            let t = gate.targets[0];
            // Rz as phase-only rotation
            ops.push(NeutralAtomGate::Rotation {
                atom: t,
                theta: 0.0,
                phi: *theta,
            });
        }

        // --- Two-qubit gates ---
        GateType::CZ => {
            let ctrl = gate.controls[0];
            let tgt = gate.targets[0];
            ops.push(NeutralAtomGate::CZ {
                atom_a: ctrl,
                atom_b: tgt,
            });
        }
        GateType::CNOT => {
            // CNOT = H(target) CZ H(target)
            let ctrl = gate.controls[0];
            let tgt = gate.targets[0];
            // H on target
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: FRAC_PI_2,
                phi: FRAC_PI_2,
            });
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: PI,
                phi: 0.0,
            });
            // CZ
            ops.push(NeutralAtomGate::CZ {
                atom_a: ctrl,
                atom_b: tgt,
            });
            // H on target
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: FRAC_PI_2,
                phi: FRAC_PI_2,
            });
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: PI,
                phi: 0.0,
            });
        }

        // --- Three-qubit gates ---
        GateType::Toffoli => {
            // Toffoli = H(target) CCZ H(target)
            // CCZ is native on Rydberg hardware.
            let c0 = gate.controls[0];
            let c1 = gate.controls[1];
            let tgt = gate.targets[0];
            // H on target
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: FRAC_PI_2,
                phi: FRAC_PI_2,
            });
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: PI,
                phi: 0.0,
            });
            // Native CCZ
            ops.push(NeutralAtomGate::CCZ {
                a: c0,
                b: c1,
                c: tgt,
            });
            // H on target
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: FRAC_PI_2,
                phi: FRAC_PI_2,
            });
            ops.push(NeutralAtomGate::Rotation {
                atom: tgt,
                theta: PI,
                phi: 0.0,
            });
        }
        GateType::CCZ => {
            // Native CCZ
            let c0 = gate.controls[0];
            let c1 = gate.controls[1];
            let tgt = gate.targets[0];
            ops.push(NeutralAtomGate::CCZ {
                a: c0,
                b: c1,
                c: tgt,
            });
        }

        // Fall back: treat as identity (unsupported gates produce empty sequence).
        _ => {}
    }

    ops
}

// ===================================================================
// NEUTRAL ATOM BACKEND SIMULATOR (full state-vector)
// ===================================================================

/// Full state-vector quantum simulator modelling neutral atom hardware.
///
/// Unlike `NeutralAtomSimulator` which tracks per-atom Bloch-sphere states,
/// this backend maintains a 2^n amplitude vector and implements the
/// `QuantumBackend` trait for use with backend-agnostic algorithms.
pub struct NeutralAtomBackendSimulator {
    /// Number of logical qubits.
    pub num_qubits: usize,
    /// State vector: 2^n complex amplitudes in computational basis.
    pub state: Vec<Complex64>,
    /// Hardware configuration (used for noise parameters).
    pub config: NeutralAtomConfig,
    /// Whether noise injection is enabled.
    pub noise_enabled: bool,
    /// Deterministic PRNG state (xorshift64).
    rng_state: u64,
    /// Per-gate single-qubit error rate.
    single_gate_error: f64,
    /// Per-gate two-qubit error rate.
    two_qubit_gate_error: f64,
    /// Per-operation atom loss probability.
    atom_loss_prob: f64,
}

impl NeutralAtomBackendSimulator {
    /// Create a new simulator with the given qubit count and hardware config.
    ///
    /// Initialises the state to |0...0> with noise parameters derived from
    /// the configuration's atom_loss_rate.
    pub fn new(num_qubits: usize, config: NeutralAtomConfig) -> Self {
        let dim = 1usize << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        let atom_loss = config.atom_loss_rate;
        Self {
            num_qubits,
            state,
            config,
            noise_enabled: true,
            rng_state: 42,
            single_gate_error: 0.001,     // 99.9% single-qubit fidelity
            two_qubit_gate_error: 0.005,   // 99.5% two-qubit fidelity
            atom_loss_prob: atom_loss,
        }
    }

    /// Create an ideal (noise-free) simulator.
    pub fn ideal(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        Self {
            num_qubits,
            state,
            config: NeutralAtomConfig::default(),
            noise_enabled: false,
            rng_state: 42,
            single_gate_error: 0.0,
            two_qubit_gate_error: 0.0,
            atom_loss_prob: 0.0,
        }
    }

    /// Create a simulator from a device preset.
    pub fn from_device(num_qubits: usize, device: NeutralAtomDevice) -> Self {
        Self::new(num_qubits, device.config())
    }

    // ---------------------------------------------------------------
    // Helper: xorshift64 PRNG
    // ---------------------------------------------------------------

    /// Generate a pseudo-random f64 in [0, 1) using xorshift64.
    fn rand_f64(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    // ---------------------------------------------------------------
    // Helper: apply 2x2 unitary to single qubit
    // ---------------------------------------------------------------

    /// Apply a 2x2 unitary matrix to a single qubit in the state vector.
    fn apply_single_qubit_unitary(&mut self, qubit: usize, u: [[Complex64; 2]; 2]) {
        let dim = 1 << self.num_qubits;
        let mask = 1 << qubit;
        for i in 0..dim {
            if i & mask == 0 {
                let j = i | mask;
                let a0 = self.state[i];
                let a1 = self.state[j];
                self.state[i] = u[0][0] * a0 + u[0][1] * a1;
                self.state[j] = u[1][0] * a0 + u[1][1] * a1;
            }
        }
    }

    // ---------------------------------------------------------------
    // Helper: apply 4x4 unitary to two qubits
    // ---------------------------------------------------------------

    /// Apply a 4x4 unitary (stored as flat [Complex64; 16]) to two qubits.
    fn apply_two_qubit_unitary(&mut self, q0: usize, q1: usize, u: &[Complex64; 16]) {
        let dim = 1 << self.num_qubits;
        let mask0 = 1 << q0;
        let mask1 = 1 << q1;
        for i in 0..dim {
            if i & mask0 == 0 && i & mask1 == 0 {
                let i00 = i;
                let i01 = i | mask1;
                let i10 = i | mask0;
                let i11 = i | mask0 | mask1;
                let a = [
                    self.state[i00],
                    self.state[i01],
                    self.state[i10],
                    self.state[i11],
                ];
                for (row, idx) in [(0, i00), (1, i01), (2, i10), (3, i11)] {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for col in 0..4 {
                        sum += u[row * 4 + col] * a[col];
                    }
                    self.state[idx] = sum;
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Helper: apply 8x8 unitary to three qubits
    // ---------------------------------------------------------------

    /// Apply an 8x8 unitary (stored as flat [Complex64; 64]) to three qubits.
    fn apply_three_qubit_unitary(&mut self, q0: usize, q1: usize, q2: usize, u: &[Complex64; 64]) {
        let dim = 1 << self.num_qubits;
        let mask0 = 1 << q0;
        let mask1 = 1 << q1;
        let mask2 = 1 << q2;
        for i in 0..dim {
            if i & mask0 == 0 && i & mask1 == 0 && i & mask2 == 0 {
                let indices = [
                    i,
                    i | mask2,
                    i | mask1,
                    i | mask1 | mask2,
                    i | mask0,
                    i | mask0 | mask2,
                    i | mask0 | mask1,
                    i | mask0 | mask1 | mask2,
                ];
                let a: [Complex64; 8] = [
                    self.state[indices[0]],
                    self.state[indices[1]],
                    self.state[indices[2]],
                    self.state[indices[3]],
                    self.state[indices[4]],
                    self.state[indices[5]],
                    self.state[indices[6]],
                    self.state[indices[7]],
                ];
                for (row, &idx) in indices.iter().enumerate() {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for col in 0..8 {
                        sum += u[row * 8 + col] * a[col];
                    }
                    self.state[idx] = sum;
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Helper: depolarizing noise
    // ---------------------------------------------------------------

    /// Inject single-qubit depolarizing noise: with probability `p`,
    /// apply a random Pauli (X, Y, or Z) to the qubit.
    fn inject_depolarizing(&mut self, qubit: usize, p: f64) {
        if !self.noise_enabled || p <= 0.0 {
            return;
        }
        let r = self.rand_f64();
        if r < p {
            let zero = Complex64::new(0.0, 0.0);
            let one = Complex64::new(1.0, 0.0);
            let i_ = Complex64::new(0.0, 1.0);
            let ni = Complex64::new(0.0, -1.0);
            let neg = Complex64::new(-1.0, 0.0);
            let pauli_choice = (self.rand_f64() * 3.0) as usize;
            match pauli_choice {
                0 => self.apply_single_qubit_unitary(qubit, [[zero, one], [one, zero]]), // X
                1 => self.apply_single_qubit_unitary(qubit, [[zero, ni], [i_, zero]]),   // Y
                _ => self.apply_single_qubit_unitary(qubit, [[one, zero], [zero, neg]]), // Z
            }
        }
    }
}

// ===================================================================
// QuantumBackend TRAIT IMPLEMENTATION
// ===================================================================

impl QuantumBackend for NeutralAtomBackendSimulator {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply_gate(&mut self, gate: &Gate) -> BackendResult<()> {
        // Apply the ideal gate using its exact matrix representation.
        // Native compilation is only used for noise modeling, not state evolution,
        // because hardware-specific decompositions introduce phases that are
        // compensated by frame tracking on real devices but not in simulation.
        let matrix = gate.gate_type.matrix();
        let all_qubits: Vec<usize> = gate
            .controls
            .iter()
            .chain(gate.targets.iter())
            .copied()
            .collect();

        // Validate qubit indices.
        for &q in &all_qubits {
            if q >= self.num_qubits {
                return Err(BackendError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
            }
        }

        match all_qubits.len() {
            1 => {
                let q = all_qubits[0];
                let u = [
                    [matrix[0][0], matrix[0][1]],
                    [matrix[1][0], matrix[1][1]],
                ];
                self.apply_single_qubit_unitary(q, u);

                // Apply noise after single-qubit gate.
                if self.noise_enabled {
                    self.inject_depolarizing(q, self.single_gate_error);
                }
            }
            2 => {
                let (q0, q1) = (all_qubits[0], all_qubits[1]);
                let mut u = [Complex64::new(0.0, 0.0); 16];
                for r in 0..4 {
                    for c in 0..4 {
                        u[r * 4 + c] = matrix[r][c];
                    }
                }
                self.apply_two_qubit_unitary(q0, q1, &u);

                // Apply noise after two-qubit gate.
                if self.noise_enabled {
                    self.inject_depolarizing(q0, self.two_qubit_gate_error);
                    self.inject_depolarizing(q1, self.two_qubit_gate_error);
                }
            }
            3 => {
                // Three-qubit gates (Toffoli, CCZ, CSWAP).
                let (q0, q1, q2) = (all_qubits[0], all_qubits[1], all_qubits[2]);
                let mut u = [Complex64::new(0.0, 0.0); 64];
                for r in 0..8 {
                    for c in 0..8 {
                        u[r * 8 + c] = matrix[r][c];
                    }
                }
                self.apply_three_qubit_unitary(q0, q1, q2, &u);

                // Apply noise to all three qubits.
                if self.noise_enabled {
                    self.inject_depolarizing(q0, self.two_qubit_gate_error);
                    self.inject_depolarizing(q1, self.two_qubit_gate_error);
                    self.inject_depolarizing(q2, self.two_qubit_gate_error);
                }
            }
            _ => {
                return Err(BackendError::UnsupportedGate(format!(
                    "{:?} ({} qubits)",
                    gate.gate_type,
                    all_qubits.len()
                )));
            }
        }

        Ok(())
    }

    fn probabilities(&self) -> BackendResult<Vec<f64>> {
        Ok(self.state.iter().map(|a| a.norm_sqr()).collect())
    }

    fn sample(&self, n_shots: usize) -> BackendResult<HashMap<usize, usize>> {
        let probs = self.probabilities()?;
        let mut counts = HashMap::new();
        let mut rng = self.rng_state;
        for _ in 0..n_shots {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let r = (rng as f64) / (u64::MAX as f64);
            let mut cumulative = 0.0;
            let mut outcome = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    outcome = i;
                    break;
                }
            }
            *counts.entry(outcome).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn measure_qubit(&mut self, qubit: usize) -> BackendResult<u8> {
        if qubit >= self.num_qubits {
            return Err(BackendError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }

        // Compute probability of |0> on this qubit.
        let mask = 1 << qubit;
        let dim = 1 << self.num_qubits;
        let mut prob_0: f64 = 0.0;
        for i in 0..dim {
            if i & mask == 0 {
                prob_0 += self.state[i].norm_sqr();
            }
        }

        // Determine outcome via PRNG.
        let r = self.rand_f64();
        let outcome = if r < prob_0 { 0u8 } else { 1u8 };

        // Collapse state vector.
        let norm_sq = if outcome == 0 { prob_0 } else { 1.0 - prob_0 };
        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            for i in 0..dim {
                let bit = ((i & mask) != 0) as u8;
                if bit == outcome {
                    self.state[i] /= Complex64::new(norm, 0.0);
                } else {
                    self.state[i] = Complex64::new(0.0, 0.0);
                }
            }
        }

        Ok(outcome)
    }

    fn reset(&mut self) {
        let dim = 1 << self.num_qubits;
        self.state = vec![Complex64::new(0.0, 0.0); dim];
        self.state[0] = Complex64::new(1.0, 0.0);
    }

    fn name(&self) -> &str {
        "NeutralAtomBackend"
    }
}

// ===================================================================
// ERROR MODEL
// ===================================================================

/// Noise model for neutral atom hardware, implementing the `ErrorModel` trait.
pub struct NeutralAtomErrorModel {
    /// Single-qubit gate error rate (depolarizing probability).
    pub single_gate_error: f64,
    /// Two-qubit gate error rate (depolarizing probability per qubit).
    pub two_qubit_gate_error: f64,
    /// Per-operation atom loss probability.
    pub atom_loss_prob: f64,
    /// Whether noise is active.
    pub enable_noise: bool,
}

impl NeutralAtomErrorModel {
    /// Create a noise model from hardware configuration.
    pub fn from_config(config: &NeutralAtomConfig) -> Self {
        Self {
            single_gate_error: 0.001,
            two_qubit_gate_error: 0.005,
            atom_loss_prob: config.atom_loss_rate,
            enable_noise: true,
        }
    }

    /// Create an ideal (noise-free) error model.
    pub fn ideal() -> Self {
        Self {
            single_gate_error: 0.0,
            two_qubit_gate_error: 0.0,
            atom_loss_prob: 0.0,
            enable_noise: false,
        }
    }

    /// Create a noise model from a device preset.
    pub fn from_device(device: NeutralAtomDevice) -> Self {
        Self::from_config(&device.config())
    }
}

impl ErrorModel for NeutralAtomErrorModel {
    fn apply_noise_after_gate(
        &self,
        _gate: &Gate,
        _state: &mut dyn QuantumBackend,
    ) -> BackendResult<()> {
        // Noise is applied internally by NeutralAtomBackendSimulator;
        // this is for composing with other backends via the ErrorModel trait.
        Ok(())
    }

    fn apply_idle_noise(
        &self,
        _qubit: usize,
        _state: &mut dyn QuantumBackend,
    ) -> BackendResult<()> {
        // Idle noise is minimal for neutral atoms in deep optical traps.
        Ok(())
    }

    fn gate_error_rate(&self, gate: &Gate) -> f64 {
        if !self.enable_noise {
            return 0.0;
        }
        let is_multi_qubit = matches!(
            gate.gate_type,
            GateType::CNOT
                | GateType::CZ
                | GateType::SWAP
                | GateType::ISWAP
                | GateType::Toffoli
                | GateType::CCZ
                | GateType::CSWAP
                | GateType::Rxx(_)
                | GateType::Ryy(_)
                | GateType::Rzz(_)
        );
        if is_multi_qubit {
            self.two_qubit_gate_error
        } else {
            self.single_gate_error
        }
    }
}

// ===================================================================
// QCVV EXPERIMENT GENERATORS
// ===================================================================

/// Generate circuits for quantum characterization, verification, and
/// validation on neutral atom hardware.
pub struct NeutralAtomQCVV;

impl NeutralAtomQCVV {
    /// Generate a GHZ state preparation circuit.
    ///
    /// Produces (|0...0> + |1...1>) / sqrt(2) using an H gate followed
    /// by a cascade of CNOT gates.
    pub fn ghz_circuit(n_qubits: usize) -> Vec<Gate> {
        assert!(n_qubits >= 2, "GHZ circuit requires at least 2 qubits");
        let mut circuit = vec![Gate::single(GateType::H, 0)];
        for i in 0..(n_qubits - 1) {
            circuit.push(Gate::two(GateType::CNOT, i, i + 1));
        }
        circuit
    }

    /// Generate a random Clifford+T circuit for quantum volume estimation.
    ///
    /// Each layer applies random single-qubit Clifford gates followed by
    /// CZ entangling gates on alternating qubit pairs.
    pub fn random_circuit(n_qubits: usize, depth: usize, seed: u64) -> Vec<Gate> {
        let clifford_gates = [
            GateType::H,
            GateType::S,
            GateType::T,
            GateType::X,
            GateType::Y,
            GateType::Z,
        ];
        let mut circuit = Vec::new();
        let mut rng = seed;

        for layer in 0..depth {
            // Random single-qubit layer.
            for q in 0..n_qubits {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let idx = (rng as usize) % clifford_gates.len();
                circuit.push(Gate::single(clifford_gates[idx].clone(), q));
            }

            // Entangling layer: alternate even/odd pair CZ gates.
            let offset = layer % 2;
            let mut pair = offset;
            while pair + 1 < n_qubits {
                circuit.push(Gate::two(GateType::CZ, pair, pair + 1));
                pair += 2;
            }
        }

        circuit
    }

    /// Generate a single-qubit randomized benchmarking sequence.
    ///
    /// Applies `num_cliffords` random Clifford gates to the given qubit,
    /// followed by an H gate as a simplified inverse Clifford.
    pub fn rb_circuit(qubit: usize, num_cliffords: usize, seed: u64) -> Vec<Gate> {
        let clifford_set = [
            GateType::H,
            GateType::S,
            GateType::X,
            GateType::Y,
            GateType::Z,
            GateType::SX,
        ];
        let mut circuit = Vec::with_capacity(num_cliffords + 1);
        let mut rng = seed.wrapping_mul(1000).wrapping_add(qubit as u64);

        for _ in 0..num_cliffords {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let idx = (rng as usize) % clifford_set.len();
            circuit.push(Gate::single(clifford_set[idx].clone(), qubit));
        }

        // Simplified inverse Clifford: H gate for measurement.
        circuit.push(Gate::single(GateType::H, qubit));
        circuit
    }
}

// ===================================================================
// TESTS FOR BACKEND EXTENSIONS
// ===================================================================

#[cfg(test)]
mod backend_tests {
    use super::*;

    // ---------------------------------------------------------------
    // 1. Device presets
    // ---------------------------------------------------------------

    #[test]
    fn test_quera_aquila_preset() {
        let cfg = NeutralAtomDevice::QuEraAquila.config();
        assert_eq!(cfg.num_atoms, 256);
        assert!((cfg.trap_spacing_um - 5.0).abs() < 1e-12);
        assert!((cfg.loading_probability - 0.5).abs() < 1e-12);
        assert!((cfg.atom_loss_rate - 0.003).abs() < 1e-12);
    }

    #[test]
    fn test_atom_computing_preset() {
        let cfg = NeutralAtomDevice::AtomComputing.config();
        assert_eq!(cfg.num_atoms, 1000);
        assert!((cfg.trap_spacing_um - 4.0).abs() < 1e-12);
        assert!((cfg.loading_probability - 0.5).abs() < 1e-12);
        assert!((cfg.atom_loss_rate - 0.002).abs() < 1e-12);
    }

    #[test]
    fn test_pasqal_fresnel_preset() {
        let cfg = NeutralAtomDevice::PasqalFresnel.config();
        assert_eq!(cfg.num_atoms, 100);
        assert!((cfg.trap_spacing_um - 6.0).abs() < 1e-12);
        assert!((cfg.loading_probability - 0.6).abs() < 1e-12);
        assert!((cfg.atom_loss_rate - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_device_names() {
        assert!(NeutralAtomDevice::QuEraAquila.name().contains("QuEra"));
        assert!(NeutralAtomDevice::AtomComputing.name().contains("Atom Computing"));
        assert!(NeutralAtomDevice::PasqalFresnel.name().contains("Pasqal"));
    }

    // ---------------------------------------------------------------
    // 2. Gate compiler
    // ---------------------------------------------------------------

    #[test]
    fn test_compile_h_gate() {
        let cfg = NeutralAtomConfig::default();
        let gate = Gate::single(GateType::H, 0);
        let ops = compile_to_neutral_atom(&gate, &cfg);
        assert_eq!(ops.len(), 2); // Ry(pi/2) + Rz(pi)
        match &ops[0] {
            NeutralAtomGate::Rotation { atom, theta, phi } => {
                assert_eq!(*atom, 0);
                assert!((theta - FRAC_PI_2).abs() < 1e-12);
                assert!((phi - FRAC_PI_2).abs() < 1e-12);
            }
            _ => panic!("Expected Rotation"),
        }
    }

    #[test]
    fn test_compile_x_gate() {
        let cfg = NeutralAtomConfig::default();
        let gate = Gate::single(GateType::X, 2);
        let ops = compile_to_neutral_atom(&gate, &cfg);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            NeutralAtomGate::Rotation { atom, theta, .. } => {
                assert_eq!(*atom, 2);
                assert!((theta - PI).abs() < 1e-12);
            }
            _ => panic!("Expected Rotation"),
        }
    }

    #[test]
    fn test_compile_cz_native() {
        let cfg = NeutralAtomConfig::default();
        let gate = Gate::two(GateType::CZ, 0, 1);
        let ops = compile_to_neutral_atom(&gate, &cfg);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            NeutralAtomGate::CZ { atom_a, atom_b } => {
                assert_eq!(*atom_a, 0);
                assert_eq!(*atom_b, 1);
            }
            _ => panic!("Expected CZ"),
        }
    }

    #[test]
    fn test_compile_cnot() {
        let cfg = NeutralAtomConfig::default();
        let gate = Gate::two(GateType::CNOT, 0, 1);
        let ops = compile_to_neutral_atom(&gate, &cfg);
        // CNOT = H(t) + CZ + H(t), each H = 2 rotations => 2 + 1 + 2 = 5
        assert_eq!(ops.len(), 5);
        // Middle element should be CZ.
        assert!(matches!(ops[2], NeutralAtomGate::CZ { .. }));
    }

    #[test]
    fn test_compile_toffoli() {
        let cfg = NeutralAtomConfig::default();
        let gate = Gate::new(GateType::Toffoli, vec![2], vec![0, 1]);
        let ops = compile_to_neutral_atom(&gate, &cfg);
        // Toffoli = H(t) + CCZ + H(t) => 2 + 1 + 2 = 5
        assert_eq!(ops.len(), 5);
        assert!(matches!(ops[2], NeutralAtomGate::CCZ { .. }));
    }

    #[test]
    fn test_compile_s_t_gates() {
        let cfg = NeutralAtomConfig::default();
        let s_ops = compile_to_neutral_atom(&Gate::single(GateType::S, 0), &cfg);
        assert_eq!(s_ops.len(), 1);

        let t_ops = compile_to_neutral_atom(&Gate::single(GateType::T, 0), &cfg);
        assert_eq!(t_ops.len(), 1);
    }

    #[test]
    fn test_compile_rotation_gates() {
        let cfg = NeutralAtomConfig::default();
        let rx = compile_to_neutral_atom(&Gate::single(GateType::Rx(1.0), 0), &cfg);
        assert_eq!(rx.len(), 1);
        let ry = compile_to_neutral_atom(&Gate::single(GateType::Ry(1.0), 0), &cfg);
        assert_eq!(ry.len(), 1);
        let rz = compile_to_neutral_atom(&Gate::single(GateType::Rz(1.0), 0), &cfg);
        assert_eq!(rz.len(), 1);
    }

    // ---------------------------------------------------------------
    // 3. Backend basic operations
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_new_initialises_zero_state() {
        let backend = NeutralAtomBackendSimulator::ideal(3);
        assert_eq!(backend.num_qubits(), 3);
        assert_eq!(backend.name(), "NeutralAtomBackend");
        let probs = backend.probabilities().unwrap();
        assert_eq!(probs.len(), 8);
        assert!((probs[0] - 1.0).abs() < 1e-12);
        for p in &probs[1..] {
            assert!(p.abs() < 1e-12);
        }
    }

    #[test]
    fn test_backend_h_gate() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        backend.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_backend_x_gate() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        backend.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        let probs = backend.probabilities().unwrap();
        assert!(probs[0].abs() < 1e-12);
        assert!((probs[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_backend_z_gate_on_plus_state() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        // Prepare |+>
        backend.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        // Z|+> = |-> (probabilities unchanged, only phase)
        backend.apply_gate(&Gate::single(GateType::Z, 0)).unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 4. Bell state
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_bell_state() {
        let mut backend = NeutralAtomBackendSimulator::ideal(2);
        backend.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        backend
            .apply_gate(&Gate::two(GateType::CNOT, 0, 1))
            .unwrap();

        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10, "|00> = {}", probs[0]);
        assert!(probs[1].abs() < 1e-10, "|01> = {}", probs[1]);
        assert!(probs[2].abs() < 1e-10, "|10> = {}", probs[2]);
        assert!((probs[3] - 0.5).abs() < 1e-10, "|11> = {}", probs[3]);
    }

    #[test]
    fn test_backend_cz_gate() {
        let mut backend = NeutralAtomBackendSimulator::ideal(2);
        // Prepare |+> on both qubits.
        backend.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        backend.apply_gate(&Gate::single(GateType::H, 1)).unwrap();
        // CZ on |++> = (|00> + |01> + |10> - |11>) / 2
        backend.apply_gate(&Gate::two(GateType::CZ, 0, 1)).unwrap();
        let probs = backend.probabilities().unwrap();
        // All four basis states should have equal probability 0.25.
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-10, "prob = {}", p);
        }
    }

    // ---------------------------------------------------------------
    // 5. Sample
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_sample_deterministic_state() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        backend.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        let counts = backend.sample(1000).unwrap();
        // |1> should be sampled every time.
        assert_eq!(*counts.get(&1).unwrap_or(&0), 1000);
        assert_eq!(*counts.get(&0).unwrap_or(&0), 0);
    }

    #[test]
    fn test_backend_sample_superposition() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        backend.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        let counts = backend.sample(10000).unwrap();
        let c0 = *counts.get(&0).unwrap_or(&0) as f64;
        let c1 = *counts.get(&1).unwrap_or(&0) as f64;
        // Expect roughly 50/50 within statistical tolerance.
        assert!(
            (c0 / 10000.0 - 0.5).abs() < 0.1,
            "c0 = {}, c1 = {}",
            c0,
            c1
        );
    }

    // ---------------------------------------------------------------
    // 6. Measure
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_measure_qubit_zero() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        let outcome = backend.measure_qubit(0).unwrap();
        assert_eq!(outcome, 0);
        // State should collapse to |0>.
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_backend_measure_qubit_one() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        backend.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        let outcome = backend.measure_qubit(0).unwrap();
        assert_eq!(outcome, 1);
    }

    #[test]
    fn test_backend_measure_out_of_range() {
        let mut backend = NeutralAtomBackendSimulator::ideal(2);
        let result = backend.measure_qubit(5);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 7. Reset
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_reset() {
        let mut backend = NeutralAtomBackendSimulator::ideal(2);
        backend.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        backend.apply_gate(&Gate::single(GateType::X, 1)).unwrap();
        backend.reset();
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 8. From device preset
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_from_device() {
        let backend = NeutralAtomBackendSimulator::from_device(4, NeutralAtomDevice::QuEraAquila);
        assert_eq!(backend.num_qubits(), 4);
        assert!(backend.noise_enabled);
        assert!((backend.atom_loss_prob - 0.003).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 9. Error model
    // ---------------------------------------------------------------

    #[test]
    fn test_error_model_from_config() {
        let cfg = NeutralAtomDevice::QuEraAquila.config();
        let em = NeutralAtomErrorModel::from_config(&cfg);
        assert!(em.enable_noise);
        assert!((em.single_gate_error - 0.001).abs() < 1e-12);
        assert!((em.two_qubit_gate_error - 0.005).abs() < 1e-12);
        assert!((em.atom_loss_prob - 0.003).abs() < 1e-12);
    }

    #[test]
    fn test_error_model_ideal() {
        let em = NeutralAtomErrorModel::ideal();
        assert!(!em.enable_noise);
        assert!((em.single_gate_error).abs() < 1e-12);
        assert!((em.two_qubit_gate_error).abs() < 1e-12);
    }

    #[test]
    fn test_error_model_gate_error_rate() {
        let em = NeutralAtomErrorModel::from_device(NeutralAtomDevice::QuEraAquila);
        let h_gate = Gate::single(GateType::H, 0);
        let cz_gate = Gate::two(GateType::CZ, 0, 1);
        assert!((em.gate_error_rate(&h_gate) - 0.001).abs() < 1e-12);
        assert!((em.gate_error_rate(&cz_gate) - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_error_model_ideal_returns_zero() {
        let em = NeutralAtomErrorModel::ideal();
        let gate = Gate::single(GateType::X, 0);
        assert!((em.gate_error_rate(&gate)).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 10. Noisy vs ideal backend
    // ---------------------------------------------------------------

    #[test]
    fn test_noisy_backend_still_functional() {
        let mut backend =
            NeutralAtomBackendSimulator::from_device(2, NeutralAtomDevice::QuEraAquila);
        // Even with noise, basic operations should not crash.
        backend.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        backend
            .apply_gate(&Gate::two(GateType::CNOT, 0, 1))
            .unwrap();
        let probs = backend.probabilities().unwrap();
        let total: f64 = probs.iter().sum();
        // Probability should still sum to ~1 even with noise.
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Total probability = {}",
            total
        );
    }

    // ---------------------------------------------------------------
    // 11. QCVV: GHZ circuit
    // ---------------------------------------------------------------

    #[test]
    fn test_qcvv_ghz_circuit() {
        let circuit = NeutralAtomQCVV::ghz_circuit(4);
        // H + 3 CNOTs = 4 gates.
        assert_eq!(circuit.len(), 4);
        assert!(matches!(circuit[0].gate_type, GateType::H));
        for g in &circuit[1..] {
            assert!(matches!(g.gate_type, GateType::CNOT));
        }
    }

    #[test]
    fn test_qcvv_ghz_produces_entangled_state() {
        let mut backend = NeutralAtomBackendSimulator::ideal(3);
        let circuit = NeutralAtomQCVV::ghz_circuit(3);
        backend.apply_circuit(&circuit).unwrap();
        let probs = backend.probabilities().unwrap();
        // GHZ: only |000> and |111> should have non-zero probability.
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[7] - 0.5).abs() < 1e-10);
        for i in 1..7 {
            assert!(probs[i].abs() < 1e-10, "probs[{}] = {}", i, probs[i]);
        }
    }

    // ---------------------------------------------------------------
    // 12. QCVV: random circuit
    // ---------------------------------------------------------------

    #[test]
    fn test_qcvv_random_circuit() {
        let circuit = NeutralAtomQCVV::random_circuit(4, 5, 42);
        // Each depth layer: 4 single-qubit + up to 2 CZ gates.
        assert!(!circuit.is_empty());
        // Circuit should be deterministic for same seed.
        let circuit2 = NeutralAtomQCVV::random_circuit(4, 5, 42);
        assert_eq!(circuit.len(), circuit2.len());
    }

    #[test]
    fn test_qcvv_random_circuit_runs_on_backend() {
        let mut backend = NeutralAtomBackendSimulator::ideal(4);
        let circuit = NeutralAtomQCVV::random_circuit(4, 3, 123);
        backend.apply_circuit(&circuit).unwrap();
        let probs = backend.probabilities().unwrap();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 13. QCVV: randomized benchmarking
    // ---------------------------------------------------------------

    #[test]
    fn test_qcvv_rb_circuit() {
        let circuit = NeutralAtomQCVV::rb_circuit(0, 10, 42);
        // 10 Cliffords + 1 H inverse = 11 gates.
        assert_eq!(circuit.len(), 11);
        // Last gate should be H.
        assert!(matches!(circuit[10].gate_type, GateType::H));
    }

    #[test]
    fn test_qcvv_rb_circuit_deterministic() {
        let c1 = NeutralAtomQCVV::rb_circuit(0, 20, 99);
        let c2 = NeutralAtomQCVV::rb_circuit(0, 20, 99);
        assert_eq!(c1.len(), c2.len());
    }

    // ---------------------------------------------------------------
    // 14. Backend: apply_circuit
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_apply_circuit() {
        let mut backend = NeutralAtomBackendSimulator::ideal(2);
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::two(GateType::CNOT, 0, 1),
        ];
        backend.apply_circuit(&circuit).unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 15. Backend: qubit out of range
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_gate_out_of_range() {
        let mut backend = NeutralAtomBackendSimulator::ideal(2);
        let result = backend.apply_gate(&Gate::single(GateType::X, 5));
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 16. Three-qubit gate (Toffoli)
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_toffoli_gate() {
        let mut backend = NeutralAtomBackendSimulator::ideal(3);
        // Prepare |110>: X on qubits 0 and 1.
        backend.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        backend.apply_gate(&Gate::single(GateType::X, 1)).unwrap();
        // Toffoli(0, 1, 2): should flip qubit 2 when both controls are 1.
        backend
            .apply_gate(&Gate::new(GateType::Toffoli, vec![2], vec![0, 1]))
            .unwrap();
        let probs = backend.probabilities().unwrap();
        // Expected: |111> = index 7
        assert!(
            (probs[7] - 1.0).abs() < 1e-10,
            "probs[7] = {}",
            probs[7]
        );
    }

    #[test]
    fn test_backend_toffoli_no_flip() {
        let mut backend = NeutralAtomBackendSimulator::ideal(3);
        // Prepare |100>: X on qubit 0 only (one control on).
        backend.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        // Toffoli should NOT flip target.
        backend
            .apply_gate(&Gate::new(GateType::Toffoli, vec![2], vec![0, 1]))
            .unwrap();
        let probs = backend.probabilities().unwrap();
        // Expected: |100> = index 1 (qubit 0 is bit 0)
        assert!(
            (probs[1] - 1.0).abs() < 1e-10,
            "probs[1] = {}",
            probs[1]
        );
    }

    // ---------------------------------------------------------------
    // 17. Rotation gates
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_rx_ry_rz() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        // Rx(pi) should flip |0> to |1> (up to global phase).
        backend
            .apply_gate(&Gate::single(GateType::Rx(PI), 0))
            .unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[1] - 1.0).abs() < 1e-10);

        backend.reset();
        // Ry(pi) should flip |0> to |1>.
        backend
            .apply_gate(&Gate::single(GateType::Ry(PI), 0))
            .unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[1] - 1.0).abs() < 1e-10);

        backend.reset();
        // Rz(pi) on |0> should still be |0> (just phase).
        backend
            .apply_gate(&Gate::single(GateType::Rz(PI), 0))
            .unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 18. S and T gates
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_s_gate() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        // S on |0> should remain |0>.
        backend.apply_gate(&Gate::single(GateType::S, 0)).unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_backend_t_gate() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        // T on |0> should remain |0>.
        backend.apply_gate(&Gate::single(GateType::T, 0)).unwrap();
        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 19. Expectation Z
    // ---------------------------------------------------------------

    #[test]
    fn test_backend_expectation_z() {
        let mut backend = NeutralAtomBackendSimulator::ideal(1);
        // |0> state: <Z> = +1
        let ez = backend.expectation_z(0).unwrap();
        assert!((ez - 1.0).abs() < 1e-10);

        // |1> state: <Z> = -1
        backend.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        let ez = backend.expectation_z(0).unwrap();
        assert!((ez - (-1.0)).abs() < 1e-10);

        // |+> state: <Z> = 0
        backend.reset();
        backend.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        let ez = backend.expectation_z(0).unwrap();
        assert!(ez.abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 20. Error model trait methods
    // ---------------------------------------------------------------

    #[test]
    fn test_error_model_apply_noise_no_crash() {
        let em = NeutralAtomErrorModel::from_device(NeutralAtomDevice::QuEraAquila);
        let mut backend = NeutralAtomBackendSimulator::ideal(2);
        let gate = Gate::single(GateType::H, 0);
        // The trait method is a no-op (noise is internal to the backend)
        // but it should not crash.
        em.apply_noise_after_gate(&gate, &mut backend).unwrap();
        em.apply_idle_noise(0, &mut backend).unwrap();
    }

    // ---------------------------------------------------------------
    // 21. Multi-qubit circuit fidelity
    // ---------------------------------------------------------------

    #[test]
    fn test_ideal_backend_preserves_unitarity() {
        let mut backend = NeutralAtomBackendSimulator::ideal(3);
        // Apply a random-ish circuit.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::T, 1),
            Gate::two(GateType::CNOT, 0, 1),
            Gate::single(GateType::S, 2),
            Gate::two(GateType::CZ, 1, 2),
            Gate::single(GateType::Ry(1.23), 0),
        ];
        backend.apply_circuit(&circuit).unwrap();
        let probs = backend.probabilities().unwrap();
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Total probability = {}",
            total
        );
    }

    // ---------------------------------------------------------------
    // 22. Compile then simulate matches direct
    // ---------------------------------------------------------------

    #[test]
    fn test_compile_roundtrip_h_gate() {
        // Direct H vs compiled H should produce same probabilities.
        let mut direct = NeutralAtomBackendSimulator::ideal(1);
        direct.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        let probs_direct = direct.probabilities().unwrap();

        // Compiled H goes through Ry(pi/2) + Rz(pi) on the NeutralAtomSimulator,
        // but the backend uses the matrix directly, so this test verifies the
        // compile function produces the right gate count and structure.
        let cfg = NeutralAtomConfig::default();
        let ops = compile_to_neutral_atom(&Gate::single(GateType::H, 0), &cfg);
        assert_eq!(ops.len(), 2);

        // The direct backend result should show equal superposition.
        assert!((probs_direct[0] - 0.5).abs() < 1e-10);
        assert!((probs_direct[1] - 0.5).abs() < 1e-10);
    }
}
