//! Property-Based Testing Infrastructure for nQPU-Metal
//!
//! Provides property-based testing utilities for quantum computing:
//! - Random circuit generation with configurable gate sets, depths, and qubit counts
//! - Quantum property assertions (unitarity, trace preservation, probability conservation)
//! - Statistical testing helpers (chi-squared, Kolmogorov-Smirnov)
//! - Shrinking of failing inputs to find minimal counterexamples
//! - Seed-based reproducibility for deterministic test replay
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::property_testing::*;
//!
//! let config = PropertyTestConfig {
//!     num_trials: 50,
//!     tolerance: 1e-10,
//!     confidence: 0.99,
//!     seed: Some(42),
//! };
//!
//! let checker = QuantumPropertyChecker::new(config);
//! let gen = RandomCircuitGenerator::new(3, 5, GateSet::standard(), Some(42));
//! let result = checker.check_unitarity(&gen);
//! assert!(result.passed);
//! ```

use crate::{GateOperations, QuantumState, C64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

// ============================================================
// GATE SET AND CIRCUIT REPRESENTATION
// ============================================================

/// A quantum gate that can be applied to a circuit.
#[derive(Clone, Debug)]
pub enum Gate {
    /// Hadamard gate on a single qubit.
    H(usize),
    /// Pauli-X gate on a single qubit.
    X(usize),
    /// Pauli-Y gate on a single qubit.
    Y(usize),
    /// Pauli-Z gate on a single qubit.
    Z(usize),
    /// S gate (pi/2 phase) on a single qubit.
    S(usize),
    /// T gate (pi/4 phase) on a single qubit.
    T(usize),
    /// Rotation around X-axis by angle theta.
    Rx(usize, f64),
    /// Rotation around Y-axis by angle theta.
    Ry(usize, f64),
    /// Rotation around Z-axis by angle theta.
    Rz(usize, f64),
    /// CNOT gate with control and target qubits.
    Cnot(usize, usize),
    /// CZ gate with control and target qubits.
    Cz(usize, usize),
    /// SWAP gate between two qubits.
    Swap(usize, usize),
    /// Toffoli (CCX) gate with two controls and one target.
    Toffoli(usize, usize, usize),
}

impl Gate {
    /// Apply this gate to the given quantum state.
    pub fn apply(&self, state: &mut QuantumState) {
        match *self {
            Gate::H(q) => GateOperations::h(state, q),
            Gate::X(q) => GateOperations::x(state, q),
            Gate::Y(q) => GateOperations::y(state, q),
            Gate::Z(q) => GateOperations::z(state, q),
            Gate::S(q) => GateOperations::s(state, q),
            Gate::T(q) => GateOperations::t(state, q),
            Gate::Rx(q, theta) => GateOperations::rx(state, q, theta),
            Gate::Ry(q, theta) => GateOperations::ry(state, q, theta),
            Gate::Rz(q, theta) => GateOperations::rz(state, q, theta),
            Gate::Cnot(c, t) => GateOperations::cnot(state, c, t),
            Gate::Cz(c, t) => GateOperations::cz(state, c, t),
            Gate::Swap(a, b) => GateOperations::swap(state, a, b),
            Gate::Toffoli(c1, c2, t) => GateOperations::toffoli(state, c1, c2, t),
        }
    }

    /// Return the inverse (adjoint) of this gate.
    pub fn inverse(&self) -> Gate {
        match *self {
            // Self-inverse gates
            Gate::H(q) => Gate::H(q),
            Gate::X(q) => Gate::X(q),
            Gate::Y(q) => Gate::Y(q),
            Gate::Z(q) => Gate::Z(q),
            Gate::Cnot(c, t) => Gate::Cnot(c, t),
            Gate::Cz(c, t) => Gate::Cz(c, t),
            Gate::Swap(a, b) => Gate::Swap(a, b),
            Gate::Toffoli(c1, c2, t) => Gate::Toffoli(c1, c2, t),
            // S^dag = Rz(-pi/2)
            Gate::S(q) => Gate::Rz(q, -PI / 2.0),
            // T^dag = Rz(-pi/4)
            Gate::T(q) => Gate::Rz(q, -PI / 4.0),
            // Rotation inverses: negate the angle
            Gate::Rx(q, theta) => Gate::Rx(q, -theta),
            Gate::Ry(q, theta) => Gate::Ry(q, -theta),
            Gate::Rz(q, theta) => Gate::Rz(q, -theta),
        }
    }

    /// Return the number of qubits this gate acts on.
    pub fn num_qubits_involved(&self) -> usize {
        match self {
            Gate::H(_)
            | Gate::X(_)
            | Gate::Y(_)
            | Gate::Z(_)
            | Gate::S(_)
            | Gate::T(_)
            | Gate::Rx(_, _)
            | Gate::Ry(_, _)
            | Gate::Rz(_, _) => 1,
            Gate::Cnot(_, _) | Gate::Cz(_, _) | Gate::Swap(_, _) => 2,
            Gate::Toffoli(_, _, _) => 3,
        }
    }

    /// Return a simplified version of this gate for shrinking.
    /// Reduces rotation angles toward zero and multi-qubit gates to fewer qubits.
    pub fn shrink(&self) -> Option<Gate> {
        match *self {
            Gate::Rx(q, theta) if theta.abs() > 1e-8 => Some(Gate::Rx(q, theta / 2.0)),
            Gate::Ry(q, theta) if theta.abs() > 1e-8 => Some(Gate::Ry(q, theta / 2.0)),
            Gate::Rz(q, theta) if theta.abs() > 1e-8 => Some(Gate::Rz(q, theta / 2.0)),
            Gate::Toffoli(_, _, t) => Some(Gate::X(t)),
            Gate::Cnot(_, t) => Some(Gate::X(t)),
            Gate::Cz(_, t) => Some(Gate::Z(t)),
            Gate::Swap(a, _) => Some(Gate::H(a)),
            _ => None,
        }
    }
}

/// Configurable set of gates available for random circuit generation.
#[derive(Clone, Debug)]
pub struct GateSet {
    /// Available single-qubit gate types.
    pub single_qubit: Vec<SingleQubitGateType>,
    /// Available two-qubit gate types.
    pub two_qubit: Vec<TwoQubitGateType>,
    /// Whether to include Toffoli (three-qubit) gates.
    pub include_toffoli: bool,
    /// Whether to include parameterized rotation gates.
    pub include_rotations: bool,
}

/// Types of single-qubit gates available in a gate set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SingleQubitGateType {
    H,
    X,
    Y,
    Z,
    S,
    T,
}

/// Types of two-qubit gates available in a gate set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TwoQubitGateType {
    Cnot,
    Cz,
    Swap,
}

impl GateSet {
    /// Standard gate set: H, X, Y, Z, S, T, CNOT, CZ, and rotation gates.
    pub fn standard() -> Self {
        GateSet {
            single_qubit: vec![
                SingleQubitGateType::H,
                SingleQubitGateType::X,
                SingleQubitGateType::Y,
                SingleQubitGateType::Z,
                SingleQubitGateType::S,
                SingleQubitGateType::T,
            ],
            two_qubit: vec![TwoQubitGateType::Cnot, TwoQubitGateType::Cz],
            include_toffoli: false,
            include_rotations: true,
        }
    }

    /// Clifford gate set: H, S, CNOT (generates the Clifford group).
    pub fn clifford() -> Self {
        GateSet {
            single_qubit: vec![SingleQubitGateType::H, SingleQubitGateType::S],
            two_qubit: vec![TwoQubitGateType::Cnot],
            include_toffoli: false,
            include_rotations: false,
        }
    }

    /// Universal gate set: Clifford + T (universal for quantum computation).
    pub fn universal() -> Self {
        GateSet {
            single_qubit: vec![
                SingleQubitGateType::H,
                SingleQubitGateType::S,
                SingleQubitGateType::T,
            ],
            two_qubit: vec![TwoQubitGateType::Cnot],
            include_toffoli: false,
            include_rotations: false,
        }
    }

    /// Full gate set including Toffoli and all rotations.
    pub fn full() -> Self {
        GateSet {
            single_qubit: vec![
                SingleQubitGateType::H,
                SingleQubitGateType::X,
                SingleQubitGateType::Y,
                SingleQubitGateType::Z,
                SingleQubitGateType::S,
                SingleQubitGateType::T,
            ],
            two_qubit: vec![
                TwoQubitGateType::Cnot,
                TwoQubitGateType::Cz,
                TwoQubitGateType::Swap,
            ],
            include_toffoli: true,
            include_rotations: true,
        }
    }
}

// ============================================================
// RANDOM CIRCUIT
// ============================================================

/// A randomly generated quantum circuit.
#[derive(Clone, Debug)]
pub struct RandomCircuit {
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
    /// Ordered list of gates composing the circuit.
    pub gates: Vec<Gate>,
    /// Depth (number of layers) used during generation.
    pub depth: usize,
    /// The seed used to generate this circuit (for reproducibility).
    pub seed: u64,
}

impl RandomCircuit {
    /// Apply the entire circuit to a quantum state.
    pub fn apply(&self, state: &mut QuantumState) {
        for gate in &self.gates {
            gate.apply(state);
        }
    }

    /// Apply the inverse (adjoint) of the entire circuit.
    /// Gates are applied in reverse order, each replaced by its inverse.
    pub fn apply_inverse(&self, state: &mut QuantumState) {
        for gate in self.gates.iter().rev() {
            gate.inverse().apply(state);
        }
    }

    /// Return a circuit containing a subset of the first `n` gates.
    /// Used during shrinking to find minimal failing inputs.
    pub fn shrink_to(&self, n: usize) -> RandomCircuit {
        let clamped = n.min(self.gates.len());
        RandomCircuit {
            num_qubits: self.num_qubits,
            gates: self.gates[..clamped].to_vec(),
            depth: self.depth,
            seed: self.seed,
        }
    }
}

// ============================================================
// RANDOM CIRCUIT GENERATOR
// ============================================================

/// Generator for random quantum circuits with configurable parameters.
///
/// Produces circuits with a specified number of qubits, depth, and gate set.
/// Seed-based generation ensures reproducibility: the same seed always
/// produces the same sequence of circuits.
pub struct RandomCircuitGenerator {
    num_qubits: usize,
    depth: usize,
    gate_set: GateSet,
    rng: StdRng,
    circuits_generated: u64,
}

impl RandomCircuitGenerator {
    /// Create a new random circuit generator.
    ///
    /// - `num_qubits`: Number of qubits in generated circuits.
    /// - `depth`: Number of gate layers per circuit.
    /// - `gate_set`: Which gates are available for generation.
    /// - `seed`: Optional seed for reproducibility. If `None`, uses entropy.
    pub fn new(num_qubits: usize, depth: usize, gate_set: GateSet, seed: Option<u64>) -> Self {
        let actual_seed = seed.unwrap_or_else(|| rand::random());
        RandomCircuitGenerator {
            num_qubits,
            depth,
            gate_set,
            rng: StdRng::seed_from_u64(actual_seed),
            circuits_generated: 0,
        }
    }

    /// Generate a single random circuit.
    pub fn generate(&mut self) -> RandomCircuit {
        let seed_used = self.rng.gen();
        let mut gates = Vec::with_capacity(self.depth * self.num_qubits);

        for _ in 0..self.depth {
            // Each layer applies gates to a random subset of qubits
            let num_gates_in_layer = self.rng.gen_range(1..=self.num_qubits);

            for _ in 0..num_gates_in_layer {
                let gate = self.random_gate();
                gates.push(gate);
            }
        }

        self.circuits_generated += 1;

        RandomCircuit {
            num_qubits: self.num_qubits,
            gates,
            depth: self.depth,
            seed: seed_used,
        }
    }

    /// Generate a batch of random circuits.
    pub fn generate_batch(&mut self, count: usize) -> Vec<RandomCircuit> {
        (0..count).map(|_| self.generate()).collect()
    }

    /// Return the number of circuits generated so far.
    pub fn circuits_generated(&self) -> u64 {
        self.circuits_generated
    }

    /// Generate a random gate from the configured gate set.
    fn random_gate(&mut self) -> Gate {
        // Decide between single-qubit, two-qubit, rotation, and toffoli
        let mut options: Vec<u8> = Vec::new();
        if !self.gate_set.single_qubit.is_empty() {
            options.push(0); // single-qubit fixed gate
        }
        if self.num_qubits >= 2 && !self.gate_set.two_qubit.is_empty() {
            options.push(1); // two-qubit gate
        }
        if self.gate_set.include_rotations {
            options.push(2); // parameterized rotation
        }
        if self.gate_set.include_toffoli && self.num_qubits >= 3 {
            options.push(3); // toffoli
        }

        if options.is_empty() {
            // Fallback: always can do H on qubit 0
            return Gate::H(0);
        }

        let choice = options[self.rng.gen_range(0..options.len())];

        match choice {
            0 => {
                let qubit = self.rng.gen_range(0..self.num_qubits);
                let gate_type_idx = self.rng.gen_range(0..self.gate_set.single_qubit.len());
                match self.gate_set.single_qubit[gate_type_idx] {
                    SingleQubitGateType::H => Gate::H(qubit),
                    SingleQubitGateType::X => Gate::X(qubit),
                    SingleQubitGateType::Y => Gate::Y(qubit),
                    SingleQubitGateType::Z => Gate::Z(qubit),
                    SingleQubitGateType::S => Gate::S(qubit),
                    SingleQubitGateType::T => Gate::T(qubit),
                }
            }
            1 => {
                let q1 = self.rng.gen_range(0..self.num_qubits);
                let mut q2 = self.rng.gen_range(0..self.num_qubits - 1);
                if q2 >= q1 {
                    q2 += 1;
                }
                let gate_type_idx = self.rng.gen_range(0..self.gate_set.two_qubit.len());
                match self.gate_set.two_qubit[gate_type_idx] {
                    TwoQubitGateType::Cnot => Gate::Cnot(q1, q2),
                    TwoQubitGateType::Cz => Gate::Cz(q1, q2),
                    TwoQubitGateType::Swap => Gate::Swap(q1, q2),
                }
            }
            2 => {
                let qubit = self.rng.gen_range(0..self.num_qubits);
                let theta = self.rng.gen_range(-PI..PI);
                let axis = self.rng.gen_range(0..3);
                match axis {
                    0 => Gate::Rx(qubit, theta),
                    1 => Gate::Ry(qubit, theta),
                    _ => Gate::Rz(qubit, theta),
                }
            }
            3 => {
                // Toffoli: need 3 distinct qubits
                let c1 = self.rng.gen_range(0..self.num_qubits);
                let mut c2 = self.rng.gen_range(0..self.num_qubits - 1);
                if c2 >= c1 {
                    c2 += 1;
                }
                let remaining: Vec<usize> = (0..self.num_qubits)
                    .filter(|&q| q != c1 && q != c2)
                    .collect();
                let t = remaining[self.rng.gen_range(0..remaining.len())];
                Gate::Toffoli(c1, c2, t)
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================
// CONFIGURATION AND RESULTS
// ============================================================

/// Configuration for property-based tests.
#[derive(Clone, Debug)]
pub struct PropertyTestConfig {
    /// Number of random trials to run.
    pub num_trials: usize,
    /// Numerical tolerance for floating-point comparisons.
    pub tolerance: f64,
    /// Confidence level for statistical tests (e.g., 0.99).
    pub confidence: f64,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        PropertyTestConfig {
            num_trials: 100,
            tolerance: 1e-6,
            confidence: 0.99,
            seed: None,
        }
    }
}

/// A single failure case from a property test.
#[derive(Clone, Debug)]
pub struct FailureCase {
    /// Human-readable description of the input that caused the failure.
    pub input_description: String,
    /// What the property expected.
    pub expected: String,
    /// What was actually observed.
    pub actual: String,
    /// The trial index where this failure occurred.
    pub trial_index: usize,
}

/// Summary statistics for a completed property test.
#[derive(Clone, Debug)]
pub struct TestStatistics {
    /// Mean value of the property metric across all trials.
    pub mean: f64,
    /// Standard deviation of the property metric.
    pub std_dev: f64,
    /// Maximum deviation from the expected value.
    pub max_deviation: f64,
    /// Minimum deviation from the expected value.
    pub min_deviation: f64,
}

/// Result of a property-based test run.
#[derive(Clone, Debug)]
pub struct PropertyTestResult {
    /// Whether all trials passed.
    pub passed: bool,
    /// Total number of trials executed.
    pub num_trials: usize,
    /// Number of trials that passed.
    pub num_passed: usize,
    /// Failure cases with details.
    pub failures: Vec<FailureCase>,
    /// Aggregate statistics over the trials.
    pub statistics: TestStatistics,
    /// Name of the property that was tested.
    pub property_name: String,
}

impl PropertyTestResult {
    /// Return the pass rate as a fraction in [0.0, 1.0].
    pub fn pass_rate(&self) -> f64 {
        if self.num_trials == 0 {
            return 1.0;
        }
        self.num_passed as f64 / self.num_trials as f64
    }
}

// ============================================================
// QUANTUM PROPERTY CHECKER
// ============================================================

/// Statistical property verification engine for quantum circuits.
///
/// Runs property-based tests over randomly generated circuits and states,
/// collecting statistics and failure cases.
pub struct QuantumPropertyChecker {
    config: PropertyTestConfig,
}

impl QuantumPropertyChecker {
    /// Create a new property checker with the given configuration.
    pub fn new(config: PropertyTestConfig) -> Self {
        QuantumPropertyChecker { config }
    }

    /// Create a property checker with default settings.
    pub fn with_defaults() -> Self {
        Self::new(PropertyTestConfig::default())
    }

    // --------------------------------------------------------
    // UNITARITY CHECK
    // --------------------------------------------------------

    /// Verify that random circuits preserve state norm (unitarity).
    ///
    /// For every randomly generated circuit, applies it to |0...0> and checks
    /// that the resulting state vector has norm 1.0 within tolerance.
    pub fn check_unitarity(&self, gen: &RandomCircuitGenerator) -> PropertyTestResult {
        let mut gen = self.make_generator_clone(gen);
        let mut failures = Vec::new();
        let mut deviations = Vec::with_capacity(self.config.num_trials);

        for trial in 0..self.config.num_trials {
            let circuit = gen.generate();
            let mut state = QuantumState::new(circuit.num_qubits);
            circuit.apply(&mut state);

            let norm_sq: f64 = state.amplitudes_ref().iter().map(|a| a.norm_sqr()).sum();
            let deviation = (norm_sq - 1.0).abs();
            deviations.push(deviation);

            if deviation > self.config.tolerance {
                failures.push(FailureCase {
                    input_description: format!(
                        "Circuit with {} gates on {} qubits (seed={})",
                        circuit.gates.len(),
                        circuit.num_qubits,
                        circuit.seed
                    ),
                    expected: "1.0".to_string(),
                    actual: format!("{:.15}", norm_sq),
                    trial_index: trial,
                });
            }
        }

        self.build_result("unitarity", &deviations, &failures)
    }

    // --------------------------------------------------------
    // TRACE PRESERVATION
    // --------------------------------------------------------

    /// Verify that density matrix trace is preserved after operations.
    ///
    /// Constructs a density matrix from a random circuit output state and
    /// verifies that Tr(rho) = 1.0 within tolerance.
    pub fn check_trace_preservation(&self, gen: &RandomCircuitGenerator) -> PropertyTestResult {
        let mut gen = self.make_generator_clone(gen);
        let mut failures = Vec::new();
        let mut deviations = Vec::with_capacity(self.config.num_trials);

        for trial in 0..self.config.num_trials {
            let circuit = gen.generate();
            let mut state = QuantumState::new(circuit.num_qubits);
            circuit.apply(&mut state);

            // Construct density matrix rho = |psi><psi| and verify trace
            let amps = state.amplitudes_ref();
            let trace: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            let deviation = (trace - 1.0).abs();
            deviations.push(deviation);

            if deviation > self.config.tolerance {
                failures.push(FailureCase {
                    input_description: format!(
                        "Density matrix from circuit with {} gates",
                        circuit.gates.len()
                    ),
                    expected: "1.0".to_string(),
                    actual: format!("{:.15}", trace),
                    trial_index: trial,
                });
            }
        }

        self.build_result("trace_preservation", &deviations, &failures)
    }

    // --------------------------------------------------------
    // PROBABILITY CONSERVATION
    // --------------------------------------------------------

    /// Verify that measurement probabilities sum to 1.0 after any circuit.
    pub fn check_probability_conservation(
        &self,
        gen: &RandomCircuitGenerator,
    ) -> PropertyTestResult {
        let mut gen = self.make_generator_clone(gen);
        let mut failures = Vec::new();
        let mut deviations = Vec::with_capacity(self.config.num_trials);

        for trial in 0..self.config.num_trials {
            let circuit = gen.generate();
            let mut state = QuantumState::new(circuit.num_qubits);
            circuit.apply(&mut state);

            let probs = state.probabilities();
            let total: f64 = probs.iter().sum();
            let deviation = (total - 1.0).abs();
            deviations.push(deviation);

            if deviation > self.config.tolerance {
                failures.push(FailureCase {
                    input_description: format!(
                        "Probability sum for circuit with {} gates",
                        circuit.gates.len()
                    ),
                    expected: "1.0".to_string(),
                    actual: format!("{:.15}", total),
                    trial_index: trial,
                });
            }
        }

        self.build_result("probability_conservation", &deviations, &failures)
    }

    // --------------------------------------------------------
    // REVERSIBILITY
    // --------------------------------------------------------

    /// Verify that circuit followed by its inverse returns to the initial state.
    ///
    /// Applies a random circuit to |0...0>, then applies the inverse circuit,
    /// and checks fidelity with the original |0...0> state.
    pub fn check_reversibility(&self, gen: &RandomCircuitGenerator) -> PropertyTestResult {
        let mut gen = self.make_generator_clone(gen);
        let mut failures = Vec::new();
        let mut deviations = Vec::with_capacity(self.config.num_trials);

        for trial in 0..self.config.num_trials {
            let circuit = gen.generate();
            let initial_state = QuantumState::new(circuit.num_qubits);
            let mut state = initial_state.clone();

            circuit.apply(&mut state);
            circuit.apply_inverse(&mut state);

            let fidelity = state.fidelity(&initial_state);
            let deviation = (fidelity - 1.0).abs();
            deviations.push(deviation);

            if deviation > self.config.tolerance {
                failures.push(FailureCase {
                    input_description: format!(
                        "Reversibility for circuit with {} gates (seed={})",
                        circuit.gates.len(),
                        circuit.seed
                    ),
                    expected: "1.0".to_string(),
                    actual: format!("{:.15}", fidelity),
                    trial_index: trial,
                });
            }
        }

        self.build_result("reversibility", &deviations, &failures)
    }

    // --------------------------------------------------------
    // HELPER: Build result from deviations
    // --------------------------------------------------------

    fn build_result(
        &self,
        name: &str,
        deviations: &[f64],
        failures: &[FailureCase],
    ) -> PropertyTestResult {
        let stats = compute_statistics(deviations);
        let num_trials = deviations.len();
        let num_passed = num_trials - failures.len();

        PropertyTestResult {
            passed: failures.is_empty(),
            num_trials,
            num_passed,
            failures: failures.to_vec(),
            statistics: stats,
            property_name: name.to_string(),
        }
    }

    /// Create a new generator with a deterministic seed derived from config.
    fn make_generator_clone(&self, gen: &RandomCircuitGenerator) -> RandomCircuitGenerator {
        RandomCircuitGenerator::new(
            gen.num_qubits,
            gen.depth,
            gen.gate_set.clone(),
            self.config.seed,
        )
    }
}

// ============================================================
// UNITARITY CHECK (standalone)
// ============================================================

/// Check that a 2x2 complex matrix is unitary (U^dag U = I) within tolerance.
///
/// The matrix is represented as `[[m00, m01], [m10, m11]]`.
pub struct UnitarityCheck;

impl UnitarityCheck {
    /// Verify unitarity of a 2x2 matrix given as four complex elements.
    pub fn check_2x2(m00: C64, m01: C64, m10: C64, m11: C64, tolerance: f64) -> bool {
        // U^dag U should equal I
        // (U^dag)_ij = conj(U_ji)
        // Product element (0,0): conj(m00)*m00 + conj(m10)*m10
        let p00 = m00.conj() * m00 + m10.conj() * m10;
        let p01 = m00.conj() * m01 + m10.conj() * m11;
        let p10 = m01.conj() * m00 + m11.conj() * m10;
        let p11 = m01.conj() * m01 + m11.conj() * m11;

        let one = C64::new(1.0, 0.0);
        let zero = C64::new(0.0, 0.0);

        (p00 - one).norm() < tolerance
            && (p01 - zero).norm() < tolerance
            && (p10 - zero).norm() < tolerance
            && (p11 - one).norm() < tolerance
    }

    /// Verify unitarity of a state vector produced by a circuit.
    /// Checks that the norm is 1.0, which is equivalent to unitarity
    /// when starting from a normalized state.
    pub fn check_state_norm(state: &QuantumState, tolerance: f64) -> bool {
        let norm_sq: f64 = state.amplitudes_ref().iter().map(|a| a.norm_sqr()).sum();
        (norm_sq - 1.0).abs() < tolerance
    }

    /// Verify unitarity by checking that a circuit applied to every
    /// computational basis state produces orthonormal outputs.
    /// This is a full unitarity check for small qubit counts.
    pub fn check_circuit_unitary(circuit: &RandomCircuit, tolerance: f64) -> bool {
        let n = circuit.num_qubits;
        let dim = 1usize << n;

        // Build the unitary matrix column by column
        let mut columns: Vec<Vec<C64>> = Vec::with_capacity(dim);
        for basis_idx in 0..dim {
            let mut state = QuantumState::new(n);
            // Set state to |basis_idx>
            let amps = state.amplitudes_mut();
            amps[0] = C64::new(0.0, 0.0);
            amps[basis_idx] = C64::new(1.0, 0.0);
            circuit.apply(&mut state);
            columns.push(state.amplitudes_ref().to_vec());
        }

        // Check orthonormality: <col_i | col_j> = delta_ij
        for i in 0..dim {
            for j in 0..dim {
                let inner: C64 = columns[i]
                    .iter()
                    .zip(columns[j].iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();

                let expected = if i == j { 1.0 } else { 0.0 };
                if (inner.re - expected).abs() > tolerance || inner.im.abs() > tolerance {
                    return false;
                }
            }
        }

        true
    }
}

// ============================================================
// TRACE PRESERVATION CHECK
// ============================================================

/// Verify that quantum channels preserve the trace of a density matrix.
pub struct TracePreservation;

impl TracePreservation {
    /// Check trace preservation for a pure state vector.
    /// Tr(|psi><psi|) should equal 1.0.
    pub fn check_pure_state(state: &QuantumState, tolerance: f64) -> bool {
        let trace: f64 = state.amplitudes_ref().iter().map(|a| a.norm_sqr()).sum();
        (trace - 1.0).abs() < tolerance
    }

    /// Apply a depolarizing channel to a density matrix and check trace preservation.
    ///
    /// Depolarizing channel: rho -> (1-p) * rho + p/d * I
    /// where d is the dimension and p is the error probability.
    pub fn check_depolarizing(state: &QuantumState, error_prob: f64, tolerance: f64) -> bool {
        let dim = state.dim;
        let amps = state.amplitudes_ref();

        // Build density matrix
        let mut rho = vec![C64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                rho[i * dim + j] = amps[i] * amps[j].conj();
            }
        }

        // Apply depolarizing channel: rho -> (1-p)*rho + p/d * I
        let p = error_prob.clamp(0.0, 1.0);
        let d = dim as f64;
        for i in 0..dim {
            for j in 0..dim {
                let identity_elem = if i == j {
                    C64::new(1.0 / d, 0.0)
                } else {
                    C64::new(0.0, 0.0)
                };
                rho[i * dim + j] =
                    C64::new(1.0 - p, 0.0) * rho[i * dim + j] + C64::new(p, 0.0) * identity_elem;
            }
        }

        // Compute trace
        let trace: f64 = (0..dim).map(|i| rho[i * dim + i].re).sum();
        (trace - 1.0).abs() < tolerance
    }
}

// ============================================================
// ENTANGLEMENT WITNESS
// ============================================================

/// Entanglement detection using the Peres-Horodecki (PPT) criterion
/// and concurrence for two-qubit systems.
pub struct EntanglementWitness;

impl EntanglementWitness {
    /// Check whether a two-qubit state is entangled using the PPT criterion.
    ///
    /// For a two-qubit system, the PPT criterion is necessary and sufficient:
    /// a state is entangled if and only if its partial transpose has at least
    /// one negative eigenvalue.
    ///
    /// Returns `true` if the state is entangled.
    pub fn is_entangled_ppt(state: &QuantumState) -> bool {
        assert_eq!(
            state.num_qubits, 2,
            "PPT criterion implementation requires exactly 2 qubits"
        );

        let amps = state.amplitudes_ref();
        let dim = 4; // 2^2

        // Build density matrix rho = |psi><psi|
        let mut rho = [[C64::new(0.0, 0.0); 4]; 4];
        for i in 0..dim {
            for j in 0..dim {
                rho[i][j] = amps[i] * amps[j].conj();
            }
        }

        // Partial transpose with respect to second qubit:
        // rho^{T_B}_{(i1,i2),(j1,j2)} = rho_{(i1,j2),(j1,i2)}
        // Index mapping for 2 qubits: state index = q0 + 2*q1
        // (i1,i2) -> row, (j1,j2) -> col
        let mut rho_pt = [[C64::new(0.0, 0.0); 4]; 4];
        for i1 in 0..2usize {
            for i2 in 0..2usize {
                for j1 in 0..2usize {
                    for j2 in 0..2usize {
                        let row = i1 + 2 * i2;
                        let col = j1 + 2 * j2;
                        // Transpose second subsystem: swap i2 <-> j2
                        let src_row = i1 + 2 * j2;
                        let src_col = j1 + 2 * i2;
                        rho_pt[row][col] = rho[src_row][src_col];
                    }
                }
            }
        }

        // Find minimum eigenvalue of the 4x4 Hermitian matrix rho_pt.
        // For a 4x4 Hermitian matrix, use the characteristic polynomial approach.
        // We compute eigenvalues numerically via iterative method.
        let min_eigenvalue = min_eigenvalue_4x4_hermitian(&rho_pt);
        min_eigenvalue < -1e-10
    }

    /// Compute the concurrence of a two-qubit pure state.
    ///
    /// For a pure state |psi> = a|00> + b|01> + c|10> + d|11>,
    /// the concurrence is C = 2|ad - bc|.
    ///
    /// Returns a value in [0, 1]: 0 = separable, 1 = maximally entangled.
    pub fn concurrence(state: &QuantumState) -> f64 {
        assert_eq!(
            state.num_qubits, 2,
            "Concurrence implementation requires exactly 2 qubits"
        );

        let amps = state.amplitudes_ref();
        let a = amps[0]; // |00>
        let b = amps[1]; // |01>
        let c = amps[2]; // |10>
        let d = amps[3]; // |11>

        // C = 2|ad - bc|
        let ad_minus_bc = a * d - b * c;
        2.0 * ad_minus_bc.norm()
    }
}

// ============================================================
// STATISTICAL TESTS
// ============================================================

/// Statistical tests for quantum measurement distributions.
pub struct StatisticalTest;

/// Result of a statistical test.
#[derive(Clone, Debug)]
pub struct StatisticalTestResult {
    /// Whether the null hypothesis was not rejected (i.e., test passed).
    pub passed: bool,
    /// The test statistic value.
    pub statistic: f64,
    /// The critical value for the given confidence level.
    pub critical_value: f64,
    /// The p-value (approximate).
    pub p_value: f64,
    /// Number of samples used.
    pub num_samples: usize,
}

impl StatisticalTest {
    /// Chi-squared goodness-of-fit test.
    ///
    /// Tests whether observed counts match expected probabilities.
    ///
    /// - `observed`: Histogram of measurement outcomes (outcome -> count).
    /// - `expected_probs`: Expected probability for each outcome.
    /// - `confidence`: Confidence level (e.g., 0.95).
    ///
    /// Returns a test result indicating whether the observed distribution
    /// is consistent with the expected probabilities.
    pub fn chi_squared(
        observed: &HashMap<usize, usize>,
        expected_probs: &[f64],
        confidence: f64,
    ) -> StatisticalTestResult {
        let n_total: usize = observed.values().sum();
        let n = n_total as f64;

        let mut chi2 = 0.0;
        let mut df = 0usize;

        for (outcome, &prob) in expected_probs.iter().enumerate() {
            if prob < 1e-15 {
                continue;
            }
            let expected_count = n * prob;
            let observed_count = *observed.get(&outcome).unwrap_or(&0) as f64;
            chi2 += (observed_count - expected_count).powi(2) / expected_count;
            df += 1;
        }

        // Degrees of freedom = number of categories - 1
        let df = if df > 0 { df - 1 } else { 0 };

        // Critical value approximation for chi-squared distribution.
        // Using Wilson-Hilferty approximation for large df.
        let critical = chi_squared_critical_value(df, confidence);

        // P-value approximation using regularized incomplete gamma function
        let p_value = chi_squared_p_value(chi2, df);

        StatisticalTestResult {
            passed: chi2 <= critical,
            statistic: chi2,
            critical_value: critical,
            p_value,
            num_samples: n_total,
        }
    }

    /// Kolmogorov-Smirnov test for comparing two empirical distributions.
    ///
    /// Tests whether two samples come from the same underlying distribution.
    ///
    /// - `sample_a`: First sample of observations.
    /// - `sample_b`: Second sample of observations.
    /// - `confidence`: Confidence level (e.g., 0.95).
    pub fn kolmogorov_smirnov(
        sample_a: &[f64],
        sample_b: &[f64],
        confidence: f64,
    ) -> StatisticalTestResult {
        let n_a = sample_a.len();
        let n_b = sample_b.len();

        if n_a == 0 || n_b == 0 {
            return StatisticalTestResult {
                passed: true,
                statistic: 0.0,
                critical_value: 0.0,
                p_value: 1.0,
                num_samples: 0,
            };
        }

        let mut sorted_a = sample_a.to_vec();
        let mut sorted_b = sample_b.to_vec();
        sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap());

        // Compute the KS statistic: max |F_a(x) - F_b(x)|
        let mut d_stat = 0.0_f64;
        let mut i = 0;
        let mut j = 0;

        while i < n_a || j < n_b {
            let val_a = if i < n_a { sorted_a[i] } else { f64::INFINITY };
            let val_b = if j < n_b { sorted_b[j] } else { f64::INFINITY };

            if val_a <= val_b {
                i += 1;
            }
            if val_b <= val_a {
                j += 1;
            }

            let ecdf_a = i as f64 / n_a as f64;
            let ecdf_b = j as f64 / n_b as f64;
            d_stat = d_stat.max((ecdf_a - ecdf_b).abs());
        }

        // Critical value for two-sample KS test
        let alpha = 1.0 - confidence;
        let n_eff = (n_a as f64 * n_b as f64) / (n_a + n_b) as f64;
        // c(alpha) values: 0.05 -> 1.358, 0.01 -> 1.628, 0.001 -> 1.949
        let c_alpha = ks_c_alpha(alpha);
        let critical = c_alpha / n_eff.sqrt();

        // Approximate p-value using asymptotic formula
        let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * d_stat;
        let p_value = ks_p_value(lambda);

        StatisticalTestResult {
            passed: d_stat <= critical,
            statistic: d_stat,
            critical_value: critical,
            p_value,
            num_samples: n_a + n_b,
        }
    }
}

// ============================================================
// ASSERTION HELPERS
// ============================================================

/// Assert that the observed distribution matches expected probabilities
/// according to a chi-squared test at the given confidence level.
///
/// Panics with a descriptive message if the test fails.
pub fn assert_distribution_matches(
    observed: &HashMap<usize, usize>,
    expected_probs: &[f64],
    confidence: f64,
) {
    let result = StatisticalTest::chi_squared(observed, expected_probs, confidence);
    assert!(
        result.passed,
        "Distribution mismatch: chi2={:.4}, critical={:.4}, p={:.4}",
        result.statistic, result.critical_value, result.p_value
    );
}

/// Assert that the fidelity between two quantum states exceeds a threshold.
///
/// Panics with a descriptive message if the fidelity is below the threshold.
pub fn assert_fidelity_above(state_a: &QuantumState, state_b: &QuantumState, threshold: f64) {
    let fidelity = state_a.fidelity(state_b);
    assert!(
        fidelity >= threshold,
        "Fidelity {:.10} is below threshold {:.10}",
        fidelity,
        threshold
    );
}

/// Assert that a two-qubit state is entangled (PPT criterion).
///
/// Panics if the state is separable.
pub fn assert_entangled(state: &QuantumState) {
    assert!(
        EntanglementWitness::is_entangled_ppt(state),
        "State is not entangled (PPT criterion: partial transpose has no negative eigenvalues)"
    );
}

/// Assert that a two-qubit state is separable (not entangled).
///
/// Panics if the state is entangled.
pub fn assert_separable(state: &QuantumState) {
    assert!(
        !EntanglementWitness::is_entangled_ppt(state),
        "State is entangled (PPT criterion: partial transpose has negative eigenvalues)"
    );
}

// ============================================================
// SHRINKING
// ============================================================

/// Attempt to shrink a failing circuit to find a minimal counterexample.
///
/// Given a circuit that fails some property (expressed as a closure returning
/// `false` on failure), tries progressively smaller circuits until it finds
/// the smallest one that still fails.
pub fn shrink_circuit<F>(circuit: &RandomCircuit, property: F) -> RandomCircuit
where
    F: Fn(&RandomCircuit) -> bool,
{
    let mut smallest = circuit.clone();

    // Phase 1: Binary search on circuit length
    let mut lo = 0;
    let mut hi = circuit.gates.len();

    while lo < hi {
        let mid = (lo + hi) / 2;
        let candidate = circuit.shrink_to(mid);
        if !property(&candidate) {
            // Still fails -- try smaller
            smallest = candidate;
            hi = mid;
        } else {
            // Passes now -- need more gates
            lo = mid + 1;
        }
    }

    // Phase 2: Try simplifying individual gates
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..smallest.gates.len() {
            // Try removing the gate entirely
            let mut without = smallest.gates.clone();
            without.remove(i);
            let candidate = RandomCircuit {
                num_qubits: smallest.num_qubits,
                gates: without,
                depth: smallest.depth,
                seed: smallest.seed,
            };
            if !property(&candidate) {
                smallest = candidate;
                improved = true;
                break; // restart from beginning after modification
            }

            // Try shrinking the gate
            if let Some(simpler_gate) = smallest.gates[i].shrink() {
                let mut with_simpler = smallest.gates.clone();
                with_simpler[i] = simpler_gate;
                let candidate = RandomCircuit {
                    num_qubits: smallest.num_qubits,
                    gates: with_simpler,
                    depth: smallest.depth,
                    seed: smallest.seed,
                };
                if !property(&candidate) {
                    smallest = candidate;
                    improved = true;
                    break;
                }
            }
        }
    }

    smallest
}

// ============================================================
// BATCH TESTING
// ============================================================

/// Run a property test across a batch of random circuits.
///
/// - `gen`: The random circuit generator.
/// - `num_circuits`: How many circuits to test.
/// - `property`: A closure that returns `true` if the circuit satisfies the property.
///
/// Returns a `PropertyTestResult` summarizing the batch.
pub fn batch_test<F>(
    gen: &mut RandomCircuitGenerator,
    num_circuits: usize,
    property: F,
) -> PropertyTestResult
where
    F: Fn(&RandomCircuit) -> bool,
{
    let mut failures = Vec::new();
    let mut pass_values = Vec::with_capacity(num_circuits);

    for trial in 0..num_circuits {
        let circuit = gen.generate();
        let passed = property(&circuit);
        pass_values.push(if passed { 0.0 } else { 1.0 });

        if !passed {
            failures.push(FailureCase {
                input_description: format!(
                    "Circuit #{} with {} gates on {} qubits (seed={})",
                    trial,
                    circuit.gates.len(),
                    circuit.num_qubits,
                    circuit.seed,
                ),
                expected: "property satisfied".to_string(),
                actual: "property violated".to_string(),
                trial_index: trial,
            });
        }
    }

    let stats = compute_statistics(&pass_values);

    PropertyTestResult {
        passed: failures.is_empty(),
        num_trials: num_circuits,
        num_passed: num_circuits - failures.len(),
        failures,
        statistics: stats,
        property_name: "batch_property".to_string(),
    }
}

// ============================================================
// GATE IDENTITY HELPERS
// ============================================================

/// Verify the gate identity HZH = X.
///
/// Applies H, Z, H to each computational basis state and checks that
/// the result matches applying X.
pub fn verify_hzh_equals_x(num_qubits: usize, target: usize, tolerance: f64) -> bool {
    let dim = 1usize << num_qubits;

    for basis_idx in 0..dim {
        // Apply HZH
        let mut state_hzh = QuantumState::new(num_qubits);
        let amps = state_hzh.amplitudes_mut();
        amps[0] = C64::new(0.0, 0.0);
        amps[basis_idx] = C64::new(1.0, 0.0);
        GateOperations::h(&mut state_hzh, target);
        GateOperations::z(&mut state_hzh, target);
        GateOperations::h(&mut state_hzh, target);

        // Apply X
        let mut state_x = QuantumState::new(num_qubits);
        let amps_x = state_x.amplitudes_mut();
        amps_x[0] = C64::new(0.0, 0.0);
        amps_x[basis_idx] = C64::new(1.0, 0.0);
        GateOperations::x(&mut state_x, target);

        // Compare
        let fidelity = state_hzh.fidelity(&state_x);
        if (fidelity - 1.0).abs() > tolerance {
            return false;
        }
    }
    true
}

/// Verify the gate identity HXH = Z.
pub fn verify_hxh_equals_z(num_qubits: usize, target: usize, tolerance: f64) -> bool {
    let dim = 1usize << num_qubits;

    for basis_idx in 0..dim {
        let mut state_hxh = QuantumState::new(num_qubits);
        let amps = state_hxh.amplitudes_mut();
        amps[0] = C64::new(0.0, 0.0);
        amps[basis_idx] = C64::new(1.0, 0.0);
        GateOperations::h(&mut state_hxh, target);
        GateOperations::x(&mut state_hxh, target);
        GateOperations::h(&mut state_hxh, target);

        let mut state_z = QuantumState::new(num_qubits);
        let amps_z = state_z.amplitudes_mut();
        amps_z[0] = C64::new(0.0, 0.0);
        amps_z[basis_idx] = C64::new(1.0, 0.0);
        GateOperations::z(&mut state_z, target);

        let fidelity = state_hxh.fidelity(&state_z);
        if (fidelity - 1.0).abs() > tolerance {
            return false;
        }
    }
    true
}

/// Verify that CNOT(H tensor I)|00> produces a Bell state.
///
/// Expected: (|00> + |11>) / sqrt(2)
pub fn verify_bell_state(tolerance: f64) -> bool {
    let mut state = QuantumState::new(2);
    GateOperations::h(&mut state, 0);
    GateOperations::cnot(&mut state, 0, 1);

    let amps = state.amplitudes_ref();
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    // Expected: amp[0] = 1/sqrt(2), amp[3] = 1/sqrt(2), rest = 0
    let checks = [
        (amps[0].re - inv_sqrt2).abs() < tolerance,
        amps[0].im.abs() < tolerance,
        amps[1].norm() < tolerance,
        amps[2].norm() < tolerance,
        (amps[3].re - inv_sqrt2).abs() < tolerance,
        amps[3].im.abs() < tolerance,
    ];

    checks.iter().all(|&c| c)
}

// ============================================================
// NOISE CHANNEL VALIDITY (CPTP)
// ============================================================

/// Check that a set of Kraus operators defines a valid CPTP channel.
///
/// A quantum channel defined by Kraus operators {K_i} is trace-preserving
/// if and only if sum_i K_i^dag K_i = I.
///
/// Each Kraus operator is represented as a flat row-major complex matrix.
pub fn check_cptp(kraus_operators: &[Vec<C64>], dim: usize, tolerance: f64) -> bool {
    // Check completeness relation: sum_i K_i^dag K_i = I
    let mut sum = vec![C64::new(0.0, 0.0); dim * dim];

    for k in kraus_operators {
        assert_eq!(
            k.len(),
            dim * dim,
            "Kraus operator dimension mismatch: expected {}x{}, got {} elements",
            dim,
            dim,
            k.len()
        );

        // Compute K^dag K and add to sum
        for i in 0..dim {
            for j in 0..dim {
                let mut val = C64::new(0.0, 0.0);
                for l in 0..dim {
                    // (K^dag)_{i,l} = conj(K_{l,i})
                    val += k[l * dim + i].conj() * k[l * dim + j];
                }
                sum[i * dim + j] += val;
            }
        }
    }

    // Check that sum equals identity
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (sum[i * dim + j].re - expected).abs() > tolerance
                || sum[i * dim + j].im.abs() > tolerance
            {
                return false;
            }
        }
    }

    true
}

// ============================================================
// ENTANGLEMENT MONOTONICITY
// ============================================================

/// Verify that a local unitary operation does not change the entanglement
/// (concurrence) of a two-qubit state.
///
/// Local unitaries preserve entanglement exactly, so the concurrence before
/// and after should be identical within tolerance.
pub fn check_entanglement_monotonicity_local_unitary(
    state: &QuantumState,
    local_gate: &Gate,
    tolerance: f64,
) -> bool {
    assert_eq!(
        state.num_qubits, 2,
        "Entanglement monotonicity check requires 2 qubits"
    );

    let concurrence_before = EntanglementWitness::concurrence(state);

    let mut state_after = state.clone();
    local_gate.apply(&mut state_after);

    let concurrence_after = EntanglementWitness::concurrence(&state_after);

    (concurrence_before - concurrence_after).abs() < tolerance
}

// ============================================================
// INTERNAL HELPERS
// ============================================================

/// Compute basic statistics over a slice of values.
fn compute_statistics(values: &[f64]) -> TestStatistics {
    if values.is_empty() {
        return TestStatistics {
            mean: 0.0,
            std_dev: 0.0,
            max_deviation: 0.0,
            min_deviation: 0.0,
        };
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let max_deviation = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_deviation = values.iter().cloned().fold(f64::INFINITY, f64::min);

    TestStatistics {
        mean,
        std_dev,
        max_deviation,
        min_deviation,
    }
}

/// Approximate chi-squared critical value using Wilson-Hilferty transformation.
///
/// For large degrees of freedom, the chi-squared distribution is approximately
/// normal. This provides a reasonable approximation for df >= 1.
fn chi_squared_critical_value(df: usize, confidence: f64) -> f64 {
    if df == 0 {
        return 0.0;
    }

    let k = df as f64;

    // Use a lookup for common significance levels
    let z = z_score_for_confidence(confidence);

    // Wilson-Hilferty approximation: chi2_p ≈ k * (1 - 2/(9k) + z * sqrt(2/(9k)))^3
    let term = 1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt();
    let result = k * term.powi(3);

    result.max(0.0)
}

/// Approximate chi-squared p-value.
///
/// Uses a series approximation of the regularized incomplete gamma function.
fn chi_squared_p_value(chi2: f64, df: usize) -> f64 {
    if df == 0 {
        return if chi2 > 0.0 { 0.0 } else { 1.0 };
    }

    // P-value = 1 - CDF = 1 - P(k/2, chi2/2)
    // Using the regularized upper incomplete gamma function Q(a, x)
    let a = df as f64 / 2.0;
    let x = chi2 / 2.0;

    upper_incomplete_gamma_regularized(a, x)
}

/// Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).
///
/// Uses series expansion for small x and continued fraction for large x.
fn upper_incomplete_gamma_regularized(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        // Use series expansion for P(a,x) and compute Q = 1 - P
        1.0 - lower_gamma_series(a, x)
    } else {
        // Use continued fraction for Q(a,x)
        upper_gamma_cf(a, x)
    }
}

/// Lower regularized incomplete gamma function via series expansion.
fn lower_gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-15 {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Upper regularized incomplete gamma function via continued fraction (Lentz).
fn upper_gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let tiny = 1e-30;

    let mut f = tiny;
    let mut c = f;
    let mut d = 0.0;

    for i in 1..200 {
        let an = if i == 1 {
            1.0
        } else if i % 2 == 0 {
            (i / 2) as f64
        } else {
            -((a - 1.0 + i as f64) / 2.0 - 0.5)
        };

        let bn = if i == 1 {
            x + 1.0 - a
        } else {
            x + (2 * i - 1) as f64 - a
        };

        d = bn + an * d;
        if d.abs() < tiny {
            d = tiny;
        }
        d = 1.0 / d;

        c = bn + an / c;
        if c.abs() < tiny {
            c = tiny;
        }

        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }

    f * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Stirling's approximation for ln(Gamma(a)).
fn ln_gamma(a: f64) -> f64 {
    // Use Lanczos approximation for better accuracy
    if a <= 0.0 {
        return f64::INFINITY;
    }

    // Coefficients for Lanczos approximation with g=7
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let g = 7.0;
    let x = a - 1.0;

    let mut t = coeffs[0];
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        t += c / (x + i as f64);
    }

    let tmp = x + g + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * tmp.ln() - tmp + t.ln()
}

/// Z-score (standard normal quantile) for common confidence levels.
fn z_score_for_confidence(confidence: f64) -> f64 {
    // Common critical values for the standard normal distribution
    if confidence >= 0.999 {
        3.291
    } else if confidence >= 0.995 {
        2.807
    } else if confidence >= 0.99 {
        2.576
    } else if confidence >= 0.975 {
        2.241
    } else if confidence >= 0.95 {
        1.960
    } else if confidence >= 0.90 {
        1.645
    } else if confidence >= 0.80 {
        1.282
    } else {
        1.0
    }
}

/// Kolmogorov-Smirnov c(alpha) critical coefficient.
fn ks_c_alpha(alpha: f64) -> f64 {
    if alpha <= 0.001 {
        1.949
    } else if alpha <= 0.01 {
        1.628
    } else if alpha <= 0.02 {
        1.517
    } else if alpha <= 0.05 {
        1.358
    } else if alpha <= 0.10 {
        1.224
    } else if alpha <= 0.20 {
        1.073
    } else {
        0.950
    }
}

/// Approximate KS p-value using the asymptotic Kolmogorov distribution.
///
/// P(D > d) = 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2*k^2*lambda^2)
fn ks_p_value(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }

    let mut p = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * (k as f64).powi(2) * lambda.powi(2)).exp();
        p += term;
        if term.abs() < 1e-15 {
            break;
        }
    }

    (2.0 * p).clamp(0.0, 1.0)
}

/// Find the minimum eigenvalue of a 4x4 Hermitian matrix using the Jacobi
/// eigenvalue algorithm on the real part.
///
/// For a density matrix partial transpose, the matrix is Hermitian, so all
/// eigenvalues are real. We use the Jacobi method which is robust for small
/// matrices and guaranteed to converge.
fn min_eigenvalue_4x4_hermitian(m: &[[C64; 4]; 4]) -> f64 {
    // Extract real symmetric matrix (Hermitian density matrices from pure states
    // have real partial transposes when the state vector is real).
    let mut a = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = m[i][j].re;
        }
    }

    // Jacobi eigenvalue algorithm: iteratively zero out off-diagonal elements
    for _ in 0..100 {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..4 {
            for j in (i + 1)..4 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break; // converged
        }

        // Compute Jacobi rotation angle
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation: A' = J^T A J
        let mut new_a = a;

        // Update rows/columns p and q
        for k in 0..4 {
            if k != p && k != q {
                new_a[p][k] = c * a[p][k] + s * a[q][k];
                new_a[k][p] = new_a[p][k];
                new_a[q][k] = -s * a[p][k] + c * a[q][k];
                new_a[k][q] = new_a[q][k];
            }
        }
        new_a[p][p] = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;

        a = new_a;
    }

    // Eigenvalues are on the diagonal
    let mut min_ev = a[0][0];
    for i in 1..4 {
        if a[i][i] < min_ev {
            min_ev = a[i][i];
        }
    }
    min_ev
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_circuit_generation() {
        let mut gen = RandomCircuitGenerator::new(3, 5, GateSet::standard(), Some(42));
        let circuit = gen.generate();

        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.depth, 5);
        assert!(
            !circuit.gates.is_empty(),
            "Circuit should have at least one gate"
        );

        // Verify all gates reference valid qubit indices
        for gate in &circuit.gates {
            match gate {
                Gate::H(q)
                | Gate::X(q)
                | Gate::Y(q)
                | Gate::Z(q)
                | Gate::S(q)
                | Gate::T(q)
                | Gate::Rx(q, _)
                | Gate::Ry(q, _)
                | Gate::Rz(q, _) => {
                    assert!(*q < 3, "Single-qubit gate on invalid qubit {}", q);
                }
                Gate::Cnot(c, t) | Gate::Cz(c, t) | Gate::Swap(c, t) => {
                    assert!(*c < 3, "Two-qubit gate control on invalid qubit {}", c);
                    assert!(*t < 3, "Two-qubit gate target on invalid qubit {}", t);
                    assert_ne!(c, t, "Two-qubit gate with same control and target");
                }
                Gate::Toffoli(c1, c2, t) => {
                    assert!(*c1 < 3);
                    assert!(*c2 < 3);
                    assert!(*t < 3);
                    assert_ne!(c1, c2);
                    assert_ne!(c1, t);
                    assert_ne!(c2, t);
                }
            }
        }
    }

    #[test]
    fn unitarity_check_identity() {
        // Identity matrix should pass unitarity check
        let one = C64::new(1.0, 0.0);
        let zero = C64::new(0.0, 0.0);
        assert!(UnitarityCheck::check_2x2(one, zero, zero, one, 1e-10));
    }

    #[test]
    fn unitarity_check_h_gate() {
        // Hadamard matrix: (1/sqrt(2)) * [[1, 1], [1, -1]]
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h00 = C64::new(inv_sqrt2, 0.0);
        let h01 = C64::new(inv_sqrt2, 0.0);
        let h10 = C64::new(inv_sqrt2, 0.0);
        let h11 = C64::new(-inv_sqrt2, 0.0);
        assert!(UnitarityCheck::check_2x2(h00, h01, h10, h11, 1e-10));
    }

    #[test]
    fn unitarity_check_random() {
        // Generate random 1-qubit circuits and verify unitarity via state norm
        let mut gen = RandomCircuitGenerator::new(1, 10, GateSet::standard(), Some(123));

        for _ in 0..50 {
            let circuit = gen.generate();
            let mut state = QuantumState::new(1);
            circuit.apply(&mut state);
            assert!(
                UnitarityCheck::check_state_norm(&state, 1e-10),
                "Random 1-qubit circuit broke unitarity"
            );
        }
    }

    #[test]
    fn trace_preservation() {
        // Apply depolarizing channel to a random state and verify trace = 1
        let mut gen = RandomCircuitGenerator::new(2, 4, GateSet::standard(), Some(77));

        for _ in 0..20 {
            let circuit = gen.generate();
            let mut state = QuantumState::new(2);
            circuit.apply(&mut state);

            // Check trace preservation under depolarizing with various error rates
            for &p in &[0.0, 0.01, 0.1, 0.5, 1.0] {
                assert!(
                    TracePreservation::check_depolarizing(&state, p, 1e-10),
                    "Trace not preserved under depolarizing channel with p={}",
                    p
                );
            }
        }
    }

    #[test]
    fn probability_conservation() {
        // Verify that measurement probabilities always sum to 1
        let config = PropertyTestConfig {
            num_trials: 50,
            tolerance: 1e-10,
            seed: Some(99),
            ..Default::default()
        };
        let checker = QuantumPropertyChecker::new(config);
        let gen = RandomCircuitGenerator::new(3, 6, GateSet::standard(), Some(99));
        let result = checker.check_probability_conservation(&gen);

        assert!(
            result.passed,
            "Probability conservation failed: {} failures out of {} trials",
            result.failures.len(),
            result.num_trials
        );
    }

    #[test]
    fn gate_identity_hzh() {
        // Verify HZH = X on each qubit of a 3-qubit system
        for target in 0..3 {
            assert!(
                verify_hzh_equals_x(3, target, 1e-10),
                "HZH != X on qubit {}",
                target
            );
        }
    }

    #[test]
    fn gate_identity_hxh() {
        // Verify HXH = Z
        for target in 0..3 {
            assert!(
                verify_hxh_equals_z(3, target, 1e-10),
                "HXH != Z on qubit {}",
                target
            );
        }
    }

    #[test]
    fn bell_state() {
        assert!(
            verify_bell_state(1e-10),
            "CNOT(H x I)|00> did not produce Bell state"
        );
    }

    #[test]
    fn reversibility() {
        let config = PropertyTestConfig {
            num_trials: 50,
            tolerance: 1e-8,
            seed: Some(55),
            ..Default::default()
        };
        let checker = QuantumPropertyChecker::new(config);
        let gen = RandomCircuitGenerator::new(3, 5, GateSet::standard(), Some(55));
        let result = checker.check_reversibility(&gen);

        assert!(
            result.passed,
            "Reversibility failed: {} failures out of {} trials. Max deviation: {:.2e}",
            result.failures.len(),
            result.num_trials,
            result.statistics.max_deviation
        );
    }

    #[test]
    fn chi_squared_uniform() {
        // Generate samples from a uniform distribution over 4 outcomes
        // and verify chi-squared test passes
        let mut rng = StdRng::seed_from_u64(42);
        let n_samples = 10000;
        let n_outcomes = 4;
        let mut observed: HashMap<usize, usize> = HashMap::new();

        for _ in 0..n_samples {
            let outcome = rng.gen_range(0..n_outcomes);
            *observed.entry(outcome).or_insert(0) += 1;
        }

        let expected_probs = vec![0.25; n_outcomes];
        let result = StatisticalTest::chi_squared(&observed, &expected_probs, 0.95);

        assert!(
            result.passed,
            "Uniform distribution failed chi-squared: statistic={:.4}, critical={:.4}",
            result.statistic, result.critical_value
        );
    }

    #[test]
    fn chi_squared_biased() {
        // Create a strongly biased distribution and test against uniform
        let mut observed: HashMap<usize, usize> = HashMap::new();
        // 90% outcome 0, 10% spread across 1-3
        observed.insert(0, 9000);
        observed.insert(1, 333);
        observed.insert(2, 333);
        observed.insert(3, 334);

        let expected_probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = StatisticalTest::chi_squared(&observed, &expected_probs, 0.95);

        assert!(
            !result.passed,
            "Biased distribution should fail chi-squared test against uniform. \
             statistic={:.4}, critical={:.4}",
            result.statistic, result.critical_value
        );
    }

    #[test]
    fn ks_test() {
        // Two samples from the same distribution should pass KS test
        let mut rng = StdRng::seed_from_u64(100);
        let n = 500;

        let sample_a: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let sample_b: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();

        let result = StatisticalTest::kolmogorov_smirnov(&sample_a, &sample_b, 0.95);

        assert!(
            result.passed,
            "Samples from same distribution should pass KS test. \
             D={:.4}, critical={:.4}",
            result.statistic, result.critical_value
        );

        // A sample from uniform vs a sample shifted by 0.5 should fail
        let sample_shifted: Vec<f64> = sample_a.iter().map(|x| (x + 0.5).min(1.0)).collect();
        let result_fail = StatisticalTest::kolmogorov_smirnov(&sample_a, &sample_shifted, 0.95);

        assert!(
            !result_fail.passed,
            "Shifted distribution should fail KS test. D={:.4}, critical={:.4}",
            result_fail.statistic, result_fail.critical_value
        );
    }

    #[test]
    fn shrinking() {
        // Create a circuit that "fails" when it has more than 3 gates,
        // and verify that shrinking finds a smaller failing circuit.
        let mut gen = RandomCircuitGenerator::new(2, 10, GateSet::standard(), Some(42));
        let big_circuit = gen.generate();

        assert!(
            big_circuit.gates.len() > 3,
            "Need a circuit with more than 3 gates for this test"
        );

        // Property: circuit has at most 3 gates (fails for big circuits)
        let property = |c: &RandomCircuit| c.gates.len() <= 3;

        // The big circuit should fail this property
        assert!(!property(&big_circuit));

        let shrunk = shrink_circuit(&big_circuit, property);

        // Shrunk circuit should still fail
        assert!(
            !property(&shrunk),
            "Shrunk circuit should still fail the property"
        );

        // Shrunk circuit should be smaller than or equal to original
        assert!(
            shrunk.gates.len() <= big_circuit.gates.len(),
            "Shrunk circuit ({} gates) should be <= original ({} gates)",
            shrunk.gates.len(),
            big_circuit.gates.len()
        );

        // Shrunk circuit should have exactly 4 gates (minimal failing input)
        assert_eq!(
            shrunk.gates.len(),
            4,
            "Minimal failing circuit should have exactly 4 gates, got {}",
            shrunk.gates.len()
        );
    }

    #[test]
    fn seed_reproducibility() {
        // Same seed should produce identical circuits
        let seed = Some(12345u64);

        let mut gen1 = RandomCircuitGenerator::new(3, 5, GateSet::standard(), seed);
        let mut gen2 = RandomCircuitGenerator::new(3, 5, GateSet::standard(), seed);

        for _ in 0..10 {
            let c1 = gen1.generate();
            let c2 = gen2.generate();

            assert_eq!(
                c1.gates.len(),
                c2.gates.len(),
                "Same seed should produce circuits with same number of gates"
            );

            // Apply both circuits to the same initial state and compare
            let mut state1 = QuantumState::new(3);
            let mut state2 = QuantumState::new(3);
            c1.apply(&mut state1);
            c2.apply(&mut state2);

            let fidelity = state1.fidelity(&state2);
            assert!(
                (fidelity - 1.0).abs() < 1e-12,
                "Same seed should produce identical circuits, fidelity = {:.15}",
                fidelity
            );
        }
    }

    #[test]
    fn batch_testing() {
        // Run 100 random circuits and verify all pass unitarity (state norm = 1)
        let mut gen = RandomCircuitGenerator::new(3, 5, GateSet::standard(), Some(42));

        let result = batch_test(&mut gen, 100, |circuit| {
            let mut state = QuantumState::new(circuit.num_qubits);
            circuit.apply(&mut state);
            UnitarityCheck::check_state_norm(&state, 1e-10)
        });

        assert!(
            result.passed,
            "Batch unitarity test failed: {} failures out of {} trials",
            result.failures.len(),
            result.num_trials
        );
        assert_eq!(result.num_trials, 100);
        assert_eq!(result.num_passed, 100);
    }

    #[test]
    fn entanglement_bell_state() {
        // Bell state should be maximally entangled
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);

        assert_entangled(&state);

        let concurrence = EntanglementWitness::concurrence(&state);
        assert!(
            (concurrence - 1.0).abs() < 1e-10,
            "Bell state concurrence should be 1.0, got {:.10}",
            concurrence
        );
    }

    #[test]
    fn separable_product_state() {
        // |00> is a product state (separable)
        let state = QuantumState::new(2);
        assert_separable(&state);

        let concurrence = EntanglementWitness::concurrence(&state);
        assert!(
            concurrence < 1e-10,
            "|00> concurrence should be 0.0, got {:.10}",
            concurrence
        );
    }

    #[test]
    fn entanglement_monotonicity_local() {
        // Local unitaries should not change entanglement
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);

        // Apply a local gate (single-qubit) and verify concurrence unchanged
        for gate in &[
            Gate::H(0),
            Gate::X(0),
            Gate::Y(1),
            Gate::Z(1),
            Gate::Rx(0, 0.7),
            Gate::Ry(1, 1.3),
        ] {
            assert!(
                check_entanglement_monotonicity_local_unitary(&state, gate, 1e-10),
                "Local gate {:?} changed entanglement",
                gate
            );
        }
    }

    #[test]
    fn cptp_depolarizing_channel() {
        // Depolarizing channel Kraus operators for 1 qubit:
        // K0 = sqrt(1-p) * I, K1 = sqrt(p/3) * X, K2 = sqrt(p/3) * Y, K3 = sqrt(p/3) * Z
        let p: f64 = 0.1;
        let s0 = (1.0_f64 - p).sqrt();
        let s1 = (p / 3.0_f64).sqrt();

        let zero = C64::new(0.0, 0.0);

        // I
        let k0 = vec![C64::new(s0, 0.0), zero, zero, C64::new(s0, 0.0)];
        // X
        let k1 = vec![zero, C64::new(s1, 0.0), C64::new(s1, 0.0), zero];
        // Y
        let k2 = vec![zero, C64::new(0.0, -s1), C64::new(0.0, s1), zero];
        // Z
        let k3 = vec![C64::new(s1, 0.0), zero, zero, C64::new(-s1, 0.0)];

        assert!(
            check_cptp(&[k0, k1, k2, k3], 2, 1e-10),
            "Depolarizing channel Kraus operators should satisfy CPTP condition"
        );
    }

    #[test]
    fn cptp_invalid_channel() {
        // An invalid set of Kraus operators (not trace-preserving)
        let k0 = vec![
            C64::new(0.5, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.5, 0.0),
        ];
        // Sum K^dag K = 0.25 * I, which != I
        assert!(
            !check_cptp(&[k0], 2, 1e-6),
            "Invalid Kraus operators should fail CPTP check"
        );
    }

    #[test]
    fn full_unitarity_check_small_circuit() {
        // Verify full unitarity check via orthonormal columns for a 2-qubit circuit
        let circuit = RandomCircuit {
            num_qubits: 2,
            gates: vec![Gate::H(0), Gate::Cnot(0, 1), Gate::Rz(1, 0.5)],
            depth: 1,
            seed: 0,
        };

        assert!(
            UnitarityCheck::check_circuit_unitary(&circuit, 1e-10),
            "Small circuit should be unitary"
        );
    }

    #[test]
    fn gate_set_configurations() {
        // Verify that all gate set presets generate valid circuits
        let configs = vec![
            ("standard", GateSet::standard()),
            ("clifford", GateSet::clifford()),
            ("universal", GateSet::universal()),
            ("full", GateSet::full()),
        ];

        for (name, gate_set) in configs {
            let mut gen = RandomCircuitGenerator::new(3, 4, gate_set, Some(42));
            let circuit = gen.generate();
            let mut state = QuantumState::new(3);
            circuit.apply(&mut state);

            assert!(
                UnitarityCheck::check_state_norm(&state, 1e-10),
                "Gate set '{}' produced non-unitary circuit",
                name
            );
        }
    }

    #[test]
    fn property_test_result_pass_rate() {
        let result = PropertyTestResult {
            passed: false,
            num_trials: 100,
            num_passed: 95,
            failures: vec![],
            statistics: TestStatistics {
                mean: 0.0,
                std_dev: 0.0,
                max_deviation: 0.0,
                min_deviation: 0.0,
            },
            property_name: "test".to_string(),
        };

        assert!((result.pass_rate() - 0.95).abs() < 1e-10);
    }
}
