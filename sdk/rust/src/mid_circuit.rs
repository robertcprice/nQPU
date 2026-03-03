//! Mid-Circuit Measurement and Classical Control
//!
//! Extends quantum simulation with:
//! - Mid-circuit measurement (measure without destroying state)
//! - Reset operations (reset qubit to |0⟩)
//! - Conditional gates (execute based on classical register)
//! - Classical register for storing measurement results
//!
//! # Usage
//!
//! ```ignore
//! use nqpu_metal::mid_circuit::{QuantumStateWithMeasurements, Operation};
//!
//! let mut sim = QuantumStateWithMeasurements::new(5);
//!
//! // Create circuit with mid-circuit measurement
//! let ops = vec![
//!     Operation::H { target: 0 },
//!     Operation::Measure { qubit: 0, bit: 0 },
//!     Operation::ConditionalGate {
//!         gate: Box::new(Operation::X { target: 1 }),
//!         condition: ClassicalCondition::BitSet(0),
//!     },
//! ];
//!
//! sim.execute(&ops);
//! ```

use crate::gates::Gate;
use crate::{GateOperations, QuantumState};
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};

/// Single-qubit Pauli basis used by Pauli-product measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliBasis {
    X,
    Y,
    Z,
}

/// Classical condition for conditional gates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassicalCondition {
    /// Single bit equals 1
    BitSet(usize),
    /// Single bit equals 0
    BitClear(usize),
    /// Single bit equals specific value
    BitEquals(usize, bool),
    /// Multiple bits AND together
    And(Vec<ClassicalCondition>),
    /// Multiple bits OR together
    Or(Vec<ClassicalCondition>),
    /// Negation of condition
    Not(Box<ClassicalCondition>),
}

impl ClassicalCondition {
    /// Evaluate condition against classical register.
    pub fn evaluate(&self, register: &[bool]) -> bool {
        match self {
            ClassicalCondition::BitSet(idx) => register.get(*idx).copied().unwrap_or(false),
            ClassicalCondition::BitClear(idx) => !register.get(*idx).copied().unwrap_or(true),
            ClassicalCondition::BitEquals(idx, value) => {
                register.get(*idx).copied().unwrap_or(false) == *value
            }
            ClassicalCondition::And(conds) => conds.iter().all(|c| c.evaluate(register)),
            ClassicalCondition::Or(conds) => conds.iter().any(|c| c.evaluate(register)),
            ClassicalCondition::Not(cond) => !cond.evaluate(register),
        }
    }

    /// Create a condition for "any of these bits are set"
    pub fn any_bit_set(bits: &[usize]) -> Self {
        if bits.len() == 1 {
            ClassicalCondition::BitSet(bits[0])
        } else {
            ClassicalCondition::Or(
                bits.iter()
                    .map(|&b| ClassicalCondition::BitSet(b))
                    .collect(),
            )
        }
    }

    /// Create a condition for "all of these bits are set"
    pub fn all_bits_set(bits: &[usize]) -> Self {
        if bits.len() == 1 {
            ClassicalCondition::BitSet(bits[0])
        } else {
            ClassicalCondition::And(
                bits.iter()
                    .map(|&b| ClassicalCondition::BitSet(b))
                    .collect(),
            )
        }
    }
}

/// Circuit operation (gate, measurement, reset, or conditional).
#[derive(Debug, Clone)]
pub enum Operation {
    /// Quantum gate
    Gate(Gate),

    /// Single-qubit measurement, stores result in classical bit
    Measure { qubit: usize, bit: usize },

    /// Multi-qubit Pauli-product measurement, stores parity in classical bit.
    ///
    /// Each entry is `(qubit, basis)`, e.g. `[(0, Z), (1, Z)]` for `Z0*Z1`.
    MeasurePauliProduct {
        paulis: Vec<(usize, PauliBasis)>,
        bit: usize,
    },

    /// Reset qubit to |0⟩ (based on measurement)
    Reset { qubit: usize },

    /// Conditional gate (only executes if condition is true)
    ConditionalGate {
        gate: Box<Operation>,
        condition: ClassicalCondition,
    },

    /// Barrier (prevents reordering across this point)
    Barrier { qubits: Vec<usize> },
}

impl Operation {
    /// Create a Hadamard gate operation
    pub fn h(target: usize) -> Self {
        Operation::Gate(Gate::h(target))
    }

    /// Create an X gate operation
    pub fn x(target: usize) -> Self {
        Operation::Gate(Gate::x(target))
    }

    /// Create a CNOT gate operation
    pub fn cnot(control: usize, target: usize) -> Self {
        Operation::Gate(Gate::cnot(control, target))
    }

    /// Create a measurement operation
    pub fn measure(qubit: usize, bit: usize) -> Self {
        Operation::Measure { qubit, bit }
    }

    /// Create a Pauli-product measurement operation.
    pub fn measure_pauli_product(paulis: Vec<(usize, PauliBasis)>, bit: usize) -> Self {
        Operation::MeasurePauliProduct { paulis, bit }
    }

    /// Create a reset operation
    pub fn reset(qubit: usize) -> Self {
        Operation::Reset { qubit }
    }

    /// Create a conditional gate operation
    pub fn conditional(gate: Operation, condition: ClassicalCondition) -> Self {
        Operation::ConditionalGate {
            gate: Box::new(gate),
            condition,
        }
    }

    /// Check if operation is a gate
    pub fn is_gate(&self) -> bool {
        matches!(self, Operation::Gate(_))
    }

    /// Get qubits involved in this operation
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Operation::Gate(gate) => {
                let mut q = gate.targets.clone();
                q.extend(gate.controls.clone());
                q
            }
            Operation::Measure { qubit, .. } => vec![*qubit],
            Operation::MeasurePauliProduct { paulis, .. } => paulis.iter().map(|(q, _)| *q).collect(),
            Operation::Reset { qubit, .. } => vec![*qubit],
            Operation::ConditionalGate { gate, .. } => gate.qubits(),
            Operation::Barrier { qubits } => qubits.clone(),
        }
    }
}

/// Quantum state with classical register for mid-circuit measurements.
#[derive(Clone)]
pub struct QuantumStateWithMeasurements {
    /// Quantum state
    pub state: QuantumState,
    /// Classical register (measurement results)
    pub classical_register: Vec<bool>,
    /// Number of classical bits allocated
    pub num_classical_bits: usize,
}

impl QuantumStateWithMeasurements {
    /// Create a new quantum state with classical register.
    pub fn new(num_qubits: usize) -> Self {
        QuantumStateWithMeasurements {
            state: QuantumState::new(num_qubits),
            classical_register: Vec::new(),
            num_classical_bits: 0,
        }
    }

    /// Create a new quantum state with pre-allocated classical register.
    pub fn with_classical_bits(num_qubits: usize, num_bits: usize) -> Self {
        QuantumStateWithMeasurements {
            state: QuantumState::new(num_qubits),
            classical_register: vec![false; num_bits],
            num_classical_bits: num_bits,
        }
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits
    }

    /// Execute a single operation.
    pub fn execute(&mut self, op: &Operation) {
        match op {
            Operation::Gate(gate) => {
                self.apply_gate(gate);
            }
            Operation::Measure { qubit, bit } => {
                self.measure(*qubit, *bit);
            }
            Operation::MeasurePauliProduct { paulis, bit } => {
                self.measure_pauli_product(paulis, *bit);
            }
            Operation::Reset { qubit } => {
                self.reset(*qubit);
            }
            Operation::ConditionalGate { gate, condition } => {
                if condition.evaluate(&self.classical_register) {
                    self.execute(gate);
                }
            }
            Operation::Barrier { .. } => {
                // Barrier is just a marker for optimization
                // No action needed for sequential execution
            }
        }
    }

    /// Execute a sequence of operations.
    pub fn execute_all(&mut self, ops: &[Operation]) {
        for op in ops {
            self.execute(op);
        }
    }

    /// Apply a gate to the quantum state.
    fn apply_gate(&mut self, gate: &Gate) {
        use crate::GateOperations;
        match &gate.gate_type {
            crate::gates::GateType::H => GateOperations::h(&mut self.state, gate.targets[0]),
            crate::gates::GateType::X => GateOperations::x(&mut self.state, gate.targets[0]),
            crate::gates::GateType::Y => GateOperations::y(&mut self.state, gate.targets[0]),
            crate::gates::GateType::Z => GateOperations::z(&mut self.state, gate.targets[0]),
            crate::gates::GateType::S => GateOperations::s(&mut self.state, gate.targets[0]),
            crate::gates::GateType::T => GateOperations::t(&mut self.state, gate.targets[0]),
            crate::gates::GateType::CNOT => {
                GateOperations::cnot(&mut self.state, gate.controls[0], gate.targets[0])
            }
            crate::gates::GateType::CZ => {
                GateOperations::cz(&mut self.state, gate.controls[0], gate.targets[0])
            }
            crate::gates::GateType::SWAP => {
                GateOperations::swap(&mut self.state, gate.targets[0], gate.targets[1])
            }
            crate::gates::GateType::Rz(theta) => {
                GateOperations::rz(&mut self.state, gate.targets[0], *theta)
            }
            _ => {
                // Fallback: use generic apply_gate_to_state
                crate::ascii_viz::apply_gate_to_state(&mut self.state, gate);
            }
        }
    }

    /// Measure a single qubit and store result in classical register.
    fn measure(&mut self, qubit: usize, bit: usize) {
        let result = self.measure_single_qubit(qubit);
        self.set_classical_bit(bit, result);
        self.collapse_to_measurement_result(qubit, result);
    }

    /// Measure a tensor product of Pauli operators and store the parity bit.
    ///
    /// `false` means +1 eigenvalue, `true` means -1 eigenvalue.
    fn measure_pauli_product(&mut self, paulis: &[(usize, PauliBasis)], bit: usize) {
        if paulis.is_empty() {
            self.set_classical_bit(bit, false);
            return;
        }

        self.validate_pauli_terms(paulis);
        self.rotate_pauli_axes_into_z(paulis);

        let mut parity = false;
        for (q, _) in paulis {
            let outcome = self.measure_single_qubit(*q);
            self.collapse_to_measurement_result(*q, outcome);
            parity ^= outcome;
        }

        self.rotate_pauli_axes_from_z(paulis);
        self.set_classical_bit(bit, parity);
    }

    fn validate_pauli_terms(&self, paulis: &[(usize, PauliBasis)]) {
        let mut seen = HashSet::with_capacity(paulis.len());
        for (q, _) in paulis {
            assert!(
                *q < self.state.num_qubits,
                "Pauli-product measurement qubit {} out of range (n={})",
                q,
                self.state.num_qubits
            );
            assert!(
                seen.insert(*q),
                "Duplicate qubit {} in Pauli-product measurement",
                q
            );
        }
    }

    fn rotate_pauli_axes_into_z(&mut self, paulis: &[(usize, PauliBasis)]) {
        for (q, basis) in paulis {
            match basis {
                PauliBasis::X => GateOperations::h(&mut self.state, *q),
                PauliBasis::Y => {
                    // S† then H maps Y-basis measurement to Z-basis measurement.
                    GateOperations::s(&mut self.state, *q);
                    GateOperations::s(&mut self.state, *q);
                    GateOperations::s(&mut self.state, *q);
                    GateOperations::h(&mut self.state, *q);
                }
                PauliBasis::Z => {}
            }
        }
    }

    fn rotate_pauli_axes_from_z(&mut self, paulis: &[(usize, PauliBasis)]) {
        for (q, basis) in paulis.iter().rev() {
            match basis {
                PauliBasis::X => GateOperations::h(&mut self.state, *q),
                PauliBasis::Y => {
                    GateOperations::h(&mut self.state, *q);
                    GateOperations::s(&mut self.state, *q);
                }
                PauliBasis::Z => {}
            }
        }
    }

    /// Measure a single qubit (0 or 1).
    fn measure_single_qubit(&self, qubit: usize) -> bool {
        let prob_one = self.measurement_probability_one(qubit);
        // Sample from distribution
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_bool(prob_one)
    }

    /// Calculate Pr(qubit = 1) without sampling.
    fn measurement_probability_one(&self, qubit: usize) -> f64 {
        let probs = self.state.probabilities();

        // Calculate probability of qubit being |1⟩
        let mut prob_one = 0.0;
        let mask = 1usize << qubit;

        for (idx, &prob) in probs.iter().enumerate() {
            if idx & mask != 0 {
                prob_one += prob;
            }
        }

        prob_one
    }

    /// Collapse state to measurement result.
    fn collapse_to_measurement_result(&mut self, qubit: usize, result: bool) {
        let mask = 1usize << qubit;

        // Project onto the measured subspace
        let mut new_amps = vec![Complex64::new(0.0, 0.0); self.state.amplitudes_ref().len()];

        for (idx, amp) in self.state.amplitudes_ref().iter().enumerate() {
            let qubit_value = (idx & mask) != 0;
            if qubit_value == result {
                // Keep this amplitude
                new_amps[idx] = *amp;
            }
        }

        // Renormalize
        let norm_sq: f64 = new_amps.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for amp in &mut new_amps {
                *amp = *amp / norm;
            }
        }

        // Update state using mutable reference
        let amplitudes = self.state.amplitudes_mut();
        for (i, amp) in new_amps.into_iter().enumerate() {
            amplitudes[i] = amp;
        }
    }

    fn set_classical_bit(&mut self, bit: usize, value: bool) {
        while self.classical_register.len() <= bit {
            self.classical_register.push(false);
        }
        self.num_classical_bits = self.num_classical_bits.max(bit + 1);
        self.classical_register[bit] = value;
    }

    /// Reset a qubit to |0⟩ (measure first, then apply X if needed).
    fn reset(&mut self, qubit: usize) {
        // Measure to check current state
        if self.measure_single_qubit(qubit) {
            // Qubit is |1⟩, apply X to flip to |0⟩
            use crate::GateOperations;
            GateOperations::x(&mut self.state, qubit);
        }
        // Qubit is now |0⟩
    }

    /// Get classical register value as integer.
    pub fn classical_value(&self, bits: &[usize]) -> usize {
        let mut value = 0;
        for (i, &bit) in bits.iter().enumerate() {
            if self.classical_register.get(bit).copied().unwrap_or(false) {
                value |= 1 << i;
            }
        }
        value
    }

    /// Get classical register as string of bits.
    pub fn classical_bits_string(&self, num_bits: usize) -> String {
        (0..num_bits)
            .map(|i| {
                if self.classical_register.get(i).copied().unwrap_or(false) {
                    '1'
                } else {
                    '0'
                }
            })
            .collect()
    }

    /// Reset classical register.
    pub fn reset_classical(&mut self) {
        self.classical_register.fill(false);
    }
}

/// Measurement result from circuit execution.
#[derive(Clone)]
pub struct ExecutionResult {
    /// Final quantum state
    pub state: QuantumState,
    /// Classical register values
    pub classical_register: Vec<bool>,
    /// Measurement counts (bit index -> count of 1s)
    pub measurement_counts: HashMap<usize, usize>,
}

/// Configuration for branch-based dynamic-circuit execution.
#[derive(Clone, Debug)]
pub struct ShotBranchingConfig {
    /// Maximum number of live branches retained after each operation.
    pub max_branches: usize,
    /// Branches below this probability are dropped.
    pub prune_probability: f64,
}

impl Default for ShotBranchingConfig {
    fn default() -> Self {
        Self {
            max_branches: 4096,
            prune_probability: 1e-12,
        }
    }
}

/// One outcome from exact branch propagation.
#[derive(Clone, Debug)]
pub struct BranchingOutcome {
    pub classical_register: Vec<bool>,
    pub probability: f64,
}

/// Shot-sampled result based on branch propagation.
#[derive(Clone, Debug)]
pub struct ShotBranchingResult {
    pub counts: HashMap<String, usize>,
    pub outcomes: Vec<BranchingOutcome>,
    pub pruned_branches: usize,
}

impl QuantumStateWithMeasurements {
    /// Execute operations and return final result.
    pub fn execute_and_finish(mut self, ops: &[Operation]) -> ExecutionResult {
        // Execute all operations
        self.execute_all(ops);

        ExecutionResult {
            state: self.state.clone(),
            classical_register: self.classical_register.clone(),
            measurement_counts: HashMap::new(), // Could track with repeated execution
        }
    }

    /// Convert to/from base QuantumState
    pub fn from_state(state: QuantumState) -> Self {
        QuantumStateWithMeasurements {
            state,
            classical_register: Vec::new(),
            num_classical_bits: 0,
        }
    }

    pub fn into_state(self) -> QuantumState {
        self.state
    }

    fn apply_operation_with_branching(
        mut sim: QuantumStateWithMeasurements,
        weight: f64,
        op: &Operation,
        prune_probability: f64,
    ) -> Vec<(QuantumStateWithMeasurements, f64)> {
        if weight <= prune_probability {
            return Vec::new();
        }

        match op {
            Operation::Gate(gate) => {
                sim.apply_gate(gate);
                vec![(sim, weight)]
            }
            Operation::Barrier { .. } => vec![(sim, weight)],
            Operation::Measure { qubit, bit } => {
                let p1 = sim.measurement_probability_one(*qubit).clamp(0.0, 1.0);
                let p0 = 1.0 - p1;
                let mut out = Vec::with_capacity(2);

                if p0 > prune_probability {
                    let mut b0 = sim.clone();
                    b0.set_classical_bit(*bit, false);
                    b0.collapse_to_measurement_result(*qubit, false);
                    out.push((b0, weight * p0));
                }
                if p1 > prune_probability {
                    let mut b1 = sim;
                    b1.set_classical_bit(*bit, true);
                    b1.collapse_to_measurement_result(*qubit, true);
                    out.push((b1, weight * p1));
                }

                out
            }
            Operation::MeasurePauliProduct { paulis, bit } => {
                sim.validate_pauli_terms(paulis);
                sim.rotate_pauli_axes_into_z(paulis);

                let mut branches: Vec<(QuantumStateWithMeasurements, f64, bool)> =
                    vec![(sim, weight, false)];

                for (q, _) in paulis {
                    let mut next = Vec::with_capacity(branches.len() * 2);
                    for (state, w, parity) in branches {
                        let p1 = state.measurement_probability_one(*q).clamp(0.0, 1.0);
                        let p0 = 1.0 - p1;

                        if p0 > prune_probability {
                            let mut b0 = state.clone();
                            b0.collapse_to_measurement_result(*q, false);
                            next.push((b0, w * p0, parity));
                        }
                        if p1 > prune_probability {
                            let mut b1 = state;
                            b1.collapse_to_measurement_result(*q, true);
                            next.push((b1, w * p1, !parity));
                        }
                    }
                    branches = next;
                    if branches.is_empty() {
                        break;
                    }
                }

                let mut out = Vec::with_capacity(branches.len());
                for (mut state, w, parity) in branches {
                    state.rotate_pauli_axes_from_z(paulis);
                    state.set_classical_bit(*bit, parity);
                    out.push((state, w));
                }
                out
            }
            Operation::Reset { qubit } => {
                let p1 = sim.measurement_probability_one(*qubit).clamp(0.0, 1.0);
                let p0 = 1.0 - p1;
                let mut out = Vec::with_capacity(2);

                if p0 > prune_probability {
                    let mut b0 = sim.clone();
                    b0.collapse_to_measurement_result(*qubit, false);
                    out.push((b0, weight * p0));
                }
                if p1 > prune_probability {
                    let mut b1 = sim;
                    b1.collapse_to_measurement_result(*qubit, true);
                    GateOperations::x(&mut b1.state, *qubit);
                    out.push((b1, weight * p1));
                }

                out
            }
            Operation::ConditionalGate { gate, condition } => {
                if condition.evaluate(&sim.classical_register) {
                    Self::apply_operation_with_branching(sim, weight, gate, prune_probability)
                } else {
                    vec![(sim, weight)]
                }
            }
        }
    }

    fn bits_to_string(bits: &[bool], width: usize) -> String {
        (0..width)
            .map(|i| {
                if bits.get(i).copied().unwrap_or(false) {
                    '1'
                } else {
                    '0'
                }
            })
            .collect()
    }

    /// Compute exact classical-outcome distribution using branch propagation.
    pub fn branching_distribution(
        &self,
        ops: &[Operation],
        config: &ShotBranchingConfig,
    ) -> Result<(Vec<BranchingOutcome>, usize), String> {
        if config.max_branches == 0 {
            return Err("max_branches must be >= 1".to_string());
        }
        if !(0.0..1.0).contains(&config.prune_probability) {
            return Err("prune_probability must be in [0, 1)".to_string());
        }

        let mut branches: Vec<(QuantumStateWithMeasurements, f64)> = vec![(self.clone(), 1.0)];
        let mut pruned_branches = 0usize;

        for op in ops {
            let mut next = Vec::new();
            for (sim, weight) in branches {
                next.extend(Self::apply_operation_with_branching(
                    sim,
                    weight,
                    op,
                    config.prune_probability,
                ));
            }

            if next.is_empty() {
                return Err("all branches were pruned; adjust prune_probability".to_string());
            }

            if next.len() > config.max_branches {
                next.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                pruned_branches += next.len() - config.max_branches;
                next.truncate(config.max_branches);
            }

            let total_weight: f64 = next.iter().map(|(_, w)| *w).sum();
            if total_weight <= 0.0 {
                return Err("invalid branch distribution: zero total weight".to_string());
            }
            for (_, w) in &mut next {
                *w /= total_weight;
            }

            branches = next;
        }

        let outcomes = branches
            .into_iter()
            .map(|(sim, probability)| BranchingOutcome {
                classical_register: sim.classical_register,
                probability,
            })
            .collect::<Vec<_>>();

        Ok((outcomes, pruned_branches))
    }

    /// Execute dynamic circuit by exact branch propagation + shot sampling.
    pub fn execute_shots_branching(
        &self,
        ops: &[Operation],
        shots: usize,
        config: &ShotBranchingConfig,
    ) -> Result<ShotBranchingResult, String> {
        if shots == 0 {
            return Err("shots must be >= 1".to_string());
        }
        let (outcomes, pruned_branches) = self.branching_distribution(ops, config)?;
        if outcomes.is_empty() {
            return Err("no branching outcomes available".to_string());
        }

        let width = outcomes
            .iter()
            .map(|o| o.classical_register.len())
            .max()
            .unwrap_or(0);
        let mut cumulative = Vec::with_capacity(outcomes.len());
        let mut acc = 0.0;
        for o in &outcomes {
            acc += o.probability.max(0.0);
            cumulative.push(acc);
        }
        if acc <= 0.0 {
            return Err("invalid branching outcomes: zero probability mass".to_string());
        }
        for c in &mut cumulative {
            *c /= acc;
        }

        let mut counts = HashMap::new();
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..shots {
            let r: f64 = rng.gen();
            let idx = cumulative
                .partition_point(|&c| c < r)
                .min(outcomes.len() - 1);
            let key = Self::bits_to_string(&outcomes[idx].classical_register, width);
            *counts.entry(key).or_insert(0) += 1;
        }

        Ok(ShotBranchingResult {
            counts,
            outcomes,
            pruned_branches,
        })
    }
}

// ============================================================================
// CONVERSION HELPERS
// ============================================================================

/// Convert a vector of Gates to Operations.
pub fn gates_to_operations(gates: &[Gate]) -> Vec<Operation> {
    gates.iter().map(|g| Operation::Gate(g.clone())).collect()
}

/// Convert Operations to Gates (extracts only Gate operations).
pub fn operations_to_gates(ops: &[Operation]) -> Vec<Gate> {
    ops.iter()
        .filter_map(|op| match op {
            Operation::Gate(gate) => Some(gate.clone()),
            _ => None,
        })
        .collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GateOperations;

    #[test]
    fn test_basic_measurement() {
        let mut sim = QuantumStateWithMeasurements::new(2);

        // Put qubit 0 in |1⟩
        sim.execute(&Operation::x(0));

        // Measure qubit 0 into bit 0
        sim.execute(&Operation::measure(0, 0));

        // Should measure 1
        assert!(sim.classical_register[0]);
    }

    #[test]
    fn test_hadamard_measurement_distribution() {
        // Run multiple times to test distribution
        let mut ones = 0;
        let trials = 100;

        for _ in 0..trials {
            let mut sim = QuantumStateWithMeasurements::new(1);
            sim.execute(&Operation::h(0));
            sim.execute(&Operation::measure(0, 0));
            if sim.classical_register[0] {
                ones += 1;
            }
        }

        // Should be approximately 50% (within 3 sigma for 100 trials)
        let ratio = ones as f64 / trials as f64;
        assert!((ratio - 0.5).abs() < 0.15, "Ratio was {}", ratio);
    }

    #[test]
    fn test_reset_operation() {
        let mut sim = QuantumStateWithMeasurements::new(2);

        // Put both qubits in |1⟩
        sim.execute(&Operation::x(0));
        sim.execute(&Operation::x(1));

        // Reset qubit 0
        sim.execute(&Operation::reset(0));

        // Qubit 0 should be |0⟩, qubit 1 should be |1⟩
        sim.execute(&Operation::measure(0, 0));
        sim.execute(&Operation::measure(1, 1));

        assert!(!sim.classical_register[0]); // Qubit 0 is 0
        assert!(sim.classical_register[1]); // Qubit 1 is 1
    }

    #[test]
    fn test_conditional_gate() {
        let mut sim = QuantumStateWithMeasurements::with_classical_bits(3, 2);

        // Set bit 0 to true
        sim.classical_register[0] = true;

        // Conditional X on qubit 1 if bit 0 is set
        let cond_op = Operation::conditional(Operation::x(1), ClassicalCondition::BitSet(0));
        sim.execute(&cond_op);

        // Qubit 1 should now be |1⟩
        sim.execute(&Operation::measure(1, 0));
        assert!(sim.classical_register[0]); // Now contains measurement of qubit 1
    }

    #[test]
    fn test_conditional_not_executed() {
        let mut sim = QuantumStateWithMeasurements::with_classical_bits(3, 2);

        // Bit 0 is false (default)
        assert!(!sim.classical_register[0]);

        // Conditional X on qubit 1 if bit 0 is set
        let cond_op = Operation::conditional(Operation::x(1), ClassicalCondition::BitSet(0));
        sim.execute(&cond_op);

        // Qubit 1 should still be |0⟩
        sim.execute(&Operation::measure(1, 1));
        assert!(!sim.classical_register[1]);
    }

    #[test]
    fn test_classical_value() {
        let mut sim = QuantumStateWithMeasurements::with_classical_bits(5, 8);

        // Set bits 0, 2, 3 to true
        sim.classical_register[0] = true;
        sim.classical_register[2] = true;
        sim.classical_register[3] = true;

        // Value of bits [0, 2, 3] in LSB-first encoding
        // Position 0 (bit 0) → 1 << 0 = 1
        // Position 1 (bit 2) → 1 << 1 = 2
        // Position 2 (bit 3) → 1 << 2 = 4
        // Total = 1 + 2 + 4 = 7
        let value = sim.classical_value(&[0, 2, 3]);
        assert_eq!(value, 7);
    }

    #[test]
    fn test_classical_bits_string() {
        let mut sim = QuantumStateWithMeasurements::with_classical_bits(5, 4);

        sim.classical_register[0] = true;
        sim.classical_register[2] = true;

        let s = sim.classical_bits_string(4);
        assert_eq!(s.len(), 4);
        assert_eq!(s.chars().nth(0).unwrap(), '1');
        assert_eq!(s.chars().nth(1).unwrap(), '0');
        assert_eq!(s.chars().nth(2).unwrap(), '1');
        assert_eq!(s.chars().nth(3).unwrap(), '0');
    }

    #[test]
    fn test_simple_bell_measurement() {
        let mut sim = QuantumStateWithMeasurements::new(2);

        // Create Bell state
        sim.execute_all(&[Operation::h(0), Operation::cnot(0, 1)]);

        // Measure both qubits
        sim.execute(&Operation::measure(0, 0));
        sim.execute(&Operation::measure(1, 1));

        // Results should be correlated
        assert_eq!(sim.classical_register[0], sim.classical_register[1]);
    }

    #[test]
    fn test_pauli_product_measure_z_single_qubit() {
        let mut sim = QuantumStateWithMeasurements::new(1);
        sim.execute(&Operation::x(0)); // |1>
        sim.execute(&Operation::measure_pauli_product(
            vec![(0, PauliBasis::Z)],
            0,
        ));
        assert!(
            sim.classical_register[0],
            "Z on |1> should give -1 eigenvalue (bit=true)"
        );
    }

    #[test]
    fn test_pauli_product_measure_x_on_plus_state() {
        let mut sim = QuantumStateWithMeasurements::new(1);
        sim.execute(&Operation::h(0)); // |+>
        sim.execute(&Operation::measure_pauli_product(
            vec![(0, PauliBasis::X)],
            0,
        ));
        assert!(
            !sim.classical_register[0],
            "X on |+> should give +1 eigenvalue (bit=false)"
        );
    }

    #[test]
    fn test_pauli_product_measure_zz_on_bell_state() {
        let mut sim = QuantumStateWithMeasurements::new(2);
        sim.execute_all(&[Operation::h(0), Operation::cnot(0, 1)]); // |Φ+>
        sim.execute(&Operation::measure_pauli_product(
            vec![(0, PauliBasis::Z), (1, PauliBasis::Z)],
            0,
        ));
        assert!(
            !sim.classical_register[0],
            "ZZ on |Φ+> should give +1 eigenvalue (bit=false)"
        );
    }

    #[test]
    fn test_condition_any_bit_set() {
        let mut sim = QuantumStateWithMeasurements::with_classical_bits(3, 3);

        sim.classical_register[1] = true;

        let cond = ClassicalCondition::any_bit_set(&[0, 1, 2]);
        assert!(cond.evaluate(&sim.classical_register));

        sim.classical_register[1] = false;
        assert!(!cond.evaluate(&sim.classical_register));
    }

    #[test]
    fn test_condition_all_bits_set() {
        let mut sim = QuantumStateWithMeasurements::with_classical_bits(3, 3);

        sim.classical_register[0] = true;
        sim.classical_register[1] = false;

        let cond = ClassicalCondition::all_bits_set(&[0, 1]);
        assert!(!cond.evaluate(&sim.classical_register));

        sim.classical_register[1] = true;
        assert!(cond.evaluate(&sim.classical_register));
    }

    #[test]
    fn test_gates_to_operations_conversion() {
        let gates = vec![Gate::h(0), Gate::x(1)];
        let ops = gates_to_operations(&gates);

        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], Operation::Gate(_)));
        assert!(matches!(ops[1], Operation::Gate(_)));
    }

    #[test]
    fn test_operations_to_gates_extraction() {
        let ops = vec![Operation::h(0), Operation::measure(0, 0), Operation::x(1)];
        let gates = operations_to_gates(&ops);

        assert_eq!(gates.len(), 2); // Only the gates, not the measurement
    }

    #[test]
    fn test_branching_distribution_for_measurement_conditioned_flow() {
        let sim = QuantumStateWithMeasurements::new(2);
        let ops = vec![
            Operation::h(0),
            Operation::measure(0, 0),
            Operation::conditional(Operation::x(1), ClassicalCondition::BitSet(0)),
            Operation::measure(1, 1),
        ];
        let cfg = ShotBranchingConfig::default();
        let (outcomes, pruned) = sim
            .branching_distribution(&ops, &cfg)
            .expect("distribution");

        assert_eq!(pruned, 0);
        assert_eq!(outcomes.len(), 2);
        let total: f64 = outcomes.iter().map(|o| o.probability).sum();
        assert!((total - 1.0).abs() < 1e-9);

        let mut by_bits = HashMap::new();
        for o in outcomes {
            let key = QuantumStateWithMeasurements::bits_to_string(&o.classical_register, 2);
            by_bits.insert(key, o.probability);
        }
        let p00 = by_bits.get("00").copied().unwrap_or(0.0);
        let p11 = by_bits.get("11").copied().unwrap_or(0.0);
        assert!((p00 - 0.5).abs() < 1e-9, "p00={}", p00);
        assert!((p11 - 0.5).abs() < 1e-9, "p11={}", p11);
    }

    #[test]
    fn test_execute_shots_branching_counts() {
        let sim = QuantumStateWithMeasurements::new(2);
        let ops = vec![
            Operation::h(0),
            Operation::measure(0, 0),
            Operation::conditional(Operation::x(1), ClassicalCondition::BitSet(0)),
            Operation::measure(1, 1),
        ];
        let cfg = ShotBranchingConfig::default();
        let result = sim
            .execute_shots_branching(&ops, 2000, &cfg)
            .expect("shot branching");

        assert_eq!(result.pruned_branches, 0);
        let c00 = result.counts.get("00").copied().unwrap_or(0);
        let c11 = result.counts.get("11").copied().unwrap_or(0);
        assert_eq!(c00 + c11, 2000);
        assert!(
            (c00 as i64 - c11 as i64).unsigned_abs() < 400,
            "c00={}, c11={}",
            c00,
            c11
        );
    }
}
