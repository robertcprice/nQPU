//! Formal Verification of Circuit Unitary Equivalence
//!
//! This module provides tools for verifying that two quantum circuits implement
//! the same unitary transformation (up to global phase). Three verification
//! methods are supported:
//!
//! - **Matrix comparison**: exact unitary reconstruction for small circuits (<10 qubits).
//! - **Statistical sampling**: Haar-random input states with fidelity comparison.
//! - **Symbolic Pauli tracking**: Clifford-circuit verification via stabilizer propagation (11-20 qubits).
//!
//! An `Auto` mode selects the best strategy based on circuit size.
//!
//! # Usage
//!
//! ```ignore
//! use nqpu_metal::circuit_equivalence::{CircuitEquivalenceChecker, EquivalenceMethod};
//! use nqpu_metal::gates::{Gate, GateType};
//!
//! let checker = CircuitEquivalenceChecker::new();
//! let h_gate = Gate::new(GateType::H, vec![0], vec![]);
//! let circuit_a = vec![h_gate.clone(), h_gate.clone()]; // HH = I
//! let circuit_b: Vec<Gate> = vec![]; // empty = I
//! let result = checker.check(&circuit_a, &circuit_b, 1, EquivalenceMethod::MatrixComparison);
//! assert!(result.equivalent);
//! ```

use crate::ascii_viz::apply_gate_to_state;
use crate::gates::{Gate, GateType};
use crate::{c64_one, c64_zero, QuantumState, C64};

use std::f64::consts::PI;

/// Default numerical tolerance for floating-point comparisons.
const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Qubit threshold below which full matrix comparison is preferred.
const MATRIX_METHOD_QUBIT_LIMIT: usize = 10;

/// Qubit threshold above which symbolic tracking is preferred over sampling.
const SYMBOLIC_METHOD_QUBIT_LOWER: usize = 11;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Strategy for checking circuit equivalence.
#[derive(Clone, Debug, PartialEq)]
pub enum EquivalenceMethod {
    /// Reconstruct the full 2^n x 2^n unitary for each circuit and compare
    /// entry-by-entry (up to global phase). Practical for fewer than 10 qubits.
    MatrixComparison,

    /// Apply both circuits to `num_samples` random Haar-distributed input
    /// states and compare output fidelities.
    StatisticalSampling { num_samples: usize },

    /// Propagate Pauli stabilizers symbolically through each circuit.
    /// Efficient for Clifford-only circuits in the 11-20 qubit range.
    SymbolicTracking,

    /// Automatically select the best method based on circuit size and gate
    /// composition.
    Auto,
}

/// Outcome of an equivalence check.
#[derive(Clone, Debug)]
pub struct EquivalenceResult {
    /// `true` when the two circuits are equivalent (within tolerance).
    pub equivalent: bool,

    /// The verification method that was actually used.
    pub method_used: EquivalenceMethod,

    /// State fidelity between outputs. 1.0 for an exact match; less for
    /// statistical checks or non-equivalent circuits.
    pub fidelity: f64,

    /// Global phase factor detected between the two unitaries, if any.
    /// `Some(e^{i*phi})` when the circuits differ only by a global phase.
    pub global_phase: Option<C64>,

    /// An input state that witnesses non-equivalence, when one is found.
    pub counterexample: Option<Vec<C64>>,
}

impl EquivalenceResult {
    /// Convenience constructor for a positive result.
    fn equivalent_with(method: EquivalenceMethod, fidelity: f64, phase: Option<C64>) -> Self {
        Self {
            equivalent: true,
            method_used: method,
            fidelity,
            global_phase: phase,
            counterexample: None,
        }
    }

    /// Convenience constructor for a negative result.
    fn not_equivalent(
        method: EquivalenceMethod,
        fidelity: f64,
        counterexample: Option<Vec<C64>>,
    ) -> Self {
        Self {
            equivalent: false,
            method_used: method,
            fidelity,
            global_phase: None,
            counterexample,
        }
    }
}

// ---------------------------------------------------------------------------
// Checker
// ---------------------------------------------------------------------------

/// Stateless checker object. Holds no mutable state, so it can be shared
/// freely across threads.
pub struct CircuitEquivalenceChecker {
    tolerance: f64,
}

impl CircuitEquivalenceChecker {
    /// Create a checker with the default tolerance (1e-10).
    pub fn new() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
        }
    }

    /// Create a checker with a custom tolerance.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self { tolerance }
    }

    // ------------------------------------------------------------------
    // Public entry points
    // ------------------------------------------------------------------

    /// Check equivalence using the specified method.
    pub fn check(
        &self,
        circuit_a: &[Gate],
        circuit_b: &[Gate],
        num_qubits: usize,
        method: EquivalenceMethod,
    ) -> EquivalenceResult {
        let resolved = match method {
            EquivalenceMethod::Auto => self.select_method(circuit_a, circuit_b, num_qubits),
            other => other,
        };

        match &resolved {
            EquivalenceMethod::MatrixComparison => {
                self.matrix_check(circuit_a, circuit_b, num_qubits)
            }
            EquivalenceMethod::StatisticalSampling { num_samples } => {
                self.statistical_check(circuit_a, circuit_b, num_qubits, *num_samples)
            }
            EquivalenceMethod::SymbolicTracking => {
                self.symbolic_check(circuit_a, circuit_b, num_qubits)
            }
            EquivalenceMethod::Auto => unreachable!("Auto was resolved above"),
        }
    }

    /// Check equivalence with an explicit tolerance, using `Auto` method
    /// selection.
    pub fn check_with_tolerance(
        &self,
        circuit_a: &[Gate],
        circuit_b: &[Gate],
        num_qubits: usize,
        tolerance: f64,
    ) -> EquivalenceResult {
        // We cannot mutate self, so build a temporary checker with the
        // caller-specified tolerance.
        let tmp = CircuitEquivalenceChecker::with_tolerance(tolerance);
        tmp.check(circuit_a, circuit_b, num_qubits, EquivalenceMethod::Auto)
    }

    // ------------------------------------------------------------------
    // Method selection
    // ------------------------------------------------------------------

    fn select_method(
        &self,
        circuit_a: &[Gate],
        circuit_b: &[Gate],
        num_qubits: usize,
    ) -> EquivalenceMethod {
        if num_qubits < MATRIX_METHOD_QUBIT_LIMIT {
            return EquivalenceMethod::MatrixComparison;
        }

        if num_qubits >= SYMBOLIC_METHOD_QUBIT_LOWER
            && is_clifford_circuit(circuit_a)
            && is_clifford_circuit(circuit_b)
        {
            return EquivalenceMethod::SymbolicTracking;
        }

        // Default fallback for 10+ qubit non-Clifford circuits.
        let samples = (100 * num_qubits).max(500);
        EquivalenceMethod::StatisticalSampling {
            num_samples: samples,
        }
    }

    // ------------------------------------------------------------------
    // Matrix comparison (exact)
    // ------------------------------------------------------------------

    fn matrix_check(
        &self,
        circuit_a: &[Gate],
        circuit_b: &[Gate],
        num_qubits: usize,
    ) -> EquivalenceResult {
        let u1 = build_unitary(circuit_a, num_qubits);
        let u2 = build_unitary(circuit_b, num_qubits);
        compare_unitaries(&u1, &u2, self.tolerance)
    }

    // ------------------------------------------------------------------
    // Statistical sampling
    // ------------------------------------------------------------------

    fn statistical_check(
        &self,
        circuit_a: &[Gate],
        circuit_b: &[Gate],
        num_qubits: usize,
        num_samples: usize,
    ) -> EquivalenceResult {
        let mut min_fidelity: f64 = 1.0;
        let mut worst_input: Option<Vec<C64>> = None;

        for _ in 0..num_samples {
            let input = random_haar_state(num_qubits);

            let mut state_a = state_from_amplitudes(num_qubits, &input);
            let mut state_b = state_from_amplitudes(num_qubits, &input);

            apply_circuit_to_state(&mut state_a, circuit_a);
            apply_circuit_to_state(&mut state_b, circuit_b);

            let fid = state_a.fidelity(&state_b);
            if fid < min_fidelity {
                min_fidelity = fid;
                if fid < 1.0 - self.tolerance {
                    worst_input = Some(input.clone());
                }
            }
        }

        let method = EquivalenceMethod::StatisticalSampling { num_samples };

        if min_fidelity >= 1.0 - self.tolerance {
            EquivalenceResult::equivalent_with(method, min_fidelity, None)
        } else {
            EquivalenceResult::not_equivalent(method, min_fidelity, worst_input)
        }
    }

    // ------------------------------------------------------------------
    // Symbolic Pauli tracking (Clifford circuits)
    // ------------------------------------------------------------------

    fn symbolic_check(
        &self,
        circuit_a: &[Gate],
        circuit_b: &[Gate],
        num_qubits: usize,
    ) -> EquivalenceResult {
        let tab_a = build_stabilizer_tableau(circuit_a, num_qubits);
        let tab_b = build_stabilizer_tableau(circuit_b, num_qubits);

        let method = EquivalenceMethod::SymbolicTracking;

        if tab_a == tab_b {
            EquivalenceResult::equivalent_with(method, 1.0, None)
        } else {
            // We cannot produce a concrete counterexample from tableaux alone
            // without reconstructing a state, so leave it as None.
            EquivalenceResult::not_equivalent(method, 0.0, None)
        }
    }
}

impl Default for CircuitEquivalenceChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience wrapper
// ---------------------------------------------------------------------------

/// Verify that an optimized circuit is equivalent to its original.
///
/// Returns an `EquivalenceResult` using `Auto` method selection.
pub fn verify_optimization(
    original: &[Gate],
    optimized: &[Gate],
    num_qubits: usize,
) -> EquivalenceResult {
    let checker = CircuitEquivalenceChecker::new();
    checker.check(original, optimized, num_qubits, EquivalenceMethod::Auto)
}

// ---------------------------------------------------------------------------
// Core algorithms (module-private)
// ---------------------------------------------------------------------------

/// Apply every gate in `circuit` to `state` in sequence.
pub fn apply_circuit_to_state(state: &mut QuantumState, circuit: &[Gate]) {
    for gate in circuit {
        apply_gate_to_state(state, gate);
    }
}

/// Construct the full unitary matrix for `circuit` acting on `num_qubits`.
///
/// Column `j` of the returned matrix is the state vector obtained by
/// applying the circuit to basis state |j>.  The matrix is stored as
/// `result[row][col]`.
pub fn build_unitary(circuit: &[Gate], num_qubits: usize) -> Vec<Vec<C64>> {
    let dim = 1usize << num_qubits;
    let mut matrix = vec![vec![c64_zero(); dim]; dim];

    for col in 0..dim {
        let mut state = QuantumState::new(num_qubits);
        // Prepare basis state |col>
        {
            let amps = state.amplitudes_mut();
            amps[0] = c64_zero();
            amps[col] = c64_one();
        }
        apply_circuit_to_state(&mut state, circuit);

        for row in 0..dim {
            matrix[row][col] = state.get(row);
        }
    }

    matrix
}

/// Compare two unitary matrices up to global phase and tolerance.
///
/// Two unitaries U1 and U2 are considered equivalent if there exists a
/// global phase factor `e^{i*phi}` such that `|U1[i][j] - e^{i*phi} * U2[i][j]| < tol`
/// for all entries.  The phase is estimated from the first non-negligible
/// entry pair.
pub fn compare_unitaries(u1: &[Vec<C64>], u2: &[Vec<C64>], tolerance: f64) -> EquivalenceResult {
    let dim = u1.len();
    assert_eq!(dim, u2.len(), "Unitary dimensions must match");

    let method = EquivalenceMethod::MatrixComparison;

    // Find the global phase from the first non-negligible entry.
    let mut global_phase: Option<C64> = None;

    for row in 0..dim {
        for col in 0..dim {
            let a = u1[row][col];
            let b = u2[row][col];

            let a_mag = a.norm();
            let b_mag = b.norm();

            // Both entries negligible -- skip.
            if a_mag < tolerance && b_mag < tolerance {
                continue;
            }

            // One negligible, the other not -- not equivalent.
            if (a_mag < tolerance) != (b_mag < tolerance) {
                let ce = find_counterexample_from_col(u1, u2, dim, tolerance);
                return EquivalenceResult::not_equivalent(method, 0.0, ce);
            }

            if global_phase.is_none() {
                // phase = a / b  -->  U1 = phase * U2
                let phase = a / b;
                // Normalise to unit magnitude.
                let phase_norm = phase.norm();
                global_phase = Some(phase / phase_norm);
            }

            let phase = global_phase.unwrap();
            let diff = a - phase * b;
            if diff.norm() > tolerance {
                let ce = find_counterexample_from_col(u1, u2, dim, tolerance);
                return EquivalenceResult::not_equivalent(method, 0.0, ce);
            }
        }
    }

    // All entries matched.
    let detected_phase = global_phase.unwrap_or(c64_one());

    // If the phase is effectively 1.0, report None.
    let phase_report = if (detected_phase - c64_one()).norm() < tolerance {
        None
    } else {
        Some(detected_phase)
    };

    EquivalenceResult::equivalent_with(method, 1.0, phase_report)
}

/// Attempt to find a basis-state counterexample by scanning columns.
fn find_counterexample_from_col(
    u1: &[Vec<C64>],
    u2: &[Vec<C64>],
    dim: usize,
    tolerance: f64,
) -> Option<Vec<C64>> {
    // For each column (basis state |j>), check if the output columns match
    // up to the global phase determined by the first pair.
    for col in 0..dim {
        let col_a: Vec<C64> = (0..dim).map(|r| u1[r][col]).collect();
        let col_b: Vec<C64> = (0..dim).map(|r| u2[r][col]).collect();

        if !columns_equivalent(&col_a, &col_b, tolerance) {
            // Return the basis state as the counterexample.
            let mut ce = vec![c64_zero(); dim];
            ce[col] = c64_one();
            return Some(ce);
        }
    }
    None
}

/// Test whether two column vectors are equal up to a single global phase.
fn columns_equivalent(a: &[C64], b: &[C64], tolerance: f64) -> bool {
    let mut phase: Option<C64> = None;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai.norm() < tolerance && bi.norm() < tolerance {
            continue;
        }
        if (ai.norm() < tolerance) != (bi.norm() < tolerance) {
            return false;
        }
        if phase.is_none() {
            let p = ai / bi;
            let pn = p.norm();
            phase = Some(p / pn);
        }
        let diff = ai - phase.unwrap() * bi;
        if diff.norm() > tolerance {
            return false;
        }
    }
    true
}

/// Find a concrete input state that differentiates two circuits.
///
/// Tries all basis states first, then falls back to a handful of random
/// states.
pub fn find_counterexample(
    circuit_a: &[Gate],
    circuit_b: &[Gate],
    num_qubits: usize,
) -> Option<Vec<C64>> {
    let dim = 1usize << num_qubits;
    let tolerance = DEFAULT_TOLERANCE;

    // Basis states.
    for idx in 0..dim {
        let mut input = vec![c64_zero(); dim];
        input[idx] = c64_one();

        let mut sa = state_from_amplitudes(num_qubits, &input);
        let mut sb = state_from_amplitudes(num_qubits, &input);
        apply_circuit_to_state(&mut sa, circuit_a);
        apply_circuit_to_state(&mut sb, circuit_b);

        if sa.fidelity(&sb) < 1.0 - tolerance {
            return Some(input);
        }
    }

    // Random states (32 attempts).
    for _ in 0..32 {
        let input = random_haar_state(num_qubits);
        let mut sa = state_from_amplitudes(num_qubits, &input);
        let mut sb = state_from_amplitudes(num_qubits, &input);
        apply_circuit_to_state(&mut sa, circuit_a);
        apply_circuit_to_state(&mut sb, circuit_b);

        if sa.fidelity(&sb) < 1.0 - tolerance {
            return Some(input);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Random Haar state generation
// ---------------------------------------------------------------------------

/// Generate a Haar-random state vector of dimension 2^num_qubits.
///
/// Uses the standard technique of sampling i.i.d. complex Gaussians and
/// normalising.
fn random_haar_state(num_qubits: usize) -> Vec<C64> {
    let dim = 1usize << num_qubits;
    let mut amps = Vec::with_capacity(dim);

    for _ in 0..dim {
        // Box-Muller for two independent N(0,1) samples.
        let (re, im) = box_muller_pair();
        amps.push(C64::new(re, im));
    }

    // Normalise.
    let norm: f64 = amps.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let inv = 1.0 / norm;
    for a in &mut amps {
        *a = C64::new(a.re * inv, a.im * inv);
    }

    amps
}

/// Box-Muller transform: returns two independent N(0,1) samples.
fn box_muller_pair() -> (f64, f64) {
    let u1: f64 = rand::random::<f64>().max(1e-300); // avoid log(0)
    let u2: f64 = rand::random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Build a `QuantumState` from an amplitude vector.
fn state_from_amplitudes(num_qubits: usize, amps: &[C64]) -> QuantumState {
    let mut state = QuantumState::new(num_qubits);
    let buf = state.amplitudes_mut();
    assert_eq!(buf.len(), amps.len());
    buf.copy_from_slice(amps);
    state
}

// ---------------------------------------------------------------------------
// Stabilizer tableau (lightweight Clifford tracking)
// ---------------------------------------------------------------------------

/// A stabilizer tableau for n qubits is a 2n x (2n+1) binary matrix.
///
/// Rows 0..n are the X stabilizers; rows n..2n are the Z stabilizers.
/// Columns 0..n are X bits, n..2n are Z bits, and the last column is the
/// phase bit (0 or 1 representing +1 or -1).
#[derive(Clone, Debug, PartialEq)]
struct StabilizerTableau {
    n: usize,
    /// Packed binary matrix: `table[row]` has `2n + 1` entries.
    table: Vec<Vec<u8>>,
}

impl StabilizerTableau {
    /// Identity tableau for n qubits.
    fn identity(n: usize) -> Self {
        let cols = 2 * n + 1;
        let rows = 2 * n;
        let mut table = vec![vec![0u8; cols]; rows];
        // X stabilizer generators: X_i at row i.
        for i in 0..n {
            table[i][i] = 1; // X bit
        }
        // Z stabilizer generators: Z_i at row n+i.
        for i in 0..n {
            table[n + i][n + i] = 1; // Z bit
        }
        Self { n, table }
    }

    fn phase_col(&self) -> usize {
        2 * self.n
    }

    /// Apply a Clifford gate to the tableau.
    fn apply_gate(&mut self, gate: &Gate) {
        match &gate.gate_type {
            GateType::H => self.apply_h(gate.targets[0]),
            GateType::X => self.apply_x(gate.targets[0]),
            GateType::Y => self.apply_y(gate.targets[0]),
            GateType::Z => self.apply_z(gate.targets[0]),
            GateType::S => self.apply_s(gate.targets[0]),
            GateType::CNOT => self.apply_cnot(gate.controls[0], gate.targets[0]),
            GateType::CZ => self.apply_cz(gate.controls[0], gate.targets[0]),
            GateType::SWAP => self.apply_swap(gate.targets[0], gate.targets[1]),
            _ => {
                // Non-Clifford gate: panic rather than silently producing
                // wrong results.
                panic!(
                    "SymbolicTracking received non-Clifford gate {:?}",
                    gate.gate_type
                );
            }
        }
    }

    // --- Single-qubit Clifford updates ---

    fn apply_h(&mut self, q: usize) {
        let n = self.n;
        let pc = self.phase_col();
        for row in 0..2 * n {
            // phase ^= x_q * z_q
            self.table[row][pc] ^= self.table[row][q] & self.table[row][n + q];
            // swap x_q and z_q
            let tmp = self.table[row][q];
            self.table[row][q] = self.table[row][n + q];
            self.table[row][n + q] = tmp;
        }
    }

    fn apply_s(&mut self, q: usize) {
        let n = self.n;
        let pc = self.phase_col();
        for row in 0..2 * n {
            // phase ^= x_q * z_q
            self.table[row][pc] ^= self.table[row][q] & self.table[row][n + q];
            // z_q ^= x_q
            self.table[row][n + q] ^= self.table[row][q];
        }
    }

    fn apply_x(&mut self, q: usize) {
        let n = self.n;
        let pc = self.phase_col();
        for row in 0..2 * n {
            // X commutes with X generators but anti-commutes with Z: phase ^= z_q
            self.table[row][pc] ^= self.table[row][n + q];
        }
    }

    fn apply_y(&mut self, q: usize) {
        let n = self.n;
        let pc = self.phase_col();
        for row in 0..2 * n {
            // Y anti-commutes with both X and Z when the respective bit is 1.
            self.table[row][pc] ^= self.table[row][q] ^ self.table[row][n + q];
        }
    }

    fn apply_z(&mut self, q: usize) {
        let n = self.n;
        let pc = self.phase_col();
        for row in 0..2 * n {
            // Z anti-commutes with X: phase ^= x_q
            self.table[row][pc] ^= self.table[row][q];
        }
    }

    // --- Two-qubit Clifford updates ---

    fn apply_cnot(&mut self, control: usize, target: usize) {
        let n = self.n;
        let pc = self.phase_col();
        for row in 0..2 * n {
            // phase ^= x_c * z_t * (x_t XOR z_c XOR 1)
            let xc = self.table[row][control];
            let zt = self.table[row][n + target];
            let xt = self.table[row][target];
            let zc = self.table[row][n + control];
            self.table[row][pc] ^= xc & zt & (xt ^ zc ^ 1);
            // x_t ^= x_c
            self.table[row][target] ^= self.table[row][control];
            // z_c ^= z_t
            self.table[row][n + control] ^= self.table[row][n + target];
        }
    }

    fn apply_cz(&mut self, a: usize, b: usize) {
        // CZ = H_b . CNOT(a,b) . H_b
        self.apply_h(b);
        self.apply_cnot(a, b);
        self.apply_h(b);
    }

    fn apply_swap(&mut self, a: usize, b: usize) {
        // SWAP = CNOT(a,b) . CNOT(b,a) . CNOT(a,b)
        self.apply_cnot(a, b);
        self.apply_cnot(b, a);
        self.apply_cnot(a, b);
    }
}

/// Build a stabilizer tableau by propagating an identity tableau through
/// the circuit.
fn build_stabilizer_tableau(circuit: &[Gate], num_qubits: usize) -> StabilizerTableau {
    let mut tab = StabilizerTableau::identity(num_qubits);
    for gate in circuit {
        tab.apply_gate(gate);
    }
    tab
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return `true` if every gate in the circuit is a Clifford gate.
fn is_clifford_circuit(circuit: &[Gate]) -> bool {
    circuit.iter().all(|g| is_clifford_gate(&g.gate_type))
}

/// Return `true` if the gate type is a member of the Clifford group.
fn is_clifford_gate(gt: &GateType) -> bool {
    matches!(
        gt,
        GateType::H
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::S
            | GateType::CNOT
            | GateType::CZ
            | GateType::SWAP
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};

    // Shorthand constructors ---------------------------------------------------

    fn h(q: usize) -> Gate {
        Gate::new(GateType::H, vec![q], vec![])
    }

    fn x(q: usize) -> Gate {
        Gate::new(GateType::X, vec![q], vec![])
    }

    fn y(q: usize) -> Gate {
        Gate::new(GateType::Y, vec![q], vec![])
    }

    fn z(q: usize) -> Gate {
        Gate::new(GateType::Z, vec![q], vec![])
    }

    fn s(q: usize) -> Gate {
        Gate::new(GateType::S, vec![q], vec![])
    }

    fn t(q: usize) -> Gate {
        Gate::new(GateType::T, vec![q], vec![])
    }

    fn cnot(ctrl: usize, tgt: usize) -> Gate {
        Gate::new(GateType::CNOT, vec![tgt], vec![ctrl])
    }

    fn cz(a: usize, b: usize) -> Gate {
        Gate::new(GateType::CZ, vec![b], vec![a])
    }

    fn swap(a: usize, b: usize) -> Gate {
        Gate::new(GateType::SWAP, vec![a, b], vec![])
    }

    fn phase(q: usize, theta: f64) -> Gate {
        Gate::new(GateType::Phase(theta), vec![q], vec![])
    }

    fn ry(q: usize, theta: f64) -> Gate {
        Gate::new(GateType::Ry(theta), vec![q], vec![])
    }

    fn rx(q: usize, theta: f64) -> Gate {
        Gate::new(GateType::Rx(theta), vec![q], vec![])
    }

    fn rz(q: usize, theta: f64) -> Gate {
        Gate::new(GateType::Rz(theta), vec![q], vec![])
    }

    // --- 1. Identity circuit equivalent to empty ---

    #[test]
    fn test_empty_circuits_equivalent() {
        let checker = CircuitEquivalenceChecker::new();
        let empty: Vec<Gate> = vec![];
        let result = checker.check(&empty, &empty, 2, EquivalenceMethod::MatrixComparison);
        assert!(result.equivalent);
        assert!((result.fidelity - 1.0).abs() < 1e-12);
    }

    // --- 2. HH = I ---

    #[test]
    fn test_hh_equals_identity() {
        let checker = CircuitEquivalenceChecker::new();
        let hh = vec![h(0), h(0)];
        let identity: Vec<Gate> = vec![];
        let result = checker.check(&hh, &identity, 1, EquivalenceMethod::MatrixComparison);
        assert!(result.equivalent, "HH should equal identity");
        assert!((result.fidelity - 1.0).abs() < 1e-12);
    }

    // --- 3. CNOT CNOT = I ---

    #[test]
    fn test_cnot_cnot_equals_identity() {
        let checker = CircuitEquivalenceChecker::new();
        let cc = vec![cnot(0, 1), cnot(0, 1)];
        let identity: Vec<Gate> = vec![];
        let result = checker.check(&cc, &identity, 2, EquivalenceMethod::MatrixComparison);
        assert!(result.equivalent, "CNOT*CNOT should equal identity");
    }

    // --- 4. X != Z (with counterexample) ---

    #[test]
    fn test_x_not_equal_z_with_counterexample() {
        let checker = CircuitEquivalenceChecker::new();
        let circ_x = vec![x(0)];
        let circ_z = vec![z(0)];
        let result = checker.check(&circ_x, &circ_z, 1, EquivalenceMethod::MatrixComparison);
        assert!(!result.equivalent, "X gate should not equal Z gate");
        assert!(
            result.counterexample.is_some(),
            "Should provide a counterexample"
        );
    }

    // --- 5. Global phase detection: Z vs Phase(pi) ---

    #[test]
    fn test_global_phase_detection() {
        // Z = diag(1, -1).  Phase(pi) = diag(1, e^{i*pi}) = diag(1, -1).
        // These are exactly the same unitary, so should be equivalent with
        // trivial (identity) phase.
        let checker = CircuitEquivalenceChecker::new();
        let circ_z = vec![z(0)];
        let circ_p = vec![phase(0, PI)];
        let result = checker.check(&circ_z, &circ_p, 1, EquivalenceMethod::MatrixComparison);
        assert!(result.equivalent, "Z and Phase(pi) should be equivalent");
    }

    // --- 6. S*S = Z (global phase equivalence) ---

    #[test]
    fn test_ss_equals_z() {
        let checker = CircuitEquivalenceChecker::new();
        let ss = vec![s(0), s(0)];
        let circ_z = vec![z(0)];
        let result = checker.check(&ss, &circ_z, 1, EquivalenceMethod::MatrixComparison);
        assert!(result.equivalent, "S*S should equal Z");
    }

    // --- 7. Statistical method on a 2-qubit circuit ---

    #[test]
    fn test_statistical_sampling_equivalent() {
        let checker = CircuitEquivalenceChecker::new();
        // SWAP decomposition: CNOT(0,1) CNOT(1,0) CNOT(0,1) == SWAP(0,1)
        let decomposed = vec![cnot(0, 1), cnot(1, 0), cnot(0, 1)];
        let native = vec![swap(0, 1)];
        let result = checker.check(
            &decomposed,
            &native,
            2,
            EquivalenceMethod::StatisticalSampling { num_samples: 200 },
        );
        assert!(
            result.equivalent,
            "CNOT decomposition of SWAP should be equivalent"
        );
    }

    // --- 8. Statistical sampling detects non-equivalence ---

    #[test]
    fn test_statistical_sampling_not_equivalent() {
        let checker = CircuitEquivalenceChecker::new();
        let circ_a = vec![h(0), cnot(0, 1)];
        let circ_b = vec![x(0), cnot(0, 1)];
        let result = checker.check(
            &circ_a,
            &circ_b,
            2,
            EquivalenceMethod::StatisticalSampling { num_samples: 100 },
        );
        assert!(
            !result.equivalent,
            "H-CNOT and X-CNOT should not be equivalent"
        );
    }

    // --- 9. Auto method selection (small circuit -> MatrixComparison) ---

    #[test]
    fn test_auto_selects_matrix_for_small() {
        let checker = CircuitEquivalenceChecker::new();
        let a = vec![h(0)];
        let b = vec![h(0)];
        let result = checker.check(&a, &b, 3, EquivalenceMethod::Auto);
        assert!(result.equivalent);
        assert_eq!(result.method_used, EquivalenceMethod::MatrixComparison);
    }

    // --- 10. Optimization verification wrapper ---

    #[test]
    fn test_verify_optimization() {
        let original = vec![h(0), z(0), h(0)];
        // HZH = X (up to global phase)
        let optimized = vec![x(0)];
        let result = verify_optimization(&original, &optimized, 1);
        assert!(result.equivalent, "HZH should equal X");
    }

    // --- 11. Tolerance handling ---

    #[test]
    fn test_tolerance_comparison() {
        let checker = CircuitEquivalenceChecker::new();
        // Rx(epsilon) is almost identity for tiny epsilon.
        let eps = 1e-12;
        let almost_id = vec![rx(0, eps)];
        let identity: Vec<Gate> = vec![];
        let result = checker.check(
            &almost_id,
            &identity,
            1,
            EquivalenceMethod::MatrixComparison,
        );
        // With default tolerance 1e-10, a rotation of 1e-12 should be
        // within tolerance of identity.
        assert!(
            result.equivalent,
            "Rx(1e-12) should be within tolerance of identity"
        );
    }

    // --- 12. Tight tolerance rejects near-miss ---

    #[test]
    fn test_tight_tolerance_rejects() {
        let checker = CircuitEquivalenceChecker::with_tolerance(1e-15);
        let eps = 1e-6;
        let almost_id = vec![rx(0, eps)];
        let identity: Vec<Gate> = vec![];
        let result = checker.check(
            &almost_id,
            &identity,
            1,
            EquivalenceMethod::MatrixComparison,
        );
        assert!(
            !result.equivalent,
            "Rx(1e-6) should NOT be within 1e-15 tolerance"
        );
    }

    // --- 13. XX = I ---

    #[test]
    fn test_xx_equals_identity() {
        let checker = CircuitEquivalenceChecker::new();
        let xx = vec![x(0), x(0)];
        let identity: Vec<Gate> = vec![];
        let result = checker.check(&xx, &identity, 1, EquivalenceMethod::MatrixComparison);
        assert!(result.equivalent, "X*X should equal identity");
    }

    // --- 14. Multi-qubit: H on each qubit ---

    #[test]
    fn test_multi_qubit_hh_identity() {
        let checker = CircuitEquivalenceChecker::new();
        let hh_all = vec![h(0), h(1), h(2), h(0), h(1), h(2)];
        let identity: Vec<Gate> = vec![];
        let result = checker.check(&hh_all, &identity, 3, EquivalenceMethod::MatrixComparison);
        assert!(
            result.equivalent,
            "H^2 on every qubit should equal identity"
        );
    }

    // --- 15. Symbolic tracking: Clifford circuit ---

    #[test]
    fn test_symbolic_clifford_equivalent() {
        let checker = CircuitEquivalenceChecker::new();
        // Two decompositions of CZ: CZ vs H-CNOT-H
        let circ_a = vec![cz(0, 1)];
        let circ_b = vec![h(1), cnot(0, 1), h(1)];
        let result = checker.check(&circ_a, &circ_b, 2, EquivalenceMethod::SymbolicTracking);
        assert!(
            result.equivalent,
            "CZ should equal H-CNOT-H via symbolic tracking"
        );
    }

    // --- 16. Symbolic tracking: non-equivalent Clifford circuits ---

    #[test]
    fn test_symbolic_clifford_not_equivalent() {
        let checker = CircuitEquivalenceChecker::new();
        let circ_a = vec![h(0), cnot(0, 1)];
        let circ_b = vec![cnot(0, 1), h(0)];
        let result = checker.check(&circ_a, &circ_b, 2, EquivalenceMethod::SymbolicTracking);
        assert!(
            !result.equivalent,
            "H-CNOT and CNOT-H should not be equivalent"
        );
    }

    // --- 17. Build unitary correctness: identity circuit ---

    #[test]
    fn test_build_unitary_identity() {
        let identity: Vec<Gate> = vec![];
        let u = build_unitary(&identity, 2);
        let dim = 4;
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (u[i][j].re - expected).abs() < 1e-12 && u[i][j].im.abs() < 1e-12,
                    "Identity unitary mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    // --- 18. Find counterexample function ---

    #[test]
    fn test_find_counterexample() {
        let circ_a = vec![x(0)];
        let circ_b = vec![z(0)];
        let ce = find_counterexample(&circ_a, &circ_b, 1);
        assert!(ce.is_some(), "Should find a counterexample for X vs Z");
    }
}
