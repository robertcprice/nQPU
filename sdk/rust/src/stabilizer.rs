//! Stabilizer Simulation (Tableau Method)
//!
//! This module implements efficient simulation of Clifford circuits using
//! the stabilizer formalism (also known as the tableau method or CH simulator).
//!
//! Clifford gates are gates that map Pauli operators to Pauli operators.
//! They include: H, S, CNOT, CZ, SWAP, and Pauli gates.
//!
//! The stabilizer representation allows simulation of n-qubit Clifford states
//! using O(n²) memory instead of O(2ⁿ) for state vector simulation.

use std::fmt;

/// A stabilizer state represented using the tableau formalism
///
/// For n qubits, we maintain:
/// - 2n generators of the stabilizer group
/// - Each generator is a Pauli operator (product of X, Y, Z, I)
/// - Represented using binary vectors (x, z) for each generator
#[derive(Clone, Debug)]
pub struct StabilizerState {
    /// Number of qubits
    num_qubits: usize,
    /// X part of the tableau (2n x n binary matrix)
    /// tableau.x[i][j] = 1 if generator i has X on qubit j
    x: Vec<Vec<bool>>,
    /// Z part of the tableau (2n x n binary matrix)
    /// tableau.z[i][j] = 1 if generator i has Z on qubit j
    z: Vec<Vec<bool>>,
    /// Phase bits for each generator (0 = +1, 1 = -1, 2 = +i, 3 = -i)
    phases: Vec<u8>,
}

impl StabilizerState {
    /// Create a new stabilizer state in the |0...0⟩ computational basis state
    ///
    /// For the |0⟩ state on qubit j, the stabilizers are Z₁, Z₂, ..., Zₙ
    pub fn new(num_qubits: usize) -> Self {
        let n = num_qubits;
        let mut x = vec![vec![false; n]; 2 * n];
        let mut z = vec![vec![false; n]; 2 * n];
        let phases = vec![0u8; 2 * n];

        // For |0...0⟩ state, stabilizers are Z₁, Z₂, ..., Zₙ
        for i in 0..n {
            z[i][i] = true; // Z on qubit i
        }

        // Destabilizers (not used for simulation but kept for completeness)
        for i in n..2 * n {
            x[i][i - n] = true; // X on qubit (i-n)
        }

        StabilizerState {
            num_qubits: n,
            x,
            z,
            phases,
        }
    }

    /// Create a stabilizer state from a computational basis state
    pub fn from_basis_state(num_qubits: usize, state: usize) -> Self {
        let mut stab = Self::new(num_qubits);

        // Flip qubits where state bit is 1 (apply X gates)
        for qubit in 0..num_qubits {
            if (state >> qubit) & 1 == 1 {
                stab.x(qubit);
            }
        }

        stab
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Crate-internal access to X tableau
    pub(crate) fn x_tableau(&self) -> &Vec<Vec<bool>> {
        &self.x
    }

    /// Crate-internal access to Z tableau
    pub(crate) fn z_tableau(&self) -> &Vec<Vec<bool>> {
        &self.z
    }

    /// Crate-internal access to phase vector
    pub(crate) fn phases(&self) -> &Vec<u8> {
        &self.phases
    }

    /// Check if this is a valid stabilizer state
    pub fn is_valid(&self) -> bool {
        // Check that stabilizer generators commute with each other
        for i in 0..self.num_qubits {
            for j in (i + 1)..self.num_qubits {
                if !self.commute(i, j) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if two generators commute
    fn commute(&self, i: usize, j: usize) -> bool {
        let mut sum = 0u32;
        for k in 0..self.num_qubits {
            // Paulis commute iff their commutator has even number of Y's
            let xi_zj = (self.x[i][k] && self.z[j][k]) as u32;
            let zi_xj = (self.z[i][k] && self.x[j][k]) as u32;
            sum += xi_zj + zi_xj;
        }
        sum % 2 == 0
    }

    /// Measure a qubit in the computational basis
    ///
    /// Returns the measurement result (0 or 1) and updates the state
    pub fn measure(&mut self, qubit: usize) -> bool {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        // Find if any generator has X on this qubit
        let mut generator_with_x = None;
        for i in 0..self.num_qubits {
            if self.x[i][qubit] {
                generator_with_x = Some(i);
                break;
            }
        }

        match generator_with_x {
            Some(p) => {
                // Random measurement result
                let result = rand::random::<bool>();

                if result {
                    // Flip the sign of generator p
                    self.phases[p] ^= 1;
                }

                // Update all generators to have Z on this qubit
                for i in 0..(2 * self.num_qubits) {
                    if i != p && self.x[i][qubit] {
                        // Multiply generator i by generator p
                        self.row_mult(i, p);
                    }
                }

                result
            }
            None => {
                // Deterministic result: check if any generator has Z with phase -1
                let mut result = false;
                for i in 0..self.num_qubits {
                    if self.z[i][qubit] && self.phases[i] == 1 {
                        result = true;
                        break;
                    }
                }

                // No state update needed for deterministic measurement
                result
            }
        }
    }

    /// Multiply row i by row p in the tableau
    fn row_mult(&mut self, i: usize, p: usize) {
        // Update x[i] and z[i]
        for k in 0..self.num_qubits {
            let x_p = self.x[p][k];
            let z_p = self.z[p][k];

            if x_p && z_p {
                // Y = iXZ, which adds a phase
                self.phases[i] ^= 1;
            }

            if x_p {
                self.x[i][k] ^= self.z[p][k];
            }
            if z_p {
                self.z[i][k] ^= self.x[p][k];
            }
        }
    }

    /// Apply Hadamard gate to a qubit
    ///
    /// H: X → Z, Z → X
    pub fn h(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        for i in 0..(2 * self.num_qubits) {
            std::mem::swap(&mut self.x[i][qubit], &mut self.z[i][qubit]);

            // Y = iXZ → -iZX = -Y, so flip phase if both were set
            if self.x[i][qubit] && self.z[i][qubit] {
                self.phases[i] ^= 1;
            }
        }
    }

    /// Apply S gate (phase gate, √Z) to a qubit
    ///
    /// S: X → Y, Z → Z
    pub fn s(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        for i in 0..(2 * self.num_qubits) {
            // S: X → Y = iXZ
            if self.x[i][qubit] && !self.z[i][qubit] {
                self.z[i][qubit] = true;
                self.phases[i] ^= 1; // Y has phase i
            }
            // S: Y → -X
            else if self.x[i][qubit] && self.z[i][qubit] {
                self.z[i][qubit] = false;
                self.phases[i] ^= 1; // -Y has phase -i
            }
        }
    }

    /// Apply S† gate (inverse phase gate) to a qubit
    ///
    /// S†: X → -Y, Z → Z
    pub fn s_dag(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        for i in 0..(2 * self.num_qubits) {
            // S†: Y → X
            if self.x[i][qubit] && self.z[i][qubit] {
                self.z[i][qubit] = false;
                self.phases[i] ^= 1;
            }
            // S†: X → -Y
            else if self.x[i][qubit] && !self.z[i][qubit] {
                self.z[i][qubit] = true;
                self.phases[i] ^= 1;
            }
        }
    }

    /// Apply Pauli-X gate to a qubit
    ///
    /// X: Z → -Z
    pub fn x(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        for i in 0..(2 * self.num_qubits) {
            if self.z[i][qubit] {
                self.phases[i] ^= 1;
            }
        }
    }

    /// Apply Pauli-Y gate to a qubit
    ///
    /// Y: X → -X, Z → -Z
    pub fn y(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        for i in 0..(2 * self.num_qubits) {
            if self.x[i][qubit] {
                self.phases[i] ^= 1;
            }
            if self.z[i][qubit] {
                self.phases[i] ^= 1;
            }
        }
    }

    /// Apply Pauli-Z gate to a qubit
    ///
    /// Z: X → -X
    pub fn z(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        for i in 0..(2 * self.num_qubits) {
            if self.x[i][qubit] {
                self.phases[i] ^= 1;
            }
        }
    }

    /// Apply CNOT gate (controlled-NOT)
    ///
    /// CNOT(c, t): X_c → X_c X_t, X_t → X_t
    ///             Z_c → Z_c,     Z_t → Z_c Z_t
    pub fn cx(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits, "Control qubit out of bounds");
        assert!(target < self.num_qubits, "Target qubit out of bounds");
        assert!(control != target, "Control and target must be different");

        for i in 0..(2 * self.num_qubits) {
            let x_c = self.x[i][control];
            let z_c = self.z[i][control];
            let x_t = self.x[i][target];
            let z_t = self.z[i][target];

            // Phase update for CNOT (Aaronson-Gottesman)
            if x_c && z_t && (x_t ^ z_c ^ true) {
                self.phases[i] ^= 1;
            }

            self.x[i][target] = x_t ^ x_c;
            self.z[i][control] = z_c ^ z_t;
        }
    }

    /// Apply CZ gate (controlled-Z)
    ///
    /// CZ(c, t): X_c → Z_c X_c, X_t → Z_t X_t
    ///            Z_c → Z_c,     Z_t → Z_t
    pub fn cz(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits, "Control qubit out of bounds");
        assert!(target < self.num_qubits, "Target qubit out of bounds");
        assert!(control != target, "Control and target must be different");

        for i in 0..(2 * self.num_qubits) {
            let x_c = self.x[i][control];
            let z_c = self.z[i][control];
            let x_t = self.x[i][target];
            let z_t = self.z[i][target];

            // Phase update for CZ (Aaronson-Gottesman)
            if x_c && x_t && (z_c ^ z_t ^ true) {
                self.phases[i] ^= 1;
            }

            self.z[i][control] = z_c ^ x_t;
            self.z[i][target] = z_t ^ x_c;
        }
    }

    /// Apply SWAP gate
    ///
    /// SWAP(a, b): Swaps qubits a and b
    pub fn swap(&mut self, a: usize, b: usize) {
        assert!(a < self.num_qubits, "Qubit a out of bounds");
        assert!(b < self.num_qubits, "Qubit b out of bounds");
        assert!(a != b, "Cannot swap a qubit with itself");

        for i in 0..(2 * self.num_qubits) {
            self.x[i].swap(a, b);
            self.z[i].swap(a, b);
        }
    }

    /// Calculate the probability of a specific measurement outcome
    pub fn probability_of(&self, qubit: usize, outcome: bool) -> f64 {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        // Find if any generator has X on this qubit
        for i in 0..self.num_qubits {
            if self.x[i][qubit] {
                // 50/50 probability
                return 0.5;
            }
        }

        // Deterministic: check if outcome matches the eigenvalue
        for i in 0..self.num_qubits {
            if self.z[i][qubit] {
                let is_minus = self.phases[i] == 1;
                let state_is_one = is_minus;
                return if outcome == state_is_one { 1.0 } else { 0.0 };
            }
        }

        // Should not reach here for valid stabilizer states
        0.5
    }

    /// Clone the stabilizer state
    pub fn clone_state(&self) -> Self {
        self.clone()
    }

    /// Check if the state is a product state (no entanglement)
    pub fn is_product_state(&self) -> bool {
        // A state is product if each qubit can be measured independently
        // This is a simplified check
        for qubit in 0..self.num_qubits {
            let mut has_x = false;
            for i in 0..self.num_qubits {
                if self.x[i][qubit] {
                    has_x = true;
                    break;
                }
            }
            if has_x {
                // Check if the X couples to other qubits
                for i in 0..self.num_qubits {
                    if self.x[i][qubit] {
                        for other in 0..self.num_qubits {
                            if other != qubit && (self.x[i][other] || self.z[i][other]) {
                                return false; // Entangled
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Get the expected value (expectation) of Pauli-Z on a qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let p0 = self.probability_of(qubit, false);
        let p1 = self.probability_of(qubit, true);
        p0 - p1
    }
}

impl fmt::Display for StabilizerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "StabilizerState ({} qubits):", self.num_qubits)?;
        writeln!(f, "  Generators:")?;
        for i in 0..self.num_qubits {
            write!(f, "    g{}: ", i)?;
            for j in 0..self.num_qubits {
                let pauli = match (self.x[i][j], self.z[i][j]) {
                    (false, false) => "I",
                    (true, false) => "X",
                    (false, true) => "Z",
                    (true, true) => "Y",
                };
                write!(f, "{}", pauli)?;
            }
            let phase = if self.phases[i] == 0 { "+" } else { "-" };
            writeln!(f, " [{}]", phase)?;
        }
        Ok(())
    }
}

/// Simulator for Clifford circuits using stabilizer formalism
pub struct StabilizerSimulator {
    state: StabilizerState,
}

impl StabilizerSimulator {
    pub fn new(num_qubits: usize) -> Self {
        StabilizerSimulator {
            state: StabilizerState::new(num_qubits),
        }
    }

    pub fn from_basis_state(num_qubits: usize, state: usize) -> Self {
        StabilizerSimulator {
            state: StabilizerState::from_basis_state(num_qubits, state),
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits()
    }

    pub fn h(&mut self, qubit: usize) {
        self.state.h(qubit);
    }

    pub fn s(&mut self, qubit: usize) {
        self.state.s(qubit);
    }

    pub fn s_dag(&mut self, qubit: usize) {
        self.state.s_dag(qubit);
    }

    pub fn x(&mut self, qubit: usize) {
        self.state.x(qubit);
    }

    pub fn y(&mut self, qubit: usize) {
        self.state.y(qubit);
    }

    pub fn z(&mut self, qubit: usize) {
        self.state.z(qubit);
    }

    pub fn cx(&mut self, control: usize, target: usize) {
        self.state.cx(control, target);
    }

    pub fn cz(&mut self, control: usize, target: usize) {
        self.state.cz(control, target);
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.state.swap(a, b);
    }

    pub fn measure(&mut self, qubit: usize) -> bool {
        self.state.measure(qubit)
    }

    pub fn probability_of(&self, qubit: usize, outcome: bool) -> f64 {
        self.state.probability_of(qubit, outcome)
    }

    pub fn expectation_z(&self, qubit: usize) -> f64 {
        self.state.expectation_z(qubit)
    }

    pub fn is_product_state(&self) -> bool {
        self.state.is_product_state()
    }

    pub fn clone(&self) -> Self {
        StabilizerSimulator {
            state: self.state.clone_state(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stabilizer_creation() {
        let state = StabilizerState::new(3);
        assert_eq!(state.num_qubits(), 3);
        assert!(state.is_valid());
    }

    #[test]
    fn test_hadamard() {
        let mut state = StabilizerState::new(1);
        // |0⟩ after H should be |+⟩ = (|0⟩ + |1⟩)/√2
        state.h(0);
        // After H, measuring should give 50/50
        let p0 = state.probability_of(0, false);
        assert!((p0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_entanglement() {
        let mut state = StabilizerState::new(2);
        // Create Bell state: H(0), CNOT(0,1)
        state.h(0);
        state.cx(0, 1);

        // Bell state should be entangled
        assert!(!state.is_product_state());

        // Measurements should be correlated
        let p00 = state.probability_of(0, false);
        // After measuring qubit 0 as 0, qubit 1 should also be 0
        // But we can't test this directly without measurement
        assert!((p00 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_phase_gate() {
        let mut state = StabilizerState::new(1);
        state.h(0);
        state.s(0);
        // |+⟩ after S should be |R⟩ = (|0⟩ + i|1⟩)/√2

        // Measurement should still be 50/50
        let p0 = state.probability_of(0, false);
        assert!((p0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_measurement() {
        let state = StabilizerState::new(1);
        // |0⟩ state should always measure as 0
        let p0 = state.probability_of(0, false);
        assert!((p0 - 1.0).abs() < 1e-10);

        let mut state = StabilizerState::new(1);
        state.x(0);
        // |1⟩ state should always measure as 1
        let p1 = state.probability_of(0, true);
        assert!((p1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_swap() {
        let mut state = StabilizerState::from_basis_state(2, 0b01); // |01⟩
        state.swap(0, 1);
        // Should now be |10⟩

        // Qubit 0 should now be |0⟩, qubit 1 should be |1⟩
        let p0_q0 = state.probability_of(0, false);
        assert!((p0_q0 - 1.0).abs() < 1e-10);

        let p1_q1 = state.probability_of(1, true);
        assert!((p1_q1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clifford_group() {
        let mut state = StabilizerState::new(2);
        // Test that Clifford operations preserve stabilizer structure
        state.h(0);
        state.s(0);
        state.cx(0, 1);
        state.h(1);
        state.cz(0, 1);

        assert!(state.is_valid());
    }

    #[test]
    fn test_expectation_z() {
        let mut state = StabilizerState::new(1);
        // |0⟩ has ⟨Z⟩ = 1
        let exp = state.expectation_z(0);
        assert!((exp - 1.0).abs() < 1e-10);

        state.h(0);
        // |+⟩ has ⟨Z⟩ = 0
        let exp = state.expectation_z(0);
        assert!(exp.abs() < 1e-10);

        state.x(0);
        // |-⟩ has ⟨Z⟩ = 0
        let exp = state.expectation_z(0);
        assert!(exp.abs() < 1e-10);
    }

    #[test]
    fn test_simulator() {
        let mut sim = StabilizerSimulator::new(3);
        assert_eq!(sim.num_qubits(), 3);

        // Create GHZ-like state
        sim.h(0);
        sim.cx(0, 1);
        sim.cx(1, 2);

        assert!(!sim.is_product_state());

        // All qubits should have equal probability of 0 and 1
        for qubit in 0..3 {
            let p0 = sim.probability_of(qubit, false);
            assert!((p0 - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_basis_state() {
        let state = StabilizerState::from_basis_state(3, 0b101); // |101⟩
        assert!(state.is_valid());

        // Check deterministic measurements
        let p1_q0 = state.probability_of(0, true);
        assert!((p1_q0 - 1.0).abs() < 1e-10);

        let p0_q1 = state.probability_of(1, false);
        assert!((p0_q1 - 1.0).abs() < 1e-10);

        let p1_q2 = state.probability_of(2, true);
        assert!((p1_q2 - 1.0).abs() < 1e-10);
    }
}
