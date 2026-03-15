//! Variational Quantum Eigensolver (VQE)
//!
//! This module implements the VQE algorithm for finding ground states
//! of quantum Hamiltonians, a key application for quantum chemistry
//! and materials science.

use crate::advanced_quantum_ml::VariationalQuantumCircuit;
use crate::QuantumState;
use std::f64::consts::PI;

/// Pauli operator for Hamiltonian representation
#[derive(Clone, Debug, PartialEq)]
pub enum PauliOperator {
    I(usize), // Identity on qubit i
    X(usize), // Pauli-X on qubit i
    Y(usize), // Pauli-Y on qubit i
    Z(usize), // Pauli-Z on qubit i
}

impl PauliOperator {
    /// Get the qubit index this operator acts on
    pub fn qubit(&self) -> usize {
        match self {
            PauliOperator::I(q)
            | PauliOperator::X(q)
            | PauliOperator::Y(q)
            | PauliOperator::Z(q) => *q,
        }
    }

    /// Compute expectation value ⟨ψ|P|ψ⟩
    pub fn expectation(&self, state: &QuantumState) -> f64 {
        match self {
            PauliOperator::I(_) => 1.0,
            PauliOperator::Z(q) => state.expectation_z(*q),
            PauliOperator::X(q) => {
                // ⟨X⟩ = ⟨ψ|X|ψ⟩ = ⟨ψ|H·Z·H|ψ⟩
                let mut state_copy = state.clone();
                crate::GateOperations::h(&mut state_copy, *q);
                state_copy.expectation_z(*q)
            }
            PauliOperator::Y(q) => {
                // ⟨Y⟩ = ⟨ψ|Y|ψ⟩ = ⟨ψ|H·S†·Z·S·H|ψ⟩
                let mut state_copy = state.clone();
                crate::GateOperations::h(&mut state_copy, *q);
                crate::GateOperations::s(&mut state_copy, *q);
                crate::GateOperations::s(&mut state_copy, *q);
                crate::GateOperations::z(&mut state_copy, *q);
                crate::GateOperations::h(&mut state_copy, *q);
                state_copy.expectation_z(*q)
            }
        }
    }
}

/// Pauli term: product of Pauli operators with coefficient
#[derive(Clone, Debug)]
pub struct PauliTerm {
    pub operators: Vec<PauliOperator>,
    pub coefficient: f64,
}

impl PauliTerm {
    /// Create a new Pauli term
    pub fn new(operators: Vec<PauliOperator>, coefficient: f64) -> Self {
        PauliTerm {
            operators,
            coefficient,
        }
    }

    /// Compute expectation value ⟨ψ|P|ψ⟩ for this term
    pub fn expectation(&self, state: &QuantumState) -> f64 {
        // For product operators, we need to measure each separately
        // and take the product (in the full implementation)
        let mut result = self.coefficient;

        for op in &self.operators {
            result *= op.expectation(state);
        }

        result
    }

    /// Check if this is a single-qubit term (easier to measure)
    pub fn is_single_qubit(&self) -> bool {
        self.operators.len() == 1
    }
}

/// Hamiltonian: sum of Pauli terms
#[derive(Clone, Debug)]
pub struct Hamiltonian {
    pub terms: Vec<PauliTerm>,
}

impl Hamiltonian {
    /// Create a new Hamiltonian
    pub fn new(terms: Vec<PauliTerm>) -> Self {
        Hamiltonian { terms }
    }

    /// Compute expectation value ⟨ψ|H|ψ⟩
    pub fn expectation(&self, state: &QuantumState) -> f64 {
        self.terms.iter().map(|term| term.expectation(state)).sum()
    }

    /// Get the number of qubits in this Hamiltonian
    pub fn num_qubits(&self) -> usize {
        self.terms
            .iter()
            .flat_map(|term| term.operators.iter().map(|op| op.qubit()))
            .max()
            .map(|q| q + 1)
            .unwrap_or(0)
    }
}

/// VQE solver
pub struct VQESolver {
    pub ansatz: VariationalQuantumCircuit,
    pub hamiltonian: Hamiltonian,
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

impl VQESolver {
    /// Create a new VQE solver
    pub fn new(
        num_qubits: usize,
        ansatz_depth: usize,
        hamiltonian: Hamiltonian,
        learning_rate: f64,
    ) -> Self {
        let ansatz = VariationalQuantumCircuit::new(num_qubits, ansatz_depth);

        VQESolver {
            ansatz,
            hamiltonian,
            learning_rate,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
        }
    }

    /// Run VQE to find the ground state energy
    pub fn find_ground_state(&mut self) -> VQEResult {
        let mut energy_history = Vec::new();
        let mut prev_energy = f64::INFINITY;

        for iteration in 0..self.max_iterations {
            // Compute current energy
            let energy = self.compute_energy();
            energy_history.push(energy);

            // Check convergence
            let energy_change = (energy - prev_energy).abs();
            if energy_change < self.convergence_threshold {
                return VQEResult {
                    ground_state_energy: energy,
                    parameters: self.ansatz.get_parameters(),
                    iterations: iteration + 1,
                    converged: true,
                    energy_history,
                };
            }

            // Compute gradients and update parameters
            self.update_parameters();

            prev_energy = energy;
        }

        VQEResult {
            ground_state_energy: prev_energy,
            parameters: self.ansatz.get_parameters(),
            iterations: self.max_iterations,
            converged: false,
            energy_history,
        }
    }

    /// Compute energy ⟨ψ(θ)|H|ψ(θ)⟩
    fn compute_energy(&self) -> f64 {
        let mut state = QuantumState::new(self.ansatz.num_qubits);

        // Execute ansatz with current parameters
        self.ansatz.execute(&mut state);

        // Compute expectation value of Hamiltonian
        self.hamiltonian.expectation(&state)
    }

    /// Update parameters using gradient descent
    fn update_parameters(&mut self) {
        let current_params = self.ansatz.get_parameters();
        let mut gradients = vec![0.0; current_params.len()];

        // Compute gradient for each parameter using parameter-shift rule
        for (param_idx, &_param) in current_params.iter().enumerate() {
            let shift = PI / 2.0;

            // Forward shift
            let mut params_plus = current_params.clone();
            params_plus[param_idx] += shift;
            let energy_plus = self.compute_energy_with_params(&params_plus);

            // Backward shift
            let mut params_minus = current_params.clone();
            params_minus[param_idx] -= shift;
            let energy_minus = self.compute_energy_with_params(&params_minus);

            // Parameter-shift rule gradient
            gradients[param_idx] = (energy_plus - energy_minus) / 2.0;
        }

        // Update parameters
        let mut new_params = current_params;
        for (i, &grad) in gradients.iter().enumerate() {
            new_params[i] -= self.learning_rate * grad;
        }

        let _ = self.ansatz.set_parameters(new_params);
    }

    /// Compute energy with given parameters
    fn compute_energy_with_params(&self, params: &[f64]) -> f64 {
        let mut state = QuantumState::new(self.ansatz.num_qubits);

        // Create temporary ansatz with given parameters
        let mut temp_ansatz = self.ansatz.clone();
        let _ = temp_ansatz.set_parameters(params.to_vec());
        temp_ansatz.execute(&mut state);

        self.hamiltonian.expectation(&state)
    }
}

/// VQE result
#[derive(Clone, Debug)]
pub struct VQEResult {
    pub ground_state_energy: f64,
    pub parameters: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub energy_history: Vec<f64>,
}

/// Common Hamiltonians for testing
pub mod hamiltonians {
    use super::*;

    /// Transverse-field Ising model Hamiltonian
    ///
    /// H = -J∑⟨i,j⟩ Z_i Z_j - h∑_i X_i
    pub fn transverse_field_ising(num_qubits: usize, j_coupling: f64, h_field: f64) -> Hamiltonian {
        let mut terms = Vec::new();

        // ZZ interaction terms
        for i in 0..num_qubits - 1 {
            terms.push(PauliTerm::new(
                vec![PauliOperator::Z(i), PauliOperator::Z(i + 1)],
                -j_coupling,
            ));
        }

        // X field terms
        for i in 0..num_qubits {
            terms.push(PauliTerm::new(vec![PauliOperator::X(i)], -h_field));
        }

        Hamiltonian::new(terms)
    }

    /// Heisenberg XYZ model Hamiltonian
    ///
    /// H = J_x∑ X_i X_j + J_y∑ Y_i Y_j + J_z∑ Z_i Z_j
    pub fn heisenberg_xyz(num_qubits: usize, jx: f64, jy: f64, jz: f64) -> Hamiltonian {
        let mut terms = Vec::new();

        for i in 0..num_qubits - 1 {
            terms.push(PauliTerm::new(
                vec![PauliOperator::X(i), PauliOperator::X(i + 1)],
                jx,
            ));
            terms.push(PauliTerm::new(
                vec![PauliOperator::Y(i), PauliOperator::Y(i + 1)],
                jy,
            ));
            terms.push(PauliTerm::new(
                vec![PauliOperator::Z(i), PauliOperator::Z(i + 1)],
                jz,
            ));
        }

        Hamiltonian::new(terms)
    }

    /// Hydrogen molecule Hamiltonian (simplified 2-qubit version)
    ///
    /// H = g0*I + g1*Z0 + g2*Z1 + g3*Z0*Z1 + g4*X0*X1 + g5*Y0*Y1
    pub fn hydrogen_molecule(g0: f64, g1: f64, g2: f64, g3: f64, g4: f64, g5: f64) -> Hamiltonian {
        let terms = vec![
            PauliTerm::new(vec![PauliOperator::I(0)], g0),
            PauliTerm::new(vec![PauliOperator::Z(0)], g1),
            PauliTerm::new(vec![PauliOperator::Z(1)], g2),
            PauliTerm::new(vec![PauliOperator::Z(0), PauliOperator::Z(1)], g3),
            PauliTerm::new(vec![PauliOperator::X(0), PauliOperator::X(1)], g4),
            PauliTerm::new(vec![PauliOperator::Y(0), PauliOperator::Y(1)], g5),
        ];

        Hamiltonian::new(terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_expectation() {
        let mut state = QuantumState::new(1);
        crate::GateOperations::h(&mut state, 0);

        // After H, ⟨Z⟩ = 0
        let z_exp = PauliOperator::Z(0).expectation(&state);
        assert!((z_exp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamiltonian_creation() {
        let h = hamiltonians::transverse_field_ising(2, 1.0, 0.5);
        assert_eq!(h.num_qubits(), 2);
        assert!(!h.terms.is_empty());
    }

    #[test]
    fn test_vqe_basic() {
        let h = hamiltonians::transverse_field_ising(2, 1.0, 0.5);
        let mut solver = VQESolver::new(2, 2, h, 0.1);

        let result = solver.find_ground_state();
        assert!(!result.energy_history.is_empty());
    }
}
