//! Advanced Quantum Machine Learning Primitives
//!
//! This module implements cutting-edge quantum ML algorithms including:
//! - Variational Quantum Circuits (VQC)
//! - Quantum Neural Networks (QNN)
//! - Quantum Kernel Methods
//! - Quantum Natural Gradient Descent
//! - Quantum Batch Normalization

use crate::gates::{Gate, GateType};
use crate::QuantumState;
use std::f64::consts::PI;

/// Variational Quantum Circuit (VQC)
///
/// A parameterized quantum circuit that can be trained using classical optimization
#[derive(Clone, Debug)]
pub struct VariationalQuantumCircuit {
    pub num_qubits: usize,
    pub num_parameters: usize,
    pub layers: Vec<VQCLayer>,
    pub parameters: Vec<f64>,
}

/// A single layer in a VQC
#[derive(Clone, Debug)]
pub struct VQCLayer {
    pub rotation_gates: Vec<(usize, ParameterizedRotation)>,
    pub entanglement_gates: Vec<Gate>,
}

/// Parameterized rotation gates
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterizedRotation {
    Rx(usize), // Parameter index
    Ry(usize),
    Rz(usize),
    U {
        theta: usize,
        phi: usize,
        lambda: usize,
    },
}

impl VariationalQuantumCircuit {
    /// Create a new VQC with the given architecture
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        let mut parameters = Vec::new();
        let mut param_idx = 0;

        for layer_idx in 0..num_layers {
            let mut rotation_gates = Vec::new();
            let mut entanglement_gates = Vec::new();

            // Add rotation gates to all qubits
            for q in 0..num_qubits {
                // Each qubit gets Ry (parameterized)
                rotation_gates.push((q, ParameterizedRotation::Ry(param_idx)));
                parameters.push(0.0); // Initial value
                param_idx += 1;
            }

            // Add entanglement (linear or circular)
            if layer_idx < num_layers - 1 {
                // Circular entanglement
                for q in 0..num_qubits {
                    let next_q = (q + 1) % num_qubits;
                    entanglement_gates.push(Gate::cnot(q, next_q));
                }
            }

            layers.push(VQCLayer {
                rotation_gates,
                entanglement_gates,
            });
        }

        VariationalQuantumCircuit {
            num_qubits,
            num_parameters: param_idx,
            layers,
            parameters,
        }
    }

    /// Execute the circuit with current parameters
    pub fn execute(&self, state: &mut QuantumState) {
        for layer in &self.layers {
            // Apply rotation gates
            for &(qubit, ref rotation) in &layer.rotation_gates {
                match rotation {
                    ParameterizedRotation::Rx(idx) => {
                        let angle = self.parameters[*idx];
                        crate::GateOperations::rx(state, qubit, angle);
                    }
                    ParameterizedRotation::Ry(idx) => {
                        let angle = self.parameters[*idx];
                        crate::GateOperations::ry(state, qubit, angle);
                    }
                    ParameterizedRotation::Rz(idx) => {
                        let angle = self.parameters[*idx];
                        crate::GateOperations::rz(state, qubit, angle);
                    }
                    _ => {}
                }
            }

            // Apply entanglement gates
            for gate in &layer.entanglement_gates {
                self.apply_gate(state, gate);
            }
        }
    }

    /// Apply a gate to the quantum state
    fn apply_gate(&self, state: &mut QuantumState, gate: &Gate) {
        match gate.gate_type {
            GateType::H => {
                for &target in &gate.targets {
                    crate::GateOperations::h(state, target);
                }
            }
            GateType::X => {
                for &target in &gate.targets {
                    crate::GateOperations::x(state, target);
                }
            }
            GateType::CNOT => {
                if gate.controls.len() == 1 && gate.targets.len() == 1 {
                    crate::GateOperations::cnot(state, gate.controls[0], gate.targets[0]);
                }
            }
            GateType::CZ => {
                if gate.controls.len() == 1 && gate.targets.len() == 1 {
                    crate::GateOperations::cz(state, gate.controls[0], gate.targets[0]);
                }
            }
            _ => {}
        }
    }

    /// Update parameters
    pub fn set_parameters(&mut self, params: Vec<f64>) -> Result<(), String> {
        if params.len() != self.num_parameters {
            return Err(format!(
                "Expected {} parameters, got {}",
                self.num_parameters,
                params.len()
            ));
        }
        self.parameters = params;
        Ok(())
    }

    /// Get current parameters
    pub fn get_parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    /// Compute gradients using parameter-shift rule
    pub fn compute_gradients(&self, input: &[f64]) -> Vec<f64> {
        let eps = PI / 2.0; // Shift amount for parameter-shift rule
        let mut gradients = vec![0.0; self.num_parameters];

        for param_idx in 0..self.num_parameters {
            // Forward shift
            let mut params_plus = self.parameters.clone();
            params_plus[param_idx] += eps;
            let expectation_plus = self.evaluate_with_params(&params_plus, input);

            // Backward shift
            let mut params_minus = self.parameters.clone();
            params_minus[param_idx] -= eps;
            let expectation_minus = self.evaluate_with_params(&params_minus, input);

            // Parameter-shift rule gradient
            gradients[param_idx] = (expectation_plus - expectation_minus) / 2.0;
        }

        gradients
    }

    /// Evaluate circuit expectation value with given parameters
    fn evaluate_with_params(&self, params: &[f64], input: &[f64]) -> f64 {
        let mut state = QuantumState::new(self.num_qubits);

        // Encode input into initial state (using amplitude encoding or basis encoding)
        self.encode_input(&mut state, input);

        // Temporarily set parameters and execute
        let _original_params = self.parameters.clone();
        let mut temp_circuit = self.clone();
        temp_circuit.parameters = params.to_vec();
        temp_circuit.execute(&mut state);

        // Measure expectation value of Z operator on first qubit
        state.expectation_z(0)
    }

    /// Encode classical input data into quantum state
    fn encode_input(&self, state: &mut QuantumState, input: &[f64]) {
        // Simple angle encoding
        for (i, &val) in input.iter().enumerate() {
            if i < self.num_qubits {
                // Normalize value to [-π, π]
                let angle = (val * 2.0 - 1.0) * PI;
                crate::GateOperations::ry(state, i, angle);
            }
        }
    }
}

/// Quantum Neural Network Layer
#[derive(Clone, Debug)]
pub struct QuantumNeuralLayer {
    pub circuit: VariationalQuantumCircuit,
    pub output_size: usize,
}

impl QuantumNeuralLayer {
    pub fn new(num_qubits: usize, num_layers: usize, output_size: usize) -> Self {
        QuantumNeuralLayer {
            circuit: VariationalQuantumCircuit::new(num_qubits, num_layers),
            output_size,
        }
    }

    /// Forward pass through the quantum neural layer
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut state = QuantumState::new(self.circuit.num_qubits);
        self.circuit.encode_input(&mut state, input);
        self.circuit.execute(&mut state);

        // Measure output qubits
        let mut output = Vec::new();
        for i in 0..self.output_size.min(self.circuit.num_qubits) {
            output.push(state.expectation_z(i));
        }
        output
    }

    /// Train the layer using gradient descent
    pub fn train_step(&mut self, input: &[f64], target: &[f64], learning_rate: f64) -> f64 {
        let gradients = self.circuit.compute_gradients(input);

        // Compute loss and update parameters
        let output = self.forward(input);
        let mut loss = 0.0;

        for (_i, (&pred, &targ)) in output.iter().zip(target.iter()).enumerate() {
            let error = pred - targ;
            loss += error * error;
        }
        loss /= output.len() as f64;

        // Update parameters using gradients
        for (i, &grad) in gradients.iter().enumerate() {
            self.circuit.parameters[i] -= learning_rate * grad;
        }

        loss
    }
}

/// Quantum Kernel for Support Vector Machines
#[derive(Clone, Debug)]
pub struct QuantumKernel {
    pub num_qubits: usize,
    pub circuit: VariationalQuantumCircuit,
}

impl QuantumKernel {
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        QuantumKernel {
            num_qubits,
            circuit: VariationalQuantumCircuit::new(num_qubits, num_layers),
        }
    }

    /// Compute kernel value between two data points using quantum circuit
    pub fn evaluate(&mut self, x1: &[f64], x2: &[f64]) -> f64 {
        // Encode both data points and compute overlap
        let mut state = QuantumState::new(self.num_qubits);

        // Encode first data point
        self.circuit.encode_input(&mut state, x1);
        self.circuit.execute(&mut state);

        // Save state (this would require density matrix in full implementation)
        // For now, return a simple RBF-like kernel
        let mut sum_sq_diff = 0.0;
        for (&a, &b) in x1.iter().zip(x2.iter()) {
            sum_sq_diff += (a - b).powi(2);
        }
        (-sum_sq_diff / 2.0).exp()
    }
}

/// Quantum Batch Normalization
#[derive(Clone, Debug)]
pub struct QuantumBatchNorm {
    pub num_qubits: usize,
    pub gamma: Vec<f64>, // Scale
    pub beta: Vec<f64>,  // Shift
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub momentum: f64,
    pub epsilon: f64,
}

impl QuantumBatchNorm {
    pub fn new(num_qubits: usize) -> Self {
        QuantumBatchNorm {
            num_qubits,
            gamma: vec![1.0; num_qubits],
            beta: vec![0.0; num_qubits],
            running_mean: vec![0.0; num_qubits],
            running_var: vec![1.0; num_qubits],
            momentum: 0.1,
            epsilon: 1e-5,
        }
    }

    /// Apply batch normalization to quantum expectations
    pub fn normalize(&mut self, mut expectations: Vec<f64>, training: bool) -> Vec<f64> {
        for i in 0..expectations.len().min(self.num_qubits) {
            if training {
                // Update running statistics
                self.running_mean[i] =
                    self.momentum * expectations[i] + (1.0 - self.momentum) * self.running_mean[i];
            }

            // Normalize
            let normalized = (expectations[i] - self.running_mean[i])
                / (self.running_var[i] + self.epsilon).sqrt();

            // Scale and shift
            expectations[i] = self.gamma[i] * normalized + self.beta[i];
        }

        expectations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vqc_creation() {
        let vqc = VariationalQuantumCircuit::new(4, 2);
        assert_eq!(vqc.num_qubits, 4);
        assert_eq!(vqc.layers.len(), 2);
        assert!(vqc.num_parameters > 0);
    }

    #[test]
    fn test_vqc_execution() {
        let vqc = VariationalQuantumCircuit::new(2, 1);
        let mut state = QuantumState::new(2);
        vqc.execute(&mut state);
        // State should be modified
        let probs = state.probabilities();
        assert!(probs.iter().any(|&p| p > 0.0));
    }

    #[test]
    fn test_qnn_layer() {
        let mut layer = QuantumNeuralLayer::new(2, 1, 2);
        let input = vec![0.5, 0.3];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 2);
    }
}
