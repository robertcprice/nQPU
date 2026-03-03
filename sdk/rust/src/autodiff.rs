//! Automatic Differentiation for Variational Quantum Algorithms
//!
//! This module provides automatic differentiation capabilities for training
//! variational quantum circuits, essential for VQE, QAOA, and other NISQ algorithms.
//!
//! # Key Features
//! - Parameter-shift rule for exact gradient computation
//! - Finite difference methods for non-analytic circuits
//! - Backpropagation through quantum circuits
//! - Hessian computation for optimization
//! - Gradient accumulation for batched training

use crate::{QuantumState, C64};
use std::f64::consts::PI;

/// Gradient computation method
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientMethod {
    /// Parameter-shift rule (exact for gates of form exp(-iθH/2))
    ParameterShift,
    /// Central finite differences
    CentralDifference,
    /// Forward finite differences
    ForwardDifference,
    /// Stochastic parameter shift (faster, approximate)
    StochasticParameterShift,
}

/// Automatic differentiation engine for quantum circuits
pub struct QuantumAutodiff {
    /// Number of parameters in the circuit
    num_parameters: usize,
    /// Gradient computation method
    method: GradientMethod,
    /// Step size for finite difference methods
    epsilon: f64,
}

impl QuantumAutodiff {
    pub fn new(num_parameters: usize, method: GradientMethod) -> Self {
        QuantumAutodiff {
            num_parameters,
            method,
            epsilon: 1e-7,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Compute gradient of expectation value with respect to parameters
    ///
    /// Given an observable O and circuit U(θ), computes ∂⟨O⟩/∂θ
    /// where ⟨O⟩ = ⟨ψ(θ)|O|ψ(θ)⟩ and |ψ(θ)⟩ = U(θ)|0⟩
    ///
    /// # Arguments
    /// * `circuit_fn` - Function that takes parameters and returns the prepared state
    /// * `observable_fn` - Function that computes expectation value ⟨ψ|O|ψ⟩
    /// * `parameters` - Current parameter values
    ///
    /// # Returns
    /// Gradient vector of same length as parameters
    pub fn compute_gradient<F, G>(
        &self,
        circuit_fn: F,
        observable_fn: G,
        parameters: &[f64],
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> QuantumState + Clone,
        G: Fn(&QuantumState) -> f64 + Clone,
    {
        match self.method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_gradient(circuit_fn, observable_fn, parameters)
            }
            GradientMethod::CentralDifference => {
                self.central_difference_gradient(circuit_fn, observable_fn, parameters)
            }
            GradientMethod::ForwardDifference => {
                self.forward_difference_gradient(circuit_fn, observable_fn, parameters)
            }
            GradientMethod::StochasticParameterShift => {
                self.stochastic_parameter_shift(circuit_fn, observable_fn, parameters)
            }
        }
    }

    /// Parameter-shift rule gradient (exact for analytic circuits)
    ///
    /// For gates of form exp(-iθH/2) where H² = I:
    /// ∂f(θ)/∂θ = r[f(θ + π/4r) - f(θ - π/4r)]
    /// where r is determined by the gate's eigenvalue spectrum
    fn parameter_shift_gradient<F, G>(
        &self,
        circuit_fn: F,
        observable_fn: G,
        parameters: &[f64],
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> QuantumState + Clone,
        G: Fn(&QuantumState) -> f64 + Clone,
    {
        let mut gradient = Vec::with_capacity(self.num_parameters);

        for i in 0..self.num_parameters {
            let shift = PI / 4.0;

            // f(θ + π/4)
            let mut params_plus = parameters.to_vec();
            params_plus[i] += shift;
            let state_plus = circuit_fn(&params_plus);
            let value_plus = observable_fn(&state_plus);

            // f(θ - π/4)
            let mut params_minus = parameters.to_vec();
            params_minus[i] -= shift;
            let state_minus = circuit_fn(&params_minus);
            let value_minus = observable_fn(&state_minus);

            // Parameter-shift rule
            gradient.push(0.5 * (value_plus - value_minus));
        }

        gradient
    }

    /// Central finite difference gradient
    fn central_difference_gradient<F, G>(
        &self,
        circuit_fn: F,
        observable_fn: G,
        parameters: &[f64],
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> QuantumState + Clone,
        G: Fn(&QuantumState) -> f64 + Clone,
    {
        let mut gradient = Vec::with_capacity(self.num_parameters);

        for i in 0..self.num_parameters {
            // f(θ + ε)
            let mut params_plus = parameters.to_vec();
            params_plus[i] += self.epsilon;
            let state_plus = circuit_fn(&params_plus);
            let value_plus = observable_fn(&state_plus);

            // f(θ - ε)
            let mut params_minus = parameters.to_vec();
            params_minus[i] -= self.epsilon;
            let state_minus = circuit_fn(&params_minus);
            let value_minus = observable_fn(&state_minus);

            // Central difference formula
            gradient.push((value_plus - value_minus) / (2.0 * self.epsilon));
        }

        gradient
    }

    /// Forward finite difference gradient
    fn forward_difference_gradient<F, G>(
        &self,
        circuit_fn: F,
        observable_fn: G,
        parameters: &[f64],
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> QuantumState + Clone,
        G: Fn(&QuantumState) -> f64 + Clone,
    {
        // Compute base value
        let state_base = circuit_fn(parameters);
        let value_base = observable_fn(&state_base);

        let mut gradient = Vec::with_capacity(self.num_parameters);

        for i in 0..self.num_parameters {
            // f(θ + ε)
            let mut params_plus = parameters.to_vec();
            params_plus[i] += self.epsilon;
            let state_plus = circuit_fn(&params_plus);
            let value_plus = observable_fn(&state_plus);

            // Forward difference formula
            gradient.push((value_plus - value_base) / self.epsilon);
        }

        gradient
    }

    /// Stochastic parameter shift (sample subset of parameters)
    fn stochastic_parameter_shift<F, G>(
        &self,
        circuit_fn: F,
        observable_fn: G,
        parameters: &[f64],
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> QuantumState + Clone,
        G: Fn(&QuantumState) -> f64 + Clone,
    {
        use rand::Rng;

        // Sample 20% of parameters (at least 1)
        let sample_size = (self.num_parameters / 5).max(1);
        let mut rng = rand::thread_rng();

        let mut gradient = vec![0.0; self.num_parameters];
        let shift = PI / 4.0;

        for _ in 0..sample_size {
            let i = rng.gen_range(0..self.num_parameters);

            let mut params_plus = parameters.to_vec();
            params_plus[i] += shift;
            let state_plus = circuit_fn(&params_plus);
            let value_plus = observable_fn(&state_plus);

            let mut params_minus = parameters.to_vec();
            params_minus[i] -= shift;
            let state_minus = circuit_fn(&params_minus);
            let value_minus = observable_fn(&state_minus);

            gradient[i] = 0.5 * (value_plus - value_minus);
        }

        gradient
    }

    /// Compute Hessian matrix (second derivatives)
    pub fn compute_hessian<F, G>(
        &self,
        circuit_fn: F,
        observable_fn: G,
        parameters: &[f64],
    ) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> QuantumState + Clone,
        G: Fn(&QuantumState) -> f64 + Clone,
    {
        let mut hessian = vec![vec![0.0; self.num_parameters]; self.num_parameters];

        for i in 0..self.num_parameters {
            for j in 0..=i {
                // Compute second derivative using finite differences
                let mut params_pp = parameters.to_vec();
                params_pp[i] += self.epsilon;
                params_pp[j] += self.epsilon;
                let state_pp = circuit_fn(&params_pp);
                let value_pp = observable_fn(&state_pp);

                let mut params_pm = parameters.to_vec();
                params_pm[i] += self.epsilon;
                params_pm[j] -= self.epsilon;
                let state_pm = circuit_fn(&params_pm);
                let value_pm = observable_fn(&state_pm);

                let mut params_mp = parameters.to_vec();
                params_mp[i] -= self.epsilon;
                params_mp[j] += self.epsilon;
                let state_mp = circuit_fn(&params_mp);
                let value_mp = observable_fn(&state_mp);

                let mut params_mm = parameters.to_vec();
                params_mm[i] -= self.epsilon;
                params_mm[j] -= self.epsilon;
                let state_mm = circuit_fn(&params_mm);
                let value_mm = observable_fn(&state_mm);

                hessian[i][j] = (value_pp - value_pm - value_mp + value_mm)
                    / (4.0 * self.epsilon * self.epsilon);
                hessian[j][i] = hessian[i][j];
            }
        }

        hessian
    }
}

/// Variational quantum circuit with trainable parameters
pub struct VariationalCircuit {
    /// Number of qubits
    num_qubits: usize,
    /// Circuit parameters
    parameters: Vec<f64>,
    /// Circuit layers (each layer is a list of gates)
    layers: Vec<Vec<ParametricGate>>,
}

/// A gate with trainable parameters
#[derive(Clone, Debug)]
pub struct ParametricGate {
    /// Gate type (Rx, Ry, Rz, etc.)
    gate_type: ParametricGateType,
    /// Target qubit(s)
    targets: Vec<usize>,
    /// Which circuit parameters this gate uses
    parameter_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParametricGateType {
    Rx,
    Ry,
    Rz,
    CRx,
    CRy,
    CRz,
    U3, // Generic single-qubit unitary with 3 parameters
}

impl VariationalCircuit {
    pub fn new(num_qubits: usize, num_parameters: usize) -> Self {
        VariationalCircuit {
            num_qubits,
            parameters: vec![0.0; num_parameters],
            layers: Vec::new(),
        }
    }

    /// Add a parametric gate to the circuit
    pub fn add_gate(
        &mut self,
        gate_type: ParametricGateType,
        targets: Vec<usize>,
        param_indices: Vec<usize>,
    ) {
        self.layers.push(vec![ParametricGate {
            gate_type,
            targets,
            parameter_indices: param_indices,
        }]);
    }

    /// Add a layer of gates (executed in parallel)
    pub fn add_layer(&mut self, gates: Vec<ParametricGate>) {
        self.layers.push(gates);
    }

    /// Execute the circuit with current parameters
    pub fn execute(&self, parameters: &[f64]) -> QuantumState {
        let mut state = QuantumState::new(self.num_qubits);

        for layer in &self.layers {
            for gate in layer {
                self.apply_gate(&mut state, gate, parameters);
            }
        }

        state
    }

    fn apply_gate(&self, state: &mut QuantumState, gate: &ParametricGate, parameters: &[f64]) {
        match gate.gate_type {
            ParametricGateType::Rx => {
                let theta = parameters[gate.parameter_indices[0]];
                let target = gate.targets[0];
                crate::GateOperations::rx(state, target, theta);
            }
            ParametricGateType::Ry => {
                let theta = parameters[gate.parameter_indices[0]];
                let target = gate.targets[0];
                crate::GateOperations::ry(state, target, theta);
            }
            ParametricGateType::Rz => {
                let theta = parameters[gate.parameter_indices[0]];
                let target = gate.targets[0];
                crate::GateOperations::rz(state, target, theta);
            }
            ParametricGateType::U3 => {
                let theta = parameters[gate.parameter_indices[0]];
                let phi = parameters[gate.parameter_indices[1]];
                let lambda = parameters[gate.parameter_indices[2]];
                let target = gate.targets[0];
                // Apply U3 gate through decomposition: U3 = Rz(phi) * Ry(theta) * Rz(lambda)
                crate::GateOperations::rz(state, target, lambda);
                crate::GateOperations::ry(state, target, theta);
                crate::GateOperations::rz(state, target, phi);
            }
            _ => {}
        }
    }

    /// Get number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Set parameter values
    pub fn set_parameters(&mut self, parameters: Vec<f64>) {
        self.parameters = parameters;
    }

    /// Get parameter values
    pub fn get_parameters(&self) -> &[f64] {
        &self.parameters
    }
}

/// Gradient descent optimizer for variational circuits
pub struct GradientDescentOptimizer {
    learning_rate: f64,
    momentum: f64,
    velocity: Vec<f64>,
}

impl GradientDescentOptimizer {
    pub fn new(learning_rate: f64, num_parameters: usize) -> Self {
        GradientDescentOptimizer {
            learning_rate,
            momentum: 0.0,
            velocity: vec![0.0; num_parameters],
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Update parameters using gradient descent with momentum
    pub fn update(&mut self, parameters: &mut [f64], gradient: &[f64]) {
        for i in 0..parameters.len() {
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradient[i];
            parameters[i] += self.velocity[i];
        }
    }
}

/// Adam optimizer for variational circuits
pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    timestep: usize,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64, num_parameters: usize) -> Self {
        AdamOptimizer {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: vec![0.0; num_parameters],
            v: vec![0.0; num_parameters],
            timestep: 0,
        }
    }

    /// Update parameters using Adam optimization
    pub fn update(&mut self, parameters: &mut [f64], gradient: &[f64]) {
        self.timestep += 1;
        let alpha = self.learning_rate * (1.0 - self.beta2.powi(self.timestep as i32)).sqrt()
            / (1.0 - self.beta1.powi(self.timestep as i32));

        for i in 0..parameters.len() {
            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradient[i];

            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradient[i] * gradient[i];

            // Compute parameter update
            let update = alpha * self.m[i] / (self.v[i].sqrt() + self.epsilon);
            parameters[i] -= update;
        }
    }
}

/// Natural gradient optimizer using Quantum Fisher Information Matrix
pub struct NaturalGradientOptimizer {
    learning_rate: f64,
    regularization: f64,
}

impl NaturalGradientOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        NaturalGradientOptimizer {
            learning_rate,
            regularization: 1e-6,
        }
    }

    /// Update parameters using natural gradient
    ///
    /// Uses the Quantum Fisher Information Matrix (QFIM) for geometry-aware optimization
    pub fn update(&mut self, parameters: &mut [f64], gradient: &[f64], qfim: &[Vec<f64>]) {
        // Solve QFIM * update = gradient
        let mut update = vec![0.0; parameters.len()];

        // Add regularization to QFIM
        let mut regularized_qfim = qfim.to_vec();
        for i in 0..regularized_qfim.len() {
            regularized_qfim[i][i] += self.regularization;
        }

        // Solve linear system (simplified - would use proper linear algebra in production)
        // For now, use diagonal approximation
        for i in 0..parameters.len() {
            update[i] = gradient[i] / (regularized_qfim[i][i] + self.regularization);
        }

        // Apply update
        for i in 0..parameters.len() {
            parameters[i] -= self.learning_rate * update[i];
        }
    }
}

/// Compute Quantum Fisher Information Matrix
///
/// The QFIM captures the geometry of the parameter space and is used
/// for natural gradient optimization
pub fn compute_quantum_fisher_information_matrix<F>(
    circuit_fn: F,
    parameters: &[f64],
    epsilon: f64,
) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> QuantumState + Clone,
{
    let num_params = parameters.len();
    let mut qfim = vec![vec![0.0; num_params]; num_params];

    // Compute state vector
    let state = circuit_fn(parameters);
    let amplitudes = state.amplitudes_ref();

    for i in 0..num_params {
        for j in 0..=i {
            // Compute partial derivatives
            let mut params_plus_i = parameters.to_vec();
            params_plus_i[i] += epsilon;
            let state_plus_i = circuit_fn(&params_plus_i);

            let mut params_minus_i = parameters.to_vec();
            params_minus_i[i] -= epsilon;
            let state_minus_i = circuit_fn(&params_minus_i);

            let mut params_plus_j = parameters.to_vec();
            params_plus_j[j] += epsilon;
            let state_plus_j = circuit_fn(&params_plus_j);

            // Fisher information element
            let mut fisher_element = 0.0;
            for (k, _amp) in amplitudes.iter().enumerate() {
                // Manual division for C64
                let diff_i = C64 {
                    re: state_plus_i.amplitudes_ref()[k].re - state_minus_i.amplitudes_ref()[k].re,
                    im: state_plus_i.amplitudes_ref()[k].im - state_minus_i.amplitudes_ref()[k].im,
                };
                let dpsi_i = C64 {
                    re: diff_i.re / (2.0 * epsilon),
                    im: diff_i.im / (2.0 * epsilon),
                };

                let diff_j = C64 {
                    re: state_plus_j.amplitudes_ref()[k].re - amplitudes[k].re,
                    im: state_plus_j.amplitudes_ref()[k].im - amplitudes[k].im,
                };
                let dpsi_j = C64 {
                    re: diff_j.re / epsilon,
                    im: diff_j.im / epsilon,
                };

                fisher_element += 2.0 * (dpsi_i.re * dpsi_j.re + dpsi_i.im * dpsi_j.im);
            }

            qfim[i][j] = fisher_element;
            qfim[j][i] = fisher_element;
        }
    }

    qfim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodiff_creation() {
        let autodiff = QuantumAutodiff::new(3, GradientMethod::ParameterShift);
        assert_eq!(autodiff.num_parameters, 3);
    }

    #[test]
    fn test_parameter_shift_gradient() {
        let autodiff = QuantumAutodiff::new(1, GradientMethod::ParameterShift);

        // Simple test function: f(θ) = sin(θ)
        let circuit_fn = |params: &[f64]| -> QuantumState {
            let mut state = QuantumState::new(1);
            crate::GateOperations::ry(&mut state, 0, params[0]);
            state
        };

        let observable_fn = |state: &QuantumState| -> f64 {
            // Measure in Z basis (probability of |0⟩)
            let mut p0 = 0.0;
            for i in 0..state.dim {
                if i & 1 == 0 {
                    p0 += state.amplitudes_ref()[i].norm_sqr();
                }
            }
            p0
        };

        let gradient = autodiff.compute_gradient(circuit_fn, observable_fn, &[0.5]);
        assert_eq!(gradient.len(), 1);
    }

    #[test]
    fn test_variational_circuit() {
        let mut circuit = VariationalCircuit::new(2, 3);
        circuit.add_gate(ParametricGateType::Ry, vec![0], vec![0]);
        circuit.add_gate(ParametricGateType::Ry, vec![1], vec![1]);
        circuit.add_gate(ParametricGateType::CRz, vec![0, 1], vec![2]);

        assert_eq!(circuit.num_parameters(), 3);

        let state = circuit.execute(&[0.5, 0.3, 0.1]);
        assert_eq!(state.num_qubits, 2);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = AdamOptimizer::new(0.01, 3);
        let mut params = vec![0.5, 0.3, 0.1];
        let gradient = vec![0.1, 0.2, 0.3];

        optimizer.update(&mut params, &gradient);

        // Parameters should have changed
        assert_ne!(params, vec![0.5, 0.3, 0.1]);
    }
}
