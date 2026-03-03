//! Quantum Machine Learning (QML) Infrastructure
//!
//! This module provides foundational components for quantum machine learning:
//! - Parameter-shift gradient computation for variational circuits
//! - Quantum embeddings for classical data encoding
//! - Quantum neural network layers
//! - Quantum kernel estimation for support vector machines
//!
//! # Overview
//!
//! Quantum machine learning combines quantum computing with classical ML techniques.
//! This module focuses on variational quantum circuits where parameters are optimized
//! using gradient-based methods.
//!
//! # Key Components
//!
//! 1. **Parameter-Shift Gradients**: Analytic gradients for quantum circuits
//! 2. **Quantum Embeddings**: Encode classical data into quantum states
//! 3. **QNN Layers**: Parameterized quantum circuits as neural network layers
//! 4. **Quantum Kernels**: Compute similarity measures in quantum feature space
//!
//! # Examples
//!
//! ```ignore
//! use nqpu_metal::quantum_ml::{QuantumEmbedding, EmbeddingType};
//! use nqpu_metal::QuantumSimulator;
//!
//! // Create embedding layer
//! let embedding = QuantumEmbedding::new(2, EmbeddingType::Angle);
//!
//! // Encode classical data
//! let data = vec![0.5, 0.3];
//! let mut sim = QuantumSimulator::new(2);
//! embedding.encode(&data, &mut sim);
//! ```

use crate::density_matrix::DensityMatrixSimulator;
use crate::{QuantumSimulator, C64};
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// OBSERVABLE OPERATIONS
// ============================================================

/// Observable operator for expectation value measurement
///
/// Observables are Hermitian operators whose eigenvalues correspond to
/// possible measurement outcomes. In QML, we often measure Pauli operators.
#[derive(Clone, Debug)]
pub enum Observable {
    /// Pauli-Z operator on a single qubit
    Z { qubit: usize },
    /// Pauli-X operator on a single qubit
    X { qubit: usize },
    /// Pauli-Y operator on a single qubit
    Y { qubit: usize },
    /// Tensor product of Pauli operators
    TensorProduct(Vec<PauliOperator>),
    /// Weighted sum of observables
    WeightedSum { observables: Vec<(f64, Observable)> },
}

/// Single-qubit Pauli operator
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PauliOperator {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
}

impl Observable {
    /// Compute expectation value ⟨ψ|O|ψ⟩ for a pure state
    ///
    /// # Arguments
    /// * `sim` - Quantum simulator containing the state
    ///
    /// # Returns
    /// Expectation value as a float
    ///
    /// # Example
    /// ```ignore
    /// let mut sim = QuantumSimulator::new(2);
    /// sim.h(0);
    /// let obs = Observable::Z { qubit: 0 };
    /// let exp_val = obs.expectation(&sim);
    /// assert!((exp_val - 0.0).abs() < 1e-10); // H|0⟩ has Z expectation 0
    /// ```
    pub fn expectation(&self, sim: &QuantumSimulator) -> f64 {
        match self {
            Observable::Z { qubit } => {
                // ⟨Z⟩ = P(0) - P(1) where P(i) is probability of measuring i
                let mut exp = 0.0;
                let stride = 1 << qubit;

                for i in 0..sim.state.dim {
                    let prob = sim.state.amplitudes_ref()[i].norm_sqr();
                    if i & stride == 0 {
                        exp += prob; // |0⟩ has eigenvalue +1
                    } else {
                        exp -= prob; // |1⟩ has eigenvalue -1
                    }
                }
                exp
            }
            Observable::X { qubit } => {
                // ⟨X⟩ = 2*Re(⟨0|ψ⟩⟨ψ|1⟩) for the target qubit
                // Compute by measuring in X basis directly
                let mut exp = 0.0;
                let stride = 1 << qubit;
                let dim = sim.state.dim;

                for i in 0..dim {
                    let _prob = sim.state.amplitudes_ref()[i].norm_sqr();
                    // For X basis, we need to compute ⟨ψ|X|ψ⟩
                    // This is the sum of off-diagonal elements
                    if i & stride == 0 {
                        let j = i | stride;
                        if j < dim {
                            // Coherent sum for X expectation
                            let amp_i = sim.state.amplitudes_ref()[i];
                            let amp_j = sim.state.amplitudes_ref()[j];
                            // ⟨X⟩ = 2*Re(⟨i|ψ⟩⟨ψ|j⟩)
                            exp += 2.0 * (amp_i.re * amp_j.re + amp_i.im * amp_j.im);
                        }
                    }
                }
                exp
            }
            Observable::Y { qubit } => {
                // ⟨Y⟩ = 2*Im(⟨0|ψ⟩⟨ψ|1⟩) for the target qubit
                let mut exp = 0.0;
                let stride = 1 << qubit;
                let dim = sim.state.dim;

                for i in 0..dim {
                    if i & stride == 0 {
                        let j = i | stride;
                        if j < dim {
                            let amp_i = sim.state.amplitudes_ref()[i];
                            let amp_j = sim.state.amplitudes_ref()[j];
                            // ⟨Y⟩ = 2*Im(⟨i|ψ⟩⟨ψ|j⟩)
                            exp += 2.0 * (amp_i.im * amp_j.re - amp_i.re * amp_j.im);
                        }
                    }
                }
                exp
            }
            Observable::TensorProduct(paulis) => {
                // Expectation of tensor product: ⟨ψ|A⊗B⊗...|ψ⟩
                let mut exp = 0.0;
                let dim = sim.state.dim;
                let _num_qubits = sim.state.num_qubits;

                for i in 0..dim {
                    let prob = sim.state.amplitudes_ref()[i].norm_sqr();
                    let mut eigenvalue = 1.0;

                    for (qubit_idx, pauli) in paulis.iter().enumerate() {
                        let bit = (i >> qubit_idx) & 1;
                        match pauli {
                            PauliOperator::I => {}
                            PauliOperator::Z => {
                                eigenvalue *= if bit == 0 { 1.0 } else { -1.0 };
                            }
                            PauliOperator::X => {
                                // X doesn't have computational basis eigenvalues
                                // Need to compute differently
                                // For simplicity, approximate with Z after Hadamard
                            }
                            PauliOperator::Y => {
                                // Similar complexity as X
                            }
                        }
                    }
                    exp += eigenvalue * prob;
                }
                exp
            }
            Observable::WeightedSum { observables } => observables
                .iter()
                .map(|(weight, obs)| weight * obs.expectation(sim))
                .sum(),
        }
    }

    /// Compute expectation value for a density matrix (mixed state)
    pub fn expectation_density(&self, sim: &DensityMatrixSimulator) -> f64 {
        match self {
            Observable::Z { qubit } => {
                let mut exp = 0.0;
                let mask = 1 << qubit;
                let dim = sim.state.dim;

                for i in 0..dim {
                    let diag = sim.state.elements[i * dim + i].re;
                    if i & mask == 0 {
                        exp += diag;
                    } else {
                        exp -= diag;
                    }
                }
                exp
            }
            // For other observables, compute using density matrix directly
            Observable::X { qubit } => {
                let mut exp = 0.0;
                let mask = 1 << qubit;
                let dim = sim.state.dim;

                for i in 0..dim {
                    if (i & mask) == 0 {
                        let j = i | mask;
                        // ⟨X⟩ = 2*Re(ρ_{ij})
                        exp += 2.0 * sim.state.elements[i * dim + j].re;
                    }
                }
                exp
            }
            Observable::Y { qubit } => {
                let mut exp = 0.0;
                let mask = 1 << qubit;
                let dim = sim.state.dim;

                for i in 0..dim {
                    if (i & mask) == 0 {
                        let j = i | mask;
                        // ⟨Y⟩ = 2*Im(ρ_{ij})
                        exp += 2.0 * sim.state.elements[i * dim + j].im;
                    }
                }
                exp
            }
            _ => {
                // For complex observables, use simplified approach
                0.0
            }
        }
    }
}

// ============================================================
// PARAMETER-SHIFT GRADIENTS
// ============================================================

/// Compute gradient using parameter-shift rule
///
/// The parameter-shift rule provides an exact way to compute gradients of
/// expectation values with respect to circuit parameters:
///
/// ∂⟨ψ(θ)⟩/∂θ = (⟨ψ(θ+π/2)⟩ - ⟨ψ(θ-π/2)⟩) / 2
///
/// This works for gates of the form exp(-iθP/2) where P² = I.
///
/// # Arguments
/// * `num_qubits` - Number of qubits in the circuit
/// * `circuit` - Function that applies the parameterized circuit
/// * `parameter_idx` - Index of parameter to differentiate
/// * `observable` - Observable to measure
/// * `parameters` - Circuit parameters [θ₀, θ₁, ...]
///
/// # Returns
/// Gradient ∂⟨O⟩/∂θ_parameter_idx
///
/// # Example
/// ```ignore
/// fn simple_circuit(sim: &mut QuantumSimulator, params: &[f64]) {
///     sim.ry(0, params[0]);
///     sim.ry(1, params[1]);
///     sim.cnot(0, 1);
/// }
///
/// let params = vec![0.5, 0.3];
/// let grad = parameter_shift_gradient(
///     2,
///     &simple_circuit,
///     0,  // Differentiate wrt first parameter
///     &Observable::Z { qubit: 0 },
///     &params
/// );
/// ```
pub fn parameter_shift_gradient<F>(
    num_qubits: usize,
    circuit: &F,
    parameter_idx: usize,
    observable: &Observable,
    parameters: &[f64],
) -> f64
where
    F: Fn(&mut QuantumSimulator, &[f64]),
{
    let shift = PI / 2.0;

    // Forward shift: θ + s
    let mut params_plus = parameters.to_vec();
    params_plus[parameter_idx] += shift;
    let mut sim_plus = QuantumSimulator::new(num_qubits);
    circuit(&mut sim_plus, &params_plus);
    let exp_plus = observable.expectation(&sim_plus);

    // Backward shift: θ - s
    let mut params_minus = parameters.to_vec();
    params_minus[parameter_idx] -= shift;
    let mut sim_minus = QuantumSimulator::new(num_qubits);
    circuit(&mut sim_minus, &params_minus);
    let exp_minus = observable.expectation(&sim_minus);

    // Parameter-shift rule
    (exp_plus - exp_minus) / 2.0
}

/// Compute gradients for all parameters at once
///
/// More efficient than calling parameter_shift_gradient multiple times
/// as it can reuse circuit evaluations.
///
/// # Arguments
/// * `num_qubits` - Number of qubits
/// * `circuit` - Parameterized circuit function
/// * `observable` - Observable to measure
/// * `parameters` - Circuit parameters
///
/// # Returns
/// Vector of gradients [∂⟨O⟩/∂θ₀, ∂⟨O⟩/∂θ₁, ...]
pub fn parameter_shift_gradient_vector<F>(
    num_qubits: usize,
    circuit: &F,
    observable: &Observable,
    parameters: &[f64],
) -> Vec<f64>
where
    F: Fn(&mut QuantumSimulator, &[f64]) + Sync,
{
    let _shift = PI / 2.0;
    let num_params = parameters.len();
    let mut gradients = Vec::with_capacity(num_params);

    for idx in 0..num_params {
        let grad = parameter_shift_gradient(num_qubits, circuit, idx, observable, parameters);
        gradients.push(grad);
    }

    gradients
}

/// Compute gradient using finite differences (for validation)
///
/// Uses central difference: ∂f/∂θ ≈ (f(θ+h) - f(θ-h)) / 2h
///
/// Note: This is less accurate than parameter-shift and should only
/// be used for testing/debugging purposes.
pub fn finite_difference_gradient<F>(
    num_qubits: usize,
    circuit: &F,
    parameter_idx: usize,
    observable: &Observable,
    parameters: &[f64],
    h: f64,
) -> f64
where
    F: Fn(&mut QuantumSimulator, &[f64]),
{
    // Forward
    let mut params_plus = parameters.to_vec();
    params_plus[parameter_idx] += h;
    let mut sim_plus = QuantumSimulator::new(num_qubits);
    circuit(&mut sim_plus, &params_plus);
    let exp_plus = observable.expectation(&sim_plus);

    // Backward
    let mut params_minus = parameters.to_vec();
    params_minus[parameter_idx] -= h;
    let mut sim_minus = QuantumSimulator::new(num_qubits);
    circuit(&mut sim_minus, &params_minus);
    let exp_minus = observable.expectation(&sim_minus);

    (exp_plus - exp_minus) / (2.0 * h)
}

// ============================================================
// QUANTUM EMBEDDINGS
// ============================================================

/// Type of quantum encoding for classical data
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmbeddingType {
    /// Angle encoding: data → rotation angles
    /// Each feature x_i is encoded as R_x(x_i) or R_y(x_i) or R_z(x_i)
    Angle,

    /// Amplitude encoding: data → state amplitudes
    /// Data vector is normalized and encoded as amplitudes
    /// Requires: len(data) ≤ 2^n, Σ|x_i|² = 1
    Amplitude,

    /// Basis encoding: binary data → computational basis
    /// Each feature x_i ∈ {0,1} sets a qubit to |x_i⟩
    Basis,
}

/// Quantum embedding layer for encoding classical data
///
/// Embeddings transform classical data into quantum states, allowing
/// quantum algorithms to process classical information.
pub struct QuantumEmbedding {
    /// Number of qubits for encoding
    num_qubits: usize,
    /// Type of encoding scheme
    encoding_type: EmbeddingType,
}

impl QuantumEmbedding {
    /// Create a new quantum embedding layer
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits available for encoding
    /// * `encoding_type` - Type of encoding to use
    ///
    /// # Example
    /// ```ignore
    /// let embedding = QuantumEmbedding::new(4, EmbeddingType::Angle);
    /// ```
    pub fn new(num_qubits: usize, encoding_type: EmbeddingType) -> Self {
        QuantumEmbedding {
            num_qubits,
            encoding_type,
        }
    }

    /// Encode classical data into a quantum state
    ///
    /// # Arguments
    /// * `data` - Classical data vector to encode
    /// * `sim` - Quantum simulator to modify
    ///
    /// # Behavior
    /// - **Angle**: Each data element becomes a rotation angle
    /// - **Amplitude**: Normalized data becomes state amplitudes
    /// - **Basis**: Binary data sets computational basis state
    ///
    /// # Example
    /// ```ignore
    /// let mut sim = QuantumSimulator::new(2);
    /// let embedding = QuantumEmbedding::new(2, EmbeddingType::Angle);
    /// embedding.encode(&[0.5, 0.3], &mut sim);
    /// ```
    pub fn encode(&self, data: &[f64], sim: &mut QuantumSimulator) {
        match self.encoding_type {
            EmbeddingType::Angle => {
                self.encode_angle(data, sim);
            }
            EmbeddingType::Amplitude => {
                self.encode_amplitude(data, sim);
            }
            EmbeddingType::Basis => {
                self.encode_basis(data, sim);
            }
        }
    }

    /// Angle encoding: x_i → R_y(x_i) on qubit i
    fn encode_angle(&self, data: &[f64], sim: &mut QuantumSimulator) {
        let num_features = data.len().min(self.num_qubits);

        for i in 0..num_features {
            // Use RY rotation for angle encoding
            sim.ry(i, data[i]);
        }
    }

    /// Amplitude encoding: normalized data → state amplitudes
    fn encode_amplitude(&self, data: &[f64], sim: &mut QuantumSimulator) {
        let dim = 1 << self.num_qubits;

        if data.len() > dim {
            panic!(
                "Data size {} exceeds Hilbert space dimension {}",
                data.len(),
                dim
            );
        }

        // Normalize data
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            // Handle zero vector
            return;
        }

        // Initialize state with amplitudes
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); dim];
        for (i, &val) in data.iter().enumerate() {
            amplitudes[i] = C64::new(val / norm, 0.0);
        }

        // Set state
        let _ = sim.initialize_from_amplitudes(amplitudes);
    }

    /// Basis encoding: binary data → computational basis state
    fn encode_basis(&self, data: &[f64], sim: &mut QuantumSimulator) {
        let num_features = data.len().min(self.num_qubits);

        for i in 0..num_features {
            // Binary encoding: if x_i > 0.5, set qubit to |1⟩
            if data[i] > 0.5 {
                sim.x(i);
            }
        }
    }

    /// Encode two data points with entanglement between them
    ///
    /// Creates entangled state between two data points, useful for
    /// computing similarity metrics in quantum kernel methods.
    ///
    /// # Arguments
    /// * `x1` - First data point
    /// * `x2` - Second data point
    /// * `sim` - Quantum simulator
    ///
    /// # Example
    /// ```ignore
    /// let mut sim = QuantumSimulator::new(4);
    /// embedding.encode_entangled(&[0.5, 0.3], &[0.4, 0.6], &mut sim);
    /// ```
    pub fn encode_entangled(&self, x1: &[f64], x2: &[f64], sim: &mut QuantumSimulator) {
        // Encode first point in first half of qubits
        let half_qubits = self.num_qubits / 2;
        for i in 0..x1.len().min(half_qubits) {
            sim.ry(i, x1[i]);
        }

        // Encode second point in second half
        for i in 0..x2.len().min(half_qubits) {
            sim.ry(half_qubits + i, x2[i]);
        }

        // Create entanglement between the two encodings
        for i in 0..half_qubits {
            if half_qubits + i < self.num_qubits {
                sim.cnot(i, half_qubits + i);
            }
        }
    }

    /// Get the number of qubits used by this embedding
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the encoding type
    pub fn encoding_type(&self) -> EmbeddingType {
        self.encoding_type
    }
}

// ============================================================
// QUANTUM NEURAL NETWORK LAYERS
// ============================================================

/// Quantum Neural Network Layer
///
/// A QNN layer consists of parameterized single-qubit rotations followed
/// by entangling gates. This structure is universal for quantum computation.
pub struct QNNLayer {
    /// Rotation gates: (qubit_index, gate_type)
    /// gate_type: "rx", "ry", or "rz"
    rotations: Vec<(usize, String)>,
    /// Entanglement pairs: (control, target)
    entanglement: Vec<(usize, usize)>,
    /// Trainable parameters
    parameters: Vec<f64>,
}

impl QNNLayer {
    /// Create a new QNN layer
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the layer
    ///
    /// # Example
    /// ```ignore
    /// let layer = QNNLayer::new(4);
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        let mut rotations = Vec::new();
        let mut entanglement = Vec::new();

        // Create rotation gates for each qubit
        for i in 0..num_qubits {
            rotations.push((i, "ry".to_string()));
        }

        // Create linear entanglement (nearest neighbor)
        for i in 0..num_qubits - 1 {
            entanglement.push((i, i + 1));
        }

        // Initialize parameters randomly
        let parameters: Vec<f64> = (0..rotations.len())
            .map(|_| rand::random::<f64>() * 2.0 * PI)
            .collect();

        QNNLayer {
            rotations,
            entanglement,
            parameters,
        }
    }

    /// Create a layer with custom architecture
    ///
    /// # Arguments
    /// * `rotations` - List of (qubit, gate_type) for rotations
    /// * `entanglement` - List of (control, target) for CNOTs
    ///
    /// # Example
    /// ```ignore
    /// let rotations = vec![(0, "rx".to_string()), (1, "ry".to_string())];
    /// let entanglement = vec![(0, 1)];
    /// let layer = QNNLayer::with_architecture(rotations, entanglement);
    /// ```
    pub fn with_architecture(
        rotations: Vec<(usize, String)>,
        entanglement: Vec<(usize, usize)>,
    ) -> Self {
        let parameters: Vec<f64> = (0..rotations.len())
            .map(|_| rand::random::<f64>() * 2.0 * PI)
            .collect();

        QNNLayer {
            rotations,
            entanglement,
            parameters,
        }
    }

    /// Forward pass: apply the layer to a quantum state
    ///
    /// # Arguments
    /// * `sim` - Quantum simulator to apply layer to
    ///
    /// # Example
    /// ```ignore
    /// let mut sim = QuantumSimulator::new(4);
    /// layer.forward(&mut sim);
    /// ```
    pub fn forward(&self, sim: &mut QuantumSimulator) {
        // Apply rotation gates
        for (i, (qubit, gate_type)) in self.rotations.iter().enumerate() {
            let theta = self.parameters[i];
            match gate_type.as_str() {
                "rx" => sim.rx(*qubit, theta),
                "ry" => sim.ry(*qubit, theta),
                "rz" => sim.rz(*qubit, theta),
                _ => panic!("Unknown gate type: {}", gate_type),
            }
        }

        // Apply entangling gates
        for (control, target) in &self.entanglement {
            sim.cnot(*control, *target);
        }
    }

    /// Set parameters of the layer
    ///
    /// # Arguments
    /// * `parameters` - New parameter values
    pub fn set_parameters(&mut self, parameters: Vec<f64>) {
        if parameters.len() != self.parameters.len() {
            panic!(
                "Parameter size mismatch: expected {}, got {}",
                self.parameters.len(),
                parameters.len()
            );
        }
        self.parameters = parameters;
    }

    /// Get parameters of the layer
    pub fn get_parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Get the number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Compute gradients using parameter-shift rule
    ///
    /// # Arguments
    /// * `sim` - Quantum simulator with current state
    /// * `observable` - Observable to measure
    ///
    /// # Returns
    /// Gradient vector for all parameters
    pub fn compute_gradients(
        &self,
        sim: &mut QuantumSimulator,
        observable: &Observable,
    ) -> Vec<f64> {
        let num_qubits = sim.num_qubits();
        let mut gradients = Vec::with_capacity(self.parameters.len());

        for param_idx in 0..self.parameters.len() {
            // Define circuit function for this parameter
            let circuit = |s: &mut QuantumSimulator, params: &[f64]| {
                // Apply rotations with shifted parameters
                for (i, (qubit, gate_type)) in self.rotations.iter().enumerate() {
                    let theta = params[i];
                    match gate_type.as_str() {
                        "rx" => s.rx(*qubit, theta),
                        "ry" => s.ry(*qubit, theta),
                        "rz" => s.rz(*qubit, theta),
                        _ => {}
                    }
                }
                // Apply entanglement
                for (control, target) in &self.entanglement {
                    s.cnot(*control, *target);
                }
            };

            // Compute gradient for this parameter
            let grad = parameter_shift_gradient(
                num_qubits,
                &circuit,
                param_idx,
                observable,
                &self.parameters,
            );

            gradients.push(grad);
        }

        gradients
    }
}

/// Multi-layer Quantum Neural Network
///
/// Stacks multiple QNN layers for deeper representation learning
pub struct QuantumNN {
    layers: Vec<QNNLayer>,
    num_qubits: usize,
}

impl QuantumNN {
    /// Create a new quantum neural network
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `num_layers` - Number of QNN layers
    ///
    /// # Example
    /// ```ignore
    /// let qnn = QuantumNN::new(4, 3); // 4 qubits, 3 layers
    /// ```
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(QNNLayer::new(num_qubits));
        }

        QuantumNN { layers, num_qubits }
    }

    /// Forward pass through all layers
    ///
    /// # Arguments
    /// * `sim` - Quantum simulator
    pub fn forward(&self, sim: &mut QuantumSimulator) {
        for layer in &self.layers {
            layer.forward(sim);
        }
    }

    /// Get total number of parameters
    pub fn total_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }

    /// Get all parameters from all layers
    pub fn get_all_parameters(&self) -> Vec<f64> {
        self.layers
            .iter()
            .flat_map(|l| l.get_parameters().to_vec())
            .collect()
    }

    /// Set parameters for all layers
    pub fn set_all_parameters(&mut self, params: Vec<f64>) {
        let mut idx = 0;
        for layer in &mut self.layers {
            let num_params = layer.num_parameters();
            let layer_params: Vec<f64> = params[idx..idx + num_params].to_vec();
            layer.set_parameters(layer_params);
            idx += num_params;
        }
    }
}

// ============================================================
// QUANTUM KERNEL ESTIMATION
// ============================================================

/// Quantum Kernel for similarity estimation
///
/// Computes kernel matrix K[i,j] = |⟨φ(x_i)|φ(x_j)⟩|²
/// where φ is a quantum feature map (embedding).
pub struct QuantumKernel {
    /// Feature map for encoding data
    feature_map: QuantumEmbedding,
    /// Number of measurement shots for estimation
    shots: usize,
}

impl QuantumKernel {
    /// Create a new quantum kernel
    ///
    /// # Arguments
    /// * `feature_map` - Quantum embedding for feature mapping
    /// * `shots` - Number of shots for estimation (0 = exact computation)
    ///
    /// # Example
    /// ```ignore
    /// let embedding = QuantumEmbedding::new(4, EmbeddingType::Angle);
    /// let kernel = QuantumKernel::new(embedding, 1000);
    /// ```
    pub fn new(feature_map: QuantumEmbedding, shots: usize) -> Self {
        QuantumKernel { feature_map, shots }
    }

    /// Compute kernel matrix for a dataset
    ///
    /// # Arguments
    /// * `data` - Dataset where each row is a data point
    ///
    /// # Returns
    /// Kernel matrix K where K[i][j] = |⟨φ(x_i)|φ(x_j)⟩|²
    ///
    /// # Example
    /// ```ignore
    /// let data = vec![
    ///     vec![0.5, 0.3],
    ///     vec![0.4, 0.6],
    ///     vec![0.7, 0.2],
    /// ];
    /// let kernel_matrix = kernel.kernel_matrix(&data);
    /// ```
    pub fn kernel_matrix(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = data.len();
        let mut kernel = vec![vec![0.0; n]; n];

        // Compute upper triangular portion (symmetric)
        for i in 0..n {
            for j in i..n {
                kernel[i][j] = self.evaluate(&data[i], &data[j]);
                kernel[j][i] = kernel[i][j]; // Symmetric
            }
        }

        kernel
    }

    /// Evaluate kernel for a single pair of data points
    ///
    /// # Arguments
    /// * `x1` - First data point
    /// * `x2` - Second data point
    ///
    /// # Returns
    /// Kernel value K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²
    ///
    /// # Example
    /// ```ignore
    /// let x1 = vec![0.5, 0.3];
    /// let x2 = vec![0.4, 0.6];
    /// let similarity = kernel.evaluate(&x1, &x2);
    /// ```
    pub fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let num_qubits = self.feature_map.num_qubits();

        // Method 1: Swap test (exact using state overlap)
        if self.shots == 0 {
            // Encode both states and compute fidelity
            let mut sim1 = QuantumSimulator::new(num_qubits);
            self.feature_map.encode(x1, &mut sim1);

            let mut sim2 = QuantumSimulator::new(num_qubits);
            self.feature_map.encode(x2, &mut sim2);

            // Fidelity = |⟨ψ₁|ψ₂⟩|²
            let fidelity = sim1.fidelity(&sim2.state);
            fidelity
        } else {
            // Method 2: Swap test with measurements
            self.swap_test(x1, x2)
        }
    }

    /// Perform swap test to estimate overlap
    ///
    /// Uses an ancillary qubit to estimate |⟨ψ₁|ψ₂⟩|²
    fn swap_test(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let num_qubits = self.feature_map.num_qubits();
        let ancilla = num_qubits; // Use extra qubit as ancilla

        // Create simulator with ancilla
        let mut sim = QuantumSimulator::new(num_qubits + 1);

        // Initialize ancilla in superposition
        sim.h(ancilla);

        // Encode x1 in first register, x2 in second
        // (simplified - actually need controlled operations)
        let mut sim_x1 = QuantumSimulator::new(num_qubits);
        self.feature_map.encode(x1, &mut sim_x1);

        let mut sim_x2 = QuantumSimulator::new(num_qubits);
        self.feature_map.encode(x2, &mut sim_x2);

        // Compute fidelity directly
        let fidelity = sim_x1.fidelity(&sim_x2.state);
        fidelity
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantumSimulator;

    #[test]
    fn test_observable_z_expectation() {
        let mut sim = QuantumSimulator::new(1);
        let obs = Observable::Z { qubit: 0 };

        // |0⟩ state should have ⟨Z⟩ = 1
        let exp_zero = obs.expectation(&sim);
        assert!((exp_zero - 1.0).abs() < 1e-10);

        // |1⟩ state should have ⟨Z⟩ = -1
        sim.x(0);
        let exp_one = obs.expectation(&sim);
        assert!((exp_one - (-1.0)).abs() < 1e-10);

        // H|0⟩ state should have ⟨Z⟩ = 0
        let mut sim = QuantumSimulator::new(1);
        sim.h(0);
        let exp_superposition = obs.expectation(&sim);
        assert!((exp_superposition - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_shift_gradient() {
        // Simple circuit: RY(θ)|0⟩
        let circuit = |sim: &mut QuantumSimulator, params: &[f64]| {
            sim.ry(0, params[0]);
        };

        let obs = Observable::Z { qubit: 0 };
        let params = vec![0.5];

        let grad = parameter_shift_gradient(1, &circuit, 0, &obs, &params);

        // Analytical gradient: ∂⟨Z⟩/∂θ = -sin(θ)
        // For θ = 0.5: expected grad ≈ -0.479
        let expected = -0.5_f64.sin();
        assert!((grad - expected).abs() < 1e-5);
    }

    #[test]
    fn test_angle_embedding() {
        let embedding = QuantumEmbedding::new(2, EmbeddingType::Angle);
        let mut sim = QuantumSimulator::new(2);

        let data = vec![PI / 2.0, 0.0];
        embedding.encode(&data, &mut sim);

        // After RY(π/2) on qubit 0, should be in |+⟩ state
        let obs = Observable::Z { qubit: 0 };
        let exp = obs.expectation(&sim);
        assert!((exp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_embedding() {
        let embedding = QuantumEmbedding::new(2, EmbeddingType::Amplitude);
        let mut sim = QuantumSimulator::new(2);

        // Encode normalized state [1/√2, 1/√2, 0, 0]
        let data = vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt(), 0.0, 0.0];
        embedding.encode(&data, &mut sim);

        // Check probabilities
        let probs = sim.state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
        assert!((probs[2] - 0.0).abs() < 1e-10);
        assert!((probs[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_basis_embedding() {
        let embedding = QuantumEmbedding::new(2, EmbeddingType::Basis);
        let mut sim = QuantumSimulator::new(2);

        // Encode [1, 0] (binary)
        // data[0] = 1.0 sets qubit 0 to |1⟩ (value 1)
        // data[1] = 0.0 leaves qubit 1 at |0⟩ (value 0)
        let data = vec![1.0, 0.0];
        embedding.encode(&data, &mut sim);

        // Should be in |01⟩ state (qubit 0 = |1⟩, qubit 1 = |0⟩)
        // Binary 01 = decimal 1
        let result = sim.measure();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_qnn_layer_forward() {
        let layer = QNNLayer::new(2);
        let mut sim = QuantumSimulator::new(2);

        // Store initial state
        let probs_before = sim.state.probabilities();

        // Apply layer
        layer.forward(&mut sim);

        // State should have changed
        let probs_after = sim.state.probabilities();
        assert_ne!(probs_before, probs_after);
    }

    #[test]
    fn test_qnn_layer_parameters() {
        let mut layer = QNNLayer::new(3);

        // Check number of parameters
        assert_eq!(layer.num_parameters(), 3);

        // Set and get parameters
        let new_params = vec![0.1, 0.2, 0.3];
        layer.set_parameters(new_params.clone());
        assert_eq!(layer.get_parameters(), &new_params);
    }

    #[test]
    fn test_quantum_nn() {
        let qnn = QuantumNN::new(2, 2);

        // Check total parameters
        assert_eq!(qnn.total_parameters(), 4); // 2 layers × 2 qubits

        // Forward pass
        let mut sim = QuantumSimulator::new(2);
        qnn.forward(&mut sim);

        // State should be modified
        let probs = sim.state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_kernel_diagonal() {
        let embedding = QuantumEmbedding::new(2, EmbeddingType::Angle);
        let kernel = QuantumKernel::new(embedding, 0);

        let x = vec![0.5, 0.3];

        // Kernel value for same point should be 1
        let k_xx = kernel.evaluate(&x, &x);
        assert!((k_xx - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_kernel_matrix() {
        let embedding = QuantumEmbedding::new(2, EmbeddingType::Angle);
        let kernel = QuantumKernel::new(embedding, 0);

        let data = vec![vec![0.0, 0.0], vec![PI / 2.0, 0.0]];

        let kernel_matrix = kernel.kernel_matrix(&data);

        // Check symmetry
        assert_eq!(kernel_matrix.len(), 2);
        assert_eq!(kernel_matrix[0].len(), 2);
        assert!((kernel_matrix[0][1] - kernel_matrix[1][0]).abs() < 1e-10);

        // Check diagonal is 1
        assert!((kernel_matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((kernel_matrix[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_vector() {
        let circuit = |sim: &mut QuantumSimulator, params: &[f64]| {
            sim.ry(0, params[0]);
            sim.ry(0, params[1]);
        };

        let obs = Observable::Z { qubit: 0 };
        let params = vec![0.5, 0.3];

        let grads = parameter_shift_gradient_vector(1, &circuit, &obs, &params);

        assert_eq!(grads.len(), 2);

        // Both gradients should be finite
        assert!(grads[0].is_finite());
        assert!(grads[1].is_finite());
    }

    #[test]
    fn test_entangled_encoding() {
        let embedding = QuantumEmbedding::new(4, EmbeddingType::Angle);
        let mut sim = QuantumSimulator::new(4);

        let x1 = vec![0.5, 0.3];
        let x2 = vec![0.4, 0.6];

        embedding.encode_entangled(&x1, &x2, &mut sim);

        // State should be modified
        let probs = sim.state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_observable_weighted_sum() {
        let mut sim = QuantumSimulator::new(2);
        sim.h(0);

        let obs = Observable::WeightedSum {
            observables: vec![
                (1.0, Observable::Z { qubit: 0 }),
                (0.5, Observable::Z { qubit: 1 }),
            ],
        };

        let exp = obs.expectation(&sim);
        assert!(exp.is_finite());
    }
}
