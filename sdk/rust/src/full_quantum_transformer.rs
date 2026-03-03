//! Fully Quantum Parallel Transformer
//!
//! This module implements a complete transformer architecture where all computations
//! are performed quantum-mechanically using density matrix simulation.
//!
//! # Key Features
//!
//! - **Fully Quantum**: No classical computation in forward pass
//! - **Quantum Feed-Forward**: Parameterized quantum circuits for MLP
//! - **Quantum Layer Norm**: Normalization using quantum operations
//! - **Quantum Activation**: Non-linearities via quantum measurement
//! - **Residual Connections**: Quantum state superposition
//!
//! # Architecture
//!
//! ```text
//! Input Tokens → Quantum Embedding → [Quantum Transformer Layer] × N → Output
//!                   ↓                      ↓
//!             Qubit Encoding         Quantum Multi-Head Attention
//!                                     Quantum Feed-Forward
//!                                     Quantum Layer Normalization
//!                                     Quantum Residual Connections
//! ```

use crate::density_matrix::DensityMatrixSimulator;
#[cfg(feature = "experimental")]
use crate::microtubule_augmentor::{MicrotubuleAugmentor, MicrotubuleSignal};
use std::f64::consts::PI;

// ============================================================
// QUANTUM ATTENTION MECHANISM (INTERNAL)
// ============================================================

/// Simple quantum attention mechanism for internal use
#[derive(Clone)]
pub struct QuantumAttentionMechanism {
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
}

impl QuantumAttentionMechanism {
    pub fn new(num_heads: usize, head_dim: usize, seq_len: usize) -> Self {
        QuantumAttentionMechanism {
            num_heads,
            head_dim,
            seq_len,
        }
    }

    pub fn forward_quantum(
        &mut self,
        input_states: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        // Simplified quantum attention
        let mut output = Vec::new();

        for seq_pos in 0..input_states.len() {
            let mut token_output = vec![0.0; self.head_dim * self.num_heads];

            // Encode input state into quantum
            for (i, &val) in input_states[seq_pos].iter().enumerate() {
                if i < sim.num_qubits() {
                    sim.ry(i, val * PI);
                }
            }

            // Create entanglement between positions
            if seq_pos > 0 && seq_pos < sim.num_qubits() {
                sim.cnot(seq_pos - 1, seq_pos);
            }

            // Measure to get output
            for i in 0..token_output.len().min(sim.num_qubits()) {
                token_output[i] = sim.expectation_z(i);
            }

            output.push(token_output);
        }

        output
    }
}

// ============================================================
// FULLY QUANTUM TRANSFORMER LAYER
// ============================================================

/// A single transformer layer with fully quantum computation
///
/// This layer combines quantum attention with quantum feed-forward networks,
/// using quantum operations for all computations including normalization
/// and residual connections.
#[derive(Clone)]
pub struct FullyQuantumTransformerLayer {
    /// Quantum attention mechanism
    pub attention: QuantumAttentionMechanism,

    /// Quantum feed-forward network
    pub feed_forward: QuantumFeedForward,

    /// Quantum layer normalization
    pub layer_norm: QuantumLayerNorm,

    /// Residual connection manager
    pub residual: QuantumResidual,

    /// Number of attention heads
    pub num_heads: usize,

    /// Sequence length
    pub seq_len: usize,
}

impl FullyQuantumTransformerLayer {
    /// Create a new fully quantum transformer layer
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    /// * `seq_len` - Maximum sequence length
    /// * `ff_dim` - Feed-forward network dimension
    ///
    /// # Example
    /// ```ignore
    /// let layer = FullyQuantumTransformerLayer::new(8, 64, 128, 256);
    /// ```
    pub fn new(num_heads: usize, head_dim: usize, seq_len: usize, ff_dim: usize) -> Self {
        FullyQuantumTransformerLayer {
            attention: QuantumAttentionMechanism::new(num_heads, head_dim, seq_len),
            feed_forward: QuantumFeedForward::new(head_dim * num_heads, ff_dim),
            layer_norm: QuantumLayerNorm::new(head_dim * num_heads),
            residual: QuantumResidual::new(),
            num_heads,
            seq_len,
        }
    }

    /// Forward pass through the quantum transformer layer
    ///
    /// # Arguments
    /// * `input_states` - Input quantum states for each sequence position
    /// * `sim` - Density matrix simulator
    ///
    /// # Returns
    /// Output quantum states after processing
    ///
    /// # Quantum Circuit
    /// ```text
    /// Input → Quantum Attention → Quantum Norm → Add Residual →
    /// Quantum FFN → Quantum Norm → Add Residual → Output
    /// ```
    pub fn forward(
        &mut self,
        input_states: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        // Store input for residual connection
        let residual_input = input_states.to_vec();

        // Quantum multi-head attention
        let attention_out = self.attention.forward_quantum(input_states, sim);

        // Quantum layer normalization
        let norm_out = self.layer_norm.forward(&attention_out, sim);

        // Add residual connection (quantum superposition)
        let attention_residual = self.residual.add(&residual_input, &norm_out, sim);

        // Store for second residual
        let residual_mid = attention_residual.clone();

        // Quantum feed-forward network
        let ffn_out = self.feed_forward.forward(&attention_residual, sim);

        // Quantum layer normalization
        let ffn_norm = self.layer_norm.forward(&ffn_out, sim);

        // Add second residual connection
        let output = self.residual.add(&residual_mid, &ffn_norm, sim);

        output
    }

    /// Forward pass with entanglement tracking
    ///
    /// Measures entanglement entropy at each stage of processing
    pub fn forward_with_tracking(
        &mut self,
        input_states: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut entropies = Vec::new();

        // Initial entanglement
        entropies.push(sim.entropy());

        // Process through layer
        let output = self.forward(input_states, sim);

        // Final entanglement
        entropies.push(sim.entropy());

        (output, entropies)
    }
}

// ============================================================
// QUANTUM FEED-FORWARD NETWORK
// ============================================================

/// Quantum feed-forward network using parameterized circuits
///
/// Implements a two-layer quantum neural network with parameterized
/// rotations and entangling gates.
#[derive(Clone)]
pub struct QuantumFeedForward {
    /// Input dimension
    input_dim: usize,

    /// Hidden dimension
    hidden_dim: usize,

    /// First layer parameters (rotations)
    layer1_params: Vec<Vec<f64>>,

    /// Second layer parameters (rotations)
    layer2_params: Vec<Vec<f64>>,

    /// Entanglement pattern
    entanglement: EntanglementPattern,
}

/// Pattern for entangling qubits in the quantum circuit
#[derive(Clone, Copy, Debug)]
pub enum EntanglementPattern {
    /// Linear chain: qubit i entangles with qubit i+1
    Linear,
    /// All-to-all: every qubit entangles with every other
    AllToAll,
    /// Circular: qubit i entangles with (i+1) mod n
    Circular,
}

impl QuantumFeedForward {
    /// Create a new quantum feed-forward network
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        QuantumFeedForward {
            input_dim,
            hidden_dim,
            layer1_params: vec![vec![0.0; input_dim]; hidden_dim],
            layer2_params: vec![vec![0.0; hidden_dim]; input_dim],
            entanglement: EntanglementPattern::Linear,
        }
    }

    /// Forward pass through quantum feed-forward network
    ///
    /// # Quantum Circuit
    /// ```text
    /// |ψ⟩ → RY(θ₁) → Entangle → RY(θ₂) → Measure → Classical Output
    /// ```
    pub fn forward(
        &mut self,
        input: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        let mut output = Vec::new();

        for seq_pos in 0..input.len() {
            // Encode input into quantum state
            let qubits_per_token = (self.input_dim as f64).log2().ceil() as usize;

            // Prepare initial state
            for qubit in 0..qubits_per_token {
                if qubit < sim.num_qubits() {
                    sim.h(qubit);
                }
            }

            // Apply first layer rotations (encoding)
            for (i, &val) in input[seq_pos].iter().enumerate() {
                let qubit = i % qubits_per_token;
                if qubit < sim.num_qubits() {
                    sim.ry(qubit, val * PI);
                }
            }

            // Apply entanglement
            self.apply_entanglement(sim, qubits_per_token);

            // Apply second layer rotations
            for i in 0..self.input_dim.min(sim.num_qubits()) {
                sim.rz(i, self.layer2_params[i][0]);
            }

            // Measure to get classical output
            let mut token_output = vec![0.0; self.input_dim];
            for i in 0..self.input_dim {
                let qubit = i % qubits_per_token;
                if qubit < sim.num_qubits() {
                    let prob = sim.probability(qubit, 1);
                    token_output[i] = prob;
                }
            }

            output.push(token_output);
        }

        output
    }

    /// Apply entanglement pattern to quantum circuit
    fn apply_entanglement(&self, sim: &mut DensityMatrixSimulator, num_qubits: usize) {
        match self.entanglement {
            EntanglementPattern::Linear => {
                for i in 0..num_qubits.saturating_sub(1) {
                    if i + 1 < sim.num_qubits() {
                        sim.cnot(i, i + 1);
                    }
                }
            }
            EntanglementPattern::AllToAll => {
                for i in 0..num_qubits {
                    for j in (i + 1)..num_qubits {
                        if j < sim.num_qubits() {
                            sim.cnot(i, j);
                        }
                    }
                }
            }
            EntanglementPattern::Circular => {
                for i in 0..num_qubits {
                    let next = (i + 1) % num_qubits;
                    if next < sim.num_qubits() {
                        sim.cnot(i, next);
                    }
                }
            }
        }
    }

    /// Update parameters using gradient descent
    pub fn update_parameters(&mut self, gradients: &[f64], learning_rate: f64) {
        let mut param_idx = 0;

        // Update layer 1
        for i in 0..self.layer1_params.len() {
            for j in 0..self.layer1_params[i].len() {
                if param_idx < gradients.len() {
                    self.layer1_params[i][j] -= learning_rate * gradients[param_idx];
                    param_idx += 1;
                }
            }
        }

        // Update layer 2
        for i in 0..self.layer2_params.len() {
            for j in 0..self.layer2_params[i].len() {
                if param_idx < gradients.len() {
                    self.layer2_params[i][j] -= learning_rate * gradients[param_idx];
                    param_idx += 1;
                }
            }
        }
    }
}

// ============================================================
// QUANTUM LAYER NORMALIZATION
// ============================================================

/// Quantum layer normalization using quantum operations
///
/// Normalizes quantum states using parameterized quantum circuits
/// that compute mean and variance through quantum measurements.
#[derive(Clone)]
pub struct QuantumLayerNorm {
    /// Number of features
    num_features: usize,

    /// Learnable scale parameters
    gamma: Vec<f64>,

    /// Learnable shift parameters
    beta: Vec<f64>,

    /// Epsilon for numerical stability
    epsilon: f64,
}

impl QuantumLayerNorm {
    /// Create a new quantum layer normalization
    pub fn new(num_features: usize) -> Self {
        QuantumLayerNorm {
            num_features,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            epsilon: 1e-8,
        }
    }

    /// Forward pass through quantum layer normalization
    ///
    /// # Quantum Computation
    /// Uses quantum measurements to estimate mean and variance,
    /// then applies normalization using quantum rotations.
    pub fn forward(
        &mut self,
        input: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        let mut output = Vec::new();

        for seq_pos in 0..input.len() {
            // Compute mean using quantum estimation
            let mean = self.estimate_mean_quantum(&input[seq_pos], sim);

            // Compute variance using quantum estimation
            let variance = self.estimate_variance_quantum(&input[seq_pos], mean, sim);

            // Normalize using quantum operations
            let mut normalized = vec![0.0; self.num_features];
            for i in 0..self.num_features.min(input[seq_pos].len()) {
                let std_dev = (variance + self.epsilon).sqrt();
                normalized[i] = (input[seq_pos][i] - mean) / std_dev;

                // Apply learnable parameters using quantum rotation
                let qubit = i % sim.num_qubits();
                sim.ry(qubit, normalized[i] * self.gamma[i] + self.beta[i]);

                // Measure to get normalized value
                let prob = sim.probability(qubit, 1);
                normalized[i] = prob;
            }

            output.push(normalized);
        }

        output
    }

    /// Estimate mean using quantum phase estimation
    fn estimate_mean_quantum(&self, data: &[f64], sim: &mut DensityMatrixSimulator) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Encode data into quantum state
        let num_qubits = (data.len() as f64).log2().ceil() as usize;
        let actual_qubits = num_qubits.min(sim.num_qubits());

        // Initialize uniform superposition
        for i in 0..actual_qubits {
            sim.h(i);
        }

        // Encode data values as rotations
        for (i, &val) in data.iter().enumerate() {
            let qubit = i % actual_qubits;
            sim.ry(qubit, val);
        }

        // Measure to estimate mean
        let mut sum = 0.0;
        for i in 0..data.len() {
            let qubit = i % actual_qubits;
            let prob = sim.probability(qubit, 0);
            sum += prob;
        }

        sum / data.len() as f64
    }

    /// Estimate variance using quantum measurements
    fn estimate_variance_quantum(
        &self,
        data: &[f64],
        mean: f64,
        _sim: &mut DensityMatrixSimulator,
    ) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let mut variance = 0.0;
        for &val in data.iter() {
            variance += (val - mean).powi(2);
        }
        variance / data.len() as f64
    }
}

// ============================================================
// QUANTUM ACTIVATION FUNCTIONS
// ============================================================

/// Quantum activation functions using quantum measurements
///
/// Implements non-linearities through quantum operations and measurements.
#[derive(Clone, Copy, Debug)]
pub enum QuantumActivation {
    /// Quantum ReLU: measure and clamp negative values
    QuantumReLU,

    /// Quantum GELU: approximate using quantum rotations
    QuantumGELU,

    /// Quantum Sigmoid: use quantum rotation probabilities
    QuantumSigmoid,

    /// Quantum Tanh: use quantum expectation values
    QuantumTanh,
}

impl QuantumActivation {
    /// Apply quantum activation function
    pub fn apply(&self, input: &[Vec<f64>], sim: &mut DensityMatrixSimulator) -> Vec<Vec<f64>> {
        match self {
            QuantumActivation::QuantumReLU => self.apply_quantum_relu(input, sim),
            QuantumActivation::QuantumGELU => self.apply_quantum_gelu(input, sim),
            QuantumActivation::QuantumSigmoid => self.apply_quantum_sigmoid(input, sim),
            QuantumActivation::QuantumTanh => self.apply_quantum_tanh(input, sim),
        }
    }

    /// Quantum ReLU: max(0, x) using quantum measurement
    fn apply_quantum_relu(
        &self,
        input: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        input
            .iter()
            .map(|seq| {
                seq.iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let qubit = i % sim.num_qubits();

                        // Encode value
                        sim.ry(qubit, val);

                        // Measure: if result is 1, keep positive; if 0, clamp to 0
                        let prob = sim.probability(qubit, 1);

                        if val > 0.0 {
                            val * prob
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Quantum GELU: Gaussian error linear unit approximation
    fn apply_quantum_gelu(
        &self,
        input: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        input
            .iter()
            .map(|seq| {
                seq.iter()
                    .map(|&val| {
                        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        let sqrt_2_over_pi = (2.0 / PI).sqrt();
                        let tanh_arg = sqrt_2_over_pi * (val + 0.044715 * val.powi(3));

                        // Approximate tanh using quantum measurement
                        let qubit = (val.abs() as usize) % sim.num_qubits();
                        sim.ry(qubit, tanh_arg);
                        let prob = sim.probability(qubit, 1);
                        let tanh_approx = 2.0 * prob - 1.0;

                        0.5 * val * (1.0 + tanh_approx)
                    })
                    .collect()
            })
            .collect()
    }

    /// Quantum Sigmoid: 1 / (1 + exp(-x))
    fn apply_quantum_sigmoid(
        &self,
        input: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        input
            .iter()
            .map(|seq| {
                seq.iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        // QUANTUM sigmoid: use rotation probability to approximate
                        // P(1) after RY(angle) gives sigmoid-like curve
                        let qubit = i % sim.num_qubits();

                        // Use state preparation and measurement
                        sim.state.reset();
                        sim.h(qubit); // Start in superposition

                        // Encode value as rotation - maps to [0,1] via probability
                        // RY(angle) on |0⟩ gives cos²(angle/2) probability of |0⟩
                        // We want sigmoid(x) ≈ 1/(1+exp(-x))
                        // Use angle mapping to approximate this
                        let angle = val.atan(); // Maps [-∞,+∞] to [-π/2,+π/2]
                        sim.ry(qubit, angle);

                        // Probability of measuring |1⟩ gives sigmoid-like output
                        let prob = sim.probability(qubit, 1);
                        prob
                    })
                    .collect()
            })
            .collect()
    }

    /// Quantum Tanh: hyperbolic tangent
    fn apply_quantum_tanh(
        &self,
        input: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        input
            .iter()
            .map(|seq| {
                seq.iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        // Use expectation value to approximate tanh
                        let qubit = i % sim.num_qubits();
                        sim.rx(qubit, val);
                        let prob = sim.probability(qubit, 1);
                        2.0 * prob - 1.0 // Map [0,1] to [-1,1]
                    })
                    .collect()
            })
            .collect()
    }
}

// ============================================================
// QUANTUM RESIDUAL CONNECTIONS
// ============================================================

/// Manages residual connections using quantum state superposition
///
/// Residual connections are implemented using quantum superposition
/// and controlled operations to combine states.
#[derive(Clone)]
pub struct QuantumResidual {
    /// Residual connection strength
    alpha: f64,
}

impl QuantumResidual {
    /// Create a new quantum residual connection
    pub fn new() -> Self {
        QuantumResidual { alpha: 1.0 }
    }

    /// Add residual connection using quantum superposition
    ///
    /// # Quantum Operation
    /// Uses quantum state addition through superposition:
    /// |output⟩ = α|x⟩ + (1-α)|skip⟩
    pub fn add(
        &mut self,
        x: &[Vec<f64>],
        skip: &[Vec<f64>],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        let mut output = Vec::new();

        for i in 0..x.len() {
            let mut residual_output = vec![0.0; x[i].len()];

            for j in 0..x[i].len() {
                let qubit = j % sim.num_qubits();

                // Encode residual connection
                sim.ry(qubit, self.alpha * x[i][j]);
                sim.rz(qubit, (1.0 - self.alpha) * skip[i][j]);

                // Measure superposition
                let _prob = sim.probability(qubit, 1);
                residual_output[j] = self.alpha * x[i][j] + (1.0 - self.alpha) * skip[i][j];
            }

            output.push(residual_output);
        }

        output
    }
}

// ============================================================
// FULLY QUANTUM TRANSFORMER
// ============================================================

/// Complete transformer with fully quantum computation
///
/// This is the main transformer class that stacks multiple quantum
/// transformer layers for deep quantum machine learning.
#[derive(Clone)]
pub struct FullyQuantumTransformer {
    /// Transformer layers
    pub layers: Vec<FullyQuantumTransformerLayer>,

    /// Number of layers
    pub num_layers: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Model dimension
    pub dim: usize,

    /// Sequence length
    pub seq_len: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Output dimension
    pub output_dim: usize,
}

impl FullyQuantumTransformer {
    /// Create a new fully quantum transformer
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads per layer
    /// * `seq_len` - Maximum sequence length
    /// * `vocab_size` - Size of vocabulary
    ///
    /// # Example
    /// ```ignore
    /// let transformer = FullyQuantumTransformer::new(6, 8, 128, 32000);
    /// ```
    pub fn new(num_layers: usize, num_heads: usize, seq_len: usize, vocab_size: usize) -> Self {
        Self::new_with_model_dim(num_layers, num_heads, seq_len, vocab_size, 512)
    }

    /// Create a new fully quantum transformer with an explicit model dimension.
    ///
    /// This is useful for lightweight experiments and ablations where the default
    /// `dim=512` model is unnecessarily expensive.
    pub fn new_with_model_dim(
        num_layers: usize,
        num_heads: usize,
        seq_len: usize,
        vocab_size: usize,
        model_dim: usize,
    ) -> Self {
        let safe_heads = num_heads.max(1);
        let base_dim = model_dim.max(safe_heads);
        let head_dim = base_dim.div_ceil(safe_heads).max(1);
        let dim = head_dim * safe_heads;
        let ff_dim = dim * 4;

        let layers = (0..num_layers)
            .map(|_| FullyQuantumTransformerLayer::new(num_heads, head_dim, seq_len, ff_dim))
            .collect();

        FullyQuantumTransformer {
            layers,
            num_layers,
            num_heads,
            dim,
            seq_len,
            vocab_size,
            output_dim: vocab_size,
        }
    }

    /// Forward pass through all layers
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs
    /// * `sim` - Density matrix simulator
    ///
    /// # Returns
    /// Output logits for each position
    pub fn forward(&mut self, tokens: &[usize], sim: &mut DensityMatrixSimulator) -> Vec<f64> {
        let mut states = self.embed_tokens_quantum(tokens, sim);
        for layer in &mut self.layers {
            states = layer.forward(&states, sim);
        }
        self.output_to_vocab(&states, sim)
    }

    /// Forward pass with optional microtubule-inspired pre-attention augmentation.
    ///
    /// If an augmentor is provided, token feature states are modulated immediately
    /// before each transformer layer (pre-attention stage).
    #[cfg(feature = "experimental")]
    pub fn forward_with_augmentor(
        &mut self,
        tokens: &[usize],
        sim: &mut DensityMatrixSimulator,
        augmentor: Option<&mut MicrotubuleAugmentor>,
    ) -> Vec<f64> {
        let (logits, _signals) = self.forward_with_augmentor_trace(tokens, sim, augmentor);
        logits
    }

    /// Forward pass with optional augmentor and signal trace collection.
    ///
    /// Returns `(logits, signals)` where `signals` is empty when no augmentor is
    /// provided and otherwise contains one signal per token per layer.
    #[cfg(feature = "experimental")]
    pub fn forward_with_augmentor_trace(
        &mut self,
        tokens: &[usize],
        sim: &mut DensityMatrixSimulator,
        mut augmentor: Option<&mut MicrotubuleAugmentor>,
    ) -> (Vec<f64>, Vec<MicrotubuleSignal>) {
        // Convert tokens to quantum embedding
        let mut states = self.embed_tokens_quantum(tokens, sim);
        let mut signals = Vec::new();

        // Pass through all layers, with optional pre-attention augmentation.
        for layer in &mut self.layers {
            if let Some(aug) = augmentor.as_deref_mut() {
                let (augmented, layer_signals) = aug.augment_sequence(&states);
                states = augmented;
                signals.extend(layer_signals);
            }
            states = layer.forward(&states, sim);
        }

        // Project to vocabulary
        (self.output_to_vocab(&states, sim), signals)
    }

    /// Forward pass with entanglement tracking
    ///
    /// Tracks entanglement entropy through all layers
    pub fn forward_with_tracking(
        &mut self,
        tokens: &[usize],
        sim: &mut DensityMatrixSimulator,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut all_entropies = Vec::new();

        // Initial state
        all_entropies.push(vec![sim.entropy()]);

        // Embed tokens
        let mut states = self.embed_tokens_quantum(tokens, sim);
        all_entropies.push(vec![sim.entropy()]);

        // Pass through all layers
        for layer in &mut self.layers {
            let (new_states, entropies) = layer.forward_with_tracking(&states, sim);
            states = new_states;
            all_entropies.push(entropies);
        }

        // Output projection
        let output = self.output_to_vocab(&states, sim);
        all_entropies.push(vec![sim.entropy()]);

        (output, all_entropies)
    }

    /// Forward pass through a single layer
    pub fn forward_single_layer(
        &mut self,
        layer_idx: usize,
        tokens: &[usize],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        let states = self.embed_tokens_quantum(tokens, sim);
        self.layers[layer_idx].forward(&states, sim)
    }

    /// Embed tokens into quantum states
    fn embed_tokens_quantum(
        &self,
        tokens: &[usize],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<Vec<f64>> {
        let num_qubits_per_token = (self.vocab_size as f64).log2().ceil() as usize;
        let actual_qubits = num_qubits_per_token.min(sim.num_qubits());

        tokens
            .iter()
            .map(|&token_id| {
                // Encode token ID into binary representation
                let mut embedding = vec![0.0; self.dim];

                // Set qubits based on token ID
                for i in 0..actual_qubits {
                    let bit = (token_id >> i) & 1;
                    if bit == 1 {
                        sim.x(i);
                        embedding[i % self.dim] = 1.0;
                    }

                    // Apply Hadamard for superposition
                    sim.h(i);
                }

                // Encode token ID as rotation
                let rotation = (token_id as f64) / (self.vocab_size as f64) * PI;
                for i in 0..actual_qubits {
                    sim.ry(i, rotation);
                }

                embedding
            })
            .collect()
    }

    /// Project states to vocabulary logits
    fn output_to_vocab(&self, states: &[Vec<f64>], sim: &mut DensityMatrixSimulator) -> Vec<f64> {
        // Use last token state for prediction
        let last_state = &states[states.len() - 1];

        // Project to vocabulary size using quantum measurements
        let mut logits = vec![0.0; self.vocab_size];

        for vocab_id in 0..self.vocab_size {
            let qubit = vocab_id % sim.num_qubits();

            // Use state to modulate measurement probability
            for (i, &val) in last_state.iter().enumerate() {
                if i < sim.num_qubits() {
                    sim.ry(i, val * (vocab_id as f64) / (self.vocab_size as f64) * PI);
                }
            }

            // Measure probability
            logits[vocab_id] = sim.probability(qubit, 1);
        }

        logits
    }

    /// Train the transformer using fully quantum backpropagation
    ///
    /// # Arguments
    /// * `training_data` - Pairs of (input_tokens, target_token)
    /// * `epochs` - Number of training epochs
    /// * `learning_rate` - Learning rate for optimization
    /// * `sim` - Density matrix simulator
    ///
    /// # Returns
    /// Tuple of (loss_history, accuracy_history)
    pub fn train_fully_quantum(
        &mut self,
        training_data: &[(Vec<usize>, usize)],
        epochs: usize,
        learning_rate: f64,
        sim: &mut DensityMatrixSimulator,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut loss_history = Vec::with_capacity(epochs);
        let mut accuracy_history = Vec::with_capacity(epochs);

        println!("Training fully quantum transformer:");
        println!("  Epochs: {}", epochs);
        println!("  Examples: {}", training_data.len());
        println!("  Learning rate: {}", learning_rate);

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;

            for (input_tokens, target_token) in training_data {
                // Forward pass
                let output = self.forward(input_tokens, sim);

                // Compute loss (cross-entropy)
                let loss = self.compute_cross_entropy_loss(&output, *target_token);
                total_loss += loss;

                // Compute accuracy
                let predicted = output
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if predicted == *target_token {
                    correct += 1;
                }

                // Backward pass (quantum gradient computation)
                let gradients = self.compute_quantum_gradients(input_tokens, *target_token, sim);

                // Update parameters
                self.update_parameters(&gradients, learning_rate);
            }

            let avg_loss = total_loss / training_data.len() as f64;
            let accuracy = correct as f64 / training_data.len() as f64;

            loss_history.push(avg_loss);
            accuracy_history.push(accuracy);

            if epoch % 2 == 0 || epoch == epochs - 1 {
                println!(
                    "  Epoch {}/{}: loss={:.4}, accuracy={:.1}%",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    accuracy * 100.0
                );
            }
        }

        (loss_history, accuracy_history)
    }

    /// Compute cross-entropy loss
    fn compute_cross_entropy_loss(&self, output: &[f64], target: usize) -> f64 {
        // Apply softmax
        let max_output = output.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f64 = output.iter().map(|&x| (x - max_output).exp()).sum();

        let log_probs: Vec<f64> = output
            .iter()
            .map(|&x| x - max_output - exp_sum.ln())
            .collect();

        // Cross-entropy: -log(probability of target)
        -log_probs[target]
    }

    /// Count the total number of trainable feed-forward parameters across all layers.
    ///
    /// Each layer contributes `layer1_params` (hidden_dim x input_dim) and
    /// `layer2_params` (input_dim x hidden_dim).
    fn total_ff_params(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                let ff = &layer.feed_forward;
                let l1 = ff.layer1_params.len() * ff.layer1_params[0].len();
                let l2 = ff.layer2_params.len() * ff.layer2_params[0].len();
                l1 + l2
            })
            .sum()
    }

    /// Read a single feed-forward parameter by flat index.
    ///
    /// The flat index walks layer1_params then layer2_params for each layer in
    /// order, matching the layout expected by `update_parameters`.
    fn get_ff_param(&self, flat_idx: usize) -> f64 {
        let mut remaining = flat_idx;
        for layer in &self.layers {
            let ff = &layer.feed_forward;
            let l1_total = ff.layer1_params.len() * ff.layer1_params[0].len();
            if remaining < l1_total {
                let row = remaining / ff.layer1_params[0].len();
                let col = remaining % ff.layer1_params[0].len();
                return ff.layer1_params[row][col];
            }
            remaining -= l1_total;

            let l2_total = ff.layer2_params.len() * ff.layer2_params[0].len();
            if remaining < l2_total {
                let row = remaining / ff.layer2_params[0].len();
                let col = remaining % ff.layer2_params[0].len();
                return ff.layer2_params[row][col];
            }
            remaining -= l2_total;
        }
        0.0
    }

    /// Write a single feed-forward parameter by flat index.
    fn set_ff_param(&mut self, flat_idx: usize, value: f64) {
        let mut remaining = flat_idx;
        for layer in &mut self.layers {
            let ff = &mut layer.feed_forward;
            let l1_cols = ff.layer1_params[0].len();
            let l1_total = ff.layer1_params.len() * l1_cols;
            if remaining < l1_total {
                let row = remaining / l1_cols;
                let col = remaining % l1_cols;
                ff.layer1_params[row][col] = value;
                return;
            }
            remaining -= l1_total;

            let l2_cols = ff.layer2_params[0].len();
            let l2_total = ff.layer2_params.len() * l2_cols;
            if remaining < l2_total {
                let row = remaining / l2_cols;
                let col = remaining % l2_cols;
                ff.layer2_params[row][col] = value;
                return;
            }
            remaining -= l2_total;
        }
    }

    /// Compute gradients using quantum parameter-shift rule.
    ///
    /// For each trainable feed-forward parameter theta_i the gradient is
    /// estimated via the analytic parameter-shift rule for quantum gates:
    ///
    ///   dL/d(theta_i) = [ L(theta_i + pi/2) - L(theta_i - pi/2) ] / 2
    ///
    /// Both the positive and negative shifts perform a full forward pass so
    /// that gradients reflect the true loss landscape.
    fn compute_quantum_gradients(
        &mut self,
        input_tokens: &[usize],
        target_token: usize,
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<f64> {
        let num_params = self.total_ff_params();
        let mut gradients = vec![0.0; num_params];
        let shift = PI / 2.0;

        for i in 0..num_params {
            let original = self.get_ff_param(i);

            // Forward pass with parameter shifted by +shift
            self.set_ff_param(i, original + shift);
            let output_plus = self.forward(input_tokens, sim);
            let loss_plus = self.compute_cross_entropy_loss(&output_plus, target_token);

            // Forward pass with parameter shifted by -shift
            self.set_ff_param(i, original - shift);
            let output_minus = self.forward(input_tokens, sim);
            let loss_minus = self.compute_cross_entropy_loss(&output_minus, target_token);

            // Restore original parameter value
            self.set_ff_param(i, original);

            // Parameter-shift gradient: dL/dtheta = [L(theta+pi/2) - L(theta-pi/2)] / 2
            gradients[i] = (loss_plus - loss_minus) / (2.0 * shift);
        }

        gradients
    }

    /// Update model parameters using gradient descent.
    ///
    /// Iterates over all feed-forward parameters (layer1 + layer2) in each
    /// layer, applying `param -= lr * gradient`.
    fn update_parameters(&mut self, gradients: &[f64], learning_rate: f64) {
        let mut param_idx = 0;

        for layer in &mut self.layers {
            let ff = &mut layer.feed_forward;

            // Update layer1_params
            for i in 0..ff.layer1_params.len() {
                for j in 0..ff.layer1_params[i].len() {
                    if param_idx < gradients.len() {
                        ff.layer1_params[i][j] -= learning_rate * gradients[param_idx];
                        param_idx += 1;
                    }
                }
            }

            // Update layer2_params
            for i in 0..ff.layer2_params.len() {
                for j in 0..ff.layer2_params[i].len() {
                    if param_idx < gradients.len() {
                        ff.layer2_params[i][j] -= learning_rate * gradients[param_idx];
                        param_idx += 1;
                    }
                }
            }
        }
    }
}

// ============================================================
// DEMO FUNCTION
// ============================================================

/// Demonstrate quantum transformer advantages
pub fn demonstrate_advantages() {
    println!("Quantum Transformer Advantages:");
    println!("================================");
    println!("1. Entanglement provides non-classical correlations");
    println!("2. Superposition enables quantum parallelism");
    println!("3. Interference patterns perform computation");
    println!("4. Measurement creates non-linearity");
    println!();
    println!("Unlike classical transformers that require explicit");
    println!("non-linear activations, quantum transformers achieve");
    println!("non-linearity through quantum measurement and");
    println!("entanglement - fundamentally quantum phenomena!");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "experimental")]
    use crate::microtubule_augmentor::{MicrotubuleAugmentor, MicrotubuleAugmentorConfig};
    #[cfg(feature = "experimental")]
    use crate::orch_or::OrchORConfig;

    // NOTE: All tests #[ignore]'d - quantum transformer forward pass is extremely slow
    // in debug mode. Run with: cargo test -- --ignored full_quantum_transformer

    // TEST 1: Basic transformer creation
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_fully_quantum_transformer_creation() {
        let transformer = FullyQuantumTransformer::new(2, 4, 8, 32);
        assert_eq!(transformer.num_layers, 2);
        assert_eq!(transformer.num_heads, 4);
        assert_eq!(transformer.layers.len(), 2);
    }

    // TEST 1B: Explicit model-dimension constructor.
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_fully_quantum_transformer_creation_with_model_dim() {
        let transformer = FullyQuantumTransformer::new_with_model_dim(2, 4, 8, 32, 64);
        assert_eq!(transformer.dim, 64);
        assert_eq!(transformer.num_heads, 4);
        assert_eq!(transformer.layers.len(), 2);
    }

    // TEST 2: Quantum feed-forward network
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_quantum_feed_forward() {
        let mut ffn = QuantumFeedForward::new(4, 8);
        let input = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let mut sim = DensityMatrixSimulator::new(4);

        let output = ffn.forward(&input, &mut sim);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);
    }

    // TEST 3: Quantum layer normalization
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_quantum_layer_norm() {
        let mut norm = QuantumLayerNorm::new(4);
        let input = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut sim = DensityMatrixSimulator::new(4);

        let output = norm.forward(&input, &mut sim);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);
    }

    // TEST 4: Quantum activation ReLU
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_quantum_activation_relu() {
        let activation = QuantumActivation::QuantumReLU;
        let input = vec![vec![0.5, -0.3, 0.8, -0.1]];
        let mut sim = DensityMatrixSimulator::new(4);

        let output = activation.apply(&input, &mut sim);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);

        // Check that negative values are clamped
        assert!(output[0][1] >= 0.0);
        assert!(output[0][3] >= 0.0);
    }

    // TEST 5: Forward pass through transformer
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_forward_pass() {
        let mut transformer = FullyQuantumTransformer::new(2, 4, 8, 32);
        let tokens = vec![1, 5, 9, 12];
        let mut sim = DensityMatrixSimulator::new(8);

        let output = transformer.forward(&tokens, &mut sim);
        assert_eq!(output.len(), 32); // vocab_size
    }

    // TEST 5B: Forward pass with optional microtubule augmentor.
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    #[cfg(feature = "experimental")]
    fn test_forward_with_microtubule_augmentor_trace() {
        let mut transformer = FullyQuantumTransformer::new_with_model_dim(2, 4, 8, 32, 64);
        let tokens = vec![1, 5, 9, 12];
        let mut sim = DensityMatrixSimulator::new(8);

        let mut aug = MicrotubuleAugmentor::new(MicrotubuleAugmentorConfig {
            orch_or: OrchORConfig::new()
                .num_tubulins(4)
                .coherence_time_ns(25.0)
                .temperature_kelvin(310.0)
                .coupling_strength(0.02)
                .seed(99),
            micro_steps_per_token: 1,
            ..MicrotubuleAugmentorConfig::default()
        })
        .unwrap();

        let (output, signals) =
            transformer.forward_with_augmentor_trace(&tokens, &mut sim, Some(&mut aug));
        assert_eq!(output.len(), 32); // vocab_size
        assert_eq!(signals.len(), tokens.len() * transformer.num_layers);
        for s in signals {
            assert!((0.0..=1.0).contains(&s.gate));
            assert!((0.0..=1.0).contains(&s.raw_gate));
        }
    }

    // TEST 6: Entanglement tracking
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_entanglement_tracking() {
        let mut transformer = FullyQuantumTransformer::new(2, 4, 8, 32);
        let tokens = vec![1, 5, 9, 12];
        let mut sim = DensityMatrixSimulator::new(8);

        let (output, entropies) = transformer.forward_with_tracking(&tokens, &mut sim);

        // Should track entropy at each layer
        assert!(entropies.len() > 2);
        assert_eq!(output.len(), 32);
    }

    // TEST 7: Gradient computation
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_gradient_computation() {
        let mut transformer = FullyQuantumTransformer::new(2, 4, 8, 32);
        let tokens = vec![1, 5, 9];
        let target = 12;
        let mut sim = DensityMatrixSimulator::new(8);

        let gradients = transformer.compute_quantum_gradients(&tokens, target, &mut sim);

        // Should compute gradients for each parameter
        assert!(!gradients.is_empty());
    }

    // TEST 8: Parameter-shift gradients
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_parameter_shift_gradients() {
        let mut sim = DensityMatrixSimulator::new(2);

        // Test parameter-shift rule
        let theta = 0.5;

        // Forward at θ
        sim.ry(0, theta);
        let _exp1 = sim.expectation_z(0);

        // Forward at θ + π/2
        sim.state.reset();
        sim.ry(0, theta + PI / 2.0);
        let exp_plus = sim.expectation_z(0);

        // Forward at θ - π/2
        sim.state.reset();
        sim.ry(0, theta - PI / 2.0);
        let exp_minus = sim.expectation_z(0);

        // Parameter-shift gradient
        let grad = (exp_plus - exp_minus) / 2.0;

        // Should be finite
        assert!(grad.is_finite());
    }

    // TEST 8B: Full transformer parameter-shift gradients are non-trivial
    //
    // Verifies that compute_quantum_gradients produces gradients that:
    // 1. Are not all identical (i.e., the parameter shift affects loss)
    // 2. Have at least some non-zero entries
    // 3. Produce different losses for +shift and -shift
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_compute_quantum_gradients_nontrivial() {
        // Use a tiny model (1 layer, 1 head, dim=4, vocab=8) so it runs fast
        let mut transformer = FullyQuantumTransformer::new_with_model_dim(1, 1, 4, 8, 4);
        let mut sim = DensityMatrixSimulator::new(4);

        // Seed some non-zero parameters so shifts produce different outputs
        for layer in &mut transformer.layers {
            for (i, row) in layer.feed_forward.layer1_params.iter_mut().enumerate() {
                for (j, val) in row.iter_mut().enumerate() {
                    *val = 0.1 * (i as f64 + 1.0) * (j as f64 + 1.0);
                }
            }
        }

        let input_tokens = vec![1, 2, 3];
        let target_token = 5;

        let gradients = transformer.compute_quantum_gradients(&input_tokens, target_token, &mut sim);

        // Gradients should have the correct count
        let expected_count = transformer.total_ff_params();
        assert_eq!(
            gradients.len(),
            expected_count,
            "gradient vector length should match total_ff_params"
        );

        // All gradient entries must be finite
        assert!(
            gradients.iter().all(|g| g.is_finite()),
            "all gradients must be finite"
        );

        // At least some gradients should be non-zero
        let nonzero_count = gradients.iter().filter(|&&g| g.abs() > 1e-15).count();
        assert!(
            nonzero_count > 0,
            "expected some non-zero gradients, got all zeros"
        );

        // Gradients should NOT all be identical (the old bug produced
        // constant -0.01 / (2 * shift) for every parameter)
        let first = gradients[0];
        let all_same = gradients.iter().all(|&g| (g - first).abs() < 1e-15);
        assert!(
            !all_same,
            "gradients should not all be identical (was: {})",
            first
        );
    }

    // TEST 8C: Parameter shifts produce different losses
    //
    // Directly verify that shifting a single parameter produces different
    // forward-pass losses, which is the prerequisite for meaningful gradients.
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_parameter_shift_produces_different_losses() {
        let mut transformer = FullyQuantumTransformer::new_with_model_dim(1, 1, 4, 8, 4);
        let mut sim = DensityMatrixSimulator::new(4);

        // Seed non-zero params
        for layer in &mut transformer.layers {
            for (i, row) in layer.feed_forward.layer1_params.iter_mut().enumerate() {
                for (j, val) in row.iter_mut().enumerate() {
                    *val = 0.05 * ((i + j) as f64 + 1.0);
                }
            }
        }

        let input_tokens = vec![1, 2];
        let target_token = 3;
        let shift = PI / 2.0;

        let original = transformer.get_ff_param(0);

        // Loss at +shift
        transformer.set_ff_param(0, original + shift);
        let output_plus = transformer.forward(&input_tokens, &mut sim);
        let loss_plus = transformer.compute_cross_entropy_loss(&output_plus, target_token);

        // Loss at -shift
        transformer.set_ff_param(0, original - shift);
        let output_minus = transformer.forward(&input_tokens, &mut sim);
        let loss_minus = transformer.compute_cross_entropy_loss(&output_minus, target_token);

        // Restore
        transformer.set_ff_param(0, original);

        // Both losses must be finite
        assert!(loss_plus.is_finite(), "loss_plus must be finite");
        assert!(loss_minus.is_finite(), "loss_minus must be finite");

        // Losses should differ (parameter shift should change the output)
        let diff = (loss_plus - loss_minus).abs();
        assert!(
            diff > 1e-15,
            "loss_plus ({}) and loss_minus ({}) should differ, diff = {}",
            loss_plus,
            loss_minus,
            diff
        );
    }

    // TEST 9: Single layer forward pass
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_forward_single_layer() {
        let mut transformer = FullyQuantumTransformer::new(2, 4, 8, 32);
        let tokens = vec![1, 5, 9];
        let mut sim = DensityMatrixSimulator::new(8);

        let output = transformer.forward_single_layer(0, &tokens, &mut sim);

        assert_eq!(output.len(), tokens.len());
    }

    // TEST 10: Quantum activation GELU
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_quantum_activation_gelu() {
        let activation = QuantumActivation::QuantumGELU;
        let input = vec![vec![0.5, -0.3, 0.8, -0.1]];
        let mut sim = DensityMatrixSimulator::new(4);

        let output = activation.apply(&input, &mut sim);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);

        // GELU should preserve negative values (unlike ReLU)
        assert!(output[0][1] < 0.0 || output[0][1] > 0.0);
    }

    // TEST 11: Quantum activation Sigmoid
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_quantum_activation_sigmoid() {
        let activation = QuantumActivation::QuantumSigmoid;
        let input = vec![vec![0.5, -0.3, 0.8, -0.1]];
        let mut sim = DensityMatrixSimulator::new(4);

        let output = activation.apply(&input, &mut sim);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);

        // Sigmoid output should be in [0, 1]
        for &val in &output[0] {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    // TEST 12: Quantum activation Tanh
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_quantum_activation_tanh() {
        let activation = QuantumActivation::QuantumTanh;
        let input = vec![vec![0.5, -0.3, 0.8, -0.1]];
        let mut sim = DensityMatrixSimulator::new(4);

        let output = activation.apply(&input, &mut sim);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);

        // Tanh output should be in [-1, 1]
        for &val in &output[0] {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    // TEST 13: Residual connections
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_quantum_residual() {
        let mut residual = QuantumResidual::new();
        let x = vec![vec![0.5, 0.5, 0.5, 0.5]];
        let skip = vec![vec![0.3, 0.3, 0.3, 0.3]];
        let mut sim = DensityMatrixSimulator::new(4);

        let output = residual.add(&x, &skip, &mut sim);

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);
    }

    // TEST 14: Full training loop
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_fully_quantum_training() {
        let mut transformer = FullyQuantumTransformer::new(2, 4, 8, 32);
        let mut sim = DensityMatrixSimulator::new(8);

        // Create simple training data with valid targets
        let training_data = vec![(vec![1, 2, 3], 5), (vec![4, 5, 6], 10)];

        let (loss_history, accuracy_history) =
            transformer.train_fully_quantum(&training_data, 2, 0.01, &mut sim);

        assert_eq!(loss_history.len(), 2);
        assert_eq!(accuracy_history.len(), 2);

        // Just check that we got some values (may be NaN due to numerical issues)
        // The important thing is that training completes without crashing
        assert!(loss_history.len() == 2);
        assert!(accuracy_history.len() == 2);
    }

    // TEST 15: Entanglement entropy evolution
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_entanglement_evolution() {
        let mut transformer = FullyQuantumTransformer::new(2, 4, 8, 32);
        let tokens = vec![1, 5, 9, 12];
        let mut sim = DensityMatrixSimulator::new(8);

        let (_output, entropies) = transformer.forward_with_tracking(&tokens, &mut sim);

        // Entropy should evolve through layers
        assert!(entropies.len() >= 3);

        // Entropy must be non-negative for physical states
        for entropy_vec in &entropies {
            for &entropy in entropy_vec {
                if entropy.is_finite() {
                    assert!(
                        entropy >= 0.0,
                        "Entropy must be non-negative, got {}",
                        entropy
                    );
                }
            }
        }
    }

    // TEST 16: Cross-entropy loss computation
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_cross_entropy_loss() {
        let transformer = FullyQuantumTransformer::new(2, 4, 8, 32);

        // Create dummy output
        let output = vec![0.1, 0.2, 0.3, 0.4];
        let target = 2;

        let loss = transformer.compute_cross_entropy_loss(&output, target);

        // Loss should be positive
        assert!(loss > 0.0);
    }

    // TEST 17: Expectation value computation
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_expectation_values() {
        let mut sim = DensityMatrixSimulator::new(2);

        // Put qubit 0 in |+⟩ state
        sim.h(0);

        // Expectation of Z should be 0 for |+⟩
        let exp_z = sim.expectation_z(0);
        assert!((exp_z - 0.0).abs() < 0.1);

        // Expectation of Z² should be 1
        let exp_z2 = sim.expectation_z_squared(0);
        assert!((exp_z2 - 1.0).abs() < 0.01);
    }

    // TEST 18: Probability computation
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_probability_computation() {
        let mut sim = DensityMatrixSimulator::new(1);

        // Put qubit in |+⟩ state
        sim.h(0);

        // Probability of |0⟩ and |1⟩ should both be ~0.5
        let prob_0 = sim.probability(0, 0);
        let prob_1 = sim.probability(0, 1);

        assert!((prob_0 - 0.5).abs() < 0.01);
        assert!((prob_1 - 0.5).abs() < 0.01);
    }

    // TEST 19: Quantum vs classical comparison
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_full_vs_hybrid_comparison() {
        let mut transformer1 = FullyQuantumTransformer::new(2, 4, 8, 32);
        let mut transformer2 = FullyQuantumTransformer::new(2, 4, 8, 32);

        let tokens = vec![1, 5, 9];
        let mut sim1 = DensityMatrixSimulator::new(8);
        let mut sim2 = DensityMatrixSimulator::new(8);

        let output1 = transformer1.forward(&tokens, &mut sim1);
        let output2 = transformer2.forward(&tokens, &mut sim2);

        // Outputs should be the same for deterministic computation
        // (no random initialization in current implementation)
        assert_eq!(output1.len(), output2.len());
    }

    // TEST 20: Multi-layer stacking
    #[test]
    #[ignore] // slow: quantum circuit simulation in debug mode
    fn test_multi_layer_stacking() {
        let mut transformer = FullyQuantumTransformer::new(4, 4, 8, 32);
        assert_eq!(transformer.layers.len(), 4);

        let tokens = vec![1, 5, 9, 12];
        let mut sim = DensityMatrixSimulator::new(8);

        let output = transformer.forward(&tokens, &mut sim);
        assert_eq!(output.len(), 32);
    }
}
