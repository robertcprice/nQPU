//! Quantum Attention Mechanism for Quantum Transformers
//!
//! This module implements quantum attention mechanisms for potential quantum advantage
//! in transformer architectures. It provides a research platform for studying:
//!
//! - Quantum attention computation using density matrices
//! - Hybrid classical-quantum transformer layers
//! - Entanglement tracking during attention
//! - Small-scale quantum transformers for language tasks
//!
//! # Architecture Overview
//!
//! The implementation follows a hybrid approach:
//! - Quantum circuits compute attention weights
//! - Classical components handle embeddings and feed-forward layers
//! - Density matrices track mixed states and entanglement
//!
//! # Research Focus
//!
//! This is a RESEARCH implementation, not production-ready:
//! - Algorithm correctness over performance
//! - Study quantum advantages in attention
//! - Document limitations honestly
//! - Consider quantum-inspired approaches as fallback
//!
//! # Example Usage
//!
//! ```rust
//! use nqpu_metal::quantum_attention::{QuantumAttention, QuantumTransformer};
//! use nqpu_metal::density_matrix::DensityMatrixSimulator;
//!
//! // Create quantum attention
//! let mut attention = QuantumAttention::new(2, 4, 8);
//! let mut sim = DensityMatrixSimulator::new(8);
//!
//! // Compute attention
//! let queries = vec![0.1; 32];
//! let keys = vec![0.2; 32];
//! let values = vec![0.3; 32];
//! let output = attention.forward(&queries, &keys, &values, &mut sim);
//!
//! // Create small transformer
//! let mut transformer = QuantumTransformer::new(2, 2, 8, 64);
//! let tokens = vec![1, 5, 10, 15];
//! let output = transformer.forward(&tokens);
//! ```

use crate::density_matrix::DensityMatrixSimulator;
use num_complex::Complex64 as C64;
use rand::Rng;
use std::f64::consts::PI;

// ================================================================
// PART 1: QUANTUM ATTENTION PRIMITIVE
// ================================================================

/// Quantum attention mechanism using density matrix simulation
///
/// Implements quantum attention where Q, K, V are encoded into quantum states
/// and attention weights are computed via quantum interference.
///
/// # Quantum Circuit Design
///
/// The attention computation uses:
/// 1. State preparation: Encode Q, K, V into quantum states
/// 2. Quantum interference: Compute similarity via overlap
/// 3. Measurement: Extract attention weights
///
/// # Entanglement Tracking
///
/// Monitors entanglement entropy during attention computation to study
/// quantum correlations in the attention mechanism.
#[derive(Clone, Debug)]
pub struct QuantumAttention {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Maximum sequence length
    pub sequence_length: usize,
    /// Number of qubits per attention head
    pub num_qubits_per_head: usize,
    /// Total qubits used (log2 of sequence length * head_dim)
    pub total_qubits: usize,
    /// Entanglement entropy during last forward pass
    pub last_entanglement_entropy: f64,
    /// Attention weights from last computation (for analysis)
    pub last_attention_weights: Vec<Vec<f64>>,
}

impl QuantumAttention {
    /// Create a new quantum attention mechanism
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads (recommended: 2-4)
    /// * `head_dim` - Dimension of each head (recommended: 4-8)
    /// * `sequence_length` - Maximum sequence length (recommended: 8-16)
    ///
    /// # Qubit Calculation
    ///
    /// Total qubits = num_heads * ceil(log2(head_dim * sequence_length))
    /// For 2 heads, dim 4, seq 8: 2 * 5 = 10 qubits
    pub fn new(num_heads: usize, head_dim: usize, sequence_length: usize) -> Self {
        // Calculate qubits needed: need to represent head_dim * sequence_length states
        let states_per_head = head_dim * sequence_length;
        let num_qubits_per_head = (states_per_head as f64).log2().ceil() as usize;
        let total_qubits = num_heads * num_qubits_per_head;

        QuantumAttention {
            num_heads,
            head_dim,
            sequence_length,
            num_qubits_per_head,
            total_qubits,
            last_entanglement_entropy: 0.0,
            last_attention_weights: Vec::new(),
        }
    }

    /// Forward pass for quantum attention
    ///
    /// Computes attention using quantum circuits:
    /// 1. Encode queries, keys, values into quantum states
    /// 2. Compute attention weights via quantum interference
    /// 3. Apply weighted aggregation
    ///
    /// # Arguments
    ///
    /// * `queries` - Query tensor [sequence_length, head_dim]
    /// * `keys` - Key tensor [sequence_length, head_dim]
    /// * `values` - Value tensor [sequence_length, head_dim]
    /// * `sim` - Density matrix simulator
    ///
    /// # Returns
    ///
    /// Output tensor [sequence_length, head_dim]
    ///
    /// # Quantum Algorithm
    ///
    /// The attention computation uses quantum parallelism:
    /// - Encode all query-key pairs into superposition
    /// - Use quantum interference to compute similarities
    /// - Measure to get attention weights
    pub fn forward(
        &mut self,
        queries: &[f64],
        keys: &[f64],
        values: &[f64],
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<f64> {
        let head_dim = self.head_dim;

        // Infer actual sequence length from input
        let actual_seq_len = queries.len() / head_dim;

        // Validate input dimensions
        assert_eq!(
            queries.len(),
            actual_seq_len * head_dim,
            "Query dimension mismatch"
        );
        assert_eq!(
            keys.len(),
            actual_seq_len * head_dim,
            "Key dimension mismatch"
        );
        assert_eq!(
            values.len(),
            actual_seq_len * head_dim,
            "Value dimension mismatch"
        );

        let mut all_attention_weights = Vec::new();
        let mut total_entropy = 0.0;

        // Process each position in sequence
        for pos in 0..actual_seq_len {
            // Extract query for this position
            let query_start = pos * head_dim;
            let query = &queries[query_start..query_start + head_dim];

            // Compute attention weights for this query against all keys
            let attention_weights =
                self.compute_attention_weights(query, keys, actual_seq_len, head_dim, sim);

            all_attention_weights.push(attention_weights);
            total_entropy += sim.entropy();
        }

        // Update metrics
        self.last_attention_weights = all_attention_weights.clone();
        self.last_entanglement_entropy = total_entropy / actual_seq_len as f64;

        // Flatten output
        let mut result = Vec::new();
        for pos in 0..actual_seq_len {
            let output = self.apply_weights(
                &all_attention_weights[pos],
                values,
                actual_seq_len,
                head_dim,
            );
            result.extend(output);
        }

        result
    }

    /// Get attention weights from the last forward pass
    pub fn get_attention_weights(&self) -> &[Vec<f64>] {
        &self.last_attention_weights
    }

    /// Encode a vector into a quantum state using amplitude encoding
    ///
    /// # Algorithm
    ///
    /// Uses angle encoding to map classical data to quantum amplitudes:
    /// - Each element becomes a rotation angle
    /// - Apply rotations to create state
    ///
    /// # Arguments
    ///
    /// * `data` - Classical data vector [head_dim]
    /// * `sim` - Density matrix simulator
    fn encode_vector(&self, data: &[f64], sim: &mut DensityMatrixSimulator) {
        let num_qubits = sim.num_qubits();

        // Normalize data to [0, π] for angle encoding
        let max_val = data.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let scale = if max_val > 0.0 { PI / max_val } else { 1.0 };

        // Apply amplitude encoding via rotations
        for (i, &val) in data.iter().enumerate() {
            if i < num_qubits {
                let angle = val * scale;
                sim.ry(i, angle);
            }
        }

        // Create superposition for quantum parallelism
        for i in 0..num_qubits.min(data.len()) {
            sim.h(i);
        }
    }

    /// Compute attention weights via quantum circuit
    ///
    /// # Quantum Algorithm
    ///
    /// 1. Encode query and key into quantum states
    /// 2. Compute overlap via quantum interference
    /// 3. Measure to get attention score
    ///
    /// This uses the quantum property that:
    /// |⟨ψ_query|ψ_key⟩|² gives the similarity
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector [head_dim]
    /// * `keys` - All keys [sequence_length, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `sim` - Density matrix simulator
    ///
    /// # Returns
    ///
    /// Attention weights [sequence_length] (softmax normalized)
    fn compute_attention_weights(
        &self,
        query: &[f64],
        keys: &[f64],
        seq_len: usize,
        head_dim: usize,
        sim: &mut DensityMatrixSimulator,
    ) -> Vec<f64> {
        let mut scores = Vec::with_capacity(seq_len);

        // Compute query-key similarity for each key
        for key_pos in 0..seq_len {
            // Reset simulator
            sim.state.reset();

            // Encode query
            self.encode_vector(query, sim);

            // Store query state
            let query_state = sim.state.clone();

            // Reset and encode key
            sim.state.reset();
            let key_start = key_pos * head_dim;
            let key = &keys[key_start..key_start + head_dim];
            self.encode_vector(key, sim);

            // Compute overlap (fidelity between states)
            let key_state = sim.state.clone();
            let fidelity = Self::compute_fidelity(&query_state, &key_state);

            // Scale by sqrt(d_k) as in standard attention
            let scaled_score = fidelity / (head_dim as f64).sqrt();
            scores.push(scaled_score);
        }

        // Apply softmax
        Self::softmax(&scores)
    }

    /// Compute fidelity (overlap) between two density matrices
    ///
    /// Fidelity F(ρ, σ) = Tr(√(√ρ σ √ρ))²
    ///
    /// For pure states: F = |⟨ψ|φ⟩|²
    fn compute_fidelity(
        state1: &crate::density_matrix::DensityMatrix,
        state2: &crate::density_matrix::DensityMatrix,
    ) -> f64 {
        let mut overlap = C64::new(0.0, 0.0);
        let dim = state1.dim;

        // Compute Tr(ρ₁ σ₂)
        for i in 0..dim {
            for j in 0..dim {
                overlap += state1.elements[i * dim + j] * state2.elements[j * dim + i];
            }
        }

        // Fidelity is |Tr(ρ₁ σ₂)|² for pure states
        overlap.norm_sqr()
    }

    /// Apply attention weights to values
    ///
    /// # Arguments
    ///
    /// * `weights` - Attention weights [sequence_length]
    /// * `values` - Value tensor [sequence_length, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    ///
    /// # Returns
    ///
    /// Weighted sum [head_dim]
    fn apply_weights(
        &self,
        weights: &[f64],
        values: &[f64],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        let mut output = vec![0.0; head_dim];

        for pos in 0..seq_len {
            let val_start = pos * head_dim;
            let weight = weights[pos];

            for i in 0..head_dim {
                output[i] += weight * values[val_start + i];
            }
        }

        output
    }

    /// Softmax function
    fn softmax(scores: &[f64]) -> Vec<f64> {
        // Find max for numerical stability
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();

        // Normalize
        exp_scores.iter().map(|&e| e / sum).collect()
    }

    /// Get classical attention for comparison
    ///
    /// Computes standard scaled dot-product attention
    pub fn classical_attention(&self, queries: &[f64], keys: &[f64], values: &[f64]) -> Vec<f64> {
        let seq_len = self.sequence_length;
        let head_dim = self.head_dim;

        let mut result = Vec::new();

        for pos in 0..seq_len {
            let query_start = pos * head_dim;
            let query = &queries[query_start..query_start + head_dim];

            // Compute QK^T / sqrt(d_k)
            let mut scores = Vec::with_capacity(seq_len);
            for key_pos in 0..seq_len {
                let key_start = key_pos * head_dim;
                let key = &keys[key_start..key_start + head_dim];

                let dot_product: f64 = query.iter().zip(key.iter()).map(|(&q, &k)| q * k).sum();
                let scaled = dot_product / (head_dim as f64).sqrt();
                scores.push(scaled);
            }

            // Softmax
            let weights = Self::softmax(&scores);

            // Apply to values
            let output = self.apply_weights(&weights, values, seq_len, head_dim);
            result.extend(output);
        }

        result
    }

    /// Compare quantum vs classical attention
    ///
    /// Returns (quantum_output, classical_output, difference)
    pub fn compare_with_classical(
        &mut self,
        queries: &[f64],
        keys: &[f64],
        values: &[f64],
        sim: &mut DensityMatrixSimulator,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let quantum_output = self.forward(queries, keys, values, sim);
        let classical_output = self.classical_attention(queries, keys, values);

        // Compute mean absolute difference
        let diff: f64 = quantum_output
            .iter()
            .zip(classical_output.iter())
            .map(|(&q, &c)| (q - c).abs())
            .sum::<f64>()
            / quantum_output.len() as f64;

        (quantum_output, classical_output, diff)
    }
}

// ================================================================
// PART 2: HYBRID TRANSFORMER LAYER
// ================================================================

/// Hybrid classical-quantum transformer layer
///
/// Combines quantum attention with classical feed-forward networks.
/// This hybrid approach leverages quantum computing where it may provide
/// advantage (attention) while using classical components for the rest.
///
/// # Architecture
///
/// 1. Quantum Self-Attention
/// 2. Add & Norm
/// 3. Classical Feed-Forward Network
/// 4. Add & Norm
///
/// # Training
///
/// Uses hybrid optimization:
/// - Quantum parameters: parameter-shift rule
/// - Classical parameters: standard backpropagation
#[derive(Clone, Debug)]
pub struct HybridTransformerLayer {
    /// Quantum attention mechanism
    pub quantum_attention: QuantumAttention,
    /// Feed-forward hidden dimension
    pub feed_forward_dim: usize,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Layer normalization parameters
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    /// Feed-forward weights
    pub ff_weights1: Vec<Vec<f64>>,
    pub ff_weights2: Vec<Vec<f64>>,
    /// Feed-forward biases
    pub ff_bias1: Vec<f64>,
    pub ff_bias2: Vec<f64>,
    // Cached forward-pass intermediates for backward
    last_ff_input: Vec<f64>,
    last_hidden_pre: Vec<f64>,  // pre-activation (before GELU)
    last_hidden_post: Vec<f64>, // post-activation (after GELU)
}

impl HybridTransformerLayer {
    /// Create a new hybrid transformer layer
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each head
    /// * `sequence_length` - Maximum sequence length
    /// * `feed_forward_dim` - Hidden dimension for FFN (recommended: 4*head_dim)
    /// * `dropout_rate` - Dropout probability (0.0 to 1.0)
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        sequence_length: usize,
        feed_forward_dim: usize,
        dropout_rate: f64,
    ) -> Self {
        let model_dim = num_heads * head_dim;

        // Initialize layer norm parameters
        let gamma = vec![1.0; model_dim];
        let beta = vec![0.0; model_dim];

        // Initialize feed-forward weights (Xavier initialization)
        let std1 = (2.0 / (model_dim + feed_forward_dim) as f64).sqrt();
        let std2 = (2.0 / (feed_forward_dim + model_dim) as f64).sqrt();

        let mut rng = rand::thread_rng();

        let ff_weights1: Vec<Vec<f64>> = (0..model_dim)
            .map(|_| {
                (0..feed_forward_dim)
                    .map(|_| rng.gen::<f64>() * 2.0 * std1 - std1)
                    .collect()
            })
            .collect();

        let ff_weights2: Vec<Vec<f64>> = (0..feed_forward_dim)
            .map(|_| {
                (0..model_dim)
                    .map(|_| rng.gen::<f64>() * 2.0 * std2 - std2)
                    .collect()
            })
            .collect();

        let ff_bias1 = vec![0.0; feed_forward_dim];
        let ff_bias2 = vec![0.0; model_dim];

        let quantum_attention = QuantumAttention::new(num_heads, head_dim, sequence_length);

        HybridTransformerLayer {
            quantum_attention,
            feed_forward_dim,
            dropout_rate,
            gamma,
            beta,
            ff_weights1,
            ff_weights2,
            ff_bias1,
            ff_bias2,
            last_ff_input: Vec::new(),
            last_hidden_pre: Vec::new(),
            last_hidden_post: Vec::new(),
        }
    }

    /// Forward pass through the hybrid layer
    ///
    /// # Architecture
    ///
    /// Input -> Quantum Attention -> Add & Norm -> FFN -> Add & Norm -> Output
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [sequence_length, model_dim]
    /// * `sim` - Density matrix simulator
    ///
    /// # Returns
    ///
    /// Output tensor [sequence_length, model_dim]
    pub fn forward(&mut self, input: &[f64], sim: &mut DensityMatrixSimulator) -> Vec<f64> {
        let model_dim = self.gamma.len();
        let actual_seq_len = input.len() / model_dim; // Actual sequence length from input

        // Reshape input to [seq_len, num_heads, head_dim]
        let num_heads = self.quantum_attention.num_heads;
        let head_dim = self.quantum_attention.head_dim;

        // Split into heads and process
        let mut attention_output = vec![0.0; input.len()];

        for head in 0..num_heads {
            // Extract head's portion of input
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;

            let mut head_queries = Vec::new();
            let mut head_keys = Vec::new();
            let mut head_values = Vec::new();

            for pos in 0..actual_seq_len {
                let pos_start = pos * model_dim;
                let pos_end = pos_start + model_dim;

                if pos_end <= input.len() {
                    let token_vec = &input[pos_start..pos_end];

                    if head_end <= token_vec.len() {
                        head_queries.extend(&token_vec[head_start..head_end]);
                        head_keys.extend(&token_vec[head_start..head_end]);
                        head_values.extend(&token_vec[head_start..head_end]);
                    }
                }
            }

            // Compute quantum attention for this head
            let seq_len_for_attention = head_queries.len() / head_dim;
            if seq_len_for_attention > 0 && !head_queries.is_empty() {
                let head_output =
                    self.quantum_attention
                        .forward(&head_queries, &head_keys, &head_values, sim);

                // Merge back into attention output
                for pos in 0..actual_seq_len.min(head_output.len() / head_dim) {
                    let out_start = pos * model_dim + head_start;
                    let out_end = out_start + head_dim;
                    let head_start_out = pos * head_dim;
                    let head_end_out = head_start_out + head_dim;

                    if out_end <= attention_output.len() && head_end_out <= head_output.len() {
                        attention_output[out_start..out_end]
                            .copy_from_slice(&head_output[head_start_out..head_end_out]);
                    }
                }
            }
        }

        // Add & Norm (residual connection)
        let mut attn_with_residual = Vec::with_capacity(input.len());
        for (i, &attn) in attention_output.iter().enumerate() {
            attn_with_residual.push(attn + input[i]);
        }

        let normalized1 = self.layer_norm(&attn_with_residual);

        // Feed-forward network
        let ff_output = self.feed_forward(&normalized1);

        // Add & Norm
        let mut output = Vec::with_capacity(input.len());
        for (i, &ff) in ff_output.iter().enumerate() {
            output.push(ff + normalized1[i]);
        }

        self.layer_norm(&output)
    }

    /// Layer normalization
    ///
    /// Normalizes across the feature dimension:
    /// output = gamma * (x - mean) / sqrt(var + eps) + beta
    fn layer_norm(&self, x: &[f64]) -> Vec<f64> {
        let model_dim = self.gamma.len();
        let actual_seq_len = x.len() / model_dim;
        let eps = 1e-6;

        let mut normalized = Vec::with_capacity(x.len());

        for pos in 0..actual_seq_len {
            let start = pos * model_dim;
            let end = start + model_dim;

            if end <= x.len() {
                let slice = &x[start..end];

                // Compute mean and variance
                let mean: f64 = slice.iter().sum::<f64>() / model_dim as f64;
                let variance: f64 =
                    slice.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / model_dim as f64;
                let std = (variance + eps).sqrt();

                // Normalize
                for (i, &val) in slice.iter().enumerate() {
                    let norm_val = (val - mean) / std;
                    normalized.push(self.gamma[i] * norm_val + self.beta[i]);
                }
            }
        }

        normalized
    }

    /// Feed-forward network
    ///
    /// FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
    fn feed_forward(&mut self, x: &[f64]) -> Vec<f64> {
        let model_dim = self.gamma.len();
        let actual_seq_len = x.len() / model_dim;
        let hidden_dim = self.feed_forward_dim;

        // Cache input for backward pass
        self.last_ff_input = x.to_vec();
        self.last_hidden_pre = Vec::with_capacity(actual_seq_len * hidden_dim);
        self.last_hidden_post = Vec::with_capacity(actual_seq_len * hidden_dim);

        let mut output = vec![0.0; x.len()];

        for pos in 0..actual_seq_len {
            let start = pos * model_dim;
            let end = start + model_dim;

            if end <= x.len() {
                let input = &x[start..end];

                // First linear layer
                let mut hidden_pre = vec![0.0; hidden_dim];
                let mut hidden_post = vec![0.0; hidden_dim];
                for j in 0..hidden_dim {
                    for i in 0..model_dim {
                        hidden_pre[j] += input[i] * self.ff_weights1[i][j];
                    }
                    hidden_pre[j] += self.ff_bias1[j];
                    // GELU activation
                    hidden_post[j] = hidden_pre[j]
                        * 0.5
                        * (1.0
                            + (hidden_pre[j] * 0.7978845608 + (hidden_pre[j] * 0.044715).powi(3))
                                .tanh());
                }

                // Cache pre/post activation for backward
                self.last_hidden_pre.extend_from_slice(&hidden_pre);
                self.last_hidden_post.extend_from_slice(&hidden_post);

                // Second linear layer
                let out_start = pos * model_dim;
                for i in 0..model_dim {
                    for j in 0..hidden_dim {
                        output[out_start + i] += hidden_post[j] * self.ff_weights2[j][i];
                    }
                    output[out_start + i] += self.ff_bias2[i];
                }
            }
        }

        output
    }

    /// Backward pass with proper gradient computation for feed-forward weights.
    ///
    /// Computes gradients for ff_weights1, ff_weights2, ff_bias1, ff_bias2 using
    /// cached forward-pass intermediates (GELU pre/post activations and input).
    /// Updates weights by learning_rate * gradient. Returns gradient for previous layer.
    ///
    /// # Note
    ///
    /// Quantum attention gradients would require parameter-shift rule with a simulator,
    /// which is not passed to backward(). Classical FF gradients are computed exactly.
    pub fn backward(&mut self, grad: &[f64], learning_rate: f64) -> Vec<f64> {
        let model_dim = self.gamma.len();
        let hidden_dim = self.feed_forward_dim;
        let actual_seq_len = grad.len() / model_dim;

        if self.last_ff_input.is_empty() || self.last_hidden_post.is_empty() {
            // No cached forward pass, fall back to identity gradient
            return grad.to_vec();
        }

        let mut input_grad = vec![0.0; grad.len()];

        for pos in 0..actual_seq_len {
            let g_start = pos * model_dim;
            let g_end = g_start + model_dim;
            let h_start = pos * hidden_dim;
            let h_end = h_start + hidden_dim;

            if g_end > grad.len() || h_end > self.last_hidden_post.len() {
                break;
            }

            let grad_slice = &grad[g_start..g_end];
            let ff_input = &self.last_ff_input[g_start..g_end.min(self.last_ff_input.len())];
            let hidden_pre = &self.last_hidden_pre[h_start..h_end];
            let hidden_post = &self.last_hidden_post[h_start..h_end];

            if ff_input.len() < model_dim {
                break;
            }

            // Backprop through second linear layer: output = W2 * hidden_post + b2
            // d_hidden_post = grad @ W2^T
            let mut d_hidden_post = vec![0.0; hidden_dim];
            for j in 0..hidden_dim {
                for i in 0..model_dim {
                    d_hidden_post[j] += grad_slice[i] * self.ff_weights2[j][i];
                }
            }
            // d_W2 -= lr * hidden_post^T @ grad
            for j in 0..hidden_dim {
                for i in 0..model_dim {
                    self.ff_weights2[j][i] -= learning_rate * hidden_post[j] * grad_slice[i];
                }
            }
            // d_b2 -= lr * grad
            for i in 0..model_dim {
                self.ff_bias2[i] -= learning_rate * grad_slice[i];
            }

            // Backprop through GELU: d_hidden_pre = d_hidden_post * gelu'(hidden_pre)
            let mut d_hidden_pre = vec![0.0; hidden_dim];
            for j in 0..hidden_dim {
                let x = hidden_pre[j];
                // GELU'(x) ≈ 0.5 * (1 + tanh(kx)) + 0.5 * x * sech²(kx) * k
                // where k = 0.7978845608 (sqrt(2/π)) and the cubic approximation
                let k = 0.7978845608;
                let inner = k * x + 0.044715 * x * x * x;
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;
                let d_inner = k + 3.0 * 0.044715 * x * x;
                let gelu_grad = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner;
                d_hidden_pre[j] = d_hidden_post[j] * gelu_grad;
            }

            // Backprop through first linear layer: hidden_pre = W1 * input + b1
            // d_input = d_hidden_pre @ W1^T
            for i in 0..model_dim {
                for j in 0..hidden_dim {
                    input_grad[g_start + i] += d_hidden_pre[j] * self.ff_weights1[i][j];
                }
            }
            // d_W1 -= lr * input^T @ d_hidden_pre
            for i in 0..model_dim {
                for j in 0..hidden_dim {
                    self.ff_weights1[i][j] -= learning_rate * ff_input[i] * d_hidden_pre[j];
                }
            }
            // d_b1 -= lr * d_hidden_pre
            for j in 0..hidden_dim {
                self.ff_bias1[j] -= learning_rate * d_hidden_pre[j];
            }
        }

        input_grad
    }

    /// Get entanglement entropy from last forward pass
    pub fn get_entanglement_entropy(&self) -> f64 {
        self.quantum_attention.last_entanglement_entropy
    }

    /// Get attention weights from last forward pass
    pub fn get_attention_weights(&self) -> &[Vec<f64>] {
        &self.quantum_attention.last_attention_weights
    }
}

// ================================================================
// PART 3: QUANTUM TRANSFORMER
// ================================================================

/// Small-scale quantum transformer for research
///
/// A complete transformer model using quantum attention mechanisms.
/// Designed for small-scale experiments to study quantum advantages.
///
/// # Scale Recommendations
///
/// - 2-4 layers
/// - 2 attention heads
/// - Sequence length: 8-16
/// - Vocabulary: 64-256 tokens
/// - ~20-25 qubits total
///
/// # Use Cases
///
/// 1. Proof-of-concept for quantum attention
/// 2. Research platform for entanglement in attention
/// 3. Algorithm development for quantum-efficient attention
/// 4. Benchmarking vs classical transformers
#[derive(Clone, Debug)]
pub struct QuantumTransformer {
    /// Transformer layers
    pub layers: Vec<HybridTransformerLayer>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Token embeddings
    pub token_embeddings: Vec<Vec<f64>>,
    /// Position embeddings
    pub position_embeddings: Vec<Vec<f64>>,
}

impl QuantumTransformer {
    /// Create a new quantum transformer
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers (recommended: 2)
    /// * `num_heads` - Number of attention heads (recommended: 2)
    /// * `max_seq_len` - Maximum sequence length (recommended: 8-16)
    /// * `vocab_size` - Vocabulary size (recommended: 64-256)
    ///
    /// # Qubit Requirements
    ///
    /// Total qubits = num_layers * num_heads * ceil(log2(head_dim * seq_len))
    /// For 2 layers, 2 heads, dim 4, seq 8: 2 * 2 * 5 = 20 qubits
    pub fn new(num_layers: usize, num_heads: usize, max_seq_len: usize, vocab_size: usize) -> Self {
        let head_dim = 4; // Small for simulation
        let embedding_dim = num_heads * head_dim;
        let feed_forward_dim = embedding_dim * 4;

        // Create layers
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(HybridTransformerLayer::new(
                num_heads,
                head_dim,
                max_seq_len,
                feed_forward_dim,
                0.1, // dropout_rate
            ));
        }

        // Initialize embeddings (random initialization)
        let mut rng = rand::thread_rng();
        let std = 1.0 / (embedding_dim as f64).sqrt();

        let token_embeddings: Vec<Vec<f64>> = (0..vocab_size)
            .map(|_| (0..embedding_dim).map(|_| rng.gen::<f64>() * std).collect())
            .collect();

        let position_embeddings: Vec<Vec<f64>> = (0..max_seq_len)
            .map(|_| (0..embedding_dim).map(|_| rng.gen::<f64>() * std).collect())
            .collect();

        QuantumTransformer {
            layers,
            vocab_size,
            max_seq_len,
            embedding_dim,
            token_embeddings,
            position_embeddings,
        }
    }

    /// Forward pass through the transformer
    ///
    /// # Pipeline
    ///
    /// 1. Token embedding lookup
    /// 2. Position embedding addition
    /// 3. Transformer layers
    /// 4. Output projection
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input token IDs [sequence_length]
    /// * `sim` - Density matrix simulator
    ///
    /// # Returns
    ///
    /// Output logits [sequence_length, vocab_size]
    pub fn forward(&mut self, tokens: &[usize], sim: &mut DensityMatrixSimulator) -> Vec<f64> {
        let seq_len = tokens.len();

        // Embed tokens
        let mut embedded = vec![0.0; seq_len * self.embedding_dim];
        for (pos, &token_id) in tokens.iter().enumerate() {
            if token_id < self.vocab_size {
                let token_emb = &self.token_embeddings[token_id];
                let pos_emb = &self.position_embeddings[pos];

                let start = pos * self.embedding_dim;
                for i in 0..self.embedding_dim {
                    embedded[start + i] = token_emb[i] + pos_emb[i];
                }
            }
        }

        // Pass through transformer layers
        let mut hidden = embedded;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, sim);
        }

        // Project to vocabulary (simple linear projection)
        // In practice, would use a proper linear layer
        let mut logits = vec![0.0; seq_len * self.vocab_size];

        for pos in 0..seq_len {
            let start = pos * self.embedding_dim;
            let end = start + self.embedding_dim;
            let hidden_vec = &hidden[start..end];

            // Simple dot product with token embeddings as projection
            for token_id in 0..self.vocab_size {
                let logit_start = pos * self.vocab_size + token_id;
                let token_emb = &self.token_embeddings[token_id];

                logits[logit_start] = hidden_vec
                    .iter()
                    .zip(token_emb.iter())
                    .map(|(&h, &e)| h * e)
                    .sum();
            }
        }

        logits
    }

    /// Train on a simple language task
    ///
    /// # Training Loop
    ///
    /// For each epoch:
    /// 1. Forward pass
    /// 2. Compute loss (cross-entropy)
    /// 3. Backward pass
    /// 4. Update parameters
    ///
    /// # Arguments
    ///
    /// * `data` - Training data: [(input_tokens, target_token), ...]
    /// * `epochs` - Number of training epochs
    /// * `learning_rate` - Learning rate for optimization
    ///
    /// # Returns
    ///
    /// Training loss history
    pub fn train(
        &mut self,
        data: &[(Vec<usize>, usize)],
        epochs: usize,
        learning_rate: f64,
    ) -> Vec<f64> {
        let mut loss_history = Vec::new();
        let mut sim = DensityMatrixSimulator::new(self.layers[0].quantum_attention.total_qubits);

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input_tokens, target_token) in data {
                // Forward pass
                let logits = self.forward(input_tokens, &mut sim);

                // Compute cross-entropy loss
                let seq_len = input_tokens.len();
                let last_pos = seq_len - 1;
                let logit_start = last_pos * self.vocab_size;

                // Softmax over logits
                let mut logits_slice = Vec::new();
                for token_id in 0..self.vocab_size {
                    logits_slice.push(logits[logit_start + token_id]);
                }

                let probs = QuantumAttention::softmax(&logits_slice);
                let loss = -probs[*target_token].ln().max(-10.0); // Clamp for stability

                total_loss += loss;

                // Compute gradient: softmax - one_hot(target)
                let mut grad = vec![0.0; logits.len()];
                for token_id in 0..self.vocab_size {
                    let idx = logit_start + token_id;
                    if idx < grad.len() {
                        grad[idx] = probs[token_id];
                        if token_id == *target_token {
                            grad[idx] -= 1.0;
                        }
                    }
                }
                let _ = self.backward(&grad, learning_rate);
            }

            let avg_loss = total_loss / data.len() as f64;
            loss_history.push(avg_loss);

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, avg_loss);
            }
        }

        loss_history
    }

    /// Backward pass through all transformer layers.
    ///
    /// Chains gradients through layers in reverse order, updating FF weights
    /// via cached forward-pass intermediates. Quantum attention gradients
    /// require parameter-shift with a simulator (not available here).
    fn backward(&mut self, grad: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut current_grad = grad.to_vec();

        // Backprop through layers (reverse order)
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad, learning_rate);
        }

        current_grad
    }

    /// Generate text token by token
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input prompt tokens
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy, higher = more random)
    ///
    /// # Returns
    ///
    /// Generated tokens
    pub fn generate(
        &mut self,
        prompt: &[usize],
        max_tokens: usize,
        temperature: f64,
    ) -> Vec<usize> {
        let mut tokens = prompt.to_vec();
        let mut sim = DensityMatrixSimulator::new(self.layers[0].quantum_attention.total_qubits);
        let mut rng = rand::thread_rng();

        for _ in 0..max_tokens {
            // Forward pass
            let logits = self.forward(&tokens, &mut sim);

            // Get last position logits
            let seq_len = tokens.len();
            let last_pos = seq_len - 1;
            let logit_start = last_pos * self.vocab_size;

            // Apply temperature and sample
            let mut logits_slice = Vec::new();
            for token_id in 0..self.vocab_size {
                let logit = logits[logit_start + token_id];
                logits_slice.push(logit / temperature);
            }

            let probs = QuantumAttention::softmax(&logits_slice);

            // Sample next token
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut next_token = 0;

            for (token_id, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if r <= cumsum {
                    next_token = token_id;
                    break;
                }
            }

            tokens.push(next_token);

            // Stop if we generate an EOS token (assumed to be 0)
            if next_token == 0 {
                break;
            }
        }

        tokens
    }

    /// Get average entanglement across all layers
    pub fn get_average_entanglement(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }

        let total: f64 = self
            .layers
            .iter()
            .map(|l| l.quantum_attention.last_entanglement_entropy)
            .sum();

        total / self.layers.len() as f64
    }

    /// Get model statistics
    pub fn get_stats(&self) -> TransformerStats {
        let total_params = self.count_parameters();
        let qubits_per_layer = self.layers[0].quantum_attention.total_qubits;
        let total_qubits = qubits_per_layer * self.layers.len();

        TransformerStats {
            num_layers: self.layers.len(),
            num_heads: self.layers[0].quantum_attention.num_heads,
            head_dim: self.layers[0].quantum_attention.head_dim,
            max_seq_len: self.max_seq_len,
            vocab_size: self.vocab_size,
            embedding_dim: self.embedding_dim,
            total_parameters: total_params,
            total_qubits,
            qubits_per_layer,
        }
    }

    /// Count total parameters
    fn count_parameters(&self) -> usize {
        let mut count = 0;

        // Embeddings
        count += self.token_embeddings.len() * self.token_embeddings[0].len();
        count += self.position_embeddings.len() * self.position_embeddings[0].len();

        // Layers
        for layer in &self.layers {
            // Layer norm
            count += layer.gamma.len() * 2;

            // Feed-forward
            count += layer.ff_weights1.len() * layer.ff_weights1[0].len();
            count += layer.ff_weights2.len() * layer.ff_weights2[0].len();
            count += layer.ff_bias1.len() + layer.ff_bias2.len();
        }

        count
    }
}

/// Transformer statistics
#[derive(Clone, Debug)]
pub struct TransformerStats {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub total_parameters: usize,
    pub total_qubits: usize,
    pub qubits_per_layer: usize,
}

// ================================================================
// TESTS
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_attention_creation() {
        let attention = QuantumAttention::new(2, 4, 8);
        assert_eq!(attention.num_heads, 2);
        assert_eq!(attention.head_dim, 4);
        assert_eq!(attention.sequence_length, 8);
    }

    #[test]
    fn test_quantum_attention_forward() {
        let mut attention = QuantumAttention::new(1, 2, 4); // Smaller for faster test
        let mut sim = DensityMatrixSimulator::new(4);

        let queries = vec![0.1; 8]; // [4, 2]
        let keys = vec![0.2; 8]; // [4, 2]
        let values = vec![0.3; 8]; // [4, 2]

        let output = attention.forward(&queries, &keys, &values, &mut sim);
        assert_eq!(output.len(), 8);

        // Check that output is not all zeros
        let sum: f64 = output.iter().sum();
        assert!(sum.abs() > 1e-10);
    }

    #[test]
    fn test_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = QuantumAttention::softmax(&scores);

        // Check probabilities sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check ordering preserved
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_hybrid_layer_creation() {
        let layer = HybridTransformerLayer::new(2, 4, 8, 16, 0.1);
        assert_eq!(layer.quantum_attention.num_heads, 2);
        assert_eq!(layer.feed_forward_dim, 16);
    }

    #[test]
    fn test_hybrid_layer_forward() {
        let mut layer = HybridTransformerLayer::new(1, 2, 4, 8, 0.1); // Smaller for faster test
        let mut sim = DensityMatrixSimulator::new(4);

        let input = vec![0.1; 8]; // [4, 2]
        let output = layer.forward(&input, &mut sim);

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_quantum_transformer_creation() {
        let transformer = QuantumTransformer::new(2, 2, 8, 64);
        assert_eq!(transformer.layers.len(), 2);
        assert_eq!(transformer.vocab_size, 64);
        assert_eq!(transformer.max_seq_len, 8);
    }

    #[test]
    fn test_quantum_transformer_forward() {
        let mut transformer = QuantumTransformer::new(1, 1, 2, 8); // Very small: seq_len=2, vocab=8
        let mut sim = DensityMatrixSimulator::new(4);

        let tokens = vec![1];
        let output = transformer.forward(&tokens, &mut sim);

        assert_eq!(output.len(), 1 * 8); // [1 position, 8 vocab]
    }

    #[test]
    fn test_quantum_vs_classical_attention() {
        let mut attention = QuantumAttention::new(1, 2, 4); // Smaller for faster test
        let mut sim = DensityMatrixSimulator::new(4);

        let queries = vec![0.1; 8]; // [4, 2]
        let keys = vec![0.2; 8]; // [4, 2]
        let values = vec![0.3; 8]; // [4, 2]

        let (quantum, classical, diff) =
            attention.compare_with_classical(&queries, &keys, &values, &mut sim);

        // Check outputs have same shape
        assert_eq!(quantum.len(), classical.len());

        // Difference should be reasonable (quantum may differ from classical)
        assert!(diff < 10.0); // Should not be wildly different
    }

    #[test]
    fn test_entanglement_tracking() {
        let mut attention = QuantumAttention::new(1, 2, 4); // Smaller for faster test
        let mut sim = DensityMatrixSimulator::new(4);

        let queries = vec![0.1; 8]; // [4, 2]
        let keys = vec![0.2; 8]; // [4, 2]
        let values = vec![0.3; 8]; // [4, 2]

        attention.forward(&queries, &keys, &values, &mut sim);

        // Entanglement should be non-negative
        assert!(attention.last_entanglement_entropy >= 0.0);
    }

    #[test]
    fn test_transformer_stats() {
        let transformer = QuantumTransformer::new(2, 2, 8, 64);
        let stats = transformer.get_stats();

        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.vocab_size, 64);
        assert!(stats.total_parameters > 0);
        assert!(stats.total_qubits > 0);
    }

    #[test]
    fn test_simple_training() {
        let mut transformer = QuantumTransformer::new(1, 1, 2, 4); // Very small: seq_len=2, vocab=4

        // Create simple synthetic data (single token inputs)
        let data = vec![(vec![1], 2), (vec![2], 3)];

        let loss_history = transformer.train(&data, 2, 0.01); // Only 2 epochs

        // Loss should decrease (at least not increase dramatically)
        assert_eq!(loss_history.len(), 2);

        // Last loss should not be higher than first by too much
        // (this is a very weak test, just checking nothing explodes)
        let first_loss = loss_history[0];
        let last_loss = loss_history[1];
        assert!(last_loss < first_loss * 2.0);
    }

    #[test]
    fn test_generation() {
        let mut transformer = QuantumTransformer::new(1, 1, 2, 4); // Very small: seq_len=2, vocab=4

        let prompt = vec![1];
        let generated = transformer.generate(&prompt, 1, 1.0); // Only 1 token

        // Should generate at least the prompt plus some tokens
        assert!(generated.len() >= prompt.len());

        // First token should match prompt
        assert_eq!(generated[0], prompt[0]);
    }
}
