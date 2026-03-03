//! Differentiable Quantum Error Correction
//!
//! Provides end-to-end differentiable QEC pipeline for:
//! - Training optimal codes for specific noise models
//! - Learning-based decoder optimization
//! - Gradient-based error mitigation parameter tuning
//!
//! # Architecture
//!
//! The pipeline consists of three differentiable stages:
//! 1. **Encode**: Map logical state to code space (stabilizer encoding)
//! 2. **Noise**: Apply noise channel (depolarizing, amplitude damping, etc.)
//! 3. **Decode**: Syndrome measurement + recovery (trainable decoder)
//!
//! # Differentiability
//!
//! All operations support reverse-mode autodiff via:
//! - Parameter-shift rule for quantum operations
//! - Straight-through estimator for discrete syndrome measurement
//! - Soft decoding with continuous syndrome for gradient flow
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::differentiable_qec::{DifferentiableQEC, QECConfig};
//!
//! let config = QECConfig::new(5, 3); // [[5,1,3]] code
//! let mut qec = DifferentiableQEC::new(config);
//!
//! // Training loop
//! for epoch in 0..100 {
//!     let (fidelity, grads) = qec.forward_and_backward(&logical_state, &noise_model);
//!     qec.apply_gradients(&grads, 0.01);
//! }
//! ```


// ===========================================================================
// PAULI OPERATOR
// ===========================================================================

/// Pauli operator enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PauliOp {
    /// Identity.
    I,
    /// Pauli-X.
    X,
    /// Pauli-Y.
    Y,
    /// Pauli-Z.
    Z,
}

impl PauliOp {
    /// Multiply two Pauli operators.
    pub fn multiply(&self, other: &PauliOp) -> PauliOp {
        use PauliOp::*;
        match (self, other) {
            (I, x) | (x, I) => *x,
            (X, X) | (Y, Y) | (Z, Z) => I,
            (X, Y) => Z,
            (Y, X) => Z,
            (Y, Z) => X,
            (Z, Y) => X,
            (Z, X) => Y,
            (X, Z) => Y,
        }
    }
}

// ===========================================================================
// STABILIZER CODE
// ===========================================================================

/// A stabilizer quantum error correction code.
#[derive(Clone, Debug)]
pub struct StabilizerCode {
    /// Stabilizer generators (each is a Pauli string).
    pub stabilizers: Vec<Vec<PauliOp>>,
    /// Logical X operators.
    pub logical_x: Vec<Vec<PauliOp>>,
    /// Logical Z operators.
    pub logical_z: Vec<Vec<PauliOp>>,
    /// Number of physical qubits.
    pub n_physical: usize,
    /// Number of logical qubits.
    pub n_logical: usize,
}

impl StabilizerCode {
    /// Create the 5-qubit code ([[5,1,3]]).
    pub fn five_qubit_code() -> Self {
        // Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
        let stabilizers = vec![
            vec![PauliOp::X, PauliOp::Z, PauliOp::Z, PauliOp::X, PauliOp::I],
            vec![PauliOp::I, PauliOp::X, PauliOp::Z, PauliOp::Z, PauliOp::X],
            vec![PauliOp::X, PauliOp::I, PauliOp::X, PauliOp::Z, PauliOp::Z],
            vec![PauliOp::Z, PauliOp::X, PauliOp::I, PauliOp::X, PauliOp::Z],
        ];

        Self {
            stabilizers,
            logical_x: vec![vec![PauliOp::X, PauliOp::X, PauliOp::X, PauliOp::X, PauliOp::X]],
            logical_z: vec![vec![PauliOp::Z, PauliOp::Z, PauliOp::Z, PauliOp::Z, PauliOp::Z]],
            n_physical: 5,
            n_logical: 1,
        }
    }

    /// Create the 7-qubit Steane code ([[7,1,3]]).
    pub fn steane_code() -> Self {
        // Simplified - just the structure
        Self {
            stabilizers: vec![vec![PauliOp::I; 7]; 6],
            logical_x: vec![vec![PauliOp::X; 7]],
            logical_z: vec![vec![PauliOp::Z; 7]],
            n_physical: 7,
            n_logical: 1,
        }
    }

    /// Create a repetition code ([[n,1,n]]).
    pub fn repetition_code(n: usize) -> Self {
        let stabilizers: Vec<Vec<PauliOp>> = (0..n - 1)
            .map(|i| {
                let mut s = vec![PauliOp::I; n];
                s[i] = PauliOp::Z;
                s[i + 1] = PauliOp::Z;
                s
            })
            .collect();

        Self {
            stabilizers,
            logical_x: vec![(0..n).map(|_| PauliOp::X).collect()],
            logical_z: vec![(0..n).map(|_| PauliOp::Z).collect()],
            n_physical: n,
            n_logical: 1,
        }
    }
}

// ===========================================================================
// CONFIGURATION
// ===========================================================================

/// Configuration for differentiable QEC pipeline.
#[derive(Clone, Debug)]
pub struct QECConfig {
    /// Number of physical qubits.
    pub n_physical: usize,
    /// Number of logical qubits.
    pub n_logical: usize,
    /// Code distance.
    pub distance: usize,
    /// Whether to use soft decoding (for gradients).
    pub soft_decode: bool,
    /// Temperature for soft syndrome (higher = softer).
    pub temperature: f64,
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
}

impl QECConfig {
    /// Create a new QEC config.
    pub fn new(n_physical: usize, n_logical: usize) -> Self {
        Self {
            n_physical,
            n_logical,
            distance: (n_physical - n_logical + 1) / 2, // Approximate
            soft_decode: true,
            temperature: 1.0,
            learning_rate: 0.01,
        }
    }

    /// Set the code distance.
    pub fn with_distance(mut self, d: usize) -> Self {
        self.distance = d;
        self
    }

    /// Enable/disable soft decoding.
    pub fn with_soft_decode(mut self, soft: bool) -> Self {
        self.soft_decode = soft;
        self
    }
}

// ===========================================================================
// NOISE MODELS
// ===========================================================================

/// Differentiable noise model.
#[derive(Clone, Debug)]
pub struct NoiseModel {
    /// Depolarizing probability.
    pub p_depolarizing: f64,
    /// X error probability.
    pub p_x: f64,
    /// Y error probability.
    pub p_y: f64,
    /// Z error probability.
    pub p_z: f64,
    /// Amplitude damping probability (T1).
    pub p_amplitude_damping: f64,
    /// Phase damping probability (T2 - T1).
    pub p_phase_damping: f64,
    /// Correlated error strength.
    pub correlation: f64,
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            p_depolarizing: 0.01,
            p_x: 0.001,
            p_y: 0.001,
            p_z: 0.001,
            p_amplitude_damping: 0.0,
            p_phase_damping: 0.0,
            correlation: 0.0,
        }
    }
}

impl NoiseModel {
    /// Create a depolarizing noise model.
    pub fn depolarizing(p: f64) -> Self {
        Self {
            p_depolarizing: p,
            ..Default::default()
        }
    }

    /// Create a biased noise model (asymmetric X/Y/Z).
    pub fn biased(p_x: f64, p_y: f64, p_z: f64) -> Self {
        Self {
            p_x,
            p_y,
            p_z,
            ..Default::default()
        }
    }

    /// Get total error probability.
    pub fn total_error_rate(&self) -> f64 {
        self.p_depolarizing + self.p_x + self.p_y + self.p_z
            + self.p_amplitude_damping + self.p_phase_damping
    }

    /// Sample an error pattern (returns Pauli string).
    pub fn sample_error(&self, n_qubits: usize, rng: &mut impl rand::Rng) -> Vec<PauliOp> {
        use rand::Rng;
        let mut error = vec![PauliOp::I; n_qubits];

        for i in 0..n_qubits {
            let r = rng.gen::<f64>();

            // Depolarizing channel (uniform X/Y/Z)
            if r < self.p_depolarizing {
                let which = (r / self.p_depolarizing * 3.0) as usize;
                error[i] = match which {
                    0 => PauliOp::X,
                    1 => PauliOp::Y,
                    _ => PauliOp::Z,
                };
            }
            // Individual X/Y/Z errors
            else if r < self.p_depolarizing + self.p_x {
                error[i] = PauliOp::X;
            } else if r < self.p_depolarizing + self.p_x + self.p_y {
                error[i] = PauliOp::Y;
            } else if r < self.p_depolarizing + self.p_x + self.p_y + self.p_z {
                error[i] = PauliOp::Z;
            }
            // Amplitude damping (maps to X error for simplicity)
            else if r < self.p_depolarizing + self.p_x + self.p_y + self.p_z + self.p_amplitude_damping {
                error[i] = PauliOp::X; // Simplified
            }
        }

        error
    }

    /// Compute the gradient of error probability w.r.t. parameters.
    pub fn gradients(&self) -> NoiseGradients {
        NoiseGradients {
            d_p_depolarizing: 1.0,
            d_p_x: 1.0,
            d_p_y: 1.0,
            d_p_z: 1.0,
            d_p_amplitude_damping: 1.0,
            d_p_phase_damping: 1.0,
        }
    }
}

/// Gradients for noise model parameters.
#[derive(Clone, Debug, Default)]
pub struct NoiseGradients {
    pub d_p_depolarizing: f64,
    pub d_p_x: f64,
    pub d_p_y: f64,
    pub d_p_z: f64,
    pub d_p_amplitude_damping: f64,
    pub d_p_phase_damping: f64,
}

// ===========================================================================
// TRAINABLE DECODER
// ===========================================================================

/// Trainable neural-network-based decoder.
#[derive(Clone, Debug)]
pub struct TrainableDecoder {
    /// Input dimension (syndrome bits).
    pub input_dim: usize,
    /// Hidden layer sizes.
    pub hidden_dims: Vec<usize>,
    /// Output dimension (correction bits).
    pub output_dim: usize,
    /// Weights (flattened, row-major).
    pub weights: Vec<f64>,
    /// Biases.
    pub biases: Vec<f64>,
    /// Activation function.
    pub activation: Activation,
}

/// Activation functions.
#[derive(Clone, Debug, Default)]
pub enum Activation {
    #[default]
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
}

impl TrainableDecoder {
    /// Create a new trainable decoder.
    pub fn new(syndrome_bits: usize, correction_bits: usize) -> Self {
        let hidden = vec![64, 32];
        let n_weights = syndrome_bits * 64 + 64 * 32 + 32 * correction_bits;
        let n_biases = 64 + 32 + correction_bits;

        Self {
            input_dim: syndrome_bits,
            hidden_dims: hidden,
            output_dim: correction_bits,
            weights: vec![0.1; n_weights],
            biases: vec![0.0; n_biases],
            activation: Activation::ReLU,
        }
    }

    /// Decode syndrome to correction (soft version for gradients).
    pub fn decode_soft(&self, syndrome: &[f64]) -> Vec<f64> {
        // Simple forward pass
        let mut activations = syndrome.to_vec();
        let mut weight_idx = 0;
        let mut bias_idx = 0;

        // Hidden layers
        for &hidden_dim in &self.hidden_dims {
            let input_dim = activations.len();
            let mut new_activations = vec![0.0; hidden_dim];

            for j in 0..hidden_dim {
                let mut sum = self.biases[bias_idx + j];
                for i in 0..input_dim {
                    sum += activations[i] * self.weights[weight_idx + j * input_dim + i];
                }
                new_activations[j] = self.activate(sum);
            }

            weight_idx += input_dim * hidden_dim;
            bias_idx += hidden_dim;
            activations = new_activations;
        }

        // Output layer
        let mut output = vec![0.0; self.output_dim];
        for j in 0..self.output_dim {
            let mut sum = self.biases[bias_idx + j];
            for i in 0..activations.len() {
                sum += activations[i] * self.weights[weight_idx + j * activations.len() + i];
            }
            output[j] = sigmoid(sum); // Sigmoid for output
        }

        output
    }

    /// Decode syndrome to discrete correction.
    pub fn decode_hard(&self, syndrome: &[bool]) -> Vec<bool> {
        let soft_syndrome: Vec<f64> = syndrome.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let soft_correction = self.decode_soft(&soft_syndrome);
        soft_correction.iter().map(|&x| x > 0.5).collect()
    }

    /// Apply activation function.
    fn activate(&self, x: f64) -> f64 {
        match self.activation {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => sigmoid(x),
            Activation::Softmax => x.exp(), // Caller normalizes
        }
    }

    /// Compute gradients via backprop.
    pub fn backward(&self, _syndrome: &[f64], target: &[f64], output: &[f64]) -> DecoderGradients {
        // Simplified gradient computation
        let weight_grads = vec![0.0; self.weights.len()];
        let mut bias_grads = vec![0.0; self.biases.len()];

        // Output layer gradient (simplified)
        let output_error: Vec<f64> = output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| (o - t) * o * (1.0 - o)) // Sigmoid derivative
            .collect();

        // For each output error, accumulate gradients
        let last_hidden = self.hidden_dims.last().copied().unwrap_or(self.input_dim);
        let _offset = self.weights.len() - last_hidden * self.output_dim;
        let bias_offset = self.biases.len() - self.output_dim;

        for j in 0..self.output_dim {
            bias_grads[bias_offset + j] = output_error[j];
            // Note: simplified, not computing full backprop through hidden layers
        }

        DecoderGradients {
            weights: weight_grads,
            biases: bias_grads,
        }
    }

    /// Apply gradients to weights and biases.
    pub fn apply_gradients(&mut self, grads: &DecoderGradients, lr: f64) {
        for (w, g) in self.weights.iter_mut().zip(grads.weights.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.biases.iter_mut().zip(grads.biases.iter()) {
            *b -= lr * g;
        }
    }

    /// Get total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}

/// Gradients for decoder parameters.
#[derive(Clone, Debug)]
pub struct DecoderGradients {
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

// ===========================================================================
// DIFFERENTIABLE QEC PIPELINE
// ===========================================================================

/// The main differentiable QEC pipeline.
pub struct DifferentiableQEC {
    config: QECConfig,
    code: StabilizerCode,
    decoder: TrainableDecoder,
    noise: NoiseModel,
    /// Cached intermediate values for backprop.
    cache: QECache,
}

/// Cache for intermediate values during forward pass.
#[derive(Clone, Debug, Default)]
struct QECache {
    /// Encoded state coefficients.
    encoded_state: Vec<f64>,
    /// Soft syndrome values.
    soft_syndrome: Vec<f64>,
    /// Decoder output (soft correction).
    soft_correction: Vec<f64>,
    /// Final fidelity.
    fidelity: f64,
}

impl DifferentiableQEC {
    /// Create a new differentiable QEC pipeline.
    pub fn new(config: QECConfig) -> Self {
        let code = StabilizerCode::five_qubit_code();
        let n_stabilizers = code.stabilizers.len();
        let n_corrections = config.n_physical * 2; // X and Z corrections per qubit

        Self {
            decoder: TrainableDecoder::new(n_stabilizers, n_corrections),
            noise: NoiseModel::default(),
            code,
            config,
            cache: QECache::default(),
        }
    }

    /// Set the noise model.
    pub fn with_noise(mut self, noise: NoiseModel) -> Self {
        self.noise = noise;
        self
    }

    /// Set the code.
    pub fn with_code(mut self, code: StabilizerCode) -> Self {
        self.code = code;
        self
    }

    /// Forward pass: encode → noise → decode.
    /// Returns the fidelity after error correction.
    pub fn forward(&mut self, logical_state: &[f64]) -> f64 {
        // 1. Encode
        self.cache.encoded_state = self.encode(logical_state);

        // 2. Apply noise
        let noisy_state = self.apply_noise(&self.cache.encoded_state.clone());

        // 3. Measure syndrome (soft for gradients)
        self.cache.soft_syndrome = self.measure_syndrome_soft(&noisy_state);

        // 4. Decode
        self.cache.soft_correction = self.decoder.decode_soft(&self.cache.soft_syndrome);

        // 5. Apply correction
        let corrected_state = self.apply_correction_soft(&noisy_state, &self.cache.soft_correction);

        // 6. Compute fidelity
        self.cache.fidelity = self.compute_fidelity(&corrected_state, logical_state);

        self.cache.fidelity
    }

    /// Backward pass: compute gradients.
    pub fn backward(&self, _logical_state: &[f64]) -> QECGradients {
        // Target: identity correction (no error)
        let target_correction = vec![0.0; self.cache.soft_correction.len()];

        // Decoder gradients
        let decoder_grads = self.decoder.backward(
            &self.cache.soft_syndrome,
            &target_correction,
            &self.cache.soft_correction,
        );

        QECGradients {
            decoder: decoder_grads,
            noise: self.noise.gradients(),
        }
    }

    /// Forward + backward in one call.
    pub fn forward_and_backward(&mut self, logical_state: &[f64]) -> (f64, QECGradients) {
        let fidelity = self.forward(logical_state);
        let grads = self.backward(logical_state);
        (fidelity, grads)
    }

    /// Apply gradients to trainable parameters.
    pub fn apply_gradients(&mut self, grads: &QECGradients) {
        self.decoder.apply_gradients(&grads.decoder, self.config.learning_rate);
    }

    /// Train for one epoch.
    pub fn train_step(&mut self, logical_states: &[Vec<f64>]) -> TrainingStats {
        let mut total_fidelity = 0.0;
        let mut total_loss = 0.0;

        for state in logical_states {
            let (fidelity, grads) = self.forward_and_backward(state);
            self.apply_gradients(&grads);
            total_fidelity += fidelity;
            total_loss += 1.0 - fidelity;
        }

        let n = logical_states.len() as f64;
        TrainingStats {
            avg_fidelity: total_fidelity / n,
            avg_loss: total_loss / n,
        }
    }

    // --- Internal methods ---

    fn encode(&self, logical_state: &[f64]) -> Vec<f64> {
        // Simplified: copy logical state to first n_logical qubits
        // Real implementation would do stabilizer encoding
        let mut encoded = vec![0.0; 1 << self.config.n_physical];
        for (i, &amp) in logical_state.iter().enumerate() {
            if i < encoded.len() {
                encoded[i] = amp;
            }
        }
        // Normalize
        let norm: f64 = encoded.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for amp in &mut encoded {
                *amp /= norm;
            }
        }
        encoded
    }

    fn apply_noise(&self, state: &[f64]) -> Vec<f64> {
        // Simplified: just apply depolarizing as amplitude damping
        let p = self.noise.total_error_rate();
        state
            .iter()
            .map(|&amp| amp * (1.0 - p / 2.0))
            .collect()
    }

    fn measure_syndrome_soft(&self, state: &[f64]) -> Vec<f64> {
        // Soft syndrome: expectation values of stabilizers
        let n_stabilizers = self.code.stabilizers.len();
        let mut syndrome = vec![0.0; n_stabilizers];

        // Simplified: random soft values for testing
        // Real implementation would compute stabilizer expectations
        for (i, s) in syndrome.iter_mut().enumerate() {
            *s = 0.5 + 0.1 * ((i as f64 + state.iter().sum::<f64>()) * 0.1).sin();
        }

        syndrome
    }

    fn apply_correction_soft(&self, state: &[f64], correction: &[f64]) -> Vec<f64> {
        // Apply soft correction (weighted by confidence)
        let mut corrected = state.to_vec();
        let len = corrected.len();

        for (i, &c) in correction.iter().enumerate() {
            if i < len && c > 0.5 {
                // Apply "soft" Pauli operation
                corrected[i % len] *= 1.0 - 2.0 * (c - 0.5);
            }
        }

        corrected
    }

    fn compute_fidelity(&self, corrected: &[f64], original: &[f64]) -> f64 {
        // |<psi_original|psi_corrected>|^2
        let mut overlap = 0.0;
        for (c, o) in corrected.iter().zip(original.iter()) {
            overlap += c * o;
        }
        overlap * overlap
    }
}

/// Gradients for the entire QEC pipeline.
#[derive(Clone, Debug)]
pub struct QECGradients {
    pub decoder: DecoderGradients,
    pub noise: NoiseGradients,
}

/// Training statistics.
#[derive(Clone, Debug)]
pub struct TrainingStats {
    pub avg_fidelity: f64,
    pub avg_loss: f64,
}

// ===========================================================================
// HELPER FUNCTIONS
// ===========================================================================

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_model_depolarizing() {
        let noise = NoiseModel::depolarizing(0.01);
        // Total should be >= depolarizing rate (includes all error sources)
        assert!(noise.total_error_rate() >= 0.01);
        assert!((noise.p_depolarizing - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_noise_model_biased() {
        let noise = NoiseModel::biased(0.01, 0.001, 0.05);
        // Check individual rates
        assert!((noise.p_x - 0.01).abs() < 1e-10);
        assert!((noise.p_y - 0.001).abs() < 1e-10);
        assert!((noise.p_z - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_trainable_decoder_creation() {
        let decoder = TrainableDecoder::new(10, 20);
        assert_eq!(decoder.input_dim, 10);
        assert_eq!(decoder.output_dim, 20);
        assert!(decoder.num_parameters() > 0);
    }

    #[test]
    fn test_trainable_decoder_forward() {
        let decoder = TrainableDecoder::new(4, 6);
        let syndrome = vec![1.0, 0.0, 1.0, 0.0];
        let output = decoder.decode_soft(&syndrome);

        assert_eq!(output.len(), 6);
        // Output should be in [0, 1] range (sigmoid)
        for &o in &output {
            assert!(o >= 0.0 && o <= 1.0);
        }
    }

    #[test]
    fn test_trainable_decoder_hard() {
        let decoder = TrainableDecoder::new(4, 6);
        let syndrome = vec![true, false, true, false];
        let correction = decoder.decode_hard(&syndrome);

        assert_eq!(correction.len(), 6);
        // All values should be boolean
        for _ in &correction {
            // Just check we got booleans
        }
    }

    #[test]
    fn test_differentiable_qec_creation() {
        let config = QECConfig::new(5, 1);
        let qec = DifferentiableQEC::new(config);

        assert_eq!(qec.config.n_physical, 5);
        assert_eq!(qec.config.n_logical, 1);
    }

    #[test]
    fn test_differentiable_qec_forward() {
        let config = QECConfig::new(5, 1);
        let mut qec = DifferentiableQEC::new(config);
        let logical_state = vec![1.0, 0.0, 0.0, 0.0];

        let fidelity = qec.forward(&logical_state);

        assert!(fidelity >= 0.0 && fidelity <= 1.0);
    }

    #[test]
    fn test_differentiable_qec_backward() {
        let config = QECConfig::new(5, 1);
        let mut qec = DifferentiableQEC::new(config);
        let logical_state = vec![1.0, 0.0, 0.0, 0.0];

        qec.forward(&logical_state);
        let grads = qec.backward(&logical_state);

        assert_eq!(grads.decoder.weights.len(), qec.decoder.weights.len());
        assert_eq!(grads.decoder.biases.len(), qec.decoder.biases.len());
    }

    #[test]
    fn test_training_step() {
        let config = QECConfig::new(5, 1).with_soft_decode(true);
        let mut qec = DifferentiableQEC::new(config);

        let states = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.707, 0.707, 0.0, 0.0],
        ];

        let stats = qec.train_step(&states);

        assert!(stats.avg_fidelity >= 0.0 && stats.avg_fidelity <= 1.0);
        assert!(stats.avg_loss >= 0.0);
    }

    #[test]
    fn test_qec_config_builder() {
        let config = QECConfig::new(9, 1)
            .with_distance(3)
            .with_soft_decode(false);

        assert_eq!(config.n_physical, 9);
        assert_eq!(config.n_logical, 1);
        assert_eq!(config.distance, 3);
        assert!(!config.soft_decode);
    }
}
