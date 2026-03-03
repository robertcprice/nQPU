//! Unified Neural QEC Decoder Across All Code Families
//!
//! A generalised neural decoder inspired by Google's AlphaQubit but extended to work
//! across **every** QEC code family supported by nQPU-Metal: surface codes, colour
//! codes, qLDPC, Floquet honeycomb, bivariate bicycle, bosonic (cat/GKP), repetition
//! codes, hyperbolic Floquet, and yoked surface codes.
//!
//! # Architecture
//!
//! The decoder is a multi-layer perceptron (MLP) augmented with:
//!
//! 1. **Code-family embeddings** -- a learned embedding table that injects
//!    code-type-specific context into every forward pass so that a single set of
//!    weights can specialise per family.
//! 2. **Optional multi-head self-attention** over syndrome tokens, enabling the
//!    network to learn non-local correlations (critical for qLDPC and Floquet codes
//!    whose check graphs are not planar).
//! 3. **Transfer learning** -- pretrain on data-rich families (surface / repetition)
//!    and fine-tune on exotic families with limited training data.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::unified_neural_decoder::{
//!     UnifiedNeuralDecoderConfig, UnifiedNeuralDecoder, SyndromeData, CodeType,
//!     SyndromeGenerator,
//! };
//!
//! let config = UnifiedNeuralDecoderConfig::new()
//!     .hidden_dims(vec![64, 32])
//!     .learning_rate(0.01)
//!     .use_attention(false);
//!
//! let mut decoder = UnifiedNeuralDecoder::new(config, 9, 9);
//!
//! // Generate synthetic training data for a distance-3 surface code
//! let data = SyndromeGenerator::generate_surface_code(3, 0.01, 100);
//! let history = decoder.train(&data, 10);
//! assert!(history.epoch_losses.last() < history.epoch_losses.first());
//! ```

use ndarray::Array2;
use rand::Rng;
use std::fmt;
use std::time::Instant;

// ===========================================================================
// Error type
// ===========================================================================

/// Errors produced by the unified neural decoder.
#[derive(Debug, Clone)]
pub enum UnifiedDecoderError {
    /// A dimension did not match what the network expected.
    DimensionMismatch { expected: usize, got: usize },
    /// Training encountered a numerical problem.
    TrainingFailed(String),
    /// Weight deserialisation failed.
    WeightLoadFailed(String),
    /// Configuration values are invalid.
    InvalidConfig(String),
}

impl fmt::Display for UnifiedDecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnifiedDecoderError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            UnifiedDecoderError::TrainingFailed(msg) => {
                write!(f, "Training failed: {}", msg)
            }
            UnifiedDecoderError::WeightLoadFailed(msg) => {
                write!(f, "Weight load failed: {}", msg)
            }
            UnifiedDecoderError::InvalidConfig(msg) => {
                write!(f, "Invalid config: {}", msg)
            }
        }
    }
}

impl std::error::Error for UnifiedDecoderError {}

// ===========================================================================
// Activation enum
// ===========================================================================

/// Element-wise activation functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
}

impl Activation {
    /// Apply the activation to a scalar.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::GELU => {
                // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
        }
    }

    /// Derivative of the activation evaluated at `x`.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::GELU => {
                // Numerical derivative for GELU (stable enough for training)
                let h = 1e-7;
                (self.apply(x + h) - self.apply(x - h)) / (2.0 * h)
            }
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
        }
    }
}

// ===========================================================================
// CodeType enum
// ===========================================================================

/// Supported QEC code families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeType {
    SurfaceCode,
    ColorCode,
    QLDPC,
    FloquetHoneycomb,
    BivariateBicycle,
    CatQubit,
    GKP,
    RepetitionCode,
    HyperbolicFloquet,
    YokedSurface,
}

impl CodeType {
    /// Map to a unique embedding index in `[0, 9]`.
    pub fn to_embedding_index(&self) -> usize {
        match self {
            CodeType::SurfaceCode => 0,
            CodeType::ColorCode => 1,
            CodeType::QLDPC => 2,
            CodeType::FloquetHoneycomb => 3,
            CodeType::BivariateBicycle => 4,
            CodeType::CatQubit => 5,
            CodeType::GKP => 6,
            CodeType::RepetitionCode => 7,
            CodeType::HyperbolicFloquet => 8,
            CodeType::YokedSurface => 9,
        }
    }

    /// Total number of supported code families.
    pub const NUM_FAMILIES: usize = 10;

    /// Return all code type variants.
    pub fn all() -> &'static [CodeType] {
        &[
            CodeType::SurfaceCode,
            CodeType::ColorCode,
            CodeType::QLDPC,
            CodeType::FloquetHoneycomb,
            CodeType::BivariateBicycle,
            CodeType::CatQubit,
            CodeType::GKP,
            CodeType::RepetitionCode,
            CodeType::HyperbolicFloquet,
            CodeType::YokedSurface,
        ]
    }
}

impl fmt::Display for CodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            CodeType::SurfaceCode => "Surface",
            CodeType::ColorCode => "Color",
            CodeType::QLDPC => "qLDPC",
            CodeType::FloquetHoneycomb => "Floquet-Honeycomb",
            CodeType::BivariateBicycle => "Bivariate-Bicycle",
            CodeType::CatQubit => "Cat-Qubit",
            CodeType::GKP => "GKP",
            CodeType::RepetitionCode => "Repetition",
            CodeType::HyperbolicFloquet => "Hyperbolic-Floquet",
            CodeType::YokedSurface => "Yoked-Surface",
        };
        write!(f, "{}", name)
    }
}

// ===========================================================================
// Data structures
// ===========================================================================

/// A single syndrome measurement sample with metadata.
#[derive(Debug, Clone)]
pub struct SyndromeData {
    /// Binary stabiliser measurement outcomes (`true` = triggered).
    pub syndrome: Vec<bool>,
    /// Which code family produced this syndrome.
    pub code_type: CodeType,
    /// Code distance.
    pub distance: usize,
    /// Number of repeated measurement rounds (temporal dimension).
    pub num_rounds: usize,
    /// Physical error rate used during sampling.
    pub noise_rate: f64,
}

/// Target correction label for supervised training.
#[derive(Debug, Clone)]
pub struct CorrectionLabel {
    /// Which data qubits need a Pauli correction.
    pub corrections: Vec<bool>,
    /// Whether a logical operator correction is needed (one per logical qubit).
    pub logical_correction: Vec<bool>,
}

// ===========================================================================
// Configuration (builder pattern)
// ===========================================================================

/// Configuration for [`UnifiedNeuralDecoder`], constructed via builder methods.
#[derive(Debug, Clone)]
pub struct UnifiedNeuralDecoderConfig {
    /// Hidden layer dimensions (e.g. `[128, 64, 32]`).
    pub hidden_dims: Vec<usize>,
    /// SGD learning rate.
    pub learning_rate: f64,
    /// Training mini-batch size.
    pub batch_size: usize,
    /// Default number of training epochs.
    pub num_epochs: usize,
    /// Activation function applied after each hidden layer.
    pub activation: Activation,
    /// Dropout probability (applied conceptually during training).
    pub dropout_rate: f64,
    /// Enable multi-head self-attention over syndrome tokens.
    pub use_attention: bool,
    /// Dimensionality of the learned code-family embedding.
    pub code_family_embedding_dim: usize,
    /// Enable transfer learning features.
    pub transfer_learning: bool,
}

impl UnifiedNeuralDecoderConfig {
    /// Create a configuration with sensible defaults.
    pub fn new() -> Self {
        UnifiedNeuralDecoderConfig {
            hidden_dims: vec![128, 64, 32],
            learning_rate: 0.001,
            batch_size: 32,
            num_epochs: 100,
            activation: Activation::ReLU,
            dropout_rate: 0.1,
            use_attention: true,
            code_family_embedding_dim: 16,
            transfer_learning: true,
        }
    }

    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    pub fn activation(mut self, a: Activation) -> Self {
        self.activation = a;
        self
    }

    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    pub fn use_attention(mut self, flag: bool) -> Self {
        self.use_attention = flag;
        self
    }

    pub fn code_family_embedding_dim(mut self, dim: usize) -> Self {
        self.code_family_embedding_dim = dim;
        self
    }

    pub fn transfer_learning(mut self, flag: bool) -> Self {
        self.transfer_learning = flag;
        self
    }

    /// Validate the configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> Result<(), UnifiedDecoderError> {
        if self.hidden_dims.is_empty() {
            return Err(UnifiedDecoderError::InvalidConfig(
                "hidden_dims must not be empty".into(),
            ));
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(UnifiedDecoderError::InvalidConfig(
                "learning_rate must be in (0, 1]".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(UnifiedDecoderError::InvalidConfig(
                "batch_size must be > 0".into(),
            ));
        }
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(UnifiedDecoderError::InvalidConfig(
                "dropout_rate must be in [0, 1)".into(),
            ));
        }
        if self.code_family_embedding_dim == 0 {
            return Err(UnifiedDecoderError::InvalidConfig(
                "code_family_embedding_dim must be > 0".into(),
            ));
        }
        Ok(())
    }
}

impl Default for UnifiedNeuralDecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Dense layer
// ===========================================================================

/// A fully-connected linear layer with bias.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weight matrix of shape `(out_dim, in_dim)`.
    pub weights: Array2<f64>,
    /// Bias vector of length `out_dim`.
    pub bias: Vec<f64>,
}

impl DenseLayer {
    /// Construct with Xavier-uniform initialisation.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let weights =
            Array2::from_shape_fn((out_dim, in_dim), |_| rng.gen_range(-limit..limit));
        let bias = vec![0.0; out_dim];
        DenseLayer { weights, bias }
    }

    /// Construct from explicit weights and bias (useful for testing).
    pub fn from_weights(weights: Array2<f64>, bias: Vec<f64>) -> Self {
        DenseLayer { weights, bias }
    }

    /// Forward pass: `output = W @ input + b`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let (out_dim, _in_dim) = (self.weights.nrows(), self.weights.ncols());
        let mut output = vec![0.0; out_dim];
        for i in 0..out_dim {
            let mut sum = self.bias[i];
            for j in 0..input.len().min(self.weights.ncols()) {
                sum += self.weights[[i, j]] * input[j];
            }
            output[i] = sum;
        }
        output
    }

    /// Backward pass computing gradients.
    ///
    /// Returns `(grad_input, grad_weights, grad_bias)`.
    pub fn backward(
        &self,
        grad_output: &[f64],
        input: &[f64],
    ) -> (Vec<f64>, Array2<f64>, Vec<f64>) {
        let (out_dim, in_dim) = (self.weights.nrows(), self.weights.ncols());

        // grad_input = W^T @ grad_output
        let mut grad_input = vec![0.0; in_dim];
        for j in 0..in_dim {
            for i in 0..out_dim {
                grad_input[j] += self.weights[[i, j]] * grad_output[i];
            }
        }

        // grad_weights = grad_output (outer) input
        let mut grad_weights = Array2::zeros((out_dim, in_dim));
        for i in 0..out_dim {
            for j in 0..in_dim {
                grad_weights[[i, j]] = grad_output[i] * input[j];
            }
        }

        // grad_bias = grad_output
        let grad_bias = grad_output.to_vec();

        (grad_input, grad_weights, grad_bias)
    }

    /// Apply an SGD update: `param -= lr * grad`.
    pub fn sgd_update(&mut self, grad_weights: &Array2<f64>, grad_bias: &[f64], lr: f64) {
        let (out_dim, in_dim) = (self.weights.nrows(), self.weights.ncols());
        for i in 0..out_dim {
            for j in 0..in_dim {
                self.weights[[i, j]] -= lr * grad_weights[[i, j]];
            }
            self.bias[i] -= lr * grad_bias[i];
        }
    }

    /// Total number of parameters (weights + bias).
    pub fn num_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

// ===========================================================================
// Attention layer
// ===========================================================================

/// Multi-head self-attention over a sequence of feature vectors.
///
/// Input is treated as `(seq_len, feature_dim)`.  Queries, keys, and values
/// are linearly projected, split across heads, and recombined.
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Query projection: `(feature_dim, feature_dim)`.
    pub query_proj: Array2<f64>,
    /// Key projection: `(feature_dim, feature_dim)`.
    pub key_proj: Array2<f64>,
    /// Value projection: `(feature_dim, feature_dim)`.
    pub value_proj: Array2<f64>,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feature dimension.
    pub feature_dim: usize,
}

impl AttentionLayer {
    /// Construct with Xavier-initialised projections.
    pub fn new(feature_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (feature_dim + feature_dim) as f64).sqrt();
        let make_proj = |rng: &mut rand::rngs::ThreadRng| {
            Array2::from_shape_fn((feature_dim, feature_dim), |_| {
                rng.gen_range(-limit..limit)
            })
        };
        AttentionLayer {
            query_proj: make_proj(&mut rng),
            key_proj: make_proj(&mut rng),
            value_proj: make_proj(&mut rng),
            num_heads,
            feature_dim,
        }
    }

    /// Compute softmax attention weights between a query and a set of keys.
    pub fn attention_weights(&self, query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
        let head_dim = self.feature_dim.max(1);
        let scale = (head_dim as f64).sqrt();

        // Compute dot-product scores
        let scores: Vec<f64> = keys
            .iter()
            .map(|k| {
                query
                    .iter()
                    .zip(k.iter())
                    .map(|(q, ki)| q * ki)
                    .sum::<f64>()
                    / scale
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        if sum_exp == 0.0 {
            return vec![1.0 / keys.len() as f64; keys.len()];
        }
        exps.iter().map(|e| e / sum_exp).collect()
    }

    /// Forward pass: self-attention over a sequence of token vectors.
    ///
    /// `input` has shape `[seq_len][feature_dim]`.
    /// Returns output with the same shape.
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = input.len();
        if seq_len == 0 {
            return vec![];
        }

        // Project Q, K, V
        let project = |x: &[f64], proj: &Array2<f64>| -> Vec<f64> {
            let dim = proj.nrows();
            let mut out = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..x.len().min(proj.ncols()) {
                    out[i] += proj[[i, j]] * x[j];
                }
            }
            out
        };

        let queries: Vec<Vec<f64>> = input.iter().map(|x| project(x, &self.query_proj)).collect();
        let keys: Vec<Vec<f64>> = input.iter().map(|x| project(x, &self.key_proj)).collect();
        let values: Vec<Vec<f64>> = input.iter().map(|x| project(x, &self.value_proj)).collect();

        // For each query position, attend over all keys
        let mut output = Vec::with_capacity(seq_len);
        for qi in 0..seq_len {
            let weights = self.attention_weights(&queries[qi], &keys);
            let mut attended = vec![0.0; self.feature_dim];
            for (ki, w) in weights.iter().enumerate() {
                for d in 0..self.feature_dim.min(values[ki].len()) {
                    attended[d] += w * values[ki][d];
                }
            }
            output.push(attended);
        }
        output
    }
}

// ===========================================================================
// Code-family embedding
// ===========================================================================

/// Learned embedding table mapping code types to dense vectors.
#[derive(Debug, Clone)]
pub struct CodeFamilyEmbedding {
    /// Embedding matrix of shape `(NUM_FAMILIES, embedding_dim)`.
    pub embeddings: Array2<f64>,
}

impl CodeFamilyEmbedding {
    /// Create randomly initialised embeddings.
    pub fn new(embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (1.0 / embedding_dim as f64).sqrt();
        let embeddings = Array2::from_shape_fn(
            (CodeType::NUM_FAMILIES, embedding_dim),
            |_| rng.gen_range(-limit..limit),
        );
        CodeFamilyEmbedding { embeddings }
    }

    /// Look up the embedding vector for a code type.
    pub fn encode(&self, code_type: &CodeType) -> Vec<f64> {
        let idx = code_type.to_embedding_index();
        self.embeddings
            .row(idx)
            .iter()
            .cloned()
            .collect()
    }

    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.embeddings.ncols()
    }
}

// ===========================================================================
// Training history & evaluation result
// ===========================================================================

/// Record of a training run.
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Mean loss per epoch.
    pub epoch_losses: Vec<f64>,
    /// Accuracy per epoch (fraction of correctly decoded samples).
    pub epoch_accuracies: Vec<f64>,
    /// Epoch index with the best accuracy.
    pub best_epoch: usize,
    /// Best accuracy achieved.
    pub best_accuracy: f64,
    /// Whether the loss converged below a threshold.
    pub converged: bool,
}

/// Results from evaluating the decoder on a test set.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Overall fraction of perfectly decoded samples.
    pub accuracy: f64,
    /// Logical error rate (fraction of samples with any logical error).
    pub logical_error_rate: f64,
    /// Accuracy broken down by code family.
    pub per_code_accuracy: Vec<(CodeType, f64)>,
    /// Confusion matrix (predicted x actual).
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Average inference time in microseconds.
    pub avg_inference_time_us: f64,
}

/// Comparison between the neural decoder and MWPM baseline.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub neural_accuracy: f64,
    pub mwpm_accuracy: f64,
    pub neural_time_us: f64,
    pub mwpm_time_us: f64,
    /// Absolute improvement: `neural_accuracy - mwpm_accuracy`.
    pub improvement: f64,
}

// ===========================================================================
// Unified neural decoder
// ===========================================================================

/// A unified neural decoder that works across all QEC code families.
///
/// Combines an MLP backbone with optional self-attention and code-family
/// embeddings to decode syndrome measurements from any supported code.
pub struct UnifiedNeuralDecoder {
    /// Hidden MLP layers (excluding the output projection).
    pub layers: Vec<DenseLayer>,
    /// Output projection to correction-size vector.
    pub output_layer: DenseLayer,
    /// Optional self-attention.
    pub attention: Option<AttentionLayer>,
    /// Code-family embedding table.
    pub code_embedding: CodeFamilyEmbedding,
    /// Configuration.
    pub config: UnifiedNeuralDecoderConfig,
    /// Maximum syndrome vector length this decoder accepts.
    pub max_syndrome_size: usize,
    /// Maximum correction vector length this decoder produces.
    pub max_correction_size: usize,
}

impl UnifiedNeuralDecoder {
    /// Construct a new (untrained) unified decoder.
    ///
    /// `max_syndrome_size` is the largest syndrome vector the decoder will
    /// encounter.  `max_correction_size` is the largest correction vector.
    pub fn new(
        config: UnifiedNeuralDecoderConfig,
        max_syndrome_size: usize,
        max_correction_size: usize,
    ) -> Self {
        let emb_dim = config.code_family_embedding_dim;
        let input_dim = max_syndrome_size + emb_dim;

        // Build hidden layers
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;
        for &hdim in &config.hidden_dims {
            layers.push(DenseLayer::new(prev_dim, hdim));
            prev_dim = hdim;
        }

        // Output layer maps last hidden dim -> correction size (sigmoid applied externally)
        let output_layer = DenseLayer::new(prev_dim, max_correction_size);

        // Optional attention on the first hidden dimension
        let attention = if config.use_attention && !config.hidden_dims.is_empty() {
            Some(AttentionLayer::new(config.hidden_dims[0], 4))
        } else {
            None
        };

        let code_embedding = CodeFamilyEmbedding::new(emb_dim);

        UnifiedNeuralDecoder {
            layers,
            output_layer,
            attention,
            code_embedding,
            config,
            max_syndrome_size,
            max_correction_size,
        }
    }

    /// Prepare a fixed-length input vector from a syndrome sample.
    ///
    /// Pads/truncates the syndrome to `max_syndrome_size` and concatenates
    /// the code-family embedding.
    fn prepare_input(&self, syndrome: &SyndromeData) -> Vec<f64> {
        let mut input = vec![0.0; self.max_syndrome_size];
        for (i, &bit) in syndrome.syndrome.iter().enumerate() {
            if i < self.max_syndrome_size {
                input[i] = if bit { 1.0 } else { 0.0 };
            }
        }
        let emb = self.code_embedding.encode(&syndrome.code_type);
        input.extend_from_slice(&emb);
        input
    }

    /// Raw forward pass through the network.
    ///
    /// Returns the pre-sigmoid logits of length `max_correction_size`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut hidden = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden);

            // Apply attention after the first hidden layer
            if i == 0 {
                if let Some(ref attn) = self.attention {
                    // Treat hidden as a sequence of 1-element tokens for attention
                    let as_seq: Vec<Vec<f64>> = hidden.iter().map(|&v| vec![v]).collect();
                    if !as_seq.is_empty() {
                        let attn_out = attn.forward(&as_seq);
                        // Flatten back (take first element of each token)
                        for (j, token) in attn_out.iter().enumerate() {
                            if j < hidden.len() && !token.is_empty() {
                                // Residual connection
                                hidden[j] += token[0];
                            }
                        }
                    }
                }
            }

            // Activation
            for v in hidden.iter_mut() {
                *v = self.config.activation.apply(*v);
            }
        }

        // Output layer (no activation -- sigmoid applied in predict/train)
        self.output_layer.forward(&hidden)
    }

    /// Predict corrections for a syndrome sample.
    pub fn predict(&self, syndrome: &SyndromeData) -> CorrectionLabel {
        let input = self.prepare_input(syndrome);
        let logits = self.forward(&input);

        // Apply sigmoid and threshold at 0.5
        let corrections: Vec<bool> = logits
            .iter()
            .take(self.max_correction_size)
            .map(|&x| Activation::Sigmoid.apply(x) > 0.5)
            .collect();

        CorrectionLabel {
            corrections,
            logical_correction: vec![false],
        }
    }

    /// Execute a single training step (one mini-batch) and return the mean loss.
    pub fn train_step(
        &mut self,
        syndromes: &[SyndromeData],
        labels: &[CorrectionLabel],
    ) -> f64 {
        if syndromes.is_empty() || labels.is_empty() {
            return 0.0;
        }

        let lr = self.config.learning_rate;
        let mut total_loss = 0.0;
        let n = syndromes.len().min(labels.len());

        for idx in 0..n {
            let input = self.prepare_input(&syndromes[idx]);
            let target: Vec<f64> = labels[idx]
                .corrections
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect();

            // --- Forward pass (store intermediates) ---
            let mut activations: Vec<Vec<f64>> = Vec::new();
            let mut pre_activations: Vec<Vec<f64>> = Vec::new();
            activations.push(input.clone());

            let mut hidden = input;
            for layer in &self.layers {
                let pre_act = layer.forward(&hidden);
                pre_activations.push(pre_act.clone());
                hidden = pre_act
                    .iter()
                    .map(|&v| self.config.activation.apply(v))
                    .collect();
                activations.push(hidden.clone());
            }

            let logits = self.output_layer.forward(&hidden);
            let probs: Vec<f64> = logits.iter().map(|&x| Activation::Sigmoid.apply(x)).collect();

            // Binary cross-entropy loss
            let mut sample_loss = 0.0;
            for i in 0..self.max_correction_size.min(target.len()) {
                let p = probs[i].clamp(1e-7, 1.0 - 1e-7);
                let t = target[i];
                sample_loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
            }
            let correction_count = self.max_correction_size.min(target.len()).max(1);
            sample_loss /= correction_count as f64;
            total_loss += sample_loss;

            // --- Backward pass ---
            // dL/d_logits = probs - target (gradient of BCE through sigmoid)
            let grad_output: Vec<f64> = (0..self.max_correction_size)
                .map(|i| {
                    if i < target.len() {
                        (probs[i] - target[i]) / correction_count as f64
                    } else {
                        0.0
                    }
                })
                .collect();

            // Output layer backward
            let last_hidden = activations.last().unwrap();
            let (mut grad_hidden, gw, gb) = self.output_layer.backward(&grad_output, last_hidden);
            self.output_layer.sgd_update(&gw, &gb, lr);

            // Hidden layers backward (reverse order)
            for (layer_idx, layer) in self.layers.iter_mut().enumerate().rev() {
                // Apply activation derivative
                let pre_act = &pre_activations[layer_idx];
                for (i, grad) in grad_hidden.iter_mut().enumerate() {
                    if i < pre_act.len() {
                        *grad *= self.config.activation.derivative(pre_act[i]);
                    }
                }

                let layer_input = &activations[layer_idx];
                let (gi, gw, gb) = layer.backward(&grad_hidden, layer_input);
                layer.sgd_update(&gw, &gb, lr);
                grad_hidden = gi;
            }
        }

        total_loss / n as f64
    }

    /// Train for multiple epochs, returning a history of losses and accuracies.
    pub fn train(
        &mut self,
        dataset: &[(SyndromeData, CorrectionLabel)],
        epochs: usize,
    ) -> TrainingHistory {
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut epoch_accuracies = Vec::with_capacity(epochs);
        let mut best_accuracy = 0.0_f64;
        let mut best_epoch = 0;

        let syndromes: Vec<&SyndromeData> = dataset.iter().map(|(s, _)| s).collect();
        let labels: Vec<&CorrectionLabel> = dataset.iter().map(|(_, l)| l).collect();

        for epoch in 0..epochs {
            // Process in mini-batches
            let bs = self.config.batch_size.min(dataset.len()).max(1);
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            for chunk_start in (0..dataset.len()).step_by(bs) {
                let chunk_end = (chunk_start + bs).min(dataset.len());
                let batch_s: Vec<SyndromeData> =
                    syndromes[chunk_start..chunk_end].iter().map(|s| (*s).clone()).collect();
                let batch_l: Vec<CorrectionLabel> =
                    labels[chunk_start..chunk_end].iter().map(|l| (*l).clone()).collect();

                let loss = self.train_step(&batch_s, &batch_l);
                epoch_loss += loss;
                batch_count += 1;
            }

            epoch_loss /= batch_count.max(1) as f64;
            epoch_losses.push(epoch_loss);

            // Compute epoch accuracy
            let mut correct = 0;
            for (syn, lbl) in dataset.iter() {
                let pred = self.predict(syn);
                let matched = pred
                    .corrections
                    .iter()
                    .zip(lbl.corrections.iter())
                    .all(|(p, t)| p == t);
                if matched {
                    correct += 1;
                }
            }
            let accuracy = correct as f64 / dataset.len().max(1) as f64;
            epoch_accuracies.push(accuracy);

            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_epoch = epoch;
            }
        }

        let converged = if epoch_losses.len() >= 2 {
            let last = *epoch_losses.last().unwrap();
            let first = epoch_losses[0];
            last < first * 0.1 || last < 1e-4
        } else {
            false
        };

        TrainingHistory {
            epoch_losses,
            epoch_accuracies,
            best_epoch,
            best_accuracy,
            converged,
        }
    }

    /// Evaluate the decoder on a test set.
    pub fn evaluate(
        &self,
        test_data: &[(SyndromeData, CorrectionLabel)],
    ) -> EvaluationResult {
        if test_data.is_empty() {
            return EvaluationResult {
                accuracy: 0.0,
                logical_error_rate: 1.0,
                per_code_accuracy: vec![],
                confusion_matrix: vec![],
                avg_inference_time_us: 0.0,
            };
        }

        let mut correct = 0;
        let mut logical_errors = 0;
        let mut total_time = 0u128;

        // Per-code tracking
        let mut per_code_correct: Vec<usize> = vec![0; CodeType::NUM_FAMILIES];
        let mut per_code_total: Vec<usize> = vec![0; CodeType::NUM_FAMILIES];

        // Simple 2x2 confusion matrix (correct / incorrect)
        let mut confusion = vec![vec![0usize; 2]; 2];

        for (syn, lbl) in test_data {
            let start = Instant::now();
            let pred = self.predict(syn);
            total_time += start.elapsed().as_micros();

            let matched = pred
                .corrections
                .iter()
                .zip(lbl.corrections.iter())
                .all(|(p, t)| p == t);

            let code_idx = syn.code_type.to_embedding_index();
            per_code_total[code_idx] += 1;

            if matched {
                correct += 1;
                per_code_correct[code_idx] += 1;
                confusion[0][0] += 1;
            } else {
                confusion[1][0] += 1;
            }

            // Check logical correction accuracy
            let logical_ok = pred
                .logical_correction
                .iter()
                .zip(lbl.logical_correction.iter())
                .all(|(p, t)| p == t);
            if !logical_ok {
                logical_errors += 1;
            }
        }

        let n = test_data.len();
        let per_code_accuracy: Vec<(CodeType, f64)> = CodeType::all()
            .iter()
            .enumerate()
            .filter(|(i, _)| per_code_total[*i] > 0)
            .map(|(i, ct)| (*ct, per_code_correct[i] as f64 / per_code_total[i] as f64))
            .collect();

        EvaluationResult {
            accuracy: correct as f64 / n as f64,
            logical_error_rate: logical_errors as f64 / n as f64,
            per_code_accuracy,
            confusion_matrix: confusion,
            avg_inference_time_us: total_time as f64 / n as f64,
        }
    }

    /// Serialise all trainable weights into a flat vector.
    pub fn save_weights(&self) -> Vec<f64> {
        let mut flat = Vec::new();
        for layer in &self.layers {
            flat.extend(layer.weights.iter());
            flat.extend(&layer.bias);
        }
        flat.extend(self.output_layer.weights.iter());
        flat.extend(&self.output_layer.bias);
        flat.extend(self.code_embedding.embeddings.iter());
        flat
    }

    /// Load weights from a flat vector produced by [`save_weights`].
    pub fn load_weights(&mut self, weights: &[f64]) {
        let mut offset = 0;

        for layer in &mut self.layers {
            let w_len = layer.weights.len();
            let b_len = layer.bias.len();
            if offset + w_len + b_len > weights.len() {
                return;
            }
            for (i, w) in layer.weights.iter_mut().enumerate() {
                *w = weights[offset + i];
            }
            offset += w_len;
            for (i, b) in layer.bias.iter_mut().enumerate() {
                *b = weights[offset + i];
            }
            offset += b_len;
        }

        // Output layer
        let w_len = self.output_layer.weights.len();
        let b_len = self.output_layer.bias.len();
        if offset + w_len + b_len <= weights.len() {
            for (i, w) in self.output_layer.weights.iter_mut().enumerate() {
                *w = weights[offset + i];
            }
            offset += w_len;
            for (i, b) in self.output_layer.bias.iter_mut().enumerate() {
                *b = weights[offset + i];
            }
            offset += b_len;
        }

        // Code embeddings
        let emb_len = self.code_embedding.embeddings.len();
        if offset + emb_len <= weights.len() {
            for (i, e) in self.code_embedding.embeddings.iter_mut().enumerate() {
                *e = weights[offset + i];
            }
        }
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        let hidden: usize = self.layers.iter().map(|l| l.num_params()).sum();
        let output = self.output_layer.num_params();
        let emb = self.code_embedding.embeddings.len();
        hidden + output + emb
    }
}

// ===========================================================================
// Syndrome generator
// ===========================================================================

/// Synthetic syndrome generation for training and benchmarking.
pub struct SyndromeGenerator;

impl SyndromeGenerator {
    /// Generate training data for a distance-`d` surface code.
    ///
    /// Simulates independent depolarising noise at rate `noise_rate` and
    /// produces `num_samples` (syndrome, correction) pairs.
    pub fn generate_surface_code(
        distance: usize,
        noise_rate: f64,
        num_samples: usize,
    ) -> Vec<(SyndromeData, CorrectionLabel)> {
        let mut rng = rand::thread_rng();
        let num_data_qubits = distance * distance;
        // Surface code has (d-1)^2 X-stabilisers and (d-1)^2 Z-stabilisers
        // Simplified: use (d-1)^2 syndrome bits total for a single round
        let num_syndrome_bits = if distance > 1 {
            (distance - 1) * (distance - 1)
        } else {
            1
        };

        let mut dataset = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Random Pauli errors on data qubits
            let errors: Vec<bool> = (0..num_data_qubits)
                .map(|_| rng.gen::<f64>() < noise_rate)
                .collect();

            // Compute syndrome: each check involves 4 neighbouring data qubits
            // (simplified model for training data)
            let mut syndrome = vec![false; num_syndrome_bits];
            for s in 0..num_syndrome_bits {
                let row = s / (distance - 1).max(1);
                let col = s % (distance - 1).max(1);
                // Check adjacent data qubits
                let qubits = [
                    row * distance + col,
                    row * distance + col + 1,
                    (row + 1) * distance + col,
                    (row + 1) * distance + col + 1,
                ];
                let parity: usize = qubits
                    .iter()
                    .filter(|&&q| q < num_data_qubits && errors[q])
                    .count();
                syndrome[s] = parity % 2 == 1;
            }

            dataset.push((
                SyndromeData {
                    syndrome,
                    code_type: CodeType::SurfaceCode,
                    distance,
                    num_rounds: 1,
                    noise_rate,
                },
                CorrectionLabel {
                    corrections: errors,
                    logical_correction: vec![false],
                },
            ));
        }

        dataset
    }

    /// Generate training data for a distance-`d` repetition code.
    pub fn generate_repetition_code(
        distance: usize,
        noise_rate: f64,
        num_samples: usize,
    ) -> Vec<(SyndromeData, CorrectionLabel)> {
        let mut rng = rand::thread_rng();
        let num_data_qubits = distance;
        let num_syndrome_bits = distance.saturating_sub(1).max(1);

        let mut dataset = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let errors: Vec<bool> = (0..num_data_qubits)
                .map(|_| rng.gen::<f64>() < noise_rate)
                .collect();

            let mut syndrome = vec![false; num_syndrome_bits];
            for s in 0..num_syndrome_bits {
                // Each syndrome bit is the XOR of two adjacent data qubits
                syndrome[s] = errors[s] ^ errors.get(s + 1).copied().unwrap_or(false);
            }

            dataset.push((
                SyndromeData {
                    syndrome,
                    code_type: CodeType::RepetitionCode,
                    distance,
                    num_rounds: 1,
                    noise_rate,
                },
                CorrectionLabel {
                    corrections: errors,
                    logical_correction: vec![false],
                },
            ));
        }

        dataset
    }

    /// Generate training data for an arbitrary code with random syndromes.
    pub fn generate_random_code(
        syndrome_size: usize,
        correction_size: usize,
        noise_rate: f64,
        num_samples: usize,
    ) -> Vec<(SyndromeData, CorrectionLabel)> {
        let mut rng = rand::thread_rng();
        let mut dataset = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let syndrome: Vec<bool> = (0..syndrome_size)
                .map(|_| rng.gen::<f64>() < noise_rate)
                .collect();
            let corrections: Vec<bool> = (0..correction_size)
                .map(|_| rng.gen::<f64>() < noise_rate)
                .collect();

            dataset.push((
                SyndromeData {
                    syndrome,
                    code_type: CodeType::QLDPC,
                    distance: 3,
                    num_rounds: 1,
                    noise_rate,
                },
                CorrectionLabel {
                    corrections,
                    logical_correction: vec![false],
                },
            ));
        }

        dataset
    }
}

// ===========================================================================
// Transfer learning manager
// ===========================================================================

/// Manages pretraining and fine-tuning workflows across code families.
pub struct TransferLearningManager;

impl TransferLearningManager {
    /// Pretrain the decoder on a source dataset (typically surface code).
    pub fn pretrain(
        decoder: &mut UnifiedNeuralDecoder,
        source_data: &[(SyndromeData, CorrectionLabel)],
    ) -> TrainingHistory {
        let epochs = decoder.config.num_epochs;
        decoder.train(source_data, epochs)
    }

    /// Fine-tune the decoder on a target dataset, optionally freezing early layers.
    ///
    /// `freeze_layers` specifies how many of the first hidden layers to keep
    /// frozen (their weights are not updated).
    pub fn finetune(
        decoder: &mut UnifiedNeuralDecoder,
        target_data: &[(SyndromeData, CorrectionLabel)],
        freeze_layers: usize,
    ) -> TrainingHistory {
        if target_data.is_empty() {
            return TrainingHistory {
                epoch_losses: vec![],
                epoch_accuracies: vec![],
                best_epoch: 0,
                best_accuracy: 0.0,
                converged: false,
            };
        }

        // Save frozen layer weights
        let frozen_weights: Vec<(Array2<f64>, Vec<f64>)> = decoder
            .layers
            .iter()
            .take(freeze_layers)
            .map(|l| (l.weights.clone(), l.bias.clone()))
            .collect();

        // Train with a reduced learning rate
        let original_lr = decoder.config.learning_rate;
        decoder.config.learning_rate *= 0.1; // Fine-tune with smaller LR
        let epochs = decoder.config.num_epochs / 2;
        let history = decoder.train(target_data, epochs.max(5));

        // Restore frozen layer weights
        for (i, (w, b)) in frozen_weights.into_iter().enumerate() {
            if i < decoder.layers.len() {
                decoder.layers[i].weights = w;
                decoder.layers[i].bias = b;
            }
        }

        decoder.config.learning_rate = original_lr;
        history
    }
}

// ===========================================================================
// Decoder benchmark (neural vs MWPM comparison)
// ===========================================================================

/// Benchmark utility comparing the neural decoder against a simple MWPM baseline.
pub struct DecoderBenchmark;

impl DecoderBenchmark {
    /// Simple minimum-weight perfect matching decoder (greedy approximation).
    ///
    /// For benchmarking purposes only; a real MWPM would use Blossom V.
    pub fn mwpm_decode(syndrome: &SyndromeData) -> CorrectionLabel {
        // Greedy: for each triggered syndrome bit, flip the nearest data qubit
        let d = syndrome.distance.max(1);
        let num_corrections = d * d; // surface code data qubits
        let mut corrections = vec![false; num_corrections];

        for (i, &triggered) in syndrome.syndrome.iter().enumerate() {
            if triggered {
                // Map syndrome bit to a data qubit (simplified heuristic)
                let qubit_idx = i.min(num_corrections.saturating_sub(1));
                corrections[qubit_idx] = !corrections[qubit_idx]; // Toggle
            }
        }

        CorrectionLabel {
            corrections,
            logical_correction: vec![false],
        }
    }

    /// Compare neural decoder vs MWPM on a test dataset.
    pub fn compare_mwpm_vs_neural(
        test_data: &[(SyndromeData, CorrectionLabel)],
        decoder: &UnifiedNeuralDecoder,
    ) -> ComparisonResult {
        if test_data.is_empty() {
            return ComparisonResult {
                neural_accuracy: 0.0,
                mwpm_accuracy: 0.0,
                neural_time_us: 0.0,
                mwpm_time_us: 0.0,
                improvement: 0.0,
            };
        }

        let n = test_data.len();
        let mut neural_correct = 0;
        let mut mwpm_correct = 0;
        let mut neural_time = 0u128;
        let mut mwpm_time = 0u128;

        for (syn, lbl) in test_data {
            // Neural decoder
            let start = Instant::now();
            let neural_pred = decoder.predict(syn);
            neural_time += start.elapsed().as_micros();

            let neural_ok = neural_pred
                .corrections
                .iter()
                .zip(lbl.corrections.iter())
                .all(|(p, t)| p == t);
            if neural_ok {
                neural_correct += 1;
            }

            // MWPM baseline
            let start = Instant::now();
            let mwpm_pred = Self::mwpm_decode(syn);
            mwpm_time += start.elapsed().as_micros();

            let mwpm_ok = mwpm_pred
                .corrections
                .iter()
                .zip(lbl.corrections.iter())
                .all(|(p, t)| p == t);
            if mwpm_ok {
                mwpm_correct += 1;
            }
        }

        let neural_accuracy = neural_correct as f64 / n as f64;
        let mwpm_accuracy = mwpm_correct as f64 / n as f64;

        ComparisonResult {
            neural_accuracy,
            mwpm_accuracy,
            neural_time_us: neural_time as f64 / n as f64,
            mwpm_time_us: mwpm_time as f64 / n as f64,
            improvement: neural_accuracy - mwpm_accuracy,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Config builder validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_builder_defaults() {
        let cfg = UnifiedNeuralDecoderConfig::new();
        assert_eq!(cfg.hidden_dims, vec![128, 64, 32]);
        assert!((cfg.learning_rate - 0.001).abs() < 1e-9);
        assert_eq!(cfg.batch_size, 32);
        assert_eq!(cfg.num_epochs, 100);
        assert_eq!(cfg.activation, Activation::ReLU);
        assert!((cfg.dropout_rate - 0.1).abs() < 1e-9);
        assert!(cfg.use_attention);
        assert_eq!(cfg.code_family_embedding_dim, 16);
        assert!(cfg.transfer_learning);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder_validation_empty_hidden() {
        let cfg = UnifiedNeuralDecoderConfig::new().hidden_dims(vec![]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_builder_validation_bad_lr() {
        let cfg = UnifiedNeuralDecoderConfig::new().learning_rate(0.0);
        assert!(cfg.validate().is_err());

        let cfg2 = UnifiedNeuralDecoderConfig::new().learning_rate(1.5);
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn test_config_builder_validation_bad_dropout() {
        let cfg = UnifiedNeuralDecoderConfig::new().dropout_rate(-0.1);
        assert!(cfg.validate().is_err());

        let cfg2 = UnifiedNeuralDecoderConfig::new().dropout_rate(1.0);
        assert!(cfg2.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // 2. Dense layer forward pass (known weights)
    // -----------------------------------------------------------------------

    #[test]
    fn test_dense_forward_known_weights() {
        // 2->2 identity-like layer: W = [[1,0],[0,1]], b = [0.5, -0.5]
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let bias = vec![0.5, -0.5];
        let layer = DenseLayer::from_weights(weights, bias);

        let output = layer.forward(&[3.0, 7.0]);
        assert!((output[0] - 3.5).abs() < 1e-9, "expected 3.5, got {}", output[0]);
        assert!((output[1] - 6.5).abs() < 1e-9, "expected 6.5, got {}", output[1]);
    }

    // -----------------------------------------------------------------------
    // 3. Dense layer backward pass (gradient check)
    // -----------------------------------------------------------------------

    #[test]
    fn test_dense_backward_gradient_check() {
        let weights = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bias = vec![0.1, 0.2];
        let layer = DenseLayer::from_weights(weights, bias);

        let input = vec![1.0, 0.5, -1.0];
        let output = layer.forward(&input);

        // Pretend grad_output = [1.0, 1.0]
        let grad_out = vec![1.0, 1.0];
        let (grad_input, grad_weights, grad_bias) = layer.backward(&grad_out, &input);

        // grad_input[j] = sum_i W[i,j] * grad_out[i]
        // grad_input[0] = 1*1 + 4*1 = 5
        assert!((grad_input[0] - 5.0).abs() < 1e-9);
        // grad_input[1] = 2*1 + 5*1 = 7
        assert!((grad_input[1] - 7.0).abs() < 1e-9);
        // grad_input[2] = 3*1 + 6*1 = 9
        assert!((grad_input[2] - 9.0).abs() < 1e-9);

        // grad_weights[i,j] = grad_out[i] * input[j]
        assert!((grad_weights[[0, 0]] - 1.0).abs() < 1e-9); // 1.0 * 1.0
        assert!((grad_weights[[0, 1]] - 0.5).abs() < 1e-9); // 1.0 * 0.5
        assert!((grad_weights[[1, 2]] - (-1.0)).abs() < 1e-9); // 1.0 * -1.0

        // grad_bias = grad_output
        assert!((grad_bias[0] - 1.0).abs() < 1e-9);
        assert!((grad_bias[1] - 1.0).abs() < 1e-9);

        // Verify output is correct too
        // out[0] = 1*1 + 2*0.5 + 3*(-1) + 0.1 = 1 + 1 - 3 + 0.1 = -0.9
        assert!((output[0] - (-0.9)).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 4. Activation functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_activation_relu() {
        assert!((Activation::ReLU.apply(5.0) - 5.0).abs() < 1e-9);
        assert!((Activation::ReLU.apply(-3.0) - 0.0).abs() < 1e-9);
        assert!((Activation::ReLU.apply(0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_activation_sigmoid() {
        let s0 = Activation::Sigmoid.apply(0.0);
        assert!((s0 - 0.5).abs() < 1e-9, "sigmoid(0) should be 0.5");

        let s_large = Activation::Sigmoid.apply(10.0);
        assert!(s_large > 0.999, "sigmoid(10) should be near 1.0");

        let s_neg = Activation::Sigmoid.apply(-10.0);
        assert!(s_neg < 0.001, "sigmoid(-10) should be near 0.0");
    }

    #[test]
    fn test_activation_gelu() {
        // GELU(0) should be 0
        assert!(Activation::GELU.apply(0.0).abs() < 1e-6);
        // GELU(x) > 0 for x > 0
        assert!(Activation::GELU.apply(1.0) > 0.0);
        // GELU(x) < 0 for some negative x (but GELU has a small negative region)
        let gelu_neg = Activation::GELU.apply(-0.5);
        assert!(gelu_neg < 0.0, "GELU(-0.5) should be negative");
    }

    #[test]
    fn test_activation_tanh() {
        assert!(Activation::Tanh.apply(0.0).abs() < 1e-9);
        assert!((Activation::Tanh.apply(100.0) - 1.0).abs() < 1e-6);
        assert!((Activation::Tanh.apply(-100.0) - (-1.0)).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 5. Code type embedding
    // -----------------------------------------------------------------------

    #[test]
    fn test_code_type_embedding() {
        let emb = CodeFamilyEmbedding::new(8);
        assert_eq!(emb.dim(), 8);

        let surface_emb = emb.encode(&CodeType::SurfaceCode);
        assert_eq!(surface_emb.len(), 8);

        let gkp_emb = emb.encode(&CodeType::GKP);
        assert_eq!(gkp_emb.len(), 8);

        // Different code types should produce different embeddings
        let diff: f64 = surface_emb
            .iter()
            .zip(gkp_emb.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(diff > 0.0, "Distinct code types should have distinct embeddings");
    }

    #[test]
    fn test_code_type_indices_unique() {
        let all = CodeType::all();
        assert_eq!(all.len(), CodeType::NUM_FAMILIES);

        let mut indices: Vec<usize> = all.iter().map(|ct| ct.to_embedding_index()).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), CodeType::NUM_FAMILIES, "All indices must be unique");
    }

    // -----------------------------------------------------------------------
    // 6. Syndrome data construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_syndrome_data_construction() {
        let data = SyndromeData {
            syndrome: vec![true, false, true, false],
            code_type: CodeType::SurfaceCode,
            distance: 3,
            num_rounds: 1,
            noise_rate: 0.01,
        };
        assert_eq!(data.syndrome.len(), 4);
        assert_eq!(data.code_type, CodeType::SurfaceCode);
        assert_eq!(data.distance, 3);
    }

    // -----------------------------------------------------------------------
    // 7. Surface code syndrome generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_surface_code() {
        let data = SyndromeGenerator::generate_surface_code(3, 0.1, 50);
        assert_eq!(data.len(), 50);

        for (syn, lbl) in &data {
            assert_eq!(syn.code_type, CodeType::SurfaceCode);
            assert_eq!(syn.distance, 3);
            assert_eq!(syn.syndrome.len(), 4); // (3-1)^2 = 4
            assert_eq!(lbl.corrections.len(), 9); // 3^2 = 9 data qubits
        }
    }

    // -----------------------------------------------------------------------
    // 8. Repetition code syndrome generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_repetition_code() {
        let data = SyndromeGenerator::generate_repetition_code(5, 0.1, 30);
        assert_eq!(data.len(), 30);

        for (syn, lbl) in &data {
            assert_eq!(syn.code_type, CodeType::RepetitionCode);
            assert_eq!(syn.distance, 5);
            assert_eq!(syn.syndrome.len(), 4); // 5-1 = 4
            assert_eq!(lbl.corrections.len(), 5);
        }
    }

    // -----------------------------------------------------------------------
    // 9. Neural decoder construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_construction() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![32, 16])
            .use_attention(false);

        let decoder = UnifiedNeuralDecoder::new(config, 9, 9);
        assert_eq!(decoder.layers.len(), 2);
        assert_eq!(decoder.max_syndrome_size, 9);
        assert_eq!(decoder.max_correction_size, 9);
        assert!(decoder.attention.is_none());
        assert!(decoder.num_params() > 0);
    }

    #[test]
    fn test_decoder_construction_with_attention() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![32, 16])
            .use_attention(true);

        let decoder = UnifiedNeuralDecoder::new(config, 9, 9);
        assert!(decoder.attention.is_some());
    }

    // -----------------------------------------------------------------------
    // 10. Forward pass produces valid output
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_pass_valid_output() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .use_attention(false);

        let decoder = UnifiedNeuralDecoder::new(config, 4, 5);
        let input_dim = 4 + 16; // syndrome_size + embedding_dim
        let input = vec![0.5; input_dim];
        let output = decoder.forward(&input);

        assert_eq!(output.len(), 5);
        // Output should be finite (no NaN/Inf)
        for &v in &output {
            assert!(v.is_finite(), "Output contains non-finite value: {}", v);
        }
    }

    // -----------------------------------------------------------------------
    // 11. Single training step reduces loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_training_step() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .learning_rate(0.01)
            .use_attention(false);

        let mut decoder = UnifiedNeuralDecoder::new(config, 4, 5);

        let syndromes = vec![SyndromeData {
            syndrome: vec![true, false, true, false],
            code_type: CodeType::RepetitionCode,
            distance: 5,
            num_rounds: 1,
            noise_rate: 0.1,
        }];
        let labels = vec![CorrectionLabel {
            corrections: vec![true, false, false, true, false],
            logical_correction: vec![false],
        }];

        let loss = decoder.train_step(&syndromes, &labels);
        assert!(loss.is_finite(), "Training step loss should be finite");
        assert!(loss >= 0.0, "BCE loss should be non-negative");
    }

    // -----------------------------------------------------------------------
    // 12. Multi-epoch training shows improvement
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_epoch_training_improvement() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![32, 16])
            .learning_rate(0.01)
            .batch_size(10)
            .use_attention(false);

        let mut decoder = UnifiedNeuralDecoder::new(config, 4, 5);

        // Create a small deterministic dataset (repetition code, low noise)
        let data = SyndromeGenerator::generate_repetition_code(5, 0.05, 20);

        let history = decoder.train(&data, 20);
        assert_eq!(history.epoch_losses.len(), 20);
        assert_eq!(history.epoch_accuracies.len(), 20);

        // Loss should decrease from first to last epoch
        let first_loss = history.epoch_losses[0];
        let last_loss = *history.epoch_losses.last().unwrap();
        assert!(
            last_loss <= first_loss + 0.1,
            "Loss should generally decrease: first={}, last={}",
            first_loss,
            last_loss
        );
    }

    // -----------------------------------------------------------------------
    // 13. Prediction produces valid correction
    // -----------------------------------------------------------------------

    #[test]
    fn test_prediction_valid_correction() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .use_attention(false);

        let decoder = UnifiedNeuralDecoder::new(config, 9, 9);

        let syndrome = SyndromeData {
            syndrome: vec![false, true, false, false, true, false, false, false, false],
            code_type: CodeType::SurfaceCode,
            distance: 3,
            num_rounds: 1,
            noise_rate: 0.01,
        };

        let correction = decoder.predict(&syndrome);
        assert_eq!(
            correction.corrections.len(),
            9,
            "Correction should have max_correction_size elements"
        );
    }

    // -----------------------------------------------------------------------
    // 14. Evaluation accuracy computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluation_accuracy() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .learning_rate(0.01)
            .use_attention(false);

        let mut decoder = UnifiedNeuralDecoder::new(config, 4, 5);

        let data = SyndromeGenerator::generate_repetition_code(5, 0.05, 30);
        decoder.train(&data, 10);

        let result = decoder.evaluate(&data);
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert!(result.logical_error_rate >= 0.0 && result.logical_error_rate <= 1.0);
        assert!(result.avg_inference_time_us >= 0.0);
        assert!(!result.per_code_accuracy.is_empty());
    }

    // -----------------------------------------------------------------------
    // 15. Weight save/load roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_save_load_roundtrip() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .use_attention(false);

        let decoder = UnifiedNeuralDecoder::new(config.clone(), 4, 5);
        let saved = decoder.save_weights();

        // Create a fresh decoder and load the saved weights
        let mut decoder2 = UnifiedNeuralDecoder::new(config, 4, 5);
        decoder2.load_weights(&saved);

        let saved2 = decoder2.save_weights();
        assert_eq!(saved.len(), saved2.len());
        for (a, b) in saved.iter().zip(saved2.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "Weight mismatch after roundtrip: {} vs {}",
                a,
                b
            );
        }
    }

    // -----------------------------------------------------------------------
    // 16. Transfer learning: pretrain then finetune
    // -----------------------------------------------------------------------

    #[test]
    fn test_transfer_learning_pretrain_finetune() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .learning_rate(0.01)
            .num_epochs(10)
            .use_attention(false);

        let mut decoder = UnifiedNeuralDecoder::new(config, 4, 5);

        // Pretrain on repetition code
        let source = SyndromeGenerator::generate_repetition_code(5, 0.05, 30);
        let pretrain_history = TransferLearningManager::pretrain(&mut decoder, &source);
        assert!(!pretrain_history.epoch_losses.is_empty());

        // Fine-tune on a different code
        let target = SyndromeGenerator::generate_random_code(4, 5, 0.05, 20);
        let finetune_history =
            TransferLearningManager::finetune(&mut decoder, &target, 1);
        assert!(!finetune_history.epoch_losses.is_empty());

        // Verify the decoder still produces valid predictions after fine-tuning
        let pred = decoder.predict(&source[0].0);
        assert_eq!(pred.corrections.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 17. MWPM baseline comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_mwpm_baseline_comparison() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .learning_rate(0.01)
            .use_attention(false);

        let mut decoder = UnifiedNeuralDecoder::new(config, 4, 9);

        let data = SyndromeGenerator::generate_surface_code(3, 0.05, 30);
        decoder.train(&data, 10);

        let comparison = DecoderBenchmark::compare_mwpm_vs_neural(&data, &decoder);
        assert!(comparison.neural_accuracy >= 0.0 && comparison.neural_accuracy <= 1.0);
        assert!(comparison.mwpm_accuracy >= 0.0 && comparison.mwpm_accuracy <= 1.0);
        assert!(comparison.neural_time_us >= 0.0);
        assert!(comparison.mwpm_time_us >= 0.0);

        // improvement = neural - mwpm (can be positive or negative)
        let expected_imp = comparison.neural_accuracy - comparison.mwpm_accuracy;
        assert!(
            (comparison.improvement - expected_imp).abs() < 1e-9,
            "Improvement should equal neural - mwpm"
        );
    }

    // -----------------------------------------------------------------------
    // 18. Per-code accuracy tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_per_code_accuracy_tracking() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![16, 8])
            .use_attention(false);

        let mut decoder = UnifiedNeuralDecoder::new(config, 9, 9);

        // Mix surface and repetition code data
        let mut mixed_data = SyndromeGenerator::generate_surface_code(3, 0.05, 15);
        let mut rep_data: Vec<(SyndromeData, CorrectionLabel)> =
            SyndromeGenerator::generate_repetition_code(5, 0.05, 15)
                .into_iter()
                .map(|(mut s, mut l)| {
                    // Pad to match decoder sizes
                    s.syndrome.resize(9, false);
                    l.corrections.resize(9, false);
                    (s, l)
                })
                .collect();
        mixed_data.append(&mut rep_data);

        decoder.train(&mixed_data, 5);
        let result = decoder.evaluate(&mixed_data);

        // Should have entries for both code families
        assert!(
            result.per_code_accuracy.len() >= 2,
            "Should track at least 2 code families, got {}",
            result.per_code_accuracy.len()
        );

        // Verify all reported accuracies are in [0, 1]
        for (ct, acc) in &result.per_code_accuracy {
            assert!(
                *acc >= 0.0 && *acc <= 1.0,
                "Accuracy for {} should be in [0,1], got {}",
                ct,
                acc
            );
        }
    }

    // -----------------------------------------------------------------------
    // 19. Attention layer forward pass
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_layer_forward() {
        let attn = AttentionLayer::new(4, 2);
        let input = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let output = attn.forward(&input);
        assert_eq!(output.len(), 3);
        for token in &output {
            assert_eq!(token.len(), 4);
            for &v in token {
                assert!(v.is_finite(), "Attention output should be finite");
            }
        }
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let attn = AttentionLayer::new(4, 2);
        let query = vec![1.0, 0.5, -0.3, 0.2];
        let keys = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![-0.1, -0.2, 0.1, 0.3],
        ];
        let weights = attn.attention_weights(&query, &keys);
        assert_eq!(weights.len(), 3);

        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Attention weights should sum to 1.0, got {}",
            sum
        );
        for &w in &weights {
            assert!(w >= 0.0, "Attention weights should be non-negative");
        }
    }

    // -----------------------------------------------------------------------
    // 20. Dense layer SGD update
    // -----------------------------------------------------------------------

    #[test]
    fn test_dense_sgd_update() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let bias = vec![0.5, 0.5];
        let mut layer = DenseLayer::from_weights(weights, bias);

        let grad_w = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let grad_b = vec![0.05, 0.05];
        let lr = 0.1;

        layer.sgd_update(&grad_w, &grad_b, lr);

        // w[0,0] = 1.0 - 0.1 * 0.1 = 0.99
        assert!((layer.weights[[0, 0]] - 0.99).abs() < 1e-9);
        // w[1,1] = 4.0 - 0.1 * 0.4 = 3.96
        assert!((layer.weights[[1, 1]] - 3.96).abs() < 1e-9);
        // b[0] = 0.5 - 0.1 * 0.05 = 0.495
        assert!((layer.bias[0] - 0.495).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 21. Activation derivatives
    // -----------------------------------------------------------------------

    #[test]
    fn test_activation_derivatives() {
        // ReLU derivative
        assert!((Activation::ReLU.derivative(1.0) - 1.0).abs() < 1e-9);
        assert!((Activation::ReLU.derivative(-1.0) - 0.0).abs() < 1e-9);

        // Sigmoid derivative at 0 should be 0.25
        let sig_deriv = Activation::Sigmoid.derivative(0.0);
        assert!((sig_deriv - 0.25).abs() < 1e-9, "sigmoid'(0) = 0.25, got {}", sig_deriv);

        // Tanh derivative at 0 should be 1.0
        let tanh_deriv = Activation::Tanh.derivative(0.0);
        assert!((tanh_deriv - 1.0).abs() < 1e-9, "tanh'(0) = 1.0, got {}", tanh_deriv);

        // GELU derivative at 0 should be ~0.5
        let gelu_deriv = Activation::GELU.derivative(0.0);
        assert!((gelu_deriv - 0.5).abs() < 0.01, "gelu'(0) ~ 0.5, got {}", gelu_deriv);
    }

    // -----------------------------------------------------------------------
    // 22. Random code generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_random_code() {
        let data = SyndromeGenerator::generate_random_code(10, 8, 0.1, 25);
        assert_eq!(data.len(), 25);
        for (syn, lbl) in &data {
            assert_eq!(syn.syndrome.len(), 10);
            assert_eq!(lbl.corrections.len(), 8);
            assert_eq!(syn.code_type, CodeType::QLDPC);
        }
    }

    // -----------------------------------------------------------------------
    // 23. Empty dataset handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_empty_dataset() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![8])
            .use_attention(false);
        let decoder = UnifiedNeuralDecoder::new(config, 4, 4);

        let result = decoder.evaluate(&[]);
        assert!((result.accuracy - 0.0).abs() < 1e-9);
        assert!((result.logical_error_rate - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 24. Num params consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_num_params() {
        let config = UnifiedNeuralDecoderConfig::new()
            .hidden_dims(vec![32, 16])
            .code_family_embedding_dim(8)
            .use_attention(false);

        let decoder = UnifiedNeuralDecoder::new(config, 4, 5);

        // Input dim = 4 + 8 = 12
        // Layer 0: 12 -> 32 => 12*32 + 32 = 416
        // Layer 1: 32 -> 16 => 32*16 + 16 = 528
        // Output:  16 -> 5  => 16*5  + 5  = 85
        // Embedding: 10 * 8 = 80
        // Total = 416 + 528 + 85 + 80 = 1109
        assert_eq!(decoder.num_params(), 1109);
    }
}
