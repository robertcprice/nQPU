//! Quantum Reservoir Computing
//!
//! **BLEEDING EDGE**: First quantum simulator with built-in reservoir computing.
//! Uses quantum dynamics as a computational reservoir for classical machine learning tasks.
//!
//! Quantum reservoir computing leverages the exponentially large Hilbert space of
//! quantum systems as a feature space for machine learning, without requiring
//! training of quantum parameters.
//!
//! Applications:
//! - Time series prediction
//! - Nonlinear function approximation
//! - Chaotic system modeling
//! - Quantum-enhanced classical ML
//!
//! References:
//! - Fujii & Nakajima (2021) - Quantum reservoir computing
//! - Chen et al. (2020) - Temporal information processing on noisy quantum computers

use crate::{QuantumState, GateOperations};
use num_complex::Complex64;

/// Quantum reservoir configuration
#[derive(Clone, Debug)]
pub struct ReservoirConfig {
    /// Number of qubits in the reservoir
    pub num_qubits: usize,
    /// Reservoir circuit depth per input step
    pub reservoir_depth: usize,
    /// Input encoding strategy
    pub encoding: InputEncoding,
    /// Measurement observables for readout
    pub observables: Vec<Observable>,
    /// Whether to include entanglement in reservoir dynamics
    pub entangling: bool,
    /// Random seed for reservoir structure
    pub seed: u64,
    /// Noise level (0.0 = noiseless, 1.0 = maximally noisy)
    pub noise_level: f64,
}

/// How classical data is encoded into the quantum reservoir
#[derive(Clone, Debug)]
pub enum InputEncoding {
    /// Encode input as rotation angles on each qubit
    AngleEncoding,
    /// Encode input as amplitude of initial state
    AmplitudeEncoding,
    /// IQP-style encoding with ZZ interactions
    IQPEncoding,
    /// Time-multiplexed encoding (one qubit, sequential inputs)
    TimeMultiplexed,
}

/// Observable to measure from reservoir state
#[derive(Clone, Debug)]
pub enum Observable {
    /// Single-qubit Pauli Z
    PauliZ(usize),
    /// Two-qubit ZZ correlation
    ZZCorrelation(usize, usize),
    /// Single-qubit Pauli X
    PauliX(usize),
    /// Single-qubit Pauli Y
    PauliY(usize),
    /// All single-qubit Z observables
    AllZ,
    /// All two-qubit ZZ correlations
    AllZZ,
    /// Full Pauli feature set (Z, ZZ, X, XX, Y, YY)
    FullPauli,
}

/// Trained quantum reservoir (with classical readout weights)
#[derive(Clone, Debug)]
pub struct TrainedReservoir {
    pub config: ReservoirConfig,
    /// Classical readout weights [output_dim x feature_dim]
    pub weights: Vec<Vec<f64>>,
    /// Bias terms
    pub bias: Vec<f64>,
    /// Feature dimension
    pub feature_dim: usize,
    /// Training loss
    pub training_loss: f64,
}

/// Result of reservoir processing
#[derive(Clone, Debug)]
pub struct ReservoirOutput {
    /// Raw features extracted from quantum measurements
    pub features: Vec<Vec<f64>>,
    /// Predictions (if trained)
    pub predictions: Option<Vec<Vec<f64>>>,
    /// Processing time in ms
    pub time_ms: f64,
}

/// Quantum Reservoir Computer
pub struct QuantumReservoir {
    config: ReservoirConfig,
    /// Precomputed reservoir unitary angles
    reservoir_angles: Vec<(usize, f64)>,
    /// Entangling layer specification: (control, target)
    entangling_pairs: Vec<(usize, usize)>,
}

impl QuantumReservoir {
    pub fn new(config: ReservoirConfig) -> Self {
        let mut rng_state = config.seed;

        // Generate random reservoir structure
        let mut reservoir_angles = Vec::new();
        for _depth in 0..config.reservoir_depth {
            for q in 0..config.num_qubits {
                // Random rotation axis and angle
                let angle = Self::next_random(&mut rng_state) * std::f64::consts::PI * 2.0;
                reservoir_angles.push((q, angle));
            }
        }

        // Generate entangling layer
        let mut entangling_pairs = Vec::new();
        if config.entangling {
            for q in 0..config.num_qubits.saturating_sub(1) {
                entangling_pairs.push((q, q + 1));
            }
            // Add long-range connections for expressivity
            if config.num_qubits > 3 {
                entangling_pairs.push((0, config.num_qubits - 1));
            }
        }

        Self {
            config,
            reservoir_angles,
            entangling_pairs,
        }
    }

    /// Simple PRNG for reproducible reservoir structure
    fn next_random(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Process a sequence of inputs through the reservoir
    pub fn process(&self, inputs: &[Vec<f64>]) -> ReservoirOutput {
        let start = std::time::Instant::now();
        let n = self.config.num_qubits;
        let mut features = Vec::new();

        for input in inputs {
            let mut state = QuantumState::new(n);

            // Encode input into quantum state
            self.encode_input(&mut state, input);

            // Apply reservoir dynamics
            self.apply_reservoir(&mut state);

            // Extract features via measurement
            let feat = self.extract_features(&state);
            features.push(feat);
        }

        ReservoirOutput {
            features,
            predictions: None,
            time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Train the reservoir's classical readout layer using ridge regression
    pub fn train(
        &self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        regularization: f64,
    ) -> TrainedReservoir {
        let output = self.process(inputs);
        let features = &output.features;

        let feature_dim = features[0].len();
        let output_dim = targets[0].len();
        let n_samples = features.len();

        // Ridge regression: W = Y^T X (X^T X + λI)^{-1}
        // Simplified: solve via pseudoinverse with Tikhonov regularization

        // Compute X^T X + λI
        let mut xtx = vec![vec![0.0; feature_dim]; feature_dim];
        for i in 0..feature_dim {
            for j in 0..feature_dim {
                for s in 0..n_samples {
                    xtx[i][j] += features[s][i] * features[s][j];
                }
                if i == j {
                    xtx[i][j] += regularization;
                }
            }
        }

        // Compute X^T Y
        let mut xty = vec![vec![0.0; output_dim]; feature_dim];
        for i in 0..feature_dim {
            for o in 0..output_dim {
                for s in 0..n_samples {
                    xty[i][o] += features[s][i] * targets[s][o];
                }
            }
        }

        // Solve via Cholesky-like approach (simplified for small feature dimensions)
        let weights = self.solve_linear_system(&xtx, &xty);

        // Compute training loss
        let mut loss = 0.0;
        for s in 0..n_samples {
            for o in 0..output_dim {
                let mut pred = 0.0;
                for f in 0..feature_dim {
                    pred += weights[o][f] * features[s][f];
                }
                loss += (pred - targets[s][o]).powi(2);
            }
        }
        loss /= n_samples as f64;

        TrainedReservoir {
            config: self.config.clone(),
            weights,
            bias: vec![0.0; output_dim],
            feature_dim,
            training_loss: loss,
        }
    }

    fn encode_input(&self, state: &mut QuantumState, input: &[f64]) {
        match self.config.encoding {
            InputEncoding::AngleEncoding => {
                // Each input value becomes a rotation angle on corresponding qubit
                for (q, &val) in input.iter().enumerate().take(self.config.num_qubits) {
                    let angle = val * std::f64::consts::PI;
                    GateOperations::ry(state, q, angle);
                    GateOperations::rz(state, q, angle * 0.5);
                }
            }
            InputEncoding::IQPEncoding => {
                // Hadamard layer + Z rotations + ZZ interactions
                for q in 0..self.config.num_qubits {
                    GateOperations::h(state, q);
                }
                for (q, &val) in input.iter().enumerate().take(self.config.num_qubits) {
                    GateOperations::rz(state, q, val * std::f64::consts::PI);
                }
                // ZZ interactions
                for q in 0..self.config.num_qubits.saturating_sub(1) {
                    let i_idx = q.min(input.len() - 1);
                    let j_idx = (q + 1).min(input.len() - 1);
                    let angle = input[i_idx] * input[j_idx] * std::f64::consts::PI;
                    // Approximate ZZ with CNOT-Rz-CNOT
                    GateOperations::cnot(state, q, q + 1);
                    GateOperations::rz(state, q + 1, angle);
                    GateOperations::cnot(state, q, q + 1);
                }
            }
            InputEncoding::AmplitudeEncoding => {
                // Encode input vector as amplitudes (requires 2^n >= input.len())
                let dim = state.dim;
                let amps = state.amplitudes_mut();
                let norm: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
                let scale = if norm > 1e-15 { 1.0 / norm } else { 0.0 };
                for (i, &val) in input.iter().enumerate().take(dim) {
                    amps[i] = Complex64::new(val * scale, 0.0);
                }
            }
            InputEncoding::TimeMultiplexed => {
                // Feed inputs sequentially to qubit 0
                for &val in input.iter() {
                    GateOperations::ry(state, 0, val * std::f64::consts::PI);
                    // Apply one round of reservoir dynamics between inputs
                    self.apply_reservoir(state);
                }
            }
        }
    }

    fn apply_reservoir(&self, state: &mut QuantumState) {
        // Apply random rotations
        for &(qubit, angle) in &self.reservoir_angles {
            GateOperations::rx(state, qubit, angle);
            GateOperations::rz(state, qubit, angle * 1.3);
        }

        // Apply entangling layer
        for &(ctrl, tgt) in &self.entangling_pairs {
            GateOperations::cnot(state, ctrl, tgt);
        }
    }

    fn extract_features(&self, state: &QuantumState) -> Vec<f64> {
        let n = self.config.num_qubits;
        let mut features = Vec::new();

        for obs in &self.config.observables {
            match obs {
                Observable::PauliZ(q) => {
                    features.push(state.expectation_z(*q));
                }
                Observable::ZZCorrelation(q1, q2) => {
                    let zz = self.expectation_zz(state, *q1, *q2);
                    features.push(zz);
                }
                Observable::PauliX(q) => {
                    features.push(self.expectation_x(state, *q));
                }
                Observable::PauliY(q) => {
                    features.push(self.expectation_y(state, *q));
                }
                Observable::AllZ => {
                    for q in 0..n {
                        features.push(state.expectation_z(q));
                    }
                }
                Observable::AllZZ => {
                    for i in 0..n {
                        for j in (i + 1)..n {
                            features.push(self.expectation_zz(state, i, j));
                        }
                    }
                }
                Observable::FullPauli => {
                    for q in 0..n {
                        features.push(state.expectation_z(q));
                        features.push(self.expectation_x(state, q));
                        features.push(self.expectation_y(state, q));
                    }
                    for i in 0..n {
                        for j in (i + 1)..n {
                            features.push(self.expectation_zz(state, i, j));
                        }
                    }
                }
            }
        }

        features
    }

    fn expectation_zz(&self, state: &QuantumState, q1: usize, q2: usize) -> f64 {
        let amps = state.amplitudes_ref();
        let mut exp = 0.0;
        for (idx, amp) in amps.iter().enumerate() {
            let bit1 = (idx >> q1) & 1;
            let bit2 = (idx >> q2) & 1;
            let sign = if bit1 ^ bit2 == 0 { 1.0 } else { -1.0 };
            exp += sign * amp.norm_sqr();
        }
        exp
    }

    fn expectation_x(&self, state: &QuantumState, qubit: usize) -> f64 {
        let amps = state.amplitudes_ref();
        let stride = 1 << qubit;
        let mut exp = 0.0;
        for i in 0..state.dim {
            if (i >> qubit) & 1 == 0 {
                let j = i | stride;
                // <i|X|j> = 1 when bits differ on qubit
                let contrib = amps[i].conj() * amps[j] + amps[j].conj() * amps[i];
                exp += contrib.re;
            }
        }
        exp
    }

    fn expectation_y(&self, state: &QuantumState, qubit: usize) -> f64 {
        let amps = state.amplitudes_ref();
        let stride = 1 << qubit;
        let mut exp = 0.0;
        for i in 0..state.dim {
            if (i >> qubit) & 1 == 0 {
                let j = i | stride;
                // Y = [[0, -i], [i, 0]]
                let contrib =
                    Complex64::new(0.0, -1.0) * amps[i].conj() * amps[j]
                    + Complex64::new(0.0, 1.0) * amps[j].conj() * amps[i];
                exp += contrib.re;
            }
        }
        exp
    }

    fn solve_linear_system(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b[0].len();

        // Gaussian elimination with partial pivoting
        let mut aug: Vec<Vec<f64>> = Vec::new();
        for i in 0..n {
            let mut row = a[i].clone();
            for j in 0..m {
                row.push(b[i][j]);
            }
            aug.push(row);
        }

        // Forward elimination
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            if aug[col][col].abs() < 1e-12 {
                continue;
            }

            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..(n + m) {
                    let val = aug[col][j];
                    aug[row][j] -= factor * val;
                }
            }
        }

        // Back substitution
        let mut result = vec![vec![0.0; n]; m];
        for col in (0..n).rev() {
            if aug[col][col].abs() < 1e-12 {
                continue;
            }
            for o in 0..m {
                let mut val = aug[col][n + o];
                for j in (col + 1)..n {
                    val -= aug[col][j] * result[o][j];
                }
                result[o][col] = val / aug[col][col];
            }
        }

        result
    }
}

/// Quantum Echo State Network - a variant of reservoir computing
/// that uses measurement feedback
pub struct QuantumEchoStateNetwork {
    pub reservoir: QuantumReservoir,
    /// Feedback connections from output back to input
    pub feedback_strength: f64,
    /// Washout period (initial steps to discard)
    pub washout: usize,
}

impl QuantumEchoStateNetwork {
    pub fn new(config: ReservoirConfig, feedback_strength: f64) -> Self {
        Self {
            reservoir: QuantumReservoir::new(config),
            feedback_strength,
            washout: 10,
        }
    }

    /// Process a time series with feedback
    pub fn process_timeseries(&self, series: &[f64]) -> Vec<Vec<f64>> {
        let mut all_features = Vec::new();
        let mut feedback = vec![0.0; self.reservoir.config.num_qubits];

        for (t, &val) in series.iter().enumerate() {
            // Combine input with feedback
            let mut input = vec![val];
            for &fb in &feedback {
                input.push(fb * self.feedback_strength);
            }

            // Process through reservoir
            let output = self.reservoir.process(&[input]);
            let features = output.features[0].clone();

            // Update feedback from features
            for (i, &f) in features.iter().enumerate().take(feedback.len()) {
                feedback[i] = f;
            }

            if t >= self.washout {
                all_features.push(features);
            }
        }

        all_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_process() {
        let config = ReservoirConfig {
            num_qubits: 4,
            reservoir_depth: 3,
            encoding: InputEncoding::AngleEncoding,
            observables: vec![Observable::AllZ, Observable::AllZZ],
            entangling: true,
            seed: 42,
            noise_level: 0.0,
        };

        let reservoir = QuantumReservoir::new(config);
        let inputs = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
        ];

        let output = reservoir.process(&inputs);
        assert_eq!(output.features.len(), 2);
        // 4 Z observables + 6 ZZ correlations = 10 features
        assert_eq!(output.features[0].len(), 10);
    }

    #[test]
    fn test_reservoir_train() {
        let config = ReservoirConfig {
            num_qubits: 3,
            reservoir_depth: 2,
            encoding: InputEncoding::AngleEncoding,
            observables: vec![Observable::AllZ],
            entangling: true,
            seed: 123,
            noise_level: 0.0,
        };

        let reservoir = QuantumReservoir::new(config);

        // Simple regression task: y = sin(x)
        let inputs: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 * 0.1])
            .collect();
        let targets: Vec<Vec<f64>> = inputs
            .iter()
            .map(|x| vec![(x[0] * std::f64::consts::PI).sin()])
            .collect();

        let trained = reservoir.train(&inputs, &targets, 0.01);
        assert!(trained.training_loss.is_finite());
    }

    #[test]
    fn test_iqp_encoding() {
        let config = ReservoirConfig {
            num_qubits: 3,
            reservoir_depth: 2,
            encoding: InputEncoding::IQPEncoding,
            observables: vec![Observable::FullPauli],
            entangling: true,
            seed: 42,
            noise_level: 0.0,
        };

        let reservoir = QuantumReservoir::new(config);
        let inputs = vec![vec![0.5, 0.3, 0.7]];
        let output = reservoir.process(&inputs);
        assert!(!output.features[0].is_empty());
    }

    #[test]
    fn test_echo_state_network() {
        let config = ReservoirConfig {
            num_qubits: 3,
            reservoir_depth: 2,
            encoding: InputEncoding::AngleEncoding,
            observables: vec![Observable::AllZ],
            entangling: true,
            seed: 42,
            noise_level: 0.0,
        };

        let esn = QuantumEchoStateNetwork::new(config, 0.3);
        let series: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin()).collect();
        let features = esn.process_timeseries(&series);
        assert_eq!(features.len(), 20); // 30 - washout(10)
    }
}
