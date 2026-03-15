//! Quantum Kernel Methods for Machine Learning
//!
//! This module implements quantum kernel methods for classification tasks,
//! including quantum feature maps, kernel matrix computation, and a simplified
//! Quantum Support Vector Machine (QSVM) using Sequential Minimal Optimization.
//!
//! # Overview
//!
//! Quantum kernel methods embed classical data into quantum Hilbert space via
//! parameterized circuits (feature maps), then compute inner products (kernel values)
//! between quantum states. These kernels can be used in classical SVMs, giving a
//! quantum advantage for certain data distributions.
//!
//! # Key Components
//!
//! 1. **Feature Maps**: ZZFeatureMap, PauliFeatureMap, AngleEncoding, AmplitudeEncoding, IQPEncoding
//! 2. **Kernel Computation**: Fidelity kernel K(x,y) = |<φ(x)|φ(y)>|²
//! 3. **QSVM**: Simplified SMO solver with quantum kernel
//! 4. **Kernel Analysis**: Alignment, concentration, effective dimension
//! 5. **Data Generation**: Moons, circles, XOR datasets for testing
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::quantum_kernels::*;
//!
//! let config = KernelConfig::default()
//!     .with_feature_map(FeatureMap::AngleEncoding { num_qubits: 2 });
//!
//! let data = vec![vec![0.5, 0.3], vec![1.2, 0.8], vec![0.1, 1.5]];
//! let km = compute_kernel_matrix(&data, &config).unwrap();
//! assert!((km.matrix[[0, 0]] - 1.0).abs() < 1e-10);
//! ```

use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from quantum kernel operations.
#[derive(Debug, Clone)]
pub enum KernelError {
    /// Data dimensions do not match the feature map requirements.
    DimensionMismatch { expected: usize, got: usize },
    /// Feature map encoding failed.
    EncodingFailed(String),
    /// SVM optimization did not converge within the iteration limit.
    SvmConvergenceFailed { iterations: usize, tolerance: f64 },
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            KernelError::EncodingFailed(msg) => write!(f, "Encoding failed: {}", msg),
            KernelError::SvmConvergenceFailed {
                iterations,
                tolerance,
            } => write!(
                f,
                "SVM did not converge after {} iterations (tol={})",
                iterations, tolerance
            ),
        }
    }
}

impl std::error::Error for KernelError {}

// ============================================================
// FEATURE MAP TYPES
// ============================================================

/// Quantum feature map for encoding classical data into quantum states.
#[derive(Clone, Debug)]
pub enum FeatureMap {
    /// ZZ feature map with entangling ZZ interactions.
    /// Applies H^n, RZ(x_i) on each qubit, RZZ(x_i * x_j) on all pairs.
    ZZFeatureMap { num_qubits: usize, reps: usize },
    /// Configurable Pauli feature map with chosen rotation axes.
    PauliFeatureMap {
        num_qubits: usize,
        /// Pauli axis per qubit: 0=X, 1=Y, 2=Z
        paulis: Vec<u8>,
        reps: usize,
    },
    /// Angle encoding: RY(x_i) on qubit i. n features = n qubits.
    AngleEncoding { num_qubits: usize },
    /// Amplitude encoding: encode normalized data as state amplitudes.
    AmplitudeEncoding { num_qubits: usize },
    /// IQP (Instantaneous Quantum Polynomial) encoding with diagonal gates.
    IQPEncoding { num_qubits: usize, reps: usize },
}

impl FeatureMap {
    /// Return the number of qubits for this feature map.
    pub fn num_qubits(&self) -> usize {
        match self {
            FeatureMap::ZZFeatureMap { num_qubits, .. }
            | FeatureMap::PauliFeatureMap { num_qubits, .. }
            | FeatureMap::AngleEncoding { num_qubits }
            | FeatureMap::AmplitudeEncoding { num_qubits }
            | FeatureMap::IQPEncoding { num_qubits, .. } => *num_qubits,
        }
    }

    /// Return a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            FeatureMap::ZZFeatureMap { .. } => "ZZFeatureMap",
            FeatureMap::PauliFeatureMap { .. } => "PauliFeatureMap",
            FeatureMap::AngleEncoding { .. } => "AngleEncoding",
            FeatureMap::AmplitudeEncoding { .. } => "AmplitudeEncoding",
            FeatureMap::IQPEncoding { .. } => "IQPEncoding",
        }
    }
}

// ============================================================
// KERNEL TYPES AND CONFIG
// ============================================================

/// Type of quantum kernel to compute.
#[derive(Clone, Debug)]
pub enum KernelType {
    /// Fidelity (overlap) kernel: K(x,y) = |<phi(x)|phi(y)>|^2
    Fidelity,
    /// Projected kernel: project onto observable subspace, then classical kernel.
    Projected {
        /// Qubit indices to measure for projection.
        observables: Vec<usize>,
    },
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::Fidelity
    }
}

/// Configuration for kernel matrix computation.
#[derive(Clone, Debug)]
pub struct KernelConfig {
    /// Feature map circuit to use.
    pub feature_map: FeatureMap,
    /// Type of kernel (fidelity or projected).
    pub kernel_type: KernelType,
    /// Number of measurement shots (None = exact statevector).
    pub shots: Option<usize>,
    /// Whether to normalize the kernel matrix.
    pub normalize: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        KernelConfig {
            feature_map: FeatureMap::AngleEncoding { num_qubits: 2 },
            kernel_type: KernelType::Fidelity,
            shots: None,
            normalize: true,
        }
    }
}

impl KernelConfig {
    /// Set the feature map.
    pub fn with_feature_map(mut self, fm: FeatureMap) -> Self {
        self.feature_map = fm;
        self
    }

    /// Set the kernel type.
    pub fn with_kernel_type(mut self, kt: KernelType) -> Self {
        self.kernel_type = kt;
        self
    }

    /// Set the number of shots.
    pub fn with_shots(mut self, shots: Option<usize>) -> Self {
        self.shots = shots;
        self
    }

    /// Set normalization.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Computed kernel matrix with metadata.
#[derive(Clone, Debug)]
pub struct KernelMatrix {
    /// The N x N kernel matrix.
    pub matrix: Array2<f64>,
    /// Number of data samples used to build the matrix.
    pub num_samples: usize,
    /// Name of the feature map used.
    pub feature_map_name: String,
}

// ============================================================
// QSVM TYPES
// ============================================================

/// Configuration for QSVM training.
#[derive(Clone, Debug)]
pub struct QsvmConfig {
    /// Kernel configuration.
    pub kernel_config: KernelConfig,
    /// SVM regularization parameter C.
    pub c_parameter: f64,
    /// Maximum SMO iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for QsvmConfig {
    fn default() -> Self {
        QsvmConfig {
            kernel_config: KernelConfig::default(),
            c_parameter: 1.0,
            max_iterations: 1000,
            tolerance: 1e-4,
        }
    }
}

impl QsvmConfig {
    /// Set the kernel configuration.
    pub fn with_kernel_config(mut self, kc: KernelConfig) -> Self {
        self.kernel_config = kc;
        self
    }

    /// Set the C parameter.
    pub fn with_c_parameter(mut self, c: f64) -> Self {
        self.c_parameter = c;
        self
    }

    /// Set the maximum iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

/// Result of QSVM training.
#[derive(Clone, Debug)]
pub struct QsvmResult {
    /// Indices of support vectors in the training set.
    pub support_vectors: Vec<usize>,
    /// Lagrange multipliers (dual variables).
    pub alphas: Vec<f64>,
    /// Bias term.
    pub bias: f64,
    /// Training accuracy.
    pub accuracy: f64,
    /// Number of support vectors.
    pub num_support_vectors: usize,
}

// ============================================================
// STATE VECTOR UTILITIES
// ============================================================

/// Create a |0...0> state for n qubits.
pub fn zero_state(num_qubits: usize) -> Vec<Complex64> {
    let dim = 1 << num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);
    state
}

/// Apply a Hadamard gate to the given qubit.
pub fn apply_h(state: &mut Vec<Complex64>, qubit: usize) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let step = 1 << (num_qubits - 1 - qubit);
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;

    let mut i = 0;
    while i < dim {
        for j in i..(i + step) {
            let k = j + step;
            let a = state[j];
            let b = state[k];
            state[j] = Complex64::new(inv_sqrt2, 0.0) * (a + b);
            state[k] = Complex64::new(inv_sqrt2, 0.0) * (a - b);
        }
        i += step << 1;
    }
}

/// Apply RX(angle) gate to the given qubit.
pub fn apply_rx(state: &mut Vec<Complex64>, qubit: usize, angle: f64) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let step = 1 << (num_qubits - 1 - qubit);
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();

    let mut i = 0;
    while i < dim {
        for j in i..(i + step) {
            let k = j + step;
            let a = state[j];
            let b = state[k];
            state[j] = Complex64::new(cos_half, 0.0) * a + Complex64::new(0.0, -sin_half) * b;
            state[k] = Complex64::new(0.0, -sin_half) * a + Complex64::new(cos_half, 0.0) * b;
        }
        i += step << 1;
    }
}

/// Apply RY(angle) gate to the given qubit.
pub fn apply_ry(state: &mut Vec<Complex64>, qubit: usize, angle: f64) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let step = 1 << (num_qubits - 1 - qubit);
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();

    let mut i = 0;
    while i < dim {
        for j in i..(i + step) {
            let k = j + step;
            let a = state[j];
            let b = state[k];
            state[j] = Complex64::new(cos_half, 0.0) * a - Complex64::new(sin_half, 0.0) * b;
            state[k] = Complex64::new(sin_half, 0.0) * a + Complex64::new(cos_half, 0.0) * b;
        }
        i += step << 1;
    }
}

/// Apply RZ(angle) gate to the given qubit.
pub fn apply_rz(state: &mut Vec<Complex64>, qubit: usize, angle: f64) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let step = 1 << (num_qubits - 1 - qubit);
    let phase_0 = Complex64::new((-angle / 2.0).cos(), (-angle / 2.0).sin());
    let phase_1 = Complex64::new((angle / 2.0).cos(), (angle / 2.0).sin());

    let mut i = 0;
    while i < dim {
        for j in i..(i + step) {
            let k = j + step;
            state[j] = phase_0 * state[j];
            state[k] = phase_1 * state[k];
        }
        i += step << 1;
    }
}

/// Apply RZZ(angle) gate to the given qubit pair.
///
/// RZZ(θ) = exp(-iθ/2 Z⊗Z) = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2})
pub fn apply_rzz(state: &mut Vec<Complex64>, q0: usize, q1: usize, angle: f64) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let phase_same = Complex64::new((-angle / 2.0).cos(), (-angle / 2.0).sin());
    let phase_diff = Complex64::new((angle / 2.0).cos(), (angle / 2.0).sin());

    let bit0 = num_qubits - 1 - q0;
    let bit1 = num_qubits - 1 - q1;

    for idx in 0..dim {
        let b0 = (idx >> bit0) & 1;
        let b1 = (idx >> bit1) & 1;
        if b0 == b1 {
            state[idx] = phase_same * state[idx];
        } else {
            state[idx] = phase_diff * state[idx];
        }
    }
}

/// Apply CNOT (CX) gate with given control and target qubits.
pub fn apply_cx(state: &mut Vec<Complex64>, control: usize, target: usize) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let ctrl_bit = num_qubits - 1 - control;
    let tgt_bit = num_qubits - 1 - target;

    for idx in 0..dim {
        if ((idx >> ctrl_bit) & 1) == 1 && ((idx >> tgt_bit) & 1) == 0 {
            let partner = idx ^ (1 << tgt_bit);
            state.swap(idx, partner);
        }
    }
}

/// Compute the overlap |<a|b>|^2 between two state vectors.
pub fn state_overlap(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len(), "State vectors must have equal length");
    let inner: Complex64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai.conj() * bi).sum();
    inner.norm_sqr()
}

// ============================================================
// FEATURE MAP ENCODING
// ============================================================

/// Encode classical data using the specified feature map circuit.
///
/// Returns the resulting quantum state vector.
pub fn encode_feature_map(
    data: &[f64],
    feature_map: &FeatureMap,
) -> Result<Vec<Complex64>, KernelError> {
    match feature_map {
        FeatureMap::AngleEncoding { num_qubits } => {
            if data.len() > *num_qubits {
                return Err(KernelError::DimensionMismatch {
                    expected: *num_qubits,
                    got: data.len(),
                });
            }
            let mut state = zero_state(*num_qubits);
            for (i, &x) in data.iter().enumerate() {
                apply_ry(&mut state, i, x);
            }
            Ok(state)
        }

        FeatureMap::ZZFeatureMap { num_qubits, reps } => {
            if data.len() > *num_qubits {
                return Err(KernelError::DimensionMismatch {
                    expected: *num_qubits,
                    got: data.len(),
                });
            }
            let n = *num_qubits;
            let mut state = zero_state(n);

            for _rep in 0..*reps {
                // Layer of Hadamards
                for q in 0..n {
                    apply_h(&mut state, q);
                }
                // RZ(x_i) on each qubit
                for (i, &x) in data.iter().enumerate() {
                    apply_rz(&mut state, i, x);
                }
                // RZZ(x_i * x_j) on all pairs
                for i in 0..data.len() {
                    for j in (i + 1)..data.len() {
                        apply_rzz(&mut state, i, j, data[i] * data[j]);
                    }
                }
            }
            Ok(state)
        }

        FeatureMap::PauliFeatureMap {
            num_qubits,
            paulis,
            reps,
        } => {
            if data.len() > *num_qubits {
                return Err(KernelError::DimensionMismatch {
                    expected: *num_qubits,
                    got: data.len(),
                });
            }
            let n = *num_qubits;
            let mut state = zero_state(n);

            for _rep in 0..*reps {
                // Layer of Hadamards
                for q in 0..n {
                    apply_h(&mut state, q);
                }
                // Pauli rotations parameterized by data
                for (i, &x) in data.iter().enumerate() {
                    let axis = if i < paulis.len() { paulis[i] } else { 2 };
                    match axis {
                        0 => apply_rx(&mut state, i, x),
                        1 => apply_ry(&mut state, i, x),
                        _ => apply_rz(&mut state, i, x),
                    }
                }
            }
            Ok(state)
        }

        FeatureMap::AmplitudeEncoding { num_qubits } => {
            let dim = 1 << *num_qubits;
            if data.len() > dim {
                return Err(KernelError::DimensionMismatch {
                    expected: dim,
                    got: data.len(),
                });
            }
            // Normalize the data vector
            let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return Err(KernelError::EncodingFailed(
                    "Cannot amplitude-encode a zero vector".to_string(),
                ));
            }
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            for (i, &x) in data.iter().enumerate() {
                state[i] = Complex64::new(x / norm, 0.0);
            }
            Ok(state)
        }

        FeatureMap::IQPEncoding { num_qubits, reps } => {
            if data.len() > *num_qubits {
                return Err(KernelError::DimensionMismatch {
                    expected: *num_qubits,
                    got: data.len(),
                });
            }
            let n = *num_qubits;
            let mut state = zero_state(n);

            for _rep in 0..*reps {
                // Layer of Hadamards
                for q in 0..n {
                    apply_h(&mut state, q);
                }
                // Diagonal RZ(x_i) gates
                for (i, &x) in data.iter().enumerate() {
                    apply_rz(&mut state, i, x);
                }
                // Quadratic diagonal RZZ(x_i * x_j) on pairs
                for i in 0..data.len() {
                    for j in (i + 1)..data.len() {
                        apply_rzz(&mut state, i, j, data[i] * data[j]);
                    }
                }
                // Final Hadamard layer
                for q in 0..n {
                    apply_h(&mut state, q);
                }
            }
            Ok(state)
        }
    }
}

/// Convenience wrapper: encode data and return the state vector.
pub fn encode_and_get_state(data: &[f64], feature_map: &FeatureMap) -> Vec<Complex64> {
    encode_feature_map(data, feature_map).expect("Encoding should succeed for valid data")
}

// ============================================================
// KERNEL COMPUTATION
// ============================================================

/// Compute the fidelity kernel K(x,y) = |<phi(x)|phi(y)>|^2.
pub fn fidelity_kernel(x: &[f64], y: &[f64], feature_map: &FeatureMap) -> f64 {
    let state_x = encode_and_get_state(x, feature_map);
    let state_y = encode_and_get_state(y, feature_map);
    state_overlap(&state_x, &state_y)
}

/// Compute the projected kernel value.
///
/// Projects encoded states onto a subspace defined by the observable qubits,
/// then computes a Gaussian RBF kernel on the projected expectation values.
pub fn projected_kernel(
    x: &[f64],
    y: &[f64],
    feature_map: &FeatureMap,
    observables: &[usize],
) -> f64 {
    let state_x = encode_and_get_state(x, feature_map);
    let state_y = encode_and_get_state(y, feature_map);

    let proj_x = project_to_observables(&state_x, observables);
    let proj_y = project_to_observables(&state_y, observables);

    // Gaussian RBF on the projected expectation values
    let diff_sq: f64 = proj_x
        .iter()
        .zip(proj_y.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    (-diff_sq / 2.0).exp()
}

/// Project a state vector onto Pauli-Z expectation values for given qubits.
fn project_to_observables(state: &[Complex64], observables: &[usize]) -> Vec<f64> {
    let num_qubits = (state.len() as f64).log2() as usize;
    observables
        .iter()
        .map(|&q| {
            let bit = num_qubits - 1 - q;
            let mut expval = 0.0;
            for (idx, amp) in state.iter().enumerate() {
                let parity = if ((idx >> bit) & 1) == 0 { 1.0 } else { -1.0 };
                expval += amp.norm_sqr() * parity;
            }
            expval
        })
        .collect()
}

/// Compute the full kernel matrix for a dataset.
///
/// Uses rayon for parallel computation. Only computes the upper triangle
/// (the matrix is symmetric). Diagonal entries are 1.0 for fidelity kernels.
pub fn compute_kernel_matrix(
    data: &[Vec<f64>],
    config: &KernelConfig,
) -> Result<KernelMatrix, KernelError> {
    let n = data.len();
    if n == 0 {
        return Err(KernelError::EncodingFailed("Empty dataset".to_string()));
    }

    // Validate dimensions
    let expected_dim = config.feature_map.num_qubits();
    for (_i, d) in data.iter().enumerate() {
        if d.len() > expected_dim {
            return Err(KernelError::DimensionMismatch {
                expected: expected_dim,
                got: d.len(),
            });
        }
    }

    // Pre-encode all states (can be parallelized)
    let states: Vec<Vec<Complex64>> = data
        .par_iter()
        .map(|d| encode_and_get_state(d, &config.feature_map))
        .collect();

    // Compute upper triangle entries in parallel
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();

    let values: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let val = match &config.kernel_type {
                KernelType::Fidelity => state_overlap(&states[i], &states[j]),
                KernelType::Projected { observables } => {
                    let proj_i = project_to_observables(&states[i], observables);
                    let proj_j = project_to_observables(&states[j], observables);
                    let diff_sq: f64 = proj_i
                        .iter()
                        .zip(proj_j.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (-diff_sq / 2.0).exp()
                }
            };
            (i, j, val)
        })
        .collect();

    let mut matrix = Array2::<f64>::zeros((n, n));
    for (i, j, val) in values {
        matrix[[i, j]] = val;
        matrix[[j, i]] = val;
    }

    if config.normalize {
        // Normalize so diagonal is exactly 1.0
        let diag: Vec<f64> = (0..n).map(|i| matrix[[i, i]]).collect();
        for i in 0..n {
            for j in 0..n {
                if diag[i] > 1e-15 && diag[j] > 1e-15 {
                    matrix[[i, j]] /= (diag[i] * diag[j]).sqrt();
                }
            }
        }
    }

    Ok(KernelMatrix {
        matrix,
        num_samples: n,
        feature_map_name: config.feature_map.name().to_string(),
    })
}

// ============================================================
// QSVM: SEQUENTIAL MINIMAL OPTIMIZATION
// ============================================================

/// Train a QSVM using simplified SMO on the quantum kernel matrix.
///
/// Solves the SVM dual problem:
///   max  Σ α_i - 1/2 ΣΣ α_i α_j y_i y_j K(x_i, x_j)
///   s.t. 0 ≤ α_i ≤ C, Σ α_i y_i = 0
pub fn train_qsvm(
    train_data: &[Vec<f64>],
    train_labels: &[i8],
    config: &QsvmConfig,
) -> Result<QsvmResult, KernelError> {
    let n = train_data.len();
    assert_eq!(
        n,
        train_labels.len(),
        "Data and labels must have same length"
    );

    // Step 1: Compute kernel matrix
    let km = compute_kernel_matrix(train_data, &config.kernel_config)?;
    let k = &km.matrix;

    // Step 2: Simplified SMO
    let c = config.c_parameter;
    let tol = config.tolerance;
    let max_iter = config.max_iterations;

    let mut alphas = vec![0.0_f64; n];
    let mut bias = 0.0_f64;

    // Cache of f(x_i) - y_i values for efficiency
    let labels_f64: Vec<f64> = train_labels.iter().map(|&y| y as f64).collect();
    let mut errors: Vec<f64> = labels_f64.iter().map(|&y| -y).collect();

    let mut passes = 0;
    let max_passes = max_iter;

    while passes < max_passes {
        let mut num_changed = 0;

        for i in 0..n {
            let ei = errors[i];
            let yi = labels_f64[i];

            // Check KKT violation
            if (yi * ei < -tol && alphas[i] < c) || (yi * ei > tol && alphas[i] > 0.0) {
                // Select j != i (simplified: pick j that maximizes |Ei - Ej|)
                let j = select_second_index(i, &errors, n);
                let ej = errors[j];
                let yj = labels_f64[j];

                let alpha_i_old = alphas[i];
                let alpha_j_old = alphas[j];

                // Compute bounds
                let (lo, hi) = if (yi - yj).abs() < 1e-10 {
                    // y_i == y_j
                    let lo = (alphas[i] + alphas[j] - c).max(0.0);
                    let hi = (alphas[i] + alphas[j]).min(c);
                    (lo, hi)
                } else {
                    let lo = (alphas[j] - alphas[i]).max(0.0);
                    let hi = (c + alphas[j] - alphas[i]).min(c);
                    (lo, hi)
                };

                if (hi - lo).abs() < 1e-10 {
                    continue;
                }

                // Compute eta
                let eta = 2.0 * k[[i, j]] - k[[i, i]] - k[[j, j]];
                if eta >= 0.0 {
                    continue;
                }

                // Update alpha_j
                alphas[j] -= yj * (ei - ej) / eta;
                alphas[j] = alphas[j].max(lo).min(hi);

                if (alphas[j] - alpha_j_old).abs() < 1e-5 {
                    continue;
                }

                // Update alpha_i
                alphas[i] += yi * yj * (alpha_j_old - alphas[j]);

                // Update bias
                let b1 = bias
                    - ei
                    - yi * (alphas[i] - alpha_i_old) * k[[i, i]]
                    - yj * (alphas[j] - alpha_j_old) * k[[i, j]];
                let b2 = bias
                    - ej
                    - yi * (alphas[i] - alpha_i_old) * k[[i, j]]
                    - yj * (alphas[j] - alpha_j_old) * k[[j, j]];

                if alphas[i] > 0.0 && alphas[i] < c {
                    bias = b1;
                } else if alphas[j] > 0.0 && alphas[j] < c {
                    bias = b2;
                } else {
                    bias = (b1 + b2) / 2.0;
                }

                // Update error cache
                for idx in 0..n {
                    errors[idx] = (0..n)
                        .map(|m| alphas[m] * labels_f64[m] * k[[m, idx]])
                        .sum::<f64>()
                        + bias
                        - labels_f64[idx];
                }

                num_changed += 1;
            }
        }

        if num_changed == 0 {
            passes += 1;
        } else {
            passes = 0;
        }
    }

    // Identify support vectors
    let sv_threshold = 1e-7;
    let support_vectors: Vec<usize> = (0..n).filter(|&i| alphas[i] > sv_threshold).collect();
    let num_support_vectors = support_vectors.len();

    // Compute training accuracy
    let mut correct = 0;
    for i in 0..n {
        let decision: f64 = (0..n)
            .map(|j| alphas[j] * labels_f64[j] * k[[j, i]])
            .sum::<f64>()
            + bias;
        let pred = if decision >= 0.0 { 1i8 } else { -1i8 };
        if pred == train_labels[i] {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / n as f64;

    Ok(QsvmResult {
        support_vectors,
        alphas,
        bias,
        accuracy,
        num_support_vectors,
    })
}

/// Select the second SMO index that maximizes |E_i - E_j|.
fn select_second_index(i: usize, errors: &[f64], n: usize) -> usize {
    let ei = errors[i];
    let mut best_j = if i == 0 { 1 } else { 0 };
    let mut max_diff = 0.0_f64;
    for j in 0..n {
        if j == i {
            continue;
        }
        let diff = (ei - errors[j]).abs();
        if diff > max_diff {
            max_diff = diff;
            best_j = j;
        }
    }
    best_j
}

/// Predict the label for a single test point using a trained QSVM.
pub fn predict_qsvm(
    model: &QsvmResult,
    train_data: &[Vec<f64>],
    train_labels: &[i8],
    test_point: &[f64],
    config: &KernelConfig,
) -> i8 {
    let labels_f64: Vec<f64> = train_labels.iter().map(|&y| y as f64).collect();
    let n = train_data.len();

    let decision: f64 = (0..n)
        .map(|j| {
            if model.alphas[j] > 1e-7 {
                let kval = match &config.kernel_type {
                    KernelType::Fidelity => {
                        fidelity_kernel(&train_data[j], test_point, &config.feature_map)
                    }
                    KernelType::Projected { observables } => projected_kernel(
                        &train_data[j],
                        test_point,
                        &config.feature_map,
                        observables,
                    ),
                };
                model.alphas[j] * labels_f64[j] * kval
            } else {
                0.0
            }
        })
        .sum::<f64>()
        + model.bias;

    if decision >= 0.0 {
        1
    } else {
        -1
    }
}

/// Evaluate QSVM accuracy on a test set.
pub fn evaluate_qsvm(
    model: &QsvmResult,
    train_data: &[Vec<f64>],
    train_labels: &[i8],
    test_data: &[Vec<f64>],
    test_labels: &[i8],
    config: &KernelConfig,
) -> f64 {
    let n = test_data.len();
    if n == 0 {
        return 0.0;
    }
    let correct = test_data
        .iter()
        .zip(test_labels.iter())
        .filter(|(point, &label)| {
            predict_qsvm(model, train_data, train_labels, point, config) == label
        })
        .count();
    correct as f64 / n as f64
}

// ============================================================
// KERNEL ANALYSIS
// ============================================================

/// Compute the kernel-target alignment.
///
/// Measures how well the kernel matrix K aligns with the ideal kernel derived
/// from the labels. A(K, K_ideal) = <K, K_ideal>_F / (||K||_F * ||K_ideal||_F)
/// where K_ideal[i,j] = y_i * y_j.
pub fn kernel_alignment(k: &KernelMatrix, labels: &[i8]) -> f64 {
    let n = k.num_samples;
    assert_eq!(n, labels.len(), "Kernel size and labels length must match");

    let mut k_ideal = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k_ideal[[i, j]] = labels[i] as f64 * labels[j] as f64;
        }
    }

    let fro_inner: f64 = k
        .matrix
        .iter()
        .zip(k_ideal.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm_k: f64 = k.matrix.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_ideal: f64 = k_ideal.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_k < 1e-15 || norm_ideal < 1e-15 {
        return 0.0;
    }
    fro_inner / (norm_k * norm_ideal)
}

/// Compute the kernel concentration (variance of off-diagonal elements).
///
/// High concentration (low variance) indicates the kernel values are all similar,
/// which is bad for classification because it means the feature map cannot
/// distinguish between data points.
pub fn kernel_concentration(k: &KernelMatrix) -> f64 {
    let n = k.num_samples;
    if n < 2 {
        return 0.0;
    }

    let mut off_diag = Vec::with_capacity(n * (n - 1));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                off_diag.push(k.matrix[[i, j]]);
            }
        }
    }

    let mean = off_diag.iter().sum::<f64>() / off_diag.len() as f64;
    let variance = off_diag
        .iter()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f64>()
        / off_diag.len() as f64;
    variance
}

/// Estimate the effective dimension of the kernel matrix.
///
/// Computed as the exponential of the Shannon entropy of the normalized eigenvalues.
/// A higher effective dimension means the kernel uses more of the Hilbert space.
pub fn effective_dimension(k: &KernelMatrix) -> f64 {
    let n = k.num_samples;
    if n == 0 {
        return 0.0;
    }

    // Compute eigenvalues using power iteration on k^T k
    // For simplicity, use the trace-based approach: effective dim from
    // the ratio of (trace)^2 / trace(K^2)
    let trace: f64 = (0..n).map(|i| k.matrix[[i, i]]).sum();
    let trace_sq: f64 = (0..n)
        .map(|i| {
            (0..n)
                .map(|m| k.matrix[[i, m]] * k.matrix[[m, i]])
                .sum::<f64>()
        })
        .sum();

    if trace_sq < 1e-15 {
        return 0.0;
    }
    // Effective rank: (trace(K))^2 / trace(K^2)
    (trace * trace) / trace_sq
}

// ============================================================
// DATA GENERATION
// ============================================================

/// Generate a "two moons" dataset (two interleaving half-circles).
pub fn make_moons(n_samples: usize, noise: f64, rng: &mut impl Rng) -> (Vec<Vec<f64>>, Vec<i8>) {
    let n_per_class = n_samples / 2;
    let mut data = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    // Upper moon
    for i in 0..n_per_class {
        let t = PI * i as f64 / n_per_class as f64;
        let x = t.cos() + rng.gen_range(-noise..noise);
        let y = t.sin() + rng.gen_range(-noise..noise);
        data.push(vec![x, y]);
        labels.push(1);
    }

    // Lower moon (shifted)
    for i in 0..n_per_class {
        let t = PI * i as f64 / n_per_class as f64;
        let x = 1.0 - t.cos() + rng.gen_range(-noise..noise);
        let y = 0.5 - t.sin() + rng.gen_range(-noise..noise);
        data.push(vec![x, y]);
        labels.push(-1);
    }

    (data, labels)
}

/// Generate a "concentric circles" dataset.
pub fn make_circles(n_samples: usize, noise: f64, rng: &mut impl Rng) -> (Vec<Vec<f64>>, Vec<i8>) {
    let n_per_class = n_samples / 2;
    let mut data = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    // Inner circle (radius ~0.5)
    for i in 0..n_per_class {
        let t = 2.0 * PI * i as f64 / n_per_class as f64;
        let r = 0.5;
        let x = r * t.cos() + rng.gen_range(-noise..noise);
        let y = r * t.sin() + rng.gen_range(-noise..noise);
        data.push(vec![x, y]);
        labels.push(1);
    }

    // Outer circle (radius ~1.0)
    for i in 0..n_per_class {
        let t = 2.0 * PI * i as f64 / n_per_class as f64;
        let r = 1.0;
        let x = r * t.cos() + rng.gen_range(-noise..noise);
        let y = r * t.sin() + rng.gen_range(-noise..noise);
        data.push(vec![x, y]);
        labels.push(-1);
    }

    (data, labels)
}

/// Generate an XOR pattern dataset (linearly inseparable).
pub fn make_xor(n_samples: usize, noise: f64, rng: &mut impl Rng) -> (Vec<Vec<f64>>, Vec<i8>) {
    let n_per_quadrant = n_samples / 4;
    let mut data = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    let centers = [(0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)];
    let xor_labels: [i8; 4] = [1, 1, -1, -1];

    for (idx, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..n_per_quadrant {
            let x = cx + rng.gen_range(-noise..noise);
            let y = cy + rng.gen_range(-noise..noise);
            data.push(vec![x, y]);
            labels.push(xor_labels[idx]);
        }
    }

    (data, labels)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_kernel_config_builder_defaults() {
        let config = KernelConfig::default();
        assert!(matches!(
            config.feature_map,
            FeatureMap::AngleEncoding { num_qubits: 2 }
        ));
        assert!(matches!(config.kernel_type, KernelType::Fidelity));
        assert!(config.shots.is_none());
        assert!(config.normalize);

        let qsvm_config = QsvmConfig::default();
        assert!((qsvm_config.c_parameter - 1.0).abs() < 1e-10);
        assert_eq!(qsvm_config.max_iterations, 1000);
        assert!((qsvm_config.tolerance - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_angle_encoding_same_data_overlap_one() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 3 };
        let data = vec![0.5, 1.2, 0.8];
        let state1 = encode_and_get_state(&data, &fm);
        let state2 = encode_and_get_state(&data, &fm);
        let overlap = state_overlap(&state1, &state2);
        assert!(
            (overlap - 1.0).abs() < 1e-10,
            "Same data should give overlap 1.0, got {}",
            overlap
        );
    }

    #[test]
    fn test_angle_encoding_different_data_overlap_less_than_one() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let data_a = vec![0.0, 0.0];
        let data_b = vec![PI / 2.0, PI / 2.0];
        let state_a = encode_and_get_state(&data_a, &fm);
        let state_b = encode_and_get_state(&data_b, &fm);
        let overlap = state_overlap(&state_a, &state_b);
        assert!(
            overlap < 1.0 - 1e-5,
            "Different data should give overlap < 1, got {}",
            overlap
        );
    }

    #[test]
    fn test_fidelity_kernel_symmetric() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let x = vec![0.3, 0.7];
        let y = vec![1.1, 0.2];
        let kxy = fidelity_kernel(&x, &y, &fm);
        let kyx = fidelity_kernel(&y, &x, &fm);
        assert!(
            (kxy - kyx).abs() < 1e-10,
            "Kernel should be symmetric: K(x,y)={} vs K(y,x)={}",
            kxy,
            kyx
        );
    }

    #[test]
    fn test_kernel_matrix_positive_semi_definite() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let config = KernelConfig::default().with_feature_map(fm);
        let data = vec![
            vec![0.1, 0.2],
            vec![0.5, 0.8],
            vec![1.0, 0.3],
            vec![0.7, 1.5],
        ];
        let km = compute_kernel_matrix(&data, &config).unwrap();

        // Check PSD via Gershgorin: all eigenvalues are non-negative
        // Simpler check: x^T K x >= 0 for random vectors
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let x: Vec<f64> = (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let xtk: f64 = (0..4)
                .map(|i| (0..4).map(|j| x[i] * km.matrix[[i, j]] * x[j]).sum::<f64>())
                .sum();
            assert!(
                xtk >= -1e-10,
                "Kernel matrix should be PSD, got x^T K x = {}",
                xtk
            );
        }
    }

    #[test]
    fn test_kernel_diagonal_is_one() {
        let fm = FeatureMap::ZZFeatureMap {
            num_qubits: 2,
            reps: 1,
        };
        let config = KernelConfig::default().with_feature_map(fm);
        let data = vec![vec![0.5, 0.3], vec![1.2, 0.8], vec![0.1, 1.5]];
        let km = compute_kernel_matrix(&data, &config).unwrap();

        for i in 0..data.len() {
            assert!(
                (km.matrix[[i, i]] - 1.0).abs() < 1e-10,
                "Diagonal should be 1.0, got K[{},{}] = {}",
                i,
                i,
                km.matrix[[i, i]]
            );
        }
    }

    #[test]
    fn test_zz_feature_map_encodes_interactions() {
        let fm_zz = FeatureMap::ZZFeatureMap {
            num_qubits: 2,
            reps: 1,
        };
        let fm_angle = FeatureMap::AngleEncoding { num_qubits: 2 };

        let data = vec![0.5, 0.3];
        let state_zz = encode_and_get_state(&data, &fm_zz);
        let state_angle = encode_and_get_state(&data, &fm_angle);

        // States should be different because ZZ has entangling gates
        let overlap = state_overlap(&state_zz, &state_angle);
        assert!(
            overlap < 1.0 - 1e-5,
            "ZZ and Angle should produce different states, overlap = {}",
            overlap
        );
    }

    #[test]
    fn test_qsvm_linearly_separable() {
        let mut rng = StdRng::seed_from_u64(42);

        // Generate well-separated data
        let mut train_data = Vec::new();
        let mut train_labels = Vec::new();

        // Class +1: centered at (2.0, 2.0)
        for _ in 0..15 {
            train_data.push(vec![
                2.0 + rng.gen_range(-0.3..0.3),
                2.0 + rng.gen_range(-0.3..0.3),
            ]);
            train_labels.push(1i8);
        }
        // Class -1: centered at (-2.0, -2.0)
        for _ in 0..15 {
            train_data.push(vec![
                -2.0 + rng.gen_range(-0.3..0.3),
                -2.0 + rng.gen_range(-0.3..0.3),
            ]);
            train_labels.push(-1i8);
        }

        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let config = QsvmConfig::default()
            .with_kernel_config(KernelConfig::default().with_feature_map(fm))
            .with_c_parameter(10.0)
            .with_max_iterations(200);

        let result = train_qsvm(&train_data, &train_labels, &config).unwrap();
        assert!(
            result.accuracy >= 0.95,
            "Linearly separable data should classify with >=95% accuracy, got {}",
            result.accuracy
        );
    }

    #[test]
    fn test_qsvm_xor_data() {
        let mut rng = StdRng::seed_from_u64(99);
        let (data, labels) = make_xor(40, 0.15, &mut rng);

        let fm = FeatureMap::ZZFeatureMap {
            num_qubits: 2,
            reps: 2,
        };
        let config = QsvmConfig::default()
            .with_kernel_config(KernelConfig::default().with_feature_map(fm))
            .with_c_parameter(5.0)
            .with_max_iterations(500);

        let result = train_qsvm(&data, &labels, &config).unwrap();
        assert!(
            result.accuracy >= 0.70,
            "XOR data with ZZ kernel should achieve >70% training accuracy, got {}",
            result.accuracy
        );
    }

    #[test]
    fn test_kernel_alignment_range() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let config = KernelConfig::default().with_feature_map(fm);

        let mut rng = StdRng::seed_from_u64(77);
        let (data, labels) = make_moons(20, 0.1, &mut rng);
        let km = compute_kernel_matrix(&data, &config).unwrap();
        let alignment = kernel_alignment(&km, &labels);

        assert!(
            alignment >= -1.0 - 1e-10 && alignment <= 1.0 + 1e-10,
            "Kernel alignment should be in [-1, 1], got {}",
            alignment
        );
    }

    #[test]
    fn test_make_moons_correct_count() {
        let mut rng = StdRng::seed_from_u64(55);
        let (data, labels) = make_moons(100, 0.1, &mut rng);
        assert_eq!(data.len(), 100, "Should generate 100 samples");
        assert_eq!(labels.len(), 100, "Should generate 100 labels");
        let pos = labels.iter().filter(|&&l| l == 1).count();
        let neg = labels.iter().filter(|&&l| l == -1).count();
        assert_eq!(pos, 50, "Should have 50 positive samples");
        assert_eq!(neg, 50, "Should have 50 negative samples");
    }

    #[test]
    fn test_amplitude_encoding_normalizes() {
        let fm = FeatureMap::AmplitudeEncoding { num_qubits: 2 };
        let data = vec![3.0, 4.0, 0.0, 0.0];
        let state = encode_and_get_state(&data, &fm);

        // Check normalization
        let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "Amplitude encoding should produce normalized state, got ||psi||^2 = {}",
            norm_sq
        );

        // Check relative amplitudes: data was [3, 4, 0, 0], norm=5
        assert!((state[0].re - 0.6).abs() < 1e-10, "Expected 3/5 = 0.6");
        assert!((state[1].re - 0.8).abs() < 1e-10, "Expected 4/5 = 0.8");
    }

    #[test]
    fn test_kernel_concentration() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let config = KernelConfig::default().with_feature_map(fm);
        let data = vec![vec![0.1, 0.2], vec![0.5, 0.8], vec![1.0, 0.3]];
        let km = compute_kernel_matrix(&data, &config).unwrap();
        let conc = kernel_concentration(&km);
        assert!(conc >= 0.0, "Variance should be non-negative, got {}", conc);
    }

    #[test]
    fn test_effective_dimension() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let config = KernelConfig::default().with_feature_map(fm);
        let data = vec![
            vec![0.1, 0.2],
            vec![0.5, 0.8],
            vec![1.0, 0.3],
            vec![0.7, 1.5],
        ];
        let km = compute_kernel_matrix(&data, &config).unwrap();
        let eff_dim = effective_dimension(&km);
        // Effective dimension should be between 1 and n
        assert!(
            eff_dim >= 0.9 && eff_dim <= 4.1,
            "Effective dimension should be in [1, 4], got {}",
            eff_dim
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let data = vec![1.0, 2.0, 3.0]; // 3 features but only 2 qubits
        let result = encode_feature_map(&data, &fm);
        assert!(result.is_err());
        if let Err(KernelError::DimensionMismatch { expected, got }) = result {
            assert_eq!(expected, 2);
            assert_eq!(got, 3);
        }
    }

    #[test]
    fn test_iqp_encoding_produces_valid_state() {
        let fm = FeatureMap::IQPEncoding {
            num_qubits: 2,
            reps: 1,
        };
        let data = vec![0.5, 0.3];
        let state = encode_and_get_state(&data, &fm);
        let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "IQP encoding should produce normalized state, got {}",
            norm_sq
        );
    }

    #[test]
    fn test_projected_kernel_value() {
        let fm = FeatureMap::AngleEncoding { num_qubits: 2 };
        let x = vec![0.3, 0.7];
        let y = vec![0.3, 0.7];
        // Same data should give projected kernel = 1.0 (distance = 0)
        let k = projected_kernel(&x, &y, &fm, &[0, 1]);
        assert!(
            (k - 1.0).abs() < 1e-10,
            "Same data should give projected kernel = 1.0, got {}",
            k
        );
    }
}
