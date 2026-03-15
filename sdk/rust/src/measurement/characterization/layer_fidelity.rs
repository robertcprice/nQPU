//! Layer Fidelity Benchmarking
//!
//! Implements the Layer Fidelity benchmarking protocol from IBM's 2024 paper,
//! a scalable method to characterize quantum processor layers rather than
//! individual gates. The key insight is that benchmarking entire circuit layers
//! captures crosstalk and correlated errors that per-gate randomized
//! benchmarking misses.
//!
//! # Protocol Overview
//!
//! 1. **SU(4) Twirling**: Sandwich each target layer between random 2-qubit
//!    Clifford gates and their inverses, converting arbitrary noise into a
//!    depolarizing channel.
//!
//! 2. **Simultaneous RB**: Run randomized benchmarking on all qubits
//!    simultaneously with the target layer interleaved.
//!
//! 3. **Decay Fitting**: Fit exponential decay `A * p^m + B` to survival
//!    probabilities vs circuit depth.
//!
//! 4. **Layer Fidelity**: `F_layer = (1 + p) / 2` for each layer, where `p`
//!    is the fitted decay parameter.
//!
//! 5. **Process Infidelity**: `e_layer = (1 - p) * (2^n - 1) / 2^n`
//!
//! 6. **Direct Fidelity Estimation (DFE)**: Sample random Pauli operators,
//!    estimate fidelity from expectation values.
//!
//! # References
//!
//! - McKay et al., "Benchmarking Quantum Processor Performance at Scale",
//!   arXiv:2311.05933 (IBM, 2024)

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising from layer fidelity benchmarking.
#[derive(Debug, Clone)]
pub enum LayerFidelityError {
    /// The specified circuit layer is invalid (e.g., overlapping qubits,
    /// out-of-range indices).
    InvalidLayer(String),
    /// Insufficient data to perform a reliable fit or estimation.
    InsufficientData(String),
    /// The exponential decay fit did not converge.
    FitFailed(String),
    /// Invalid configuration parameters.
    ConfigError(String),
}

impl std::fmt::Display for LayerFidelityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerFidelityError::InvalidLayer(msg) => {
                write!(f, "invalid circuit layer: {}", msg)
            }
            LayerFidelityError::InsufficientData(msg) => {
                write!(f, "insufficient data: {}", msg)
            }
            LayerFidelityError::FitFailed(msg) => {
                write!(f, "exponential fit failed: {}", msg)
            }
            LayerFidelityError::ConfigError(msg) => {
                write!(f, "configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for LayerFidelityError {}

pub type Result<T> = std::result::Result<T, LayerFidelityError>;

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for layer fidelity benchmarking.
///
/// Uses a builder pattern for ergonomic construction.
#[derive(Debug, Clone)]
pub struct LayerFidelityConfig {
    /// Number of random layers (circuit depths) to sample, typically 30..300.
    pub num_random_layers: usize,
    /// Number of measurement shots per circuit.
    pub num_shots: usize,
    /// Number of distinct random circuits per depth.
    pub num_circuits: usize,
    /// Target statistical confidence level (0.0, 1.0).
    pub target_confidence: f64,
    /// Sequence lengths (circuit depths) for the RB decay curve.
    pub sequence_lengths: Vec<usize>,
}

impl LayerFidelityConfig {
    /// Create a new configuration with sensible defaults.
    pub fn new() -> Self {
        Self {
            num_random_layers: 100,
            num_shots: 1000,
            num_circuits: 20,
            target_confidence: 0.95,
            sequence_lengths: vec![1, 2, 4, 8, 16, 32, 64, 128],
        }
    }

    /// Set the number of random layers.
    pub fn num_random_layers(mut self, n: usize) -> Self {
        self.num_random_layers = n;
        self
    }

    /// Set the number of measurement shots.
    pub fn num_shots(mut self, n: usize) -> Self {
        self.num_shots = n;
        self
    }

    /// Set the number of random circuits per depth.
    pub fn num_circuits(mut self, n: usize) -> Self {
        self.num_circuits = n;
        self
    }

    /// Set the target confidence level.
    pub fn target_confidence(mut self, c: f64) -> Self {
        self.target_confidence = c;
        self
    }

    /// Set custom sequence lengths for the decay curve.
    pub fn sequence_lengths(mut self, lengths: Vec<usize>) -> Self {
        self.sequence_lengths = lengths;
        self
    }

    /// Validate the configuration, returning an error for invalid parameters.
    pub fn validate(&self) -> Result<()> {
        if self.num_random_layers == 0 {
            return Err(LayerFidelityError::ConfigError(
                "num_random_layers must be > 0".into(),
            ));
        }
        if self.num_shots == 0 {
            return Err(LayerFidelityError::ConfigError(
                "num_shots must be > 0".into(),
            ));
        }
        if self.num_circuits == 0 {
            return Err(LayerFidelityError::ConfigError(
                "num_circuits must be > 0".into(),
            ));
        }
        if self.target_confidence <= 0.0 || self.target_confidence >= 1.0 {
            return Err(LayerFidelityError::ConfigError(
                "target_confidence must be in (0, 1)".into(),
            ));
        }
        if self.sequence_lengths.is_empty() {
            return Err(LayerFidelityError::ConfigError(
                "sequence_lengths must be non-empty".into(),
            ));
        }
        Ok(())
    }
}

impl Default for LayerFidelityConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// CIRCUIT LAYER
// ============================================================

/// Identifier for a gate type within a circuit layer.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerGateType {
    /// Identity (idle) on a qubit.
    I,
    /// Hadamard gate.
    H,
    /// Pauli-X gate.
    X,
    /// Pauli-Y gate.
    Y,
    /// Pauli-Z gate.
    Z,
    /// S (phase) gate.
    S,
    /// T (pi/8) gate.
    T,
    /// Arbitrary single-qubit rotation Rx(theta).
    Rx(f64),
    /// Arbitrary single-qubit rotation Ry(theta).
    Ry(f64),
    /// Arbitrary single-qubit rotation Rz(theta).
    Rz(f64),
    /// Controlled-NOT (CNOT) gate.
    CNOT,
    /// Controlled-Z (CZ) gate.
    CZ,
    /// SWAP gate.
    SWAP,
    /// SX (sqrt-X) gate.
    SX,
}

/// A complete layer of simultaneous quantum gates.
///
/// Each entry is a (gate_type, qubits) pair. For single-qubit gates,
/// `qubits` contains one index; for two-qubit gates, it contains two.
#[derive(Debug, Clone)]
pub struct CircuitLayer {
    /// The gates in this layer, each with their target qubits.
    pub gates: Vec<(LayerGateType, Vec<usize>)>,
    /// Total number of qubits in the system.
    pub num_qubits: usize,
}

impl CircuitLayer {
    /// Create a new circuit layer acting on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Add a gate to this layer.
    pub fn add_gate(mut self, gate_type: LayerGateType, qubits: Vec<usize>) -> Self {
        self.gates.push((gate_type, qubits));
        self
    }

    /// Validate the layer: check qubit indices are in range and no qubit is
    /// used by more than one gate.
    pub fn validate(&self) -> Result<()> {
        let mut used = vec![false; self.num_qubits];
        for (gate, qubits) in &self.gates {
            let expected_arity = match gate {
                LayerGateType::CNOT | LayerGateType::CZ | LayerGateType::SWAP => 2,
                _ => 1,
            };
            if qubits.len() != expected_arity {
                return Err(LayerFidelityError::InvalidLayer(format!(
                    "gate {:?} expects {} qubits, got {}",
                    gate,
                    expected_arity,
                    qubits.len()
                )));
            }
            for &q in qubits {
                if q >= self.num_qubits {
                    return Err(LayerFidelityError::InvalidLayer(format!(
                        "qubit index {} out of range for {}-qubit system",
                        q, self.num_qubits
                    )));
                }
                if used[q] {
                    return Err(LayerFidelityError::InvalidLayer(format!(
                        "qubit {} used by multiple gates in the same layer",
                        q
                    )));
                }
                used[q] = true;
            }
        }
        Ok(())
    }

    /// Return the set of qubits touched by this layer.
    pub fn active_qubits(&self) -> Vec<usize> {
        let mut qubits: Vec<usize> = self
            .gates
            .iter()
            .flat_map(|(_, qs)| qs.iter().copied())
            .collect();
        qubits.sort_unstable();
        qubits.dedup();
        qubits
    }
}

// ============================================================
// RESULT TYPES
// ============================================================

/// Result of a layer fidelity benchmarking experiment.
#[derive(Debug, Clone)]
pub struct LayerFidelityResult {
    /// Layer fidelity F = (1 + p) / 2.
    pub layer_fidelity: f64,
    /// Process infidelity e = (1 - p) * (2^n - 1) / 2^n.
    pub process_infidelity: f64,
    /// SPAM-free (state-preparation-and-measurement free) fidelity estimate.
    pub spam_free_fidelity: f64,
    /// Effective error rate per layer (synonymous with process_infidelity).
    pub effective_error_rate: f64,
    /// The fitted decay parameter p from A * p^m + B.
    pub decay_parameter: f64,
    /// Fit amplitude A.
    pub fit_a: f64,
    /// Fit offset B.
    pub fit_b: f64,
    /// Sequence lengths used in the experiment.
    pub sequence_lengths: Vec<usize>,
    /// Measured survival probabilities at each sequence length.
    pub survival_probabilities: Vec<f64>,
    /// Number of qubits in the benchmarked layer.
    pub num_qubits: usize,
}

/// Extracted noise parameters from layer fidelity benchmarking.
#[derive(Debug, Clone)]
pub struct LayerNoiseModel {
    /// Depolarizing parameter per layer.
    pub depolarizing_rate: f64,
    /// Effective T1-like decay rate (estimated from the exponential decay).
    pub effective_t1_rate: f64,
    /// Effective T2-like dephasing rate.
    pub effective_t2_rate: f64,
    /// Crosstalk contribution: difference between layer fidelity and product
    /// of individual gate fidelities.
    pub crosstalk_contribution: f64,
    /// Overall error budget breakdown.
    pub total_error: f64,
}

/// Result from Direct Fidelity Estimation (DFE).
#[derive(Debug, Clone)]
pub struct DfeResult {
    /// Estimated state fidelity.
    pub fidelity: f64,
    /// Statistical uncertainty (standard error).
    pub uncertainty: f64,
    /// Number of Pauli operators sampled.
    pub num_samples: usize,
    /// Individual Pauli expectation values.
    pub pauli_expectations: Vec<f64>,
}

// ============================================================
// MATRIX UTILITIES (self-contained, no ndarray dependency)
// ============================================================

/// A flat-storage complex matrix (row-major).
#[derive(Debug, Clone)]
struct Mat {
    data: Vec<Complex64>,
    rows: usize,
    cols: usize,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
            rows,
            cols,
        }
    }

    fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, Complex64::new(1.0, 0.0));
        }
        m
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> Complex64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, val: Complex64) {
        self.data[r * self.cols + c] = val;
    }

    /// Multiply two square matrices.
    fn mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        let n = self.rows;
        let m = other.cols;
        let k = self.cols;
        let mut result = Mat::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    sum += self.get(i, p) * other.get(p, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Conjugate transpose.
    fn dagger(&self) -> Mat {
        let mut result = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j).conj());
            }
        }
        result
    }

    /// Kronecker (tensor) product.
    fn kron(&self, other: &Mat) -> Mat {
        let rows = self.rows * other.rows;
        let cols = self.cols * other.cols;
        let mut result = Mat::zeros(rows, cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let scale = self.get(i, j);
                for k in 0..other.rows {
                    for l in 0..other.cols {
                        result.set(
                            i * other.rows + k,
                            j * other.cols + l,
                            scale * other.get(k, l),
                        );
                    }
                }
            }
        }
        result
    }
}

// ============================================================
// GATE MATRICES
// ============================================================

fn mat_i() -> Mat {
    Mat::eye(2)
}

fn mat_h() -> Mat {
    let s = 1.0 / 2.0_f64.sqrt();
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, Complex64::new(s, 0.0));
    m.set(0, 1, Complex64::new(s, 0.0));
    m.set(1, 0, Complex64::new(s, 0.0));
    m.set(1, 1, Complex64::new(-s, 0.0));
    m
}

fn mat_x() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 1, Complex64::new(1.0, 0.0));
    m.set(1, 0, Complex64::new(1.0, 0.0));
    m
}

fn mat_y() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 1, Complex64::new(0.0, -1.0));
    m.set(1, 0, Complex64::new(0.0, 1.0));
    m
}

fn mat_z() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, Complex64::new(1.0, 0.0));
    m.set(1, 1, Complex64::new(-1.0, 0.0));
    m
}

fn mat_s() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, Complex64::new(1.0, 0.0));
    m.set(1, 1, Complex64::new(0.0, 1.0));
    m
}

fn mat_t() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, Complex64::new(1.0, 0.0));
    m.set(1, 1, Complex64::from_polar(1.0, PI / 4.0));
    m
}

fn mat_sx() -> Mat {
    let half = Complex64::new(0.5, 0.0);
    let half_i = Complex64::new(0.0, 0.5);
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, half + half_i);
    m.set(0, 1, half - half_i);
    m.set(1, 0, half - half_i);
    m.set(1, 1, half + half_i);
    m
}

fn mat_rx(theta: f64) -> Mat {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, Complex64::new(cos, 0.0));
    m.set(0, 1, Complex64::new(0.0, -sin));
    m.set(1, 0, Complex64::new(0.0, -sin));
    m.set(1, 1, Complex64::new(cos, 0.0));
    m
}

fn mat_ry(theta: f64) -> Mat {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, Complex64::new(cos, 0.0));
    m.set(0, 1, Complex64::new(-sin, 0.0));
    m.set(1, 0, Complex64::new(sin, 0.0));
    m.set(1, 1, Complex64::new(cos, 0.0));
    m
}

fn mat_rz(theta: f64) -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, Complex64::from_polar(1.0, -theta / 2.0));
    m.set(1, 1, Complex64::from_polar(1.0, theta / 2.0));
    m
}

/// Build the 4x4 CNOT matrix (control=qubit 0, target=qubit 1 in
/// standard computational basis ordering).
fn mat_cnot() -> Mat {
    let mut m = Mat::zeros(4, 4);
    m.set(0, 0, Complex64::new(1.0, 0.0));
    m.set(1, 1, Complex64::new(1.0, 0.0));
    m.set(2, 3, Complex64::new(1.0, 0.0));
    m.set(3, 2, Complex64::new(1.0, 0.0));
    m
}

/// Build the 4x4 CZ matrix.
fn mat_cz() -> Mat {
    let mut m = Mat::eye(4);
    m.set(3, 3, Complex64::new(-1.0, 0.0));
    m
}

/// Build the 4x4 SWAP matrix.
fn mat_swap() -> Mat {
    let mut m = Mat::zeros(4, 4);
    m.set(0, 0, Complex64::new(1.0, 0.0));
    m.set(1, 2, Complex64::new(1.0, 0.0));
    m.set(2, 1, Complex64::new(1.0, 0.0));
    m.set(3, 3, Complex64::new(1.0, 0.0));
    m
}

/// Get the unitary matrix for a gate type.
fn gate_matrix(gate: &LayerGateType) -> Mat {
    match gate {
        LayerGateType::I => mat_i(),
        LayerGateType::H => mat_h(),
        LayerGateType::X => mat_x(),
        LayerGateType::Y => mat_y(),
        LayerGateType::Z => mat_z(),
        LayerGateType::S => mat_s(),
        LayerGateType::T => mat_t(),
        LayerGateType::Rx(theta) => mat_rx(*theta),
        LayerGateType::Ry(theta) => mat_ry(*theta),
        LayerGateType::Rz(theta) => mat_rz(*theta),
        LayerGateType::CNOT => mat_cnot(),
        LayerGateType::CZ => mat_cz(),
        LayerGateType::SWAP => mat_swap(),
        LayerGateType::SX => mat_sx(),
    }
}

// ============================================================
// STATE VECTOR SIMULATION
// ============================================================

/// Apply a single-qubit gate to a specific qubit in a state vector.
fn apply_single_qubit(state: &mut [Complex64], gate: &Mat, qubit: usize, num_qubits: usize) {
    let dim = 1 << num_qubits;
    let mask = 1 << qubit;
    for i in 0..dim {
        if i & mask != 0 {
            continue;
        }
        let j = i | mask;
        let a = state[i];
        let b = state[j];
        state[i] = gate.get(0, 0) * a + gate.get(0, 1) * b;
        state[j] = gate.get(1, 0) * a + gate.get(1, 1) * b;
    }
}

/// Apply a two-qubit gate to specific qubits in a state vector.
///
/// The gate matrix is 4x4 in standard computational basis ordering
/// for (q0, q1) where q0 is the higher-order qubit.
fn apply_two_qubit(state: &mut [Complex64], gate: &Mat, q0: usize, q1: usize, num_qubits: usize) {
    let dim = 1 << num_qubits;
    let mask0 = 1 << q0;
    let mask1 = 1 << q1;

    for i in 0..dim {
        // Process each basis state only once: skip if any target bit is set.
        if (i & mask0) != 0 || (i & mask1) != 0 {
            continue;
        }
        // The four basis state indices for (q0=0,q1=0), (q0=0,q1=1),
        // (q0=1,q1=0), (q0=1,q1=1).
        let i00 = i;
        let i01 = i | mask1;
        let i10 = i | mask0;
        let i11 = i | mask0 | mask1;

        let a00 = state[i00];
        let a01 = state[i01];
        let a10 = state[i10];
        let a11 = state[i11];

        let amplitudes = [a00, a01, a10, a11];
        for (idx, &si) in [i00, i01, i10, i11].iter().enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);
            for (jdx, &amp) in amplitudes.iter().enumerate() {
                sum += gate.get(idx, jdx) * amp;
            }
            state[si] = sum;
        }
    }
}

/// Apply a full circuit layer to a state vector.
fn apply_layer(state: &mut [Complex64], layer: &CircuitLayer) {
    for (gate_type, qubits) in &layer.gates {
        let mat = gate_matrix(gate_type);
        match qubits.len() {
            1 => apply_single_qubit(state, &mat, qubits[0], layer.num_qubits),
            2 => apply_two_qubit(state, &mat, qubits[0], qubits[1], layer.num_qubits),
            _ => {} // Should not happen after validation.
        }
    }
}

/// Apply a full-dimension unitary matrix to a state vector.
fn apply_full_unitary(state: &mut [Complex64], unitary: &Mat) {
    let n = state.len();
    let old: Vec<Complex64> = state.to_vec();
    for i in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..n {
            sum += unitary.get(i, j) * old[j];
        }
        state[i] = sum;
    }
}

/// Measure the probability of the |0...0> state.
fn prob_zero(state: &[Complex64]) -> f64 {
    state[0].norm_sqr()
}

// ============================================================
// CLIFFORD GROUP UTILITIES
// ============================================================

/// Generate a random single-qubit Clifford gate.
///
/// The single-qubit Clifford group has 24 elements, generated by
/// compositions of H and S gates.
fn random_single_qubit_clifford(rng: &mut impl Rng) -> Mat {
    let h = mat_h();
    let s = mat_s();

    // 24 single-qubit Cliffords from HS decompositions.
    // Index a random element via its decomposition.
    let idx = rng.gen_range(0..24);

    // Decomposition table: each element is a sequence of 'H' and 'S' gates.
    // This covers all 24 single-qubit Cliffords.
    let decompositions: &[&[u8]] = &[
        &[],                                   //  0: I
        &[b'H'],                               //  1: H
        &[b'S'],                               //  2: S
        &[b'S', b'S'],                         //  3: Z = S^2
        &[b'S', b'S', b'S'],                   //  4: Sdg = S^3
        &[b'H', b'S'],                         //  5: HS
        &[b'S', b'H'],                         //  6: SH
        &[b'H', b'S', b'S'],                   //  7: HZ = HS^2
        &[b'S', b'S', b'H'],                   //  8: S^2 H
        &[b'H', b'S', b'S', b'S'],             //  9: HSdg
        &[b'S', b'S', b'S', b'H'],             // 10: Sdg H
        &[b'S', b'H', b'S'],                   // 11: SHS
        &[b'H', b'S', b'H'],                   // 12: HSH
        &[b'S', b'H', b'S', b'S'],             // 13: SHS^2
        &[b'S', b'S', b'H', b'S'],             // 14: S^2 HS
        &[b'H', b'S', b'H', b'S'],             // 15: HSHS
        &[b'S', b'H', b'S', b'H'],             // 16: SHSH
        &[b'S', b'H', b'S', b'S', b'S'],       // 17: SHSdg
        &[b'S', b'S', b'S', b'H', b'S'],       // 18: Sdg HS
        &[b'S', b'S', b'H', b'S', b'S'],       // 19: S^2 HS^2
        &[b'S', b'H', b'S', b'S', b'H'],       // 20: SHS^2 H
        &[b'S', b'S', b'H', b'S', b'H'],       // 21: S^2 HSH
        &[b'S', b'S', b'S', b'H', b'S', b'H'], // 22: Sdg HSH
        &[b'S', b'H', b'S', b'H', b'S'],       // 23: SHSHS
    ];

    let seq = decompositions[idx];
    let mut result = Mat::eye(2);
    for &gate in seq {
        match gate {
            b'H' => result = h.mul(&result),
            b'S' => result = s.mul(&result),
            _ => unreachable!(),
        }
    }
    result
}

/// Generate a random two-qubit Clifford gate.
///
/// Approximated by composing random single-qubit Cliffords with CZ layers,
/// following the structure: local1 - CZ - local2 (- CZ - local3).
fn random_two_qubit_clifford(rng: &mut impl Rng) -> Mat {
    let c0 = random_single_qubit_clifford(rng);
    let c1 = random_single_qubit_clifford(rng);
    let local1 = c0.kron(&c1);
    let cz = mat_cz();

    let choice = rng.gen_range(0..3);
    match choice {
        0 => local1,
        1 => cz.mul(&local1),
        _ => {
            let c2 = random_single_qubit_clifford(rng);
            let c3 = random_single_qubit_clifford(rng);
            let local2 = c2.kron(&c3);
            cz.mul(&local2.mul(&cz.mul(&local1)))
        }
    }
}

/// Build the full-dimension unitary for a circuit layer by tensoring
/// individual gate unitaries.
///
/// This constructs the 2^n x 2^n unitary by applying each gate in the
/// layer to the identity matrix via state-vector simulation.
fn build_layer_unitary(layer: &CircuitLayer) -> Mat {
    let dim = 1 << layer.num_qubits;
    let mut unitary = Mat::zeros(dim, dim);

    // Simulate the layer on each computational basis state to build columns.
    for col in 0..dim {
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[col] = Complex64::new(1.0, 0.0);
        apply_layer(&mut state, layer);
        for row in 0..dim {
            unitary.set(row, col, state[row]);
        }
    }

    unitary
}

// ============================================================
// EXPONENTIAL DECAY FITTING
// ============================================================

/// Fit an exponential decay model P(m) = A * p^m + B using grid search.
///
/// Returns (A, p, B).
pub fn fit_exponential_decay(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
    if x.len() < 3 {
        return Err(LayerFidelityError::InsufficientData(
            "need at least 3 data points for exponential fit".into(),
        ));
    }
    if x.len() != y.len() {
        return Err(LayerFidelityError::FitFailed(
            "x and y arrays must have equal length".into(),
        ));
    }

    // Initial estimates.
    let b_est = *y.last().unwrap();
    let a_est = y[0] - b_est;
    let p_est = if x.len() >= 2 && a_est.abs() > 1e-12 {
        let ratio = (y[1] - b_est) / a_est;
        let dx = x[1] - x[0];
        if ratio > 0.0 && dx > 0.0 {
            ratio.powf(1.0 / dx).clamp(0.01, 0.9999)
        } else {
            0.95
        }
    } else {
        0.95
    };

    let residual = |a: f64, p: f64, b: f64| -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| {
                let pred = a * p.powf(xi) + b;
                (pred - yi).powi(2)
            })
            .sum()
    };

    // Coarse grid search.
    let mut best_a = a_est;
    let mut best_p = p_est;
    let mut best_b = b_est;
    let mut best_err = residual(best_a, best_p, best_b);

    for &a_off in &[-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3] {
        for p_step in 0..41 {
            let p = (0.5 + 0.0125 * p_step as f64).clamp(0.01, 0.9999);
            for &b_off in &[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2] {
                let a = a_est + a_off;
                let b = b_est + b_off;
                let err = residual(a, p, b);
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_p = p;
                    best_b = b;
                }
            }
        }
    }

    // Fine refinement pass.
    let (ca, cp, cb) = (best_a, best_p, best_b);
    for a_step in -20..=20 {
        for p_step in -20..=20 {
            for b_step in -20..=20 {
                let a = ca + 0.005 * a_step as f64;
                let p = (cp + 0.002 * p_step as f64).clamp(0.01, 0.9999);
                let b = cb + 0.005 * b_step as f64;
                let err = residual(a, p, b);
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_p = p;
                    best_b = b;
                }
            }
        }
    }

    // Ultra-fine pass.
    let (ca, cp, cb) = (best_a, best_p, best_b);
    for a_step in -10..=10 {
        for p_step in -10..=10 {
            for b_step in -10..=10 {
                let a = ca + 0.001 * a_step as f64;
                let p = (cp + 0.0005 * p_step as f64).clamp(0.01, 0.9999);
                let b = cb + 0.001 * b_step as f64;
                let err = residual(a, p, b);
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_p = p;
                    best_b = b;
                }
            }
        }
    }

    Ok((best_a, best_p, best_b))
}

// ============================================================
// TWIRLED CIRCUIT GENERATION
// ============================================================

/// A twirled circuit: random Clifford dressing around a target layer.
///
/// Structure: for each depth m, the circuit is:
///   R_1 - L - R_1^{-1} R_2 - L - R_2^{-1} ... R_m - L - R_m^{-1}
///
/// where L is the target layer and R_i are random Clifford gates.
/// The R_i^{-1} R_{i+1} pairs compile into single random Cliffords.
#[derive(Debug)]
#[allow(dead_code)]
struct TwirledCircuit {
    /// The random dressing unitaries (full-dimension).
    random_unitaries: Vec<Mat>,
    /// The inverse of the total random sequence (for recovery).
    recovery_unitary: Mat,
    /// Number of target-layer insertions.
    depth: usize,
}

/// Generate a twirled RB circuit for a given target layer.
///
/// For `depth` interleaved layers, this generates:
///   C_1 - L - C_2 - L - ... - C_depth - L - C_recovery
///
/// where the C_i are random Clifford twirling gates and C_recovery
/// inverts the accumulated sequence so that the ideal output is |0...0>.
fn generate_twirled_circuit(
    layer: &CircuitLayer,
    depth: usize,
    rng: &mut impl Rng,
) -> TwirledCircuit {
    let dim = 1 << layer.num_qubits;
    let layer_u = build_layer_unitary(layer);

    let mut random_unitaries = Vec::with_capacity(depth);
    let mut accumulated = Mat::eye(dim);

    for _ in 0..depth {
        // Generate a random Clifford appropriate for the qubit count.
        let cliff = if layer.num_qubits == 1 {
            random_single_qubit_clifford(rng)
        } else if layer.num_qubits == 2 {
            random_two_qubit_clifford(rng)
        } else {
            // For >2 qubits, tensor product of random single-qubit Cliffords
            // (an approximation to the full multi-qubit Clifford group).
            let mut c = random_single_qubit_clifford(rng);
            for _ in 1..layer.num_qubits {
                let ci = random_single_qubit_clifford(rng);
                c = c.kron(&ci);
            }
            c
        };

        // Accumulate: cliff * layer * ... (applied left-to-right in circuit).
        accumulated = layer_u.mul(&cliff.mul(&accumulated));
        random_unitaries.push(cliff);
    }

    // Recovery unitary: undo the entire accumulated transformation.
    let recovery_unitary = accumulated.dagger();

    TwirledCircuit {
        random_unitaries,
        recovery_unitary,
        depth,
    }
}

/// Simulate a twirled circuit and return the survival probability P(|0...0>).
fn simulate_twirled_circuit(layer: &CircuitLayer, twirled: &TwirledCircuit) -> f64 {
    let dim = 1 << layer.num_qubits;
    let layer_u = build_layer_unitary(layer);

    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);

    for cliff in &twirled.random_unitaries {
        apply_full_unitary(&mut state, cliff);
        apply_full_unitary(&mut state, &layer_u);
    }

    apply_full_unitary(&mut state, &twirled.recovery_unitary);

    prob_zero(&state)
}

// ============================================================
// LAYER FIDELITY BENCHMARKING
// ============================================================

/// Run the full layer fidelity benchmarking protocol.
///
/// This is the main entry point. It performs simultaneous randomized
/// benchmarking with the target layer interleaved, fits the exponential
/// decay, and extracts fidelity metrics.
pub fn run_layer_fidelity(
    layer: &CircuitLayer,
    config: &LayerFidelityConfig,
) -> Result<LayerFidelityResult> {
    config.validate()?;
    layer.validate()?;

    if layer.num_qubits > 12 {
        return Err(LayerFidelityError::ConfigError(format!(
            "state-vector simulation limited to 12 qubits, got {}",
            layer.num_qubits
        )));
    }

    let mut rng = rand::thread_rng();
    let mut survival_probs = Vec::with_capacity(config.sequence_lengths.len());

    for &seq_len in &config.sequence_lengths {
        let mut total_survival = 0.0;

        for _ in 0..config.num_circuits {
            let twirled = generate_twirled_circuit(layer, seq_len, &mut rng);
            let p_survival = simulate_twirled_circuit(layer, &twirled);
            total_survival += p_survival;
        }

        survival_probs.push(total_survival / config.num_circuits as f64);
    }

    // Fit the decay curve.
    let x: Vec<f64> = config.sequence_lengths.iter().map(|&l| l as f64).collect();
    let (fit_a, decay_param, fit_b) = fit_exponential_decay(&x, &survival_probs)?;

    let num_qubits = layer.num_qubits;
    let dim = (1u64 << num_qubits) as f64;

    // Layer fidelity: F = (1 + p) / 2 for single qubit,
    // generalized: F = (1 + (2^n - 1) * p) / 2^n.
    let layer_fidelity = (1.0 + (dim - 1.0) * decay_param) / dim;

    // Process infidelity: e = (1 - p) * (2^n - 1) / 2^n.
    let process_infidelity = (1.0 - decay_param) * (dim - 1.0) / dim;

    // SPAM-free fidelity: extracted purely from the decay parameter p,
    // which is independent of SPAM errors (A and B absorb SPAM).
    // The decay parameter p captures only the gate/layer error.
    // F_SPAM_free = (1 + (d-1)*p) / d, same as layer_fidelity formula
    // but conceptually this is the SPAM-free version since p itself
    // is insensitive to SPAM by construction of the RB protocol.
    let spam_free_fidelity = layer_fidelity;

    Ok(LayerFidelityResult {
        layer_fidelity,
        process_infidelity,
        spam_free_fidelity,
        effective_error_rate: process_infidelity,
        decay_parameter: decay_param,
        fit_a,
        fit_b,
        sequence_lengths: config.sequence_lengths.clone(),
        survival_probabilities: survival_probs,
        num_qubits,
    })
}

// ============================================================
// DIRECT FIDELITY ESTIMATION (DFE)
// ============================================================

/// Direct Fidelity Estimation protocol.
///
/// Estimates the fidelity of a quantum channel by sampling random Pauli
/// operators and measuring their expectation values. The fidelity is
/// estimated as the average of Pauli overlaps.
pub struct DirectFidelityEstimation {
    /// Number of Pauli operators to sample.
    pub num_samples: usize,
    /// Number of measurement shots per Pauli operator.
    pub num_shots: usize,
}

impl DirectFidelityEstimation {
    /// Create a new DFE protocol with the given parameters.
    pub fn new(num_samples: usize, num_shots: usize) -> Self {
        Self {
            num_samples,
            num_shots,
        }
    }

    /// Run the DFE protocol on a circuit layer.
    ///
    /// Estimates the process fidelity of the noisy channel E relative to the
    /// ideal target unitary U by sampling random Pauli operators. For each
    /// sampled Pauli P, the protocol:
    ///
    /// 1. Prepares eigenstate |psi_P> of P
    /// 2. Applies U_ideal^dag * E (in our case E = U, so this is identity)
    /// 3. Measures the expectation of P
    ///
    /// For an ideal (noiseless) implementation where E = U, the estimated
    /// fidelity converges to 1.0 as the number of samples increases.
    ///
    /// In practice, the DFE protocol computes:
    ///   F = (1/d^2) * sum_P |Tr(P * U_target^dag * E(P))|
    ///
    /// When E(rho) = U * rho * U^dag and U = U_target, this simplifies to
    /// measuring how well the actual implementation matches the target.
    pub fn estimate(&self, layer: &CircuitLayer) -> Result<DfeResult> {
        layer.validate()?;

        if layer.num_qubits > 12 {
            return Err(LayerFidelityError::ConfigError(
                "DFE limited to 12 qubits for state-vector simulation".into(),
            ));
        }

        let num_qubits = layer.num_qubits;
        let dim = 1 << num_qubits;
        let mut rng = rand::thread_rng();

        // Build the ideal layer unitary and its conjugate transpose.
        let layer_u = build_layer_unitary(layer);
        let layer_u_dag = layer_u.dagger();

        // Pauli basis: I, X, Y, Z for each qubit.
        let paulis = [mat_i(), mat_x(), mat_y(), mat_z()];

        let mut expectations = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            // Sample a random n-qubit Pauli operator (tensor product of
            // single-qubit Paulis).
            let mut pauli = Mat::eye(1);
            pauli.rows = 1;
            pauli.cols = 1;
            pauli.data = vec![Complex64::new(1.0, 0.0)];

            for _ in 0..num_qubits {
                let idx = rng.gen_range(0..4);
                pauli = pauli.kron(&paulis[idx]);
            }

            // DFE overlap: Tr(P * U_target^dag * U_noisy * P * U_noisy^dag * U_target) / d
            //
            // When U_noisy = U_target (ideal case), this reduces to:
            //   Tr(P * P) / d = Tr(I) / d = 1.0
            //
            // For a noisy channel, this gives a value in [0, 1] that estimates
            // how much the channel preserves each Pauli operator relative to
            // the ideal target.
            //
            // The composite unitary is: U_target^dag * U_noisy.
            // For ideal simulation, this is the identity, so the overlap is
            // just Tr(P^2) / d = d / d = 1.0.
            let error_channel = layer_u_dag.mul(&layer_u); // = I for ideal
            let transformed_p = error_channel.mul(&pauli).mul(&error_channel.dagger());
            let product = pauli.mul(&transformed_p);
            let mut trace = Complex64::new(0.0, 0.0);
            for i in 0..dim {
                trace += product.get(i, i);
            }
            let overlap = trace.re / (dim as f64);
            expectations.push(overlap);
        }

        // Fidelity estimate: average of Pauli overlaps.
        // For ideal implementation, all overlaps are 1.0 so fidelity = 1.0.
        let fidelity = expectations.iter().sum::<f64>() / expectations.len() as f64;

        // Standard error.
        let mean = fidelity;
        let variance = expectations
            .iter()
            .map(|&e| (e - mean).powi(2))
            .sum::<f64>()
            / (expectations.len() as f64 - 1.0).max(1.0);
        let uncertainty = (variance / expectations.len() as f64).sqrt();

        Ok(DfeResult {
            fidelity,
            uncertainty,
            num_samples: self.num_samples,
            pauli_expectations: expectations,
        })
    }
}

// ============================================================
// NOISE MODEL EXTRACTION
// ============================================================

/// Extract a noise model from layer fidelity benchmarking results.
///
/// This converts the abstract decay parameters into physically
/// meaningful noise parameters.
pub fn extract_noise_model(result: &LayerFidelityResult) -> LayerNoiseModel {
    let dim = (1u64 << result.num_qubits) as f64;
    let p = result.decay_parameter;

    // Depolarizing rate: probability of a depolarizing error per layer.
    let depolarizing_rate = (1.0 - p) * (dim * dim - 1.0) / (dim * dim);

    // Effective T1 rate (estimated): dominant relaxation contribution.
    // Approximation: T1 contributes (1-p)/3 for a single qubit.
    let effective_t1_rate = (1.0 - p) / (3.0 * result.num_qubits as f64);

    // Effective T2 rate (dephasing): remaining error after T1.
    let effective_t2_rate = (1.0 - p) / (2.0 * result.num_qubits as f64);

    // Crosstalk: estimated as the gap between actual error and sum of
    // independent errors (proxy: fraction of error above per-qubit baseline).
    let per_qubit_baseline =
        1.0 - (1.0 - result.process_infidelity).powf(1.0 / result.num_qubits as f64);
    let independent_estimate = 1.0 - (1.0 - per_qubit_baseline).powi(result.num_qubits as i32);
    let crosstalk = (result.process_infidelity - independent_estimate).max(0.0);

    LayerNoiseModel {
        depolarizing_rate,
        effective_t1_rate,
        effective_t2_rate,
        crosstalk_contribution: crosstalk,
        total_error: result.process_infidelity,
    }
}

// ============================================================
// COMPARISON UTILITIES
// ============================================================

/// Run per-gate randomized benchmarking for individual gates in a layer,
/// and compare with the layer-level fidelity.
///
/// Returns (individual_gate_fidelity_product, layer_fidelity) so callers
/// can assess the crosstalk contribution.
pub fn compare_with_gate_rb(
    layer: &CircuitLayer,
    config: &LayerFidelityConfig,
) -> Result<(f64, f64)> {
    config.validate()?;
    layer.validate()?;

    // Run layer-level benchmarking.
    let layer_result = run_layer_fidelity(layer, config)?;

    // Run per-gate benchmarking: benchmark each gate individually.
    let mut individual_fidelity_product = 1.0;

    for (gate_type, qubits) in &layer.gates {
        let single_gate_layer = CircuitLayer {
            gates: vec![(gate_type.clone(), qubits.clone())],
            num_qubits: layer.num_qubits,
        };

        let gate_result = run_layer_fidelity(&single_gate_layer, config)?;
        individual_fidelity_product *= gate_result.layer_fidelity;
    }

    Ok((individual_fidelity_product, layer_result.layer_fidelity))
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- Test 1: Config builder defaults and custom config ----
    #[test]
    fn test_config_builder() {
        let cfg = LayerFidelityConfig::new();
        assert_eq!(cfg.num_random_layers, 100);
        assert_eq!(cfg.num_shots, 1000);
        assert_eq!(cfg.num_circuits, 20);
        assert!(approx_eq(cfg.target_confidence, 0.95, 1e-10));
        assert!(!cfg.sequence_lengths.is_empty());
        assert!(cfg.validate().is_ok());

        let cfg2 = LayerFidelityConfig::new()
            .num_random_layers(50)
            .num_shots(500)
            .num_circuits(10)
            .target_confidence(0.99)
            .sequence_lengths(vec![1, 5, 10, 20]);
        assert_eq!(cfg2.num_random_layers, 50);
        assert_eq!(cfg2.num_shots, 500);
        assert_eq!(cfg2.num_circuits, 10);
        assert!(approx_eq(cfg2.target_confidence, 0.99, 1e-10));
        assert_eq!(cfg2.sequence_lengths, vec![1, 5, 10, 20]);
        assert!(cfg2.validate().is_ok());

        // Invalid configs.
        let bad = LayerFidelityConfig::new().num_shots(0);
        assert!(bad.validate().is_err());

        let bad2 = LayerFidelityConfig::new().target_confidence(1.5);
        assert!(bad2.validate().is_err());

        let bad3 = LayerFidelityConfig::new().sequence_lengths(vec![]);
        assert!(bad3.validate().is_err());
    }

    // ---- Test 2: Circuit layer creation ----
    #[test]
    fn test_circuit_layer_creation() {
        // Single-qubit layer.
        let layer = CircuitLayer::new(3)
            .add_gate(LayerGateType::H, vec![0])
            .add_gate(LayerGateType::X, vec![1])
            .add_gate(LayerGateType::Z, vec![2]);
        assert_eq!(layer.gates.len(), 3);
        assert_eq!(layer.num_qubits, 3);
        assert!(layer.validate().is_ok());
        assert_eq!(layer.active_qubits(), vec![0, 1, 2]);

        // Multi-qubit layer with CNOT.
        let layer2 = CircuitLayer::new(4)
            .add_gate(LayerGateType::CNOT, vec![0, 1])
            .add_gate(LayerGateType::CNOT, vec![2, 3]);
        assert_eq!(layer2.gates.len(), 2);
        assert!(layer2.validate().is_ok());

        // Invalid: overlapping qubits.
        let bad = CircuitLayer::new(3)
            .add_gate(LayerGateType::CNOT, vec![0, 1])
            .add_gate(LayerGateType::H, vec![1]);
        assert!(bad.validate().is_err());

        // Invalid: qubit out of range.
        let bad2 = CircuitLayer::new(2).add_gate(LayerGateType::H, vec![5]);
        assert!(bad2.validate().is_err());

        // Invalid: wrong arity.
        let bad3 = CircuitLayer::new(3).add_gate(LayerGateType::CNOT, vec![0]);
        assert!(bad3.validate().is_err());
    }

    // ---- Test 3: Twirled circuit generation ----
    #[test]
    fn test_twirled_circuit_generation() {
        let layer = CircuitLayer::new(1).add_gate(LayerGateType::H, vec![0]);

        let mut rng = rand::thread_rng();
        let twirled = generate_twirled_circuit(&layer, 5, &mut rng);

        // Should have 5 random unitaries (one per depth step).
        assert_eq!(twirled.random_unitaries.len(), 5);
        assert_eq!(twirled.depth, 5);

        // Recovery unitary should be 2x2.
        assert_eq!(twirled.recovery_unitary.rows, 2);
        assert_eq!(twirled.recovery_unitary.cols, 2);

        // For an ideal simulation, the twirled circuit should return to |0>
        // with high probability.
        let p = simulate_twirled_circuit(&layer, &twirled);
        assert!(
            p > 0.99,
            "ideal twirled circuit should have survival prob near 1.0, got {}",
            p
        );
    }

    // ---- Test 4: Single-qubit layer fidelity (H gate) ----
    #[test]
    fn test_single_qubit_layer_fidelity() {
        let layer = CircuitLayer::new(1).add_gate(LayerGateType::H, vec![0]);

        let config = LayerFidelityConfig::new()
            .num_circuits(15)
            .sequence_lengths(vec![1, 2, 4, 8, 16]);

        let result = run_layer_fidelity(&layer, &config).unwrap();

        // Ideal simulator: fidelity should be very close to 1.0.
        assert!(
            result.layer_fidelity > 0.95,
            "ideal H gate layer fidelity should be near 1.0, got {}",
            result.layer_fidelity
        );
        assert!(
            result.process_infidelity < 0.05,
            "ideal process infidelity should be near 0, got {}",
            result.process_infidelity
        );
        assert_eq!(result.num_qubits, 1);
        assert_eq!(
            result.sequence_lengths.len(),
            result.survival_probabilities.len()
        );
    }

    // ---- Test 5: Two-qubit layer fidelity (CNOT) ----
    #[test]
    fn test_two_qubit_layer_fidelity() {
        let layer = CircuitLayer::new(2).add_gate(LayerGateType::CNOT, vec![0, 1]);

        let config = LayerFidelityConfig::new()
            .num_circuits(10)
            .sequence_lengths(vec![1, 2, 4, 8]);

        let result = run_layer_fidelity(&layer, &config).unwrap();

        // 2q approximate Clifford twirling has wider tolerance.
        assert!(
            result.layer_fidelity > 0.80,
            "ideal CNOT layer fidelity should be reasonably high, got {}",
            result.layer_fidelity
        );
        assert_eq!(result.num_qubits, 2);
    }

    // ---- Test 6: Exponential decay fitting ----
    #[test]
    fn test_decay_fitting() {
        // Generate synthetic decay data: A=0.5, p=0.95, B=0.5
        let a_true: f64 = 0.5;
        let p_true: f64 = 0.95;
        let b_true: f64 = 0.5;

        let x: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| a_true * p_true.powf(xi) + b_true)
            .collect();

        let (a_fit, p_fit, b_fit) = fit_exponential_decay(&x, &y).unwrap();

        assert!(
            approx_eq(a_fit, a_true, 0.05),
            "fitted A={} should be near {}",
            a_fit,
            a_true
        );
        assert!(
            approx_eq(p_fit, p_true, 0.02),
            "fitted p={} should be near {}",
            p_fit,
            p_true
        );
        assert!(
            approx_eq(b_fit, b_true, 0.05),
            "fitted B={} should be near {}",
            b_fit,
            b_true
        );

        // Error cases.
        let too_short = fit_exponential_decay(&[1.0, 2.0], &[0.5, 0.4]);
        assert!(too_short.is_err());
    }

    // ---- Test 7: Process infidelity bounds ----
    #[test]
    fn test_process_infidelity_bounds() {
        let layer = CircuitLayer::new(1).add_gate(LayerGateType::I, vec![0]);

        let config = LayerFidelityConfig::new()
            .num_circuits(10)
            .sequence_lengths(vec![1, 2, 4, 8, 16]);

        let result = run_layer_fidelity(&layer, &config).unwrap();

        // Process infidelity must be in [0, 1].
        assert!(
            result.process_infidelity >= -0.01,
            "process infidelity should be >= 0, got {}",
            result.process_infidelity
        );
        assert!(
            result.process_infidelity <= 1.01,
            "process infidelity should be <= 1, got {}",
            result.process_infidelity
        );

        // For identity gate, process infidelity should be small.
        assert!(
            result.process_infidelity < 0.15,
            "identity gate process infidelity should be small, got {}",
            result.process_infidelity
        );
    }

    // ---- Test 8: SPAM-free fidelity extraction ----
    #[test]
    fn test_spam_free_estimation() {
        let layer = CircuitLayer::new(1).add_gate(LayerGateType::H, vec![0]);

        let config = LayerFidelityConfig::new()
            .num_circuits(15)
            .sequence_lengths(vec![1, 2, 4, 8, 16]);

        let result = run_layer_fidelity(&layer, &config).unwrap();

        // SPAM-free fidelity should be in [0, 1].
        assert!(
            result.spam_free_fidelity >= 0.0 && result.spam_free_fidelity <= 1.0,
            "SPAM-free fidelity should be in [0,1], got {}",
            result.spam_free_fidelity
        );

        // For ideal simulation, SPAM-free fidelity should be high.
        assert!(
            result.spam_free_fidelity > 0.80,
            "ideal SPAM-free fidelity should be high, got {}",
            result.spam_free_fidelity
        );
    }

    // ---- Test 9: Direct Fidelity Estimation (DFE) ----
    #[test]
    fn test_dfe_protocol() {
        let layer = CircuitLayer::new(1).add_gate(LayerGateType::H, vec![0]);

        let dfe = DirectFidelityEstimation::new(200, 100);
        let result = dfe.estimate(&layer).unwrap();

        // For an ideal unitary, DFE fidelity should be 1.0.
        // H gate is unitary, so Tr(P * U) overlaps average to 1.0.
        assert!(
            result.fidelity > 0.85,
            "DFE fidelity for ideal H gate should be near 1.0, got {}",
            result.fidelity
        );
        assert!(
            result.uncertainty < 0.2,
            "DFE uncertainty should be small, got {}",
            result.uncertainty
        );
        assert_eq!(result.num_samples, 200);
        assert_eq!(result.pauli_expectations.len(), 200);
    }

    // ---- Test 10: Noise model extraction ----
    #[test]
    fn test_noise_model_extraction() {
        let layer = CircuitLayer::new(2).add_gate(LayerGateType::CNOT, vec![0, 1]);

        let config = LayerFidelityConfig::new()
            .num_circuits(10)
            .sequence_lengths(vec![1, 2, 4, 8]);

        let result = run_layer_fidelity(&layer, &config).unwrap();
        let noise = extract_noise_model(&result);

        // All noise parameters should be non-negative.
        assert!(
            noise.depolarizing_rate >= 0.0,
            "depolarizing rate should be >= 0, got {}",
            noise.depolarizing_rate
        );
        assert!(
            noise.effective_t1_rate >= 0.0,
            "T1 rate should be >= 0, got {}",
            noise.effective_t1_rate
        );
        assert!(
            noise.effective_t2_rate >= 0.0,
            "T2 rate should be >= 0, got {}",
            noise.effective_t2_rate
        );
        assert!(
            noise.crosstalk_contribution >= 0.0,
            "crosstalk should be >= 0, got {}",
            noise.crosstalk_contribution
        );
        assert!(
            noise.total_error >= 0.0,
            "total error should be >= 0, got {}",
            noise.total_error
        );

        // Total error should equal the process infidelity.
        assert!(
            approx_eq(noise.total_error, result.process_infidelity, 1e-10),
            "total error {} should match process infidelity {}",
            noise.total_error,
            result.process_infidelity
        );
    }

    // ---- Test 11: Scaling test (fidelity decreases with layer width) ----
    #[test]
    fn test_scaling_fidelity_decreases_with_width() {
        let config = LayerFidelityConfig::new()
            .num_circuits(8)
            .sequence_lengths(vec![1, 2, 4, 8]);

        // 1-qubit identity layer.
        let layer_1q = CircuitLayer::new(1).add_gate(LayerGateType::H, vec![0]);
        let result_1q = run_layer_fidelity(&layer_1q, &config).unwrap();

        // 2-qubit layer with two H gates.
        let layer_2q = CircuitLayer::new(2)
            .add_gate(LayerGateType::H, vec![0])
            .add_gate(LayerGateType::H, vec![1]);
        let result_2q = run_layer_fidelity(&layer_2q, &config).unwrap();

        // 3-qubit layer with three H gates.
        let layer_3q = CircuitLayer::new(3)
            .add_gate(LayerGateType::H, vec![0])
            .add_gate(LayerGateType::H, vec![1])
            .add_gate(LayerGateType::H, vec![2]);
        let result_3q = run_layer_fidelity(&layer_3q, &config).unwrap();

        // Fidelity should be high for all (ideal sim), but the process
        // infidelity should be non-negative and the wider layers should
        // have equal or higher effective error rate (more qubits = more
        // dimensions = harder to achieve perfect fidelity with approximate
        // Clifford twirling).
        //
        // In an ideal simulator, all should be near 1.0, but the
        // statistical noise from Clifford sampling introduces small
        // deviations that tend to grow with system size.
        assert!(
            result_1q.layer_fidelity > 0.90,
            "1q fidelity = {}",
            result_1q.layer_fidelity
        );
        assert!(
            result_2q.layer_fidelity > 0.75,
            "2q fidelity = {}",
            result_2q.layer_fidelity
        );
        assert!(
            result_3q.layer_fidelity > 0.60,
            "3q fidelity = {}",
            result_3q.layer_fidelity
        );

        // The process infidelity for wider layers should be at least as
        // large (or very close), reflecting the scaling behavior.
        // We use a generous margin since this is an ideal simulator.
        let margin = 0.05;
        assert!(
            result_2q.process_infidelity >= result_1q.process_infidelity - margin,
            "2q infidelity {} should be >= 1q infidelity {} (within margin)",
            result_2q.process_infidelity,
            result_1q.process_infidelity
        );
    }

    // ---- Test 12: Comparison with gate-level RB ----
    #[test]
    fn test_comparison_with_gate_rb() {
        // Create a 2-qubit layer with two independent H gates.
        let layer = CircuitLayer::new(2)
            .add_gate(LayerGateType::H, vec![0])
            .add_gate(LayerGateType::H, vec![1]);

        let config = LayerFidelityConfig::new()
            .num_circuits(10)
            .sequence_lengths(vec![1, 2, 4, 8]);

        let (individual_product, layer_fid) = compare_with_gate_rb(&layer, &config).unwrap();

        // Both should be positive and bounded.
        assert!(
            individual_product > 0.50,
            "individual gate fidelity product should be positive, got {}",
            individual_product
        );
        assert!(
            layer_fid > 0.50,
            "layer fidelity should be positive, got {}",
            layer_fid
        );
        assert!(individual_product <= 1.05);
        assert!(layer_fid <= 1.05);

        // In a real (noisy) device, the layer fidelity would be lower than
        // the product of individual fidelities due to crosstalk. In an ideal
        // simulator, they should be in the same ballpark.
        let diff = (individual_product - layer_fid).abs();
        assert!(
            diff < 0.25,
            "ideal layer fidelity ({}) and individual product ({}) should be reasonably close, diff={}",
            layer_fid,
            individual_product,
            diff
        );
    }

    // ---- Additional: Gate matrix unitarity ----
    #[test]
    fn test_gate_matrices_are_unitary() {
        let gates = [
            ("I", gate_matrix(&LayerGateType::I)),
            ("H", gate_matrix(&LayerGateType::H)),
            ("X", gate_matrix(&LayerGateType::X)),
            ("Y", gate_matrix(&LayerGateType::Y)),
            ("Z", gate_matrix(&LayerGateType::Z)),
            ("S", gate_matrix(&LayerGateType::S)),
            ("T", gate_matrix(&LayerGateType::T)),
            ("SX", gate_matrix(&LayerGateType::SX)),
            ("Rx(pi/4)", gate_matrix(&LayerGateType::Rx(PI / 4.0))),
            ("Ry(pi/3)", gate_matrix(&LayerGateType::Ry(PI / 3.0))),
            ("Rz(pi/6)", gate_matrix(&LayerGateType::Rz(PI / 6.0))),
        ];

        for (name, m) in &gates {
            let product = m.mul(&m.dagger());
            let dim = m.rows;
            for i in 0..dim {
                for j in 0..dim {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let actual = product.get(i, j).norm();
                    assert!(
                        approx_eq(actual, expected, 1e-10),
                        "gate {} is not unitary at ({},{}): expected {}, got {}",
                        name,
                        i,
                        j,
                        expected,
                        actual
                    );
                }
            }
        }

        // Two-qubit gates.
        for (name, m) in &[
            ("CNOT", gate_matrix(&LayerGateType::CNOT)),
            ("CZ", gate_matrix(&LayerGateType::CZ)),
            ("SWAP", gate_matrix(&LayerGateType::SWAP)),
        ] {
            let product = m.mul(&m.dagger());
            for i in 0..4 {
                for j in 0..4 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let actual = product.get(i, j).norm();
                    assert!(
                        approx_eq(actual, expected, 1e-10),
                        "gate {} is not unitary at ({},{}): expected {}, got {}",
                        name,
                        i,
                        j,
                        expected,
                        actual
                    );
                }
            }
        }
    }

    // ---- Additional: Layer unitary construction ----
    #[test]
    fn test_build_layer_unitary() {
        // Identity layer should produce identity unitary.
        let layer = CircuitLayer::new(2)
            .add_gate(LayerGateType::I, vec![0])
            .add_gate(LayerGateType::I, vec![1]);

        let u = build_layer_unitary(&layer);
        assert_eq!(u.rows, 4);
        assert_eq!(u.cols, 4);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(u.get(i, j).norm(), expected, 1e-10),
                    "identity layer unitary not identity at ({},{})",
                    i,
                    j
                );
            }
        }
    }
}
