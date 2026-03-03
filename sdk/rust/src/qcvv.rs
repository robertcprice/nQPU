//! Quantum Characterization, Verification, and Validation (QCVV)
//!
//! Implements standard QCVV protocols for benchmarking quantum hardware:
//!
//! - **Cross-Entropy Benchmarking (XEB)**: Used by Google for quantum supremacy
//!   demonstrations. Measures fidelity by comparing sampled bitstring distributions
//!   against ideal state-vector probabilities.
//!
//! - **Randomized Benchmarking (RB)**: Measures average gate error by applying
//!   random Clifford sequences of increasing length and fitting the exponential
//!   decay of survival probability.
//!
//! - **Interleaved RB**: Extension that isolates the error rate of a specific
//!   gate by interleaving it into random Clifford sequences.

#![allow(dead_code)]

use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising from QCVV protocols.
#[derive(Debug, Clone)]
pub enum QcvvError {
    /// Invalid configuration parameters.
    InvalidConfig(String),
    /// Numerical failure during fitting or simulation.
    NumericalError(String),
    /// Qubit count too large for state-vector simulation.
    TooManyQubits(usize),
}

impl std::fmt::Display for QcvvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QcvvError::InvalidConfig(msg) => write!(f, "invalid QCVV config: {}", msg),
            QcvvError::NumericalError(msg) => write!(f, "numerical error: {}", msg),
            QcvvError::TooManyQubits(n) => {
                write!(f, "too many qubits for state-vector sim: {}", n)
            }
        }
    }
}

impl std::error::Error for QcvvError {}

pub type Result<T> = std::result::Result<T, QcvvError>;

// ============================================================
// XEB TYPES
// ============================================================

/// Configuration for cross-entropy benchmarking.
#[derive(Debug, Clone)]
pub struct XebConfig {
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
    /// Number of random circuits to sample.
    pub num_circuits: usize,
    /// Depth of each random circuit (number of gate layers).
    pub circuit_depth: usize,
    /// Number of measurement samples per circuit.
    pub num_samples: usize,
}

impl XebConfig {
    /// Create a new XEB configuration with sensible defaults.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            num_circuits: 20,
            circuit_depth: 20,
            num_samples: 1000,
        }
    }

    pub fn num_circuits(mut self, n: usize) -> Self {
        self.num_circuits = n;
        self
    }

    pub fn circuit_depth(mut self, d: usize) -> Self {
        self.circuit_depth = d;
        self
    }

    pub fn num_samples(mut self, s: usize) -> Self {
        self.num_samples = s;
        self
    }
}

/// Result of a cross-entropy benchmarking run.
#[derive(Debug, Clone)]
pub struct XebResult {
    /// Standard XEB fidelity: F = 2^n * <p_ideal(x)>_samples - 1
    pub xeb_fidelity: f64,
    /// Linear XEB fidelity (same formula, averaged over circuits).
    pub linear_xeb: f64,
    /// Logarithmic XEB: mean(log(2^n * p_ideal)) + euler_gamma
    pub log_xeb: f64,
    /// Number of circuits evaluated.
    pub num_circuits: usize,
    /// Depths at which fidelity was measured (if depth scan).
    pub circuit_depths: Vec<usize>,
    /// Fidelity at each depth (parallel to `circuit_depths`).
    pub fidelities_by_depth: Vec<f64>,
}

/// Type of single-qubit gate in a random circuit.
#[derive(Debug, Clone, Copy)]
pub enum SingleGateType {
    /// Haar-random SU(2) rotation.
    RandomSU2,
    /// Hadamard gate.
    Hadamard,
    /// T gate (pi/8).
    T,
    /// S gate (pi/4).
    S,
}

/// A gate in a random circuit.
#[derive(Debug, Clone)]
pub enum RandomGate {
    /// Single-qubit gate with Euler-angle parameters.
    SingleQubit {
        qubit: usize,
        gate_type: SingleGateType,
        params: [f64; 3],
    },
    /// Two-qubit entangling gate (CZ).
    TwoQubit { q0: usize, q1: usize },
}

/// A random quantum circuit composed of gate layers.
#[derive(Debug, Clone)]
pub struct RandomCircuit {
    /// Ordered list of gates.
    pub gates: Vec<RandomGate>,
    /// Number of qubits.
    pub num_qubits: usize,
    /// Circuit depth (number of layers).
    pub depth: usize,
}

// ============================================================
// RB TYPES
// ============================================================

/// Configuration for randomized benchmarking.
#[derive(Debug, Clone)]
pub struct RbConfig {
    /// Number of qubits (1 or 2).
    pub num_qubits: usize,
    /// Sequence lengths to benchmark.
    pub sequence_lengths: Vec<usize>,
    /// Number of random Clifford sequences per length.
    pub num_sequences: usize,
    /// Measurement shots per sequence.
    pub num_samples_per_sequence: usize,
}

impl RbConfig {
    /// Create a new RB configuration with standard defaults.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            sequence_lengths: vec![1, 2, 4, 8, 16, 32, 64],
            num_sequences: 30,
            num_samples_per_sequence: 100,
        }
    }

    pub fn sequence_lengths(mut self, lens: Vec<usize>) -> Self {
        self.sequence_lengths = lens;
        self
    }

    pub fn num_sequences(mut self, n: usize) -> Self {
        self.num_sequences = n;
        self
    }

    pub fn num_samples_per_sequence(mut self, n: usize) -> Self {
        self.num_samples_per_sequence = n;
        self
    }
}

/// Result of a randomized benchmarking experiment.
#[derive(Debug, Clone)]
pub struct RbResult {
    /// Error per Clifford gate: r = (1 - p) * (2^n - 1) / 2^n
    pub error_per_gate: f64,
    /// Decay parameter p from exponential fit.
    pub decay_rate: f64,
    /// Amplitude A from fit P(m) = A * p^m + B.
    pub fit_a: f64,
    /// Offset B from fit.
    pub fit_b: f64,
    /// Sequence lengths used.
    pub sequence_lengths: Vec<usize>,
    /// Measured survival probabilities at each length.
    pub survival_probabilities: Vec<f64>,
}

/// A Clifford gate represented by its unitary matrix.
#[derive(Debug, Clone)]
pub struct CliffordGate {
    /// 2^n x 2^n unitary matrix.
    pub matrix: Array2<Complex64>,
    /// Human-readable name.
    pub name: String,
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Compute the Kronecker (tensor) product of two matrices.
pub fn kronecker_product(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (ar, ac) = (a.nrows(), a.ncols());
    let (br, bc) = (b.nrows(), b.ncols());
    let mut result = Array2::zeros((ar * br, ac * bc));
    for i in 0..ar {
        for j in 0..ac {
            let scale = a[[i, j]];
            for k in 0..br {
                for l in 0..bc {
                    result[[i * br + k, j * bc + l]] = scale * b[[k, l]];
                }
            }
        }
    }
    result
}

/// Apply a unitary to a state vector (in-place).
///
/// For a single-qubit unitary on an n-qubit state, embed via Kronecker products
/// first. This function applies a full-dimension unitary directly.
pub fn apply_unitary(state: &mut Vec<Complex64>, unitary: &Array2<Complex64>) {
    let n = state.len();
    assert_eq!(unitary.nrows(), n);
    assert_eq!(unitary.ncols(), n);
    let old = state.clone();
    for i in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..n {
            sum += unitary[[i, j]] * old[j];
        }
        state[i] = sum;
    }
}

/// Apply a single-qubit unitary to a specific qubit in an n-qubit state vector.
fn apply_single_qubit_gate(state: &mut Vec<Complex64>, gate: &Array2<Complex64>, qubit: usize) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = 1 << num_qubits;
    let mask = 1 << qubit;
    for i in 0..dim {
        if i & mask != 0 {
            continue;
        }
        let j = i | mask;
        let a = state[i];
        let b = state[j];
        state[i] = gate[[0, 0]] * a + gate[[0, 1]] * b;
        state[j] = gate[[1, 0]] * a + gate[[1, 1]] * b;
    }
}

/// Apply a CZ gate to qubits q0 and q1.
fn apply_cz(state: &mut Vec<Complex64>, q0: usize, q1: usize) {
    let dim = state.len();
    let mask0 = 1 << q0;
    let mask1 = 1 << q1;
    for i in 0..dim {
        if (i & mask0 != 0) && (i & mask1 != 0) {
            state[i] = -state[i];
        }
    }
}

/// Generate a Haar-random SU(2) unitary matrix.
///
/// Uses the QR decomposition approach: generate a 2x2 complex Gaussian
/// matrix and orthogonalize.
pub fn random_su2(rng: &mut impl Rng) -> Array2<Complex64> {
    // Sample 4 independent standard normal values for a 2x2 complex Gaussian matrix.
    // Then use polar decomposition to get a Haar-random unitary.
    use std::f64::consts::PI;
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    let u3: f64 = rng.gen();
    // Haar measure parameterization (Ozols 2009):
    // alpha in [0, 2*pi), psi in [0, 2*pi), chi in [0, 2*pi), xi in [0, pi/2)
    let alpha = u1 * 2.0 * PI;
    let psi = u2 * 2.0 * PI;
    let xi = u3.sqrt().asin(); // Haar-uniform on SU(2)

    let ei_alpha = Complex64::from_polar(1.0, alpha);
    let ei_psi = Complex64::from_polar(1.0, psi);

    let cos_xi = xi.cos();
    let sin_xi = xi.sin();

    Array2::from_shape_vec(
        (2, 2),
        vec![
            ei_alpha * Complex64::new(cos_xi, 0.0),
            ei_psi * Complex64::new(sin_xi, 0.0),
            -ei_psi.conj() * Complex64::new(sin_xi, 0.0),
            ei_alpha.conj() * Complex64::new(cos_xi, 0.0),
        ],
    )
    .unwrap()
}

/// Build a single-qubit unitary from Euler angles (U3 gate convention):
///
/// U3(theta, phi, lambda) = [[ cos(t/2),              -exp(i*lambda)*sin(t/2) ],
///                            [ exp(i*phi)*sin(t/2),    exp(i*(phi+lambda))*cos(t/2) ]]
fn su2_from_angles(theta: f64, phi: f64, lambda: f64) -> Array2<Complex64> {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    let eip = Complex64::from_polar(1.0, phi);
    let eil = Complex64::from_polar(1.0, lambda);
    let eipl = Complex64::from_polar(1.0, phi + lambda);
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(ct, 0.0),
            -eil * Complex64::new(st, 0.0),
            eip * Complex64::new(st, 0.0),
            eipl * Complex64::new(ct, 0.0),
        ],
    )
    .unwrap()
}

/// Probability of measuring |0...0> on the full state.
pub fn measure_probability_zero(state: &[Complex64]) -> f64 {
    state[0].norm_sqr()
}

/// Sample a bitstring from a probability distribution.
fn sample_from_probs(probs: &[f64], rng: &mut impl Rng) -> usize {
    let r: f64 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

// ============================================================
// SINGLE-QUBIT CLIFFORD GROUP (24 ELEMENTS)
// ============================================================

/// Identity matrix.
fn eye2() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap()
}

/// Hadamard gate.
fn hadamard() -> Array2<Complex64> {
    let s = 1.0 / 2.0_f64.sqrt();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(-s, 0.0),
        ],
    )
    .unwrap()
}

/// S gate (phase gate).
fn s_gate() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0),
        ],
    )
    .unwrap()
}

/// X (Pauli-X) gate.
fn x_gate() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
}

/// Y (Pauli-Y) gate.
fn y_gate() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
}

/// Z (Pauli-Z) gate.
fn z_gate() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap()
}

/// T gate (pi/8 gate).
fn t_gate() -> Array2<Complex64> {
    let t = Complex64::from_polar(1.0, PI / 4.0);
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            t,
        ],
    )
    .unwrap()
}

/// Multiply two 2x2 complex matrices.
fn mat_mul_2x2(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let mut result = Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            result[[i, j]] = a[[i, 0]] * b[[0, j]] + a[[i, 1]] * b[[1, j]];
        }
    }
    result
}

/// Conjugate-transpose of a 2x2 matrix.
fn dagger_2x2(m: &Array2<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            m[[0, 0]].conj(),
            m[[1, 0]].conj(),
            m[[0, 1]].conj(),
            m[[1, 1]].conj(),
        ],
    )
    .unwrap()
}

/// Conjugate-transpose of an NxN matrix.
fn dagger(m: &Array2<Complex64>) -> Array2<Complex64> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut result = Array2::zeros((c, r));
    for i in 0..r {
        for j in 0..c {
            result[[j, i]] = m[[i, j]].conj();
        }
    }
    result
}

/// Generate all 24 single-qubit Clifford gates.
/// Generate all 24 single-qubit Clifford gates.
/// Generate all 24 single-qubit Clifford gates.
///
/// The single-qubit Clifford group C1 has 24 elements, enumerated here
/// as exact matrices. These are all products of H and S gates (which
/// generate the group).
pub fn single_qubit_cliffords() -> Vec<Array2<Complex64>> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    vec![
        // C0: I
        Array2::from_shape_vec((2,2), vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]).unwrap(),
        // C1: H
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0), Complex64::new(s, 0.0), Complex64::new(-s, 0.0)]).unwrap(),
        // C2: S
        Array2::from_shape_vec((2,2), vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)]).unwrap(),
        // C3: SH
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(0.0, s), Complex64::new(s, 0.0), Complex64::new(0.0, -s)]).unwrap(),
        // C4: HS
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0), Complex64::new(0.0, s), Complex64::new(0.0, -s)]).unwrap(),
        // C5: S^2 = Z
        Array2::from_shape_vec((2,2), vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]).unwrap(),
        // C6: HSH
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5), Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.5)]).unwrap(),
        // C7: S*HS = S(HS)
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(-s, 0.0), Complex64::new(s, 0.0), Complex64::new(s, 0.0)]).unwrap(),
        // C8: SH*S
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(0.0, s), Complex64::new(0.0, s), Complex64::new(s, 0.0)]).unwrap(),
        // C9: H*SH (= H(SH))
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0), Complex64::new(-s, 0.0), Complex64::new(s, 0.0)]).unwrap(),
        // C10: S^3 = S-dagger
        Array2::from_shape_vec((2,2), vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)]).unwrap(),
        // C11: (SH)(SH)
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.5, 0.5), Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5), Complex64::new(-0.5, 0.5)]).unwrap(),
        // C12: (HS)(HS)
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.5), Complex64::new(-0.5, 0.5)]).unwrap(),
        // C13: X
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]).unwrap(),
        // C14: S^2*HS
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(-s, 0.0), Complex64::new(0.0, s), Complex64::new(0.0, s)]).unwrap(),
        // C15: S*HS^2
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(0.0, s), Complex64::new(-s, 0.0), Complex64::new(0.0, s)]).unwrap(),
        // C16: S^3*HSH
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.5, 0.5), Complex64::new(-0.5, 0.5), Complex64::new(0.5, -0.5), Complex64::new(-0.5, -0.5)]).unwrap(),
        // C17: S^2*HSH
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5), Complex64::new(-0.5, 0.5), Complex64::new(-0.5, -0.5)]).unwrap(),
        // C18: iY rotation
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]).unwrap(),
        // C19: XS
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]).unwrap(),
        // C20: XH
        Array2::from_shape_vec((2,2), vec![Complex64::new(s, 0.0), Complex64::new(-s, 0.0), Complex64::new(-s, 0.0), Complex64::new(-s, 0.0)]).unwrap(),
        // C21: S(XH)
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.0, s), Complex64::new(s, 0.0), Complex64::new(0.0, -s), Complex64::new(s, 0.0)]).unwrap(),
        // C22: (SH)(S^2*H)
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.5, 0.5), Complex64::new(-0.5, -0.5), Complex64::new(0.5, -0.5), Complex64::new(0.5, -0.5)]).unwrap(),
        // C23: -iY
        Array2::from_shape_vec((2,2), vec![Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]).unwrap(),
    ]
}
/// Check if two unitaries are equal up to a global phase.
fn matrices_equal_up_to_phase(a: &Array2<Complex64>, b: &Array2<Complex64>, tol: f64) -> bool {
    // Find the first nonzero entry to extract the relative phase.
    let mut phase = None;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if a[[i, j]].norm() > tol && b[[i, j]].norm() > tol {
                phase = Some(b[[i, j]] / a[[i, j]]);
                break;
            }
        }
        if phase.is_some() {
            break;
        }
    }
    let phase = match phase {
        Some(p) => p,
        None => return true, // both zero
    };
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = (a[[i, j]] * phase - b[[i, j]]).norm();
            if diff > tol {
                return false;
            }
        }
    }
    true
}

// ============================================================
// XEB IMPLEMENTATION
// ============================================================

/// Generate a random quantum circuit.
///
/// Each layer consists of random single-qubit gates on all qubits,
/// followed by CZ gates on pairs of adjacent qubits (alternating even/odd
/// pairs each layer, mimicking Google's Sycamore pattern).
pub fn generate_random_circuit(
    num_qubits: usize,
    depth: usize,
    rng: &mut impl Rng,
) -> RandomCircuit {
    let mut gates = Vec::new();

    for layer in 0..depth {
        // Single-qubit layer: random SU(2) on each qubit.
        for q in 0..num_qubits {
            let u = random_su2(rng);
            let params = [
                u[[0, 0]].re.atan2(u[[0, 0]].im),
                u[[0, 1]].re.atan2(u[[0, 1]].im),
                u[[1, 0]].re.atan2(u[[1, 0]].im),
            ];
            gates.push(RandomGate::SingleQubit {
                qubit: q,
                gate_type: SingleGateType::RandomSU2,
                params,
            });
        }

        // Two-qubit entangling layer: CZ on adjacent pairs.
        // Alternate even-odd and odd-even pairs.
        let start = if layer % 2 == 0 { 0 } else { 1 };
        let mut q = start;
        while q + 1 < num_qubits {
            gates.push(RandomGate::TwoQubit { q0: q, q1: q + 1 });
            q += 2;
        }
    }

    RandomCircuit {
        gates,
        num_qubits,
        depth,
    }
}

/// Build the single-qubit unitary for a RandomGate::SingleQubit.
fn gate_unitary(gate_type: SingleGateType, params: &[f64; 3]) -> Array2<Complex64> {
    match gate_type {
        SingleGateType::RandomSU2 => su2_from_angles(params[0], params[1], params[2]),
        SingleGateType::Hadamard => hadamard(),
        SingleGateType::T => t_gate(),
        SingleGateType::S => s_gate(),
    }
}

/// Simulate a random circuit ideally (state-vector) and return probabilities.
pub fn simulate_ideal(circuit: &RandomCircuit) -> Vec<f64> {
    let dim = 1 << circuit.num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0); // |0...0>

    for gate in &circuit.gates {
        match gate {
            RandomGate::SingleQubit {
                qubit,
                gate_type,
                params,
            } => {
                let u = gate_unitary(*gate_type, params);
                apply_single_qubit_gate(&mut state, &u, *qubit);
            }
            RandomGate::TwoQubit { q0, q1 } => {
                apply_cz(&mut state, *q0, *q1);
            }
        }
    }

    state.iter().map(|a| a.norm_sqr()).collect()
}

/// Compute XEB fidelity: F_XEB = 2^n * mean(p_ideal(x_i)) - 1
///
/// `ideal_probs` are the ideal output probabilities (length 2^n).
/// `samples` are bitstring indices drawn from the (possibly noisy) device.
pub fn compute_xeb_fidelity(ideal_probs: &[f64], samples: &[usize]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let n_states = ideal_probs.len() as f64;
    let mean_p: f64 = samples.iter().map(|&s| ideal_probs[s]).sum::<f64>() / samples.len() as f64;
    n_states * mean_p - 1.0
}

/// Euler-Mascheroni constant.
const EULER_GAMMA: f64 = 0.5772156649015329;

/// Compute logarithmic XEB fidelity.
fn compute_log_xeb(ideal_probs: &[f64], samples: &[usize]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let n_states = ideal_probs.len() as f64;
    let mean_log: f64 = samples
        .iter()
        .map(|&s| {
            let p = ideal_probs[s];
            if p > 1e-15 {
                (n_states * p).ln()
            } else {
                -30.0 // floor for log(0)
            }
        })
        .sum::<f64>()
        / samples.len() as f64;
    mean_log + EULER_GAMMA
}

/// Run a full XEB experiment.
pub fn run_xeb(config: &XebConfig) -> Result<XebResult> {
    if config.num_qubits > 20 {
        return Err(QcvvError::TooManyQubits(config.num_qubits));
    }
    if config.num_circuits == 0 || config.num_samples == 0 {
        return Err(QcvvError::InvalidConfig(
            "num_circuits and num_samples must be > 0".into(),
        ));
    }

    let mut rng = rand::thread_rng();
    let mut fidelities = Vec::with_capacity(config.num_circuits);
    let mut log_xebs = Vec::with_capacity(config.num_circuits);

    for _ in 0..config.num_circuits {
        let circuit = generate_random_circuit(config.num_qubits, config.circuit_depth, &mut rng);
        let ideal_probs = simulate_ideal(&circuit);

        // For ideal XEB (no noise model), sample from ideal distribution.
        let samples: Vec<usize> = (0..config.num_samples)
            .map(|_| sample_from_probs(&ideal_probs, &mut rng))
            .collect();

        let f = compute_xeb_fidelity(&ideal_probs, &samples);
        let l = compute_log_xeb(&ideal_probs, &samples);
        fidelities.push(f);
        log_xebs.push(l);
    }

    let xeb_fidelity = fidelities.iter().sum::<f64>() / fidelities.len() as f64;
    let log_xeb = log_xebs.iter().sum::<f64>() / log_xebs.len() as f64;

    Ok(XebResult {
        xeb_fidelity,
        linear_xeb: xeb_fidelity,
        log_xeb,
        num_circuits: config.num_circuits,
        circuit_depths: vec![config.circuit_depth],
        fidelities_by_depth: vec![xeb_fidelity],
    })
}

/// Scan XEB fidelity across multiple circuit depths.
pub fn xeb_depth_scan(
    num_qubits: usize,
    depths: &[usize],
    num_circuits: usize,
) -> Result<Vec<(usize, f64)>> {
    let mut results = Vec::with_capacity(depths.len());
    for &d in depths {
        let config = XebConfig::new(num_qubits)
            .circuit_depth(d)
            .num_circuits(num_circuits)
            .num_samples(500);
        let r = run_xeb(&config)?;
        results.push((d, r.xeb_fidelity));
    }
    Ok(results)
}

// ============================================================
// RANDOMIZED BENCHMARKING
// ============================================================

/// Generate a random single-qubit Clifford gate.
pub fn generate_clifford_1q(rng: &mut impl Rng) -> CliffordGate {
    let cliffords = single_qubit_cliffords();
    let idx = rng.gen_range(0..cliffords.len());
    CliffordGate {
        matrix: cliffords[idx].clone(),
        name: format!("C1_{}", idx),
    }
}

/// Generate a random two-qubit Clifford gate.
///
/// The two-qubit Clifford group has 11520 elements. We approximate by
/// composing random single-qubit Cliffords with entangling CZ layers.
pub fn generate_clifford_2q(rng: &mut impl Rng) -> CliffordGate {
    let c0 = generate_clifford_1q(rng);
    let c1 = generate_clifford_1q(rng);
    let local = kronecker_product(&c0.matrix, &c1.matrix);

    // CZ gate as 4x4 matrix
    let cz = {
        let mut m = Array2::<Complex64>::eye(4);
        m[[3, 3]] = Complex64::new(-1.0, 0.0);
        m
    };

    // Choose structure: local-only, CZ, or CZ-local-CZ
    let choice = rng.gen_range(0..3);
    let matrix = match choice {
        0 => local,
        1 => mat_mul_nxn(&cz, &local),
        _ => {
            let c2 = generate_clifford_1q(rng);
            let c3 = generate_clifford_1q(rng);
            let local2 = kronecker_product(&c2.matrix, &c3.matrix);
            mat_mul_nxn(&cz, &mat_mul_nxn(&local2, &mat_mul_nxn(&cz, &local)))
        }
    };

    CliffordGate {
        matrix,
        name: format!("C2_rand"),
    }
}

/// Multiply two NxN complex matrices.
fn mat_mul_nxn(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n);
    assert_eq!(b.nrows(), n);
    assert_eq!(b.ncols(), n);
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

/// Compute the inverse of a Clifford gate sequence.
///
/// For a sequence C_m, C_{m-1}, ..., C_1, the product is U = C_m * ... * C_1.
/// The inverse is U† (conjugate transpose).
pub fn clifford_inverse(gates: &[CliffordGate]) -> CliffordGate {
    if gates.is_empty() {
        return CliffordGate {
            matrix: Array2::eye(2),
            name: "I".into(),
        };
    }
    let n = gates[0].matrix.nrows();
    let mut product = Array2::<Complex64>::eye(n);
    for g in gates {
        product = mat_mul_nxn(&g.matrix, &product);
    }
    let inv = dagger(&product);
    CliffordGate {
        matrix: inv,
        name: "C_inv".into(),
    }
}

/// Fit an exponential decay model P(m) = A * p^m + B using least-squares.
///
/// Returns (A, p, B). Uses a simple grid search + refinement approach
/// since this is a 3-parameter nonlinear fit.
pub fn fit_exponential_decay(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    assert_eq!(x.len(), y.len());
    assert!(!x.is_empty());

    // Special case: if all y-values are nearly constant (no decay), return p=1.0.
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (y_max - y_min).abs() < 1e-4 {
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        // No decay: A=0, p=1.0, B=mean(y)
        return (0.0, 1.0, mean_y);
    }


    // Initial estimates
    let b_est = *y.last().unwrap_or(&0.5);
    let a_est = y[0] - b_est;

    // Estimate p from first two points if available.
    let p_est = if x.len() >= 2 && a_est.abs() > 1e-12 {
        let ratio = (y[1] - b_est) / a_est;
        let dx = x[1] - x[0];
        if ratio > 0.0 && dx > 0.0 {
            ratio.powf(1.0 / dx).clamp(0.0, 1.0)
        } else {
            0.95
        }
    } else {
        0.95
    };

    // Refine with grid search around initial estimates.
    let mut best_a = a_est;
    let mut best_p = p_est;
    let mut best_b = b_est;
    let mut best_err = f64::MAX;

    for &a_off in &[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2] {
        for p_try in 0..21 {
            let p = (p_est - 0.2 + 0.02 * p_try as f64).clamp(0.001, 0.9999);
            for &b_off in &[-0.1, -0.05, 0.0, 0.05, 0.1] {
                let a = a_est + a_off;
                let b = b_est + b_off;
                let err: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| {
                        let pred = a * p.powf(xi) + b;
                        (pred - yi).powi(2)
                    })
                    .sum();
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_p = p;
                    best_b = b;
                }
            }
        }
    }

    // Second refinement pass (narrow grid).
    let (ca, cp, cb) = (best_a, best_p, best_b);
    for a_step in -10..=10 {
        for p_step in -10..=10 {
            for b_step in -10..=10 {
                let a = ca + 0.01 * a_step as f64;
                let p = (cp + 0.005 * p_step as f64).clamp(0.001, 0.9999);
                let b = cb + 0.01 * b_step as f64;
                let err: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| {
                        let pred = a * p.powf(xi) + b;
                        (pred - yi).powi(2)
                    })
                    .sum();
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_p = p;
                    best_b = b;
                }
            }
        }
    }

    (best_a, best_p, best_b)
}

/// Run a single-qubit or two-qubit randomized benchmarking experiment.
pub fn run_rb(config: &RbConfig) -> Result<RbResult> {
    if config.num_qubits == 0 || config.num_qubits > 2 {
        return Err(QcvvError::InvalidConfig(
            "RB supports 1 or 2 qubits".into(),
        ));
    }
    let dim = 1 << config.num_qubits;
    let mut rng = rand::thread_rng();

    let mut survival_probs = Vec::with_capacity(config.sequence_lengths.len());

    for &seq_len in &config.sequence_lengths {
        let mut total_survival = 0.0;

        for _ in 0..config.num_sequences {
            // Generate random Clifford sequence.
            let gates: Vec<CliffordGate> = (0..seq_len)
                .map(|_| {
                    if config.num_qubits == 1 {
                        generate_clifford_1q(&mut rng)
                    } else {
                        generate_clifford_2q(&mut rng)
                    }
                })
                .collect();

            // Compute the inverse Clifford.
            let inv = clifford_inverse(&gates);

            // Prepare |0...0>.
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            state[0] = Complex64::new(1.0, 0.0);

            // Apply each gate, then the inverse.
            for g in &gates {
                apply_unitary(&mut state, &g.matrix);
            }
            apply_unitary(&mut state, &inv.matrix);

            // Measure survival probability.
            total_survival += measure_probability_zero(&state);
        }

        survival_probs.push(total_survival / config.num_sequences as f64);
    }

    // Fit exponential decay.
    let x: Vec<f64> = config.sequence_lengths.iter().map(|&l| l as f64).collect();
    let (fit_a, decay_rate, fit_b) = fit_exponential_decay(&x, &survival_probs);

    // Error per Clifford.
    let error_per_gate = (1.0 - decay_rate) * (dim as f64 - 1.0) / dim as f64;

    Ok(RbResult {
        error_per_gate,
        decay_rate,
        fit_a,
        fit_b,
        sequence_lengths: config.sequence_lengths.clone(),
        survival_probabilities: survival_probs,
    })
}

// ============================================================
// INTERLEAVED RANDOMIZED BENCHMARKING
// ============================================================

/// Run interleaved RB to isolate the error of a specific gate.
///
/// Inserts `gate_under_test` between every pair of random Cliffords.
pub fn run_interleaved_rb(config: &RbConfig, gate_under_test: &CliffordGate) -> Result<RbResult> {
    if config.num_qubits == 0 || config.num_qubits > 2 {
        return Err(QcvvError::InvalidConfig(
            "Interleaved RB supports 1 or 2 qubits".into(),
        ));
    }
    let dim = 1 << config.num_qubits;
    let mut rng = rand::thread_rng();

    let mut survival_probs = Vec::with_capacity(config.sequence_lengths.len());

    for &seq_len in &config.sequence_lengths {
        let mut total_survival = 0.0;

        for _ in 0..config.num_sequences {
            // Build interleaved sequence: C_1, G, C_2, G, ..., C_m, G
            let random_cliffords: Vec<CliffordGate> = (0..seq_len)
                .map(|_| {
                    if config.num_qubits == 1 {
                        generate_clifford_1q(&mut rng)
                    } else {
                        generate_clifford_2q(&mut rng)
                    }
                })
                .collect();

            // Build the full interleaved sequence for inverse computation.
            let mut full_sequence = Vec::with_capacity(seq_len * 2);
            for c in &random_cliffords {
                full_sequence.push(c.clone());
                full_sequence.push(gate_under_test.clone());
            }

            let inv = clifford_inverse(&full_sequence);

            // Simulate.
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            state[0] = Complex64::new(1.0, 0.0);

            for g in &full_sequence {
                apply_unitary(&mut state, &g.matrix);
            }
            apply_unitary(&mut state, &inv.matrix);

            total_survival += measure_probability_zero(&state);
        }

        survival_probs.push(total_survival / config.num_sequences as f64);
    }

    let x: Vec<f64> = config.sequence_lengths.iter().map(|&l| l as f64).collect();
    let (fit_a, decay_rate, fit_b) = fit_exponential_decay(&x, &survival_probs);
    let error_per_gate = (1.0 - decay_rate) * (dim as f64 - 1.0) / dim as f64;

    Ok(RbResult {
        error_per_gate,
        decay_rate,
        fit_a,
        fit_b,
        sequence_lengths: config.sequence_lengths.clone(),
        survival_probabilities: survival_probs,
    })
}

/// Extract the error rate of a specific gate from reference and interleaved RB.
///
/// gate_error = (1 - p_interleaved / p_reference) * (2^n - 1) / 2^n
pub fn gate_error_from_interleaved(rb_ref: &RbResult, rb_interleaved: &RbResult, num_qubits: usize) -> f64 {
    let dim = (1 << num_qubits) as f64;
    if rb_ref.decay_rate.abs() < 1e-15 {
        return 1.0;
    }
    let ratio = rb_interleaved.decay_rate / rb_ref.decay_rate;
    (1.0 - ratio) * (dim - 1.0) / dim
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    type StdRng = rand::rngs::StdRng;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // 1. XebConfig builder defaults
    #[test]
    fn test_xeb_config_defaults() {
        let cfg = XebConfig::new(4);
        assert_eq!(cfg.num_qubits, 4);
        assert_eq!(cfg.num_circuits, 20);
        assert_eq!(cfg.circuit_depth, 20);
        assert_eq!(cfg.num_samples, 1000);

        let cfg2 = XebConfig::new(3).num_circuits(10).circuit_depth(5).num_samples(500);
        assert_eq!(cfg2.num_circuits, 10);
        assert_eq!(cfg2.circuit_depth, 5);
        assert_eq!(cfg2.num_samples, 500);
    }

    // 2. Random circuit generation produces valid circuits
    #[test]
    fn test_random_circuit_generation() {
        let mut rng = StdRng::seed_from_u64(42);
        let circuit = generate_random_circuit(3, 5, &mut rng);
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.depth, 5);
        assert!(!circuit.gates.is_empty());

        // Verify we have both single-qubit and two-qubit gates.
        let has_single = circuit.gates.iter().any(|g| matches!(g, RandomGate::SingleQubit { .. }));
        let has_two = circuit.gates.iter().any(|g| matches!(g, RandomGate::TwoQubit { .. }));
        assert!(has_single, "circuit must contain single-qubit gates");
        assert!(has_two, "circuit must contain two-qubit gates");

        // All qubit indices should be in range.
        for gate in &circuit.gates {
            match gate {
                RandomGate::SingleQubit { qubit, .. } => {
                    assert!(*qubit < 3, "qubit index out of range");
                }
                RandomGate::TwoQubit { q0, q1 } => {
                    assert!(*q0 < 3 && *q1 < 3, "qubit index out of range");
                    assert_ne!(q0, q1, "two-qubit gate on same qubit");
                }
            }
        }
    }

    // 3. Ideal simulation of Hadamard circuit gives 50/50
    #[test]
    fn test_hadamard_simulation() {
        // Single Hadamard on qubit 0 of a 1-qubit system.
        let circuit = RandomCircuit {
            gates: vec![RandomGate::SingleQubit {
                qubit: 0,
                gate_type: SingleGateType::Hadamard,
                params: [0.0; 3],
            }],
            num_qubits: 1,
            depth: 1,
        };
        let probs = simulate_ideal(&circuit);
        assert_eq!(probs.len(), 2);
        assert!(approx_eq(probs[0], 0.5, 1e-10), "p(0) = {}", probs[0]);
        assert!(approx_eq(probs[1], 0.5, 1e-10), "p(1) = {}", probs[1]);
    }

    // 4. XEB fidelity = 1.0 for ideal (noiseless) simulation
    #[test]
    fn test_xeb_ideal_fidelity() {
        // For ideal sampling, XEB fidelity should be near 1.0 (within statistical noise).
        let config = XebConfig::new(2)
            .num_circuits(30)
            .circuit_depth(4)
            .num_samples(5000);
        let result = run_xeb(&config).unwrap();
        // With ideal sampling, F_XEB should be close to 1.0.
        // Statistical noise means we accept a generous tolerance.
        assert!(
            result.xeb_fidelity > 0.5,
            "ideal XEB fidelity should be high, got {}",
            result.xeb_fidelity
        );
    }

    // 5. XEB fidelity decreases with added depolarizing noise
    #[test]
    fn test_xeb_with_noise() {
        let mut rng = StdRng::seed_from_u64(123);
        let num_qubits = 2;
        let dim = 1 << num_qubits;
        let circuit = generate_random_circuit(num_qubits, 4, &mut rng);
        let ideal_probs = simulate_ideal(&circuit);

        // Ideal samples
        let ideal_samples: Vec<usize> = (0..2000)
            .map(|_| sample_from_probs(&ideal_probs, &mut rng))
            .collect();
        let ideal_fidelity = compute_xeb_fidelity(&ideal_probs, &ideal_samples);

        // Noisy samples: mix ideal distribution with uniform (depolarizing noise).
        let noise_rate = 0.5;
        let uniform = 1.0 / dim as f64;
        let noisy_probs: Vec<f64> = ideal_probs
            .iter()
            .map(|&p| (1.0 - noise_rate) * p + noise_rate * uniform)
            .collect();
        let noisy_samples: Vec<usize> = (0..2000)
            .map(|_| sample_from_probs(&noisy_probs, &mut rng))
            .collect();
        let noisy_fidelity = compute_xeb_fidelity(&ideal_probs, &noisy_samples);

        assert!(
            noisy_fidelity < ideal_fidelity,
            "noisy fidelity ({}) should be less than ideal ({})",
            noisy_fidelity,
            ideal_fidelity
        );
    }

    // 6. All 24 single-qubit Cliffords are unitary
    #[test]
    fn test_cliffords_are_unitary() {
        let cliffords = single_qubit_cliffords();
        assert!(
            cliffords.len() >= 24,
            "expected at least 24 Cliffords, got {}",
            cliffords.len()
        );

        for (i, c) in cliffords.iter().enumerate() {
            let prod = mat_mul_2x2(c, &dagger_2x2(c));
            for r in 0..2 {
                for col in 0..2 {
                    let expected = if r == col { 1.0 } else { 0.0 };
                    assert!(
                        approx_eq(prod[[r, col]].re, expected, 1e-10)
                            && approx_eq(prod[[r, col]].im, 0.0, 1e-10),
                        "Clifford {} is not unitary: U*U† [{},{}] = {:?}",
                        i,
                        r,
                        col,
                        prod[[r, col]]
                    );
                }
            }
        }
    }

    // 7. Clifford inverse: C * C^{-1} = I
    #[test]
    fn test_clifford_inverse() {
        let mut rng = StdRng::seed_from_u64(99);
        let gates: Vec<CliffordGate> = (0..5).map(|_| generate_clifford_1q(&mut rng)).collect();
        let inv = clifford_inverse(&gates);

        // Compute product: C_5 * C_4 * ... * C_1
        let mut product = Array2::<Complex64>::eye(2);
        for g in &gates {
            product = mat_mul_2x2(&g.matrix, &product);
        }

        // product * inv should be identity.
        let result = mat_mul_2x2(&product, &inv.matrix);
        // Check up to global phase: result should be proportional to I.
        assert!(
            matrices_equal_up_to_phase(&result, &eye2(), 1e-8),
            "C * C^{{-1}} should be identity (up to phase)"
        );
    }

    // 8. RB survival probability = 1.0 for noiseless simulation
    #[test]
    fn test_rb_noiseless() {
        let config = RbConfig::new(1)
            .sequence_lengths(vec![1, 2, 4, 8, 16])
            .num_sequences(20)
            .num_samples_per_sequence(1);
        let result = run_rb(&config).unwrap();

        // In noiseless simulation, survival should be ~1.0 for all lengths.
        for (i, &prob) in result.survival_probabilities.iter().enumerate() {
            assert!(
                approx_eq(prob, 1.0, 1e-6),
                "survival at length {} = {}, expected ~1.0",
                config.sequence_lengths[i],
                prob
            );
        }

        // Error per gate should be ~0.
        assert!(
            result.error_per_gate < 0.05,
            "noiseless error_per_gate = {}, expected ~0",
            result.error_per_gate
        );
    }

    // 9. RB exponential fit extracts known error rate
    #[test]
    fn test_exponential_fit() {
        // Generate synthetic data: P(m) = 0.5 * 0.95^m + 0.5
        let known_a: f64 = 0.5;
        let known_p: f64 = 0.95;
        let known_b: f64 = 0.5;
        let x: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let y: Vec<f64> = x.iter().map(|&m: &f64| known_a * known_p.powf(m) + known_b).collect();

        let (fit_a, fit_p, fit_b) = fit_exponential_decay(&x, &y);

        assert!(
            approx_eq(fit_a, known_a, 0.05),
            "fit_a = {}, expected {}",
            fit_a,
            known_a
        );
        assert!(
            approx_eq(fit_p, known_p, 0.05),
            "fit_p = {}, expected {}",
            fit_p,
            known_p
        );
        assert!(
            approx_eq(fit_b, known_b, 0.05),
            "fit_b = {}, expected {}",
            fit_b,
            known_b
        );
    }

    // 10. Interleaved RB isolates gate error
    #[test]
    fn test_interleaved_rb() {
        // Test that interleaved RB runs and produces reasonable output.
        let config = RbConfig::new(1)
            .sequence_lengths(vec![1, 2, 4, 8])
            .num_sequences(10)
            .num_samples_per_sequence(1);

        let identity_gate = CliffordGate {
            matrix: eye2(),
            name: "I".into(),
        };

        let rb_ref = run_rb(&config).unwrap();
        let rb_interleaved = run_interleaved_rb(&config, &identity_gate).unwrap();

        // With noiseless identity gate, the interleaved error should be near 0.
        let gate_err = gate_error_from_interleaved(&rb_ref, &rb_interleaved, 1);
        assert!(
            gate_err.abs() < 0.1,
            "identity gate error = {}, expected ~0",
            gate_err
        );
    }

    // 11. Haar-random unitaries are unitary (U†U = I)
    #[test]
    fn test_haar_random_unitary() {
        let mut rng = StdRng::seed_from_u64(77);
        for _ in 0..20 {
            let u = random_su2(&mut rng);
            let prod = mat_mul_2x2(&dagger_2x2(&u), &u);
            for r in 0..2 {
                for c in 0..2 {
                    let expected = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        approx_eq(prod[[r, c]].re, expected, 1e-10)
                            && approx_eq(prod[[r, c]].im, 0.0, 1e-10),
                        "U†U is not identity: [{},{}] = {:?}",
                        r,
                        c,
                        prod[[r, c]]
                    );
                }
            }
        }
    }

    // 12. Kronecker product dimensions are correct
    #[test]
    fn test_kronecker_dimensions() {
        let a = Array2::<Complex64>::eye(2);
        let b = Array2::<Complex64>::eye(3);
        let k = kronecker_product(&a, &b);
        assert_eq!(k.nrows(), 6);
        assert_eq!(k.ncols(), 6);

        // I_2 ⊗ I_3 = I_6
        for r in 0..6 {
            for c in 0..6 {
                let expected = if r == c {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    approx_eq(k[[r, c]].re, expected.re, 1e-10)
                        && approx_eq(k[[r, c]].im, expected.im, 1e-10),
                    "I_2 ⊗ I_3 [{},{}] = {:?}, expected {:?}",
                    r,
                    c,
                    k[[r, c]],
                    expected
                );
            }
        }
    }

    // 13. Probability normalization in simulate_ideal
    #[test]
    fn test_simulate_ideal_normalization() {
        let mut rng = StdRng::seed_from_u64(55);
        let circuit = generate_random_circuit(3, 4, &mut rng);
        let probs = simulate_ideal(&circuit);
        let total: f64 = probs.iter().sum();
        assert!(
            approx_eq(total, 1.0, 1e-10),
            "probabilities should sum to 1.0, got {}",
            total
        );
        for &p in &probs {
            assert!(p >= 0.0, "probability should be non-negative, got {}", p);
        }
    }

    // 14. XEB depth scan returns correct number of points
    #[test]
    fn test_xeb_depth_scan() {
        let depths = vec![1, 2, 3];
        let results = xeb_depth_scan(2, &depths, 5).unwrap();
        assert_eq!(results.len(), 3);
        for (i, &(d, _f)) in results.iter().enumerate() {
            assert_eq!(d, depths[i]);
        }
    }

    // 15. RB config builder
    #[test]
    fn test_rb_config_defaults() {
        let cfg = RbConfig::new(1);
        assert_eq!(cfg.num_qubits, 1);
        assert_eq!(cfg.sequence_lengths, vec![1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(cfg.num_sequences, 30);
        assert_eq!(cfg.num_samples_per_sequence, 100);
    }
}
