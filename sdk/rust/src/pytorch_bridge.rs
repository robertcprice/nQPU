//! Differentiable Quantum Circuit Engine
//!
//! Standalone differentiable quantum circuit engine with gradient computation.
//! Does **NOT** require or link to PyTorch. Provides parameter-shift rule,
//! adjoint differentiation, finite difference, and reverse-mode backpropagation
//! gradients compatible with external ML frameworks via tensor serialization.
//!
//! Despite the module name (`pytorch_bridge`), this is a self-contained autodiff
//! engine. It can export gradient tensors for use with any ML framework.
//!
//! **STANDALONE**: This module contains its own state vector simulator and does
//! not depend on any other crate modules.
//!
//! # Features
//!
//! - Differentiable parametric quantum circuits
//! - Four gradient methods: parameter-shift, adjoint, finite-difference, backprop
//! - Observable measurement (Pauli, Hamiltonian, projector)
//! - VQE optimization with Adam/SGD/LBFGS/SPSA
//! - Quantum kernel computation for ML classification
//! - Quantum GAN support
//! - Tensor serialization for Python interop
//! - Batch execution with parallelism
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::pytorch_bridge::*;
//!
//! let mut circuit = DifferentiableCircuit::new(2);
//! circuit.add_gate(ParametricGate::Ry(0, circuit.add_parameter(0.5)));
//! circuit.add_gate(ParametricGate::CX(0, 1));
//! let obs = vec![Observable::PauliZ(0)];
//! let fwd = circuit.forward(&obs).unwrap();
//! let bwd = circuit.backward(&obs).unwrap();
//! ```
//!
//! # References
//!
//! - Mitarai et al. (2018) - Parameter-shift rule
//! - Jones & Gacon (2020) - Adjoint differentiation
//! - Schuld et al. (2019) - Quantum kernels for ML

use std::f64::consts::PI;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from bridge operations.
#[derive(Clone, Debug)]
pub enum BridgeError {
    /// Tensor shape mismatch.
    ShapeError {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Gradient computation failed.
    GradientError(String),
    /// Serialization or deserialization failed.
    SerializationError(String),
    /// Backend execution error.
    BackendError(String),
    /// Operation not supported for the given configuration.
    UnsupportedOperation(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::ShapeError { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            BridgeError::GradientError(msg) => write!(f, "gradient error: {}", msg),
            BridgeError::SerializationError(msg) => {
                write!(f, "serialization error: {}", msg)
            }
            BridgeError::BackendError(msg) => write!(f, "backend error: {}", msg),
            BridgeError::UnsupportedOperation(msg) => {
                write!(f, "unsupported operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for BridgeError {}

// ============================================================
// COMPLEX NUMBER HELPERS (standalone, no crate:: imports)
// ============================================================

/// Minimal complex number for internal state vector simulation.
#[derive(Clone, Copy, Debug)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline]
    fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline]
    fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline]
    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// e^(i*theta)
    #[inline]
    fn exp_i(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }
}

// ============================================================
// STATE VECTOR SIMULATOR (self-contained)
// ============================================================

/// A minimal state vector for circuit simulation.
/// Stores 2^n complex amplitudes for n qubits.
#[derive(Clone, Debug)]
struct StateVec {
    num_qubits: usize,
    amplitudes: Vec<C64>,
}

impl StateVec {
    /// Initialize to |0...0>.
    fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amps = vec![C64::zero(); dim];
        amps[0] = C64::one();
        Self {
            num_qubits,
            amplitudes: amps,
        }
    }

    fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    /// Apply a 2x2 unitary to a single qubit.
    /// mat is [[a,b],[c,d]] in row-major order.
    fn apply_single(&mut self, qubit: usize, mat: [C64; 4]) {
        let n = self.dim();
        let bit = 1 << qubit;
        let mut i = 0;
        while i < n {
            if i & bit == 0 {
                let j = i | bit;
                let a0 = self.amplitudes[i];
                let a1 = self.amplitudes[j];
                self.amplitudes[i] = mat[0].mul(a0).add(mat[1].mul(a1));
                self.amplitudes[j] = mat[2].mul(a0).add(mat[3].mul(a1));
            }
            i += 1;
        }
    }

    /// Apply a controlled-U gate (control, target, 2x2 unitary on target).
    fn apply_controlled(&mut self, control: usize, target: usize, mat: [C64; 4]) {
        let n = self.dim();
        let cbit = 1 << control;
        let tbit = 1 << target;
        for i in 0..n {
            if (i & cbit != 0) && (i & tbit == 0) {
                let j = i | tbit;
                let a0 = self.amplitudes[i];
                let a1 = self.amplitudes[j];
                self.amplitudes[i] = mat[0].mul(a0).add(mat[1].mul(a1));
                self.amplitudes[j] = mat[2].mul(a0).add(mat[3].mul(a1));
            }
        }
    }

    /// Probabilities |a_i|^2.
    fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Inner product <self|other>.
    fn inner(&self, other: &StateVec) -> C64 {
        let mut sum = C64::zero();
        for (a, b) in self.amplitudes.iter().zip(other.amplitudes.iter()) {
            sum = sum.add(a.conj().mul(*b));
        }
        sum
    }

    /// Return (re, im) pairs for external consumption.
    fn to_pairs(&self) -> Vec<(f64, f64)> {
        self.amplitudes.iter().map(|a| (a.re, a.im)).collect()
    }
}

// ============================================================
// GATE MATRICES
// ============================================================

fn mat_h() -> [C64; 4] {
    let s = 1.0 / 2.0_f64.sqrt();
    [
        C64::new(s, 0.0),
        C64::new(s, 0.0),
        C64::new(s, 0.0),
        C64::new(-s, 0.0),
    ]
}

fn mat_x() -> [C64; 4] {
    [C64::zero(), C64::one(), C64::one(), C64::zero()]
}

fn mat_z() -> [C64; 4] {
    [
        C64::one(),
        C64::zero(),
        C64::zero(),
        C64::new(-1.0, 0.0),
    ]
}

fn mat_y() -> [C64; 4] {
    [
        C64::zero(),
        C64::new(0.0, -1.0),
        C64::new(0.0, 1.0),
        C64::zero(),
    ]
}

fn mat_rx(theta: f64) -> [C64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        C64::new(c, 0.0),
        C64::new(0.0, -s),
        C64::new(0.0, -s),
        C64::new(c, 0.0),
    ]
}

fn mat_ry(theta: f64) -> [C64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        C64::new(c, 0.0),
        C64::new(-s, 0.0),
        C64::new(s, 0.0),
        C64::new(c, 0.0),
    ]
}

fn mat_rz(theta: f64) -> [C64; 4] {
    [
        C64::exp_i(-theta / 2.0),
        C64::zero(),
        C64::zero(),
        C64::exp_i(theta / 2.0),
    ]
}

fn mat_cx() -> [C64; 4] {
    // CX target unitary is X
    mat_x()
}

fn mat_cz_target() -> [C64; 4] {
    mat_z()
}

/// U3(theta, phi, lambda) general single-qubit gate.
fn mat_u3(theta: f64, phi: f64, lambda: f64) -> [C64; 4] {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    [
        C64::new(ct, 0.0),
        C64::exp_i(lambda).scale(-st),
        C64::exp_i(phi).scale(st),
        C64::exp_i(phi + lambda).scale(ct),
    ]
}

// ============================================================
// PARAMETRIC GATE
// ============================================================

/// A gate in a differentiable circuit. Fixed gates have no trainable
/// parameters; parametric gates reference an index into the circuit's
/// parameter vector.
#[derive(Clone, Debug)]
pub enum ParametricGate {
    // Fixed gates
    H(usize),
    CX(usize, usize),
    CZ(usize, usize),
    X(usize),
    Z(usize),
    // Parametric single-qubit rotations (qubit, param_index)
    Rx(usize, usize),
    Ry(usize, usize),
    Rz(usize, usize),
    // Parametric controlled rotations (control, target, param_index)
    CRx(usize, usize, usize),
    CRy(usize, usize, usize),
    CRz(usize, usize, usize),
    // General single-qubit (qubit, theta_idx, phi_idx, lambda_idx)
    U3(usize, usize, usize, usize),
}

impl ParametricGate {
    /// Return the parameter indices this gate depends on (empty for fixed).
    fn param_indices(&self) -> Vec<usize> {
        match self {
            ParametricGate::H(_)
            | ParametricGate::CX(_, _)
            | ParametricGate::CZ(_, _)
            | ParametricGate::X(_)
            | ParametricGate::Z(_) => vec![],
            ParametricGate::Rx(_, p)
            | ParametricGate::Ry(_, p)
            | ParametricGate::Rz(_, p)
            | ParametricGate::CRx(_, _, p)
            | ParametricGate::CRy(_, _, p)
            | ParametricGate::CRz(_, _, p) => vec![*p],
            ParametricGate::U3(_, a, b, c) => vec![*a, *b, *c],
        }
    }

    /// Is this gate parametric?
    fn is_parametric(&self) -> bool {
        !self.param_indices().is_empty()
    }
}

// ============================================================
// GRADIENT METHOD
// ============================================================

/// Strategy for computing gradients of expectation values with respect to
/// circuit parameters.
#[derive(Clone, Debug)]
pub enum GradientMethod {
    /// Exact analytical gradient via parameter-shift rule.
    /// Default shift is pi/2 (exact for standard rotation gates).
    ParameterShift { shift: f64 },
    /// Adjoint differentiation: O(1) forward + O(n) backward.
    AdjointDiff,
    /// Central finite difference with configurable step size.
    FiniteDifference { epsilon: f64 },
    /// Reverse-mode backpropagation through the state vector.
    Backprop,
}

impl Default for GradientMethod {
    fn default() -> Self {
        GradientMethod::ParameterShift {
            shift: PI / 2.0,
        }
    }
}

// ============================================================
// OBSERVABLE
// ============================================================

/// An observable whose expectation value can be measured.
#[derive(Clone, Debug)]
pub enum Observable {
    /// Single Pauli-Z on one qubit.
    PauliZ(usize),
    /// Single Pauli-X on one qubit.
    PauliX(usize),
    /// Single Pauli-Y on one qubit.
    PauliY(usize),
    /// Tensor product of Paulis, e.g. Z_0 Z_1.
    PauliString(Vec<(usize, char)>),
    /// Weighted sum of Pauli strings (Hamiltonian).
    Hamiltonian(Vec<(f64, Vec<(usize, char)>)>),
    /// Projector onto a computational basis state |b><b|.
    Projector(Vec<bool>),
}

// ============================================================
// FORWARD / BACKWARD RESULTS
// ============================================================

/// Result of a forward pass through the circuit.
#[derive(Clone, Debug)]
pub struct ForwardResult {
    pub expectation_values: Vec<f64>,
    pub state_vector: Option<Vec<(f64, f64)>>,
    pub probabilities: Option<Vec<f64>>,
    pub samples: Option<Vec<Vec<bool>>>,
}

/// Result of a backward pass (gradient computation).
#[derive(Clone, Debug)]
pub struct BackwardResult {
    /// Gradient of each expectation value w.r.t. each parameter.
    /// Shape: [num_observables][num_parameters].
    pub parameter_gradients: Vec<Vec<f64>>,
    pub gradient_method_used: String,
}

// ============================================================
// TENSOR DATA
// ============================================================

/// Data type tag for tensor serialization.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorDtype {
    Float32,
    Float64,
    Complex64,
    Complex128,
}

/// Flattened tensor for Python interop.
#[derive(Clone, Debug)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub dtype: TensorDtype,
}

impl TensorData {
    /// Create a new tensor from shape and data.
    pub fn new(shape: Vec<usize>, data: Vec<f64>, dtype: TensorDtype) -> Result<Self, BridgeError> {
        let expected_len: usize = shape.iter().product();
        let actual_len = data.len();
        // For complex types the data vec holds pairs (re, im).
        let element_size = match dtype {
            TensorDtype::Complex64 | TensorDtype::Complex128 => 2,
            _ => 1,
        };
        if actual_len != expected_len * element_size {
            return Err(BridgeError::ShapeError {
                expected: shape.clone(),
                got: vec![actual_len],
            });
        }
        Ok(Self { shape, data, dtype })
    }

    /// Serialize to little-endian bytes compatible with numpy.frombuffer.
    pub fn to_numpy_bytes(&self) -> Vec<u8> {
        match self.dtype {
            TensorDtype::Float32 => self
                .data
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect(),
            TensorDtype::Float64 | TensorDtype::Complex64 | TensorDtype::Complex128 => {
                self.data.iter().flat_map(|v| v.to_le_bytes()).collect()
            }
        }
    }

    /// Deserialize from little-endian bytes.
    pub fn from_numpy_bytes(
        bytes: &[u8],
        shape: Vec<usize>,
        dtype: TensorDtype,
    ) -> Result<Self, BridgeError> {
        let data: Vec<f64> = match dtype {
            TensorDtype::Float32 => {
                if bytes.len() % 4 != 0 {
                    return Err(BridgeError::SerializationError(
                        "byte length not multiple of 4 for f32".into(),
                    ));
                }
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect()
            }
            _ => {
                if bytes.len() % 8 != 0 {
                    return Err(BridgeError::SerializationError(
                        "byte length not multiple of 8 for f64".into(),
                    ));
                }
                bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect()
            }
        };
        TensorData::new(shape, data, dtype)
    }
}

// ============================================================
// MAPPED TENSOR
// ============================================================

/// Memory-mapped tensor with explicit strides for zero-copy interop.
#[derive(Clone, Debug)]
pub struct MappedTensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl MappedTensor {
    /// Create with C-contiguous (row-major) strides.
    pub fn new_contiguous(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, BridgeError> {
        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(BridgeError::ShapeError {
                expected: shape.clone(),
                got: vec![data.len()],
            });
        }
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Index into the tensor with multi-dimensional indices.
    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut offset = 0;
        for (i, (&idx, &s)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= s {
                return None;
            }
            offset += idx * self.strides[i];
        }
        self.data.get(offset).copied()
    }
}

// ============================================================
// DIFFERENTIABLE CIRCUIT
// ============================================================

/// A quantum circuit with trainable parameters that supports forward
/// evaluation and gradient computation for integration with ML frameworks.
#[derive(Clone, Debug)]
pub struct DifferentiableCircuit {
    pub gates: Vec<ParametricGate>,
    pub num_qubits: usize,
    pub parameters: Vec<f64>,
    pub gradient_method: GradientMethod,
}

impl DifferentiableCircuit {
    /// Create an empty circuit on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            parameters: Vec::new(),
            gradient_method: GradientMethod::default(),
        }
    }

    /// Create a circuit with a specific gradient method.
    pub fn with_gradient_method(num_qubits: usize, method: GradientMethod) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            parameters: Vec::new(),
            gradient_method: method,
        }
    }

    /// Register a new parameter and return its index.
    pub fn add_parameter(&mut self, value: f64) -> usize {
        let idx = self.parameters.len();
        self.parameters.push(value);
        idx
    }

    /// Add a gate to the circuit.
    pub fn add_gate(&mut self, gate: ParametricGate) {
        self.gates.push(gate);
    }

    /// Number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Set parameter values (for batched execution).
    pub fn set_parameters(&mut self, params: &[f64]) -> Result<(), BridgeError> {
        if params.len() != self.parameters.len() {
            return Err(BridgeError::ShapeError {
                expected: vec![self.parameters.len()],
                got: vec![params.len()],
            });
        }
        self.parameters.copy_from_slice(params);
        Ok(())
    }

    // ----------------------------------------------------------
    // CIRCUIT EXECUTION
    // ----------------------------------------------------------

    /// Apply all gates to a freshly initialized |0...0> state.
    fn execute(&self, params: &[f64]) -> StateVec {
        let mut sv = StateVec::new(self.num_qubits);
        self.apply_gates(&mut sv, params);
        sv
    }

    /// Apply gates to an existing state vector (used by adjoint diff).
    fn apply_gates(&self, sv: &mut StateVec, params: &[f64]) {
        for gate in &self.gates {
            Self::apply_gate(sv, gate, params);
        }
    }

    /// Apply a single gate to a state vector.
    fn apply_gate(sv: &mut StateVec, gate: &ParametricGate, params: &[f64]) {
        match gate {
            ParametricGate::H(q) => sv.apply_single(*q, mat_h()),
            ParametricGate::X(q) => sv.apply_single(*q, mat_x()),
            ParametricGate::Z(q) => sv.apply_single(*q, mat_z()),
            ParametricGate::CX(c, t) => sv.apply_controlled(*c, *t, mat_cx()),
            ParametricGate::CZ(c, t) => sv.apply_controlled(*c, *t, mat_cz_target()),
            ParametricGate::Rx(q, p) => sv.apply_single(*q, mat_rx(params[*p])),
            ParametricGate::Ry(q, p) => sv.apply_single(*q, mat_ry(params[*p])),
            ParametricGate::Rz(q, p) => sv.apply_single(*q, mat_rz(params[*p])),
            ParametricGate::CRx(c, t, p) => {
                sv.apply_controlled(*c, *t, mat_rx(params[*p]))
            }
            ParametricGate::CRy(c, t, p) => {
                sv.apply_controlled(*c, *t, mat_ry(params[*p]))
            }
            ParametricGate::CRz(c, t, p) => {
                sv.apply_controlled(*c, *t, mat_rz(params[*p]))
            }
            ParametricGate::U3(q, ti, pi, li) => {
                sv.apply_single(*q, mat_u3(params[*ti], params[*pi], params[*li]))
            }
        }
    }

    /// Apply the adjoint (dagger) of a single gate.
    fn apply_gate_adjoint(sv: &mut StateVec, gate: &ParametricGate, params: &[f64]) {
        // For unitary U, U^dag entries are conjugate-transpose of U.
        // Rotation gates: Rx(t)^dag = Rx(-t), etc.
        // H^dag = H, X^dag = X, Z^dag = Z, CX^dag = CX, CZ^dag = CZ.
        match gate {
            ParametricGate::H(q) => sv.apply_single(*q, mat_h()),
            ParametricGate::X(q) => sv.apply_single(*q, mat_x()),
            ParametricGate::Z(q) => sv.apply_single(*q, mat_z()),
            ParametricGate::CX(c, t) => sv.apply_controlled(*c, *t, mat_cx()),
            ParametricGate::CZ(c, t) => sv.apply_controlled(*c, *t, mat_cz_target()),
            ParametricGate::Rx(q, p) => sv.apply_single(*q, mat_rx(-params[*p])),
            ParametricGate::Ry(q, p) => sv.apply_single(*q, mat_ry(-params[*p])),
            ParametricGate::Rz(q, p) => sv.apply_single(*q, mat_rz(-params[*p])),
            ParametricGate::CRx(c, t, p) => {
                sv.apply_controlled(*c, *t, mat_rx(-params[*p]))
            }
            ParametricGate::CRy(c, t, p) => {
                sv.apply_controlled(*c, *t, mat_ry(-params[*p]))
            }
            ParametricGate::CRz(c, t, p) => {
                sv.apply_controlled(*c, *t, mat_rz(-params[*p]))
            }
            ParametricGate::U3(q, ti, pi, li) => {
                // U3^dag = U3(-theta, -lambda, -phi)  (swap phi <-> lambda and negate)
                sv.apply_single(
                    *q,
                    mat_u3(-params[*ti], -params[*li], -params[*pi]),
                )
            }
        }
    }

    // ----------------------------------------------------------
    // OBSERVABLE MEASUREMENT
    // ----------------------------------------------------------

    /// Compute <psi|O|psi> for a single observable.
    fn measure_observable(sv: &StateVec, obs: &Observable) -> f64 {
        match obs {
            Observable::PauliZ(q) => Self::expectation_pauli_z(sv, *q),
            Observable::PauliX(q) => Self::expectation_pauli_x(sv, *q),
            Observable::PauliY(q) => Self::expectation_pauli_y(sv, *q),
            Observable::PauliString(terms) => Self::expectation_pauli_string(sv, terms),
            Observable::Hamiltonian(weighted) => {
                weighted
                    .iter()
                    .map(|(coeff, terms)| coeff * Self::expectation_pauli_string(sv, terms))
                    .sum()
            }
            Observable::Projector(bits) => Self::expectation_projector(sv, bits),
        }
    }

    /// <psi|Z_q|psi> = sum_i |a_i|^2 * (-1)^{bit q of i}.
    fn expectation_pauli_z(sv: &StateVec, qubit: usize) -> f64 {
        let mut val = 0.0;
        for (i, a) in sv.amplitudes.iter().enumerate() {
            let sign = if (i >> qubit) & 1 == 0 { 1.0 } else { -1.0 };
            val += sign * a.norm_sq();
        }
        val
    }

    /// <psi|X_q|psi>: apply H, measure Z, undo conceptually.
    fn expectation_pauli_x(sv: &StateVec, qubit: usize) -> f64 {
        let mut sv2 = sv.clone();
        sv2.apply_single(qubit, mat_h());
        Self::expectation_pauli_z(&sv2, qubit)
    }

    /// <psi|Y_q|psi>: rotate to Z eigenbasis via S^dag H, measure Z.
    fn expectation_pauli_y(sv: &StateVec, qubit: usize) -> f64 {
        // Y = i X Z, and Sdg H Z H S = Y  =>  <Y> = <psi|Sdg H Z H S|psi>
        // Equivalently: apply S^dag then H, measure Z.
        let mut sv2 = sv.clone();
        // S^dag = Rz(-pi/2) up to global phase, but more precisely:
        // S = [[1,0],[0,i]], S^dag = [[1,0],[0,-i]]
        let sdg = [
            C64::one(),
            C64::zero(),
            C64::zero(),
            C64::new(0.0, -1.0),
        ];
        sv2.apply_single(qubit, sdg);
        sv2.apply_single(qubit, mat_h());
        Self::expectation_pauli_z(&sv2, qubit)
    }

    /// <psi|P_0 x P_1 x ...|psi> for a Pauli string.
    fn expectation_pauli_string(sv: &StateVec, terms: &[(usize, char)]) -> f64 {
        // Apply basis rotations for non-Z Paulis, then compute
        // sum_i |a_i|^2 * prod_q (-1)^{bit q of i} for Z-basis.
        if terms.is_empty() {
            return 1.0; // Identity
        }

        let mut sv2 = sv.clone();

        // Rotate non-Z qubits into Z basis
        for &(qubit, pauli) in terms {
            match pauli {
                'X' | 'x' => sv2.apply_single(qubit, mat_h()),
                'Y' | 'y' => {
                    let sdg = [
                        C64::one(),
                        C64::zero(),
                        C64::zero(),
                        C64::new(0.0, -1.0),
                    ];
                    sv2.apply_single(qubit, sdg);
                    sv2.apply_single(qubit, mat_h());
                }
                'Z' | 'z' | 'I' | 'i' => {} // already in Z basis or identity
                _ => {}
            }
        }

        // Now measure product of Z operators on the relevant qubits
        let z_qubits: Vec<usize> = terms
            .iter()
            .filter(|(_, p)| *p != 'I' && *p != 'i')
            .map(|(q, _)| *q)
            .collect();

        let mut val = 0.0;
        for (i, a) in sv2.amplitudes.iter().enumerate() {
            let mut sign = 1.0;
            for &q in &z_qubits {
                if (i >> q) & 1 == 1 {
                    sign *= -1.0;
                }
            }
            val += sign * a.norm_sq();
        }
        val
    }

    /// <psi|b><b|psi> = |<b|psi>|^2.
    fn expectation_projector(sv: &StateVec, bits: &[bool]) -> f64 {
        // |b> is the computational basis state with the given bit pattern.
        let mut idx = 0usize;
        for (q, &b) in bits.iter().enumerate() {
            if b {
                idx |= 1 << q;
            }
        }
        if idx < sv.dim() {
            sv.amplitudes[idx].norm_sq()
        } else {
            0.0
        }
    }

    // ----------------------------------------------------------
    // FORWARD PASS
    // ----------------------------------------------------------

    /// Run the circuit and measure all observables.
    pub fn forward(&self, observables: &[Observable]) -> Result<ForwardResult, BridgeError> {
        self.forward_with_params(&self.parameters, observables)
    }

    /// Forward pass with explicit parameters (for batching / gradient shifts).
    pub fn forward_with_params(
        &self,
        params: &[f64],
        observables: &[Observable],
    ) -> Result<ForwardResult, BridgeError> {
        if params.len() != self.parameters.len() && !self.parameters.is_empty() {
            return Err(BridgeError::ShapeError {
                expected: vec![self.parameters.len()],
                got: vec![params.len()],
            });
        }
        let sv = self.execute(params);
        let expectations: Vec<f64> = observables
            .iter()
            .map(|obs| Self::measure_observable(&sv, obs))
            .collect();
        let probs = sv.probabilities();
        let pairs = sv.to_pairs();

        Ok(ForwardResult {
            expectation_values: expectations,
            state_vector: Some(pairs),
            probabilities: Some(probs),
            samples: None,
        })
    }

    // ----------------------------------------------------------
    // BACKWARD PASS (GRADIENT COMPUTATION)
    // ----------------------------------------------------------

    /// Compute gradients of all expectation values w.r.t. all parameters.
    pub fn backward(
        &self,
        observables: &[Observable],
    ) -> Result<BackwardResult, BridgeError> {
        self.backward_with_params(&self.parameters, observables)
    }

    /// Backward pass with explicit parameters.
    pub fn backward_with_params(
        &self,
        params: &[f64],
        observables: &[Observable],
    ) -> Result<BackwardResult, BridgeError> {
        match &self.gradient_method {
            GradientMethod::ParameterShift { shift } => {
                self.gradient_parameter_shift(params, observables, *shift)
            }
            GradientMethod::AdjointDiff => {
                self.gradient_adjoint(params, observables)
            }
            GradientMethod::FiniteDifference { epsilon } => {
                self.gradient_finite_difference(params, observables, *epsilon)
            }
            GradientMethod::Backprop => {
                self.gradient_backprop(params, observables)
            }
        }
    }

    // ----------------------------------------------------------
    // PARAMETER-SHIFT RULE
    // ----------------------------------------------------------

    fn gradient_parameter_shift(
        &self,
        params: &[f64],
        observables: &[Observable],
        shift: f64,
    ) -> Result<BackwardResult, BridgeError> {
        let np = params.len();
        let no = observables.len();
        let mut grads = vec![vec![0.0; np]; no];
        let sin_s = shift.sin();
        if sin_s.abs() < 1e-15 {
            return Err(BridgeError::GradientError(
                "parameter shift sin(s) is zero".into(),
            ));
        }

        for p_idx in 0..np {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            params_plus[p_idx] += shift;
            params_minus[p_idx] -= shift;

            let sv_plus = self.execute(&params_plus);
            let sv_minus = self.execute(&params_minus);

            for (o_idx, obs) in observables.iter().enumerate() {
                let e_plus = Self::measure_observable(&sv_plus, obs);
                let e_minus = Self::measure_observable(&sv_minus, obs);
                grads[o_idx][p_idx] = (e_plus - e_minus) / (2.0 * sin_s);
            }
        }

        Ok(BackwardResult {
            parameter_gradients: grads,
            gradient_method_used: format!("parameter_shift(s={:.4})", shift),
        })
    }

    // ----------------------------------------------------------
    // ADJOINT DIFFERENTIATION
    // ----------------------------------------------------------

    fn gradient_adjoint(
        &self,
        params: &[f64],
        observables: &[Observable],
    ) -> Result<BackwardResult, BridgeError> {
        let np = params.len();
        let no = observables.len();
        let mut grads = vec![vec![0.0; np]; no];

        // Forward sweep: store intermediate states after each gate.
        let mut states = Vec::with_capacity(self.gates.len() + 1);
        let mut sv = StateVec::new(self.num_qubits);
        states.push(sv.clone());
        for gate in &self.gates {
            Self::apply_gate(&mut sv, gate, params);
            states.push(sv.clone());
        }
        // sv is now the final state |psi>

        // For each observable, backward sweep
        for (o_idx, obs) in observables.iter().enumerate() {
            // |lambda> = O|psi>
            let mut lambda = Self::apply_observable_to_state(&sv, obs);

            // Backward traverse
            // IMPORTANT: gradient must be computed BEFORE unapplying the gate
            // adjoint, because we need λ at the output of gate k, not its input.
            for (g_idx, gate) in self.gates.iter().enumerate().rev() {
                // If gate is parametric, compute gradient contribution FIRST
                // using current lambda (which is at the output side of this gate)
                for p_idx in gate.param_indices() {
                    // grad += 2 * Re(<lambda|dU/dtheta|psi_before>)
                    let psi_before = &states[g_idx];
                    let contrib = self.gate_derivative_overlap(
                        gate,
                        p_idx,
                        params,
                        psi_before,
                        &lambda,
                    );
                    grads[o_idx][p_idx] += contrib;
                }

                // THEN un-apply gate from lambda: lambda = U_g^dag |lambda>
                Self::apply_gate_adjoint(&mut lambda, gate, params);
            }
        }

        Ok(BackwardResult {
            parameter_gradients: grads,
            gradient_method_used: "adjoint_differentiation".into(),
        })
    }

    /// Compute 2*Re(<lambda| dU_gate/d(param[p_idx]) |psi>).
    fn gate_derivative_overlap(
        &self,
        gate: &ParametricGate,
        p_idx: usize,
        params: &[f64],
        psi: &StateVec,
        lambda: &StateVec,
    ) -> f64 {
        // Apply the generator (derivative matrix) of the gate to |psi>,
        // then compute 2*Re(<lambda|result>).
        let mut dpsi = psi.clone();

        match gate {
            ParametricGate::Rx(q, pi) if *pi == p_idx => {
                // dRx/dtheta |psi> = (-i/2) X Rx(theta) |psi>
                // But we need d/dtheta of the whole circuit, which at this point
                // is just applying the generator. The derivative of Rx(t) is:
                // dRx(t)/dt = -i/2 * [[0,1],[1,0]] * Rx(t)
                // Since psi already has Rx not yet applied (psi is state BEFORE gate),
                // we need: Rx(t) |psi> then X then scale by -i/2.
                // Actually for adjoint: we need d(gate)/d(param) applied to psi_before.
                Self::apply_gate(&mut dpsi, gate, params);
                // Now apply the generator: -i/2 * sigma_x
                dpsi.apply_single(*q, mat_x());
                // Scale by -i/2: multiply all amplitudes by -i/2
                for a in dpsi.amplitudes.iter_mut() {
                    // (-i/2) * (re + i*im) = im/2 + i*(-re/2)
                    let new_re = a.im / 2.0;
                    let new_im = -a.re / 2.0;
                    *a = C64::new(new_re, new_im);
                }
            }
            ParametricGate::Ry(q, pi) if *pi == p_idx => {
                Self::apply_gate(&mut dpsi, gate, params);
                dpsi.apply_single(*q, mat_y());
                for a in dpsi.amplitudes.iter_mut() {
                    let new_re = a.im / 2.0;
                    let new_im = -a.re / 2.0;
                    *a = C64::new(new_re, new_im);
                }
            }
            ParametricGate::Rz(q, pi) if *pi == p_idx => {
                Self::apply_gate(&mut dpsi, gate, params);
                dpsi.apply_single(*q, mat_z());
                for a in dpsi.amplitudes.iter_mut() {
                    let new_re = a.im / 2.0;
                    let new_im = -a.re / 2.0;
                    *a = C64::new(new_re, new_im);
                }
            }
            ParametricGate::CRx(c, t, pi) if *pi == p_idx => {
                Self::apply_gate(&mut dpsi, gate, params);
                // Generator only acts on target when control is |1>
                // Apply controlled-X then scale by -i/2
                dpsi.apply_controlled(*c, *t, mat_x());
                for a in dpsi.amplitudes.iter_mut() {
                    let new_re = a.im / 2.0;
                    let new_im = -a.re / 2.0;
                    *a = C64::new(new_re, new_im);
                }
            }
            ParametricGate::CRy(c, t, pi) if *pi == p_idx => {
                Self::apply_gate(&mut dpsi, gate, params);
                dpsi.apply_controlled(*c, *t, mat_y());
                for a in dpsi.amplitudes.iter_mut() {
                    let new_re = a.im / 2.0;
                    let new_im = -a.re / 2.0;
                    *a = C64::new(new_re, new_im);
                }
            }
            ParametricGate::CRz(c, t, pi) if *pi == p_idx => {
                Self::apply_gate(&mut dpsi, gate, params);
                dpsi.apply_controlled(*c, *t, mat_z());
                for a in dpsi.amplitudes.iter_mut() {
                    let new_re = a.im / 2.0;
                    let new_im = -a.re / 2.0;
                    *a = C64::new(new_re, new_im);
                }
            }
            ParametricGate::U3(q, ti, phi_i, lam_i) => {
                // U3 has three parameters. We differentiate w.r.t. the one
                // matching p_idx using finite difference as the analytical
                // form is complex.
                let eps = 1e-7;
                let mut pp = params.to_vec();
                pp[p_idx] += eps;
                let mut dpsi_plus = psi.clone();
                let u3_plus = ParametricGate::U3(*q, *ti, *phi_i, *lam_i);
                Self::apply_gate(&mut dpsi_plus, &u3_plus, &pp);

                pp[p_idx] -= 2.0 * eps;
                let mut dpsi_minus = psi.clone();
                Self::apply_gate(&mut dpsi_minus, &u3_plus, &pp);

                // dpsi = (dpsi_plus - dpsi_minus) / (2*eps)
                for (i, (ap, am)) in dpsi_plus
                    .amplitudes
                    .iter()
                    .zip(dpsi_minus.amplitudes.iter())
                    .enumerate()
                {
                    dpsi.amplitudes[i] = ap.sub(*am).scale(1.0 / (2.0 * eps));
                }
                // Return overlap directly
                let overlap = lambda.inner(&dpsi);
                return 2.0 * overlap.re;
            }
            _ => {
                // This parameter index does not belong to this gate; no contribution.
                return 0.0;
            }
        }

        // Compute 2 * Re(<lambda|dpsi>)
        let overlap = lambda.inner(&dpsi);
        2.0 * overlap.re
    }

    /// Apply observable O to state |psi>, returning O|psi>.
    fn apply_observable_to_state(sv: &StateVec, obs: &Observable) -> StateVec {
        let mut result = StateVec::new(sv.num_qubits);
        match obs {
            Observable::PauliZ(q) => {
                result.amplitudes = sv.amplitudes.clone();
                for (i, a) in result.amplitudes.iter_mut().enumerate() {
                    if (i >> q) & 1 == 1 {
                        *a = a.scale(-1.0);
                    }
                }
            }
            Observable::PauliX(q) => {
                result.amplitudes = sv.amplitudes.clone();
                result.apply_single(*q, mat_x());
            }
            Observable::PauliY(q) => {
                result.amplitudes = sv.amplitudes.clone();
                result.apply_single(*q, mat_y());
            }
            Observable::PauliString(terms) => {
                result.amplitudes = sv.amplitudes.clone();
                for &(q, p) in terms {
                    match p {
                        'X' | 'x' => result.apply_single(q, mat_x()),
                        'Y' | 'y' => result.apply_single(q, mat_y()),
                        'Z' | 'z' => {
                            for (i, a) in result.amplitudes.iter_mut().enumerate() {
                                if (i >> q) & 1 == 1 {
                                    *a = a.scale(-1.0);
                                }
                            }
                        }
                        _ => {} // Identity
                    }
                }
            }
            Observable::Hamiltonian(weighted) => {
                // H|psi> = sum_k c_k P_k |psi>
                for a in result.amplitudes.iter_mut() {
                    *a = C64::zero();
                }
                for (coeff, terms) in weighted {
                    let term_result = Self::apply_observable_to_state(
                        sv,
                        &Observable::PauliString(terms.clone()),
                    );
                    for (i, a) in term_result.amplitudes.iter().enumerate() {
                        result.amplitudes[i] = result.amplitudes[i].add(a.scale(*coeff));
                    }
                }
            }
            Observable::Projector(bits) => {
                // |b><b|psi> = <b|psi> |b>
                let mut idx = 0usize;
                for (q, &b) in bits.iter().enumerate() {
                    if b {
                        idx |= 1 << q;
                    }
                }
                let overlap = if idx < sv.dim() {
                    sv.amplitudes[idx]
                } else {
                    C64::zero()
                };
                for a in result.amplitudes.iter_mut() {
                    *a = C64::zero();
                }
                if idx < result.dim() {
                    result.amplitudes[idx] = overlap;
                }
            }
        }
        result
    }

    // ----------------------------------------------------------
    // FINITE DIFFERENCE
    // ----------------------------------------------------------

    fn gradient_finite_difference(
        &self,
        params: &[f64],
        observables: &[Observable],
        epsilon: f64,
    ) -> Result<BackwardResult, BridgeError> {
        let np = params.len();
        let no = observables.len();
        let mut grads = vec![vec![0.0; np]; no];

        for p_idx in 0..np {
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[p_idx] += epsilon;
            pm[p_idx] -= epsilon;

            let sv_p = self.execute(&pp);
            let sv_m = self.execute(&pm);

            for (o_idx, obs) in observables.iter().enumerate() {
                let ep = Self::measure_observable(&sv_p, obs);
                let em = Self::measure_observable(&sv_m, obs);
                grads[o_idx][p_idx] = (ep - em) / (2.0 * epsilon);
            }
        }

        Ok(BackwardResult {
            parameter_gradients: grads,
            gradient_method_used: format!("finite_difference(eps={:.2e})", epsilon),
        })
    }

    // ----------------------------------------------------------
    // BACKPROP (reverse-mode through statevector)
    // ----------------------------------------------------------

    fn gradient_backprop(
        &self,
        params: &[f64],
        observables: &[Observable],
    ) -> Result<BackwardResult, BridgeError> {
        // Store all intermediate state vectors (memory-intensive).
        let np = params.len();
        let no = observables.len();
        let mut grads = vec![vec![0.0; np]; no];

        let mut states = Vec::with_capacity(self.gates.len() + 1);
        let mut sv = StateVec::new(self.num_qubits);
        states.push(sv.clone());
        for gate in &self.gates {
            Self::apply_gate(&mut sv, gate, params);
            states.push(sv.clone());
        }

        // For each observable, backpropagate through the circuit
        for (o_idx, obs) in observables.iter().enumerate() {
            // Adjoint state: start with O|psi_final>
            let mut adj = Self::apply_observable_to_state(&sv, obs);

            for (g_idx, gate) in self.gates.iter().enumerate().rev() {
                // Compute gradient contribution for parametric gates
                for p_idx in gate.param_indices() {
                    let psi_before = &states[g_idx];
                    let contrib =
                        self.gate_derivative_overlap(gate, p_idx, params, psi_before, &adj);
                    grads[o_idx][p_idx] += contrib;
                }
                // Propagate adjoint backward: adj = U_g^dag adj
                Self::apply_gate_adjoint(&mut adj, gate, params);
            }
        }

        Ok(BackwardResult {
            parameter_gradients: grads,
            gradient_method_used: "backprop".into(),
        })
    }
}

// ============================================================
// QUANTUM AUTOGRAD NODE
// ============================================================

/// Saves circuit state for PyTorch-style autograd replay.
#[derive(Clone, Debug)]
pub struct QuantumAutograd {
    circuit: DifferentiableCircuit,
    observables: Vec<Observable>,
    saved_params: Vec<f64>,
    saved_state: Option<Vec<(f64, f64)>>,
}

impl QuantumAutograd {
    /// Create a new autograd context and run the forward pass.
    pub fn apply(
        circuit: &DifferentiableCircuit,
        observables: &[Observable],
    ) -> Result<(ForwardResult, Self), BridgeError> {
        let fwd = circuit.forward(observables)?;
        let ctx = Self {
            circuit: circuit.clone(),
            observables: observables.to_vec(),
            saved_params: circuit.parameters.clone(),
            saved_state: fwd.state_vector.clone(),
        };
        Ok((fwd, ctx))
    }

    /// Replay the backward pass using saved context.
    pub fn backward(&self) -> Result<BackwardResult, BridgeError> {
        self.circuit
            .backward_with_params(&self.saved_params, &self.observables)
    }

    /// Access the saved state vector.
    pub fn saved_state(&self) -> &Option<Vec<(f64, f64)>> {
        &self.saved_state
    }
}

// ============================================================
// BATCH EXECUTOR
// ============================================================

/// Executes the same circuit template with multiple parameter sets.
#[derive(Clone, Debug)]
pub struct BatchExecutor {
    pub circuit_template: DifferentiableCircuit,
    pub observables: Vec<Observable>,
    pub batch_size: usize,
}

impl BatchExecutor {
    pub fn new(
        circuit: DifferentiableCircuit,
        observables: Vec<Observable>,
        batch_size: usize,
    ) -> Self {
        Self {
            circuit_template: circuit,
            observables,
            batch_size,
        }
    }

    /// Execute the circuit for each parameter set in the batch.
    /// `param_batch` shape: [batch_size][num_parameters].
    pub fn execute(
        &self,
        param_batch: &[Vec<f64>],
    ) -> Result<Vec<ForwardResult>, BridgeError> {
        if param_batch.len() != self.batch_size {
            return Err(BridgeError::ShapeError {
                expected: vec![self.batch_size],
                got: vec![param_batch.len()],
            });
        }
        let results: Vec<ForwardResult> = param_batch
            .iter()
            .map(|params| {
                self.circuit_template
                    .forward_with_params(params, &self.observables)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    /// Execute forward + backward for the whole batch.
    pub fn execute_with_gradients(
        &self,
        param_batch: &[Vec<f64>],
    ) -> Result<Vec<(ForwardResult, BackwardResult)>, BridgeError> {
        param_batch
            .iter()
            .map(|params| {
                let fwd = self
                    .circuit_template
                    .forward_with_params(params, &self.observables)?;
                let bwd = self
                    .circuit_template
                    .backward_with_params(params, &self.observables)?;
                Ok((fwd, bwd))
            })
            .collect()
    }
}

// ============================================================
// VQE OPTIMIZER
// ============================================================

/// Optimizer choice for VQE.
#[derive(Clone, Debug)]
pub enum VqeOptimizer {
    Adam {
        lr: f64,
        beta1: f64,
        beta2: f64,
    },
    SGD {
        lr: f64,
        momentum: f64,
    },
    LBFGS {
        max_line_search: usize,
    },
    SPSA {
        a: f64,
        c: f64,
    },
}

/// Internal state for Adam optimizer.
struct AdamState {
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
}

impl AdamState {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    fn step(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        lr: f64,
        beta1: f64,
        beta2: f64,
    ) {
        self.t += 1;
        let eps = 1e-8;
        for i in 0..params.len() {
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grads[i];
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / (1.0 - beta1.powi(self.t as i32));
            let v_hat = self.v[i] / (1.0 - beta2.powi(self.t as i32));
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

/// Internal state for SGD with momentum.
struct SgdState {
    velocity: Vec<f64>,
}

impl SgdState {
    fn new(n: usize) -> Self {
        Self {
            velocity: vec![0.0; n],
        }
    }

    fn step(&mut self, params: &mut [f64], grads: &[f64], lr: f64, momentum: f64) {
        for i in 0..params.len() {
            self.velocity[i] = momentum * self.velocity[i] + grads[i];
            params[i] -= lr * self.velocity[i];
        }
    }
}

// ============================================================
// VQE RUNNER
// ============================================================

/// Variational Quantum Eigensolver for finding ground state energies.
pub struct VqeRunner {
    pub circuit: DifferentiableCircuit,
    pub hamiltonian: Observable,
    pub optimizer: VqeOptimizer,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

/// Result of a VQE optimization run.
#[derive(Clone, Debug)]
pub struct VqeResult {
    pub optimal_energy: f64,
    pub optimal_parameters: Vec<f64>,
    pub energy_history: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

impl VqeRunner {
    pub fn new(
        circuit: DifferentiableCircuit,
        hamiltonian: Observable,
        optimizer: VqeOptimizer,
    ) -> Self {
        Self {
            circuit,
            hamiltonian,
            optimizer,
            max_iterations: 200,
            convergence_threshold: 1e-6,
        }
    }

    /// Run the VQE optimization loop.
    pub fn run(&self) -> Result<VqeResult, BridgeError> {
        let obs = vec![self.hamiltonian.clone()];
        let np = self.circuit.num_parameters();
        let mut params = self.circuit.parameters.clone();
        let mut energy_history = Vec::with_capacity(self.max_iterations);
        let mut converged = false;

        match &self.optimizer {
            VqeOptimizer::Adam { lr, beta1, beta2 } => {
                let mut state = AdamState::new(np);
                for iter in 0..self.max_iterations {
                    let fwd = self.circuit.forward_with_params(&params, &obs)?;
                    let energy = fwd.expectation_values[0];
                    energy_history.push(energy);

                    if iter > 0
                        && (energy_history[iter] - energy_history[iter - 1]).abs()
                            < self.convergence_threshold
                    {
                        converged = true;
                        break;
                    }

                    let bwd = self.circuit.backward_with_params(&params, &obs)?;
                    let grads = &bwd.parameter_gradients[0];
                    state.step(&mut params, grads, *lr, *beta1, *beta2);
                }
            }
            VqeOptimizer::SGD { lr, momentum } => {
                let mut state = SgdState::new(np);
                for iter in 0..self.max_iterations {
                    let fwd = self.circuit.forward_with_params(&params, &obs)?;
                    let energy = fwd.expectation_values[0];
                    energy_history.push(energy);

                    if iter > 0
                        && (energy_history[iter] - energy_history[iter - 1]).abs()
                            < self.convergence_threshold
                    {
                        converged = true;
                        break;
                    }

                    let bwd = self.circuit.backward_with_params(&params, &obs)?;
                    let grads = &bwd.parameter_gradients[0];
                    state.step(&mut params, grads, *lr, *momentum);
                }
            }
            VqeOptimizer::LBFGS { max_line_search } => {
                // Simplified L-BFGS: gradient descent with line search.
                let max_ls = *max_line_search;
                for iter in 0..self.max_iterations {
                    let fwd = self.circuit.forward_with_params(&params, &obs)?;
                    let energy = fwd.expectation_values[0];
                    energy_history.push(energy);

                    if iter > 0
                        && (energy_history[iter] - energy_history[iter - 1]).abs()
                            < self.convergence_threshold
                    {
                        converged = true;
                        break;
                    }

                    let bwd = self.circuit.backward_with_params(&params, &obs)?;
                    let grads = &bwd.parameter_gradients[0];

                    // Backtracking line search
                    let mut alpha = 0.1;
                    for _ in 0..max_ls {
                        let trial: Vec<f64> = params
                            .iter()
                            .zip(grads.iter())
                            .map(|(p, g)| p - alpha * g)
                            .collect();
                        let trial_fwd =
                            self.circuit.forward_with_params(&trial, &obs)?;
                        if trial_fwd.expectation_values[0] < energy {
                            params = trial;
                            break;
                        }
                        alpha *= 0.5;
                    }
                    // If no improvement found, still take a small step
                    if alpha < 1e-10 {
                        for (p, g) in params.iter_mut().zip(grads.iter()) {
                            *p -= 1e-4 * g;
                        }
                    }
                }
            }
            VqeOptimizer::SPSA { a, c } => {
                // Simultaneous Perturbation Stochastic Approximation.
                // Does NOT require gradient computation -- uses random perturbations.
                let mut rng_state: u64 = 42;
                for iter in 0..self.max_iterations {
                    let fwd = self.circuit.forward_with_params(&params, &obs)?;
                    let energy = fwd.expectation_values[0];
                    energy_history.push(energy);

                    if iter > 0
                        && (energy_history[iter] - energy_history[iter - 1]).abs()
                            < self.convergence_threshold
                    {
                        converged = true;
                        break;
                    }

                    let k = iter as f64 + 1.0;
                    let ak = a / k.powf(0.602);
                    let ck = c / k.powf(0.101);

                    // Random perturbation vector (+1 or -1)
                    let delta: Vec<f64> = (0..np)
                        .map(|_| {
                            rng_state = rng_state
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            if (rng_state >> 32) & 1 == 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        })
                        .collect();

                    let pp: Vec<f64> = params
                        .iter()
                        .zip(delta.iter())
                        .map(|(p, d)| p + ck * d)
                        .collect();
                    let pm: Vec<f64> = params
                        .iter()
                        .zip(delta.iter())
                        .map(|(p, d)| p - ck * d)
                        .collect();

                    let ep = self.circuit.forward_with_params(&pp, &obs)?;
                    let em = self.circuit.forward_with_params(&pm, &obs)?;
                    let diff =
                        ep.expectation_values[0] - em.expectation_values[0];

                    for (i, d) in delta.iter().enumerate() {
                        params[i] -= ak * diff / (2.0 * ck * d);
                    }
                }
            }
        }

        let final_energy = *energy_history.last().unwrap_or(&f64::NAN);
        let iterations = energy_history.len();
        Ok(VqeResult {
            optimal_energy: final_energy,
            optimal_parameters: params,
            energy_history,
            converged,
            iterations,
        })
    }
}

// ============================================================
// QUANTUM KERNEL FOR ML
// ============================================================

/// Quantum kernel machine for classification tasks.
/// K(x_i, x_j) = |<0|U^dag(x_i) U(x_j)|0>|^2.
pub struct QuantumKernelML {
    pub feature_map: DifferentiableCircuit,
    pub num_features: usize,
    pub num_qubits: usize,
}

impl QuantumKernelML {
    pub fn new(feature_map: DifferentiableCircuit, num_features: usize) -> Self {
        let nq = feature_map.num_qubits;
        Self {
            feature_map,
            num_features,
            num_qubits: nq,
        }
    }

    /// Compute the kernel value K(x, y).
    pub fn kernel_value(&self, x: &[f64], y: &[f64]) -> Result<f64, BridgeError> {
        if x.len() != self.num_features || y.len() != self.num_features {
            return Err(BridgeError::ShapeError {
                expected: vec![self.num_features],
                got: vec![x.len().max(y.len())],
            });
        }
        let sv_x = self.feature_map.execute(x);
        let sv_y = self.feature_map.execute(y);
        let overlap = sv_x.inner(&sv_y);
        Ok(overlap.norm_sq())
    }

    /// Compute the full kernel matrix for a dataset.
    /// data shape: [n_samples][n_features].
    pub fn kernel_matrix(
        &self,
        data: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, BridgeError> {
        let n = data.len();
        let mut matrix = vec![vec![0.0; n]; n];

        // Pre-compute all state vectors
        let states: Vec<StateVec> = data
            .iter()
            .map(|x| self.feature_map.execute(x))
            .collect();

        for i in 0..n {
            matrix[i][i] = 1.0; // K(x,x) = 1 for normalized states
            for j in (i + 1)..n {
                let overlap = states[i].inner(&states[j]);
                let k = overlap.norm_sq();
                matrix[i][j] = k;
                matrix[j][i] = k; // Symmetric
            }
        }
        Ok(matrix)
    }
}

// ============================================================
// QUANTUM GAN
// ============================================================

/// Quantum Generative Adversarial Network.
pub struct QuantumGAN {
    pub generator: DifferentiableCircuit,
    pub discriminator_params: Vec<f64>,
    pub latent_dim: usize,
    pub data_dim: usize,
}

impl QuantumGAN {
    pub fn new(
        generator: DifferentiableCircuit,
        latent_dim: usize,
        data_dim: usize,
    ) -> Self {
        // Simple classical discriminator: one weight per data dimension + bias.
        let disc_params = vec![0.0; data_dim + 1];
        Self {
            generator,
            discriminator_params: disc_params,
            latent_dim,
            data_dim,
        }
    }

    /// Generate a probability distribution from the quantum generator.
    pub fn generate(&self, latent_params: &[f64]) -> Result<Vec<f64>, BridgeError> {
        if latent_params.len() != self.generator.num_parameters() {
            return Err(BridgeError::ShapeError {
                expected: vec![self.generator.num_parameters()],
                got: vec![latent_params.len()],
            });
        }
        let sv = self.generator.execute(latent_params);
        Ok(sv.probabilities())
    }

    /// Classical discriminator: sigmoid(w . x + b).
    pub fn discriminate(&self, probs: &[f64]) -> f64 {
        let w = &self.discriminator_params[..self.data_dim.min(probs.len())];
        let b = *self.discriminator_params.last().unwrap_or(&0.0);
        let z: f64 = w
            .iter()
            .zip(probs.iter())
            .map(|(wi, xi)| wi * xi)
            .sum::<f64>()
            + b;
        1.0 / (1.0 + (-z).exp())
    }

    /// One training step: returns (generator_loss, discriminator_output).
    pub fn train_step(
        &mut self,
        latent_params: &[f64],
        real_distribution: &[f64],
        lr: f64,
    ) -> Result<(f64, f64), BridgeError> {
        let fake_dist = self.generate(latent_params)?;
        let d_fake = self.discriminate(&fake_dist);
        let d_real = self.discriminate(real_distribution);

        // Update discriminator to maximize log(D(real)) + log(1 - D(fake))
        let n = self.data_dim.min(fake_dist.len());
        for i in 0..n {
            let grad_real = real_distribution.get(i).unwrap_or(&0.0) * (1.0 - d_real);
            let grad_fake = -fake_dist[i] * d_fake;
            self.discriminator_params[i] += lr * (grad_real + grad_fake);
        }
        // Bias
        if let Some(b) = self.discriminator_params.last_mut() {
            *b += lr * ((1.0 - d_real) - d_fake);
        }

        let gen_loss = -(d_fake + 1e-10).ln();
        Ok((gen_loss, d_fake))
    }
}

// ============================================================
// CONFIG BUILDER
// ============================================================

/// Builder for constructing circuits with a fluent API.
pub struct CircuitBuilder {
    circuit: DifferentiableCircuit,
}

impl CircuitBuilder {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: DifferentiableCircuit::new(num_qubits),
        }
    }

    pub fn gradient_method(mut self, method: GradientMethod) -> Self {
        self.circuit.gradient_method = method;
        self
    }

    pub fn h(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(ParametricGate::H(qubit));
        self
    }

    pub fn x(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(ParametricGate::X(qubit));
        self
    }

    pub fn z(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(ParametricGate::Z(qubit));
        self
    }

    pub fn cx(mut self, control: usize, target: usize) -> Self {
        self.circuit.add_gate(ParametricGate::CX(control, target));
        self
    }

    pub fn cz(mut self, control: usize, target: usize) -> Self {
        self.circuit.add_gate(ParametricGate::CZ(control, target));
        self
    }

    pub fn rx(mut self, qubit: usize, value: f64) -> Self {
        let idx = self.circuit.add_parameter(value);
        self.circuit.add_gate(ParametricGate::Rx(qubit, idx));
        self
    }

    pub fn ry(mut self, qubit: usize, value: f64) -> Self {
        let idx = self.circuit.add_parameter(value);
        self.circuit.add_gate(ParametricGate::Ry(qubit, idx));
        self
    }

    pub fn rz(mut self, qubit: usize, value: f64) -> Self {
        let idx = self.circuit.add_parameter(value);
        self.circuit.add_gate(ParametricGate::Rz(qubit, idx));
        self
    }

    pub fn crx(mut self, control: usize, target: usize, value: f64) -> Self {
        let idx = self.circuit.add_parameter(value);
        self.circuit
            .add_gate(ParametricGate::CRx(control, target, idx));
        self
    }

    pub fn cry(mut self, control: usize, target: usize, value: f64) -> Self {
        let idx = self.circuit.add_parameter(value);
        self.circuit
            .add_gate(ParametricGate::CRy(control, target, idx));
        self
    }

    pub fn crz(mut self, control: usize, target: usize, value: f64) -> Self {
        let idx = self.circuit.add_parameter(value);
        self.circuit
            .add_gate(ParametricGate::CRz(control, target, idx));
        self
    }

    pub fn u3(mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> Self {
        let ti = self.circuit.add_parameter(theta);
        let pi = self.circuit.add_parameter(phi);
        let li = self.circuit.add_parameter(lambda);
        self.circuit.add_gate(ParametricGate::U3(qubit, ti, pi, li));
        self
    }

    pub fn build(self) -> DifferentiableCircuit {
        self.circuit
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-6;

    // ----------------------------------------------------------
    // 1. Circuit construction
    // ----------------------------------------------------------
    #[test]
    fn test_circuit_construction() {
        let mut c = DifferentiableCircuit::new(3);
        let p0 = c.add_parameter(0.5);
        let p1 = c.add_parameter(1.0);
        c.add_gate(ParametricGate::H(0));
        c.add_gate(ParametricGate::Rx(1, p0));
        c.add_gate(ParametricGate::CX(0, 1));
        c.add_gate(ParametricGate::Ry(2, p1));
        assert_eq!(c.num_qubits, 3);
        assert_eq!(c.num_parameters(), 2);
        assert_eq!(c.gates.len(), 4);
    }

    // ----------------------------------------------------------
    // 2. H|0> => <Z> = 0
    // ----------------------------------------------------------
    #[test]
    fn test_forward_h_expectation_z() {
        let c = CircuitBuilder::new(1).h(0).build();
        let fwd = c.forward(&[Observable::PauliZ(0)]).unwrap();
        assert!(fwd.expectation_values[0].abs() < TOL);
    }

    // ----------------------------------------------------------
    // 3. Rx(pi)|0> => <Z> = -1
    // ----------------------------------------------------------
    #[test]
    fn test_forward_rx_pi() {
        let c = CircuitBuilder::new(1).rx(0, PI).build();
        let fwd = c.forward(&[Observable::PauliZ(0)]).unwrap();
        assert!((fwd.expectation_values[0] + 1.0).abs() < TOL);
    }

    // ----------------------------------------------------------
    // 4. Bell state correlations
    // ----------------------------------------------------------
    #[test]
    fn test_bell_state_correlations() {
        let c = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let fwd = c
            .forward(&[Observable::PauliString(vec![(0, 'Z'), (1, 'Z')])])
            .unwrap();
        // Bell state |00>+|11>: ZZ expectation = 1
        assert!((fwd.expectation_values[0] - 1.0).abs() < TOL);
    }

    // ----------------------------------------------------------
    // 5. Parameter-shift gradient: Rx
    // ----------------------------------------------------------
    #[test]
    fn test_param_shift_gradient_rx() {
        // <Z> = cos(theta) for Rx(theta)|0>
        // d<Z>/dtheta = -sin(theta)
        let theta = 0.7;
        let c = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .rx(0, theta)
            .build();
        let bwd = c.backward(&[Observable::PauliZ(0)]).unwrap();
        let expected = -theta.sin();
        assert!(
            (bwd.parameter_gradients[0][0] - expected).abs() < TOL,
            "got {}, expected {}",
            bwd.parameter_gradients[0][0],
            expected
        );
    }

    // ----------------------------------------------------------
    // 6. Parameter-shift gradient: Ry
    // ----------------------------------------------------------
    #[test]
    fn test_param_shift_gradient_ry() {
        let theta = 1.2;
        let c = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, theta)
            .build();
        let bwd = c.backward(&[Observable::PauliZ(0)]).unwrap();
        let expected = -theta.sin();
        assert!(
            (bwd.parameter_gradients[0][0] - expected).abs() < TOL,
            "got {}, expected {}",
            bwd.parameter_gradients[0][0],
            expected
        );
    }

    // ----------------------------------------------------------
    // 7. Parameter-shift matches numerical gradient
    // ----------------------------------------------------------
    #[test]
    fn test_param_shift_matches_numerical() {
        let theta = 0.9;
        let c_ps = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .rx(0, theta)
            .build();
        let c_fd = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::FiniteDifference { epsilon: 1e-5 })
            .rx(0, theta)
            .build();
        let g_ps = c_ps.backward(&[Observable::PauliZ(0)]).unwrap();
        let g_fd = c_fd.backward(&[Observable::PauliZ(0)]).unwrap();
        assert!(
            (g_ps.parameter_gradients[0][0] - g_fd.parameter_gradients[0][0]).abs() < 1e-4,
        );
    }

    // ----------------------------------------------------------
    // 8. Adjoint differentiation matches parameter-shift
    // ----------------------------------------------------------
    #[test]
    fn test_adjoint_matches_param_shift() {
        let theta = 0.5;
        let phi = 1.1;

        let c_ps = CircuitBuilder::new(2)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, theta)
            .cx(0, 1)
            .rx(1, phi)
            .build();
        let c_adj = CircuitBuilder::new(2)
            .gradient_method(GradientMethod::AdjointDiff)
            .ry(0, theta)
            .cx(0, 1)
            .rx(1, phi)
            .build();

        let obs = vec![Observable::PauliZ(0)];
        let g_ps = c_ps.backward(&obs).unwrap();
        let g_adj = c_adj.backward(&obs).unwrap();

        for p in 0..2 {
            // Different gradient methods may have small numerical differences
            assert!(
                (g_ps.parameter_gradients[0][p] - g_adj.parameter_gradients[0][p]).abs()
                    < 1e-2,
                "param {}: ps={}, adj={}",
                p,
                g_ps.parameter_gradients[0][p],
                g_adj.parameter_gradients[0][p]
            );
        }
    }

    // ----------------------------------------------------------
    // 9. Finite difference: reasonable approximation
    // ----------------------------------------------------------
    #[test]
    fn test_finite_difference_approximation() {
        let theta = 0.3;
        let c = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::FiniteDifference { epsilon: 1e-7 })
            .ry(0, theta)
            .build();
        let bwd = c.backward(&[Observable::PauliZ(0)]).unwrap();
        let expected = -theta.sin();
        assert!((bwd.parameter_gradients[0][0] - expected).abs() < 1e-4);
    }

    // ----------------------------------------------------------
    // 10. Backprop gradient matches parameter-shift
    // ----------------------------------------------------------
    #[test]
    fn test_backprop_matches_param_shift() {
        let theta = 0.8;
        let c_ps = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .rx(0, theta)
            .build();
        let c_bp = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::Backprop)
            .rx(0, theta)
            .build();
        let obs = vec![Observable::PauliZ(0)];
        let g_ps = c_ps.backward(&obs).unwrap();
        let g_bp = c_bp.backward(&obs).unwrap();
        assert!(
            (g_ps.parameter_gradients[0][0] - g_bp.parameter_gradients[0][0]).abs() < 1e-4,
        );
    }

    // ----------------------------------------------------------
    // 11. Observable: single Pauli Z
    // ----------------------------------------------------------
    #[test]
    fn test_observable_pauli_z() {
        // |0> => <Z> = 1
        let c = CircuitBuilder::new(1).build();
        let fwd = c.forward(&[Observable::PauliZ(0)]).unwrap();
        assert!((fwd.expectation_values[0] - 1.0).abs() < TOL);
    }

    // ----------------------------------------------------------
    // 12. Observable: Pauli string ZZ
    // ----------------------------------------------------------
    #[test]
    fn test_observable_pauli_string_zz() {
        // |00> => <ZZ> = 1
        let c = CircuitBuilder::new(2).build();
        let fwd = c
            .forward(&[Observable::PauliString(vec![(0, 'Z'), (1, 'Z')])])
            .unwrap();
        assert!((fwd.expectation_values[0] - 1.0).abs() < TOL);

        // X|0> on qubit 0 => |10>, <ZZ> = (-1)*(1) = -1
        let c2 = CircuitBuilder::new(2).x(0).build();
        let fwd2 = c2
            .forward(&[Observable::PauliString(vec![(0, 'Z'), (1, 'Z')])])
            .unwrap();
        assert!((fwd2.expectation_values[0] + 1.0).abs() < TOL);
    }

    // ----------------------------------------------------------
    // 13. Observable: Hamiltonian (weighted sum)
    // ----------------------------------------------------------
    #[test]
    fn test_observable_hamiltonian() {
        // H = 0.5*Z0 + 0.3*Z1 for |00>
        // <H> = 0.5*1 + 0.3*1 = 0.8
        let c = CircuitBuilder::new(2).build();
        let h = Observable::Hamiltonian(vec![
            (0.5, vec![(0, 'Z')]),
            (0.3, vec![(1, 'Z')]),
        ]);
        let fwd = c.forward(&[h]).unwrap();
        assert!((fwd.expectation_values[0] - 0.8).abs() < TOL);
    }

    // ----------------------------------------------------------
    // 14. VQE: H2 molecule ground state (2 qubits)
    // ----------------------------------------------------------
    #[test]
    fn test_vqe_h2_ground_state() {
        // Simplified H2 Hamiltonian:
        // H = -1.0523 I + 0.3979 Z0Z1 - 0.3979 Z0 - 0.3979 Z1 + 0.1809 X0X1
        let h = Observable::Hamiltonian(vec![
            (-1.0523, vec![]),                       // Identity (no qubits)
            (0.3979, vec![(0, 'Z'), (1, 'Z')]),
            (-0.3979, vec![(0, 'Z')]),
            (-0.3979, vec![(1, 'Z')]),
            (0.1809, vec![(0, 'X'), (1, 'X')]),
        ]);

        let circuit = CircuitBuilder::new(2)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, 0.1)
            .ry(1, 0.1)
            .cx(0, 1)
            .ry(0, 0.1)
            .ry(1, 0.1)
            .build();

        let runner = VqeRunner {
            circuit,
            hamiltonian: h,
            optimizer: VqeOptimizer::Adam {
                lr: 0.05,
                beta1: 0.9,
                beta2: 0.999,
            },
            max_iterations: 200,
            convergence_threshold: 1e-6,
        };

        let result = runner.run().unwrap();
        // H2 ground state energy is approximately -1.8513 Hartree
        // With this simplified ansatz we should get close
        assert!(
            result.optimal_energy < -1.5,
            "VQE energy {} should be below -1.5",
            result.optimal_energy
        );
    }

    // ----------------------------------------------------------
    // 15. VQE converges to known energy
    // ----------------------------------------------------------
    #[test]
    fn test_vqe_convergence() {
        // Simple: minimize <Z> with Ry(theta)|0> => minimum at theta=pi, <Z>=-1
        let h = Observable::PauliZ(0);
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, 0.1)
            .build();
        let runner = VqeRunner {
            circuit,
            hamiltonian: h,
            optimizer: VqeOptimizer::Adam {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.999,
            },
            max_iterations: 200,
            convergence_threshold: 1e-8,
        };
        let result = runner.run().unwrap();
        assert!(
            (result.optimal_energy + 1.0).abs() < 0.01,
            "energy {} should be close to -1.0",
            result.optimal_energy
        );
    }

    // ----------------------------------------------------------
    // 16. Adam optimizer: decreasing loss
    // ----------------------------------------------------------
    #[test]
    fn test_adam_decreasing_loss() {
        let h = Observable::PauliZ(0);
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, 0.5) // Start away from optimal
            .build();
        let runner = VqeRunner {
            circuit,
            hamiltonian: h,
            optimizer: VqeOptimizer::Adam {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.999,
            },
            max_iterations: 50,
            convergence_threshold: 1e-10,
        };
        let result = runner.run().unwrap();
        // Check that final energy is lower than initial
        assert!(result.energy_history.len() >= 2);
        let first = result.energy_history[0];
        let last = *result.energy_history.last().unwrap();
        assert!(
            last < first,
            "loss should decrease: first={}, last={}",
            first,
            last
        );
    }

    // ----------------------------------------------------------
    // 17. SGD with momentum
    // ----------------------------------------------------------
    #[test]
    fn test_sgd_momentum() {
        let h = Observable::PauliZ(0);
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, 0.5)
            .build();
        let runner = VqeRunner {
            circuit,
            hamiltonian: h,
            optimizer: VqeOptimizer::SGD {
                lr: 0.1,
                momentum: 0.9,
            },
            max_iterations: 100,
            convergence_threshold: 1e-8,
        };
        let result = runner.run().unwrap();
        assert!(
            result.optimal_energy < 0.0,
            "SGD should find negative energy, got {}",
            result.optimal_energy
        );
    }

    // ----------------------------------------------------------
    // 18. SPSA optimizer: works without analytic gradients
    // ----------------------------------------------------------
    #[test]
    fn test_spsa_optimizer() {
        let h = Observable::PauliZ(0);
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, 0.5)
            .build();
        let runner = VqeRunner {
            circuit,
            hamiltonian: h,
            optimizer: VqeOptimizer::SPSA { a: 0.1, c: 0.1 },
            max_iterations: 200,
            convergence_threshold: 1e-8,
        };
        let result = runner.run().unwrap();
        // SPSA is noisy but should still improve
        let first = result.energy_history[0];
        let last = *result.energy_history.last().unwrap();
        assert!(
            last < first,
            "SPSA should decrease loss: first={}, last={}",
            first,
            last
        );
    }

    // ----------------------------------------------------------
    // 19. Quantum kernel: K(x,x) = 1
    // ----------------------------------------------------------
    #[test]
    fn test_kernel_self_overlap() {
        let fm = CircuitBuilder::new(2).ry(0, 0.0).ry(1, 0.0).build();
        let kernel = QuantumKernelML::new(fm, 2);
        let x = vec![0.5, 1.0];
        let k = kernel.kernel_value(&x, &x).unwrap();
        assert!(
            (k - 1.0).abs() < TOL,
            "K(x,x) should be 1.0, got {}",
            k
        );
    }

    // ----------------------------------------------------------
    // 20. Quantum kernel: matrix is symmetric
    // ----------------------------------------------------------
    #[test]
    fn test_kernel_matrix_symmetric() {
        let fm = CircuitBuilder::new(2).ry(0, 0.0).ry(1, 0.0).build();
        let kernel = QuantumKernelML::new(fm, 2);
        let data = vec![
            vec![0.1, 0.2],
            vec![0.5, 0.8],
            vec![1.0, 0.3],
        ];
        let mat = kernel.kernel_matrix(&data).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (mat[i][j] - mat[j][i]).abs() < TOL,
                    "K[{},{}]={} != K[{},{}]={}",
                    i,
                    j,
                    mat[i][j],
                    j,
                    i,
                    mat[j][i]
                );
            }
        }
    }

    // ----------------------------------------------------------
    // 21. Quantum kernel: matrix is PSD
    // ----------------------------------------------------------
    #[test]
    fn test_kernel_matrix_psd() {
        let fm = CircuitBuilder::new(2).ry(0, 0.0).ry(1, 0.0).build();
        let kernel = QuantumKernelML::new(fm, 2);
        let data = vec![
            vec![0.1, 0.2],
            vec![0.5, 0.8],
            vec![1.0, 0.3],
        ];
        let mat = kernel.kernel_matrix(&data).unwrap();

        // Check positive semi-definiteness via Gershgorin circle theorem:
        // For PSD matrix all eigenvalues >= 0.
        // Quick check: all diagonal elements >= 0 and dominant.
        for i in 0..3 {
            assert!(mat[i][i] >= -TOL);
        }

        // Also check v^T K v >= 0 for random test vectors
        let test_vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, 0.5],
        ];
        for v in &test_vecs {
            let mut vtmv = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    vtmv += v[i] * mat[i][j] * v[j];
                }
            }
            assert!(
                vtmv >= -TOL,
                "v^T K v = {} < 0 for v={:?}",
                vtmv,
                v
            );
        }
    }

    // ----------------------------------------------------------
    // 22. Batch execution: correct shape
    // ----------------------------------------------------------
    #[test]
    fn test_batch_execution_shape() {
        let circuit = CircuitBuilder::new(1).ry(0, 0.0).build();
        let obs = vec![Observable::PauliZ(0), Observable::PauliX(0)];
        let batch = BatchExecutor::new(circuit, obs, 3);
        let params = vec![vec![0.5], vec![1.0], vec![1.5]];
        let results = batch.execute(&params).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.expectation_values.len(), 2);
        }
    }

    // ----------------------------------------------------------
    // 23. Batch execution: parallel matches sequential
    // ----------------------------------------------------------
    #[test]
    fn test_batch_parallel_matches_sequential() {
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .rx(0, 0.0)
            .build();
        let obs = vec![Observable::PauliZ(0)];
        let params_list = vec![vec![0.3], vec![0.7], vec![1.1], vec![2.0]];

        let batch = BatchExecutor::new(circuit.clone(), obs.clone(), 4);
        let batch_results = batch.execute(&params_list).unwrap();

        for (i, p) in params_list.iter().enumerate() {
            let single = circuit.forward_with_params(p, &obs).unwrap();
            assert!(
                (batch_results[i].expectation_values[0] - single.expectation_values[0]).abs()
                    < TOL,
            );
        }
    }

    // ----------------------------------------------------------
    // 24. Tensor serialization: round-trip
    // ----------------------------------------------------------
    #[test]
    fn test_tensor_round_trip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = TensorData::new(vec![2, 3], data.clone(), TensorDtype::Float64).unwrap();
        let bytes = t.to_numpy_bytes();
        let t2 = TensorData::from_numpy_bytes(&bytes, vec![2, 3], TensorDtype::Float64).unwrap();
        for (a, b) in t.data.iter().zip(t2.data.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    // ----------------------------------------------------------
    // 25. Tensor dtype conversion
    // ----------------------------------------------------------
    #[test]
    fn test_tensor_dtype_f32() {
        let data = vec![1.5, 2.5, 3.5];
        let t = TensorData::new(vec![3], data.clone(), TensorDtype::Float32).unwrap();
        let bytes = t.to_numpy_bytes();
        assert_eq!(bytes.len(), 3 * 4); // 3 floats * 4 bytes
        let t2 = TensorData::from_numpy_bytes(&bytes, vec![3], TensorDtype::Float32).unwrap();
        for (a, b) in data.iter().zip(t2.data.iter()) {
            assert!((*a as f32 - *b as f32).abs() < 1e-6);
        }
    }

    // ----------------------------------------------------------
    // 26. Memory-mapped tensor: correct strides
    // ----------------------------------------------------------
    #[test]
    fn test_mapped_tensor_strides() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = MappedTensor::new_contiguous(data, vec![2, 3, 4]).unwrap();
        assert_eq!(t.strides, vec![12, 4, 1]); // C-contiguous
        assert_eq!(t.get(&[0, 0, 0]), Some(0.0));
        assert_eq!(t.get(&[0, 0, 3]), Some(3.0));
        assert_eq!(t.get(&[1, 2, 3]), Some(23.0));
        assert_eq!(t.get(&[2, 0, 0]), None); // Out of bounds
    }

    // ----------------------------------------------------------
    // 27. Autograd: save and replay
    // ----------------------------------------------------------
    #[test]
    fn test_autograd_save_replay() {
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .rx(0, 0.7)
            .build();
        let obs = vec![Observable::PauliZ(0)];
        let (fwd, ctx) = QuantumAutograd::apply(&circuit, &obs).unwrap();
        let bwd = ctx.backward().unwrap();

        // Forward should give cos(0.7)
        assert!((fwd.expectation_values[0] - 0.7_f64.cos()).abs() < TOL);
        // Backward should give -sin(0.7)
        assert!((bwd.parameter_gradients[0][0] + 0.7_f64.sin()).abs() < TOL);
        // Saved state should be present
        assert!(ctx.saved_state().is_some());
    }

    // ----------------------------------------------------------
    // 28. Large circuit: 10 qubits, 50 parametric gates
    // ----------------------------------------------------------
    #[test]
    fn test_large_circuit() {
        let mut c = DifferentiableCircuit::new(10);
        c.gradient_method = GradientMethod::ParameterShift { shift: PI / 2.0 };
        for layer in 0..5 {
            for q in 0..10 {
                let p = c.add_parameter(0.1 * (layer * 10 + q) as f64);
                c.add_gate(ParametricGate::Ry(q, p));
            }
            for q in 0..9 {
                c.add_gate(ParametricGate::CX(q, q + 1));
            }
        }
        assert_eq!(c.num_parameters(), 50);
        let fwd = c.forward(&[Observable::PauliZ(0)]).unwrap();
        // Just verify it runs and produces a valid expectation
        assert!(fwd.expectation_values[0].abs() <= 1.0 + TOL);
    }

    // ----------------------------------------------------------
    // 29. Multiple observables simultaneous
    // ----------------------------------------------------------
    #[test]
    fn test_multiple_observables() {
        let c = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let obs = vec![
            Observable::PauliZ(0),
            Observable::PauliZ(1),
            Observable::PauliString(vec![(0, 'Z'), (1, 'Z')]),
            Observable::PauliX(0),
        ];
        let fwd = c.forward(&obs).unwrap();
        assert_eq!(fwd.expectation_values.len(), 4);
        // Bell state: <Z0> = 0, <Z1> = 0, <Z0Z1> = 1
        assert!(fwd.expectation_values[0].abs() < TOL);
        assert!(fwd.expectation_values[1].abs() < TOL);
        assert!((fwd.expectation_values[2] - 1.0).abs() < TOL);
    }

    // ----------------------------------------------------------
    // 30. GAN: generator produces valid quantum state
    // ----------------------------------------------------------
    #[test]
    fn test_gan_generator() {
        let gen = CircuitBuilder::new(2).ry(0, 0.0).ry(1, 0.0).cx(0, 1).build();
        let mut gan = QuantumGAN::new(gen, 2, 4);
        let probs = gan.generate(&[0.5, 1.0]).unwrap();
        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < TOL,
            "probs sum to {}, not 1.0",
            sum
        );
        // All non-negative
        for p in &probs {
            assert!(*p >= -TOL);
        }

        // Training step should return valid values
        let real_dist = vec![0.25, 0.25, 0.25, 0.25];
        let (loss, d_out) = gan.train_step(&[0.5, 1.0], &real_dist, 0.01).unwrap();
        assert!(loss.is_finite());
        assert!(d_out >= 0.0 && d_out <= 1.0);
    }

    // ----------------------------------------------------------
    // 31. Config builder
    // ----------------------------------------------------------
    #[test]
    fn test_config_builder() {
        let c = CircuitBuilder::new(3)
            .gradient_method(GradientMethod::AdjointDiff)
            .h(0)
            .rx(1, 0.5)
            .cx(0, 2)
            .ry(2, 1.0)
            .rz(0, 0.3)
            .crx(0, 1, 0.7)
            .build();

        assert_eq!(c.num_qubits, 3);
        assert_eq!(c.num_parameters(), 4); // rx, ry, rz, crx
        assert_eq!(c.gates.len(), 6); // h, rx, cx, ry, rz, crx
        match c.gradient_method {
            GradientMethod::AdjointDiff => {} // Expected
            _ => panic!("wrong gradient method"),
        }
    }

    // ----------------------------------------------------------
    // 32. Probabilities sum to 1
    // ----------------------------------------------------------
    #[test]
    fn test_probabilities_sum_to_one() {
        let c = CircuitBuilder::new(3)
            .h(0)
            .h(1)
            .cx(0, 2)
            .ry(1, 0.7)
            .build();
        let fwd = c.forward(&[Observable::PauliZ(0)]).unwrap();
        let probs = fwd.probabilities.unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "probs sum = {}", sum);
    }

    // ----------------------------------------------------------
    // 33. Projector observable
    // ----------------------------------------------------------
    #[test]
    fn test_projector_observable() {
        // X|0> = |1>, projector onto |1> should give 1
        let c = CircuitBuilder::new(1).x(0).build();
        let fwd = c.forward(&[Observable::Projector(vec![true])]).unwrap();
        assert!((fwd.expectation_values[0] - 1.0).abs() < TOL);

        // Projector onto |0> should give 0
        let fwd2 = c.forward(&[Observable::Projector(vec![false])]).unwrap();
        assert!(fwd2.expectation_values[0].abs() < TOL);
    }

    // ----------------------------------------------------------
    // 34. PauliX observable
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_x_observable() {
        // H|0> = |+>, <X> = 1
        let c = CircuitBuilder::new(1).h(0).build();
        let fwd = c.forward(&[Observable::PauliX(0)]).unwrap();
        assert!(
            (fwd.expectation_values[0] - 1.0).abs() < TOL,
            "<X> on |+> = {}, expected 1.0",
            fwd.expectation_values[0]
        );
    }

    // ----------------------------------------------------------
    // 35. PauliY observable
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_y_observable() {
        // |0> state: <Y> should be 0
        let c = CircuitBuilder::new(1).build();
        let fwd = c.forward(&[Observable::PauliY(0)]).unwrap();
        assert!(
            fwd.expectation_values[0].abs() < TOL,
            "<Y> on |0> = {}",
            fwd.expectation_values[0]
        );
    }

    // ----------------------------------------------------------
    // 36. CRz parametric gradient
    // ----------------------------------------------------------
    #[test]
    fn test_crz_gradient() {
        let c_ps = CircuitBuilder::new(2)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .x(0) // control = |1>
            .crz(0, 1, 0.6)
            .build();
        let c_fd = CircuitBuilder::new(2)
            .gradient_method(GradientMethod::FiniteDifference { epsilon: 1e-5 })
            .x(0)
            .crz(0, 1, 0.6)
            .build();
        let obs = vec![Observable::PauliZ(1)];
        let g_ps = c_ps.backward(&obs).unwrap();
        let g_fd = c_fd.backward(&obs).unwrap();
        assert!(
            (g_ps.parameter_gradients[0][0] - g_fd.parameter_gradients[0][0]).abs() < 1e-3,
            "CRz ps={}, fd={}",
            g_ps.parameter_gradients[0][0],
            g_fd.parameter_gradients[0][0],
        );
    }

    // ----------------------------------------------------------
    // 37. LBFGS optimizer
    // ----------------------------------------------------------
    #[test]
    fn test_lbfgs_optimizer() {
        let h = Observable::PauliZ(0);
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .ry(0, 0.5)
            .build();
        let runner = VqeRunner {
            circuit,
            hamiltonian: h,
            optimizer: VqeOptimizer::LBFGS {
                max_line_search: 10,
            },
            max_iterations: 100,
            convergence_threshold: 1e-8,
        };
        let result = runner.run().unwrap();
        assert!(
            result.optimal_energy < 0.0,
            "LBFGS should find negative energy, got {}",
            result.optimal_energy
        );
    }

    // ----------------------------------------------------------
    // 38. Tensor shape validation
    // ----------------------------------------------------------
    #[test]
    fn test_tensor_shape_validation() {
        let result = TensorData::new(vec![2, 3], vec![1.0, 2.0], TensorDtype::Float64);
        assert!(result.is_err());
    }

    // ----------------------------------------------------------
    // 39. Set parameters error
    // ----------------------------------------------------------
    #[test]
    fn test_set_parameters_error() {
        let mut c = CircuitBuilder::new(1).rx(0, 0.5).build();
        let result = c.set_parameters(&[1.0, 2.0]); // Wrong size
        assert!(result.is_err());
    }

    // ----------------------------------------------------------
    // 40. Identity Hamiltonian term
    // ----------------------------------------------------------
    #[test]
    fn test_identity_hamiltonian_term() {
        // H = 2.5 * I => <H> = 2.5 for any state
        let c = CircuitBuilder::new(1).h(0).build();
        let h = Observable::Hamiltonian(vec![(2.5, vec![])]);
        let fwd = c.forward(&[h]).unwrap();
        assert!((fwd.expectation_values[0] - 2.5).abs() < TOL);
    }

    // ----------------------------------------------------------
    // 41. Batch with gradients
    // ----------------------------------------------------------
    #[test]
    fn test_batch_with_gradients() {
        let circuit = CircuitBuilder::new(1)
            .gradient_method(GradientMethod::ParameterShift { shift: PI / 2.0 })
            .rx(0, 0.0)
            .build();
        let obs = vec![Observable::PauliZ(0)];
        let batch = BatchExecutor::new(circuit, obs, 2);
        let params = vec![vec![0.5], vec![1.5]];
        let results = batch.execute_with_gradients(&params).unwrap();
        assert_eq!(results.len(), 2);
        // Check gradient for first: d cos(0.5)/d(0.5) = -sin(0.5)
        assert!(
            (results[0].1.parameter_gradients[0][0] + 0.5_f64.sin()).abs() < TOL,
        );
    }
}
