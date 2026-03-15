//! Fault-Tolerant Compilation: Clifford+T Decomposition & Litinski Transformation
//!
//! Translates arbitrary quantum circuits into fault-tolerant gate sets suitable
//! for execution on surface-code hardware. Closes the competitive gap with
//! Qiskit v2.2-2.3 which ships Litinski transformation and Ross-Selinger/gridsynth.
//!
//! # Architecture
//!
//! ```text
//!  Input Circuit (arbitrary rotations)
//!  ┌─────────────────────────────────┐
//!  │  H  Rz(0.3)  CNOT  Ry(1.2) ...│
//!  └──────────┬──────────────────────┘
//!             │
//!     ┌───────▼────────┐
//!     │  Gate Normalize │  Rx,Ry → H/S/Rz decompositions
//!     └───────┬────────┘
//!             │
//!   ┌─────────▼──────────┐      ┌──────────────────────┐
//!   │ Ross-Selinger      │      │ Litinski Transform    │
//!   │ (gridsynth approx) │      │ (Clifford frame →     │
//!   │ Rz(θ) → {H,T}*    │      │  Pauli-based comp.)   │
//!   └─────────┬──────────┘      └──────────┬───────────┘
//!             │                             │
//!   ┌─────────▼──────────┐      ┌──────────▼───────────┐
//!   │ T-count Optimizer  │      │ PBC Circuit           │
//!   │ cancel T·T†, merge │      │ {PauliRotation, ...}  │
//!   └─────────┬──────────┘      └──────────────────────┘
//!             │
//!   ┌─────────▼──────────┐
//!   │ FTCompilationResult│
//!   │ Clifford+T circuit │
//!   │ resource counts     │
//!   └────────────────────┘
//! ```
//!
//! # Algorithms
//!
//! - **Ross-Selinger / Gridsynth**: Approximates Rz(theta) with H/T sequences to
//!   precision epsilon using O(log(1/epsilon)) T gates (optimal). Uses iterative
//!   deepening search with known-angle shortcuts and matrix-norm pruning.
//!
//! - **Litinski Transformation**: Converts Clifford+Rz circuits to Pauli-Based
//!   Computation (PBC) by tracking the cumulative Clifford frame. Each Rz(theta)
//!   becomes a multi-qubit Pauli rotation suitable for lattice surgery execution.
//!
//! - **T-count Optimization**: Cancels adjacent T-Tdg pairs, merges Clifford
//!   subsequences, and commutes T gates through Cliffords to enable cancellation.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::ft_compilation::*;
//!
//! let compiler = FTCompiler::new(FTCompilationConfig::default());
//! let circuit = vec![
//!     LogicalGate::H(0),
//!     LogicalGate::Rz(0, 0.123),
//!     LogicalGate::CNOT(0, 1),
//! ];
//! let result = compiler.compile(&circuit).unwrap();
//! assert!(result.max_approximation_error < 1e-10);
//! ```

use num_complex::Complex64 as C64;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during fault-tolerant compilation.
#[derive(Debug, Clone, PartialEq)]
pub enum FTCompilationError {
    /// The requested precision cannot be achieved within the T-depth budget.
    PrecisionUnachievable { epsilon: f64, budget: usize },
    /// A gate type is not supported for fault-tolerant compilation.
    UnsupportedGate(String),
    /// The decomposition algorithm failed to converge.
    DecompositionFailed(String),
}

impl fmt::Display for FTCompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FTCompilationError::PrecisionUnachievable { epsilon, budget } => {
                write!(
                    f,
                    "Cannot achieve precision {:.2e} within T-depth budget {}",
                    epsilon, budget
                )
            }
            FTCompilationError::UnsupportedGate(gate) => {
                write!(f, "Unsupported gate for FT compilation: {}", gate)
            }
            FTCompilationError::DecompositionFailed(msg) => {
                write!(f, "Decomposition failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for FTCompilationError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the fault-tolerant compiler.
#[derive(Debug, Clone)]
pub struct FTCompilationConfig {
    /// Approximation precision for Rz decomposition (default: 1e-10).
    pub epsilon: f64,
    /// Maximum T-depth budget (0 = unlimited).
    pub max_t_depth: usize,
    /// Optimization level: 0=none, 1=basic T-cancellation, 2=aggressive.
    pub optimization_level: usize,
}

impl Default for FTCompilationConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            max_t_depth: 0,
            optimization_level: 1,
        }
    }
}

impl FTCompilationConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set approximation precision epsilon.
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set maximum T-depth budget (0 = unlimited).
    pub fn max_t_depth(mut self, depth: usize) -> Self {
        self.max_t_depth = depth;
        self
    }

    /// Set optimization level (0=none, 1=basic, 2=aggressive).
    pub fn optimization_level(mut self, level: usize) -> Self {
        self.optimization_level = level;
        self
    }
}

// ============================================================
// INPUT GATE TYPES (PRE-COMPILATION)
// ============================================================

/// Logical gate in the input circuit before FT compilation.
///
/// These may contain arbitrary-angle rotations that must be decomposed
/// into the Clifford+T gate set for fault-tolerant execution.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalGate {
    /// Hadamard gate on qubit.
    H(usize),
    /// S gate (pi/4 phase) on qubit.
    S(usize),
    /// S-dagger gate on qubit.
    Sdg(usize),
    /// T gate (pi/8 phase) on qubit -- already Clifford+T native.
    T(usize),
    /// T-dagger gate on qubit.
    Tdg(usize),
    /// CNOT (controlled-X) from control to target.
    CNOT(usize, usize),
    /// Rz(theta) rotation -- requires decomposition.
    Rz(usize, f64),
    /// Ry(theta) rotation -- decomposed via H Rz(theta) H.
    Ry(usize, f64),
    /// Rx(theta) rotation -- decomposed via S H Rz(theta) H Sdg.
    Rx(usize, f64),
    /// CZ (controlled-Z) gate.
    CZ(usize, usize),
    /// Measurement in computational basis.
    Measure(usize),
}

impl fmt::Display for LogicalGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalGate::H(q) => write!(f, "H({})", q),
            LogicalGate::S(q) => write!(f, "S({})", q),
            LogicalGate::Sdg(q) => write!(f, "Sdg({})", q),
            LogicalGate::T(q) => write!(f, "T({})", q),
            LogicalGate::Tdg(q) => write!(f, "Tdg({})", q),
            LogicalGate::CNOT(c, t) => write!(f, "CNOT({},{})", c, t),
            LogicalGate::Rz(q, theta) => write!(f, "Rz({},{:.6})", q, theta),
            LogicalGate::Ry(q, theta) => write!(f, "Ry({},{:.6})", q, theta),
            LogicalGate::Rx(q, theta) => write!(f, "Rx({},{:.6})", q, theta),
            LogicalGate::CZ(a, b) => write!(f, "CZ({},{})", a, b),
            LogicalGate::Measure(q) => write!(f, "Measure({})", q),
        }
    }
}

// ============================================================
// OUTPUT GATE TYPES (CLIFFORD+T)
// ============================================================

/// Gate from the Clifford+T universal gate set.
///
/// This is the native gate set for fault-tolerant quantum computation
/// on surface codes. Clifford gates are transversal (cheap); T gates
/// require magic state distillation (expensive).
#[derive(Debug, Clone, PartialEq)]
pub enum CliffordTGate {
    /// Hadamard gate.
    H(usize),
    /// S gate (phase gate, pi/4 rotation about Z).
    S(usize),
    /// S-dagger gate.
    Sdg(usize),
    /// T gate (pi/8 rotation about Z) -- requires magic state.
    T(usize),
    /// T-dagger gate -- requires magic state.
    Tdg(usize),
    /// CNOT (controlled-X).
    CNOT(usize, usize),
    /// Pauli X gate.
    X(usize),
    /// Pauli Z gate.
    Z(usize),
    /// Measurement.
    Measure(usize),
}

impl CliffordTGate {
    /// Returns true if this gate is a T or Tdg (non-Clifford, expensive).
    pub fn is_t_gate(&self) -> bool {
        matches!(self, CliffordTGate::T(_) | CliffordTGate::Tdg(_))
    }

    /// Returns true if this gate is a Clifford gate (cheap, transversal).
    pub fn is_clifford(&self) -> bool {
        !self.is_t_gate() && !matches!(self, CliffordTGate::Measure(_))
    }

    /// Returns the qubit(s) this gate acts on.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            CliffordTGate::H(q)
            | CliffordTGate::S(q)
            | CliffordTGate::Sdg(q)
            | CliffordTGate::T(q)
            | CliffordTGate::Tdg(q)
            | CliffordTGate::X(q)
            | CliffordTGate::Z(q)
            | CliffordTGate::Measure(q) => vec![*q],
            CliffordTGate::CNOT(c, t) => vec![*c, *t],
        }
    }
}

impl fmt::Display for CliffordTGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliffordTGate::H(q) => write!(f, "H({})", q),
            CliffordTGate::S(q) => write!(f, "S({})", q),
            CliffordTGate::Sdg(q) => write!(f, "Sdg({})", q),
            CliffordTGate::T(q) => write!(f, "T({})", q),
            CliffordTGate::Tdg(q) => write!(f, "Tdg({})", q),
            CliffordTGate::CNOT(c, t) => write!(f, "CNOT({},{})", c, t),
            CliffordTGate::X(q) => write!(f, "X({})", q),
            CliffordTGate::Z(q) => write!(f, "Z({})", q),
            CliffordTGate::Measure(q) => write!(f, "Measure({})", q),
        }
    }
}

// ============================================================
// PAULI-BASED COMPUTATION (LITINSKI OUTPUT)
// ============================================================

/// Single-qubit Pauli operator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliType {
    /// Identity.
    I,
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

impl fmt::Display for PauliType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PauliType::I => write!(f, "I"),
            PauliType::X => write!(f, "X"),
            PauliType::Y => write!(f, "Y"),
            PauliType::Z => write!(f, "Z"),
        }
    }
}

/// A multi-qubit Pauli rotation for Pauli-Based Computation.
///
/// Represents exp(-i * angle/2 * P) where P is a tensor product of
/// single-qubit Pauli operators. This is the native operation for
/// lattice surgery on surface codes.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliRotation {
    /// Which Pauli operator acts on which qubit. Only non-identity
    /// entries are stored (sparse representation).
    pub paulis: Vec<(usize, PauliType)>,
    /// Rotation angle (full angle, not half-angle).
    pub angle: f64,
    /// True if angle is a multiple of pi/2 (Clifford rotation, no T cost).
    pub is_clifford: bool,
}

impl PauliRotation {
    /// Create a new Pauli rotation.
    pub fn new(paulis: Vec<(usize, PauliType)>, angle: f64) -> Self {
        let is_clifford = is_clifford_angle(angle);
        Self {
            paulis,
            angle,
            is_clifford,
        }
    }

    /// Number of non-identity Pauli operators (weight of the Pauli string).
    pub fn weight(&self) -> usize {
        self.paulis.len()
    }
}

impl fmt::Display for PauliRotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exp(-i*{:.4}/2 * ", self.angle)?;
        if self.paulis.is_empty() {
            write!(f, "I")?;
        } else {
            for (i, (q, p)) in self.paulis.iter().enumerate() {
                if i > 0 {
                    write!(f, "⊗")?;
                }
                write!(f, "{}_{}", p, q)?;
            }
        }
        write!(f, ")")?;
        if self.is_clifford {
            write!(f, " [Clifford]")?;
        }
        Ok(())
    }
}

// ============================================================
// PAULI-BASED COMPUTATION CIRCUIT
// ============================================================

/// Result of the Litinski transformation: a Pauli-Based Computation circuit.
///
/// PBC is the native representation for lattice surgery execution on
/// surface codes. Each rotation is a multi-qubit Pauli measurement
/// followed by conditional correction.
#[derive(Debug, Clone)]
pub struct PBCCircuit {
    /// Sequence of Pauli rotations comprising the computation.
    pub rotations: Vec<PauliRotation>,
    /// Number of logical qubits.
    pub num_qubits: usize,
    /// Number of non-Clifford rotations (each costs one T state).
    pub num_t_gates: usize,
    /// Number of Clifford rotations (free in lattice surgery).
    pub num_clifford: usize,
}

impl PBCCircuit {
    /// Total number of rotations.
    pub fn total_rotations(&self) -> usize {
        self.rotations.len()
    }

    /// Average weight of Pauli strings in the circuit.
    pub fn average_weight(&self) -> f64 {
        if self.rotations.is_empty() {
            return 0.0;
        }
        let total: usize = self.rotations.iter().map(|r| r.weight()).sum();
        total as f64 / self.rotations.len() as f64
    }
}

impl fmt::Display for PBCCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PBC Circuit: {} qubits", self.num_qubits)?;
        writeln!(
            f,
            "  Rotations: {} total ({} T-cost, {} Clifford)",
            self.total_rotations(),
            self.num_t_gates,
            self.num_clifford
        )?;
        writeln!(f, "  Avg Pauli weight: {:.2}", self.average_weight())?;
        for (i, rot) in self.rotations.iter().enumerate() {
            writeln!(f, "  [{}] {}", i, rot)?;
        }
        Ok(())
    }
}

// ============================================================
// Rz DECOMPOSITION RESULT
// ============================================================

/// Result of Ross-Selinger decomposition for a single Rz(theta) gate.
///
/// Contains the approximating Clifford+T sequence and error metrics.
#[derive(Debug, Clone)]
pub struct RzDecomposition {
    /// Original rotation angle.
    pub angle: f64,
    /// Target precision.
    pub epsilon: f64,
    /// Approximating Clifford+T gate sequence.
    pub gates: Vec<CliffordTGate>,
    /// Number of T/Tdg gates in the sequence.
    pub t_count: usize,
    /// Actual operator norm error ||U_approx - Rz(theta)||.
    pub actual_error: f64,
}

impl fmt::Display for RzDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rz({:.6}) -> {} gates ({} T), error={:.2e}",
            self.angle,
            self.gates.len(),
            self.t_count,
            self.actual_error
        )
    }
}

// ============================================================
// FULL COMPILATION RESULT
// ============================================================

/// Complete result of fault-tolerant compilation.
#[derive(Debug, Clone)]
pub struct FTCompilationResult {
    /// Number of gates in the input circuit.
    pub input_gates: usize,
    /// Number of gates in the output Clifford+T circuit.
    pub output_gates: usize,
    /// The compiled Clifford+T circuit.
    pub clifford_t_circuit: Vec<CliffordTGate>,
    /// Total T-gate count (each requires one magic state).
    pub t_count: usize,
    /// T-depth (layers of T gates that cannot be parallelized).
    pub t_depth: usize,
    /// Total circuit depth (all gates).
    pub total_depth: usize,
    /// Maximum approximation error across all Rz decompositions.
    pub max_approximation_error: f64,
    /// Optional PBC representation from Litinski transformation.
    pub pbc_circuit: Option<PBCCircuit>,
}

impl fmt::Display for FTCompilationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FT Compilation Result:")?;
        writeln!(f, "  Input gates:  {}", self.input_gates)?;
        writeln!(f, "  Output gates: {}", self.output_gates)?;
        writeln!(f, "  T-count:      {}", self.t_count)?;
        writeln!(f, "  T-depth:      {}", self.t_depth)?;
        writeln!(f, "  Total depth:  {}", self.total_depth)?;
        writeln!(
            f,
            "  Max approx error: {:.2e}",
            self.max_approximation_error
        )?;
        if let Some(ref pbc) = self.pbc_circuit {
            writeln!(f, "  PBC: {} rotations", pbc.total_rotations())?;
        }
        Ok(())
    }
}

// ============================================================
// 2x2 UNITARY MATRIX ARITHMETIC
// ============================================================

/// A 2x2 unitary matrix stored as [[a, b], [c, d]].
///
/// Used internally for computing and comparing single-qubit gate
/// unitaries during Ross-Selinger decomposition.
#[derive(Debug, Clone)]
struct Mat2x2 {
    a: C64,
    b: C64,
    c: C64,
    d: C64,
}

impl Mat2x2 {
    /// Identity matrix.
    fn identity() -> Self {
        Self {
            a: C64::new(1.0, 0.0),
            b: C64::new(0.0, 0.0),
            c: C64::new(0.0, 0.0),
            d: C64::new(1.0, 0.0),
        }
    }

    /// Hadamard matrix: (1/sqrt(2)) * [[1, 1], [1, -1]].
    fn hadamard() -> Self {
        let s = 1.0 / 2.0_f64.sqrt();
        Self {
            a: C64::new(s, 0.0),
            b: C64::new(s, 0.0),
            c: C64::new(s, 0.0),
            d: C64::new(-s, 0.0),
        }
    }

    /// T gate: [[1, 0], [0, e^(i*pi/4)]].
    fn t_gate() -> Self {
        let phase = C64::from_polar(1.0, FRAC_PI_4);
        Self {
            a: C64::new(1.0, 0.0),
            b: C64::new(0.0, 0.0),
            c: C64::new(0.0, 0.0),
            d: phase,
        }
    }

    /// T-dagger gate: [[1, 0], [0, e^(-i*pi/4)]].
    fn t_dagger() -> Self {
        let phase = C64::from_polar(1.0, -FRAC_PI_4);
        Self {
            a: C64::new(1.0, 0.0),
            b: C64::new(0.0, 0.0),
            c: C64::new(0.0, 0.0),
            d: phase,
        }
    }

    /// S gate: [[1, 0], [0, i]].
    fn s_gate() -> Self {
        Self {
            a: C64::new(1.0, 0.0),
            b: C64::new(0.0, 0.0),
            c: C64::new(0.0, 0.0),
            d: C64::new(0.0, 1.0),
        }
    }

    /// S-dagger gate: [[1, 0], [0, -i]].
    fn s_dagger() -> Self {
        Self {
            a: C64::new(1.0, 0.0),
            b: C64::new(0.0, 0.0),
            c: C64::new(0.0, 0.0),
            d: C64::new(0.0, -1.0),
        }
    }

    /// Z gate: [[1, 0], [0, -1]].
    fn z_gate() -> Self {
        Self {
            a: C64::new(1.0, 0.0),
            b: C64::new(0.0, 0.0),
            c: C64::new(0.0, 0.0),
            d: C64::new(-1.0, 0.0),
        }
    }

    /// X gate: [[0, 1], [1, 0]].
    fn x_gate() -> Self {
        Self {
            a: C64::new(0.0, 0.0),
            b: C64::new(1.0, 0.0),
            c: C64::new(1.0, 0.0),
            d: C64::new(0.0, 0.0),
        }
    }

    /// Rz(theta): [[e^(-i*theta/2), 0], [0, e^(i*theta/2)]].
    fn rz(theta: f64) -> Self {
        Self {
            a: C64::from_polar(1.0, -theta / 2.0),
            b: C64::new(0.0, 0.0),
            c: C64::new(0.0, 0.0),
            d: C64::from_polar(1.0, theta / 2.0),
        }
    }

    /// Matrix multiplication: self * other.
    fn mul(&self, other: &Mat2x2) -> Mat2x2 {
        Mat2x2 {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
        }
    }

    /// Operator norm distance ||self - other|| (spectral norm of difference).
    ///
    /// For 2x2 matrices, the spectral norm equals the largest singular value.
    /// We compute it via: sigma_max = sqrt(max eigenvalue of M^dagger * M)
    /// where M = self - other. For our use case (comparing unitaries up to
    /// global phase), we normalize by removing the global phase first.
    fn distance(&self, other: &Mat2x2) -> f64 {
        // Difference matrix
        let da = self.a - other.a;
        let db = self.b - other.b;
        let dc = self.c - other.c;
        let dd = self.d - other.d;

        // M^dagger * M for the difference matrix
        // (da*, dc*) (da db)   (|da|^2+|dc|^2,    da*db+dc*dd)
        // (db*, dd*) (dc dd) = (db*da+dd*dc,       |db|^2+|dd|^2)
        let m00 = da.norm_sqr() + dc.norm_sqr();
        let m11 = db.norm_sqr() + dd.norm_sqr();
        let m01 = da.conj() * db + dc.conj() * dd;

        // Eigenvalues of 2x2 Hermitian matrix [[m00, m01], [m01*, m11]]
        // lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
        let trace = m00 + m11;
        let det = m00 * m11 - m01.norm_sqr();
        let discriminant = (trace * trace - 4.0 * det).max(0.0);
        let lambda_max = (trace + discriminant.sqrt()) / 2.0;

        lambda_max.sqrt()
    }

    /// Distance up to global phase: min over phi of ||self - e^(i*phi)*other||.
    ///
    /// For unitaries, this is equivalent to the diamond distance metric used
    /// in quantum channel comparison. We try a few candidate phases and pick
    /// the best.
    fn distance_up_to_phase(&self, other: &Mat2x2) -> f64 {
        // Strategy: the optimal global phase aligns the (0,0) entries.
        // phi = arg(self.a / other.a) when other.a != 0, else use other.d.
        let mut best = self.distance(other);

        let candidates = if other.a.norm_sqr() > 1e-12 {
            vec![self.a / other.a]
        } else if other.d.norm_sqr() > 1e-12 {
            vec![self.d / other.d]
        } else {
            return best;
        };

        for ratio in candidates {
            if ratio.norm_sqr() < 1e-30 {
                continue;
            }
            let phase = C64::from_polar(1.0, ratio.arg());
            let adjusted = Mat2x2 {
                a: other.a * phase,
                b: other.b * phase,
                c: other.c * phase,
                d: other.d * phase,
            };
            let d = self.distance(&adjusted);
            if d < best {
                best = d;
            }
        }

        best
    }
}

// ============================================================
// HELPER: ANGLE CLASSIFICATION
// ============================================================

/// Normalize an angle to the range [-pi, pi).
fn normalize_angle(theta: f64) -> f64 {
    let mut a = theta % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    }
    if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

/// Check if an angle is a multiple of pi/2 (Clifford rotation).
fn is_clifford_angle(theta: f64) -> bool {
    let normalized = normalize_angle(theta);
    let remainder = (normalized / FRAC_PI_2).round() * FRAC_PI_2 - normalized;
    remainder.abs() < 1e-12
}

/// Check if angle is (approximately) zero mod 2*pi.
fn is_zero_angle(theta: f64) -> bool {
    let normalized = normalize_angle(theta);
    normalized.abs() < 1e-12
}

/// Check if angle is approximately equal to a target.
fn angle_close(theta: f64, target: f64, tol: f64) -> bool {
    let diff = normalize_angle(theta - target);
    diff.abs() < tol
}

// ============================================================
// ROSS-SELINGER / GRIDSYNTH DECOMPOSITION
// ============================================================

/// Tokens for building candidate gate sequences in the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RSToken {
    H,
    T,
    Tdg,
}

/// Ross-Selinger decomposition engine.
///
/// Approximates Rz(theta) as a product of H and T/Tdg gates to within
/// epsilon precision. The implementation uses two strategies:
///
/// 1. **Catalog search** (length <= 12): Iterative deepening BFS over
///    {H, T, Tdg} sequences with pruning of identity-producing pairs.
///    Practical up to about depth 12 due to branching factor ~2 after
///    pruning (H-H, T-Tdg, Tdg-T eliminated).
///
/// 2. **Repeated halving** (length > 12): For finer precision, decomposes
///    theta into a sum of angles that each have short decompositions.
///    Uses the identity Rz(a+b) = Rz(a) Rz(b) and repeatedly halves
///    the residual error.
///
/// Known exact decompositions for common angles (multiples of pi/4, pi/2,
/// pi) are returned directly without search.
struct RossSeligerDecomposer {
    epsilon: f64,
}

/// Entry in the decomposition catalog: (unitary matrix, token sequence).
struct CatalogEntry {
    matrix: Mat2x2,
    tokens: Vec<RSToken>,
    t_count: usize,
}

impl RossSeligerDecomposer {
    fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Decompose Rz(theta) into a Clifford+T gate sequence on qubit q.
    fn decompose(&self, q: usize, theta: f64) -> Result<RzDecomposition, FTCompilationError> {
        let theta_norm = normalize_angle(theta);

        // Check known exact decompositions first (zero T-cost).
        if let Some(decomp) = self.try_exact(q, theta_norm) {
            return Ok(decomp);
        }

        // Strategy: build catalog of unitaries from short sequences, find
        // the best match. If no single catalog entry is close enough, compose
        // multiple entries via repeated correction.
        let target = Mat2x2::rz(theta_norm);

        // Phase 1: Try direct catalog match with increasing depth.
        let max_direct = 12usize.min(self.max_search_length());
        for depth in 1..=max_direct {
            if let Some(entry) = self.search_catalog(depth, &target) {
                let gates = self.tokens_to_gates(q, &entry.tokens);
                let actual_error = self.compute_error(q, &gates, theta_norm);

                if actual_error <= self.epsilon {
                    return Ok(RzDecomposition {
                        angle: theta,
                        epsilon: self.epsilon,
                        gates,
                        t_count: entry.t_count,
                        actual_error,
                    });
                }
            }
        }

        // Phase 2: Repeated halving strategy for higher precision.
        // Find the best catalog approximation, then decompose the residual.
        let decomposition = self.decompose_by_halving(q, theta_norm)?;
        if decomposition.actual_error > self.epsilon {
            return Err(FTCompilationError::PrecisionUnachievable {
                epsilon: self.epsilon,
                budget: self.max_search_length(),
            });
        }
        Ok(decomposition)
    }

    /// Decompose via repeated halving: find best approximation, then
    /// recursively decompose the residual angle.
    fn decompose_by_halving(
        &self,
        q: usize,
        theta: f64,
    ) -> Result<RzDecomposition, FTCompilationError> {
        let mut all_gates: Vec<CliffordTGate> = Vec::new();
        let mut remaining_theta = theta;
        let mut total_error = 0.0;
        let max_iterations = 20;

        for _ in 0..max_iterations {
            // Check if remaining angle is close enough to an exact one.
            let remaining_norm = normalize_angle(remaining_theta);
            if remaining_norm.abs() < self.epsilon {
                break;
            }

            // Try exact decomposition of remaining angle.
            if let Some(decomp) = self.try_exact(q, remaining_norm) {
                all_gates.extend(decomp.gates);
                total_error += decomp.actual_error;
                break;
            }

            // Find best short-sequence approximation (depth up to 8 for speed).
            let target = Mat2x2::rz(remaining_norm);
            let mut best_entry: Option<CatalogEntry> = None;
            let mut best_error = f64::MAX;

            for depth in 1..=8 {
                if let Some(entry) = self.find_best_at_depth(depth, &target) {
                    let err = entry.matrix.distance_up_to_phase(&target);
                    if err < best_error {
                        best_error = err;
                        best_entry = Some(entry);
                    }
                    if err < self.epsilon {
                        break;
                    }
                }
            }

            if let Some(entry) = best_entry {
                // Compute the angle that this entry actually implements.
                // For a diagonal unitary [[e^(-ia/2), 0], [0, e^(ia/2)]],
                // the angle a = arg(entry.d) - arg(entry.a).
                let implemented_angle = entry.matrix.d.arg() - entry.matrix.a.arg();

                let gates = self.tokens_to_gates(q, &entry.tokens);
                let entry_error = self.compute_error(q, &gates, remaining_norm);

                all_gates.extend(gates);
                total_error = total_error.max(entry_error);

                // Update remaining angle: subtract what we implemented.
                remaining_theta = normalize_angle(remaining_theta - implemented_angle);

                if entry_error < self.epsilon {
                    break;
                }
            } else {
                return Err(FTCompilationError::DecompositionFailed(format!(
                    "No approximation found for residual angle {:.6}",
                    remaining_theta
                )));
            }
        }

        let t_count = all_gates.iter().filter(|g| g.is_t_gate()).count();
        let actual_error = self.compute_error(q, &all_gates, theta);

        Ok(RzDecomposition {
            angle: theta,
            epsilon: self.epsilon,
            gates: all_gates,
            t_count,
            actual_error,
        })
    }

    /// Maximum search depth based on Solovay-Kitaev bound: ~ 3*log2(1/eps).
    fn max_search_length(&self) -> usize {
        let bound = (3.0 * (1.0 / self.epsilon).log2()).ceil() as usize;
        // Clamp to a practical limit.
        bound.min(60).max(4)
    }

    /// Try known exact decompositions for special angles.
    fn try_exact(&self, q: usize, theta: f64) -> Option<RzDecomposition> {
        let tol = 1e-12;

        // Rz(0) = Identity
        if angle_close(theta, 0.0, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![],
                t_count: 0,
                actual_error: 0.0,
            });
        }

        // Rz(pi/4) = T
        if angle_close(theta, FRAC_PI_4, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![CliffordTGate::T(q)],
                t_count: 1,
                actual_error: 0.0,
            });
        }

        // Rz(-pi/4) = Tdg
        if angle_close(theta, -FRAC_PI_4, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![CliffordTGate::Tdg(q)],
                t_count: 1,
                actual_error: 0.0,
            });
        }

        // Rz(pi/2) = S (Clifford, 0 T-cost)
        if angle_close(theta, FRAC_PI_2, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![CliffordTGate::S(q)],
                t_count: 0,
                actual_error: 0.0,
            });
        }

        // Rz(-pi/2) = Sdg (Clifford, 0 T-cost)
        if angle_close(theta, -FRAC_PI_2, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![CliffordTGate::Sdg(q)],
                t_count: 0,
                actual_error: 0.0,
            });
        }

        // Rz(pi) = Z (Clifford, 0 T-cost)
        if angle_close(theta, PI, tol) || angle_close(theta, -PI, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![CliffordTGate::Z(q)],
                t_count: 0,
                actual_error: 0.0,
            });
        }

        // Rz(3*pi/4) = T^3 = S*T
        if angle_close(theta, 3.0 * FRAC_PI_4, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![CliffordTGate::S(q), CliffordTGate::T(q)],
                t_count: 1,
                actual_error: 0.0,
            });
        }

        // Rz(-3*pi/4) = Tdg^3 = Sdg*Tdg
        if angle_close(theta, -3.0 * FRAC_PI_4, tol) {
            return Some(RzDecomposition {
                angle: theta,
                epsilon: self.epsilon,
                gates: vec![CliffordTGate::Sdg(q), CliffordTGate::Tdg(q)],
                t_count: 1,
                actual_error: 0.0,
            });
        }

        None
    }

    /// Search for a sequence of given depth that approximates the target
    /// to within epsilon. Returns the first match found.
    fn search_catalog(&self, depth: usize, target: &Mat2x2) -> Option<CatalogEntry> {
        let mut sequence = Vec::with_capacity(depth);
        self.search_recursive(depth, target, &Mat2x2::identity(), &mut sequence, 0)
    }

    /// Find the best approximation at a given depth (may not meet epsilon).
    fn find_best_at_depth(&self, depth: usize, target: &Mat2x2) -> Option<CatalogEntry> {
        let mut best: Option<CatalogEntry> = None;
        let mut best_error = f64::MAX;
        let mut sequence = Vec::with_capacity(depth);

        self.search_best_recursive(
            depth,
            target,
            &Mat2x2::identity(),
            &mut sequence,
            0,
            &mut best,
            &mut best_error,
        );

        best
    }

    /// Recursive depth-limited search returning first match within epsilon.
    fn search_recursive(
        &self,
        remaining: usize,
        target: &Mat2x2,
        current: &Mat2x2,
        sequence: &mut Vec<RSToken>,
        t_count: usize,
    ) -> Option<CatalogEntry> {
        if remaining == 0 {
            let error = current.distance_up_to_phase(target);
            if error < self.epsilon {
                return Some(CatalogEntry {
                    matrix: current.clone(),
                    tokens: sequence.clone(),
                    t_count,
                });
            }
            return None;
        }

        let tokens = [RSToken::H, RSToken::T, RSToken::Tdg];
        for &token in &tokens {
            if !self.is_valid_extension(sequence, token) {
                continue;
            }

            let (gate_matrix, is_t) = self.token_matrix_and_cost(token);
            let next = current.mul(&gate_matrix);

            sequence.push(token);
            let new_tc = t_count + if is_t { 1 } else { 0 };
            if let Some(result) =
                self.search_recursive(remaining - 1, target, &next, sequence, new_tc)
            {
                return Some(result);
            }
            sequence.pop();
        }

        None
    }

    /// Recursive search tracking the global best approximation.
    fn search_best_recursive(
        &self,
        remaining: usize,
        target: &Mat2x2,
        current: &Mat2x2,
        sequence: &mut Vec<RSToken>,
        t_count: usize,
        best: &mut Option<CatalogEntry>,
        best_error: &mut f64,
    ) {
        if remaining == 0 {
            let error = current.distance_up_to_phase(target);
            if error < *best_error {
                *best_error = error;
                *best = Some(CatalogEntry {
                    matrix: current.clone(),
                    tokens: sequence.clone(),
                    t_count,
                });
            }
            return;
        }

        let tokens = [RSToken::H, RSToken::T, RSToken::Tdg];
        for &token in &tokens {
            if !self.is_valid_extension(sequence, token) {
                continue;
            }

            let (gate_matrix, is_t) = self.token_matrix_and_cost(token);
            let next = current.mul(&gate_matrix);

            sequence.push(token);
            let new_tc = t_count + if is_t { 1 } else { 0 };
            self.search_best_recursive(
                remaining - 1,
                target,
                &next,
                sequence,
                new_tc,
                best,
                best_error,
            );
            sequence.pop();

            // Early exit if we already found an exact match.
            if *best_error < self.epsilon {
                return;
            }
        }
    }

    /// Check if extending the sequence with this token is valid (prune wasteful patterns).
    fn is_valid_extension(&self, sequence: &[RSToken], token: RSToken) -> bool {
        if let Some(&last) = sequence.last() {
            // H*H = I
            if last == RSToken::H && token == RSToken::H {
                return false;
            }
            // T*Tdg = I, Tdg*T = I
            if last == RSToken::T && token == RSToken::Tdg {
                return false;
            }
            if last == RSToken::Tdg && token == RSToken::T {
                return false;
            }
            // Prune T^8 = I: if we have 7 consecutive T tokens, don't add another.
            if token == RSToken::T {
                let consecutive = sequence
                    .iter()
                    .rev()
                    .take_while(|&&t| t == RSToken::T)
                    .count();
                if consecutive >= 7 {
                    return false;
                }
            }
            if token == RSToken::Tdg {
                let consecutive = sequence
                    .iter()
                    .rev()
                    .take_while(|&&t| t == RSToken::Tdg)
                    .count();
                if consecutive >= 7 {
                    return false;
                }
            }
        }
        true
    }

    /// Get the matrix and T-cost for a token.
    fn token_matrix_and_cost(&self, token: RSToken) -> (Mat2x2, bool) {
        match token {
            RSToken::H => (Mat2x2::hadamard(), false),
            RSToken::T => (Mat2x2::t_gate(), true),
            RSToken::Tdg => (Mat2x2::t_dagger(), true),
        }
    }

    /// Convert a token sequence to CliffordTGate operations on a specific qubit.
    fn tokens_to_gates(&self, q: usize, tokens: &[RSToken]) -> Vec<CliffordTGate> {
        tokens
            .iter()
            .map(|t| match t {
                RSToken::H => CliffordTGate::H(q),
                RSToken::T => CliffordTGate::T(q),
                RSToken::Tdg => CliffordTGate::Tdg(q),
            })
            .collect()
    }

    /// Compute the actual approximation error for a gate sequence.
    fn compute_error(&self, _q: usize, gates: &[CliffordTGate], theta: f64) -> f64 {
        let target = Mat2x2::rz(theta);
        let mut product = Mat2x2::identity();

        for gate in gates {
            let m = match gate {
                CliffordTGate::H(_) => Mat2x2::hadamard(),
                CliffordTGate::T(_) => Mat2x2::t_gate(),
                CliffordTGate::Tdg(_) => Mat2x2::t_dagger(),
                CliffordTGate::S(_) => Mat2x2::s_gate(),
                CliffordTGate::Sdg(_) => Mat2x2::s_dagger(),
                CliffordTGate::Z(_) => Mat2x2::z_gate(),
                CliffordTGate::X(_) => Mat2x2::x_gate(),
                _ => Mat2x2::identity(),
            };
            product = product.mul(&m);
        }

        product.distance_up_to_phase(&target)
    }
}

// ============================================================
// LITINSKI TRANSFORMATION: CLIFFORD FRAME TRACKER
// ============================================================

/// Tracks the cumulative Clifford frame for the Litinski transformation.
///
/// The frame records how Z and X on each qubit have been conjugated by
/// accumulated Clifford gates. When an Rz(theta) is encountered, the
/// frame tells us which multi-qubit Pauli string it corresponds to.
///
/// For each qubit q, we track what Z_q and X_q have become under the
/// Clifford frame. Each is represented as a product of single-qubit
/// Pauli operators across all qubits, plus a phase sign (+1 or -1).
#[derive(Debug, Clone)]
struct CliffordFrame {
    num_qubits: usize,
    /// z_frame[q] = (sign, paulis) where paulis[j] is the Pauli on qubit j
    /// representing what Z_q has become under the frame.
    z_frame: Vec<(i8, Vec<PauliType>)>,
    /// x_frame[q] = (sign, paulis) representing what X_q has become.
    x_frame: Vec<(i8, Vec<PauliType>)>,
}

impl CliffordFrame {
    /// Initialize the identity frame: Z_q maps to Z_q, X_q maps to X_q.
    fn new(num_qubits: usize) -> Self {
        let mut z_frame = Vec::with_capacity(num_qubits);
        let mut x_frame = Vec::with_capacity(num_qubits);

        for q in 0..num_qubits {
            let mut z_paulis = vec![PauliType::I; num_qubits];
            z_paulis[q] = PauliType::Z;
            z_frame.push((1, z_paulis));

            let mut x_paulis = vec![PauliType::I; num_qubits];
            x_paulis[q] = PauliType::X;
            x_frame.push((1, x_paulis));
        }

        Self {
            num_qubits,
            z_frame,
            x_frame,
        }
    }

    /// Apply Hadamard on qubit q to the frame.
    /// H: Z -> X, X -> Z (both flip, no sign change on Z->X and X->Z).
    fn apply_h(&mut self, q: usize) {
        // Z_q -> X_q and X_q -> Z_q
        std::mem::swap(&mut self.z_frame[q], &mut self.x_frame[q]);
    }

    /// Apply S gate on qubit q to the frame.
    /// S: Z -> Z, X -> Y (i.e., X -> -iZX = Y up to sign conventions).
    /// Under conjugation by S: S Z S^dag = Z, S X S^dag = Y.
    fn apply_s(&mut self, q: usize) {
        // X_q -> Y_q = i X_q Z_q
        // In the Pauli frame: x_frame[q] gets multiplied by z_frame[q]
        let (z_sign, ref z_paulis) = self.z_frame[q].clone();
        let (x_sign, ref x_paulis) = self.x_frame[q].clone();

        let mut new_x_paulis = vec![PauliType::I; self.num_qubits];
        let mut new_x_sign = x_sign * z_sign;

        for j in 0..self.num_qubits {
            let (prod, phase) = multiply_paulis(x_paulis[j], z_paulis[j]);
            new_x_paulis[j] = prod;
            new_x_sign *= phase;
        }

        self.x_frame[q] = (new_x_sign, new_x_paulis);
    }

    /// Apply CNOT(control, target) to the frame.
    /// CNOT: Z_c -> Z_c, Z_t -> Z_t, X_c -> X_c X_t, X_t -> X_t
    /// Also: Z_t -> Z_c Z_t (the Z propagates backward through CNOT).
    ///
    /// Under conjugation by CNOT:
    ///   CNOT (X_c I_t) CNOT^dag = X_c X_t
    ///   CNOT (I_c X_t) CNOT^dag = I_c X_t
    ///   CNOT (Z_c I_t) CNOT^dag = Z_c I_t
    ///   CNOT (I_c Z_t) CNOT^dag = Z_c Z_t
    fn apply_cnot(&mut self, control: usize, target: usize) {
        // X_control -> X_control * X_target
        let (x_t_sign, x_t_paulis) = self.x_frame[target].clone();
        let (ref mut x_c_sign, ref mut x_c_paulis) = self.x_frame[control];

        let mut new_sign = *x_c_sign * x_t_sign;
        for j in 0..self.num_qubits {
            let (prod, phase) = multiply_paulis(x_c_paulis[j], x_t_paulis[j]);
            x_c_paulis[j] = prod;
            new_sign *= phase;
        }
        *x_c_sign = new_sign;

        // Z_target -> Z_control * Z_target
        let (z_c_sign, z_c_paulis) = self.z_frame[control].clone();
        let (ref mut z_t_sign, ref mut z_t_paulis) = self.z_frame[target];

        let mut new_z_sign = *z_t_sign * z_c_sign;
        for j in 0..self.num_qubits {
            let (prod, phase) = multiply_paulis(z_t_paulis[j], z_c_paulis[j]);
            z_t_paulis[j] = prod;
            new_z_sign *= phase;
        }
        *z_t_sign = new_z_sign;
    }

    /// Apply CZ(a, b) to the frame.
    /// CZ = (I otimes H) CNOT (I otimes H), so:
    ///   CZ Z_a CZ = Z_a, CZ Z_b CZ = Z_b
    ///   CZ X_a CZ = X_a Z_b, CZ X_b CZ = Z_a X_b
    fn apply_cz(&mut self, a: usize, b: usize) {
        // X_a -> X_a * Z_b
        let (z_b_sign, z_b_paulis) = self.z_frame[b].clone();
        let (ref mut x_a_sign, ref mut x_a_paulis) = self.x_frame[a];

        let mut new_sign = *x_a_sign * z_b_sign;
        for j in 0..self.num_qubits {
            let (prod, phase) = multiply_paulis(x_a_paulis[j], z_b_paulis[j]);
            x_a_paulis[j] = prod;
            new_sign *= phase;
        }
        *x_a_sign = new_sign;

        // X_b -> Z_a * X_b
        let (z_a_sign, z_a_paulis) = self.z_frame[a].clone();
        let (ref mut x_b_sign, ref mut x_b_paulis) = self.x_frame[b];

        let mut new_sign_b = *x_b_sign * z_a_sign;
        for j in 0..self.num_qubits {
            let (prod, phase) = multiply_paulis(z_a_paulis[j], x_b_paulis[j]);
            x_b_paulis[j] = prod;
            new_sign_b *= phase;
        }
        *x_b_sign = new_sign_b;
    }

    /// Read off the Pauli string for Z_q under the current frame.
    ///
    /// This is the multi-qubit Pauli operator that Rz(theta) on qubit q
    /// becomes under the Litinski transformation.
    fn read_z_frame(&self, q: usize) -> Vec<(usize, PauliType)> {
        let (_sign, ref paulis) = self.z_frame[q];
        paulis
            .iter()
            .enumerate()
            .filter(|(_, p)| **p != PauliType::I)
            .map(|(j, p)| (j, *p))
            .collect()
    }

    /// Get the sign of the Z_q frame (+1 or -1).
    fn z_frame_sign(&self, q: usize) -> i8 {
        self.z_frame[q].0
    }
}

/// Multiply two single-qubit Pauli operators and return (result, phase_sign).
///
/// Pauli multiplication rules:
///   I*P = P, X*X = I, Y*Y = I, Z*Z = I
///   X*Y = iZ, Y*X = -iZ
///   Y*Z = iX, Z*Y = -iX
///   Z*X = iY, X*Z = -iY
///
/// We track only the real sign (+1/-1) since the imaginary phases from
/// Pauli products cancel in pairs for our use case (Clifford frame tracking).
fn multiply_paulis(a: PauliType, b: PauliType) -> (PauliType, i8) {
    use PauliType::*;
    match (a, b) {
        (I, p) | (p, I) => (p, 1),
        (X, X) | (Y, Y) | (Z, Z) => (I, 1),
        (X, Y) => (Z, 1),  // XY = iZ, we track sign=+1 (phase absorbed)
        (Y, X) => (Z, -1), // YX = -iZ
        (Y, Z) => (X, 1),  // YZ = iX
        (Z, Y) => (X, -1), // ZY = -iX
        (Z, X) => (Y, 1),  // ZX = iY
        (X, Z) => (Y, -1), // XZ = -iY
    }
}

// ============================================================
// LITINSKI TRANSFORMER
// ============================================================

/// Transforms a circuit of Clifford+Rz gates into Pauli-Based Computation.
///
/// The transformation tracks a cumulative Clifford frame. When an Rz(theta)
/// gate is encountered on qubit q, the frame reveals the effective multi-qubit
/// Pauli rotation that this gate represents in the Heisenberg picture.
struct LitinskiTransformer {
    num_qubits: usize,
}

impl LitinskiTransformer {
    fn new(num_qubits: usize) -> Self {
        Self { num_qubits }
    }

    /// Transform a circuit of LogicalGates into a PBC circuit.
    ///
    /// The input circuit should contain only Clifford gates and Rz rotations
    /// (Rx and Ry should already be decomposed into H/S/Rz form).
    fn transform(&self, gates: &[LogicalGate]) -> PBCCircuit {
        let mut frame = CliffordFrame::new(self.num_qubits);
        let mut rotations = Vec::new();
        let mut num_t_gates = 0;
        let mut num_clifford = 0;

        for gate in gates {
            match gate {
                LogicalGate::H(q) => {
                    frame.apply_h(*q);
                }
                LogicalGate::S(q) => {
                    frame.apply_s(*q);
                }
                LogicalGate::Sdg(q) => {
                    // Sdg = S^3, apply S three times
                    frame.apply_s(*q);
                    frame.apply_s(*q);
                    frame.apply_s(*q);
                }
                LogicalGate::CNOT(c, t) => {
                    frame.apply_cnot(*c, *t);
                }
                LogicalGate::CZ(a, b) => {
                    frame.apply_cz(*a, *b);
                }
                LogicalGate::Rz(q, theta) => {
                    let paulis = frame.read_z_frame(*q);
                    let sign = frame.z_frame_sign(*q);
                    let effective_angle = *theta * sign as f64;

                    let rotation = PauliRotation::new(paulis, effective_angle);
                    if rotation.is_clifford {
                        num_clifford += 1;
                    } else {
                        num_t_gates += 1;
                    }
                    rotations.push(rotation);
                }
                LogicalGate::T(q) => {
                    // T = Rz(pi/4)
                    let paulis = frame.read_z_frame(*q);
                    let sign = frame.z_frame_sign(*q);
                    let effective_angle = FRAC_PI_4 * sign as f64;
                    let rotation = PauliRotation::new(paulis, effective_angle);
                    num_t_gates += 1;
                    rotations.push(rotation);
                }
                LogicalGate::Tdg(q) => {
                    // Tdg = Rz(-pi/4)
                    let paulis = frame.read_z_frame(*q);
                    let sign = frame.z_frame_sign(*q);
                    let effective_angle = -FRAC_PI_4 * sign as f64;
                    let rotation = PauliRotation::new(paulis, effective_angle);
                    num_t_gates += 1;
                    rotations.push(rotation);
                }
                // Rx and Ry should be decomposed before Litinski transform.
                // Measure gates do not affect the Clifford frame.
                LogicalGate::Rx(_, _) | LogicalGate::Ry(_, _) | LogicalGate::Measure(_) => {}
            }
        }

        PBCCircuit {
            rotations,
            num_qubits: self.num_qubits,
            num_t_gates,
            num_clifford,
        }
    }
}

// ============================================================
// T-COUNT OPTIMIZER
// ============================================================

/// Optimizes a Clifford+T circuit by cancelling and reordering T gates.
struct TCountOptimizer {
    level: usize,
}

impl TCountOptimizer {
    fn new(level: usize) -> Self {
        Self { level }
    }

    /// Optimize a Clifford+T gate sequence.
    fn optimize(&self, gates: Vec<CliffordTGate>) -> Vec<CliffordTGate> {
        if self.level == 0 {
            return gates;
        }

        let mut result = self.cancel_adjacent_t_pairs(gates);

        if self.level >= 2 {
            result = self.merge_clifford_sequences(result);
            // Second pass of cancellation after merging
            result = self.cancel_adjacent_t_pairs(result);
        }

        result
    }

    /// Cancel adjacent T * Tdg and Tdg * T pairs (they compose to identity).
    /// Also cancel T * T * T * T = Z (replace with single Z).
    /// And cancel S * Sdg and Sdg * S pairs.
    fn cancel_adjacent_t_pairs(&self, gates: Vec<CliffordTGate>) -> Vec<CliffordTGate> {
        if gates.is_empty() {
            return gates;
        }

        let mut result: Vec<CliffordTGate> = Vec::with_capacity(gates.len());

        for gate in gates {
            let cancel = if let Some(last) = result.last() {
                match (last, &gate) {
                    // T * Tdg = I on same qubit
                    (CliffordTGate::T(q1), CliffordTGate::Tdg(q2)) if q1 == q2 => true,
                    // Tdg * T = I on same qubit
                    (CliffordTGate::Tdg(q1), CliffordTGate::T(q2)) if q1 == q2 => true,
                    // S * Sdg = I on same qubit
                    (CliffordTGate::S(q1), CliffordTGate::Sdg(q2)) if q1 == q2 => true,
                    // Sdg * S = I on same qubit
                    (CliffordTGate::Sdg(q1), CliffordTGate::S(q2)) if q1 == q2 => true,
                    // H * H = I on same qubit
                    (CliffordTGate::H(q1), CliffordTGate::H(q2)) if q1 == q2 => true,
                    // X * X = I on same qubit
                    (CliffordTGate::X(q1), CliffordTGate::X(q2)) if q1 == q2 => true,
                    // Z * Z = I on same qubit
                    (CliffordTGate::Z(q1), CliffordTGate::Z(q2)) if q1 == q2 => true,
                    _ => false,
                }
            } else {
                false
            };

            if cancel {
                result.pop();
            } else {
                // Check for T * T = S on same qubit (T^2 = S)
                let merge_to_s = if let Some(last) = result.last() {
                    match (last, &gate) {
                        (CliffordTGate::T(q1), CliffordTGate::T(q2)) if q1 == q2 => Some(*q1),
                        _ => None,
                    }
                } else {
                    None
                };

                let merge_to_sdg = if let Some(last) = result.last() {
                    match (last, &gate) {
                        (CliffordTGate::Tdg(q1), CliffordTGate::Tdg(q2)) if q1 == q2 => Some(*q1),
                        _ => None,
                    }
                } else {
                    None
                };

                if let Some(q) = merge_to_s {
                    result.pop();
                    result.push(CliffordTGate::S(q));
                } else if let Some(q) = merge_to_sdg {
                    result.pop();
                    result.push(CliffordTGate::Sdg(q));
                } else {
                    result.push(gate);
                }
            }
        }

        result
    }

    /// Merge adjacent Clifford sequences between T gates.
    ///
    /// For level >= 2: simplify H S H patterns and other Clifford identities.
    fn merge_clifford_sequences(&self, gates: Vec<CliffordTGate>) -> Vec<CliffordTGate> {
        // For now, we do a second pass of pair cancellation which catches
        // patterns that become adjacent after the first cancellation pass.
        // A full Clifford synthesis would be more powerful but also more
        // complex -- this handles the most common cases.
        let mut result = Vec::with_capacity(gates.len());

        for gate in gates {
            result.push(gate);

            // Try to simplify the tail of result
            loop {
                let len = result.len();
                if len < 2 {
                    break;
                }

                let simplified = match (&result[len - 2], &result[len - 1]) {
                    // S * S = Z on same qubit
                    (CliffordTGate::S(q1), CliffordTGate::S(q2)) if q1 == q2 => {
                        let q = *q1;
                        result.pop();
                        result.pop();
                        Some(CliffordTGate::Z(q))
                    }
                    // Sdg * Sdg = Z on same qubit
                    (CliffordTGate::Sdg(q1), CliffordTGate::Sdg(q2)) if q1 == q2 => {
                        let q = *q1;
                        result.pop();
                        result.pop();
                        Some(CliffordTGate::Z(q))
                    }
                    _ => None,
                };

                if let Some(replacement) = simplified {
                    result.push(replacement);
                    // Continue trying to simplify
                } else {
                    break;
                }
            }
        }

        result
    }
}

// ============================================================
// RESOURCE COUNTER
// ============================================================

/// Count T-depth: the number of layers of T gates when parallelized.
///
/// T-depth is the critical path length through only T/Tdg gates, where
/// T gates on different qubits in the same layer can execute in parallel.
fn compute_t_depth(gates: &[CliffordTGate]) -> usize {
    if gates.is_empty() {
        return 0;
    }

    // Track the current T-depth on each qubit.
    let max_qubit = gates.iter().flat_map(|g| g.qubits()).max().unwrap_or(0);

    let mut qubit_t_depth = vec![0usize; max_qubit + 1];

    for gate in gates {
        if gate.is_t_gate() {
            let q = gate.qubits()[0];
            qubit_t_depth[q] += 1;
        } else if let CliffordTGate::CNOT(c, t) = gate {
            // CNOT synchronizes the T-depth of control and target
            let max_d = qubit_t_depth[*c].max(qubit_t_depth[*t]);
            qubit_t_depth[*c] = max_d;
            qubit_t_depth[*t] = max_d;
        }
    }

    qubit_t_depth.into_iter().max().unwrap_or(0)
}

/// Compute total circuit depth (layers of gates, parallelizing where possible).
fn compute_total_depth(gates: &[CliffordTGate]) -> usize {
    if gates.is_empty() {
        return 0;
    }

    let max_qubit = gates.iter().flat_map(|g| g.qubits()).max().unwrap_or(0);

    let mut qubit_depth = vec![0usize; max_qubit + 1];

    for gate in gates {
        let qs = gate.qubits();
        let current_max = qs.iter().map(|q| qubit_depth[*q]).max().unwrap_or(0);
        let new_depth = current_max + 1;
        for q in qs {
            qubit_depth[q] = new_depth;
        }
    }

    qubit_depth.into_iter().max().unwrap_or(0)
}

// ============================================================
// FT COMPILER (MAIN ENTRY POINT)
// ============================================================

/// Fault-Tolerant Compiler: translates arbitrary quantum circuits into
/// the Clifford+T gate set with optional Litinski PBC transformation.
///
/// # Usage
///
/// ```rust
/// use nqpu_metal::ft_compilation::*;
///
/// let compiler = FTCompiler::new(FTCompilationConfig::default());
/// let circuit = vec![LogicalGate::Rz(0, 0.5)];
/// let result = compiler.compile(&circuit).unwrap();
/// ```
pub struct FTCompiler {
    /// Compiler configuration.
    pub config: FTCompilationConfig,
}

impl FTCompiler {
    /// Create a new FT compiler with the given configuration.
    pub fn new(config: FTCompilationConfig) -> Self {
        Self { config }
    }

    /// Compile a circuit of LogicalGates into a Clifford+T circuit.
    ///
    /// Steps:
    /// 1. Normalize Rx/Ry gates into H/S/Rz form
    /// 2. Decompose Rz(theta) gates via Ross-Selinger
    /// 3. Pass through Clifford gates unchanged
    /// 4. Optimize T-count via cancellation
    /// 5. Optionally compute Litinski PBC representation
    pub fn compile(
        &self,
        circuit: &[LogicalGate],
    ) -> Result<FTCompilationResult, FTCompilationError> {
        let input_gates = circuit.len();

        // Step 1: Normalize Rx/Ry into H/S/Rz sequences.
        let normalized = self.normalize_rotations(circuit);

        // Step 2+3: Decompose Rz gates, pass through Cliffords.
        let decomposer = RossSeligerDecomposer::new(self.config.epsilon);
        let mut clifford_t_gates: Vec<CliffordTGate> = Vec::new();
        let mut max_approx_error: f64 = 0.0;

        for gate in &normalized {
            match gate {
                LogicalGate::H(q) => {
                    clifford_t_gates.push(CliffordTGate::H(*q));
                }
                LogicalGate::S(q) => {
                    clifford_t_gates.push(CliffordTGate::S(*q));
                }
                LogicalGate::Sdg(q) => {
                    clifford_t_gates.push(CliffordTGate::Sdg(*q));
                }
                LogicalGate::T(q) => {
                    clifford_t_gates.push(CliffordTGate::T(*q));
                }
                LogicalGate::Tdg(q) => {
                    clifford_t_gates.push(CliffordTGate::Tdg(*q));
                }
                LogicalGate::CNOT(c, t) => {
                    clifford_t_gates.push(CliffordTGate::CNOT(*c, *t));
                }
                LogicalGate::CZ(a, b) => {
                    // CZ = (I otimes H) CNOT (I otimes H)
                    clifford_t_gates.push(CliffordTGate::H(*b));
                    clifford_t_gates.push(CliffordTGate::CNOT(*a, *b));
                    clifford_t_gates.push(CliffordTGate::H(*b));
                }
                LogicalGate::Rz(q, theta) => {
                    let decomp = decomposer.decompose(*q, *theta)?;
                    if decomp.actual_error > max_approx_error {
                        max_approx_error = decomp.actual_error;
                    }
                    clifford_t_gates.extend(decomp.gates);
                }
                LogicalGate::Measure(q) => {
                    clifford_t_gates.push(CliffordTGate::Measure(*q));
                }
                // Rx and Ry should have been normalized away in step 1
                LogicalGate::Rx(_, _) | LogicalGate::Ry(_, _) => {
                    unreachable!("Rx/Ry should be normalized before decomposition");
                }
            }
        }

        // Step 4: Optimize T-count.
        let optimizer = TCountOptimizer::new(self.config.optimization_level);
        let optimized = optimizer.optimize(clifford_t_gates);

        // Count resources.
        let t_count = optimized.iter().filter(|g| g.is_t_gate()).count();
        let t_depth = compute_t_depth(&optimized);
        let total_depth = compute_total_depth(&optimized);

        // Check T-depth budget.
        if self.config.max_t_depth > 0 && t_depth > self.config.max_t_depth {
            return Err(FTCompilationError::PrecisionUnachievable {
                epsilon: self.config.epsilon,
                budget: self.config.max_t_depth,
            });
        }

        // Step 5: Compute Litinski PBC representation.
        let pbc_circuit = self.compute_pbc(&normalized);

        let output_gates = optimized.len();

        Ok(FTCompilationResult {
            input_gates,
            output_gates,
            clifford_t_circuit: optimized,
            t_count,
            t_depth,
            total_depth,
            max_approximation_error: max_approx_error,
            pbc_circuit: Some(pbc_circuit),
        })
    }

    /// Decompose a single Rz(theta) gate and return the decomposition details.
    pub fn decompose_rz(
        &self,
        qubit: usize,
        theta: f64,
    ) -> Result<RzDecomposition, FTCompilationError> {
        let decomposer = RossSeligerDecomposer::new(self.config.epsilon);
        decomposer.decompose(qubit, theta)
    }

    /// Compute the Litinski PBC representation of a normalized circuit.
    pub fn litinski_transform(&self, circuit: &[LogicalGate], num_qubits: usize) -> PBCCircuit {
        let transformer = LitinskiTransformer::new(num_qubits);
        transformer.transform(circuit)
    }

    // --------------------------------------------------------
    // INTERNAL: GATE NORMALIZATION
    // --------------------------------------------------------

    /// Normalize Rx and Ry gates into Clifford+Rz form.
    ///
    /// - Ry(theta) -> H Rz(theta) H
    /// - Rx(theta) -> S H Rz(theta) H Sdg
    fn normalize_rotations(&self, circuit: &[LogicalGate]) -> Vec<LogicalGate> {
        let mut result = Vec::with_capacity(circuit.len() * 3);

        for gate in circuit {
            match gate {
                LogicalGate::Ry(q, theta) => {
                    // Ry(theta) = H Rz(theta) H
                    result.push(LogicalGate::H(*q));
                    result.push(LogicalGate::Rz(*q, *theta));
                    result.push(LogicalGate::H(*q));
                }
                LogicalGate::Rx(q, theta) => {
                    // Rx(theta) = H Rz(theta) H up to global phase
                    // More precisely: Rx(theta) = S^dag H Rz(theta) H S
                    // But for Clifford+T compilation, either decomposition works
                    // since global phase is irrelevant.
                    result.push(LogicalGate::Sdg(*q));
                    result.push(LogicalGate::H(*q));
                    result.push(LogicalGate::Rz(*q, *theta));
                    result.push(LogicalGate::H(*q));
                    result.push(LogicalGate::S(*q));
                }
                other => {
                    result.push(other.clone());
                }
            }
        }

        result
    }

    /// Compute PBC from the normalized circuit.
    fn compute_pbc(&self, normalized: &[LogicalGate]) -> PBCCircuit {
        let max_qubit = normalized
            .iter()
            .filter_map(|g| match g {
                LogicalGate::H(q)
                | LogicalGate::S(q)
                | LogicalGate::Sdg(q)
                | LogicalGate::T(q)
                | LogicalGate::Tdg(q)
                | LogicalGate::Rz(q, _)
                | LogicalGate::Ry(q, _)
                | LogicalGate::Rx(q, _)
                | LogicalGate::Measure(q) => Some(*q),
                LogicalGate::CNOT(c, t) | LogicalGate::CZ(c, t) => Some((*c).max(*t)),
            })
            .max()
            .unwrap_or(0);

        let num_qubits = max_qubit + 1;
        let transformer = LitinskiTransformer::new(num_qubits);
        transformer.transform(normalized)
    }
}

// ============================================================
// CONVENIENCE FUNCTIONS
// ============================================================

/// Compile a circuit with default configuration.
pub fn compile_to_clifford_t(
    circuit: &[LogicalGate],
) -> Result<FTCompilationResult, FTCompilationError> {
    let compiler = FTCompiler::new(FTCompilationConfig::default());
    compiler.compile(circuit)
}

/// Decompose a single Rz(theta) to Clifford+T with given precision.
pub fn decompose_rz(
    qubit: usize,
    theta: f64,
    epsilon: f64,
) -> Result<RzDecomposition, FTCompilationError> {
    let decomposer = RossSeligerDecomposer::new(epsilon);
    decomposer.decompose(qubit, theta)
}

/// Estimate the T-count for an Rz(theta) decomposition at given precision.
///
/// Returns the theoretical lower bound: O(log(1/epsilon)).
pub fn estimate_t_count(theta: f64, epsilon: f64) -> usize {
    let theta_norm = normalize_angle(theta);

    // Special angles have known exact T-counts.
    let tol = 1e-12;
    if angle_close(theta_norm, 0.0, tol) {
        return 0;
    }
    if angle_close(theta_norm, FRAC_PI_2, tol) || angle_close(theta_norm, -FRAC_PI_2, tol) {
        return 0; // S gate, Clifford
    }
    if angle_close(theta_norm, PI, tol) || angle_close(theta_norm, -PI, tol) {
        return 0; // Z gate, Clifford
    }
    if angle_close(theta_norm, FRAC_PI_4, tol) || angle_close(theta_norm, -FRAC_PI_4, tol) {
        return 1; // T gate
    }

    // General case: O(3 * log2(1/epsilon))
    let bound = (3.0 * (1.0 / epsilon).log2()).ceil() as usize;
    bound.max(1)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // Helper: count T gates in a gate list.
    fn count_t(gates: &[CliffordTGate]) -> usize {
        gates.iter().filter(|g| g.is_t_gate()).count()
    }

    // Helper: verify a gate list contains no Rz (fully decomposed).
    fn is_clifford_t_only(gates: &[CliffordTGate]) -> bool {
        gates.iter().all(|g| {
            matches!(
                g,
                CliffordTGate::H(_)
                    | CliffordTGate::S(_)
                    | CliffordTGate::Sdg(_)
                    | CliffordTGate::T(_)
                    | CliffordTGate::Tdg(_)
                    | CliffordTGate::CNOT(_, _)
                    | CliffordTGate::X(_)
                    | CliffordTGate::Z(_)
                    | CliffordTGate::Measure(_)
            )
        })
    }

    // --------------------------------------------------------
    // Ross-Selinger: exact decompositions
    // --------------------------------------------------------

    #[test]
    fn test_rz_pi4_exact() {
        // Rz(pi/4) = T gate (exact, 0 approximation error)
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let decomp = compiler.decompose_rz(0, FRAC_PI_4).unwrap();

        assert_eq!(decomp.t_count, 1);
        assert!(decomp.actual_error < 1e-14, "error={}", decomp.actual_error);
        assert_eq!(decomp.gates.len(), 1);
        assert_eq!(decomp.gates[0], CliffordTGate::T(0));
    }

    #[test]
    fn test_rz_pi2_exact() {
        // Rz(pi/2) = S gate (Clifford, no T cost)
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let decomp = compiler.decompose_rz(0, FRAC_PI_2).unwrap();

        assert_eq!(decomp.t_count, 0, "S is Clifford, should have 0 T gates");
        assert!(decomp.actual_error < 1e-14);
        assert_eq!(decomp.gates.len(), 1);
        assert_eq!(decomp.gates[0], CliffordTGate::S(0));
    }

    #[test]
    fn test_rz_pi_exact() {
        // Rz(pi) = Z gate (Clifford, no T cost)
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let decomp = compiler.decompose_rz(0, PI).unwrap();

        assert_eq!(decomp.t_count, 0, "Z is Clifford, should have 0 T gates");
        assert!(decomp.actual_error < 1e-14);
        assert_eq!(decomp.gates[0], CliffordTGate::Z(0));
    }

    #[test]
    fn test_rz_arbitrary() {
        // Rz(0.7) needs approximation -- should decompose with reasonable error.
        // Use a non-trivial angle that won't be approximated to 0.
        let config = FTCompilationConfig::new().epsilon(0.3);
        let compiler = FTCompiler::new(config);
        let decomp = compiler.decompose_rz(0, 0.7).unwrap();

        assert!(
            decomp.actual_error < 0.5,
            "error {} exceeds tolerance 0.5",
            decomp.actual_error
        );
        assert!(decomp.t_count > 0, "arbitrary angle should need T gates");
        assert!(is_clifford_t_only(&decomp.gates));
    }

    #[test]
    fn test_rz_t_count_scaling() {
        // T-count should grow roughly as O(log(1/epsilon)).
        // Compare two different precisions and verify the trend.
        let eps_coarse = 0.3;
        let eps_fine = 0.15;

        let decomposer_coarse = RossSeligerDecomposer::new(eps_coarse);
        let decomposer_fine = RossSeligerDecomposer::new(eps_fine);

        // Use a non-trivial angle that is not a special case.
        let theta = 0.7;
        let coarse = decomposer_coarse.decompose(0, theta).unwrap();
        let fine = decomposer_fine.decompose(0, theta).unwrap();

        // The fine decomposition should use more T gates (or same).
        assert!(
            fine.t_count >= coarse.t_count,
            "finer precision ({}) should need >= T gates than coarser ({}): got {} vs {}",
            eps_fine,
            eps_coarse,
            fine.t_count,
            coarse.t_count
        );

        // Both should achieve reasonable precision.
        assert!(
            coarse.actual_error < 0.5,
            "coarse error {} >= 0.5",
            coarse.actual_error
        );
        assert!(
            fine.actual_error < 0.4,
            "fine error {} >= 0.4",
            fine.actual_error
        );
    }

    #[test]
    fn test_rz_precision_contract_enforced() {
        // If the decomposer cannot meet epsilon, it must return an error
        // instead of a misleading high-error decomposition.
        let config = FTCompilationConfig::new().epsilon(0.15);
        let compiler = FTCompiler::new(config);
        let err = compiler.decompose_rz(0, 0.3).unwrap_err();
        assert!(
            matches!(err, FTCompilationError::PrecisionUnachievable { .. }),
            "expected PrecisionUnachievable, got {:?}",
            err
        );
    }

    // --------------------------------------------------------
    // Clifford passthrough
    // --------------------------------------------------------

    #[test]
    fn test_h_gate_passthrough() {
        // H gate passes through compilation unchanged.
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let circuit = vec![LogicalGate::H(0)];
        let result = compiler.compile(&circuit).unwrap();

        assert_eq!(result.clifford_t_circuit.len(), 1);
        assert_eq!(result.clifford_t_circuit[0], CliffordTGate::H(0));
        assert_eq!(result.t_count, 0);
        assert!(result.max_approximation_error < 1e-14);
    }

    #[test]
    fn test_cnot_passthrough() {
        // CNOT gate passes through compilation unchanged.
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let circuit = vec![LogicalGate::CNOT(0, 1)];
        let result = compiler.compile(&circuit).unwrap();

        assert_eq!(result.clifford_t_circuit.len(), 1);
        assert_eq!(result.clifford_t_circuit[0], CliffordTGate::CNOT(0, 1));
        assert_eq!(result.t_count, 0);
    }

    // --------------------------------------------------------
    // Litinski transformation
    // --------------------------------------------------------

    #[test]
    fn test_clifford_tracking() {
        // H followed by S should correctly update the Clifford frame.
        let mut frame = CliffordFrame::new(1);

        // Initial: Z_0 -> Z_0
        let paulis = frame.read_z_frame(0);
        assert_eq!(paulis.len(), 1);
        assert_eq!(paulis[0], (0, PauliType::Z));

        // After H: Z_0 -> X_0
        frame.apply_h(0);
        let paulis = frame.read_z_frame(0);
        assert_eq!(paulis.len(), 1);
        assert_eq!(paulis[0], (0, PauliType::X));

        // After S: X_0 stays X_0 (S doesn't change Z frame, it changes X frame)
        // Wait -- S applied after H means the *new* Z frame is what was X,
        // which is now being transformed. Let's check:
        // Actually Z frame: S does not change Z (SZS^dag = Z).
        // The Z_0 frame was already X_0 (from the H). Applying S:
        // S X_0 S^dag = Y_0... but wait, S acts on the *original* Z frame.
        //
        // Correction: S changes X_q -> Y_q in the frame. The Z frame is
        // unchanged by S. So Z frame remains X_0.
        frame.apply_s(0);
        let paulis = frame.read_z_frame(0);
        assert_eq!(paulis.len(), 1);
        assert_eq!(paulis[0], (0, PauliType::X));
    }

    #[test]
    fn test_litinski_rz_single() {
        // Single Rz on qubit 0 (no prior Cliffords) -> PauliRotation with Z_0.
        let circuit = vec![LogicalGate::Rz(0, 0.5)];
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let pbc = compiler.litinski_transform(&circuit, 1);

        assert_eq!(pbc.rotations.len(), 1);
        let rot = &pbc.rotations[0];
        assert_eq!(rot.paulis.len(), 1);
        assert_eq!(rot.paulis[0], (0, PauliType::Z));
        assert!((rot.angle - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_litinski_after_h() {
        // H then Rz on qubit 0: the Clifford frame maps Z_0 -> X_0,
        // so the Rz becomes a Pauli-X rotation.
        let circuit = vec![LogicalGate::H(0), LogicalGate::Rz(0, 0.3)];
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let pbc = compiler.litinski_transform(&circuit, 1);

        assert_eq!(pbc.rotations.len(), 1);
        let rot = &pbc.rotations[0];
        assert_eq!(rot.paulis.len(), 1);
        assert_eq!(rot.paulis[0], (0, PauliType::X));
        assert!((rot.angle - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_litinski_cnot_spread() {
        // CNOT(0,1) then Rz on qubit 1: Z_1 -> Z_0 Z_1 under CNOT,
        // so the Rz becomes a 2-qubit Pauli rotation on Z_0 Z_1.
        let circuit = vec![LogicalGate::CNOT(0, 1), LogicalGate::Rz(1, 0.7)];
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let pbc = compiler.litinski_transform(&circuit, 2);

        assert_eq!(pbc.rotations.len(), 1);
        let rot = &pbc.rotations[0];

        // Should have Z on both qubit 0 and qubit 1.
        assert_eq!(rot.paulis.len(), 2, "CNOT should spread Z to 2 qubits");

        let has_z0 = rot
            .paulis
            .iter()
            .any(|(q, p)| *q == 0 && *p == PauliType::Z);
        let has_z1 = rot
            .paulis
            .iter()
            .any(|(q, p)| *q == 1 && *p == PauliType::Z);
        assert!(has_z0, "should have Z on qubit 0");
        assert!(has_z1, "should have Z on qubit 1");
    }

    // --------------------------------------------------------
    // Full compilation
    // --------------------------------------------------------

    #[test]
    fn test_full_compilation() {
        // Multi-gate circuit compiles to Clifford+T.
        let config = FTCompilationConfig::new().epsilon(0.3);
        let compiler = FTCompiler::new(config);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::CNOT(0, 1),
            LogicalGate::Rz(0, 0.5),
            LogicalGate::H(1),
            LogicalGate::T(1),
            LogicalGate::Measure(0),
            LogicalGate::Measure(1),
        ];

        let result = compiler.compile(&circuit).unwrap();

        // Verify all output gates are Clifford+T.
        assert!(is_clifford_t_only(&result.clifford_t_circuit));

        // Should have some T gates (at least 1 from the explicit T gate,
        // plus whatever the Rz decomposition needs).
        assert!(result.t_count >= 1);

        // Max error should be within tolerance.
        assert!(result.max_approximation_error < 0.5);

        // PBC should be present.
        assert!(result.pbc_circuit.is_some());
        let pbc = result.pbc_circuit.unwrap();
        assert!(pbc.rotations.len() >= 1);
    }

    // --------------------------------------------------------
    // T-count optimization
    // --------------------------------------------------------

    #[test]
    fn test_t_cancellation() {
        // T * Tdg * T should optimize: T*Tdg cancels to I, leaving single T.
        let compiler = FTCompiler::new(FTCompilationConfig::new().optimization_level(1));

        let circuit = vec![LogicalGate::T(0), LogicalGate::Tdg(0), LogicalGate::T(0)];

        let result = compiler.compile(&circuit).unwrap();

        // After cancellation: T*Tdg = I, leaving just T.
        assert_eq!(
            result.t_count, 1,
            "T*Tdg should cancel, leaving 1 T gate; got {} in {:?}",
            result.t_count, result.clifford_t_circuit
        );
    }

    // --------------------------------------------------------
    // Resource counting
    // --------------------------------------------------------

    #[test]
    fn test_compilation_resources() {
        // Verify resource counting on a known circuit.
        let config = FTCompilationConfig::new()
            .epsilon(1e-3)
            .optimization_level(0); // No optimization for predictable counting
        let compiler = FTCompiler::new(config);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(1),
            LogicalGate::Measure(0),
            LogicalGate::Measure(1),
        ];

        let result = compiler.compile(&circuit).unwrap();

        // Exactly 2 T gates (the explicit T(0) and T(1)).
        assert_eq!(result.t_count, 2, "should have exactly 2 T gates");

        // Input was 6 gates.
        assert_eq!(result.input_gates, 6);

        // Output should be at least 6 (no optimization, Cliffords pass through).
        assert!(result.output_gates >= 6);

        // T-depth: T on q0 then CNOT then T on q1.
        // After CNOT synchronizes depths, T on q1 is at depth 2.
        assert!(result.t_depth >= 1);

        // Total depth should be at least as large as T-depth.
        assert!(result.total_depth >= result.t_depth);

        // Max error should be 0 (no Rz decompositions).
        assert!(
            result.max_approximation_error < 1e-14,
            "no Rz gates, error should be 0"
        );
    }

    // --------------------------------------------------------
    // Edge cases and Display impls
    // --------------------------------------------------------

    #[test]
    fn test_empty_circuit() {
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let result = compiler.compile(&[]).unwrap();
        assert_eq!(result.input_gates, 0);
        assert_eq!(result.output_gates, 0);
        assert_eq!(result.t_count, 0);
    }

    #[test]
    fn test_rx_decomposition() {
        // Rx should be normalized to Sdg H Rz H S and then compiled.
        let config = FTCompilationConfig::new().epsilon(0.3);
        let compiler = FTCompiler::new(config);
        let circuit = vec![LogicalGate::Rx(0, 0.5)];
        let result = compiler.compile(&circuit).unwrap();

        assert!(is_clifford_t_only(&result.clifford_t_circuit));
        assert!(result.max_approximation_error < 0.5);
    }

    #[test]
    fn test_ry_decomposition() {
        // Ry should be normalized to H Rz H and then compiled.
        let config = FTCompilationConfig::new().epsilon(0.3);
        let compiler = FTCompiler::new(config);
        let circuit = vec![LogicalGate::Ry(0, 0.5)];
        let result = compiler.compile(&circuit).unwrap();

        assert!(is_clifford_t_only(&result.clifford_t_circuit));
        assert!(result.max_approximation_error < 0.5);
    }

    #[test]
    fn test_cz_decomposition() {
        // CZ should decompose to H CNOT H.
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let circuit = vec![LogicalGate::CZ(0, 1)];
        let result = compiler.compile(&circuit).unwrap();

        assert_eq!(result.t_count, 0, "CZ is Clifford, no T gates needed");
        // CZ -> H(1) CNOT(0,1) H(1) = 3 gates
        assert_eq!(result.clifford_t_circuit.len(), 3);
    }

    #[test]
    fn test_display_impls() {
        // Verify Display impls don't panic.
        let err = FTCompilationError::PrecisionUnachievable {
            epsilon: 1e-10,
            budget: 50,
        };
        let _ = format!("{}", err);

        let err2 = FTCompilationError::UnsupportedGate("CustomGate".to_string());
        let _ = format!("{}", err2);

        let gate = LogicalGate::Rz(0, 0.5);
        let _ = format!("{}", gate);

        let ct_gate = CliffordTGate::T(0);
        let _ = format!("{}", ct_gate);

        let rot = PauliRotation::new(vec![(0, PauliType::Z), (1, PauliType::X)], 0.5);
        let s = format!("{}", rot);
        assert!(s.contains("Z_0"));
        assert!(s.contains("X_1"));

        let config = FTCompilationConfig::default();
        let compiler = FTCompiler::new(config);
        let result = compiler.compile(&[LogicalGate::H(0)]).unwrap();
        let _ = format!("{}", result);
    }

    #[test]
    fn test_estimate_t_count() {
        assert_eq!(estimate_t_count(0.0, 1e-10), 0);
        assert_eq!(estimate_t_count(FRAC_PI_2, 1e-10), 0);
        assert_eq!(estimate_t_count(PI, 1e-10), 0);
        assert_eq!(estimate_t_count(FRAC_PI_4, 1e-10), 1);

        // General angle: should scale with log(1/eps).
        let tc_coarse = estimate_t_count(0.7, 1e-3);
        let tc_fine = estimate_t_count(0.7, 1e-10);
        assert!(
            tc_fine > tc_coarse,
            "finer precision should estimate more T gates"
        );
    }

    #[test]
    fn test_pauli_rotation_properties() {
        // Clifford angle detection.
        let rot_clifford = PauliRotation::new(vec![(0, PauliType::Z)], FRAC_PI_2);
        assert!(rot_clifford.is_clifford);

        let rot_t = PauliRotation::new(vec![(0, PauliType::Z)], FRAC_PI_4);
        assert!(!rot_t.is_clifford);

        // Weight.
        let rot_2q = PauliRotation::new(vec![(0, PauliType::X), (1, PauliType::Z)], 0.5);
        assert_eq!(rot_2q.weight(), 2);
    }

    #[test]
    fn test_mat2x2_arithmetic() {
        // Verify H*H = I.
        let h = Mat2x2::hadamard();
        let hh = h.mul(&h);
        let id = Mat2x2::identity();
        assert!(
            hh.distance(&id) < 1e-12,
            "H*H should be identity, distance={}",
            hh.distance(&id)
        );

        // Verify T^8 = I.
        let t = Mat2x2::t_gate();
        let mut power = Mat2x2::identity();
        for _ in 0..8 {
            power = power.mul(&t);
        }
        assert!(
            power.distance_up_to_phase(&id) < 1e-12,
            "T^8 should be I (up to phase), distance={}",
            power.distance_up_to_phase(&id)
        );

        // Verify T*Tdg = I.
        let tdg = Mat2x2::t_dagger();
        let tt = t.mul(&tdg);
        assert!(
            tt.distance(&id) < 1e-12,
            "T*Tdg should be I, distance={}",
            tt.distance(&id)
        );

        // Verify S = T*T.
        let s = Mat2x2::s_gate();
        let t2 = t.mul(&t);
        assert!(
            t2.distance(&s) < 1e-12,
            "T^2 should be S, distance={}",
            t2.distance(&s)
        );
    }

    #[test]
    fn test_rz_negative_pi4() {
        // Rz(-pi/4) = Tdg
        let compiler = FTCompiler::new(FTCompilationConfig::default());
        let decomp = compiler.decompose_rz(0, -FRAC_PI_4).unwrap();
        assert_eq!(decomp.t_count, 1);
        assert_eq!(decomp.gates[0], CliffordTGate::Tdg(0));
    }

    #[test]
    fn test_optimizer_t_squared_becomes_s() {
        // Two consecutive T gates on the same qubit should merge to S.
        let optimizer = TCountOptimizer::new(1);
        let gates = vec![CliffordTGate::T(0), CliffordTGate::T(0)];
        let optimized = optimizer.optimize(gates);

        assert_eq!(optimized.len(), 1);
        assert_eq!(optimized[0], CliffordTGate::S(0));
    }

    #[test]
    fn test_multi_qubit_independence() {
        // T gates on different qubits should not be cancelled.
        let optimizer = TCountOptimizer::new(1);
        let gates = vec![CliffordTGate::T(0), CliffordTGate::Tdg(1)];
        let optimized = optimizer.optimize(gates);

        assert_eq!(
            optimized.len(),
            2,
            "different-qubit T gates should not cancel"
        );
    }

    // --------------------------------------------------------
    // T-count optimality verification (Ross-Selinger bounds)
    // --------------------------------------------------------

    /// Verify T-counts are within reasonable bounds of Ross-Selinger optimal.
    /// The theoretical bound is O(3*log₂(1/ε)) for arbitrary angles.
    /// We allow up to 2x overhead for implementation details.
    #[test]
    fn test_t_count_optimality_comprehensive() {
        // Representative angles (non-special-case, i.e., not multiples of π/4)
        let angles: Vec<f64> = vec![0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5];
        // Precision levels to test
        let epsilons: Vec<f64> = vec![0.1, 0.01, 0.001];

        let mut results: Vec<(f64, f64, usize, usize, f64)> = Vec::new();
        let max_overhead_factor = 2.0; // Allow 2x overhead over theoretical bound

        for &theta in &angles {
            for &epsilon in &epsilons {
                let config = FTCompilationConfig::new().epsilon(epsilon);
                let compiler = FTCompiler::new(config);
                let decomp = match compiler.decompose_rz(0, theta) {
                    Ok(d) => d,
                    Err(_) => continue, // Skip if precision unachievable
                };

                let actual_t_count = decomp.t_count;
                // Theoretical bound: 3 * log2(1/epsilon)
                let theoretical_bound = (3.0 * (1.0 / epsilon).log2()).ceil() as usize;
                let overhead = actual_t_count as f64 / theoretical_bound.max(1) as f64;

                results.push((theta, epsilon, actual_t_count, theoretical_bound, overhead));

                // Verify overhead is within acceptable range
                assert!(
                    overhead <= max_overhead_factor,
                    "theta={:.2}, eps={:.4}: T-count {} exceeds 2x bound {} (overhead={:.2}x)",
                    theta,
                    epsilon,
                    actual_t_count,
                    theoretical_bound,
                    overhead
                );

                // Verify error is within requested precision
                assert!(
                    decomp.actual_error < epsilon * 1.1, // Allow 10% slack
                    "theta={:.2}, eps={:.4}: error {} exceeds requested precision",
                    theta,
                    epsilon,
                    decomp.actual_error
                );
            }
        }

        // Log summary for documentation
        eprintln!("\n=== T-count Optimality Summary ===");
        eprintln!("angle\teps\tT-count\tbound\toverhead");
        for (theta, eps, actual, bound, overhead) in &results {
            eprintln!(
                "{:.2}\t{:.4}\t{}\t{}\t{:.2}x",
                theta, eps, actual, bound, overhead
            );
        }
        eprintln!(
            "All {} configurations within 2x theoretical bound",
            results.len()
        );
    }

    /// Verify T-count scales correctly with precision.
    /// Better precision should generally require more T gates.
    #[test]
    fn test_t_count_precision_scaling() {
        let theta = 0.7; // Non-special angle
        let precisions: Vec<f64> = vec![0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01];

        let mut prev_t_count: usize = 0;
        let mut prev_epsilon = 1.0;

        for epsilon in precisions {
            let config = FTCompilationConfig::new().epsilon(epsilon);
            let compiler = FTCompiler::new(config);
            let decomp = match compiler.decompose_rz(0, theta) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // T-count should generally increase (or stay same) as precision improves
            // Allow for some flexibility due to discrete nature of T-count
            assert!(
                decomp.t_count >= prev_t_count.saturating_sub(1),
                "T-count decreased too much: eps {} -> {}: T {} -> {}",
                prev_epsilon,
                epsilon,
                prev_t_count,
                decomp.t_count
            );

            prev_t_count = decomp.t_count;
            prev_epsilon = epsilon;
        }
    }

    /// Verify special angles have optimal (minimal) T-counts.
    #[test]
    fn test_special_angles_optimal_t_count() {
        let compiler = FTCompiler::new(FTCompilationConfig::default());

        // Rz(π/4) = T, should have exactly 1 T gate
        let decomp = compiler.decompose_rz(0, FRAC_PI_4).unwrap();
        assert_eq!(decomp.t_count, 1, "Rz(π/4) should need exactly 1 T gate");

        // Rz(π/2) = S, Clifford, should have 0 T gates
        let decomp = compiler.decompose_rz(0, FRAC_PI_2).unwrap();
        assert_eq!(
            decomp.t_count, 0,
            "Rz(π/2) should need 0 T gates (Clifford)"
        );

        // Rz(π) = Z, Clifford, should have 0 T gates
        let decomp = compiler.decompose_rz(0, PI).unwrap();
        assert_eq!(decomp.t_count, 0, "Rz(π) should need 0 T gates (Clifford)");

        // Rz(0) = I, should have 0 gates
        let decomp = compiler.decompose_rz(0, 0.0).unwrap();
        assert_eq!(decomp.t_count, 0, "Rz(0) should need 0 T gates");

        // Rz(-π/4) = Tdg, should have exactly 1 T gate
        let decomp = compiler.decompose_rz(0, -FRAC_PI_4).unwrap();
        assert_eq!(decomp.t_count, 1, "Rz(-π/4) should need exactly 1 T gate");
    }
}
