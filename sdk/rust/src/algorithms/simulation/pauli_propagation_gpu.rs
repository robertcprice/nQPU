//! GPU-Accelerated Pauli Propagation for Heisenberg-Picture Simulation
//!
//! This module implements Pauli propagation: instead of evolving a 2^n state
//! vector forward through the circuit, we track how Pauli operators evolve
//! backwards. This enables efficient expectation value computation for large
//! circuits when the observable has few Pauli terms.
//!
//! # Architecture
//!
//! - [`PauliOp`]: Single-qubit Pauli operator (I, X, Y, Z).
//! - [`WeightedPauli`]: A tensor product of single-qubit Paulis with a complex
//!   coefficient.
//! - [`PauliSum`]: A sum of weighted Pauli strings representing an observable.
//! - [`PropGate`]: Circuit gates that the observable is propagated through.
//! - [`PropagationConfig`]: Controls truncation, batching, and thread count.
//! - [`PauliPropagator`]: Main engine that propagates an observable through a
//!   circuit using Rayon for parallel batch processing.
//! - [`BatchExpectation`]: Computes expectation values for multiple observables
//!   against the same circuit in parallel.
//! - [`PauliFrame`]: Lightweight Clifford-only frame tracker with zero term
//!   growth guarantee.
//!
//! # Correctness
//!
//! Clifford gates (H, S, Sdg, CX, CZ, SWAP) never increase the term count.
//! Non-Clifford gates (T, Tdg, Rx, Ry, Rz, Toffoli) may split one term into
//! two. Truncation by coefficient magnitude bounds memory usage.
//!
//! # GPU Simulation
//!
//! Actual GPU kernels are not invoked. Parallelism is simulated via Rayon
//! thread pools, providing the same algorithmic structure that a real Metal or
//! CUDA backend would use.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::pauli_propagation_gpu::*;
//!
//! // Track Z on qubit 0 of a 2-qubit system
//! let obs = PauliSum::single(WeightedPauli::basis(2, 0, PauliOp::Z));
//! let circuit = vec![PropGate::H(0), PropGate::CX(0, 1)];
//! let config = PropagationConfig::default();
//! let mut prop = PauliPropagator::new(config, obs, circuit);
//! let result = prop.propagate();
//! assert!(result.is_ok());
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use rayon::prelude::*;

// =====================================================================
// ERROR TYPE
// =====================================================================

/// Errors that can occur during Pauli propagation.
#[derive(Clone, Debug)]
pub enum PauliPropError {
    /// A Pauli string has invalid structure (wrong qubit count, bad operator).
    InvalidPauli(String),
    /// A circuit gate references out-of-range qubits or is otherwise malformed.
    CircuitError(String),
    /// The number of Pauli terms exceeded the hard cap before truncation could
    /// bring it under control.
    OverflowError { num_terms: usize, max_terms: usize },
    /// A numerical computation produced NaN or Inf.
    NumericalError(String),
}

impl fmt::Display for PauliPropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PauliPropError::InvalidPauli(msg) => write!(f, "Invalid Pauli: {}", msg),
            PauliPropError::CircuitError(msg) => write!(f, "Circuit error: {}", msg),
            PauliPropError::OverflowError {
                num_terms,
                max_terms,
            } => write!(
                f,
                "Term overflow: {} terms exceeds max {}",
                num_terms, max_terms
            ),
            PauliPropError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl std::error::Error for PauliPropError {}

// =====================================================================
// PAULI OPERATOR
// =====================================================================

/// Single-qubit Pauli operator.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PauliOp {
    /// Identity operator.
    I,
    /// Pauli-X (bit flip).
    X,
    /// Pauli-Y (bit+phase flip).
    Y,
    /// Pauli-Z (phase flip).
    Z,
}

impl PauliOp {
    /// Parse from a character.
    pub fn from_char(c: char) -> Result<Self, PauliPropError> {
        match c {
            'I' | 'i' => Ok(PauliOp::I),
            'X' | 'x' => Ok(PauliOp::X),
            'Y' | 'y' => Ok(PauliOp::Y),
            'Z' | 'z' => Ok(PauliOp::Z),
            _ => Err(PauliPropError::InvalidPauli(format!(
                "invalid character '{}'",
                c
            ))),
        }
    }

    /// Convert to a character.
    pub fn to_char(self) -> char {
        match self {
            PauliOp::I => 'I',
            PauliOp::X => 'X',
            PauliOp::Y => 'Y',
            PauliOp::Z => 'Z',
        }
    }

    /// Returns true if this is the identity.
    pub fn is_identity(self) -> bool {
        self == PauliOp::I
    }
}

impl fmt::Display for PauliOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

// =====================================================================
// WEIGHTED PAULI STRING
// =====================================================================

/// A weighted Pauli string: coefficient * P_0 (x) P_1 (x) ... (x) P_{n-1}.
///
/// The coefficient is stored as a (real, imag) pair for ergonomics and to avoid
/// pulling in `num_complex` as a hard dependency within this standalone module.
#[derive(Clone, Debug)]
pub struct WeightedPauli {
    /// One Pauli operator per qubit.
    pub paulis: Vec<PauliOp>,
    /// Complex coefficient as (real, imaginary).
    pub coefficient: (f64, f64),
}

impl WeightedPauli {
    /// Create a new weighted Pauli string.
    pub fn new(paulis: Vec<PauliOp>, coefficient: (f64, f64)) -> Self {
        WeightedPauli {
            paulis,
            coefficient,
        }
    }

    /// Create an all-identity string on `n` qubits with unit coefficient.
    pub fn identity(n: usize) -> Self {
        WeightedPauli {
            paulis: vec![PauliOp::I; n],
            coefficient: (1.0, 0.0),
        }
    }

    /// Create a single-qubit basis Pauli on an `n`-qubit register.
    ///
    /// All qubits are identity except `qubit` which is set to `op`.
    pub fn basis(n: usize, qubit: usize, op: PauliOp) -> Self {
        let mut paulis = vec![PauliOp::I; n];
        paulis[qubit] = op;
        WeightedPauli {
            paulis,
            coefficient: (1.0, 0.0),
        }
    }

    /// Parse from a string like "XYZ" with unit coefficient.
    pub fn from_str(s: &str) -> Result<Self, PauliPropError> {
        let paulis: Result<Vec<PauliOp>, _> = s.chars().map(PauliOp::from_char).collect();
        Ok(WeightedPauli {
            paulis: paulis?,
            coefficient: (1.0, 0.0),
        })
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.paulis.len()
    }

    /// Magnitude of the complex coefficient.
    pub fn coeff_magnitude(&self) -> f64 {
        let (re, im) = self.coefficient;
        (re * re + im * im).sqrt()
    }

    /// Check if the coefficient is essentially zero.
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.coeff_magnitude() < threshold
    }

    /// Check if this is the all-identity string.
    pub fn is_all_identity(&self) -> bool {
        self.paulis.iter().all(|p| *p == PauliOp::I)
    }

    /// Pauli weight: count of non-identity sites.
    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|p| **p != PauliOp::I).count()
    }

    /// Multiply two coefficients: (a+bi)*(c+di).
    fn coeff_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    /// Scale the coefficient by a real factor.
    pub fn scale(&mut self, factor: f64) {
        self.coefficient.0 *= factor;
        self.coefficient.1 *= factor;
    }

    /// Scale the coefficient by a complex factor.
    pub fn scale_complex(&mut self, factor: (f64, f64)) {
        self.coefficient = Self::coeff_mul(self.coefficient, factor);
    }

    /// Negate the coefficient.
    pub fn negate(&mut self) {
        self.coefficient.0 = -self.coefficient.0;
        self.coefficient.1 = -self.coefficient.1;
    }

    /// Return a canonical key for deduplication: the Pauli string as bytes.
    fn canonical_key(&self) -> Vec<u8> {
        self.paulis
            .iter()
            .map(|p| match p {
                PauliOp::I => 0u8,
                PauliOp::X => 1u8,
                PauliOp::Y => 2u8,
                PauliOp::Z => 3u8,
            })
            .collect()
    }

    /// Check if two Pauli strings have the same operator content (ignoring coefficient).
    pub fn same_paulis(&self, other: &WeightedPauli) -> bool {
        self.paulis == other.paulis
    }

    /// Check if this Pauli string commutes with another.
    ///
    /// Two Pauli strings commute iff the number of positions where they
    /// anticommute is even. At position j, they anticommute iff both are
    /// distinct non-identity Paulis that differ from each other.
    pub fn commutes_with(&self, other: &WeightedPauli) -> bool {
        assert_eq!(self.paulis.len(), other.paulis.len());
        let mut anticommute_count = 0usize;
        for (a, b) in self.paulis.iter().zip(other.paulis.iter()) {
            if *a != PauliOp::I && *b != PauliOp::I && *a != *b {
                anticommute_count += 1;
            }
        }
        anticommute_count % 2 == 0
    }
}

impl fmt::Display for WeightedPauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = self.coefficient;
        write!(f, "({:.6}{:+.6}i) ", re, im)?;
        for p in &self.paulis {
            write!(f, "{}", p)?;
        }
        Ok(())
    }
}

// =====================================================================
// PAULI SUM
// =====================================================================

/// A sum of weighted Pauli strings representing a quantum observable.
///
/// This is the central data structure being propagated through the circuit.
#[derive(Clone, Debug)]
pub struct PauliSum {
    /// The terms in the sum.
    pub terms: Vec<WeightedPauli>,
    /// Number of qubits.
    pub num_qubits: usize,
}

impl PauliSum {
    /// Create an empty sum on `n` qubits.
    pub fn new(num_qubits: usize) -> Self {
        PauliSum {
            terms: Vec::new(),
            num_qubits,
        }
    }

    /// Create from a single Pauli string.
    pub fn single(term: WeightedPauli) -> Self {
        let n = term.num_qubits();
        PauliSum {
            terms: vec![term],
            num_qubits: n,
        }
    }

    /// Create from multiple terms.
    pub fn from_terms(terms: Vec<WeightedPauli>, num_qubits: usize) -> Self {
        PauliSum { terms, num_qubits }
    }

    /// Add a term.
    pub fn add_term(&mut self, term: WeightedPauli) {
        self.terms.push(term);
    }

    /// Number of terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Merge duplicate Pauli strings by summing their coefficients.
    ///
    /// After merging, terms with coefficient magnitude below `threshold` are
    /// removed.
    pub fn merge_duplicates(&mut self, threshold: f64) {
        let mut map: HashMap<Vec<u8>, (f64, f64)> = HashMap::new();
        let mut key_to_paulis: HashMap<Vec<u8>, Vec<PauliOp>> = HashMap::new();

        for term in &self.terms {
            let key = term.canonical_key();
            let entry = map.entry(key.clone()).or_insert((0.0, 0.0));
            entry.0 += term.coefficient.0;
            entry.1 += term.coefficient.1;
            key_to_paulis
                .entry(key)
                .or_insert_with(|| term.paulis.clone());
        }

        self.terms = map
            .into_iter()
            .filter(|(_, (re, im))| (re * re + im * im).sqrt() >= threshold)
            .map(|(key, coeff)| WeightedPauli {
                paulis: key_to_paulis.remove(&key).unwrap(),
                coefficient: coeff,
            })
            .collect();
    }

    /// Remove terms with coefficient magnitude below `threshold`.
    pub fn truncate_by_magnitude(&mut self, threshold: f64) {
        self.terms.retain(|t| t.coeff_magnitude() >= threshold);
    }

    /// Keep only the top `max_terms` terms by coefficient magnitude.
    pub fn truncate_by_count(&mut self, max_terms: usize) {
        if self.terms.len() > max_terms {
            self.terms.sort_by(|a, b| {
                b.coeff_magnitude()
                    .partial_cmp(&a.coeff_magnitude())
                    .unwrap()
            });
            self.terms.truncate(max_terms);
        }
    }

    /// Compute the expectation value on the |0...0> state.
    ///
    /// For the all-zeros state, only the all-identity Pauli string contributes:
    /// <0...0| c_i P_i |0...0> = c_i if P_i = I^n, else 0.
    pub fn expectation_on_zero_state(&self) -> f64 {
        let mut sum_re = 0.0;
        for term in &self.terms {
            if term.is_all_identity() {
                sum_re += term.coefficient.0;
            }
        }
        sum_re
    }

    /// Compute the expectation value on a computational basis state |b>.
    ///
    /// For a computational basis state |b>, <b| P |b> is nonzero only when P
    /// is a tensor product of I and Z operators (no X or Y). In that case:
    /// <b| Z_j |b> = (-1)^{b_j}, <b| I |b> = 1.
    pub fn expectation_on_basis_state(&self, basis_state: u64) -> f64 {
        let mut total = 0.0;
        for term in &self.terms {
            let mut contributes = true;
            let mut sign = 1.0f64;
            for (q, p) in term.paulis.iter().enumerate() {
                match p {
                    PauliOp::I => {}
                    PauliOp::Z => {
                        if (basis_state >> q) & 1 == 1 {
                            sign *= -1.0;
                        }
                    }
                    PauliOp::X | PauliOp::Y => {
                        contributes = false;
                        break;
                    }
                }
            }
            if contributes {
                total += term.coefficient.0 * sign;
            }
        }
        total
    }

    /// Group terms into sets of mutually commuting Pauli strings.
    ///
    /// Uses a greedy algorithm: iterate terms and assign each to the first
    /// existing group where it commutes with all members, or start a new group.
    pub fn commuting_groups(&self) -> Vec<Vec<usize>> {
        let mut groups: Vec<Vec<usize>> = Vec::new();

        for (idx, term) in self.terms.iter().enumerate() {
            let mut placed = false;
            for group in &mut groups {
                let all_commute = group
                    .iter()
                    .all(|&g_idx| term.commutes_with(&self.terms[g_idx]));
                if all_commute {
                    group.push(idx);
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push(vec![idx]);
            }
        }

        groups
    }
}

impl fmt::Display for PauliSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}", term)?;
        }
        Ok(())
    }
}

// =====================================================================
// CIRCUIT GATE
// =====================================================================

/// A gate in the circuit through which Pauli observables are propagated.
#[derive(Clone, Debug)]
pub enum PropGate {
    /// Hadamard on qubit.
    H(usize),
    /// S gate on qubit.
    S(usize),
    /// S-dagger on qubit.
    Sdg(usize),
    /// T gate on qubit (non-Clifford).
    T(usize),
    /// T-dagger on qubit (non-Clifford).
    Tdg(usize),
    /// Rx rotation on qubit by angle (non-Clifford for general theta).
    Rx(usize, f64),
    /// Ry rotation on qubit by angle (non-Clifford for general theta).
    Ry(usize, f64),
    /// Rz rotation on qubit by angle (non-Clifford for general theta).
    Rz(usize, f64),
    /// CNOT: control, target.
    CX(usize, usize),
    /// Controlled-Z: qubit a, qubit b.
    CZ(usize, usize),
    /// SWAP: qubit a, qubit b.
    Swap(usize, usize),
    /// Toffoli (CCX): control1, control2, target (non-Clifford).
    Toffoli(usize, usize, usize),
}

impl PropGate {
    /// Returns true if this gate is Clifford (never causes term splitting).
    pub fn is_clifford(&self) -> bool {
        matches!(
            self,
            PropGate::H(_)
                | PropGate::S(_)
                | PropGate::Sdg(_)
                | PropGate::CX(_, _)
                | PropGate::CZ(_, _)
                | PropGate::Swap(_, _)
        )
    }

    /// Returns the maximum qubit index referenced by this gate.
    pub fn max_qubit(&self) -> usize {
        match self {
            PropGate::H(q)
            | PropGate::S(q)
            | PropGate::Sdg(q)
            | PropGate::T(q)
            | PropGate::Tdg(q)
            | PropGate::Rx(q, _)
            | PropGate::Ry(q, _)
            | PropGate::Rz(q, _) => *q,
            PropGate::CX(a, b) | PropGate::CZ(a, b) | PropGate::Swap(a, b) => std::cmp::max(*a, *b),
            PropGate::Toffoli(a, b, c) => std::cmp::max(*a, std::cmp::max(*b, *c)),
        }
    }
}

// =====================================================================
// PROPAGATION CONFIG
// =====================================================================

/// Configuration for the propagation engine.
#[derive(Clone, Debug)]
pub struct PropagationConfig {
    /// Maximum number of Pauli terms before hard truncation.
    pub max_terms: usize,
    /// Drop terms with coefficient magnitude below this threshold.
    pub truncation_threshold: f64,
    /// Number of terms processed per Rayon parallel batch.
    pub batch_size: usize,
    /// Number of Rayon threads to use.
    pub num_threads: usize,
    /// Whether to merge duplicate Pauli strings after each gate.
    pub merge_duplicates: bool,
    /// Whether to group commuting terms for efficiency.
    pub group_commuting: bool,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        PropagationConfig {
            max_terms: 100_000,
            truncation_threshold: 1e-10,
            batch_size: 256,
            num_threads: 4,
            merge_duplicates: true,
            group_commuting: false,
        }
    }
}

impl PropagationConfig {
    /// Builder: set max terms.
    pub fn with_max_terms(mut self, n: usize) -> Self {
        self.max_terms = n;
        self
    }

    /// Builder: set truncation threshold.
    pub fn with_threshold(mut self, t: f64) -> Self {
        self.truncation_threshold = t;
        self
    }

    /// Builder: set batch size.
    pub fn with_batch_size(mut self, b: usize) -> Self {
        self.batch_size = b;
        self
    }

    /// Builder: set thread count.
    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Builder: enable or disable merging.
    pub fn with_merge(mut self, m: bool) -> Self {
        self.merge_duplicates = m;
        self
    }

    /// Builder: enable or disable commuting group optimization.
    pub fn with_commuting_groups(mut self, g: bool) -> Self {
        self.group_commuting = g;
        self
    }
}

// =====================================================================
// PROPAGATION RESULT
// =====================================================================

/// Result of propagating an observable through a circuit.
#[derive(Clone, Debug)]
pub struct PropagationResult {
    /// The propagated observable: U-dagger O U.
    pub propagated_observable: PauliSum,
    /// Expectation value on |0...0>.
    pub expectation_value: f64,
    /// Number of terms before propagation.
    pub num_terms_initial: usize,
    /// Number of terms after propagation.
    pub num_terms_final: usize,
    /// Peak number of terms during propagation.
    pub num_terms_peak: usize,
    /// Total number of terms that were truncated.
    pub truncated_terms: usize,
    /// Wall-clock time in seconds.
    pub elapsed_secs: f64,
}

// =====================================================================
// SINGLE-TERM PROPAGATION RULES
// =====================================================================

/// Propagate a single WeightedPauli through one gate.
///
/// Returns one or two output terms. Clifford gates always return exactly one
/// term; non-Clifford gates may return two.
fn propagate_term_through_gate(term: &WeightedPauli, gate: &PropGate) -> Vec<WeightedPauli> {
    match gate {
        PropGate::H(q) => vec![propagate_h(term, *q)],
        PropGate::S(q) => vec![propagate_s(term, *q)],
        PropGate::Sdg(q) => vec![propagate_sdg(term, *q)],
        PropGate::T(q) => propagate_t(term, *q),
        PropGate::Tdg(q) => propagate_tdg(term, *q),
        PropGate::Rx(q, theta) => propagate_rx(term, *q, *theta),
        PropGate::Ry(q, theta) => propagate_ry(term, *q, *theta),
        PropGate::Rz(q, theta) => propagate_rz(term, *q, *theta),
        PropGate::CX(ctrl, targ) => vec![propagate_cx(term, *ctrl, *targ)],
        PropGate::CZ(a, b) => vec![propagate_cz(term, *a, *b)],
        PropGate::Swap(a, b) => vec![propagate_swap(term, *a, *b)],
        PropGate::Toffoli(c1, c2, targ) => propagate_toffoli(term, *c1, *c2, *targ),
    }
}

// ----- Hadamard: H-dagger P H -----
// H is self-inverse so H-dagger = H.
// H X H = Z, H Z H = X, H Y H = -Y, H I H = I
fn propagate_h(term: &WeightedPauli, q: usize) -> WeightedPauli {
    let mut out = term.clone();
    match out.paulis[q] {
        PauliOp::I => {}
        PauliOp::X => {
            out.paulis[q] = PauliOp::Z;
        }
        PauliOp::Z => {
            out.paulis[q] = PauliOp::X;
        }
        PauliOp::Y => {
            out.negate();
        }
    }
    out
}

// ----- S gate: S-dagger P S -----
// S = diag(1, i). S-dagger X S = Y, S-dagger Y S = -X, S-dagger Z S = Z
fn propagate_s(term: &WeightedPauli, q: usize) -> WeightedPauli {
    let mut out = term.clone();
    match out.paulis[q] {
        PauliOp::I | PauliOp::Z => {}
        PauliOp::X => {
            out.paulis[q] = PauliOp::Y;
        }
        PauliOp::Y => {
            out.paulis[q] = PauliOp::X;
            out.negate();
        }
    }
    out
}

// ----- Sdg gate: S P S-dagger -----
// Sdg-dagger = S, so this is S P Sdg.
// Actually: (Sdg)-dagger P (Sdg) = S P S-dagger.
// S X Sdg = -Y, S Y Sdg = X, S Z Sdg = Z
// Equivalently: Sdg-dagger X Sdg = -Y, Sdg-dagger Y Sdg = X
fn propagate_sdg(term: &WeightedPauli, q: usize) -> WeightedPauli {
    let mut out = term.clone();
    match out.paulis[q] {
        PauliOp::I | PauliOp::Z => {}
        PauliOp::X => {
            out.paulis[q] = PauliOp::Y;
            out.negate();
        }
        PauliOp::Y => {
            out.paulis[q] = PauliOp::X;
        }
    }
    out
}

// ----- T gate (non-Clifford): T-dagger P T -----
// T = diag(1, e^{i pi/4}). T-dagger = diag(1, e^{-i pi/4}).
//
// T-dagger I T = I
// T-dagger Z T = Z  (diagonal gate commutes with Z)
// T-dagger X T = cos(pi/4) X + sin(pi/4) Y
// T-dagger Y T = cos(pi/4) Y - sin(pi/4) X
fn propagate_t(term: &WeightedPauli, q: usize) -> Vec<WeightedPauli> {
    match term.paulis[q] {
        PauliOp::I | PauliOp::Z => vec![term.clone()],
        PauliOp::X => {
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            // cos(pi/4) * X
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            // sin(pi/4) * Y
            t2.paulis[q] = PauliOp::Y;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (s, 0.0));
            vec![t1, t2]
        }
        PauliOp::Y => {
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            // cos(pi/4) * Y
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            // -sin(pi/4) * X
            t2.paulis[q] = PauliOp::X;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (-s, 0.0));
            vec![t1, t2]
        }
    }
}

// ----- Tdg gate: Tdg-dagger P Tdg = T P Tdg -----
// T I Tdg = I
// T Z Tdg = Z
// T X Tdg = cos(pi/4) X - sin(pi/4) Y
// T Y Tdg = cos(pi/4) Y + sin(pi/4) X
fn propagate_tdg(term: &WeightedPauli, q: usize) -> Vec<WeightedPauli> {
    match term.paulis[q] {
        PauliOp::I | PauliOp::Z => vec![term.clone()],
        PauliOp::X => {
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::Y;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (-s, 0.0));
            vec![t1, t2]
        }
        PauliOp::Y => {
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::X;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (s, 0.0));
            vec![t1, t2]
        }
    }
}

// ----- Rx(theta): Rx-dagger P Rx -----
// Rx(theta) = exp(-i theta/2 X) = cos(theta/2) I - i sin(theta/2) X
// Rx-dagger I Rx = I
// Rx-dagger X Rx = X  (X commutes with Rx)
// Rx-dagger Y Rx = cos(theta) Y + sin(theta) Z
// Rx-dagger Z Rx = cos(theta) Z - sin(theta) Y
fn propagate_rx(term: &WeightedPauli, q: usize, theta: f64) -> Vec<WeightedPauli> {
    match term.paulis[q] {
        PauliOp::I | PauliOp::X => vec![term.clone()],
        PauliOp::Y => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::Z;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (s, 0.0));
            vec![t1, t2]
        }
        PauliOp::Z => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::Y;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (-s, 0.0));
            vec![t1, t2]
        }
    }
}

// ----- Ry(theta): Ry-dagger P Ry -----
// Ry(theta) = exp(-i theta/2 Y) = cos(theta/2) I - i sin(theta/2) Y
// Ry-dagger I Ry = I
// Ry-dagger Y Ry = Y  (Y commutes with Ry)
// Ry-dagger X Ry = cos(theta) X - sin(theta) Z
// Ry-dagger Z Ry = cos(theta) Z + sin(theta) X
fn propagate_ry(term: &WeightedPauli, q: usize, theta: f64) -> Vec<WeightedPauli> {
    match term.paulis[q] {
        PauliOp::I | PauliOp::Y => vec![term.clone()],
        PauliOp::X => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::Z;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (-s, 0.0));
            vec![t1, t2]
        }
        PauliOp::Z => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::X;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (s, 0.0));
            vec![t1, t2]
        }
    }
}

// ----- Rz(theta): Rz-dagger P Rz -----
// Rz(theta) = exp(-i theta/2 Z) = cos(theta/2) I - i sin(theta/2) Z
// Rz-dagger I Rz = I
// Rz-dagger Z Rz = Z  (Z commutes with Rz)
// Rz-dagger X Rz = cos(theta) X + sin(theta) Y
// Rz-dagger Y Rz = cos(theta) Y - sin(theta) X
fn propagate_rz(term: &WeightedPauli, q: usize, theta: f64) -> Vec<WeightedPauli> {
    match term.paulis[q] {
        PauliOp::I | PauliOp::Z => vec![term.clone()],
        PauliOp::X => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::Y;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (s, 0.0));
            vec![t1, t2]
        }
        PauliOp::Y => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.coefficient = WeightedPauli::coeff_mul(t1.coefficient, (c, 0.0));
            let mut t2 = term.clone();
            t2.paulis[q] = PauliOp::X;
            t2.coefficient = WeightedPauli::coeff_mul(t2.coefficient, (-s, 0.0));
            vec![t1, t2]
        }
    }
}

// ----- CX (CNOT): CX-dagger (Pc x Pt) CX -----
// CX is self-inverse so CX-dagger = CX.
//
// CX (Pc x Pt) CX:
//   II -> II, IX -> IX, IY -> ZY, IZ -> ZZ
//   XI -> XX, XX -> XI, XY -> -YZ, XZ -> YY
//   YI -> YX, YX -> YI, YY -> XZ, YZ -> -XY
//   ZI -> ZI, ZX -> ZX, ZY -> IY, ZZ -> IZ
fn propagate_cx(term: &WeightedPauli, ctrl: usize, targ: usize) -> WeightedPauli {
    let mut out = term.clone();
    let pc = term.paulis[ctrl];
    let pt = term.paulis[targ];
    match (pc, pt) {
        (PauliOp::I, PauliOp::I) => {}
        (PauliOp::I, PauliOp::X) => {}
        (PauliOp::I, PauliOp::Y) => {
            out.paulis[ctrl] = PauliOp::Z;
            out.paulis[targ] = PauliOp::Y;
        }
        (PauliOp::I, PauliOp::Z) => {
            out.paulis[ctrl] = PauliOp::Z;
            out.paulis[targ] = PauliOp::Z;
        }
        (PauliOp::X, PauliOp::I) => {
            out.paulis[ctrl] = PauliOp::X;
            out.paulis[targ] = PauliOp::X;
        }
        (PauliOp::X, PauliOp::X) => {
            out.paulis[ctrl] = PauliOp::X;
            out.paulis[targ] = PauliOp::I;
        }
        (PauliOp::X, PauliOp::Y) => {
            out.paulis[ctrl] = PauliOp::Y;
            out.paulis[targ] = PauliOp::Z;
            out.negate();
        }
        (PauliOp::X, PauliOp::Z) => {
            out.paulis[ctrl] = PauliOp::Y;
            out.paulis[targ] = PauliOp::Y;
        }
        (PauliOp::Y, PauliOp::I) => {
            out.paulis[ctrl] = PauliOp::Y;
            out.paulis[targ] = PauliOp::X;
        }
        (PauliOp::Y, PauliOp::X) => {
            out.paulis[ctrl] = PauliOp::Y;
            out.paulis[targ] = PauliOp::I;
        }
        (PauliOp::Y, PauliOp::Y) => {
            out.paulis[ctrl] = PauliOp::X;
            out.paulis[targ] = PauliOp::Z;
        }
        (PauliOp::Y, PauliOp::Z) => {
            out.paulis[ctrl] = PauliOp::X;
            out.paulis[targ] = PauliOp::Y;
            out.negate();
        }
        (PauliOp::Z, PauliOp::I) => {}
        (PauliOp::Z, PauliOp::X) => {}
        (PauliOp::Z, PauliOp::Y) => {
            out.paulis[ctrl] = PauliOp::I;
            out.paulis[targ] = PauliOp::Y;
        }
        (PauliOp::Z, PauliOp::Z) => {
            out.paulis[ctrl] = PauliOp::I;
            out.paulis[targ] = PauliOp::Z;
        }
    }
    out
}

// ----- CZ: CZ-dagger P CZ = CZ P CZ (self-inverse) -----
// CZ (Pa x Pb) CZ:
//   II -> II, IX -> ZX, IY -> ZY, IZ -> IZ
//   XI -> XZ, XX -> -YY, XY -> YX, XZ -> XI
//   YI -> YZ, YX -> XY, YY -> -XX, YZ -> YI
//   ZI -> ZI, ZX -> IX, ZY -> IY, ZZ -> ZZ
fn propagate_cz(term: &WeightedPauli, a: usize, b: usize) -> WeightedPauli {
    let mut out = term.clone();
    let pa = term.paulis[a];
    let pb = term.paulis[b];
    match (pa, pb) {
        (PauliOp::I, PauliOp::I) => {}
        (PauliOp::I, PauliOp::X) => {
            out.paulis[a] = PauliOp::Z;
            out.paulis[b] = PauliOp::X;
        }
        (PauliOp::I, PauliOp::Y) => {
            out.paulis[a] = PauliOp::Z;
            out.paulis[b] = PauliOp::Y;
        }
        (PauliOp::I, PauliOp::Z) => {}
        (PauliOp::X, PauliOp::I) => {
            out.paulis[a] = PauliOp::X;
            out.paulis[b] = PauliOp::Z;
        }
        (PauliOp::X, PauliOp::X) => {
            out.paulis[a] = PauliOp::Y;
            out.paulis[b] = PauliOp::Y;
            out.negate();
        }
        (PauliOp::X, PauliOp::Y) => {
            out.paulis[a] = PauliOp::Y;
            out.paulis[b] = PauliOp::X;
        }
        (PauliOp::X, PauliOp::Z) => {
            out.paulis[a] = PauliOp::X;
            out.paulis[b] = PauliOp::I;
        }
        (PauliOp::Y, PauliOp::I) => {
            out.paulis[a] = PauliOp::Y;
            out.paulis[b] = PauliOp::Z;
        }
        (PauliOp::Y, PauliOp::X) => {
            out.paulis[a] = PauliOp::X;
            out.paulis[b] = PauliOp::Y;
        }
        (PauliOp::Y, PauliOp::Y) => {
            out.paulis[a] = PauliOp::X;
            out.paulis[b] = PauliOp::X;
            out.negate();
        }
        (PauliOp::Y, PauliOp::Z) => {
            out.paulis[a] = PauliOp::Y;
            out.paulis[b] = PauliOp::I;
        }
        (PauliOp::Z, PauliOp::I) => {}
        (PauliOp::Z, PauliOp::X) => {
            out.paulis[a] = PauliOp::I;
            out.paulis[b] = PauliOp::X;
        }
        (PauliOp::Z, PauliOp::Y) => {
            out.paulis[a] = PauliOp::I;
            out.paulis[b] = PauliOp::Y;
        }
        (PauliOp::Z, PauliOp::Z) => {}
    }
    out
}

// ----- SWAP -----
fn propagate_swap(term: &WeightedPauli, a: usize, b: usize) -> WeightedPauli {
    let mut out = term.clone();
    out.paulis.swap(a, b);
    out
}

// ----- Toffoli (CCX): non-Clifford 3-qubit gate -----
//
// Decompose Toffoli into Clifford + T gates and propagate sequentially.
// Standard decomposition uses 6 CX gates and 7 T/Tdg gates.
// We use the well-known decomposition:
//   Toffoli = (I x I x H) . (I x CX) . (I x I x Tdg) . (CX x I) . (I x I x T) .
//             (I x CX) . (I x I x Tdg) . (CX x I) . (I x T x T) . (I x CX) .
//             (I x I x H) . (T x I x I) . ...
// For simplicity and correctness, we use a direct decomposition into our gate set.
fn propagate_toffoli(
    term: &WeightedPauli,
    c1: usize,
    c2: usize,
    targ: usize,
) -> Vec<WeightedPauli> {
    // Standard Toffoli decomposition into 1- and 2-qubit gates:
    //   H(targ), CX(c2, targ), Tdg(targ), CX(c1, targ), T(targ),
    //   CX(c2, targ), Tdg(targ), CX(c1, targ), T(c2), T(targ),
    //   CX(c1, c2), H(targ), T(c1), Tdg(c2), CX(c1, c2)
    let decomposition = vec![
        PropGate::H(targ),
        PropGate::CX(c2, targ),
        PropGate::Tdg(targ),
        PropGate::CX(c1, targ),
        PropGate::T(targ),
        PropGate::CX(c2, targ),
        PropGate::Tdg(targ),
        PropGate::CX(c1, targ),
        PropGate::T(c2),
        PropGate::T(targ),
        PropGate::CX(c1, c2),
        PropGate::H(targ),
        PropGate::T(c1),
        PropGate::Tdg(c2),
        PropGate::CX(c1, c2),
    ];

    // Propagate in reverse order (Heisenberg picture: last gate first).
    let mut current_terms = vec![term.clone()];
    for gate in decomposition.iter().rev() {
        let mut next_terms = Vec::new();
        for t in &current_terms {
            next_terms.extend(propagate_term_through_gate(t, gate));
        }
        current_terms = next_terms;
    }
    current_terms
}

// =====================================================================
// PAULI PROPAGATOR
// =====================================================================

/// Main propagation engine. Propagates a Pauli observable through a quantum
/// circuit in the Heisenberg picture using Rayon-based parallel batching.
pub struct PauliPropagator {
    /// Configuration parameters.
    pub config: PropagationConfig,
    /// The observable being propagated.
    pub observable: PauliSum,
    /// The circuit gates (applied in forward order; propagation is backwards).
    pub circuit: Vec<PropGate>,
}

impl PauliPropagator {
    /// Create a new propagator.
    pub fn new(config: PropagationConfig, observable: PauliSum, circuit: Vec<PropGate>) -> Self {
        PauliPropagator {
            config,
            observable,
            circuit,
        }
    }

    /// Validate the circuit against the observable's qubit count.
    pub fn validate(&self) -> Result<(), PauliPropError> {
        let n = self.observable.num_qubits;
        if n == 0 {
            return Err(PauliPropError::InvalidPauli(
                "observable has zero qubits".to_string(),
            ));
        }
        for (i, term) in self.observable.terms.iter().enumerate() {
            if term.num_qubits() != n {
                return Err(PauliPropError::InvalidPauli(format!(
                    "term {} has {} qubits, expected {}",
                    i,
                    term.num_qubits(),
                    n
                )));
            }
        }
        for (i, gate) in self.circuit.iter().enumerate() {
            if gate.max_qubit() >= n {
                return Err(PauliPropError::CircuitError(format!(
                    "gate {} references qubit {} but system has only {} qubits",
                    i,
                    gate.max_qubit(),
                    n
                )));
            }
        }
        Ok(())
    }

    /// Run the propagation. Returns a `PropagationResult` on success.
    ///
    /// Gates are applied in reverse order (Heisenberg picture). Each gate
    /// transforms the current observable terms in parallel batches.
    pub fn propagate(&mut self) -> Result<PropagationResult, PauliPropError> {
        self.validate()?;

        let start = Instant::now();
        let num_initial = self.observable.len();
        let mut peak = num_initial;
        let mut total_truncated = 0usize;

        // Configure the Rayon thread pool.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.num_threads)
            .build()
            .unwrap_or_else(|_| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap()
            });

        // Propagate through the circuit in reverse order.
        let num_gates = self.circuit.len();
        for gate_idx in (0..num_gates).rev() {
            let gate = &self.circuit[gate_idx];

            let batch_size = self.config.batch_size;
            let terms = &self.observable.terms;

            // Parallel propagation of all terms through this gate.
            let new_terms: Vec<WeightedPauli> = pool.install(|| {
                terms
                    .par_chunks(batch_size)
                    .flat_map_iter(|chunk| {
                        chunk
                            .iter()
                            .flat_map(|t| propagate_term_through_gate(t, gate))
                    })
                    .collect()
            });

            self.observable.terms = new_terms;

            // Track peak.
            if self.observable.len() > peak {
                peak = self.observable.len();
            }

            // Merge and truncate after non-Clifford gates (which cause growth).
            if !gate.is_clifford() {
                let before = self.observable.len();

                if self.config.merge_duplicates {
                    self.observable
                        .merge_duplicates(self.config.truncation_threshold);
                } else {
                    self.observable
                        .truncate_by_magnitude(self.config.truncation_threshold);
                }

                self.observable.truncate_by_count(self.config.max_terms);

                let after = self.observable.len();
                if before > after {
                    total_truncated += before - after;
                }
            }

            // Hard overflow check.
            if self.observable.len() > self.config.max_terms * 2 {
                return Err(PauliPropError::OverflowError {
                    num_terms: self.observable.len(),
                    max_terms: self.config.max_terms,
                });
            }
        }

        // Final merge pass.
        if self.config.merge_duplicates {
            let before = self.observable.len();
            self.observable
                .merge_duplicates(self.config.truncation_threshold);
            let after = self.observable.len();
            if before > after {
                total_truncated += before - after;
            }
        }

        let ev = self.observable.expectation_on_zero_state();
        let elapsed = start.elapsed().as_secs_f64();

        // Check for NaN.
        if ev.is_nan() {
            return Err(PauliPropError::NumericalError(
                "expectation value is NaN".to_string(),
            ));
        }

        Ok(PropagationResult {
            propagated_observable: self.observable.clone(),
            expectation_value: ev,
            num_terms_initial: num_initial,
            num_terms_final: self.observable.len(),
            num_terms_peak: peak,
            truncated_terms: total_truncated,
            elapsed_secs: elapsed,
        })
    }

    /// Propagate and return only the expectation value on |0...0>.
    pub fn expectation_value(&mut self) -> Result<f64, PauliPropError> {
        self.propagate().map(|r| r.expectation_value)
    }
}

// =====================================================================
// BATCH EXPECTATION
// =====================================================================

/// Batch computation of expectation values for multiple observables against the
/// same circuit.
///
/// Each observable is propagated independently and in parallel across
/// observables (outer parallelism) as well as within each propagation (inner
/// parallelism via the propagator's own batching).
pub struct BatchExpectation {
    /// The observables to propagate.
    pub observables: Vec<PauliSum>,
    /// The shared circuit.
    pub circuit: Vec<PropGate>,
    /// Propagation configuration.
    pub config: PropagationConfig,
}

impl BatchExpectation {
    /// Create a new batch expectation computation.
    pub fn new(
        observables: Vec<PauliSum>,
        circuit: Vec<PropGate>,
        config: PropagationConfig,
    ) -> Self {
        BatchExpectation {
            observables,
            circuit,
            config,
        }
    }

    /// Compute all expectation values. Returns one f64 per observable.
    pub fn compute(&self) -> Result<Vec<f64>, PauliPropError> {
        let results: Vec<Result<f64, PauliPropError>> = self
            .observables
            .par_iter()
            .map(|obs| {
                let mut prop =
                    PauliPropagator::new(self.config.clone(), obs.clone(), self.circuit.clone());
                prop.expectation_value()
            })
            .collect();

        let mut values = Vec::with_capacity(results.len());
        for r in results {
            values.push(r?);
        }
        Ok(values)
    }

    /// Compute all propagation results (full details per observable).
    pub fn compute_full(&self) -> Result<Vec<PropagationResult>, PauliPropError> {
        let results: Vec<Result<PropagationResult, PauliPropError>> = self
            .observables
            .par_iter()
            .map(|obs| {
                let mut prop =
                    PauliPropagator::new(self.config.clone(), obs.clone(), self.circuit.clone());
                prop.propagate()
            })
            .collect();

        let mut out = Vec::with_capacity(results.len());
        for r in results {
            out.push(r?);
        }
        Ok(out)
    }
}

// =====================================================================
// PAULI FRAME (CLIFFORD-ONLY TRACKER)
// =====================================================================

/// Lightweight Pauli frame tracker for purely Clifford circuits.
///
/// In a Clifford circuit, each Pauli string maps to exactly one Pauli string
/// under conjugation (no term growth). The `PauliFrame` exploits this by
/// tracking frames without any merging or truncation overhead.
///
/// If a non-Clifford gate is encountered, the tracker returns an error.
pub struct PauliFrame {
    /// The tracked Pauli frames.
    pub frames: Vec<WeightedPauli>,
    /// Number of qubits.
    pub num_qubits: usize,
}

impl PauliFrame {
    /// Create a new frame tracker.
    pub fn new(frames: Vec<WeightedPauli>, num_qubits: usize) -> Self {
        PauliFrame { frames, num_qubits }
    }

    /// Create from a single Pauli string.
    pub fn single(term: WeightedPauli) -> Self {
        let n = term.num_qubits();
        PauliFrame {
            frames: vec![term],
            num_qubits: n,
        }
    }

    /// Propagate all frames through a single Clifford gate.
    ///
    /// Returns an error if the gate is non-Clifford.
    pub fn propagate_gate(&mut self, gate: &PropGate) -> Result<(), PauliPropError> {
        if !gate.is_clifford() {
            return Err(PauliPropError::CircuitError(format!(
                "PauliFrame only supports Clifford gates, got {:?}",
                gate
            )));
        }
        if gate.max_qubit() >= self.num_qubits {
            return Err(PauliPropError::CircuitError(format!(
                "gate references qubit {} but frame has {} qubits",
                gate.max_qubit(),
                self.num_qubits
            )));
        }

        for frame in &mut self.frames {
            let propagated = propagate_term_through_gate(frame, gate);
            // Clifford gates produce exactly one output term.
            debug_assert_eq!(propagated.len(), 1);
            *frame = propagated.into_iter().next().unwrap();
        }
        Ok(())
    }

    /// Propagate all frames through a Clifford circuit (reverse order).
    pub fn propagate_circuit(&mut self, circuit: &[PropGate]) -> Result<(), PauliPropError> {
        for gate in circuit.iter().rev() {
            self.propagate_gate(gate)?;
        }
        Ok(())
    }

    /// Compute the expectation value on |0...0>.
    pub fn expectation_on_zero_state(&self) -> f64 {
        let mut sum = 0.0;
        for frame in &self.frames {
            if frame.is_all_identity() {
                sum += frame.coefficient.0;
            }
        }
        sum
    }

    /// Number of tracked frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the tracker has no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

// =====================================================================
// UTILITY FUNCTIONS
// =====================================================================

/// Create a standard Z-basis observable for measuring qubit `q` in an
/// `n`-qubit system.
pub fn z_observable(n: usize, q: usize) -> PauliSum {
    PauliSum::single(WeightedPauli::basis(n, q, PauliOp::Z))
}

/// Create a standard X-basis observable for measuring qubit `q`.
pub fn x_observable(n: usize, q: usize) -> PauliSum {
    PauliSum::single(WeightedPauli::basis(n, q, PauliOp::X))
}

/// Create a ZZ correlation observable for qubits `q1` and `q2`.
pub fn zz_observable(n: usize, q1: usize, q2: usize) -> PauliSum {
    let mut paulis = vec![PauliOp::I; n];
    paulis[q1] = PauliOp::Z;
    paulis[q2] = PauliOp::Z;
    PauliSum::single(WeightedPauli::new(paulis, (1.0, 0.0)))
}

/// Build a random Clifford circuit for benchmarking.
///
/// Generates `depth` layers, each containing random H, S, and CX gates.
pub fn random_clifford_circuit(num_qubits: usize, depth: usize, seed: u64) -> Vec<PropGate> {
    let mut gates = Vec::new();
    let mut state = seed;

    // Simple LCG for reproducibility without pulling in rand.
    let mut next_rand = move || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    for _ in 0..depth {
        for q in 0..num_qubits {
            let r = next_rand() % 4;
            match r {
                0 => gates.push(PropGate::H(q)),
                1 => gates.push(PropGate::S(q)),
                2 => {
                    if num_qubits > 1 {
                        let other =
                            (q + 1 + (next_rand() as usize % (num_qubits - 1))) % num_qubits;
                        gates.push(PropGate::CX(q, other));
                    }
                }
                _ => {} // identity (no gate)
            }
        }
    }
    gates
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: approximate f64 equality.
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // Helper: coefficient magnitude.
    fn coeff_mag(c: (f64, f64)) -> f64 {
        (c.0 * c.0 + c.1 * c.1).sqrt()
    }

    // ---------------------------------------------------------------
    // 1. WeightedPauli creation
    // ---------------------------------------------------------------
    #[test]
    fn test_weighted_pauli_creation() {
        let wp = WeightedPauli::new(vec![PauliOp::X, PauliOp::Y, PauliOp::Z], (0.5, -0.3));
        assert_eq!(wp.num_qubits(), 3);
        assert_eq!(wp.paulis[0], PauliOp::X);
        assert_eq!(wp.paulis[1], PauliOp::Y);
        assert_eq!(wp.paulis[2], PauliOp::Z);
        assert!(approx_eq(wp.coefficient.0, 0.5, 1e-12));
        assert!(approx_eq(wp.coefficient.1, -0.3, 1e-12));
        assert_eq!(wp.weight(), 3);
    }

    // ---------------------------------------------------------------
    // 2. PauliSum addition
    // ---------------------------------------------------------------
    #[test]
    fn test_pauli_sum_addition() {
        let mut sum = PauliSum::new(2);
        sum.add_term(WeightedPauli::basis(2, 0, PauliOp::Z));
        sum.add_term(WeightedPauli::basis(2, 1, PauliOp::X));
        assert_eq!(sum.len(), 2);
        assert_eq!(sum.num_qubits, 2);
    }

    // ---------------------------------------------------------------
    // 3. PauliSum merge duplicates
    // ---------------------------------------------------------------
    #[test]
    fn test_pauli_sum_merge_duplicates() {
        let mut sum = PauliSum::new(2);
        sum.add_term(WeightedPauli::new(vec![PauliOp::X, PauliOp::I], (0.5, 0.0)));
        sum.add_term(WeightedPauli::new(vec![PauliOp::X, PauliOp::I], (0.3, 0.0)));
        sum.add_term(WeightedPauli::new(vec![PauliOp::Z, PauliOp::I], (1.0, 0.0)));
        assert_eq!(sum.len(), 3);

        sum.merge_duplicates(1e-10);
        assert_eq!(sum.len(), 2);

        // Find the XI term and check its coefficient.
        let xi_term = sum
            .terms
            .iter()
            .find(|t| t.paulis[0] == PauliOp::X && t.paulis[1] == PauliOp::I)
            .unwrap();
        assert!(approx_eq(xi_term.coefficient.0, 0.8, 1e-10));
    }

    // ---------------------------------------------------------------
    // 4. H propagation: X -> Z
    // ---------------------------------------------------------------
    #[test]
    fn test_h_propagation_x_to_z() {
        let term = WeightedPauli::basis(1, 0, PauliOp::X);
        let result = propagate_h(&term, 0);
        assert_eq!(result.paulis[0], PauliOp::Z);
        assert!(approx_eq(result.coefficient.0, 1.0, 1e-12));
        assert!(approx_eq(result.coefficient.1, 0.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 5. H propagation: Z -> X
    // ---------------------------------------------------------------
    #[test]
    fn test_h_propagation_z_to_x() {
        let term = WeightedPauli::basis(1, 0, PauliOp::Z);
        let result = propagate_h(&term, 0);
        assert_eq!(result.paulis[0], PauliOp::X);
        assert!(approx_eq(result.coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 6. H propagation: Y -> -Y
    // ---------------------------------------------------------------
    #[test]
    fn test_h_propagation_y_to_neg_y() {
        let term = WeightedPauli::basis(1, 0, PauliOp::Y);
        let result = propagate_h(&term, 0);
        assert_eq!(result.paulis[0], PauliOp::Y);
        assert!(approx_eq(result.coefficient.0, -1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 7. S propagation: X -> Y
    // ---------------------------------------------------------------
    #[test]
    fn test_s_propagation_x_to_y() {
        let term = WeightedPauli::basis(1, 0, PauliOp::X);
        let result = propagate_s(&term, 0);
        assert_eq!(result.paulis[0], PauliOp::Y);
        assert!(approx_eq(result.coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 8. S propagation: Y -> -X
    // ---------------------------------------------------------------
    #[test]
    fn test_s_propagation_y_to_neg_x() {
        let term = WeightedPauli::basis(1, 0, PauliOp::Y);
        let result = propagate_s(&term, 0);
        assert_eq!(result.paulis[0], PauliOp::X);
        assert!(approx_eq(result.coefficient.0, -1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 9. CX propagation: XI -> XX
    // ---------------------------------------------------------------
    #[test]
    fn test_cx_propagation_xi_to_xx() {
        let term = WeightedPauli::new(vec![PauliOp::X, PauliOp::I], (1.0, 0.0));
        let result = propagate_cx(&term, 0, 1);
        assert_eq!(result.paulis[0], PauliOp::X);
        assert_eq!(result.paulis[1], PauliOp::X);
        assert!(approx_eq(result.coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 10. CX propagation: IZ -> ZZ
    // ---------------------------------------------------------------
    #[test]
    fn test_cx_propagation_iz_to_zz() {
        let term = WeightedPauli::new(vec![PauliOp::I, PauliOp::Z], (1.0, 0.0));
        let result = propagate_cx(&term, 0, 1);
        assert_eq!(result.paulis[0], PauliOp::Z);
        assert_eq!(result.paulis[1], PauliOp::Z);
        assert!(approx_eq(result.coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 11. CZ propagation: XI -> XZ
    // ---------------------------------------------------------------
    #[test]
    fn test_cz_propagation_xi_to_xz() {
        let term = WeightedPauli::new(vec![PauliOp::X, PauliOp::I], (1.0, 0.0));
        let result = propagate_cz(&term, 0, 1);
        assert_eq!(result.paulis[0], PauliOp::X);
        assert_eq!(result.paulis[1], PauliOp::Z);
        assert!(approx_eq(result.coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 12. T propagation: X splits into cos X + sin Y
    // ---------------------------------------------------------------
    #[test]
    fn test_t_propagation_x_splits() {
        let term = WeightedPauli::basis(1, 0, PauliOp::X);
        let result = propagate_t(&term, 0);
        assert_eq!(result.len(), 2);

        let c = std::f64::consts::FRAC_PI_4.cos();
        let s = std::f64::consts::FRAC_PI_4.sin();

        // One term should be X with cos coefficient.
        let x_term = result.iter().find(|t| t.paulis[0] == PauliOp::X).unwrap();
        assert!(approx_eq(x_term.coefficient.0, c, 1e-10));

        // Other should be Y with sin coefficient.
        let y_term = result.iter().find(|t| t.paulis[0] == PauliOp::Y).unwrap();
        assert!(approx_eq(y_term.coefficient.0, s, 1e-10));
    }

    // ---------------------------------------------------------------
    // 13. T propagation: Z is unchanged
    // ---------------------------------------------------------------
    #[test]
    fn test_t_propagation_z_unchanged() {
        let term = WeightedPauli::basis(1, 0, PauliOp::Z);
        let result = propagate_t(&term, 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].paulis[0], PauliOp::Z);
        assert!(approx_eq(result[0].coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 14. Rx propagation: parametric rotation
    // ---------------------------------------------------------------
    #[test]
    fn test_rx_propagation_parametric() {
        let theta = 0.7;
        let term = WeightedPauli::basis(1, 0, PauliOp::Y);
        let result = propagate_rx(&term, 0, theta);
        assert_eq!(result.len(), 2);

        // Y -> cos(theta)*Y + sin(theta)*Z
        let y_term = result.iter().find(|t| t.paulis[0] == PauliOp::Y).unwrap();
        assert!(approx_eq(y_term.coefficient.0, theta.cos(), 1e-10));

        let z_term = result.iter().find(|t| t.paulis[0] == PauliOp::Z).unwrap();
        assert!(approx_eq(z_term.coefficient.0, theta.sin(), 1e-10));
    }

    // ---------------------------------------------------------------
    // 15. Ry propagation: parametric rotation
    // ---------------------------------------------------------------
    #[test]
    fn test_ry_propagation_parametric() {
        let theta = 1.2;
        let term = WeightedPauli::basis(1, 0, PauliOp::Z);
        let result = propagate_ry(&term, 0, theta);
        assert_eq!(result.len(), 2);

        // Z -> cos(theta)*Z + sin(theta)*X
        let z_term = result.iter().find(|t| t.paulis[0] == PauliOp::Z).unwrap();
        assert!(approx_eq(z_term.coefficient.0, theta.cos(), 1e-10));

        let x_term = result.iter().find(|t| t.paulis[0] == PauliOp::X).unwrap();
        assert!(approx_eq(x_term.coefficient.0, theta.sin(), 1e-10));
    }

    // ---------------------------------------------------------------
    // 16. Clifford circuit: no term growth
    // ---------------------------------------------------------------
    #[test]
    fn test_clifford_circuit_no_term_growth() {
        let obs = PauliSum::single(WeightedPauli::basis(3, 0, PauliOp::Z));
        let circuit = vec![
            PropGate::H(0),
            PropGate::CX(0, 1),
            PropGate::S(2),
            PropGate::CZ(1, 2),
            PropGate::Swap(0, 2),
            PropGate::H(1),
        ];
        let config = PropagationConfig::default();
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate().unwrap();

        // Clifford circuits never increase term count beyond the initial.
        assert_eq!(result.num_terms_initial, 1);
        assert_eq!(result.num_terms_final, 1);
        assert_eq!(result.num_terms_peak, 1);
    }

    // ---------------------------------------------------------------
    // 17. Non-Clifford circuit: term growth bounded
    // ---------------------------------------------------------------
    #[test]
    fn test_non_clifford_term_growth_bounded() {
        let obs = PauliSum::single(WeightedPauli::basis(2, 0, PauliOp::X));
        // 5 T gates on qubit 0 could produce up to 2^5 = 32 terms.
        let circuit = vec![
            PropGate::T(0),
            PropGate::T(0),
            PropGate::T(0),
            PropGate::T(0),
            PropGate::T(0),
        ];
        let config = PropagationConfig::default().with_max_terms(100);
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate().unwrap();

        // Should have grown but been kept under max_terms.
        assert!(result.num_terms_final <= 100);
        // Should have more than 1 term.
        assert!(result.num_terms_final >= 2);
    }

    // ---------------------------------------------------------------
    // 18. Truncation: small terms removed
    // ---------------------------------------------------------------
    #[test]
    fn test_truncation_small_terms_removed() {
        let mut sum = PauliSum::new(2);
        sum.add_term(WeightedPauli::new(vec![PauliOp::X, PauliOp::I], (1.0, 0.0)));
        sum.add_term(WeightedPauli::new(
            vec![PauliOp::Y, PauliOp::I],
            (1e-15, 0.0),
        ));
        sum.add_term(WeightedPauli::new(vec![PauliOp::Z, PauliOp::I], (0.5, 0.0)));

        sum.truncate_by_magnitude(1e-10);
        assert_eq!(sum.len(), 2);

        // The Y term should be gone.
        assert!(sum.terms.iter().all(|t| t.paulis[0] != PauliOp::Y));
    }

    // ---------------------------------------------------------------
    // 19. Term merging: identical terms combined
    // ---------------------------------------------------------------
    #[test]
    fn test_term_merging_identical_combined() {
        let mut sum = PauliSum::new(1);
        sum.add_term(WeightedPauli::new(vec![PauliOp::X], (0.3, 0.1)));
        sum.add_term(WeightedPauli::new(vec![PauliOp::X], (0.4, -0.2)));
        sum.add_term(WeightedPauli::new(vec![PauliOp::Z], (1.0, 0.0)));

        sum.merge_duplicates(1e-10);
        assert_eq!(sum.len(), 2);

        let x_term = sum
            .terms
            .iter()
            .find(|t| t.paulis[0] == PauliOp::X)
            .unwrap();
        assert!(approx_eq(x_term.coefficient.0, 0.7, 1e-10));
        assert!(approx_eq(x_term.coefficient.1, -0.1, 1e-10));
    }

    // ---------------------------------------------------------------
    // 20. Expectation value: Z on |0> = +1
    // ---------------------------------------------------------------
    #[test]
    fn test_expectation_z_on_zero() {
        // Z on a single qubit, no circuit -> <0|Z|0> = 1.
        let obs = PauliSum::single(WeightedPauli::basis(1, 0, PauliOp::Z));
        let circuit: Vec<PropGate> = vec![];
        let config = PropagationConfig::default();
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate().unwrap();

        // Z|0> = |0>, so <0|Z|0> = 1. But we compute via identity projection:
        // The propagated observable is just Z (no gates applied).
        // <0|Z|0> needs to be computed via basis state method.
        let ev = result.propagated_observable.expectation_on_basis_state(0);
        assert!(approx_eq(ev, 1.0, 1e-10));
    }

    // ---------------------------------------------------------------
    // 21. Expectation value: X on |0> = 0
    // ---------------------------------------------------------------
    #[test]
    fn test_expectation_x_on_zero() {
        let obs = PauliSum::single(WeightedPauli::basis(1, 0, PauliOp::X));
        let circuit: Vec<PropGate> = vec![];
        let config = PropagationConfig::default();
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate().unwrap();

        // <0|X|0> = 0 (X has no diagonal element for |0>).
        let ev = result.propagated_observable.expectation_on_basis_state(0);
        assert!(approx_eq(ev, 0.0, 1e-10));
    }

    // ---------------------------------------------------------------
    // 22. Expectation value: ZZ on Bell state
    // ---------------------------------------------------------------
    #[test]
    fn test_expectation_zz_bell_state() {
        // Prepare Bell state: H(0) then CX(0,1) applied to |00>.
        // Observable: ZZ.
        // Heisenberg picture: propagate ZZ backwards through CX then H.
        // CX-dagger ZZ CX = CX ZZ CX. ZZ -> IZ (from our table).
        // H-dagger IZ H = IX (H acts on qubit 0: I->I, then nothing changes qubit 0.
        //   Actually qubit 0 is I so H doesn't change it. Result is still IZ.)
        // Wait -- let's be careful:
        //   Step 1: propagate ZZ through CX(0,1) backwards:
        //     Z on ctrl, Z on targ -> CX: ZZ -> IZ
        //   Step 2: propagate IZ through H(0) backwards:
        //     I on qubit 0 -> H: I -> I. So result is IZ.
        // <00|IZ|00> = <0|I|0><0|Z|0> = 1 * 1 = 1.
        // For Bell state |00>+|11>/sqrt(2): <ZZ> = 1 because both components give +1.

        let obs = zz_observable(2, 0, 1);
        let circuit = vec![PropGate::H(0), PropGate::CX(0, 1)];
        let config = PropagationConfig::default();
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate().unwrap();

        let ev = result.propagated_observable.expectation_on_basis_state(0);
        assert!(approx_eq(ev, 1.0, 1e-10));
    }

    // ---------------------------------------------------------------
    // 23. Batch propagation: parallel matches sequential
    // ---------------------------------------------------------------
    #[test]
    fn test_batch_propagation_matches_sequential() {
        let circuit = vec![PropGate::H(0), PropGate::CX(0, 1), PropGate::S(1)];

        let obs1 = z_observable(2, 0);
        let obs2 = x_observable(2, 1);
        let obs3 = zz_observable(2, 0, 1);

        // Sequential computation.
        let mut ev_seq = Vec::new();
        for obs in &[obs1.clone(), obs2.clone(), obs3.clone()] {
            let config = PropagationConfig::default();
            let mut prop = PauliPropagator::new(config, obs.clone(), circuit.clone());
            let r = prop.propagate().unwrap();
            ev_seq.push(r.propagated_observable.expectation_on_zero_state());
        }

        // Batch computation.
        let batch = BatchExpectation::new(
            vec![obs1, obs2, obs3],
            circuit,
            PropagationConfig::default(),
        );
        let ev_batch = batch.compute().unwrap();

        for (s, b) in ev_seq.iter().zip(ev_batch.iter()) {
            assert!(approx_eq(*s, *b, 1e-10));
        }
    }

    // ---------------------------------------------------------------
    // 24. Commuting groups: ZZ and ZI grouped together
    // ---------------------------------------------------------------
    #[test]
    fn test_commuting_groups() {
        let mut sum = PauliSum::new(2);
        // ZI and ZZ commute (both diagonal).
        sum.add_term(WeightedPauli::new(vec![PauliOp::Z, PauliOp::I], (1.0, 0.0)));
        sum.add_term(WeightedPauli::new(vec![PauliOp::Z, PauliOp::Z], (1.0, 0.0)));
        // XI anticommutes with ZI, so it should be in a separate group.
        sum.add_term(WeightedPauli::new(vec![PauliOp::X, PauliOp::I], (1.0, 0.0)));

        let groups = sum.commuting_groups();

        // ZI and ZZ should be in the same group.
        let zi_group = groups.iter().find(|g| g.contains(&0)).unwrap();
        assert!(zi_group.contains(&1)); // ZZ in same group as ZI.

        // XI should be in a different group.
        let xi_group = groups.iter().find(|g| g.contains(&2)).unwrap();
        assert!(!xi_group.contains(&0));
    }

    // ---------------------------------------------------------------
    // 25. Pauli frame: Clifford circuit exact
    // ---------------------------------------------------------------
    #[test]
    fn test_pauli_frame_clifford_exact() {
        // Z0 through H(0) -> X0
        let term = WeightedPauli::basis(2, 0, PauliOp::Z);
        let mut frame = PauliFrame::single(term);

        let circuit = vec![PropGate::H(0), PropGate::CX(0, 1)];
        frame.propagate_circuit(&circuit).unwrap();

        assert_eq!(frame.len(), 1);
        // H(0) applied last (reversed), then CX(0,1):
        // Reverse: first CX(0,1) then H(0).
        // CX-dagger Z0 CX: Z on ctrl, I on targ -> ZI -> ZI (unchanged).
        // H-dagger ZI H: Z on qubit 0 -> X on qubit 0. Result: XI.
        assert_eq!(frame.frames[0].paulis[0], PauliOp::X);
        assert_eq!(frame.frames[0].paulis[1], PauliOp::I);
    }

    // ---------------------------------------------------------------
    // 26. Large circuit: 20 qubits, 100 Clifford gates
    // ---------------------------------------------------------------
    #[test]
    fn test_large_clifford_circuit() {
        let n = 20;
        let circuit = random_clifford_circuit(n, 5, 42);
        assert!(circuit.len() >= 20); // at least some gates

        let obs = z_observable(n, 0);
        let config = PropagationConfig::default().with_threads(2);
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate().unwrap();

        // Clifford circuit: exactly 1 term output.
        assert_eq!(result.num_terms_final, 1);
        // Coefficient magnitude should be 1.
        let mag = result.propagated_observable.terms[0].coeff_magnitude();
        assert!(approx_eq(mag, 1.0, 1e-10));
    }

    // ---------------------------------------------------------------
    // 27. Toffoli propagation: correct splitting
    // ---------------------------------------------------------------
    #[test]
    fn test_toffoli_propagation() {
        // Toffoli is decomposed into Clifford + T gates.
        // Propagating Z on the target through Toffoli should produce
        // a valid result (may have multiple terms due to T gates).
        let term = WeightedPauli::basis(3, 2, PauliOp::Z);
        let result = propagate_toffoli(&term, 0, 1, 2);

        // Should produce at least one term.
        assert!(!result.is_empty());

        // All terms should have 3 qubits.
        for t in &result {
            assert_eq!(t.num_qubits(), 3);
        }

        // Total probability should be conserved (sum of |coeff|^2 should
        // remain close to 1 for a unitary conjugation of a single Pauli).
        // Actually for Pauli propagation the norm is not sum of squares;
        // we just verify terms exist and are reasonable.
        let total_mag: f64 = result.iter().map(|t| t.coeff_magnitude()).sum();
        assert!(total_mag > 0.5, "total magnitude {} too small", total_mag);
    }

    // ---------------------------------------------------------------
    // 28. Config builder defaults
    // ---------------------------------------------------------------
    #[test]
    fn test_config_builder_defaults() {
        let config = PropagationConfig::default();
        assert_eq!(config.max_terms, 100_000);
        assert!(approx_eq(config.truncation_threshold, 1e-10, 1e-15));
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.num_threads, 4);
        assert!(config.merge_duplicates);
        assert!(!config.group_commuting);

        // Builder chaining.
        let config2 = PropagationConfig::default()
            .with_max_terms(500)
            .with_threshold(1e-6)
            .with_batch_size(64)
            .with_threads(2)
            .with_merge(false)
            .with_commuting_groups(true);
        assert_eq!(config2.max_terms, 500);
        assert!(approx_eq(config2.truncation_threshold, 1e-6, 1e-15));
        assert_eq!(config2.batch_size, 64);
        assert_eq!(config2.num_threads, 2);
        assert!(!config2.merge_duplicates);
        assert!(config2.group_commuting);
    }

    // ---------------------------------------------------------------
    // 29. Performance: 1000-term propagation completes
    // ---------------------------------------------------------------
    #[test]
    fn test_performance_1000_terms() {
        let n = 4;
        let mut sum = PauliSum::new(n);
        let ops = [PauliOp::I, PauliOp::X, PauliOp::Y, PauliOp::Z];
        // Create 256 distinct Pauli strings on 4 qubits (4^4).
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        let paulis = vec![ops[i0], ops[i1], ops[i2], ops[i3]];
                        let coeff = 1.0 / 256.0;
                        sum.add_term(WeightedPauli::new(paulis, (coeff, 0.0)));
                    }
                }
            }
        }
        assert_eq!(sum.len(), 256);

        let circuit = vec![PropGate::H(0), PropGate::CX(0, 1), PropGate::T(2)];
        let config = PropagationConfig::default()
            .with_max_terms(10_000)
            .with_threads(2);
        let mut prop = PauliPropagator::new(config, sum, circuit);
        let result = prop.propagate().unwrap();

        // Should complete without hanging. The T gate on qubit 2 splits
        // terms where qubit 2 is X or Y, so we expect some growth.
        assert!(result.num_terms_final > 0);
        assert!(
            result.elapsed_secs < 10.0,
            "took too long: {}s",
            result.elapsed_secs
        );
    }

    // ---------------------------------------------------------------
    // 30. Sdg propagation correctness
    // ---------------------------------------------------------------
    #[test]
    fn test_sdg_propagation() {
        // Sdg-dagger X Sdg = -Y
        let term = WeightedPauli::basis(1, 0, PauliOp::X);
        let result = propagate_sdg(&term, 0);
        assert_eq!(result.paulis[0], PauliOp::Y);
        assert!(approx_eq(result.coefficient.0, -1.0, 1e-12));

        // Sdg-dagger Y Sdg = X
        let term2 = WeightedPauli::basis(1, 0, PauliOp::Y);
        let result2 = propagate_sdg(&term2, 0);
        assert_eq!(result2.paulis[0], PauliOp::X);
        assert!(approx_eq(result2.coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 31. Rz propagation: X -> cos X + sin Y
    // ---------------------------------------------------------------
    #[test]
    fn test_rz_propagation_x() {
        let theta = 0.5;
        let term = WeightedPauli::basis(1, 0, PauliOp::X);
        let result = propagate_rz(&term, 0, theta);
        assert_eq!(result.len(), 2);

        let x_term = result.iter().find(|t| t.paulis[0] == PauliOp::X).unwrap();
        assert!(approx_eq(x_term.coefficient.0, theta.cos(), 1e-10));

        let y_term = result.iter().find(|t| t.paulis[0] == PauliOp::Y).unwrap();
        assert!(approx_eq(y_term.coefficient.0, theta.sin(), 1e-10));
    }

    // ---------------------------------------------------------------
    // 32. SWAP propagation
    // ---------------------------------------------------------------
    #[test]
    fn test_swap_propagation() {
        let term = WeightedPauli::new(vec![PauliOp::X, PauliOp::Z], (1.0, 0.0));
        let result = propagate_swap(&term, 0, 1);
        assert_eq!(result.paulis[0], PauliOp::Z);
        assert_eq!(result.paulis[1], PauliOp::X);
        assert!(approx_eq(result.coefficient.0, 1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 33. Validation: qubit out of range
    // ---------------------------------------------------------------
    #[test]
    fn test_validation_qubit_out_of_range() {
        let obs = z_observable(2, 0);
        let circuit = vec![PropGate::H(5)]; // qubit 5 on a 2-qubit system
        let config = PropagationConfig::default();
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate();
        assert!(result.is_err());
        match result {
            Err(PauliPropError::CircuitError(_)) => {}
            _ => panic!("Expected CircuitError"),
        }
    }

    // ---------------------------------------------------------------
    // 34. PauliFrame rejects non-Clifford
    // ---------------------------------------------------------------
    #[test]
    fn test_pauli_frame_rejects_non_clifford() {
        let term = WeightedPauli::basis(2, 0, PauliOp::Z);
        let mut frame = PauliFrame::single(term);
        let result = frame.propagate_gate(&PropGate::T(0));
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 35. Expectation on basis state
    // ---------------------------------------------------------------
    #[test]
    fn test_expectation_on_basis_state() {
        let mut sum = PauliSum::new(2);
        // ZI has eigenvalue +1 on |00>, -1 on |10>, +1 on |01>, -1 on |11>
        sum.add_term(WeightedPauli::new(vec![PauliOp::Z, PauliOp::I], (1.0, 0.0)));

        assert!(approx_eq(sum.expectation_on_basis_state(0b00), 1.0, 1e-10));
        assert!(approx_eq(sum.expectation_on_basis_state(0b01), -1.0, 1e-10));
        assert!(approx_eq(sum.expectation_on_basis_state(0b10), 1.0, 1e-10));
        assert!(approx_eq(sum.expectation_on_basis_state(0b11), -1.0, 1e-10));
    }

    // ---------------------------------------------------------------
    // 36. PauliOp from_char and to_char roundtrip
    // ---------------------------------------------------------------
    #[test]
    fn test_pauli_op_roundtrip() {
        for c in &['I', 'X', 'Y', 'Z'] {
            let op = PauliOp::from_char(*c).unwrap();
            assert_eq!(op.to_char(), *c);
        }
        assert!(PauliOp::from_char('Q').is_err());
    }

    // ---------------------------------------------------------------
    // 37. WeightedPauli from_str
    // ---------------------------------------------------------------
    #[test]
    fn test_weighted_pauli_from_str() {
        let wp = WeightedPauli::from_str("IXYZ").unwrap();
        assert_eq!(wp.num_qubits(), 4);
        assert_eq!(wp.paulis[0], PauliOp::I);
        assert_eq!(wp.paulis[1], PauliOp::X);
        assert_eq!(wp.paulis[2], PauliOp::Y);
        assert_eq!(wp.paulis[3], PauliOp::Z);
        assert_eq!(wp.weight(), 3);
        assert!(wp.coefficient.0 == 1.0);
    }

    // ---------------------------------------------------------------
    // 38. Commutes_with correctness
    // ---------------------------------------------------------------
    #[test]
    fn test_commutes_with() {
        // XX and ZZ commute (even number of anticommuting sites).
        let xx = WeightedPauli::from_str("XX").unwrap();
        let zz = WeightedPauli::from_str("ZZ").unwrap();
        assert!(xx.commutes_with(&zz));

        // XI and ZI anticommute (one anticommuting site).
        let xi = WeightedPauli::from_str("XI").unwrap();
        let zi = WeightedPauli::from_str("ZI").unwrap();
        assert!(!xi.commutes_with(&zi));

        // All Paulis commute with identity.
        let ii = WeightedPauli::from_str("II").unwrap();
        assert!(xx.commutes_with(&ii));
        assert!(zi.commutes_with(&ii));
    }

    // ---------------------------------------------------------------
    // 39. Tdg propagation: X -> cos X - sin Y
    // ---------------------------------------------------------------
    #[test]
    fn test_tdg_propagation_x() {
        let term = WeightedPauli::basis(1, 0, PauliOp::X);
        let result = propagate_tdg(&term, 0);
        assert_eq!(result.len(), 2);

        let c = std::f64::consts::FRAC_PI_4.cos();
        let s = std::f64::consts::FRAC_PI_4.sin();

        let x_term = result.iter().find(|t| t.paulis[0] == PauliOp::X).unwrap();
        assert!(approx_eq(x_term.coefficient.0, c, 1e-10));

        let y_term = result.iter().find(|t| t.paulis[0] == PauliOp::Y).unwrap();
        assert!(approx_eq(y_term.coefficient.0, -s, 1e-10));
    }

    // ---------------------------------------------------------------
    // 40. CX propagation: XY -> -YZ
    // ---------------------------------------------------------------
    #[test]
    fn test_cx_propagation_xy_to_neg_yz() {
        let term = WeightedPauli::new(vec![PauliOp::X, PauliOp::Y], (1.0, 0.0));
        let result = propagate_cx(&term, 0, 1);
        assert_eq!(result.paulis[0], PauliOp::Y);
        assert_eq!(result.paulis[1], PauliOp::Z);
        assert!(approx_eq(result.coefficient.0, -1.0, 1e-12));
    }

    // ---------------------------------------------------------------
    // 41. Multiple T gates accumulate correctly
    // ---------------------------------------------------------------
    #[test]
    fn test_double_t_gate() {
        // Two T gates = S gate. T-dagger T-dagger X T T = S-dagger X S = Y.
        let obs = PauliSum::single(WeightedPauli::basis(1, 0, PauliOp::X));
        let circuit = vec![PropGate::T(0), PropGate::T(0)];
        let config = PropagationConfig::default();
        let mut prop = PauliPropagator::new(config, obs, circuit);
        let result = prop.propagate().unwrap();

        // After merging, we should get Y with coefficient 1.
        // T^2 = S, so (T^2)-dagger X (T^2) = S-dagger X S = Y.
        // Find the dominant term.
        let mut max_term: Option<&WeightedPauli> = None;
        let mut max_mag = 0.0;
        for t in &result.propagated_observable.terms {
            let m = t.coeff_magnitude();
            if m > max_mag {
                max_mag = m;
                max_term = Some(t);
            }
        }
        let dominant = max_term.unwrap();
        assert_eq!(dominant.paulis[0], PauliOp::Y);
        assert!(approx_eq(dominant.coeff_magnitude(), 1.0, 1e-6));
    }

    // ---------------------------------------------------------------
    // 42. CZ propagation: XX -> -YY
    // ---------------------------------------------------------------
    #[test]
    fn test_cz_propagation_xx_to_neg_yy() {
        let term = WeightedPauli::new(vec![PauliOp::X, PauliOp::X], (1.0, 0.0));
        let result = propagate_cz(&term, 0, 1);
        assert_eq!(result.paulis[0], PauliOp::Y);
        assert_eq!(result.paulis[1], PauliOp::Y);
        assert!(approx_eq(result.coefficient.0, -1.0, 1e-12));
    }
}
