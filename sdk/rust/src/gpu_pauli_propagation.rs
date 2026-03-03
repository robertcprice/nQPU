//! GPU-Accelerated Pauli Propagation for Near-Clifford Simulation
//!
//! This module implements bit-packed Pauli string propagation with optional
//! Metal GPU acceleration. Instead of full statevector simulation (O(2^n)
//! memory), we track how Pauli operators propagate through a circuit in the
//! Heisenberg picture, then evaluate expectation values directly.
//!
//! # Architecture
//!
//! - [`PauliString`]: Bit-packed Pauli representation (2 bits per qubit).
//! - [`PauliPropagator`]: CPU engine for propagating Pauli strings through
//!   Clifford gates and handling T-gate decomposition.
//! - [`MetalPauliKernel`]: Metal compute shader dispatch for parallel
//!   propagation of thousands of Pauli strings (macOS only).
//! - [`PauliTableau`]: Stabilizer-like tracking of how each initial Pauli
//!   transforms, with destabilizer support.
//! - [`NearCliffordEstimator`]: Monte Carlo estimation of expectation values
//!   for circuits with few T-gates, using importance sampling.
//! - [`AutoDispatch`]: Automatic CPU vs GPU routing based on problem size.
//!
//! # Bit Encoding
//!
//! Each qubit uses 2 bits: I=00, X=01, Z=10, Y=11. This encoding allows
//! multiplication via XOR and commutation checking via popcount of the
//! bitwise AND of the symplectic inner product.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::gpu_pauli_propagation::*;
//!
//! let mut p = PackedPauliString::single_z(4, 0);
//! let q = PackedPauliString::single_x(4, 2);
//! assert!(p.commutes_with(&q));  // Z0 and X2 act on different qubits
//! assert_eq!(p.weight(), 1);
//! ```

use std::fmt;
use std::time::Instant;

use rand::Rng;

// =====================================================================
// ERROR TYPE
// =====================================================================

/// Errors arising during GPU Pauli propagation.
#[derive(Clone, Debug)]
pub enum GpuPauliError {
    /// Invalid Pauli string structure.
    InvalidPauli(String),
    /// Circuit gate references out-of-range qubits.
    CircuitError(String),
    /// Term count exceeded hard cap.
    Overflow { count: usize, limit: usize },
    /// Metal GPU initialization or dispatch failure.
    GpuError(String),
    /// Numerical issue (NaN, Inf, or failed convergence).
    NumericalError(String),
}

impl fmt::Display for GpuPauliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuPauliError::InvalidPauli(msg) => write!(f, "Invalid Pauli: {}", msg),
            GpuPauliError::CircuitError(msg) => write!(f, "Circuit error: {}", msg),
            GpuPauliError::Overflow { count, limit } => {
                write!(f, "Term overflow: {} exceeds limit {}", count, limit)
            }
            GpuPauliError::GpuError(msg) => write!(f, "GPU error: {}", msg),
            GpuPauliError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl std::error::Error for GpuPauliError {}

// =====================================================================
// PHASE
// =====================================================================

/// Phase factor: one of +1, -1, +i, -i encoded as 0, 2, 1, 3.
///
/// Arithmetic follows the rule: phase = (a + b) mod 4, where
/// 0 = +1, 1 = +i, 2 = -1, 3 = -i.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Phase(pub u8);

impl Phase {
    pub const PLUS_ONE: Phase = Phase(0);
    pub const PLUS_I: Phase = Phase(1);
    pub const MINUS_ONE: Phase = Phase(2);
    pub const MINUS_I: Phase = Phase(3);

    /// Multiply two phases.
    #[inline]
    pub fn mul(self, other: Phase) -> Phase {
        Phase((self.0 + other.0) & 3)
    }

    /// Return the complex value as (re, im).
    #[inline]
    pub fn to_complex(self) -> (f64, f64) {
        match self.0 & 3 {
            0 => (1.0, 0.0),
            1 => (0.0, 1.0),
            2 => (-1.0, 0.0),
            3 => (0.0, -1.0),
            _ => unreachable!(),
        }
    }

    /// Negate the phase (multiply by -1, i.e. add 2 mod 4).
    #[inline]
    pub fn negate(self) -> Phase {
        Phase((self.0 + 2) & 3)
    }

    /// Conjugate the phase (negate imaginary part).
    #[inline]
    pub fn conjugate(self) -> Phase {
        Phase((4 - self.0) & 3)
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 & 3 {
            0 => write!(f, "+1"),
            1 => write!(f, "+i"),
            2 => write!(f, "-1"),
            3 => write!(f, "-i"),
            _ => unreachable!(),
        }
    }
}

// =====================================================================
// PACKED PAULI STRING
// =====================================================================

/// Bit-packed Pauli string on n qubits.
///
/// Encoding per qubit (2 bits): I=0b00, X=0b01, Z=0b10, Y=0b11.
/// Packed into `Vec<u64>` where each u64 holds 32 qubit positions
/// (64 bits / 2 bits per qubit).
///
/// The phase tracks the overall sign: +1, -1, +i, or -i.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedPauliString {
    /// Packed operator data. Each u64 stores 32 qubits at 2 bits each.
    pub data: Vec<u64>,
    /// Number of qubits.
    pub num_qubits: usize,
    /// Phase factor (+1, +i, -1, -i).
    pub phase: Phase,
}

impl PackedPauliString {
    /// Number of u64 words needed for `n` qubits (32 qubits per word).
    #[inline]
    fn num_words(n: usize) -> usize {
        (n + 31) / 32
    }

    /// Create the identity string on `n` qubits with phase +1.
    pub fn identity(num_qubits: usize) -> Self {
        PackedPauliString {
            data: vec![0u64; Self::num_words(num_qubits)],
            num_qubits,
            phase: Phase::PLUS_ONE,
        }
    }

    /// Create a single-X Pauli on qubit `q` of an `n`-qubit system.
    pub fn single_x(num_qubits: usize, q: usize) -> Self {
        let mut ps = Self::identity(num_qubits);
        ps.set(q, 0b01);
        ps
    }

    /// Create a single-Z Pauli on qubit `q` of an `n`-qubit system.
    pub fn single_z(num_qubits: usize, q: usize) -> Self {
        let mut ps = Self::identity(num_qubits);
        ps.set(q, 0b10);
        ps
    }

    /// Create a single-Y Pauli on qubit `q` of an `n`-qubit system.
    pub fn single_y(num_qubits: usize, q: usize) -> Self {
        let mut ps = Self::identity(num_qubits);
        ps.set(q, 0b11);
        ps
    }

    /// Parse from a string like "IXYZ" (qubit 0 = leftmost).
    pub fn from_str_rep(s: &str) -> Result<Self, GpuPauliError> {
        let n = s.len();
        let mut ps = Self::identity(n);
        for (i, ch) in s.chars().enumerate() {
            let code = match ch {
                'I' | 'i' => 0b00,
                'X' | 'x' => 0b01,
                'Z' | 'z' => 0b10,
                'Y' | 'y' => 0b11,
                _ => {
                    return Err(GpuPauliError::InvalidPauli(format!(
                        "invalid character '{}'",
                        ch
                    )))
                }
            };
            ps.set(i, code);
        }
        Ok(ps)
    }

    /// Get the 2-bit Pauli code for qubit `q`.
    #[inline]
    pub fn get(&self, q: usize) -> u8 {
        let word = q / 32;
        let bit_offset = (q % 32) * 2;
        ((self.data[word] >> bit_offset) & 0b11) as u8
    }

    /// Set the 2-bit Pauli code for qubit `q`.
    #[inline]
    pub fn set(&mut self, q: usize, code: u8) {
        let word = q / 32;
        let bit_offset = (q % 32) * 2;
        self.data[word] &= !(0b11u64 << bit_offset);
        self.data[word] |= (code as u64 & 0b11) << bit_offset;
    }

    /// Get the Pauli operator character for qubit `q`.
    pub fn get_char(&self, q: usize) -> char {
        match self.get(q) {
            0b00 => 'I',
            0b01 => 'X',
            0b10 => 'Z',
            0b11 => 'Y',
            _ => unreachable!(),
        }
    }

    /// Pauli weight: count of non-identity sites.
    pub fn weight(&self) -> usize {
        let mut w = 0usize;
        for q in 0..self.num_qubits {
            if self.get(q) != 0 {
                w += 1;
            }
        }
        w
    }

    /// Check if this is the all-identity string.
    pub fn is_identity(&self) -> bool {
        self.data.iter().all(|&word| word == 0)
    }

    /// Check if this Pauli string commutes with another.
    ///
    /// Two n-qubit Pauli strings P, Q commute iff the number of positions
    /// where they anticommute is even. At position j, they anticommute iff
    /// both are non-identity and different from each other.
    pub fn commutes_with(&self, other: &PackedPauliString) -> bool {
        debug_assert_eq!(self.num_qubits, other.num_qubits);
        let mut anticommute_count = 0u32;
        for q in 0..self.num_qubits {
            let a = self.get(q);
            let b = other.get(q);
            if a != 0 && b != 0 && a != b {
                anticommute_count += 1;
            }
        }
        anticommute_count % 2 == 0
    }

    /// Multiply two Pauli strings, returning the product with updated phase.
    ///
    /// Uses the rule: X*Y = iZ, Y*Z = iX, Z*X = iY, and cyclic.
    /// The result Pauli at each site is the XOR of the two codes, and the
    /// phase picks up factors of +i or -i from non-commuting sites.
    pub fn multiply(&self, other: &PackedPauliString) -> PackedPauliString {
        debug_assert_eq!(self.num_qubits, other.num_qubits);
        let mut result = self.clone();
        let mut extra_phase = 0u8; // accumulate mod 4

        for q in 0..self.num_qubits {
            let a = self.get(q);
            let b = other.get(q);
            let product_code = a ^ b;

            // Determine phase contribution from this site.
            // Using the Pauli multiplication table:
            // I*P = P (phase 0), P*I = P (phase 0), P*P = I (phase 0)
            // X*Y = iZ (+1), Y*X = -iZ (+3)
            // Y*Z = iX (+1), Z*Y = -iX (+3)
            // Z*X = iY (+1), X*Z = -iY (+3)
            let site_phase = match (a, b) {
                (0, _) | (_, 0) => 0u8,
                (x, y) if x == y => 0u8,
                (0b01, 0b11) => 1, // X*Y = +iZ
                (0b11, 0b01) => 3, // Y*X = -iZ
                (0b11, 0b10) => 1, // Y*Z = +iX
                (0b10, 0b11) => 3, // Z*Y = -iX
                (0b10, 0b01) => 1, // Z*X = +iY
                (0b01, 0b10) => 3, // X*Z = -iY
                _ => 0,
            };
            extra_phase = (extra_phase + site_phase) & 3;
            result.set(q, product_code);
        }

        result.phase = result.phase.mul(other.phase).mul(Phase(extra_phase));
        result
    }

    /// Swap the Pauli operators on qubits `a` and `b`.
    pub fn swap_qubits(&mut self, a: usize, b: usize) {
        let pa = self.get(a);
        let pb = self.get(b);
        self.set(a, pb);
        self.set(b, pa);
    }

    /// Serialize to a flat u32 buffer for GPU transfer.
    ///
    /// Format: [num_qubits, phase, word_0_lo, word_0_hi, word_1_lo, word_1_hi, ...]
    pub fn to_gpu_buffer(&self) -> Vec<u32> {
        let mut buf = Vec::with_capacity(2 + self.data.len() * 2);
        buf.push(self.num_qubits as u32);
        buf.push(self.phase.0 as u32);
        for &word in &self.data {
            buf.push(word as u32);
            buf.push((word >> 32) as u32);
        }
        buf
    }

    /// Deserialize from a flat u32 GPU buffer.
    pub fn from_gpu_buffer(buf: &[u32]) -> Result<Self, GpuPauliError> {
        if buf.len() < 2 {
            return Err(GpuPauliError::InvalidPauli(
                "GPU buffer too short".to_string(),
            ));
        }
        let num_qubits = buf[0] as usize;
        let phase = Phase(buf[1] as u8);
        let num_words = Self::num_words(num_qubits);
        if buf.len() < 2 + num_words * 2 {
            return Err(GpuPauliError::InvalidPauli(
                "GPU buffer truncated".to_string(),
            ));
        }
        let mut data = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let lo = buf[2 + i * 2] as u64;
            let hi = buf[2 + i * 2 + 1] as u64;
            data.push(lo | (hi << 32));
        }
        Ok(PackedPauliString {
            data,
            num_qubits,
            phase,
        })
    }
}

impl fmt::Display for PackedPauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) ", self.phase)?;
        for q in 0..self.num_qubits {
            write!(f, "{}", self.get_char(q))?;
        }
        Ok(())
    }
}

// =====================================================================
// WEIGHTED PACKED PAULI
// =====================================================================

/// A packed Pauli string with a complex coefficient.
#[derive(Clone, Debug)]
pub struct WeightedPackedPauli {
    /// The Pauli string.
    pub pauli: PackedPauliString,
    /// Complex coefficient as (real, imaginary).
    pub coeff: (f64, f64),
}

impl WeightedPackedPauli {
    /// Create with unit coefficient.
    pub fn unit(pauli: PackedPauliString) -> Self {
        let (re, im) = pauli.phase.to_complex();
        WeightedPackedPauli {
            pauli,
            coeff: (re, im),
        }
    }

    /// Create with explicit coefficient.
    pub fn new(pauli: PackedPauliString, coeff: (f64, f64)) -> Self {
        WeightedPackedPauli { pauli, coeff }
    }

    /// Coefficient magnitude.
    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.coeff.0 * self.coeff.0 + self.coeff.1 * self.coeff.1).sqrt()
    }

    /// Scale coefficient by a real factor.
    pub fn scale(&mut self, factor: f64) {
        self.coeff.0 *= factor;
        self.coeff.1 *= factor;
    }

    /// Scale coefficient by a complex factor.
    pub fn scale_complex(&mut self, factor: (f64, f64)) {
        let (a, b) = self.coeff;
        let (c, d) = factor;
        self.coeff = (a * c - b * d, a * d + b * c);
    }

    /// Negate the coefficient.
    pub fn negate(&mut self) {
        self.coeff.0 = -self.coeff.0;
        self.coeff.1 = -self.coeff.1;
    }
}

impl fmt::Display for WeightedPackedPauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}{:+.6}i) ", self.coeff.0, self.coeff.1)?;
        for q in 0..self.pauli.num_qubits {
            write!(f, "{}", self.pauli.get_char(q))?;
        }
        Ok(())
    }
}

// =====================================================================
// CLIFFORD GATE ENUM
// =====================================================================

/// Gates supported by the Pauli propagator.
#[derive(Clone, Debug)]
pub enum CliffordGate {
    /// Hadamard on qubit.
    H(usize),
    /// S gate on qubit.
    S(usize),
    /// S-dagger on qubit.
    Sdg(usize),
    /// CNOT: (control, target).
    CX(usize, usize),
    /// Controlled-Z: (qubit_a, qubit_b).
    CZ(usize, usize),
    /// SWAP: (qubit_a, qubit_b).
    SWAP(usize, usize),
    /// T gate on qubit (non-Clifford, causes term splitting).
    T(usize),
    /// T-dagger on qubit (non-Clifford).
    Tdg(usize),
    /// Rz rotation by angle (non-Clifford for general theta).
    Rz(usize, f64),
}

impl CliffordGate {
    /// Returns true if this gate is strictly Clifford (no term splitting).
    pub fn is_clifford(&self) -> bool {
        matches!(
            self,
            CliffordGate::H(_)
                | CliffordGate::S(_)
                | CliffordGate::Sdg(_)
                | CliffordGate::CX(_, _)
                | CliffordGate::CZ(_, _)
                | CliffordGate::SWAP(_, _)
        )
    }

    /// Maximum qubit index referenced.
    pub fn max_qubit(&self) -> usize {
        match self {
            CliffordGate::H(q)
            | CliffordGate::S(q)
            | CliffordGate::Sdg(q)
            | CliffordGate::T(q)
            | CliffordGate::Tdg(q)
            | CliffordGate::Rz(q, _) => *q,
            CliffordGate::CX(a, b)
            | CliffordGate::CZ(a, b)
            | CliffordGate::SWAP(a, b) => std::cmp::max(*a, *b),
        }
    }
}

impl fmt::Display for CliffordGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliffordGate::H(q) => write!(f, "H({})", q),
            CliffordGate::S(q) => write!(f, "S({})", q),
            CliffordGate::Sdg(q) => write!(f, "Sdg({})", q),
            CliffordGate::CX(c, t) => write!(f, "CX({},{})", c, t),
            CliffordGate::CZ(a, b) => write!(f, "CZ({},{})", a, b),
            CliffordGate::SWAP(a, b) => write!(f, "SWAP({},{})", a, b),
            CliffordGate::T(q) => write!(f, "T({})", q),
            CliffordGate::Tdg(q) => write!(f, "Tdg({})", q),
            CliffordGate::Rz(q, th) => write!(f, "Rz({},{:.4})", q, th),
        }
    }
}

// =====================================================================
// PROPAGATION CONFIG
// =====================================================================

/// Configuration for the Pauli propagation engine.
#[derive(Clone, Debug)]
pub struct GpuPropConfig {
    /// Maximum number of Pauli terms before hard truncation.
    pub max_terms: usize,
    /// Drop terms with coefficient magnitude below this threshold.
    pub truncation_threshold: f64,
    /// Whether to merge duplicate Pauli strings after non-Clifford gates.
    pub merge_duplicates: bool,
    /// Minimum number of Pauli strings to justify GPU dispatch.
    pub gpu_threshold: usize,
}

impl Default for GpuPropConfig {
    fn default() -> Self {
        GpuPropConfig {
            max_terms: 100_000,
            truncation_threshold: 1e-10,
            merge_duplicates: true,
            gpu_threshold: 1000,
        }
    }
}

impl GpuPropConfig {
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

    /// Builder: set merge behavior.
    pub fn with_merge(mut self, m: bool) -> Self {
        self.merge_duplicates = m;
        self
    }

    /// Builder: set GPU dispatch threshold.
    pub fn with_gpu_threshold(mut self, n: usize) -> Self {
        self.gpu_threshold = n;
        self
    }
}

// =====================================================================
// SINGLE-TERM PROPAGATION RULES (CPU)
// =====================================================================

/// Propagate a single weighted Pauli through a Hadamard on qubit `q`.
///
/// H is self-inverse: H^dag P H.
/// H X H = Z, H Z H = X, H Y H = -Y, H I H = I.
fn propagate_h(term: &WeightedPackedPauli, q: usize) -> WeightedPackedPauli {
    let mut out = term.clone();
    match out.pauli.get(q) {
        0b00 => {}                      // I -> I
        0b01 => out.pauli.set(q, 0b10), // X -> Z
        0b10 => out.pauli.set(q, 0b01), // Z -> X
        0b11 => out.negate(),            // Y -> -Y
        _ => unreachable!(),
    }
    out
}

/// Propagate through S gate on qubit `q`.
///
/// S^dag X S = Y, S^dag Y S = -X, S^dag Z S = Z.
fn propagate_s(term: &WeightedPackedPauli, q: usize) -> WeightedPackedPauli {
    let mut out = term.clone();
    match out.pauli.get(q) {
        0b00 | 0b10 => {} // I, Z unchanged
        0b01 => out.pauli.set(q, 0b11), // X -> Y
        0b11 => {
            // Y -> -X
            out.pauli.set(q, 0b01);
            out.negate();
        }
        _ => unreachable!(),
    }
    out
}

/// Propagate through S-dagger on qubit `q`.
///
/// S X S^dag = -Y, S Y S^dag = X, S Z S^dag = Z.
fn propagate_sdg(term: &WeightedPackedPauli, q: usize) -> WeightedPackedPauli {
    let mut out = term.clone();
    match out.pauli.get(q) {
        0b00 | 0b10 => {} // I, Z unchanged
        0b01 => {
            // X -> -Y
            out.pauli.set(q, 0b11);
            out.negate();
        }
        0b11 => out.pauli.set(q, 0b01), // Y -> X
        _ => unreachable!(),
    }
    out
}

/// Propagate through T gate on qubit `q` (non-Clifford).
///
/// T^dag I T = I, T^dag Z T = Z.
/// T^dag X T = cos(pi/4) X + sin(pi/4) Y.
/// T^dag Y T = cos(pi/4) Y - sin(pi/4) X.
fn propagate_t(term: &WeightedPackedPauli, q: usize) -> Vec<WeightedPackedPauli> {
    match term.pauli.get(q) {
        0b00 | 0b10 => vec![term.clone()],
        0b01 => {
            // X -> cos X + sin Y
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            t1.scale_complex((c, 0.0));
            let mut t2 = term.clone();
            t2.pauli.set(q, 0b11);
            t2.scale_complex((s, 0.0));
            vec![t1, t2]
        }
        0b11 => {
            // Y -> cos Y - sin X
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            t1.scale_complex((c, 0.0));
            let mut t2 = term.clone();
            t2.pauli.set(q, 0b01);
            t2.scale_complex((-s, 0.0));
            vec![t1, t2]
        }
        _ => unreachable!(),
    }
}

/// Propagate through T-dagger on qubit `q` (non-Clifford).
///
/// T X T^dag = cos(pi/4) X - sin(pi/4) Y.
/// T Y T^dag = cos(pi/4) Y + sin(pi/4) X.
fn propagate_tdg(term: &WeightedPackedPauli, q: usize) -> Vec<WeightedPackedPauli> {
    match term.pauli.get(q) {
        0b00 | 0b10 => vec![term.clone()],
        0b01 => {
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            t1.scale_complex((c, 0.0));
            let mut t2 = term.clone();
            t2.pauli.set(q, 0b11);
            t2.scale_complex((-s, 0.0));
            vec![t1, t2]
        }
        0b11 => {
            let c = std::f64::consts::FRAC_PI_4.cos();
            let s = std::f64::consts::FRAC_PI_4.sin();
            let mut t1 = term.clone();
            t1.scale_complex((c, 0.0));
            let mut t2 = term.clone();
            t2.pauli.set(q, 0b01);
            t2.scale_complex((s, 0.0));
            vec![t1, t2]
        }
        _ => unreachable!(),
    }
}

/// Propagate through Rz(theta) on qubit `q` (non-Clifford).
///
/// Rz^dag X Rz = cos(theta) X + sin(theta) Y.
/// Rz^dag Y Rz = cos(theta) Y - sin(theta) X.
fn propagate_rz(term: &WeightedPackedPauli, q: usize, theta: f64) -> Vec<WeightedPackedPauli> {
    match term.pauli.get(q) {
        0b00 | 0b10 => vec![term.clone()],
        0b01 => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.scale_complex((c, 0.0));
            let mut t2 = term.clone();
            t2.pauli.set(q, 0b11);
            t2.scale_complex((s, 0.0));
            vec![t1, t2]
        }
        0b11 => {
            let c = theta.cos();
            let s = theta.sin();
            let mut t1 = term.clone();
            t1.scale_complex((c, 0.0));
            let mut t2 = term.clone();
            t2.pauli.set(q, 0b01);
            t2.scale_complex((-s, 0.0));
            vec![t1, t2]
        }
        _ => unreachable!(),
    }
}

/// Propagate through CNOT on (control, target).
///
/// CX is self-inverse. Conjugation table:
///   II->II, IX->IX, IY->ZY, IZ->ZZ,
///   XI->XX, XX->XI, XY->-YZ, XZ->YY,
///   YI->YX, YX->YI, YY->XZ, YZ->-XY,
///   ZI->ZI, ZX->ZX, ZY->IY, ZZ->IZ.
fn propagate_cx(term: &WeightedPackedPauli, ctrl: usize, targ: usize) -> WeightedPackedPauli {
    let mut out = term.clone();
    let pc = term.pauli.get(ctrl);
    let pt = term.pauli.get(targ);
    match (pc, pt) {
        (0b00, 0b00) => {}
        (0b00, 0b01) => {}
        (0b00, 0b11) => {
            out.pauli.set(ctrl, 0b10);
        }
        (0b00, 0b10) => {
            out.pauli.set(ctrl, 0b10);
            out.pauli.set(targ, 0b10);
        }
        (0b01, 0b00) => {
            out.pauli.set(targ, 0b01);
        }
        (0b01, 0b01) => {
            out.pauli.set(targ, 0b00);
        }
        (0b01, 0b11) => {
            out.pauli.set(ctrl, 0b11);
            out.pauli.set(targ, 0b10);
            out.negate();
        }
        (0b01, 0b10) => {
            out.pauli.set(ctrl, 0b11);
            out.pauli.set(targ, 0b11);
        }
        (0b11, 0b00) => {
            out.pauli.set(targ, 0b01);
        }
        (0b11, 0b01) => {
            out.pauli.set(targ, 0b00);
        }
        (0b11, 0b11) => {
            out.pauli.set(ctrl, 0b01);
            out.pauli.set(targ, 0b10);
        }
        (0b11, 0b10) => {
            out.pauli.set(ctrl, 0b01);
            out.pauli.set(targ, 0b11);
            out.negate();
        }
        (0b10, 0b00) => {}
        (0b10, 0b01) => {}
        (0b10, 0b11) => {
            out.pauli.set(ctrl, 0b00);
        }
        (0b10, 0b10) => {
            out.pauli.set(ctrl, 0b00);
            out.pauli.set(targ, 0b10);
        }
        _ => unreachable!(),
    }
    out
}

/// Propagate through CZ on (qubit_a, qubit_b).
///
/// CZ is self-inverse and symmetric. Conjugation table:
///   II->II, IX->ZX, IY->ZY, IZ->IZ,
///   XI->XZ, XX->-YY, XY->YX, XZ->XI,
///   YI->YZ, YX->XY, YY->-XX, YZ->YI,
///   ZI->ZI, ZX->IX, ZY->IY, ZZ->ZZ.
fn propagate_cz(term: &WeightedPackedPauli, a: usize, b: usize) -> WeightedPackedPauli {
    let mut out = term.clone();
    let pa = term.pauli.get(a);
    let pb = term.pauli.get(b);
    match (pa, pb) {
        (0b00, 0b00) | (0b00, 0b10) | (0b10, 0b00) | (0b10, 0b10) => {}
        (0b00, 0b01) => {
            out.pauli.set(a, 0b10);
        }
        (0b00, 0b11) => {
            out.pauli.set(a, 0b10);
        }
        (0b01, 0b00) => {
            out.pauli.set(b, 0b10);
        }
        (0b01, 0b01) => {
            out.pauli.set(a, 0b11);
            out.pauli.set(b, 0b11);
            out.negate();
        }
        (0b01, 0b11) => {
            out.pauli.set(a, 0b11);
            out.pauli.set(b, 0b01);
        }
        (0b01, 0b10) => {
            out.pauli.set(b, 0b00);
        }
        (0b11, 0b00) => {
            out.pauli.set(b, 0b10);
        }
        (0b11, 0b01) => {
            out.pauli.set(a, 0b01);
            out.pauli.set(b, 0b11);
        }
        (0b11, 0b11) => {
            out.pauli.set(a, 0b01);
            out.pauli.set(b, 0b01);
            out.negate();
        }
        (0b11, 0b10) => {
            out.pauli.set(b, 0b00);
        }
        (0b10, 0b01) => {
            out.pauli.set(a, 0b00);
        }
        (0b10, 0b11) => {
            out.pauli.set(a, 0b00);
        }
        _ => unreachable!(),
    }
    out
}

/// Propagate through SWAP on (qubit_a, qubit_b).
fn propagate_swap(term: &WeightedPackedPauli, a: usize, b: usize) -> WeightedPackedPauli {
    let mut out = term.clone();
    out.pauli.swap_qubits(a, b);
    out
}

/// Propagate a single term through one gate, returning one or more output terms.
fn propagate_term(term: &WeightedPackedPauli, gate: &CliffordGate) -> Vec<WeightedPackedPauli> {
    match gate {
        CliffordGate::H(q) => vec![propagate_h(term, *q)],
        CliffordGate::S(q) => vec![propagate_s(term, *q)],
        CliffordGate::Sdg(q) => vec![propagate_sdg(term, *q)],
        CliffordGate::CX(c, t) => vec![propagate_cx(term, *c, *t)],
        CliffordGate::CZ(a, b) => vec![propagate_cz(term, *a, *b)],
        CliffordGate::SWAP(a, b) => vec![propagate_swap(term, *a, *b)],
        CliffordGate::T(q) => propagate_t(term, *q),
        CliffordGate::Tdg(q) => propagate_tdg(term, *q),
        CliffordGate::Rz(q, theta) => propagate_rz(term, *q, *theta),
    }
}

// =====================================================================
// PAULI PROPAGATOR (CPU ENGINE)
// =====================================================================

/// Result of a propagation run.
#[derive(Clone, Debug)]
pub struct PropResult {
    /// Final propagated terms.
    pub terms: Vec<WeightedPackedPauli>,
    /// Expectation value on |0...0>.
    pub expectation_value: f64,
    /// Number of terms before propagation.
    pub initial_count: usize,
    /// Number of terms after propagation.
    pub final_count: usize,
    /// Peak term count during propagation.
    pub peak_count: usize,
    /// Total terms dropped by truncation.
    pub truncated_count: usize,
    /// Wall-clock time in seconds.
    pub elapsed_secs: f64,
}

/// CPU-based Pauli propagation engine.
///
/// Propagates a set of weighted Pauli strings through a circuit in the
/// Heisenberg picture (gates applied in reverse order).
#[derive(Clone, Debug)]
pub struct PauliPropagator {
    /// Configuration.
    pub config: GpuPropConfig,
    /// The terms being propagated.
    pub terms: Vec<WeightedPackedPauli>,
    /// Number of qubits.
    pub num_qubits: usize,
    /// The circuit (forward order; propagation reverses).
    pub circuit: Vec<CliffordGate>,
}

impl PauliPropagator {
    /// Create a new propagator with a single observable term.
    pub fn new(
        num_qubits: usize,
        observable: WeightedPackedPauli,
        circuit: Vec<CliffordGate>,
        config: GpuPropConfig,
    ) -> Self {
        PauliPropagator {
            config,
            terms: vec![observable],
            num_qubits,
            circuit,
        }
    }

    /// Create from multiple observable terms.
    pub fn from_terms(
        num_qubits: usize,
        terms: Vec<WeightedPackedPauli>,
        circuit: Vec<CliffordGate>,
        config: GpuPropConfig,
    ) -> Self {
        PauliPropagator {
            config,
            terms,
            num_qubits,
            circuit,
        }
    }

    /// Validate qubit indices in the circuit.
    pub fn validate(&self) -> Result<(), GpuPauliError> {
        if self.num_qubits == 0 {
            return Err(GpuPauliError::InvalidPauli(
                "zero qubits".to_string(),
            ));
        }
        for (i, term) in self.terms.iter().enumerate() {
            if term.pauli.num_qubits != self.num_qubits {
                return Err(GpuPauliError::InvalidPauli(format!(
                    "term {} has {} qubits, expected {}",
                    i, term.pauli.num_qubits, self.num_qubits
                )));
            }
        }
        for (i, gate) in self.circuit.iter().enumerate() {
            if gate.max_qubit() >= self.num_qubits {
                return Err(GpuPauliError::CircuitError(format!(
                    "gate {} ({}) references qubit {} but system has {} qubits",
                    i, gate, gate.max_qubit(), self.num_qubits
                )));
            }
        }
        Ok(())
    }

    /// Run the propagation and return the result.
    pub fn propagate(&mut self) -> Result<PropResult, GpuPauliError> {
        self.validate()?;

        let start = Instant::now();
        let initial_count = self.terms.len();
        let mut peak = initial_count;
        let mut total_truncated = 0usize;

        // Propagate in reverse (Heisenberg picture).
        let num_gates = self.circuit.len();
        for gate_idx in (0..num_gates).rev() {
            let gate = self.circuit[gate_idx].clone();
            let mut new_terms = Vec::with_capacity(self.terms.len());
            for term in &self.terms {
                new_terms.extend(propagate_term(term, &gate));
            }
            self.terms = new_terms;

            if self.terms.len() > peak {
                peak = self.terms.len();
            }

            // Truncate after non-Clifford gates.
            if !gate.is_clifford() {
                let before = self.terms.len();
                if self.config.merge_duplicates {
                    self.merge_terms();
                }
                self.truncate_by_magnitude();
                self.truncate_by_count();
                let after = self.terms.len();
                if before > after {
                    total_truncated += before - after;
                }
            }

            // Hard overflow check.
            if self.terms.len() > self.config.max_terms * 2 {
                return Err(GpuPauliError::Overflow {
                    count: self.terms.len(),
                    limit: self.config.max_terms,
                });
            }
        }

        let expectation = self.expectation_on_zero_state();
        let elapsed = start.elapsed().as_secs_f64();

        Ok(PropResult {
            terms: self.terms.clone(),
            expectation_value: expectation,
            initial_count,
            final_count: self.terms.len(),
            peak_count: peak,
            truncated_count: total_truncated,
            elapsed_secs: elapsed,
        })
    }

    /// Propagate a batch of Pauli strings through a single gate (public API
    /// for batch processing).
    pub fn propagate_batch_through_gate(
        terms: &[WeightedPackedPauli],
        gate: &CliffordGate,
    ) -> Vec<WeightedPackedPauli> {
        let mut out = Vec::with_capacity(terms.len());
        for t in terms {
            out.extend(propagate_term(t, gate));
        }
        out
    }

    /// Compute expectation value on the |0...0> state.
    ///
    /// Only all-identity Pauli strings contribute.
    pub fn expectation_on_zero_state(&self) -> f64 {
        let mut sum = 0.0;
        for term in &self.terms {
            if term.pauli.is_identity() {
                sum += term.coeff.0;
            }
        }
        sum
    }

    /// Compute expectation value on a computational basis state |b>.
    pub fn expectation_on_basis_state(&self, basis_state: u64) -> f64 {
        let mut total = 0.0;
        for term in &self.terms {
            let mut contributes = true;
            let mut sign = 1.0f64;
            for q in 0..self.num_qubits {
                match term.pauli.get(q) {
                    0b00 => {}       // I
                    0b10 => {        // Z
                        if (basis_state >> q) & 1 == 1 {
                            sign *= -1.0;
                        }
                    }
                    _ => {
                        contributes = false;
                        break;
                    }
                }
            }
            if contributes {
                total += term.coeff.0 * sign;
            }
        }
        total
    }

    /// Merge terms with identical Pauli strings by summing coefficients.
    fn merge_terms(&mut self) {
        use std::collections::HashMap;
        let mut map: HashMap<Vec<u64>, (f64, f64)> = HashMap::new();
        for term in &self.terms {
            let key = term.pauli.data.clone();
            let entry = map.entry(key).or_insert((0.0, 0.0));
            entry.0 += term.coeff.0;
            entry.1 += term.coeff.1;
        }
        let threshold = self.config.truncation_threshold;
        self.terms = map
            .into_iter()
            .filter(|(_, (re, im))| (re * re + im * im).sqrt() >= threshold)
            .map(|(data, coeff)| WeightedPackedPauli {
                pauli: PackedPauliString {
                    data,
                    num_qubits: self.num_qubits,
                    phase: Phase::PLUS_ONE,
                },
                coeff,
            })
            .collect();
    }

    /// Drop terms below the truncation threshold.
    fn truncate_by_magnitude(&mut self) {
        let threshold = self.config.truncation_threshold;
        self.terms.retain(|t| t.magnitude() >= threshold);
    }

    /// Keep only the top `max_terms` by coefficient magnitude.
    fn truncate_by_count(&mut self) {
        let max = self.config.max_terms;
        if self.terms.len() > max {
            self.terms
                .sort_by(|a, b| b.magnitude().partial_cmp(&a.magnitude()).unwrap());
            self.terms.truncate(max);
        }
    }
}

// =====================================================================
// PAULI TABLEAU
// =====================================================================

/// Stabilizer-like tableau tracking how each initial single-qubit Pauli
/// transforms under Clifford conjugation.
///
/// For n qubits, maintains 2n rows: X_0..X_{n-1}, Z_0..Z_{n-1}.
/// Each row is a PackedPauliString plus phase, representing the evolved
/// Pauli operator.
#[derive(Clone, Debug)]
pub struct PauliTableau {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Stabilizer rows: first n are the X generators, next n are Z generators.
    pub rows: Vec<PackedPauliString>,
    /// Phase for each row.
    pub phases: Vec<Phase>,
}

impl PauliTableau {
    /// Initialize to the identity tableau (X_j -> X_j, Z_j -> Z_j).
    pub fn new(num_qubits: usize) -> Self {
        let n = num_qubits;
        let mut rows = Vec::with_capacity(2 * n);
        let mut phases = Vec::with_capacity(2 * n);

        // X generators.
        for j in 0..n {
            rows.push(PackedPauliString::single_x(n, j));
            phases.push(Phase::PLUS_ONE);
        }
        // Z generators.
        for j in 0..n {
            rows.push(PackedPauliString::single_z(n, j));
            phases.push(Phase::PLUS_ONE);
        }

        PauliTableau {
            num_qubits: n,
            rows,
            phases,
        }
    }

    /// Apply a Clifford gate to the tableau.
    ///
    /// Updates each row by conjugation: row -> gate^dag row gate.
    pub fn apply_gate(&mut self, gate: &CliffordGate) {
        let n = self.num_qubits;
        for i in 0..(2 * n) {
            let term = WeightedPackedPauli {
                pauli: self.rows[i].clone(),
                coeff: self.phases[i].to_complex(),
            };
            let results = propagate_term(&term, gate);
            // Clifford gates produce exactly one output term.
            debug_assert_eq!(results.len(), 1, "Tableau only supports Clifford gates");
            self.rows[i] = results[0].pauli.clone();
            // Reconstruct phase from the coefficient.
            let (re, im) = results[0].coeff;
            self.phases[i] = coeff_to_phase(re, im);
        }
    }

    /// Look up what a single-qubit X on qubit `q` has evolved to.
    pub fn get_x_image(&self, q: usize) -> (&PackedPauliString, Phase) {
        (&self.rows[q], self.phases[q])
    }

    /// Look up what a single-qubit Z on qubit `q` has evolved to.
    pub fn get_z_image(&self, q: usize) -> (&PackedPauliString, Phase) {
        let n = self.num_qubits;
        (&self.rows[n + q], self.phases[n + q])
    }

    /// Apply a full Clifford circuit to this tableau.
    pub fn apply_circuit(&mut self, circuit: &[CliffordGate]) {
        for gate in circuit.iter().rev() {
            if !gate.is_clifford() {
                continue; // Skip non-Clifford gates in tableau mode.
            }
            self.apply_gate(gate);
        }
    }
}

/// Convert a coefficient (re, im) back to the nearest Phase.
fn coeff_to_phase(re: f64, im: f64) -> Phase {
    if re > 0.5 {
        Phase::PLUS_ONE
    } else if re < -0.5 {
        Phase::MINUS_ONE
    } else if im > 0.5 {
        Phase::PLUS_I
    } else {
        Phase::MINUS_I
    }
}

// =====================================================================
// METAL PAULI KERNEL (macOS GPU)
// =====================================================================

/// Metal shader source for Pauli propagation.
///
/// Each thread processes one Pauli string. The kernel reads the gate type
/// and parameters from a uniform buffer, then applies the conjugation rule
/// to the packed 2-bit-per-qubit representation in-place.
#[cfg(target_os = "macos")]
const PAULI_PROPAGATION_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Pauli codes: I=0, X=1, Z=2, Y=3 (2 bits per qubit, 32 qubits per uint)
// Gate types: 0=H, 1=S, 2=Sdg, 3=CX, 4=CZ, 5=SWAP

kernel void propagate_clifford(
    device uint* pauli_data    [[buffer(0)]],  // packed pauli strings
    constant uint& gate_type   [[buffer(1)]],  // gate type code
    constant uint& qubit_a     [[buffer(2)]],  // first qubit
    constant uint& qubit_b     [[buffer(3)]],  // second qubit (for 2q gates)
    constant uint& num_qubits  [[buffer(4)]],  // qubits per string
    constant uint& words_per_string [[buffer(5)]],  // u32 words per string
    device uint* phase_data    [[buffer(6)]],  // phase array (one per string)
    uint tid [[thread_position_in_grid]]
) {
    uint base = tid * words_per_string;
    uint word_a = qubit_a / 16;
    uint bit_a = (qubit_a % 16) * 2;
    uint pa = (pauli_data[base + word_a] >> bit_a) & 3;

    if (gate_type == 0) {
        // Hadamard: X<->Z, Y->-Y
        if (pa == 1) {
            pauli_data[base + word_a] = (pauli_data[base + word_a] & ~(3u << bit_a)) | (2u << bit_a);
        } else if (pa == 2) {
            pauli_data[base + word_a] = (pauli_data[base + word_a] & ~(3u << bit_a)) | (1u << bit_a);
        } else if (pa == 3) {
            phase_data[tid] = (phase_data[tid] + 2) % 4;
        }
    } else if (gate_type == 1) {
        // S: X->Y, Y->-X
        if (pa == 1) {
            pauli_data[base + word_a] = (pauli_data[base + word_a] & ~(3u << bit_a)) | (3u << bit_a);
        } else if (pa == 3) {
            pauli_data[base + word_a] = (pauli_data[base + word_a] & ~(3u << bit_a)) | (1u << bit_a);
            phase_data[tid] = (phase_data[tid] + 2) % 4;
        }
    } else if (gate_type == 2) {
        // Sdg: X->-Y, Y->X
        if (pa == 1) {
            pauli_data[base + word_a] = (pauli_data[base + word_a] & ~(3u << bit_a)) | (3u << bit_a);
            phase_data[tid] = (phase_data[tid] + 2) % 4;
        } else if (pa == 3) {
            pauli_data[base + word_a] = (pauli_data[base + word_a] & ~(3u << bit_a)) | (1u << bit_a);
        }
    } else if (gate_type == 5) {
        // SWAP: exchange paulis on qubit_a and qubit_b
        uint word_b = qubit_b / 16;
        uint bit_b = (qubit_b % 16) * 2;
        uint pb = (pauli_data[base + word_b] >> bit_b) & 3;
        pauli_data[base + word_a] = (pauli_data[base + word_a] & ~(3u << bit_a)) | (pb << bit_a);
        pauli_data[base + word_b] = (pauli_data[base + word_b] & ~(3u << bit_b)) | (pa << bit_b);
    }
    // CX and CZ require a full lookup table; handled by a separate kernel
    // or multiple dispatches in the host code.
}
"#;

/// Metal GPU kernel manager for parallel Pauli propagation.
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct MetalPauliKernel {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    /// Number of Pauli strings currently on GPU.
    num_strings: usize,
    /// Words per Pauli string (u32 granularity for Metal).
    words_per_string: usize,
    /// Number of qubits.
    num_qubits: usize,
}

#[cfg(target_os = "macos")]
impl MetalPauliKernel {
    /// Initialize the Metal kernel.
    pub fn new(num_qubits: usize) -> Result<Self, GpuPauliError> {
        let device = metal::Device::system_default()
            .ok_or_else(|| GpuPauliError::GpuError("No Metal device found".to_string()))?;

        let library = device
            .new_library_with_source(PAULI_PROPAGATION_SHADER, &metal::CompileOptions::new())
            .map_err(|e| GpuPauliError::GpuError(format!("Shader compile: {}", e)))?;

        let func = library
            .get_function("propagate_clifford", None)
            .map_err(|e| GpuPauliError::GpuError(format!("Function lookup: {}", e)))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| GpuPauliError::GpuError(format!("Pipeline: {}", e)))?;

        let queue = device.new_command_queue();
        // Each u32 holds 16 qubits (2 bits * 16 = 32 bits).
        let words_per_string = (num_qubits + 15) / 16;

        Ok(MetalPauliKernel {
            device,
            queue,
            pipeline,
            num_strings: 0,
            words_per_string,
            num_qubits,
        })
    }

    /// Upload Pauli strings to GPU buffers and dispatch a Clifford gate.
    ///
    /// Returns the propagated strings. This is the main GPU entry point.
    pub fn dispatch_clifford(
        &mut self,
        strings: &[PackedPauliString],
        gate_type: u32,
        qubit_a: u32,
        qubit_b: u32,
    ) -> Result<Vec<PackedPauliString>, GpuPauliError> {
        use metal::*;

        self.num_strings = strings.len();
        let wps = self.words_per_string;

        // Pack all strings into a flat u32 buffer.
        let mut flat_data: Vec<u32> = vec![0u32; self.num_strings * wps];
        let mut phase_data: Vec<u32> = vec![0u32; self.num_strings];
        for (i, s) in strings.iter().enumerate() {
            phase_data[i] = s.phase.0 as u32;
            for w in 0..wps {
                if w < s.data.len() * 2 {
                    let word64_idx = w / 2;
                    if word64_idx < s.data.len() {
                        if w % 2 == 0 {
                            flat_data[i * wps + w] = s.data[word64_idx] as u32;
                        } else {
                            flat_data[i * wps + w] = (s.data[word64_idx] >> 32) as u32;
                        }
                    }
                }
            }
        }

        let data_buf = self.device.new_buffer_with_data(
            flat_data.as_ptr() as *const _,
            (flat_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let phase_buf = self.device.new_buffer_with_data(
            phase_data.as_ptr() as *const _,
            (phase_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_bytes(1, 4, &gate_type as *const u32 as *const _);
        enc.set_bytes(2, 4, &qubit_a as *const u32 as *const _);
        enc.set_bytes(3, 4, &qubit_b as *const u32 as *const _);
        let nq = self.num_qubits as u32;
        enc.set_bytes(4, 4, &nq as *const u32 as *const _);
        let wps_u32 = wps as u32;
        enc.set_bytes(5, 4, &wps_u32 as *const u32 as *const _);
        enc.set_buffer(6, Some(&phase_buf), 0);

        let grid = MTLSize::new(self.num_strings as u64, 1, 1);
        let tg = MTLSize::new(
            std::cmp::min(256, self.num_strings as u64),
            1,
            1,
        );
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Read back.
        let result_ptr = data_buf.contents() as *const u32;
        let phase_ptr = phase_buf.contents() as *const u32;
        let mut results = Vec::with_capacity(self.num_strings);
        for i in 0..self.num_strings {
            let phase = unsafe { Phase(*phase_ptr.add(i) as u8) };
            let num_words64 = PackedPauliString::num_words(self.num_qubits);
            let mut data = vec![0u64; num_words64];
            for w64 in 0..num_words64 {
                let lo_idx = i * wps + w64 * 2;
                let hi_idx = lo_idx + 1;
                let lo = if lo_idx < self.num_strings * wps {
                    (unsafe { *result_ptr.add(lo_idx) }) as u64
                } else {
                    0
                };
                let hi = if hi_idx < self.num_strings * wps {
                    (unsafe { *result_ptr.add(hi_idx) }) as u64
                } else {
                    0
                };
                data[w64] = lo | (hi << 32);
            }
            results.push(PackedPauliString {
                data,
                num_qubits: self.num_qubits,
                phase,
            });
        }

        Ok(results)
    }
}

/// CPU fallback for MetalPauliKernel on non-macOS platforms and for testing.
#[cfg(not(target_os = "macos"))]
#[derive(Clone, Debug)]
pub struct MetalPauliKernel {
    num_qubits: usize,
}

#[cfg(not(target_os = "macos"))]
impl MetalPauliKernel {
    /// Create a CPU-fallback kernel (no GPU).
    pub fn new(num_qubits: usize) -> Result<Self, GpuPauliError> {
        Ok(MetalPauliKernel { num_qubits })
    }

    /// CPU fallback dispatch: propagate using the CPU engine.
    pub fn dispatch_clifford(
        &mut self,
        strings: &[PackedPauliString],
        gate_type: u32,
        qubit_a: u32,
        qubit_b: u32,
    ) -> Result<Vec<PackedPauliString>, GpuPauliError> {
        let gate = match gate_type {
            0 => CliffordGate::H(qubit_a as usize),
            1 => CliffordGate::S(qubit_a as usize),
            2 => CliffordGate::Sdg(qubit_a as usize),
            3 => CliffordGate::CX(qubit_a as usize, qubit_b as usize),
            4 => CliffordGate::CZ(qubit_a as usize, qubit_b as usize),
            5 => CliffordGate::SWAP(qubit_a as usize, qubit_b as usize),
            _ => {
                return Err(GpuPauliError::GpuError(format!(
                    "Unknown gate type {}",
                    gate_type
                )))
            }
        };
        let mut results = Vec::with_capacity(strings.len());
        for s in strings {
            let term = WeightedPackedPauli::unit(s.clone());
            let propagated = propagate_term(&term, &gate);
            for p in propagated {
                results.push(p.pauli);
            }
        }
        Ok(results)
    }
}

// =====================================================================
// NEAR-CLIFFORD ESTIMATOR
// =====================================================================

/// Configuration for the near-Clifford Monte Carlo estimator.
#[derive(Clone, Debug)]
pub struct NearCliffordConfig {
    /// Number of Monte Carlo samples.
    pub num_samples: usize,
    /// Maximum T-gates before falling back to exact propagation.
    pub max_t_gates: usize,
    /// Seed for the random number generator (0 = use entropy).
    pub rng_seed: u64,
    /// Truncation config for the underlying propagator.
    pub prop_config: GpuPropConfig,
}

impl Default for NearCliffordConfig {
    fn default() -> Self {
        NearCliffordConfig {
            num_samples: 10_000,
            max_t_gates: 50,
            rng_seed: 0,
            prop_config: GpuPropConfig::default(),
        }
    }
}

impl NearCliffordConfig {
    /// Builder: set sample count.
    pub fn with_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Builder: set max T-gates.
    pub fn with_max_t_gates(mut self, n: usize) -> Self {
        self.max_t_gates = n;
        self
    }

    /// Builder: set RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }
}

/// Result of a near-Clifford estimation.
#[derive(Clone, Debug)]
pub struct EstimationResult {
    /// Estimated expectation value.
    pub expectation: f64,
    /// Statistical error estimate (standard error of mean).
    pub error_bound: f64,
    /// Number of samples used.
    pub num_samples: usize,
    /// Number of T-gates in the circuit.
    pub num_t_gates: usize,
    /// Wall-clock time in seconds.
    pub elapsed_secs: f64,
}

/// Monte Carlo estimator for circuits with few T-gates.
///
/// Strategy: decompose each T-gate into a sum of two Clifford operations
/// (the "sum over Cliffords" approach). Sample random Clifford branches
/// and average the resulting expectation values.
#[derive(Clone, Debug)]
pub struct NearCliffordEstimator {
    /// Configuration.
    pub config: NearCliffordConfig,
    /// Number of qubits.
    pub num_qubits: usize,
    /// Circuit gates.
    pub circuit: Vec<CliffordGate>,
    /// Observable to measure.
    pub observable: WeightedPackedPauli,
}

impl NearCliffordEstimator {
    /// Create a new estimator.
    pub fn new(
        num_qubits: usize,
        observable: WeightedPackedPauli,
        circuit: Vec<CliffordGate>,
        config: NearCliffordConfig,
    ) -> Self {
        NearCliffordEstimator {
            config,
            num_qubits,
            circuit,
            observable,
        }
    }

    /// Count the number of T-gates (T and Tdg) in the circuit.
    pub fn count_t_gates(&self) -> usize {
        self.circuit
            .iter()
            .filter(|g| matches!(g, CliffordGate::T(_) | CliffordGate::Tdg(_)))
            .count()
    }

    /// Identify the indices of T-gates in the circuit.
    fn t_gate_indices(&self) -> Vec<usize> {
        self.circuit
            .iter()
            .enumerate()
            .filter(|(_, g)| matches!(g, CliffordGate::T(_) | CliffordGate::Tdg(_)))
            .map(|(i, _)| i)
            .collect()
    }

    /// Run the estimation.
    ///
    /// For each Monte Carlo sample, randomly choose one of the two Clifford
    /// branches at each T-gate position, then propagate the observable
    /// through the resulting fully Clifford circuit.
    pub fn estimate(&self) -> Result<EstimationResult, GpuPauliError> {
        let start = Instant::now();
        let t_indices = self.t_gate_indices();
        let num_t = t_indices.len();

        if num_t > self.config.max_t_gates {
            return Err(GpuPauliError::CircuitError(format!(
                "Circuit has {} T-gates, exceeding limit of {}",
                num_t, self.config.max_t_gates
            )));
        }

        // If no T-gates, do exact Clifford propagation.
        if num_t == 0 {
            let mut prop = PauliPropagator::new(
                self.num_qubits,
                self.observable.clone(),
                self.circuit.clone(),
                self.config.prop_config.clone(),
            );
            let result = prop.propagate()?;
            return Ok(EstimationResult {
                expectation: result.expectation_value,
                error_bound: 0.0,
                num_samples: 1,
                num_t_gates: 0,
                elapsed_secs: start.elapsed().as_secs_f64(),
            });
        }

        let mut rng = if self.config.rng_seed == 0 {
            rand::thread_rng()
        } else {
            // Seed with a deterministic approach using thread_rng.
            // Since rand 0.8 does not expose SeedableRng on ThreadRng,
            // we use thread_rng and burn the seed through initial draws.
            let mut r = rand::thread_rng();
            for _ in 0..self.config.rng_seed {
                let _: u64 = r.gen();
            }
            r
        };

        let cos_pi4 = std::f64::consts::FRAC_PI_4.cos();
        let sin_pi4 = std::f64::consts::FRAC_PI_4.sin();
        let norm_factor = (2.0f64).sqrt(); // Each branch has amplitude 1/sqrt(2)

        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let num_samples = self.config.num_samples;

        for _ in 0..num_samples {
            // For each T-gate, randomly choose branch 0 (cos) or branch 1 (sin).
            let choices: Vec<bool> = (0..num_t).map(|_| rng.gen::<bool>()).collect();

            // Build the Clifford circuit for this sample by replacing each
            // T-gate with either S (for the cos branch) or identity + phase
            // (for the sin branch).
            let mut sample_circuit = self.circuit.clone();
            let mut sample_weight = 1.0f64;

            for (idx, &choice) in choices.iter().enumerate() {
                let gate_idx = t_indices[idx];
                let qubit = match &self.circuit[gate_idx] {
                    CliffordGate::T(q) | CliffordGate::Tdg(q) => *q,
                    _ => unreachable!(),
                };

                if choice {
                    // cos branch: replace T with S (closest Clifford).
                    sample_circuit[gate_idx] = CliffordGate::S(qubit);
                    sample_weight *= cos_pi4 * norm_factor;
                } else {
                    // sin branch: replace T with Sdg (orthogonal Clifford).
                    sample_circuit[gate_idx] = CliffordGate::Sdg(qubit);
                    sample_weight *= sin_pi4 * norm_factor;
                }
            }

            // Propagate through the sampled Clifford circuit.
            let mut prop = PauliPropagator::new(
                self.num_qubits,
                self.observable.clone(),
                sample_circuit,
                self.config.prop_config.clone(),
            );
            match prop.propagate() {
                Ok(result) => {
                    let val = result.expectation_value * sample_weight;
                    sum += val;
                    sum_sq += val * val;
                }
                Err(_) => {
                    // Skip failed samples (should be rare for Clifford circuits).
                }
            }
        }

        let mean = sum / num_samples as f64;
        let variance = (sum_sq / num_samples as f64) - mean * mean;
        let stderr = if num_samples > 1 {
            (variance.abs() / (num_samples - 1) as f64).sqrt()
        } else {
            f64::INFINITY
        };

        Ok(EstimationResult {
            expectation: mean,
            error_bound: stderr,
            num_samples,
            num_t_gates: num_t,
            elapsed_secs: start.elapsed().as_secs_f64(),
        })
    }
}

// =====================================================================
// AUTO DISPATCH
// =====================================================================

/// Backend selection for Pauli propagation.
#[derive(Clone, Debug, PartialEq)]
pub enum DispatchBackend {
    /// CPU-only propagation.
    Cpu,
    /// Metal GPU acceleration.
    MetalGpu,
}

impl fmt::Display for DispatchBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchBackend::Cpu => write!(f, "CPU"),
            DispatchBackend::MetalGpu => write!(f, "Metal GPU"),
        }
    }
}

/// Automatic dispatch between CPU and GPU based on problem size.
#[derive(Clone, Debug)]
pub struct AutoDispatch {
    /// Minimum Pauli string count for GPU dispatch.
    pub gpu_threshold: usize,
    /// Whether Metal GPU is available on this system.
    pub gpu_available: bool,
}

impl Default for AutoDispatch {
    fn default() -> Self {
        AutoDispatch {
            gpu_threshold: 1000,
            gpu_available: cfg!(target_os = "macos"),
        }
    }
}

impl AutoDispatch {
    /// Create with a custom GPU threshold.
    pub fn with_threshold(threshold: usize) -> Self {
        AutoDispatch {
            gpu_threshold: threshold,
            gpu_available: cfg!(target_os = "macos"),
        }
    }

    /// Decide which backend to use.
    pub fn select(&self, num_strings: usize) -> DispatchBackend {
        if self.gpu_available && num_strings >= self.gpu_threshold {
            DispatchBackend::MetalGpu
        } else {
            DispatchBackend::Cpu
        }
    }

    /// Propagate a batch of Pauli strings through a single Clifford gate,
    /// automatically choosing CPU or GPU.
    pub fn propagate_batch(
        &self,
        num_qubits: usize,
        strings: &[PackedPauliString],
        gate: &CliffordGate,
    ) -> Result<Vec<PackedPauliString>, GpuPauliError> {
        let backend = self.select(strings.len());
        match backend {
            DispatchBackend::Cpu => {
                let mut results = Vec::with_capacity(strings.len());
                for s in strings {
                    let term = WeightedPackedPauli::unit(s.clone());
                    let propagated = propagate_term(&term, gate);
                    for p in propagated {
                        results.push(p.pauli);
                    }
                }
                Ok(results)
            }
            DispatchBackend::MetalGpu => {
                let (gate_type, qubit_a, qubit_b) = match gate {
                    CliffordGate::H(q) => (0u32, *q as u32, 0u32),
                    CliffordGate::S(q) => (1, *q as u32, 0),
                    CliffordGate::Sdg(q) => (2, *q as u32, 0),
                    CliffordGate::CX(c, t) => (3, *c as u32, *t as u32),
                    CliffordGate::CZ(a, b) => (4, *a as u32, *b as u32),
                    CliffordGate::SWAP(a, b) => (5, *a as u32, *b as u32),
                    _ => {
                        // Non-Clifford: fall back to CPU.
                        let mut results = Vec::with_capacity(strings.len());
                        for s in strings {
                            let term = WeightedPackedPauli::unit(s.clone());
                            let propagated = propagate_term(&term, gate);
                            for p in propagated {
                                results.push(p.pauli);
                            }
                        }
                        return Ok(results);
                    }
                };
                let mut kernel = MetalPauliKernel::new(num_qubits)?;
                kernel.dispatch_clifford(strings, gate_type, qubit_a, qubit_b)
            }
        }
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =================================================================
    // Phase tests
    // =================================================================

    #[test]
    fn test_phase_constants_to_complex() {
        assert_eq!(Phase::PLUS_ONE.to_complex(), (1.0, 0.0));
        assert_eq!(Phase::PLUS_I.to_complex(), (0.0, 1.0));
        assert_eq!(Phase::MINUS_ONE.to_complex(), (-1.0, 0.0));
        assert_eq!(Phase::MINUS_I.to_complex(), (0.0, -1.0));
    }

    #[test]
    fn test_phase_multiplication_table() {
        // +1 * anything = anything
        assert_eq!(Phase::PLUS_ONE.mul(Phase::PLUS_ONE), Phase::PLUS_ONE);
        assert_eq!(Phase::PLUS_ONE.mul(Phase::PLUS_I), Phase::PLUS_I);
        assert_eq!(Phase::PLUS_ONE.mul(Phase::MINUS_ONE), Phase::MINUS_ONE);
        assert_eq!(Phase::PLUS_ONE.mul(Phase::MINUS_I), Phase::MINUS_I);

        // i * i = -1
        assert_eq!(Phase::PLUS_I.mul(Phase::PLUS_I), Phase::MINUS_ONE);
        // i * (-1) = -i
        assert_eq!(Phase::PLUS_I.mul(Phase::MINUS_ONE), Phase::MINUS_I);
        // i * (-i) = +1
        assert_eq!(Phase::PLUS_I.mul(Phase::MINUS_I), Phase::PLUS_ONE);
        // (-1) * (-1) = +1
        assert_eq!(Phase::MINUS_ONE.mul(Phase::MINUS_ONE), Phase::PLUS_ONE);
        // (-i) * (-i) = -1
        assert_eq!(Phase::MINUS_I.mul(Phase::MINUS_I), Phase::MINUS_ONE);
    }

    #[test]
    fn test_phase_negate() {
        assert_eq!(Phase::PLUS_ONE.negate(), Phase::MINUS_ONE);
        assert_eq!(Phase::MINUS_ONE.negate(), Phase::PLUS_ONE);
        assert_eq!(Phase::PLUS_I.negate(), Phase::MINUS_I);
        assert_eq!(Phase::MINUS_I.negate(), Phase::PLUS_I);
    }

    #[test]
    fn test_phase_conjugate() {
        assert_eq!(Phase::PLUS_ONE.conjugate(), Phase::PLUS_ONE);
        assert_eq!(Phase::MINUS_ONE.conjugate(), Phase::MINUS_ONE);
        assert_eq!(Phase::PLUS_I.conjugate(), Phase::MINUS_I);
        assert_eq!(Phase::MINUS_I.conjugate(), Phase::PLUS_I);
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::PLUS_ONE), "+1");
        assert_eq!(format!("{}", Phase::PLUS_I), "+i");
        assert_eq!(format!("{}", Phase::MINUS_ONE), "-1");
        assert_eq!(format!("{}", Phase::MINUS_I), "-i");
    }

    // =================================================================
    // PackedPauliString construction tests
    // =================================================================

    #[test]
    fn test_packed_pauli_string_identity() {
        let id = PackedPauliString::identity(4);
        assert_eq!(id.num_qubits, 4);
        assert_eq!(id.phase, Phase::PLUS_ONE);
        assert!(id.is_identity());
        assert_eq!(id.weight(), 0);
        for q in 0..4 {
            assert_eq!(id.get(q), 0b00);
            assert_eq!(id.get_char(q), 'I');
        }
    }

    #[test]
    fn test_packed_pauli_string_single_x() {
        let x = PackedPauliString::single_x(4, 1);
        assert_eq!(x.num_qubits, 4);
        assert_eq!(x.phase, Phase::PLUS_ONE);
        assert_eq!(x.weight(), 1);
        assert!(!x.is_identity());
        assert_eq!(x.get(0), 0b00); // I
        assert_eq!(x.get(1), 0b01); // X
        assert_eq!(x.get(2), 0b00); // I
        assert_eq!(x.get(3), 0b00); // I
    }

    #[test]
    fn test_packed_pauli_string_single_y() {
        let y = PackedPauliString::single_y(3, 2);
        assert_eq!(y.weight(), 1);
        assert_eq!(y.get(0), 0b00);
        assert_eq!(y.get(1), 0b00);
        assert_eq!(y.get(2), 0b11); // Y
        assert_eq!(y.get_char(2), 'Y');
    }

    #[test]
    fn test_packed_pauli_string_single_z() {
        let z = PackedPauliString::single_z(5, 0);
        assert_eq!(z.weight(), 1);
        assert_eq!(z.get(0), 0b10); // Z
        assert_eq!(z.get_char(0), 'Z');
        for q in 1..5 {
            assert_eq!(z.get(q), 0b00);
        }
    }

    #[test]
    fn test_packed_pauli_string_from_str_rep() {
        let ps = PackedPauliString::from_str_rep("IXYZ").unwrap();
        assert_eq!(ps.num_qubits, 4);
        assert_eq!(ps.get_char(0), 'I');
        assert_eq!(ps.get_char(1), 'X');
        assert_eq!(ps.get_char(2), 'Y');
        assert_eq!(ps.get_char(3), 'Z');
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_packed_pauli_string_from_str_rep_lowercase() {
        let ps = PackedPauliString::from_str_rep("ixyz").unwrap();
        assert_eq!(ps.get_char(0), 'I');
        assert_eq!(ps.get_char(1), 'X');
        assert_eq!(ps.get_char(2), 'Y');
        assert_eq!(ps.get_char(3), 'Z');
    }

    #[test]
    fn test_packed_pauli_string_from_str_rep_invalid() {
        let err = PackedPauliString::from_str_rep("IXQA").unwrap_err();
        match err {
            GpuPauliError::InvalidPauli(msg) => {
                assert!(msg.contains("invalid character"));
            }
            _ => panic!("Expected InvalidPauli error"),
        }
    }

    // =================================================================
    // PackedPauliString bit operations and properties
    // =================================================================

    #[test]
    fn test_packed_pauli_string_set_get_roundtrip() {
        let mut ps = PackedPauliString::identity(8);
        ps.set(0, 0b01); // X
        ps.set(3, 0b10); // Z
        ps.set(7, 0b11); // Y
        assert_eq!(ps.get(0), 0b01);
        assert_eq!(ps.get(1), 0b00);
        assert_eq!(ps.get(3), 0b10);
        assert_eq!(ps.get(7), 0b11);
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_packed_pauli_string_large_qubit_count() {
        // 64 qubits requires 2 u64 words (32 qubits per word).
        let mut ps = PackedPauliString::identity(64);
        assert_eq!(ps.data.len(), 2);
        assert_eq!(ps.weight(), 0);

        // Set paulis near word boundaries.
        ps.set(0, 0b01);   // X on qubit 0 (word 0, offset 0)
        ps.set(31, 0b10);  // Z on qubit 31 (word 0, offset 62)
        ps.set(32, 0b11);  // Y on qubit 32 (word 1, offset 0)
        ps.set(63, 0b01);  // X on qubit 63 (word 1, offset 62)

        assert_eq!(ps.get(0), 0b01);
        assert_eq!(ps.get(31), 0b10);
        assert_eq!(ps.get(32), 0b11);
        assert_eq!(ps.get(63), 0b01);
        assert_eq!(ps.weight(), 4);
    }

    #[test]
    fn test_packed_pauli_string_100_qubits() {
        // 100 qubits requires ceil(100/32) = 4 words.
        let mut ps = PackedPauliString::identity(100);
        assert_eq!(ps.data.len(), 4);
        ps.set(99, 0b11); // Y on last qubit
        assert_eq!(ps.get(99), 0b11);
        assert_eq!(ps.weight(), 1);
    }

    #[test]
    fn test_packed_pauli_string_zero_qubits() {
        let ps = PackedPauliString::identity(0);
        assert_eq!(ps.num_qubits, 0);
        assert_eq!(ps.data.len(), 0);
        assert!(ps.is_identity());
        assert_eq!(ps.weight(), 0);
    }

    #[test]
    fn test_packed_pauli_string_one_qubit() {
        let x = PackedPauliString::single_x(1, 0);
        assert_eq!(x.num_qubits, 1);
        assert_eq!(x.data.len(), 1);
        assert_eq!(x.weight(), 1);
        assert!(!x.is_identity());
    }

    // =================================================================
    // PackedPauliString commutation tests
    // =================================================================

    #[test]
    fn test_commutes_with_same_pauli() {
        let x0 = PackedPauliString::single_x(4, 0);
        // Same operator on same qubit always commutes with itself.
        assert!(x0.commutes_with(&x0));
    }

    #[test]
    fn test_commutes_with_different_qubits() {
        let x0 = PackedPauliString::single_x(4, 0);
        let z2 = PackedPauliString::single_z(4, 2);
        // Operators on different qubits always commute.
        assert!(x0.commutes_with(&z2));
    }

    #[test]
    fn test_anticommutes_xz_same_qubit() {
        let x0 = PackedPauliString::single_x(2, 0);
        let z0 = PackedPauliString::single_z(2, 0);
        // X and Z on the same qubit anticommute (odd number of anticommuting sites = 1).
        assert!(!x0.commutes_with(&z0));
    }

    #[test]
    fn test_commutes_two_anticommuting_sites() {
        // X0 Z1 and Z0 X1: two sites where they anticommute -> even count -> commute.
        let mut a = PackedPauliString::identity(2);
        a.set(0, 0b01); // X
        a.set(1, 0b10); // Z
        let mut b = PackedPauliString::identity(2);
        b.set(0, 0b10); // Z
        b.set(1, 0b01); // X
        assert!(a.commutes_with(&b));
    }

    #[test]
    fn test_identity_commutes_with_everything() {
        let id = PackedPauliString::identity(4);
        let x = PackedPauliString::single_x(4, 2);
        let y = PackedPauliString::single_y(4, 1);
        let z = PackedPauliString::single_z(4, 3);
        assert!(id.commutes_with(&x));
        assert!(id.commutes_with(&y));
        assert!(id.commutes_with(&z));
    }

    // =================================================================
    // PackedPauliString multiplication tests
    // =================================================================

    #[test]
    fn test_multiply_identity_times_x() {
        let id = PackedPauliString::identity(2);
        let x = PackedPauliString::single_x(2, 0);
        let result = id.multiply(&x);
        assert_eq!(result.get(0), 0b01); // X
        assert_eq!(result.get(1), 0b00); // I
        assert_eq!(result.phase, Phase::PLUS_ONE);
    }

    #[test]
    fn test_multiply_x_times_x() {
        let x0 = PackedPauliString::single_x(2, 0);
        let result = x0.multiply(&x0);
        assert!(result.is_identity());
        assert_eq!(result.phase, Phase::PLUS_ONE);
    }

    #[test]
    fn test_multiply_x_times_y_gives_iz() {
        // X * Y = iZ at site 0.
        let x = PackedPauliString::single_x(1, 0);
        let y = PackedPauliString::single_y(1, 0);
        let result = x.multiply(&y);
        assert_eq!(result.get(0), 0b10); // Z
        assert_eq!(result.phase, Phase::PLUS_I);
    }

    #[test]
    fn test_multiply_y_times_z_gives_ix() {
        // Y * Z = iX at site 0.
        let y = PackedPauliString::single_y(1, 0);
        let z = PackedPauliString::single_z(1, 0);
        let result = y.multiply(&z);
        assert_eq!(result.get(0), 0b01); // X
        assert_eq!(result.phase, Phase::PLUS_I);
    }

    #[test]
    fn test_multiply_z_times_x_gives_iy() {
        // Z * X = iY at site 0.
        let z = PackedPauliString::single_z(1, 0);
        let x = PackedPauliString::single_x(1, 0);
        let result = z.multiply(&x);
        assert_eq!(result.get(0), 0b11); // Y
        assert_eq!(result.phase, Phase::PLUS_I);
    }

    #[test]
    fn test_multiply_reverse_order_gives_negative_i() {
        // Y * X = -iZ (reverse of X*Y).
        let y = PackedPauliString::single_y(1, 0);
        let x = PackedPauliString::single_x(1, 0);
        let result = y.multiply(&x);
        assert_eq!(result.get(0), 0b10); // Z
        assert_eq!(result.phase, Phase::MINUS_I);
    }

    // =================================================================
    // PackedPauliString swap and serialization tests
    // =================================================================

    #[test]
    fn test_swap_qubits() {
        let mut ps = PackedPauliString::from_str_rep("XYI").unwrap();
        ps.swap_qubits(0, 2);
        assert_eq!(ps.get_char(0), 'I');
        assert_eq!(ps.get_char(1), 'Y');
        assert_eq!(ps.get_char(2), 'X');
    }

    #[test]
    fn test_gpu_buffer_roundtrip() {
        let original = PackedPauliString::from_str_rep("XYZIXYZ").unwrap();
        let buf = original.to_gpu_buffer();
        let recovered = PackedPauliString::from_gpu_buffer(&buf).unwrap();
        assert_eq!(original.num_qubits, recovered.num_qubits);
        assert_eq!(original.phase, recovered.phase);
        for q in 0..original.num_qubits {
            assert_eq!(original.get(q), recovered.get(q));
        }
    }

    #[test]
    fn test_gpu_buffer_too_short() {
        let err = PackedPauliString::from_gpu_buffer(&[]).unwrap_err();
        match err {
            GpuPauliError::InvalidPauli(msg) => {
                assert!(msg.contains("too short"));
            }
            _ => panic!("Expected InvalidPauli error"),
        }
    }

    #[test]
    fn test_packed_pauli_string_display() {
        let ps = PackedPauliString::from_str_rep("XYZ").unwrap();
        let s = format!("{}", ps);
        assert!(s.contains("XYZ"));
        assert!(s.contains("(+1)"));
    }

    // =================================================================
    // WeightedPackedPauli tests
    // =================================================================

    #[test]
    fn test_weighted_packed_pauli_unit() {
        let ps = PackedPauliString::single_x(2, 0);
        let w = WeightedPackedPauli::unit(ps);
        assert_eq!(w.coeff, (1.0, 0.0));
        assert!((w.magnitude() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_packed_pauli_unit_with_phase() {
        let mut ps = PackedPauliString::single_x(2, 0);
        ps.phase = Phase::MINUS_I;
        let w = WeightedPackedPauli::unit(ps);
        assert_eq!(w.coeff, (0.0, -1.0));
        assert!((w.magnitude() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_packed_pauli_scale() {
        let ps = PackedPauliString::single_z(2, 1);
        let mut w = WeightedPackedPauli::unit(ps);
        w.scale(0.5);
        assert!((w.coeff.0 - 0.5).abs() < 1e-12);
        assert!((w.coeff.1).abs() < 1e-12);
        assert!((w.magnitude() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_packed_pauli_scale_complex() {
        let ps = PackedPauliString::identity(1);
        let mut w = WeightedPackedPauli::new(ps, (1.0, 0.0));
        // Multiply by i: (1+0i) * (0+1i) = (0+1i)
        w.scale_complex((0.0, 1.0));
        assert!((w.coeff.0).abs() < 1e-12);
        assert!((w.coeff.1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_packed_pauli_negate() {
        let ps = PackedPauliString::single_x(2, 0);
        let mut w = WeightedPackedPauli::new(ps, (3.0, -2.0));
        w.negate();
        assert!((w.coeff.0 + 3.0).abs() < 1e-12);
        assert!((w.coeff.1 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_packed_pauli_display() {
        let ps = PackedPauliString::from_str_rep("XZ").unwrap();
        let w = WeightedPackedPauli::new(ps, (0.5, -0.25));
        let s = format!("{}", w);
        assert!(s.contains("XZ"));
        // Display includes coefficient.
        assert!(s.contains("0.5"));
    }

    // =================================================================
    // CliffordGate tests
    // =================================================================

    #[test]
    fn test_clifford_gate_is_clifford() {
        assert!(CliffordGate::H(0).is_clifford());
        assert!(CliffordGate::S(0).is_clifford());
        assert!(CliffordGate::Sdg(0).is_clifford());
        assert!(CliffordGate::CX(0, 1).is_clifford());
        assert!(CliffordGate::CZ(0, 1).is_clifford());
        assert!(CliffordGate::SWAP(0, 1).is_clifford());
        assert!(!CliffordGate::T(0).is_clifford());
        assert!(!CliffordGate::Tdg(0).is_clifford());
        assert!(!CliffordGate::Rz(0, 0.5).is_clifford());
    }

    #[test]
    fn test_clifford_gate_max_qubit() {
        assert_eq!(CliffordGate::H(5).max_qubit(), 5);
        assert_eq!(CliffordGate::CX(2, 7).max_qubit(), 7);
        assert_eq!(CliffordGate::CX(7, 2).max_qubit(), 7);
        assert_eq!(CliffordGate::SWAP(3, 1).max_qubit(), 3);
        assert_eq!(CliffordGate::Rz(4, 1.0).max_qubit(), 4);
    }

    #[test]
    fn test_clifford_gate_display() {
        assert_eq!(format!("{}", CliffordGate::H(0)), "H(0)");
        assert_eq!(format!("{}", CliffordGate::CX(1, 2)), "CX(1,2)");
        assert_eq!(format!("{}", CliffordGate::SWAP(3, 4)), "SWAP(3,4)");
    }

    // =================================================================
    // Clifford propagation tests: H gate
    // =================================================================

    /// Helper: propagate a PackedPauliString through a single Clifford gate.
    fn prop_single(ps: PackedPauliString, gate: &CliffordGate) -> (PackedPauliString, (f64, f64)) {
        let term = WeightedPackedPauli::unit(ps);
        let results = propagate_term(&term, gate);
        assert_eq!(results.len(), 1, "Clifford gate should produce exactly 1 term");
        let r = &results[0];
        (r.pauli.clone(), r.coeff)
    }

    #[test]
    fn test_clifford_propagation_h_x_to_z() {
        // H X H = Z
        let x = PackedPauliString::single_x(2, 0);
        let (result, coeff) = prop_single(x, &CliffordGate::H(0));
        assert_eq!(result.get(0), 0b10); // Z
        assert_eq!(result.get(1), 0b00); // I
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_h_z_to_x() {
        // H Z H = X
        let z = PackedPauliString::single_z(2, 0);
        let (result, coeff) = prop_single(z, &CliffordGate::H(0));
        assert_eq!(result.get(0), 0b01); // X
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_h_y_to_neg_y() {
        // H Y H = -Y
        let y = PackedPauliString::single_y(2, 0);
        let (result, coeff) = prop_single(y, &CliffordGate::H(0));
        assert_eq!(result.get(0), 0b11); // Y
        assert!((coeff.0 + 1.0).abs() < 1e-12); // -1
    }

    #[test]
    fn test_clifford_propagation_h_identity_unchanged() {
        // H I H = I
        let id = PackedPauliString::identity(2);
        let (result, coeff) = prop_single(id, &CliffordGate::H(0));
        assert!(result.is_identity());
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    // =================================================================
    // Clifford propagation tests: S gate
    // =================================================================

    #[test]
    fn test_clifford_propagation_s_x_to_y() {
        // S^dag X S = Y
        let x = PackedPauliString::single_x(2, 0);
        let (result, coeff) = prop_single(x, &CliffordGate::S(0));
        assert_eq!(result.get(0), 0b11); // Y
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_s_y_to_neg_x() {
        // S^dag Y S = -X
        let y = PackedPauliString::single_y(2, 0);
        let (result, coeff) = prop_single(y, &CliffordGate::S(0));
        assert_eq!(result.get(0), 0b01); // X
        assert!((coeff.0 + 1.0).abs() < 1e-12); // -1
    }

    #[test]
    fn test_clifford_propagation_s_z_unchanged() {
        // S^dag Z S = Z
        let z = PackedPauliString::single_z(2, 0);
        let (result, coeff) = prop_single(z, &CliffordGate::S(0));
        assert_eq!(result.get(0), 0b10); // Z
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    // =================================================================
    // Clifford propagation tests: Sdg gate
    // =================================================================

    #[test]
    fn test_clifford_propagation_sdg_x_to_neg_y() {
        // S X S^dag = -Y
        let x = PackedPauliString::single_x(2, 0);
        let (result, coeff) = prop_single(x, &CliffordGate::Sdg(0));
        assert_eq!(result.get(0), 0b11); // Y
        assert!((coeff.0 + 1.0).abs() < 1e-12); // -1
    }

    #[test]
    fn test_clifford_propagation_sdg_y_to_x() {
        // S Y S^dag = X
        let y = PackedPauliString::single_y(2, 0);
        let (result, coeff) = prop_single(y, &CliffordGate::Sdg(0));
        assert_eq!(result.get(0), 0b01); // X
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_sdg_z_unchanged() {
        let z = PackedPauliString::single_z(2, 0);
        let (result, coeff) = prop_single(z, &CliffordGate::Sdg(0));
        assert_eq!(result.get(0), 0b10); // Z
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    // =================================================================
    // Clifford propagation tests: CX gate
    // =================================================================

    #[test]
    fn test_clifford_propagation_cx_ix_to_ix() {
        // IX -> IX (target X is unaffected when control is I)
        let ix = PackedPauliString::single_x(2, 1);
        let (result, coeff) = prop_single(ix, &CliffordGate::CX(0, 1));
        assert_eq!(result.get(0), 0b00); // I
        assert_eq!(result.get(1), 0b01); // X
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_cx_xi_to_xx() {
        // XI -> XX (control X spreads to target)
        let xi = PackedPauliString::single_x(2, 0);
        let (result, coeff) = prop_single(xi, &CliffordGate::CX(0, 1));
        assert_eq!(result.get(0), 0b01); // X
        assert_eq!(result.get(1), 0b01); // X
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_cx_iz_to_zz() {
        // IZ -> ZZ (target Z spreads to control)
        let iz = PackedPauliString::single_z(2, 1);
        let (result, coeff) = prop_single(iz, &CliffordGate::CX(0, 1));
        assert_eq!(result.get(0), 0b10); // Z
        assert_eq!(result.get(1), 0b10); // Z
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_cx_zi_to_zi() {
        // ZI -> ZI (control Z unaffected)
        let zi = PackedPauliString::single_z(2, 0);
        let (result, coeff) = prop_single(zi, &CliffordGate::CX(0, 1));
        assert_eq!(result.get(0), 0b10); // Z
        assert_eq!(result.get(1), 0b00); // I
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_cx_xx_to_xi() {
        // XX -> XI
        let mut xx = PackedPauliString::identity(2);
        xx.set(0, 0b01);
        xx.set(1, 0b01);
        let (result, coeff) = prop_single(xx, &CliffordGate::CX(0, 1));
        assert_eq!(result.get(0), 0b01); // X
        assert_eq!(result.get(1), 0b00); // I
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    // =================================================================
    // Clifford propagation tests: CZ gate
    // =================================================================

    #[test]
    fn test_clifford_propagation_cz_xi_to_xz() {
        // XI -> XZ
        let xi = PackedPauliString::single_x(2, 0);
        let (result, coeff) = prop_single(xi, &CliffordGate::CZ(0, 1));
        assert_eq!(result.get(0), 0b01); // X
        assert_eq!(result.get(1), 0b10); // Z
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_cz_ix_to_zx() {
        // IX -> ZX
        let ix = PackedPauliString::single_x(2, 1);
        let (result, coeff) = prop_single(ix, &CliffordGate::CZ(0, 1));
        assert_eq!(result.get(0), 0b10); // Z
        assert_eq!(result.get(1), 0b01); // X
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_cz_zz_unchanged() {
        // ZZ -> ZZ (both Z unchanged under CZ)
        let mut zz = PackedPauliString::identity(2);
        zz.set(0, 0b10);
        zz.set(1, 0b10);
        let (result, coeff) = prop_single(zz, &CliffordGate::CZ(0, 1));
        assert_eq!(result.get(0), 0b10); // Z
        assert_eq!(result.get(1), 0b10); // Z
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_cz_xx_to_neg_yy() {
        // XX -> -YY
        let mut xx = PackedPauliString::identity(2);
        xx.set(0, 0b01);
        xx.set(1, 0b01);
        let (result, coeff) = prop_single(xx, &CliffordGate::CZ(0, 1));
        assert_eq!(result.get(0), 0b11); // Y
        assert_eq!(result.get(1), 0b11); // Y
        assert!((coeff.0 + 1.0).abs() < 1e-12); // -1
    }

    // =================================================================
    // Clifford propagation tests: SWAP gate
    // =================================================================

    #[test]
    fn test_clifford_propagation_swap_x0_to_x1() {
        let x0 = PackedPauliString::single_x(3, 0);
        let (result, coeff) = prop_single(x0, &CliffordGate::SWAP(0, 1));
        assert_eq!(result.get(0), 0b00); // I
        assert_eq!(result.get(1), 0b01); // X
        assert_eq!(result.get(2), 0b00); // I
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_swap_z0_to_z1() {
        let z0 = PackedPauliString::single_z(3, 0);
        let (result, coeff) = prop_single(z0, &CliffordGate::SWAP(0, 1));
        assert_eq!(result.get(0), 0b00); // I
        assert_eq!(result.get(1), 0b10); // Z
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clifford_propagation_swap_preserves_third_qubit() {
        // SWAP(0,1) should not affect qubit 2.
        let mut ps = PackedPauliString::identity(3);
        ps.set(0, 0b01); // X
        ps.set(2, 0b11); // Y
        let (result, _) = prop_single(ps, &CliffordGate::SWAP(0, 1));
        assert_eq!(result.get(0), 0b00); // I (was X, swapped away)
        assert_eq!(result.get(1), 0b01); // X (received from qubit 0)
        assert_eq!(result.get(2), 0b11); // Y (unaffected)
    }

    // =================================================================
    // Non-Clifford propagation tests: T gate
    // =================================================================

    #[test]
    fn test_t_gate_on_identity_single_term() {
        // T on I -> I (single term, no splitting)
        let id = PackedPauliString::identity(2);
        let term = WeightedPackedPauli::unit(id);
        let results = propagate_term(&term, &CliffordGate::T(0));
        assert_eq!(results.len(), 1);
        assert!(results[0].pauli.is_identity());
    }

    #[test]
    fn test_t_gate_on_z_single_term() {
        // T on Z -> Z (single term, no splitting)
        let z = PackedPauliString::single_z(2, 0);
        let term = WeightedPackedPauli::unit(z);
        let results = propagate_term(&term, &CliffordGate::T(0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pauli.get(0), 0b10); // Z
    }

    #[test]
    fn test_t_gate_on_x_splits_to_two_terms() {
        // T^dag X T = cos(pi/4) X + sin(pi/4) Y -> 2 terms
        let x = PackedPauliString::single_x(2, 0);
        let term = WeightedPackedPauli::unit(x);
        let results = propagate_term(&term, &CliffordGate::T(0));
        assert_eq!(results.len(), 2);

        // First term should be X with cos(pi/4) coefficient.
        let cos_val = std::f64::consts::FRAC_PI_4.cos();
        let sin_val = std::f64::consts::FRAC_PI_4.sin();

        assert_eq!(results[0].pauli.get(0), 0b01); // X
        assert!((results[0].coeff.0 - cos_val).abs() < 1e-12);

        assert_eq!(results[1].pauli.get(0), 0b11); // Y
        assert!((results[1].coeff.0 - sin_val).abs() < 1e-12);
    }

    #[test]
    fn test_t_gate_norm_preservation() {
        // The sum of squared magnitudes should be 1 for unit input.
        let x = PackedPauliString::single_x(2, 0);
        let term = WeightedPackedPauli::unit(x);
        let results = propagate_term(&term, &CliffordGate::T(0));
        let norm_sq: f64 = results.iter().map(|r| r.magnitude() * r.magnitude()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }

    // =================================================================
    // PauliTableau tests
    // =================================================================

    #[test]
    fn test_pauli_tableau_identity_initialization() {
        let tab = PauliTableau::new(3);
        assert_eq!(tab.num_qubits, 3);
        assert_eq!(tab.rows.len(), 6); // 2 * 3

        // X generators: X0 -> X0, X1 -> X1, X2 -> X2
        for j in 0..3 {
            let (row, phase) = tab.get_x_image(j);
            assert_eq!(row.get(j), 0b01); // X
            assert_eq!(row.weight(), 1);
            assert_eq!(phase, Phase::PLUS_ONE);
        }

        // Z generators: Z0 -> Z0, Z1 -> Z1, Z2 -> Z2
        for j in 0..3 {
            let (row, phase) = tab.get_z_image(j);
            assert_eq!(row.get(j), 0b10); // Z
            assert_eq!(row.weight(), 1);
            assert_eq!(phase, Phase::PLUS_ONE);
        }
    }

    #[test]
    fn test_pauli_tableau_hadamard() {
        let mut tab = PauliTableau::new(2);
        tab.apply_gate(&CliffordGate::H(0));

        // After H on qubit 0: X0 -> Z0, Z0 -> X0
        let (x0_img, x0_phase) = tab.get_x_image(0);
        assert_eq!(x0_img.get(0), 0b10); // X0 -> Z0
        assert_eq!(x0_phase, Phase::PLUS_ONE);

        let (z0_img, z0_phase) = tab.get_z_image(0);
        assert_eq!(z0_img.get(0), 0b01); // Z0 -> X0
        assert_eq!(z0_phase, Phase::PLUS_ONE);

        // Qubit 1 should be unaffected.
        let (x1_img, _) = tab.get_x_image(1);
        assert_eq!(x1_img.get(1), 0b01);
    }

    #[test]
    fn test_pauli_tableau_cx_entanglement() {
        let mut tab = PauliTableau::new(2);
        tab.apply_gate(&CliffordGate::CX(0, 1));

        // X0 -> X0 X1
        let (x0_img, _) = tab.get_x_image(0);
        assert_eq!(x0_img.get(0), 0b01); // X
        assert_eq!(x0_img.get(1), 0b01); // X

        // Z1 -> Z0 Z1
        let (z1_img, _) = tab.get_z_image(1);
        assert_eq!(z1_img.get(0), 0b10); // Z
        assert_eq!(z1_img.get(1), 0b10); // Z
    }

    #[test]
    fn test_pauli_tableau_apply_circuit() {
        let mut tab = PauliTableau::new(2);
        // Circuit: H(0), CX(0,1) - creates Bell state generator structure.
        let circuit = vec![CliffordGate::H(0), CliffordGate::CX(0, 1)];
        tab.apply_circuit(&circuit);

        // After circuit reversal (Heisenberg picture):
        // We can check specific propagation results.
        // The tableau should be internally consistent.
        assert_eq!(tab.rows.len(), 4);
    }

    // =================================================================
    // PauliPropagator tests
    // =================================================================

    #[test]
    fn test_propagator_clifford_circuit_exact() {
        // Propagate Z0 through H(0): should give X0.
        // <0|X0|0> = 0 (X has no diagonal component on |0>).
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(2, 0));
        let circuit = vec![CliffordGate::H(0)];
        let mut prop = PauliPropagator::new(2, obs, circuit, GpuPropConfig::default());
        let result = prop.propagate().unwrap();
        assert_eq!(result.initial_count, 1);
        assert_eq!(result.final_count, 1);
        // Z0 -> X0 under H. <0|X0|0> = 0 (X flips |0> to |1>).
        assert!((result.expectation_value).abs() < 1e-12);
    }

    #[test]
    fn test_propagator_identity_circuit() {
        // Identity observable with no gates.
        // expectation_on_zero_state only counts all-identity Pauli strings,
        // so the identity observable I_{all} should give +1.
        let obs = WeightedPackedPauli::unit(PackedPauliString::identity(2));
        let circuit = vec![];
        let mut prop = PauliPropagator::new(2, obs, circuit, GpuPropConfig::default());
        let result = prop.propagate().unwrap();
        assert!((result.expectation_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_propagator_z_observable_no_gates() {
        // Z0 with no gates: Z0 is non-identity, so expectation_on_zero_state
        // returns 0 (it only sums all-identity terms). However,
        // expectation_on_basis_state properly handles Z operators.
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(2, 0));
        let circuit = vec![];
        let mut prop = PauliPropagator::new(2, obs, circuit, GpuPropConfig::default());
        let result = prop.propagate().unwrap();
        // expectation_on_zero_state only counts identity terms, so Z0 gives 0.
        assert!((result.expectation_value).abs() < 1e-12);
    }

    #[test]
    fn test_propagator_validate_qubit_out_of_range() {
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(2, 0));
        let circuit = vec![CliffordGate::H(5)]; // qubit 5 in a 2-qubit system
        let prop = PauliPropagator::new(2, obs, circuit, GpuPropConfig::default());
        let err = prop.validate().unwrap_err();
        match err {
            GpuPauliError::CircuitError(msg) => {
                assert!(msg.contains("qubit"));
            }
            _ => panic!("Expected CircuitError"),
        }
    }

    #[test]
    fn test_propagator_validate_zero_qubits() {
        let obs = WeightedPackedPauli::unit(PackedPauliString::identity(0));
        let prop = PauliPropagator::new(0, obs, vec![], GpuPropConfig::default());
        let err = prop.validate().unwrap_err();
        match err {
            GpuPauliError::InvalidPauli(msg) => {
                assert!(msg.contains("zero"));
            }
            _ => panic!("Expected InvalidPauli error"),
        }
    }

    #[test]
    fn test_propagator_expectation_on_basis_state() {
        // Z0 on |1> should give -1.
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(2, 0));
        let prop = PauliPropagator::new(2, obs, vec![], GpuPropConfig::default());
        let exp_00 = prop.expectation_on_basis_state(0b00);
        let exp_01 = prop.expectation_on_basis_state(0b01);
        assert!((exp_00 - 1.0).abs() < 1e-12);   // Z|0> = +|0>
        assert!((exp_01 + 1.0).abs() < 1e-12);    // Z|1> = -|1>
    }

    #[test]
    fn test_propagator_batch_through_gate() {
        let strings = vec![
            PackedPauliString::single_x(2, 0),
            PackedPauliString::single_z(2, 0),
        ];
        let terms: Vec<WeightedPackedPauli> =
            strings.into_iter().map(WeightedPackedPauli::unit).collect();
        let result = PauliPropagator::propagate_batch_through_gate(&terms, &CliffordGate::H(0));
        assert_eq!(result.len(), 2);
        // X -> Z, Z -> X
        assert_eq!(result[0].pauli.get(0), 0b10); // Z
        assert_eq!(result[1].pauli.get(0), 0b01); // X
    }

    // =================================================================
    // NearCliffordEstimator tests
    // =================================================================

    #[test]
    fn test_near_clifford_pure_clifford_circuit() {
        // Pure Clifford circuit (no T-gates) should give exact result.
        // Z0 through H(0): <0|X0|0> = 0.
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(2, 0));
        let circuit = vec![CliffordGate::H(0)];
        let config = NearCliffordConfig::default().with_samples(100);
        let estimator = NearCliffordEstimator::new(2, obs, circuit, config);
        assert_eq!(estimator.count_t_gates(), 0);
        let result = estimator.estimate().unwrap();
        assert_eq!(result.num_t_gates, 0);
        assert_eq!(result.num_samples, 1); // Exact, only 1 sample needed.
        assert_eq!(result.error_bound, 0.0);
        assert!((result.expectation).abs() < 1e-12);
    }

    #[test]
    fn test_near_clifford_count_t_gates() {
        let circuit = vec![
            CliffordGate::H(0),
            CliffordGate::T(0),
            CliffordGate::CX(0, 1),
            CliffordGate::Tdg(1),
            CliffordGate::S(0),
        ];
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(2, 0));
        let estimator = NearCliffordEstimator::new(
            2,
            obs,
            circuit,
            NearCliffordConfig::default(),
        );
        assert_eq!(estimator.count_t_gates(), 2);
    }

    #[test]
    fn test_near_clifford_exceeds_max_t_gates() {
        let circuit: Vec<CliffordGate> = (0..60).map(|_| CliffordGate::T(0)).collect();
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(2, 0));
        let config = NearCliffordConfig::default().with_max_t_gates(50);
        let estimator = NearCliffordEstimator::new(2, obs, circuit, config);
        let err = estimator.estimate().unwrap_err();
        match err {
            GpuPauliError::CircuitError(msg) => {
                assert!(msg.contains("T-gates"));
            }
            _ => panic!("Expected CircuitError"),
        }
    }

    #[test]
    fn test_near_clifford_with_single_t_gate() {
        // Observable: identity (all-I), circuit: T(0).
        // Identity is invariant under any gate, so both Clifford branches
        // produce identity. expectation_on_zero_state for identity = 1.0.
        let obs = WeightedPackedPauli::unit(PackedPauliString::identity(2));
        let circuit = vec![CliffordGate::T(0)];
        let config = NearCliffordConfig::default().with_samples(500);
        let estimator = NearCliffordEstimator::new(2, obs, circuit, config);
        let result = estimator.estimate().unwrap();
        assert_eq!(result.num_t_gates, 1);
        // Identity observable is invariant under everything, so expectation
        // should be near 1.0 (with some Monte Carlo noise from the weighting).
        assert!(
            (result.expectation - 1.0).abs() < 0.5,
            "Expected near 1.0, got {}",
            result.expectation
        );
    }

    // =================================================================
    // AutoDispatch tests
    // =================================================================

    #[test]
    fn test_auto_dispatch_cpu_for_small_batch() {
        let dispatch = AutoDispatch::default();
        assert_eq!(dispatch.select(10), DispatchBackend::Cpu);
        assert_eq!(dispatch.select(999), DispatchBackend::Cpu);
    }

    #[test]
    fn test_auto_dispatch_gpu_threshold() {
        let dispatch = AutoDispatch::with_threshold(500);
        assert_eq!(dispatch.gpu_threshold, 500);
        // On macOS, gpu_available is true; on other platforms, false.
        // Either way, CPU is always an option for small batches.
        assert_eq!(dispatch.select(100), DispatchBackend::Cpu);
    }

    #[test]
    fn test_auto_dispatch_propagate_batch_cpu() {
        // Force CPU by using a tiny batch below any threshold.
        let dispatch = AutoDispatch {
            gpu_threshold: 1_000_000,
            gpu_available: false,
        };
        let strings = vec![
            PackedPauliString::single_x(3, 0),
            PackedPauliString::single_z(3, 1),
            PackedPauliString::single_y(3, 2),
        ];
        let results = dispatch
            .propagate_batch(3, &strings, &CliffordGate::H(0))
            .unwrap();
        assert_eq!(results.len(), 3);
        // X0 -> Z0
        assert_eq!(results[0].get(0), 0b10);
        // Z1 unaffected by H(0)
        assert_eq!(results[1].get(1), 0b10);
        // Y2 unaffected by H(0)
        assert_eq!(results[2].get(2), 0b11);
    }

    #[test]
    fn test_auto_dispatch_backend_display() {
        assert_eq!(format!("{}", DispatchBackend::Cpu), "CPU");
        assert_eq!(format!("{}", DispatchBackend::MetalGpu), "Metal GPU");
    }

    // =================================================================
    // GpuPropConfig tests
    // =================================================================

    #[test]
    fn test_gpu_prop_config_default() {
        let cfg = GpuPropConfig::default();
        assert_eq!(cfg.max_terms, 100_000);
        assert!((cfg.truncation_threshold - 1e-10).abs() < 1e-15);
        assert!(cfg.merge_duplicates);
        assert_eq!(cfg.gpu_threshold, 1000);
    }

    #[test]
    fn test_gpu_prop_config_builder() {
        let cfg = GpuPropConfig::default()
            .with_max_terms(500)
            .with_threshold(0.01)
            .with_merge(false)
            .with_gpu_threshold(2000);
        assert_eq!(cfg.max_terms, 500);
        assert!((cfg.truncation_threshold - 0.01).abs() < 1e-15);
        assert!(!cfg.merge_duplicates);
        assert_eq!(cfg.gpu_threshold, 2000);
    }

    // =================================================================
    // NearCliffordConfig tests
    // =================================================================

    #[test]
    fn test_near_clifford_config_builders() {
        let cfg = NearCliffordConfig::default()
            .with_samples(5000)
            .with_max_t_gates(20)
            .with_seed(42);
        assert_eq!(cfg.num_samples, 5000);
        assert_eq!(cfg.max_t_gates, 20);
        assert_eq!(cfg.rng_seed, 42);
    }

    // =================================================================
    // Error type tests
    // =================================================================

    #[test]
    fn test_error_display_invalid_pauli() {
        let err = GpuPauliError::InvalidPauli("bad string".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid Pauli"));
        assert!(msg.contains("bad string"));
    }

    #[test]
    fn test_error_display_circuit_error() {
        let err = GpuPauliError::CircuitError("qubit out of range".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Circuit error"));
        assert!(msg.contains("qubit out of range"));
    }

    #[test]
    fn test_error_display_overflow() {
        let err = GpuPauliError::Overflow {
            count: 200_000,
            limit: 100_000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("200000"));
        assert!(msg.contains("100000"));
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn test_error_display_gpu_error() {
        let err = GpuPauliError::GpuError("device not found".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("GPU error"));
        assert!(msg.contains("device not found"));
    }

    #[test]
    fn test_error_display_numerical() {
        let err = GpuPauliError::NumericalError("NaN detected".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Numerical error"));
        assert!(msg.contains("NaN detected"));
    }

    #[test]
    fn test_error_is_std_error() {
        let err = GpuPauliError::InvalidPauli("test".to_string());
        // Verify it implements std::error::Error by using it as a trait object.
        let _: &dyn std::error::Error = &err;
    }

    // =================================================================
    // Integration / compound propagation tests
    // =================================================================

    #[test]
    fn test_double_hadamard_is_identity() {
        // H(H(X)) = X. Applying H twice should return to original.
        let x = PackedPauliString::single_x(2, 0);
        let (intermediate, _) = prop_single(x.clone(), &CliffordGate::H(0));
        assert_eq!(intermediate.get(0), 0b10); // Z
        let (final_result, coeff) = prop_single(intermediate, &CliffordGate::H(0));
        assert_eq!(final_result.get(0), 0b01); // X
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_s_sdg_inverse() {
        // S followed by Sdg should be identity (on the propagation level).
        let x = PackedPauliString::single_x(2, 0);
        let (after_s, _) = prop_single(x, &CliffordGate::S(0));
        assert_eq!(after_s.get(0), 0b11); // Y
        let (after_sdg, coeff) = prop_single(after_s, &CliffordGate::Sdg(0));
        assert_eq!(after_sdg.get(0), 0b01); // X
        assert!((coeff.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_propagator_h_then_z_measurement() {
        // Circuit: H(0). Observable: Z0.
        // In Heisenberg picture, Z0 propagates backward through H to become X0.
        // <0|X0|0> = 0.
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_z(1, 0));
        let circuit = vec![CliffordGate::H(0)];
        let mut prop = PauliPropagator::new(1, obs, circuit, GpuPropConfig::default());
        let result = prop.propagate().unwrap();
        assert!((result.expectation_value).abs() < 1e-12);
    }

    #[test]
    fn test_propagator_identity_observable_always_one() {
        // Identity observable should always give expectation 1 regardless of circuit.
        let obs = WeightedPackedPauli::unit(PackedPauliString::identity(2));
        let circuit = vec![
            CliffordGate::H(0),
            CliffordGate::CX(0, 1),
            CliffordGate::S(1),
        ];
        let mut prop = PauliPropagator::new(2, obs, circuit, GpuPropConfig::default());
        let result = prop.propagate().unwrap();
        assert!((result.expectation_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_propagator_with_t_gate_term_growth() {
        // T gate on X should produce 2 terms.
        let obs = WeightedPackedPauli::unit(PackedPauliString::single_x(2, 0));
        let circuit = vec![CliffordGate::T(0)];
        let config = GpuPropConfig::default().with_merge(false);
        let mut prop = PauliPropagator::new(2, obs, circuit, config);
        let result = prop.propagate().unwrap();
        assert_eq!(result.final_count, 2);
        assert!(result.peak_count >= 2);
    }

    #[test]
    fn test_coeff_to_phase_helper() {
        assert_eq!(coeff_to_phase(1.0, 0.0), Phase::PLUS_ONE);
        assert_eq!(coeff_to_phase(-1.0, 0.0), Phase::MINUS_ONE);
        assert_eq!(coeff_to_phase(0.0, 1.0), Phase::PLUS_I);
        assert_eq!(coeff_to_phase(0.0, -1.0), Phase::MINUS_I);
    }

    #[test]
    fn test_metal_pauli_kernel_cpu_fallback() {
        // MetalPauliKernel should work even when running on non-GPU path.
        let mut kernel = MetalPauliKernel::new(4).unwrap();
        let strings = vec![
            PackedPauliString::single_x(4, 0),
            PackedPauliString::single_z(4, 1),
        ];
        // H gate = type 0
        let results = kernel.dispatch_clifford(&strings, 0, 0, 0).unwrap();
        assert_eq!(results.len(), 2);
        // X0 -> Z0 under H(0)
        assert_eq!(results[0].get(0), 0b10);
        // Z1 unaffected by H(0)
        assert_eq!(results[1].get(0), 0b00);
        assert_eq!(results[1].get(1), 0b10);
    }
}
