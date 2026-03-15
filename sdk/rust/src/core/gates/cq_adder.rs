//! Classical-Quantum Hybrid Arithmetic
//!
//! Quantum arithmetic circuits (adders, multipliers, comparators) that serve as
//! essential building blocks for Shor's algorithm, quantum chemistry, and
//! combinatorial optimization.
//!
//! # Implemented Algorithms
//!
//! - **Draper QFT Adder**: O(n^2) controlled rotations, no ancilla qubits.
//!   Transforms via QFT, applies phase kicks, then inverse QFT.
//!   Reference: Draper, arXiv:quant-ph/0008033.
//!
//! - **Cuccaro Ripple-Carry Adder**: O(n) depth with 1 ancilla qubit.
//!   Uses CNOT/Toffoli carry propagation. Better for near-term devices.
//!   Reference: Cuccaro et al., arXiv:quant-ph/0410184.
//!
//! - **Classical-Quantum Adder**: Add a known classical constant `c` to a
//!   quantum register |x> -> |x+c>. Fewer gates since one operand is classical.
//!
//! - **Modular Arithmetic**: (a + b) mod N and a * x mod N for Shor's algorithm.
//!
//! - **Quantum Comparator**: |a>|b>|0> -> |a>|b>|a >= b> via subtraction and
//!   sign-bit extraction.
//!
//! # Integration
//!
//! Used by [`crate::shor`] for modular exponentiation. Self-contained with an
//! inline statevector simulator for correctness testing.

use std::f64::consts::PI;

use crate::{GateOperations, QuantumState};

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from quantum arithmetic operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ArithmeticError {
    /// The two quantum registers have different sizes.
    RegisterSizeMismatch { a_size: usize, b_size: usize },
    /// The result may overflow the target register width.
    OverflowRisk {
        register_bits: usize,
        max_value: u64,
    },
    /// The modulus N is invalid (must be >= 2 and fit in register).
    InvalidModulus { modulus: u64, reason: &'static str },
    /// A register index is out of bounds.
    QubitIndexOutOfBounds { index: usize, total: usize },
}

impl std::fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArithmeticError::RegisterSizeMismatch { a_size, b_size } => {
                write!(
                    f,
                    "Register size mismatch: a has {} qubits, b has {}",
                    a_size, b_size
                )
            }
            ArithmeticError::OverflowRisk {
                register_bits,
                max_value,
            } => {
                write!(
                    f,
                    "Overflow risk: {} bits cannot hold value {}",
                    register_bits, max_value
                )
            }
            ArithmeticError::InvalidModulus { modulus, reason } => {
                write!(f, "Invalid modulus {}: {}", modulus, reason)
            }
            ArithmeticError::QubitIndexOutOfBounds { index, total } => {
                write!(f, "Qubit index {} out of bounds (total: {})", index, total)
            }
        }
    }
}

impl std::error::Error for ArithmeticError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Selects which adder implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdderType {
    /// QFT-based Draper adder: O(n^2) gates, 0 ancilla.
    Draper,
    /// Ripple-carry Cuccaro adder: O(n) depth, 1 ancilla.
    Cuccaro,
    /// Automatically choose based on register size.
    Auto,
}

/// Configuration for quantum arithmetic operations.
#[derive(Debug, Clone)]
pub struct ArithmeticConfig {
    /// Which adder circuit to use.
    pub adder_type: AdderType,
    /// Whether to use approximate QFT (truncate small rotations).
    pub use_approximation: bool,
    /// Rotation precision: keep controlled rotations where k <= this value.
    /// Rotations of angle 2*pi / 2^k with k > precision are dropped.
    pub rotation_precision: usize,
}

impl Default for ArithmeticConfig {
    fn default() -> Self {
        ArithmeticConfig {
            adder_type: AdderType::Auto,
            use_approximation: false,
            rotation_precision: 10,
        }
    }
}

impl ArithmeticConfig {
    /// Create a new configuration builder starting from defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the adder type.
    pub fn with_adder_type(mut self, adder_type: AdderType) -> Self {
        self.adder_type = adder_type;
        self
    }

    /// Enable or disable approximate QFT.
    pub fn with_approximation(mut self, enabled: bool) -> Self {
        self.use_approximation = enabled;
        self
    }

    /// Set the rotation precision threshold.
    pub fn with_rotation_precision(mut self, precision: usize) -> Self {
        self.rotation_precision = precision;
        self
    }

    /// Resolve `Auto` to a concrete adder type based on register size.
    pub fn resolve_adder(&self, register_bits: usize) -> AdderType {
        match self.adder_type {
            AdderType::Auto => {
                if register_bits < 8 {
                    AdderType::Cuccaro
                } else {
                    AdderType::Draper
                }
            }
            concrete => concrete,
        }
    }

    /// Check whether a rotation of order k should be kept.
    fn keep_rotation(&self, k: usize) -> bool {
        !self.use_approximation || k <= self.rotation_precision
    }
}

// ============================================================
// ARITHMETIC GATE ENUM
// ============================================================

/// A gate instruction emitted by arithmetic circuit generators.
#[derive(Debug, Clone, PartialEq)]
pub enum ArithmeticGate {
    /// Hadamard on a single qubit.
    H(usize),
    /// Pauli-X (NOT) on a single qubit.
    X(usize),
    /// CNOT: control, target.
    CNOT(usize, usize),
    /// Toffoli (CCX): control1, control2, target.
    Toffoli(usize, usize, usize),
    /// Controlled phase rotation: control, target, angle (radians).
    CPhase(usize, usize, f64),
    /// Single-qubit phase rotation: qubit, angle (radians).
    Phase(usize, f64),
}

impl ArithmeticGate {
    /// Apply this gate to a quantum state using the nQPU-Metal backend.
    pub fn apply(&self, state: &mut QuantumState) {
        match *self {
            ArithmeticGate::H(q) => GateOperations::h(state, q),
            ArithmeticGate::X(q) => GateOperations::x(state, q),
            ArithmeticGate::CNOT(c, t) => GateOperations::cnot(state, c, t),
            ArithmeticGate::Toffoli(c1, c2, t) => GateOperations::toffoli(state, c1, c2, t),
            ArithmeticGate::CPhase(c, t, phi) => GateOperations::cphase(state, c, t, phi),
            ArithmeticGate::Phase(q, phi) => GateOperations::phase(state, q, phi),
        }
    }

    /// Return the inverse (adjoint) of this gate.
    pub fn inverse(&self) -> ArithmeticGate {
        match *self {
            ArithmeticGate::H(q) => ArithmeticGate::H(q),
            ArithmeticGate::X(q) => ArithmeticGate::X(q),
            ArithmeticGate::CNOT(c, t) => ArithmeticGate::CNOT(c, t),
            ArithmeticGate::Toffoli(c1, c2, t) => ArithmeticGate::Toffoli(c1, c2, t),
            ArithmeticGate::CPhase(c, t, phi) => ArithmeticGate::CPhase(c, t, -phi),
            ArithmeticGate::Phase(q, phi) => ArithmeticGate::Phase(q, -phi),
        }
    }
}

// ============================================================
// QUANTUM ADDER TRAIT
// ============================================================

/// Trait for quantum adder circuits.
///
/// An adder takes two registers (a, b) specified by qubit indices and produces
/// a gate sequence that computes |a>|b> -> |a>|a+b> (result stored in b).
pub trait QuantumAdder {
    /// Generate the gate sequence for addition.
    ///
    /// `a_qubits` and `b_qubits` are qubit indices (LSB first).
    /// The result |a+b> is stored in the `b_qubits` register.
    fn add(
        &self,
        a_qubits: &[usize],
        b_qubits: &[usize],
    ) -> Result<Vec<ArithmeticGate>, ArithmeticError>;

    /// Generate the inverse (subtraction) gate sequence.
    fn subtract(
        &self,
        a_qubits: &[usize],
        b_qubits: &[usize],
    ) -> Result<Vec<ArithmeticGate>, ArithmeticError> {
        let forward = self.add(a_qubits, b_qubits)?;
        Ok(forward.iter().rev().map(|g| g.inverse()).collect())
    }
}

// ============================================================
// QFT HELPERS (without bit-reversal swaps)
// ============================================================

/// Generate QFT gate sequence on the given qubits WITHOUT the final
/// bit-reversal swap step.
///
/// The qubits array is indexed from MSB (index 0) to LSB (index n-1).
/// This is the convention used inside the Draper adder where the QFT
/// and inverse QFT swaps cancel, so we omit them.
///
/// For qubit ordering: qubits[0] is the most significant bit.
fn qft_no_swap(qubits: &[usize], config: &ArithmeticConfig) -> Vec<ArithmeticGate> {
    let n = qubits.len();
    let mut gates = Vec::new();

    for i in 0..n {
        gates.push(ArithmeticGate::H(qubits[i]));
        for j in (i + 1)..n {
            let k = j - i + 1; // rotation order: R_k has angle 2*pi/2^k
            if config.keep_rotation(k) {
                let angle = 2.0 * PI / (1u64 << k) as f64;
                gates.push(ArithmeticGate::CPhase(qubits[j], qubits[i], angle));
            }
        }
    }

    gates
}

/// Generate inverse QFT gate sequence WITHOUT bit-reversal swaps.
fn inverse_qft_no_swap(qubits: &[usize], config: &ArithmeticConfig) -> Vec<ArithmeticGate> {
    let forward = qft_no_swap(qubits, config);
    forward.iter().rev().map(|g| g.inverse()).collect()
}

// ============================================================
// DRAPER QFT ADDER
// ============================================================

/// Draper's QFT-based adder.
///
/// Computes |a>|b> -> |a>|a+b mod 2^n> using:
/// 1. QFT on the b register (without swaps)
/// 2. Controlled phase rotations from a into b (in the Fourier basis)
/// 3. Inverse QFT on b (without swaps)
///
/// Gate count: O(n^2). Ancilla: 0.
///
/// The QFT transforms |b> into the Fourier basis where addition becomes
/// phase accumulation. Each bit a[j] (with weight 2^j) contributes a
/// controlled phase rotation of angle 2*pi * 2^j / 2^n to the appropriate
/// Fourier-basis qubit.
pub struct DraperAdder {
    config: ArithmeticConfig,
}

impl DraperAdder {
    /// Create a Draper adder with the given configuration.
    pub fn new(config: ArithmeticConfig) -> Self {
        DraperAdder { config }
    }

    /// Create a Draper adder with default configuration.
    pub fn default_config() -> Self {
        DraperAdder {
            config: ArithmeticConfig {
                adder_type: AdderType::Draper,
                ..ArithmeticConfig::default()
            },
        }
    }
}

impl QuantumAdder for DraperAdder {
    fn add(
        &self,
        a_qubits: &[usize],
        b_qubits: &[usize],
    ) -> Result<Vec<ArithmeticGate>, ArithmeticError> {
        let n = a_qubits.len();
        if n != b_qubits.len() {
            return Err(ArithmeticError::RegisterSizeMismatch {
                a_size: n,
                b_size: b_qubits.len(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        // Both registers are LSB-first: qubits[0] = bit 0 (weight 1),
        // qubits[n-1] = bit n-1 (weight 2^(n-1)).
        //
        // The QFT operates on qubits listed MSB-first. So we reverse b.
        let b_msb: Vec<usize> = b_qubits.iter().copied().rev().collect();

        let mut gates = Vec::new();

        // Step 1: QFT on b register (no swap).
        gates.extend(qft_no_swap(&b_msb, &self.config));

        // Step 2: Phase rotations from a into b (Fourier domain).
        //
        // After QFT (no swap), b_msb[i] corresponds to the (n-1-i)-th
        // Fourier coefficient. To add the integer a = sum_j a_j * 2^j,
        // we need to rotate b_msb[i] by angle:
        //   sum_j a_j * 2*pi * 2^j / 2^(n-i)
        // where j ranges over all a-bits such that the rotation is
        // well-defined.
        //
        // Equivalently, for each a-bit j and b-Fourier qubit i (MSB-first),
        // the rotation angle is 2*pi * 2^j / 2^(n-i) = 2*pi / 2^(n-i-j).
        // This is only applied when n-i-j >= 1, i.e., j <= n-1-i.
        //
        // The rotation order k = n - i - j, so we keep it when k <= precision.
        for i in 0..n {
            for j in 0..n {
                let k_signed: isize = n as isize - i as isize - j as isize;
                if k_signed >= 1 {
                    let k = k_signed as usize;
                    if self.config.keep_rotation(k) {
                        let angle = 2.0 * PI / (1u64 << k) as f64;
                        gates.push(ArithmeticGate::CPhase(a_qubits[j], b_msb[i], angle));
                    }
                }
            }
        }

        // Step 3: Inverse QFT on b register (no swap).
        gates.extend(inverse_qft_no_swap(&b_msb, &self.config));

        Ok(gates)
    }
}

// ============================================================
// CUCCARO RIPPLE-CARRY ADDER
// ============================================================

/// Cuccaro's linear-depth ripple-carry adder.
///
/// Computes |a>|b>|0_ancilla> -> |a>|a+b mod 2^n>|carry>.
/// Uses a single ancilla qubit initialized to |0> which holds the
/// carry propagation state during computation.
///
/// Gate count: O(n). Ancilla: 1.
///
/// The circuit uses a MAJ (majority) and UMA (unmajority-and-add)
/// decomposition for each bit position, propagating the carry from
/// LSB to MSB and then un-computing it while producing the sum bits.
pub struct CuccaroAdder {
    /// Qubit index of the ancilla (carry propagation scratch qubit).
    pub ancilla: usize,
    config: ArithmeticConfig,
}

impl CuccaroAdder {
    /// Create a Cuccaro adder with the specified ancilla qubit.
    pub fn new(ancilla: usize, config: ArithmeticConfig) -> Self {
        CuccaroAdder {
            ancilla,
            config: ArithmeticConfig {
                adder_type: AdderType::Cuccaro,
                ..config
            },
        }
    }

    /// Create with default config and a specified ancilla.
    pub fn with_ancilla(ancilla: usize) -> Self {
        CuccaroAdder {
            ancilla,
            config: ArithmeticConfig {
                adder_type: AdderType::Cuccaro,
                ..ArithmeticConfig::default()
            },
        }
    }

    /// MAJ (Majority) gate on three qubits (c, b, a):
    /// Computes carry = majority(a, b, c_in) and stores it in a.
    fn maj(c: usize, b: usize, a: usize) -> Vec<ArithmeticGate> {
        vec![
            ArithmeticGate::CNOT(a, b),
            ArithmeticGate::CNOT(a, c),
            ArithmeticGate::Toffoli(c, b, a),
        ]
    }

    /// UMA (Un-Majority and Add) gate on three qubits (c, b, a):
    /// Reverses MAJ and writes the sum bit into b.
    fn uma(c: usize, b: usize, a: usize) -> Vec<ArithmeticGate> {
        vec![
            ArithmeticGate::Toffoli(c, b, a),
            ArithmeticGate::CNOT(a, c),
            ArithmeticGate::CNOT(c, b),
        ]
    }
}

impl QuantumAdder for CuccaroAdder {
    fn add(
        &self,
        a_qubits: &[usize],
        b_qubits: &[usize],
    ) -> Result<Vec<ArithmeticGate>, ArithmeticError> {
        let n = a_qubits.len();
        if n != b_qubits.len() {
            return Err(ArithmeticError::RegisterSizeMismatch {
                a_size: n,
                b_size: b_qubits.len(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            // 1-bit addition mod 2: just XOR (CNOT).
            return Ok(vec![ArithmeticGate::CNOT(a_qubits[0], b_qubits[0])]);
        }

        let mut gates = Vec::new();

        // ---- Forward MAJ sweep (bits 0 through n-2) ----
        //
        // Position 0: MAJ(ancilla, b[0], a[0])
        //   After: a[0] = carry_out_0 = majority(ancilla, b[0], a[0])
        //          b[0] is modified (XOR'd)
        //          ancilla is modified (XOR'd)
        gates.extend(Self::maj(self.ancilla, b_qubits[0], a_qubits[0]));

        // Positions 1 through n-2: MAJ(a[i-1], b[i], a[i])
        //   After: a[i] = carry_out_i
        for i in 1..(n - 1) {
            gates.extend(Self::maj(a_qubits[i - 1], b_qubits[i], a_qubits[i]));
        }

        // ---- MSB sum computation ----
        //
        // The carry into position n-1 is in a[n-2].
        // The MSB sum = a[n-1] XOR b[n-1] XOR carry_in(n-1)
        // We compute this with two CNOTs:
        //   b[n-1] ^= carry_in(n-1) = a[n-2]
        //   b[n-1] ^= a[n-1]
        gates.push(ArithmeticGate::CNOT(a_qubits[n - 2], b_qubits[n - 1]));
        gates.push(ArithmeticGate::CNOT(a_qubits[n - 1], b_qubits[n - 1]));

        // ---- Reverse UMA sweep (bits n-2 down to 0) ----
        //
        // Each UMA un-computes the MAJ at that position and writes sum into b[i].
        for i in (1..(n - 1)).rev() {
            gates.extend(Self::uma(a_qubits[i - 1], b_qubits[i], a_qubits[i]));
        }

        // Position 0: UMA(ancilla, b[0], a[0])
        gates.extend(Self::uma(self.ancilla, b_qubits[0], a_qubits[0]));

        Ok(gates)
    }
}

// ============================================================
// CLASSICAL-QUANTUM ADDER
// ============================================================

/// Add a known classical integer `c` to a quantum register |x> -> |x + c mod 2^n>.
///
/// This is more efficient than the quantum-quantum adder because the
/// classical bits are known at circuit compilation time, so we only
/// need unconditional phase rotations (no control qubits from a quantum
/// register).
///
/// Uses QFT-based addition in the Fourier domain.
pub struct ClassicalQuantumAdder {
    /// The classical constant to add.
    pub constant: u64,
    config: ArithmeticConfig,
}

impl ClassicalQuantumAdder {
    /// Create a new classical-quantum adder for the given constant.
    pub fn new(constant: u64, config: ArithmeticConfig) -> Self {
        ClassicalQuantumAdder { constant, config }
    }

    /// Create with default configuration.
    pub fn with_constant(constant: u64) -> Self {
        ClassicalQuantumAdder {
            constant,
            config: ArithmeticConfig::default(),
        }
    }

    /// Generate the gate sequence for |x> -> |x + c mod 2^n>.
    ///
    /// `x_qubits` are the qubit indices of the quantum register (LSB first).
    pub fn add_to_register(&self, x_qubits: &[usize]) -> Vec<ArithmeticGate> {
        let n = x_qubits.len();
        if n == 0 {
            return Vec::new();
        }

        let x_msb: Vec<usize> = x_qubits.iter().copied().rev().collect();

        let mut gates = Vec::new();

        // Step 1: QFT on x register (no swap).
        gates.extend(qft_no_swap(&x_msb, &self.config));

        // Step 2: Phase rotations from classical constant c.
        //
        // To add integer c = sum_j c_j * 2^j to x in the Fourier domain,
        // we rotate x_msb[i] by angle:
        //   sum_j c_j * 2*pi * 2^j / 2^(n-i)
        // for each j where bit j of c is set.
        //
        // Combine all contributions into a single phase per qubit.
        for i in 0..n {
            let mut total_angle = 0.0_f64;
            for j in 0..n {
                let k_signed: isize = n as isize - i as isize - j as isize;
                if k_signed >= 1 && (self.constant >> j) & 1 == 1 {
                    let k = k_signed as usize;
                    total_angle += 2.0 * PI / (1u64 << k) as f64;
                }
            }
            if total_angle.abs() > 1e-15 {
                gates.push(ArithmeticGate::Phase(x_msb[i], total_angle));
            }
        }

        // Step 3: Inverse QFT on x register (no swap).
        gates.extend(inverse_qft_no_swap(&x_msb, &self.config));

        gates
    }

    /// Generate the inverse (subtraction) gate sequence: |x> -> |x - c mod 2^n>.
    pub fn subtract_from_register(&self, x_qubits: &[usize]) -> Vec<ArithmeticGate> {
        let forward = self.add_to_register(x_qubits);
        forward.iter().rev().map(|g| g.inverse()).collect()
    }
}

// ============================================================
// QUANTUM MULTIPLIER
// ============================================================

/// Schoolbook quantum multiplier via repeated addition.
///
/// Computes |a>|0...0_result> -> |a>|a * b_classical mod 2^(2n)> where
/// `b_classical` is a known classical value.
///
/// Uses shift-and-add: for each bit of b that is set, add a shifted by
/// that bit position into the result register using the Draper adder.
pub struct QuantumMultiplier {
    config: ArithmeticConfig,
}

impl QuantumMultiplier {
    /// Create a multiplier with the given configuration.
    pub fn new(config: ArithmeticConfig) -> Self {
        QuantumMultiplier { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        QuantumMultiplier {
            config: ArithmeticConfig::default(),
        }
    }

    /// Generate gate sequence for multiplication by a classical value.
    ///
    /// `a_qubits`: n-qubit multiplicand register (LSB first).
    /// `result_qubits`: 2n-qubit result register, initialized to |0>.
    /// `b_value`: classical multiplier value.
    ///
    /// Result: |a> unchanged, |result> = |a * b_value mod 2^(2n)>.
    pub fn multiply_by_classical(
        &self,
        a_qubits: &[usize],
        result_qubits: &[usize],
        b_value: u64,
    ) -> Result<Vec<ArithmeticGate>, ArithmeticError> {
        let n = a_qubits.len();
        if result_qubits.len() < 2 * n {
            return Err(ArithmeticError::OverflowRisk {
                register_bits: result_qubits.len(),
                max_value: ((1u64 << n) - 1) * b_value,
            });
        }

        let mut gates = Vec::new();

        // For each bit position j of b_value, if bit j is set,
        // add the a register into result[j..j+n] using Draper addition.
        for j in 0..n {
            if (b_value >> j) & 1 == 1 {
                let result_sub: Vec<usize> = (0..n)
                    .filter_map(|i| {
                        if i + j < result_qubits.len() {
                            Some(result_qubits[i + j])
                        } else {
                            None
                        }
                    })
                    .collect();

                if result_sub.len() == n {
                    let adder = DraperAdder::new(self.config.clone());
                    let add_gates = adder.add(a_qubits, &result_sub)?;
                    gates.extend(add_gates);
                }
            }
        }

        Ok(gates)
    }
}

// ============================================================
// QUANTUM COMPARATOR
// ============================================================

/// Quantum comparator: |x>|0> -> |x>|x >= c> for classical value c.
///
/// Uses subtraction: compute x - c, check the sign bit (MSB), and
/// then un-compute the subtraction leaving only the comparison result.
pub struct QuantumComparator {
    config: ArithmeticConfig,
}

impl QuantumComparator {
    /// Create a comparator with the given configuration.
    pub fn new(config: ArithmeticConfig) -> Self {
        QuantumComparator { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        QuantumComparator {
            config: ArithmeticConfig::default(),
        }
    }

    /// Compare quantum register against a classical value.
    ///
    /// `x_qubits`: quantum register (LSB first).
    /// `classical_value`: the classical value to compare against.
    /// `result_qubit`: single qubit initialized to |0>, set to |1> if x >= classical_value.
    ///
    /// Strategy: compute |x - c> in place, read sign bit, un-compute.
    pub fn compare_with_classical(
        &self,
        x_qubits: &[usize],
        classical_value: u64,
        result_qubit: usize,
    ) -> Vec<ArithmeticGate> {
        let n = x_qubits.len();
        let mut gates = Vec::new();

        // Subtract c from x (in place): |x> -> |x - c mod 2^n>.
        let subtractor = ClassicalQuantumAdder::new(classical_value, self.config.clone());
        gates.extend(subtractor.subtract_from_register(x_qubits));

        // If x >= c, then x - c is in [0, 2^(n-1)), so the MSB is 0.
        // If x < c, then x - c wraps around and MSB is 1.
        // Copy the MSB into result, then flip to get "x >= c".
        let msb = x_qubits[n - 1];
        gates.push(ArithmeticGate::CNOT(msb, result_qubit));
        gates.push(ArithmeticGate::X(result_qubit));

        // Un-compute the subtraction: add c back.
        gates.extend(subtractor.add_to_register(x_qubits));

        gates
    }
}

// ============================================================
// MODULAR ARITHMETIC
// ============================================================

/// Modular arithmetic operations for Shor's algorithm.
///
/// Provides modular addition (x + a) mod N and modular multiplication
/// a * x mod N using QFT-based addition as the underlying building block.
pub struct ModularArithmetic {
    /// The modulus N.
    pub modulus: u64,
    config: ArithmeticConfig,
}

impl ModularArithmetic {
    /// Create a new modular arithmetic engine for modulus N.
    pub fn new(modulus: u64, config: ArithmeticConfig) -> Result<Self, ArithmeticError> {
        if modulus < 2 {
            return Err(ArithmeticError::InvalidModulus {
                modulus,
                reason: "modulus must be >= 2",
            });
        }
        Ok(ModularArithmetic { modulus, config })
    }

    /// Create with default configuration.
    pub fn with_modulus(modulus: u64) -> Result<Self, ArithmeticError> {
        Self::new(modulus, ArithmeticConfig::default())
    }

    /// Minimum number of qubits needed to hold values up to modulus - 1.
    pub fn register_bits(&self) -> usize {
        let mut bits = 0usize;
        let mut val = self.modulus;
        while val > 0 {
            bits += 1;
            val >>= 1;
        }
        bits
    }

    /// Modular addition: |x> -> |(x + a) mod N> where a is classical.
    ///
    /// Algorithm (Beauregard-style):
    /// 1. Add a to x: |x> -> |x + a>
    /// 2. Subtract N: |x + a> -> |x + a - N>
    /// 3. If x + a - N < 0 (MSB set), add N back.
    /// 4. Un-compute the comparison ancilla.
    ///
    /// Requires one ancilla qubit for the comparison flag.
    pub fn modular_add_classical(
        &self,
        x_qubits: &[usize],
        classical_a: u64,
        ancilla: usize,
    ) -> Vec<ArithmeticGate> {
        let n = x_qubits.len();
        let mut gates = Vec::new();
        let a_mod = classical_a % self.modulus;

        // Step 1: Add a to x.
        let adder_a = ClassicalQuantumAdder::new(a_mod, self.config.clone());
        gates.extend(adder_a.add_to_register(x_qubits));

        // Step 2: Subtract N.
        let sub_n = ClassicalQuantumAdder::new(self.modulus, self.config.clone());
        gates.extend(sub_n.subtract_from_register(x_qubits));

        // Step 3: Check MSB. If x + a - N < 0 (wrapped), MSB = 1.
        let msb = x_qubits[n - 1];
        gates.push(ArithmeticGate::CNOT(msb, ancilla));

        // Step 4: If ancilla = 1, add N back (controlled addition).
        // We do QFT on x, then controlled-phase from ancilla for each set bit of N.
        let x_msb: Vec<usize> = x_qubits.iter().copied().rev().collect();
        gates.extend(qft_no_swap(&x_msb, &self.config));
        for i in 0..n {
            let mut total_angle = 0.0_f64;
            for j in 0..n {
                let k_signed: isize = n as isize - i as isize - j as isize;
                if k_signed >= 1 && (self.modulus >> j) & 1 == 1 {
                    let k = k_signed as usize;
                    total_angle += 2.0 * PI / (1u64 << k) as f64;
                }
            }
            if total_angle.abs() > 1e-15 {
                gates.push(ArithmeticGate::CPhase(ancilla, x_msb[i], total_angle));
            }
        }
        gates.extend(inverse_qft_no_swap(&x_msb, &self.config));

        // Step 5: Un-compute ancilla.
        // Subtract a to get x back, check MSB, flip, add a again.
        gates.extend(adder_a.subtract_from_register(x_qubits));
        gates.push(ArithmeticGate::X(ancilla));
        gates.push(ArithmeticGate::CNOT(msb, ancilla));
        gates.push(ArithmeticGate::X(ancilla));
        gates.extend(adder_a.add_to_register(x_qubits));

        gates
    }

    /// Modular multiplication: |x> -> |a * x mod N> where a is classical.
    ///
    /// Beauregard-style: for each bit x[j], conditionally add (a * 2^j mod N)
    /// to scratch with modular reduction. Then swap scratch and x.
    ///
    /// `x_qubits`: input/output register (LSB first).
    /// `scratch_qubits`: same size as x, initialized to |0>.
    /// `ancilla`: single qubit initialized to |0>.
    /// `classical_a`: the classical multiplier.
    pub fn modular_multiply_classical(
        &self,
        x_qubits: &[usize],
        scratch_qubits: &[usize],
        ancilla: usize,
        classical_a: u64,
    ) -> Result<Vec<ArithmeticGate>, ArithmeticError> {
        let n = x_qubits.len();
        if n != scratch_qubits.len() {
            return Err(ArithmeticError::RegisterSizeMismatch {
                a_size: n,
                b_size: scratch_qubits.len(),
            });
        }

        let a_mod = classical_a % self.modulus;
        let mut gates = Vec::new();

        // For each bit x[j], conditionally add (a * 2^j mod N) to scratch
        // with modular reduction.
        //
        // The controlled modular addition uses:
        // 1. Controlled add value to scratch (in QFT domain, controlled by x[j])
        // 2. Subtract N from scratch
        // 3. Check sign (MSB), copy to ancilla
        // 4. Controlled-on-ancilla add N back
        // 5. Un-compute ancilla
        for j in 0..n {
            let shifted_a = (a_mod * (1u64 << j)) % self.modulus;
            if shifted_a == 0 {
                continue;
            }
            let s_msb: Vec<usize> = scratch_qubits.iter().copied().rev().collect();

            // Step 1: Controlled add shifted_a to scratch (controlled by x[j]).
            gates.extend(qft_no_swap(&s_msb, &self.config));
            for i in 0..n {
                let mut total_angle = 0.0_f64;
                for k in 0..n {
                    let rot_signed: isize = n as isize - i as isize - k as isize;
                    if rot_signed >= 1 && (shifted_a >> k) & 1 == 1 {
                        let rot = rot_signed as usize;
                        total_angle += 2.0 * PI / (1u64 << rot) as f64;
                    }
                }
                if total_angle.abs() > 1e-15 {
                    gates.push(ArithmeticGate::CPhase(x_qubits[j], s_msb[i], total_angle));
                }
            }
            // Step 2: Subtract N (unconditional, in QFT domain still).
            for i in 0..n {
                let mut total_angle = 0.0_f64;
                for k in 0..n {
                    let rot_signed: isize = n as isize - i as isize - k as isize;
                    if rot_signed >= 1 && (self.modulus >> k) & 1 == 1 {
                        let rot = rot_signed as usize;
                        total_angle -= 2.0 * PI / (1u64 << rot) as f64;
                    }
                }
                if total_angle.abs() > 1e-15 {
                    gates.push(ArithmeticGate::Phase(s_msb[i], total_angle));
                }
            }
            gates.extend(inverse_qft_no_swap(&s_msb, &self.config));

            // Step 3: Check sign (MSB of scratch). If negative (wrapped), MSB=1.
            let scratch_msb = scratch_qubits[n - 1];
            gates.push(ArithmeticGate::CNOT(scratch_msb, ancilla));

            // Step 4: Controlled add N back (controlled by ancilla).
            gates.extend(qft_no_swap(&s_msb, &self.config));
            for i in 0..n {
                let mut total_angle = 0.0_f64;
                for k in 0..n {
                    let rot_signed: isize = n as isize - i as isize - k as isize;
                    if rot_signed >= 1 && (self.modulus >> k) & 1 == 1 {
                        let rot = rot_signed as usize;
                        total_angle += 2.0 * PI / (1u64 << rot) as f64;
                    }
                }
                if total_angle.abs() > 1e-15 {
                    gates.push(ArithmeticGate::CPhase(ancilla, s_msb[i], total_angle));
                }
            }
            gates.extend(inverse_qft_no_swap(&s_msb, &self.config));

            // Step 5: Un-compute ancilla.
            // Subtract shifted_a from scratch, check MSB (inverted), restore.
            let sub_val = ClassicalQuantumAdder::new(shifted_a, self.config.clone());
            gates.extend(sub_val.subtract_from_register(scratch_qubits));
            gates.push(ArithmeticGate::X(ancilla));
            gates.push(ArithmeticGate::CNOT(scratch_msb, ancilla));
            gates.push(ArithmeticGate::X(ancilla));
            gates.extend(sub_val.add_to_register(scratch_qubits));
        }

        // Swap scratch and x to move the result into x.
        for i in 0..n {
            gates.push(ArithmeticGate::CNOT(x_qubits[i], scratch_qubits[i]));
            gates.push(ArithmeticGate::CNOT(scratch_qubits[i], x_qubits[i]));
            gates.push(ArithmeticGate::CNOT(x_qubits[i], scratch_qubits[i]));
        }

        Ok(gates)
    }
}

// ============================================================
// INLINE SIMULATION HELPERS
// ============================================================

/// Prepare a quantum state with a computational basis state encoding the
/// given integer in the specified qubit range (LSB first).
fn encode_integer(state: &mut QuantumState, value: u64, qubits: &[usize]) {
    for (i, &q) in qubits.iter().enumerate() {
        if (value >> i) & 1 == 1 {
            GateOperations::x(state, q);
        }
    }
}

/// Measure a register of qubits and return the integer value (LSB first).
/// Uses the maximum-probability basis state (deterministic readout).
fn read_register(state: &QuantumState, qubits: &[usize]) -> u64 {
    let probs = state.probabilities();
    let max_idx = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut value = 0u64;
    for (i, &q) in qubits.iter().enumerate() {
        if (max_idx >> q) & 1 == 1 {
            value |= 1 << i;
        }
    }
    value
}

/// Apply a sequence of arithmetic gates to a quantum state.
fn apply_gates(state: &mut QuantumState, gates: &[ArithmeticGate]) {
    for gate in gates {
        gate.apply(state);
    }
}

/// Apply the inverse of a gate sequence (reversed order, each gate inverted).
fn apply_inverse_gates(state: &mut QuantumState, gates: &[ArithmeticGate]) {
    for gate in gates.iter().rev() {
        gate.inverse().apply(state);
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::C64;

    // ----------------------------------------------------------
    // Test 1: Config builder defaults and custom
    // ----------------------------------------------------------

    #[test]
    fn config_builder() {
        let default = ArithmeticConfig::default();
        assert_eq!(default.adder_type, AdderType::Auto);
        assert!(!default.use_approximation);
        assert_eq!(default.rotation_precision, 10);

        let custom = ArithmeticConfig::new()
            .with_adder_type(AdderType::Draper)
            .with_approximation(true)
            .with_rotation_precision(5);
        assert_eq!(custom.adder_type, AdderType::Draper);
        assert!(custom.use_approximation);
        assert_eq!(custom.rotation_precision, 5);

        let auto = ArithmeticConfig::default();
        assert_eq!(auto.resolve_adder(4), AdderType::Cuccaro);
        assert_eq!(auto.resolve_adder(12), AdderType::Draper);
    }

    // ----------------------------------------------------------
    // Test 2: Draper add small (2 + 3 = 5)
    // ----------------------------------------------------------

    #[test]
    fn draper_add_small() {
        let n = 4;
        let a_qubits: Vec<usize> = (0..n).collect();
        let b_qubits: Vec<usize> = (n..2 * n).collect();

        let mut state = QuantumState::new(2 * n);
        encode_integer(&mut state, 2, &a_qubits);
        encode_integer(&mut state, 3, &b_qubits);

        let adder = DraperAdder::default_config();
        let gates = adder.add(&a_qubits, &b_qubits).unwrap();
        apply_gates(&mut state, &gates);

        let a_val = read_register(&state, &a_qubits);
        let b_val = read_register(&state, &b_qubits);
        assert_eq!(a_val, 2, "a register should be unchanged");
        assert_eq!(b_val, 5, "b register should hold 2 + 3 = 5");
    }

    // ----------------------------------------------------------
    // Test 3: Draper add overflow (wrap-around)
    // ----------------------------------------------------------

    #[test]
    fn draper_add_overflow() {
        let n = 3;
        let a_qubits: Vec<usize> = (0..n).collect();
        let b_qubits: Vec<usize> = (n..2 * n).collect();

        let mut state = QuantumState::new(2 * n);
        encode_integer(&mut state, 5, &a_qubits);
        encode_integer(&mut state, 6, &b_qubits);

        let adder = DraperAdder::default_config();
        let gates = adder.add(&a_qubits, &b_qubits).unwrap();
        apply_gates(&mut state, &gates);

        let b_val = read_register(&state, &b_qubits);
        assert_eq!(b_val, (5 + 6) % 8, "should wrap around: (5+6) mod 8 = 3");
    }

    // ----------------------------------------------------------
    // Test 4: Cuccaro add small (2 + 3 = 5)
    // ----------------------------------------------------------

    #[test]
    fn cuccaro_add_small() {
        let n = 4;
        let ancilla = 2 * n;
        let a_qubits: Vec<usize> = (0..n).collect();
        let b_qubits: Vec<usize> = (n..2 * n).collect();

        let mut state = QuantumState::new(2 * n + 1);
        encode_integer(&mut state, 2, &a_qubits);
        encode_integer(&mut state, 3, &b_qubits);

        let adder = CuccaroAdder::with_ancilla(ancilla);
        let gates = adder.add(&a_qubits, &b_qubits).unwrap();
        apply_gates(&mut state, &gates);

        let b_val = read_register(&state, &b_qubits);
        assert_eq!(b_val, 5, "Cuccaro: b should hold 2 + 3 = 5");
    }

    // ----------------------------------------------------------
    // Test 5: Classical-quantum add (constant 5 + |3> = |8>)
    // ----------------------------------------------------------

    #[test]
    fn classical_quantum_add() {
        let n = 4;
        let x_qubits: Vec<usize> = (0..n).collect();

        let mut state = QuantumState::new(n);
        encode_integer(&mut state, 3, &x_qubits);

        let adder = ClassicalQuantumAdder::with_constant(5);
        let gates = adder.add_to_register(&x_qubits);
        apply_gates(&mut state, &gates);

        let result = read_register(&state, &x_qubits);
        assert_eq!(result, 8, "3 + 5 = 8");
    }

    // ----------------------------------------------------------
    // Test 6: Draper vs Cuccaro produce same result for all 3-bit inputs
    // ----------------------------------------------------------

    #[test]
    fn draper_vs_cuccaro() {
        let n = 3;
        let a_qubits: Vec<usize> = (0..n).collect();
        let b_qubits: Vec<usize> = (n..2 * n).collect();
        let ancilla = 2 * n;

        for a_val in 0..8u64 {
            for b_val in 0..8u64 {
                let expected = (a_val + b_val) % 8;

                // Draper.
                let mut state_d = QuantumState::new(2 * n);
                encode_integer(&mut state_d, a_val, &a_qubits);
                encode_integer(&mut state_d, b_val, &b_qubits);
                let draper = DraperAdder::default_config();
                let gates_d = draper.add(&a_qubits, &b_qubits).unwrap();
                apply_gates(&mut state_d, &gates_d);
                let result_d = read_register(&state_d, &b_qubits);

                // Cuccaro.
                let mut state_c = QuantumState::new(2 * n + 1);
                encode_integer(&mut state_c, a_val, &a_qubits);
                encode_integer(&mut state_c, b_val, &b_qubits);
                let cuccaro = CuccaroAdder::with_ancilla(ancilla);
                let gates_c = cuccaro.add(&a_qubits, &b_qubits).unwrap();
                apply_gates(&mut state_c, &gates_c);
                let result_c = read_register(&state_c, &b_qubits);

                assert_eq!(
                    result_d, expected,
                    "Draper: {} + {} should be {} (got {})",
                    a_val, b_val, expected, result_d
                );
                assert_eq!(
                    result_c, expected,
                    "Cuccaro: {} + {} should be {} (got {})",
                    a_val, b_val, expected, result_c
                );
            }
        }
    }

    // ----------------------------------------------------------
    // Test 7: Quantum multiplier (3 * 4 = 12)
    // ----------------------------------------------------------

    #[test]
    fn quantum_multiplier() {
        let n = 3;
        let a_qubits: Vec<usize> = (0..n).collect();
        let result_qubits: Vec<usize> = (n..(n + 2 * n)).collect();

        let mut state = QuantumState::new(n + 2 * n);
        encode_integer(&mut state, 3, &a_qubits);

        let multiplier = QuantumMultiplier::default_config();
        let gates = multiplier
            .multiply_by_classical(&a_qubits, &result_qubits, 4)
            .unwrap();
        apply_gates(&mut state, &gates);

        let result = read_register(&state, &result_qubits);
        assert_eq!(result, 12, "3 * 4 = 12");
    }

    // ----------------------------------------------------------
    // Test 8: Quantum comparator (5 >= 3 and 2 < 3)
    // ----------------------------------------------------------

    #[test]
    fn quantum_comparator() {
        let n = 4;
        let x_qubits: Vec<usize> = (0..n).collect();
        let result_qubit = n;
        let comparator = QuantumComparator::default_config();

        // Test 5 >= 3: should set result to 1.
        {
            let mut state = QuantumState::new(n + 1);
            encode_integer(&mut state, 5, &x_qubits);
            let gates = comparator.compare_with_classical(&x_qubits, 3, result_qubit);
            apply_gates(&mut state, &gates);

            let probs = state.probabilities();
            let max_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let result_bit = (max_idx >> result_qubit) & 1;
            assert_eq!(result_bit, 1, "5 >= 3 should give result = 1");
        }

        // Test 2 < 3: should set result to 0.
        {
            let mut state = QuantumState::new(n + 1);
            encode_integer(&mut state, 2, &x_qubits);
            let gates = comparator.compare_with_classical(&x_qubits, 3, result_qubit);
            apply_gates(&mut state, &gates);

            let probs = state.probabilities();
            let max_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let result_bit = (max_idx >> result_qubit) & 1;
            assert_eq!(result_bit, 0, "2 < 3 should give result = 0");
        }
    }

    // ----------------------------------------------------------
    // Test 9: Modular addition ((5 + 4) mod 7 = 2)
    // ----------------------------------------------------------

    #[test]
    fn modular_addition() {
        let modulus = 7u64;
        let n = 4;
        let x_qubits: Vec<usize> = (0..n).collect();
        let ancilla = n;

        let mut state = QuantumState::new(n + 1);
        encode_integer(&mut state, 5, &x_qubits);

        let mod_arith = ModularArithmetic::with_modulus(modulus).unwrap();
        let gates = mod_arith.modular_add_classical(&x_qubits, 4, ancilla);
        apply_gates(&mut state, &gates);

        let result = read_register(&state, &x_qubits);
        assert_eq!(result, (5 + 4) % 7, "(5 + 4) mod 7 = 2, got {}", result);
    }

    // ----------------------------------------------------------
    // Test 10: Modular multiplication ((3 * 5) mod 7 = 1)
    // ----------------------------------------------------------

    #[test]
    fn modular_multiplication() {
        let modulus = 7u64;
        let n = 4;
        let x_qubits: Vec<usize> = (0..n).collect();
        let scratch_qubits: Vec<usize> = (n..(2 * n)).collect();
        let ancilla = 2 * n;

        let mut state = QuantumState::new(2 * n + 1);
        encode_integer(&mut state, 5, &x_qubits);

        let mod_arith = ModularArithmetic::with_modulus(modulus).unwrap();
        let gates = mod_arith
            .modular_multiply_classical(&x_qubits, &scratch_qubits, ancilla, 3)
            .unwrap();
        apply_gates(&mut state, &gates);

        let result = read_register(&state, &x_qubits);
        assert_eq!(result, (3 * 5) % 7, "(3 * 5) mod 7 = 1, got {}", result);
    }

    // ----------------------------------------------------------
    // Test 11: Gate count comparison (Draper O(n^2) vs Cuccaro O(n))
    // ----------------------------------------------------------

    #[test]
    fn gate_count_comparison() {
        let draper = DraperAdder::default_config();
        let sizes = [4, 8, 16];
        let mut draper_counts = Vec::new();
        let mut cuccaro_counts = Vec::new();

        for &n in &sizes {
            let a: Vec<usize> = (0..n).collect();
            let b: Vec<usize> = (n..2 * n).collect();
            let ancilla = 2 * n;

            let dg = draper.add(&a, &b).unwrap();
            draper_counts.push(dg.len());

            let cuccaro = CuccaroAdder::with_ancilla(ancilla);
            let cg = cuccaro.add(&a, &b).unwrap();
            cuccaro_counts.push(cg.len());
        }

        let draper_ratio = draper_counts[2] as f64 / draper_counts[0] as f64;
        let cuccaro_ratio = cuccaro_counts[2] as f64 / cuccaro_counts[0] as f64;

        assert!(
            draper_ratio > cuccaro_ratio,
            "Draper should grow faster than Cuccaro: draper_ratio={:.1}, cuccaro_ratio={:.1}",
            draper_ratio,
            cuccaro_ratio
        );

        assert!(
            draper_ratio > 6.0,
            "Draper should show roughly quadratic growth, ratio={:.1}",
            draper_ratio
        );

        assert!(
            cuccaro_ratio < 8.0,
            "Cuccaro should show roughly linear growth, ratio={:.1}",
            cuccaro_ratio
        );
    }

    // ----------------------------------------------------------
    // Test 12: Approximate Draper (truncated rotations)
    // ----------------------------------------------------------

    #[test]
    fn approximate_draper() {
        let n = 8;
        let a: Vec<usize> = (0..n).collect();
        let b: Vec<usize> = (n..2 * n).collect();

        let full_config = ArithmeticConfig::new()
            .with_adder_type(AdderType::Draper)
            .with_approximation(false);
        let full_draper = DraperAdder::new(full_config);
        let full_gates = full_draper.add(&a, &b).unwrap();

        let approx_config = ArithmeticConfig::new()
            .with_adder_type(AdderType::Draper)
            .with_approximation(true)
            .with_rotation_precision(4);
        let approx_draper = DraperAdder::new(approx_config);
        let approx_gates = approx_draper.add(&a, &b).unwrap();

        assert!(
            approx_gates.len() < full_gates.len(),
            "Approximate Draper should have fewer gates: {} vs {}",
            approx_gates.len(),
            full_gates.len()
        );

        // Verify the approximate version still produces correct results
        // for small values.
        let mut state = QuantumState::new(2 * n);
        encode_integer(&mut state, 3, &a);
        encode_integer(&mut state, 5, &b);
        apply_gates(&mut state, &approx_gates);
        let result = read_register(&state, &b);
        assert_eq!(result, 8, "Approximate Draper: 3 + 5 = 8 (got {})", result);
    }

    // ----------------------------------------------------------
    // Test 13: Reversibility (apply + inverse = identity)
    // ----------------------------------------------------------

    #[test]
    fn reversibility() {
        let n = 3;
        let a_qubits: Vec<usize> = (0..n).collect();
        let b_qubits: Vec<usize> = (n..2 * n).collect();

        // Draper reversibility.
        for a_val in 0..8u64 {
            for b_val in 0..8u64 {
                let mut state = QuantumState::new(2 * n);
                encode_integer(&mut state, a_val, &a_qubits);
                encode_integer(&mut state, b_val, &b_qubits);
                let initial_amps: Vec<C64> = state.amplitudes_ref().to_vec();

                let draper = DraperAdder::default_config();
                let gates = draper.add(&a_qubits, &b_qubits).unwrap();
                apply_gates(&mut state, &gates);
                apply_inverse_gates(&mut state, &gates);

                let final_amps = state.amplitudes_ref();
                for (i, (init, fin)) in initial_amps.iter().zip(final_amps.iter()).enumerate() {
                    let diff = (init - fin).norm();
                    assert!(
                        diff < 1e-10,
                        "Draper reversibility failed at index {} for a={}, b={}: diff={}",
                        i,
                        a_val,
                        b_val,
                        diff
                    );
                }
            }
        }

        // Classical-Quantum add reversibility.
        for x_val in 0..8u64 {
            let x_qubits: Vec<usize> = (0..n).collect();
            let mut state = QuantumState::new(n);
            encode_integer(&mut state, x_val, &x_qubits);
            let initial_amps: Vec<C64> = state.amplitudes_ref().to_vec();

            let cq_adder = ClassicalQuantumAdder::with_constant(5);
            let gates = cq_adder.add_to_register(&x_qubits);
            apply_gates(&mut state, &gates);
            apply_inverse_gates(&mut state, &gates);

            let final_amps = state.amplitudes_ref();
            for (i, (init, fin)) in initial_amps.iter().zip(final_amps.iter()).enumerate() {
                let diff = (init - fin).norm();
                assert!(
                    diff < 1e-10,
                    "CQ reversibility failed at index {} for x={}: diff={}",
                    i,
                    x_val,
                    diff
                );
            }
        }
    }

    // ----------------------------------------------------------
    // Additional: Error handling tests
    // ----------------------------------------------------------

    #[test]
    fn register_size_mismatch_error() {
        let draper = DraperAdder::default_config();
        let result = draper.add(&[0, 1, 2], &[3, 4]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArithmeticError::RegisterSizeMismatch { a_size, b_size } => {
                assert_eq!(a_size, 3);
                assert_eq!(b_size, 2);
            }
            other => panic!("Expected RegisterSizeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn invalid_modulus_error() {
        let result = ModularArithmetic::with_modulus(0);
        assert!(result.is_err());
        let result = ModularArithmetic::with_modulus(1);
        assert!(result.is_err());
        let result = ModularArithmetic::with_modulus(2);
        assert!(result.is_ok());
    }

    #[test]
    fn gate_inverse_roundtrip() {
        let gates = vec![
            ArithmeticGate::H(0),
            ArithmeticGate::X(0),
            ArithmeticGate::CNOT(0, 1),
            ArithmeticGate::Toffoli(0, 1, 2),
            ArithmeticGate::CPhase(0, 1, 1.234),
            ArithmeticGate::Phase(0, 2.345),
        ];

        for gate in &gates {
            let inv = gate.inverse();
            let inv_inv = inv.inverse();
            match gate {
                ArithmeticGate::CPhase(_, _, angle) => {
                    if let ArithmeticGate::CPhase(_, _, inv_angle) = inv {
                        assert!((inv_angle + angle).abs() < 1e-12);
                    }
                    assert_eq!(*gate, inv_inv);
                }
                ArithmeticGate::Phase(_, angle) => {
                    if let ArithmeticGate::Phase(_, inv_angle) = inv {
                        assert!((inv_angle + angle).abs() < 1e-12);
                    }
                    assert_eq!(*gate, inv_inv);
                }
                _ => {
                    assert_eq!(*gate, inv);
                }
            }
        }
    }

    #[test]
    fn classical_quantum_subtraction() {
        let n = 4;
        let x_qubits: Vec<usize> = (0..n).collect();

        let mut state = QuantumState::new(n);
        encode_integer(&mut state, 8, &x_qubits);

        let adder = ClassicalQuantumAdder::with_constant(5);
        let gates = adder.subtract_from_register(&x_qubits);
        apply_gates(&mut state, &gates);

        let result = read_register(&state, &x_qubits);
        assert_eq!(result, 3, "8 - 5 = 3, got {}", result);
    }
}
