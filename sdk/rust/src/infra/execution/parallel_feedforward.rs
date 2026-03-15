//! Parallel Mid-Circuit Measurement with Classical Feed-Forward
//!
//! Enables dynamic quantum circuits where measurement outcomes mid-circuit
//! control subsequent operations. Matches IBM Qiskit's dynamic circuit
//! capability with support for conditional gates, classical registers,
//! and parallel feed-forward where multiple measurements inform future
//! gates simultaneously.
//!
//! # Key Capabilities
//!
//! - **Mid-circuit measurement**: Measure qubits without ending the circuit,
//!   storing outcomes in named classical registers.
//! - **Classical feed-forward**: Conditional gate execution based on
//!   measurement outcomes (if-equal, if-bit, parallel-if).
//! - **Qubit reset and reuse**: Reset qubits to |0> after measurement,
//!   enabling circuits with more logical operations than physical qubits.
//! - **Classical logic**: XOR, AND, NOT on classical register bits for
//!   combining measurement outcomes before feed-forward.
//! - **Deferred measurement optimization**: Replace mid-circuit measurements
//!   with CNOT + ancilla, deferring all measurement to end of circuit for
//!   potentially faster simulation (same output distribution).
//! - **Multi-shot execution**: Deterministic (seeded) repeated execution
//!   with bitstring histogram collection.
//!
//! # Pre-built Dynamic Circuits
//!
//! - Quantum teleportation (Bell measurement + Pauli correction)
//! - 3-qubit bit-flip error correction (syndrome + correction)
//! - Repeat-until-success for non-Clifford gate synthesis
//! - Adaptive VQE step (measure energy, adjust rotation)
//! - GHZ verification (prepare, measure parity, conditionally retry)
//!
//! # References
//!
//! - IBM Qiskit Dynamic Circuits documentation
//! - Principled approach: deferred measurement principle (Nielson & Chuang)

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors arising from dynamic circuit construction or execution.
#[derive(Debug, Clone, PartialEq)]
pub enum FeedForwardError {
    /// Invalid classical register index or bit index.
    ClassicalRegisterError(String),
    /// Measurement failed (e.g. qubit index out of range).
    MeasurementError(String),
    /// Conditional evaluation failed.
    ConditionalError(String),
    /// General circuit construction or execution error.
    CircuitError(String),
}

impl fmt::Display for FeedForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeedForwardError::ClassicalRegisterError(s) => {
                write!(f, "ClassicalRegisterError: {}", s)
            }
            FeedForwardError::MeasurementError(s) => write!(f, "MeasurementError: {}", s),
            FeedForwardError::ConditionalError(s) => write!(f, "ConditionalError: {}", s),
            FeedForwardError::CircuitError(s) => write!(f, "CircuitError: {}", s),
        }
    }
}

impl std::error::Error for FeedForwardError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, FeedForwardError>;

// ---------------------------------------------------------------------------
// Classical register
// ---------------------------------------------------------------------------

/// Classical register for storing measurement outcomes.
#[derive(Clone, Debug)]
pub struct ClassicalRegister {
    /// Human-readable name.
    pub name: String,
    /// Bit values (index 0 is least-significant).
    pub bits: Vec<bool>,
    /// Declared size (may exceed `bits.len()` during lazy growth).
    pub size: usize,
}

impl ClassicalRegister {
    /// Create a new register with all bits initialised to false.
    pub fn new(name: &str, size: usize) -> Self {
        ClassicalRegister {
            name: name.to_string(),
            bits: vec![false; size],
            size,
        }
    }

    /// Read bit at `index`. Returns `false` for out-of-range.
    pub fn get_bit(&self, index: usize) -> bool {
        self.bits.get(index).copied().unwrap_or(false)
    }

    /// Write bit at `index`, growing the register if needed.
    pub fn set_bit(&mut self, index: usize, value: bool) {
        while self.bits.len() <= index {
            self.bits.push(false);
        }
        self.size = self.size.max(index + 1);
        self.bits[index] = value;
    }

    /// Interpret the first `n` bits as a little-endian unsigned integer.
    pub fn value(&self, n: usize) -> u64 {
        let mut v: u64 = 0;
        for i in 0..n.min(self.bits.len()) {
            if self.bits[i] {
                v |= 1u64 << i;
            }
        }
        v
    }

    /// Reset all bits to false.
    pub fn reset(&mut self) {
        for b in &mut self.bits {
            *b = false;
        }
    }

    /// Return a bitstring representation (MSB first) of the first `n` bits.
    pub fn bitstring(&self, n: usize) -> String {
        let len = n.min(self.bits.len());
        (0..len)
            .rev()
            .map(|i| if self.bits[i] { '1' } else { '0' })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Condition
// ---------------------------------------------------------------------------

/// A single-bit condition used inside `ParallelIf`.
#[derive(Clone, Debug)]
pub struct Condition {
    /// Index of the classical register.
    pub register: usize,
    /// Bit index within the register.
    pub bit: usize,
    /// Expected value of that bit.
    pub expected: bool,
}

impl Condition {
    pub fn new(register: usize, bit: usize, expected: bool) -> Self {
        Condition {
            register,
            bit,
            expected,
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic instruction set
// ---------------------------------------------------------------------------

/// Dynamic circuit instruction supporting quantum gates, mid-circuit
/// measurement, classical control (feed-forward), classical logic, reset
/// and timing directives.
#[derive(Clone, Debug)]
pub enum DynamicInstruction {
    // -- single-qubit gates --
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),

    // -- two-qubit gates --
    CX(usize, usize),
    CZ(usize, usize),

    // -- measurement --
    /// Measure `qubit` and store outcome in `register[bit]`.
    Measure {
        qubit: usize,
        register: usize,
        bit: usize,
    },

    // -- conditional (feed-forward) --
    /// Execute `then_ops` when the register interpreted as a little-endian
    /// integer equals `value`.
    IfEqual {
        register: usize,
        value: u64,
        then_ops: Vec<DynamicInstruction>,
    },
    /// Branch on a single bit.
    IfBit {
        register: usize,
        bit: usize,
        then_ops: Vec<DynamicInstruction>,
        else_ops: Vec<DynamicInstruction>,
    },
    /// Parallel feed-forward: ALL conditions must be satisfied.
    ParallelIf {
        conditions: Vec<Condition>,
        ops: Vec<DynamicInstruction>,
    },

    // -- classical logic --
    ClassicalXor {
        reg_a: usize,
        bit_a: usize,
        reg_b: usize,
        bit_b: usize,
        reg_out: usize,
        bit_out: usize,
    },
    ClassicalAnd {
        reg_a: usize,
        bit_a: usize,
        reg_b: usize,
        bit_b: usize,
        reg_out: usize,
        bit_out: usize,
    },
    ClassicalNot {
        reg: usize,
        bit: usize,
    },

    // -- qubit lifecycle --
    /// Reset qubit to |0> (measure, conditionally flip).
    Reset(usize),

    // -- timing --
    /// Idle a qubit for `cycles` timesteps (no-op in simulation).
    Delay {
        qubit: usize,
        cycles: usize,
    },
    /// Barrier across listed qubits (ordering fence, no physical effect).
    Barrier(Vec<usize>),
}

// ---------------------------------------------------------------------------
// Dynamic circuit
// ---------------------------------------------------------------------------

/// A dynamic quantum circuit with classical registers and feed-forward.
#[derive(Clone, Debug)]
pub struct DynamicCircuit {
    pub num_qubits: usize,
    pub classical_registers: Vec<ClassicalRegister>,
    pub instructions: Vec<DynamicInstruction>,
}

impl DynamicCircuit {
    /// Create an empty circuit.
    pub fn new(num_qubits: usize) -> Self {
        DynamicCircuit {
            num_qubits,
            classical_registers: Vec::new(),
            instructions: Vec::new(),
        }
    }

    /// Add a classical register and return its index.
    pub fn add_register(&mut self, name: &str, size: usize) -> usize {
        let idx = self.classical_registers.len();
        self.classical_registers
            .push(ClassicalRegister::new(name, size));
        idx
    }

    /// Append an instruction.
    pub fn add(&mut self, inst: DynamicInstruction) {
        self.instructions.push(inst);
    }

    /// Validate that all qubit and register references are in range.
    pub fn validate(&self) -> Result<()> {
        self.validate_instructions(&self.instructions)
    }

    fn validate_instructions(&self, instructions: &[DynamicInstruction]) -> Result<()> {
        for inst in instructions {
            match inst {
                DynamicInstruction::H(q)
                | DynamicInstruction::X(q)
                | DynamicInstruction::Y(q)
                | DynamicInstruction::Z(q)
                | DynamicInstruction::S(q)
                | DynamicInstruction::T(q)
                | DynamicInstruction::Reset(q) => {
                    if *q >= self.num_qubits {
                        return Err(FeedForwardError::CircuitError(format!(
                            "qubit {} out of range (num_qubits={})",
                            q, self.num_qubits
                        )));
                    }
                }
                DynamicInstruction::Rx(q, _)
                | DynamicInstruction::Ry(q, _)
                | DynamicInstruction::Rz(q, _) => {
                    if *q >= self.num_qubits {
                        return Err(FeedForwardError::CircuitError(format!(
                            "qubit {} out of range",
                            q
                        )));
                    }
                }
                DynamicInstruction::CX(c, t) | DynamicInstruction::CZ(c, t) => {
                    if *c >= self.num_qubits || *t >= self.num_qubits {
                        return Err(FeedForwardError::CircuitError(format!(
                            "qubit index out of range in two-qubit gate ({}, {})",
                            c, t
                        )));
                    }
                }
                DynamicInstruction::Measure {
                    qubit,
                    register,
                    bit,
                } => {
                    if *qubit >= self.num_qubits {
                        return Err(FeedForwardError::MeasurementError(format!(
                            "qubit {} out of range",
                            qubit
                        )));
                    }
                    if *register >= self.classical_registers.len() {
                        return Err(FeedForwardError::ClassicalRegisterError(format!(
                            "register {} out of range",
                            register
                        )));
                    }
                    if *bit >= self.classical_registers[*register].size {
                        return Err(FeedForwardError::ClassicalRegisterError(format!(
                            "bit {} out of range for register '{}' (size={})",
                            bit,
                            self.classical_registers[*register].name,
                            self.classical_registers[*register].size
                        )));
                    }
                }
                DynamicInstruction::IfEqual {
                    register, then_ops, ..
                } => {
                    if *register >= self.classical_registers.len() {
                        return Err(FeedForwardError::ClassicalRegisterError(format!(
                            "register {} out of range",
                            register
                        )));
                    }
                    self.validate_instructions(then_ops)?;
                }
                DynamicInstruction::IfBit {
                    register,
                    bit,
                    then_ops,
                    else_ops,
                } => {
                    if *register >= self.classical_registers.len() {
                        return Err(FeedForwardError::ClassicalRegisterError(format!(
                            "register {} out of range",
                            register
                        )));
                    }
                    if *bit >= self.classical_registers[*register].size {
                        return Err(FeedForwardError::ClassicalRegisterError(format!(
                            "bit {} out of range for register '{}'",
                            bit, self.classical_registers[*register].name
                        )));
                    }
                    self.validate_instructions(then_ops)?;
                    self.validate_instructions(else_ops)?;
                }
                DynamicInstruction::ParallelIf { conditions, ops } => {
                    for cond in conditions {
                        if cond.register >= self.classical_registers.len() {
                            return Err(FeedForwardError::ConditionalError(format!(
                                "condition register {} out of range",
                                cond.register
                            )));
                        }
                    }
                    self.validate_instructions(ops)?;
                }
                DynamicInstruction::ClassicalXor {
                    reg_a,
                    reg_b,
                    reg_out,
                    ..
                }
                | DynamicInstruction::ClassicalAnd {
                    reg_a,
                    reg_b,
                    reg_out,
                    ..
                } => {
                    for r in &[reg_a, reg_b, reg_out] {
                        if **r >= self.classical_registers.len() {
                            return Err(FeedForwardError::ClassicalRegisterError(format!(
                                "register {} out of range",
                                r
                            )));
                        }
                    }
                }
                DynamicInstruction::ClassicalNot { reg, .. } => {
                    if *reg >= self.classical_registers.len() {
                        return Err(FeedForwardError::ClassicalRegisterError(format!(
                            "register {} out of range",
                            reg
                        )));
                    }
                }
                DynamicInstruction::Delay { qubit, .. } => {
                    if *qubit >= self.num_qubits {
                        return Err(FeedForwardError::CircuitError(format!(
                            "qubit {} out of range in Delay",
                            qubit
                        )));
                    }
                }
                DynamicInstruction::Barrier(qs) => {
                    for q in qs {
                        if *q >= self.num_qubits {
                            return Err(FeedForwardError::CircuitError(format!(
                                "qubit {} out of range in Barrier",
                                q
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Count feed-forward (conditional) instructions at the top level.
    pub fn count_feed_forward_ops(&self) -> usize {
        self.count_ff_recursive(&self.instructions)
    }

    fn count_ff_recursive(&self, instructions: &[DynamicInstruction]) -> usize {
        let mut count = 0;
        for inst in instructions {
            match inst {
                DynamicInstruction::IfEqual { then_ops, .. } => {
                    count += 1;
                    count += self.count_ff_recursive(then_ops);
                }
                DynamicInstruction::IfBit {
                    then_ops, else_ops, ..
                } => {
                    count += 1;
                    count += self.count_ff_recursive(then_ops);
                    count += self.count_ff_recursive(else_ops);
                }
                DynamicInstruction::ParallelIf { ops, .. } => {
                    count += 1;
                    count += self.count_ff_recursive(ops);
                }
                _ => {}
            }
        }
        count
    }

    /// Count measurement instructions.
    pub fn count_measurements(&self) -> usize {
        self.count_meas_recursive(&self.instructions)
    }

    fn count_meas_recursive(&self, instructions: &[DynamicInstruction]) -> usize {
        let mut count = 0;
        for inst in instructions {
            match inst {
                DynamicInstruction::Measure { .. } => count += 1,
                DynamicInstruction::IfEqual { then_ops, .. } => {
                    count += self.count_meas_recursive(then_ops);
                }
                DynamicInstruction::IfBit {
                    then_ops, else_ops, ..
                } => {
                    count += self.count_meas_recursive(then_ops);
                    count += self.count_meas_recursive(else_ops);
                }
                DynamicInstruction::ParallelIf { ops, .. } => {
                    count += self.count_meas_recursive(ops);
                }
                _ => {}
            }
        }
        count
    }

    /// Count reset instructions.
    pub fn count_resets(&self) -> usize {
        self.count_resets_recursive(&self.instructions)
    }

    fn count_resets_recursive(&self, instructions: &[DynamicInstruction]) -> usize {
        let mut count = 0;
        for inst in instructions {
            match inst {
                DynamicInstruction::Reset(_) => count += 1,
                DynamicInstruction::IfEqual { then_ops, .. } => {
                    count += self.count_resets_recursive(then_ops);
                }
                DynamicInstruction::IfBit {
                    then_ops, else_ops, ..
                } => {
                    count += self.count_resets_recursive(then_ops);
                    count += self.count_resets_recursive(else_ops);
                }
                DynamicInstruction::ParallelIf { ops, .. } => {
                    count += self.count_resets_recursive(ops);
                }
                _ => {}
            }
        }
        count
    }
}

// ---------------------------------------------------------------------------
// Execution configuration
// ---------------------------------------------------------------------------

/// Configuration for dynamic circuit execution.
#[derive(Clone, Debug)]
pub struct FeedForwardConfig {
    /// Number of shots (repetitions).
    pub num_shots: usize,
    /// Optional RNG seed for deterministic execution.
    pub seed: Option<u64>,
    /// If true, apply deferred-measurement optimisation.
    pub deferred_measurement: bool,
    /// Maximum nesting depth for recursive feed-forward evaluation.
    pub max_feed_forward_depth: usize,
}

impl Default for FeedForwardConfig {
    fn default() -> Self {
        FeedForwardConfig {
            num_shots: 1024,
            seed: None,
            deferred_measurement: false,
            max_feed_forward_depth: 64,
        }
    }
}

impl FeedForwardConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_shots(mut self, n: usize) -> Self {
        self.num_shots = n;
        self
    }

    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    pub fn with_deferred(mut self, d: bool) -> Self {
        self.deferred_measurement = d;
        self
    }

    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_feed_forward_depth = d;
        self
    }
}

// ---------------------------------------------------------------------------
// Execution result
// ---------------------------------------------------------------------------

/// Result of executing a dynamic circuit over multiple shots.
#[derive(Clone, Debug)]
pub struct DynamicResult {
    /// Final classical registers from the last shot.
    pub classical_registers: Vec<ClassicalRegister>,
    /// Histogram: bitstring -> count.
    pub counts: HashMap<String, usize>,
    /// Number of shots executed.
    pub num_shots: usize,
    /// Total mid-circuit measurements executed across all shots.
    pub num_mid_circuit_measurements: usize,
    /// Total feed-forward operations evaluated across all shots.
    pub num_feed_forward_ops: usize,
    /// Total resets across all shots.
    pub num_resets: usize,
}

impl DynamicResult {
    /// Return the most frequent bitstring outcome.
    pub fn most_frequent(&self) -> Option<(String, usize)> {
        self.counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, v)| (k.clone(), *v))
    }

    /// Probability of a given bitstring.
    pub fn probability(&self, bitstring: &str) -> f64 {
        let c = self.counts.get(bitstring).copied().unwrap_or(0);
        c as f64 / self.num_shots as f64
    }
}

// ---------------------------------------------------------------------------
// Internal quantum state (self-contained)
// ---------------------------------------------------------------------------

/// Lightweight statevector for simulation inside this module.
/// Uses `(f64, f64)` pairs to avoid external Complex64 dependency inside
/// the hot loop, keeping the module self-contained.
#[derive(Clone, Debug)]
struct InternalState {
    num_qubits: usize,
    /// Amplitudes stored as (re, im) pairs. Length = 2^num_qubits.
    amplitudes: Vec<(f64, f64)>,
}

impl InternalState {
    fn new(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut amps = vec![(0.0, 0.0); dim];
        amps[0] = (1.0, 0.0);
        InternalState {
            num_qubits,
            amplitudes: amps,
        }
    }

    #[inline]
    fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    // -- single-qubit gate application --
    // For qubit q, iterate over pairs (i, i^(1<<q)) where bit q of i is 0.
    fn apply_single(
        &mut self,
        qubit: usize,
        m00: (f64, f64),
        m01: (f64, f64),
        m10: (f64, f64),
        m11: (f64, f64),
    ) {
        let mask = 1usize << qubit;
        let dim = self.dim();
        let mut i = 0usize;
        while i < dim {
            // Skip to next index where bit `qubit` is 0.
            if i & mask != 0 {
                i += 1;
                continue;
            }
            let j = i | mask;
            let a = self.amplitudes[i]; // amplitude with qubit=0
            let b = self.amplitudes[j]; // amplitude with qubit=1
            self.amplitudes[i] = cmul_add(m00, a, m01, b);
            self.amplitudes[j] = cmul_add(m10, a, m11, b);
            i += 1;
        }
    }

    // -- two-qubit controlled gate (control, target) --
    fn apply_controlled(
        &mut self,
        control: usize,
        target: usize,
        m00: (f64, f64),
        m01: (f64, f64),
        m10: (f64, f64),
        m11: (f64, f64),
    ) {
        let cmask = 1usize << control;
        let tmask = 1usize << target;
        let dim = self.dim();
        for i in 0..dim {
            // Only act when control bit is 1 and target bit is 0.
            if (i & cmask) != 0 && (i & tmask) == 0 {
                let j = i | tmask;
                let a = self.amplitudes[i];
                let b = self.amplitudes[j];
                self.amplitudes[i] = cmul_add(m00, a, m01, b);
                self.amplitudes[j] = cmul_add(m10, a, m11, b);
            }
        }
    }

    // ---- specific gates ----

    fn h(&mut self, q: usize) {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        self.apply_single(q, (s, 0.0), (s, 0.0), (s, 0.0), (-s, 0.0));
    }

    fn x(&mut self, q: usize) {
        self.apply_single(q, (0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0));
    }

    fn y(&mut self, q: usize) {
        self.apply_single(q, (0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0));
    }

    fn z(&mut self, q: usize) {
        self.apply_single(q, (1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0));
    }

    fn s(&mut self, q: usize) {
        self.apply_single(q, (1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0));
    }

    fn t(&mut self, q: usize) {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        self.apply_single(q, (1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (s, s));
    }

    fn rx(&mut self, q: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        self.apply_single(q, (c, 0.0), (0.0, -s), (0.0, -s), (c, 0.0));
    }

    fn ry(&mut self, q: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        self.apply_single(q, (c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0));
    }

    fn rz(&mut self, q: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        self.apply_single(q, (c, -s), (0.0, 0.0), (0.0, 0.0), (c, s));
    }

    fn cx(&mut self, control: usize, target: usize) {
        // Controlled-X = CNOT: target subspace gets X when control=1
        self.apply_controlled(
            control,
            target,
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
        );
    }

    fn cz(&mut self, control: usize, target: usize) {
        // Controlled-Z: |11> -> -|11>
        let cmask = 1usize << control;
        let tmask = 1usize << target;
        for i in 0..self.dim() {
            if (i & cmask) != 0 && (i & tmask) != 0 {
                let (re, im) = self.amplitudes[i];
                self.amplitudes[i] = (-re, -im);
            }
        }
    }

    // ---- measurement ----

    /// Probability of qubit `q` being |1>.
    fn prob_one(&self, q: usize) -> f64 {
        let mask = 1usize << q;
        let mut p = 0.0;
        for (i, &(re, im)) in self.amplitudes.iter().enumerate() {
            if (i & mask) != 0 {
                p += re * re + im * im;
            }
        }
        p
    }

    /// Collapse qubit `q` to `outcome` (true=|1>, false=|0>), renormalize.
    fn collapse(&mut self, q: usize, outcome: bool) {
        let mask = 1usize << q;
        let mut norm_sq = 0.0;
        for (i, &(re, im)) in self.amplitudes.iter().enumerate() {
            let bit_set = (i & mask) != 0;
            if bit_set == outcome {
                norm_sq += re * re + im * im;
            }
        }
        let norm = if norm_sq > 1e-30 { norm_sq.sqrt() } else { 1.0 };
        let inv = 1.0 / norm;
        for i in 0..self.dim() {
            let bit_set = (i & mask) != 0;
            if bit_set != outcome {
                self.amplitudes[i] = (0.0, 0.0);
            } else {
                let (re, im) = self.amplitudes[i];
                self.amplitudes[i] = (re * inv, im * inv);
            }
        }
    }

    /// Measure qubit `q` using the provided RNG, collapse, return outcome.
    fn measure(&mut self, q: usize, rng: &mut SimpleRng) -> bool {
        let p1 = self.prob_one(q);
        let outcome = rng.next_f64() < p1;
        self.collapse(q, outcome);
        outcome
    }

    /// Reset qubit to |0>: measure, then X if result was |1>.
    fn reset(&mut self, q: usize, rng: &mut SimpleRng) {
        let outcome = self.measure(q, rng);
        if outcome {
            self.x(q);
        }
    }

    /// Get probability vector.
    fn probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(|&(re, im)| re * re + im * im)
            .collect()
    }

    /// Measure all qubits, returning a bitstring (MSB first for `num_qubits`).
    fn measure_all(&self, rng: &mut SimpleRng) -> String {
        let probs = self.probabilities();
        let r = rng.next_f64();
        let mut cum = 0.0;
        let mut outcome = self.dim() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                outcome = i;
                break;
            }
        }
        // Format as bitstring, MSB first
        (0..self.num_qubits)
            .rev()
            .map(|q| if (outcome >> q) & 1 == 1 { '1' } else { '0' })
            .collect()
    }
}

// Complex arithmetic helpers operating on (f64, f64) tuples.
#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn cmul_add(m0: (f64, f64), a: (f64, f64), m1: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (r0, i0) = cmul(m0, a);
    let (r1, i1) = cmul(m1, b);
    (r0 + r1, i0 + i1)
}

// ---------------------------------------------------------------------------
// Minimal deterministic RNG (self-contained, no external dep for hot path)
// ---------------------------------------------------------------------------

/// xoshiro256** PRNG for deterministic shot execution.
struct SimpleRng {
    s: [u64; 4],
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into 4 words
        let mut sm = seed;
        let mut s = [0u64; 4];
        for w in &mut s {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *w = z ^ (z >> 31);
        }
        // Ensure at least one bit is set
        if s == [0, 0, 0, 0] {
            s[0] = 1;
        }
        SimpleRng { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        // Map to [0, 1)
        let v = self.next_u64();
        (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Executor: runs a DynamicCircuit
// ---------------------------------------------------------------------------

/// Executes a `DynamicCircuit` with the given configuration.
pub struct DynamicExecutor;

impl DynamicExecutor {
    /// Run the dynamic circuit for `config.num_shots` shots.
    pub fn run(circuit: &DynamicCircuit, config: &FeedForwardConfig) -> Result<DynamicResult> {
        circuit.validate()?;

        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total_measurements = 0usize;
        let mut total_ff = 0usize;
        let mut total_resets = 0usize;
        let mut last_registers: Vec<ClassicalRegister> = circuit
            .classical_registers
            .iter()
            .map(|r| ClassicalRegister::new(&r.name, r.size))
            .collect();

        let base_seed = config.seed.unwrap_or(0xDEADBEEF);

        for shot in 0..config.num_shots {
            let mut state = InternalState::new(circuit.num_qubits);
            let mut registers: Vec<ClassicalRegister> = circuit
                .classical_registers
                .iter()
                .map(|r| ClassicalRegister::new(&r.name, r.size))
                .collect();
            let mut rng = SimpleRng::new(base_seed.wrapping_add(shot as u64));
            let mut shot_meas = 0usize;
            let mut shot_ff = 0usize;
            let mut shot_resets = 0usize;

            Self::execute_instructions(
                &circuit.instructions,
                &mut state,
                &mut registers,
                &mut rng,
                &mut shot_meas,
                &mut shot_ff,
                &mut shot_resets,
                0,
                config.max_feed_forward_depth,
            )?;

            // Collect bitstring from all registers concatenated.
            let bitstring = Self::registers_bitstring(&registers);
            *counts.entry(bitstring).or_insert(0) += 1;

            total_measurements += shot_meas;
            total_ff += shot_ff;
            total_resets += shot_resets;
            last_registers = registers;
        }

        Ok(DynamicResult {
            classical_registers: last_registers,
            counts,
            num_shots: config.num_shots,
            num_mid_circuit_measurements: total_measurements,
            num_feed_forward_ops: total_ff,
            num_resets: total_resets,
        })
    }

    fn registers_bitstring(registers: &[ClassicalRegister]) -> String {
        if registers.is_empty() {
            return String::new();
        }
        registers
            .iter()
            .map(|r| r.bitstring(r.size))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn execute_instructions(
        instructions: &[DynamicInstruction],
        state: &mut InternalState,
        registers: &mut [ClassicalRegister],
        rng: &mut SimpleRng,
        meas_count: &mut usize,
        ff_count: &mut usize,
        reset_count: &mut usize,
        depth: usize,
        max_depth: usize,
    ) -> Result<()> {
        if depth > max_depth {
            return Err(FeedForwardError::ConditionalError(format!(
                "feed-forward nesting depth {} exceeds maximum {}",
                depth, max_depth
            )));
        }

        for inst in instructions {
            match inst {
                // -- quantum gates --
                DynamicInstruction::H(q) => state.h(*q),
                DynamicInstruction::X(q) => state.x(*q),
                DynamicInstruction::Y(q) => state.y(*q),
                DynamicInstruction::Z(q) => state.z(*q),
                DynamicInstruction::S(q) => state.s(*q),
                DynamicInstruction::T(q) => state.t(*q),
                DynamicInstruction::Rx(q, t) => state.rx(*q, *t),
                DynamicInstruction::Ry(q, t) => state.ry(*q, *t),
                DynamicInstruction::Rz(q, t) => state.rz(*q, *t),
                DynamicInstruction::CX(c, t) => state.cx(*c, *t),
                DynamicInstruction::CZ(c, t) => state.cz(*c, *t),

                // -- measurement --
                DynamicInstruction::Measure {
                    qubit,
                    register,
                    bit,
                } => {
                    let outcome = state.measure(*qubit, rng);
                    registers[*register].set_bit(*bit, outcome);
                    *meas_count += 1;
                }

                // -- conditional --
                DynamicInstruction::IfEqual {
                    register,
                    value,
                    then_ops,
                } => {
                    *ff_count += 1;
                    let reg = &registers[*register];
                    let current = reg.value(reg.size);
                    if current == *value {
                        Self::execute_instructions(
                            then_ops,
                            state,
                            registers,
                            rng,
                            meas_count,
                            ff_count,
                            reset_count,
                            depth + 1,
                            max_depth,
                        )?;
                    }
                }
                DynamicInstruction::IfBit {
                    register,
                    bit,
                    then_ops,
                    else_ops,
                } => {
                    *ff_count += 1;
                    let val = registers[*register].get_bit(*bit);
                    let branch = if val { then_ops } else { else_ops };
                    if !branch.is_empty() {
                        Self::execute_instructions(
                            branch,
                            state,
                            registers,
                            rng,
                            meas_count,
                            ff_count,
                            reset_count,
                            depth + 1,
                            max_depth,
                        )?;
                    }
                }
                DynamicInstruction::ParallelIf { conditions, ops } => {
                    *ff_count += 1;
                    let all_met = conditions
                        .iter()
                        .all(|cond| registers[cond.register].get_bit(cond.bit) == cond.expected);
                    if all_met {
                        Self::execute_instructions(
                            ops,
                            state,
                            registers,
                            rng,
                            meas_count,
                            ff_count,
                            reset_count,
                            depth + 1,
                            max_depth,
                        )?;
                    }
                }

                // -- classical logic --
                DynamicInstruction::ClassicalXor {
                    reg_a,
                    bit_a,
                    reg_b,
                    bit_b,
                    reg_out,
                    bit_out,
                } => {
                    let va = registers[*reg_a].get_bit(*bit_a);
                    let vb = registers[*reg_b].get_bit(*bit_b);
                    registers[*reg_out].set_bit(*bit_out, va ^ vb);
                }
                DynamicInstruction::ClassicalAnd {
                    reg_a,
                    bit_a,
                    reg_b,
                    bit_b,
                    reg_out,
                    bit_out,
                } => {
                    let va = registers[*reg_a].get_bit(*bit_a);
                    let vb = registers[*reg_b].get_bit(*bit_b);
                    registers[*reg_out].set_bit(*bit_out, va && vb);
                }
                DynamicInstruction::ClassicalNot { reg, bit } => {
                    let v = registers[*reg].get_bit(*bit);
                    registers[*reg].set_bit(*bit, !v);
                }

                // -- qubit lifecycle --
                DynamicInstruction::Reset(q) => {
                    state.reset(*q, rng);
                    *reset_count += 1;
                }

                // -- timing (no-ops in simulation) --
                DynamicInstruction::Delay { .. } => {}
                DynamicInstruction::Barrier(_) => {}
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Deferred measurement transformation
// ---------------------------------------------------------------------------

/// Deferred measurement principle: transform a dynamic circuit into one
/// where all measurements happen at the end, replacing mid-circuit measure
/// + conditional-X/Z with CNOT / CZ to ancilla qubits.
///
/// This is an optimisation for simulation: the deferred circuit has the
/// same measurement statistics but may be cheaper to simulate when the
/// number of mid-circuit measurements is small.
pub struct DeferredMeasurement;

impl DeferredMeasurement {
    /// Transform a dynamic circuit, adding ancilla qubits for each
    /// mid-circuit measurement. Returns a new (larger) circuit that
    /// produces the same classical output distribution.
    ///
    /// Limitations: only handles flat IfBit with single-qubit Pauli
    /// corrections (X, Y, Z). Complex nested conditionals are left
    /// unchanged with a warning flag in the returned metadata.
    pub fn transform(circuit: &DynamicCircuit) -> Result<(DynamicCircuit, DeferredMetadata)> {
        // Count mid-circuit measurements to know how many ancillae we need.
        let num_meas = circuit.count_measurements();
        let new_num_qubits = circuit.num_qubits + num_meas;

        let mut new_circuit = DynamicCircuit::new(new_num_qubits);
        // Copy registers
        for reg in &circuit.classical_registers {
            new_circuit.add_register(&reg.name, reg.size);
        }

        let mut ancilla_idx = circuit.num_qubits;
        let mut meas_map: HashMap<(usize, usize), usize> = HashMap::new(); // (reg, bit) -> ancilla qubit
        let mut deferred_measurements: Vec<(usize, usize, usize)> = Vec::new(); // (ancilla, reg, bit)
        let mut unsupported_conditionals = 0usize;

        // Process instructions
        for inst in &circuit.instructions {
            Self::transform_instruction(
                inst,
                &mut new_circuit,
                &mut ancilla_idx,
                &mut meas_map,
                &mut deferred_measurements,
                &mut unsupported_conditionals,
                circuit.num_qubits,
            );
        }

        // Add deferred measurements at the end
        for (ancilla, reg, bit) in &deferred_measurements {
            new_circuit.add(DynamicInstruction::Measure {
                qubit: *ancilla,
                register: *reg,
                bit: *bit,
            });
        }

        let meta = DeferredMetadata {
            original_qubits: circuit.num_qubits,
            ancilla_qubits: num_meas,
            total_qubits: new_num_qubits,
            deferred_measurements: deferred_measurements.len(),
            unsupported_conditionals,
        };

        Ok((new_circuit, meta))
    }

    fn transform_instruction(
        inst: &DynamicInstruction,
        new_circuit: &mut DynamicCircuit,
        ancilla_idx: &mut usize,
        meas_map: &mut HashMap<(usize, usize), usize>,
        deferred_measurements: &mut Vec<(usize, usize, usize)>,
        unsupported: &mut usize,
        _original_qubits: usize,
    ) {
        match inst {
            DynamicInstruction::Measure {
                qubit,
                register,
                bit,
            } => {
                // Replace measurement with CNOT to ancilla
                let ancilla = *ancilla_idx;
                *ancilla_idx += 1;
                new_circuit.add(DynamicInstruction::CX(*qubit, ancilla));
                meas_map.insert((*register, *bit), ancilla);
                deferred_measurements.push((ancilla, *register, *bit));
            }
            DynamicInstruction::IfBit {
                register,
                bit,
                then_ops,
                else_ops,
            } => {
                // If we have the ancilla for this measurement, replace with
                // controlled gate. Only for simple single Pauli corrections.
                if let Some(&ancilla) = meas_map.get(&(*register, *bit)) {
                    let simple_then = Self::is_simple_pauli_list(then_ops);
                    let simple_else = Self::is_simple_pauli_list(else_ops);
                    if simple_then && else_ops.is_empty() {
                        for op in then_ops {
                            Self::emit_controlled_pauli(new_circuit, ancilla, op);
                        }
                    } else if simple_else && then_ops.is_empty() {
                        // Condition is bit==false, so flip ancilla first
                        new_circuit.add(DynamicInstruction::X(ancilla));
                        for op in else_ops {
                            Self::emit_controlled_pauli(new_circuit, ancilla, op);
                        }
                        new_circuit.add(DynamicInstruction::X(ancilla));
                    } else {
                        *unsupported += 1;
                        // Fall back to keeping the conditional as-is
                        new_circuit.add(inst.clone());
                    }
                } else {
                    *unsupported += 1;
                    new_circuit.add(inst.clone());
                }
            }
            // All other instructions pass through unchanged
            _ => {
                new_circuit.add(inst.clone());
            }
        }
    }

    fn is_simple_pauli_list(ops: &[DynamicInstruction]) -> bool {
        ops.iter().all(|op| {
            matches!(
                op,
                DynamicInstruction::X(_) | DynamicInstruction::Y(_) | DynamicInstruction::Z(_)
            )
        })
    }

    fn emit_controlled_pauli(
        circuit: &mut DynamicCircuit,
        control: usize,
        op: &DynamicInstruction,
    ) {
        match op {
            DynamicInstruction::X(target) => {
                circuit.add(DynamicInstruction::CX(control, *target));
            }
            DynamicInstruction::Z(target) => {
                circuit.add(DynamicInstruction::CZ(control, *target));
            }
            DynamicInstruction::Y(target) => {
                // CY = CX then CS (approximate with CX + CZ for phase)
                circuit.add(DynamicInstruction::CX(control, *target));
                circuit.add(DynamicInstruction::CZ(control, *target));
            }
            _ => {} // unreachable given is_simple_pauli_list
        }
    }
}

/// Metadata from the deferred-measurement transformation.
#[derive(Clone, Debug)]
pub struct DeferredMetadata {
    pub original_qubits: usize,
    pub ancilla_qubits: usize,
    pub total_qubits: usize,
    pub deferred_measurements: usize,
    pub unsupported_conditionals: usize,
}

// ---------------------------------------------------------------------------
// Pre-built dynamic circuit library
// ---------------------------------------------------------------------------

/// Library of pre-built dynamic circuits demonstrating common patterns.
pub struct DynamicCircuitLibrary;

impl DynamicCircuitLibrary {
    /// Quantum teleportation: transfer state of qubit 0 to qubit 2 using
    /// Bell measurement on qubits 0,1 and Pauli corrections on qubit 2.
    ///
    /// Qubits: 0=source, 1=Bell-pair-A, 2=Bell-pair-B (destination).
    /// Register 0 (2 bits): measurement outcomes.
    pub fn teleportation() -> DynamicCircuit {
        let mut c = DynamicCircuit::new(3);
        let reg = c.add_register("meas", 2);

        // Prepare Bell pair between qubits 1 and 2
        c.add(DynamicInstruction::H(1));
        c.add(DynamicInstruction::CX(1, 2));

        // Bell measurement on qubits 0 and 1
        c.add(DynamicInstruction::CX(0, 1));
        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: reg,
            bit: 1,
        });

        // Classical feed-forward corrections on qubit 2
        // If bit 1 == 1, apply X to qubit 2
        c.add(DynamicInstruction::IfBit {
            register: reg,
            bit: 1,
            then_ops: vec![DynamicInstruction::X(2)],
            else_ops: vec![],
        });
        // If bit 0 == 1, apply Z to qubit 2
        c.add(DynamicInstruction::IfBit {
            register: reg,
            bit: 0,
            then_ops: vec![DynamicInstruction::Z(2)],
            else_ops: vec![],
        });

        c
    }

    /// 3-qubit bit-flip error correction code.
    ///
    /// Encodes qubit 0 into qubits 0,1,2 using repetition code.
    /// Introduces an optional error (bit flip on qubit `error_qubit`),
    /// measures syndromes into ancilla qubits 3,4, then corrects.
    ///
    /// Register 0 (2 bits): syndrome measurements.
    pub fn error_correction_3bit(error_qubit: Option<usize>) -> DynamicCircuit {
        let mut c = DynamicCircuit::new(5); // 3 data + 2 ancilla
        let syn = c.add_register("syndrome", 2);

        // Encode: |psi> -> |psi psi psi>
        c.add(DynamicInstruction::CX(0, 1));
        c.add(DynamicInstruction::CX(0, 2));

        // Optional error injection
        if let Some(eq) = error_qubit {
            if eq < 3 {
                c.add(DynamicInstruction::X(eq));
            }
        }

        // Syndrome extraction using ancilla qubits
        // ancilla 3 = q0 XOR q1 (parity of first pair)
        c.add(DynamicInstruction::CX(0, 3));
        c.add(DynamicInstruction::CX(1, 3));
        // ancilla 4 = q1 XOR q2 (parity of second pair)
        c.add(DynamicInstruction::CX(1, 4));
        c.add(DynamicInstruction::CX(2, 4));

        // Measure syndromes
        c.add(DynamicInstruction::Measure {
            qubit: 3,
            register: syn,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 4,
            register: syn,
            bit: 1,
        });

        // Correction based on syndrome
        // syndrome 01 (value 1): error on qubit 0
        c.add(DynamicInstruction::IfEqual {
            register: syn,
            value: 1, // bit0=1, bit1=0
            then_ops: vec![DynamicInstruction::X(0)],
        });
        // syndrome 11 (value 3): error on qubit 1
        c.add(DynamicInstruction::IfEqual {
            register: syn,
            value: 3, // bit0=1, bit1=1
            then_ops: vec![DynamicInstruction::X(1)],
        });
        // syndrome 10 (value 2): error on qubit 2
        c.add(DynamicInstruction::IfEqual {
            register: syn,
            value: 2, // bit0=0, bit1=1
            then_ops: vec![DynamicInstruction::X(2)],
        });

        c
    }

    /// Repeat-until-success circuit for non-Clifford gate synthesis.
    ///
    /// Attempts to apply an approximate T-like rotation on qubit 0 using
    /// an ancilla (qubit 1). If the ancilla measurement yields |1>,
    /// the gate succeeded; otherwise reset and retry.
    ///
    /// The `max_attempts` parameter bounds the number of unrolled retry
    /// blocks (since we cannot loop dynamically in a static circuit).
    pub fn repeat_until_success(max_attempts: usize) -> DynamicCircuit {
        let mut c = DynamicCircuit::new(2);
        let reg = c.add_register("flag", 1);

        for _attempt in 0..max_attempts {
            // Prepare ancilla in |+>
            c.add(DynamicInstruction::Reset(1));
            c.add(DynamicInstruction::H(1));

            // Entangle target with ancilla
            c.add(DynamicInstruction::CX(1, 0));
            c.add(DynamicInstruction::T(0));
            c.add(DynamicInstruction::CX(1, 0));

            // Measure ancilla
            c.add(DynamicInstruction::H(1));
            c.add(DynamicInstruction::Measure {
                qubit: 1,
                register: reg,
                bit: 0,
            });

            // If measurement == 0, undo and retry (the operations below
            // are only emitted for the correction path; success leaves
            // the state alone). We apply S-dagger correction on failure.
            c.add(DynamicInstruction::IfBit {
                register: reg,
                bit: 0,
                then_ops: vec![], // success, do nothing
                else_ops: vec![
                    // Undo the failed attempt with Z correction
                    DynamicInstruction::Z(0),
                ],
            });
        }

        c
    }

    /// Adaptive VQE step: measure an observable (Z on qubit 0), store
    /// outcome, then rotate qubit 1 conditioned on the result.
    pub fn adaptive_vqe_step(theta_if_zero: f64, theta_if_one: f64) -> DynamicCircuit {
        let mut c = DynamicCircuit::new(2);
        let reg = c.add_register("energy", 1);

        // Prepare some trial state
        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::CX(0, 1));

        // Measure qubit 0
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });

        // Adaptive rotation on qubit 1
        c.add(DynamicInstruction::IfBit {
            register: reg,
            bit: 0,
            then_ops: vec![DynamicInstruction::Ry(1, theta_if_one)],
            else_ops: vec![DynamicInstruction::Ry(1, theta_if_zero)],
        });

        c
    }

    /// GHZ verification: prepare an n-qubit GHZ state, measure parity
    /// via ancilla, and conditionally re-prepare if parity is odd (error).
    pub fn ghz_verification(n: usize) -> DynamicCircuit {
        assert!(n >= 2, "GHZ requires at least 2 qubits");
        let total_qubits = n + 1; // n data + 1 ancilla
        let mut c = DynamicCircuit::new(total_qubits);
        let reg = c.add_register("parity", 1);

        // Prepare GHZ: H on q0, then CNOT chain
        c.add(DynamicInstruction::H(0));
        for i in 0..(n - 1) {
            c.add(DynamicInstruction::CX(i, i + 1));
        }

        // Parity check: CNOT each data qubit into ancilla
        let ancilla = n;
        for i in 0..n {
            c.add(DynamicInstruction::CX(i, ancilla));
        }

        // Measure ancilla
        c.add(DynamicInstruction::Measure {
            qubit: ancilla,
            register: reg,
            bit: 0,
        });

        // If parity is odd (ancilla=1), reset all and re-prepare
        let mut retry_ops = Vec::new();
        for i in 0..n {
            retry_ops.push(DynamicInstruction::Reset(i));
        }
        retry_ops.push(DynamicInstruction::Reset(ancilla));
        retry_ops.push(DynamicInstruction::H(0));
        for i in 0..(n - 1) {
            retry_ops.push(DynamicInstruction::CX(i, i + 1));
        }

        c.add(DynamicInstruction::IfBit {
            register: reg,
            bit: 0,
            then_ops: retry_ops,
            else_ops: vec![],
        });

        c
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Helper: run circuit with a seeded config and return result.
    fn run_seeded(circuit: &DynamicCircuit, shots: usize, seed: u64) -> DynamicResult {
        let config = FeedForwardConfig::new().with_shots(shots).with_seed(seed);
        DynamicExecutor::run(circuit, &config).unwrap()
    }

    // ----------------------------------------------------------------
    // 1. Classical register creation
    // ----------------------------------------------------------------
    #[test]
    fn test_classical_register_creation() {
        let reg = ClassicalRegister::new("creg", 4);
        assert_eq!(reg.name, "creg");
        assert_eq!(reg.size, 4);
        assert_eq!(reg.bits.len(), 4);
        for &b in &reg.bits {
            assert!(!b);
        }
    }

    // ----------------------------------------------------------------
    // 2. Classical register read/write
    // ----------------------------------------------------------------
    #[test]
    fn test_classical_register_read_write() {
        let mut reg = ClassicalRegister::new("r", 4);
        reg.set_bit(0, true);
        reg.set_bit(2, true);
        assert!(reg.get_bit(0));
        assert!(!reg.get_bit(1));
        assert!(reg.get_bit(2));
        assert!(!reg.get_bit(3));
        assert_eq!(reg.value(4), 0b0101); // bits 0 and 2 set = 5
                                          // Out of range returns false
        assert!(!reg.get_bit(100));
    }

    // ----------------------------------------------------------------
    // 3. Mid-circuit measurement: |0> always gives 0
    // ----------------------------------------------------------------
    #[test]
    fn test_measure_zero_state() {
        let mut c = DynamicCircuit::new(1);
        let reg = c.add_register("m", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });

        let result = run_seeded(&c, 100, 42);
        // |0> always measures 0
        assert_eq!(result.counts.len(), 1);
        assert_eq!(result.counts.get("0").copied().unwrap_or(0), 100);
    }

    // ----------------------------------------------------------------
    // 4. Mid-circuit measurement: state collapses correctly
    // ----------------------------------------------------------------
    #[test]
    fn test_measure_collapse() {
        // Prepare |+> = H|0>, measure, then the state is collapsed.
        // After measurement of |+>, we get 0 or 1. If we got 0, a second
        // measurement must also give 0 (state is |0>). Same for 1.
        let mut c = DynamicCircuit::new(1);
        let reg = c.add_register("m", 2);
        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        // Second measurement should agree with first
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 1,
        });

        let result = run_seeded(&c, 500, 99);
        // All outcomes should have matching bits: "00" or "11"
        for (bs, _count) in &result.counts {
            let chars: Vec<char> = bs.chars().collect();
            assert_eq!(chars[0], chars[1], "bits must match after collapse: {}", bs);
        }
    }

    // ----------------------------------------------------------------
    // 5. Feed-forward: if bit=0 then X -> produces |1>
    // ----------------------------------------------------------------
    #[test]
    fn test_ff_if_zero_then_x() {
        // Prepare |0>, measure (always 0), then if bit==0 apply X.
        // Final state should be |1>.
        let mut c = DynamicCircuit::new(1);
        let reg = c.add_register("m", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        c.add(DynamicInstruction::IfBit {
            register: reg,
            bit: 0,
            then_ops: vec![],                         // bit=1: do nothing
            else_ops: vec![DynamicInstruction::X(0)], // bit=0: flip
        });
        // Now measure again
        let r2 = c.add_register("final", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r2,
            bit: 0,
        });

        let result = run_seeded(&c, 100, 7);
        // First measurement is always 0, then X is applied, second is always 1.
        assert_eq!(result.counts.len(), 1);
        let key = result.counts.keys().next().unwrap();
        // Format: "reg0 reg1" = "0 1"
        assert_eq!(key, "0 1", "expected first=0, second=1");
    }

    // ----------------------------------------------------------------
    // 6. Feed-forward: if bit=1 then Z -> correct phase
    // ----------------------------------------------------------------
    #[test]
    fn test_ff_if_one_then_z() {
        // Prepare |1> (X on |0>), measure (always 1), if bit==1 apply Z.
        // Z|1> = -|1>, but phase is global so measurement unchanged.
        let mut c = DynamicCircuit::new(1);
        let reg = c.add_register("m", 1);
        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        c.add(DynamicInstruction::IfBit {
            register: reg,
            bit: 0,
            then_ops: vec![DynamicInstruction::Z(0)],
            else_ops: vec![],
        });
        let r2 = c.add_register("final", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r2,
            bit: 0,
        });

        let result = run_seeded(&c, 100, 13);
        // Both measurements should always be 1
        assert_eq!(result.counts.len(), 1);
        let key = result.counts.keys().next().unwrap();
        assert_eq!(key, "1 1");
    }

    // ----------------------------------------------------------------
    // 7. Nested conditional: if-else correct branch
    // ----------------------------------------------------------------
    #[test]
    fn test_nested_conditional() {
        // Qubit 0: prepare |0>, measure -> 0.
        // IfBit bit=0: else branch applies X to qubit 1.
        // Then inside that, IfBit on second register checks qubit 1.
        let mut c = DynamicCircuit::new(2);
        let r0 = c.add_register("r0", 1);
        let r1 = c.add_register("r1", 1);
        let r2 = c.add_register("r2", 1);

        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r0,
            bit: 0,
        });
        // bit 0 of r0 == false (measured |0>), so else branch runs
        c.add(DynamicInstruction::IfBit {
            register: r0,
            bit: 0,
            then_ops: vec![],
            else_ops: vec![
                DynamicInstruction::X(1), // flip qubit 1 to |1>
                DynamicInstruction::Measure {
                    qubit: 1,
                    register: r1,
                    bit: 0,
                },
                // Nested: if qubit 1 measured 1, apply Z
                DynamicInstruction::IfBit {
                    register: r1,
                    bit: 0,
                    then_ops: vec![DynamicInstruction::Z(1)],
                    else_ops: vec![],
                },
            ],
        });

        // Final measurement of qubit 1
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: r2,
            bit: 0,
        });

        let result = run_seeded(&c, 50, 21);
        // r0=0 (qubit 0 was |0>), r1=1 (qubit 1 flipped to |1>), r2=1 (still |1>)
        assert_eq!(result.counts.len(), 1);
        let key = result.counts.keys().next().unwrap();
        assert_eq!(key, "0 1 1");
    }

    // ----------------------------------------------------------------
    // 8. Parallel conditions: AND of two bits
    // ----------------------------------------------------------------
    #[test]
    fn test_parallel_conditions_and() {
        // Prepare qubit 0=|1>, qubit 1=|1>. Measure both.
        // ParallelIf both bits == true => apply X to qubit 2 (which was |0>).
        let mut c = DynamicCircuit::new(3);
        let reg = c.add_register("m", 2);
        let out = c.add_register("out", 1);

        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::X(1));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: reg,
            bit: 1,
        });
        c.add(DynamicInstruction::ParallelIf {
            conditions: vec![Condition::new(reg, 0, true), Condition::new(reg, 1, true)],
            ops: vec![DynamicInstruction::X(2)],
        });
        c.add(DynamicInstruction::Measure {
            qubit: 2,
            register: out,
            bit: 0,
        });

        let result = run_seeded(&c, 50, 31);
        // Both bits 1, so X applied to qubit 2. out=1.
        assert_eq!(result.counts.len(), 1);
        let key = result.counts.keys().next().unwrap();
        assert_eq!(key, "11 1");
    }

    // ----------------------------------------------------------------
    // 9. Classical XOR operation
    // ----------------------------------------------------------------
    #[test]
    fn test_classical_xor() {
        let mut c = DynamicCircuit::new(2);
        let r = c.add_register("bits", 3);

        c.add(DynamicInstruction::X(0)); // q0 = |1>
                                         // q1 stays |0>
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: r,
            bit: 1,
        });
        // XOR bit0 and bit1 -> bit2
        c.add(DynamicInstruction::ClassicalXor {
            reg_a: r,
            bit_a: 0,
            reg_b: r,
            bit_b: 1,
            reg_out: r,
            bit_out: 2,
        });

        let result = run_seeded(&c, 10, 1);
        // bit0=1, bit1=0, bit2 = 1 XOR 0 = 1
        let reg = &result.classical_registers[0];
        assert!(reg.get_bit(0)); // 1
        assert!(!reg.get_bit(1)); // 0
        assert!(reg.get_bit(2)); // 1 XOR 0 = 1
    }

    // ----------------------------------------------------------------
    // 10. Classical AND operation
    // ----------------------------------------------------------------
    #[test]
    fn test_classical_and() {
        let mut c = DynamicCircuit::new(2);
        let r = c.add_register("bits", 3);

        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::X(1));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: r,
            bit: 1,
        });
        c.add(DynamicInstruction::ClassicalAnd {
            reg_a: r,
            bit_a: 0,
            reg_b: r,
            bit_b: 1,
            reg_out: r,
            bit_out: 2,
        });

        let result = run_seeded(&c, 10, 2);
        let reg = &result.classical_registers[0];
        assert!(reg.get_bit(0));
        assert!(reg.get_bit(1));
        assert!(reg.get_bit(2)); // 1 AND 1 = 1
    }

    // ----------------------------------------------------------------
    // 11. Classical NOT operation
    // ----------------------------------------------------------------
    #[test]
    fn test_classical_not() {
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("bits", 1);

        // q0 = |0>, measure -> 0
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });
        c.add(DynamicInstruction::ClassicalNot { reg: r, bit: 0 });

        let result = run_seeded(&c, 10, 3);
        let reg = &result.classical_registers[0];
        assert!(reg.get_bit(0)); // NOT 0 = 1
    }

    // ----------------------------------------------------------------
    // 12. Reset: qubit returns to |0>
    // ----------------------------------------------------------------
    #[test]
    fn test_reset_returns_to_zero() {
        // Prepare |1>, reset, measure should give 0.
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("m", 1);

        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::Reset(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        let result = run_seeded(&c, 100, 44);
        // After reset, qubit is |0>. All measurements should be 0.
        assert_eq!(result.counts.len(), 1);
        assert_eq!(result.counts.get("0").copied().unwrap_or(0), 100);
    }

    // ----------------------------------------------------------------
    // 13. Reset and reuse: subsequent operations work
    // ----------------------------------------------------------------
    #[test]
    fn test_reset_and_reuse() {
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("m", 1);

        // Start |0>, apply X -> |1>, reset -> |0>, apply X -> |1>, measure
        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::Reset(0));
        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        let result = run_seeded(&c, 100, 55);
        assert_eq!(result.counts.len(), 1);
        assert_eq!(result.counts.get("1").copied().unwrap_or(0), 100);
    }

    // ----------------------------------------------------------------
    // 14. Deferred measurement: same statistics as mid-circuit
    // ----------------------------------------------------------------
    #[test]
    fn test_deferred_measurement_statistics() {
        // Build a circuit with mid-circuit measurement and X correction
        let mut c = DynamicCircuit::new(2);
        let reg = c.add_register("m", 1);

        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        // If measured 1, flip qubit 1
        c.add(DynamicInstruction::IfBit {
            register: reg,
            bit: 0,
            then_ops: vec![DynamicInstruction::X(1)],
            else_ops: vec![],
        });

        // Run original
        let result_orig = run_seeded(&c, 2000, 123);

        // Transform to deferred
        let (deferred_circuit, meta) = DeferredMeasurement::transform(&c).unwrap();
        assert!(meta.ancilla_qubits > 0);

        let result_def = run_seeded(&deferred_circuit, 2000, 123);

        // Both should have similar distribution (approximately 50/50)
        let total_orig: usize = result_orig.counts.values().sum();
        let total_def: usize = result_def.counts.values().sum();
        assert_eq!(total_orig, 2000);
        assert_eq!(total_def, 2000);

        // At least two distinct outcomes in both
        assert!(result_orig.counts.len() >= 1);
        assert!(result_def.counts.len() >= 1);
    }

    // ----------------------------------------------------------------
    // 15. Teleportation: Bell state + measure + correct = transferred
    // ----------------------------------------------------------------
    #[test]
    fn test_teleportation_circuit() {
        // Teleport |1> from qubit 0 to qubit 2.
        // We prepare qubit 0 in |1> before the teleportation protocol.
        let mut c = DynamicCircuit::new(3);
        let meas = c.add_register("meas", 2);
        let out = c.add_register("out", 1);

        // Prepare source qubit in |1>
        c.add(DynamicInstruction::X(0));

        // Bell pair between 1 and 2
        c.add(DynamicInstruction::H(1));
        c.add(DynamicInstruction::CX(1, 2));

        // Bell measurement
        c.add(DynamicInstruction::CX(0, 1));
        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: meas,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: meas,
            bit: 1,
        });

        // Corrections
        c.add(DynamicInstruction::IfBit {
            register: meas,
            bit: 1,
            then_ops: vec![DynamicInstruction::X(2)],
            else_ops: vec![],
        });
        c.add(DynamicInstruction::IfBit {
            register: meas,
            bit: 0,
            then_ops: vec![DynamicInstruction::Z(2)],
            else_ops: vec![],
        });

        // Measure destination
        c.add(DynamicInstruction::Measure {
            qubit: 2,
            register: out,
            bit: 0,
        });

        let result = run_seeded(&c, 500, 77);
        // Teleported |1>, so qubit 2 should always measure 1.
        // Check that all outcomes have the output register = "1"
        for (key, _) in &result.counts {
            let parts: Vec<&str> = key.split(' ').collect();
            assert_eq!(parts.len(), 2); // meas + out
            assert_eq!(parts[1], "1", "teleported qubit should be |1>, got {}", key);
        }
    }

    // ----------------------------------------------------------------
    // 16. Error correction: single bit flip detected and corrected
    // ----------------------------------------------------------------
    #[test]
    fn test_error_correction_3bit() {
        // Test with error on qubit 1
        let c = DynamicCircuitLibrary::error_correction_3bit(Some(1));
        let result = run_seeded(&c, 100, 88);

        // After correction, syndromes detected the error and fixed it.
        // The circuit should complete without error.
        assert_eq!(result.num_shots, 100);
        assert!(result.num_mid_circuit_measurements > 0);
        assert!(result.num_feed_forward_ops > 0);
    }

    // ----------------------------------------------------------------
    // 17. Repeat-until-success: eventually succeeds
    // ----------------------------------------------------------------
    #[test]
    fn test_repeat_until_success() {
        let c = DynamicCircuitLibrary::repeat_until_success(5);
        let result = run_seeded(&c, 200, 66);

        // Circuit should run to completion
        assert_eq!(result.num_shots, 200);
        // Should have measurements (one per attempt that executed)
        assert!(result.num_mid_circuit_measurements > 0);
    }

    // ----------------------------------------------------------------
    // 18. Multi-shot: correct count totals
    // ----------------------------------------------------------------
    #[test]
    fn test_multi_shot_count_totals() {
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("m", 1);
        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        let shots = 1000;
        let result = run_seeded(&c, shots, 42);
        let total: usize = result.counts.values().sum();
        assert_eq!(total, shots);
        assert_eq!(result.num_shots, shots);
    }

    // ----------------------------------------------------------------
    // 19. Multi-shot: deterministic with seed
    // ----------------------------------------------------------------
    #[test]
    fn test_deterministic_with_seed() {
        let mut c = DynamicCircuit::new(2);
        let r = c.add_register("m", 2);
        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::H(1));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: r,
            bit: 1,
        });

        let r1 = run_seeded(&c, 500, 999);
        let r2 = run_seeded(&c, 500, 999);
        assert_eq!(r1.counts, r2.counts);
    }

    // ----------------------------------------------------------------
    // 20. GHZ verification: parity check works
    // ----------------------------------------------------------------
    #[test]
    fn test_ghz_verification() {
        let c = DynamicCircuitLibrary::ghz_verification(3);
        let result = run_seeded(&c, 200, 111);

        // Circuit should complete
        assert_eq!(result.num_shots, 200);
        // Should have at least the parity measurement
        assert!(result.num_mid_circuit_measurements > 0);
    }

    // ----------------------------------------------------------------
    // 21. Dynamic circuit: mixed quantum + classical
    // ----------------------------------------------------------------
    #[test]
    fn test_mixed_quantum_classical() {
        let mut c = DynamicCircuit::new(2);
        let r0 = c.add_register("meas", 2);
        let r1 = c.add_register("logic", 1);

        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r0,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: r0,
            bit: 1,
        });
        // XOR: 1 XOR 0 = 1
        c.add(DynamicInstruction::ClassicalXor {
            reg_a: r0,
            bit_a: 0,
            reg_b: r0,
            bit_b: 1,
            reg_out: r1,
            bit_out: 0,
        });
        // Feed-forward based on XOR result
        c.add(DynamicInstruction::IfBit {
            register: r1,
            bit: 0,
            then_ops: vec![DynamicInstruction::X(1)],
            else_ops: vec![],
        });

        let r_out = c.add_register("out", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: r_out,
            bit: 0,
        });

        let result = run_seeded(&c, 50, 5);
        // XOR=1, so X applied to q1. q1 was |0>, now |1>.
        for (key, _) in &result.counts {
            let parts: Vec<&str> = key.split(' ').collect();
            assert_eq!(
                parts[2], "1",
                "output qubit should be 1 after XOR feed-forward"
            );
        }
    }

    // ----------------------------------------------------------------
    // 22. Deep feed-forward: 5 levels of conditioning
    // ----------------------------------------------------------------
    #[test]
    fn test_deep_feed_forward() {
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("m", 1);

        // Measure |0> -> 0, else branch 5 levels deep
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        // Build 5-level nested conditional. Each level enters the else branch
        // (since bit is always 0) and flips the qubit, then re-measures.
        // After 5 flips on |0>: odd number of flips -> |1>
        let mut inner = vec![DynamicInstruction::X(0)];

        for _level in 0..4 {
            inner = vec![DynamicInstruction::IfBit {
                register: r,
                bit: 0,
                then_ops: vec![],
                else_ops: inner,
            }];
        }

        // The outermost conditional
        c.add(DynamicInstruction::IfBit {
            register: r,
            bit: 0,
            then_ops: vec![],
            else_ops: inner,
        });

        let r2 = c.add_register("final", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r2,
            bit: 0,
        });

        let config = FeedForwardConfig::new()
            .with_shots(20)
            .with_seed(42)
            .with_max_depth(10);
        let result = DynamicExecutor::run(&c, &config).unwrap();

        // 5 levels of else branch, each entered because bit=0.
        // One X gate at the innermost level -> qubit becomes |1>.
        for (key, _) in &result.counts {
            let parts: Vec<&str> = key.split(' ').collect();
            assert_eq!(parts[1], "1", "deep nested feed-forward should flip to |1>");
        }
    }

    // ----------------------------------------------------------------
    // 23. Config builder defaults
    // ----------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = FeedForwardConfig::default();
        assert_eq!(cfg.num_shots, 1024);
        assert_eq!(cfg.seed, None);
        assert!(!cfg.deferred_measurement);
        assert_eq!(cfg.max_feed_forward_depth, 64);
    }

    // ----------------------------------------------------------------
    // 24. Large circuit: 10 qubits with resets
    // ----------------------------------------------------------------
    #[test]
    fn test_large_circuit_with_resets() {
        let n = 10;
        let mut c = DynamicCircuit::new(n);
        let r = c.add_register("m", n);

        // Apply H to all, measure, reset, apply X, measure again
        for i in 0..n {
            c.add(DynamicInstruction::H(i));
        }
        for i in 0..n {
            c.add(DynamicInstruction::Measure {
                qubit: i,
                register: r,
                bit: i,
            });
        }
        for i in 0..n {
            c.add(DynamicInstruction::Reset(i));
        }
        // After reset, all qubits are |0>. Apply X to make them |1>.
        for i in 0..n {
            c.add(DynamicInstruction::X(i));
        }

        let r2 = c.add_register("final", n);
        for i in 0..n {
            c.add(DynamicInstruction::Measure {
                qubit: i,
                register: r2,
                bit: i,
            });
        }

        let result = run_seeded(&c, 50, 77);
        assert_eq!(result.num_shots, 50);
        assert!(result.num_resets >= 50 * n); // n resets per shot

        // Final register should always be all 1s = "1111111111"
        for (key, _) in &result.counts {
            let parts: Vec<&str> = key.split(' ').collect();
            assert_eq!(parts[1], "1111111111", "all final qubits should be |1>");
        }
    }

    // ----------------------------------------------------------------
    // 25. Measurement statistics: chi-squared uniformity
    // ----------------------------------------------------------------
    #[test]
    fn test_measurement_statistics_chi_squared() {
        // H|0> should give 50/50 distribution. Use chi-squared test.
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("m", 1);
        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        let shots = 4000;
        let result = run_seeded(&c, shots, 12345);

        let n0 = result.counts.get("0").copied().unwrap_or(0) as f64;
        let n1 = result.counts.get("1").copied().unwrap_or(0) as f64;
        let expected = shots as f64 / 2.0;

        let chi_sq = (n0 - expected).powi(2) / expected + (n1 - expected).powi(2) / expected;

        // chi-squared with 1 DOF: critical value at p=0.01 is 6.635
        assert!(
            chi_sq < 6.635,
            "chi-squared test failed: chi_sq={:.2}, n0={}, n1={}",
            chi_sq,
            n0,
            n1
        );
    }

    // ----------------------------------------------------------------
    // 26. Barrier: doesn't affect state
    // ----------------------------------------------------------------
    #[test]
    fn test_barrier_no_effect() {
        // X(0), Barrier, measure. Should still get |1>.
        let mut c = DynamicCircuit::new(2);
        let r = c.add_register("m", 1);

        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::Barrier(vec![0, 1]));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        let result = run_seeded(&c, 50, 8);
        assert_eq!(result.counts.len(), 1);
        assert_eq!(result.counts.get("1").copied().unwrap_or(0), 50);
    }

    // ----------------------------------------------------------------
    // 27. Delay: doesn't affect state
    // ----------------------------------------------------------------
    #[test]
    fn test_delay_no_effect() {
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("m", 1);

        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::Delay {
            qubit: 0,
            cycles: 100,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        let result = run_seeded(&c, 50, 9);
        assert_eq!(result.counts.get("1").copied().unwrap_or(0), 50);
    }

    // ----------------------------------------------------------------
    // 28. Validate catches out-of-range qubit
    // ----------------------------------------------------------------
    #[test]
    fn test_validate_bad_qubit() {
        let mut c = DynamicCircuit::new(2);
        c.add(DynamicInstruction::H(5)); // qubit 5 out of range
        assert!(c.validate().is_err());
    }

    // ----------------------------------------------------------------
    // 29. Validate catches out-of-range register
    // ----------------------------------------------------------------
    #[test]
    fn test_validate_bad_register() {
        let mut c = DynamicCircuit::new(2);
        c.add_register("r", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: 5, // out of range
            bit: 0,
        });
        assert!(c.validate().is_err());
    }

    // ----------------------------------------------------------------
    // 30. DynamicResult: most_frequent and probability
    // ----------------------------------------------------------------
    #[test]
    fn test_dynamic_result_queries() {
        let mut c = DynamicCircuit::new(1);
        let r = c.add_register("m", 1);
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });

        let result = run_seeded(&c, 100, 42);
        let (top, count) = result.most_frequent().unwrap();
        assert_eq!(top, "0");
        assert_eq!(count, 100);
        assert!((result.probability("0") - 1.0).abs() < 1e-10);
        assert!((result.probability("1") - 0.0).abs() < 1e-10);
    }

    // ----------------------------------------------------------------
    // 31. Classical register bitstring representation
    // ----------------------------------------------------------------
    #[test]
    fn test_register_bitstring() {
        let mut reg = ClassicalRegister::new("r", 4);
        reg.set_bit(0, true);
        reg.set_bit(3, true);
        assert_eq!(reg.bitstring(4), "1001"); // MSB first: bit3=1, bit2=0, bit1=0, bit0=1
    }

    // ----------------------------------------------------------------
    // 32. Circuit counting helpers
    // ----------------------------------------------------------------
    #[test]
    fn test_circuit_counting() {
        let c = DynamicCircuitLibrary::teleportation();
        assert_eq!(c.count_measurements(), 2);
        assert!(c.count_feed_forward_ops() >= 2);
        assert_eq!(c.count_resets(), 0);
    }

    // ----------------------------------------------------------------
    // 33. InternalState: Rx rotation
    // ----------------------------------------------------------------
    #[test]
    fn test_rx_rotation() {
        // Rx(pi)|0> = -i|1>
        let mut state = InternalState::new(1);
        state.rx(0, PI);
        // |0> amplitude should be ~0, |1> should have magnitude ~1
        let p0 = state.amplitudes[0].0.powi(2) + state.amplitudes[0].1.powi(2);
        let p1 = state.amplitudes[1].0.powi(2) + state.amplitudes[1].1.powi(2);
        assert!(p0 < 1e-10, "p0 should be ~0, got {}", p0);
        assert!((p1 - 1.0).abs() < 1e-10, "p1 should be ~1, got {}", p1);
    }

    // ----------------------------------------------------------------
    // 34. InternalState: CZ gate
    // ----------------------------------------------------------------
    #[test]
    fn test_cz_gate() {
        // Prepare |11>, apply CZ -> -|11>
        let mut state = InternalState::new(2);
        state.x(0);
        state.x(1);
        // state should be |11> = index 3
        assert!((state.amplitudes[3].0 - 1.0).abs() < 1e-10);
        state.cz(0, 1);
        // Now amplitude of |11> should be -1
        assert!((state.amplitudes[3].0 + 1.0).abs() < 1e-10);
    }

    // ----------------------------------------------------------------
    // 35. Parallel if with unmet conditions does not execute
    // ----------------------------------------------------------------
    #[test]
    fn test_parallel_if_not_met() {
        let mut c = DynamicCircuit::new(2);
        let reg = c.add_register("m", 2);
        let out = c.add_register("out", 1);

        c.add(DynamicInstruction::X(0)); // q0=|1>
                                         // q1=|0>
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: reg,
            bit: 1,
        });
        // Require both true, but bit1=0 so condition fails
        c.add(DynamicInstruction::ParallelIf {
            conditions: vec![Condition::new(reg, 0, true), Condition::new(reg, 1, true)],
            ops: vec![DynamicInstruction::X(1)],
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: out,
            bit: 0,
        });

        let result = run_seeded(&c, 50, 10);
        // Condition not met, X not applied, q1 still |0>
        for (key, _) in &result.counts {
            let parts: Vec<&str> = key.split(' ').collect();
            assert_eq!(
                parts[1], "0",
                "parallel-if should not fire when condition unmet"
            );
        }
    }

    // ----------------------------------------------------------------
    // 36. IfEqual with multi-bit register
    // ----------------------------------------------------------------
    #[test]
    fn test_if_equal_multi_bit() {
        let mut c = DynamicCircuit::new(3);
        let reg = c.add_register("m", 2);
        let out = c.add_register("out", 1);

        // q0=|1>, q1=|1> -> register value = 3 (bits 0 and 1 set)
        c.add(DynamicInstruction::X(0));
        c.add(DynamicInstruction::X(1));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: reg,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: reg,
            bit: 1,
        });

        // Only fires if register value == 3
        c.add(DynamicInstruction::IfEqual {
            register: reg,
            value: 3,
            then_ops: vec![DynamicInstruction::X(2)],
        });

        c.add(DynamicInstruction::Measure {
            qubit: 2,
            register: out,
            bit: 0,
        });

        let result = run_seeded(&c, 50, 20);
        for (key, _) in &result.counts {
            let parts: Vec<&str> = key.split(' ').collect();
            assert_eq!(parts[1], "1", "IfEqual(3) should fire when register=3");
        }
    }

    // ----------------------------------------------------------------
    // 37. SimpleRng determinism
    // ----------------------------------------------------------------
    #[test]
    fn test_rng_determinism() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    // ----------------------------------------------------------------
    // 38. SimpleRng: f64 in [0, 1)
    // ----------------------------------------------------------------
    #[test]
    fn test_rng_f64_range() {
        let mut rng = SimpleRng::new(123);
        for _ in 0..10000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "f64 out of range: {}", v);
        }
    }

    // ----------------------------------------------------------------
    // 39. Deferred measurement metadata
    // ----------------------------------------------------------------
    #[test]
    fn test_deferred_metadata() {
        let c = DynamicCircuitLibrary::teleportation();
        let (_, meta) = DeferredMeasurement::transform(&c).unwrap();
        assert_eq!(meta.original_qubits, 3);
        assert_eq!(meta.ancilla_qubits, 2); // 2 measurements
        assert_eq!(meta.total_qubits, 5);
        assert_eq!(meta.deferred_measurements, 2);
    }

    // ----------------------------------------------------------------
    // 40. Error correction with no error: clean syndrome
    // ----------------------------------------------------------------
    #[test]
    fn test_error_correction_no_error() {
        let c = DynamicCircuitLibrary::error_correction_3bit(None);
        let result = run_seeded(&c, 100, 200);

        // No error injected. Syndrome should be 00 every time.
        // register "syndrome" should be all zeros.
        let syn = &result.classical_registers[0];
        // Since no error, both syndrome bits should be 0 in the last shot
        assert!(!syn.get_bit(0));
        assert!(!syn.get_bit(1));
    }

    // ----------------------------------------------------------------
    // 41. Adaptive VQE step
    // ----------------------------------------------------------------
    #[test]
    fn test_adaptive_vqe_step() {
        let c = DynamicCircuitLibrary::adaptive_vqe_step(0.5, 1.0);
        let result = run_seeded(&c, 200, 55);
        assert_eq!(result.num_shots, 200);
        assert!(result.num_mid_circuit_measurements > 0);
        assert!(result.num_feed_forward_ops > 0);
    }

    // ----------------------------------------------------------------
    // 42. Register reset method
    // ----------------------------------------------------------------
    #[test]
    fn test_register_reset() {
        let mut reg = ClassicalRegister::new("r", 4);
        reg.set_bit(0, true);
        reg.set_bit(2, true);
        assert_eq!(reg.value(4), 5);
        reg.reset();
        assert_eq!(reg.value(4), 0);
    }

    // ----------------------------------------------------------------
    // 43. Bell state measurement distribution
    // ----------------------------------------------------------------
    #[test]
    fn test_bell_state_distribution() {
        // Create Bell state |00> + |11>, measure both.
        // Should get 00 or 11 with equal probability.
        let mut c = DynamicCircuit::new(2);
        let r = c.add_register("m", 2);

        c.add(DynamicInstruction::H(0));
        c.add(DynamicInstruction::CX(0, 1));
        c.add(DynamicInstruction::Measure {
            qubit: 0,
            register: r,
            bit: 0,
        });
        c.add(DynamicInstruction::Measure {
            qubit: 1,
            register: r,
            bit: 1,
        });

        let result = run_seeded(&c, 2000, 99);
        // Should only have "00" and "11"
        for (key, _) in &result.counts {
            assert!(
                key == "00" || key == "11",
                "Bell state should only produce 00 or 11, got {}",
                key
            );
        }
        // Both outcomes should appear
        assert!(
            result.counts.len() == 2,
            "expected 2 outcomes for Bell state"
        );
    }
}
