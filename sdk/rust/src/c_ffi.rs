//! C Foreign Function Interface for nQPU-Metal Quantum Simulator
//!
//! Exposes nQPU-Metal's core quantum simulation capabilities to C, C++,
//! Fortran, Julia, and other languages via a stable C ABI. This enables HPC
//! integration where quantum simulation runs as a library call from existing
//! scientific codes.
//!
//! # Design Principles
//!
//! - **Opaque handles**: Callers receive pointers to opaque structs. Internal
//!   layout is never exposed across the ABI boundary.
//! - **Error codes**: Every fallible operation returns `NqpuError`. The zero
//!   value (`Ok`) indicates success.
//! - **Null safety**: All `extern "C"` entry points check pointer arguments
//!   before dereferencing.
//! - **Panic safety**: Critical paths are wrapped in `std::panic::catch_unwind`
//!   so a Rust panic never unwinds into the C caller.
//! - **Self-contained**: This module carries its own gate implementations and
//!   does not depend on `crate::GateOperations` or `crate::QuantumState`.
//!   This avoids tight coupling and allows the C API to evolve independently.
//!
//! # Capability Parity
//!
//! The API surface matches Qiskit v2.x C API capability:
//! - State vector creation, cloning, reset, amplitude access
//! - Full Clifford+T gate set (H, X, Y, Z, S, T, Rx, Ry, Rz)
//! - Controlled gates (CX, CZ), SWAP
//! - Custom 1-qubit and 2-qubit unitary gates
//! - Single-qubit measurement with state collapse
//! - Full register measurement
//! - Pauli-Z expectation values
//! - Shot-based sampling
//! - Circuit construction and batch execution
//! - Thread-local RNG seeding for reproducibility
//!
//! # C Header Generation
//!
//! Call [`generate_c_header`] to obtain a complete `nqpu_metal.h` header file
//! suitable for inclusion in C/C++ projects.

use std::cell::RefCell;
use std::os::raw::c_char;
use std::ptr;

// ---------------------------------------------------------------------------
// Thread-local RNG
// ---------------------------------------------------------------------------

thread_local! {
    static THREAD_RNG_SEED: RefCell<u64> = RefCell::new(0);
    static THREAD_RNG_STATE: RefCell<u64> = RefCell::new(0x853c49e6748fea9b);
}

/// Simple xorshift64* PRNG for deterministic, reproducible measurement.
/// This avoids pulling in `rand` as a hard dependency for the FFI surface.
fn next_random_f64() -> f64 {
    THREAD_RNG_STATE.with(|cell| {
        let mut s = *cell.borrow();
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        *cell.borrow_mut() = s;
        // Map to [0, 1)
        let val = s.wrapping_mul(0x2545F4914F6CDD1D);
        (val >> 11) as f64 / (1u64 << 53) as f64
    })
}

// ===================================================================
// C ABI TYPES
// ===================================================================

/// Opaque handle to a quantum state vector.
///
/// Callers must not inspect or modify the contents. Use the `nqpu_state_*`
/// family of functions for all interactions.
#[repr(C)]
pub struct NqpuState {
    _private: [u8; 0],
}

/// Opaque handle to a quantum circuit (sequence of gates).
///
/// Callers must not inspect or modify the contents. Use the `nqpu_circuit_*`
/// family of functions for all interactions.
#[repr(C)]
pub struct NqpuCircuit {
    _private: [u8; 0],
}

/// Error codes returned by all fallible FFI functions.
///
/// `Ok` (0) indicates success. Any other value indicates an error condition
/// that the caller should handle.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NqpuError {
    /// Operation completed successfully.
    Ok = 0,
    /// A qubit index exceeds the number of qubits in the state.
    InvalidQubit = 1,
    /// A function parameter is out of range or otherwise invalid.
    InvalidParameter = 2,
    /// Memory allocation failed (out of memory).
    AllocationFailed = 3,
    /// The simulation encountered an unrecoverable error.
    SimulationFailed = 4,
    /// A required pointer argument was null.
    NullPointer = 5,
    /// The opaque handle does not point to a valid object.
    InvalidHandle = 6,
}

/// Double-precision complex number for C interop.
///
/// Layout is guaranteed to match two contiguous `f64` values (real, imag),
/// which is binary-compatible with C99 `_Complex double` on most platforms.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NqpuComplex {
    pub re: f64,
    pub im: f64,
}

impl NqpuComplex {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        NqpuComplex { re, im }
    }

    #[inline]
    pub fn zero() -> Self {
        NqpuComplex { re: 0.0, im: 0.0 }
    }

    #[inline]
    pub fn one() -> Self {
        NqpuComplex { re: 1.0, im: 0.0 }
    }

    #[inline]
    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    #[inline]
    pub fn mul(&self, other: &NqpuComplex) -> NqpuComplex {
        NqpuComplex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Complex addition
    #[inline]
    pub fn add(&self, other: &NqpuComplex) -> NqpuComplex {
        NqpuComplex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Scale by a real factor
    #[inline]
    pub fn scale(&self, factor: f64) -> NqpuComplex {
        NqpuComplex {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
}

/// Result of measuring a single qubit.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NqpuMeasurement {
    /// Measurement outcome: 0 or 1.
    pub outcome: u32,
    /// Born-rule probability of the observed outcome.
    pub probability: f64,
}

/// Simulation configuration (reserved for future use).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NqpuConfig {
    pub num_qubits: u32,
    pub seed: u64,
    pub use_gpu: bool,
    pub num_threads: u32,
}

/// Semantic version triple.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NqpuVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

// ===================================================================
// INTERNAL STATE (not exposed across ABI)
// ===================================================================

/// Internal quantum state representation.
///
/// The state vector stores 2^n complex amplitudes where n = num_qubits.
/// Amplitude at index i corresponds to the computational basis state |i>.
struct InternalQuantumState {
    num_qubits: usize,
    amplitudes: Vec<NqpuComplex>,
}

impl InternalQuantumState {
    /// Create |0...0> state for `n` qubits.
    fn new(n: usize) -> Self {
        let dim = 1usize << n;
        let mut amplitudes = vec![NqpuComplex::zero(); dim];
        amplitudes[0] = NqpuComplex::one();
        InternalQuantumState {
            num_qubits: n,
            amplitudes,
        }
    }

    #[inline]
    fn dim(&self) -> usize {
        1usize << self.num_qubits
    }

    /// Reset to |0...0>.
    fn reset(&mut self) {
        for a in self.amplitudes.iter_mut() {
            *a = NqpuComplex::zero();
        }
        self.amplitudes[0] = NqpuComplex::one();
    }

    /// Compute squared norm (should be 1.0 for a valid state).
    fn norm(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>()
    }

    /// Compute all probabilities.
    fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Validate that a qubit index is in range.
    #[inline]
    fn check_qubit(&self, qubit: u32) -> Result<(), NqpuError> {
        if (qubit as usize) >= self.num_qubits {
            Err(NqpuError::InvalidQubit)
        } else {
            Ok(())
        }
    }

    // ---------------------------------------------------------------
    // 1-qubit gate kernel
    // ---------------------------------------------------------------

    /// Apply an arbitrary 2x2 unitary matrix to a single qubit.
    ///
    /// The matrix is given in row-major order: [m00, m01, m10, m11].
    fn apply_1q(&mut self, qubit: usize, m: &[NqpuComplex; 4]) {
        let stride = 1usize << qubit;
        let dim = self.dim();
        let mut i = 0;
        while i < dim {
            for k in 0..stride {
                let idx0 = i + k;
                let idx1 = idx0 + stride;
                let a = self.amplitudes[idx0];
                let b = self.amplitudes[idx1];
                self.amplitudes[idx0] = m[0].mul(&a).add(&m[1].mul(&b));
                self.amplitudes[idx1] = m[2].mul(&a).add(&m[3].mul(&b));
            }
            i += stride * 2;
        }
    }

    /// Apply a diagonal phase gate to a single qubit.
    ///
    /// The diagonal is [1, phase] meaning |0> is unchanged and |1> picks
    /// up the given complex phase.
    fn apply_phase(&mut self, qubit: usize, phase: NqpuComplex) {
        let stride = 1usize << qubit;
        let dim = self.dim();
        let mut i = 0;
        while i < dim {
            for k in 0..stride {
                let idx1 = i + k + stride;
                let b = self.amplitudes[idx1];
                self.amplitudes[idx1] = phase.mul(&b);
            }
            i += stride * 2;
        }
    }

    // ---------------------------------------------------------------
    // 2-qubit gate kernel
    // ---------------------------------------------------------------

    /// Apply an arbitrary 4x4 unitary matrix to two qubits.
    ///
    /// The matrix is given in row-major order as a 16-element array.
    /// Basis ordering: |q1=0,q2=0>, |q1=0,q2=1>, |q1=1,q2=0>, |q1=1,q2=1>
    /// where q1 is the higher-order qubit and q2 is the lower-order qubit.
    fn apply_2q(&mut self, q1: usize, q2: usize, m: &[NqpuComplex; 16]) {
        let (hi, lo) = if q1 > q2 { (q1, q2) } else { (q2, q1) };
        let stride_lo = 1usize << lo;
        let stride_hi = 1usize << hi;
        let dim = self.dim();

        // Iterate over all basis states, grouped by the two qubit bits
        let mut base = 0;
        while base < dim {
            // Skip indices where the hi or lo bit is already set
            // by iterating in blocks
            for j in 0..stride_lo {
                let idx = base + j;
                // The four basis states for qubits (q1, q2):
                // We need to compute indices with the correct bit positions
                let i00 = idx & !(stride_lo | stride_hi); // clear both bits
                let _i00 = i00 | (idx & (stride_lo - 1)); // restore bits below lo
                // Actually, let's just use the direct approach:
                let bit_lo = (idx >> lo) & 1;
                let bit_hi = (idx >> hi) & 1;
                if bit_lo != 0 || bit_hi != 0 {
                    continue;
                }

                let i00 = idx;
                let i01 = idx | stride_lo;
                let i10 = idx | stride_hi;
                let i11 = idx | stride_lo | stride_hi;

                let a00 = self.amplitudes[i00];
                let a01 = self.amplitudes[i01];
                let a10 = self.amplitudes[i10];
                let a11 = self.amplitudes[i11];

                // Map (q1, q2) → matrix index:
                // If q1 > q2: |q1,q2> order matches (hi,lo), matrix row ordering
                // is |00>, |01>, |10>, |11> where first index is q1, second is q2.
                let (v0, v1, v2, v3) = if q1 > q2 {
                    (a00, a01, a10, a11)
                } else {
                    // q2 > q1: swap roles. Matrix expects |q1,q2> but our hi/lo
                    // indexing has q2 as hi. So |q1=0,q2=0>=i00, |q1=0,q2=1>=i10,
                    // |q1=1,q2=0>=i01, |q1=1,q2=1>=i11.
                    (a00, a10, a01, a11)
                };

                let r0 = m[0].mul(&v0).add(&m[1].mul(&v1)).add(&m[2].mul(&v2)).add(&m[3].mul(&v3));
                let r1 = m[4].mul(&v0).add(&m[5].mul(&v1)).add(&m[6].mul(&v2)).add(&m[7].mul(&v3));
                let r2 = m[8].mul(&v0).add(&m[9].mul(&v1)).add(&m[10].mul(&v2)).add(&m[11].mul(&v3));
                let r3 = m[12].mul(&v0).add(&m[13].mul(&v1)).add(&m[14].mul(&v2)).add(&m[15].mul(&v3));

                if q1 > q2 {
                    self.amplitudes[i00] = r0;
                    self.amplitudes[i01] = r1;
                    self.amplitudes[i10] = r2;
                    self.amplitudes[i11] = r3;
                } else {
                    self.amplitudes[i00] = r0;
                    self.amplitudes[i10] = r1;
                    self.amplitudes[i01] = r2;
                    self.amplitudes[i11] = r3;
                }
            }
            // Advance to next block
            // We need to iterate over all indices where both target bits are 0.
            // Simple approach: iterate over all indices, skip non-zero target bits.
            base += 1;
            // Skip if we hit a target bit
            while base < dim && (((base >> lo) & 1) != 0 || ((base >> hi) & 1) != 0) {
                base += 1;
            }
        }
    }

    // ---------------------------------------------------------------
    // Standard gates
    // ---------------------------------------------------------------

    fn gate_h(&mut self, qubit: usize) {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let m = [
            NqpuComplex::new(inv_sqrt2, 0.0),
            NqpuComplex::new(inv_sqrt2, 0.0),
            NqpuComplex::new(inv_sqrt2, 0.0),
            NqpuComplex::new(-inv_sqrt2, 0.0),
        ];
        self.apply_1q(qubit, &m);
    }

    fn gate_x(&mut self, qubit: usize) {
        let stride = 1usize << qubit;
        let dim = self.dim();
        let mut i = 0;
        while i < dim {
            for k in 0..stride {
                self.amplitudes.swap(i + k, i + k + stride);
            }
            i += stride * 2;
        }
    }

    fn gate_y(&mut self, qubit: usize) {
        let m = [
            NqpuComplex::zero(),
            NqpuComplex::new(0.0, -1.0),
            NqpuComplex::new(0.0, 1.0),
            NqpuComplex::zero(),
        ];
        self.apply_1q(qubit, &m);
    }

    fn gate_z(&mut self, qubit: usize) {
        self.apply_phase(qubit, NqpuComplex::new(-1.0, 0.0));
    }

    fn gate_s(&mut self, qubit: usize) {
        self.apply_phase(qubit, NqpuComplex::new(0.0, 1.0));
    }

    fn gate_t(&mut self, qubit: usize) {
        let angle = std::f64::consts::FRAC_PI_4;
        self.apply_phase(qubit, NqpuComplex::new(angle.cos(), angle.sin()));
    }

    fn gate_rx(&mut self, qubit: usize, angle: f64) {
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        let m = [
            NqpuComplex::new(c, 0.0),
            NqpuComplex::new(0.0, -s),
            NqpuComplex::new(0.0, -s),
            NqpuComplex::new(c, 0.0),
        ];
        self.apply_1q(qubit, &m);
    }

    fn gate_ry(&mut self, qubit: usize, angle: f64) {
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        let m = [
            NqpuComplex::new(c, 0.0),
            NqpuComplex::new(-s, 0.0),
            NqpuComplex::new(s, 0.0),
            NqpuComplex::new(c, 0.0),
        ];
        self.apply_1q(qubit, &m);
    }

    fn gate_rz(&mut self, qubit: usize, angle: f64) {
        let half = angle / 2.0;
        // Rz = diag(e^{-i*theta/2}, e^{i*theta/2})
        // We apply as: |0> *= e^{-i*theta/2}, |1> *= e^{i*theta/2}
        let phase0 = NqpuComplex::new(half.cos(), -half.sin());
        let phase1 = NqpuComplex::new(half.cos(), half.sin());

        let stride = 1usize << qubit;
        let dim = self.dim();
        let mut i = 0;
        while i < dim {
            for k in 0..stride {
                let idx0 = i + k;
                let idx1 = idx0 + stride;
                let a = self.amplitudes[idx0];
                let b = self.amplitudes[idx1];
                self.amplitudes[idx0] = phase0.mul(&a);
                self.amplitudes[idx1] = phase1.mul(&b);
            }
            i += stride * 2;
        }
    }

    fn gate_cx(&mut self, control: usize, target: usize) {
        let _stride_c = 1usize << control;
        let stride_t = 1usize << target;
        let dim = self.dim();

        for i in 0..dim {
            // Only process when control bit is 1 and target bit is 0
            if ((i >> control) & 1) == 1 && ((i >> target) & 1) == 0 {
                let j = i | stride_t;
                self.amplitudes.swap(i, j);
            }
        }
    }

    fn gate_cz(&mut self, q1: usize, q2: usize) {
        let dim = self.dim();
        for i in 0..dim {
            if ((i >> q1) & 1) == 1 && ((i >> q2) & 1) == 1 {
                let a = self.amplitudes[i];
                self.amplitudes[i] = NqpuComplex::new(-a.re, -a.im);
            }
        }
    }

    fn gate_swap(&mut self, q1: usize, q2: usize) {
        let dim = self.dim();
        for i in 0..dim {
            let bit1 = (i >> q1) & 1;
            let bit2 = (i >> q2) & 1;
            if bit1 != bit2 && bit1 < bit2 {
                // Swap |...1_q1...0_q2...> with |...0_q1...1_q2...>
                let j = i ^ (1 << q1) ^ (1 << q2);
                self.amplitudes.swap(i, j);
            }
        }
    }

    /// Measure a single qubit, collapsing the state.
    fn measure_qubit(&mut self, qubit: usize) -> NqpuMeasurement {
        let _stride = 1usize << qubit;
        let dim = self.dim();

        // Compute probability of measuring |0> on this qubit
        let mut prob0 = 0.0;
        for i in 0..dim {
            if ((i >> qubit) & 1) == 0 {
                prob0 += self.amplitudes[i].norm_sqr();
            }
        }
        let prob1 = 1.0 - prob0;

        // Sample outcome
        let r = next_random_f64();
        let (outcome, prob_outcome) = if r < prob0 {
            (0u32, prob0)
        } else {
            (1u32, prob1)
        };

        // Collapse: zero out amplitudes inconsistent with outcome, renormalize
        let renorm = 1.0 / prob_outcome.sqrt();
        for i in 0..dim {
            let bit = ((i >> qubit) & 1) as u32;
            if bit == outcome {
                self.amplitudes[i] = self.amplitudes[i].scale(renorm);
            } else {
                self.amplitudes[i] = NqpuComplex::zero();
            }
        }

        NqpuMeasurement {
            outcome,
            probability: prob_outcome,
        }
    }

    /// Measure all qubits, returning the outcome bits.
    fn measure_all(&mut self) -> Vec<u32> {
        let mut outcomes = Vec::with_capacity(self.num_qubits);
        // Measure from qubit 0 to num_qubits-1 sequentially
        for q in 0..self.num_qubits {
            let m = self.measure_qubit(q);
            outcomes.push(m.outcome);
        }
        outcomes
    }

    /// Expectation value of Pauli-Z on a single qubit.
    fn expectation_z(&self, qubit: usize) -> f64 {
        let dim = self.dim();
        let mut exp = 0.0;
        for i in 0..dim {
            let prob = self.amplitudes[i].norm_sqr();
            if ((i >> qubit) & 1) == 0 {
                exp += prob;
            } else {
                exp -= prob;
            }
        }
        exp
    }

    /// Sample `num_shots` measurement outcomes without collapsing the state.
    fn sample(&self, num_shots: u32) -> Vec<u64> {
        let probs = self.probabilities();
        let mut results = Vec::with_capacity(num_shots as usize);
        for _ in 0..num_shots {
            let r = next_random_f64();
            let mut cumsum = 0.0;
            let mut outcome = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    outcome = i;
                    break;
                }
            }
            results.push(outcome as u64);
        }
        results
    }
}

// ===================================================================
// INTERNAL CIRCUIT REPRESENTATION
// ===================================================================

/// A gate in the circuit.
#[derive(Clone, Debug)]
enum CircuitGate {
    H(u32),
    X(u32),
    Y(u32),
    Z(u32),
    S(u32),
    T(u32),
    Rx(u32, f64),
    Ry(u32, f64),
    Rz(u32, f64),
    Cx(u32, u32),
    Cz(u32, u32),
    Swap(u32, u32),
    Custom1Q(u32, [NqpuComplex; 4]),
    Custom2Q(u32, u32, [NqpuComplex; 16]),
}

impl CircuitGate {
    /// Return the qubits this gate acts on.
    fn qubits(&self) -> Vec<u32> {
        match self {
            CircuitGate::H(q)
            | CircuitGate::X(q)
            | CircuitGate::Y(q)
            | CircuitGate::Z(q)
            | CircuitGate::S(q)
            | CircuitGate::T(q)
            | CircuitGate::Rx(q, _)
            | CircuitGate::Ry(q, _)
            | CircuitGate::Rz(q, _) => vec![*q],
            CircuitGate::Cx(c, t) | CircuitGate::Cz(c, t) | CircuitGate::Swap(c, t) => {
                vec![*c, *t]
            }
            CircuitGate::Custom1Q(q, _) => vec![*q],
            CircuitGate::Custom2Q(q1, q2, _) => vec![*q1, *q2],
        }
    }
}

struct InternalCircuit {
    num_qubits: u32,
    gates: Vec<CircuitGate>,
}

impl InternalCircuit {
    fn new(num_qubits: u32) -> Self {
        InternalCircuit {
            num_qubits,
            gates: Vec::new(),
        }
    }

    fn gate_count(&self) -> u32 {
        self.gates.len() as u32
    }

    /// Compute circuit depth via a greedy layer assignment.
    ///
    /// Each gate is assigned to the earliest layer where none of its qubits
    /// conflict with an already-assigned gate in that layer. The depth is
    /// the number of layers.
    fn depth(&self) -> u32 {
        if self.gates.is_empty() {
            return 0;
        }
        // Track the next available layer for each qubit
        let mut qubit_depth = vec![0u32; self.num_qubits as usize];
        let mut max_depth = 0u32;

        for gate in &self.gates {
            let qubits = gate.qubits();
            // This gate must go in a layer >= max of all its qubit depths
            let layer = qubits
                .iter()
                .map(|&q| qubit_depth[q as usize])
                .max()
                .unwrap_or(0);
            // Update all involved qubits to layer + 1
            for &q in &qubits {
                qubit_depth[q as usize] = layer + 1;
            }
            if layer + 1 > max_depth {
                max_depth = layer + 1;
            }
        }
        max_depth
    }

    /// Execute all gates on the given state.
    fn execute(&self, state: &mut InternalQuantumState) -> Result<(), NqpuError> {
        for gate in &self.gates {
            match gate {
                CircuitGate::H(q) => state.gate_h(*q as usize),
                CircuitGate::X(q) => state.gate_x(*q as usize),
                CircuitGate::Y(q) => state.gate_y(*q as usize),
                CircuitGate::Z(q) => state.gate_z(*q as usize),
                CircuitGate::S(q) => state.gate_s(*q as usize),
                CircuitGate::T(q) => state.gate_t(*q as usize),
                CircuitGate::Rx(q, angle) => state.gate_rx(*q as usize, *angle),
                CircuitGate::Ry(q, angle) => state.gate_ry(*q as usize, *angle),
                CircuitGate::Rz(q, angle) => state.gate_rz(*q as usize, *angle),
                CircuitGate::Cx(c, t) => state.gate_cx(*c as usize, *t as usize),
                CircuitGate::Cz(q1, q2) => state.gate_cz(*q1 as usize, *q2 as usize),
                CircuitGate::Swap(q1, q2) => state.gate_swap(*q1 as usize, *q2 as usize),
                CircuitGate::Custom1Q(q, m) => state.apply_1q(*q as usize, m),
                CircuitGate::Custom2Q(q1, q2, m) => {
                    state.apply_2q(*q1 as usize, *q2 as usize, m)
                }
            }
        }
        Ok(())
    }
}

// ===================================================================
// HANDLE CONVERSION HELPERS
// ===================================================================

/// Convert raw NqpuState pointer to a reference, returning NullPointer on null.
unsafe fn state_ref<'a>(ptr: *const NqpuState) -> Result<&'a InternalQuantumState, NqpuError> {
    if ptr.is_null() {
        return Err(NqpuError::NullPointer);
    }
    Ok(&*(ptr as *const InternalQuantumState))
}

/// Convert raw NqpuState pointer to a mutable reference.
unsafe fn state_mut<'a>(ptr: *mut NqpuState) -> Result<&'a mut InternalQuantumState, NqpuError> {
    if ptr.is_null() {
        return Err(NqpuError::NullPointer);
    }
    Ok(&mut *(ptr as *mut InternalQuantumState))
}

/// Convert raw NqpuCircuit pointer to a reference.
unsafe fn circuit_ref<'a>(ptr: *const NqpuCircuit) -> Result<&'a InternalCircuit, NqpuError> {
    if ptr.is_null() {
        return Err(NqpuError::NullPointer);
    }
    Ok(&*(ptr as *const InternalCircuit))
}

/// Convert raw NqpuCircuit pointer to a mutable reference.
unsafe fn circuit_mut<'a>(ptr: *mut NqpuCircuit) -> Result<&'a mut InternalCircuit, NqpuError> {
    if ptr.is_null() {
        return Err(NqpuError::NullPointer);
    }
    Ok(&mut *(ptr as *mut InternalCircuit))
}

// ===================================================================
// STATE MANAGEMENT — extern "C" API
// ===================================================================

/// Create a new quantum state initialized to |0...0> with the given number
/// of qubits.
///
/// Returns a non-null pointer on success, or null if allocation fails or
/// `num_qubits` is 0 or > 30.
///
/// The caller owns the returned pointer and must free it with
/// `nqpu_state_destroy`.
#[no_mangle]
pub extern "C" fn nqpu_state_create(num_qubits: u32) -> *mut NqpuState {
    if num_qubits == 0 || num_qubits > 30 {
        return ptr::null_mut();
    }
    let result = std::panic::catch_unwind(|| {
        let state = Box::new(InternalQuantumState::new(num_qubits as usize));
        Box::into_raw(state) as *mut NqpuState
    });
    result.unwrap_or(ptr::null_mut())
}

/// Destroy a quantum state and free its memory.
///
/// It is safe to pass a null pointer (no-op). Passing an invalid non-null
/// pointer is undefined behavior.
#[no_mangle]
pub extern "C" fn nqpu_state_destroy(state: *mut NqpuState) {
    if !state.is_null() {
        unsafe {
            let _ = Box::from_raw(state as *mut InternalQuantumState);
        }
    }
}

/// Create an independent deep copy of a quantum state.
///
/// Returns null if the input pointer is null or allocation fails.
#[no_mangle]
pub extern "C" fn nqpu_state_clone(state: *const NqpuState) -> *mut NqpuState {
    if state.is_null() {
        return ptr::null_mut();
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let internal = &*(state as *const InternalQuantumState);
        let cloned = Box::new(InternalQuantumState {
            num_qubits: internal.num_qubits,
            amplitudes: internal.amplitudes.clone(),
        });
        Box::into_raw(cloned) as *mut NqpuState
    }));
    result.unwrap_or(ptr::null_mut())
}

/// Return the number of qubits in the state.
///
/// Returns 0 if the pointer is null.
#[no_mangle]
pub extern "C" fn nqpu_state_num_qubits(state: *const NqpuState) -> u32 {
    if state.is_null() {
        return 0;
    }
    unsafe {
        let internal = &*(state as *const InternalQuantumState);
        internal.num_qubits as u32
    }
}

/// Read the complex amplitude at `index` in the state vector.
///
/// `index` must be in [0, 2^num_qubits).
#[no_mangle]
pub extern "C" fn nqpu_state_get_amplitude(
    state: *const NqpuState,
    index: u64,
    out: *mut NqpuComplex,
) -> NqpuError {
    unsafe {
        let s = match state_ref(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if out.is_null() {
            return NqpuError::NullPointer;
        }
        let idx = index as usize;
        if idx >= s.dim() {
            return NqpuError::InvalidParameter;
        }
        *out = s.amplitudes[idx];
        NqpuError::Ok
    }
}

/// Set the complex amplitude at `index` in the state vector.
///
/// **Warning**: This does not renormalize the state. The caller is
/// responsible for maintaining unit norm.
#[no_mangle]
pub extern "C" fn nqpu_state_set_amplitude(
    state: *mut NqpuState,
    index: u64,
    amp: NqpuComplex,
) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let idx = index as usize;
        if idx >= s.dim() {
            return NqpuError::InvalidParameter;
        }
        s.amplitudes[idx] = amp;
        NqpuError::Ok
    }
}

/// Write all 2^n probabilities into the caller-provided buffer.
///
/// `len` must be >= 2^num_qubits.
#[no_mangle]
pub extern "C" fn nqpu_state_probabilities(
    state: *const NqpuState,
    out: *mut f64,
    len: u64,
) -> NqpuError {
    unsafe {
        let s = match state_ref(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if out.is_null() {
            return NqpuError::NullPointer;
        }
        let dim = s.dim();
        if (len as usize) < dim {
            return NqpuError::InvalidParameter;
        }
        let probs = s.probabilities();
        for (i, &p) in probs.iter().enumerate() {
            *out.add(i) = p;
        }
        NqpuError::Ok
    }
}

/// Return the squared norm of the state vector.
///
/// A properly normalized state returns 1.0. Returns -1.0 on null pointer.
#[no_mangle]
pub extern "C" fn nqpu_state_norm(state: *const NqpuState) -> f64 {
    if state.is_null() {
        return -1.0;
    }
    unsafe {
        let s = &*(state as *const InternalQuantumState);
        s.norm()
    }
}

/// Reset the state to |0...0>.
#[no_mangle]
pub extern "C" fn nqpu_state_reset(state: *mut NqpuState) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        s.reset();
        NqpuError::Ok
    }
}

// ===================================================================
// GATE OPERATIONS — extern "C" API
// ===================================================================

macro_rules! gate_1q {
    ($fn_name:ident, $method:ident) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(state: *mut NqpuState, qubit: u32) -> NqpuError {
            unsafe {
                let s = match state_mut(state) {
                    Ok(s) => s,
                    Err(e) => return e,
                };
                if let Err(e) = s.check_qubit(qubit) {
                    return e;
                }
                s.$method(qubit as usize);
                NqpuError::Ok
            }
        }
    };
}

macro_rules! gate_1q_param {
    ($fn_name:ident, $method:ident) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(
            state: *mut NqpuState,
            qubit: u32,
            angle: f64,
        ) -> NqpuError {
            unsafe {
                let s = match state_mut(state) {
                    Ok(s) => s,
                    Err(e) => return e,
                };
                if let Err(e) = s.check_qubit(qubit) {
                    return e;
                }
                if !angle.is_finite() {
                    return NqpuError::InvalidParameter;
                }
                s.$method(qubit as usize, angle);
                NqpuError::Ok
            }
        }
    };
}

gate_1q!(nqpu_gate_h, gate_h);
gate_1q!(nqpu_gate_x, gate_x);
gate_1q!(nqpu_gate_y, gate_y);
gate_1q!(nqpu_gate_z, gate_z);
gate_1q!(nqpu_gate_s, gate_s);
gate_1q!(nqpu_gate_t, gate_t);

gate_1q_param!(nqpu_gate_rx, gate_rx);
gate_1q_param!(nqpu_gate_ry, gate_ry);
gate_1q_param!(nqpu_gate_rz, gate_rz);

/// Controlled-X (CNOT) gate.
#[no_mangle]
pub extern "C" fn nqpu_gate_cx(state: *mut NqpuState, control: u32, target: u32) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if let Err(e) = s.check_qubit(control) {
            return e;
        }
        if let Err(e) = s.check_qubit(target) {
            return e;
        }
        if control == target {
            return NqpuError::InvalidParameter;
        }
        s.gate_cx(control as usize, target as usize);
        NqpuError::Ok
    }
}

/// Controlled-Z gate.
#[no_mangle]
pub extern "C" fn nqpu_gate_cz(state: *mut NqpuState, control: u32, target: u32) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if let Err(e) = s.check_qubit(control) {
            return e;
        }
        if let Err(e) = s.check_qubit(target) {
            return e;
        }
        if control == target {
            return NqpuError::InvalidParameter;
        }
        s.gate_cz(control as usize, target as usize);
        NqpuError::Ok
    }
}

/// SWAP gate.
#[no_mangle]
pub extern "C" fn nqpu_gate_swap(state: *mut NqpuState, q1: u32, q2: u32) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if let Err(e) = s.check_qubit(q1) {
            return e;
        }
        if let Err(e) = s.check_qubit(q2) {
            return e;
        }
        if q1 == q2 {
            return NqpuError::InvalidParameter;
        }
        s.gate_swap(q1 as usize, q2 as usize);
        NqpuError::Ok
    }
}

/// Apply a custom 1-qubit unitary gate.
///
/// `matrix` must point to 4 contiguous `NqpuComplex` values in row-major
/// order: [[m00, m01], [m10, m11]].
#[no_mangle]
pub extern "C" fn nqpu_gate_custom_1q(
    state: *mut NqpuState,
    qubit: u32,
    matrix: *const NqpuComplex,
) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if matrix.is_null() {
            return NqpuError::NullPointer;
        }
        if let Err(e) = s.check_qubit(qubit) {
            return e;
        }
        let m: [NqpuComplex; 4] = [*matrix.add(0), *matrix.add(1), *matrix.add(2), *matrix.add(3)];
        s.apply_1q(qubit as usize, &m);
        NqpuError::Ok
    }
}

/// Apply a custom 2-qubit unitary gate.
///
/// `matrix` must point to 16 contiguous `NqpuComplex` values in row-major
/// order for the 4x4 unitary.
#[no_mangle]
pub extern "C" fn nqpu_gate_custom_2q(
    state: *mut NqpuState,
    q1: u32,
    q2: u32,
    matrix: *const NqpuComplex,
) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if matrix.is_null() {
            return NqpuError::NullPointer;
        }
        if let Err(e) = s.check_qubit(q1) {
            return e;
        }
        if let Err(e) = s.check_qubit(q2) {
            return e;
        }
        if q1 == q2 {
            return NqpuError::InvalidParameter;
        }
        let mut m = [NqpuComplex::zero(); 16];
        for i in 0..16 {
            m[i] = *matrix.add(i);
        }
        s.apply_2q(q1 as usize, q2 as usize, &m);
        NqpuError::Ok
    }
}

// ===================================================================
// MEASUREMENT — extern "C" API
// ===================================================================

/// Measure a single qubit, collapsing the state.
///
/// The measurement outcome and its Born-rule probability are written to `out`.
#[no_mangle]
pub extern "C" fn nqpu_measure(
    state: *mut NqpuState,
    qubit: u32,
    out: *mut NqpuMeasurement,
) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if out.is_null() {
            return NqpuError::NullPointer;
        }
        if let Err(e) = s.check_qubit(qubit) {
            return e;
        }
        *out = s.measure_qubit(qubit as usize);
        NqpuError::Ok
    }
}

/// Measure all qubits, writing per-qubit outcomes (0 or 1) into `out`.
///
/// `num_qubits` must match the state's qubit count. `out` must have room
/// for at least `num_qubits` elements.
#[no_mangle]
pub extern "C" fn nqpu_measure_all(
    state: *mut NqpuState,
    out: *mut u32,
    num_qubits: u32,
) -> NqpuError {
    unsafe {
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if out.is_null() {
            return NqpuError::NullPointer;
        }
        if num_qubits as usize != s.num_qubits {
            return NqpuError::InvalidParameter;
        }
        let outcomes = s.measure_all();
        for (i, &o) in outcomes.iter().enumerate() {
            *out.add(i) = o;
        }
        NqpuError::Ok
    }
}

/// Compute the expectation value of the Pauli-Z operator on `qubit`.
///
/// Returns <psi|Z_q|psi> in the range [-1, +1]. Does not modify the state.
#[no_mangle]
pub extern "C" fn nqpu_expectation_z(
    state: *const NqpuState,
    qubit: u32,
    out: *mut f64,
) -> NqpuError {
    unsafe {
        let s = match state_ref(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if out.is_null() {
            return NqpuError::NullPointer;
        }
        if let Err(e) = s.check_qubit(qubit) {
            return e;
        }
        *out = s.expectation_z(qubit as usize);
        NqpuError::Ok
    }
}

/// Sample `num_shots` measurement outcomes without collapsing the state.
///
/// Each outcome is a basis-state index in [0, 2^n). The caller must provide
/// a buffer of at least `num_shots` `u64` values.
#[no_mangle]
pub extern "C" fn nqpu_sample(
    state: *const NqpuState,
    num_shots: u32,
    out: *mut u64,
) -> NqpuError {
    unsafe {
        let s = match state_ref(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if out.is_null() {
            return NqpuError::NullPointer;
        }
        if num_shots == 0 {
            return NqpuError::Ok;
        }
        let results = s.sample(num_shots);
        for (i, &r) in results.iter().enumerate() {
            *out.add(i) = r;
        }
        NqpuError::Ok
    }
}

// ===================================================================
// CIRCUIT API — extern "C"
// ===================================================================

/// Create a new empty circuit for the given number of qubits.
///
/// Returns null if `num_qubits` is 0, > 30, or on allocation failure.
#[no_mangle]
pub extern "C" fn nqpu_circuit_create(num_qubits: u32) -> *mut NqpuCircuit {
    if num_qubits == 0 || num_qubits > 30 {
        return ptr::null_mut();
    }
    let result = std::panic::catch_unwind(|| {
        let circuit = Box::new(InternalCircuit::new(num_qubits));
        Box::into_raw(circuit) as *mut NqpuCircuit
    });
    result.unwrap_or(ptr::null_mut())
}

/// Destroy a circuit and free its memory.
#[no_mangle]
pub extern "C" fn nqpu_circuit_destroy(circuit: *mut NqpuCircuit) {
    if !circuit.is_null() {
        unsafe {
            let _ = Box::from_raw(circuit as *mut InternalCircuit);
        }
    }
}

/// Append a Hadamard gate to the circuit.
#[no_mangle]
pub extern "C" fn nqpu_circuit_add_h(circuit: *mut NqpuCircuit, qubit: u32) -> NqpuError {
    unsafe {
        let c = match circuit_mut(circuit) {
            Ok(c) => c,
            Err(e) => return e,
        };
        if qubit >= c.num_qubits {
            return NqpuError::InvalidQubit;
        }
        c.gates.push(CircuitGate::H(qubit));
        NqpuError::Ok
    }
}

/// Append a CNOT gate to the circuit.
#[no_mangle]
pub extern "C" fn nqpu_circuit_add_cx(
    circuit: *mut NqpuCircuit,
    control: u32,
    target: u32,
) -> NqpuError {
    unsafe {
        let c = match circuit_mut(circuit) {
            Ok(c) => c,
            Err(e) => return e,
        };
        if control >= c.num_qubits || target >= c.num_qubits {
            return NqpuError::InvalidQubit;
        }
        if control == target {
            return NqpuError::InvalidParameter;
        }
        c.gates.push(CircuitGate::Cx(control, target));
        NqpuError::Ok
    }
}

/// Append an Rx rotation gate to the circuit.
#[no_mangle]
pub extern "C" fn nqpu_circuit_add_rx(
    circuit: *mut NqpuCircuit,
    qubit: u32,
    angle: f64,
) -> NqpuError {
    unsafe {
        let c = match circuit_mut(circuit) {
            Ok(c) => c,
            Err(e) => return e,
        };
        if qubit >= c.num_qubits {
            return NqpuError::InvalidQubit;
        }
        if !angle.is_finite() {
            return NqpuError::InvalidParameter;
        }
        c.gates.push(CircuitGate::Rx(qubit, angle));
        NqpuError::Ok
    }
}

/// Execute all gates in the circuit on the given state, in order.
///
/// The circuit's `num_qubits` must be <= the state's `num_qubits`.
#[no_mangle]
pub extern "C" fn nqpu_circuit_execute(
    circuit: *const NqpuCircuit,
    state: *mut NqpuState,
) -> NqpuError {
    unsafe {
        let c = match circuit_ref(circuit) {
            Ok(c) => c,
            Err(e) => return e,
        };
        let s = match state_mut(state) {
            Ok(s) => s,
            Err(e) => return e,
        };
        if c.num_qubits as usize > s.num_qubits {
            return NqpuError::InvalidParameter;
        }
        match c.execute(s) {
            Ok(()) => NqpuError::Ok,
            Err(e) => e,
        }
    }
}

/// Return the total number of gates in the circuit.
///
/// Returns 0 if the pointer is null.
#[no_mangle]
pub extern "C" fn nqpu_circuit_gate_count(circuit: *const NqpuCircuit) -> u32 {
    if circuit.is_null() {
        return 0;
    }
    unsafe {
        let c = &*(circuit as *const InternalCircuit);
        c.gate_count()
    }
}

/// Return the circuit depth (number of time steps under parallel execution).
///
/// Returns 0 if the pointer is null.
#[no_mangle]
pub extern "C" fn nqpu_circuit_depth(circuit: *const NqpuCircuit) -> u32 {
    if circuit.is_null() {
        return 0;
    }
    unsafe {
        let c = &*(circuit as *const InternalCircuit);
        c.depth()
    }
}

// ===================================================================
// UTILITY — extern "C" API
// ===================================================================

/// Return the library version.
#[no_mangle]
pub extern "C" fn nqpu_version() -> NqpuVersion {
    NqpuVersion {
        major: 0,
        minor: 1,
        patch: 0,
    }
}

// Static error message strings (null-terminated, stored in read-only data)
static ERROR_OK: &[u8] = b"Success\0";
static ERROR_INVALID_QUBIT: &[u8] = b"Invalid qubit index\0";
static ERROR_INVALID_PARAMETER: &[u8] = b"Invalid parameter\0";
static ERROR_ALLOCATION_FAILED: &[u8] = b"Memory allocation failed\0";
static ERROR_SIMULATION_FAILED: &[u8] = b"Simulation failed\0";
static ERROR_NULL_POINTER: &[u8] = b"Null pointer\0";
static ERROR_INVALID_HANDLE: &[u8] = b"Invalid handle\0";
static ERROR_UNKNOWN: &[u8] = b"Unknown error\0";

/// Return a human-readable error message string for the given error code.
///
/// The returned pointer is valid for the lifetime of the library (static
/// storage). The caller must not free or modify it.
#[no_mangle]
pub extern "C" fn nqpu_error_message(error: NqpuError) -> *const c_char {
    let msg = match error {
        NqpuError::Ok => ERROR_OK,
        NqpuError::InvalidQubit => ERROR_INVALID_QUBIT,
        NqpuError::InvalidParameter => ERROR_INVALID_PARAMETER,
        NqpuError::AllocationFailed => ERROR_ALLOCATION_FAILED,
        NqpuError::SimulationFailed => ERROR_SIMULATION_FAILED,
        NqpuError::NullPointer => ERROR_NULL_POINTER,
        NqpuError::InvalidHandle => ERROR_INVALID_HANDLE,
    };
    msg.as_ptr() as *const c_char
}

/// Set the thread-local RNG seed for reproducible measurements and sampling.
///
/// This affects only the calling thread. Different threads may use different
/// seeds for independent random streams.
#[no_mangle]
pub extern "C" fn nqpu_set_seed(seed: u64) {
    THREAD_RNG_SEED.with(|cell| {
        *cell.borrow_mut() = seed;
    });
    // Re-initialize the xorshift state from seed (avoid zero state)
    THREAD_RNG_STATE.with(|cell| {
        *cell.borrow_mut() = if seed == 0 {
            0x853c49e6748fea9b
        } else {
            seed
        };
    });
}

/// Set the number of threads for parallel simulation.
///
/// Configures the Rayon global thread pool used by parallel gate operations.
/// Must be called before any simulation; calling after pool initialization has no effect.
/// Pass 0 to use the default (number of logical CPUs).
#[no_mangle]
pub extern "C" fn nqpu_set_num_threads(threads: u32) {
    let num = if threads == 0 {
        rayon::current_num_threads()
    } else {
        threads as usize
    };
    // Rayon's global pool can only be initialized once; ignore errors if already set.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num)
        .build_global();
}

// ===================================================================
// C HEADER GENERATION
// ===================================================================

/// Generate a complete C header file for this FFI API.
///
/// The returned string can be written to `nqpu_metal.h` and included in
/// C/C++ projects.
pub fn generate_c_header() -> String {
    r#"/*
 * nqpu_metal.h — C API for nQPU-Metal Quantum Simulator
 *
 * Auto-generated. Do not edit.
 * Version: 1.0.0
 */

#ifndef NQPU_METAL_H
#define NQPU_METAL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

/** Opaque handle to a quantum state vector. */
typedef struct NqpuState NqpuState;

/** Opaque handle to a quantum circuit. */
typedef struct NqpuCircuit NqpuCircuit;

/** Error codes. */
typedef enum {
    NQPU_OK                = 0,
    NQPU_INVALID_QUBIT     = 1,
    NQPU_INVALID_PARAMETER = 2,
    NQPU_ALLOCATION_FAILED = 3,
    NQPU_SIMULATION_FAILED = 4,
    NQPU_NULL_POINTER      = 5,
    NQPU_INVALID_HANDLE    = 6
} NqpuError;

/** Double-precision complex number (binary-compatible with C99 _Complex double). */
typedef struct {
    double re;
    double im;
} NqpuComplex;

/** Measurement result. */
typedef struct {
    uint32_t outcome;
    double   probability;
} NqpuMeasurement;

/** Simulation configuration. */
typedef struct {
    uint32_t num_qubits;
    uint64_t seed;
    bool     use_gpu;
    uint32_t num_threads;
} NqpuConfig;

/** Semantic version. */
typedef struct {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} NqpuVersion;

/* ------------------------------------------------------------------ */
/* State Management                                                    */
/* ------------------------------------------------------------------ */

NqpuState*  nqpu_state_create(uint32_t num_qubits);
void        nqpu_state_destroy(NqpuState* state);
NqpuState*  nqpu_state_clone(const NqpuState* state);
uint32_t    nqpu_state_num_qubits(const NqpuState* state);
NqpuError   nqpu_state_get_amplitude(const NqpuState* state, uint64_t index, NqpuComplex* out);
NqpuError   nqpu_state_set_amplitude(NqpuState* state, uint64_t index, NqpuComplex amp);
NqpuError   nqpu_state_probabilities(const NqpuState* state, double* out, uint64_t len);
double      nqpu_state_norm(const NqpuState* state);
NqpuError   nqpu_state_reset(NqpuState* state);

/* ------------------------------------------------------------------ */
/* Gate Operations                                                     */
/* ------------------------------------------------------------------ */

NqpuError   nqpu_gate_h(NqpuState* state, uint32_t qubit);
NqpuError   nqpu_gate_x(NqpuState* state, uint32_t qubit);
NqpuError   nqpu_gate_y(NqpuState* state, uint32_t qubit);
NqpuError   nqpu_gate_z(NqpuState* state, uint32_t qubit);
NqpuError   nqpu_gate_s(NqpuState* state, uint32_t qubit);
NqpuError   nqpu_gate_t(NqpuState* state, uint32_t qubit);
NqpuError   nqpu_gate_rx(NqpuState* state, uint32_t qubit, double angle);
NqpuError   nqpu_gate_ry(NqpuState* state, uint32_t qubit, double angle);
NqpuError   nqpu_gate_rz(NqpuState* state, uint32_t qubit, double angle);
NqpuError   nqpu_gate_cx(NqpuState* state, uint32_t control, uint32_t target);
NqpuError   nqpu_gate_cz(NqpuState* state, uint32_t control, uint32_t target);
NqpuError   nqpu_gate_swap(NqpuState* state, uint32_t q1, uint32_t q2);
NqpuError   nqpu_gate_custom_1q(NqpuState* state, uint32_t qubit, const NqpuComplex* matrix);
NqpuError   nqpu_gate_custom_2q(NqpuState* state, uint32_t q1, uint32_t q2, const NqpuComplex* matrix);

/* ------------------------------------------------------------------ */
/* Measurement                                                         */
/* ------------------------------------------------------------------ */

NqpuError   nqpu_measure(NqpuState* state, uint32_t qubit, NqpuMeasurement* out);
NqpuError   nqpu_measure_all(NqpuState* state, uint32_t* out, uint32_t num_qubits);
NqpuError   nqpu_expectation_z(const NqpuState* state, uint32_t qubit, double* out);
NqpuError   nqpu_sample(const NqpuState* state, uint32_t num_shots, uint64_t* out);

/* ------------------------------------------------------------------ */
/* Circuit API                                                         */
/* ------------------------------------------------------------------ */

NqpuCircuit* nqpu_circuit_create(uint32_t num_qubits);
void         nqpu_circuit_destroy(NqpuCircuit* circuit);
NqpuError    nqpu_circuit_add_h(NqpuCircuit* circuit, uint32_t qubit);
NqpuError    nqpu_circuit_add_cx(NqpuCircuit* circuit, uint32_t control, uint32_t target);
NqpuError    nqpu_circuit_add_rx(NqpuCircuit* circuit, uint32_t qubit, double angle);
NqpuError    nqpu_circuit_execute(const NqpuCircuit* circuit, NqpuState* state);
uint32_t     nqpu_circuit_gate_count(const NqpuCircuit* circuit);
uint32_t     nqpu_circuit_depth(const NqpuCircuit* circuit);

/* ------------------------------------------------------------------ */
/* Utility                                                             */
/* ------------------------------------------------------------------ */

NqpuVersion  nqpu_version(void);
const char*  nqpu_error_message(NqpuError error);
void         nqpu_set_seed(uint64_t seed);
void         nqpu_set_num_threads(uint32_t threads);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NQPU_METAL_H */
"#
    .to_string()
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::ffi::CStr;

    /// Tolerance for floating-point comparisons in quantum simulation.
    const EPS: f64 = 1e-10;

    // ---------------------------------------------------------------
    // Helper: get amplitude from state handle (panics on error)
    // ---------------------------------------------------------------
    fn get_amp(state: *const NqpuState, index: u64) -> NqpuComplex {
        let mut out = NqpuComplex::zero();
        let err = nqpu_state_get_amplitude(state, index, &mut out);
        assert_eq!(err, NqpuError::Ok);
        out
    }

    // ---------------------------------------------------------------
    // 1. State creation and destruction
    // ---------------------------------------------------------------
    #[test]
    fn test_state_create_destroy() {
        let state = nqpu_state_create(3);
        assert!(!state.is_null());
        nqpu_state_destroy(state);
        // Null destruction is a no-op
        nqpu_state_destroy(std::ptr::null_mut());
    }

    // ---------------------------------------------------------------
    // 2. State: correct number of qubits
    // ---------------------------------------------------------------
    #[test]
    fn test_state_num_qubits() {
        let state = nqpu_state_create(5);
        assert_eq!(nqpu_state_num_qubits(state), 5);
        nqpu_state_destroy(state);

        // Null pointer returns 0
        assert_eq!(nqpu_state_num_qubits(std::ptr::null()), 0);
    }

    // ---------------------------------------------------------------
    // 3. State: initial amplitude |0...0> = 1
    // ---------------------------------------------------------------
    #[test]
    fn test_state_initial_amplitude() {
        let state = nqpu_state_create(3);
        let a0 = get_amp(state, 0);
        assert!((a0.re - 1.0).abs() < EPS);
        assert!(a0.im.abs() < EPS);

        // All other amplitudes should be zero
        for i in 1..8u64 {
            let a = get_amp(state, i);
            assert!(a.re.abs() < EPS, "amplitude[{}].re = {} != 0", i, a.re);
            assert!(a.im.abs() < EPS, "amplitude[{}].im = {} != 0", i, a.im);
        }
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 4. State: get/set amplitude round-trip
    // ---------------------------------------------------------------
    #[test]
    fn test_state_get_set_amplitude() {
        let state = nqpu_state_create(2);
        let val = NqpuComplex::new(0.6, 0.8);
        let err = nqpu_state_set_amplitude(state, 2, val);
        assert_eq!(err, NqpuError::Ok);

        let got = get_amp(state, 2);
        assert!((got.re - 0.6).abs() < EPS);
        assert!((got.im - 0.8).abs() < EPS);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 5. Gate H: |0> -> |+> = (|0> + |1>) / sqrt(2)
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_h_zero_to_plus() {
        let state = nqpu_state_create(1);
        let err = nqpu_gate_h(state, 0);
        assert_eq!(err, NqpuError::Ok);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let a0 = get_amp(state, 0);
        let a1 = get_amp(state, 1);

        assert!(
            (a0.re - inv_sqrt2).abs() < EPS,
            "H|0>: a0.re = {} != {}",
            a0.re,
            inv_sqrt2
        );
        assert!(a0.im.abs() < EPS);
        assert!(
            (a1.re - inv_sqrt2).abs() < EPS,
            "H|0>: a1.re = {} != {}",
            a1.re,
            inv_sqrt2
        );
        assert!(a1.im.abs() < EPS);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 6. Gate X: |0> -> |1>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_x_zero_to_one() {
        let state = nqpu_state_create(1);
        nqpu_gate_x(state, 0);

        let a0 = get_amp(state, 0);
        let a1 = get_amp(state, 1);
        assert!(a0.re.abs() < EPS);
        assert!((a1.re - 1.0).abs() < EPS);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 7. Gate CX: Bell state creation
    //    H(0) then CX(0,1) on |00> -> (|00> + |11>) / sqrt(2)
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_cx_bell_state() {
        let state = nqpu_state_create(2);
        nqpu_gate_h(state, 0);
        nqpu_gate_cx(state, 0, 1);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let a00 = get_amp(state, 0); // |00>
        let a01 = get_amp(state, 1); // |01>
        let a10 = get_amp(state, 2); // |10>
        let a11 = get_amp(state, 3); // |11>

        assert!(
            (a00.re - inv_sqrt2).abs() < EPS,
            "Bell: a00 = {:?}",
            a00
        );
        assert!(a01.norm_sqr() < EPS, "Bell: a01 = {:?}", a01);
        assert!(a10.norm_sqr() < EPS, "Bell: a10 = {:?}", a10);
        assert!(
            (a11.re - inv_sqrt2).abs() < EPS,
            "Bell: a11 = {:?}",
            a11
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 8. Gate Rx: rotation by pi = X (up to global phase)
    //    Rx(pi) = [[0, -i], [-i, 0]] which maps |0> -> -i|1>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_rx_pi_equals_x() {
        let state = nqpu_state_create(1);
        nqpu_gate_rx(state, 0, PI);

        let a0 = get_amp(state, 0);
        let a1 = get_amp(state, 1);

        // |0> should be ~0
        assert!(a0.norm_sqr() < EPS, "Rx(pi)|0>: a0 = {:?}", a0);
        // |1> should have magnitude 1
        assert!(
            (a1.norm_sqr() - 1.0).abs() < EPS,
            "Rx(pi)|0>: |a1|^2 = {}",
            a1.norm_sqr()
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 9. Gate Ry: rotation by pi maps |0> -> |1>
    //    Ry(pi) = [[0, -1], [1, 0]], |0> -> |1>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_ry_pi() {
        let state = nqpu_state_create(1);
        nqpu_gate_ry(state, 0, PI);

        let a0 = get_amp(state, 0);
        let a1 = get_amp(state, 1);

        assert!(a0.norm_sqr() < EPS, "Ry(pi)|0>: a0 = {:?}", a0);
        assert!(
            (a1.norm_sqr() - 1.0).abs() < EPS,
            "Ry(pi)|0>: |a1|^2 = {}",
            a1.norm_sqr()
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 10. Gate Rz: rotation by pi/2 = S (up to global phase)
    //     Rz(pi/2)|1> = e^{i*pi/4}|1>, which is the T-like phase.
    //     Actually Rz(pi/2) = diag(e^{-i*pi/4}, e^{i*pi/4}).
    //     S = diag(1, i) = e^{i*pi/4} * Rz(pi/2) (global phase).
    //     Check: Rz(pi/2)|1> has phase e^{i*pi/4} = (1+i)/sqrt(2).
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_rz_pi_over_2() {
        let state = nqpu_state_create(1);
        // Prepare |1>
        nqpu_gate_x(state, 0);
        // Apply Rz(pi/2)
        nqpu_gate_rz(state, 0, PI / 2.0);

        let a1 = get_amp(state, 1);
        // Expected: e^{i*pi/4} = cos(pi/4) + i*sin(pi/4)
        let expected_re = (PI / 4.0).cos();
        let expected_im = (PI / 4.0).sin();
        assert!(
            (a1.re - expected_re).abs() < EPS,
            "Rz(pi/2)|1>: re = {} != {}",
            a1.re,
            expected_re
        );
        assert!(
            (a1.im - expected_im).abs() < EPS,
            "Rz(pi/2)|1>: im = {} != {}",
            a1.im,
            expected_im
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 11. Measurement: |0> always gives 0
    // ---------------------------------------------------------------
    #[test]
    fn test_measure_zero_always_zero() {
        for _ in 0..20 {
            let state = nqpu_state_create(1);
            let mut result = NqpuMeasurement {
                outcome: 99,
                probability: 0.0,
            };
            let err = nqpu_measure(state, 0, &mut result);
            assert_eq!(err, NqpuError::Ok);
            assert_eq!(result.outcome, 0);
            assert!((result.probability - 1.0).abs() < EPS);
            nqpu_state_destroy(state);
        }
    }

    // ---------------------------------------------------------------
    // 12. Measurement: |1> always gives 1
    // ---------------------------------------------------------------
    #[test]
    fn test_measure_one_always_one() {
        for _ in 0..20 {
            let state = nqpu_state_create(1);
            nqpu_gate_x(state, 0);
            let mut result = NqpuMeasurement {
                outcome: 99,
                probability: 0.0,
            };
            nqpu_measure(state, 0, &mut result);
            assert_eq!(result.outcome, 1);
            assert!((result.probability - 1.0).abs() < EPS);
            nqpu_state_destroy(state);
        }
    }

    // ---------------------------------------------------------------
    // 13. Expectation Z: |0> = +1
    // ---------------------------------------------------------------
    #[test]
    fn test_expectation_z_zero() {
        let state = nqpu_state_create(1);
        let mut exp = 0.0;
        let err = nqpu_expectation_z(state, 0, &mut exp);
        assert_eq!(err, NqpuError::Ok);
        assert!((exp - 1.0).abs() < EPS, "<0|Z|0> = {} != 1.0", exp);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 14. Expectation Z: |1> = -1
    // ---------------------------------------------------------------
    #[test]
    fn test_expectation_z_one() {
        let state = nqpu_state_create(1);
        nqpu_gate_x(state, 0);
        let mut exp = 0.0;
        nqpu_expectation_z(state, 0, &mut exp);
        assert!((exp + 1.0).abs() < EPS, "<1|Z|1> = {} != -1.0", exp);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 15. Probabilities: sum to 1
    // ---------------------------------------------------------------
    #[test]
    fn test_probabilities_sum_to_one() {
        let state = nqpu_state_create(3);
        // Create a non-trivial state
        nqpu_gate_h(state, 0);
        nqpu_gate_h(state, 1);
        nqpu_gate_cx(state, 0, 2);

        let dim = 8u64;
        let mut probs = vec![0.0f64; dim as usize];
        let err = nqpu_state_probabilities(state, probs.as_mut_ptr(), dim);
        assert_eq!(err, NqpuError::Ok);

        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < EPS,
            "Probability sum = {} != 1.0",
            sum
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 16. State clone: independent copy
    // ---------------------------------------------------------------
    #[test]
    fn test_state_clone_independent() {
        let state = nqpu_state_create(2);
        nqpu_gate_h(state, 0);

        let cloned = nqpu_state_clone(state);
        assert!(!cloned.is_null());

        // Modify original
        nqpu_gate_x(state, 1);

        // Clone should be unaffected
        let orig_a = get_amp(state, 0);
        let clone_a = get_amp(cloned, 0);

        // After X(1) on original, amplitudes differ
        let orig_a2 = get_amp(state, 2); // |10> on original
        let clone_a2 = get_amp(cloned, 2); // |10> on clone

        // Original had H(0), then X(1) moves |00>-component to |10> and |01> to |11>
        // Clone only has H(0)
        // Clone's |10> amplitude should still be 0
        assert!(clone_a2.norm_sqr() < EPS);
        // Original's |10> should be nonzero
        assert!(orig_a2.norm_sqr() > EPS);

        nqpu_state_destroy(state);
        nqpu_state_destroy(cloned);
    }

    // ---------------------------------------------------------------
    // 17. State reset: back to |0...0>
    // ---------------------------------------------------------------
    #[test]
    fn test_state_reset() {
        let state = nqpu_state_create(2);
        nqpu_gate_h(state, 0);
        nqpu_gate_h(state, 1);

        let err = nqpu_state_reset(state);
        assert_eq!(err, NqpuError::Ok);

        let a0 = get_amp(state, 0);
        assert!((a0.re - 1.0).abs() < EPS);
        for i in 1..4u64 {
            let a = get_amp(state, i);
            assert!(a.norm_sqr() < EPS);
        }
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 18. Circuit: create and add gates
    // ---------------------------------------------------------------
    #[test]
    fn test_circuit_create_add_gates() {
        let circuit = nqpu_circuit_create(3);
        assert!(!circuit.is_null());

        assert_eq!(nqpu_circuit_add_h(circuit, 0), NqpuError::Ok);
        assert_eq!(nqpu_circuit_add_cx(circuit, 0, 1), NqpuError::Ok);
        assert_eq!(nqpu_circuit_add_rx(circuit, 2, PI / 4.0), NqpuError::Ok);

        assert_eq!(nqpu_circuit_gate_count(circuit), 3);
        nqpu_circuit_destroy(circuit);
    }

    // ---------------------------------------------------------------
    // 19. Circuit: execute on state
    // ---------------------------------------------------------------
    #[test]
    fn test_circuit_execute() {
        let circuit = nqpu_circuit_create(2);
        nqpu_circuit_add_h(circuit, 0);
        nqpu_circuit_add_cx(circuit, 0, 1);

        let state = nqpu_state_create(2);
        let err = nqpu_circuit_execute(circuit, state);
        assert_eq!(err, NqpuError::Ok);

        // Should produce Bell state
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let a00 = get_amp(state, 0);
        let a11 = get_amp(state, 3);
        assert!((a00.re - inv_sqrt2).abs() < EPS);
        assert!((a11.re - inv_sqrt2).abs() < EPS);

        nqpu_circuit_destroy(circuit);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 20. Circuit: gate count
    // ---------------------------------------------------------------
    #[test]
    fn test_circuit_gate_count() {
        let circuit = nqpu_circuit_create(4);
        assert_eq!(nqpu_circuit_gate_count(circuit), 0);

        for i in 0..4 {
            nqpu_circuit_add_h(circuit, i);
        }
        assert_eq!(nqpu_circuit_gate_count(circuit), 4);

        nqpu_circuit_add_cx(circuit, 0, 1);
        nqpu_circuit_add_cx(circuit, 2, 3);
        assert_eq!(nqpu_circuit_gate_count(circuit), 6);

        nqpu_circuit_destroy(circuit);
    }

    // ---------------------------------------------------------------
    // 21. Circuit: depth calculation
    //     H(0) H(1) CX(0,1) -> depth 2
    //     Layer 0: H(0), H(1) in parallel
    //     Layer 1: CX(0,1)
    // ---------------------------------------------------------------
    #[test]
    fn test_circuit_depth() {
        let circuit = nqpu_circuit_create(2);
        nqpu_circuit_add_h(circuit, 0);
        nqpu_circuit_add_h(circuit, 1);
        nqpu_circuit_add_cx(circuit, 0, 1);

        assert_eq!(nqpu_circuit_depth(circuit), 2);
        nqpu_circuit_destroy(circuit);
    }

    // ---------------------------------------------------------------
    // 22. Null pointer: returns NullPointer error
    // ---------------------------------------------------------------
    #[test]
    fn test_null_pointer_errors() {
        let null_state: *mut NqpuState = std::ptr::null_mut();
        let null_circuit: *mut NqpuCircuit = std::ptr::null_mut();

        assert_eq!(nqpu_gate_h(null_state, 0), NqpuError::NullPointer);
        assert_eq!(nqpu_gate_x(null_state, 0), NqpuError::NullPointer);
        assert_eq!(nqpu_gate_cx(null_state, 0, 1), NqpuError::NullPointer);

        let mut out = NqpuComplex::zero();
        assert_eq!(
            nqpu_state_get_amplitude(std::ptr::null(), 0, &mut out),
            NqpuError::NullPointer
        );
        assert_eq!(
            nqpu_state_set_amplitude(null_state, 0, NqpuComplex::one()),
            NqpuError::NullPointer
        );
        assert_eq!(nqpu_state_reset(null_state), NqpuError::NullPointer);

        let mut meas = NqpuMeasurement {
            outcome: 0,
            probability: 0.0,
        };
        assert_eq!(
            nqpu_measure(null_state, 0, &mut meas),
            NqpuError::NullPointer
        );

        assert_eq!(
            nqpu_circuit_add_h(null_circuit, 0),
            NqpuError::NullPointer
        );
        assert_eq!(
            nqpu_circuit_execute(std::ptr::null(), null_state),
            NqpuError::NullPointer
        );
    }

    // ---------------------------------------------------------------
    // 23. Invalid qubit: returns InvalidQubit error
    // ---------------------------------------------------------------
    #[test]
    fn test_invalid_qubit_errors() {
        let state = nqpu_state_create(2);

        assert_eq!(nqpu_gate_h(state, 5), NqpuError::InvalidQubit);
        assert_eq!(nqpu_gate_x(state, 2), NqpuError::InvalidQubit);
        assert_eq!(nqpu_gate_rx(state, 3, 0.0), NqpuError::InvalidQubit);
        assert_eq!(nqpu_gate_cx(state, 0, 5), NqpuError::InvalidQubit);
        assert_eq!(nqpu_gate_cx(state, 5, 0), NqpuError::InvalidQubit);

        let mut exp = 0.0;
        assert_eq!(
            nqpu_expectation_z(state, 10, &mut exp),
            NqpuError::InvalidQubit
        );

        nqpu_state_destroy(state);

        // Circuit also validates
        let circuit = nqpu_circuit_create(2);
        assert_eq!(nqpu_circuit_add_h(circuit, 3), NqpuError::InvalidQubit);
        assert_eq!(
            nqpu_circuit_add_cx(circuit, 0, 5),
            NqpuError::InvalidQubit
        );
        nqpu_circuit_destroy(circuit);
    }

    // ---------------------------------------------------------------
    // 24. Custom 1Q gate: identity matrix
    // ---------------------------------------------------------------
    #[test]
    fn test_custom_1q_identity() {
        let state = nqpu_state_create(1);
        nqpu_gate_h(state, 0);

        // Save amplitudes before
        let before_0 = get_amp(state, 0);
        let before_1 = get_amp(state, 1);

        // Identity matrix
        let identity = [
            NqpuComplex::new(1.0, 0.0),
            NqpuComplex::zero(),
            NqpuComplex::zero(),
            NqpuComplex::new(1.0, 0.0),
        ];
        let err = nqpu_gate_custom_1q(state, 0, identity.as_ptr());
        assert_eq!(err, NqpuError::Ok);

        let after_0 = get_amp(state, 0);
        let after_1 = get_amp(state, 1);

        assert!((after_0.re - before_0.re).abs() < EPS);
        assert!((after_0.im - before_0.im).abs() < EPS);
        assert!((after_1.re - before_1.re).abs() < EPS);
        assert!((after_1.im - before_1.im).abs() < EPS);

        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 25. Custom 2Q gate: SWAP matrix
    // ---------------------------------------------------------------
    #[test]
    fn test_custom_2q_swap() {
        let state = nqpu_state_create(2);
        // Prepare |10> (X on qubit 1)
        nqpu_gate_x(state, 1);

        // Verify we are in |10> (index 2)
        assert!((get_amp(state, 2).re - 1.0).abs() < EPS);

        // SWAP matrix (4x4 identity with rows/cols 1,2 swapped)
        // Basis: |00>, |01>, |10>, |11>
        // SWAP|10> = |01>
        let z = NqpuComplex::zero();
        let o = NqpuComplex::one();
        #[rustfmt::skip]
        let swap_matrix = [
            o, z, z, z,  // |00> -> |00>
            z, z, o, z,  // |01> -> |10>
            z, o, z, z,  // |10> -> |01>
            z, z, z, o,  // |11> -> |11>
        ];

        let err = nqpu_gate_custom_2q(state, 1, 0, swap_matrix.as_ptr());
        assert_eq!(err, NqpuError::Ok);

        // After SWAP: |10> -> |01> (index 1)
        assert!(
            (get_amp(state, 1).re - 1.0).abs() < EPS,
            "SWAP |10> should give |01>, got amp[1] = {:?}",
            get_amp(state, 1)
        );
        assert!(get_amp(state, 2).norm_sqr() < EPS);

        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 26. Version: returns valid version
    // ---------------------------------------------------------------
    #[test]
    fn test_version() {
        let v = nqpu_version();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
    }

    // ---------------------------------------------------------------
    // 27. Error message: non-null string
    // ---------------------------------------------------------------
    #[test]
    fn test_error_message() {
        let errors = [
            NqpuError::Ok,
            NqpuError::InvalidQubit,
            NqpuError::InvalidParameter,
            NqpuError::AllocationFailed,
            NqpuError::SimulationFailed,
            NqpuError::NullPointer,
            NqpuError::InvalidHandle,
        ];
        for err in &errors {
            let msg = nqpu_error_message(*err);
            assert!(!msg.is_null());
            let s = unsafe { CStr::from_ptr(msg) };
            assert!(!s.to_str().unwrap().is_empty());
        }
    }

    // ---------------------------------------------------------------
    // 28. Sample: 100 shots of Bell state
    // ---------------------------------------------------------------
    #[test]
    fn test_sample_bell_state() {
        nqpu_set_seed(42);

        let state = nqpu_state_create(2);
        nqpu_gate_h(state, 0);
        nqpu_gate_cx(state, 0, 1);

        let num_shots = 100u32;
        let mut results = vec![0u64; num_shots as usize];
        let err = nqpu_sample(state, num_shots, results.as_mut_ptr());
        assert_eq!(err, NqpuError::Ok);

        // Bell state: only |00> (0) and |11> (3) should appear
        let mut count_00 = 0u32;
        let mut count_11 = 0u32;
        for &r in &results {
            match r {
                0 => count_00 += 1,
                3 => count_11 += 1,
                other => panic!("Unexpected outcome {} in Bell state sampling", other),
            }
        }
        assert_eq!(count_00 + count_11, num_shots);
        // With 100 shots, both should appear (extremely unlikely to get all one outcome)
        assert!(count_00 > 0, "Expected some |00> outcomes");
        assert!(count_11 > 0, "Expected some |11> outcomes");

        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 29. State norm is 1.0 after gates
    // ---------------------------------------------------------------
    #[test]
    fn test_state_norm_preserved() {
        let state = nqpu_state_create(3);
        nqpu_gate_h(state, 0);
        nqpu_gate_cx(state, 0, 1);
        nqpu_gate_ry(state, 2, 1.234);
        nqpu_gate_rz(state, 0, 0.567);
        nqpu_gate_t(state, 1);
        nqpu_gate_s(state, 2);

        let norm = nqpu_state_norm(state);
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "Norm after gates = {} != 1.0",
            norm
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 30. Gate Y: |0> -> i|1>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_y() {
        let state = nqpu_state_create(1);
        nqpu_gate_y(state, 0);

        let a0 = get_amp(state, 0);
        let a1 = get_amp(state, 1);

        assert!(a0.norm_sqr() < EPS, "Y|0>: a0 = {:?}", a0);
        // Y|0> = i|1>
        assert!(a1.re.abs() < EPS, "Y|0>: a1.re = {}", a1.re);
        assert!(
            (a1.im - 1.0).abs() < EPS,
            "Y|0>: a1.im = {} != 1.0",
            a1.im
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 31. Gate Z: |1> -> -|1>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_z() {
        let state = nqpu_state_create(1);
        nqpu_gate_x(state, 0); // |1>
        nqpu_gate_z(state, 0); // -|1>

        let a1 = get_amp(state, 1);
        assert!(
            (a1.re + 1.0).abs() < EPS,
            "Z|1>: a1.re = {} != -1.0",
            a1.re
        );
        assert!(a1.im.abs() < EPS);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 32. Gate S: |1> -> i|1>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_s() {
        let state = nqpu_state_create(1);
        nqpu_gate_x(state, 0); // |1>
        nqpu_gate_s(state, 0); // i|1>

        let a1 = get_amp(state, 1);
        assert!(a1.re.abs() < EPS, "S|1>: re = {}", a1.re);
        assert!((a1.im - 1.0).abs() < EPS, "S|1>: im = {} != 1.0", a1.im);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 33. Gate T: |1> -> e^{i*pi/4}|1>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_t() {
        let state = nqpu_state_create(1);
        nqpu_gate_x(state, 0);
        nqpu_gate_t(state, 0);

        let a1 = get_amp(state, 1);
        let expected_re = (PI / 4.0).cos();
        let expected_im = (PI / 4.0).sin();
        assert!(
            (a1.re - expected_re).abs() < EPS,
            "T|1>: re = {} != {}",
            a1.re,
            expected_re
        );
        assert!(
            (a1.im - expected_im).abs() < EPS,
            "T|1>: im = {} != {}",
            a1.im,
            expected_im
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 34. SWAP gate: |01> -> |10>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_swap() {
        let state = nqpu_state_create(2);
        nqpu_gate_x(state, 0); // |01> (qubit 0 is 1)

        assert!((get_amp(state, 1).re - 1.0).abs() < EPS); // |01> = index 1

        nqpu_gate_swap(state, 0, 1);

        // After SWAP: |01> -> |10> = index 2
        assert!(
            (get_amp(state, 2).re - 1.0).abs() < EPS,
            "SWAP|01> should give |10>, amp[2] = {:?}",
            get_amp(state, 2)
        );
        assert!(get_amp(state, 1).norm_sqr() < EPS);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 35. CZ gate: |11> -> -|11>
    // ---------------------------------------------------------------
    #[test]
    fn test_gate_cz() {
        let state = nqpu_state_create(2);
        nqpu_gate_x(state, 0);
        nqpu_gate_x(state, 1);
        // Now in |11> = index 3

        nqpu_gate_cz(state, 0, 1);

        let a11 = get_amp(state, 3);
        assert!(
            (a11.re + 1.0).abs() < EPS,
            "CZ|11>: a11.re = {} != -1.0",
            a11.re
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 36. Invalid parameter: same qubit for CX
    // ---------------------------------------------------------------
    #[test]
    fn test_same_qubit_cx_error() {
        let state = nqpu_state_create(2);
        assert_eq!(nqpu_gate_cx(state, 0, 0), NqpuError::InvalidParameter);
        assert_eq!(nqpu_gate_cz(state, 1, 1), NqpuError::InvalidParameter);
        assert_eq!(nqpu_gate_swap(state, 0, 0), NqpuError::InvalidParameter);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 37. State creation: invalid qubit counts
    // ---------------------------------------------------------------
    #[test]
    fn test_state_create_invalid() {
        assert!(nqpu_state_create(0).is_null());
        assert!(nqpu_state_create(31).is_null());
        assert!(nqpu_state_create(100).is_null());
    }

    // ---------------------------------------------------------------
    // 38. H gate is self-inverse: H*H = I
    // ---------------------------------------------------------------
    #[test]
    fn test_h_self_inverse() {
        let state = nqpu_state_create(1);
        nqpu_gate_h(state, 0);
        nqpu_gate_h(state, 0);

        let a0 = get_amp(state, 0);
        let a1 = get_amp(state, 1);
        assert!((a0.re - 1.0).abs() < EPS, "H^2|0>: a0 = {:?}", a0);
        assert!(a1.norm_sqr() < EPS, "H^2|0>: a1 = {:?}", a1);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 39. X gate is self-inverse: X*X = I
    // ---------------------------------------------------------------
    #[test]
    fn test_x_self_inverse() {
        let state = nqpu_state_create(1);
        nqpu_gate_x(state, 0);
        nqpu_gate_x(state, 0);

        let a0 = get_amp(state, 0);
        assert!((a0.re - 1.0).abs() < EPS);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 40. Probabilities buffer too small
    // ---------------------------------------------------------------
    #[test]
    fn test_probabilities_buffer_too_small() {
        let state = nqpu_state_create(3);
        let mut probs = [0.0f64; 4]; // Need 8 for 3 qubits
        let err = nqpu_state_probabilities(state, probs.as_mut_ptr(), 4);
        assert_eq!(err, NqpuError::InvalidParameter);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 41. Set seed produces deterministic results
    // ---------------------------------------------------------------
    #[test]
    fn test_seed_determinism() {
        let run = |seed: u64| -> Vec<u64> {
            nqpu_set_seed(seed);
            let state = nqpu_state_create(2);
            nqpu_gate_h(state, 0);
            nqpu_gate_h(state, 1);

            let mut results = vec![0u64; 50];
            nqpu_sample(state, 50, results.as_mut_ptr());
            nqpu_state_destroy(state);
            results
        };

        let r1 = run(12345);
        let r2 = run(12345);
        assert_eq!(r1, r2, "Same seed should produce same samples");

        let r3 = run(99999);
        // Different seeds should (very likely) produce different results
        assert_ne!(r1, r3, "Different seeds should produce different samples");
    }

    // ---------------------------------------------------------------
    // 42. C header generation produces valid content
    // ---------------------------------------------------------------
    #[test]
    fn test_c_header_generation() {
        let header = generate_c_header();
        assert!(header.contains("#ifndef NQPU_METAL_H"));
        assert!(header.contains("#define NQPU_METAL_H"));
        assert!(header.contains("nqpu_state_create"));
        assert!(header.contains("nqpu_gate_h"));
        assert!(header.contains("nqpu_measure"));
        assert!(header.contains("nqpu_circuit_create"));
        assert!(header.contains("NqpuComplex"));
        assert!(header.contains("NqpuError"));
        assert!(header.contains("#endif"));
    }

    // ---------------------------------------------------------------
    // 43. Multi-qubit state: H on qubit 1 of 3-qubit system
    // ---------------------------------------------------------------
    #[test]
    fn test_h_on_middle_qubit() {
        let state = nqpu_state_create(3);
        // |000> -> H on qubit 1 -> (|000> + |010>) / sqrt(2)
        nqpu_gate_h(state, 1);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        // |000> = index 0
        assert!((get_amp(state, 0).re - inv_sqrt2).abs() < EPS);
        // |010> = index 2
        assert!((get_amp(state, 2).re - inv_sqrt2).abs() < EPS);
        // All others zero
        for i in [1u64, 3, 4, 5, 6, 7] {
            assert!(get_amp(state, i).norm_sqr() < EPS, "amp[{}] should be 0", i);
        }
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 44. Circuit depth: parallel H gates
    // ---------------------------------------------------------------
    #[test]
    fn test_circuit_depth_parallel() {
        // 4 H gates on 4 different qubits = depth 1
        let circuit = nqpu_circuit_create(4);
        for i in 0..4 {
            nqpu_circuit_add_h(circuit, i);
        }
        // All act on different qubits, so depth = 1... but they are sequential
        // in the gate list. The depth algorithm assigns them to layers greedily:
        // H(0) -> layer 0, H(1) -> layer 0, H(2) -> layer 0, H(3) -> layer 0
        // Actually: qubit_depth starts at 0 for all. H(0) -> qubit_depth[0]=1, layer=1.
        // Wait, let me re-check: layer = max of qubit depths = 0, then set to 0+1=1.
        // H(1): layer = qubit_depth[1] = 0, set qubit_depth[1] = 1. max_depth = 1.
        // So all 4 H gates go in layer 0 (the first layer), depth = 1.
        assert_eq!(nqpu_circuit_depth(circuit), 1);
        nqpu_circuit_destroy(circuit);
    }

    // ---------------------------------------------------------------
    // 45. Norm is -1.0 for null state
    // ---------------------------------------------------------------
    #[test]
    fn test_norm_null_state() {
        let norm = nqpu_state_norm(std::ptr::null());
        assert!((norm - (-1.0)).abs() < EPS);
    }

    // ---------------------------------------------------------------
    // 46. Get amplitude out-of-bounds
    // ---------------------------------------------------------------
    #[test]
    fn test_get_amplitude_out_of_bounds() {
        let state = nqpu_state_create(2); // dim = 4
        let mut out = NqpuComplex::zero();
        let err = nqpu_state_get_amplitude(state, 4, &mut out);
        assert_eq!(err, NqpuError::InvalidParameter);
        let err2 = nqpu_state_get_amplitude(state, 100, &mut out);
        assert_eq!(err2, NqpuError::InvalidParameter);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 47. NaN angle rejected
    // ---------------------------------------------------------------
    #[test]
    fn test_nan_angle_rejected() {
        let state = nqpu_state_create(2);
        assert_eq!(nqpu_gate_rx(state, 0, f64::NAN), NqpuError::InvalidParameter);
        assert_eq!(nqpu_gate_ry(state, 0, f64::INFINITY), NqpuError::InvalidParameter);
        assert_eq!(
            nqpu_gate_rz(state, 0, f64::NEG_INFINITY),
            NqpuError::InvalidParameter
        );
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 48. Measure all on 1-qubit |1> state
    // ---------------------------------------------------------------
    #[test]
    fn test_measure_all_one_qubit() {
        let state = nqpu_state_create(1);
        nqpu_gate_x(state, 0);

        let mut outcomes = [99u32; 1];
        let err = nqpu_measure_all(state, outcomes.as_mut_ptr(), 1);
        assert_eq!(err, NqpuError::Ok);
        assert_eq!(outcomes[0], 1);
        nqpu_state_destroy(state);
    }

    // ---------------------------------------------------------------
    // 49. Circuit: empty circuit has depth 0
    // ---------------------------------------------------------------
    #[test]
    fn test_empty_circuit_depth() {
        let circuit = nqpu_circuit_create(3);
        assert_eq!(nqpu_circuit_depth(circuit), 0);
        assert_eq!(nqpu_circuit_gate_count(circuit), 0);
        nqpu_circuit_destroy(circuit);
    }

    // ---------------------------------------------------------------
    // 50. GHZ state preparation
    //     H(0), CX(0,1), CX(1,2) -> (|000> + |111>) / sqrt(2)
    // ---------------------------------------------------------------
    #[test]
    fn test_ghz_state() {
        let state = nqpu_state_create(3);
        nqpu_gate_h(state, 0);
        nqpu_gate_cx(state, 0, 1);
        nqpu_gate_cx(state, 1, 2);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        // |000> = 0, |111> = 7
        assert!((get_amp(state, 0).re - inv_sqrt2).abs() < EPS);
        assert!((get_amp(state, 7).re - inv_sqrt2).abs() < EPS);

        // All others zero
        for i in [1u64, 2, 3, 4, 5, 6] {
            assert!(
                get_amp(state, i).norm_sqr() < EPS,
                "GHZ: amp[{}] = {:?} should be 0",
                i,
                get_amp(state, i)
            );
        }
        nqpu_state_destroy(state);
    }
}
