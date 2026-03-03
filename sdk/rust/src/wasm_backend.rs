//! WASM-Compatible Quantum Simulation Backend
//!
//! Provides a fully self-contained quantum simulator that compiles to WebAssembly
//! without Metal, CUDA, rayon, or any platform-specific dependencies. All operations
//! are single-threaded and use only core Rust primitives.
//!
//! # Architecture
//!
//! - [`WasmQuantumState`]: State vector representation (amplitudes as `Vec<C64>`)
//! - [`WasmGateEngine`]: Stateless gate application engine (single/controlled/circuit)
//! - [`WasmSimulator`]: High-level simulator with measurement and JS interop
//! - [`WasmCircuitBuilder`]: Builder pattern for constructing gate sequences
//! - [`XorShift64`]: Deterministic PRNG for measurement (no `rand` dependency)
//!
//! # Usage
//!
//! ```rust,ignore
//! use nqpu_metal::wasm_backend::{WasmSimulator, WasmCircuitBuilder};
//!
//! let mut sim = WasmSimulator::new(2);
//! sim.h(0);
//! sim.cnot(0, 1);
//! let probs = sim.probabilities();
//! // probs ~= [0.5, 0.0, 0.0, 0.5] (Bell state)
//! ```

#![cfg(feature = "wasm")]

use crate::gates::{Gate, GateType};
use crate::C64;

use std::f64::consts::{FRAC_1_SQRT_2, PI};

// ============================================================
// LOCAL HELPERS
// ============================================================

/// Zero complex amplitude.
#[inline]
fn zero() -> C64 {
    C64::new(0.0, 0.0)
}

/// Unity complex amplitude.
#[inline]
fn one() -> C64 {
    C64::new(1.0, 0.0)
}

// ============================================================
// XORSHIFT64 PRNG
// ============================================================

/// Minimal xorshift64 pseudo-random number generator.
///
/// Provides deterministic randomness without requiring the `rand` crate,
/// which has WASM compatibility issues in certain configurations. The
/// algorithm is George Marsaglia's xorshift64 (2003).
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new PRNG with the given seed.
    ///
    /// # Panics
    ///
    /// Panics if `seed` is zero (xorshift requires a nonzero seed).
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "XorShift64 seed must be nonzero");
        Self { state: seed }
    }

    /// Create a PRNG seeded from a combination of state properties.
    ///
    /// Useful when no external entropy source is available. Combines the
    /// qubit count and dimension to produce a nonzero seed.
    pub fn from_state_properties(num_qubits: usize, dim: usize) -> Self {
        let seed = (num_qubits as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(dim as u64)
            .wrapping_add(1); // guarantee nonzero
                              // Ensure nonzero after wrapping
        let seed = if seed == 0 { 1 } else { seed };
        Self { state: seed }
    }

    /// Advance the state and return the next raw u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a uniformly distributed f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for full double-precision mantissa coverage.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ============================================================
// WASM QUANTUM STATE
// ============================================================

/// Quantum state vector for WASM targets.
///
/// Stores `2^n` complex amplitudes in a contiguous `Vec<C64>`. No GPU
/// buffers, no threading primitives -- pure single-threaded computation
/// suitable for WebAssembly execution.
pub struct WasmQuantumState {
    /// Complex amplitudes of the quantum state. Length is always `dim`.
    pub amplitudes: Vec<C64>,
    /// Number of qubits in the register.
    pub num_qubits: usize,
    /// State-space dimension (`2^num_qubits`).
    pub dim: usize,
}

impl WasmQuantumState {
    /// Create a new quantum state initialized to |0...0>.
    ///
    /// The state vector has `2^num_qubits` entries, with only the first
    /// amplitude set to 1.0 (computational basis state zero).
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut amplitudes = vec![zero(); dim];
        amplitudes[0] = one();
        Self {
            amplitudes,
            num_qubits,
            dim,
        }
    }

    /// Read the amplitude at basis state `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= dim`.
    #[inline]
    pub fn get(&self, idx: usize) -> C64 {
        self.amplitudes[idx]
    }

    /// Write the amplitude at basis state `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= dim`.
    #[inline]
    pub fn set(&mut self, idx: usize, val: C64) {
        self.amplitudes[idx] = val;
    }

    /// Compute the probability distribution over computational basis states.
    ///
    /// Returns a vector of length `dim` where `probs[i] = |amplitudes[i]|^2`.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Normalize the state vector to unit length.
    ///
    /// After normalization, the sum of squared magnitudes equals 1.0.
    /// If the state is the zero vector (norm < 1e-30), no normalization
    /// is performed to avoid division by zero.
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq < 1e-30 {
            return;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        for a in &mut self.amplitudes {
            *a = C64::new(a.re * inv_norm, a.im * inv_norm);
        }
    }

    /// Compute the state fidelity with another quantum state.
    ///
    /// Fidelity is defined as `|<self|other>|^2`. Returns a value in [0, 1].
    /// Both states must have the same number of qubits.
    ///
    /// # Panics
    ///
    /// Panics if the two states have different dimensions.
    pub fn fidelity(&self, other: &WasmQuantumState) -> f64 {
        assert_eq!(
            self.dim, other.dim,
            "Cannot compute fidelity between states of different dimensions"
        );
        let inner: C64 = self
            .amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        inner.norm_sqr()
    }

    /// Reset the state to |0...0>.
    pub fn reset(&mut self) {
        for a in &mut self.amplitudes {
            *a = zero();
        }
        self.amplitudes[0] = one();
    }

    /// Return the number of qubits.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Return the state-space dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Clone for WasmQuantumState {
    fn clone(&self) -> Self {
        Self {
            amplitudes: self.amplitudes.clone(),
            num_qubits: self.num_qubits,
            dim: self.dim,
        }
    }
}

// ============================================================
// WASM GATE ENGINE
// ============================================================

/// Stateless engine for applying quantum gates to a `WasmQuantumState`.
///
/// All methods are purely functional transformations of the state vector.
/// No internal state is maintained between calls. The engine supports
/// single-qubit gates, controlled gates, and full circuit dispatch.
pub struct WasmGateEngine;

impl WasmGateEngine {
    /// Apply a single-qubit unitary to the target qubit.
    ///
    /// The `matrix` is a 2x2 unitary in row-major form. For each pair of
    /// basis states differing only in bit `target`, the amplitudes are
    /// updated by left-multiplying with the gate matrix.
    pub fn apply_single_qubit_gate(
        state: &mut WasmQuantumState,
        target: usize,
        matrix: &[[C64; 2]; 2],
    ) {
        let stride = 1usize << target;
        let dim = state.dim;

        let mut i = 0;
        while i < dim {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;

                let a0 = state.amplitudes[idx0];
                let a1 = state.amplitudes[idx1];

                state.amplitudes[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1;
                state.amplitudes[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1;
            }
            i += stride * 2;
        }
    }

    /// Apply a controlled single-qubit gate.
    ///
    /// The `matrix` is applied to the `target` qubit only when the `control`
    /// qubit is in state |1>. This is the general mechanism for CNOT, CZ,
    /// and controlled rotations.
    pub fn apply_controlled_gate(
        state: &mut WasmQuantumState,
        control: usize,
        target: usize,
        matrix: &[[C64; 2]; 2],
    ) {
        let dim = state.dim;
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;

        for idx in 0..dim {
            // Only process indices where control bit is 1 and target bit is 0
            // to avoid double-processing.
            if (idx & ctrl_mask) != 0 && (idx & tgt_mask) == 0 {
                let idx0 = idx; // target bit = 0
                let idx1 = idx | tgt_mask; // target bit = 1

                let a0 = state.amplitudes[idx0];
                let a1 = state.amplitudes[idx1];

                state.amplitudes[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1;
                state.amplitudes[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1;
            }
        }
    }

    /// Apply a SWAP gate between two qubits.
    ///
    /// Swaps the amplitudes of basis states that differ only in the two
    /// specified qubit positions.
    pub fn apply_swap(state: &mut WasmQuantumState, q1: usize, q2: usize) {
        let dim = state.dim;
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;

        for idx in 0..dim {
            let bit1 = (idx >> q1) & 1;
            let bit2 = (idx >> q2) & 1;
            // Only swap when the two bits differ, and process each pair once.
            if bit1 != bit2 && bit1 < bit2 {
                let partner = (idx ^ mask1) ^ mask2;
                state.amplitudes.swap(idx, partner);
            }
        }
    }

    /// Get the 2x2 unitary matrix for a single-qubit gate type.
    ///
    /// Returns `None` for multi-qubit gate types (CNOT, CZ, SWAP, etc.).
    pub fn gate_matrix(gate_type: &GateType) -> Option<[[C64; 2]; 2]> {
        let m = match gate_type {
            GateType::H => [
                [C64::new(FRAC_1_SQRT_2, 0.0), C64::new(FRAC_1_SQRT_2, 0.0)],
                [C64::new(FRAC_1_SQRT_2, 0.0), C64::new(-FRAC_1_SQRT_2, 0.0)],
            ],
            GateType::X => [[zero(), one()], [one(), zero()]],
            GateType::Y => [[zero(), C64::new(0.0, -1.0)], [C64::new(0.0, 1.0), zero()]],
            GateType::Z => [[one(), zero()], [zero(), C64::new(-1.0, 0.0)]],
            GateType::S => [[one(), zero()], [zero(), C64::new(0.0, 1.0)]],
            GateType::T => {
                let angle = PI / 4.0;
                [
                    [one(), zero()],
                    [zero(), C64::new(angle.cos(), angle.sin())],
                ]
            }
            GateType::Rx(theta) => {
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                [
                    [C64::new(c, 0.0), C64::new(0.0, -s)],
                    [C64::new(0.0, -s), C64::new(c, 0.0)],
                ]
            }
            GateType::Ry(theta) => {
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                [
                    [C64::new(c, 0.0), C64::new(-s, 0.0)],
                    [C64::new(s, 0.0), C64::new(c, 0.0)],
                ]
            }
            GateType::Rz(theta) => {
                let neg_half = -theta / 2.0;
                let pos_half = theta / 2.0;
                [
                    [C64::new(neg_half.cos(), neg_half.sin()), zero()],
                    [zero(), C64::new(pos_half.cos(), pos_half.sin())],
                ]
            }
            GateType::SX => [
                [C64::new(0.5, 0.5), C64::new(0.5, -0.5)],
                [C64::new(0.5, -0.5), C64::new(0.5, 0.5)],
            ],
            GateType::Phase(theta) => [
                [one(), zero()],
                [zero(), C64::new(theta.cos(), theta.sin())],
            ],
            GateType::U { theta, phi, lambda } => {
                let ct = (theta / 2.0).cos();
                let st = (theta / 2.0).sin();
                [
                    [
                        C64::new(ct, 0.0),
                        C64::new(-st * lambda.cos(), -st * lambda.sin()),
                    ],
                    [
                        C64::new(st * phi.cos(), st * phi.sin()),
                        C64::new(ct * (phi + lambda).cos(), ct * (phi + lambda).sin()),
                    ],
                ]
            }
            // Multi-qubit gate types -- no single 2x2 matrix.
            _ => return None,
        };
        Some(m)
    }

    /// Dispatch and apply a `Gate` to the quantum state.
    ///
    /// Handles single-qubit gates, controlled gates (CNOT, CZ, controlled
    /// rotations), and SWAP. Unsupported gate types (Toffoli, CCZ, Custom,
    /// ISWAP) are silently ignored in this lightweight backend.
    pub fn apply_gate(state: &mut WasmQuantumState, gate: &Gate) {
        match &gate.gate_type {
            // ----- Single-qubit gates -----
            GateType::H
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::S
            | GateType::T
            | GateType::SX
            | GateType::Rx(_)
            | GateType::Ry(_)
            | GateType::Rz(_)
            | GateType::Phase(_)
            | GateType::U { .. } => {
                if let Some(matrix) = Self::gate_matrix(&gate.gate_type) {
                    let target = gate.targets[0];
                    if gate.controls.is_empty() {
                        Self::apply_single_qubit_gate(state, target, &matrix);
                    } else {
                        // Controlled variant: use first control qubit.
                        Self::apply_controlled_gate(state, gate.controls[0], target, &matrix);
                    }
                }
            }
            // ----- Two-qubit gates -----
            GateType::CNOT => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                let x_matrix = [[zero(), one()], [one(), zero()]];
                Self::apply_controlled_gate(state, control, target, &x_matrix);
            }
            GateType::CZ => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                let z_matrix = [[one(), zero()], [zero(), C64::new(-1.0, 0.0)]];
                Self::apply_controlled_gate(state, control, target, &z_matrix);
            }
            GateType::SWAP => {
                // SWAP is stored with two targets, no controls.
                let q1 = gate.targets[0];
                let q2 = gate.targets[1];
                Self::apply_swap(state, q1, q2);
            }
            // ----- Controlled rotations -----
            GateType::CRx(angle) => {
                let c = (angle / 2.0).cos();
                let s = (angle / 2.0).sin();
                let matrix = [
                    [C64::new(c, 0.0), C64::new(0.0, -s)],
                    [C64::new(0.0, -s), C64::new(c, 0.0)],
                ];
                Self::apply_controlled_gate(state, gate.controls[0], gate.targets[0], &matrix);
            }
            GateType::CRy(angle) => {
                let c = (angle / 2.0).cos();
                let s = (angle / 2.0).sin();
                let matrix = [
                    [C64::new(c, 0.0), C64::new(-s, 0.0)],
                    [C64::new(s, 0.0), C64::new(c, 0.0)],
                ];
                Self::apply_controlled_gate(state, gate.controls[0], gate.targets[0], &matrix);
            }
            GateType::CRz(angle) => {
                let neg = -angle / 2.0;
                let pos = angle / 2.0;
                let matrix = [
                    [C64::new(neg.cos(), neg.sin()), zero()],
                    [zero(), C64::new(pos.cos(), pos.sin())],
                ];
                Self::apply_controlled_gate(state, gate.controls[0], gate.targets[0], &matrix);
            }
            GateType::CR(angle) => {
                let matrix = [
                    [one(), zero()],
                    [zero(), C64::new(angle.cos(), angle.sin())],
                ];
                Self::apply_controlled_gate(state, gate.controls[0], gate.targets[0], &matrix);
            }
            // ----- Unsupported in WASM lightweight backend -----
            GateType::Toffoli | GateType::CCZ | GateType::ISWAP | GateType::Custom(_) => {
                // Toffoli, CCZ, ISWAP, and custom unitaries are not
                // implemented in the lightweight WASM backend. These
                // require either 3-qubit dispatch or arbitrary-dimension
                // matrix application, which adds complexity beyond the
                // scope of a browser-oriented simulator.
            }
        }
    }

    /// Apply a sequence of gates (a circuit) to the state.
    pub fn apply_circuit(state: &mut WasmQuantumState, gates: &[Gate]) {
        for gate in gates {
            Self::apply_gate(state, gate);
        }
    }
}

// ============================================================
// WASM SIMULATOR
// ============================================================

/// High-level quantum simulator for browser/WASM environments.
///
/// Wraps `WasmQuantumState` and `WasmGateEngine` behind a convenient API
/// with named gate methods, measurement, and JS-interop-friendly output
/// formats. Includes an internal PRNG for probabilistic measurement.
pub struct WasmSimulator {
    state: WasmQuantumState,
    rng: XorShift64,
}

impl WasmSimulator {
    /// Create a new simulator with `num_qubits` initialized to |0...0>.
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        Self {
            state: WasmQuantumState::new(num_qubits),
            rng: XorShift64::from_state_properties(num_qubits, dim),
        }
    }

    /// Create a simulator with a specific PRNG seed for reproducibility.
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            state: WasmQuantumState::new(num_qubits),
            rng: XorShift64::new(seed),
        }
    }

    // ---- Single-qubit gates ----

    /// Apply Hadamard gate.
    pub fn h(&mut self, target: usize) {
        let m = [
            [C64::new(FRAC_1_SQRT_2, 0.0), C64::new(FRAC_1_SQRT_2, 0.0)],
            [C64::new(FRAC_1_SQRT_2, 0.0), C64::new(-FRAC_1_SQRT_2, 0.0)],
        ];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply Pauli-X (NOT) gate.
    pub fn x(&mut self, target: usize) {
        let m = [[zero(), one()], [one(), zero()]];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply Pauli-Y gate.
    pub fn y(&mut self, target: usize) {
        let m = [[zero(), C64::new(0.0, -1.0)], [C64::new(0.0, 1.0), zero()]];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply Pauli-Z gate.
    pub fn z(&mut self, target: usize) {
        let m = [[one(), zero()], [zero(), C64::new(-1.0, 0.0)]];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply S (phase pi/2) gate.
    pub fn s(&mut self, target: usize) {
        let m = [[one(), zero()], [zero(), C64::new(0.0, 1.0)]];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply T (pi/8) gate.
    pub fn t(&mut self, target: usize) {
        let angle = PI / 4.0;
        let m = [
            [one(), zero()],
            [zero(), C64::new(angle.cos(), angle.sin())],
        ];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply Rx rotation gate.
    pub fn rx(&mut self, target: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let m = [
            [C64::new(c, 0.0), C64::new(0.0, -s)],
            [C64::new(0.0, -s), C64::new(c, 0.0)],
        ];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply Ry rotation gate.
    pub fn ry(&mut self, target: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let m = [
            [C64::new(c, 0.0), C64::new(-s, 0.0)],
            [C64::new(s, 0.0), C64::new(c, 0.0)],
        ];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    /// Apply Rz rotation gate.
    pub fn rz(&mut self, target: usize, theta: f64) {
        let neg = -theta / 2.0;
        let pos = theta / 2.0;
        let m = [
            [C64::new(neg.cos(), neg.sin()), zero()],
            [zero(), C64::new(pos.cos(), pos.sin())],
        ];
        WasmGateEngine::apply_single_qubit_gate(&mut self.state, target, &m);
    }

    // ---- Two-qubit gates ----

    /// Apply CNOT (controlled-X) gate.
    pub fn cnot(&mut self, control: usize, target: usize) {
        let x_matrix = [[zero(), one()], [one(), zero()]];
        WasmGateEngine::apply_controlled_gate(&mut self.state, control, target, &x_matrix);
    }

    /// Apply CZ (controlled-Z) gate.
    pub fn cz(&mut self, control: usize, target: usize) {
        let z_matrix = [[one(), zero()], [zero(), C64::new(-1.0, 0.0)]];
        WasmGateEngine::apply_controlled_gate(&mut self.state, control, target, &z_matrix);
    }

    /// Apply SWAP gate.
    pub fn swap(&mut self, q1: usize, q2: usize) {
        WasmGateEngine::apply_swap(&mut self.state, q1, q2);
    }

    // ---- Measurement ----

    /// Measure a single qubit, collapsing the state.
    ///
    /// Returns `true` for outcome |1> and `false` for outcome |0>.
    /// The state vector is projected and renormalized after measurement.
    pub fn measure(&mut self, target: usize) -> bool {
        let tgt_mask = 1usize << target;

        // Compute probability of measuring |1> on the target qubit.
        let mut prob_one: f64 = 0.0;
        for idx in 0..self.state.dim {
            if (idx & tgt_mask) != 0 {
                prob_one += self.state.amplitudes[idx].norm_sqr();
            }
        }

        let r = self.rng.next_f64();
        let outcome = r < prob_one;

        // Collapse: zero out amplitudes inconsistent with the outcome.
        for idx in 0..self.state.dim {
            let bit_is_one = (idx & tgt_mask) != 0;
            if bit_is_one != outcome {
                self.state.amplitudes[idx] = zero();
            }
        }

        // Renormalize.
        self.state.normalize();

        outcome
    }

    /// Measure all qubits, collapsing the entire state.
    ///
    /// Returns a vector of bools (one per qubit, index 0 = qubit 0).
    /// The state is fully collapsed to a single computational basis state.
    pub fn measure_all(&mut self) -> Vec<bool> {
        let probs = self.state.probabilities();
        let r = self.rng.next_f64();

        // Sample from the cumulative distribution.
        let mut cumulative = 0.0;
        let mut outcome_idx = 0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                outcome_idx = i;
                break;
            }
        }

        // Collapse to the sampled basis state.
        for i in 0..self.state.dim {
            self.state.amplitudes[i] = zero();
        }
        self.state.amplitudes[outcome_idx] = one();

        // Extract per-qubit outcomes.
        let mut results = Vec::with_capacity(self.state.num_qubits);
        for q in 0..self.state.num_qubits {
            results.push((outcome_idx >> q) & 1 == 1);
        }
        results
    }

    /// Get the probability distribution without collapsing the state.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.probabilities()
    }

    /// Return the state vector as (re, im) pairs for JavaScript interop.
    ///
    /// This avoids exposing `C64` directly, which is not `#[repr(C)]`
    /// compatible with wasm-bindgen. Each tuple is `(real_part, imag_part)`.
    pub fn state_vector(&self) -> Vec<(f64, f64)> {
        self.state.amplitudes.iter().map(|a| (a.re, a.im)).collect()
    }

    /// Reset the simulator to |0...0>.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Get a reference to the internal quantum state.
    pub fn state(&self) -> &WasmQuantumState {
        &self.state
    }

    /// Get a mutable reference to the internal quantum state.
    pub fn state_mut(&mut self) -> &mut WasmQuantumState {
        &mut self.state
    }
}

// ============================================================
// WASM CIRCUIT BUILDER
// ============================================================

/// Builder for constructing quantum circuits as `Vec<Gate>`.
///
/// Provides a fluent API where gate methods return `&mut Self` for chaining.
/// Call `build()` to extract the final gate sequence.
///
/// # Example
///
/// ```rust,ignore
/// let circuit = WasmCircuitBuilder::new()
///     .h(0)
///     .cnot(0, 1)
///     .measure_z(0)
///     .build();
/// ```
pub struct WasmCircuitBuilder {
    gates: Vec<Gate>,
}

impl WasmCircuitBuilder {
    /// Create an empty circuit builder.
    pub fn new() -> Self {
        Self { gates: Vec::new() }
    }

    /// Append a Hadamard gate.
    pub fn h(&mut self, target: usize) -> &mut Self {
        self.gates.push(Gate::h(target));
        self
    }

    /// Append a Pauli-X gate.
    pub fn x(&mut self, target: usize) -> &mut Self {
        self.gates.push(Gate::x(target));
        self
    }

    /// Append a Pauli-Y gate.
    pub fn y(&mut self, target: usize) -> &mut Self {
        self.gates.push(Gate::y(target));
        self
    }

    /// Append a Pauli-Z gate.
    pub fn z(&mut self, target: usize) -> &mut Self {
        self.gates.push(Gate::z(target));
        self
    }

    /// Append an S gate.
    pub fn s(&mut self, target: usize) -> &mut Self {
        self.gates.push(Gate::s(target));
        self
    }

    /// Append a T gate.
    pub fn t(&mut self, target: usize) -> &mut Self {
        self.gates.push(Gate::t(target));
        self
    }

    /// Append an Rx rotation gate.
    pub fn rx(&mut self, target: usize, angle: f64) -> &mut Self {
        self.gates.push(Gate::rx(target, angle));
        self
    }

    /// Append an Ry rotation gate.
    pub fn ry(&mut self, target: usize, angle: f64) -> &mut Self {
        self.gates.push(Gate::ry(target, angle));
        self
    }

    /// Append an Rz rotation gate.
    pub fn rz(&mut self, target: usize, angle: f64) -> &mut Self {
        self.gates.push(Gate::rz(target, angle));
        self
    }

    /// Append a CNOT gate.
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.gates.push(Gate::cnot(control, target));
        self
    }

    /// Append a CZ gate.
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.gates.push(Gate::cz(control, target));
        self
    }

    /// Append a SWAP gate.
    pub fn swap(&mut self, q1: usize, q2: usize) -> &mut Self {
        self.gates.push(Gate::swap(q1, q2));
        self
    }

    /// Append a Toffoli (CCX) gate.
    pub fn toffoli(&mut self, c1: usize, c2: usize, target: usize) -> &mut Self {
        self.gates.push(Gate::toffoli(c1, c2, target));
        self
    }

    /// Append an arbitrary `Gate`.
    pub fn gate(&mut self, g: Gate) -> &mut Self {
        self.gates.push(g);
        self
    }

    /// Return the number of gates currently in the builder.
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Check if the builder has no gates.
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Consume the builder and return the gate sequence.
    pub fn build(self) -> Vec<Gate> {
        self.gates
    }
}

impl Default for WasmCircuitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(all(test, feature = "wasm"))]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn c64_approx_eq(a: C64, b: C64) -> bool {
        (a.re - b.re).abs() < EPSILON && (a.im - b.im).abs() < EPSILON
    }

    // ---- State creation and initialization ----

    #[test]
    fn test_state_new_single_qubit() {
        let state = WasmQuantumState::new(1);
        assert_eq!(state.num_qubits(), 1);
        assert_eq!(state.dim(), 2);
        assert!(c64_approx_eq(state.get(0), one()));
        assert!(c64_approx_eq(state.get(1), zero()));
    }

    #[test]
    fn test_state_new_multi_qubit() {
        let state = WasmQuantumState::new(3);
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.dim(), 8);
        assert!(c64_approx_eq(state.get(0), one()));
        for i in 1..8 {
            assert!(c64_approx_eq(state.get(i), zero()));
        }
    }

    #[test]
    fn test_state_set_and_get() {
        let mut state = WasmQuantumState::new(2);
        let val = C64::new(0.3, 0.4);
        state.set(2, val);
        assert!(c64_approx_eq(state.get(2), val));
    }

    // ---- Normalization ----

    #[test]
    fn test_normalize() {
        let mut state = WasmQuantumState::new(1);
        state.set(0, C64::new(3.0, 0.0));
        state.set(1, C64::new(4.0, 0.0));
        state.normalize();
        let probs = state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!(approx_eq(total, 1.0));
        assert!(approx_eq(probs[0], 9.0 / 25.0));
        assert!(approx_eq(probs[1], 16.0 / 25.0));
    }

    #[test]
    fn test_normalize_zero_state() {
        let mut state = WasmQuantumState::new(1);
        state.set(0, zero());
        state.set(1, zero());
        // Should not panic or produce NaN.
        state.normalize();
        assert!(c64_approx_eq(state.get(0), zero()));
        assert!(c64_approx_eq(state.get(1), zero()));
    }

    // ---- Fidelity ----

    #[test]
    fn test_fidelity_identical_states() {
        let state = WasmQuantumState::new(2);
        let f = state.fidelity(&state);
        assert!(approx_eq(f, 1.0));
    }

    #[test]
    fn test_fidelity_orthogonal_states() {
        let state_a = WasmQuantumState::new(1); // |0>
        let mut state_b = WasmQuantumState::new(1);
        state_b.set(0, zero());
        state_b.set(1, one()); // |1>
        let f = state_a.fidelity(&state_b);
        assert!(approx_eq(f, 0.0));
    }

    // ---- Single-qubit gates ----

    #[test]
    fn test_hadamard_on_zero() {
        let mut state = WasmQuantumState::new(1);
        let h_matrix = WasmGateEngine::gate_matrix(&GateType::H).unwrap();
        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &h_matrix);

        let probs = state.probabilities();
        assert!(approx_eq(probs[0], 0.5));
        assert!(approx_eq(probs[1], 0.5));
    }

    #[test]
    fn test_x_gate_flips() {
        let mut state = WasmQuantumState::new(1);
        let x_matrix = WasmGateEngine::gate_matrix(&GateType::X).unwrap();
        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &x_matrix);

        assert!(c64_approx_eq(state.get(0), zero()));
        assert!(c64_approx_eq(state.get(1), one()));
    }

    #[test]
    fn test_z_gate_phase() {
        // Z|+> = |->
        let mut state = WasmQuantumState::new(1);
        let h_matrix = WasmGateEngine::gate_matrix(&GateType::H).unwrap();
        let z_matrix = WasmGateEngine::gate_matrix(&GateType::Z).unwrap();

        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &h_matrix);
        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &z_matrix);

        // |-> = (|0> - |1>) / sqrt(2)
        assert!(c64_approx_eq(state.get(0), C64::new(FRAC_1_SQRT_2, 0.0)));
        assert!(c64_approx_eq(state.get(1), C64::new(-FRAC_1_SQRT_2, 0.0)));
    }

    #[test]
    fn test_double_x_is_identity() {
        let mut state = WasmQuantumState::new(1);
        let x_matrix = WasmGateEngine::gate_matrix(&GateType::X).unwrap();
        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &x_matrix);
        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &x_matrix);

        assert!(c64_approx_eq(state.get(0), one()));
        assert!(c64_approx_eq(state.get(1), zero()));
    }

    // ---- CNOT and Bell state ----

    #[test]
    fn test_cnot_creates_bell_state() {
        let mut state = WasmQuantumState::new(2);

        // H on qubit 0 -> |+0>
        let h_matrix = WasmGateEngine::gate_matrix(&GateType::H).unwrap();
        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &h_matrix);

        // CNOT(0, 1) -> Bell state (|00> + |11>) / sqrt(2)
        let x_matrix = WasmGateEngine::gate_matrix(&GateType::X).unwrap();
        WasmGateEngine::apply_controlled_gate(&mut state, 0, 1, &x_matrix);

        let probs = state.probabilities();
        assert!(approx_eq(probs[0], 0.5)); // |00>
        assert!(approx_eq(probs[1], 0.0)); // |01>
        assert!(approx_eq(probs[2], 0.0)); // |10>
        assert!(approx_eq(probs[3], 0.5)); // |11>
    }

    // ---- Measurement probabilities ----

    #[test]
    fn test_measurement_probabilities_sum_to_one() {
        let mut sim = WasmSimulator::with_seed(3, 42);
        sim.h(0);
        sim.cnot(0, 1);
        sim.h(2);

        let probs = sim.probabilities();
        let total: f64 = probs.iter().sum();
        assert!(approx_eq(total, 1.0));
    }

    #[test]
    fn test_measure_deterministic_state() {
        // |1> state: measuring should always return true.
        let mut sim = WasmSimulator::with_seed(1, 12345);
        sim.x(0);
        let result = sim.measure(0);
        assert!(result);
    }

    // ---- Circuit builder ----

    #[test]
    fn test_circuit_builder_bell_state() {
        let circuit = WasmCircuitBuilder::new().h(0).cnot(0, 1).build();

        assert_eq!(circuit.len(), 2);

        let mut state = WasmQuantumState::new(2);
        WasmGateEngine::apply_circuit(&mut state, &circuit);

        let probs = state.probabilities();
        assert!(approx_eq(probs[0], 0.5));
        assert!(approx_eq(probs[3], 0.5));
    }

    #[test]
    fn test_circuit_builder_chaining() {
        let mut builder = WasmCircuitBuilder::new();
        builder.h(0).x(1).cnot(0, 1).z(0);
        assert_eq!(builder.len(), 4);
        assert!(!builder.is_empty());
    }

    #[test]
    fn test_circuit_builder_empty() {
        let builder = WasmCircuitBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
        let circuit = builder.build();
        assert!(circuit.is_empty());
    }

    // ---- XorShift64 ----

    #[test]
    fn test_xorshift_range() {
        let mut rng = XorShift64::new(42);
        for _ in 0..1000 {
            let val = rng.next_f64();
            assert!(val >= 0.0);
            assert!(val < 1.0);
        }
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = XorShift64::new(12345);
        let mut rng2 = XorShift64::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    #[should_panic(expected = "nonzero")]
    fn test_xorshift_zero_seed_panics() {
        XorShift64::new(0);
    }

    // ---- Simulator JS interop ----

    #[test]
    fn test_state_vector_interop() {
        let sim = WasmSimulator::new(1);
        let sv = sim.state_vector();
        assert_eq!(sv.len(), 2);
        assert!(approx_eq(sv[0].0, 1.0));
        assert!(approx_eq(sv[0].1, 0.0));
        assert!(approx_eq(sv[1].0, 0.0));
        assert!(approx_eq(sv[1].1, 0.0));
    }

    // ---- Reset ----

    #[test]
    fn test_simulator_reset() {
        let mut sim = WasmSimulator::new(2);
        sim.h(0);
        sim.cnot(0, 1);
        sim.reset();

        let sv = sim.state_vector();
        assert!(approx_eq(sv[0].0, 1.0));
        for i in 1..4 {
            assert!(approx_eq(sv[i].0, 0.0));
            assert!(approx_eq(sv[i].1, 0.0));
        }
    }

    // ---- SWAP gate ----

    #[test]
    fn test_swap_gate() {
        // Prepare |10> then swap -> |01>
        let mut state = WasmQuantumState::new(2);
        let x_matrix = WasmGateEngine::gate_matrix(&GateType::X).unwrap();
        WasmGateEngine::apply_single_qubit_gate(&mut state, 0, &x_matrix);

        // State is now |01> in little-endian (qubit 0 = 1, qubit 1 = 0) -> index 1
        assert!(c64_approx_eq(state.get(1), one()));

        WasmGateEngine::apply_swap(&mut state, 0, 1);

        // After swap: qubit 0 = 0, qubit 1 = 1 -> index 2
        assert!(c64_approx_eq(state.get(2), one()));
        assert!(c64_approx_eq(state.get(1), zero()));
    }
}
