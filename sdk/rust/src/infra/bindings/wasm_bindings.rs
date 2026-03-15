//! WebAssembly bindings for nQPU-Metal quantum simulator.
//!
//! This module provides JavaScript-friendly wrappers around the core quantum
//! simulator via `wasm-bindgen`. The entire module is gated behind
//! `#[cfg(feature = "wasm")]` so the crate compiles cleanly without the
//! `wasm-bindgen` dependency.
//!
//! # Exported types
//!
//! - [`JsQuantumSimulator`] -- stateful simulator with single-gate methods
//! - [`JsCircuitRunner`]    -- gate-list builder that executes in one shot
//!
//! # Exported free functions
//!
//! - [`bell_state`]      -- returns Bell-state probabilities
//! - [`ghz_state`]       -- returns GHZ-state probabilities for *n* qubits
//! - [`random_circuit`]  -- returns probabilities from a random circuit

// ---------------------------------------------------------------------------
// Guard: everything below requires the "wasm" feature.
// ---------------------------------------------------------------------------
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::{GateOperations, QuantumState, C64};

// =========================================================================
// JsQuantumSimulator
// =========================================================================

/// JavaScript-facing quantum simulator.
///
/// Wraps the core [`QuantumState`] and [`GateOperations`] into a single
/// mutable object whose methods map one-to-one onto standard quantum gates.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct JsQuantumSimulator {
    state: QuantumState,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl JsQuantumSimulator {
    // ----- construction / meta -----

    /// Create a new simulator initialised to |0...0>.
    #[wasm_bindgen(constructor)]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            state: QuantumState::new(num_qubits),
        }
    }

    /// Number of qubits in the register.
    #[wasm_bindgen]
    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits
    }

    /// Reset the state back to |0...0>.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.state = QuantumState::new(self.state.num_qubits);
    }

    // ----- single-qubit gates -----

    /// Hadamard gate.
    #[wasm_bindgen]
    pub fn h(&mut self, target: usize) {
        GateOperations::h(&mut self.state, target);
    }

    /// Pauli-X (NOT) gate.
    #[wasm_bindgen]
    pub fn x(&mut self, target: usize) {
        GateOperations::x(&mut self.state, target);
    }

    /// Pauli-Y gate.
    #[wasm_bindgen]
    pub fn y(&mut self, target: usize) {
        GateOperations::y(&mut self.state, target);
    }

    /// Pauli-Z gate.
    #[wasm_bindgen]
    pub fn z(&mut self, target: usize) {
        GateOperations::z(&mut self.state, target);
    }

    /// S (phase) gate -- sqrt(Z).
    #[wasm_bindgen]
    pub fn s(&mut self, target: usize) {
        GateOperations::s(&mut self.state, target);
    }

    /// T gate -- fourth root of Z.
    #[wasm_bindgen]
    pub fn t(&mut self, target: usize) {
        GateOperations::t(&mut self.state, target);
    }

    /// Rotation about X axis by angle `theta`.
    #[wasm_bindgen]
    pub fn rx(&mut self, target: usize, theta: f64) {
        GateOperations::rx(&mut self.state, target, theta);
    }

    /// Rotation about Y axis by angle `theta`.
    #[wasm_bindgen]
    pub fn ry(&mut self, target: usize, theta: f64) {
        GateOperations::ry(&mut self.state, target, theta);
    }

    /// Rotation about Z axis by angle `theta`.
    #[wasm_bindgen]
    pub fn rz(&mut self, target: usize, theta: f64) {
        GateOperations::rz(&mut self.state, target, theta);
    }

    // ----- two-qubit gates -----

    /// Controlled-NOT (CNOT) gate.
    #[wasm_bindgen]
    pub fn cnot(&mut self, control: usize, target: usize) {
        GateOperations::cnot(&mut self.state, control, target);
    }

    /// Controlled-Z (CZ) gate.
    #[wasm_bindgen]
    pub fn cz(&mut self, control: usize, target: usize) {
        GateOperations::cz(&mut self.state, control, target);
    }

    /// SWAP gate.
    #[wasm_bindgen]
    pub fn swap(&mut self, q1: usize, q2: usize) {
        GateOperations::swap(&mut self.state, q1, q2);
    }

    // ----- measurement -----

    /// Measure a single qubit, collapsing the state.
    ///
    /// Returns `true` for |1> and `false` for |0>.
    #[wasm_bindgen]
    pub fn measure(&mut self, target: usize) -> bool {
        let stride = 1usize << target;
        let dim = self.state.dim;
        let amplitudes = self.state.amplitudes_ref();

        // Probability of measuring |1> on this qubit.
        let mut prob_one: f64 = 0.0;
        for i in 0..dim {
            if i & stride != 0 {
                prob_one += amplitudes[i].norm_sqr();
            }
        }

        let outcome: bool = rand::random::<f64>() < prob_one;

        // Collapse: zero out incompatible amplitudes and renormalise.
        let amps = self.state.amplitudes_mut();
        let mut norm_sq: f64 = 0.0;
        for i in 0..dim {
            let bit_set = (i & stride) != 0;
            if bit_set != outcome {
                amps[i] = C64::new(0.0, 0.0);
            } else {
                norm_sq += amps[i].norm_sqr();
            }
        }

        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for i in 0..dim {
                let a = amps[i];
                amps[i] = C64::new(a.re * inv_norm, a.im * inv_norm);
            }
        }

        outcome
    }

    /// Measure all qubits.
    ///
    /// Returns a `Vec<u8>` where each byte is 0 or 1, indexed by qubit
    /// number (qubit 0 is index 0). This representation avoids
    /// `wasm-bindgen` limitations with `Vec<bool>`.
    #[wasm_bindgen]
    pub fn measure_all(&mut self) -> Vec<u8> {
        let n = self.state.num_qubits;
        let mut results = Vec::with_capacity(n);
        // Measure from highest to lowest so earlier collapses do not shift
        // indices, though single-qubit collapse is index-independent.
        for q in 0..n {
            results.push(if self.measure(q) { 1u8 } else { 0u8 });
        }
        results
    }

    // ----- state inspection -----

    /// Probability distribution over all basis states.
    #[wasm_bindgen]
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.probabilities()
    }

    /// Real parts of the state-vector amplitudes.
    #[wasm_bindgen]
    pub fn state_vector_real(&self) -> Vec<f64> {
        self.state.amplitudes_ref().iter().map(|a| a.re).collect()
    }

    /// Imaginary parts of the state-vector amplitudes.
    #[wasm_bindgen]
    pub fn state_vector_imag(&self) -> Vec<f64> {
        self.state.amplitudes_ref().iter().map(|a| a.im).collect()
    }
}

// =========================================================================
// JsCircuitRunner
// =========================================================================

/// A gate is stored as (name, target, optional control, optional parameter).
#[cfg(feature = "wasm")]
struct GateEntry {
    name: String,
    target: usize,
    control: Option<usize>,
    param: Option<f64>,
}

/// Builder that accumulates gates and executes them in a single `run()` call.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct JsCircuitRunner {
    num_qubits: usize,
    gates: Vec<GateEntry>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl JsCircuitRunner {
    /// Create a new circuit runner for `num_qubits` qubits.
    #[wasm_bindgen(constructor)]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }

    /// Append a gate to the circuit.
    ///
    /// `gate_name` is case-insensitive and supports: h, x, y, z, s, t,
    /// rx, ry, rz, cnot, cx, cz, swap.
    ///
    /// For controlled gates (`cnot`/`cx`/`cz`), `control` must be `Some`.
    /// For parameterised gates (`rx`/`ry`/`rz`), `param` must be `Some`.
    #[wasm_bindgen]
    pub fn add_gate(
        &mut self,
        gate_name: &str,
        target: usize,
        control: Option<usize>,
        param: Option<f64>,
    ) {
        self.gates.push(GateEntry {
            name: gate_name.to_ascii_lowercase(),
            target,
            control,
            param,
        });
    }

    /// Execute all accumulated gates and return the probability distribution.
    #[wasm_bindgen]
    pub fn run(&mut self) -> Vec<f64> {
        let mut state = QuantumState::new(self.num_qubits);

        for gate in &self.gates {
            match gate.name.as_str() {
                "h" => GateOperations::h(&mut state, gate.target),
                "x" => GateOperations::x(&mut state, gate.target),
                "y" => GateOperations::y(&mut state, gate.target),
                "z" => GateOperations::z(&mut state, gate.target),
                "s" => GateOperations::s(&mut state, gate.target),
                "t" => GateOperations::t(&mut state, gate.target),
                "rx" => {
                    let theta = gate.param.unwrap_or(0.0);
                    GateOperations::rx(&mut state, gate.target, theta);
                }
                "ry" => {
                    let theta = gate.param.unwrap_or(0.0);
                    GateOperations::ry(&mut state, gate.target, theta);
                }
                "rz" => {
                    let theta = gate.param.unwrap_or(0.0);
                    GateOperations::rz(&mut state, gate.target, theta);
                }
                "cnot" | "cx" => {
                    if let Some(ctrl) = gate.control {
                        GateOperations::cnot(&mut state, ctrl, gate.target);
                    }
                }
                "cz" => {
                    if let Some(ctrl) = gate.control {
                        GateOperations::cz(&mut state, ctrl, gate.target);
                    }
                }
                "swap" => {
                    if let Some(other) = gate.control {
                        GateOperations::swap(&mut state, gate.target, other);
                    }
                }
                _ => {} // unknown gate -- silently skip
            }
        }

        state.probabilities()
    }

    /// Remove all gates from the circuit.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.gates.clear();
    }
}

// =========================================================================
// Free functions
// =========================================================================

/// Create a Bell state (|00> + |11>) / sqrt(2) and return its probabilities.
///
/// Expected output: [0.5, 0.0, 0.0, 0.5].
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bell_state() -> Vec<f64> {
    let mut state = QuantumState::new(2);
    GateOperations::h(&mut state, 0);
    GateOperations::cnot(&mut state, 0, 1);
    state.probabilities()
}

/// Create a GHZ state (|00...0> + |11...1>) / sqrt(2) for `n` qubits and
/// return its probabilities.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ghz_state(n: usize) -> Vec<f64> {
    let n = if n < 2 { 2 } else { n };
    let mut state = QuantumState::new(n);
    GateOperations::h(&mut state, 0);
    for q in 1..n {
        GateOperations::cnot(&mut state, 0, q);
    }
    state.probabilities()
}

/// Build a pseudo-random circuit of the given `depth` on `num_qubits` qubits
/// and return its probability distribution.
///
/// The circuit alternates layers of random single-qubit gates (H, X, Y, Z,
/// S, T) with layers of CNOT gates on adjacent pairs. Randomness comes from
/// `rand::random`, so results differ on each call.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn random_circuit(num_qubits: usize, depth: usize) -> Vec<f64> {
    let num_qubits = if num_qubits < 1 { 1 } else { num_qubits };
    let mut state = QuantumState::new(num_qubits);

    for layer in 0..depth {
        // Single-qubit layer
        for q in 0..num_qubits {
            let gate_choice: u8 = (rand::random::<u8>()) % 6;
            match gate_choice {
                0 => GateOperations::h(&mut state, q),
                1 => GateOperations::x(&mut state, q),
                2 => GateOperations::y(&mut state, q),
                3 => GateOperations::z(&mut state, q),
                4 => GateOperations::s(&mut state, q),
                5 => GateOperations::t(&mut state, q),
                _ => unreachable!(),
            }
        }

        // Entangling layer (CNOT on adjacent pairs, offset by layer parity)
        if num_qubits > 1 {
            let start = layer % 2;
            let mut q = start;
            while q + 1 < num_qubits {
                GateOperations::cnot(&mut state, q, q + 1);
                q += 2;
            }
        }
    }

    state.probabilities()
}

// =========================================================================
// Interactive Wave Visualization Engine (Rust/WASM)
// =========================================================================

#[cfg(feature = "wasm")]
#[derive(Clone, Copy)]
struct VizRipple {
    cx: f32,
    cy: f32,
    birth: f32,
    strength: f32,
}

#[cfg(feature = "wasm")]
#[derive(Clone, Copy)]
struct VizWakePoint {
    x: f32,
    y: f32,
    time: f32,
}

#[cfg(feature = "wasm")]
const ASCII_RAMP: [u8; 12] = *b" .,-~:;=+*#^";

/// High-performance wave-based visualization engine for the web UI.
///
/// This engine is intentionally data-oriented: all buffers are preallocated,
/// updates are in-place, and each `tick()` produces both an ASCII frame and
/// an RGBA frame so the UI can switch render modes with near-zero overhead.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct JsQuantumVizEngine {
    cols: usize,
    rows: usize,
    time: f32,
    dye: Vec<f32>,
    dye_next: Vec<f32>,
    rgba: Vec<u8>,
    ascii_bytes: Vec<u8>,
    ripples: Vec<VizRipple>,
    wake: Vec<VizWakePoint>,
    pointer_x: f32,
    pointer_y: f32,
    pointer_last_x: f32,
    pointer_last_y: f32,
    pointer_vx: f32,
    pointer_vy: f32,
    pointer_active: bool,
    ripple_cooldown: f32,
    wake_cooldown: f32,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl JsQuantumVizEngine {
    /// Create a new visualization engine.
    #[wasm_bindgen(constructor)]
    pub fn new(cols: usize, rows: usize) -> Self {
        let mut engine = Self {
            cols: cols.clamp(32, 240),
            rows: rows.clamp(20, 140),
            time: 0.0,
            dye: Vec::new(),
            dye_next: Vec::new(),
            rgba: Vec::new(),
            ascii_bytes: Vec::new(),
            ripples: Vec::with_capacity(64),
            wake: Vec::with_capacity(64),
            pointer_x: -1.0,
            pointer_y: -1.0,
            pointer_last_x: -1.0,
            pointer_last_y: -1.0,
            pointer_vx: 0.0,
            pointer_vy: 0.0,
            pointer_active: false,
            ripple_cooldown: 0.0,
            wake_cooldown: 0.0,
        };
        engine.reset_buffers();
        engine.render_frame();
        engine
    }

    /// Current frame width in cells/pixels.
    #[wasm_bindgen]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Current frame height in cells/pixels.
    #[wasm_bindgen]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Resize the simulation and rendering buffers.
    #[wasm_bindgen]
    pub fn resize(&mut self, cols: usize, rows: usize) {
        self.cols = cols.clamp(32, 240);
        self.rows = rows.clamp(20, 140);
        self.time = 0.0;
        self.ripples.clear();
        self.wake.clear();
        self.pointer_leave();
        self.reset_buffers();
        self.render_frame();
    }

    /// Advance simulation state and render fresh buffers.
    ///
    /// `dt_seconds` should be the wall-clock frame delta (e.g. ~0.016 for 60 FPS).
    #[wasm_bindgen]
    pub fn tick(&mut self, dt_seconds: f32) {
        let dt = dt_seconds.clamp(0.001, 0.050);
        self.time += dt * 1.35;
        self.ripple_cooldown = (self.ripple_cooldown - dt).max(0.0);
        self.wake_cooldown = (self.wake_cooldown - dt).max(0.0);

        self.update_dye();

        if self.pointer_active && self.pointer_x >= 0.0 && self.pointer_y >= 0.0 {
            self.paint_dye(self.pointer_x, self.pointer_y, 0.17);

            if self.wake_cooldown <= 0.0 {
                self.wake.push(VizWakePoint {
                    x: self.pointer_x,
                    y: self.pointer_y,
                    time: self.time,
                });
                self.wake_cooldown = 0.018;
                if self.wake.len() > 24 {
                    let drop_n = self.wake.len() - 24;
                    self.wake.drain(0..drop_n);
                }
            }
        }

        self.ripples
            .retain(|r| (self.time - r.birth) <= 4.0 && r.strength > 0.01);
        self.wake.retain(|w| (self.time - w.time) <= 2.0);

        self.render_frame();
    }

    /// Update pointer state from normalized coordinates in [0, 1].
    #[wasm_bindgen]
    pub fn set_pointer(&mut self, x_norm: f32, y_norm: f32, active: bool) {
        if !active {
            self.pointer_leave();
            return;
        }

        let gx = x_norm.clamp(0.0, 1.0) * (self.cols.saturating_sub(1) as f32);
        let gy = y_norm.clamp(0.0, 1.0) * (self.rows.saturating_sub(1) as f32);

        if self.pointer_last_x >= 0.0 {
            self.pointer_vx = gx - self.pointer_last_x;
            self.pointer_vy = gy - self.pointer_last_y;
        } else {
            self.pointer_vx = 0.0;
            self.pointer_vy = 0.0;
        }

        self.pointer_active = true;
        self.pointer_x = gx;
        self.pointer_y = gy;
        self.pointer_last_x = gx;
        self.pointer_last_y = gy;

        let speed = (self.pointer_vx * self.pointer_vx + self.pointer_vy * self.pointer_vy).sqrt();
        let paint_strength = (speed * 0.24).clamp(0.28, 0.92);
        self.paint_dye(gx, gy, paint_strength);

        if self.ripple_cooldown <= 0.0 && speed > 0.04 {
            let strength = (speed * 1.4).clamp(1.9, 5.2);
            self.ripples.push(VizRipple {
                cx: gx,
                cy: gy,
                birth: self.time,
                strength,
            });
            self.ripple_cooldown = 0.045;
            if self.ripples.len() > 48 {
                let drop_n = self.ripples.len() - 48;
                self.ripples.drain(0..drop_n);
            }
        }
    }

    /// Mark pointer as inactive.
    #[wasm_bindgen]
    pub fn pointer_leave(&mut self) {
        self.pointer_active = false;
        self.pointer_x = -1.0;
        self.pointer_y = -1.0;
        self.pointer_last_x = -1.0;
        self.pointer_last_y = -1.0;
        self.pointer_vx = 0.0;
        self.pointer_vy = 0.0;
    }

    /// Trigger a stronger splash at a normalized position.
    #[wasm_bindgen]
    pub fn pointer_click(&mut self, x_norm: f32, y_norm: f32, strength: f32) {
        let gx = x_norm.clamp(0.0, 1.0) * (self.cols.saturating_sub(1) as f32);
        let gy = y_norm.clamp(0.0, 1.0) * (self.rows.saturating_sub(1) as f32);
        let s = strength.clamp(1.0, 12.0);
        self.paint_dye(gx, gy, 0.85);
        self.ripples.push(VizRipple {
            cx: gx,
            cy: gy,
            birth: self.time,
            strength: s,
        });
        if self.ripples.len() > 48 {
            let drop_n = self.ripples.len() - 48;
            self.ripples.drain(0..drop_n);
        }
    }

    /// Inject a pulse derived from quantum state metrics.
    #[wasm_bindgen]
    pub fn inject_quantum_pulse(&mut self, strength: f32) {
        let s = strength.clamp(0.2, 8.0);
        let cx = self.cols as f32 * 0.5;
        let cy = self.rows as f32 * 0.55;
        self.paint_dye(cx, cy, (0.2 + s * 0.08).clamp(0.2, 1.0));
        self.ripples.push(VizRipple {
            cx,
            cy,
            birth: self.time,
            strength: s,
        });
        if self.ripples.len() > 48 {
            let drop_n = self.ripples.len() - 48;
            self.ripples.drain(0..drop_n);
        }
    }

    /// Return the latest ASCII frame.
    #[wasm_bindgen]
    pub fn ascii_frame(&self) -> String {
        String::from_utf8_lossy(&self.ascii_bytes).into_owned()
    }

    /// Return the latest RGBA frame buffer (length = cols * rows * 4).
    #[wasm_bindgen]
    pub fn rgba_frame(&self) -> Vec<u8> {
        self.rgba.clone()
    }
}

#[cfg(feature = "wasm")]
impl JsQuantumVizEngine {
    fn reset_buffers(&mut self) {
        let n = self.cols * self.rows;
        self.dye = vec![0.0; n];
        self.dye_next = vec![0.0; n];
        self.rgba = vec![0; n * 4];
        self.ascii_bytes = vec![b' '; self.rows * (self.cols + 1)];
        for y in 0..self.rows {
            self.ascii_bytes[y * (self.cols + 1) + self.cols] = b'\n';
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.cols + x
    }

    fn update_dye(&mut self) {
        const DIFFUSE_RATE: f32 = 0.04;
        const DECAY_RATE: f32 = 0.985;

        for y in 0..self.rows {
            for x in 0..self.cols {
                let mut sum = 0.0f32;
                let mut count = 0.0f32;

                if x > 0 {
                    sum += self.dye[self.idx(x - 1, y)];
                    count += 1.0;
                }
                if x + 1 < self.cols {
                    sum += self.dye[self.idx(x + 1, y)];
                    count += 1.0;
                }
                if y > 0 {
                    sum += self.dye[self.idx(x, y - 1)];
                    count += 1.0;
                }
                if y + 1 < self.rows {
                    sum += self.dye[self.idx(x, y + 1)];
                    count += 1.0;
                }

                let avg = if count > 0.0 { sum / count } else { 0.0 };
                let i = self.idx(x, y);
                self.dye_next[i] =
                    (self.dye[i] * (1.0 - DIFFUSE_RATE) + avg * DIFFUSE_RATE) * DECAY_RATE;
            }
        }

        std::mem::swap(&mut self.dye, &mut self.dye_next);
    }

    fn paint_dye(&mut self, gx: f32, gy: f32, strength: f32) {
        let radius = 4.0f32;
        let x0 = ((gx - radius).floor() as isize).max(0) as usize;
        let y0 = ((gy - radius).floor() as isize).max(0) as usize;
        let x1 = ((gx + radius).ceil() as isize).min((self.cols as isize) - 1) as usize;
        let y1 = ((gy + radius).ceil() as isize).min((self.rows as isize) - 1) as usize;

        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let dx = xx as f32 - gx;
                let dy = (yy as f32 - gy) * 1.8;
                let dist2 = dx * dx + dy * dy;
                let falloff = (-dist2 * 0.15).exp() * strength;
                let i = self.idx(xx, yy);
                self.dye[i] = (self.dye[i] + falloff).min(1.0);
            }
        }
    }

    #[inline]
    fn clamp01(v: f32) -> f32 {
        v.clamp(0.0, 1.0)
    }

    fn current_x(&self, t: f32) -> f32 {
        (t * 0.018).sin() * 0.7 + (t * 0.042).sin() * 0.35
    }

    fn current_y(&self, t: f32) -> f32 {
        (t * 0.025 + 1.3).sin() * 0.6 + (t * 0.035 + 2.7).sin() * 0.3
    }

    fn wave_height(&self, x: f32, y: f32, t: f32) -> f32 {
        let cx = self.current_x(t);
        let cy = self.current_y(t);
        let dx = x + cx * t * 0.35;
        let dy = y + cy * t * 0.35;
        let depth = y / (self.rows as f32).max(1.0);
        let mut v = 0.0f32;

        v += (dy * 0.038 - t * 0.10 + x * 0.010).sin() * 1.6;
        v += (dy * 0.065 + t * 0.075 - x * 0.014).sin() * 1.1;
        v += (dy * 0.10 - t * 0.13 + x * 0.007).sin() * 0.75;
        v += (dy * 0.15 + t * 0.055 - x * 0.020).sin() * 0.45;

        v += (dx * 0.035 - t * 0.12 + y * 0.020).sin() * 1.3;
        v += (dx * 0.07 + t * 0.16 - y * 0.030).sin() * 0.85;
        v += (dx * 0.12 - t * 0.095 + y * 0.008).sin() * 0.55;
        v += (dx * 0.18 + t * 0.20 - y * 0.040).sin() * 0.35;

        v += ((dx * 0.6 + dy * 0.8) * 0.042 - t * 0.13).sin() * 0.7;
        v += ((dx * 0.8 - dy * 0.6) * 0.055 + t * 0.15).sin() * 0.6;

        let chop_angle = cy.atan2(cx);
        let chop_coord = chop_angle.cos() * x + chop_angle.sin() * y;
        v += (chop_coord * 0.12 - t * 0.28).sin() * 0.45;
        v += (chop_coord * 0.20 + t * 0.22).sin() * 0.30;

        let turb_x = (t * 0.02).sin() * 34.0 + self.cols as f32 * 0.48;
        let turb_y = (t * 0.018 + 2.0).sin() * 14.0 + self.rows as f32 * 0.58;
        let td = (x - turb_x) * (x - turb_x) + (y - turb_y) * (y - turb_y);
        let tb = (-td * 0.0012).exp() * 0.72;
        v += (x * 0.18 - y * 0.23 + t * 0.46).sin() * tb;

        let depth_mod = 0.45 + depth * 0.75;
        v * depth_mod
    }

    fn ripple_at(&self, x: f32, y: f32) -> f32 {
        let mut v = 0.0f32;
        for r in &self.ripples {
            let age = self.time - r.birth;
            if age > 4.0 {
                continue;
            }
            let ddx = x - r.cx;
            let ddy = (y - r.cy) * 2.0;
            let d2 = ddx * ddx + ddy * ddy;
            let max_r = age * 10.0 + 8.0;
            if d2 > max_r * max_r {
                continue;
            }
            let dist = d2.sqrt();
            let splash_r = 1.0 + age * 3.0;
            let splash = (-(dist * dist) / (splash_r * splash_r)).exp()
                * (-age * 3.0).exp()
                * r.strength
                * 1.5;
            let front = age * 10.0;
            let rd = (dist - front).abs();
            let ring = (-rd * 0.5).exp()
                * (dist * 0.6 - age * 4.0).sin()
                * (-age * 0.8).exp()
                * r.strength;
            v += splash + ring;
        }
        v
    }

    fn wake_at(&self, x: f32, y: f32) -> f32 {
        let mut v = 0.0f32;
        for w in &self.wake {
            let age = self.time - w.time;
            if age > 2.0 {
                continue;
            }
            let ddx = x - w.x;
            let ddy = (y - w.y) * 2.0;
            let d2 = ddx * ddx + ddy * ddy;
            if d2 > 80.0 {
                continue;
            }
            let fade = 1.0 - age * 0.5;
            v -= (-d2 * 0.12).exp() * fade * 1.8;
        }
        v
    }

    fn pointer_dent(&self, x: f32, y: f32) -> f32 {
        if !self.pointer_active || self.pointer_x < 0.0 {
            return 0.0;
        }
        let ddx = x - self.pointer_x;
        let ddy = (y - self.pointer_y) * 2.0;
        let d2 = ddx * ddx + ddy * ddy;
        if d2 > 100.0 {
            return 0.0;
        }
        let dip = (-d2 * 0.18).exp() * -3.0;
        let rim = (-(d2.sqrt() - 3.0).powi(2) * 0.15).exp() * 1.5;
        dip + rim
    }

    fn pointer_glow(&self, x: f32, y: f32) -> f32 {
        if !self.pointer_active || self.pointer_x < 0.0 {
            return 0.0;
        }
        let ddx = x - self.pointer_x;
        let ddy = (y - self.pointer_y) * 2.0;
        let d2 = ddx * ddx + ddy * ddy;
        if d2 > 400.0 {
            return 0.0;
        }
        (-d2 * 0.008).exp() * 1.2
    }

    fn shimmer(&self, x: f32, y: f32, t: f32) -> f32 {
        let s1 = (x * 3.77 + y * 7.13 + t * 2.5).sin();
        let s2 = (x * 5.91 - y * 2.37 + t * 3.8).sin();
        let s3 = (x * 1.23 + y * 11.9 - t * 2.0).sin();
        s1 * s2 * s3
    }

    fn dye_to_rgb(&self, dye_amount: f32, brightness: f32) -> (f32, f32, f32) {
        let d = Self::clamp01(dye_amount);
        let max_val = 170.0f32;
        let (r, b) = if d < 0.5 {
            let t = d / 0.5;
            (t * 0.6, 1.0 - t * 0.15)
        } else {
            let t = (d - 0.5) / 0.5;
            (0.6 + t * 0.4, 0.85 - t * 0.7)
        };
        (r * brightness * max_val, 0.0, b * brightness * max_val)
    }

    fn wave_char(&self, wave: f32) -> u8 {
        let c = (wave * 0.5 + 0.5).clamp(0.0, 0.999);
        let idx = (c * ASCII_RAMP.len() as f32) as usize;
        ASCII_RAMP[idx.min(ASCII_RAMP.len() - 1)]
    }

    fn render_frame(&mut self) {
        let stride = self.cols + 1;
        for y in 0..self.rows {
            for x in 0..self.cols {
                let xf = x as f32;
                let yf = y as f32;

                let mut wave = self.wave_height(xf, yf, self.time);
                let rip = self.ripple_at(xf, yf);
                wave += rip * 2.0;
                wave += self.pointer_dent(xf, yf);
                let wake_val = self.wake_at(xf, yf);
                wave += wake_val;

                let intensity = (wave * 0.5 + 0.5).clamp(0.0, 1.0);
                let mut brightness = intensity * intensity * 1.4 + intensity * 0.18;
                brightness = brightness.clamp(0.02, 1.8);

                let dye_val = self.dye[self.idx(x, y)];
                let (mut fr, fg, mut fb) = self.dye_to_rgb(dye_val, brightness);

                let sh = self.shimmer(xf, yf, self.time);
                if sh > 0.85 {
                    let spk = 1.0 + ((sh - 0.85) / 0.15) * 0.5;
                    fr = (fr * spk).min(170.0);
                    fb = (fb * spk).min(170.0);
                }

                let mut glow = 0.0f32;
                if intensity > 0.55 {
                    glow = (intensity - 0.55) / 0.45;
                }
                if sh > 0.82 {
                    glow = (glow + (sh - 0.82) * 3.0).min(1.0);
                }
                if rip.abs() > 0.08 {
                    glow = (glow + rip.abs() * 0.6).min(1.0);
                }
                if dye_val > 0.1 {
                    glow = (glow + dye_val * 0.5).min(1.0);
                }
                if wake_val.abs() > 0.1 {
                    glow = (glow + wake_val.abs() * 0.4).min(1.0);
                }
                let pg = self.pointer_glow(xf, yf);
                if pg > 0.02 {
                    glow = (glow + pg).min(1.0);
                }

                let boost = 1.0 + glow * 0.55;
                let r = (fr * boost).clamp(0.0, 255.0) as u8;
                let g = (fg * boost).clamp(0.0, 255.0) as u8;
                let b = (fb * boost).clamp(0.0, 255.0) as u8;

                let p = self.idx(x, y) * 4;
                self.rgba[p] = r;
                self.rgba[p + 1] = g;
                self.rgba[p + 2] = b;
                self.rgba[p + 3] = 255;

                self.ascii_bytes[y * stride + x] = self.wave_char(wave);
            }
        }
    }
}

// =========================================================================
// Tests  (only compiled under `cargo test --features wasm`)
// =========================================================================

#[cfg(all(test, feature = "wasm"))]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn prob_sum_is_one(probs: &[f64]) {
        let total: f64 = probs.iter().sum();
        assert!(
            approx_eq(total, 1.0),
            "probabilities sum to {} (expected 1.0)",
            total
        );
    }

    #[test]
    fn test_simulator_creation() {
        let sim = JsQuantumSimulator::new(3);
        assert_eq!(sim.num_qubits(), 3);

        let probs = sim.probabilities();
        assert_eq!(probs.len(), 8); // 2^3
        assert!(approx_eq(probs[0], 1.0)); // |000>
        prob_sum_is_one(&probs);
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut sim = JsQuantumSimulator::new(1);
        sim.h(0);

        let probs = sim.probabilities();
        assert!(approx_eq(probs[0], 0.5));
        assert!(approx_eq(probs[1], 0.5));
        prob_sum_is_one(&probs);
    }

    #[test]
    fn test_x_gate_flips() {
        let mut sim = JsQuantumSimulator::new(1);
        sim.x(0);

        let probs = sim.probabilities();
        assert!(approx_eq(probs[0], 0.0));
        assert!(approx_eq(probs[1], 1.0));
    }

    #[test]
    fn test_bell_state_probabilities() {
        let probs = bell_state();
        assert_eq!(probs.len(), 4);
        assert!(approx_eq(probs[0], 0.5));
        assert!(approx_eq(probs[1], 0.0));
        assert!(approx_eq(probs[2], 0.0));
        assert!(approx_eq(probs[3], 0.5));
        prob_sum_is_one(&probs);
    }

    #[test]
    fn test_ghz_state_probabilities() {
        let probs = ghz_state(3);
        assert_eq!(probs.len(), 8);
        // Only |000> and |111> should have probability
        assert!(approx_eq(probs[0], 0.5));
        assert!(approx_eq(probs[7], 0.5));
        for i in 1..7 {
            assert!(approx_eq(probs[i], 0.0));
        }
        prob_sum_is_one(&probs);
    }

    #[test]
    fn test_circuit_runner_bell() {
        let mut runner = JsCircuitRunner::new(2);
        runner.add_gate("h", 0, None, None);
        runner.add_gate("cnot", 1, Some(0), None);
        let probs = runner.run();
        assert!(approx_eq(probs[0], 0.5));
        assert!(approx_eq(probs[3], 0.5));
        prob_sum_is_one(&probs);
    }

    #[test]
    fn test_reset_returns_to_zero() {
        let mut sim = JsQuantumSimulator::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        // After reset, should be back to |00>
        sim.reset();
        let probs = sim.probabilities();
        assert!(approx_eq(probs[0], 1.0));
        prob_sum_is_one(&probs);
    }

    #[test]
    fn test_random_circuit_probabilities_sum() {
        let probs = random_circuit(3, 5);
        assert_eq!(probs.len(), 8);
        prob_sum_is_one(&probs);
    }

    #[test]
    fn test_viz_engine_frame_shapes() {
        let mut viz = JsQuantumVizEngine::new(80, 40);
        viz.tick(0.016);

        let ascii = viz.ascii_frame();
        let rows: Vec<&str> = ascii.lines().collect();
        assert_eq!(rows.len(), 40);
        assert!(rows.iter().all(|line| line.len() == 80));

        let rgba = viz.rgba_frame();
        assert_eq!(rgba.len(), 80 * 40 * 4);
    }

    #[test]
    fn test_viz_engine_quantum_pulse_changes_frame() {
        let mut viz = JsQuantumVizEngine::new(64, 32);
        viz.tick(0.016);
        let before = viz.ascii_frame();

        viz.inject_quantum_pulse(6.0);
        viz.tick(0.016);
        let after = viz.ascii_frame();

        assert_ne!(before, after, "pulse should alter visualization output");
    }
}
