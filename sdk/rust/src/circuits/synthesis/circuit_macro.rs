//! Circuit DSL: Fluent API for Quantum Circuit Construction
//!
//! Provides a Qiskit-like builder pattern for constructing quantum circuits
//! with minimal boilerplate. Supports gate chaining, range operations,
//! circuit composition, statistics, and ASCII visualization.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::circuit_macro::CircuitBuilder;
//!
//! // Build a Bell state circuit
//! let circuit = CircuitBuilder::new(2)
//!     .h(0)
//!     .cx(0, 1)
//!     .measure_all()
//!     .build();
//!
//! assert_eq!(circuit.gate_count(), 2);
//! assert_eq!(circuit.num_qubits(), 2);
//! ```

use std::fmt;
use std::ops::Range;

// ============================================================
// GATE OPERATION ENUM
// ============================================================

/// A single gate operation in the circuit DSL.
///
/// Each variant captures the target qubit(s), control qubit(s),
/// and any continuous parameters required by the gate.
#[derive(Clone, Debug, PartialEq)]
pub enum GateOp {
    // Single-qubit gates
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),
    Sdg(usize),
    Tdg(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
    U3(usize, f64, f64, f64),

    // Two-qubit gates
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Crx(usize, usize, f64),
    Crz(usize, usize, f64),

    // Three-qubit gates
    Ccx(usize, usize, usize),
    Cswap(usize, usize, usize),

    // Measurement
    Measure(usize, usize),

    // Barrier (scheduling hint, not a physical gate)
    Barrier(Vec<usize>),
}

impl GateOp {
    /// Returns the set of qubits this operation touches.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            GateOp::H(q)
            | GateOp::X(q)
            | GateOp::Y(q)
            | GateOp::Z(q)
            | GateOp::S(q)
            | GateOp::T(q)
            | GateOp::Sdg(q)
            | GateOp::Tdg(q)
            | GateOp::Rx(q, _)
            | GateOp::Ry(q, _)
            | GateOp::Rz(q, _)
            | GateOp::U3(q, _, _, _) => vec![*q],

            GateOp::Cx(c, t)
            | GateOp::Cz(c, t)
            | GateOp::Swap(c, t)
            | GateOp::Crx(c, t, _)
            | GateOp::Crz(c, t, _) => vec![*c, *t],

            GateOp::Ccx(c0, c1, t) | GateOp::Cswap(c0, c1, t) => vec![*c0, *c1, *t],

            GateOp::Measure(q, _) => vec![*q],
            GateOp::Barrier(qs) => qs.clone(),
        }
    }

    /// True if this is a measurement operation.
    pub fn is_measurement(&self) -> bool {
        matches!(self, GateOp::Measure(_, _))
    }

    /// True if this is a barrier (not a physical gate).
    pub fn is_barrier(&self) -> bool {
        matches!(self, GateOp::Barrier(_))
    }

    /// True if this is a two-qubit gate.
    pub fn is_two_qubit(&self) -> bool {
        matches!(
            self,
            GateOp::Cx(_, _)
                | GateOp::Cz(_, _)
                | GateOp::Swap(_, _)
                | GateOp::Crx(_, _, _)
                | GateOp::Crz(_, _, _)
        )
    }

    /// True if this is a three-qubit gate.
    pub fn is_three_qubit(&self) -> bool {
        matches!(self, GateOp::Ccx(_, _, _) | GateOp::Cswap(_, _, _))
    }

    /// Short label for ASCII drawing.
    fn label(&self) -> String {
        match self {
            GateOp::H(_) => "H".into(),
            GateOp::X(_) => "X".into(),
            GateOp::Y(_) => "Y".into(),
            GateOp::Z(_) => "Z".into(),
            GateOp::S(_) => "S".into(),
            GateOp::T(_) => "T".into(),
            GateOp::Sdg(_) => "Sdg".into(),
            GateOp::Tdg(_) => "Tdg".into(),
            GateOp::Rx(_, theta) => format!("Rx({:.2})", theta),
            GateOp::Ry(_, theta) => format!("Ry({:.2})", theta),
            GateOp::Rz(_, theta) => format!("Rz({:.2})", theta),
            GateOp::U3(_, t, p, l) => format!("U3({:.1},{:.1},{:.1})", t, p, l),
            GateOp::Cx(_, _) => "CX".into(),
            GateOp::Cz(_, _) => "CZ".into(),
            GateOp::Swap(_, _) => "SWAP".into(),
            GateOp::Crx(_, _, theta) => format!("CRx({:.2})", theta),
            GateOp::Crz(_, _, theta) => format!("CRz({:.2})", theta),
            GateOp::Ccx(_, _, _) => "CCX".into(),
            GateOp::Cswap(_, _, _) => "CSWAP".into(),
            GateOp::Measure(_, _) => "M".into(),
            GateOp::Barrier(_) => "|||".into(),
        }
    }

    /// Produce the inverse (adjoint) of this gate operation.
    /// Returns None for measurements and barriers which have no inverse.
    pub fn inverse(&self) -> Option<GateOp> {
        match self {
            // Self-inverse gates
            GateOp::H(q) => Some(GateOp::H(*q)),
            GateOp::X(q) => Some(GateOp::X(*q)),
            GateOp::Y(q) => Some(GateOp::Y(*q)),
            GateOp::Z(q) => Some(GateOp::Z(*q)),
            GateOp::Cx(c, t) => Some(GateOp::Cx(*c, *t)),
            GateOp::Cz(c, t) => Some(GateOp::Cz(*c, *t)),
            GateOp::Swap(a, b) => Some(GateOp::Swap(*a, *b)),
            GateOp::Ccx(c0, c1, t) => Some(GateOp::Ccx(*c0, *c1, *t)),
            GateOp::Cswap(c, a, b) => Some(GateOp::Cswap(*c, *a, *b)),

            // S <-> Sdg
            GateOp::S(q) => Some(GateOp::Sdg(*q)),
            GateOp::Sdg(q) => Some(GateOp::S(*q)),

            // T <-> Tdg
            GateOp::T(q) => Some(GateOp::Tdg(*q)),
            GateOp::Tdg(q) => Some(GateOp::T(*q)),

            // Rotation inverses: negate the angle
            GateOp::Rx(q, theta) => Some(GateOp::Rx(*q, -theta)),
            GateOp::Ry(q, theta) => Some(GateOp::Ry(*q, -theta)),
            GateOp::Rz(q, theta) => Some(GateOp::Rz(*q, -theta)),
            GateOp::U3(q, theta, phi, lambda) => Some(GateOp::U3(*q, -theta, -lambda, -phi)),
            GateOp::Crx(c, t, theta) => Some(GateOp::Crx(*c, *t, -theta)),
            GateOp::Crz(c, t, theta) => Some(GateOp::Crz(*c, *t, -theta)),

            // Measurements and barriers have no inverse
            GateOp::Measure(_, _) | GateOp::Barrier(_) => None,
        }
    }
}

impl fmt::Display for GateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ============================================================
// CIRCUIT STATISTICS
// ============================================================

/// Computed statistics for a built circuit.
#[derive(Clone, Debug)]
pub struct CircuitStats {
    /// Total circuit depth (maximum number of sequential time steps).
    pub depth: usize,
    /// Total number of gate operations (excluding barriers and measurements).
    pub gate_count: usize,
    /// Number of two-qubit gates.
    pub two_qubit_count: usize,
    /// Number of three-qubit gates.
    pub three_qubit_count: usize,
    /// Number of T and Tdg gates (important for fault-tolerant cost).
    pub t_count: usize,
    /// Number of CNOT (CX) gates.
    pub cnot_count: usize,
    /// Number of measurement operations.
    pub measurement_count: usize,
}

// ============================================================
// CIRCUIT DRAWING
// ============================================================

/// ASCII circuit diagram.
pub struct CircuitDrawing {
    text: String,
}

impl CircuitDrawing {
    /// Get the drawing as a string reference.
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for CircuitDrawing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

// ============================================================
// BUILT CIRCUIT
// ============================================================

/// A fully constructed quantum circuit produced by [`CircuitBuilder`].
///
/// Contains the ordered list of gate operations, qubit count, and
/// classical bit count. Immutable after construction.
#[derive(Clone, Debug)]
pub struct BuiltCircuit {
    gates: Vec<GateOp>,
    num_qubits: usize,
    num_clbits: usize,
}

impl BuiltCircuit {
    /// Number of qubits in this circuit.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of classical bits in this circuit.
    pub fn num_clbits(&self) -> usize {
        self.num_clbits
    }

    /// Ordered slice of gate operations.
    pub fn gates(&self) -> &[GateOp] {
        &self.gates
    }

    /// Total number of gate operations (excluding barriers and measurements).
    pub fn gate_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| !g.is_barrier() && !g.is_measurement())
            .count()
    }

    /// Number of two-qubit gates.
    pub fn two_qubit_count(&self) -> usize {
        self.gates.iter().filter(|g| g.is_two_qubit()).count()
    }

    /// Number of T and Tdg gates.
    pub fn t_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, GateOp::T(_) | GateOp::Tdg(_)))
            .count()
    }

    /// Number of CNOT (CX) gates.
    pub fn cnot_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, GateOp::Cx(_, _)))
            .count()
    }

    /// Compute the circuit depth.
    ///
    /// Depth is the number of sequential time steps required, where
    /// gates on disjoint qubits can execute in the same step.
    pub fn depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }

        // Track the earliest available time step for each qubit.
        let mut qubit_time: Vec<usize> = vec![0; self.num_qubits];

        for gate in &self.gates {
            if gate.is_barrier() {
                // Barriers synchronize all involved qubits.
                let qs = gate.qubits();
                if !qs.is_empty() {
                    let max_t = qs.iter().map(|&q| qubit_time[q]).max().unwrap_or(0);
                    for &q in &qs {
                        qubit_time[q] = max_t;
                    }
                }
                continue;
            }

            let qs = gate.qubits();
            let start = qs.iter().map(|&q| qubit_time[q]).max().unwrap_or(0);
            for &q in &qs {
                qubit_time[q] = start + 1;
            }
        }

        qubit_time.into_iter().max().unwrap_or(0)
    }

    /// Compute comprehensive circuit statistics.
    pub fn stats(&self) -> CircuitStats {
        CircuitStats {
            depth: self.depth(),
            gate_count: self.gate_count(),
            two_qubit_count: self.two_qubit_count(),
            three_qubit_count: self.gates.iter().filter(|g| g.is_three_qubit()).count(),
            t_count: self.t_count(),
            cnot_count: self.cnot_count(),
            measurement_count: self.gates.iter().filter(|g| g.is_measurement()).count(),
        }
    }

    /// Generate an ASCII circuit diagram.
    ///
    /// Produces a simple text-based visualization of the circuit with
    /// qubit wires and gate labels. Multi-qubit gates show control
    /// and target connections.
    pub fn draw(&self) -> CircuitDrawing {
        if self.num_qubits == 0 {
            return CircuitDrawing {
                text: String::new(),
            };
        }

        // Each qubit gets a line. We build columns of gate labels.
        let mut columns: Vec<Vec<String>> = Vec::new();

        for gate in &self.gates {
            let mut col: Vec<String> = vec!["---".to_string(); self.num_qubits];

            match gate {
                GateOp::Cx(c, t) => {
                    col[*c] = "-*-".to_string();
                    col[*t] = "-X-".to_string();
                    // Draw vertical connections
                    let (lo, hi) = if c < t { (*c, *t) } else { (*t, *c) };
                    for q in (lo + 1)..hi {
                        col[q] = "-|-".to_string();
                    }
                }
                GateOp::Cz(c, t) => {
                    col[*c] = "-*-".to_string();
                    col[*t] = "-Z-".to_string();
                    let (lo, hi) = if c < t { (*c, *t) } else { (*t, *c) };
                    for q in (lo + 1)..hi {
                        col[q] = "-|-".to_string();
                    }
                }
                GateOp::Ccx(c0, c1, t) => {
                    col[*c0] = "-*-".to_string();
                    col[*c1] = "-*-".to_string();
                    col[*t] = "-X-".to_string();
                    let qs = vec![*c0, *c1, *t];
                    let lo = *qs.iter().min().unwrap();
                    let hi = *qs.iter().max().unwrap();
                    for q in (lo + 1)..hi {
                        if !qs.contains(&q) {
                            col[q] = "-|-".to_string();
                        }
                    }
                }
                GateOp::Barrier(qs) => {
                    for &q in qs {
                        if q < self.num_qubits {
                            col[q] = "|||".to_string();
                        }
                    }
                }
                _ => {
                    let qs = gate.qubits();
                    if let Some(&q) = qs.first() {
                        let lbl = gate.label();
                        col[q] = format!("[{}]", lbl);
                    }
                }
            }

            columns.push(col);
        }

        // Assemble lines
        let mut lines: Vec<String> = Vec::with_capacity(self.num_qubits);
        for q in 0..self.num_qubits {
            let mut line = format!("q{}: ", q);
            for col in &columns {
                line.push_str(&col[q]);
            }
            line.push_str("---");
            lines.push(line);
        }

        CircuitDrawing {
            text: lines.join("\n"),
        }
    }
}

impl fmt::Display for BuiltCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BuiltCircuit(qubits={}, gates={}, depth={})",
            self.num_qubits,
            self.gate_count(),
            self.depth()
        )
    }
}

// ============================================================
// CIRCUIT BUILDER ERROR
// ============================================================

/// Errors that can occur during circuit construction.
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitError {
    /// A qubit index exceeds the circuit's qubit count.
    QubitOutOfBounds { qubit: usize, num_qubits: usize },
    /// A classical bit index exceeds the circuit's classical bit count.
    ClbitOutOfBounds { clbit: usize, num_clbits: usize },
    /// Two distinct qubits were expected but the same qubit was given.
    DuplicateQubit { qubit: usize },
    /// The appended circuit has more qubits than the current one.
    IncompatibleQubitCount { expected: usize, actual: usize },
}

impl fmt::Display for CircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitError::QubitOutOfBounds { qubit, num_qubits } => {
                write!(
                    f,
                    "qubit index {} out of bounds for {}-qubit circuit",
                    qubit, num_qubits
                )
            }
            CircuitError::ClbitOutOfBounds { clbit, num_clbits } => {
                write!(
                    f,
                    "classical bit index {} out of bounds (have {} clbits)",
                    clbit, num_clbits
                )
            }
            CircuitError::DuplicateQubit { qubit } => {
                write!(f, "duplicate qubit index {}", qubit)
            }
            CircuitError::IncompatibleQubitCount { expected, actual } => {
                write!(
                    f,
                    "cannot append {}-qubit circuit to {}-qubit circuit",
                    actual, expected
                )
            }
        }
    }
}

impl std::error::Error for CircuitError {}

// ============================================================
// CIRCUIT BUILDER
// ============================================================

/// Fluent API for constructing quantum circuits.
///
/// Validates qubit indices at each step and collects gate operations
/// into a [`BuiltCircuit`] on `.build()`.
///
/// # Example
///
/// ```ignore
/// use nqpu_metal::circuit_macro::CircuitBuilder;
///
/// let ghz = CircuitBuilder::new(3)
///     .h(0)
///     .cx(0, 1)
///     .cx(1, 2)
///     .measure_all()
///     .build();
///
/// assert_eq!(ghz.gate_count(), 3);
/// assert_eq!(ghz.depth(), 3);
/// ```
pub struct CircuitBuilder {
    num_qubits: usize,
    num_clbits: usize,
    gates: Vec<GateOp>,
    errors: Vec<CircuitError>,
}

impl CircuitBuilder {
    /// Create a new builder for a circuit with `n_qubits` qubits.
    ///
    /// Classical bits start at zero and are allocated on demand by
    /// `measure` or `measure_all`.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            num_qubits: n_qubits,
            num_clbits: 0,
            gates: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Create a new builder with explicit qubit and classical bit counts.
    pub fn with_clbits(n_qubits: usize, n_clbits: usize) -> Self {
        Self {
            num_qubits: n_qubits,
            num_clbits: n_clbits,
            gates: Vec::new(),
            errors: Vec::new(),
        }
    }

    // -------------------------------------------------------
    // Validation helpers
    // -------------------------------------------------------

    fn check_qubit(&mut self, q: usize) -> bool {
        if q >= self.num_qubits {
            self.errors.push(CircuitError::QubitOutOfBounds {
                qubit: q,
                num_qubits: self.num_qubits,
            });
            false
        } else {
            true
        }
    }

    fn check_two_qubits(&mut self, a: usize, b: usize) -> bool {
        let mut ok = true;
        if a >= self.num_qubits {
            self.errors.push(CircuitError::QubitOutOfBounds {
                qubit: a,
                num_qubits: self.num_qubits,
            });
            ok = false;
        }
        if b >= self.num_qubits {
            self.errors.push(CircuitError::QubitOutOfBounds {
                qubit: b,
                num_qubits: self.num_qubits,
            });
            ok = false;
        }
        if a == b && ok {
            self.errors.push(CircuitError::DuplicateQubit { qubit: a });
            ok = false;
        }
        ok
    }

    fn check_three_qubits(&mut self, a: usize, b: usize, c: usize) -> bool {
        let mut ok = true;
        for &q in &[a, b, c] {
            if q >= self.num_qubits {
                self.errors.push(CircuitError::QubitOutOfBounds {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
                ok = false;
            }
        }
        if ok {
            if a == b || a == c || b == c {
                let dup = if a == b {
                    a
                } else if a == c {
                    a
                } else {
                    b
                };
                self.errors
                    .push(CircuitError::DuplicateQubit { qubit: dup });
                ok = false;
            }
        }
        ok
    }

    // -------------------------------------------------------
    // Single-qubit gates
    // -------------------------------------------------------

    /// Apply a Hadamard gate to `qubit`.
    pub fn h(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::H(qubit));
        }
        self
    }

    /// Apply a Pauli-X gate to `qubit`.
    pub fn x(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::X(qubit));
        }
        self
    }

    /// Apply a Pauli-Y gate to `qubit`.
    pub fn y(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::Y(qubit));
        }
        self
    }

    /// Apply a Pauli-Z gate to `qubit`.
    pub fn z(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::Z(qubit));
        }
        self
    }

    /// Apply an S gate (phase gate, pi/2 rotation) to `qubit`.
    pub fn s(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::S(qubit));
        }
        self
    }

    /// Apply a T gate (pi/8 gate) to `qubit`.
    pub fn t(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::T(qubit));
        }
        self
    }

    /// Apply an S-dagger gate to `qubit`.
    pub fn sdg(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::Sdg(qubit));
        }
        self
    }

    /// Apply a T-dagger gate to `qubit`.
    pub fn tdg(mut self, qubit: usize) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::Tdg(qubit));
        }
        self
    }

    /// Apply an Rx rotation by `theta` radians to `qubit`.
    pub fn rx(mut self, qubit: usize, theta: f64) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::Rx(qubit, theta));
        }
        self
    }

    /// Apply an Ry rotation by `theta` radians to `qubit`.
    pub fn ry(mut self, qubit: usize, theta: f64) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::Ry(qubit, theta));
        }
        self
    }

    /// Apply an Rz rotation by `theta` radians to `qubit`.
    pub fn rz(mut self, qubit: usize, theta: f64) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::Rz(qubit, theta));
        }
        self
    }

    /// Apply a general U3 gate with parameters (theta, phi, lambda) to `qubit`.
    pub fn u3(mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> Self {
        if self.check_qubit(qubit) {
            self.gates.push(GateOp::U3(qubit, theta, phi, lambda));
        }
        self
    }

    // -------------------------------------------------------
    // Two-qubit gates
    // -------------------------------------------------------

    /// Apply a CNOT (controlled-X) gate with `control` and `target`.
    pub fn cx(mut self, control: usize, target: usize) -> Self {
        if self.check_two_qubits(control, target) {
            self.gates.push(GateOp::Cx(control, target));
        }
        self
    }

    /// Apply a controlled-Z gate with `control` and `target`.
    pub fn cz(mut self, control: usize, target: usize) -> Self {
        if self.check_two_qubits(control, target) {
            self.gates.push(GateOp::Cz(control, target));
        }
        self
    }

    /// Apply a SWAP gate between `q0` and `q1`.
    pub fn swap(mut self, q0: usize, q1: usize) -> Self {
        if self.check_two_qubits(q0, q1) {
            self.gates.push(GateOp::Swap(q0, q1));
        }
        self
    }

    /// Apply a controlled-Rx rotation with `control`, `target`, and angle `theta`.
    pub fn crx(mut self, control: usize, target: usize, theta: f64) -> Self {
        if self.check_two_qubits(control, target) {
            self.gates.push(GateOp::Crx(control, target, theta));
        }
        self
    }

    /// Apply a controlled-Rz rotation with `control`, `target`, and angle `theta`.
    pub fn crz(mut self, control: usize, target: usize, theta: f64) -> Self {
        if self.check_two_qubits(control, target) {
            self.gates.push(GateOp::Crz(control, target, theta));
        }
        self
    }

    // -------------------------------------------------------
    // Three-qubit gates
    // -------------------------------------------------------

    /// Apply a Toffoli (CCX) gate with controls `c0`, `c1` and `target`.
    pub fn ccx(mut self, c0: usize, c1: usize, target: usize) -> Self {
        if self.check_three_qubits(c0, c1, target) {
            self.gates.push(GateOp::Ccx(c0, c1, target));
        }
        self
    }

    /// Apply a Fredkin (CSWAP) gate with `control` swapping `q0` and `q1`.
    pub fn cswap(mut self, control: usize, q0: usize, q1: usize) -> Self {
        if self.check_three_qubits(control, q0, q1) {
            self.gates.push(GateOp::Cswap(control, q0, q1));
        }
        self
    }

    // -------------------------------------------------------
    // Batch / range operations
    // -------------------------------------------------------

    /// Apply a Hadamard gate to every qubit.
    pub fn h_all(mut self) -> Self {
        for q in 0..self.num_qubits {
            self.gates.push(GateOp::H(q));
        }
        self
    }

    /// Apply a Pauli-X gate to every qubit.
    pub fn x_all(mut self) -> Self {
        for q in 0..self.num_qubits {
            self.gates.push(GateOp::X(q));
        }
        self
    }

    /// Apply a Hadamard gate to each qubit in `range`.
    pub fn h_range(mut self, range: Range<usize>) -> Self {
        for q in range {
            if self.check_qubit(q) {
                self.gates.push(GateOp::H(q));
            }
        }
        self
    }

    /// Apply a Pauli-X gate to each qubit in `range`.
    pub fn x_range(mut self, range: Range<usize>) -> Self {
        for q in range {
            if self.check_qubit(q) {
                self.gates.push(GateOp::X(q));
            }
        }
        self
    }

    // -------------------------------------------------------
    // Measurement
    // -------------------------------------------------------

    /// Measure `qubit` into classical bit `cbit`.
    pub fn measure(mut self, qubit: usize, cbit: usize) -> Self {
        if self.check_qubit(qubit) {
            // Auto-expand classical register if needed.
            if cbit >= self.num_clbits {
                self.num_clbits = cbit + 1;
            }
            self.gates.push(GateOp::Measure(qubit, cbit));
        }
        self
    }

    /// Measure all qubits into classical bits 0..n_qubits.
    pub fn measure_all(mut self) -> Self {
        self.num_clbits = self.num_clbits.max(self.num_qubits);
        for q in 0..self.num_qubits {
            self.gates.push(GateOp::Measure(q, q));
        }
        self
    }

    // -------------------------------------------------------
    // Barriers
    // -------------------------------------------------------

    /// Insert a barrier on all qubits.
    pub fn barrier_all(mut self) -> Self {
        let qs: Vec<usize> = (0..self.num_qubits).collect();
        self.gates.push(GateOp::Barrier(qs));
        self
    }

    /// Insert a barrier on the specified qubits.
    pub fn barrier(mut self) -> Self {
        let qs: Vec<usize> = (0..self.num_qubits).collect();
        self.gates.push(GateOp::Barrier(qs));
        self
    }

    /// Insert a barrier on a specific set of qubits.
    pub fn barrier_qubits(mut self, qubits: &[usize]) -> Self {
        for &q in qubits {
            if q >= self.num_qubits {
                self.errors.push(CircuitError::QubitOutOfBounds {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
                return self;
            }
        }
        self.gates.push(GateOp::Barrier(qubits.to_vec()));
        self
    }

    // -------------------------------------------------------
    // Circuit composition
    // -------------------------------------------------------

    /// Append all gates from `other` to this circuit.
    ///
    /// The other circuit must have at most `num_qubits` qubits.
    pub fn append(mut self, other: &BuiltCircuit) -> Self {
        if other.num_qubits > self.num_qubits {
            self.errors.push(CircuitError::IncompatibleQubitCount {
                expected: self.num_qubits,
                actual: other.num_qubits,
            });
            return self;
        }
        self.gates.extend(other.gates.iter().cloned());
        self.num_clbits = self.num_clbits.max(other.num_clbits);
        self
    }

    /// Repeat the current circuit `n` times.
    ///
    /// The existing gates are repeated in order.
    pub fn repeat(mut self, n: usize) -> Self {
        if n == 0 {
            self.gates.clear();
            return self;
        }
        let original = self.gates.clone();
        for _ in 1..n {
            self.gates.extend(original.iter().cloned());
        }
        self
    }

    /// Append the inverse (adjoint) of the current circuit.
    ///
    /// Reverses the gate order and takes the adjoint of each gate.
    /// Measurements and barriers are skipped in the inverse.
    pub fn inverse(mut self) -> Self {
        let mut inv_gates: Vec<GateOp> = Vec::new();
        for gate in self.gates.iter().rev() {
            if let Some(inv) = gate.inverse() {
                inv_gates.push(inv);
            }
        }
        self.gates = inv_gates;
        self
    }

    // -------------------------------------------------------
    // Analysis (on the builder, before build)
    // -------------------------------------------------------

    /// Compute depth of the circuit so far.
    pub fn depth(&self) -> usize {
        // Build a temporary circuit to reuse the depth calculation.
        let tmp = BuiltCircuit {
            gates: self.gates.clone(),
            num_qubits: self.num_qubits,
            num_clbits: self.num_clbits,
        };
        tmp.depth()
    }

    /// Count gates so far (excluding barriers and measurements).
    pub fn current_gate_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| !g.is_barrier() && !g.is_measurement())
            .count()
    }

    /// Compute stats on the circuit so far.
    pub fn stats(&self) -> CircuitStats {
        let tmp = BuiltCircuit {
            gates: self.gates.clone(),
            num_qubits: self.num_qubits,
            num_clbits: self.num_clbits,
        };
        tmp.stats()
    }

    /// Generate an ASCII drawing of the circuit so far.
    pub fn draw(&self) -> CircuitDrawing {
        let tmp = BuiltCircuit {
            gates: self.gates.clone(),
            num_qubits: self.num_qubits,
            num_clbits: self.num_clbits,
        };
        tmp.draw()
    }

    /// Return any validation errors accumulated so far.
    pub fn errors(&self) -> &[CircuitError] {
        &self.errors
    }

    /// True if the builder has accumulated validation errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    // -------------------------------------------------------
    // Build
    // -------------------------------------------------------

    /// Consume the builder and produce a [`BuiltCircuit`].
    ///
    /// Returns `Err` if any validation errors were accumulated during
    /// construction.
    pub fn try_build(self) -> Result<BuiltCircuit, Vec<CircuitError>> {
        if self.errors.is_empty() {
            Ok(BuiltCircuit {
                gates: self.gates,
                num_qubits: self.num_qubits,
                num_clbits: self.num_clbits,
            })
        } else {
            Err(self.errors)
        }
    }

    /// Consume the builder and produce a [`BuiltCircuit`].
    ///
    /// Panics if there are validation errors. For fallible construction,
    /// use [`try_build`](Self::try_build).
    pub fn build(self) -> BuiltCircuit {
        if !self.errors.is_empty() {
            panic!(
                "CircuitBuilder has {} error(s): {:?}",
                self.errors.len(),
                self.errors
            );
        }
        BuiltCircuit {
            gates: self.gates,
            num_qubits: self.num_qubits,
            num_clbits: self.num_clbits,
        }
    }
}

// ============================================================
// QUBIT RANGE HELPER
// ============================================================

/// Helper for specifying ranges of qubits in batch operations.
///
/// This is a thin wrapper around `Range<usize>` providing a named
/// constructor for clarity in circuit-building code.
#[derive(Clone, Debug)]
pub struct QubitRange {
    /// Start qubit (inclusive).
    pub start: usize,
    /// End qubit (exclusive).
    pub end: usize,
}

impl QubitRange {
    /// Create a new qubit range [start, end).
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Convert to a standard `Range<usize>`.
    pub fn to_range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Number of qubits in the range.
    pub fn len(&self) -> usize {
        if self.end > self.start {
            self.end - self.start
        } else {
            0
        }
    }

    /// True if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl From<Range<usize>> for QubitRange {
    fn from(r: Range<usize>) -> Self {
        Self {
            start: r.start,
            end: r.end,
        }
    }
}

impl IntoIterator for QubitRange {
    type Item = usize;
    type IntoIter = Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.start..self.end
    }
}

// ============================================================
// DECLARATIVE MACRO
// ============================================================

/// Declarative macro for building quantum circuits with less boilerplate.
///
/// Single-qubit gates take a bare literal argument. Multi-argument gates
/// use parenthesized arguments to avoid macro ambiguity with semicolons.
/// Zero-argument operations stand alone.
///
/// # Example
///
/// ```ignore
/// use nqpu_metal::circuit;
///
/// let bell = circuit!(2;
///     h 0;
///     cx(0, 1);
///     measure_all;
/// );
/// assert_eq!(bell.gate_count(), 2);
/// ```
#[macro_export]
macro_rules! circuit {
    // Entry point: qubit count followed by semicolon-separated operations.
    ($n:expr; $( $rest:tt )* ) => {{
        let _builder = $crate::circuit_macro::CircuitBuilder::new($n);
        let _builder = $crate::circuit!(@munch _builder; $( $rest )* );
        _builder.build()
    }};

    // -- Base case: no more tokens --------------------------------
    (@munch $b:expr; ) => { $b };

    // -- Zero-argument operations ---------------------------------
    (@munch $b:expr; measure_all ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.measure_all(); $( $rest )* )
    };
    (@munch $b:expr; measure_all ) => { $b.measure_all() };

    (@munch $b:expr; barrier ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.barrier(); $( $rest )* )
    };
    (@munch $b:expr; barrier ) => { $b.barrier() };

    (@munch $b:expr; barrier_all ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.barrier_all(); $( $rest )* )
    };
    (@munch $b:expr; barrier_all ) => { $b.barrier_all() };

    (@munch $b:expr; h_all ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.h_all(); $( $rest )* )
    };
    (@munch $b:expr; h_all ) => { $b.h_all() };

    (@munch $b:expr; x_all ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.x_all(); $( $rest )* )
    };
    (@munch $b:expr; x_all ) => { $b.x_all() };

    // -- Single-argument (bare literal) ---------------------------
    (@munch $b:expr; h $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.h($q); $( $rest )* )
    };
    (@munch $b:expr; h $q:literal ) => { $b.h($q) };

    (@munch $b:expr; x $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.x($q); $( $rest )* )
    };
    (@munch $b:expr; x $q:literal ) => { $b.x($q) };

    (@munch $b:expr; y $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.y($q); $( $rest )* )
    };
    (@munch $b:expr; y $q:literal ) => { $b.y($q) };

    (@munch $b:expr; z $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.z($q); $( $rest )* )
    };
    (@munch $b:expr; z $q:literal ) => { $b.z($q) };

    (@munch $b:expr; s $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.s($q); $( $rest )* )
    };
    (@munch $b:expr; s $q:literal ) => { $b.s($q) };

    (@munch $b:expr; t $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.t($q); $( $rest )* )
    };
    (@munch $b:expr; t $q:literal ) => { $b.t($q) };

    (@munch $b:expr; sdg $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.sdg($q); $( $rest )* )
    };
    (@munch $b:expr; sdg $q:literal ) => { $b.sdg($q) };

    (@munch $b:expr; tdg $q:literal ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.tdg($q); $( $rest )* )
    };
    (@munch $b:expr; tdg $q:literal ) => { $b.tdg($q) };

    // -- Multi-argument (parenthesized) ---------------------------
    (@munch $b:expr; cx ( $c:expr, $t:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.cx($c, $t); $( $rest )* )
    };
    (@munch $b:expr; cx ( $c:expr, $t:expr ) ) => { $b.cx($c, $t) };

    (@munch $b:expr; cz ( $c:expr, $t:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.cz($c, $t); $( $rest )* )
    };
    (@munch $b:expr; cz ( $c:expr, $t:expr ) ) => { $b.cz($c, $t) };

    (@munch $b:expr; swap ( $a:expr, $b2:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.swap($a, $b2); $( $rest )* )
    };
    (@munch $b:expr; swap ( $a:expr, $b2:expr ) ) => { $b.swap($a, $b2) };

    (@munch $b:expr; rx ( $q:expr, $theta:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.rx($q, $theta); $( $rest )* )
    };
    (@munch $b:expr; rx ( $q:expr, $theta:expr ) ) => { $b.rx($q, $theta) };

    (@munch $b:expr; ry ( $q:expr, $theta:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.ry($q, $theta); $( $rest )* )
    };
    (@munch $b:expr; ry ( $q:expr, $theta:expr ) ) => { $b.ry($q, $theta) };

    (@munch $b:expr; rz ( $q:expr, $theta:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.rz($q, $theta); $( $rest )* )
    };
    (@munch $b:expr; rz ( $q:expr, $theta:expr ) ) => { $b.rz($q, $theta) };

    (@munch $b:expr; measure ( $q:expr, $c:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.measure($q, $c); $( $rest )* )
    };
    (@munch $b:expr; measure ( $q:expr, $c:expr ) ) => { $b.measure($q, $c) };

    (@munch $b:expr; crx ( $c:expr, $t:expr, $theta:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.crx($c, $t, $theta); $( $rest )* )
    };
    (@munch $b:expr; crx ( $c:expr, $t:expr, $theta:expr ) ) => { $b.crx($c, $t, $theta) };

    (@munch $b:expr; crz ( $c:expr, $t:expr, $theta:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.crz($c, $t, $theta); $( $rest )* )
    };
    (@munch $b:expr; crz ( $c:expr, $t:expr, $theta:expr ) ) => { $b.crz($c, $t, $theta) };

    (@munch $b:expr; ccx ( $c0:expr, $c1:expr, $t:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.ccx($c0, $c1, $t); $( $rest )* )
    };
    (@munch $b:expr; ccx ( $c0:expr, $c1:expr, $t:expr ) ) => { $b.ccx($c0, $c1, $t) };

    (@munch $b:expr; cswap ( $c:expr, $a:expr, $b2:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.cswap($c, $a, $b2); $( $rest )* )
    };
    (@munch $b:expr; cswap ( $c:expr, $a:expr, $b2:expr ) ) => { $b.cswap($c, $a, $b2) };

    (@munch $b:expr; u3 ( $q:expr, $t:expr, $p:expr, $l:expr ) ; $( $rest:tt )* ) => {
        $crate::circuit!(@munch $b.u3($q, $t, $p, $l); $( $rest )* )
    };
    (@munch $b:expr; u3 ( $q:expr, $t:expr, $p:expr, $l:expr ) ) => { $b.u3($q, $t, $p, $l) };
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn empty_circuit() {
        let circuit = CircuitBuilder::new(3).build();
        assert_eq!(circuit.gate_count(), 0);
        assert_eq!(circuit.num_qubits(), 3);
        assert_eq!(circuit.num_clbits(), 0);
        assert_eq!(circuit.depth(), 0);
        assert!(circuit.gates().is_empty());
    }

    #[test]
    fn single_h() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        assert_eq!(circuit.gate_count(), 1);
        assert_eq!(circuit.gates().len(), 1);
        assert_eq!(circuit.gates()[0], GateOp::H(0));
        assert_eq!(circuit.depth(), 1);
    }

    #[test]
    fn bell_state() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        assert_eq!(circuit.gate_count(), 2);
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.depth(), 2);
        assert_eq!(circuit.cnot_count(), 1);
        assert_eq!(circuit.gates()[0], GateOp::H(0));
        assert_eq!(circuit.gates()[1], GateOp::Cx(0, 1));
    }

    #[test]
    fn ghz_state() {
        let circuit = CircuitBuilder::new(3).h(0).cx(0, 1).cx(1, 2).build();
        assert_eq!(circuit.gate_count(), 3);
        assert_eq!(circuit.num_qubits(), 3);
        assert_eq!(circuit.depth(), 3);
        assert_eq!(circuit.cnot_count(), 2);
        assert_eq!(circuit.two_qubit_count(), 2);
    }

    #[test]
    fn fluent_chaining() {
        let circuit = CircuitBuilder::new(4)
            .h(0)
            .h(1)
            .cx(0, 2)
            .cx(1, 3)
            .rz(0, PI / 4.0)
            .t(1)
            .s(2)
            .sdg(3)
            .measure_all()
            .build();

        // 8 gates: 2 H + 2 CX + 1 Rz + 1 T + 1 S + 1 Sdg
        assert_eq!(circuit.gate_count(), 8);
        assert_eq!(circuit.num_qubits(), 4);
        assert_eq!(circuit.num_clbits(), 4);
    }

    #[test]
    fn h_all() {
        let circuit = CircuitBuilder::new(5).h_all().build();
        assert_eq!(circuit.gate_count(), 5);
        assert_eq!(circuit.depth(), 1);
        for (i, gate) in circuit.gates().iter().enumerate() {
            assert_eq!(*gate, GateOp::H(i));
        }
    }

    #[test]
    fn measure_all() {
        let circuit = CircuitBuilder::new(4).h_all().measure_all().build();
        assert_eq!(circuit.num_clbits(), 4);
        let measurements: Vec<_> = circuit
            .gates()
            .iter()
            .filter(|g| g.is_measurement())
            .collect();
        assert_eq!(measurements.len(), 4);
        for (i, m) in measurements.iter().enumerate() {
            assert_eq!(**m, GateOp::Measure(i, i));
        }
    }

    #[test]
    fn circuit_depth() {
        // Two parallel H gates = depth 1
        let c1 = CircuitBuilder::new(2).h(0).h(1).build();
        assert_eq!(c1.depth(), 1);

        // Sequential gates on same qubit = depth 2
        let c2 = CircuitBuilder::new(1).h(0).x(0).build();
        assert_eq!(c2.depth(), 2);

        // Mixed: H on q0, H on q1 (parallel, depth 1), then CX(0,1) (depth 2)
        let c3 = CircuitBuilder::new(2).h(0).h(1).cx(0, 1).build();
        assert_eq!(c3.depth(), 2);

        // Chain of CX gates on overlapping qubits
        let c4 = CircuitBuilder::new(3).cx(0, 1).cx(1, 2).build();
        assert_eq!(c4.depth(), 2);
    }

    #[test]
    fn gate_count() {
        let circuit = CircuitBuilder::new(3)
            .h(0)
            .x(1)
            .cx(0, 1)
            .ccx(0, 1, 2)
            .build();

        assert_eq!(circuit.gate_count(), 4);
        assert_eq!(circuit.two_qubit_count(), 1);
        assert_eq!(circuit.cnot_count(), 1);
    }

    #[test]
    fn t_count() {
        let circuit = CircuitBuilder::new(2).t(0).t(1).tdg(0).h(0).t(1).build();

        assert_eq!(circuit.t_count(), 4); // 2 T + 1 Tdg + 1 T
        assert_eq!(circuit.gate_count(), 5);
    }

    #[test]
    fn circuit_stats() {
        let circuit = CircuitBuilder::new(3)
            .h(0)
            .t(0)
            .cx(0, 1)
            .ccx(0, 1, 2)
            .tdg(2)
            .measure_all()
            .build();

        let stats = circuit.stats();
        assert_eq!(stats.gate_count, 5); // H + T + CX + CCX + Tdg
        assert_eq!(stats.two_qubit_count, 1); // CX
        assert_eq!(stats.three_qubit_count, 1); // CCX
        assert_eq!(stats.t_count, 2); // T + Tdg
        assert_eq!(stats.cnot_count, 1); // CX
        assert_eq!(stats.measurement_count, 3); // measure_all on 3 qubits
        assert!(stats.depth >= 4); // At least 4 sequential steps
    }

    #[test]
    fn ascii_drawing() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let drawing = circuit.draw();
        let text = drawing.as_str();

        // Must be a non-empty multi-line string
        assert!(!text.is_empty());
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2); // 2 qubits = 2 lines
        assert!(lines[0].contains("q0:"));
        assert!(lines[1].contains("q1:"));
        assert!(lines[0].contains("H"));
    }

    #[test]
    fn circuit_inverse() {
        let circuit = CircuitBuilder::new(2)
            .h(0)
            .t(1)
            .cx(0, 1)
            .s(0)
            .inverse()
            .build();

        // Inverse reverses order and inverts each gate.
        // Original: H(0), T(1), CX(0,1), S(0)
        // Inverse:  Sdg(0), CX(0,1), Tdg(1), H(0)
        let gates = circuit.gates();
        assert_eq!(gates.len(), 4);
        assert_eq!(gates[0], GateOp::Sdg(0));
        assert_eq!(gates[1], GateOp::Cx(0, 1));
        assert_eq!(gates[2], GateOp::Tdg(1));
        assert_eq!(gates[3], GateOp::H(0));
    }

    #[test]
    fn circuit_repeat() {
        let circuit = CircuitBuilder::new(1).h(0).x(0).repeat(3).build();

        // H, X repeated 3 times = 6 gates
        assert_eq!(circuit.gate_count(), 6);
        assert_eq!(circuit.gates()[0], GateOp::H(0));
        assert_eq!(circuit.gates()[1], GateOp::X(0));
        assert_eq!(circuit.gates()[2], GateOp::H(0));
        assert_eq!(circuit.gates()[3], GateOp::X(0));
        assert_eq!(circuit.gates()[4], GateOp::H(0));
        assert_eq!(circuit.gates()[5], GateOp::X(0));
    }

    #[test]
    fn circuit_append() {
        let sub = CircuitBuilder::new(2).cx(0, 1).build();
        let circuit = CircuitBuilder::new(2).h(0).h(1).append(&sub).build();

        assert_eq!(circuit.gate_count(), 3); // 2 H + 1 CX
        assert_eq!(circuit.gates()[2], GateOp::Cx(0, 1));
    }

    #[test]
    fn out_of_bounds_error() {
        let result = CircuitBuilder::new(2).h(5).try_build();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0],
            CircuitError::QubitOutOfBounds {
                qubit: 5,
                num_qubits: 2,
            }
        );
    }

    #[test]
    fn barrier_no_gates() {
        let circuit = CircuitBuilder::new(3).h(0).barrier().h(1).build();

        // Barrier is not counted as a gate.
        assert_eq!(circuit.gate_count(), 2);
        // But it is in the gate list.
        assert_eq!(circuit.gates().len(), 3);
        assert!(circuit.gates()[1].is_barrier());
    }

    #[test]
    fn duplicate_qubit_error() {
        let result = CircuitBuilder::new(3).cx(1, 1).try_build();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, CircuitError::DuplicateQubit { qubit: 1 })));
    }

    #[test]
    fn qubit_range_operations() {
        let circuit = CircuitBuilder::new(4).h_range(1..3).build();
        assert_eq!(circuit.gate_count(), 2);
        assert_eq!(circuit.gates()[0], GateOp::H(1));
        assert_eq!(circuit.gates()[1], GateOp::H(2));
    }

    #[test]
    fn x_range() {
        let circuit = CircuitBuilder::new(4).x_range(0..4).build();
        assert_eq!(circuit.gate_count(), 4);
        for (i, gate) in circuit.gates().iter().enumerate() {
            assert_eq!(*gate, GateOp::X(i));
        }
    }

    #[test]
    fn rotation_gates() {
        let circuit = CircuitBuilder::new(1)
            .rx(0, PI / 2.0)
            .ry(0, PI)
            .rz(0, PI / 4.0)
            .build();

        assert_eq!(circuit.gate_count(), 3);
        assert_eq!(circuit.depth(), 3);
    }

    #[test]
    fn u3_gate() {
        let circuit = CircuitBuilder::new(1).u3(0, PI, PI / 2.0, PI / 4.0).build();

        assert_eq!(circuit.gate_count(), 1);
        match &circuit.gates()[0] {
            GateOp::U3(q, t, p, l) => {
                assert_eq!(*q, 0);
                assert!((t - PI).abs() < 1e-10);
                assert!((p - PI / 2.0).abs() < 1e-10);
                assert!((l - PI / 4.0).abs() < 1e-10);
            }
            _ => panic!("expected U3 gate"),
        }
    }

    #[test]
    fn controlled_rotation_gates() {
        let circuit = CircuitBuilder::new(2)
            .crx(0, 1, PI)
            .crz(0, 1, PI / 2.0)
            .build();

        assert_eq!(circuit.gate_count(), 2);
        assert_eq!(circuit.two_qubit_count(), 2);
    }

    #[test]
    fn three_qubit_gates() {
        let circuit = CircuitBuilder::new(3).ccx(0, 1, 2).cswap(0, 1, 2).build();

        assert_eq!(circuit.gate_count(), 2);
        let stats = circuit.stats();
        assert_eq!(stats.three_qubit_count, 2);
    }

    #[test]
    fn inverse_skips_measurements() {
        let circuit = CircuitBuilder::new(2)
            .h(0)
            .measure(0, 0)
            .x(1)
            .inverse()
            .build();

        // Measurements are skipped in inverse.
        // Original had H(0), Measure(0,0), X(1) => inverse skips Measure
        // Inverse: X(1), H(0) [measurements dropped]
        assert_eq!(circuit.gate_count(), 2);
        assert!(!circuit.gates().iter().any(|g| g.is_measurement()));
    }

    #[test]
    fn circuit_display() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let display = format!("{}", circuit);
        assert!(display.contains("qubits=2"));
        assert!(display.contains("gates=2"));
    }

    #[test]
    fn qubit_range_struct() {
        let qr = QubitRange::new(2, 5);
        assert_eq!(qr.len(), 3);
        assert!(!qr.is_empty());
        assert_eq!(qr.to_range(), 2..5);

        let qr_empty = QubitRange::new(5, 5);
        assert!(qr_empty.is_empty());
        assert_eq!(qr_empty.len(), 0);
    }

    #[test]
    fn qubit_range_from_range() {
        let qr: QubitRange = (1..4).into();
        assert_eq!(qr.start, 1);
        assert_eq!(qr.end, 4);
        assert_eq!(qr.len(), 3);
    }

    #[test]
    fn qubit_range_iter() {
        let qr = QubitRange::new(0, 3);
        let collected: Vec<usize> = qr.into_iter().collect();
        assert_eq!(collected, vec![0, 1, 2]);
    }

    #[test]
    fn gate_op_qubits() {
        assert_eq!(GateOp::H(2).qubits(), vec![2]);
        assert_eq!(GateOp::Cx(0, 1).qubits(), vec![0, 1]);
        assert_eq!(GateOp::Ccx(0, 1, 2).qubits(), vec![0, 1, 2]);
        assert_eq!(GateOp::Measure(3, 0).qubits(), vec![3]);
    }

    #[test]
    fn gate_op_inverse_roundtrip() {
        let gates = vec![
            GateOp::H(0),
            GateOp::X(0),
            GateOp::S(0),
            GateOp::T(0),
            GateOp::Rx(0, 1.5),
            GateOp::Cx(0, 1),
        ];

        for gate in &gates {
            let inv = gate.inverse().expect("gate should be invertible");
            let back = inv.inverse().expect("inverse should be invertible");
            assert_eq!(*gate, back, "double inverse should return original gate");
        }
    }

    #[test]
    fn measurement_not_invertible() {
        assert!(GateOp::Measure(0, 0).inverse().is_none());
    }

    #[test]
    fn barrier_not_invertible() {
        assert!(GateOp::Barrier(vec![0, 1]).inverse().is_none());
    }

    #[test]
    fn append_incompatible_error() {
        let big = CircuitBuilder::new(5).h_all().build();
        let result = CircuitBuilder::new(3).append(&big).try_build();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            CircuitError::IncompatibleQubitCount {
                expected: 3,
                actual: 5
            }
        )));
    }

    #[test]
    fn repeat_zero_clears() {
        let circuit = CircuitBuilder::new(2).h(0).repeat(0).build();
        assert_eq!(circuit.gate_count(), 0);
    }

    #[test]
    fn builder_error_accumulation() {
        let result = CircuitBuilder::new(2).h(10).cx(20, 1).try_build();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        // h(10) produces 1 error, cx(20, 1) produces 1 error for qubit 20
        assert!(errors.len() >= 2);
    }

    #[test]
    fn macro_bell_state() {
        let bell = circuit!(2;
            h 0;
            cx(0, 1);
            measure_all
        );
        assert_eq!(bell.gate_count(), 2);
        assert_eq!(bell.num_qubits(), 2);
        assert_eq!(bell.num_clbits(), 2);
    }

    #[test]
    fn macro_ghz() {
        let ghz = circuit!(3;
            h 0;
            cx(0, 1);
            cx(1, 2)
        );
        assert_eq!(ghz.gate_count(), 3);
    }

    #[test]
    fn depth_with_barrier() {
        // Barrier should synchronize qubits but not add to gate depth.
        let circuit = CircuitBuilder::new(2).h(0).barrier_all().h(1).build();

        // Without barrier: H(0) and H(1) are on different qubits, depth = 1.
        // With barrier: H(0) sets q0 to time 1, barrier syncs q1 to time 1,
        //               then H(1) goes at time 2. Depth = 2.
        assert_eq!(circuit.depth(), 2);
    }

    #[test]
    fn swap_gate() {
        let circuit = CircuitBuilder::new(2).swap(0, 1).build();
        assert_eq!(circuit.gate_count(), 1);
        assert_eq!(circuit.two_qubit_count(), 1);
        assert_eq!(circuit.gates()[0], GateOp::Swap(0, 1));
    }

    #[test]
    fn builder_has_errors() {
        let builder = CircuitBuilder::new(1).h(5);
        assert!(builder.has_errors());
        assert_eq!(builder.errors().len(), 1);
    }

    #[test]
    fn with_clbits() {
        let circuit = CircuitBuilder::with_clbits(2, 4).h(0).measure(0, 3).build();

        assert_eq!(circuit.num_clbits(), 4);
    }
}
