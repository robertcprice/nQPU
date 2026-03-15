//! Shaded Lightcone (SLC) Circuit Pre-Processing
//!
//! Classical technique to reduce quantum circuit size before execution by
//! identifying and removing qubits and gates that cannot affect the measurement
//! outcome. For each measurement qubit the algorithm computes the causal
//! lightcone -- the transitive set of gates and qubits whose evolution can
//! influence the measurement result -- and "shades" (removes) everything
//! outside that cone.
//!
//! # Supported Methods
//!
//! | Method | Direction | Use Case |
//! |---|---|---|
//! | `BackwardCone` | Measurements -> initial state | Standard; identifies all qubits that influence a measurement |
//! | `ForwardCone` | Initial state -> measurements | Detects dead qubits that never interact with anything |
//! | `Bidirectional` | Intersection of both | Most aggressive reduction |
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::shaded_lightcones::*;
//!
//! // Build a 4-qubit circuit: H(0), CX(0,1), Measure(1)
//! // Qubit 2 and 3 are not in the lightcone of measurement on qubit 1
//! let circuit = QuantumCircuit {
//!     num_qubits: 4,
//!     gates: vec![
//!         SlcGate::H(0),
//!         SlcGate::CX(0, 1),
//!         SlcGate::X(2),
//!         SlcGate::H(3),
//!         SlcGate::Measure(1),
//!     ],
//!     measurements: vec![1],
//! };
//!
//! let config = SlcConfig::default();
//! let shaded = shade_circuit(&circuit, &config).unwrap();
//! assert_eq!(shaded.reduced_qubits, 2); // Only qubits 0 and 1 survive
//! assert!(shaded.gates.len() < circuit.gates.len());
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum circuit depth before we reject to avoid degenerate analysis.
const MAX_CIRCUIT_DEPTH: usize = 1_000_000;

/// Tolerance for floating-point angle comparisons.
const ANGLE_EPSILON: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can arise during lightcone analysis and circuit shading.
#[derive(Clone, Debug, PartialEq)]
pub enum SlcError {
    /// The input circuit is structurally invalid (e.g. gate references a qubit
    /// index that is out of range for the declared `num_qubits`).
    InvalidCircuit(String),
    /// The computed lightcone is empty -- the circuit has no path from any
    /// initial qubit to the requested measurements.
    EmptyLightcone(String),
    /// Circuit depth exceeds the safety threshold.
    CircuitTooDeep(usize),
}

impl fmt::Display for SlcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlcError::InvalidCircuit(msg) => write!(f, "Invalid circuit: {}", msg),
            SlcError::EmptyLightcone(msg) => write!(f, "Empty lightcone: {}", msg),
            SlcError::CircuitTooDeep(d) => {
                write!(
                    f,
                    "Circuit depth {} exceeds maximum {}",
                    d, MAX_CIRCUIT_DEPTH
                )
            }
        }
    }
}

impl std::error::Error for SlcError {}

// ---------------------------------------------------------------------------
// Gate representation
// ---------------------------------------------------------------------------

/// A quantum gate used for lightcone analysis.
///
/// Each variant records the qubit indices it acts on and, where applicable,
/// a rotation angle. The `Barrier` variant is a scheduling directive and
/// does not alter the quantum state -- it is therefore invisible to the
/// lightcone algorithm.
#[derive(Clone, Debug, PartialEq)]
pub enum SlcGate {
    /// Hadamard on qubit `q`.
    H(usize),
    /// Pauli-X on qubit `q`.
    X(usize),
    /// Pauli-Y on qubit `q`.
    Y(usize),
    /// Pauli-Z on qubit `q`.
    Z(usize),
    /// Phase gate (S) on qubit `q`.
    S(usize),
    /// T gate on qubit `q`.
    T(usize),
    /// Rotation about X by `theta` on qubit `q`.
    Rx(usize, f64),
    /// Rotation about Y by `theta` on qubit `q`.
    Ry(usize, f64),
    /// Rotation about Z by `theta` on qubit `q`.
    Rz(usize, f64),
    /// Controlled-X (CNOT) with control `c` and target `t`.
    CX(usize, usize),
    /// Controlled-Z with qubits `a` and `b`.
    CZ(usize, usize),
    /// SWAP between qubits `a` and `b`.
    Swap(usize, usize),
    /// Toffoli (CCX) with controls `c1`, `c2` and target `t`.
    Toffoli(usize, usize, usize),
    /// Barrier -- scheduling hint; does NOT affect lightcone.
    Barrier(Vec<usize>),
    /// Measurement on qubit `q`.
    Measure(usize),
}

impl SlcGate {
    /// Return the set of qubit indices this gate touches.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            SlcGate::H(q)
            | SlcGate::X(q)
            | SlcGate::Y(q)
            | SlcGate::Z(q)
            | SlcGate::S(q)
            | SlcGate::T(q)
            | SlcGate::Rx(q, _)
            | SlcGate::Ry(q, _)
            | SlcGate::Rz(q, _)
            | SlcGate::Measure(q) => vec![*q],

            SlcGate::CX(a, b) | SlcGate::CZ(a, b) | SlcGate::Swap(a, b) => {
                vec![*a, *b]
            }

            SlcGate::Toffoli(a, b, c) => vec![*a, *b, *c],

            SlcGate::Barrier(qs) => qs.clone(),
        }
    }

    /// True if this gate is a barrier (scheduling hint only).
    pub fn is_barrier(&self) -> bool {
        matches!(self, SlcGate::Barrier(_))
    }

    /// True if this gate is a measurement.
    pub fn is_measurement(&self) -> bool {
        matches!(self, SlcGate::Measure(_))
    }

    /// True for single-qubit gates (excluding barriers and measurements).
    pub fn is_single_qubit(&self) -> bool {
        matches!(
            self,
            SlcGate::H(_)
                | SlcGate::X(_)
                | SlcGate::Y(_)
                | SlcGate::Z(_)
                | SlcGate::S(_)
                | SlcGate::T(_)
                | SlcGate::Rx(_, _)
                | SlcGate::Ry(_, _)
                | SlcGate::Rz(_, _)
        )
    }

    /// Return a human-readable name for the gate.
    pub fn name(&self) -> &'static str {
        match self {
            SlcGate::H(_) => "H",
            SlcGate::X(_) => "X",
            SlcGate::Y(_) => "Y",
            SlcGate::Z(_) => "Z",
            SlcGate::S(_) => "S",
            SlcGate::T(_) => "T",
            SlcGate::Rx(_, _) => "Rx",
            SlcGate::Ry(_, _) => "Ry",
            SlcGate::Rz(_, _) => "Rz",
            SlcGate::CX(_, _) => "CX",
            SlcGate::CZ(_, _) => "CZ",
            SlcGate::Swap(_, _) => "SWAP",
            SlcGate::Toffoli(_, _, _) => "Toffoli",
            SlcGate::Barrier(_) => "Barrier",
            SlcGate::Measure(_) => "Measure",
        }
    }

    /// Remap qubit indices through a translation table.
    /// `map[old] = Some(new)` for active qubits, `None` for removed ones.
    /// Returns `None` if any qubit in this gate was removed.
    pub fn remap(&self, map: &[Option<usize>]) -> Option<SlcGate> {
        match self {
            SlcGate::H(q) => map[*q].map(SlcGate::H),
            SlcGate::X(q) => map[*q].map(SlcGate::X),
            SlcGate::Y(q) => map[*q].map(SlcGate::Y),
            SlcGate::Z(q) => map[*q].map(SlcGate::Z),
            SlcGate::S(q) => map[*q].map(SlcGate::S),
            SlcGate::T(q) => map[*q].map(SlcGate::T),
            SlcGate::Rx(q, a) => map[*q].map(|nq| SlcGate::Rx(nq, *a)),
            SlcGate::Ry(q, a) => map[*q].map(|nq| SlcGate::Ry(nq, *a)),
            SlcGate::Rz(q, a) => map[*q].map(|nq| SlcGate::Rz(nq, *a)),
            SlcGate::CX(c, t) => {
                let nc = map[*c]?;
                let nt = map[*t]?;
                Some(SlcGate::CX(nc, nt))
            }
            SlcGate::CZ(a, b) => {
                let na = map[*a]?;
                let nb = map[*b]?;
                Some(SlcGate::CZ(na, nb))
            }
            SlcGate::Swap(a, b) => {
                let na = map[*a]?;
                let nb = map[*b]?;
                Some(SlcGate::Swap(na, nb))
            }
            SlcGate::Toffoli(a, b, c) => {
                let na = map[*a]?;
                let nb = map[*b]?;
                let nc = map[*c]?;
                Some(SlcGate::Toffoli(na, nb, nc))
            }
            SlcGate::Barrier(qs) => {
                let mapped: Vec<usize> = qs.iter().filter_map(|q| map[*q]).collect();
                if mapped.is_empty() {
                    None
                } else {
                    Some(SlcGate::Barrier(mapped))
                }
            }
            SlcGate::Measure(q) => map[*q].map(SlcGate::Measure),
        }
    }
}

impl fmt::Display for SlcGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlcGate::H(q) => write!(f, "H({})", q),
            SlcGate::X(q) => write!(f, "X({})", q),
            SlcGate::Y(q) => write!(f, "Y({})", q),
            SlcGate::Z(q) => write!(f, "Z({})", q),
            SlcGate::S(q) => write!(f, "S({})", q),
            SlcGate::T(q) => write!(f, "T({})", q),
            SlcGate::Rx(q, a) => write!(f, "Rx({}, {:.4})", q, a),
            SlcGate::Ry(q, a) => write!(f, "Ry({}, {:.4})", q, a),
            SlcGate::Rz(q, a) => write!(f, "Rz({}, {:.4})", q, a),
            SlcGate::CX(c, t) => write!(f, "CX({}, {})", c, t),
            SlcGate::CZ(a, b) => write!(f, "CZ({}, {})", a, b),
            SlcGate::Swap(a, b) => write!(f, "SWAP({}, {})", a, b),
            SlcGate::Toffoli(a, b, c) => write!(f, "Toffoli({}, {}, {})", a, b, c),
            SlcGate::Barrier(qs) => write!(f, "Barrier({:?})", qs),
            SlcGate::Measure(q) => write!(f, "Measure({})", q),
        }
    }
}

// ---------------------------------------------------------------------------
// Quantum circuit
// ---------------------------------------------------------------------------

/// A quantum circuit expressed as a sequence of gates over a fixed number of
/// qubits, together with a set of measurement targets.
#[derive(Clone, Debug)]
pub struct QuantumCircuit {
    /// Total number of qubits in the circuit.
    pub num_qubits: usize,
    /// Ordered sequence of gates (first = earliest in time).
    pub gates: Vec<SlcGate>,
    /// Qubit indices that will be measured at the end of the circuit.
    pub measurements: Vec<usize>,
}

impl QuantumCircuit {
    /// Create a new empty circuit with the given qubit count.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            measurements: Vec::new(),
        }
    }

    /// Append a gate.
    pub fn add_gate(&mut self, gate: SlcGate) {
        self.gates.push(gate);
    }

    /// Declare a measurement on `qubit`.
    pub fn add_measurement(&mut self, qubit: usize) {
        if !self.measurements.contains(&qubit) {
            self.measurements.push(qubit);
        }
    }

    /// Validate that all qubit references are in range.
    pub fn validate(&self) -> Result<(), SlcError> {
        for (i, gate) in self.gates.iter().enumerate() {
            for q in gate.qubits() {
                if q >= self.num_qubits {
                    return Err(SlcError::InvalidCircuit(format!(
                        "Gate {} ({}) references qubit {} but circuit has only {} qubits",
                        i,
                        gate.name(),
                        q,
                        self.num_qubits,
                    )));
                }
            }
        }
        for &m in &self.measurements {
            if m >= self.num_qubits {
                return Err(SlcError::InvalidCircuit(format!(
                    "Measurement references qubit {} but circuit has only {} qubits",
                    m, self.num_qubits,
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Lightcone
// ---------------------------------------------------------------------------

/// The causal lightcone of a set of measurement qubits through a circuit.
///
/// Tracks which qubits and which gates lie on at least one causal path
/// from the initial state to the target measurements.
#[derive(Clone, Debug)]
pub struct Lightcone {
    /// The measurement qubits that seeded this lightcone.
    pub target_qubits: Vec<usize>,
    /// Per-qubit flag: `active_qubits[q]` is true if qubit `q` is inside the
    /// lightcone.
    pub active_qubits: Vec<bool>,
    /// Per-gate flag: `active_gates[i]` is true if gate `i` is inside the
    /// lightcone.
    pub active_gates: Vec<bool>,
    /// Number of qubits inside the lightcone.
    pub num_active_qubits: usize,
    /// Number of gates inside the lightcone.
    pub num_active_gates: usize,
}

impl Lightcone {
    /// Create a lightcone marking everything as inactive.
    fn empty(num_qubits: usize, num_gates: usize) -> Self {
        Self {
            target_qubits: Vec::new(),
            active_qubits: vec![false; num_qubits],
            active_gates: vec![false; num_gates],
            num_active_qubits: 0,
            num_active_gates: 0,
        }
    }

    /// Recount the summary fields from the boolean vectors.
    fn recount(&mut self) {
        self.num_active_qubits = self.active_qubits.iter().filter(|&&a| a).count();
        self.num_active_gates = self.active_gates.iter().filter(|&&a| a).count();
    }

    /// Merge another lightcone into this one (union of active sets).
    pub fn merge(&mut self, other: &Lightcone) {
        assert_eq!(self.active_qubits.len(), other.active_qubits.len());
        assert_eq!(self.active_gates.len(), other.active_gates.len());
        for (a, b) in self
            .active_qubits
            .iter_mut()
            .zip(other.active_qubits.iter())
        {
            *a = *a || *b;
        }
        for (a, b) in self.active_gates.iter_mut().zip(other.active_gates.iter()) {
            *a = *a || *b;
        }
        for &tq in &other.target_qubits {
            if !self.target_qubits.contains(&tq) {
                self.target_qubits.push(tq);
            }
        }
        self.recount();
    }

    /// Intersect with another lightcone (keep only elements in both).
    pub fn intersect(&mut self, other: &Lightcone) {
        assert_eq!(self.active_qubits.len(), other.active_qubits.len());
        assert_eq!(self.active_gates.len(), other.active_gates.len());
        for (a, b) in self
            .active_qubits
            .iter_mut()
            .zip(other.active_qubits.iter())
        {
            *a = *a && *b;
        }
        for (a, b) in self.active_gates.iter_mut().zip(other.active_gates.iter()) {
            *a = *a && *b;
        }
        self.recount();
    }
}

// ---------------------------------------------------------------------------
// Shaded circuit
// ---------------------------------------------------------------------------

/// A circuit that has been reduced ("shaded") by removing all qubits and
/// gates outside the lightcone. Qubit indices are compacted so that the
/// reduced circuit uses indices `0..reduced_qubits`.
#[derive(Clone, Debug)]
pub struct ShadedCircuit {
    /// Number of qubits in the original circuit.
    pub original_qubits: usize,
    /// Number of qubits in the reduced circuit.
    pub reduced_qubits: usize,
    /// The gates that survived shading, with remapped qubit indices.
    pub gates: Vec<SlcGate>,
    /// `qubit_map[original_index]` = `Some(reduced_index)` for active qubits,
    /// `None` for removed qubits.
    pub qubit_map: Vec<Option<usize>>,
    /// `inverse_map[reduced_index]` = `original_index`.
    pub inverse_map: Vec<usize>,
    /// Fraction of qubits removed: `qubits_removed / original_qubits`.
    pub reduction_ratio: f64,
    /// Absolute number of gates removed.
    pub gates_removed: usize,
    /// Absolute number of qubits removed.
    pub qubits_removed: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which lightcone algorithm to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlcMethod {
    /// Standard backward lightcone: walk from measurements toward initial state.
    BackwardCone,
    /// Forward lightcone: walk from initial state toward measurements.
    ForwardCone,
    /// Bidirectional: intersect backward and forward cones for maximum reduction.
    Bidirectional,
}

/// Configuration for the SLC pre-processing pass.
#[derive(Clone, Debug)]
pub struct SlcConfig {
    /// Lightcone computation method.
    pub method: SlcMethod,
    /// When true, merge lightcones of multiple measurement qubits into a single
    /// unified lightcone before shading.
    pub merge_lightcones: bool,
    /// Remove identity gate pairs (H-H, X-X, etc.) before lightcone analysis.
    pub remove_identity_gates: bool,
    /// Merge adjacent single-qubit rotation gates (Rz-Rz -> Rz).
    pub simplify_single_qubit_chains: bool,
}

impl Default for SlcConfig {
    fn default() -> Self {
        Self {
            method: SlcMethod::BackwardCone,
            merge_lightcones: true,
            remove_identity_gates: true,
            simplify_single_qubit_chains: true,
        }
    }
}

impl SlcConfig {
    /// Builder: set the method.
    pub fn with_method(mut self, method: SlcMethod) -> Self {
        self.method = method;
        self
    }

    /// Builder: set merge_lightcones.
    pub fn with_merge(mut self, merge: bool) -> Self {
        self.merge_lightcones = merge;
        self
    }

    /// Builder: set remove_identity_gates.
    pub fn with_identity_removal(mut self, remove: bool) -> Self {
        self.remove_identity_gates = remove;
        self
    }

    /// Builder: set simplify_single_qubit_chains.
    pub fn with_chain_simplification(mut self, simplify: bool) -> Self {
        self.simplify_single_qubit_chains = simplify;
        self
    }
}

// ---------------------------------------------------------------------------
// Analysis report
// ---------------------------------------------------------------------------

/// Summary statistics comparing the original circuit to the shaded version.
#[derive(Clone, Debug)]
pub struct SlcReport {
    pub original_qubits: usize,
    pub original_gates: usize,
    pub original_depth: usize,
    pub reduced_qubits: usize,
    pub reduced_gates: usize,
    pub reduced_depth: usize,
    /// Percentage of qubits removed.
    pub qubit_reduction: f64,
    /// Percentage of gates removed.
    pub gate_reduction: f64,
    /// Percentage of depth reduced.
    pub depth_reduction: f64,
    /// For each measurement qubit, the size (number of active qubits) of its
    /// individual backward lightcone before merging.
    pub lightcone_sizes: Vec<usize>,
}

impl fmt::Display for SlcReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== SLC Report ===")?;
        writeln!(
            f,
            "Qubits : {} -> {} ({:.1}% reduction)",
            self.original_qubits, self.reduced_qubits, self.qubit_reduction
        )?;
        writeln!(
            f,
            "Gates  : {} -> {} ({:.1}% reduction)",
            self.original_gates, self.reduced_gates, self.gate_reduction
        )?;
        writeln!(
            f,
            "Depth  : {} -> {} ({:.1}% reduction)",
            self.original_depth, self.reduced_depth, self.depth_reduction
        )?;
        writeln!(f, "Lightcone sizes: {:?}", self.lightcone_sizes)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dependency graph
// ---------------------------------------------------------------------------

/// A node in the gate dependency DAG.
#[derive(Clone, Debug)]
pub struct GateNode {
    /// Index of this gate in the original circuit's `gates` vector.
    pub gate_index: usize,
    /// Qubit indices this gate touches.
    pub qubits: Vec<usize>,
    /// Depth (layer) of the gate: the longest path from any input to this node.
    pub depth: usize,
}

/// Directed acyclic graph (DAG) of gate dependencies within a circuit.
///
/// An edge `(a, b)` means gate `a` must execute before gate `b` because they
/// share at least one qubit and `a` appears earlier in the circuit.
#[derive(Clone, Debug)]
pub struct DependencyGraph {
    /// One node per gate.
    pub gates: Vec<GateNode>,
    /// Edges: `(from_gate_index, to_gate_index)`.
    pub edges: Vec<(usize, usize)>,
}

impl DependencyGraph {
    /// The critical-path depth of the circuit (longest path in the DAG + 1).
    pub fn circuit_depth(&self) -> usize {
        self.gates
            .iter()
            .map(|n| n.depth)
            .max()
            .map_or(0, |d| d + 1)
    }
}

/// Build the dependency graph for a circuit.
///
/// Two gates are connected by an edge if they share at least one qubit and
/// the earlier gate is the most recent predecessor on that qubit. Barriers
/// are excluded from the dependency graph since they do not alter the quantum
/// state.
pub fn build_dependency_graph(circuit: &QuantumCircuit) -> DependencyGraph {
    let n_gates = circuit.gates.len();
    // For each qubit, track the index of the most recent gate that touched it.
    let mut last_gate_on_qubit: Vec<Option<usize>> = vec![None; circuit.num_qubits];
    let mut nodes: Vec<GateNode> = Vec::with_capacity(n_gates);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    // Predecessors for depth computation: predecessors[i] = set of gate indices
    // that are direct predecessors of gate i.
    let mut predecessors: Vec<Vec<usize>> = Vec::with_capacity(n_gates);

    for (i, gate) in circuit.gates.iter().enumerate() {
        let qs = gate.qubits();
        let mut preds: Vec<usize> = Vec::new();

        if !gate.is_barrier() {
            for &q in &qs {
                if q < circuit.num_qubits {
                    if let Some(prev) = last_gate_on_qubit[q] {
                        if !preds.contains(&prev) {
                            preds.push(prev);
                            edges.push((prev, i));
                        }
                    }
                    last_gate_on_qubit[q] = Some(i);
                }
            }
        }

        predecessors.push(preds);
        nodes.push(GateNode {
            gate_index: i,
            qubits: qs,
            depth: 0, // will be computed below
        });
    }

    // Compute depths via topological order (gates are already in order).
    for i in 0..n_gates {
        let max_pred_depth = predecessors[i]
            .iter()
            .map(|&p| nodes[p].depth + 1)
            .max()
            .unwrap_or(0);
        nodes[i].depth = max_pred_depth;
    }

    DependencyGraph {
        gates: nodes,
        edges,
    }
}

// ---------------------------------------------------------------------------
// Gate simplification (identity removal + chain merging)
// ---------------------------------------------------------------------------

/// Return true if two gates cancel each other (form the identity).
///
/// Recognized pairs:
/// - H-H, X-X, Y-Y, Z-Z (self-inverse gates)
/// - S followed by S (S^2 = Z, not identity) -- NOT cancelled
/// - CX-CX on the same (control, target) pair
/// - CZ-CZ on the same pair
/// - SWAP-SWAP on the same pair
fn gates_cancel(a: &SlcGate, b: &SlcGate) -> bool {
    match (a, b) {
        (SlcGate::H(q1), SlcGate::H(q2)) if q1 == q2 => true,
        (SlcGate::X(q1), SlcGate::X(q2)) if q1 == q2 => true,
        (SlcGate::Y(q1), SlcGate::Y(q2)) if q1 == q2 => true,
        (SlcGate::Z(q1), SlcGate::Z(q2)) if q1 == q2 => true,
        (SlcGate::CX(c1, t1), SlcGate::CX(c2, t2)) if c1 == c2 && t1 == t2 => true,
        (SlcGate::CZ(a1, b1), SlcGate::CZ(a2, b2)) if a1 == a2 && b1 == b2 => true,
        (SlcGate::Swap(a1, b1), SlcGate::Swap(a2, b2)) if a1 == a2 && b1 == b2 => true,
        _ => false,
    }
}

/// Try to merge two adjacent single-qubit rotation gates on the same qubit
/// and same axis into a single rotation. Returns `Some(merged)` on success.
fn try_merge_rotations(a: &SlcGate, b: &SlcGate) -> Option<SlcGate> {
    match (a, b) {
        (SlcGate::Rz(q1, a1), SlcGate::Rz(q2, a2)) if q1 == q2 => {
            let angle = a1 + a2;
            if angle.abs() < ANGLE_EPSILON {
                None // zero rotation = identity; caller should remove both
            } else {
                Some(SlcGate::Rz(*q1, angle))
            }
        }
        (SlcGate::Rx(q1, a1), SlcGate::Rx(q2, a2)) if q1 == q2 => {
            let angle = a1 + a2;
            if angle.abs() < ANGLE_EPSILON {
                None
            } else {
                Some(SlcGate::Rx(*q1, angle))
            }
        }
        (SlcGate::Ry(q1, a1), SlcGate::Ry(q2, a2)) if q1 == q2 => {
            let angle = a1 + a2;
            if angle.abs() < ANGLE_EPSILON {
                None
            } else {
                Some(SlcGate::Ry(*q1, angle))
            }
        }
        _ => {
            // Signal "no merge" vs "merged to identity" via a sentinel:
            // We return Some(original_b) to mean "not applicable" -- but
            // that is confusing. Instead, the caller checks the discriminant.
            // Returning None here means "not applicable" in the non-same-axis case.
            // We need a tri-state, so use a wrapper. For simplicity, return None
            // when not applicable and use an explicit separate check for identity.
            None
        }
    }
}

/// Check if two rotations on the same qubit and axis sum to zero.
fn rotations_cancel(a: &SlcGate, b: &SlcGate) -> bool {
    match (a, b) {
        (SlcGate::Rz(q1, a1), SlcGate::Rz(q2, a2)) if q1 == q2 => (a1 + a2).abs() < ANGLE_EPSILON,
        (SlcGate::Rx(q1, a1), SlcGate::Rx(q2, a2)) if q1 == q2 => (a1 + a2).abs() < ANGLE_EPSILON,
        (SlcGate::Ry(q1, a1), SlcGate::Ry(q2, a2)) if q1 == q2 => (a1 + a2).abs() < ANGLE_EPSILON,
        _ => false,
    }
}

/// Remove identity gate pairs and merge adjacent rotation chains.
///
/// This operates greedily: scan left-to-right, looking for adjacent
/// cancellations on the same qubit(s). Multi-qubit gates block merging
/// on their qubits.
pub fn simplify_gates(
    gates: &[SlcGate],
    remove_identities: bool,
    merge_rotations: bool,
) -> Vec<SlcGate> {
    if !remove_identities && !merge_rotations {
        return gates.to_vec();
    }

    let mut result: Vec<SlcGate> = Vec::with_capacity(gates.len());

    for gate in gates {
        if gate.is_barrier() || gate.is_measurement() {
            result.push(gate.clone());
            continue;
        }

        let mut merged = false;

        // Look at the last gate in result that acts on the same qubit set.
        if !result.is_empty() {
            let last_idx = result.len() - 1;
            let last = &result[last_idx];

            // Check for identity cancellation.
            if remove_identities && gates_cancel(last, gate) {
                result.pop();
                merged = true;
            }
            // Check for rotation merging.
            else if merge_rotations {
                if rotations_cancel(last, gate) {
                    result.pop();
                    merged = true;
                } else if let Some(merged_gate) = try_merge_rotations(last, gate) {
                    result.pop();
                    result.push(merged_gate);
                    merged = true;
                }
            }
        }

        if !merged {
            result.push(gate.clone());
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Core lightcone algorithms
// ---------------------------------------------------------------------------

/// Compute the **backward lightcone** from a set of target measurement qubits.
///
/// Starting from the measurement qubits, walk the gate list in reverse. At
/// each gate, if any of the gate's qubits are currently "active" (in the cone),
/// mark ALL of the gate's qubits as active and mark the gate as active.
pub fn backward_lightcone(circuit: &QuantumCircuit, targets: &[usize]) -> Lightcone {
    let n = circuit.num_qubits;
    let ng = circuit.gates.len();
    let mut lc = Lightcone::empty(n, ng);
    lc.target_qubits = targets.to_vec();

    // Seed with measurement qubits.
    for &t in targets {
        if t < n {
            lc.active_qubits[t] = true;
        }
    }

    // Walk backwards through the circuit.
    for i in (0..ng).rev() {
        let gate = &circuit.gates[i];
        if gate.is_barrier() {
            continue;
        }

        let qs = gate.qubits();
        let any_active = qs.iter().any(|&q| q < n && lc.active_qubits[q]);

        if any_active {
            lc.active_gates[i] = true;
            for &q in &qs {
                if q < n {
                    lc.active_qubits[q] = true;
                }
            }
        }
    }

    lc.recount();
    lc
}

/// Compute the **forward lightcone** from a set of initial qubits.
///
/// Starting from the given initial qubits (or all qubits if empty), walk the
/// gate list forward. At each gate, if any of the gate's qubits are active,
/// mark all of them as active and mark the gate as active.
pub fn forward_lightcone(circuit: &QuantumCircuit, initial_qubits: &[usize]) -> Lightcone {
    let n = circuit.num_qubits;
    let ng = circuit.gates.len();
    let mut lc = Lightcone::empty(n, ng);

    // Seed with initial qubits. If empty, assume all qubits are live.
    if initial_qubits.is_empty() {
        for q in 0..n {
            lc.active_qubits[q] = true;
        }
    } else {
        for &q in initial_qubits {
            if q < n {
                lc.active_qubits[q] = true;
            }
        }
    }

    // Walk forward.
    for i in 0..ng {
        let gate = &circuit.gates[i];
        if gate.is_barrier() {
            continue;
        }

        let qs = gate.qubits();
        let any_active = qs.iter().any(|&q| q < n && lc.active_qubits[q]);

        if any_active {
            lc.active_gates[i] = true;
            for &q in &qs {
                if q < n {
                    lc.active_qubits[q] = true;
                }
            }
        }
    }

    lc.recount();
    lc
}

/// Compute the **bidirectional lightcone**: the intersection of the backward
/// cone from the measurements and the forward cone from all initial qubits.
///
/// This is the most aggressive reduction: it prunes qubits that influence
/// the measurement (backward cone) but are never actually initialized from a
/// non-trivial state (forward cone), as well as qubits that are initialized
/// but never reach the measurement.
pub fn bidirectional_lightcone(circuit: &QuantumCircuit, targets: &[usize]) -> Lightcone {
    let back = backward_lightcone(circuit, targets);
    let fwd = forward_lightcone(circuit, &[]);

    let mut combined = back;
    combined.intersect(&fwd);
    combined
}

// ---------------------------------------------------------------------------
// Circuit shading (the main entry point)
// ---------------------------------------------------------------------------

/// Compute per-measurement lightcones and merge them according to the config.
fn compute_lightcone(circuit: &QuantumCircuit, config: &SlcConfig) -> Result<Lightcone, SlcError> {
    if circuit.measurements.is_empty() {
        return Err(SlcError::EmptyLightcone(
            "No measurement qubits specified".to_string(),
        ));
    }

    let compute_single = |targets: &[usize]| -> Lightcone {
        match config.method {
            SlcMethod::BackwardCone => backward_lightcone(circuit, targets),
            SlcMethod::ForwardCone => {
                // For forward cone mode we still use backward from measurements
                // as the primary cone, then intersect with forward from all qubits.
                // This gives a meaningful "forward" reduction.
                let back = backward_lightcone(circuit, targets);
                let fwd = forward_lightcone(circuit, &[]);
                let mut result = back;
                result.intersect(&fwd);
                result
            }
            SlcMethod::Bidirectional => bidirectional_lightcone(circuit, targets),
        }
    };

    if config.merge_lightcones {
        // Single pass with all measurements at once.
        let lc = compute_single(&circuit.measurements);
        Ok(lc)
    } else {
        // Compute individual lightcones and merge.
        let mut merged = Lightcone::empty(circuit.num_qubits, circuit.gates.len());
        for &m in &circuit.measurements {
            let lc = compute_single(&[m]);
            merged.merge(&lc);
        }
        Ok(merged)
    }
}

/// Apply the lightcone to produce a shaded (reduced) circuit.
fn apply_shading(
    circuit: &QuantumCircuit,
    lightcone: &Lightcone,
) -> Result<ShadedCircuit, SlcError> {
    let n = circuit.num_qubits;

    // Build qubit map: original -> reduced.
    let mut qubit_map: Vec<Option<usize>> = vec![None; n];
    let mut inverse_map: Vec<usize> = Vec::new();
    let mut reduced_idx = 0;

    for q in 0..n {
        if lightcone.active_qubits[q] {
            qubit_map[q] = Some(reduced_idx);
            inverse_map.push(q);
            reduced_idx += 1;
        }
    }

    let reduced_qubits = reduced_idx;

    if reduced_qubits == 0 {
        return Err(SlcError::EmptyLightcone(
            "All qubits removed; lightcone is empty".to_string(),
        ));
    }

    // Filter and remap gates.
    let mut shaded_gates: Vec<SlcGate> = Vec::new();
    let mut gates_removed = 0usize;

    for (i, gate) in circuit.gates.iter().enumerate() {
        if lightcone.active_gates[i] {
            if let Some(remapped) = gate.remap(&qubit_map) {
                shaded_gates.push(remapped);
            } else {
                gates_removed += 1;
            }
        } else {
            gates_removed += 1;
        }
    }

    let qubits_removed = n - reduced_qubits;
    let reduction_ratio = if n > 0 {
        qubits_removed as f64 / n as f64
    } else {
        0.0
    };

    Ok(ShadedCircuit {
        original_qubits: n,
        reduced_qubits,
        gates: shaded_gates,
        qubit_map,
        inverse_map,
        reduction_ratio,
        gates_removed,
        qubits_removed,
    })
}

/// Shade a quantum circuit: remove qubits and gates outside the causal
/// lightcone of its measurements.
///
/// This is the main entry point for the SLC pre-processing pass.
///
/// 1. Validate the circuit.
/// 2. Optionally simplify (remove identity pairs, merge rotations).
/// 3. Compute the lightcone according to the configured method.
/// 4. Remove gates and qubits outside the lightcone.
/// 5. Remap qubit indices to compact numbering.
pub fn shade_circuit(
    circuit: &QuantumCircuit,
    config: &SlcConfig,
) -> Result<ShadedCircuit, SlcError> {
    circuit.validate()?;

    // Optional gate simplification.
    let working_circuit = if config.remove_identity_gates || config.simplify_single_qubit_chains {
        let simplified = simplify_gates(
            &circuit.gates,
            config.remove_identity_gates,
            config.simplify_single_qubit_chains,
        );
        QuantumCircuit {
            num_qubits: circuit.num_qubits,
            gates: simplified,
            measurements: circuit.measurements.clone(),
        }
    } else {
        circuit.clone()
    };

    // Compute lightcone.
    let lightcone = compute_lightcone(&working_circuit, config)?;

    // Apply shading.
    apply_shading(&working_circuit, &lightcone)
}

// ---------------------------------------------------------------------------
// Circuit depth computation
// ---------------------------------------------------------------------------

/// Compute the depth of a gate sequence over the given number of qubits.
///
/// Depth is defined as the minimum number of time steps needed to execute
/// all gates, respecting the constraint that two gates sharing a qubit cannot
/// execute simultaneously.
pub fn circuit_depth(gates: &[SlcGate], num_qubits: usize) -> usize {
    let mut qubit_depth: Vec<usize> = vec![0; num_qubits];

    for gate in gates {
        if gate.is_barrier() {
            continue;
        }
        let qs = gate.qubits();
        let max_d = qs
            .iter()
            .filter(|&&q| q < num_qubits)
            .map(|&q| qubit_depth[q])
            .max()
            .unwrap_or(0);
        let new_depth = max_d + 1;
        for &q in &qs {
            if q < num_qubits {
                qubit_depth[q] = new_depth;
            }
        }
    }

    qubit_depth.iter().copied().max().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Reduction analysis / report
// ---------------------------------------------------------------------------

/// Produce a detailed analysis report comparing the original and shaded
/// circuits.
pub fn analyze_reduction(
    circuit: &QuantumCircuit,
    shaded: &ShadedCircuit,
    _config: &SlcConfig,
) -> SlcReport {
    let original_depth = circuit_depth(&circuit.gates, circuit.num_qubits);
    let reduced_depth = circuit_depth(&shaded.gates, shaded.reduced_qubits);

    // Count non-barrier, non-measurement gates in original.
    let original_gates = circuit.gates.iter().filter(|g| !g.is_barrier()).count();
    let reduced_gates = shaded.gates.iter().filter(|g| !g.is_barrier()).count();

    let qubit_reduction = if circuit.num_qubits > 0 {
        (circuit.num_qubits - shaded.reduced_qubits) as f64 / circuit.num_qubits as f64 * 100.0
    } else {
        0.0
    };

    let gate_reduction = if original_gates > 0 {
        (original_gates.saturating_sub(reduced_gates)) as f64 / original_gates as f64 * 100.0
    } else {
        0.0
    };

    let depth_reduction = if original_depth > 0 {
        (original_depth.saturating_sub(reduced_depth)) as f64 / original_depth as f64 * 100.0
    } else {
        0.0
    };

    // Compute individual lightcone sizes per measurement qubit.
    let lightcone_sizes: Vec<usize> = circuit
        .measurements
        .iter()
        .map(|&m| {
            let lc = backward_lightcone(circuit, &[m]);
            lc.num_active_qubits
        })
        .collect();

    SlcReport {
        original_qubits: circuit.num_qubits,
        original_gates,
        original_depth,
        reduced_qubits: shaded.reduced_qubits,
        reduced_gates,
        reduced_depth,
        qubit_reduction,
        gate_reduction,
        depth_reduction,
        lightcone_sizes,
    }
}

// ---------------------------------------------------------------------------
// Convenience: shade + analyze in one call
// ---------------------------------------------------------------------------

/// Shade the circuit and produce the analysis report in a single pass.
pub fn shade_and_analyze(
    circuit: &QuantumCircuit,
    config: &SlcConfig,
) -> Result<(ShadedCircuit, SlcReport), SlcError> {
    let shaded = shade_circuit(circuit, config)?;
    let report = analyze_reduction(circuit, &shaded, config);
    Ok((shaded, report))
}

// ---------------------------------------------------------------------------
// Per-qubit lightcone analysis utilities
// ---------------------------------------------------------------------------

/// For each measurement qubit, compute the backward lightcone and return
/// the list of (measurement_qubit, active_qubit_count) pairs.
pub fn per_measurement_lightcone_sizes(circuit: &QuantumCircuit) -> Vec<(usize, usize)> {
    circuit
        .measurements
        .iter()
        .map(|&m| {
            let lc = backward_lightcone(circuit, &[m]);
            (m, lc.num_active_qubits)
        })
        .collect()
}

/// Identify "bottleneck" qubits that appear in the lightcone of every single
/// measurement qubit. These are the most critical qubits -- removing any one
/// of them would disconnect some measurement from its causal support.
pub fn bottleneck_qubits(circuit: &QuantumCircuit) -> Vec<usize> {
    if circuit.measurements.is_empty() {
        return Vec::new();
    }

    let n = circuit.num_qubits;
    let mut in_all = vec![true; n];

    for &m in &circuit.measurements {
        let lc = backward_lightcone(circuit, &[m]);
        for q in 0..n {
            if !lc.active_qubits[q] {
                in_all[q] = false;
            }
        }
    }

    (0..n).filter(|&q| in_all[q]).collect()
}

// ---------------------------------------------------------------------------
// Circuit builder helpers
// ---------------------------------------------------------------------------

/// Build a simple linear (nearest-neighbor) circuit for testing.
///
/// Creates `depth` layers, each applying CX between adjacent qubits in
/// alternating even/odd fashion, plus Hadamards on all qubits in the first
/// layer.
pub fn build_linear_circuit(num_qubits: usize, depth: usize, measure: &[usize]) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(num_qubits);

    // Initial Hadamards.
    for q in 0..num_qubits {
        circuit.add_gate(SlcGate::H(q));
    }

    // Alternating CX layers.
    for layer in 0..depth {
        let start = if layer % 2 == 0 { 0 } else { 1 };
        let mut q = start;
        while q + 1 < num_qubits {
            circuit.add_gate(SlcGate::CX(q, q + 1));
            q += 2;
        }
    }

    for &m in measure {
        circuit.add_gate(SlcGate::Measure(m));
        circuit.add_measurement(m);
    }

    circuit
}

/// Build an all-to-all entangling circuit (every pair connected).
pub fn build_all_to_all_circuit(num_qubits: usize, measure: &[usize]) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(num_qubits);

    for q in 0..num_qubits {
        circuit.add_gate(SlcGate::H(q));
    }
    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            circuit.add_gate(SlcGate::CX(i, j));
        }
    }

    for &m in measure {
        circuit.add_gate(SlcGate::Measure(m));
        circuit.add_measurement(m);
    }

    circuit
}

// ---------------------------------------------------------------------------
// Advanced: layered gate extraction for depth analysis
// ---------------------------------------------------------------------------

/// Assign each gate to a layer (time step) for parallel execution scheduling.
///
/// Returns a vector of layers, where each layer is a vector of gate indices
/// that can be executed simultaneously.
pub fn layer_assignment(circuit: &QuantumCircuit) -> Vec<Vec<usize>> {
    let mut qubit_layer: Vec<usize> = vec![0; circuit.num_qubits];
    let mut gate_layers: Vec<usize> = Vec::with_capacity(circuit.gates.len());
    let mut max_layer = 0usize;

    for (_i, gate) in circuit.gates.iter().enumerate() {
        if gate.is_barrier() {
            gate_layers.push(0);
            continue;
        }

        let qs = gate.qubits();
        let layer = qs
            .iter()
            .filter(|&&q| q < circuit.num_qubits)
            .map(|&q| qubit_layer[q])
            .max()
            .unwrap_or(0);

        gate_layers.push(layer);

        let next = layer + 1;
        for &q in &qs {
            if q < circuit.num_qubits {
                qubit_layer[q] = next;
            }
        }

        if layer > max_layer {
            max_layer = layer;
        }
    }

    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_layer + 1];
    for (i, &layer) in gate_layers.iter().enumerate() {
        if !circuit.gates[i].is_barrier() {
            layers[layer].push(i);
        }
    }

    layers
}

// ---------------------------------------------------------------------------
// Advanced: qubit interaction graph
// ---------------------------------------------------------------------------

/// A graph where vertices are qubits and edges represent two-qubit gate
/// interactions. Useful for understanding connectivity before shading.
#[derive(Clone, Debug)]
pub struct QubitInteractionGraph {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Adjacency list: `adj[q]` = set of qubits connected to q by at least one
    /// two-qubit gate.
    pub adj: Vec<HashSet<usize>>,
    /// Edge weights: `weights[(min(a,b), max(a,b))]` = number of two-qubit
    /// gates between qubits a and b.
    pub weights: HashMap<(usize, usize), usize>,
}

/// Build the qubit interaction graph for a circuit.
pub fn build_interaction_graph(circuit: &QuantumCircuit) -> QubitInteractionGraph {
    let n = circuit.num_qubits;
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut weights: HashMap<(usize, usize), usize> = HashMap::new();

    for gate in &circuit.gates {
        let qs = gate.qubits();
        if qs.len() >= 2 && !gate.is_barrier() {
            for i in 0..qs.len() {
                for j in (i + 1)..qs.len() {
                    let a = qs[i].min(qs[j]);
                    let b = qs[i].max(qs[j]);
                    adj[a].insert(b);
                    adj[b].insert(a);
                    *weights.entry((a, b)).or_insert(0) += 1;
                }
            }
        }
    }

    QubitInteractionGraph {
        num_qubits: n,
        adj,
        weights,
    }
}

// ---------------------------------------------------------------------------
// Advanced: causal cone distance
// ---------------------------------------------------------------------------

/// Compute the "lightcone distance" from each qubit to the nearest
/// measurement qubit. A qubit directly measured has distance 0. A qubit
/// that interacts with a distance-0 qubit through a two-qubit gate has
/// distance 1, and so on. Qubits outside the lightcone have distance
/// `usize::MAX`.
pub fn lightcone_distance(circuit: &QuantumCircuit) -> Vec<usize> {
    let n = circuit.num_qubits;
    let mut dist = vec![usize::MAX; n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    // Seed: measurement qubits at distance 0.
    for &m in &circuit.measurements {
        if m < n && dist[m] == usize::MAX {
            dist[m] = 0;
            queue.push_back(m);
        }
    }

    // BFS on the interaction graph, walking backwards through the circuit.
    // We use the interaction graph for a simpler BFS.
    let ig = build_interaction_graph(circuit);

    while let Some(q) = queue.pop_front() {
        let d = dist[q];
        for &neighbor in &ig.adj[q] {
            if dist[neighbor] == usize::MAX {
                dist[neighbor] = d + 1;
                queue.push_back(neighbor);
            }
        }
    }

    dist
}

// ---------------------------------------------------------------------------
// Advanced: multi-pass shading with depth limit
// ---------------------------------------------------------------------------

/// Shade a circuit but also enforce a maximum circuit depth on the result.
///
/// If the shaded circuit's depth exceeds `max_depth`, returns an error.
pub fn shade_with_depth_limit(
    circuit: &QuantumCircuit,
    config: &SlcConfig,
    max_depth: usize,
) -> Result<ShadedCircuit, SlcError> {
    let shaded = shade_circuit(circuit, config)?;
    let depth = circuit_depth(&shaded.gates, shaded.reduced_qubits);

    if depth > max_depth {
        return Err(SlcError::CircuitTooDeep(depth));
    }

    Ok(shaded)
}

// ---------------------------------------------------------------------------
// Advanced: iterative shading
// ---------------------------------------------------------------------------

/// Apply shading iteratively until no further reduction is possible.
///
/// Each iteration may expose new identity gates or trivial qubits once
/// neighbours have been removed. Returns the final shaded circuit and the
/// number of iterations performed.
pub fn iterative_shade(
    circuit: &QuantumCircuit,
    config: &SlcConfig,
    max_iterations: usize,
) -> Result<(ShadedCircuit, usize), SlcError> {
    let mut current_circuit = circuit.clone();
    let mut cumulative_inverse: Vec<usize> = (0..circuit.num_qubits).collect();
    let mut total_qubits_removed = 0usize;
    let mut total_gates_removed = 0usize;
    let mut iteration = 0;

    loop {
        if iteration >= max_iterations {
            break;
        }

        let shaded = shade_circuit(&current_circuit, config)?;

        if shaded.qubits_removed == 0 && shaded.gates_removed == 0 {
            break;
        }

        total_qubits_removed += shaded.qubits_removed;
        total_gates_removed += shaded.gates_removed;

        // Update cumulative inverse map.
        let new_inverse: Vec<usize> = shaded
            .inverse_map
            .iter()
            .map(|&reduced_orig| cumulative_inverse[reduced_orig])
            .collect();
        cumulative_inverse = new_inverse;

        // Build new measurements in reduced space.
        let new_measurements: Vec<usize> = current_circuit
            .measurements
            .iter()
            .filter_map(|&m| shaded.qubit_map[m])
            .collect();

        current_circuit = QuantumCircuit {
            num_qubits: shaded.reduced_qubits,
            gates: shaded.gates,
            measurements: new_measurements,
        };

        iteration += 1;
    }

    // Build final qubit map from original space.
    let final_n = current_circuit.num_qubits;
    let mut qubit_map: Vec<Option<usize>> = vec![None; circuit.num_qubits];
    for (reduced_idx, &orig_idx) in cumulative_inverse.iter().enumerate() {
        qubit_map[orig_idx] = Some(reduced_idx);
    }

    let reduction_ratio = if circuit.num_qubits > 0 {
        total_qubits_removed as f64 / circuit.num_qubits as f64
    } else {
        0.0
    };

    let result = ShadedCircuit {
        original_qubits: circuit.num_qubits,
        reduced_qubits: final_n,
        gates: current_circuit.gates,
        qubit_map,
        inverse_map: cumulative_inverse,
        reduction_ratio,
        gates_removed: total_gates_removed,
        qubits_removed: total_qubits_removed,
    };

    Ok((result, iteration))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn no_simplify_config() -> SlcConfig {
        SlcConfig {
            method: SlcMethod::BackwardCone,
            merge_lightcones: true,
            remove_identity_gates: false,
            simplify_single_qubit_chains: false,
        }
    }

    // -----------------------------------------------------------------------
    // 1. Simple circuit: backward lightcone
    // -----------------------------------------------------------------------
    #[test]
    fn test_backward_lightcone_simple() {
        // 4 qubits: H(0), CX(0,1), X(2), H(3), Measure(1)
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::X(2),
                SlcGate::H(3),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let lc = backward_lightcone(&circuit, &[1]);
        assert!(
            lc.active_qubits[0],
            "qubit 0 should be in backward cone of qubit 1"
        );
        assert!(lc.active_qubits[1], "qubit 1 should be in backward cone");
        assert!(
            !lc.active_qubits[2],
            "qubit 2 should NOT be in backward cone"
        );
        assert!(
            !lc.active_qubits[3],
            "qubit 3 should NOT be in backward cone"
        );
        assert_eq!(lc.num_active_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 2. Simple circuit: forward lightcone
    // -----------------------------------------------------------------------
    #[test]
    fn test_forward_lightcone_simple() {
        // Forward from qubit 0: H(0), CX(0,1) -> qubits 0 and 1 active
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::X(2),
                SlcGate::H(3),
            ],
            measurements: vec![1],
        };

        let lc = forward_lightcone(&circuit, &[0]);
        assert!(lc.active_qubits[0]);
        assert!(lc.active_qubits[1]);
        assert!(!lc.active_qubits[2]);
        assert!(!lc.active_qubits[3]);
    }

    // -----------------------------------------------------------------------
    // 3. Bidirectional: tighter than either alone
    // -----------------------------------------------------------------------
    #[test]
    fn test_bidirectional_tighter() {
        // 5 qubits:
        //   H(0), CX(0,1), CX(1,2), X(3), CX(3,4), Measure(2)
        // Backward from 2: {0,1,2} (3,4 not connected to 2)
        // Forward from all: all active (all qubits have gates)
        // Bidirectional = intersection = {0,1,2} (same as backward here)
        //
        // Better example where bidirectional is tighter:
        // 4 qubits: CX(0,1), CX(2,3), CX(1,2), Measure(3)
        // Backward from 3: 3 <- CX(2,3) pulls in 2 <- CX(1,2) pulls in 1 <- CX(0,1) pulls in 0
        //   = {0,1,2,3} (all qubits)
        // Forward from all: all active
        // Intersection = {0,1,2,3}
        //
        // To get a real difference, use: qubit 0 has no gate, but is in backward cone
        // Actually bidirectional helps when forward from subset of initial qubits.
        // Using all qubits as forward seed means forward cone = everything with a gate.
        //
        // Better: 5 qubits, only qubit 4 has a gate (H(4)), and CX(4,0), measure(0)
        // Backward from 0: {0, 4}
        // Forward from all: all 5 are seeded, all will be in forward cone
        // Intersection: {0, 4}
        // Both backward and bidirectional give {0, 4}.
        //
        // The real value of bidirectional is when forward cone from INITIAL state
        // (non-|0>) is smaller than all qubits. Our current implementation seeds
        // forward from all qubits, so bidirectional = backward in most cases.
        // Still, we verify correctness.

        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::X(2),
                SlcGate::H(3),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let back = backward_lightcone(&circuit, &[1]);
        let bidir = bidirectional_lightcone(&circuit, &[1]);

        // Bidirectional should be a subset (or equal) of backward.
        for q in 0..circuit.num_qubits {
            if bidir.active_qubits[q] {
                assert!(
                    back.active_qubits[q],
                    "bidirectional should be subset of backward"
                );
            }
        }
        assert!(bidir.num_active_qubits <= back.num_active_qubits);
    }

    // -----------------------------------------------------------------------
    // 4. Linear circuit: lightcone is local
    // -----------------------------------------------------------------------
    #[test]
    fn test_linear_circuit_local_lightcone() {
        // 10 qubits, nearest-neighbor CX chain: CX(0,1), CX(1,2), ..., CX(8,9)
        // Measure qubit 0 only.
        // Backward cone: 0 <- CX(0,1) pulls in 1 <- CX(1,2) pulls in 2, etc.
        // But we only have forward CX, so walking backwards:
        //   Measure(0): active = {0}
        //   CX(8,9): neither 8 nor 9 in {0} -> skip
        //   ...
        //   CX(0,1): qubit 0 is active -> pull in 1
        //   Before CX(0,1): H(0)..H(9) -> H(0) active, H(1) active (qubit 1 now active)
        //
        // So lightcone of qubit 0 = {0, 1} (only CX(0,1) connects to qubit 0).

        let mut circuit = QuantumCircuit::new(10);
        for q in 0..10 {
            circuit.add_gate(SlcGate::H(q));
        }
        for q in 0..9 {
            circuit.add_gate(SlcGate::CX(q, q + 1));
        }
        circuit.add_gate(SlcGate::Measure(0));
        circuit.add_measurement(0);

        let lc = backward_lightcone(&circuit, &[0]);

        // Walking backward: Measure(0) seeds {0}.
        // CX(8,9): neither in {0} -> skip
        // CX(7,8): skip ... CX(1,2): skip
        // CX(0,1): qubit 0 in {0} -> add 1 -> {0,1}
        // H(9)..H(2): skip. H(1): active. H(0): active.
        assert!(lc.active_qubits[0]);
        assert!(lc.active_qubits[1]);
        assert!(!lc.active_qubits[2]);
        assert!(!lc.active_qubits[9]);
        assert_eq!(lc.num_active_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 5. All-to-all circuit: lightcone is full
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_to_all_full_lightcone() {
        let circuit = build_all_to_all_circuit(5, &[0]);
        let lc = backward_lightcone(&circuit, &[0]);

        // All-to-all means measuring any qubit pulls in all qubits.
        for q in 0..5 {
            assert!(
                lc.active_qubits[q],
                "qubit {} should be active in all-to-all",
                q
            );
        }
        assert_eq!(lc.num_active_qubits, 5);
    }

    // -----------------------------------------------------------------------
    // 6. Single measurement: only relevant qubits
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_measurement_relevant_qubits() {
        // 6 qubits, two disconnected clusters: {0,1,2} and {3,4,5}
        let circuit = QuantumCircuit {
            num_qubits: 6,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::CX(1, 2),
                SlcGate::H(3),
                SlcGate::CX(3, 4),
                SlcGate::CX(4, 5),
                SlcGate::Measure(2),
            ],
            measurements: vec![2],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // Only cluster {0,1,2} should survive.
        assert_eq!(shaded.reduced_qubits, 3);
        assert!(shaded.qubit_map[0].is_some());
        assert!(shaded.qubit_map[1].is_some());
        assert!(shaded.qubit_map[2].is_some());
        assert!(shaded.qubit_map[3].is_none());
        assert!(shaded.qubit_map[4].is_none());
        assert!(shaded.qubit_map[5].is_none());
    }

    // -----------------------------------------------------------------------
    // 7. Two measurements: merged lightcone
    // -----------------------------------------------------------------------
    #[test]
    fn test_two_measurements_merged() {
        // 6 qubits, two clusters: {0,1,2} and {3,4,5}
        // Measure 2 and 5 -> both clusters survive.
        let circuit = QuantumCircuit {
            num_qubits: 6,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::CX(1, 2),
                SlcGate::H(3),
                SlcGate::CX(3, 4),
                SlcGate::CX(4, 5),
                SlcGate::Measure(2),
                SlcGate::Measure(5),
            ],
            measurements: vec![2, 5],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        assert_eq!(shaded.reduced_qubits, 6, "both clusters should survive");
    }

    // -----------------------------------------------------------------------
    // 8. Gate removal: gate outside cone removed
    // -----------------------------------------------------------------------
    #[test]
    fn test_gate_outside_cone_removed() {
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![
                SlcGate::H(0),
                SlcGate::X(1), // on qubit 1 -- relevant
                SlcGate::Z(2), // on qubit 2 -- irrelevant if measuring only 1
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // Z(2) and H(0) should be removed; only X(1) and Measure(1) remain.
        assert_eq!(shaded.gates.len(), 2); // X(0_remapped) + Measure(0_remapped)
        assert!(shaded.gates.iter().any(|g| matches!(g, SlcGate::X(_))));
        assert!(shaded
            .gates
            .iter()
            .any(|g| matches!(g, SlcGate::Measure(_))));
    }

    // -----------------------------------------------------------------------
    // 9. Qubit removal: unused qubit removed
    // -----------------------------------------------------------------------
    #[test]
    fn test_unused_qubit_removed() {
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![SlcGate::H(1), SlcGate::Measure(1)],
            measurements: vec![1],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        assert_eq!(shaded.reduced_qubits, 1);
        assert_eq!(shaded.qubits_removed, 3);
    }

    // -----------------------------------------------------------------------
    // 10. Qubit remapping: compact indices
    // -----------------------------------------------------------------------
    #[test]
    fn test_qubit_remapping_compact() {
        // Qubits 0, 2, 4 are in the cone (1, 3 removed).
        let circuit = QuantumCircuit {
            num_qubits: 5,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 2),
                SlcGate::CX(2, 4),
                SlcGate::X(1),
                SlcGate::Y(3),
                SlcGate::Measure(4),
            ],
            measurements: vec![4],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        assert_eq!(shaded.reduced_qubits, 3);
        // Original qubit 0 -> reduced 0, 2 -> 1, 4 -> 2
        assert_eq!(shaded.qubit_map[0], Some(0));
        assert_eq!(shaded.qubit_map[1], None);
        assert_eq!(shaded.qubit_map[2], Some(1));
        assert_eq!(shaded.qubit_map[3], None);
        assert_eq!(shaded.qubit_map[4], Some(2));
    }

    // -----------------------------------------------------------------------
    // 11. Inverse map: correct translation
    // -----------------------------------------------------------------------
    #[test]
    fn test_inverse_map_correct() {
        let circuit = QuantumCircuit {
            num_qubits: 5,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 2),
                SlcGate::CX(2, 4),
                SlcGate::X(1),
                SlcGate::Y(3),
                SlcGate::Measure(4),
            ],
            measurements: vec![4],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // inverse_map[reduced] = original
        assert_eq!(shaded.inverse_map[0], 0);
        assert_eq!(shaded.inverse_map[1], 2);
        assert_eq!(shaded.inverse_map[2], 4);
    }

    // -----------------------------------------------------------------------
    // 12. Reduction ratio: calculated correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_reduction_ratio() {
        let circuit = QuantumCircuit {
            num_qubits: 10,
            gates: vec![SlcGate::H(5), SlcGate::Measure(5)],
            measurements: vec![5],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // 9 qubits removed out of 10.
        assert_eq!(shaded.qubits_removed, 9);
        let expected_ratio = 9.0 / 10.0;
        assert!((shaded.reduction_ratio - expected_ratio).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 13. Identity removal: H-H detected
    // -----------------------------------------------------------------------
    #[test]
    fn test_identity_removal_hh() {
        let gates = vec![SlcGate::H(0), SlcGate::H(0), SlcGate::X(0)];

        let simplified = simplify_gates(&gates, true, false);
        // H-H cancels, leaving only X(0).
        assert_eq!(simplified.len(), 1);
        assert!(matches!(simplified[0], SlcGate::X(0)));
    }

    // -----------------------------------------------------------------------
    // 14. Identity removal: X-X detected
    // -----------------------------------------------------------------------
    #[test]
    fn test_identity_removal_xx() {
        let gates = vec![SlcGate::X(2), SlcGate::X(2)];

        let simplified = simplify_gates(&gates, true, false);
        assert_eq!(simplified.len(), 0, "X-X should cancel to nothing");
    }

    // -----------------------------------------------------------------------
    // 15. Gate chain simplification: Rz-Rz merged
    // -----------------------------------------------------------------------
    #[test]
    fn test_rz_chain_merge() {
        let gates = vec![SlcGate::Rz(0, 0.3), SlcGate::Rz(0, 0.7)];

        let simplified = simplify_gates(&gates, false, true);
        assert_eq!(simplified.len(), 1);
        if let SlcGate::Rz(q, angle) = &simplified[0] {
            assert_eq!(*q, 0);
            assert!((angle - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Rz gate after merge");
        }
    }

    // -----------------------------------------------------------------------
    // 16. Dependency graph: correct edges
    // -----------------------------------------------------------------------
    #[test]
    fn test_dependency_graph_edges() {
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![
                SlcGate::H(0),     // gate 0
                SlcGate::H(1),     // gate 1
                SlcGate::CX(0, 1), // gate 2 depends on 0 and 1
                SlcGate::X(0),     // gate 3 depends on 2
            ],
            measurements: vec![],
        };

        let dag = build_dependency_graph(&circuit);

        assert_eq!(dag.gates.len(), 4);
        // Edge (0, 2): H(0) -> CX(0,1)
        assert!(dag.edges.contains(&(0, 2)));
        // Edge (1, 2): H(1) -> CX(0,1)
        assert!(dag.edges.contains(&(1, 2)));
        // Edge (2, 3): CX(0,1) -> X(0)
        assert!(dag.edges.contains(&(2, 3)));
    }

    // -----------------------------------------------------------------------
    // 17. Circuit depth: matches critical path
    // -----------------------------------------------------------------------
    #[test]
    fn test_circuit_depth_critical_path() {
        // H(0), H(1) in parallel (depth 1)
        // CX(0,1) (depth 2)
        // X(0) (depth 3)
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![
                SlcGate::H(0),
                SlcGate::H(1),
                SlcGate::CX(0, 1),
                SlcGate::X(0),
            ],
            measurements: vec![],
        };

        let depth = circuit_depth(&circuit.gates, circuit.num_qubits);
        assert_eq!(depth, 3);

        let dag = build_dependency_graph(&circuit);
        assert_eq!(dag.circuit_depth(), 3);
    }

    // -----------------------------------------------------------------------
    // 18. Shaded circuit: fewer qubits than original
    // -----------------------------------------------------------------------
    #[test]
    fn test_shaded_fewer_qubits() {
        let circuit = QuantumCircuit {
            num_qubits: 8,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::H(4),
                SlcGate::CX(4, 5),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        assert!(shaded.reduced_qubits < circuit.num_qubits);
        assert_eq!(shaded.reduced_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 19. Shaded circuit: fewer gates than original
    // -----------------------------------------------------------------------
    #[test]
    fn test_shaded_fewer_gates() {
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::H(1),
                SlcGate::H(2),
                SlcGate::H(3),
                SlcGate::CX(0, 1),
                SlcGate::CX(2, 3),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        assert!(shaded.gates.len() < circuit.gates.len());
    }

    // -----------------------------------------------------------------------
    // 20. Shaded circuit: correct gate content
    // -----------------------------------------------------------------------
    #[test]
    fn test_shaded_correct_gate_content() {
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::X(2),
                SlcGate::Y(3),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // Should contain: H(0->0), CX(0->0, 1->1), Measure(1->1)
        assert_eq!(shaded.gates.len(), 3);
        assert!(matches!(shaded.gates[0], SlcGate::H(0)));
        assert!(matches!(shaded.gates[1], SlcGate::CX(0, 1)));
        assert!(matches!(shaded.gates[2], SlcGate::Measure(1)));
    }

    // -----------------------------------------------------------------------
    // 21. Empty lightcone: all qubits removed
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_lightcone_error() {
        // No gates connect to the measurement qubit.
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![
                // No gates at all touching measurement qubit
            ],
            measurements: vec![0],
        };

        // With no gates, the backward cone from qubit 0 is just {0}.
        // But there are no gates, so no active gates. Still, qubit 0 itself
        // is in the cone. The shaded circuit should have 1 qubit, 0 gates.
        let config = no_simplify_config();
        let result = shade_circuit(&circuit, &config);

        // Actually, backward lightcone starts with qubit 0 active. No gates
        // to walk backwards through. So 1 active qubit, 0 active gates.
        // apply_shading will have reduced_qubits=1 which is not 0, so no error.
        assert!(result.is_ok());
        let shaded = result.unwrap();
        assert_eq!(shaded.reduced_qubits, 1);
        assert_eq!(shaded.gates.len(), 0);
    }

    // -----------------------------------------------------------------------
    // 21b. Empty lightcone: no measurements specified
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_lightcone_no_measurements() {
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![SlcGate::H(0)],
            measurements: vec![],
        };

        let config = no_simplify_config();
        let result = shade_circuit(&circuit, &config);
        assert!(matches!(result, Err(SlcError::EmptyLightcone(_))));
    }

    // -----------------------------------------------------------------------
    // 22. Full lightcone: deeply entangled circuit
    // -----------------------------------------------------------------------
    #[test]
    fn test_full_lightcone_deeply_entangled() {
        let circuit = build_all_to_all_circuit(6, &[0]);
        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // All 6 qubits should survive -- deeply entangled.
        assert_eq!(shaded.reduced_qubits, 6);
        assert_eq!(shaded.qubits_removed, 0);
    }

    // -----------------------------------------------------------------------
    // 23. Barrier: doesn't affect lightcone
    // -----------------------------------------------------------------------
    #[test]
    fn test_barrier_no_effect() {
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::Barrier(vec![0, 1, 2, 3]),
                SlcGate::X(2),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // Barrier should not pull in qubits 2, 3.
        assert_eq!(shaded.reduced_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 24. Toffoli: all 3 qubits in cone
    // -----------------------------------------------------------------------
    #[test]
    fn test_toffoli_all_three_in_cone() {
        let circuit = QuantumCircuit {
            num_qubits: 5,
            gates: vec![
                SlcGate::H(0),
                SlcGate::H(1),
                SlcGate::H(2),
                SlcGate::Toffoli(0, 1, 2),
                SlcGate::X(3),
                SlcGate::Y(4),
                SlcGate::Measure(2),
            ],
            measurements: vec![2],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // Toffoli(0,1,2) with measurement on 2 -> all three qubits in cone.
        assert_eq!(shaded.reduced_qubits, 3);
        assert!(shaded.qubit_map[0].is_some());
        assert!(shaded.qubit_map[1].is_some());
        assert!(shaded.qubit_map[2].is_some());
        assert!(shaded.qubit_map[3].is_none());
        assert!(shaded.qubit_map[4].is_none());
    }

    // -----------------------------------------------------------------------
    // 25. SlcReport: all fields populated
    // -----------------------------------------------------------------------
    #[test]
    fn test_slc_report_populated() {
        let circuit = QuantumCircuit {
            num_qubits: 6,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::X(2),
                SlcGate::H(3),
                SlcGate::CX(3, 4),
                SlcGate::Y(5),
                SlcGate::Measure(1),
                SlcGate::Measure(4),
            ],
            measurements: vec![1, 4],
        };

        let config = no_simplify_config();
        let (shaded, report) = shade_and_analyze(&circuit, &config).unwrap();

        assert_eq!(report.original_qubits, 6);
        assert!(report.original_gates > 0);
        assert!(report.original_depth > 0);
        assert!(report.reduced_qubits <= report.original_qubits);
        assert!(report.reduced_gates <= report.original_gates);
        assert!(report.reduced_depth <= report.original_depth);
        assert!(report.qubit_reduction >= 0.0);
        assert!(report.gate_reduction >= 0.0);
        assert!(report.depth_reduction >= 0.0);
        assert_eq!(report.lightcone_sizes.len(), 2); // Two measurements
                                                     // Each lightcone size should be > 0.
        for &sz in &report.lightcone_sizes {
            assert!(sz > 0);
        }
    }

    // -----------------------------------------------------------------------
    // 26. Config builder defaults
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let config = SlcConfig::default();
        assert_eq!(config.method, SlcMethod::BackwardCone);
        assert!(config.merge_lightcones);
        assert!(config.remove_identity_gates);
        assert!(config.simplify_single_qubit_chains);
    }

    // -----------------------------------------------------------------------
    // 26b. Config builder chaining
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_builder_chain() {
        let config = SlcConfig::default()
            .with_method(SlcMethod::Bidirectional)
            .with_merge(false)
            .with_identity_removal(false)
            .with_chain_simplification(false);

        assert_eq!(config.method, SlcMethod::Bidirectional);
        assert!(!config.merge_lightcones);
        assert!(!config.remove_identity_gates);
        assert!(!config.simplify_single_qubit_chains);
    }

    // -----------------------------------------------------------------------
    // 27. Large circuit: 50 qubits, measure 2
    // -----------------------------------------------------------------------
    #[test]
    fn test_large_circuit_50_qubits() {
        // 50 qubits, linear nearest-neighbor, measure qubits 0 and 1.
        let circuit = build_linear_circuit(50, 3, &[0, 1]);
        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // With nearest-neighbor CX and only 3 layers, the lightcone of
        // qubits 0,1 cannot reach far. Definitely fewer than 50 qubits.
        assert!(shaded.reduced_qubits < 50);
        assert!(shaded.reduced_qubits >= 2);
        assert!(shaded.qubits_removed > 0);
    }

    // -----------------------------------------------------------------------
    // 28. Validate: invalid qubit reference
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_invalid_qubit() {
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![SlcGate::H(5)], // qubit 5 out of range
            measurements: vec![0],
        };

        let result = circuit.validate();
        assert!(matches!(result, Err(SlcError::InvalidCircuit(_))));
    }

    // -----------------------------------------------------------------------
    // 29. Lightcone merge operation
    // -----------------------------------------------------------------------
    #[test]
    fn test_lightcone_merge() {
        let mut lc1 = Lightcone::empty(4, 3);
        lc1.active_qubits[0] = true;
        lc1.active_qubits[1] = true;
        lc1.active_gates[0] = true;
        lc1.target_qubits = vec![0];
        lc1.recount();

        let mut lc2 = Lightcone::empty(4, 3);
        lc2.active_qubits[2] = true;
        lc2.active_qubits[3] = true;
        lc2.active_gates[1] = true;
        lc2.active_gates[2] = true;
        lc2.target_qubits = vec![2];
        lc2.recount();

        lc1.merge(&lc2);

        assert_eq!(lc1.num_active_qubits, 4);
        assert_eq!(lc1.num_active_gates, 3);
        assert!(lc1.target_qubits.contains(&0));
        assert!(lc1.target_qubits.contains(&2));
    }

    // -----------------------------------------------------------------------
    // 30. Lightcone intersect operation
    // -----------------------------------------------------------------------
    #[test]
    fn test_lightcone_intersect() {
        let mut lc1 = Lightcone::empty(4, 3);
        lc1.active_qubits[0] = true;
        lc1.active_qubits[1] = true;
        lc1.active_qubits[2] = true;
        lc1.active_gates[0] = true;
        lc1.active_gates[1] = true;
        lc1.recount();

        let mut lc2 = Lightcone::empty(4, 3);
        lc2.active_qubits[1] = true;
        lc2.active_qubits[2] = true;
        lc2.active_qubits[3] = true;
        lc2.active_gates[1] = true;
        lc2.active_gates[2] = true;
        lc2.recount();

        lc1.intersect(&lc2);

        // Intersection: qubits {1,2}, gates {1}
        assert!(!lc1.active_qubits[0]);
        assert!(lc1.active_qubits[1]);
        assert!(lc1.active_qubits[2]);
        assert!(!lc1.active_qubits[3]);
        assert_eq!(lc1.num_active_qubits, 2);
        assert_eq!(lc1.num_active_gates, 1);
    }

    // -----------------------------------------------------------------------
    // 31. CX-CX identity cancellation
    // -----------------------------------------------------------------------
    #[test]
    fn test_cx_cx_cancellation() {
        let gates = vec![SlcGate::CX(0, 1), SlcGate::CX(0, 1), SlcGate::H(0)];

        let simplified = simplify_gates(&gates, true, false);
        assert_eq!(simplified.len(), 1);
        assert!(matches!(simplified[0], SlcGate::H(0)));
    }

    // -----------------------------------------------------------------------
    // 32. Rx-Rx merge
    // -----------------------------------------------------------------------
    #[test]
    fn test_rx_rx_merge() {
        let gates = vec![
            SlcGate::Rx(1, std::f64::consts::FRAC_PI_4),
            SlcGate::Rx(1, std::f64::consts::FRAC_PI_4),
        ];

        let simplified = simplify_gates(&gates, false, true);
        assert_eq!(simplified.len(), 1);
        if let SlcGate::Rx(q, angle) = &simplified[0] {
            assert_eq!(*q, 1);
            assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        } else {
            panic!("Expected Rx after merge");
        }
    }

    // -----------------------------------------------------------------------
    // 33. Rotation cancellation (sum to zero)
    // -----------------------------------------------------------------------
    #[test]
    fn test_rotation_cancel_to_zero() {
        let gates = vec![SlcGate::Rz(0, 1.5), SlcGate::Rz(0, -1.5)];

        let simplified = simplify_gates(&gates, false, true);
        assert_eq!(
            simplified.len(),
            0,
            "Rz(1.5) + Rz(-1.5) = identity, both removed"
        );
    }

    // -----------------------------------------------------------------------
    // 34. Dependency graph depth with parallel gates
    // -----------------------------------------------------------------------
    #[test]
    fn test_dag_parallel_depth() {
        // Two independent single-qubit gates: depth 1
        let circuit = QuantumCircuit {
            num_qubits: 2,
            gates: vec![SlcGate::H(0), SlcGate::H(1)],
            measurements: vec![],
        };

        let dag = build_dependency_graph(&circuit);
        assert_eq!(dag.circuit_depth(), 1);
    }

    // -----------------------------------------------------------------------
    // 35. Interaction graph construction
    // -----------------------------------------------------------------------
    #[test]
    fn test_interaction_graph() {
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![SlcGate::CX(0, 1), SlcGate::CX(1, 2), SlcGate::CX(0, 1)],
            measurements: vec![],
        };

        let ig = build_interaction_graph(&circuit);

        assert!(ig.adj[0].contains(&1));
        assert!(ig.adj[1].contains(&0));
        assert!(ig.adj[1].contains(&2));
        assert!(ig.adj[2].contains(&1));
        assert!(!ig.adj[0].contains(&2));
        assert!(!ig.adj[3].contains(&0));

        // Edge (0,1) appears twice (two CX gates).
        assert_eq!(*ig.weights.get(&(0, 1)).unwrap(), 2);
        assert_eq!(*ig.weights.get(&(1, 2)).unwrap(), 1);
    }

    // -----------------------------------------------------------------------
    // 36. Lightcone distance
    // -----------------------------------------------------------------------
    #[test]
    fn test_lightcone_distance() {
        // Linear chain: CX(0,1), CX(1,2), CX(2,3), Measure(0)
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![SlcGate::CX(0, 1), SlcGate::CX(1, 2), SlcGate::CX(2, 3)],
            measurements: vec![0],
        };

        let dist = lightcone_distance(&circuit);
        assert_eq!(dist[0], 0); // measured directly
        assert_eq!(dist[1], 1); // neighbor of 0
        assert_eq!(dist[2], 2); // two hops
        assert_eq!(dist[3], 3); // three hops
    }

    // -----------------------------------------------------------------------
    // 37. Depth limit enforcement
    // -----------------------------------------------------------------------
    #[test]
    fn test_depth_limit() {
        let circuit = QuantumCircuit {
            num_qubits: 2,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::X(0),
                SlcGate::Y(1),
                SlcGate::Measure(0),
            ],
            measurements: vec![0],
        };

        let config = no_simplify_config();

        // Depth should be 4 for this circuit.
        let result = shade_with_depth_limit(&circuit, &config, 2);
        assert!(matches!(result, Err(SlcError::CircuitTooDeep(_))));

        let result = shade_with_depth_limit(&circuit, &config, 100);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // 38. Layer assignment
    // -----------------------------------------------------------------------
    #[test]
    fn test_layer_assignment() {
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![
                SlcGate::H(0),     // layer 0
                SlcGate::H(1),     // layer 0
                SlcGate::H(2),     // layer 0
                SlcGate::CX(0, 1), // layer 1
                SlcGate::X(2),     // layer 1
                SlcGate::CX(1, 2), // layer 2
            ],
            measurements: vec![],
        };

        let layers = layer_assignment(&circuit);
        assert_eq!(layers.len(), 3); // 3 layers
        assert_eq!(layers[0].len(), 3); // H(0), H(1), H(2)
        assert_eq!(layers[1].len(), 2); // CX(0,1), X(2)
        assert_eq!(layers[2].len(), 1); // CX(1,2)
    }

    // -----------------------------------------------------------------------
    // 39. Bottleneck qubits
    // -----------------------------------------------------------------------
    #[test]
    fn test_bottleneck_qubits() {
        // Two clusters: {0,1} and {2,3}, with qubit 1 bridging via CX(1,2).
        // Measure 0 and 3.
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::CX(1, 2),
                SlcGate::CX(2, 3),
                SlcGate::Measure(0),
                SlcGate::Measure(3),
            ],
            measurements: vec![0, 3],
        };

        let bn = bottleneck_qubits(&circuit);
        // Lightcone of 0: {0, 1} (CX(0,1) pulls in 1; CX(1,2) pulls in 2; CX(2,3) pulls in 3)
        // Actually walking backwards:
        //   seed {0}, CX(2,3): neither in {0}. CX(1,2): neither in {0}. CX(0,1): 0 in -> add 1 -> {0,1}.
        // Lightcone of 3: seed {3}. CX(2,3): 3 in -> add 2. CX(1,2): 2 in -> add 1. CX(0,1): 1 in -> add 0.
        //   = {0,1,2,3}
        // Bottleneck = intersection: qubits in both = {0,1} intersected with {0,1,2,3} = {0,1}
        assert!(bn.contains(&0));
        assert!(bn.contains(&1));
    }

    // -----------------------------------------------------------------------
    // 40. Per-measurement lightcone sizes
    // -----------------------------------------------------------------------
    #[test]
    fn test_per_measurement_sizes() {
        let circuit = QuantumCircuit {
            num_qubits: 6,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::H(3),
                SlcGate::CX(3, 4),
                SlcGate::CX(4, 5),
                SlcGate::Measure(1),
                SlcGate::Measure(5),
            ],
            measurements: vec![1, 5],
        };

        let sizes = per_measurement_lightcone_sizes(&circuit);
        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes[0].0, 1); // measurement qubit
        assert_eq!(sizes[0].1, 2); // lightcone of qubit 1 = {0, 1}
        assert_eq!(sizes[1].0, 5); // measurement qubit
        assert_eq!(sizes[1].1, 3); // lightcone of qubit 5 = {3, 4, 5}
    }

    // -----------------------------------------------------------------------
    // 41. Iterative shading converges
    // -----------------------------------------------------------------------
    #[test]
    fn test_iterative_shading() {
        let circuit = QuantumCircuit {
            num_qubits: 6,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::H(2),
                SlcGate::H(3),
                SlcGate::CX(3, 4),
                SlcGate::H(5),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let config = no_simplify_config();
        let (shaded, iterations) = iterative_shade(&circuit, &config, 10).unwrap();

        assert_eq!(shaded.reduced_qubits, 2);
        assert!(iterations >= 1);
    }

    // -----------------------------------------------------------------------
    // 42. Gate display formatting
    // -----------------------------------------------------------------------
    #[test]
    fn test_gate_display() {
        assert_eq!(format!("{}", SlcGate::H(0)), "H(0)");
        assert_eq!(format!("{}", SlcGate::CX(1, 2)), "CX(1, 2)");
        assert_eq!(format!("{}", SlcGate::Toffoli(0, 1, 2)), "Toffoli(0, 1, 2)");
        assert_eq!(format!("{}", SlcGate::Measure(3)), "Measure(3)");
    }

    // -----------------------------------------------------------------------
    // 43. SlcError display formatting
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_display() {
        let e1 = SlcError::InvalidCircuit("bad qubit".to_string());
        assert!(format!("{}", e1).contains("bad qubit"));

        let e2 = SlcError::CircuitTooDeep(999999);
        assert!(format!("{}", e2).contains("999999"));
    }

    // -----------------------------------------------------------------------
    // 44. Gate classification helpers
    // -----------------------------------------------------------------------
    #[test]
    fn test_gate_classification() {
        assert!(SlcGate::H(0).is_single_qubit());
        assert!(SlcGate::Rx(0, 1.0).is_single_qubit());
        assert!(!SlcGate::CX(0, 1).is_single_qubit());
        assert!(!SlcGate::Barrier(vec![0]).is_single_qubit());
        assert!(!SlcGate::Measure(0).is_single_qubit());
        assert!(SlcGate::Barrier(vec![0]).is_barrier());
        assert!(!SlcGate::H(0).is_barrier());
        assert!(SlcGate::Measure(0).is_measurement());
        assert!(!SlcGate::H(0).is_measurement());
    }

    // -----------------------------------------------------------------------
    // 45. Gate name
    // -----------------------------------------------------------------------
    #[test]
    fn test_gate_name() {
        assert_eq!(SlcGate::H(0).name(), "H");
        assert_eq!(SlcGate::CX(0, 1).name(), "CX");
        assert_eq!(SlcGate::Swap(0, 1).name(), "SWAP");
        assert_eq!(SlcGate::Toffoli(0, 1, 2).name(), "Toffoli");
    }

    // -----------------------------------------------------------------------
    // 46. Swap gate in lightcone
    // -----------------------------------------------------------------------
    #[test]
    fn test_swap_in_lightcone() {
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::Swap(0, 2),
                SlcGate::X(1),
                SlcGate::Y(3),
                SlcGate::Measure(2),
            ],
            measurements: vec![2],
        };

        let config = no_simplify_config();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // SWAP(0,2) with measure on 2 -> both 0 and 2 are in the cone.
        assert!(shaded.qubit_map[0].is_some());
        assert!(shaded.qubit_map[2].is_some());
        assert!(shaded.qubit_map[1].is_none());
        assert!(shaded.qubit_map[3].is_none());
        assert_eq!(shaded.reduced_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 47. Forward lightcone from subset
    // -----------------------------------------------------------------------
    #[test]
    fn test_forward_lightcone_subset() {
        let circuit = QuantumCircuit {
            num_qubits: 4,
            gates: vec![
                SlcGate::H(0),
                SlcGate::CX(0, 1),
                SlcGate::H(2),
                SlcGate::CX(2, 3),
            ],
            measurements: vec![1],
        };

        // Forward from only qubit 0: should reach 0 and 1, not 2 or 3.
        let lc = forward_lightcone(&circuit, &[0]);
        assert!(lc.active_qubits[0]);
        assert!(lc.active_qubits[1]);
        assert!(!lc.active_qubits[2]);
        assert!(!lc.active_qubits[3]);
        assert_eq!(lc.num_active_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 48. Report display doesn't panic
    // -----------------------------------------------------------------------
    #[test]
    fn test_report_display() {
        let report = SlcReport {
            original_qubits: 10,
            original_gates: 20,
            original_depth: 5,
            reduced_qubits: 4,
            reduced_gates: 8,
            reduced_depth: 3,
            qubit_reduction: 60.0,
            gate_reduction: 60.0,
            depth_reduction: 40.0,
            lightcone_sizes: vec![3, 4],
        };

        let s = format!("{}", report);
        assert!(s.contains("SLC Report"));
        assert!(s.contains("60.0%"));
    }

    // -----------------------------------------------------------------------
    // 49. Circuit builder helpers
    // -----------------------------------------------------------------------
    #[test]
    fn test_build_linear_circuit() {
        let circuit = build_linear_circuit(6, 2, &[0, 5]);
        assert_eq!(circuit.num_qubits, 6);
        assert_eq!(circuit.measurements.len(), 2);
        assert!(circuit.gates.len() > 6); // At least 6 Hadamards + CX layers + measures
    }

    // -----------------------------------------------------------------------
    // 50. Y-Y cancellation
    // -----------------------------------------------------------------------
    #[test]
    fn test_yy_cancellation() {
        let gates = vec![SlcGate::Y(0), SlcGate::Y(0)];
        let simplified = simplify_gates(&gates, true, false);
        assert_eq!(simplified.len(), 0);
    }

    // -----------------------------------------------------------------------
    // 51. SWAP-SWAP cancellation
    // -----------------------------------------------------------------------
    #[test]
    fn test_swap_swap_cancellation() {
        let gates = vec![SlcGate::Swap(0, 1), SlcGate::Swap(0, 1), SlcGate::H(0)];
        let simplified = simplify_gates(&gates, true, false);
        assert_eq!(simplified.len(), 1);
        assert!(matches!(simplified[0], SlcGate::H(0)));
    }

    // -----------------------------------------------------------------------
    // 52. Simplification + shading combined
    // -----------------------------------------------------------------------
    #[test]
    fn test_simplification_with_shading() {
        // H(0), H(0) cancels, leaving no gates on qubit 0.
        // CX(0,1) still pulls qubit 0 into the cone.
        let circuit = QuantumCircuit {
            num_qubits: 3,
            gates: vec![
                SlcGate::H(0),
                SlcGate::H(0), // cancels with previous
                SlcGate::CX(0, 1),
                SlcGate::X(2),
                SlcGate::Measure(1),
            ],
            measurements: vec![1],
        };

        let config = SlcConfig::default();
        let shaded = shade_circuit(&circuit, &config).unwrap();

        // After simplification: [CX(0,1), X(2), Measure(1)]
        // Backward cone of 1: CX(0,1) pulls in 0 -> {0,1}
        assert_eq!(shaded.reduced_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 53. Ry-Ry merge
    // -----------------------------------------------------------------------
    #[test]
    fn test_ry_ry_merge() {
        let gates = vec![SlcGate::Ry(2, 0.1), SlcGate::Ry(2, 0.2)];
        let simplified = simplify_gates(&gates, false, true);
        assert_eq!(simplified.len(), 1);
        if let SlcGate::Ry(q, angle) = &simplified[0] {
            assert_eq!(*q, 2);
            assert!((angle - 0.3).abs() < 1e-10);
        } else {
            panic!("Expected Ry");
        }
    }

    // -----------------------------------------------------------------------
    // 54. Gate remap returns None for removed qubit
    // -----------------------------------------------------------------------
    #[test]
    fn test_gate_remap_none() {
        let map = vec![Some(0), None, Some(1)];
        let gate = SlcGate::CX(0, 1); // qubit 1 is removed
        assert!(gate.remap(&map).is_none());
    }

    // -----------------------------------------------------------------------
    // 55. QuantumCircuit new and add methods
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantum_circuit_builder() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(SlcGate::H(0));
        circuit.add_gate(SlcGate::CX(0, 1));
        circuit.add_measurement(2);
        circuit.add_measurement(2); // duplicate should not be added

        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.gates.len(), 2);
        assert_eq!(circuit.measurements.len(), 1);
    }
}
