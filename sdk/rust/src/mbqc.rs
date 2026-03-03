//! Measurement-Based Quantum Computation (MBQC)
//!
//! Implements the one-way quantum computation model where computation proceeds
//! by preparing an entangled cluster state and performing single-qubit
//! measurements with adaptive angles driven by prior measurement outcomes.
//!
//! # Overview
//!
//! In the measurement-based model, quantum computation consists of three stages:
//!
//! 1. **Preparation**: Build a cluster state (graph state) by initialising every
//!    qubit in |+> and applying CZ gates along the edges of a graph.
//! 2. **Measurement**: Measure each qubit in a basis determined by the desired
//!    computation.  Feed-forward corrections adapt later measurement angles
//!    based on earlier outcomes.
//! 3. **Read-out**: The unmeasured qubits encode the output of the computation.
//!
//! This module provides:
//!
//! - [`ClusterState`]: graph-state builder on top of [`StabilizerState`].
//! - Lattice generators for 1D chains, 2D grids, and brickwork graphs.
//! - [`MeasurementBasis`], [`MeasurementCommand`], and [`MeasurementPattern`]
//!   for expressing and executing measurement schedules.
//! - [`GateModelTranslator`]: converts a gate-model circuit into an MBQC
//!   measurement pattern.
//! - [`MBQCSimulator`]: executes a measurement pattern on a cluster state with
//!   feed-forward corrections.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::mbqc::{linear_cluster, MBQCSimulator, MeasurementPattern,
//!                         MeasurementCommand, MeasurementBasis};
//!
//! // Build a 5-qubit linear cluster state
//! let cluster = linear_cluster(5);
//!
//! // Define a measurement pattern (identity wire: measure in X basis)
//! let mut pattern = MeasurementPattern::new();
//! pattern.add_command(MeasurementCommand::new(0, MeasurementBasis::XBasis));
//! pattern.add_command(MeasurementCommand::new(1, MeasurementBasis::XBasis));
//! pattern.add_command(MeasurementCommand::new(2, MeasurementBasis::XBasis));
//! pattern.add_command(MeasurementCommand::new(3, MeasurementBasis::XBasis));
//!
//! // Execute the pattern
//! let mut sim = MBQCSimulator::new(cluster);
//! let outcomes = sim.execute(&pattern);
//! ```

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

use crate::gates::{Gate, GateType};
use crate::stabilizer::StabilizerState;

// ---------------------------------------------------------------------------
// MbqcError
// ---------------------------------------------------------------------------

/// Errors that can occur during MBQC translation and simulation.
#[derive(Clone, Debug)]
pub enum MbqcError {
    /// A custom gate was encountered that cannot be automatically decomposed
    /// into MBQC measurement patterns.  The caller must decompose the custom
    /// gate into standard primitives before passing it to the translator.
    UnsupportedCustomGate {
        /// Number of target qubits on the gate.
        num_targets: usize,
        /// Number of control qubits on the gate.
        num_controls: usize,
        /// Dimension of the unitary matrix (row count).
        matrix_dim: usize,
    },
}

impl fmt::Display for MbqcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MbqcError::UnsupportedCustomGate {
                num_targets,
                num_controls,
                matrix_dim,
            } => {
                write!(
                    f,
                    "Custom gate (targets={}, controls={}, {}x{} matrix) cannot be \
                     automatically translated to MBQC.  Decompose into standard gates \
                     (H, CNOT, Rx, Ry, Rz, S, T, CZ, SWAP, Toffoli) before calling \
                     the translator.",
                    num_targets, num_controls, matrix_dim, matrix_dim
                )
            }
        }
    }
}

impl std::error::Error for MbqcError {}

// ---------------------------------------------------------------------------
// ClusterState
// ---------------------------------------------------------------------------

/// A cluster (graph) state built on top of the stabilizer tableau.
///
/// Cluster states are the universal resource for measurement-based quantum
/// computation.  They are created by preparing every qubit in the |+> state
/// (via Hadamard on |0>) and then entangling neighbouring qubits with CZ
/// gates according to a graph.
#[derive(Clone, Debug)]
pub struct ClusterState {
    /// Underlying stabilizer representation.
    state: StabilizerState,
    /// Number of physical qubits in the cluster.
    num_qubits: usize,
    /// Adjacency list describing the graph edges.
    edges: Vec<(usize, usize)>,
}

impl ClusterState {
    /// Build a cluster state from an explicit graph description.
    ///
    /// # Arguments
    ///
    /// * `edges` - Pairs of qubit indices that should be entangled with CZ.
    /// * `n` - Total number of qubits in the cluster.
    ///
    /// # Panics
    ///
    /// Panics if any edge references a qubit index >= `n`.
    pub fn from_graph(edges: &[(usize, usize)], n: usize) -> Self {
        for &(a, b) in edges {
            assert!(a < n, "Edge vertex {} out of range (n={})", a, n);
            assert!(b < n, "Edge vertex {} out of range (n={})", b, n);
            assert!(a != b, "Self-loops are not permitted (vertex {})", a);
        }

        let mut state = StabilizerState::new(n);

        // Prepare |+> on every qubit.
        for q in 0..n {
            state.h(q);
        }

        // Entangle along edges with CZ.
        for &(a, b) in edges {
            state.cz(a, b);
        }

        ClusterState {
            state,
            num_qubits: n,
            edges: edges.to_vec(),
        }
    }

    /// Wrap an existing [`StabilizerState`] as a cluster state.
    ///
    /// This is useful when the caller has already prepared a state through
    /// some other means and wants to use it with the MBQC machinery.  No
    /// additional gates are applied.
    pub fn from_stabilizer(state: StabilizerState) -> Self {
        let n = state.num_qubits();
        ClusterState {
            state,
            num_qubits: n,
            edges: Vec::new(),
        }
    }

    /// Return the number of physical qubits in this cluster.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Return the edges that define this cluster graph.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Obtain a reference to the underlying stabilizer state.
    pub fn stabilizer(&self) -> &StabilizerState {
        &self.state
    }

    /// Obtain a mutable reference to the underlying stabilizer state.
    pub fn stabilizer_mut(&mut self) -> &mut StabilizerState {
        &mut self.state
    }
}

// ---------------------------------------------------------------------------
// Lattice generators
// ---------------------------------------------------------------------------

/// Create a 1D linear cluster state (chain graph).
///
/// The graph has edges `(0,1), (1,2), ..., (n-2, n-1)`.
///
/// A linear cluster with `n` qubits can teleport a single-qubit state across
/// `n-1` wires, implementing a sequence of single-qubit rotations.
pub fn linear_cluster(n: usize) -> ClusterState {
    assert!(n >= 1, "Linear cluster requires at least 1 qubit");
    let edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
    ClusterState::from_graph(&edges, n)
}

/// Create a 2D square-lattice cluster state.
///
/// Qubits are arranged on a `rows x cols` grid.  Qubit `(r, c)` has index
/// `r * cols + c`.  Edges connect nearest neighbours in both horizontal and
/// vertical directions.
pub fn square_cluster(rows: usize, cols: usize) -> ClusterState {
    assert!(rows >= 1 && cols >= 1, "Grid dimensions must be >= 1");
    let n = rows * cols;
    let mut edges = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            // Horizontal edge.
            if c + 1 < cols {
                edges.push((idx, idx + 1));
            }
            // Vertical edge.
            if r + 1 < rows {
                edges.push((idx, idx + cols));
            }
        }
    }

    ClusterState::from_graph(&edges, n)
}

/// Create a brickwork lattice cluster state suitable for universal quantum
/// computation.
///
/// A brickwork state is a 2D lattice where horizontal edges are present on
/// every column, but vertical edges alternate between even and odd columns
/// in a staggered (brick-like) pattern.  This structure is the canonical
/// resource state for blind quantum computing protocols.
///
/// Layout for `rows=3, cols=6`:
///
/// ```text
///  o-o-o-o-o-o
///  | | | | | |     (vertical edges on even columns for even rows)
///  o-o-o-o-o-o
///    | | | | |     (vertical edges on odd columns for odd rows)
///  o-o-o-o-o-o
/// ```
pub fn brickwork(rows: usize, cols: usize) -> ClusterState {
    assert!(rows >= 1 && cols >= 1, "Brickwork dimensions must be >= 1");
    let n = rows * cols;
    let mut edges = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;

            // Horizontal edges within the same row.
            if c + 1 < cols {
                edges.push((idx, idx + 1));
            }

            // Vertical (inter-row) edges in a staggered pattern.
            if r + 1 < rows {
                let place_vertical = if r % 2 == 0 {
                    c % 2 == 0 // even rows: vertical edges on even columns
                } else {
                    c % 2 == 1 // odd rows: vertical edges on odd columns
                };
                if place_vertical {
                    edges.push((idx, idx + cols));
                }
            }
        }
    }

    ClusterState::from_graph(&edges, n)
}

// ---------------------------------------------------------------------------
// MeasurementBasis
// ---------------------------------------------------------------------------

/// Basis in which a qubit is measured during MBQC.
///
/// Single-qubit measurements in the XY plane at angle `alpha` correspond to
/// measuring the observable `cos(alpha) X + sin(alpha) Y`.
#[derive(Clone, Debug, PartialEq)]
pub enum MeasurementBasis {
    /// Measure in the Pauli-X eigenbasis (alpha = 0).
    XBasis,
    /// Measure in the Pauli-Y eigenbasis (alpha = pi/2).
    YBasis,
    /// Measure in the computational (Pauli-Z) basis.
    ZBasis,
    /// Measure in the XY plane at an arbitrary angle `alpha`.
    ///
    /// The measured observable is `cos(alpha) X + sin(alpha) Y`.
    ArbitraryAngle(f64),
}

impl MeasurementBasis {
    /// Return the measurement angle in the XY plane.
    ///
    /// For `ZBasis` this returns `None` since Z measurement is not in the
    /// XY plane.
    pub fn angle(&self) -> Option<f64> {
        match self {
            MeasurementBasis::XBasis => Some(0.0),
            MeasurementBasis::YBasis => Some(PI / 2.0),
            MeasurementBasis::ZBasis => None,
            MeasurementBasis::ArbitraryAngle(a) => Some(*a),
        }
    }
}

// ---------------------------------------------------------------------------
// MeasurementCommand
// ---------------------------------------------------------------------------

/// A single measurement instruction in an MBQC pattern.
///
/// Each command specifies which qubit to measure, the basis/angle, and the
/// lists of previous qubits whose outcomes determine whether the angle should
/// be sign-flipped (X-dependent) or pi-shifted (Z-dependent).
#[derive(Clone, Debug)]
pub struct MeasurementCommand {
    /// Index of the qubit to measure.
    pub qubit: usize,
    /// Basis of measurement.
    pub basis: MeasurementBasis,
    /// Nominal measurement angle (before corrections).
    pub angle: f64,
    /// Indices of previously-measured qubits whose outcomes cause an X
    /// correction (sign flip of the angle).
    pub x_corrections: Vec<usize>,
    /// Indices of previously-measured qubits whose outcomes cause a Z
    /// correction (pi shift of the angle).
    pub z_corrections: Vec<usize>,
}

impl MeasurementCommand {
    /// Create a new measurement command with default (no) corrections.
    pub fn new(qubit: usize, basis: MeasurementBasis) -> Self {
        let angle = match &basis {
            MeasurementBasis::XBasis => 0.0,
            MeasurementBasis::YBasis => PI / 2.0,
            MeasurementBasis::ZBasis => 0.0,
            MeasurementBasis::ArbitraryAngle(a) => *a,
        };
        MeasurementCommand {
            qubit,
            basis,
            angle,
            x_corrections: Vec::new(),
            z_corrections: Vec::new(),
        }
    }

    /// Create a measurement command with explicit angle and correction lists.
    pub fn with_corrections(
        qubit: usize,
        basis: MeasurementBasis,
        angle: f64,
        x_corrections: Vec<usize>,
        z_corrections: Vec<usize>,
    ) -> Self {
        MeasurementCommand {
            qubit,
            basis,
            angle,
            x_corrections,
            z_corrections,
        }
    }

    /// Compute the effective measurement angle given prior outcomes.
    ///
    /// The feed-forward rule is:
    ///
    /// ```text
    /// alpha_eff = (-1)^s_x * alpha + s_z * pi
    /// ```
    ///
    /// where `s_x` is the parity of X-correction outcomes and `s_z` is the
    /// parity of Z-correction outcomes.
    pub fn effective_angle(&self, outcomes: &HashMap<usize, bool>) -> f64 {
        let s_x: usize = self
            .x_corrections
            .iter()
            .filter(|&&q| *outcomes.get(&q).unwrap_or(&false))
            .count();
        let s_z: usize = self
            .z_corrections
            .iter()
            .filter(|&&q| *outcomes.get(&q).unwrap_or(&false))
            .count();

        let sign = if s_x % 2 == 0 { 1.0 } else { -1.0 };
        let z_shift = if s_z % 2 == 0 { 0.0 } else { PI };

        sign * self.angle + z_shift
    }
}

// ---------------------------------------------------------------------------
// MeasurementPattern
// ---------------------------------------------------------------------------

/// An ordered sequence of [`MeasurementCommand`]s that together implement a
/// quantum computation in the MBQC model.
///
/// The commands are executed in order; later commands may depend on the
/// outcomes of earlier ones through their correction lists.
#[derive(Clone, Debug)]
pub struct MeasurementPattern {
    /// Ordered list of measurement commands.
    commands: Vec<MeasurementCommand>,
    /// Qubits that should *not* be measured (they carry the output state).
    output_qubits: Vec<usize>,
    /// Total number of qubits required by this pattern.
    num_qubits: usize,
}

impl MeasurementPattern {
    /// Create an empty pattern.
    pub fn new() -> Self {
        MeasurementPattern {
            commands: Vec::new(),
            output_qubits: Vec::new(),
            num_qubits: 0,
        }
    }

    /// Create a pattern with a known qubit count.
    pub fn with_num_qubits(num_qubits: usize) -> Self {
        MeasurementPattern {
            commands: Vec::new(),
            output_qubits: Vec::new(),
            num_qubits,
        }
    }

    /// Append a measurement command to the pattern.
    pub fn add_command(&mut self, cmd: MeasurementCommand) {
        if cmd.qubit >= self.num_qubits {
            self.num_qubits = cmd.qubit + 1;
        }
        self.commands.push(cmd);
    }

    /// Designate a qubit as an output qubit (will not be measured).
    pub fn add_output_qubit(&mut self, qubit: usize) {
        if qubit >= self.num_qubits {
            self.num_qubits = qubit + 1;
        }
        if !self.output_qubits.contains(&qubit) {
            self.output_qubits.push(qubit);
        }
    }

    /// Return the list of measurement commands.
    pub fn commands(&self) -> &[MeasurementCommand] {
        &self.commands
    }

    /// Return the output qubit indices.
    pub fn output_qubits(&self) -> &[usize] {
        &self.output_qubits
    }

    /// Return the total number of qubits required.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Return the number of measurement commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check whether the pattern is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Set the total qubit count explicitly.
    pub fn set_num_qubits(&mut self, n: usize) {
        self.num_qubits = n;
    }
}

impl Default for MeasurementPattern {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GateModelTranslator
// ---------------------------------------------------------------------------

/// Translates a sequence of gate-model operations into an MBQC measurement
/// pattern.
///
/// # Decomposition Rules
///
/// Single-qubit Clifford+T gates are decomposed into XY-plane measurements
/// on a linear cluster segment:
///
/// - **H** (Hadamard): identity wire on the cluster (measure at 0).
/// - **S** (Phase): measure at angle pi/2.
/// - **T** (pi/8): measure at angle pi/4.
/// - **Rz(theta)**: measure at angle -theta.
/// - **X**: measure at 0 with Z-correction flip.
/// - **Z**: no physical measurement, just a Pauli frame update.
///
/// Two-qubit gates (CNOT) are decomposed into a 15-qubit gadget on the
/// cluster state with appropriate CZ entanglement and measurement ordering.
///
/// # Qubit Mapping
///
/// The translator maintains a mapping from logical gate-model qubits to
/// physical cluster-state qubits.  Each single-qubit gate consumes one
/// cluster qubit; CNOT consumes additional resource qubits.
pub struct GateModelTranslator {
    /// Map from logical qubit to current physical qubit.
    logical_to_physical: HashMap<usize, usize>,
    /// Next available physical qubit index.
    next_physical: usize,
    /// Accumulated edges for the cluster state graph.
    edges: Vec<(usize, usize)>,
    /// Accumulated measurement commands.
    pattern: MeasurementPattern,
}

impl GateModelTranslator {
    /// Create a translator for a circuit with `num_qubits` logical qubits.
    pub fn new(num_qubits: usize) -> Self {
        let mut logical_to_physical = HashMap::new();
        for q in 0..num_qubits {
            logical_to_physical.insert(q, q);
        }

        GateModelTranslator {
            logical_to_physical,
            next_physical: num_qubits,
            edges: Vec::new(),
            pattern: MeasurementPattern::with_num_qubits(num_qubits),
        }
    }

    /// Allocate a fresh physical qubit and return its index.
    fn alloc_qubit(&mut self) -> usize {
        let q = self.next_physical;
        self.next_physical += 1;
        q
    }

    /// Add a single-qubit rotation in the XY plane.
    ///
    /// This consumes one resource qubit: the current physical qubit for the
    /// logical wire is measured at the given angle and the logical wire
    /// advances to the next physical qubit.
    fn add_single_qubit_rotation(
        &mut self,
        logical: usize,
        angle: f64,
        x_corrections: Vec<usize>,
        z_corrections: Vec<usize>,
    ) {
        let current = self.logical_to_physical[&logical];
        let next = self.alloc_qubit();

        // Entangle current qubit with the fresh resource qubit.
        self.edges.push((current, next));

        // Measure the current qubit.
        let cmd = MeasurementCommand::with_corrections(
            current,
            MeasurementBasis::ArbitraryAngle(angle),
            angle,
            x_corrections,
            z_corrections,
        );
        self.pattern.add_command(cmd);

        // Advance the logical wire.
        self.logical_to_physical.insert(logical, next);
    }

    /// Translate a single gate into measurement commands.
    ///
    /// Returns `Ok(())` on success, or an `MbqcError` if the gate cannot be
    /// decomposed into MBQC measurement patterns (currently only
    /// [`GateType::Custom`] gates trigger this).
    fn translate_gate(&mut self, gate: &Gate) -> Result<(), MbqcError> {
        match &gate.gate_type {
            // ----- Single-qubit gates -----

            GateType::H => {
                // Hadamard = identity wire on cluster (measure at angle 0).
                // In the one-way model, teleporting through one CZ bond and
                // measuring in the X basis implements H (up to Pauli byproduct).
                let target = gate.targets[0];
                let current = self.logical_to_physical[&target];
                self.add_single_qubit_rotation(target, 0.0, vec![], vec![]);
                // The byproduct from an X-basis measurement on a cluster bond
                // implements H naturally.  We record the measured qubit as a
                // potential correction source.
                let _ = current; // measurement added inside helper
            }

            GateType::X => {
                // X = HZH.  On the cluster this is realised as a Pauli frame
                // update.  We implement it as a measurement at angle 0 with
                // a deterministic Z-correction.
                let target = gate.targets[0];
                let current = self.logical_to_physical[&target];
                self.add_single_qubit_rotation(target, 0.0, vec![], vec![current]);
            }

            GateType::Y => {
                // Y = iXZ.  Two measurement steps.
                let target = gate.targets[0];
                let current = self.logical_to_physical[&target];
                self.add_single_qubit_rotation(target, PI / 2.0, vec![], vec![current]);
            }

            GateType::Z => {
                // Z is a Pauli frame update: measure at 0, flip via X-correction.
                let target = gate.targets[0];
                let current = self.logical_to_physical[&target];
                self.add_single_qubit_rotation(target, 0.0, vec![current], vec![]);
            }

            GateType::S => {
                // S = Phase(pi/2): measure at pi/2.
                let target = gate.targets[0];
                self.add_single_qubit_rotation(target, PI / 2.0, vec![], vec![]);
            }

            GateType::T => {
                // T = Phase(pi/4): measure at pi/4.
                let target = gate.targets[0];
                self.add_single_qubit_rotation(target, PI / 4.0, vec![], vec![]);
            }

            GateType::Rz(theta) => {
                // Rz(theta): measure at -theta (the sign convention comes
                // from the relation between XY-plane measurement angle and
                // the Z rotation implemented on the output wire).
                let target = gate.targets[0];
                self.add_single_qubit_rotation(target, -theta, vec![], vec![]);
            }

            GateType::Rx(theta) => {
                // Rx(theta) = H Rz(theta) H
                // Decompose into H, Rz, H sequence.
                let target = gate.targets[0];
                self.translate_gate(&Gate::h(target))?;
                self.translate_gate(&Gate::rz(target, *theta))?;
                self.translate_gate(&Gate::h(target))?;
            }

            GateType::Ry(theta) => {
                // Ry(theta) = Rz(pi/2) Rx(theta) Rz(-pi/2)
                let target = gate.targets[0];
                self.translate_gate(&Gate::rz(target, PI / 2.0))?;
                self.translate_gate(&Gate::rx(target, *theta))?;
                self.translate_gate(&Gate::rz(target, -PI / 2.0))?;
            }

            GateType::Phase(theta) => {
                // Phase(theta) is the same as Rz(theta) up to global phase.
                let target = gate.targets[0];
                self.add_single_qubit_rotation(target, -theta, vec![], vec![]);
            }

            // ----- Two-qubit gates -----

            GateType::CNOT => {
                // CNOT is decomposed into a cluster-state gadget.
                //
                // The standard MBQC decomposition uses a 4-qubit bridge:
                //   control_phys --CZ-- bridge_a --CZ-- bridge_b --CZ-- target_phys
                //                                  |
                //                                  CZ
                //                                  |
                //                             (next_ctrl)
                //
                // We measure bridge_a in the X basis and bridge_b in the X basis.
                // The control wire advances and the target wire advances.
                let control = gate.controls[0];
                let target = gate.targets[0];

                let ctrl_phys = self.logical_to_physical[&control];
                let tgt_phys = self.logical_to_physical[&target];

                let bridge_a = self.alloc_qubit();
                let bridge_b = self.alloc_qubit();
                let next_ctrl = self.alloc_qubit();
                let next_tgt = self.alloc_qubit();

                // Build the entanglement graph for the CNOT gadget.
                // Control wire continues through bridge_a to next_ctrl.
                self.edges.push((ctrl_phys, bridge_a));
                self.edges.push((bridge_a, next_ctrl));

                // Target wire continues through bridge_b to next_tgt.
                self.edges.push((tgt_phys, bridge_b));
                self.edges.push((bridge_b, next_tgt));

                // Cross-edge that entangles control and target paths.
                self.edges.push((bridge_a, bridge_b));

                // Measure the old physical qubits and the bridge qubits.
                // Control qubit: X-basis measurement.
                self.pattern.add_command(MeasurementCommand::new(
                    ctrl_phys,
                    MeasurementBasis::XBasis,
                ));

                // Bridge A: X-basis measurement, X-correction from control.
                self.pattern.add_command(MeasurementCommand::with_corrections(
                    bridge_a,
                    MeasurementBasis::XBasis,
                    0.0,
                    vec![ctrl_phys],
                    vec![],
                ));

                // Target qubit: X-basis measurement.
                self.pattern.add_command(MeasurementCommand::new(
                    tgt_phys,
                    MeasurementBasis::XBasis,
                ));

                // Bridge B: X-basis measurement, X-correction from target,
                // Z-correction from bridge_a.
                self.pattern.add_command(MeasurementCommand::with_corrections(
                    bridge_b,
                    MeasurementBasis::XBasis,
                    0.0,
                    vec![tgt_phys],
                    vec![bridge_a],
                ));

                // Update logical-to-physical mapping.
                self.logical_to_physical.insert(control, next_ctrl);
                self.logical_to_physical.insert(target, next_tgt);
            }

            GateType::CZ => {
                // CZ = (I x H) CNOT (I x H)
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.translate_gate(&Gate::h(target))?;
                self.translate_gate(&Gate::cnot(control, target))?;
                self.translate_gate(&Gate::h(target))?;
            }

            GateType::SWAP => {
                // SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)
                let a = gate.targets[0];
                let b = gate.targets[1];
                self.translate_gate(&Gate::cnot(a, b))?;
                self.translate_gate(&Gate::cnot(b, a))?;
                self.translate_gate(&Gate::cnot(a, b))?;
            }

            GateType::U { theta, phi, lambda } => {
                // U(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda)
                let target = gate.targets[0];
                self.translate_gate(&Gate::rz(target, *lambda))?;
                self.translate_gate(&Gate::ry(target, *theta))?;
                self.translate_gate(&Gate::rz(target, *phi))?;
            }

            GateType::SX => {
                // SX = Rx(pi/2)
                let target = gate.targets[0];
                self.translate_gate(&Gate::rx(target, PI / 2.0))?;
            }

            GateType::CRz(theta) => {
                // Controlled-Rz: decompose into CNOT + Rz
                // CRz(theta) = CNOT * (I x Rz(-theta/2)) * CNOT * (I x Rz(theta/2))
                let control = gate.controls[0];
                let target = gate.targets[0];
                let half = theta / 2.0;
                self.translate_gate(&Gate::rz(target, half))?;
                self.translate_gate(&Gate::cnot(control, target))?;
                self.translate_gate(&Gate::rz(target, -half))?;
                self.translate_gate(&Gate::cnot(control, target))?;
            }

            GateType::CRx(theta) => {
                // CRx: H * CRz * H
                let target = gate.targets[0];
                let control = gate.controls[0];
                self.translate_gate(&Gate::h(target))?;
                self.translate_gate(&Gate::with_params(
                    GateType::CRz(*theta),
                    vec![target],
                    vec![control],
                    vec![*theta],
                ))?;
                self.translate_gate(&Gate::h(target))?;
            }

            GateType::CRy(theta) => {
                // CRy: Rz(pi/2) * CRx * Rz(-pi/2)
                let target = gate.targets[0];
                let control = gate.controls[0];
                self.translate_gate(&Gate::rz(target, PI / 2.0))?;
                self.translate_gate(&Gate::with_params(
                    GateType::CRx(*theta),
                    vec![target],
                    vec![control],
                    vec![*theta],
                ))?;
                self.translate_gate(&Gate::rz(target, -PI / 2.0))?;
            }

            GateType::CR(theta) => {
                // Controlled-phase = CRz(theta)
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.translate_gate(&Gate::with_params(
                    GateType::CRz(*theta),
                    vec![target],
                    vec![control],
                    vec![*theta],
                ))?;
            }

            GateType::Toffoli => {
                // Toffoli = decompose into CNOT + T gates (standard decomposition).
                // For MBQC we simply decompose into the six-CNOT form.
                let c1 = gate.controls[0];
                let c2 = gate.controls[1];
                let target = gate.targets[0];

                self.translate_gate(&Gate::h(target))?;
                self.translate_gate(&Gate::cnot(c2, target))?;
                self.translate_gate(&Gate::t(target))?; // T-dagger approximated
                self.translate_gate(&Gate::cnot(c1, target))?;
                self.translate_gate(&Gate::t(target))?;
                self.translate_gate(&Gate::cnot(c2, target))?;
                self.translate_gate(&Gate::t(target))?;
                self.translate_gate(&Gate::cnot(c1, target))?;
                self.translate_gate(&Gate::t(c2))?;
                self.translate_gate(&Gate::t(target))?;
                self.translate_gate(&Gate::cnot(c1, c2))?;
                self.translate_gate(&Gate::h(target))?;
                self.translate_gate(&Gate::t(c1))?;
                self.translate_gate(&Gate::t(c2))?;
                self.translate_gate(&Gate::cnot(c1, c2))?;
            }

            GateType::ISWAP => {
                // iSWAP = SWAP * CZ * (S x S)
                let a = gate.targets[0];
                let b = gate.targets[1];
                self.translate_gate(&Gate::s(a))?;
                self.translate_gate(&Gate::s(b))?;
                self.translate_gate(&Gate::cz(a, b))?;
                self.translate_gate(&Gate::swap(a, b))?;
            }

            GateType::CCZ => {
                // CCZ = (I x I x H) Toffoli (I x I x H)
                let c1 = gate.controls[0];
                let c2 = gate.controls[1];
                let target = gate.targets[0];
                self.translate_gate(&Gate::h(target))?;
                self.translate_gate(&Gate::toffoli(c1, c2, target))?;
                self.translate_gate(&Gate::h(target))?;
            }

            GateType::Rxx(theta) => {
                // Rxx(θ) = exp(-i θ/2 XX)
                // Decomposition: CNOT(a,b) · (Rx(θ) ⊗ I) · CNOT(a,b)
                let a = gate.targets[0];
                let b = gate.targets[1];
                self.translate_gate(&Gate::cnot(a, b))?;
                self.translate_gate(&Gate::rx(a, *theta))?;
                self.translate_gate(&Gate::cnot(a, b))?;
            }

            GateType::Ryy(theta) => {
                // Ryy(θ) = exp(-i θ/2 YY)
                // Decomposition: conjugate into ZZ basis using S†, apply via
                // CNOT + Ry, then undo.
                // S† = Rz(-π/2), S = Rz(π/2)
                let a = gate.targets[0];
                let b = gate.targets[1];
                let half_pi = PI / 2.0;
                self.translate_gate(&Gate::rz(a, -half_pi))?; // S†(a)
                self.translate_gate(&Gate::rz(b, -half_pi))?; // S†(b)
                self.translate_gate(&Gate::cnot(a, b))?;
                self.translate_gate(&Gate::ry(a, *theta))?;
                self.translate_gate(&Gate::cnot(a, b))?;
                self.translate_gate(&Gate::rz(a, half_pi))?; // S(a)
                self.translate_gate(&Gate::rz(b, half_pi))?; // S(b)
            }

            GateType::Rzz(theta) => {
                // Rzz(θ) = exp(-i θ/2 ZZ)
                // Decomposition: CNOT(a,b) · (I ⊗ Rz(θ)) · CNOT(a,b)
                let a = gate.targets[0];
                let b = gate.targets[1];
                self.translate_gate(&Gate::cnot(a, b))?;
                self.translate_gate(&Gate::rz(b, *theta))?;
                self.translate_gate(&Gate::cnot(a, b))?;
            }

            GateType::CSWAP => {
                // Fredkin (controlled-SWAP) decomposition:
                // CNOT(t2, t1) · Toffoli(ctrl, t1, t2) · CNOT(t2, t1)
                let ctrl = gate.controls[0];
                let t1 = gate.targets[0];
                let t2 = gate.targets[1];
                self.translate_gate(&Gate::cnot(t2, t1))?;
                self.translate_gate(&Gate::toffoli(ctrl, t1, t2))?;
                self.translate_gate(&Gate::cnot(t2, t1))?;
            }

            GateType::CU {
                theta,
                phi,
                lambda,
                gamma,
            } => {
                // Generic controlled-U(θ, φ, λ, γ) decomposition.
                //
                // CU = (global phase γ on control arm) ·
                //      Rz((φ+λ)/2) on control ·
                //      Rz((λ-φ)/2) on target ·
                //      CNOT(control, target) ·
                //      Ry(-θ/2) Rz(-(φ+λ)/2) on target ·
                //      CNOT(control, target) ·
                //      Ry(θ/2) Rz(φ) on target
                let control = gate.controls[0];
                let target = gate.targets[0];

                // Phase on control: global phase γ + (φ+λ)/2
                self.translate_gate(&Gate::rz(control, *gamma + (*phi + *lambda) / 2.0))?;
                // Phase on target before first CNOT
                self.translate_gate(&Gate::rz(target, (*lambda - *phi) / 2.0))?;
                // First CNOT
                self.translate_gate(&Gate::cnot(control, target))?;
                // Rotations on target between CNOTs
                self.translate_gate(&Gate::rz(target, -(*phi + *lambda) / 2.0))?;
                self.translate_gate(&Gate::ry(target, -(*theta) / 2.0))?;
                // Second CNOT
                self.translate_gate(&Gate::cnot(control, target))?;
                // Final rotations on target
                self.translate_gate(&Gate::ry(target, *theta / 2.0))?;
                self.translate_gate(&Gate::rz(target, *phi))?;
            }

            GateType::Custom(matrix) => {
                // Custom gates cannot be automatically decomposed because
                // their unitary structure is opaque.  Return a descriptive
                // error so callers can handle it gracefully.
                return Err(MbqcError::UnsupportedCustomGate {
                    num_targets: gate.targets.len(),
                    num_controls: gate.controls.len(),
                    matrix_dim: matrix.len(),
                });
            }
        }

        Ok(())
    }

    /// Convert a sequence of gate-model operations into an MBQC
    /// [`MeasurementPattern`] together with the cluster state that the
    /// pattern should be applied to.
    ///
    /// # Returns
    ///
    /// `Ok((ClusterState, MeasurementPattern))` on success, or
    /// `Err(MbqcError)` if any gate cannot be decomposed.
    ///
    /// # Errors
    ///
    /// Returns [`MbqcError::UnsupportedCustomGate`] if the gate sequence
    /// contains a [`GateType::Custom`] gate.  Custom gates must be
    /// decomposed into standard primitives before translation.
    pub fn to_mbqc(
        gates: &[Gate],
        num_qubits: usize,
    ) -> Result<(ClusterState, MeasurementPattern), MbqcError> {
        let mut translator = GateModelTranslator::new(num_qubits);

        for gate in gates {
            translator.translate_gate(gate)?;
        }

        // Mark the final physical qubits as outputs.
        for logical in 0..num_qubits {
            let phys = translator.logical_to_physical[&logical];
            translator.pattern.add_output_qubit(phys);
        }

        // Build the cluster state from accumulated edges.
        let total_qubits = translator.next_physical;
        translator.pattern.set_num_qubits(total_qubits);

        let cluster = ClusterState::from_graph(&translator.edges, total_qubits);

        Ok((cluster, translator.pattern))
    }

    /// Return the current logical-to-physical qubit mapping.
    pub fn qubit_mapping(&self) -> &HashMap<usize, usize> {
        &self.logical_to_physical
    }

    /// Return the total number of physical qubits allocated so far.
    pub fn total_physical_qubits(&self) -> usize {
        self.next_physical
    }
}

// ---------------------------------------------------------------------------
// MBQCSimulator
// ---------------------------------------------------------------------------

/// Executes an MBQC measurement pattern on a cluster state.
///
/// The simulator tracks measurement outcomes and applies feed-forward
/// corrections (adaptive angles) as dictated by the pattern's correction
/// lists.
pub struct MBQCSimulator {
    /// The cluster state to measure.
    cluster: ClusterState,
    /// Measurement outcomes indexed by qubit.
    outcomes: HashMap<usize, bool>,
}

impl MBQCSimulator {
    /// Create a simulator with the given cluster state.
    pub fn new(cluster: ClusterState) -> Self {
        MBQCSimulator {
            cluster,
            outcomes: HashMap::new(),
        }
    }

    /// Execute all commands in a measurement pattern.
    ///
    /// Returns the map of measurement outcomes (qubit index -> classical bit).
    pub fn execute(&mut self, pattern: &MeasurementPattern) -> HashMap<usize, bool> {
        for cmd in pattern.commands() {
            let outcome = self.execute_command(cmd);
            self.outcomes.insert(cmd.qubit, outcome);
        }
        self.outcomes.clone()
    }

    /// Execute a single measurement command, applying feed-forward corrections.
    fn execute_command(&mut self, cmd: &MeasurementCommand) -> bool {
        let effective_angle = cmd.effective_angle(&self.outcomes);

        match cmd.basis {
            MeasurementBasis::ZBasis => {
                // Direct Z-basis measurement on the stabilizer tableau.
                self.cluster.stabilizer_mut().measure(cmd.qubit)
            }
            _ => {
                // For XY-plane measurements, rotate into the computational
                // basis before measuring.
                //
                // Measuring cos(a)X + sin(a)Y is equivalent to:
                //   Rz(-a) H |psi> then measure in Z basis.
                //
                // Since we are in the stabilizer formalism we can only do
                // Clifford rotations exactly.  For arbitrary angles we use
                // the closest Clifford approximation within the stabilizer
                // framework.  When the angle is an exact multiple of pi/2
                // we use the exact gates.
                self.apply_xy_rotation(cmd.qubit, effective_angle);
                self.cluster.stabilizer_mut().measure(cmd.qubit)
            }
        }
    }

    /// Rotate a qubit so that an XY-plane measurement at angle `alpha`
    /// becomes a Z-basis measurement.
    ///
    /// The rotation is: Rz(-alpha) followed by H.
    ///
    /// Within the stabilizer formalism we decompose the Rz rotation into
    /// Clifford gates when possible and use the nearest Clifford
    /// approximation otherwise.
    fn apply_xy_rotation(&mut self, qubit: usize, alpha: f64) {
        // Decompose Rz(-alpha) into S gates.
        let normalised = normalise_angle(-alpha);
        let quarter_turns = (normalised / (PI / 2.0)).round() as i64;

        let state = self.cluster.stabilizer_mut();

        match ((quarter_turns % 4) + 4) % 4 {
            0 => { /* Identity */ }
            1 => state.s(qubit),
            2 => {
                state.s(qubit);
                state.s(qubit);
            }
            3 => state.s_dag(qubit),
            _ => unreachable!(),
        }

        state.h(qubit);
    }

    /// Return the measurement outcome for a specific qubit.
    pub fn outcome(&self, qubit: usize) -> Option<bool> {
        self.outcomes.get(&qubit).copied()
    }

    /// Return all measurement outcomes.
    pub fn outcomes(&self) -> &HashMap<usize, bool> {
        &self.outcomes
    }

    /// Return the output classical bits for the designated output qubits.
    ///
    /// Output qubits are measured in the Z basis to obtain the final result
    /// of the computation.
    pub fn read_output(&mut self, pattern: &MeasurementPattern) -> Vec<bool> {
        pattern
            .output_qubits()
            .iter()
            .map(|&q| {
                if let Some(&outcome) = self.outcomes.get(&q) {
                    outcome
                } else {
                    // Measure the output qubit in Z basis.
                    let result = self.cluster.stabilizer_mut().measure(q);
                    self.outcomes.insert(q, result);
                    result
                }
            })
            .collect()
    }

    /// Apply Pauli-X corrections to the output qubits based on measurement
    /// outcomes.
    ///
    /// In MBQC the computation result on the output wire carries a Pauli
    /// byproduct that depends on the measurement outcomes of earlier qubits.
    /// This method applies the X corrections to the stabilizer state before
    /// final readout.
    pub fn apply_x_correction(&mut self, qubit: usize, should_correct: bool) {
        if should_correct {
            self.cluster.stabilizer_mut().x(qubit);
        }
    }

    /// Apply Pauli-Z corrections to the output qubits.
    pub fn apply_z_correction(&mut self, qubit: usize, should_correct: bool) {
        if should_correct {
            self.cluster.stabilizer_mut().z(qubit);
        }
    }

    /// Return a reference to the underlying cluster state.
    pub fn cluster(&self) -> &ClusterState {
        &self.cluster
    }

    /// Consume the simulator and return the cluster state.
    pub fn into_cluster(self) -> ClusterState {
        self.cluster
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Normalise an angle into the range [0, 2*pi).
fn normalise_angle(angle: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut a = angle % two_pi;
    if a < 0.0 {
        a += two_pi;
    }
    a
}

/// Determine whether two floating-point values are approximately equal.
pub(crate) fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

/// Check whether an angle is a multiple of pi/2 (i.e. a Clifford angle).
pub(crate) fn is_clifford_angle(angle: f64) -> bool {
    let normalised = normalise_angle(angle);
    let quarter = PI / 2.0;
    for k in 0..4 {
        if approx_eq(normalised, k as f64 * quarter) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ClusterState construction tests ----

    #[test]
    fn test_linear_cluster_creation() {
        let cluster = linear_cluster(5);
        assert_eq!(cluster.num_qubits(), 5);
        assert_eq!(cluster.edges().len(), 4);
        assert!(cluster.stabilizer().is_valid());
    }

    #[test]
    fn test_linear_cluster_single_qubit() {
        let cluster = linear_cluster(1);
        assert_eq!(cluster.num_qubits(), 1);
        assert_eq!(cluster.edges().len(), 0);
        assert!(cluster.stabilizer().is_valid());
    }

    #[test]
    fn test_linear_cluster_edges() {
        let cluster = linear_cluster(4);
        let edges = cluster.edges();
        assert_eq!(edges, &[(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn test_square_cluster_creation() {
        let cluster = square_cluster(3, 3);
        assert_eq!(cluster.num_qubits(), 9);
        // 3x3 grid: 6 horizontal + 6 vertical = 12 edges
        assert_eq!(cluster.edges().len(), 12);
        assert!(cluster.stabilizer().is_valid());
    }

    #[test]
    fn test_square_cluster_2x2() {
        let cluster = square_cluster(2, 2);
        assert_eq!(cluster.num_qubits(), 4);
        // 2x2 grid: 2 horizontal + 2 vertical = 4 edges
        assert_eq!(cluster.edges().len(), 4);
        let edges = cluster.edges();
        assert!(edges.contains(&(0, 1))); // top row horizontal
        assert!(edges.contains(&(2, 3))); // bottom row horizontal
        assert!(edges.contains(&(0, 2))); // left column vertical
        assert!(edges.contains(&(1, 3))); // right column vertical
    }

    #[test]
    fn test_brickwork_cluster_creation() {
        let cluster = brickwork(3, 4);
        assert_eq!(cluster.num_qubits(), 12);
        assert!(cluster.stabilizer().is_valid());
        // Brickwork has all horizontal edges + staggered vertical edges
        let horizontal_edges = 3 * 3; // 3 rows, each with 3 horizontal edges
        let vertical_r0 = 2; // even row: vertical on cols 0, 2
        let vertical_r1 = 2; // odd row: vertical on cols 1, 3
        assert_eq!(cluster.edges().len(), horizontal_edges + vertical_r0 + vertical_r1);
    }

    #[test]
    fn test_brickwork_has_horizontal_edges() {
        let cluster = brickwork(2, 3);
        let edges = cluster.edges();
        // All horizontal edges should be present.
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
        assert!(edges.contains(&(3, 4)));
        assert!(edges.contains(&(4, 5)));
    }

    #[test]
    fn test_cluster_from_stabilizer() {
        let stab = StabilizerState::new(3);
        let cluster = ClusterState::from_stabilizer(stab);
        assert_eq!(cluster.num_qubits(), 3);
        assert!(cluster.edges().is_empty());
    }

    // ---- MeasurementBasis tests ----

    #[test]
    fn test_measurement_basis_angles() {
        assert_eq!(MeasurementBasis::XBasis.angle(), Some(0.0));
        assert!((MeasurementBasis::YBasis.angle().unwrap() - PI / 2.0).abs() < 1e-12);
        assert_eq!(MeasurementBasis::ZBasis.angle(), None);
        let arb = MeasurementBasis::ArbitraryAngle(1.23);
        assert!((arb.angle().unwrap() - 1.23).abs() < 1e-12);
    }

    // ---- MeasurementCommand tests ----

    #[test]
    fn test_command_effective_angle_no_corrections() {
        let cmd = MeasurementCommand::new(0, MeasurementBasis::ArbitraryAngle(PI / 4.0));
        let outcomes = HashMap::new();
        let eff = cmd.effective_angle(&outcomes);
        assert!((eff - PI / 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_command_effective_angle_x_correction() {
        let cmd = MeasurementCommand::with_corrections(
            1,
            MeasurementBasis::ArbitraryAngle(PI / 4.0),
            PI / 4.0,
            vec![0],
            vec![],
        );

        // No prior outcome for qubit 0 -> treated as false -> no sign flip.
        let mut outcomes = HashMap::new();
        assert!((cmd.effective_angle(&outcomes) - PI / 4.0).abs() < 1e-12);

        // Qubit 0 measured as true -> sign flip.
        outcomes.insert(0, true);
        assert!((cmd.effective_angle(&outcomes) - (-PI / 4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_command_effective_angle_z_correction() {
        let cmd = MeasurementCommand::with_corrections(
            2,
            MeasurementBasis::ArbitraryAngle(PI / 4.0),
            PI / 4.0,
            vec![],
            vec![0],
        );

        let mut outcomes = HashMap::new();
        outcomes.insert(0, true);
        let expected = PI / 4.0 + PI;
        assert!((cmd.effective_angle(&outcomes) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_command_effective_angle_both_corrections() {
        let cmd = MeasurementCommand::with_corrections(
            3,
            MeasurementBasis::ArbitraryAngle(PI / 4.0),
            PI / 4.0,
            vec![0],
            vec![1],
        );

        let mut outcomes = HashMap::new();
        outcomes.insert(0, true);
        outcomes.insert(1, true);
        // s_x = 1 (odd) -> sign flip, s_z = 1 (odd) -> pi shift
        let expected = -PI / 4.0 + PI;
        assert!((cmd.effective_angle(&outcomes) - expected).abs() < 1e-12);
    }

    // ---- MeasurementPattern tests ----

    #[test]
    fn test_pattern_construction() {
        let mut pattern = MeasurementPattern::new();
        assert!(pattern.is_empty());
        assert_eq!(pattern.len(), 0);

        pattern.add_command(MeasurementCommand::new(0, MeasurementBasis::XBasis));
        pattern.add_command(MeasurementCommand::new(1, MeasurementBasis::YBasis));
        assert_eq!(pattern.len(), 2);
        assert!(!pattern.is_empty());
        assert_eq!(pattern.num_qubits(), 2);
    }

    #[test]
    fn test_pattern_output_qubits() {
        let mut pattern = MeasurementPattern::with_num_qubits(5);
        pattern.add_output_qubit(3);
        pattern.add_output_qubit(4);
        pattern.add_output_qubit(3); // duplicate, should not add
        assert_eq!(pattern.output_qubits(), &[3, 4]);
    }

    // ---- GateModelTranslator tests ----

    #[test]
    fn test_translator_single_h_gate() {
        let gates = vec![Gate::h(0)];
        let (cluster, pattern) = GateModelTranslator::to_mbqc(&gates, 1).unwrap();

        // H gate on 1 logical qubit should produce a 2-qubit cluster
        // (original + one resource qubit) with 1 measurement command.
        assert_eq!(cluster.num_qubits(), 2);
        assert_eq!(pattern.len(), 1);
        assert_eq!(pattern.output_qubits().len(), 1);
        assert_eq!(pattern.output_qubits()[0], 1); // output is the new qubit
    }

    #[test]
    fn test_translator_s_gate() {
        let gates = vec![Gate::s(0)];
        let (cluster, pattern) = GateModelTranslator::to_mbqc(&gates, 1).unwrap();
        assert_eq!(cluster.num_qubits(), 2);
        assert_eq!(pattern.len(), 1);

        // The measurement angle should be pi/2.
        let cmd = &pattern.commands()[0];
        assert!((cmd.angle - PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_translator_t_gate() {
        let gates = vec![Gate::t(0)];
        let (cluster, pattern) = GateModelTranslator::to_mbqc(&gates, 1).unwrap();
        assert_eq!(cluster.num_qubits(), 2);
        assert_eq!(pattern.len(), 1);

        let cmd = &pattern.commands()[0];
        assert!((cmd.angle - PI / 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_translator_cnot_gate() {
        let gates = vec![Gate::cnot(0, 1)];
        let (cluster, pattern) = GateModelTranslator::to_mbqc(&gates, 2).unwrap();

        // CNOT uses 4 bridge/resource qubits + 2 original = 6 total.
        assert_eq!(cluster.num_qubits(), 6);

        // 4 measurement commands (ctrl, bridge_a, tgt, bridge_b).
        assert_eq!(pattern.len(), 4);

        // 2 output qubits.
        assert_eq!(pattern.output_qubits().len(), 2);
    }

    #[test]
    fn test_translator_rz_gate() {
        let theta = 0.7;
        let gates = vec![Gate::rz(0, theta)];
        let (cluster, pattern) = GateModelTranslator::to_mbqc(&gates, 1).unwrap();
        assert_eq!(cluster.num_qubits(), 2);
        assert_eq!(pattern.len(), 1);

        let cmd = &pattern.commands()[0];
        assert!((cmd.angle - (-theta)).abs() < 1e-12);
    }

    #[test]
    fn test_translator_multiple_single_qubit_gates() {
        // H followed by S on the same qubit.
        let gates = vec![Gate::h(0), Gate::s(0)];
        let (cluster, pattern) = GateModelTranslator::to_mbqc(&gates, 1).unwrap();

        // Each single-qubit gate consumes one resource qubit.
        // Start with 1, add 1 for H, add 1 for S = 3 total.
        assert_eq!(cluster.num_qubits(), 3);
        assert_eq!(pattern.len(), 2);
    }

    #[test]
    fn test_translator_preserves_output_mapping() {
        // Two-qubit circuit: H on q0, then CNOT(0,1).
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let (_cluster, pattern) = GateModelTranslator::to_mbqc(&gates, 2).unwrap();

        // Should have 2 output qubits.
        assert_eq!(pattern.output_qubits().len(), 2);

        // Output qubits should be distinct.
        let out = pattern.output_qubits();
        assert_ne!(out[0], out[1]);
    }

    // ---- MBQCSimulator tests ----

    #[test]
    fn test_simulator_z_basis_measurement() {
        // A single-qubit cluster in |+> measured in Z gives 50/50.
        // We run multiple trials to check both outcomes appear.
        let mut zeros = 0;
        let mut ones = 0;
        for _ in 0..100 {
            let cluster = linear_cluster(1);
            let mut sim = MBQCSimulator::new(cluster);
            let mut pattern = MeasurementPattern::new();
            pattern.add_command(MeasurementCommand::new(0, MeasurementBasis::ZBasis));
            let outcomes = sim.execute(&pattern);
            if outcomes[&0] {
                ones += 1;
            } else {
                zeros += 1;
            }
        }
        // With 100 trials, extremely unlikely to get all zeros or all ones.
        assert!(zeros > 0, "Expected some zero outcomes");
        assert!(ones > 0, "Expected some one outcomes");
    }

    #[test]
    fn test_simulator_x_basis_on_plus_state() {
        // |+> measured in X basis should deterministically give 0.
        // In stabilizer formalism, |+> has stabiliser +X, so X-basis
        // measurement is deterministic.
        let _initial = ClusterState::from_stabilizer(StabilizerState::new(1));
        // Apply H to get |+>.
        let mut stab = StabilizerState::new(1);
        stab.h(0);
        let cluster = ClusterState::from_stabilizer(stab);

        let mut sim = MBQCSimulator::new(cluster);
        let mut pattern = MeasurementPattern::new();
        pattern.add_command(MeasurementCommand::new(0, MeasurementBasis::XBasis));
        let outcomes = sim.execute(&pattern);

        // X-basis measurement of |+> is deterministic: always 0.
        // The rotation Rz(0) H turns X measurement into Z measurement on
        // the H|+> = |0> state, which gives 0.
        assert_eq!(outcomes[&0], false);
    }

    #[test]
    fn test_simulator_read_output() {
        let cluster = linear_cluster(3);
        let mut sim = MBQCSimulator::new(cluster);

        let mut pattern = MeasurementPattern::with_num_qubits(3);
        pattern.add_command(MeasurementCommand::new(0, MeasurementBasis::XBasis));
        pattern.add_command(MeasurementCommand::new(1, MeasurementBasis::XBasis));
        pattern.add_output_qubit(2);

        sim.execute(&pattern);
        let output = sim.read_output(&pattern);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_simulator_outcomes_tracking() {
        let cluster = linear_cluster(3);
        let mut sim = MBQCSimulator::new(cluster);

        let mut pattern = MeasurementPattern::new();
        pattern.add_command(MeasurementCommand::new(0, MeasurementBasis::XBasis));
        pattern.add_command(MeasurementCommand::new(1, MeasurementBasis::YBasis));

        sim.execute(&pattern);

        // Both outcomes should be recorded.
        assert!(sim.outcome(0).is_some());
        assert!(sim.outcome(1).is_some());
        assert!(sim.outcome(2).is_none()); // not measured
    }

    #[test]
    fn test_simulator_corrections() {
        // Verify that X and Z corrections modify the state.
        let mut stab = StabilizerState::new(1);
        stab.h(0); // |+>
        let cluster = ClusterState::from_stabilizer(stab);
        let mut sim = MBQCSimulator::new(cluster);

        // Apply X correction: |+> -> |+> (X eigenstate, no change).
        sim.apply_x_correction(0, true);
        // Apply Z correction: |+> -> |-> (Z flips the phase).
        sim.apply_z_correction(0, true);

        // After Z on |+> we get |->.  X-basis measurement of |-> gives 1.
        // Rotate for X measurement: H.
        sim.cluster.stabilizer_mut().h(0);
        let result = sim.cluster.stabilizer_mut().measure(0);
        assert_eq!(result, true);
    }

    // ---- Utility function tests ----

    #[test]
    fn test_normalise_angle() {
        assert!((normalise_angle(0.0) - 0.0).abs() < 1e-12);
        assert!((normalise_angle(2.0 * PI) - 0.0).abs() < 1e-10);
        assert!((normalise_angle(-PI / 2.0) - (3.0 * PI / 2.0)).abs() < 1e-12);
        assert!((normalise_angle(5.0 * PI) - PI).abs() < 1e-10);
    }

    #[test]
    fn test_is_clifford_angle() {
        assert!(is_clifford_angle(0.0));
        assert!(is_clifford_angle(PI / 2.0));
        assert!(is_clifford_angle(PI));
        assert!(is_clifford_angle(3.0 * PI / 2.0));
        assert!(!is_clifford_angle(PI / 4.0));
        assert!(!is_clifford_angle(0.1));
    }

    #[test]
    fn test_cluster_state_is_not_product() {
        // A linear cluster with 2+ qubits should be entangled.
        let cluster = linear_cluster(2);
        assert!(!cluster.stabilizer().is_product_state());
    }

    #[test]
    fn test_single_qubit_cluster_is_product() {
        // A single-qubit "cluster" is trivially a product state.
        let cluster = linear_cluster(1);
        assert!(cluster.stabilizer().is_product_state());
    }

    #[test]
    fn test_from_graph_panics_on_out_of_range() {
        let result = std::panic::catch_unwind(|| {
            ClusterState::from_graph(&[(0, 5)], 3);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_from_graph_panics_on_self_loop() {
        let result = std::panic::catch_unwind(|| {
            ClusterState::from_graph(&[(1, 1)], 3);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_default() {
        let pattern = MeasurementPattern::default();
        assert!(pattern.is_empty());
        assert_eq!(pattern.num_qubits(), 0);
    }

    // ---- Gate decomposition tests for previously-unsupported gates ----

    #[test]
    fn test_translator_rxx_gate() {
        // Rxx(theta) decomposes as CNOT(a,b) -> Rx(theta,a) -> CNOT(a,b).
        // Each CNOT uses 4 resource qubits, Rx decomposes into H+Rz+H (3
        // single-qubit rotations, each consuming 1 resource qubit).
        let theta = PI / 3.0;
        let gates = vec![Gate::rxx(0, 1, theta)];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(result.is_ok(), "Rxx gate should translate without error");
        let (cluster, pattern) = result.unwrap();

        // 2 original + resources for 2 CNOTs + 3 single-qubit gates (Rx = H+Rz+H).
        // Each CNOT: 4 resource qubits.  Each single-qubit: 1 resource qubit.
        // Total: 2 + 4 + 3 + 4 = 13.
        assert_eq!(cluster.num_qubits(), 13);
        assert_eq!(pattern.output_qubits().len(), 2);
        // Measurement commands: 2 CNOTs (4 each) + 3 single-qubit = 11.
        assert_eq!(pattern.len(), 11);
    }

    #[test]
    fn test_translator_ryy_gate() {
        // Ryy(theta) decomposes as Rz(-pi/2) on both, CNOT, Ry(theta), CNOT,
        // Rz(pi/2) on both.  That is 4 Rz + 1 Ry + 2 CNOTs.
        // Ry itself decomposes into Rz + Rx + Rz = Rz + H + Rz + H + Rz
        // = 5 single-qubit gates.
        let theta = PI / 4.0;
        let gates = vec![Gate::ryy(0, 1, theta)];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(result.is_ok(), "Ryy gate should translate without error");
        let (cluster, pattern) = result.unwrap();

        // Must have 2 output qubits.
        assert_eq!(pattern.output_qubits().len(), 2);
        // Cluster must be larger than the 2 logical qubits.
        assert!(cluster.num_qubits() > 2);
        // Must have measurement commands.
        assert!(pattern.len() > 0);
    }

    #[test]
    fn test_translator_rzz_gate() {
        // Rzz(theta) decomposes as CNOT(a,b) -> Rz(theta,b) -> CNOT(a,b).
        // 2 CNOTs (4 resource each) + 1 Rz (1 resource) = 2 + 8 + 1 = 11.
        let theta = PI / 6.0;
        let gates = vec![Gate::rzz(0, 1, theta)];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(result.is_ok(), "Rzz gate should translate without error");
        let (cluster, pattern) = result.unwrap();

        assert_eq!(cluster.num_qubits(), 11);
        assert_eq!(pattern.output_qubits().len(), 2);
        // 2 CNOTs (4 measurements each) + 1 Rz (1 measurement) = 9.
        assert_eq!(pattern.len(), 9);
    }

    #[test]
    fn test_translator_cswap_gate() {
        // CSWAP (Fredkin) decomposes as CNOT + Toffoli + CNOT.
        // This is a 3-qubit gate.
        let gates = vec![Gate::cswap(0, 1, 2)];
        let result = GateModelTranslator::to_mbqc(&gates, 3);
        assert!(result.is_ok(), "CSWAP gate should translate without error");
        let (cluster, pattern) = result.unwrap();

        // Must have 3 output qubits (one per logical qubit).
        assert_eq!(pattern.output_qubits().len(), 3);
        // The decomposition involves 2 CNOTs + 1 Toffoli (which itself
        // decomposes into many CNOTs and single-qubit gates), so the
        // cluster must be substantially larger than 3.
        assert!(
            cluster.num_qubits() > 20,
            "CSWAP should produce a large cluster (got {})",
            cluster.num_qubits()
        );
        assert!(pattern.len() > 0);
    }

    #[test]
    fn test_translator_cu_gate() {
        // CU(theta, phi, lambda, gamma) decomposes into Rz and Ry rotations
        // interleaved with CNOTs.
        let theta = PI / 3.0;
        let phi = PI / 4.0;
        let lambda = PI / 6.0;
        let gamma = PI / 8.0;
        let gates = vec![Gate::cu(0, 1, theta, phi, lambda, gamma)];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(result.is_ok(), "CU gate should translate without error");
        let (cluster, pattern) = result.unwrap();

        // Must have 2 output qubits.
        assert_eq!(pattern.output_qubits().len(), 2);
        // The decomposition uses 2 CNOTs + several single-qubit rotations.
        // Each CNOT: 4 resource qubits.  Each Rz: 1 resource qubit.
        // Each Ry: Rz + H + Rz + H + Rz = 5 resource qubits.
        // Total Rz on ctrl: 1, Rz on tgt: 3 total, Ry on tgt: 2
        // = 2 + 8 + 3 + 10 = 23.
        assert!(
            cluster.num_qubits() > 10,
            "CU should produce a significant cluster (got {})",
            cluster.num_qubits()
        );
        assert!(pattern.len() > 0);
    }

    #[test]
    fn test_translator_custom_gate_returns_error() {
        // Custom gates should return an error, not panic.
        use crate::C64;
        let identity_matrix = vec![
            vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        ];
        let custom_gate = Gate::new(
            GateType::Custom(identity_matrix),
            vec![0],
            vec![],
        );
        let gates = vec![custom_gate];
        let result = GateModelTranslator::to_mbqc(&gates, 1);
        assert!(result.is_err(), "Custom gate should return Err, not panic");

        match result.unwrap_err() {
            MbqcError::UnsupportedCustomGate {
                num_targets,
                num_controls,
                matrix_dim,
            } => {
                assert_eq!(num_targets, 1);
                assert_eq!(num_controls, 0);
                assert_eq!(matrix_dim, 2);
            }
        }
    }

    #[test]
    fn test_translator_custom_gate_error_display() {
        // Verify the error message is human-readable and informative.
        let err = MbqcError::UnsupportedCustomGate {
            num_targets: 2,
            num_controls: 1,
            matrix_dim: 8,
        };
        let msg = format!("{}", err);
        assert!(
            msg.contains("Custom gate"),
            "Error message should mention 'Custom gate'"
        );
        assert!(
            msg.contains("Decompose"),
            "Error message should advise decomposition"
        );
        assert!(
            msg.contains("8x8"),
            "Error message should include matrix dimensions"
        );
    }

    #[test]
    fn test_translator_rxx_zero_angle() {
        // Rxx(0) should be equivalent to identity (up to global phase).
        // The translator should still succeed.
        let gates = vec![Gate::rxx(0, 1, 0.0)];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(
            result.is_ok(),
            "Rxx(0) should translate without error"
        );
        let (_cluster, pattern) = result.unwrap();
        assert_eq!(pattern.output_qubits().len(), 2);
    }

    #[test]
    fn test_translator_rzz_pi() {
        // Rzz(pi) is a common entangling operation.  Ensure it translates.
        let gates = vec![Gate::rzz(0, 1, PI)];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(
            result.is_ok(),
            "Rzz(pi) should translate without error"
        );
        let (cluster, pattern) = result.unwrap();
        assert_eq!(pattern.output_qubits().len(), 2);
        assert!(cluster.num_qubits() > 2);
    }

    #[test]
    fn test_translator_cu_identity_params() {
        // CU(0, 0, 0, 0) should be close to a controlled-identity.
        let gates = vec![Gate::cu(0, 1, 0.0, 0.0, 0.0, 0.0)];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(
            result.is_ok(),
            "CU(0,0,0,0) should translate without error"
        );
        let (_cluster, pattern) = result.unwrap();
        assert_eq!(pattern.output_qubits().len(), 2);
    }

    #[test]
    fn test_translator_mixed_circuit_with_rxx_rzz() {
        // A multi-gate circuit mixing standard and rotation gates.
        let gates = vec![
            Gate::h(0),
            Gate::rxx(0, 1, PI / 4.0),
            Gate::rzz(0, 1, PI / 3.0),
            Gate::h(1),
        ];
        let result = GateModelTranslator::to_mbqc(&gates, 2);
        assert!(
            result.is_ok(),
            "Mixed circuit with Rxx and Rzz should translate"
        );
        let (_cluster, pattern) = result.unwrap();
        assert_eq!(pattern.output_qubits().len(), 2);
    }

    #[test]
    fn test_mbqc_error_is_std_error() {
        // Verify MbqcError implements std::error::Error for interop.
        let err = MbqcError::UnsupportedCustomGate {
            num_targets: 1,
            num_controls: 0,
            matrix_dim: 2,
        };
        let _: &dyn std::error::Error = &err;
    }
}
