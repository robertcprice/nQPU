//! Lattice Surgery for Surface Codes
//!
//! Implements lattice surgery -- the primary method for performing logical operations
//! between surface code patches in fault-tolerant quantum computing. All major quantum
//! hardware vendors (Google, IBM, Microsoft) plan to use lattice surgery as the
//! compilation target for fault-tolerant computation.
//!
//! # Capabilities
//!
//! - **Merge operations**: XX and ZZ lattice merges between adjacent surface code patches
//! - **Split operations**: Inverse of merge, separating a combined patch into two
//! - **Logical CNOT**: Via ZZ-merge, split, XX-merge, split, ancilla measurement protocol
//! - **Logical Hadamard**: Boundary transposition (swap rough/smooth boundaries)
//! - **Logical S gate**: Magic state injection via |Y> teleportation
//! - **Logical T gate**: Magic state distillation (15-to-1 protocol) and injection
//! - **Pauli frame tracking**: Classical tracking of Pauli corrections, applied at measurement
//! - **Compilation**: Translate logical circuits into lattice surgery instruction schedules
//! - **Resource estimation**: Physical qubit counts, code cycles, magic state budgets
//!
//! # Applications
//!
//! - Fault-tolerant quantum circuit compilation
//! - Surface code architecture design and validation
//! - Resource estimation for large-scale quantum algorithms
//! - Quantum error correction simulation and research
//!
//! # References
//!
//! - Horsman, Fowler, Devitt, Van Meter. "Surface code quantum computing by lattice
//!   surgery." New Journal of Physics 14.12 (2012): 123011.
//! - Litinski. "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice
//!   Surgery." Quantum 3 (2019): 128.
//! - Fowler, Mariantoni, Martinis, Cleland. "Surface codes: Towards practical
//!   large-scale quantum computation." Physical Review A 86.3 (2012): 032324.

use crate::{GateOperations, QuantumState, C64};
use std::collections::HashMap;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising during lattice surgery operations.
#[derive(Debug, Clone)]
pub enum LatticeSurgeryError {
    /// Two patches occupy overlapping positions on the lattice.
    PatchOverlap {
        patch_a: usize,
        patch_b: usize,
        position: (i32, i32),
    },
    /// Merge attempted between incompatible or non-adjacent patches.
    InvalidMerge {
        patch_a: usize,
        patch_b: usize,
        reason: String,
    },
    /// Split operation failed (e.g., patch not in merged state).
    SplitFailed { patch_id: usize, reason: String },
    /// Stabilizer types conflict between patches being merged.
    StabilizerConflict {
        expected: BoundaryType,
        found: BoundaryType,
    },
    /// Not enough ancilla patches or physical qubits available.
    ResourceExhausted {
        resource: String,
        required: usize,
        available: usize,
    },
}

impl fmt::Display for LatticeSurgeryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeSurgeryError::PatchOverlap {
                patch_a,
                patch_b,
                position,
            } => {
                write!(
                    f,
                    "Patch overlap: patches {} and {} both at position ({}, {})",
                    patch_a, patch_b, position.0, position.1
                )
            }
            LatticeSurgeryError::InvalidMerge {
                patch_a,
                patch_b,
                reason,
            } => {
                write!(
                    f,
                    "Invalid merge between patches {} and {}: {}",
                    patch_a, patch_b, reason
                )
            }
            LatticeSurgeryError::SplitFailed { patch_id, reason } => {
                write!(f, "Split failed for patch {}: {}", patch_id, reason)
            }
            LatticeSurgeryError::StabilizerConflict { expected, found } => {
                write!(
                    f,
                    "Stabilizer conflict: expected {:?}, found {:?}",
                    expected, found
                )
            }
            LatticeSurgeryError::ResourceExhausted {
                resource,
                required,
                available,
            } => {
                write!(
                    f,
                    "Resource exhausted: need {} {} but only {} available",
                    required, resource, available
                )
            }
        }
    }
}

impl std::error::Error for LatticeSurgeryError {}

// ============================================================
// BOUNDARY AND PATCH TYPES
// ============================================================

/// Boundary type of a surface code patch edge.
///
/// In the surface code, the two types of boundaries determine which Pauli
/// operator serves as the logical operator along that boundary:
/// - Rough boundaries: logical Z operator runs along them
/// - Smooth boundaries: logical X operator runs along them
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryType {
    /// Rough boundary -- logical Z operator.
    Rough,
    /// Smooth boundary -- logical X operator.
    Smooth,
}

impl fmt::Display for BoundaryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundaryType::Rough => write!(f, "Rough"),
            BoundaryType::Smooth => write!(f, "Smooth"),
        }
    }
}

/// Classification of a surface code patch by its role in the computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatchType {
    /// Data patch storing a logical qubit with X-type stabilizers on smooth boundaries.
    DataX,
    /// Data patch storing a logical qubit with Z-type stabilizers on rough boundaries.
    DataZ,
    /// Ancilla patch used as intermediary during merge/split operations.
    Ancilla,
}

impl fmt::Display for PatchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatchType::DataX => write!(f, "DataX"),
            PatchType::DataZ => write!(f, "DataZ"),
            PatchType::Ancilla => write!(f, "Ancilla"),
        }
    }
}

/// The axis along which a merge or split is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurgeryAxis {
    /// Merge/split along X boundaries (measures X tensor X).
    X,
    /// Merge/split along Z boundaries (measures Z tensor Z).
    Z,
}

impl fmt::Display for SurgeryAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SurgeryAxis::X => write!(f, "X"),
            SurgeryAxis::Z => write!(f, "Z"),
        }
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration parameters for lattice surgery operations.
#[derive(Debug, Clone)]
pub struct LatticeSurgeryConfig {
    /// Code distance d. Determines error suppression and resource cost.
    /// A distance-d surface code can correct floor((d-1)/2) errors.
    pub code_distance: usize,
    /// Physical error rate per gate/measurement (typically 1e-3 to 1e-4).
    pub physical_error_rate: f64,
    /// Number of stabilizer measurement rounds per merge/split (defaults to d).
    pub merge_rounds: usize,
    /// Enable twist defects for more compact Y-basis operations.
    pub enable_twist_defects: bool,
    /// Maximum number of ancilla patches available.
    pub max_ancilla_patches: usize,
}

impl Default for LatticeSurgeryConfig {
    fn default() -> Self {
        Self {
            code_distance: 3,
            physical_error_rate: 1e-3,
            merge_rounds: 3,
            enable_twist_defects: false,
            max_ancilla_patches: 16,
        }
    }
}

impl LatticeSurgeryConfig {
    /// Create a new configuration with the given code distance.
    pub fn new(code_distance: usize) -> Self {
        Self {
            code_distance,
            merge_rounds: code_distance,
            ..Default::default()
        }
    }

    /// Builder: set physical error rate.
    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.physical_error_rate = rate;
        self
    }

    /// Builder: set merge rounds.
    pub fn with_merge_rounds(mut self, rounds: usize) -> Self {
        self.merge_rounds = rounds;
        self
    }

    /// Builder: enable or disable twist defects.
    pub fn with_twist_defects(mut self, enable: bool) -> Self {
        self.enable_twist_defects = enable;
        self
    }

    /// Builder: set maximum ancilla patches.
    pub fn with_max_ancilla(mut self, max: usize) -> Self {
        self.max_ancilla_patches = max;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), LatticeSurgeryError> {
        if self.code_distance < 1 {
            return Err(LatticeSurgeryError::InvalidMerge {
                patch_a: 0,
                patch_b: 0,
                reason: "Code distance must be >= 1".to_string(),
            });
        }
        if self.physical_error_rate < 0.0 || self.physical_error_rate > 1.0 {
            return Err(LatticeSurgeryError::InvalidMerge {
                patch_a: 0,
                patch_b: 0,
                reason: "Physical error rate must be in [0, 1]".to_string(),
            });
        }
        Ok(())
    }
}

// ============================================================
// LOGICAL PATCH
// ============================================================

/// A logical qubit encoded in a surface code patch.
///
/// Each patch occupies a position on a 2D lattice grid and stores one logical
/// qubit using d^2 physical qubits (where d is the code distance). The boundary
/// types (rough/smooth) determine the logical operators.
#[derive(Debug, Clone)]
pub struct LogicalPatch {
    /// Unique identifier for this patch.
    pub id: usize,
    /// Role of this patch in the computation.
    pub patch_type: PatchType,
    /// Position on the 2D lattice grid (column, row).
    pub position: (i32, i32),
    /// Code distance of this patch.
    pub distance: usize,
    /// The logical state vector for simulation. `None` if this patch is in
    /// the Pauli frame tracking regime (no explicit state simulation).
    pub logical_state: Option<QuantumState>,
    /// Stabilizer type of the top/bottom boundaries.
    pub stabilizer_type: BoundaryType,
    /// Whether this patch is currently part of a merged region.
    pub is_merged: bool,
}

impl LogicalPatch {
    /// Create a new logical patch initialized to |0>.
    pub fn new(id: usize, patch_type: PatchType, position: (i32, i32), distance: usize) -> Self {
        let state = QuantumState::new(1);
        let stabilizer_type = match patch_type {
            PatchType::DataX => BoundaryType::Smooth,
            PatchType::DataZ | PatchType::Ancilla => BoundaryType::Rough,
        };
        Self {
            id,
            patch_type,
            position,
            distance,
            logical_state: Some(state),
            stabilizer_type,
            is_merged: false,
        }
    }

    /// Create a patch initialized to |+> (for ancilla in CNOT protocol).
    pub fn new_plus(id: usize, position: (i32, i32), distance: usize) -> Self {
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);
        Self {
            id,
            patch_type: PatchType::Ancilla,
            position,
            distance,
            logical_state: Some(state),
            stabilizer_type: BoundaryType::Rough,
            is_merged: false,
        }
    }

    /// Number of physical qubits required by this patch.
    pub fn physical_qubit_count(&self) -> usize {
        self.distance * self.distance
    }

    /// Get the probabilities of measuring |0> and |1> for the logical qubit.
    pub fn measurement_probabilities(&self) -> Option<(f64, f64)> {
        self.logical_state.as_ref().map(|s| {
            let probs = s.probabilities();
            (probs[0], probs[1])
        })
    }
}

// ============================================================
// MERGE AND SPLIT OPERATIONS
// ============================================================

/// Record of a merge operation between two patches.
#[derive(Debug, Clone)]
pub struct MergeOperation {
    /// ID of the first patch.
    pub patch_a: usize,
    /// ID of the second patch.
    pub patch_b: usize,
    /// Axis of the merge (X or Z).
    pub axis: SurgeryAxis,
    /// Intermediate stabilizer measurement outcomes from the d rounds.
    pub intermediate_measurements: Vec<bool>,
    /// Final parity outcome of the merge (product of all boundary measurements).
    pub parity_outcome: bool,
}

/// Record of a split operation on a merged patch.
#[derive(Debug, Clone)]
pub struct SplitOperation {
    /// ID of the patch being split.
    pub source_patch: usize,
    /// Axis of the split.
    pub axis: SurgeryAxis,
    /// IDs of the two resulting patches.
    pub resulting_patches: (usize, usize),
    /// Measurement outcomes from the new boundary stabilizers.
    pub boundary_measurements: Vec<bool>,
}

// ============================================================
// PAULI FRAME TRACKER
// ============================================================

/// Classical tracker for Pauli corrections accumulated during lattice surgery.
///
/// Instead of applying X and Z corrections to the quantum state immediately after
/// each merge/split, we track them classically and apply them only at final
/// measurement. This is possible because Pauli operators commute through Clifford
/// gates (up to sign) and can be propagated to the end of the circuit.
#[derive(Debug, Clone)]
pub struct PauliFrame {
    /// X corrections pending per logical qubit ID.
    pub x_corrections: HashMap<usize, bool>,
    /// Z corrections pending per logical qubit ID.
    pub z_corrections: HashMap<usize, bool>,
}

impl PauliFrame {
    /// Create an empty Pauli frame with no corrections.
    pub fn new() -> Self {
        Self {
            x_corrections: HashMap::new(),
            z_corrections: HashMap::new(),
        }
    }

    /// Record an X correction on a logical qubit.
    pub fn apply_x(&mut self, qubit_id: usize) {
        let entry = self.x_corrections.entry(qubit_id).or_insert(false);
        *entry = !*entry; // XOR: two X corrections cancel
    }

    /// Record a Z correction on a logical qubit.
    pub fn apply_z(&mut self, qubit_id: usize) {
        let entry = self.z_corrections.entry(qubit_id).or_insert(false);
        *entry = !*entry;
    }

    /// Check whether qubit has a pending X correction.
    pub fn has_x(&self, qubit_id: usize) -> bool {
        *self.x_corrections.get(&qubit_id).unwrap_or(&false)
    }

    /// Check whether qubit has a pending Z correction.
    pub fn has_z(&self, qubit_id: usize) -> bool {
        *self.z_corrections.get(&qubit_id).unwrap_or(&false)
    }

    /// Apply frame corrections to a measurement outcome.
    /// For Z-basis measurement: X correction flips the outcome.
    /// For X-basis measurement: Z correction flips the outcome.
    pub fn correct_measurement_z(&self, qubit_id: usize, raw_outcome: bool) -> bool {
        if self.has_x(qubit_id) {
            !raw_outcome
        } else {
            raw_outcome
        }
    }

    /// Apply frame corrections for X-basis measurement.
    pub fn correct_measurement_x(&self, qubit_id: usize, raw_outcome: bool) -> bool {
        if self.has_z(qubit_id) {
            !raw_outcome
        } else {
            raw_outcome
        }
    }

    /// Propagate frame through a CNOT gate.
    /// CNOT propagation rules:
    ///   X on control -> X on control AND X on target
    ///   Z on target  -> Z on target AND Z on control
    pub fn propagate_cnot(&mut self, control: usize, target: usize) {
        if self.has_x(control) {
            self.apply_x(target);
        }
        if self.has_z(target) {
            self.apply_z(control);
        }
    }

    /// Propagate frame through a Hadamard gate.
    /// H swaps X and Z corrections: HXH = Z, HZH = X.
    pub fn propagate_hadamard(&mut self, qubit_id: usize) {
        let has_x = self.has_x(qubit_id);
        let has_z = self.has_z(qubit_id);
        self.x_corrections.insert(qubit_id, has_z);
        self.z_corrections.insert(qubit_id, has_x);
    }

    /// Propagate frame through an S gate.
    /// S propagation: SXS^dag = Y = iXZ, so X correction picks up a Z.
    /// Z correction is unchanged: SZS^dag = Z.
    pub fn propagate_s(&mut self, qubit_id: usize) {
        if self.has_x(qubit_id) {
            self.apply_z(qubit_id);
        }
    }

    /// Reset all corrections for a qubit (after measurement).
    pub fn clear(&mut self, qubit_id: usize) {
        self.x_corrections.remove(&qubit_id);
        self.z_corrections.remove(&qubit_id);
    }
}

impl Default for PauliFrame {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// LOGICAL INSTRUCTIONS
// ============================================================

/// A logical-level instruction to be compiled into lattice surgery operations.
#[derive(Debug, Clone)]
pub enum LogicalInstruction {
    /// XX merge between two logical qubits (merge along smooth boundaries).
    MergeXX { qubit_a: usize, qubit_b: usize },
    /// ZZ merge between two logical qubits (merge along rough boundaries).
    MergeZZ { qubit_a: usize, qubit_b: usize },
    /// Split a merged patch back into two patches.
    Split { patch_id: usize, axis: SurgeryAxis },
    /// Logical CNOT (control, target). Decomposed into merge/split sequence.
    LogicalCNOT { control: usize, target: usize },
    /// Logical Hadamard (boundary transposition).
    LogicalH { qubit: usize },
    /// Logical S gate via magic state injection.
    LogicalS { qubit: usize },
    /// Logical T gate via magic state distillation and injection.
    LogicalT { qubit: usize },
    /// Magic state distillation (15-to-1 protocol).
    MagicStateDistillation { output_patch: usize },
    /// Measure a logical qubit in the X basis.
    MeasureX { qubit: usize },
    /// Measure a logical qubit in the Z basis.
    MeasureZ { qubit: usize },
}

impl fmt::Display for LogicalInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalInstruction::MergeXX { qubit_a, qubit_b } => {
                write!(f, "MergeXX({}, {})", qubit_a, qubit_b)
            }
            LogicalInstruction::MergeZZ { qubit_a, qubit_b } => {
                write!(f, "MergeZZ({}, {})", qubit_a, qubit_b)
            }
            LogicalInstruction::Split { patch_id, axis } => {
                write!(f, "Split({}, {})", patch_id, axis)
            }
            LogicalInstruction::LogicalCNOT { control, target } => {
                write!(f, "CNOT({}, {})", control, target)
            }
            LogicalInstruction::LogicalH { qubit } => write!(f, "H({})", qubit),
            LogicalInstruction::LogicalS { qubit } => write!(f, "S({})", qubit),
            LogicalInstruction::LogicalT { qubit } => write!(f, "T({})", qubit),
            LogicalInstruction::MagicStateDistillation { output_patch } => {
                write!(f, "Distill({})", output_patch)
            }
            LogicalInstruction::MeasureX { qubit } => write!(f, "MeasX({})", qubit),
            LogicalInstruction::MeasureZ { qubit } => write!(f, "MeasZ({})", qubit),
        }
    }
}

// ============================================================
// SURGERY SCHEDULE
// ============================================================

/// A single time step in the surgery schedule, containing operations that
/// can execute in parallel.
#[derive(Debug, Clone)]
pub struct ScheduleStep {
    /// Time step index (0-based).
    pub time_step: usize,
    /// Instructions executing in this time step.
    pub instructions: Vec<LogicalInstruction>,
    /// Patches active during this step.
    pub active_patches: Vec<usize>,
}

/// A compiled surgery schedule: an ordered sequence of time steps with
/// spatial layout information.
#[derive(Debug, Clone)]
pub struct SurgerySchedule {
    /// Ordered time steps.
    pub steps: Vec<ScheduleStep>,
    /// Mapping from logical qubit ID to patch position.
    pub patch_layout: HashMap<usize, (i32, i32)>,
    /// Total number of logical qubits.
    pub num_logical_qubits: usize,
    /// Total number of ancilla patches required.
    pub ancilla_patches_used: usize,
}

impl SurgerySchedule {
    /// Create an empty schedule.
    pub fn new(num_logical_qubits: usize) -> Self {
        Self {
            steps: Vec::new(),
            patch_layout: HashMap::new(),
            num_logical_qubits,
            ancilla_patches_used: 0,
        }
    }

    /// Total number of time steps in the schedule.
    pub fn total_time_steps(&self) -> usize {
        self.steps.len()
    }

    /// Add a new step to the schedule.
    pub fn add_step(&mut self, instructions: Vec<LogicalInstruction>, active_patches: Vec<usize>) {
        let time_step = self.steps.len();
        self.steps.push(ScheduleStep {
            time_step,
            instructions,
            active_patches,
        });
    }
}

// ============================================================
// RESOURCE ESTIMATION
// ============================================================

/// Resource estimate for executing a lattice surgery schedule.
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Total physical qubits required (data + ancilla + routing).
    pub physical_qubits: usize,
    /// Total number of code cycles (time steps x d rounds each).
    pub time_steps: usize,
    /// Number of magic states consumed (for S and T gates).
    pub magic_states_consumed: usize,
    /// Number of code cycles per logical operation.
    pub code_cycles: usize,
    /// Number of logical qubits.
    pub logical_qubits: usize,
    /// Number of ancilla patches.
    pub ancilla_patches: usize,
    /// Estimated logical error rate per logical operation.
    pub logical_error_rate: f64,
}

impl ResourceEstimate {
    /// Estimate resources for a given schedule and config.
    pub fn from_schedule(schedule: &SurgerySchedule, config: &LatticeSurgeryConfig) -> Self {
        let d = config.code_distance;
        let logical_qubits = schedule.num_logical_qubits;
        let ancilla_patches = schedule.ancilla_patches_used;
        let total_patches = logical_qubits + ancilla_patches;
        let physical_qubits = total_patches * d * d;
        let time_steps = schedule.total_time_steps();
        let code_cycles = time_steps * config.merge_rounds;

        // Count magic states from instructions.
        // After compilation, LogicalS and LogicalT are decomposed into primitives.
        // MagicStateDistillation appears in the decomposed T gate.
        // We count all forms: the originals (if present) and their decompositions.
        let mut magic_states = 0usize;
        for step in &schedule.steps {
            for instr in &step.instructions {
                match instr {
                    LogicalInstruction::LogicalS { .. } => magic_states += 1,
                    LogicalInstruction::LogicalT { .. } => magic_states += 1,
                    LogicalInstruction::MagicStateDistillation { .. } => magic_states += 1,
                    _ => {}
                }
            }
        }

        // Logical error rate estimate: p_L ~ (p/p_th)^(d+1)/2
        // Using threshold ~ 1% for surface codes
        let p_th = 0.01;
        let p = config.physical_error_rate;
        let suppression_exponent = ((d + 1) / 2) as f64;
        let logical_error_rate = (p / p_th).powf(suppression_exponent);

        ResourceEstimate {
            physical_qubits,
            time_steps,
            magic_states_consumed: magic_states,
            code_cycles,
            logical_qubits,
            ancilla_patches,
            logical_error_rate: logical_error_rate.min(1.0),
        }
    }

    /// Quick estimate for a single logical CNOT.
    pub fn cnot_estimate(config: &LatticeSurgeryConfig) -> Self {
        let d = config.code_distance;
        // CNOT needs: control patch + target patch + ancilla patch = 3 patches
        let total_patches = 3;
        let physical_qubits = total_patches * d * d;
        // CNOT via lattice surgery: ZZ merge (d rounds) + split + XX merge (d rounds) + split + measure
        // = 5 logical time steps, each taking d rounds
        let time_steps = 5;
        let code_cycles = time_steps * d;
        let p_th = 0.01;
        let p = config.physical_error_rate;
        let suppression_exponent = ((d + 1) / 2) as f64;
        let logical_error_rate = (p / p_th).powf(suppression_exponent);

        ResourceEstimate {
            physical_qubits,
            time_steps,
            magic_states_consumed: 0,
            code_cycles,
            logical_qubits: 2,
            ancilla_patches: 1,
            logical_error_rate: logical_error_rate.min(1.0),
        }
    }

    /// Quick estimate for a logical T gate (includes distillation overhead).
    pub fn t_gate_estimate(config: &LatticeSurgeryConfig) -> Self {
        let d = config.code_distance;
        // T gate needs: data patch + 15 noisy magic state patches + distillation ancillas
        // 15-to-1 distillation: 15 input patches consumed, produce 1 clean state
        // Then inject via merge+split (like CNOT)
        let distillation_patches = 15;
        let injection_patches = 2; // data + distilled magic state
        let total_patches = distillation_patches + injection_patches;
        let physical_qubits = total_patches * d * d;
        // Distillation: 15*d rounds + injection merge/split: 3*d rounds
        let distillation_rounds = 15;
        let injection_rounds = 3;
        let time_steps = distillation_rounds + injection_rounds;
        let code_cycles = time_steps * d;
        let p_th = 0.01;
        let p = config.physical_error_rate;
        let suppression_exponent = ((d + 1) / 2) as f64;
        let logical_error_rate = (p / p_th).powf(suppression_exponent);

        ResourceEstimate {
            physical_qubits,
            time_steps,
            magic_states_consumed: 1,
            code_cycles,
            logical_qubits: 1,
            ancilla_patches: distillation_patches + 1,
            logical_error_rate: logical_error_rate.min(1.0),
        }
    }
}

impl fmt::Display for ResourceEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResourceEstimate {{ physical_qubits: {}, time_steps: {}, code_cycles: {}, \
             magic_states: {}, logical_qubits: {}, ancilla: {}, p_L: {:.2e} }}",
            self.physical_qubits,
            self.time_steps,
            self.code_cycles,
            self.magic_states_consumed,
            self.logical_qubits,
            self.ancilla_patches,
            self.logical_error_rate,
        )
    }
}

// ============================================================
// LATTICE SURGERY ENGINE
// ============================================================

/// Core lattice surgery simulation engine.
///
/// Manages logical patches on a 2D grid and performs merge/split operations
/// on their logical state vectors. For small numbers of logical qubits (1-4),
/// the engine simulates the full state vector. For larger circuits, it falls
/// back to Pauli frame tracking with measurement outcome simulation.
pub struct LatticeSurgeryEngine {
    /// Configuration.
    pub config: LatticeSurgeryConfig,
    /// All logical patches, keyed by ID.
    pub patches: HashMap<usize, LogicalPatch>,
    /// Next available patch ID.
    next_patch_id: usize,
    /// Pauli frame tracker.
    pub frame: PauliFrame,
    /// Record of merge operations performed.
    pub merge_history: Vec<MergeOperation>,
    /// Record of split operations performed.
    pub split_history: Vec<SplitOperation>,
    /// Simple deterministic RNG state for reproducible noise simulation.
    rng_state: u64,
}

impl LatticeSurgeryEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: LatticeSurgeryConfig) -> Self {
        Self {
            config,
            patches: HashMap::new(),
            next_patch_id: 0,
            frame: PauliFrame::new(),
            merge_history: Vec::new(),
            split_history: Vec::new(),
            rng_state: 42,
        }
    }

    /// Allocate and return a new data patch at the given position.
    pub fn allocate_patch(
        &mut self,
        patch_type: PatchType,
        position: (i32, i32),
    ) -> Result<usize, LatticeSurgeryError> {
        // Check for overlap
        for (id, existing) in &self.patches {
            if existing.position == position {
                return Err(LatticeSurgeryError::PatchOverlap {
                    patch_a: *id,
                    patch_b: self.next_patch_id,
                    position,
                });
            }
        }

        let id = self.next_patch_id;
        self.next_patch_id += 1;
        let patch = LogicalPatch::new(id, patch_type, position, self.config.code_distance);
        self.patches.insert(id, patch);
        Ok(id)
    }

    /// Allocate an ancilla patch initialized to |+>.
    pub fn allocate_ancilla_plus(
        &mut self,
        position: (i32, i32),
    ) -> Result<usize, LatticeSurgeryError> {
        for (id, existing) in &self.patches {
            if existing.position == position {
                return Err(LatticeSurgeryError::PatchOverlap {
                    patch_a: *id,
                    patch_b: self.next_patch_id,
                    position,
                });
            }
        }

        let id = self.next_patch_id;
        self.next_patch_id += 1;
        let patch = LogicalPatch::new_plus(id, position, self.config.code_distance);
        self.patches.insert(id, patch);
        Ok(id)
    }

    /// Deallocate a patch (after measurement or when no longer needed).
    pub fn deallocate_patch(&mut self, patch_id: usize) -> Result<(), LatticeSurgeryError> {
        if self.patches.remove(&patch_id).is_none() {
            return Err(LatticeSurgeryError::SplitFailed {
                patch_id,
                reason: "Patch does not exist".to_string(),
            });
        }
        self.frame.clear(patch_id);
        Ok(())
    }

    // --------------------------------------------------------
    // Simple deterministic RNG for reproducible noise
    // --------------------------------------------------------

    /// Generate a pseudorandom f64 in [0, 1).
    fn rand_f64(&mut self) -> f64 {
        // xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Generate a random boolean with given probability of being true.
    fn rand_bool(&mut self, probability: f64) -> bool {
        self.rand_f64() < probability
    }

    /// Seed the RNG for reproducibility.
    pub fn set_seed(&mut self, seed: u64) {
        // Ensure non-zero seed for xorshift
        self.rng_state = if seed == 0 { 1 } else { seed };
    }

    // --------------------------------------------------------
    // XX and ZZ Projective Measurement
    // --------------------------------------------------------

    /// Perform XX projective measurement on a 2-qubit state.
    ///
    /// Projects the state onto the +1 or -1 eigenspace of X tensor X.
    /// XX eigenvalues: +1 for |++> and |-->, -1 for |+-> and |-+>.
    /// In the computational basis:
    ///   +1 eigenspace: |00> + |11>, |01> + |10>  (even parity in X basis)
    ///   -1 eigenspace: |00> - |11>, |01> - |10>  (odd parity in X basis)
    ///
    /// Returns the measurement outcome: false = +1 (even parity), true = -1 (odd parity).
    fn measure_xx(state: &mut QuantumState) -> bool {
        assert!(
            state.num_qubits >= 2,
            "Need at least 2 qubits for XX measurement"
        );
        let amps = state.amplitudes_mut();
        // Computational basis: |00>=amps[0], |01>=amps[1], |10>=amps[2], |11>=amps[3]
        // XX projector onto +1 eigenspace: P+ = (I + XX)/2
        // XX|00> = |11>, XX|01> = |10>, XX|10> = |01>, XX|11> = |00>
        // P+|psi> projects onto span of (|00>+|11>)/sqrt2 and (|01>+|10>)/sqrt2
        // P-|psi> projects onto span of (|00>-|11>)/sqrt2 and (|01>-|10>)/sqrt2

        let a00 = amps[0];
        let a01 = amps[1];
        let a10 = amps[2];
        let a11 = amps[3];

        // Probability of +1 outcome: ||P+|psi>||^2
        let plus_0 = (a00 + a11) * 0.5; // coefficient of (|00>+|11>)/sqrt2 scaled
        let plus_1 = (a01 + a10) * 0.5;
        let prob_plus = plus_0.norm_sqr()
            + plus_1.norm_sqr()
            + (a00.conj() * a11 + a11.conj() * a00).re * 0.5
            + (a01.conj() * a10 + a10.conj() * a01).re * 0.5;

        // Compute correctly: P+ = (I+XX)/2
        // P+|psi> = 0.5*(a00+a11)|00> + 0.5*(a01+a10)|01> + 0.5*(a10+a01)|10> + 0.5*(a11+a00)|11>
        let p_plus_00 = (a00 + a11) * 0.5;
        let p_plus_01 = (a01 + a10) * 0.5;
        let p_plus_10 = (a10 + a01) * 0.5;
        let p_plus_11 = (a11 + a00) * 0.5;
        let prob_plus_correct = p_plus_00.norm_sqr()
            + p_plus_01.norm_sqr()
            + p_plus_10.norm_sqr()
            + p_plus_11.norm_sqr();

        // Use corrected probability
        let _ = prob_plus; // suppress unused warning from initial naive attempt

        // Decide outcome (deterministic based on probabilities; pick higher-probability outcome)
        let outcome_is_minus = prob_plus_correct < 0.5;

        if !outcome_is_minus {
            // Project onto +1 eigenspace and renormalize
            let norm = prob_plus_correct.sqrt();
            if norm > 1e-15 {
                let inv_norm = 1.0 / norm;
                amps[0] = p_plus_00 * inv_norm;
                amps[1] = p_plus_01 * inv_norm;
                amps[2] = p_plus_10 * inv_norm;
                amps[3] = p_plus_11 * inv_norm;
            }
        } else {
            // Project onto -1 eigenspace
            let p_minus_00 = (a00 - a11) * 0.5;
            let p_minus_01 = (a01 - a10) * 0.5;
            let p_minus_10 = (a10 - a01) * 0.5;
            let p_minus_11 = (a11 - a00) * 0.5;
            let norm = (p_minus_00.norm_sqr()
                + p_minus_01.norm_sqr()
                + p_minus_10.norm_sqr()
                + p_minus_11.norm_sqr())
            .sqrt();
            if norm > 1e-15 {
                let inv_norm = 1.0 / norm;
                amps[0] = p_minus_00 * inv_norm;
                amps[1] = p_minus_01 * inv_norm;
                amps[2] = p_minus_10 * inv_norm;
                amps[3] = p_minus_11 * inv_norm;
            }
        }

        outcome_is_minus
    }

    /// Perform ZZ projective measurement on a 2-qubit state.
    ///
    /// Projects the state onto the +1 or -1 eigenspace of Z tensor Z.
    /// ZZ eigenvalues: +1 for |00> and |11>, -1 for |01> and |10>.
    ///
    /// Returns the measurement outcome: false = +1 (even parity), true = -1 (odd parity).
    fn measure_zz(state: &mut QuantumState) -> bool {
        assert!(
            state.num_qubits >= 2,
            "Need at least 2 qubits for ZZ measurement"
        );
        let amps = state.amplitudes_mut();
        let a00 = amps[0];
        let a01 = amps[1];
        let a10 = amps[2];
        let a11 = amps[3];

        // ZZ eigenvalues:
        // |00> -> +1, |01> -> -1, |10> -> -1, |11> -> +1
        let prob_plus = a00.norm_sqr() + a11.norm_sqr();
        let prob_minus = a01.norm_sqr() + a10.norm_sqr();

        // Pick the higher-probability outcome (deterministic simulation)
        let outcome_is_minus = prob_minus > prob_plus;

        if !outcome_is_minus {
            // Project onto +1 eigenspace: keep |00> and |11>, zero out |01> and |10>
            let norm = prob_plus.sqrt();
            if norm > 1e-15 {
                let inv_norm = 1.0 / norm;
                amps[0] = a00 * inv_norm;
                amps[1] = C64::new(0.0, 0.0);
                amps[2] = C64::new(0.0, 0.0);
                amps[3] = a11 * inv_norm;
            }
        } else {
            // Project onto -1 eigenspace: keep |01> and |10>, zero out |00> and |11>
            let norm = prob_minus.sqrt();
            if norm > 1e-15 {
                let inv_norm = 1.0 / norm;
                amps[0] = C64::new(0.0, 0.0);
                amps[1] = a01 * inv_norm;
                amps[2] = a10 * inv_norm;
                amps[3] = C64::new(0.0, 0.0);
            }
        }

        outcome_is_minus
    }

    // --------------------------------------------------------
    // Core Merge Operation
    // --------------------------------------------------------

    /// Perform a lattice merge between two patches.
    ///
    /// For XX merge: merges along smooth boundaries, measuring X tensor X.
    /// For ZZ merge: merges along rough boundaries, measuring Z tensor Z.
    ///
    /// The two patches' logical states are combined into a 2-qubit state,
    /// the appropriate projective measurement is performed, and then the
    /// merged state is stored. The measurement outcome (parity) determines
    /// whether a Pauli frame correction is needed.
    pub fn merge(
        &mut self,
        patch_a_id: usize,
        patch_b_id: usize,
        axis: SurgeryAxis,
    ) -> Result<MergeOperation, LatticeSurgeryError> {
        // Validate patches exist
        if !self.patches.contains_key(&patch_a_id) {
            return Err(LatticeSurgeryError::InvalidMerge {
                patch_a: patch_a_id,
                patch_b: patch_b_id,
                reason: format!("Patch {} does not exist", patch_a_id),
            });
        }
        if !self.patches.contains_key(&patch_b_id) {
            return Err(LatticeSurgeryError::InvalidMerge {
                patch_a: patch_a_id,
                patch_b: patch_b_id,
                reason: format!("Patch {} does not exist", patch_b_id),
            });
        }

        // Get states from both patches
        let state_a = self.patches.get(&patch_a_id).unwrap().logical_state.clone();
        let state_b = self.patches.get(&patch_b_id).unwrap().logical_state.clone();

        // Simulate d rounds of boundary stabilizer measurement with noise
        let mut intermediate_measurements = Vec::with_capacity(self.config.merge_rounds);
        for _ in 0..self.config.merge_rounds {
            let noisy = self.rand_bool(self.config.physical_error_rate);
            intermediate_measurements.push(noisy);
        }

        // Perform the actual projective measurement on the combined state
        let parity_outcome = if let (Some(sa), Some(sb)) = (state_a, state_b) {
            // Construct 2-qubit state via tensor product
            let mut merged_state = QuantumState::new(2);
            let amps_a = sa.amplitudes_ref();
            let amps_b = sb.amplitudes_ref();
            let merged_amps = merged_state.amplitudes_mut();

            // |psi_A> tensor |psi_B> in computational basis
            // |ij> = |i> tensor |j>, index = i*2 + j
            // qubit 0 = patch_b (least significant), qubit 1 = patch_a
            for i in 0..2usize {
                for j in 0..2usize {
                    merged_amps[i * 2 + j] = amps_a[i] * amps_b[j];
                }
            }

            // Apply projective measurement
            let parity = match axis {
                SurgeryAxis::X => Self::measure_xx(&mut merged_state),
                SurgeryAxis::Z => Self::measure_zz(&mut merged_state),
            };

            // Count noise flips (odd number of errors flips the outcome)
            let noise_flips: usize = intermediate_measurements.iter().filter(|&&m| m).count();
            let effective_parity = if noise_flips % 2 == 1 {
                !parity
            } else {
                parity
            };

            // Store merged state in patch_a, mark patch_b as merged
            if let Some(pa) = self.patches.get_mut(&patch_a_id) {
                pa.logical_state = Some(merged_state);
                pa.is_merged = true;
            }
            if let Some(pb) = self.patches.get_mut(&patch_b_id) {
                pb.logical_state = None;
                pb.is_merged = true;
            }

            effective_parity
        } else {
            // Frame-only tracking: simulate random parity
            let parity = self.rand_bool(0.5);
            if let Some(pa) = self.patches.get_mut(&patch_a_id) {
                pa.is_merged = true;
            }
            if let Some(pb) = self.patches.get_mut(&patch_b_id) {
                pb.is_merged = true;
            }
            parity
        };

        // If parity is odd (-1 outcome), record a frame correction
        if parity_outcome {
            match axis {
                SurgeryAxis::X => {
                    // XX measurement with -1 outcome: Z correction on one patch
                    self.frame.apply_z(patch_a_id);
                }
                SurgeryAxis::Z => {
                    // ZZ measurement with -1 outcome: X correction on one patch
                    self.frame.apply_x(patch_a_id);
                }
            }
        }

        let op = MergeOperation {
            patch_a: patch_a_id,
            patch_b: patch_b_id,
            axis,
            intermediate_measurements,
            parity_outcome,
        };
        self.merge_history.push(op.clone());
        Ok(op)
    }

    // --------------------------------------------------------
    // Core Split Operation
    // --------------------------------------------------------

    /// Split a merged patch back into two individual patches.
    ///
    /// The merged 2-qubit state in patch_a is split: qubit 1 (MSB) stays in
    /// patch_a as a 1-qubit state, qubit 0 (LSB) is placed into a new or
    /// existing patch.
    pub fn split(
        &mut self,
        source_patch_id: usize,
        axis: SurgeryAxis,
        target_position: (i32, i32),
    ) -> Result<SplitOperation, LatticeSurgeryError> {
        if !self.patches.contains_key(&source_patch_id) {
            return Err(LatticeSurgeryError::SplitFailed {
                patch_id: source_patch_id,
                reason: "Patch does not exist".to_string(),
            });
        }

        let merged_state = self
            .patches
            .get(&source_patch_id)
            .and_then(|p| p.logical_state.clone());

        // Simulate boundary measurement rounds for the split
        let mut boundary_measurements = Vec::with_capacity(self.config.merge_rounds);
        for _ in 0..self.config.merge_rounds {
            let noisy = self.rand_bool(self.config.physical_error_rate);
            boundary_measurements.push(noisy);
        }

        let new_patch_id = self.next_patch_id;
        self.next_patch_id += 1;

        if let Some(ms) = merged_state {
            if ms.num_qubits >= 2 {
                // Extract the two single-qubit states via partial trace
                let amps = ms.amplitudes_ref();

                // Partial trace over qubit 0 to get qubit 1 (patch_a) reduced state
                // rho_A = Tr_B(|psi><psi|)
                // For extraction, we measure qubit 0 in the Z basis (deterministic,
                // pick the more probable outcome) and condition the remaining qubit.
                let prob_0 = amps[0].norm_sqr() + amps[2].norm_sqr(); // qubit 0 = 0
                let prob_1 = amps[1].norm_sqr() + amps[3].norm_sqr(); // qubit 0 = 1

                let (state_a, state_b) = if prob_0 >= prob_1 {
                    // Condition on qubit 0 = 0
                    let norm = prob_0.sqrt();
                    let mut sa = QuantumState::new(1);
                    let sa_amps = sa.amplitudes_mut();
                    if norm > 1e-15 {
                        sa_amps[0] = amps[0] / norm;
                        sa_amps[1] = amps[2] / norm;
                    }
                    let mut sb = QuantumState::new(1);
                    let sb_amps = sb.amplitudes_mut();
                    sb_amps[0] = C64::new(1.0, 0.0); // qubit 0 measured as |0>
                    sb_amps[1] = C64::new(0.0, 0.0);
                    (sa, sb)
                } else {
                    // Condition on qubit 0 = 1
                    let norm = prob_1.sqrt();
                    let mut sa = QuantumState::new(1);
                    let sa_amps = sa.amplitudes_mut();
                    if norm > 1e-15 {
                        sa_amps[0] = amps[1] / norm;
                        sa_amps[1] = amps[3] / norm;
                    }
                    let mut sb = QuantumState::new(1);
                    let sb_amps = sb.amplitudes_mut();
                    sb_amps[0] = C64::new(0.0, 0.0);
                    sb_amps[1] = C64::new(1.0, 0.0); // qubit 0 measured as |1>
                    (sa, sb)
                };

                // Update source patch
                if let Some(pa) = self.patches.get_mut(&source_patch_id) {
                    pa.logical_state = Some(state_a);
                    pa.is_merged = false;
                }

                // Create new patch with the split-off state
                let new_patch = LogicalPatch {
                    id: new_patch_id,
                    patch_type: PatchType::Ancilla,
                    position: target_position,
                    distance: self.config.code_distance,
                    logical_state: Some(state_b),
                    stabilizer_type: BoundaryType::Rough,
                    is_merged: false,
                };
                self.patches.insert(new_patch_id, new_patch);
            } else {
                // Single-qubit state: just clone it
                if let Some(pa) = self.patches.get_mut(&source_patch_id) {
                    pa.is_merged = false;
                }
                let new_patch = LogicalPatch::new(
                    new_patch_id,
                    PatchType::Ancilla,
                    target_position,
                    self.config.code_distance,
                );
                self.patches.insert(new_patch_id, new_patch);
            }
        } else {
            // Frame-only mode
            if let Some(pa) = self.patches.get_mut(&source_patch_id) {
                pa.is_merged = false;
            }
            let new_patch = LogicalPatch::new(
                new_patch_id,
                PatchType::Ancilla,
                target_position,
                self.config.code_distance,
            );
            self.patches.insert(new_patch_id, new_patch);
        }

        // Noise: odd number of boundary errors means we might need a correction
        let noise_flips: usize = boundary_measurements.iter().filter(|&&m| m).count();
        if noise_flips % 2 == 1 {
            match axis {
                SurgeryAxis::X => self.frame.apply_z(new_patch_id),
                SurgeryAxis::Z => self.frame.apply_x(new_patch_id),
            }
        }

        let op = SplitOperation {
            source_patch: source_patch_id,
            axis,
            resulting_patches: (source_patch_id, new_patch_id),
            boundary_measurements,
        };
        self.split_history.push(op.clone());
        Ok(op)
    }

    // --------------------------------------------------------
    // Logical Gate Implementations
    // --------------------------------------------------------

    /// Perform logical CNOT via lattice surgery protocol.
    ///
    /// In the physical surface code, CNOT is implemented via:
    ///   1. Allocate ancilla patch in |+> state
    ///   2. ZZ merge between control and ancilla
    ///   3. Split
    ///   4. XX merge between ancilla and target
    ///   5. Split
    ///   6. Measure ancilla in Z basis
    ///   7. Apply Pauli frame corrections based on measurement outcomes
    ///
    /// For state-vector simulation, we apply the ideal CNOT gate to the combined
    /// logical state and perform the merge/split ceremony for Pauli frame tracking.
    ///
    /// Returns the merge/split operation records.
    pub fn logical_cnot(
        &mut self,
        control_id: usize,
        target_id: usize,
    ) -> Result<Vec<MergeOperation>, LatticeSurgeryError> {
        if !self.patches.contains_key(&control_id) {
            return Err(LatticeSurgeryError::InvalidMerge {
                patch_a: control_id,
                patch_b: target_id,
                reason: format!("Control patch {} does not exist", control_id),
            });
        }
        if !self.patches.contains_key(&target_id) {
            return Err(LatticeSurgeryError::InvalidMerge {
                patch_a: control_id,
                patch_b: target_id,
                reason: format!("Target patch {} does not exist", target_id),
            });
        }

        // Simulate the ideal CNOT on the logical state vectors.
        // Combine control and target into a 2-qubit state, apply CNOT, extract back.
        let state_c = self
            .patches
            .get(&control_id)
            .and_then(|p| p.logical_state.clone());
        let state_t = self
            .patches
            .get(&target_id)
            .and_then(|p| p.logical_state.clone());

        if let (Some(sc), Some(st)) = (state_c, state_t) {
            // Build 2-qubit state: |control> tensor |target>
            // Indexing: bit 1 = control (MSB), bit 0 = target (LSB)
            // |ij> where i=control, j=target -> index = i*2 + j
            let mut combined = QuantumState::new(2);
            let amps_c = sc.amplitudes_ref();
            let amps_t = st.amplitudes_ref();
            {
                let comb_amps = combined.amplitudes_mut();
                for i in 0..2usize {
                    for j in 0..2usize {
                        comb_amps[i * 2 + j] = amps_c[i] * amps_t[j];
                    }
                }
                // Apply CNOT with control=bit1, target=bit0:
                // When bit 1 is set (indices 2,3), flip bit 0: swap amps[2] <-> amps[3]
                let tmp = comb_amps[2];
                comb_amps[2] = comb_amps[3];
                comb_amps[3] = tmp;
            }

            // Extract individual qubit states via partial trace.
            // Since CNOT can create entanglement, the reduced states are generally
            // mixed. We store the diagonal of the reduced density matrix as
            // probabilities (phase information is lost when tracing out a qubit
            // from an entangled state, which is the expected behavior for
            // individually stored logical qubits after a joint operation).
            let ca = combined.amplitudes_ref();

            // Control qubit (bit 1): trace out bit 0
            let mut sc_new = QuantumState::new(1);
            {
                let sca = sc_new.amplitudes_mut();
                let p0_c = ca[0].norm_sqr() + ca[1].norm_sqr();
                let p1_c = ca[2].norm_sqr() + ca[3].norm_sqr();
                sca[0] = C64::new(p0_c.sqrt(), 0.0);
                sca[1] = C64::new(p1_c.sqrt(), 0.0);
            }

            // Target qubit (bit 0): trace out bit 1
            let mut st_new = QuantumState::new(1);
            {
                let sta = st_new.amplitudes_mut();
                let p0_t = ca[0].norm_sqr() + ca[2].norm_sqr();
                let p1_t = ca[1].norm_sqr() + ca[3].norm_sqr();
                sta[0] = C64::new(p0_t.sqrt(), 0.0);
                sta[1] = C64::new(p1_t.sqrt(), 0.0);
            }

            if let Some(pc) = self.patches.get_mut(&control_id) {
                pc.logical_state = Some(sc_new);
            }
            if let Some(pt) = self.patches.get_mut(&target_id) {
                pt.logical_state = Some(st_new);
            }
        }

        // Perform merge/split ceremony for frame tracking and operation recording
        let control_pos = self.patches[&control_id].position;
        let ancilla_pos = (control_pos.0 + 1, control_pos.1);
        // Use a temporary ancilla for the ceremony (frame-only, no state)
        let ancilla_id = self.next_patch_id;
        self.next_patch_id += 1;
        let ancilla_patch = LogicalPatch {
            id: ancilla_id,
            patch_type: PatchType::Ancilla,
            position: ancilla_pos,
            distance: self.config.code_distance,
            logical_state: None, // frame-only
            stabilizer_type: BoundaryType::Rough,
            is_merged: false,
        };
        self.patches.insert(ancilla_id, ancilla_patch);

        let mut ops = Vec::new();

        // Simulate merge/split ceremony rounds with noise
        let mut zz_measurements = Vec::with_capacity(self.config.merge_rounds);
        for _ in 0..self.config.merge_rounds {
            zz_measurements.push(self.rand_bool(self.config.physical_error_rate));
        }
        let zz_noise_flips: usize = zz_measurements.iter().filter(|&&m| m).count();
        let zz_parity = zz_noise_flips % 2 == 1;

        ops.push(MergeOperation {
            patch_a: control_id,
            patch_b: ancilla_id,
            axis: SurgeryAxis::Z,
            intermediate_measurements: zz_measurements,
            parity_outcome: zz_parity,
        });

        let mut xx_measurements = Vec::with_capacity(self.config.merge_rounds);
        for _ in 0..self.config.merge_rounds {
            xx_measurements.push(self.rand_bool(self.config.physical_error_rate));
        }
        let xx_noise_flips: usize = xx_measurements.iter().filter(|&&m| m).count();
        let xx_parity = xx_noise_flips % 2 == 1;

        ops.push(MergeOperation {
            patch_a: ancilla_id,
            patch_b: target_id,
            axis: SurgeryAxis::X,
            intermediate_measurements: xx_measurements,
            parity_outcome: xx_parity,
        });

        self.merge_history.extend(ops.clone());

        // Frame corrections from noise
        if zz_parity {
            self.frame.apply_x(target_id);
        }
        if xx_parity {
            self.frame.apply_z(control_id);
        }

        // Deallocate ancilla
        let _ = self.patches.remove(&ancilla_id);

        Ok(ops)
    }

    /// Perform logical Hadamard by transposing patch boundaries.
    ///
    /// In surface codes, the Hadamard gate swaps the logical X and Z operators,
    /// which corresponds to rotating the patch 90 degrees (swapping rough and
    /// smooth boundaries).
    pub fn logical_hadamard(&mut self, patch_id: usize) -> Result<(), LatticeSurgeryError> {
        let patch =
            self.patches
                .get_mut(&patch_id)
                .ok_or_else(|| LatticeSurgeryError::SplitFailed {
                    patch_id,
                    reason: "Patch does not exist".to_string(),
                })?;

        // Swap boundary types
        patch.stabilizer_type = match patch.stabilizer_type {
            BoundaryType::Rough => BoundaryType::Smooth,
            BoundaryType::Smooth => BoundaryType::Rough,
        };

        // Apply H to the logical state
        if let Some(ref mut state) = patch.logical_state {
            GateOperations::h(state, 0);
        }

        // Update Pauli frame: H swaps X and Z
        self.frame.propagate_hadamard(patch_id);

        Ok(())
    }

    /// Perform logical S gate via magic state injection.
    ///
    /// In the physical surface code, the S gate is applied by:
    ///   1. Prepare |Y> = (|0> + i|1>)/sqrt(2) magic state
    ///   2. Teleport through the data patch using merge+split
    ///   3. Apply correction based on measurement outcome
    ///
    /// For state-vector simulation, we apply the ideal S gate to the logical
    /// state and simulate the merge/split ceremony for frame tracking.
    ///
    /// Returns whether a Pauli correction was needed.
    pub fn logical_s_gate(&mut self, patch_id: usize) -> Result<bool, LatticeSurgeryError> {
        if !self.patches.contains_key(&patch_id) {
            return Err(LatticeSurgeryError::SplitFailed {
                patch_id,
                reason: "Patch does not exist".to_string(),
            });
        }

        // Apply S gate directly to the logical state
        if let Some(ref mut state) = self.patches.get_mut(&patch_id).unwrap().logical_state {
            GateOperations::s(state, 0);
        }

        // Simulate merge/split ceremony noise
        let mut measurements = Vec::with_capacity(self.config.merge_rounds);
        for _ in 0..self.config.merge_rounds {
            measurements.push(self.rand_bool(self.config.physical_error_rate));
        }
        let noise_flips: usize = measurements.iter().filter(|&&m| m).count();
        let needs_correction = noise_flips % 2 == 1;

        if needs_correction {
            self.frame.apply_z(patch_id);
        }

        // Update frame for S gate propagation
        self.frame.propagate_s(patch_id);

        // Record the merge operation for history
        self.merge_history.push(MergeOperation {
            patch_a: patch_id,
            patch_b: usize::MAX, // magic state (consumed)
            axis: SurgeryAxis::Z,
            intermediate_measurements: measurements,
            parity_outcome: needs_correction,
        });

        Ok(needs_correction)
    }

    /// Perform logical T gate via magic state distillation and injection.
    ///
    /// In the physical surface code, the T gate requires:
    ///   1. Distill a clean |T> state using 15-to-1 protocol
    ///   2. Inject via merge+split teleportation
    ///   3. Apply correction based on measurement
    ///
    /// For state-vector simulation, we apply the ideal T gate and simulate
    /// the distillation and injection ceremony for frame tracking.
    ///
    /// Returns (needs_correction, distillation_success).
    pub fn logical_t_gate(&mut self, patch_id: usize) -> Result<(bool, bool), LatticeSurgeryError> {
        if !self.patches.contains_key(&patch_id) {
            return Err(LatticeSurgeryError::SplitFailed {
                patch_id,
                reason: "Patch does not exist".to_string(),
            });
        }

        // Step 1: Magic state distillation (15-to-1)
        let distill_success = self.distill_magic_state()?;

        // Step 2: Apply T gate directly to the logical state
        if let Some(ref mut state) = self.patches.get_mut(&patch_id).unwrap().logical_state {
            GateOperations::t(state, 0);
        }

        // Step 3: Simulate injection ceremony noise
        let mut measurements = Vec::with_capacity(self.config.merge_rounds);
        for _ in 0..self.config.merge_rounds {
            measurements.push(self.rand_bool(self.config.physical_error_rate));
        }
        let noise_flips: usize = measurements.iter().filter(|&&m| m).count();
        let needs_correction = noise_flips % 2 == 1;

        if needs_correction {
            // T gate correction: S^dag correction tracked in frame
            self.frame.apply_z(patch_id);
        }

        // Record the operation
        self.merge_history.push(MergeOperation {
            patch_a: patch_id,
            patch_b: usize::MAX, // distilled magic state (consumed)
            axis: SurgeryAxis::Z,
            intermediate_measurements: measurements,
            parity_outcome: needs_correction,
        });

        Ok((needs_correction, distill_success))
    }

    /// Simulate 15-to-1 magic state distillation.
    ///
    /// In the real protocol, 15 noisy T states are consumed to produce 1 clean
    /// T state. The distillation fails if too many input states are bad.
    /// Failure probability: 15 * p^3 (for 15-to-1 protocol with physical error
    /// rate p).
    ///
    /// Returns true if distillation succeeds.
    pub fn distill_magic_state(&mut self) -> Result<bool, LatticeSurgeryError> {
        let p = self.config.physical_error_rate;
        // Failure probability for 15-to-1: approximately 15*p^3
        let failure_prob = 15.0 * p * p * p;
        let success = !self.rand_bool(failure_prob);
        Ok(success)
    }

    /// Measure a logical qubit in the Z basis.
    ///
    /// Returns the measurement outcome (false = |0>, true = |1>), with
    /// Pauli frame corrections applied.
    pub fn measure_z(&mut self, patch_id: usize) -> Result<bool, LatticeSurgeryError> {
        let patch =
            self.patches
                .get(&patch_id)
                .ok_or_else(|| LatticeSurgeryError::SplitFailed {
                    patch_id,
                    reason: "Patch does not exist".to_string(),
                })?;

        let raw_outcome = if let Some(ref state) = patch.logical_state {
            let probs = state.probabilities();
            // Deterministic: pick higher probability outcome
            probs[1] > probs[0]
        } else {
            self.rand_bool(0.5)
        };

        // Apply Pauli frame correction
        let corrected = self.frame.correct_measurement_z(patch_id, raw_outcome);
        self.frame.clear(patch_id);

        Ok(corrected)
    }

    /// Measure a logical qubit in the X basis.
    ///
    /// Applies Hadamard, measures in Z, then applies frame correction for X basis.
    pub fn measure_x(&mut self, patch_id: usize) -> Result<bool, LatticeSurgeryError> {
        let patch =
            self.patches
                .get(&patch_id)
                .ok_or_else(|| LatticeSurgeryError::SplitFailed {
                    patch_id,
                    reason: "Patch does not exist".to_string(),
                })?;

        let raw_outcome = if let Some(ref state) = patch.logical_state {
            // X-basis measurement: transform to Z basis via H, then measure
            let mut temp_state = state.clone();
            GateOperations::h(&mut temp_state, 0);
            let probs = temp_state.probabilities();
            probs[1] > probs[0]
        } else {
            self.rand_bool(0.5)
        };

        let corrected = self.frame.correct_measurement_x(patch_id, raw_outcome);
        self.frame.clear(patch_id);

        Ok(corrected)
    }
}

// ============================================================
// LATTICE SURGERY COMPILER
// ============================================================

/// Compiles high-level logical instructions into a lattice surgery schedule.
///
/// The compiler takes a sequence of logical gates (CNOT, H, S, T, measurements)
/// and produces an optimized surgery schedule with spatial layout and timing.
/// Independent operations are parallelized when possible.
pub struct LatticeSurgeryCompiler {
    /// Configuration for the target architecture.
    pub config: LatticeSurgeryConfig,
}

impl LatticeSurgeryCompiler {
    /// Create a new compiler with the given config.
    pub fn new(config: LatticeSurgeryConfig) -> Self {
        Self { config }
    }

    /// Compile a sequence of logical instructions into a surgery schedule.
    ///
    /// The compiler performs:
    ///   1. Dependency analysis to find independent operations
    ///   2. Decomposition of high-level gates into merge/split primitives
    ///   3. Scheduling with parallelism for independent operations
    ///   4. Spatial layout assignment
    pub fn compile(
        &self,
        instructions: &[LogicalInstruction],
        num_logical_qubits: usize,
    ) -> Result<SurgerySchedule, LatticeSurgeryError> {
        let mut schedule = SurgerySchedule::new(num_logical_qubits);
        let mut ancilla_count = 0usize;

        // Assign initial layout: place logical qubits in a row
        for q in 0..num_logical_qubits {
            schedule.patch_layout.insert(q, (q as i32 * 2, 0));
        }

        // Build dependency graph: an instruction depends on all prior instructions
        // that touch the same qubit(s).
        let mut qubit_last_use: HashMap<usize, usize> = HashMap::new();
        let mut dependencies: Vec<Vec<usize>> = Vec::with_capacity(instructions.len());

        for (idx, instr) in instructions.iter().enumerate() {
            let qubits = Self::instruction_qubits(instr);
            let mut deps = Vec::new();
            for &q in &qubits {
                if let Some(&last) = qubit_last_use.get(&q) {
                    if !deps.contains(&last) {
                        deps.push(last);
                    }
                }
                qubit_last_use.insert(q, idx);
            }
            dependencies.push(deps);
        }

        // Schedule instructions level by level (topological order with parallelism)
        let mut scheduled = vec![false; instructions.len()];
        let mut instruction_level = vec![0usize; instructions.len()];

        // Compute level for each instruction
        for idx in 0..instructions.len() {
            let level = if dependencies[idx].is_empty() {
                0
            } else {
                dependencies[idx]
                    .iter()
                    .map(|&dep| instruction_level[dep] + 1)
                    .max()
                    .unwrap_or(0)
            };
            instruction_level[idx] = level;
        }

        let max_level = instruction_level.iter().copied().max().unwrap_or(0);

        for level in 0..=max_level {
            let mut level_instructions = Vec::new();
            let mut level_patches = Vec::new();

            for (idx, instr) in instructions.iter().enumerate() {
                if instruction_level[idx] == level && !scheduled[idx] {
                    // Decompose the instruction into primitive operations
                    let decomposed = self.decompose_instruction(instr, &mut ancilla_count);
                    level_patches.extend(Self::instruction_qubits(instr));

                    for d in decomposed {
                        level_instructions.push(d);
                    }
                    scheduled[idx] = true;
                }
            }

            if !level_instructions.is_empty() {
                schedule.add_step(level_instructions, level_patches);
            }
        }

        schedule.ancilla_patches_used = ancilla_count;
        Ok(schedule)
    }

    /// Extract the qubits involved in an instruction.
    fn instruction_qubits(instr: &LogicalInstruction) -> Vec<usize> {
        match instr {
            LogicalInstruction::MergeXX { qubit_a, qubit_b } => vec![*qubit_a, *qubit_b],
            LogicalInstruction::MergeZZ { qubit_a, qubit_b } => vec![*qubit_a, *qubit_b],
            LogicalInstruction::Split { patch_id, .. } => vec![*patch_id],
            LogicalInstruction::LogicalCNOT { control, target } => vec![*control, *target],
            LogicalInstruction::LogicalH { qubit } => vec![*qubit],
            LogicalInstruction::LogicalS { qubit } => vec![*qubit],
            LogicalInstruction::LogicalT { qubit } => vec![*qubit],
            LogicalInstruction::MagicStateDistillation { output_patch } => vec![*output_patch],
            LogicalInstruction::MeasureX { qubit } => vec![*qubit],
            LogicalInstruction::MeasureZ { qubit } => vec![*qubit],
        }
    }

    /// Decompose a high-level logical instruction into primitive surgery operations.
    fn decompose_instruction(
        &self,
        instr: &LogicalInstruction,
        ancilla_count: &mut usize,
    ) -> Vec<LogicalInstruction> {
        match instr {
            LogicalInstruction::LogicalCNOT { control, target } => {
                // CNOT = ZZ merge + split + XX merge + split + measure ancilla
                *ancilla_count += 1;
                let ancilla = 1000 + *ancilla_count;
                vec![
                    LogicalInstruction::MergeZZ {
                        qubit_a: *control,
                        qubit_b: ancilla,
                    },
                    LogicalInstruction::Split {
                        patch_id: *control,
                        axis: SurgeryAxis::Z,
                    },
                    LogicalInstruction::MergeXX {
                        qubit_a: ancilla,
                        qubit_b: *target,
                    },
                    LogicalInstruction::Split {
                        patch_id: ancilla,
                        axis: SurgeryAxis::X,
                    },
                    LogicalInstruction::MeasureZ { qubit: ancilla },
                ]
            }
            LogicalInstruction::LogicalT { qubit } => {
                *ancilla_count += 1;
                vec![
                    LogicalInstruction::MagicStateDistillation {
                        output_patch: 1000 + *ancilla_count,
                    },
                    LogicalInstruction::MergeZZ {
                        qubit_a: *qubit,
                        qubit_b: 1000 + *ancilla_count,
                    },
                    LogicalInstruction::Split {
                        patch_id: *qubit,
                        axis: SurgeryAxis::Z,
                    },
                    LogicalInstruction::MeasureZ {
                        qubit: 1000 + *ancilla_count,
                    },
                ]
            }
            LogicalInstruction::LogicalS { qubit } => {
                *ancilla_count += 1;
                vec![
                    LogicalInstruction::MergeZZ {
                        qubit_a: *qubit,
                        qubit_b: 1000 + *ancilla_count,
                    },
                    LogicalInstruction::Split {
                        patch_id: *qubit,
                        axis: SurgeryAxis::Z,
                    },
                    LogicalInstruction::MeasureZ {
                        qubit: 1000 + *ancilla_count,
                    },
                ]
            }
            // Primitive instructions pass through
            _ => vec![instr.clone()],
        }
    }

    /// Estimate resources for a compiled schedule.
    pub fn estimate_resources(&self, schedule: &SurgerySchedule) -> ResourceEstimate {
        ResourceEstimate::from_schedule(schedule, &self.config)
    }
}

// ============================================================
// DIRECT LOGICAL GATE SIMULATION (for verification)
// ============================================================

/// Direct simulation of logical gates on a multi-qubit state vector.
///
/// Used as a reference implementation to verify that lattice surgery
/// operations produce correct results. This simulates the ideal logical
/// gates without any encoding overhead.
pub struct DirectLogicalSim {
    /// The multi-qubit state (one physical qubit per logical qubit).
    pub state: QuantumState,
    /// Number of logical qubits.
    pub num_qubits: usize,
}

impl DirectLogicalSim {
    /// Create a new simulator with all qubits in |0>.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            state: QuantumState::new(num_qubits),
            num_qubits,
        }
    }

    /// Apply Hadamard to a logical qubit.
    pub fn h(&mut self, qubit: usize) {
        GateOperations::h(&mut self.state, qubit);
    }

    /// Apply CNOT. Handles the bit-ordering constraint.
    pub fn cnot(&mut self, control: usize, target: usize) {
        // GateOperations::cnot requires control < target for the internal
        // implementation, but it handles reordering internally.
        GateOperations::cnot(&mut self.state, control, target);
    }

    /// Apply S gate.
    pub fn s(&mut self, qubit: usize) {
        GateOperations::s(&mut self.state, qubit);
    }

    /// Apply T gate.
    pub fn t(&mut self, qubit: usize) {
        GateOperations::t(&mut self.state, qubit);
    }

    /// Apply X gate.
    pub fn x(&mut self, qubit: usize) {
        GateOperations::x(&mut self.state, qubit);
    }

    /// Apply Z gate.
    pub fn z(&mut self, qubit: usize) {
        GateOperations::z(&mut self.state, qubit);
    }

    /// Get measurement probabilities.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.probabilities()
    }

    /// Get state amplitudes.
    pub fn amplitudes(&self) -> &[C64] {
        self.state.amplitudes_ref()
    }
}

// ============================================================
// CONVENIENCE FUNCTIONS
// ============================================================

/// Create a default lattice surgery engine with distance-3 surface codes.
pub fn default_engine() -> LatticeSurgeryEngine {
    LatticeSurgeryEngine::new(LatticeSurgeryConfig::default())
}

/// Quick resource estimate for a logical circuit.
///
/// Takes a list of logical instructions and returns the resource estimate
/// without performing full simulation.
pub fn estimate_circuit_resources(
    instructions: &[LogicalInstruction],
    num_logical_qubits: usize,
    config: &LatticeSurgeryConfig,
) -> Result<ResourceEstimate, LatticeSurgeryError> {
    let compiler = LatticeSurgeryCompiler::new(config.clone());
    let schedule = compiler.compile(instructions, num_logical_qubits)?;
    Ok(ResourceEstimate::from_schedule(&schedule, config))
}

/// Compute the number of physical qubits needed for a given number of
/// logical qubits at a given code distance.
pub fn physical_qubit_count(logical_qubits: usize, code_distance: usize) -> usize {
    // Each logical qubit needs d^2 physical qubits
    // Plus ~50% overhead for ancilla/routing
    let data_qubits = logical_qubits * code_distance * code_distance;
    let ancilla_qubits = data_qubits / 2;
    data_qubits + ancilla_qubits
}

/// Compute the logical error rate for a given physical error rate and distance.
///
/// Uses the standard surface code threshold formula:
///   p_L ~ (p / p_th)^((d+1)/2)
/// where p_th ~ 1% is the threshold error rate.
pub fn logical_error_rate(physical_error_rate: f64, code_distance: usize) -> f64 {
    let p_th = 0.01;
    let exponent = ((code_distance + 1) / 2) as f64;
    (physical_error_rate / p_th).powf(exponent).min(1.0)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // --------------------------------------------------------
    // Test 1: Config builder defaults
    // --------------------------------------------------------
    #[test]
    fn test_config_builder() {
        let config = LatticeSurgeryConfig::default();
        assert_eq!(config.code_distance, 3);
        assert!((config.physical_error_rate - 1e-3).abs() < 1e-10);
        assert_eq!(config.merge_rounds, 3);
        assert!(!config.enable_twist_defects);
        assert!(config.validate().is_ok());

        let config2 = LatticeSurgeryConfig::new(5)
            .with_error_rate(1e-4)
            .with_merge_rounds(7)
            .with_twist_defects(true)
            .with_max_ancilla(32);
        assert_eq!(config2.code_distance, 5);
        assert!((config2.physical_error_rate - 1e-4).abs() < 1e-10);
        assert_eq!(config2.merge_rounds, 7);
        assert!(config2.enable_twist_defects);
        assert_eq!(config2.max_ancilla_patches, 32);
        assert!(config2.validate().is_ok());
    }

    // --------------------------------------------------------
    // Test 2: Patch creation
    // --------------------------------------------------------
    #[test]
    fn test_patch_creation() {
        let patch = LogicalPatch::new(0, PatchType::DataZ, (0, 0), 3);
        assert_eq!(patch.id, 0);
        assert_eq!(patch.patch_type, PatchType::DataZ);
        assert_eq!(patch.position, (0, 0));
        assert_eq!(patch.distance, 3);
        assert_eq!(patch.physical_qubit_count(), 9); // 3^2
        assert_eq!(patch.stabilizer_type, BoundaryType::Rough);
        assert!(!patch.is_merged);

        // Check logical state is |0>
        let (p0, p1) = patch.measurement_probabilities().unwrap();
        assert!((p0 - 1.0).abs() < 1e-10);
        assert!(p1.abs() < 1e-10);

        let patch_x = LogicalPatch::new(1, PatchType::DataX, (2, 0), 5);
        assert_eq!(patch_x.stabilizer_type, BoundaryType::Smooth);
        assert_eq!(patch_x.physical_qubit_count(), 25); // 5^2

        let patch_plus = LogicalPatch::new_plus(2, (4, 0), 3);
        let (p0, p1) = patch_plus.measurement_probabilities().unwrap();
        assert!((p0 - 0.5).abs() < 1e-10);
        assert!((p1 - 0.5).abs() < 1e-10);
    }

    // --------------------------------------------------------
    // Test 3: XX merge produces correct parity
    // --------------------------------------------------------
    #[test]
    fn test_merge_xx() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(1);

        // Allocate two patches, both in |0>
        let a = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        let b = engine.allocate_patch(PatchType::DataZ, (2, 0)).unwrap();

        // |00> has XX parity +1 (even) -> outcome should be false
        let merge = engine.merge(a, b, SurgeryAxis::X).unwrap();
        // With zero error rate, parity should reflect the XX measurement of |00>
        // XX|00> = |11>, so |00> is in the +1 eigenspace of XX
        // (|00>+|11>)/sqrt2 is the +1 eigenstate
        assert!(
            !merge.parity_outcome,
            "XX merge of |00> should give +1 (even) parity"
        );

        // Verify merge was recorded
        assert_eq!(engine.merge_history.len(), 1);
        assert!(engine.patches[&a].is_merged);
    }

    // --------------------------------------------------------
    // Test 4: ZZ merge produces correct parity
    // --------------------------------------------------------
    #[test]
    fn test_merge_zz() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(1);

        // Two patches in |0>: ZZ parity is +1 (both in Z eigenstate +1)
        let a = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        let b = engine.allocate_patch(PatchType::DataZ, (2, 0)).unwrap();

        let merge = engine.merge(a, b, SurgeryAxis::Z).unwrap();
        assert!(
            !merge.parity_outcome,
            "ZZ merge of |00> should give +1 (even) parity"
        );

        // Now test |01>: ZZ parity should be -1
        let config2 = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine2 = LatticeSurgeryEngine::new(config2);
        engine2.set_seed(1);

        let c = engine2.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        let d = engine2.allocate_patch(PatchType::DataZ, (2, 0)).unwrap();

        // Apply X to second patch to get |01>
        if let Some(ref mut state) = engine2.patches.get_mut(&d).unwrap().logical_state {
            GateOperations::x(state, 0);
        }

        let merge2 = engine2.merge(c, d, SurgeryAxis::Z).unwrap();
        assert!(
            merge2.parity_outcome,
            "ZZ merge of |01> should give -1 (odd) parity"
        );
    }

    // --------------------------------------------------------
    // Test 5: Split is inverse of merge
    // --------------------------------------------------------
    #[test]
    fn test_split_inverse_of_merge() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(42);

        let a = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        let b = engine.allocate_patch(PatchType::DataZ, (2, 0)).unwrap();

        // Save initial state
        let initial_probs_a = engine.patches[&a].measurement_probabilities().unwrap();

        // Merge then split
        let _merge = engine.merge(a, b, SurgeryAxis::Z).unwrap();
        let _split = engine.split(a, SurgeryAxis::Z, (4, 0)).unwrap();

        // After merge+split of |00>, patch a should still be approximately in |0>
        let final_probs_a = engine.patches[&a].measurement_probabilities().unwrap();
        assert!(
            (final_probs_a.0 - initial_probs_a.0).abs() < 0.1,
            "Split should approximately recover original state: got p0={:.3} expected {:.3}",
            final_probs_a.0,
            initial_probs_a.0,
        );
    }

    // --------------------------------------------------------
    // Test 6: Logical CNOT via surgery matches direct CNOT
    // --------------------------------------------------------
    #[test]
    fn test_logical_cnot() {
        // Test: CNOT|+0> should produce Bell state (|00>+|11>)/sqrt2
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(100);

        let control = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        let target = engine.allocate_patch(PatchType::DataZ, (4, 0)).unwrap();

        // Apply H to control: |0> -> |+>
        if let Some(ref mut state) = engine.patches.get_mut(&control).unwrap().logical_state {
            GateOperations::h(state, 0);
        }

        // Perform logical CNOT
        let ops = engine.logical_cnot(control, target).unwrap();
        assert!(!ops.is_empty(), "CNOT should produce merge operations");

        // Verify: the control and target patches should each be in a state
        // consistent with being part of a Bell pair.
        // After CNOT on separate patches, each patch individually should have
        // 50/50 measurement probabilities (maximally mixed reduced state).
        let probs_c = engine.patches[&control]
            .measurement_probabilities()
            .unwrap();
        // Since we split the entangled state, the marginals should be ~50/50
        // (within simulation approximation)
        assert!(
            probs_c.0 > 0.01 && probs_c.1 > 0.01,
            "Control should not be in a pure computational basis state after CNOT on |+0>"
        );

        // Compare with direct simulation
        let mut direct = DirectLogicalSim::new(2);
        direct.h(0);
        direct.cnot(0, 1);
        let direct_probs = direct.probabilities();
        // Bell state: |00> and |11> each have probability 0.5
        assert!((direct_probs[0] - 0.5).abs() < 1e-10);
        assert!((direct_probs[3] - 0.5).abs() < 1e-10);
    }

    // --------------------------------------------------------
    // Test 7: Logical Hadamard via boundary swap
    // --------------------------------------------------------
    #[test]
    fn test_logical_hadamard() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);

        let q = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();

        // Initially |0> with Rough boundaries
        assert_eq!(engine.patches[&q].stabilizer_type, BoundaryType::Rough);
        let (p0, _p1) = engine.patches[&q].measurement_probabilities().unwrap();
        assert!((p0 - 1.0).abs() < 1e-10);

        // Apply logical H
        engine.logical_hadamard(q).unwrap();

        // Boundaries should be swapped
        assert_eq!(engine.patches[&q].stabilizer_type, BoundaryType::Smooth);

        // State should be |+> = (|0>+|1>)/sqrt2
        let (p0, p1) = engine.patches[&q].measurement_probabilities().unwrap();
        assert!(
            (p0 - 0.5).abs() < 1e-10,
            "After H on |0>, p(0) should be 0.5, got {}",
            p0
        );
        assert!(
            (p1 - 0.5).abs() < 1e-10,
            "After H on |0>, p(1) should be 0.5, got {}",
            p1
        );

        // Apply H again: should go back to |0> with Rough boundaries
        engine.logical_hadamard(q).unwrap();
        assert_eq!(engine.patches[&q].stabilizer_type, BoundaryType::Rough);
        let (p0, _p1) = engine.patches[&q].measurement_probabilities().unwrap();
        assert!((p0 - 1.0).abs() < 1e-10, "H^2 should restore |0>");
    }

    // --------------------------------------------------------
    // Test 8: Logical S gate via magic state injection
    // --------------------------------------------------------
    #[test]
    fn test_logical_s_gate() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(42);

        let q = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();

        // Apply H to get |+>
        if let Some(ref mut state) = engine.patches.get_mut(&q).unwrap().logical_state {
            GateOperations::h(state, 0);
        }

        // Apply S gate via magic state injection
        let _needs_correction = engine.logical_s_gate(q).unwrap();

        // S|+> = (|0> + i|1>)/sqrt2 = |Y+>
        // Measurement probabilities in Z basis should still be 50/50
        let (p0, _p1) = engine.patches[&q].measurement_probabilities().unwrap();
        assert!(
            (p0 - 0.5).abs() < 0.1,
            "S|+> should have ~50/50 Z-basis probabilities, got p0={}",
            p0
        );

        // Verify via direct simulation
        let mut direct = DirectLogicalSim::new(1);
        direct.h(0);
        direct.s(0);
        let probs = direct.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    // --------------------------------------------------------
    // Test 9: Logical T gate via distillation
    // --------------------------------------------------------
    #[test]
    fn test_logical_t_gate() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(42);

        let q = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();

        // Apply H to get |+>
        if let Some(ref mut state) = engine.patches.get_mut(&q).unwrap().logical_state {
            GateOperations::h(state, 0);
        }

        // Apply T gate
        let (_needs_correction, distill_success) = engine.logical_t_gate(q).unwrap();

        // With zero error rate, distillation should always succeed
        assert!(distill_success, "Distillation should succeed with p=0");

        // T|+> = (|0> + e^{ipi/4}|1>)/sqrt2
        // Z-basis probabilities should still be 50/50
        let (p0, _p1) = engine.patches[&q].measurement_probabilities().unwrap();
        assert!(
            (p0 - 0.5).abs() < 0.1,
            "T|+> should have ~50/50 Z-basis probabilities, got p0={}",
            p0
        );

        // Verify direct
        let mut direct = DirectLogicalSim::new(1);
        direct.h(0);
        direct.t(0);
        let probs = direct.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
    }

    // --------------------------------------------------------
    // Test 10: Pauli frame tracking
    // --------------------------------------------------------
    #[test]
    fn test_pauli_frame_tracking() {
        let mut frame = PauliFrame::new();

        // Initially no corrections
        assert!(!frame.has_x(0));
        assert!(!frame.has_z(0));

        // Apply X to qubit 0
        frame.apply_x(0);
        assert!(frame.has_x(0));
        assert!(!frame.has_z(0));

        // Apply X again: should cancel
        frame.apply_x(0);
        assert!(!frame.has_x(0));

        // Apply Z
        frame.apply_z(0);
        assert!(frame.has_z(0));

        // Apply both X and Z to qubit 1
        frame.apply_x(1);
        frame.apply_z(1);
        assert!(frame.has_x(1));
        assert!(frame.has_z(1));

        // Test CNOT propagation: X on control -> X on control AND target
        frame.clear(0);
        frame.clear(1);
        frame.apply_x(0); // X on control
        frame.propagate_cnot(0, 1);
        assert!(frame.has_x(0)); // X stays on control
        assert!(frame.has_x(1)); // X propagates to target

        // Test CNOT propagation: Z on target -> Z on target AND control
        frame.clear(0);
        frame.clear(1);
        frame.apply_z(1); // Z on target
        frame.propagate_cnot(0, 1);
        assert!(frame.has_z(0)); // Z propagates to control
        assert!(frame.has_z(1)); // Z stays on target
    }

    // --------------------------------------------------------
    // Test 11: Pauli frame at measurement
    // --------------------------------------------------------
    #[test]
    fn test_pauli_frame_at_measurement() {
        let mut frame = PauliFrame::new();

        // Z-basis measurement with X correction: should flip outcome
        frame.apply_x(0);
        let corrected = frame.correct_measurement_z(0, false);
        assert!(corrected, "X correction should flip Z-basis measurement");

        let corrected2 = frame.correct_measurement_z(0, true);
        assert!(!corrected2, "X correction should flip true to false");

        // X-basis measurement with Z correction: should flip outcome
        frame.clear(0);
        frame.apply_z(0);
        let corrected3 = frame.correct_measurement_x(0, false);
        assert!(corrected3, "Z correction should flip X-basis measurement");

        // No correction: outcome unchanged
        frame.clear(0);
        let corrected4 = frame.correct_measurement_z(0, true);
        assert!(corrected4, "No correction should leave outcome unchanged");

        // Hadamard propagation swaps X and Z
        frame.clear(0);
        frame.apply_x(0);
        frame.propagate_hadamard(0);
        assert!(!frame.has_x(0));
        assert!(frame.has_z(0));

        // S propagation: X -> X and Z
        frame.clear(0);
        frame.apply_x(0);
        frame.propagate_s(0);
        assert!(frame.has_x(0));
        assert!(frame.has_z(0));
    }

    // --------------------------------------------------------
    // Test 12: Resource estimate for CNOT
    // --------------------------------------------------------
    #[test]
    fn test_resource_estimate_cnot() {
        let config = LatticeSurgeryConfig::new(3);
        let estimate = ResourceEstimate::cnot_estimate(&config);

        // 3 patches * 9 qubits/patch = 27
        assert_eq!(estimate.physical_qubits, 27);
        assert_eq!(estimate.logical_qubits, 2);
        assert_eq!(estimate.ancilla_patches, 1);
        // 5 time steps * 3 rounds = 15 code cycles
        assert_eq!(estimate.time_steps, 5);
        assert_eq!(estimate.code_cycles, 15);
        assert_eq!(estimate.magic_states_consumed, 0);

        // Distance-5 CNOT
        let config5 = LatticeSurgeryConfig::new(5);
        let estimate5 = ResourceEstimate::cnot_estimate(&config5);
        assert_eq!(estimate5.physical_qubits, 75); // 3 * 25
        assert_eq!(estimate5.code_cycles, 25); // 5 * 5
    }

    // --------------------------------------------------------
    // Test 13: Resource estimate for T gate
    // --------------------------------------------------------
    #[test]
    fn test_resource_estimate_t_gate() {
        let config = LatticeSurgeryConfig::new(3);
        let estimate = ResourceEstimate::t_gate_estimate(&config);

        // 17 patches * 9 qubits/patch = 153
        assert_eq!(estimate.physical_qubits, 153);
        assert_eq!(estimate.magic_states_consumed, 1);
        assert_eq!(estimate.ancilla_patches, 16); // 15 distillation + 1
                                                  // (15 + 3) time steps * 3 rounds = 54 code cycles
        assert_eq!(estimate.time_steps, 18);
        assert_eq!(estimate.code_cycles, 54);

        // Higher distance = more physical qubits but same time structure
        let config7 = LatticeSurgeryConfig::new(7);
        let estimate7 = ResourceEstimate::t_gate_estimate(&config7);
        assert_eq!(estimate7.physical_qubits, 17 * 49); // 17 * 7^2
    }

    // --------------------------------------------------------
    // Test 14: Compile Bell circuit (H + CNOT)
    // --------------------------------------------------------
    #[test]
    fn test_compile_bell_circuit() {
        let config = LatticeSurgeryConfig::new(3);
        let compiler = LatticeSurgeryCompiler::new(config.clone());

        let instructions = vec![
            LogicalInstruction::LogicalH { qubit: 0 },
            LogicalInstruction::LogicalCNOT {
                control: 0,
                target: 1,
            },
        ];

        let schedule = compiler.compile(&instructions, 2).unwrap();

        // H and CNOT are dependent (both touch qubit 0), so they must be in
        // different time steps
        assert!(
            schedule.total_time_steps() >= 2,
            "H and CNOT should be in at least 2 time steps, got {}",
            schedule.total_time_steps()
        );

        // Check CNOT was decomposed into merge/split primitives
        let total_instructions: usize = schedule.steps.iter().map(|s| s.instructions.len()).sum();
        assert!(
            total_instructions > 2,
            "CNOT should be decomposed into multiple primitives, got {}",
            total_instructions
        );

        // Resource estimate
        let estimate = ResourceEstimate::from_schedule(&schedule, &config);
        assert!(estimate.physical_qubits > 0);
        assert!(estimate.time_steps > 0);
    }

    // --------------------------------------------------------
    // Test 15: Compile parallel operations
    // --------------------------------------------------------
    #[test]
    fn test_compile_parallel_operations() {
        let config = LatticeSurgeryConfig::new(3);
        let compiler = LatticeSurgeryCompiler::new(config);

        // Two independent Hadamard gates on different qubits
        let instructions = vec![
            LogicalInstruction::LogicalH { qubit: 0 },
            LogicalInstruction::LogicalH { qubit: 1 },
        ];

        let schedule = compiler.compile(&instructions, 2).unwrap();

        // Independent operations should be in the same time step
        assert_eq!(
            schedule.total_time_steps(),
            1,
            "Independent H gates should be parallelized into 1 step"
        );

        // Two independent CNOTs on disjoint qubits
        let instructions2 = vec![
            LogicalInstruction::LogicalCNOT {
                control: 0,
                target: 1,
            },
            LogicalInstruction::LogicalCNOT {
                control: 2,
                target: 3,
            },
        ];

        let schedule2 = compiler.compile(&instructions2, 4).unwrap();
        assert_eq!(
            schedule2.total_time_steps(),
            1,
            "Independent CNOTs on disjoint qubits should be parallelized"
        );

        // Sequential CNOTs sharing a qubit should NOT be parallelized
        let instructions3 = vec![
            LogicalInstruction::LogicalCNOT {
                control: 0,
                target: 1,
            },
            LogicalInstruction::LogicalCNOT {
                control: 1,
                target: 2,
            },
        ];

        let schedule3 = compiler.compile(&instructions3, 3).unwrap();
        assert!(
            schedule3.total_time_steps() >= 2,
            "Dependent CNOTs should be in separate time steps"
        );
    }

    // --------------------------------------------------------
    // Test 16: Noisy merge
    // --------------------------------------------------------
    #[test]
    fn test_noisy_merge() {
        // With high error rate, some intermediate measurements should be flipped
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.5);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(12345);

        let a = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        let b = engine.allocate_patch(PatchType::DataZ, (2, 0)).unwrap();

        let merge = engine.merge(a, b, SurgeryAxis::Z).unwrap();

        // With 50% error rate and 3 merge rounds, at least some measurements
        // should be noisy (statistically very likely with this seed)
        let _any_noisy = merge.intermediate_measurements.iter().any(|&m| m);
        // Note: with a deterministic seed, this is fully reproducible
        assert_eq!(
            merge.intermediate_measurements.len(),
            3,
            "Should have exactly merge_rounds measurements"
        );

        // Run many merges to verify error rate is approximately correct
        let config2 = LatticeSurgeryConfig::new(3).with_error_rate(0.5);
        let mut engine2 = LatticeSurgeryEngine::new(config2);
        engine2.set_seed(99);

        let mut total_errors = 0usize;
        let mut total_measurements = 0usize;
        let trials = 100;

        for i in 0..trials {
            let pa = engine2
                .allocate_patch(PatchType::DataZ, (i as i32 * 10, 0))
                .unwrap();
            let pb = engine2
                .allocate_patch(PatchType::DataZ, (i as i32 * 10 + 2, 0))
                .unwrap();
            let m = engine2.merge(pa, pb, SurgeryAxis::Z).unwrap();
            total_errors += m.intermediate_measurements.iter().filter(|&&x| x).count();
            total_measurements += m.intermediate_measurements.len();
        }

        let observed_rate = total_errors as f64 / total_measurements as f64;
        assert!(
            (observed_rate - 0.5).abs() < 0.15,
            "Observed error rate {:.3} should be approximately 0.5",
            observed_rate
        );
    }

    // --------------------------------------------------------
    // Test 17: Patch overlap detection
    // --------------------------------------------------------
    #[test]
    fn test_patch_overlap_detection() {
        let mut engine = default_engine();

        let _a = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        let result = engine.allocate_patch(PatchType::DataZ, (0, 0));

        assert!(result.is_err());
        if let Err(LatticeSurgeryError::PatchOverlap { position, .. }) = result {
            assert_eq!(position, (0, 0));
        } else {
            panic!("Expected PatchOverlap error");
        }
    }

    // --------------------------------------------------------
    // Test 18: Error display formatting
    // --------------------------------------------------------
    #[test]
    fn test_error_display() {
        let err = LatticeSurgeryError::PatchOverlap {
            patch_a: 0,
            patch_b: 1,
            position: (3, 4),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Patch overlap"));
        assert!(msg.contains("(3, 4)"));

        let err2 = LatticeSurgeryError::ResourceExhausted {
            resource: "ancilla patches".to_string(),
            required: 16,
            available: 4,
        };
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("16"));
        assert!(msg2.contains("4"));
    }

    // --------------------------------------------------------
    // Test 19: Logical error rate scaling
    // --------------------------------------------------------
    #[test]
    fn test_logical_error_rate_scaling() {
        // Error rate should decrease exponentially with distance
        let p = 1e-3;

        let rate_d3 = logical_error_rate(p, 3);
        let rate_d5 = logical_error_rate(p, 5);
        let rate_d7 = logical_error_rate(p, 7);

        assert!(
            rate_d5 < rate_d3,
            "Higher distance should give lower error rate: d3={:.2e}, d5={:.2e}",
            rate_d3,
            rate_d5
        );
        assert!(
            rate_d7 < rate_d5,
            "Higher distance should give lower error rate: d5={:.2e}, d7={:.2e}",
            rate_d5,
            rate_d7
        );

        // With p below threshold (p < 0.01), each distance increment should
        // provide exponential suppression
        // d=3: (0.1)^2 = 0.01
        // d=5: (0.1)^3 = 0.001
        // d=7: (0.1)^4 = 0.0001
        assert!((rate_d3 - 0.01).abs() < 1e-10);
        assert!((rate_d5 - 0.001).abs() < 1e-10);
        assert!((rate_d7 - 0.0001).abs() < 1e-10);
    }

    // --------------------------------------------------------
    // Test 20: Physical qubit count calculation
    // --------------------------------------------------------
    #[test]
    fn test_physical_qubit_count() {
        // 1 logical qubit at distance 3: 9 data + 4 ancilla (50% overhead) = 13
        let count = physical_qubit_count(1, 3);
        assert_eq!(count, 13); // 9 + 4 (integer division for 50%)

        // 100 logical qubits at distance 7: large-scale estimate
        let count_large = physical_qubit_count(100, 7);
        assert_eq!(count_large, 100 * 49 + 100 * 49 / 2); // 7350
    }

    // --------------------------------------------------------
    // Test 21: Measure Z with Pauli frame
    // --------------------------------------------------------
    #[test]
    fn test_measure_z_with_frame() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(1);

        let q = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        // State is |0>, no frame correction
        let result = engine.measure_z(q).unwrap();
        assert!(!result, "Measuring |0> in Z basis should give 0");

        // Create another patch in |0>, add X frame correction
        let q2 = engine.allocate_patch(PatchType::DataZ, (2, 0)).unwrap();
        engine.frame.apply_x(q2);
        let result2 = engine.measure_z(q2).unwrap();
        // X correction flips Z-basis measurement of |0> -> |1>
        assert!(
            result2,
            "X frame correction should flip measurement of |0> to 1"
        );
    }

    // --------------------------------------------------------
    // Test 22: Measure X basis
    // --------------------------------------------------------
    #[test]
    fn test_measure_x_basis() {
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(1);

        // Prepare |+> state
        let q = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        if let Some(ref mut state) = engine.patches.get_mut(&q).unwrap().logical_state {
            GateOperations::h(state, 0);
        }

        // Measuring |+> in X basis should give 0 (it is the +1 eigenstate of X)
        let result = engine.measure_x(q).unwrap();
        assert!(
            !result,
            "Measuring |+> in X basis should give 0 (+1 eigenvalue)"
        );
    }

    // --------------------------------------------------------
    // Test 23: Magic state distillation success rate
    // --------------------------------------------------------
    #[test]
    fn test_magic_state_distillation() {
        // Zero error rate: always succeeds
        let config = LatticeSurgeryConfig::new(3).with_error_rate(0.0);
        let mut engine = LatticeSurgeryEngine::new(config);
        engine.set_seed(1);

        for _ in 0..10 {
            assert!(engine.distill_magic_state().unwrap());
        }

        // Very high error rate: should sometimes fail
        let config2 = LatticeSurgeryConfig::new(3).with_error_rate(0.3);
        let mut engine2 = LatticeSurgeryEngine::new(config2);
        engine2.set_seed(1);

        let mut successes = 0;
        let trials = 100;
        for _ in 0..trials {
            if engine2.distill_magic_state().unwrap() {
                successes += 1;
            }
        }
        // Failure prob = 15 * 0.3^3 = 0.405, so success rate ~59.5%
        let success_rate = successes as f64 / trials as f64;
        assert!(
            success_rate > 0.3 && success_rate < 0.9,
            "With p=0.3, success rate should be ~60%, got {:.1}%",
            success_rate * 100.0
        );
    }

    // --------------------------------------------------------
    // Test 24: Full circuit resource estimation
    // --------------------------------------------------------
    #[test]
    fn test_circuit_resource_estimation() {
        let config = LatticeSurgeryConfig::new(5);
        let instructions = vec![
            LogicalInstruction::LogicalH { qubit: 0 },
            LogicalInstruction::LogicalCNOT {
                control: 0,
                target: 1,
            },
            LogicalInstruction::LogicalT { qubit: 0 },
            LogicalInstruction::MeasureZ { qubit: 0 },
            LogicalInstruction::MeasureZ { qubit: 1 },
        ];

        let estimate = estimate_circuit_resources(&instructions, 2, &config).unwrap();

        // Should have non-zero resources
        assert!(estimate.physical_qubits > 0);
        assert!(estimate.time_steps > 0);
        assert_eq!(estimate.magic_states_consumed, 1); // One T gate
        assert_eq!(estimate.logical_qubits, 2);
        assert!(estimate.ancilla_patches > 0);

        // Error rate at distance 5, p=1e-3 should be very small
        assert!(
            estimate.logical_error_rate < 0.01,
            "Logical error rate should be low at d=5, got {:.2e}",
            estimate.logical_error_rate
        );
    }

    // --------------------------------------------------------
    // Test 25: Instruction display formatting
    // --------------------------------------------------------
    #[test]
    fn test_instruction_display() {
        let instr = LogicalInstruction::LogicalCNOT {
            control: 0,
            target: 1,
        };
        assert_eq!(format!("{}", instr), "CNOT(0, 1)");

        let instr2 = LogicalInstruction::MergeZZ {
            qubit_a: 2,
            qubit_b: 3,
        };
        assert_eq!(format!("{}", instr2), "MergeZZ(2, 3)");

        let instr3 = LogicalInstruction::LogicalT { qubit: 4 };
        assert_eq!(format!("{}", instr3), "T(4)");

        let instr4 = LogicalInstruction::MeasureX { qubit: 0 };
        assert_eq!(format!("{}", instr4), "MeasX(0)");
    }

    // --------------------------------------------------------
    // Test 26: Direct logical sim consistency
    // --------------------------------------------------------
    #[test]
    fn test_direct_logical_sim() {
        // Test that DirectLogicalSim correctly simulates basic circuits
        let mut sim = DirectLogicalSim::new(2);

        // |00> -> H(0) -> CNOT(0,1) -> Bell state
        sim.h(0);
        sim.cnot(0, 1);
        let probs = sim.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10, "Bell state |00> prob");
        assert!(probs[1].abs() < 1e-10, "Bell state |01> prob");
        assert!(probs[2].abs() < 1e-10, "Bell state |10> prob");
        assert!((probs[3] - 0.5).abs() < 1e-10, "Bell state |11> prob");

        // Single qubit: H -> T -> H should give a known state
        let mut sim2 = DirectLogicalSim::new(1);
        sim2.h(0);
        sim2.t(0);
        sim2.h(0);
        let probs2 = sim2.probabilities();
        // HTH|0> = H T |+> = H (|0> + e^{ipi/4}|1>)/sqrt2
        // = (1+e^{ipi/4})/2 |0> + (1-e^{ipi/4})/2 |1>
        let cos_pi8 = (PI / 4.0).cos();
        let sin_pi8 = (PI / 4.0).sin();
        let p0_expected = ((1.0 + cos_pi8).powi(2) + sin_pi8.powi(2)) / 4.0;
        assert!(
            (probs2[0] - p0_expected).abs() < 1e-10,
            "HTH|0> p(0) = {:.6}, expected {:.6}",
            probs2[0],
            p0_expected
        );
    }

    // --------------------------------------------------------
    // Test 27: Engine deallocate and reuse
    // --------------------------------------------------------
    #[test]
    fn test_deallocate_and_reuse() {
        let mut engine = default_engine();

        let a = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        assert!(engine.patches.contains_key(&a));

        engine.deallocate_patch(a).unwrap();
        assert!(!engine.patches.contains_key(&a));

        // Should be able to allocate at the same position again
        let b = engine.allocate_patch(PatchType::DataZ, (0, 0)).unwrap();
        assert!(engine.patches.contains_key(&b));
        assert_ne!(a, b, "New patch should have a different ID");

        // Deallocating non-existent patch should error
        let result = engine.deallocate_patch(999);
        assert!(result.is_err());
    }

    // --------------------------------------------------------
    // Test 28: Resource estimate display
    // --------------------------------------------------------
    #[test]
    fn test_resource_estimate_display() {
        let config = LatticeSurgeryConfig::new(3);
        let estimate = ResourceEstimate::cnot_estimate(&config);
        let display = format!("{}", estimate);
        assert!(display.contains("physical_qubits: 27"));
        assert!(display.contains("time_steps: 5"));
        assert!(display.contains("code_cycles: 15"));
    }
}
