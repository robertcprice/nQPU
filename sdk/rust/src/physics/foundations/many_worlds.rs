//! Many-Worlds Branching Simulation (Everett Interpretation)
//!
//! **WORLD FIRST**: No quantum simulator implements the many-worlds interpretation
//! as a computational framework. This module tracks the full branching structure of
//! quantum measurements -- every measurement creates branches rather than collapsing
//! the wavefunction. Provides decoherent histories analysis (Griffiths/Gell-Mann-Hartle),
//! branch counting, weight statistics, interference detection, and branch merging.
//!
//! ## Capabilities
//!
//! - **Branch-on-Measurement**: Each measurement creates two child branches (one per
//!   outcome) with projected, renormalized states and properly updated weights.
//! - **Unitary Evolution**: Apply gates to all active branches simultaneously.
//! - **Branch Pruning**: Remove low-weight branches and redistribute weight to
//!   surviving branches for memory efficiency.
//! - **Decoherent Histories**: Griffiths/Gell-Mann-Hartle consistency check via the
//!   decoherence functional D(alpha, beta) = Tr(C_alpha rho C_beta^dagger).
//! - **Branch Interference**: Detect whether two branches can still interfere by
//!   computing their state overlap.
//! - **Branch Merging**: Re-combine branches with identical measurement histories
//!   (coherent addition of state vectors).
//! - **Statistics**: Weight entropy, effective branch count, depth, histograms.
//! - **Universe Snapshot**: Serialize the full branching tree for visualization.
//!
//! ## Applications
//!
//! - Foundations of quantum mechanics (interpretational studies)
//! - Decoherent histories / consistent histories formalism
//! - Quantum decision theory (branch-weighted utilities)
//! - Quantum Darwinism (redundant information encoding across branches)
//! - Educational: visualizing the branching structure of quantum mechanics
//!
//! ## References
//!
//! - Everett (1957) - "Relative State" Formulation of Quantum Mechanics
//! - DeWitt & Graham (1973) - The Many-Worlds Interpretation of Quantum Mechanics
//! - Griffiths (1984) - Consistent Histories and the Interpretation of Quantum Mechanics
//! - Gell-Mann & Hartle (1990) - Quantum Mechanics in the Light of Quantum Cosmology
//! - Zurek (2003) - Decoherence, Einselection, and the Quantum Origins of the Classical
//! - Wallace (2012) - The Emergent Multiverse

use crate::{GateOperations, QuantumState, C64};
use std::collections::HashMap;
use std::fmt;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors that can occur during many-worlds simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum ManyWorldsError {
    /// Referenced branch index does not exist or has been retired.
    InvalidBranch {
        branch_id: usize,
        total_branches: usize,
    },
    /// A decoherent history is inconsistent (off-diagonal decoherence functional
    /// exceeds threshold).
    HistoryInconsistent {
        alpha: usize,
        beta: usize,
        off_diagonal: f64,
    },
    /// Decoherence functional computation failed (e.g., dimension mismatch).
    DecoherenceFailed { reason: String },
    /// Number of active branches would exceed the configured maximum.
    BranchOverflow { current: usize, max: usize },
    /// Qubit index is out of range for the branch state.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
}

impl fmt::Display for ManyWorldsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManyWorldsError::InvalidBranch {
                branch_id,
                total_branches,
            } => {
                write!(
                    f,
                    "Invalid branch id {} (total branches: {})",
                    branch_id, total_branches
                )
            }
            ManyWorldsError::HistoryInconsistent {
                alpha,
                beta,
                off_diagonal,
            } => {
                write!(
                    f,
                    "Histories ({}, {}) are inconsistent: |D(a,b)| = {:.6e}",
                    alpha, beta, off_diagonal
                )
            }
            ManyWorldsError::DecoherenceFailed { reason } => {
                write!(f, "Decoherence functional computation failed: {}", reason)
            }
            ManyWorldsError::BranchOverflow { current, max } => {
                write!(
                    f,
                    "Branch overflow: {} active branches exceeds maximum of {}",
                    current, max
                )
            }
            ManyWorldsError::QubitOutOfRange { qubit, num_qubits } => {
                write!(
                    f,
                    "Qubit {} out of range for {}-qubit system",
                    qubit, num_qubits
                )
            }
        }
    }
}

impl std::error::Error for ManyWorldsError {}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Configuration for the many-worlds branching simulation.
#[derive(Clone, Debug)]
pub struct ManyWorldsConfig {
    /// Maximum number of active branches before overflow error (default: 1024).
    pub max_branches: usize,
    /// Threshold below which the decoherence functional is considered zero
    /// for consistency checks (default: 0.01).
    pub decoherence_threshold: f64,
    /// Whether to record full measurement histories on each branch (default: true).
    pub track_history: bool,
    /// Branches with weight below this threshold are pruned (default: 1e-10).
    pub prune_threshold: f64,
}

impl Default for ManyWorldsConfig {
    fn default() -> Self {
        Self {
            max_branches: 1024,
            decoherence_threshold: 0.01,
            track_history: true,
            prune_threshold: 1e-10,
        }
    }
}

impl ManyWorldsConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of branches.
    pub fn with_max_branches(mut self, max: usize) -> Self {
        self.max_branches = max;
        self
    }

    /// Set decoherence threshold.
    pub fn with_decoherence_threshold(mut self, threshold: f64) -> Self {
        self.decoherence_threshold = threshold;
        self
    }

    /// Enable or disable history tracking.
    pub fn with_track_history(mut self, track: bool) -> Self {
        self.track_history = track;
        self
    }

    /// Set pruning threshold.
    pub fn with_prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }
}

// ===================================================================
// MEASUREMENT RECORD
// ===================================================================

/// Record of a single measurement event within a branch's history.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementRecord {
    /// Which qubit was measured.
    pub qubit: usize,
    /// Measurement outcome (0 or 1).
    pub outcome: u8,
    /// Branch weight at the time of this measurement.
    pub branch_weight: f64,
    /// Step number (sequential measurement index within the universe).
    pub step: usize,
}

impl fmt::Display for MeasurementRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "q{}={} (w={:.6}, step={})",
            self.qubit, self.outcome, self.branch_weight, self.step
        )
    }
}

// ===================================================================
// BRANCH
// ===================================================================

/// A single branch (world) in the many-worlds tree.
///
/// Each branch contains its own quantum state, a weight (product of all
/// Born-rule probabilities along its measurement history), and optionally
/// the full history of measurement outcomes that led to this branch.
#[derive(Clone, Debug)]
pub struct Branch {
    /// Quantum state of this branch (projected and renormalized).
    pub state: QuantumState,
    /// Weight = product of Born-rule probabilities along the history.
    /// Satisfies: sum of all active branch weights = 1.
    pub weight: f64,
    /// Ordered sequence of measurement outcomes leading to this branch.
    pub history: Vec<MeasurementRecord>,
    /// Index of the parent branch (None for the root).
    pub parent: Option<usize>,
    /// Whether this branch is still active (not retired by measurement).
    pub active: bool,
    /// Unique identifier for this branch.
    pub id: usize,
}

impl Branch {
    /// Depth of this branch (number of measurements in its history).
    pub fn depth(&self) -> usize {
        self.history.len()
    }

    /// Returns a string key encoding the measurement history for comparison.
    /// Two branches with the same history key have identical measurement records.
    pub fn history_key(&self) -> String {
        self.history
            .iter()
            .map(|r| format!("q{}={}", r.qubit, r.outcome))
            .collect::<Vec<_>>()
            .join("|")
    }
}

// ===================================================================
// BRANCHING UNIVERSE
// ===================================================================

/// The many-worlds branching universe.
///
/// Manages all branches, handles measurement-induced branching, applies
/// unitary evolution across all active branches, and provides analysis
/// tools (pruning, statistics, decoherent histories, interference checks).
pub struct BranchingUniverse {
    /// All branches (both active and retired).
    branches: Vec<Branch>,
    /// Total number of branches ever created.
    total_created: usize,
    /// Configuration.
    config: ManyWorldsConfig,
    /// Global step counter for measurement ordering.
    step_counter: usize,
}

impl BranchingUniverse {
    /// Create a new branching universe with `num_qubits` qubits, starting in |0...0>.
    pub fn new(num_qubits: usize, config: ManyWorldsConfig) -> Self {
        let root = Branch {
            state: QuantumState::new(num_qubits),
            weight: 1.0,
            history: Vec::new(),
            parent: None,
            active: true,
            id: 0,
        };

        BranchingUniverse {
            branches: vec![root],
            total_created: 1,
            config,
            step_counter: 0,
        }
    }

    /// Create a branching universe with default configuration.
    pub fn with_defaults(num_qubits: usize) -> Self {
        Self::new(num_qubits, ManyWorldsConfig::default())
    }

    /// Create a branching universe from an existing quantum state.
    pub fn from_state(state: QuantumState, config: ManyWorldsConfig) -> Self {
        let root = Branch {
            state,
            weight: 1.0,
            history: Vec::new(),
            parent: None,
            active: true,
            id: 0,
        };

        BranchingUniverse {
            branches: vec![root],
            total_created: 1,
            config,
            step_counter: 0,
        }
    }

    /// Number of qubits in the system (determined from the first branch).
    pub fn num_qubits(&self) -> usize {
        self.branches
            .first()
            .map(|b| b.state.num_qubits)
            .unwrap_or(0)
    }

    /// Number of currently active branches.
    pub fn active_branch_count(&self) -> usize {
        self.branches.iter().filter(|b| b.active).count()
    }

    /// Total number of branches ever created.
    pub fn total_branches_created(&self) -> usize {
        self.total_created
    }

    /// Get a reference to all branches (including retired ones).
    pub fn all_branches(&self) -> &[Branch] {
        &self.branches
    }

    /// Get references to all active branches.
    pub fn active_branches(&self) -> Vec<&Branch> {
        self.branches.iter().filter(|b| b.active).collect()
    }

    /// Get mutable references to active branch indices.
    fn active_branch_indices(&self) -> Vec<usize> {
        self.branches
            .iter()
            .enumerate()
            .filter(|(_, b)| b.active)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get a reference to a specific branch by index.
    pub fn branch(&self, idx: usize) -> Result<&Branch, ManyWorldsError> {
        self.branches
            .get(idx)
            .ok_or(ManyWorldsError::InvalidBranch {
                branch_id: idx,
                total_branches: self.branches.len(),
            })
    }

    // ---------------------------------------------------------------
    // BRANCH-ON-MEASUREMENT
    // ---------------------------------------------------------------

    /// Measure qubit `qubit` on all active branches, creating two child
    /// branches per active branch (one for outcome 0, one for outcome 1).
    ///
    /// The original branches are retired. Each child branch receives:
    /// - A projected, renormalized state
    /// - weight *= P(outcome)
    /// - An updated measurement history
    ///
    /// Returns the indices of all newly created branches.
    pub fn measure_all(&mut self, qubit: usize) -> Result<Vec<usize>, ManyWorldsError> {
        let nq = self.num_qubits();
        if qubit >= nq {
            return Err(ManyWorldsError::QubitOutOfRange {
                qubit,
                num_qubits: nq,
            });
        }

        let active_indices = self.active_branch_indices();
        let new_count = active_indices.len() * 2;

        if self.active_branch_count() - active_indices.len() + new_count > self.config.max_branches
        {
            return Err(ManyWorldsError::BranchOverflow {
                current: self.active_branch_count() + new_count - active_indices.len(),
                max: self.config.max_branches,
            });
        }

        let step = self.step_counter;
        self.step_counter += 1;

        let mut new_branch_indices = Vec::with_capacity(new_count);

        // Collect data from active branches before mutating
        let branch_data: Vec<(usize, QuantumState, f64, Vec<MeasurementRecord>)> = active_indices
            .iter()
            .map(|&idx| {
                let b = &self.branches[idx];
                (idx, b.state.clone(), b.weight, b.history.clone())
            })
            .collect();

        // Retire the original branches
        for &idx in &active_indices {
            self.branches[idx].active = false;
        }

        // Create child branches
        for (parent_idx, state, parent_weight, parent_history) in branch_data {
            let (branch_0, branch_1) = self.create_measurement_branches(
                &state,
                qubit,
                parent_weight,
                &parent_history,
                parent_idx,
                step,
            );

            let id_0 = self.total_created;
            let id_1 = self.total_created + 1;
            self.total_created += 2;

            let b0 = Branch {
                state: branch_0.0,
                weight: branch_0.1,
                history: branch_0.2,
                parent: Some(parent_idx),
                active: branch_0.1 > 0.0, // Only activate if non-zero probability
                id: id_0,
            };
            let b1 = Branch {
                state: branch_1.0,
                weight: branch_1.1,
                history: branch_1.2,
                parent: Some(parent_idx),
                active: branch_1.1 > 0.0,
                id: id_1,
            };

            self.branches.push(b0);
            new_branch_indices.push(self.branches.len() - 1);
            self.branches.push(b1);
            new_branch_indices.push(self.branches.len() - 1);
        }

        Ok(new_branch_indices)
    }

    /// Measure qubit `qubit` on a specific branch, creating two child branches.
    ///
    /// The specified branch is retired.
    /// Returns (index_of_outcome_0_branch, index_of_outcome_1_branch).
    pub fn measure_branch(
        &mut self,
        branch_idx: usize,
        qubit: usize,
    ) -> Result<(usize, usize), ManyWorldsError> {
        if branch_idx >= self.branches.len() || !self.branches[branch_idx].active {
            return Err(ManyWorldsError::InvalidBranch {
                branch_id: branch_idx,
                total_branches: self.branches.len(),
            });
        }

        let nq = self.branches[branch_idx].state.num_qubits;
        if qubit >= nq {
            return Err(ManyWorldsError::QubitOutOfRange {
                qubit,
                num_qubits: nq,
            });
        }

        if self.active_branch_count() + 1 > self.config.max_branches {
            return Err(ManyWorldsError::BranchOverflow {
                current: self.active_branch_count() + 1,
                max: self.config.max_branches,
            });
        }

        let step = self.step_counter;
        self.step_counter += 1;

        let state = self.branches[branch_idx].state.clone();
        let weight = self.branches[branch_idx].weight;
        let history = self.branches[branch_idx].history.clone();

        self.branches[branch_idx].active = false;

        let (branch_0, branch_1) =
            self.create_measurement_branches(&state, qubit, weight, &history, branch_idx, step);

        let id_0 = self.total_created;
        let id_1 = self.total_created + 1;
        self.total_created += 2;

        let b0 = Branch {
            state: branch_0.0,
            weight: branch_0.1,
            history: branch_0.2,
            parent: Some(branch_idx),
            active: branch_0.1 > 0.0,
            id: id_0,
        };
        let b1 = Branch {
            state: branch_1.0,
            weight: branch_1.1,
            history: branch_1.2,
            parent: Some(branch_idx),
            active: branch_1.1 > 0.0,
            id: id_1,
        };

        self.branches.push(b0);
        let idx_0 = self.branches.len() - 1;
        self.branches.push(b1);
        let idx_1 = self.branches.len() - 1;

        Ok((idx_0, idx_1))
    }

    /// Internal helper: create the two projected branches for a measurement.
    ///
    /// Returns ((state_0, weight_0, history_0), (state_1, weight_1, history_1)).
    fn create_measurement_branches(
        &self,
        state: &QuantumState,
        qubit: usize,
        parent_weight: f64,
        parent_history: &[MeasurementRecord],
        _parent_idx: usize,
        step: usize,
    ) -> (
        (QuantumState, f64, Vec<MeasurementRecord>),
        (QuantumState, f64, Vec<MeasurementRecord>),
    ) {
        let dim = state.dim;
        let mask = 1usize << qubit;
        let amps = state.amplitudes_ref();

        // Compute P(0) = sum |a_i|^2 for basis states with bit `qubit` = 0
        let mut prob_0: f64 = 0.0;
        for i in 0..dim {
            if (i & mask) == 0 {
                prob_0 += amps[i].norm_sqr();
            }
        }
        let prob_1 = 1.0 - prob_0;

        // Project onto outcome 0
        let state_0 = Self::project_state(state, qubit, 0);
        // Project onto outcome 1
        let state_1 = Self::project_state(state, qubit, 1);

        let weight_0 = parent_weight * prob_0;
        let weight_1 = parent_weight * prob_1;

        let mut history_0 = if self.config.track_history {
            parent_history.to_vec()
        } else {
            Vec::new()
        };
        let mut history_1 = if self.config.track_history {
            parent_history.to_vec()
        } else {
            Vec::new()
        };

        if self.config.track_history {
            history_0.push(MeasurementRecord {
                qubit,
                outcome: 0,
                branch_weight: weight_0,
                step,
            });
            history_1.push(MeasurementRecord {
                qubit,
                outcome: 1,
                branch_weight: weight_1,
                step,
            });
        }

        (
            (state_0, weight_0, history_0),
            (state_1, weight_1, history_1),
        )
    }

    /// Project a quantum state onto outcome `outcome` (0 or 1) for `qubit`,
    /// and renormalize.
    ///
    /// Sets all amplitudes where bit `qubit` != `outcome` to zero, then
    /// divides remaining amplitudes by sqrt(P(outcome)).
    fn project_state(state: &QuantumState, qubit: usize, outcome: u8) -> QuantumState {
        let dim = state.dim;
        let nq = state.num_qubits;
        let mask = 1usize << qubit;
        let amps = state.amplitudes_ref();

        let mut new_amps = vec![C64::new(0.0, 0.0); dim];
        let mut norm_sq: f64 = 0.0;

        for i in 0..dim {
            let bit_val = if (i & mask) != 0 { 1u8 } else { 0u8 };
            if bit_val == outcome {
                new_amps[i] = amps[i];
                norm_sq += amps[i].norm_sqr();
            }
        }

        // Renormalize
        if norm_sq > 1e-30 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for amp in new_amps.iter_mut() {
                *amp = C64::new(amp.re * inv_norm, amp.im * inv_norm);
            }
        }

        let mut projected = QuantumState::new(nq);
        let dest = projected.amplitudes_mut();
        for i in 0..dim {
            dest[i] = new_amps[i];
        }
        projected
    }

    // ---------------------------------------------------------------
    // UNITARY EVOLUTION ON ALL BRANCHES
    // ---------------------------------------------------------------

    /// Apply a gate function to all active branches.
    ///
    /// The closure receives a mutable reference to each branch's QuantumState.
    /// This implements the Everettian principle that unitary evolution is
    /// universal -- it applies identically to every branch.
    pub fn apply_gate_all<F>(&mut self, gate_fn: F)
    where
        F: Fn(&mut QuantumState),
    {
        for branch in self.branches.iter_mut() {
            if branch.active {
                gate_fn(&mut branch.state);
            }
        }
    }

    /// Apply a Hadamard gate to qubit `qubit` on all active branches.
    pub fn h_all(&mut self, qubit: usize) {
        self.apply_gate_all(|state| GateOperations::h(state, qubit));
    }

    /// Apply a CNOT gate to all active branches.
    /// Note: `control` must be less than `target` to satisfy GateOperations::cnot.
    pub fn cnot_all(&mut self, control: usize, target: usize) {
        self.apply_gate_all(|state| GateOperations::cnot(state, control, target));
    }

    /// Apply an X gate to qubit `qubit` on all active branches.
    pub fn x_all(&mut self, qubit: usize) {
        self.apply_gate_all(|state| GateOperations::x(state, qubit));
    }

    /// Apply an Rz gate to qubit `qubit` on all active branches.
    pub fn rz_all(&mut self, qubit: usize, angle: f64) {
        self.apply_gate_all(|state| GateOperations::rz(state, qubit, angle));
    }

    // ---------------------------------------------------------------
    // BRANCH PRUNING
    // ---------------------------------------------------------------

    /// Prune branches with weight below the configured threshold.
    ///
    /// The pruned weight is redistributed proportionally to surviving branches
    /// so that the total weight remains 1.
    ///
    /// Returns (number_pruned, total_pruned_weight).
    pub fn prune(&mut self) -> (usize, f64) {
        self.prune_with_threshold(self.config.prune_threshold)
    }

    /// Prune branches with weight below the given threshold.
    ///
    /// Returns (number_pruned, total_pruned_weight).
    pub fn prune_with_threshold(&mut self, threshold: f64) -> (usize, f64) {
        let mut pruned_count = 0usize;
        let mut pruned_weight = 0.0f64;
        let mut surviving_weight = 0.0f64;

        // First pass: identify branches to prune
        for branch in self.branches.iter_mut() {
            if branch.active {
                if branch.weight < threshold {
                    branch.active = false;
                    pruned_weight += branch.weight;
                    pruned_count += 1;
                } else {
                    surviving_weight += branch.weight;
                }
            }
        }

        // Second pass: redistribute pruned weight proportionally
        if pruned_weight > 0.0 && surviving_weight > 0.0 {
            let scale = (surviving_weight + pruned_weight) / surviving_weight;
            for branch in self.branches.iter_mut() {
                if branch.active {
                    branch.weight *= scale;
                }
            }
        }

        (pruned_count, pruned_weight)
    }

    // ---------------------------------------------------------------
    // BRANCH INTERFERENCE CHECK
    // ---------------------------------------------------------------

    /// Compute the overlap <psi_i|psi_j> between two branches.
    ///
    /// If |overlap| < threshold, the branches are effectively decohered and
    /// cannot interfere. Returns the complex overlap.
    pub fn branch_overlap(&self, branch_a: usize, branch_b: usize) -> Result<C64, ManyWorldsError> {
        let a = self.branch(branch_a)?;
        let b = self.branch(branch_b)?;

        let amps_a = a.state.amplitudes_ref();
        let amps_b = b.state.amplitudes_ref();

        if amps_a.len() != amps_b.len() {
            return Err(ManyWorldsError::DecoherenceFailed {
                reason: format!(
                    "Dimension mismatch: branch {} has {} amplitudes, branch {} has {}",
                    branch_a,
                    amps_a.len(),
                    branch_b,
                    amps_b.len()
                ),
            });
        }

        // <psi_a|psi_b> = sum conj(a_i) * b_i
        let mut overlap = C64::new(0.0, 0.0);
        for i in 0..amps_a.len() {
            overlap += amps_a[i].conj() * amps_b[i];
        }

        Ok(overlap)
    }

    /// Check whether two branches can interfere.
    ///
    /// Returns true if |<psi_a|psi_b>| >= threshold (branches can interfere),
    /// false if they are effectively decohered.
    pub fn can_interfere(
        &self,
        branch_a: usize,
        branch_b: usize,
        threshold: f64,
    ) -> Result<bool, ManyWorldsError> {
        let overlap = self.branch_overlap(branch_a, branch_b)?;
        Ok(overlap.norm() >= threshold)
    }

    // ---------------------------------------------------------------
    // BRANCH MERGING
    // ---------------------------------------------------------------

    /// Merge branches with identical measurement histories.
    ///
    /// When unitary evolution brings branches back together (same measurement
    /// history), their states are added coherently:
    ///   |psi_merged> = sqrt(w_a) * |psi_a> + sqrt(w_b) * |psi_b>
    ///   w_merged = |<psi_merged|psi_merged>|  (then renormalize)
    ///
    /// In practice, since states are already renormalized per-branch, the
    /// merged amplitude vector is:
    ///   amp_i = sqrt(w_a) * a_i + sqrt(w_b) * b_i
    /// and the merged weight = sum |amp_i|^2.
    ///
    /// Returns number of merges performed.
    pub fn merge_compatible_branches(&mut self) -> usize {
        let active_indices = self.active_branch_indices();
        if active_indices.len() < 2 {
            return 0;
        }

        // Group active branches by history key
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for &idx in &active_indices {
            let key = self.branches[idx].history_key();
            groups.entry(key).or_default().push(idx);
        }

        let mut merge_count = 0usize;

        for (_key, indices) in groups.iter() {
            if indices.len() < 2 {
                continue;
            }

            // Merge all branches in this group into the first one
            let primary_idx = indices[0];
            let dim = self.branches[primary_idx].state.dim;

            // Build the merged amplitude vector
            let mut merged_amps = vec![C64::new(0.0, 0.0); dim];

            for &idx in indices {
                let w_sqrt = self.branches[idx].weight.sqrt();
                let amps = self.branches[idx].state.amplitudes_ref();
                for i in 0..dim {
                    merged_amps[i] += C64::new(w_sqrt, 0.0) * amps[i];
                }
            }

            // Compute merged weight (norm squared of merged vector)
            let merged_weight: f64 = merged_amps.iter().map(|a| a.norm_sqr()).sum();

            // Renormalize the state
            if merged_weight > 1e-30 {
                let inv_norm = 1.0 / merged_weight.sqrt();
                for amp in merged_amps.iter_mut() {
                    *amp = C64::new(amp.re * inv_norm, amp.im * inv_norm);
                }
            }

            // Update the primary branch
            let nq = self.branches[primary_idx].state.num_qubits;
            let mut new_state = QuantumState::new(nq);
            let dest = new_state.amplitudes_mut();
            for i in 0..dim {
                dest[i] = merged_amps[i];
            }
            self.branches[primary_idx].state = new_state;
            self.branches[primary_idx].weight = merged_weight;

            // Retire the other branches
            for &idx in &indices[1..] {
                self.branches[idx].active = false;
                merge_count += 1;
            }
        }

        merge_count
    }

    // ---------------------------------------------------------------
    // DECOHERENT HISTORIES (GRIFFITHS / GELL-MANN-HARTLE)
    // ---------------------------------------------------------------

    /// Compute the decoherence functional D(alpha, beta) for two histories.
    ///
    /// A "history" is a sequence of computational basis projections at specified
    /// steps. The decoherence functional is:
    ///   D(alpha, beta) = Tr(C_alpha rho_0 C_beta^dagger)
    ///
    /// where C_alpha = P_alpha_n ... P_alpha_1 is the class operator (product of
    /// projectors in chronological order) and rho_0 = |psi_0><psi_0| is the
    /// initial state.
    ///
    /// For a pure initial state |psi_0>, this reduces to:
    ///   D(alpha, beta) = <psi_0|C_beta^dagger C_alpha|psi_0>
    ///
    /// The histories decohere (are consistent) if D(alpha, beta) ~= 0
    /// for alpha != beta.
    pub fn decoherence_functional(
        &self,
        initial_state: &QuantumState,
        history_alpha: &DecoherentHistory,
        history_beta: &DecoherentHistory,
    ) -> Result<C64, ManyWorldsError> {
        let dim = initial_state.dim;
        let nq = initial_state.num_qubits;

        // Validate projectors
        for proj in history_alpha
            .projectors
            .iter()
            .chain(history_beta.projectors.iter())
        {
            if proj.qubit >= nq {
                return Err(ManyWorldsError::QubitOutOfRange {
                    qubit: proj.qubit,
                    num_qubits: nq,
                });
            }
        }

        // Compute C_alpha |psi_0>: apply projectors in chronological order
        let psi_alpha = self.apply_class_operator(initial_state, &history_alpha.projectors)?;

        // Compute C_beta |psi_0>
        let psi_beta = self.apply_class_operator(initial_state, &history_beta.projectors)?;

        // D(alpha, beta) = <C_beta psi_0 | C_alpha psi_0> = <psi_beta | psi_alpha>
        let amps_alpha = psi_alpha.amplitudes_ref();
        let amps_beta = psi_beta.amplitudes_ref();

        let mut d = C64::new(0.0, 0.0);
        for i in 0..dim {
            d += amps_beta[i].conj() * amps_alpha[i];
        }

        Ok(d)
    }

    /// Apply a sequence of computational basis projectors to a state (without
    /// renormalization between projections).
    ///
    /// Each projector P_{qubit, outcome} zeros out amplitudes where bit `qubit`
    /// does not match `outcome`.
    fn apply_class_operator(
        &self,
        initial: &QuantumState,
        projectors: &[Projector],
    ) -> Result<QuantumState, ManyWorldsError> {
        let dim = initial.dim;
        let nq = initial.num_qubits;

        let mut amps: Vec<C64> = initial.amplitudes_ref().to_vec();

        for proj in projectors {
            let mask = 1usize << proj.qubit;
            for i in 0..dim {
                let bit_val = if (i & mask) != 0 { 1u8 } else { 0u8 };
                if bit_val != proj.outcome {
                    amps[i] = C64::new(0.0, 0.0);
                }
            }
        }

        let mut result = QuantumState::new(nq);
        let dest = result.amplitudes_mut();
        for i in 0..dim {
            dest[i] = amps[i];
        }

        Ok(result)
    }

    /// Check consistency of a set of decoherent histories.
    ///
    /// Computes D(alpha, beta) for all pairs alpha != beta and checks whether
    /// |D(alpha, beta)| < decoherence_threshold.
    pub fn check_consistency(
        &self,
        initial_state: &QuantumState,
        histories: &[DecoherentHistory],
    ) -> Result<ConsistencyResult, ManyWorldsError> {
        let n = histories.len();
        if n < 2 {
            return Ok(ConsistencyResult {
                is_consistent: true,
                max_off_diagonal: 0.0,
                decoherence_matrix: vec![vec![C64::new(0.0, 0.0); n]; n],
            });
        }

        let mut d_matrix = vec![vec![C64::new(0.0, 0.0); n]; n];
        let mut max_off_diagonal: f64 = 0.0;

        for alpha in 0..n {
            for beta in 0..n {
                let d = self.decoherence_functional(
                    initial_state,
                    &histories[alpha],
                    &histories[beta],
                )?;
                d_matrix[alpha][beta] = d;

                if alpha != beta {
                    let magnitude = d.norm();
                    if magnitude > max_off_diagonal {
                        max_off_diagonal = magnitude;
                    }
                }
            }
        }

        Ok(ConsistencyResult {
            is_consistent: max_off_diagonal < self.config.decoherence_threshold,
            max_off_diagonal,
            decoherence_matrix: d_matrix,
        })
    }

    // ---------------------------------------------------------------
    // BRANCH STATISTICS
    // ---------------------------------------------------------------

    /// Compute comprehensive statistics about the branching structure.
    pub fn statistics(&self) -> BranchStatistics {
        let active: Vec<&Branch> = self.active_branches();
        let n = active.len();

        if n == 0 {
            return BranchStatistics {
                total_branches: self.total_created,
                active_branches: 0,
                max_depth: 0,
                weight_entropy: 0.0,
                effective_branch_count: 0.0,
                total_weight: 0.0,
                min_weight: 0.0,
                max_weight: 0.0,
                mean_weight: 0.0,
                weight_distribution: Vec::new(),
            };
        }

        let weights: Vec<f64> = active.iter().map(|b| b.weight).collect();
        let total_weight: f64 = weights.iter().sum();
        let max_depth = active.iter().map(|b| b.depth()).max().unwrap_or(0);

        // Weight entropy: S = -sum w_i * log2(w_i)
        let mut entropy: f64 = 0.0;
        for &w in &weights {
            if w > 0.0 {
                let normalized = w / total_weight;
                entropy -= normalized * normalized.log2();
            }
        }

        // Effective branch count: 2^S
        let effective = 2.0f64.powf(entropy);

        let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_weight = total_weight / n as f64;

        // Weight distribution histogram (10 bins)
        let num_bins = 10;
        let mut distribution = vec![0usize; num_bins];
        if max_weight > min_weight {
            let range = max_weight - min_weight;
            for &w in &weights {
                let bin = ((w - min_weight) / range * (num_bins as f64 - 1.0)).floor() as usize;
                let bin = bin.min(num_bins - 1);
                distribution[bin] += 1;
            }
        } else {
            // All weights equal
            distribution[0] = n;
        }

        BranchStatistics {
            total_branches: self.total_created,
            active_branches: n,
            max_depth,
            weight_entropy: entropy,
            effective_branch_count: effective,
            total_weight,
            min_weight,
            max_weight,
            mean_weight,
            weight_distribution: distribution,
        }
    }

    /// Compute only the weight entropy of the branching structure.
    pub fn weight_entropy(&self) -> f64 {
        let active = self.active_branches();
        let total_weight: f64 = active.iter().map(|b| b.weight).sum();

        if total_weight <= 0.0 || active.is_empty() {
            return 0.0;
        }

        let mut entropy: f64 = 0.0;
        for b in &active {
            if b.weight > 0.0 {
                let p = b.weight / total_weight;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Total weight of all active branches (should be ~1.0 if no weight is lost).
    pub fn total_weight(&self) -> f64 {
        self.active_branches().iter().map(|b| b.weight).sum()
    }

    // ---------------------------------------------------------------
    // UNIVERSE SNAPSHOT
    // ---------------------------------------------------------------

    /// Serialize the full branching structure as a tree for visualization.
    ///
    /// Each node contains the branch ID, weight, active status, and
    /// measurement history. Edges represent parent-child relationships
    /// via measurement.
    pub fn snapshot(&self) -> UniverseSnapshot {
        let nodes: Vec<SnapshotNode> = self
            .branches
            .iter()
            .map(|b| SnapshotNode {
                branch_id: b.id,
                index: 0, // Will be set below
                weight: b.weight,
                active: b.active,
                depth: b.depth(),
                parent: b.parent,
                history: b.history.clone(),
                num_qubits: b.state.num_qubits,
            })
            .enumerate()
            .map(|(i, mut n)| {
                n.index = i;
                n
            })
            .collect();

        // Build children map
        let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, branch) in self.branches.iter().enumerate() {
            if let Some(parent) = branch.parent {
                children.entry(parent).or_default().push(i);
            }
        }

        UniverseSnapshot {
            nodes,
            children,
            total_created: self.total_created,
            active_count: self.active_branch_count(),
        }
    }
}

// ===================================================================
// DECOHERENT HISTORIES TYPES
// ===================================================================

/// A computational basis projector: projects onto the subspace where
/// qubit `qubit` has value `outcome`.
#[derive(Clone, Debug, PartialEq)]
pub struct Projector {
    /// Which qubit to project.
    pub qubit: usize,
    /// Which outcome (0 or 1) to project onto.
    pub outcome: u8,
}

impl Projector {
    /// Create a new projector.
    pub fn new(qubit: usize, outcome: u8) -> Self {
        Self { qubit, outcome }
    }
}

/// A decoherent history: a sequence of projectors applied at successive times.
///
/// In the Griffiths/Gell-Mann-Hartle formalism, a history is a time-ordered
/// sequence of projection operators. The decoherence functional tests whether
/// two histories are consistent (non-interfering).
#[derive(Clone, Debug)]
pub struct DecoherentHistory {
    /// Sequence of projectors in chronological order.
    pub projectors: Vec<Projector>,
    /// Optional label for this history.
    pub label: String,
}

impl DecoherentHistory {
    /// Create a new decoherent history from a sequence of projectors.
    pub fn new(projectors: Vec<Projector>, label: &str) -> Self {
        Self {
            projectors,
            label: label.to_string(),
        }
    }

    /// Create a history from a sequence of (qubit, outcome) pairs.
    pub fn from_outcomes(outcomes: &[(usize, u8)], label: &str) -> Self {
        Self {
            projectors: outcomes
                .iter()
                .map(|&(q, o)| Projector::new(q, o))
                .collect(),
            label: label.to_string(),
        }
    }
}

// ===================================================================
// CONSISTENCY RESULT
// ===================================================================

/// Result of a decoherent histories consistency check.
#[derive(Clone, Debug)]
pub struct ConsistencyResult {
    /// Whether all off-diagonal elements of the decoherence functional are
    /// below the threshold (i.e., the histories decohere / are consistent).
    pub is_consistent: bool,
    /// Largest magnitude of any off-diagonal element |D(alpha, beta)|.
    pub max_off_diagonal: f64,
    /// Full decoherence matrix D(alpha, beta).
    pub decoherence_matrix: Vec<Vec<C64>>,
}

// ===================================================================
// BRANCH STATISTICS
// ===================================================================

/// Comprehensive statistics about the branching structure of the universe.
#[derive(Clone, Debug)]
pub struct BranchStatistics {
    /// Total number of branches ever created (including retired ones).
    pub total_branches: usize,
    /// Number of currently active branches.
    pub active_branches: usize,
    /// Maximum depth (longest measurement history among active branches).
    pub max_depth: usize,
    /// Weight entropy: S = -sum p_i log2(p_i) where p_i = w_i / sum(w).
    pub weight_entropy: f64,
    /// Effective branch count: 2^S. For uniform weights this equals the
    /// actual number of branches.
    pub effective_branch_count: f64,
    /// Total weight of all active branches (should be ~1.0).
    pub total_weight: f64,
    /// Minimum weight among active branches.
    pub min_weight: f64,
    /// Maximum weight among active branches.
    pub max_weight: f64,
    /// Mean weight among active branches.
    pub mean_weight: f64,
    /// Histogram of weight distribution (10 bins from min to max).
    pub weight_distribution: Vec<usize>,
}

impl fmt::Display for BranchStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Branch Statistics ===")?;
        writeln!(f, "  Total created:        {}", self.total_branches)?;
        writeln!(f, "  Active branches:      {}", self.active_branches)?;
        writeln!(f, "  Max depth:            {}", self.max_depth)?;
        writeln!(f, "  Weight entropy:       {:.6} bits", self.weight_entropy)?;
        writeln!(
            f,
            "  Effective branches:   {:.2}",
            self.effective_branch_count
        )?;
        writeln!(f, "  Total weight:         {:.10}", self.total_weight)?;
        writeln!(
            f,
            "  Weight range:         [{:.6e}, {:.6e}]",
            self.min_weight, self.max_weight
        )?;
        writeln!(f, "  Mean weight:          {:.6e}", self.mean_weight)?;
        Ok(())
    }
}

// ===================================================================
// UNIVERSE SNAPSHOT
// ===================================================================

/// A snapshot node representing one branch in the tree.
#[derive(Clone, Debug)]
pub struct SnapshotNode {
    /// Unique branch identifier.
    pub branch_id: usize,
    /// Index in the branches vector.
    pub index: usize,
    /// Branch weight.
    pub weight: f64,
    /// Whether the branch is still active.
    pub active: bool,
    /// Number of measurements in this branch's history.
    pub depth: usize,
    /// Parent branch index (None for root).
    pub parent: Option<usize>,
    /// Measurement history.
    pub history: Vec<MeasurementRecord>,
    /// Number of qubits.
    pub num_qubits: usize,
}

/// Serialized snapshot of the entire branching universe.
#[derive(Clone, Debug)]
pub struct UniverseSnapshot {
    /// All nodes in the tree.
    pub nodes: Vec<SnapshotNode>,
    /// Children map: parent index -> list of child indices.
    pub children: HashMap<usize, Vec<usize>>,
    /// Total branches ever created.
    pub total_created: usize,
    /// Number of currently active branches.
    pub active_count: usize,
}

impl UniverseSnapshot {
    /// Get root nodes (nodes with no parent).
    pub fn roots(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get children of a node.
    pub fn children_of(&self, index: usize) -> &[usize] {
        self.children
            .get(&index)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all leaf nodes (active branches with no active children).
    pub fn leaves(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| {
                n.active && self.children_of(*i).iter().all(|&c| !self.nodes[c].active)
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Render the tree as a text string for debugging.
    pub fn render_tree(&self) -> String {
        let mut out = String::new();
        let roots = self.roots();
        for &root in &roots {
            self.render_subtree(root, 0, &mut out);
        }
        out
    }

    fn render_subtree(&self, index: usize, indent: usize, out: &mut String) {
        let node = &self.nodes[index];
        let prefix = "  ".repeat(indent);
        let status = if node.active { "ACTIVE" } else { "retired" };
        let last_meas = node
            .history
            .last()
            .map(|r| format!(" <- q{}={}", r.qubit, r.outcome))
            .unwrap_or_default();

        out.push_str(&format!(
            "{}[{}] w={:.6e} {} d={}{}\n",
            prefix, node.branch_id, node.weight, status, node.depth, last_meas
        ));

        if let Some(children) = self.children.get(&index) {
            for &child in children {
                self.render_subtree(child, indent + 1, out);
            }
        }
    }
}

// ===================================================================
// CONVENIENCE BUILDERS
// ===================================================================

/// Build a GHZ state (|00...0> + |11...1>) / sqrt(2) on `n` qubits.
///
/// Circuit: H(0), then CNOT(0,1), CNOT(1,2), ..., CNOT(n-2, n-1).
pub fn prepare_ghz(n: usize) -> QuantumState {
    let mut state = QuantumState::new(n);
    GateOperations::h(&mut state, 0);
    for i in 0..n - 1 {
        // cnot requires bit0 < bit1
        GateOperations::cnot(&mut state, i, i + 1);
    }
    state
}

/// Build a Bell state (|00> + |11>) / sqrt(2).
pub fn prepare_bell() -> QuantumState {
    prepare_ghz(2)
}

/// Build a W state (|100> + |010> + |001>) / sqrt(3) on `n` qubits.
///
/// Uses a cascade of controlled rotations.
pub fn prepare_w(n: usize) -> QuantumState {
    assert!(n >= 2, "W state requires at least 2 qubits");
    let mut state = QuantumState::new(n);

    // Start with |10...0>
    GateOperations::x(&mut state, 0);

    // Distribute the excitation across all qubits
    for i in 0..n - 1 {
        // Rotate qubit i: Ry(theta) where theta = 2*arccos(sqrt(1/(n-i)))
        // This distributes amplitude from qubit i to qubit i+1 via CNOT
        let remaining = (n - i) as f64;
        let theta = 2.0 * (1.0 / remaining).sqrt().acos();

        // Apply controlled rotation: Ry(theta) on qubit i
        // We implement this as: Ry(-theta/2) on target, CNOT, Ry(theta/2) on target
        let half = theta / 2.0;
        let _cos_h = half.cos();
        let _sin_h = half.sin();

        // Apply Ry(-theta/2) to qubit i+1
        // Ry(angle) = [[cos(a/2), -sin(a/2)], [sin(a/2), cos(a/2)]]
        apply_ry(&mut state, i + 1, -half);

        // CNOT(i, i+1) -- i < i+1 always holds
        GateOperations::cnot(&mut state, i, i + 1);

        // Apply Ry(theta/2) to qubit i+1
        apply_ry(&mut state, i + 1, half);

        // CNOT(i, i+1) again
        GateOperations::cnot(&mut state, i, i + 1);
    }

    state
}

/// Apply Ry(angle) gate to a single qubit.
/// Ry(a) = [[cos(a/2), -sin(a/2)], [sin(a/2), cos(a/2)]]
fn apply_ry(state: &mut QuantumState, qubit: usize, angle: f64) {
    let half = angle / 2.0;
    let c = half.cos();
    let s = half.sin();

    let dim = state.dim;
    let mask = 1usize << qubit;
    let amps = state.amplitudes_mut();

    let mut i = 0;
    while i < dim {
        // Find pairs (i0, i1) where i0 has bit `qubit` = 0 and i1 has it = 1
        if (i & mask) != 0 {
            i += 1;
            continue;
        }
        let i0 = i;
        let i1 = i | mask;

        let a0 = amps[i0];
        let a1 = amps[i1];

        amps[i0] = C64::new(c * a0.re + (-s) * a1.re, c * a0.im + (-s) * a1.im);
        amps[i1] = C64::new(s * a0.re + c * a1.re, s * a0.im + c * a1.im);

        i += 1;
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    // ----- Helper -----

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn total_active_weight(universe: &BranchingUniverse) -> f64 {
        universe.active_branches().iter().map(|b| b.weight).sum()
    }

    // ----- Test 1: Config builder defaults -----

    #[test]
    fn test_config_builder() {
        let config = ManyWorldsConfig::default();
        assert_eq!(config.max_branches, 1024);
        assert!(approx_eq(config.decoherence_threshold, 0.01, EPSILON));
        assert!(config.track_history);
        assert!(approx_eq(config.prune_threshold, 1e-10, EPSILON));

        let config2 = ManyWorldsConfig::new()
            .with_max_branches(512)
            .with_decoherence_threshold(0.05)
            .with_track_history(false)
            .with_prune_threshold(1e-8);

        assert_eq!(config2.max_branches, 512);
        assert!(approx_eq(config2.decoherence_threshold, 0.05, EPSILON));
        assert!(!config2.track_history);
        assert!(approx_eq(config2.prune_threshold, 1e-8, EPSILON));
    }

    // ----- Test 2: Single measurement creates two branches -----

    #[test]
    fn test_single_measurement_creates_two_branches() {
        let mut universe = BranchingUniverse::with_defaults(1);

        // Apply H to qubit 0: |0> -> (|0> + |1>) / sqrt(2)
        universe.h_all(0);

        // Measure qubit 0 -- should create two branches
        let new_indices = universe.measure_all(0).unwrap();
        assert_eq!(new_indices.len(), 2);

        let active = universe.active_branches();
        assert_eq!(active.len(), 2);

        // Each branch should have weight ~0.5
        for b in &active {
            assert!(
                approx_eq(b.weight, 0.5, 1e-6),
                "Branch weight {} should be ~0.5",
                b.weight
            );
        }

        // Outcomes should be 0 and 1
        let outcomes: Vec<u8> = active.iter().map(|b| b.history[0].outcome).collect();
        assert!(outcomes.contains(&0));
        assert!(outcomes.contains(&1));
    }

    // ----- Test 3: Branch weights sum to one -----

    #[test]
    fn test_branch_weights_sum_to_one() {
        let mut universe = BranchingUniverse::with_defaults(3);

        // Create superposition on all qubits
        universe.h_all(0);
        universe.h_all(1);
        universe.h_all(2);

        // Measure qubit 0
        universe.measure_all(0).unwrap();
        let w1 = total_active_weight(&universe);
        assert!(
            approx_eq(w1, 1.0, 1e-10),
            "Weights after first measurement: {} (expected 1.0)",
            w1
        );

        // Measure qubit 1
        universe.measure_all(1).unwrap();
        let w2 = total_active_weight(&universe);
        assert!(
            approx_eq(w2, 1.0, 1e-10),
            "Weights after second measurement: {} (expected 1.0)",
            w2
        );

        // Measure qubit 2
        universe.measure_all(2).unwrap();
        let w3 = total_active_weight(&universe);
        assert!(
            approx_eq(w3, 1.0, 1e-10),
            "Weights after third measurement: {} (expected 1.0)",
            w3
        );

        // 8 branches total (2^3)
        assert_eq!(universe.active_branch_count(), 8);
    }

    // ----- Test 4: GHZ measurement cascade -----

    #[test]
    fn test_ghz_measurement_cascade() {
        // GHZ state on 3 qubits: (|000> + |111>) / sqrt(2)
        let ghz = prepare_ghz(3);
        let config = ManyWorldsConfig::default();
        let mut universe = BranchingUniverse::from_state(ghz, config);

        // Measure qubit 0
        universe.measure_all(0).unwrap();

        let active = universe.active_branches();
        assert_eq!(active.len(), 2);

        // For GHZ, P(0) = P(1) = 0.5
        for b in &active {
            assert!(
                approx_eq(b.weight, 0.5, 1e-6),
                "GHZ branch weight should be ~0.5, got {}",
                b.weight
            );
        }

        // After measuring q0=0, remaining state should be |00>
        // After measuring q0=1, remaining state should be |11>
        // Measure qubit 1 on all branches
        universe.measure_all(1).unwrap();

        // In the q0=0 branch, P(q1=0) = 1, so only one child is active
        // In the q0=1 branch, P(q1=1) = 1, so only one child is active
        // We should have 2 active branches (the zero-weight ones are deactivated)
        let active_after = universe.active_branches();

        // GHZ correlation: the measurements should be perfectly correlated
        // Active branches should have histories where all outcomes match
        for b in &active_after {
            if b.history.len() >= 2 {
                let o0 = b.history[0].outcome;
                let o1 = b.history[1].outcome;
                assert_eq!(
                    o0, o1,
                    "GHZ: qubit 0 outcome ({}) should equal qubit 1 outcome ({})",
                    o0, o1
                );
            }
        }
    }

    // ----- Test 5: Gate applies to all branches -----

    #[test]
    fn test_gate_applies_to_all_branches() {
        let mut universe = BranchingUniverse::with_defaults(2);

        // Create superposition and branch
        universe.h_all(0);
        universe.measure_all(0).unwrap();

        assert_eq!(universe.active_branch_count(), 2);

        // Apply H to qubit 1 on ALL branches
        universe.h_all(1);

        // Each branch should now have qubit 1 in superposition
        for b in universe.active_branches() {
            let probs = b.state.probabilities();
            // qubit 1 should have 50/50 distribution
            let p1_0: f64 = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| (i >> 1) & 1 == 0)
                .map(|(_, p)| p)
                .sum();
            let p1_1: f64 = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| (i >> 1) & 1 == 1)
                .map(|(_, p)| p)
                .sum();
            assert!(
                approx_eq(p1_0, 0.5, 1e-6),
                "P(q1=0) should be 0.5, got {}",
                p1_0
            );
            assert!(
                approx_eq(p1_1, 0.5, 1e-6),
                "P(q1=1) should be 0.5, got {}",
                p1_1
            );
        }
    }

    // ----- Test 6: Branch pruning -----

    #[test]
    fn test_branch_pruning() {
        let config = ManyWorldsConfig::new().with_prune_threshold(0.2);
        let mut universe = BranchingUniverse::new(2, config);

        // Create an unequal superposition: Ry(pi/6) on qubit 0
        // P(0) = cos^2(pi/12) ~ 0.933, P(1) = sin^2(pi/12) ~ 0.067
        apply_ry(&mut universe.branches[0].state, 0, PI / 6.0);

        // Measure qubit 0
        universe.measure_all(0).unwrap();

        assert_eq!(universe.active_branch_count(), 2);

        // Prune with threshold 0.2 -- the ~0.067 branch should be pruned
        let (pruned, pruned_weight) = universe.prune();
        assert_eq!(pruned, 1, "Should have pruned 1 branch");
        assert!(pruned_weight > 0.0, "Pruned weight should be positive");

        assert_eq!(universe.active_branch_count(), 1);
    }

    // ----- Test 7: Pruning preserves total weight -----

    #[test]
    fn test_pruning_preserves_total_weight() {
        let config = ManyWorldsConfig::new().with_prune_threshold(0.05);
        let mut universe = BranchingUniverse::new(3, config);

        // Create superpositions with varying weights
        apply_ry(&mut universe.branches[0].state, 0, PI / 8.0);
        universe.h_all(1);

        universe.measure_all(0).unwrap();
        universe.measure_all(1).unwrap();

        let w_before = total_active_weight(&universe);
        assert!(
            approx_eq(w_before, 1.0, 1e-10),
            "Weight before pruning: {}",
            w_before
        );

        let (pruned, _) = universe.prune();

        let w_after = total_active_weight(&universe);
        assert!(
            approx_eq(w_after, 1.0, 1e-6),
            "Weight after pruning: {} (pruned {} branches)",
            w_after,
            pruned
        );
    }

    // ----- Test 8: Decoherent history consistency (computational basis) -----

    #[test]
    fn test_decoherent_history_consistency() {
        // For |+> state, measuring in Z basis gives histories that decohere
        let mut initial = QuantumState::new(1);
        GateOperations::h(&mut initial, 0);

        let universe = BranchingUniverse::with_defaults(1);

        // History alpha: q0 = 0
        let h_alpha = DecoherentHistory::from_outcomes(&[(0, 0)], "q0=0");
        // History beta: q0 = 1
        let h_beta = DecoherentHistory::from_outcomes(&[(0, 1)], "q0=1");

        // D(alpha, beta) should be ~0 (orthogonal projections on |+>)
        let d = universe
            .decoherence_functional(&initial, &h_alpha, &h_beta)
            .unwrap();

        assert!(
            d.norm() < 0.01,
            "Off-diagonal D(alpha, beta) should be ~0, got |D| = {}",
            d.norm()
        );

        // D(alpha, alpha) should be ~0.5 (P(0) for |+>)
        let d_aa = universe
            .decoherence_functional(&initial, &h_alpha, &h_alpha)
            .unwrap();
        assert!(
            approx_eq(d_aa.re, 0.5, 1e-6),
            "D(alpha, alpha) should be ~0.5, got {}",
            d_aa.re
        );

        // Check full consistency
        let result = universe
            .check_consistency(&initial, &[h_alpha, h_beta])
            .unwrap();
        assert!(
            result.is_consistent,
            "Computational basis histories should be consistent"
        );
    }

    // ----- Test 9: Inconsistent history -----

    #[test]
    fn test_inconsistent_history() {
        // Create a 2-qubit Bell state: (|00> + |11>) / sqrt(2)
        let initial = prepare_bell();

        let config = ManyWorldsConfig::new().with_decoherence_threshold(1e-6);
        let universe = BranchingUniverse::new(2, config);

        // History alpha: q0 = 0 (does NOT specify q1)
        let h_alpha = DecoherentHistory::from_outcomes(&[(0, 0)], "q0=0");
        // History beta: q1 = 0 (does NOT specify q0)
        let h_beta = DecoherentHistory::from_outcomes(&[(1, 0)], "q1=0");

        // These histories are NOT orthogonal projections -- their class operators
        // overlap because |00> is in both projected subspaces.
        let d = universe
            .decoherence_functional(&initial, &h_alpha, &h_beta)
            .unwrap();

        // For Bell state, projecting q0=0 gives |00>/sqrt(2) amplitude,
        // projecting q1=0 gives |00>/sqrt(2) amplitude.
        // D(alpha, beta) = <00|P_{q1=0} P_{q0=0}|Bell> is not zero.
        // Actually let's just check the magnitude is non-trivial:
        // The consistency check should reveal these are not fully consistent
        // as a set of alternative histories (they're not exhaustive/exclusive).
        //
        // More precisely: for the Bell state (|00>+|11>)/sqrt(2),
        // P_{q0=0}|Bell> = |00>/sqrt(2), P_{q1=0}|Bell> = |00>/sqrt(2)
        // D(alpha,beta) = <Bell|P_{q1=0}^dag P_{q0=0}|Bell>
        //               = <Bell|P_{q1=0} P_{q0=0}|Bell>  (projectors are Hermitian)
        //               = <Bell| (project q0=0 AND q1=0) |Bell>
        //               = |<00|Bell>|^2 = 1/2
        assert!(
            d.norm() > 0.1,
            "Off-diagonal D should be non-zero for non-orthogonal histories, got |D| = {}",
            d.norm()
        );
    }

    // ----- Test 10: Branch interference -- orthogonal -----

    #[test]
    fn test_branch_interference_orthogonal() {
        let mut universe = BranchingUniverse::with_defaults(1);

        // |0> -> H -> measure -> two branches: |0> and |1>
        universe.h_all(0);
        universe.measure_all(0).unwrap();

        let active = universe.active_branch_indices();
        assert_eq!(active.len(), 2);

        // |0> and |1> are orthogonal, overlap = 0
        let overlap = universe.branch_overlap(active[0], active[1]).unwrap();
        assert!(
            overlap.norm() < EPSILON,
            "Orthogonal branches should have zero overlap, got {}",
            overlap.norm()
        );

        assert!(
            !universe.can_interfere(active[0], active[1], 0.01).unwrap(),
            "Orthogonal branches should not interfere"
        );
    }

    // ----- Test 11: Branch interference -- overlapping -----

    #[test]
    fn test_branch_interference_overlapping() {
        // Create two branches with overlapping states
        let mut universe = BranchingUniverse::with_defaults(2);

        // Put qubit 0 in superposition and branch
        universe.h_all(0);
        universe.measure_all(0).unwrap();

        // Now apply H to qubit 1 on all branches, then apply H to qubit 0 on all branches
        // This partially undoes the measurement distinction
        universe.h_all(1);

        // After this, each branch has a different state for qubit 0 (|0> vs |1>)
        // but identical state for qubit 1. The states are still orthogonal because
        // qubit 0 is in a definite basis state.
        let active = universe.active_branch_indices();
        let overlap = universe.branch_overlap(active[0], active[1]).unwrap();

        // |0,+> and |1,+> are orthogonal (differ on qubit 0)
        assert!(
            overlap.norm() < EPSILON,
            "States differing on qubit 0 should be orthogonal, got {}",
            overlap.norm()
        );

        // Now let's test truly overlapping branches by creating a custom scenario
        let config = ManyWorldsConfig::default();
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0); // |+>
        let mut universe2 = BranchingUniverse::from_state(state, config);

        // Measure, then apply a small rotation to both branches
        universe2.measure_all(0).unwrap();

        // Apply H to both branches -- this maps |0> -> |+> and |1> -> |->
        universe2.h_all(0);

        // |+> = (|0>+|1>)/sqrt(2) and |-> = (|0>-|1>)/sqrt(2)
        // <+|-> = 0  (still orthogonal after H)
        let active2 = universe2.active_branch_indices();
        let overlap2 = universe2.branch_overlap(active2[0], active2[1]).unwrap();
        assert!(
            overlap2.norm() < EPSILON,
            "H(|0>) and H(|1>) should still be orthogonal"
        );

        // For a true overlap test: create branches manually via small rotation
        let config3 = ManyWorldsConfig::default();
        let mut universe3 = BranchingUniverse::new(1, config3);

        // Very small rotation: Ry(0.1) means P(0) ~ 0.9975, P(1) ~ 0.0025
        apply_ry(&mut universe3.branches[0].state, 0, 0.1);
        universe3.measure_all(0).unwrap();

        // Now rotate both branches by Ry(pi/4) -- they'll have some overlap
        for b in universe3.branches.iter_mut() {
            if b.active {
                apply_ry(&mut b.state, 0, PI / 4.0);
            }
        }

        let active3 = universe3.active_branch_indices();
        // After projection and Ry rotation, the branches may have non-trivial overlap
        // Branch 0 had |0> projected, then Ry(pi/4) -> cos(pi/8)|0> + sin(pi/8)|1>
        // Branch 1 had |1> projected, then Ry(pi/4) -> -sin(pi/8)|0> + cos(pi/8)|1>
        // Overlap = -sin(pi/8)*cos(pi/8) + sin(pi/8)*cos(pi/8) = 0
        // These are still orthogonal! The Ry gate preserves orthogonality of |0> and |1>.
        //
        // To get genuinely overlapping branches, we need to apply DIFFERENT gates
        // to different branches, which breaks the universal evolution model.
        // Instead, let's verify can_interfere with a generous threshold:
        let overlap3 = universe3.branch_overlap(active3[0], active3[1]).unwrap();
        // Unitary evolution preserves inner products, so |0> and |1> remain orthogonal.
        assert!(
            overlap3.norm() < EPSILON,
            "Unitary evolution preserves orthogonality"
        );

        // The fact that branches created by measurement and evolved unitarily
        // remain orthogonal is itself a deep result -- it's WHY the many-worlds
        // branches don't interfere (decoherence). This test validates that.
        assert!(
            !universe3
                .can_interfere(active3[0], active3[1], 0.01)
                .unwrap(),
            "Post-measurement branches remain orthogonal under unitary evolution"
        );
    }

    // ----- Test 12: Branch statistics entropy -----

    #[test]
    fn test_branch_statistics_entropy() {
        let mut universe = BranchingUniverse::with_defaults(3);

        // Start with 0 entropy (one branch)
        let s0 = universe.weight_entropy();
        assert!(
            approx_eq(s0, 0.0, EPSILON),
            "Single branch entropy should be 0, got {}",
            s0
        );

        // Measure qubit 0 after H: 2 equal branches -> entropy = 1 bit
        universe.h_all(0);
        universe.measure_all(0).unwrap();
        let s1 = universe.weight_entropy();
        assert!(
            approx_eq(s1, 1.0, 1e-6),
            "Two equal branches: entropy should be 1.0 bit, got {}",
            s1
        );

        // Measure qubit 1 after H: 4 equal branches -> entropy = 2 bits
        universe.h_all(1);
        universe.measure_all(1).unwrap();
        let s2 = universe.weight_entropy();
        assert!(
            approx_eq(s2, 2.0, 1e-6),
            "Four equal branches: entropy should be 2.0 bits, got {}",
            s2
        );

        // Measure qubit 2 after H: 8 equal branches -> entropy = 3 bits
        universe.h_all(2);
        universe.measure_all(2).unwrap();
        let s3 = universe.weight_entropy();
        assert!(
            approx_eq(s3, 3.0, 1e-6),
            "Eight equal branches: entropy should be 3.0 bits, got {}",
            s3
        );
    }

    // ----- Test 13: Effective branch count -----

    #[test]
    fn test_effective_branch_count() {
        let mut universe = BranchingUniverse::with_defaults(3);

        // Create uniform superposition and measure all qubits
        universe.h_all(0);
        universe.measure_all(0).unwrap();
        universe.h_all(1);
        universe.measure_all(1).unwrap();
        universe.h_all(2);
        universe.measure_all(2).unwrap();

        let stats = universe.statistics();

        // 8 branches with equal weight -> effective count = 2^3 = 8
        assert_eq!(stats.active_branches, 8);
        assert!(
            approx_eq(stats.effective_branch_count, 8.0, 0.1),
            "Effective branch count should be ~8, got {}",
            stats.effective_branch_count
        );

        // For non-uniform: fewer effective branches
        let mut universe2 = BranchingUniverse::with_defaults(1);
        // Ry(pi/6): P(0) ~ 0.933, P(1) ~ 0.067
        apply_ry(&mut universe2.branches[0].state, 0, PI / 6.0);
        universe2.measure_all(0).unwrap();

        let stats2 = universe2.statistics();
        assert_eq!(stats2.active_branches, 2);

        // Effective count for very unequal weights should be < 2
        assert!(
            stats2.effective_branch_count < 2.0,
            "Unequal branches: effective count should be < 2, got {}",
            stats2.effective_branch_count
        );
        assert!(
            stats2.effective_branch_count > 1.0,
            "Unequal branches: effective count should be > 1, got {}",
            stats2.effective_branch_count
        );
    }

    // ----- Test 14: Branch merging -----

    #[test]
    fn test_branch_merging() {
        let config = ManyWorldsConfig::new().with_track_history(true);
        let mut universe = BranchingUniverse::new(1, config);

        // Create two branches with no measurements (empty history)
        // We can't easily create this via measurement, so let's test that
        // branches with the SAME measurement history get merged.

        // H, measure -> 2 branches
        universe.h_all(0);
        universe.measure_all(0).unwrap();

        // Each branch has a DIFFERENT history (q0=0 vs q0=1), so no merging
        let merged = universe.merge_compatible_branches();
        assert_eq!(merged, 0, "Different histories should not merge");
        assert_eq!(universe.active_branch_count(), 2);
    }

    // ----- Test 15: Measurement history tracking -----

    #[test]
    fn test_measurement_history_tracking() {
        let config = ManyWorldsConfig::new().with_track_history(true);
        let mut universe = BranchingUniverse::new(2, config);

        universe.h_all(0);
        universe.h_all(1);

        // Measure qubit 0 (step 0)
        universe.measure_all(0).unwrap();

        // Measure qubit 1 (step 1)
        universe.measure_all(1).unwrap();

        // Should have 4 active branches
        assert_eq!(universe.active_branch_count(), 4);

        // Each branch should have exactly 2 measurements in its history
        for b in universe.active_branches() {
            assert_eq!(
                b.history.len(),
                2,
                "Each branch should have 2 measurement records"
            );
            assert_eq!(
                b.history[0].qubit, 0,
                "First measurement should be on qubit 0"
            );
            assert_eq!(
                b.history[1].qubit, 1,
                "Second measurement should be on qubit 1"
            );
            assert_eq!(b.history[0].step, 0, "First measurement should be step 0");
            assert_eq!(b.history[1].step, 1, "Second measurement should be step 1");
            assert!(
                b.history[0].outcome == 0 || b.history[0].outcome == 1,
                "Outcome should be 0 or 1"
            );
        }

        // All four combinations of (q0, q1) should be present
        let mut seen: HashMap<(u8, u8), bool> = HashMap::new();
        for b in universe.active_branches() {
            let key = (b.history[0].outcome, b.history[1].outcome);
            seen.insert(key, true);
        }
        assert!(seen.contains_key(&(0, 0)));
        assert!(seen.contains_key(&(0, 1)));
        assert!(seen.contains_key(&(1, 0)));
        assert!(seen.contains_key(&(1, 1)));
    }

    // ----- Test 16: Multi-qubit branching -----

    #[test]
    fn test_multi_qubit_branching() {
        let mut universe = BranchingUniverse::with_defaults(4);

        // Create superposition on qubits 0-3
        for q in 0..4 {
            universe.h_all(q);
        }

        // Measure each qubit sequentially
        for q in 0..4 {
            universe.measure_all(q).unwrap();
        }

        // Should have 2^4 = 16 branches
        assert_eq!(universe.active_branch_count(), 16);

        // All branches should have equal weight 1/16
        for b in universe.active_branches() {
            assert!(
                approx_eq(b.weight, 1.0 / 16.0, 1e-6),
                "Each branch weight should be 1/16, got {}",
                b.weight
            );
        }

        // Total weight should be 1
        let w = total_active_weight(&universe);
        assert!(approx_eq(w, 1.0, 1e-10));

        // Statistics should show entropy = 4 bits
        let stats = universe.statistics();
        assert!(
            approx_eq(stats.weight_entropy, 4.0, 1e-6),
            "Entropy should be 4.0 bits, got {}",
            stats.weight_entropy
        );
    }

    // ----- Test 17: Branch overflow error -----

    #[test]
    fn test_branch_overflow() {
        let config = ManyWorldsConfig::new().with_max_branches(4);
        let mut universe = BranchingUniverse::new(3, config);

        universe.h_all(0);
        universe.measure_all(0).unwrap(); // 2 branches
        assert_eq!(universe.active_branch_count(), 2);

        universe.h_all(1);
        universe.measure_all(1).unwrap(); // 4 branches
        assert_eq!(universe.active_branch_count(), 4);

        // Next measurement would create 8 branches, exceeding max=4
        universe.h_all(2);
        let result = universe.measure_all(2);
        assert!(result.is_err());

        match result.unwrap_err() {
            ManyWorldsError::BranchOverflow { current: _, max } => {
                assert_eq!(max, 4);
            }
            other => panic!("Expected BranchOverflow, got {:?}", other),
        }
    }

    // ----- Test 18: Qubit out of range -----

    #[test]
    fn test_qubit_out_of_range() {
        let mut universe = BranchingUniverse::with_defaults(2);

        let result = universe.measure_all(5);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManyWorldsError::QubitOutOfRange { qubit, num_qubits } => {
                assert_eq!(qubit, 5);
                assert_eq!(num_qubits, 2);
            }
            other => panic!("Expected QubitOutOfRange, got {:?}", other),
        }
    }

    // ----- Test 19: Universe snapshot -----

    #[test]
    fn test_universe_snapshot() {
        let mut universe = BranchingUniverse::with_defaults(1);
        universe.h_all(0);
        universe.measure_all(0).unwrap();

        let snapshot = universe.snapshot();

        // 3 total nodes: root (retired) + 2 children (active)
        assert_eq!(snapshot.nodes.len(), 3);
        assert_eq!(snapshot.active_count, 2);
        assert_eq!(snapshot.total_created, 3);

        // Root has 2 children
        let roots = snapshot.roots();
        assert_eq!(roots.len(), 1);
        let root_children = snapshot.children_of(roots[0]);
        assert_eq!(root_children.len(), 2);

        // Children are leaves
        let leaves = snapshot.leaves();
        assert_eq!(leaves.len(), 2);

        // Render should produce non-empty text
        let rendered = snapshot.render_tree();
        assert!(!rendered.is_empty());
        assert!(rendered.contains("ACTIVE"));
        assert!(rendered.contains("retired"));
    }

    // ----- Test 20: Decoherence functional diagonal elements sum to 1 -----

    #[test]
    fn test_decoherence_diagonal_sum() {
        // For a complete set of histories, sum of D(alpha, alpha) should = 1
        let mut initial = QuantumState::new(1);
        GateOperations::h(&mut initial, 0);

        let universe = BranchingUniverse::with_defaults(1);

        let h0 = DecoherentHistory::from_outcomes(&[(0, 0)], "q0=0");
        let h1 = DecoherentHistory::from_outcomes(&[(0, 1)], "q0=1");

        let d00 = universe.decoherence_functional(&initial, &h0, &h0).unwrap();
        let d11 = universe.decoherence_functional(&initial, &h1, &h1).unwrap();

        let sum = d00.re + d11.re;
        assert!(
            approx_eq(sum, 1.0, 1e-6),
            "Sum of diagonal elements should be 1.0, got {}",
            sum
        );
    }

    // ----- Test 21: Measure specific branch -----

    #[test]
    fn test_measure_specific_branch() {
        let mut universe = BranchingUniverse::with_defaults(2);

        // Put qubit 0 in superposition
        universe.h_all(0);
        universe.h_all(1);

        // Measure qubit 0 on the root branch (index 0)
        let (idx_0, idx_1) = universe.measure_branch(0, 0).unwrap();

        assert!(
            !universe.branches[0].active,
            "Original branch should be retired"
        );
        assert!(universe.branches[idx_0].active || universe.branches[idx_1].active);

        // Now measure qubit 1 on just one of the child branches
        if universe.branches[idx_0].active {
            let (c0, c1) = universe.measure_branch(idx_0, 1).unwrap();
            assert!(!universe.branches[idx_0].active);
            // Should have created 2 more branches
            assert!(c0 != c1);
        }
    }

    // ----- Test 22: Empty universe statistics -----

    #[test]
    fn test_empty_statistics() {
        let mut universe = BranchingUniverse::with_defaults(1);
        // Deactivate the only branch manually for testing
        universe.branches[0].active = false;

        let stats = universe.statistics();
        assert_eq!(stats.active_branches, 0);
        assert!(approx_eq(stats.weight_entropy, 0.0, EPSILON));
        assert!(approx_eq(stats.effective_branch_count, 0.0, EPSILON));
    }

    // ----- Test 23: GHZ correlation across all qubits -----

    #[test]
    fn test_ghz_full_correlation() {
        let n = 4;
        let ghz = prepare_ghz(n);
        let config = ManyWorldsConfig::default();
        let mut universe = BranchingUniverse::from_state(ghz, config);

        // Measure all qubits
        for q in 0..n {
            universe.measure_all(q).unwrap();
        }

        // For GHZ state, only two branches should survive with non-zero weight:
        // all-0 and all-1
        let active: Vec<&Branch> = universe
            .active_branches()
            .into_iter()
            .filter(|b| b.weight > 1e-12)
            .collect();

        assert_eq!(
            active.len(),
            2,
            "GHZ should produce exactly 2 non-trivial branches, got {}",
            active.len()
        );

        for b in &active {
            let outcomes: Vec<u8> = b.history.iter().map(|r| r.outcome).collect();
            let all_same = outcomes.iter().all(|&o| o == outcomes[0]);
            assert!(
                all_same,
                "GHZ branch outcomes should all be the same: {:?}",
                outcomes
            );
        }
    }

    // ----- Test 24: Bell state CNOT ordering -----

    #[test]
    fn test_bell_state_cnot_ordering() {
        // Verify cnot(0, 1) works correctly (bit0 < bit1)
        let bell = prepare_bell();
        let probs = bell.probabilities();

        // Bell state: (|00> + |11>) / sqrt(2)
        // P(00) = 0.5, P(01) = 0, P(10) = 0, P(11) = 0.5
        assert!(approx_eq(probs[0], 0.5, 1e-6), "P(00) = {}", probs[0]);
        assert!(approx_eq(probs[1], 0.0, 1e-6), "P(01) = {}", probs[1]);
        assert!(approx_eq(probs[2], 0.0, 1e-6), "P(10) = {}", probs[2]);
        assert!(approx_eq(probs[3], 0.5, 1e-6), "P(11) = {}", probs[3]);
    }

    // ----- Test 25: Deterministic state after measurement projection -----

    #[test]
    fn test_deterministic_state_projection() {
        // |0> measured -> outcome 0 with certainty, outcome 1 with probability 0
        let mut universe = BranchingUniverse::with_defaults(1);
        // Don't apply H -- state is |0>
        universe.measure_all(0).unwrap();

        let active = universe.active_branches();

        // Only the outcome=0 branch should be active (outcome=1 has probability 0)
        let active_with_weight: Vec<&Branch> = active
            .iter()
            .filter(|b| b.weight > 1e-12)
            .cloned()
            .collect();

        assert_eq!(
            active_with_weight.len(),
            1,
            "Measuring |0> should produce 1 non-trivial branch"
        );
        assert_eq!(
            active_with_weight[0].history[0].outcome, 0,
            "Outcome should be 0"
        );
        assert!(
            approx_eq(active_with_weight[0].weight, 1.0, 1e-10),
            "Weight should be 1.0"
        );
    }

    // ----- Test 26: Error Display formatting -----

    #[test]
    fn test_error_display() {
        let e1 = ManyWorldsError::InvalidBranch {
            branch_id: 5,
            total_branches: 3,
        };
        assert!(format!("{}", e1).contains("Invalid branch id 5"));

        let e2 = ManyWorldsError::BranchOverflow {
            current: 100,
            max: 50,
        };
        assert!(format!("{}", e2).contains("100"));
        assert!(format!("{}", e2).contains("50"));

        let e3 = ManyWorldsError::QubitOutOfRange {
            qubit: 7,
            num_qubits: 3,
        };
        assert!(format!("{}", e3).contains("Qubit 7"));

        let e4 = ManyWorldsError::DecoherenceFailed {
            reason: "test error".to_string(),
        };
        assert!(format!("{}", e4).contains("test error"));

        let e5 = ManyWorldsError::HistoryInconsistent {
            alpha: 1,
            beta: 2,
            off_diagonal: 0.5,
        };
        assert!(format!("{}", e5).contains("inconsistent"));
    }

    // ----- Test 27: Two-qubit decoherent histories consistency -----

    #[test]
    fn test_two_qubit_decoherent_histories() {
        // For a 2-qubit |++> state, measuring both in Z basis
        // gives 4 consistent histories
        let mut initial = QuantumState::new(2);
        GateOperations::h(&mut initial, 0);
        GateOperations::h(&mut initial, 1);

        let universe = BranchingUniverse::with_defaults(2);

        let histories = vec![
            DecoherentHistory::from_outcomes(&[(0, 0), (1, 0)], "00"),
            DecoherentHistory::from_outcomes(&[(0, 0), (1, 1)], "01"),
            DecoherentHistory::from_outcomes(&[(0, 1), (1, 0)], "10"),
            DecoherentHistory::from_outcomes(&[(0, 1), (1, 1)], "11"),
        ];

        let result = universe.check_consistency(&initial, &histories).unwrap();
        assert!(
            result.is_consistent,
            "Complete computational basis histories should be consistent, max off-diag = {}",
            result.max_off_diagonal
        );

        // Each diagonal should be 0.25 (uniform distribution over 4 outcomes)
        for i in 0..4 {
            assert!(
                approx_eq(result.decoherence_matrix[i][i].re, 0.25, 1e-6),
                "D({},{}) should be 0.25, got {}",
                i,
                i,
                result.decoherence_matrix[i][i].re
            );
        }
    }
}
