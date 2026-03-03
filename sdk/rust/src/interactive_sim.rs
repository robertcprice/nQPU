//! Interactive stateful simulation API (E2).
//!
//! Provides first-class `step`, `undo`, and `fork` operations for notebook-style
//! iterative exploration of a circuit.

use crate::ascii_viz::apply_gate_to_state;
use crate::gates::Gate;
use crate::state_checkpoint::{CheckpointManager, StateDiff};
use crate::QuantumState;

/// Interactive simulator with checkpoint-backed time travel.
pub struct InteractiveSimulator {
    gates: Vec<Gate>,
    state: QuantumState,
    next_gate_index: usize,
    checkpoints: CheckpointManager,
    checkpoint_history: Vec<usize>,
}

impl InteractiveSimulator {
    /// Create a new interactive simulator from a fixed gate list.
    pub fn new(num_qubits: usize, gates: Vec<Gate>) -> Self {
        let state = QuantumState::new(num_qubits);
        let mut checkpoints = CheckpointManager::new();
        let init_id = checkpoints.checkpoint(&state, 0, "init");
        Self {
            gates,
            state,
            next_gate_index: 0,
            checkpoints,
            checkpoint_history: vec![init_id],
        }
    }

    /// Return immutable state access for inspection.
    pub fn state(&self) -> &QuantumState {
        &self.state
    }

    /// Return mutable state access for advanced use cases.
    pub fn state_mut(&mut self) -> &mut QuantumState {
        &mut self.state
    }

    /// Current position in the circuit (next gate to execute).
    pub fn next_gate_index(&self) -> usize {
        self.next_gate_index
    }

    /// Total number of gates in the circuit.
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Whether all gates have been executed.
    pub fn is_finished(&self) -> bool {
        self.next_gate_index >= self.gates.len()
    }

    /// Execute one gate and checkpoint the resulting state.
    ///
    /// Returns `Ok(Some(checkpoint_id))` if a gate was executed, or `Ok(None)`
    /// when already at end-of-circuit.
    pub fn step(&mut self) -> Result<Option<usize>, String> {
        if self.is_finished() {
            return Ok(None);
        }

        let gate_idx = self.next_gate_index;
        let gate = self
            .gates
            .get(gate_idx)
            .ok_or_else(|| format!("missing gate at index {}", gate_idx))?;

        apply_gate_to_state(&mut self.state, gate);
        self.next_gate_index += 1;

        let checkpoint_id = self.checkpoints.checkpoint(
            &self.state,
            self.next_gate_index,
            &format!("after_gate_{}", gate_idx),
        );
        self.checkpoint_history.push(checkpoint_id);

        Ok(Some(checkpoint_id))
    }

    /// Execute up to `n` gates, stopping early at end-of-circuit.
    ///
    /// Returns the number of gates actually executed.
    pub fn step_n(&mut self, n: usize) -> Result<usize, String> {
        let mut executed = 0usize;
        for _ in 0..n {
            if self.step()?.is_none() {
                break;
            }
            executed += 1;
        }
        Ok(executed)
    }

    /// Undo one step by restoring the previous checkpoint.
    ///
    /// Returns `Ok(true)` when a step was undone and `Ok(false)` when already
    /// at the initial state.
    pub fn undo(&mut self) -> Result<bool, String> {
        if self.next_gate_index == 0 || self.checkpoint_history.len() <= 1 {
            return Ok(false);
        }

        // Drop current checkpoint and restore the previous one.
        self.checkpoint_history.pop();
        let restore_id = *self
            .checkpoint_history
            .last()
            .ok_or_else(|| "checkpoint history unexpectedly empty".to_string())?;

        let restored = self
            .checkpoints
            .restore(restore_id)
            .ok_or_else(|| format!("failed to restore checkpoint {}", restore_id))?;
        self.state = restored;
        self.next_gate_index = self.next_gate_index.saturating_sub(1);
        Ok(true)
    }

    /// Fork a new simulator instance at the current state.
    ///
    /// The child receives the remaining circuit and an independent checkpoint
    /// timeline starting from the fork point.
    pub fn fork(&self) -> Result<Self, String> {
        let current_checkpoint_id = *self
            .checkpoint_history
            .last()
            .ok_or_else(|| "cannot fork without a checkpoint".to_string())?;
        let (fork_manager, fork_state) = self
            .checkpoints
            .fork(current_checkpoint_id)
            .ok_or_else(|| format!("failed to fork from checkpoint {}", current_checkpoint_id))?;
        let fork_id = fork_manager
            .latest()
            .map(|c| c.id)
            .ok_or_else(|| "fork manager missing fork checkpoint".to_string())?;

        Ok(Self {
            gates: self.gates.clone(),
            state: fork_state,
            next_gate_index: self.next_gate_index,
            checkpoints: fork_manager,
            checkpoint_history: vec![fork_id],
        })
    }

    /// Diff the current state against the latest checkpoint (if available).
    pub fn diff_from_latest_checkpoint(&self) -> Option<StateDiff> {
        let latest = self.checkpoints.latest()?;
        self.checkpoints.diff_with_current(&self.state, latest.id)
    }

    /// Snapshot list from the underlying checkpoint manager.
    pub fn checkpoint_list(&self) -> Vec<(usize, String, usize, f64)> {
        self.checkpoints.list()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_and_undo_flow() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let mut sim = InteractiveSimulator::new(2, gates);

        assert_eq!(sim.next_gate_index(), 0);
        assert!(!sim.is_finished());

        sim.step().unwrap().expect("expected first step checkpoint");
        assert_eq!(sim.next_gate_index(), 1);
        let probs_after_h = sim.state().probabilities();
        assert!((probs_after_h[0] - 0.5).abs() < 1e-10);
        assert!((probs_after_h[1] - 0.5).abs() < 1e-10);

        sim.step().unwrap().expect("expected second step checkpoint");
        assert_eq!(sim.next_gate_index(), 2);
        assert!(sim.is_finished());
        let probs_after_cnot = sim.state().probabilities();
        assert!((probs_after_cnot[0] - 0.5).abs() < 1e-10);
        assert!((probs_after_cnot[3] - 0.5).abs() < 1e-10);

        assert!(sim.undo().unwrap(), "undo should succeed");
        assert_eq!(sim.next_gate_index(), 1);
        let probs_undo = sim.state().probabilities();
        assert!((probs_undo[0] - 0.5).abs() < 1e-10);
        assert!((probs_undo[1] - 0.5).abs() < 1e-10);
        assert!(probs_undo[3].abs() < 1e-10);
    }

    #[test]
    fn test_step_n_respects_end_of_circuit() {
        let gates = vec![Gate::h(0), Gate::x(0)];
        let mut sim = InteractiveSimulator::new(1, gates);
        let executed = sim.step_n(10).unwrap();
        assert_eq!(executed, 2);
        assert!(sim.is_finished());
    }

    #[test]
    fn test_fork_produces_independent_branch() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::x(0)];
        let mut parent = InteractiveSimulator::new(2, gates);

        parent.step().unwrap().expect("parent step 1");
        let mut child = parent.fork().expect("fork should succeed");

        // Advance only the child.
        child.step().unwrap().expect("child step 2");
        child.step().unwrap().expect("child step 3");

        // Parent should remain at gate index 1.
        assert_eq!(parent.next_gate_index(), 1);
        assert_eq!(child.next_gate_index(), 3);

        let p_parent = parent.state().probabilities();
        let p_child = child.state().probabilities();
        assert_ne!(p_parent, p_child, "forked branch should evolve independently");
    }

    #[test]
    fn test_undo_at_start_is_noop() {
        let mut sim = InteractiveSimulator::new(1, vec![Gate::h(0)]);
        assert!(!sim.undo().unwrap());
        assert_eq!(sim.next_gate_index(), 0);
    }
}

