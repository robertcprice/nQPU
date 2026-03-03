// Adaptive Matrix Product State (MPS) with Automatic Bond Dimension Management
//
// This module extends the basic MPS with automatic bond dimension expansion
// when entanglement exceeds thresholds.


use crate::tensor_network::MPSSimulator;

/// Configuration for adaptive bond dimension behavior
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Initial bond dimension
    pub initial_bond_dim: usize,

    /// Maximum bond dimension (memory limit)
    pub max_bond_dim: usize,

    /// Entanglement threshold for expansion (0.0-1.0)
    /// When bond usage exceeds this fraction, expand
    pub expansion_threshold: f64,

    /// Whether to warn when bond dimension is increased
    pub verbose: bool,

    /// Growth factor when expanding (usually 2.0)
    pub growth_factor: usize,

    /// Optional target maximum entanglement entropy for adaptive truncation.
    /// When set, truncation threshold is adjusted to keep entropy near this value.
    pub target_max_entropy: Option<f64>,

    /// Entropy hysteresis band (avoid oscillation).
    pub entropy_hysteresis: f64,

    /// Minimum truncation threshold (more accuracy).
    pub truncation_min: f64,

    /// Maximum truncation threshold (more aggressive compression).
    pub truncation_max: f64,

    /// Multiplicative adjustment factor for truncation threshold.
    pub truncation_adjust: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            initial_bond_dim: 2,
            max_bond_dim: 64,
            expansion_threshold: 0.7, // Expand at 70% of max
            verbose: true,
            growth_factor: 2,
            target_max_entropy: None,
            entropy_hysteresis: 0.05,
            truncation_min: 1e-14,
            truncation_max: 1e-4,
            truncation_adjust: 2.0,
        }
    }
}

impl AdaptiveConfig {
    /// Create a new adaptive config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set initial bond dimension
    pub fn with_initial_bond_dim(mut self, dim: usize) -> Self {
        self.initial_bond_dim = dim;
        self
    }

    /// Set maximum bond dimension
    pub fn with_max_bond_dim(mut self, dim: usize) -> Self {
        self.max_bond_dim = dim;
        self
    }

    /// Set expansion threshold (0.0 to 1.0)
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold >= 0.0 && threshold <= 1.0,
            "Threshold must be in [0, 1]"
        );
        self.expansion_threshold = threshold;
        self
    }

    /// Set verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set growth factor
    pub fn with_growth_factor(mut self, factor: usize) -> Self {
        assert!(factor >= 1, "Growth factor must be >= 1");
        self.growth_factor = factor;
        self
    }

    /// Enable entropy control with a target maximum entropy.
    pub fn with_entropy_target(mut self, target_max: f64) -> Self {
        assert!(target_max >= 0.0, "Entropy target must be >= 0");
        self.target_max_entropy = Some(target_max);
        self
    }

    /// Configure entropy control parameters.
    pub fn with_entropy_control(
        mut self,
        target_max: f64,
        hysteresis: f64,
        trunc_min: f64,
        trunc_max: f64,
        adjust: f64,
    ) -> Self {
        assert!(target_max >= 0.0, "Entropy target must be >= 0");
        assert!(hysteresis >= 0.0, "Hysteresis must be >= 0");
        assert!(
            trunc_min >= 0.0 && trunc_max >= trunc_min,
            "Invalid truncation bounds"
        );
        assert!(adjust >= 1.0, "Adjustment factor must be >= 1");
        self.target_max_entropy = Some(target_max);
        self.entropy_hysteresis = hysteresis;
        self.truncation_min = trunc_min;
        self.truncation_max = trunc_max;
        self.truncation_adjust = adjust;
        self
    }
}

/// MPS with adaptive bond dimension management
///
/// Automatically expands bond dimension when entanglement exceeds threshold.
pub struct AdaptiveMPS {
    mps: MPSSimulator,
    config: AdaptiveConfig,
    current_bond_dim: usize,
    expansion_count: usize,
}

impl AdaptiveMPS {
    /// Create a new adaptive MPS simulator
    pub fn new(num_qubits: usize, config: AdaptiveConfig) -> Self {
        let initial_bond_dim = config.initial_bond_dim;
        let mut mps = MPSSimulator::new(num_qubits, Some(initial_bond_dim));
        if config.target_max_entropy.is_some() {
            mps.enable_entanglement_tracking(true);
        }

        Self {
            mps,
            config,
            current_bond_dim: initial_bond_dim,
            expansion_count: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(num_qubits: usize) -> Self {
        Self::new(num_qubits, AdaptiveConfig::default())
    }

    /// Get current bond dimension
    pub fn current_bond_dim(&self) -> usize {
        self.current_bond_dim
    }

    /// Get maximum bond dimension
    pub fn max_bond_dim(&self) -> usize {
        self.config.max_bond_dim
    }

    /// Get number of times bond dimension was expanded
    pub fn expansion_count(&self) -> usize {
        self.expansion_count
    }

    /// Get bond dimensions at each bond
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.mps.bond_dimensions()
    }

    /// Check if bond dimension can be expanded
    pub fn can_expand(&self) -> bool {
        self.current_bond_dim < self.config.max_bond_dim
    }

    /// Check if bond dimension should be expanded based on usage
    fn should_expand(&self) -> bool {
        if !self.can_expand() {
            return false;
        }

        let bond_dims = self.mps.bond_dimensions();
        if bond_dims.is_empty() {
            return false;
        }

        // Check if any bond is near current limit
        let max_used = bond_dims.iter().cloned().fold(0_usize, usize::max);
        let threshold = (self.current_bond_dim as f64) * self.config.expansion_threshold;

        max_used as f64 > threshold
    }

    /// Expand bond dimension
    fn expand_bond_dimension(&mut self) {
        if !self.can_expand() {
            if self.config.verbose {
                eprintln!(
                    "Warning: Cannot expand beyond max bond dimension {}",
                    self.config.max_bond_dim
                );
            }
            return;
        }

        let new_bond =
            (self.current_bond_dim * self.config.growth_factor).min(self.config.max_bond_dim);

        if self.config.verbose {
            eprintln!(
                "AdaptiveMPS: Expanding bond dimension {} -> {}",
                self.current_bond_dim, new_bond
            );
        }

        // Update MPS bond dimension limit without discarding state
        self.mps.set_max_bond_dim(Some(new_bond));
        self.current_bond_dim = new_bond;
        self.expansion_count += 1;
    }

    // ============================================================
    // Standard gate operations (delegate to MPS with adaptive check)
    // ============================================================

    /// Apply Hadamard gate
    pub fn h(&mut self, qubit: usize) {
        self.mps.h(qubit);
        self.check_expand();
    }

    /// Apply X (NOT) gate
    pub fn x(&mut self, qubit: usize) {
        self.mps.x(qubit);
        // Single-qubit gates don't increase entanglement much
    }

    /// Apply Y gate
    pub fn y(&mut self, qubit: usize) {
        self.mps.y(qubit);
    }

    /// Apply Z gate
    pub fn z(&mut self, qubit: usize) {
        self.mps.z(qubit);
    }

    /// Apply CNOT gate (main entanglement creator)
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.mps.cnot(control, target);
        self.check_expand();
        self.check_entropy_control();
    }

    /// Apply RX rotation
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        self.mps.rx(qubit, theta);
    }

    /// Apply RY rotation
    pub fn ry(&mut self, qubit: usize, theta: f64) {
        self.mps.ry(qubit, theta);
    }

    /// Apply RZ rotation
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        self.mps.rz(qubit, theta);
    }

    /// Check for expansion after potentially entangling operation
    fn check_expand(&mut self) {
        if self.should_expand() {
            self.expand_bond_dimension();
        }
    }

    /// Adjust truncation threshold to keep entanglement near target.
    fn check_entropy_control(&mut self) {
        let target = match self.config.target_max_entropy {
            Some(v) => v,
            None => return,
        };
        let stats = match self.mps.entanglement_stats() {
            Some(s) => s,
            None => return,
        };
        let max_ent = stats.max;
        let mut thr = self.mps.truncation_threshold();

        if max_ent > target + self.config.entropy_hysteresis {
            thr = (thr * self.config.truncation_adjust).min(self.config.truncation_max);
            self.mps.set_truncation_threshold(thr);
        } else if max_ent + self.config.entropy_hysteresis < target {
            thr = (thr / self.config.truncation_adjust).max(self.config.truncation_min);
            self.mps.set_truncation_threshold(thr);
        }
    }

    /// Measure all qubits
    pub fn measure(&mut self) -> u64 {
        self.mps.measure() as u64
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.mps.num_qubits()
    }

    /// Enable entanglement tracking (required for entropy control).
    pub fn enable_entanglement_tracking(&mut self, enabled: bool) {
        self.mps.enable_entanglement_tracking(enabled);
    }

    /// Get current truncation threshold (for debugging).
    pub fn truncation_threshold(&self) -> f64 {
        self.mps.truncation_threshold()
    }

    /// Get statistics about the adaptive MPS
    pub fn statistics(&self) -> AdaptiveStatistics {
        let bond_dims = self.bond_dimensions();
        let max_bond = bond_dims.iter().cloned().fold(0_usize, usize::max);
        let avg_bond = if bond_dims.is_empty() {
            0.0
        } else {
            bond_dims.iter().sum::<usize>() as f64 / bond_dims.len() as f64
        };

        AdaptiveStatistics {
            current_bond_dim: self.current_bond_dim,
            max_bond_dim: self.config.max_bond_dim,
            expansion_count: self.expansion_count,
            avg_bond_dimension: avg_bond,
            max_bond_dimension: max_bond as f64,
            num_qubits: self.mps.num_qubits(),
        }
    }
}

/// Statistics about adaptive MPS behavior
#[derive(Debug, Clone)]
pub struct AdaptiveStatistics {
    pub current_bond_dim: usize,
    pub max_bond_dim: usize,
    pub expansion_count: usize,
    pub avg_bond_dimension: f64,
    pub max_bond_dimension: f64,
    pub num_qubits: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveConfig::default();
        assert_eq!(config.initial_bond_dim, 2);
        assert_eq!(config.max_bond_dim, 64);
        assert_eq!(config.expansion_threshold, 0.7);
    }

    #[test]
    fn test_adaptive_config_builder() {
        let config = AdaptiveConfig::new()
            .with_initial_bond_dim(4)
            .with_max_bond_dim(32)
            .with_threshold(0.8)
            .with_verbose(false)
            .with_growth_factor(3);

        assert_eq!(config.initial_bond_dim, 4);
        assert_eq!(config.max_bond_dim, 32);
        assert_eq!(config.expansion_threshold, 0.8);
        assert_eq!(config.verbose, false);
        assert_eq!(config.growth_factor, 3);
    }

    #[test]
    fn test_adaptive_mps_creation() {
        let sim = AdaptiveMPS::with_defaults(10);
        assert_eq!(sim.num_qubits(), 10);
        assert_eq!(sim.current_bond_dim(), 2);
        assert_eq!(sim.max_bond_dim(), 64);
        assert_eq!(sim.expansion_count(), 0);
    }

    #[test]
    fn test_bond_dimensions() {
        let sim = AdaptiveMPS::with_defaults(20);
        let dims = sim.bond_dimensions();
        // Should have dimensions for each bond (n-1 bonds)
        assert_eq!(dims.len(), 19);
    }

    #[test]
    fn test_adaptive_statistics() {
        let mut sim = AdaptiveMPS::with_defaults(20);
        sim.h(0);
        sim.cnot(0, 1);

        let stats = sim.statistics();
        assert_eq!(stats.num_qubits, 20);
        assert!(stats.avg_bond_dimension >= 0.0);
        assert!(stats.current_bond_dim <= stats.max_bond_dim);

        println!("Statistics: {:?}", stats);
    }

    #[test]
    fn test_adaptive_expansion_preserves_state() {
        let cfg = AdaptiveConfig::default()
            .with_initial_bond_dim(2)
            .with_max_bond_dim(4)
            .with_threshold(0.1)
            .with_verbose(false)
            .with_growth_factor(2);
        let mut sim = AdaptiveMPS::new(2, cfg);
        sim.h(0);
        sim.cnot(0, 1);
        let max_bond = sim.bond_dimensions().into_iter().max().unwrap_or(1);
        assert!(max_bond >= 2);
        assert!(sim.expansion_count() >= 1);
    }
}
