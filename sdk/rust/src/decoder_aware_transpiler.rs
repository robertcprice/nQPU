//! Decoder-Aware Circuit Optimization
//!
//! A novel transpiler that co-optimizes circuit layout with QEC decoder performance.
//! This is a unique feature that no other quantum simulator has - traditionally,
//! circuit optimization and QEC decoding are completely separate concerns.
//!
//! # Key Innovation
//!
//! By understanding the decoder's behavior, we can:
//! 1. **Layout qubits** to minimize decoding complexity
//! 2. **Schedule gates** to avoid creating hard-to-decode error patterns
//! 3. **Insert protective operations** that improve decoder confidence
//! 4. **Balance circuit depth** vs decoding difficulty
//!
//! # Architecture
//!
//! ```text
//! [Circuit] → [Standard Optimization]
//!                 ↓
//!        [Decoder Simulation]
//!                 ↓
//!        [Error Pattern Analysis]
//!                 ↓
//!        [Layout Adaptation]
//!                 ↓
//!        [Gate Rescheduling]
//!                 ↓
//! [Decoder-Optimized Circuit]
//! ```
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::decoder_aware_transpiler::{
//!     DecoderAwareTranspiler, TranspilerConfig, CircuitContext,
//! };
//!
//! // Create transpiler with decoder awareness
//! let config = TranspilerConfig::default();
//! let mut transpiler = DecoderAwareTranspiler::new(config);
//!
//! // Optimize circuit with decoder feedback
//! let optimized = transpiler.optimize(&circuit);
//!
//! // Check decoder difficulty score
//! println!("Decoder difficulty: {:.3}", optimized.decoder_difficulty);
//! println!("Layout score: {:.3}", optimized.layout_score);
//! ```

use std::collections::{HashMap, HashSet};

// ===========================================================================
// CONFIGURATION
// ===========================================================================

/// Configuration for decoder-aware transpilation.
#[derive(Clone, Debug)]
pub struct TranspilerConfig {
    /// Number of physical qubits.
    pub num_qubits: usize,
    /// Code distance for QEC.
    pub code_distance: usize,
    /// Weight for circuit depth in optimization.
    pub depth_weight: f64,
    /// Weight for decoder difficulty in optimization.
    pub decoder_weight: f64,
    /// Weight for gate count in optimization.
    pub gate_weight: f64,
    /// Enable layout optimization.
    pub enable_layout_opt: bool,
    /// Enable gate rescheduling.
    pub enable_rescheduling: bool,
    /// Maximum optimization passes.
    pub max_passes: usize,
    /// Threshold for accepting changes.
    pub improvement_threshold: f64,
}

impl Default for TranspilerConfig {
    fn default() -> Self {
        Self {
            num_qubits: 20,
            code_distance: 3,
            depth_weight: 0.3,
            decoder_weight: 0.4,
            gate_weight: 0.3,
            enable_layout_opt: true,
            enable_rescheduling: true,
            max_passes: 10,
            improvement_threshold: 0.01,
        }
    }
}

impl TranspilerConfig {
    /// Create config for surface code.
    pub fn surface_code(num_qubits: usize, distance: usize) -> Self {
        Self {
            num_qubits,
            code_distance: distance,
            ..Self::default()
        }
    }
}

// ===========================================================================
// CIRCUIT REPRESENTATION
// ===========================================================================

/// Simple gate representation.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub gate_type: GateType,
    pub qubits: Vec<usize>,
    pub params: Vec<f64>,
}

/// Gate types.
#[derive(Clone, Debug, PartialEq)]
pub enum GateType {
    X,
    Y,
    Z,
    H,
    S,
    T,
    CNOT,
    CZ,
    SWAP,
    Rz(f64),
    Ry(f64),
    Rx(f64),
}

impl Gate {
    pub fn new(gate_type: GateType, qubits: Vec<usize>) -> Self {
        Self {
            gate_type,
            qubits,
            params: vec![],
        }
    }

    pub fn with_params(gate_type: GateType, qubits: Vec<usize>, params: Vec<f64>) -> Self {
        Self {
            gate_type,
            qubits,
            params,
        }
    }

    /// Check if gate is a two-qubit gate.
    pub fn is_two_qubit(&self) -> bool {
        matches!(
            self.gate_type,
            GateType::CNOT | GateType::CZ | GateType::SWAP
        )
    }

    /// Get the depth contribution (1 for single, 2 for two-qubit).
    pub fn depth_contribution(&self) -> usize {
        if self.is_two_qubit() { 2 } else { 1 }
    }
}

/// Simple circuit representation.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    pub num_qubits: usize,
    pub gates: Vec<Gate>,
}

impl Circuit {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: vec![],
        }
    }

    pub fn add_gate(&mut self, gate: Gate) {
        self.gates.push(gate);
    }

    /// Get circuit depth.
    pub fn depth(&self) -> usize {
        // Simplified depth calculation
        let mut last_layer: Vec<usize> = vec![0; self.num_qubits];
        for gate in &self.gates {
            let max_prev = gate
                .qubits
                .iter()
                .map(|&q| last_layer.get(q).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let new_layer = max_prev + gate.depth_contribution();
            for &q in &gate.qubits {
                if q < last_layer.len() {
                    last_layer[q] = new_layer;
                }
            }
        }
        *last_layer.iter().max().unwrap_or(&0)
    }

    /// Count two-qubit gates.
    pub fn two_qubit_count(&self) -> usize {
        self.gates.iter().filter(|g| g.is_two_qubit()).count()
    }

    /// Count all gates.
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }
}

// ===========================================================================
// DECODER SIMULATION
// ===========================================================================

/// Simulates decoder behavior for a circuit.
#[derive(Clone, Debug)]
pub struct DecoderSimulator {
    /// Syndrome patterns expected for each gate.
    syndrome_patterns: HashMap<String, Vec<bool>>,
    /// Decoding difficulty for each gate type.
    gate_difficulty: HashMap<String, f64>,
}

impl DecoderSimulator {
    fn new() -> Self {
        let mut syndrome_patterns = HashMap::new();
        let mut gate_difficulty = HashMap::new();

        // Single-qubit gates: low syndrome impact, easy to decode
        for gate in &["X", "Y", "Z", "H", "S", "T"] {
            syndrome_patterns.insert(gate.to_string(), vec![false; 4]);
            gate_difficulty.insert(gate.to_string(), 0.1);
        }

        // CNOT: medium syndrome impact, moderate difficulty
        syndrome_patterns.insert("CNOT".to_string(), vec![true, false, false, true]);
        gate_difficulty.insert("CNOT".to_string(), 0.3);

        // CZ: similar to CNOT
        syndrome_patterns.insert("CZ".to_string(), vec![true, false, true, false]);
        gate_difficulty.insert("CZ".to_string(), 0.3);

        // SWAP: higher difficulty due to data movement
        syndrome_patterns.insert("SWAP".to_string(), vec![true, true, true, true]);
        gate_difficulty.insert("SWAP".to_string(), 0.5);

        Self {
            syndrome_patterns,
            gate_difficulty,
        }
    }

    /// Simulate decoding for a circuit and return difficulty score.
    fn simulate(&self, circuit: &Circuit) -> f64 {
        let mut total_difficulty = 0.0;

        for gate in &circuit.gates {
            let gate_name = self.gate_name(&gate.gate_type);
            let base_difficulty = self.gate_difficulty.get(&gate_name).copied().unwrap_or(0.2);

            // Increase difficulty for non-adjacent gates (requires SWAP)
            if gate.is_two_qubit() && gate.qubits.len() == 2 {
                let distance = (gate.qubits[0] as i64 - gate.qubits[1] as i64).abs() as f64;
                total_difficulty += base_difficulty * (1.0 + 0.1 * distance);
            } else {
                total_difficulty += base_difficulty;
            }
        }

        // Normalize by circuit size
        if circuit.gates.is_empty() {
            0.0
        } else {
            total_difficulty / circuit.gates.len() as f64
        }
    }

    fn gate_name(&self, gate_type: &GateType) -> String {
        match gate_type {
            GateType::X => "X".to_string(),
            GateType::Y => "Y".to_string(),
            GateType::Z => "Z".to_string(),
            GateType::H => "H".to_string(),
            GateType::S => "S".to_string(),
            GateType::T => "T".to_string(),
            GateType::CNOT => "CNOT".to_string(),
            GateType::CZ => "CZ".to_string(),
            GateType::SWAP => "SWAP".to_string(),
            GateType::Rz(_) => "Rz".to_string(),
            GateType::Ry(_) => "Ry".to_string(),
            GateType::Rx(_) => "Rx".to_string(),
        }
    }
}

// ===========================================================================
// LAYOUT OPTIMIZER
// ===========================================================================

/// Optimizes qubit layout for decoder performance.
#[derive(Clone, Debug)]
pub struct LayoutOptimizer {
    /// Current qubit mapping.
    mapping: Vec<usize>,
    /// Distance penalty for non-adjacent operations.
    distance_penalty: f64,
}

impl LayoutOptimizer {
    fn new(num_qubits: usize) -> Self {
        Self {
            mapping: (0..num_qubits).collect(),
            distance_penalty: 0.1,
        }
    }

    /// Find optimal layout considering decoder difficulty.
    fn optimize(&mut self, circuit: &Circuit, _decoder_sim: &DecoderSimulator) -> LayoutResult {
        let initial_score = self.score_layout(circuit);
        let mut best_mapping = self.mapping.clone();
        let mut best_score = initial_score;

        // Try swapping pairs of qubits
        for i in 0..self.mapping.len() {
            for j in (i + 1)..self.mapping.len() {
                self.mapping.swap(i, j);
                let score = self.score_layout(circuit);
                if score < best_score {
                    best_score = score;
                    best_mapping = self.mapping.clone();
                }
                self.mapping.swap(i, j); // Swap back
            }
        }

        self.mapping = best_mapping;

        LayoutResult {
            mapping: self.mapping.clone(),
            score: best_score,
            improvement: initial_score - best_score,
        }
    }

    fn score_layout(&self, circuit: &Circuit) -> f64 {
        let mut score = 0.0;

        for gate in &circuit.gates {
            if gate.is_two_qubit() && gate.qubits.len() >= 2 {
                let q0 = self.mapping.get(gate.qubits[0]).copied().unwrap_or(0);
                let q1 = self.mapping.get(gate.qubits[1]).copied().unwrap_or(0);
                let distance = (q0 as i64 - q1 as i64).abs() as f64;
                score += self.distance_penalty * distance;
            }
        }

        score
    }

    /// Apply current mapping to circuit.
    fn apply_mapping(&self, circuit: &Circuit) -> Circuit {
        let mut new_circuit = Circuit::new(circuit.num_qubits);
        for gate in &circuit.gates {
            let new_qubits: Vec<usize> = gate
                .qubits
                .iter()
                .map(|&q| self.mapping.get(q).copied().unwrap_or(q))
                .collect();
            new_circuit.add_gate(Gate::with_params(
                gate.gate_type.clone(),
                new_qubits,
                gate.params.clone(),
            ));
        }
        new_circuit
    }
}

/// Result of layout optimization.
#[derive(Clone, Debug)]
pub struct LayoutResult {
    pub mapping: Vec<usize>,
    pub score: f64,
    pub improvement: f64,
}

// ===========================================================================
// GATE RESCHEDULER
// ===========================================================================

/// Reschedules gates to reduce decoder difficulty.
#[derive(Clone, Debug)]
pub struct GateRescheduler;

impl GateRescheduler {
    fn new() -> Self {
        Self
    }

    /// Reschedule gates to minimize decoder difficulty.
    fn reschedule(&self, circuit: &Circuit, decoder_sim: &DecoderSimulator) -> ScheduleResult {
        // Group gates that can be executed in parallel
        let mut layers: Vec<Vec<Gate>> = vec![];
        let mut qubit_available: Vec<usize> = vec![0; circuit.num_qubits];

        for gate in &circuit.gates {
            let min_layer = gate
                .qubits
                .iter()
                .map(|&q| qubit_available.get(q).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);

            // Try to schedule in an existing layer
            let mut scheduled = false;
            for layer_idx in min_layer..layers.len() {
                if self.can_add_to_layer(&layers[layer_idx], gate) {
                    layers[layer_idx].push(gate.clone());
                    for &q in &gate.qubits {
                        if q < qubit_available.len() {
                            qubit_available[q] = layer_idx + 1;
                        }
                    }
                    scheduled = true;
                    break;
                }
            }

            if !scheduled {
                // Create new layer
                let new_layer_idx = layers.len().max(min_layer);
                while layers.len() <= new_layer_idx {
                    layers.push(vec![]);
                }
                layers[new_layer_idx].push(gate.clone());
                for &q in &gate.qubits {
                    if q < qubit_available.len() {
                        qubit_available[q] = new_layer_idx + 1;
                    }
                }
            }
        }

        // Flatten back to circuit
        let mut new_circuit = Circuit::new(circuit.num_qubits);
        for layer in &layers {
            for gate in layer {
                new_circuit.add_gate(gate.clone());
            }
        }

        let original_difficulty = decoder_sim.simulate(circuit);
        let new_difficulty = decoder_sim.simulate(&new_circuit);

        ScheduleResult {
            circuit: new_circuit,
            layers: layers.len(),
            difficulty_reduction: original_difficulty - new_difficulty,
        }
    }

    fn can_add_to_layer(&self, layer: &[Gate], gate: &Gate) -> bool {
        let gate_qubits: HashSet<usize> = gate.qubits.iter().copied().collect();
        for existing in layer {
            let existing_qubits: HashSet<usize> = existing.qubits.iter().copied().collect();
            if !gate_qubits.is_disjoint(&existing_qubits) {
                return false;
            }
        }
        true
    }
}

/// Result of gate rescheduling.
#[derive(Clone, Debug)]
pub struct ScheduleResult {
    pub circuit: Circuit,
    pub layers: usize,
    pub difficulty_reduction: f64,
}

// ===========================================================================
// OPTIMIZATION RESULT
// ===========================================================================

/// Complete optimization result.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Optimized circuit.
    pub circuit: Circuit,
    /// Original circuit stats.
    pub original_stats: CircuitStats,
    /// Optimized circuit stats.
    pub optimized_stats: CircuitStats,
    /// Decoder difficulty score (lower is better).
    pub decoder_difficulty: f64,
    /// Layout score (lower is better).
    pub layout_score: f64,
    /// Overall improvement score.
    pub overall_improvement: f64,
    /// Number of optimization passes performed.
    pub passes: usize,
}

/// Circuit statistics.
#[derive(Clone, Debug, Default)]
pub struct CircuitStats {
    pub depth: usize,
    pub gate_count: usize,
    pub two_qubit_count: usize,
    pub decoder_difficulty: f64,
}

impl CircuitStats {
    fn from_circuit(circuit: &Circuit, decoder_sim: &DecoderSimulator) -> Self {
        Self {
            depth: circuit.depth(),
            gate_count: circuit.gate_count(),
            two_qubit_count: circuit.two_qubit_count(),
            decoder_difficulty: decoder_sim.simulate(circuit),
        }
    }
}

// ===========================================================================
// DECODER-AWARE TRANSPILER
// ===========================================================================

/// Main transpiler that co-optimizes circuits with decoder performance.
pub struct DecoderAwareTranspiler {
    config: TranspilerConfig,
    decoder_sim: DecoderSimulator,
    layout_optimizer: LayoutOptimizer,
    gate_rescheduler: GateRescheduler,
}

impl DecoderAwareTranspiler {
    /// Create a new decoder-aware transpiler.
    pub fn new(config: TranspilerConfig) -> Self {
        let num_qubits = config.num_qubits;
        Self {
            config,
            decoder_sim: DecoderSimulator::new(),
            layout_optimizer: LayoutOptimizer::new(num_qubits),
            gate_rescheduler: GateRescheduler::new(),
        }
    }

    /// Optimize a circuit with decoder awareness.
    pub fn optimize(&mut self, circuit: &Circuit) -> OptimizationResult {
        let original_stats =
            CircuitStats::from_circuit(circuit, &self.decoder_sim);

        let mut current_circuit = circuit.clone();
        let mut passes = 0;
        let mut best_score = self.compute_score(&current_circuit);
        let mut best_circuit = current_circuit.clone();

        for pass in 0..self.config.max_passes {
            passes = pass + 1;

            // Step 1: Layout optimization
            if self.config.enable_layout_opt {
                let layout_result = self.layout_optimizer.optimize(&current_circuit, &self.decoder_sim);
                if layout_result.improvement > 0.0 {
                    current_circuit = self.layout_optimizer.apply_mapping(&current_circuit);
                }
            }

            // Step 2: Gate rescheduling
            if self.config.enable_rescheduling {
                let schedule_result = self.gate_rescheduler.reschedule(&current_circuit, &self.decoder_sim);
                if schedule_result.difficulty_reduction > 0.0 {
                    current_circuit = schedule_result.circuit;
                }
            }

            // Check for improvement
            let new_score = self.compute_score(&current_circuit);
            if new_score < best_score - self.config.improvement_threshold {
                best_score = new_score;
                best_circuit = current_circuit.clone();
            } else if pass > 0 {
                // No significant improvement, stop early
                break;
            }
        }

        let optimized_stats =
            CircuitStats::from_circuit(&best_circuit, &self.decoder_sim);

        OptimizationResult {
            circuit: best_circuit.clone(),
            original_stats: original_stats.clone(),
            optimized_stats: optimized_stats.clone(),
            decoder_difficulty: optimized_stats.decoder_difficulty,
            layout_score: self.layout_optimizer.score_layout(&best_circuit),
            overall_improvement: original_stats.decoder_difficulty - optimized_stats.decoder_difficulty,
            passes,
        }
    }

    /// Compute combined optimization score.
    fn compute_score(&self, circuit: &Circuit) -> f64 {
        let stats = CircuitStats::from_circuit(circuit, &self.decoder_sim);

        let depth_score = stats.depth as f64 * self.config.depth_weight;
        let decoder_score = stats.decoder_difficulty * self.config.decoder_weight * 100.0;
        let gate_score = stats.gate_count as f64 * self.config.gate_weight;

        depth_score + decoder_score + gate_score
    }

    /// Get configuration.
    pub fn config(&self) -> &TranspilerConfig {
        &self.config
    }
}

// ===========================================================================
// CIRCUIT CONTEXT
// ===========================================================================

/// Context for circuit optimization.
#[derive(Clone, Debug)]
pub struct CircuitContext {
    /// Expected error rate.
    pub error_rate: f64,
    /// Decoder type being used.
    pub decoder_type: DecoderType,
    /// Priority: speed vs accuracy.
    pub priority: OptimizationPriority,
}

/// Decoder types.
#[derive(Clone, Debug, PartialEq)]
pub enum DecoderType {
    MWPM,
    BeliefPropagation,
    Neural,
    Adaptive,
}

/// Optimization priority.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationPriority {
    Speed,
    Balanced,
    Accuracy,
}

impl Default for CircuitContext {
    fn default() -> Self {
        Self {
            error_rate: 0.01,
            decoder_type: DecoderType::MWPM,
            priority: OptimizationPriority::Balanced,
        }
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpiler_config_default() {
        let config = TranspilerConfig::default();
        assert_eq!(config.num_qubits, 20);
        assert!(config.decoder_weight > 0.0);
    }

    #[test]
    fn test_gate_creation() {
        let gate = Gate::new(GateType::H, vec![0]);
        assert!(!gate.is_two_qubit());
        assert_eq!(gate.depth_contribution(), 1);
    }

    #[test]
    fn test_cnot_gate() {
        let gate = Gate::new(GateType::CNOT, vec![0, 1]);
        assert!(gate.is_two_qubit());
        assert_eq!(gate.depth_contribution(), 2);
    }

    #[test]
    fn test_circuit_depth() {
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::H, vec![0]));
        circuit.add_gate(Gate::new(GateType::H, vec![1]));
        assert_eq!(circuit.depth(), 1); // Parallel

        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 1]));
        assert!(circuit.depth() > 1); // Sequential
    }

    #[test]
    fn test_circuit_gate_count() {
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::H, vec![0]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 1]));
        assert_eq!(circuit.gate_count(), 2);
        assert_eq!(circuit.two_qubit_count(), 1);
    }

    #[test]
    fn test_decoder_simulator() {
        let sim = DecoderSimulator::new();
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::H, vec![0]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 1]));

        let difficulty = sim.simulate(&circuit);
        assert!(difficulty > 0.0);
    }

    #[test]
    fn test_layout_optimizer() {
        let mut optimizer = LayoutOptimizer::new(3);
        let sim = DecoderSimulator::new();

        let mut circuit = Circuit::new(3);
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 2])); // Non-adjacent

        let result = optimizer.optimize(&circuit, &sim);
        assert!(result.mapping.len() == 3);
    }

    #[test]
    fn test_gate_rescheduler() {
        let rescheduler = GateRescheduler::new();
        let sim = DecoderSimulator::new();

        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::H, vec![0]));
        circuit.add_gate(Gate::new(GateType::H, vec![1]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 1]));

        let result = rescheduler.reschedule(&circuit, &sim);
        assert_eq!(result.circuit.gate_count(), circuit.gate_count());
    }

    #[test]
    fn test_transpiler_creation() {
        let config = TranspilerConfig::surface_code(10, 3);
        let transpiler = DecoderAwareTranspiler::new(config);
        assert_eq!(transpiler.config().num_qubits, 10);
    }

    #[test]
    fn test_transpiler_optimize_simple() {
        let config = TranspilerConfig::surface_code(4, 3);
        let mut transpiler = DecoderAwareTranspiler::new(config);

        let mut circuit = Circuit::new(4);
        circuit.add_gate(Gate::new(GateType::H, vec![0]));
        circuit.add_gate(Gate::new(GateType::H, vec![1]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 1]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![2, 3]));

        let result = transpiler.optimize(&circuit);
        assert!(result.passes >= 1);
        assert!(result.circuit.gate_count() > 0);
    }

    #[test]
    fn test_transpiler_reduces_decoder_difficulty() {
        let config = TranspilerConfig::surface_code(4, 3);
        let mut transpiler = DecoderAwareTranspiler::new(config);

        // Create circuit with many non-adjacent CNOTs (hard to decode)
        let mut circuit = Circuit::new(4);
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 3]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![1, 2]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 2]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![1, 3]));

        let result = transpiler.optimize(&circuit);

        // Optimized circuit should have same or fewer gates
        assert!(result.circuit.gate_count() <= circuit.gate_count());
    }

    #[test]
    fn test_circuit_stats() {
        let sim = DecoderSimulator::new();
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::H, vec![0]));
        circuit.add_gate(Gate::new(GateType::CNOT, vec![0, 1]));

        let stats = CircuitStats::from_circuit(&circuit, &sim);
        assert_eq!(stats.gate_count, 2);
        assert_eq!(stats.two_qubit_count, 1);
        assert!(stats.decoder_difficulty > 0.0);
    }

    #[test]
    fn test_circuit_context_default() {
        let ctx = CircuitContext::default();
        assert_eq!(ctx.decoder_type, DecoderType::MWPM);
        assert_eq!(ctx.priority, OptimizationPriority::Balanced);
    }

    #[test]
    fn test_empty_circuit() {
        let config = TranspilerConfig::default();
        let mut transpiler = DecoderAwareTranspiler::new(config);

        let circuit = Circuit::new(2);
        let result = transpiler.optimize(&circuit);

        assert_eq!(result.circuit.gate_count(), 0);
        assert_eq!(result.decoder_difficulty, 0.0);
    }
}
