//! Auto-Backend Selection for Quantum Simulation
//!
//! Intelligently selects the best execution backend based on circuit properties:
//! - MPS for large circuits with low entanglement (30+ qubits)
//! - GPU for medium circuits with high gate count
//! - Fused CPU for small-to-medium circuits
//! - CPU sequential for tiny circuits
//!
//! # Usage
//!
//! ```ignore
//! use nqpu_metal::auto_backend::{AutoBackend, BackendType};
//!
//! let backend = AutoBackend::select(&gates);
//! match backend {
//!     BackendType::MPS => { /* use MPS */ }
//!     BackendType::MetalGPU => { /* use GPU */ }
//!     BackendType::Fused => { /* use fused CPU */ }
//!     BackendType::CPU => { /* use sequential CPU */ }
//! }
//! ```

use crate::gates::{Gate, GateType};

/// Backend type for circuit execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Matrix Product State (for large, low-entanglement circuits)
    MPS,
    /// Multi-node distributed simulation (for very large states, 30+ qubits)
    Distributed,
    /// Metal GPU (for macOS medium-to-large circuits)
    MetalGPU,
    /// CUDA GPU (for Linux/Windows medium-to-large circuits)
    CudaGPU,
    /// F32 + fused CPU execution (for memory-bandwidth constrained circuits)
    F32Fused,
    /// Fused CPU execution (for small-to-medium circuits)
    Fused,
    /// Sequential CPU execution (for tiny circuits)
    CPU,
    /// Trapped-ion backend (Molmer-Sorensen native gate set)
    TrappedIon,
}

impl BackendType {
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::MPS => "MPS",
            BackendType::Distributed => "DistributedMPI",
            BackendType::MetalGPU => "MetalGPU",
            BackendType::CudaGPU => "CudaGPU",
            BackendType::F32Fused => "F32FusionCPU",
            BackendType::Fused => "FusedCPU",
            BackendType::CPU => "CPU",
            BackendType::TrappedIon => "TrappedIon",
        }
    }
}

/// Circuit analysis results.
#[derive(Debug, Clone)]
pub struct CircuitAnalysis {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of gates
    pub num_gates: usize,
    /// Estimated depth
    pub depth: usize,
    /// Entanglement estimate (0.0 = product state, 1.0 = maximally entangled)
    pub entanglement_estimate: f64,
    /// Whether circuit is Clifford (can simulate efficiently)
    pub is_clifford: bool,
    /// Recommended backend
    pub recommended_backend: BackendType,
    /// Reasoning for recommendation
    pub reasoning: String,
}

/// Auto-backend selector.
pub struct AutoBackend {
    /// Qubit threshold for MPS consideration
    mps_qubit_threshold: usize,
    /// Entanglement threshold for MPS (below this = use MPS)
    mps_entanglement_threshold: f64,
    /// Gate count threshold for GPU consideration
    gpu_gate_threshold: usize,
    /// Gate count threshold for fusion consideration
    fusion_gate_threshold: usize,
}

impl Default for AutoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoBackend {
    /// Create a new auto-backend selector with default thresholds.
    pub fn new() -> Self {
        AutoBackend {
            mps_qubit_threshold: 25,
            mps_entanglement_threshold: 0.3,
            gpu_gate_threshold: 50,
            fusion_gate_threshold: 5,
        }
    }

    /// Create a new auto-backend selector with custom thresholds.
    pub fn with_thresholds(
        mps_qubit_threshold: usize,
        mps_entanglement_threshold: f64,
        gpu_gate_threshold: usize,
        fusion_gate_threshold: usize,
    ) -> Self {
        AutoBackend {
            mps_qubit_threshold,
            mps_entanglement_threshold,
            gpu_gate_threshold,
            fusion_gate_threshold,
        }
    }

    /// Analyze a circuit and recommend the best backend.
    pub fn analyze(&self, gates: &[Gate]) -> CircuitAnalysis {
        if gates.is_empty() {
            return CircuitAnalysis {
                num_qubits: 0,
                num_gates: 0,
                depth: 0,
                entanglement_estimate: 0.0,
                is_clifford: true,
                recommended_backend: BackendType::CPU,
                reasoning: "Empty circuit".to_string(),
            };
        }

        let num_qubits = self.count_qubits(gates);
        let num_gates = gates.len();
        let depth = self.estimate_depth(gates);
        let is_clifford = self.is_clifford_circuit(gates);
        let entanglement_estimate = self.estimate_entanglement(gates);

        // Determine backend
        let (recommended_backend, reasoning) = self.select_backend(
            num_qubits,
            num_gates,
            depth,
            is_clifford,
            entanglement_estimate,
        );

        CircuitAnalysis {
            num_qubits,
            num_gates,
            depth,
            entanglement_estimate,
            is_clifford,
            recommended_backend,
            reasoning,
        }
    }

    /// Select the best backend for a circuit.
    fn select_backend(
        &self,
        num_qubits: usize,
        num_gates: usize,
        _depth: usize,
        _is_clifford: bool,
        entanglement_estimate: f64,
    ) -> (BackendType, String) {
        // Distributed for very large circuits (30+ qubits)
        if num_qubits >= 30 {
            return (
                BackendType::Distributed,
                format!(
                    "Very large circuit ({} qubits), multi-node distributed execution required",
                    num_qubits
                ),
            );
        }

        // MPS for large circuits with low entanglement
        if num_qubits >= self.mps_qubit_threshold
            && entanglement_estimate < self.mps_entanglement_threshold
        {
            return (
                BackendType::MPS,
                format!(
                    "Large circuit ({} qubits) with low entanglement ({:.2}), MPS efficient",
                    num_qubits, entanglement_estimate
                ),
            );
        }

        // GPU for circuits with many gates (if available)
        #[cfg(target_os = "macos")]
        {
            if num_gates >= self.gpu_gate_threshold {
                return (
                    BackendType::MetalGPU,
                    format!(
                        "Large circuit ({} gates), Metal GPU acceleration beneficial",
                        num_gates
                    ),
                );
            }
        }

        // CUDA for non-macOS systems (if feature enabled)
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            if num_gates >= self.gpu_gate_threshold {
                return (
                    BackendType::CudaGPU,
                    format!(
                        "Large circuit ({} gates), CUDA acceleration beneficial",
                        num_gates
                    ),
                );
            }
        }

        // Fused CPU for medium circuits
        if num_qubits >= 10 && num_gates >= self.fusion_gate_threshold * 2 {
            return (
                BackendType::F32Fused,
                format!(
                    "Medium-large circuit ({} qubits, {} gates), f32+fusion reduces memory traffic",
                    num_qubits, num_gates
                ),
            );
        }

        // Fused CPU for medium circuits
        if num_gates >= self.fusion_gate_threshold {
            return (
                BackendType::Fused,
                format!(
                    "Medium circuit ({} gates), fusion reduces overhead",
                    num_gates
                ),
            );
        }

        // CPU sequential for tiny circuits
        (
            BackendType::CPU,
            format!(
                "Small circuit ({} gates), sequential execution fastest",
                num_gates
            ),
        )
    }

    /// Quick backend selection (returns only the backend type).
    pub fn select(&self, gates: &[Gate]) -> BackendType {
        self.analyze(gates).recommended_backend
    }

    /// Count unique qubits used in the circuit.
    fn count_qubits(&self, gates: &[Gate]) -> usize {
        let mut qubits = std::collections::HashSet::new();
        for gate in gates {
            for &q in &gate.targets {
                qubits.insert(q);
            }
            for &q in &gate.controls {
                qubits.insert(q);
            }
        }
        qubits.len().max(1)
    }

    /// Estimate circuit depth.
    fn estimate_depth(&self, gates: &[Gate]) -> usize {
        let mut qubit_last_use: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut depth = 0;

        for gate in gates {
            let mut gate_qubits = Vec::new();
            for &q in &gate.targets {
                gate_qubits.push(q);
            }
            for &q in &gate.controls {
                gate_qubits.push(q);
            }

            let min_layer = gate_qubits
                .iter()
                .filter_map(|q| qubit_last_use.get(q))
                .max()
                .copied()
                .unwrap_or(0);

            let layer = min_layer + 1;
            for q in gate_qubits {
                qubit_last_use.insert(q, layer);
            }
            depth = depth.max(layer);
        }

        depth
    }

    /// Check if circuit is Clifford-only.
    fn is_clifford_circuit(&self, gates: &[Gate]) -> bool {
        gates.iter().all(|gate| match &gate.gate_type {
            GateType::H
            | GateType::S
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::CNOT
            | GateType::CZ
            | GateType::SWAP
            | GateType::ISWAP => true,
            _ => false,
        })
    }

    /// Estimate circuit entanglement based on gate patterns.
    ///
    /// Returns 0.0 for product states, 1.0 for maximally entangled.
    fn estimate_entanglement(&self, gates: &[Gate]) -> f64 {
        if gates.is_empty() {
            return 0.0;
        }

        let mut entangling_gates = 0.0_f64;
        let mut total_gates = 0;

        for gate in gates {
            total_gates += 1;
            match &gate.gate_type {
                // Two-qubit gates create entanglement
                GateType::CNOT | GateType::CZ | GateType::CR(_) | GateType::SWAP => {
                    entangling_gates += 1.0;
                }
                // Hadamard can create superposition (potential entanglement)
                GateType::H => {
                    // H followed by 2Q gate = entanglement
                    entangling_gates += 0.5;
                }
                _ => {}
            }
        }

        if total_gates == 0 {
            return 0.0;
        }

        // Normalize by qubit count (more qubits need more entanglement to be "highly entangled")
        let num_qubits = self.count_qubits(gates);

        // Base entanglement from entangling gate ratio
        let base_entanglement = (entangling_gates as f64 / total_gates as f64).min(1.0);

        // Scale by qubit count (2-qubit circuits max entangled at 0.5, 10-qubit at 1.0)
        let qubit_factor = (num_qubits as f64 / 10.0).min(1.0);

        (base_entanglement * qubit_factor).min(1.0)
    }
}

// ============================================================================
// EXECUTION HELPERS
// ============================================================================

/// Execution configuration with auto-backend selection.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Whether to enable auto-backend selection
    pub auto_backend: bool,
    /// Backend override (None = auto-select)
    pub backend_override: Option<BackendType>,
    /// Bond dimension for MPS (None = auto-select)
    pub mps_bond_dim: Option<usize>,
    /// Whether to use thermal-aware GPU scheduling
    pub thermal_aware: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            auto_backend: true,
            backend_override: None,
            mps_bond_dim: None,
            thermal_aware: true,
        }
    }
}

impl ExecutionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.backend_override = Some(backend);
        self.auto_backend = false;
        self
    }

    pub fn with_mps_bond_dim(mut self, bond_dim: usize) -> Self {
        self.mps_bond_dim = Some(bond_dim);
        self
    }

    pub fn with_thermal_aware(mut self, aware: bool) -> Self {
        self.thermal_aware = aware;
        self
    }

    pub fn with_auto_backend(mut self, auto: bool) -> Self {
        self.auto_backend = auto;
        self
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GateOperations;

    #[test]
    fn test_backend_names() {
        assert_eq!(BackendType::MPS.name(), "MPS");
        assert_eq!(BackendType::Distributed.name(), "DistributedMPI");
        assert_eq!(BackendType::MetalGPU.name(), "MetalGPU");
        assert_eq!(BackendType::CudaGPU.name(), "CudaGPU");
        assert_eq!(BackendType::F32Fused.name(), "F32FusionCPU");
        assert_eq!(BackendType::Fused.name(), "FusedCPU");
        assert_eq!(BackendType::CPU.name(), "CPU");
    }

    #[test]
    fn test_auto_backend_small_circuit() {
        let selector = AutoBackend::new();
        let gates = vec![Gate::h(0), Gate::h(1)];
        let backend = selector.select(&gates);
        assert_eq!(backend, BackendType::CPU); // < 5 gates
    }

    #[test]
    fn test_auto_backend_medium_circuit() {
        let selector = AutoBackend::new();
        let mut gates = Vec::new();
        for i in 0..20 {
            gates.push(Gate::h(i % 5));
        }
        let backend = selector.select(&gates);
        assert!(
            backend == BackendType::Fused
                || backend == BackendType::MetalGPU
                || backend == BackendType::CudaGPU
                || backend == BackendType::F32Fused
        );
    }

    #[test]
    fn test_qubit_counting() {
        let selector = AutoBackend::new();
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::cnot(1, 2)];
        let count = selector.count_qubits(&gates);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_depth_estimation() {
        let selector = AutoBackend::new();
        // Sequential circuit on same qubit = depth = num_gates
        let gates = vec![Gate::h(0), Gate::x(0), Gate::y(0)];
        let depth = selector.estimate_depth(&gates);
        assert_eq!(depth, 3);
    }

    #[test]
    fn test_depth_estimation_parallel() {
        let selector = AutoBackend::new();
        // Parallel gates = depth = 1
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2)];
        let depth = selector.estimate_depth(&gates);
        assert_eq!(depth, 1);
    }

    #[test]
    fn test_clifford_detection() {
        let selector = AutoBackend::new();
        let clifford_gates = vec![Gate::h(0), Gate::s(0), Gate::cnot(0, 1)];
        assert!(selector.is_clifford_circuit(&clifford_gates));

        let non_clifford_gates = vec![
            Gate::h(0),
            Gate::t(0), // T is non-Clifford
        ];
        assert!(!selector.is_clifford_circuit(&non_clifford_gates));
    }

    #[test]
    fn test_entanglement_estimation() {
        let selector = AutoBackend::new();

        // Product state = 0 entanglement
        let product_gates = vec![Gate::h(0), Gate::h(1)];
        let ent1 = selector.estimate_entanglement(&product_gates);
        assert!(ent1 < 0.3); // Low entanglement

        // Highly entangled = high entanglement
        let mut entangled_gates = Vec::new();
        for i in 0..10 {
            entangled_gates.push(Gate::h(i));
            if i + 1 < 10 {
                entangled_gates.push(Gate::cnot(i, i + 1));
            }
        }
        let ent2 = selector.estimate_entanglement(&entangled_gates);
        assert!(ent2 > 0.5); // High entanglement
    }

    #[test]
    fn test_full_analysis() {
        let selector = AutoBackend::new();
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::h(2),
            Gate::h(3),
            Gate::cnot(2, 3),
        ];

        let analysis = selector.analyze(&gates);
        assert!(analysis.num_qubits > 0);
        assert!(analysis.depth > 0);
        assert!(analysis.reasoning.len() > 0);
    }

    #[test]
    fn test_execution_config() {
        let config = ExecutionConfig::new()
            .with_backend(BackendType::MetalGPU)
            .with_mps_bond_dim(16)
            .with_thermal_aware(true);

        assert!(!config.auto_backend);
        assert_eq!(config.backend_override, Some(BackendType::MetalGPU));
        assert_eq!(config.mps_bond_dim, Some(16));
        assert!(config.thermal_aware);
    }
}
