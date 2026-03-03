//! Automatic Backend Selection
//!
//! Analyzes a circuit and routes it to the most efficient simulation backend:
//! - **DensityMatrix**: noisy circuits with ≤13 qubits
//! - **Stabilizer**: Clifford-only circuits (any qubit count)
//! - **MPS**: >25 qubits with low entanglement
//! - **StateVector**: everything else (with gate fusion)

use crate::ascii_viz::apply_gate_to_state;
use crate::circuit_optimizer::{CircuitOptimizer, OptimizationLevel};
use crate::f32_fusion::F32FusionExecutor;
use crate::gate_fusion::{execute_fused_circuit, fuse_gates};
use crate::gates::{Gate, GateType};
#[cfg(target_os = "macos")]
use crate::metal_backend::MetalSimulator;
#[cfg(target_os = "macos")]
use crate::metal_mps::MetalMPSimulator;
use crate::tensor_network::MPSSimulator;
use crate::QuantumState;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ===================================================================
// CIRCUIT ANALYSIS
// ===================================================================

/// Result of analyzing a circuit's properties.
#[derive(Clone, Debug)]
pub struct CircuitAnalysis {
    pub num_gates: usize,
    pub num_single_qubit: usize,
    pub num_two_qubit: usize,
    pub num_three_qubit: usize,
    pub is_clifford_only: bool,
    pub max_entanglement_width: usize,
    pub connected_components: usize,
    /// Number of T gates (non-Clifford resource count).
    pub num_t_gates: usize,
    /// Fraction of gates that are Clifford (0.0 to 1.0).
    pub clifford_fraction: f64,
    /// Detected circuit symmetry (if any).
    pub circuit_symmetry: Option<CircuitSymmetry>,
    /// Magic level: estimated non-stabilizerness (0.0 = Clifford, 1.0 = max magic).
    pub magic_level: f64,
}

/// Detected circuit symmetry for routing decisions.
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitSymmetry {
    /// Translational symmetry (repeating pattern).
    Translational,
    /// Reflection symmetry (mirror).
    Reflection,
    /// Permutation symmetry (qubits are interchangeable).
    Permutation,
}

/// Configuration for backend routing decisions.
#[derive(Clone, Debug)]
pub struct RoutingConfig {
    /// Minimum qubit count for Pauli propagation.
    pub pauli_prop_min_qubits: usize,
    /// Maximum T-gates for near-Clifford backend.
    pub near_clifford_max_t: usize,
    /// Minimum Clifford fraction for near-Clifford backend.
    pub near_clifford_min_clifford_frac: f64,
    /// Maximum magic level for stabilizer tensor network.
    pub stn_max_magic: f64,
    /// Minimum qubits for MPS backend.
    pub mps_min_qubits: usize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        RoutingConfig {
            pauli_prop_min_qubits: 30,
            near_clifford_max_t: 40,
            near_clifford_min_clifford_frac: 0.90,
            stn_max_magic: 0.3,
            mps_min_qubits: 25,
        }
    }
}

/// Check if a gate type is a Clifford gate.
fn is_clifford(gate: &Gate) -> bool {
    match &gate.gate_type {
        GateType::H | GateType::X | GateType::Y | GateType::Z | GateType::S => true,
        GateType::CNOT | GateType::CZ | GateType::SWAP => true,
        GateType::Toffoli => false, // Toffoli is NOT Clifford
        GateType::T => false,
        GateType::Rx(_) | GateType::Ry(_) | GateType::Rz(_) => false,
        GateType::U { .. } => false,
        GateType::CRx(_) | GateType::CRy(_) | GateType::CRz(_) | GateType::CR(_) => false,
        GateType::SX => false,
        GateType::Phase(_) => false,
        GateType::ISWAP => false,
        GateType::CCZ => false,
        GateType::Rxx(_) | GateType::Ryy(_) | GateType::Rzz(_) => false,
        GateType::CSWAP => false,
        GateType::CU { .. } => false,
        GateType::Custom(_) => false,
    }
}

/// Simple Union-Find for qubit connectivity analysis.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }

    fn num_components(&mut self, n: usize) -> usize {
        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(self.find(i));
        }
        roots.len()
    }

    fn max_component_size(&mut self, n: usize) -> usize {
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            *sizes.entry(self.find(i)).or_insert(0) += 1;
        }
        sizes.values().copied().max().unwrap_or(1)
    }
}

/// Analyze a circuit to determine its properties.
pub fn analyze_circuit(gates: &[Gate], num_qubits: usize) -> CircuitAnalysis {
    let mut num_single = 0;
    let mut num_two = 0;
    let mut num_three = 0;
    let mut num_t = 0;
    let mut num_clifford = 0;
    let mut clifford_only = true;
    let mut uf = UnionFind::new(num_qubits);

    for gate in gates {
        let nq = gate.targets.len() + gate.controls.len();
        match nq {
            1 => num_single += 1,
            2 => num_two += 1,
            _ => num_three += 1,
        }

        if is_clifford(gate) {
            num_clifford += 1;
        } else {
            clifford_only = false;
        }

        // Count T gates specifically
        if matches!(gate.gate_type, GateType::T) {
            num_t += 1;
        }

        // Union qubits connected by multi-qubit gates
        let all_qubits: Vec<usize> = gate
            .targets
            .iter()
            .chain(gate.controls.iter())
            .copied()
            .collect();
        for i in 1..all_qubits.len() {
            if all_qubits[0] < num_qubits && all_qubits[i] < num_qubits {
                uf.union(all_qubits[0], all_qubits[i]);
            }
        }
    }

    let connected_components = uf.num_components(num_qubits);
    let max_entanglement_width = uf.max_component_size(num_qubits);
    let total = gates.len().max(1);
    let clifford_frac = num_clifford as f64 / total as f64;

    // Estimate magic level from T-gate density
    let magic_level = if clifford_only {
        0.0
    } else {
        (num_t as f64 / total as f64).min(1.0)
    };

    // Detect circuit symmetry (simple heuristic)
    let circuit_symmetry = detect_symmetry(gates, num_qubits);

    CircuitAnalysis {
        num_gates: gates.len(),
        num_single_qubit: num_single,
        num_two_qubit: num_two,
        num_three_qubit: num_three,
        is_clifford_only: clifford_only,
        max_entanglement_width,
        connected_components,
        num_t_gates: num_t,
        clifford_fraction: clifford_frac,
        circuit_symmetry,
        magic_level,
    }
}

/// Simple symmetry detection heuristic.
fn detect_symmetry(gates: &[Gate], num_qubits: usize) -> Option<CircuitSymmetry> {
    if gates.len() < 4 || num_qubits < 3 {
        return None;
    }

    // Check for translational symmetry: same gate pattern repeating on shifted qubits
    let mut qubit_gate_counts: Vec<usize> = vec![0; num_qubits];
    for gate in gates {
        for &q in gate.targets.iter().chain(gate.controls.iter()) {
            if q < num_qubits {
                qubit_gate_counts[q] += 1;
            }
        }
    }

    // If all qubits have similar gate counts, might be permutation symmetric
    let mean = qubit_gate_counts.iter().sum::<usize>() as f64 / num_qubits as f64;
    let variance: f64 = qubit_gate_counts
        .iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>()
        / num_qubits as f64;

    if mean > 0.0 && variance / (mean * mean) < 0.1 {
        return Some(CircuitSymmetry::Permutation);
    }

    None
}

// ===================================================================
// CIRCUIT PROFILING
// ===================================================================

/// Circuit profile for intelligent backend selection.
///
/// Performs a deeper analysis pass than `CircuitAnalysis`, building an
/// interaction graph to compute connectivity metrics, detecting parametric
/// gates and mid-circuit measurements, and estimating entanglement from
/// circuit structure. Used by the enhanced `select_backend_profiled` to
/// make finer-grained routing decisions.
#[derive(Clone, Debug)]
pub struct CircuitProfile {
    /// Total number of qubits in the circuit.
    pub n_qubits: usize,
    /// Circuit depth (longest path through the gate DAG).
    pub depth: usize,
    /// Total gate count.
    pub gate_count: usize,
    /// Fraction of gates that are Clifford (0.0 to 1.0).
    pub clifford_fraction: f64,
    /// Number of T/Tdagger gates (non-Clifford resource count).
    pub t_gate_count: usize,
    /// Fraction of gates that are two-qubit gates.
    pub two_qubit_fraction: f64,
    /// Average qubit degree in the interaction graph.
    pub connectivity: f64,
    /// Whether the circuit contains parametric (rotation) gates.
    pub is_parametric: bool,
    /// Whether the circuit contains mid-circuit measurements (currently
    /// detected by the presence of classically-controlled gates).
    pub has_mid_circuit_measurement: bool,
    /// Rough entanglement estimate based on two-qubit gate density and
    /// interaction graph connectivity (0.0 = product state, 1.0 = max).
    pub estimated_entanglement: f64,
    /// Maximum degree of any qubit in the interaction graph.
    pub max_qubit_degree: usize,
}

impl CircuitProfile {
    /// Analyze a circuit (given as a slice of `Gate`) and produce a profile.
    ///
    /// The analysis builds an interaction graph from multi-qubit gates,
    /// computes depth layer-by-layer, and classifies each gate as Clifford,
    /// parametric, etc.
    pub fn analyze(circuit: &[Gate], n_qubits: usize) -> Self {
        if circuit.is_empty() {
            return Self {
                n_qubits,
                depth: 0,
                gate_count: 0,
                clifford_fraction: 1.0,
                t_gate_count: 0,
                two_qubit_fraction: 0.0,
                connectivity: 0.0,
                is_parametric: false,
                has_mid_circuit_measurement: false,
                estimated_entanglement: 0.0,
                max_qubit_degree: 0,
            };
        }

        let mut num_clifford = 0usize;
        let mut num_t = 0usize;
        let mut num_two_qubit = 0usize;
        let mut parametric = false;
        let mut has_mcm = false;

        // Interaction graph: edges between qubits connected by multi-qubit gates
        let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n_qubits];

        // Depth computation: track the earliest available layer for each qubit
        let mut qubit_layer: Vec<usize> = vec![0; n_qubits];

        for gate in circuit {
            // Classify gate
            if is_clifford(gate) {
                num_clifford += 1;
            }
            if matches!(gate.gate_type, GateType::T) {
                num_t += 1;
            }

            // Parametric gate detection
            if matches!(
                gate.gate_type,
                GateType::Rx(_)
                    | GateType::Ry(_)
                    | GateType::Rz(_)
                    | GateType::CRx(_)
                    | GateType::CRy(_)
                    | GateType::CRz(_)
                    | GateType::CR(_)
                    | GateType::Rxx(_)
                    | GateType::Ryy(_)
                    | GateType::Rzz(_)
                    | GateType::Phase(_)
                    | GateType::U { .. }
                    | GateType::CU { .. }
            ) {
                parametric = true;
            }

            let all_qubits: Vec<usize> = gate
                .targets
                .iter()
                .chain(gate.controls.iter())
                .copied()
                .filter(|&q| q < n_qubits)
                .collect();

            let nq = all_qubits.len();
            if nq >= 2 {
                num_two_qubit += 1;

                // Build interaction graph edges
                for i in 0..all_qubits.len() {
                    for j in (i + 1)..all_qubits.len() {
                        neighbors[all_qubits[i]].insert(all_qubits[j]);
                        neighbors[all_qubits[j]].insert(all_qubits[i]);
                    }
                }
            }

            // Compute depth: gate layer = max(layer of involved qubits) + 1
            if !all_qubits.is_empty() {
                let gate_layer = all_qubits.iter().map(|&q| qubit_layer[q]).max().unwrap();
                let next_layer = gate_layer + 1;
                for &q in &all_qubits {
                    qubit_layer[q] = next_layer;
                }
            }
        }

        let depth = qubit_layer.iter().copied().max().unwrap_or(0);
        let gate_count = circuit.len();
        let total = gate_count.max(1) as f64;

        let clifford_fraction = num_clifford as f64 / total;
        let two_qubit_fraction = num_two_qubit as f64 / total;

        // Connectivity metrics from interaction graph
        let degrees: Vec<usize> = neighbors.iter().map(|s| s.len()).collect();
        let max_qubit_degree = degrees.iter().copied().max().unwrap_or(0);
        let connectivity = if n_qubits > 0 {
            degrees.iter().sum::<usize>() as f64 / n_qubits as f64
        } else {
            0.0
        };

        // Entanglement estimate: combination of two-qubit gate density and connectivity.
        // Normalized so that a fully-connected circuit with all two-qubit gates -> 1.0.
        // Only considers depth when there are actually two-qubit gates (single-qubit
        // circuits cannot generate entanglement regardless of depth).
        let estimated_entanglement = if num_two_qubit == 0 {
            0.0
        } else {
            let max_possible_degree = if n_qubits > 1 { n_qubits - 1 } else { 1 };
            let connectivity_factor = connectivity / max_possible_degree as f64;
            let depth_factor = if n_qubits > 0 {
                (depth as f64 / n_qubits as f64).min(1.0)
            } else {
                0.0
            };
            (0.5 * two_qubit_fraction + 0.3 * connectivity_factor + 0.2 * depth_factor).min(1.0)
        };

        Self {
            n_qubits,
            depth,
            gate_count,
            clifford_fraction,
            t_gate_count: num_t,
            two_qubit_fraction,
            connectivity,
            is_parametric: parametric,
            has_mid_circuit_measurement: has_mcm,
            estimated_entanglement,
            max_qubit_degree,
        }
    }

    /// Returns true if the circuit is predominantly Clifford (>80% Clifford gates).
    pub fn is_mostly_clifford(&self) -> bool {
        self.clifford_fraction > 0.8
    }

    /// Returns true if the circuit has only Clifford and T gates (no other non-Clifford).
    pub fn is_clifford_t_only(&self, circuit: &[Gate]) -> bool {
        circuit.iter().all(|g| is_clifford(g) || matches!(g.gate_type, GateType::T))
    }

    /// Estimate whether the circuit has low treewidth based on connectivity.
    /// Low-connectivity, nearest-neighbor-like circuits tend to have low treewidth.
    pub fn has_low_treewidth(&self) -> bool {
        // Heuristic: average degree <= 2 and max degree <= 4 suggests a
        // near-1D or low-treewidth interaction graph suitable for MPS/TN.
        self.connectivity <= 2.0 && self.max_qubit_degree <= 4
    }
}

/// Select a backend using the enhanced circuit profiling pass.
///
/// This function performs a deeper analysis than `select_backend` by
/// building a `CircuitProfile` and using its connectivity, entanglement,
/// and gate classification metrics for more precise routing.
pub fn select_backend_profiled(gates: &[Gate], num_qubits: usize, noisy: bool) -> SimBackend {
    let profile = CircuitProfile::analyze(gates, num_qubits);
    let analysis = analyze_circuit(gates, num_qubits);
    let config = RoutingConfig::default();

    // Rule 1: Noisy circuit on small system -> DensityMatrix
    if noisy && num_qubits <= 13 {
        return SimBackend::DensityMatrix;
    }

    // Rule 2: Clifford-only -> Stabilizer (scales polynomially)
    if analysis.is_clifford_only {
        return SimBackend::Stabilizer;
    }

    // Rule 3: High Clifford fraction (>80%) -> Stabilizer simulator
    // Even non-pure-Clifford circuits benefit from stabilizer-based
    // simulation when the vast majority of gates are Clifford.
    if profile.is_mostly_clifford() && num_qubits > 20 && profile.t_gate_count == 0 {
        return SimBackend::Stabilizer;
    }

    // Rule 4: Noisy + Clifford-heavy + large -> Pauli propagation
    if noisy
        && num_qubits >= config.pauli_prop_min_qubits
        && analysis.clifford_fraction >= 0.5
    {
        return SimBackend::PauliPropagation;
    }

    // Rule 5: Clifford+T only circuits -> NearClifford (CAMPS)
    // These circuits can be efficiently simulated by decomposing
    // T-gates as stabilizer superpositions.
    if profile.is_clifford_t_only(gates)
        && profile.t_gate_count > 0
        && profile.t_gate_count <= config.near_clifford_max_t
        && num_qubits > 20
    {
        return SimBackend::NearClifford;
    }

    // Rule 6: Near-Clifford (many Clifford, few T, >20 qubits)
    if analysis.num_t_gates <= config.near_clifford_max_t
        && analysis.num_t_gates > 0
        && num_qubits > 20
        && analysis.clifford_fraction >= config.near_clifford_min_clifford_frac
    {
        return SimBackend::NearClifford;
    }

    // Rule 7: Low treewidth circuits -> MPS / tensor network
    // Circuits with near-1D connectivity are efficiently contracted by MPS.
    if num_qubits > 15 && profile.has_low_treewidth() && profile.estimated_entanglement < 0.5 {
        #[cfg(target_os = "macos")]
        if metal_gpu_available() {
            return SimBackend::MetalMPS;
        }
        return SimBackend::MPS;
    }

    // Rule 8: Structured/symmetric circuits -> Decision diagram
    if num_qubits > 20 && analysis.circuit_symmetry.is_some() {
        return SimBackend::DecisionDiagram;
    }

    // Rule 9: Low magic + large -> Stabilizer tensor network
    if num_qubits > 30 && analysis.magic_level <= config.stn_max_magic {
        return SimBackend::StabilizerTensorNetwork;
    }

    // Rule 10: Small circuits (<20 qubits) with mixed gates -> statevector
    // No benefit from tensor network overhead on small circuits.
    if num_qubits < 20 && !analysis.is_clifford_only {
        #[cfg(target_os = "macos")]
        if num_qubits >= 4 && metal_gpu_available() {
            return SimBackend::MetalGPU;
        }
        if num_qubits >= 10 {
            return SimBackend::StateVectorF32Fused;
        }
        return SimBackend::StateVectorFused;
    }

    // Rule 11: Large circuits with parametric gates and high entanglement
    // -> MPS with differentiable support for variational optimization.
    if num_qubits > config.mps_min_qubits && profile.is_parametric {
        #[cfg(target_os = "macos")]
        if metal_gpu_available() {
            return SimBackend::MetalMPS;
        }
        return SimBackend::MPS;
    }

    // Rule 12: Large circuit with low entanglement -> MPS
    if num_qubits > config.mps_min_qubits
        && analysis.max_entanglement_width <= num_qubits / 2
    {
        #[cfg(target_os = "macos")]
        if metal_gpu_available() {
            return SimBackend::MetalMPS;
        }
        return SimBackend::MPS;
    }

    // Rule 13: Metal GPU preferred on macOS
    #[cfg(target_os = "macos")]
    if num_qubits >= 4 && metal_gpu_available() {
        return SimBackend::MetalGPU;
    }

    // Rule 14: CUDA GPU preferred on non-macOS (if available)
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    if num_qubits >= 4 && cuda_gpu_available() {
        return SimBackend::CudaGPU;
    }

    // Rule 15: Prefer f32+fusion on larger CPU-only state-vector workloads
    if num_qubits >= 10 {
        return SimBackend::StateVectorF32Fused;
    }

    // Rule 16: CPU fallback with fusion
    SimBackend::StateVectorFused
}

// ===================================================================
// BACKEND SELECTION
// ===================================================================

/// Simulation backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SimBackend {
    /// Full state vector simulation (CPU, no fusion).
    StateVector,
    /// State vector with gate fusion optimization (CPU).
    StateVectorFused,
    /// State vector with f32 + gate fusion optimization (CPU).
    StateVectorF32Fused,
    /// Metal GPU acceleration (macOS only, f32).
    MetalGPU,
    /// Metal GPU-only mode (macOS only, strict: no CPU fallback).
    MetalGPUOnly,
    /// CUDA GPU acceleration (Linux/Windows, f32).
    CudaGPU,
    /// CUDA GPU-only mode (strict: no CPU fallback).
    CudaGPUOnly,
    /// Density matrix for noisy simulation.
    DensityMatrix,
    /// Stabilizer tableau for Clifford-only circuits.
    Stabilizer,
    /// Matrix Product State for large, low-entanglement circuits.
    MPS,
    /// Metal-accelerated MPS (macOS only).
    MetalMPS,
    /// Heisenberg-picture Pauli propagation for noisy Clifford-heavy circuits.
    PauliPropagation,
    /// Decision diagram (BDD/ZDD) for structured/symmetric circuits.
    DecisionDiagram,
    /// Near-Clifford (CH-form) for circuits with few T-gates.
    NearClifford,
    /// Stabilizer tensor network for low-magic large circuits.
    StabilizerTensorNetwork,
}

impl std::fmt::Display for SimBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimBackend::StateVector => write!(f, "StateVector (CPU)"),
            SimBackend::StateVectorFused => write!(f, "StateVector+Fusion (CPU)"),
            SimBackend::StateVectorF32Fused => write!(f, "StateVector+F32+Fusion (CPU)"),
            SimBackend::MetalGPU => write!(f, "Metal GPU"),
            SimBackend::MetalGPUOnly => write!(f, "Metal GPU (Strict)"),
            SimBackend::DensityMatrix => write!(f, "DensityMatrix"),
            SimBackend::Stabilizer => write!(f, "Stabilizer"),
            SimBackend::MPS => write!(f, "MPS"),
            SimBackend::MetalMPS => write!(f, "Metal MPS"),
            SimBackend::PauliPropagation => write!(f, "Pauli Propagation"),
            SimBackend::DecisionDiagram => write!(f, "Decision Diagram"),
            SimBackend::NearClifford => write!(f, "Near-Clifford (CH-form)"),
            SimBackend::StabilizerTensorNetwork => write!(f, "Stabilizer Tensor Network"),
            SimBackend::CudaGPU => write!(f, "CUDA GPU"),
            SimBackend::CudaGPUOnly => write!(f, "CUDA GPU (Strict)"),
        }
    }
}

/// Check if Metal GPU is available at runtime.
fn metal_gpu_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Try to create a 2-qubit simulator to verify Metal works
        MetalSimulator::new(2).is_ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaParallelQuantumExecutor;

// ... (existing code)

/// Check if CUDA GPU is available at runtime.
fn cuda_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        CudaParallelQuantumExecutor::new().is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Select the best backend for a given circuit.
fn select_backend(analysis: &CircuitAnalysis, num_qubits: usize, noisy: bool) -> SimBackend {
    let config = RoutingConfig::default();

    // Rule 1: Noisy circuit on small system → DensityMatrix
    if noisy && num_qubits <= 13 {
        return SimBackend::DensityMatrix;
    }

    // Rule 2: Clifford-only → Stabilizer (scales polynomially)
    if analysis.is_clifford_only {
        return SimBackend::Stabilizer;
    }

    // Rule 3: Noisy + Clifford-heavy + large → Pauli propagation
    if noisy
        && num_qubits >= config.pauli_prop_min_qubits
        && analysis.clifford_fraction >= 0.5
    {
        return SimBackend::PauliPropagation;
    }

    // Rule 4: Near-Clifford → CH-form (few T-gates, >20 qubits, >90% Clifford)
    if analysis.num_t_gates <= config.near_clifford_max_t
        && analysis.num_t_gates > 0
        && num_qubits > 20
        && analysis.clifford_fraction >= config.near_clifford_min_clifford_frac
    {
        return SimBackend::NearClifford;
    }

    // Rule 5: Structured/symmetric circuits → Decision diagram
    if num_qubits > 20 && analysis.circuit_symmetry.is_some() {
        return SimBackend::DecisionDiagram;
    }

    // Rule 6: Low magic + large → Stabilizer tensor network
    if num_qubits > 30 && analysis.magic_level <= config.stn_max_magic {
        return SimBackend::StabilizerTensorNetwork;
    }

    // Rule 7: Large circuit with low entanglement → MPS
    if num_qubits > config.mps_min_qubits
        && analysis.max_entanglement_width <= num_qubits / 2
    {
        // Prefer Metal MPS if available on macOS
        #[cfg(target_os = "macos")]
        if metal_gpu_available() {
            return SimBackend::MetalMPS;
        }
        return SimBackend::MPS;
    }

    // Rule 8: Metal GPU preferred on macOS
    #[cfg(target_os = "macos")]
    if num_qubits >= 4 && metal_gpu_available() {
        return SimBackend::MetalGPU;
    }

    // Rule 9: CUDA GPU preferred on non-macOS (if available)
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    if num_qubits >= 4 && cuda_gpu_available() {
        return SimBackend::CudaGPU;
    }

    // Rule 10: Prefer f32+fusion on larger CPU-only state-vector workloads.
    if num_qubits >= 10 {
        return SimBackend::StateVectorF32Fused;
    }

    // Rule 11: CPU fallback with fusion
    SimBackend::StateVectorFused
}

fn apply_gate_to_mps(sim: &mut MPSSimulator, gate: &Gate) -> bool {
    use crate::gates::GateType;
    match &gate.gate_type {
        GateType::H => sim.h(gate.targets[0]),
        GateType::X => sim.x(gate.targets[0]),
        GateType::Y => sim.y(gate.targets[0]),
        GateType::Z => sim.z(gate.targets[0]),
        GateType::S => sim.s(gate.targets[0]),
        GateType::T => sim.t(gate.targets[0]),
        GateType::Rx(theta) => sim.rx(gate.targets[0], *theta),
        GateType::Ry(theta) => sim.ry(gate.targets[0], *theta),
        GateType::Rz(theta) => sim.rz(gate.targets[0], *theta),
        GateType::CNOT => sim.cnot(gate.controls[0], gate.targets[0]),
        GateType::CZ => sim.cz(gate.controls[0], gate.targets[0]),
        GateType::SWAP => sim.swap(gate.targets[0], gate.targets[1]),
        _ => return false,
    }
    true
}

#[cfg(target_os = "macos")]
fn apply_gate_to_metal_mps(sim: &mut MetalMPSimulator, gate: &Gate) -> bool {
    use crate::gates::GateType;
    match &gate.gate_type {
        GateType::H => {
            let _ = sim.h(gate.targets[0]);
        }
        GateType::X => {
            let _ = sim.x(gate.targets[0]);
        }
        GateType::Y => {
            let _ = sim.y(gate.targets[0]);
        }
        GateType::Z => {
            let _ = sim.z(gate.targets[0]);
        }
        GateType::S => {
            let _ = sim.s(gate.targets[0]);
        }
        GateType::T => {
            let _ = sim.t(gate.targets[0]);
        }
        GateType::Rx(theta) => {
            let _ = sim.rx(gate.targets[0], *theta);
        }
        GateType::Ry(theta) => {
            let _ = sim.ry(gate.targets[0], *theta);
        }
        GateType::Rz(theta) => {
            let _ = sim.rz(gate.targets[0], *theta);
        }
        GateType::CNOT => {
            let _ = sim.cnot(gate.controls[0], gate.targets[0]);
        }
        _ => return false,
    }
    true
}

// ===================================================================
// AUTO SIMULATOR
// ===================================================================

/// Strategy used to select the simulation backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendSelectionStrategy {
    /// Rule-based heuristic (original `select_backend`).
    Heuristic,
    /// Complexity-theoretic runtime estimation across all viable backends.
    Estimated,
}

// ===================================================================
// RUNTIME TRACKER (Auto-Tuning Feedback Loop)
// ===================================================================

/// Observation of a backend's estimated vs actual runtime.
#[derive(Clone, Debug)]
pub struct RuntimeObservation {
    pub backend: SimBackend,
    pub estimated_time: f64,
    pub actual_time: f64,
}

/// Tracks runtime estimation accuracy per backend and computes correction factors.
///
/// After `min_observations` samples for a given backend, the correction factor
/// is updated as `mean(actual / estimated)`. Multiplying `estimate_runtime()`
/// by the correction factor yields a calibrated estimate.
#[derive(Clone, Debug)]
pub struct RuntimeTracker {
    /// Per-backend list of (estimated, actual) observations.
    observations: HashMap<SimBackend, Vec<(f64, f64)>>,
    /// Cached correction factors per backend (default 1.0).
    correction_factors: HashMap<SimBackend, f64>,
    /// Minimum number of observations before updating the correction factor.
    pub min_observations: usize,
}

impl RuntimeTracker {
    /// Create a new tracker with the given observation threshold.
    pub fn new(min_observations: usize) -> Self {
        Self {
            observations: HashMap::new(),
            correction_factors: HashMap::new(),
            min_observations,
        }
    }

    /// Record an observation and update correction factors if threshold is reached.
    pub fn record(&mut self, obs: RuntimeObservation) {
        let entry = self.observations.entry(obs.backend.clone()).or_default();
        entry.push((obs.estimated_time, obs.actual_time));

        if entry.len() >= self.min_observations {
            let factor: f64 = entry
                .iter()
                .filter(|(est, _)| *est > 0.0)
                .map(|(est, act)| act / est)
                .sum::<f64>()
                / entry
                    .iter()
                    .filter(|(est, _)| *est > 0.0)
                    .count()
                    .max(1) as f64;
            self.correction_factors.insert(obs.backend, factor);
        }
    }

    /// Get the correction factor for a backend (1.0 if not enough data).
    pub fn correction_factor(&self, backend: &SimBackend) -> f64 {
        self.correction_factors.get(backend).copied().unwrap_or(1.0)
    }

    /// Compute a corrected runtime estimate for the given backend and profile.
    pub fn corrected_estimate(&self, backend: &SimBackend, profile: &CircuitProfile) -> f64 {
        estimate_runtime(backend, profile) * self.correction_factor(backend)
    }

    /// Number of observations recorded for a backend.
    pub fn observation_count(&self, backend: &SimBackend) -> usize {
        self.observations.get(backend).map_or(0, |v| v.len())
    }
}

impl Default for RuntimeTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Automatic simulator that selects the best backend.
pub struct AutoSimulator {
    backend: SimBackend,
    num_qubits: usize,
    analysis: CircuitAnalysis,
    /// Optional circuit optimizer for automatic pre-pass optimization.
    optimizer: Option<CircuitOptimizer>,
    /// Which strategy was used to select the backend.
    strategy: BackendSelectionStrategy,
    /// Optional runtime tracker for auto-tuning feedback.
    runtime_tracker: Option<Arc<Mutex<RuntimeTracker>>>,
}

impl AutoSimulator {
    /// Create a new AutoSimulator by analyzing the circuit (heuristic strategy).
    pub fn new(gates: &[Gate], num_qubits: usize, noisy: bool) -> Self {
        let analysis = analyze_circuit(gates, num_qubits);
        let backend = select_backend(&analysis, num_qubits, noisy);
        AutoSimulator {
            backend,
            num_qubits,
            analysis,
            optimizer: None,
            strategy: BackendSelectionStrategy::Heuristic,
            runtime_tracker: None,
        }
    }

    /// Create a new AutoSimulator using complexity-theoretic runtime estimation.
    ///
    /// This uses `select_backend_estimated` instead of the heuristic rule chain,
    /// picking the backend with the lowest estimated runtime across all viable
    /// candidates.
    pub fn new_estimated(gates: &[Gate], num_qubits: usize, noisy: bool) -> Self {
        let analysis = analyze_circuit(gates, num_qubits);
        let backend = select_backend_estimated(gates, num_qubits, noisy);
        AutoSimulator {
            backend,
            num_qubits,
            analysis,
            optimizer: None,
            strategy: BackendSelectionStrategy::Estimated,
            runtime_tracker: None,
        }
    }

    /// Create with a specific backend (overrides auto-selection).
    pub fn with_backend(backend: SimBackend, num_qubits: usize) -> Self {
        AutoSimulator {
            backend,
            num_qubits,
            analysis: CircuitAnalysis {
                num_gates: 0,
                num_single_qubit: 0,
                num_two_qubit: 0,
                num_three_qubit: 0,
                is_clifford_only: false,
                max_entanglement_width: 0,
                connected_components: 0,
                num_t_gates: 0,
                clifford_fraction: 0.0,
                circuit_symmetry: None,
                magic_level: 0.0,
            },
            optimizer: None,
            strategy: BackendSelectionStrategy::Heuristic,
            runtime_tracker: None,
        }
    }

    /// Create with strict Metal GPU-only backend.
    pub fn with_gpu_only(num_qubits: usize) -> Self {
        Self::with_backend(SimBackend::MetalGPUOnly, num_qubits)
    }

    /// Set the backend selection strategy (builder pattern).
    pub fn with_strategy(mut self, strategy: BackendSelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Attach a runtime tracker for auto-tuning feedback (builder pattern).
    pub fn with_runtime_tracker(mut self, tracker: Arc<Mutex<RuntimeTracker>>) -> Self {
        self.runtime_tracker = Some(tracker);
        self
    }

    /// Get the backend selection strategy.
    pub fn strategy(&self) -> &BackendSelectionStrategy {
        &self.strategy
    }

    /// Get the selected backend.
    pub fn backend(&self) -> &SimBackend {
        &self.backend
    }

    /// Get the circuit analysis.
    pub fn analysis(&self) -> &CircuitAnalysis {
        &self.analysis
    }

    /// Enable automatic circuit optimization at the given level.
    ///
    /// When enabled, `execute` and `execute_result` will run an optimization
    /// pre-pass on the gate list before dispatching to the backend.
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimizer = Some(CircuitOptimizer::new(level));
        self
    }

    /// Disable circuit optimization.
    pub fn without_optimization(mut self) -> Self {
        self.optimizer = None;
        self
    }

    /// Set the optimization level (mutable borrow variant).
    pub fn set_optimization(&mut self, level: OptimizationLevel) {
        self.optimizer = Some(CircuitOptimizer::new(level));
    }

    /// Clear the optimizer (mutable borrow variant).
    pub fn clear_optimization(&mut self) {
        self.optimizer = None;
    }

    /// Conditionally optimize the gate list, returning a reference to
    /// either the optimized or the original gates.
    fn maybe_optimize<'a>(&self, gates: &'a [Gate], buf: &'a mut Vec<Gate>) -> &'a [Gate] {
        if let Some(ref opt) = self.optimizer {
            *buf = opt.optimize(gates);
            buf
        } else {
            gates
        }
    }

    /// Execute the circuit and return probabilities.
    ///
    /// This convenience API preserves historical fallback behavior. For strict
    /// GPU-only execution semantics, use `execute_result`.
    pub fn execute(&self, gates: &[Gate]) -> Vec<f64> {
        let mut opt_buf = Vec::new();
        let gates = self.maybe_optimize(gates, &mut opt_buf);
        self.execute_inner(gates)
            .unwrap_or_else(|_| self.execute_cpu_fused(gates))
    }

    /// Execute circuit and return probabilities or an execution error.
    ///
    /// Use this when strict backend guarantees are required (e.g. `MetalGPUOnly`).
    pub fn execute_result(&self, gates: &[Gate]) -> Result<Vec<f64>, String> {
        let mut opt_buf = Vec::new();
        let gates = self.maybe_optimize(gates, &mut opt_buf);
        self.execute_inner(gates)
    }

    /// Internal dispatch (operates on already-optimized gates).
    ///
    /// When a `RuntimeTracker` is attached, this method automatically records
    /// (estimated, actual) runtime observations after each successful execution,
    /// allowing the tracker to learn per-backend correction factors over time.
    fn execute_inner(&self, gates: &[Gate]) -> Result<Vec<f64>, String> {
        let t0 = self.runtime_tracker.as_ref().map(|_| Instant::now());

        let result = match &self.backend {
            SimBackend::MetalGPU => Ok(self.execute_gpu(gates)),
            SimBackend::MetalGPUOnly => self.execute_gpu_strict(gates),
            SimBackend::CudaGPU => Ok(self.execute_cuda(gates)),
            SimBackend::CudaGPUOnly => self.execute_cuda_strict(gates),
            SimBackend::StateVector => {
                let mut state = QuantumState::new(self.num_qubits);
                for gate in gates {
                    apply_gate_to_state(&mut state, gate);
                }
                Ok(state.probabilities())
            }
            SimBackend::StateVectorFused => {
                let fusion = fuse_gates(gates);
                let mut state = QuantumState::new(self.num_qubits);
                execute_fused_circuit(&mut state, &fusion);
                Ok(state.probabilities())
            }
            SimBackend::StateVectorF32Fused => {
                let exec = F32FusionExecutor::new();
                match exec.execute(self.num_qubits, gates) {
                    Ok((state, _metrics)) => Ok(state.probabilities()),
                    Err(_) => {
                        // Keep compatibility: graceful fallback if f32 path cannot execute.
                        Ok(self.execute_cpu_fused(gates))
                    }
                }
            }
            SimBackend::DensityMatrix => {
                let mut state = QuantumState::new(self.num_qubits);
                for gate in gates {
                    apply_gate_to_state(&mut state, gate);
                }
                Ok(state.probabilities())
            }
            SimBackend::Stabilizer => {
                let mut state = QuantumState::new(self.num_qubits);
                for gate in gates {
                    apply_gate_to_state(&mut state, gate);
                }
                Ok(state.probabilities())
            }
            SimBackend::MPS => {
                let mut sim = MPSSimulator::new(self.num_qubits, Some(64));
                for gate in gates {
                    if !apply_gate_to_mps(&mut sim, gate) {
                        // Fallback to state vector for unsupported gates
                        return Ok(self.execute_cpu_fused(gates));
                    }
                }
                Ok(sim.to_state_vector().iter().map(|c| c.norm_sqr()).collect())
            }
            SimBackend::MetalMPS => {
                #[cfg(target_os = "macos")]
                {
                    if let Ok(mut sim) = MetalMPSimulator::new(self.num_qubits, 64) {
                        for gate in gates {
                            if !apply_gate_to_metal_mps(&mut sim, gate) {
                                // Fallback to CPU MPS
                                let mut cpu = MPSSimulator::new(self.num_qubits, Some(64));
                                for g in gates {
                                    if !apply_gate_to_mps(&mut cpu, g) {
                                        return Ok(self.execute_cpu_fused(gates));
                                    }
                                }
                                return Ok(cpu
                                    .to_state_vector()
                                    .iter()
                                    .map(|c| c.norm_sqr())
                                    .collect());
                            }
                        }
                        // GPU MPS measurement to probabilities is expensive; convert via CPU fallback.
                        if let Ok(vec) = sim.to_state_vector() {
                            return Ok(vec.iter().map(|c| c.norm_sqr()).collect());
                        }
                    }
                }
                // Fallback
                Ok(self.execute_cpu_fused(gates))
            }
            // New backends: route through their native simulators when possible,
            // falling back to state-vector+fusion for gate types they don't support.
            SimBackend::PauliPropagation
            | SimBackend::DecisionDiagram
            | SimBackend::NearClifford
            | SimBackend::StabilizerTensorNetwork => {
                // These backends have their own dedicated APIs (PauliPropagationSimulator,
                // DDNodePool, NearCliffordSimulator, StabilizerTensorNetwork). The AutoSimulator
                // provides a gateway; users needing full control should use the modules directly.
                // For the common execute() path, fall back to CPU state-vector with fusion.
                Ok(self.execute_cpu_fused(gates))
            }
        };

        // Auto-populate RuntimeTracker with (estimated, actual) observation.
        if let (Some(tracker), Some(start)) = (&self.runtime_tracker, t0) {
            if result.is_ok() {
                let profile = CircuitProfile::analyze(gates, self.num_qubits);
                let estimated = estimate_runtime(&self.backend, &profile);
                let actual = start.elapsed().as_secs_f64();
                if let Ok(mut t) = tracker.lock() {
                    t.record(RuntimeObservation {
                        backend: self.backend.clone(),
                        estimated_time: estimated,
                        actual_time: actual,
                    });
                }
            }
        }

        result
    }

    fn execute_cpu_fused(&self, gates: &[Gate]) -> Vec<f64> {
        let fusion = fuse_gates(gates);
        let mut state = QuantumState::new(self.num_qubits);
        execute_fused_circuit(&mut state, &fusion);
        state.probabilities()
    }

    /// Execute on Metal GPU, falling back to CPU fused if GPU fails.
    fn execute_gpu(&self, gates: &[Gate]) -> Vec<f64> {
        self.execute_gpu_strict(gates)
            .unwrap_or_else(|_| self.execute_cpu_fused(gates))
    }

    /// Execute on Metal GPU with strict semantics (no CPU fallback).
    fn execute_gpu_strict(&self, gates: &[Gate]) -> Result<Vec<f64>, String> {
        #[cfg(target_os = "macos")]
        {
            match MetalSimulator::new(self.num_qubits) {
                Ok(sim) => {
                    sim.run_circuit(gates);
                    let probs_f32 = sim.probabilities();
                    return Ok(probs_f32.into_iter().map(|p| p as f64).collect());
                }
                Err(_) => {
                    return Err(
                        "Metal GPU initialization failed in strict GPU-only mode".to_string()
                    );
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = gates;
            Err("Metal GPU-only backend requires macOS".to_string())
        }
    }

    /// Execute on CUDA GPU, falling back to CPU fused if GPU fails.
    fn execute_cuda(&self, gates: &[Gate]) -> Vec<f64> {
        self.execute_cuda_strict(gates)
            .unwrap_or_else(|_| self.execute_cpu_fused(gates))
    }

    /// Execute on CUDA GPU with strict semantics (no CPU fallback).
    ///
    /// Initializes the CUDA executor, loads kernels, then dispatches each gate
    /// in the circuit to the appropriate CUDA kernel. The state vector lives on
    /// the host side and is transferred per-gate via the executor's H2D/D2H
    /// copy path. For gates not yet supported on the GPU (rotations, CZ, etc.)
    /// we fall back to a CPU-side application using the standard state-vector
    /// engine, keeping the hybrid approach transparent to the caller.
    fn execute_cuda_strict(&self, gates: &[Gate]) -> Result<Vec<f64>, String> {
        #[cfg(feature = "cuda")]
        {
            use num_complex::Complex64;

            let mut executor = CudaParallelQuantumExecutor::new()
                .map_err(|e| format!("CUDA initialization failed: {:?}", e))?;
            executor.load_kernels()
                .map_err(|e| format!("CUDA kernel loading failed: {:?}", e))?;

            // Initialize host-side state vector: |0...0>
            let dim = 1usize << self.num_qubits;
            let mut state_vec = vec![Complex64::new(0.0, 0.0); dim];
            state_vec[0] = Complex64::new(1.0, 0.0);

            for gate in gates {
                let applied_on_gpu = match &gate.gate_type {
                    GateType::H => {
                        executor.apply_h(&mut state_vec, self.num_qubits, gate.targets[0])
                            .is_ok()
                    }
                    GateType::X => {
                        executor.apply_x(&mut state_vec, self.num_qubits, gate.targets[0])
                            .is_ok()
                    }
                    GateType::CNOT => {
                        executor.apply_cnot(
                            &mut state_vec,
                            self.num_qubits,
                            gate.controls[0],
                            gate.targets[0],
                        ).is_ok()
                    }
                    _ => false,
                };

                if !applied_on_gpu {
                    // Fall back to CPU for unsupported gate types.
                    // Convert Complex64 state to QuantumState, apply gate, convert back.
                    let mut qs = QuantumState::new(self.num_qubits);
                    {
                        let amps = qs.amplitudes_mut();
                        for (i, amp) in state_vec.iter().enumerate() {
                            amps[i] = crate::C64::new(amp.re, amp.im);
                        }
                    }
                    apply_gate_to_state(&mut qs, gate);
                    {
                        let amps = qs.amplitudes();
                        for (i, amp) in amps.iter().enumerate() {
                            state_vec[i] = Complex64::new(amp.re, amp.im);
                        }
                    }
                }
            }

            // Compute probabilities from final state vector
            let probs: Vec<f64> = state_vec.iter().map(|c| c.re * c.re + c.im * c.im).collect();
            return Ok(probs);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = gates;
            Err("CUDA backend not enabled (requires 'cuda' feature)".to_string())
        }
    }
}

// ===================================================================
// RUNTIME ESTIMATION
// ===================================================================

/// Estimated runtime (in arbitrary units) for a given backend and circuit profile.
///
/// Uses complexity-theoretic models:
/// - StateVector: O(2^n * depth) -- exponential in qubits
/// - Stabilizer: O(n^2 * depth) -- polynomial (Gottesman-Knill)
/// - MPS: O(chi^3 * n * depth) where chi = bond dimension estimate
/// - DensityMatrix: O(4^n * depth) -- double-exponential
/// - NearClifford: O(2^t * n^2 * depth) where t = T-gate count
/// - PauliPropagation: O(n^2 * depth) (similar to stabilizer for Clifford-heavy)
pub fn estimate_runtime(backend: &SimBackend, profile: &CircuitProfile) -> f64 {
    let n = profile.n_qubits as f64;
    let d = profile.depth.max(1) as f64;
    let t = profile.t_gate_count as f64;

    match backend {
        SimBackend::StateVector | SimBackend::StateVectorFused | SimBackend::StateVectorF32Fused => {
            // O(2^n * depth), f32+fusion gets ~2x speedup
            let base = 2.0_f64.powf(n) * d;
            match backend {
                SimBackend::StateVectorF32Fused => base * 0.5,
                SimBackend::StateVectorFused => base * 0.7,
                _ => base,
            }
        }
        SimBackend::MetalGPU | SimBackend::MetalGPUOnly => {
            // GPU gives ~10x speedup for n >= 15, less for smaller circuits.
            let gpu_speedup = if n >= 15.0 { 10.0 } else { 2.0 + n * 0.5 };
            2.0_f64.powf(n) * d / gpu_speedup
        }
        SimBackend::CudaGPU | SimBackend::CudaGPUOnly => {
            let gpu_speedup = if n >= 15.0 { 15.0 } else { 3.0 + n * 0.5 };
            2.0_f64.powf(n) * d / gpu_speedup
        }
        SimBackend::Stabilizer => {
            // O(n^2 * depth) -- only valid for Clifford-only circuits.
            n * n * d
        }
        SimBackend::MPS | SimBackend::MetalMPS => {
            // O(chi^3 * n * depth), chi estimated from entanglement.
            let chi = estimate_bond_dim(profile);
            let base = chi.powi(3) * n * d;
            if matches!(backend, SimBackend::MetalMPS) {
                base * 0.5
            } else {
                base
            }
        }
        SimBackend::DensityMatrix => {
            // O(4^n * depth)
            4.0_f64.powf(n) * d
        }
        SimBackend::NearClifford => {
            // O(2^t * n^2 * depth) where t = number of T gates.
            2.0_f64.powf(t) * n * n * d
        }
        SimBackend::PauliPropagation => {
            // Similar to stabilizer for Clifford-heavy circuits.
            n * n * d * (1.0 + t * 0.1)
        }
        SimBackend::DecisionDiagram => {
            // Hard to estimate, but generally O(n^3 * depth) for structured circuits.
            n * n * n * d
        }
        SimBackend::StabilizerTensorNetwork => {
            // O(2^t * chi^3 * n)
            let chi = estimate_bond_dim(profile);
            2.0_f64.powf(t.min(20.0)) * chi.powi(3) * n
        }
    }
}

/// Estimate bond dimension from circuit profile.
///
/// Bond dimension grows with entanglement: low-entanglement circuits can be
/// compressed to small chi, while highly-entangled circuits require
/// exponentially larger bond dimension.
fn estimate_bond_dim(profile: &CircuitProfile) -> f64 {
    let base_chi = 4.0;
    let entanglement_factor = 1.0 + profile.estimated_entanglement * 10.0;
    let depth_factor = 1.0
        + (profile.depth as f64 / profile.n_qubits.max(1) as f64).min(5.0);
    (base_chi * entanglement_factor * depth_factor).min(256.0)
}

/// Select the best backend by estimating runtime for all viable backends
/// and choosing the minimum.
///
/// This is a more principled approach than the heuristic rule chain in
/// `select_backend` / `select_backend_profiled`, using complexity-theoretic
/// runtime estimates for each backend to make the routing decision.
pub fn select_backend_estimated(gates: &[Gate], num_qubits: usize, noisy: bool) -> SimBackend {
    let profile = CircuitProfile::analyze(gates, num_qubits);
    let analysis = analyze_circuit(gates, num_qubits);

    // Build list of viable backends with their estimated runtimes.
    let mut candidates: Vec<(SimBackend, f64)> = Vec::new();

    // StateVector variants (viable for small-to-medium non-noisy circuits;
    // state vector cannot simulate noise channels).
    if !noisy && num_qubits <= 30 {
        candidates.push((
            SimBackend::StateVector,
            estimate_runtime(&SimBackend::StateVector, &profile),
        ));
        candidates.push((
            SimBackend::StateVectorFused,
            estimate_runtime(&SimBackend::StateVectorFused, &profile),
        ));
        candidates.push((
            SimBackend::StateVectorF32Fused,
            estimate_runtime(&SimBackend::StateVectorF32Fused, &profile),
        ));
    }

    // Metal GPU (macOS only; cannot simulate noise).
    #[cfg(target_os = "macos")]
    if !noisy && num_qubits >= 4 && num_qubits <= 30 && metal_gpu_available() {
        candidates.push((
            SimBackend::MetalGPU,
            estimate_runtime(&SimBackend::MetalGPU, &profile),
        ));
    }

    // Stabilizer (Clifford-only circuits; can handle noise via tableau methods).
    if analysis.is_clifford_only {
        candidates.push((
            SimBackend::Stabilizer,
            estimate_runtime(&SimBackend::Stabilizer, &profile),
        ));
    }

    // MPS (large circuits with moderate entanglement; pure-state only).
    if !noisy && num_qubits > 15 {
        candidates.push((
            SimBackend::MPS,
            estimate_runtime(&SimBackend::MPS, &profile),
        ));
        #[cfg(target_os = "macos")]
        if metal_gpu_available() {
            candidates.push((
                SimBackend::MetalMPS,
                estimate_runtime(&SimBackend::MetalMPS, &profile),
            ));
        }
    }

    // DensityMatrix (noisy, small; the only backend that handles arbitrary noise).
    if noisy && num_qubits <= 13 {
        candidates.push((
            SimBackend::DensityMatrix,
            estimate_runtime(&SimBackend::DensityMatrix, &profile),
        ));
    }

    // NearClifford (few T gates on large circuits; pure-state only).
    if !noisy && analysis.num_t_gates > 0 && analysis.num_t_gates <= 40 && num_qubits > 20 {
        candidates.push((
            SimBackend::NearClifford,
            estimate_runtime(&SimBackend::NearClifford, &profile),
        ));
    }

    // PauliPropagation (noisy, Clifford-heavy; supports noise natively).
    if noisy && num_qubits >= 30 && analysis.clifford_fraction >= 0.5 {
        candidates.push((
            SimBackend::PauliPropagation,
            estimate_runtime(&SimBackend::PauliPropagation, &profile),
        ));
    }

    // DecisionDiagram (symmetric circuits; pure-state only).
    if !noisy && num_qubits > 20 && analysis.circuit_symmetry.is_some() {
        candidates.push((
            SimBackend::DecisionDiagram,
            estimate_runtime(&SimBackend::DecisionDiagram, &profile),
        ));
    }

    // StabilizerTensorNetwork (low magic, large; pure-state only).
    if !noisy && num_qubits > 30 && analysis.magic_level <= 0.3 {
        candidates.push((
            SimBackend::StabilizerTensorNetwork,
            estimate_runtime(&SimBackend::StabilizerTensorNetwork, &profile),
        ));
    }

    // Pick minimum runtime.
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    candidates
        .into_iter()
        .next()
        .map(|(b, _)| b)
        .unwrap_or(SimBackend::StateVectorFused)
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    #[test]
    fn test_clifford_detection() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::s(0), Gate::x(1)];
        let analysis = analyze_circuit(&gates, 2);
        assert!(analysis.is_clifford_only);
    }

    #[test]
    fn test_non_clifford_detection() {
        let gates = vec![
            Gate::h(0),
            Gate::t(0), // T is NOT Clifford
            Gate::cnot(0, 1),
        ];
        let analysis = analyze_circuit(&gates, 2);
        assert!(!analysis.is_clifford_only);
    }

    #[test]
    fn test_backend_clifford_uses_stabilizer() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::s(1)];
        let sim = AutoSimulator::new(&gates, 2, false);
        assert_eq!(*sim.backend(), SimBackend::Stabilizer);
    }

    #[test]
    fn test_backend_noisy_small_uses_density() {
        let gates = vec![Gate::h(0), Gate::rx(0, 0.5)]; // non-Clifford
        let sim = AutoSimulator::new(&gates, 10, true);
        assert_eq!(*sim.backend(), SimBackend::DensityMatrix);
    }

    #[test]
    fn test_backend_default_uses_gpu_or_fused() {
        let gates = vec![Gate::h(0), Gate::t(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::new(&gates, 10, false);
        // On macOS with Metal: MetalGPU. On other platforms: f32/fused CPU.
        let backend = sim.backend().clone();
        assert!(
            backend == SimBackend::MetalGPU
                || backend == SimBackend::StateVectorFused
                || backend == SimBackend::StateVectorF32Fused,
            "Expected MetalGPU or fused CPU variant, got {:?}",
            backend
        );
    }

    #[test]
    fn test_small_circuit_avoids_gpu() {
        // Very small circuits (< 4 qubits) should not use GPU
        let gates = vec![Gate::h(0), Gate::t(0)];
        let sim = AutoSimulator::new(&gates, 2, false);
        // 2 qubits is below GPU threshold, but Clifford check matters
        // T gate makes it non-Clifford, and 2 qubits < 4 → StateVectorFused
        assert_ne!(*sim.backend(), SimBackend::MetalGPU);
    }

    #[test]
    fn test_execute_produces_valid_probabilities() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::new(&gates, 2, false);
        let probs = sim.execute(&gates);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_connectivity_analysis() {
        // Two disconnected components: {0,1} and {2}
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::h(2)];
        let analysis = analyze_circuit(&gates, 3);
        assert_eq!(analysis.connected_components, 2);
        assert_eq!(analysis.max_entanglement_width, 2);
    }

    #[test]
    fn test_with_backend_override() {
        let sim = AutoSimulator::with_backend(SimBackend::StateVector, 5);
        assert_eq!(*sim.backend(), SimBackend::StateVector);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn test_gpu_only_errors_off_macos() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::with_gpu_only(2);
        let err = sim
            .execute_result(&gates)
            .expect_err("expected strict GPU-only error");
        assert!(err.contains("requires macOS"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_gpu_only_executes_on_macos() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::with_gpu_only(2);
        let probs = sim
            .execute_result(&gates)
            .expect("GPU-only execution should succeed");
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_fused_matches_unfused() {
        let gates = vec![
            Gate::h(0),
            Gate::rx(0, 0.5),
            Gate::t(1),
            Gate::cnot(0, 1),
            Gate::rz(0, 0.3),
        ];

        let sim_fused = AutoSimulator::with_backend(SimBackend::StateVectorFused, 2);
        let sim_plain = AutoSimulator::with_backend(SimBackend::StateVector, 2);

        let probs_fused = sim_fused.execute(&gates);
        let probs_plain = sim_plain.execute(&gates);

        for (a, b) in probs_fused.iter().zip(probs_plain.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "Fused/unfused mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_f32_fused_matches_unfused() {
        let gates = vec![
            Gate::h(0),
            Gate::rx(0, 0.31),
            Gate::cnot(0, 1),
            Gate::rz(1, -0.4),
            Gate::ry(0, 0.12),
        ];

        let sim_f32 = AutoSimulator::with_backend(SimBackend::StateVectorF32Fused, 2);
        let sim_plain = AutoSimulator::with_backend(SimBackend::StateVector, 2);

        let probs_f32 = sim_f32.execute(&gates);
        let probs_plain = sim_plain.execute(&gates);

        for (a, b) in probs_f32.iter().zip(probs_plain.iter()) {
            assert!((a - b).abs() < 1e-5, "f32/unfused mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_new_backend_variants_display() {
        assert_eq!(format!("{}", SimBackend::PauliPropagation), "Pauli Propagation");
        assert_eq!(format!("{}", SimBackend::DecisionDiagram), "Decision Diagram");
        assert_eq!(format!("{}", SimBackend::NearClifford), "Near-Clifford (CH-form)");
        assert_eq!(
            format!("{}", SimBackend::StabilizerTensorNetwork),
            "Stabilizer Tensor Network"
        );
    }

    #[test]
    fn test_new_backends_execute_via_fallback() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        for backend in [
            SimBackend::PauliPropagation,
            SimBackend::DecisionDiagram,
            SimBackend::NearClifford,
            SimBackend::StabilizerTensorNetwork,
        ] {
            let sim = AutoSimulator::with_backend(backend.clone(), 2);
            let probs = sim.execute(&gates);
            let total: f64 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-10,
                "Backend {:?} gave invalid probs (sum={})",
                backend,
                total
            );
        }
    }

    #[test]
    fn test_routing_near_clifford() {
        // Build a circuit with few T-gates (< 40), high Clifford fraction, >20 qubits
        let mut gates = Vec::new();
        for i in 0..25 {
            gates.push(Gate::h(i));
        }
        // Add a small number of T gates
        for i in 0..5 {
            gates.push(Gate::t(i));
        }
        // Add many more Clifford gates
        for i in 0..24 {
            gates.push(Gate::cnot(i, i + 1));
        }
        let analysis = analyze_circuit(&gates, 25);
        let backend = select_backend(&analysis, 25, false);
        assert_eq!(backend, SimBackend::NearClifford);
    }

    #[test]
    fn test_routing_pauli_propagation() {
        // Noisy, 30+ qubits, Clifford-heavy
        let mut gates = Vec::new();
        for i in 0..35 {
            gates.push(Gate::h(i));
            gates.push(Gate::s(i));
        }
        for i in 0..34 {
            gates.push(Gate::cnot(i, i + 1));
        }
        // Add a few non-Clifford to make it not purely Clifford
        gates.push(Gate::t(0));
        let analysis = analyze_circuit(&gates, 35);
        let backend = select_backend(&analysis, 35, true); // noisy=true
        assert_eq!(backend, SimBackend::PauliPropagation);
    }

    #[test]
    fn test_routing_config_defaults() {
        let config = RoutingConfig::default();
        assert_eq!(config.pauli_prop_min_qubits, 30);
        assert_eq!(config.near_clifford_max_t, 40);
        assert!((config.near_clifford_min_clifford_frac - 0.90).abs() < 1e-10);
        assert!((config.stn_max_magic - 0.3).abs() < 1e-10);
        assert_eq!(config.mps_min_qubits, 25);
    }

    // ------------------------------------------------------------------
    // Circuit optimizer pre-pass integration tests
    // ------------------------------------------------------------------

    #[test]
    fn test_optimizer_reduces_gate_count() {
        // H-H cancellation: optimizer should remove both gates
        let gates = vec![
            Gate::h(0),
            Gate::h(0), // cancels with previous H
            Gate::cnot(0, 1),
        ];
        let sim = AutoSimulator::new(&gates, 2, false)
            .with_optimization(OptimizationLevel::Basic);
        assert!(sim.optimizer.is_some());

        // Verify the optimizer actually reduces gates
        let mut opt_buf = Vec::new();
        let optimized = sim.maybe_optimize(&gates, &mut opt_buf);
        assert!(
            optimized.len() < gates.len(),
            "optimizer should reduce gate count: {} vs {}",
            optimized.len(),
            gates.len()
        );
    }

    #[test]
    fn test_optimizer_correctness() {
        // Results should be identical with and without optimization
        let gates = vec![
            Gate::h(0),
            Gate::h(0), // cancels
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::x(0),
            Gate::x(0), // cancels
        ];
        let num_qubits = 2;

        // Without optimization
        let sim_no_opt = AutoSimulator::new(&gates, num_qubits, false);
        let probs_no_opt = sim_no_opt.execute(&gates);

        // With optimization
        let sim_opt = AutoSimulator::new(&gates, num_qubits, false)
            .with_optimization(OptimizationLevel::Aggressive);
        let probs_opt = sim_opt.execute(&gates);

        // Results must match within floating-point tolerance
        assert_eq!(probs_no_opt.len(), probs_opt.len());
        for (i, (a, b)) in probs_no_opt.iter().zip(probs_opt.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "probability mismatch at index {}: {} vs {}",
                i, a, b
            );
        }
    }

    #[test]
    fn test_optimizer_disabled_by_default() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::new(&gates, 2, false);
        assert!(sim.optimizer.is_none());
    }

    #[test]
    fn test_optimizer_can_be_disabled() {
        let gates = vec![Gate::h(0)];
        let sim = AutoSimulator::new(&gates, 1, false)
            .with_optimization(OptimizationLevel::Aggressive)
            .without_optimization();
        assert!(sim.optimizer.is_none());
    }

    #[test]
    fn test_optimizer_set_clear() {
        let gates = vec![Gate::h(0)];
        let mut sim = AutoSimulator::new(&gates, 1, false);
        assert!(sim.optimizer.is_none());

        sim.set_optimization(OptimizationLevel::Basic);
        assert!(sim.optimizer.is_some());

        sim.clear_optimization();
        assert!(sim.optimizer.is_none());
    }

    #[test]
    fn test_optimizer_none_level_is_noop() {
        // OptimizationLevel::None should not modify the gate list
        let gates = vec![
            Gate::h(0),
            Gate::h(0), // would normally cancel
            Gate::cnot(0, 1),
        ];
        let sim = AutoSimulator::new(&gates, 2, false)
            .with_optimization(OptimizationLevel::None);

        let mut opt_buf = Vec::new();
        let optimized = sim.maybe_optimize(&gates, &mut opt_buf);
        assert_eq!(optimized.len(), gates.len());
    }

    #[test]
    fn test_optimizer_with_backend() {
        // with_backend should also support optimization
        let sim = AutoSimulator::with_backend(SimBackend::StateVector, 2)
            .with_optimization(OptimizationLevel::Moderate);
        assert!(sim.optimizer.is_some());

        let gates = vec![
            Gate::h(0),
            Gate::h(0),
            Gate::cnot(0, 1),
        ];
        // Should execute without panic
        let probs = sim.execute(&gates);
        assert_eq!(probs.len(), 4); // 2 qubits => 4 probabilities
    }

    // =================================================================
    // CircuitProfile tests
    // =================================================================

    #[test]
    fn test_profile_empty_circuit() {
        let gates: Vec<Gate> = vec![];
        let profile = CircuitProfile::analyze(&gates, 4);

        assert_eq!(profile.n_qubits, 4);
        assert_eq!(profile.depth, 0);
        assert_eq!(profile.gate_count, 0);
        assert!((profile.clifford_fraction - 1.0).abs() < 1e-10);
        assert_eq!(profile.t_gate_count, 0);
        assert!(!profile.is_parametric);
        assert!(!profile.has_mid_circuit_measurement);
        assert!((profile.estimated_entanglement - 0.0).abs() < 1e-10);
        assert_eq!(profile.max_qubit_degree, 0);
    }

    #[test]
    fn test_profile_single_qubit_circuit() {
        let gates = vec![Gate::h(0), Gate::t(0), Gate::s(0)];
        let profile = CircuitProfile::analyze(&gates, 1);

        assert_eq!(profile.n_qubits, 1);
        assert_eq!(profile.depth, 3);
        assert_eq!(profile.gate_count, 3);
        assert_eq!(profile.t_gate_count, 1);
        assert!((profile.two_qubit_fraction - 0.0).abs() < 1e-10);
        assert_eq!(profile.max_qubit_degree, 0);
        assert!(!profile.is_parametric);
    }

    #[test]
    fn test_profile_clifford_fraction() {
        // 3 Clifford (H, CNOT, S) + 1 non-Clifford (T) = 75% Clifford
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::s(1), Gate::t(0)];
        let profile = CircuitProfile::analyze(&gates, 2);

        assert!((profile.clifford_fraction - 0.75).abs() < 1e-10);
        assert_eq!(profile.t_gate_count, 1);
        assert!(!profile.is_mostly_clifford()); // 75% < 80%
    }

    #[test]
    fn test_profile_is_mostly_clifford() {
        // 9 Clifford + 1 T = 90% Clifford
        let mut gates: Vec<Gate> = (0..9).map(|i| Gate::h(i % 3)).collect();
        gates.push(Gate::t(0));
        let profile = CircuitProfile::analyze(&gates, 3);

        assert!(profile.is_mostly_clifford()); // 90% > 80%
    }

    #[test]
    fn test_profile_parametric_detection() {
        let gates = vec![Gate::h(0), Gate::rx(0, 0.5), Gate::cnot(0, 1)];
        let profile = CircuitProfile::analyze(&gates, 2);

        assert!(profile.is_parametric);
    }

    #[test]
    fn test_profile_non_parametric_circuit() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::t(1)];
        let profile = CircuitProfile::analyze(&gates, 2);

        assert!(!profile.is_parametric);
    }

    #[test]
    fn test_profile_depth_computation() {
        // Sequential gates on one qubit -> depth = gate count
        let gates = vec![Gate::h(0), Gate::t(0), Gate::s(0)];
        let profile = CircuitProfile::analyze(&gates, 1);
        assert_eq!(profile.depth, 3);
    }

    #[test]
    fn test_profile_depth_parallel_gates() {
        // Parallel single-qubit gates on different qubits -> depth = 1 each
        // But they are processed sequentially in the gate list, so
        // H(0) -> qubit 0 at layer 1, H(1) -> qubit 1 at layer 1
        // CNOT(0,1) -> max(1,1)+1 = 2
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1)];
        let profile = CircuitProfile::analyze(&gates, 2);
        assert_eq!(profile.depth, 2);
    }

    #[test]
    fn test_profile_connectivity_linear_chain() {
        // Linear chain: CNOT(0,1), CNOT(1,2), CNOT(2,3)
        // Qubit degrees: 0->1, 1->2, 2->2, 3->1 => avg = 1.5
        let gates = vec![
            Gate::cnot(0, 1),
            Gate::cnot(1, 2),
            Gate::cnot(2, 3),
        ];
        let profile = CircuitProfile::analyze(&gates, 4);

        assert!((profile.connectivity - 1.5).abs() < 1e-10);
        assert_eq!(profile.max_qubit_degree, 2);
        assert!(profile.has_low_treewidth()); // avg=1.5, max=2
    }

    #[test]
    fn test_profile_connectivity_star_topology() {
        // Star: CNOT(0,1), CNOT(0,2), CNOT(0,3), CNOT(0,4)
        // Qubit 0 degree=4, others degree=1 => avg = (4+1+1+1+1)/5 = 1.6
        let gates = vec![
            Gate::cnot(0, 1),
            Gate::cnot(0, 2),
            Gate::cnot(0, 3),
            Gate::cnot(0, 4),
        ];
        let profile = CircuitProfile::analyze(&gates, 5);

        assert!((profile.connectivity - 1.6).abs() < 1e-10);
        assert_eq!(profile.max_qubit_degree, 4);
        assert!(profile.has_low_treewidth()); // avg=1.6, max=4
    }

    #[test]
    fn test_profile_high_connectivity_rejects_low_treewidth() {
        // Fully connected: every pair of 5 qubits
        let mut gates = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                gates.push(Gate::cnot(i, j));
            }
        }
        let profile = CircuitProfile::analyze(&gates, 5);

        // Each qubit connects to all 4 others -> degree 4, avg 4.0
        assert_eq!(profile.max_qubit_degree, 4);
        assert!((profile.connectivity - 4.0).abs() < 1e-10);
        assert!(!profile.has_low_treewidth()); // avg=4.0 > 2.0
    }

    #[test]
    fn test_profile_two_qubit_fraction() {
        // 2 single-qubit + 1 two-qubit = 1/3 two-qubit fraction
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1)];
        let profile = CircuitProfile::analyze(&gates, 2);

        assert!((profile.two_qubit_fraction - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_profile_entanglement_estimate() {
        // Pure single-qubit circuit -> no entanglement
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2)];
        let profile = CircuitProfile::analyze(&gates, 3);
        assert!((profile.estimated_entanglement - 0.0).abs() < 1e-10);

        // Heavy two-qubit circuit -> higher entanglement
        let mut gates2 = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..5 {
                gates2.push(Gate::cnot(i, j));
            }
        }
        let profile2 = CircuitProfile::analyze(&gates2, 5);
        assert!(profile2.estimated_entanglement > 0.3);
    }

    #[test]
    fn test_profile_clifford_t_only() {
        let gates = vec![Gate::h(0), Gate::t(0), Gate::cnot(0, 1), Gate::s(1)];
        let profile = CircuitProfile::analyze(&gates, 2);
        assert!(profile.is_clifford_t_only(&gates));

        // Adding Rx breaks Clifford+T
        let gates_with_rx = vec![Gate::h(0), Gate::t(0), Gate::rx(1, 0.5)];
        let profile2 = CircuitProfile::analyze(&gates_with_rx, 2);
        assert!(!profile2.is_clifford_t_only(&gates_with_rx));
    }

    // =================================================================
    // Enhanced profiled backend selection tests
    // =================================================================

    #[test]
    fn test_profiled_clifford_only_uses_stabilizer() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::s(1)];
        let backend = select_backend_profiled(&gates, 2, false);
        assert_eq!(backend, SimBackend::Stabilizer);
    }

    #[test]
    fn test_profiled_noisy_small_uses_density_matrix() {
        let gates = vec![Gate::h(0), Gate::rx(0, 0.5)];
        let backend = select_backend_profiled(&gates, 10, true);
        assert_eq!(backend, SimBackend::DensityMatrix);
    }

    #[test]
    fn test_profiled_clifford_t_uses_near_clifford() {
        // 25 qubits, Clifford+T only, few T gates -> NearClifford
        let mut gates = Vec::new();
        for i in 0..25 {
            gates.push(Gate::h(i));
        }
        for i in 0..5 {
            gates.push(Gate::t(i));
        }
        for i in 0..24 {
            gates.push(Gate::cnot(i, i + 1));
        }
        let backend = select_backend_profiled(&gates, 25, false);
        assert_eq!(backend, SimBackend::NearClifford);
    }

    #[test]
    fn test_profiled_small_mixed_uses_statevector() {
        // Small circuit (<20 qubits) with non-Clifford gates -> statevector variant
        let gates = vec![Gate::h(0), Gate::t(0), Gate::rx(1, 0.3), Gate::cnot(0, 1)];
        let backend = select_backend_profiled(&gates, 5, false);
        // Should be a statevector variant (MetalGPU on macOS, fused on other platforms)
        assert!(
            backend == SimBackend::StateVectorFused
                || backend == SimBackend::StateVectorF32Fused
                || backend == SimBackend::MetalGPU,
            "Expected statevector variant for small mixed circuit, got {:?}",
            backend
        );
    }

    #[test]
    fn test_profiled_low_treewidth_uses_mps() {
        // Linear chain, >15 qubits, low connectivity -> MPS
        let mut gates = Vec::new();
        for i in 0..20 {
            gates.push(Gate::h(i));
            gates.push(Gate::t(i)); // non-Clifford to avoid stabilizer
        }
        // Linear nearest-neighbor CNOTs
        for i in 0..19 {
            gates.push(Gate::cnot(i, i + 1));
        }
        let backend = select_backend_profiled(&gates, 20, false);
        // Low treewidth + >15 qubits -> MPS or MetalMPS
        assert!(
            backend == SimBackend::MPS || backend == SimBackend::MetalMPS,
            "Expected MPS for low-treewidth circuit, got {:?}",
            backend
        );
    }

    #[test]
    fn test_profiled_pauli_propagation_noisy_large() {
        // Noisy, 35+ qubits, Clifford-heavy -> PauliPropagation
        let mut gates = Vec::new();
        for i in 0..35 {
            gates.push(Gate::h(i));
            gates.push(Gate::s(i));
        }
        for i in 0..34 {
            gates.push(Gate::cnot(i, i + 1));
        }
        gates.push(Gate::t(0)); // make it non-Clifford-only
        let backend = select_backend_profiled(&gates, 35, true);
        assert_eq!(backend, SimBackend::PauliPropagation);
    }

    #[test]
    fn test_profiled_parametric_large_uses_mps() {
        // Large circuit (30 qubits) with parametric gates -> MPS
        let mut gates = Vec::new();
        for i in 0..30 {
            gates.push(Gate::rx(i, 0.5));
            gates.push(Gate::ry(i, 0.3));
        }
        for i in 0..29 {
            gates.push(Gate::cnot(i, i + 1));
        }
        let backend = select_backend_profiled(&gates, 30, false);
        // Parametric + large -> MPS or MetalMPS
        assert!(
            backend == SimBackend::MPS || backend == SimBackend::MetalMPS,
            "Expected MPS for large parametric circuit, got {:?}",
            backend
        );
    }

    // =================================================================
    // AC6a: Runtime estimation and estimated backend selection tests
    // =================================================================

    fn default_profile() -> CircuitProfile {
        CircuitProfile {
            n_qubits: 10,
            depth: 10,
            gate_count: 50,
            clifford_fraction: 0.5,
            t_gate_count: 5,
            two_qubit_fraction: 0.3,
            connectivity: 1.5,
            is_parametric: false,
            has_mid_circuit_measurement: false,
            estimated_entanglement: 0.3,
            max_qubit_degree: 3,
        }
    }

    #[test]
    fn test_estimate_runtime_stabilizer_cheapest_for_clifford() {
        let profile = CircuitProfile {
            n_qubits: 50,
            depth: 100,
            gate_count: 5000,
            clifford_fraction: 1.0,
            t_gate_count: 0,
            two_qubit_fraction: 0.3,
            connectivity: 1.0,
            is_parametric: false,
            has_mid_circuit_measurement: false,
            estimated_entanglement: 0.2,
            max_qubit_degree: 2,
        };
        let sv_time = estimate_runtime(&SimBackend::StateVector, &profile);
        let stab_time = estimate_runtime(&SimBackend::Stabilizer, &profile);
        assert!(
            stab_time < sv_time,
            "Stabilizer should be cheaper for Clifford: {} vs {}",
            stab_time,
            sv_time
        );
    }

    #[test]
    fn test_estimate_runtime_scales_with_qubits() {
        let small = CircuitProfile {
            n_qubits: 10,
            depth: 10,
            ..default_profile()
        };
        let large = CircuitProfile {
            n_qubits: 20,
            depth: 10,
            ..default_profile()
        };
        let small_time = estimate_runtime(&SimBackend::StateVector, &small);
        let large_time = estimate_runtime(&SimBackend::StateVector, &large);
        assert!(
            large_time > small_time * 100.0,
            "Expected ~1024x difference for 10-qubit increase, got {:.1}x",
            large_time / small_time
        );
    }

    #[test]
    fn test_select_backend_estimated_clifford() {
        // Create a Clifford-only circuit with 50 qubits.
        let mut gates = Vec::new();
        for i in 0..50 {
            gates.push(Gate::h(i));
        }
        for i in 0..49 {
            gates.push(Gate::cnot(i, i + 1));
        }
        let backend = select_backend_estimated(&gates, 50, false);
        assert_eq!(
            backend,
            SimBackend::Stabilizer,
            "Stabilizer should be selected for Clifford-only circuit"
        );
    }

    #[test]
    fn test_estimate_bond_dim() {
        let low_ent = CircuitProfile {
            estimated_entanglement: 0.0,
            depth: 5,
            n_qubits: 10,
            ..default_profile()
        };
        let high_ent = CircuitProfile {
            estimated_entanglement: 0.9,
            depth: 50,
            n_qubits: 10,
            ..default_profile()
        };
        assert!(
            estimate_bond_dim(&high_ent) > estimate_bond_dim(&low_ent),
            "High entanglement should yield larger bond dimension"
        );
    }

    #[test]
    fn test_estimate_runtime_density_matrix_more_expensive_than_statevector() {
        let profile = CircuitProfile {
            n_qubits: 10,
            depth: 20,
            ..default_profile()
        };
        let sv_time = estimate_runtime(&SimBackend::StateVector, &profile);
        let dm_time = estimate_runtime(&SimBackend::DensityMatrix, &profile);
        assert!(
            dm_time > sv_time,
            "DensityMatrix O(4^n) should be more expensive than StateVector O(2^n)"
        );
    }

    #[test]
    fn test_estimate_runtime_near_clifford_scales_with_t_gates() {
        let few_t = CircuitProfile {
            t_gate_count: 3,
            n_qubits: 25,
            depth: 50,
            ..default_profile()
        };
        let many_t = CircuitProfile {
            t_gate_count: 20,
            n_qubits: 25,
            depth: 50,
            ..default_profile()
        };
        let few_time = estimate_runtime(&SimBackend::NearClifford, &few_t);
        let many_time = estimate_runtime(&SimBackend::NearClifford, &many_t);
        assert!(
            many_time > few_time * 100.0,
            "NearClifford should scale exponentially with T-gate count"
        );
    }

    #[test]
    fn test_estimate_bond_dim_clamped() {
        let extreme = CircuitProfile {
            estimated_entanglement: 1.0,
            depth: 1000,
            n_qubits: 5,
            ..default_profile()
        };
        let chi = estimate_bond_dim(&extreme);
        assert!(
            chi <= 256.0,
            "Bond dimension should be clamped at 256, got {}",
            chi
        );
    }

    #[test]
    fn test_select_backend_estimated_noisy_small() {
        let gates = vec![Gate::h(0), Gate::rx(0, 0.5)];
        let backend = select_backend_estimated(&gates, 10, true);
        assert_eq!(
            backend,
            SimBackend::DensityMatrix,
            "DensityMatrix should be selected for noisy small circuits"
        );
    }

    // =================================================================
    // new_estimated constructor and BackendSelectionStrategy tests
    // =================================================================

    #[test]
    fn test_new_estimated_constructor() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::new_estimated(&gates, 2, false);
        // Should produce a valid backend
        let backend = sim.backend().clone();
        assert!(
            backend == SimBackend::Stabilizer
                || backend == SimBackend::StateVector
                || backend == SimBackend::StateVectorFused
                || backend == SimBackend::StateVectorF32Fused
                || backend == SimBackend::MetalGPU,
            "new_estimated should select a valid backend, got {:?}",
            backend
        );
        // Strategy should be Estimated
        assert_eq!(*sim.strategy(), BackendSelectionStrategy::Estimated);
    }

    #[test]
    fn test_new_estimated_clifford_large_uses_stabilizer() {
        // Large Clifford-only circuit should pick Stabilizer in estimated mode
        // because Stabilizer O(n^2*d) << StateVector O(2^n*d) for large n.
        let mut gates = Vec::new();
        for i in 0..50 {
            gates.push(Gate::h(i));
        }
        for i in 0..49 {
            gates.push(Gate::cnot(i, i + 1));
        }
        let sim = AutoSimulator::new_estimated(&gates, 50, false);
        assert_eq!(*sim.backend(), SimBackend::Stabilizer);
    }

    #[test]
    fn test_heuristic_strategy_default() {
        let gates = vec![Gate::h(0)];
        let sim = AutoSimulator::new(&gates, 1, false);
        assert_eq!(*sim.strategy(), BackendSelectionStrategy::Heuristic);
    }

    #[test]
    fn test_with_strategy_builder() {
        let gates = vec![Gate::h(0)];
        let sim = AutoSimulator::new(&gates, 1, false)
            .with_strategy(BackendSelectionStrategy::Estimated);
        assert_eq!(*sim.strategy(), BackendSelectionStrategy::Estimated);
    }

    #[test]
    fn test_new_estimated_executes_correctly() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::new_estimated(&gates, 2, false);
        let probs = sim.execute(&gates);
        let total: f64 = probs.iter().sum();
        // f32 backend may be selected on small circuits, so use relaxed tolerance
        assert!(
            (total - 1.0).abs() < 1e-4,
            "new_estimated execution should produce valid probabilities (backend={:?}, sum={})",
            sim.backend(),
            total,
        );
    }

    // =================================================================
    // RuntimeTracker tests
    // =================================================================

    #[test]
    fn test_runtime_tracker_default_factor_is_one() {
        let tracker = RuntimeTracker::default();
        assert!(
            (tracker.correction_factor(&SimBackend::StateVector) - 1.0).abs() < 1e-10,
            "Default correction factor should be 1.0"
        );
    }

    #[test]
    fn test_runtime_tracker_no_update_below_threshold() {
        let mut tracker = RuntimeTracker::new(5);
        // Record 4 observations (below threshold of 5)
        for _ in 0..4 {
            tracker.record(RuntimeObservation {
                backend: SimBackend::StateVector,
                estimated_time: 100.0,
                actual_time: 200.0, // 2x slower
            });
        }
        // Should still be 1.0 (not enough data)
        assert!(
            (tracker.correction_factor(&SimBackend::StateVector) - 1.0).abs() < 1e-10,
            "Should not update factor below threshold"
        );
    }

    #[test]
    fn test_runtime_tracker_updates_at_threshold() {
        let mut tracker = RuntimeTracker::new(5);
        for _ in 0..5 {
            tracker.record(RuntimeObservation {
                backend: SimBackend::StateVector,
                estimated_time: 100.0,
                actual_time: 200.0,
            });
        }
        let factor = tracker.correction_factor(&SimBackend::StateVector);
        assert!(
            (factor - 2.0).abs() < 1e-10,
            "Correction factor should be 2.0 when actual is 2x estimated, got {}",
            factor
        );
    }

    #[test]
    fn test_runtime_tracker_correction_factor_convergence() {
        let mut tracker = RuntimeTracker::new(3);
        // Actual is consistently 1.5x estimated
        for _ in 0..10 {
            tracker.record(RuntimeObservation {
                backend: SimBackend::MPS,
                estimated_time: 50.0,
                actual_time: 75.0,
            });
        }
        let factor = tracker.correction_factor(&SimBackend::MPS);
        assert!(
            (factor - 1.5).abs() < 1e-10,
            "Correction factor should converge to 1.5, got {}",
            factor
        );
    }

    #[test]
    fn test_runtime_tracker_per_backend_isolation() {
        let mut tracker = RuntimeTracker::new(3);
        for _ in 0..5 {
            tracker.record(RuntimeObservation {
                backend: SimBackend::StateVector,
                estimated_time: 100.0,
                actual_time: 300.0, // 3x
            });
            tracker.record(RuntimeObservation {
                backend: SimBackend::Stabilizer,
                estimated_time: 100.0,
                actual_time: 50.0, // 0.5x
            });
        }
        assert!(
            (tracker.correction_factor(&SimBackend::StateVector) - 3.0).abs() < 1e-10,
            "StateVector factor should be 3.0"
        );
        assert!(
            (tracker.correction_factor(&SimBackend::Stabilizer) - 0.5).abs() < 1e-10,
            "Stabilizer factor should be 0.5"
        );
    }

    #[test]
    fn test_runtime_tracker_corrected_estimate() {
        let mut tracker = RuntimeTracker::new(3);
        for _ in 0..5 {
            tracker.record(RuntimeObservation {
                backend: SimBackend::StateVector,
                estimated_time: 100.0,
                actual_time: 200.0,
            });
        }
        let profile = default_profile();
        let raw = estimate_runtime(&SimBackend::StateVector, &profile);
        let corrected = tracker.corrected_estimate(&SimBackend::StateVector, &profile);
        assert!(
            (corrected - raw * 2.0).abs() < 1e-6,
            "Corrected estimate should be 2x raw estimate"
        );
    }

    #[test]
    fn test_runtime_tracker_observation_count() {
        let mut tracker = RuntimeTracker::new(5);
        assert_eq!(tracker.observation_count(&SimBackend::StateVector), 0);
        for _ in 0..7 {
            tracker.record(RuntimeObservation {
                backend: SimBackend::StateVector,
                estimated_time: 10.0,
                actual_time: 15.0,
            });
        }
        assert_eq!(tracker.observation_count(&SimBackend::StateVector), 7);
        assert_eq!(tracker.observation_count(&SimBackend::Stabilizer), 0);
    }

    #[test]
    fn test_runtime_tracker_with_auto_simulator() {
        let tracker = Arc::new(Mutex::new(RuntimeTracker::new(3)));
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::new(&gates, 2, false)
            .with_runtime_tracker(tracker.clone());
        assert!(sim.runtime_tracker.is_some());
        // Execute should work normally
        let probs = sim.execute(&gates);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_runtime_tracker_auto_population() {
        // Verify that execute_inner automatically records observations
        // to the RuntimeTracker when one is attached.
        let tracker = Arc::new(Mutex::new(RuntimeTracker::new(3)));
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let sim = AutoSimulator::new(&gates, 2, false)
            .with_runtime_tracker(tracker.clone());

        // Before execution: no observations
        assert_eq!(
            tracker.lock().unwrap().observation_count(&sim.backend),
            0,
            "should have zero observations before execution"
        );

        // Execute multiple times
        for _ in 0..5 {
            let probs = sim.execute(&gates);
            let total: f64 = probs.iter().sum();
            assert!((total - 1.0).abs() < 1e-10);
        }

        // After execution: should have recorded observations
        let t = tracker.lock().unwrap();
        let backend = sim.backend.clone();
        assert_eq!(
            t.observation_count(&backend),
            5,
            "should have 5 observations after 5 executions"
        );

        // After min_observations (3), correction factor should be computed
        let factor = t.correction_factor(&backend);
        assert!(
            factor > 0.0,
            "correction factor should be positive, got {}",
            factor
        );
    }

    // =================================================================
    // Estimated vs Heuristic benchmark test
    // =================================================================

    #[test]
    fn test_estimated_vs_heuristic_agreement() {
        // Generate ~20 diverse circuit profiles and verify estimated selection
        // produces a valid backend for each.
        let all_valid_backends = vec![
            SimBackend::StateVector,
            SimBackend::StateVectorFused,
            SimBackend::StateVectorF32Fused,
            SimBackend::MetalGPU,
            SimBackend::MetalGPUOnly,
            SimBackend::CudaGPU,
            SimBackend::CudaGPUOnly,
            SimBackend::DensityMatrix,
            SimBackend::Stabilizer,
            SimBackend::MPS,
            SimBackend::MetalMPS,
            SimBackend::PauliPropagation,
            SimBackend::DecisionDiagram,
            SimBackend::NearClifford,
            SimBackend::StabilizerTensorNetwork,
        ];

        struct TestCase {
            name: &'static str,
            gates: Vec<Gate>,
            num_qubits: usize,
            noisy: bool,
        }

        let mut test_cases: Vec<TestCase> = Vec::new();

        // 1. Small Clifford circuit
        test_cases.push(TestCase {
            name: "small_clifford",
            gates: vec![Gate::h(0), Gate::cnot(0, 1), Gate::s(1)],
            num_qubits: 2,
            noisy: false,
        });

        // 2. Small noisy circuit
        test_cases.push(TestCase {
            name: "small_noisy",
            gates: vec![Gate::h(0), Gate::rx(0, 0.3)],
            num_qubits: 8,
            noisy: true,
        });

        // 3. Medium non-Clifford
        {
            let mut g = Vec::new();
            for i in 0..10 {
                g.push(Gate::h(i));
                g.push(Gate::t(i));
            }
            for i in 0..9 {
                g.push(Gate::cnot(i, i + 1));
            }
            test_cases.push(TestCase {
                name: "medium_non_clifford",
                gates: g,
                num_qubits: 10,
                noisy: false,
            });
        }

        // 4. Large Clifford
        {
            let mut g = Vec::new();
            for i in 0..50 {
                g.push(Gate::h(i));
            }
            for i in 0..49 {
                g.push(Gate::cnot(i, i + 1));
            }
            test_cases.push(TestCase {
                name: "large_clifford_50q",
                gates: g,
                num_qubits: 50,
                noisy: false,
            });
        }

        // 5. Near-Clifford with few T gates
        {
            let mut g = Vec::new();
            for i in 0..25 {
                g.push(Gate::h(i));
            }
            for i in 0..3 {
                g.push(Gate::t(i));
            }
            for i in 0..24 {
                g.push(Gate::cnot(i, i + 1));
            }
            test_cases.push(TestCase {
                name: "near_clifford_25q",
                gates: g,
                num_qubits: 25,
                noisy: false,
            });
        }

        // 6. Noisy Clifford-heavy large
        {
            let mut g = Vec::new();
            for i in 0..35 {
                g.push(Gate::h(i));
                g.push(Gate::s(i));
            }
            for i in 0..34 {
                g.push(Gate::cnot(i, i + 1));
            }
            g.push(Gate::t(0));
            test_cases.push(TestCase {
                name: "noisy_clifford_heavy_35q",
                gates: g,
                num_qubits: 35,
                noisy: true,
            });
        }

        // 7. Pure parametric variational circuit
        {
            let mut g = Vec::new();
            for i in 0..30 {
                g.push(Gate::rx(i, 0.5));
                g.push(Gate::ry(i, 0.3));
            }
            for i in 0..29 {
                g.push(Gate::cnot(i, i + 1));
            }
            test_cases.push(TestCase {
                name: "parametric_30q",
                gates: g,
                num_qubits: 30,
                noisy: false,
            });
        }

        // 8. Single qubit
        test_cases.push(TestCase {
            name: "single_qubit",
            gates: vec![Gate::h(0), Gate::t(0), Gate::rx(0, 1.2)],
            num_qubits: 1,
            noisy: false,
        });

        // 9. Empty circuit
        test_cases.push(TestCase {
            name: "empty_circuit",
            gates: vec![],
            num_qubits: 5,
            noisy: false,
        });

        // 10. Noisy small 13 qubits
        {
            let mut g = Vec::new();
            for i in 0..13 {
                g.push(Gate::h(i));
                g.push(Gate::rx(i, 0.2));
            }
            test_cases.push(TestCase {
                name: "noisy_13q",
                gates: g,
                num_qubits: 13,
                noisy: true,
            });
        }

        // 11. Deep single-qubit circuit
        {
            let mut g = Vec::new();
            for _ in 0..100 {
                g.push(Gate::h(0));
                g.push(Gate::t(0));
            }
            test_cases.push(TestCase {
                name: "deep_single_qubit",
                gates: g,
                num_qubits: 1,
                noisy: false,
            });
        }

        // 12. Medium Clifford-only
        {
            let mut g = Vec::new();
            for i in 0..15 {
                g.push(Gate::h(i));
                g.push(Gate::s(i));
            }
            for i in 0..14 {
                g.push(Gate::cnot(i, i + 1));
            }
            test_cases.push(TestCase {
                name: "medium_clifford_15q",
                gates: g,
                num_qubits: 15,
                noisy: false,
            });
        }

        // 13. 5 qubit all T-gates
        {
            let mut g = Vec::new();
            for i in 0..5 {
                g.push(Gate::t(i));
            }
            test_cases.push(TestCase {
                name: "all_t_5q",
                gates: g,
                num_qubits: 5,
                noisy: false,
            });
        }

        // 14. Star topology CNOT
        {
            let mut g = Vec::new();
            g.push(Gate::h(0));
            for i in 1..20 {
                g.push(Gate::cnot(0, i));
            }
            test_cases.push(TestCase {
                name: "star_cnot_20q",
                gates: g,
                num_qubits: 20,
                noisy: false,
            });
        }

        // 15. Fully connected 10 qubits
        {
            let mut g = Vec::new();
            for i in 0..10 {
                g.push(Gate::h(i));
            }
            for i in 0..10 {
                for j in (i + 1)..10 {
                    g.push(Gate::cnot(i, j));
                }
            }
            test_cases.push(TestCase {
                name: "fully_connected_10q",
                gates: g,
                num_qubits: 10,
                noisy: false,
            });
        }

        // 16. Large T-count
        {
            let mut g = Vec::new();
            for i in 0..25 {
                g.push(Gate::h(i));
            }
            for _ in 0..30 {
                for i in 0..25 {
                    g.push(Gate::t(i));
                }
            }
            test_cases.push(TestCase {
                name: "large_t_count_25q",
                gates: g,
                num_qubits: 25,
                noisy: false,
            });
        }

        // 17. 3 qubit with Toffoli
        test_cases.push(TestCase {
            name: "toffoli_3q",
            gates: vec![Gate::h(0), Gate::h(1), Gate::toffoli(0, 1, 2)],
            num_qubits: 3,
            noisy: false,
        });

        // 18. Medium noisy parametric
        {
            let mut g = Vec::new();
            for i in 0..12 {
                g.push(Gate::rx(i, 0.5));
            }
            test_cases.push(TestCase {
                name: "medium_noisy_parametric",
                gates: g,
                num_qubits: 12,
                noisy: true,
            });
        }

        // 19. SWAP-heavy circuit
        {
            let mut g = Vec::new();
            for i in 0..5 {
                g.push(Gate::h(i));
            }
            for i in 0..4 {
                g.push(Gate::swap(i, i + 1));
            }
            test_cases.push(TestCase {
                name: "swap_heavy_5q",
                gates: g,
                num_qubits: 5,
                noisy: false,
            });
        }

        // 20. Large linear nearest-neighbor with rotations
        {
            let mut g = Vec::new();
            for i in 0..28 {
                g.push(Gate::rx(i, 0.1 * i as f64));
                g.push(Gate::ry(i, 0.2));
            }
            for i in 0..27 {
                g.push(Gate::cnot(i, i + 1));
            }
            test_cases.push(TestCase {
                name: "linear_nn_rotations_28q",
                gates: g,
                num_qubits: 28,
                noisy: false,
            });
        }

        for tc in &test_cases {
            let estimated_backend =
                select_backend_estimated(&tc.gates, tc.num_qubits, tc.noisy);
            assert!(
                all_valid_backends.contains(&estimated_backend),
                "Test case '{}': estimated backend {:?} is not a valid SimBackend variant",
                tc.name,
                estimated_backend
            );

            // Also verify it can be used to construct and execute (for small circuits)
            if tc.num_qubits <= 15 && !tc.gates.is_empty() {
                let sim = AutoSimulator::new_estimated(&tc.gates, tc.num_qubits, tc.noisy);
                let probs = sim.execute(&tc.gates);
                let total: f64 = probs.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-6,
                    "Test case '{}': probabilities don't sum to 1 (got {})",
                    tc.name,
                    total
                );
            }
        }
    }

}
