//! UMA Gate-Level CPU/GPU Dispatch for Apple Silicon
//!
//! Exploits Unified Memory Architecture for per-gate routing decisions.
//! On Apple Silicon the state vector resides in a single unified address space
//! (Metal `StorageModeShared`), so both CPU and GPU can touch it with zero-copy.
//! This is impossible on discrete-GPU platforms (CUDA/ROCm) where PCIe forces
//! the entire computation onto the GPU side.
//!
//! # Dispatch Strategy
//!
//! | Gate class                | Target | Rationale                                       |
//! |---------------------------|--------|-------------------------------------------------|
//! | Single-qubit              | CPU    | ~50 ns on CPU; GPU dispatch overhead ~5 us      |
//! | Diagonal (Rz, CZ, Phase)  | CPU    | Element-wise multiply, cache-friendly            |
//! | Multi-qubit (>threshold)  | GPU    | Massive parallelism pays off at scale            |
//! | Disjoint pair             | Both   | CPU handles one gate while GPU handles another   |
//!
//! # Adaptive Learning
//!
//! When `enable_adaptive` is set, observed wall-clock times refine the cost
//! model so that thresholds converge to the actual hardware breakeven point.
//!
//! # GPU Fallback
//!
//! Since the Metal compute API is not linked in this crate, the GPU path is a
//! documented stub that falls back to CPU execution via
//! [`apply_gate_to_state`](crate::ascii_viz::apply_gate_to_state).
//! The dispatch *decision* logic is fully real and production-ready.

use crate::ascii_viz::apply_gate_to_state;
use crate::gates::{Gate, GateType};
use crate::QuantumState;
use std::fmt;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising during UMA dispatch operations.
#[derive(Debug, Clone)]
pub enum UmaError {
    /// A gate references a qubit index that exceeds the state size.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// The circuit is invalid (e.g. gate has no targets).
    InvalidGate(String),
    /// GPU execution failed (stub -- would wrap Metal errors).
    GpuExecutionFailed(String),
    /// Concurrent dispatch scheduling failed.
    ConcurrentSchedulingFailed(String),
}

impl fmt::Display for UmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QubitOutOfRange { qubit, num_qubits } => {
                write!(
                    f,
                    "qubit index {} out of range for {}-qubit state",
                    qubit, num_qubits
                )
            }
            Self::InvalidGate(msg) => write!(f, "invalid gate: {}", msg),
            Self::GpuExecutionFailed(msg) => write!(f, "GPU execution failed: {}", msg),
            Self::ConcurrentSchedulingFailed(msg) => {
                write!(f, "concurrent scheduling failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for UmaError {}

// ============================================================
// DISPATCH TARGET
// ============================================================

/// Where a gate executes within the unified memory architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchTarget {
    /// Execute entirely on CPU cores.
    Cpu,
    /// Execute on the Metal GPU.
    Gpu,
    /// Execute concurrently: CPU handles one qubit range, GPU another.
    Concurrent {
        cpu_qubits: (usize, usize),
        gpu_qubits: (usize, usize),
    },
}

impl fmt::Display for DispatchTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Concurrent {
                cpu_qubits,
                gpu_qubits,
            } => {
                write!(
                    f,
                    "Concurrent(CPU:{}-{}, GPU:{}-{})",
                    cpu_qubits.0, cpu_qubits.1, gpu_qubits.0, gpu_qubits.1
                )
            }
        }
    }
}

// ============================================================
// COST MODEL
// ============================================================

/// Cost model for routing decisions.  These defaults are calibrated for
/// Apple M-series chips; users can override for specific hardware.
#[derive(Clone, Debug)]
pub struct DispatchCostModel {
    /// GPU command buffer dispatch overhead in microseconds (~5 us on M4 Pro).
    pub gpu_dispatch_overhead_us: f64,
    /// Throughput multiplier for multi-qubit gates on GPU vs CPU.
    pub gpu_throughput_factor: f64,
    /// CPU cost to apply a single-qubit gate, in nanoseconds.
    pub cpu_single_qubit_ns: f64,
    /// Qubit-count threshold: gates touching more qubits than this route to GPU.
    pub qubit_threshold: usize,
}

impl Default for DispatchCostModel {
    fn default() -> Self {
        Self {
            gpu_dispatch_overhead_us: 5.0,
            gpu_throughput_factor: 10.0,
            cpu_single_qubit_ns: 50.0,
            qubit_threshold: 4, // log2(16) -- gates touching >4 qubits go to GPU
        }
    }
}

impl DispatchCostModel {
    /// Builder: set GPU dispatch overhead.
    pub fn with_gpu_dispatch_overhead_us(mut self, us: f64) -> Self {
        self.gpu_dispatch_overhead_us = us;
        self
    }

    /// Builder: set GPU throughput factor.
    pub fn with_gpu_throughput_factor(mut self, factor: f64) -> Self {
        self.gpu_throughput_factor = factor;
        self
    }

    /// Builder: set CPU single-qubit gate cost.
    pub fn with_cpu_single_qubit_ns(mut self, ns: f64) -> Self {
        self.cpu_single_qubit_ns = ns;
        self
    }

    /// Builder: set qubit threshold for GPU routing.
    pub fn with_qubit_threshold(mut self, threshold: usize) -> Self {
        self.qubit_threshold = threshold;
        self
    }
}

// ============================================================
// DISPATCH CONFIG
// ============================================================

/// Configuration for the UMA dispatcher.
#[derive(Clone, Debug)]
pub struct UmaDispatchConfig {
    /// Cost model parameters.
    pub cost_model: DispatchCostModel,
    /// Allow simultaneous CPU + GPU execution on disjoint qubit subsets.
    pub enable_concurrent: bool,
    /// Learn thresholds from measured execution times.
    pub enable_adaptive: bool,
    /// Total number of qubits in the system.
    pub num_qubits: usize,
}

impl Default for UmaDispatchConfig {
    fn default() -> Self {
        Self {
            cost_model: DispatchCostModel::default(),
            enable_concurrent: false,
            enable_adaptive: false,
            num_qubits: 10,
        }
    }
}

impl UmaDispatchConfig {
    /// Builder: set the cost model.
    pub fn with_cost_model(mut self, cost_model: DispatchCostModel) -> Self {
        self.cost_model = cost_model;
        self
    }

    /// Builder: enable or disable concurrent CPU+GPU execution.
    pub fn with_concurrent(mut self, enable: bool) -> Self {
        self.enable_concurrent = enable;
        self
    }

    /// Builder: enable or disable adaptive threshold learning.
    pub fn with_adaptive(mut self, enable: bool) -> Self {
        self.enable_adaptive = enable;
        self
    }

    /// Builder: set the number of qubits.
    pub fn with_num_qubits(mut self, n: usize) -> Self {
        self.num_qubits = n;
        self
    }
}

// ============================================================
// GATE ANALYSIS
// ============================================================

/// Analysis of a single gate's properties relevant to dispatch routing.
#[derive(Clone, Debug)]
pub struct GateAnalysis {
    /// Gate type name (for diagnostics).
    pub gate_type: String,
    /// All qubit indices this gate operates on (targets + controls, sorted, deduped).
    pub target_qubits: Vec<usize>,
    /// Span: max qubit index - min qubit index (0 for single-qubit gates).
    pub qubit_span: usize,
    /// True if the gate touches exactly one qubit.
    pub is_single_qubit: bool,
    /// True if the gate matrix is diagonal (element-wise phase multiply).
    pub is_diagonal: bool,
    /// Fraction of total system qubits that this gate touches.
    pub touches_fraction: f64,
}

/// Check whether a `GateType` is diagonal in the computational basis.
/// Diagonal gates only modify phases, never create superpositions.
fn gate_type_is_diagonal(gt: &GateType) -> bool {
    matches!(
        gt,
        GateType::Z
            | GateType::S
            | GateType::T
            | GateType::Rz(_)
            | GateType::Phase(_)
            | GateType::CZ
            | GateType::CRz(_)
            | GateType::CR(_)
            | GateType::CCZ
    )
}

/// Return a human-readable name for a gate type.
fn gate_type_name(gt: &GateType) -> String {
    match gt {
        GateType::H => "H".to_string(),
        GateType::X => "X".to_string(),
        GateType::Y => "Y".to_string(),
        GateType::Z => "Z".to_string(),
        GateType::S => "S".to_string(),
        GateType::T => "T".to_string(),
        GateType::Rx(a) => format!("Rx({:.4})", a),
        GateType::Ry(a) => format!("Ry({:.4})", a),
        GateType::Rz(a) => format!("Rz({:.4})", a),
        GateType::U { .. } => "U".to_string(),
        GateType::CNOT => "CNOT".to_string(),
        GateType::CZ => "CZ".to_string(),
        GateType::SWAP => "SWAP".to_string(),
        GateType::Toffoli => "Toffoli".to_string(),
        GateType::CRx(a) => format!("CRx({:.4})", a),
        GateType::CRy(a) => format!("CRy({:.4})", a),
        GateType::CRz(a) => format!("CRz({:.4})", a),
        GateType::CR(a) => format!("CR({:.4})", a),
        GateType::SX => "SX".to_string(),
        GateType::Phase(a) => format!("Phase({:.4})", a),
        GateType::ISWAP => "ISWAP".to_string(),
        GateType::CCZ => "CCZ".to_string(),
        GateType::Rxx(a) => format!("Rxx({:.4})", a),
        GateType::Ryy(a) => format!("Ryy({:.4})", a),
        GateType::Rzz(a) => format!("Rzz({:.4})", a),
        GateType::CSWAP => "CSWAP".to_string(),
        GateType::CU { .. } => "CU".to_string(),
        GateType::Custom(_) => "Custom".to_string(),
    }
}

/// Collect all qubit indices a gate touches (targets union controls), sorted and deduped.
fn all_qubits(gate: &Gate) -> Vec<usize> {
    let mut qs: Vec<usize> = gate.targets.iter().copied().collect();
    qs.extend(gate.controls.iter().copied());
    qs.sort_unstable();
    qs.dedup();
    qs
}

// ============================================================
// DISPATCH DECISION
// ============================================================

/// The result of a dispatch routing decision for a single gate.
#[derive(Clone, Debug)]
pub struct DispatchDecision {
    /// Where the gate should execute.
    pub target: DispatchTarget,
    /// Estimated execution cost in nanoseconds.
    pub estimated_cost_ns: f64,
    /// Human-readable reasoning for the decision.
    pub reasoning: String,
}

// ============================================================
// DISPATCH STATISTICS
// ============================================================

/// Accumulated statistics from dispatching a circuit.
#[derive(Clone, Debug)]
pub struct DispatchStats {
    pub total_gates: usize,
    pub cpu_gates: usize,
    pub gpu_gates: usize,
    pub concurrent_gates: usize,
    pub total_cpu_time_ns: f64,
    pub total_gpu_time_ns: f64,
    /// What total GPU-only execution would have cost.
    pub estimated_gpu_only_time_ns: f64,
    /// What total CPU-only execution would have cost.
    pub estimated_cpu_only_time_ns: f64,
    /// Speedup of UMA dispatch over pure GPU execution.
    pub speedup_vs_gpu_only: f64,
    /// Speedup of UMA dispatch over pure CPU execution.
    pub speedup_vs_cpu_only: f64,
}

impl Default for DispatchStats {
    fn default() -> Self {
        Self {
            total_gates: 0,
            cpu_gates: 0,
            gpu_gates: 0,
            concurrent_gates: 0,
            total_cpu_time_ns: 0.0,
            total_gpu_time_ns: 0.0,
            estimated_gpu_only_time_ns: 0.0,
            estimated_cpu_only_time_ns: 0.0,
            speedup_vs_gpu_only: 1.0,
            speedup_vs_cpu_only: 1.0,
        }
    }
}

impl fmt::Display for DispatchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DispatchStats {{ gates: {} (CPU:{}, GPU:{}, concurrent:{}), \
             speedup vs GPU-only: {:.2}x, vs CPU-only: {:.2}x }}",
            self.total_gates,
            self.cpu_gates,
            self.gpu_gates,
            self.concurrent_gates,
            self.speedup_vs_gpu_only,
            self.speedup_vs_cpu_only,
        )
    }
}

// ============================================================
// GATE LAYER
// ============================================================

/// A layer of independent (non-overlapping) gates that can execute in parallel.
#[derive(Clone, Debug)]
pub struct GateLayer {
    /// Gates in this layer: `(original_index_in_circuit, gate)`.
    pub gates: Vec<(usize, Gate)>,
    /// Per-qubit mask: `qubit_mask[q]` is true if qubit `q` is touched.
    pub qubit_mask: Vec<bool>,
}

/// Group a flat gate list into layers of non-overlapping operations.
///
/// Two gates are in the same layer if and only if they operate on
/// completely disjoint qubit sets.  This is a greedy front-to-back
/// assignment: each gate goes into the earliest layer where it fits.
pub fn layer_circuit(gates: &[Gate], num_qubits: usize) -> Vec<GateLayer> {
    let mut layers: Vec<GateLayer> = Vec::new();

    for (idx, gate) in gates.iter().enumerate() {
        let qs = all_qubits(gate);

        // Find the first layer where none of these qubits are occupied.
        let mut placed = false;
        for layer in layers.iter_mut() {
            let conflict = qs.iter().any(|&q| q < num_qubits && layer.qubit_mask[q]);
            if !conflict {
                for &q in &qs {
                    if q < num_qubits {
                        layer.qubit_mask[q] = true;
                    }
                }
                layer.gates.push((idx, gate.clone()));
                placed = true;
                break;
            }
        }

        if !placed {
            let mut mask = vec![false; num_qubits];
            for &q in &qs {
                if q < num_qubits {
                    mask[q] = true;
                }
            }
            layers.push(GateLayer {
                gates: vec![(idx, gate.clone())],
                qubit_mask: mask,
            });
        }
    }

    layers
}

/// Check if two gates operate on completely disjoint qubit sets.
pub fn gates_disjoint(g1: &Gate, g2: &Gate) -> bool {
    let qs1 = all_qubits(g1);
    let qs2 = all_qubits(g2);
    for &q in &qs1 {
        if qs2.contains(&q) {
            return false;
        }
    }
    true
}

// ============================================================
// UMA DISPATCHER
// ============================================================

/// Per-gate CPU/GPU dispatcher exploiting Apple Silicon unified memory.
///
/// The dispatcher analyzes each gate, estimates whether CPU or GPU
/// execution is cheaper, and routes accordingly.  Because the state
/// vector lives in shared memory, the CPU path avoids the ~5 us
/// Metal command-buffer dispatch overhead entirely.
pub struct UmaDispatcher {
    config: UmaDispatchConfig,
    stats: DispatchStats,
    // Adaptive learning: observed wall-clock times for CPU and GPU paths.
    observed_cpu_times: Vec<f64>,
    observed_gpu_times: Vec<f64>,
}

impl UmaDispatcher {
    /// Create a new dispatcher with the given configuration.
    pub fn new(config: UmaDispatchConfig) -> Self {
        Self {
            config,
            stats: DispatchStats::default(),
            observed_cpu_times: Vec::new(),
            observed_gpu_times: Vec::new(),
        }
    }

    /// Analyze a single gate for dispatch-relevant properties.
    pub fn analyze_gate(&self, gate: &Gate, num_qubits: usize) -> GateAnalysis {
        let qs = all_qubits(gate);
        let span = if qs.len() <= 1 {
            0
        } else {
            qs.last().copied().unwrap_or(0) - qs.first().copied().unwrap_or(0)
        };
        let touches_fraction = if num_qubits == 0 {
            0.0
        } else {
            qs.len() as f64 / num_qubits as f64
        };

        GateAnalysis {
            gate_type: gate_type_name(&gate.gate_type),
            target_qubits: qs,
            qubit_span: span,
            is_single_qubit: gate.is_single_qubit(),
            is_diagonal: gate_type_is_diagonal(&gate.gate_type),
            touches_fraction,
        }
    }

    /// Decide where a gate should execute based on its analysis.
    ///
    /// Decision rules (priority order):
    /// 1. Single-qubit gates always go to CPU (avoid GPU dispatch overhead).
    /// 2. Diagonal multi-qubit gates go to CPU (element-wise, cache-friendly).
    /// 3. Gates touching more qubits than the threshold go to GPU.
    /// 4. Everything else defaults to CPU.
    pub fn decide(&self, analysis: &GateAnalysis) -> DispatchDecision {
        let cm = &self.config.cost_model;
        let gpu_overhead_ns = cm.gpu_dispatch_overhead_us * 1000.0;

        // Rule 1: single-qubit gate -> CPU
        if analysis.is_single_qubit {
            return DispatchDecision {
                target: DispatchTarget::Cpu,
                estimated_cost_ns: cm.cpu_single_qubit_ns,
                reasoning: format!(
                    "{}: single-qubit -> CPU (~{:.0} ns, avoids {:.0} ns GPU overhead)",
                    analysis.gate_type, cm.cpu_single_qubit_ns, gpu_overhead_ns
                ),
            };
        }

        // Rule 2: diagonal gates -> CPU (element-wise phase multiply)
        if analysis.is_diagonal {
            let n_touched = analysis.target_qubits.len();
            let cpu_cost = cm.cpu_single_qubit_ns * n_touched as f64 * 2.0;
            return DispatchDecision {
                target: DispatchTarget::Cpu,
                estimated_cost_ns: cpu_cost,
                reasoning: format!(
                    "{}: diagonal {}-qubit -> CPU (element-wise, cache-friendly, ~{:.0} ns)",
                    analysis.gate_type, n_touched, cpu_cost
                ),
            };
        }

        // Rule 3: multi-qubit gate touching many qubits -> GPU
        let n_touched = analysis.target_qubits.len();
        let effective_threshold = if self.config.enable_adaptive
            && !self.observed_cpu_times.is_empty()
            && !self.observed_gpu_times.is_empty()
        {
            self.adaptive_threshold()
        } else {
            cm.qubit_threshold
        };

        if n_touched > effective_threshold {
            let state_dim = 1usize << self.config.num_qubits;
            let gpu_cost_ns =
                gpu_overhead_ns + (state_dim as f64 * 0.5 / cm.gpu_throughput_factor);
            return DispatchDecision {
                target: DispatchTarget::Gpu,
                estimated_cost_ns: gpu_cost_ns,
                reasoning: format!(
                    "{}: {}-qubit (>{} threshold) -> GPU (throughput {:.0}x, ~{:.0} ns)",
                    analysis.gate_type,
                    n_touched,
                    effective_threshold,
                    cm.gpu_throughput_factor,
                    gpu_cost_ns
                ),
            };
        }

        // Rule 4: moderate multi-qubit gate -> CPU
        let cpu_cost = cm.cpu_single_qubit_ns * (1usize << n_touched) as f64;
        DispatchDecision {
            target: DispatchTarget::Cpu,
            estimated_cost_ns: cpu_cost,
            reasoning: format!(
                "{}: {}-qubit (<={} threshold) -> CPU (~{:.0} ns)",
                analysis.gate_type, n_touched, effective_threshold, cpu_cost
            ),
        }
    }

    /// Dispatch and execute an entire circuit, returning accumulated statistics.
    ///
    /// Currently the GPU path falls back to CPU execution because we do not
    /// link against the Metal API in this crate.  The dispatch *decision* logic
    /// is fully real; only the final execution step is stubbed.
    pub fn dispatch_circuit(
        &mut self,
        gates: &[Gate],
        state: &mut QuantumState,
    ) -> Result<DispatchStats, UmaError> {
        let num_qubits = state.num_qubits;
        self.reset_stats();

        // Validate all gates first.
        for (i, gate) in gates.iter().enumerate() {
            let qs = all_qubits(gate);
            if qs.is_empty() {
                return Err(UmaError::InvalidGate(format!(
                    "gate {} has no target qubits",
                    i
                )));
            }
            for &q in &qs {
                if q >= num_qubits {
                    return Err(UmaError::QubitOutOfRange {
                        qubit: q,
                        num_qubits,
                    });
                }
            }
        }

        let cm = &self.config.cost_model.clone();
        let gpu_overhead_ns = cm.gpu_dispatch_overhead_us * 1000.0;

        for gate in gates {
            let analysis = self.analyze_gate(gate, num_qubits);
            let decision = self.decide(&analysis);

            // Estimate what each pure strategy would cost for this gate.
            let cpu_only_cost = if analysis.is_single_qubit {
                cm.cpu_single_qubit_ns
            } else {
                let n = analysis.target_qubits.len();
                cm.cpu_single_qubit_ns * (1usize << n) as f64
            };
            let state_dim = 1usize << num_qubits;
            let gpu_only_cost =
                gpu_overhead_ns + (state_dim as f64 * 0.5 / cm.gpu_throughput_factor);

            self.stats.estimated_cpu_only_time_ns += cpu_only_cost;
            self.stats.estimated_gpu_only_time_ns += gpu_only_cost;
            self.stats.total_gates += 1;

            match decision.target {
                DispatchTarget::Cpu => {
                    self.stats.cpu_gates += 1;
                    self.stats.total_cpu_time_ns += decision.estimated_cost_ns;
                }
                DispatchTarget::Gpu => {
                    self.stats.gpu_gates += 1;
                    self.stats.total_gpu_time_ns += decision.estimated_cost_ns;
                }
                DispatchTarget::Concurrent { .. } => {
                    self.stats.concurrent_gates += 1;
                    // Concurrent time is the max of the two; approximate with
                    // the larger sub-cost.
                    self.stats.total_cpu_time_ns += decision.estimated_cost_ns * 0.5;
                    self.stats.total_gpu_time_ns += decision.estimated_cost_ns * 0.5;
                }
            }

            // Execute the gate.  GPU path falls back to CPU for now.
            apply_gate_to_state(state, gate);
        }

        // Compute speedup ratios.
        let actual_time_ns = self.stats.total_cpu_time_ns + self.stats.total_gpu_time_ns;
        if actual_time_ns > 0.0 {
            if self.stats.estimated_gpu_only_time_ns > 0.0 {
                self.stats.speedup_vs_gpu_only =
                    self.stats.estimated_gpu_only_time_ns / actual_time_ns;
            }
            if self.stats.estimated_cpu_only_time_ns > 0.0 {
                self.stats.speedup_vs_cpu_only =
                    self.stats.estimated_cpu_only_time_ns / actual_time_ns;
            }
        }

        Ok(self.stats.clone())
    }

    /// Analyze and decide dispatch for every gate in a circuit.
    pub fn batch_analyze(
        &self,
        gates: &[Gate],
        num_qubits: usize,
    ) -> Vec<(GateAnalysis, DispatchDecision)> {
        gates
            .iter()
            .map(|g| {
                let a = self.analyze_gate(g, num_qubits);
                let d = self.decide(&a);
                (a, d)
            })
            .collect()
    }

    /// Find pairs of gates on disjoint qubit sets that could execute concurrently.
    ///
    /// Returns pairs of indices into the `gates` slice.  Uses layer analysis:
    /// within each layer all gates are on disjoint qubits by construction.
    pub fn find_concurrent_pairs(
        &self,
        gates: &[Gate],
        num_qubits: usize,
    ) -> Vec<(usize, usize)> {
        if !self.config.enable_concurrent {
            return Vec::new();
        }

        let mut pairs = Vec::new();
        let layers = layer_circuit(gates, num_qubits);

        for layer in &layers {
            if layer.gates.len() < 2 {
                continue;
            }
            // Within each layer all gates are on disjoint qubits by construction.
            // Emit all unique index pairs.
            for i in 0..layer.gates.len() {
                for j in (i + 1)..layer.gates.len() {
                    pairs.push((layer.gates[i].0, layer.gates[j].0));
                }
            }
        }
        pairs
    }

    /// Feed back an observed execution time for adaptive threshold learning.
    pub fn update_adaptive(&mut self, target: DispatchTarget, actual_time_ns: f64) {
        match target {
            DispatchTarget::Cpu => self.observed_cpu_times.push(actual_time_ns),
            DispatchTarget::Gpu => self.observed_gpu_times.push(actual_time_ns),
            DispatchTarget::Concurrent { .. } => {
                // Split evenly as a rough heuristic.
                self.observed_cpu_times.push(actual_time_ns * 0.5);
                self.observed_gpu_times.push(actual_time_ns * 0.5);
            }
        }
    }

    /// Get a reference to the accumulated statistics.
    pub fn stats(&self) -> &DispatchStats {
        &self.stats
    }

    /// Reset all accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DispatchStats::default();
    }

    // ----------------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------------

    /// Compute an adaptive qubit threshold from observed times.
    ///
    /// If average observed CPU time is close to or less than average
    /// observed GPU time, raise the threshold (prefer CPU for more gates).
    /// Otherwise lower it (let GPU handle more).
    fn adaptive_threshold(&self) -> usize {
        let base = self.config.cost_model.qubit_threshold;
        if self.observed_cpu_times.is_empty() || self.observed_gpu_times.is_empty() {
            return base;
        }
        let avg_cpu: f64 =
            self.observed_cpu_times.iter().sum::<f64>() / self.observed_cpu_times.len() as f64;
        let avg_gpu: f64 =
            self.observed_gpu_times.iter().sum::<f64>() / self.observed_gpu_times.len() as f64;

        if avg_cpu <= avg_gpu {
            // CPU is faster on average -- raise threshold so more gates stay on CPU.
            (base + 1).min(16)
        } else {
            // GPU is faster on average -- lower threshold to offload more.
            base.saturating_sub(1).max(1)
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};

    fn default_dispatcher(num_qubits: usize) -> UmaDispatcher {
        let config = UmaDispatchConfig::default().with_num_qubits(num_qubits);
        UmaDispatcher::new(config)
    }

    // ---- GateAnalysis tests ----

    #[test]
    fn test_analysis_single_qubit() {
        let d = default_dispatcher(8);
        let g = Gate::h(3);
        let a = d.analyze_gate(&g, 8);
        assert!(a.is_single_qubit);
        assert_eq!(a.target_qubits, vec![3]);
        assert_eq!(a.qubit_span, 0);
        assert!(!a.is_diagonal);
        assert!((a.touches_fraction - 1.0 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_analysis_two_qubit_cnot() {
        let d = default_dispatcher(8);
        let g = Gate::cnot(1, 5);
        let a = d.analyze_gate(&g, 8);
        assert!(!a.is_single_qubit);
        assert_eq!(a.target_qubits, vec![1, 5]);
        assert_eq!(a.qubit_span, 4);
        assert!(!a.is_diagonal);
        assert!((a.touches_fraction - 2.0 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_analysis_multi_qubit_toffoli() {
        let d = default_dispatcher(10);
        let g = Gate::toffoli(0, 4, 9);
        let a = d.analyze_gate(&g, 10);
        assert!(!a.is_single_qubit);
        assert_eq!(a.target_qubits.len(), 3);
        assert_eq!(a.qubit_span, 9);
        assert!((a.touches_fraction - 3.0 / 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_analysis_diagonal_rz() {
        let d = default_dispatcher(4);
        let g = Gate::rz(2, 0.5);
        let a = d.analyze_gate(&g, 4);
        assert!(a.is_single_qubit);
        assert!(a.is_diagonal);
    }

    #[test]
    fn test_analysis_diagonal_cz() {
        let d = default_dispatcher(8);
        let g = Gate::cz(0, 1);
        let a = d.analyze_gate(&g, 8);
        assert!(!a.is_single_qubit);
        assert!(a.is_diagonal);
    }

    #[test]
    fn test_analysis_diagonal_phase() {
        let d = default_dispatcher(4);
        let g = Gate::phase(0, 1.0);
        let a = d.analyze_gate(&g, 4);
        assert!(a.is_single_qubit);
        assert!(a.is_diagonal);
    }

    // ---- DispatchDecision tests ----

    #[test]
    fn test_decide_single_qubit_goes_to_cpu() {
        let d = default_dispatcher(20);
        let g = Gate::h(0);
        let a = d.analyze_gate(&g, 20);
        let dec = d.decide(&a);
        assert_eq!(dec.target, DispatchTarget::Cpu);
        assert!(dec.reasoning.contains("single-qubit"));
    }

    #[test]
    fn test_decide_diagonal_goes_to_cpu() {
        let d = default_dispatcher(20);
        let g = Gate::cz(0, 1);
        let a = d.analyze_gate(&g, 20);
        let dec = d.decide(&a);
        assert_eq!(dec.target, DispatchTarget::Cpu);
        assert!(dec.reasoning.contains("diagonal"));
    }

    #[test]
    fn test_decide_large_entangling_goes_to_gpu() {
        // Custom gate touching 6 qubits (exceeds default threshold of 4).
        let g = Gate::new(GateType::Custom(vec![]), vec![0, 1, 2, 3, 4, 5], vec![]);
        let d = default_dispatcher(20);
        let a = d.analyze_gate(&g, 20);
        assert_eq!(a.target_qubits.len(), 6);
        let dec = d.decide(&a);
        assert_eq!(dec.target, DispatchTarget::Gpu);
        assert!(dec.reasoning.contains("GPU"));
    }

    #[test]
    fn test_decide_moderate_multi_qubit_stays_cpu() {
        // CNOT touches 2 qubits, well below default threshold of 4.
        let d = default_dispatcher(20);
        let g = Gate::cnot(0, 1);
        let a = d.analyze_gate(&g, 20);
        let dec = d.decide(&a);
        assert_eq!(dec.target, DispatchTarget::Cpu);
    }

    // ---- gates_disjoint tests ----

    #[test]
    fn test_gates_disjoint_non_overlapping() {
        let g1 = Gate::h(0);
        let g2 = Gate::h(1);
        assert!(gates_disjoint(&g1, &g2));
    }

    #[test]
    fn test_gates_disjoint_overlapping() {
        let g1 = Gate::cnot(0, 1);
        let g2 = Gate::h(1);
        assert!(!gates_disjoint(&g1, &g2));
    }

    #[test]
    fn test_gates_disjoint_complex() {
        let g1 = Gate::cnot(0, 1);
        let g2 = Gate::cnot(2, 3);
        assert!(gates_disjoint(&g1, &g2));
    }

    // ---- layer_circuit tests ----

    #[test]
    fn test_layer_independent_gates_grouped() {
        // H(0), H(1), H(2) are on disjoint qubits -> same layer.
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2)];
        let layers = layer_circuit(&gates, 4);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].gates.len(), 3);
    }

    #[test]
    fn test_layer_dependent_gates_separated() {
        // CNOT(0,1) and H(1) overlap on qubit 1 -> different layers.
        let gates = vec![Gate::cnot(0, 1), Gate::h(1)];
        let layers = layer_circuit(&gates, 4);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].gates.len(), 1);
        assert_eq!(layers[1].gates.len(), 1);
    }

    #[test]
    fn test_layer_mixed_dependencies() {
        // H(0), H(1), CNOT(0,1), H(2)
        // Layer 0: H(0), H(1), H(2) -- all disjoint
        // Layer 1: CNOT(0,1)         -- overlaps with H(0) and H(1)
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1), Gate::h(2)];
        let layers = layer_circuit(&gates, 4);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].gates.len(), 3);
        assert_eq!(layers[1].gates.len(), 1);
    }

    // ---- find_concurrent_pairs tests ----

    #[test]
    fn test_find_concurrent_pairs_disabled() {
        let d = default_dispatcher(4);
        let gates = vec![Gate::h(0), Gate::h(1)];
        let pairs = d.find_concurrent_pairs(&gates, 4);
        assert!(pairs.is_empty(), "concurrent disabled by default");
    }

    #[test]
    fn test_find_concurrent_pairs_enabled() {
        let config = UmaDispatchConfig::default()
            .with_num_qubits(4)
            .with_concurrent(true);
        let d = UmaDispatcher::new(config);
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(2, 3)];
        let pairs = d.find_concurrent_pairs(&gates, 4);
        // All three gates are on disjoint qubits -> one layer with 3 pairs.
        assert_eq!(pairs.len(), 3);
    }

    // ---- dispatch_circuit tests ----

    #[test]
    fn test_dispatch_circuit_basic() {
        let config = UmaDispatchConfig::default().with_num_qubits(4);
        let mut d = UmaDispatcher::new(config);
        let mut state = QuantumState::new(4);
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::h(2)];

        let stats = d.dispatch_circuit(&gates, &mut state).unwrap();
        assert_eq!(stats.total_gates, 3);
        assert!(stats.cpu_gates > 0);
        // State should be modified.
        let probs = state.probabilities();
        assert!(probs.iter().any(|&p| p > 0.0));
    }

    #[test]
    fn test_dispatch_circuit_empty() {
        let config = UmaDispatchConfig::default().with_num_qubits(4);
        let mut d = UmaDispatcher::new(config);
        let mut state = QuantumState::new(4);

        let stats = d.dispatch_circuit(&[], &mut state).unwrap();
        assert_eq!(stats.total_gates, 0);
        assert_eq!(stats.cpu_gates, 0);
        assert_eq!(stats.gpu_gates, 0);
    }

    #[test]
    fn test_dispatch_circuit_single_gate() {
        let config = UmaDispatchConfig::default().with_num_qubits(2);
        let mut d = UmaDispatcher::new(config);
        let mut state = QuantumState::new(2);

        let stats = d.dispatch_circuit(&[Gate::h(0)], &mut state).unwrap();
        assert_eq!(stats.total_gates, 1);
        assert_eq!(stats.cpu_gates, 1);
        assert_eq!(stats.gpu_gates, 0);
    }

    #[test]
    fn test_dispatch_circuit_qubit_out_of_range() {
        let config = UmaDispatchConfig::default().with_num_qubits(4);
        let mut d = UmaDispatcher::new(config);
        let mut state = QuantumState::new(4);

        let result = d.dispatch_circuit(&[Gate::h(10)], &mut state);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                UmaError::QubitOutOfRange {
                    qubit: 10,
                    num_qubits: 4
                }
            ),
            "expected QubitOutOfRange, got {:?}",
            err
        );
    }

    // ---- Adaptive threshold learning ----

    #[test]
    fn test_adaptive_raises_threshold_when_cpu_faster() {
        let config = UmaDispatchConfig::default()
            .with_num_qubits(10)
            .with_adaptive(true);
        let mut d = UmaDispatcher::new(config);

        // Simulate CPU being consistently faster.
        for _ in 0..10 {
            d.update_adaptive(DispatchTarget::Cpu, 20.0);
            d.update_adaptive(DispatchTarget::Gpu, 100.0);
        }

        let threshold = d.adaptive_threshold();
        assert!(
            threshold > DispatchCostModel::default().qubit_threshold,
            "expected raised threshold, got {}",
            threshold
        );
    }

    #[test]
    fn test_adaptive_lowers_threshold_when_gpu_faster() {
        let config = UmaDispatchConfig::default()
            .with_num_qubits(10)
            .with_adaptive(true);
        let mut d = UmaDispatcher::new(config);

        // Simulate GPU being consistently faster.
        for _ in 0..10 {
            d.update_adaptive(DispatchTarget::Cpu, 500.0);
            d.update_adaptive(DispatchTarget::Gpu, 20.0);
        }

        let threshold = d.adaptive_threshold();
        assert!(
            threshold < DispatchCostModel::default().qubit_threshold,
            "expected lowered threshold, got {}",
            threshold
        );
    }

    // ---- DispatchStats correctness ----

    #[test]
    fn test_stats_correctness_after_circuit() {
        let config = UmaDispatchConfig::default().with_num_qubits(4);
        let mut d = UmaDispatcher::new(config);
        let mut state = QuantumState::new(4);

        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::rz(2, 1.0),
            Gate::cz(2, 3),
        ];

        let stats = d.dispatch_circuit(&gates, &mut state).unwrap();
        assert_eq!(stats.total_gates, 5);
        assert_eq!(
            stats.cpu_gates + stats.gpu_gates + stats.concurrent_gates,
            5
        );
        assert!(stats.estimated_gpu_only_time_ns > 0.0);
        assert!(stats.estimated_cpu_only_time_ns > 0.0);
        assert!(stats.speedup_vs_gpu_only > 0.0);
    }

    // ---- Default cost model values ----

    #[test]
    fn test_default_cost_model_values() {
        let cm = DispatchCostModel::default();
        assert!((cm.gpu_dispatch_overhead_us - 5.0).abs() < 1e-9);
        assert!((cm.gpu_throughput_factor - 10.0).abs() < 1e-9);
        assert!((cm.cpu_single_qubit_ns - 50.0).abs() < 1e-9);
        assert_eq!(cm.qubit_threshold, 4);
    }

    // ---- Builder pattern ----

    #[test]
    fn test_cost_model_builder() {
        let cm = DispatchCostModel::default()
            .with_gpu_dispatch_overhead_us(3.0)
            .with_gpu_throughput_factor(20.0)
            .with_cpu_single_qubit_ns(30.0)
            .with_qubit_threshold(6);
        assert!((cm.gpu_dispatch_overhead_us - 3.0).abs() < 1e-9);
        assert!((cm.gpu_throughput_factor - 20.0).abs() < 1e-9);
        assert!((cm.cpu_single_qubit_ns - 30.0).abs() < 1e-9);
        assert_eq!(cm.qubit_threshold, 6);
    }

    #[test]
    fn test_config_builder() {
        let config = UmaDispatchConfig::default()
            .with_num_qubits(20)
            .with_concurrent(true)
            .with_adaptive(true);
        assert_eq!(config.num_qubits, 20);
        assert!(config.enable_concurrent);
        assert!(config.enable_adaptive);
    }

    // ---- Reset stats ----

    #[test]
    fn test_reset_stats() {
        let config = UmaDispatchConfig::default().with_num_qubits(2);
        let mut d = UmaDispatcher::new(config);
        let mut state = QuantumState::new(2);

        d.dispatch_circuit(&[Gate::h(0)], &mut state).unwrap();
        assert_eq!(d.stats().total_gates, 1);

        d.reset_stats();
        assert_eq!(d.stats().total_gates, 0);
        assert_eq!(d.stats().cpu_gates, 0);
    }

    // ---- Batch analyze ----

    #[test]
    fn test_batch_analyze() {
        let d = default_dispatcher(8);
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::rz(2, 0.5)];
        let results = d.batch_analyze(&gates, 8);
        assert_eq!(results.len(), 3);
        assert!(results[0].0.is_single_qubit);
        assert!(!results[1].0.is_single_qubit);
        assert!(results[2].0.is_diagonal);
    }

    // ---- Circuit execution correctness ----

    #[test]
    fn test_dispatch_preserves_bell_state() {
        let config = UmaDispatchConfig::default().with_num_qubits(2);
        let mut d = UmaDispatcher::new(config);
        let mut state = QuantumState::new(2);

        // Bell state: H(0) then CNOT(0,1) -> (|00> + |11>) / sqrt(2)
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        d.dispatch_circuit(&gates, &mut state).unwrap();

        let probs = state.probabilities();
        assert!(
            (probs[0] - 0.5).abs() < 1e-10,
            "|00> probability: {}",
            probs[0]
        );
        assert!(probs[1].abs() < 1e-10, "|01> probability: {}", probs[1]);
        assert!(probs[2].abs() < 1e-10, "|10> probability: {}", probs[2]);
        assert!(
            (probs[3] - 0.5).abs() < 1e-10,
            "|11> probability: {}",
            probs[3]
        );
    }

    // ---- Display impls ----

    #[test]
    fn test_dispatch_target_display() {
        assert_eq!(format!("{}", DispatchTarget::Cpu), "CPU");
        assert_eq!(format!("{}", DispatchTarget::Gpu), "GPU");
        let c = DispatchTarget::Concurrent {
            cpu_qubits: (0, 3),
            gpu_qubits: (4, 7),
        };
        assert!(format!("{}", c).contains("Concurrent"));
    }

    #[test]
    fn test_uma_error_display() {
        let e = UmaError::QubitOutOfRange {
            qubit: 10,
            num_qubits: 4,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("10"));
        assert!(msg.contains("4"));
    }

    #[test]
    fn test_dispatch_stats_display() {
        let stats = DispatchStats {
            total_gates: 10,
            cpu_gates: 7,
            gpu_gates: 3,
            concurrent_gates: 0,
            total_cpu_time_ns: 350.0,
            total_gpu_time_ns: 150.0,
            estimated_gpu_only_time_ns: 1000.0,
            estimated_cpu_only_time_ns: 800.0,
            speedup_vs_gpu_only: 2.0,
            speedup_vs_cpu_only: 1.6,
        };
        let msg = format!("{}", stats);
        assert!(msg.contains("gates: 10"));
        assert!(msg.contains("CPU:7"));
        assert!(msg.contains("GPU:3"));
    }

    // ---- Layer empty circuit edge case ----

    #[test]
    fn test_layer_empty_circuit() {
        let layers = layer_circuit(&[], 4);
        assert!(layers.is_empty());
    }
}
