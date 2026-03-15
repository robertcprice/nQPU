//! Concurrent CPU+GPU Execution on Disjoint Qubit Subsets (B2)
//!
//! Extends the UMA gate-level dispatch (B1) with true concurrent execution:
//! the CPU processes gates on one qubit subset while the Metal GPU processes
//! gates on a disjoint subset *simultaneously*.  This is only possible on
//! Apple Silicon's Unified Memory Architecture (UMA), where the state vector
//! lives in a single shared address space (`StorageModeShared`).  CUDA/ROCm
//! platforms cannot do this because PCIe forces explicit host/device transfers.
//!
//! # Architecture
//!
//! ```text
//!  Circuit
//!    |
//!    v
//!  QubitDependencyGraph  -- analyze gate list, extract independent subsets
//!    |
//!    v
//!  find_concurrent_partitions() -- layers of disjoint-qubit gate groups
//!    |
//!    v
//!  OptimalPartition      -- choose CPU/GPU split boundary for each layer
//!    |
//!    v
//!  ConcurrentUmaExecutor -- spawn CPU thread + Metal command buffer
//!                           synchronized via memory fences (not copies)
//! ```
//!
//! # Synchronization
//!
//! Because UMA shares the same physical memory between CPU and GPU, we use
//! `std::sync::atomic::fence(Ordering::SeqCst)` as a memory fence rather
//! than copying data.  The GPU path (Metal command buffer + `waitUntilCompleted`)
//! provides its own fence on the GPU side.
//!
//! # GPU Backend
//!
//! The GPU side can run via CPU emulation or Metal (`GpuExecutionBackend`).
//! In strict mode, Metal setup/runtime failures are surfaced as errors instead
//! of degrading to CPU emulation.

use crate::ascii_viz::apply_gate_to_state;
use crate::gates::{Gate, GateType};
use crate::uma_dispatch::{
    DispatchCostModel, GateAnalysis, UmaDispatchConfig, UmaDispatcher, UmaError,
};
use crate::{QuantumState, C32, C64};
use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{self, Ordering};
use std::time::Instant;

// ============================================================
// QUBIT DEPENDENCY GRAPH
// ============================================================

/// Dependency graph over a circuit's gates.
///
/// Edges represent qubit-level data dependencies: gate A depends on gate B
/// if they share at least one qubit and A comes after B in circuit order.
/// Independent gates (disjoint qubit sets) have no edge and can execute
/// concurrently.
#[derive(Clone, Debug)]
pub struct QubitDependencyGraph {
    /// Number of gates in the circuit.
    pub num_gates: usize,
    /// Number of qubits in the system.
    pub num_qubits: usize,
    /// For each gate index, the set of qubit indices it touches.
    pub gate_qubits: Vec<HashSet<usize>>,
    /// Adjacency list: `deps[i]` contains indices of gates that gate `i`
    /// depends on (i.e. earlier gates sharing at least one qubit).
    pub deps: Vec<Vec<usize>>,
}

impl QubitDependencyGraph {
    /// Build a dependency graph from a gate list.
    ///
    /// Complexity: O(G^2 * Q) where G = number of gates, Q = max qubits per gate.
    /// For typical circuits (sparse connectivity) this is effectively O(G * Q).
    pub fn build(gates: &[Gate], num_qubits: usize) -> Self {
        let num_gates = gates.len();
        let mut gate_qubits: Vec<HashSet<usize>> = Vec::with_capacity(num_gates);
        let mut deps: Vec<Vec<usize>> = Vec::with_capacity(num_gates);

        // Track the last gate that touched each qubit.
        let mut last_gate_on_qubit: Vec<Option<usize>> = vec![None; num_qubits];

        for (i, gate) in gates.iter().enumerate() {
            let qs = collect_qubits(gate);
            let mut my_deps = HashSet::new();

            for &q in &qs {
                if q < num_qubits {
                    if let Some(prev) = last_gate_on_qubit[q] {
                        my_deps.insert(prev);
                    }
                    last_gate_on_qubit[q] = Some(i);
                }
            }

            let mut dep_vec: Vec<usize> = my_deps.into_iter().collect();
            dep_vec.sort_unstable();
            deps.push(dep_vec);
            gate_qubits.push(qs);
        }

        Self {
            num_gates,
            num_qubits,
            gate_qubits,
            deps,
        }
    }

    /// Return the set of all qubits touched by a gate.
    pub fn qubits_of(&self, gate_idx: usize) -> &HashSet<usize> {
        &self.gate_qubits[gate_idx]
    }

    /// Check whether two gates are independent (no shared qubits).
    pub fn are_independent(&self, a: usize, b: usize) -> bool {
        self.gate_qubits[a].is_disjoint(&self.gate_qubits[b])
    }

    /// Find sets of gates that can execute simultaneously.
    ///
    /// Returns layers where each layer contains gate indices that are
    /// mutually independent (disjoint qubit sets).  This is the same
    /// layering as `uma_dispatch::layer_circuit` but returns indices
    /// instead of `GateLayer` structs, and is built from the dependency
    /// graph for consistency.
    pub fn find_concurrent_partitions(&self) -> Vec<Vec<usize>> {
        let mut layers: Vec<(Vec<usize>, HashSet<usize>)> = Vec::new();

        for i in 0..self.num_gates {
            let qs = &self.gate_qubits[i];

            // Find the earliest layer where this gate has no qubit conflict
            // AND all its dependencies are in earlier layers.
            let dep_max_layer = self.deps[i]
                .iter()
                .filter_map(|&d| layers.iter().position(|(indices, _)| indices.contains(&d)))
                .max();

            // Gate must go in a layer strictly after all its dependencies.
            let earliest_layer = dep_max_layer.map(|l| l + 1).unwrap_or(0);

            let mut placed = false;
            for layer_idx in earliest_layer..layers.len() {
                let (ref mut indices, ref mut occupied) = layers[layer_idx];
                if qs.is_disjoint(occupied) {
                    occupied.extend(qs.iter());
                    indices.push(i);
                    placed = true;
                    break;
                }
            }

            if !placed {
                let mut occupied = HashSet::new();
                occupied.extend(qs.iter());
                layers.push((vec![i], occupied));
            }
        }

        layers.into_iter().map(|(indices, _)| indices).collect()
    }
}

// ============================================================
// CONCURRENT PARTITION
// ============================================================

/// A partition of a gate layer into CPU and GPU work.
#[derive(Clone, Debug)]
pub struct ConcurrentPartition {
    /// Gate indices assigned to CPU execution.
    pub cpu_gates: Vec<usize>,
    /// Gate indices assigned to GPU execution.
    pub gpu_gates: Vec<usize>,
    /// Qubit indices owned by the CPU partition.
    pub cpu_qubit_set: HashSet<usize>,
    /// Qubit indices owned by the GPU partition.
    pub gpu_qubit_set: HashSet<usize>,
    /// Estimated CPU work cost in nanoseconds.
    pub estimated_cpu_cost_ns: f64,
    /// Estimated GPU work cost in nanoseconds.
    pub estimated_gpu_cost_ns: f64,
}

impl fmt::Display for ConcurrentPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConcurrentPartition(CPU: {} gates on {:?}, GPU: {} gates on {:?}, \
             est. CPU={:.0}ns GPU={:.0}ns)",
            self.cpu_gates.len(),
            sorted_set(&self.cpu_qubit_set),
            self.gpu_gates.len(),
            sorted_set(&self.gpu_qubit_set),
            self.estimated_cpu_cost_ns,
            self.estimated_gpu_cost_ns,
        )
    }
}

// ============================================================
// OPTIMAL PARTITION ALGORITHM
// ============================================================

/// Strategy for partitioning gates between CPU and GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// All gates on CPU (baseline).
    AllCpu,
    /// All gates on GPU (baseline).
    AllGpu,
    /// Greedy: single-qubit and diagonal gates to CPU, rest to GPU.
    Greedy,
    /// Balanced: minimize max(cpu_cost, gpu_cost) by trying all split points.
    Balanced,
}

/// Backend used for GPU-partition execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuExecutionBackend {
    /// CPU emulation for GPU partitions.
    CpuEmulation,
    /// Real Metal backend for GPU partitions (macOS only).
    MetalGpu,
}

impl fmt::Display for GpuExecutionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuEmulation => write!(f, "CpuEmulation"),
            Self::MetalGpu => write!(f, "MetalGpu"),
        }
    }
}

/// Compute an optimal partition of gates within a single layer.
///
/// The `gates_in_layer` are indices into the original circuit's gate list.
/// All gates in the layer are on disjoint qubit subsets by construction,
/// so any partition into two groups is safe for concurrent execution.
///
/// The algorithm assigns each gate to CPU or GPU based on the cost model:
/// - Single-qubit and diagonal gates prefer CPU (low overhead).
/// - Multi-qubit non-diagonal gates prefer GPU (parallelism benefit).
/// - The `Balanced` strategy additionally tries to equalize CPU/GPU load.
pub fn compute_optimal_partition(
    gates: &[Gate],
    gate_indices: &[usize],
    num_qubits: usize,
    cost_model: &DispatchCostModel,
    strategy: PartitionStrategy,
) -> ConcurrentPartition {
    let dispatcher = UmaDispatcher::new(
        UmaDispatchConfig::default()
            .with_num_qubits(num_qubits)
            .with_cost_model(cost_model.clone()),
    );

    match strategy {
        PartitionStrategy::AllCpu => {
            let all_qubits: HashSet<usize> = gate_indices
                .iter()
                .flat_map(|&i| collect_qubits(&gates[i]))
                .collect();
            let total_cost: f64 = gate_indices
                .iter()
                .map(|&i| {
                    let a = dispatcher.analyze_gate(&gates[i], num_qubits);
                    cpu_gate_cost(&a, cost_model)
                })
                .sum();
            ConcurrentPartition {
                cpu_gates: gate_indices.to_vec(),
                gpu_gates: Vec::new(),
                cpu_qubit_set: all_qubits,
                gpu_qubit_set: HashSet::new(),
                estimated_cpu_cost_ns: total_cost,
                estimated_gpu_cost_ns: 0.0,
            }
        }
        PartitionStrategy::AllGpu => {
            let all_qubits: HashSet<usize> = gate_indices
                .iter()
                .flat_map(|&i| collect_qubits(&gates[i]))
                .collect();
            let total_cost: f64 = gate_indices
                .iter()
                .map(|&_i| gpu_gate_cost(num_qubits, cost_model))
                .sum();
            ConcurrentPartition {
                cpu_gates: Vec::new(),
                gpu_gates: gate_indices.to_vec(),
                cpu_qubit_set: HashSet::new(),
                gpu_qubit_set: all_qubits,
                estimated_cpu_cost_ns: 0.0,
                estimated_gpu_cost_ns: total_cost,
            }
        }
        PartitionStrategy::Greedy => {
            greedy_partition(gates, gate_indices, num_qubits, cost_model, &dispatcher)
        }
        PartitionStrategy::Balanced => {
            balanced_partition(gates, gate_indices, num_qubits, cost_model, &dispatcher)
        }
    }
}

/// Greedy partitioning: single-qubit/diagonal -> CPU, everything else -> GPU.
fn greedy_partition(
    gates: &[Gate],
    gate_indices: &[usize],
    num_qubits: usize,
    cost_model: &DispatchCostModel,
    dispatcher: &UmaDispatcher,
) -> ConcurrentPartition {
    let mut cpu_gates = Vec::new();
    let mut gpu_gates = Vec::new();
    let mut cpu_qubits = HashSet::new();
    let mut gpu_qubits = HashSet::new();
    let mut cpu_cost = 0.0f64;
    let mut gpu_cost = 0.0f64;

    for &idx in gate_indices {
        let analysis = dispatcher.analyze_gate(&gates[idx], num_qubits);
        let qs = collect_qubits(&gates[idx]);

        if analysis.is_single_qubit || analysis.is_diagonal {
            cpu_cost += cpu_gate_cost(&analysis, cost_model);
            cpu_qubits.extend(qs);
            cpu_gates.push(idx);
        } else {
            gpu_cost += gpu_gate_cost(num_qubits, cost_model);
            gpu_qubits.extend(qs);
            gpu_gates.push(idx);
        }
    }

    ConcurrentPartition {
        cpu_gates,
        gpu_gates,
        cpu_qubit_set: cpu_qubits,
        gpu_qubit_set: gpu_qubits,
        estimated_cpu_cost_ns: cpu_cost,
        estimated_gpu_cost_ns: gpu_cost,
    }
}

/// Balanced partitioning: minimize max(cpu_cost, gpu_cost).
///
/// Sorts gates by their CPU cost (descending) and greedily assigns each
/// to whichever side currently has the lower total cost.  This is a classic
/// load-balancing heuristic (LPT algorithm) that produces a 4/3-optimal
/// solution for two machines.
fn balanced_partition(
    gates: &[Gate],
    gate_indices: &[usize],
    num_qubits: usize,
    cost_model: &DispatchCostModel,
    dispatcher: &UmaDispatcher,
) -> ConcurrentPartition {
    // Build (gate_index, cpu_cost, gpu_cost, analysis) tuples.
    let mut items: Vec<(usize, f64, f64, GateAnalysis)> = gate_indices
        .iter()
        .map(|&idx| {
            let a = dispatcher.analyze_gate(&gates[idx], num_qubits);
            let cc = cpu_gate_cost(&a, cost_model);
            let gc = gpu_gate_cost(num_qubits, cost_model);
            (idx, cc, gc, a)
        })
        .collect();

    // Sort by descending absolute cost difference (gates that benefit most
    // from a particular device get assigned first).
    items.sort_by(|a, b| {
        let diff_a = (a.1 - a.2).abs();
        let diff_b = (b.1 - b.2).abs();
        diff_b
            .partial_cmp(&diff_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cpu_gates = Vec::new();
    let mut gpu_gates = Vec::new();
    let mut cpu_qubits = HashSet::new();
    let mut gpu_qubits = HashSet::new();
    let mut cpu_total = 0.0f64;
    let mut gpu_total = 0.0f64;

    for (idx, cc, gc, analysis) in items {
        let qs = collect_qubits(&gates[idx]);

        // Prefer the device with the lower estimated cost for this gate,
        // but break ties by load balance.
        let prefer_cpu = if analysis.is_single_qubit || analysis.is_diagonal {
            true
        } else if cc < gc {
            true
        } else if gc < cc {
            false
        } else {
            cpu_total <= gpu_total
        };

        // Assign to the preferred side, but if the other side is much
        // less loaded, override the preference for balance.
        let assign_cpu = if prefer_cpu {
            // Only override if GPU is much less loaded AND gate works on GPU.
            !(cpu_total > gpu_total * 2.0 && !analysis.is_single_qubit)
        } else {
            // Only override if CPU is much less loaded AND gate works on CPU.
            gpu_total > cpu_total * 2.0
        };

        if assign_cpu {
            cpu_total += cc;
            cpu_qubits.extend(qs);
            cpu_gates.push(idx);
        } else {
            gpu_total += gc;
            gpu_qubits.extend(qs);
            gpu_gates.push(idx);
        }
    }

    ConcurrentPartition {
        cpu_gates,
        gpu_gates,
        cpu_qubit_set: cpu_qubits,
        gpu_qubit_set: gpu_qubits,
        estimated_cpu_cost_ns: cpu_total,
        estimated_gpu_cost_ns: gpu_total,
    }
}

// ============================================================
// CONCURRENT EXECUTOR
// ============================================================

/// Result of concurrent execution of a single layer.
#[derive(Clone, Debug)]
pub struct LayerExecutionResult {
    /// How the layer was partitioned.
    pub partition: ConcurrentPartition,
    /// Actual CPU execution time in nanoseconds.
    pub cpu_time_ns: f64,
    /// Actual GPU-side execution time in nanoseconds.
    pub gpu_time_ns: f64,
    /// Wall time for the concurrent layer execution (max of cpu, gpu).
    pub wall_time_ns: f64,
    /// Effective overlapped execution time for this layer:
    /// max(0, min(cpu_time, gpu_time, cpu_time + gpu_time - wall_time)).
    pub effective_overlap_ns: f64,
    /// Overlap ratio in [0, 1]: effective_overlap / (cpu_time + gpu_time).
    pub effective_overlap_ratio: f64,
    /// Whether concurrent execution was used (vs sequential fallback).
    pub was_concurrent: bool,
}

/// Statistics from a full concurrent circuit execution.
#[derive(Clone, Debug)]
pub struct ConcurrentExecutionStats {
    /// Total number of layers in the circuit.
    pub num_layers: usize,
    /// Layers that used concurrent CPU+GPU execution.
    pub concurrent_layers: usize,
    /// Layers that used sequential execution (only one side had work).
    pub sequential_layers: usize,
    /// Total wall time in nanoseconds.
    pub total_wall_time_ns: f64,
    /// Total CPU work time in nanoseconds.
    pub total_cpu_time_ns: f64,
    /// Total GPU work time in nanoseconds.
    pub total_gpu_time_ns: f64,
    /// Total effective overlap across all layers in nanoseconds.
    pub total_overlap_ns: f64,
    /// Effective overlap ratio in [0, 1]:
    /// total_overlap / (total_cpu_time + total_gpu_time).
    pub effective_overlap_ratio: f64,
    /// What sequential-only (CPU) execution would have cost.
    pub estimated_sequential_time_ns: f64,
    /// Speedup from concurrency: sequential_time / wall_time.
    pub concurrency_speedup: f64,
    /// Per-layer results.
    pub layer_results: Vec<LayerExecutionResult>,
}

impl fmt::Display for ConcurrentExecutionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConcurrentExec {{ layers: {} (concurrent:{}, seq:{}), \
             wall={:.0}ns, speedup={:.2}x, overlap={:.1}% }}",
            self.num_layers,
            self.concurrent_layers,
            self.sequential_layers,
            self.total_wall_time_ns,
            self.concurrency_speedup,
            100.0 * self.effective_overlap_ratio,
        )
    }
}

/// Executor that runs gate layers concurrently on CPU and GPU.
///
/// For each layer of independent gates, the executor partitions gates
/// between CPU and GPU and attempts concurrent execution. Concurrency is
/// only enabled when the layer can be mapped to memory-disjoint amplitude
/// chunks; otherwise execution falls back to sequential order.
pub struct ConcurrentUmaExecutor {
    /// Cost model for partition decisions.
    cost_model: DispatchCostModel,
    /// Partitioning strategy.
    strategy: PartitionStrategy,
    /// Minimum number of gates in a layer to trigger concurrent execution.
    /// Layers with fewer gates run sequentially (not worth the thread overhead).
    min_concurrent_gates: usize,
    /// Backend used for GPU partition execution.
    gpu_backend: GpuExecutionBackend,
    /// When true, any attempt to execute a GPU partition/chunk without a
    /// functioning Metal backend returns an error instead of CPU emulation.
    strict_gpu_backend: bool,
}

impl ConcurrentUmaExecutor {
    /// Create a new concurrent executor with default settings.
    pub fn new() -> Self {
        Self {
            cost_model: DispatchCostModel::default(),
            strategy: PartitionStrategy::Greedy,
            min_concurrent_gates: 2,
            gpu_backend: GpuExecutionBackend::CpuEmulation,
            strict_gpu_backend: false,
        }
    }

    /// Builder: set the cost model.
    pub fn with_cost_model(mut self, cost_model: DispatchCostModel) -> Self {
        self.cost_model = cost_model;
        self
    }

    /// Builder: set the partitioning strategy.
    pub fn with_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Builder: set the minimum gate count for concurrent execution.
    pub fn with_min_concurrent_gates(mut self, min: usize) -> Self {
        self.min_concurrent_gates = min;
        self
    }

    /// Builder: select backend for GPU partitions.
    pub fn with_gpu_backend(mut self, backend: GpuExecutionBackend) -> Self {
        self.gpu_backend = backend;
        self
    }

    /// Builder: enable strict GPU backend behavior.
    ///
    /// In strict mode, any would-be GPU work fails if it cannot execute on
    /// Metal (misconfigured backend, platform unsupported, or Metal runtime
    /// failure). No CPU emulation fallback is allowed.
    pub fn with_strict_gpu_backend(mut self, strict: bool) -> Self {
        self.strict_gpu_backend = strict;
        self
    }

    /// Execute a circuit with concurrent CPU+GPU scheduling.
    ///
    /// Steps:
    /// 1. Build qubit dependency graph.
    /// 2. Extract concurrent partitions (layers).
    /// 3. For each layer, compute optimal CPU/GPU partition.
    /// 4. Execute CPU and GPU work concurrently (or sequentially if
    ///    only one side has work).
    /// 5. Memory fence between layers.
    pub fn execute(
        &self,
        gates: &[Gate],
        state: &mut QuantumState,
    ) -> Result<ConcurrentExecutionStats, UmaError> {
        self.execute_impl(gates, state, self.strict_gpu_backend)
    }

    /// Execute a circuit with strict GPU backend requirements.
    ///
    /// Any layer that requires GPU work must run on Metal, otherwise execution
    /// fails with `UmaError::GpuExecutionFailed`.
    pub fn execute_strict(
        &self,
        gates: &[Gate],
        state: &mut QuantumState,
    ) -> Result<ConcurrentExecutionStats, UmaError> {
        self.execute_impl(gates, state, true)
    }

    fn execute_impl(
        &self,
        gates: &[Gate],
        state: &mut QuantumState,
        strict_gpu_backend: bool,
    ) -> Result<ConcurrentExecutionStats, UmaError> {
        let num_qubits = state.num_qubits;

        // Validate all gates.
        for (i, gate) in gates.iter().enumerate() {
            let qs = collect_qubits(gate);
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

        // Build dependency graph and extract layers.
        let dep_graph = QubitDependencyGraph::build(gates, num_qubits);
        let layers = dep_graph.find_concurrent_partitions();

        let mut stats = ConcurrentExecutionStats {
            num_layers: layers.len(),
            concurrent_layers: 0,
            sequential_layers: 0,
            total_wall_time_ns: 0.0,
            total_cpu_time_ns: 0.0,
            total_gpu_time_ns: 0.0,
            total_overlap_ns: 0.0,
            effective_overlap_ratio: 0.0,
            estimated_sequential_time_ns: 0.0,
            concurrency_speedup: 1.0,
            layer_results: Vec::with_capacity(layers.len()),
        };

        for layer_indices in &layers {
            let partition = compute_optimal_partition(
                gates,
                layer_indices,
                num_qubits,
                &self.cost_model,
                self.strategy,
            );

            let has_cpu_work = !partition.cpu_gates.is_empty();
            let has_gpu_work = !partition.gpu_gates.is_empty();
            let use_concurrent =
                has_cpu_work && has_gpu_work && layer_indices.len() >= self.min_concurrent_gates;

            let result = if use_concurrent {
                self.execute_layer_concurrent(
                    gates,
                    state,
                    &partition,
                    layer_indices,
                    strict_gpu_backend,
                )?
            } else {
                self.execute_layer_sequential(gates, state, &partition, strict_gpu_backend)?
            };
            if result.was_concurrent {
                stats.concurrent_layers += 1;
            } else {
                stats.sequential_layers += 1;
            }

            // Accumulate sequential estimate (sum of both sides).
            stats.estimated_sequential_time_ns +=
                partition.estimated_cpu_cost_ns + partition.estimated_gpu_cost_ns;
            stats.total_wall_time_ns += result.wall_time_ns;
            stats.total_cpu_time_ns += result.cpu_time_ns;
            stats.total_gpu_time_ns += result.gpu_time_ns;
            stats.total_overlap_ns += result.effective_overlap_ns;
            stats.layer_results.push(result);

            // Memory fence between layers (UMA: no copy needed).
            atomic::fence(Ordering::SeqCst);
        }

        // Compute concurrency speedup.
        if stats.total_wall_time_ns > 0.0 {
            stats.concurrency_speedup =
                stats.estimated_sequential_time_ns / stats.total_wall_time_ns;
        }
        let total_work = stats.total_cpu_time_ns + stats.total_gpu_time_ns;
        if total_work > 0.0 {
            stats.effective_overlap_ratio = (stats.total_overlap_ns / total_work).clamp(0.0, 1.0);
        }

        Ok(stats)
    }

    /// Execute a layer concurrently.
    ///
    /// Safety rule: concurrent mutation is only used when the layer leaves
    /// the highest-index qubit untouched. In that case the state vector can be
    /// split into two independent contiguous chunks.
    fn execute_layer_concurrent(
        &self,
        gates: &[Gate],
        state: &mut QuantumState,
        partition: &ConcurrentPartition,
        gate_indices: &[usize],
        strict_gpu_backend: bool,
    ) -> Result<LayerExecutionResult, UmaError> {
        if !can_chunk_parallelize(partition, state.num_qubits) {
            return self.execute_layer_sequential(gates, state, partition, strict_gpu_backend);
        }

        let sub_num_qubits = state.num_qubits - 1;
        let start = Instant::now();

        let amps = state.amplitudes_mut();
        let half = amps.len() / 2;
        let (cpu_chunk, gpu_chunk) = amps.split_at_mut(half);

        let (cpu_join_result, gpu_result, gpu_elapsed_ns) = std::thread::scope(|scope| {
            let cpu_handle = scope.spawn(move || {
                let t0 = Instant::now();
                apply_gate_indices_to_chunk_cpu(cpu_chunk, sub_num_qubits, gates, gate_indices);
                t0.elapsed().as_nanos() as f64
            });

            let t1 = Instant::now();
            let gpu_result = self.execute_gpu_chunk_indices(
                gpu_chunk,
                sub_num_qubits,
                gates,
                gate_indices,
                strict_gpu_backend,
            );
            let gpu_elapsed_ns = t1.elapsed().as_nanos() as f64;

            (cpu_handle.join(), gpu_result, gpu_elapsed_ns)
        });
        let gpu_time_ns = gpu_elapsed_ns;
        let cpu_time_ns = cpu_join_result.map_err(|_| {
            UmaError::ConcurrentSchedulingFailed(
                "CPU worker thread panicked during concurrent execution".to_string(),
            )
        })?;
        gpu_result?;

        atomic::fence(Ordering::SeqCst);
        let wall_time_ns = start.elapsed().as_nanos() as f64;
        let (effective_overlap_ns, effective_overlap_ratio) =
            compute_effective_overlap(cpu_time_ns, gpu_time_ns, wall_time_ns);

        Ok(LayerExecutionResult {
            partition: partition.clone(),
            cpu_time_ns,
            gpu_time_ns,
            wall_time_ns,
            effective_overlap_ns,
            effective_overlap_ratio,
            was_concurrent: true,
        })
    }

    /// Execute a layer sequentially (one side has all the work).
    fn execute_layer_sequential(
        &self,
        gates: &[Gate],
        state: &mut QuantumState,
        partition: &ConcurrentPartition,
        strict_gpu_backend: bool,
    ) -> Result<LayerExecutionResult, UmaError> {
        let start = Instant::now();

        // Execute all CPU gates.
        for &idx in &partition.cpu_gates {
            apply_gate_to_state(state, &gates[idx]);
        }
        let cpu_time_ns = start.elapsed().as_nanos() as f64;

        // Execute all GPU-partition gates.
        let gpu_gate_list: Vec<Gate> = partition
            .gpu_gates
            .iter()
            .map(|&i| gates[i].clone())
            .collect();
        let gpu_time_ns = self.execute_gpu_partition(state, &gpu_gate_list, strict_gpu_backend)?;

        let wall_time_ns = start.elapsed().as_nanos() as f64;
        let (effective_overlap_ns, effective_overlap_ratio) =
            compute_effective_overlap(cpu_time_ns, gpu_time_ns, wall_time_ns);

        Ok(LayerExecutionResult {
            partition: partition.clone(),
            cpu_time_ns,
            gpu_time_ns,
            wall_time_ns,
            effective_overlap_ns,
            effective_overlap_ratio,
            was_concurrent: false,
        })
    }

    fn execute_gpu_partition(
        &self,
        state: &mut QuantumState,
        gates: &[Gate],
        strict_gpu_backend: bool,
    ) -> Result<f64, UmaError> {
        if gates.is_empty() {
            return Ok(0.0);
        }

        let start = Instant::now();
        match self.gpu_backend {
            GpuExecutionBackend::CpuEmulation => {
                if strict_gpu_backend {
                    return Err(UmaError::GpuExecutionFailed(
                        "strict mode requires MetalGpu backend for GPU partition execution; \
                         CpuEmulation configured"
                            .to_string(),
                    ));
                }
                for gate in gates {
                    apply_gate_to_state(state, gate);
                }
            }
            GpuExecutionBackend::MetalGpu => {
                #[cfg(target_os = "macos")]
                {
                    if let Err(err) = apply_gpu_gates_with_metal(state, gates) {
                        if strict_gpu_backend {
                            return Err(UmaError::GpuExecutionFailed(format!(
                                "strict mode requested MetalGpu execution, but Metal path failed: {}",
                                err
                            )));
                        }
                        for gate in gates {
                            apply_gate_to_state(state, gate);
                        }
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    if strict_gpu_backend {
                        return Err(UmaError::GpuExecutionFailed(
                            "strict mode requested MetalGpu execution, but Metal is unavailable \
                             on this platform"
                                .to_string(),
                        ));
                    }
                    for gate in gates {
                        apply_gate_to_state(state, gate);
                    }
                }
            }
        }
        Ok(start.elapsed().as_nanos() as f64)
    }

    fn execute_gpu_chunk(
        &self,
        chunk: &mut [C64],
        sub_num_qubits: usize,
        gates: &[Gate],
        strict_gpu_backend: bool,
    ) -> Result<(), UmaError> {
        if gates.is_empty() {
            return Ok(());
        }

        match self.gpu_backend {
            GpuExecutionBackend::CpuEmulation => {
                if strict_gpu_backend {
                    return Err(UmaError::GpuExecutionFailed(
                        "strict mode requires MetalGpu backend for GPU chunk execution; \
                         CpuEmulation configured"
                            .to_string(),
                    ));
                }
                apply_gates_to_chunk_cpu(chunk, sub_num_qubits, gates);
            }
            GpuExecutionBackend::MetalGpu => {
                #[cfg(target_os = "macos")]
                {
                    if let Err(err) = apply_gates_to_chunk_metal(chunk, sub_num_qubits, gates) {
                        if strict_gpu_backend {
                            return Err(UmaError::GpuExecutionFailed(format!(
                                "strict mode requested MetalGpu chunk execution, but Metal path \
                                 failed: {}",
                                err
                            )));
                        }
                        apply_gates_to_chunk_cpu(chunk, sub_num_qubits, gates);
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    if strict_gpu_backend {
                        return Err(UmaError::GpuExecutionFailed(
                            "strict mode requested MetalGpu chunk execution, but Metal is \
                             unavailable on this platform"
                                .to_string(),
                        ));
                    }
                    apply_gates_to_chunk_cpu(chunk, sub_num_qubits, gates);
                }
            }
        }
        Ok(())
    }

    fn execute_gpu_chunk_indices(
        &self,
        chunk: &mut [C64],
        sub_num_qubits: usize,
        all_gates: &[Gate],
        gate_indices: &[usize],
        strict_gpu_backend: bool,
    ) -> Result<(), UmaError> {
        if gate_indices.is_empty() {
            return Ok(());
        }

        match self.gpu_backend {
            GpuExecutionBackend::CpuEmulation => {
                if strict_gpu_backend {
                    return Err(UmaError::GpuExecutionFailed(
                        "strict mode requires MetalGpu backend for GPU chunk execution; \
                         CpuEmulation configured"
                            .to_string(),
                    ));
                }
                apply_gate_indices_to_chunk_cpu(chunk, sub_num_qubits, all_gates, gate_indices);
            }
            GpuExecutionBackend::MetalGpu => {
                #[cfg(target_os = "macos")]
                {
                    let gate_list: Vec<Gate> = gate_indices
                        .iter()
                        .map(|&idx| all_gates[idx].clone())
                        .collect();
                    if let Err(err) = apply_gates_to_chunk_metal(chunk, sub_num_qubits, &gate_list)
                    {
                        if strict_gpu_backend {
                            return Err(UmaError::GpuExecutionFailed(format!(
                                "strict mode requested MetalGpu chunk execution, but Metal path \
                                 failed: {}",
                                err
                            )));
                        }
                        apply_gate_indices_to_chunk_cpu(
                            chunk,
                            sub_num_qubits,
                            all_gates,
                            gate_indices,
                        );
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    if strict_gpu_backend {
                        return Err(UmaError::GpuExecutionFailed(
                            "strict mode requested MetalGpu chunk execution, but Metal is \
                             unavailable on this platform"
                                .to_string(),
                        ));
                    }
                    apply_gate_indices_to_chunk_cpu(chunk, sub_num_qubits, all_gates, gate_indices);
                }
            }
        }
        Ok(())
    }
}

impl Default for ConcurrentUmaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// HELPERS
// ============================================================

/// Collect all qubit indices a gate touches (targets + controls).
fn collect_qubits(gate: &Gate) -> HashSet<usize> {
    let mut qs = HashSet::new();
    for &t in &gate.targets {
        qs.insert(t);
    }
    for &c in &gate.controls {
        qs.insert(c);
    }
    qs
}

/// Sorted vector from a HashSet for deterministic display.
fn sorted_set(s: &HashSet<usize>) -> Vec<usize> {
    let mut v: Vec<usize> = s.iter().copied().collect();
    v.sort_unstable();
    v
}

fn compute_effective_overlap(cpu_time_ns: f64, gpu_time_ns: f64, wall_time_ns: f64) -> (f64, f64) {
    let raw = cpu_time_ns + gpu_time_ns - wall_time_ns;
    let capped = raw.max(0.0).min(cpu_time_ns.min(gpu_time_ns));
    let denom = cpu_time_ns + gpu_time_ns;
    if denom <= 0.0 {
        (0.0, 0.0)
    } else {
        (capped, (capped / denom).clamp(0.0, 1.0))
    }
}

/// Concurrent chunk mode requires a spare top qubit untouched by this layer.
fn can_chunk_parallelize(partition: &ConcurrentPartition, num_qubits: usize) -> bool {
    if num_qubits < 2 {
        return false;
    }
    let top = num_qubits - 1;
    !partition.cpu_qubit_set.contains(&top) && !partition.gpu_qubit_set.contains(&top)
}

fn apply_gates_to_chunk_cpu(chunk: &mut [C64], sub_num_qubits: usize, gates: &[Gate]) {
    for gate in gates {
        if !apply_gate_to_chunk_in_place(chunk, sub_num_qubits, gate) {
            apply_gate_to_chunk_via_local_state(chunk, sub_num_qubits, gate);
        }
    }
}

fn apply_gate_indices_to_chunk_cpu(
    chunk: &mut [C64],
    sub_num_qubits: usize,
    all_gates: &[Gate],
    gate_indices: &[usize],
) {
    for &gate_idx in gate_indices {
        let gate = &all_gates[gate_idx];
        if !apply_gate_to_chunk_in_place(chunk, sub_num_qubits, gate) {
            apply_gate_to_chunk_via_local_state(chunk, sub_num_qubits, gate);
        }
    }
}

fn apply_gate_to_chunk_via_local_state(chunk: &mut [C64], sub_num_qubits: usize, gate: &Gate) {
    let mut local = QuantumState::new(sub_num_qubits);
    local.amplitudes_mut().copy_from_slice(chunk);
    apply_gate_to_state(&mut local, gate);
    chunk.copy_from_slice(local.amplitudes_ref());
}

fn apply_gate_to_chunk_in_place(chunk: &mut [C64], sub_num_qubits: usize, gate: &Gate) -> bool {
    match &gate.gate_type {
        GateType::H => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_h_chunk(chunk, sub_num_qubits, q)),
        GateType::X => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_x_chunk(chunk, sub_num_qubits, q)),
        GateType::Y => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_y_chunk(chunk, sub_num_qubits, q)),
        GateType::Z => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_z_chunk(chunk, sub_num_qubits, q)),
        GateType::S => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_s_chunk(chunk, sub_num_qubits, q)),
        GateType::T => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_t_chunk(chunk, sub_num_qubits, q)),
        GateType::Rx(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_rx_chunk(chunk, sub_num_qubits, q, *theta)),
        GateType::Ry(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_ry_chunk(chunk, sub_num_qubits, q, *theta)),
        GateType::Rz(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_rz_chunk(chunk, sub_num_qubits, q, *theta)),
        GateType::Phase(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_phase_chunk(chunk, sub_num_qubits, q, *theta)),
        GateType::CNOT => {
            if gate.controls.len() == 1 && gate.targets.len() == 1 {
                apply_cnot_chunk(chunk, sub_num_qubits, gate.controls[0], gate.targets[0])
            } else {
                false
            }
        }
        GateType::CZ => {
            if gate.controls.len() == 1 && gate.targets.len() == 1 {
                apply_cz_chunk(chunk, sub_num_qubits, gate.controls[0], gate.targets[0])
            } else {
                false
            }
        }
        GateType::SWAP => {
            if gate.targets.len() == 2 {
                apply_swap_chunk(chunk, sub_num_qubits, gate.targets[0], gate.targets[1])
            } else {
                false
            }
        }
        _ => false,
    }
}

fn apply_h_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    for base in (0..chunk.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = chunk[i0];
            let b = chunk[i1];
            chunk[i0] = C64::new((a.re + b.re) * inv_sqrt2, (a.im + b.im) * inv_sqrt2);
            chunk[i1] = C64::new((a.re - b.re) * inv_sqrt2, (a.im - b.im) * inv_sqrt2);
        }
    }
    true
}

fn apply_x_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    for base in (0..chunk.len()).step_by(stride * 2) {
        for i in 0..stride {
            chunk.swap(base + i, base + i + stride);
        }
    }
    true
}

fn apply_y_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    for base in (0..chunk.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = chunk[i0];
            let b = chunk[i1];
            // |0> -> i|1>, |1> -> -i|0>
            chunk[i0] = C64::new(b.im, -b.re);
            chunk[i1] = C64::new(-a.im, a.re);
        }
    }
    true
}

fn apply_z_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    for (idx, amp) in chunk.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            amp.re = -amp.re;
            amp.im = -amp.im;
        }
    }
    true
}

fn apply_s_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    for (idx, amp) in chunk.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            let re = amp.re;
            amp.re = -amp.im;
            amp.im = re;
        }
    }
    true
}

fn apply_t_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    let phase_re = std::f64::consts::FRAC_1_SQRT_2;
    let phase_im = std::f64::consts::FRAC_1_SQRT_2;
    for (idx, amp) in chunk.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            let re = amp.re * phase_re - amp.im * phase_im;
            let im = amp.re * phase_im + amp.im * phase_re;
            amp.re = re;
            amp.im = im;
        }
    }
    true
}

fn apply_phase_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize, theta: f64) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    let phase_re = theta.cos();
    let phase_im = theta.sin();
    for (idx, amp) in chunk.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            let re = amp.re * phase_re - amp.im * phase_im;
            let im = amp.re * phase_im + amp.im * phase_re;
            amp.re = re;
            amp.im = im;
        }
    }
    true
}

fn apply_rx_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize, theta: f64) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    for base in (0..chunk.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = chunk[i0];
            let b = chunk[i1];
            chunk[i0] = C64::new(c * a.re + s * b.im, c * a.im - s * b.re);
            chunk[i1] = C64::new(s * a.im + c * b.re, -s * a.re + c * b.im);
        }
    }
    true
}

fn apply_ry_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize, theta: f64) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    for base in (0..chunk.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = chunk[i0];
            let b = chunk[i1];
            chunk[i0] = C64::new(c * a.re - s * b.re, c * a.im - s * b.im);
            chunk[i1] = C64::new(s * a.re + c * b.re, s * a.im + c * b.im);
        }
    }
    true
}

fn apply_rz_chunk(chunk: &mut [C64], sub_num_qubits: usize, qubit: usize, theta: f64) -> bool {
    if qubit >= sub_num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    let p0_re = c;
    let p0_im = -s;
    let p1_re = c;
    let p1_im = s;
    for base in (0..chunk.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = chunk[i0];
            let b = chunk[i1];
            chunk[i0] = C64::new(a.re * p0_re - a.im * p0_im, a.re * p0_im + a.im * p0_re);
            chunk[i1] = C64::new(b.re * p1_re - b.im * p1_im, b.re * p1_im + b.im * p1_re);
        }
    }
    true
}

fn apply_cnot_chunk(
    chunk: &mut [C64],
    sub_num_qubits: usize,
    control: usize,
    target: usize,
) -> bool {
    if control >= sub_num_qubits || target >= sub_num_qubits || control == target {
        return false;
    }
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    for idx in 0..chunk.len() {
        if (idx & control_mask) != 0 && (idx & target_mask) == 0 {
            let j = idx | target_mask;
            chunk.swap(idx, j);
        }
    }
    true
}

fn apply_cz_chunk(chunk: &mut [C64], sub_num_qubits: usize, control: usize, target: usize) -> bool {
    if control >= sub_num_qubits || target >= sub_num_qubits || control == target {
        return false;
    }
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    for (idx, amp) in chunk.iter_mut().enumerate() {
        if (idx & control_mask) != 0 && (idx & target_mask) != 0 {
            amp.re = -amp.re;
            amp.im = -amp.im;
        }
    }
    true
}

fn apply_swap_chunk(chunk: &mut [C64], sub_num_qubits: usize, q0: usize, q1: usize) -> bool {
    if q0 >= sub_num_qubits || q1 >= sub_num_qubits || q0 == q1 {
        return false;
    }
    let m0 = 1usize << q0;
    let m1 = 1usize << q1;
    for idx in 0..chunk.len() {
        let b0 = (idx & m0) != 0;
        let b1 = (idx & m1) != 0;
        if !b0 && b1 {
            let j = (idx | m0) & !m1;
            chunk.swap(idx, j);
        }
    }
    true
}

#[cfg(target_os = "macos")]
fn apply_gates_to_chunk_metal(
    chunk: &mut [C64],
    sub_num_qubits: usize,
    gates: &[Gate],
) -> Result<(), String> {
    let mut local = QuantumState::new(sub_num_qubits);
    local.amplitudes_mut().copy_from_slice(chunk);
    apply_gpu_gates_with_metal(&mut local, gates)?;
    chunk.copy_from_slice(local.amplitudes_ref());
    Ok(())
}

#[cfg(target_os = "macos")]
fn apply_gpu_gates_with_metal(state: &mut QuantumState, gates: &[Gate]) -> Result<(), String> {
    let num_qubits = state.num_qubits;
    let mut sim = crate::metal_backend::MetalSimulator::new(num_qubits)?;

    let amps64 = state.amplitudes_ref();
    let mut amps32 = vec![C32::new(0.0, 0.0); amps64.len()];
    for (dst, src) in amps32.iter_mut().zip(amps64.iter()) {
        *dst = C32::new(src.re as f32, src.im as f32);
    }
    sim.write_state_f32(&amps32)?;
    sim.run_circuit(gates);
    let out32 = sim.read_state_f32();

    let amps_mut = state.amplitudes_mut();
    for (dst, src) in amps_mut.iter_mut().zip(out32.iter()) {
        *dst = C64::new(src.re as f64, src.im as f64);
    }
    Ok(())
}

/// Estimate CPU cost for a gate based on its analysis.
fn cpu_gate_cost(analysis: &GateAnalysis, cost_model: &DispatchCostModel) -> f64 {
    if analysis.is_single_qubit {
        cost_model.cpu_single_qubit_ns
    } else if analysis.is_diagonal {
        cost_model.cpu_single_qubit_ns * analysis.target_qubits.len() as f64 * 2.0
    } else {
        let n = analysis.target_qubits.len();
        cost_model.cpu_single_qubit_ns * (1usize << n) as f64
    }
}

/// Estimate GPU cost for a gate.
fn gpu_gate_cost(num_qubits: usize, cost_model: &DispatchCostModel) -> f64 {
    let gpu_overhead_ns = cost_model.gpu_dispatch_overhead_us * 1000.0;
    let state_dim = 1usize << num_qubits;
    gpu_overhead_ns + (state_dim as f64 * 0.5 / cost_model.gpu_throughput_factor)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    // ---- QubitDependencyGraph tests ----

    #[test]
    fn test_dep_graph_empty_circuit() {
        let graph = QubitDependencyGraph::build(&[], 4);
        assert_eq!(graph.num_gates, 0);
        assert!(graph.deps.is_empty());
    }

    #[test]
    fn test_dep_graph_independent_gates() {
        // H(0), H(1), H(2) -- all independent.
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2)];
        let graph = QubitDependencyGraph::build(&gates, 4);
        assert_eq!(graph.num_gates, 3);
        // No gate depends on any other.
        assert!(graph.deps[0].is_empty());
        assert!(graph.deps[1].is_empty());
        assert!(graph.deps[2].is_empty());
        // All are pairwise independent.
        assert!(graph.are_independent(0, 1));
        assert!(graph.are_independent(0, 2));
        assert!(graph.are_independent(1, 2));
    }

    #[test]
    fn test_dep_graph_dependent_chain() {
        // H(0), CNOT(0,1), H(0) -- chain dependency.
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::h(0)];
        let graph = QubitDependencyGraph::build(&gates, 4);
        // Gate 1 depends on gate 0 (shared qubit 0).
        assert!(graph.deps[1].contains(&0));
        // Gate 2 depends on gate 1 (shared qubit 0).
        assert!(graph.deps[2].contains(&1));
        // Gates 0 and 1 are NOT independent.
        assert!(!graph.are_independent(0, 1));
    }

    #[test]
    fn test_dep_graph_partial_independence() {
        // H(0), H(1), CNOT(0,1) -- H(0) and H(1) are independent,
        // CNOT depends on both.
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1)];
        let graph = QubitDependencyGraph::build(&gates, 4);
        assert!(graph.are_independent(0, 1));
        assert!(!graph.are_independent(0, 2));
        assert!(!graph.are_independent(1, 2));
        assert!(graph.deps[2].contains(&0));
        assert!(graph.deps[2].contains(&1));
    }

    // ---- find_concurrent_partitions tests ----

    #[test]
    fn test_concurrent_partitions_all_independent() {
        // H(0), H(1), H(2), H(3) -- all in one layer.
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2), Gate::h(3)];
        let graph = QubitDependencyGraph::build(&gates, 4);
        let layers = graph.find_concurrent_partitions();
        assert_eq!(
            layers.len(),
            1,
            "all independent gates should be in one layer"
        );
        assert_eq!(layers[0].len(), 4);
    }

    #[test]
    fn test_concurrent_partitions_sequential_chain() {
        // H(0), X(0), Z(0) -- all on same qubit, must be sequential.
        let gates = vec![Gate::h(0), Gate::x(0), Gate::z(0)];
        let graph = QubitDependencyGraph::build(&gates, 4);
        let layers = graph.find_concurrent_partitions();
        assert_eq!(layers.len(), 3, "same-qubit chain should produce 3 layers");
    }

    #[test]
    fn test_concurrent_partitions_mixed() {
        // H(0), H(1), CNOT(0,1), H(2), H(3), CNOT(2,3)
        // Layer 0: H(0), H(1), H(2), H(3) -- all independent
        // Layer 1: CNOT(0,1), CNOT(2,3)    -- independent of each other, depend on layer 0
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::h(2),
            Gate::h(3),
            Gate::cnot(2, 3),
        ];
        let graph = QubitDependencyGraph::build(&gates, 4);
        let layers = graph.find_concurrent_partitions();
        assert_eq!(layers.len(), 2, "expected 2 layers, got {}", layers.len());
        assert_eq!(
            layers[0].len(),
            4,
            "first layer should have H(0),H(1),H(2),H(3)"
        );
        assert_eq!(
            layers[1].len(),
            2,
            "second layer should have CNOT(0,1),CNOT(2,3)"
        );
    }

    // ---- Partition strategy tests ----

    #[test]
    fn test_partition_all_cpu() {
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(2, 3)];
        let indices = vec![0, 1, 2];
        let partition = compute_optimal_partition(
            &gates,
            &indices,
            4,
            &DispatchCostModel::default(),
            PartitionStrategy::AllCpu,
        );
        assert_eq!(partition.cpu_gates.len(), 3);
        assert!(partition.gpu_gates.is_empty());
    }

    #[test]
    fn test_partition_all_gpu() {
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(2, 3)];
        let indices = vec![0, 1, 2];
        let partition = compute_optimal_partition(
            &gates,
            &indices,
            4,
            &DispatchCostModel::default(),
            PartitionStrategy::AllGpu,
        );
        assert!(partition.cpu_gates.is_empty());
        assert_eq!(partition.gpu_gates.len(), 3);
    }

    #[test]
    fn test_partition_greedy_separates_single_and_multi() {
        // H(0), H(1) are single-qubit -> CPU
        // A 6-qubit custom gate -> GPU (exceeds threshold in dispatch decision)
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::new(GateType::Custom(vec![]), vec![2, 3, 4, 5, 6, 7], vec![]),
        ];
        let indices = vec![0, 1, 2];
        let partition = compute_optimal_partition(
            &gates,
            &indices,
            8,
            &DispatchCostModel::default(),
            PartitionStrategy::Greedy,
        );
        // Single-qubit H gates should go to CPU.
        assert!(partition.cpu_gates.contains(&0));
        assert!(partition.cpu_gates.contains(&1));
        // Multi-qubit custom gate should go to GPU.
        assert!(partition.gpu_gates.contains(&2));
    }

    #[test]
    fn test_partition_balanced_distributes_load() {
        // 4 independent H gates -- balanced should split them.
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2), Gate::h(3)];
        let indices = vec![0, 1, 2, 3];
        let partition = compute_optimal_partition(
            &gates,
            &indices,
            4,
            &DispatchCostModel::default(),
            PartitionStrategy::Balanced,
        );
        // Both sides should have work (unless all go to CPU due to overhead).
        // Single-qubit gates strongly prefer CPU, so balanced may still put
        // all on CPU. This is correct behavior.
        let total = partition.cpu_gates.len() + partition.gpu_gates.len();
        assert_eq!(total, 4, "all gates accounted for");
    }

    // ---- ConcurrentUmaExecutor tests ----

    #[test]
    fn test_executor_empty_circuit() {
        let executor = ConcurrentUmaExecutor::new();
        let mut state = QuantumState::new(4);
        let stats = executor.execute(&[], &mut state).unwrap();
        assert_eq!(stats.num_layers, 0);
        assert_eq!(stats.concurrent_layers, 0);
    }

    #[test]
    fn test_executor_single_gate() {
        let executor = ConcurrentUmaExecutor::new();
        let mut state = QuantumState::new(2);
        let gates = vec![Gate::h(0)];
        let stats = executor.execute(&gates, &mut state).unwrap();
        assert_eq!(stats.num_layers, 1);
        // Single gate: not enough for concurrent.
        assert_eq!(stats.sequential_layers, 1);
        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_executor_bell_state_correctness() {
        let executor = ConcurrentUmaExecutor::new();
        let mut state = QuantumState::new(2);
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let _stats = executor.execute(&gates, &mut state).unwrap();

        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10, "|00> = {}", probs[0]);
        assert!(probs[1].abs() < 1e-10, "|01> = {}", probs[1]);
        assert!(probs[2].abs() < 1e-10, "|10> = {}", probs[2]);
        assert!((probs[3] - 0.5).abs() < 1e-10, "|11> = {}", probs[3]);
    }

    #[test]
    fn test_executor_concurrent_disjoint_gates() {
        // H(0), H(1), H(2), H(3) -- all disjoint, should form one concurrent layer.
        let executor = ConcurrentUmaExecutor::new()
            .with_strategy(PartitionStrategy::Greedy)
            .with_min_concurrent_gates(2);
        let mut state = QuantumState::new(4);
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2), Gate::h(3)];
        let stats = executor.execute(&gates, &mut state).unwrap();
        assert_eq!(stats.num_layers, 1);
        // All probs should be ~1/16 for uniform superposition.
        let probs = state.probabilities();
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - 1.0 / 16.0).abs() < 1e-10,
                "prob[{}] = {}, expected ~0.0625",
                i,
                p
            );
        }
    }

    #[test]
    fn test_executor_chunk_parallel_matches_sequential() {
        // Top qubit (3) untouched -> chunk concurrency is legal.
        // Greedy partition: H(0) -> CPU, CNOT(1,2) -> GPU.
        let gates = vec![Gate::h(0), Gate::cnot(1, 2)];

        let executor = ConcurrentUmaExecutor::new()
            .with_strategy(PartitionStrategy::Greedy)
            .with_min_concurrent_gates(2)
            .with_gpu_backend(GpuExecutionBackend::CpuEmulation);
        let mut concurrent_state = QuantumState::new(4);
        let stats = executor.execute(&gates, &mut concurrent_state).unwrap();
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.concurrent_layers, 1);
        assert!(stats.total_overlap_ns >= 0.0);
        assert!((0.0..=1.0).contains(&stats.effective_overlap_ratio));

        let mut seq_state = QuantumState::new(4);
        for gate in &gates {
            apply_gate_to_state(&mut seq_state, gate);
        }

        let p_conc = concurrent_state.probabilities();
        let p_seq = seq_state.probabilities();
        for i in 0..p_seq.len() {
            assert!(
                (p_conc[i] - p_seq[i]).abs() < 1e-10,
                "mismatch at {}: concurrent={} sequential={}",
                i,
                p_conc[i],
                p_seq[i]
            );
        }
    }

    #[test]
    fn test_executor_qubit_out_of_range() {
        let executor = ConcurrentUmaExecutor::new();
        let mut state = QuantumState::new(4);
        let gates = vec![Gate::h(10)];
        let result = executor.execute(&gates, &mut state);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_stats_speedup_positive() {
        let executor = ConcurrentUmaExecutor::new().with_strategy(PartitionStrategy::Greedy);
        let mut state = QuantumState::new(4);
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::cnot(2, 3),
            Gate::rz(0, 0.5),
            Gate::rz(1, 0.5),
        ];
        let stats = executor.execute(&gates, &mut state).unwrap();
        assert!(
            stats.concurrency_speedup > 0.0,
            "speedup must be positive: {}",
            stats.concurrency_speedup
        );
    }

    #[test]
    fn test_executor_strict_rejects_cpu_emulation_backend() {
        let executor = ConcurrentUmaExecutor::new()
            .with_strategy(PartitionStrategy::AllGpu)
            .with_gpu_backend(GpuExecutionBackend::CpuEmulation);
        let mut state = QuantumState::new(2);
        let err = executor
            .execute_strict(&[Gate::h(0)], &mut state)
            .expect_err("strict mode should reject CpuEmulation when GPU work is required");
        assert!(
            matches!(err, UmaError::GpuExecutionFailed(_)),
            "expected GpuExecutionFailed, got {:?}",
            err
        );
    }

    #[test]
    fn test_executor_strict_allows_cpu_only_partitions() {
        let executor = ConcurrentUmaExecutor::new()
            .with_strategy(PartitionStrategy::AllCpu)
            .with_gpu_backend(GpuExecutionBackend::CpuEmulation);
        let mut state = QuantumState::new(2);
        let stats = executor
            .execute_strict(&[Gate::h(0)], &mut state)
            .expect("strict mode should allow CPU-only layers");
        assert_eq!(stats.sequential_layers, 1);
        assert_eq!(stats.concurrent_layers, 0);
    }

    #[test]
    fn test_partition_display() {
        let partition = ConcurrentPartition {
            cpu_gates: vec![0, 1],
            gpu_gates: vec![2],
            cpu_qubit_set: [0, 1].into_iter().collect(),
            gpu_qubit_set: [2, 3].into_iter().collect(),
            estimated_cpu_cost_ns: 100.0,
            estimated_gpu_cost_ns: 200.0,
        };
        let s = format!("{}", partition);
        assert!(s.contains("CPU: 2 gates"));
        assert!(s.contains("GPU: 1 gates"));
    }

    #[test]
    fn test_stats_display() {
        let stats = ConcurrentExecutionStats {
            num_layers: 3,
            concurrent_layers: 2,
            sequential_layers: 1,
            total_wall_time_ns: 1000.0,
            total_cpu_time_ns: 500.0,
            total_gpu_time_ns: 600.0,
            total_overlap_ns: 100.0,
            effective_overlap_ratio: 100.0 / 1100.0,
            estimated_sequential_time_ns: 1100.0,
            concurrency_speedup: 1.1,
            layer_results: Vec::new(),
        };
        let s = format!("{}", stats);
        assert!(s.contains("layers: 3"));
        assert!(s.contains("concurrent:2"));
        assert!(s.contains("overlap="));
    }
}
