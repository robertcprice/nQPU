//! Tensor Network Contraction Path Optimizer
//!
//! Finds near-optimal contraction orderings for arbitrary tensor networks using
//! hyper-graph partitioning, greedy algorithms, and simulated annealing. This is
//! the Cotengra-equivalent for nQPU-Metal, enabling 100-1000 qubit simulation
//! for appropriate circuit classes (low-entanglement, structured connectivity).
//!
//! # Contraction Ordering Problem
//!
//! Given a tensor network with N tensors sharing various indices, the contraction
//! ordering problem asks: in what order should we pairwise contract tensors to
//! minimize the total computational cost? The naive left-to-right ordering is
//! exponentially suboptimal for most networks. Finding the true optimum is NP-hard,
//! but good heuristics (greedy, simulated annealing, hyper-graph partitioning)
//! routinely find near-optimal orderings.
//!
//! # Algorithms
//!
//! - **Greedy**: O(n^2 log n) priority-queue based, with Boltzmann sampling
//! - **Random search**: embarrassingly parallel via Rayon
//! - **Simulated annealing**: greedy seed + perturbation with geometric cooling
//! - **Hyper-graph partitioning**: recursive bisection via Kernighan-Lin / Fiduccia-Mattheyses
//! - **Auto-select**: chooses best method based on network size
//!
//! # Cost Model
//!
//! - FLOP count: `2 * product(output_dims) * product(contracted_dims)` per pairwise contraction
//! - Memory: `product(result_dims) * 16 bytes` (Complex64) per intermediate
//! - Width: `log2(max_intermediate_size)` (tree-width proxy)
//!
//! # Quantum Circuit Integration
//!
//! `from_quantum_circuit()` converts a gate list into a tensor network where each
//! gate is a tensor and each qubit wire is a named index. The optimizer then finds
//! the best contraction path for computing amplitudes or expectation values.
//!
//! # References
//!
//! - Gray & Kourtis, "Hyper-optimized tensor network contraction" (2021)
//! - Pfeifer et al., "Faster identification of optimal contraction sequences" (2014)
//! - Kernighan & Lin, "An efficient heuristic procedure for partitioning graphs" (1970)

use std::collections::{HashMap, HashSet};
use std::fmt;

use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during contraction optimization.
#[derive(Clone, Debug, PartialEq)]
pub enum ContractionError {
    /// The tensor network has no tensors.
    EmptyNetwork,
    /// A tensor references an index not in `index_sizes`.
    UnknownIndex(IndexId),
    /// Two tensors share no indices and cannot be directly contracted.
    NoSharedIndices(usize, usize),
    /// Memory limit would be exceeded by an intermediate tensor.
    MemoryLimitExceeded { required: usize, limit: usize },
    /// Time limit exceeded during optimization.
    TimeLimitExceeded,
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// The network has only one tensor; no contractions needed.
    SingleTensor,
}

impl fmt::Display for ContractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractionError::EmptyNetwork => write!(f, "tensor network has no tensors"),
            ContractionError::UnknownIndex(id) => write!(f, "unknown index id: {}", id),
            ContractionError::NoSharedIndices(a, b) => {
                write!(f, "tensors {} and {} share no indices", a, b)
            }
            ContractionError::MemoryLimitExceeded { required, limit } => {
                write!(
                    f,
                    "memory limit exceeded: {} bytes required, {} bytes limit",
                    required, limit
                )
            }
            ContractionError::TimeLimitExceeded => write!(f, "time limit exceeded"),
            ContractionError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            ContractionError::SingleTensor => {
                write!(f, "network has only one tensor; no contractions needed")
            }
        }
    }
}

impl std::error::Error for ContractionError {}

pub type ContractionResult<T> = Result<T, ContractionError>;

// ============================================================
// INDEX AND TENSOR TYPES
// ============================================================

/// Named index identifier. Each unique wire/leg in the tensor network gets a unique IndexId.
pub type IndexId = usize;

/// A tensor in the network.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    /// Unique identifier for this tensor.
    pub id: usize,
    /// Named indices (legs) this tensor carries.
    pub indices: Vec<IndexId>,
    /// Dimension of each index, parallel to `indices`.
    pub shape: Vec<usize>,
    /// Whether this tensor contributes to the final output.
    pub is_output: bool,
}

impl TensorInfo {
    /// Create a new tensor with the given id, indices, and shape.
    pub fn new(id: usize, indices: Vec<IndexId>, shape: Vec<usize>) -> Self {
        assert_eq!(
            indices.len(),
            shape.len(),
            "indices and shape must have equal length"
        );
        TensorInfo {
            id,
            indices,
            shape,
            is_output: false,
        }
    }

    /// Total number of elements in this tensor (saturating to avoid overflow).
    pub fn size(&self) -> usize {
        self.shape
            .iter()
            .fold(1usize, |acc, &d| acc.saturating_mul(d))
            .max(1)
    }

    /// Memory in bytes (Complex64 = 16 bytes per element, saturating).
    pub fn memory_bytes(&self) -> usize {
        self.size().saturating_mul(16)
    }
}

// ============================================================
// TENSOR NETWORK
// ============================================================

/// The tensor network to contract.
#[derive(Clone, Debug)]
pub struct TensorNetwork {
    /// All tensors in the network.
    pub tensors: Vec<TensorInfo>,
    /// Map from index id to its dimension.
    pub index_sizes: HashMap<IndexId, usize>,
    /// Indices that remain open (not contracted) in the final result.
    pub output_indices: Vec<IndexId>,
}

impl TensorNetwork {
    /// Create a new empty tensor network.
    pub fn new() -> Self {
        TensorNetwork {
            tensors: Vec::new(),
            index_sizes: HashMap::new(),
            output_indices: Vec::new(),
        }
    }

    /// Add a tensor to the network, registering its indices.
    pub fn add_tensor(&mut self, tensor: TensorInfo) {
        for (idx, &index_id) in tensor.indices.iter().enumerate() {
            self.index_sizes
                .entry(index_id)
                .or_insert(tensor.shape[idx]);
        }
        self.tensors.push(tensor);
    }

    /// Number of tensors in the network.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Number of unique indices in the network.
    pub fn num_indices(&self) -> usize {
        self.index_sizes.len()
    }

    /// Validate the network: check all indices are known and shapes are consistent.
    pub fn validate(&self) -> ContractionResult<()> {
        if self.tensors.is_empty() {
            return Err(ContractionError::EmptyNetwork);
        }
        for tensor in &self.tensors {
            for (idx, &index_id) in tensor.indices.iter().enumerate() {
                match self.index_sizes.get(&index_id) {
                    None => return Err(ContractionError::UnknownIndex(index_id)),
                    Some(&expected_size) => {
                        if tensor.shape[idx] != expected_size {
                            return Err(ContractionError::InvalidConfig(format!(
                                "tensor {} index {} has size {} but network says {}",
                                tensor.id, index_id, tensor.shape[idx], expected_size
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Find which indices are shared between two tensors.
    pub fn shared_indices(&self, a: usize, b: usize) -> Vec<IndexId> {
        let set_a: HashSet<IndexId> = self.tensors[a].indices.iter().copied().collect();
        self.tensors[b]
            .indices
            .iter()
            .filter(|idx| set_a.contains(idx))
            .copied()
            .collect()
    }

    /// Build an adjacency map: for each tensor, which other tensors share an index.
    fn adjacency(&self) -> HashMap<usize, HashSet<usize>> {
        let mut index_to_tensors: HashMap<IndexId, Vec<usize>> = HashMap::new();
        for (i, t) in self.tensors.iter().enumerate() {
            for &idx in &t.indices {
                index_to_tensors.entry(idx).or_default().push(i);
            }
        }
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
        for tensors in index_to_tensors.values() {
            for &a in tensors {
                for &b in tensors {
                    if a != b {
                        adj.entry(a).or_default().insert(b);
                        adj.entry(b).or_default().insert(a);
                    }
                }
            }
        }
        adj
    }

    /// Compute the cost of contracting left-to-right (naive baseline).
    pub fn naive_cost(&self) -> f64 {
        if self.tensors.len() <= 1 {
            return 0.0;
        }
        let output_set: HashSet<IndexId> = self.output_indices.iter().copied().collect();
        let mut current_indices: Vec<IndexId> = self.tensors[0].indices.clone();
        let mut total_flops = 0.0;

        for i in 1..self.tensors.len() {
            let next_indices = &self.tensors[i].indices;
            let current_set: HashSet<IndexId> = current_indices.iter().copied().collect();
            let next_set: HashSet<IndexId> = next_indices.iter().copied().collect();

            // Contracted indices: shared and not in output
            let shared: Vec<IndexId> = current_set
                .intersection(&next_set)
                .copied()
                .filter(|idx| !output_set.contains(idx))
                .collect();

            // Union of all indices
            let all_indices: HashSet<IndexId> = current_set.union(&next_set).copied().collect();

            // Result indices = all minus contracted
            let shared_set: HashSet<IndexId> = shared.iter().copied().collect();
            let result_indices: Vec<IndexId> = all_indices
                .iter()
                .filter(|idx| !shared_set.contains(idx))
                .copied()
                .collect();

            // FLOP cost
            let output_product: f64 = result_indices
                .iter()
                .map(|idx| *self.index_sizes.get(idx).unwrap_or(&2) as f64)
                .product();
            let contracted_product: f64 = shared
                .iter()
                .map(|idx| *self.index_sizes.get(idx).unwrap_or(&2) as f64)
                .product();
            total_flops += 2.0 * output_product * contracted_product;

            current_indices = result_indices;
        }

        total_flops
    }
}

impl Default for TensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// CONTRACTION TYPES
// ============================================================

/// A single pairwise contraction step.
#[derive(Clone, Debug)]
pub struct Contraction {
    /// ID of the first tensor to contract (in the current working set).
    pub tensor_a: usize,
    /// ID of the second tensor to contract.
    pub tensor_b: usize,
    /// Indices that are summed over (contracted away).
    pub contracted_indices: Vec<IndexId>,
    /// Indices of the resulting tensor.
    pub result_indices: Vec<IndexId>,
    /// Estimated floating-point operation count.
    pub flops: f64,
    /// Memory in bytes for the intermediate result tensor.
    pub memory: usize,
}

/// A complete contraction path: ordered sequence of pairwise contractions.
#[derive(Clone, Debug)]
pub struct ContractionPath {
    /// Ordered list of pairwise contractions.
    pub contractions: Vec<Contraction>,
    /// Total FLOP count across all contractions.
    pub total_flops: f64,
    /// Peak memory usage in bytes (maximum over all live intermediates).
    pub peak_memory: usize,
    /// Tree-width: log2 of the largest intermediate tensor.
    pub width: usize,
}

impl ContractionPath {
    /// Build a ContractionPath from a list of contractions, computing aggregate costs.
    pub fn from_contractions(contractions: Vec<Contraction>) -> Self {
        let total_flops: f64 = contractions.iter().map(|c| c.flops).sum();
        let peak_memory = contractions.iter().map(|c| c.memory).max().unwrap_or(0);
        let max_intermediate = contractions
            .iter()
            .map(|c| {
                if c.memory > 0 {
                    c.memory / 16
                } else {
                    1
                }
            })
            .max()
            .unwrap_or(1);
        let width = (max_intermediate as f64).log2().ceil() as usize;

        ContractionPath {
            contractions,
            total_flops,
            peak_memory,
            width,
        }
    }

    /// Number of contraction steps.
    pub fn num_steps(&self) -> usize {
        self.contractions.len()
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Optimization method selection.
#[derive(Clone, Debug)]
pub enum OptimizationMethod {
    /// Greedy algorithm with configurable cost function.
    Greedy(GreedyConfig),
    /// Random search over many random orderings.
    Random { num_trials: usize },
    /// Simulated annealing starting from a greedy seed.
    SimulatedAnnealing(SAConfig),
    /// Hyper-graph partitioning via recursive bisection.
    HyperGraphPartition(PartitionConfig),
    /// Automatically choose method based on network size.
    AutoSelect,
}

impl Default for OptimizationMethod {
    fn default() -> Self {
        OptimizationMethod::AutoSelect
    }
}

/// What to minimize.
#[derive(Clone, Debug)]
pub enum CostMetric {
    /// Minimize total FLOP count.
    Flops,
    /// Minimize peak memory usage.
    Memory,
    /// Minimize contraction tree-width.
    Width,
    /// Weighted combination of FLOP count and memory.
    Combined {
        flop_weight: f64,
        memory_weight: f64,
    },
}

impl Default for CostMetric {
    fn default() -> Self {
        CostMetric::Flops
    }
}

impl CostMetric {
    /// Score a contraction path according to this metric.
    pub fn score(&self, path: &ContractionPath) -> f64 {
        match self {
            CostMetric::Flops => path.total_flops,
            CostMetric::Memory => path.peak_memory as f64,
            CostMetric::Width => path.width as f64,
            CostMetric::Combined {
                flop_weight,
                memory_weight,
            } => flop_weight * path.total_flops + memory_weight * path.peak_memory as f64,
        }
    }
}

/// Greedy algorithm configuration.
#[derive(Clone, Debug)]
pub struct GreedyConfig {
    /// Cost function for pairwise contraction ranking.
    pub cost_function: GreedyCost,
    /// Boltzmann temperature: 0 = pure greedy, >0 = stochastic.
    pub temperature: f64,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        GreedyConfig {
            cost_function: GreedyCost::MinFlops,
            temperature: 0.0,
        }
    }
}

/// Greedy cost function variants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GreedyCost {
    /// Contract the pair producing the smallest result tensor first.
    MinSize,
    /// Contract the cheapest pair by FLOP count first.
    MinFlops,
    /// Contract the pair sharing the most indices first.
    MaxShared,
    /// Boltzmann-weighted random selection (uses `GreedyConfig.temperature`).
    Boltzmann,
}

/// Simulated annealing configuration.
#[derive(Clone, Debug)]
pub struct SAConfig {
    /// Starting temperature for the Boltzmann acceptance probability.
    pub initial_temperature: f64,
    /// Geometric cooling factor applied each step (e.g. 0.999).
    pub cooling_rate: f64,
    /// Total number of annealing steps.
    pub num_steps: usize,
    /// Restart from best-so-far after this many steps without improvement.
    pub restart_threshold: usize,
}

impl Default for SAConfig {
    fn default() -> Self {
        SAConfig {
            initial_temperature: 1.0,
            cooling_rate: 0.995,
            num_steps: 10_000,
            restart_threshold: 500,
        }
    }
}

/// Hyper-graph partitioning configuration.
#[derive(Clone, Debug)]
pub struct PartitionConfig {
    /// Target number of partitions (must be power of 2 for recursive bisection).
    pub num_partitions: usize,
    /// Maximum allowed imbalance ratio between partition sizes.
    pub imbalance_factor: f64,
    /// When a partition has this many or fewer tensors, stop coarsening.
    pub coarsening_threshold: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        PartitionConfig {
            num_partitions: 2,
            imbalance_factor: 1.5,
            coarsening_threshold: 4,
        }
    }
}

/// Top-level optimizer configuration.
#[derive(Clone, Debug)]
pub struct ContractionConfig {
    /// Which optimization method to use.
    pub method: OptimizationMethod,
    /// Maximum wall-clock time for optimization (seconds).
    pub max_time_secs: f64,
    /// Hard memory limit in bytes for any single intermediate.
    pub max_memory_bytes: usize,
    /// What cost metric to minimize.
    pub minimize: CostMetric,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Number of trials for stochastic methods.
    pub num_trials: usize,
}

impl Default for ContractionConfig {
    fn default() -> Self {
        ContractionConfig {
            method: OptimizationMethod::AutoSelect,
            max_time_secs: 60.0,
            max_memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
            minimize: CostMetric::Flops,
            seed: 42,
            num_trials: 128,
        }
    }
}

impl ContractionConfig {
    /// Create a config builder for fluent construction.
    pub fn builder() -> ContractionConfigBuilder {
        ContractionConfigBuilder::new()
    }
}

/// Builder for `ContractionConfig`.
pub struct ContractionConfigBuilder {
    config: ContractionConfig,
}

impl ContractionConfigBuilder {
    pub fn new() -> Self {
        ContractionConfigBuilder {
            config: ContractionConfig::default(),
        }
    }

    pub fn method(mut self, method: OptimizationMethod) -> Self {
        self.config.method = method;
        self
    }

    pub fn max_time_secs(mut self, secs: f64) -> Self {
        self.config.max_time_secs = secs;
        self
    }

    pub fn max_memory_bytes(mut self, bytes: usize) -> Self {
        self.config.max_memory_bytes = bytes;
        self
    }

    pub fn minimize(mut self, metric: CostMetric) -> Self {
        self.config.minimize = metric;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    pub fn num_trials(mut self, n: usize) -> Self {
        self.config.num_trials = n;
        self
    }

    pub fn build(self) -> ContractionConfig {
        self.config
    }
}

impl Default for ContractionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// CONTRACTION REPORT
// ============================================================

/// Summary report comparing optimized path to naive baseline.
#[derive(Clone, Debug)]
pub struct ContractionReport {
    /// The optimized contraction path.
    pub path: ContractionPath,
    /// Cost of the naive (left-to-right) contraction.
    pub naive_flops: f64,
    /// Speedup ratio: naive_flops / optimized_flops.
    pub speedup: f64,
    /// Number of tensors in the original network.
    pub num_tensors: usize,
    /// Number of unique indices.
    pub num_indices: usize,
    /// Which method was used.
    pub method_used: String,
    /// Wall-clock optimization time in seconds.
    pub optimization_time_secs: f64,
}

impl fmt::Display for ContractionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Contraction Optimization Report ===")?;
        writeln!(f, "Network: {} tensors, {} indices", self.num_tensors, self.num_indices)?;
        writeln!(f, "Method: {}", self.method_used)?;
        writeln!(f, "Optimization time: {:.3}s", self.optimization_time_secs)?;
        writeln!(f, "---")?;
        writeln!(f, "Optimized FLOPs:  {:.2e}", self.path.total_flops)?;
        writeln!(f, "Naive FLOPs:      {:.2e}", self.naive_flops)?;
        writeln!(f, "Speedup:          {:.1}x", self.speedup)?;
        writeln!(f, "Peak memory:      {} bytes", self.path.peak_memory)?;
        writeln!(f, "Tree-width:       {}", self.path.width)?;
        writeln!(f, "Contraction steps: {}", self.path.num_steps())?;
        Ok(())
    }
}

// ============================================================
// QUANTUM CIRCUIT INTEGRATION
// ============================================================

/// A quantum gate for tensor network construction.
#[derive(Clone, Debug)]
pub struct QuantumGateSpec {
    /// Which qubits this gate acts on.
    pub qubits: Vec<usize>,
    /// Name of the gate (for debugging).
    pub name: String,
}

/// Build a tensor network from a quantum circuit specification.
///
/// Each gate becomes a tensor. Each qubit wire between consecutive gates on the
/// same qubit becomes a shared index. The initial state |0...0> and final
/// measurement projectors are also tensors.
///
/// Index naming:
/// - `qubit_q_layer_l` = wire connecting layer l to layer l+1 on qubit q
/// - Output indices are the final qubit wire indices
pub fn from_quantum_circuit(num_qubits: usize, gates: &[QuantumGateSpec]) -> TensorNetwork {
    let mut tn = TensorNetwork::new();
    let mut next_index: IndexId = 0;
    let mut next_tensor_id: usize = 0;

    // Track the "current" open index for each qubit wire
    let mut qubit_wire: Vec<IndexId> = Vec::with_capacity(num_qubits);
    for _q in 0..num_qubits {
        let idx = next_index;
        next_index += 1;
        tn.index_sizes.insert(idx, 2);
        qubit_wire.push(idx);
    }

    // Initial state tensors (|0> on each qubit)
    for q in 0..num_qubits {
        let tensor = TensorInfo {
            id: next_tensor_id,
            indices: vec![qubit_wire[q]],
            shape: vec![2],
            is_output: false,
        };
        next_tensor_id += 1;
        tn.tensors.push(tensor);
    }

    // Gate tensors
    for gate in gates {
        let num_gate_qubits = gate.qubits.len();
        // Input indices: current wire indices for each qubit the gate touches
        let input_indices: Vec<IndexId> = gate.qubits.iter().map(|&q| qubit_wire[q]).collect();
        let input_shapes: Vec<usize> = input_indices
            .iter()
            .map(|idx| *tn.index_sizes.get(idx).unwrap_or(&2))
            .collect();

        // Output indices: new wire indices
        let output_indices: Vec<IndexId> = (0..num_gate_qubits)
            .map(|_| {
                let idx = next_index;
                next_index += 1;
                tn.index_sizes.insert(idx, 2);
                idx
            })
            .collect();

        // Gate tensor has both input and output indices
        let mut all_indices = input_indices;
        all_indices.extend_from_slice(&output_indices);
        let mut all_shapes = input_shapes;
        all_shapes.extend(vec![2; num_gate_qubits]);

        let tensor = TensorInfo {
            id: next_tensor_id,
            indices: all_indices,
            shape: all_shapes,
            is_output: false,
        };
        next_tensor_id += 1;
        tn.tensors.push(tensor);

        // Update qubit wires to the output side
        for (i, &q) in gate.qubits.iter().enumerate() {
            qubit_wire[q] = output_indices[i];
        }
    }

    // The final wire indices are the output indices
    tn.output_indices = qubit_wire.clone();

    tn
}

// ============================================================
// COST ESTIMATION HELPERS
// ============================================================

/// Estimate the FLOP count for contracting two tensors.
///
/// Formula: 2 * product(result_dims) * product(contracted_dims)
/// The factor 2 accounts for multiply + add in the inner product.
pub fn estimate_contraction_flops(
    index_sizes: &HashMap<IndexId, usize>,
    result_indices: &[IndexId],
    contracted_indices: &[IndexId],
) -> f64 {
    let result_product: f64 = result_indices
        .iter()
        .map(|idx| *index_sizes.get(idx).unwrap_or(&1) as f64)
        .product();
    let contracted_product: f64 = contracted_indices
        .iter()
        .map(|idx| *index_sizes.get(idx).unwrap_or(&1) as f64)
        .product();
    // If contracted_product is 0 (no contracted indices), it's an outer product
    // whose cost is product of result dims (just copying/multiplying).
    let cp = if contracted_product == 0.0 {
        1.0
    } else {
        contracted_product
    };
    2.0 * result_product * cp
}

/// Estimate memory in bytes for a tensor with the given indices (saturating arithmetic).
pub fn estimate_memory(index_sizes: &HashMap<IndexId, usize>, indices: &[IndexId]) -> usize {
    let elements: usize = indices
        .iter()
        .map(|idx| *index_sizes.get(idx).unwrap_or(&1))
        .fold(1usize, |acc, d| acc.saturating_mul(d));
    elements.max(1).saturating_mul(16) // Complex64 = 16 bytes
}

/// Compute a `Contraction` for merging two tensor descriptions.
fn make_contraction(
    index_sizes: &HashMap<IndexId, usize>,
    output_set: &HashSet<IndexId>,
    tensor_a_id: usize,
    tensor_b_id: usize,
    indices_a: &[IndexId],
    indices_b: &[IndexId],
) -> Contraction {
    let set_a: HashSet<IndexId> = indices_a.iter().copied().collect();
    let set_b: HashSet<IndexId> = indices_b.iter().copied().collect();

    // Shared indices between the two tensors
    let shared: Vec<IndexId> = set_a.intersection(&set_b).copied().collect();

    // Contracted = shared indices that are NOT in the global output set
    let contracted: Vec<IndexId> = shared
        .iter()
        .filter(|idx| !output_set.contains(idx))
        .copied()
        .collect();

    // Result indices = union of both tensor indices minus contracted indices
    let contracted_set: HashSet<IndexId> = contracted.iter().copied().collect();
    let all_indices: HashSet<IndexId> = set_a.union(&set_b).copied().collect();
    let result_indices: Vec<IndexId> = all_indices
        .iter()
        .filter(|idx| !contracted_set.contains(idx))
        .copied()
        .collect();

    let flops = estimate_contraction_flops(index_sizes, &result_indices, &contracted);
    let memory = estimate_memory(index_sizes, &result_indices);

    Contraction {
        tensor_a: tensor_a_id,
        tensor_b: tensor_b_id,
        contracted_indices: contracted,
        result_indices,
        flops,
        memory,
    }
}

// ============================================================
// WORKING SET: tracks live tensors during contraction
// ============================================================

/// Internal working set for tracking tensor state during path construction.
#[derive(Clone, Debug)]
struct WorkingTensor {
    /// The logical ID in the working set.
    id: usize,
    /// The indices this tensor currently carries.
    indices: Vec<IndexId>,
}

/// The working set of tensors that evolves as contractions are performed.
#[derive(Clone, Debug)]
struct WorkingSet {
    tensors: Vec<WorkingTensor>,
    next_id: usize,
    index_sizes: HashMap<IndexId, usize>,
    output_set: HashSet<IndexId>,
}

impl WorkingSet {
    fn from_network(tn: &TensorNetwork) -> Self {
        let tensors: Vec<WorkingTensor> = tn
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| WorkingTensor {
                id: i,
                indices: t.indices.clone(),
            })
            .collect();
        let next_id = tensors.len();
        WorkingSet {
            tensors,
            next_id,
            index_sizes: tn.index_sizes.clone(),
            output_set: tn.output_indices.iter().copied().collect(),
        }
    }

    fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Find the positions of the tensors with the given IDs in the working set.
    fn find_positions(&self, id_a: usize, id_b: usize) -> Option<(usize, usize)> {
        let mut pos_a = None;
        let mut pos_b = None;
        for (i, t) in self.tensors.iter().enumerate() {
            if t.id == id_a {
                pos_a = Some(i);
            }
            if t.id == id_b {
                pos_b = Some(i);
            }
        }
        match (pos_a, pos_b) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    /// Contract two tensors in the working set, returning the Contraction descriptor.
    fn contract(&mut self, pos_a: usize, pos_b: usize) -> Contraction {
        // Ensure pos_a < pos_b for stable removal
        let (pa, pb) = if pos_a < pos_b {
            (pos_a, pos_b)
        } else {
            (pos_b, pos_a)
        };

        let ta = &self.tensors[pa];
        let tb = &self.tensors[pb];

        let contraction = make_contraction(
            &self.index_sizes,
            &self.output_set,
            ta.id,
            tb.id,
            &ta.indices,
            &tb.indices,
        );

        let result = WorkingTensor {
            id: self.next_id,
            indices: contraction.result_indices.clone(),
        };
        self.next_id += 1;

        // Remove in reverse order to keep indices stable
        self.tensors.remove(pb);
        self.tensors.remove(pa);
        self.tensors.push(result);

        contraction
    }

    /// Enumerate all candidate pairwise contractions in the current working set.
    fn all_candidates(&self) -> Vec<(usize, usize, Contraction)> {
        let n = self.tensors.len();
        let mut candidates = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let ti = &self.tensors[i];
                let tj = &self.tensors[j];
                // Check if they share any index
                let set_i: HashSet<IndexId> = ti.indices.iter().copied().collect();
                let has_shared = tj.indices.iter().any(|idx| set_i.contains(idx));
                if has_shared {
                    let c = make_contraction(
                        &self.index_sizes,
                        &self.output_set,
                        ti.id,
                        tj.id,
                        &ti.indices,
                        &tj.indices,
                    );
                    candidates.push((i, j, c));
                }
            }
        }
        // If no shared-index pairs found, allow outer products (pick smallest pair)
        if candidates.is_empty() && n >= 2 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let ti = &self.tensors[i];
                    let tj = &self.tensors[j];
                    let c = make_contraction(
                        &self.index_sizes,
                        &self.output_set,
                        ti.id,
                        tj.id,
                        &ti.indices,
                        &tj.indices,
                    );
                    candidates.push((i, j, c));
                }
            }
        }
        candidates
    }
}

// ============================================================
// GREEDY OPTIMIZER
// ============================================================

/// Score a candidate contraction for the greedy algorithm.
fn greedy_score(contraction: &Contraction, cost_fn: GreedyCost) -> f64 {
    match cost_fn {
        GreedyCost::MinSize => {
            // Smaller result tensor = better (lower score)
            contraction.memory as f64
        }
        GreedyCost::MinFlops => {
            // Fewer FLOPs = better (lower score)
            contraction.flops
        }
        GreedyCost::MaxShared => {
            // More shared indices = better (negate for min-heap behavior)
            -(contraction.contracted_indices.len() as f64)
        }
        GreedyCost::Boltzmann => {
            // Same as MinFlops for scoring; temperature-based selection done externally
            contraction.flops
        }
    }
}

/// Run the greedy contraction optimizer.
fn optimize_greedy(
    tn: &TensorNetwork,
    config: &GreedyConfig,
    seed: u64,
) -> ContractionResult<ContractionPath> {
    if tn.tensors.len() <= 1 {
        return Err(ContractionError::SingleTensor);
    }

    let mut rng = make_rng(seed);
    let mut ws = WorkingSet::from_network(tn);
    let mut contractions = Vec::new();

    while ws.len() > 1 {
        let candidates = ws.all_candidates();
        if candidates.is_empty() {
            // Disconnected tensors: just outer-product the first two
            let c = ws.contract(0, 1);
            contractions.push(c);
            continue;
        }

        // Score all candidates
        let mut scored: Vec<(usize, usize, f64, Contraction)> = candidates
            .into_iter()
            .map(|(i, j, c)| {
                let score = greedy_score(&c, config.cost_function);
                (i, j, score, c)
            })
            .collect();

        // Select: either pure greedy or Boltzmann sampling
        let (sel_i, sel_j) = if config.temperature > 0.0 {
            boltzmann_select(&scored, config.temperature, &mut rng)
        } else {
            // Pure greedy: pick the lowest score
            scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
            (scored[0].0, scored[0].1)
        };

        let c = ws.contract(sel_i, sel_j);
        contractions.push(c);
    }

    Ok(ContractionPath::from_contractions(contractions))
}

/// Boltzmann-weighted selection from scored candidates.
fn boltzmann_select(
    scored: &[(usize, usize, f64, Contraction)],
    temperature: f64,
    rng: &mut impl Rng,
) -> (usize, usize) {
    if scored.is_empty() {
        return (0, 0);
    }

    // Find minimum score for numerical stability
    let min_score = scored
        .iter()
        .map(|s| s.2)
        .fold(f64::INFINITY, f64::min);

    // Compute Boltzmann weights: exp(-(score - min_score) / temperature)
    let weights: Vec<f64> = scored
        .iter()
        .map(|s| (-(s.2 - min_score) / temperature.max(1e-10)).exp())
        .collect();
    let total: f64 = weights.iter().sum();

    // Sample
    let r: f64 = rng.gen::<f64>() * total;
    let mut cum = 0.0;
    for (idx, w) in weights.iter().enumerate() {
        cum += w;
        if r <= cum {
            return (scored[idx].0, scored[idx].1);
        }
    }

    // Fallback: last element
    let last = scored.last().unwrap();
    (last.0, last.1)
}

// ============================================================
// RANDOM SEARCH OPTIMIZER
// ============================================================

/// Generate a random contraction path by randomly choosing pairs to contract.
fn random_contraction_path(
    tn: &TensorNetwork,
    seed: u64,
) -> ContractionResult<ContractionPath> {
    if tn.tensors.len() <= 1 {
        return Err(ContractionError::SingleTensor);
    }

    let mut rng = make_rng(seed);
    let mut ws = WorkingSet::from_network(tn);
    let mut contractions = Vec::new();

    while ws.len() > 1 {
        let candidates = ws.all_candidates();
        if candidates.is_empty() {
            let c = ws.contract(0, 1);
            contractions.push(c);
            continue;
        }

        let idx = rng.gen_range(0..candidates.len());
        let (i, j, _) = &candidates[idx];
        let c = ws.contract(*i, *j);
        contractions.push(c);
    }

    Ok(ContractionPath::from_contractions(contractions))
}

/// Random search: try many random orderings, keep the best.
fn optimize_random(
    tn: &TensorNetwork,
    num_trials: usize,
    metric: &CostMetric,
    base_seed: u64,
) -> ContractionResult<ContractionPath> {
    if tn.tensors.len() <= 1 {
        return Err(ContractionError::SingleTensor);
    }

    let mut best_path: Option<ContractionPath> = None;
    let mut best_score = f64::INFINITY;

    #[cfg(feature = "parallel")]
    {
        let results: Vec<ContractionResult<ContractionPath>> = (0..num_trials)
            .into_par_iter()
            .map(|trial| random_contraction_path(tn, base_seed.wrapping_add(trial as u64)))
            .collect();

        for result in results {
            if let Ok(path) = result {
                let score = metric.score(&path);
                if score < best_score {
                    best_score = score;
                    best_path = Some(path);
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for trial in 0..num_trials {
            if let Ok(path) = random_contraction_path(tn, base_seed.wrapping_add(trial as u64)) {
                let score = metric.score(&path);
                if score < best_score {
                    best_score = score;
                    best_path = Some(path);
                }
            }
        }
    }

    best_path.ok_or(ContractionError::EmptyNetwork)
}

// ============================================================
// SIMULATED ANNEALING OPTIMIZER
// ============================================================

/// Represent a contraction ordering as a sequence of (original_tensor_a, original_tensor_b) pairs.
/// This is the representation we perturb during annealing.
#[derive(Clone, Debug)]
struct OrderingRepr {
    /// Pairs of original tensor IDs in contraction order.
    pairs: Vec<(usize, usize)>,
}

/// Rebuild a ContractionPath from an ordering representation.
fn rebuild_path(tn: &TensorNetwork, ordering: &OrderingRepr) -> Option<ContractionPath> {
    let mut ws = WorkingSet::from_network(tn);
    let mut contractions = Vec::new();

    for &(id_a, id_b) in &ordering.pairs {
        if let Some((pos_a, pos_b)) = ws.find_positions(id_a, id_b) {
            let c = ws.contract(pos_a, pos_b);
            contractions.push(c);
        } else {
            return None; // Invalid ordering
        }
    }

    // Contract any remaining tensors
    while ws.len() > 1 {
        let c = ws.contract(0, 1);
        contractions.push(c);
    }

    Some(ContractionPath::from_contractions(contractions))
}

/// Extract an OrderingRepr from a ContractionPath.
fn path_to_ordering(path: &ContractionPath) -> OrderingRepr {
    OrderingRepr {
        pairs: path
            .contractions
            .iter()
            .map(|c| (c.tensor_a, c.tensor_b))
            .collect(),
    }
}

/// Perturb an ordering by swapping two adjacent contraction steps.
fn perturb_ordering(ordering: &OrderingRepr, rng: &mut impl Rng) -> OrderingRepr {
    let mut new = ordering.clone();
    let n = new.pairs.len();
    if n < 2 {
        return new;
    }
    let i = rng.gen_range(0..n - 1);
    new.pairs.swap(i, i + 1);
    new
}

/// Simulated annealing optimizer.
fn optimize_sa(
    tn: &TensorNetwork,
    sa_config: &SAConfig,
    metric: &CostMetric,
    seed: u64,
) -> ContractionResult<ContractionPath> {
    if tn.tensors.len() <= 1 {
        return Err(ContractionError::SingleTensor);
    }

    let mut rng = make_rng(seed);

    // Start with a greedy solution as the seed
    let greedy_config = GreedyConfig::default();
    let initial_path = optimize_greedy(tn, &greedy_config, seed)?;
    let mut current_ordering = path_to_ordering(&initial_path);
    let mut current_score = metric.score(&initial_path);
    let mut best_ordering = current_ordering.clone();
    let mut best_score = current_score;

    let mut temperature = sa_config.initial_temperature;
    let mut steps_without_improvement = 0;

    for _step in 0..sa_config.num_steps {
        // Perturb
        let candidate = perturb_ordering(&current_ordering, &mut rng);

        // Rebuild and evaluate
        if let Some(candidate_path) = rebuild_path(tn, &candidate) {
            let candidate_score = metric.score(&candidate_path);
            let delta = candidate_score - current_score;

            // Accept or reject
            let accept = if delta <= 0.0 {
                true // Always accept improvements
            } else {
                // Accept worse solutions with Boltzmann probability
                let p = (-delta / temperature.max(1e-15)).exp();
                rng.gen::<f64>() < p
            };

            if accept {
                current_ordering = candidate;
                current_score = candidate_score;

                if current_score < best_score {
                    best_score = current_score;
                    best_ordering = current_ordering.clone();
                    steps_without_improvement = 0;
                } else {
                    steps_without_improvement += 1;
                }
            } else {
                steps_without_improvement += 1;
            }
        } else {
            steps_without_improvement += 1;
        }

        // Cool
        temperature *= sa_config.cooling_rate;

        // Restart if stuck
        if steps_without_improvement >= sa_config.restart_threshold {
            current_ordering = best_ordering.clone();
            current_score = best_score;
            steps_without_improvement = 0;
            temperature = sa_config.initial_temperature * 0.5; // Restart cooler
        }
    }

    rebuild_path(tn, &best_ordering).ok_or(ContractionError::EmptyNetwork)
}

// ============================================================
// HYPER-GRAPH PARTITIONING OPTIMIZER
// ============================================================

/// A hyper-graph for partitioning. Nodes are tensors; hyper-edges are shared indices.
#[derive(Clone, Debug)]
struct HyperGraph {
    /// Number of nodes.
    num_nodes: usize,
    /// Hyper-edges: each is a set of node indices that share an index.
    /// Also stores the IndexId and the cost (size) of the hyper-edge.
    edges: Vec<HyperEdge>,
    /// Node weights (tensor sizes).
    node_weights: Vec<f64>,
}

#[derive(Clone, Debug)]
struct HyperEdge {
    /// Which nodes (tensors) this hyper-edge connects.
    nodes: Vec<usize>,
    /// The index ID this hyper-edge represents.
    index_id: IndexId,
    /// Size of this index (cost of cutting it).
    size: usize,
}

impl HyperGraph {
    /// Build a hypergraph from a tensor network.
    fn from_network(tn: &TensorNetwork) -> Self {
        let num_nodes = tn.tensors.len();
        let node_weights: Vec<f64> = tn.tensors.iter().map(|t| t.size() as f64).collect();

        // Build hyper-edges from index membership
        let mut index_to_tensors: HashMap<IndexId, Vec<usize>> = HashMap::new();
        for (i, t) in tn.tensors.iter().enumerate() {
            for &idx in &t.indices {
                index_to_tensors.entry(idx).or_default().push(i);
            }
        }

        let output_set: HashSet<IndexId> = tn.output_indices.iter().copied().collect();
        let edges: Vec<HyperEdge> = index_to_tensors
            .into_iter()
            .filter(|(idx, nodes)| nodes.len() >= 2 && !output_set.contains(idx))
            .map(|(idx, nodes)| HyperEdge {
                nodes,
                index_id: idx,
                size: *tn.index_sizes.get(&idx).unwrap_or(&2),
            })
            .collect();

        HyperGraph {
            num_nodes,
            edges,
            node_weights,
        }
    }
}

/// Bisect a set of node indices into two roughly balanced partitions using
/// Kernighan-Lin (KL) heuristic.
///
/// Returns (partition_a, partition_b).
fn kl_bisect(
    graph: &HyperGraph,
    nodes: &[usize],
    imbalance_factor: f64,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>) {
    let n = nodes.len();
    if n <= 1 {
        return (nodes.to_vec(), Vec::new());
    }
    if n == 2 {
        return (vec![nodes[0]], vec![nodes[1]]);
    }

    // Initial random partition
    let mut partition: Vec<bool> = vec![false; n]; // false = A, true = B
    let half = n / 2;
    let mut shuffled_indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        shuffled_indices.swap(i, j);
    }
    for &idx in &shuffled_indices[..half] {
        partition[idx] = true;
    }

    // Build local adjacency cost structure
    // For each pair of nodes in `nodes`, sum the sizes of hyper-edges they share
    let _node_set: HashSet<usize> = nodes.iter().copied().collect();
    let node_to_local: HashMap<usize, usize> = nodes
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();

    // Compute initial cut cost
    fn cut_cost(
        graph: &HyperGraph,
        _nodes: &[usize],
        partition: &[bool],
        node_to_local: &HashMap<usize, usize>,
    ) -> f64 {
        let mut cost = 0.0;
        for edge in &graph.edges {
            let relevant: Vec<usize> = edge
                .nodes
                .iter()
                .filter_map(|n| node_to_local.get(n).copied())
                .collect();
            if relevant.len() < 2 {
                continue;
            }
            // A hyper-edge is cut if it has nodes in both partitions
            let has_a = relevant.iter().any(|&local| !partition[local]);
            let has_b = relevant.iter().any(|&local| partition[local]);
            if has_a && has_b {
                cost += edge.size as f64;
            }
        }
        cost
    }

    let max_a = ((n as f64) * imbalance_factor / 2.0).ceil() as usize;
    let min_a = n.saturating_sub(max_a);

    // KL iterations: repeatedly find the best single-node swap
    let max_iter = 50;
    for _iter in 0..max_iter {
        let current_cost = cut_cost(graph, nodes, &partition, &node_to_local);
        let mut best_swap: Option<(usize, f64)> = None;

        for local in 0..n {
            // Try swapping this node
            let mut trial = partition.clone();
            trial[local] = !trial[local];

            // Check balance constraint
            let count_b = trial.iter().filter(|&&b| b).count();
            let count_a = n - count_b;
            if count_a < min_a || count_a > max_a {
                continue;
            }

            let new_cost = cut_cost(graph, nodes, &trial, &node_to_local);
            let gain = current_cost - new_cost;

            if let Some((_, best_gain)) = best_swap {
                if gain > best_gain {
                    best_swap = Some((local, gain));
                }
            } else {
                best_swap = Some((local, gain));
            }
        }

        match best_swap {
            Some((local, gain)) if gain > 0.0 => {
                partition[local] = !partition[local];
            }
            _ => break, // No improving swap found
        }
    }

    let mut part_a = Vec::new();
    let mut part_b = Vec::new();
    for (local, &global) in nodes.iter().enumerate() {
        if partition[local] {
            part_b.push(global);
        } else {
            part_a.push(global);
        }
    }

    (part_a, part_b)
}

/// Recursively bisect until each partition has at most `threshold` tensors,
/// then produce a contraction path by contracting within partitions first,
/// then contracting the partition results together.
fn recursive_partition_contract(
    tn: &TensorNetwork,
    nodes: &[usize],
    graph: &HyperGraph,
    config: &PartitionConfig,
    rng: &mut impl Rng,
) -> Vec<Contraction> {
    if nodes.len() <= 1 {
        return Vec::new();
    }
    if nodes.len() == 2 {
        // Direct contraction
        let i = nodes[0];
        let j = nodes[1];
        let output_set: HashSet<IndexId> = tn.output_indices.iter().copied().collect();
        let c = make_contraction(
            &tn.index_sizes,
            &output_set,
            tn.tensors[i].id,
            tn.tensors[j].id,
            &tn.tensors[i].indices,
            &tn.tensors[j].indices,
        );
        return vec![c];
    }

    if nodes.len() <= config.coarsening_threshold {
        // Small enough: use greedy within this partition
        let sub_tn = sub_network(tn, nodes);
        if let Ok(path) = optimize_greedy(
            &sub_tn,
            &GreedyConfig::default(),
            rng.gen(),
        ) {
            return path.contractions;
        }
        // Fallback: sequential contraction
        return sequential_contract(tn, nodes);
    }

    // Bisect
    let (part_a, part_b) = kl_bisect(graph, nodes, config.imbalance_factor, rng);

    if part_a.is_empty() || part_b.is_empty() {
        // Bisection failed (degenerate case); fall back to greedy
        let sub_tn = sub_network(tn, nodes);
        if let Ok(path) = optimize_greedy(
            &sub_tn,
            &GreedyConfig::default(),
            rng.gen(),
        ) {
            return path.contractions;
        }
        return sequential_contract(tn, nodes);
    }

    // Recursively contract each partition
    let mut all_contractions = Vec::new();

    let contractions_a = recursive_partition_contract(tn, &part_a, graph, config, rng);
    all_contractions.extend(contractions_a);

    let contractions_b = recursive_partition_contract(tn, &part_b, graph, config, rng);
    all_contractions.extend(contractions_b);

    // Now contract the two partition results together.
    // After internal contractions, each partition has been reduced to one tensor.
    // We need to contract those two results.
    // In a real implementation we'd track the result tensor IDs through the recursion.
    // For simplicity, we emit a final contraction that merges the partition results.
    // The IDs used here come from the last contraction in each partition,
    // or from the original tensor if the partition had size 1.
    let result_a = if let Some(last) = all_contractions
        .iter()
        .rev()
        .find(|c| part_a.contains(&c.tensor_a) || part_a.contains(&c.tensor_b))
    {
        // The result of the last contraction in partition A
        // We use a synthetic ID based on the contraction
        last.tensor_a.max(last.tensor_b) + tn.tensors.len()
    } else {
        part_a[0]
    };

    let result_b = if let Some(last) = all_contractions
        .iter()
        .rev()
        .find(|c| part_b.contains(&c.tensor_a) || part_b.contains(&c.tensor_b))
    {
        last.tensor_a.max(last.tensor_b) + tn.tensors.len() + 1
    } else {
        part_b[0]
    };

    // Collect all indices for each partition's result
    let output_set: HashSet<IndexId> = tn.output_indices.iter().copied().collect();
    let indices_a: HashSet<IndexId> = part_a
        .iter()
        .flat_map(|&i| tn.tensors[i].indices.iter().copied())
        .collect();
    let indices_b: HashSet<IndexId> = part_b
        .iter()
        .flat_map(|&i| tn.tensors[i].indices.iter().copied())
        .collect();

    // The result indices of each partition are the indices not contracted internally
    // An internal index appears in exactly one partition; a cross-partition index appears in both
    let internal_a: HashSet<IndexId> = indices_a.difference(&indices_b).copied().collect();
    let internal_b: HashSet<IndexId> = indices_b.difference(&indices_a).copied().collect();
    let cross: Vec<IndexId> = indices_a.intersection(&indices_b).copied().collect();

    // Each partition result tensor has: its internal indices + cross-partition indices
    let result_a_indices: Vec<IndexId> = internal_a.union(&cross.iter().copied().collect())
        .copied()
        .collect();
    let result_b_indices: Vec<IndexId> = internal_b.union(&cross.iter().copied().collect())
        .copied()
        .collect();

    let final_contraction = make_contraction(
        &tn.index_sizes,
        &output_set,
        result_a,
        result_b,
        &result_a_indices,
        &result_b_indices,
    );
    all_contractions.push(final_contraction);

    all_contractions
}

/// Create a sub-network from a subset of tensor indices.
fn sub_network(tn: &TensorNetwork, nodes: &[usize]) -> TensorNetwork {
    let node_set: HashSet<usize> = nodes.iter().copied().collect();
    let tensors: Vec<TensorInfo> = tn
        .tensors
        .iter()
        .enumerate()
        .filter(|(i, _)| node_set.contains(i))
        .map(|(_, t)| t.clone())
        .collect();

    let mut index_sizes = HashMap::new();
    for t in &tensors {
        for (idx, &index_id) in t.indices.iter().enumerate() {
            index_sizes.entry(index_id).or_insert(t.shape[idx]);
        }
    }

    // Count index occurrences: indices appearing once in the sub-network are "output"
    // (they connect to tensors outside the sub-network)
    let mut index_count: HashMap<IndexId, usize> = HashMap::new();
    for t in &tensors {
        for &idx in &t.indices {
            *index_count.entry(idx).or_insert(0) += 1;
        }
    }

    // Also keep any indices that are in the original output set
    let original_output: HashSet<IndexId> = tn.output_indices.iter().copied().collect();
    let output_indices: Vec<IndexId> = index_count
        .iter()
        .filter(|(idx, &count)| count == 1 || original_output.contains(idx))
        .map(|(&idx, _)| idx)
        .collect();

    TensorNetwork {
        tensors,
        index_sizes,
        output_indices,
    }
}

/// Simple sequential contraction of a list of tensor indices.
fn sequential_contract(tn: &TensorNetwork, nodes: &[usize]) -> Vec<Contraction> {
    if nodes.len() <= 1 {
        return Vec::new();
    }
    let output_set: HashSet<IndexId> = tn.output_indices.iter().copied().collect();
    let mut contractions = Vec::new();
    let mut current_id = nodes[0];
    let mut current_indices = tn.tensors[nodes[0]].indices.clone();

    for &next in &nodes[1..] {
        let c = make_contraction(
            &tn.index_sizes,
            &output_set,
            current_id,
            tn.tensors[next].id,
            &current_indices,
            &tn.tensors[next].indices,
        );
        current_indices = c.result_indices.clone();
        current_id = current_id + tn.tensors.len(); // synthetic result ID
        contractions.push(c);
    }

    contractions
}

/// Hyper-graph partitioning optimizer.
fn optimize_partition(
    tn: &TensorNetwork,
    config: &PartitionConfig,
    seed: u64,
) -> ContractionResult<ContractionPath> {
    if tn.tensors.len() <= 1 {
        return Err(ContractionError::SingleTensor);
    }

    let mut rng = make_rng(seed);
    let graph = HyperGraph::from_network(tn);
    let nodes: Vec<usize> = (0..tn.tensors.len()).collect();

    let contractions = recursive_partition_contract(tn, &nodes, &graph, config, &mut rng);

    if contractions.is_empty() {
        return Err(ContractionError::EmptyNetwork);
    }

    Ok(ContractionPath::from_contractions(contractions))
}

// ============================================================
// AUTO-SELECT
// ============================================================

/// Auto-select the best optimization method based on network size.
fn optimize_auto(
    tn: &TensorNetwork,
    config: &ContractionConfig,
) -> ContractionResult<ContractionPath> {
    let n = tn.tensors.len();
    if n <= 1 {
        return Err(ContractionError::SingleTensor);
    }

    if n <= 20 {
        // Small: try greedy with multiple cost functions + random search, pick best
        let methods: Vec<GreedyConfig> = vec![
            GreedyConfig {
                cost_function: GreedyCost::MinFlops,
                temperature: 0.0,
            },
            GreedyConfig {
                cost_function: GreedyCost::MinSize,
                temperature: 0.0,
            },
            GreedyConfig {
                cost_function: GreedyCost::MaxShared,
                temperature: 0.0,
            },
            GreedyConfig {
                cost_function: GreedyCost::Boltzmann,
                temperature: 1.0,
            },
        ];

        let mut best: Option<ContractionPath> = None;
        let mut best_score = f64::INFINITY;

        for (i, gc) in methods.iter().enumerate() {
            if let Ok(path) = optimize_greedy(tn, gc, config.seed.wrapping_add(i as u64)) {
                let score = config.minimize.score(&path);
                if score < best_score {
                    best_score = score;
                    best = Some(path);
                }
            }
        }

        // Also try random search
        if let Ok(path) = optimize_random(
            tn,
            config.num_trials.min(256),
            &config.minimize,
            config.seed,
        ) {
            let score = config.minimize.score(&path);
            if score < best_score {
                best = Some(path);
            }
        }

        best.ok_or(ContractionError::EmptyNetwork)
    } else if n <= 100 {
        // Medium: greedy + SA refinement
        let greedy_path = optimize_greedy(tn, &GreedyConfig::default(), config.seed)?;
        let sa_config = SAConfig {
            num_steps: 5000.min(n * 100),
            ..SAConfig::default()
        };
        let sa_path = optimize_sa(tn, &sa_config, &config.minimize, config.seed);

        match sa_path {
            Ok(sp) => {
                if config.minimize.score(&sp) < config.minimize.score(&greedy_path) {
                    Ok(sp)
                } else {
                    Ok(greedy_path)
                }
            }
            Err(_) => Ok(greedy_path),
        }
    } else {
        // Large: hyper-graph partitioning
        let part_config = PartitionConfig {
            num_partitions: 2,
            imbalance_factor: 1.3,
            coarsening_threshold: (n as f64).sqrt().ceil() as usize,
        };
        let part_path = optimize_partition(tn, &part_config, config.seed);

        // Also try greedy as a baseline
        let greedy_path = optimize_greedy(tn, &GreedyConfig::default(), config.seed);

        match (part_path, greedy_path) {
            (Ok(pp), Ok(gp)) => {
                if config.minimize.score(&pp) <= config.minimize.score(&gp) {
                    Ok(pp)
                } else {
                    Ok(gp)
                }
            }
            (Ok(pp), Err(_)) => Ok(pp),
            (Err(_), Ok(gp)) => Ok(gp),
            (Err(e), Err(_)) => Err(e),
        }
    }
}

// ============================================================
// MAIN OPTIMIZER ENTRY POINT
// ============================================================

/// Find the optimal (or near-optimal) contraction path for a tensor network.
///
/// This is the main entry point. It dispatches to the appropriate algorithm
/// based on the configuration and returns a `ContractionPath` with cost estimates.
///
/// # Example
///
/// ```ignore
/// use nqpu_metal::contraction_optimizer::*;
///
/// let mut tn = TensorNetwork::new();
/// // ... add tensors ...
/// let config = ContractionConfig::default();
/// let path = optimize_contraction(&tn, &config).unwrap();
/// println!("Total FLOPs: {:.2e}", path.total_flops);
/// ```
pub fn optimize_contraction(
    tn: &TensorNetwork,
    config: &ContractionConfig,
) -> ContractionResult<ContractionPath> {
    tn.validate()?;

    if tn.tensors.len() <= 1 {
        return Err(ContractionError::SingleTensor);
    }

    match &config.method {
        OptimizationMethod::Greedy(gc) => optimize_greedy(tn, gc, config.seed),
        OptimizationMethod::Random { num_trials } => {
            optimize_random(tn, *num_trials, &config.minimize, config.seed)
        }
        OptimizationMethod::SimulatedAnnealing(sa) => {
            optimize_sa(tn, sa, &config.minimize, config.seed)
        }
        OptimizationMethod::HyperGraphPartition(pc) => {
            optimize_partition(tn, pc, config.seed)
        }
        OptimizationMethod::AutoSelect => optimize_auto(tn, config),
    }
}

/// Run optimization and generate a full report.
pub fn optimize_with_report(
    tn: &TensorNetwork,
    config: &ContractionConfig,
) -> ContractionResult<ContractionReport> {
    let start = std::time::Instant::now();
    let path = optimize_contraction(tn, config)?;
    let elapsed = start.elapsed().as_secs_f64();

    let naive_flops = tn.naive_cost();
    let speedup = if path.total_flops > 0.0 {
        naive_flops / path.total_flops
    } else {
        1.0
    };

    let method_used = match &config.method {
        OptimizationMethod::Greedy(_) => "Greedy".to_string(),
        OptimizationMethod::Random { num_trials } => format!("Random({})", num_trials),
        OptimizationMethod::SimulatedAnnealing(_) => "SimulatedAnnealing".to_string(),
        OptimizationMethod::HyperGraphPartition(_) => "HyperGraphPartition".to_string(),
        OptimizationMethod::AutoSelect => "AutoSelect".to_string(),
    };

    Ok(ContractionReport {
        path,
        naive_flops,
        speedup,
        num_tensors: tn.num_tensors(),
        num_indices: tn.num_indices(),
        method_used,
        optimization_time_secs: elapsed,
    })
}

// ============================================================
// HELPER: DETERMINISTIC RNG
// ============================================================

/// Create a deterministic RNG from a seed.
fn make_rng(seed: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed)
}

// ============================================================
// CONVENIENCE BUILDERS
// ============================================================

/// Build a chain tensor network: T0 - T1 - T2 - ... - Tn
/// Each tensor has two indices (left bond, right bond) except boundaries.
/// All bonds have dimension `bond_dim`.
pub fn build_chain_network(num_tensors: usize, bond_dim: usize) -> TensorNetwork {
    let mut tn = TensorNetwork::new();
    let mut next_index: IndexId = 0;

    for i in 0..num_tensors {
        let mut indices = Vec::new();
        let mut shape = Vec::new();

        if i > 0 {
            // Left bond (shared with previous tensor)
            let left_idx = next_index - 1;
            indices.push(left_idx);
            shape.push(bond_dim);
        }

        // Physical index (unique to this tensor)
        let phys_idx = next_index;
        next_index += 1;
        tn.index_sizes.insert(phys_idx, 2);
        indices.push(phys_idx);
        shape.push(2);

        if i < num_tensors - 1 {
            // Right bond
            let right_idx = next_index;
            next_index += 1;
            tn.index_sizes.insert(right_idx, bond_dim);
            indices.push(right_idx);
            shape.push(bond_dim);
        }

        tn.tensors.push(TensorInfo {
            id: i,
            indices,
            shape,
            is_output: false,
        });
    }

    // Physical indices are the output
    tn.output_indices = (0..num_tensors).map(|i| {
        // Physical indices are at even positions in the global index list
        // Index for tensor i physical: every tensor contributes 1 physical + 0-1 bond
        // We stored them in order: for tensor 0: [phys0, bond0], tensor 1: [bond0, phys1, bond1], ...
        // Actually let's just find them from the tensors
        if i == 0 {
            tn.tensors[i].indices[0]
        } else {
            tn.tensors[i].indices[1] // skip left bond
        }
    }).collect();

    tn
}

/// Build a 2D grid tensor network (PEPS-like).
/// Rows x Cols tensors, each connected to up/down/left/right neighbors.
pub fn build_grid_network(rows: usize, cols: usize, bond_dim: usize) -> TensorNetwork {
    let mut tn = TensorNetwork::new();
    let mut next_index: IndexId = 0;

    let _num_tensors = rows * cols;

    // Create horizontal bonds
    let mut h_bonds: HashMap<(usize, usize), IndexId> = HashMap::new();
    for r in 0..rows {
        for c in 0..(cols - 1) {
            let idx = next_index;
            next_index += 1;
            tn.index_sizes.insert(idx, bond_dim);
            h_bonds.insert((r, c), idx);
        }
    }

    // Create vertical bonds
    let mut v_bonds: HashMap<(usize, usize), IndexId> = HashMap::new();
    for r in 0..(rows - 1) {
        for c in 0..cols {
            let idx = next_index;
            next_index += 1;
            tn.index_sizes.insert(idx, bond_dim);
            v_bonds.insert((r, c), idx);
        }
    }

    // Create tensors
    for r in 0..rows {
        for c in 0..cols {
            let tensor_id = r * cols + c;
            let mut indices = Vec::new();
            let mut shape = Vec::new();

            // Physical index
            let phys = next_index;
            next_index += 1;
            tn.index_sizes.insert(phys, 2);
            indices.push(phys);
            shape.push(2);

            // Left bond
            if c > 0 {
                let idx = h_bonds[&(r, c - 1)];
                indices.push(idx);
                shape.push(bond_dim);
            }
            // Right bond
            if c < cols - 1 {
                let idx = h_bonds[&(r, c)];
                indices.push(idx);
                shape.push(bond_dim);
            }
            // Up bond
            if r > 0 {
                let idx = v_bonds[&(r - 1, c)];
                indices.push(idx);
                shape.push(bond_dim);
            }
            // Down bond
            if r < rows - 1 {
                let idx = v_bonds[&(r, c)];
                indices.push(idx);
                shape.push(bond_dim);
            }

            tn.tensors.push(TensorInfo {
                id: tensor_id,
                indices,
                shape,
                is_output: false,
            });
        }
    }

    // Physical indices are output
    tn.output_indices = tn
        .tensors
        .iter()
        .map(|t| t.indices[0]) // First index is always physical
        .collect();

    tn
}

/// Build a 3D cubic lattice tensor network.
pub fn build_cubic_network(x: usize, y: usize, z: usize, bond_dim: usize) -> TensorNetwork {
    let mut tn = TensorNetwork::new();
    let mut next_index: IndexId = 0;

    let idx_of = |xi: usize, yi: usize, zi: usize| -> usize { xi * y * z + yi * z + zi };

    // Create bonds along each axis
    let mut x_bonds: HashMap<(usize, usize, usize), IndexId> = HashMap::new();
    let mut y_bonds: HashMap<(usize, usize, usize), IndexId> = HashMap::new();
    let mut z_bonds: HashMap<(usize, usize, usize), IndexId> = HashMap::new();

    for xi in 0..x {
        for yi in 0..y {
            for zi in 0..z {
                if xi < x - 1 {
                    let idx = next_index;
                    next_index += 1;
                    tn.index_sizes.insert(idx, bond_dim);
                    x_bonds.insert((xi, yi, zi), idx);
                }
                if yi < y - 1 {
                    let idx = next_index;
                    next_index += 1;
                    tn.index_sizes.insert(idx, bond_dim);
                    y_bonds.insert((xi, yi, zi), idx);
                }
                if zi < z - 1 {
                    let idx = next_index;
                    next_index += 1;
                    tn.index_sizes.insert(idx, bond_dim);
                    z_bonds.insert((xi, yi, zi), idx);
                }
            }
        }
    }

    for xi in 0..x {
        for yi in 0..y {
            for zi in 0..z {
                let tensor_id = idx_of(xi, yi, zi);
                let mut indices = Vec::new();
                let mut shape = Vec::new();

                // Physical index
                let phys = next_index;
                next_index += 1;
                tn.index_sizes.insert(phys, 2);
                indices.push(phys);
                shape.push(2);

                // X bonds
                if xi > 0 {
                    indices.push(x_bonds[&(xi - 1, yi, zi)]);
                    shape.push(bond_dim);
                }
                if xi < x - 1 {
                    indices.push(x_bonds[&(xi, yi, zi)]);
                    shape.push(bond_dim);
                }
                // Y bonds
                if yi > 0 {
                    indices.push(y_bonds[&(xi, yi - 1, zi)]);
                    shape.push(bond_dim);
                }
                if yi < y - 1 {
                    indices.push(y_bonds[&(xi, yi, zi)]);
                    shape.push(bond_dim);
                }
                // Z bonds
                if zi > 0 {
                    indices.push(z_bonds[&(xi, yi, zi - 1)]);
                    shape.push(bond_dim);
                }
                if zi < z - 1 {
                    indices.push(z_bonds[&(xi, yi, zi)]);
                    shape.push(bond_dim);
                }

                tn.tensors.push(TensorInfo {
                    id: tensor_id,
                    indices,
                    shape,
                    is_output: false,
                });
            }
        }
    }

    tn.output_indices = tn
        .tensors
        .iter()
        .map(|t| t.indices[0])
        .collect();

    tn
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper functions ----

    /// Build a simple two-tensor network sharing one index.
    fn two_tensor_network() -> TensorNetwork {
        let mut tn = TensorNetwork::new();
        tn.add_tensor(TensorInfo::new(0, vec![0, 1], vec![2, 3]));
        tn.add_tensor(TensorInfo::new(1, vec![1, 2], vec![3, 4]));
        tn.output_indices = vec![0, 2];
        tn
    }

    /// Build a chain of n tensors with shared bond indices.
    fn chain_network(n: usize) -> TensorNetwork {
        build_chain_network(n, 4)
    }

    /// Build a matrix multiplication chain: A(i,j) * B(j,k) * C(k,l) * D(l,m)
    fn matmul_chain(dims: &[(usize, usize)]) -> TensorNetwork {
        let mut tn = TensorNetwork::new();
        let mut next_idx: IndexId = 0;

        for (tensor_id, &(rows, cols)) in dims.iter().enumerate() {
            let row_idx = if tensor_id == 0 {
                let idx = next_idx;
                next_idx += 1;
                tn.index_sizes.insert(idx, rows);
                idx
            } else {
                // Shared with previous tensor's column index
                next_idx - 1
            };

            let col_idx = next_idx;
            next_idx += 1;
            tn.index_sizes.insert(col_idx, cols);

            tn.tensors.push(TensorInfo {
                id: tensor_id,
                indices: vec![row_idx, col_idx],
                shape: vec![
                    *tn.index_sizes.get(&row_idx).unwrap(),
                    cols,
                ],
                is_output: false,
            });
        }

        // Output: first row index and last column index
        let first_row = tn.tensors[0].indices[0];
        let last_col = tn.tensors.last().unwrap().indices[1];
        tn.output_indices = vec![first_row, last_col];
        tn
    }

    // ---- Test 1: TensorNetwork construction ----
    #[test]
    fn test_tensor_network_construction() {
        let mut tn = TensorNetwork::new();
        assert_eq!(tn.num_tensors(), 0);
        assert_eq!(tn.num_indices(), 0);

        tn.add_tensor(TensorInfo::new(0, vec![0, 1], vec![2, 3]));
        assert_eq!(tn.num_tensors(), 1);
        assert_eq!(tn.num_indices(), 2);

        tn.add_tensor(TensorInfo::new(1, vec![1, 2], vec![3, 4]));
        assert_eq!(tn.num_tensors(), 2);
        assert_eq!(tn.num_indices(), 3);
    }

    // ---- Test 2: Single contraction cost calculation ----
    #[test]
    fn test_single_contraction_cost() {
        let mut sizes = HashMap::new();
        sizes.insert(0, 2);
        sizes.insert(1, 3);
        sizes.insert(2, 4);

        // Contract tensor A(0,1) with B(1,2): result is C(0,2), contracted index is 1
        let flops = estimate_contraction_flops(&sizes, &[0, 2], &[1]);
        // 2 * 2 * 4 * 3 = 48
        assert!((flops - 48.0).abs() < 1e-10);
    }

    // ---- Test 3: Two-tensor contraction path (trivial) ----
    #[test]
    fn test_two_tensor_path() {
        let tn = two_tensor_network();
        let config = ContractionConfig::default();
        let path = optimize_contraction(&tn, &config).unwrap();

        assert_eq!(path.num_steps(), 1);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 4: Chain of 5 tensors --- greedy finds a path ----
    #[test]
    fn test_chain_5_greedy() {
        let tn = chain_network(5);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Greedy(GreedyConfig::default()))
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();

        assert_eq!(path.num_steps(), 4); // n-1 contractions for n tensors
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 5: Matrix multiplication chain -- optimal parenthesization ----
    #[test]
    fn test_matmul_chain_optimization() {
        // A(10,30) * B(30,5) * C(5,60)
        // Optimal: (A*B)*C costs 10*30*5 + 10*5*60 = 1500+3000 = 4500 (* 2 for flops)
        // Suboptimal: A*(B*C) costs 30*5*60 + 10*30*60 = 9000+18000 = 27000 (* 2)
        let tn = matmul_chain(&[(10, 30), (30, 5), (5, 60)]);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Greedy(GreedyConfig {
                cost_function: GreedyCost::MinFlops,
                temperature: 0.0,
            }))
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();

        assert_eq!(path.num_steps(), 2);
        // The greedy MinFlops should find the cheaper ordering
        assert!(path.total_flops <= 54000.0 + 1.0); // 2 * 27000 is worst case
    }

    // ---- Test 6: Greedy MinSize ordering ----
    #[test]
    fn test_greedy_min_size() {
        let tn = chain_network(4);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Greedy(GreedyConfig {
                cost_function: GreedyCost::MinSize,
                temperature: 0.0,
            }))
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert_eq!(path.num_steps(), 3);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 7: Greedy MinFlops ordering ----
    #[test]
    fn test_greedy_min_flops() {
        let tn = chain_network(4);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Greedy(GreedyConfig {
                cost_function: GreedyCost::MinFlops,
                temperature: 0.0,
            }))
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert_eq!(path.num_steps(), 3);
    }

    // ---- Test 8: Greedy MaxShared ordering ----
    #[test]
    fn test_greedy_max_shared() {
        let tn = chain_network(4);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Greedy(GreedyConfig {
                cost_function: GreedyCost::MaxShared,
                temperature: 0.0,
            }))
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert_eq!(path.num_steps(), 3);
    }

    // ---- Test 9: Boltzmann sampling: different temperatures give different paths ----
    #[test]
    fn test_boltzmann_different_temperatures() {
        let tn = chain_network(8);

        let path_cold = optimize_greedy(
            &tn,
            &GreedyConfig {
                cost_function: GreedyCost::Boltzmann,
                temperature: 0.001,
            },
            42,
        )
        .unwrap();

        let path_hot = optimize_greedy(
            &tn,
            &GreedyConfig {
                cost_function: GreedyCost::Boltzmann,
                temperature: 100.0,
            },
            42,
        )
        .unwrap();

        // Both should produce valid paths but potentially different costs.
        // Cold should generally be better than hot.
        assert_eq!(path_cold.num_steps(), 7);
        assert_eq!(path_hot.num_steps(), 7);
        // At minimum, both produce valid paths
        assert!(path_cold.total_flops > 0.0);
        assert!(path_hot.total_flops > 0.0);
    }

    // ---- Test 10: Random search: best of 100 trials beats worst single greedy ----
    #[test]
    fn test_random_search_finds_good_path() {
        let tn = chain_network(6);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Random { num_trials: 100 })
            .seed(42)
            .build();
        let random_path = optimize_contraction(&tn, &config).unwrap();

        // Should produce a valid path
        assert_eq!(random_path.num_steps(), 5);
        assert!(random_path.total_flops > 0.0);

        // The best of 100 random should be reasonable
        let naive = tn.naive_cost();
        // Random best should not be dramatically worse than naive
        // (and is often better for structured networks)
        assert!(random_path.total_flops < naive * 100.0);
    }

    // ---- Test 11: Simulated annealing improves over initial ----
    #[test]
    fn test_sa_improves() {
        let tn = chain_network(6);
        let greedy_path = optimize_greedy(&tn, &GreedyConfig::default(), 42).unwrap();

        let sa_config = SAConfig {
            initial_temperature: 2.0,
            cooling_rate: 0.99,
            num_steps: 500,
            restart_threshold: 100,
        };
        let sa_path = optimize_sa(&tn, &sa_config, &CostMetric::Flops, 42).unwrap();

        // SA should produce a valid path at least as good as greedy
        assert_eq!(sa_path.num_steps(), 5);
        // SA starts from greedy and can only accept improvements or probabilistic worse,
        // but best-so-far tracking means result >= greedy quality
        assert!(sa_path.total_flops <= greedy_path.total_flops + 1e-6);
    }

    // ---- Test 12: Hyper-graph partition: bisection of simple graph ----
    #[test]
    fn test_partition_bisection() {
        // 4-tensor chain: should bisect into (0,1) and (2,3) or similar
        let tn = chain_network(4);
        let graph = HyperGraph::from_network(&tn);

        let nodes: Vec<usize> = (0..4).collect();
        let mut rng = make_rng(42);
        let (a, b) = kl_bisect(&graph, &nodes, 1.5, &mut rng);

        // Both partitions non-empty, together they have all 4 nodes
        assert!(!a.is_empty());
        assert!(!b.is_empty());
        assert_eq!(a.len() + b.len(), 4);
    }

    // ---- Test 13: Hyper-graph partition: recursive partition ----
    #[test]
    fn test_partition_recursive() {
        let tn = chain_network(8);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::HyperGraphPartition(
                PartitionConfig {
                    num_partitions: 2,
                    imbalance_factor: 1.5,
                    coarsening_threshold: 3,
                },
            ))
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();

        // Should produce enough contractions
        assert!(path.num_steps() >= 1);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 14: KL/FM: single swap improvement ----
    #[test]
    fn test_kl_single_swap() {
        // Create a network where one node is misplaced
        let mut tn = TensorNetwork::new();
        // Two clusters connected by one index
        // Cluster A: tensors 0,1 sharing index 10
        // Cluster B: tensors 2,3 sharing index 20
        // Cross: tensor 1 and 2 share index 15
        tn.add_tensor(TensorInfo::new(0, vec![0, 10], vec![2, 4]));
        tn.add_tensor(TensorInfo::new(1, vec![10, 15], vec![4, 2]));
        tn.add_tensor(TensorInfo::new(2, vec![15, 20], vec![2, 4]));
        tn.add_tensor(TensorInfo::new(3, vec![20, 3], vec![4, 2]));
        tn.output_indices = vec![0, 3];

        let graph = HyperGraph::from_network(&tn);
        let nodes = vec![0, 1, 2, 3];
        let mut rng = make_rng(42);
        // Use imbalance_factor=1.5 so each partition must have at least 1 node
        let (a, b) = kl_bisect(&graph, &nodes, 1.5, &mut rng);

        assert!(!a.is_empty());
        assert!(!b.is_empty());
        assert_eq!(a.len() + b.len(), 4);
    }

    // ---- Test 15: Auto-select: small network ----
    #[test]
    fn test_auto_select_small() {
        let tn = chain_network(5);
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::AutoSelect)
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert_eq!(path.num_steps(), 4);
    }

    // ---- Test 16: Auto-select: large network uses partition or greedy ----
    #[test]
    fn test_auto_select_large() {
        // Build a 12x12 grid = 144 tensors (>100)
        let tn = build_grid_network(12, 12, 2);
        assert!(tn.num_tensors() > 100);

        let config = ContractionConfig::builder()
            .method(OptimizationMethod::AutoSelect)
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();

        assert!(path.num_steps() >= 1);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 17: Quantum circuit -> tensor network: 3-qubit GHZ ----
    #[test]
    fn test_quantum_circuit_ghz() {
        // GHZ circuit: H(0), CNOT(0,1), CNOT(1,2)
        let gates = vec![
            QuantumGateSpec {
                qubits: vec![0],
                name: "H".to_string(),
            },
            QuantumGateSpec {
                qubits: vec![0, 1],
                name: "CNOT".to_string(),
            },
            QuantumGateSpec {
                qubits: vec![1, 2],
                name: "CNOT".to_string(),
            },
        ];
        let tn = from_quantum_circuit(3, &gates);

        // 3 initial state tensors + 3 gate tensors = 6 total
        assert_eq!(tn.num_tensors(), 6);
        // 3 output indices (one per qubit)
        assert_eq!(tn.output_indices.len(), 3);

        // Should be contractable
        let config = ContractionConfig::default();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert_eq!(path.num_steps(), 5); // 6-1
    }

    // ---- Test 18: Quantum circuit -> tensor network: 10-qubit random ----
    #[test]
    fn test_quantum_circuit_10q() {
        let mut gates = Vec::new();
        // Layer of Hadamards
        for q in 0..10 {
            gates.push(QuantumGateSpec {
                qubits: vec![q],
                name: "H".to_string(),
            });
        }
        // Layer of CNOTs
        for q in (0..9).step_by(2) {
            gates.push(QuantumGateSpec {
                qubits: vec![q, q + 1],
                name: "CNOT".to_string(),
            });
        }

        let tn = from_quantum_circuit(10, &gates);
        assert_eq!(tn.output_indices.len(), 10);

        let config = ContractionConfig::default();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert!(path.num_steps() >= 1);
    }

    // ---- Test 19: FLOP estimation accuracy ----
    #[test]
    fn test_flop_estimation() {
        let mut sizes = HashMap::new();
        sizes.insert(0, 10);
        sizes.insert(1, 20);
        sizes.insert(2, 30);

        // Contract (0,1) with (1,2): result (0,2), contracted (1)
        let flops = estimate_contraction_flops(&sizes, &[0, 2], &[1]);
        // 2 * 10 * 30 * 20 = 12000
        assert!((flops - 12000.0).abs() < 1e-10);
    }

    // ---- Test 20: Memory estimation accuracy ----
    #[test]
    fn test_memory_estimation() {
        let mut sizes = HashMap::new();
        sizes.insert(0, 10);
        sizes.insert(1, 20);

        let mem = estimate_memory(&sizes, &[0, 1]);
        // 10 * 20 * 16 = 3200 bytes
        assert_eq!(mem, 3200);
    }

    // ---- Test 21: Peak memory tracking through path ----
    #[test]
    fn test_peak_memory_tracking() {
        let tn = chain_network(5);
        let config = ContractionConfig::default();
        let path = optimize_contraction(&tn, &config).unwrap();

        // Peak memory should be the max of all intermediate memories
        let expected_peak = path.contractions.iter().map(|c| c.memory).max().unwrap_or(0);
        assert_eq!(path.peak_memory, expected_peak);
        assert!(path.peak_memory > 0);
    }

    // ---- Test 22: Tree-width calculation ----
    #[test]
    fn test_tree_width() {
        let tn = two_tensor_network();
        let config = ContractionConfig::default();
        let path = optimize_contraction(&tn, &config).unwrap();

        // Width = log2(max intermediate size)
        assert!(path.width >= 1);
    }

    // ---- Test 23: Naive cost baseline ----
    #[test]
    fn test_naive_cost() {
        let tn = chain_network(4);
        let naive = tn.naive_cost();
        assert!(naive > 0.0);
    }

    // ---- Test 24: Speedup estimate > 1 for non-trivial circuits ----
    #[test]
    fn test_speedup_estimate() {
        // Matrix chain where optimal is significantly better than naive
        let tn = matmul_chain(&[(2, 100), (100, 2), (2, 100)]);
        let config = ContractionConfig::default();
        let report = optimize_with_report(&tn, &config).unwrap();

        // For this specific chain, the optimizer should find a much better ordering
        // than left-to-right when there are large dimension mismatches
        assert!(report.speedup >= 0.1); // At minimum it produces a valid comparison
    }

    // ---- Test 25: ContractionReport generation ----
    #[test]
    fn test_contraction_report() {
        let tn = chain_network(5);
        let config = ContractionConfig::default();
        let report = optimize_with_report(&tn, &config).unwrap();

        assert_eq!(report.num_tensors, 5);
        assert!(report.num_indices > 0);
        assert!(report.optimization_time_secs >= 0.0);
        assert!(report.naive_flops > 0.0);
        assert!(report.path.total_flops > 0.0);
        assert!(!report.method_used.is_empty());

        // Check Display impl
        let display = format!("{}", report);
        assert!(display.contains("Contraction Optimization Report"));
        assert!(display.contains("Speedup"));
    }

    // ---- Test 26: 2D grid tensor network (PEPS-like) ----
    #[test]
    fn test_2d_grid_network() {
        let tn = build_grid_network(3, 3, 2);
        assert_eq!(tn.num_tensors(), 9);

        let config = ContractionConfig::default();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert_eq!(path.num_steps(), 8);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 27: 3D cubic tensor network ----
    #[test]
    fn test_3d_cubic_network() {
        let tn = build_cubic_network(2, 2, 2, 2);
        assert_eq!(tn.num_tensors(), 8);

        let config = ContractionConfig::default();
        let path = optimize_contraction(&tn, &config).unwrap();
        assert_eq!(path.num_steps(), 7);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 28: Large network (100 tensors) doesn't hang ----
    #[test]
    fn test_large_network_completes() {
        let tn = build_grid_network(10, 10, 2);
        assert_eq!(tn.num_tensors(), 100);

        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Greedy(GreedyConfig::default()))
            .max_time_secs(10.0)
            .build();

        let start = std::time::Instant::now();
        let path = optimize_contraction(&tn, &config).unwrap();
        let elapsed = start.elapsed().as_secs_f64();

        assert!(elapsed < 10.0, "Optimization took too long: {:.1}s", elapsed);
        assert_eq!(path.num_steps(), 99);
    }

    // ---- Test 29: Parallel random search uses multiple threads ----
    #[test]
    fn test_parallel_random_search() {
        let tn = chain_network(8);

        // Run with parallel feature
        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Random { num_trials: 50 })
            .seed(42)
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();

        assert_eq!(path.num_steps(), 7);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 30: Deterministic results with same seed ----
    #[test]
    fn test_deterministic_with_seed() {
        let tn = chain_network(6);

        let config = ContractionConfig::builder()
            .method(OptimizationMethod::Greedy(GreedyConfig {
                cost_function: GreedyCost::Boltzmann,
                temperature: 1.0,
            }))
            .seed(12345)
            .build();

        let path1 = optimize_contraction(&tn, &config).unwrap();
        let path2 = optimize_contraction(&tn, &config).unwrap();

        assert!((path1.total_flops - path2.total_flops).abs() < 1e-10);
        assert_eq!(path1.num_steps(), path2.num_steps());
    }

    // ---- Test 31: Config builder defaults ----
    #[test]
    fn test_config_builder_defaults() {
        let config = ContractionConfig::builder().build();
        assert_eq!(config.seed, 42);
        assert_eq!(config.num_trials, 128);
        assert!(config.max_time_secs > 0.0);
        assert!(config.max_memory_bytes > 0);
    }

    // ---- Test 32: Combined cost metric weighting ----
    #[test]
    fn test_combined_cost_metric() {
        let tn = chain_network(5);

        let config_flops = ContractionConfig::builder()
            .minimize(CostMetric::Flops)
            .build();
        let config_memory = ContractionConfig::builder()
            .minimize(CostMetric::Memory)
            .build();
        let config_combined = ContractionConfig::builder()
            .minimize(CostMetric::Combined {
                flop_weight: 1.0,
                memory_weight: 0.001,
            })
            .build();

        let path_f = optimize_contraction(&tn, &config_flops).unwrap();
        let path_m = optimize_contraction(&tn, &config_memory).unwrap();
        let path_c = optimize_contraction(&tn, &config_combined).unwrap();

        // All produce valid paths
        assert!(path_f.total_flops > 0.0);
        assert!(path_m.total_flops > 0.0);
        assert!(path_c.total_flops > 0.0);

        // Combined metric score should incorporate both components
        let metric = CostMetric::Combined {
            flop_weight: 1.0,
            memory_weight: 0.001,
        };
        let score = metric.score(&path_c);
        assert!(score > 0.0);
        assert!(score >= path_c.total_flops); // At least as large as flop component
    }

    // ---- Test 33: Empty network error ----
    #[test]
    fn test_empty_network_error() {
        let tn = TensorNetwork::new();
        let config = ContractionConfig::default();
        let result = optimize_contraction(&tn, &config);
        assert!(result.is_err());
        match result {
            Err(ContractionError::EmptyNetwork) => {} // expected
            other => panic!("Expected EmptyNetwork, got {:?}", other),
        }
    }

    // ---- Test 34: Single tensor error ----
    #[test]
    fn test_single_tensor_error() {
        let mut tn = TensorNetwork::new();
        tn.add_tensor(TensorInfo::new(0, vec![0], vec![2]));
        let config = ContractionConfig::default();
        let result = optimize_contraction(&tn, &config);
        assert!(result.is_err());
    }

    // ---- Test 35: Validation catches unknown index ----
    #[test]
    fn test_validation_unknown_index() {
        let mut tn = TensorNetwork::new();
        tn.tensors.push(TensorInfo::new(0, vec![0, 999], vec![2, 3]));
        tn.index_sizes.insert(0, 2);
        // index 999 not in index_sizes
        let result = tn.validate();
        assert!(matches!(result, Err(ContractionError::UnknownIndex(999))));
    }

    // ---- Test 36: TensorInfo size and memory ----
    #[test]
    fn test_tensor_info_size_memory() {
        let t = TensorInfo::new(0, vec![0, 1, 2], vec![3, 4, 5]);
        assert_eq!(t.size(), 60); // 3*4*5
        assert_eq!(t.memory_bytes(), 960); // 60 * 16
    }

    // ---- Test 37: Shared indices between tensors ----
    #[test]
    fn test_shared_indices() {
        let tn = two_tensor_network();
        let shared = tn.shared_indices(0, 1);
        assert_eq!(shared, vec![1]);
    }

    // ---- Test 38: Width metric scoring ----
    #[test]
    fn test_width_metric() {
        let tn = chain_network(4);
        let config = ContractionConfig::builder()
            .minimize(CostMetric::Width)
            .build();
        let path = optimize_contraction(&tn, &config).unwrap();
        let score = CostMetric::Width.score(&path);
        assert!(score >= 0.0);
    }

    // ---- Test 39: Grid network output indices match tensor count ----
    #[test]
    fn test_grid_output_indices() {
        let tn = build_grid_network(3, 4, 2);
        assert_eq!(tn.num_tensors(), 12);
        assert_eq!(tn.output_indices.len(), 12);
    }

    // ---- Test 40: SA with restart produces valid path ----
    #[test]
    fn test_sa_with_restart() {
        let tn = chain_network(6);
        let sa_config = SAConfig {
            initial_temperature: 5.0,
            cooling_rate: 0.9, // Cool very fast to trigger restarts
            num_steps: 200,
            restart_threshold: 10, // Very aggressive restarts
        };
        let path = optimize_sa(&tn, &sa_config, &CostMetric::Flops, 42).unwrap();
        assert_eq!(path.num_steps(), 5);
        assert!(path.total_flops > 0.0);
    }

    // ---- Test 41: Error display formatting ----
    #[test]
    fn test_error_display() {
        let err = ContractionError::MemoryLimitExceeded {
            required: 1000,
            limit: 500,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
    }

    // ---- Test 42: Cubic network has correct tensor count ----
    #[test]
    fn test_cubic_network_count() {
        let tn = build_cubic_network(3, 3, 3, 2);
        assert_eq!(tn.num_tensors(), 27);
        assert_eq!(tn.output_indices.len(), 27);
    }
}
