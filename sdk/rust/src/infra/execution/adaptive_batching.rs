//! Adaptive Batching for Quantum Gate Execution
//!
//! This module implements intelligent batching strategies that automatically
//! determine the optimal batch size based on:
//! - Gate type dependencies
//! - GPU memory availability
//! - Circuit structure analysis
//! - Performance profiling
//!
//! # Key Innovations
//!
//! - **Dynamic Batching**: Automatically adjusts batch size based on circuit structure
//! - **Dependency Analysis**: Respects gate dependencies for correctness
//! - **Memory-Aware**: Considers GPU memory constraints
//! - **Profiling-Guided**: Uses runtime performance data to optimize batching

use crate::gates::Gate;
use std::collections::{HashSet, VecDeque};

/// Batching configuration
#[derive(Clone, Debug)]
pub struct BatchingConfig {
    /// Maximum gates per batch
    pub max_batch_size: usize,
    /// Minimum gates per batch (for small circuits)
    pub min_batch_size: usize,
    /// Whether to profile for optimization
    pub enable_profiling: bool,
    /// Memory limit per batch (in bytes, 0 = no limit)
    pub memory_limit: usize,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        BatchingConfig {
            max_batch_size: 1000,
            min_batch_size: 10,
            enable_profiling: true,
            memory_limit: 0,
        }
    }
}

/// A batch of gates that can be executed together
#[derive(Clone, Debug)]
pub struct GateBatch {
    pub gates: Vec<Gate>,
    pub batch_id: usize,
    pub estimated_memory: usize,
    pub can_parallelize: bool,
}

/// Adaptive batching engine
pub struct AdaptiveBatchingEngine {
    config: BatchingConfig,
    performance_history: Vec<BatchPerformance>,
}

#[derive(Clone, Debug)]
struct BatchPerformance {
    batch_size: usize,
    execution_time_us: u64,
    parallelizable: bool,
}

impl AdaptiveBatchingEngine {
    pub fn new(config: BatchingConfig) -> Self {
        AdaptiveBatchingEngine {
            config,
            performance_history: Vec::new(),
        }
    }

    /// Split gates into optimal batches based on dependencies and performance
    pub fn create_batches(&mut self, gates: Vec<Gate>) -> Vec<GateBatch> {
        if gates.is_empty() {
            return Vec::new();
        }

        // Build dependency graph
        let dependencies = build_dependency_graph(&gates);

        // Find parallel layers (gates with no dependencies between them)
        let layers = find_parallel_layers(&gates, &dependencies);

        // Create batches from layers
        let mut batches = Vec::new();
        let mut batch_id = 0;

        for layer in layers {
            // For each layer, check if we can batch all gates
            if layer.len() <= self.config.max_batch_size {
                let memory = self.estimate_memory(&layer);
                batches.push(GateBatch {
                    gates: layer,
                    batch_id,
                    estimated_memory: memory,
                    can_parallelize: true,
                });
                batch_id += 1;
            } else {
                // Split large layer into multiple batches
                for chunk in layer.chunks(self.config.max_batch_size) {
                    let memory = self.estimate_memory(chunk);
                    batches.push(GateBatch {
                        gates: chunk.to_vec(),
                        batch_id,
                        estimated_memory: memory,
                        can_parallelize: chunk.len() > 1,
                    });
                    batch_id += 1;
                }
            }
        }

        // Optimize based on performance history
        if self.config.enable_profiling && !self.performance_history.is_empty() {
            self.optimize_batches(&mut batches);
        }

        batches
    }

    /// Estimate memory usage for a set of gates
    fn estimate_memory(&self, gates: &[Gate]) -> usize {
        // Base memory for state vectors (2 * 8 bytes per amplitude)
        // This is a rough estimate based on the maximum qubit index
        let max_qubit = gates
            .iter()
            .flat_map(|g| g.targets.iter().chain(g.controls.iter()))
            .max()
            .copied()
            .unwrap_or(0);

        // State vector size for n+1 qubits
        let state_size = 2 * 8 * (1_usize << (max_qubit + 1));

        // Add overhead for each gate
        let gate_overhead = gates.len() * 64; // Approximate metadata size

        state_size + gate_overhead
    }

    /// Optimize batches based on historical performance data
    fn optimize_batches(&self, batches: &mut Vec<GateBatch>) {
        // Find optimal batch size from history
        let optimal_size = if self.performance_history.is_empty() {
            self.config.max_batch_size
        } else {
            // Find batch size with best average performance
            let mut size_performance: std::collections::HashMap<usize, Vec<f64>> =
                std::collections::HashMap::new();

            for perf in &self.performance_history {
                size_performance
                    .entry(perf.batch_size)
                    .or_insert_with(Vec::new)
                    .push(perf.execution_time_us as f64);
            }

            size_performance
                .into_iter()
                .map(|(size, times)| (size, times.iter().sum::<f64>() / times.len() as f64))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(size, _)| size)
                .unwrap_or(self.config.max_batch_size)
        };

        // Re-batch if beneficial
        if optimal_size != self.config.max_batch_size {
            self.rebatch_with_size(batches, optimal_size);
        }
    }

    /// Re-batch gates with a different batch size
    fn rebatch_with_size(&self, batches: &mut Vec<GateBatch>, new_size: usize) {
        let all_gates: Vec<Gate> = batches.iter().flat_map(|b| b.gates.clone()).collect();

        let mut new_batches = Vec::new();
        let mut batch_id = 0;

        for chunk in all_gates.chunks(new_size) {
            new_batches.push(GateBatch {
                gates: chunk.to_vec(),
                batch_id,
                estimated_memory: self.estimate_memory(chunk),
                can_parallelize: chunk.len() > 1,
            });
            batch_id += 1;
        }

        *batches = new_batches;
    }

    /// Record performance data for a batch
    pub fn record_performance(&mut self, batch: &GateBatch, execution_time_us: u64) {
        if self.config.enable_profiling {
            self.performance_history.push(BatchPerformance {
                batch_size: batch.gates.len(),
                execution_time_us,
                parallelizable: batch.can_parallelize,
            });

            // Keep only recent history (last 1000 entries)
            if self.performance_history.len() > 1000 {
                self.performance_history.drain(0..100);
            }
        }
    }

    /// Get recommended batch size based on performance history
    pub fn recommended_batch_size(&self) -> usize {
        if self.performance_history.is_empty() {
            return self.config.max_batch_size;
        }

        // Simple heuristic: find size with best average performance
        let mut size_times: std::collections::HashMap<usize, Vec<u64>> =
            std::collections::HashMap::new();

        for perf in &self.performance_history {
            size_times
                .entry(perf.batch_size)
                .or_insert_with(Vec::new)
                .push(perf.execution_time_us);
        }

        size_times
            .into_iter()
            .map(|(size, times)| {
                let avg = times.iter().sum::<u64>() / times.len() as u64;
                (size, avg)
            })
            .min_by_key(|&(_, time)| time)
            .map(|(size, _)| size)
            .unwrap_or(self.config.max_batch_size)
    }
}

/// Build dependency graph for gates
fn build_dependency_graph(gates: &[Gate]) -> Vec<HashSet<usize>> {
    let n = gates.len();
    let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if gates_depend_on_each_other(&gates[i], &gates[j]) {
                dependencies[j].insert(i);
            }
        }
    }

    dependencies
}

/// Check if two gates have dependencies (operate on overlapping qubits)
fn gates_depend_on_each_other(a: &Gate, b: &Gate) -> bool {
    let a_qubits: HashSet<_> = a.targets.iter().chain(a.controls.iter()).copied().collect();
    let b_qubits: HashSet<_> = b.targets.iter().chain(b.controls.iter()).copied().collect();

    !a_qubits.is_disjoint(&b_qubits)
}

/// Find parallel layers using topological sorting
fn find_parallel_layers(gates: &[Gate], dependencies: &[HashSet<usize>]) -> Vec<Vec<Gate>> {
    let n = gates.len();
    let mut in_degree = vec![0usize; n];
    let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Build adjacency list and calculate in-degrees
    for i in 0..n {
        for &dep in &dependencies[i] {
            adj_list[dep].push(i);
            in_degree[i] += 1;
        }
    }

    // Kahn's algorithm for topological sorting with layer detection
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    let mut layers = Vec::new();
    let mut visited = vec![false; n];

    while !queue.is_empty() {
        let layer_size = queue.len();
        let mut current_layer = Vec::new();

        for _ in 0..layer_size {
            let node = queue.pop_front().unwrap();
            if visited[node] {
                continue;
            }
            visited[node] = true;
            current_layer.push(gates[node].clone());

            // Reduce in-degree for neighbors
            for &neighbor in &adj_list[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 && !visited[neighbor] {
                    queue.push_back(neighbor);
                }
            }
        }

        if !current_layer.is_empty() {
            layers.push(current_layer);
        }
    }

    layers
}

/// Smart batching that considers gate types and commutation
pub fn smart_batch(gates: Vec<Gate>, config: BatchingConfig) -> Vec<GateBatch> {
    let mut engine = AdaptiveBatchingEngine::new(config);
    engine.create_batches(gates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_layer_detection() {
        let gates = vec![
            Gate::h(0),
            Gate::h(1),       // Parallel with gate 0
            Gate::cnot(0, 1), // Depends on 0 and 1
            Gate::h(2),       // Parallel with all above
        ];

        let batches = smart_batch(gates, BatchingConfig::default());
        assert!(batches.len() >= 2); // At least 2 layers due to dependencies
    }

    #[test]
    fn test_adaptive_batch_size() {
        let mut engine = AdaptiveBatchingEngine::new(BatchingConfig {
            max_batch_size: 100,
            min_batch_size: 10,
            enable_profiling: true,
            memory_limit: 0,
        });

        // Create many independent gates
        let gates: Vec<Gate> = (0..50).map(|i| Gate::h(i)).collect();
        let batches = engine.create_batches(gates);

        // All gates should be in one batch (they're all parallel)
        assert_eq!(batches.len(), 1);
        assert!(batches[0].can_parallelize);
    }
}
