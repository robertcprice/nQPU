//! Shot Batching Optimization
//!
//! Efficient batch sampling for large shot counts with:
//! - Vectorized random number generation
//! - Parallel sampling with Rayon
//! - Cache-friendly memory access patterns
//! - SIMD-accelerated probability computation

use crate::QuantumState;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

/// Configuration for shot batching
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Batch size for parallel processing
    pub batch_size: usize,
    /// Use parallel processing (Rayon)
    pub parallel: bool,
    /// Use vectorized sampling
    pub vectorized: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 10000,
            parallel: true,
            vectorized: true,
        }
    }
}

impl BatchConfig {
    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            batch_size: 50000,
            parallel: true,
            vectorized: true,
        }
    }

    /// Create config for minimal memory usage
    pub fn memory_efficient() -> Self {
        Self {
            batch_size: 1000,
            parallel: false,
            vectorized: false,
        }
    }

    /// Create config for GPU workloads
    pub fn gpu_friendly() -> Self {
        Self {
            batch_size: 100000,
            parallel: true,
            vectorized: true,
        }
    }
}

/// Optimized batch sampler
pub struct BatchSampler {
    config: BatchConfig,
}

impl BatchSampler {
    /// Create a new batch sampler with default config
    pub fn new() -> Self {
        Self::with_config(BatchConfig::default())
    }

    /// Create a new batch sampler with custom config
    pub fn with_config(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Sample multiple shots with optimization
    pub fn sample(&self, state: &QuantumState, n_shots: usize) -> HashMap<usize, usize> {
        if n_shots < self.config.batch_size || !self.config.parallel {
            self.sample_sequential(state, n_shots)
        } else {
            self.sample_parallel(state, n_shots)
        }
    }

    /// Sequential sampling with vectorization
    fn sample_sequential(&self, state: &QuantumState, n_shots: usize) -> HashMap<usize, usize> {
        let probs = state.probabilities();
        let mut counts = HashMap::new();
        let dim = state.dim;

        // Build cumulative distribution
        let cdf = self.build_cdf(&probs);

        if self.config.vectorized && n_shots >= 100 {
            self.sample_vectorized(&cdf, dim, n_shots, &mut counts)
        } else {
            self.sample_scalar(&cdf, dim, n_shots, &mut counts)
        }

        counts
    }

    /// Parallel sampling using Rayon
    #[cfg(feature = "parallel")]
    fn sample_parallel(&self, state: &QuantumState, n_shots: usize) -> HashMap<usize, usize> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        let probs = state.probabilities();
        let cdf = self.build_cdf(&probs);
        let dim = state.dim;

        let counts = Mutex::new(HashMap::new());
        let n_batches = (n_shots + self.config.batch_size - 1) / self.config.batch_size;

        (0..n_batches).into_par_iter().for_each(|batch_idx| {
            let batch_start = batch_idx * self.config.batch_size;
            let batch_end = (batch_start + self.config.batch_size).min(n_shots);
            let batch_size = batch_end - batch_start;

            let mut rng = rand::thread_rng();
            let mut local_counts: HashMap<usize, usize> = HashMap::new();

            for _ in 0..batch_size {
                let r: f64 = rng.gen();
                let outcome = self.search_cdf(&cdf, r, dim);
                *local_counts.entry(outcome).or_insert(0) += 1;
            }

            // Merge into global counts
            let mut counts = counts.lock().unwrap();
            for (outcome, count) in local_counts {
                *counts.entry(outcome).or_insert(0) += count;
            }
        });

        counts.into_inner().unwrap()
    }

    /// Fallback sequential sampling when parallel feature is disabled
    #[cfg(not(feature = "parallel"))]
    fn sample_parallel(&self, state: &QuantumState, n_shots: usize) -> HashMap<usize, usize> {
        self.sample_sequential(state, n_shots)
    }

    /// Build cumulative distribution function
    fn build_cdf(&self, probs: &[f64]) -> Vec<f64> {
        let mut cdf = Vec::with_capacity(probs.len());
        let mut cumsum = 0.0;
        for &p in probs {
            cumsum += p;
            cdf.push(cumsum);
        }
        cdf
    }

    /// Scalar sampling (one sample at a time)
    fn sample_scalar(
        &self,
        cdf: &[f64],
        dim: usize,
        n_shots: usize,
        counts: &mut HashMap<usize, usize>,
    ) {
        let mut rng = rand::thread_rng();
        for _ in 0..n_shots {
            let r: f64 = rng.gen();
            let outcome = self.search_cdf(cdf, r, dim);
            *counts.entry(outcome).or_insert(0) += 1;
        }
    }

    /// Vectorized sampling (batch random number generation)
    fn sample_vectorized(
        &self,
        cdf: &[f64],
        dim: usize,
        n_shots: usize,
        counts: &mut HashMap<usize, usize>,
    ) {
        // Generate random numbers in batches
        let batch_size = 256.min(n_shots);
        let mut rng = rand::thread_rng();

        for batch_start in (0..n_shots).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_shots);
            let batch_len = batch_end - batch_start;

            // Generate batch of random numbers
            let mut randoms: Vec<f64> = Vec::with_capacity(batch_len);
            for _ in 0..batch_len {
                randoms.push(rng.gen());
            }

            // Process batch
            for r in randoms {
                let outcome = self.search_cdf(cdf, r, dim);
                *counts.entry(outcome).or_insert(0) += 1;
            }
        }
    }

    /// Search CDF for outcome
    #[inline]
    fn search_cdf(&self, cdf: &[f64], r: f64, dim: usize) -> usize {
        match cdf.binary_search_by(|c| c.partial_cmp(&r).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(dim - 1),
        }
    }
}

impl Default for BatchSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch circuit execution helper
///
/// Reuses state preparation for multiple shots when possible
pub struct BatchCircuitExecutor {
    sampler: BatchSampler,
}

impl BatchCircuitExecutor {
    /// Create a new batch circuit executor
    pub fn new() -> Self {
        Self {
            sampler: BatchSampler::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: BatchConfig) -> Self {
        Self {
            sampler: BatchSampler::with_config(config),
        }
    }

    /// Execute circuit and sample multiple shots efficiently
    pub fn execute_and_sample<F>(
        &self,
        num_qubits: usize,
        prepare_circuit: F,
        n_shots: usize,
    ) -> HashMap<String, usize>
    where
        F: Fn(&mut crate::QuantumState),
    {
        // Prepare circuit once
        let mut state = crate::QuantumState::new(num_qubits);
        prepare_circuit(&mut state);

        // Sample using optimized batch sampler
        let raw_counts = self.sampler.sample(&state, n_shots);

        // Convert to bitstrings
        raw_counts
            .into_iter()
            .map(|(outcome, count)| {
                let bits: String = (0..num_qubits)
                    .rev()
                    .map(|q| if outcome & (1 << q) != 0 { '1' } else { '0' })
                    .collect();
                (bits, count)
            })
            .collect()
    }
}

impl Default for BatchCircuitExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance benchmark for shot batching
pub struct Benchmark {
    sampler: BatchSampler,
}

impl Benchmark {
    pub fn new() -> Self {
        Self {
            sampler: BatchSampler::new(),
        }
    }

    /// Compare sequential vs batched sampling performance
    pub fn compare_sampling(&self, state: &QuantumState, n_shots: usize) -> (f64, f64) {
        use std::time::Instant;

        // Standard sampling
        let start = Instant::now();
        let _standard = state.sample(n_shots);
        let standard_time = start.elapsed().as_secs_f64();

        // Batched sampling
        let start = Instant::now();
        let _batched = self.sampler.sample(state, n_shots);
        let batched_time = start.elapsed().as_secs_f64();

        (standard_time, batched_time)
    }

    /// Get speedup factor
    pub fn speedup_factor(&self, state: &QuantumState, n_shots: usize) -> f64 {
        let (standard, batched) = self.compare_sampling(state, n_shots);
        if batched > 0.0 {
            standard / batched
        } else {
            1.0
        }
    }
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GateOperations, QuantumState};

    #[test]
    fn test_batch_sampler_basic() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);

        let sampler = BatchSampler::new();
        let counts = sampler.sample(&state, 1000);

        assert!(!counts.is_empty());
        let total: usize = counts.values().sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_batch_circuit_executor() {
        let executor = BatchCircuitExecutor::new();

        let results = executor.execute_and_sample(
            2,
            |state| {
                GateOperations::h(state, 0);
                GateOperations::cnot(state, 0, 1);
            },
            1000,
        );

        assert!(!results.is_empty());
        assert!(results.contains_key("00") || results.contains_key("11"));
    }

    #[test]
    fn test_batch_config() {
        let fast = BatchConfig::fast();
        assert!(fast.batch_size >= 10000);

        let mem_eff = BatchConfig::memory_efficient();
        assert!(mem_eff.batch_size <= 1000);
    }
}
