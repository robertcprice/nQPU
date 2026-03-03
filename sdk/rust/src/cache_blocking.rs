//! Cache Blocking Optimization for CPU Quantum Operations
//!
//! CPU cache optimization using cache-aware blocking strategies.
//!
//! **M4 Pro CPU Cache Hierarchy**:
//! - L1: 128KB per core (data + instruction)
//! - L2: 12MB shared (vs 4MB on M4)
//! - L3: 24MB unified
//!
//! **Optimizations**:
//! - Block size tuning for cache hierarchy
//! - Cache-aware gate fusion
//! - Prefetching for sequential operations
//! - NEON SIMD with cache alignment

use crate::{QuantumState, C64};
use std::time::Instant;

/// Cache configuration for M4 Pro.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// L1 cache size per core (bytes).
    pub l1_size: usize,
    /// L2 cache size (bytes).
    pub l2_size: usize,
    /// L3 cache size (bytes).
    pub l3_size: usize,
    /// Cache line size (bytes).
    pub cache_line_size: usize,
    /// Optimal block size for L1.
    pub l1_block_size: usize,
    /// Optimal block size for L2.
    pub l2_block_size: usize,
    /// Optimal block size for L3.
    pub l3_block_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 128 * 1024,       // 128KB
            l2_size: 12 * 1024 * 1024, // 12MB (M4 Pro)
            l3_size: 24 * 1024 * 1024, // 24MB
            cache_line_size: 64,       // 64 bytes
            // Block sizes tuned to fit 3 working sets in cache
            l1_block_size: 1024,   // Fits in L1
            l2_block_size: 32768,  // Fits in L2
            l3_block_size: 262144, // Fits in L3
        }
    }
}

impl CacheConfig {
    /// Get optimal block size for given problem size.
    pub fn optimal_block_size(&self, state_size: usize) -> usize {
        // Try to fit in L1, fall back to L2, then L3
        if state_size <= self.l1_block_size {
            state_size
        } else if state_size <= self.l2_block_size {
            self.l1_block_size
        } else if state_size <= self.l3_block_size {
            self.l2_block_size
        } else {
            self.l3_block_size
        }
    }

    /// Calculate number of cache lines for a given size.
    pub fn cache_lines(&self, size: usize) -> usize {
        (size * std::mem::size_of::<C64>() + self.cache_line_size - 1) / self.cache_line_size
    }
}

/// Cache-aware quantum state with optimized operations.
pub struct CacheAwareQuantumState {
    inner: QuantumState,
    cache_config: CacheConfig,
}

impl CacheAwareQuantumState {
    /// Create a new cache-aware quantum state.
    pub fn new(num_qubits: usize) -> Self {
        let inner = QuantumState::new(num_qubits);
        let cache_config = CacheConfig::default();

        Self {
            inner,
            cache_config,
        }
    }

    /// Create with custom cache config.
    pub fn with_config(num_qubits: usize, cache_config: CacheConfig) -> Self {
        let inner = QuantumState::new(num_qubits);
        Self {
            inner,
            cache_config,
        }
    }

    /// Get inner quantum state.
    pub fn inner(&self) -> &QuantumState {
        &self.inner
    }

    /// Get mutable inner quantum state.
    pub fn inner_mut(&mut self) -> &mut QuantumState {
        &mut self.inner
    }

    /// Apply single-qubit gate with cache blocking.
    pub fn apply_single_qubit_gate_cached(
        &mut self,
        qubit: usize,
        matrix: [[C64; 2]; 2],
    ) -> Result<(), String> {
        let num_qubits = self.inner.num_qubits;
        let dim = 1usize << num_qubits;
        let stride = 1usize << qubit;

        // Use cache blocking for large states
        if dim > self.cache_config.l2_block_size {
            self.apply_single_qubit_blocked(qubit, matrix, stride, dim)?;
        } else {
            // Direct application for small states
            self.apply_single_qubit_direct(qubit, matrix, stride, dim);
        }

        Ok(())
    }

    /// Apply gate with cache blocking and real ARM prefetch.
    fn apply_single_qubit_blocked(
        &mut self,
        _qubit: usize,
        matrix: [[C64; 2]; 2],
        stride: usize,
        dim: usize,
    ) -> Result<(), String> {
        let block_size = self.cache_config.optimal_block_size(dim);
        let cache_line = self.cache_config.cache_line_size;
        let state = self.inner.amplitudes_mut();
        // Elements per cache line (C64 = 16 bytes, cache line = 64 bytes → 4 elements)
        let elems_per_line = cache_line / std::mem::size_of::<C64>();

        // Process in cache-friendly blocks
        for block_start in (0..dim).step_by(block_size) {
            let block_end = (block_start + block_size).min(dim);

            // Prefetch next block into L1 (read) and L2 (further ahead)
            let next_block_start = block_end;
            if next_block_start < dim {
                let next_block_end = (next_block_start + block_size).min(dim);
                for pf in (next_block_start..next_block_end).step_by(elems_per_line.max(1)) {
                    if pf < dim {
                        unsafe {
                            let addr = state.as_ptr().add(pf) as *const i8;
                            llvm_prefetch(addr, 0);
                        }
                    }
                }
                // Prefetch 2 blocks ahead into L2
                let far_start = next_block_end;
                if far_start < dim {
                    let far_end = (far_start + block_size).min(dim);
                    for pf in (far_start..far_end).step_by(elems_per_line.max(1) * 4) {
                        if pf < dim {
                            unsafe {
                                prefetch_l2(state.as_ptr().add(pf) as *const i8);
                            }
                        }
                    }
                }
            }

            // Apply gate to current block
            for i in (block_start..block_end).step_by(stride * 2) {
                for j in 0..stride {
                    let idx0 = i + j;
                    let idx1 = idx0 + stride;

                    if idx1 < dim {
                        let a0 = state[idx0];
                        let a1 = state[idx1];

                        state[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1;
                        state[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply gate directly without blocking.
    fn apply_single_qubit_direct(
        &mut self,
        _qubit: usize,
        matrix: [[C64; 2]; 2],
        stride: usize,
        dim: usize,
    ) {
        let state = self.inner.amplitudes_mut();

        for i in (0..dim).step_by(stride * 2) {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;

                if idx1 < dim {
                    let a0 = state[idx0];
                    let a1 = state[idx1];

                    state[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1;
                    state[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1;
                }
            }
        }
    }

    /// Apply batch of single-qubit gates with cache-aware fusion.
    pub fn apply_batch_cached(&mut self, gates: &[(usize, [[C64; 2]; 2])]) -> Result<(), String> {
        // Fuse gates that operate on non-overlapping qubits
        let fused_groups = Self::fuse_gates_internal(gates);

        for group in fused_groups {
            for (qubit, matrix) in group {
                self.apply_single_qubit_gate_cached(qubit, matrix)?;
            }
        }

        Ok(())
    }

    /// Fuse gates that can be applied in parallel.
    fn fuse_gates_internal(gates: &[(usize, [[C64; 2]; 2])]) -> Vec<Vec<(usize, [[C64; 2]; 2])>> {
        let mut groups = Vec::new();
        let mut used_qubits = std::collections::HashSet::new();
        let mut current_group = Vec::new();

        for &(qubit, matrix) in gates {
            if !used_qubits.contains(&qubit) {
                used_qubits.insert(qubit);
                current_group.push((qubit, matrix));
            } else {
                // Start new group
                if !current_group.is_empty() {
                    groups.push(current_group);
                    current_group = Vec::new();
                    used_qubits.clear();
                }
                used_qubits.insert(qubit);
                current_group.push((qubit, matrix));
            }
        }

        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
    }

    /// NEON SIMD optimized single-qubit gate application for AArch64.
    ///
    /// Uses ARM NEON f64x2 intrinsics to vectorize the 2x2 matrix-vector
    /// multiplication at each butterfly pair of amplitudes.
    ///
    /// Gate matrix [[a, b], [c, d]] applied to amplitude pair (amp0, amp1):
    ///   new_amp0 = a * amp0 + b * amp1
    ///   new_amp1 = c * amp0 + d * amp1
    ///
    /// Complex multiplication (x+iy)(u+iv) = (xu-yv) + i(xv+yu) is
    /// decomposed into NEON f64x2 operations on [re, im] pairs.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn apply_single_qubit_neon(state: &mut [C64], qubit: usize, matrix: [[C64; 2]; 2]) {
        use std::arch::aarch64::*;

        let dim = state.len();
        let stride = 1usize << qubit;

        // Extract gate matrix elements.
        let a = matrix[0][0]; // gate[0][0]
        let b = matrix[0][1]; // gate[0][1]
        let c = matrix[1][0]; // gate[1][0]
        let d = matrix[1][1]; // gate[1][1]

        // Broadcast real and imaginary parts into NEON f64x2 registers.
        // Each register holds [val, val] for use in vectorized complex mul.
        let a_re = vdupq_n_f64(a.re);
        let a_im = vdupq_n_f64(a.im);
        let b_re = vdupq_n_f64(b.re);
        let b_im = vdupq_n_f64(b.im);
        let c_re = vdupq_n_f64(c.re);
        let c_im = vdupq_n_f64(c.im);
        let d_re = vdupq_n_f64(d.re);
        let d_im = vdupq_n_f64(d.im);

        // Sign mask for complex multiplication: negate the imaginary-times-imaginary
        // product to get (re*re - im*im) and add (re*im + im*re).
        let neg_mask = vld1q_f64([1.0f64, -1.0f64].as_ptr());

        // Helper closure: multiply complex number z (as f64x2 [re, im]) by
        // complex number with broadcasted parts (m_re, m_im).
        // Result: [(z.re * m_re - z.im * m_im), (z.re * m_im + z.im * m_re)]
        //
        // Implementation:
        //   prod_rr = [z.re * m_re, z.im * m_re]
        //   prod_ii = [z.im * m_im, z.re * m_im]  (after shuffle)
        //   result  = prod_rr +/- prod_ii (using fma and sign mask)
        #[inline(always)]
        unsafe fn complex_mul_neon(
            z: float64x2_t,
            m_re: float64x2_t,
            m_im: float64x2_t,
            neg_mask: float64x2_t,
        ) -> float64x2_t {
            // z = [z.re, z.im]
            // prod_re_parts = [z.re * m_re, z.im * m_re]
            let prod_re_parts = vmulq_f64(z, m_re);

            // Swap re/im of z: [z.im, z.re]
            let z_swap = vextq_f64(z, z, 1);

            // prod_im_parts = [z.im * m_im, z.re * m_im]
            let prod_im_parts = vmulq_f64(z_swap, m_im);

            // Apply sign: [z.im * m_im * 1.0, z.re * m_im * (-1.0)]
            //           = [z.im * m_im, -(z.re * m_im)]
            // Wait -- we need:
            //   result.re = z.re * m_re - z.im * m_im
            //   result.im = z.im * m_re + z.re * m_im
            //
            // prod_re_parts = [z.re * m_re, z.im * m_re]
            // prod_im_parts = [z.im * m_im, z.re * m_im]
            //
            // We want to subtract prod_im_parts[0] from prod_re_parts[0]
            // and add prod_im_parts[1] to prod_re_parts[1].
            // Use neg_mask = [1.0, -1.0] on prod_im_parts, then subtract.
            let prod_im_signed = vmulq_f64(prod_im_parts, neg_mask);
            // prod_im_signed = [z.im * m_im, -(z.re * m_im)]
            // result = prod_re_parts - prod_im_signed
            //        = [z.re*m_re - z.im*m_im, z.im*m_re + z.re*m_im]
            vsubq_f64(prod_re_parts, prod_im_signed)
        }

        let mut i = 0;
        while i < dim {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;

                if idx1 < dim {
                    // Load amp0 and amp1 as f64x2 vectors [re, im].
                    let ptr0 = state.as_ptr().add(idx0) as *const f64;
                    let ptr1 = state.as_ptr().add(idx1) as *const f64;
                    let amp0 = vld1q_f64(ptr0);
                    let amp1 = vld1q_f64(ptr1);

                    // new_amp0 = a * amp0 + b * amp1
                    let term0 = complex_mul_neon(amp0, a_re, a_im, neg_mask);
                    let term1 = complex_mul_neon(amp1, b_re, b_im, neg_mask);
                    let new_amp0 = vaddq_f64(term0, term1);

                    // new_amp1 = c * amp0 + d * amp1
                    let term2 = complex_mul_neon(amp0, c_re, c_im, neg_mask);
                    let term3 = complex_mul_neon(amp1, d_re, d_im, neg_mask);
                    let new_amp1 = vaddq_f64(term2, term3);

                    // Store results back.
                    let out_ptr0 = state.as_mut_ptr().add(idx0) as *mut f64;
                    let out_ptr1 = state.as_mut_ptr().add(idx1) as *mut f64;
                    vst1q_f64(out_ptr0, new_amp0);
                    vst1q_f64(out_ptr1, new_amp1);
                }
            }
            i += stride << 1;
        }
    }

    /// Fallback for non-AArch64 platforms: scalar implementation.
    #[cfg(not(target_arch = "aarch64"))]
    unsafe fn apply_single_qubit_neon(state: &mut [C64], qubit: usize, matrix: [[C64; 2]; 2]) {
        let dim = state.len();
        let stride = 1usize << qubit;

        let mut i = 0;
        while i < dim {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;
                if idx1 < dim {
                    let a0 = state[idx0];
                    let a1 = state[idx1];
                    state[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1;
                    state[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1;
                }
            }
            i += stride << 1;
        }
    }

    /// Benchmark cache-aware vs standard operations.
    pub fn benchmark_cache_aware(num_qubits: usize, iterations: usize) -> CacheBenchmarkResults {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Cache Blocking Benchmark: {} qubits", num_qubits);
        println!("═══════════════════════════════════════════════════════════════");

        let h_matrix = [
            [C64::new(0.70710678, 0.0), C64::new(0.70710678, 0.0)],
            [C64::new(0.70710678, 0.0), C64::new(-0.70710678, 0.0)],
        ];

        let config = CacheConfig::default();
        println!("Cache Configuration:");
        println!("  L1:  {} KB", config.l1_size / 1024);
        println!("  L2:  {} MB", config.l2_size / (1024 * 1024));
        println!("  L3:  {} MB", config.l3_size / (1024 * 1024));
        println!(
            "  Block sizes: L1={}, L2={}, L3={}",
            config.l1_block_size, config.l2_block_size, config.l3_block_size
        );
        println!();

        // Standard state
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = QuantumState::new(num_qubits);
            for q in 0..num_qubits {
                crate::GateOperations::u(&mut state, q, &h_matrix);
            }
        }
        let standard_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Cache-aware state
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = Self::new(num_qubits);
            for q in 0..num_qubits {
                let _ = state.apply_single_qubit_gate_cached(q, h_matrix);
            }
        }
        let cache_aware_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = standard_time / cache_aware_time;

        println!("Standard operations:  {:.6} sec", standard_time);
        println!("Cache-aware:          {:.6} sec", cache_aware_time);
        println!("Cache speedup:        {:.2}x", speedup);
        println!();

        CacheBenchmarkResults {
            standard_time,
            cache_aware_time,
            speedup,
            cache_config: config,
        }
    }
}

/// Cache benchmark results.
#[derive(Clone, Debug)]
pub struct CacheBenchmarkResults {
    pub standard_time: f64,
    pub cache_aware_time: f64,
    pub speedup: f64,
    pub cache_config: CacheConfig,
}

/// Prefetch for read into L1 data cache.
#[inline(always)]
fn llvm_prefetch(addr: *const i8, _rw: i32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // prfm pldl1keep: prefetch for load, L1 data cache, temporal (keep)
        std::arch::asm!(
            "prfm pldl1keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = addr;
    }
}

/// Prefetch for write (store) into L1 data cache.
#[inline(always)]
#[allow(dead_code)]
fn prefetch_write(addr: *const i8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // prfm pstl1keep: prefetch for store, L1 data cache, temporal (keep)
        std::arch::asm!(
            "prfm pstl1keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = addr;
    }
}

/// Prefetch for read into L2 cache.
#[inline(always)]
#[allow(dead_code)]
fn prefetch_l2(addr: *const i8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl2keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = addr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.l1_size, 128 * 1024);
        assert_eq!(config.l2_size, 12 * 1024 * 1024);
        assert_eq!(config.l3_size, 24 * 1024 * 1024);
    }

    #[test]
    fn test_optimal_block_size() {
        let config = CacheConfig::default();
        assert_eq!(config.optimal_block_size(512), 512);
        assert_eq!(config.optimal_block_size(10000), config.l1_block_size);
        assert_eq!(config.optimal_block_size(100000), config.l2_block_size);
    }

    #[test]
    fn test_cache_aware_state_creation() {
        let state = CacheAwareQuantumState::new(10);
        assert_eq!(state.inner().num_qubits, 10);
    }

    #[test]
    fn test_gate_fusion() {
        let state = CacheAwareQuantumState::new(10);
        let h_matrix = [
            [C64::new(0.70710678, 0.0), C64::new(0.70710678, 0.0)],
            [C64::new(0.70710678, 0.0), C64::new(-0.70710678, 0.0)],
        ];

        let gates = vec![(0, h_matrix), (1, h_matrix), (0, h_matrix), (2, h_matrix)];
        let fused = CacheAwareQuantumState::fuse_gates_internal(&gates);

        // Should create groups: [(0,1), (0), (2)]
        assert!(fused.len() >= 2);
    }
}
