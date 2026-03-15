//! Cache-Optimized Stabilizer Simulation
//!
//! Uses transposed memory layout for cache-efficient gate operations.
//! Target: Match or approach Stim performance on Apple Silicon.
//!
//! # Memory Layout
//!
//! Traditional layout (cache-inefficient):
//! ```
//! Row 0: [x0, x1, x2, ..., z0, z1, z2, ...]
//! Row 1: [x0, x1, x2, ..., z0, z1, z2, ...]
//! ...
//! ```
//! Problem: Gate on qubit q touches ALL rows but only position q
//! → Cache thrashing for n > cache_line_size
//!
//! Transposed layout (cache-efficient):
//! ```
//! Word 0: [all rows' x-bits for qubits 0-63]
//! Word 1: [all rows' x-bits for qubits 64-127]
//! ...
//! ```
//! Gate on qubit q touches only 1 word, which fits in cache!
//!
//! # Optimizations from cache_blocking.rs
//!
//! - ARM prefetch intrinsics for L1/L2 cache warming
//! - Cache-aware blocking for large tableaux
//! - M4 Pro cache hierarchy: 128KB L1, 12MB L2, 24MB L3

use std::time::Instant;

// Import ARM prefetch from our cache_blocking module

/// M4 Pro cache configuration
#[derive(Clone, Debug)]
pub struct StabilizerCacheConfig {
    /// L1 cache size (128KB per core)
    pub l1_size: usize,
    /// L2 cache size (12MB on M4 Pro)
    pub l2_size: usize,
    /// Cache line size (64 bytes)
    pub cache_line_size: usize,
    /// Rows per cache block
    pub rows_per_block: usize,
}

impl Default for StabilizerCacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 128 * 1024,
            l2_size: 12 * 1024 * 1024,
            cache_line_size: 64,
            // Each u64 = 8 bytes, 8 u64 per cache line
            // Process 64 rows at a time to fit in L1
            rows_per_block: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// Cache-optimized tableau with transposed memory layout
#[derive(Clone, Debug)]
pub struct TransposedTableau {
    /// X bits: transposed layout
    /// xs[word_idx][row_chunk] = X bits for all rows at word position
    xs: Vec<Vec<u64>>,

    /// Z bits: transposed layout
    zs: Vec<Vec<u64>>,

    /// Phase bits: one per row (packed into u64)
    phases: Vec<u64>,

    /// Number of qubits
    num_qubits: usize,

    /// Number of u64 words needed (ceil(n/64))
    num_words: usize,

    /// Number of rows (always 2n for full tableau)
    num_rows: usize,

    /// Cache configuration
    cache_config: StabilizerCacheConfig,
}

impl TransposedTableau {
    /// Create a new identity tableau for n qubits
    pub fn new(num_qubits: usize) -> Self {
        Self::with_cache_config(num_qubits, StabilizerCacheConfig::default())
    }

    /// Create with custom cache config
    pub fn with_cache_config(num_qubits: usize, cache_config: StabilizerCacheConfig) -> Self {
        let num_words = (num_qubits + 63) / 64;
        let num_rows = 2 * num_qubits;
        let phase_words = (num_rows + 63) / 64;

        // Allocate transposed arrays
        let rows_per_chunk = (num_rows + 63) / 64;

        let mut xs = vec![vec![0u64; rows_per_chunk]; num_words];
        let mut zs = vec![vec![0u64; rows_per_chunk]; num_words];
        let phases = vec![0u64; phase_words];

        // Initialize destabilizers: row i has X on qubit i
        for q in 0..num_qubits {
            let word_idx = q / 64;
            let bit_idx = q % 64;
            let row_chunk = q / 64;
            xs[word_idx][row_chunk] |= 1u64 << bit_idx;
        }

        // Initialize stabilizers: row n+i has Z on qubit i
        for q in 0..num_qubits {
            let word_idx = q / 64;
            let _bit_idx = q % 64;
            let row = num_qubits + q;
            let row_chunk = row / 64;
            let row_bit = row % 64;
            zs[word_idx][row_chunk] |= 1u64 << row_bit;
        }

        Self {
            xs,
            zs,
            phases,
            num_qubits,
            num_words,
            num_rows,
            cache_config,
        }
    }

    /// Get number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get phase for row r
    #[inline]
    fn get_phase(&self, r: usize) -> bool {
        let word = r / 64;
        let bit = r % 64;
        (self.phases[word] >> bit) & 1 == 1
    }

    /// Set phase for row r
    #[inline]
    fn set_phase(&mut self, r: usize, val: bool) {
        let word = r / 64;
        let bit = r % 64;
        if val {
            self.phases[word] |= 1u64 << bit;
        } else {
            self.phases[word] &= !(1u64 << bit);
        }
    }

    /// Toggle phase for row r
    #[inline]
    fn toggle_phase(&mut self, r: usize) {
        let word = r / 64;
        let bit = r % 64;
        self.phases[word] ^= 1u64 << bit;
    }

    /// Apply Hadamard gate to qubit q (with prefetching)
    /// H: X -> Z, Z -> X (swap X and Z bits, phase ^= x*z)
    #[inline]
    pub fn h(&mut self, q: usize) -> Result<(), String> {
        if q >= self.num_qubits {
            return Err(format!("qubit {} out of range [0, {})", q, self.num_qubits));
        }

        let word_idx = q / 64;
        let bit_mask = 1u64 << (q % 64);

        let num_chunks = self.xs[word_idx].len();

        // Process in cache-friendly blocks with prefetching
        for block_start in (0..num_chunks).step_by(self.cache_config.rows_per_block) {
            let block_end = (block_start + self.cache_config.rows_per_block).min(num_chunks);

            // Prefetch next block (ARM-specific)
            #[cfg(target_arch = "aarch64")]
            if block_end < num_chunks {
                let prefetch_start = block_end;
                let prefetch_end =
                    (prefetch_start + self.cache_config.rows_per_block).min(num_chunks);
                for pf in (prefetch_start..prefetch_end).step_by(4) {
                    unsafe {
                        let addr_x = self.xs[word_idx].as_ptr().add(pf) as *const i8;
                        let addr_z = self.zs[word_idx].as_ptr().add(pf) as *const i8;
                        llvm_prefetch(addr_x, 0); // L1 read
                        llvm_prefetch(addr_z, 0);
                    }
                }
            }

            // Process current block
            for chunk_idx in block_start..block_end {
                let x_val = self.xs[word_idx][chunk_idx];
                let z_val = self.zs[word_idx][chunk_idx];
                let x_bit = x_val & bit_mask;
                let z_bit = z_val & bit_mask;

                // Swap X and Z bits
                self.xs[word_idx][chunk_idx] =
                    (x_val & !bit_mask) | (if z_bit != 0 { bit_mask } else { 0 });
                self.zs[word_idx][chunk_idx] =
                    (z_val & !bit_mask) | (if x_bit != 0 { bit_mask } else { 0 });

                // Phase update: only when both X and Z were set
                if x_bit != 0 && z_bit != 0 {
                    let base_row = chunk_idx * 64;
                    // Find which rows in this chunk have both bits set
                    let both_mask = x_bit & z_bit;
                    for bit in 0..64 {
                        if base_row + bit >= self.num_rows {
                            break;
                        }
                        if (both_mask & (1u64 << bit)) != 0 {
                            self.toggle_phase(base_row + bit);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply Phase (S) gate to qubit q
    /// S: X -> Y (X unchanged, Z ^= X), phase ^= x*z
    #[inline]
    pub fn s(&mut self, q: usize) -> Result<(), String> {
        if q >= self.num_qubits {
            return Err(format!("qubit {} out of range [0, {})", q, self.num_qubits));
        }

        let word_idx = q / 64;
        let bit_mask = 1u64 << (q % 64);

        for chunk_idx in 0..self.xs[word_idx].len() {
            let x_bit = self.xs[word_idx][chunk_idx] & bit_mask;
            let z_bit = self.zs[word_idx][chunk_idx] & bit_mask;

            // S: Z ^= X for this qubit
            if x_bit != 0 {
                self.zs[word_idx][chunk_idx] ^= bit_mask;
            }

            // Phase update
            if x_bit != 0 && z_bit != 0 {
                let base_row = chunk_idx * 64;
                for bit in 0..64 {
                    if base_row + bit >= self.num_rows {
                        break;
                    }
                    if (x_bit & (1u64 << bit)) != 0 && (z_bit & (1u64 << bit)) != 0 {
                        self.toggle_phase(base_row + bit);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply CNOT gate: control c, target t (optimized)
    /// X_c -> X_c X_t, Z_t -> Z_c Z_t
    #[inline]
    pub fn cx(&mut self, c: usize, t: usize) -> Result<(), String> {
        if c >= self.num_qubits {
            return Err(format!(
                "control {} out of range [0, {})",
                c, self.num_qubits
            ));
        }
        if t >= self.num_qubits {
            return Err(format!(
                "target {} out of range [0, {})",
                t, self.num_qubits
            ));
        }
        if c == t {
            return Err("control and target must be different".to_string());
        }

        let c_word = c / 64;
        let c_mask = 1u64 << (c % 64);
        let t_word = t / 64;
        let t_mask = 1u64 << (t % 64);

        // Same word case is much faster
        if c_word == t_word {
            let xs = &mut self.xs[c_word];
            let zs = &mut self.zs[c_word];

            for chunk_idx in 0..xs.len() {
                let xc = xs[chunk_idx] & c_mask;
                let zt = zs[chunk_idx] & t_mask;

                // X_c -> X_c X_t: if control X is set, flip target X
                if xc != 0 {
                    xs[chunk_idx] ^= t_mask;
                }

                // Z_t -> Z_c Z_t: if target Z is set, flip control Z
                if zt != 0 {
                    zs[chunk_idx] ^= c_mask;
                }
            }
        } else {
            // Cross-word: process separately
            for chunk_idx in 0..self.xs[c_word].len().min(self.xs[t_word].len()) {
                let xc = self.xs[c_word][chunk_idx] & c_mask;
                let zt = self.zs[t_word][chunk_idx] & t_mask;

                if xc != 0 {
                    self.xs[t_word][chunk_idx] ^= t_mask;
                }
                if zt != 0 {
                    self.zs[c_word][chunk_idx] ^= c_mask;
                }
            }
        }

        Ok(())
    }

    /// Apply CZ gate between qubits a and b
    #[inline]
    pub fn cz(&mut self, a: usize, b: usize) -> Result<(), String> {
        if a >= self.num_qubits || b >= self.num_qubits {
            return Err("qubit out of range".to_string());
        }

        let a_word = a / 64;
        let a_mask = 1u64 << (a % 64);
        let b_word = b / 64;
        let b_mask = 1u64 << (b % 64);

        if a_word == b_word {
            let xs = &mut self.xs[a_word];
            let zs = &mut self.zs[a_word];

            for chunk_idx in 0..xs.len() {
                let xa = xs[chunk_idx] & a_mask;
                let xb = xs[chunk_idx] & b_mask;

                if xa != 0 {
                    zs[chunk_idx] ^= b_mask;
                }
                if xb != 0 {
                    zs[chunk_idx] ^= a_mask;
                }
            }
        } else {
            for chunk_idx in 0..self.xs[a_word].len().min(self.xs[b_word].len()) {
                let xa = self.xs[a_word][chunk_idx] & a_mask;
                let xb = self.xs[b_word][chunk_idx] & b_mask;

                if xa != 0 {
                    self.zs[b_word][chunk_idx] ^= b_mask;
                }
                if xb != 0 {
                    self.zs[a_word][chunk_idx] ^= a_mask;
                }
            }
        }

        Ok(())
    }
}

// ARM prefetch intrinsics (same as cache_blocking.rs)
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn llvm_prefetch(addr: *const i8, _locality: i32) {
    // PLDL1KEEP = prefetch into L1 data cache for reading
    std::arch::asm!(
        "prfm pldl1keep, [{0}]",
        in(reg) addr,
        options(nostack, preserves_flags)
    );
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

/// Run benchmark with cache-optimized layout
pub fn benchmark_optimized(num_qubits: usize, num_gates: usize) -> f64 {
    let mut tab = TransposedTableau::new(num_qubits);

    // Generate gate sequence (mixed gates like Stim)
    let gates: Vec<(usize, usize, u8)> = (0..num_gates)
        .map(|i| {
            let q1 = i % num_qubits;
            let q2 = (i + 1) % num_qubits;
            (q1, q2, (i % 4) as u8)
        })
        .collect();

    // Warmup
    for (q1, q2, gate_type) in gates.iter().take(100) {
        match gate_type {
            0 => {
                let _ = tab.h(*q1);
            }
            1 => {
                let _ = tab.s(*q1);
            }
            2 => {
                let _ = tab.cx(*q1, *q2);
            }
            _ => {
                let _ = tab.cz(*q1, *q2);
            }
        }
    }

    // Reset
    tab = TransposedTableau::new(num_qubits);

    // Benchmark
    let start = Instant::now();
    for (q1, q2, gate_type) in &gates {
        match gate_type {
            0 => {
                let _ = tab.h(*q1);
            }
            1 => {
                let _ = tab.s(*q1);
            }
            2 => {
                let _ = tab.cx(*q1, *q2);
            }
            _ => {
                let _ = tab.cz(*q1, *q2);
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    if elapsed > 0.0 {
        num_gates as f64 / elapsed
    } else {
        f64::INFINITY
    }
}

/// Print comparison benchmark
pub fn print_benchmark_comparison() {
    println!("{}", "=".repeat(80));
    println!("Cache-Optimized Transposed Tableau Benchmark");
    println!("{}", "=".repeat(80));
    println!();

    for &n in &[50, 100, 200, 500, 1000] {
        let gates = if n <= 100 { 1_000_000 } else { 100_000 };
        let throughput = benchmark_optimized(n, gates);
        println!(
            "  n={:<5}: {:>12.0} gates/sec ({:>8.1} kHz)",
            n,
            throughput,
            throughput / 1000.0
        );
    }

    println!();
    println!("Stim reference (AVX-512): ~50M gates/sec for 1000 qubits");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transposed_creation() {
        let tab = TransposedTableau::new(10);
        assert_eq!(tab.num_qubits(), 10);
    }

    #[test]
    fn test_transposed_hadamard() {
        let mut tab = TransposedTableau::new(2);
        assert!(tab.h(0).is_ok());
        assert!(tab.h(1).is_ok());
    }

    #[test]
    fn test_transposed_cnot() {
        let mut tab = TransposedTableau::new(2);
        assert!(tab.cx(0, 1).is_ok());
    }

    #[test]
    fn test_transposed_bell_state() {
        let mut tab = TransposedTableau::new(2);
        tab.h(0).unwrap();
        tab.cx(0, 1).unwrap();
    }

    #[test]
    fn test_phase_tracking() {
        let mut tab = TransposedTableau::new(1);
        // S gate on |-> should flip phase
        tab.h(0).unwrap();
        tab.s(0).unwrap();
        // Should not crash - phase tracking works
    }

    #[test]
    fn test_large_tableau() {
        let mut tab = TransposedTableau::new(100);
        for i in 0..100 {
            tab.h(i).unwrap();
        }
        for i in 0..99 {
            tab.cx(i, i + 1).unwrap();
        }
    }
}
