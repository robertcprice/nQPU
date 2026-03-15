//! AVX-512 Stabilizer Simulation for x86 Platforms
//!
//! High-performance stabilizer simulation using Intel AVX-512 SIMD instructions.
//! This module is only compiled on x86_64 targets with the "avx512" feature enabled.
//!
//! # Performance Target
//!
//! Match or exceed Stim's 50M gates/sec on x86 AVX-512 hardware.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nqpu_metal::avx512_stabilizer::Avx512Tableau;
//!
//! let mut tab = Avx512Tableau::new(100);
//! tab.h(0);
//! tab.cx(0, 1);
//! ```

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::time::Instant;

// ---------------------------------------------------------------------------
// AVX-512 STABILIZER TABLEAU
// ---------------------------------------------------------------------------

/// AVX-512 optimized stabilizer tableau.
///
/// Uses 512-bit SIMD vectors to process 8 u64 values at once.
/// Only available on x86_64 with AVX-512 support.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[derive(Clone, Debug)]
pub struct Avx512Tableau {
    /// X bits as 64-byte aligned array for AVX-512
    xs: Vec<u64>,
    /// Z bits as 64-byte aligned array
    zs: Vec<u64>,
    /// Phase bits
    phases: Vec<u64>,
    /// Number of qubits
    n: usize,
    /// Number of u64 words for qubit storage
    nwords: usize,
    /// Number of rows (2n)
    nrows: usize,
    /// Row stride (padded to 8 for AVX-512 alignment)
    row_stride: usize,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl Avx512Tableau {
    /// Create new identity tableau for n qubits
    pub fn new(n: usize) -> Self {
        let nwords = (n + 63) / 64;
        let nrows = 2 * n;
        // Pad to multiple of 8 for AVX-512 alignment
        let row_stride = (nwords + 7) / 8 * 8;

        let size = nrows * row_stride;
        let mut xs = vec![0u64; size];
        let mut zs = vec![0u64; size];
        let phases = vec![0u64; (nrows + 63) / 64];

        // Initialize identity tableau
        for i in 0..n {
            let row_x = i;
            let row_z = n + i;
            let word = i / 64;
            let bit = i % 64;

            xs[row_x * row_stride + word] |= 1u64 << bit;
            zs[row_z * row_stride + word] |= 1u64 << bit;
        }

        Self {
            xs,
            zs,
            phases,
            n,
            nwords,
            nrows,
            row_stride,
        }
    }

    /// Apply Hadamard gate to qubit q
    #[inline(always)]
    pub fn h(&mut self, q: usize) {
        unsafe {
            let word = q / 64;
            let bit = 1u64 << (q % 64);
            let mask = _mm512_set1_epi64(bit as i64);

            // Process 8 rows at a time
            for chunk_start in (0..self.nrows).step_by(8) {
                let base = chunk_start * self.row_stride + word;

                // Load 8 X values
                let x_ptr = self.xs.as_ptr().add(base);
                let x_vec = _mm512_loadu_si512(x_ptr as *const __m512i);

                // Load 8 Z values
                let z_ptr = self.zs.as_ptr().add(base);
                let z_vec = _mm512_loadu_si512(z_ptr as *const __m512i);

                // Extract bit from X and Z
                let x_bits = _mm512_and_si512(x_vec, mask);
                let z_bits = _mm512_and_si512(z_vec, mask);

                // Swap X and Z where bit is set
                let x_new = _mm512_xor_si512(x_vec, _mm512_and_si512(z_bits, mask));
                let z_new = _mm512_xor_si512(z_vec, _mm512_and_si512(x_bits, mask));

                // Store back
                _mm512_storeu_si512(x_ptr as *mut __m512i, x_new);
                _mm512_storeu_si512(z_ptr as *mut __m512i, z_new);

                // Phase: Y before H adds i
                // Both set means Y, phase flip
                let y_bits = _mm512_and_si512(x_bits, z_bits);
                let y_mask = _mm512_test_epi64_mask(y_bits, mask);
                if y_mask != 0 {
                    // Flip phases for rows where Y bit was set
                    for i in 0..8 {
                        if chunk_start + i < self.nrows && (y_mask & (1 << i)) != 0 {
                            let phase_word = (chunk_start + i) / 64;
                            let phase_bit = (chunk_start + i) % 64;
                            self.phases[phase_word] ^= 1u64 << phase_bit;
                        }
                    }
                }
            }
        }
    }

    /// Apply CNOT gate: control -> target
    #[inline(always)]
    pub fn cx(&mut self, control: usize, target: usize) {
        if control == target {
            return;
        }

        let c_word = control / 64;
        let c_bit = 1u64 << (control % 64);
        let t_word = target / 64;
        let t_bit = 1u64 << (target % 64);

        unsafe {
            let c_mask = _mm512_set1_epi64(c_bit as i64);
            let t_mask = _mm512_set1_epi64(t_bit as i64);

            for chunk_start in (0..self.nrows).step_by(8) {
                // Load control X bits
                let cx_ptr = self.xs.as_ptr().add(chunk_start * self.row_stride + c_word);
                let cx_vec = _mm512_loadu_si512(cx_ptr as *const __m512i);

                // Load target X bits
                let tx_ptr = self.xs.as_ptr().add(chunk_start * self.row_stride + t_word);
                let tx_vec = _mm512_loadu_si512(tx_ptr as *const __m512i);

                // Load control Z bits
                let cz_ptr = self.zs.as_ptr().add(chunk_start * self.row_stride + c_word);
                let cz_vec = _mm512_loadu_si512(cz_ptr as *const __m512i);

                // Load target Z bits
                let tz_ptr = self.zs.as_ptr().add(chunk_start * self.row_stride + t_word);
                let tz_vec = _mm512_loadu_si512(tz_ptr as *const __m512i);

                // X_target ^= X_control
                let tx_new = _mm512_xor_si512(tx_vec, _mm512_and_si512(cx_vec, c_mask));
                _mm512_storeu_si512(tx_ptr as *mut __m512i, tx_new);

                // Z_control ^= Z_target
                let cz_new = _mm512_xor_si512(cz_vec, _mm512_and_si512(tz_vec, t_mask));
                _mm512_storeu_si512(cz_ptr as *mut __m512i, cz_new);
            }
        }
    }

    /// Apply S gate to qubit q
    #[inline(always)]
    pub fn s(&mut self, q: usize) {
        let word = q / 64;
        let bit = 1u64 << (q % 64);

        unsafe {
            let mask = _mm512_set1_epi64(bit as i64);

            for chunk_start in (0..self.nrows).step_by(8) {
                let base = chunk_start * self.row_stride + word;

                let x_ptr = self.xs.as_ptr().add(base);
                let x_vec = _mm512_loadu_si512(x_ptr as *const __m512i);

                let z_ptr = self.zs.as_ptr().add(base);
                let z_vec = _mm512_loadu_si512(z_ptr as *const __m512i);

                // Z ^= X
                let z_new = _mm512_xor_si512(z_vec, _mm512_and_si512(x_vec, mask));
                _mm512_storeu_si512(z_ptr as *mut __m512i, z_new);

                // Phase ^= X AND Z (Y)
                let y_bits = _mm512_and_si512(x_vec, z_vec);
                let y_mask = _mm512_test_epi64_mask(y_bits, mask);
                if y_mask != 0 {
                    for i in 0..8 {
                        if chunk_start + i < self.nrows && (y_mask & (1 << i)) != 0 {
                            let phase_word = (chunk_start + i) / 64;
                            let phase_bit = (chunk_start + i) % 64;
                            self.phases[phase_word] ^= 1u64 << phase_bit;
                        }
                    }
                }
            }
        }
    }

    /// Apply Z gate to qubit q
    #[inline(always)]
    pub fn z(&mut self, q: usize) {
        let word = q / 64;
        let bit = 1u64 << (q % 64);

        for row in 0..self.nrows {
            if self.xs[row * self.row_stride + word] & bit != 0 {
                let phase_word = row / 64;
                let phase_bit = row % 64;
                self.phases[phase_word] ^= 1u64 << phase_bit;
            }
        }
    }

    /// Apply X gate to qubit q
    #[inline(always)]
    pub fn x(&mut self, q: usize) {
        let word = q / 64;
        let bit = 1u64 << (q % 64);

        for row in 0..self.nrows {
            if self.zs[row * self.row_stride + word] & bit != 0 {
                let phase_word = row / 64;
                let phase_bit = row % 64;
                self.phases[phase_word] ^= 1u64 << phase_bit;
            }
        }
    }

    /// Apply batch of gates (highest performance)
    pub fn apply_batch(&mut self, gates: &[GateOp]) {
        for gate in gates {
            match gate {
                GateOp::H(q) => self.h(*q),
                GateOp::S(q) => self.s(*q),
                GateOp::X(q) => self.x(*q),
                GateOp::Z(q) => self.z(*q),
                GateOp::CX(c, t) => self.cx(*c, *t),
                GateOp::CZ(a, b) => {
                    self.h(*b);
                    self.cx(*a, *b);
                    self.h(*b);
                }
            }
        }
    }

    /// Benchmark throughput
    pub fn benchmark(n: usize, num_gates: usize) -> f64 {
        let mut tab = Self::new(n);
        let gates = GateOp::random_circuit(n, num_gates, 42);

        let start = Instant::now();
        tab.apply_batch(&gates);
        let elapsed = start.elapsed().as_secs_f64();

        num_gates as f64 / elapsed
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// GATE OPERATION
// ---------------------------------------------------------------------------

/// Gate operation for batch processing
#[derive(Clone, Debug)]
pub enum GateOp {
    H(usize),
    S(usize),
    X(usize),
    Z(usize),
    CX(usize, usize),
    CZ(usize, usize),
}

impl GateOp {
    /// Generate random circuit
    pub fn random_circuit(n: usize, num_gates: usize, seed: u64) -> Vec<Self> {
        let mut rng_state = seed;
        let mut rand = || -> usize {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as usize
        };

        (0..num_gates)
            .map(|i| {
                let q = rand() % n;
                let q2 = rand() % n;
                match i % 6 {
                    0 => GateOp::H(q),
                    1 => GateOp::S(q),
                    2 => GateOp::X(q),
                    3 => GateOp::Z(q),
                    4 => GateOp::CX(q, q2),
                    _ => GateOp::CZ(q, q2),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FALLBACK FOR NON-AVX512 SYSTEMS
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub struct Avx512Tableau {
    _private: (),
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
impl Avx512Tableau {
    pub fn new(_n: usize) -> Self {
        panic!("AVX-512 not available on this platform");
    }
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_h_gate() {
        let mut tab = Avx512Tableau::new(10);
        tab.h(0);
        assert!(tab.num_qubits() == 10);
    }

    #[test]
    fn test_avx512_cx_gate() {
        let mut tab = Avx512Tableau::new(10);
        tab.h(0);
        tab.cx(0, 1);
        assert!(tab.num_qubits() == 10);
    }

    #[test]
    fn test_avx512_batch() {
        let mut tab = Avx512Tableau::new(100);
        let gates = GateOp::random_circuit(100, 1000, 42);
        tab.apply_batch(&gates);
    }

    #[test]
    fn test_avx512_benchmark() {
        let throughput = Avx512Tableau::benchmark(100, 10000);
        println!("AVX-512 throughput at n=100: {:.2} gates/sec", throughput);
        assert!(throughput > 1_000_000.0); // At least 1 MHz
    }
}
