//! SIMD-accelerated quantum state operations.
//!
//! Provides NEON intrinsics for Apple Silicon (aarch64) with a scalar fallback
//! for other architectures. All functions operate on (a, b) amplitude pairs
//! within a `par_chunks_mut` chunk, where `a = chunk[i]`, `b = chunk[i | stride]`.
//!
//! # Safety
//!
//! The NEON functions use `unsafe` for SIMD intrinsics. They rely on `C64` being
//! `#[repr(C)]` with layout `(re: f64, im: f64)`, which is memory-compatible
//! with `[f64; 2]` for NEON loads/stores.

use crate::C64;

// ===================================================================
// SIMD MATRIX TYPE
// ===================================================================

/// 2×2 complex matrix stored as plain f64 for efficient broadcast into SIMD regs.
#[derive(Clone, Debug)]
pub struct SimdMatrix2x2 {
    pub m00_re: f64,
    pub m00_im: f64,
    pub m01_re: f64,
    pub m01_im: f64,
    pub m10_re: f64,
    pub m10_im: f64,
    pub m11_re: f64,
    pub m11_im: f64,
}

impl SimdMatrix2x2 {
    /// Create from C64 matrix entries.
    #[inline]
    pub fn from_c64(m: &[[C64; 2]; 2]) -> Self {
        SimdMatrix2x2 {
            m00_re: m[0][0].re,
            m00_im: m[0][0].im,
            m01_re: m[0][1].re,
            m01_im: m[0][1].im,
            m10_re: m[1][0].re,
            m10_im: m[1][0].im,
            m11_re: m[1][1].re,
            m11_im: m[1][1].im,
        }
    }
}

// ===================================================================
// NEON (aarch64) IMPLEMENTATION
// ===================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Complex multiply: (m_re + i*m_im) * (x.re + i*x.im)
    /// x is a float64x2_t with [re, im].
    /// Returns float64x2_t with [result_re, result_im].
    #[inline(always)]
    unsafe fn complex_mul_neon(m_re: f64, m_im: f64, x: float64x2_t) -> float64x2_t {
        // x = [x_re, x_im]
        let x_re = vdupq_laneq_f64::<0>(x); // [x_re, x_re]
        let x_im = vdupq_laneq_f64::<1>(x); // [x_im, x_im]

        // coeff = [m_re, m_im]
        let coeff_re_im = vld1q_f64([m_re, m_im].as_ptr());
        // coeff_neg = [-m_im, m_re]
        let coeff_neg_im_re = vld1q_f64([-m_im, m_re].as_ptr());

        // result = x_re * [m_re, m_im] + x_im * [-m_im, m_re]
        //        = [x_re*m_re - x_im*m_im, x_re*m_im + x_im*m_re]
        let t = vmulq_f64(x_re, coeff_re_im);
        vfmaq_f64(t, x_im, coeff_neg_im_re)
    }

    /// Apply a 2×2 unitary matrix to one (a, b) pair using NEON.
    /// `a_ptr` and `b_ptr` point to C64 values (16 bytes each: re, im as f64).
    #[inline(always)]
    pub unsafe fn apply_unitary_pair_neon(a_ptr: *mut C64, b_ptr: *mut C64, m: &SimdMatrix2x2) {
        let a = vld1q_f64(a_ptr as *const f64);
        let b = vld1q_f64(b_ptr as *const f64);

        // new_a = m00 * a + m01 * b
        let t0 = complex_mul_neon(m.m00_re, m.m00_im, a);
        let t1 = complex_mul_neon(m.m01_re, m.m01_im, b);
        let new_a = vaddq_f64(t0, t1);

        // new_b = m10 * a + m11 * b
        let t2 = complex_mul_neon(m.m10_re, m.m10_im, a);
        let t3 = complex_mul_neon(m.m11_re, m.m11_im, b);
        let new_b = vaddq_f64(t2, t3);

        vst1q_f64(a_ptr as *mut f64, new_a);
        vst1q_f64(b_ptr as *mut f64, new_b);
    }

    /// Apply unitary to all pairs in a chunk. Called from par_chunks_mut context.
    /// `chunk` has length `stride * 2`, pairs are `(chunk[i], chunk[i | stride])`.
    #[inline]
    pub unsafe fn apply_unitary_chunk_neon(chunk: &mut [C64], stride: usize, m: &SimdMatrix2x2) {
        let half = chunk.len() / 2;
        let base_ptr = chunk.as_mut_ptr();
        for i in 0..half.min(stride) {
            let a_ptr = base_ptr.add(i);
            let b_ptr = base_ptr.add(i | stride);
            if (i | stride) < chunk.len() {
                apply_unitary_pair_neon(a_ptr, b_ptr, m);
            }
        }
    }

    /// Apply unitary to 2 (a,b) pairs simultaneously using 4 NEON registers.
    /// Doubles throughput when stride >= 2 by interleaving independent pair operations.
    #[inline(always)]
    pub unsafe fn apply_unitary_2pair_neon(
        a0_ptr: *mut C64,
        b0_ptr: *mut C64,
        a1_ptr: *mut C64,
        b1_ptr: *mut C64,
        m: &SimdMatrix2x2,
    ) {
        // Load both pairs simultaneously
        let a0 = vld1q_f64(a0_ptr as *const f64);
        let b0 = vld1q_f64(b0_ptr as *const f64);
        let a1 = vld1q_f64(a1_ptr as *const f64);
        let b1 = vld1q_f64(b1_ptr as *const f64);

        // new_a0 = m00*a0 + m01*b0, new_a1 = m00*a1 + m01*b1
        let t0_a0 = complex_mul_neon(m.m00_re, m.m00_im, a0);
        let t1_a0 = complex_mul_neon(m.m01_re, m.m01_im, b0);
        let t0_a1 = complex_mul_neon(m.m00_re, m.m00_im, a1);
        let t1_a1 = complex_mul_neon(m.m01_re, m.m01_im, b1);
        let new_a0 = vaddq_f64(t0_a0, t1_a0);
        let new_a1 = vaddq_f64(t0_a1, t1_a1);

        // new_b0 = m10*a0 + m11*b0, new_b1 = m10*a1 + m11*b1
        let t2_b0 = complex_mul_neon(m.m10_re, m.m10_im, a0);
        let t3_b0 = complex_mul_neon(m.m11_re, m.m11_im, b0);
        let t2_b1 = complex_mul_neon(m.m10_re, m.m10_im, a1);
        let t3_b1 = complex_mul_neon(m.m11_re, m.m11_im, b1);
        let new_b0 = vaddq_f64(t2_b0, t3_b0);
        let new_b1 = vaddq_f64(t2_b1, t3_b1);

        // Store all 4 results
        vst1q_f64(a0_ptr as *mut f64, new_a0);
        vst1q_f64(b0_ptr as *mut f64, new_b0);
        vst1q_f64(a1_ptr as *mut f64, new_a1);
        vst1q_f64(b1_ptr as *mut f64, new_b1);
    }

    /// Apply unitary to chunk with 2-pair pipeline. Falls back to single-pair for remainder.
    #[inline]
    pub unsafe fn apply_unitary_chunk_2pair_neon(
        chunk: &mut [C64],
        stride: usize,
        m: &SimdMatrix2x2,
    ) {
        let half = chunk.len() / 2;
        let count = half.min(stride);
        let base_ptr = chunk.as_mut_ptr();
        let mut i = 0;

        // Process 2 pairs at a time
        while i + 1 < count {
            let a0_ptr = base_ptr.add(i);
            let b0_ptr = base_ptr.add(i | stride);
            let a1_ptr = base_ptr.add(i + 1);
            let b1_ptr = base_ptr.add((i + 1) | stride);
            if ((i + 1) | stride) < chunk.len() {
                apply_unitary_2pair_neon(a0_ptr, b0_ptr, a1_ptr, b1_ptr, m);
            }
            i += 2;
        }
        // Handle remainder
        if i < count {
            let a_ptr = base_ptr.add(i);
            let b_ptr = base_ptr.add(i | stride);
            if (i | stride) < chunk.len() {
                apply_unitary_pair_neon(a_ptr, b_ptr, m);
            }
        }
    }

    /// NEON vectorized probability computation: sum of |amp|^2 for a slice.
    /// Processes 2 complex numbers per iteration using NEON multiply + pairwise add.
    #[inline]
    pub unsafe fn probability_sum_neon(amps: &[C64]) -> f64 {
        let n = amps.len();
        let ptr = amps.as_ptr() as *const f64;
        let mut acc = vdupq_n_f64(0.0); // [sum_even, sum_odd]
        let mut i = 0;

        // Process 2 complex numbers per iteration (4 f64 values)
        while i + 1 < n {
            let v0 = vld1q_f64(ptr.add(i * 2)); // [re0, im0]
            let v1 = vld1q_f64(ptr.add(i * 2 + 2)); // [re1, im1]
                                                    // |amp|^2 = re^2 + im^2
            let sq0 = vmulq_f64(v0, v0); // [re0^2, im0^2]
            let sq1 = vmulq_f64(v1, v1); // [re1^2, im1^2]
            acc = vaddq_f64(acc, sq0);
            acc = vaddq_f64(acc, sq1);
            i += 2;
        }
        // Handle remainder
        if i < n {
            let v = vld1q_f64(ptr.add(i * 2));
            let sq = vmulq_f64(v, v);
            acc = vaddq_f64(acc, sq);
        }
        // Horizontal sum: acc = [sum_re, sum_im] → total = sum_re + sum_im
        vgetq_lane_f64::<0>(acc) + vgetq_lane_f64::<1>(acc)
    }

    /// Specialized Hadamard: coefficients are real-only (±1/√2), so we skip
    /// imaginary multiplies.
    #[inline]
    pub unsafe fn apply_hadamard_chunk_neon(chunk: &mut [C64], stride: usize, inv_sqrt2: f64) {
        let half = chunk.len() / 2;
        let scale = vdupq_n_f64(inv_sqrt2);
        let base_ptr = chunk.as_mut_ptr();

        for i in 0..half.min(stride) {
            let idx2 = i | stride;
            if idx2 < chunk.len() {
                let a_ptr = base_ptr.add(i) as *mut f64;
                let b_ptr = base_ptr.add(idx2) as *mut f64;

                let a = vld1q_f64(a_ptr);
                let b = vld1q_f64(b_ptr);

                // new_a = (a + b) * inv_sqrt2
                let sum = vaddq_f64(a, b);
                let new_a = vmulq_f64(sum, scale);

                // new_b = (a - b) * inv_sqrt2
                let diff = vsubq_f64(a, b);
                let new_b = vmulq_f64(diff, scale);

                vst1q_f64(a_ptr, new_a);
                vst1q_f64(b_ptr, new_b);
            }
        }
    }

    /// Diagonal gate (Z/S/T/Rz): only modify |1⟩ component.
    /// phase = (phase_re + i*phase_im), applied to entries where qubit bit = 1.
    #[inline]
    pub unsafe fn apply_diagonal_chunk_neon(
        chunk: &mut [C64],
        stride: usize,
        phase_re: f64,
        phase_im: f64,
    ) {
        let base_ptr = chunk.as_mut_ptr();
        let half = chunk.len() / 2;

        for i in 0..half.min(stride) {
            let idx2 = i | stride;
            if idx2 < chunk.len() {
                let b_ptr = base_ptr.add(idx2) as *mut f64;
                let b = vld1q_f64(b_ptr);
                let result = complex_mul_neon(phase_re, phase_im, b);
                vst1q_f64(b_ptr, result);
            }
        }
    }

    /// Full diagonal gate (Rz): apply DIFFERENT phases to |0⟩ and |1⟩ components.
    /// Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
    /// phase0 applies to |0⟩ entries (chunk[i] where i < stride)
    /// phase1 applies to |1⟩ entries (chunk[i | stride] where i < stride)
    /// Uses only 6 NEON ops per pair vs 14 for general unitary.
    #[inline]
    pub unsafe fn apply_full_diagonal_chunk_neon(
        chunk: &mut [C64],
        stride: usize,
        phase0_re: f64,
        phase0_im: f64,
        phase1_re: f64,
        phase1_im: f64,
    ) {
        let base_ptr = chunk.as_mut_ptr();
        let half = chunk.len() / 2;

        for i in 0..half.min(stride) {
            let idx2 = i | stride;
            if idx2 < chunk.len() {
                // Apply phase0 to |0⟩ component (chunk[i])
                let a_ptr = base_ptr.add(i) as *mut f64;
                let a = vld1q_f64(a_ptr);
                let result0 = complex_mul_neon(phase0_re, phase0_im, a);
                vst1q_f64(a_ptr, result0);

                // Apply phase1 to |1⟩ component (chunk[idx2])
                let b_ptr = base_ptr.add(idx2) as *mut f64;
                let b = vld1q_f64(b_ptr);
                let result1 = complex_mul_neon(phase1_re, phase1_im, b);
                vst1q_f64(b_ptr, result1);
            }
        }
    }
}

// ===================================================================
// SCALAR FALLBACK (any architecture)
// ===================================================================

/// Scalar fallback for the general 2×2 unitary.
#[inline]
pub fn apply_unitary_chunk_scalar(chunk: &mut [C64], stride: usize, m: &SimdMatrix2x2) {
    let half = chunk.len() / 2;
    for i in 0..half.min(stride) {
        let idx2 = i | stride;
        if idx2 < chunk.len() {
            let a = chunk[i];
            let b = chunk[idx2];

            chunk[i] = C64 {
                re: m.m00_re * a.re - m.m00_im * a.im + m.m01_re * b.re - m.m01_im * b.im,
                im: m.m00_re * a.im + m.m00_im * a.re + m.m01_re * b.im + m.m01_im * b.re,
            };
            chunk[idx2] = C64 {
                re: m.m10_re * a.re - m.m10_im * a.im + m.m11_re * b.re - m.m11_im * b.im,
                im: m.m10_re * a.im + m.m10_im * a.re + m.m11_re * b.im + m.m11_im * b.re,
            };
        }
    }
}

/// Scalar fallback for Hadamard.
#[inline]
pub fn apply_hadamard_chunk_scalar(chunk: &mut [C64], stride: usize, inv_sqrt2: f64) {
    let half = chunk.len() / 2;
    for i in 0..half.min(stride) {
        let idx2 = i | stride;
        if idx2 < chunk.len() {
            let a = chunk[i];
            let b = chunk[idx2];

            chunk[i] = C64 {
                re: (a.re + b.re) * inv_sqrt2,
                im: (a.im + b.im) * inv_sqrt2,
            };
            chunk[idx2] = C64 {
                re: (a.re - b.re) * inv_sqrt2,
                im: (a.im - b.im) * inv_sqrt2,
            };
        }
    }
}

/// Scalar fallback for diagonal gate.
#[inline]
pub fn apply_diagonal_chunk_scalar(chunk: &mut [C64], stride: usize, phase_re: f64, phase_im: f64) {
    let half = chunk.len() / 2;
    for i in 0..half.min(stride) {
        let idx2 = i | stride;
        if idx2 < chunk.len() {
            let b = chunk[idx2];
            chunk[idx2] = C64 {
                re: phase_re * b.re - phase_im * b.im,
                im: phase_re * b.im + phase_im * b.re,
            };
        }
    }
}

/// Scalar fallback for full diagonal gate (Rz).
#[inline]
pub fn apply_full_diagonal_chunk_scalar(
    chunk: &mut [C64],
    stride: usize,
    phase0_re: f64,
    phase0_im: f64,
    phase1_re: f64,
    phase1_im: f64,
) {
    let half = chunk.len() / 2;
    for i in 0..half.min(stride) {
        let idx2 = i | stride;
        if idx2 < chunk.len() {
            // Apply phase0 to |0⟩ component
            let a = chunk[i];
            chunk[i] = C64 {
                re: phase0_re * a.re - phase0_im * a.im,
                im: phase0_re * a.im + phase0_im * a.re,
            };

            // Apply phase1 to |1⟩ component
            let b = chunk[idx2];
            chunk[idx2] = C64 {
                re: phase1_re * b.re - phase1_im * b.im,
                im: phase1_re * b.im + phase1_im * b.re,
            };
        }
    }
}

// ===================================================================
// PUBLIC DISPATCH FUNCTIONS
// ===================================================================

/// Apply a 2×2 unitary to all pairs in a chunk. Dispatches to NEON on aarch64.
#[inline]
pub fn apply_unitary_chunk(chunk: &mut [C64], stride: usize, m: &SimdMatrix2x2) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon::apply_unitary_chunk_neon(chunk, stride, m);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_unitary_chunk_scalar(chunk, stride, m);
    }
}

/// Apply unitary with 2-pair NEON pipeline. 2x throughput over single-pair.
#[inline]
pub fn apply_unitary_chunk_2pair(chunk: &mut [C64], stride: usize, m: &SimdMatrix2x2) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon::apply_unitary_chunk_2pair_neon(chunk, stride, m);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_unitary_chunk_scalar(chunk, stride, m);
    }
}

/// Compute sum of |amp|^2 for a slice of complex amplitudes. NEON-accelerated on aarch64.
#[inline]
pub fn probability_sum(amps: &[C64]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return neon::probability_sum_neon(amps);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        amps.iter().map(|a| a.re * a.re + a.im * a.im).sum()
    }
}

/// Apply Hadamard to all pairs in a chunk. Dispatches to NEON on aarch64.
#[inline]
pub fn apply_hadamard_chunk(chunk: &mut [C64], stride: usize, inv_sqrt2: f64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon::apply_hadamard_chunk_neon(chunk, stride, inv_sqrt2);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_hadamard_chunk_scalar(chunk, stride, inv_sqrt2);
    }
}

/// Apply diagonal phase to all |1⟩ entries in a chunk. Dispatches to NEON on aarch64.
#[inline]
pub fn apply_diagonal_chunk(chunk: &mut [C64], stride: usize, phase_re: f64, phase_im: f64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon::apply_diagonal_chunk_neon(chunk, stride, phase_re, phase_im);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_diagonal_chunk_scalar(chunk, stride, phase_re, phase_im);
    }
}

/// Apply full diagonal matrix with different phases to |0⟩ and |1⟩ entries.
/// Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]. Dispatches to NEON on aarch64.
#[inline]
pub fn apply_full_diagonal_chunk(
    chunk: &mut [C64],
    stride: usize,
    phase0_re: f64,
    phase0_im: f64,
    phase1_re: f64,
    phase1_im: f64,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon::apply_full_diagonal_chunk_neon(
            chunk, stride, phase0_re, phase0_im, phase1_re, phase1_im,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_full_diagonal_chunk_scalar(chunk, stride, phase0_re, phase0_im, phase1_re, phase1_im);
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    #[test]
    fn test_unitary_scalar_identity() {
        let id = SimdMatrix2x2 {
            m00_re: 1.0,
            m00_im: 0.0,
            m01_re: 0.0,
            m01_im: 0.0,
            m10_re: 0.0,
            m10_im: 0.0,
            m11_re: 1.0,
            m11_im: 0.0,
        };
        let mut chunk = vec![c(0.6, 0.1), c(0.3, -0.2)];
        let orig = chunk.clone();
        apply_unitary_chunk_scalar(&mut chunk, 1, &id);
        for i in 0..2 {
            assert!((chunk[i].re - orig[i].re).abs() < 1e-15);
            assert!((chunk[i].im - orig[i].im).abs() < 1e-15);
        }
    }

    #[test]
    fn test_hadamard_scalar() {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut chunk = vec![c(1.0, 0.0), c(0.0, 0.0)];
        apply_hadamard_chunk_scalar(&mut chunk, 1, inv_sqrt2);
        assert!((chunk[0].re - inv_sqrt2).abs() < 1e-15);
        assert!((chunk[1].re - inv_sqrt2).abs() < 1e-15);
    }

    #[test]
    fn test_diagonal_scalar() {
        // Apply i (phase = 0 + 1i) to |1⟩
        let mut chunk = vec![c(1.0, 0.0), c(0.5, 0.3)];
        apply_diagonal_chunk_scalar(&mut chunk, 1, 0.0, 1.0);
        // |0⟩ unchanged
        assert!((chunk[0].re - 1.0).abs() < 1e-15);
        // |1⟩: i * (0.5 + 0.3i) = -0.3 + 0.5i
        assert!((chunk[1].re - (-0.3)).abs() < 1e-15);
        assert!((chunk[1].im - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_neon_matches_scalar() {
        // This test validates that the dispatched version matches scalar
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        // Hadamard test
        let mut chunk_scalar = vec![c(0.7, 0.1), c(0.3, -0.4)];
        let mut chunk_dispatch = chunk_scalar.clone();
        apply_hadamard_chunk_scalar(&mut chunk_scalar, 1, inv_sqrt2);
        apply_hadamard_chunk(&mut chunk_dispatch, 1, inv_sqrt2);
        for i in 0..2 {
            assert!(
                (chunk_scalar[i].re - chunk_dispatch[i].re).abs() < 1e-12,
                "Hadamard mismatch at [{}].re",
                i
            );
            assert!(
                (chunk_scalar[i].im - chunk_dispatch[i].im).abs() < 1e-12,
                "Hadamard mismatch at [{}].im",
                i
            );
        }

        // General unitary test (Rx(0.5) matrix)
        let theta: f64 = 0.5;
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        let m = SimdMatrix2x2 {
            m00_re: cos,
            m00_im: 0.0,
            m01_re: 0.0,
            m01_im: -sin,
            m10_re: 0.0,
            m10_im: -sin,
            m11_re: cos,
            m11_im: 0.0,
        };

        let mut chunk_s = vec![c(0.6, 0.2), c(0.4, -0.1)];
        let mut chunk_d = chunk_s.clone();
        apply_unitary_chunk_scalar(&mut chunk_s, 1, &m);
        apply_unitary_chunk(&mut chunk_d, 1, &m);
        for i in 0..2 {
            assert!(
                (chunk_s[i].re - chunk_d[i].re).abs() < 1e-12,
                "Unitary mismatch at [{}].re",
                i
            );
            assert!(
                (chunk_s[i].im - chunk_d[i].im).abs() < 1e-12,
                "Unitary mismatch at [{}].im",
                i
            );
        }
    }

    #[test]
    fn test_larger_chunk_unitary() {
        // Test with stride=2, chunk size=4
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let m = SimdMatrix2x2 {
            m00_re: inv_sqrt2,
            m00_im: 0.0,
            m01_re: inv_sqrt2,
            m01_im: 0.0,
            m10_re: inv_sqrt2,
            m10_im: 0.0,
            m11_re: -inv_sqrt2,
            m11_im: 0.0,
        };

        let mut chunk = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
        apply_unitary_chunk(&mut chunk, 2, &m);
        // After H on qubit 1 of a 2-qubit |00⟩:
        // pairs: (chunk[0], chunk[2]) and (chunk[1], chunk[3])
        assert!((chunk[0].re - inv_sqrt2).abs() < 1e-12);
        assert!((chunk[2].re - inv_sqrt2).abs() < 1e-12);
    }

    #[test]
    fn test_2pair_matches_single_pair() {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let m = SimdMatrix2x2 {
            m00_re: inv_sqrt2,
            m00_im: 0.0,
            m01_re: inv_sqrt2,
            m01_im: 0.0,
            m10_re: inv_sqrt2,
            m10_im: 0.0,
            m11_re: -inv_sqrt2,
            m11_im: 0.0,
        };

        // stride=2, 4 elements: pairs are (0,2) and (1,3)
        let mut chunk_single = vec![c(0.8, 0.1), c(0.3, -0.2), c(0.5, 0.0), c(-0.1, 0.4)];
        let mut chunk_2pair = chunk_single.clone();

        apply_unitary_chunk(&mut chunk_single, 2, &m);
        apply_unitary_chunk_2pair(&mut chunk_2pair, 2, &m);

        for i in 0..4 {
            assert!(
                (chunk_single[i].re - chunk_2pair[i].re).abs() < 1e-12,
                "2-pair mismatch at [{}].re: {} vs {}",
                i,
                chunk_single[i].re,
                chunk_2pair[i].re
            );
            assert!(
                (chunk_single[i].im - chunk_2pair[i].im).abs() < 1e-12,
                "2-pair mismatch at [{}].im",
                i
            );
        }
    }

    #[test]
    fn test_probability_sum() {
        let amps = vec![c(0.5, 0.0), c(0.0, 0.5), c(0.5, 0.0), c(0.0, 0.5)];
        let prob = probability_sum(&amps);
        // 0.25 + 0.25 + 0.25 + 0.25 = 1.0
        assert!((prob - 1.0).abs() < 1e-12, "probability sum: {}", prob);
    }

    #[test]
    fn test_probability_sum_single() {
        let amps = vec![c(1.0, 0.0)];
        let prob = probability_sum(&amps);
        assert!((prob - 1.0).abs() < 1e-12);
    }
}
