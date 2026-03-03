//! Float32 Quantum State and Gate Operations
//!
//! Provides a single-precision quantum simulator for ~4x throughput improvement
//! over f64 on ARM NEON (2 complex per 128-bit register instead of 1).
//! Memory footprint is halved, enabling +1 qubit at the same RAM budget.
//!
//! # Precision
//!
//! f32 gives ~7 decimal digits of precision. For circuits <100 gates on <25 qubits,
//! fidelity loss is typically <1e-5. For research requiring high precision, use the
//! standard f64 `QuantumState`.

use crate::{c32_one, c32_zero, QuantumState, C32, C64};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ===================================================================
// QUANTUM STATE (f32)
// ===================================================================

/// Single-precision quantum state vector.
///
/// Stores 2^n amplitudes as `C32` (8 bytes each, vs 16 for C64).
/// At n=24: 128MB instead of 256MB. At n=25: 256MB instead of 512MB.
pub struct QuantumStateF32 {
    amplitudes: Vec<C32>,
    pub num_qubits: usize,
    pub dim: usize,
}

impl QuantumStateF32 {
    /// Create a new f32 quantum state in |0...0⟩.
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amplitudes = vec![c32_zero(); dim];
        amplitudes[0] = c32_one();
        QuantumStateF32 {
            amplitudes,
            num_qubits,
            dim,
        }
    }

    /// Convert from f64 state (lossy downcast).
    pub fn from_f64(state: &QuantumState) -> Self {
        let dim = state.dim;
        let mut amplitudes = Vec::with_capacity(dim);
        for i in 0..dim {
            let c = state.get(i);
            amplitudes.push(C32 {
                re: c.re as f32,
                im: c.im as f32,
            });
        }
        QuantumStateF32 {
            amplitudes,
            num_qubits: state.num_qubits,
            dim,
        }
    }

    /// Convert to f64 state (lossless upcast).
    pub fn to_f64(&self) -> QuantumState {
        let mut state = QuantumState::new(self.num_qubits);
        let amps = state.amplitudes_mut();
        for i in 0..self.dim {
            amps[i] = C64 {
                re: self.amplitudes[i].re as f64,
                im: self.amplitudes[i].im as f64,
            };
        }
        state
    }

    #[inline]
    pub fn get(&self, idx: usize) -> C32 {
        self.amplitudes[idx]
    }

    #[inline]
    pub fn amplitudes_mut(&mut self) -> &mut [C32] {
        &mut self.amplitudes
    }

    #[inline]
    pub fn amplitudes_ref(&self) -> &[C32] {
        &self.amplitudes
    }

    /// Compute fidelity with an f64 state (upcast self to f64 for comparison).
    pub fn fidelity_vs_f64(&self, other: &QuantumState) -> f64 {
        if self.dim != other.dim {
            return 0.0;
        }
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for i in 0..self.dim {
            let a = self.amplitudes[i];
            let b = other.get(i);
            // ⟨a|b⟩ = a* · b
            re += (a.re as f64) * b.re + (a.im as f64) * b.im;
            im += (a.re as f64) * b.im - (a.im as f64) * b.re;
        }
        re * re + im * im
    }

    /// Compute fidelity with another f32 state.
    pub fn fidelity(&self, other: &QuantumStateF32) -> f64 {
        if self.dim != other.dim {
            return 0.0;
        }
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for i in 0..self.dim {
            let a = self.amplitudes[i];
            let b = other.amplitudes[i];
            re += (a.re as f64) * (b.re as f64) + (a.im as f64) * (b.im as f64);
            im += (a.re as f64) * (b.im as f64) - (a.im as f64) * (b.re as f64);
        }
        re * re + im * im
    }

    /// Probabilities (squared magnitudes).
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(|a| (a.re as f64).powi(2) + (a.im as f64).powi(2))
            .collect()
    }
}

// ===================================================================
// F32 SIMD MATRIX TYPE
// ===================================================================

/// 2x2 complex matrix with f32 entries for SIMD broadcast.
#[derive(Clone, Debug)]
pub struct SimdMatrix2x2F32 {
    pub m00_re: f32,
    pub m00_im: f32,
    pub m01_re: f32,
    pub m01_im: f32,
    pub m10_re: f32,
    pub m10_im: f32,
    pub m11_re: f32,
    pub m11_im: f32,
}

// ===================================================================
// GATE OPERATIONS (f32)
// ===================================================================

/// Float32 gate operations. Mirrors GateOperations but for QuantumStateF32.
pub struct GateOpsF32;

impl GateOpsF32 {
    /// Hadamard gate with f32 NEON SIMD.
    pub fn h(state: &mut QuantumStateF32, qubit: usize) {
        let stride = 1 << qubit;
        let inv_sqrt2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                apply_hadamard_chunk_f32(chunk, stride, inv_sqrt2);
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                apply_hadamard_chunk_f32(chunk, stride, inv_sqrt2);
            }
        }
    }

    /// General 2x2 unitary gate.
    pub fn u(state: &mut QuantumStateF32, qubit: usize, matrix: &[[f32; 2]; 4]) {
        let stride = 1 << qubit;
        let m = SimdMatrix2x2F32 {
            m00_re: matrix[0][0],
            m00_im: matrix[0][1],
            m01_re: matrix[1][0],
            m01_im: matrix[1][1],
            m10_re: matrix[2][0],
            m10_im: matrix[2][1],
            m11_re: matrix[3][0],
            m11_im: matrix[3][1],
        };
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                apply_unitary_chunk_f32(chunk, stride, &m);
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                apply_unitary_chunk_f32(chunk, stride, &m);
            }
        }
    }

    /// CNOT gate (control-target swap of |1x⟩ amplitudes).
    pub fn cnot(state: &mut QuantumStateF32, control: usize, target: usize) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            let (bit0, bit1) = if control < target {
                (control, target)
            } else {
                (target, control)
            };
            let num_pairs = dim / 4;
            let ptr = amplitudes.as_mut_ptr();

            unsafe {
                let raw = ptr as usize;
                (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                    let base = crate::insert_zero_bits(pair_idx, bit0, bit1);
                    let i = base | control_mask;
                    let j = i | target_mask;
                    let p = raw as *mut C32;
                    std::ptr::swap(p.add(i), p.add(j));
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if i & control_mask != 0 {
                    let j = i ^ target_mask;
                    if i < j {
                        amplitudes.swap(i, j);
                    }
                }
            }
        }
    }

    /// General 2-qubit unitary (4x4 matrix).
    pub fn u2(state: &mut QuantumStateF32, q0: usize, q1: usize, matrix: &[[C32; 4]; 4]) {
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let dim = state.dim;
        let num_groups = dim / 4;
        let mask_lo = 1usize << lo;
        let mask_hi = 1usize << hi;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            let ptr = amplitudes.as_mut_ptr();
            unsafe {
                let raw = ptr as usize;
                (0..num_groups).into_par_iter().for_each(|group_idx| {
                    let base = crate::insert_zero_bits(group_idx, lo, hi);
                    let i00 = base;
                    let i01 = base | mask_lo;
                    let i10 = base | mask_hi;
                    let i11 = base | mask_lo | mask_hi;

                    let p = raw as *mut C32;
                    let a00 = *p.add(i00);
                    let a01 = *p.add(i01);
                    let a10 = *p.add(i10);
                    let a11 = *p.add(i11);

                    let vals = [a00, a01, a10, a11];
                    for (row, idx) in [(0, i00), (1, i01), (2, i10), (3, i11)] {
                        let mut re = 0.0f32;
                        let mut im = 0.0f32;
                        for col in 0..4 {
                            let m = matrix[row][col];
                            let v = vals[col];
                            re += m.re * v.re - m.im * v.im;
                            im += m.re * v.im + m.im * v.re;
                        }
                        *p.add(idx) = C32 { re, im };
                    }
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for group_idx in 0..num_groups {
                let base = crate::insert_zero_bits(group_idx, lo, hi);
                let i00 = base;
                let i01 = base | mask_lo;
                let i10 = base | mask_hi;
                let i11 = base | mask_lo | mask_hi;

                let vals = [
                    amplitudes[i00],
                    amplitudes[i01],
                    amplitudes[i10],
                    amplitudes[i11],
                ];
                for (row, idx) in [(0, i00), (1, i01), (2, i10), (3, i11)] {
                    let mut re = 0.0f32;
                    let mut im = 0.0f32;
                    for col in 0..4 {
                        let m = matrix[row][col];
                        let v = vals[col];
                        re += m.re * v.re - m.im * v.im;
                        im += m.re * v.im + m.im * v.re;
                    }
                    amplitudes[idx] = C32 { re, im };
                }
            }
        }
    }

    /// X gate (Pauli-X / NOT).
    pub fn x(state: &mut QuantumStateF32, qubit: usize) {
        let stride = 1 << qubit;
        let amplitudes = state.amplitudes_mut();
        for chunk in amplitudes.chunks_mut(stride * 2) {
            for i in 0..stride.min(chunk.len() / 2) {
                chunk.swap(i, i + stride);
            }
        }
    }

    /// Z gate (Pauli-Z).
    pub fn z(state: &mut QuantumStateF32, qubit: usize) {
        let mask = 1 << qubit;
        let amplitudes = state.amplitudes_mut();
        for (i, a) in amplitudes.iter_mut().enumerate() {
            if i & mask != 0 {
                a.re = -a.re;
                a.im = -a.im;
            }
        }
    }

    /// S gate (√Z).
    pub fn s(state: &mut QuantumStateF32, qubit: usize) {
        let mask = 1 << qubit;
        let amplitudes = state.amplitudes_mut();
        for (i, a) in amplitudes.iter_mut().enumerate() {
            if i & mask != 0 {
                // Multiply by i: (re, im) -> (-im, re)
                let tmp = a.re;
                a.re = -a.im;
                a.im = tmp;
            }
        }
    }

    /// T gate (π/8).
    pub fn t(state: &mut QuantumStateF32, qubit: usize) {
        let mask = 1 << qubit;
        let phase_re: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let phase_im: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let amplitudes = state.amplitudes_mut();
        for (i, a) in amplitudes.iter_mut().enumerate() {
            if i & mask != 0 {
                let re = a.re * phase_re - a.im * phase_im;
                let im = a.re * phase_im + a.im * phase_re;
                a.re = re;
                a.im = im;
            }
        }
    }

    /// Rotation around X-axis: Rx(θ) - with adaptive stride parallelism
    pub fn rx(state: &mut QuantumStateF32, qubit: usize, theta: f64) {
        let stride = 1 << qubit;
        let cos_half: f32 = (theta / 2.0).cos() as f32;
        let sin_half: f32 = (theta / 2.0).sin() as f32;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            const HIGH_STRIDE_THRESHOLD: usize = 4096;
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    apply_rx_chunk_f32(chunk, stride, cos_half, sin_half);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = crate::insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C32;
                        let a = *p.add(i);
                        let b = *p.add(j);
                        *p.add(i) = C32 {
                            re: a.re * cos_half + a.im * sin_half,
                            im: a.im * cos_half - a.re * sin_half,
                        };
                        *p.add(j) = C32 {
                            re: b.re * cos_half - b.im * sin_half,
                            im: b.im * cos_half + b.re * sin_half,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                apply_rx_chunk_f32(chunk, stride, cos_half, sin_half);
            }
        }
    }

    /// Rotation around Y-axis: Ry(θ) - with adaptive stride parallelism
    pub fn ry(state: &mut QuantumStateF32, qubit: usize, theta: f64) {
        let stride = 1 << qubit;
        let cos_half: f32 = (theta / 2.0).cos() as f32;
        let sin_half: f32 = (theta / 2.0).sin() as f32;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            const HIGH_STRIDE_THRESHOLD: usize = 4096;
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    apply_ry_chunk_f32(chunk, stride, cos_half, sin_half);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = crate::insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C32;
                        let a = *p.add(i);
                        let b = *p.add(j);
                        *p.add(i) = C32 {
                            re: a.re * cos_half - b.re * sin_half,
                            im: a.im * cos_half - b.im * sin_half,
                        };
                        *p.add(j) = C32 {
                            re: a.re * sin_half + b.re * cos_half,
                            im: a.im * sin_half + b.im * cos_half,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                apply_ry_chunk_f32(chunk, stride, cos_half, sin_half);
            }
        }
    }

    /// Rotation around Z-axis: Rz(θ) - full diagonal with adaptive stride parallelism
    pub fn rz(state: &mut QuantumStateF32, qubit: usize, theta: f64) {
        let stride = 1 << qubit;
        let cos_half: f32 = (theta / 2.0).cos() as f32;
        let sin_half: f32 = (theta / 2.0).sin() as f32;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // Rz is fully diagonal: phase0 on |0⟩, phase1 on |1⟩
        let phase0_re = cos_half;
        let phase0_im = -sin_half; // exp(-iθ/2)
        let phase1_re = cos_half;
        let phase1_im = sin_half; // exp(iθ/2)

        #[cfg(feature = "parallel")]
        {
            const HIGH_STRIDE_THRESHOLD: usize = 4096;
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    apply_full_diagonal_chunk_f32(
                        chunk, stride, phase0_re, phase0_im, phase1_re, phase1_im,
                    );
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = crate::insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C32;

                        // Apply phase0 to |0⟩ component
                        let a = *p.add(i);
                        *p.add(i) = C32 {
                            re: phase0_re * a.re - phase0_im * a.im,
                            im: phase0_re * a.im + phase0_im * a.re,
                        };

                        // Apply phase1 to |1⟩ component
                        let b = *p.add(j);
                        *p.add(j) = C32 {
                            re: phase1_re * b.re - phase1_im * b.im,
                            im: phase1_re * b.im + phase1_im * b.re,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                apply_full_diagonal_chunk_f32(
                    chunk, stride, phase0_re, phase0_im, phase1_re, phase1_im,
                );
            }
        }
    }

    /// Y gate (Pauli-Y) - with parallelism.
    pub fn y(state: &mut QuantumStateF32, qubit: usize) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            const HIGH_STRIDE_THRESHOLD: usize = 4096;
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    let n = stride.min(chunk.len() / 2);
                    for i in 0..n {
                        let idx_b = i + stride;
                        if idx_b < chunk.len() {
                            let a = chunk[i];
                            let b = chunk[idx_b];
                            // Y = i*X*Z†, effectively: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
                            chunk[i] = C32 {
                                re: b.im,
                                im: -b.re,
                            };
                            chunk[idx_b] = C32 {
                                re: -a.im,
                                im: a.re,
                            };
                        }
                    }
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = crate::insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C32;
                        let a = *p.add(i);
                        let b = *p.add(j);
                        *p.add(i) = C32 {
                            re: b.im,
                            im: -b.re,
                        };
                        *p.add(j) = C32 {
                            re: -a.im,
                            im: a.re,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                let n = stride.min(chunk.len() / 2);
                for i in 0..n {
                    let idx_b = i + stride;
                    if idx_b < chunk.len() {
                        let a = chunk[i];
                        let b = chunk[idx_b];
                        // Y = i*X*Z†, effectively: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
                        chunk[i] = C32 {
                            re: b.im,
                            im: -b.re,
                        };
                        chunk[idx_b] = C32 {
                            re: -a.im,
                            im: a.re,
                        };
                    }
                }
            }
        }
    }

    // ===================================================================
    // FUSION SUPPORT FOR F32
    // ===================================================================

    /// Execute a fused single-qubit gate on f32 state.
    /// `matrix_data` is the 2x2 matrix in [[C64; 2]; 2] format (from gate_fusion::Matrix2x2).
    pub fn execute_fused_single_qubit(
        state: &mut QuantumStateF32,
        qubit: usize,
        matrix_data: &[[C64; 2]; 2],
    ) {
        let stride = 1 << qubit;
        // Convert Matrix2x2 format to SimdMatrix2x2F32
        let m = SimdMatrix2x2F32 {
            m00_re: matrix_data[0][0].re as f32,
            m00_im: matrix_data[0][0].im as f32,
            m01_re: matrix_data[0][1].re as f32,
            m01_im: matrix_data[0][1].im as f32,
            m10_re: matrix_data[1][0].re as f32,
            m10_im: matrix_data[1][0].im as f32,
            m11_re: matrix_data[1][1].re as f32,
            m11_im: matrix_data[1][1].im as f32,
        };
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            const HIGH_STRIDE_THRESHOLD: usize = 4096;
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    apply_unitary_chunk_f32(chunk, stride, &m);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = crate::insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C32;
                        let a = *p.add(i);
                        let b = *p.add(j);

                        *p.add(i) = C32 {
                            re: m.m00_re * a.re - m.m00_im * a.im + m.m01_re * b.re
                                - m.m01_im * b.im,
                            im: m.m00_re * a.im
                                + m.m00_im * a.re
                                + m.m01_re * b.im
                                + m.m01_im * b.re,
                        };
                        *p.add(j) = C32 {
                            re: m.m10_re * a.re - m.m10_im * a.im + m.m11_re * b.re
                                - m.m11_im * b.im,
                            im: m.m10_re * a.im
                                + m.m10_im * a.re
                                + m.m11_re * b.im
                                + m.m11_im * b.re,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                apply_unitary_chunk_f32(chunk, stride, &m);
            }
        }
    }

    /// Execute a fused two-qubit gate on f32 state.
    /// `matrix` is the accumulated 4x4 unitary matrix.
    pub fn execute_fused_two_qubit(
        state: &mut QuantumStateF32,
        q0: usize,
        q1: usize,
        matrix: &[[C64; 4]; 4],
    ) {
        // Convert f64 matrix to f32 and execute
        let matrix_f32: [[C32; 4]; 4] = matrix.map(|row| {
            row.map(|c| C32 {
                re: c.re as f32,
                im: c.im as f32,
            })
        });
        Self::u2(state, q0, q1, &matrix_f32);
    }
}

// ===================================================================
// F32 NEON SIMD
// ===================================================================

#[cfg(target_arch = "aarch64")]
mod neon_f32 {
    use super::*;
    use std::arch::aarch64::*;

    /// Apply Hadamard to all pairs in an f32 chunk using NEON float32x4_t.
    /// Processes 2 pairs per iteration (4 complex f32 = 2 pairs).
    #[inline]
    pub unsafe fn apply_hadamard_chunk_neon_f32(chunk: &mut [C32], stride: usize, inv_sqrt2: f32) {
        let scale = vdupq_n_f32(inv_sqrt2);
        let base_ptr = chunk.as_mut_ptr() as *mut f32;
        let n = stride.min(chunk.len() / 2);

        // Process 2 pairs at a time (each C32 is 2xf32, so 2 pairs = 8 f32 = 2 float32x4_t)
        let pairs_of_2 = n / 2;
        let mut i = 0usize;
        while i < pairs_of_2 * 2 {
            let idx_a = i;
            let idx_b = i + stride;
            if idx_b + 1 < chunk.len() {
                // Load 2 consecutive |0⟩ amplitudes: [a0_re, a0_im, a1_re, a1_im]
                let a = vld1q_f32(base_ptr.add(idx_a * 2));
                // Load 2 consecutive |1⟩ amplitudes: [b0_re, b0_im, b1_re, b1_im]
                let b = vld1q_f32(base_ptr.add(idx_b * 2));

                let sum = vaddq_f32(a, b);
                let diff = vsubq_f32(a, b);
                let new_a = vmulq_f32(sum, scale);
                let new_b = vmulq_f32(diff, scale);

                vst1q_f32(base_ptr.add(idx_a * 2), new_a);
                vst1q_f32(base_ptr.add(idx_b * 2), new_b);
            }
            i += 2;
        }

        // Handle remaining odd pair
        if n % 2 == 1 {
            let idx_a = n - 1;
            let idx_b = idx_a + stride;
            if idx_b < chunk.len() {
                let a_re = chunk[idx_a].re;
                let a_im = chunk[idx_a].im;
                let b_re = chunk[idx_b].re;
                let b_im = chunk[idx_b].im;
                chunk[idx_a] = C32 {
                    re: (a_re + b_re) * inv_sqrt2,
                    im: (a_im + b_im) * inv_sqrt2,
                };
                chunk[idx_b] = C32 {
                    re: (a_re - b_re) * inv_sqrt2,
                    im: (a_im - b_im) * inv_sqrt2,
                };
            }
        }
    }

    /// Complex multiply for f32: returns (re, im) of (m_re + i*m_im) * (x_re + i*x_im)
    #[inline(always)]
    unsafe fn complex_mul_neon_f32_pair(m_re: f32, m_im: f32, x_re: f32, x_im: f32) -> (f32, f32) {
        let re = m_re * x_re - m_im * x_im;
        let im = m_re * x_im + m_im * x_re;
        (re, im)
    }

    /// Apply a general 2x2 unitary to all pairs in an f32 chunk using NEON.
    /// Processes 2 pairs at a time using float32x4_t (4 f32 = 2 complex numbers).
    #[inline]
    pub unsafe fn apply_unitary_chunk_neon_f32(
        chunk: &mut [C32],
        stride: usize,
        m: &SimdMatrix2x2F32,
    ) {
        let base_ptr = chunk.as_mut_ptr() as *mut f32;
        let n = stride.min(chunk.len() / 2);

        // Process 2 pairs at a time (4 complex f32 = 2 pairs per iteration)
        let pairs_of_2 = n / 2;
        let mut i = 0usize;

        while i < pairs_of_2 * 2 {
            let idx_a0 = i;
            let idx_a1 = i + 1;
            let idx_b0 = i + stride;
            let idx_b1 = i + 1 + stride;

            if idx_b1 >= chunk.len() {
                break;
            }

            // Load amplitudes directly
            let a0_re = *base_ptr.add(idx_a0 * 2);
            let a0_im = *base_ptr.add(idx_a0 * 2 + 1);
            let a1_re = *base_ptr.add(idx_a1 * 2);
            let a1_im = *base_ptr.add(idx_a1 * 2 + 1);
            let b0_re = *base_ptr.add(idx_b0 * 2);
            let b0_im = *base_ptr.add(idx_b0 * 2 + 1);
            let b1_re = *base_ptr.add(idx_b1 * 2);
            let b1_im = *base_ptr.add(idx_b1 * 2 + 1);

            // new_a0 = m00 * a0 + m01 * b0
            let (na0_re, na0_im) = complex_mul_neon_f32_pair(m.m00_re, m.m00_im, a0_re, a0_im);
            let (t0_re, t0_im) = complex_mul_neon_f32_pair(m.m01_re, m.m01_im, b0_re, b0_im);
            let na0_re = na0_re + t0_re;
            let na0_im = na0_im + t0_im;

            // new_a1 = m00 * a1 + m01 * b1
            let (na1_re, na1_im) = complex_mul_neon_f32_pair(m.m00_re, m.m00_im, a1_re, a1_im);
            let (t1_re, t1_im) = complex_mul_neon_f32_pair(m.m01_re, m.m01_im, b1_re, b1_im);
            let na1_re = na1_re + t1_re;
            let na1_im = na1_im + t1_im;

            // new_b0 = m10 * a0 + m11 * b0
            let (nb0_re, nb0_im) = complex_mul_neon_f32_pair(m.m10_re, m.m10_im, a0_re, a0_im);
            let (t2_re, t2_im) = complex_mul_neon_f32_pair(m.m11_re, m.m11_im, b0_re, b0_im);
            let nb0_re = nb0_re + t2_re;
            let nb0_im = nb0_im + t2_im;

            // new_b1 = m10 * a1 + m11 * b1
            let (nb1_re, nb1_im) = complex_mul_neon_f32_pair(m.m10_re, m.m10_im, a1_re, a1_im);
            let (t3_re, t3_im) = complex_mul_neon_f32_pair(m.m11_re, m.m11_im, b1_re, b1_im);
            let nb1_re = nb1_re + t3_re;
            let nb1_im = nb1_im + t3_im;

            // Store results
            *base_ptr.add(idx_a0 * 2) = na0_re;
            *base_ptr.add(idx_a0 * 2 + 1) = na0_im;
            *base_ptr.add(idx_a1 * 2) = na1_re;
            *base_ptr.add(idx_a1 * 2 + 1) = na1_im;
            *base_ptr.add(idx_b0 * 2) = nb0_re;
            *base_ptr.add(idx_b0 * 2 + 1) = nb0_im;
            *base_ptr.add(idx_b1 * 2) = nb1_re;
            *base_ptr.add(idx_b1 * 2 + 1) = nb1_im;

            i += 2;
        }

        // Handle remaining odd pair
        if n % 2 == 1 {
            let idx_a = n - 1;
            let idx_b = idx_a + stride;
            if idx_b < chunk.len() {
                let a = chunk[idx_a];
                let b = chunk[idx_b];

                chunk[idx_a] = C32 {
                    re: m.m00_re * a.re - m.m00_im * a.im + m.m01_re * b.re - m.m01_im * b.im,
                    im: m.m00_re * a.im + m.m00_im * a.re + m.m01_re * b.im + m.m01_im * b.re,
                };
                chunk[idx_b] = C32 {
                    re: m.m10_re * a.re - m.m10_im * a.im + m.m11_re * b.re - m.m11_im * b.im,
                    im: m.m10_re * a.im + m.m10_im * a.re + m.m11_re * b.im + m.m11_im * b.re,
                };
            }
        }
    }
}

// ===================================================================
// F32 SCALAR FALLBACK
// ===================================================================

fn apply_hadamard_chunk_f32_scalar(chunk: &mut [C32], stride: usize, inv_sqrt2: f32) {
    let n = stride.min(chunk.len() / 2);
    for i in 0..n {
        let idx_b = i + stride;
        if idx_b < chunk.len() {
            let a = chunk[i];
            let b = chunk[idx_b];
            chunk[i] = C32 {
                re: (a.re + b.re) * inv_sqrt2,
                im: (a.im + b.im) * inv_sqrt2,
            };
            chunk[idx_b] = C32 {
                re: (a.re - b.re) * inv_sqrt2,
                im: (a.im - b.im) * inv_sqrt2,
            };
        }
    }
}

fn apply_unitary_chunk_f32_scalar(chunk: &mut [C32], stride: usize, m: &SimdMatrix2x2F32) {
    let n = stride.min(chunk.len() / 2);
    for i in 0..n {
        let idx_b = i + stride;
        if idx_b < chunk.len() {
            let a = chunk[i];
            let b = chunk[idx_b];

            chunk[i] = C32 {
                re: m.m00_re * a.re - m.m00_im * a.im + m.m01_re * b.re - m.m01_im * b.im,
                im: m.m00_re * a.im + m.m00_im * a.re + m.m01_re * b.im + m.m01_im * b.re,
            };
            chunk[idx_b] = C32 {
                re: m.m10_re * a.re - m.m10_im * a.im + m.m11_re * b.re - m.m11_im * b.im,
                im: m.m10_re * a.im + m.m10_im * a.re + m.m11_re * b.im + m.m11_im * b.re,
            };
        }
    }
}

// ===================================================================
// DISPATCH FUNCTIONS
// ===================================================================

#[inline]
fn apply_hadamard_chunk_f32(chunk: &mut [C32], stride: usize, inv_sqrt2: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_f32::apply_hadamard_chunk_neon_f32(chunk, stride, inv_sqrt2);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_hadamard_chunk_f32_scalar(chunk, stride, inv_sqrt2);
    }
}

#[inline]
fn apply_unitary_chunk_f32(chunk: &mut [C32], stride: usize, m: &SimdMatrix2x2F32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_f32::apply_unitary_chunk_neon_f32(chunk, stride, m);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_unitary_chunk_f32_scalar(chunk, stride, m);
    }
}

// ===================================================================
// ADDITIONAL F32 SIMD HELPERS
// ===================================================================

/// Apply Rx rotation to all pairs in a chunk.
#[inline]
fn apply_rx_chunk_f32(chunk: &mut [C32], stride: usize, cos_half: f32, sin_half: f32) {
    let n = stride.min(chunk.len() / 2);
    for i in 0..n {
        let idx_b = i + stride;
        if idx_b < chunk.len() {
            let a = chunk[i];
            let b = chunk[idx_b];
            chunk[i] = C32 {
                re: a.re * cos_half + a.im * sin_half,
                im: a.im * cos_half - a.re * sin_half,
            };
            chunk[idx_b] = C32 {
                re: b.re * cos_half - b.im * sin_half,
                im: b.im * cos_half + b.re * sin_half,
            };
        }
    }
}

/// Apply Ry rotation to all pairs in a chunk.
#[inline]
fn apply_ry_chunk_f32(chunk: &mut [C32], stride: usize, cos_half: f32, sin_half: f32) {
    let n = stride.min(chunk.len() / 2);
    for i in 0..n {
        let idx_b = i + stride;
        if idx_b < chunk.len() {
            let a = chunk[i];
            let b = chunk[idx_b];
            chunk[i] = C32 {
                re: a.re * cos_half - b.re * sin_half,
                im: a.im * cos_half - b.im * sin_half,
            };
            chunk[idx_b] = C32 {
                re: a.re * sin_half + b.re * cos_half,
                im: a.im * sin_half + b.im * cos_half,
            };
        }
    }
}

/// Apply full diagonal (different phases to |0⟩ and |1⟩) to all pairs in a chunk.
#[inline]
fn apply_full_diagonal_chunk_f32(
    chunk: &mut [C32],
    stride: usize,
    phase0_re: f32,
    phase0_im: f32,
    phase1_re: f32,
    phase1_im: f32,
) {
    let n = stride.min(chunk.len() / 2);
    for i in 0..n {
        let idx_b = i + stride;
        if idx_b < chunk.len() {
            // Apply phase0 to |0⟩ component
            let a = chunk[i];
            chunk[i] = C32 {
                re: phase0_re * a.re - phase0_im * a.im,
                im: phase0_re * a.im + phase0_im * a.re,
            };

            // Apply phase1 to |1⟩ component
            let b = chunk[idx_b];
            chunk[idx_b] = C32 {
                re: phase1_re * b.re - phase1_im * b.im,
                im: phase1_re * b.im + phase1_im * b.re,
            };
        }
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_state_init() {
        let state = QuantumStateF32::new(3);
        assert_eq!(state.dim, 8);
        assert_eq!(state.num_qubits, 3);
        assert!((state.get(0).re - 1.0).abs() < 1e-7);
        for i in 1..8 {
            assert!(state.get(i).re.abs() < 1e-7);
            assert!(state.get(i).im.abs() < 1e-7);
        }
    }

    #[test]
    fn test_f32_hadamard() {
        let mut state = QuantumStateF32::new(1);
        GateOpsF32::h(&mut state, 0);
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((state.get(0).re - inv_sqrt2).abs() < 1e-6);
        assert!((state.get(1).re - inv_sqrt2).abs() < 1e-6);
    }

    #[test]
    fn test_f32_cnot() {
        // |10⟩ → CNOT → |11⟩
        let mut state = QuantumStateF32::new(2);
        GateOpsF32::x(&mut state, 1); // |10⟩
        GateOpsF32::cnot(&mut state, 1, 0); // |11⟩
        assert!((state.get(3).re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_roundtrip_fidelity() {
        // Create f64 state, apply gates, convert to f32, compare
        let mut state64 = QuantumState::new(4);
        crate::GateOperations::h(&mut state64, 0);
        crate::GateOperations::cnot(&mut state64, 0, 1);
        crate::GateOperations::h(&mut state64, 2);

        let state32 = QuantumStateF32::from_f64(&state64);
        let fidelity = state32.fidelity_vs_f64(&state64);
        assert!(fidelity > 1.0 - 1e-6, "Roundtrip fidelity: {}", fidelity);
    }

    #[test]
    fn test_f32_vs_f64_circuit() {
        // Run same circuit on both f32 and f64, compare fidelity
        let n = 8;

        // f64 reference
        let mut state64 = QuantumState::new(n);
        for q in 0..n {
            crate::GateOperations::h(&mut state64, q);
        }
        for q in 0..n - 1 {
            crate::GateOperations::cnot(&mut state64, q, q + 1);
        }
        for q in 0..n {
            crate::GateOperations::h(&mut state64, q);
        }

        // f32
        let mut state32 = QuantumStateF32::new(n);
        for q in 0..n {
            GateOpsF32::h(&mut state32, q);
        }
        for q in 0..n - 1 {
            GateOpsF32::cnot(&mut state32, q, q + 1);
        }
        for q in 0..n {
            GateOpsF32::h(&mut state32, q);
        }

        let fidelity = state32.fidelity_vs_f64(&state64);
        assert!(
            fidelity > 1.0 - 1e-4,
            "f32 vs f64 fidelity too low: {}",
            fidelity
        );
    }

    #[test]
    fn test_f32_bell_state() {
        let mut state = QuantumStateF32::new(2);
        GateOpsF32::h(&mut state, 0);
        GateOpsF32::cnot(&mut state, 0, 1);
        // Should be (|00⟩ + |11⟩) / √2
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((state.get(0).re - inv_sqrt2).abs() < 1e-6);
        assert!(state.get(1).re.abs() < 1e-6);
        assert!(state.get(2).re.abs() < 1e-6);
        assert!((state.get(3).re - inv_sqrt2).abs() < 1e-6);
    }

    #[test]
    fn test_f32_to_f64_conversion() {
        let mut state32 = QuantumStateF32::new(2);
        GateOpsF32::h(&mut state32, 0);
        let state64 = state32.to_f64();
        let fidelity = state32.fidelity_vs_f64(&state64);
        assert!(
            fidelity > 1.0 - 1e-6,
            "f32->f64 roundtrip fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_f32_x_gate() {
        let mut state = QuantumStateF32::new(1);
        GateOpsF32::x(&mut state, 0);
        assert!(state.get(0).re.abs() < 1e-7);
        assert!((state.get(1).re - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_f32_z_gate() {
        let mut state = QuantumStateF32::new(1);
        GateOpsF32::h(&mut state, 0);
        GateOpsF32::z(&mut state, 0);
        // H|0⟩ = (|0⟩+|1⟩)/√2, Z gives (|0⟩-|1⟩)/√2, which is H|1⟩
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((state.get(0).re - inv_sqrt2).abs() < 1e-6);
        assert!((state.get(1).re + inv_sqrt2).abs() < 1e-6);
    }

    #[test]
    fn test_f32_deep_circuit_precision() {
        // 50 H gates on same qubit (H^50 = I or H depending on parity)
        // 50 is even, so H^50 = I
        let mut state32 = QuantumStateF32::new(1);
        for _ in 0..50 {
            GateOpsF32::h(&mut state32, 0);
        }
        assert!(
            (state32.get(0).re - 1.0).abs() < 1e-4,
            "H^50 should be identity, got amp[0]={:?}",
            state32.get(0)
        );
        assert!(
            state32.get(1).re.abs() < 1e-4,
            "H^50 should be identity, got amp[1]={:?}",
            state32.get(1)
        );
    }

    #[test]
    fn test_f32_neon_matches_scalar_hadamard() {
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let mut chunk_scalar = vec![C32 { re: 0.7, im: 0.1 }, C32 { re: 0.3, im: -0.4 }];
        let mut chunk_dispatch = chunk_scalar.clone();

        apply_hadamard_chunk_f32_scalar(&mut chunk_scalar, 1, inv_sqrt2);
        apply_hadamard_chunk_f32(&mut chunk_dispatch, 1, inv_sqrt2);

        for i in 0..2 {
            assert!(
                (chunk_scalar[i].re - chunk_dispatch[i].re).abs() < 1e-6,
                "Hadamard f32 mismatch at [{}].re: {} vs {}",
                i,
                chunk_scalar[i].re,
                chunk_dispatch[i].re
            );
            assert!(
                (chunk_scalar[i].im - chunk_dispatch[i].im).abs() < 1e-6,
                "Hadamard f32 mismatch at [{}].im",
                i
            );
        }
    }
}
