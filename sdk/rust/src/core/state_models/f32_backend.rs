//! Single-Precision (F32) Backend
//!
//! F32 quantum state implementation for reduced memory usage and improved performance.
//!
//! **Benefits**:
//! - 2x memory reduction (f32 vs f64)
//! - Faster GPU operations (simd_float vs simd_double)
//! - Better cache utilization
//! - Sufficient accuracy for many quantum circuits
//!
//! **Tradeoffs**:
//! - Reduced numerical precision (7 digits vs 15 digits)
//! - Potential error accumulation in deep circuits
//! - Not suitable for precision-critical applications

use num_complex::Complex32;
use std::time::Instant;

/// F32 complex number alias.
pub type C32 = Complex32;

/// F32 quantum state.
pub struct F32QuantumState {
    /// State amplitudes in f32 precision.
    amplitudes: Vec<C32>,
    /// Number of qubits.
    num_qubits: usize,
    /// Dimension (2^n).
    dim: usize,
}

impl F32QuantumState {
    /// Create a new F32 quantum state initialized to |0⟩.
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut amplitudes = vec![C32::new(0.0, 0.0); dim];
        amplitudes[0] = C32::new(1.0, 0.0);

        Self {
            amplitudes,
            num_qubits,
            dim,
        }
    }

    /// Get number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get state dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get reference to amplitudes.
    pub fn get_amplitudes(&self) -> &[C32] {
        &self.amplitudes
    }

    /// Get mutable reference to amplitudes.
    pub fn get_amplitudes_mut(&mut self) -> &mut [C32] {
        &mut self.amplitudes
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.amplitudes.len() * std::mem::size_of::<C32>()
    }

    /// Apply single-qubit gate.
    pub fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        matrix: [[C32; 2]; 2],
    ) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of bounds".to_string());
        }

        let stride = 1usize << qubit;

        for i in (0..self.dim).step_by(stride * 2) {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;

                let a0 = self.amplitudes[idx0];
                let a1 = self.amplitudes[idx1];

                self.amplitudes[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1;
                self.amplitudes[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1;
            }
        }

        Ok(())
    }

    /// Apply two-qubit gate.
    pub fn apply_two_qubit_gate(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        matrix: [[C32; 4]; 4],
    ) -> Result<(), String> {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Err("Qubit index out of bounds".to_string());
        }

        if qubit1 == qubit2 {
            return Err("Qubits must be different".to_string());
        }

        let (lo, hi) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        let stride_lo = 1usize << lo;
        let stride_hi = 1usize << hi;

        for i in (0..self.dim).step_by(stride_hi * 2) {
            for j in (i..i + stride_hi).step_by(stride_lo * 2) {
                for k in 0..stride_lo {
                    let i0 = j + k;
                    let i1 = i0 + stride_lo;
                    let i2 = i0 + stride_hi;
                    let i3 = i1 + stride_hi;

                    let a0 = self.amplitudes[i0];
                    let a1 = self.amplitudes[i1];
                    let a2 = self.amplitudes[i2];
                    let a3 = self.amplitudes[i3];

                    self.amplitudes[i0] = matrix[0][0] * a0
                        + matrix[0][1] * a1
                        + matrix[0][2] * a2
                        + matrix[0][3] * a3;
                    self.amplitudes[i1] = matrix[1][0] * a0
                        + matrix[1][1] * a1
                        + matrix[1][2] * a2
                        + matrix[1][3] * a3;
                    self.amplitudes[i2] = matrix[2][0] * a0
                        + matrix[2][1] * a1
                        + matrix[2][2] * a2
                        + matrix[2][3] * a3;
                    self.amplitudes[i3] = matrix[3][0] * a0
                        + matrix[3][1] * a1
                        + matrix[3][2] * a2
                        + matrix[3][3] * a3;
                }
            }
        }

        Ok(())
    }

    /// Compute probabilities (|amplitude|^2).
    pub fn probabilities(&self) -> Vec<f32> {
        self.amplitudes.iter().map(|&a| a.norm_sqr()).collect()
    }

    /// Measure a qubit.
    pub fn measure(&mut self, qubit: usize) -> Result<usize, String> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of bounds".to_string());
        }

        let stride = 1usize << qubit;

        // Compute probability of measuring 0
        let mut p0 = 0.0f32;
        for i in (0..self.dim).step_by(stride * 2) {
            for j in 0..stride {
                p0 += self.amplitudes[i + j].norm_sqr();
            }
        }

        // Sample from distribution
        let outcome = if rand::random::<f32>() < p0 { 0 } else { 1 };

        // Collapse state
        let norm = if outcome == 0 {
            1.0 / p0.sqrt()
        } else {
            1.0 / (1.0 - p0).sqrt()
        };

        for i in (0..self.dim).step_by(stride * 2) {
            for j in 0..stride {
                let idx = i + j + outcome * stride;
                if idx < self.dim {
                    self.amplitudes[idx] *= norm;
                }
                // Zero out the other outcome
                let other_idx = i + j + (1 - outcome) * stride;
                if other_idx < self.dim {
                    self.amplitudes[other_idx] = C32::new(0.0, 0.0);
                }
            }
        }

        Ok(outcome)
    }

    /// Normalize the state.
    pub fn normalize(&mut self) {
        let norm_sq: f32 = self.amplitudes.iter().map(|&a| a.norm_sqr()).sum();

        if norm_sq > 0.0 {
            let norm = 1.0 / norm_sq.sqrt();
            for amp in &mut self.amplitudes {
                *amp *= norm;
            }
        }
    }

    /// Convert to F64 state (for comparison).
    pub fn to_f64_state(&self) -> crate::QuantumState {
        let mut state = crate::QuantumState::new(self.num_qubits);
        let amplitudes = state.amplitudes_mut();

        for (i, &amp) in self.amplitudes.iter().enumerate() {
            if i < amplitudes.len() {
                amplitudes[i] = crate::C64::new(amp.re as f64, amp.im as f64);
            }
        }

        state
    }

    /// Compute fidelity with another state.
    pub fn fidelity(&self, other: &Self) -> Result<f32, String> {
        if self.dim != other.dim {
            return Err("State dimensions don't match".to_string());
        }

        let mut inner_product = C32::new(0.0, 0.0);
        for (i, &a) in self.amplitudes.iter().enumerate() {
            inner_product += a.conj() * other.amplitudes[i];
        }

        Ok(inner_product.norm_sqr())
    }
}

/// GPU-accelerated F32 quantum state (Metal).
#[cfg(target_os = "macos")]
pub struct GPUF32QuantumState {
    state_buffer: metal::Buffer,
    command_queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    num_qubits: usize,
    dim: usize,
}

#[cfg(target_os = "macos")]
impl GPUF32QuantumState {
    /// Create a new GPU F32 quantum state.
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        use metal::*;

        let device = Device::system_default().ok_or("No Metal device found")?;

        let command_queue = device.new_command_queue();

        // Load F32 shaders
        let shader_source = r#"
#include <metal_stdlib>
using namespace metal;

struct ComplexF32 {
    float real;
    float imag;
};

ComplexF32 complex_mul_f32(ComplexF32 a, ComplexF32 b) {
    return ComplexF32{
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

ComplexF32 complex_add_f32(ComplexF32 a, ComplexF32 b) {
    return ComplexF32{a.real + b.real, a.imag + b.imag};
}

struct SingleQParamsF32 {
    uint qubit;
    ComplexF32 matrix[2][2];
};

kernel void single_qubit_f32(
    device ComplexF32* state [[buffer(0)]],
    constant SingleQParamsF32& params [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    uint n = params.qubit;
    uint stride = 1 << n;
    uint base = (index / stride) * stride * 2;
    uint offset = index % stride;

    uint i0 = base + offset;
    uint i1 = i0 + stride;

    ComplexF32 a0 = state[i0];
    ComplexF32 a1 = state[i1];

    state[i0] = complex_add_f32(
        complex_mul_f32(params.matrix[0][0], a0),
        complex_mul_f32(params.matrix[0][1], a1)
    );
    state[i1] = complex_add_f32(
        complex_mul_f32(params.matrix[1][0], a0),
        complex_mul_f32(params.matrix[1][1], a1)
    );
}
"#;

        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Shader compilation failed: {}", e))?;

        let func = library
            .get_function("single_qubit_f32", None)
            .map_err(|e| format!("Shader function: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("Pipeline creation: {}", e))?;

        let dim = 1usize << num_qubits;
        let buffer_size = (dim * std::mem::size_of::<C32>()) as u64;
        let state_buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        // Initialize to |0⟩ state
        let ptr = state_buffer.contents() as *mut C32;
        unsafe {
            *ptr = C32::new(1.0, 0.0);
            for i in 1..dim {
                *ptr.add(i) = C32::new(0.0, 0.0);
            }
        }

        Ok(Self {
            state_buffer,
            command_queue,
            pipeline,
            num_qubits,
            dim,
        })
    }

    /// Apply single-qubit gate on GPU.
    pub fn apply_single_qubit_gate(
        &self,
        qubit: usize,
        matrix: [[C32; 2]; 2],
    ) -> Result<(), String> {
        use metal::*;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);

        let params = SingleQParamsF32 {
            qubit: qubit as u32,
            matrix: unsafe { std::mem::transmute(matrix) },
        };

        let params_buffer = self.state_buffer.device().new_buffer(
            std::mem::size_of::<SingleQParamsF32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            let ptr = params_buffer.contents() as *mut SingleQParamsF32;
            std::ptr::write(ptr, params);
        }

        encoder.set_buffer(1, Some(&params_buffer), 0);

        let threads_per_grid = (self.dim / 2) as u64;
        let threads_per_threadgroup = std::cmp::min(256, threads_per_grid);
        let threadgroups =
            (threads_per_grid + threads_per_threadgroup - 1) / threads_per_threadgroup;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SingleQParamsF32 {
    qubit: u32,
    matrix: [[C32; 2]; 2],
}

/// F32 benchmark results.
#[derive(Clone, Debug)]
pub struct F32BenchmarkResults {
    pub f64_time: f64,
    pub f32_time: f64,
    pub f64_memory: usize,
    pub f32_memory: usize,
    pub speedup: f64,
    pub fidelity_loss: f32,
}

impl F32QuantumState {
    /// Benchmark F32 vs F64 performance.
    pub fn benchmark_f32_vs_f64(
        num_qubits: usize,
        depth: usize,
        iterations: usize,
    ) -> F32BenchmarkResults {
        println!("═══════════════════════════════════════════════════════════════");
        println!(
            "F32 vs F64 Backend Benchmark: {} qubits, depth {}",
            num_qubits, depth
        );
        println!("═══════════════════════════════════════════════════════════════");

        let h_matrix_f64 = [
            [
                crate::C64::new(0.70710678, 0.0),
                crate::C64::new(0.70710678, 0.0),
            ],
            [
                crate::C64::new(0.70710678, 0.0),
                crate::C64::new(-0.70710678, 0.0),
            ],
        ];

        let h_matrix_f32 = [
            [C32::new(0.70710678, 0.0), C32::new(0.70710678, 0.0)],
            [C32::new(0.70710678, 0.0), C32::new(-0.70710678, 0.0)],
        ];

        // Benchmark F64
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = crate::QuantumState::new(num_qubits);
            for _ in 0..depth {
                for q in 0..num_qubits {
                    crate::GateOperations::u(&mut state, q, &h_matrix_f64);
                }
            }
        }
        let f64_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Benchmark F32
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = Self::new(num_qubits);
            for _ in 0..depth {
                for q in 0..num_qubits {
                    let _ = state.apply_single_qubit_gate(q, h_matrix_f32);
                }
            }
        }
        let f32_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Memory comparison
        let f64_memory = (1usize << num_qubits) * std::mem::size_of::<crate::C64>();
        let f32_memory = (1usize << num_qubits) * std::mem::size_of::<C32>();

        // Fidelity comparison (run shallow circuit)
        let mut state_f64 = crate::QuantumState::new(num_qubits);
        let mut state_f32 = Self::new(num_qubits);

        for q in 0..num_qubits.min(5) {
            crate::GateOperations::u(&mut state_f64, q, &h_matrix_f64);
            let _ = state_f32.apply_single_qubit_gate(q, h_matrix_f32);
        }

        let state_f32_converted = state_f32.to_f64_state();
        let fidelity = state_f64.fidelity(&state_f32_converted);

        let speedup = f64_time / f32_time;

        println!("F64 time:       {:.6} sec", f64_time);
        println!("F32 time:       {:.6} sec", f32_time);
        println!("Speedup:        {:.2}x", speedup);
        println!();
        println!(
            "F64 memory:     {} bytes ({:.2} MB)",
            f64_memory,
            f64_memory as f64 / (1024.0 * 1024.0)
        );
        println!(
            "F32 memory:     {} bytes ({:.2} MB)",
            f32_memory,
            f32_memory as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Memory savings: {:.1}%",
            100.0 * (1.0 - f32_memory as f64 / f64_memory as f64)
        );
        println!();
        println!(
            "Fidelity loss:  {:.6} (should be < 1e-6 for shallow circuits)",
            1.0 - fidelity as f32
        );
        println!();

        F32BenchmarkResults {
            f64_time,
            f32_time,
            f64_memory,
            f32_memory,
            speedup,
            fidelity_loss: (1.0 - fidelity) as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_state_creation() {
        let state = F32QuantumState::new(10);
        assert_eq!(state.num_qubits(), 10);
        assert_eq!(state.dim(), 1024);
        assert_eq!(state.amplitudes[0], C32::new(1.0, 0.0));
    }

    #[test]
    fn test_f32_memory_usage() {
        let state = F32QuantumState::new(20);
        let f32_usage = state.memory_usage();
        let f64_usage = state.dim * std::mem::size_of::<crate::C64>();

        assert_eq!(f32_usage, f64_usage / 2);
    }

    #[test]
    fn test_f32_single_qubit_gate() {
        let mut state = F32QuantumState::new(10);
        let h_matrix = [
            [C32::new(0.70710678, 0.0), C32::new(0.70710678, 0.0)],
            [C32::new(0.70710678, 0.0), C32::new(-0.70710678, 0.0)],
        ];

        state.apply_single_qubit_gate(0, h_matrix).unwrap();

        // Check normalization
        let probs = state.probabilities();
        let total_prob: f32 = probs.iter().sum();
        assert!((total_prob - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_to_f64_conversion() {
        let state_f32 = F32QuantumState::new(10);
        let state_f64 = state_f32.to_f64_state();

        assert_eq!(state_f32.num_qubits, state_f64.num_qubits);
    }
}
