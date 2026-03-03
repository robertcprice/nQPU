//! Full Metal GPU Quantum Simulator
//!
//! Complete GPU-accelerated quantum simulator using Apple Metal framework.

use num_complex::Complex64;
use std::ptr;

#[cfg(target_os = "macos")]
use metal::*;

use crate::C64;

pub struct MetalQuantumSimulator {
    #[cfg(target_os = "macos")]
    device: metal::Device,
    #[cfg(target_os = "macos")]
    command_queue: metal::CommandQueue,
    num_qubits: usize,
    dim: usize,
}

impl MetalQuantumSimulator {
    #[cfg(target_os = "macos")]
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        let device = Device::system_default().ok_or("Failed to get Metal device".to_string())?;
        let command_queue = device.new_command_queue();

        Ok(MetalQuantumSimulator {
            device,
            command_queue,
            num_qubits,
            dim: 1 << num_qubits,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        Err("Metal GPU is only supported on macOS".to_string())
    }

    #[cfg(target_os = "macos")]
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Create a state buffer on GPU initialized to |0...0⟩
    #[cfg(target_os = "macos")]
    pub fn create_state_buffer(&self) -> metal::Buffer {
        let mut state = vec![Complex64::new(0.0, 0.0); self.dim];
        state[0] = Complex64::new(1.0, 0.0);

        let buffer = self.device.new_buffer(
            (state.len() * std::mem::size_of::<C64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            ptr::copy_nonoverlapping(
                state.as_ptr() as *const u8,
                buffer.contents() as *mut u8,
                state.len() * std::mem::size_of::<C64>(),
            );
        }

        buffer
    }

    /// Load shader library
    #[cfg(target_os = "macos")]
    fn load_shader_library(&self) -> Result<metal::Library, String> {
        let shader_source = include_str!("metal/shaders.metal");
        self.device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile Metal shaders: {:?}", e))
    }

    /// Create compute pipeline for a shader function
    #[cfg(target_os = "macos")]
    fn create_pipeline(
        &self,
        library: &metal::Library,
        function_name: &str,
    ) -> metal::ComputePipelineState {
        let function = library
            .get_function(function_name, None)
            .expect(&format!("Failed to find function: {}", function_name));
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create compute pipeline")
    }

    /// Execute Hadamard gate on GPU
    #[cfg(target_os = "macos")]
    pub fn hadamard_gate(&self, state_buffer: &metal::Buffer, qubit: usize) {
        let library = self.load_shader_library().unwrap();
        let pipeline = self.create_pipeline(&library, "hadamard_gate");
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let qubit_val = qubit as u32;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(state_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&qubit_val) as u64,
            &qubit_val as *const u32 as *const _,
        );

        let thread_groups = metal::MTLSize::new((self.dim / 2) as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, metal::MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Execute X gate on GPU
    #[cfg(target_os = "macos")]
    pub fn x_gate(&self, state_buffer: &metal::Buffer, qubit: usize) {
        let library = self.load_shader_library().unwrap();
        let pipeline = self.create_pipeline(&library, "x_gate");
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let qubit_val = qubit as u32;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(state_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&qubit_val) as u64,
            &qubit_val as *const u32 as *const _,
        );

        let thread_groups = metal::MTLSize::new((self.dim / 2) as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, metal::MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Execute Z gate on GPU
    #[cfg(target_os = "macos")]
    pub fn z_gate(&self, state_buffer: &metal::Buffer, qubit: usize) {
        let library = self.load_shader_library().unwrap();
        let pipeline = self.create_pipeline(&library, "z_gate");
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let qubit_val = qubit as u32;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(state_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&qubit_val) as u64,
            &qubit_val as *const u32 as *const _,
        );

        let thread_groups = metal::MTLSize::new(self.dim as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, metal::MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Read state from GPU buffer
    #[cfg(target_os = "macos")]
    pub fn read_state(&self, state_buffer: &metal::Buffer) -> Vec<C64> {
        let mut state = vec![Complex64::new(0.0, 0.0); self.dim];

        unsafe {
            ptr::copy_nonoverlapping(
                state_buffer.contents() as *const u8,
                state.as_mut_ptr() as *mut u8,
                self.dim * std::mem::size_of::<C64>(),
            );
        }

        state
    }

    /// Run Grover's search on GPU
    #[cfg(target_os = "macos")]
    pub fn grover_search(&self, target: usize, num_iterations: usize) -> (Vec<C64>, usize) {
        let state_buffer = self.create_state_buffer();

        // Initialize uniform superposition with H gates
        for i in 0..self.num_qubits {
            self.hadamard_gate(&state_buffer, i);
        }

        // Grover iterations
        for _ in 0..num_iterations {
            // Oracle: phase flip on target
            self.oracle(&state_buffer, target);

            // Diffusion: 2|s⟩⟨s| - I
            self.diffusion(&state_buffer);
        }

        // Read final state
        let state = self.read_state(&state_buffer);

        // Find most probable state
        let max_idx = state
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let prob_a = a.1.norm_sqr();
                let prob_b = b.1.norm_sqr();
                prob_a.partial_cmp(&prob_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        (state, max_idx)
    }

    /// Oracle: phase flip on target state
    #[cfg(target_os = "macos")]
    fn oracle(&self, state_buffer: &metal::Buffer, target: usize) {
        let library = self.load_shader_library().unwrap();
        let pipeline = self.create_pipeline(&library, "oracle_phase_flip");
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let target_val = target as u32;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(state_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&target_val) as u64,
            &target_val as *const u32 as *const _,
        );

        let thread_groups = metal::MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, metal::MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Diffusion operator: 2|s⟩⟨s| - I
    #[cfg(target_os = "macos")]
    fn diffusion(&self, state_buffer: &metal::Buffer) {
        // Apply H⊗n
        for i in 0..self.num_qubits {
            self.hadamard_gate(state_buffer, i);
        }

        // Phase flip on |0⟩
        let library = self.load_shader_library().unwrap();
        let pipeline = self.create_pipeline(&library, "phase_flip_zero");
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(state_buffer), 0);

        let thread_groups = metal::MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, metal::MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Apply H⊗n again
        for i in 0..self.num_qubits {
            self.hadamard_gate(state_buffer, i);
        }
    }

    /// Benchmark GPU Grover search
    #[cfg(target_os = "macos")]
    pub fn benchmark_grover(&self, target: usize) -> (f64, usize) {
        use std::time::Instant;
        let num_iterations = (std::f64::consts::PI / 4.0 * self.dim as f64).sqrt() as usize;

        let start = Instant::now();
        let (_state, result) = self.grover_search(target, num_iterations);
        let elapsed = start.elapsed().as_secs_f64();

        (elapsed, result)
    }

    /// Benchmark GPU gate operations
    #[cfg(target_os = "macos")]
    pub fn benchmark_gates(&self, num_gates: usize) -> f64 {
        use std::time::Instant;
        let state_buffer = self.create_state_buffer();

        let start = Instant::now();
        for i in 0..num_gates {
            self.hadamard_gate(&state_buffer, i % self.num_qubits);
        }
        let elapsed = start.elapsed().as_secs_f64();

        elapsed
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

// Benchmark functions for use from main
#[cfg(target_os = "macos")]
pub fn benchmark_gpu_grover(num_qubits: usize, target: usize) -> Result<(f64, usize), String> {
    let simulator = MetalQuantumSimulator::new(num_qubits)?;
    Ok(simulator.benchmark_grover(target))
}

#[cfg(target_os = "macos")]
pub fn benchmark_gpu_gates(num_qubits: usize, num_gates: usize) -> Result<f64, String> {
    let simulator = MetalQuantumSimulator::new(num_qubits)?;
    Ok(simulator.benchmark_gates(num_gates))
}

#[cfg(not(target_os = "macos"))]
pub fn benchmark_gpu_grover(_num_qubits: usize, _target: usize) -> Result<(f64, usize), String> {
    Err("Metal GPU is only supported on macOS".to_string())
}

#[cfg(not(target_os = "macos"))]
pub fn benchmark_gpu_gates(_num_qubits: usize, _num_gates: usize) -> Result<f64, String> {
    Err("Metal GPU is only supported on macOS".to_string())
}
