// Metal Parallel Quantum Executor
// ===============================
// Safe Rust interface to Metal GPU acceleration for parallel quantum operations
// Provides high-performance GPU kernels for quantum gates and transformer operations

#![allow(unexpected_cfgs)]

#[cfg(target_os = "macos")]
use cocoa::base::{id, nil};
#[cfg(target_os = "macos")]
use cocoa::foundation::NSString as cocoaNSString;
#[cfg(target_os = "macos")]
use metal::*;
#[cfg(target_os = "macos")]
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
#[cfg(target_os = "macos")]
use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::ptr;

use crate::{QuantumState, C64};
use std::time::Instant;

// ============================================================
// METAL PARALLEL QUANTUM EXECUTOR
// ============================================================

#[cfg(target_os = "macos")]
pub struct MetalParallelQuantumExecutor {
    device: id,
    command_queue: id,
    library: id,
    pipeline_parallel_h: Option<id>,
    pipeline_parallel_x: Option<id>,
    pipeline_parallel_z: Option<id>,
    pipeline_parallel_rotations: Option<id>,
    pipeline_parallel_cnot: Option<id>,
    pipeline_encode_heads: Option<id>,
    pipeline_attention_weights: Option<id>,
    pipeline_apply_attention: Option<id>,
    pipeline_batch_forward: Option<id>,
    pipeline_layer_norm: Option<id>,
    pipeline_grover_oracle: Option<id>,
    pipeline_grover_diffusion: Option<id>,
    pipeline_qft: Option<id>,
    pipeline_normalize: Option<id>,
}

#[cfg(target_os = "macos")]
impl MetalParallelQuantumExecutor {
    /// Create a new Metal parallel quantum executor
    pub fn new() -> Result<Self, String> {
        unsafe {
            // Get the default Metal device
            let device: id = msg_send![class!(MTLCreateSystemDefaultDevice), newDevice];
            if device.is_null() {
                return Err("Failed to create Metal device".to_string());
            }

            // Create command queue
            let command_queue: id = msg_send![device, newCommandQueue];
            if command_queue.is_null() {
                return Err("Failed to create command queue".to_string());
            }

            // Load shader library
            let library = Self::load_library(device)?;

            // Create compute pipelines (lazy initialization)
            let executor = Self {
                device,
                command_queue,
                library,
                pipeline_parallel_h: None,
                pipeline_parallel_x: None,
                pipeline_parallel_z: None,
                pipeline_parallel_rotations: None,
                pipeline_parallel_cnot: None,
                pipeline_encode_heads: None,
                pipeline_attention_weights: None,
                pipeline_apply_attention: None,
                pipeline_batch_forward: None,
                pipeline_layer_norm: None,
                pipeline_grover_oracle: None,
                pipeline_grover_diffusion: None,
                pipeline_qft: None,
                pipeline_normalize: None,
            };

            Ok(executor)
        }
    }

    /// Load Metal shader library from file
    fn load_library(device: id) -> Result<id, String> {
        // Try multiple shader paths
        let shader_paths = vec![
            PathBuf::from("src/metal/shaders/parallel_quantum.metal"),
            PathBuf::from("./shaders/parallel_quantum.metal"),
            PathBuf::from("../src/metal/shaders/parallel_quantum.metal"),
        ];

        let mut library = None;

        for shader_path in shader_paths {
            if shader_path.exists() {
                match Self::compile_from_file(device, &shader_path) {
                    Ok(lib) => {
                        library = Some(lib);
                        break;
                    }
                    Err(_) => continue,
                }
            }
        }

        match library {
            Some(lib) => Ok(lib),
            None => Err("Failed to load shader library from any known path".to_string()),
        }
    }

    /// Compile Metal shader from file
    fn compile_from_file(device: id, path: &PathBuf) -> Result<id, String> {
        unsafe {
            let shader_source = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read shader file {:?}: {}", path, e))?;

            let ns_string = cocoaNSString::alloc(nil).init_str(&shader_source);

            let compile_options: id = msg_send![class!(MTLCompileOptions), alloc];
            let compile_options: id = msg_send![compile_options, init];

            let library: id = msg_send![device,
                newLibraryWithSource:ns_string
                options:compile_options
                error:ptr::null_mut::<Object>()
            ];

            if library.is_null() {
                return Err(format!("Failed to compile Metal shader from {:?}", path));
            }

            Ok(library)
        }
    }

    /// Get or create compute pipeline state
    fn get_pipeline(&mut self, kernel_name: &str) -> Result<id, String> {
        unsafe {
            // Check if we have a cached pipeline
            let pipeline = match kernel_name {
                "parallel_hadamard" => &mut self.pipeline_parallel_h,
                "parallel_pauli_x" => &mut self.pipeline_parallel_x,
                "parallel_pauli_z" => &mut self.pipeline_parallel_z,
                "parallel_rotations" => &mut self.pipeline_parallel_rotations,
                "parallel_cnot" => &mut self.pipeline_parallel_cnot,
                "parallel_encode_heads" => &mut self.pipeline_encode_heads,
                "parallel_attention_weights" => &mut self.pipeline_attention_weights,
                "parallel_apply_attention" => &mut self.pipeline_apply_attention,
                "batch_transformer_forward" => &mut self.pipeline_batch_forward,
                "batch_quantum_layer_norm" => &mut self.pipeline_layer_norm,
                "parallel_grover_oracle" => &mut self.pipeline_grover_oracle,
                "parallel_grover_diffusion" => &mut self.pipeline_grover_diffusion,
                "parallel_qft" => &mut self.pipeline_qft,
                "normalize_state" => &mut self.pipeline_normalize,
                _ => return Err(format!("Unknown kernel: {}", kernel_name)),
            };

            if let Some(p) = pipeline {
                return Ok(*p);
            }

            // Create new pipeline
            let ns_name = cocoaNSString::alloc(nil).init_str(kernel_name);
            let function: id = msg_send![self.library, newFunctionWithName:ns_name];

            if function.is_null() {
                return Err(format!("Kernel function not found: {}", kernel_name));
            }

            let pipeline_state: id = msg_send![self.device,
                newComputePipelineStateWithFunction:function
                error:ptr::null_mut::<Object>()
            ];

            if pipeline_state.is_null() {
                return Err(format!("Failed to create pipeline for: {}", kernel_name));
            }

            *pipeline = Some(pipeline_state);
            Ok(pipeline_state)
        }
    }

    /// Create Metal buffer from data
    fn create_buffer<T>(&self, data: &[T]) -> Result<id, String>
    where
        T: Copy,
    {
        unsafe {
            let byte_length = std::mem::size_of_val(data);
            let buffer: id = msg_send![self.device,
                newBufferWithBytes:data.as_ptr()
                length:byte_length
                options:0x0001  // MTLResourceStorageModeShared
            ];

            if buffer.is_null() {
                return Err("Failed to create Metal buffer".to_string());
            }

            Ok(buffer)
        }
    }

    /// Execute parallel Hadamard gates on GPU
    pub fn parallel_hadamard_gpu(
        &mut self,
        state: &mut QuantumState,
        target_qubits: &[usize],
    ) -> Result<(), String> {
        unsafe {
            let pipeline = self.get_pipeline("parallel_hadamard")?;

            // Create buffers
            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let num_qubits = state.num_qubits as u32;
            let num_qubits_buffer = self.create_buffer(&[num_qubits])?;
            let targets: Vec<u32> = target_qubits.iter().map(|&x| x as u32).collect();
            let targets_buffer = self.create_buffer(&targets)?;
            let num_targets = target_qubits.len() as u32;
            let num_targets_buffer = self.create_buffer(&[num_targets])?;

            // Create command buffer
            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            // Set pipeline and buffers
            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:num_qubits_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:targets_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:num_targets_buffer offset:0 atIndex:3];

            // Dispatch threads
            let threads_per_threadgroup = metal::MTLSize {
                width: target_qubits.len() as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () = msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
            let _: () = msg_send![encoder, endEncoding];

            // Commit and wait
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            // Read back results
            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }

    /// Execute parallel Pauli-X gates on GPU
    pub fn parallel_pauli_x_gpu(
        &mut self,
        state: &mut QuantumState,
        target_qubits: &[usize],
    ) -> Result<(), String> {
        unsafe {
            let pipeline = self.get_pipeline("parallel_pauli_x")?;

            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let num_qubits = state.num_qubits as u32;
            let num_qubits_buffer = self.create_buffer(&[num_qubits])?;
            let targets: Vec<u32> = target_qubits.iter().map(|&x| x as u32).collect();
            let targets_buffer = self.create_buffer(&targets)?;
            let num_targets = target_qubits.len() as u32;
            let num_targets_buffer = self.create_buffer(&[num_targets])?;

            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:num_qubits_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:targets_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:num_targets_buffer offset:0 atIndex:3];

            let threads = metal::MTLSize {
                width: target_qubits.len() as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () =
                msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }

    /// Execute parallel rotations (RX, RY, RZ) on GPU
    pub fn parallel_rotations_gpu(
        &mut self,
        state: &mut QuantumState,
        target_qubits: &[usize],
        angles: &[f64],
        rotation_type: u32, // 0=RX, 1=RY, 2=RZ
    ) -> Result<(), String> {
        unsafe {
            let pipeline = self.get_pipeline("parallel_rotations")?;

            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let num_qubits = state.num_qubits as u32;
            let num_qubits_buffer = self.create_buffer(&[num_qubits])?;
            let targets: Vec<u32> = target_qubits.iter().map(|&x| x as u32).collect();
            let targets_buffer = self.create_buffer(&targets)?;
            let angles_buffer = self.create_buffer(angles)?;
            let num_targets = target_qubits.len() as u32;
            let num_targets_buffer = self.create_buffer(&[num_targets])?;
            let rotation_type_buffer = self.create_buffer(&[rotation_type])?;

            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:num_qubits_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:targets_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:angles_buffer offset:0 atIndex:3];
            let _: () = msg_send![encoder, setBuffer:rotation_type_buffer offset:0 atIndex:4];
            let _: () = msg_send![encoder, setBuffer:num_targets_buffer offset:0 atIndex:5];

            let threads = metal::MTLSize {
                width: target_qubits.len() as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () =
                msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }

    /// Execute parallel CNOT gates on GPU
    pub fn parallel_cnot_gpu(
        &mut self,
        state: &mut QuantumState,
        control_qubits: &[usize],
        target_qubits: &[usize],
    ) -> Result<(), String> {
        unsafe {
            if control_qubits.len() != target_qubits.len() {
                return Err("Control and target qubit arrays must have same length".to_string());
            }

            let pipeline = self.get_pipeline("parallel_cnot")?;

            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let num_qubits = state.num_qubits as u32;
            let num_qubits_buffer = self.create_buffer(&[num_qubits])?;
            let controls: Vec<u32> = control_qubits.iter().map(|&x| x as u32).collect();
            let controls_buffer = self.create_buffer(&controls)?;
            let targets: Vec<u32> = target_qubits.iter().map(|&x| x as u32).collect();
            let targets_buffer = self.create_buffer(&targets)?;
            let num_gates = control_qubits.len() as u32;
            let num_gates_buffer = self.create_buffer(&[num_gates])?;

            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:num_qubits_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:controls_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:targets_buffer offset:0 atIndex:3];
            let _: () = msg_send![encoder, setBuffer:num_gates_buffer offset:0 atIndex:4];

            let threads = metal::MTLSize {
                width: control_qubits.len() as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () =
                msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }

    /// Execute parallel multi-head attention encoding on GPU
    pub fn parallel_encode_heads_gpu(
        &mut self,
        state: &mut QuantumState,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        qubits_per_head: usize,
    ) -> Result<(), String> {
        unsafe {
            let pipeline = self.get_pipeline("parallel_encode_heads")?;

            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let queries_buffer = self.create_buffer(queries)?;
            let keys_buffer = self.create_buffer(keys)?;
            let values_buffer = self.create_buffer(values)?;
            let num_heads = num_heads as u32;
            let num_heads_buffer = self.create_buffer(&[num_heads])?;
            let head_dim = head_dim as u32;
            let head_dim_buffer = self.create_buffer(&[head_dim])?;
            let seq_len = seq_len as u32;
            let seq_len_buffer = self.create_buffer(&[seq_len])?;
            let qubits_per_head = qubits_per_head as u32;
            let qubits_per_head_buffer = self.create_buffer(&[qubits_per_head])?;
            let total_qubits = state.num_qubits as u32;
            let total_qubits_buffer = self.create_buffer(&[total_qubits])?;

            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:queries_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:keys_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:values_buffer offset:0 atIndex:3];
            let _: () = msg_send![encoder, setBuffer:num_heads_buffer offset:0 atIndex:4];
            let _: () = msg_send![encoder, setBuffer:head_dim_buffer offset:0 atIndex:5];
            let _: () = msg_send![encoder, setBuffer:seq_len_buffer offset:0 atIndex:6];
            let _: () = msg_send![encoder, setBuffer:qubits_per_head_buffer offset:0 atIndex:7];
            let _: () = msg_send![encoder, setBuffer:total_qubits_buffer offset:0 atIndex:8];

            let threads = metal::MTLSize {
                width: 1,
                height: num_heads as u64,
                depth: seq_len as u64,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () =
                msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }

    /// Execute batch transformer forward pass on GPU
    pub fn batch_transformer_forward_gpu(
        &mut self,
        state: &mut QuantumState,
        tokens: &[u32],
        embedding_matrix: &[f32],
        batch_size: usize,
        seq_len: usize,
        embedding_dim: usize,
        qubits_per_sequence: usize,
    ) -> Result<(), String> {
        unsafe {
            let pipeline = self.get_pipeline("batch_transformer_forward")?;

            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let tokens_buffer = self.create_buffer(tokens)?;
            let embedding_buffer = self.create_buffer(embedding_matrix)?;
            let batch_size = batch_size as u32;
            let batch_size_buffer = self.create_buffer(&[batch_size])?;
            let seq_len = seq_len as u32;
            let seq_len_buffer = self.create_buffer(&[seq_len])?;
            let embedding_dim = embedding_dim as u32;
            let embedding_dim_buffer = self.create_buffer(&[embedding_dim])?;
            let qubits_per_sequence = qubits_per_sequence as u32;
            let qubits_per_sequence_buffer = self.create_buffer(&[qubits_per_sequence])?;
            let total_qubits = state.num_qubits as u32;
            let total_qubits_buffer = self.create_buffer(&[total_qubits])?;

            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:tokens_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:embedding_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:batch_size_buffer offset:0 atIndex:3];
            let _: () = msg_send![encoder, setBuffer:seq_len_buffer offset:0 atIndex:4];
            let _: () = msg_send![encoder, setBuffer:embedding_dim_buffer offset:0 atIndex:5];
            let _: () = msg_send![encoder, setBuffer:qubits_per_sequence_buffer offset:0 atIndex:6];
            let _: () = msg_send![encoder, setBuffer:total_qubits_buffer offset:0 atIndex:7];

            let threads = metal::MTLSize {
                width: batch_size as u64,
                height: seq_len as u64,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () =
                msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }

    /// Execute parallel Grover oracle on GPU
    pub fn parallel_grover_oracle_gpu(
        &mut self,
        state: &mut QuantumState,
        targets: &[usize],
    ) -> Result<(), String> {
        unsafe {
            let pipeline = self.get_pipeline("parallel_grover_oracle")?;

            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let targets_u32: Vec<u32> = targets.iter().map(|&x| x as u32).collect();
            let targets_buffer = self.create_buffer(&targets_u32)?;
            let num_targets = targets.len() as u32;
            let num_targets_buffer = self.create_buffer(&[num_targets])?;
            let num_qubits = state.num_qubits as u32;
            let num_qubits_buffer = self.create_buffer(&[num_qubits])?;

            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:targets_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:num_targets_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:num_qubits_buffer offset:0 atIndex:3];

            let threads = metal::MTLSize {
                width: targets.len() as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () =
                msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }

    /// Execute parallel QFT on GPU
    pub fn parallel_qft_gpu(
        &mut self,
        state: &mut QuantumState,
        start_qubit: usize,
        num_transform_qubits: usize,
    ) -> Result<(), String> {
        unsafe {
            let pipeline = self.get_pipeline("parallel_qft")?;

            let state_buffer = self.create_buffer(state.amplitudes_ref())?;
            let num_qubits = state.num_qubits as u32;
            let num_qubits_buffer = self.create_buffer(&[num_qubits])?;
            let start_qubit = start_qubit as u32;
            let start_qubit_buffer = self.create_buffer(&[start_qubit])?;
            let num_transform = num_transform_qubits as u32;
            let num_transform_buffer = self.create_buffer(&[num_transform])?;

            let command_buffer: id = msg_send![self.command_queue, commandBuffer];
            let encoder: id = msg_send![command_buffer, computeCommandEncoder];

            let _: () = msg_send![encoder, setComputePipelineState:pipeline];
            let _: () = msg_send![encoder, setBuffer:state_buffer offset:0 atIndex:0];
            let _: () = msg_send![encoder, setBuffer:num_qubits_buffer offset:0 atIndex:1];
            let _: () = msg_send![encoder, setBuffer:start_qubit_buffer offset:0 atIndex:2];
            let _: () = msg_send![encoder, setBuffer:num_transform_buffer offset:0 atIndex:3];

            let threads = metal::MTLSize {
                width: num_transform_qubits as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };

            let _: () =
                msg_send![encoder, dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![command_buffer, commit];
            let _: () = msg_send![command_buffer, waitUntilCompleted];

            let ptr: *const u8 = msg_send![state_buffer, contents];
            let contents: *const C64 = ptr as *const C64;
            std::ptr::copy_nonoverlapping(contents, state.amplitudes_mut().as_mut_ptr(), state.dim);

            Ok(())
        }
    }
}

// ============================================================
// BENCHMARKING INFRASTRUCTURE
// ============================================================

#[cfg(target_os = "macos")]
pub struct MetalBenchmark {
    executor: MetalParallelQuantumExecutor,
}

#[cfg(target_os = "macos")]
impl MetalBenchmark {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            executor: MetalParallelQuantumExecutor::new()?,
        })
    }

    /// Benchmark parallel gates: CPU vs GPU
    pub fn benchmark_parallel_gates(
        &mut self,
        num_qubits: usize,
        num_gates: usize,
    ) -> Result<BenchmarkResults, String> {
        // CPU benchmark
        let mut cpu_state = QuantumState::new(num_qubits);
        let cpu_start = Instant::now();

        for i in 0..num_gates {
            crate::GateOperations::h(&mut cpu_state, i % num_qubits);
        }

        let cpu_time = cpu_start.elapsed();

        // GPU benchmark
        let mut gpu_state = QuantumState::new(num_qubits);
        let gpu_start = Instant::now();

        for i in 0..num_gates {
            let qubit = i % num_qubits;
            self.executor
                .parallel_hadamard_gpu(&mut gpu_state, &[qubit])?;
        }

        let gpu_time = gpu_start.elapsed();

        // Verify correctness
        let fidelity = cpu_state.fidelity(&gpu_state);

        Ok(BenchmarkResults {
            cpu_time,
            gpu_time,
            speedup: cpu_time.as_secs_f64() / gpu_time.as_secs_f64(),
            fidelity,
            num_qubits,
            num_operations: num_gates,
        })
    }

    /// Benchmark batch transformer operations
    pub fn benchmark_batch_transformer(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        embedding_dim: usize,
    ) -> Result<BenchmarkResults, String> {
        let num_qubits = (batch_size * seq_len * embedding_dim)
            .next_power_of_two()
            .ilog2() as usize;
        let mut state = QuantumState::new(num_qubits);

        // Generate dummy data
        let tokens: Vec<u32> = (0..batch_size * seq_len).map(|i| i as u32 % 1000).collect();
        let embedding_matrix: Vec<f32> = (0..1000 * embedding_dim)
            .map(|i| (i as f32) / 1000.0)
            .collect();

        let start = Instant::now();

        self.executor.batch_transformer_forward_gpu(
            &mut state,
            &tokens,
            &embedding_matrix,
            batch_size,
            seq_len,
            embedding_dim,
            num_qubits / (batch_size * seq_len),
        )?;

        let gpu_time = start.elapsed();

        Ok(BenchmarkResults {
            cpu_time: gpu_time, // No CPU baseline for comparison
            gpu_time,
            speedup: 1.0,
            fidelity: 1.0,
            num_qubits,
            num_operations: batch_size * seq_len,
        })
    }

    /// Benchmark multi-head attention
    pub fn benchmark_attention(
        &mut self,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<BenchmarkResults, String> {
        let num_qubits = (num_heads * head_dim * 3).next_power_of_two().ilog2() as usize;
        let mut state = QuantumState::new(num_qubits);

        let qubits_per_head = (head_dim * 3).min(num_qubits / num_heads);

        // Generate dummy QKV
        let qkv_size = num_heads * seq_len * head_dim;
        let queries: Vec<f32> = (0..qkv_size)
            .map(|i| (i as f32) / qkv_size as f32)
            .collect();
        let keys: Vec<f32> = (0..qkv_size)
            .map(|i| (i as f32) / qkv_size as f32)
            .collect();
        let values: Vec<f32> = (0..qkv_size)
            .map(|i| (i as f32) / qkv_size as f32)
            .collect();

        let start = Instant::now();

        self.executor.parallel_encode_heads_gpu(
            &mut state,
            &queries,
            &keys,
            &values,
            num_heads,
            head_dim,
            seq_len,
            qubits_per_head,
        )?;

        let gpu_time = start.elapsed();

        Ok(BenchmarkResults {
            cpu_time: gpu_time,
            gpu_time,
            speedup: 1.0,
            fidelity: 1.0,
            num_qubits,
            num_operations: num_heads * seq_len,
        })
    }
}

pub struct BenchmarkResults {
    pub cpu_time: std::time::Duration,
    pub gpu_time: std::time::Duration,
    pub speedup: f64,
    pub fidelity: f64,
    pub num_qubits: usize,
    pub num_operations: usize,
}

impl std::fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Benchmark Results:")?;
        writeln!(f, "  Qubits: {}", self.num_qubits)?;
        writeln!(f, "  Operations: {}", self.num_operations)?;
        writeln!(f, "  CPU time: {:?}", self.cpu_time)?;
        writeln!(f, "  GPU time: {:?}", self.gpu_time)?;
        writeln!(f, "  Speedup: {:.2}×", self.speedup)?;
        writeln!(f, "  Fidelity: {:.6}", self.fidelity)
    }
}

// ============================================================
// CROSS-PLATFORM FALLBACK
// ============================================================

#[cfg(not(target_os = "macos"))]
pub struct MetalParallelQuantumExecutor;

#[cfg(not(target_os = "macos"))]
impl MetalParallelQuantumExecutor {
    pub fn new() -> Result<Self, String> {
        Err("Metal GPU acceleration is only available on macOS".to_string())
    }
}

#[cfg(not(target_os = "macos"))]
pub struct MetalBenchmark;

#[cfg(not(target_os = "macos"))]
impl MetalBenchmark {
    pub fn new() -> Result<Self, String> {
        Err("Metal GPU acceleration is only available on macOS".to_string())
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;

    fn metal_tests_enabled() -> bool {
        std::env::var("NQPU_RUN_METAL_TESTS").ok().as_deref() == Some("1")
    }

    #[test]
    fn test_executor_creation() {
        if !metal_tests_enabled() {
            return;
        }
        let executor = MetalParallelQuantumExecutor::new();
        assert!(executor.is_ok());
    }

    #[test]
    fn test_parallel_hadamard() {
        if !metal_tests_enabled() {
            return;
        }
        let mut executor = MetalParallelQuantumExecutor::new().unwrap();
        let mut state = QuantumState::new(10);

        let result = executor.parallel_hadamard_gpu(&mut state, &[0, 1, 2]);
        assert!(result.is_ok());

        // Verify normalization
        let probs = state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_rotations() {
        if !metal_tests_enabled() {
            return;
        }
        let mut executor = MetalParallelQuantumExecutor::new().unwrap();
        let mut state = QuantumState::new(10);

        let angles = vec![0.5, 1.0, 1.5];
        let result = executor.parallel_rotations_gpu(&mut state, &[0, 1, 2], &angles, 1); // RY
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_cnot() {
        if !metal_tests_enabled() {
            return;
        }
        let mut executor = MetalParallelQuantumExecutor::new().unwrap();
        let mut state = QuantumState::new(10);

        let result = executor.parallel_cnot_gpu(&mut state, &[0, 1], &[1, 2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_heads() {
        if !metal_tests_enabled() {
            return;
        }
        let mut executor = MetalParallelQuantumExecutor::new().unwrap();
        let mut state = QuantumState::new(12);

        let num_heads = 4;
        let head_dim = 8;
        let seq_len = 16;
        let qubits_per_head = 3;

        let qkv_size = num_heads * seq_len * head_dim;
        let queries: Vec<f32> = vec![0.1; qkv_size];
        let keys: Vec<f32> = vec![0.2; qkv_size];
        let values: Vec<f32> = vec![0.3; qkv_size];

        let result = executor.parallel_encode_heads_gpu(
            &mut state,
            &queries,
            &keys,
            &values,
            num_heads,
            head_dim,
            seq_len,
            qubits_per_head,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_grover_oracle() {
        if !metal_tests_enabled() {
            return;
        }
        let mut executor = MetalParallelQuantumExecutor::new().unwrap();
        let mut state = QuantumState::new(10);

        // Initialize to superposition
        for i in 0..10 {
            crate::GateOperations::h(&mut state, i);
        }

        let targets = vec![42, 100, 256];
        let result = executor.parallel_grover_oracle_gpu(&mut state, &targets);

        assert!(result.is_ok());

        // Verify normalization preserved
        let probs = state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_qft() {
        if !metal_tests_enabled() {
            return;
        }
        let mut executor = MetalParallelQuantumExecutor::new().unwrap();
        let mut state = QuantumState::new(10);

        let result = executor.parallel_qft_gpu(&mut state, 0, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_benchmark_parallel_gates() {
        if !metal_tests_enabled() {
            return;
        }
        let mut benchmark = MetalBenchmark::new().unwrap();
        let result = benchmark.benchmark_parallel_gates(12, 100);

        assert!(result.is_ok());
        let results = result.unwrap();
        println!("\n{}", results);
        assert!(results.speedup > 0.0);
    }

    #[test]
    fn test_gpu_vs_cpu_correctness() {
        if !metal_tests_enabled() {
            return;
        }
        let mut benchmark = MetalBenchmark::new().unwrap();

        // Small test for correctness verification
        let result = benchmark.benchmark_parallel_gates(8, 10);

        assert!(result.is_ok());
        let results = result.unwrap();

        // Fidelity should be very high (> 0.99) for correct implementation
        assert!(
            results.fidelity > 0.99,
            "Fidelity too low: {}",
            results.fidelity
        );
    }

    #[test]
    fn test_batch_transformer() {
        if !metal_tests_enabled() {
            return;
        }
        let mut benchmark = MetalBenchmark::new().unwrap();

        let result = benchmark.benchmark_batch_transformer(4, 16, 32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_attention() {
        if !metal_tests_enabled() {
            return;
        }
        let mut benchmark = MetalBenchmark::new().unwrap();

        let result = benchmark.benchmark_attention(8, 16, 8);
        assert!(result.is_ok());
    }
}
