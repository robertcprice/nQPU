//! Metal GPU Backend for Quantum Simulation
//!
//! High-performance quantum simulator using Apple Metal compute shaders.
//!
//! # Design
//! - State vector lives on GPU in f32 (Metal lacks f64 compute)
//! - Shader library compiled once at init, all pipelines cached
//! - Circuit execution batches all gates into a single command buffer
//! - Uses StorageModeShared (Apple unified memory) — zero-copy between CPU/GPU
//! - Separate compute encoder per gate for implicit memory barriers
//!
//! # Usage
//! ```rust,no_run
//! use nqpu_metal::metal_backend::MetalSimulator;
//! use nqpu_metal::gates::Gate;
//!
//! let mut sim = MetalSimulator::new(10).unwrap();
//! sim.run_circuit(&[Gate::h(0), Gate::cnot(0, 1)]);
//! let probs = sim.probabilities();
//! ```

#[cfg(target_os = "macos")]
use metal::*;

use crate::gate_fusion::{self, FusedGate, FusedOrOriginal, FusedTwoQubitGate};
use crate::gates::{Gate, GateType};
use crate::thermal_scheduler::ThermalScheduler;
use crate::{c32_zero, QuantumState, C32, C64};

const DEFAULT_THREADGROUP_SIZE: u64 = 256;
const SHADER_SOURCE: &str = include_str!("metal/quantum_gates.metal");

/// Matrix passed to the generic 2x2 unitary kernel.
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuMatrix2x2 {
    m00_re: f32,
    m00_im: f32,
    m01_re: f32,
    m01_im: f32,
    m10_re: f32,
    m10_im: f32,
    m11_re: f32,
    m11_im: f32,
}

/// Matrix passed to the 4x4 unitary kernel (fused two-qubit gates).
/// Layout: 32 floats = 4x4 complex matrix, row-major, [re, im] per element.
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuMatrix4x4 {
    data: [f32; 32],
}

impl GpuMatrix4x4 {
    fn from_fusion_matrix(mat: &gate_fusion::Matrix4x4) -> Self {
        let mut data = [0.0f32; 32];
        for r in 0..4 {
            for c in 0..4 {
                data[2 * (4 * r + c)] = mat.data[r][c].re as f32;
                data[2 * (4 * r + c) + 1] = mat.data[r][c].im as f32;
            }
        }
        GpuMatrix4x4 { data }
    }
}

// ============================================================
// CACHED PIPELINES
// ============================================================

#[cfg(target_os = "macos")]
struct Pipelines {
    hadamard: ComputePipelineState,
    pauli_x: ComputePipelineState,
    pauli_y: ComputePipelineState,
    pauli_z: ComputePipelineState,
    phase_s: ComputePipelineState,
    t_gate: ComputePipelineState,
    rotation_x: ComputePipelineState,
    rotation_y: ComputePipelineState,
    rotation_z: ComputePipelineState,
    unitary: ComputePipelineState,
    unitary4x4: ComputePipelineState,
    cnot: ComputePipelineState,
    cz: ComputePipelineState,
    cphase: ComputePipelineState,
    swap: ComputePipelineState,
    toffoli: ComputePipelineState,
    probabilities: ComputePipelineState,
    init_zero: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl Pipelines {
    fn new(device: &DeviceRef, library: &LibraryRef) -> Result<Self, String> {
        let make = |name: &str| -> Result<ComputePipelineState, String> {
            let func = library
                .get_function(name, None)
                .map_err(|e| format!("Shader function '{}': {}", name, e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Pipeline '{}': {}", name, e))
        };
        Ok(Pipelines {
            hadamard: make("gpu_hadamard")?,
            pauli_x: make("gpu_pauli_x")?,
            pauli_y: make("gpu_pauli_y")?,
            pauli_z: make("gpu_pauli_z")?,
            phase_s: make("gpu_phase_s")?,
            t_gate: make("gpu_t_gate")?,
            rotation_x: make("gpu_rotation_x")?,
            rotation_y: make("gpu_rotation_y")?,
            rotation_z: make("gpu_rotation_z")?,
            unitary: make("gpu_unitary")?,
            unitary4x4: make("gpu_unitary4x4")?,
            cnot: make("gpu_cnot")?,
            cz: make("gpu_cz")?,
            cphase: make("gpu_cphase")?,
            swap: make("gpu_swap")?,
            toffoli: make("gpu_toffoli")?,
            probabilities: make("gpu_probabilities")?,
            init_zero: make("gpu_init_zero")?,
        })
    }
}

// ============================================================
// METAL SIMULATOR
// ============================================================

#[cfg(target_os = "macos")]
pub struct MetalSimulator {
    device: Device,
    queue: CommandQueue,
    pipelines: Pipelines,
    state_buffer: Buffer,
    prob_buffer: Buffer,
    num_qubits: usize,
    dim: usize,
    /// Adaptive threadgroup size based on pipeline query and thermal state.
    threadgroup_size: u64,
    /// Thermal-aware scheduler for performance consistency.
    thermal_scheduler: ThermalScheduler,
}

#[cfg(target_os = "macos")]
impl MetalSimulator {
    /// Create a new GPU simulator for `num_qubits` qubits.
    /// State is initialized to |0...0⟩ on the GPU.
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        // Compile shaders once
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|e| format!("Shader compilation failed: {}", e))?;

        let pipelines = Pipelines::new(&device, &library)?;

        let dim = 1usize << num_qubits;
        let state_bytes = (dim * std::mem::size_of::<C32>()) as u64;
        let prob_bytes = (dim * std::mem::size_of::<f32>()) as u64;

        // Allocate in shared memory (unified on Apple Silicon — zero-copy)
        let state_buffer = device.new_buffer(state_bytes, MTLResourceOptions::StorageModeShared);
        let prob_buffer = device.new_buffer(prob_bytes, MTLResourceOptions::StorageModeShared);

        // Phase 1C: Adaptive threadgroup sizing
        // Query pipeline capability and adapt to state size
        let max_tg = pipelines.hadamard.max_total_threads_per_threadgroup();
        let initial_tg = if dim < 4096 {
            64.min(max_tg) // small states: minimize overhead
        } else if dim < 1048576 {
            256.min(max_tg) // medium states: balanced
        } else {
            512.min(max_tg) // large states: maximize occupancy
        };

        // Create thermal scheduler for consistent performance
        let thermal_scheduler = ThermalScheduler::new(initial_tg, max_tg, 32);

        let mut sim = MetalSimulator {
            device,
            queue,
            pipelines,
            state_buffer,
            prob_buffer,
            num_qubits,
            dim,
            threadgroup_size: initial_tg,
            thermal_scheduler,
        };

        // Initialize to |0⟩ on GPU
        sim.reset();
        Ok(sim)
    }

    /// Device name (e.g. "Apple M4 Pro")
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// State vector dimension (2^n)
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Reset state to |0...0⟩ using GPU kernel.
    pub fn reset(&mut self) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.init_zero);
        enc.set_buffer(0, Some(&self.state_buffer), 0);
        let ns = self.dim as u32;
        enc.set_bytes(1, 4, &ns as *const u32 as *const _);
        self.dispatch_dim(enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // --------------------------------------------------------
    // CIRCUIT EXECUTION
    // --------------------------------------------------------

    /// Run a complete circuit on GPU. All gates are encoded into a single
    /// command buffer with one encoder per gate (implicit memory barriers).
    pub fn run_circuit(&self, gates: &[Gate]) {
        if gates.is_empty() {
            return;
        }

        let cmd = self.queue.new_command_buffer();
        for gate in gates {
            let enc = cmd.new_compute_command_encoder();
            self.encode_gate(enc, gate);
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Run a circuit with layer-based batching (Phase 1A).
    /// Gates are first fused, then decomposed into layers of non-conflicting gates.
    /// All gates in a layer share one encoder → fewer memory barriers.
    pub fn run_circuit_batched(&self, gates: &[Gate]) {
        if gates.is_empty() {
            return;
        }

        let fusion = gate_fusion::fuse_gates(gates);
        let layers = gate_fusion::decompose_fused_layers(&fusion.gates);

        let cmd = self.queue.new_command_buffer();
        for layer in &layers {
            let enc = cmd.new_compute_command_encoder();
            for item in layer {
                self.encode_fused_item(enc, item);
            }
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Encode a fused-or-original gate item into a compute encoder.
    fn encode_fused_item(&self, enc: &ComputeCommandEncoderRef, item: &FusedOrOriginal) {
        match item {
            FusedOrOriginal::Original(gate) => {
                self.encode_gate(enc, gate);
            }
            FusedOrOriginal::Fused(fg) => {
                self.encode_fused_1q(enc, fg);
            }
            FusedOrOriginal::FusedTwo(fg) => {
                self.encode_fused_2q(enc, fg);
            }
            FusedOrOriginal::Diagonal(diag) => {
                // Encode each single-qubit diagonal as phase gate via generic unitary pipeline.
                for &(qubit, phase0, phase1) in &diag.single_phases {
                    let mat = GpuMatrix2x2 {
                        m00_re: phase0.re as f32,
                        m00_im: phase0.im as f32,
                        m01_re: 0.0,
                        m01_im: 0.0,
                        m10_re: 0.0,
                        m10_im: 0.0,
                        m11_re: phase1.re as f32,
                        m11_im: phase1.im as f32,
                    };
                    let ns = self.dim as u32;
                    let q = qubit as u32;
                    enc.set_compute_pipeline_state(&self.pipelines.unitary);
                    enc.set_buffer(0, Some(&self.state_buffer), 0);
                    enc.set_bytes(1, 4, &q as *const u32 as *const _);
                    enc.set_bytes(
                        2,
                        std::mem::size_of::<GpuMatrix2x2>() as u64,
                        &mat as *const GpuMatrix2x2 as *const _,
                    );
                    enc.set_bytes(3, 4, &ns as *const u32 as *const _);
                    self.dispatch_half(enc);
                }
                // Two-qubit controlled phases encoded via 4x4 unitary kernel.
                for &(q_lo, q_hi, phase11) in &diag.pair_phases {
                    let o = C64::new(1.0, 0.0);
                    let z = C64::new(0.0, 0.0);
                    // diag(1, 1, 1, phase11) in computational basis ordered by (lo, hi)
                    let mat4x4 = crate::gate_fusion::Matrix4x4 {
                        data: [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, phase11]],
                    };
                    let fg = crate::gate_fusion::FusedTwoQubitGate {
                        qubit_lo: q_lo,
                        qubit_hi: q_hi,
                        matrix: mat4x4,
                        fused_count: 1,
                    };
                    self.encode_fused_2q(enc, &fg);
                }
            }
        }
    }

    /// Encode a fused single-qubit gate via the generic 2x2 unitary kernel.
    fn encode_fused_1q(&self, enc: &ComputeCommandEncoderRef, fg: &FusedGate) {
        let ns = self.dim as u32;
        let mat = GpuMatrix2x2 {
            m00_re: fg.matrix.data[0][0].re as f32,
            m00_im: fg.matrix.data[0][0].im as f32,
            m01_re: fg.matrix.data[0][1].re as f32,
            m01_im: fg.matrix.data[0][1].im as f32,
            m10_re: fg.matrix.data[1][0].re as f32,
            m10_im: fg.matrix.data[1][0].im as f32,
            m11_re: fg.matrix.data[1][1].re as f32,
            m11_im: fg.matrix.data[1][1].im as f32,
        };
        enc.set_compute_pipeline_state(&self.pipelines.unitary);
        enc.set_buffer(0, Some(&self.state_buffer), 0);
        let q = fg.target as u32;
        enc.set_bytes(1, 4, &q as *const u32 as *const _);
        enc.set_bytes(
            2,
            std::mem::size_of::<GpuMatrix2x2>() as u64,
            &mat as *const GpuMatrix2x2 as *const _,
        );
        enc.set_bytes(3, 4, &ns as *const u32 as *const _);
        self.dispatch_half(enc);
    }

    /// Encode a fused two-qubit gate via the 4x4 unitary kernel (Phase 1B).
    fn encode_fused_2q(&self, enc: &ComputeCommandEncoderRef, fg: &FusedTwoQubitGate) {
        let ns = self.dim as u32;
        let gpu_mat = GpuMatrix4x4::from_fusion_matrix(&fg.matrix);
        let q_lo = fg.qubit_lo as u32;
        let q_hi = fg.qubit_hi as u32;

        enc.set_compute_pipeline_state(&self.pipelines.unitary4x4);
        enc.set_buffer(0, Some(&self.state_buffer), 0);
        enc.set_bytes(1, 4, &q_lo as *const u32 as *const _);
        enc.set_bytes(2, 4, &q_hi as *const u32 as *const _);
        enc.set_bytes(
            3,
            std::mem::size_of::<GpuMatrix4x4>() as u64,
            &gpu_mat as *const GpuMatrix4x4 as *const _,
        );
        enc.set_bytes(4, 4, &ns as *const u32 as *const _);
        self.dispatch_quarter(enc);
    }

    /// Encode a single gate into a compute encoder.
    fn encode_gate(&self, enc: &ComputeCommandEncoderRef, gate: &Gate) {
        let ns = self.dim as u32;
        match &gate.gate_type {
            // ---- Pair-indexed single-qubit gates (dim/2 threads) ----
            GateType::H => {
                enc.set_compute_pipeline_state(&self.pipelines.hadamard);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &ns as *const u32 as *const _);
                self.dispatch_half(enc);
            }
            GateType::X => {
                enc.set_compute_pipeline_state(&self.pipelines.pauli_x);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &ns as *const u32 as *const _);
                self.dispatch_half(enc);
            }
            GateType::Y => {
                enc.set_compute_pipeline_state(&self.pipelines.pauli_y);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &ns as *const u32 as *const _);
                self.dispatch_half(enc);
            }
            GateType::Rx(theta) => {
                enc.set_compute_pipeline_state(&self.pipelines.rotation_x);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                let th = *theta as f32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &th as *const f32 as *const _);
                enc.set_bytes(3, 4, &ns as *const u32 as *const _);
                self.dispatch_half(enc);
            }
            GateType::Ry(theta) => {
                enc.set_compute_pipeline_state(&self.pipelines.rotation_y);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                let th = *theta as f32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &th as *const f32 as *const _);
                enc.set_bytes(3, 4, &ns as *const u32 as *const _);
                self.dispatch_half(enc);
            }

            // ---- Diagonal single-qubit gates (dim threads) ----
            GateType::Z => {
                enc.set_compute_pipeline_state(&self.pipelines.pauli_z);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }
            GateType::S => {
                enc.set_compute_pipeline_state(&self.pipelines.phase_s);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }
            GateType::T => {
                enc.set_compute_pipeline_state(&self.pipelines.t_gate);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }
            GateType::Rz(theta) => {
                enc.set_compute_pipeline_state(&self.pipelines.rotation_z);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q = gate.targets[0] as u32;
                let ph = *theta as f32;
                enc.set_bytes(1, 4, &q as *const u32 as *const _);
                enc.set_bytes(2, 4, &ph as *const f32 as *const _);
                enc.set_bytes(3, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }

            // ---- Two-qubit gates (dim threads) ----
            GateType::CNOT => {
                enc.set_compute_pipeline_state(&self.pipelines.cnot);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let ctrl = gate.controls[0] as u32;
                let tgt = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &ctrl as *const u32 as *const _);
                enc.set_bytes(2, 4, &tgt as *const u32 as *const _);
                enc.set_bytes(3, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }
            GateType::CZ => {
                enc.set_compute_pipeline_state(&self.pipelines.cz);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let ctrl = gate.controls[0] as u32;
                let tgt = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &ctrl as *const u32 as *const _);
                enc.set_bytes(2, 4, &tgt as *const u32 as *const _);
                enc.set_bytes(3, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }
            GateType::SWAP => {
                enc.set_compute_pipeline_state(&self.pipelines.swap);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let q1 = gate.targets[0] as u32;
                let q2 = gate.targets[1] as u32;
                enc.set_bytes(1, 4, &q1 as *const u32 as *const _);
                enc.set_bytes(2, 4, &q2 as *const u32 as *const _);
                enc.set_bytes(3, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }

            // ---- Controlled-phase (dim threads) ----
            GateType::CR(angle) => {
                enc.set_compute_pipeline_state(&self.pipelines.cphase);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let ctrl = gate.controls[0] as u32;
                let tgt = gate.targets[0] as u32;
                let ph = *angle as f32;
                enc.set_bytes(1, 4, &ctrl as *const u32 as *const _);
                enc.set_bytes(2, 4, &tgt as *const u32 as *const _);
                enc.set_bytes(3, 4, &ph as *const f32 as *const _);
                enc.set_bytes(4, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }
            GateType::CRz(angle) => {
                // CRz = controlled phase on |11⟩ with exp(i*angle/2) relative phase
                enc.set_compute_pipeline_state(&self.pipelines.cphase);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let ctrl = gate.controls[0] as u32;
                let tgt = gate.targets[0] as u32;
                let ph = *angle as f32;
                enc.set_bytes(1, 4, &ctrl as *const u32 as *const _);
                enc.set_bytes(2, 4, &tgt as *const u32 as *const _);
                enc.set_bytes(3, 4, &ph as *const f32 as *const _);
                enc.set_bytes(4, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }

            // ---- Three-qubit (dim threads) ----
            GateType::Toffoli => {
                enc.set_compute_pipeline_state(&self.pipelines.toffoli);
                enc.set_buffer(0, Some(&self.state_buffer), 0);
                let c1 = gate.controls[0] as u32;
                let c2 = gate.controls[1] as u32;
                let tgt = gate.targets[0] as u32;
                enc.set_bytes(1, 4, &c1 as *const u32 as *const _);
                enc.set_bytes(2, 4, &c2 as *const u32 as *const _);
                enc.set_bytes(3, 4, &tgt as *const u32 as *const _);
                enc.set_bytes(4, 4, &ns as *const u32 as *const _);
                self.dispatch_dim(enc);
            }

            // ---- Fallback: generic unitary via matrix ----
            _ => {
                self.encode_generic_unitary(enc, gate);
            }
        }
    }

    /// Fallback: compute the gate matrix and apply via generic unitary kernel.
    fn encode_generic_unitary(&self, enc: &ComputeCommandEncoderRef, gate: &Gate) {
        let ns = self.dim as u32;

        if gate.is_single_qubit() {
            let mat_vecs = gate.gate_type.matrix();
            let mat = GpuMatrix2x2 {
                m00_re: mat_vecs[0][0].re as f32,
                m00_im: mat_vecs[0][0].im as f32,
                m01_re: mat_vecs[0][1].re as f32,
                m01_im: mat_vecs[0][1].im as f32,
                m10_re: mat_vecs[1][0].re as f32,
                m10_im: mat_vecs[1][0].im as f32,
                m11_re: mat_vecs[1][1].re as f32,
                m11_im: mat_vecs[1][1].im as f32,
            };
            enc.set_compute_pipeline_state(&self.pipelines.unitary);
            enc.set_buffer(0, Some(&self.state_buffer), 0);
            let q = gate.targets[0] as u32;
            enc.set_bytes(1, 4, &q as *const u32 as *const _);
            enc.set_bytes(
                2,
                std::mem::size_of::<GpuMatrix2x2>() as u64,
                &mat as *const GpuMatrix2x2 as *const _,
            );
            enc.set_bytes(3, 4, &ns as *const u32 as *const _);
            self.dispatch_half(enc);
        }
        // Multi-qubit gates without specialized kernels: decompose or skip
        // (all common 2-qubit gates are handled above)
    }

    // --------------------------------------------------------
    // DISPATCH HELPERS (Phase 1C: adaptive threadgroup sizing)
    // --------------------------------------------------------

    /// Dispatch dim/2 threads (for pair-indexed gates).
    fn dispatch_half(&self, enc: &ComputeCommandEncoderRef) {
        let n = (self.dim / 2) as u64;
        let tg = self.threadgroup_size;
        let groups = (n + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// Dispatch dim threads (for diagonal/full-state gates).
    fn dispatch_dim(&self, enc: &ComputeCommandEncoderRef) {
        let n = self.dim as u64;
        let tg = self.threadgroup_size;
        let groups = (n + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// Dispatch dim/4 threads (for 4x4 unitary / fused two-qubit gates).
    fn dispatch_quarter(&self, enc: &ComputeCommandEncoderRef) {
        let n = (self.dim / 4) as u64;
        let tg = self.threadgroup_size;
        let groups = (n + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }

    // --------------------------------------------------------
    // STATE READOUT
    // --------------------------------------------------------

    /// Read state vector from GPU as f32 complex numbers.
    pub fn read_state_f32(&self) -> Vec<C32> {
        let mut out = vec![c32_zero(); self.dim];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.state_buffer.contents() as *const C32,
                out.as_mut_ptr(),
                self.dim,
            );
        }
        out
    }

    /// Read state vector from GPU, converting to f64.
    pub fn read_state(&self) -> Vec<C64> {
        self.read_state_f32()
            .into_iter()
            .map(|c| C64::new(c.re as f64, c.im as f64))
            .collect()
    }

    /// Overwrite state vector on GPU from f32 amplitudes.
    pub fn write_state_f32(&mut self, amplitudes: &[C32]) -> Result<(), String> {
        if amplitudes.len() != self.dim {
            return Err(format!(
                "state length mismatch: got {}, expected {}",
                amplitudes.len(),
                self.dim
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                amplitudes.as_ptr(),
                self.state_buffer.contents() as *mut C32,
                self.dim,
            );
        }
        Ok(())
    }

    /// Overwrite state vector on GPU from f64 amplitudes.
    pub fn write_state(&mut self, amplitudes: &[C64]) -> Result<(), String> {
        if amplitudes.len() != self.dim {
            return Err(format!(
                "state length mismatch: got {}, expected {}",
                amplitudes.len(),
                self.dim
            ));
        }
        let mut converted = vec![c32_zero(); self.dim];
        for (dst, src) in converted.iter_mut().zip(amplitudes.iter()) {
            *dst = C32::new(src.re as f32, src.im as f32);
        }
        self.write_state_f32(&converted)
    }

    /// Compute probabilities on GPU and read back.
    pub fn probabilities(&self) -> Vec<f32> {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.probabilities);
        enc.set_buffer(0, Some(&self.state_buffer), 0);
        enc.set_buffer(1, Some(&self.prob_buffer), 0);
        let ns = self.dim as u32;
        enc.set_bytes(2, 4, &ns as *const u32 as *const _);
        self.dispatch_dim(enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut probs = vec![0.0f32; self.dim];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.prob_buffer.contents() as *const f32,
                probs.as_mut_ptr(),
                self.dim,
            );
        }
        probs
    }

    /// Compute fidelity against a CPU QuantumState (f64).
    /// Converts GPU f32 state to f64 and computes |⟨gpu|cpu⟩|².
    pub fn fidelity_vs_cpu(&self, cpu_state: &QuantumState) -> f64 {
        let gpu_amps = self.read_state();
        let cpu_amps = cpu_state.amplitudes_ref();
        assert_eq!(gpu_amps.len(), cpu_amps.len());

        let mut inner_re = 0.0f64;
        let mut inner_im = 0.0f64;
        for (g, c) in gpu_amps.iter().zip(cpu_amps.iter()) {
            // <gpu|cpu> = conj(gpu) * cpu
            inner_re += g.re * c.re + g.im * c.im;
            inner_im += g.re * c.im - g.im * c.re;
        }
        inner_re * inner_re + inner_im * inner_im
    }

    // ============================================================
    // THERMAL-AWARE EXECUTION
    // ============================================================

    /// Run circuit with thermal-aware scheduling.
    ///
    /// Automatically adjusts threadgroup size based on performance to
    /// maintain consistent execution under varying thermal conditions.
    pub fn run_circuit_thermal(&mut self, gates: &[Gate]) -> f64 {
        let start = std::time::Instant::now();
        self.run_circuit_batched(gates);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update thermal scheduler and adjust threadgroup size if needed
        if let Some(new_size) = self.thermal_scheduler.adjust(elapsed_ms) {
            self.threadgroup_size = new_size;
        }

        elapsed_ms
    }

    /// Get current thermal state for debugging/monitoring.
    pub fn thermal_state(&self) -> crate::thermal_scheduler::ThermalState {
        self.thermal_scheduler.thermal_state()
    }

    /// Reset thermal scheduler (e.g., after long idle period).
    pub fn reset_thermal_scheduler(&mut self) {
        self.thermal_scheduler.reset(self.threadgroup_size);
    }

    /// Manually set threadgroup size (for advanced tuning).
    pub fn set_threadgroup_size(&mut self, size: u64) {
        let max = self.thermal_scheduler.max_threadgroup_size();
        self.threadgroup_size = size.min(max);
    }

    /// Get current threadgroup size.
    pub fn get_threadgroup_size(&self) -> u64 {
        self.threadgroup_size
    }
}

// ============================================================
// GPU BENCHMARK HARNESS
// ============================================================

/// Result of a GPU vs CPU benchmark.
#[derive(Clone, Debug)]
pub struct GpuBenchmarkResult {
    pub name: String,
    pub num_qubits: usize,
    pub num_gates: usize,
    pub cpu_time_ms: f64,
    pub gpu_time_ms: f64,
    pub speedup: f64,
    pub fidelity: f64,
}

impl std::fmt::Display for GpuBenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<20} n={:<3} gates={:<5} cpu={:>8.2}ms gpu={:>8.2}ms {:>6.2}x fid={:.8}",
            self.name,
            self.num_qubits,
            self.num_gates,
            self.cpu_time_ms,
            self.gpu_time_ms,
            self.speedup,
            self.fidelity
        )
    }
}

/// Run a GPU vs CPU benchmark on a circuit.
#[cfg(target_os = "macos")]
pub fn run_gpu_benchmark(
    name: &str,
    gates: &[Gate],
    num_qubits: usize,
) -> Result<GpuBenchmarkResult, String> {
    use crate::ascii_viz::apply_gate_to_state;
    use std::time::Instant;

    // CPU execution (f64, fused)
    let start = Instant::now();
    let mut cpu_state = QuantumState::new(num_qubits);
    for gate in gates {
        apply_gate_to_state(&mut cpu_state, gate);
    }
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    // GPU execution (f32, batched)
    let mut sim = MetalSimulator::new(num_qubits)?;

    // Warmup run
    sim.run_circuit(gates);
    sim.reset();

    // Timed run
    let start = Instant::now();
    sim.run_circuit(gates);
    let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

    let fidelity = sim.fidelity_vs_cpu(&cpu_state);
    let speedup = if gpu_time > 0.0 {
        cpu_time / gpu_time
    } else {
        f64::INFINITY
    };

    Ok(GpuBenchmarkResult {
        name: name.to_string(),
        num_qubits,
        num_gates: gates.len(),
        cpu_time_ms: cpu_time,
        gpu_time_ms: gpu_time,
        speedup,
        fidelity,
    })
}

/// Run the full GPU vs CPU benchmark suite.
#[cfg(target_os = "macos")]
pub fn run_gpu_suite() -> Vec<GpuBenchmarkResult> {
    use crate::benchmark_suite::*;

    let mut results = Vec::new();

    // QFT benchmarks
    for &n in &[10, 15, 20, 22, 24] {
        let gates = qft_circuit(n);
        match run_gpu_benchmark(&format!("QFT-{}", n), &gates, n) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("QFT-{} failed: {}", n, e),
        }
    }

    // Random circuit benchmarks
    for &n in &[10, 15, 20, 22] {
        let gates = random_circuit(n, 20, 42);
        match run_gpu_benchmark(&format!("Random-{}", n), &gates, n) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("Random-{} failed: {}", n, e),
        }
    }

    // Grover benchmarks
    for &n in &[10, 14, 18] {
        let target = (1 << n) / 2;
        let gates = grover_circuit(n, target);
        match run_gpu_benchmark(&format!("Grover-{}", n), &gates, n) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("Grover-{} failed: {}", n, e),
        }
    }

    // Bell chain benchmarks
    for &n in &[10, 20, 25] {
        let gates = bell_circuit(n);
        match run_gpu_benchmark(&format!("Bell-{}", n), &gates, n) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("Bell-{} failed: {}", n, e),
        }
    }

    results
}

/// Print formatted GPU benchmark results.
#[cfg(target_os = "macos")]
pub fn print_gpu_results(results: &[GpuBenchmarkResult]) {
    println!("{}", "=".repeat(100));
    println!("nQPU-Metal GPU vs CPU BENCHMARK SUITE");
    if let Ok(sim) = MetalSimulator::new(2) {
        println!("GPU Device: {}", sim.device_name());
    }
    println!("{}", "=".repeat(100));
    println!(
        "{:<20} {:<5} {:<7} {:>10} {:>10} {:>8} {}",
        "Benchmark", "n", "Gates", "CPU(ms)", "GPU(ms)", "Speedup", "Fidelity"
    );
    println!("{}", "-".repeat(100));

    for r in results {
        println!("{}", r);
    }

    println!("{}", "-".repeat(100));
    let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    let min_fidelity = results
        .iter()
        .map(|r| r.fidelity)
        .fold(f64::INFINITY, f64::min);
    let gpu_faster: usize = results.iter().filter(|r| r.speedup > 1.0).count();

    println!("Average speedup:    {:.2}x", avg_speedup);
    println!(
        "GPU faster in:      {}/{} benchmarks",
        gpu_faster,
        results.len()
    );
    println!("Minimum fidelity:   {:.8}", min_fidelity);
    println!(
        "Fidelity check:     {} (threshold: 0.999 for f32)",
        if min_fidelity > 0.999 { "PASS" } else { "FAIL" }
    );
}

// ============================================================
// CROSS-PLATFORM STUB
// ============================================================

#[cfg(not(target_os = "macos"))]
pub struct MetalSimulator;

#[cfg(not(target_os = "macos"))]
impl MetalSimulator {
    pub fn new(_num_qubits: usize) -> Result<Self, String> {
        Err("Metal GPU is only available on macOS".to_string())
    }

    pub fn write_state_f32(&mut self, _amplitudes: &[C32]) -> Result<(), String> {
        Err("Metal GPU is only available on macOS".to_string())
    }

    pub fn write_state(&mut self, _amplitudes: &[C64]) -> Result<(), String> {
        Err("Metal GPU is only available on macOS".to_string())
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;
    use crate::GateOperations;

    #[test]
    fn test_metal_simulator_creation() {
        let sim = MetalSimulator::new(4);
        match &sim {
            Ok(s) => println!("GPU: {}", s.device_name()),
            Err(e) => panic!("MetalSimulator::new failed: {}", e),
        }
        let sim = sim.unwrap();
        assert_eq!(sim.dim(), 16);
    }

    #[test]
    fn test_gpu_init_zero() {
        let sim = MetalSimulator::new(3).unwrap();
        let state = sim.read_state_f32();
        assert!((state[0].re - 1.0).abs() < 1e-6);
        for i in 1..8 {
            assert!(state[i].re.abs() < 1e-6);
            assert!(state[i].im.abs() < 1e-6);
        }
    }

    #[test]
    fn test_gpu_hadamard_single() {
        let sim = MetalSimulator::new(1).unwrap();
        sim.run_circuit(&[Gate::h(0)]);
        let probs = sim.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-5);
        assert!((probs[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_bell_state() {
        let sim = MetalSimulator::new(2).unwrap();
        sim.run_circuit(&[Gate::h(0), Gate::cnot(0, 1)]);
        let probs = sim.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-5, "|00⟩ prob = {}", probs[0]);
        assert!(probs[1].abs() < 1e-5, "|01⟩ prob = {}", probs[1]);
        assert!(probs[2].abs() < 1e-5, "|10⟩ prob = {}", probs[2]);
        assert!((probs[3] - 0.5).abs() < 1e-5, "|11⟩ prob = {}", probs[3]);
    }

    #[test]
    fn test_gpu_x_gate() {
        let sim = MetalSimulator::new(2).unwrap();
        sim.run_circuit(&[Gate::x(0)]);
        let probs = sim.probabilities();
        // |00⟩ → |01⟩
        assert!(probs[0].abs() < 1e-5);
        assert!((probs[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_z_gate() {
        let sim = MetalSimulator::new(1).unwrap();
        // H then Z then H should give X
        sim.run_circuit(&[Gate::h(0), Gate::z(0), Gate::h(0)]);
        let probs = sim.probabilities();
        // Result should be |1⟩
        assert!(probs[0].abs() < 1e-5);
        assert!((probs[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_swap() {
        let sim = MetalSimulator::new(2).unwrap();
        // Put qubit 0 in |1⟩, then SWAP → qubit 1 in |1⟩
        sim.run_circuit(&[Gate::x(0), Gate::swap(0, 1)]);
        let probs = sim.probabilities();
        // Should be |10⟩ = index 2
        assert!((probs[2] - 1.0).abs() < 1e-5, "probs = {:?}", probs);
    }

    #[test]
    fn test_gpu_toffoli() {
        let sim = MetalSimulator::new(3).unwrap();
        // Set controls to |1⟩, Toffoli should flip target
        sim.run_circuit(&[Gate::x(0), Gate::x(1), Gate::toffoli(0, 1, 2)]);
        let probs = sim.probabilities();
        // |110⟩ → |111⟩ = index 7
        assert!((probs[7] - 1.0).abs() < 1e-5, "probs[7] = {}", probs[7]);
    }

    #[test]
    fn test_gpu_fidelity_qft10() {
        let gates = crate::benchmark_suite::qft_circuit(10);
        let sim = MetalSimulator::new(10).unwrap();
        sim.run_circuit(&gates);

        let mut cpu = QuantumState::new(10);
        for g in &gates {
            crate::ascii_viz::apply_gate_to_state(&mut cpu, g);
        }

        let fid = sim.fidelity_vs_cpu(&cpu);
        assert!(fid > 0.999, "QFT-10 GPU/CPU fidelity too low: {:.8}", fid);
    }

    #[test]
    fn test_gpu_fidelity_random20() {
        let gates = crate::benchmark_suite::random_circuit(12, 20, 42);
        let sim = MetalSimulator::new(12).unwrap();
        sim.run_circuit(&gates);

        let mut cpu = QuantumState::new(12);
        for g in &gates {
            crate::ascii_viz::apply_gate_to_state(&mut cpu, g);
        }

        let fid = sim.fidelity_vs_cpu(&cpu);
        assert!(
            fid > 0.999,
            "Random-12 GPU/CPU fidelity too low: {:.8}",
            fid
        );
    }

    #[test]
    fn test_gpu_normalization_preserved() {
        let gates = crate::benchmark_suite::grover_circuit(8, 42);
        let sim = MetalSimulator::new(8).unwrap();
        sim.run_circuit(&gates);
        let probs = sim.probabilities();
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Normalization violated: total = {}",
            total
        );
    }

    #[test]
    fn test_gpu_cr_gate() {
        // Test controlled-rotation (used heavily in QFT)
        let gates = vec![
            Gate::h(0),
            Gate::new(GateType::CR(std::f64::consts::PI / 2.0), vec![0], vec![1]),
        ];
        let sim = MetalSimulator::new(2).unwrap();
        sim.run_circuit(&gates);

        let mut cpu = QuantumState::new(2);
        for g in &gates {
            crate::ascii_viz::apply_gate_to_state(&mut cpu, g);
        }

        let fid = sim.fidelity_vs_cpu(&cpu);
        assert!(fid > 0.999, "CR gate fidelity: {:.8}", fid);
    }

    #[test]
    fn test_gpu_batched_bell_state() {
        let sim = MetalSimulator::new(2).unwrap();
        sim.run_circuit_batched(&[Gate::h(0), Gate::cnot(0, 1)]);
        let probs = sim.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-4, "|00⟩ prob = {}", probs[0]);
        assert!(probs[1].abs() < 1e-4, "|01⟩ prob = {}", probs[1]);
        assert!(probs[2].abs() < 1e-4, "|10⟩ prob = {}", probs[2]);
        assert!((probs[3] - 0.5).abs() < 1e-4, "|11⟩ prob = {}", probs[3]);
    }

    #[test]
    fn test_gpu_batched_fidelity_qft10() {
        let gates = crate::benchmark_suite::qft_circuit(10);
        let sim = MetalSimulator::new(10).unwrap();
        sim.run_circuit_batched(&gates);

        let mut cpu = QuantumState::new(10);
        for g in &gates {
            crate::ascii_viz::apply_gate_to_state(&mut cpu, g);
        }

        let fid = sim.fidelity_vs_cpu(&cpu);
        assert!(
            fid > 0.999,
            "Batched QFT-10 GPU/CPU fidelity too low: {:.8}",
            fid
        );
    }

    #[test]
    fn test_gpu_batched_vs_sequential_fidelity() {
        // Verify batched dispatch produces same results as sequential
        let gates = crate::benchmark_suite::random_circuit(8, 20, 42);
        let sim1 = MetalSimulator::new(8).unwrap();
        sim1.run_circuit(&gates);
        let state1 = sim1.read_state();

        let sim2 = MetalSimulator::new(8).unwrap();
        sim2.run_circuit_batched(&gates);
        let state2 = sim2.read_state();

        // Compute fidelity between the two GPU results
        let mut inner_re = 0.0f64;
        let mut inner_im = 0.0f64;
        for (a, b) in state1.iter().zip(state2.iter()) {
            inner_re += a.re * b.re + a.im * b.im;
            inner_im += a.re * b.im - a.im * b.re;
        }
        let fid = inner_re * inner_re + inner_im * inner_im;
        assert!(fid > 0.999, "Batched vs sequential fidelity: {:.8}", fid);
    }

    #[test]
    fn test_gpu_high_qubit_cnot() {
        // Regression test: CNOT on high qubit indices
        let n = 8;
        let sim = MetalSimulator::new(n).unwrap();
        sim.run_circuit(&[Gate::x(6), Gate::cnot(6, 7)]);
        let probs = sim.probabilities();
        // |11000000⟩ = bits 6,7 set = index 192
        let expected_idx = (1 << 6) | (1 << 7);
        assert!(
            (probs[expected_idx] - 1.0).abs() < 1e-5,
            "CNOT on high qubits failed: probs[{}] = {}",
            expected_idx,
            probs[expected_idx]
        );
    }
}
