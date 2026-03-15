//! Metal GPU Backend for Pulse-Level Quantum Simulation
//!
//! Dispatches Lindblad master equation integration, matrix multiplication,
//! matrix exponentials, and gate fidelity computations to Apple Metal compute
//! shaders for hardware-accelerated pulse simulation.
//!
//! # Architecture
//!
//! - Shader source is embedded via `include_str!` and compiled once at init
//! - One `ComputePipelineState` is cached per kernel for zero-overhead dispatch
//! - Data flows: CPU `DenseMatrix` (f64) -> GPU buffer (f32) -> GPU compute -> read back -> f64
//! - Uses `StorageModeShared` (Apple Silicon unified memory) for zero-copy transfers
//!
//! # Kernels
//!
//! | Kernel                | Purpose                                        | Grid          |
//! |-----------------------|------------------------------------------------|---------------|
//! | `complex_matmul`      | NxN complex matrix multiplication              | (N, N)        |
//! | `complex_matmul_tiled`| Tiled matmul with threadgroup memory           | (N_pad, N_pad)|
//! | `lindblad_rhs`        | Lindblad master equation RHS                   | (N, N)        |
//! | `batch_rk4_lindblad`  | Batched RK4 Lindblad steps (for GRAPE)         | M             |
//! | `hermitian_expm`      | Matrix exp via eigendecomposition               | (N, N)        |
//! | `batch_gate_fidelity` | Parallel gate fidelity computation             | M             |
//!
//! # Example
//!
//! ```rust,ignore
//! use nqpu_metal::metal_pulse_backend::MetalPulseBackend;
//! use nqpu_metal::pulse_simulation::DenseMatrix;
//!
//! let backend = MetalPulseBackend::new().unwrap();
//! let a = DenseMatrix::identity(4);
//! let b = DenseMatrix::identity(4);
//! let c = backend.matmul_gpu(&a, &b);
//! ```

#[cfg(target_os = "macos")]
use metal::*;

use crate::pulse_simulation::{DenseMatrix, PulseError};
use num_complex::Complex64;

/// Embedded Metal shader source for pulse-level simulation kernels.
#[cfg(target_os = "macos")]
const PULSE_SHADER_SOURCE: &str = include_str!("../../metal/pulse_kernels.metal");

/// Default threadgroup width for 2D dispatch kernels.
const THREADGROUP_2D: u64 = 16;

/// Default threadgroup width for 1D dispatch kernels.
const THREADGROUP_1D: u64 = 256;

/// Maximum matrix dimension supported by the batch RK4 kernel
/// (limited by thread-private array size in the shader: 64 = 8*8).
const MAX_BATCH_RK4_DIM: usize = 8;

// ============================================================
// GPU-SIDE DATA LAYOUT
// ============================================================

/// GPU-friendly complex number (f32 precision).
/// Must match the `Complex` struct in `pulse_kernels.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuComplex {
    re: f32,
    im: f32,
}

impl GpuComplex {
    fn from_c64(c: Complex64) -> Self {
        GpuComplex {
            re: c.re as f32,
            im: c.im as f32,
        }
    }

    fn to_c64(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
}

// ============================================================
// CONVERSION HELPERS
// ============================================================

/// Convert a DenseMatrix (Complex64) to a flat Vec<GpuComplex> (f32).
fn matrix_to_gpu(m: &DenseMatrix) -> Vec<GpuComplex> {
    m.data.iter().map(|&c| GpuComplex::from_c64(c)).collect()
}

/// Convert a flat slice of GpuComplex back to a DenseMatrix.
fn gpu_to_matrix(data: &[GpuComplex], dim: usize) -> DenseMatrix {
    DenseMatrix {
        dim,
        data: data.iter().map(|c| c.to_c64()).collect(),
    }
}

/// Pack multiple DenseMatrix into a single contiguous GPU buffer.
fn matrices_to_gpu_batch(matrices: &[DenseMatrix]) -> Vec<GpuComplex> {
    let mut buf = Vec::with_capacity(matrices.iter().map(|m| m.data.len()).sum());
    for m in matrices {
        buf.extend(m.data.iter().map(|&c| GpuComplex::from_c64(c)));
    }
    buf
}

// ============================================================
// METAL PULSE BACKEND (macOS only)
// ============================================================

/// Cached compute pipeline states for all pulse simulation kernels.
#[cfg(target_os = "macos")]
struct PulsePipelines {
    complex_matmul: ComputePipelineState,
    complex_matmul_tiled: ComputePipelineState,
    lindblad_rhs: ComputePipelineState,
    batch_rk4_lindblad: ComputePipelineState,
    hermitian_expm: ComputePipelineState,
    batch_gate_fidelity: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl PulsePipelines {
    fn new(device: &DeviceRef, library: &LibraryRef) -> Result<Self, PulseError> {
        let make = |name: &str| -> Result<ComputePipelineState, PulseError> {
            let func = library
                .get_function(name, None)
                .map_err(|e| {
                    PulseError::InvalidSystem(format!("Shader function '{}': {}", name, e))
                })?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| {
                    PulseError::InvalidSystem(format!("Pipeline '{}': {}", name, e))
                })
        };
        Ok(PulsePipelines {
            complex_matmul: make("complex_matmul")?,
            complex_matmul_tiled: make("complex_matmul_tiled")?,
            lindblad_rhs: make("lindblad_rhs")?,
            batch_rk4_lindblad: make("batch_rk4_lindblad")?,
            hermitian_expm: make("hermitian_expm")?,
            batch_gate_fidelity: make("batch_gate_fidelity")?,
        })
    }
}

/// Metal GPU backend for pulse-level quantum simulation.
///
/// Holds the Metal device, command queue, and pre-compiled pipeline states
/// for all pulse simulation kernels. Created once and reused for all
/// GPU-accelerated operations in a pulse optimization workflow.
#[cfg(target_os = "macos")]
pub struct MetalPulseBackend {
    device: Device,
    queue: CommandQueue,
    pipelines: PulsePipelines,
}

#[cfg(target_os = "macos")]
impl MetalPulseBackend {
    /// Create a new Metal pulse backend.
    ///
    /// Finds the default Metal GPU device, compiles the embedded shader library,
    /// and builds all compute pipeline states. Returns an error if no Metal
    /// device is available or shader compilation fails.
    pub fn new() -> Result<Self, PulseError> {
        let device = Device::system_default().ok_or_else(|| {
            PulseError::InvalidSystem("No Metal GPU device found".into())
        })?;
        let queue = device.new_command_queue();

        let library = device
            .new_library_with_source(PULSE_SHADER_SOURCE, &CompileOptions::new())
            .map_err(|e| {
                PulseError::InvalidSystem(format!("Pulse shader compilation failed: {}", e))
            })?;

        let pipelines = PulsePipelines::new(&device, &library)?;

        Ok(MetalPulseBackend {
            device,
            queue,
            pipelines,
        })
    }

    // --------------------------------------------------------
    // BUFFER ALLOCATION
    // --------------------------------------------------------

    /// Allocate a shared-memory GPU buffer from a slice of GpuComplex.
    fn buffer_from_complex(&self, data: &[GpuComplex]) -> Buffer {
        let bytes = (data.len() * std::mem::size_of::<GpuComplex>()) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            bytes,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate an output buffer of the given byte size.
    fn output_buffer(&self, bytes: u64) -> Buffer {
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate a buffer from a u32 slice (for scalar constants passed as buffers).
    fn buffer_from_u32(&self, val: u32) -> Buffer {
        self.device.new_buffer_with_data(
            &val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate a buffer from a f32 value.
    fn buffer_from_f32(&self, val: f32) -> Buffer {
        self.device.new_buffer_with_data(
            &val as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate a buffer from a f32 slice (eigenvalues).
    fn buffer_from_f32_slice(&self, data: &[f32]) -> Buffer {
        let bytes = (data.len() * std::mem::size_of::<f32>()) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            bytes,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Read back a GpuComplex buffer into a Vec.
    fn read_complex_buffer(&self, buffer: &Buffer, count: usize) -> Vec<GpuComplex> {
        let ptr = buffer.contents() as *const GpuComplex;
        let mut out = vec![GpuComplex::default(); count];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
        }
        out
    }

    /// Read back a f32 buffer into a Vec.
    fn read_f32_buffer(&self, buffer: &Buffer, count: usize) -> Vec<f32> {
        let ptr = buffer.contents() as *const f32;
        let mut out = vec![0.0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
        }
        out
    }

    // --------------------------------------------------------
    // DISPATCH HELPERS
    // --------------------------------------------------------

    /// Dispatch a 2D grid of (width, height) threads with THREADGROUP_2D x THREADGROUP_2D threadgroups.
    fn dispatch_2d(
        &self,
        encoder: &ComputeCommandEncoderRef,
        width: u64,
        height: u64,
    ) {
        let tg = THREADGROUP_2D;
        let groups_x = (width + tg - 1) / tg;
        let groups_y = (height + tg - 1) / tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(groups_x, groups_y, 1),
            MTLSize::new(tg, tg, 1),
        );
    }

    /// Dispatch a 1D grid of `count` threads.
    fn dispatch_1d(&self, encoder: &ComputeCommandEncoderRef, count: u64) {
        let tg = THREADGROUP_1D;
        let groups = (count + tg - 1) / tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
    }

    // --------------------------------------------------------
    // PUBLIC API: MATRIX MULTIPLY
    // --------------------------------------------------------

    /// GPU-accelerated complex matrix multiplication: C = A * B.
    ///
    /// Both matrices must be square with the same dimension.
    /// Uses f32 precision on GPU; results are converted back to f64.
    pub fn matmul_gpu(&self, a: &DenseMatrix, b: &DenseMatrix) -> Result<DenseMatrix, PulseError> {
        if a.dim != b.dim {
            return Err(PulseError::DimensionMismatch(format!(
                "matmul_gpu: A is {}x{}, B is {}x{}",
                a.dim, a.dim, b.dim, b.dim
            )));
        }
        let n = a.dim;
        let nn = n * n;

        let buf_a = self.buffer_from_complex(&matrix_to_gpu(a));
        let buf_b = self.buffer_from_complex(&matrix_to_gpu(b));
        let buf_c = self.output_buffer((nn * std::mem::size_of::<GpuComplex>()) as u64);
        let buf_n = self.buffer_from_u32(n as u32);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // Use tiled kernel for larger matrices, simple kernel for small ones.
        if n >= 32 {
            enc.set_compute_pipeline_state(&self.pipelines.complex_matmul_tiled);
        } else {
            enc.set_compute_pipeline_state(&self.pipelines.complex_matmul);
        }

        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        enc.set_buffer(3, Some(&buf_n), 0);
        self.dispatch_2d(enc, n as u64, n as u64);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let result_data = self.read_complex_buffer(&buf_c, nn);
        Ok(gpu_to_matrix(&result_data, n))
    }

    // --------------------------------------------------------
    // PUBLIC API: LINDBLAD STEP
    // --------------------------------------------------------

    /// Single Lindblad master equation step on GPU.
    ///
    /// Computes `drho/dt` for the given Hamiltonian, density matrix, and collapse
    /// operators, then returns `rho + dt * drho/dt` (forward Euler).
    /// For production use, prefer `batch_evolve` which uses RK4.
    ///
    /// # Arguments
    /// - `h`: Hamiltonian (NxN)
    /// - `rho`: density matrix (NxN)
    /// - `l_ops`: collapse operators, each NxN
    /// - `dt`: time step
    pub fn lindblad_step_gpu(
        &self,
        h: &DenseMatrix,
        rho: &DenseMatrix,
        l_ops: &[DenseMatrix],
        dt: f64,
    ) -> Result<DenseMatrix, PulseError> {
        let n = h.dim;
        if rho.dim != n {
            return Err(PulseError::DimensionMismatch(format!(
                "lindblad_step: H is {}x{}, rho is {}x{}",
                n, n, rho.dim, rho.dim
            )));
        }
        for (i, l) in l_ops.iter().enumerate() {
            if l.dim != n {
                return Err(PulseError::DimensionMismatch(format!(
                    "lindblad_step: L_ops[{}] is {}x{}, expected {}x{}",
                    i, l.dim, l.dim, n, n
                )));
            }
        }

        let nn = n * n;
        let k = l_ops.len();

        let buf_h = self.buffer_from_complex(&matrix_to_gpu(h));
        let buf_rho = self.buffer_from_complex(&matrix_to_gpu(rho));
        let buf_l = self.buffer_from_complex(&matrices_to_gpu_batch(l_ops));
        let buf_drho = self.output_buffer((nn * std::mem::size_of::<GpuComplex>()) as u64);
        let buf_n = self.buffer_from_u32(n as u32);
        let buf_k = self.buffer_from_u32(k as u32);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.lindblad_rhs);
        enc.set_buffer(0, Some(&buf_h), 0);
        enc.set_buffer(1, Some(&buf_rho), 0);
        enc.set_buffer(2, Some(&buf_l), 0);
        enc.set_buffer(3, Some(&buf_drho), 0);
        enc.set_buffer(4, Some(&buf_n), 0);
        enc.set_buffer(5, Some(&buf_k), 0);
        self.dispatch_2d(enc, n as u64, n as u64);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let drho_data = self.read_complex_buffer(&buf_drho, nn);
        let drho = gpu_to_matrix(&drho_data, n);

        // Forward Euler: rho_new = rho + dt * drho
        Ok(rho.add(&drho.scale(Complex64::new(dt, 0.0))))
    }

    // --------------------------------------------------------
    // PUBLIC API: BATCH EVOLVE
    // --------------------------------------------------------

    /// Evolve M density matrices in parallel, each with its own Hamiltonian.
    ///
    /// This is the core GRAPE acceleration kernel: each time slice of the
    /// piecewise-constant control pulse gets its own Hamiltonian, and all
    /// M slices evolve simultaneously on the GPU via RK4.
    ///
    /// # Arguments
    /// - `hamiltonians`: M Hamiltonians, one per time slice (each NxN)
    /// - `rho_init`: initial density matrix (NxN), replicated across all M slices
    /// - `l_ops`: collapse operators (shared across all slices)
    /// - `dt`: time step per slice
    ///
    /// # Returns
    /// M evolved density matrices (one per time slice).
    pub fn batch_evolve(
        &self,
        hamiltonians: &[DenseMatrix],
        rho_init: &DenseMatrix,
        l_ops: &[DenseMatrix],
        dt: f64,
    ) -> Result<Vec<DenseMatrix>, PulseError> {
        let m = hamiltonians.len();
        if m == 0 {
            return Ok(vec![]);
        }
        let n = rho_init.dim;
        if n > MAX_BATCH_RK4_DIM {
            return Err(PulseError::InvalidSystem(format!(
                "batch_evolve: dimension {} exceeds max {} for batch RK4 kernel",
                n, MAX_BATCH_RK4_DIM
            )));
        }
        for (i, h) in hamiltonians.iter().enumerate() {
            if h.dim != n {
                return Err(PulseError::DimensionMismatch(format!(
                    "batch_evolve: hamiltonians[{}] is {}x{}, expected {}x{}",
                    i, h.dim, h.dim, n, n
                )));
            }
        }
        let k = l_ops.len();
        let nn = n * n;

        // Pack M copies of rho_init into the batch buffer
        let rho_gpu = matrix_to_gpu(rho_init);
        let mut rho_batch_data: Vec<GpuComplex> = Vec::with_capacity(m * nn);
        for _ in 0..m {
            rho_batch_data.extend_from_slice(&rho_gpu);
        }

        let buf_h = self.buffer_from_complex(&matrices_to_gpu_batch(hamiltonians));
        let buf_rho = self.buffer_from_complex(&rho_batch_data);
        let buf_l = if k > 0 {
            self.buffer_from_complex(&matrices_to_gpu_batch(l_ops))
        } else {
            // Even with no collapse ops, we need a valid buffer pointer
            self.output_buffer(std::mem::size_of::<GpuComplex>() as u64)
        };
        let buf_dt = self.buffer_from_f32(dt as f32);
        let buf_n = self.buffer_from_u32(n as u32);
        let buf_k = self.buffer_from_u32(k as u32);
        let buf_m = self.buffer_from_u32(m as u32);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.batch_rk4_lindblad);
        enc.set_buffer(0, Some(&buf_h), 0);
        enc.set_buffer(1, Some(&buf_rho), 0);
        enc.set_buffer(2, Some(&buf_l), 0);
        enc.set_buffer(3, Some(&buf_dt), 0);
        enc.set_buffer(4, Some(&buf_n), 0);
        enc.set_buffer(5, Some(&buf_k), 0);
        enc.set_buffer(6, Some(&buf_m), 0);
        self.dispatch_1d(enc, m as u64);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        // Read back all M density matrices
        let result_data = self.read_complex_buffer(&buf_rho, m * nn);
        let mut results = Vec::with_capacity(m);
        for i in 0..m {
            let start = i * nn;
            results.push(gpu_to_matrix(&result_data[start..start + nn], n));
        }
        Ok(results)
    }

    // --------------------------------------------------------
    // PUBLIC API: MATRIX EXPONENTIAL
    // --------------------------------------------------------

    /// GPU-accelerated Hermitian matrix exponential via eigendecomposition.
    ///
    /// Computes `exp(-i * H * dt)` given pre-computed eigenvalues and eigenvectors.
    /// The caller is responsible for diagonalizing H on the CPU (e.g., via Jacobi
    /// or LAPACK); this kernel just does the fast reconstruction.
    ///
    /// # Arguments
    /// - `eigenvalues`: N real eigenvalues
    /// - `eigenvectors`: NxN matrix whose columns are the eigenvectors (stored row-major)
    /// - `dt`: time parameter
    pub fn hermitian_expm(
        &self,
        eigenvalues: &[f64],
        eigenvectors: &DenseMatrix,
        dt: f64,
    ) -> Result<DenseMatrix, PulseError> {
        let n = eigenvalues.len();
        if eigenvectors.dim != n {
            return Err(PulseError::DimensionMismatch(format!(
                "hermitian_expm: {} eigenvalues but eigenvector matrix is {}x{}",
                n, eigenvectors.dim, eigenvectors.dim
            )));
        }
        let nn = n * n;

        let eig_f32: Vec<f32> = eigenvalues.iter().map(|&v| v as f32).collect();
        let buf_eig = self.buffer_from_f32_slice(&eig_f32);
        let buf_v = self.buffer_from_complex(&matrix_to_gpu(eigenvectors));
        let buf_result = self.output_buffer((nn * std::mem::size_of::<GpuComplex>()) as u64);
        let buf_dt = self.buffer_from_f32(dt as f32);
        let buf_n = self.buffer_from_u32(n as u32);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.hermitian_expm);
        enc.set_buffer(0, Some(&buf_eig), 0);
        enc.set_buffer(1, Some(&buf_v), 0);
        enc.set_buffer(2, Some(&buf_result), 0);
        enc.set_buffer(3, Some(&buf_dt), 0);
        enc.set_buffer(4, Some(&buf_n), 0);
        self.dispatch_2d(enc, n as u64, n as u64);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let result_data = self.read_complex_buffer(&buf_result, nn);
        Ok(gpu_to_matrix(&result_data, n))
    }

    // --------------------------------------------------------
    // PUBLIC API: BATCH FIDELITY
    // --------------------------------------------------------

    /// Compute process fidelity for M unitaries against a single target, in parallel.
    ///
    /// `F = |Tr(U_target† @ U)|^2 / d^2` for each U in the batch.
    /// This is the inner loop of GRAPE: after propagating each time slice,
    /// we need the fidelity gradient to update pulse amplitudes.
    ///
    /// # Arguments
    /// - `target`: d x d target unitary
    /// - `unitaries`: M unitaries, each d x d
    ///
    /// # Returns
    /// M fidelity values in `[0, 1]`.
    pub fn batch_fidelity(
        &self,
        target: &DenseMatrix,
        unitaries: &[DenseMatrix],
    ) -> Result<Vec<f64>, PulseError> {
        let m = unitaries.len();
        if m == 0 {
            return Ok(vec![]);
        }
        let d = target.dim;
        for (i, u) in unitaries.iter().enumerate() {
            if u.dim != d {
                return Err(PulseError::DimensionMismatch(format!(
                    "batch_fidelity: unitaries[{}] is {}x{}, target is {}x{}",
                    i, u.dim, u.dim, d, d
                )));
            }
        }

        let buf_target = self.buffer_from_complex(&matrix_to_gpu(target));
        let buf_u_batch = self.buffer_from_complex(&matrices_to_gpu_batch(unitaries));
        let buf_fid = self.output_buffer((m * std::mem::size_of::<f32>()) as u64);
        let buf_d = self.buffer_from_u32(d as u32);
        let buf_n = self.buffer_from_u32(d as u32); // N = d for now
        let buf_m = self.buffer_from_u32(m as u32);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.batch_gate_fidelity);
        enc.set_buffer(0, Some(&buf_target), 0);
        enc.set_buffer(1, Some(&buf_u_batch), 0);
        enc.set_buffer(2, Some(&buf_fid), 0);
        enc.set_buffer(3, Some(&buf_d), 0);
        enc.set_buffer(4, Some(&buf_n), 0);
        enc.set_buffer(5, Some(&buf_m), 0);
        self.dispatch_1d(enc, m as u64);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let fid_f32 = self.read_f32_buffer(&buf_fid, m);
        Ok(fid_f32.iter().map(|&f| f as f64).collect())
    }
}

// ============================================================
// CUDA STUB (feature-gated placeholder)
// ============================================================

/// CUDA pulse-level backend placeholder.
///
/// Provides the same API as `MetalPulseBackend` but falls back to CPU
/// implementations. Replace with actual CUDA kernel dispatch when the
/// CUDA backend is implemented.
#[cfg(feature = "cuda")]
pub struct CudaPulseBackend;

#[cfg(feature = "cuda")]
impl CudaPulseBackend {
    /// Create a new CUDA pulse backend (currently CPU fallback).
    pub fn new() -> Result<Self, PulseError> {
        // TODO: Initialize CUDA device, compile PTX kernels, create streams
        Ok(CudaPulseBackend)
    }

    /// GPU matrix multiply (CPU fallback).
    pub fn matmul_gpu(&self, a: &DenseMatrix, b: &DenseMatrix) -> Result<DenseMatrix, PulseError> {
        // TODO: Dispatch to CUDA complex_matmul kernel
        if a.dim != b.dim {
            return Err(PulseError::DimensionMismatch(format!(
                "matmul_gpu: A is {}x{}, B is {}x{}",
                a.dim, a.dim, b.dim, b.dim
            )));
        }
        Ok(a.matmul(b))
    }

    /// Single Lindblad step (CPU fallback).
    pub fn lindblad_step_gpu(
        &self,
        h: &DenseMatrix,
        rho: &DenseMatrix,
        l_ops: &[DenseMatrix],
        dt: f64,
    ) -> Result<DenseMatrix, PulseError> {
        // TODO: Dispatch to CUDA lindblad_rhs kernel
        let _ = (h, rho, l_ops, dt);
        Err(PulseError::InvalidSystem(
            "CUDA Lindblad step not yet implemented; use CPU fallback".into(),
        ))
    }

    /// Batch evolve (CPU fallback).
    pub fn batch_evolve(
        &self,
        hamiltonians: &[DenseMatrix],
        rho_init: &DenseMatrix,
        l_ops: &[DenseMatrix],
        dt: f64,
    ) -> Result<Vec<DenseMatrix>, PulseError> {
        // TODO: Dispatch to CUDA batch_rk4_lindblad kernel
        let _ = (hamiltonians, rho_init, l_ops, dt);
        Err(PulseError::InvalidSystem(
            "CUDA batch evolve not yet implemented; use CPU fallback".into(),
        ))
    }

    /// Batch fidelity (CPU fallback).
    pub fn batch_fidelity(
        &self,
        target: &DenseMatrix,
        unitaries: &[DenseMatrix],
    ) -> Result<Vec<f64>, PulseError> {
        // TODO: Dispatch to CUDA batch_gate_fidelity kernel
        let d = target.dim;
        let mut fidelities = Vec::with_capacity(unitaries.len());
        for u in unitaries {
            // F = |Tr(U_target† @ U)|^2 / d^2
            let prod = target.dagger().matmul(u);
            let tr = prod.trace();
            let f = tr.norm_sqr() / (d as f64 * d as f64);
            fidelities.push(f);
        }
        Ok(fidelities)
    }
}

// ============================================================
// CPU-SIDE REFERENCE IMPLEMENTATIONS (for testing)
// ============================================================

/// CPU reference implementation of Lindblad RHS for test validation.
///
/// Computes `drho/dt = -i[H, rho] + sum_k (L_k rho L_k† - 0.5 {L_k†L_k, rho})`
fn cpu_lindblad_rhs(
    h: &DenseMatrix,
    rho: &DenseMatrix,
    l_ops: &[DenseMatrix],
) -> DenseMatrix {
    let i_neg = Complex64::new(0.0, -1.0);

    // Unitary part: -i[H, rho]
    let comm = h.commutator(rho);
    let mut drho = comm.scale(i_neg);

    // Dissipative part
    for l in l_ops {
        let ld = l.dagger();
        let ldl = ld.matmul(l);

        // L rho L†
        let l_rho_ld = l.matmul(rho).matmul(&ld);

        // {L†L, rho} = L†L*rho + rho*L†L
        let anti = ldl.matmul(rho).add(&rho.matmul(&ldl));

        drho = drho.add(&l_rho_ld.sub(&anti.scale(Complex64::new(0.5, 0.0))));
    }
    drho
}

/// Check whether a matrix is approximately Hermitian (rho = rho†).
fn is_hermitian(m: &DenseMatrix, tol: f64) -> bool {
    let n = m.dim;
    for i in 0..n {
        for j in 0..n {
            let diff = m.get(i, j) - m.get(j, i).conj();
            if diff.norm() > tol {
                return false;
            }
        }
    }
    true
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    // ---- CPU-only tests (always run) ----

    #[test]
    fn test_gpu_complex_roundtrip() {
        let c = Complex64::new(1.5, -2.7);
        let gc = GpuComplex::from_c64(c);
        let back = gc.to_c64();
        // f32 roundtrip loses precision beyond ~7 decimal digits
        assert!((back.re - c.re).abs() < 1e-6);
        assert!((back.im - c.im).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_to_gpu_roundtrip() {
        let m = DenseMatrix::identity(3);
        let gpu = matrix_to_gpu(&m);
        let back = gpu_to_matrix(&gpu, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((back.get(i, j).re - expected).abs() < 1e-6);
                assert!(back.get(i, j).im.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_cpu_matmul_3x3() {
        // Known 3x3 matrix multiply for validation
        let mut a = DenseMatrix::zeros(3);
        let mut b = DenseMatrix::zeros(3);

        // A = [[1, 2, 0], [0, 1, 1], [1, 0, 1]]
        a.set(0, 0, Complex64::new(1.0, 0.0));
        a.set(0, 1, Complex64::new(2.0, 0.0));
        a.set(1, 1, Complex64::new(1.0, 0.0));
        a.set(1, 2, Complex64::new(1.0, 0.0));
        a.set(2, 0, Complex64::new(1.0, 0.0));
        a.set(2, 2, Complex64::new(1.0, 0.0));

        // B = identity
        b = DenseMatrix::identity(3);

        let c = a.matmul(&b);
        // A * I = A
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (c.get(i, j) - a.get(i, j)).norm() < 1e-12,
                    "A*I != A at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_cpu_lindblad_rhs_hermitian_output() {
        // For a Hermitian H, Hermitian rho, and physical L_ops,
        // the Lindblad RHS should preserve Hermiticity of drho.
        let n = 3;
        let mut h = DenseMatrix::zeros(n);
        h.set(0, 0, Complex64::new(1.0, 0.0));
        h.set(1, 1, Complex64::new(2.0, 0.0));
        h.set(2, 2, Complex64::new(3.0, 0.0));
        // Off-diagonal coupling (Hermitian)
        h.set(0, 1, Complex64::new(0.1, 0.0));
        h.set(1, 0, Complex64::new(0.1, 0.0));

        // rho = |0><0| (pure state, Hermitian)
        let mut rho = DenseMatrix::zeros(n);
        rho.set(0, 0, Complex64::new(1.0, 0.0));

        // Collapse operator: lowering operator |0><1|
        let mut l = DenseMatrix::zeros(n);
        l.set(0, 1, Complex64::new(0.05, 0.0));

        let drho = cpu_lindblad_rhs(&h, &rho, &[l]);
        assert!(
            is_hermitian(&drho, 1e-12),
            "Lindblad RHS should produce Hermitian output"
        );
    }

    #[test]
    fn test_cpu_fidelity_identity() {
        // Fidelity of identity with itself should be 1.0
        let d = 4;
        let id = DenseMatrix::identity(d);
        let prod = id.dagger().matmul(&id);
        let tr = prod.trace();
        let f = tr.norm_sqr() / (d as f64 * d as f64);
        assert!(
            (f - 1.0).abs() < 1e-12,
            "Fidelity of I with I should be 1.0, got {}",
            f
        );
    }

    #[test]
    fn test_is_hermitian() {
        let id = DenseMatrix::identity(3);
        assert!(is_hermitian(&id, 1e-15));

        let mut non_herm = DenseMatrix::zeros(2);
        non_herm.set(0, 1, Complex64::new(1.0, 0.0));
        // (0,1) = 1, (1,0) = 0 => not Hermitian
        assert!(!is_hermitian(&non_herm, 1e-15));
    }

    #[test]
    fn test_batch_packing() {
        let m1 = DenseMatrix::identity(2);
        let m2 = DenseMatrix::identity(2);
        let packed = matrices_to_gpu_batch(&[m1, m2]);
        // 2 matrices of 2x2 = 8 GpuComplex values
        assert_eq!(packed.len(), 8);
        // First matrix: diag = (1,0), off-diag = (0,0)
        assert!((packed[0].re - 1.0).abs() < 1e-6); // (0,0) = 1
        assert!((packed[1].re).abs() < 1e-6); // (0,1) = 0
        assert!((packed[2].re).abs() < 1e-6); // (1,0) = 0
        assert!((packed[3].re - 1.0).abs() < 1e-6); // (1,1) = 1
    }

    // ---- GPU tests (macOS only) ----

    #[cfg(target_os = "macos")]
    mod gpu_tests {
        use super::*;

        fn make_backend() -> MetalPulseBackend {
            MetalPulseBackend::new().expect("Metal device should be available on macOS")
        }

        #[test]
        fn test_gpu_matmul_identity() {
            let backend = make_backend();
            let id = DenseMatrix::identity(4);
            let result = backend.matmul_gpu(&id, &id).unwrap();
            for i in 0..4 {
                for j in 0..4 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (result.get(i, j).re - expected).abs() < 1e-5,
                        "I*I at ({},{}) = {:?}, expected {}",
                        i,
                        j,
                        result.get(i, j),
                        expected
                    );
                    assert!(
                        result.get(i, j).im.abs() < 1e-5,
                        "I*I imaginary at ({},{}) should be 0",
                        i,
                        j
                    );
                }
            }
        }

        #[test]
        fn test_gpu_matmul_known_3x3() {
            let backend = make_backend();

            // A = [[1+i, 2, 0], [0, 1, i], [1, 0, 1]]
            let mut a = DenseMatrix::zeros(3);
            a.set(0, 0, Complex64::new(1.0, 1.0));
            a.set(0, 1, Complex64::new(2.0, 0.0));
            a.set(1, 1, Complex64::new(1.0, 0.0));
            a.set(1, 2, Complex64::new(0.0, 1.0));
            a.set(2, 0, Complex64::new(1.0, 0.0));
            a.set(2, 2, Complex64::new(1.0, 0.0));

            let cpu_result = a.matmul(&a);
            let gpu_result = backend.matmul_gpu(&a, &a).unwrap();

            for i in 0..3 {
                for j in 0..3 {
                    let diff = (cpu_result.get(i, j) - gpu_result.get(i, j)).norm();
                    assert!(
                        diff < 1e-4,
                        "A*A mismatch at ({},{}): CPU={:?}, GPU={:?}",
                        i,
                        j,
                        cpu_result.get(i, j),
                        gpu_result.get(i, j)
                    );
                }
            }
        }

        #[test]
        fn test_gpu_lindblad_step_hermitian() {
            let backend = make_backend();
            let n = 3;

            // Diagonal Hamiltonian
            let mut h = DenseMatrix::zeros(n);
            h.set(0, 0, Complex64::new(1.0, 0.0));
            h.set(1, 1, Complex64::new(2.0, 0.0));
            h.set(2, 2, Complex64::new(3.0, 0.0));

            // rho = |0><0|
            let mut rho = DenseMatrix::zeros(n);
            rho.set(0, 0, Complex64::new(1.0, 0.0));

            // Small collapse operator
            let mut l = DenseMatrix::zeros(n);
            l.set(0, 1, Complex64::new(0.05, 0.0));

            let result = backend
                .lindblad_step_gpu(&h, &rho, &[l], 0.01)
                .unwrap();

            // The output should be approximately Hermitian
            assert!(
                is_hermitian(&result, 1e-4),
                "Lindblad step output should be approximately Hermitian"
            );

            // Trace should be approximately 1 (trace preservation)
            let tr = result.trace();
            assert!(
                (tr.re - 1.0).abs() < 1e-4,
                "Trace should be ~1.0, got {}",
                tr.re
            );
        }

        #[test]
        fn test_gpu_batch_fidelity_identity() {
            let backend = make_backend();
            let d = 2;
            let id = DenseMatrix::identity(d);

            // Batch of 5 identity matrices
            let batch: Vec<DenseMatrix> = (0..5).map(|_| DenseMatrix::identity(d)).collect();
            let fidelities = backend.batch_fidelity(&id, &batch).unwrap();

            assert_eq!(fidelities.len(), 5);
            for (i, &f) in fidelities.iter().enumerate() {
                assert!(
                    (f - 1.0).abs() < 1e-4,
                    "Fidelity[{}] should be 1.0, got {}",
                    i,
                    f
                );
            }
        }

        #[test]
        fn test_gpu_batch_evolve_no_collapse() {
            let backend = make_backend();
            let n = 2;

            // Simple diagonal Hamiltonian
            let mut h = DenseMatrix::zeros(n);
            h.set(0, 0, Complex64::new(1.0, 0.0));
            h.set(1, 1, Complex64::new(2.0, 0.0));

            // 3 identical Hamiltonians
            let hamiltonians = vec![h.clone(), h.clone(), h.clone()];

            // rho = |0><0|
            let mut rho = DenseMatrix::zeros(n);
            rho.set(0, 0, Complex64::new(1.0, 0.0));

            let results = backend
                .batch_evolve(&hamiltonians, &rho, &[], 0.001)
                .unwrap();

            assert_eq!(results.len(), 3);
            // With no collapse operators and small dt, rho should stay close to initial
            for (i, r) in results.iter().enumerate() {
                assert!(
                    is_hermitian(r, 1e-4),
                    "batch_evolve[{}] should be Hermitian",
                    i
                );
                let tr = r.trace();
                assert!(
                    (tr.re - 1.0).abs() < 1e-4,
                    "batch_evolve[{}] trace should be ~1.0, got {}",
                    i,
                    tr.re
                );
            }
        }

        #[test]
        fn test_gpu_hermitian_expm_identity() {
            let backend = make_backend();
            let n = 3;

            // Zero eigenvalues -> exp(-i * 0 * dt) = I
            let eigenvalues = vec![0.0f64; n];
            let eigvecs = DenseMatrix::identity(n);
            let result = backend.hermitian_expm(&eigenvalues, &eigvecs, 1.0).unwrap();

            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (result.get(i, j).re - expected).abs() < 1e-5,
                        "expm(0) at ({},{}) should be {}, got {:?}",
                        i,
                        j,
                        expected,
                        result.get(i, j)
                    );
                }
            }
        }

        #[test]
        fn test_gpu_matmul_dimension_mismatch() {
            let backend = make_backend();
            let a = DenseMatrix::identity(3);
            let b = DenseMatrix::identity(4);
            assert!(backend.matmul_gpu(&a, &b).is_err());
        }
    }
}
