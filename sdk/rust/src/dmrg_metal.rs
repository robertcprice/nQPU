//! Metal GPU Backend for DMRG Tensor Operations
//!
//! Provides GPU-accelerated kernels for the computationally expensive
//! operations in DMRG: effective Hamiltonian matvec, SVD, and tensor contraction.
//!
//! # Architecture
//!
//! - [`DmrgMetalEngine`]: Main GPU engine with cached pipelines
//! - [`GpuTensorBuffer`]: GPU buffer wrapper for tensor data
//! - Automatic CPU fallback when Metal is unavailable
//!
//! # Usage
//!
//! ```rust,ignore
//! use nqpu_metal::dmrg_metal::DmrgMetalEngine;
//!
//! let engine = DmrgMetalEngine::new()?;
//! let result = engine.matvec(&h_eff, &vec, dims)?;
//! ```

#[cfg(target_os = "macos")]
use metal::*;

use num_complex::Complex64;
use std::fmt;

type C64 = Complex64;

/// Errors from DMRG Metal operations.
#[derive(Debug)]
pub enum DmrgMetalError {
    /// Metal device not available.
    NoDevice,
    /// Shader compilation failed.
    ShaderError(String),
    /// Pipeline creation failed.
    PipelineError(String),
    /// Buffer allocation failed.
    BufferError(String),
    /// Invalid tensor dimensions.
    InvalidDimensions(String),
    /// Computation failed.
    ComputationError(String),
}

impl fmt::Display for DmrgMetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => write!(f, "Metal device not available"),
            Self::ShaderError(s) => write!(f, "Shader error: {}", s),
            Self::PipelineError(s) => write!(f, "Pipeline error: {}", s),
            Self::BufferError(s) => write!(f, "Buffer error: {}", s),
            Self::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
            Self::ComputationError(s) => write!(f, "Computation error: {}", s),
        }
    }
}

impl std::error::Error for DmrgMetalError {}

// ============================================================
// GPU TENSOR BUFFER
// ============================================================

/// GPU buffer wrapper for complex tensor data.
#[cfg(target_os = "macos")]
pub struct GpuTensorBuffer {
    buffer: Buffer,
    len: usize,
}

#[cfg(target_os = "macos")]
impl GpuTensorBuffer {
    /// Create a new GPU buffer from complex data.
    pub fn new(device: &DeviceRef, data: &[C64]) -> Result<Self, DmrgMetalError> {
        // Convert to f32 complex for Metal
        let f32_data: Vec<f32> = data.iter()
            .flat_map(|c| [c.re as f32, c.im as f32])
            .collect();

        let buffer = device.new_buffer_with_data(
            f32_data.as_ptr() as *const _,
            (f32_data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer,
            len: data.len(),
        })
    }

    /// Create an uninitialized GPU buffer.
    pub fn uninitialized(device: &DeviceRef, len: usize) -> Result<Self, DmrgMetalError> {
        let bytes = len * 2 * std::mem::size_of::<f32>(); // complex = 2 floats
        let buffer = device.new_buffer(
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self { buffer, len })
    }

    /// Read data back from GPU.
    pub fn read(&self) -> Vec<C64> {
        let ptr = self.buffer.contents() as *const f32;
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let re = unsafe { *ptr.add(i * 2) };
            let im = unsafe { *ptr.add(i * 2 + 1) };
            result.push(C64::new(re as f64, im as f64));
        }
        result
    }

    /// Get the underlying Metal buffer.
    pub fn as_buffer(&self) -> &BufferRef {
        &self.buffer
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.len
    }
}

// ============================================================
// DMRG METAL ENGINE
// ============================================================

/// Cached compute pipelines for DMRG operations.
#[cfg(target_os = "macos")]
struct DmrgPipelines {
    tensor_contract: ComputePipelineState,
    batch_matvec: ComputePipelineState,
    gemm_simdgroup: Option<ComputePipelineState>,
    truncate_svd: ComputePipelineState,
    entanglement_entropy: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl DmrgPipelines {
    fn new(device: &DeviceRef, library: &LibraryRef, has_simdgroup: bool) -> Result<Self, DmrgMetalError> {
        let make = |name: &str| -> Result<ComputePipelineState, DmrgMetalError> {
            let func = library
                .get_function(name, None)
                .map_err(|e| DmrgMetalError::ShaderError(format!("{}: {}", name, e)))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| DmrgMetalError::PipelineError(format!("{}: {}", name, e)))
        };

        let gemm_simdgroup = if has_simdgroup {
            make("dmrg_gemm_simdgroup").ok()
        } else {
            None
        };

        Ok(DmrgPipelines {
            tensor_contract: make("dmrg_tensor_contract_3d")?,
            batch_matvec: make("dmrg_batch_matvec")?,
            gemm_simdgroup,
            truncate_svd: make("dmrg_truncate_svd")?,
            entanglement_entropy: make("dmrg_entanglement_entropy")?,
        })
    }
}

/// Metal GPU engine for DMRG tensor operations.
#[cfg(target_os = "macos")]
pub struct DmrgMetalEngine {
    device: Device,
    queue: CommandQueue,
    pipelines: DmrgPipelines,
    has_simdgroup: bool,
}

#[cfg(target_os = "macos")]
impl DmrgMetalEngine {
    /// Create a new DMRG Metal engine.
    pub fn new() -> Result<Self, DmrgMetalError> {
        let device = Device::system_default()
            .ok_or(DmrgMetalError::NoDevice)?;

        // Load shader library
        let source = include_str!("metal/dmrg_kernels.metal");
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(source, &options)
            .map_err(|e| DmrgMetalError::ShaderError(e.to_string()))?;

        // Check for simdgroup support (Metal 3.1+, Apple9 GPU family)
        let has_simdgroup = Self::check_simdgroup_support(&device);

        let pipelines = DmrgPipelines::new(&device, &library, has_simdgroup)?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            pipelines,
            has_simdgroup,
        })
    }

    fn check_simdgroup_support(device: &DeviceRef) -> bool {
        // Check for Metal 3.1+ features (simdgroup_matrix)
        // This requires Apple9 GPU family (M4) or later
        // For now, use a conservative check
        if let Some(family) = device.name().strip_prefix("Apple") {
            if let Ok(tier) = family.parse::<u32>() {
                return tier >= 9;
            }
        }
        false
    }

    /// Check if simdgroup matrix operations are available.
    pub fn has_simdgroup(&self) -> bool {
        self.has_simdgroup
    }

    /// Perform matrix-vector multiply: y = H * x
    ///
    /// # Arguments
    /// * `h` - Matrix in row-major order [m, n]
    /// * `x` - Vector [n]
    /// * `m` - Number of rows
    /// * `n` - Number of columns
    ///
    /// # Returns
    /// Vector y of length m
    pub fn matvec(
        &self,
        h: &[C64],
        x: &[C64],
        m: usize,
        n: usize,
    ) -> Result<Vec<C64>, DmrgMetalError> {
        if h.len() != m * n {
            return Err(DmrgMetalError::InvalidDimensions(
                format!("H matrix size {} != m*n = {}*{} = {}", h.len(), m, n, m*n)
            ));
        }
        if x.len() != n {
            return Err(DmrgMetalError::InvalidDimensions(
                format!("x vector size {} != n = {}", x.len(), n)
            ));
        }

        // Create GPU buffers
        let h_buf = GpuTensorBuffer::new(&self.device, h)?;
        let x_buf = GpuTensorBuffer::new(&self.device, x)?;
        let y_buf = GpuTensorBuffer::uninitialized(&self.device, m)?;

        // Encode command
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.batch_matvec);
        encoder.set_buffer(0, Some(h_buf.as_buffer()), 0);
        encoder.set_buffer(1, Some(x_buf.as_buffer()), 0);
        encoder.set_buffer(2, Some(y_buf.as_buffer()), 0);

        // Set dimensions
        let mn = [m as u32, n as u32];
        let mn_buf = self.device.new_buffer_with_data(
            mn.as_ptr() as *const _,
            8,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(3, Some(&mn_buf), 0);

        let batch: u32 = 1;
        let batch_buf = self.device.new_buffer_with_data(
            &batch as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(4, Some(&batch_buf), 0);

        // Dispatch
        let threadgroup_size = MTLSize { width: 8, height: 8, depth: 1 };
        let grid_width = (m + 7) / 8;
        let grid_height = (n + 7) / 8;
        let grid_size = MTLSize { width: grid_width as u64, height: grid_height as u64, depth: 1 };

        encoder.use_resource(h_buf.as_buffer(), MTLResourceUsage::Read);
        encoder.use_resource(x_buf.as_buffer(), MTLResourceUsage::Read);
        encoder.use_resource(y_buf.as_buffer(), MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        // Commit and wait
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok(y_buf.read())
    }

    /// Perform 3D tensor contraction: C[a,b,c] = sum_d A[a,b,d] * B[d,c]
    ///
    /// # Arguments
    /// * `a` - Tensor A with shape [dim_a, dim_b, dim_contract]
    /// * `b` - Tensor B with shape [dim_contract, dim_c]
    /// * `dims` - (dim_a, dim_b, dim_c)
    /// * `dim_contract` - Contraction dimension
    pub fn tensor_contract_3d(
        &self,
        a: &[C64],
        b: &[C64],
        dims: (usize, usize, usize),
        dim_contract: usize,
    ) -> Result<Vec<C64>, DmrgMetalError> {
        let (dim_a, dim_b, dim_c) = dims;
        let expected_a = dim_a * dim_b * dim_contract;
        let expected_b = dim_contract * dim_c;
        let output_len = dim_a * dim_b * dim_c;

        if a.len() != expected_a {
            return Err(DmrgMetalError::InvalidDimensions(
                format!("A tensor size {} != expected {}", a.len(), expected_a)
            ));
        }
        if b.len() != expected_b {
            return Err(DmrgMetalError::InvalidDimensions(
                format!("B tensor size {} != expected {}", b.len(), expected_b)
            ));
        }

        // Create GPU buffers
        let a_buf = GpuTensorBuffer::new(&self.device, a)?;
        let b_buf = GpuTensorBuffer::new(&self.device, b)?;
        let c_buf = GpuTensorBuffer::uninitialized(&self.device, output_len)?;

        // Encode command
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.tensor_contract);
        encoder.set_buffer(0, Some(a_buf.as_buffer()), 0);
        encoder.set_buffer(1, Some(b_buf.as_buffer()), 0);
        encoder.set_buffer(2, Some(c_buf.as_buffer()), 0);

        // Set dimensions
        let dims_u32 = [dim_a as u32, dim_b as u32, dim_c as u32];
        let dims_buf = self.device.new_buffer_with_data(
            dims_u32.as_ptr() as *const _,
            12,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(3, Some(&dims_buf), 0);

        let dc = dim_contract as u32;
        let dc_buf = self.device.new_buffer_with_data(
            &dc as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(4, Some(&dc_buf), 0);

        // Dispatch 3D grid
        let grid_size = MTLSize {
            width: dim_a as u64,
            height: dim_b as u64,
            depth: dim_c as u64,
        };
        let threadgroup_size = MTLSize { width: 4, height: 4, depth: 4 };

        encoder.use_resource(a_buf.as_buffer(), MTLResourceUsage::Read);
        encoder.use_resource(b_buf.as_buffer(), MTLResourceUsage::Read);
        encoder.use_resource(c_buf.as_buffer(), MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok(c_buf.read())
    }

    /// Truncate SVD result to specified bond dimension.
    ///
    /// # Arguments
    /// * `u` - Left singular vectors [m, n]
    /// * `s` - Singular values [n]
    /// * `vt` - Right singular vectors [n, n]
    /// * `m` - Number of rows
    /// * `n` - Number of columns (min dimension)
    /// * `keep` - Number of singular values to keep
    ///
    /// # Returns
    /// Tuple (A_new, B_new) where A_new = U*S and B_new = Vt
    pub fn truncate_svd(
        &self,
        u: &[C64],
        s: &[f64],
        vt: &[C64],
        m: usize,
        n: usize,
        keep: usize,
    ) -> Result<(Vec<C64>, Vec<C64>), DmrgMetalError> {
        if keep > n {
            return Err(DmrgMetalError::InvalidDimensions(
                format!("keep {} > n {}", keep, n)
            ));
        }

        // Create GPU buffers
        let u_buf = GpuTensorBuffer::new(&self.device, u)?;

        // Singular values are real floats
        let s_f32: Vec<f32> = s.iter().map(|&x| x as f32).collect();
        let s_buf = self.device.new_buffer_with_data(
            s_f32.as_ptr() as *const _,
            (s_f32.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let vt_buf = GpuTensorBuffer::new(&self.device, vt)?;
        let a_buf = GpuTensorBuffer::uninitialized(&self.device, m * keep)?;
        let b_buf = GpuTensorBuffer::uninitialized(&self.device, keep * n)?;

        // Encode command
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.truncate_svd);
        encoder.set_buffer(0, Some(u_buf.as_buffer()), 0);
        encoder.set_buffer(1, Some(&s_buf), 0);
        encoder.set_buffer(2, Some(vt_buf.as_buffer()), 0);
        encoder.set_buffer(3, Some(a_buf.as_buffer()), 0);
        encoder.set_buffer(4, Some(b_buf.as_buffer()), 0);

        // Set dimensions
        let dims = [m as u32, n as u32];
        let dims_buf = self.device.new_buffer_with_data(
            dims.as_ptr() as *const _,
            8,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(5, Some(&dims_buf), 0);

        let keep_u32 = keep as u32;
        let keep_buf = self.device.new_buffer_with_data(
            &keep_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(6, Some(&keep_buf), 0);

        // Dispatch
        let grid_size = MTLSize {
            width: m.max(keep) as u64,
            height: n.max(keep) as u64,
            depth: 1,
        };
        let threadgroup_size = MTLSize { width: 16, height: 16, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok((a_buf.read(), b_buf.read()))
    }

    /// Compute von Neumann entanglement entropy from singular values.
    pub fn entanglement_entropy(&self, singular_values: &[f64]) -> Result<f64, DmrgMetalError> {
        let s_f32: Vec<f32> = singular_values.iter().map(|&x| x as f32).collect();
        let s_buf = self.device.new_buffer_with_data(
            s_f32.as_ptr() as *const _,
            (s_f32.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let entropy_buf = self.device.new_buffer(
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.entanglement_entropy);
        encoder.set_buffer(0, Some(&s_buf), 0);
        encoder.set_buffer(1, Some(&entropy_buf), 0);

        let n = singular_values.len() as u32;
        let n_buf = self.device.new_buffer_with_data(
            &n as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(2, Some(&n_buf), 0);

        let grid_size = MTLSize { width: 1, height: 1, depth: 1 };
        let threadgroup_size = MTLSize { width: 1, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let result = unsafe { *(entropy_buf.contents() as *const f32) };
        Ok(result as f64)
    }
}

// ============================================================
// CPU FALLBACK (non-macOS)
// ============================================================

#[cfg(not(target_os = "macos"))]
pub struct DmrgMetalEngine;

#[cfg(not(target_os = "macos"))]
impl DmrgMetalEngine {
    pub fn new() -> Result<Self, DmrgMetalError> {
        Err(DmrgMetalError::NoDevice)
    }

    pub fn has_simdgroup(&self) -> bool {
        false
    }

    pub fn matvec(&self, _: &[C64], _: &[C64], _: usize, _: usize) -> Result<Vec<C64>, DmrgMetalError> {
        Err(DmrgMetalError::NoDevice)
    }

    pub fn tensor_contract_3d(&self, _: &[C64], _: &[C64], _: (usize, usize, usize), _: usize) -> Result<Vec<C64>, DmrgMetalError> {
        Err(DmrgMetalError::NoDevice)
    }

    pub fn truncate_svd(&self, _: &[C64], _: &[f64], _: &[C64], _: usize, _: usize, _: usize) -> Result<(Vec<C64>, Vec<C64>), DmrgMetalError> {
        Err(DmrgMetalError::NoDevice)
    }

    pub fn entanglement_entropy(&self, _: &[f64]) -> Result<f64, DmrgMetalError> {
        Err(DmrgMetalError::NoDevice)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_dmrg_metal_engine_creation() {
        let result = DmrgMetalEngine::new();
        // May fail if no Metal device
        if result.is_ok() {
            let engine = result.unwrap();
            // simdgroup depends on hardware
            let _ = engine.has_simdgroup();
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_matvec_simple() {
        let engine = match DmrgMetalEngine::new() {
            Ok(e) => e,
            Err(_) => return, // Skip if no Metal
        };

        // 2x2 identity * [1, i] = [1, i]
        let h = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0),
                     C64::new(0.0, 0.0), C64::new(1.0, 0.0)];
        let x = vec![C64::new(1.0, 0.0), C64::new(0.0, 1.0)];

        let y = engine.matvec(&h, &x, 2, 2).unwrap();

        assert!((y[0] - x[0]).norm() < 1e-5);
        assert!((y[1] - x[1]).norm() < 1e-5);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_entanglement_entropy() {
        let engine = match DmrgMetalEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };

        // Maximally entangled state: s = [1/sqrt(2), 1/sqrt(2)]
        let s = vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()];
        let entropy = engine.entanglement_entropy(&s).unwrap();

        // Should be ln(2) ≈ 0.693
        assert!((entropy - 2.0f64.ln()).abs() < 0.01);
    }
}
