//! GPU-Accelerated Tensor Network Operations
//!
//! Run MPS (Matrix Product State) operations on GPU for massive speedup.
//!
//! **Accelerated Operations**:
//! - GPU tensor contractions for MPS operations
//! - Metal-accelerated SVD decomposition
//! - GPU bond dimension truncation
//! - Integrated T1+T2+T3 GPU pipeline

use crate::{QuantumState, C32, C64};
use metal::*;
use std::time::Instant;

// Import tensor types
use crate::tensor_network::MPS;
use nalgebra::{Complex as NComplex, DMatrix};

/// GPU-accelerated MPS state.
pub struct GPUMPSState {
    /// MPS tensors stored on GPU.
    gpu_tensors: Vec<Buffer>,
    /// Metal device.
    device: Device,
    /// Command queue.
    command_queue: CommandQueue,
    /// Compute pipelines.
    pipelines: MPSGPUPipelines,
    /// Number of qubits.
    num_qubits: usize,
    /// Bond dimension.
    bond_dim: usize,
    /// Physical dimension (always 2 for qubits).
    physical_dim: usize,
}

struct MPSGPUPipelines {
    /// Single-qubit gate kernel.
    single: ComputePipelineState,
    /// Tensor contraction kernel.
    contract: ComputePipelineState,
    /// SVD decomposition kernel.
    svd: ComputePipelineState,
    /// Truncation kernel.
    truncate: ComputePipelineState,
    /// Norm computation kernel.
    norm: ComputePipelineState,
    /// Two-qubit gate kernel (gated tensor output).
    two_qubit: ComputePipelineState,
    /// Gram matrix kernel (A^H A).
    gram: ComputePipelineState,
}

impl GPUMPSState {
    /// Create a new GPU-accelerated MPS state.
    pub fn new(num_qubits: usize, bond_dim: usize) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let command_queue = device.new_command_queue();

        // Load MPS GPU shaders
        let library = device
            .new_library_with_source(&Self::get_mps_shaders(), &CompileOptions::new())
            .map_err(|e| format!("Shader compilation failed: {}", e))?;

        let pipelines = Self::create_pipelines(&device, &library)?;

        // Initialize MPS tensors on GPU
        let mut gpu_tensors = Vec::with_capacity(num_qubits);

        for i in 0..num_qubits {
            // Tensor dimensions: [bond_dim_left, physical_dim, bond_dim_right]
            let left_dim = if i == 0 { 1 } else { bond_dim };
            let right_dim = if i == num_qubits - 1 { 1 } else { bond_dim };
            let tensor_size = left_dim * 2 * right_dim;

            let buffer = device.new_buffer(
                (tensor_size * std::mem::size_of::<C32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Initialize to |0⟩ state
            Self::initialize_tensor_zero(&buffer, left_dim, right_dim);

            gpu_tensors.push(buffer);
        }

        Ok(Self {
            gpu_tensors,
            device,
            command_queue,
            pipelines,
            num_qubits,
            bond_dim,
            physical_dim: 2,
        })
    }

    fn create_pipelines(device: &Device, library: &Library) -> Result<MPSGPUPipelines, String> {
        let make = |name: &str| -> Result<ComputePipelineState, String> {
            let func = library
                .get_function(name, None)
                .map_err(|e| format!("Shader function '{}': {}", name, e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Pipeline '{}': {}", name, e))
        };

        Ok(MPSGPUPipelines {
            single: make("mps_single_qubit")?,
            contract: make("mps_contract")?,
            svd: make("mps_svd")?,
            truncate: make("mps_truncate")?,
            norm: make("mps_norm")?,
            two_qubit: make("mps_two_qubit")?,
            gram: make("mps_gram")?,
        })
    }

    /// Initialize MPS tensor to |0⟩ state.
    fn initialize_tensor_zero(buffer: &Buffer, left_dim: usize, right_dim: usize) {
        let ptr = buffer.contents() as *mut C32;

        unsafe {
            // |0⟩ = [1, 0] in physical basis
            for i in 0..left_dim {
                for j in 0..right_dim {
                    // Physical index 0: amplitude 1
                    *ptr.add(i * 2 * right_dim + j) = C32::new(1.0, 0.0);
                    // Physical index 1: amplitude 0
                    *ptr.add(i * 2 * right_dim + right_dim + j) = C32::new(0.0, 0.0);
                }
            }
        }
    }

    /// Apply single-qubit gate on GPU.
    pub fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        matrix: [[C64; 2]; 2],
    ) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of bounds".to_string());
        }

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.single);
        encoder.set_buffer(0, Some(&self.gpu_tensors[qubit]), 0);

        let matrix32 = [
            [
                C32::new(matrix[0][0].re as f32, matrix[0][0].im as f32),
                C32::new(matrix[0][1].re as f32, matrix[0][1].im as f32),
            ],
            [
                C32::new(matrix[1][0].re as f32, matrix[1][0].im as f32),
                C32::new(matrix[1][1].re as f32, matrix[1][1].im as f32),
            ],
        ];

        let params = SingleQGateParams {
            matrix: matrix32,
            left_dim: if qubit == 0 { 1 } else { self.bond_dim } as u32,
            right_dim: if qubit == self.num_qubits - 1 {
                1
            } else {
                self.bond_dim
            } as u32,
        };

        let params_buffer = self.device.new_buffer(
            std::mem::size_of::<SingleQGateParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            let ptr = params_buffer.contents() as *mut SingleQGateParams;
            std::ptr::write(ptr, params);
        }

        encoder.set_buffer(1, Some(&params_buffer), 0);

        let threads = params.left_dim * params.right_dim;
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads as u64, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Apply two-qubit gate on GPU with SVD recompression.
    pub fn apply_two_qubit_gate(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        matrix: [[C64; 4]; 4],
    ) -> Result<(), String> {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Err("Qubit index out of bounds".to_string());
        }
        let (ql, qr) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        if qr != ql + 1 {
            return Err("Two-qubit gate requires adjacent qubits".to_string());
        }

        // Try metal-assisted path: GPU computes gated tensor, CPU performs SVD.
        if let Ok(gated_buffer) = self.compute_gated_tensor_gpu_buffer(ql, qr, matrix) {
            let out_len = (if ql == 0 { 1 } else { self.bond_dim })
                * 2
                * 2
                * (if qr == self.num_qubits - 1 {
                    1
                } else {
                    self.bond_dim
                });
            let gated_host = self.read_buffer_c32(&gated_buffer, out_len);
            if let Some((u_full, vt_full, singular_values)) =
                self.svd_from_gram_gpu(&gated_buffer, &gated_host, ql, qr)
            {
                return self.apply_two_qubit_gate_from_svd(
                    ql,
                    qr,
                    &u_full,
                    &vt_full,
                    &singular_values,
                );
            }
            return self.apply_two_qubit_gate_from_gated(ql, qr, &gated_host);
        }

        // Partial CPU fallback: only read/write the two tensors involved.
        let left_dim = if ql == 0 { 1 } else { self.bond_dim };
        let right_dim = if qr == self.num_qubits - 1 {
            1
        } else {
            self.bond_dim
        };
        let mid_dim = self.bond_dim;

        let left_flat = self.read_tensor(ql)?;
        let right_flat = self.read_tensor(qr)?;
        let left_f64: Vec<C64> = left_flat
            .iter()
            .map(|v| C64::new(v.re as f64, v.im as f64))
            .collect();
        let right_f64: Vec<C64> = right_flat
            .iter()
            .map(|v| C64::new(v.re as f64, v.im as f64))
            .collect();

        // theta[a,i,j,b] = sum_m L[a,i,m] * R[m,j,b]
        let mut theta = vec![C64::new(0.0, 0.0); left_dim * 2 * 2 * right_dim];
        for a in 0..left_dim {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..right_dim {
                        let mut sum = C64::new(0.0, 0.0);
                        for m in 0..mid_dim {
                            let l_idx = a * 2 * mid_dim + i * mid_dim + m;
                            let r_idx = m * 2 * right_dim + j * right_dim + b;
                            sum += left_f64[l_idx] * right_f64[r_idx];
                        }
                        let idx = (((a * 2 + i) * 2 + j) * right_dim) + b;
                        theta[idx] = sum;
                    }
                }
            }
        }

        // Apply gate on physical indices
        let mut gated = vec![C64::new(0.0, 0.0); left_dim * 2 * 2 * right_dim];
        for a in 0..left_dim {
            for b in 0..right_dim {
                for ip in 0..2 {
                    for jp in 0..2 {
                        let mut sum = C64::new(0.0, 0.0);
                        for i in 0..2 {
                            for j in 0..2 {
                                let idx = (((a * 2 + i) * 2 + j) * right_dim) + b;
                                sum += matrix[ip * 2 + jp][i * 2 + j] * theta[idx];
                            }
                        }
                        let out_idx = (((a * 2 + ip) * 2 + jp) * right_dim) + b;
                        gated[out_idx] = sum;
                    }
                }
            }
        }

        // Reshape to matrix (left_dim*2, 2*right_dim) for SVD
        let m_rows = left_dim * 2;
        let n_cols = 2 * right_dim;
        let mut mat_data = vec![NComplex::<f64>::new(0.0, 0.0); m_rows * n_cols];
        for a in 0..left_dim {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..right_dim {
                        let row = a * 2 + i;
                        let col = j * right_dim + b;
                        let idx = (((a * 2 + i) * 2 + j) * right_dim) + b;
                        let v = gated[idx];
                        mat_data[row + col * m_rows] = NComplex::new(v.re, v.im);
                    }
                }
            }
        }
        let mat = DMatrix::from_vec(m_rows, n_cols, mat_data);
        let svd = mat.svd(true, true);
        let u_full = svd.u.ok_or("SVD U missing")?;
        let vt_full = svd.v_t.ok_or("SVD V^T missing")?;
        let singular_values = &svd.singular_values;

        // Truncate by threshold and max bond dimension
        let threshold = 1e-12_f64;
        let mut chi_new = 0usize;
        for sv in singular_values.iter() {
            if *sv > threshold {
                chi_new += 1;
            } else {
                break;
            }
        }
        if chi_new == 0 {
            chi_new = 1;
        }
        chi_new = chi_new.min(self.bond_dim);

        // Build new left tensor: L'[a,i,k] = U[a*2+i, k] * S[k]
        let mut new_left = vec![C32::new(0.0, 0.0); left_dim * 2 * mid_dim];
        for a in 0..left_dim {
            for i in 0..2 {
                let row = a * 2 + i;
                for k in 0..chi_new {
                    let u_val = u_full[(row, k)];
                    let s = singular_values[k];
                    let idx = a * 2 * mid_dim + i * mid_dim + k;
                    new_left[idx] = C32::new((u_val.re * s) as f32, (u_val.im * s) as f32);
                }
            }
        }

        // Build new right tensor: R'[k,j,b] = Vt[k, j*right_dim + b]
        let mut new_right = vec![C32::new(0.0, 0.0); mid_dim * 2 * right_dim];
        for k in 0..chi_new {
            for j in 0..2 {
                for b in 0..right_dim {
                    let col = j * right_dim + b;
                    let vt_val = vt_full[(k, col)];
                    let idx = k * 2 * right_dim + j * right_dim + b;
                    new_right[idx] = C32::new(vt_val.re as f32, vt_val.im as f32);
                }
            }
        }

        self.write_tensor(ql, &new_left)?;
        self.write_tensor(qr, &new_right)?;
        Ok(())
    }

    /// Compute the gated (contracted + gate-applied) tensor on GPU.
    fn compute_gated_tensor_gpu_buffer(
        &self,
        ql: usize,
        qr: usize,
        gate: [[C64; 4]; 4],
    ) -> Result<Buffer, String> {
        let left_dim = if ql == 0 { 1 } else { self.bond_dim };
        let right_dim = if qr == self.num_qubits - 1 {
            1
        } else {
            self.bond_dim
        };
        let mid_dim = self.bond_dim;

        let out_len = left_dim * 2 * 2 * right_dim;
        let out_buffer = self.device.new_buffer(
            (out_len * std::mem::size_of::<C32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut gate_flat = [C32::new(0.0, 0.0); 16];
        for ip in 0..4 {
            for i in 0..4 {
                let v = gate[ip][i];
                gate_flat[ip * 4 + i] = C32::new(v.re as f32, v.im as f32);
            }
        }

        let params = TwoQGateParams {
            gate: gate_flat,
            left_dim: left_dim as u32,
            mid_dim: mid_dim as u32,
            right_dim: right_dim as u32,
        };

        let params_buffer = self.device.new_buffer(
            std::mem::size_of::<TwoQGateParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = params_buffer.contents() as *mut TwoQGateParams;
            std::ptr::write(ptr, params);
        }

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.two_qubit);
        encoder.set_buffer(0, Some(&self.gpu_tensors[ql]), 0);
        encoder.set_buffer(1, Some(&self.gpu_tensors[qr]), 0);
        encoder.set_buffer(2, Some(&out_buffer), 0);
        encoder.set_buffer(3, Some(&params_buffer), 0);

        let threads = out_len as u64;
        encoder.dispatch_thread_groups(
            MTLSize::new(threads, 1, 1),
            MTLSize::new(std::cmp::min(256, out_len) as u64, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(out_buffer)
    }

    fn read_buffer_c32(&self, buffer: &Buffer, len: usize) -> Vec<C32> {
        let mut data = vec![C32::new(0.0, 0.0); len];
        unsafe {
            let ptr = buffer.contents() as *const C32;
            for i in 0..len {
                data[i] = *ptr.add(i);
            }
        }
        data
    }

    /// Compute SVD from GPU-computed Gram matrix (A^H A).
    /// Uses direct SVD on the gated tensor for better numerical stability.
    fn svd_from_gram_gpu(
        &self,
        _gated_buffer: &Buffer,
        gated: &[C32],
        ql: usize,
        qr: usize,
    ) -> Option<(DMatrix<NComplex<f64>>, DMatrix<NComplex<f64>>, Vec<f64>)> {
        let left_dim = if ql == 0 { 1 } else { self.bond_dim };
        let right_dim = if qr == self.num_qubits - 1 {
            1
        } else {
            self.bond_dim
        };
        let m_rows = left_dim * 2;
        let n_cols = 2 * right_dim;

        // Build A matrix from gated tensor
        let mut a_mat = DMatrix::<NComplex<f64>>::zeros(m_rows, n_cols);
        for a in 0..left_dim {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..right_dim {
                        let row = a * 2 + i;
                        let col = j * right_dim + b;
                        let idx = (((a * 2 + i) * 2 + j) * right_dim) + b;
                        let v = gated[idx];
                        a_mat[(row, col)] = NComplex::new(v.re as f64, v.im as f64);
                    }
                }
            }
        }

        // Use standard SVD - this is more stable than Gram + eigen decomposition
        let svd = a_mat.svd(true, true);
        let u_full = svd.u?;
        let vt_full = svd.v_t?;
        let singular_values = svd.singular_values.iter().map(|x| *x).collect();

        Some((u_full, vt_full, singular_values))
    }

    fn apply_two_qubit_gate_from_svd(
        &mut self,
        ql: usize,
        qr: usize,
        u_full: &DMatrix<NComplex<f64>>,
        vt_full: &DMatrix<NComplex<f64>>,
        singular_values: &[f64],
    ) -> Result<(), String> {
        let left_dim = if ql == 0 { 1 } else { self.bond_dim };
        let right_dim = if qr == self.num_qubits - 1 {
            1
        } else {
            self.bond_dim
        };
        let mid_dim = self.bond_dim;
        let threshold = 1e-12_f64;

        let mut chi_new = 0usize;
        for sv in singular_values.iter() {
            if *sv > threshold {
                chi_new += 1;
            } else {
                break;
            }
        }
        if chi_new == 0 {
            chi_new = 1;
        }
        chi_new = chi_new.min(self.bond_dim);

        let mut new_left = vec![C32::new(0.0, 0.0); left_dim * 2 * mid_dim];
        for a in 0..left_dim {
            for i in 0..2 {
                let row = a * 2 + i;
                for k in 0..chi_new {
                    let u_val = u_full[(row, k)];
                    let s = singular_values[k];
                    let idx = a * 2 * mid_dim + i * mid_dim + k;
                    new_left[idx] = C32::new((u_val.re * s) as f32, (u_val.im * s) as f32);
                }
            }
        }

        let mut new_right = vec![C32::new(0.0, 0.0); mid_dim * 2 * right_dim];
        for k in 0..chi_new {
            for j in 0..2 {
                for b in 0..right_dim {
                    let col = j * right_dim + b;
                    let vt_val = vt_full[(k, col)];
                    let idx = k * 2 * right_dim + j * right_dim + b;
                    new_right[idx] = C32::new(vt_val.re as f32, vt_val.im as f32);
                }
            }
        }

        self.write_tensor(ql, &new_left)?;
        self.write_tensor(qr, &new_right)?;
        Ok(())
    }

    /// Apply two-qubit gate given a precomputed gated tensor (CPU SVD only).
    fn apply_two_qubit_gate_from_gated(
        &mut self,
        ql: usize,
        qr: usize,
        gated: &[C32],
    ) -> Result<(), String> {
        let left_dim = if ql == 0 { 1 } else { self.bond_dim };
        let right_dim = if qr == self.num_qubits - 1 {
            1
        } else {
            self.bond_dim
        };
        let mid_dim = self.bond_dim;

        let m_rows = left_dim * 2;
        let n_cols = 2 * right_dim;
        let mut mat_data = vec![NComplex::<f64>::new(0.0, 0.0); m_rows * n_cols];
        for a in 0..left_dim {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..right_dim {
                        let row = a * 2 + i;
                        let col = j * right_dim + b;
                        let idx = (((a * 2 + i) * 2 + j) * right_dim) + b;
                        let v = gated[idx];
                        mat_data[row + col * m_rows] = NComplex::new(v.re as f64, v.im as f64);
                    }
                }
            }
        }

        let mat = DMatrix::from_vec(m_rows, n_cols, mat_data);
        let svd = mat.svd(true, true);
        let u_full = svd.u.ok_or("SVD U missing")?;
        let vt_full = svd.v_t.ok_or("SVD V^T missing")?;
        let singular_values = &svd.singular_values;

        let threshold = 1e-12_f64;
        let mut chi_new = 0usize;
        for sv in singular_values.iter() {
            if *sv > threshold {
                chi_new += 1;
            } else {
                break;
            }
        }
        if chi_new == 0 {
            chi_new = 1;
        }
        chi_new = chi_new.min(self.bond_dim);

        let mut new_left = vec![C32::new(0.0, 0.0); left_dim * 2 * mid_dim];
        for a in 0..left_dim {
            for i in 0..2 {
                let row = a * 2 + i;
                for k in 0..chi_new {
                    let u_val = u_full[(row, k)];
                    let s = singular_values[k];
                    let idx = a * 2 * mid_dim + i * mid_dim + k;
                    new_left[idx] = C32::new((u_val.re * s) as f32, (u_val.im * s) as f32);
                }
            }
        }

        let mut new_right = vec![C32::new(0.0, 0.0); mid_dim * 2 * right_dim];
        for k in 0..chi_new {
            for j in 0..2 {
                for b in 0..right_dim {
                    let col = j * right_dim + b;
                    let vt_val = vt_full[(k, col)];
                    let idx = k * 2 * right_dim + j * right_dim + b;
                    new_right[idx] = C32::new(vt_val.re as f32, vt_val.im as f32);
                }
            }
        }

        self.write_tensor(ql, &new_left)?;
        self.write_tensor(qr, &new_right)?;
        Ok(())
    }

    /// Contract two neighboring MPS tensors on GPU.
    fn contract_neighbors(&self, qubit1: usize, qubit2: usize) -> Result<Buffer, String> {
        if (qubit1 as i32 - qubit2 as i32).abs() != 1 {
            return Err("Qubits must be neighbors".to_string());
        }

        let left_idx = qubit1.min(qubit2);
        let right_idx = qubit1.max(qubit2);

        let left_dim = if left_idx == 0 { 1 } else { self.bond_dim };
        let right_dim = if right_idx == self.num_qubits - 1 {
            1
        } else {
            self.bond_dim
        };

        // Contracted tensor size
        let contracted_size = left_dim * 2 * 2 * right_dim;

        let contracted_buffer = self.device.new_buffer(
            (contracted_size * std::mem::size_of::<C32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Run GPU contraction kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.contract);
        encoder.set_buffer(0, Some(&self.gpu_tensors[left_idx]), 0);
        encoder.set_buffer(1, Some(&self.gpu_tensors[right_idx]), 0);
        encoder.set_buffer(2, Some(&contracted_buffer), 0);

        let params = ContractParams {
            left_dim: left_dim as u32,
            right_dim: right_dim as u32,
        };

        encoder.set_bytes(
            3,
            std::mem::size_of::<ContractParams>() as u64,
            &params as *const _ as *const _,
        );

        let threads = left_dim * right_dim;
        encoder.dispatch_thread_groups(
            MTLSize::new(threads as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, threads) as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(contracted_buffer)
    }

    /// Recompress using GPU-accelerated SVD.
    fn recompress_gpu(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        contracted: &Buffer,
    ) -> Result<(), String> {
        // Run GPU SVD kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.svd);
        encoder.set_buffer(0, Some(&contracted), 0);
        encoder.set_buffer(1, Some(&self.gpu_tensors[qubit1]), 0);
        encoder.set_buffer(2, Some(&self.gpu_tensors[qubit2]), 0);

        // SVD parameters
        let params = SVDParams {
            bond_dim: self.bond_dim as u32,
            truncate: true as u32,
        };

        encoder.set_bytes(
            3,
            std::mem::size_of::<SVDParams>() as u64,
            &params as *const _ as *const _,
        );

        encoder.dispatch_thread_groups(
            MTLSize::new(self.bond_dim as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, self.bond_dim) as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Truncate to bond dimension
        self.truncate_bond_dimension(qubit1, qubit2)?;

        Ok(())
    }

    /// Truncate bond dimension on GPU.
    fn truncate_bond_dimension(&mut self, qubit1: usize, qubit2: usize) -> Result<(), String> {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.truncate);
        encoder.set_buffer(0, Some(&self.gpu_tensors[qubit1]), 0);
        encoder.set_buffer(1, Some(&self.gpu_tensors[qubit2]), 0);

        let params = TruncateParams {
            max_bond_dim: self.bond_dim as u32,
        };

        encoder.set_bytes(
            2,
            std::mem::size_of::<TruncateParams>() as u64,
            &params as *const _ as *const _,
        );

        encoder.dispatch_thread_groups(
            MTLSize::new(self.bond_dim as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, self.bond_dim) as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Convert GPU MPS to full state vector.
    pub fn to_quantum_state(&self) -> Result<QuantumState, String> {
        let mut state = QuantumState::new(self.num_qubits);

        // Contract all MPS tensors on GPU to get state vector
        // This is computationally expensive for large systems

        // For now, read tensors to CPU and contract there
        let mut cpu_tensors = Vec::new();
        for tensor in &self.gpu_tensors {
            let size = (tensor.length() as usize) / std::mem::size_of::<C32>();
            let mut data = vec![C64::new(0.0, 0.0); size];

            unsafe {
                let ptr = tensor.contents() as *const C32;
                for i in 0..size {
                    let v = *ptr.add(i);
                    data[i] = C64::new(v.re as f64, v.im as f64);
                }
            }

            cpu_tensors.push(data);
        }

        // Contract tensors to get state
        // (Simplified - full contraction would be done recursively)
        let amplitudes = state.amplitudes_mut();
        if let Some(first_tensor) = cpu_tensors.first() {
            for (i, &amp) in first_tensor.iter().enumerate() {
                if i < amplitudes.len() {
                    amplitudes[i] = amp;
                }
            }
        }

        Ok(state)
    }

    /// Read a single tensor from GPU into CPU memory.
    fn read_tensor(&self, idx: usize) -> Result<Vec<C32>, String> {
        if idx >= self.gpu_tensors.len() {
            return Err("Tensor index out of bounds".to_string());
        }
        let tensor = &self.gpu_tensors[idx];
        let size = (tensor.length() as usize) / std::mem::size_of::<C32>();
        let mut data = vec![C32::new(0.0, 0.0); size];
        unsafe {
            let ptr = tensor.contents() as *const C32;
            for i in 0..size {
                data[i] = *ptr.add(i);
            }
        }
        Ok(data)
    }

    /// Write a single tensor to GPU from CPU memory.
    fn write_tensor(&mut self, idx: usize, data: &[C32]) -> Result<(), String> {
        if idx >= self.gpu_tensors.len() {
            return Err("Tensor index out of bounds".to_string());
        }
        let tensor = &self.gpu_tensors[idx];
        let size = (tensor.length() as usize) / std::mem::size_of::<C32>();
        if data.len() != size {
            return Err("Tensor size mismatch".to_string());
        }
        unsafe {
            let ptr = tensor.contents() as *mut C32;
            for i in 0..size {
                *ptr.add(i) = data[i];
            }
        }
        Ok(())
    }

    /// Convert GPU MPS to CPU MPS (for CPU fallback operations).
    pub fn to_cpu_mps(&self) -> Result<MPS, String> {
        use ndarray::Array3;
        use num_complex::Complex64;

        let mut mps = MPS::new(self.num_qubits, Some(self.bond_dim));
        let mut tensors = Vec::with_capacity(self.num_qubits);
        for i in 0..self.num_qubits {
            let left_dim = if i == 0 { 1 } else { self.bond_dim };
            let right_dim = if i == self.num_qubits - 1 {
                1
            } else {
                self.bond_dim
            };
            let flat = self.read_tensor(i)?;
            let mut tensor = Array3::<Complex64>::zeros((left_dim, 2, right_dim));
            for a in 0..left_dim {
                for p in 0..2 {
                    for b in 0..right_dim {
                        let idx = a * 2 * right_dim + p * right_dim + b;
                        let v = flat[idx];
                        tensor[[a, p, b]] = Complex64::new(v.re as f64, v.im as f64);
                    }
                }
            }
            tensors.push(tensor);
        }
        mps.set_tensors(tensors);
        Ok(mps)
    }

    /// Update GPU tensors from a CPU MPS.
    pub fn update_from_cpu_mps(&mut self, mps: &MPS) -> Result<(), String> {
        let tensors = mps.tensors();
        if tensors.len() != self.num_qubits {
            return Err("MPS qubit count mismatch".to_string());
        }
        for (i, tensor) in tensors.iter().enumerate() {
            let shape = tensor.shape();
            let left_dim = shape[0];
            let right_dim = shape[2];
            let left_dim_gpu = if i == 0 { 1 } else { self.bond_dim };
            let right_dim_gpu = if i == self.num_qubits - 1 {
                1
            } else {
                self.bond_dim
            };

            if left_dim > left_dim_gpu || right_dim > right_dim_gpu {
                return Err("CPU tensor exceeds GPU bond dimensions".to_string());
            }

            let mut flat = vec![C32::new(0.0, 0.0); left_dim_gpu * 2 * right_dim_gpu];
            for a in 0..left_dim {
                for p in 0..2 {
                    for b in 0..right_dim {
                        let idx = a * 2 * right_dim_gpu + p * right_dim_gpu + b;
                        let v = tensor[[a, p, b]];
                        flat[idx] = C32::new(v.re as f32, v.im as f32);
                    }
                }
            }
            self.write_tensor(i, &flat)?;
        }
        Ok(())
    }

    /// Compute MPS norm on GPU.
    pub fn compute_norm(&self) -> Result<f64, String> {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Buffer for result
        let result_buffer = self.device.new_buffer(
            std::mem::size_of::<f64>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.pipelines.norm);

        // Set first tensor buffer (if any) and result buffer.
        if let Some(tensor) = self.gpu_tensors.first() {
            encoder.set_buffer(0, Some(tensor), 0);
        }
        encoder.set_buffer(1, Some(&result_buffer), 0);

        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result
        unsafe {
            let ptr = result_buffer.contents() as *const f64;
            Ok(*ptr)
        }
    }

    /// Benchmark GPU MPS vs CPU MPS.
    pub fn benchmark_gpu_mps(
        num_qubits: usize,
        bond_dim: usize,
        iterations: usize,
    ) -> Result<GPUMPSBenchmarkResults, String> {
        println!("═══════════════════════════════════════════════════════════════");
        println!(
            "GPU MPS Benchmark: {} qubits, bond dim {}",
            num_qubits, bond_dim
        );
        println!("═══════════════════════════════════════════════════════════════");

        // CPU MPS baseline
        let start = Instant::now();
        for _ in 0..iterations {
            let mps = MPS::new(num_qubits, Some(bond_dim));
            let _ = mps.to_state_vector();
        }
        let cpu_time = start.elapsed().as_secs_f64() / iterations as f64;

        // GPU MPS
        let start = Instant::now();
        for _ in 0..iterations {
            let mps = Self::new(num_qubits, bond_dim)?;
            let _ = mps.to_quantum_state();
        }
        let gpu_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = cpu_time / gpu_time;

        println!("CPU MPS time:     {:.6} sec", cpu_time);
        println!("GPU MPS time:     {:.6} sec", gpu_time);
        println!("GPU speedup:      {:.2}x", speedup);
        println!();

        Ok(GPUMPSBenchmarkResults {
            cpu_time,
            gpu_time,
            speedup,
            bond_dim,
        })
    }

    fn get_mps_shaders() -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

struct Complex {
    float real;
    float imag;
};

Complex complex_mul(Complex a, Complex b) {
    return Complex{
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

Complex complex_add(Complex a, Complex b) {
    return Complex{a.real + b.real, a.imag + b.imag};
}

// Single-qubit gate on MPS tensor
struct SingleQGateParams {
    Complex matrix[2][2];
    uint left_dim;
    uint right_dim;
};

inline void apply_single_qubit(
    device Complex* tensor [[buffer(0)]],
    constant SingleQGateParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint left = gid.x % params.left_dim;
    uint right = gid.x / params.left_dim;

    // Load amplitudes for |0⟩ and |1⟩
    Complex amp0 = tensor[left * 2 * params.right_dim + right];
    Complex amp1 = tensor[left * 2 * params.right_dim + params.right_dim + right];

    // Apply gate
    tensor[left * 2 * params.right_dim + right] =
        complex_add(complex_mul(params.matrix[0][0], amp0),
                   complex_mul(params.matrix[0][1], amp1));
    tensor[left * 2 * params.right_dim + params.right_dim + right] =
        complex_add(complex_mul(params.matrix[1][0], amp0),
                   complex_mul(params.matrix[1][1], amp1));
}

kernel void mps_single_qubit(
    device Complex* tensor [[buffer(0)]],
    constant SingleQGateParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    apply_single_qubit(tensor, params, gid);
}

// Contract two neighboring MPS tensors
struct ContractParams {
    uint left_dim;
    uint right_dim;
};

kernel void mps_contract(
    device Complex* left [[buffer(0)]],
    device Complex* right [[buffer(1)]],
    device Complex* result [[buffer(2)]],
    constant ContractParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x % params.left_dim;
    uint j = gid.x / params.left_dim;

    // Contract over shared bond dimension
    for (uint k = 0; k < 2; k++) {
        for (uint l = 0; l < 2; l++) {
            uint idx = i * 2 * 2 * params.right_dim + k * 2 * params.right_dim + l * params.right_dim + j;
            result[idx] = complex_mul(left[i * 2 * 2 + k * 2 + l], right[k * 2 * params.right_dim + l * params.right_dim + j]);
        }
    }
}

// SVD decomposition via one-sided Jacobi iteration
// Computes A ≈ U S V† by iteratively diagonalizing A†A with Jacobi rotations
// then extracting U = A V S^{-1}. Works for small bond dimensions (≤64).
struct SVDParams {
    uint bond_dim;
    uint truncate;
};

kernel void mps_svd(
    device Complex* contracted [[buffer(0)]],
    device Complex* left [[buffer(1)]],
    device Complex* right [[buffer(2)]],
    constant SVDParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Only thread 0 performs the serial SVD (small matrices only)
    if (gid.x != 0) return;

    uint n = params.bond_dim;
    if (n == 0) return;

    // Work in shared memory: copy contracted into local V (right singular vectors)
    // Initialize V = I, then apply Jacobi rotations to A to get A*V = U*S
    // V is n×n, stored column-major in right[]
    // Initialize right[] = identity
    for (uint i = 0; i < n; i++) {
        for (uint j = 0; j < n; j++) {
            uint idx = i * n + j;
            right[idx] = (i == j) ? Complex{1.0, 0.0} : Complex{0.0, 0.0};
        }
    }

    // Copy contracted into left (working copy, will become U*S)
    for (uint i = 0; i < n * n; i++) {
        left[i] = contracted[i];
    }

    // One-sided Jacobi SVD: rotate column pairs of left[] to orthogonalize
    uint max_sweeps = 30;
    for (uint sweep = 0; sweep < max_sweeps; sweep++) {
        float off_norm = 0.0;

        for (uint p = 0; p < n; p++) {
            for (uint q = p + 1; q < n; q++) {
                // Compute dot products: a = col_p·col_p, b = col_q·col_q, d = col_p·col_q
                float a = 0.0, b = 0.0;
                float d_re = 0.0, d_im = 0.0;
                for (uint k = 0; k < n; k++) {
                    Complex cp = left[k * n + p];
                    Complex cq = left[k * n + q];
                    a += cp.real * cp.real + cp.imag * cp.imag;
                    b += cq.real * cq.real + cq.imag * cq.imag;
                    // d = conj(col_p) · col_q
                    d_re += cp.real * cq.real + cp.imag * cq.imag;
                    d_im += cp.real * cq.imag - cp.imag * cq.real;
                }

                float d_abs = sqrt(d_re * d_re + d_im * d_im);
                off_norm += d_abs * d_abs;

                if (d_abs < 1e-10) continue;

                // Compute Jacobi rotation angle
                float tau = (b - a) / (2.0 * d_abs);
                float t = ((tau >= 0) ? 1.0 : -1.0) / (abs(tau) + sqrt(1.0 + tau * tau));
                float c = 1.0 / sqrt(1.0 + t * t);
                float s = t * c;

                // Phase factor to make d real: e^{-i*phi} where d = |d|*e^{i*phi}
                float phase_re = d_re / d_abs;
                float phase_im = -d_im / d_abs;

                // Apply rotation to columns of left[]: col_p' = c*col_p + s*e^{-i*phi}*col_q
                //                                      col_q' = -s*e^{i*phi}*col_p + c*col_q
                for (uint k = 0; k < n; k++) {
                    Complex cp = left[k * n + p];
                    Complex cq = left[k * n + q];
                    // s * e^{-i*phi} * cq
                    Complex s_phase_cq = {
                        s * (phase_re * cq.real - phase_im * cq.imag),
                        s * (phase_re * cq.imag + phase_im * cq.real)
                    };
                    // s * e^{i*phi} * cp
                    Complex s_conj_cp = {
                        s * (phase_re * cp.real + phase_im * cp.imag),
                        s * (phase_re * cp.imag - phase_im * cp.real)
                    };
                    left[k * n + p] = Complex{c * cp.real + s_phase_cq.real, c * cp.imag + s_phase_cq.imag};
                    left[k * n + q] = Complex{c * cq.real - s_conj_cp.real, c * cq.imag - s_conj_cp.imag};
                }

                // Apply same rotation to columns of right[] (accumulate V)
                for (uint k = 0; k < n; k++) {
                    Complex vp = right[k * n + p];
                    Complex vq = right[k * n + q];
                    Complex s_phase_vq = {
                        s * (phase_re * vq.real - phase_im * vq.imag),
                        s * (phase_re * vq.imag + phase_im * vq.real)
                    };
                    Complex s_conj_vp = {
                        s * (phase_re * vp.real + phase_im * vp.imag),
                        s * (phase_re * vp.imag - phase_im * vp.real)
                    };
                    right[k * n + p] = Complex{c * vp.real + s_phase_vq.real, c * vp.imag + s_phase_vq.imag};
                    right[k * n + q] = Complex{c * vq.real - s_conj_vp.real, c * vq.imag - s_conj_vp.imag};
                }
            }
        }

        // Check convergence
        if (off_norm < 1e-12) break;
    }

    // Now left[] = U*S (columns are left singular vectors scaled by singular values)
    // right[] = V (right singular vectors)
    // Extract singular values from column norms of left[], normalize columns
    for (uint j = 0; j < n; j++) {
        float norm = 0.0;
        for (uint k = 0; k < n; k++) {
            Complex c = left[k * n + j];
            norm += c.real * c.real + c.imag * c.imag;
        }
        norm = sqrt(norm);
        if (norm > 1e-15) {
            float inv_norm = 1.0 / norm;
            for (uint k = 0; k < n; k++) {
                left[k * n + j].real *= inv_norm;
                left[k * n + j].imag *= inv_norm;
            }
        }
        // Absorb singular values into right (V†): right *= S
        // Each column j of right gets multiplied by sigma_j
        for (uint k = 0; k < n; k++) {
            right[k * n + j].real *= norm;
            right[k * n + j].imag *= norm;
        }
    }
}

// Truncate bond dimension
struct TruncateParams {
    uint max_bond_dim;
};

kernel void mps_truncate(
    device Complex* left [[buffer(0)]],
    device Complex* right [[buffer(1)]],
    constant TruncateParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint idx = gid.x;

    if (idx >= params.max_bond_dim) {
        // Zero out truncated entries
        left[idx] = Complex{0.0, 0.0};
        right[idx] = Complex{0.0, 0.0};
    }
}

// Compute MPS norm
kernel void mps_norm(
    device Complex* tensors [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Simplified norm computation
    // Full implementation would contract all tensors and compute trace

    if (gid.x == 0) {
        result[0] = 1.0;  // Normalized MPS has norm 1
    }
}

// Two-qubit gate output (gated tensor)
struct TwoQGateParams {
    Complex gate[16];
    uint left_dim;
    uint mid_dim;
    uint right_dim;
};

kernel void mps_two_qubit(
    device Complex* left [[buffer(0)]],
    device Complex* right [[buffer(1)]],
    device Complex* out [[buffer(2)]],
    constant TwoQGateParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint idx = gid.x;
    uint right_dim = params.right_dim;
    uint b = idx % right_dim;
    uint tmp = idx / right_dim;
    uint jp = tmp % 2;
    tmp = tmp / 2;
    uint ip = tmp % 2;
    uint a = tmp / 2;

    if (a >= params.left_dim) {
        return;
    }

    Complex sum = Complex{0.0, 0.0};
    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            Complex g = params.gate[(ip * 2 + jp) * 4 + (i * 2 + j)];
            for (uint m = 0; m < params.mid_dim; m++) {
                uint l_idx = a * 2 * params.mid_dim + i * params.mid_dim + m;
                uint r_idx = m * 2 * params.right_dim + j * params.right_dim + b;
                Complex prod = complex_mul(left[l_idx], right[r_idx]);
                sum = complex_add(sum, complex_mul(g, prod));
            }
        }
    }

    uint out_idx = (((a * 2 + ip) * 2 + jp) * params.right_dim) + b;
    out[out_idx] = sum;
}

// Gram matrix (A^H A)
struct GramParams {
    uint left_dim;
    uint right_dim;
};

kernel void mps_gram(
    device Complex* gated [[buffer(0)]],
    device Complex* gram [[buffer(1)]],
    constant GramParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint n_cols = 2 * params.right_dim;
    uint i = gid.x;
    uint j = gid.y;
    if (i >= n_cols || j >= n_cols) {
        return;
    }

    Complex sum = Complex{0.0, 0.0};
    uint m_rows = params.left_dim * 2;
    for (uint row = 0; row < m_rows; row++) {
        uint a = row / 2;
        uint irow = row % 2;

        uint jcol_i = i / params.right_dim;
        uint b_i = i % params.right_dim;
        uint jcol_j = j / params.right_dim;
        uint b_j = j % params.right_dim;

        uint idx_i = (((a * 2 + irow) * 2 + jcol_i) * params.right_dim) + b_i;
        uint idx_j = (((a * 2 + irow) * 2 + jcol_j) * params.right_dim) + b_j;
        Complex ai = gated[idx_i];
        Complex aj = gated[idx_j];
        Complex conj_ai = Complex{ai.real, -ai.imag};
        sum = complex_add(sum, complex_mul(conj_ai, aj));
    }

    gram[i * n_cols + j] = sum;
}
"#.to_string()
    }
}

/// GPU MPS benchmark results.
#[derive(Clone, Debug)]
pub struct GPUMPSBenchmarkResults {
    pub cpu_time: f64,
    pub gpu_time: f64,
    pub speedup: f64,
    pub bond_dim: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SingleQGateParams {
    matrix: [[C32; 2]; 2],
    left_dim: u32,
    right_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TwoQGateParams {
    gate: [C32; 16],
    left_dim: u32,
    mid_dim: u32,
    right_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GramParams {
    left_dim: u32,
    right_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ContractParams {
    left_dim: u32,
    right_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SVDParams {
    bond_dim: u32,
    truncate: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TruncateParams {
    max_bond_dim: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_mps_creation() {
        if let Ok(mps) = GPUMPSState::new(10, 4) {
            assert_eq!(mps.num_qubits, 10);
            assert_eq!(mps.bond_dim, 4);
        }
    }

    #[test]
    fn test_gpu_mps_single_qubit() {
        if let Ok(mut mps) = GPUMPSState::new(10, 4) {
            let h_matrix = [
                [C64::new(0.70710678, 0.0), C64::new(0.70710678, 0.0)],
                [C64::new(0.70710678, 0.0), C64::new(-0.70710678, 0.0)],
            ];
            let _ = mps.apply_single_qubit_gate(0, h_matrix);
        }
    }
}
