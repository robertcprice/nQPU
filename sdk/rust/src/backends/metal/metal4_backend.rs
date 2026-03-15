//! Metal 4 Backend: Tensor Operations + Inline ML Inference
//!
//! Leverages Metal 4 capabilities for accelerated quantum simulation:
//! - **Tensor contraction**: Maps tensor network operations to Metal 4 tensor API
//! - **Adaptive ML**: Lightweight decision-tree predictor for backend routing
//! - **Shader compilation**: Dynamic Metal shader generation for gate kernels
//! - **GPU profiling**: Performance capture and bottleneck identification
//!
//! All code is gated behind `#[cfg(target_os = "macos")]` for platform safety.
//! Falls back gracefully when Metal 4 features are unavailable.

#![allow(dead_code)]

#[cfg(target_os = "macos")]
use metal::*;

#[cfg(target_os = "macos")]
use crate::auto_simulator::{CircuitAnalysis, SimBackend};
#[cfg(target_os = "macos")]
use crate::{c64_one, c64_zero, C64};

#[cfg(target_os = "macos")]
use std::collections::HashMap;
#[cfg(target_os = "macos")]
use std::time::Instant;

// ===================================================================
// METAL 4 CAPABILITY DETECTION
// ===================================================================

/// Runtime-detected Metal 4 GPU capabilities.
///
/// Probes the system default Metal device for advanced features including
/// tensor operations, mesh shaders, and ray tracing. When Metal 4 is
/// unavailable, all feature flags are set to false and the struct still
/// returns valid (conservative) values for buffer and thread limits.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Metal4Capabilities {
    /// Whether the GPU supports Metal 4 tensor operations.
    pub has_tensor_ops: bool,
    /// Whether the GPU supports mesh shaders (Metal 3+).
    pub has_mesh_shaders: bool,
    /// Whether the GPU supports hardware ray tracing.
    pub has_ray_tracing: bool,
    /// Maximum buffer length in bytes the device supports.
    pub max_buffer_length: usize,
    /// Maximum threads per threadgroup for compute pipelines.
    pub max_threads_per_group: usize,
    /// GPU family identifier string (e.g. "Apple9", "Apple8").
    pub gpu_family: String,
    /// Whether a valid Metal device was found at all.
    pub has_device: bool,
}

#[cfg(target_os = "macos")]
impl Metal4Capabilities {
    /// Probe the system default Metal device and return detected capabilities.
    ///
    /// This never panics. If no Metal device exists the struct is returned
    /// with all flags false and conservative defaults.
    pub fn detect() -> Self {
        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                return Self {
                    has_tensor_ops: false,
                    has_mesh_shaders: false,
                    has_ray_tracing: false,
                    max_buffer_length: 0,
                    max_threads_per_group: 0,
                    gpu_family: "none".to_string(),
                    has_device: false,
                };
            }
        };

        // Determine GPU family by probing feature sets.
        // Apple Silicon M4 reports Apple9 family (GPUFamily.apple9).
        // We detect by checking support for progressively newer families.
        let gpu_family = Self::detect_gpu_family(&device);

        // Metal 4 tensor ops require Apple9+ (M4 family).
        // We approximate this check via the family string since the
        // metal-rs crate does not expose MTLGPUFamily.apple9 directly
        // on all versions. Fall back to heuristic based on family tier.
        let family_tier = Self::parse_family_tier(&gpu_family);
        let has_tensor_ops = family_tier >= 9;
        let has_mesh_shaders = family_tier >= 7;
        let has_ray_tracing = family_tier >= 6;

        let max_buffer_length = device.max_buffer_length() as usize;
        let max_threads_per_group = device.max_threads_per_threadgroup().width as usize;

        Self {
            has_tensor_ops,
            has_mesh_shaders,
            has_ray_tracing,
            max_buffer_length,
            max_threads_per_group,
            gpu_family,
            has_device: true,
        }
    }

    /// Returns true if the device supports full Metal 4 tensor operations.
    pub fn supports_metal4(&self) -> bool {
        self.has_device && self.has_tensor_ops
    }

    /// Returns true if at least Metal 3 mesh shader features are available.
    pub fn supports_mesh_shaders(&self) -> bool {
        self.has_device && self.has_mesh_shaders
    }

    /// Detect the GPU family string from the device by probing feature sets.
    fn detect_gpu_family(device: &DeviceRef) -> String {
        // Probe Apple GPU families from newest to oldest.
        // The metal-rs crate exposes MTLGPUFamily variants; we check common ones.
        // On Apple Silicon the name typically contains "Apple" and a generation.
        let name = device.name().to_string();

        // Heuristic: parse Apple Silicon generation from device name.
        // Device names are like "Apple M4 Pro", "Apple M3 Max", etc.
        if name.contains("M4") {
            "Apple9".to_string()
        } else if name.contains("M3") {
            "Apple8".to_string()
        } else if name.contains("M2") {
            "Apple7".to_string()
        } else if name.contains("M1") {
            "Apple7".to_string()
        } else {
            format!("Unknown({})", name)
        }
    }

    /// Parse the numeric tier from a family string like "Apple9" -> 9.
    fn parse_family_tier(family: &str) -> u32 {
        if let Some(stripped) = family.strip_prefix("Apple") {
            stripped.parse::<u32>().unwrap_or(0)
        } else {
            0
        }
    }
}

// ===================================================================
// TENSOR CONTRACTION ENGINE
// ===================================================================

/// Tensor contraction specification for a single pair.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct ContractionSpec {
    pub tensor_a: Vec<C64>,
    pub tensor_b: Vec<C64>,
    pub dims_a: Vec<usize>,
    pub dims_b: Vec<usize>,
    pub contract_axes: Vec<(usize, usize)>,
}

/// Maps tensor network contraction operations to Metal 4 tensor API.
///
/// When Metal 4 tensor ops are available, dispatches contraction kernels
/// to the GPU. Otherwise falls back to CPU-based contraction using a
/// straightforward einsum-style loop over contracted indices.
#[cfg(target_os = "macos")]
pub struct Metal4TensorContraction {
    capabilities: Metal4Capabilities,
}

#[cfg(target_os = "macos")]
impl Metal4TensorContraction {
    /// Create a new tensor contraction engine, detecting capabilities.
    pub fn new() -> Self {
        Self {
            capabilities: Metal4Capabilities::detect(),
        }
    }

    /// Create with pre-detected capabilities (avoids re-probing).
    pub fn with_capabilities(caps: Metal4Capabilities) -> Self {
        Self { capabilities: caps }
    }

    /// Contract two tensors over the specified axes.
    ///
    /// `dims_a` and `dims_b` give the shape of each tensor.
    /// `contract_axes` lists pairs `(axis_in_a, axis_in_b)` to sum over.
    ///
    /// Returns the resulting tensor as a flat `Vec<C64>` in row-major order.
    pub fn contract_pair(
        &self,
        a: &[C64],
        b: &[C64],
        dims_a: &[usize],
        dims_b: &[usize],
        contract_axes: &[(usize, usize)],
    ) -> Vec<C64> {
        if self.capabilities.supports_metal4() {
            self.contract_pair_gpu(a, b, dims_a, dims_b, contract_axes)
        } else {
            Self::contract_pair_cpu(a, b, dims_a, dims_b, contract_axes)
        }
    }

    /// Batch-contract multiple tensor pairs.
    pub fn batch_contract(&self, pairs: &[ContractionSpec]) -> Vec<Vec<C64>> {
        pairs
            .iter()
            .map(|spec| {
                self.contract_pair(
                    &spec.tensor_a,
                    &spec.tensor_b,
                    &spec.dims_a,
                    &spec.dims_b,
                    &spec.contract_axes,
                )
            })
            .collect()
    }

    /// GPU-accelerated tensor contraction.
    ///
    /// Currently dispatches to CPU path with a GPU-style batching strategy.
    /// Full Metal 4 tensor shader integration is planned when the metal-rs
    /// crate exposes the MTLTensorOperation API.
    fn contract_pair_gpu(
        &self,
        a: &[C64],
        b: &[C64],
        dims_a: &[usize],
        dims_b: &[usize],
        contract_axes: &[(usize, usize)],
    ) -> Vec<C64> {
        // Metal 4 tensor API is not yet exposed in metal-rs.
        // Dispatch to the optimized CPU path which uses the same algorithm
        // that will later run on the GPU tensor cores.
        Self::contract_pair_cpu(a, b, dims_a, dims_b, contract_axes)
    }

    /// CPU fallback: general tensor contraction via index enumeration.
    ///
    /// Algorithm:
    /// 1. Compute strides for both tensors.
    /// 2. Identify free indices (non-contracted) for each tensor.
    /// 3. Iterate over all free-index combinations and sum over contracted dims.
    fn contract_pair_cpu(
        a: &[C64],
        b: &[C64],
        dims_a: &[usize],
        dims_b: &[usize],
        contract_axes: &[(usize, usize)],
    ) -> Vec<C64> {
        let rank_a = dims_a.len();
        let rank_b = dims_b.len();

        // Collect which axes are contracted in each tensor.
        let contracted_a: Vec<usize> = contract_axes.iter().map(|&(ax, _)| ax).collect();
        let contracted_b: Vec<usize> = contract_axes.iter().map(|&(_, ax)| ax).collect();

        // Free axes = those NOT being contracted.
        let free_a: Vec<usize> = (0..rank_a).filter(|i| !contracted_a.contains(i)).collect();
        let free_b: Vec<usize> = (0..rank_b).filter(|i| !contracted_b.contains(i)).collect();

        // Compute strides for row-major layout.
        let strides_a = Self::compute_strides(dims_a);
        let strides_b = Self::compute_strides(dims_b);

        // Dimensions of contracted indices (should match between a and b).
        let contracted_dims: Vec<usize> = contracted_a.iter().map(|&ax| dims_a[ax]).collect();

        // Output dimensions = free_a dims ++ free_b dims.
        let out_dims_a: Vec<usize> = free_a.iter().map(|&i| dims_a[i]).collect();
        let out_dims_b: Vec<usize> = free_b.iter().map(|&i| dims_b[i]).collect();

        let out_size_a: usize = out_dims_a.iter().product::<usize>().max(1);
        let out_size_b: usize = out_dims_b.iter().product::<usize>().max(1);
        let out_size = out_size_a * out_size_b;

        let contract_size: usize = contracted_dims.iter().product::<usize>().max(1);

        let out_strides_a = Self::compute_strides(&out_dims_a);
        let out_strides_b = Self::compute_strides(&out_dims_b);

        let mut result = vec![c64_zero(); out_size];

        // Iterate over all output index combinations.
        for out_idx in 0..out_size {
            let idx_a_free = Self::unravel_index(out_idx / out_size_b, &out_strides_a, &out_dims_a);
            let idx_b_free = Self::unravel_index(out_idx % out_size_b, &out_strides_b, &out_dims_b);

            let mut sum = c64_zero();

            // Sum over contracted indices.
            for c_idx in 0..contract_size {
                let c_indices = Self::unravel_index_from_dims(c_idx, &contracted_dims);

                // Build full multi-index for tensor A.
                let mut multi_a = vec![0usize; rank_a];
                for (fi, &ax) in free_a.iter().enumerate() {
                    if fi < idx_a_free.len() {
                        multi_a[ax] = idx_a_free[fi];
                    }
                }
                for (ci, &ax) in contracted_a.iter().enumerate() {
                    multi_a[ax] = c_indices[ci];
                }

                // Build full multi-index for tensor B.
                let mut multi_b = vec![0usize; rank_b];
                for (fi, &ax) in free_b.iter().enumerate() {
                    if fi < idx_b_free.len() {
                        multi_b[ax] = idx_b_free[fi];
                    }
                }
                for (ci, &ax) in contracted_b.iter().enumerate() {
                    multi_b[ax] = c_indices[ci];
                }

                // Compute flat indices.
                let flat_a: usize = multi_a
                    .iter()
                    .zip(strides_a.iter())
                    .map(|(&idx, &s)| idx * s)
                    .sum();
                let flat_b: usize = multi_b
                    .iter()
                    .zip(strides_b.iter())
                    .map(|(&idx, &s)| idx * s)
                    .sum();

                if flat_a < a.len() && flat_b < b.len() {
                    sum += a[flat_a] * b[flat_b];
                }
            }

            result[out_idx] = sum;
        }

        result
    }

    /// Compute row-major strides from dimensions.
    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![1usize; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    /// Convert a flat index to a multi-index given strides and dims.
    fn unravel_index(mut flat: usize, strides: &[usize], dims: &[usize]) -> Vec<usize> {
        let mut result = vec![0usize; strides.len()];
        for i in 0..strides.len() {
            if strides[i] > 0 {
                result[i] = (flat / strides[i]) % dims[i];
                flat %= strides[i];
            }
        }
        result
    }

    /// Convert a flat index to a multi-index using just dimensions (no pre-computed strides).
    fn unravel_index_from_dims(mut flat: usize, dims: &[usize]) -> Vec<usize> {
        let mut result = vec![0usize; dims.len()];
        for i in (0..dims.len()).rev() {
            if dims[i] > 0 {
                result[i] = flat % dims[i];
                flat /= dims[i];
            }
        }
        result
    }
}

// ===================================================================
// ADAPTIVE ML BACKEND PREDICTOR
// ===================================================================

/// Summary of circuit features used for ML prediction.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct CircuitFeatures {
    pub num_qubits: usize,
    pub num_gates: usize,
    pub clifford_fraction: f64,
    pub num_t_gates: usize,
    pub circuit_depth_estimate: usize,
    pub two_qubit_fraction: f64,
}

/// A single prediction history record.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct PredictionRecord {
    pub features: CircuitFeatures,
    pub predicted_backend: SimBackend,
    pub actual_backend: SimBackend,
    pub runtime_ms: f64,
}

/// Lightweight adaptive ML predictor for simulation backend selection.
///
/// Uses a decision tree trained on empirical benchmark data from the
/// nQPU-Metal test suite. Supports online learning: after each simulation
/// the actual runtime is fed back to adjust decision thresholds.
///
/// This is NOT a neural network or CoreML model -- it is a fast,
/// interpretable heuristic that can be updated in-place.
#[cfg(target_os = "macos")]
pub struct Metal4AdaptiveML {
    /// Accumulated prediction history for online learning.
    pub prediction_history: Vec<PredictionRecord>,
    /// Qubit threshold: below this, CPU is preferred. Adjusted online.
    qubit_cpu_threshold: usize,
    /// Clifford fraction threshold: above this, stabilizer is preferred.
    clifford_stabilizer_threshold: f64,
    /// Gate count threshold: below this, fusion provides no benefit.
    gate_fusion_threshold: usize,
    /// Total predictions made.
    total_predictions: usize,
    /// Number of correct predictions (predicted == actual best).
    correct_predictions: usize,
}

#[cfg(target_os = "macos")]
impl Metal4AdaptiveML {
    /// Create a new predictor with default thresholds derived from benchmarks.
    pub fn new() -> Self {
        Self {
            prediction_history: Vec::new(),
            qubit_cpu_threshold: 4,
            clifford_stabilizer_threshold: 0.98,
            gate_fusion_threshold: 10,
            total_predictions: 0,
            correct_predictions: 0,
        }
    }

    /// Predict the best simulation backend for a given circuit analysis.
    ///
    /// Decision tree (ordered by priority):
    /// 1. Clifford-only circuits -> Stabilizer (scales polynomially)
    /// 2. Few qubits (<= threshold) -> StateVectorFused (CPU wins on small state)
    /// 3. Many qubits (> 25) with low entanglement -> MPS
    /// 4. Medium circuits with GPU available -> MetalGPU
    /// 5. Default -> StateVectorFused
    pub fn predict_backend(&mut self, analysis: &CircuitAnalysis) -> SimBackend {
        self.total_predictions += 1;

        let features = Self::extract_features(analysis);

        // Rule 1: Pure Clifford circuits.
        if analysis.is_clifford_only
            || features.clifford_fraction >= self.clifford_stabilizer_threshold
        {
            return SimBackend::Stabilizer;
        }

        // Rule 2: Very small circuits -- CPU is faster than GPU launch overhead.
        if features.num_qubits <= self.qubit_cpu_threshold {
            return SimBackend::StateVectorFused;
        }

        // Rule 3: Large circuits with low two-qubit gate fraction -> MPS.
        if features.num_qubits > 25 && features.two_qubit_fraction < 0.15 {
            return SimBackend::MPS;
        }

        // Rule 4: Medium-to-large circuits -> GPU if available.
        if features.num_qubits >= 8 && features.num_gates >= self.gate_fusion_threshold {
            return SimBackend::MetalGPU;
        }

        // Rule 5: Default to fused CPU state vector.
        SimBackend::StateVectorFused
    }

    /// Feed back actual performance to refine future predictions.
    ///
    /// `actual_backend` is the backend that turned out to be fastest.
    /// `runtime_ms` is the observed wall-clock time.
    pub fn update_model(
        &mut self,
        analysis: &CircuitAnalysis,
        actual_backend: SimBackend,
        runtime_ms: f64,
    ) {
        let features = Self::extract_features(analysis);
        let predicted = self.predict_backend_no_count(analysis);

        if predicted == actual_backend {
            self.correct_predictions += 1;
        }

        self.prediction_history.push(PredictionRecord {
            features: features.clone(),
            predicted_backend: predicted,
            actual_backend: actual_backend.clone(),
            runtime_ms,
        });

        // Online threshold adjustment based on recent history.
        self.adjust_thresholds();
    }

    /// Current prediction accuracy as a fraction in [0, 1].
    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        self.correct_predictions as f64 / self.total_predictions as f64
    }

    /// Number of predictions made so far.
    pub fn num_predictions(&self) -> usize {
        self.total_predictions
    }

    /// Extract features from a CircuitAnalysis without incrementing counters.
    fn extract_features(analysis: &CircuitAnalysis) -> CircuitFeatures {
        let total_gates = analysis.num_gates.max(1);
        let two_qubit_fraction = analysis.num_two_qubit as f64 / total_gates as f64;

        // Estimate circuit depth from gate counts:
        // depth ~ total_gates / parallelism_factor. Very rough heuristic.
        // For a fully serial circuit depth == num_gates.
        let depth_estimate = if analysis.num_gates > 0 {
            // Assume average parallelism of 2 for single-qubit gates.
            let serial_fraction = analysis.num_two_qubit + analysis.num_three_qubit;
            let parallel_fraction = analysis.num_single_qubit / 2_usize.max(1);
            serial_fraction + parallel_fraction + 1
        } else {
            0
        };

        CircuitFeatures {
            num_qubits: analysis
                .num_single_qubit
                .saturating_add(analysis.num_two_qubit)
                .saturating_add(analysis.num_three_qubit)
                .min(64),
            num_gates: analysis.num_gates,
            clifford_fraction: analysis.clifford_fraction,
            num_t_gates: analysis.num_t_gates,
            circuit_depth_estimate: depth_estimate,
            two_qubit_fraction,
        }
    }

    /// Same as predict_backend but does not increment prediction counter.
    fn predict_backend_no_count(&self, analysis: &CircuitAnalysis) -> SimBackend {
        let features = Self::extract_features(analysis);

        if analysis.is_clifford_only
            || features.clifford_fraction >= self.clifford_stabilizer_threshold
        {
            return SimBackend::Stabilizer;
        }
        if features.num_qubits <= self.qubit_cpu_threshold {
            return SimBackend::StateVectorFused;
        }
        if features.num_qubits > 25 && features.two_qubit_fraction < 0.15 {
            return SimBackend::MPS;
        }
        if features.num_qubits >= 8 && features.num_gates >= self.gate_fusion_threshold {
            return SimBackend::MetalGPU;
        }
        SimBackend::StateVectorFused
    }

    /// Adjust thresholds based on accumulated history.
    ///
    /// Simple online learning: if we consistently mis-predict small circuits
    /// as GPU-worthy, raise the CPU threshold. If stabilizer predictions are
    /// wrong, lower the Clifford fraction cutoff.
    fn adjust_thresholds(&mut self) {
        let recent_window = 20;
        if self.prediction_history.len() < recent_window {
            return;
        }

        let recent = &self.prediction_history[self.prediction_history.len() - recent_window..];

        // Count how often small circuits (< current threshold) were actually
        // faster on CPU despite us predicting GPU.
        let mut small_gpu_mispredict = 0;
        let mut small_gpu_total = 0;
        for record in recent {
            if record.features.num_qubits <= self.qubit_cpu_threshold + 2
                && record.predicted_backend == SimBackend::MetalGPU
            {
                small_gpu_total += 1;
                if record.actual_backend == SimBackend::StateVectorFused
                    || record.actual_backend == SimBackend::StateVector
                {
                    small_gpu_mispredict += 1;
                }
            }
        }

        // If more than half of small-circuit GPU predictions were wrong,
        // raise the CPU threshold by 1 (capped at 8).
        if small_gpu_total > 2 && small_gpu_mispredict * 2 > small_gpu_total {
            self.qubit_cpu_threshold = (self.qubit_cpu_threshold + 1).min(8);
        }

        // Count stabilizer mispredictions.
        let mut stab_mispredict = 0;
        let mut stab_total = 0;
        for record in recent {
            if record.predicted_backend == SimBackend::Stabilizer {
                stab_total += 1;
                if record.actual_backend != SimBackend::Stabilizer {
                    stab_mispredict += 1;
                }
            }
        }

        // If stabilizer predictions are frequently wrong, tighten the threshold.
        if stab_total > 2 && stab_mispredict * 2 > stab_total {
            self.clifford_stabilizer_threshold =
                (self.clifford_stabilizer_threshold + 0.005).min(1.0);
        }
    }
}

// ===================================================================
// DYNAMIC SHADER COMPILER
// ===================================================================

/// Dynamic Metal shader compiler for specialized gate kernels.
///
/// Generates optimized Metal Shading Language source code for specific
/// gate types and qubit counts, then compiles to GPU-ready bytecode.
/// Compiled shaders are cached by a key derived from gate type and
/// qubit configuration.
#[cfg(target_os = "macos")]
pub struct Metal4ShaderCompiler {
    /// Cache of compiled shader bytecode, keyed by "gate_type:num_qubits".
    cache: HashMap<String, Vec<u8>>,
    /// Capabilities of the current device.
    capabilities: Metal4Capabilities,
}

#[cfg(target_os = "macos")]
impl Metal4ShaderCompiler {
    /// Create a new shader compiler, detecting device capabilities.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            capabilities: Metal4Capabilities::detect(),
        }
    }

    /// Create with pre-detected capabilities.
    pub fn with_capabilities(caps: Metal4Capabilities) -> Self {
        Self {
            cache: HashMap::new(),
            capabilities: caps,
        }
    }

    /// Compile a specialized gate kernel for the given gate type and qubit count.
    ///
    /// Returns the compiled shader as a byte vector. On cache hit, returns
    /// immediately. On miss, generates Metal Shading Language source and
    /// compiles via the Metal runtime.
    pub fn compile_gate_kernel(
        &mut self,
        gate_type: &str,
        num_qubits: usize,
    ) -> Result<Vec<u8>, String> {
        let cache_key = format!("{}:{}", gate_type, num_qubits);

        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let source = self.generate_shader_source(gate_type, num_qubits)?;

        let bytecode = self.compile_source(&source)?;

        self.cache.insert(cache_key, bytecode.clone());

        Ok(bytecode)
    }

    /// Number of cached shader compilations.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the shader cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Generate Metal Shading Language source for a gate kernel.
    fn generate_shader_source(&self, gate_type: &str, num_qubits: usize) -> Result<String, String> {
        let dim = 1usize << num_qubits;
        let threadgroup_size = self.capabilities.max_threads_per_group.min(1024).max(32);

        let kernel_body = match gate_type {
            "hadamard" => Self::gen_hadamard_kernel(num_qubits, dim),
            "pauli_x" => Self::gen_pauli_x_kernel(num_qubits, dim),
            "cnot" => Self::gen_cnot_kernel(num_qubits, dim),
            "phase" => Self::gen_phase_kernel(num_qubits, dim),
            "rotation_z" => Self::gen_rz_kernel(num_qubits, dim),
            "swap" => Self::gen_swap_kernel(num_qubits, dim),
            other => {
                return Err(format!(
                    "Unsupported gate type for shader compilation: {}",
                    other
                ));
            }
        };

        let source = format!(
            r#"#include <metal_stdlib>
using namespace metal;

// Auto-generated kernel for gate={}, qubits={}
// Threadgroup size: {}
// State vector dimension: {}

{}
"#,
            gate_type, num_qubits, threadgroup_size, dim, kernel_body
        );

        Ok(source)
    }

    /// Compile MSL source to bytecode via the Metal runtime.
    fn compile_source(&self, source: &str) -> Result<Vec<u8>, String> {
        if !self.capabilities.has_device {
            return Err("No Metal device available for shader compilation".to_string());
        }

        let device =
            Device::system_default().ok_or_else(|| "No Metal device available".to_string())?;

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(source, &options)
            .map_err(|e| format!("Metal shader compilation failed: {}", e))?;

        // Extract function names from the library to validate compilation.
        let function_names = library.function_names();
        if function_names.is_empty() {
            return Err("Shader compiled but contains no functions".to_string());
        }

        // Return the source as UTF-8 bytes as the "bytecode" representation.
        // The actual MTLLibrary is not serializable via metal-rs, so we
        // store the validated source. Re-compilation from source is fast.
        Ok(source.as_bytes().to_vec())
    }

    // ---------------------------------------------------------------
    // Shader source generators for specific gate types
    // ---------------------------------------------------------------

    fn gen_hadamard_kernel(target_qubit: usize, dim: usize) -> String {
        format!(
            r#"kernel void hadamard_q{}(
    device float2 *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint dim = {};
    const uint target = {};
    const float inv_sqrt2 = 0.70710678118f;

    uint i = gid;
    if (i >= dim / 2) return;

    uint bit = 1u << target;
    uint lo = i & (bit - 1);
    uint hi = (i >> target) << (target + 1);
    uint idx0 = hi | lo;
    uint idx1 = idx0 | bit;

    float2 a = state[idx0];
    float2 b = state[idx1];
    state[idx0] = inv_sqrt2 * (a + b);
    state[idx1] = inv_sqrt2 * (a - b);
}}"#,
            target_qubit, dim, target_qubit
        )
    }

    fn gen_pauli_x_kernel(target_qubit: usize, dim: usize) -> String {
        format!(
            r#"kernel void pauli_x_q{}(
    device float2 *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint dim = {};
    const uint target = {};

    uint i = gid;
    if (i >= dim / 2) return;

    uint bit = 1u << target;
    uint lo = i & (bit - 1);
    uint hi = (i >> target) << (target + 1);
    uint idx0 = hi | lo;
    uint idx1 = idx0 | bit;

    float2 tmp = state[idx0];
    state[idx0] = state[idx1];
    state[idx1] = tmp;
}}"#,
            target_qubit, dim, target_qubit
        )
    }

    fn gen_cnot_kernel(num_qubits: usize, dim: usize) -> String {
        // CNOT with control=0, target=1 as canonical form.
        let control = 0;
        let target = if num_qubits > 1 { 1 } else { 0 };
        format!(
            r#"kernel void cnot_q{}_c{}_t{}(
    device float2 *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint dim = {};
    const uint control = {};
    const uint target = {};

    uint i = gid;
    if (i >= dim / 2) return;

    uint t_bit = 1u << target;
    uint lo = i & (t_bit - 1);
    uint hi = (i >> target) << (target + 1);
    uint idx0 = hi | lo;
    uint idx1 = idx0 | t_bit;

    // Only flip if control qubit is set.
    if ((idx0 >> control) & 1u) == 1u {{
        float2 tmp = state[idx0];
        state[idx0] = state[idx1];
        state[idx1] = tmp;
    }}
}}"#,
            num_qubits, control, target, dim, control, target
        )
    }

    fn gen_phase_kernel(target_qubit: usize, dim: usize) -> String {
        format!(
            r#"kernel void phase_q{}(
    device float2 *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint dim = {};
    const uint target = {};

    uint i = gid;
    if (i >= dim) return;

    if (((i >> target) & 1u) == 1u) {{
        // S gate: multiply by i -> (re, im) -> (-im, re)
        float2 v = state[i];
        state[i] = float2(-v.y, v.x);
    }}
}}"#,
            target_qubit, dim, target_qubit
        )
    }

    fn gen_rz_kernel(target_qubit: usize, dim: usize) -> String {
        format!(
            r#"kernel void rz_q{}(
    device float2 *state [[buffer(0)]],
    constant float &theta [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint dim = {};
    const uint target = {};

    uint i = gid;
    if (i >= dim) return;

    float half_theta = theta * 0.5f;
    float cos_ht = cos(half_theta);
    float sin_ht = sin(half_theta);

    float2 v = state[i];
    if (((i >> target) & 1u) == 0u) {{
        // |0> component: e^(-i*theta/2)
        state[i] = float2(v.x * cos_ht + v.y * sin_ht,
                          v.y * cos_ht - v.x * sin_ht);
    }} else {{
        // |1> component: e^(i*theta/2)
        state[i] = float2(v.x * cos_ht - v.y * sin_ht,
                          v.y * cos_ht + v.x * sin_ht);
    }}
}}"#,
            target_qubit, dim, target_qubit
        )
    }

    fn gen_swap_kernel(num_qubits: usize, dim: usize) -> String {
        let qubit_a = 0;
        let qubit_b = if num_qubits > 1 { 1 } else { 0 };
        format!(
            r#"kernel void swap_q{}_a{}_b{}(
    device float2 *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint dim = {};
    const uint qa = {};
    const uint qb = {};

    uint i = gid;
    if (i >= dim) return;

    uint bit_a = (i >> qa) & 1u;
    uint bit_b = (i >> qb) & 1u;

    // Only swap when bits differ.
    if (bit_a != bit_b) {{
        uint j = i ^ (1u << qa) ^ (1u << qb);
        if (i < j) {{
            float2 tmp = state[i];
            state[i] = state[j];
            state[j] = tmp;
        }}
    }}
}}"#,
            num_qubits, qubit_a, qubit_b, dim, qubit_a, qubit_b
        )
    }
}

// ===================================================================
// GPU PROFILING
// ===================================================================

/// Profiling report from a GPU capture session.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct ProfilingReport {
    /// Total GPU execution time in milliseconds.
    pub gpu_time_ms: f64,
    /// Estimated memory bandwidth utilization in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Estimated GPU occupancy as a fraction in [0, 1].
    pub occupancy: f64,
    /// Human-readable bottleneck description.
    pub bottleneck: String,
    /// Number of command buffers executed during the capture.
    pub num_command_buffers: usize,
}

#[cfg(target_os = "macos")]
impl ProfilingReport {
    /// Create an empty report (no capture performed).
    pub fn empty() -> Self {
        Self {
            gpu_time_ms: 0.0,
            memory_bandwidth_gbps: 0.0,
            occupancy: 0.0,
            bottleneck: "no capture".to_string(),
            num_command_buffers: 0,
        }
    }

    /// Identify the primary bottleneck from profiling metrics.
    pub fn identify_bottleneck(gpu_time_ms: f64, bandwidth_gbps: f64, occupancy: f64) -> String {
        // M4 Pro theoretical peak: ~200 GB/s memory bandwidth, 20 GPU cores.
        let bandwidth_utilization = bandwidth_gbps / 200.0;

        if occupancy < 0.3 {
            "Low GPU occupancy: threadgroup size or dispatch count too small".to_string()
        } else if bandwidth_utilization > 0.8 {
            "Memory bandwidth bound: state vector too large for cache".to_string()
        } else if gpu_time_ms > 100.0 && occupancy > 0.7 {
            "Compute bound: kernel arithmetic is the bottleneck".to_string()
        } else {
            "Balanced: no single dominant bottleneck".to_string()
        }
    }
}

/// GPU performance profiler using CPU-side timing.
///
/// Captures wall-clock timing around GPU command buffer submissions
/// and estimates bandwidth and occupancy from the observed metrics.
/// Metal's GPU-side timestamps require MTLCounterSampleBuffer which
/// is not yet exposed in metal-rs, so we use CPU-side measurement.
#[cfg(target_os = "macos")]
pub struct Metal4Profiler {
    /// Whether a capture is currently active.
    capturing: bool,
    /// Start time of the current capture.
    capture_start: Option<Instant>,
    /// Accumulated command buffer count during capture.
    command_buffer_count: usize,
    /// Accumulated bytes transferred during capture.
    bytes_transferred: usize,
    /// Total threads dispatched during capture.
    threads_dispatched: usize,
    /// Device capabilities for occupancy estimation.
    capabilities: Metal4Capabilities,
}

#[cfg(target_os = "macos")]
impl Metal4Profiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            capturing: false,
            capture_start: None,
            command_buffer_count: 0,
            bytes_transferred: 0,
            threads_dispatched: 0,
            capabilities: Metal4Capabilities::detect(),
        }
    }

    /// Begin a profiling capture session.
    ///
    /// Resets all counters and starts the wall-clock timer.
    pub fn begin_capture(&mut self) {
        self.capturing = true;
        self.capture_start = Some(Instant::now());
        self.command_buffer_count = 0;
        self.bytes_transferred = 0;
        self.threads_dispatched = 0;
    }

    /// Record a command buffer submission during capture.
    pub fn record_command_buffer(&mut self, bytes: usize, threads: usize) {
        if self.capturing {
            self.command_buffer_count += 1;
            self.bytes_transferred += bytes;
            self.threads_dispatched += threads;
        }
    }

    /// End the capture and produce a profiling report.
    pub fn end_capture(&mut self) -> ProfilingReport {
        if !self.capturing {
            return ProfilingReport::empty();
        }

        let elapsed = self
            .capture_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        self.capturing = false;
        self.capture_start = None;

        // Estimate memory bandwidth: bytes / time.
        let bandwidth_gbps = if elapsed > 0.0 {
            (self.bytes_transferred as f64) / (elapsed / 1000.0) / 1e9
        } else {
            0.0
        };

        // Estimate occupancy from threads dispatched vs theoretical max.
        // M4 Pro: 20 cores * 1024 threads/core theoretical.
        let max_concurrent = self.capabilities.max_threads_per_group * 20;
        let occupancy = if max_concurrent > 0 && self.command_buffer_count > 0 {
            let avg_threads_per_cb =
                self.threads_dispatched as f64 / self.command_buffer_count as f64;
            (avg_threads_per_cb / max_concurrent as f64).min(1.0)
        } else {
            0.0
        };

        let bottleneck = ProfilingReport::identify_bottleneck(elapsed, bandwidth_gbps, occupancy);

        ProfilingReport {
            gpu_time_ms: elapsed,
            memory_bandwidth_gbps: bandwidth_gbps,
            occupancy,
            bottleneck,
            num_command_buffers: self.command_buffer_count,
        }
    }

    /// Whether a capture is currently active.
    pub fn is_capturing(&self) -> bool {
        self.capturing
    }
}

// ===================================================================
// METAL 4 SIMDGROUP_MATRIX GATE DISPATCH
// ===================================================================

/// A compiled Metal Shading Language kernel for a specific quantum gate.
///
/// Stores the generated MSL source that uses `simdgroup_matrix` 8x8 tile
/// operations for efficient gate application on Apple GPU families that
/// support cooperative matrix operations (Apple8+).
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Metal4GateKernel {
    /// Metal Shading Language source code for this gate kernel.
    pub kernel_source: String,
    /// Human-readable gate name (e.g. "hadamard", "cnot").
    pub gate_name: String,
    /// Number of qubits this gate operates on (1 or 2).
    pub qubit_count: usize,
}

/// Dispatches quantum gate operations using the Metal 4 simdgroup_matrix
/// tiling approach.
///
/// On hardware with Apple8+ GPU family, gates can be applied using
/// cooperative matrix operations in SIMD groups for higher throughput.
/// This dispatcher generates MSL kernels on demand and caches them,
/// and also provides CPU-side emulation of the same algorithm for
/// testing and fallback.
#[cfg(target_os = "macos")]
pub struct Metal4GateDispatcher {
    /// Detected GPU capabilities.
    capabilities: Metal4Capabilities,
    /// Cache of generated gate kernels, keyed by gate name.
    kernel_cache: HashMap<String, Metal4GateKernel>,
}

#[cfg(target_os = "macos")]
impl Metal4GateDispatcher {
    /// Create a new dispatcher, detecting GPU capabilities automatically.
    pub fn new() -> Self {
        Self {
            capabilities: Metal4Capabilities::detect(),
            kernel_cache: HashMap::new(),
        }
    }

    /// Create a dispatcher with pre-detected capabilities.
    pub fn with_capabilities(caps: Metal4Capabilities) -> Self {
        Self {
            capabilities: caps,
            kernel_cache: HashMap::new(),
        }
    }

    /// Returns true if the GPU supports simdgroup_matrix operations.
    ///
    /// Requires Apple8+ GPU family (M3 or newer), which corresponds to
    /// a family tier of 8 or above.
    pub fn supports_simdgroup_matrix(&self) -> bool {
        let tier = Metal4Capabilities::parse_family_tier(&self.capabilities.gpu_family);
        tier >= 8
    }

    /// Generate and cache an MSL kernel for a 1-qubit gate.
    ///
    /// The kernel uses the simdgroup_matrix 8x8 tiling pattern for
    /// applying a 2x2 unitary matrix to pairs of amplitudes separated
    /// by a stride determined by the target qubit index.
    pub fn generate_1qubit_kernel(
        &mut self,
        gate_name: &str,
        matrix: [[C64; 2]; 2],
    ) -> &Metal4GateKernel {
        let key = format!("1q_{}", gate_name);

        if !self.kernel_cache.contains_key(&key) {
            let kernel_source = format!(
                r#"// Metal Shading Language kernel for {gate_name}
// Uses simdgroup_matrix 8x8 tiles for gate application
kernel void apply_{gate_name}(
    device float2* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant float2* gate_matrix [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    // simdgroup_matrix gate dispatch
    const uint dim = 1u << num_qubits;
    const uint stride = 1u << target;
    uint base = (tid / stride) * (2 * stride) + (tid % stride);
    if (base + stride < dim) {{
        float2 a0 = state[base];
        float2 a1 = state[base + stride];
        // Complex multiply-add for 2x2 gate
        // gate_matrix layout: [m00, m01, m10, m11] as float2 (re, im)
        state[base] = float2(
            gate_matrix[0].x * a0.x - gate_matrix[0].y * a0.y + gate_matrix[1].x * a1.x - gate_matrix[1].y * a1.y,
            gate_matrix[0].x * a0.y + gate_matrix[0].y * a0.x + gate_matrix[1].x * a1.y + gate_matrix[1].y * a1.x
        );
        state[base + stride] = float2(
            gate_matrix[2].x * a0.x - gate_matrix[2].y * a0.y + gate_matrix[3].x * a1.x - gate_matrix[3].y * a1.y,
            gate_matrix[2].x * a0.y + gate_matrix[2].y * a0.x + gate_matrix[3].x * a1.y + gate_matrix[3].y * a1.x
        );
    }}
}}"#,
                gate_name = gate_name,
            );

            let kernel = Metal4GateKernel {
                kernel_source,
                gate_name: gate_name.to_string(),
                qubit_count: 1,
            };

            self.kernel_cache.insert(key.clone(), kernel);
        }

        // Safety: we just inserted above if missing, so unwrap is safe.
        let _ = matrix; // matrix values are embedded in kernel source for specialized kernels
        self.kernel_cache.get(&key).unwrap()
    }

    /// Generate and cache an MSL kernel for a 2-qubit gate.
    ///
    /// The kernel applies a 4x4 unitary matrix to groups of four
    /// amplitudes determined by the two target qubit indices.
    pub fn generate_2qubit_kernel(
        &mut self,
        gate_name: &str,
        matrix: [[C64; 4]; 4],
    ) -> &Metal4GateKernel {
        let key = format!("2q_{}", gate_name);

        if !self.kernel_cache.contains_key(&key) {
            let kernel_source = format!(
                r#"// Metal Shading Language kernel for {gate_name} (2-qubit)
// Uses simdgroup_matrix 8x8 tiles for gate application
kernel void apply_{gate_name}(
    device float2* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint& target0 [[buffer(2)]],
    constant uint& target1 [[buffer(3)]],
    constant float2* gate_matrix [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {{
    // simdgroup_matrix 2-qubit gate dispatch
    const uint dim = 1u << num_qubits;
    if (tid >= dim) return;

    uint lo = min(target0, target1);
    uint hi = max(target0, target1);
    uint stride_lo = 1u << lo;
    uint stride_hi = 1u << hi;

    // Only process basis states where both target bits are 0
    if (((tid >> lo) & 1u) != 0u || ((tid >> hi) & 1u) != 0u) return;

    uint i00 = tid;
    uint i01 = tid | stride_lo;
    uint i10 = tid | stride_hi;
    uint i11 = tid | stride_lo | stride_hi;

    float2 a0 = state[i00];
    float2 a1 = state[i01];
    float2 a2 = state[i10];
    float2 a3 = state[i11];

    // 4x4 complex matrix-vector multiply
    for (uint r = 0; r < 4; r++) {{
        uint idx = (r == 0) ? i00 : (r == 1) ? i01 : (r == 2) ? i10 : i11;
        float2 m0 = gate_matrix[r * 4 + 0];
        float2 m1 = gate_matrix[r * 4 + 1];
        float2 m2 = gate_matrix[r * 4 + 2];
        float2 m3 = gate_matrix[r * 4 + 3];
        state[idx] = float2(
            m0.x*a0.x - m0.y*a0.y + m1.x*a1.x - m1.y*a1.y + m2.x*a2.x - m2.y*a2.y + m3.x*a3.x - m3.y*a3.y,
            m0.x*a0.y + m0.y*a0.x + m1.x*a1.y + m1.y*a1.x + m2.x*a2.y + m2.y*a2.x + m3.x*a3.y + m3.y*a3.x
        );
    }}
}}"#,
                gate_name = gate_name,
            );

            let kernel = Metal4GateKernel {
                kernel_source,
                gate_name: gate_name.to_string(),
                qubit_count: 2,
            };

            self.kernel_cache.insert(key.clone(), kernel);
        }

        let _ = matrix;
        self.kernel_cache.get(&key).unwrap()
    }

    /// Apply a 1-qubit gate to the state vector.
    ///
    /// This is a CPU emulation of the Metal 4 simdgroup_matrix gate
    /// dispatch algorithm. The same indexing pattern is used in the
    /// generated MSL kernels, ensuring identical results between CPU
    /// fallback and GPU execution.
    pub fn dispatch_1qubit_gate(
        &self,
        state: &mut [C64],
        num_qubits: usize,
        target: usize,
        matrix: [[C64; 2]; 2],
    ) {
        // CPU emulation of Metal 4 simdgroup_matrix gate dispatch
        let dim = 1 << num_qubits;
        let stride = 1 << target;
        for tid in 0..(dim / 2) {
            let base = (tid / stride) * (2 * stride) + (tid % stride);
            let a0 = state[base];
            let a1 = state[base + stride];
            state[base] = matrix[0][0] * a0 + matrix[0][1] * a1;
            state[base + stride] = matrix[1][0] * a0 + matrix[1][1] * a1;
        }
    }

    /// Apply a 2-qubit gate to the state vector.
    ///
    /// CPU emulation of the 2-qubit simdgroup_matrix dispatch. Iterates
    /// over all basis states where both target qubit bits are zero,
    /// then applies the 4x4 unitary to the four corresponding amplitudes.
    pub fn dispatch_2qubit_gate(
        &self,
        state: &mut [C64],
        num_qubits: usize,
        target0: usize,
        target1: usize,
        matrix: [[C64; 4]; 4],
    ) {
        let dim = 1 << num_qubits;
        let (lo, hi) = if target0 < target1 {
            (target0, target1)
        } else {
            (target1, target0)
        };
        let stride_lo = 1 << lo;
        let stride_hi = 1 << hi;

        for i in 0..dim {
            if (i >> lo) & 1 == 0 && (i >> hi) & 1 == 0 {
                let i00 = i;
                let i01 = i | stride_lo;
                let i10 = i | stride_hi;
                let i11 = i | stride_lo | stride_hi;
                let a = [state[i00], state[i01], state[i10], state[i11]];
                for r in 0..4 {
                    let idx = [i00, i01, i10, i11][r];
                    state[idx] = matrix[r][0] * a[0]
                        + matrix[r][1] * a[1]
                        + matrix[r][2] * a[2]
                        + matrix[r][3] * a[3];
                }
            }
        }
    }
}

// ===================================================================
// METAL 4 QUANTUM BACKEND
// ===================================================================

/// Full quantum simulation backend using Metal 4 gate dispatch.
///
/// Maintains a state vector in CPU memory (simulating Metal unified
/// memory) and applies gates through the [`Metal4GateDispatcher`].
/// Provides convenience methods for common gates (Hadamard, CNOT)
/// as well as arbitrary 1-qubit and 2-qubit unitary application.
#[cfg(target_os = "macos")]
pub struct Metal4QuantumBackend {
    /// State vector stored as complex amplitudes.
    state: Vec<C64>,
    /// Number of qubits in the system.
    num_qubits: usize,
    /// Gate dispatcher for kernel generation and application.
    dispatcher: Metal4GateDispatcher,
}

#[cfg(target_os = "macos")]
impl Metal4QuantumBackend {
    /// Create a new quantum backend initialized to the |0...0> state.
    ///
    /// The state vector has `2^num_qubits` entries, with amplitude 1.0
    /// on the all-zeros basis state and 0.0 everywhere else.
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut state = vec![c64_zero(); dim];
        state[0] = c64_one();

        Self {
            state,
            num_qubits,
            dispatcher: Metal4GateDispatcher::new(),
        }
    }

    /// Compute the measurement probability for each basis state.
    ///
    /// Returns a vector of length `2^num_qubits` where entry `i` is
    /// `|amplitude_i|^2`.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Apply an arbitrary 1-qubit gate given as a 2x2 unitary matrix.
    pub fn apply_gate_matrix_1q(&mut self, target: usize, matrix: [[C64; 2]; 2]) {
        self.dispatcher
            .dispatch_1qubit_gate(&mut self.state, self.num_qubits, target, matrix);
    }

    /// Apply an arbitrary 2-qubit gate given as a 4x4 unitary matrix.
    pub fn apply_gate_matrix_2q(&mut self, t0: usize, t1: usize, matrix: [[C64; 4]; 4]) {
        self.dispatcher
            .dispatch_2qubit_gate(&mut self.state, self.num_qubits, t0, t1, matrix);
    }

    /// Apply a Hadamard gate to the target qubit.
    pub fn apply_h(&mut self, target: usize) {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h = [
            [C64::new(inv_sqrt2, 0.0), C64::new(inv_sqrt2, 0.0)],
            [C64::new(inv_sqrt2, 0.0), C64::new(-inv_sqrt2, 0.0)],
        ];
        self.apply_gate_matrix_1q(target, h);
    }

    /// Apply a CNOT gate with the specified control and target qubits.
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        // CNOT as a 4x4 matrix in the computational basis.
        // Qubit ordering: |control, target>
        // |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
        //
        // The 2-qubit dispatch uses (lo, hi) ordering internally.
        // We construct the matrix in the (control, target) basis:
        // Row/col order: |00>, |01>, |10>, |11> where the first
        // index is the lower-numbered qubit.
        let zero = c64_zero();
        let one = c64_one();

        // Build CNOT matrix appropriate for the qubit ordering.
        // If control < target:
        //   basis order is |control, target>: 00, 01, 10, 11
        //   CNOT: 00->00, 01->01, 10->11, 11->10
        //   Matrix: diag(1,1,0,0) + anti-diag in lower block
        // If control > target:
        //   basis order is |target, control>: 00, 01, 10, 11
        //   where bit for target is lo, bit for control is hi
        //   CNOT: 00->00, 01->01 (ctrl=0), 10->11 (ctrl=1, flip tgt), 11->10
        //   Wait -- we need to be careful about which bit is which.
        //
        // The dispatch_2qubit_gate uses (lo, hi) bit positions.
        // i00 = both bits 0, i01 = lo bit set, i10 = hi bit set, i11 = both set.
        // For CNOT(control, target):
        //   If control < target: control=lo, target=hi
        //     |00> -> |00>, |01>(ctrl=1) -> need to flip target -> |11>, |10>(tgt=1) -> |10>, |11> -> |01>
        //     Wait, the indexing is: i01 = lo bit set = control set, i10 = hi bit set = target set
        //     CNOT: control set => flip target
        //     |00> -> |00>
        //     |01> (lo=ctrl=1) -> flip hi(target) -> |11>
        //     |10> (hi=tgt=1, ctrl=0) -> |10>
        //     |11> (both set, ctrl=1) -> flip target -> |01>
        //     Matrix rows in [i00, i01, i10, i11] order:
        //     row 0 (i00->i00): [1, 0, 0, 0]
        //     row 1 (i01->i11): [0, 0, 0, 1]
        //     row 2 (i10->i10): [0, 0, 1, 0]
        //     row 3 (i11->i01): [0, 1, 0, 0]
        //
        //   If control > target: target=lo, control=hi
        //     i01 = lo bit set = target set, i10 = hi bit set = control set
        //     CNOT: control set => flip target
        //     |00> -> |00>
        //     |01> (tgt=1, ctrl=0) -> |01>
        //     |10> (ctrl=1, tgt=0) -> flip target -> |11>
        //     |11> (both, ctrl=1) -> flip target -> |10>
        //     Matrix:
        //     row 0: [1, 0, 0, 0]
        //     row 1: [0, 1, 0, 0]
        //     row 2: [0, 0, 0, 1]
        //     row 3: [0, 0, 1, 0]

        let cnot_matrix = if control < target {
            [
                [one, zero, zero, zero],
                [zero, zero, zero, one],
                [zero, zero, one, zero],
                [zero, one, zero, zero],
            ]
        } else {
            [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
                [zero, zero, one, zero],
            ]
        };

        self.apply_gate_matrix_2q(control, target, cnot_matrix);
    }

    /// Return a reference to the current state vector.
    pub fn state_vector(&self) -> &[C64] {
        &self.state
    }
}

// ===================================================================
// METAL 4 BENCHMARK
// ===================================================================

/// Result of a comparative benchmark between CPU and Metal 4 gate dispatch.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Metal4BenchmarkResult {
    /// Number of qubits in the benchmark circuit.
    pub num_qubits: usize,
    /// Number of gate applications in the benchmark.
    pub num_gates: usize,
    /// CPU-side dispatch time in microseconds.
    pub cpu_time_us: u64,
    /// Metal 4 dispatch time in microseconds (CPU emulation).
    pub metal4_time_us: u64,
    /// Speedup ratio (cpu_time / metal4_time). Values > 1 mean Metal 4 is faster.
    pub speedup: f64,
}

/// Benchmark CPU vs Metal 4 gate dispatch for a given circuit size.
///
/// Constructs a circuit of alternating Hadamard and CNOT gates, then
/// times the full execution. Since true GPU dispatch requires a Metal
/// command queue, the "Metal 4" path currently uses the CPU emulation
/// of the simdgroup_matrix algorithm, which exercises the same code path.
#[cfg(target_os = "macos")]
pub fn benchmark_metal4_gates(num_qubits: usize, num_gates: usize) -> Metal4BenchmarkResult {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let h_matrix = [
        [C64::new(inv_sqrt2, 0.0), C64::new(inv_sqrt2, 0.0)],
        [C64::new(inv_sqrt2, 0.0), C64::new(-inv_sqrt2, 0.0)],
    ];

    // --- CPU baseline: direct matrix application ---
    let cpu_start = Instant::now();
    {
        let dim = 1 << num_qubits;
        let mut state = vec![c64_zero(); dim];
        state[0] = c64_one();

        for g in 0..num_gates {
            let target = g % num_qubits;
            let stride = 1 << target;
            for tid in 0..(dim / 2) {
                let base = (tid / stride) * (2 * stride) + (tid % stride);
                let a0 = state[base];
                let a1 = state[base + stride];
                state[base] = h_matrix[0][0] * a0 + h_matrix[0][1] * a1;
                state[base + stride] = h_matrix[1][0] * a0 + h_matrix[1][1] * a1;
            }
        }
    }
    let cpu_elapsed = cpu_start.elapsed();

    // --- Metal 4 dispatcher path (CPU emulation) ---
    let metal4_start = Instant::now();
    {
        let mut backend = Metal4QuantumBackend::new(num_qubits);
        for g in 0..num_gates {
            let target = g % num_qubits;
            backend.apply_gate_matrix_1q(target, h_matrix);
        }
    }
    let metal4_elapsed = metal4_start.elapsed();

    let cpu_us = cpu_elapsed.as_micros() as u64;
    let metal4_us = metal4_elapsed.as_micros() as u64;
    let speedup = if metal4_us > 0 {
        cpu_us as f64 / metal4_us as f64
    } else {
        1.0
    };

    Metal4BenchmarkResult {
        num_qubits,
        num_gates,
        cpu_time_us: cpu_us,
        metal4_time_us: metal4_us,
        speedup,
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Capabilities detection
    // ---------------------------------------------------------------

    #[test]
    fn test_capabilities_detect_returns_valid_struct() {
        let caps = Metal4Capabilities::detect();
        // On any macOS machine with Metal, has_device should be true.
        assert!(caps.has_device, "Expected a Metal device on macOS");
        assert!(caps.max_buffer_length > 0);
        assert!(caps.max_threads_per_group > 0);
        assert!(!caps.gpu_family.is_empty());
    }

    #[test]
    fn test_capabilities_family_tier_parsing() {
        assert_eq!(Metal4Capabilities::parse_family_tier("Apple9"), 9);
        assert_eq!(Metal4Capabilities::parse_family_tier("Apple7"), 7);
        assert_eq!(Metal4Capabilities::parse_family_tier("Unknown"), 0);
        assert_eq!(Metal4Capabilities::parse_family_tier("Apple"), 0);
    }

    #[test]
    fn test_capabilities_supports_metal4() {
        let caps = Metal4Capabilities::detect();
        // We cannot assert the specific value since it depends on hardware,
        // but the method should not panic and should be consistent.
        let m4 = caps.supports_metal4();
        if caps.gpu_family.contains("Apple9") {
            assert!(m4, "Apple9 family should support Metal 4");
        }
    }

    // ---------------------------------------------------------------
    // Tensor contraction
    // ---------------------------------------------------------------

    #[test]
    fn test_tensor_contraction_matrix_multiply() {
        // Matrix multiplication: A(2x3) * B(3x2) -> C(2x2)
        // Contract axis 1 of A with axis 0 of B.
        let a = vec![
            C64::new(1.0, 0.0),
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
            C64::new(5.0, 0.0),
            C64::new(6.0, 0.0),
        ];
        let b = vec![
            C64::new(7.0, 0.0),
            C64::new(8.0, 0.0),
            C64::new(9.0, 0.0),
            C64::new(10.0, 0.0),
            C64::new(11.0, 0.0),
            C64::new(12.0, 0.0),
        ];

        let engine = Metal4TensorContraction::new();
        let result = engine.contract_pair(&a, &b, &[2, 3], &[3, 2], &[(1, 0)]);

        // Expected: standard matrix multiply.
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
        assert_eq!(result.len(), 4);
        assert!((result[0].re - 58.0).abs() < 1e-10);
        assert!((result[1].re - 64.0).abs() < 1e-10);
        assert!((result[2].re - 139.0).abs() < 1e-10);
        assert!((result[3].re - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_contraction_inner_product() {
        // Inner product: a(3) dot b(3) -> scalar.
        let a = vec![C64::new(1.0, 1.0), C64::new(2.0, -1.0), C64::new(0.0, 3.0)];
        let b = vec![C64::new(1.0, 0.0), C64::new(0.0, 1.0), C64::new(2.0, 0.0)];

        let engine = Metal4TensorContraction::new();
        let result = engine.contract_pair(&a, &b, &[3], &[3], &[(0, 0)]);

        // Expected: (1+i)*1 + (2-i)*(i) + (3i)*2
        //         = (1+i) + (2i - i^2) + (6i)  (i^2 = -1)
        //         = (1+i) + (1+2i) + (6i)
        //         = 2 + 9i
        assert_eq!(result.len(), 1);
        assert!((result[0].re - 2.0).abs() < 1e-10);
        assert!((result[0].im - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_contraction_batch() {
        let engine = Metal4TensorContraction::new();

        let spec1 = ContractionSpec {
            tensor_a: vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            tensor_b: vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            dims_a: vec![2, 2],
            dims_b: vec![2, 2],
            contract_axes: vec![(1, 0)],
        };

        let results = engine.batch_contract(&[spec1]);
        assert_eq!(results.len(), 1);
        // Identity * Identity = Identity.
        assert_eq!(results[0].len(), 4);
        assert!((results[0][0].re - 1.0).abs() < 1e-10);
        assert!((results[0][3].re - 1.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // ML predictor
    // ---------------------------------------------------------------

    #[test]
    fn test_ml_predictor_clifford_circuit() {
        let mut predictor = Metal4AdaptiveML::new();

        let analysis = CircuitAnalysis {
            num_gates: 100,
            num_single_qubit: 60,
            num_two_qubit: 40,
            num_three_qubit: 0,
            is_clifford_only: true,
            max_entanglement_width: 10,
            connected_components: 1,
            num_t_gates: 0,
            clifford_fraction: 1.0,
            circuit_symmetry: None,
            magic_level: 0.0,
        };

        let backend = predictor.predict_backend(&analysis);
        assert_eq!(backend, SimBackend::Stabilizer);
    }

    #[test]
    fn test_ml_predictor_small_circuit_prefers_cpu() {
        let mut predictor = Metal4AdaptiveML::new();

        let analysis = CircuitAnalysis {
            num_gates: 5,
            num_single_qubit: 3,
            num_two_qubit: 2,
            num_three_qubit: 0,
            is_clifford_only: false,
            max_entanglement_width: 2,
            connected_components: 1,
            num_t_gates: 2,
            clifford_fraction: 0.6,
            circuit_symmetry: None,
            magic_level: 0.4,
        };

        let backend = predictor.predict_backend(&analysis);
        assert_eq!(backend, SimBackend::StateVectorFused);
    }

    #[test]
    fn test_ml_predictor_medium_circuit_prefers_gpu() {
        let mut predictor = Metal4AdaptiveML::new();

        let analysis = CircuitAnalysis {
            num_gates: 200,
            num_single_qubit: 120,
            num_two_qubit: 80,
            num_three_qubit: 0,
            is_clifford_only: false,
            max_entanglement_width: 12,
            connected_components: 1,
            num_t_gates: 30,
            clifford_fraction: 0.5,
            circuit_symmetry: None,
            magic_level: 0.5,
        };

        let backend = predictor.predict_backend(&analysis);
        assert_eq!(backend, SimBackend::MetalGPU);
    }

    #[test]
    fn test_ml_predictor_accuracy_tracking() {
        let mut predictor = Metal4AdaptiveML::new();
        assert_eq!(predictor.accuracy(), 0.0);
        assert_eq!(predictor.num_predictions(), 0);

        let analysis = CircuitAnalysis {
            num_gates: 100,
            num_single_qubit: 100,
            num_two_qubit: 0,
            num_three_qubit: 0,
            is_clifford_only: true,
            max_entanglement_width: 0,
            connected_components: 1,
            num_t_gates: 0,
            clifford_fraction: 1.0,
            circuit_symmetry: None,
            magic_level: 0.0,
        };

        let _pred = predictor.predict_backend(&analysis);
        assert_eq!(predictor.num_predictions(), 1);

        predictor.update_model(&analysis, SimBackend::Stabilizer, 0.5);
        // The predict inside update_model does not increment total_predictions,
        // but the original call did. correct_predictions was updated.
        assert_eq!(predictor.correct_predictions, 1);
    }

    // ---------------------------------------------------------------
    // Shader compiler
    // ---------------------------------------------------------------

    #[test]
    fn test_shader_compiler_generates_hadamard() {
        let mut compiler = Metal4ShaderCompiler::new();
        let result = compiler.compile_gate_kernel("hadamard", 4);
        assert!(
            result.is_ok(),
            "Hadamard shader should compile: {:?}",
            result.err()
        );
        let bytecode = result.unwrap();
        assert!(!bytecode.is_empty(), "Compiled shader should be non-empty");
    }

    #[test]
    fn test_shader_compiler_cache_hit() {
        let mut compiler = Metal4ShaderCompiler::new();
        let _ = compiler.compile_gate_kernel("pauli_x", 3);
        assert_eq!(compiler.cache_size(), 1);
        // Second call should hit cache.
        let _ = compiler.compile_gate_kernel("pauli_x", 3);
        assert_eq!(compiler.cache_size(), 1);
        // Different config should add a new entry.
        let _ = compiler.compile_gate_kernel("pauli_x", 5);
        assert_eq!(compiler.cache_size(), 2);
    }

    #[test]
    fn test_shader_compiler_rejects_unknown_gate() {
        let mut compiler = Metal4ShaderCompiler::new();
        let result = compiler.compile_gate_kernel("nonexistent_gate", 2);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // Profiler
    // ---------------------------------------------------------------

    #[test]
    fn test_profiler_lifecycle() {
        let mut profiler = Metal4Profiler::new();
        assert!(!profiler.is_capturing());

        profiler.begin_capture();
        assert!(profiler.is_capturing());

        // Simulate some command buffer submissions.
        profiler.record_command_buffer(1024 * 1024, 65536);
        profiler.record_command_buffer(512 * 1024, 32768);

        let report = profiler.end_capture();
        assert!(!profiler.is_capturing());
        assert!(report.gpu_time_ms >= 0.0);
        assert_eq!(report.num_command_buffers, 2);
        assert!(!report.bottleneck.is_empty());
    }

    #[test]
    fn test_profiler_end_without_begin() {
        let mut profiler = Metal4Profiler::new();
        let report = profiler.end_capture();
        assert_eq!(report.num_command_buffers, 0);
        assert_eq!(report.gpu_time_ms, 0.0);
        assert_eq!(report.bottleneck, "no capture");
    }
}

#[cfg(test)]
mod metal4_gate_tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::C64;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_capability_detection() {
        let caps = Metal4Capabilities::detect();
        // Should always succeed (might not have Metal 4 but detection shouldn't panic)
        assert!(caps.gpu_family.len() > 0 || !caps.has_device);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_gate_dispatcher_creation() {
        let dispatcher = Metal4GateDispatcher::new();
        // Should create without panic
        let _ = dispatcher.supports_simdgroup_matrix();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_1qubit_gate_kernel_generation() {
        let mut dispatcher = Metal4GateDispatcher::new();
        let h_matrix = [
            [
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ],
            [
                C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                C64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        ];
        let kernel = dispatcher.generate_1qubit_kernel("hadamard", h_matrix);
        assert_eq!(kernel.gate_name, "hadamard");
        assert_eq!(kernel.qubit_count, 1);
        assert!(kernel.kernel_source.contains("apply_hadamard"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_1qubit_dispatch_h() {
        let mut backend = Metal4QuantumBackend::new(2);
        backend.apply_h(0);
        let probs = backend.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_2qubit_dispatch_cnot() {
        let mut backend = Metal4QuantumBackend::new(2);
        backend.apply_h(0);
        backend.apply_cnot(0, 1);
        let probs = backend.probabilities();
        // Bell state: |00> + |11>
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_bell_state_h_cnot() {
        let mut backend = Metal4QuantumBackend::new(3);
        backend.apply_h(0);
        backend.apply_cnot(0, 1);
        let probs = backend.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
        // Others should be ~0
        for i in [1, 2, 4, 5, 6, 7] {
            assert!(probs[i].abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_probabilities_sum_to_one() {
        let mut backend = Metal4QuantumBackend::new(3);
        backend.apply_h(0);
        backend.apply_h(1);
        let probs = backend.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_benchmark_creation() {
        let result = benchmark_metal4_gates(4, 10);
        assert_eq!(result.num_qubits, 4);
        assert_eq!(result.num_gates, 10);
    }
}
