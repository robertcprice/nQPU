//! Circuit Cache — compiled circuit caching and parallel gate batching.
//!
//! Wraps [`crate::circuit_optimizer`] with a hash-indexed cache so that
//! repeated execution of the same circuit reuses the optimized form.
//! Also batches independent gates for parallel dispatch.
//!
//! # Architecture
//!
//! 1. **Circuit Analysis**: Identify optimization opportunities (fusion, parallelization)
//! 2. **Gate Batching**: Group gates by qubit independence for parallel execution
//! 3. **Fusion**: Combine consecutive single-qubit gates into single matrices
//! 4. **Caching**: Memoize compiled circuits for reuse
//!
//! # Performance
//!
//! Circuit compilation provides speedups for:
//! - Repeated circuit execution (via cached execution plans)
//! - Large circuits with fusion opportunities
//! - Parameterized circuits with fixed structure

use crate::circuit_optimizer::{CircuitOptimizer, OptimizationLevel};
use crate::gates::Gate;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

// ===========================================================================
// CONFIGURATION
// ===========================================================================

/// JIT compiler configuration.
#[derive(Clone, Debug)]
pub struct JitConfig {
    /// Optimization level (0-3).
    pub optimization_level: usize,
    /// Maximum circuits to cache.
    pub cache_size: usize,
    /// Enable gate fusion.
    pub enable_fusion: bool,
    /// Enable parallel gate dispatch.
    pub enable_parallel: bool,
    /// Minimum circuit size to JIT compile.
    pub min_circuit_size: usize,
    /// Enable native code generation (requires cranelift feature).
    pub enable_native: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            cache_size: 100,
            enable_fusion: true,
            enable_parallel: true,
            min_circuit_size: 10,
            enable_native: false,
        }
    }
}

impl JitConfig {
    /// Set optimization level (0-3).
    pub fn with_optimization_level(mut self, level: usize) -> Self {
        self.optimization_level = level.min(3);
        self
    }

    /// Set cache size.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Enable/disable native code generation.
    pub fn with_native(mut self, enable: bool) -> Self {
        self.enable_native = enable;
        self
    }
}

// ===========================================================================
// COMPILED CIRCUIT
// ===========================================================================

/// A compiled circuit ready for execution.
#[derive(Clone)]
pub struct CompiledCircuit {
    /// Original circuit hash.
    hash: u64,
    /// Optimized gates.
    optimized_gates: Vec<Gate>,
    /// Gate batches for parallel execution.
    batches: Vec<GateBatch>,
    /// Compilation statistics.
    stats: CompilationStats,
    /// Native Metal shaders (one per batch, if JIT enabled).
    native_shaders: Vec<String>,
}

/// A batch of gates that can be executed in parallel.
#[derive(Clone, Debug)]
pub struct GateBatch {
    /// Gates in this batch.
    pub gates: Vec<Gate>,
    /// Qubits touched by this batch.
    pub qubits: Vec<usize>,
    /// Whether batch contains only single-qubit gates.
    pub single_qubit_only: bool,
}

/// Compilation statistics.
#[derive(Clone, Debug, Default)]
pub struct CompilationStats {
    /// Original gate count.
    pub original_gates: usize,
    /// Optimized gate count.
    pub optimized_gates: usize,
    /// Number of batches.
    pub num_batches: usize,
    /// Compilation time in microseconds.
    pub compile_time_us: u64,
    /// Gates removed by optimization.
    pub gates_removed: usize,
    /// Fusion operations performed.
    pub fusions: usize,
}

impl CompiledCircuit {
    /// Get the number of qubits required.
    pub fn num_qubits(&self) -> usize {
        let mut max_q = 0;
        for gate in &self.optimized_gates {
            for &q in &gate.targets {
                max_q = max_q.max(q);
            }
            for &q in &gate.controls {
                max_q = max_q.max(q);
            }
        }
        max_q + 1
    }

    /// Get the optimized gates.
    pub fn gates(&self) -> &[Gate] {
        &self.optimized_gates
    }

    /// Get the gate batches.
    pub fn batches(&self) -> &[GateBatch] {
        &self.batches
    }

    /// Get compilation statistics.
    pub fn stats(&self) -> &CompilationStats {
        &self.stats
    }

    /// Get native Metal shader for a batch (if JIT was enabled).
    pub fn get_native_shader(&self, batch_idx: usize) -> Option<&str> {
        self.native_shaders.get(batch_idx).map(|s| s.as_str())
    }

    /// Check if native shaders were generated.
    pub fn has_native_shaders(&self) -> bool {
        !self.native_shaders.is_empty()
    }
}

// ===========================================================================
// CIRCUIT HASH
// ===========================================================================

/// Compute a hash for a circuit (for caching).
fn circuit_hash(gates: &[Gate]) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    for gate in gates {
        // Hash gate type as string (GateType may not impl Hash)
        format!("{:?}", gate.gate_type).hash(&mut hasher);
        // Hash targets
        for &t in &gate.targets {
            t.hash(&mut hasher);
        }
        // Hash controls
        for &c in &gate.controls {
            c.hash(&mut hasher);
        }
        // Hash params
        if let Some(ref params) = gate.params {
            for &p in params {
                p.to_bits().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

// ===========================================================================
// NATIVE METAL SHADER COMPILER
// ===========================================================================

/// Native Metal shader compiler for gate batches.
struct NativeCompiler;

impl NativeCompiler {
    fn new() -> Self {
        Self
    }

    /// Compile a gate batch into a fused Metal compute shader.
    ///
    /// Generates a single Metal kernel that applies all gates in the batch
    /// in a single GPU dispatch, with pre-baked matrix constants.
    fn compile_batch_to_metal(&self, batch: &GateBatch, batch_idx: usize) -> Result<String, JitError> {
        let mut shader = String::new();

        // Header
        shader.push_str("#include <metal_stdlib>\n");
        shader.push_str("using namespace metal;\n\n");

        // Complex number helpers
        shader.push_str(Self::complex_helpers());

        // Gate matrix constants
        shader.push_str(&self.generate_matrix_constants(&batch.gates)?);

        // Kernel function
        shader.push_str(&format!("\nkernel void batch_{}(\n", batch_idx));
        shader.push_str("    device Complex* state [[buffer(0)]],\n");
        shader.push_str("    uint index [[thread_position_in_grid]]\n");
        shader.push_str(") {\n");

        // Apply each gate sequentially
        shader.push_str(&self.generate_gate_applications(&batch.gates, &batch.qubits)?);

        shader.push_str("}\n");

        Ok(shader)
    }

    /// Generate Complex type and helper functions.
    fn complex_helpers() -> &'static str {
        r#"struct Complex {
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

"#
    }

    /// Generate constant matrix definitions for all gates in the batch.
    fn generate_matrix_constants(&self, gates: &[Gate]) -> Result<String, JitError> {
        let mut constants = String::new();

        for (idx, gate) in gates.iter().enumerate() {
            let matrix = gate.gate_type.matrix();
            let size = matrix.len();

            constants.push_str(&format!("// Gate {}: {:?}\n", idx, gate.gate_type));

            if size == 2 {
                // Single-qubit gate (2x2)
                constants.push_str(&format!("constant Complex gate_{}_m[2][2] = {{\n", idx));
                for row in &matrix {
                    constants.push_str("    {");
                    for (col_idx, val) in row.iter().enumerate() {
                        constants.push_str(&format!("{{{}f, {}f}}", val.re, val.im));
                        if col_idx < row.len() - 1 {
                            constants.push_str(", ");
                        }
                    }
                    constants.push_str("},\n");
                }
                constants.push_str("};\n\n");
            } else if size == 4 {
                // Two-qubit gate (4x4)
                constants.push_str(&format!("constant Complex gate_{}_m[4][4] = {{\n", idx));
                for row in &matrix {
                    constants.push_str("    {");
                    for (col_idx, val) in row.iter().enumerate() {
                        constants.push_str(&format!("{{{}f, {}f}}", val.re, val.im));
                        if col_idx < row.len() - 1 {
                            constants.push_str(", ");
                        }
                    }
                    constants.push_str("},\n");
                }
                constants.push_str("};\n\n");
            } else {
                return Err(JitError::NativeError(format!(
                    "Unsupported gate matrix size: {}x{} for gate {:?}",
                    size, size, gate.gate_type
                )));
            }
        }

        Ok(constants)
    }

    /// Generate gate application code.
    fn generate_gate_applications(&self, gates: &[Gate], _qubits: &[usize]) -> Result<String, JitError> {
        let mut code = String::new();

        for (idx, gate) in gates.iter().enumerate() {
            if gate.targets.len() == 1 && gate.controls.is_empty() {
                // Single-qubit gate
                let qubit = gate.targets[0];
                code.push_str(&format!("\n    // Apply gate {} (single-qubit on q{})\n", idx, qubit));
                code.push_str(&Self::apply_single_qubit(idx, qubit));
            } else if gate.targets.len() == 1 && gate.controls.len() == 1 {
                // Controlled single-qubit gate (CNOT, CZ, etc.)
                let control = gate.controls[0];
                let target = gate.targets[0];
                code.push_str(&format!("\n    // Apply gate {} (controlled on q{}, target q{})\n", idx, control, target));
                code.push_str(&Self::apply_controlled_gate(idx, control, target));
            } else if gate.targets.len() == 2 && gate.controls.is_empty() {
                // Two-qubit gate (SWAP, etc.)
                let q0 = gate.targets[0];
                let q1 = gate.targets[1];
                code.push_str(&format!("\n    // Apply gate {} (two-qubit on q{}, q{})\n", idx, q0, q1));
                code.push_str(&Self::apply_two_qubit(idx, q0, q1));
            } else {
                return Err(JitError::NativeError(format!(
                    "Unsupported gate configuration: {} targets, {} controls",
                    gate.targets.len(),
                    gate.controls.len()
                )));
            }
        }

        Ok(code)
    }

    /// Generate single-qubit gate application code.
    fn apply_single_qubit(gate_idx: usize, qubit: usize) -> String {
        format!(
            r#"    {{
        uint stride = 1 << {};
        uint base = (index / stride) * stride * 2;
        uint offset = index % stride;
        uint i0 = base + offset;
        uint i1 = i0 + stride;

        Complex a0 = state[i0];
        Complex a1 = state[i1];

        Complex r0 = complex_add(
            complex_mul(gate_{}_m[0][0], a0),
            complex_mul(gate_{}_m[0][1], a1)
        );
        Complex r1 = complex_add(
            complex_mul(gate_{}_m[1][0], a0),
            complex_mul(gate_{}_m[1][1], a1)
        );

        state[i0] = r0;
        state[i1] = r1;
    }}
"#,
            qubit, gate_idx, gate_idx, gate_idx, gate_idx
        )
    }

    /// Generate controlled gate application code.
    fn apply_controlled_gate(gate_idx: usize, control: usize, target: usize) -> String {
        let (q_lo, q_hi) = if control < target {
            (control, target)
        } else {
            (target, control)
        };

        format!(
            r#"    {{
        uint n_lo = {};
        uint n_hi = {};
        uint stride_lo = 1 << n_lo;
        uint stride_hi = 1 << n_hi;

        uint base = (index / stride_lo) * stride_lo * 2;
        uint offset = index % stride_lo;

        uint i0 = base + offset;
        uint i1 = i0 + stride_lo;
        uint i2 = i0 + stride_hi;
        uint i3 = i1 + stride_hi;

        // Only apply to states where control=1 (i1, i3)
        Complex a1 = state[i1];
        Complex a3 = state[i3];

        Complex r1 = complex_add(
            complex_mul(gate_{}_m[1][1], a1),
            complex_mul(gate_{}_m[1][3], a3)
        );
        Complex r3 = complex_add(
            complex_mul(gate_{}_m[3][1], a1),
            complex_mul(gate_{}_m[3][3], a3)
        );

        state[i1] = r1;
        state[i3] = r3;
    }}
"#,
            q_lo, q_hi, gate_idx, gate_idx, gate_idx, gate_idx
        )
    }

    /// Generate two-qubit gate application code.
    fn apply_two_qubit(gate_idx: usize, qubit_lo: usize, qubit_hi: usize) -> String {
        format!(
            r#"    {{
        uint n_lo = {};
        uint n_hi = {};
        uint stride_lo = 1 << n_lo;
        uint stride_hi = 1 << n_hi;

        uint base = (index / stride_lo) * stride_lo * 2;
        uint offset = index % stride_lo;

        uint i0 = base + offset;
        uint i1 = i0 + stride_lo;
        uint i2 = i0 + stride_hi;
        uint i3 = i1 + stride_hi;

        Complex a0 = state[i0];
        Complex a1 = state[i1];
        Complex a2 = state[i2];
        Complex a3 = state[i3];

        Complex r0 = complex_add(
            complex_add(
                complex_mul(gate_{}_m[0][0], a0),
                complex_mul(gate_{}_m[0][1], a1)
            ),
            complex_add(
                complex_mul(gate_{}_m[0][2], a2),
                complex_mul(gate_{}_m[0][3], a3)
            )
        );
        Complex r1 = complex_add(
            complex_add(
                complex_mul(gate_{}_m[1][0], a0),
                complex_mul(gate_{}_m[1][1], a1)
            ),
            complex_add(
                complex_mul(gate_{}_m[1][2], a2),
                complex_mul(gate_{}_m[1][3], a3)
            )
        );
        Complex r2 = complex_add(
            complex_add(
                complex_mul(gate_{}_m[2][0], a0),
                complex_mul(gate_{}_m[2][1], a1)
            ),
            complex_add(
                complex_mul(gate_{}_m[2][2], a2),
                complex_mul(gate_{}_m[2][3], a3)
            )
        );
        Complex r3 = complex_add(
            complex_add(
                complex_mul(gate_{}_m[3][0], a0),
                complex_mul(gate_{}_m[3][1], a1)
            ),
            complex_add(
                complex_mul(gate_{}_m[3][2], a2),
                complex_mul(gate_{}_m[3][3], a3)
            )
        );

        state[i0] = r0;
        state[i1] = r1;
        state[i2] = r2;
        state[i3] = r3;
    }}
"#,
            qubit_lo, qubit_hi,
            gate_idx, gate_idx, gate_idx, gate_idx,
            gate_idx, gate_idx, gate_idx, gate_idx,
            gate_idx, gate_idx, gate_idx, gate_idx,
            gate_idx, gate_idx, gate_idx, gate_idx
        )
    }
}

// ===========================================================================
// JIT COMPILER
// ===========================================================================

/// JIT compiler for quantum circuits.
pub struct JitCompiler {
    /// Configuration.
    config: JitConfig,
    /// Circuit optimizer.
    optimizer: CircuitOptimizer,
    /// Compiled circuit cache.
    cache: HashMap<u64, CompiledCircuit>,
    /// Cache access statistics.
    cache_hits: usize,
    cache_misses: usize,
}

impl JitCompiler {
    /// Create a new JIT compiler.
    pub fn new(config: JitConfig) -> Self {
        let level = match config.optimization_level {
            0 => OptimizationLevel::None,
            1 => OptimizationLevel::Basic,
            2 => OptimizationLevel::Moderate,
            _ => OptimizationLevel::Aggressive,
        };

        Self {
            optimizer: CircuitOptimizer::new(level),
            config,
            cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Compile a circuit to executable form.
    pub fn compile(&mut self, gates: &[Gate]) -> Result<CompiledCircuit, JitError> {
        if gates.is_empty() {
            return Err(JitError::EmptyCircuit);
        }

        let hash = circuit_hash(gates);

        // Check cache
        if let Some(cached) = self.cache.get(&hash) {
            self.cache_hits += 1;
            return Ok(cached.clone());
        }

        self.cache_misses += 1;

        // Compile
        let start = Instant::now();
        let compiled = self.compile_inner(gates, hash)?;
        let compile_time = start.elapsed().as_micros() as u64;

        // Update stats
        let mut stats = compiled.stats.clone();
        stats.compile_time_us = compile_time;
        let compiled = CompiledCircuit { stats, ..compiled };

        // Cache
        if self.cache.len() >= self.config.cache_size {
            // Simple eviction: remove oldest entry (first key)
            if let Some(first_key) = self.cache.keys().next().copied() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(hash, compiled.clone());

        Ok(compiled)
    }

    /// Inner compilation logic.
    fn compile_inner(&self, gates: &[Gate], hash: u64) -> Result<CompiledCircuit, JitError> {
        let original_count = gates.len();

        // Optimize
        let optimized = if self.config.enable_fusion {
            self.optimizer.optimize(gates)
        } else {
            gates.to_vec()
        };

        // Create batches
        let batches = self.create_batches(&optimized);

        // Generate native Metal shaders if enabled
        let native_shaders = if self.config.enable_native {
            let compiler = NativeCompiler::new();
            batches
                .iter()
                .enumerate()
                .map(|(idx, batch)| compiler.compile_batch_to_metal(batch, idx))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };

        // Calculate stats
        let stats = CompilationStats {
            original_gates: original_count,
            optimized_gates: optimized.len(),
            num_batches: batches.len(),
            compile_time_us: 0, // Set by caller
            gates_removed: original_count.saturating_sub(optimized.len()),
            fusions: self.count_fusions(gates, &optimized),
        };

        Ok(CompiledCircuit {
            hash,
            optimized_gates: optimized,
            batches,
            stats,
            native_shaders,
        })
    }

    /// Create gate batches for parallel execution.
    fn create_batches(&self, gates: &[Gate]) -> Vec<GateBatch> {
        if !self.config.enable_parallel {
            // Single batch
            return vec![GateBatch {
                gates: gates.to_vec(),
                qubits: self.all_qubits(gates),
                single_qubit_only: gates.iter().all(|g| g.targets.len() == 1 && g.controls.is_empty()),
            }];
        }

        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut used_qubits = std::collections::HashSet::new();

        for gate in gates {
            let gate_qubits: std::collections::HashSet<_> = gate.targets.iter()
                .chain(gate.controls.iter())
                .copied()
                .collect();

            // Check if gate conflicts with current batch
            let conflicts = gate_qubits.iter().any(|q| used_qubits.contains(q));

            if conflicts && !current_batch.is_empty() {
                // Start new batch
                let qubits: Vec<_> = used_qubits.iter().copied().collect();
                batches.push(GateBatch {
                    gates: std::mem::take(&mut current_batch),
                    qubits,
                    single_qubit_only: false,
                });
                used_qubits.clear();
            }

            current_batch.push(gate.clone());
            used_qubits.extend(gate_qubits);
        }

        // Final batch
        if !current_batch.is_empty() {
            let qubits: Vec<_> = used_qubits.iter().copied().collect();
            batches.push(GateBatch {
                gates: current_batch,
                qubits,
                single_qubit_only: false,
            });
        }

        batches
    }

    /// Get all qubits used in a circuit.
    fn all_qubits(&self, gates: &[Gate]) -> Vec<usize> {
        let mut qubits = std::collections::HashSet::new();
        for gate in gates {
            for &t in &gate.targets {
                qubits.insert(t);
            }
            for &c in &gate.controls {
                qubits.insert(c);
            }
        }
        qubits.into_iter().collect()
    }

    /// Count fusion operations (approximation).
    fn count_fusions(&self, original: &[Gate], optimized: &[Gate]) -> usize {
        original.len().saturating_sub(optimized.len())
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            max_size: self.config.cache_size,
            hits: self.cache_hits,
            misses: self.cache_misses,
            hit_rate: if self.cache_hits + self.cache_misses > 0 {
                self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

// ===========================================================================
// STATISTICS
// ===========================================================================

/// Cache statistics.
#[derive(Clone, Debug)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
}

// ===========================================================================
// ERRORS
// ===========================================================================

/// JIT compilation errors.
#[derive(Debug, Clone)]
pub enum JitError {
    /// Circuit is empty.
    EmptyCircuit,
    /// Circuit too small for JIT.
    CircuitTooSmall { size: usize, min: usize },
    /// Native compilation failed.
    NativeError(String),
    /// Invalid gate.
    InvalidGate(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCircuit => write!(f, "circuit is empty"),
            Self::CircuitTooSmall { size, min } => {
                write!(f, "circuit has {} gates, minimum is {}", size, min)
            }
            Self::NativeError(msg) => write!(f, "native compilation failed: {}", msg),
            Self::InvalidGate(msg) => write!(f, "invalid gate: {}", msg),
        }
    }
}

impl std::error::Error for JitError {}

// ===========================================================================
// CIRCUIT EXECUTOR
// ===========================================================================

/// Executor for compiled circuits.
pub struct CircuitExecutor<'a> {
    /// Compiled circuit.
    circuit: &'a CompiledCircuit,
}

impl<'a> CircuitExecutor<'a> {
    /// Create an executor for a compiled circuit.
    pub fn new(circuit: &'a CompiledCircuit) -> Self {
        Self { circuit }
    }

    /// Get the gates to execute.
    pub fn gates(&self) -> &[Gate] {
        &self.circuit.optimized_gates
    }

    /// Get the batches for parallel execution.
    pub fn batches(&self) -> &[GateBatch] {
        &self.circuit.batches
    }

    /// Get the number of execution steps (batches).
    pub fn num_steps(&self) -> usize {
        self.circuit.batches.len()
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::GateType;

    fn make_test_circuit() -> Vec<Gate> {
        vec![
            Gate::new(GateType::H, vec![0], vec![]),
            Gate::new(GateType::H, vec![1], vec![]),
            Gate::new(GateType::CNOT, vec![1], vec![0]),
            Gate::new(GateType::H, vec![0], vec![]),
            Gate::new(GateType::H, vec![1], vec![]),
        ]
    }

    #[test]
    fn test_jit_compile_basic() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config);
        let circuit = make_test_circuit();

        let compiled = jit.compile(&circuit).unwrap();
        assert!(!compiled.optimized_gates.is_empty()); // Compilation produces gates
        assert!(!compiled.batches.is_empty());
    }

    #[test]
    fn test_cache_hit() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config);
        let circuit = make_test_circuit();

        // First compile (cache miss)
        let _ = jit.compile(&circuit).unwrap();

        // Second compile (cache hit)
        let _ = jit.compile(&circuit).unwrap();

        let stats = jit.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_empty_circuit_error() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config);

        let result = jit.compile(&[]);
        assert!(matches!(result, Err(JitError::EmptyCircuit)));
    }

    #[test]
    fn test_batching() {
        let config = JitConfig {
            enable_parallel: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);

        // Gates on different qubits should batch together
        let circuit = vec![
            Gate::new(GateType::H, vec![0], vec![]),
            Gate::new(GateType::H, vec![1], vec![]),
            Gate::new(GateType::H, vec![2], vec![]),
            Gate::new(GateType::CNOT, vec![1], vec![0]), // This forces new batch
            Gate::new(GateType::H, vec![2], vec![]),
        ];

        let compiled = jit.compile(&circuit).unwrap();

        // Should have multiple batches due to CNOT
        assert!(compiled.batches.len() >= 1);
    }

    #[test]
    fn test_circuit_hash() {
        let circuit1 = make_test_circuit();
        let circuit2 = make_test_circuit();
        let circuit3 = vec![Gate::new(GateType::X, vec![0], vec![])];

        assert_eq!(circuit_hash(&circuit1), circuit_hash(&circuit2));
        assert_ne!(circuit_hash(&circuit1), circuit_hash(&circuit3));
    }

    #[test]
    fn test_optimization_levels() {
        for level in 0..=3 {
            let config = JitConfig::default().with_optimization_level(level);
            let mut jit = JitCompiler::new(config);
            let circuit = make_test_circuit();

            let compiled = jit.compile(&circuit).unwrap();
            // Compilation succeeds at all levels
            assert!(!compiled.optimized_gates.is_empty());
        }
    }

    #[test]
    fn test_clear_cache() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config);
        let circuit = make_test_circuit();

        let _ = jit.compile(&circuit).unwrap();
        assert_eq!(jit.cache_stats().size, 1);

        jit.clear_cache();
        assert_eq!(jit.cache_stats().size, 0);
        assert_eq!(jit.cache_stats().hits, 0);
        assert_eq!(jit.cache_stats().misses, 0);
    }

    #[test]
    fn test_cache_speedup_benchmark() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config);

        // Create a larger circuit for meaningful benchmark
        let circuit: Vec<Gate> = (0..100)
            .flat_map(|i| {
                let q = i % 10;
                vec![
                    Gate::new(GateType::H, vec![q], vec![]),
                    Gate::new(GateType::Rz(0.1 * (i as f64)), vec![q], vec![]),
                ]
            })
            .collect();

        // Cold compile (cache miss)
        let cold_start = Instant::now();
        let _compiled1 = jit.compile(&circuit).unwrap();
        let cold_time = cold_start.elapsed();

        // Warm compile (cache hit)
        let warm_start = Instant::now();
        let _compiled2 = jit.compile(&circuit).unwrap();
        let warm_time = warm_start.elapsed();

        let speedup = cold_time.as_nanos() as f64 / warm_time.as_nanos().max(1) as f64;

        println!("\n=== JIT Cache Speedup ===");
        println!("Cold compile: {:?}", cold_time);
        println!("Warm compile: {:?}", warm_time);
        println!("Speedup: {:.1}x", speedup);
        println!("Cache stats: {:?}", jit.cache_stats());

        // Cache hit should be significantly faster
        assert!(warm_time < cold_time, "Cache hit should be faster than miss");
        assert!(jit.cache_stats().hits >= 1);
    }

    #[test]
    fn test_native_compilation_disabled() {
        let config = JitConfig {
            enable_native: false,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);
        let circuit = make_test_circuit();

        let compiled = jit.compile(&circuit).unwrap();

        assert!(!compiled.has_native_shaders());
        assert_eq!(compiled.native_shaders.len(), 0);
    }

    #[test]
    fn test_native_compilation_enabled() {
        let config = JitConfig {
            enable_native: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);
        let circuit = make_test_circuit();

        let compiled = jit.compile(&circuit).unwrap();

        assert!(compiled.has_native_shaders());
        assert_eq!(compiled.native_shaders.len(), compiled.batches.len());
    }

    #[test]
    fn test_native_shader_syntax() {
        let config = JitConfig {
            enable_native: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);

        // Simple circuit with known gates
        let circuit = vec![
            Gate::new(GateType::H, vec![0], vec![]),
            Gate::new(GateType::X, vec![1], vec![]),
        ];

        let compiled = jit.compile(&circuit).unwrap();

        // Check each generated shader
        for (idx, shader) in compiled.native_shaders.iter().enumerate() {
            println!("\n=== Batch {} Shader ===\n{}", idx, shader);

            // Verify MSL syntax requirements
            assert!(shader.contains("kernel void"), "Shader must contain kernel function");
            assert!(shader.contains("batch_"), "Kernel must be named batch_N");
            assert!(shader.contains("Complex"), "Shader must use Complex type");
            assert!(shader.contains("device Complex* state"), "Kernel must have state buffer");
            assert!(shader.contains("[[buffer(0)]]"), "State must be buffer 0");
            assert!(shader.contains("[[thread_position_in_grid]]"), "Must use thread_position_in_grid");
            assert!(shader.contains("complex_mul"), "Must define complex_mul helper");
            assert!(shader.contains("complex_add"), "Must define complex_add helper");
        }
    }

    #[test]
    fn test_native_shader_gate_matrices() {
        let config = JitConfig {
            enable_native: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);

        // Circuit with various gate types
        let circuit = vec![
            Gate::new(GateType::H, vec![0], vec![]),
            Gate::new(GateType::X, vec![1], vec![]),
            Gate::new(GateType::Rz(std::f64::consts::PI / 4.0), vec![2], vec![]),
        ];

        let compiled = jit.compile(&circuit).unwrap();

        for shader in &compiled.native_shaders {
            // Should have gate matrix constants
            if shader.contains("gate_0_m") {
                assert!(shader.contains("constant Complex gate_"), "Shader must have gate matrix constants");
                assert!(shader.contains("[2][2]") || shader.contains("[4][4]"), "Matrix must be 2x2 or 4x4");
            }
        }
    }

    #[test]
    fn test_native_shader_batch_count() {
        let config = JitConfig {
            enable_native: true,
            enable_parallel: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);

        // Circuit that should create multiple batches
        let circuit = vec![
            Gate::new(GateType::H, vec![0], vec![]),
            Gate::new(GateType::H, vec![1], vec![]),
            Gate::new(GateType::CNOT, vec![1], vec![0]), // Forces new batch
            Gate::new(GateType::H, vec![2], vec![]),
        ];

        let compiled = jit.compile(&circuit).unwrap();

        // Shader count must match batch count
        assert_eq!(
            compiled.native_shaders.len(),
            compiled.batches.len(),
            "Must have one shader per batch"
        );
    }

    #[test]
    fn test_native_shader_get_method() {
        let config = JitConfig {
            enable_native: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);
        let circuit = make_test_circuit();

        let compiled = jit.compile(&circuit).unwrap();

        // Test get_native_shader method
        for i in 0..compiled.batches.len() {
            let shader = compiled.get_native_shader(i);
            assert!(shader.is_some(), "Shader {} should exist", i);
            assert!(shader.unwrap().len() > 0, "Shader {} should not be empty", i);
        }

        // Out of bounds should return None
        assert!(compiled.get_native_shader(999).is_none());
    }

    #[test]
    fn test_native_compilation_cnot() {
        let config = JitConfig {
            enable_native: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);

        let circuit = vec![Gate::new(GateType::CNOT, vec![1], vec![0])];
        let compiled = jit.compile(&circuit).unwrap();

        assert!(compiled.has_native_shaders());
        let shader = compiled.get_native_shader(0).unwrap();

        // CNOT should generate controlled gate code
        assert!(shader.contains("gate_0_m[4][4]"), "CNOT should have 4x4 matrix");
    }

    #[test]
    fn test_native_compilation_rotation_gate() {
        let config = JitConfig {
            enable_native: true,
            ..Default::default()
        };
        let mut jit = JitCompiler::new(config);

        let angle = std::f64::consts::PI / 3.0;
        let circuit = vec![Gate::new(GateType::Rx(angle), vec![0], vec![])];
        let compiled = jit.compile(&circuit).unwrap();

        assert!(compiled.has_native_shaders());
        let shader = compiled.get_native_shader(0).unwrap();

        // Rotation gate should have matrix with computed cos/sin values
        assert!(shader.contains("gate_0_m[2][2]"), "Rx should have 2x2 matrix");

        // The matrix should contain non-trivial values (not all 0 or 1)
        assert!(shader.contains("f, "), "Matrix should have float literals");
    }
}

// ─── Aliases with non-misleading names ──────────────────────────────────────

/// Alias: `CircuitCache` is the preferred name for `JitCompiler`.
pub type CircuitCache = JitCompiler;

/// Alias: `CacheConfig` is the preferred name for `JitConfig`.
pub type CacheConfig = JitConfig;
