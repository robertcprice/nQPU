//! On-Demand Precision Switching for Apple Silicon UMA (B3)
//!
//! Mixed-precision quantum simulation that exploits Apple Silicon's unified
//! memory to switch between f32 (Metal GPU, fast) and f64 (CPU, precise)
//! with **zero transfer cost**.  On discrete-GPU platforms (CUDA/ROCm),
//! precision switching requires a PCIe copy of the entire state vector;
//! on UMA, both f32 and f64 representations share the same physical memory
//! and promotion/demotion is a local O(N) pointer-chase with no DMA.
//!
//! # Precision Policies
//!
//! | Policy      | f32 Usage           | f64 Usage                  |
//! |-------------|---------------------|----------------------------|
//! | AlwaysF64   | Never               | All gates                  |
//! | AlwaysF32   | All gates           | Never                      |
//! | Adaptive    | Forward pass gates  | Gradient / measurement     |
//! | Custom      | User-specified      | User-specified             |
//!
//! # Error Tracking
//!
//! The simulator tracks accumulated precision error by periodically
//! computing the norm difference between f32 and f64 representations.
//! When the error exceeds a configurable threshold, f64 is forced
//! for subsequent gates until a checkpoint restores accuracy.
//!
//! # Backend Selection
//!
//! On macOS, the simulator can execute f32 gates through the Metal backend.
//! When Metal is unavailable (or on non-macOS platforms), f32 execution
//! falls back to CPU-side emulation.

use crate::ascii_viz::apply_gate_to_state;
use crate::gates::{Gate, GateType};
use crate::{c32_one, c32_zero, C32, C64, QuantumState};
#[cfg(target_os = "macos")]
use crate::metal_backend::MetalSimulator;
use std::fmt;

// ============================================================
// PRECISION POLICY
// ============================================================

/// Policy governing when to use f32 vs f64 precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionPolicy {
    /// Always use f64 (maximum precision, CPU only).
    AlwaysF64,
    /// Always use f32 (maximum speed, GPU-friendly).
    AlwaysF32,
    /// Adaptive: use f32 for forward passes, f64 for gradients and measurement.
    /// Switches automatically based on gate context.
    Adaptive,
    /// Custom: use the per-gate precision selector callback.
    Custom,
}

impl fmt::Display for PrecisionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlwaysF64 => write!(f, "AlwaysF64"),
            Self::AlwaysF32 => write!(f, "AlwaysF32"),
            Self::Adaptive => write!(f, "Adaptive"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// ============================================================
// GATE PRECISION CLASSIFICATION
// ============================================================

/// The precision used for a specific gate execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatePrecision {
    /// Execute in f32 (fast, GPU-friendly).
    F32,
    /// Execute in f64 (precise, CPU).
    F64,
}

/// Backend used for f32 gate execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32ExecutionBackend {
    /// CPU fallback path using f32 round-trip emulation.
    CpuFallback,
    /// Real Metal backend execution path (macOS only).
    MetalGpu,
}

impl fmt::Display for F32ExecutionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuFallback => write!(f, "CpuFallback"),
            Self::MetalGpu => write!(f, "MetalGpu"),
        }
    }
}

impl fmt::Display for GatePrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
        }
    }
}

/// Context for a gate that influences precision selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateContext {
    /// Normal forward-pass gate application.
    ForwardPass,
    /// Gate applied during gradient computation (needs precision).
    GradientComputation,
    /// Gate immediately before measurement (needs precision).
    PreMeasurement,
    /// Parametric gate where precision affects optimization convergence.
    ParametricSensitive,
}

/// Classify a gate for adaptive precision selection.
///
/// Returns the recommended precision and a reasoning string.
pub fn classify_gate_precision(
    gate: &Gate,
    context: GateContext,
    accumulated_error: f64,
    error_threshold: f64,
) -> (GatePrecision, String) {
    // If accumulated error exceeds threshold, force f64.
    if accumulated_error > error_threshold {
        return (
            GatePrecision::F64,
            format!(
                "accumulated error ({:.2e}) exceeds threshold ({:.2e}), forcing f64",
                accumulated_error, error_threshold
            ),
        );
    }

    match context {
        GateContext::GradientComputation => (
            GatePrecision::F64,
            "gradient computation requires f64 precision".to_string(),
        ),
        GateContext::PreMeasurement => (
            GatePrecision::F64,
            "pre-measurement gate requires f64 for accurate probabilities".to_string(),
        ),
        GateContext::ParametricSensitive => {
            // Parametric gates with small angles are precision-sensitive.
            let small_angle = match &gate.gate_type {
                GateType::Rx(a) | GateType::Ry(a) | GateType::Rz(a) => a.abs() < 0.01,
                GateType::Phase(a) => a.abs() < 0.01,
                GateType::CRx(a) | GateType::CRy(a) | GateType::CRz(a) | GateType::CR(a) => {
                    a.abs() < 0.01
                }
                _ => false,
            };
            if small_angle {
                (
                    GatePrecision::F64,
                    "small-angle parametric gate: f64 for precision".to_string(),
                )
            } else {
                (
                    GatePrecision::F32,
                    "parametric gate with sufficient angle: f32 ok".to_string(),
                )
            }
        }
        GateContext::ForwardPass => {
            // For forward pass, most gates are fine in f32.
            // Exception: multi-qubit entangling gates accumulate more error.
            let is_multi_qubit =
                gate.targets.len() + gate.controls.len() > 2;
            if is_multi_qubit {
                (
                    GatePrecision::F64,
                    "multi-qubit entangling gate: f64 to limit error accumulation".to_string(),
                )
            } else {
                (
                    GatePrecision::F32,
                    "forward pass, standard gate: f32 for speed".to_string(),
                )
            }
        }
    }
}

// ============================================================
// PRECISION STATISTICS
// ============================================================

/// Statistics from a mixed-precision circuit execution.
#[derive(Clone, Debug)]
pub struct PrecisionStats {
    /// Total gates executed.
    pub total_gates: usize,
    /// Gates executed in f32.
    pub f32_gates: usize,
    /// Gates executed in f64.
    pub f64_gates: usize,
    /// Number of f32->f64 promotions.
    pub promotions: usize,
    /// Number of f64->f32 demotions.
    pub demotions: usize,
    /// Total time spent in f32->f64 promotions.
    pub promotion_time_ns: f64,
    /// Total time spent in f64->f32 demotions.
    pub demotion_time_ns: f64,
    /// Total time spent executing gates in f32 path.
    pub f32_gate_time_ns: f64,
    /// Total time spent executing gates in f64 path.
    pub f64_gate_time_ns: f64,
    /// End-to-end mixed precision execution time.
    pub total_exec_time_ns: f64,
    /// Final accumulated precision error estimate.
    pub accumulated_error: f64,
    /// Peak accumulated error during execution.
    pub peak_error: f64,
    /// Whether the error threshold was ever exceeded.
    pub threshold_exceeded: bool,
    /// Per-gate precision decisions (gate_index, precision, reasoning).
    pub decisions: Vec<(usize, GatePrecision, String)>,
}

impl Default for PrecisionStats {
    fn default() -> Self {
        Self {
            total_gates: 0,
            f32_gates: 0,
            f64_gates: 0,
            promotions: 0,
            demotions: 0,
            promotion_time_ns: 0.0,
            demotion_time_ns: 0.0,
            f32_gate_time_ns: 0.0,
            f64_gate_time_ns: 0.0,
            total_exec_time_ns: 0.0,
            accumulated_error: 0.0,
            peak_error: 0.0,
            threshold_exceeded: false,
            decisions: Vec::new(),
        }
    }
}

impl fmt::Display for PrecisionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrecisionStats {{ gates: {} (f32:{}, f64:{}), promotions:{}, demotions:{}, \
             trans_overhead_ms:{:.3}, error:{:.2e}, peak:{:.2e} }}",
            self.total_gates,
            self.f32_gates,
            self.f64_gates,
            self.promotions,
            self.demotions,
            (self.promotion_time_ns + self.demotion_time_ns) * 1e-6,
            self.accumulated_error,
            self.peak_error,
        )
    }
}

impl PrecisionStats {
    /// Fraction of end-to-end time spent in precision transitions.
    pub fn transition_overhead_ratio(&self) -> f64 {
        if self.total_exec_time_ns <= 0.0 {
            return 0.0;
        }
        ((self.promotion_time_ns + self.demotion_time_ns) / self.total_exec_time_ns).clamp(0.0, 1.0)
    }
}

// ============================================================
// MIXED PRECISION SIMULATOR
// ============================================================

/// Mixed-precision quantum simulator exploiting Apple Silicon UMA.
///
/// Maintains both an f64 (primary, CPU) and f32 (shadow, GPU-ready) state
/// vector.  On UMA, both reside in the same physical memory space, so
/// promotion (f32->f64) and demotion (f64->f32) are zero-copy local
/// operations (no DMA/PCIe transfer).
///
/// The simulator applies gates in the precision dictated by the active
/// [`PrecisionPolicy`], tracks accumulated error, and can auto-promote
/// to f64 when error exceeds the configured threshold.
pub struct MixedPrecisionSimulator {
    /// Primary f64 state (always authoritative).
    state_f64: QuantumState,
    /// Shadow f32 state (kept in sync when f32 path is used).
    state_f32: Vec<C32>,
    /// Number of qubits.
    num_qubits: usize,
    /// State vector dimension (2^num_qubits).
    dim: usize,
    /// Active precision policy.
    policy: PrecisionPolicy,
    /// Current working precision (may differ from policy default
    /// if error threshold triggers a switch).
    current_precision: GatePrecision,
    /// Accumulated precision error estimate.
    accumulated_error: f64,
    /// Error threshold: when accumulated_error exceeds this, force f64.
    error_threshold: f64,
    /// Per-gate context override (None = use default ForwardPass).
    gate_contexts: Vec<GateContext>,
    /// Custom per-gate precision selector (used when policy = Custom).
    custom_selector: Option<Box<dyn Fn(&Gate, usize) -> GatePrecision + Send + Sync>>,
    /// Backend used when executing f32 gates.
    f32_backend: F32ExecutionBackend,
    /// Optional reusable Metal simulator for f32 execution.
    #[cfg(target_os = "macos")]
    metal_sim: Option<MetalSimulator>,
}

impl MixedPrecisionSimulator {
    /// Create a new mixed-precision simulator.
    ///
    /// Both f64 and f32 states are initialized to |0...0>.
    pub fn new(num_qubits: usize, policy: PrecisionPolicy) -> Self {
        let dim = 1usize << num_qubits;
        let state_f64 = QuantumState::new(num_qubits);
        let mut state_f32 = vec![c32_zero(); dim];
        state_f32[0] = c32_one();
        #[cfg(target_os = "macos")]
        let (f32_backend, metal_sim) = match MetalSimulator::new(num_qubits) {
            Ok(sim) => (F32ExecutionBackend::MetalGpu, Some(sim)),
            Err(_) => (F32ExecutionBackend::CpuFallback, None),
        };
        #[cfg(not(target_os = "macos"))]
        let f32_backend = F32ExecutionBackend::CpuFallback;

        Self {
            state_f64,
            state_f32,
            num_qubits,
            dim,
            policy,
            current_precision: match policy {
                PrecisionPolicy::AlwaysF32 => GatePrecision::F32,
                _ => GatePrecision::F64,
            },
            accumulated_error: 0.0,
            error_threshold: 1e-6,
            gate_contexts: Vec::new(),
            custom_selector: None,
            f32_backend,
            #[cfg(target_os = "macos")]
            metal_sim,
        }
    }

    /// Builder: set the error threshold for automatic f64 promotion.
    pub fn with_error_threshold(mut self, threshold: f64) -> Self {
        self.error_threshold = threshold;
        self
    }

    /// Builder: set per-gate contexts for adaptive precision.
    ///
    /// `contexts[i]` gives the context for gate `i` in the circuit.
    /// If the vector is shorter than the circuit, remaining gates use
    /// `ForwardPass`.
    pub fn with_gate_contexts(mut self, contexts: Vec<GateContext>) -> Self {
        self.gate_contexts = contexts;
        self
    }

    /// Builder: set a custom per-gate precision selector.
    pub fn with_custom_selector<F>(mut self, selector: F) -> Self
    where
        F: Fn(&Gate, usize) -> GatePrecision + Send + Sync + 'static,
    {
        self.custom_selector = Some(Box::new(selector));
        self
    }

    /// Builder: select the execution backend for f32 gates.
    ///
    /// On non-macOS targets, `MetalGpu` is ignored and falls back to CPU.
    pub fn with_f32_backend(mut self, backend: F32ExecutionBackend) -> Self {
        #[cfg(target_os = "macos")]
        {
            match backend {
                F32ExecutionBackend::CpuFallback => {
                    self.f32_backend = F32ExecutionBackend::CpuFallback;
                    self.metal_sim = None;
                }
                F32ExecutionBackend::MetalGpu => {
                    if self.metal_sim.is_none() {
                        self.metal_sim = MetalSimulator::new(self.num_qubits).ok();
                    }
                    self.f32_backend = if self.metal_sim.is_some() {
                        F32ExecutionBackend::MetalGpu
                    } else {
                        F32ExecutionBackend::CpuFallback
                    };
                }
            }
            return self;
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = backend;
            self.f32_backend = F32ExecutionBackend::CpuFallback;
            self
        }
    }

    /// Get the current precision policy.
    pub fn policy(&self) -> PrecisionPolicy {
        self.policy
    }

    /// Get the current working precision.
    pub fn current_precision(&self) -> GatePrecision {
        self.current_precision
    }

    /// Get the accumulated precision error.
    pub fn accumulated_error(&self) -> f64 {
        self.accumulated_error
    }

    /// Get the number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Backend currently used for f32 gate execution.
    pub fn f32_execution_backend(&self) -> F32ExecutionBackend {
        self.f32_backend
    }

    /// Get state dimension (2^n).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Read the f64 state.
    pub fn state_f64(&self) -> &QuantumState {
        &self.state_f64
    }

    /// Read the f32 state.
    pub fn state_f32(&self) -> &[C32] {
        &self.state_f32
    }

    /// Read probabilities from the authoritative (f64) state.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state_f64.probabilities()
    }

    /// Reset both states to |0...0>.
    pub fn reset(&mut self) {
        self.state_f64 = QuantumState::new(self.num_qubits);
        self.state_f32 = vec![c32_zero(); self.dim];
        self.state_f32[0] = c32_one();
        #[cfg(target_os = "macos")]
        if let Some(sim) = self.metal_sim.as_mut() {
            let _ = sim.write_state_f32(&self.state_f32);
        }
        self.accumulated_error = 0.0;
        self.current_precision = match self.policy {
            PrecisionPolicy::AlwaysF32 => GatePrecision::F32,
            _ => GatePrecision::F64,
        };
    }

    /// Promote f32 state to f64 (zero-copy on UMA: just a local conversion).
    ///
    /// Overwrites the f64 state with values from the f32 state.
    /// This is the "precision promotion" operation.
    pub fn promote_f32_to_f64(&mut self) {
        let amps = self.state_f64.amplitudes_mut();
        for (dst, src) in amps.iter_mut().zip(self.state_f32.iter()) {
            *dst = C64::new(src.re as f64, src.im as f64);
        }
        self.current_precision = GatePrecision::F64;
    }

    /// Demote f64 state to f32 (zero-copy on UMA: just a local conversion).
    ///
    /// Overwrites the f32 state with values from the f64 state.
    /// This is the "precision demotion" operation.
    pub fn demote_f64_to_f32(&mut self) {
        let amps = self.state_f64.amplitudes_ref();
        for (dst, src) in self.state_f32.iter_mut().zip(amps.iter()) {
            *dst = C32::new(src.re as f32, src.im as f32);
        }
        self.current_precision = GatePrecision::F32;
    }

    /// Compute the precision error.
    ///
    /// When running in f64 mode, compare the authoritative f64 amplitudes
    /// against an on-the-fly f32 quantization of those same amplitudes. This
    /// avoids reporting a false large error when the f32 shadow is stale.
    ///
    /// When running in f32 mode, compare the f64 shadow state against the f32
    /// amplitudes directly.
    pub fn compute_precision_error(&self) -> f64 {
        let amps64 = self.state_f64.amplitudes_ref();
        let mut error_sq = 0.0f64;
        match self.current_precision {
            GatePrecision::F64 => {
                for a64 in amps64 {
                    // Quantize to f32 and lift back to f64 for a local truncation estimate.
                    let q_re = (a64.re as f32) as f64;
                    let q_im = (a64.im as f32) as f64;
                    let diff_re = a64.re - q_re;
                    let diff_im = a64.im - q_im;
                    error_sq += diff_re * diff_re + diff_im * diff_im;
                }
            }
            GatePrecision::F32 => {
                for (a64, a32) in amps64.iter().zip(self.state_f32.iter()) {
                    let diff_re = a64.re - a32.re as f64;
                    let diff_im = a64.im - a32.im as f64;
                    error_sq += diff_re * diff_re + diff_im * diff_im;
                }
            }
        }
        error_sq.sqrt()
    }

    /// Execute a circuit with mixed precision.
    ///
    /// For each gate, the simulator:
    /// 1. Determines the gate's precision (policy + context + error tracking).
    /// 2. Applies the gate in the chosen precision.
    /// 3. Updates the shadow state if needed.
    /// 4. Tracks accumulated error.
    pub fn execute_circuit(&mut self, gates: &[Gate]) -> PrecisionStats {
        self.execute_circuit_impl(gates, false)
            .expect("non-strict mixed-precision execution should not fail")
    }

    /// Execute a circuit in strict mode.
    ///
    /// Strict mode rejects f32 execution if it cannot run on the configured
    /// backend (e.g. Metal requested but unavailable). This prevents silent
    /// degradation to CPU emulation in production.
    pub fn execute_circuit_strict(
        &mut self,
        gates: &[Gate],
    ) -> Result<PrecisionStats, String> {
        self.execute_circuit_impl(gates, true)
    }

    fn execute_circuit_impl(
        &mut self,
        gates: &[Gate],
        strict_backend: bool,
    ) -> Result<PrecisionStats, String> {
        const ERROR_CHECK_INTERVAL: usize = 128;
        let start = std::time::Instant::now();
        let mut stats = PrecisionStats::default();
        let mut prev_precision = self.current_precision;

        for (i, gate) in gates.iter().enumerate() {
            let context = self
                .gate_contexts
                .get(i)
                .copied()
                .unwrap_or(GateContext::ForwardPass);

            let gate_precision = self.select_precision(gate, i, context);

            // Track promotions/demotions.
            if gate_precision != prev_precision {
                match gate_precision {
                    GatePrecision::F64 => {
                        if prev_precision == GatePrecision::F32 {
                            // Promote: sync f64 from f32 before switching.
                            let t0 = std::time::Instant::now();
                            self.promote_f32_to_f64();
                            stats.promotion_time_ns += t0.elapsed().as_nanos() as f64;
                            stats.promotions += 1;
                        }
                    }
                    GatePrecision::F32 => {
                        if prev_precision == GatePrecision::F64 {
                            // Demote: sync f32 from f64 before switching.
                            let t0 = std::time::Instant::now();
                            self.demote_f64_to_f32();
                            stats.demotion_time_ns += t0.elapsed().as_nanos() as f64;
                            stats.demotions += 1;
                        }
                    }
                }
            }

            // Apply gate in the chosen precision.
            match gate_precision {
                GatePrecision::F64 => {
                    let t0 = std::time::Instant::now();
                    apply_gate_to_state(&mut self.state_f64, gate);
                    stats.f64_gate_time_ns += t0.elapsed().as_nanos() as f64;
                    stats.f64_gates += 1;
                }
                GatePrecision::F32 => {
                    let t0 = std::time::Instant::now();
                    self.apply_gate_f32(gate, strict_backend)?;
                    stats.f32_gate_time_ns += t0.elapsed().as_nanos() as f64;
                    stats.f32_gates += 1;
                }
            }
            stats.total_gates += 1;

            // Update error tracking periodically or when precision changes.
            if i % ERROR_CHECK_INTERVAL == 0 || gate_precision != prev_precision {
                // Keep both states in sync for error measurement.
                if gate_precision == GatePrecision::F64 {
                    // f64 is authoritative; update f32 shadow.
                    let t0 = std::time::Instant::now();
                    self.demote_f64_to_f32();
                    stats.demotion_time_ns += t0.elapsed().as_nanos() as f64;
                    self.current_precision = GatePrecision::F64; // restore after demote
                } else {
                    // f32 is being used; update f64 shadow for comparison.
                    let t0 = std::time::Instant::now();
                    self.promote_f32_to_f64();
                    stats.promotion_time_ns += t0.elapsed().as_nanos() as f64;
                    self.current_precision = GatePrecision::F32; // restore after promote
                }
                self.accumulated_error = self.compute_precision_error();
                if self.accumulated_error > stats.peak_error {
                    stats.peak_error = self.accumulated_error;
                }
                if self.accumulated_error > self.error_threshold {
                    stats.threshold_exceeded = true;
                }
            }

            let reason = format!(
                "gate {}: {} ({}) in {} [err={:.2e}]",
                i,
                gate_type_name(&gate.gate_type),
                context_name(context),
                gate_precision,
                self.accumulated_error,
            );
            stats.decisions.push((i, gate_precision, reason));
            prev_precision = gate_precision;
            self.current_precision = gate_precision;
        }

        // Final sync: ensure f64 is authoritative.
        if self.current_precision == GatePrecision::F32 {
            let t0 = std::time::Instant::now();
            self.promote_f32_to_f64();
            stats.promotion_time_ns += t0.elapsed().as_nanos() as f64;
        }

        stats.accumulated_error = self.accumulated_error;
        stats.total_exec_time_ns = start.elapsed().as_nanos() as f64;
        Ok(stats)
    }

    /// Apply a single gate in f32 precision.
    ///
    /// Uses the selected f32 execution backend (Metal on macOS when available,
    /// otherwise CPU fallback emulation).
    fn apply_gate_f32(&mut self, gate: &Gate, strict_backend: bool) -> Result<(), String> {
        if strict_backend && self.f32_backend == F32ExecutionBackend::CpuFallback {
            return Err(
                "strict mode requires MetalGpu backend for f32 execution; CpuFallback configured"
                    .to_string(),
            );
        }

        #[cfg(target_os = "macos")]
        if self.f32_backend == F32ExecutionBackend::MetalGpu {
            if let Some(sim) = self.metal_sim.as_mut() {
                if sim.write_state_f32(&self.state_f32).is_ok() {
                    sim.run_circuit(std::slice::from_ref(gate));
                    self.state_f32 = sim.read_state_f32();
                    return Ok(());
                }
                if strict_backend {
                    return Err(
                        "strict mode requested MetalGpu f32 execution, but Metal state upload failed"
                            .to_string(),
                    );
                }
                // If Metal write fails, downgrade to CPU fallback for safety.
                self.f32_backend = F32ExecutionBackend::CpuFallback;
                self.metal_sim = None;
            } else if strict_backend {
                return Err(
                    "strict mode requested MetalGpu f32 execution, but Metal backend is unavailable"
                        .to_string(),
                );
            }
        }

        #[cfg(not(target_os = "macos"))]
        if strict_backend {
            return Err(
                "strict mode requested f32 backend execution, but MetalGpu is unavailable on this platform"
                    .to_string(),
            );
        }

        // CPU fallback path: apply directly on f32 state when supported.
        if apply_gate_f32_in_place(&mut self.state_f32, self.num_qubits, gate) {
            return Ok(());
        }

        // Generic fallback for uncommon gates: round-trip through f64 implementation.
        let amps = self.state_f64.amplitudes_mut();
        for (dst, src) in amps.iter_mut().zip(self.state_f32.iter()) {
            *dst = C64::new(src.re as f64, src.im as f64);
        }
        apply_gate_to_state(&mut self.state_f64, gate);
        let amps_ref = self.state_f64.amplitudes_ref();
        for (dst, src) in self.state_f32.iter_mut().zip(amps_ref.iter()) {
            *dst = C32::new(src.re as f32, src.im as f32);
        }
        Ok(())
    }

    /// Select precision for a gate based on policy, context, and error state.
    fn select_precision(
        &self,
        gate: &Gate,
        gate_index: usize,
        context: GateContext,
    ) -> GatePrecision {
        match self.policy {
            PrecisionPolicy::AlwaysF64 => GatePrecision::F64,
            PrecisionPolicy::AlwaysF32 => GatePrecision::F32,
            PrecisionPolicy::Adaptive => {
                let (precision, _reason) = classify_gate_precision(
                    gate,
                    context,
                    self.accumulated_error,
                    self.error_threshold,
                );
                precision
            }
            PrecisionPolicy::Custom => {
                if let Some(ref selector) = self.custom_selector {
                    selector(gate, gate_index)
                } else {
                    // Fallback to f64 if no custom selector provided.
                    GatePrecision::F64
                }
            }
        }
    }
}

fn apply_gate_f32_in_place(state: &mut [C32], num_qubits: usize, gate: &Gate) -> bool {
    match &gate.gate_type {
        GateType::H => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_h_f32(state, num_qubits, q)),
        GateType::X => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_x_f32(state, num_qubits, q)),
        GateType::Y => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_y_f32(state, num_qubits, q)),
        GateType::Z => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_z_f32(state, num_qubits, q)),
        GateType::S => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_s_f32(state, num_qubits, q)),
        GateType::T => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_t_f32(state, num_qubits, q)),
        GateType::Rx(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_rx_f32(state, num_qubits, q, *theta as f32)),
        GateType::Ry(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_ry_f32(state, num_qubits, q, *theta as f32)),
        GateType::Rz(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_rz_f32(state, num_qubits, q, *theta as f32)),
        GateType::Phase(theta) => gate
            .targets
            .first()
            .copied()
            .is_some_and(|q| apply_phase_f32(state, num_qubits, q, *theta as f32)),
        GateType::CNOT => {
            if gate.controls.len() == 1 && gate.targets.len() == 1 {
                apply_cnot_f32(state, num_qubits, gate.controls[0], gate.targets[0])
            } else {
                false
            }
        }
        GateType::CZ => {
            if gate.controls.len() == 1 && gate.targets.len() == 1 {
                apply_cz_f32(state, num_qubits, gate.controls[0], gate.targets[0])
            } else {
                false
            }
        }
        GateType::SWAP => {
            if gate.targets.len() == 2 {
                apply_swap_f32(state, num_qubits, gate.targets[0], gate.targets[1])
            } else {
                false
            }
        }
        _ => false,
    }
}

fn apply_h_f32(state: &mut [C32], num_qubits: usize, qubit: usize) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    for base in (0..state.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = state[i0];
            let b = state[i1];
            state[i0] = C32::new((a.re + b.re) * inv_sqrt2, (a.im + b.im) * inv_sqrt2);
            state[i1] = C32::new((a.re - b.re) * inv_sqrt2, (a.im - b.im) * inv_sqrt2);
        }
    }
    true
}

fn apply_x_f32(state: &mut [C32], num_qubits: usize, qubit: usize) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    for base in (0..state.len()).step_by(stride * 2) {
        for i in 0..stride {
            state.swap(base + i, base + i + stride);
        }
    }
    true
}

fn apply_y_f32(state: &mut [C32], num_qubits: usize, qubit: usize) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    for base in (0..state.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = state[i0];
            let b = state[i1];
            state[i0] = C32::new(b.im, -b.re);
            state[i1] = C32::new(-a.im, a.re);
        }
    }
    true
}

fn apply_z_f32(state: &mut [C32], num_qubits: usize, qubit: usize) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            amp.re = -amp.re;
            amp.im = -amp.im;
        }
    }
    true
}

fn apply_s_f32(state: &mut [C32], num_qubits: usize, qubit: usize) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            let re = amp.re;
            amp.re = -amp.im;
            amp.im = re;
        }
    }
    true
}

fn apply_t_f32(state: &mut [C32], num_qubits: usize, qubit: usize) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    let phase_re = std::f32::consts::FRAC_1_SQRT_2;
    let phase_im = std::f32::consts::FRAC_1_SQRT_2;
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            let re = amp.re * phase_re - amp.im * phase_im;
            let im = amp.re * phase_im + amp.im * phase_re;
            amp.re = re;
            amp.im = im;
        }
    }
    true
}

fn apply_phase_f32(state: &mut [C32], num_qubits: usize, qubit: usize, theta: f32) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let mask = 1usize << qubit;
    let phase_re = theta.cos();
    let phase_im = theta.sin();
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & mask) != 0 {
            let re = amp.re * phase_re - amp.im * phase_im;
            let im = amp.re * phase_im + amp.im * phase_re;
            amp.re = re;
            amp.im = im;
        }
    }
    true
}

fn apply_rx_f32(state: &mut [C32], num_qubits: usize, qubit: usize, theta: f32) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    for base in (0..state.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = state[i0];
            let b = state[i1];
            state[i0] = C32::new(c * a.re + s * b.im, c * a.im - s * b.re);
            state[i1] = C32::new(s * a.im + c * b.re, -s * a.re + c * b.im);
        }
    }
    true
}

fn apply_ry_f32(state: &mut [C32], num_qubits: usize, qubit: usize, theta: f32) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    for base in (0..state.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = state[i0];
            let b = state[i1];
            state[i0] = C32::new(c * a.re - s * b.re, c * a.im - s * b.im);
            state[i1] = C32::new(s * a.re + c * b.re, s * a.im + c * b.im);
        }
    }
    true
}

fn apply_rz_f32(state: &mut [C32], num_qubits: usize, qubit: usize, theta: f32) -> bool {
    if qubit >= num_qubits {
        return false;
    }
    let stride = 1usize << qubit;
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    let p0_re = c;
    let p0_im = -s;
    let p1_re = c;
    let p1_im = s;
    for base in (0..state.len()).step_by(stride * 2) {
        for i in 0..stride {
            let i0 = base + i;
            let i1 = i0 + stride;
            let a = state[i0];
            let b = state[i1];
            state[i0] = C32::new(a.re * p0_re - a.im * p0_im, a.re * p0_im + a.im * p0_re);
            state[i1] = C32::new(b.re * p1_re - b.im * p1_im, b.re * p1_im + b.im * p1_re);
        }
    }
    true
}

fn apply_cnot_f32(state: &mut [C32], num_qubits: usize, control: usize, target: usize) -> bool {
    if control >= num_qubits || target >= num_qubits || control == target {
        return false;
    }
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    for idx in 0..state.len() {
        if (idx & control_mask) != 0 && (idx & target_mask) == 0 {
            let j = idx | target_mask;
            state.swap(idx, j);
        }
    }
    true
}

fn apply_cz_f32(state: &mut [C32], num_qubits: usize, control: usize, target: usize) -> bool {
    if control >= num_qubits || target >= num_qubits || control == target {
        return false;
    }
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & control_mask) != 0 && (idx & target_mask) != 0 {
            amp.re = -amp.re;
            amp.im = -amp.im;
        }
    }
    true
}

fn apply_swap_f32(state: &mut [C32], num_qubits: usize, q0: usize, q1: usize) -> bool {
    if q0 >= num_qubits || q1 >= num_qubits || q0 == q1 {
        return false;
    }
    let m0 = 1usize << q0;
    let m1 = 1usize << q1;
    for idx in 0..state.len() {
        let b0 = (idx & m0) != 0;
        let b1 = (idx & m1) != 0;
        if !b0 && b1 {
            let j = (idx | m0) & !m1;
            state.swap(idx, j);
        }
    }
    true
}

// ============================================================
// VQE MIXED PRECISION HELPERS
// ============================================================

/// Execute a VQE-style circuit with mixed precision.
///
/// Forward pass uses f32 for speed, gradient computation uses f64 for
/// precision.  The `parameter_gates` indices mark which gates are
/// parametric (these get f64 in gradient mode).
///
/// Returns (energy_estimate, gradient_precision_stats).
pub fn vqe_mixed_precision(
    gates: &[Gate],
    num_qubits: usize,
    parameter_gate_indices: &[usize],
) -> (Vec<f64>, PrecisionStats) {
    // Build contexts: parametric gates get ParametricSensitive,
    // last gate gets PreMeasurement, rest get ForwardPass.
    let mut contexts = vec![GateContext::ForwardPass; gates.len()];
    for &idx in parameter_gate_indices {
        if idx < contexts.len() {
            contexts[idx] = GateContext::ParametricSensitive;
        }
    }
    if !contexts.is_empty() {
        let last = contexts.len() - 1;
        contexts[last] = GateContext::PreMeasurement;
    }

    let mut sim = MixedPrecisionSimulator::new(num_qubits, PrecisionPolicy::Adaptive)
        .with_gate_contexts(contexts)
        .with_error_threshold(1e-6);

    let stats = sim.execute_circuit(gates);
    let probs = sim.probabilities();

    (probs, stats)
}

/// Execute a QAOA-style circuit with mixed precision.
///
/// All layers use f32 except measurement-adjacent gates.
pub fn qaoa_mixed_precision(
    gates: &[Gate],
    num_qubits: usize,
) -> (Vec<f64>, PrecisionStats) {
    let mut contexts = vec![GateContext::ForwardPass; gates.len()];
    // Mark last 10% of gates as pre-measurement.
    let meas_start = gates.len().saturating_sub(gates.len() / 10 + 1);
    for i in meas_start..gates.len() {
        contexts[i] = GateContext::PreMeasurement;
    }

    let mut sim = MixedPrecisionSimulator::new(num_qubits, PrecisionPolicy::Adaptive)
        .with_gate_contexts(contexts)
        .with_error_threshold(1e-5); // QAOA can tolerate slightly more error.

    let stats = sim.execute_circuit(gates);
    let probs = sim.probabilities();

    (probs, stats)
}

// ============================================================
// HELPERS
// ============================================================

/// Human-readable gate type name.
fn gate_type_name(gt: &GateType) -> String {
    match gt {
        GateType::H => "H".into(),
        GateType::X => "X".into(),
        GateType::Y => "Y".into(),
        GateType::Z => "Z".into(),
        GateType::S => "S".into(),
        GateType::T => "T".into(),
        GateType::Rx(a) => format!("Rx({:.4})", a),
        GateType::Ry(a) => format!("Ry({:.4})", a),
        GateType::Rz(a) => format!("Rz({:.4})", a),
        GateType::U { .. } => "U".into(),
        GateType::CNOT => "CNOT".into(),
        GateType::CZ => "CZ".into(),
        GateType::SWAP => "SWAP".into(),
        GateType::Toffoli => "Toffoli".into(),
        GateType::CRx(a) => format!("CRx({:.4})", a),
        GateType::CRy(a) => format!("CRy({:.4})", a),
        GateType::CRz(a) => format!("CRz({:.4})", a),
        GateType::CR(a) => format!("CR({:.4})", a),
        GateType::SX => "SX".into(),
        GateType::Phase(a) => format!("Phase({:.4})", a),
        GateType::ISWAP => "ISWAP".into(),
        GateType::CCZ => "CCZ".into(),
        GateType::Rxx(a) => format!("Rxx({:.4})", a),
        GateType::Ryy(a) => format!("Ryy({:.4})", a),
        GateType::Rzz(a) => format!("Rzz({:.4})", a),
        GateType::CSWAP => "CSWAP".into(),
        GateType::CU { .. } => "CU".into(),
        GateType::Custom(_) => "Custom".into(),
    }
}

/// Human-readable context name.
fn context_name(ctx: GateContext) -> &'static str {
    match ctx {
        GateContext::ForwardPass => "forward",
        GateContext::GradientComputation => "gradient",
        GateContext::PreMeasurement => "pre-meas",
        GateContext::ParametricSensitive => "parametric",
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    // ---- PrecisionPolicy tests ----

    #[test]
    fn test_policy_display() {
        assert_eq!(format!("{}", PrecisionPolicy::AlwaysF64), "AlwaysF64");
        assert_eq!(format!("{}", PrecisionPolicy::AlwaysF32), "AlwaysF32");
        assert_eq!(format!("{}", PrecisionPolicy::Adaptive), "Adaptive");
        assert_eq!(format!("{}", PrecisionPolicy::Custom), "Custom");
    }

    #[test]
    fn test_gate_precision_display() {
        assert_eq!(format!("{}", GatePrecision::F32), "f32");
        assert_eq!(format!("{}", GatePrecision::F64), "f64");
    }

    // ---- classify_gate_precision tests ----

    #[test]
    fn test_classify_gradient_always_f64() {
        let gate = Gate::h(0);
        let (prec, reason) =
            classify_gate_precision(&gate, GateContext::GradientComputation, 0.0, 1e-6);
        assert_eq!(prec, GatePrecision::F64);
        assert!(reason.contains("gradient"));
    }

    #[test]
    fn test_classify_pre_measurement_always_f64() {
        let gate = Gate::h(0);
        let (prec, _) =
            classify_gate_precision(&gate, GateContext::PreMeasurement, 0.0, 1e-6);
        assert_eq!(prec, GatePrecision::F64);
    }

    #[test]
    fn test_classify_forward_single_qubit_f32() {
        let gate = Gate::h(0);
        let (prec, _) = classify_gate_precision(&gate, GateContext::ForwardPass, 0.0, 1e-6);
        assert_eq!(prec, GatePrecision::F32);
    }

    #[test]
    fn test_classify_forward_toffoli_f64() {
        // Toffoli touches 3 qubits -> multi-qubit -> f64.
        let gate = Gate::toffoli(0, 1, 2);
        let (prec, _) = classify_gate_precision(&gate, GateContext::ForwardPass, 0.0, 1e-6);
        assert_eq!(prec, GatePrecision::F64);
    }

    #[test]
    fn test_classify_small_angle_parametric_f64() {
        let gate = Gate::rz(0, 0.001);
        let (prec, reason) =
            classify_gate_precision(&gate, GateContext::ParametricSensitive, 0.0, 1e-6);
        assert_eq!(prec, GatePrecision::F64);
        assert!(reason.contains("small-angle"));
    }

    #[test]
    fn test_classify_large_angle_parametric_f32() {
        let gate = Gate::rz(0, 1.5);
        let (prec, _) =
            classify_gate_precision(&gate, GateContext::ParametricSensitive, 0.0, 1e-6);
        assert_eq!(prec, GatePrecision::F32);
    }

    #[test]
    fn test_classify_error_exceeds_threshold_forces_f64() {
        let gate = Gate::h(0);
        let (prec, reason) =
            classify_gate_precision(&gate, GateContext::ForwardPass, 1e-5, 1e-6);
        assert_eq!(prec, GatePrecision::F64);
        assert!(reason.contains("exceeds threshold"));
    }

    // ---- MixedPrecisionSimulator tests ----

    #[test]
    fn test_simulator_creation() {
        let sim = MixedPrecisionSimulator::new(4, PrecisionPolicy::AlwaysF64);
        assert_eq!(sim.num_qubits(), 4);
        assert_eq!(sim.dim(), 16);
        assert_eq!(sim.policy(), PrecisionPolicy::AlwaysF64);
        assert_eq!(sim.current_precision(), GatePrecision::F64);
    }

    #[test]
    fn test_simulator_always_f64_bell_state() {
        let mut sim = MixedPrecisionSimulator::new(2, PrecisionPolicy::AlwaysF64);
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let stats = sim.execute_circuit(&gates);

        assert_eq!(stats.f64_gates, 2);
        assert_eq!(stats.f32_gates, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.demotions, 0);

        let probs = sim.probabilities();
        assert!(
            (probs[0] - 0.5).abs() < 1e-10,
            "|00> = {}",
            probs[0]
        );
        assert!(probs[1].abs() < 1e-10, "|01> = {}", probs[1]);
        assert!(probs[2].abs() < 1e-10, "|10> = {}", probs[2]);
        assert!(
            (probs[3] - 0.5).abs() < 1e-10,
            "|11> = {}",
            probs[3]
        );
    }

    #[test]
    fn test_simulator_always_f32_bell_state() {
        let mut sim = MixedPrecisionSimulator::new(2, PrecisionPolicy::AlwaysF32);
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let stats = sim.execute_circuit(&gates);

        assert_eq!(stats.f32_gates, 2);
        assert_eq!(stats.f64_gates, 0);

        // f32 should still produce correct Bell state (within f32 tolerance).
        let probs = sim.probabilities();
        assert!(
            (probs[0] - 0.5).abs() < 1e-5,
            "|00> = {}",
            probs[0]
        );
        assert!(
            (probs[3] - 0.5).abs() < 1e-5,
            "|11> = {}",
            probs[3]
        );
    }

    #[test]
    fn test_simulator_adaptive_uses_both_precisions() {
        let contexts = vec![
            GateContext::ForwardPass,
            GateContext::ForwardPass,
            GateContext::GradientComputation,
        ];
        let mut sim = MixedPrecisionSimulator::new(2, PrecisionPolicy::Adaptive)
            .with_gate_contexts(contexts);
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1)];
        let stats = sim.execute_circuit(&gates);

        // H gates in forward pass should be f32, CNOT in gradient should be f64.
        assert!(stats.f32_gates > 0, "should have f32 gates");
        assert!(stats.f64_gates > 0, "should have f64 gates");
        assert_eq!(stats.total_gates, 3);
    }

    #[test]
    fn test_simulator_custom_selector() {
        // Custom: even gates f32, odd gates f64.
        let mut sim =
            MixedPrecisionSimulator::new(2, PrecisionPolicy::Custom).with_custom_selector(
                |_gate, idx| {
                    if idx % 2 == 0 {
                        GatePrecision::F32
                    } else {
                        GatePrecision::F64
                    }
                },
            );
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1)];
        let stats = sim.execute_circuit(&gates);

        // Gates 0, 2 -> f32; gate 1 -> f64.
        assert_eq!(stats.f32_gates, 2);
        assert_eq!(stats.f64_gates, 1);
    }

    #[test]
    fn test_promote_demote_roundtrip() {
        let mut sim = MixedPrecisionSimulator::new(3, PrecisionPolicy::AlwaysF64);
        // Apply some gates to create non-trivial state.
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::h(2)];
        sim.execute_circuit(&gates);

        // Demote to f32 then promote back; check error is small.
        sim.demote_f64_to_f32();
        let error_after_demote = sim.compute_precision_error();
        // f64 was overwritten by promote, so error should be 0 after full roundtrip.
        sim.promote_f32_to_f64();
        let probs = sim.probabilities();
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "normalization after roundtrip: {}",
            total
        );
    }

    #[test]
    fn test_precision_error_computation() {
        let mut sim = MixedPrecisionSimulator::new(2, PrecisionPolicy::AlwaysF64);
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        sim.execute_circuit(&gates);

        // Error should be very small (both states were kept in sync).
        let error = sim.compute_precision_error();
        assert!(
            error < 1e-6,
            "precision error should be small: {}",
            error
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sim = MixedPrecisionSimulator::new(2, PrecisionPolicy::AlwaysF64);
        sim.execute_circuit(&[Gate::h(0)]);
        sim.reset();

        let probs = sim.probabilities();
        assert!(
            (probs[0] - 1.0).abs() < 1e-10,
            "reset should restore |00>: prob[0]={}",
            probs[0]
        );
        assert!(
            sim.accumulated_error().abs() < 1e-15,
            "reset should clear error"
        );
    }

    #[test]
    fn test_stats_display() {
        let stats = PrecisionStats {
            total_gates: 10,
            f32_gates: 7,
            f64_gates: 3,
            promotions: 1,
            demotions: 1,
            promotion_time_ns: 10_000.0,
            demotion_time_ns: 8_000.0,
            f32_gate_time_ns: 40_000.0,
            f64_gate_time_ns: 20_000.0,
            total_exec_time_ns: 100_000.0,
            accumulated_error: 1.5e-7,
            peak_error: 2.0e-7,
            threshold_exceeded: false,
            decisions: Vec::new(),
        };
        let s = format!("{}", stats);
        assert!(s.contains("gates: 10"));
        assert!(s.contains("f32:7"));
        assert!(s.contains("f64:3"));
        assert!(stats.transition_overhead_ratio() > 0.0);
    }

    // ---- VQE/QAOA helper tests ----

    #[test]
    fn test_vqe_mixed_precision_basic() {
        let gates = vec![
            Gate::h(0),
            Gate::ry(1, 0.5),
            Gate::cnot(0, 1),
            Gate::rz(0, 0.3),
        ];
        let param_indices = vec![1, 3]; // Ry and Rz are parametric.
        let (probs, stats) = vqe_mixed_precision(&gates, 2, &param_indices);

        assert_eq!(probs.len(), 4);
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "VQE probs should sum to 1: {}",
            total
        );
        assert_eq!(stats.total_gates, 4);
    }

    #[test]
    fn test_qaoa_mixed_precision_basic() {
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::rz(0, 0.5),
            Gate::rz(1, 0.5),
        ];
        let (probs, stats) = qaoa_mixed_precision(&gates, 2);

        assert_eq!(probs.len(), 4);
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "QAOA probs should sum to 1: {}",
            total
        );
        assert_eq!(stats.total_gates, 5);
    }

    #[test]
    fn test_strict_mode_rejects_cpu_fallback_backend() {
        let mut sim = MixedPrecisionSimulator::new(1, PrecisionPolicy::AlwaysF32)
            .with_f32_backend(F32ExecutionBackend::CpuFallback);
        let err = sim
            .execute_circuit_strict(&[Gate::h(0)])
            .expect_err("strict mode should reject CpuFallback for f32 execution");
        assert!(err.contains("strict mode"));
    }

    #[test]
    fn test_strict_mode_allows_f64_only_execution() {
        let mut sim = MixedPrecisionSimulator::new(1, PrecisionPolicy::AlwaysF64)
            .with_f32_backend(F32ExecutionBackend::CpuFallback);
        let stats = sim
            .execute_circuit_strict(&[Gate::h(0)])
            .expect("strict mode should permit f64-only execution");
        assert_eq!(stats.f64_gates, 1);
        assert_eq!(stats.f32_gates, 0);
    }
}
