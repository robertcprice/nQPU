//! Compilation-Informed Probabilistic Error Cancellation (CI-PEC)
//!
//! Bridges quantum error mitigation (QEM) and quantum error correction (QEC)
//! by making PEC aware of the fault-tolerant compilation strategy. Rather than
//! treating noise as a monolithic channel, CI-PEC decomposes the total noise
//! budget into physical noise, compilation-induced noise, and PEC sampling
//! overhead, then optimally allocates across these three axes.
//!
//! # Motivation (arXiv:2508.20174)
//!
//! Standard PEC treats the noisy channel as a black box. However, when running
//! fault-tolerant circuits, much of the effective noise budget comes from the
//! compilation itself: T-gate injection fidelity, Clifford synthesis depth,
//! and SWAP routing overhead. By incorporating this structure, CI-PEC can:
//!
//! 1. Reduce sampling overhead by exploiting partial QEC protection
//! 2. Target mitigation at high-error layers (compilation-heavy gates)
//! 3. Find the optimal balance between QEC distance and PEC budget
//!
//! # Architecture
//!
//! ```text
//!   Circuit (arbitrary rotations)
//!        │
//!   ┌────▼─────────────────┐
//!   │  FT Compilation      │  ← ft_compilation.rs
//!   │  (Clifford+T output) │
//!   └────┬─────────────────┘
//!        │
//!   ┌────▼─────────────────────────────┐
//!   │  CompilationAwareNoise           │
//!   │  ├─ Physical noise (hardware)    │
//!   │  ├─ T-gate injection noise       │
//!   │  ├─ Clifford synthesis noise     │
//!   │  └─ Routing overhead noise       │
//!   └────┬─────────────────────────────┘
//!        │
//!   ┌────▼────────────────┐    ┌───────────────────────┐
//!   │  ErrorBudgetOptimizer│    │  TwirledPec           │
//!   │  Pareto QEC vs PEC  │    │  Twirl + PEC combined │
//!   └────┬────────────────┘    └──────────┬────────────┘
//!        │                                │
//!   ┌────▼────────────────────────────────▼──┐
//!   │  FtPec: Logical-level PEC              │
//!   │  PEC overhead reduced by partial QEC   │
//!   └────┬───────────────────────────────────┘
//!        │
//!   ┌────▼──────────────────┐
//!   │  PecAnalysis          │
//!   │  Compare: PEC / FT-PEC│
//!   │  / full QEC / none    │
//!   └───────────────────────┘
//! ```
//!
//! # References
//!
//! - arXiv:2508.20174, "Compilation-Informed Error Mitigation" (2025)
//! - Temme, Bravyi, Gambetta, PRL 119, 180509 (2017) [PEC foundations]
//! - van den Berg et al., Nature Physics 19, 1116 (2023) [sparse PEC]
//! - Wallman & Emerson, PRA 94, 052325 (2016) [randomized compiling]

use rand::rngs::StdRng;
use rand::Rng;
use std::fmt;

use crate::ft_compilation::{CliffordTGate, FTCompilationResult};

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during compilation-informed PEC operations.
#[derive(Debug, Clone)]
pub enum CiPecError {
    /// A noise parameter is outside its valid range.
    InvalidNoiseParameter(String),
    /// Configuration parameter is invalid.
    InvalidConfig(String),
    /// Sampling overhead exceeds the allowed budget.
    OverheadExceeded { computed: f64, max_allowed: f64 },
    /// The noise channel is not invertible at the given parameters.
    NonInvertible(String),
    /// Optimization did not converge.
    OptimizationFailed(String),
}

impl fmt::Display for CiPecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CiPecError::InvalidNoiseParameter(msg) => {
                write!(f, "Invalid noise parameter: {}", msg)
            }
            CiPecError::InvalidConfig(msg) => {
                write!(f, "Invalid CI-PEC config: {}", msg)
            }
            CiPecError::OverheadExceeded {
                computed,
                max_allowed,
            } => {
                write!(
                    f,
                    "Sampling overhead {:.4} exceeds max allowed {:.4}",
                    computed, max_allowed
                )
            }
            CiPecError::NonInvertible(msg) => {
                write!(f, "Non-invertible channel: {}", msg)
            }
            CiPecError::OptimizationFailed(msg) => {
                write!(f, "Optimization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for CiPecError {}

// ============================================================
// COMPILATION STRATEGY
// ============================================================

/// The fault-tolerant compilation strategy used, which determines the
/// shape of the compilation-induced noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationStrategy {
    /// Direct synthesis: rotations are approximated directly in Clifford+T.
    /// Noise dominated by T-gate injection fidelity and synthesis depth.
    DirectSynthesis,
    /// Clifford+T decomposition with T-count optimization.
    /// Lower T-count but potentially deeper Clifford circuits.
    CliffordT,
    /// Pauli-Based Computation (Litinski transformation).
    /// Noise from multi-qubit Pauli measurements via lattice surgery.
    PauliFrame,
    /// User-supplied noise model for custom compilation pipelines.
    Custom,
}

impl fmt::Display for CompilationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilationStrategy::DirectSynthesis => write!(f, "DirectSynthesis"),
            CompilationStrategy::CliffordT => write!(f, "Clifford+T"),
            CompilationStrategy::PauliFrame => write!(f, "PauliFrame"),
            CompilationStrategy::Custom => write!(f, "Custom"),
        }
    }
}

// ============================================================
// PEC CONFIGURATION
// ============================================================

/// Configuration for the core PEC engine.
#[derive(Debug, Clone)]
pub struct PecConfig {
    /// Maximum allowed sampling overhead before aborting.
    pub max_overhead: f64,
    /// Number of Monte Carlo samples for expectation estimation.
    pub num_samples: usize,
    /// Depolarizing noise rate for the noise model (physical error rate).
    pub noise_rate: f64,
    /// Random seed for reproducibility (None = use entropy).
    pub seed: Option<u64>,
}

impl Default for PecConfig {
    fn default() -> Self {
        Self {
            max_overhead: 1000.0,
            num_samples: 10_000,
            noise_rate: 0.01,
            seed: None,
        }
    }
}

impl PecConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), CiPecError> {
        if self.max_overhead <= 0.0 {
            return Err(CiPecError::InvalidConfig(
                "max_overhead must be positive".into(),
            ));
        }
        if self.num_samples == 0 {
            return Err(CiPecError::InvalidConfig(
                "num_samples must be > 0".into(),
            ));
        }
        if self.noise_rate < 0.0 || self.noise_rate > 1.0 {
            return Err(CiPecError::InvalidNoiseParameter(format!(
                "noise_rate must be in [0, 1], got {}",
                self.noise_rate
            )));
        }
        Ok(())
    }
}

// ============================================================
// QUASI-PROBABILITY DECOMPOSITION ENTRY
// ============================================================

/// A single term in a quasi-probability decomposition.
///
/// Represents one implementable operation with its quasi-probability
/// coefficient. Negative coefficients encode the "virtual" operations
/// needed to invert noise.
#[derive(Debug, Clone)]
pub struct QpdTerm {
    /// Pauli operation index: 0=I, 1=X, 2=Y, 3=Z.
    pub pauli_index: usize,
    /// Quasi-probability coefficient (can be negative).
    pub coefficient: f64,
}

/// Result of a quasi-probability decomposition for a single gate.
#[derive(Debug, Clone)]
pub struct QpdDecomposition {
    /// The QPD terms (one per implementable operation).
    pub terms: Vec<QpdTerm>,
    /// One-norm overhead gamma = sum |eta_i|.
    pub overhead: f64,
    /// Normalized sampling distribution |eta_i| / gamma.
    pub sampling_probs: Vec<f64>,
}

impl QpdDecomposition {
    /// Build QPD for a depolarizing channel with error rate p.
    ///
    /// For depolarizing noise E(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ),
    /// the inverse channel has Pauli decomposition:
    ///
    ///   c_I = (1 + 3*eta) / 4
    ///   c_X = c_Y = c_Z = (1 - eta) / 4
    ///
    /// where eta = 1 / (1 - 4p/3).
    pub fn from_depolarizing(p: f64) -> Result<Self, CiPecError> {
        if p < 0.0 || p >= 0.75 {
            return Err(CiPecError::InvalidNoiseParameter(format!(
                "Depolarizing rate must be in [0, 0.75), got {}",
                p
            )));
        }
        if p < 1e-15 {
            // Noiseless: identity decomposition
            return Ok(Self {
                terms: vec![QpdTerm {
                    pauli_index: 0,
                    coefficient: 1.0,
                }],
                overhead: 1.0,
                sampling_probs: vec![1.0],
            });
        }

        let lambda = 1.0 - 4.0 * p / 3.0;
        let eta = 1.0 / lambda;

        let c_i = (1.0 + 3.0 * eta) / 4.0;
        let c_xyz = (1.0 - eta) / 4.0;

        let coeffs = vec![c_i, c_xyz, c_xyz, c_xyz];
        let overhead: f64 = coeffs.iter().map(|c| c.abs()).sum();
        let sampling_probs: Vec<f64> = coeffs.iter().map(|c| c.abs() / overhead).collect();

        let terms: Vec<QpdTerm> = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| QpdTerm {
                pauli_index: i,
                coefficient: c,
            })
            .collect();

        Ok(Self {
            terms,
            overhead,
            sampling_probs,
        })
    }

    /// Validate internal consistency.
    pub fn is_valid(&self) -> bool {
        let computed_overhead: f64 = self.terms.iter().map(|t| t.coefficient.abs()).sum();
        if (computed_overhead - self.overhead).abs() > 1e-10 {
            return false;
        }
        let prob_sum: f64 = self.sampling_probs.iter().sum();
        (prob_sum - 1.0).abs() < 1e-10
    }
}

// ============================================================
// PEC ENGINE
// ============================================================

/// Core Probabilistic Error Cancellation engine.
///
/// Performs quasi-probability decomposition of ideal gates into noisy
/// operations, draws signed circuit samples, and averages to recover
/// the ideal expectation value.
pub struct PecEngine {
    /// Configuration parameters.
    pub config: PecConfig,
}

impl PecEngine {
    /// Create a new PEC engine with the given configuration.
    pub fn new(config: PecConfig) -> Result<Self, CiPecError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute the per-gate QPD for a depolarizing noise model.
    pub fn decompose_gate(&self, noise_rate: f64) -> Result<QpdDecomposition, CiPecError> {
        QpdDecomposition::from_depolarizing(noise_rate)
    }

    /// Compute the total sampling overhead for a circuit of given depth.
    ///
    /// For identical per-gate noise, gamma_total = gamma_per_gate ^ depth.
    pub fn total_overhead(&self, depth: usize) -> Result<f64, CiPecError> {
        let qpd = self.decompose_gate(self.config.noise_rate)?;
        Ok(qpd.overhead.powi(depth as i32))
    }

    /// Sample a single PEC instance: for each gate layer, draw an operation
    /// from the QPD and accumulate the sign.
    ///
    /// Returns (sampled_pauli_indices, accumulated_sign).
    pub fn sample_instance(
        &self,
        qpds: &[QpdDecomposition],
        rng: &mut StdRng,
    ) -> (Vec<usize>, f64) {
        let mut indices = Vec::with_capacity(qpds.len());
        let mut sign = 1.0;

        for qpd in qpds {
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut chosen = 0;

            for (idx, &prob) in qpd.sampling_probs.iter().enumerate() {
                cumulative += prob;
                if r <= cumulative {
                    chosen = idx;
                    break;
                }
            }

            let coeff = qpd.terms[chosen].coefficient;
            sign *= coeff.signum() * qpd.overhead;
            indices.push(qpd.terms[chosen].pauli_index);
        }

        (indices, sign)
    }

    /// Estimate the mitigated expectation value from signed samples.
    ///
    /// Each sample is (measurement_value, sign). The mitigated estimator is:
    ///   <O>_mitigated = (1/N) sum_i sign_i * value_i
    ///
    /// Returns (mean, standard_error).
    pub fn estimate_expectation(samples: &[(f64, f64)]) -> (f64, f64) {
        if samples.is_empty() {
            return (0.0, f64::INFINITY);
        }
        let n = samples.len() as f64;
        let values: Vec<f64> = samples.iter().map(|(val, sign)| sign * val).collect();

        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let stderr = (variance / n).sqrt();

        (mean, stderr)
    }

    /// Compute the number of samples needed for a target precision.
    ///
    /// N = gamma_total^2 * z^2 / epsilon^2
    /// where z is the z-score for 95% confidence (1.96).
    pub fn samples_needed(&self, depth: usize, precision: f64) -> Result<usize, CiPecError> {
        let gamma = self.total_overhead(depth)?;
        let z = 1.96; // 95% confidence
        let n = (gamma * gamma * z * z) / (precision * precision);
        Ok(n.ceil() as usize)
    }
}

// ============================================================
// COMPILATION-AWARE NOISE MODEL
// ============================================================

/// Noise model that accounts for the FT compilation strategy.
///
/// Decomposes the total effective noise into four additive components,
/// each arising from a distinct source in the compilation pipeline.
#[derive(Debug, Clone)]
pub struct CompilationAwareNoise {
    /// Physical noise rate from hardware (depolarizing per gate).
    pub physical_noise: f64,
    /// T-gate injection noise from magic state distillation fidelity.
    /// Each T gate has infidelity ~ 1 - F_distillation.
    pub t_gate_noise: f64,
    /// Clifford synthesis noise from decomposition depth.
    /// Scales with the number of Clifford gates in the synthesis.
    pub clifford_noise: f64,
    /// Routing overhead noise from SWAP gate insertion.
    /// Each SWAP ~ 3 CNOTs, each with physical error rate.
    pub routing_noise: f64,
    /// The compilation strategy that produced these noise components.
    pub strategy: CompilationStrategy,
}

impl CompilationAwareNoise {
    /// Create a noise model for a given compilation strategy.
    ///
    /// # Arguments
    /// * `physical_error_rate` - Base physical error rate per gate
    /// * `t_count` - Number of T gates in the compiled circuit
    /// * `clifford_depth` - Depth of Clifford synthesis layers
    /// * `num_swaps` - Number of routing SWAP gates inserted
    /// * `distillation_fidelity` - Magic state distillation fidelity
    /// * `strategy` - Compilation strategy used
    pub fn new(
        physical_error_rate: f64,
        t_count: usize,
        clifford_depth: usize,
        num_swaps: usize,
        distillation_fidelity: f64,
        strategy: CompilationStrategy,
    ) -> Self {
        // T-gate noise: each T gate introduces infidelity from distillation
        let t_gate_noise = t_count as f64 * (1.0 - distillation_fidelity);

        // Clifford noise: each Clifford layer accumulates physical error
        let clifford_noise = clifford_depth as f64 * physical_error_rate;

        // Routing noise: each SWAP = 3 CNOTs, each with 2-qubit error rate
        // Two-qubit gates are roughly 10x noisier than single-qubit
        let two_qubit_rate = physical_error_rate * 10.0;
        let routing_noise = num_swaps as f64 * 3.0 * two_qubit_rate;

        Self {
            physical_noise: physical_error_rate,
            t_gate_noise,
            clifford_noise,
            routing_noise,
            strategy,
        }
    }

    /// Build a compilation-aware noise model directly from FT compilation output.
    ///
    /// This integrates CI-PEC with `ft_compilation.rs` real metrics:
    /// - `t_count` comes from the compiled Clifford+T result
    /// - `clifford_depth` is estimated as `total_depth - t_depth`
    /// - `num_swaps` is estimated from long-range CNOT interactions
    pub fn from_ft_compilation_result(
        physical_error_rate: f64,
        compilation: &FTCompilationResult,
        distillation_fidelity: f64,
        strategy: CompilationStrategy,
    ) -> Self {
        let clifford_depth = compilation.total_depth.saturating_sub(compilation.t_depth);
        let num_swaps = estimate_routing_swaps(&compilation.clifford_t_circuit);
        Self::new(
            physical_error_rate,
            compilation.t_count,
            clifford_depth,
            num_swaps,
            distillation_fidelity,
            strategy,
        )
    }

    /// Total effective noise budget (sum of all components).
    ///
    /// This is the effective per-circuit-layer noise that PEC must cancel.
    pub fn total_noise(&self) -> f64 {
        self.physical_noise + self.t_gate_noise + self.clifford_noise + self.routing_noise
    }

    /// Fraction of total noise attributable to compilation (not hardware).
    pub fn compilation_fraction(&self) -> f64 {
        let total = self.total_noise();
        if total < 1e-15 {
            return 0.0;
        }
        (self.t_gate_noise + self.clifford_noise + self.routing_noise) / total
    }

    /// PEC overhead (gamma) for this noise model, treating the combined
    /// noise as a depolarizing channel.
    pub fn pec_overhead(&self) -> Result<f64, CiPecError> {
        let p = self.total_noise().min(0.749);
        let qpd = QpdDecomposition::from_depolarizing(p)?;
        Ok(qpd.overhead)
    }

    /// Per-component overhead breakdown.
    ///
    /// Returns (physical_gamma, t_gate_gamma, clifford_gamma, routing_gamma).
    pub fn component_overheads(&self) -> (f64, f64, f64, f64) {
        let gamma = |p: f64| -> f64 {
            if p < 1e-15 {
                return 1.0;
            }
            let p_clamped = p.min(0.749);
            let lambda = 1.0 - 4.0 * p_clamped / 3.0;
            1.0 / lambda.abs()
        };
        (
            gamma(self.physical_noise),
            gamma(self.t_gate_noise),
            gamma(self.clifford_noise),
            gamma(self.routing_noise),
        )
    }

    /// Create a default noise model for a given compilation strategy.
    pub fn for_strategy(
        strategy: CompilationStrategy,
        physical_error_rate: f64,
        circuit_depth: usize,
    ) -> Self {
        match strategy {
            CompilationStrategy::DirectSynthesis => Self::new(
                physical_error_rate,
                circuit_depth * 3, // ~3 T gates per rotation
                circuit_depth * 5, // ~5 Clifford gates per decomposition
                circuit_depth / 2, // moderate routing
                0.9999,            // high fidelity distillation
                strategy,
            ),
            CompilationStrategy::CliffordT => Self::new(
                physical_error_rate,
                circuit_depth * 2,  // optimized T count
                circuit_depth * 10, // deeper Clifford
                circuit_depth / 3,  // less routing
                0.99999,            // better distillation
                strategy,
            ),
            CompilationStrategy::PauliFrame => Self::new(
                physical_error_rate,
                circuit_depth,     // T gates from non-Clifford rotations
                circuit_depth * 2, // shallow Clifford frame
                circuit_depth,     // lattice surgery routing
                0.9999,
                strategy,
            ),
            CompilationStrategy::Custom => Self::new(
                physical_error_rate,
                0,
                0,
                0,
                1.0,
                strategy,
            ),
        }
    }
}

/// Estimate routing SWAP count from compiled Clifford+T gates.
///
/// For each non-adjacent CNOT, estimate `distance - 1` swaps to bring qubits
/// adjacent in a nearest-neighbor layout.
fn estimate_routing_swaps(gates: &[CliffordTGate]) -> usize {
    gates
        .iter()
        .map(|gate| match gate {
            CliffordTGate::CNOT(c, t) => c.abs_diff(*t).saturating_sub(1),
            _ => 0,
        })
        .sum()
}

/// Derive per-layer error rates directly from an FT-compiled Clifford+T circuit.
///
/// This converts concrete compilation artifacts into layerwise mitigation inputs,
/// so CI-PEC planning can target high-error compiled layers (T-heavy and
/// routing-heavy) rather than synthetic placeholder profiles.
pub fn layer_errors_from_ft_compilation(
    compilation: &FTCompilationResult,
    physical_error_rate: f64,
    distillation_fidelity: f64,
) -> Vec<f64> {
    let p_phys = physical_error_rate.max(0.0);
    if compilation.clifford_t_circuit.is_empty() {
        return Vec::new();
    }

    let max_q = compilation
        .clifford_t_circuit
        .iter()
        .flat_map(|g| g.qubits())
        .max()
        .unwrap_or(0);
    let mut qubit_depth = vec![0usize; max_q + 1];
    let mut layers: Vec<Vec<&CliffordTGate>> = Vec::new();

    // Build a qubit-aware layerization from the actual compiled gate sequence.
    for gate in &compilation.clifford_t_circuit {
        let qs = gate.qubits();
        if qs.is_empty() {
            continue;
        }
        let layer_idx = qs.iter().map(|q| qubit_depth[*q]).max().unwrap_or(0);
        if layers.len() <= layer_idx {
            layers.resize_with(layer_idx + 1, Vec::new);
        }
        layers[layer_idx].push(gate);
        for q in qs {
            qubit_depth[q] = layer_idx + 1;
        }
    }

    let two_qubit_rate = p_phys * 10.0;
    let t_noise = (1.0 - distillation_fidelity).max(0.0);
    let mut layer_errors = Vec::with_capacity(layers.len());

    for layer in layers {
        let mut err = p_phys;
        for gate in layer {
            match gate {
                CliffordTGate::T(_) | CliffordTGate::Tdg(_) => {
                    err += t_noise;
                }
                CliffordTGate::CNOT(c, t) => {
                    err += two_qubit_rate;
                    let distance = c.abs_diff(*t);
                    if distance > 1 {
                        // Approximate routing insertion cost: (distance-1) SWAPs,
                        // each SWAP ~ 3 CNOTs at two-qubit error rate.
                        err += (distance - 1) as f64 * 3.0 * two_qubit_rate;
                    }
                }
                CliffordTGate::Measure(_) => {
                    err += 2.0 * p_phys;
                }
                _ => {
                    err += p_phys;
                }
            }
        }
        layer_errors.push(err.min(0.749));
    }

    layer_errors
}

impl fmt::Display for CompilationAwareNoise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompilationAwareNoise(strategy={}, physical={:.2e}, \
             T-gate={:.2e}, Clifford={:.2e}, routing={:.2e}, total={:.2e})",
            self.strategy,
            self.physical_noise,
            self.t_gate_noise,
            self.clifford_noise,
            self.routing_noise,
            self.total_noise()
        )
    }
}

// ============================================================
// FAULT-TOLERANT PEC CONFIGURATION
// ============================================================

/// Configuration for fault-tolerant PEC.
#[derive(Debug, Clone)]
pub struct FtPecConfig {
    /// Surface code distance.
    pub code_distance: usize,
    /// Physical error rate per gate.
    pub physical_error_rate: f64,
    /// Compilation strategy used.
    pub compilation_strategy: CompilationStrategy,
    /// Maximum allowed sampling overhead.
    pub max_overhead: f64,
    /// Number of PEC samples.
    pub num_samples: usize,
    /// Magic state distillation fidelity.
    pub distillation_fidelity: f64,
}

impl Default for FtPecConfig {
    fn default() -> Self {
        Self {
            code_distance: 3,
            physical_error_rate: 1e-3,
            compilation_strategy: CompilationStrategy::CliffordT,
            max_overhead: 1e6,
            num_samples: 100_000,
            distillation_fidelity: 0.9999,
        }
    }
}

impl FtPecConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), CiPecError> {
        if self.code_distance < 1 || self.code_distance % 2 == 0 {
            return Err(CiPecError::InvalidConfig(format!(
                "code_distance must be odd and >= 1, got {}",
                self.code_distance
            )));
        }
        if self.physical_error_rate < 0.0 || self.physical_error_rate > 1.0 {
            return Err(CiPecError::InvalidNoiseParameter(format!(
                "physical_error_rate must be in [0, 1], got {}",
                self.physical_error_rate
            )));
        }
        if self.distillation_fidelity < 0.0 || self.distillation_fidelity > 1.0 {
            return Err(CiPecError::InvalidNoiseParameter(format!(
                "distillation_fidelity must be in [0, 1], got {}",
                self.distillation_fidelity
            )));
        }
        if self.max_overhead <= 0.0 {
            return Err(CiPecError::InvalidConfig(
                "max_overhead must be positive".into(),
            ));
        }
        if self.num_samples == 0 {
            return Err(CiPecError::InvalidConfig(
                "num_samples must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ============================================================
// FAULT-TOLERANT PEC
// ============================================================

/// PEC operating at the logical (QEC-encoded) level.
///
/// Combines partial QEC protection with PEC to reduce sampling overhead.
/// The logical noise rate is exponentially suppressed by code distance:
///
///   p_logical ~ (p_physical / p_threshold)^((d+1)/2)
///
/// PEC then only needs to cancel this reduced logical noise, yielding
/// dramatically lower overhead compared to bare PEC on physical qubits.
pub struct FtPec {
    /// Configuration.
    pub config: FtPecConfig,
    /// Compilation-aware noise model.
    pub noise_model: CompilationAwareNoise,
}

impl FtPec {
    /// Create a new FT-PEC instance.
    pub fn new(config: FtPecConfig) -> Result<Self, CiPecError> {
        config.validate()?;

        let noise_model = CompilationAwareNoise::for_strategy(
            config.compilation_strategy,
            config.physical_error_rate,
            1, // per-layer noise
        );

        Ok(Self {
            config,
            noise_model,
        })
    }

    /// Create FT-PEC using metrics extracted from a real FT compilation result.
    pub fn from_compilation_result(
        config: FtPecConfig,
        compilation: &FTCompilationResult,
    ) -> Result<Self, CiPecError> {
        config.validate()?;
        let noise_model = CompilationAwareNoise::from_ft_compilation_result(
            config.physical_error_rate,
            compilation,
            config.distillation_fidelity,
            config.compilation_strategy,
        );
        Ok(Self {
            config,
            noise_model,
        })
    }

    /// Create FT-PEC with a custom noise model.
    pub fn with_noise_model(
        config: FtPecConfig,
        noise_model: CompilationAwareNoise,
    ) -> Result<Self, CiPecError> {
        config.validate()?;
        Ok(Self {
            config,
            noise_model,
        })
    }

    /// Compute the logical error rate after QEC encoding.
    ///
    /// Uses the standard surface code scaling:
    ///   p_L = alpha * (p / p_th)^((d+1)/2)
    ///
    /// where alpha ~ 0.01, p_th ~ 0.01 (threshold), d = code distance.
    pub fn logical_error_rate(&self) -> f64 {
        let p = self.config.physical_error_rate;
        let d = self.config.code_distance;
        let p_th = 0.01; // Surface code threshold
        let alpha = 0.01; // Prefactor: keep d=3 below physical error in typical regimes.

        let exponent = (d + 1) as f64 / 2.0;
        alpha * (p / p_th).powf(exponent)
    }

    /// Compute the effective noise rate including compilation overhead.
    ///
    /// The logical noise is further increased by compilation-induced noise
    /// that is not fully protected by QEC (e.g., T-gate injection errors).
    pub fn effective_noise_rate(&self) -> f64 {
        let logical = self.logical_error_rate();
        let t_noise = self.noise_model.t_gate_noise;

        // T-gate injection errors are at the logical level and are only weakly
        // reduced by encoding, so keep them as an additive residual term.
        // Clifford and routing noise are protected by code distance; model that
        // by converting them to an effective per-round rate before threshold scaling.
        let protected_noise = self.noise_model.clifford_noise + self.noise_model.routing_noise;
        let p_th = 0.01;
        let alpha = 0.01;
        // Effective round-level contribution from compilation noise.
        let protected_round = protected_noise / 10.0;
        let protected_logical =
            alpha * (protected_round / p_th).powf((self.config.code_distance + 1) as f64 / 2.0);

        (logical + t_noise + protected_logical).min(0.749)
    }

    /// PEC overhead at the logical level.
    ///
    /// Because QEC suppresses the noise exponentially, the PEC overhead
    /// per layer is much smaller than bare PEC.
    pub fn logical_pec_overhead(&self) -> Result<f64, CiPecError> {
        let p = self.effective_noise_rate();
        if p >= 0.75 {
            return Err(CiPecError::NonInvertible(
                "Effective noise rate >= 0.75, PEC not viable".into(),
            ));
        }
        let qpd = QpdDecomposition::from_depolarizing(p)?;
        Ok(qpd.overhead)
    }

    /// Total overhead for a circuit of given depth.
    pub fn total_overhead(&self, depth: usize) -> Result<f64, CiPecError> {
        let per_layer = self.logical_pec_overhead()?;
        let total = per_layer.powi(depth as i32);
        if total > self.config.max_overhead {
            return Err(CiPecError::OverheadExceeded {
                computed: total,
                max_allowed: self.config.max_overhead,
            });
        }
        Ok(total)
    }

    /// Compare bare PEC overhead vs FT-PEC overhead for a given depth.
    ///
    /// Returns (bare_overhead, ft_pec_overhead, reduction_factor).
    pub fn overhead_comparison(&self, depth: usize) -> Result<(f64, f64, f64), CiPecError> {
        // Bare PEC on physical noise
        let bare_p = self.noise_model.total_noise().min(0.749);
        let bare_qpd = QpdDecomposition::from_depolarizing(bare_p)?;
        let bare_overhead = bare_qpd.overhead.powi(depth as i32);

        // FT-PEC on logical noise
        let ft_overhead = self.total_overhead(depth).unwrap_or(f64::INFINITY);

        let reduction = if ft_overhead > 0.0 && ft_overhead.is_finite() {
            bare_overhead / ft_overhead
        } else {
            f64::INFINITY
        };

        Ok((bare_overhead, ft_overhead, reduction))
    }

    /// Compute the QEC qubit overhead for the current code distance.
    ///
    /// Surface code requires ~ 2 * d^2 physical qubits per logical qubit.
    pub fn qec_qubit_overhead(&self, num_logical_qubits: usize) -> usize {
        let d = self.config.code_distance;
        num_logical_qubits * 2 * d * d
    }
}

// ============================================================
// ERROR BUDGET OPTIMIZER
// ============================================================

/// Allocation of the error budget across three axes.
#[derive(Debug, Clone)]
pub struct BudgetAllocation {
    /// Fraction of budget allocated to physical noise (hardware improvement).
    pub physical_fraction: f64,
    /// Fraction allocated to compilation noise (synthesis optimization).
    pub compilation_fraction: f64,
    /// Fraction allocated to PEC sampling overhead.
    pub pec_fraction: f64,
    /// Resulting total overhead at this allocation.
    pub total_overhead: f64,
    /// Optimal code distance for this allocation.
    pub optimal_distance: usize,
    /// Number of PEC samples needed.
    pub pec_samples_needed: usize,
}

/// Point on the Pareto frontier: QEC overhead vs PEC overhead trade-off.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    /// Code distance.
    pub code_distance: usize,
    /// QEC qubit overhead (physical qubits per logical qubit).
    pub qec_overhead: usize,
    /// PEC sampling overhead (gamma^L).
    pub pec_overhead: f64,
    /// Total resource cost (qec_overhead * pec_overhead).
    pub total_cost: f64,
}

/// Optimizer for distributing the error budget across QEC, compilation,
/// and PEC to minimize total resource cost.
pub struct ErrorBudgetOptimizer {
    /// Physical error rate.
    physical_error_rate: f64,
    /// Compilation strategy.
    strategy: CompilationStrategy,
    /// Circuit depth.
    circuit_depth: usize,
    /// Number of logical qubits.
    num_logical_qubits: usize,
}

impl ErrorBudgetOptimizer {
    /// Create a new optimizer.
    pub fn new(
        physical_error_rate: f64,
        strategy: CompilationStrategy,
        circuit_depth: usize,
        num_logical_qubits: usize,
    ) -> Self {
        Self {
            physical_error_rate,
            strategy,
            circuit_depth,
            num_logical_qubits,
        }
    }

    /// Compute the Pareto frontier of QEC vs PEC trade-offs.
    ///
    /// Sweeps code distances from 1 to max_distance (odd only) and
    /// computes the resulting PEC overhead at each point.
    pub fn pareto_frontier(&self, max_distance: usize) -> Vec<ParetoPoint> {
        let mut points = Vec::new();

        for d in (1..=max_distance).step_by(2) {
            let config = FtPecConfig {
                code_distance: d,
                physical_error_rate: self.physical_error_rate,
                compilation_strategy: self.strategy,
                max_overhead: 1e30,
                num_samples: 10_000,
                distillation_fidelity: 0.9999,
            };

            if let Ok(ft_pec) = FtPec::new(config) {
                let qec_overhead = ft_pec.qec_qubit_overhead(self.num_logical_qubits);
                let pec_overhead = ft_pec
                    .total_overhead(self.circuit_depth)
                    .unwrap_or(f64::INFINITY);

                if pec_overhead.is_finite() {
                    let total_cost = qec_overhead as f64 * pec_overhead;
                    points.push(ParetoPoint {
                        code_distance: d,
                        qec_overhead,
                        pec_overhead,
                        total_cost,
                    });
                }
            }
        }

        points
    }

    /// Find the optimal code distance that minimizes total cost.
    ///
    /// Total cost = QEC qubit overhead * PEC sampling overhead.
    pub fn optimize(&self) -> Result<BudgetAllocation, CiPecError> {
        let frontier = self.pareto_frontier(21);

        if frontier.is_empty() {
            return Err(CiPecError::OptimizationFailed(
                "No viable point on Pareto frontier".into(),
            ));
        }

        // Find the minimum total cost point
        let best = frontier
            .iter()
            .min_by(|a, b| a.total_cost.partial_cmp(&b.total_cost).unwrap())
            .unwrap();

        // Compute noise breakdown at optimal distance
        let noise = CompilationAwareNoise::for_strategy(
            self.strategy,
            self.physical_error_rate,
            self.circuit_depth,
        );
        let total = noise.total_noise().max(1e-15);

        let z = 1.96_f64;
        let epsilon = 0.01;
        let pec_samples = (best.pec_overhead * best.pec_overhead * z * z / (epsilon * epsilon))
            .ceil() as usize;

        Ok(BudgetAllocation {
            physical_fraction: noise.physical_noise / total,
            compilation_fraction: noise.compilation_fraction(),
            pec_fraction: 1.0 - noise.compilation_fraction() - noise.physical_noise / total,
            total_overhead: best.total_cost,
            optimal_distance: best.code_distance,
            pec_samples_needed: pec_samples,
        })
    }

    /// Find the break-even depth where full QEC becomes cheaper than FT-PEC.
    ///
    /// As circuit depth increases, PEC overhead grows exponentially while
    /// QEC overhead is constant. At some depth, pure QEC (high distance,
    /// no PEC) wins. This finds that crossover.
    pub fn breakeven_depth(&self, high_distance: usize) -> Option<usize> {
        // Cost of pure QEC at high distance (no PEC needed if logical error is negligible)
        let high_config = FtPecConfig {
            code_distance: high_distance,
            physical_error_rate: self.physical_error_rate,
            compilation_strategy: self.strategy,
            max_overhead: 1e30,
            num_samples: 10_000,
            distillation_fidelity: 0.9999,
        };
        let high_ft_pec = FtPec::new(high_config).ok()?;
        let qec_only_cost =
            high_ft_pec.qec_qubit_overhead(self.num_logical_qubits) as f64;

        // Scan depths to find crossover
        for depth in 1..1000 {
            let low_config = FtPecConfig {
                code_distance: 3,
                physical_error_rate: self.physical_error_rate,
                compilation_strategy: self.strategy,
                max_overhead: 1e30,
                num_samples: 10_000,
                distillation_fidelity: 0.9999,
            };

            if let Ok(low_ft) = FtPec::new(low_config) {
                let low_qec = low_ft.qec_qubit_overhead(self.num_logical_qubits) as f64;
                let low_pec = low_ft.total_overhead(depth).unwrap_or(f64::INFINITY);

                if low_qec * low_pec > qec_only_cost {
                    return Some(depth);
                }
            }
        }

        None
    }
}

// ============================================================
// TWIRLED PEC
// ============================================================

/// Combined Pauli twirling + PEC engine.
///
/// Twirling converts coherent errors to stochastic Pauli noise, which
/// PEC can cancel more efficiently (lower overhead). The key insight is
/// that coherent errors have PEC overhead proportional to the diamond
/// norm, while stochastic errors have overhead proportional to the
/// average infidelity -- often significantly smaller.
pub struct TwirledPec {
    /// Number of twirling instances to average over.
    pub num_twirls: usize,
    /// PEC configuration.
    pub pec_config: PecConfig,
    /// Coherent noise rate (before twirling).
    pub coherent_rate: f64,
    /// Stochastic rate after twirling (typically smaller).
    pub stochastic_rate: f64,
}

impl TwirledPec {
    /// Create a new TwirledPec instance.
    ///
    /// # Arguments
    /// * `coherent_rate` - Total coherent error rate before twirling
    /// * `num_twirls` - Number of randomized compilations to average
    /// * `pec_config` - PEC configuration
    pub fn new(
        coherent_rate: f64,
        num_twirls: usize,
        pec_config: PecConfig,
    ) -> Result<Self, CiPecError> {
        if coherent_rate < 0.0 || coherent_rate >= 0.75 {
            return Err(CiPecError::InvalidNoiseParameter(format!(
                "coherent_rate must be in [0, 0.75), got {}",
                coherent_rate
            )));
        }
        pec_config.validate()?;

        // After twirling, the effective stochastic rate is bounded by
        // the average gate infidelity rather than the diamond norm.
        // For a unitary error U = exp(-i * epsilon * H), the diamond
        // norm is ~ 2*epsilon but the average infidelity is ~ epsilon^2.
        // Twirling converts the former to the latter.
        let stochastic_rate = coherent_rate * coherent_rate;

        Ok(Self {
            num_twirls,
            pec_config,
            coherent_rate,
            stochastic_rate,
        })
    }

    /// PEC overhead without twirling (coherent noise).
    pub fn overhead_without_twirling(&self) -> Result<f64, CiPecError> {
        QpdDecomposition::from_depolarizing(self.coherent_rate).map(|q| q.overhead)
    }

    /// PEC overhead with twirling (stochastic noise).
    pub fn overhead_with_twirling(&self) -> Result<f64, CiPecError> {
        let rate = self.stochastic_rate.min(0.749);
        QpdDecomposition::from_depolarizing(rate).map(|q| q.overhead)
    }

    /// Overhead reduction factor from twirling.
    pub fn twirling_reduction(&self) -> Result<f64, CiPecError> {
        let without = self.overhead_without_twirling()?;
        let with = self.overhead_with_twirling()?;
        Ok(without / with)
    }

    /// Total overhead for a circuit of given depth, with twirling.
    pub fn total_overhead(&self, depth: usize) -> Result<f64, CiPecError> {
        let per_layer = self.overhead_with_twirling()?;
        Ok(per_layer.powi(depth as i32))
    }

    /// Compute the number of total circuit executions needed.
    ///
    /// Total shots = num_twirls * pec_samples_per_twirl.
    pub fn total_shots(&self, depth: usize, precision: f64) -> Result<usize, CiPecError> {
        let gamma = self.total_overhead(depth)?;
        let z = 1.96_f64;
        let pec_samples = (gamma * gamma * z * z / (precision * precision)).ceil() as usize;
        Ok(self.num_twirls * pec_samples)
    }
}

// ============================================================
// LAYER MITIGATION PLAN
// ============================================================

/// Mitigation strategy for a single circuit layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerStrategy {
    /// No mitigation: leave the layer as-is (low error).
    None,
    /// Apply PEC to this layer.
    Pec,
    /// Apply twirled PEC to this layer.
    TwirledPec,
    /// This layer is QEC-protected (no PEC needed).
    QecProtected,
}

impl fmt::Display for LayerStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerStrategy::None => write!(f, "None"),
            LayerStrategy::Pec => write!(f, "PEC"),
            LayerStrategy::TwirledPec => write!(f, "TwirledPEC"),
            LayerStrategy::QecProtected => write!(f, "QEC"),
        }
    }
}

/// Per-layer mitigation plan for a circuit.
///
/// Rather than applying uniform PEC to all gates, this plan targets
/// mitigation at the layers with the highest error rates, leaving
/// low-error layers unmitigated to reduce total overhead.
#[derive(Debug, Clone)]
pub struct LayerMitigationPlan {
    /// Strategy for each layer.
    pub strategies: Vec<LayerStrategy>,
    /// Error rate for each layer (used to determine strategy).
    pub layer_errors: Vec<f64>,
    /// Threshold below which no mitigation is applied.
    pub mitigation_threshold: f64,
}

impl LayerMitigationPlan {
    /// Create a mitigation plan from per-layer error rates.
    ///
    /// Layers with error rate above the threshold get PEC;
    /// those below are left unmitigated.
    pub fn from_errors(layer_errors: Vec<f64>, threshold: f64) -> Self {
        let strategies = layer_errors
            .iter()
            .map(|&err| {
                if err < threshold {
                    LayerStrategy::None
                } else {
                    LayerStrategy::Pec
                }
            })
            .collect();

        Self {
            strategies,
            layer_errors,
            mitigation_threshold: threshold,
        }
    }

    /// Build a mitigation plan from real FT compilation artifacts.
    pub fn from_ft_compilation(
        compilation: &FTCompilationResult,
        physical_error_rate: f64,
        distillation_fidelity: f64,
        threshold: f64,
    ) -> Self {
        let layer_errors = layer_errors_from_ft_compilation(
            compilation,
            physical_error_rate,
            distillation_fidelity,
        );
        Self::from_errors(layer_errors, threshold)
    }

    /// Create a plan that applies PEC only to the top-k highest-error layers.
    pub fn top_k(layer_errors: Vec<f64>, k: usize) -> Self {
        let mut indexed: Vec<(usize, f64)> =
            layer_errors.iter().enumerate().map(|(i, &e)| (i, e)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut strategies = vec![LayerStrategy::None; layer_errors.len()];
        for (i, &(idx, _)) in indexed.iter().enumerate() {
            if i < k {
                strategies[idx] = LayerStrategy::Pec;
            }
        }

        let threshold = if k < indexed.len() {
            indexed[k].1
        } else {
            0.0
        };

        Self {
            strategies,
            layer_errors,
            mitigation_threshold: threshold,
        }
    }

    /// Compute the total PEC overhead for this plan.
    ///
    /// Only layers marked as PEC contribute to the sampling overhead.
    pub fn total_overhead(&self) -> Result<f64, CiPecError> {
        let mut overhead = 1.0;
        for (strategy, &err) in self.strategies.iter().zip(self.layer_errors.iter()) {
            match strategy {
                LayerStrategy::Pec | LayerStrategy::TwirledPec => {
                    let p = err.min(0.749);
                    if p > 1e-15 {
                        let qpd = QpdDecomposition::from_depolarizing(p)?;
                        overhead *= qpd.overhead;
                    }
                }
                _ => {}
            }
        }
        Ok(overhead)
    }

    /// Number of layers that are mitigated.
    pub fn num_mitigated(&self) -> usize {
        self.strategies
            .iter()
            .filter(|s| matches!(s, LayerStrategy::Pec | LayerStrategy::TwirledPec))
            .count()
    }

    /// Number of layers left unmitigated.
    pub fn num_unmitigated(&self) -> usize {
        self.strategies.len() - self.num_mitigated()
    }
}

/// Wrapper around a circuit with its associated mitigation strategy.
#[derive(Debug, Clone)]
pub struct MitigatedCircuit {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Circuit depth (number of layers).
    pub depth: usize,
    /// Per-layer error rates.
    pub layer_errors: Vec<f64>,
    /// The mitigation plan.
    pub plan: LayerMitigationPlan,
}

impl MitigatedCircuit {
    /// Create a new mitigated circuit with automatic plan generation.
    pub fn new(num_qubits: usize, layer_errors: Vec<f64>, threshold: f64) -> Self {
        let depth = layer_errors.len();
        let plan = LayerMitigationPlan::from_errors(layer_errors.clone(), threshold);
        Self {
            num_qubits,
            depth,
            layer_errors,
            plan,
        }
    }

    /// Create a mitigated circuit from FT compilation artifacts.
    pub fn from_ft_compilation(
        num_qubits: usize,
        compilation: &FTCompilationResult,
        physical_error_rate: f64,
        distillation_fidelity: f64,
        threshold: f64,
    ) -> Self {
        let layer_errors = layer_errors_from_ft_compilation(
            compilation,
            physical_error_rate,
            distillation_fidelity,
        );
        let plan = LayerMitigationPlan::from_errors(layer_errors.clone(), threshold);
        let depth = layer_errors.len();
        Self {
            num_qubits,
            depth,
            layer_errors,
            plan,
        }
    }

    /// Total PEC overhead for this circuit.
    pub fn total_overhead(&self) -> Result<f64, CiPecError> {
        self.plan.total_overhead()
    }
}

// ============================================================
// PEC ANALYSIS
// ============================================================

/// Comparison metrics between different mitigation strategies.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Strategy name.
    pub strategy: String,
    /// Total sampling overhead (gamma^L).
    pub sampling_overhead: f64,
    /// Expected variance of the estimator.
    pub expected_variance: f64,
    /// 95% confidence interval half-width.
    pub confidence_interval: f64,
    /// Number of samples needed for 1% precision.
    pub samples_for_1pct: usize,
    /// Number of physical qubits needed.
    pub physical_qubits: usize,
    /// Total cost metric (qubits * samples).
    pub total_cost: f64,
}

/// Analysis engine for comparing mitigation strategies.
pub struct PecAnalysis {
    /// Physical error rate.
    physical_error_rate: f64,
    /// Circuit depth.
    circuit_depth: usize,
    /// Number of logical qubits.
    num_logical_qubits: usize,
}

impl PecAnalysis {
    /// Create a new analysis engine.
    pub fn new(
        physical_error_rate: f64,
        circuit_depth: usize,
        num_logical_qubits: usize,
    ) -> Self {
        Self {
            physical_error_rate,
            circuit_depth,
            num_logical_qubits,
        }
    }

    /// Analyze the "no mitigation" baseline.
    fn analyze_no_mitigation(&self) -> AnalysisResult {
        AnalysisResult {
            strategy: "No mitigation".to_string(),
            sampling_overhead: 1.0,
            expected_variance: self.physical_error_rate * self.circuit_depth as f64,
            confidence_interval: 1.96
                * (self.physical_error_rate * self.circuit_depth as f64).sqrt(),
            samples_for_1pct: 1,
            physical_qubits: self.num_logical_qubits,
            total_cost: self.num_logical_qubits as f64,
        }
    }

    /// Analyze bare PEC (no QEC).
    fn analyze_bare_pec(&self) -> AnalysisResult {
        let p = self.physical_error_rate.min(0.749);
        let qpd = QpdDecomposition::from_depolarizing(p).unwrap_or(QpdDecomposition {
            terms: vec![],
            overhead: f64::INFINITY,
            sampling_probs: vec![],
        });

        let gamma_total = qpd.overhead.powi(self.circuit_depth as i32);
        let variance = gamma_total * gamma_total;
        let z = 1.96_f64;
        let epsilon = 0.01;
        let samples = (gamma_total * gamma_total * z * z / (epsilon * epsilon)).ceil() as usize;

        AnalysisResult {
            strategy: "Bare PEC".to_string(),
            sampling_overhead: gamma_total,
            expected_variance: variance,
            confidence_interval: z * variance.sqrt(),
            samples_for_1pct: samples,
            physical_qubits: self.num_logical_qubits,
            total_cost: self.num_logical_qubits as f64 * samples as f64,
        }
    }

    /// Analyze FT-PEC at a given code distance.
    fn analyze_ft_pec(&self, code_distance: usize) -> AnalysisResult {
        let config = FtPecConfig {
            code_distance,
            physical_error_rate: self.physical_error_rate,
            compilation_strategy: CompilationStrategy::CliffordT,
            max_overhead: 1e30,
            num_samples: 10_000,
            distillation_fidelity: 0.9999,
        };

        let ft_pec = match FtPec::new(config) {
            Ok(f) => f,
            Err(_) => {
                return AnalysisResult {
                    strategy: format!("FT-PEC (d={})", code_distance),
                    sampling_overhead: f64::INFINITY,
                    expected_variance: f64::INFINITY,
                    confidence_interval: f64::INFINITY,
                    samples_for_1pct: usize::MAX,
                    physical_qubits: 0,
                    total_cost: f64::INFINITY,
                }
            }
        };

        let gamma_total = ft_pec
            .total_overhead(self.circuit_depth)
            .unwrap_or(f64::INFINITY);
        let phys_qubits = ft_pec.qec_qubit_overhead(self.num_logical_qubits);
        let variance = gamma_total * gamma_total;
        let z = 1.96_f64;
        let epsilon = 0.01;
        let samples = if gamma_total.is_finite() {
            (gamma_total * gamma_total * z * z / (epsilon * epsilon)).ceil() as usize
        } else {
            usize::MAX
        };

        AnalysisResult {
            strategy: format!("FT-PEC (d={})", code_distance),
            sampling_overhead: gamma_total,
            expected_variance: variance,
            confidence_interval: z * variance.sqrt(),
            samples_for_1pct: samples,
            physical_qubits: phys_qubits,
            total_cost: phys_qubits as f64 * samples as f64,
        }
    }

    /// Analyze full QEC (no PEC needed, assumes negligible logical error).
    fn analyze_full_qec(&self, code_distance: usize) -> AnalysisResult {
        let d = code_distance;
        let phys_qubits = self.num_logical_qubits * 2 * d * d;

        // Logical error rate
        let p = self.physical_error_rate;
        let p_logical = 0.1 * (p / 0.01).powf((d + 1) as f64 / 2.0);

        AnalysisResult {
            strategy: format!("Full QEC (d={})", code_distance),
            sampling_overhead: 1.0,
            expected_variance: p_logical * self.circuit_depth as f64,
            confidence_interval: 1.96
                * (p_logical * self.circuit_depth as f64).sqrt(),
            samples_for_1pct: 1,
            physical_qubits: phys_qubits,
            total_cost: phys_qubits as f64,
        }
    }

    /// Run the full comparison: no mitigation, bare PEC, FT-PEC, full QEC.
    pub fn compare_all(&self) -> Vec<AnalysisResult> {
        vec![
            self.analyze_no_mitigation(),
            self.analyze_bare_pec(),
            self.analyze_ft_pec(3),
            self.analyze_ft_pec(5),
            self.analyze_full_qec(7),
            self.analyze_full_qec(11),
        ]
    }

    /// Find the break-even depth where full QEC becomes cheaper than FT-PEC.
    pub fn breakeven_depth(&self, _ft_distance: usize, qec_distance: usize) -> Option<usize> {
        let optimizer = ErrorBudgetOptimizer::new(
            self.physical_error_rate,
            CompilationStrategy::CliffordT,
            1,
            self.num_logical_qubits,
        );
        optimizer.breakeven_depth(qec_distance)
    }
}

// ============================================================
// DEMO FUNCTION
// ============================================================

/// Demonstration of Compilation-Informed PEC capabilities.
///
/// Shows noise decomposition, overhead comparison, Pareto optimization,
/// twirling integration, and strategy analysis.
pub fn demo() {
    println!("=== Compilation-Informed PEC (CI-PEC) Demo ===\n");

    // 1. Compilation-aware noise model
    let physical_rate = 1e-3;
    let noise = CompilationAwareNoise::new(
        physical_rate,
        20,     // T gates
        50,     // Clifford depth
        10,     // SWAPs
        0.9999, // distillation fidelity
        CompilationStrategy::CliffordT,
    );

    println!("1. Compilation-Aware Noise Model:");
    println!("   {}", noise);
    println!("   Physical:    {:.2e}", noise.physical_noise);
    println!("   T-gate:      {:.2e}", noise.t_gate_noise);
    println!("   Clifford:    {:.2e}", noise.clifford_noise);
    println!("   Routing:     {:.2e}", noise.routing_noise);
    println!("   Total:       {:.2e}", noise.total_noise());
    println!(
        "   Compilation fraction: {:.1}%",
        noise.compilation_fraction() * 100.0
    );
    println!();

    // 2. FT-PEC overhead comparison
    println!("2. FT-PEC Overhead Comparison (depth=100):");
    for d in [3, 5, 7].iter() {
        let config = FtPecConfig {
            code_distance: *d,
            physical_error_rate: physical_rate,
            compilation_strategy: CompilationStrategy::CliffordT,
            max_overhead: 1e30,
            num_samples: 10_000,
            distillation_fidelity: 0.9999,
        };

        if let Ok(ft_pec) = FtPec::new(config) {
            let logical_rate = ft_pec.logical_error_rate();
            let effective_rate = ft_pec.effective_noise_rate();
            let overhead = ft_pec.total_overhead(100).unwrap_or(f64::INFINITY);
            println!(
                "   d={}: logical_err={:.2e}, effective_err={:.2e}, overhead={:.2e}",
                d, logical_rate, effective_rate, overhead
            );
        }
    }
    println!();

    // 3. Pareto frontier
    println!("3. Pareto Frontier (QEC vs PEC overhead):");
    let optimizer = ErrorBudgetOptimizer::new(physical_rate, CompilationStrategy::CliffordT, 50, 4);
    let frontier = optimizer.pareto_frontier(11);
    for point in &frontier {
        println!(
            "   d={}: QEC qubits={}, PEC overhead={:.2e}, total_cost={:.2e}",
            point.code_distance, point.qec_overhead, point.pec_overhead, point.total_cost
        );
    }
    println!();

    // 4. Budget optimization
    println!("4. Error Budget Optimization:");
    match optimizer.optimize() {
        Ok(alloc) => {
            println!("   Optimal code distance: {}", alloc.optimal_distance);
            println!("   Physical fraction:     {:.1}%", alloc.physical_fraction * 100.0);
            println!(
                "   Compilation fraction:  {:.1}%",
                alloc.compilation_fraction * 100.0
            );
            println!("   PEC fraction:          {:.1}%", alloc.pec_fraction * 100.0);
            println!("   PEC samples needed:    {}", alloc.pec_samples_needed);
        }
        Err(e) => println!("   Optimization failed: {}", e),
    }
    println!();

    // 5. Twirled PEC
    println!("5. Twirled PEC Overhead Reduction:");
    let pec_config = PecConfig {
        noise_rate: 0.05,
        ..Default::default()
    };
    if let Ok(twirled) = TwirledPec::new(0.05, 32, pec_config) {
        let without = twirled.overhead_without_twirling().unwrap_or(f64::INFINITY);
        let with = twirled.overhead_with_twirling().unwrap_or(f64::INFINITY);
        let reduction = twirled.twirling_reduction().unwrap_or(0.0);
        println!("   Without twirling: gamma = {:.4}", without);
        println!("   With twirling:    gamma = {:.4}", with);
        println!("   Reduction factor: {:.2}x", reduction);
    }
    println!();

    // 6. Layer mitigation plan
    println!("6. Layer Mitigation Plan:");
    let layer_errors = vec![0.001, 0.05, 0.002, 0.08, 0.001, 0.03, 0.001, 0.06];
    let plan = LayerMitigationPlan::top_k(layer_errors.clone(), 3);
    for (i, (strategy, &err)) in plan.strategies.iter().zip(layer_errors.iter()).enumerate() {
        println!("   Layer {}: err={:.3}, strategy={}", i, err, strategy);
    }
    println!(
        "   Total overhead (targeted): {:.4}",
        plan.total_overhead().unwrap_or(f64::INFINITY)
    );
    let uniform = LayerMitigationPlan::from_errors(layer_errors.clone(), 0.0);
    println!(
        "   Total overhead (uniform):  {:.4}",
        uniform.total_overhead().unwrap_or(f64::INFINITY)
    );
    println!();

    // 7. Full comparison
    println!("7. Strategy Comparison (p={:.0e}, depth=50, 4 qubits):", physical_rate);
    let analysis = PecAnalysis::new(physical_rate, 50, 4);
    let results = analysis.compare_all();
    println!(
        "   {:25} {:>14} {:>14} {:>14}",
        "Strategy", "Overhead", "Samples@1%", "Phys Qubits"
    );
    println!("   {}", "-".repeat(70));
    for r in &results {
        let overhead_str = if r.sampling_overhead.is_finite() && r.sampling_overhead < 1e15 {
            format!("{:.2e}", r.sampling_overhead)
        } else {
            "INF".to_string()
        };
        let samples_str = if r.samples_for_1pct < usize::MAX {
            format!("{}", r.samples_for_1pct)
        } else {
            "INF".to_string()
        };
        println!(
            "   {:25} {:>14} {:>14} {:>14}",
            r.strategy, overhead_str, samples_str, r.physical_qubits
        );
    }
    println!();

    println!("=== CI-PEC Demo Complete ===");
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use crate::ft_compilation::{compile_to_clifford_t, LogicalGate};

    const TOL: f64 = 1e-10;

    // --------------------------------------------------------
    // QPD decomposition tests
    // --------------------------------------------------------

    #[test]
    fn test_qpd_noiseless() {
        let qpd = QpdDecomposition::from_depolarizing(0.0).unwrap();
        assert!((qpd.overhead - 1.0).abs() < TOL);
        assert_eq!(qpd.terms.len(), 1);
        assert!(qpd.is_valid());
    }

    #[test]
    fn test_qpd_low_noise() {
        let qpd = QpdDecomposition::from_depolarizing(0.01).unwrap();
        // For p=0.01: lambda = 1 - 4/300 = 0.9867
        // gamma = |c_I| + 3*|c_xyz|
        assert!(qpd.overhead > 1.0);
        assert!(qpd.overhead < 1.1); // Small overhead for small noise
        assert!(qpd.is_valid());
    }

    #[test]
    fn test_qpd_moderate_noise() {
        let qpd = QpdDecomposition::from_depolarizing(0.1).unwrap();
        assert!(qpd.overhead > 1.0);
        assert_eq!(qpd.terms.len(), 4);
        assert!(qpd.is_valid());
    }

    #[test]
    fn test_qpd_high_noise_boundary() {
        // Should fail at p >= 0.75 (channel non-invertible)
        assert!(QpdDecomposition::from_depolarizing(0.75).is_err());
        assert!(QpdDecomposition::from_depolarizing(0.8).is_err());
    }

    #[test]
    fn test_qpd_negative_noise() {
        assert!(QpdDecomposition::from_depolarizing(-0.01).is_err());
    }

    #[test]
    fn test_qpd_coefficients_sum() {
        // For a valid QPD, coefficients should sum to 1 (identity channel)
        let qpd = QpdDecomposition::from_depolarizing(0.05).unwrap();
        let coeff_sum: f64 = qpd.terms.iter().map(|t| t.coefficient).sum();
        assert!(
            (coeff_sum - 1.0).abs() < 1e-8,
            "QPD coefficients should sum to 1, got {}",
            coeff_sum
        );
    }

    // --------------------------------------------------------
    // PEC engine tests
    // --------------------------------------------------------

    #[test]
    fn test_pec_engine_creation() {
        let config = PecConfig::default();
        let engine = PecEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_pec_engine_invalid_config() {
        let config = PecConfig {
            noise_rate: 1.5,
            ..Default::default()
        };
        assert!(PecEngine::new(config).is_err());
    }

    #[test]
    fn test_pec_total_overhead_scales_with_depth() {
        let config = PecConfig {
            noise_rate: 0.01,
            ..Default::default()
        };
        let engine = PecEngine::new(config).unwrap();

        let overhead_10 = engine.total_overhead(10).unwrap();
        let overhead_20 = engine.total_overhead(20).unwrap();

        // Overhead at depth 20 should be (overhead per layer)^2 of depth 10
        let per_layer = engine.decompose_gate(0.01).unwrap().overhead;
        let expected_20 = per_layer.powi(20);
        assert!((overhead_20 - expected_20).abs() / expected_20 < 1e-8);
        assert!(overhead_20 > overhead_10);
    }

    #[test]
    fn test_pec_sampling_instance() {
        let config = PecConfig {
            noise_rate: 0.05,
            ..Default::default()
        };
        let engine = PecEngine::new(config).unwrap();

        let qpd = engine.decompose_gate(0.05).unwrap();
        let qpds = vec![qpd; 5];

        let mut rng = StdRng::seed_from_u64(42);
        let (indices, sign) = engine.sample_instance(&qpds, &mut rng);

        assert_eq!(indices.len(), 5);
        for &idx in &indices {
            assert!(idx < 4, "Pauli index must be in [0, 3]");
        }
        // Sign can be positive or negative but should be non-zero
        assert!(sign.abs() > 0.0);
    }

    #[test]
    fn test_pec_expectation_estimation() {
        // With no noise, signed samples should average to the true value
        let samples: Vec<(f64, f64)> = vec![
            (0.8, 1.0),
            (0.9, 1.0),
            (0.85, 1.0),
            (0.82, 1.0),
            (0.88, 1.0),
        ];
        let (mean, stderr) = PecEngine::estimate_expectation(&samples);
        assert!((mean - 0.85).abs() < 0.1);
        assert!(stderr < 0.1);
    }

    // --------------------------------------------------------
    // Compilation-aware noise tests
    // --------------------------------------------------------

    #[test]
    fn test_compilation_noise_components() {
        let noise = CompilationAwareNoise::new(
            1e-3,   // physical
            10,     // T gates
            20,     // Clifford depth
            5,      // SWAPs
            0.999,  // distillation fidelity
            CompilationStrategy::CliffordT,
        );

        assert!(noise.physical_noise > 0.0);
        assert!(noise.t_gate_noise > 0.0);
        assert!(noise.clifford_noise > 0.0);
        assert!(noise.routing_noise > 0.0);
        assert!(noise.total_noise() > noise.physical_noise);
    }

    #[test]
    fn test_compilation_fraction() {
        let noise = CompilationAwareNoise::new(
            1e-3, 10, 20, 5, 0.999, CompilationStrategy::CliffordT,
        );

        let frac = noise.compilation_fraction();
        assert!(frac > 0.0 && frac <= 1.0);
        // Compilation noise should dominate for typical parameters
        assert!(frac > 0.5, "Compilation should dominate, got {}", frac);
    }

    #[test]
    fn test_compilation_noise_custom_zero() {
        // Custom strategy with zero compilation noise
        let noise = CompilationAwareNoise::new(
            1e-3, 0, 0, 0, 1.0, CompilationStrategy::Custom,
        );

        assert!((noise.compilation_fraction()).abs() < TOL);
        assert!((noise.total_noise() - 1e-3).abs() < TOL);
    }

    #[test]
    fn test_compilation_noise_from_ft_result() {
        let circuit = vec![
            LogicalGate::T(0),
            LogicalGate::T(3),
            LogicalGate::CNOT(0, 3),
            LogicalGate::H(1),
        ];
        let compiled = compile_to_clifford_t(&circuit).unwrap();

        let noise = CompilationAwareNoise::from_ft_compilation_result(
            1e-3,
            &compiled,
            0.9999,
            CompilationStrategy::CliffordT,
        );

        assert_eq!(noise.t_gate_noise > 0.0, compiled.t_count > 0);
        assert!(noise.clifford_noise >= 0.0);
        assert!(
            noise.routing_noise > 0.0,
            "Expected routing noise from non-adjacent CNOT"
        );
    }

    #[test]
    fn test_strategy_noise_profiles() {
        let physical = 1e-3;
        let depth = 10;

        let direct = CompilationAwareNoise::for_strategy(
            CompilationStrategy::DirectSynthesis,
            physical,
            depth,
        );
        let ct = CompilationAwareNoise::for_strategy(
            CompilationStrategy::CliffordT,
            physical,
            depth,
        );
        let pf = CompilationAwareNoise::for_strategy(
            CompilationStrategy::PauliFrame,
            physical,
            depth,
        );

        // All should have the same physical noise
        assert!((direct.physical_noise - ct.physical_noise).abs() < TOL);
        assert!((ct.physical_noise - pf.physical_noise).abs() < TOL);

        // Clifford+T should have higher Clifford depth noise
        assert!(ct.clifford_noise > direct.clifford_noise);
    }

    // --------------------------------------------------------
    // FT-PEC tests
    // --------------------------------------------------------

    #[test]
    fn test_ft_pec_creation() {
        let config = FtPecConfig::default();
        let ft_pec = FtPec::new(config);
        assert!(ft_pec.is_ok());
    }

    #[test]
    fn test_ft_pec_invalid_distance() {
        let config = FtPecConfig {
            code_distance: 4, // even, invalid
            ..Default::default()
        };
        assert!(FtPec::new(config).is_err());
    }

    #[test]
    fn test_ft_pec_logical_error_suppression() {
        // Higher distance should suppress logical error
        let config_3 = FtPecConfig {
            code_distance: 3,
            physical_error_rate: 1e-3,
            ..Default::default()
        };
        let config_5 = FtPecConfig {
            code_distance: 5,
            physical_error_rate: 1e-3,
            ..Default::default()
        };

        let ft3 = FtPec::new(config_3).unwrap();
        let ft5 = FtPec::new(config_5).unwrap();

        assert!(
            ft5.logical_error_rate() < ft3.logical_error_rate(),
            "d=5 should have lower logical error than d=3"
        );
    }

    #[test]
    fn test_ft_pec_overhead_reduction() {
        let config = FtPecConfig {
            code_distance: 5,
            physical_error_rate: 1e-3,
            ..Default::default()
        };
        let ft_pec = FtPec::new(config).unwrap();

        let (bare, ft, reduction) = ft_pec.overhead_comparison(50).unwrap();
        assert!(
            ft < bare,
            "FT-PEC overhead ({}) should be less than bare PEC ({})",
            ft,
            bare
        );
        assert!(
            reduction > 1.0,
            "Reduction factor should be > 1, got {}",
            reduction
        );
    }

    #[test]
    fn test_ft_pec_qubit_overhead() {
        let config = FtPecConfig {
            code_distance: 5,
            ..Default::default()
        };
        let ft_pec = FtPec::new(config).unwrap();

        // d=5: 2*5^2 = 50 physical qubits per logical qubit
        assert_eq!(ft_pec.qec_qubit_overhead(1), 50);
        assert_eq!(ft_pec.qec_qubit_overhead(4), 200);
    }

    #[test]
    fn test_ft_pec_from_compilation_result() {
        let circuit = vec![LogicalGate::T(0), LogicalGate::CNOT(0, 2), LogicalGate::T(2)];
        let compiled = compile_to_clifford_t(&circuit).unwrap();

        let config = FtPecConfig {
            code_distance: 3,
            physical_error_rate: 1e-3,
            compilation_strategy: CompilationStrategy::CliffordT,
            max_overhead: 1e10,
            num_samples: 10_000,
            distillation_fidelity: 0.9999,
        };
        let ft_pec = FtPec::from_compilation_result(config, &compiled).unwrap();

        assert!(ft_pec.noise_model.t_gate_noise > 0.0);
        assert!(ft_pec.noise_model.routing_noise > 0.0);
    }

    // --------------------------------------------------------
    // Error budget optimizer tests
    // --------------------------------------------------------

    #[test]
    fn test_pareto_frontier() {
        let optimizer = ErrorBudgetOptimizer::new(
            1e-3,
            CompilationStrategy::CliffordT,
            50,
            4,
        );
        let frontier = optimizer.pareto_frontier(11);

        assert!(!frontier.is_empty());
        // QEC overhead should increase with distance
        for i in 1..frontier.len() {
            assert!(frontier[i].qec_overhead >= frontier[i - 1].qec_overhead);
        }
    }

    #[test]
    fn test_budget_optimization() {
        let optimizer = ErrorBudgetOptimizer::new(
            1e-3,
            CompilationStrategy::CliffordT,
            20,
            2,
        );
        let alloc = optimizer.optimize().unwrap();

        // Fractions should approximately sum to 1 (accounting for normalization)
        assert!(alloc.optimal_distance >= 1);
        assert!(alloc.optimal_distance % 2 == 1); // Must be odd
        assert!(alloc.pec_samples_needed > 0);
    }

    // --------------------------------------------------------
    // Twirled PEC tests
    // --------------------------------------------------------

    #[test]
    fn test_twirled_pec_creation() {
        let config = PecConfig::default();
        let twirled = TwirledPec::new(0.05, 32, config);
        assert!(twirled.is_ok());
    }

    #[test]
    fn test_twirled_pec_stochastic_rate() {
        let config = PecConfig::default();
        let twirled = TwirledPec::new(0.1, 32, config).unwrap();

        // Stochastic rate should be coherent_rate^2
        assert!(
            (twirled.stochastic_rate - 0.01).abs() < TOL,
            "Expected stochastic_rate = 0.01, got {}",
            twirled.stochastic_rate
        );
    }

    #[test]
    fn test_twirled_pec_overhead_reduction() {
        let config = PecConfig::default();
        let twirled = TwirledPec::new(0.1, 32, config).unwrap();

        let without = twirled.overhead_without_twirling().unwrap();
        let with = twirled.overhead_with_twirling().unwrap();

        assert!(
            with < without,
            "Twirled overhead ({}) should be less than untwirled ({})",
            with,
            without
        );
    }

    #[test]
    fn test_twirled_pec_invalid_rate() {
        let config = PecConfig::default();
        assert!(TwirledPec::new(0.8, 32, config).is_err());
    }

    // --------------------------------------------------------
    // Layer mitigation plan tests
    // --------------------------------------------------------

    #[test]
    fn test_layer_plan_threshold() {
        let errors = vec![0.001, 0.05, 0.002, 0.08, 0.001];
        let plan = LayerMitigationPlan::from_errors(errors, 0.01);

        assert_eq!(plan.strategies[0], LayerStrategy::None);
        assert_eq!(plan.strategies[1], LayerStrategy::Pec);
        assert_eq!(plan.strategies[2], LayerStrategy::None);
        assert_eq!(plan.strategies[3], LayerStrategy::Pec);
        assert_eq!(plan.strategies[4], LayerStrategy::None);
        assert_eq!(plan.num_mitigated(), 2);
        assert_eq!(plan.num_unmitigated(), 3);
    }

    #[test]
    fn test_layer_plan_top_k() {
        let errors = vec![0.01, 0.05, 0.02, 0.08, 0.03];
        let plan = LayerMitigationPlan::top_k(errors, 2);

        // Top-2 errors are at indices 3 (0.08) and 1 (0.05)
        assert_eq!(plan.strategies[3], LayerStrategy::Pec);
        assert_eq!(plan.strategies[1], LayerStrategy::Pec);
        assert_eq!(plan.num_mitigated(), 2);
    }

    #[test]
    fn test_layer_plan_overhead() {
        let errors = vec![0.001, 0.05, 0.002, 0.08, 0.001];
        let plan_targeted = LayerMitigationPlan::from_errors(errors.clone(), 0.01);
        let plan_uniform = LayerMitigationPlan::from_errors(errors, 0.0);

        let overhead_targeted = plan_targeted.total_overhead().unwrap();
        let overhead_uniform = plan_uniform.total_overhead().unwrap();

        assert!(
            overhead_targeted < overhead_uniform,
            "Targeted ({}) should be less than uniform ({})",
            overhead_targeted,
            overhead_uniform
        );
    }

    #[test]
    fn test_layer_errors_from_ft_compilation_artifacts() {
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::CNOT(0, 3), // forces routing estimate
            LogicalGate::T(3),
        ];
        let compiled = compile_to_clifford_t(&circuit).unwrap();
        let layer_errors = layer_errors_from_ft_compilation(&compiled, 1e-3, 0.9999);

        assert!(
            !layer_errors.is_empty(),
            "expected non-empty layer errors from compiled circuit"
        );
        assert!(
            layer_errors.iter().all(|e| *e >= 0.0 && *e < 0.75),
            "layer errors must stay in [0, 0.75)"
        );
        assert!(
            layer_errors.iter().any(|e| *e > 1e-3),
            "compiled artifacts should induce above-physical per-layer error"
        );
    }

    #[test]
    fn test_layer_plan_from_ft_compilation() {
        let circuit = vec![
            LogicalGate::T(0),
            LogicalGate::CNOT(0, 2),
            LogicalGate::H(1),
            LogicalGate::T(2),
        ];
        let compiled = compile_to_clifford_t(&circuit).unwrap();
        let plan = LayerMitigationPlan::from_ft_compilation(&compiled, 1e-3, 0.9999, 0.005);
        assert_eq!(plan.layer_errors.len(), plan.strategies.len());
        assert!(plan.layer_errors.len() > 0);
        assert!(
            plan.strategies.iter().any(|s| matches!(s, LayerStrategy::Pec)),
            "expected at least one mitigated layer from compilation-aware plan"
        );
    }

    // --------------------------------------------------------
    // Mitigated circuit tests
    // --------------------------------------------------------

    #[test]
    fn test_mitigated_circuit() {
        let errors = vec![0.01, 0.05, 0.02, 0.08, 0.03, 0.01, 0.06, 0.02];
        let circuit = MitigatedCircuit::new(4, errors, 0.03);

        assert_eq!(circuit.num_qubits, 4);
        assert_eq!(circuit.depth, 8);
        assert!(circuit.total_overhead().unwrap() > 1.0);
    }

    // --------------------------------------------------------
    // PEC analysis tests
    // --------------------------------------------------------

    #[test]
    fn test_analysis_compare_all() {
        let analysis = PecAnalysis::new(1e-3, 20, 2);
        let results = analysis.compare_all();

        assert_eq!(results.len(), 6);

        // No mitigation should have overhead = 1
        assert!((results[0].sampling_overhead - 1.0).abs() < TOL);

        // Bare PEC should have overhead > 1
        assert!(results[1].sampling_overhead > 1.0);

        // Full QEC should have overhead = 1 (no sampling)
        assert!((results[4].sampling_overhead - 1.0).abs() < TOL);
    }

    #[test]
    fn test_analysis_ft_pec_beats_bare() {
        let analysis = PecAnalysis::new(1e-3, 50, 4);
        let results = analysis.compare_all();

        let bare_overhead = results[1].sampling_overhead;
        let ft_pec_d3 = results[2].sampling_overhead;

        // FT-PEC should have lower sampling overhead than bare PEC
        assert!(
            ft_pec_d3 < bare_overhead,
            "FT-PEC (d=3) overhead ({}) should be less than bare PEC ({})",
            ft_pec_d3,
            bare_overhead
        );
    }

    // --------------------------------------------------------
    // Integration / end-to-end tests
    // --------------------------------------------------------

    #[test]
    fn test_end_to_end_pec_pipeline() {
        // Create PEC engine
        let config = PecConfig {
            noise_rate: 0.02,
            num_samples: 1000,
            ..Default::default()
        };
        let engine = PecEngine::new(config).unwrap();

        // Decompose gates
        let qpd = engine.decompose_gate(0.02).unwrap();
        assert!(qpd.is_valid());

        // Check total overhead
        let total = engine.total_overhead(10).unwrap();
        assert!(total > 1.0);
        assert!(total.is_finite());

        // Sample instances
        let qpds = vec![qpd; 10];
        let mut rng = StdRng::seed_from_u64(42);
        let mut signed_samples = Vec::new();

        for _ in 0..100 {
            let (_indices, sign) = engine.sample_instance(&qpds, &mut rng);
            // Simulate a measurement outcome
            let measurement = 0.85 + 0.1 * rng.gen::<f64>();
            signed_samples.push((measurement, sign));
        }

        // Estimate expectation
        let (mean, stderr) = PecEngine::estimate_expectation(&signed_samples);
        assert!(mean.is_finite());
        assert!(stderr.is_finite());
        assert!(stderr >= 0.0);
    }

    #[test]
    fn test_end_to_end_ft_pec_pipeline() {
        // Create FT-PEC
        let config = FtPecConfig {
            code_distance: 3,
            physical_error_rate: 1e-3,
            compilation_strategy: CompilationStrategy::CliffordT,
            max_overhead: 1e10,
            num_samples: 10_000,
            distillation_fidelity: 0.9999,
        };
        let ft_pec = FtPec::new(config).unwrap();

        // Check logical error rate
        let logical = ft_pec.logical_error_rate();
        assert!(logical > 0.0);
        assert!(logical < ft_pec.config.physical_error_rate);

        // Check overhead comparison
        let (bare, ft, reduction) = ft_pec.overhead_comparison(20).unwrap();
        assert!(bare.is_finite());
        assert!(ft.is_finite());
        assert!(reduction > 0.0);
    }

    #[test]
    fn test_samples_needed_scaling() {
        let config = PecConfig {
            noise_rate: 0.01,
            ..Default::default()
        };
        let engine = PecEngine::new(config).unwrap();

        let n_depth5 = engine.samples_needed(5, 0.01).unwrap();
        let n_depth10 = engine.samples_needed(10, 0.01).unwrap();

        // Samples needed should grow exponentially with depth
        assert!(
            n_depth10 > n_depth5,
            "More samples needed at depth 10 ({}) than depth 5 ({})",
            n_depth10,
            n_depth5
        );
    }
}
