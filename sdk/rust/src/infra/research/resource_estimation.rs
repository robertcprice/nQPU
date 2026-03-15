//! Fault-Tolerant Quantum Resource Estimation
//!
//! Estimates the physical resources (qubits, time, factories) needed to run
//! a quantum algorithm fault-tolerantly under a given QEC scheme. Inspired by
//! Microsoft's Azure Quantum Resource Estimator.
//!
//! # Overview
//!
//! Given a logical circuit description (qubit count, T-count, T-depth, etc.)
//! and a configuration (QEC scheme, physical error rate, target logical error
//! rate, magic state factory), this module computes:
//!
//! - Optimal QEC code distance
//! - Physical qubit counts (data + routing + factory)
//! - Number of T-state factories needed
//! - Wall-clock time estimate
//! - Total space-time volume
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::resource_estimation::*;
//!
//! let circuit = LogicalCircuit {
//!     num_logical_qubits: 100,
//!     t_count: 10_000,
//!     t_depth: 1_000,
//!     clifford_count: 50_000,
//!     measurement_count: 100,
//!     rotation_count: 0,
//!     rotation_precision: 0.0,
//! };
//!
//! let config = ResourceEstimationConfig::default();
//! let estimate = estimate_resources(&circuit, &config).unwrap();
//! println!("{}", estimate.summary());
//! ```

use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during resource estimation.
#[derive(Clone, Debug, PartialEq)]
pub enum ResourceError {
    /// The target logical error rate is too low to achieve with any
    /// reasonable code distance at the given physical error rate.
    InfeasibleErrorBudget { target: f64, best_achievable: f64 },
    /// The required code distance exceeds the maximum allowed (practical limit).
    CodeDistanceTooHigh { required: usize, maximum: usize },
    /// No valid magic-state factory configuration can supply T states
    /// at the required rate.
    NoValidFactory,
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceError::InfeasibleErrorBudget {
                target,
                best_achievable,
            } => {
                write!(
                    f,
                    "Infeasible error budget: target {:.2e} but best achievable is {:.2e}",
                    target, best_achievable
                )
            }
            ResourceError::CodeDistanceTooHigh { required, maximum } => {
                write!(
                    f,
                    "Required code distance {} exceeds maximum {}",
                    required, maximum
                )
            }
            ResourceError::NoValidFactory => {
                write!(f, "No valid magic state factory configuration found")
            }
        }
    }
}

// ============================================================
// QEC SCHEME
// ============================================================

/// Quantum Error Correction scheme.
#[derive(Clone, Debug)]
pub enum QecScheme {
    /// Standard surface code: 2*d^2 physical qubits per logical qubit,
    /// threshold ~1%.
    SurfaceCode,
    /// Bivariate bicycle code [[144,12,12]]: higher encoding rate,
    /// fewer physical qubits per logical qubit.
    BicycleCode144,
    /// Heavy-hex code used by IBM hardware.
    HeavyHexCode,
    /// Custom code with parameters.
    CustomCode {
        /// Block length (physical qubits per code block)
        n: usize,
        /// Number of logical qubits per block
        k: usize,
        /// Function mapping code distance to block length
        d_func: fn(usize) -> usize,
    },
}

impl PartialEq for QecScheme {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QecScheme::SurfaceCode, QecScheme::SurfaceCode) => true,
            (QecScheme::BicycleCode144, QecScheme::BicycleCode144) => true,
            (QecScheme::HeavyHexCode, QecScheme::HeavyHexCode) => true,
            _ => false,
        }
    }
}

impl fmt::Display for QecScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QecScheme::SurfaceCode => write!(f, "Surface Code"),
            QecScheme::BicycleCode144 => write!(f, "Bivariate Bicycle [[144,12,12]]"),
            QecScheme::HeavyHexCode => write!(f, "Heavy-Hex Code"),
            QecScheme::CustomCode { n, k, .. } => write!(f, "Custom [[{},{}]]", n, k),
        }
    }
}

// ============================================================
// MAGIC STATE FACTORY
// ============================================================

/// Magic state distillation factory configuration.
#[derive(Clone, Debug)]
pub enum MagicStateFactory {
    /// Standard 15-to-1 distillation: 15 noisy T states -> 1 clean T state.
    Standard15to1,
    /// More efficient 20-to-4 protocol.
    Efficient20to4,
    /// Litinski factory with improved layout.
    Litinski,
    /// Custom factory parameters.
    Custom {
        input_t_states: usize,
        output_t_states: usize,
        output_error: f64,
        num_qubits: usize,
        cycles: usize,
    },
}

impl MagicStateFactory {
    /// Number of input T states consumed per distillation round.
    pub fn input_t_states(&self) -> usize {
        match self {
            MagicStateFactory::Standard15to1 => 15,
            MagicStateFactory::Efficient20to4 => 20,
            MagicStateFactory::Litinski => 15,
            MagicStateFactory::Custom { input_t_states, .. } => *input_t_states,
        }
    }

    /// Number of output (distilled) T states per round.
    pub fn output_t_states(&self) -> usize {
        match self {
            MagicStateFactory::Standard15to1 => 1,
            MagicStateFactory::Efficient20to4 => 4,
            MagicStateFactory::Litinski => 1,
            MagicStateFactory::Custom {
                output_t_states, ..
            } => *output_t_states,
        }
    }

    /// Output error rate of the distilled T state.
    pub fn output_error(&self, physical_error_rate: f64) -> f64 {
        match self {
            MagicStateFactory::Standard15to1 => {
                // 15-to-1: output error ~ 35 * p^3
                35.0 * physical_error_rate.powi(3)
            }
            MagicStateFactory::Efficient20to4 => {
                // 20-to-4: slightly higher output error ~ 50 * p^3
                50.0 * physical_error_rate.powi(3)
            }
            MagicStateFactory::Litinski => {
                // Litinski: output error ~ 20 * p^3 (improved layout)
                20.0 * physical_error_rate.powi(3)
            }
            MagicStateFactory::Custom { output_error, .. } => *output_error,
        }
    }

    /// Physical qubits required for one factory instance at given code distance.
    pub fn qubits_per_factory(&self, code_distance: usize) -> usize {
        let d2 = code_distance * code_distance;
        match self {
            MagicStateFactory::Standard15to1 => {
                // ~15 logical qubits worth of space at 2*d^2 each, plus ancilla
                // Typically ~30*d^2 for the factory footprint
                30 * d2
            }
            MagicStateFactory::Efficient20to4 => {
                // Larger factory but produces 4 T states
                40 * d2
            }
            MagicStateFactory::Litinski => {
                // Litinski layout is more compact: ~20*d^2
                20 * d2
            }
            MagicStateFactory::Custom { num_qubits, .. } => *num_qubits,
        }
    }

    /// Number of QEC cycles per distillation round.
    pub fn cycles_per_round(&self, code_distance: usize) -> usize {
        match self {
            MagicStateFactory::Standard15to1 => {
                // ~5*d cycles for one round of 15-to-1 distillation
                5 * code_distance
            }
            MagicStateFactory::Efficient20to4 => {
                // ~6*d cycles (slightly longer but produces 4)
                6 * code_distance
            }
            MagicStateFactory::Litinski => {
                // ~4*d cycles (optimized layout)
                4 * code_distance
            }
            MagicStateFactory::Custom { cycles, .. } => *cycles,
        }
    }

    /// T-state output rate: states per QEC cycle for one factory instance.
    pub fn output_rate(&self, code_distance: usize) -> f64 {
        let out = self.output_t_states() as f64;
        let cyc = self.cycles_per_round(code_distance) as f64;
        out / cyc
    }
}

impl fmt::Display for MagicStateFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MagicStateFactory::Standard15to1 => write!(f, "15-to-1 Standard"),
            MagicStateFactory::Efficient20to4 => write!(f, "20-to-4 Efficient"),
            MagicStateFactory::Litinski => write!(f, "Litinski Compact"),
            MagicStateFactory::Custom {
                input_t_states,
                output_t_states,
                ..
            } => {
                write!(f, "Custom {}-to-{}", input_t_states, output_t_states)
            }
        }
    }
}

// ============================================================
// LOGICAL CIRCUIT DESCRIPTION
// ============================================================

/// Description of a logical quantum circuit for resource estimation.
///
/// This captures the high-level resource requirements without needing
/// the full gate-level circuit.
#[derive(Clone, Debug)]
pub struct LogicalCircuit {
    /// Number of logical qubits used by the algorithm.
    pub num_logical_qubits: usize,
    /// Total number of T gates (non-Clifford resource).
    pub t_count: usize,
    /// T-gate depth (longest chain of sequential T gates).
    pub t_depth: usize,
    /// Total number of Clifford gates.
    pub clifford_count: usize,
    /// Number of measurements during the circuit.
    pub measurement_count: usize,
    /// Number of arbitrary rotation gates (require decomposition).
    pub rotation_count: usize,
    /// Precision for rotation gate synthesis (in radians).
    /// Each rotation decomposes to ~log2(1/precision) T gates.
    pub rotation_precision: f64,
}

impl LogicalCircuit {
    /// Effective T-count including T gates from rotation decomposition.
    pub fn effective_t_count(&self) -> usize {
        if self.rotation_count == 0 || self.rotation_precision <= 0.0 {
            return self.t_count;
        }
        // Solovay-Kitaev / Ross-Selinger: ~3*log2(1/eps) T gates per rotation
        let t_per_rotation = (3.0 * (1.0 / self.rotation_precision).log2()).ceil() as usize;
        self.t_count + self.rotation_count * t_per_rotation
    }

    /// Effective T-depth including rotation decomposition overhead.
    pub fn effective_t_depth(&self) -> usize {
        if self.rotation_count == 0 || self.rotation_precision <= 0.0 {
            return self.t_depth;
        }
        let t_depth_per_rotation = ((1.0 / self.rotation_precision).log2()).ceil() as usize;
        self.t_depth + self.rotation_count * t_depth_per_rotation
    }

    /// Estimate Clifford depth (rough: assume depth ~ sqrt(count) for random circuits).
    pub fn estimated_clifford_depth(&self) -> usize {
        if self.clifford_count == 0 {
            return 0;
        }
        // Rough estimate: depth proportional to count / width
        let width = self.num_logical_qubits.max(1);
        (self.clifford_count + width - 1) / width
    }
}

// ============================================================
// RESOURCE ESTIMATION CONFIG
// ============================================================

/// Configuration for fault-tolerant resource estimation.
#[derive(Clone, Debug)]
pub struct ResourceEstimationConfig {
    /// Physical error rate of the hardware (default: 1e-3).
    pub physical_error_rate: f64,
    /// QEC scheme to use (default: SurfaceCode).
    pub qec_scheme: QecScheme,
    /// Magic state factory type (default: Standard15to1).
    pub magic_state_factory: MagicStateFactory,
    /// Target total logical error rate (default: 1e-12).
    pub target_logical_error_rate: f64,
    /// Maximum code distance to consider (default: 51).
    pub max_code_distance: usize,
    /// Physical gate cycle time in nanoseconds (default: 100 ns).
    pub cycle_time_ns: f64,
}

impl Default for ResourceEstimationConfig {
    fn default() -> Self {
        Self {
            physical_error_rate: 1e-3,
            qec_scheme: QecScheme::SurfaceCode,
            magic_state_factory: MagicStateFactory::Standard15to1,
            target_logical_error_rate: 1e-12,
            max_code_distance: 51,
            cycle_time_ns: 100.0,
        }
    }
}

impl ResourceEstimationConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the physical error rate.
    pub fn with_physical_error_rate(mut self, rate: f64) -> Self {
        self.physical_error_rate = rate;
        self
    }

    /// Set the QEC scheme.
    pub fn with_qec_scheme(mut self, scheme: QecScheme) -> Self {
        self.qec_scheme = scheme;
        self
    }

    /// Set the magic state factory.
    pub fn with_magic_state_factory(mut self, factory: MagicStateFactory) -> Self {
        self.magic_state_factory = factory;
        self
    }

    /// Set the target logical error rate.
    pub fn with_target_logical_error_rate(mut self, rate: f64) -> Self {
        self.target_logical_error_rate = rate;
        self
    }

    /// Set the maximum code distance.
    pub fn with_max_code_distance(mut self, d: usize) -> Self {
        self.max_code_distance = d;
        self
    }

    /// Set the physical cycle time in nanoseconds.
    pub fn with_cycle_time_ns(mut self, ns: f64) -> Self {
        self.cycle_time_ns = ns;
        self
    }
}

// ============================================================
// RESOURCE ESTIMATE OUTPUT
// ============================================================

/// Complete fault-tolerant resource estimate.
#[derive(Clone, Debug)]
pub struct ResourceEstimate {
    /// Total physical qubits required.
    pub physical_qubits: usize,
    /// QEC code distance used.
    pub code_distance: usize,
    /// Number of T-state distillation factories.
    pub num_t_factories: usize,
    /// Physical qubits used by all T-factories.
    pub factory_qubits: usize,
    /// Physical qubits used for data (logical qubit storage).
    pub data_qubits: usize,
    /// Physical qubits used for routing (lattice surgery).
    pub routing_qubits: usize,
    /// Total wall-clock QEC cycles.
    pub wall_clock_cycles: usize,
    /// Estimated wall-clock time in seconds.
    pub wall_clock_time_seconds: f64,
    /// Achieved logical error rate.
    pub logical_error_rate: f64,
    /// Total T states consumed.
    pub t_states_consumed: usize,
    /// Total space-time volume (physical_qubits * wall_clock_cycles).
    pub total_volume: f64,
}

impl ResourceEstimate {
    /// Human-readable summary of the resource estimate.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Fault-Tolerant Resource Estimate ===\n");
        s.push_str(&format!(
            "Physical qubits:     {:>15}\n",
            format_with_commas(self.physical_qubits)
        ));
        s.push_str(&format!(
            "  Data qubits:       {:>15}\n",
            format_with_commas(self.data_qubits)
        ));
        s.push_str(&format!(
            "  Routing qubits:    {:>15}\n",
            format_with_commas(self.routing_qubits)
        ));
        s.push_str(&format!(
            "  Factory qubits:    {:>15}\n",
            format_with_commas(self.factory_qubits)
        ));
        s.push_str(&format!(
            "Code distance:       {:>15}\n",
            self.code_distance
        ));
        s.push_str(&format!(
            "T-factories:         {:>15}\n",
            self.num_t_factories
        ));
        s.push_str(&format!(
            "T-states consumed:   {:>15}\n",
            format_with_commas(self.t_states_consumed)
        ));
        s.push_str(&format!(
            "Logical error rate:  {:>15.2e}\n",
            self.logical_error_rate
        ));
        s.push_str(&format!(
            "Wall-clock cycles:   {:>15}\n",
            format_with_commas(self.wall_clock_cycles)
        ));
        s.push_str(&format!(
            "Wall-clock time:     {:>15}\n",
            format_duration(self.wall_clock_time_seconds)
        ));
        s.push_str(&format!(
            "Space-time volume:   {:>15.2e}\n",
            self.total_volume
        ));
        s
    }

    /// Formatted table of the resource estimate.
    pub fn to_table(&self) -> String {
        let mut s = String::new();
        let bar = "+---------------------------+-----------------+";
        s.push_str(bar);
        s.push('\n');
        s.push_str("| Parameter                 |           Value |\n");
        s.push_str(bar);
        s.push('\n');
        s.push_str(&format!(
            "| Physical qubits           | {:>15} |\n",
            format_with_commas(self.physical_qubits)
        ));
        s.push_str(&format!(
            "| Data qubits               | {:>15} |\n",
            format_with_commas(self.data_qubits)
        ));
        s.push_str(&format!(
            "| Routing qubits            | {:>15} |\n",
            format_with_commas(self.routing_qubits)
        ));
        s.push_str(&format!(
            "| Factory qubits            | {:>15} |\n",
            format_with_commas(self.factory_qubits)
        ));
        s.push_str(&format!(
            "| Code distance             | {:>15} |\n",
            self.code_distance
        ));
        s.push_str(&format!(
            "| T-factories               | {:>15} |\n",
            self.num_t_factories
        ));
        s.push_str(&format!(
            "| T-states consumed          | {:>15} |\n",
            format_with_commas(self.t_states_consumed)
        ));
        s.push_str(&format!(
            "| Logical error rate         | {:>15.2e} |\n",
            self.logical_error_rate
        ));
        s.push_str(&format!(
            "| Wall-clock cycles          | {:>15} |\n",
            format_with_commas(self.wall_clock_cycles)
        ));
        s.push_str(&format!(
            "| Wall-clock time            | {:>15} |\n",
            format_duration(self.wall_clock_time_seconds)
        ));
        s.push_str(&format!(
            "| Space-time volume          | {:>15.2e} |\n",
            self.total_volume
        ));
        s.push_str(bar);
        s.push('\n');
        s
    }
}

// ============================================================
// CORE ESTIMATION FUNCTIONS
// ============================================================

/// Compute the logical error rate for a surface code at given distance.
///
/// Uses the standard model: p_L ≈ 0.1 * (p / p_th)^((d+1)/2)
/// where p_th ≈ 0.01 is the surface code threshold.
pub fn logical_error_rate_surface_code(physical_error_rate: f64, distance: usize) -> f64 {
    let p_th = 0.01; // Surface code threshold
    let ratio = physical_error_rate / p_th;
    let exponent = ((distance + 1) as f64) / 2.0;
    0.1 * ratio.powf(exponent)
}

/// Compute the logical error rate for a bivariate bicycle code at given distance.
///
/// Bicycle codes have better encoding rates. We model:
/// p_L ≈ 0.05 * (p / p_th)^((d+1)/2) with p_th ≈ 0.007
/// (slightly lower threshold but better qubit overhead).
pub fn logical_error_rate_bicycle(physical_error_rate: f64, distance: usize) -> f64 {
    let p_th = 0.007; // Bicycle code threshold (lower than surface code)
    let ratio = physical_error_rate / p_th;
    let exponent = ((distance + 1) as f64) / 2.0;
    0.05 * ratio.powf(exponent)
}

/// Compute the logical error rate for heavy-hex code at given distance.
///
/// Heavy-hex codes: p_L ≈ 0.08 * (p / p_th)^((d+1)/2), p_th ≈ 0.005
pub fn logical_error_rate_heavy_hex(physical_error_rate: f64, distance: usize) -> f64 {
    let p_th = 0.005;
    let ratio = physical_error_rate / p_th;
    let exponent = ((distance + 1) as f64) / 2.0;
    0.08 * ratio.powf(exponent)
}

/// Compute the logical error rate per logical qubit per QEC cycle for a given scheme.
pub fn logical_error_rate_for_scheme(
    physical_error_rate: f64,
    distance: usize,
    scheme: &QecScheme,
) -> f64 {
    match scheme {
        QecScheme::SurfaceCode => logical_error_rate_surface_code(physical_error_rate, distance),
        QecScheme::BicycleCode144 => logical_error_rate_bicycle(physical_error_rate, distance),
        QecScheme::HeavyHexCode => logical_error_rate_heavy_hex(physical_error_rate, distance),
        QecScheme::CustomCode { .. } => {
            // For custom codes, fall back to surface code model
            logical_error_rate_surface_code(physical_error_rate, distance)
        }
    }
}

/// Find the optimal (minimum) code distance that achieves the target logical error rate.
///
/// The total error budget is distributed across all logical qubits and all QEC cycles:
///   p_total = num_logical_qubits * num_cycles * p_L(d) < target
///
/// Returns the minimum odd distance d >= 3 satisfying this constraint.
pub fn optimal_code_distance(
    physical_error_rate: f64,
    target_logical_rate: f64,
    num_logical_qubits: usize,
    num_cycles: usize,
    scheme: &QecScheme,
    max_distance: usize,
) -> Result<usize, ResourceError> {
    let budget_per_qubit_cycle =
        target_logical_rate / (num_logical_qubits.max(1) as f64 * num_cycles.max(1) as f64);

    // Search odd distances from 3 up to max_distance
    for d in (3..=max_distance).step_by(2) {
        let p_l = logical_error_rate_for_scheme(physical_error_rate, d, scheme);
        if p_l < budget_per_qubit_cycle {
            return Ok(d);
        }
    }

    // If we exhausted the search, report infeasible
    let best_p_l = logical_error_rate_for_scheme(physical_error_rate, max_distance, scheme);
    let best_total = best_p_l * num_logical_qubits.max(1) as f64 * num_cycles.max(1) as f64;

    if best_total < target_logical_rate {
        // max_distance (even) might work, return it
        Ok(max_distance)
    } else {
        Err(ResourceError::InfeasibleErrorBudget {
            target: target_logical_rate,
            best_achievable: best_total,
        })
    }
}

// ============================================================
// PHYSICAL QUBIT COUNTING
// ============================================================

/// Compute the number of physical data qubits needed.
///
/// - Surface code: 2 * d^2 physical qubits per logical qubit (data + syndrome).
/// - Bicycle code [[144,12,12]]: 144/12 = 12 physical qubits per logical qubit
///   at base distance, scaling as ceil(n_logical/k) * n_block.
/// - Heavy-hex: ~2.5 * d^2 per logical qubit.
pub fn data_qubits(num_logical: usize, code_distance: usize, scheme: &QecScheme) -> usize {
    match scheme {
        QecScheme::SurfaceCode => {
            // 2*d^2 per logical qubit: d^2 data + d^2 syndrome ancilla
            num_logical * 2 * code_distance * code_distance
        }
        QecScheme::BicycleCode144 => {
            // [[144, 12, 12]] code: 144 physical per 12 logical
            // Scale distance: physical_per_block ~ 144 * (d/12)^2
            let base_n = 144;
            let base_k = 12;
            let base_d = 12;
            let distance_scale = ((code_distance as f64) / (base_d as f64)).max(1.0);
            let physical_per_block =
                (base_n as f64 * distance_scale * distance_scale).ceil() as usize;
            let blocks_needed = (num_logical + base_k - 1) / base_k;
            blocks_needed * physical_per_block
        }
        QecScheme::HeavyHexCode => {
            // Heavy-hex: ~2.5 * d^2 per logical qubit
            let per_logical = ((2.5 * (code_distance * code_distance) as f64).ceil()) as usize;
            num_logical * per_logical
        }
        QecScheme::CustomCode { n, k, d_func } => {
            let block_size = d_func(code_distance).max(*n);
            let blocks_needed = (num_logical + k - 1) / k;
            blocks_needed * block_size
        }
    }
}

/// Compute routing qubits for lattice surgery.
///
/// Routing overhead is approximately 50% of data qubits for surface code
/// lattice surgery, and 30% for more efficient schemes.
pub fn routing_qubits(num_logical: usize, code_distance: usize, scheme: &QecScheme) -> usize {
    let data = data_qubits(num_logical, code_distance, scheme);
    match scheme {
        QecScheme::SurfaceCode => data / 2,         // 50% overhead
        QecScheme::BicycleCode144 => data * 3 / 10, // 30% overhead
        QecScheme::HeavyHexCode => data * 4 / 10,   // 40% overhead
        QecScheme::CustomCode { .. } => data / 2,   // conservative 50%
    }
}

/// Compute total factory qubits for all T-state factories.
pub fn factory_qubits(
    factory: &MagicStateFactory,
    code_distance: usize,
    num_factories: usize,
) -> usize {
    num_factories * factory.qubits_per_factory(code_distance)
}

// ============================================================
// T-FACTORY PLANNING
// ============================================================

/// Plan the number of T-state factories needed to match consumption rate.
///
/// Returns `(num_factories, total_factory_qubits)`.
///
/// The algorithm ensures factories produce T states at a rate that matches
/// or exceeds the circuit's consumption rate (t_count / t_depth).
pub fn plan_t_factories(
    t_count: usize,
    t_depth: usize,
    factory: &MagicStateFactory,
    code_distance: usize,
) -> (usize, usize) {
    if t_count == 0 {
        return (0, 0);
    }

    let t_depth_eff = t_depth.max(1);

    // Consumption rate: T states needed per logical cycle layer
    let consumption_rate = (t_count as f64) / (t_depth_eff as f64);

    // Each factory's output rate
    let factory_rate = factory.output_rate(code_distance);

    if factory_rate <= 0.0 {
        return (1, factory.qubits_per_factory(code_distance));
    }

    // Number of factories needed (ceiling)
    let num_factories = (consumption_rate / factory_rate).ceil() as usize;
    let num_factories = num_factories.max(1);

    let total_qubits = factory_qubits(factory, code_distance, num_factories);

    (num_factories, total_qubits)
}

// ============================================================
// WALL CLOCK ESTIMATION
// ============================================================

/// Estimate wall-clock time for executing the logical circuit.
///
/// Returns `(total_qec_cycles, total_seconds)`.
///
/// - Logical cycle time = code_distance * cycle_time_ns (for syndrome extraction).
/// - Total cycles = max(t_depth, clifford_depth) + measurement_rounds.
pub fn estimate_wall_clock(
    t_depth: usize,
    clifford_depth: usize,
    measurement_count: usize,
    code_distance: usize,
    cycle_time_ns: f64,
) -> (usize, f64) {
    // Each logical time step requires `code_distance` rounds of syndrome extraction
    let logical_cycle_time_ns = code_distance as f64 * cycle_time_ns;

    // Total logical time steps: the circuit depth is dominated by the longer
    // of T-depth and Clifford-depth, plus measurement rounds
    let total_logical_steps = t_depth.max(clifford_depth) + measurement_count;

    // Total QEC cycles = logical_steps * code_distance (each step = d rounds)
    let total_qec_cycles = total_logical_steps * code_distance;

    // Total time in seconds
    let total_time_ns = total_logical_steps as f64 * logical_cycle_time_ns;
    let total_time_seconds = total_time_ns * 1e-9;

    (total_qec_cycles, total_time_seconds)
}

// ============================================================
// MAIN ENTRY POINT
// ============================================================

/// Estimate the full fault-tolerant physical resources for a logical circuit.
///
/// This is the primary entry point. It:
/// 1. Computes effective T-count/depth (including rotation decomposition)
/// 2. Estimates total QEC cycles for code distance search
/// 3. Finds the optimal code distance
/// 4. Counts data + routing qubits
/// 5. Plans T-factories
/// 6. Estimates wall-clock time
/// 7. Computes total space-time volume
pub fn estimate_resources(
    circuit: &LogicalCircuit,
    config: &ResourceEstimationConfig,
) -> Result<ResourceEstimate, ResourceError> {
    let eff_t_count = circuit.effective_t_count();
    let eff_t_depth = circuit.effective_t_depth();
    let clifford_depth = circuit.estimated_clifford_depth();

    // Rough cycle estimate for code distance search
    let rough_cycles = eff_t_depth.max(clifford_depth) + circuit.measurement_count;
    let rough_cycles = rough_cycles.max(1);

    // Step 1: Find optimal code distance
    let code_distance = optimal_code_distance(
        config.physical_error_rate,
        config.target_logical_error_rate,
        circuit.num_logical_qubits,
        rough_cycles * config.max_code_distance, // conservative: multiply by d for QEC cycles
        &config.qec_scheme,
        config.max_code_distance,
    )?;

    // Step 2: Compute data and routing qubits
    let data = data_qubits(
        circuit.num_logical_qubits,
        code_distance,
        &config.qec_scheme,
    );
    let routing = routing_qubits(
        circuit.num_logical_qubits,
        code_distance,
        &config.qec_scheme,
    );

    // Step 3: Plan T-factories
    let (num_factories, fac_qubits) = plan_t_factories(
        eff_t_count,
        eff_t_depth,
        &config.magic_state_factory,
        code_distance,
    );

    // Step 4: Wall clock estimation
    let (wall_cycles, wall_seconds) = estimate_wall_clock(
        eff_t_depth,
        clifford_depth,
        circuit.measurement_count,
        code_distance,
        config.cycle_time_ns,
    );

    // Step 5: Achieved logical error rate
    let p_l = logical_error_rate_for_scheme(
        config.physical_error_rate,
        code_distance,
        &config.qec_scheme,
    );
    let achieved_error = p_l * circuit.num_logical_qubits as f64 * wall_cycles as f64;

    // Step 6: Total physical qubits
    let total_physical = data + routing + fac_qubits;

    // Step 7: Space-time volume
    let total_volume = total_physical as f64 * wall_cycles as f64;

    Ok(ResourceEstimate {
        physical_qubits: total_physical,
        code_distance,
        num_t_factories: num_factories,
        factory_qubits: fac_qubits,
        data_qubits: data,
        routing_qubits: routing,
        wall_clock_cycles: wall_cycles,
        wall_clock_time_seconds: wall_seconds,
        logical_error_rate: achieved_error,
        t_states_consumed: eff_t_count,
        total_volume,
    })
}

// ============================================================
// CONVENIENCE: WELL-KNOWN ALGORITHMS
// ============================================================

/// Estimate resources for Shor's algorithm factoring RSA-2048.
///
/// Parameters from Gidney & Ekera (2021): ~20 million physical qubits
/// on surface code with ~8 hours runtime.
pub fn estimate_shor_rsa2048() -> Result<ResourceEstimate, ResourceError> {
    let circuit = LogicalCircuit {
        num_logical_qubits: 4098,       // 2*2048 + 2 ancilla
        t_count: 2_000_000_000,         // ~2 billion T gates
        t_depth: 400_000_000,           // ~400M T-depth layers
        clifford_count: 10_000_000_000, // ~10 billion Cliffords
        measurement_count: 4098,
        rotation_count: 0,
        rotation_precision: 0.0,
    };

    let config = ResourceEstimationConfig::new()
        .with_physical_error_rate(1e-3)
        .with_qec_scheme(QecScheme::SurfaceCode)
        .with_magic_state_factory(MagicStateFactory::Litinski)
        .with_target_logical_error_rate(1e-12)
        .with_max_code_distance(51);

    estimate_resources(&circuit, &config)
}

/// Estimate resources for VQE on molecular hydrogen (H2).
///
/// Small chemistry example: 4 qubits, low gate counts.
pub fn estimate_vqe_h2() -> Result<ResourceEstimate, ResourceError> {
    let circuit = LogicalCircuit {
        num_logical_qubits: 4,
        t_count: 280,
        t_depth: 70,
        clifford_count: 1200,
        measurement_count: 4,
        rotation_count: 24,
        rotation_precision: 1e-4,
    };

    let config = ResourceEstimationConfig::new()
        .with_physical_error_rate(1e-3)
        .with_target_logical_error_rate(1e-6);

    estimate_resources(&circuit, &config)
}

/// Estimate resources for Grover search targeting AES-128 key.
///
/// Grover's algorithm on AES-128: 2^64 iterations, large circuit.
pub fn estimate_grover_aes128() -> Result<ResourceEstimate, ResourceError> {
    let circuit = LogicalCircuit {
        num_logical_qubits: 2953, // AES-128 Grover (Grassl et al.)
        t_count: 1_500_000_000,   // ~1.5 billion T gates per iteration (amortized)
        t_depth: 300_000_000,
        clifford_count: 8_000_000_000,
        measurement_count: 2953,
        rotation_count: 0,
        rotation_precision: 0.0,
    };

    let config = ResourceEstimationConfig::new()
        .with_physical_error_rate(1e-3)
        .with_target_logical_error_rate(1e-15);

    estimate_resources(&circuit, &config)
}

/// Compare multiple QEC schemes for the same logical circuit.
///
/// Returns a vector of (scheme_name, estimate) pairs, skipping schemes
/// that fail feasibility checks.
pub fn compare_qec_schemes(
    circuit: &LogicalCircuit,
    schemes: &[QecScheme],
    physical_error_rate: f64,
    target_logical_error_rate: f64,
) -> Vec<(String, ResourceEstimate)> {
    let mut results = Vec::new();

    for scheme in schemes {
        let config = ResourceEstimationConfig::new()
            .with_physical_error_rate(physical_error_rate)
            .with_qec_scheme(scheme.clone())
            .with_target_logical_error_rate(target_logical_error_rate);

        if let Ok(est) = estimate_resources(circuit, &config) {
            results.push((format!("{}", scheme), est));
        }
    }

    results
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Format a number with comma separators.
fn format_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format seconds into a human-readable duration string.
fn format_duration(seconds: f64) -> String {
    if seconds < 1e-6 {
        format!("{:.2} ns", seconds * 1e9)
    } else if seconds < 1e-3 {
        format!("{:.2} us", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:.2} ms", seconds * 1e3)
    } else if seconds < 60.0 {
        format!("{:.2} s", seconds)
    } else if seconds < 3600.0 {
        format!("{:.1} min", seconds / 60.0)
    } else if seconds < 86400.0 {
        format!("{:.1} hrs", seconds / 3600.0)
    } else if seconds < 86400.0 * 365.0 {
        format!("{:.1} days", seconds / 86400.0)
    } else {
        format!("{:.1} years", seconds / (86400.0 * 365.0))
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_defaults() {
        let config = ResourceEstimationConfig::default();
        assert!((config.physical_error_rate - 1e-3).abs() < 1e-15);
        assert!((config.target_logical_error_rate - 1e-12).abs() < 1e-25);
        assert_eq!(config.max_code_distance, 51);
        assert!((config.cycle_time_ns - 100.0).abs() < 1e-10);
        assert!(matches!(config.qec_scheme, QecScheme::SurfaceCode));
        assert!(matches!(
            config.magic_state_factory,
            MagicStateFactory::Standard15to1
        ));
    }

    #[test]
    fn test_surface_code_logical_error_decreases_with_distance() {
        let p = 1e-3;
        let rate_d3 = logical_error_rate_surface_code(p, 3);
        let rate_d5 = logical_error_rate_surface_code(p, 5);
        let rate_d7 = logical_error_rate_surface_code(p, 7);
        let rate_d11 = logical_error_rate_surface_code(p, 11);

        assert!(rate_d5 < rate_d3, "d=5 should be lower than d=3");
        assert!(rate_d7 < rate_d5, "d=7 should be lower than d=5");
        assert!(rate_d11 < rate_d7, "d=11 should be lower than d=7");

        // At p=1e-3 and p_th=0.01, ratio=0.1, so error drops exponentially
        // d=3: 0.1 * 0.1^2 = 1e-3
        // d=5: 0.1 * 0.1^3 = 1e-4
        assert!(
            rate_d3 < 1e-2,
            "d=3 error should be reasonable: {}",
            rate_d3
        );
        assert!(rate_d5 < 1e-3, "d=5 error should be <1e-3: {}", rate_d5);
    }

    #[test]
    fn test_optimal_distance_increases_with_lower_target() {
        let p = 1e-3;
        let n_qubits = 100;
        let n_cycles = 1000;

        let d_1e6 = optimal_code_distance(p, 1e-6, n_qubits, n_cycles, &QecScheme::SurfaceCode, 51)
            .unwrap();
        let d_1e9 = optimal_code_distance(p, 1e-9, n_qubits, n_cycles, &QecScheme::SurfaceCode, 51)
            .unwrap();
        let d_1e12 =
            optimal_code_distance(p, 1e-12, n_qubits, n_cycles, &QecScheme::SurfaceCode, 51)
                .unwrap();

        assert!(
            d_1e9 >= d_1e6,
            "Tighter target should need >= distance: d(1e-9)={} vs d(1e-6)={}",
            d_1e9,
            d_1e6
        );
        assert!(
            d_1e12 >= d_1e9,
            "Tighter target should need >= distance: d(1e-12)={} vs d(1e-9)={}",
            d_1e12,
            d_1e9
        );
    }

    #[test]
    fn test_data_qubits_surface_code_formula() {
        // Surface code: 2 * d^2 per logical qubit
        let d = 7;
        let n_logical = 10;
        let expected = n_logical * 2 * d * d;
        let actual = data_qubits(n_logical, d, &QecScheme::SurfaceCode);
        assert_eq!(actual, expected, "Data qubits should be 2*d^2 per logical");

        // Check d=11
        let d = 11;
        let expected = n_logical * 2 * d * d;
        let actual = data_qubits(n_logical, d, &QecScheme::SurfaceCode);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_routing_overhead_approximately_50_percent() {
        let d = 7;
        let n_logical = 100;
        let data = data_qubits(n_logical, d, &QecScheme::SurfaceCode);
        let routing = routing_qubits(n_logical, d, &QecScheme::SurfaceCode);

        // Routing should be ~50% of data for surface code
        let ratio = routing as f64 / data as f64;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "Routing should be ~50% of data, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_t_factory_count_increases_with_t_depth() {
        let factory = MagicStateFactory::Standard15to1;
        let code_distance = 7;
        let t_count = 10_000;

        // With small T-depth, we need more factories to produce T states fast enough
        let (nf_small_depth, _) = plan_t_factories(t_count, 100, &factory, code_distance);
        let (nf_large_depth, _) = plan_t_factories(t_count, 10_000, &factory, code_distance);

        assert!(
            nf_small_depth >= nf_large_depth,
            "Smaller T-depth should need more factories: small_depth={} vs large_depth={}",
            nf_small_depth,
            nf_large_depth
        );
    }

    #[test]
    fn test_wall_clock_proportional_to_code_distance() {
        let t_depth = 100;
        let clifford_depth = 50;
        let measurements = 10;

        let (cycles_d5, time_d5) =
            estimate_wall_clock(t_depth, clifford_depth, measurements, 5, 100.0);
        let (cycles_d11, time_d11) =
            estimate_wall_clock(t_depth, clifford_depth, measurements, 11, 100.0);

        // Cycles should scale linearly with d (total_logical_steps * d)
        // Ratio of cycles should be approximately 11/5
        let cycle_ratio = cycles_d11 as f64 / cycles_d5 as f64;
        assert!(
            (cycle_ratio - 11.0 / 5.0).abs() < 0.01,
            "Cycle ratio should be 11/5, got {:.3}",
            cycle_ratio
        );

        // Time scales linearly with d: time = logical_steps * d * cycle_time_ns
        let time_ratio = time_d11 / time_d5;
        assert!(
            (time_ratio - 11.0 / 5.0).abs() < 0.01,
            "Time ratio should be 11/5, got {:.3}",
            time_ratio
        );
    }

    #[test]
    fn test_vqe_h2_produces_reasonable_numbers() {
        let est = estimate_vqe_h2().expect("VQE H2 should succeed");

        // Small circuit: should need < 10M physical qubits
        // (rotation decomposition and factory overhead inflate this)
        assert!(
            est.physical_qubits < 10_000_000,
            "VQE H2 should need < 10M qubits, got {}",
            est.physical_qubits
        );
        assert!(est.physical_qubits > 0);

        // Should need at least some T-factories
        assert!(est.num_t_factories >= 1);

        // Code distance should be modest
        assert!(est.code_distance >= 3);
        assert!(est.code_distance <= 25);

        // Should complete in reasonable time (< 1 second for small circuit)
        assert!(est.wall_clock_time_seconds < 1.0);
    }

    #[test]
    fn test_shor_rsa2048_millions_of_qubits() {
        let est = estimate_shor_rsa2048().expect("Shor RSA-2048 should succeed");

        // RSA-2048 factoring requires millions of physical qubits
        assert!(
            est.physical_qubits > 1_000_000,
            "Shor RSA-2048 should need >1M qubits, got {}",
            est.physical_qubits
        );

        // T-state consumption should be very high
        assert!(est.t_states_consumed > 1_000_000_000);

        // Should take significant time
        assert!(est.wall_clock_time_seconds > 1.0);
    }

    #[test]
    fn test_bicycle_code_fewer_qubits_than_surface_code() {
        let circuit = LogicalCircuit {
            num_logical_qubits: 50,
            t_count: 5000,
            t_depth: 500,
            clifford_count: 20_000,
            measurement_count: 50,
            rotation_count: 0,
            rotation_precision: 0.0,
        };

        let d = 7;
        let surface_data = data_qubits(circuit.num_logical_qubits, d, &QecScheme::SurfaceCode);
        let bicycle_data = data_qubits(circuit.num_logical_qubits, d, &QecScheme::BicycleCode144);

        assert!(
            bicycle_data < surface_data,
            "Bicycle code should use fewer data qubits: bicycle={} vs surface={}",
            bicycle_data,
            surface_data
        );
    }

    #[test]
    fn test_infeasible_error_budget_returns_error() {
        // Extremely low target with high physical error rate
        let result = optimal_code_distance(
            1e-2,      // High physical error rate (right at threshold)
            1e-30,     // Impossibly low target
            1000,      // Many qubits
            1_000_000, // Many cycles
            &QecScheme::SurfaceCode,
            51, // Limited max distance
        );

        assert!(result.is_err(), "Should fail with infeasible error budget");
        if let Err(ResourceError::InfeasibleErrorBudget { .. }) = result {
            // correct error variant
        } else {
            panic!("Expected InfeasibleErrorBudget error");
        }
    }

    #[test]
    fn test_total_volume_equals_qubits_times_cycles() {
        let circuit = LogicalCircuit {
            num_logical_qubits: 20,
            t_count: 1000,
            t_depth: 100,
            clifford_count: 5000,
            measurement_count: 20,
            rotation_count: 0,
            rotation_precision: 0.0,
        };

        let config = ResourceEstimationConfig::default();
        let est = estimate_resources(&circuit, &config).unwrap();

        let expected_volume = est.physical_qubits as f64 * est.wall_clock_cycles as f64;
        assert!(
            (est.total_volume - expected_volume).abs() < 1.0,
            "Volume should be qubits * cycles: {:.0} vs {:.0}",
            est.total_volume,
            expected_volume
        );
    }

    #[test]
    fn test_compare_schemes_bicycle_fewer_qubits() {
        // At a low physical error rate, bicycle code's higher encoding rate
        // (12 logical per 144 physical) gives fewer total physical qubits
        // than surface code (1 logical per 2*d^2 physical).
        let circuit = LogicalCircuit {
            num_logical_qubits: 50,
            t_count: 5000,
            t_depth: 500,
            clifford_count: 20_000,
            measurement_count: 50,
            rotation_count: 0,
            rotation_precision: 0.0,
        };

        let schemes = vec![QecScheme::SurfaceCode, QecScheme::BicycleCode144];
        // Use low physical error rate so both codes achieve targets at similar distances
        let results = compare_qec_schemes(&circuit, &schemes, 1e-4, 1e-10);

        assert!(results.len() >= 2, "Both schemes should produce estimates");

        let surface_qubits = results
            .iter()
            .find(|(name, _)| name.contains("Surface"))
            .map(|(_, est)| est.data_qubits)
            .expect("Surface code result");

        let bicycle_qubits = results
            .iter()
            .find(|(name, _)| name.contains("Bicycle"))
            .map(|(_, est)| est.data_qubits)
            .expect("Bicycle code result");

        assert!(
            bicycle_qubits < surface_qubits,
            "Bicycle should use fewer data qubits: bicycle={} vs surface={}",
            bicycle_qubits,
            surface_qubits
        );
    }

    #[test]
    fn test_factory_qubits_scale_with_num_factories() {
        let factory = MagicStateFactory::Standard15to1;
        let d = 7;

        let q1 = factory_qubits(&factory, d, 1);
        let q3 = factory_qubits(&factory, d, 3);
        let q10 = factory_qubits(&factory, d, 10);

        assert_eq!(q3, 3 * q1, "3 factories should use 3x qubits");
        assert_eq!(q10, 10 * q1, "10 factories should use 10x qubits");
    }

    #[test]
    fn test_effective_t_count_with_rotations() {
        let circuit = LogicalCircuit {
            num_logical_qubits: 10,
            t_count: 100,
            t_depth: 50,
            clifford_count: 500,
            measurement_count: 10,
            rotation_count: 20,
            rotation_precision: 1e-4,
        };

        let eff_t = circuit.effective_t_count();
        // Each rotation ~ 3 * log2(1e4) ≈ 3 * 13.3 ≈ 40 T gates
        // Total: 100 + 20 * 40 = 900
        assert!(
            eff_t > circuit.t_count,
            "Effective T-count should be higher than base: {} vs {}",
            eff_t,
            circuit.t_count
        );
        assert!(eff_t > 500, "Should include rotation overhead: {}", eff_t);
    }

    #[test]
    fn test_zero_t_count_no_factories() {
        let circuit = LogicalCircuit {
            num_logical_qubits: 10,
            t_count: 0,
            t_depth: 0,
            clifford_count: 500,
            measurement_count: 10,
            rotation_count: 0,
            rotation_precision: 0.0,
        };

        let config = ResourceEstimationConfig::default();
        let est = estimate_resources(&circuit, &config).unwrap();

        assert_eq!(est.num_t_factories, 0, "No T gates = no factories needed");
        assert_eq!(est.factory_qubits, 0, "No factories = no factory qubits");
        assert_eq!(est.t_states_consumed, 0);
    }

    #[test]
    fn test_summary_and_table_not_empty() {
        let est = estimate_vqe_h2().expect("VQE H2 should succeed");

        let summary = est.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Physical qubits"));
        assert!(summary.contains("Code distance"));

        let table = est.to_table();
        assert!(!table.is_empty());
        assert!(table.contains("Physical qubits"));
        assert!(table.contains("+--"));
    }

    #[test]
    fn test_format_with_commas() {
        assert_eq!(format_with_commas(0), "0");
        assert_eq!(format_with_commas(999), "999");
        assert_eq!(format_with_commas(1000), "1,000");
        assert_eq!(format_with_commas(1_000_000), "1,000,000");
        assert_eq!(format_with_commas(123_456_789), "123,456,789");
    }

    #[test]
    fn test_format_duration() {
        assert!(format_duration(0.5e-9).contains("ns"));
        assert!(format_duration(0.5e-3).contains("us"));
        assert!(format_duration(0.5).contains("ms"));
        assert!(format_duration(5.0).contains(" s"));
        assert!(format_duration(120.0).contains("min"));
        assert!(format_duration(7200.0).contains("hrs"));
        assert!(format_duration(172800.0).contains("days"));
    }

    #[test]
    fn test_bicycle_code_logical_error_rate() {
        let p = 1e-3;
        let rate_d5 = logical_error_rate_bicycle(p, 5);
        let rate_d11 = logical_error_rate_bicycle(p, 11);

        assert!(
            rate_d11 < rate_d5,
            "Higher distance should give lower error"
        );
        assert!(
            rate_d5 > 0.0 && rate_d5 < 1.0,
            "Error rate should be in (0,1)"
        );
    }
}
