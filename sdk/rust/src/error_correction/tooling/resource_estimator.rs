//! Fault-Tolerant Quantum Resource Estimation
//!
//! Answers the fundamental question in quantum computing: "How many physical
//! qubits and how long will it take to run a fault-tolerant quantum algorithm?"
//!
//! This module bridges the gap between logical algorithm requirements (number of
//! logical qubits, T-gate count, circuit depth) and physical hardware constraints
//! (gate error rates, cycle times, qubit connectivity). It models the full
//! resource pipeline: surface code encoding, magic state distillation, and
//! time-limited execution scheduling.
//!
//! **Surface Code Model**:
//! Uses the rotated surface code with logical error rate per round:
//!   `p_L(d) = 0.1 * (100 * p_phys)^((d+1)/2)`
//! which is valid below the ~1% threshold. The code distance d is chosen as
//! the minimum odd integer such that the total logical failure probability
//! over the entire circuit depth remains below the target.
//!
//! **Magic State Distillation**:
//! T gates (the dominant non-Clifford operation) cannot be implemented
//! transversally on surface codes. Instead, they require offline magic state
//! preparation via distillation. Two protocols are supported:
//! - **15-to-1**: Standard Bravyi-Kitaev with cubic error suppression.
//! - **Two-level 15-to-1**: Higher fidelity at greater qubit cost.
//!
//! **References**:
//! - Fowler et al., "Surface codes: Towards practical large-scale quantum
//!   computation", PRA 86, 032324 (2012).
//! - Gidney & Ekera, "How to factor 2048 bit RSA integers in 8 hours using
//!   20 million noisy qubits", Quantum 5, 433 (2021).
//! - Litinski, "Magic State Distillation: Not as Costly as You Think",
//!   Quantum 3, 205 (2019).
//! - Bravyi & Kitaev, "Universal quantum computation with ideal Clifford gates
//!   and noisy ancillas", PRA 71, 022316 (2005).

use std::fmt;

// ---------------------------------------------------------------------------
// Algorithm specification
// ---------------------------------------------------------------------------

/// Describes the logical-level requirements of a quantum algorithm.
///
/// These numbers come from circuit-level analysis of the algorithm and are
/// independent of the physical hardware. The T-gate count dominates the
/// resource estimate because T gates require magic state distillation.
#[derive(Clone, Debug)]
pub struct AlgorithmSpec {
    /// Number of logical qubits needed by the algorithm.
    pub logical_qubits: usize,
    /// Number of T gates in the circuit (dominates resource cost).
    pub t_gate_count: usize,
    /// Number of Clifford gates (essentially free on surface codes).
    pub clifford_gate_count: usize,
    /// Number of measurements in the circuit.
    pub measurement_count: usize,
    /// Logical circuit depth (for time estimation).
    pub circuit_depth: usize,
    /// Algorithm name for display purposes.
    pub name: String,
}

impl AlgorithmSpec {
    /// Create a new algorithm specification with all parameters.
    pub fn new(
        logical_qubits: usize,
        t_gate_count: usize,
        clifford_gate_count: usize,
        measurement_count: usize,
        circuit_depth: usize,
    ) -> Self {
        Self {
            logical_qubits,
            t_gate_count,
            clifford_gate_count,
            measurement_count,
            circuit_depth,
            name: String::from("Custom Algorithm"),
        }
    }

    /// Preset for Shor's algorithm factoring an RSA-n key.
    ///
    /// Based on Gidney & Ekera (2021), "How to factor 2048 bit RSA integers
    /// in 8 hours using 20 million noisy qubits":
    /// - Logical qubits: 2n + 3 (two n-bit registers + ancilla)
    /// - T-gate count: ~64n^3 (dominated by modular exponentiation)
    /// - Clifford gates: ~500n^3 (from controlled additions)
    /// - Circuit depth: ~8n^3 (sequential modular multiplications)
    pub fn shor_rsa(key_bits: usize) -> Self {
        let n = key_bits;
        let n3 = n as u64 * n as u64 * n as u64;
        Self {
            logical_qubits: 2 * n + 3,
            t_gate_count: (64 * n3) as usize,
            clifford_gate_count: (500 * n3) as usize,
            measurement_count: n,
            circuit_depth: (8 * n3) as usize,
            name: format!("Shor RSA-{}", key_bits),
        }
    }

    /// Preset for Grover's search algorithm over a database of size N.
    ///
    /// - Logical qubits: ceil(log2(N)) + 1 (index register + ancilla)
    /// - T-gate count: ~sqrt(N) * O(n) per oracle call (n = log2 N)
    /// - Circuit depth: ~sqrt(N) * n (sequential Grover iterations)
    pub fn grover_search(database_size: usize) -> Self {
        let n = (database_size as f64).log2().ceil() as usize;
        let sqrt_n = (database_size as f64).sqrt().ceil() as usize;
        Self {
            logical_qubits: n + 1,
            t_gate_count: sqrt_n * n,
            clifford_gate_count: sqrt_n * n * 4,
            measurement_count: n,
            circuit_depth: sqrt_n * n,
            name: format!("Grover N={}", database_size),
        }
    }

    /// Preset for VQE (Variational Quantum Eigensolver) chemistry simulation.
    ///
    /// Uses Jordan-Wigner encoding with Trotterized evolution:
    /// - Logical qubits: num_orbitals (one qubit per spin-orbital)
    /// - T-gate count: ~num_orbitals^4 * trotter_steps (quartic scaling
    ///   from the number of two-electron integrals)
    /// - Circuit depth: ~num_orbitals^2 * trotter_steps (parallelizable terms)
    pub fn vqe_molecule(num_orbitals: usize, trotter_steps: usize) -> Self {
        let m = num_orbitals;
        let m4 = m * m * m * m;
        Self {
            logical_qubits: m,
            t_gate_count: m4 * trotter_steps,
            clifford_gate_count: m4 * trotter_steps * 2,
            measurement_count: m * trotter_steps,
            circuit_depth: m * m * trotter_steps,
            name: format!("VQE {}orb {}steps", num_orbitals, trotter_steps),
        }
    }

    /// Simple preset with just the key parameters.
    pub fn custom(name: &str, logical_qubits: usize, t_gates: usize) -> Self {
        Self {
            logical_qubits,
            t_gate_count: t_gates,
            clifford_gate_count: t_gates * 10,
            measurement_count: logical_qubits,
            circuit_depth: t_gates,
            name: name.to_string(),
        }
    }
}

impl fmt::Display for AlgorithmSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} logical qubits, {} T gates, depth {}",
            self.name, self.logical_qubits, self.t_gate_count, self.circuit_depth
        )
    }
}

// ---------------------------------------------------------------------------
// Hardware model
// ---------------------------------------------------------------------------

/// Describes the physical capabilities of a quantum hardware platform.
///
/// These parameters determine how expensive each logical operation is in
/// terms of physical resources and wall-clock time.
#[derive(Clone, Debug)]
pub struct HardwareModel {
    /// Physical gate error rate (p_phys). Must be below the ~1% surface code
    /// threshold for fault tolerance to work. Typical values: 1e-3 to 1e-4.
    pub physical_gate_error_rate: f64,
    /// Readout/measurement error rate.
    pub measurement_error_rate: f64,
    /// Physical gate execution time in nanoseconds.
    pub gate_time_ns: f64,
    /// Measurement execution time in nanoseconds.
    pub measurement_time_ns: f64,
    /// QEC syndrome extraction cycle time in microseconds.
    /// One round of surface code error correction takes this long.
    pub code_cycle_time_us: f64,
    /// Hardware platform name for display.
    pub name: String,
}

impl HardwareModel {
    /// Current-generation superconducting hardware (IBM/Google class).
    ///
    /// p_phys ~1e-3, gate time ~25ns, code cycle ~1us.
    pub fn superconducting_current() -> Self {
        Self {
            physical_gate_error_rate: 1e-3,
            measurement_error_rate: 1e-2,
            gate_time_ns: 25.0,
            measurement_time_ns: 500.0,
            code_cycle_time_us: 1.0,
            name: String::from("Superconducting (current)"),
        }
    }

    /// Near-term target superconducting hardware.
    ///
    /// p_phys ~1e-4, gate time ~20ns, code cycle ~0.8us.
    pub fn superconducting_target() -> Self {
        Self {
            physical_gate_error_rate: 1e-4,
            measurement_error_rate: 1e-3,
            gate_time_ns: 20.0,
            measurement_time_ns: 400.0,
            code_cycle_time_us: 0.8,
            name: String::from("Superconducting (target)"),
        }
    }

    /// Current-generation trapped-ion hardware (IonQ/Quantinuum class).
    ///
    /// p_phys ~1e-3, gate time ~100us, code cycle ~500us.
    pub fn trapped_ion_current() -> Self {
        Self {
            physical_gate_error_rate: 1e-3,
            measurement_error_rate: 5e-3,
            gate_time_ns: 100_000.0,
            measurement_time_ns: 200_000.0,
            code_cycle_time_us: 500.0,
            name: String::from("Trapped Ion (current)"),
        }
    }

    /// Near-term target trapped-ion hardware.
    ///
    /// p_phys ~5e-5, gate time ~50us, code cycle ~200us.
    pub fn trapped_ion_target() -> Self {
        Self {
            physical_gate_error_rate: 5e-5,
            measurement_error_rate: 1e-3,
            gate_time_ns: 50_000.0,
            measurement_time_ns: 100_000.0,
            code_cycle_time_us: 200.0,
            name: String::from("Trapped Ion (target)"),
        }
    }

    /// Current-generation neutral atom hardware (QuEra class).
    ///
    /// p_phys ~5e-3, gate time ~1us, code cycle ~100us.
    pub fn neutral_atom_current() -> Self {
        Self {
            physical_gate_error_rate: 5e-3,
            measurement_error_rate: 2e-2,
            gate_time_ns: 1_000.0,
            measurement_time_ns: 10_000.0,
            code_cycle_time_us: 100.0,
            name: String::from("Neutral Atom (current)"),
        }
    }

    /// Near-term target neutral atom hardware.
    ///
    /// p_phys ~1e-4, gate time ~0.5us, code cycle ~50us.
    pub fn neutral_atom_target() -> Self {
        Self {
            physical_gate_error_rate: 1e-4,
            measurement_error_rate: 1e-3,
            gate_time_ns: 500.0,
            measurement_time_ns: 5_000.0,
            code_cycle_time_us: 50.0,
            name: String::from("Neutral Atom (target)"),
        }
    }
}

impl fmt::Display for HardwareModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: p_phys={:.1e}, cycle={:.1}us",
            self.name, self.physical_gate_error_rate, self.code_cycle_time_us
        )
    }
}

// ---------------------------------------------------------------------------
// Distillation protocol
// ---------------------------------------------------------------------------

/// Magic state distillation protocol choice.
///
/// T gates require distilled magic states. The protocol determines the
/// trade-off between qubit overhead and output fidelity.
#[derive(Clone, Debug, PartialEq)]
pub enum DistillationProtocol {
    /// Standard 15-to-1 Bravyi-Kitaev protocol.
    ///
    /// Consumes 15 noisy T states to produce 1 clean T state.
    /// Factory footprint: ~15 logical patches.
    /// Generation time: ~10 code cycles.
    /// Error suppression: cubic (p_out = 35 * p_in^3).
    Standard15to1,

    /// Two-level cascaded 15-to-1 protocol.
    ///
    /// Runs two rounds of 15-to-1 distillation in series for much
    /// higher output fidelity at the cost of ~15x more physical qubits.
    /// Factory footprint: ~225 logical patches (15 * 15).
    /// Generation time: ~100 code cycles (10 * 10).
    TwoLevel15to1,
}

impl DistillationProtocol {
    /// Number of logical qubit patches per factory.
    pub fn factory_patches(&self) -> usize {
        match self {
            DistillationProtocol::Standard15to1 => 15,
            DistillationProtocol::TwoLevel15to1 => 225,
        }
    }

    /// Code cycles per magic state produced.
    pub fn cycles_per_state(&self) -> usize {
        match self {
            DistillationProtocol::Standard15to1 => 10,
            DistillationProtocol::TwoLevel15to1 => 100,
        }
    }
}

impl fmt::Display for DistillationProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistillationProtocol::Standard15to1 => write!(f, "15-to-1 (standard)"),
            DistillationProtocol::TwoLevel15to1 => write!(f, "15-to-1 (two-level)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Resource estimate output
// ---------------------------------------------------------------------------

/// Complete resource estimate for running a fault-tolerant quantum algorithm.
///
/// This is the main output of the resource estimator, containing all the
/// numbers needed to understand the physical cost of an algorithm.
#[derive(Clone, Debug)]
pub struct ResourceEstimate {
    /// Required surface code distance d (always odd, >= 3).
    pub code_distance: usize,
    /// Physical qubits per logical qubit: 2d^2 for rotated surface code.
    pub physical_qubits_per_logical: usize,
    /// Total data qubits: physical_qubits_per_logical * logical_qubits.
    pub total_data_qubits: usize,
    /// Number of magic state distillation factories.
    pub magic_state_factories: usize,
    /// Physical qubits consumed by each factory.
    pub factory_qubits: usize,
    /// Total physical qubits: data + all factory qubits.
    pub total_physical_qubits: usize,
    /// Logical error rate per QEC round at the chosen code distance.
    pub logical_error_rate: f64,
    /// Estimated wall-clock execution time in seconds.
    pub total_time_seconds: f64,
    /// Time to generate one magic state in microseconds.
    pub magic_state_generation_time_us: f64,
    /// The algorithm specification that was estimated.
    pub algorithm_spec: AlgorithmSpec,
    /// The hardware model used for the estimate.
    pub hardware_model: HardwareModel,
}

impl ResourceEstimate {
    /// Generate a human-readable summary of the resource estimate.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("====================================================\n");
        s.push_str("  Fault-Tolerant Resource Estimate\n");
        s.push_str("====================================================\n");
        s.push_str(&format!("  Algorithm:         {}\n", self.algorithm_spec.name));
        s.push_str(&format!("  Hardware:          {}\n", self.hardware_model.name));
        s.push_str("----------------------------------------------------\n");
        s.push_str(&format!(
            "  Logical qubits:    {}\n",
            self.algorithm_spec.logical_qubits
        ));
        s.push_str(&format!(
            "  T-gate count:      {}\n",
            self.algorithm_spec.t_gate_count
        ));
        s.push_str(&format!(
            "  Circuit depth:     {}\n",
            self.algorithm_spec.circuit_depth
        ));
        s.push_str("----------------------------------------------------\n");
        s.push_str(&format!("  Code distance:     {}\n", self.code_distance));
        s.push_str(&format!(
            "  Qubits/logical:    {}\n",
            self.physical_qubits_per_logical
        ));
        s.push_str(&format!("  Data qubits:       {}\n", self.total_data_qubits));
        s.push_str(&format!(
            "  Factories:         {}\n",
            self.magic_state_factories
        ));
        s.push_str(&format!(
            "  Factory qubits:    {} (each)\n",
            self.factory_qubits
        ));
        s.push_str(&format!(
            "  TOTAL phys qubits: {}\n",
            self.total_physical_qubits
        ));
        s.push_str("----------------------------------------------------\n");
        s.push_str(&format!(
            "  Logical error/rnd: {:.2e}\n",
            self.logical_error_rate
        ));
        s.push_str(&format!(
            "  Magic state time:  {:.1} us\n",
            self.magic_state_generation_time_us
        ));

        // Format time in human-readable units.
        let time = self.total_time_seconds;
        if time < 1.0 {
            s.push_str(&format!("  Total time:        {:.2} ms\n", time * 1e3));
        } else if time < 3600.0 {
            s.push_str(&format!("  Total time:        {:.2} s\n", time));
        } else if time < 86400.0 {
            s.push_str(&format!("  Total time:        {:.2} hours\n", time / 3600.0));
        } else if time < 86400.0 * 365.0 {
            s.push_str(&format!("  Total time:        {:.2} days\n", time / 86400.0));
        } else {
            s.push_str(&format!(
                "  Total time:        {:.2} years\n",
                time / (86400.0 * 365.0)
            ));
        }

        s.push_str("====================================================\n");
        s
    }
}

impl fmt::Display for ResourceEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Resource estimator
// ---------------------------------------------------------------------------

/// Main resource estimator for fault-tolerant quantum computation.
///
/// Given an algorithm specification and hardware model, computes the minimum
/// surface code distance, number of physical qubits, magic state factories,
/// and wall-clock execution time required to run the algorithm with the
/// target logical failure probability.
///
/// # Example
/// ```
/// use nqpu_metal::resource_estimator::*;
///
/// let estimator = ResourceEstimator::new();
/// let alg = AlgorithmSpec::shor_rsa(2048);
/// let hw = HardwareModel::superconducting_current();
/// let estimate = estimator.estimate(&alg, &hw);
/// assert!(estimate.total_physical_qubits > 1_000_000);
/// ```
pub struct ResourceEstimator {
    /// Target total logical failure probability for the entire computation.
    /// Default: 0.01 (1%). The algorithm succeeds with probability >= 1 - target.
    pub target_logical_error_rate: f64,
    /// Which magic state distillation protocol to use.
    pub distillation_protocol: DistillationProtocol,
}

impl ResourceEstimator {
    /// Create a new estimator with default settings:
    /// - Target failure probability: 1% (1e-2)
    /// - Distillation: Standard 15-to-1
    pub fn new() -> Self {
        Self {
            target_logical_error_rate: 1e-2,
            distillation_protocol: DistillationProtocol::Standard15to1,
        }
    }

    /// Create an estimator with a custom target failure probability.
    pub fn with_target_error(target: f64) -> Self {
        Self {
            target_logical_error_rate: target,
            distillation_protocol: DistillationProtocol::Standard15to1,
        }
    }

    /// Compute the full resource estimate for an algorithm on a hardware platform.
    ///
    /// # Algorithm
    ///
    /// 1. **Code distance**: Find minimum odd d >= 3 such that the per-round
    ///    logical error rate p_L(d) satisfies:
    ///    `p_L * circuit_depth * logical_qubits < target_logical_error_rate`
    ///
    /// 2. **Physical qubits**: Each logical qubit uses 2d^2 physical qubits
    ///    (rotated surface code: d^2 data + d^2 - 1 syndrome, rounded to 2d^2).
    ///
    /// 3. **Magic state factories**: Enough factories to pipeline one T-state
    ///    delivery per circuit depth layer.
    ///
    /// 4. **Execution time**: max(circuit execution time, T-state bottleneck time).
    pub fn estimate(&self, algorithm: &AlgorithmSpec, hardware: &HardwareModel) -> ResourceEstimate {
        // Step 1: Find minimum code distance.
        let code_distance = self.find_code_distance(algorithm, hardware);

        // Step 2: Physical qubits per logical qubit (rotated surface code).
        let physical_qubits_per_logical = 2 * code_distance * code_distance;

        // Step 3: Total data qubits.
        let total_data_qubits = physical_qubits_per_logical * algorithm.logical_qubits;

        // Step 4: Logical error rate at this code distance.
        let logical_error_rate =
            surface_code_logical_error_rate(code_distance, hardware.physical_gate_error_rate);

        // Step 5: Magic state factory sizing.
        let magic_generation_time_us =
            self.distillation_protocol.cycles_per_state() as f64 * hardware.code_cycle_time_us;

        // Number of factories: enough to produce T states at the rate the
        // circuit consumes them. The circuit needs t_gate_count T states over
        // circuit_depth layers, so on average t_gate_count/circuit_depth per
        // layer. Each factory produces one T state per magic_generation_time_us
        // interval, and one layer takes code_cycle_time_us. So we need:
        //   factories >= ceil(t_gates_per_layer * magic_time / cycle_time)
        // Simplified: factories = ceil(t_gate_count / circuit_depth) but
        // at least 1.
        let t_gates_per_layer = if algorithm.circuit_depth > 0 {
            (algorithm.t_gate_count as f64 / algorithm.circuit_depth as f64).ceil() as usize
        } else {
            algorithm.t_gate_count
        };
        let magic_state_factories = t_gates_per_layer.max(1);

        // Step 6: Factory qubits per factory.
        let factory_qubits =
            self.distillation_protocol.factory_patches() * physical_qubits_per_logical;

        // Step 7: Total physical qubits.
        let total_factory_qubits = factory_qubits * magic_state_factories;
        let total_physical_qubits = total_data_qubits + total_factory_qubits;

        // Step 8: Execution time.
        // Circuit execution time (Cliffords are free, one layer per code cycle).
        let circuit_time_us = algorithm.circuit_depth as f64 * hardware.code_cycle_time_us;

        // T-gate bottleneck: total T-states / (factories * production rate).
        let t_gate_time_us = if magic_state_factories > 0 {
            algorithm.t_gate_count as f64 * magic_generation_time_us
                / magic_state_factories as f64
        } else {
            0.0
        };

        // Total time is the maximum of circuit execution and T-gate production.
        let total_time_us = circuit_time_us.max(t_gate_time_us);
        let total_time_seconds = total_time_us / 1e6;

        ResourceEstimate {
            code_distance,
            physical_qubits_per_logical,
            total_data_qubits,
            magic_state_factories,
            factory_qubits,
            total_physical_qubits,
            logical_error_rate,
            total_time_seconds,
            magic_state_generation_time_us: magic_generation_time_us,
            algorithm_spec: algorithm.clone(),
            hardware_model: hardware.clone(),
        }
    }

    /// Find the minimum odd code distance d >= 3 such that the total logical
    /// failure probability stays below the target.
    ///
    /// The condition is:
    ///   p_L(d) * circuit_depth * logical_qubits < target_logical_error_rate
    ///
    /// We iterate d = 3, 5, 7, ... up to a reasonable maximum.
    fn find_code_distance(&self, algorithm: &AlgorithmSpec, hardware: &HardwareModel) -> usize {
        let p_phys = hardware.physical_gate_error_rate;

        // The per-round error rate budget: divide the total target by the
        // number of "error opportunities" (rounds * qubits).
        let error_budget = if algorithm.circuit_depth > 0 && algorithm.logical_qubits > 0 {
            self.target_logical_error_rate
                / (algorithm.circuit_depth as f64 * algorithm.logical_qubits as f64)
        } else {
            self.target_logical_error_rate
        };

        // Iterate odd distances from 3 up to 101 (well beyond practical needs).
        let mut d = 3usize;
        while d <= 101 {
            let p_l = surface_code_logical_error_rate(d, p_phys);
            if p_l < error_budget {
                return d;
            }
            d += 2;
        }

        // If we reach here, even d=101 is insufficient (very noisy hardware
        // or extremely demanding algorithm). Return the maximum tried.
        101
    }
}

impl Default for ResourceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Surface code logical error rate
// ---------------------------------------------------------------------------

/// Compute the surface code logical error rate per QEC round for code
/// distance d and physical error rate p_phys.
///
/// Uses the standard approximation:
///   p_L(d) = 0.1 * (100 * p_phys)^((d+1)/2)
///
/// This formula is valid when p_phys < p_threshold ~ 1%. The factor 0.1
/// is an empirical prefactor from numerical simulations, and the base
/// (100 * p_phys) encodes the ratio of physical error rate to the ~1%
/// threshold.
///
/// At p_phys = 1e-3 (0.1%), the base is 0.1, so:
///   d=3:  p_L = 0.1 * 0.1^2 = 1e-3
///   d=5:  p_L = 0.1 * 0.1^3 = 1e-4
///   d=7:  p_L = 0.1 * 0.1^4 = 1e-5
///   d=9:  p_L = 0.1 * 0.1^5 = 1e-6
pub fn surface_code_logical_error_rate(d: usize, p_phys: f64) -> f64 {
    let base = 100.0 * p_phys;
    let exponent = (d as f64 + 1.0) / 2.0;
    0.1 * base.powf(exponent)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- 1. Shor RSA-2048 on current superconducting hardware --

    #[test]
    fn test_shor_rsa_2048_current_hardware() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::shor_rsa(2048);
        let hw = HardwareModel::superconducting_current();
        let est = estimator.estimate(&alg, &hw);

        // Shor RSA-2048 should require millions of physical qubits on current hardware.
        // Gidney & Ekera estimate ~20 million; our simplified model should be in the
        // same ballpark (millions, not thousands or billions).
        assert!(
            est.total_physical_qubits > 1_000_000,
            "RSA-2048 on current SC should need millions of qubits, got {}",
            est.total_physical_qubits
        );

        // Should need a non-trivial code distance.
        assert!(
            est.code_distance >= 7,
            "RSA-2048 should need code distance >= 7, got {}",
            est.code_distance
        );

        // Time should be substantial (many hours to days).
        assert!(
            est.total_time_seconds > 1.0,
            "RSA-2048 should take more than 1 second, got {:.2}s",
            est.total_time_seconds
        );

        // Logical qubits should be 2*2048 + 3 = 4099.
        assert_eq!(
            est.algorithm_spec.logical_qubits, 4099,
            "Shor RSA-2048 should need 4099 logical qubits"
        );
    }

    // -- 2. Better hardware reduces qubit count --

    #[test]
    fn test_shor_rsa_2048_target_hardware_fewer_qubits() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::shor_rsa(2048);

        let hw_current = HardwareModel::superconducting_current();
        let hw_target = HardwareModel::superconducting_target();

        let est_current = estimator.estimate(&alg, &hw_current);
        let est_target = estimator.estimate(&alg, &hw_target);

        // Better hardware (lower p_phys) should require fewer total qubits
        // because the code distance can be smaller.
        assert!(
            est_target.total_physical_qubits < est_current.total_physical_qubits,
            "Target hardware ({}) should need fewer qubits than current ({})",
            est_target.total_physical_qubits,
            est_current.total_physical_qubits
        );

        assert!(
            est_target.code_distance <= est_current.code_distance,
            "Target hardware distance ({}) should be <= current ({})",
            est_target.code_distance,
            est_current.code_distance
        );
    }

    // -- 3. Simple VQE has reasonable qubit count --

    #[test]
    fn test_simple_vqe_reasonable_qubit_count() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::vqe_molecule(4, 10);
        let hw = HardwareModel::superconducting_current();
        let est = estimator.estimate(&alg, &hw);

        // 4-orbital VQE with 10 Trotter steps should be modest.
        // Only 4 logical qubits, T-gate count = 4^4 * 10 = 2560.
        assert!(
            est.total_physical_qubits < 100_000,
            "Simple VQE should need < 100k qubits, got {}",
            est.total_physical_qubits
        );

        assert_eq!(
            est.algorithm_spec.logical_qubits, 4,
            "4-orbital VQE should need 4 logical qubits"
        );

        // T-gate count should be 4^4 * 10 = 2560.
        assert_eq!(est.algorithm_spec.t_gate_count, 2560);
    }

    // -- 4. Lower physical error rate requires lower code distance --

    #[test]
    fn test_code_distance_scales_with_error_rate() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::custom("Test", 10, 10_000);

        let hw_noisy = HardwareModel::superconducting_current(); // p = 1e-3
        let hw_clean = HardwareModel::superconducting_target(); // p = 1e-4

        let est_noisy = estimator.estimate(&alg, &hw_noisy);
        let est_clean = estimator.estimate(&alg, &hw_clean);

        assert!(
            est_clean.code_distance <= est_noisy.code_distance,
            "Cleaner hardware (d={}) should need <= distance than noisy (d={})",
            est_clean.code_distance,
            est_noisy.code_distance
        );
    }

    // -- 5. Grover search estimate --

    #[test]
    fn test_grover_search_estimate() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::grover_search(1_000_000);
        let hw = HardwareModel::superconducting_current();
        let est = estimator.estimate(&alg, &hw);

        // log2(1e6) ~ 20, so ~21 logical qubits.
        assert!(
            est.algorithm_spec.logical_qubits >= 20
                && est.algorithm_spec.logical_qubits <= 22,
            "Grover 1M should need ~21 logical qubits, got {}",
            est.algorithm_spec.logical_qubits
        );

        // T-gate count: ~sqrt(1e6) * 20 = 1000 * 20 = 20000.
        assert!(
            est.algorithm_spec.t_gate_count > 10_000,
            "Grover 1M should have substantial T-gate count, got {}",
            est.algorithm_spec.t_gate_count
        );

        // Should produce a valid estimate.
        assert!(est.total_physical_qubits > 0);
        assert!(est.total_time_seconds > 0.0);
    }

    // -- 6. Hardware comparison: same algorithm, different platforms --

    #[test]
    fn test_hardware_comparison() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::custom("Benchmark", 100, 1_000_000);

        let hw_sc = HardwareModel::superconducting_current();
        let hw_ion = HardwareModel::trapped_ion_current();
        let hw_atom = HardwareModel::neutral_atom_current();

        let est_sc = estimator.estimate(&alg, &hw_sc);
        let est_ion = estimator.estimate(&alg, &hw_ion);
        let est_atom = estimator.estimate(&alg, &hw_atom);

        // All should produce valid estimates.
        assert!(est_sc.total_physical_qubits > 0);
        assert!(est_ion.total_physical_qubits > 0);
        assert!(est_atom.total_physical_qubits > 0);

        // SC and ion have the same p_phys (1e-3) so same code distance.
        assert_eq!(
            est_sc.code_distance, est_ion.code_distance,
            "SC and ion have same p_phys, should have same distance"
        );

        // Neutral atom has higher p_phys (5e-3), should need higher distance.
        assert!(
            est_atom.code_distance >= est_sc.code_distance,
            "Neutral atom (p=5e-3, d={}) should need >= distance than SC (p=1e-3, d={})",
            est_atom.code_distance,
            est_sc.code_distance
        );

        // Trapped ion has slower cycle time, so should take longer.
        assert!(
            est_ion.total_time_seconds > est_sc.total_time_seconds,
            "Trapped ion ({:.2}s) should be slower than SC ({:.2}s)",
            est_ion.total_time_seconds,
            est_sc.total_time_seconds
        );
    }

    // -- 7. Custom algorithm with known T-gate count --

    #[test]
    fn test_custom_algorithm_formula() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::custom("Known", 10, 1000);
        let hw = HardwareModel::superconducting_current();
        let est = estimator.estimate(&alg, &hw);

        // Verify physical_qubits_per_logical = 2 * d^2.
        let d = est.code_distance;
        assert_eq!(
            est.physical_qubits_per_logical,
            2 * d * d,
            "physical_qubits_per_logical should be 2*d^2 = 2*{}^2 = {}",
            d,
            2 * d * d
        );

        // Verify total_data_qubits = physical_qubits_per_logical * logical_qubits.
        assert_eq!(
            est.total_data_qubits,
            est.physical_qubits_per_logical * 10,
            "total_data_qubits should be {} * 10",
            est.physical_qubits_per_logical
        );

        // Verify total = data + factory.
        let total_factory = est.factory_qubits * est.magic_state_factories;
        assert_eq!(
            est.total_physical_qubits,
            est.total_data_qubits + total_factory,
            "total should be data ({}) + factory ({})",
            est.total_data_qubits,
            total_factory
        );
    }

    // -- 8. Stricter target error increases code distance --

    #[test]
    fn test_target_error_rate_effect() {
        let alg = AlgorithmSpec::custom("Test", 50, 100_000);
        let hw = HardwareModel::superconducting_current();

        let est_loose = ResourceEstimator::with_target_error(1e-2).estimate(&alg, &hw);
        let est_strict = ResourceEstimator::with_target_error(1e-4).estimate(&alg, &hw);

        // Stricter target should require a higher (or equal) code distance.
        assert!(
            est_strict.code_distance >= est_loose.code_distance,
            "Strict target (d={}) should need >= distance than loose (d={})",
            est_strict.code_distance,
            est_loose.code_distance
        );

        // And therefore more physical qubits.
        assert!(
            est_strict.total_physical_qubits >= est_loose.total_physical_qubits,
            "Strict target ({} qubits) should need >= qubits than loose ({})",
            est_strict.total_physical_qubits,
            est_loose.total_physical_qubits
        );
    }

    // -- 9. More T gates require more factories or longer time --

    #[test]
    fn test_factory_count_scales_with_t_gates() {
        let estimator = ResourceEstimator::new();
        let hw = HardwareModel::superconducting_current();

        let alg_few = AlgorithmSpec::custom("Few T", 10, 100);
        let alg_many = AlgorithmSpec::custom("Many T", 10, 1_000_000);

        let est_few = estimator.estimate(&alg_few, &hw);
        let est_many = estimator.estimate(&alg_many, &hw);

        // More T gates should mean either more factories or longer time (or both).
        let resource_few =
            est_few.magic_state_factories as f64 * est_few.total_time_seconds;
        let resource_many =
            est_many.magic_state_factories as f64 * est_many.total_time_seconds;

        assert!(
            resource_many > resource_few,
            "Many T gates (factories*time={:.2e}) should exceed few ({:.2e})",
            resource_many,
            resource_few
        );
    }

    // -- 10. Summary string contains all key metrics --

    #[test]
    fn test_summary_string() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::shor_rsa(2048);
        let hw = HardwareModel::superconducting_current();
        let est = estimator.estimate(&alg, &hw);

        let summary = est.summary();

        // Check that all key sections are present.
        assert!(
            summary.contains("Shor RSA-2048"),
            "Summary should contain algorithm name"
        );
        assert!(
            summary.contains("Superconducting"),
            "Summary should contain hardware name"
        );
        assert!(
            summary.contains("Code distance"),
            "Summary should contain code distance"
        );
        assert!(
            summary.contains("TOTAL phys qubits"),
            "Summary should contain total qubit count"
        );
        assert!(
            summary.contains("Logical error"),
            "Summary should contain logical error rate"
        );
        assert!(
            summary.contains("Total time"),
            "Summary should contain total time"
        );
        assert!(
            summary.contains("Factories"),
            "Summary should contain factory count"
        );
        assert!(
            summary.contains("T-gate count"),
            "Summary should contain T-gate count"
        );
    }

    // -- 11. Logical error rate formula gives correct values --

    #[test]
    fn test_logical_error_rate_formula() {
        // d=3, p_phys=1e-3:
        // p_L = 0.1 * (100 * 1e-3)^((3+1)/2) = 0.1 * (0.1)^2 = 0.1 * 0.01 = 1e-3
        let p_l_d3 = surface_code_logical_error_rate(3, 1e-3);
        assert!(
            (p_l_d3 - 1e-3).abs() < 1e-15,
            "d=3, p=1e-3: expected 1e-3, got {:.6e}",
            p_l_d3
        );

        // d=5, p_phys=1e-3:
        // p_L = 0.1 * (0.1)^3 = 0.1 * 1e-3 = 1e-4
        let p_l_d5 = surface_code_logical_error_rate(5, 1e-3);
        assert!(
            (p_l_d5 - 1e-4).abs() < 1e-16,
            "d=5, p=1e-3: expected 1e-4, got {:.6e}",
            p_l_d5
        );

        // d=7, p_phys=1e-3:
        // p_L = 0.1 * (0.1)^4 = 1e-5
        let p_l_d7 = surface_code_logical_error_rate(7, 1e-3);
        assert!(
            (p_l_d7 - 1e-5).abs() < 1e-17,
            "d=7, p=1e-3: expected 1e-5, got {:.6e}",
            p_l_d7
        );

        // d=9, p_phys=1e-3:
        // p_L = 0.1 * (0.1)^5 = 1e-6
        let p_l_d9 = surface_code_logical_error_rate(9, 1e-3);
        assert!(
            (p_l_d9 - 1e-6).abs() < 1e-18,
            "d=9, p=1e-3: expected 1e-6, got {:.6e}",
            p_l_d9
        );
    }

    // -- 12. Distance d=3 with known p_phys gives expected p_L --

    #[test]
    fn test_distance_3_known_values() {
        // d=3 with p_phys=1e-3:
        // p_L = 0.1 * (100 * 1e-3)^2 = 0.1 * 0.01 = 1e-3
        let p_l = surface_code_logical_error_rate(3, 1e-3);
        let expected = 0.1 * (0.1_f64).powi(2);
        assert!(
            (p_l - expected).abs() < 1e-15,
            "d=3, p=1e-3: got {:.6e}, expected {:.6e}",
            p_l,
            expected
        );

        // d=3 with p_phys=1e-4:
        // p_L = 0.1 * (100 * 1e-4)^2 = 0.1 * (0.01)^2 = 0.1 * 1e-4 = 1e-5
        let p_l_clean = surface_code_logical_error_rate(3, 1e-4);
        let expected_clean = 0.1 * (0.01_f64).powi(2);
        assert!(
            (p_l_clean - expected_clean).abs() < 1e-17,
            "d=3, p=1e-4: got {:.6e}, expected {:.6e}",
            p_l_clean,
            expected_clean
        );
    }

    // -- Additional tests for thorough coverage --

    #[test]
    fn test_algorithm_spec_display() {
        let alg = AlgorithmSpec::shor_rsa(2048);
        let display = format!("{}", alg);
        assert!(display.contains("Shor RSA-2048"));
        assert!(display.contains("4099"));
    }

    #[test]
    fn test_hardware_model_display() {
        let hw = HardwareModel::superconducting_current();
        let display = format!("{}", hw);
        assert!(display.contains("Superconducting"));
        assert!(display.contains("1.0e-3") || display.contains("1e-3") || display.contains("1.0"));
    }

    #[test]
    fn test_distillation_protocol_display() {
        let proto = DistillationProtocol::Standard15to1;
        assert_eq!(format!("{}", proto), "15-to-1 (standard)");

        let proto2 = DistillationProtocol::TwoLevel15to1;
        assert_eq!(format!("{}", proto2), "15-to-1 (two-level)");
    }

    #[test]
    fn test_two_level_distillation_uses_more_qubits() {
        let alg = AlgorithmSpec::custom("Test", 20, 50_000);
        let hw = HardwareModel::superconducting_current();

        let mut est_standard = ResourceEstimator::new();
        est_standard.distillation_protocol = DistillationProtocol::Standard15to1;
        let result_standard = est_standard.estimate(&alg, &hw);

        let mut est_two_level = ResourceEstimator::new();
        est_two_level.distillation_protocol = DistillationProtocol::TwoLevel15to1;
        let result_two_level = est_two_level.estimate(&alg, &hw);

        // Two-level uses 225 patches vs 15 patches, so factory qubits should be larger.
        assert!(
            result_two_level.factory_qubits > result_standard.factory_qubits,
            "Two-level factory ({}) should use more qubits than standard ({})",
            result_two_level.factory_qubits,
            result_standard.factory_qubits
        );

        // Both should have the same code distance (same algorithm + hardware).
        assert_eq!(
            result_two_level.code_distance, result_standard.code_distance,
            "Same algorithm and hardware should give same code distance"
        );
    }

    #[test]
    fn test_resource_estimate_display_trait() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::custom("Test", 5, 100);
        let hw = HardwareModel::superconducting_current();
        let est = estimator.estimate(&alg, &hw);

        // Display trait should work without panicking and produce content.
        let display = format!("{}", est);
        assert!(!display.is_empty());
        assert!(display.contains("Fault-Tolerant Resource Estimate"));
    }

    #[test]
    fn test_default_estimator() {
        let est = ResourceEstimator::default();
        assert!((est.target_logical_error_rate - 1e-2).abs() < 1e-15);
        assert_eq!(
            est.distillation_protocol,
            DistillationProtocol::Standard15to1
        );
    }

    #[test]
    fn test_very_small_algorithm() {
        // Edge case: 1 logical qubit, 1 T gate.
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::custom("Tiny", 1, 1);
        let hw = HardwareModel::superconducting_current();
        let est = estimator.estimate(&alg, &hw);

        // Should still produce a valid estimate.
        assert!(est.code_distance >= 3);
        assert!(est.total_physical_qubits > 0);
        assert!(est.total_time_seconds > 0.0);
        assert!(est.magic_state_factories >= 1);
    }

    #[test]
    fn test_error_rate_scaling_monotone() {
        // Logical error rate should monotonically decrease with code distance.
        let p_phys = 1e-3;
        let mut prev_error = f64::MAX;
        for d in (3..=21).step_by(2) {
            let p_l = surface_code_logical_error_rate(d, p_phys);
            assert!(
                p_l < prev_error,
                "p_L should decrease: d={} gave {:.2e} >= previous {:.2e}",
                d,
                p_l,
                prev_error
            );
            prev_error = p_l;
        }
    }

    #[test]
    fn test_neutral_atom_target_vs_current() {
        let estimator = ResourceEstimator::new();
        let alg = AlgorithmSpec::custom("Test", 50, 100_000);

        let est_current = estimator.estimate(&alg, &HardwareModel::neutral_atom_current());
        let est_target = estimator.estimate(&alg, &HardwareModel::neutral_atom_target());

        // Target hardware should need fewer qubits.
        assert!(
            est_target.total_physical_qubits < est_current.total_physical_qubits,
            "Target neutral atom ({}) should need fewer qubits than current ({})",
            est_target.total_physical_qubits,
            est_current.total_physical_qubits
        );
    }
}
