//! Magic State Distillation Factory Simulation
//!
//! Simulates the full distillation pipeline that fault-tolerant quantum computers
//! use to produce high-fidelity T gates from noisy ancilla states. Tracks resource
//! costs including physical qubit counts, surface code cycles, and space-time volume.
//!
//! **Background**:
//! Universal fault-tolerant quantum computation requires non-Clifford gates (like T).
//! Clifford gates can be implemented transversally on surface codes, but T gates
//! cannot. Instead, we prepare "magic states" offline via distillation and then
//! inject them into the computation via gate teleportation.
//!
//! **Protocols Implemented**:
//! - **15-to-1**: The original Bravyi-Kitaev protocol. 15 noisy T states yield 1
//!   cleaner T state with cubic error suppression: `p_out = 35 * p_in^3`.
//! - **20-to-4**: Golay-code protocol. 20 noisy states yield 4 cleaner states with
//!   quadratic suppression: `p_out = 20 * p_in^2`.
//! - **Reed-Muller 116-to-12**: For very low target error rates. Aggressive
//!   suppression at higher raw state cost.
//! - **Litinski Compact**: Optimized factory layout that reduces the surface code
//!   patch footprint by exploiting lattice surgery scheduling.
//!
//! **References**:
//! - Bravyi & Kitaev, "Universal quantum computation with ideal Clifford gates and
//!   noisy ancillas", PRA 71, 022316 (2005).
//! - Litinski, "Magic State Distillation: Not as Costly as You Think",
//!   Quantum 3, 205 (2019).
//! - Fowler et al., "Surface codes: Towards practical large-scale quantum
//!   computation", PRA 86, 032324 (2012).
//! - Haah et al., "Magic state distillation with low space overhead and optimal
//!   asymptotic input count", Quantum 1, 31 (2017).

use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Type of magic state being distilled.
///
/// Different state types enable different non-Clifford operations when consumed
/// via gate teleportation.
#[derive(Clone, Debug, PartialEq)]
pub enum MagicStateType {
    /// |T> = (|0> + e^{i pi/4} |1>) / sqrt(2)
    ///
    /// Enables the single-qubit T gate (pi/8 rotation). The workhorse of most
    /// fault-tolerant algorithms; T-gate count dominates resource estimates.
    TState,

    /// CCZ magic state for direct three-qubit CCZ injection.
    ///
    /// Some algorithms benefit from direct CCZ synthesis rather than
    /// decomposing into T gates (which costs 4-7 T gates per CCZ).
    CCZState,

    /// CSS-code-based magic state used in certain distillation protocols.
    ///
    /// Exploits the CSS structure of the underlying error-correcting code
    /// to simplify distillation circuits.
    CSSState,
}

impl fmt::Display for MagicStateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MagicStateType::TState => write!(f, "|T>"),
            MagicStateType::CCZState => write!(f, "|CCZ>"),
            MagicStateType::CSSState => write!(f, "|CSS>"),
        }
    }
}

/// A single magic state with tracked fidelity.
///
/// `fidelity = 1 - error_rate`. After distillation the error rate drops
/// according to the protocol's suppression polynomial.
#[derive(Clone, Debug)]
pub struct MagicState {
    /// Fidelity of the magic state: `1 - error_rate`.
    pub fidelity: f64,
    /// Probability that the state is in the wrong computational branch.
    pub error_rate: f64,
    /// Which type of magic state this represents.
    pub state_type: MagicStateType,
}

impl MagicState {
    /// Create a new magic state with the given error rate.
    ///
    /// # Panics
    /// Panics if `error_rate` is not in [0, 1].
    pub fn new(error_rate: f64, state_type: MagicStateType) -> Self {
        assert!(
            (0.0..=1.0).contains(&error_rate),
            "error_rate must be in [0, 1], got {}",
            error_rate
        );
        Self {
            fidelity: 1.0 - error_rate,
            error_rate,
            state_type,
        }
    }

    /// Create a raw (noisy) T state with the given physical error rate.
    pub fn raw_t_state(physical_error_rate: f64) -> Self {
        Self::new(physical_error_rate, MagicStateType::TState)
    }

    /// Create a perfect (zero-error) T state (for testing).
    pub fn perfect_t_state() -> Self {
        Self::new(0.0, MagicStateType::TState)
    }
}

impl fmt::Display for MagicState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MagicState({}, error={:.2e}, fidelity={:.10})",
            self.state_type, self.error_rate, self.fidelity
        )
    }
}

// ---------------------------------------------------------------------------
// Distillation protocols
// ---------------------------------------------------------------------------

/// Distillation protocol choice.
///
/// Each protocol trades off raw-state consumption, output count, and error
/// suppression strength. Multi-level distillation cascades these to reach
/// arbitrarily low output error rates.
#[derive(Clone, Debug, PartialEq)]
pub enum DistillationProtocol {
    /// **15-to-1** (Bravyi-Kitaev 2005).
    ///
    /// Consumes 15 noisy T states to produce 1 cleaner T state.
    /// Error suppression: `p_out = 35 * p_in^3` (cubic).
    /// This is the original and most widely-analyzed protocol.
    FifteenToOne,

    /// **20-to-4** (Golay-code protocol).
    ///
    /// Consumes 20 noisy T states to produce 4 cleaner T states.
    /// Error suppression: `p_out = 20 * p_in^2` (quadratic).
    /// More efficient per output state at moderate noise levels, but weaker
    /// suppression per level means more levels may be needed for very low
    /// target error rates.
    TwentyToFour,

    /// **Reed-Muller 116-to-12**.
    ///
    /// Uses the punctured Reed-Muller code [[2^m - 1, 1, 2^{m-1} - 1]].
    /// Consumes 116 raw states to produce 12 distilled states.
    /// Error suppression: `p_out = C * p_in^4` with C ~ 1000.
    /// Aggressive suppression makes this attractive for extremely low
    /// target error rates despite the high raw state cost.
    ReedMuller116,

    /// **Litinski Compact** (Litinski 2019).
    ///
    /// Optimized factory layout for 15-to-1 that reduces the surface code
    /// patch count via lattice surgery scheduling tricks. Same error
    /// suppression as 15-to-1 (`p_out = 35 * p_in^3`) but uses roughly
    /// half the physical qubits by overlapping distillation rounds in time.
    LitinskiCompact,
}

impl DistillationProtocol {
    /// Number of raw input states consumed per distillation round.
    pub fn states_in(&self) -> usize {
        match self {
            DistillationProtocol::FifteenToOne => 15,
            DistillationProtocol::TwentyToFour => 20,
            DistillationProtocol::ReedMuller116 => 116,
            DistillationProtocol::LitinskiCompact => 15,
        }
    }

    /// Number of distilled output states produced per round.
    pub fn states_out(&self) -> usize {
        match self {
            DistillationProtocol::FifteenToOne => 1,
            DistillationProtocol::TwentyToFour => 4,
            DistillationProtocol::ReedMuller116 => 12,
            DistillationProtocol::LitinskiCompact => 1,
        }
    }

    /// Compute the output error rate from a single distillation round.
    ///
    /// Each protocol has a characteristic suppression polynomial:
    /// - 15-to-1: `35 * p^3`
    /// - 20-to-4: `20 * p^2`
    /// - Reed-Muller 116: `1000 * p^4`
    /// - Litinski Compact: `35 * p^3` (same math, smaller footprint)
    pub fn output_error(&self, input_error: f64) -> f64 {
        let result = match self {
            DistillationProtocol::FifteenToOne => {
                // Bravyi-Kitaev: C(15,3) = 455 but dominant term is 35*p^3
                // The leading-order coefficient comes from the 15-qubit
                // Reed-Muller code's weight-3 error paths.
                35.0 * input_error.powi(3)
            }
            DistillationProtocol::TwentyToFour => {
                // Golay code: quadratic suppression with ~20 weight-2 paths.
                20.0 * input_error.powi(2)
            }
            DistillationProtocol::ReedMuller116 => {
                // Higher-order Reed-Muller: quartic suppression.
                // Coefficient from Haah et al. analysis.
                1000.0 * input_error.powi(4)
            }
            DistillationProtocol::LitinskiCompact => {
                // Same distillation math as 15-to-1, just a more efficient
                // surface code layout.
                35.0 * input_error.powi(3)
            }
        };
        // Output error rate cannot exceed 1.0.
        result.min(1.0)
    }

    /// Error suppression exponent (the polynomial degree).
    pub fn suppression_order(&self) -> u32 {
        match self {
            DistillationProtocol::FifteenToOne => 3,
            DistillationProtocol::TwentyToFour => 2,
            DistillationProtocol::ReedMuller116 => 4,
            DistillationProtocol::LitinskiCompact => 3,
        }
    }

    /// Number of surface code patches required per factory unit.
    ///
    /// A "patch" is one logical qubit's worth of physical qubits
    /// (2 * d^2 physical qubits for a distance-d rotated surface code).
    pub fn factory_patches(&self) -> usize {
        match self {
            DistillationProtocol::FifteenToOne => 16,     // 15 ancilla + 1 output
            DistillationProtocol::TwentyToFour => 24,     // 20 ancilla + 4 output
            DistillationProtocol::ReedMuller116 => 128,   // large code block
            // Litinski's key insight: time-optimal scheduling halves the patch count.
            DistillationProtocol::LitinskiCompact => 8,
        }
    }

    /// Number of surface code cycles per distillation round.
    ///
    /// Each round involves syndrome extraction and Clifford circuits that
    /// take O(d) surface code cycles where d is the code distance.
    /// The multiplier depends on the protocol's circuit depth.
    pub fn cycles_multiplier(&self) -> usize {
        match self {
            DistillationProtocol::FifteenToOne => 10,
            DistillationProtocol::TwentyToFour => 12,
            DistillationProtocol::ReedMuller116 => 20,
            // Litinski schedules operations in parallel, reducing depth.
            DistillationProtocol::LitinskiCompact => 5,
        }
    }

    /// Threshold input error rate above which distillation cannot help.
    ///
    /// If the input error exceeds this, the output error will be worse than
    /// the input. Derived from solving `protocol(p) = p`.
    pub fn threshold(&self) -> f64 {
        match self {
            // 35 * p^3 = p => p^2 = 1/35 => p ~ 0.169
            DistillationProtocol::FifteenToOne => (1.0 / 35.0_f64).sqrt(),
            // 20 * p^2 = p => p = 1/20 = 0.05
            DistillationProtocol::TwentyToFour => 1.0 / 20.0,
            // 1000 * p^4 = p => p^3 = 1/1000 => p = 0.1
            DistillationProtocol::ReedMuller116 => (1.0 / 1000.0_f64).cbrt(),
            DistillationProtocol::LitinskiCompact => (1.0 / 35.0_f64).sqrt(),
        }
    }
}

impl fmt::Display for DistillationProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistillationProtocol::FifteenToOne => write!(f, "15-to-1"),
            DistillationProtocol::TwentyToFour => write!(f, "20-to-4"),
            DistillationProtocol::ReedMuller116 => write!(f, "RM 116-to-12"),
            DistillationProtocol::LitinskiCompact => write!(f, "Litinski Compact"),
        }
    }
}

// ---------------------------------------------------------------------------
// Factory and results
// ---------------------------------------------------------------------------

/// Detailed information about a single distillation level.
#[derive(Clone, Debug)]
pub struct LevelDetail {
    /// Which level (0-indexed from the bottom / noisiest).
    pub level: usize,
    /// Error rate of the input states at this level.
    pub input_error_rate: f64,
    /// Error rate of the output states after this level's distillation.
    pub output_error_rate: f64,
    /// Number of states consumed at this level.
    pub states_in: usize,
    /// Number of states produced at this level.
    pub states_out: usize,
    /// Physical qubits used by this level's factory.
    pub qubits_per_factory: usize,
    /// Surface code cycles required by this level.
    pub cycles: usize,
}

/// Result of running a distillation factory.
#[derive(Clone, Debug)]
pub struct FactoryResult {
    /// The output magic states (all at the final fidelity).
    pub output_states: Vec<MagicState>,
    /// Error rate of the output states.
    pub output_error_rate: f64,
    /// Total number of raw (level-0) states consumed across all levels.
    pub total_raw_states_consumed: usize,
    /// Total physical qubits required (summed across all levels).
    pub total_physical_qubits: usize,
    /// Total distillation time in surface code cycles.
    pub distillation_time_cycles: usize,
    /// Space-time volume: `total_physical_qubits * distillation_time_cycles`.
    pub space_time_volume: usize,
    /// Number of distillation levels used.
    pub levels_used: usize,
    /// Per-level breakdown.
    pub level_details: Vec<LevelDetail>,
}

impl fmt::Display for FactoryResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Magic State Factory Result ===")?;
        writeln!(f, "Output error rate:       {:.2e}", self.output_error_rate)?;
        writeln!(f, "Output states:           {}", self.output_states.len())?;
        writeln!(f, "Raw states consumed:     {}", self.total_raw_states_consumed)?;
        writeln!(f, "Physical qubits:         {}", self.total_physical_qubits)?;
        writeln!(f, "Distillation cycles:     {}", self.distillation_time_cycles)?;
        writeln!(f, "Space-time volume:       {}", self.space_time_volume)?;
        writeln!(f, "Levels:                  {}", self.levels_used)?;
        writeln!(f, "--- Level Details ---")?;
        for detail in &self.level_details {
            writeln!(
                f,
                "  Level {}: {:.2e} -> {:.2e}  ({} in, {} out, {} qubits, {} cycles)",
                detail.level,
                detail.input_error_rate,
                detail.output_error_rate,
                detail.states_in,
                detail.states_out,
                detail.qubits_per_factory,
                detail.cycles,
            )?;
        }
        Ok(())
    }
}

/// Resource estimate for a given target error rate.
#[derive(Clone, Debug)]
pub struct ResourceEstimate {
    /// The target output error rate.
    pub target_error_rate: f64,
    /// Which protocol is recommended.
    pub recommended_protocol: DistillationProtocol,
    /// How many distillation levels are needed.
    pub recommended_levels: usize,
    /// Total physical qubits for the factory.
    pub total_physical_qubits: usize,
    /// Estimated T gates producible per second (assuming 1 us cycle time).
    pub t_gates_per_second: f64,
    /// Number of surface code patches in the factory footprint.
    pub factory_footprint_patches: usize,
}

impl fmt::Display for ResourceEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Resource Estimate ===")?;
        writeln!(f, "Target error:         {:.2e}", self.target_error_rate)?;
        writeln!(f, "Protocol:             {}", self.recommended_protocol)?;
        writeln!(f, "Levels:               {}", self.recommended_levels)?;
        writeln!(f, "Physical qubits:      {}", self.total_physical_qubits)?;
        writeln!(f, "T gates/second:       {:.0}", self.t_gates_per_second)?;
        writeln!(f, "Factory patches:      {}", self.factory_footprint_patches)?;
        Ok(())
    }
}

/// A magic state distillation factory.
///
/// Encapsulates the protocol choice, number of distillation levels, surface code
/// distance, and physical error rate. Call [`MagicStateFactory::distill`] to run
/// the simulation and get a [`FactoryResult`] with full resource accounting.
///
/// # Example
/// ```
/// use nqpu_metal::magic_state_factory::*;
///
/// let factory = MagicStateFactory::new(
///     DistillationProtocol::FifteenToOne,
///     2,      // two distillation levels
///     17,     // code distance 17
///     1e-3,   // physical error rate 0.1%
/// );
/// let result = factory.distill();
/// assert!(result.output_error_rate < 1e-10);
/// ```
pub struct MagicStateFactory {
    /// Which distillation protocol to use.
    protocol: DistillationProtocol,
    /// Number of distillation levels (cascaded rounds).
    levels: usize,
    /// Surface code distance (determines physical qubit overhead).
    code_distance: usize,
    /// Physical qubit error rate (gate error probability).
    physical_error_rate: f64,
}

impl MagicStateFactory {
    /// Create a new factory with explicit parameters.
    ///
    /// # Arguments
    /// * `protocol` - Which distillation protocol to use.
    /// * `levels` - Number of cascaded distillation levels.
    /// * `code_distance` - Surface code distance d (must be odd and >= 3).
    /// * `physical_error_rate` - Physical gate error probability.
    ///
    /// # Panics
    /// Panics if `code_distance < 3` or `physical_error_rate` is not in (0, 1).
    pub fn new(
        protocol: DistillationProtocol,
        levels: usize,
        code_distance: usize,
        physical_error_rate: f64,
    ) -> Self {
        assert!(code_distance >= 3, "code_distance must be >= 3, got {}", code_distance);
        assert!(
            physical_error_rate > 0.0 && physical_error_rate < 1.0,
            "physical_error_rate must be in (0, 1), got {}",
            physical_error_rate
        );
        Self {
            protocol,
            levels,
            code_distance,
            physical_error_rate,
        }
    }

    /// Number of physical qubits per surface code patch.
    ///
    /// For a rotated surface code of distance d, we need `2 * d^2` physical
    /// qubits (d^2 data qubits + d^2 - 1 syndrome qubits, rounded up).
    pub fn qubits_per_patch(&self) -> usize {
        2 * self.code_distance * self.code_distance
    }

    /// Run the distillation simulation.
    ///
    /// Cascades through `self.levels` rounds of distillation, tracking error
    /// suppression and resource costs at each level. The first level consumes
    /// raw (noisy) T states; subsequent levels consume the output of the
    /// previous level.
    pub fn distill(&self) -> FactoryResult {
        let mut current_error = self.physical_error_rate;
        let mut level_details = Vec::with_capacity(self.levels);
        let mut total_raw = 1_usize; // multiplicative accumulator
        let mut total_qubits = 0_usize;
        let mut total_cycles = 0_usize;

        for level in 0..self.levels {
            let input_error = current_error;
            let output_error = self.protocol.output_error(input_error);

            let states_in = self.protocol.states_in();
            let states_out = self.protocol.states_out();

            // Physical qubits for this level's factory.
            let patches = self.protocol.factory_patches();
            let qubits = patches * self.qubits_per_patch();

            // Cycles for this distillation round.
            let cycles = self.protocol.cycles_multiplier() * self.code_distance;

            // How many raw states feed into this level (cascade multiplication).
            // Level 0 directly consumes raw states. Level k consumes the output
            // of level k-1, so the raw-state cost multiplies.
            if level == 0 {
                total_raw = states_in;
            } else {
                // Each state entering this level required (states_in / states_out)
                // of the previous level's raw cost, but we account for the ratio.
                // For a full cascade: total_raw at level k is
                //   product_{j=0}^{k} ceil(states_in / states_out) for each level.
                // We simplify to the dominant term: states_in multiplied upward.
                total_raw *= states_in;
            }

            total_qubits += qubits;
            total_cycles += cycles;

            level_details.push(LevelDetail {
                level,
                input_error_rate: input_error,
                output_error_rate: output_error,
                states_in,
                states_out,
                qubits_per_factory: qubits,
                cycles,
            });

            current_error = output_error;
        }

        // Produce output states at the final fidelity.
        let final_states_out = self.protocol.states_out();
        let output_states: Vec<MagicState> = (0..final_states_out)
            .map(|_| MagicState::new(current_error, MagicStateType::TState))
            .collect();

        let space_time_volume = total_qubits * total_cycles;

        FactoryResult {
            output_states,
            output_error_rate: current_error,
            total_raw_states_consumed: total_raw,
            total_physical_qubits: total_qubits,
            distillation_time_cycles: total_cycles,
            space_time_volume,
            levels_used: self.levels,
            level_details,
        }
    }

    /// Estimate the resources needed to reach a target error rate.
    ///
    /// Evaluates all four protocols and selects the one with the smallest
    /// space-time volume (physical qubits x cycles) that achieves the target.
    ///
    /// # Arguments
    /// * `target_error` - Desired output error rate (e.g., 1e-15).
    /// * `physical_error` - Physical gate error rate (e.g., 1e-3).
    /// * `code_distance` - Surface code distance to assume.
    ///
    /// # Returns
    /// A [`ResourceEstimate`] with the recommended protocol and costs.
    pub fn estimate_resources(
        target_error: f64,
        physical_error: f64,
        code_distance: usize,
    ) -> ResourceEstimate {
        let protocols = [
            DistillationProtocol::FifteenToOne,
            DistillationProtocol::TwentyToFour,
            DistillationProtocol::ReedMuller116,
            DistillationProtocol::LitinskiCompact,
        ];

        let mut best: Option<(DistillationProtocol, usize, FactoryResult)> = None;

        for protocol in &protocols {
            // Skip if physical error is above the protocol's threshold.
            if physical_error >= protocol.threshold() {
                continue;
            }

            // Find the minimum number of levels needed.
            let max_levels = 10; // sanity bound
            for levels in 1..=max_levels {
                let factory = MagicStateFactory::new(
                    protocol.clone(),
                    levels,
                    code_distance,
                    physical_error,
                );
                let result = factory.distill();

                if result.output_error_rate <= target_error {
                    let dominated = best.as_ref().map_or(false, |(_, _, prev)| {
                        result.space_time_volume >= prev.space_time_volume
                    });
                    if !dominated {
                        best = Some((protocol.clone(), levels, result));
                    }
                    break; // found min levels for this protocol
                }
            }
        }

        // Fall back to 15-to-1 with max levels if nothing found (extreme case).
        let (rec_protocol, rec_levels, rec_result) = best.unwrap_or_else(|| {
            let factory = MagicStateFactory::new(
                DistillationProtocol::FifteenToOne,
                10,
                code_distance,
                physical_error,
            );
            let result = factory.distill();
            (DistillationProtocol::FifteenToOne, 10, result)
        });

        let patches = rec_protocol.factory_patches() * rec_levels;

        // Estimate T gates per second.
        // Assume 1 microsecond per surface code cycle (typical superconducting).
        let cycle_time_us = 1.0;
        let total_time_us =
            rec_result.distillation_time_cycles as f64 * cycle_time_us;
        let t_gates_per_second = if total_time_us > 0.0 {
            (rec_protocol.states_out() as f64 / total_time_us) * 1e6
        } else {
            0.0
        };

        ResourceEstimate {
            target_error_rate: target_error,
            recommended_protocol: rec_protocol,
            recommended_levels: rec_levels,
            total_physical_qubits: rec_result.total_physical_qubits,
            t_gates_per_second,
            factory_footprint_patches: patches,
        }
    }

    /// Compute just the output error rate without full resource accounting.
    ///
    /// Useful for quick parameter sweeps.
    pub fn output_error_rate(&self) -> f64 {
        let mut error = self.physical_error_rate;
        for _ in 0..self.levels {
            error = self.protocol.output_error(error);
        }
        error
    }

    /// Check whether the physical error rate is below the protocol's threshold.
    pub fn is_below_threshold(&self) -> bool {
        self.physical_error_rate < self.protocol.threshold()
    }
}

// ---------------------------------------------------------------------------
// Multi-protocol cascade factory
// ---------------------------------------------------------------------------

/// A multi-protocol cascade where different levels can use different protocols.
///
/// This models the common practice of using a cheap protocol (e.g., 20-to-4)
/// for early levels where the error rate is high, then switching to a more
/// aggressive protocol (e.g., 15-to-1) for the final level.
pub struct CascadeFactory {
    /// Each entry is (protocol, code_distance) for that level.
    pub levels: Vec<(DistillationProtocol, usize)>,
    /// Physical error rate of raw states.
    pub physical_error_rate: f64,
}

impl CascadeFactory {
    /// Create a cascade factory with per-level protocol choices.
    pub fn new(
        levels: Vec<(DistillationProtocol, usize)>,
        physical_error_rate: f64,
    ) -> Self {
        assert!(!levels.is_empty(), "cascade must have at least one level");
        Self {
            levels,
            physical_error_rate,
        }
    }

    /// Run the cascade distillation and return a full result.
    pub fn distill(&self) -> FactoryResult {
        let mut current_error = self.physical_error_rate;
        let mut level_details = Vec::with_capacity(self.levels.len());
        let mut total_raw = 1_usize;
        let mut total_qubits = 0_usize;
        let mut total_cycles = 0_usize;

        for (level_idx, (protocol, distance)) in self.levels.iter().enumerate() {
            let input_error = current_error;
            let output_error = protocol.output_error(input_error);

            let states_in = protocol.states_in();
            let states_out = protocol.states_out();

            let qubits_per_patch = 2 * distance * distance;
            let patches = protocol.factory_patches();
            let qubits = patches * qubits_per_patch;
            let cycles = protocol.cycles_multiplier() * distance;

            if level_idx == 0 {
                total_raw = states_in;
            } else {
                total_raw *= states_in;
            }

            total_qubits += qubits;
            total_cycles += cycles;

            level_details.push(LevelDetail {
                level: level_idx,
                input_error_rate: input_error,
                output_error_rate: output_error,
                states_in,
                states_out,
                qubits_per_factory: qubits,
                cycles,
            });

            current_error = output_error;
        }

        // Use the last protocol's output count.
        let final_protocol = &self.levels.last().unwrap().0;
        let final_states_out = final_protocol.states_out();
        let output_states: Vec<MagicState> = (0..final_states_out)
            .map(|_| MagicState::new(current_error, MagicStateType::TState))
            .collect();

        let space_time_volume = total_qubits * total_cycles;

        FactoryResult {
            output_states,
            output_error_rate: current_error,
            total_raw_states_consumed: total_raw,
            total_physical_qubits: total_qubits,
            distillation_time_cycles: total_cycles,
            space_time_volume,
            levels_used: self.levels.len(),
            level_details,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Magic state creation --

    #[test]
    fn test_magic_state_creation() {
        let state = MagicState::new(0.01, MagicStateType::TState);
        assert_eq!(state.state_type, MagicStateType::TState);
        assert!((state.error_rate - 0.01).abs() < 1e-15);
        assert!((state.fidelity - 0.99).abs() < 1e-15);

        let raw = MagicState::raw_t_state(0.001);
        assert_eq!(raw.state_type, MagicStateType::TState);
        assert!((raw.error_rate - 0.001).abs() < 1e-15);

        let perfect = MagicState::perfect_t_state();
        assert!((perfect.error_rate).abs() < 1e-15);
        assert!((perfect.fidelity - 1.0).abs() < 1e-15);

        let ccz = MagicState::new(0.05, MagicStateType::CCZState);
        assert_eq!(ccz.state_type, MagicStateType::CCZState);

        let css = MagicState::new(0.02, MagicStateType::CSSState);
        assert_eq!(css.state_type, MagicStateType::CSSState);
    }

    #[test]
    #[should_panic(expected = "error_rate must be in [0, 1]")]
    fn test_magic_state_invalid_error_rate() {
        MagicState::new(1.5, MagicStateType::TState);
    }

    // -- Error suppression formulas --

    #[test]
    fn test_fifteen_to_one_error_suppression() {
        let protocol = DistillationProtocol::FifteenToOne;
        let p_in = 0.001; // 0.1% physical error rate
        let p_out = protocol.output_error(p_in);

        // p_out = 35 * (0.001)^3 = 35 * 1e-9 = 3.5e-8
        let expected = 35.0 * p_in.powi(3);
        assert!(
            (p_out - expected).abs() < 1e-20,
            "15-to-1 output error mismatch: got {:.2e}, expected {:.2e}",
            p_out,
            expected
        );

        // Verify cubic suppression: doubling input should ~8x the output.
        let p_in_2x = 0.002;
        let p_out_2x = protocol.output_error(p_in_2x);
        let ratio = p_out_2x / p_out;
        assert!(
            (ratio - 8.0).abs() < 0.01,
            "cubic suppression ratio should be ~8, got {:.4}",
            ratio
        );
    }

    #[test]
    fn test_twenty_to_four_error_suppression() {
        let protocol = DistillationProtocol::TwentyToFour;
        let p_in = 0.001;
        let p_out = protocol.output_error(p_in);

        // p_out = 20 * (0.001)^2 = 20 * 1e-6 = 2e-5
        let expected = 20.0 * p_in.powi(2);
        assert!(
            (p_out - expected).abs() < 1e-15,
            "20-to-4 output error mismatch: got {:.2e}, expected {:.2e}",
            p_out,
            expected
        );

        // Quadratic: doubling input should ~4x the output.
        let p_in_2x = 0.002;
        let p_out_2x = protocol.output_error(p_in_2x);
        let ratio = p_out_2x / p_out;
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "quadratic suppression ratio should be ~4, got {:.4}",
            ratio
        );
    }

    #[test]
    fn test_reed_muller_error_suppression() {
        let protocol = DistillationProtocol::ReedMuller116;
        let p_in = 0.01;
        let p_out = protocol.output_error(p_in);

        // p_out = 1000 * (0.01)^4 = 1000 * 1e-8 = 1e-5
        let expected = 1000.0 * p_in.powi(4);
        assert!(
            (p_out - expected).abs() < 1e-15,
            "RM 116 output error mismatch: got {:.2e}, expected {:.2e}",
            p_out,
            expected
        );

        // Quartic: doubling input should ~16x the output.
        let p_in_2x = 0.02;
        let p_out_2x = protocol.output_error(p_in_2x);
        let ratio = p_out_2x / p_out;
        assert!(
            (ratio - 16.0).abs() < 0.1,
            "quartic suppression ratio should be ~16, got {:.4}",
            ratio
        );
    }

    // -- Single-level distillation --

    #[test]
    fn test_single_level_distillation() {
        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            1,
            15,
            1e-3,
        );
        let result = factory.distill();

        // Single level of 15-to-1 with p = 1e-3:
        // p_out = 35 * (1e-3)^3 = 3.5e-8
        let expected_error = 35.0 * (1e-3_f64).powi(3);
        assert!(
            (result.output_error_rate - expected_error).abs() < 1e-20,
            "single-level error: got {:.2e}, expected {:.2e}",
            result.output_error_rate,
            expected_error
        );

        assert_eq!(result.levels_used, 1);
        assert_eq!(result.output_states.len(), 1); // 15-to-1 produces 1
        assert_eq!(result.total_raw_states_consumed, 15);
        assert_eq!(result.level_details.len(), 1);
    }

    // -- Multi-level distillation --

    #[test]
    fn test_multi_level_distillation() {
        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            2,
            17,
            1e-3,
        );
        let result = factory.distill();

        // Level 0: p_out = 35 * (1e-3)^3 = 3.5e-8
        // Level 1: p_out = 35 * (3.5e-8)^3 = 35 * 4.2875e-23 ~ 1.5e-21
        let p1 = 35.0 * (1e-3_f64).powi(3);
        let p2 = 35.0 * p1.powi(3);
        assert!(
            (result.output_error_rate - p2).abs() / p2 < 1e-6,
            "two-level error: got {:.2e}, expected {:.2e}",
            result.output_error_rate,
            p2
        );

        assert_eq!(result.levels_used, 2);
        assert_eq!(result.level_details.len(), 2);

        // Level details should show cascading error rates.
        let d0 = &result.level_details[0];
        let d1 = &result.level_details[1];
        assert!((d0.input_error_rate - 1e-3).abs() < 1e-15);
        assert!((d0.output_error_rate - p1).abs() < 1e-20);
        assert!((d1.input_error_rate - p1).abs() < 1e-20);
        assert!((d1.output_error_rate - p2).abs() / p2 < 1e-6);

        // Raw states: 15 * 15 = 225 for two cascaded 15-to-1 levels.
        assert_eq!(result.total_raw_states_consumed, 225);
    }

    // -- Resource estimation --

    #[test]
    fn test_resource_estimation() {
        let estimate = MagicStateFactory::estimate_resources(
            1e-15,  // very low target
            1e-3,   // typical physical error
            17,     // code distance
        );

        // Should achieve the target.
        // The recommended protocol should reach the target error rate.
        let factory = MagicStateFactory::new(
            estimate.recommended_protocol.clone(),
            estimate.recommended_levels,
            17,
            1e-3,
        );
        let result = factory.distill();
        assert!(
            result.output_error_rate <= 1e-15,
            "estimate should achieve target: got {:.2e}",
            result.output_error_rate
        );

        assert!(estimate.total_physical_qubits > 0);
        assert!(estimate.t_gates_per_second > 0.0);
        assert!(estimate.factory_footprint_patches > 0);
    }

    #[test]
    fn test_resource_estimation_moderate_target() {
        let estimate = MagicStateFactory::estimate_resources(
            1e-6,   // moderate target
            1e-3,   // typical physical error
            13,     // smaller code distance
        );

        // At least one level should suffice for moderate targets with some protocols.
        assert!(estimate.recommended_levels >= 1);
        assert!(estimate.total_physical_qubits > 0);
    }

    // -- Physical qubit count --

    #[test]
    fn test_physical_qubit_count() {
        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            1,
            7,
            1e-3,
        );

        // Qubits per patch: 2 * 7^2 = 98.
        assert_eq!(factory.qubits_per_patch(), 98);

        let result = factory.distill();
        let d = &result.level_details[0];

        // 15-to-1 uses 16 patches, each patch = 98 qubits.
        let expected_qubits = 16 * 98;
        assert_eq!(d.qubits_per_factory, expected_qubits);
        assert_eq!(result.total_physical_qubits, expected_qubits);
    }

    // -- Distillation time --

    #[test]
    fn test_distillation_time() {
        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            1,
            11,
            1e-3,
        );
        let result = factory.distill();

        // Cycles = cycles_multiplier * code_distance = 10 * 11 = 110.
        let expected_cycles = 10 * 11;
        assert_eq!(result.distillation_time_cycles, expected_cycles);
    }

    // -- Space-time volume --

    #[test]
    fn test_space_time_volume() {
        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            1,
            7,
            1e-3,
        );
        let result = factory.distill();

        // qubits = 16 * 2 * 49 = 1568
        // cycles = 10 * 7 = 70
        // STV = 1568 * 70 = 109760
        let expected_stv = result.total_physical_qubits * result.distillation_time_cycles;
        assert_eq!(result.space_time_volume, expected_stv);
        assert_eq!(result.space_time_volume, 1568 * 70);
    }

    // -- Level details --

    #[test]
    fn test_level_details() {
        let factory = MagicStateFactory::new(
            DistillationProtocol::TwentyToFour,
            3,
            13,
            1e-2,
        );
        let result = factory.distill();

        assert_eq!(result.level_details.len(), 3);

        // Each level should show decreasing error rates.
        for i in 0..result.level_details.len() {
            let d = &result.level_details[i];
            assert_eq!(d.level, i);
            assert!(
                d.output_error_rate < d.input_error_rate,
                "level {} should reduce error: {:.2e} -> {:.2e}",
                i,
                d.input_error_rate,
                d.output_error_rate
            );
            assert_eq!(d.states_in, 20);
            assert_eq!(d.states_out, 4);
        }

        // Verify cascading: output of level i = input of level i+1.
        for i in 0..result.level_details.len() - 1 {
            let out_i = result.level_details[i].output_error_rate;
            let in_next = result.level_details[i + 1].input_error_rate;
            assert!(
                (out_i - in_next).abs() < 1e-30,
                "level {} output {:.2e} should equal level {} input {:.2e}",
                i,
                out_i,
                i + 1,
                in_next
            );
        }
    }

    // -- Error threshold --

    #[test]
    fn test_error_threshold() {
        // 15-to-1 threshold: sqrt(1/35) ~ 0.169
        let threshold_15 = DistillationProtocol::FifteenToOne.threshold();
        assert!(
            (threshold_15 - (1.0 / 35.0_f64).sqrt()).abs() < 1e-10,
            "15-to-1 threshold: got {:.6}, expected {:.6}",
            threshold_15,
            (1.0 / 35.0_f64).sqrt()
        );

        // Below threshold: error should decrease.
        let p_below = 0.1;
        assert!(p_below < threshold_15);
        let p_out = DistillationProtocol::FifteenToOne.output_error(p_below);
        assert!(
            p_out < p_below,
            "below threshold ({:.4}), error should decrease: {:.4e} -> {:.4e}",
            threshold_15,
            p_below,
            p_out
        );

        // 20-to-4 threshold: 1/20 = 0.05
        let threshold_20 = DistillationProtocol::TwentyToFour.threshold();
        assert!(
            (threshold_20 - 0.05).abs() < 1e-10,
            "20-to-4 threshold should be 0.05, got {:.6}",
            threshold_20
        );

        // At threshold, output should equal input (fixed point).
        let p_at_threshold = threshold_20;
        let p_out_at = DistillationProtocol::TwentyToFour.output_error(p_at_threshold);
        assert!(
            (p_out_at - p_at_threshold).abs() < 1e-10,
            "at threshold, output should equal input: {:.6e} vs {:.6e}",
            p_out_at,
            p_at_threshold
        );

        // Factory below-threshold check.
        let factory_ok = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            1,
            7,
            0.001,
        );
        assert!(factory_ok.is_below_threshold());

        let factory_bad = MagicStateFactory::new(
            DistillationProtocol::TwentyToFour,
            1,
            7,
            0.04, // below 0.05 threshold
        );
        assert!(factory_bad.is_below_threshold());
    }

    // -- Factory comparison across protocols --

    #[test]
    fn test_factory_comparison() {
        let p_in = 1e-3;
        let distance = 13;

        let protocols = [
            DistillationProtocol::FifteenToOne,
            DistillationProtocol::TwentyToFour,
            DistillationProtocol::ReedMuller116,
            DistillationProtocol::LitinskiCompact,
        ];

        let mut results: Vec<(DistillationProtocol, FactoryResult)> = Vec::new();

        for proto in &protocols {
            let factory = MagicStateFactory::new(proto.clone(), 1, distance, p_in);
            let result = factory.distill();
            results.push((proto.clone(), result));
        }

        // Litinski Compact should use fewer qubits than standard 15-to-1.
        let qubits_15to1 = results
            .iter()
            .find(|(p, _)| *p == DistillationProtocol::FifteenToOne)
            .unwrap()
            .1
            .total_physical_qubits;
        let qubits_litinski = results
            .iter()
            .find(|(p, _)| *p == DistillationProtocol::LitinskiCompact)
            .unwrap()
            .1
            .total_physical_qubits;
        assert!(
            qubits_litinski < qubits_15to1,
            "Litinski ({}) should use fewer qubits than 15-to-1 ({})",
            qubits_litinski,
            qubits_15to1
        );

        // 15-to-1 and Litinski should have the same output error (same math).
        let error_15to1 = results
            .iter()
            .find(|(p, _)| *p == DistillationProtocol::FifteenToOne)
            .unwrap()
            .1
            .output_error_rate;
        let error_litinski = results
            .iter()
            .find(|(p, _)| *p == DistillationProtocol::LitinskiCompact)
            .unwrap()
            .1
            .output_error_rate;
        assert!(
            (error_15to1 - error_litinski).abs() < 1e-30,
            "15-to-1 and Litinski should produce same error: {:.2e} vs {:.2e}",
            error_15to1,
            error_litinski
        );

        // 20-to-4 should produce MORE output states per round.
        let states_20to4 = results
            .iter()
            .find(|(p, _)| *p == DistillationProtocol::TwentyToFour)
            .unwrap()
            .1
            .output_states
            .len();
        let states_15to1 = results
            .iter()
            .find(|(p, _)| *p == DistillationProtocol::FifteenToOne)
            .unwrap()
            .1
            .output_states
            .len();
        assert!(
            states_20to4 > states_15to1,
            "20-to-4 should produce more states: {} vs {}",
            states_20to4,
            states_15to1
        );

        // All protocols should reduce error below input for p_in = 1e-3.
        for (proto, result) in &results {
            assert!(
                result.output_error_rate < p_in,
                "{} should reduce error: {:.2e} < {:.2e}",
                proto,
                result.output_error_rate,
                p_in
            );
        }
    }

    // -- Litinski compact efficiency --

    #[test]
    fn test_litinski_compact_efficiency() {
        let p_in = 1e-3;
        let distance = 17;

        let standard = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            2,
            distance,
            p_in,
        );
        let compact = MagicStateFactory::new(
            DistillationProtocol::LitinskiCompact,
            2,
            distance,
            p_in,
        );

        let r_std = standard.distill();
        let r_cmp = compact.distill();

        // Same output error (same suppression polynomial).
        assert!(
            (r_std.output_error_rate - r_cmp.output_error_rate).abs() < 1e-30,
            "same error: {:.2e} vs {:.2e}",
            r_std.output_error_rate,
            r_cmp.output_error_rate
        );

        // Litinski uses fewer physical qubits (8 patches vs 16).
        assert!(
            r_cmp.total_physical_qubits < r_std.total_physical_qubits,
            "Litinski qubits ({}) < standard ({})",
            r_cmp.total_physical_qubits,
            r_std.total_physical_qubits
        );

        // Litinski also has fewer cycles (multiplier 5 vs 10).
        assert!(
            r_cmp.distillation_time_cycles < r_std.distillation_time_cycles,
            "Litinski cycles ({}) < standard ({})",
            r_cmp.distillation_time_cycles,
            r_std.distillation_time_cycles
        );

        // Space-time volume: Litinski should be substantially better.
        // Patches: 8/16 = 0.5x. Cycles: 5/10 = 0.5x. Combined: 0.25x.
        let ratio = r_cmp.space_time_volume as f64 / r_std.space_time_volume as f64;
        assert!(
            ratio < 0.30,
            "Litinski STV should be ~0.25x of standard, got {:.2}x",
            ratio
        );
    }

    // -- Cascade factory --

    #[test]
    fn test_cascade_factory() {
        // Use 20-to-4 for level 0 (cheap quadratic), then 15-to-1 for level 1.
        let cascade = CascadeFactory::new(
            vec![
                (DistillationProtocol::TwentyToFour, 13),
                (DistillationProtocol::FifteenToOne, 17),
            ],
            1e-3,
        );
        let result = cascade.distill();

        // Level 0: 20 * (1e-3)^2 = 2e-5
        let p1 = 20.0 * (1e-3_f64).powi(2);
        // Level 1: 35 * (2e-5)^3 = 35 * 8e-15 = 2.8e-13
        let p2 = 35.0 * p1.powi(3);

        assert!(
            (result.output_error_rate - p2).abs() / p2 < 1e-6,
            "cascade error: got {:.2e}, expected {:.2e}",
            result.output_error_rate,
            p2
        );

        assert_eq!(result.levels_used, 2);
        // Final output count should match the last protocol (15-to-1 = 1).
        assert_eq!(result.output_states.len(), 1);
    }

    // -- Output error rate shortcut --

    #[test]
    fn test_output_error_rate_shortcut() {
        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            2,
            17,
            1e-3,
        );

        let quick = factory.output_error_rate();
        let full = factory.distill().output_error_rate;

        assert!(
            (quick - full).abs() / full < 1e-10,
            "shortcut should match full: {:.2e} vs {:.2e}",
            quick,
            full
        );
    }

    // -- Display trait coverage --

    #[test]
    fn test_display_traits() {
        let state = MagicState::new(0.01, MagicStateType::TState);
        let display = format!("{}", state);
        assert!(display.contains("|T>"));
        assert!(display.contains("error="));

        let proto = DistillationProtocol::FifteenToOne;
        assert_eq!(format!("{}", proto), "15-to-1");

        let proto_lit = DistillationProtocol::LitinskiCompact;
        assert_eq!(format!("{}", proto_lit), "Litinski Compact");

        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            1,
            7,
            1e-3,
        );
        let result = factory.distill();
        let result_display = format!("{}", result);
        assert!(result_display.contains("Magic State Factory Result"));
        assert!(result_display.contains("Level 0"));
    }

    // -- Suppression order --

    #[test]
    fn test_suppression_order() {
        assert_eq!(DistillationProtocol::FifteenToOne.suppression_order(), 3);
        assert_eq!(DistillationProtocol::TwentyToFour.suppression_order(), 2);
        assert_eq!(DistillationProtocol::ReedMuller116.suppression_order(), 4);
        assert_eq!(DistillationProtocol::LitinskiCompact.suppression_order(), 3);
    }

    // -- Edge case: very low physical error --

    #[test]
    fn test_very_low_physical_error() {
        // Even a single level of 15-to-1 at p = 1e-4 gives extremely low output.
        let factory = MagicStateFactory::new(
            DistillationProtocol::FifteenToOne,
            1,
            7,
            1e-4,
        );
        let result = factory.distill();

        // 35 * (1e-4)^3 = 3.5e-11
        let expected = 35.0 * (1e-4_f64).powi(3);
        assert!(
            (result.output_error_rate - expected).abs() < 1e-23,
            "very low input: got {:.2e}, expected {:.2e}",
            result.output_error_rate,
            expected
        );
    }
}
