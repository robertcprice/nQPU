//! Dynamical Decoupling (DD) -- Circuit Transformation Pass
//!
//! Implements circuit-level dynamical decoupling passes that insert
//! identity-equivalent pulse sequences on idling qubits to suppress
//! decoherence noise.  The module provides:
//!
//! - **XY4**: The canonical 4-pulse sequence {X, Y, X, Y}.
//! - **CPMG(N)**: Carr--Purcell--Meiboom--Gill with N Y-pulses.
//! - **UDD(N)**: Uhrig dynamical decoupling with optimal timings.
//! - **PlatonicXY**: Extended 8-pulse XY-8 sequence {X, Y, X, Y, Y, X, Y, X}.
//! - **Custom**: User-defined pulse sequence.
//! - **Identity**: No-op for comparison baselines.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::dynamical_decoupling::{DDConfig, DDPass, DDSequence};
//! use nqpu_metal::gates::{Gate, GateType};
//!
//! let config = DDConfig::new()
//!     .sequence(DDSequence::XY4)
//!     .min_idle_slots(2);
//!
//! let pass = DDPass::new(config);
//!
//! let circuit = vec![
//!     Gate::single(GateType::H, 0),
//!     // qubit 1 is idle for many layers
//!     Gate::single(GateType::X, 0),
//!     Gate::single(GateType::Z, 0),
//!     Gate::single(GateType::H, 0),
//! ];
//!
//! let augmented = pass.insert_dd(&circuit, 2);
//! // qubit 1 now has DD pulses inserted during its idle period
//! ```

use crate::gates::{Gate, GateType};
use crate::C64;

// ============================================================
// DDSequence -- Available dynamical decoupling sequences
// ============================================================

/// Enumeration of supported dynamical decoupling pulse sequences.
///
/// Each sequence is designed to compose to the identity operator when
/// applied in full, ensuring the DD pass does not alter the logical
/// computation.
#[derive(Clone, Debug, PartialEq)]
pub enum DDSequence {
    /// XY4: {X, Y, X, Y} -- the canonical 4-pulse DD sequence.
    /// Suppresses both bit-flip and phase-flip noise symmetrically.
    XY4,

    /// Carr--Purcell--Meiboom--Gill with N Y-pulses equally spaced.
    /// For identity composition, N must be even.
    CPMG(usize),

    /// Uhrig Dynamical Decoupling with N pulses at optimal timings
    /// t_j = sin^2(pi * j / (2N + 2)) for j = 1..N.
    /// Uses Y-pulses; N must be even for identity composition.
    UDD(usize),

    /// Extended 8-pulse XY sequence: {X, Y, X, Y, Y, X, Y, X}.
    /// This is the standard XY-8 sequence (XY4 followed by its
    /// time-reversal YX4), providing higher-order suppression of
    /// correlated errors.
    PlatonicXY,

    /// No DD insertion -- used as a comparison baseline.
    Identity,

    /// User-defined pulse sequence of arbitrary single-qubit gate types.
    Custom(Vec<GateType>),
}

impl DDSequence {
    /// Returns the number of pulses in this sequence.
    pub fn num_pulses(&self) -> usize {
        dd_sequence_gates(self).len()
    }
}

// ============================================================
// DDConfig -- Builder-pattern configuration
// ============================================================

/// Configuration for the dynamical decoupling pass.
///
/// Uses a builder pattern for ergonomic construction:
///
/// ```rust
/// use nqpu_metal::dynamical_decoupling::{DDConfig, DDSequence};
///
/// let config = DDConfig::new()
///     .sequence(DDSequence::CPMG(4))
///     .min_idle_slots(3)
///     .exclude_qubits(vec![2, 5])
///     .respect_barriers(true);
/// ```
#[derive(Clone, Debug)]
pub struct DDConfig {
    /// Minimum number of consecutive idle time slots required before
    /// DD pulses are inserted.  Default: 2.
    pub min_idle_slots: usize,

    /// The DD pulse sequence to use.  Default: XY4.
    pub sequence: DDSequence,

    /// Qubits that should never receive DD pulses (e.g. ancillas
    /// managed by an error-correction code).
    pub exclude_qubits: Vec<usize>,

    /// When true, idle periods that span a barrier boundary are not
    /// treated as contiguous.  Default: true.
    ///
    /// Note: The current gate set does not include a Barrier gate type,
    /// so this flag is a forward-compatibility provision.  When a Barrier
    /// gate is added, the scheduling logic will respect it.
    pub respect_barriers: bool,
}

impl Default for DDConfig {
    fn default() -> Self {
        Self {
            min_idle_slots: 2,
            sequence: DDSequence::XY4,
            exclude_qubits: Vec::new(),
            respect_barriers: true,
        }
    }
}

impl DDConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the DD sequence.
    pub fn sequence(mut self, seq: DDSequence) -> Self {
        self.sequence = seq;
        self
    }

    /// Set the minimum idle slot threshold.
    pub fn min_idle_slots(mut self, slots: usize) -> Self {
        self.min_idle_slots = slots;
        self
    }

    /// Set the list of excluded qubits.
    pub fn exclude_qubits(mut self, qubits: Vec<usize>) -> Self {
        self.exclude_qubits = qubits;
        self
    }

    /// Set whether to respect barriers.
    pub fn respect_barriers(mut self, respect: bool) -> Self {
        self.respect_barriers = respect;
        self
    }
}

// ============================================================
// DDStats -- Statistics from a DD pass
// ============================================================

/// Statistics collected during a DD insertion pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DDStats {
    /// Number of gates in the original circuit.
    pub original_gates: usize,

    /// Number of DD gates inserted.
    pub inserted_gates: usize,

    /// Total number of idle periods detected (across all qubits).
    pub idle_periods_found: usize,

    /// Number of idle periods that were long enough to fill with DD.
    pub idle_periods_filled: usize,

    /// Sorted, deduplicated list of qubits that received DD protection.
    pub qubits_protected: Vec<usize>,
}

// ============================================================
// CircuitScheduler -- Time-slot scheduling helper
// ============================================================

/// Helper that converts a flat gate list into a 2-D time-slot schedule.
///
/// The schedule is a `Vec<Vec<Option<usize>>>` where
/// `schedule[time_slot][qubit]` is `Some(gate_index)` if that qubit is
/// busy at that time slot, or `None` if it is idle.
pub struct CircuitScheduler;

impl CircuitScheduler {
    /// Schedule a gate list into a 2-D grid of time slots.
    ///
    /// The algorithm performs a greedy ASAP (As-Soon-As-Possible)
    /// scheduling: each gate is placed in the earliest time slot where
    /// none of its target/control qubits are already occupied.
    ///
    /// Returns `(schedule, gate_slot_map)` where `gate_slot_map[i]`
    /// is the time slot assigned to gate `i`.
    pub fn schedule(gates: &[Gate], num_qubits: usize) -> (Vec<Vec<Option<usize>>>, Vec<usize>) {
        if gates.is_empty() || num_qubits == 0 {
            return (Vec::new(), Vec::new());
        }

        // Track the next available slot for each qubit.
        let mut next_available = vec![0usize; num_qubits];
        let mut gate_slot_map = Vec::with_capacity(gates.len());
        let mut max_slot = 0usize;

        for (gate_idx, gate) in gates.iter().enumerate() {
            let involved = Self::involved_qubits(gate);

            // Find the earliest slot where all involved qubits are free.
            let earliest = involved
                .iter()
                .filter(|&&q| q < num_qubits)
                .map(|&q| next_available[q])
                .max()
                .unwrap_or(0);

            gate_slot_map.push(earliest);

            // Mark all involved qubits as busy until after this slot.
            for &q in &involved {
                if q < num_qubits {
                    next_available[q] = earliest + 1;
                }
            }

            if earliest > max_slot {
                max_slot = earliest;
            }

            let _ = gate_idx; // used implicitly via enumerate index
        }

        // Build the 2-D schedule grid.
        let num_slots = if gates.is_empty() { 0 } else { max_slot + 1 };
        let mut schedule = vec![vec![None; num_qubits]; num_slots];

        for (gate_idx, &slot) in gate_slot_map.iter().enumerate() {
            let involved = Self::involved_qubits(&gates[gate_idx]);
            for &q in &involved {
                if q < num_qubits {
                    schedule[slot][q] = Some(gate_idx);
                }
            }
        }

        (schedule, gate_slot_map)
    }

    /// Find all idle periods for a given qubit in the schedule.
    ///
    /// Returns a list of `(start_slot, length)` pairs representing
    /// consecutive time slots where the qubit has no gate.
    pub fn find_idle_periods(schedule: &[Vec<Option<usize>>], qubit: usize) -> Vec<(usize, usize)> {
        if schedule.is_empty() {
            return Vec::new();
        }

        let num_slots = schedule.len();
        let mut periods = Vec::new();
        let mut start: Option<usize> = None;

        for slot in 0..num_slots {
            let is_idle = if qubit < schedule[slot].len() {
                schedule[slot][qubit].is_none()
            } else {
                true // qubit index beyond schedule width is idle
            };

            if is_idle {
                if start.is_none() {
                    start = Some(slot);
                }
            } else if let Some(s) = start {
                periods.push((s, slot - s));
                start = None;
            }
        }

        // Close any trailing idle period.
        if let Some(s) = start {
            periods.push((s, num_slots - s));
        }

        periods
    }

    /// Extract all qubit indices involved in a gate (targets + controls).
    fn involved_qubits(gate: &Gate) -> Vec<usize> {
        let mut qubits = Vec::with_capacity(gate.targets.len() + gate.controls.len());
        qubits.extend_from_slice(&gate.targets);
        qubits.extend_from_slice(&gate.controls);
        qubits
    }
}

// ============================================================
// DDPass -- The main transformation pass
// ============================================================

/// The dynamical decoupling transformation pass.
///
/// Given a circuit (as a `Vec<Gate>`) and the number of qubits, this
/// pass identifies idle periods and inserts the configured DD sequence
/// to protect qubits from decoherence.
pub struct DDPass {
    config: DDConfig,
}

impl DDPass {
    /// Create a new DD pass with the given configuration.
    pub fn new(config: DDConfig) -> Self {
        Self { config }
    }

    /// Create a DD pass with default XY4 configuration.
    pub fn default_xy4() -> Self {
        Self::new(DDConfig::new())
    }

    /// Insert DD pulses into the circuit, returning the augmented circuit.
    ///
    /// The algorithm:
    /// 1. Schedule the input circuit into time layers via ASAP scheduling.
    /// 2. For each qubit (not in the exclude list), find idle periods.
    /// 3. For each idle period >= `min_idle_slots`, distribute DD pulses
    ///    evenly across the idle slots.
    /// 4. Build the output circuit from the augmented schedule.
    pub fn insert_dd(&self, circuit: &[Gate], num_qubits: usize) -> Vec<Gate> {
        let (augmented, _stats) = self.insert_dd_with_stats(circuit, num_qubits);
        augmented
    }

    /// Insert DD pulses and return both the augmented circuit and statistics.
    pub fn insert_dd_with_stats(
        &self,
        circuit: &[Gate],
        num_qubits: usize,
    ) -> (Vec<Gate>, DDStats) {
        let mut stats = DDStats {
            original_gates: circuit.len(),
            ..Default::default()
        };

        // Identity sequence means no insertion.
        if matches!(self.config.sequence, DDSequence::Identity) {
            return (circuit.to_vec(), stats);
        }

        if circuit.is_empty() || num_qubits == 0 {
            return (circuit.to_vec(), stats);
        }

        let dd_gates = dd_sequence_gates(&self.config.sequence);
        if dd_gates.is_empty() {
            return (circuit.to_vec(), stats);
        }

        // Step 1: Schedule the circuit.
        let (schedule, gate_slot_map) = CircuitScheduler::schedule(circuit, num_qubits);
        let num_slots = schedule.len();

        if num_slots == 0 {
            return (circuit.to_vec(), stats);
        }

        // Step 2: Build a mutable augmented schedule.
        // Each cell holds either an original gate or an inserted DD gate.
        // We track insertions separately per (slot, qubit).
        let mut dd_insertions: Vec<Vec<Option<GateType>>> = vec![vec![None; num_qubits]; num_slots];
        let mut protected_qubits = Vec::new();

        for qubit in 0..num_qubits {
            // Skip excluded qubits.
            if self.config.exclude_qubits.contains(&qubit) {
                continue;
            }

            let idle_periods = CircuitScheduler::find_idle_periods(&schedule, qubit);

            for &(start, length) in &idle_periods {
                stats.idle_periods_found += 1;

                if length < self.config.min_idle_slots {
                    continue;
                }

                // Distribute DD pulses evenly across idle slots.
                let pulse_positions = Self::compute_pulse_positions(
                    &self.config.sequence,
                    &dd_gates,
                    start,
                    length,
                );

                if pulse_positions.is_empty() {
                    continue;
                }

                stats.idle_periods_filled += 1;

                for (slot, gate_type) in &pulse_positions {
                    if *slot < num_slots {
                        dd_insertions[*slot][qubit] = Some(gate_type.clone());
                        stats.inserted_gates += 1;
                    }
                }

                if !protected_qubits.contains(&qubit) {
                    protected_qubits.push(qubit);
                }
            }
        }

        protected_qubits.sort();
        protected_qubits.dedup();
        stats.qubits_protected = protected_qubits;

        // Step 3: Reconstruct the circuit from the augmented schedule.
        let augmented = self.reconstruct_circuit(
            circuit,
            &schedule,
            &gate_slot_map,
            &dd_insertions,
            num_qubits,
            num_slots,
        );

        (augmented, stats)
    }

    /// Compute which time slots receive which DD pulse gate types.
    ///
    /// For UDD sequences, pulse placement follows the Uhrig formula.
    /// For all other sequences, pulses are distributed as evenly as
    /// possible across the idle window.
    fn compute_pulse_positions(
        sequence: &DDSequence,
        dd_gates: &[GateType],
        start: usize,
        length: usize,
    ) -> Vec<(usize, GateType)> {
        let n_pulses = dd_gates.len();
        if n_pulses == 0 || length == 0 {
            return Vec::new();
        }

        // If the idle period is shorter than the number of pulses,
        // we cannot fit them all.  In that case, truncate.
        let actual_pulses = n_pulses.min(length);

        match sequence {
            DDSequence::UDD(n) => {
                // Use Uhrig timings: t_j = sin^2(pi * j / (2N + 2))
                // Map these fractional positions onto the idle window.
                let timings = udd_timings(*n);
                let mut positions = Vec::with_capacity(actual_pulses);

                for (i, &t) in timings.iter().enumerate().take(actual_pulses) {
                    let slot = start + (t * (length as f64)).round() as usize;
                    // Clamp to the idle window.
                    let slot = slot.min(start + length - 1);
                    positions.push((slot, dd_gates[i].clone()));
                }

                positions
            }
            _ => {
                // Equally spaced placement.
                let mut positions = Vec::with_capacity(actual_pulses);

                if actual_pulses == 1 {
                    // Single pulse in the middle.
                    let slot = start + length / 2;
                    positions.push((slot, dd_gates[0].clone()));
                } else {
                    // Distribute evenly within the window.
                    // Place pulses at fractional positions (i+1)/(n+1) of the window.
                    for i in 0..actual_pulses {
                        let frac = (i as f64 + 1.0) / (actual_pulses as f64 + 1.0);
                        let slot = start + (frac * length as f64).round() as usize;
                        let slot = slot.min(start + length - 1);
                        positions.push((slot, dd_gates[i].clone()));
                    }
                }

                positions
            }
        }
    }

    /// Reconstruct the output circuit from the time-slot schedule plus
    /// DD insertions.
    ///
    /// The output preserves the original gate ordering within each time
    /// slot and appends DD gates at the end of each slot.
    fn reconstruct_circuit(
        &self,
        original: &[Gate],
        _schedule: &[Vec<Option<usize>>],
        gate_slot_map: &[usize],
        dd_insertions: &[Vec<Option<GateType>>],
        num_qubits: usize,
        num_slots: usize,
    ) -> Vec<Gate> {
        // We need to emit gates in time-slot order.
        // Within each slot: original gates first, then DD insertions.

        // Build a mapping from slot -> list of original gate indices.
        let mut slot_to_gates: Vec<Vec<usize>> = vec![Vec::new(); num_slots];
        for (gate_idx, &slot) in gate_slot_map.iter().enumerate() {
            slot_to_gates[slot].push(gate_idx);
        }

        let mut result = Vec::with_capacity(original.len() + dd_insertions.len());

        for slot in 0..num_slots {
            // Emit original gates for this slot.
            for &gate_idx in &slot_to_gates[slot] {
                result.push(original[gate_idx].clone());
            }

            // Emit DD insertions for this slot.
            for qubit in 0..num_qubits {
                if let Some(ref gate_type) = dd_insertions[slot][qubit] {
                    result.push(Gate::single(gate_type.clone(), qubit));
                }
            }
        }

        result
    }

    /// Get a reference to this pass's configuration.
    pub fn config(&self) -> &DDConfig {
        &self.config
    }
}

// ============================================================
// Public helper functions
// ============================================================

/// Expand a `DDSequence` into its constituent gate types.
///
/// For parameterized sequences (CPMG, UDD), the function generates
/// the full list of pulses.  For `Identity`, returns an empty vec.
pub fn dd_sequence_gates(seq: &DDSequence) -> Vec<GateType> {
    match seq {
        DDSequence::XY4 => vec![GateType::X, GateType::Y, GateType::X, GateType::Y],

        DDSequence::CPMG(n) => {
            // N Y-pulses.
            vec![GateType::Y; *n]
        }

        DDSequence::UDD(n) => {
            // UDD uses N Y-pulses placed at Uhrig-optimal timings.
            vec![GateType::Y; *n]
        }

        DDSequence::PlatonicXY => vec![
            GateType::X,
            GateType::Y,
            GateType::X,
            GateType::Y,
            GateType::Y,
            GateType::X,
            GateType::Y,
            GateType::X,
        ],

        DDSequence::Identity => Vec::new(),

        DDSequence::Custom(gates) => gates.clone(),
    }
}

/// Compute Uhrig Dynamical Decoupling pulse timings.
///
/// Returns the fractional positions t_j = sin^2(pi * j / (2N + 2))
/// for j = 1, 2, ..., N.
///
/// These positions are in the open interval (0, 1) and are symmetric
/// about 0.5 for even N.
pub fn udd_timings(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let denom = 2.0 * n as f64 + 2.0;
    (1..=n)
        .map(|j| {
            let arg = std::f64::consts::PI * j as f64 / denom;
            arg.sin().powi(2)
        })
        .collect()
}

/// Verify that a DD sequence composes to the identity operator
/// (up to a global phase).
///
/// This function multiplies the 2x2 unitary matrices of each pulse
/// in the sequence and checks whether the result is proportional to
/// the identity matrix.
///
/// # Notes
///
/// - For `CPMG(n)` and `UDD(n)`, the sequence is identity only when
///   n is even (since Y^2 = -I and (-I)^k = I when k is even).
/// - `Identity` returns true trivially (empty sequence = identity).
/// - For `Custom` sequences, the check is performed numerically.
pub fn verify_dd_identity(seq: &DDSequence) -> bool {
    let gates = dd_sequence_gates(seq);

    if gates.is_empty() {
        return true; // empty sequence is identity
    }

    // Start with the 2x2 identity matrix.
    let mut product = [[C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                        [C64::new(0.0, 0.0), C64::new(1.0, 0.0)]];

    for gate_type in &gates {
        let mat = gate_type.matrix();
        // We only handle single-qubit gates (2x2 matrices).
        if mat.len() != 2 || mat[0].len() != 2 {
            return false;
        }

        let g = [[mat[0][0], mat[0][1]],
                  [mat[1][0], mat[1][1]]];

        // Multiply: product = g * product
        let new = mat_mul_2x2(&g, &product);
        product = new;
    }

    // Check if result is proportional to identity: product = e^{i*phi} * I
    // This means product[0][1] ~ 0, product[1][0] ~ 0,
    // and product[0][0] ~ product[1][1].
    let eps = 1e-10;

    let off_diag_ok = product[0][1].norm() < eps && product[1][0].norm() < eps;
    if !off_diag_ok {
        return false;
    }

    // Diagonal elements should be equal (same global phase).
    let diag_diff = (product[0][0] - product[1][1]).norm();
    diag_diff < eps
}

/// Multiply two 2x2 complex matrices.
fn mat_mul_2x2(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
    let mut result = [[C64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Identity verification tests
    // ----------------------------------------------------------

    #[test]
    fn test_xy4_is_identity() {
        // XY4 = X Y X Y
        // X*Y = i*Z, then (i*Z)*(X) = i*Z*X = i*(i*Y) = -Y
        // then (-Y)*Y = -(Y*Y) = -(-I) = I
        assert!(
            verify_dd_identity(&DDSequence::XY4),
            "XY4 sequence should compose to identity"
        );
    }

    #[test]
    fn test_cpmg_even_is_identity() {
        // Y^2 = -I, Y^4 = I, Y^6 = -I, Y^8 = I, ...
        // Even number of Y-pairs: Y^(2k) for k even -> identity up to global phase.
        // Actually Y^2 = -I, so Y^4 = (-I)^2 = I.
        assert!(
            verify_dd_identity(&DDSequence::CPMG(2)),
            "CPMG(2) = YY should compose to -I (identity up to phase)"
        );
        assert!(
            verify_dd_identity(&DDSequence::CPMG(4)),
            "CPMG(4) = YYYY should compose to identity"
        );
        assert!(
            verify_dd_identity(&DDSequence::CPMG(8)),
            "CPMG(8) should compose to identity"
        );
    }

    #[test]
    fn test_cpmg_odd_not_identity() {
        // Y^1 = Y (not identity), Y^3 = -Y (not identity)
        assert!(
            !verify_dd_identity(&DDSequence::CPMG(1)),
            "CPMG(1) = Y should NOT compose to identity"
        );
        assert!(
            !verify_dd_identity(&DDSequence::CPMG(3)),
            "CPMG(3) = YYY should NOT compose to identity"
        );
    }

    #[test]
    fn test_udd_even_is_identity() {
        // UDD uses Y-pulses, so identity checks follow CPMG parity.
        assert!(
            verify_dd_identity(&DDSequence::UDD(2)),
            "UDD(2) should compose to identity (up to phase)"
        );
        assert!(
            verify_dd_identity(&DDSequence::UDD(4)),
            "UDD(4) should compose to identity"
        );
    }

    #[test]
    fn test_platonic_xy_is_identity() {
        // PlatonicXY = X Y X Y X Y = (XYXY) * (XY)
        // XYXY = I (or -I up to phase), XY = iZ
        // Actually let's verify numerically.  The sequence should
        // compose to identity up to global phase.
        assert!(
            verify_dd_identity(&DDSequence::PlatonicXY),
            "PlatonicXY should compose to identity (up to global phase)"
        );
    }

    #[test]
    fn test_identity_sequence_is_identity() {
        assert!(
            verify_dd_identity(&DDSequence::Identity),
            "Identity (empty) sequence should trivially be identity"
        );
    }

    #[test]
    fn test_custom_identity_sequence() {
        // X X = I
        let custom = DDSequence::Custom(vec![GateType::X, GateType::X]);
        assert!(
            verify_dd_identity(&custom),
            "Custom XX should compose to identity"
        );

        // H H = I
        let custom_hh = DDSequence::Custom(vec![GateType::H, GateType::H]);
        assert!(
            verify_dd_identity(&custom_hh),
            "Custom HH should compose to identity"
        );
    }

    #[test]
    fn test_custom_non_identity() {
        let custom = DDSequence::Custom(vec![GateType::X]);
        assert!(
            !verify_dd_identity(&custom),
            "Custom single X is not identity"
        );
    }

    // ----------------------------------------------------------
    // UDD timing tests
    // ----------------------------------------------------------

    #[test]
    fn test_udd_timings_in_unit_interval() {
        for n in 1..=10 {
            let timings = udd_timings(n);
            assert_eq!(timings.len(), n);
            for &t in &timings {
                assert!(
                    t > 0.0 && t < 1.0,
                    "UDD timing {} should be in (0, 1) for n={}",
                    t,
                    n
                );
            }
        }
    }

    #[test]
    fn test_udd_timings_monotonic() {
        for n in 2..=8 {
            let timings = udd_timings(n);
            for i in 1..timings.len() {
                assert!(
                    timings[i] > timings[i - 1],
                    "UDD timings should be strictly increasing for n={}",
                    n
                );
            }
        }
    }

    #[test]
    fn test_udd_timings_symmetric() {
        // For even N, timings should be symmetric about 0.5:
        // t_j + t_{N+1-j} = 1.0
        for n in &[2, 4, 6, 8] {
            let timings = udd_timings(*n);
            for j in 0..*n {
                let complement = *n - 1 - j;
                let sum = timings[j] + timings[complement];
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "UDD timings for n={} should be symmetric: t[{}] + t[{}] = {} (expected 1.0)",
                    n,
                    j,
                    complement,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_udd_timings_empty() {
        assert!(udd_timings(0).is_empty());
    }

    // ----------------------------------------------------------
    // DD sequence expansion tests
    // ----------------------------------------------------------

    #[test]
    fn test_dd_sequence_gates_xy4() {
        let gates = dd_sequence_gates(&DDSequence::XY4);
        assert_eq!(gates.len(), 4);
        assert_eq!(gates[0], GateType::X);
        assert_eq!(gates[1], GateType::Y);
        assert_eq!(gates[2], GateType::X);
        assert_eq!(gates[3], GateType::Y);
    }

    #[test]
    fn test_dd_sequence_gates_cpmg() {
        let gates = dd_sequence_gates(&DDSequence::CPMG(6));
        assert_eq!(gates.len(), 6);
        for g in &gates {
            assert_eq!(*g, GateType::Y);
        }
    }

    #[test]
    fn test_dd_sequence_gates_identity_empty() {
        let gates = dd_sequence_gates(&DDSequence::Identity);
        assert!(gates.is_empty());
    }

    // ----------------------------------------------------------
    // Circuit scheduling tests
    // ----------------------------------------------------------

    #[test]
    fn test_schedule_single_qubit_circuit() {
        // Three gates on qubit 0 -- should be in 3 separate slots.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
        ];

        let (schedule, slot_map) = CircuitScheduler::schedule(&circuit, 1);
        assert_eq!(schedule.len(), 3);
        assert_eq!(slot_map, vec![0, 1, 2]);

        // Each slot should have qubit 0 occupied.
        for slot in &schedule {
            assert!(slot[0].is_some());
        }
    }

    #[test]
    fn test_schedule_parallel_gates() {
        // Two gates on different qubits can run in parallel.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 1),
        ];

        let (schedule, slot_map) = CircuitScheduler::schedule(&circuit, 2);
        // Both gates should be in slot 0.
        assert_eq!(slot_map[0], 0);
        assert_eq!(slot_map[1], 0);
        assert_eq!(schedule.len(), 1);
    }

    #[test]
    fn test_find_idle_periods_basic() {
        // 2 qubits, 5 slots.  Qubit 0 busy at slots 0 and 4.
        // Qubit 1 busy at slot 2 only.
        let schedule: Vec<Vec<Option<usize>>> = vec![
            vec![Some(0), None],   // slot 0
            vec![None, None],      // slot 1
            vec![None, Some(1)],   // slot 2
            vec![None, None],      // slot 3
            vec![Some(2), None],   // slot 4
        ];

        let q0_idle = CircuitScheduler::find_idle_periods(&schedule, 0);
        // Qubit 0: idle at slots 1, 2, 3 (length 3)
        assert_eq!(q0_idle, vec![(1, 3)]);

        let q1_idle = CircuitScheduler::find_idle_periods(&schedule, 1);
        // Qubit 1: idle at slots 0-1 (length 2), then 3-4 (length 2)
        assert_eq!(q1_idle, vec![(0, 2), (3, 2)]);
    }

    // ----------------------------------------------------------
    // DD insertion tests
    // ----------------------------------------------------------

    #[test]
    fn test_insert_dd_simple_two_qubit() {
        // Qubit 0 has gates at every step; qubit 1 is idle throughout.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
            Gate::single(GateType::H, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        // Qubit 1 should have DD pulses inserted.
        assert!(stats.inserted_gates > 0, "DD gates should be inserted");
        assert!(
            stats.qubits_protected.contains(&1),
            "Qubit 1 should be protected"
        );
        assert!(augmented.len() > circuit.len());

        // Original gates should still be present.
        assert_eq!(stats.original_gates, 4);
    }

    #[test]
    fn test_no_insertion_below_threshold() {
        // 1 qubit, min_idle_slots = 5, only 2 slots total.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(5);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        // Qubit 1 has a 2-slot idle period but threshold is 5.
        assert_eq!(stats.inserted_gates, 0);
        assert_eq!(stats.idle_periods_filled, 0);
        assert_eq!(augmented.len(), circuit.len());
    }

    #[test]
    fn test_excluded_qubits_skipped() {
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(2)
            .exclude_qubits(vec![1]);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        assert_eq!(stats.inserted_gates, 0);
        assert!(stats.qubits_protected.is_empty());
        assert_eq!(augmented.len(), circuit.len());
    }

    #[test]
    fn test_no_idle_periods_unchanged() {
        // Both qubits busy at every slot.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 1),
            Gate::single(GateType::Z, 0),
            Gate::single(GateType::Y, 1),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(1);

        let pass = DDPass::new(config);
        let (schedule, _) = CircuitScheduler::schedule(&circuit, 2);

        // With ASAP scheduling, gates on different qubits can parallelize.
        // H on q0 and X on q1 go to slot 0; Z on q0 and Y on q1 to slot 1.
        assert_eq!(schedule.len(), 2);

        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);
        assert_eq!(stats.inserted_gates, 0);
        assert_eq!(augmented.len(), circuit.len());
    }

    #[test]
    fn test_stats_accuracy() {
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        assert_eq!(stats.original_gates, 5);
        assert_eq!(
            augmented.len(),
            stats.original_gates + stats.inserted_gates
        );
        assert!(stats.idle_periods_found > 0);
        assert!(stats.idle_periods_filled <= stats.idle_periods_found);

        // Verify all protected qubits are sorted and deduplicated.
        for i in 1..stats.qubits_protected.len() {
            assert!(stats.qubits_protected[i] > stats.qubits_protected[i - 1]);
        }
    }

    #[test]
    fn test_identity_sequence_no_change() {
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::Identity)
            .min_idle_slots(1);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        assert_eq!(stats.inserted_gates, 0);
        assert_eq!(augmented.len(), circuit.len());
    }

    #[test]
    fn test_custom_sequence_insertion() {
        // Custom sequence: Z Z (composes to identity)
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::Custom(vec![GateType::Z, GateType::Z]))
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        assert!(stats.inserted_gates > 0, "Custom DD should be inserted");
        assert!(augmented.len() > circuit.len());

        // Verify the inserted gates are Z gates on qubit 1.
        let dd_gates: Vec<&Gate> = augmented
            .iter()
            .filter(|g| g.targets.contains(&1))
            .collect();
        assert!(!dd_gates.is_empty());
        for g in &dd_gates {
            assert_eq!(g.gate_type, GateType::Z);
        }
    }

    #[test]
    fn test_single_qubit_no_idle() {
        // Single qubit circuit: no other qubits to protect.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 1);

        // Only 1 qubit, always busy -- no idle periods.
        assert_eq!(stats.idle_periods_found, 0);
        assert_eq!(stats.inserted_gates, 0);
        assert_eq!(augmented.len(), circuit.len());
    }

    #[test]
    fn test_staggered_multi_qubit() {
        // 3 qubits with staggered activity.
        // q0: H at slot 0
        // q1: X at slot 0 (parallel with H on q0)
        // q2: idle for all slots
        // q0: Z at slot 1
        // q1: Y at slot 1 (parallel with Z on q0)
        // q0: X at slot 2
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 1),
            Gate::single(GateType::Z, 0),
            Gate::single(GateType::Y, 1),
            Gate::single(GateType::X, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::CPMG(2))
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 3);

        // q2 should be protected (idle for 3 slots >= threshold of 2).
        assert!(
            stats.qubits_protected.contains(&2),
            "Qubit 2 should be protected"
        );
        assert!(stats.inserted_gates > 0);
        assert!(augmented.len() > circuit.len());
    }

    #[test]
    fn test_cpmg_insertion() {
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
            Gate::single(GateType::H, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::CPMG(2))
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        assert!(stats.inserted_gates > 0);

        // Verify inserted gates are Y gates.
        let dd_gates: Vec<&Gate> = augmented
            .iter()
            .filter(|g| g.targets.contains(&1))
            .collect();
        for g in &dd_gates {
            assert_eq!(g.gate_type, GateType::Y);
        }
    }

    #[test]
    fn test_platonic_xy_insertion() {
        // Need enough idle slots for 6 pulses.
        let circuit: Vec<Gate> = (0..8)
            .map(|_| Gate::single(GateType::H, 0))
            .collect();

        let config = DDConfig::new()
            .sequence(DDSequence::PlatonicXY)
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        assert!(stats.inserted_gates > 0);
        assert!(stats.qubits_protected.contains(&1));
        assert!(augmented.len() > circuit.len());
    }

    #[test]
    fn test_respect_barriers_config() {
        // Verify the respect_barriers flag is stored correctly.
        let config = DDConfig::new().respect_barriers(false);
        assert!(!config.respect_barriers);

        let config2 = DDConfig::new().respect_barriers(true);
        assert!(config2.respect_barriers);
    }

    #[test]
    fn test_udd_insertion() {
        let circuit: Vec<Gate> = (0..6)
            .map(|_| Gate::single(GateType::H, 0))
            .collect();

        let config = DDConfig::new()
            .sequence(DDSequence::UDD(4))
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        assert!(stats.inserted_gates > 0);
        assert!(stats.qubits_protected.contains(&1));

        // Verify inserted gates are Y gates (UDD uses Y).
        let dd_gates: Vec<&Gate> = augmented
            .iter()
            .filter(|g| g.targets.contains(&1))
            .collect();
        for g in &dd_gates {
            assert_eq!(g.gate_type, GateType::Y);
        }
    }

    #[test]
    fn test_empty_circuit() {
        let config = DDConfig::new();
        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&[], 4);

        assert_eq!(augmented.len(), 0);
        assert_eq!(stats.original_gates, 0);
        assert_eq!(stats.inserted_gates, 0);
    }

    #[test]
    fn test_cnot_scheduling() {
        // CNOT occupies both control and target qubits.
        let circuit = vec![
            Gate::two(GateType::CNOT, 0, 1),
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 1),
        ];

        let (schedule, slot_map) = CircuitScheduler::schedule(&circuit, 3);

        // CNOT at slot 0, H and X at slot 1 (both depend on q0/q1).
        assert_eq!(slot_map[0], 0); // CNOT
        assert_eq!(slot_map[1], 1); // H on q0
        assert_eq!(slot_map[2], 1); // X on q1

        // q2 should be idle for all slots.
        let q2_idle = CircuitScheduler::find_idle_periods(&schedule, 2);
        assert_eq!(q2_idle, vec![(0, 2)]);

        // Now run DD pass.
        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let (_augmented, stats) = pass.insert_dd_with_stats(&circuit, 3);

        assert!(
            stats.qubits_protected.contains(&2),
            "Qubit 2 should be protected during CNOT"
        );
    }

    #[test]
    fn test_min_idle_slots_one() {
        // With min_idle_slots = 1, even a single idle slot should trigger DD.
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::Custom(vec![GateType::X, GateType::X]))
            .min_idle_slots(1);

        let pass = DDPass::new(config);
        let (augmented, stats) = pass.insert_dd_with_stats(&circuit, 2);

        // Qubit 1 is idle for 2 slots (>= 1), should get DD.
        assert!(stats.inserted_gates > 0);
        assert!(augmented.len() > circuit.len());
    }

    #[test]
    fn test_dd_config_builder() {
        let config = DDConfig::new()
            .sequence(DDSequence::CPMG(6))
            .min_idle_slots(4)
            .exclude_qubits(vec![0, 3, 7])
            .respect_barriers(false);

        assert_eq!(config.min_idle_slots, 4);
        assert_eq!(config.exclude_qubits, vec![0, 3, 7]);
        assert!(!config.respect_barriers);
        assert_eq!(config.sequence, DDSequence::CPMG(6));
    }

    #[test]
    fn test_dd_preserves_original_gate_order() {
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::single(GateType::X, 0),
            Gate::single(GateType::Z, 0),
        ];

        let config = DDConfig::new()
            .sequence(DDSequence::XY4)
            .min_idle_slots(2);

        let pass = DDPass::new(config);
        let augmented = pass.insert_dd(&circuit, 2);

        // Extract gates that target qubit 0 -- they should appear in
        // the original order.
        let q0_gates: Vec<&GateType> = augmented
            .iter()
            .filter(|g| g.targets.contains(&0) && g.controls.is_empty())
            .map(|g| &g.gate_type)
            .collect();

        assert!(q0_gates.len() >= 3);
        assert_eq!(*q0_gates[0], GateType::H);
        assert_eq!(*q0_gates[1], GateType::X);
        assert_eq!(*q0_gates[2], GateType::Z);
    }

    #[test]
    fn test_verify_dd_identity_all_builtins() {
        // All built-in sequences with even pulse counts should verify.
        assert!(verify_dd_identity(&DDSequence::XY4));
        assert!(verify_dd_identity(&DDSequence::CPMG(2)));
        assert!(verify_dd_identity(&DDSequence::CPMG(4)));
        assert!(verify_dd_identity(&DDSequence::UDD(2)));
        assert!(verify_dd_identity(&DDSequence::UDD(4)));
        assert!(verify_dd_identity(&DDSequence::PlatonicXY));
        assert!(verify_dd_identity(&DDSequence::Identity));
    }

    #[test]
    fn test_num_pulses() {
        assert_eq!(DDSequence::XY4.num_pulses(), 4);
        assert_eq!(DDSequence::CPMG(6).num_pulses(), 6);
        assert_eq!(DDSequence::UDD(3).num_pulses(), 3);
        assert_eq!(DDSequence::PlatonicXY.num_pulses(), 8);
        assert_eq!(DDSequence::Identity.num_pulses(), 0);
        assert_eq!(
            DDSequence::Custom(vec![GateType::X, GateType::Y, GateType::Z]).num_pulses(),
            3
        );
    }

    #[test]
    fn test_default_xy4_constructor() {
        let pass = DDPass::default_xy4();
        let config = pass.config();
        assert_eq!(config.sequence, DDSequence::XY4);
        assert_eq!(config.min_idle_slots, 2);
        assert!(config.exclude_qubits.is_empty());
        assert!(config.respect_barriers);
    }
}
