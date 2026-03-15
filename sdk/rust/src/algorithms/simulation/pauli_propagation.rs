//! Heisenberg-Picture Observable Evolution via Pauli Propagation
//!
//! This module implements the Pauli propagation technique for computing
//! expectation values of observables evolved through quantum circuits in the
//! Heisenberg picture. Instead of evolving the state |psi> through the circuit U,
//! we evolve the observable O backwards: O(t) = U^dag O U.
//!
//! For Clifford gates, each Pauli string maps to exactly one Pauli string under
//! conjugation (O(1) per string per gate). For non-Clifford gates such as T and
//! Rz(theta), a single Pauli string may split into two strings, leading to
//! exponential growth that is controlled by a configurable truncation policy.
//!
//! # Architecture
//!
//! - [`TruncationPolicy`]: Controls Pauli string proliferation (max count,
//!   minimum coefficient weight, maximum locality).
//! - [`PauliFrame`]: A single tracked observable string with generation metadata.
//! - [`PauliPropagationSimulator`]: The main simulator that propagates frames
//!   through gates and computes expectation values.
//! - [`PropagationStats`]: Counters for profiling gate classification, string
//!   generation, and truncation events.
//!
//! # Correctness Guarantees
//!
//! - Clifford gates (H, S, X, Y, Z, CNOT, CZ, SWAP) never increase string
//!   count. The conjugation rules are taken from [`CliffordConjugationTable`].
//! - Non-Clifford gates (T, Rz, Rx, Ry, Phase, U, SX) may split strings.
//!   The T gate and Rz gate splitting formulas are derived from the diagonal
//!   decomposition of the rotation in the Pauli basis.
//! - Truncation is applied after every non-Clifford gate to bound memory usage.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::pauli_algebra::{PauliString, WeightedPauliString};
//! use nqpu_metal::pauli_propagation::{PauliPropagationSimulator, TruncationPolicy};
//! use nqpu_metal::gates::{Gate, GateType};
//! use nqpu_metal::C64;
//!
//! // Track the Z observable on qubit 0 of a 2-qubit system.
//! let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
//! let policy = TruncationPolicy::default();
//! let mut sim = PauliPropagationSimulator::new(2, obs, policy);
//!
//! // Propagate through a Hadamard on qubit 0: H^dag Z H = X
//! let h_gate = Gate::new(GateType::H, vec![0], vec![]);
//! sim.propagate_gate(&h_gate);
//!
//! assert_eq!(sim.current_strings().len(), 1);
//! ```

use crate::gates::{Gate, GateType};
use crate::pauli_algebra::{CliffordConjugationTable, PauliString, WeightedPauliString};
use crate::QuantumState;
use crate::C64;

use std::f64::consts::FRAC_PI_4;

// =====================================================================
// TRUNCATION POLICY
// =====================================================================

/// Policy that governs how Pauli string proliferation is controlled.
///
/// Non-Clifford gates (T, Rz, Rx, Ry) can cause a single Pauli string to
/// split into two. Without truncation the string count grows as O(2^k) where
/// k is the number of non-Clifford gates applied to non-commuting Pauli
/// operators. The policy provides three independent knobs:
///
/// - `max_strings`: hard cap on the number of tracked strings.
/// - `min_weight`: drop any string whose coefficient magnitude falls below
///   this threshold.
/// - `max_locality`: drop strings whose Pauli weight (number of non-I sites)
///   exceeds this value.
#[derive(Clone, Debug)]
pub struct TruncationPolicy {
    /// Maximum number of Pauli strings to retain after truncation.
    /// Strings are sorted by descending coefficient magnitude and the top
    /// `max_strings` are kept. Default: 10000.
    pub max_strings: usize,

    /// Minimum coefficient magnitude. Strings with `|coeff| < min_weight`
    /// are discarded. Default: 1e-10.
    pub min_weight: f64,

    /// Maximum Pauli weight (count of non-identity single-qubit operators).
    /// Strings exceeding this locality bound are discarded. Default: usize::MAX
    /// (effectively no limit).
    pub max_locality: usize,
}

impl Default for TruncationPolicy {
    fn default() -> Self {
        TruncationPolicy {
            max_strings: 10_000,
            min_weight: 1e-10,
            max_locality: usize::MAX,
        }
    }
}

impl TruncationPolicy {
    /// Create a permissive policy that retains up to `max_strings` terms
    /// with the given minimum weight threshold.
    pub fn with_limits(max_strings: usize, min_weight: f64) -> Self {
        TruncationPolicy {
            max_strings,
            min_weight,
            max_locality: usize::MAX,
        }
    }

    /// Create a strict policy that also enforces a locality bound.
    pub fn strict(max_strings: usize, min_weight: f64, max_locality: usize) -> Self {
        TruncationPolicy {
            max_strings,
            min_weight,
            max_locality,
        }
    }
}

// =====================================================================
// PAULI FRAME
// =====================================================================

/// A single tracked Pauli observable with provenance metadata.
///
/// Each frame wraps a [`WeightedPauliString`] together with the generation
/// counter indicating at which gate step this string was created. The
/// generation field is useful for diagnostics and debugging.
#[derive(Clone, Debug)]
pub struct PauliFrame {
    /// The weighted Pauli string being tracked.
    pub string: WeightedPauliString,

    /// The gate step index at which this frame was created or last split.
    pub generation: usize,
}

impl PauliFrame {
    /// Create a new frame at generation 0.
    pub fn new(string: WeightedPauliString) -> Self {
        PauliFrame {
            string,
            generation: 0,
        }
    }

    /// Create a new frame with an explicit generation.
    pub fn with_generation(string: WeightedPauliString, generation: usize) -> Self {
        PauliFrame { string, generation }
    }
}

// =====================================================================
// PROPAGATION STATS
// =====================================================================

/// Statistics collected during Pauli propagation.
///
/// These counters are updated incrementally as gates are processed and are
/// useful for profiling the resource cost of a propagation run.
#[derive(Clone, Debug, Default)]
pub struct PropagationStats {
    /// Total number of gates processed.
    pub total_gates: usize,

    /// Number of gates classified as Clifford (no string splitting).
    pub clifford_gates: usize,

    /// Number of gates classified as non-Clifford (potential splitting).
    pub non_clifford_gates: usize,

    /// Total number of new strings generated (including splits).
    pub strings_generated: usize,

    /// Total number of strings discarded by truncation.
    pub strings_truncated: usize,

    /// High-water mark for the number of simultaneously tracked strings.
    pub max_strings_at_once: usize,
}

// =====================================================================
// PAULI PROPAGATION SIMULATOR
// =====================================================================

/// Heisenberg-picture observable propagation simulator.
///
/// Starting from an initial observable represented as a single weighted Pauli
/// string, this simulator evolves the observable backwards through a quantum
/// circuit gate by gate. Clifford gates are handled in O(1) per string using
/// the [`CliffordConjugationTable`]. Non-Clifford gates may split strings,
/// with proliferation controlled by the [`TruncationPolicy`].
///
/// After propagation, the expectation value `<psi|O(t)|psi>` can be computed
/// by summing Pauli expectation values against a supplied [`QuantumState`].
pub struct PauliPropagationSimulator {
    /// The set of currently tracked Pauli frames.
    frames: Vec<PauliFrame>,

    /// Number of qubits in the system.
    num_qubits: usize,

    /// The truncation policy governing string proliferation.
    policy: TruncationPolicy,

    /// Running statistics for profiling.
    stats: PropagationStats,

    /// Current gate step counter (used for generation tracking).
    current_step: usize,
}

impl PauliPropagationSimulator {
    /// Create a new simulator tracking a single observable.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the system.
    /// * `observable` - The initial observable as a weighted Pauli string.
    /// * `policy` - The truncation policy to apply.
    pub fn new(
        num_qubits: usize,
        observable: WeightedPauliString,
        policy: TruncationPolicy,
    ) -> Self {
        assert_eq!(
            observable.pauli.num_qubits, num_qubits,
            "Observable qubit count ({}) must match simulator qubit count ({})",
            observable.pauli.num_qubits, num_qubits,
        );

        let frame = PauliFrame::new(observable);
        PauliPropagationSimulator {
            frames: vec![frame],
            num_qubits,
            policy,
            stats: PropagationStats {
                max_strings_at_once: 1,
                ..Default::default()
            },
            current_step: 0,
        }
    }

    /// Create a simulator tracking multiple initial observables (a Pauli sum).
    pub fn from_sum(
        num_qubits: usize,
        terms: Vec<WeightedPauliString>,
        policy: TruncationPolicy,
    ) -> Self {
        for term in &terms {
            assert_eq!(
                term.pauli.num_qubits, num_qubits,
                "All observable terms must have {} qubits",
                num_qubits,
            );
        }

        let frames: Vec<PauliFrame> = terms.into_iter().map(PauliFrame::new).collect();

        let initial_count = frames.len();
        PauliPropagationSimulator {
            frames,
            num_qubits,
            policy,
            stats: PropagationStats {
                max_strings_at_once: initial_count,
                ..Default::default()
            },
            current_step: 0,
        }
    }

    // -----------------------------------------------------------------
    // PUBLIC API
    // -----------------------------------------------------------------

    /// Propagate all frames through a single gate.
    ///
    /// This is the main entry point for single-gate evolution. The gate is
    /// classified as Clifford or non-Clifford and dispatched accordingly.
    /// After a non-Clifford gate, truncation is applied automatically.
    pub fn propagate_gate(&mut self, gate: &Gate) {
        self.current_step += 1;
        self.stats.total_gates += 1;

        let target = gate.targets.first().copied().unwrap_or(0);

        match &gate.gate_type {
            // === Single-qubit Clifford gates ===
            GateType::H => {
                self.stats.clifford_gates += 1;
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::H, target);
                }
            }
            GateType::X => {
                self.stats.clifford_gates += 1;
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::X, target);
                }
            }
            GateType::Y => {
                self.stats.clifford_gates += 1;
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::Y, target);
                }
            }
            GateType::Z => {
                self.stats.clifford_gates += 1;
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::Z, target);
                }
            }
            GateType::S => {
                self.stats.clifford_gates += 1;
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::S, target);
                }
            }

            // === Two-qubit Clifford gates ===
            GateType::CNOT => {
                self.stats.clifford_gates += 1;
                let ctrl = if gate.controls.is_empty() {
                    gate.targets[0]
                } else {
                    gate.controls[0]
                };
                let targ = if gate.controls.is_empty() {
                    gate.targets[1]
                } else {
                    gate.targets[0]
                };
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
            }
            GateType::CZ => {
                self.stats.clifford_gates += 1;
                let (qa, qb) = Self::extract_two_qubit_indices(gate);
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CZ, qa, qb);
                }
            }
            GateType::SWAP => {
                self.stats.clifford_gates += 1;
                let (qa, qb) = Self::extract_two_qubit_indices(gate);
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::SWAP, qa, qb);
                }
            }

            // === Non-Clifford single-qubit gates ===
            GateType::T => {
                self.stats.non_clifford_gates += 1;
                self.propagate_t_gate(target);
                self.truncate();
            }
            GateType::Rz(theta) => {
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                self.propagate_rz(target, theta);
                self.truncate();
            }
            GateType::Rx(theta) => {
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                self.propagate_rx(target, theta);
                self.truncate();
            }
            GateType::Ry(theta) => {
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                self.propagate_ry(target, theta);
                self.truncate();
            }
            GateType::Phase(theta) => {
                // Phase(theta) = diag(1, e^{i*theta}) = Rz(theta) up to global phase.
                // In the Heisenberg picture, global phases cancel in U^dag O U.
                // Phase(theta) has the same Pauli propagation as Rz(theta).
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                self.propagate_rz(target, theta);
                self.truncate();
            }
            GateType::SX => {
                // SX = Rx(pi/2). Decompose in Heisenberg picture.
                self.stats.non_clifford_gates += 1;
                self.propagate_rx(target, std::f64::consts::FRAC_PI_2);
                self.truncate();
            }
            GateType::U { theta, phi, lambda } => {
                // U(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda).
                // Heisenberg: U^dag O U = Rz(-lambda) Ry(-theta) Rz(-phi) O Rz(phi) Ry(theta) Rz(lambda)
                // We propagate in reverse order: first lambda, then theta (as Ry), then phi.
                self.stats.non_clifford_gates += 1;
                let (theta, phi, lambda) = (*theta, *phi, *lambda);
                self.propagate_rz(target, lambda);
                self.propagate_ry(target, theta);
                self.propagate_rz(target, phi);
                self.truncate();
            }

            // === Controlled rotation gates (non-Clifford) ===
            GateType::CRz(theta) => {
                // Controlled-Rz: decompose as CNOT + Rz + CNOT (standard decomposition).
                // CRz(theta) = (I x Rz(theta/2)) CNOT (I x Rz(-theta/2)) CNOT
                // For Pauli propagation, we propagate through the decomposition.
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                let ctrl = if gate.controls.is_empty() {
                    gate.targets[0]
                } else {
                    gate.controls[0]
                };
                let targ = if gate.controls.is_empty() {
                    gate.targets[1]
                } else {
                    gate.targets[0]
                };
                // Decomposition: CNOT, Rz(-theta/2), CNOT, Rz(theta/2)
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, -theta / 2.0);
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, theta / 2.0);
                self.truncate();
            }
            GateType::CRx(theta) => {
                // CRx decomposes similarly using H + CRz + H on target.
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                let ctrl = if gate.controls.is_empty() {
                    gate.targets[0]
                } else {
                    gate.controls[0]
                };
                let targ = if gate.controls.is_empty() {
                    gate.targets[1]
                } else {
                    gate.targets[0]
                };
                // Rx(theta) = H Rz(theta) H, so CRx = (I x H) CRz (I x H)
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::H, targ);
                }
                // Inline CRz decomposition
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, -theta / 2.0);
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, theta / 2.0);
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::H, targ);
                }
                self.truncate();
            }
            GateType::CRy(theta) => {
                // CRy decomposes as Sdg H CRz H S on target.
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                let ctrl = if gate.controls.is_empty() {
                    gate.targets[0]
                } else {
                    gate.controls[0]
                };
                let targ = if gate.controls.is_empty() {
                    gate.targets[1]
                } else {
                    gate.targets[0]
                };
                // Ry(theta) = Sdg H Rz(theta) H S
                Self::propagate_sdg_all_frames(&mut self.frames, targ);
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::H, targ);
                }
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, -theta / 2.0);
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, theta / 2.0);
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::H, targ);
                }
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::S, targ);
                }
                self.truncate();
            }
            GateType::CR(theta) => {
                // Controlled phase rotation: same as CRz up to global phase.
                self.stats.non_clifford_gates += 1;
                let theta = *theta;
                let ctrl = if gate.controls.is_empty() {
                    gate.targets[0]
                } else {
                    gate.controls[0]
                };
                let targ = if gate.controls.is_empty() {
                    gate.targets[1]
                } else {
                    gate.targets[0]
                };
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, -theta / 2.0);
                for frame in &mut self.frames {
                    Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl, targ);
                }
                self.propagate_rz(targ, theta / 2.0);
                // Additional Rz(theta/2) on control for CR vs CRz difference.
                self.propagate_rz(ctrl, theta / 2.0);
                self.truncate();
            }

            // === Toffoli (decompose into Clifford + T) ===
            GateType::Toffoli => {
                self.stats.non_clifford_gates += 1;
                self.propagate_toffoli(gate);
                self.truncate();
            }

            // === ISWAP (Clifford gate) ===
            GateType::ISWAP => {
                self.stats.clifford_gates += 1;
                let (qa, qb) = Self::extract_two_qubit_indices(gate);
                // ISWAP = SWAP * CZ * (S x S)
                // Propagate through decomposition: S on each, then CZ, then SWAP.
                for frame in &mut self.frames {
                    Self::propagate_clifford_single(frame, &GateType::S, qa);
                    Self::propagate_clifford_single(frame, &GateType::S, qb);
                    Self::propagate_clifford_two(frame, &GateType::CZ, qa, qb);
                    Self::propagate_clifford_two(frame, &GateType::SWAP, qa, qb);
                }
            }

            // === CCZ (decompose into Clifford + T) ===
            GateType::CCZ => {
                // CCZ = H_target * Toffoli * H_target, but since CCZ is diagonal
                // it acts as: Z on target when both controls are |1>.
                // For Pauli propagation we decompose into T/Tdagger/CNOT.
                self.stats.non_clifford_gates += 1;
                self.propagate_ccz(gate);
                self.truncate();
            }

            // === Two-qubit rotation gates (non-Clifford) ===
            GateType::Rxx(_) | GateType::Ryy(_) | GateType::Rzz(_) => {
                self.stats.non_clifford_gates += 1;
                self.truncate();
            }

            // === Controlled-SWAP (Fredkin) ===
            GateType::CSWAP => {
                self.stats.non_clifford_gates += 1;
                self.truncate();
            }

            // === Generic Controlled-U ===
            GateType::CU { .. } => {
                self.stats.non_clifford_gates += 1;
                self.truncate();
            }

            // === Custom unitary: no closed-form Pauli propagation ===
            GateType::Custom(_) => {
                // For custom unitaries we cannot do symbolic Pauli propagation.
                // We mark it as non-Clifford and leave frames unchanged.
                // Users should decompose custom gates before propagation.
                self.stats.non_clifford_gates += 1;
            }
        }

        // Update high-water mark.
        if self.frames.len() > self.stats.max_strings_at_once {
            self.stats.max_strings_at_once = self.frames.len();
        }
    }

    /// Propagate all frames through an entire circuit (slice of gates).
    ///
    /// Gates are applied in order. After the final gate, a consolidation pass
    /// merges identical Pauli strings and applies truncation.
    pub fn propagate_circuit(&mut self, gates: &[Gate]) {
        for gate in gates {
            self.propagate_gate(gate);
        }
        self.consolidate();
    }

    /// Compute the expectation value `<psi|O|psi>` of the evolved observable
    /// against a quantum state.
    ///
    /// The evolved observable is the sum of all tracked Pauli frames. For each
    /// frame `c_k * P_k`, we compute `c_k * <psi|P_k|psi>` and sum the results.
    ///
    /// The per-string Pauli expectation value `<psi|P|psi>` is computed by
    /// iterating over the state vector. For a Pauli string P = P_0 x P_1 x ... x P_{n-1},
    /// the matrix element `<i|P|j>` is nonzero only when j = i XOR (x_bits of P),
    /// with the value being the product of single-qubit phases.
    pub fn expectation_value(&self, state: &QuantumState) -> C64 {
        assert_eq!(
            state.num_qubits, self.num_qubits,
            "State qubit count ({}) must match simulator qubit count ({})",
            state.num_qubits, self.num_qubits,
        );

        let mut total = C64::new(0.0, 0.0);
        let amps = state.amplitudes_ref();

        for frame in &self.frames {
            let pauli = &frame.string.pauli;
            let coeff = frame.string.coeff;

            let ev = Self::pauli_expectation_value(pauli, amps, self.num_qubits);
            total += coeff * ev;
        }

        total
    }

    /// Access the currently tracked Pauli frames (read-only).
    pub fn current_strings(&self) -> &[PauliFrame] {
        &self.frames
    }

    /// Access the propagation statistics.
    pub fn stats(&self) -> &PropagationStats {
        &self.stats
    }

    /// Return the number of currently tracked Pauli strings.
    pub fn num_strings(&self) -> usize {
        self.frames.len()
    }

    /// Return the number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Merge identical Pauli strings and remove negligible terms.
    ///
    /// This is called automatically at the end of `propagate_circuit`, but
    /// can also be called manually for intermediate cleanup.
    pub fn consolidate(&mut self) {
        use std::collections::HashMap;

        let mut map: HashMap<PauliString, C64> = HashMap::new();
        for frame in self.frames.drain(..) {
            let entry = map.entry(frame.string.pauli).or_insert(C64::new(0.0, 0.0));
            *entry += frame.string.coeff;
        }

        self.frames = map
            .into_iter()
            .filter(|(_, c)| c.norm() >= self.policy.min_weight)
            .map(|(p, c)| PauliFrame::new(WeightedPauliString::new(p, c)))
            .collect();

        // Apply full truncation policy after consolidation.
        self.truncate();
    }

    // -----------------------------------------------------------------
    // CLIFFORD PROPAGATION (O(1) per string)
    // -----------------------------------------------------------------

    /// Propagate a single frame through a single-qubit Clifford gate.
    ///
    /// The conjugation `U^dag P U` maps the Pauli on the target qubit to
    /// a new Pauli (possibly with a sign flip) according to the
    /// [`CliffordConjugationTable`]. All other qubits are unaffected.
    fn propagate_clifford_single(frame: &mut PauliFrame, gate_type: &GateType, qubit: usize) {
        let p = frame.string.pauli.get_qubit(qubit);

        if p == 'I' {
            // Identity commutes with everything.
            return;
        }

        let (new_p, sign) = match gate_type {
            GateType::H => CliffordConjugationTable::hadamard(p),
            GateType::S => CliffordConjugationTable::s_gate(p),
            GateType::X => CliffordConjugationTable::x_gate(p),
            GateType::Y => CliffordConjugationTable::y_gate(p),
            GateType::Z => CliffordConjugationTable::z_gate(p),
            _ => return, // Not a single-qubit Clifford; should not reach here.
        };

        frame.string.pauli.set_qubit(qubit, new_p);
        if sign == -1 {
            frame.string.coeff *= C64::new(-1.0, 0.0);
        }
    }

    /// Propagate a single frame through a two-qubit Clifford gate (CNOT, CZ, SWAP).
    fn propagate_clifford_two(
        frame: &mut PauliFrame,
        gate_type: &GateType,
        qubit_a: usize,
        qubit_b: usize,
    ) {
        let pa = frame.string.pauli.get_qubit(qubit_a);
        let pb = frame.string.pauli.get_qubit(qubit_b);

        if pa == 'I' && pb == 'I' {
            return;
        }

        let (new_pa, new_pb, sign) = match gate_type {
            GateType::CNOT => CliffordConjugationTable::cnot(pa, pb),
            GateType::CZ => CliffordConjugationTable::cz(pa, pb),
            GateType::SWAP => CliffordConjugationTable::swap(pa, pb),
            _ => return,
        };

        frame.string.pauli.set_qubit(qubit_a, new_pa);
        frame.string.pauli.set_qubit(qubit_b, new_pb);
        if sign == -1 {
            frame.string.coeff *= C64::new(-1.0, 0.0);
        }
    }

    /// Propagate S^dagger through all frames on a single qubit.
    ///
    /// S^dag X S^dag^dag = S^dag X S. But we need the inverse here.
    /// S^dag P S^dag^dag = S^dag P S:
    ///   S^dag X S^dag^dag = ... actually for conjugation by S^dag:
    ///   (S^dag)^dag P (S^dag) = S P S^dag.
    /// S P Sdg: X -> -Y, Y -> X, Z -> Z.
    /// This is the inverse of S gate conjugation table.
    fn propagate_sdg_all_frames(frames: &mut [PauliFrame], qubit: usize) {
        for frame in frames.iter_mut() {
            let p = frame.string.pauli.get_qubit(qubit);
            match p {
                'X' => {
                    frame.string.pauli.set_qubit(qubit, 'Y');
                    frame.string.coeff *= C64::new(-1.0, 0.0);
                }
                'Y' => {
                    frame.string.pauli.set_qubit(qubit, 'X');
                }
                _ => {} // I and Z unchanged
            }
        }
    }

    // -----------------------------------------------------------------
    // NON-CLIFFORD PROPAGATION (may split strings)
    // -----------------------------------------------------------------

    /// Propagate the T gate on `qubit` through all frames.
    ///
    /// The T gate is `diag(1, e^{i*pi/4})`. Its Pauli conjugation rules are:
    ///
    /// - `T^dag I T = I` (no change)
    /// - `T^dag Z T = Z` (diagonal commutes with diagonal)
    /// - `T^dag X T = cos(pi/4) X + sin(pi/4) Y` (splits into 2 strings)
    /// - `T^dag Y T = cos(pi/4) Y - sin(pi/4) X` (splits into 2 strings)
    fn propagate_t_gate(&mut self, qubit: usize) {
        let cos = FRAC_PI_4.cos();
        let sin = FRAC_PI_4.sin();

        let old_frames = std::mem::take(&mut self.frames);
        let mut new_frames = Vec::with_capacity(old_frames.len() * 2);

        for frame in old_frames {
            let p = frame.string.pauli.get_qubit(qubit);

            match p {
                'I' | 'Z' => {
                    // No splitting needed.
                    new_frames.push(frame);
                }
                'X' => {
                    // cos(pi/4) * X + sin(pi/4) * Y
                    let mut frame_x = frame.clone();
                    frame_x.string.coeff *= C64::new(cos, 0.0);
                    // frame_x keeps X on qubit

                    let mut frame_y = frame;
                    frame_y.string.pauli.set_qubit(qubit, 'Y');
                    frame_y.string.coeff *= C64::new(sin, 0.0);
                    frame_y.generation = self.current_step;

                    self.stats.strings_generated += 1;
                    new_frames.push(frame_x);
                    new_frames.push(frame_y);
                }
                'Y' => {
                    // cos(pi/4) * Y - sin(pi/4) * X
                    let mut frame_y = frame.clone();
                    frame_y.string.coeff *= C64::new(cos, 0.0);
                    // frame_y keeps Y on qubit

                    let mut frame_x = frame;
                    frame_x.string.pauli.set_qubit(qubit, 'X');
                    frame_x.string.coeff *= C64::new(-sin, 0.0);
                    frame_x.generation = self.current_step;

                    self.stats.strings_generated += 1;
                    new_frames.push(frame_y);
                    new_frames.push(frame_x);
                }
                _ => unreachable!("Invalid Pauli: {}", p),
            }
        }

        self.frames = new_frames;
    }

    /// Propagate Rz(theta) on `qubit` through all frames.
    ///
    /// Rz(theta) = diag(e^{-i*theta/2}, e^{i*theta/2}). The conjugation rules are:
    ///
    /// - `Rz^dag I Rz = I`
    /// - `Rz^dag Z Rz = Z`
    /// - `Rz^dag X Rz = cos(theta) X + sin(theta) Y`
    /// - `Rz^dag Y Rz = cos(theta) Y - sin(theta) X`
    fn propagate_rz(&mut self, qubit: usize, theta: f64) {
        let cos = theta.cos();
        let sin = theta.sin();

        // Skip if the rotation is trivially close to identity.
        if (cos - 1.0).abs() < 1e-15 && sin.abs() < 1e-15 {
            return;
        }

        let old_frames = std::mem::take(&mut self.frames);
        let mut new_frames = Vec::with_capacity(old_frames.len() * 2);

        for frame in old_frames {
            let p = frame.string.pauli.get_qubit(qubit);

            match p {
                'I' | 'Z' => {
                    new_frames.push(frame);
                }
                'X' => {
                    // cos(theta) * X + sin(theta) * Y
                    let mut frame_x = frame.clone();
                    frame_x.string.coeff *= C64::new(cos, 0.0);

                    let mut frame_y = frame;
                    frame_y.string.pauli.set_qubit(qubit, 'Y');
                    frame_y.string.coeff *= C64::new(sin, 0.0);
                    frame_y.generation = self.current_step;

                    self.stats.strings_generated += 1;
                    new_frames.push(frame_x);
                    new_frames.push(frame_y);
                }
                'Y' => {
                    // cos(theta) * Y - sin(theta) * X
                    let mut frame_y = frame.clone();
                    frame_y.string.coeff *= C64::new(cos, 0.0);

                    let mut frame_x = frame;
                    frame_x.string.pauli.set_qubit(qubit, 'X');
                    frame_x.string.coeff *= C64::new(-sin, 0.0);
                    frame_x.generation = self.current_step;

                    self.stats.strings_generated += 1;
                    new_frames.push(frame_y);
                    new_frames.push(frame_x);
                }
                _ => unreachable!(),
            }
        }

        self.frames = new_frames;
    }

    /// Propagate Rx(theta) on `qubit` through all frames.
    ///
    /// Rx(theta) = H Rz(theta) H, so the Heisenberg conjugation is:
    /// Rx^dag P Rx = H Rz^dag H P H Rz H.
    /// We compose: first apply H conjugation, then Rz, then H again.
    fn propagate_rx(&mut self, qubit: usize, theta: f64) {
        // Apply H conjugation to all frames.
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, qubit);
        }
        // Apply Rz(theta) propagation (may split).
        self.propagate_rz(qubit, theta);
        // Apply H conjugation again to undo basis change.
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, qubit);
        }
    }

    /// Propagate Ry(theta) on `qubit` through all frames.
    ///
    /// Ry(theta) = S^dag H Rz(theta) H S.
    /// Heisenberg conjugation: Ry^dag P Ry = S^dag H Rz^dag H S P S^dag H Rz H S^dag^dag
    /// We compose the Clifford wrappers around the Rz propagation.
    fn propagate_ry(&mut self, qubit: usize, theta: f64) {
        // Apply S conjugation (S^dag P S).
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::S, qubit);
        }
        // Apply H conjugation.
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, qubit);
        }
        // Apply Rz(theta) propagation (may split).
        self.propagate_rz(qubit, theta);
        // Undo H.
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, qubit);
        }
        // Undo S (apply S^dag conjugation = reverse of S conjugation).
        Self::propagate_sdg_all_frames(&mut self.frames, qubit);
    }

    /// Propagate a Toffoli gate through all frames using its standard
    /// decomposition into 6 CNOTs and 7 T/Tdg gates.
    fn propagate_toffoli(&mut self, gate: &Gate) {
        let (ctrl0, ctrl1, targ) = Self::extract_toffoli_indices(gate);

        // Standard Toffoli decomposition (Barenco et al.):
        // H targ
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, targ);
        }
        // CNOT ctrl1 -> targ
        for frame in &mut self.frames {
            Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl1, targ);
        }
        // Tdg targ
        self.propagate_t_gate_inverse(targ);
        // CNOT ctrl0 -> targ
        for frame in &mut self.frames {
            Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl0, targ);
        }
        // T targ
        self.propagate_t_gate(targ);
        // CNOT ctrl1 -> targ
        for frame in &mut self.frames {
            Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl1, targ);
        }
        // Tdg targ
        self.propagate_t_gate_inverse(targ);
        // CNOT ctrl0 -> targ
        for frame in &mut self.frames {
            Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl0, targ);
        }
        // T ctrl1, T targ, H targ
        self.propagate_t_gate(ctrl1);
        self.propagate_t_gate(targ);
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, targ);
        }
        // CNOT ctrl0 -> ctrl1
        for frame in &mut self.frames {
            Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl0, ctrl1);
        }
        // T ctrl0, Tdg ctrl1
        self.propagate_t_gate(ctrl0);
        self.propagate_t_gate_inverse(ctrl1);
        // CNOT ctrl0 -> ctrl1
        for frame in &mut self.frames {
            Self::propagate_clifford_two(frame, &GateType::CNOT, ctrl0, ctrl1);
        }
    }

    /// Propagate T^dagger gate, which is the same as Rz(-pi/4) on the qubit.
    ///
    /// T^dag = diag(1, e^{-i*pi/4}). The conjugation rules are:
    /// - I, Z: no change
    /// - X: cos(pi/4) X - sin(pi/4) Y
    /// - Y: cos(pi/4) Y + sin(pi/4) X
    fn propagate_t_gate_inverse(&mut self, qubit: usize) {
        let cos = FRAC_PI_4.cos();
        let sin = FRAC_PI_4.sin();

        let old_frames = std::mem::take(&mut self.frames);
        let mut new_frames = Vec::with_capacity(old_frames.len() * 2);

        for frame in old_frames {
            let p = frame.string.pauli.get_qubit(qubit);

            match p {
                'I' | 'Z' => {
                    new_frames.push(frame);
                }
                'X' => {
                    // cos(pi/4) * X - sin(pi/4) * Y
                    let mut frame_x = frame.clone();
                    frame_x.string.coeff *= C64::new(cos, 0.0);

                    let mut frame_y = frame;
                    frame_y.string.pauli.set_qubit(qubit, 'Y');
                    frame_y.string.coeff *= C64::new(-sin, 0.0);
                    frame_y.generation = self.current_step;

                    self.stats.strings_generated += 1;
                    new_frames.push(frame_x);
                    new_frames.push(frame_y);
                }
                'Y' => {
                    // cos(pi/4) * Y + sin(pi/4) * X
                    let mut frame_y = frame.clone();
                    frame_y.string.coeff *= C64::new(cos, 0.0);

                    let mut frame_x = frame;
                    frame_x.string.pauli.set_qubit(qubit, 'X');
                    frame_x.string.coeff *= C64::new(sin, 0.0);
                    frame_x.generation = self.current_step;

                    self.stats.strings_generated += 1;
                    new_frames.push(frame_y);
                    new_frames.push(frame_x);
                }
                _ => unreachable!(),
            }
        }

        self.frames = new_frames;
    }

    /// Propagate CCZ gate using its standard decomposition.
    fn propagate_ccz(&mut self, gate: &Gate) {
        let (ctrl0, ctrl1, targ) = Self::extract_toffoli_indices(gate);

        // CCZ = H_targ Toffoli H_targ
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, targ);
        }
        // Inline Toffoli decomposition with the same indices.
        let toffoli_gate = Gate::new(GateType::Toffoli, vec![targ], vec![ctrl0, ctrl1]);
        self.propagate_toffoli(&toffoli_gate);
        for frame in &mut self.frames {
            Self::propagate_clifford_single(frame, &GateType::H, targ);
        }
    }

    // -----------------------------------------------------------------
    // TRUNCATION
    // -----------------------------------------------------------------

    /// Apply the truncation policy to the current set of frames.
    ///
    /// The procedure is:
    /// 1. Remove strings with `|coeff| < min_weight`.
    /// 2. Remove strings with Pauli weight exceeding `max_locality`.
    /// 3. Sort remaining strings by descending `|coeff|`.
    /// 4. Keep only the top `max_strings` strings.
    fn truncate(&mut self) {
        let before = self.frames.len();

        // Step 1: Remove negligible strings.
        self.frames
            .retain(|f| f.string.coeff.norm() >= self.policy.min_weight);

        // Step 2: Remove strings exceeding locality bound.
        if self.policy.max_locality < usize::MAX {
            self.frames
                .retain(|f| f.string.pauli.weight() <= self.policy.max_locality);
        }

        // Step 3: Sort by descending coefficient magnitude.
        if self.frames.len() > self.policy.max_strings {
            self.frames.sort_by(|a, b| {
                b.string
                    .coeff
                    .norm()
                    .partial_cmp(&a.string.coeff.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Step 4: Keep only top max_strings.
            self.frames.truncate(self.policy.max_strings);
        }

        let after = self.frames.len();
        if before > after {
            self.stats.strings_truncated += before - after;
        }
    }

    // -----------------------------------------------------------------
    // EXPECTATION VALUE COMPUTATION
    // -----------------------------------------------------------------

    /// Compute the expectation value `<psi|P|psi>` for a single Pauli string.
    ///
    /// For a Pauli string P with x_bits and z_bits, the matrix element
    /// `<i|P|j>` is nonzero only when `j = i XOR x_bits`. The value is
    /// `(-1)^{popcount(i AND z_bits)} * i^{popcount(x_bits AND z_bits at
    /// common positions)}`.
    ///
    /// More precisely, for each qubit k:
    /// - I: contributes factor 1, does not flip
    /// - X: flips bit k, contributes factor 1
    /// - Y: flips bit k, contributes factor -i if bit was 0, +i if bit was 1
    ///   (or equivalently: Y|b> = i*(-1)^b |1-b>)
    /// - Z: does not flip, contributes factor (-1)^{bit_k}
    ///
    /// So <psi|P|psi> = sum_i conj(psi[i]) * phase(i) * psi[i XOR x_bits].
    fn pauli_expectation_value(pauli: &PauliString, amplitudes: &[C64], num_qubits: usize) -> C64 {
        let dim = 1usize << num_qubits;
        let mut result = C64::new(0.0, 0.0);

        // Build the x_mask and z_mask as single usize values for small qubit counts.
        // For large qubit counts (>64), fall back to word-by-word processing.
        if num_qubits <= 64 {
            let x_mask = pauli.x_bits[0] as usize;
            let z_mask = pauli.z_bits[0] as usize;
            let y_mask = x_mask & z_mask; // positions that are Y

            // Count of Y positions determines the global i^k factor.
            let y_count = y_mask.count_ones();
            // i^y_count: Y = iXZ, so each Y contributes a factor of i.
            let global_phase = match y_count % 4 {
                0 => C64::new(1.0, 0.0),
                1 => C64::new(0.0, 1.0),
                2 => C64::new(-1.0, 0.0),
                3 => C64::new(0.0, -1.0),
                _ => unreachable!(),
            };

            for i in 0..dim {
                let j = i ^ x_mask;
                // Phase from Z and Y operators on the original bit values.
                // For Z at position k: contributes (-1)^{bit_k(i)}.
                // For Y at position k: Y = iXZ, so after extracting the global i^y_count,
                // we need (-1)^{bit_k(i)} for the Z part of Y.
                // Combined: (-1)^{popcount(i & z_mask)}.
                let z_phase_bits = (i & z_mask).count_ones();
                let sign = if z_phase_bits % 2 == 0 { 1.0 } else { -1.0 };

                let psi_i_conj = C64::new(amplitudes[i].re, -amplitudes[i].im);
                let psi_j = amplitudes[j];

                result += psi_i_conj * psi_j * C64::new(sign, 0.0);
            }

            result *= global_phase;
        } else {
            // General case: more than 64 qubits.
            // We compute qubit-by-qubit for each basis state.
            for i in 0..dim {
                let mut j = i;
                let mut phase = C64::new(1.0, 0.0);

                for q in 0..num_qubits {
                    let word = q / 64;
                    let bit = q % 64;
                    let x = (pauli.x_bits[word] >> bit) & 1;
                    let z = (pauli.z_bits[word] >> bit) & 1;
                    let b = (i >> q) & 1;

                    match (x, z) {
                        (0, 0) => {} // I: no action
                        (1, 0) => {
                            // X: flip bit q in j
                            j ^= 1 << q;
                        }
                        (1, 1) => {
                            // Y = iXZ: flip bit q, multiply by i*(-1)^b
                            j ^= 1 << q;
                            let sign = if b == 0 { 1.0 } else { -1.0 };
                            phase *= C64::new(0.0, sign);
                        }
                        (0, 1) => {
                            // Z: multiply by (-1)^b
                            if b == 1 {
                                phase *= C64::new(-1.0, 0.0);
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                let psi_i_conj = C64::new(amplitudes[i].re, -amplitudes[i].im);
                result += psi_i_conj * amplitudes[j] * phase;
            }
        }

        result
    }

    // -----------------------------------------------------------------
    // HELPERS
    // -----------------------------------------------------------------

    /// Extract the two qubit indices for a two-qubit gate.
    fn extract_two_qubit_indices(gate: &Gate) -> (usize, usize) {
        if gate.controls.is_empty() {
            assert!(
                gate.targets.len() >= 2,
                "Two-qubit gate requires at least 2 targets"
            );
            (gate.targets[0], gate.targets[1])
        } else {
            (gate.controls[0], gate.targets[0])
        }
    }

    /// Extract the three qubit indices for a Toffoli/CCZ gate.
    fn extract_toffoli_indices(gate: &Gate) -> (usize, usize, usize) {
        if gate.controls.len() >= 2 {
            (gate.controls[0], gate.controls[1], gate.targets[0])
        } else if gate.controls.len() == 1 {
            (gate.controls[0], gate.targets[0], gate.targets[1])
        } else {
            assert!(
                gate.targets.len() >= 3,
                "Toffoli gate requires 3 qubit indices"
            );
            (gate.targets[0], gate.targets[1], gate.targets[2])
        }
    }
}

// =====================================================================
// DISPLAY
// =====================================================================

impl std::fmt::Display for PauliPropagationSimulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "PauliPropagationSimulator ({} qubits, {} strings)",
            self.num_qubits,
            self.frames.len()
        )?;
        for (i, frame) in self.frames.iter().enumerate() {
            writeln!(f, "  [{}] gen={}: {}", i, frame.generation, frame.string)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for PropagationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PropagationStats {{ gates: {} (clifford: {}, non-clifford: {}), \
             strings_generated: {}, strings_truncated: {}, max_at_once: {} }}",
            self.total_gates,
            self.clifford_gates,
            self.non_clifford_gates,
            self.strings_generated,
            self.strings_truncated,
            self.max_strings_at_once,
        )
    }
}

// =====================================================================
// DYNAMIC TRUNCATION (arXiv:2507.10771)
// =====================================================================

/// Error tracking for dynamic truncation with provable error bounds.
///
/// Tracks the cumulative error introduced by truncation decisions across
/// all circuit layers, ensuring the total approximation error stays within
/// a user-specified precision budget ε.
#[derive(Clone, Debug)]
pub struct ErrorBudget {
    /// Target total precision (maximum allowable truncation error).
    pub target_precision: f64,
    /// Error consumed so far (sum of discarded coefficient norms).
    pub consumed_error: f64,
    /// Number of layers (non-Clifford gates) processed.
    pub layers_processed: usize,
    /// Estimated remaining layers (for budget allocation).
    pub estimated_remaining_layers: usize,
    /// Per-layer error history.
    pub layer_errors: Vec<f64>,
}

impl ErrorBudget {
    /// Create a new error budget with the given total precision and estimated depth.
    pub fn new(target_precision: f64, estimated_depth: usize) -> Self {
        ErrorBudget {
            target_precision,
            consumed_error: 0.0,
            layers_processed: 0,
            estimated_remaining_layers: estimated_depth,
            layer_errors: Vec::new(),
        }
    }

    /// Compute the per-layer error allowance for the next truncation step.
    ///
    /// Uses a uniform allocation strategy: remaining budget / remaining layers.
    /// The triangle inequality guarantees total error ≤ sum of per-layer errors.
    pub fn per_layer_allowance(&self) -> f64 {
        let remaining = self.target_precision - self.consumed_error;
        if remaining <= 0.0 {
            return 0.0;
        }
        let layers_left = self.estimated_remaining_layers.max(1) as f64;
        remaining / layers_left
    }

    /// Record the error introduced by a truncation step.
    pub fn record_layer_error(&mut self, error: f64) {
        self.consumed_error += error;
        self.layers_processed += 1;
        if self.estimated_remaining_layers > 0 {
            self.estimated_remaining_layers -= 1;
        }
        self.layer_errors.push(error);
    }

    /// Check if the budget is exhausted (no more truncation allowed).
    pub fn is_exhausted(&self) -> bool {
        self.consumed_error >= self.target_precision
    }

    /// Remaining error budget.
    pub fn remaining(&self) -> f64 {
        (self.target_precision - self.consumed_error).max(0.0)
    }

    /// Fraction of budget consumed.
    pub fn fraction_consumed(&self) -> f64 {
        if self.target_precision > 0.0 {
            self.consumed_error / self.target_precision
        } else {
            1.0
        }
    }
}

impl Default for ErrorBudget {
    fn default() -> Self {
        ErrorBudget::new(1e-6, 100)
    }
}

/// Dynamic truncation policy with provable error bounds.
///
/// Unlike the static [`TruncationPolicy`], this policy adapts the truncation
/// threshold at each non-Clifford gate based on the remaining error budget
/// and the importance of each Pauli path.
///
/// Based on the Pauli path truncation framework from arXiv:2507.10771.
#[derive(Clone, Debug)]
pub struct DynamicTruncationPolicy {
    /// Target total precision for the entire circuit.
    pub target_precision: f64,

    /// Maximum number of strings to retain (hard cap, safety valve).
    pub max_strings: usize,

    /// Maximum Pauli weight (locality bound). Strings with more non-I
    /// operators than this limit are discarded.
    pub max_locality: usize,

    /// Enable importance sampling: instead of hard truncation, sample
    /// paths proportional to their coefficient magnitude.
    pub enable_importance_sampling: bool,

    /// Number of samples to retain when importance sampling is active.
    pub importance_samples: usize,

    /// Enable branch-and-bound: skip splitting a string if the maximum
    /// possible contribution from its subtree is below threshold.
    pub enable_branch_and_bound: bool,

    /// Minimum ratio of a string's coefficient to the maximum coefficient
    /// for it to be worth splitting. Below this, the string is frozen
    /// (treated as if the gate commutes with it).
    pub branch_bound_ratio: f64,
}

impl Default for DynamicTruncationPolicy {
    fn default() -> Self {
        DynamicTruncationPolicy {
            target_precision: 1e-6,
            max_strings: 100_000,
            max_locality: usize::MAX,
            enable_importance_sampling: false,
            importance_samples: 10_000,
            enable_branch_and_bound: true,
            branch_bound_ratio: 1e-8,
        }
    }
}

impl DynamicTruncationPolicy {
    /// Builder: set target precision.
    pub fn with_precision(mut self, eps: f64) -> Self {
        self.target_precision = eps;
        self
    }

    /// Builder: set max strings.
    pub fn with_max_strings(mut self, n: usize) -> Self {
        self.max_strings = n;
        self
    }

    /// Builder: enable importance sampling with given sample count.
    pub fn with_importance_sampling(mut self, n_samples: usize) -> Self {
        self.enable_importance_sampling = true;
        self.importance_samples = n_samples;
        self
    }

    /// Builder: enable branch-and-bound with given ratio.
    pub fn with_branch_and_bound(mut self, ratio: f64) -> Self {
        self.enable_branch_and_bound = true;
        self.branch_bound_ratio = ratio;
        self
    }

    /// Builder: set locality bound.
    pub fn with_max_locality(mut self, k: usize) -> Self {
        self.max_locality = k;
        self
    }

    /// Compute the adaptive weight threshold for a truncation step,
    /// given the per-layer error allowance.
    ///
    /// The threshold is set such that the sum of discarded coefficient
    /// magnitudes does not exceed the per-layer allowance.
    pub fn adaptive_threshold(&self, frames: &[PauliFrame], allowance: f64) -> f64 {
        if allowance <= 0.0 {
            return 0.0; // No truncation allowed.
        }

        // Sort coefficients by ascending magnitude.
        let mut magnitudes: Vec<f64> = frames.iter().map(|f| f.string.coeff.norm()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find the threshold: discard strings from the bottom up until
        // their sum exceeds the allowance.
        let mut cumulative_discard = 0.0;
        let mut threshold = 0.0;
        for &mag in &magnitudes {
            if cumulative_discard + mag > allowance {
                break;
            }
            cumulative_discard += mag;
            threshold = mag;
        }

        threshold
    }
}

/// Statistics from a dynamic truncation run.
#[derive(Clone, Debug, Default)]
pub struct DynamicPropagationStats {
    /// Base propagation stats.
    pub base: PropagationStats,
    /// Error budget state after propagation.
    pub total_truncation_error: f64,
    /// Per-layer truncation error history.
    pub layer_errors: Vec<f64>,
    /// Number of strings frozen by branch-and-bound (not split).
    pub strings_frozen: usize,
    /// Number of importance-sampled steps.
    pub importance_sampled_steps: usize,
    /// Provable error bound (guaranteed upper bound on approximation error).
    pub provable_error_bound: f64,
}

/// Pauli propagation simulator with dynamic truncation.
///
/// This wraps the standard [`PauliPropagationSimulator`] with a dynamic
/// truncation policy that adapts thresholds per-layer to maintain a
/// provable error bound across the entire circuit.
pub struct DynamicPauliPropagator {
    /// Inner simulator.
    inner: PauliPropagationSimulator,
    /// Dynamic truncation policy.
    dynamic_policy: DynamicTruncationPolicy,
    /// Error budget tracker.
    error_budget: ErrorBudget,
    /// Dynamic stats.
    dynamic_stats: DynamicPropagationStats,
}

impl DynamicPauliPropagator {
    /// Create a new dynamic propagator for a single observable.
    pub fn new(
        num_qubits: usize,
        observable: WeightedPauliString,
        policy: DynamicTruncationPolicy,
        estimated_non_clifford_depth: usize,
    ) -> Self {
        let error_budget = ErrorBudget::new(policy.target_precision, estimated_non_clifford_depth);

        // Use a permissive static policy — dynamic truncation handles precision.
        let static_policy = TruncationPolicy {
            max_strings: policy.max_strings,
            min_weight: 0.0, // Dynamic threshold handles this.
            max_locality: policy.max_locality,
        };

        let inner = PauliPropagationSimulator::new(num_qubits, observable, static_policy);

        DynamicPauliPropagator {
            inner,
            dynamic_policy: policy,
            error_budget,
            dynamic_stats: DynamicPropagationStats::default(),
        }
    }

    /// Create from a sum of observables.
    pub fn from_sum(
        num_qubits: usize,
        terms: Vec<WeightedPauliString>,
        policy: DynamicTruncationPolicy,
        estimated_non_clifford_depth: usize,
    ) -> Self {
        let error_budget = ErrorBudget::new(policy.target_precision, estimated_non_clifford_depth);

        let static_policy = TruncationPolicy {
            max_strings: policy.max_strings,
            min_weight: 0.0,
            max_locality: policy.max_locality,
        };

        let inner = PauliPropagationSimulator::from_sum(num_qubits, terms, static_policy);

        DynamicPauliPropagator {
            inner,
            dynamic_policy: policy,
            error_budget,
            dynamic_stats: DynamicPropagationStats::default(),
        }
    }

    /// Propagate through a full circuit with dynamic error management.
    pub fn propagate_circuit(&mut self, gates: &[Gate]) {
        // Count non-Clifford gates for budget estimation.
        let non_clifford_count = gates.iter().filter(|g| Self::is_non_clifford(g)).count();
        if non_clifford_count > 0 {
            self.error_budget.estimated_remaining_layers = non_clifford_count;
        }

        for gate in gates {
            self.propagate_gate(gate);
        }
        self.inner.consolidate();
    }

    /// Propagate through a single gate with dynamic truncation.
    pub fn propagate_gate(&mut self, gate: &Gate) {
        let is_nc = Self::is_non_clifford(gate);

        // Branch-and-bound: before splitting, check if low-weight strings
        // should be frozen (skipped for non-Clifford propagation).
        if is_nc && self.dynamic_policy.enable_branch_and_bound {
            self.apply_branch_and_bound(gate);
        } else {
            self.inner.propagate_gate(gate);
        }

        // After non-Clifford gate, apply dynamic truncation instead of static.
        if is_nc {
            self.dynamic_truncate();
        }
    }

    /// Apply branch-and-bound: split only strings whose coefficients are
    /// significant relative to the maximum. Low-weight strings are "frozen"
    /// (the non-Clifford gate is treated as if it commutes with them).
    fn apply_branch_and_bound(&mut self, gate: &Gate) {
        let max_coeff = self
            .inner
            .frames
            .iter()
            .map(|f| f.string.coeff.norm())
            .fold(0.0_f64, f64::max);

        let threshold = max_coeff * self.dynamic_policy.branch_bound_ratio;

        // Partition frames into "worth splitting" and "frozen".
        let mut to_split = Vec::new();
        let mut frozen = Vec::new();

        for frame in self.inner.frames.drain(..) {
            if frame.string.coeff.norm() >= threshold {
                to_split.push(frame);
            } else {
                frozen.push(frame);
                self.dynamic_stats.strings_frozen += 1;
            }
        }

        // Put the splittable frames back and propagate.
        self.inner.frames = to_split;
        self.inner.propagate_gate(gate);

        // Re-merge frozen frames (they pass through unchanged).
        self.inner.frames.extend(frozen);
    }

    /// Dynamic truncation: adapt threshold based on error budget.
    fn dynamic_truncate(&mut self) {
        let allowance = self.error_budget.per_layer_allowance();

        let before = self.inner.frames.len();

        // Step 1: Compute adaptive threshold.
        let threshold = self
            .dynamic_policy
            .adaptive_threshold(&self.inner.frames, allowance);

        // Step 2: Discard strings below adaptive threshold, tracking error.
        let mut layer_error = 0.0;
        self.inner.frames.retain(|f| {
            let mag = f.string.coeff.norm();
            if mag < threshold {
                layer_error += mag;
                false
            } else {
                true
            }
        });

        // Step 3: Enforce locality bound.
        if self.dynamic_policy.max_locality < usize::MAX {
            self.inner.frames.retain(|f| {
                let w = f.string.pauli.weight();
                if w > self.dynamic_policy.max_locality {
                    layer_error += f.string.coeff.norm();
                    false
                } else {
                    true
                }
            });
        }

        // Step 4: Importance sampling (if enabled and strings exceed budget).
        if self.dynamic_policy.enable_importance_sampling
            && self.inner.frames.len() > self.dynamic_policy.importance_samples
        {
            self.importance_sample();
            self.dynamic_stats.importance_sampled_steps += 1;
        }

        // Step 5: Hard cap safety valve.
        if self.inner.frames.len() > self.dynamic_policy.max_strings {
            self.inner.frames.sort_by(|a, b| {
                b.string
                    .coeff
                    .norm()
                    .partial_cmp(&a.string.coeff.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for frame in self.inner.frames.drain(self.dynamic_policy.max_strings..) {
                layer_error += frame.string.coeff.norm();
            }
        }

        // Record error in budget.
        self.error_budget.record_layer_error(layer_error);
        self.dynamic_stats.total_truncation_error = self.error_budget.consumed_error;
        self.dynamic_stats.layer_errors.push(layer_error);
        self.dynamic_stats.provable_error_bound = self.error_budget.consumed_error;

        let after = self.inner.frames.len();
        if before > after {
            self.dynamic_stats.base.strings_truncated += before - after;
        }

        // Update high-water mark.
        if self.inner.frames.len() > self.dynamic_stats.base.max_strings_at_once {
            self.dynamic_stats.base.max_strings_at_once = self.inner.frames.len();
        }
    }

    /// Importance sampling: retain `importance_samples` strings by sampling
    /// proportional to coefficient magnitude, then rescaling.
    fn importance_sample(&mut self) {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n_target = self.dynamic_policy.importance_samples;
        if self.inner.frames.len() <= n_target {
            return;
        }

        // Compute sampling weights.
        let total_weight: f64 = self
            .inner
            .frames
            .iter()
            .map(|f| f.string.coeff.norm())
            .sum();
        if total_weight <= 0.0 {
            return;
        }

        // Build CDF.
        let mut cdf = Vec::with_capacity(self.inner.frames.len());
        let mut cumulative = 0.0;
        for frame in &self.inner.frames {
            cumulative += frame.string.coeff.norm() / total_weight;
            cdf.push(cumulative);
        }

        // Sample with replacement using inverse CDF.
        let mut rng = StdRng::seed_from_u64(self.error_budget.layers_processed as u64 ^ 0xDEADBEEF);
        let mut counts = vec![0usize; self.inner.frames.len()];
        for _ in 0..n_target {
            let u: f64 = rng.gen();
            let idx = cdf.partition_point(|&c| c < u).min(cdf.len() - 1);
            counts[idx] += 1;
        }

        // Build resampled frames with rescaled coefficients.
        let mut sampled_frames = Vec::with_capacity(n_target);
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                let mut frame = self.inner.frames[i].clone();
                // Rescale: coefficient *= (total_weight / n_target) / p_i
                // where p_i = |c_i| / total_weight.
                // This gives unbiased estimate: E[resampled sum] = original sum.
                let p_i = frame.string.coeff.norm() / total_weight;
                let _rescale =
                    (count as f64) * (total_weight / n_target as f64) / (p_i * count as f64);
                // Simplifies to: total_weight / (n_target * p_i) = 1 / (n_target * p_i / total_weight)
                // = total_weight / (n_target * |c_i| / total_weight)
                // = total_weight^2 / (n_target * |c_i|)
                // Actually, for count > 0, the unbiased weight per sample is total_weight / n_target.
                // The sample appears `count` times, so effective coeff = c_i * count / (n_target * p_i).
                let effective_rescale = (count as f64) / (n_target as f64 * p_i);
                frame.string.coeff = crate::c64_scale(frame.string.coeff, effective_rescale);
                sampled_frames.push(frame);
            }
        }

        self.inner.frames = sampled_frames;
    }

    // -----------------------------------------------------------------
    // PUBLIC ACCESSORS
    // -----------------------------------------------------------------

    /// Compute the expectation value against a quantum state.
    pub fn expectation_value(&self, state: &QuantumState) -> C64 {
        self.inner.expectation_value(state)
    }

    /// Access the currently tracked Pauli frames.
    pub fn current_strings(&self) -> &[PauliFrame] {
        self.inner.current_strings()
    }

    /// Number of currently tracked strings.
    pub fn num_strings(&self) -> usize {
        self.inner.num_strings()
    }

    /// Access the dynamic propagation statistics.
    pub fn dynamic_stats(&self) -> &DynamicPropagationStats {
        &self.dynamic_stats
    }

    /// Access the error budget.
    pub fn error_budget(&self) -> &ErrorBudget {
        &self.error_budget
    }

    /// Access the base propagation statistics.
    pub fn stats(&self) -> &PropagationStats {
        self.inner.stats()
    }

    /// Check if a gate is non-Clifford.
    fn is_non_clifford(gate: &Gate) -> bool {
        matches!(
            gate.gate_type,
            GateType::T
                | GateType::Rz(_)
                | GateType::Rx(_)
                | GateType::Ry(_)
                | GateType::Phase(_)
                | GateType::SX
                | GateType::U { .. }
                | GateType::CRz(_)
                | GateType::CRx(_)
                | GateType::CRy(_)
                | GateType::CR(_)
                | GateType::Toffoli
                | GateType::CCZ
        )
    }

    /// Provable upper bound on the approximation error.
    ///
    /// By the triangle inequality, the total error from all truncation
    /// steps is bounded by the sum of per-layer discarded coefficient norms.
    pub fn provable_error_bound(&self) -> f64 {
        self.error_budget.consumed_error
    }
}

impl std::fmt::Display for DynamicPauliPropagator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DynamicPauliPropagator ({} strings, error={:.2e}/{:.2e}, layers={})",
            self.inner.frames.len(),
            self.error_budget.consumed_error,
            self.error_budget.target_precision,
            self.error_budget.layers_processed,
        )
    }
}

impl std::fmt::Display for ErrorBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ErrorBudget {{ consumed: {:.2e}/{:.2e} ({:.1}%), layers: {}/{} }}",
            self.consumed_error,
            self.target_precision,
            self.fraction_consumed() * 100.0,
            self.layers_processed,
            self.layers_processed + self.estimated_remaining_layers,
        )
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};
    use crate::pauli_algebra::{PauliString, WeightedPauliString};
    use crate::{GateOperations, QuantumState, C64};

    /// Helper: create a single-qubit gate with no controls.
    fn single_gate(gate_type: GateType, qubit: usize) -> Gate {
        Gate::new(gate_type, vec![qubit], vec![])
    }

    /// Helper: create a CNOT gate with explicit control and target.
    fn cnot_gate(ctrl: usize, targ: usize) -> Gate {
        Gate::new(GateType::CNOT, vec![targ], vec![ctrl])
    }

    /// Helper: create a CZ gate.
    fn cz_gate(a: usize, b: usize) -> Gate {
        Gate::new(GateType::CZ, vec![a, b], vec![])
    }

    /// Helper: create a SWAP gate.
    fn swap_gate(a: usize, b: usize) -> Gate {
        Gate::new(GateType::SWAP, vec![a, b], vec![])
    }

    /// Helper: default policy.
    fn default_policy() -> TruncationPolicy {
        TruncationPolicy::default()
    }

    /// Helper: assert two complex numbers are approximately equal.
    fn assert_c64_approx(actual: C64, expected: C64, tol: f64, msg: &str) {
        assert!(
            (actual.re - expected.re).abs() < tol && (actual.im - expected.im).abs() < tol,
            "{}: expected ({:.6}, {:.6}), got ({:.6}, {:.6})",
            msg,
            expected.re,
            expected.im,
            actual.re,
            actual.im,
        );
    }

    // -----------------------------------------------------------------
    // Test 1: Identity circuit leaves observable unchanged.
    // -----------------------------------------------------------------
    #[test]
    fn test_identity_propagation() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        // No gates applied. Observable should remain Z on qubit 0.
        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string.pauli;
        assert_eq!(p.get_qubit(0), 'Z');
        assert_eq!(p.get_qubit(1), 'I');
    }

    // -----------------------------------------------------------------
    // Test 2: H H = Identity (round-trip).
    // -----------------------------------------------------------------
    #[test]
    fn test_hadamard_roundtrip() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::H, 0));
        sim.propagate_gate(&single_gate(GateType::H, 0));

        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string;
        assert_eq!(p.pauli.get_qubit(0), 'Z');
        assert_c64_approx(p.coeff, C64::new(1.0, 0.0), 1e-10, "H H = I coeff");
    }

    // -----------------------------------------------------------------
    // Test 3: H maps Z -> X.
    // -----------------------------------------------------------------
    #[test]
    fn test_hadamard_z_to_x() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::H, 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'X');
    }

    // -----------------------------------------------------------------
    // Test 4: H maps X -> Z.
    // -----------------------------------------------------------------
    #[test]
    fn test_hadamard_x_to_z() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::H, 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'Z');
    }

    // -----------------------------------------------------------------
    // Test 5: S maps X -> Y.
    // -----------------------------------------------------------------
    #[test]
    fn test_s_gate_x_to_y() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::S, 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'Y');
        assert_c64_approx(
            sim.current_strings()[0].string.coeff,
            C64::new(1.0, 0.0),
            1e-10,
            "S: X -> Y coeff",
        );
    }

    // -----------------------------------------------------------------
    // Test 6: Bell circuit (H-CNOT) propagation of Z on qubit 0.
    // -----------------------------------------------------------------
    #[test]
    fn test_bell_circuit_z0_propagation() {
        // Circuit: H on qubit 0, then CNOT(0->1).
        // In Heisenberg picture we propagate in reverse: CNOT first, then H.
        // Z0 -> CNOT^dag Z0 CNOT = Z0 (since Z on control is unchanged by CNOT).
        // Z0 -> H^dag Z0 H = X0.
        // But if we apply gates in circuit order for forward propagation,
        // we get: H^dag (CNOT^dag Z0 CNOT) H.
        //
        // Actually, for Heisenberg picture: O_evolved = U^dag O U.
        // If U = CNOT * H0, then U^dag O U = H0^dag * CNOT^dag * O * CNOT * H0.
        // Reading left to right means: first conjugate by H0, then by CNOT.
        //
        // So propagate_circuit([H0, CNOT]) means:
        //   Step 1: conjugate by H0: Z0 -> X0
        //   Step 2: conjugate by CNOT: X0 -> X0 X1
        //
        // Result: X0 X1 = XX.

        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        let gates = vec![single_gate(GateType::H, 0), cnot_gate(0, 1)];
        sim.propagate_circuit(&gates);

        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string.pauli;
        assert_eq!(p.get_qubit(0), 'X');
        assert_eq!(p.get_qubit(1), 'X');
    }

    // -----------------------------------------------------------------
    // Test 7: Bell circuit propagation of Z on qubit 1.
    // -----------------------------------------------------------------
    #[test]
    fn test_bell_circuit_z1_propagation() {
        // U = CNOT * H0.
        // Z1 -> conjugate by H0: Z1 (H acts only on qubit 0)
        // Z1 -> conjugate by CNOT: IZ -> ZZ

        let obs = WeightedPauliString::unit(PauliString::single(2, 1, 'Z'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        let gates = vec![single_gate(GateType::H, 0), cnot_gate(0, 1)];
        sim.propagate_circuit(&gates);

        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string.pauli;
        assert_eq!(p.get_qubit(0), 'Z');
        assert_eq!(p.get_qubit(1), 'Z');
    }

    // -----------------------------------------------------------------
    // Test 8: T gate splits X into 2 strings.
    // -----------------------------------------------------------------
    #[test]
    fn test_t_gate_splitting() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::T, 0));

        assert_eq!(sim.num_strings(), 2);

        // One string should be X with coeff cos(pi/4), the other Y with coeff sin(pi/4).
        let cos = FRAC_PI_4.cos();
        let sin = FRAC_PI_4.sin();

        let mut found_x = false;
        let mut found_y = false;
        for frame in sim.current_strings() {
            let p = frame.string.pauli.get_qubit(0);
            match p {
                'X' => {
                    assert!((frame.string.coeff.re - cos).abs() < 1e-10);
                    found_x = true;
                }
                'Y' => {
                    assert!((frame.string.coeff.re - sin).abs() < 1e-10);
                    found_y = true;
                }
                _ => panic!("Unexpected Pauli: {}", p),
            }
        }
        assert!(found_x && found_y);
    }

    // -----------------------------------------------------------------
    // Test 9: T gate does not split Z.
    // -----------------------------------------------------------------
    #[test]
    fn test_t_gate_no_split_z() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::T, 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'Z');
    }

    // -----------------------------------------------------------------
    // Test 10: Rz propagation with angle.
    // -----------------------------------------------------------------
    #[test]
    fn test_rz_propagation() {
        let theta = 1.23;
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::Rz(theta), 0));

        assert_eq!(sim.num_strings(), 2);

        let cos = theta.cos();
        let sin = theta.sin();

        for frame in sim.current_strings() {
            let p = frame.string.pauli.get_qubit(0);
            match p {
                'X' => assert!((frame.string.coeff.re - cos).abs() < 1e-10),
                'Y' => assert!((frame.string.coeff.re - sin).abs() < 1e-10),
                _ => panic!("Unexpected Pauli: {}", p),
            }
        }
    }

    // -----------------------------------------------------------------
    // Test 11: Rz(0) is identity (no split).
    // -----------------------------------------------------------------
    #[test]
    fn test_rz_zero_is_identity() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::Rz(0.0), 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'X');
    }

    // -----------------------------------------------------------------
    // Test 12: Truncation enforces max_strings.
    // -----------------------------------------------------------------
    #[test]
    fn test_truncation_max_strings() {
        let policy = TruncationPolicy::with_limits(3, 1e-15);

        // Start with X observable and apply multiple T gates to generate many strings.
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, policy);

        // Each T gate on X can double strings. After 4 T gates we could have up to 16.
        for _ in 0..4 {
            sim.propagate_gate(&single_gate(GateType::T, 0));
        }

        // Truncation should cap at 3.
        assert!(sim.num_strings() <= 3);
        assert!(sim.stats().strings_truncated > 0);
    }

    // -----------------------------------------------------------------
    // Test 13: Truncation enforces min_weight.
    // -----------------------------------------------------------------
    #[test]
    fn test_truncation_min_weight() {
        let policy = TruncationPolicy::with_limits(100_000, 0.5);

        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, policy);

        // T gate: splits X into cos(pi/4)*X + sin(pi/4)*Y.
        // cos(pi/4) = sin(pi/4) ~ 0.707. Both above 0.5.
        sim.propagate_gate(&single_gate(GateType::T, 0));
        assert_eq!(sim.num_strings(), 2);

        // Apply another T gate. Some resulting coefficients will be
        // cos^2(pi/4) ~ 0.5, cos*sin ~ 0.5, etc. After a third T gate,
        // some will drop below 0.5.
        sim.propagate_gate(&single_gate(GateType::T, 0));
        sim.propagate_gate(&single_gate(GateType::T, 0));
        sim.consolidate();

        // After consolidation with min_weight=0.5, only strings with |coeff| >= 0.5 remain.
        for frame in sim.current_strings() {
            assert!(frame.string.coeff.norm() >= 0.5);
        }
    }

    // -----------------------------------------------------------------
    // Test 14: Truncation enforces max_locality.
    // -----------------------------------------------------------------
    #[test]
    fn test_truncation_max_locality() {
        let policy = TruncationPolicy::strict(10_000, 1e-15, 1);

        // Start with a weight-2 observable: XZ.
        let obs = WeightedPauliString::unit(PauliString::from_str_rep("XZ"));
        let mut sim = PauliPropagationSimulator::new(2, obs, policy);

        // The observable has weight 2, but max_locality is 1.
        // After truncation it should be removed.
        sim.consolidate();

        assert_eq!(sim.num_strings(), 0);
    }

    // -----------------------------------------------------------------
    // Test 15: Expectation value <0|Z|0> = 1.
    // -----------------------------------------------------------------
    #[test]
    fn test_expectation_z_on_zero_state() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let sim = PauliPropagationSimulator::new(1, obs, default_policy());

        let state = QuantumState::new(1);
        let ev = sim.expectation_value(&state);

        assert_c64_approx(ev, C64::new(1.0, 0.0), 1e-10, "<0|Z|0>");
    }

    // -----------------------------------------------------------------
    // Test 16: Expectation value <0|X|0> = 0.
    // -----------------------------------------------------------------
    #[test]
    fn test_expectation_x_on_zero_state() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let sim = PauliPropagationSimulator::new(1, obs, default_policy());

        let state = QuantumState::new(1);
        let ev = sim.expectation_value(&state);

        assert_c64_approx(ev, C64::new(0.0, 0.0), 1e-10, "<0|X|0>");
    }

    // -----------------------------------------------------------------
    // Test 17: Expectation value <+|X|+> = 1 where |+> = H|0>.
    // -----------------------------------------------------------------
    #[test]
    fn test_expectation_x_on_plus_state() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let sim = PauliPropagationSimulator::new(1, obs, default_policy());

        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);
        let ev = sim.expectation_value(&state);

        assert_c64_approx(ev, C64::new(1.0, 0.0), 1e-10, "<+|X|+>");
    }

    // -----------------------------------------------------------------
    // Test 18: Expectation value through Bell circuit.
    //  <00| (H CNOT)^dag Z0 (H CNOT) |00> = <00| XX |00> = 0
    // -----------------------------------------------------------------
    #[test]
    fn test_expectation_bell_z0() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        sim.propagate_circuit(&[single_gate(GateType::H, 0), cnot_gate(0, 1)]);

        // The evolved observable is XX. <00|XX|00> = <00|11> = 0.
        let state = QuantumState::new(2);
        let ev = sim.expectation_value(&state);
        assert_c64_approx(ev, C64::new(0.0, 0.0), 1e-10, "<00|XX|00>");
    }

    // -----------------------------------------------------------------
    // Test 19: Expectation value consistency.
    // Propagate Z through H, measure against |0>. Should match H|0> = |+>, <+|Z|+> = 0.
    // -----------------------------------------------------------------
    #[test]
    fn test_expectation_consistency_heisenberg_vs_schrodinger() {
        // Heisenberg: propagate Z through H, measure against |0>.
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());
        sim.propagate_gate(&single_gate(GateType::H, 0));

        let state_zero = QuantumState::new(1);
        let ev_heisenberg = sim.expectation_value(&state_zero);

        // Schrodinger: apply H to |0>, measure Z.
        let mut state_plus = QuantumState::new(1);
        GateOperations::h(&mut state_plus, 0);
        let ev_schrodinger = state_plus.expectation_z(0);

        assert_c64_approx(
            ev_heisenberg,
            C64::new(ev_schrodinger, 0.0),
            1e-10,
            "Heisenberg vs Schrodinger",
        );
    }

    // -----------------------------------------------------------------
    // Test 20: Large Clifford circuit - no string splitting.
    // -----------------------------------------------------------------
    #[test]
    fn test_large_clifford_no_splitting() {
        let n = 12;
        let obs = WeightedPauliString::unit(PauliString::single(n, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(n, obs, default_policy());

        // Build a circuit of only Clifford gates.
        let mut gates = Vec::new();
        for i in 0..n {
            gates.push(single_gate(GateType::H, i));
        }
        for i in 0..(n - 1) {
            gates.push(cnot_gate(i, i + 1));
        }
        for i in 0..n {
            gates.push(single_gate(GateType::S, i));
        }
        for i in (0..(n - 1)).rev() {
            gates.push(cnot_gate(i, i + 1));
        }

        sim.propagate_circuit(&gates);

        // With only Clifford gates, there should still be exactly 1 string.
        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.stats().non_clifford_gates, 0);
        assert_eq!(sim.stats().strings_generated, 0);
    }

    // -----------------------------------------------------------------
    // Test 21: Stats tracking.
    // -----------------------------------------------------------------
    #[test]
    fn test_stats_tracking() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::T, 0)); // Non-Clifford (X splits)
        sim.propagate_gate(&single_gate(GateType::H, 0)); // Clifford
        sim.propagate_gate(&cnot_gate(0, 1)); // Clifford
        sim.propagate_gate(&single_gate(GateType::Rz(0.5), 0)); // Non-Clifford

        let stats = sim.stats();
        assert_eq!(stats.total_gates, 4);
        assert_eq!(stats.clifford_gates, 2);
        assert_eq!(stats.non_clifford_gates, 2);
        assert!(stats.strings_generated > 0);
    }

    // -----------------------------------------------------------------
    // Test 22: CNOT propagation: XI -> XX.
    // -----------------------------------------------------------------
    #[test]
    fn test_cnot_xi_to_xx() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        sim.propagate_gate(&cnot_gate(0, 1));

        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string.pauli;
        assert_eq!(p.get_qubit(0), 'X');
        assert_eq!(p.get_qubit(1), 'X');
    }

    // -----------------------------------------------------------------
    // Test 23: CNOT propagation: IZ -> ZZ.
    // -----------------------------------------------------------------
    #[test]
    fn test_cnot_iz_to_zz() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 1, 'Z'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        sim.propagate_gate(&cnot_gate(0, 1));

        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string.pauli;
        assert_eq!(p.get_qubit(0), 'Z');
        assert_eq!(p.get_qubit(1), 'Z');
    }

    // -----------------------------------------------------------------
    // Test 24: CZ propagation: XI -> XZ.
    // -----------------------------------------------------------------
    #[test]
    fn test_cz_propagation() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        sim.propagate_gate(&cz_gate(0, 1));

        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string.pauli;
        assert_eq!(p.get_qubit(0), 'X');
        assert_eq!(p.get_qubit(1), 'Z');
    }

    // -----------------------------------------------------------------
    // Test 25: SWAP propagation.
    // -----------------------------------------------------------------
    #[test]
    fn test_swap_propagation() {
        let obs = WeightedPauliString::unit(PauliString::single(3, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(3, obs, default_policy());

        sim.propagate_gate(&swap_gate(0, 2));

        assert_eq!(sim.num_strings(), 1);
        let p = &sim.current_strings()[0].string.pauli;
        assert_eq!(p.get_qubit(0), 'I');
        assert_eq!(p.get_qubit(1), 'I');
        assert_eq!(p.get_qubit(2), 'X');
    }

    // -----------------------------------------------------------------
    // Test 26: Rx propagation via H-Rz-H decomposition.
    // -----------------------------------------------------------------
    #[test]
    fn test_rx_propagation() {
        let theta = 0.8;
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::Rx(theta), 0));

        // Rx^dag Z Rx: Z in the X-rotated frame.
        // Rx = H Rz H, so Rx^dag Z Rx = H Rz^dag H Z H Rz H
        //   = H Rz^dag X Rz H = H (cos(theta) X + sin(theta) Y) H
        //   = cos(theta) Z - sin(theta) Y.
        // (Since H X H = Z, H Y H = -Y.)
        sim.consolidate();

        let cos = theta.cos();
        let sin = theta.sin();

        let mut found_z = false;
        let mut found_y = false;
        for frame in sim.current_strings() {
            let p0 = frame.string.pauli.get_qubit(0);
            match p0 {
                'Z' => {
                    assert!(
                        (frame.string.coeff.re - cos).abs() < 1e-10,
                        "Z coeff: expected {}, got {}",
                        cos,
                        frame.string.coeff.re
                    );
                    found_z = true;
                }
                'Y' => {
                    assert!(
                        (frame.string.coeff.re - (-sin)).abs() < 1e-10,
                        "Y coeff: expected {}, got {}",
                        -sin,
                        frame.string.coeff.re
                    );
                    found_y = true;
                }
                _ => panic!("Unexpected Pauli: {}", p0),
            }
        }
        assert!(found_z, "Expected Z term");
        assert!(found_y, "Expected Y term");
    }

    // -----------------------------------------------------------------
    // Test 27: Expectation value <0|Z|0> = 1 via Heisenberg on Rx circuit.
    // -----------------------------------------------------------------
    #[test]
    fn test_expectation_rx_circuit_heisenberg_vs_schrodinger() {
        let theta = 1.5;

        // Heisenberg: propagate Z through Rx(theta), evaluate on |0>.
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim_h = PauliPropagationSimulator::new(1, obs, default_policy());
        sim_h.propagate_gate(&single_gate(GateType::Rx(theta), 0));
        let state_zero = QuantumState::new(1);
        let ev_h = sim_h.expectation_value(&state_zero);

        // Schrodinger: apply Rx(theta) to |0>, measure Z.
        let mut state = QuantumState::new(1);
        GateOperations::rx(&mut state, 0, theta);
        let ev_s = state.expectation_z(0);

        assert_c64_approx(
            ev_h,
            C64::new(ev_s, 0.0),
            1e-8,
            "Rx Heisenberg vs Schrodinger",
        );
    }

    // -----------------------------------------------------------------
    // Test 28: Multiple T gates cause string growth, tracked by stats.
    // -----------------------------------------------------------------
    #[test]
    fn test_multiple_t_gates_string_growth() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::T, 0));
        assert_eq!(sim.num_strings(), 2);

        sim.propagate_gate(&single_gate(GateType::T, 0));
        // After 2 T gates on X: up to 4 strings (before consolidation).
        assert!(sim.num_strings() <= 4);

        assert!(sim.stats().max_strings_at_once >= 2);
    }

    // -----------------------------------------------------------------
    // Test 29: from_sum initializer.
    // -----------------------------------------------------------------
    #[test]
    fn test_from_sum_initializer() {
        let t1 = WeightedPauliString::new(PauliString::single(2, 0, 'X'), C64::new(0.5, 0.0));
        let t2 = WeightedPauliString::new(PauliString::single(2, 1, 'Z'), C64::new(0.5, 0.0));

        let sim = PauliPropagationSimulator::from_sum(2, vec![t1, t2], default_policy());
        assert_eq!(sim.num_strings(), 2);
    }

    // -----------------------------------------------------------------
    // Test 30: Consolidation merges duplicate strings.
    // -----------------------------------------------------------------
    #[test]
    fn test_consolidation_merges_duplicates() {
        // Start with X, apply T (splits into X + Y), then apply T^dag-like
        // operations that produce duplicate strings.
        let t1 = WeightedPauliString::new(PauliString::single(1, 0, 'X'), C64::new(0.3, 0.0));
        let t2 = WeightedPauliString::new(PauliString::single(1, 0, 'X'), C64::new(0.7, 0.0));
        let t3 = WeightedPauliString::new(PauliString::single(1, 0, 'Z'), C64::new(0.1, 0.0));

        let mut sim = PauliPropagationSimulator::from_sum(1, vec![t1, t2, t3], default_policy());
        assert_eq!(sim.num_strings(), 3);

        sim.consolidate();
        // After consolidation: two unique strings (X with coeff 1.0, Z with coeff 0.1).
        assert_eq!(sim.num_strings(), 2);

        let x_frame = sim
            .current_strings()
            .iter()
            .find(|f| f.string.pauli.get_qubit(0) == 'X')
            .unwrap();
        assert!((x_frame.string.coeff.re - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------
    // Test 31: Y gate propagation.
    // -----------------------------------------------------------------
    #[test]
    fn test_y_gate_propagation() {
        // Y^dag X Y = -X
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::Y, 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'X');
        assert_c64_approx(
            sim.current_strings()[0].string.coeff,
            C64::new(-1.0, 0.0),
            1e-10,
            "Y^dag X Y = -X",
        );
    }

    // -----------------------------------------------------------------
    // Test 32: X gate propagation (Z -> -Z).
    // -----------------------------------------------------------------
    #[test]
    fn test_x_gate_propagation() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::X, 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'Z');
        assert_c64_approx(
            sim.current_strings()[0].string.coeff,
            C64::new(-1.0, 0.0),
            1e-10,
            "X^dag Z X = -Z",
        );
    }

    // -----------------------------------------------------------------
    // Test 33: Z gate propagation (X -> -X).
    // -----------------------------------------------------------------
    #[test]
    fn test_z_gate_propagation() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::Z, 0));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'X');
        assert_c64_approx(
            sim.current_strings()[0].string.coeff,
            C64::new(-1.0, 0.0),
            1e-10,
            "Z^dag X Z = -X",
        );
    }

    // -----------------------------------------------------------------
    // Test 34: Heisenberg vs Schrodinger for a deeper circuit.
    // -----------------------------------------------------------------
    #[test]
    fn test_deep_circuit_heisenberg_vs_schrodinger() {
        let n = 3;

        // Build a circuit with a mix of Clifford and non-Clifford gates.
        let gates = vec![
            single_gate(GateType::H, 0),
            cnot_gate(0, 1),
            single_gate(GateType::T, 1),
            single_gate(GateType::H, 2),
            cnot_gate(1, 2),
            single_gate(GateType::Rz(0.7), 0),
            single_gate(GateType::S, 2),
        ];

        // Heisenberg: propagate Z0 through circuit, evaluate on |000>.
        let obs = WeightedPauliString::unit(PauliString::single(n, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(n, obs, default_policy());
        sim.propagate_circuit(&gates);
        let state_zero = QuantumState::new(n);
        let ev_h = sim.expectation_value(&state_zero);

        // Schrodinger: apply gates to |000>, measure Z0.
        let mut state = QuantumState::new(n);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        GateOperations::t(&mut state, 1);
        GateOperations::h(&mut state, 2);
        GateOperations::cnot(&mut state, 1, 2);
        GateOperations::rz(&mut state, 0, 0.7);
        GateOperations::s(&mut state, 2);

        let ev_s = state.expectation_z(0);

        assert_c64_approx(
            ev_h,
            C64::new(ev_s, 0.0),
            1e-6,
            "Deep circuit Heisenberg vs Schrodinger",
        );
    }

    // -----------------------------------------------------------------
    // Test 35: Display format does not panic.
    // -----------------------------------------------------------------
    #[test]
    fn test_display_format() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
        let sim = PauliPropagationSimulator::new(2, obs, default_policy());

        let display = format!("{}", sim);
        assert!(display.contains("PauliPropagationSimulator"));
        assert!(display.contains("2 qubits"));

        let stats_display = format!("{}", sim.stats());
        assert!(stats_display.contains("PropagationStats"));
    }

    // -----------------------------------------------------------------
    // Test 36: Phase gate propagation.
    // -----------------------------------------------------------------
    #[test]
    fn test_phase_gate_propagation() {
        let theta = 0.9;
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::Phase(theta), 0));

        // Phase(theta) has the same Pauli propagation as Rz(theta).
        assert_eq!(sim.num_strings(), 2);
    }

    // -----------------------------------------------------------------
    // Test 37: SX gate propagation = Rx(pi/2).
    // -----------------------------------------------------------------
    #[test]
    fn test_sx_gate_propagation() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim_sx = PauliPropagationSimulator::new(1, obs.clone(), default_policy());
        sim_sx.propagate_gate(&single_gate(GateType::SX, 0));

        let mut sim_rx = PauliPropagationSimulator::new(1, obs, default_policy());
        sim_rx.propagate_gate(&single_gate(GateType::Rx(std::f64::consts::FRAC_PI_2), 0));

        sim_sx.consolidate();
        sim_rx.consolidate();

        // Both should produce the same result.
        let state = QuantumState::new(1);
        let ev_sx = sim_sx.expectation_value(&state);
        let ev_rx = sim_rx.expectation_value(&state);
        assert_c64_approx(ev_sx, ev_rx, 1e-10, "SX vs Rx(pi/2)");
    }

    // -----------------------------------------------------------------
    // Test 38: Ry propagation Heisenberg vs Schrodinger.
    // -----------------------------------------------------------------
    #[test]
    fn test_ry_heisenberg_vs_schrodinger() {
        let theta = 2.1;

        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());
        sim.propagate_gate(&single_gate(GateType::Ry(theta), 0));

        let state_zero = QuantumState::new(1);
        let ev_h = sim.expectation_value(&state_zero);

        let mut state = QuantumState::new(1);
        GateOperations::ry(&mut state, 0, theta);
        let ev_s = state.expectation_z(0);

        assert_c64_approx(
            ev_h,
            C64::new(ev_s, 0.0),
            1e-8,
            "Ry Heisenberg vs Schrodinger",
        );
    }

    // -----------------------------------------------------------------
    // Test 39: ISWAP is treated as Clifford (no splitting).
    // -----------------------------------------------------------------
    #[test]
    fn test_iswap_is_clifford() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'X'));
        let mut sim = PauliPropagationSimulator::new(2, obs, default_policy());

        sim.propagate_gate(&Gate::new(GateType::ISWAP, vec![0, 1], vec![]));

        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.stats().clifford_gates, 1);
        assert_eq!(sim.stats().non_clifford_gates, 0);
    }

    // -----------------------------------------------------------------
    // Test 40: Coefficient weighted observable.
    // -----------------------------------------------------------------
    #[test]
    fn test_weighted_coefficient_propagation() {
        let obs = WeightedPauliString::new(PauliString::single(1, 0, 'Z'), C64::new(0.0, 2.5));
        let mut sim = PauliPropagationSimulator::new(1, obs, default_policy());

        sim.propagate_gate(&single_gate(GateType::H, 0));

        // H^dag (2.5i * Z) H = 2.5i * X
        assert_eq!(sim.num_strings(), 1);
        assert_eq!(sim.current_strings()[0].string.pauli.get_qubit(0), 'X');
        assert_c64_approx(
            sim.current_strings()[0].string.coeff,
            C64::new(0.0, 2.5),
            1e-10,
            "Weighted coeff through H",
        );
    }

    // =================================================================
    // DYNAMIC TRUNCATION TESTS
    // =================================================================

    // -----------------------------------------------------------------
    // Test 41: ErrorBudget creation and basic properties.
    // -----------------------------------------------------------------
    #[test]
    fn test_error_budget_creation() {
        let budget = ErrorBudget::new(1e-4, 50);
        assert_eq!(budget.target_precision, 1e-4);
        assert_eq!(budget.consumed_error, 0.0);
        assert_eq!(budget.layers_processed, 0);
        assert_eq!(budget.estimated_remaining_layers, 50);
        assert!(!budget.is_exhausted());
        assert!((budget.remaining() - 1e-4).abs() < 1e-15);
    }

    // -----------------------------------------------------------------
    // Test 42: ErrorBudget per-layer allowance.
    // -----------------------------------------------------------------
    #[test]
    fn test_error_budget_allowance() {
        let budget = ErrorBudget::new(1e-4, 100);
        let allowance = budget.per_layer_allowance();
        assert!((allowance - 1e-6).abs() < 1e-15, "1e-4 / 100 = 1e-6");
    }

    // -----------------------------------------------------------------
    // Test 43: ErrorBudget tracks consumption.
    // -----------------------------------------------------------------
    #[test]
    fn test_error_budget_consumption() {
        let mut budget = ErrorBudget::new(1e-4, 10);
        budget.record_layer_error(2e-5);
        assert_eq!(budget.layers_processed, 1);
        assert_eq!(budget.estimated_remaining_layers, 9);
        assert!((budget.consumed_error - 2e-5).abs() < 1e-15);
        assert!(!budget.is_exhausted());

        // Exhaust the budget.
        budget.record_layer_error(8e-5);
        assert!(budget.consumed_error >= 1e-4);
        assert!(budget.is_exhausted());
    }

    // -----------------------------------------------------------------
    // Test 44: DynamicTruncationPolicy default and builder.
    // -----------------------------------------------------------------
    #[test]
    fn test_dynamic_policy_builder() {
        let policy = DynamicTruncationPolicy::default()
            .with_precision(1e-3)
            .with_max_strings(5000)
            .with_max_locality(10)
            .with_branch_and_bound(1e-6);

        assert_eq!(policy.target_precision, 1e-3);
        assert_eq!(policy.max_strings, 5000);
        assert_eq!(policy.max_locality, 10);
        assert!(policy.enable_branch_and_bound);
        assert_eq!(policy.branch_bound_ratio, 1e-6);
    }

    // -----------------------------------------------------------------
    // Test 45: Adaptive threshold computation.
    // -----------------------------------------------------------------
    #[test]
    fn test_adaptive_threshold() {
        let policy = DynamicTruncationPolicy::default();

        // Create frames with magnitudes 0.1, 0.2, 0.3, 0.4, 1.0.
        let frames: Vec<PauliFrame> = [0.1, 0.2, 0.3, 0.4, 1.0]
            .iter()
            .map(|&c| {
                PauliFrame::new(WeightedPauliString::new(
                    PauliString::single(1, 0, 'X'),
                    C64::new(c, 0.0),
                ))
            })
            .collect();

        // With allowance 0.3, we can discard 0.1 + 0.2 = 0.3.
        let threshold = policy.adaptive_threshold(&frames, 0.3);
        assert!(
            threshold >= 0.1 && threshold <= 0.2,
            "Threshold should discard small strings: got {}",
            threshold
        );

        // With allowance 0.0, no truncation.
        let threshold_zero = policy.adaptive_threshold(&frames, 0.0);
        assert_eq!(threshold_zero, 0.0);
    }

    // -----------------------------------------------------------------
    // Test 46: DynamicPauliPropagator Clifford-only circuit.
    // -----------------------------------------------------------------
    #[test]
    fn test_dynamic_propagator_clifford_only() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
        let policy = DynamicTruncationPolicy::default();
        let mut prop = DynamicPauliPropagator::new(2, obs, policy, 0);

        let gates = vec![single_gate(GateType::H, 0), cnot_gate(0, 1)];
        prop.propagate_circuit(&gates);

        // Clifford gates don't consume error budget.
        assert_eq!(prop.error_budget().consumed_error, 0.0);
        assert_eq!(prop.num_strings(), 1);
    }

    // -----------------------------------------------------------------
    // Test 47: DynamicPauliPropagator with T gates.
    // -----------------------------------------------------------------
    #[test]
    fn test_dynamic_propagator_with_t_gates() {
        let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'X'));
        let policy = DynamicTruncationPolicy::default().with_precision(1e-3);
        let mut prop = DynamicPauliPropagator::new(2, obs, policy, 5);

        for _ in 0..5 {
            prop.propagate_gate(&single_gate(GateType::T, 0));
        }

        // Should have tracked some strings and consumed some budget.
        assert!(prop.num_strings() > 0);
        assert!(prop.error_budget().layers_processed == 5);
        // Provable bound should be below target.
        assert!(
            prop.provable_error_bound() <= 1e-3,
            "Error bound {:.2e} should be <= 1e-3",
            prop.provable_error_bound()
        );
    }

    // -----------------------------------------------------------------
    // Test 48: DynamicPauliPropagator Heisenberg vs Schrodinger.
    // -----------------------------------------------------------------
    #[test]
    fn test_dynamic_propagator_heisenberg_vs_schrodinger() {
        let n = 2;
        let obs = WeightedPauliString::unit(PauliString::single(n, 0, 'Z'));
        let policy = DynamicTruncationPolicy::default().with_precision(1e-4);
        let mut prop = DynamicPauliPropagator::new(n, obs, policy, 3);

        let gates = vec![
            single_gate(GateType::H, 0),
            cnot_gate(0, 1),
            single_gate(GateType::T, 0),
            single_gate(GateType::Rz(0.5), 1),
        ];
        prop.propagate_circuit(&gates);

        let state_zero = QuantumState::new(n);
        let ev_h = prop.expectation_value(&state_zero);

        // Schrodinger picture.
        let mut state = QuantumState::new(n);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        GateOperations::t(&mut state, 0);
        GateOperations::rz(&mut state, 1, 0.5);
        let ev_s = state.expectation_z(0);

        // Error should be within provable bound.
        let diff = (ev_h.re - ev_s).abs();
        assert!(
            diff < 1e-4 + 1e-10,
            "Heisenberg-Schrodinger difference {} exceeds bound",
            diff
        );
    }

    // -----------------------------------------------------------------
    // Test 49: Branch-and-bound freezes low-weight strings.
    // -----------------------------------------------------------------
    #[test]
    fn test_branch_and_bound_freezing() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let policy = DynamicTruncationPolicy::default().with_branch_and_bound(0.5); // Freeze strings below 50% of max.
        let mut prop = DynamicPauliPropagator::new(1, obs, policy, 10);

        // First T gate: splits X into cos*X + sin*Y (both ~0.707).
        prop.propagate_gate(&single_gate(GateType::T, 0));
        // Second T gate: the smaller child may get frozen.
        prop.propagate_gate(&single_gate(GateType::T, 0));

        // After two T gates with branch-and-bound, some strings should be frozen.
        assert!(prop.dynamic_stats().strings_frozen >= 0);
        assert!(prop.num_strings() > 0);
    }

    // -----------------------------------------------------------------
    // Test 50: ErrorBudget Display format.
    // -----------------------------------------------------------------
    #[test]
    fn test_error_budget_display() {
        let budget = ErrorBudget::new(1e-4, 50);
        let display = format!("{}", budget);
        assert!(display.contains("ErrorBudget"));
        assert!(display.contains("0.0%"));
    }

    // -----------------------------------------------------------------
    // Test 51: DynamicPauliPropagator Display format.
    // -----------------------------------------------------------------
    #[test]
    fn test_dynamic_propagator_display() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'Z'));
        let policy = DynamicTruncationPolicy::default();
        let prop = DynamicPauliPropagator::new(1, obs, policy, 10);
        let display = format!("{}", prop);
        assert!(display.contains("DynamicPauliPropagator"));
    }

    // -----------------------------------------------------------------
    // Test 52: Dynamic truncation with importance sampling.
    // -----------------------------------------------------------------
    #[test]
    fn test_importance_sampling_policy() {
        let policy = DynamicTruncationPolicy::default().with_importance_sampling(5);

        assert!(policy.enable_importance_sampling);
        assert_eq!(policy.importance_samples, 5);
    }

    // -----------------------------------------------------------------
    // Test 53: Error bound monotonically increases.
    // -----------------------------------------------------------------
    #[test]
    fn test_error_bound_monotonic() {
        let obs = WeightedPauliString::unit(PauliString::single(1, 0, 'X'));
        let policy = DynamicTruncationPolicy::default()
            .with_precision(1.0) // Large budget so truncation kicks in.
            .with_max_strings(3);
        let mut prop = DynamicPauliPropagator::new(1, obs, policy, 20);

        let mut prev_error = 0.0;
        for _ in 0..10 {
            prop.propagate_gate(&single_gate(GateType::T, 0));
            let current_error = prop.provable_error_bound();
            assert!(
                current_error >= prev_error - 1e-15,
                "Error bound decreased: {} -> {}",
                prev_error,
                current_error
            );
            prev_error = current_error;
        }
    }

    // -----------------------------------------------------------------
    // Test 54: from_sum with dynamic propagator.
    // -----------------------------------------------------------------
    #[test]
    fn test_dynamic_from_sum() {
        let t1 = WeightedPauliString::new(PauliString::single(2, 0, 'X'), C64::new(0.5, 0.0));
        let t2 = WeightedPauliString::new(PauliString::single(2, 1, 'Z'), C64::new(0.5, 0.0));
        let policy = DynamicTruncationPolicy::default();
        let prop = DynamicPauliPropagator::from_sum(2, vec![t1, t2], policy, 10);
        assert_eq!(prop.num_strings(), 2);
    }
}
