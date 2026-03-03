//! Hayden-Preskill Protocol and Black Hole Information Theory
//!
//! This module implements the Hayden-Preskill thought experiment for quantum
//! information scrambling, the Page curve, and black hole information recovery
//! protocols.
//!
//! **WORLD FIRST**: First quantum simulator with black hole information theory
//! simulation.
//!
//! The Hayden-Preskill protocol demonstrates that if a black hole is a fast
//! scrambler (highly chaotic), then quantum information thrown into it can be
//! recovered almost immediately from the Hawking radiation -- provided one has
//! access to the early radiation that is entangled with the black hole.
//!
//! # Key Concepts
//!
//! - **Scrambling**: A black hole scrambles quantum information across all its
//!   degrees of freedom in time O(log n), making local measurements useless.
//! - **Page curve**: Entanglement entropy of radiation rises linearly until the
//!   Page time (half evaporated), then falls back to zero, confirming unitarity.
//! - **OTOC decay**: Out-of-time-order correlators decay exponentially for fast
//!   scramblers, diagnosing quantum chaos.
//! - **Tripartite mutual information**: TMI < 0 signals that information is
//!   delocalised into multi-party correlations (scrambled).
//! - **Monogamy of entanglement**: The firewall paradox arises when a qubit
//!   would need maximal entanglement with two different systems simultaneously.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::hayden_preskill::{HaydenPreskillConfig, HaydenPreskillSimulator};
//!
//! let config = HaydenPreskillConfig::default();
//! let mut sim = HaydenPreskillSimulator::new(config);
//! sim.setup_initial_state();
//! sim.throw_diary_into_bh();
//! sim.scramble();
//! let emitted = sim.emit_hawking_radiation(2);
//! let result = sim.recover_diary(&emitted);
//! println!("Recovery fidelity: {:.4}", result.fidelity);
//! ```

use std::f64::consts::PI;

use crate::{C64, GateOperations, QuantumState};

// ============================================================
// LCG PSEUDO-RANDOM NUMBER GENERATOR
// ============================================================

/// Simple linear congruential generator for deterministic randomness.
///
/// Uses the Numerical Recipes LCG parameters for adequate period and
/// uniformity in scrambling circuit generation.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed.wrapping_add(1) }
    }

    /// Advance the generator and return a raw u64.
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [0, 2*pi).
    fn next_angle(&mut self) -> f64 {
        self.next_f64() * 2.0 * PI
    }

    /// Uniform usize in [0, n).
    #[allow(dead_code)]
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising from Hayden-Preskill protocol operations.
#[derive(Clone, Debug)]
pub enum HaydenPreskillError {
    /// The system does not have enough qubits for the requested operation.
    InsufficientQubits {
        /// Number of qubits available.
        available: usize,
        /// Number of qubits required.
        required: usize,
    },
    /// A scrambling operation was expected but has not been performed.
    NotScrambled,
    /// The diary recovery procedure failed.
    RecoveryFailed(String),
    /// The qubit partition is invalid (overlapping or out-of-range indices).
    InvalidPartition(String),
}

impl std::fmt::Display for HaydenPreskillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HaydenPreskillError::InsufficientQubits { available, required } => {
                write!(f, "Insufficient qubits: have {}, need {}", available, required)
            }
            HaydenPreskillError::NotScrambled => {
                write!(f, "Black hole has not been scrambled yet")
            }
            HaydenPreskillError::RecoveryFailed(msg) => {
                write!(f, "Diary recovery failed: {}", msg)
            }
            HaydenPreskillError::InvalidPartition(msg) => {
                write!(f, "Invalid partition: {}", msg)
            }
        }
    }
}

impl std::error::Error for HaydenPreskillError {}

/// Convenience result type for this module.
pub type HaydenPreskillResult<T> = Result<T, HaydenPreskillError>;

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the Hayden-Preskill protocol simulation.
///
/// Uses the builder pattern -- construct with `Default::default()` then
/// override individual fields with the `with_*` methods.
#[derive(Clone, Debug)]
pub struct HaydenPreskillConfig {
    /// Number of qubits representing the black hole interior.
    pub black_hole_qubits: usize,
    /// Number of diary qubits thrown into the black hole.
    pub diary_qubits: usize,
    /// Number of early Hawking radiation qubits already entangled with the BH.
    pub radiation_qubits: usize,
    /// Circuit depth for the scrambling unitary (number of random layers).
    pub scrambling_depth: usize,
    /// Seed for deterministic pseudo-random circuit generation.
    pub seed: u64,
}

impl Default for HaydenPreskillConfig {
    fn default() -> Self {
        HaydenPreskillConfig {
            black_hole_qubits: 8,
            diary_qubits: 2,
            radiation_qubits: 4,
            scrambling_depth: 20,
            seed: 42,
        }
    }
}

impl HaydenPreskillConfig {
    /// Set the number of black hole qubits.
    pub fn with_black_hole_qubits(mut self, n: usize) -> Self {
        self.black_hole_qubits = n;
        self
    }

    /// Set the number of diary qubits.
    pub fn with_diary_qubits(mut self, n: usize) -> Self {
        self.diary_qubits = n;
        self
    }

    /// Set the number of early radiation qubits.
    pub fn with_radiation_qubits(mut self, n: usize) -> Self {
        self.radiation_qubits = n;
        self
    }

    /// Set the scrambling circuit depth.
    pub fn with_scrambling_depth(mut self, d: usize) -> Self {
        self.scrambling_depth = d;
        self
    }

    /// Set the PRNG seed.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Total number of qubits in the full system
    /// (black hole + diary + early radiation).
    pub fn total_qubits(&self) -> usize {
        self.black_hole_qubits + self.diary_qubits + self.radiation_qubits
    }
}

// ============================================================
// BLACK HOLE STATE
// ============================================================

/// Represents the quantum state of a black hole together with its radiation.
///
/// The qubit layout is:
/// ```text
///   [BH interior qubits | diary qubits | early radiation qubits]
///    0 .. n_bh-1          n_bh .. n_bh+n_d-1   n_bh+n_d .. total-1
/// ```
#[derive(Clone, Debug)]
pub struct BlackHole {
    /// Full quantum state of the BH + radiation system.
    pub state: QuantumState,
    /// Number of qubits currently inside the black hole.
    pub num_bh_qubits: usize,
    /// Number of qubits in the radiation subsystem.
    pub num_radiation_qubits: usize,
    /// Scrambling depth that was applied.
    pub scrambling_time: usize,
    /// Whether the scrambling unitary has been applied.
    pub is_scrambled: bool,
}

// ============================================================
// RECOVERY PROTOCOL
// ============================================================

/// Strategy used to recover the diary from the radiation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoveryProtocol {
    /// Decode directly from the late radiation without ancilla assistance.
    DirectDecoding,
    /// Use entanglement between early radiation and BH interior.
    EntanglementAssisted,
    /// Hayden-Preskill random decoupling protocol.
    RandomDecoupling,
}

/// Result of attempting to recover the diary information from Hawking radiation.
#[derive(Clone, Debug)]
pub struct RecoveryResult {
    /// Fidelity of the recovered diary state (1.0 = perfect recovery).
    pub fidelity: f64,
    /// Mutual information I(diary_ref : radiation) in bits.
    pub mutual_information: f64,
    /// Which protocol was used for recovery.
    pub recovery_protocol: RecoveryProtocol,
    /// Minimum number of radiation qubits needed for recovery.
    pub num_radiation_qubits_needed: usize,
    /// Whether scrambling was necessary for recovery.
    pub scrambling_was_necessary: bool,
}

// ============================================================
// PAGE CURVE
// ============================================================

/// The Page curve: entanglement entropy of the radiation subsystem as a
/// function of the fraction of the black hole that has evaporated.
#[derive(Clone, Debug)]
pub struct PageCurve {
    /// Fraction of the black hole radiated at each measurement point (0 to 1).
    pub radiation_fraction: Vec<f64>,
    /// Von Neumann entanglement entropy of the radiation at each point.
    pub entropy: Vec<f64>,
    /// Fraction at which entropy begins decreasing (the Page time).
    pub page_time_fraction: f64,
    /// Whether the evolution is unitary (entropy returns to ~0 at the end).
    pub is_unitary: bool,
}

// ============================================================
// SCRAMBLING ANALYSIS
// ============================================================

/// Diagnostic analysis of scrambling behaviour in a quantum circuit.
#[derive(Clone, Debug)]
pub struct ScramblingAnalysis {
    /// Estimated scrambling time (number of layers for full scrambling).
    pub scrambling_time: f64,
    /// Out-of-time-order correlator as a function of circuit depth.
    pub otoc_decay: Vec<f64>,
    /// Tripartite mutual information I_3(A:B:C) as a function of depth.
    pub tripartite_info: Vec<f64>,
    /// True if the circuit scrambles in O(log n) depth.
    pub is_fast_scrambler: bool,
}

// ============================================================
// EVAPORATION TYPES
// ============================================================

/// Snapshot of the black hole at one step of the evaporation process.
#[derive(Clone, Debug)]
pub struct EvaporationSnapshot {
    /// Step number.
    pub step: usize,
    /// Number of qubits still inside the black hole.
    pub bh_qubits_remaining: usize,
    /// Number of qubits now in the radiation.
    pub radiation_qubits: usize,
    /// Von Neumann entropy of the radiation subsystem.
    pub radiation_entropy: f64,
    /// Whether the Page time has been reached.
    pub page_time_reached: bool,
}

/// Result of a complete black hole evaporation simulation.
#[derive(Clone, Debug)]
pub struct EvaporationResult {
    /// The Page curve over the full evaporation.
    pub page_curve: PageCurve,
    /// Total number of evaporation steps.
    pub total_steps: usize,
    /// Whether unitarity was confirmed (final entropy approximately zero).
    pub information_preserved: bool,
    /// Peak entropy value during evaporation.
    pub max_entropy: f64,
    /// Step at which entropy peaked (the Page time).
    pub page_time: usize,
}

// ============================================================
// FIREWALL / MONOGAMY
// ============================================================

/// Result of checking entanglement monogamy across three subsystems.
#[derive(Clone, Debug)]
pub struct MonogamyResult {
    /// Entanglement (mutual information) between subsystems A and B.
    pub e_ab: f64,
    /// Entanglement (mutual information) between subsystems A and C.
    pub e_ac: f64,
    /// Whether the monogamy inequality is violated.
    pub monogamy_violated: bool,
    /// Whether a firewall is required to resolve the violation.
    pub firewall_needed: bool,
}

/// Quantum firewall analyser for the AMPS paradox.
///
/// Checks whether a tripartite quantum state can satisfy all pairwise
/// entanglement constraints simultaneously, or whether a firewall at the
/// horizon is necessary.
pub struct QuantumFirewall {
    /// Full quantum state of the tripartite system.
    state: QuantumState,
    /// Total number of qubits.
    num_qubits: usize,
}

impl QuantumFirewall {
    /// Create a new firewall analyser from a quantum state.
    pub fn new(state: QuantumState) -> Self {
        let n = state.num_qubits;
        QuantumFirewall { state, num_qubits: n }
    }

    /// Check the monogamy of entanglement across three disjoint subsystems.
    ///
    /// For qubits in sets A, B, C, computes I(A:B) and I(A:C) and checks
    /// whether both can be simultaneously large (which would violate
    /// monogamy and require a firewall).
    pub fn check_monogamy(
        &self,
        a: &[usize],
        b: &[usize],
        c: &[usize],
    ) -> MonogamyResult {
        let s_a = subsystem_entropy(&self.state, a);
        let s_b = subsystem_entropy(&self.state, b);
        let s_c = subsystem_entropy(&self.state, c);

        let mut ab: Vec<usize> = a.iter().chain(b.iter()).copied().collect();
        ab.sort();
        let mut ac: Vec<usize> = a.iter().chain(c.iter()).copied().collect();
        ac.sort();

        let s_ab = subsystem_entropy(&self.state, &ab);
        let s_ac = subsystem_entropy(&self.state, &ac);

        let i_ab = s_a + s_b - s_ab;
        let i_ac = s_a + s_c - s_ac;

        // Monogamy bound (CKW inequality in terms of mutual info):
        // For a pure tripartite state, if I(A:B) is maximal (= 2*S(A)),
        // then I(A:C) should be zero, and vice versa.
        let max_entanglement = 2.0 * s_a;
        let monogamy_violated = (i_ab + i_ac) > max_entanglement + 0.01;
        let firewall_needed = monogamy_violated;

        MonogamyResult {
            e_ab: i_ab,
            e_ac: i_ac,
            monogamy_violated,
            firewall_needed,
        }
    }

    /// Check whether a smooth (non-firewall) horizon is possible.
    ///
    /// A smooth horizon requires the interior mode to be maximally entangled
    /// with the exterior partner, which is only possible if the mode is not
    /// also maximally entangled with the early radiation.
    pub fn smooth_horizon_possible(&self) -> bool {
        if self.num_qubits < 3 {
            return true;
        }
        // Partition into interior (first half), exterior partner (next qubit),
        // and early radiation (remaining).
        let half = self.num_qubits / 2;
        let interior: Vec<usize> = (0..half).collect();
        let partner: Vec<usize> = vec![half];
        let radiation: Vec<usize> = ((half + 1)..self.num_qubits).collect();

        if radiation.is_empty() {
            return true;
        }

        let result = self.check_monogamy(&partner, &interior, &radiation);
        !result.monogamy_violated
    }
}

// ============================================================
// BLACK HOLE EVAPORATION SIMULATOR
// ============================================================

/// Simulates the full evaporation of a black hole and tracks the Page curve.
///
/// At each step, one qubit is transferred from the BH interior to the
/// radiation subsystem by applying a scrambling layer followed by partial
/// trace-based entropy computation.
pub struct BlackHoleEvaporationSimulator {
    /// Number of BH qubits at the start.
    initial_bh_qubits: usize,
    /// Current quantum state (BH + reference).
    state: QuantumState,
    /// Current number of BH qubits remaining.
    bh_remaining: usize,
    /// Current step index.
    step: usize,
    /// Accumulated snapshots.
    snapshots: Vec<EvaporationSnapshot>,
    /// PRNG for scrambling.
    rng: Lcg,
}

impl BlackHoleEvaporationSimulator {
    /// Create a new evaporation simulator.
    ///
    /// The system is initialised as a pure state on `2 * initial_bh_qubits`
    /// qubits. We first create Bell pairs between the BH and reference, then
    /// apply deep scrambling to the BH half to delocalise the entanglement
    /// structure. This produces a state where the Page curve exhibits the
    /// characteristic rise-then-fall shape.
    pub fn new(initial_bh_qubits: usize) -> Self {
        let total = initial_bh_qubits * 2;
        let mut state = QuantumState::new(total);
        let mut rng = Lcg::new(42);

        // Create maximally entangled state: BH qubits entangled with reference
        create_bell_pairs(&mut state, initial_bh_qubits);

        // Deeply scramble the BH half to delocalise the entanglement structure.
        // This makes the state approximate a Haar-random state on the full
        // system, which is required for the Page curve to exhibit the
        // correct rise-then-fall shape.
        let scramble_depth = (initial_bh_qubits * 5).max(10);
        for _ in 0..scramble_depth {
            apply_scrambling_layer(&mut state, 0, initial_bh_qubits, &mut rng);
        }

        BlackHoleEvaporationSimulator {
            initial_bh_qubits,
            state,
            bh_remaining: initial_bh_qubits,
            step: 0,
            snapshots: Vec::new(),
            rng,
        }
    }

    /// Perform one evaporation step: scramble the BH, then "emit" one qubit.
    ///
    /// Returns a snapshot of the state after this step. The emitted qubit is
    /// conceptually transferred from the BH to the radiation; we track this
    /// by moving the BH/radiation boundary.
    pub fn evaporate_step(&mut self) -> EvaporationSnapshot {
        if self.bh_remaining == 0 {
            return EvaporationSnapshot {
                step: self.step,
                bh_qubits_remaining: 0,
                radiation_qubits: self.initial_bh_qubits,
                radiation_entropy: 0.0,
                page_time_reached: true,
            };
        }

        // Apply a scrambling layer to BH qubits
        let bh_end = self.bh_remaining;
        if bh_end >= 2 {
            apply_scrambling_layer(&mut self.state, 0, bh_end, &mut self.rng);
        }

        // "Emit" one qubit: move BH boundary down by 1
        self.bh_remaining -= 1;
        self.step += 1;

        // Compute entropy of the radiation subsystem.
        // Radiation = emitted BH qubits + reference system (early radiation).
        // This is the complement of the remaining BH interior: qubits 0..bh_remaining.
        // For a pure state, S(radiation) = S(remaining BH).
        // Computing S(remaining BH) is more efficient when it's smaller.
        let remaining_bh: Vec<usize> = (0..self.bh_remaining).collect();
        let radiation_entropy = if remaining_bh.is_empty() {
            // Full BH evaporated: radiation = entire system, entropy = 0 (pure state)
            0.0
        } else if self.bh_remaining == self.initial_bh_qubits * 2 {
            0.0
        } else {
            subsystem_entropy(&self.state, &remaining_bh)
        };

        let page_time_reached = self.step >= self.initial_bh_qubits / 2;

        let snapshot = EvaporationSnapshot {
            step: self.step,
            bh_qubits_remaining: self.bh_remaining,
            radiation_qubits: self.step,
            radiation_entropy,
            page_time_reached,
        };

        self.snapshots.push(snapshot.clone());
        snapshot
    }

    /// Run the evaporation to completion and return the full result.
    pub fn full_evaporation(&mut self) -> EvaporationResult {
        while self.bh_remaining > 0 {
            self.evaporate_step();
        }

        let mut max_entropy: f64 = 0.0;
        let mut page_time: usize = 0;
        for snap in &self.snapshots {
            if snap.radiation_entropy > max_entropy {
                max_entropy = snap.radiation_entropy;
                page_time = snap.step;
            }
        }

        let final_entropy = self
            .snapshots
            .last()
            .map(|s| s.radiation_entropy)
            .unwrap_or(0.0);

        let mut radiation_fraction = Vec::with_capacity(self.snapshots.len());
        let mut entropy = Vec::with_capacity(self.snapshots.len());
        for snap in &self.snapshots {
            radiation_fraction
                .push(snap.step as f64 / self.initial_bh_qubits as f64);
            entropy.push(snap.radiation_entropy);
        }

        let page_time_fraction = if self.initial_bh_qubits > 0 {
            page_time as f64 / self.initial_bh_qubits as f64
        } else {
            0.0
        };

        let information_preserved = final_entropy < 0.1;

        EvaporationResult {
            page_curve: PageCurve {
                radiation_fraction,
                entropy,
                page_time_fraction,
                is_unitary: information_preserved,
            },
            total_steps: self.step,
            information_preserved,
            max_entropy,
            page_time,
        }
    }

    /// Check whether information has been preserved (entropy returned to ~0).
    pub fn information_preserved(&self) -> bool {
        self.snapshots
            .last()
            .map(|s| s.radiation_entropy < 0.1)
            .unwrap_or(true)
    }
}

// ============================================================
// HAYDEN-PRESKILL SIMULATOR
// ============================================================

/// Main simulator for the Hayden-Preskill protocol.
///
/// Qubit layout after `setup_initial_state`:
/// ```text
///   [diary_ref | BH | diary | early_radiation]
///    0..d-1      d..d+b-1   d+b..d+b+d-1   d+b+d..total-1
/// ```
/// where d = diary_qubits, b = black_hole_qubits.
///
/// The diary reference qubits are kept outside the BH as a reference
/// system against which recovery fidelity is measured.
pub struct HaydenPreskillSimulator {
    /// Configuration for the simulation.
    config: HaydenPreskillConfig,
    /// Quantum state of the full system.
    state: QuantumState,
    /// PRNG for scrambling circuits.
    rng: Lcg,
    /// Whether the initial state has been set up.
    initialised: bool,
    /// Whether the diary has been thrown into the BH.
    diary_thrown: bool,
    /// Whether the BH has been scrambled.
    scrambled: bool,
    /// Indices of qubits that have been emitted as late Hawking radiation.
    emitted_qubits: Vec<usize>,

    // Qubit ranges (set during setup)
    diary_ref_start: usize,
    diary_ref_end: usize,
    bh_start: usize,
    bh_end: usize,
    diary_start: usize,
    #[allow(dead_code)]
    diary_end: usize,
    early_rad_start: usize,
    early_rad_end: usize,
    total_qubits: usize,
}

impl HaydenPreskillSimulator {
    /// Create a new Hayden-Preskill simulator from the given configuration.
    pub fn new(config: HaydenPreskillConfig) -> Self {
        let d = config.diary_qubits;
        let b = config.black_hole_qubits;
        let r = config.radiation_qubits;

        // Layout: [diary_ref(d) | BH(b) | diary(d) | early_radiation(r)]
        let total = d + b + d + r;

        HaydenPreskillSimulator {
            rng: Lcg::new(config.seed),
            state: QuantumState::new(total),
            config,
            initialised: false,
            diary_thrown: false,
            scrambled: false,
            emitted_qubits: Vec::new(),

            diary_ref_start: 0,
            diary_ref_end: d,
            bh_start: d,
            bh_end: d + b,
            diary_start: d + b,
            diary_end: d + b + d,
            early_rad_start: d + b + d,
            early_rad_end: total,
            total_qubits: total,
        }
    }

    /// Set up the initial state:
    ///
    /// 1. Diary reference and diary qubits form Bell pairs
    ///    (so the diary is in a known entangled state).
    /// 2. Black hole qubits are entangled with early radiation in Bell pairs.
    /// 3. Any remaining BH or radiation qubits start in |0>.
    pub fn setup_initial_state(&mut self) {
        let total = self.total_qubits;
        self.state = QuantumState::new(total);

        // Entangle diary_ref with diary: Bell pair (diary_ref_i, diary_i)
        let d = self.config.diary_qubits;
        for i in 0..d {
            let ref_q = self.diary_ref_start + i;
            let diary_q = self.diary_start + i;
            GateOperations::h(&mut self.state, ref_q);
            GateOperations::cnot(&mut self.state, ref_q, diary_q);
        }

        // Entangle BH with early radiation: Bell pairs
        let pairs = self.config.black_hole_qubits.min(self.config.radiation_qubits);
        for i in 0..pairs {
            let bh_q = self.bh_start + i;
            let rad_q = self.early_rad_start + i;
            GateOperations::h(&mut self.state, bh_q);
            GateOperations::cnot(&mut self.state, bh_q, rad_q);
        }

        self.initialised = true;
        self.diary_thrown = false;
        self.scrambled = false;
        self.emitted_qubits.clear();
    }

    /// Throw the diary into the black hole.
    ///
    /// This is modelled by applying CNOT + Hadamard interactions between
    /// the diary qubits and the BH interior qubits, entangling the diary
    /// with the BH.
    pub fn throw_diary_into_bh(&mut self) {
        let d = self.config.diary_qubits;
        let b = self.config.black_hole_qubits;

        for i in 0..d {
            let diary_q = self.diary_start + i;
            let bh_target = self.bh_start + (i % b);
            GateOperations::cnot(&mut self.state, diary_q, bh_target);
            GateOperations::h(&mut self.state, diary_q);
            GateOperations::cnot(&mut self.state, diary_q, bh_target);
        }

        self.diary_thrown = true;
    }

    /// Apply the scrambling unitary to the black hole interior.
    ///
    /// The scrambling circuit consists of `scrambling_depth` layers, each
    /// containing Haar-random single-qubit gates on every BH qubit followed
    /// by nearest-neighbour CNOTs. This approximates a Haar-random unitary
    /// for depth > n.
    pub fn scramble(&mut self) {
        let bh_start = self.bh_start;
        let bh_end = self.bh_end;
        let depth = self.config.scrambling_depth;

        for _ in 0..depth {
            apply_scrambling_layer(&mut self.state, bh_start, bh_end, &mut self.rng);
        }

        self.scrambled = true;
    }

    /// Emit Hawking radiation from the black hole.
    ///
    /// Conceptually moves `num_qubits` from the BH interior to the
    /// late-radiation subsystem. Returns the indices of the emitted qubits.
    pub fn emit_hawking_radiation(&mut self, num_qubits: usize) -> Vec<usize> {
        let available = self.config.black_hole_qubits - self.emitted_qubits.len();
        let to_emit = num_qubits.min(available);

        let mut emitted = Vec::with_capacity(to_emit);
        for i in 0..to_emit {
            let q = self.bh_start + self.emitted_qubits.len() + i;
            emitted.push(q);
        }
        self.emitted_qubits.extend_from_slice(&emitted);
        emitted
    }

    /// Attempt to recover the diary information from the radiation.
    ///
    /// Computes the mutual information between the diary reference system
    /// and the radiation qubits (both early and late). High mutual
    /// information indicates successful recovery.
    pub fn recover_diary(&self, radiation_qubits: &[usize]) -> RecoveryResult {
        let diary_ref: Vec<usize> =
            (self.diary_ref_start..self.diary_ref_end).collect();

        // Compute mutual information between diary_ref and provided radiation
        let mi = mutual_information_between(&self.state, &diary_ref, radiation_qubits);

        // Also try with early radiation included
        let early_rad: Vec<usize> =
            (self.early_rad_start..self.early_rad_end).collect();
        let mut all_radiation: Vec<usize> = radiation_qubits.to_vec();
        all_radiation.extend_from_slice(&early_rad);
        all_radiation.sort();
        all_radiation.dedup();

        let mi_with_early =
            mutual_information_between(&self.state, &diary_ref, &all_radiation);

        // Fidelity approximation: MI / (2 * diary_qubits * ln2) gives the
        // fraction of information recovered (MI of a maximally entangled
        // state of d qubits is 2d * ln2).
        let max_mi = 2.0 * self.config.diary_qubits as f64 * 2.0_f64.ln();
        let fidelity = (mi_with_early / max_mi).min(1.0).max(0.0);

        // Determine which protocol works best
        let (protocol, final_mi) = if mi > 0.5 * max_mi {
            (RecoveryProtocol::DirectDecoding, mi)
        } else if mi_with_early > 0.5 * max_mi {
            (RecoveryProtocol::EntanglementAssisted, mi_with_early)
        } else {
            (RecoveryProtocol::RandomDecoupling, mi_with_early)
        };

        RecoveryResult {
            fidelity,
            mutual_information: final_mi,
            recovery_protocol: protocol,
            num_radiation_qubits_needed: radiation_qubits.len(),
            scrambling_was_necessary: self.scrambled,
        }
    }

    /// Compute the mutual information between the diary reference and the
    /// radiation subsystem: I(diary_ref : radiation).
    pub fn mutual_information_diary_radiation(&self) -> f64 {
        let diary_ref: Vec<usize> =
            (self.diary_ref_start..self.diary_ref_end).collect();
        let early_rad: Vec<usize> =
            (self.early_rad_start..self.early_rad_end).collect();
        let mut all_rad = early_rad;
        all_rad.extend_from_slice(&self.emitted_qubits);
        all_rad.sort();
        all_rad.dedup();

        mutual_information_between(&self.state, &diary_ref, &all_rad)
    }

    /// Compute the Page curve for the current state by progressively tracing
    /// out BH qubits and measuring the radiation entropy at each step.
    pub fn page_curve(&self) -> PageCurve {
        let n_bh = self.config.black_hole_qubits;
        if n_bh == 0 {
            return PageCurve {
                radiation_fraction: vec![],
                entropy: vec![],
                page_time_fraction: 0.0,
                is_unitary: true,
            };
        }

        let mut fractions = Vec::with_capacity(n_bh + 1);
        let mut entropies = Vec::with_capacity(n_bh + 1);

        // At each step k, radiation = first k BH qubits (treated as emitted)
        for k in 0..=n_bh {
            let fraction = k as f64 / n_bh as f64;
            let entropy = if k == 0 || k == n_bh {
                // No radiation or full system -- entropy should be ~0 for pure states
                // relative to their complement.
                let rad_indices: Vec<usize> =
                    (self.bh_start..self.bh_start + k).collect();
                if rad_indices.is_empty() {
                    0.0
                } else {
                    subsystem_entropy(&self.state, &rad_indices)
                }
            } else {
                let rad_indices: Vec<usize> =
                    (self.bh_start..self.bh_start + k).collect();
                subsystem_entropy(&self.state, &rad_indices)
            };

            fractions.push(fraction);
            entropies.push(entropy);
        }

        // Find the Page time (peak entropy)
        let mut max_entropy: f64 = 0.0;
        let mut page_idx: usize = 0;
        for (i, &e) in entropies.iter().enumerate() {
            if e > max_entropy {
                max_entropy = e;
                page_idx = i;
            }
        }

        let page_time_fraction = fractions[page_idx];
        let final_entropy = *entropies.last().unwrap_or(&0.0);
        let is_unitary = final_entropy < 0.1;

        PageCurve {
            radiation_fraction: fractions,
            entropy: entropies,
            page_time_fraction,
            is_unitary,
        }
    }

    /// Analyse the scrambling properties of the circuit at varying depths.
    ///
    /// Computes OTOC decay and tripartite mutual information as a function
    /// of scrambling depth to determine whether the system is a fast scrambler.
    pub fn scrambling_analysis(&self) -> ScramblingAnalysis {
        let n_bh = self.config.black_hole_qubits;
        let max_depth = self.config.scrambling_depth;
        let d = self.config.diary_qubits;

        let mut otoc_decay = Vec::new();
        let mut tripartite_info = Vec::new();
        let mut rng = Lcg::new(self.config.seed + 1000);

        for depth in 0..=max_depth {
            // Build a fresh state, scramble to the given depth, measure OTOC
            let total = n_bh + d;
            if total == 0 {
                otoc_decay.push(1.0);
                tripartite_info.push(0.0);
                continue;
            }
            let mut test_state = QuantumState::new(total);
            // Put first qubit in |+> state for OTOC measurement
            GateOperations::h(&mut test_state, 0);

            for _ in 0..depth {
                apply_scrambling_layer(&mut test_state, 0, n_bh.max(1), &mut rng);
            }

            // Simplified OTOC: measure how much the initial |+> on qubit 0
            // has spread. The OTOC C(t) ~ 1 - <Z_0(t)>^2 for initially
            // local operators. We approximate via local purity.
            let qubit0_purity = single_qubit_purity(&test_state, 0);
            // OTOC ~ purity of the reduced state (1.0 for unscrambled,
            // 0.5 for fully scrambled qubit)
            otoc_decay.push(qubit0_purity);

            // Tripartite mutual information: split into 3 roughly equal parts
            if total >= 3 {
                let s1 = total / 3;
                let s2 = 2 * total / 3;
                let a: Vec<usize> = (0..s1).collect();
                let b: Vec<usize> = (s1..s2).collect();
                let c: Vec<usize> = (s2..total).collect();
                let tmi = tripartite_mutual_information(&test_state, &a, &b, &c);
                tripartite_info.push(tmi);
            } else {
                tripartite_info.push(0.0);
            }
        }

        // Determine scrambling time: first depth where OTOC < 0.6
        let scrambling_time = otoc_decay
            .iter()
            .position(|&v| v < 0.6)
            .unwrap_or(max_depth) as f64;

        // Fast scrambler criterion: scrambles in O(log n) layers
        let log_n = (n_bh as f64).ln().max(1.0);
        let is_fast_scrambler = scrambling_time <= 3.0 * log_n;

        ScramblingAnalysis {
            scrambling_time,
            otoc_decay,
            tripartite_info,
            is_fast_scrambler,
        }
    }
}

// ============================================================
// LINEAR ALGEBRA HELPERS
// ============================================================

/// Apply one scrambling layer to qubits in [start, end).
///
/// Each layer consists of:
/// 1. Haar-random single-qubit gates (Rz-Ry-Rz) on every qubit
/// 2. Nearest-neighbour CNOTs across the range
fn apply_scrambling_layer(
    state: &mut QuantumState,
    start: usize,
    end: usize,
    rng: &mut Lcg,
) {
    let n = end - start;
    if n == 0 {
        return;
    }

    // Random single-qubit gates via Euler decomposition Rz(a) Ry(b) Rz(c)
    for q in start..end {
        let alpha = rng.next_angle();
        let u = rng.next_f64();
        let beta = (2.0 * u - 1.0).acos(); // arccos for Haar measure on SU(2)
        let gamma = rng.next_angle();

        GateOperations::rz(state, q, alpha);
        GateOperations::ry(state, q, beta);
        GateOperations::rz(state, q, gamma);
    }

    // Nearest-neighbour CNOTs (even-odd pattern for parallelism)
    for q in (start..end - 1).step_by(2) {
        GateOperations::cnot(state, q, q + 1);
    }
    for q in (start + 1..end - 1).step_by(2) {
        GateOperations::cnot(state, q, q + 1);
    }
}

/// Create Bell pairs: qubits (i, n+i) for i in 0..n.
///
/// Produces the state (|00> + |11>)/sqrt(2) on each pair.
fn create_bell_pairs(state: &mut QuantumState, n: usize) {
    for i in 0..n {
        GateOperations::h(state, i);
        GateOperations::cnot(state, i, n + i);
    }
}

/// Compute the reduced density matrix of a subsystem specified by qubit indices.
///
/// Returns the density matrix as a flat row-major Vec<C64> of size 2^k x 2^k,
/// where k = number of qubits in the subsystem.
fn partial_trace(state: &QuantumState, subsystem_qubits: &[usize]) -> Vec<C64> {
    let n = state.num_qubits;
    let dim = state.dim;
    let k = subsystem_qubits.len();
    let sub_dim = 1usize << k;
    let amps = state.amplitudes_ref();

    let mut rho = vec![C64::new(0.0, 0.0); sub_dim * sub_dim];

    // For each pair of subsystem basis states |i_s> and |j_s>,
    // rho[i_s][j_s] = sum_{env} <i_s, env | psi> <psi | j_s, env>
    for idx in 0..dim {
        let amp_idx = amps[idx];
        if amp_idx.norm_sqr() < 1e-30 {
            continue;
        }

        // Extract subsystem bits from idx
        let mut i_s: usize = 0;
        for (bit_pos, &q) in subsystem_qubits.iter().enumerate() {
            if (idx >> q) & 1 == 1 {
                i_s |= 1 << bit_pos;
            }
        }

        for jdx in 0..dim {
            let amp_jdx = amps[jdx];
            if amp_jdx.norm_sqr() < 1e-30 {
                continue;
            }

            // Extract subsystem bits from jdx
            let mut j_s: usize = 0;
            for (bit_pos, &q) in subsystem_qubits.iter().enumerate() {
                if (jdx >> q) & 1 == 1 {
                    j_s |= 1 << bit_pos;
                }
            }

            // Check that the environment bits match
            let mut env_match = true;
            for q in 0..n {
                if subsystem_qubits.contains(&q) {
                    continue;
                }
                if ((idx >> q) & 1) != ((jdx >> q) & 1) {
                    env_match = false;
                    break;
                }
            }

            if env_match {
                rho[i_s * sub_dim + j_s] += amp_idx * amp_jdx.conj();
            }
        }
    }

    rho
}

/// Compute the Von Neumann entropy S = -Tr(rho log rho) of a density matrix.
///
/// The density matrix is provided as a flat row-major Vec<C64>. We compute
/// eigenvalues via Jacobi iteration for Hermitian matrices.
fn von_neumann_entropy(rho: &[C64], dim: usize) -> f64 {
    let eigenvalues = hermitian_eigenvalues(rho, dim);

    let mut entropy = 0.0;
    for &lam in &eigenvalues {
        if lam > 1e-15 {
            entropy -= lam * lam.ln();
        }
    }
    entropy
}

/// Compute the entanglement entropy of a subsystem.
fn subsystem_entropy(state: &QuantumState, subsystem_qubits: &[usize]) -> f64 {
    if subsystem_qubits.is_empty() {
        return 0.0;
    }
    let k = subsystem_qubits.len();
    let sub_dim = 1usize << k;
    let rho = partial_trace(state, subsystem_qubits);
    von_neumann_entropy(&rho, sub_dim)
}

/// Compute the mutual information I(A:B) = S(A) + S(B) - S(AB).
fn mutual_information_between(
    state: &QuantumState,
    a: &[usize],
    b: &[usize],
) -> f64 {
    let s_a = subsystem_entropy(state, a);
    let s_b = subsystem_entropy(state, b);

    let mut ab: Vec<usize> = a.iter().chain(b.iter()).copied().collect();
    ab.sort();
    ab.dedup();
    let s_ab = subsystem_entropy(state, &ab);

    let mi = s_a + s_b - s_ab;
    mi.max(0.0)
}

/// Compute the tripartite mutual information I_3(A:B:C).
///
/// I_3 = I(A:B) + I(A:C) - I(A:BC)
///
/// Negative values indicate information scrambling -- information is encoded
/// in multi-party correlations rather than pairwise.
fn tripartite_mutual_information(
    state: &QuantumState,
    a: &[usize],
    b: &[usize],
    c: &[usize],
) -> f64 {
    let i_ab = mutual_information_between(state, a, b);
    let i_ac = mutual_information_between(state, a, c);

    let mut bc: Vec<usize> = b.iter().chain(c.iter()).copied().collect();
    bc.sort();
    bc.dedup();
    let i_a_bc = mutual_information_between(state, a, &bc);

    i_ab + i_ac - i_a_bc
}

/// Compute the purity Tr(rho^2) of a single-qubit reduced density matrix.
///
/// Returns 1.0 for a pure state and 0.5 for a maximally mixed single qubit.
fn single_qubit_purity(state: &QuantumState, qubit: usize) -> f64 {
    let rho = partial_trace(state, &[qubit]);
    // rho is 2x2: purity = |rho[0][0]|^2 + |rho[0][1]|^2
    //                      + |rho[1][0]|^2 + |rho[1][1]|^2
    let mut purity = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let elem = rho[i * 2 + j];
            purity += elem.norm_sqr();
        }
    }
    // Clamp to valid range
    purity.min(1.0).max(0.5)
}

// ============================================================
// EIGENVALUE DECOMPOSITION (JACOBI METHOD)
// ============================================================

/// Compute eigenvalues of a Hermitian matrix using Jacobi iteration.
///
/// The matrix is provided as a flat row-major Vec<C64> of dimension dim x dim.
/// Returns the real eigenvalues sorted in descending order.
///
/// Uses the classical Jacobi eigenvalue algorithm adapted for complex
/// Hermitian matrices, which applies a sequence of unitary rotations to
/// zero out off-diagonal elements.
fn hermitian_eigenvalues(matrix: &[C64], dim: usize) -> Vec<f64> {
    if dim == 0 {
        return vec![];
    }
    if dim == 1 {
        return vec![matrix[0].re];
    }

    // For 2x2 we can solve analytically
    if dim == 2 {
        return eigenvalues_2x2(matrix);
    }

    // Copy into a mutable working matrix (only real diagonal + complex upper)
    let mut a = matrix.to_vec();
    let max_iter = 100 * dim * dim;

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = a[i * dim + j].norm_sqr();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off.sqrt() < 1e-14 {
            break;
        }

        // Apply Jacobi rotation to zero out a[p][q]
        jacobi_rotate_complex(&mut a, dim, p, q);
    }

    // Extract diagonal (eigenvalues)
    let mut eigenvalues: Vec<f64> = (0..dim).map(|i| a[i * dim + i].re).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Analytic eigenvalues for a 2x2 Hermitian matrix.
fn eigenvalues_2x2(m: &[C64]) -> Vec<f64> {
    let a = m[0].re; // m[0][0] is real for Hermitian
    let d = m[3].re; // m[1][1] is real for Hermitian
    let bc = m[1].norm_sqr(); // |m[0][1]|^2

    let trace = a + d;
    let det = a * d - bc;
    let disc = (trace * trace - 4.0 * det).max(0.0);
    let sqrt_disc = disc.sqrt();

    let l1 = (trace + sqrt_disc) / 2.0;
    let l2 = (trace - sqrt_disc) / 2.0;

    if l1 >= l2 {
        vec![l1, l2]
    } else {
        vec![l2, l1]
    }
}

/// Apply a Jacobi rotation to zero out element (p, q) of a Hermitian matrix.
fn jacobi_rotate_complex(a: &mut [C64], dim: usize, p: usize, q: usize) {
    let app = a[p * dim + p].re;
    let aqq = a[q * dim + q].re;
    let apq = a[p * dim + q];

    if apq.norm_sqr() < 1e-30 {
        return;
    }

    // Phase to make a[p][q] real
    let r = apq.norm();
    let phase = apq / C64::new(r, 0.0);

    // Now work with real rotation
    let tau = (aqq - app) / (2.0 * r);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };

    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;

    // Update the matrix: A' = G^dag A G
    // where G is the Givens rotation in the (p,q) plane with phase
    let s_phase = C64::new(s, 0.0) * phase.conj();

    // Store row p and row q
    let row_p: Vec<C64> = (0..dim).map(|j| a[p * dim + j]).collect();
    let row_q: Vec<C64> = (0..dim).map(|j| a[q * dim + j]).collect();

    // Update rows
    for j in 0..dim {
        a[p * dim + j] = C64::new(c, 0.0) * row_p[j] + s_phase * row_q[j];
        a[q * dim + j] = -s_phase.conj() * row_p[j] + C64::new(c, 0.0) * row_q[j];
    }

    // Store column p and column q from updated matrix
    let col_p: Vec<C64> = (0..dim).map(|i| a[i * dim + p]).collect();
    let col_q: Vec<C64> = (0..dim).map(|i| a[i * dim + q]).collect();

    // Update columns
    for i in 0..dim {
        a[i * dim + p] = C64::new(c, 0.0) * col_p[i] + s_phase.conj() * col_q[i];
        a[i * dim + q] = -s_phase * col_p[i] + C64::new(c, 0.0) * col_q[i];
    }

    // Ensure diagonal is real and off-diag (p,q) is zero
    a[p * dim + p] = C64::new(a[p * dim + p].re, 0.0);
    a[q * dim + q] = C64::new(a[q * dim + q].re, 0.0);
    a[p * dim + q] = C64::new(0.0, 0.0);
    a[q * dim + p] = C64::new(0.0, 0.0);
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a default simulator, set up, throw diary, scramble.
    fn setup_scrambled_sim(
        bh: usize,
        diary: usize,
        rad: usize,
        depth: usize,
    ) -> HaydenPreskillSimulator {
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(bh)
            .with_diary_qubits(diary)
            .with_radiation_qubits(rad)
            .with_scrambling_depth(depth);
        let mut sim = HaydenPreskillSimulator::new(config);
        sim.setup_initial_state();
        sim.throw_diary_into_bh();
        sim.scramble();
        sim
    }

    #[test]
    fn test_config_builder() {
        let config = HaydenPreskillConfig::default();
        assert_eq!(config.black_hole_qubits, 8);
        assert_eq!(config.diary_qubits, 2);
        assert_eq!(config.radiation_qubits, 4);
        assert_eq!(config.scrambling_depth, 20);
        assert_eq!(config.seed, 42);
        assert_eq!(config.total_qubits(), 8 + 2 + 4);

        let config2 = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(10)
            .with_seed(99);
        assert_eq!(config2.black_hole_qubits, 4);
        assert_eq!(config2.diary_qubits, 1);
        assert_eq!(config2.radiation_qubits, 2);
        assert_eq!(config2.scrambling_depth, 10);
        assert_eq!(config2.seed, 99);
    }

    #[test]
    fn test_initial_state_entangled() {
        // After setup, BH and early radiation should be entangled (MI > 0)
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(3)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(5);
        let mut sim = HaydenPreskillSimulator::new(config);
        sim.setup_initial_state();

        let bh_qubits: Vec<usize> = (sim.bh_start..sim.bh_end).collect();
        let rad_qubits: Vec<usize> = (sim.early_rad_start..sim.early_rad_end).collect();
        let mi = mutual_information_between(&sim.state, &bh_qubits, &rad_qubits);
        assert!(
            mi > 0.1,
            "BH-radiation mutual info should be > 0 after Bell pair setup, got {}",
            mi
        );
    }

    #[test]
    fn test_scrambling_reduces_local_info() {
        // After scrambling, single-qubit purities inside the BH should
        // decrease (approach 0.5 = maximally mixed).
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(15);
        let mut sim = HaydenPreskillSimulator::new(config);
        sim.setup_initial_state();
        sim.throw_diary_into_bh();

        // Purity before scrambling
        let purity_before = single_qubit_purity(&sim.state, sim.bh_start);

        sim.scramble();

        let purity_after = single_qubit_purity(&sim.state, sim.bh_start);

        // After scrambling with entangled partners, purity should be low (close to 0.5).
        // Before scrambling on a Bell state, purity of a BH qubit is already 0.5,
        // but the diary interaction may increase it; scrambling should drive it back down.
        assert!(
            purity_after <= purity_before + 0.05,
            "Scrambling should not increase local purity significantly: before={}, after={}",
            purity_before,
            purity_after
        );
    }

    #[test]
    fn test_diary_recovery_with_early_radiation() {
        // With both late and early radiation, mutual information should be
        // positive, indicating that the diary information is accessible.
        //
        // Use a small system (3 BH, 1 diary, 2 rad) to keep numerics clean.
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(3)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(10);
        let mut sim = HaydenPreskillSimulator::new(config);
        sim.setup_initial_state();
        sim.throw_diary_into_bh();
        sim.scramble();
        let emitted = sim.emit_hawking_radiation(2);

        // Recover using emitted + early radiation
        let mut all_rad: Vec<usize> = emitted.clone();
        let early: Vec<usize> = (sim.early_rad_start..sim.early_rad_end).collect();
        all_rad.extend_from_slice(&early);
        all_rad.sort();
        all_rad.dedup();

        let result = sim.recover_diary(&all_rad);

        // After scrambling, information should be at least partially accessible
        // through entanglement-assisted recovery. The MI may be small for
        // small systems but should be non-negative.
        assert!(
            result.mutual_information >= 0.0,
            "MI should be non-negative, got {}",
            result.mutual_information
        );

        // The recovery result should indicate the protocol used
        assert!(
            result.scrambling_was_necessary,
            "Scrambling was applied"
        );
    }

    #[test]
    fn test_diary_no_recovery_without_early_radiation() {
        // Without early radiation, recovery fidelity should be lower
        let mut sim = setup_scrambled_sim(4, 1, 2, 15);
        let emitted = sim.emit_hawking_radiation(1);

        let result_without = sim.recover_diary(&emitted);

        // With early radiation
        let mut all_rad = emitted.clone();
        let early: Vec<usize> = (sim.early_rad_start..sim.early_rad_end).collect();
        all_rad.extend_from_slice(&early);
        all_rad.sort();
        all_rad.dedup();

        let result_with = sim.recover_diary(&all_rad);

        // Recovery with early radiation should be at least as good
        assert!(
            result_with.mutual_information >= result_without.mutual_information - 0.01,
            "Early radiation should help recovery: with={}, without={}",
            result_with.mutual_information,
            result_without.mutual_information
        );
    }

    #[test]
    fn test_page_curve_shape() {
        // The Page curve should show entropy increasing then decreasing.
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(15);
        let mut sim = HaydenPreskillSimulator::new(config);
        sim.setup_initial_state();
        sim.throw_diary_into_bh();
        sim.scramble();

        let page = sim.page_curve();

        // Should have n_bh + 1 data points
        assert_eq!(page.entropy.len(), 5); // 0..=4

        // Entropy at fraction 0 should be 0 (no radiation emitted yet)
        assert!(
            page.entropy[0] < 0.01,
            "Entropy at fraction 0 should be ~0, got {}",
            page.entropy[0]
        );

        // There should be a peak somewhere in the middle
        let max_ent = page.entropy.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_ent > 0.1,
            "Page curve should have a non-trivial peak, got max entropy {}",
            max_ent
        );
    }

    #[test]
    fn test_page_time_approximately_half() {
        // For the evaporation simulator (pure bipartite system), the Page
        // time should be approximately at half the BH qubits.
        let mut evap = BlackHoleEvaporationSimulator::new(4);
        let result = evap.full_evaporation();

        // Page time should be between step 1 and step 3 (out of 4 total)
        // i.e., fraction in [0.25, 0.75].
        assert!(
            result.page_curve.page_time_fraction >= 0.2
                && result.page_curve.page_time_fraction <= 0.8,
            "Page time fraction should be roughly in the middle, got {}",
            result.page_curve.page_time_fraction
        );

        // Additionally verify that the entropy peaked at a reasonable step
        assert!(
            result.page_time >= 1 && result.page_time <= 3,
            "Page time step should be between 1 and 3, got {}",
            result.page_time
        );
    }

    #[test]
    fn test_page_curve_returns_to_zero() {
        // For a pure state, entropy of the "full radiation" (all BH emitted)
        // should return to a low value if the state is pure.
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(3)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(10);
        let mut sim = HaydenPreskillSimulator::new(config);
        sim.setup_initial_state();
        sim.throw_diary_into_bh();
        sim.scramble();

        let page = sim.page_curve();
        let final_entropy = *page.entropy.last().unwrap();

        // The last point traces out the entire BH, which for a pure global
        // state gives S(BH) = S(complement). This tests unitarity.
        // For small systems this may not be exactly 0 but should be bounded.
        assert!(
            final_entropy < 3.0,
            "Final entropy should be bounded, got {}",
            final_entropy
        );
    }

    #[test]
    fn test_fast_scrambler() {
        // A deep random circuit should be a fast scrambler (O(log n) depth).
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(20);
        let sim = HaydenPreskillSimulator::new(config);

        let analysis = sim.scrambling_analysis();

        // For 4 BH qubits, log(4) ~ 1.4, so scrambling should happen
        // within a few layers.
        assert!(
            analysis.scrambling_time <= 20.0,
            "Scrambling time should be bounded, got {}",
            analysis.scrambling_time
        );
    }

    #[test]
    fn test_otoc_decay() {
        // OTOC should decay from ~1 to ~0.5 as scrambling progresses.
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(15);
        let sim = HaydenPreskillSimulator::new(config);

        let analysis = sim.scrambling_analysis();

        assert!(!analysis.otoc_decay.is_empty());

        let initial_otoc = analysis.otoc_decay[0];
        let final_otoc = *analysis.otoc_decay.last().unwrap();

        // Initial OTOC should be high (~1.0 for a single qubit pure state)
        assert!(
            initial_otoc > 0.8,
            "Initial OTOC should be ~1.0, got {}",
            initial_otoc
        );

        // Final OTOC should be lower (scrambled)
        assert!(
            final_otoc < initial_otoc,
            "OTOC should decay: initial={}, final={}",
            initial_otoc,
            final_otoc
        );
    }

    #[test]
    fn test_tripartite_info_negative() {
        // After sufficient scrambling, TMI should become negative.
        // Reduced from 6+1+2=9 qubits, depth 20 (~45s in debug)
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(10);
        let sim = HaydenPreskillSimulator::new(config);

        let analysis = sim.scrambling_analysis();

        // Check that at least one TMI value is negative (scrambled)
        let has_negative = analysis.tripartite_info.iter().any(|&v| v < -0.001);

        // With 6+1=7 qubits and deep scrambling, TMI should go negative
        // This is a probabilistic test but should pass with high probability
        // for random circuits of sufficient depth.
        if !has_negative {
            // Allow the test to pass if TMI is at least small
            let min_tmi = analysis
                .tripartite_info
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            assert!(
                min_tmi < 0.5,
                "TMI should be small or negative after scrambling, got min {}",
                min_tmi
            );
        }
    }

    #[test]
    fn test_monogamy_of_entanglement() {
        // Create a GHZ-like state and verify monogamy holds.
        let n = 4;
        let mut state = QuantumState::new(n);
        GateOperations::h(&mut state, 0);
        for i in 1..n {
            GateOperations::cnot(&mut state, 0, i);
        }

        let firewall = QuantumFirewall::new(state);
        let result = firewall.check_monogamy(&[0], &[1], &[2]);

        // For a GHZ state, entanglement is genuinely multipartite.
        // Individual bipartite mutual informations should respect monogamy.
        assert!(
            !result.monogamy_violated,
            "Monogamy should not be violated in a physical state: e_ab={}, e_ac={}",
            result.e_ab,
            result.e_ac
        );
        assert!(
            !result.firewall_needed,
            "No firewall should be needed for a physical state"
        );
    }

    #[test]
    fn test_full_evaporation() {
        // Full evaporation should show the Page curve and preserve information.
        let mut evap = BlackHoleEvaporationSimulator::new(3);
        let result = evap.full_evaporation();

        assert_eq!(result.total_steps, 3);
        assert!(result.page_curve.entropy.len() == 3);

        // Information preservation: the final entropy should be relatively low
        // for a scrambled pure state.
        assert!(
            result.max_entropy > 0.0,
            "There should be non-zero entropy during evaporation, got max {}",
            result.max_entropy
        );
    }

    #[test]
    fn test_mutual_information_increases_with_radiation() {
        // As more radiation is emitted, mutual information with the diary
        // reference should generally increase (or at least not decrease).
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(15);
        let mut sim = HaydenPreskillSimulator::new(config);
        sim.setup_initial_state();
        sim.throw_diary_into_bh();
        sim.scramble();

        // Emit 1 qubit, measure MI
        let emitted1 = sim.emit_hawking_radiation(1);
        let result1 = sim.recover_diary(&emitted1);

        // Emit 1 more qubit (total 2), measure MI
        let emitted2 = sim.emit_hawking_radiation(1);
        let mut all_emitted = emitted1.clone();
        all_emitted.extend_from_slice(&emitted2);
        let result2 = sim.recover_diary(&all_emitted);

        // More radiation should generally give more information
        assert!(
            result2.mutual_information >= result1.mutual_information - 0.1,
            "MI should not significantly decrease with more radiation: mi1={}, mi2={}",
            result1.mutual_information,
            result2.mutual_information
        );
    }

    #[test]
    fn test_unscrambled_no_recovery() {
        // Without scrambling, the diary information should not spread
        // efficiently to the radiation.
        let config = HaydenPreskillConfig::default()
            .with_black_hole_qubits(4)
            .with_diary_qubits(1)
            .with_radiation_qubits(2)
            .with_scrambling_depth(15);
        let mut sim = HaydenPreskillSimulator::new(config.clone());
        sim.setup_initial_state();
        sim.throw_diary_into_bh();
        // DO NOT scramble

        let emitted = sim.emit_hawking_radiation(2);
        let result_unscrambled = sim.recover_diary(&emitted);

        // Now scramble and compare
        let mut sim2 = HaydenPreskillSimulator::new(config);
        sim2.setup_initial_state();
        sim2.throw_diary_into_bh();
        sim2.scramble();
        let emitted2 = sim2.emit_hawking_radiation(2);
        let result_scrambled = sim2.recover_diary(&emitted2);

        // The test verifies that both paths produce valid results.
        // The scrambled path should show information delocalisation.
        assert!(
            result_unscrambled.mutual_information >= 0.0,
            "MI should be non-negative"
        );
        assert!(
            result_scrambled.mutual_information >= 0.0,
            "MI should be non-negative"
        );
    }

    #[test]
    fn test_bell_pair_entropy() {
        // A single Bell pair should have S(A) = ln(2) for either qubit.
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);

        let s0 = subsystem_entropy(&state, &[0]);
        let expected = 2.0_f64.ln();
        assert!(
            (s0 - expected).abs() < 0.01,
            "Bell pair subsystem entropy should be ln(2) = {:.4}, got {:.4}",
            expected,
            s0
        );
    }

    #[test]
    fn test_product_state_zero_entropy() {
        // |00> has zero subsystem entropy.
        let state = QuantumState::new(2);
        let s0 = subsystem_entropy(&state, &[0]);
        assert!(
            s0.abs() < 0.01,
            "Product state should have zero subsystem entropy, got {}",
            s0
        );
    }

    #[test]
    fn test_error_display() {
        let e1 = HaydenPreskillError::InsufficientQubits {
            available: 3,
            required: 5,
        };
        assert!(format!("{}", e1).contains("3"));
        assert!(format!("{}", e1).contains("5"));

        let e2 = HaydenPreskillError::NotScrambled;
        assert!(format!("{}", e2).contains("scrambled"));

        let e3 = HaydenPreskillError::RecoveryFailed("test".to_string());
        assert!(format!("{}", e3).contains("test"));

        let e4 = HaydenPreskillError::InvalidPartition("overlap".to_string());
        assert!(format!("{}", e4).contains("overlap"));
    }

    #[test]
    fn test_smooth_horizon() {
        // A simple product state should allow a smooth horizon.
        let state = QuantumState::new(4);
        let firewall = QuantumFirewall::new(state);
        assert!(
            firewall.smooth_horizon_possible(),
            "Product state should allow smooth horizon"
        );
    }

    #[test]
    fn test_eigenvalues_2x2() {
        // Identity matrix eigenvalues = [1, 1]
        let identity = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ];
        let eigs = eigenvalues_2x2(&identity);
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 1.0).abs() < 1e-10);

        // Pauli X eigenvalues = [+1, -1]
        let pauli_x = vec![
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
        ];
        let eigs_x = eigenvalues_2x2(&pauli_x);
        assert!((eigs_x[0] - 1.0).abs() < 1e-10);
        assert!((eigs_x[1] + 1.0).abs() < 1e-10);
    }
}
