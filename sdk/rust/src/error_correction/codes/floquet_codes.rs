//! Floquet Code Simulation
//!
//! Dynamical quantum error correcting codes where the stabilizer group changes
//! periodically through a sequence of two-qubit measurements. Unlike static
//! codes (e.g. surface codes), the codespace changes each time step -- only
//! the periodic sequence stabilizes logical information.
//!
//! # Implemented Codes
//!
//! - **Honeycomb Code** (Hastings-Haah): 3-round period with XX, YY, ZZ
//!   measurements on a honeycomb lattice with 3-colorable edges.
//! - **X3Z3 Code**: Bias-tailored Floquet code optimized for Z-biased noise
//!   (e.g. cat qubits), with alternating X-type and Z-type measurement rounds.
//!
//! # Architecture
//!
//! ```text
//! FloquetSchedule ─── MeasurementRound ─── PauliMeasurement
//!        │
//!        ├── HoneycombCode (builds schedule from honeycomb lattice)
//!        ├── X3Z3Code (builds schedule for biased noise)
//!        │
//!        └── FloquetSimulator (executes schedule on state vector)
//!               │
//!               ├── FloquetDecoder (greedy matching on syndrome graph)
//!               ├── FloquetStats (runtime statistics)
//!               └── FloquetBenchmark (Monte Carlo error rate estimation)
//! ```

use num_complex::Complex64 as C64;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

// ============================================================
// PAULI OPERATOR
// ============================================================

/// Single-qubit Pauli operator for Floquet measurement specifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauliOp {
    /// Pauli-X operator
    X,
    /// Pauli-Y operator
    Y,
    /// Pauli-Z operator
    Z,
}

impl PauliOp {
    /// Multiply two Pauli operators (ignoring global phase).
    ///
    /// Returns `None` if the product is the identity (up to phase),
    /// or `Some(P)` if the product is Pauli P (up to phase).
    pub fn multiply(self, other: PauliOp) -> Option<PauliOp> {
        match (self, other) {
            (PauliOp::X, PauliOp::X) => None,
            (PauliOp::Y, PauliOp::Y) => None,
            (PauliOp::Z, PauliOp::Z) => None,
            (PauliOp::X, PauliOp::Y) | (PauliOp::Y, PauliOp::X) => Some(PauliOp::Z),
            (PauliOp::Y, PauliOp::Z) | (PauliOp::Z, PauliOp::Y) => Some(PauliOp::X),
            (PauliOp::X, PauliOp::Z) | (PauliOp::Z, PauliOp::X) => Some(PauliOp::Y),
        }
    }

    /// Check if two Pauli operators commute.
    pub fn commutes_with(self, other: PauliOp) -> bool {
        self == other
    }

    /// Return the 2x2 matrix representation.
    pub fn matrix(&self) -> [[C64; 2]; 2] {
        let zero = C64::new(0.0, 0.0);
        let one = C64::new(1.0, 0.0);
        let neg_one = C64::new(-1.0, 0.0);
        let i = C64::new(0.0, 1.0);
        let neg_i = C64::new(0.0, -1.0);

        match self {
            PauliOp::X => [[zero, one], [one, zero]],
            PauliOp::Y => [[zero, neg_i], [i, zero]],
            PauliOp::Z => [[one, zero], [zero, neg_one]],
        }
    }
}

impl fmt::Display for PauliOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PauliOp::X => write!(f, "X"),
            PauliOp::Y => write!(f, "Y"),
            PauliOp::Z => write!(f, "Z"),
        }
    }
}

// ============================================================
// EDGE COLOR
// ============================================================

/// Edge color for 3-colorable honeycomb lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeColor {
    /// Red edges -- XX measurements in honeycomb code
    Red,
    /// Green edges -- YY measurements in honeycomb code
    Green,
    /// Blue edges -- ZZ measurements in honeycomb code
    Blue,
}

impl fmt::Display for EdgeColor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EdgeColor::Red => write!(f, "Red"),
            EdgeColor::Green => write!(f, "Green"),
            EdgeColor::Blue => write!(f, "Blue"),
        }
    }
}

// ============================================================
// PAULI MEASUREMENT
// ============================================================

/// A two-qubit Pauli measurement specification.
///
/// Represents a projective measurement of the operator P_a tensor P_b
/// on qubits (qubit_a, qubit_b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliMeasurement {
    /// First qubit index
    pub qubit_a: usize,
    /// Second qubit index
    pub qubit_b: usize,
    /// Pauli operator on first qubit
    pub pauli_a: PauliOp,
    /// Pauli operator on second qubit
    pub pauli_b: PauliOp,
}

impl PauliMeasurement {
    /// Create a new two-qubit Pauli measurement.
    pub fn new(qubit_a: usize, qubit_b: usize, pauli_a: PauliOp, pauli_b: PauliOp) -> Self {
        Self {
            qubit_a,
            qubit_b,
            pauli_a,
            pauli_b,
        }
    }
}

impl fmt::Display for PauliMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}{}({},{})",
            self.pauli_a, self.pauli_b, self.qubit_a, self.qubit_b
        )
    }
}

// ============================================================
// MEASUREMENT ROUND
// ============================================================

/// A single round of two-qubit Pauli measurements in a Floquet code.
///
/// Each round consists of a set of commuting two-qubit Pauli measurements
/// that can be performed simultaneously.
#[derive(Debug, Clone)]
pub struct MeasurementRound {
    /// The measurements in this round
    pub measurements: Vec<PauliMeasurement>,
}

impl MeasurementRound {
    /// Create a new measurement round.
    pub fn new(measurements: Vec<PauliMeasurement>) -> Self {
        Self { measurements }
    }

    /// Number of measurements in this round.
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Whether this round has no measurements.
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Get the set of qubits involved in this round.
    pub fn qubits(&self) -> Vec<usize> {
        let mut qs: Vec<usize> = self
            .measurements
            .iter()
            .flat_map(|m| vec![m.qubit_a, m.qubit_b])
            .collect();
        qs.sort_unstable();
        qs.dedup();
        qs
    }
}

// ============================================================
// FLOQUET SCHEDULE
// ============================================================

/// Defines the periodic measurement pattern for a Floquet code.
///
/// A Floquet code is specified by a sequence of measurement rounds that
/// repeats periodically. The codespace is only well-defined after a
/// complete period of measurements.
#[derive(Debug, Clone)]
pub struct FloquetSchedule {
    /// One period of measurement rounds
    pub rounds: Vec<MeasurementRound>,
    /// Total number of physical qubits
    pub num_qubits: usize,
}

impl FloquetSchedule {
    /// Create a new Floquet schedule.
    pub fn new(rounds: Vec<MeasurementRound>, num_qubits: usize) -> Self {
        Self { rounds, num_qubits }
    }

    /// Number of rounds in one period.
    pub fn period(&self) -> usize {
        self.rounds.len()
    }

    /// Get the measurement round at time step t (modulo the period).
    pub fn round(&self, t: usize) -> &MeasurementRound {
        &self.rounds[t % self.rounds.len()]
    }

    /// Total number of measurements per period.
    pub fn measurements_per_period(&self) -> usize {
        self.rounds.iter().map(|r| r.len()).sum()
    }
}

// ============================================================
// FLOQUET CODE TRAIT
// ============================================================

/// Trait for Floquet code implementations.
///
/// Any code that implements this trait can be used with the generic
/// `FloquetSimulator` and `FloquetBenchmark` infrastructure.
pub trait FloquetCode {
    /// Get the measurement schedule for this code.
    fn schedule(&self) -> FloquetSchedule;

    /// Number of physical qubits.
    fn num_physical_qubits(&self) -> usize;

    /// Number of logical qubits.
    fn num_logical_qubits(&self) -> usize;

    /// Code distance.
    fn distance(&self) -> usize;
}

// ============================================================
// HONEYCOMB LATTICE GENERATION
// ============================================================

/// Generate a honeycomb lattice with 3-colorable edges.
///
/// Returns a list of edges `(qubit_a, qubit_b, color)` for a honeycomb
/// lattice that supports a Floquet code of the given distance.
///
/// The honeycomb lattice has vertices at positions determined by the
/// distance parameter. Each vertex has degree 3, with one edge of each
/// color (Red, Green, Blue).
///
/// For a distance-d honeycomb code on a planar patch:
/// - The lattice is built as a brick-wall representation of the honeycomb
/// - Qubits sit on vertices
/// - Edges are 3-colored so that each vertex touches exactly one edge of each color
pub fn honeycomb_lattice(distance: usize) -> Vec<(usize, usize, EdgeColor)> {
    let d = distance.max(2);

    // Build a honeycomb lattice as a brick-wall graph.
    // Rows alternate between having horizontal bonds shifted left or right.
    // We use a grid of size (2*d) rows x (d) columns for the planar patch.
    let rows = 2 * d;
    let cols = d;

    let qubit_index = |r: usize, c: usize| -> usize { r * cols + c };
    let num_qubits = rows * cols;

    let mut edges: Vec<(usize, usize, EdgeColor)> = Vec::new();
    let mut vertex_colors: HashMap<usize, Vec<EdgeColor>> = HashMap::new();

    for v in 0..num_qubits {
        vertex_colors.insert(v, Vec::new());
    }

    // Vertical bonds (alternating Red and Blue)
    for r in 0..(rows - 1) {
        for c in 0..cols {
            let a = qubit_index(r, c);
            let b = qubit_index(r + 1, c);
            let color = if r % 2 == 0 {
                EdgeColor::Red
            } else {
                EdgeColor::Blue
            };
            edges.push((a, b, color));
            vertex_colors.get_mut(&a).unwrap().push(color);
            vertex_colors.get_mut(&b).unwrap().push(color);
        }
    }

    // Horizontal bonds (Green) -- staggered by row parity
    for r in 0..rows {
        let start = if r % 2 == 0 { 0 } else { 1 };
        let mut c = start;
        while c + 1 < cols {
            let a = qubit_index(r, c);
            let b = qubit_index(r, c + 1);
            let color = EdgeColor::Green;
            edges.push((a, b, color));
            vertex_colors.get_mut(&a).unwrap().push(color);
            vertex_colors.get_mut(&b).unwrap().push(color);
            c += 2;
        }
    }

    edges
}

/// Count the number of qubits in a honeycomb lattice for given distance.
fn honeycomb_qubit_count(distance: usize) -> usize {
    let d = distance.max(2);
    2 * d * d
}

// ============================================================
// HONEYCOMB CODE
// ============================================================

/// The Hastings-Haah honeycomb Floquet code.
///
/// A topological Floquet code on a honeycomb lattice with a 3-round
/// measurement period. In each round, one color of edges is measured:
///
/// - Round 0 (Red edges): XX measurements
/// - Round 1 (Green edges): YY measurements
/// - Round 2 (Blue edges): ZZ measurements
///
/// The code encodes 1 logical qubit on a planar patch.
///
/// # Reference
///
/// M. B. Hastings and J. Haah, "Dynamically Generated Logical Qubits,"
/// Quantum 5, 564 (2021).
#[derive(Debug, Clone)]
pub struct HoneycombCode {
    /// Code distance
    dist: usize,
    /// Honeycomb lattice edges with colors
    edges: Vec<(usize, usize, EdgeColor)>,
    /// Number of physical qubits
    n_qubits: usize,
}

impl HoneycombCode {
    /// Create a new honeycomb Floquet code with the given distance.
    ///
    /// The distance must be at least 2.
    pub fn new(distance: usize) -> Self {
        let dist = distance.max(2);
        let edges = honeycomb_lattice(dist);
        let n_qubits = honeycomb_qubit_count(dist);

        HoneycombCode {
            dist,
            edges,
            n_qubits,
        }
    }

    /// Get the code distance.
    pub fn code_distance(&self) -> usize {
        self.dist
    }

    /// Number of physical qubits.
    pub fn num_physical_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of logical qubits (always 1 for planar honeycomb).
    pub fn num_logical_qubits(&self) -> usize {
        1
    }

    /// Get the measurement schedule (period 3).
    ///
    /// Returns a `FloquetSchedule` with three rounds:
    /// - Round 0: XX on Red edges
    /// - Round 1: YY on Green edges
    /// - Round 2: ZZ on Blue edges
    pub fn schedule(&self) -> FloquetSchedule {
        let color_to_pauli = |color: EdgeColor| -> (PauliOp, PauliOp) {
            match color {
                EdgeColor::Red => (PauliOp::X, PauliOp::X),
                EdgeColor::Green => (PauliOp::Y, PauliOp::Y),
                EdgeColor::Blue => (PauliOp::Z, PauliOp::Z),
            }
        };

        let colors = [EdgeColor::Red, EdgeColor::Green, EdgeColor::Blue];
        let mut rounds = Vec::with_capacity(3);

        for &color in &colors {
            let measurements: Vec<PauliMeasurement> = self
                .edges
                .iter()
                .filter(|&&(_, _, c)| c == color)
                .map(|&(a, b, c)| {
                    let (pa, pb) = color_to_pauli(c);
                    PauliMeasurement::new(a, b, pa, pb)
                })
                .collect();

            rounds.push(MeasurementRound::new(measurements));
        }

        FloquetSchedule::new(rounds, self.n_qubits)
    }

    /// Get the edges of the honeycomb lattice.
    pub fn edges(&self) -> &[(usize, usize, EdgeColor)] {
        &self.edges
    }

    /// Get edges of a specific color.
    pub fn edges_of_color(&self, color: EdgeColor) -> Vec<(usize, usize)> {
        self.edges
            .iter()
            .filter(|&&(_, _, c)| c == color)
            .map(|&(a, b, _)| (a, b))
            .collect()
    }
}

impl FloquetCode for HoneycombCode {
    fn schedule(&self) -> FloquetSchedule {
        HoneycombCode::schedule(self)
    }

    fn num_physical_qubits(&self) -> usize {
        self.n_qubits
    }

    fn num_logical_qubits(&self) -> usize {
        1
    }

    fn distance(&self) -> usize {
        self.dist
    }
}

// ============================================================
// X3Z3 CODE (BIAS-TAILORED)
// ============================================================

/// Bias-tailored Floquet code optimized for Z-biased noise channels.
///
/// The X3Z3 code uses alternating rounds of X-type and Z-type measurements,
/// structured to exploit asymmetric noise where Z errors dominate (as in
/// cat qubits or other bosonic encodings).
///
/// The schedule has a 6-round period:
/// - Rounds 0, 1, 2: X-type measurements (XX on different edge subsets)
/// - Rounds 3, 4, 5: Z-type measurements (ZZ on different edge subsets)
///
/// Under pure Z noise, the X-type measurements provide redundant syndrome
/// information that enhances error correction below threshold.
#[derive(Debug, Clone)]
pub struct X3Z3Code {
    /// Code distance
    dist: usize,
    /// Number of physical qubits
    n_qubits: usize,
    /// Lattice edges partitioned into 3 subsets
    edge_subsets: Vec<Vec<(usize, usize)>>,
}

impl X3Z3Code {
    /// Create a new X3Z3 code with the given distance.
    ///
    /// Uses the same honeycomb lattice geometry but with a different
    /// measurement schedule optimized for biased noise.
    pub fn new(distance: usize) -> Self {
        let dist = distance.max(2);
        let all_edges = honeycomb_lattice(dist);
        let n_qubits = honeycomb_qubit_count(dist);

        // Partition edges into 3 subsets by color
        let mut subsets = vec![Vec::new(); 3];
        for (a, b, color) in &all_edges {
            let idx = match color {
                EdgeColor::Red => 0,
                EdgeColor::Green => 1,
                EdgeColor::Blue => 2,
            };
            subsets[idx].push((*a, *b));
        }

        X3Z3Code {
            dist,
            n_qubits,
            edge_subsets: subsets,
        }
    }

    /// Get the measurement schedule (period 6).
    ///
    /// - Rounds 0..3: XX on each of 3 edge subsets
    /// - Rounds 3..6: ZZ on each of 3 edge subsets
    pub fn schedule(&self) -> FloquetSchedule {
        let mut rounds = Vec::with_capacity(6);

        // X-type rounds
        for subset in &self.edge_subsets {
            let measurements: Vec<PauliMeasurement> = subset
                .iter()
                .map(|&(a, b)| PauliMeasurement::new(a, b, PauliOp::X, PauliOp::X))
                .collect();
            rounds.push(MeasurementRound::new(measurements));
        }

        // Z-type rounds
        for subset in &self.edge_subsets {
            let measurements: Vec<PauliMeasurement> = subset
                .iter()
                .map(|&(a, b)| PauliMeasurement::new(a, b, PauliOp::Z, PauliOp::Z))
                .collect();
            rounds.push(MeasurementRound::new(measurements));
        }

        FloquetSchedule::new(rounds, self.n_qubits)
    }

    /// Estimated error threshold for a given noise bias eta = p_Z / p_X.
    ///
    /// For highly biased noise (eta >> 1), the X3Z3 code has a significantly
    /// higher threshold than the standard honeycomb code.
    ///
    /// This is an analytical estimate, not a numerically computed value.
    pub fn threshold_estimate(bias: f64) -> f64 {
        // At infinite bias (pure Z noise), threshold approaches ~4.5%
        // At no bias (eta = 1), threshold is ~0.7% (comparable to honeycomb)
        // Interpolation based on published results
        let eta = bias.max(1.0);
        let p_inf = 0.045; // Threshold at infinite bias
        let p_1 = 0.007; // Threshold at no bias
        let alpha = 2.0; // Interpolation exponent

        p_1 + (p_inf - p_1) * (1.0 - 1.0 / eta.powf(1.0 / alpha))
    }
}

impl FloquetCode for X3Z3Code {
    fn schedule(&self) -> FloquetSchedule {
        X3Z3Code::schedule(self)
    }

    fn num_physical_qubits(&self) -> usize {
        self.n_qubits
    }

    fn num_logical_qubits(&self) -> usize {
        1
    }

    fn distance(&self) -> usize {
        self.dist
    }
}

// ============================================================
// PROJECTIVE TWO-QUBIT PAULI MEASUREMENT (STATE VECTOR)
// ============================================================

/// Perform a projective measurement of P_1 tensor P_2 on a state vector.
///
/// Measures the two-qubit Pauli operator `p1 x p2` on qubits `q1` and `q2`.
/// The state vector is projected onto the +1 or -1 eigenspace with the
/// correct Born-rule probability, and renormalized.
///
/// Returns `false` for +1 outcome, `true` for -1 outcome.
///
/// # Panics
///
/// Panics if `q1 == q2` or if qubit indices exceed the state vector size.
pub fn measure_two_qubit_pauli(
    state: &mut Vec<C64>,
    q1: usize,
    q2: usize,
    p1: PauliOp,
    p2: PauliOp,
) -> bool {
    assert_ne!(q1, q2, "Measurement qubits must be distinct");
    let n_states = state.len();
    let n_qubits = (n_states as f64).log2() as usize;
    assert!(q1 < n_qubits && q2 < n_qubits, "Qubit index out of range");
    assert_eq!(
        1 << n_qubits,
        n_states,
        "State vector size must be a power of 2"
    );

    // Build the projectors P_+ = (I + P1 x P2) / 2 and P_- = (I - P1 x P2) / 2
    // by computing the action of P1 x P2 on each basis state.

    // For each basis state |k>, compute P1_q1 x P2_q2 |k>
    // The result is a linear combination of at most one basis state (Paulis permute basis states).
    let mut p_state = vec![C64::new(0.0, 0.0); n_states];

    let m1 = p1.matrix();
    let m2 = p2.matrix();

    for k in 0..n_states {
        let b1 = (k >> q1) & 1; // bit of qubit q1
        let b2 = (k >> q2) & 1; // bit of qubit q2

        // Apply P1 on qubit q1: sum over b1' of m1[b1'][b1] * |...b1'...>
        // Apply P2 on qubit q2: sum over b2' of m2[b2'][b2] * |...b2'...>
        for b1p in 0..2usize {
            for b2p in 0..2usize {
                let coeff = m1[b1p][b1] * m2[b2p][b2];
                if coeff.norm() < 1e-15 {
                    continue;
                }
                // New basis state: k with q1 bit set to b1p and q2 bit set to b2p
                let mut kp = k;
                // Clear bits q1 and q2
                kp &= !(1 << q1);
                kp &= !(1 << q2);
                // Set new bits
                kp |= b1p << q1;
                kp |= b2p << q2;

                p_state[kp] += coeff * state[k];
            }
        }
    }

    // Compute probability of +1 outcome: p_plus = <psi| P_+ |psi> = (1 + <psi|P|psi>) / 2
    let mut expectation = C64::new(0.0, 0.0);
    for k in 0..n_states {
        expectation += state[k].conj() * p_state[k];
    }

    let p_plus = (1.0 + expectation.re) / 2.0;
    let p_plus_clamped = p_plus.clamp(0.0, 1.0);

    // Sample outcome
    let mut rng = rand::thread_rng();
    let outcome_plus = rng.gen::<f64>() < p_plus_clamped;

    // Project onto the chosen eigenspace
    // P_+ = (I + P) / 2, P_- = (I - P) / 2
    let sign = if outcome_plus { 1.0 } else { -1.0 };
    let mut norm_sq = 0.0;

    for k in 0..n_states {
        state[k] = C64::new(0.5, 0.0) * (state[k] + C64::new(sign, 0.0) * p_state[k]);
        norm_sq += state[k].norm_sqr();
    }

    // Renormalize
    if norm_sq > 1e-30 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for amp in state.iter_mut() {
            *amp = C64::new(amp.re * inv_norm, amp.im * inv_norm);
        }
    }

    // Return true for -1 eigenvalue (defect), false for +1
    !outcome_plus
}

// ============================================================
// DEFECT
// ============================================================

/// A defect in the Floquet syndrome history.
///
/// A defect occurs when a measurement outcome changes between consecutive
/// rounds of the same type (i.e., same position in the schedule period).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Defect {
    /// The round number in which the defect was detected
    pub round: usize,
    /// The index of the measurement within the round
    pub measurement_idx: usize,
}

impl Defect {
    /// Create a new defect.
    pub fn new(round: usize, measurement_idx: usize) -> Self {
        Self {
            round,
            measurement_idx,
        }
    }
}

impl fmt::Display for Defect {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Defect(r={}, m={})", self.round, self.measurement_idx)
    }
}

// ============================================================
// FLOQUET STATS
// ============================================================

/// Runtime statistics for Floquet code simulation.
#[derive(Debug, Clone, Default)]
pub struct FloquetStats {
    /// Total number of measurement rounds executed
    pub total_rounds: usize,
    /// Total number of defects detected
    pub total_defects: usize,
    /// Running defect rate (defects per measurement)
    pub defect_rate: f64,
    /// Number of logical errors detected
    pub logical_error_count: usize,
    /// Total measurements performed
    total_measurements: usize,
}

impl FloquetStats {
    /// Create a new statistics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a round of syndrome measurements.
    pub fn record_round(&mut self, num_measurements: usize, num_defects: usize) {
        self.total_rounds += 1;
        self.total_defects += num_defects;
        self.total_measurements += num_measurements;
        if self.total_measurements > 0 {
            self.defect_rate = self.total_defects as f64 / self.total_measurements as f64;
        }
    }

    /// Record a logical error.
    pub fn record_logical_error(&mut self) {
        self.logical_error_count += 1;
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Logical error rate (errors per round).
    pub fn logical_error_rate(&self) -> f64 {
        if self.total_rounds == 0 {
            0.0
        } else {
            self.logical_error_count as f64 / self.total_rounds as f64
        }
    }
}

impl fmt::Display for FloquetStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FloquetStats {{ rounds: {}, defects: {}, rate: {:.6}, logical_errors: {} }}",
            self.total_rounds, self.total_defects, self.defect_rate, self.logical_error_count
        )
    }
}

// ============================================================
// FLOQUET SIMULATOR
// ============================================================

/// Simulates Floquet code dynamics using a state vector backend.
///
/// For small systems (up to ~14 qubits), this simulator maintains
/// a full state vector and performs projective two-qubit Pauli
/// measurements according to the Floquet schedule.
///
/// For larger systems, consider using a stabilizer-based backend
/// (all operations in Floquet codes are Clifford).
pub struct FloquetSimulator {
    /// The Floquet measurement schedule
    schedule: FloquetSchedule,
    /// State vector (2^n amplitudes)
    state: Vec<C64>,
    /// Number of qubits
    n_qubits: usize,
    /// Current round index (absolute, not mod period)
    current_round: usize,
    /// Syndrome history: one Vec<bool> per round executed
    syndrome_history: Vec<Vec<bool>>,
    /// Runtime statistics
    pub stats: FloquetStats,
}

impl FloquetSimulator {
    /// Create a new Floquet simulator from a schedule.
    ///
    /// Initializes the state to |0...0> (computational basis).
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits > 20` (state vector too large).
    pub fn new(schedule: FloquetSchedule) -> Self {
        let n = schedule.num_qubits;
        assert!(
            n <= 20,
            "State vector simulation limited to 20 qubits (got {})",
            n
        );

        let dim = 1 << n;
        let mut state = vec![C64::new(0.0, 0.0); dim];
        state[0] = C64::new(1.0, 0.0); // |0...0>

        FloquetSimulator {
            schedule,
            state,
            n_qubits: n,
            current_round: 0,
            syndrome_history: Vec::new(),
            stats: FloquetStats::new(),
        }
    }

    /// Execute one measurement round and return the syndrome bits.
    ///
    /// Each bit corresponds to one measurement in the round: `false` (+1)
    /// or `true` (-1).
    pub fn step(&mut self) -> Vec<bool> {
        let round = self.schedule.round(self.current_round);
        let mut syndrome = Vec::with_capacity(round.len());

        for m in &round.measurements {
            let outcome = measure_two_qubit_pauli(
                &mut self.state,
                m.qubit_a,
                m.qubit_b,
                m.pauli_a,
                m.pauli_b,
            );
            syndrome.push(outcome);
        }

        let num_defects = syndrome.iter().filter(|&&b| b).count();
        self.stats.record_round(round.len(), num_defects);

        self.syndrome_history.push(syndrome.clone());
        self.current_round += 1;

        syndrome
    }

    /// Execute multiple measurement rounds.
    ///
    /// Returns the syndrome bits for each round.
    pub fn evolve(&mut self, num_rounds: usize) -> Vec<Vec<bool>> {
        let mut all_syndromes = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            all_syndromes.push(self.step());
        }
        all_syndromes
    }

    /// Detect defects by comparing consecutive syndromes from rounds
    /// at the same position in the schedule period.
    ///
    /// A defect at (round, measurement_idx) means the syndrome bit changed
    /// compared to the previous round at the same schedule position.
    pub fn detect_defects(&self, syndromes: &[Vec<bool>]) -> Vec<Defect> {
        let mut defects = Vec::new();
        let period = self.schedule.period();

        for (r, syn) in syndromes.iter().enumerate() {
            if r < period {
                // First period: defects are any -1 outcomes
                // (compared to implicit all-+1 initialization)
                for (m_idx, &bit) in syn.iter().enumerate() {
                    if bit {
                        defects.push(Defect::new(r, m_idx));
                    }
                }
            } else {
                // Compare with the syndrome from one period ago
                let prev = &syndromes[r - period];
                for (m_idx, (&curr, &prev_bit)) in syn.iter().zip(prev.iter()).enumerate() {
                    if curr != prev_bit {
                        defects.push(Defect::new(r, m_idx));
                    }
                }
            }
        }

        defects
    }

    /// Inject a single-qubit Pauli error on the given qubit.
    ///
    /// This modifies the state vector by applying the specified Pauli
    /// operator, simulating a physical error.
    pub fn inject_error(&mut self, qubit: usize, error: PauliOp) {
        assert!(qubit < self.n_qubits, "Qubit index out of bounds");
        let mat = error.matrix();
        let dim = self.state.len();
        let mut new_state = vec![C64::new(0.0, 0.0); dim];

        for k in 0..dim {
            let b = (k >> qubit) & 1;
            for bp in 0..2usize {
                let coeff = mat[bp][b];
                if coeff.norm() < 1e-15 {
                    continue;
                }
                let mut kp = k;
                kp &= !(1 << qubit);
                kp |= bp << qubit;
                new_state[kp] += coeff * self.state[k];
            }
        }

        self.state = new_state;
    }

    /// Measure the logical Z operator.
    ///
    /// For the honeycomb code on a planar patch, the logical Z operator
    /// is a string of Z operators along one boundary. We measure it
    /// non-destructively by computing the expectation value.
    ///
    /// Returns `false` for +1 (logical |0>), `true` for -1 (logical |1>).
    pub fn logical_measurement(&self) -> bool {
        // Logical Z is a product of Z operators on qubits in the first column
        // For a 2d x d lattice, column 0 qubits are indices 0, d, 2d, 3d, ...
        let cols = (self.n_qubits / 2).max(1);
        let rows = if cols > 0 { self.n_qubits / cols } else { 1 };

        // Collect qubits in the first column
        let logical_z_qubits: Vec<usize> = (0..rows)
            .map(|r| r * cols)
            .filter(|&q| q < self.n_qubits)
            .collect();

        // Compute <psi| Z_logical |psi>
        let mut expectation = 0.0f64;
        for k in 0..self.state.len() {
            // The eigenvalue of Z_q1 Z_q2 ... on |k> is (-1)^(sum of bits)
            let parity: usize = logical_z_qubits.iter().map(|&q| (k >> q) & 1).sum();
            let eigenvalue = if parity % 2 == 0 { 1.0 } else { -1.0 };
            expectation += eigenvalue * self.state[k].norm_sqr();
        }

        // Return true if the expectation is negative (logical |1>)
        expectation < 0.0
    }

    /// Get the current round index.
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get the full syndrome history.
    pub fn syndrome_history(&self) -> &[Vec<bool>] {
        &self.syndrome_history
    }

    /// Get a reference to the internal state vector.
    pub fn state_vector(&self) -> &[C64] {
        &self.state
    }

    /// Reset the simulator to |0...0>.
    pub fn reset(&mut self) {
        let dim = 1 << self.n_qubits;
        self.state = vec![C64::new(0.0, 0.0); dim];
        self.state[0] = C64::new(1.0, 0.0);
        self.current_round = 0;
        self.syndrome_history.clear();
        self.stats.reset();
    }
}

// ============================================================
// SYNDROME GRAPH
// ============================================================

/// A node in the 3D syndrome graph (space + time).
#[derive(Debug, Clone)]
pub struct SyndromeNode {
    /// Unique node identifier
    pub id: usize,
    /// Round number (time coordinate)
    pub round: usize,
    /// Measurement index within the round (space coordinate)
    pub measurement_idx: usize,
    /// Whether this node is a boundary node (for matching to vacuum)
    pub is_boundary: bool,
}

/// An edge in the syndrome graph connecting two defect sites.
#[derive(Debug, Clone)]
pub struct SyndromeEdge {
    /// Source node index
    pub from: usize,
    /// Destination node index
    pub to: usize,
    /// Edge weight (negative log-likelihood of the error chain)
    pub weight: f64,
    /// The correction associated with this edge (qubit, PauliOp)
    pub correction: Vec<(usize, PauliOp)>,
}

/// The 3D syndrome graph for Floquet code decoding.
///
/// Nodes correspond to measurement locations in spacetime.
/// Edges connect nodes that could be linked by a single error,
/// with weights derived from the noise model.
#[derive(Debug, Clone)]
pub struct SyndromeGraph {
    /// Nodes in the graph
    pub nodes: Vec<SyndromeNode>,
    /// Edges in the graph
    pub edges: Vec<SyndromeEdge>,
    /// Map from (round, measurement_idx) to node index
    node_map: HashMap<(usize, usize), usize>,
}

impl SyndromeGraph {
    /// Create an empty syndrome graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, round: usize, measurement_idx: usize, is_boundary: bool) -> usize {
        let id = self.nodes.len();
        self.nodes.push(SyndromeNode {
            id,
            round,
            measurement_idx,
            is_boundary,
        });
        self.node_map.insert((round, measurement_idx), id);
        id
    }

    /// Add an edge to the graph.
    pub fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        weight: f64,
        correction: Vec<(usize, PauliOp)>,
    ) {
        self.edges.push(SyndromeEdge {
            from,
            to,
            weight,
            correction,
        });
    }

    /// Look up a node by (round, measurement_idx).
    pub fn get_node(&self, round: usize, measurement_idx: usize) -> Option<usize> {
        self.node_map.get(&(round, measurement_idx)).copied()
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

impl Default for SyndromeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// FLOQUET DECODER
// ============================================================

/// Decoder for Floquet code syndromes.
///
/// Uses a greedy nearest-neighbor matching algorithm on the 3D syndrome
/// graph to pair defects and produce a correction.
///
/// For production use, this should be replaced with a full MWPM decoder
/// (see `decoding::mwpm`), but the greedy decoder is sufficient for
/// testing and small-distance codes.
pub struct FloquetDecoder;

impl FloquetDecoder {
    /// Build the 3D syndrome graph for a Floquet schedule.
    ///
    /// Creates nodes for each measurement location across `num_rounds`
    /// time steps, plus boundary nodes. Edges connect temporally and
    /// spatially adjacent measurement sites.
    pub fn build_syndrome_graph(schedule: &FloquetSchedule, num_rounds: usize) -> SyndromeGraph {
        let mut graph = SyndromeGraph::new();
        let period = schedule.period();

        // Add measurement nodes for each round
        for r in 0..num_rounds {
            let round_def = schedule.round(r);
            for m_idx in 0..round_def.len() {
                graph.add_node(r, m_idx, false);
            }
        }

        // Add boundary nodes (one per measurement in each round type)
        let num_measurement_types = schedule.rounds.iter().map(|r| r.len()).max().unwrap_or(0);

        for m_idx in 0..num_measurement_types {
            graph.add_node(num_rounds, m_idx, true);
        }

        // Add temporal edges (same measurement index, consecutive periods)
        for r in 0..num_rounds {
            if r + period < num_rounds {
                let round_def = schedule.round(r);
                for m_idx in 0..round_def.len() {
                    let from = graph.get_node(r, m_idx);
                    let to = graph.get_node(r + period, m_idx);
                    if let (Some(f), Some(t)) = (from, to) {
                        // The correction for a temporal edge is a single-qubit error
                        // on one of the qubits involved in the measurement
                        let meas = &round_def.measurements[m_idx];
                        let correction = vec![(meas.qubit_a, meas.pauli_a)];
                        graph.add_edge(f, t, 1.0, correction);
                    }
                }
            }
        }

        // Add spatial edges (within the same round, adjacent measurements)
        for r in 0..num_rounds {
            let round_def = schedule.round(r);
            for i in 0..round_def.len() {
                for j in (i + 1)..round_def.len() {
                    let mi = &round_def.measurements[i];
                    let mj = &round_def.measurements[j];

                    // Check if measurements share a qubit (spatial adjacency)
                    let shared = mi.qubit_a == mj.qubit_a
                        || mi.qubit_a == mj.qubit_b
                        || mi.qubit_b == mj.qubit_a
                        || mi.qubit_b == mj.qubit_b;

                    if shared {
                        let from = graph.get_node(r, i);
                        let to = graph.get_node(r, j);
                        if let (Some(f), Some(t)) = (from, to) {
                            // Find the shared qubit for the correction
                            let shared_qubit =
                                if mi.qubit_a == mj.qubit_a || mi.qubit_a == mj.qubit_b {
                                    mi.qubit_a
                                } else {
                                    mi.qubit_b
                                };
                            let correction = vec![(shared_qubit, PauliOp::X)];
                            graph.add_edge(f, t, 1.0, correction);
                        }
                    }
                }
            }
        }

        // Add boundary edges
        for r in 0..num_rounds {
            let round_def = schedule.round(r);
            for m_idx in 0..round_def.len() {
                if m_idx < num_measurement_types {
                    let from = graph.get_node(r, m_idx);
                    let to = graph.get_node(num_rounds, m_idx);
                    if let (Some(f), Some(t)) = (from, to) {
                        let meas = &round_def.measurements[m_idx];
                        let correction = vec![(meas.qubit_a, meas.pauli_a)];
                        graph.add_edge(f, t, 0.5, correction);
                    }
                }
            }
        }

        graph
    }

    /// Decode defects using greedy nearest-neighbor matching.
    ///
    /// Pairs defects greedily by minimum spacetime distance and returns
    /// the correction operators to apply.
    ///
    /// This is a simple O(n^2) algorithm. For better decoding performance,
    /// use the MWPM decoder in `decoding::mwpm`.
    pub fn decode_defects(defects: &[Defect], schedule: &FloquetSchedule) -> Vec<(usize, PauliOp)> {
        if defects.is_empty() {
            return Vec::new();
        }

        let mut corrections = Vec::new();
        let mut matched = vec![false; defects.len()];
        let period = schedule.period();

        // Greedy matching: pair closest unmatched defects
        for i in 0..defects.len() {
            if matched[i] {
                continue;
            }

            let mut best_j = None;
            let mut best_dist = f64::INFINITY;

            for j in (i + 1)..defects.len() {
                if matched[j] {
                    continue;
                }

                // Only match defects from the same round type (mod period)
                if defects[i].round % period != defects[j].round % period {
                    continue;
                }

                // Spacetime distance
                let dt = (defects[i].round as f64 - defects[j].round as f64).abs();
                let dm =
                    (defects[i].measurement_idx as f64 - defects[j].measurement_idx as f64).abs();
                let dist = dt + dm;

                if dist < best_dist {
                    best_dist = dist;
                    best_j = Some(j);
                }
            }

            if let Some(j) = best_j {
                matched[i] = true;
                matched[j] = true;

                // Generate correction: apply Pauli on the qubit(s) between the paired defects
                let round_def = schedule.round(defects[i].round);
                if defects[i].measurement_idx < round_def.len() {
                    let meas = &round_def.measurements[defects[i].measurement_idx];
                    corrections.push((meas.qubit_a, meas.pauli_a));
                }
            } else {
                // Unmatched defect: match to boundary
                let round_def = schedule.round(defects[i].round);
                if defects[i].measurement_idx < round_def.len() {
                    let meas = &round_def.measurements[defects[i].measurement_idx];
                    corrections.push((meas.qubit_a, meas.pauli_a));
                }
            }
        }

        // Deduplicate corrections (same qubit, same operator cancels)
        let mut correction_map: HashMap<usize, Vec<PauliOp>> = HashMap::new();
        for (q, p) in &corrections {
            correction_map.entry(*q).or_default().push(*p);
        }

        let mut final_corrections = Vec::new();
        for (q, ops) in &correction_map {
            // Count each operator type; odd count means apply it
            let mut x_count = 0;
            let mut y_count = 0;
            let mut z_count = 0;
            for op in ops {
                match op {
                    PauliOp::X => x_count += 1,
                    PauliOp::Y => y_count += 1,
                    PauliOp::Z => z_count += 1,
                }
            }
            if x_count % 2 == 1 {
                final_corrections.push((*q, PauliOp::X));
            }
            if y_count % 2 == 1 {
                final_corrections.push((*q, PauliOp::Y));
            }
            if z_count % 2 == 1 {
                final_corrections.push((*q, PauliOp::Z));
            }
        }

        final_corrections
    }
}

// ============================================================
// FLOQUET BENCHMARK
// ============================================================

/// Monte Carlo estimation of logical error rates for Floquet codes.
///
/// Runs multiple trials of the Floquet code under depolarizing noise,
/// applying decoding and checking for logical errors.
pub struct FloquetBenchmark;

impl FloquetBenchmark {
    /// Estimate the logical error rate by Monte Carlo sampling.
    ///
    /// For each trial:
    /// 1. Initialize the simulator
    /// 2. Evolve for `num_rounds` measurement rounds
    /// 3. Inject depolarizing noise at rate `error_rate` between rounds
    /// 4. Decode syndromes and apply corrections
    /// 5. Check for logical errors
    ///
    /// Returns the fraction of trials with a logical error.
    pub fn estimate_logical_error_rate(
        code: &dyn FloquetCode,
        error_rate: f64,
        num_rounds: usize,
        num_trials: usize,
    ) -> f64 {
        // Guard against empty trials
        if num_trials == 0 || num_rounds == 0 {
            return 0.0;
        }

        // Only run state-vector simulation for small codes
        let n_qubits = code.num_physical_qubits();
        if n_qubits > 20 {
            // Return analytical estimate for large codes
            let d = code.distance() as f64;
            let p = error_rate;
            return (p / 0.01_f64).powf((d + 1.0) / 2.0).min(1.0);
        }

        let schedule = code.schedule();
        let mut rng = rand::thread_rng();
        let mut logical_errors = 0usize;

        let pauli_ops = [PauliOp::X, PauliOp::Y, PauliOp::Z];

        for _ in 0..num_trials {
            let mut sim = FloquetSimulator::new(schedule.clone());

            let mut all_syndromes = Vec::with_capacity(num_rounds);

            for _ in 0..num_rounds {
                // Inject depolarizing noise
                for qubit in 0..n_qubits {
                    if rng.gen::<f64>() < error_rate {
                        let op = pauli_ops[rng.gen_range(0..3)];
                        sim.inject_error(qubit, op);
                    }
                }

                // Measure
                let syndrome = sim.step();
                all_syndromes.push(syndrome);
            }

            // Detect defects
            let defects = sim.detect_defects(&all_syndromes);

            // Decode and apply corrections
            let corrections = FloquetDecoder::decode_defects(&defects, &schedule);
            for (qubit, op) in &corrections {
                sim.inject_error(*qubit, *op);
            }

            // Check logical error
            if sim.logical_measurement() {
                logical_errors += 1;
            }
        }

        logical_errors as f64 / num_trials as f64
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Helper: create a minimal 2-qubit schedule for testing
    // ----------------------------------------------------------
    fn minimal_schedule() -> FloquetSchedule {
        let round =
            MeasurementRound::new(vec![PauliMeasurement::new(0, 1, PauliOp::Z, PauliOp::Z)]);
        FloquetSchedule::new(vec![round], 2)
    }

    // ----------------------------------------------------------
    // Test 1: Honeycomb code construction (distance 2)
    // ----------------------------------------------------------
    #[test]
    fn test_honeycomb_code_construction_d2() {
        let code = HoneycombCode::new(2);
        assert!(code.num_physical_qubits() > 0);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.code_distance(), 2);
    }

    // ----------------------------------------------------------
    // Test 2: Honeycomb code construction (distance 3)
    // ----------------------------------------------------------
    #[test]
    fn test_honeycomb_code_construction_d3() {
        let code = HoneycombCode::new(3);
        assert_eq!(code.code_distance(), 3);
        assert_eq!(code.num_physical_qubits(), honeycomb_qubit_count(3));
        assert_eq!(code.num_logical_qubits(), 1);
    }

    // ----------------------------------------------------------
    // Test 3: Honeycomb code construction (distance 4)
    // ----------------------------------------------------------
    #[test]
    fn test_honeycomb_code_construction_d4() {
        let code = HoneycombCode::new(4);
        assert_eq!(code.code_distance(), 4);
        let n = code.num_physical_qubits();
        assert!(n > 0, "Should have physical qubits");
        // d=4 -> 2*4*4 = 32 qubits
        assert_eq!(n, 32);
    }

    // ----------------------------------------------------------
    // Test 4: Honeycomb schedule has period 3
    // ----------------------------------------------------------
    #[test]
    fn test_honeycomb_schedule_period_3() {
        let code = HoneycombCode::new(3);
        let schedule = code.schedule();
        assert_eq!(schedule.period(), 3, "Honeycomb code must have period 3");
    }

    // ----------------------------------------------------------
    // Test 5: Measurement round structure
    // ----------------------------------------------------------
    #[test]
    fn test_measurement_round_structure() {
        let code = HoneycombCode::new(2);
        let schedule = code.schedule();

        // Each round should have measurements
        for r in 0..schedule.period() {
            let round = schedule.round(r);
            assert!(!round.is_empty(), "Round {} should have measurements", r);

            // Each measurement should reference valid qubits
            for m in &round.measurements {
                assert!(m.qubit_a < schedule.num_qubits);
                assert!(m.qubit_b < schedule.num_qubits);
                assert_ne!(m.qubit_a, m.qubit_b);
            }
        }
    }

    // ----------------------------------------------------------
    // Test 6: Round 0 has XX, Round 1 has YY, Round 2 has ZZ
    // ----------------------------------------------------------
    #[test]
    fn test_honeycomb_pauli_types() {
        let code = HoneycombCode::new(2);
        let schedule = code.schedule();

        // Round 0: XX (Red edges)
        for m in &schedule.round(0).measurements {
            assert_eq!(m.pauli_a, PauliOp::X);
            assert_eq!(m.pauli_b, PauliOp::X);
        }

        // Round 1: YY (Green edges)
        for m in &schedule.round(1).measurements {
            assert_eq!(m.pauli_a, PauliOp::Y);
            assert_eq!(m.pauli_b, PauliOp::Y);
        }

        // Round 2: ZZ (Blue edges)
        for m in &schedule.round(2).measurements {
            assert_eq!(m.pauli_a, PauliOp::Z);
            assert_eq!(m.pauli_b, PauliOp::Z);
        }
    }

    // ----------------------------------------------------------
    // Test 7: No-error simulation produces no defects after warmup
    // ----------------------------------------------------------
    #[test]
    fn test_no_error_no_defects_after_warmup() {
        // Use a minimal 2-qubit ZZ schedule
        let schedule = minimal_schedule();
        let mut sim = FloquetSimulator::new(schedule);

        // Run 2 rounds to warm up (first round sets the baseline)
        let s0 = sim.step();
        let s1 = sim.step();

        // After 2 rounds of ZZ measurement on |00>, both should be +1
        // and no defects between them
        assert_eq!(
            s0, s1,
            "Repeated ZZ measurement on |00> should be deterministic"
        );
    }

    // ----------------------------------------------------------
    // Test 8: Single X error produces detectable defects
    // ----------------------------------------------------------
    #[test]
    fn test_x_error_produces_defects() {
        let schedule = minimal_schedule();
        let mut sim = FloquetSimulator::new(schedule.clone());

        // Warm up: measure once to establish baseline
        let baseline = sim.step();

        // Inject X error on qubit 0 (flips Z parity)
        sim.inject_error(0, PauliOp::X);

        // Measure again
        let after_error = sim.step();

        // The ZZ measurement should change because X anticommutes with Z
        assert_ne!(
            baseline, after_error,
            "X error should change ZZ measurement outcome"
        );
    }

    // ----------------------------------------------------------
    // Test 9: Single Z error detection on XX measurement
    // ----------------------------------------------------------
    #[test]
    fn test_z_error_on_xx_measurement() {
        // Schedule with XX measurement
        let round =
            MeasurementRound::new(vec![PauliMeasurement::new(0, 1, PauliOp::X, PauliOp::X)]);
        let schedule = FloquetSchedule::new(vec![round], 2);
        let mut sim = FloquetSimulator::new(schedule.clone());

        // Warm up
        let baseline = sim.step();

        // Z error anticommutes with X measurement
        sim.inject_error(0, PauliOp::Z);

        let after_error = sim.step();

        assert_ne!(
            baseline, after_error,
            "Z error should flip XX measurement outcome"
        );
    }

    // ----------------------------------------------------------
    // Test 10: Defect detection from consecutive syndromes
    // ----------------------------------------------------------
    #[test]
    fn test_defect_detection() {
        let schedule = minimal_schedule();
        let sim = FloquetSimulator::new(schedule);

        // Manually construct syndrome histories
        let syndromes = vec![
            vec![false], // round 0: +1
            vec![true],  // round 1: -1 (defect!)
            vec![true],  // round 2: -1 (same as round 1, so no defect relative to period)
        ];

        let defects = sim.detect_defects(&syndromes);
        assert!(!defects.is_empty(), "Should detect at least one defect");

        // First period has defect at round 1 (differs from initial all-false)
        let has_round_1_defect = defects.iter().any(|d| d.round == 1);
        assert!(has_round_1_defect, "Should have defect at round 1");
    }

    // ----------------------------------------------------------
    // Test 11: Greedy decoder finds correction
    // ----------------------------------------------------------
    #[test]
    fn test_greedy_decoder() {
        let code = HoneycombCode::new(2);
        let schedule = code.schedule();

        let defects = vec![Defect::new(0, 0), Defect::new(0, 1)];

        let corrections = FloquetDecoder::decode_defects(&defects, &schedule);

        // Decoder should produce at least one correction
        assert!(
            !corrections.is_empty(),
            "Decoder should produce corrections for paired defects"
        );

        // Each correction should reference a valid qubit
        for (q, _) in &corrections {
            assert!(*q < schedule.num_qubits);
        }
    }

    // ----------------------------------------------------------
    // Test 12: X3Z3 code construction
    // ----------------------------------------------------------
    #[test]
    fn test_x3z3_code_construction() {
        let code = X3Z3Code::new(3);
        assert_eq!(code.dist, 3);
        assert_eq!(code.n_qubits, honeycomb_qubit_count(3));

        let schedule = code.schedule();
        assert_eq!(schedule.period(), 6, "X3Z3 code should have period 6");
    }

    // ----------------------------------------------------------
    // Test 13: X3Z3 schedule has XX then ZZ rounds
    // ----------------------------------------------------------
    #[test]
    fn test_x3z3_schedule_structure() {
        let code = X3Z3Code::new(2);
        let schedule = code.schedule();

        // First 3 rounds: XX measurements
        for r in 0..3 {
            for m in &schedule.round(r).measurements {
                assert_eq!(m.pauli_a, PauliOp::X);
                assert_eq!(m.pauli_b, PauliOp::X);
            }
        }

        // Last 3 rounds: ZZ measurements
        for r in 3..6 {
            for m in &schedule.round(r).measurements {
                assert_eq!(m.pauli_a, PauliOp::Z);
                assert_eq!(m.pauli_b, PauliOp::Z);
            }
        }
    }

    // ----------------------------------------------------------
    // Test 14: Logical measurement without errors
    // ----------------------------------------------------------
    #[test]
    fn test_logical_measurement_no_error() {
        let schedule = minimal_schedule();
        let sim = FloquetSimulator::new(schedule);

        // |00> should give logical |0>
        let result = sim.logical_measurement();
        assert!(!result, "Logical measurement of |00> should be +1 (false)");
    }

    // ----------------------------------------------------------
    // Test 15: FloquetStats tracking
    // ----------------------------------------------------------
    #[test]
    fn test_floquet_stats() {
        let mut stats = FloquetStats::new();
        assert_eq!(stats.total_rounds, 0);
        assert_eq!(stats.total_defects, 0);
        assert_eq!(stats.defect_rate, 0.0);

        stats.record_round(10, 2);
        assert_eq!(stats.total_rounds, 1);
        assert_eq!(stats.total_defects, 2);
        assert!((stats.defect_rate - 0.2).abs() < 1e-10);

        stats.record_round(10, 0);
        assert_eq!(stats.total_rounds, 2);
        assert_eq!(stats.total_defects, 2);
        assert!((stats.defect_rate - 0.1).abs() < 1e-10);

        stats.record_logical_error();
        assert_eq!(stats.logical_error_count, 1);
        assert!((stats.logical_error_rate() - 0.5).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // Test 16: Edge coloring validity
    // ----------------------------------------------------------
    #[test]
    fn test_edge_coloring_valid() {
        let edges = honeycomb_lattice(3);

        // Collect colors per vertex
        let mut vertex_colors: HashMap<usize, Vec<EdgeColor>> = HashMap::new();
        for &(a, b, color) in &edges {
            vertex_colors.entry(a).or_default().push(color);
            vertex_colors.entry(b).or_default().push(color);
        }

        // Each vertex should have at most 3 edges, and no two edges of the same color
        for (v, colors) in &vertex_colors {
            assert!(
                colors.len() <= 3,
                "Vertex {} has {} edges (max 3 for honeycomb)",
                v,
                colors.len()
            );

            // Check no duplicate colors
            let mut seen = std::collections::HashSet::new();
            for c in colors {
                assert!(seen.insert(c), "Vertex {} has duplicate color {}", v, c);
            }
        }
    }

    // ----------------------------------------------------------
    // Test 17: Syndrome graph construction
    // ----------------------------------------------------------
    #[test]
    fn test_syndrome_graph_construction() {
        let code = HoneycombCode::new(2);
        let schedule = code.schedule();
        let num_rounds = 6;

        let graph = FloquetDecoder::build_syndrome_graph(&schedule, num_rounds);

        assert!(graph.num_nodes() > 0, "Syndrome graph should have nodes");
        assert!(graph.num_edges() > 0, "Syndrome graph should have edges");

        // Should have boundary nodes
        let boundary_count = graph.nodes.iter().filter(|n| n.is_boundary).count();
        assert!(boundary_count > 0, "Should have boundary nodes");
    }

    // ----------------------------------------------------------
    // Test 18: Multiple rounds of error-free evolution
    // ----------------------------------------------------------
    #[test]
    fn test_error_free_evolution() {
        let schedule = minimal_schedule();
        let mut sim = FloquetSimulator::new(schedule);

        let syndromes = sim.evolve(6);
        assert_eq!(syndromes.len(), 6, "Should have 6 rounds of syndromes");

        // In error-free simulation on |00>, ZZ measurements should all be +1
        for (r, syn) in syndromes.iter().enumerate() {
            assert_eq!(syn.len(), 1, "Each round should have 1 measurement");
            assert!(!syn[0], "Error-free ZZ on |00> should be +1 at round {}", r);
        }
    }

    // ----------------------------------------------------------
    // Test 19: Error injection and detection
    // ----------------------------------------------------------
    #[test]
    fn test_error_injection_and_detection() {
        let schedule = minimal_schedule();
        let mut sim = FloquetSimulator::new(schedule.clone());

        // Evolve error-free for 2 rounds
        let s0 = sim.step();
        let s1 = sim.step();

        // Inject error
        sim.inject_error(0, PauliOp::X);

        // Next measurement should show the error
        let s2 = sim.step();

        // s0 and s1 should be the same; s2 should differ from s1
        assert_eq!(s0, s1);
        assert_ne!(s1, s2, "Error should change syndrome");

        assert_eq!(sim.stats.total_rounds, 3);
    }

    // ----------------------------------------------------------
    // Test 20: Small distance code (d=2) basic functionality
    // ----------------------------------------------------------
    #[test]
    fn test_small_distance_d2() {
        let code = HoneycombCode::new(2);
        assert_eq!(code.code_distance(), 2);

        let schedule = code.schedule();
        assert_eq!(schedule.period(), 3);

        // Verify we can create a simulator
        // d=2 gives 8 qubits, 2^8 = 256 amplitudes -- manageable
        let mut sim = FloquetSimulator::new(schedule);
        let syn = sim.step();
        assert!(!syn.is_empty());
    }

    // ----------------------------------------------------------
    // Test 21: PauliOp multiplication
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_multiplication() {
        // X * X = I (None)
        assert_eq!(PauliOp::X.multiply(PauliOp::X), None);
        // Y * Y = I
        assert_eq!(PauliOp::Y.multiply(PauliOp::Y), None);
        // Z * Z = I
        assert_eq!(PauliOp::Z.multiply(PauliOp::Z), None);
        // X * Y = iZ
        assert_eq!(PauliOp::X.multiply(PauliOp::Y), Some(PauliOp::Z));
        // Y * Z = iX
        assert_eq!(PauliOp::Y.multiply(PauliOp::Z), Some(PauliOp::X));
        // X * Z = -iY
        assert_eq!(PauliOp::X.multiply(PauliOp::Z), Some(PauliOp::Y));
    }

    // ----------------------------------------------------------
    // Test 22: PauliOp commutativity
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_commutativity() {
        // Same Paulis commute
        assert!(PauliOp::X.commutes_with(PauliOp::X));
        assert!(PauliOp::Y.commutes_with(PauliOp::Y));
        assert!(PauliOp::Z.commutes_with(PauliOp::Z));

        // Different Paulis anticommute
        assert!(!PauliOp::X.commutes_with(PauliOp::Y));
        assert!(!PauliOp::Y.commutes_with(PauliOp::Z));
        assert!(!PauliOp::X.commutes_with(PauliOp::Z));
    }

    // ----------------------------------------------------------
    // Test 23: FloquetCode trait implementation
    // ----------------------------------------------------------
    #[test]
    fn test_floquet_code_trait() {
        let code = HoneycombCode::new(3);
        let code_ref: &dyn FloquetCode = &code;

        assert_eq!(code_ref.distance(), 3);
        assert_eq!(code_ref.num_logical_qubits(), 1);
        assert!(code_ref.num_physical_qubits() > 0);

        let schedule = code_ref.schedule();
        assert_eq!(schedule.period(), 3);
    }

    // ----------------------------------------------------------
    // Test 24: X3Z3 threshold estimate
    // ----------------------------------------------------------
    #[test]
    fn test_x3z3_threshold_estimate() {
        // At no bias (eta=1), threshold should be around 0.7%
        let t1 = X3Z3Code::threshold_estimate(1.0);
        assert!(t1 >= 0.0 && t1 < 0.01, "Threshold at eta=1 should be ~0.7%");

        // At high bias, threshold should be higher
        let t100 = X3Z3Code::threshold_estimate(100.0);
        assert!(t100 > t1, "Higher bias should give higher threshold");

        // Threshold should be at most ~4.5%
        let t_inf = X3Z3Code::threshold_estimate(1e6);
        assert!(t_inf < 0.05, "Threshold should not exceed ~4.5%");
    }

    // ----------------------------------------------------------
    // Test 25: Schedule round modular access
    // ----------------------------------------------------------
    #[test]
    fn test_schedule_modular_access() {
        let code = HoneycombCode::new(2);
        let schedule = code.schedule();

        // round(t) should equal round(t + period)
        for t in 0..3 {
            let r1 = schedule.round(t);
            let r2 = schedule.round(t + 3);
            assert_eq!(r1.len(), r2.len());
            for (m1, m2) in r1.measurements.iter().zip(r2.measurements.iter()) {
                assert_eq!(m1.qubit_a, m2.qubit_a);
                assert_eq!(m1.qubit_b, m2.qubit_b);
                assert_eq!(m1.pauli_a, m2.pauli_a);
                assert_eq!(m1.pauli_b, m2.pauli_b);
            }
        }
    }
}
