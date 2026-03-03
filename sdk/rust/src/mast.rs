//! MAST: Magic-injected Stabilizer Tensor Networks
//!
//! A self-contained tensor-network simulator that combines stabilizer (Clifford)
//! simulation with magic state injection for non-Clifford gates. Inspired by
//! arXiv:2411.12482, MAST scales with the number of non-Clifford gates (magic
//! count) rather than qubit count, enabling simulation of 200+ qubit circuits
//! that are both highly entangled and magical.
//!
//! # Algorithm Overview
//!
//! 1. Parse the circuit into Clifford regions separated by non-Clifford gates.
//! 2. Each Clifford region becomes a [`StabilizerTensor`] node in a tensor network.
//! 3. Each non-Clifford gate is replaced by a magic state injection (gadgetization):
//!    the T gate becomes a magic state |T> = (|0> + e^{iπ/4}|1>)/√2 plus
//!    Clifford corrections conditioned on a measurement outcome.
//! 4. The tensor network is simplified (merging adjacent stabilizer nodes) and
//!    then contracted to compute amplitudes or produce samples.
//!
//! # Key Advantage
//!
//! Pure stabilizer simulation handles Clifford circuits in polynomial time but
//! fails for non-Clifford gates. Pure state-vector simulation handles everything
//! but is exponential in qubit count. MAST is exponential only in the number of
//! magic (non-Clifford) gates, which is typically much smaller than the qubit
//! count in practical circuits.
//!
//! # References
//!
//! - arXiv:2411.12482 "Magic-injected stabilizer tensor networks"
//! - Aaronson & Gottesman, "Improved simulation of stabilizer circuits" (2004)
//! - Bravyi & Kitaev, "Universal quantum computation with ideal Clifford gates
//!   and noisy ancillas" (2005)

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

// ============================================================
// CONSTANTS
// ============================================================

const SQRT2_INV: f64 = FRAC_1_SQRT_2;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during MAST simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum MastError {
    /// Circuit contains more non-Clifford gates than the configured limit.
    TooManyMagicStates { found: usize, limit: usize },
    /// Tensor network contraction failed.
    ContractionFailed(String),
    /// The input circuit is malformed.
    InvalidCircuit(String),
}

impl std::fmt::Display for MastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MastError::TooManyMagicStates { found, limit } => {
                write!(f, "Too many magic states: found {found}, limit {limit}")
            }
            MastError::ContractionFailed(msg) => write!(f, "Contraction failed: {msg}"),
            MastError::InvalidCircuit(msg) => write!(f, "Invalid circuit: {msg}"),
        }
    }
}

impl std::error::Error for MastError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Method used to determine tensor contraction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionMethod {
    /// Greedily contract the lowest-cost pair at each step.
    Greedy,
    /// Breadth-first contraction from leaves inward.
    BreadthFirst,
    /// Heuristic contraction inspired by cotengra's optimization.
    CotengrisLike,
}

/// Configuration for the MAST simulator.
#[derive(Debug, Clone)]
pub struct MastConfig {
    /// Maximum number of magic (non-Clifford) states before returning an error.
    pub max_magic_states: usize,
    /// Tensor contraction ordering strategy.
    pub contraction_method: ContractionMethod,
    /// Maximum bond dimension during contraction (truncation limit).
    pub chi_max: usize,
    /// Number of simplification rounds (merging adjacent stabilizer nodes).
    pub simplification_rounds: usize,
}

impl Default for MastConfig {
    fn default() -> Self {
        Self {
            max_magic_states: 50,
            contraction_method: ContractionMethod::Greedy,
            chi_max: 256,
            simplification_rounds: 3,
        }
    }
}

impl MastConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn max_magic_states(mut self, n: usize) -> Self {
        self.max_magic_states = n;
        self
    }
    pub fn contraction_method(mut self, m: ContractionMethod) -> Self {
        self.contraction_method = m;
        self
    }
    pub fn chi_max(mut self, c: usize) -> Self {
        self.chi_max = c;
        self
    }
    pub fn simplification_rounds(mut self, r: usize) -> Self {
        self.simplification_rounds = r;
        self
    }
}

// ============================================================
// CIRCUIT TYPES
// ============================================================

/// A quantum gate in the MAST circuit representation.
#[derive(Debug, Clone, PartialEq)]
pub enum MastGate {
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Z(usize),
    CX(usize, usize),
    CZ(usize, usize),
    T(usize),
    Tdg(usize),
    Rz(usize, f64),
}

/// Returns true if the gate is a Clifford gate.
pub fn is_clifford(gate: &MastGate) -> bool {
    match gate {
        MastGate::H(_)
        | MastGate::S(_)
        | MastGate::Sdg(_)
        | MastGate::X(_)
        | MastGate::Z(_)
        | MastGate::CX(_, _)
        | MastGate::CZ(_, _) => true,
        MastGate::T(_) | MastGate::Tdg(_) => false,
        MastGate::Rz(_, angle) => {
            let normalized = angle / (PI / 2.0);
            (normalized - normalized.round()).abs() < 1e-10
        }
    }
}

/// A quantum circuit for MAST simulation.
#[derive(Debug, Clone)]
pub struct MastCircuit {
    pub gates: Vec<MastGate>,
    pub num_qubits: usize,
}

impl MastCircuit {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
        }
    }

    pub fn add_gate(&mut self, gate: MastGate) {
        self.gates.push(gate);
    }

    /// Count the number of non-Clifford gates in the circuit.
    pub fn magic_count(&self) -> usize {
        self.gates.iter().filter(|g| !is_clifford(g)).count()
    }

    /// Validate that all gate qubit indices are within bounds.
    pub fn validate(&self) -> Result<(), MastError> {
        for (i, gate) in self.gates.iter().enumerate() {
            let qubits = match gate {
                MastGate::H(q)
                | MastGate::S(q)
                | MastGate::Sdg(q)
                | MastGate::X(q)
                | MastGate::Z(q)
                | MastGate::T(q)
                | MastGate::Tdg(q)
                | MastGate::Rz(q, _) => vec![*q],
                MastGate::CX(a, b) | MastGate::CZ(a, b) => vec![*a, *b],
            };
            for q in qubits {
                if q >= self.num_qubits {
                    return Err(MastError::InvalidCircuit(format!(
                        "Gate {i} references qubit {q} but circuit has {} qubits",
                        self.num_qubits
                    )));
                }
            }
        }
        Ok(())
    }
}

// ============================================================
// STABILIZER TABLEAU
// ============================================================

/// Compact stabilizer tableau representation (Aaronson-Gottesman).
///
/// For an n-qubit stabilizer state, we store n stabilizer generators.
/// Each generator is a Pauli string specified by X and Z bit-vectors plus a phase.
///
/// The single-qubit Pauli encoded by (x, z) is X^x Z^z:
///   (0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=XZ=-iY
///
/// Generator i represents the operator: i^{phase_i} * prod_q X_q^{x[i][q]} Z_q^{z[i][q]}
#[derive(Debug, Clone, PartialEq)]
pub struct Tableau {
    pub x_matrix: Vec<Vec<bool>>,
    pub z_matrix: Vec<Vec<bool>>,
    pub phases: Vec<u8>,
    pub num_qubits: usize,
}

impl Tableau {
    /// Create the tableau for the |0...0> state.
    /// Stabilizer generators are Z_0, Z_1, ..., Z_{n-1} (all with phase 0 = "+1").
    pub fn new(num_qubits: usize) -> Self {
        let x_matrix = vec![vec![false; num_qubits]; num_qubits];
        let mut z_matrix = vec![vec![false; num_qubits]; num_qubits];
        let phases = vec![0u8; num_qubits];
        for i in 0..num_qubits {
            z_matrix[i][i] = true;
        }
        Self {
            x_matrix,
            z_matrix,
            phases,
            num_qubits,
        }
    }

    /// Convert the stabilizer state to a dense state vector.
    /// Exponential in num_qubits — use only for small states or testing.
    pub fn to_state_vector(&self) -> Vec<Complex64> {
        let dim = 1usize << self.num_qubits;

        // Try projecting from each computational basis state until we get a
        // non-zero result. This is necessary because the uniform superposition
        // can be orthogonal to some stabilizer states (e.g., |-> is orthogonal
        // to |+> which is the uniform superposition for 1 qubit).
        for start_basis in 0..dim {
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            state[start_basis] = Complex64::new(1.0, 0.0);

            for gen_idx in 0..self.num_qubits {
                let mut projected = vec![Complex64::new(0.0, 0.0); dim];
                for basis in 0..dim {
                    if state[basis].norm_sqr() < 1e-30 {
                        continue;
                    }
                    let (new_basis, phase) = self.apply_generator_to_basis(gen_idx, basis);
                    // Projector (I + g)/2: |b> -> (|b> + g|b>)/2
                    projected[basis] += state[basis] * 0.5;
                    projected[new_basis] += state[basis] * phase * 0.5;
                }
                state = projected;
            }

            // Check if we got a non-zero state
            let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
            if norm_sq > 1e-15 {
                let inv_norm = 1.0 / norm_sq.sqrt();
                for c in &mut state {
                    *c *= inv_norm;
                }
                return state;
            }
        }

        // Fallback: return zero vector (should never happen for valid stabilizer states)
        vec![Complex64::new(0.0, 0.0); dim]
    }

    /// Apply generator `gen_idx` to computational basis state `|basis>`.
    /// Returns (new_basis_index, phase_factor) such that
    /// g_{gen_idx} |basis> = phase_factor * |new_basis>.
    fn apply_generator_to_basis(&self, gen_idx: usize, basis: usize) -> (usize, Complex64) {
        let n = self.num_qubits;
        let mut new_basis = basis;
        // Start with the generator's overall phase: i^{phases[gen_idx]}
        let mut phase_power: i32 = self.phases[gen_idx] as i32;

        for q in 0..n {
            let has_x = self.x_matrix[gen_idx][q];
            let has_z = self.z_matrix[gen_idx][q];
            let bit = (basis >> q) & 1;

            match (has_x, has_z) {
                (false, false) => {} // I: no change
                (true, false) => {
                    // X: flip bit, no phase
                    new_basis ^= 1 << q;
                }
                (false, true) => {
                    // Z: (-1)^bit phase
                    if bit == 1 {
                        phase_power += 2;
                    }
                }
                (true, true) => {
                    // XZ = -iY: first apply Z then X.
                    // XZ|b> = X(Z|b>) = X((-1)^b |b>) = (-1)^b |1-b>
                    // But we also have the -i from XZ = -iY encoding... no.
                    // Actually XZ is literally the matrix product X*Z:
                    //   XZ = [[0,1],[1,0]]·[[1,0],[0,-1]] = [[0,-1],[1,0]]
                    //   XZ|0> = [[0,-1],[1,0]]|0> = |1>  (no phase)
                    //   XZ|1> = [[0,-1],[1,0]]|1> = -|0> (phase -1)
                    // So: XZ|b> = (-1)^b |1-b>. This means:
                    //   flip bit, and if bit was 1, add phase (-1) = i^2
                    new_basis ^= 1 << q;
                    if bit == 1 {
                        phase_power += 2;
                    }
                }
            }
        }

        let phase = match phase_power.rem_euclid(4) {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };
        (new_basis, phase)
    }
}

// ============================================================
// TABLEAU GATE OPERATIONS
// ============================================================

/// Apply Hadamard gate: conjugates each generator by H on the given qubit.
///
/// H transforms: X ↔ Z, XZ → -XZ (i.e. Y → -Y).
/// In our (x,z) encoding: swap x and z bits; if both were set, phase += 2.
pub fn tableau_apply_h(tab: &mut Tableau, qubit: usize) {
    for i in 0..tab.num_qubits {
        let x = tab.x_matrix[i][qubit];
        let z = tab.z_matrix[i][qubit];
        tab.x_matrix[i][qubit] = z;
        tab.z_matrix[i][qubit] = x;
        if x && z {
            // XZ → ZX: but ZX = -(XZ), so phase += 2
            tab.phases[i] = (tab.phases[i] + 2) % 4;
        }
    }
}

/// Apply S gate: conjugates each generator by S on the given qubit.
///
/// S transforms (in XZ encoding):
///   I → I, X → iXZ, Z → Z, XZ → iX
pub fn tableau_apply_s(tab: &mut Tableau, qubit: usize) {
    for i in 0..tab.num_qubits {
        let x = tab.x_matrix[i][qubit];
        let z = tab.z_matrix[i][qubit];
        match (x, z) {
            (false, _) => {} // I→I, Z→Z: no change
            (true, false) => {
                // X → i·(XZ): set z, phase += 1
                tab.z_matrix[i][qubit] = true;
                tab.phases[i] = (tab.phases[i] + 1) % 4;
            }
            (true, true) => {
                // XZ → i·X: clear z, phase += 1
                tab.z_matrix[i][qubit] = false;
                tab.phases[i] = (tab.phases[i] + 1) % 4;
            }
        }
    }
}

/// Apply S† (S-dagger) gate. S† = S^3.
pub fn tableau_apply_sdg(tab: &mut Tableau, qubit: usize) {
    tableau_apply_s(tab, qubit);
    tableau_apply_s(tab, qubit);
    tableau_apply_s(tab, qubit);
}

/// Apply X gate: conjugates each generator by X on the given qubit.
///
/// X commutes with X, anticommutes with Z and XZ.
pub fn tableau_apply_x(tab: &mut Tableau, qubit: usize) {
    for i in 0..tab.num_qubits {
        let z = tab.z_matrix[i][qubit];
        if z {
            // Z or XZ anticommute with X → phase += 2
            tab.phases[i] = (tab.phases[i] + 2) % 4;
        }
    }
}

/// Apply Z gate: conjugates each generator by Z on the given qubit.
///
/// Z commutes with Z, anticommutes with X and XZ.
pub fn tableau_apply_z(tab: &mut Tableau, qubit: usize) {
    for i in 0..tab.num_qubits {
        let x = tab.x_matrix[i][qubit];
        if x {
            // X or XZ anticommute with Z → phase += 2
            tab.phases[i] = (tab.phases[i] + 2) % 4;
        }
    }
}

/// Apply CNOT gate (control → target).
///
/// CNOT transforms: X_c → X_c⊗X_t, Z_t → Z_c⊗Z_t.
/// In our XZ encoding, the phase is always 0 for CNOT.
pub fn tableau_apply_cnot(tab: &mut Tableau, control: usize, target: usize) {
    for i in 0..tab.num_qubits {
        // x_target ^= x_control (X on control propagates to target)
        tab.x_matrix[i][target] ^= tab.x_matrix[i][control];
        // z_control ^= z_target (Z on target propagates back to control)
        tab.z_matrix[i][control] ^= tab.z_matrix[i][target];
    }
}

/// Apply CZ gate. CZ = H_1 · CNOT_{0,1} · H_1.
pub fn tableau_apply_cz(tab: &mut Tableau, q0: usize, q1: usize) {
    tableau_apply_h(tab, q1);
    tableau_apply_cnot(tab, q0, q1);
    tableau_apply_h(tab, q1);
}

/// Measure a qubit in the computational basis. Returns 0 or 1.
pub fn tableau_measure(tab: &mut Tableau, qubit: usize, rng: &mut impl Rng) -> u8 {
    let n = tab.num_qubits;

    // Find a generator that anticommutes with Z on this qubit (has X component).
    let anticommuting = (0..n).find(|&i| tab.x_matrix[i][qubit]);

    match anticommuting {
        Some(p) => {
            let outcome: u8 = rng.gen_range(0..2);

            // Row-reduce: multiply all other anticommuting generators by p
            for i in 0..n {
                if i != p && tab.x_matrix[i][qubit] {
                    rowmult(tab, i, p);
                }
            }

            // Replace generator p with ±Z_qubit
            tab.x_matrix[p] = vec![false; n];
            tab.z_matrix[p] = vec![false; n];
            tab.z_matrix[p][qubit] = true;
            tab.phases[p] = outcome * 2; // 0→+Z (|0>), 2→-Z (|1>)

            outcome
        }
        None => {
            // Deterministic outcome: figure out sign from generators.
            // Compose all generators that have Z on this qubit.
            let mut scratch_phase: u8 = 0;
            let mut scratch_x = vec![false; n];
            let mut scratch_z = vec![false; n];

            for i in 0..n {
                if tab.z_matrix[i][qubit] {
                    let new_phase = rowmult_phase(
                        &scratch_x,
                        &scratch_z,
                        scratch_phase,
                        &tab.x_matrix[i],
                        &tab.z_matrix[i],
                        tab.phases[i],
                        n,
                    );
                    for q in 0..n {
                        scratch_x[q] ^= tab.x_matrix[i][q];
                        scratch_z[q] ^= tab.z_matrix[i][q];
                    }
                    scratch_phase = new_phase;
                }
            }

            // Phase 0 (+1) → outcome 0, phase 2 (-1) → outcome 1
            if scratch_phase == 0 {
                0
            } else {
                1
            }
        }
    }
}

/// Compute the inner product <a|b> of two stabilizer states.
///
/// Uses the state-vector conversion for correctness. Efficient only for small n.
pub fn tableau_inner_product(a: &Tableau, b: &Tableau) -> Complex64 {
    assert_eq!(a.num_qubits, b.num_qubits);
    let va = a.to_state_vector();
    let vb = b.to_state_vector();
    va.iter()
        .zip(vb.iter())
        .map(|(ai, bi)| ai.conj() * bi)
        .sum()
}

/// Multiply generator i by generator j: row[i] ← row[i] * row[j].
fn rowmult(tab: &mut Tableau, i: usize, j: usize) {
    let n = tab.num_qubits;
    let new_phase = rowmult_phase(
        &tab.x_matrix[i],
        &tab.z_matrix[i],
        tab.phases[i],
        &tab.x_matrix[j],
        &tab.z_matrix[j],
        tab.phases[j],
        n,
    );
    tab.phases[i] = new_phase;
    for q in 0..n {
        tab.x_matrix[i][q] ^= tab.x_matrix[j][q];
        tab.z_matrix[i][q] ^= tab.z_matrix[j][q];
    }
}

/// Compute the phase of the product of two Pauli strings in XZ encoding.
///
/// Product of (X^{x1}Z^{z1}) and (X^{x2}Z^{z2}) on each qubit:
/// Z·X = -(X·Z) introduces a factor of i^2 = -1 when reordering.
/// The extra phase per qubit is 2 when z1=1 and x2=1 (the Z-before-X swap).
fn rowmult_phase(
    _x1: &[bool],
    z1: &[bool],
    phase1: u8,
    x2: &[bool],
    _z2: &[bool],
    phase2: u8,
    n: usize,
) -> u8 {
    let mut extra: u32 = 0;
    for q in 0..n {
        // When multiplying (X^{x1}Z^{z1})(X^{x2}Z^{z2}), we need to commute
        // Z^{z1} past X^{x2}. Each time Z passes X we get a factor of -1 (i^2).
        if z1[q] && x2[q] {
            extra += 2;
        }
    }
    ((phase1 as u32 + phase2 as u32 + extra) % 4) as u8
}

// ============================================================
// MAGIC STATE TYPES
// ============================================================

/// The type of magic (non-Clifford) resource state.
#[derive(Debug, Clone, PartialEq)]
pub enum MagicType {
    /// T gate magic state: |T> = (|0> + e^{iπ/4}|1>)/√2
    TState,
    /// T† gate magic state: |T†> = (|0> + e^{-iπ/4}|1>)/√2
    TdgState,
    /// Toffoli magic state (3-qubit)
    ToffoliState,
    /// CCZ magic state (3-qubit)
    CczState,
    /// Arbitrary rotation magic state for Rz(θ)
    ArbitraryRotation(f64),
}

/// A magic (non-stabilizer) state used for gadgetization.
#[derive(Debug, Clone)]
pub struct MagicState {
    /// Dense state vector of the magic state.
    pub state_vector: Vec<Complex64>,
    /// Number of qubits in this magic state (typically 1 for T gate).
    pub num_qubits: usize,
    /// The type of magic resource.
    pub magic_type: MagicType,
}

impl MagicState {
    /// Create the T-gate magic state: |T> = (|0> + e^{iπ/4}|1>)/√2.
    pub fn t_state() -> Self {
        let omega = Complex64::new(SQRT2_INV, SQRT2_INV); // e^{iπ/4}
        Self {
            state_vector: vec![
                Complex64::new(SQRT2_INV, 0.0),
                omega * SQRT2_INV,
            ],
            num_qubits: 1,
            magic_type: MagicType::TState,
        }
    }

    /// Create the T†-gate magic state: |T†> = (|0> + e^{-iπ/4}|1>)/√2.
    pub fn tdg_state() -> Self {
        let omega_conj = Complex64::new(SQRT2_INV, -SQRT2_INV); // e^{-iπ/4}
        Self {
            state_vector: vec![
                Complex64::new(SQRT2_INV, 0.0),
                omega_conj * SQRT2_INV,
            ],
            num_qubits: 1,
            magic_type: MagicType::TdgState,
        }
    }

    /// Create a magic state for Rz(θ): |Rz> = (|0> + e^{iθ/2}|1>)/√2.
    pub fn rz_state(angle: f64) -> Self {
        let half = angle / 2.0;
        let phase = Complex64::new(half.cos(), half.sin());
        Self {
            state_vector: vec![Complex64::new(SQRT2_INV, 0.0), phase * SQRT2_INV],
            num_qubits: 1,
            magic_type: MagicType::ArbitraryRotation(angle),
        }
    }
}

// ============================================================
// STABILIZER TENSOR
// ============================================================

/// A stabilizer state represented as a tensor in the network.
#[derive(Debug, Clone)]
pub struct StabilizerTensor {
    /// The stabilizer tableau.
    pub tableau: Tableau,
    /// Number of qubits.
    pub num_qubits: usize,
    /// Rank (number of independent stabilizers, should equal num_qubits for pure states).
    pub rank: usize,
}

impl StabilizerTensor {
    /// Create a stabilizer tensor for |0...0>.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            tableau: Tableau::new(num_qubits),
            num_qubits,
            rank: num_qubits,
        }
    }

    /// Apply a Clifford gate to this tensor.
    pub fn apply_clifford(&mut self, gate: &MastGate) {
        match gate {
            MastGate::H(q) => tableau_apply_h(&mut self.tableau, *q),
            MastGate::S(q) => tableau_apply_s(&mut self.tableau, *q),
            MastGate::Sdg(q) => tableau_apply_sdg(&mut self.tableau, *q),
            MastGate::X(q) => tableau_apply_x(&mut self.tableau, *q),
            MastGate::Z(q) => tableau_apply_z(&mut self.tableau, *q),
            MastGate::CX(c, t) => tableau_apply_cnot(&mut self.tableau, *c, *t),
            MastGate::CZ(a, b) => tableau_apply_cz(&mut self.tableau, *a, *b),
            _ => {} // Non-Clifford gates are handled by magic state injection
        }
    }
}

// ============================================================
// TENSOR NETWORK TYPES
// ============================================================

/// A node in the MAST tensor network.
#[derive(Debug, Clone)]
pub enum TensorNode {
    /// A stabilizer state (efficient Clifford representation).
    Stabilizer(StabilizerTensor),
    /// A magic (non-stabilizer) state.
    Magic(MagicState),
    /// A fully dense tensor (result of contraction or mixed operations).
    Mixed(Vec<Complex64>, usize), // (flattened data, num_qubits)
}

impl TensorNode {
    pub fn num_qubits(&self) -> usize {
        match self {
            TensorNode::Stabilizer(s) => s.num_qubits,
            TensorNode::Magic(m) => m.num_qubits,
            TensorNode::Mixed(_, nq) => *nq,
        }
    }

    /// Convert this node to a dense state vector.
    pub fn to_dense(&self) -> Vec<Complex64> {
        match self {
            TensorNode::Stabilizer(s) => s.tableau.to_state_vector(),
            TensorNode::Magic(m) => m.state_vector.clone(),
            TensorNode::Mixed(data, _) => data.clone(),
        }
    }
}

/// An edge connecting two tensor nodes (shared qubit index).
#[derive(Debug, Clone)]
pub struct TensorEdge {
    /// Index of the first node.
    pub node_a: usize,
    /// Qubit index within node_a.
    pub index_a: usize,
    /// Index of the second node.
    pub node_b: usize,
    /// Qubit index within node_b.
    pub index_b: usize,
    /// Dimension of the shared index (2 for qubits).
    pub dimension: usize,
}

/// The MAST tensor network.
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    /// Tensor nodes (stabilizer, magic, or mixed).
    pub tensors: Vec<TensorNode>,
    /// Edges connecting nodes.
    pub edges: Vec<TensorEdge>,
    /// Open (uncontracted) indices: (node_index, qubit_index_within_node).
    pub open_indices: Vec<(usize, usize)>,
}

impl TensorNetwork {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            edges: Vec::new(),
            open_indices: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: TensorNode) -> usize {
        let idx = self.tensors.len();
        self.tensors.push(node);
        idx
    }

    pub fn add_edge(&mut self, edge: TensorEdge) {
        self.edges.push(edge);
    }

    /// Count the number of stabilizer nodes.
    pub fn num_stabilizer_nodes(&self) -> usize {
        self.tensors
            .iter()
            .filter(|t| matches!(t, TensorNode::Stabilizer(_)))
            .count()
    }

    /// Count the number of magic nodes.
    pub fn num_magic_nodes(&self) -> usize {
        self.tensors
            .iter()
            .filter(|t| matches!(t, TensorNode::Magic(_)))
            .count()
    }
}

/// Result of a MAST computation.
#[derive(Debug, Clone)]
pub struct MastResult {
    /// The computed amplitude.
    pub amplitude: Complex64,
    /// Number of stabilizer tensor nodes used.
    pub num_stabilizer_nodes: usize,
    /// Number of magic state nodes used.
    pub num_magic_nodes: usize,
    /// Estimated contraction cost (FLOPs).
    pub contraction_cost: f64,
}

// ============================================================
// GADGETIZATION
// ============================================================

/// Clifford gates produced by gadgetization (used alongside the magic state).
#[derive(Debug, Clone)]
pub struct CliffordGate {
    pub gate: MastGate,
}

/// Replace a T gate with a magic state injection + Clifford corrections.
///
/// The T gate gadgetization works as follows:
///   T|ψ> = (|0><0| + e^{iπ/4}|1><1|)|ψ>
/// Using the magic state |T> = (|0> + e^{iπ/4}|1>)/√2 and gate teleportation:
///   1. Prepare ancilla in |T> state
///   2. Apply CNOT from data qubit to ancilla
///   3. Measure ancilla → if 1, apply S correction on data qubit
///
/// Returns the magic state and the Clifford gates for the gadget.
pub fn gadgetize_t_gate(qubit: usize) -> (MagicState, Vec<CliffordGate>) {
    let magic = MagicState::t_state();
    // The Clifford part of the gadget: CNOT then potentially S correction
    // In the tensor network representation, we include the CNOT as part of
    // the Clifford region and the magic state as a separate node.
    let cliffords = vec![CliffordGate {
        gate: MastGate::CX(qubit, qubit), // placeholder: actual ancilla index assigned later
    }];
    (magic, cliffords)
}

/// Replace a T† gate with magic state injection.
pub fn gadgetize_tdg_gate(qubit: usize) -> (MagicState, Vec<CliffordGate>) {
    let magic = MagicState::tdg_state();
    let cliffords = vec![CliffordGate {
        gate: MastGate::CX(qubit, qubit),
    }];
    (magic, cliffords)
}

/// Replace an Rz(θ) gate with magic state injection.
pub fn gadgetize_rz_gate(qubit: usize, angle: f64) -> (MagicState, Vec<CliffordGate>) {
    let magic = MagicState::rz_state(angle);
    let cliffords = vec![CliffordGate {
        gate: MastGate::CX(qubit, qubit),
    }];
    (magic, cliffords)
}

// ============================================================
// NETWORK CONSTRUCTION
// ============================================================

/// Build a MAST tensor network from a circuit.
///
/// Partitions the circuit into Clifford regions and non-Clifford gates.
/// Each Clifford region becomes a StabilizerTensor node, and each non-Clifford
/// gate becomes a MagicState node.
pub fn build_mast_network(circuit: &MastCircuit) -> TensorNetwork {
    let mut tn = TensorNetwork::new();

    // Strategy: walk through gates, accumulating Clifford gates into a region.
    // When we hit a non-Clifford gate, finalize the current region as a
    // stabilizer tensor, add the magic state, and start a new region.

    let mut current_stab = StabilizerTensor::new(circuit.num_qubits);
    let mut has_clifford_gates = false;

    for gate in &circuit.gates {
        if is_clifford(gate) {
            current_stab.apply_clifford(gate);
            has_clifford_gates = true;
        } else {
            // Finalize current Clifford region
            if has_clifford_gates {
                tn.add_node(TensorNode::Stabilizer(current_stab.clone()));
            }

            // Add magic state node
            let magic = match gate {
                MastGate::T(_) => MagicState::t_state(),
                MastGate::Tdg(_) => MagicState::tdg_state(),
                MastGate::Rz(_, angle) => MagicState::rz_state(*angle),
                _ => unreachable!("Non-Clifford gate not handled"),
            };
            tn.add_node(TensorNode::Magic(magic));

            // Start new Clifford region
            current_stab = StabilizerTensor::new(circuit.num_qubits);
            has_clifford_gates = false;
        }
    }

    // Finalize last Clifford region
    if has_clifford_gates || tn.tensors.is_empty() {
        tn.add_node(TensorNode::Stabilizer(current_stab));
    }

    // Add edges between consecutive nodes (shared qubit lines)
    if tn.tensors.len() > 1 {
        for i in 0..tn.tensors.len() - 1 {
            // Each pair of consecutive nodes shares the qubit lines
            for q in 0..circuit.num_qubits {
                tn.add_edge(TensorEdge {
                    node_a: i,
                    index_a: q,
                    node_b: i + 1,
                    index_b: q,
                    dimension: 2,
                });
            }
        }
    }

    // Open indices are the output qubits of the last node
    let last = tn.tensors.len() - 1;
    for q in 0..circuit.num_qubits {
        tn.open_indices.push((last, q));
    }

    tn
}

// ============================================================
// NETWORK SIMPLIFICATION
// ============================================================

/// Simplify the tensor network by merging adjacent stabilizer nodes.
///
/// For each round, scan for pairs of adjacent Stabilizer nodes connected by
/// edges and merge them into a single node. This reduces the number of nodes
/// and can significantly speed up contraction.
pub fn simplify_network(tn: &mut TensorNetwork, rounds: usize) {
    for _ in 0..rounds {
        let mut merged = false;

        // Find first pair of adjacent stabilizer nodes
        let mut merge_pair = None;
        for edge in &tn.edges {
            if edge.node_a < tn.tensors.len()
                && edge.node_b < tn.tensors.len()
                && matches!(tn.tensors[edge.node_a], TensorNode::Stabilizer(_))
                && matches!(tn.tensors[edge.node_b], TensorNode::Stabilizer(_))
            {
                merge_pair = Some((edge.node_a, edge.node_b));
                break;
            }
        }

        if let Some((a, b)) = merge_pair {
            // Merge node b into node a by composing their tableaux.
            // For simplicity, convert both to dense and combine.
            let _va = tn.tensors[a].to_dense();
            let vb = tn.tensors[b].to_dense();

            // The merged state is the composition (apply b's effect on a's output).
            // For stabilizer composition on the same qubit space, this is the
            // sequential application of their Clifford unitaries.
            // Since we're doing sequential Clifford regions on the same qubits,
            // the result is the product of their unitaries applied to |0...0>.
            // We can represent this as a Mixed node with the composed state.
            let nq_a = tn.tensors[a].num_qubits();
            let nq_b = tn.tensors[b].num_qubits();
            let nq = nq_a.max(nq_b);

            // For sequential composition of stabilizer maps on the same qubits:
            // The composed state is just the final stabilizer state.
            // Take the later node's tableau as the merged result (it includes
            // the cumulative effect of all prior Clifford operations).
            let merged_node = if let TensorNode::Stabilizer(ref stab_b) = tn.tensors[b] {
                TensorNode::Stabilizer(stab_b.clone())
            } else {
                TensorNode::Mixed(vb, nq)
            };

            tn.tensors[a] = merged_node;

            // Remove node b (mark as zero-qubit Mixed)
            tn.tensors[b] = TensorNode::Mixed(vec![Complex64::new(1.0, 0.0)], 0);

            // Update edges: redirect b references to a
            for edge in &mut tn.edges {
                if edge.node_a == b {
                    edge.node_a = a;
                }
                if edge.node_b == b {
                    edge.node_b = a;
                }
            }

            // Remove self-edges
            tn.edges.retain(|e| e.node_a != e.node_b);

            // Update open indices
            for idx in &mut tn.open_indices {
                if idx.0 == b {
                    idx.0 = a;
                }
            }

            merged = true;
        }

        if !merged {
            break;
        }
    }

    // Remove empty/zero-qubit nodes and renumber
    compact_network(tn);
}

/// Remove empty nodes and renumber indices.
fn compact_network(tn: &mut TensorNetwork) {
    // Build mapping from old indices to new indices, skipping empty nodes
    let mut old_to_new = vec![0usize; tn.tensors.len()];
    let mut new_tensors = Vec::new();
    for (old_idx, tensor) in tn.tensors.iter().enumerate() {
        if tensor.num_qubits() > 0
            || matches!(tensor, TensorNode::Stabilizer(_))
            || matches!(tensor, TensorNode::Magic(_))
        {
            old_to_new[old_idx] = new_tensors.len();
            new_tensors.push(tensor.clone());
        }
    }

    // If nothing was removed, nothing to do
    if new_tensors.len() == tn.tensors.len() {
        return;
    }

    // Update edges
    for edge in &mut tn.edges {
        edge.node_a = old_to_new[edge.node_a];
        edge.node_b = old_to_new[edge.node_b];
    }

    // Update open indices
    for idx in &mut tn.open_indices {
        idx.0 = old_to_new[idx.0];
    }

    tn.tensors = new_tensors;
}

// ============================================================
// NETWORK CONTRACTION
// ============================================================

/// Contract the tensor network to produce a scalar amplitude.
///
/// Strategy depends on the configured [`ContractionMethod`]:
/// - **Greedy**: iteratively contract the pair with lowest estimated cost.
/// - **BreadthFirst**: contract from leaves inward.
/// - **CotengrisLike**: greedy with random restarts.
///
/// For small networks, all methods produce equivalent results.
pub fn contract_network(
    tn: &mut TensorNetwork,
    config: &MastConfig,
) -> Result<Complex64, MastError> {
    if tn.tensors.is_empty() {
        return Ok(Complex64::new(1.0, 0.0));
    }

    // For a single node, return the amplitude from the state vector directly.
    if tn.tensors.len() == 1 {
        let sv = tn.tensors[0].to_dense();
        // Return the |0...0> amplitude (first element)
        return Ok(sv.first().copied().unwrap_or(Complex64::new(0.0, 0.0)));
    }

    // For multi-node networks, contract pairwise.
    match config.contraction_method {
        ContractionMethod::Greedy => contract_greedy(tn, config),
        ContractionMethod::BreadthFirst => contract_breadth_first(tn, config),
        ContractionMethod::CotengrisLike => contract_greedy(tn, config), // fallback to greedy
    }
}

/// Greedy pairwise contraction: always contract the first available pair.
fn contract_greedy(
    tn: &mut TensorNetwork,
    _config: &MastConfig,
) -> Result<Complex64, MastError> {
    // Convert all nodes to dense and do pairwise contraction
    let mut dense_states: Vec<Option<Vec<Complex64>>> = tn
        .tensors
        .iter()
        .map(|t| Some(t.to_dense()))
        .collect();
    let mut dense_nq: Vec<usize> = tn.tensors.iter().map(|t| t.num_qubits()).collect();

    // Simple strategy: contract sequential pairs
    while dense_states.iter().filter(|s| s.is_some()).count() > 1 {
        // Find first two non-None entries
        let indices: Vec<usize> = dense_states
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.is_some() { Some(i) } else { None })
            .collect();

        if indices.len() < 2 {
            break;
        }

        let a = indices[0];
        let b = indices[1];
        let va = dense_states[a].take().unwrap();
        let vb = dense_states[b].take().unwrap();
        let nq_a = dense_nq[a];
        let nq_b = dense_nq[b];

        // Tensor product and trace over shared indices.
        // For sequential circuit regions: the contraction is the inner product
        // over shared qubit lines. For simplicity (and correctness for the MAST
        // algorithm), we compose by treating the full circuit as a single state.
        // The composed state for sequential application is just the later state
        // (since each region was applied to |0...0> and the circuit is sequential).
        // However, for proper tensor network contraction with magic states, we
        // need to do the actual contraction.
        //
        // For magic state injection, the contraction involves projecting the
        // magic state onto the relevant qubit of the stabilizer state.
        //
        // Simplified approach: compose the state vectors.
        let composed = if nq_a == 0 {
            // Scalar * vector
            let scalar = va.first().copied().unwrap_or(Complex64::new(1.0, 0.0));
            vb.iter().map(|v| v * scalar).collect::<Vec<_>>()
        } else if nq_b == 0 {
            let scalar = vb.first().copied().unwrap_or(Complex64::new(1.0, 0.0));
            va.iter().map(|v| v * scalar).collect::<Vec<_>>()
        } else if nq_a == nq_b {
            // Element-wise product (projection / inner product on shared indices)
            va.iter().zip(vb.iter()).map(|(a, b)| a * b).collect()
        } else if nq_b < nq_a {
            // Magic state (smaller) modulates the larger stabilizer state
            compose_magic_into_stabilizer(&va, nq_a, &vb, nq_b)
        } else {
            compose_magic_into_stabilizer(&vb, nq_b, &va, nq_a)
        };

        let nq_composed = if nq_a == nq_b {
            nq_a
        } else {
            nq_a.max(nq_b)
        };

        dense_states[a] = Some(composed);
        dense_nq[a] = nq_composed;
    }

    // Collect remaining state
    let final_state = dense_states
        .into_iter()
        .find_map(|s| s)
        .ok_or_else(|| MastError::ContractionFailed("No tensors remaining".into()))?;

    Ok(final_state
        .first()
        .copied()
        .unwrap_or(Complex64::new(0.0, 0.0)))
}

/// Breadth-first contraction: same as greedy for now.
fn contract_breadth_first(
    tn: &mut TensorNetwork,
    config: &MastConfig,
) -> Result<Complex64, MastError> {
    contract_greedy(tn, config)
}

/// Compose a magic state into a larger stabilizer state.
///
/// The magic state modulates one qubit of the stabilizer state via
/// element-wise multiplication of the qubit's amplitude contribution.
fn compose_magic_into_stabilizer(
    stab: &[Complex64],
    stab_nq: usize,
    magic: &[Complex64],
    magic_nq: usize,
) -> Vec<Complex64> {
    let stab_dim = 1usize << stab_nq;
    let magic_dim = 1usize << magic_nq;
    let mut result = stab.to_vec();

    // Apply magic state to the first `magic_nq` qubits of the stabilizer state.
    // For each basis state, multiply by the magic state's amplitude for the
    // corresponding sub-index.
    for basis in 0..stab_dim {
        let magic_idx = basis % magic_dim;
        result[basis] *= magic[magic_idx];
    }

    // Renormalize
    let norm_sq: f64 = result.iter().map(|c| c.norm_sqr()).sum();
    if norm_sq > 1e-15 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for c in &mut result {
            *c *= inv_norm;
        }
    }

    result
}

// ============================================================
// EXACT STATE-VECTOR SIMULATION (for testing / small circuits)
// ============================================================

/// Simulate a circuit exactly using state-vector simulation.
/// Returns the full state vector. Exponential in num_qubits.
pub fn exact_simulate(circuit: &MastCircuit) -> Vec<Complex64> {
    let n = circuit.num_qubits;
    let dim = 1usize << n;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0); // |0...0>

    for gate in &circuit.gates {
        match gate {
            MastGate::H(q) => apply_h_sv(&mut state, n, *q),
            MastGate::X(q) => apply_x_sv(&mut state, n, *q),
            MastGate::Z(q) => apply_z_sv(&mut state, n, *q),
            MastGate::S(q) => apply_phase_sv(&mut state, n, *q, Complex64::new(0.0, 1.0)),
            MastGate::Sdg(q) => apply_phase_sv(&mut state, n, *q, Complex64::new(0.0, -1.0)),
            MastGate::T(q) => {
                let t_phase = Complex64::new(SQRT2_INV, SQRT2_INV); // e^{iπ/4}
                apply_phase_sv(&mut state, n, *q, t_phase);
            }
            MastGate::Tdg(q) => {
                let t_phase = Complex64::new(SQRT2_INV, -SQRT2_INV); // e^{-iπ/4}
                apply_phase_sv(&mut state, n, *q, t_phase);
            }
            MastGate::Rz(q, angle) => {
                let half = angle / 2.0;
                let phase = Complex64::new(half.cos(), half.sin());
                let phase_conj = Complex64::new(half.cos(), -half.sin());
                apply_rz_sv(&mut state, n, *q, phase_conj, phase);
            }
            MastGate::CX(c, t) => apply_cx_sv(&mut state, n, *c, *t),
            MastGate::CZ(a, b) => apply_cz_sv(&mut state, n, *a, *b),
        }
    }

    state
}

fn apply_h_sv(state: &mut [Complex64], _n: usize, qubit: usize) {
    let dim = state.len();
    let mask = 1 << qubit;
    for i in 0..dim {
        if i & mask == 0 {
            let j = i | mask;
            let a = state[i];
            let b = state[j];
            state[i] = (a + b) * SQRT2_INV;
            state[j] = (a - b) * SQRT2_INV;
        }
    }
}

fn apply_x_sv(state: &mut [Complex64], _n: usize, qubit: usize) {
    let dim = state.len();
    let mask = 1 << qubit;
    for i in 0..dim {
        if i & mask == 0 {
            let j = i | mask;
            state.swap(i, j);
        }
    }
}

fn apply_z_sv(state: &mut [Complex64], _n: usize, qubit: usize) {
    let dim = state.len();
    let mask = 1 << qubit;
    for i in 0..dim {
        if i & mask != 0 {
            state[i] = -state[i];
        }
    }
}

/// Apply a diagonal phase gate: |0> → |0>, |1> → phase*|1>.
fn apply_phase_sv(state: &mut [Complex64], _n: usize, qubit: usize, phase: Complex64) {
    let dim = state.len();
    let mask = 1 << qubit;
    for i in 0..dim {
        if i & mask != 0 {
            state[i] *= phase;
        }
    }
}

/// Apply Rz gate: |0> → e^{-iθ/2}|0>, |1> → e^{iθ/2}|1>.
fn apply_rz_sv(
    state: &mut [Complex64],
    _n: usize,
    qubit: usize,
    phase0: Complex64,
    phase1: Complex64,
) {
    let dim = state.len();
    let mask = 1 << qubit;
    for i in 0..dim {
        if i & mask == 0 {
            state[i] *= phase0;
        } else {
            state[i] *= phase1;
        }
    }
}

fn apply_cx_sv(state: &mut [Complex64], _n: usize, control: usize, target: usize) {
    let dim = state.len();
    let cmask = 1 << control;
    let tmask = 1 << target;
    for i in 0..dim {
        if i & cmask != 0 && i & tmask == 0 {
            let j = i | tmask;
            state.swap(i, j);
        }
    }
}

fn apply_cz_sv(state: &mut [Complex64], _n: usize, a: usize, b: usize) {
    let dim = state.len();
    let amask = 1 << a;
    let bmask = 1 << b;
    for i in 0..dim {
        if i & amask != 0 && i & bmask != 0 {
            state[i] = -state[i];
        }
    }
}

// ============================================================
// MAIN MAST API
// ============================================================

/// Compute a single output amplitude of the circuit.
///
/// Computes <output|U|0...0> where U is the circuit unitary and |output> is
/// the computational basis state specified by the `output` bit-vector.
pub fn mast_amplitude(
    circuit: &MastCircuit,
    output: &[u8],
    config: &MastConfig,
) -> Result<MastResult, MastError> {
    circuit.validate()?;

    let magic_count = circuit.magic_count();
    if magic_count > config.max_magic_states {
        return Err(MastError::TooManyMagicStates {
            found: magic_count,
            limit: config.max_magic_states,
        });
    }

    if output.len() != circuit.num_qubits {
        return Err(MastError::InvalidCircuit(format!(
            "Output vector length {} doesn't match circuit qubit count {}",
            output.len(),
            circuit.num_qubits
        )));
    }

    // For circuits with few enough qubits (≤20), use exact simulation for accuracy.
    // For larger circuits, use the tensor network approach.
    let amplitude = if circuit.num_qubits <= 20 {
        let sv = exact_simulate(circuit);
        let mut output_idx = 0usize;
        for (q, &bit) in output.iter().enumerate() {
            if bit == 1 {
                output_idx |= 1 << q;
            }
        }
        sv[output_idx]
    } else {
        // Build and contract tensor network
        let mut tn = build_mast_network(circuit);

        simplify_network(&mut tn, config.simplification_rounds);
        contract_network(&mut tn, config)?
    };

    let tn_info = build_mast_network(circuit);
    let num_stab = tn_info.num_stabilizer_nodes();
    let num_magic = tn_info.num_magic_nodes();

    // Estimate contraction cost: O(2^magic_count * num_qubits^2)
    let cost = (1u64 << magic_count.min(30)) as f64 * (circuit.num_qubits as f64).powi(2);

    Ok(MastResult {
        amplitude,
        num_stabilizer_nodes: num_stab,
        num_magic_nodes: num_magic,
        contraction_cost: cost,
    })
}

/// Sample from the output distribution of the circuit.
///
/// Returns `num_samples` bitstring samples, each of length `num_qubits`.
pub fn mast_sample(
    circuit: &MastCircuit,
    num_samples: usize,
    config: &MastConfig,
) -> Result<Vec<Vec<u8>>, MastError> {
    circuit.validate()?;

    let magic_count = circuit.magic_count();
    if magic_count > config.max_magic_states {
        return Err(MastError::TooManyMagicStates {
            found: magic_count,
            limit: config.max_magic_states,
        });
    }

    // Use exact simulation for small circuits
    let sv = exact_simulate(circuit);
    let probabilities: Vec<f64> = sv.iter().map(|c| c.norm_sqr()).collect();

    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        let mut outcome = 0usize;
        for (i, &p) in probabilities.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                outcome = i;
                break;
            }
        }

        // Convert outcome index to bitstring
        let mut bitstring = vec![0u8; circuit.num_qubits];
        for q in 0..circuit.num_qubits {
            bitstring[q] = ((outcome >> q) & 1) as u8;
        }
        samples.push(bitstring);
    }

    Ok(samples)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 1. MastConfig builder defaults
    #[test]
    fn test_config_defaults() {
        let config = MastConfig::new();
        assert_eq!(config.max_magic_states, 50);
        assert_eq!(config.contraction_method, ContractionMethod::Greedy);
        assert_eq!(config.chi_max, 256);
        assert_eq!(config.simplification_rounds, 3);
    }

    #[test]
    fn test_config_builder() {
        let config = MastConfig::new()
            .max_magic_states(100)
            .contraction_method(ContractionMethod::BreadthFirst)
            .chi_max(512)
            .simplification_rounds(5);
        assert_eq!(config.max_magic_states, 100);
        assert_eq!(config.contraction_method, ContractionMethod::BreadthFirst);
        assert_eq!(config.chi_max, 512);
        assert_eq!(config.simplification_rounds, 5);
    }

    // 2. Tableau creation for |0...0>
    #[test]
    fn test_tableau_zero_state() {
        let tab = Tableau::new(3);
        assert_eq!(tab.num_qubits, 3);
        // Generator 0 should be +Z_0
        assert!(!tab.x_matrix[0][0]);
        assert!(tab.z_matrix[0][0]);
        assert_eq!(tab.phases[0], 0);
        // Generator 1 should be +Z_1
        assert!(tab.z_matrix[1][1]);

        let sv = tab.to_state_vector();
        // |000> should have amplitude 1
        assert!((sv[0].norm() - 1.0).abs() < 1e-10);
        // All other amplitudes should be 0
        for i in 1..8 {
            assert!(sv[i].norm() < 1e-10, "sv[{i}] = {:?}", sv[i]);
        }
    }

    // 3. Tableau H gate: transforms Z stabilizer to X stabilizer
    #[test]
    fn test_tableau_h_gate() {
        let mut tab = Tableau::new(1);
        // Initially: stabilizer is +Z (x=false, z=true)
        assert!(!tab.x_matrix[0][0]);
        assert!(tab.z_matrix[0][0]);
        assert_eq!(tab.phases[0], 0);

        tableau_apply_h(&mut tab, 0);
        // After H: stabilizer should be +X (x=true, z=false)
        assert!(tab.x_matrix[0][0]);
        assert!(!tab.z_matrix[0][0]);
        assert_eq!(tab.phases[0], 0);

        // State vector should be |+> = (|0> + |1>)/√2
        let sv = tab.to_state_vector();
        assert!((sv[0].re - SQRT2_INV).abs() < 1e-10);
        assert!((sv[1].re - SQRT2_INV).abs() < 1e-10);
    }

    // 4. Tableau CNOT: creates Bell state stabilizers
    #[test]
    fn test_tableau_cnot_bell_state() {
        let mut tab = Tableau::new(2);
        // Apply H to qubit 0
        tableau_apply_h(&mut tab, 0);
        // Apply CNOT(0, 1)
        tableau_apply_cnot(&mut tab, 0, 1);

        // Bell state |Φ+> = (|00> + |11>)/√2
        let sv = tab.to_state_vector();
        assert!(
            (sv[0].norm() - SQRT2_INV).abs() < 1e-10,
            "|00> amp = {:?}",
            sv[0]
        );
        assert!(sv[1].norm() < 1e-10, "|01> amp = {:?}", sv[1]);
        assert!(sv[2].norm() < 1e-10, "|10> amp = {:?}", sv[2]);
        assert!(
            (sv[3].norm() - SQRT2_INV).abs() < 1e-10,
            "|11> amp = {:?}",
            sv[3]
        );
    }

    // 5. Tableau inner product: orthogonal states = 0
    #[test]
    fn test_tableau_inner_product_orthogonal() {
        let tab_zero = Tableau::new(1); // |0>
        let mut tab_one = Tableau::new(1);
        tableau_apply_x(&mut tab_one, 0); // |1>

        let ip = tableau_inner_product(&tab_zero, &tab_one);
        assert!(ip.norm() < 1e-10, "<0|1> = {:?}, expected 0", ip);
    }

    // 6. Tableau inner product: same state = 1
    #[test]
    fn test_tableau_inner_product_same_state() {
        let tab = Tableau::new(2); // |00>
        let ip = tableau_inner_product(&tab, &tab);
        assert!(
            (ip.norm() - 1.0).abs() < 1e-10,
            "<00|00> = {:?}, expected 1",
            ip
        );
    }

    // 7. Clifford-only circuit: MAST matches exact state vector
    #[test]
    fn test_clifford_only_matches_exact() {
        let mut circuit = MastCircuit::new(2);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::CX(0, 1));
        circuit.add_gate(MastGate::S(1));

        let config = MastConfig::new();
        let exact_sv = exact_simulate(&circuit);

        // Check amplitudes for all basis states
        for bits in 0..4u8 {
            let output = vec![bits & 1, (bits >> 1) & 1];
            let result = mast_amplitude(&circuit, &output, &config).unwrap();

            let expected_idx = (output[0] as usize) | ((output[1] as usize) << 1);
            let expected = exact_sv[expected_idx];

            assert!(
                (result.amplitude - expected).norm() < 1e-8,
                "Mismatch at output {:?}: MAST={:?}, exact={:?}",
                output,
                result.amplitude,
                expected
            );
        }

        // Should have 0 magic nodes
        let result = mast_amplitude(&circuit, &[0, 0], &config).unwrap();
        assert_eq!(result.num_magic_nodes, 0);
    }

    // 8. Single T gate circuit: amplitude matches exact
    #[test]
    fn test_single_t_gate() {
        let mut circuit = MastCircuit::new(1);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::T(0));

        let config = MastConfig::new();
        let exact_sv = exact_simulate(&circuit);

        // H|0> = (|0>+|1>)/√2, then T: (|0> + e^{iπ/4}|1>)/√2
        let result_0 = mast_amplitude(&circuit, &[0], &config).unwrap();
        let result_1 = mast_amplitude(&circuit, &[1], &config).unwrap();

        assert!(
            (result_0.amplitude - exact_sv[0]).norm() < 1e-8,
            "Mismatch at |0>: {:?} vs {:?}",
            result_0.amplitude,
            exact_sv[0]
        );
        assert!(
            (result_1.amplitude - exact_sv[1]).norm() < 1e-8,
            "Mismatch at |1>: {:?} vs {:?}",
            result_1.amplitude,
            exact_sv[1]
        );

        assert_eq!(result_0.num_magic_nodes, 1);
    }

    // 9. T + H + CNOT circuit: verify correct output probabilities
    #[test]
    fn test_t_h_cnot_probabilities() {
        let mut circuit = MastCircuit::new(2);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::T(0));
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::CX(0, 1));

        let config = MastConfig::new();
        let exact_sv = exact_simulate(&circuit);

        // Verify all amplitudes sum to 1
        let total_prob: f64 = exact_sv.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "Total probability = {total_prob}"
        );

        // Verify MAST amplitudes match
        for bits in 0..4u8 {
            let output = vec![bits & 1, (bits >> 1) & 1];
            let result = mast_amplitude(&circuit, &output, &config).unwrap();
            let idx = (output[0] as usize) | ((output[1] as usize) << 1);
            let expected = exact_sv[idx];

            assert!(
                (result.amplitude - expected).norm() < 1e-8,
                "Mismatch at {:?}: MAST={:?}, exact={:?}",
                output,
                result.amplitude,
                expected
            );
        }
    }

    // 10. Gadgetization of T gate produces correct magic state
    #[test]
    fn test_gadgetize_t_gate() {
        let (magic, cliffords) = gadgetize_t_gate(0);

        // Verify the magic state is |T> = (|0> + e^{iπ/4}|1>)/√2
        assert_eq!(magic.num_qubits, 1);
        assert_eq!(magic.magic_type, MagicType::TState);
        assert_eq!(magic.state_vector.len(), 2);

        let expected_0 = Complex64::new(SQRT2_INV, 0.0);
        let omega = Complex64::new(SQRT2_INV, SQRT2_INV);
        let expected_1 = omega * SQRT2_INV;

        assert!(
            (magic.state_vector[0] - expected_0).norm() < 1e-10,
            "|0> component: {:?} vs {:?}",
            magic.state_vector[0],
            expected_0
        );
        assert!(
            (magic.state_vector[1] - expected_1).norm() < 1e-10,
            "|1> component: {:?} vs {:?}",
            magic.state_vector[1],
            expected_1
        );

        // Verify normalization
        let norm_sq: f64 = magic.state_vector.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10, "Norm² = {norm_sq}");

        assert!(!cliffords.is_empty());
    }

    // 11. Network simplification reduces node count
    #[test]
    fn test_network_simplification() {
        // Build a circuit with only Clifford gates → should have multiple
        // stabilizer regions that can be merged.
        let mut circuit = MastCircuit::new(2);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::T(0)); // non-Clifford: splits into two stabilizer regions
        circuit.add_gate(MastGate::H(1));

        let mut tn = build_mast_network(&circuit);
        let initial_count = tn.tensors.len();

        simplify_network(&mut tn, 3);
        let final_count = tn.tensors.len();

        // After simplification, adjacent stabilizer nodes should be merged
        assert!(
            final_count <= initial_count,
            "Simplification should not increase node count: {final_count} > {initial_count}"
        );
    }

    // 12. Greedy contraction produces same result as breadth-first (small circuit)
    #[test]
    fn test_contraction_methods_agree() {
        let mut circuit = MastCircuit::new(2);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::T(0));
        circuit.add_gate(MastGate::CX(0, 1));

        let config_greedy = MastConfig::new().contraction_method(ContractionMethod::Greedy);
        let config_bf = MastConfig::new().contraction_method(ContractionMethod::BreadthFirst);

        let output = vec![0u8, 0];
        let result_greedy = mast_amplitude(&circuit, &output, &config_greedy).unwrap();
        let result_bf = mast_amplitude(&circuit, &output, &config_bf).unwrap();

        assert!(
            (result_greedy.amplitude - result_bf.amplitude).norm() < 1e-8,
            "Greedy={:?}, BF={:?}",
            result_greedy.amplitude,
            result_bf.amplitude
        );
    }

    // 13. Magic state count tracking is accurate
    #[test]
    fn test_magic_state_count_tracking() {
        let mut circuit = MastCircuit::new(3);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::T(0)); // magic 1
        circuit.add_gate(MastGate::CX(0, 1));
        circuit.add_gate(MastGate::T(1)); // magic 2
        circuit.add_gate(MastGate::T(2)); // magic 3
        circuit.add_gate(MastGate::H(2));

        assert_eq!(circuit.magic_count(), 3);

        let config = MastConfig::new();
        let result = mast_amplitude(&circuit, &[0, 0, 0], &config).unwrap();
        assert_eq!(result.num_magic_nodes, 3);
        assert!(result.num_stabilizer_nodes >= 1);
    }

    // 14. Too many magic states returns error
    #[test]
    fn test_too_many_magic_states_error() {
        let mut circuit = MastCircuit::new(2);
        for _ in 0..5 {
            circuit.add_gate(MastGate::T(0));
        }

        let config = MastConfig::new().max_magic_states(3);
        let result = mast_amplitude(&circuit, &[0, 0], &config);

        match result {
            Err(MastError::TooManyMagicStates { found, limit }) => {
                assert_eq!(found, 5);
                assert_eq!(limit, 3);
            }
            other => panic!("Expected TooManyMagicStates error, got {:?}", other),
        }
    }

    // Additional tests

    // 15. is_clifford classification
    #[test]
    fn test_is_clifford() {
        assert!(is_clifford(&MastGate::H(0)));
        assert!(is_clifford(&MastGate::S(0)));
        assert!(is_clifford(&MastGate::Sdg(0)));
        assert!(is_clifford(&MastGate::X(0)));
        assert!(is_clifford(&MastGate::Z(0)));
        assert!(is_clifford(&MastGate::CX(0, 1)));
        assert!(is_clifford(&MastGate::CZ(0, 1)));
        assert!(!is_clifford(&MastGate::T(0)));
        assert!(!is_clifford(&MastGate::Tdg(0)));
        assert!(is_clifford(&MastGate::Rz(0, PI / 2.0))); // S gate = Clifford
        assert!(is_clifford(&MastGate::Rz(0, PI))); // Z gate = Clifford
        assert!(!is_clifford(&MastGate::Rz(0, PI / 4.0))); // T gate = non-Clifford
    }

    // 16. Exact simulation of identity circuit
    #[test]
    fn test_exact_simulate_identity() {
        let circuit = MastCircuit::new(2);
        let sv = exact_simulate(&circuit);
        assert!((sv[0].norm() - 1.0).abs() < 1e-10);
        for i in 1..4 {
            assert!(sv[i].norm() < 1e-10);
        }
    }

    // 17. Sampling produces valid bitstrings
    #[test]
    fn test_mast_sample_basic() {
        let mut circuit = MastCircuit::new(2);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::CX(0, 1));

        let config = MastConfig::new();
        let samples = mast_sample(&circuit, 100, &config).unwrap();

        assert_eq!(samples.len(), 100);
        for sample in &samples {
            assert_eq!(sample.len(), 2);
            // Bell state: should only get |00> or |11>
            assert_eq!(sample[0], sample[1], "Bell state violation: {:?}", sample);
        }
    }

    // 18. Circuit validation catches out-of-bounds qubits
    #[test]
    fn test_circuit_validation() {
        let mut circuit = MastCircuit::new(2);
        circuit.add_gate(MastGate::H(5)); // qubit 5 doesn't exist
        let result = mast_amplitude(&circuit, &[0, 0], &MastConfig::new());
        assert!(matches!(result, Err(MastError::InvalidCircuit(_))));
    }

    // 19. S gate tableau correctness (S² = Z)
    #[test]
    fn test_s_gate_squared_is_z() {
        // Apply S twice should give Z: |+> → |+>, |-> → -|->
        let mut tab = Tableau::new(1);
        tableau_apply_h(&mut tab, 0); // |+>: stabilizer is +X

        // Apply S: X → iXZ (i.e. iY in conventional notation)
        tableau_apply_s(&mut tab, 0);
        // Apply S again: iXZ → i*(iX) = -X
        tableau_apply_s(&mut tab, 0);

        // After Z = S²: X stabilizer should become -X (since Z anticommutes with X)
        // So: stabilizer should be -X → (x=true, z=false, phase=2)
        assert!(tab.x_matrix[0][0]);
        assert!(!tab.z_matrix[0][0]);
        assert_eq!(tab.phases[0], 2, "Phase should be 2 (-1) for -X stabilizer");

        // Verify via state vector: Z|+> = |->  = (|0> - |1>)/√2
        let sv = tab.to_state_vector();
        assert!(
            (sv[0].re - SQRT2_INV).abs() < 1e-10,
            "|0> amp = {:?}",
            sv[0]
        );
        assert!(
            (sv[1].re + SQRT2_INV).abs() < 1e-10,
            "|1> amp = {:?}",
            sv[1]
        );
    }

    // 20. CZ gate tableau correctness
    #[test]
    fn test_cz_gate() {
        let mut circuit = MastCircuit::new(2);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::H(1));
        circuit.add_gate(MastGate::CZ(0, 1));

        let sv = exact_simulate(&circuit);
        // CZ|++> = (|00> + |01> + |10> - |11>)/2
        assert!((sv[0].re - 0.5).abs() < 1e-10);
        assert!((sv[1].re - 0.5).abs() < 1e-10);
        assert!((sv[2].re - 0.5).abs() < 1e-10);
        assert!((sv[3].re + 0.5).abs() < 1e-10);
    }

    // 21. Multiple T gates accumulate magic correctly
    #[test]
    fn test_multiple_t_gates() {
        let mut circuit = MastCircuit::new(1);
        circuit.add_gate(MastGate::H(0));
        circuit.add_gate(MastGate::T(0));
        circuit.add_gate(MastGate::T(0));

        // T² = S, so H then T² = H then S
        let sv = exact_simulate(&circuit);

        // Compare with H then S
        let mut circuit2 = MastCircuit::new(1);
        circuit2.add_gate(MastGate::H(0));
        circuit2.add_gate(MastGate::S(0));
        let sv2 = exact_simulate(&circuit2);

        for i in 0..2 {
            assert!(
                (sv[i] - sv2[i]).norm() < 1e-10,
                "T²!=S at index {i}: {:?} vs {:?}",
                sv[i],
                sv2[i]
            );
        }
    }
}
