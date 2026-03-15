//! CAMPS: Clifford-Augmented Matrix Product States
//!
//! Implements the CAMPS algorithm from arXiv:2412.17209, which represents
//! quantum states as a Clifford tableau combined with a low-bond-dimension
//! Matrix Product State (MPS). Clifford gates are absorbed into the tableau
//! in O(n^2) time, while only non-Clifford gates (T, Rz, Rx, Ry) grow the
//! MPS bond dimension. This gives exponential savings for circuits with few
//! non-Clifford gates.
//!
//! # Architecture
//!
//! - [`CliffordTableau`]: Binary symplectic representation with 2n generators
//!   (n stabilizers + n destabilizers), following Aaronson-Gottesman.
//! - [`MpsResidual`]: Compact MPS for the non-Clifford residual state.
//! - [`CampsState`]: Combined Clifford+MPS state with automatic gate dispatch.
//! - [`CampsConfig`]: Configuration with builder pattern.
//! - [`GateKind`]: Gate enumeration with Clifford/non-Clifford classification.
//!
//! # Key Physics
//!
//! Clifford gates (H, S, X, Y, Z, CNOT, CZ, SWAP) map Pauli operators to Pauli
//! operators and are tracked exactly in the O(n^2) tableau. Non-Clifford gates
//! (T, Rz, Rx, Ry for generic angles) require updating the MPS, whose bond
//! dimension grows only with the number of non-Clifford gates, not with the
//! number of qubits.
//!
//! # References
//!
//! - Begusic & Chan, "CAMPS: Clifford-Augmented MPS", arXiv:2412.17209 (2024)
//! - Aaronson & Gottesman, "Improved simulation of stabilizer circuits",
//!   Phys. Rev. A 70, 052328 (2004)
//! - Vidal, "Efficient classical simulation of slightly entangled quantum
//!   computations", Phys. Rev. Lett. 91, 147902 (2003)

use crate::{c64_one, c64_scale, c64_zero, C64};
use ndarray::Array3;
use std::fmt;

// ============================================================
// CONSTANTS
// ============================================================

const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;
const FRAC_PI_4: f64 = std::f64::consts::FRAC_PI_4;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising during CAMPS simulation.
#[derive(Debug, Clone)]
pub enum CampsError {
    /// Qubit index exceeds system size.
    QubitOutOfBounds { qubit: usize, num_qubits: usize },
    /// SVD decomposition failed during MPS truncation.
    SvdFailed(String),
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Gate application failed.
    GateFailed(String),
    /// MPS bond dimension exceeded the configured maximum.
    BondDimOverflow { current: usize, max: usize },
    /// MPS tensor contraction or reshape error.
    MpsContractError(String),
    /// Error during measurement computation.
    MeasurementError(String),
}

impl fmt::Display for CampsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QubitOutOfBounds { qubit, num_qubits } => {
                write!(f, "Qubit {} out of bounds (n={})", qubit, num_qubits)
            }
            Self::SvdFailed(msg) => write!(f, "SVD failed: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid CAMPS config: {}", msg),
            Self::GateFailed(msg) => write!(f, "Gate application failed: {}", msg),
            Self::BondDimOverflow { current, max } => {
                write!(
                    f,
                    "MPS bond dimension overflow: {} exceeds max {}",
                    current, max
                )
            }
            Self::MpsContractError(msg) => write!(f, "MPS contraction error: {}", msg),
            Self::MeasurementError(msg) => write!(f, "Measurement error: {}", msg),
        }
    }
}

impl std::error::Error for CampsError {}

// ============================================================
// GATE KIND
// ============================================================

/// Enumeration of quantum gate types supported by CAMPS.
#[derive(Debug, Clone, PartialEq)]
pub enum GateKind {
    /// Hadamard gate (Clifford).
    H,
    /// S gate = sqrt(Z) (Clifford).
    S,
    /// S-dagger gate (Clifford).
    Sdg,
    /// Pauli X gate (Clifford).
    X,
    /// Pauli Y gate (Clifford).
    Y,
    /// Pauli Z gate (Clifford).
    Z,
    /// Controlled-NOT gate (Clifford). Qubits: [control, target].
    Cx,
    /// Controlled-Z gate (Clifford). Qubits: [q1, q2].
    Cz,
    /// SWAP gate (Clifford). Qubits: [q1, q2].
    Swap,
    /// T gate = diag(1, e^{i pi/4}) (non-Clifford).
    T,
    /// T-dagger gate = diag(1, e^{-i pi/4}) (non-Clifford).
    Tdg,
    /// Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
    Rz(f64),
    /// Rx(theta) = exp(-i theta/2 X).
    Rx(f64),
    /// Ry(theta) = exp(-i theta/2 Y).
    Ry(f64),
}

impl fmt::Display for GateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateKind::H => write!(f, "H"),
            GateKind::S => write!(f, "S"),
            GateKind::Sdg => write!(f, "Sdg"),
            GateKind::X => write!(f, "X"),
            GateKind::Y => write!(f, "Y"),
            GateKind::Z => write!(f, "Z"),
            GateKind::Cx => write!(f, "CX"),
            GateKind::Cz => write!(f, "CZ"),
            GateKind::Swap => write!(f, "SWAP"),
            GateKind::T => write!(f, "T"),
            GateKind::Tdg => write!(f, "Tdg"),
            GateKind::Rz(theta) => write!(f, "Rz({:.4})", theta),
            GateKind::Rx(theta) => write!(f, "Rx({:.4})", theta),
            GateKind::Ry(theta) => write!(f, "Ry({:.4})", theta),
        }
    }
}

/// Returns true if the gate is a Clifford gate.
pub fn is_clifford(gate: &GateKind) -> bool {
    match gate {
        GateKind::H
        | GateKind::S
        | GateKind::Sdg
        | GateKind::X
        | GateKind::Y
        | GateKind::Z
        | GateKind::Cx
        | GateKind::Cz
        | GateKind::Swap => true,
        GateKind::T | GateKind::Tdg => false,
        GateKind::Rz(theta) | GateKind::Rx(theta) | GateKind::Ry(theta) => {
            is_clifford_angle(*theta)
        }
    }
}

fn is_clifford_angle(theta: f64) -> bool {
    let ratio = theta / FRAC_PI_2;
    (ratio - ratio.round()).abs() < 1e-10
}

fn clifford_quarter_turns(theta: f64) -> usize {
    let ratio = theta / FRAC_PI_2;
    let rounded = ratio.round() as i64;
    ((rounded % 4 + 4) % 4) as usize
}

// ============================================================
// CLIFFORD TABLEAU
// ============================================================

/// Binary symplectic representation of a Clifford transformation.
///
/// For n qubits, maintains 2n generators (rows 0..n are stabilizers,
/// rows n..2n are destabilizers) following Aaronson-Gottesman.
#[derive(Clone, Debug)]
pub struct CliffordTableau {
    num_qubits: usize,
    x_table: Vec<Vec<bool>>,
    z_table: Vec<Vec<bool>>,
    phases: Vec<u8>,
}

impl CliffordTableau {
    /// Create a new tableau representing the |0...0> state.
    pub fn new(num_qubits: usize) -> Self {
        let n = num_qubits;
        let mut x_table = vec![vec![false; n]; 2 * n];
        let mut z_table = vec![vec![false; n]; 2 * n];
        let phases = vec![0u8; 2 * n];

        for i in 0..n {
            z_table[i][i] = true;
        }
        for i in 0..n {
            x_table[n + i][i] = true;
        }

        CliffordTableau {
            num_qubits: n,
            x_table,
            z_table,
            phases,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Apply Hadamard gate: X -> Z, Z -> X, Y -> -Y.
    pub fn apply_h(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        for i in 0..(2 * self.num_qubits) {
            if self.x_table[i][qubit] && self.z_table[i][qubit] {
                self.phases[i] = (self.phases[i] + 2) % 4;
            }
            let tmp = self.x_table[i][qubit];
            self.x_table[i][qubit] = self.z_table[i][qubit];
            self.z_table[i][qubit] = tmp;
        }
    }

    /// Apply S gate: X -> Y, Y -> -X, Z -> Z.
    pub fn apply_s(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        for i in 0..(2 * self.num_qubits) {
            if self.x_table[i][qubit] {
                self.phases[i] = (self.phases[i] + if self.z_table[i][qubit] { 2 } else { 0 }) % 4;
                self.z_table[i][qubit] ^= true;
            }
        }
    }

    /// Apply S-dagger gate (S^3).
    pub fn apply_sdg(&mut self, qubit: usize) {
        self.apply_s(qubit);
        self.apply_s(qubit);
        self.apply_s(qubit);
    }

    /// Apply CNOT(control, target).
    pub fn apply_cx(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits, "Control qubit out of bounds");
        assert!(target < self.num_qubits, "Target qubit out of bounds");
        assert_ne!(control, target, "Control and target must differ");
        for i in 0..(2 * self.num_qubits) {
            let xc = self.x_table[i][control];
            let zc = self.z_table[i][control];
            let xt = self.x_table[i][target];
            let zt = self.z_table[i][target];
            if xc && zt && (xt ^ zc ^ true) {
                self.phases[i] = (self.phases[i] + 2) % 4;
            }
            self.x_table[i][target] = xt ^ xc;
            self.z_table[i][control] = zc ^ zt;
        }
    }

    /// Apply CZ gate.
    pub fn apply_cz(&mut self, q1: usize, q2: usize) {
        assert!(
            q1 < self.num_qubits && q2 < self.num_qubits,
            "Qubit out of bounds"
        );
        assert_ne!(q1, q2, "CZ qubits must differ");
        for i in 0..(2 * self.num_qubits) {
            let x1 = self.x_table[i][q1];
            let z1 = self.z_table[i][q1];
            let x2 = self.x_table[i][q2];
            let z2 = self.z_table[i][q2];
            if x1 && x2 && (z1 ^ z2 ^ true) {
                self.phases[i] = (self.phases[i] + 2) % 4;
            }
            self.z_table[i][q1] = z1 ^ x2;
            self.z_table[i][q2] = z2 ^ x1;
        }
    }

    /// Apply Pauli X gate.
    pub fn apply_x(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        for i in 0..(2 * self.num_qubits) {
            if self.z_table[i][qubit] {
                self.phases[i] = (self.phases[i] + 2) % 4;
            }
        }
    }

    /// Apply Pauli Y gate.
    pub fn apply_y(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        for i in 0..(2 * self.num_qubits) {
            if self.x_table[i][qubit] ^ self.z_table[i][qubit] {
                self.phases[i] = (self.phases[i] + 2) % 4;
            }
        }
    }

    /// Apply Pauli Z gate.
    pub fn apply_z(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        for i in 0..(2 * self.num_qubits) {
            if self.x_table[i][qubit] {
                self.phases[i] = (self.phases[i] + 2) % 4;
            }
        }
    }

    /// Apply SWAP gate.
    pub fn apply_swap(&mut self, q1: usize, q2: usize) {
        assert!(
            q1 < self.num_qubits && q2 < self.num_qubits,
            "Qubit out of bounds"
        );
        assert_ne!(q1, q2, "SWAP qubits must differ");
        for i in 0..(2 * self.num_qubits) {
            self.x_table[i].swap(q1, q2);
            self.z_table[i].swap(q1, q2);
        }
    }

    /// Check if generator `row` commutes with Z on the given qubit.
    pub fn commutes_with_z(&self, row: usize, qubit: usize) -> bool {
        !self.x_table[row][qubit]
    }

    /// Multiply row `target_row` by row `source_row` (Pauli product with phase).
    pub fn row_multiply(&mut self, target_row: usize, source_row: usize) {
        let n = self.num_qubits;
        let mut phase_contrib: i64 = 0;
        for j in 0..n {
            phase_contrib += pauli_product_phase(
                self.x_table[source_row][j],
                self.z_table[source_row][j],
                self.x_table[target_row][j],
                self.z_table[target_row][j],
            );
        }
        let total =
            (self.phases[source_row] as i64) + (self.phases[target_row] as i64) + phase_contrib;
        self.phases[target_row] = ((total % 4 + 4) % 4) as u8;
        for j in 0..n {
            self.x_table[target_row][j] ^= self.x_table[source_row][j];
            self.z_table[target_row][j] ^= self.z_table[source_row][j];
        }
    }

    /// Measure qubit in the Z basis.
    ///
    /// Returns `(outcome, deterministic)`.
    pub fn measure_z(&mut self, qubit: usize) -> (bool, bool) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        let n = self.num_qubits;

        let mut anticommuting = None;
        for i in 0..n {
            if !self.commutes_with_z(i, qubit) {
                anticommuting = Some(i);
                break;
            }
        }

        match anticommuting {
            Some(p) => {
                for i in 0..(2 * n) {
                    if i != p && !self.commutes_with_z(i, qubit) {
                        self.row_multiply(i, p);
                    }
                }
                let outcome = rand::random::<bool>();
                for j in 0..n {
                    self.x_table[p][j] = false;
                    self.z_table[p][j] = false;
                }
                self.z_table[p][qubit] = true;
                self.phases[p] = if outcome { 2 } else { 0 };
                (outcome, false)
            }
            None => {
                let mut combined_phase: i64 = 0;
                let mut found = false;
                for i in 0..n {
                    if self.z_table[i][qubit] && !self.x_table[i][qubit] {
                        combined_phase += self.phases[i] as i64;
                        found = true;
                    }
                }
                if found {
                    let eff = ((combined_phase % 4 + 4) % 4) as u8;
                    (eff == 2, true)
                } else {
                    (false, true)
                }
            }
        }
    }

    /// Check whether the tableau is in the identity configuration.
    pub fn is_identity(&self) -> bool {
        let n = self.num_qubits;
        for i in 0..n {
            if self.phases[i] != 0 {
                return false;
            }
            for j in 0..n {
                if self.x_table[i][j] || self.z_table[i][j] != (i == j) {
                    return false;
                }
            }
        }
        for i in 0..n {
            if self.phases[n + i] != 0 {
                return false;
            }
            for j in 0..n {
                if self.x_table[n + i][j] != (i == j) || self.z_table[n + i][j] {
                    return false;
                }
            }
        }
        true
    }
}

fn pauli_product_phase(xa: bool, za: bool, xb: bool, zb: bool) -> i64 {
    let a = pauli_index(xa, za);
    let b = pauli_index(xb, zb);
    if a == 0 || b == 0 || a == b {
        return 0;
    }
    match (a, b) {
        (1, 2) | (2, 3) | (3, 1) => 1,
        (2, 1) | (3, 2) | (1, 3) => -1,
        _ => 0,
    }
}

fn pauli_index(x: bool, z: bool) -> u8 {
    match (x, z) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

impl fmt::Display for CliffordTableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.num_qubits;
        writeln!(f, "CliffordTableau ({} qubits):", n)?;
        for i in 0..n {
            let phase_str = match self.phases[i] {
                0 => "+",
                2 => "-",
                1 => "+i",
                3 => "-i",
                _ => "?",
            };
            write!(f, "  S{}: {} ", i, phase_str)?;
            for j in 0..n {
                let p = match (self.x_table[i][j], self.z_table[i][j]) {
                    (false, false) => "I",
                    (true, false) => "X",
                    (true, true) => "Y",
                    (false, true) => "Z",
                };
                write!(f, "{}", p)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ============================================================
// MPS RESIDUAL
// ============================================================

/// Matrix Product State for the non-Clifford residual in CAMPS.
#[derive(Clone, Debug)]
pub struct MpsResidual {
    num_qubits: usize,
    tensors: Vec<Array3<C64>>,
    bond_dims: Vec<usize>,
    max_bond_dim: usize,
    svd_cutoff: f64,
}

impl MpsResidual {
    /// Create an identity (product state |0...0>) MPS.
    pub fn identity(num_qubits: usize, d_phys: usize) -> Self {
        assert!(num_qubits > 0, "Need at least 1 qubit");
        assert!(d_phys >= 2, "Physical dimension must be at least 2");
        let mut tensors = Vec::with_capacity(num_qubits);
        for _ in 0..num_qubits {
            let mut t = Array3::zeros((1, d_phys, 1));
            t[[0, 0, 0]] = c64_one();
            tensors.push(t);
        }
        MpsResidual {
            num_qubits,
            tensors,
            bond_dims: vec![1; num_qubits.saturating_sub(1)],
            max_bond_dim: 64,
            svd_cutoff: 1e-10,
        }
    }

    /// Apply a single-qubit 2x2 gate.
    pub fn apply_single_qubit(&mut self, qubit: usize, gate: &[[C64; 2]; 2]) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        let t = &self.tensors[qubit];
        let (bl, d, br) = (t.dim().0, t.dim().1, t.dim().2);
        assert_eq!(d, 2);
        let mut new_t = Array3::zeros((bl, 2, br));
        for l in 0..bl {
            for r in 0..br {
                let a0 = t[[l, 0, r]];
                let a1 = t[[l, 1, r]];
                new_t[[l, 0, r]] = gate[0][0] * a0 + gate[0][1] * a1;
                new_t[[l, 1, r]] = gate[1][0] * a0 + gate[1][1] * a1;
            }
        }
        self.tensors[qubit] = new_t;
    }

    /// Apply a two-qubit 4x4 gate to adjacent sites with SVD truncation.
    pub fn apply_two_qubit(&mut self, q1: usize, q2: usize, gate: &[[C64; 4]; 4]) {
        assert!(
            q1 < self.num_qubits && q2 < self.num_qubits,
            "Qubit out of bounds"
        );
        assert_eq!(
            (q1 as i64 - q2 as i64).unsigned_abs(),
            1,
            "Two-qubit gate requires adjacent qubits"
        );
        let (left, right) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
        let tl = &self.tensors[left];
        let tr = &self.tensors[right];
        let bl = tl.dim().0;
        let br = tr.dim().2;
        let bond = tl.dim().2;

        let mut theta = ndarray::Array4::<C64>::zeros((bl, 2, 2, br));
        for l in 0..bl {
            for r in 0..br {
                for k in 0..bond {
                    for d1 in 0..2usize {
                        for d2 in 0..2usize {
                            theta[[l, d1, d2, r]] += tl[[l, d1, k]] * tr[[k, d2, r]];
                        }
                    }
                }
            }
        }

        let mut new_theta = ndarray::Array4::<C64>::zeros((bl, 2, 2, br));
        for l in 0..bl {
            for r in 0..br {
                for d1 in 0..2usize {
                    for d2 in 0..2usize {
                        let row = d1 * 2 + d2;
                        for d1p in 0..2usize {
                            for d2p in 0..2usize {
                                new_theta[[l, d1, d2, r]] +=
                                    gate[row][d1p * 2 + d2p] * theta[[l, d1p, d2p, r]];
                            }
                        }
                    }
                }
            }
        }

        let m = bl * 2;
        let n = 2 * br;
        let mut mat = ndarray::Array2::<C64>::zeros((m, n));
        for l in 0..bl {
            for d1 in 0..2usize {
                for d2 in 0..2usize {
                    for r in 0..br {
                        mat[[l * 2 + d1, d2 * br + r]] = new_theta[[l, d1, d2, r]];
                    }
                }
            }
        }

        let (u, s_vals, vt) = thin_svd_c64(&mat);
        let mut keep = s_vals.len();
        for i in (0..s_vals.len()).rev() {
            if s_vals[i] < self.svd_cutoff {
                keep = i;
            } else {
                break;
            }
        }
        keep = keep.max(1).min(self.max_bond_dim);

        let mut new_tl = Array3::zeros((bl, 2, keep));
        for l in 0..bl {
            for d1 in 0..2usize {
                for k in 0..keep {
                    new_tl[[l, d1, k]] = u[[l * 2 + d1, k]];
                }
            }
        }
        let mut new_tr = Array3::zeros((keep, 2, br));
        for k in 0..keep {
            for d2 in 0..2usize {
                for r in 0..br {
                    new_tr[[k, d2, r]] = c64_scale(vt[[k, d2 * br + r]], s_vals[k]);
                }
            }
        }
        self.tensors[left] = new_tl;
        self.tensors[right] = new_tr;
        if left < self.bond_dims.len() {
            self.bond_dims[left] = keep;
        }
    }

    /// SVD truncation at a given site.
    pub fn svd_truncate(&mut self, site: usize) {
        if site + 1 >= self.num_qubits {
            return;
        }
        let tl = &self.tensors[site];
        let (bl, d, br) = (tl.dim().0, tl.dim().1, tl.dim().2);
        let m = bl * d;
        let mut mat = ndarray::Array2::<C64>::zeros((m, br));
        for l in 0..bl {
            for p in 0..d {
                for r in 0..br {
                    mat[[l * d + p, r]] = tl[[l, p, r]];
                }
            }
        }
        let (u, s_vals, vt) = thin_svd_c64(&mat);
        let mut keep = s_vals.len();
        for i in (0..s_vals.len()).rev() {
            if s_vals[i] < self.svd_cutoff {
                keep = i;
            } else {
                break;
            }
        }
        keep = keep.max(1).min(self.max_bond_dim);

        let mut new_tl = Array3::zeros((bl, d, keep));
        for l in 0..bl {
            for p in 0..d {
                for k in 0..keep {
                    new_tl[[l, p, k]] = u[[l * d + p, k]];
                }
            }
        }
        self.tensors[site] = new_tl;

        let tr = &self.tensors[site + 1];
        let (br_old, d2, br2) = (tr.dim().0, tr.dim().1, tr.dim().2);
        let mut sv = ndarray::Array2::<C64>::zeros((keep, br_old));
        for k in 0..keep {
            for j in 0..br_old.min(vt.dim().1) {
                sv[[k, j]] = c64_scale(vt[[k, j]], s_vals[k]);
            }
        }
        let mut new_tr = Array3::zeros((keep, d2, br2));
        for k in 0..keep {
            for p in 0..d2 {
                for r in 0..br2 {
                    for j in 0..br_old {
                        new_tr[[k, p, r]] += sv[[k, j]] * tr[[j, p, r]];
                    }
                }
            }
        }
        self.tensors[site + 1] = new_tr;
        if site < self.bond_dims.len() {
            self.bond_dims[site] = keep;
        }
    }

    pub fn total_bond_dim(&self) -> usize {
        self.bond_dims.iter().sum()
    }
    pub fn max_bond_dim_actual(&self) -> usize {
        self.bond_dims.iter().copied().max().unwrap_or(1)
    }

    /// Compute the norm of the MPS.
    pub fn norm(&self) -> f64 {
        if self.num_qubits == 0 {
            return 0.0;
        }
        let t0 = &self.tensors[0];
        let bl = t0.dim().0;
        let mut env = ndarray::Array2::<C64>::zeros((bl, bl));
        for a in 0..bl {
            env[[a, a]] = c64_one();
        }
        for site in 0..self.num_qubits {
            let t = &self.tensors[site];
            let (bl_s, d_s, br_s) = (t.dim().0, t.dim().1, t.dim().2);
            let mut new_env = ndarray::Array2::<C64>::zeros((br_s, br_s));
            for a in 0..bl_s {
                for ap in 0..bl_s {
                    let e = env[[a, ap]];
                    if e.norm() < 1e-15 {
                        continue;
                    }
                    for p in 0..d_s {
                        for b in 0..br_s {
                            for bp in 0..br_s {
                                new_env[[b, bp]] += e * t[[a, p, b]] * t[[ap, p, bp]].conj();
                            }
                        }
                    }
                }
            }
            env = new_env;
        }
        let mut norm_sq = c64_zero();
        let sz = env.dim().0;
        for a in 0..sz {
            norm_sq += env[[a, a]];
        }
        norm_sq.re.max(0.0).sqrt()
    }

    /// Compute <psi|Z_qubit|psi> for the MPS.
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");
        if self.num_qubits == 0 {
            return 0.0;
        }
        let t0 = &self.tensors[0];
        let bl = t0.dim().0;
        let mut env = ndarray::Array2::<C64>::zeros((bl, bl));
        for a in 0..bl {
            env[[a, a]] = c64_one();
        }
        for site in 0..self.num_qubits {
            let t = &self.tensors[site];
            let (bl_s, d_s, br_s) = (t.dim().0, t.dim().1, t.dim().2);
            let mut new_env = ndarray::Array2::<C64>::zeros((br_s, br_s));
            for a in 0..bl_s {
                for ap in 0..bl_s {
                    let e = env[[a, ap]];
                    if e.norm() < 1e-15 {
                        continue;
                    }
                    for p in 0..d_s {
                        let z_val: f64 = if site == qubit {
                            if p == 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        } else {
                            1.0
                        };
                        for b in 0..br_s {
                            for bp in 0..br_s {
                                new_env[[b, bp]] +=
                                    c64_scale(e * t[[a, p, b]] * t[[ap, p, bp]].conj(), z_val);
                            }
                        }
                    }
                }
            }
            env = new_env;
        }
        let mut result = c64_zero();
        let sz = env.dim().0;
        for a in 0..sz {
            result += env[[a, a]];
        }
        result.re
    }

    /// Check if the MPS is the identity product state |0...0>.
    pub fn is_identity(&self) -> bool {
        for t in &self.tensors {
            let (bl, d, br) = (t.dim().0, t.dim().1, t.dim().2);
            if bl != 1 || br != 1 {
                return false;
            }
            if (t[[0, 0, 0]] - c64_one()).norm() > 1e-10 {
                return false;
            }
            for p in 1..d {
                if t[[0, p, 0]].norm() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }
}

// ============================================================
// SVD HELPER
// ============================================================

fn thin_svd_c64(
    mat: &ndarray::Array2<C64>,
) -> (ndarray::Array2<C64>, Vec<f64>, ndarray::Array2<C64>) {
    let (m, n) = (mat.dim().0, mat.dim().1);
    let k = m.min(n);
    if k == 0 {
        return (
            ndarray::Array2::zeros((m, 0)),
            vec![],
            ndarray::Array2::zeros((0, n)),
        );
    }
    if m <= n {
        let mut a = ndarray::Array2::<C64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut sum = c64_zero();
                for l in 0..n {
                    sum += mat[[i, l]] * mat[[j, l]].conj();
                }
                a[[i, j]] = sum;
            }
        }
        let (eigenvalues, eigenvectors) = hermitian_eigen_deflation(&a);
        let mut idx: Vec<usize> = (0..m).collect();
        idx.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
        let mut sigma = Vec::with_capacity(k);
        let mut u = ndarray::Array2::<C64>::zeros((m, k));
        let mut vt = ndarray::Array2::<C64>::zeros((k, n));
        for (ki, &i) in idx.iter().take(k).enumerate() {
            let sv = eigenvalues[i].max(0.0).sqrt();
            sigma.push(sv);
            for r in 0..m {
                u[[r, ki]] = eigenvectors[[r, i]];
            }
            if sv > 1e-15 {
                for c in 0..n {
                    let mut sum = c64_zero();
                    for r in 0..m {
                        sum += eigenvectors[[r, i]].conj() * mat[[r, c]];
                    }
                    vt[[ki, c]] = c64_scale(sum, 1.0 / sv);
                }
            }
        }
        (u, sigma, vt)
    } else {
        let mut b = ndarray::Array2::<C64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = c64_zero();
                for l in 0..m {
                    sum += mat[[l, i]].conj() * mat[[l, j]];
                }
                b[[i, j]] = sum;
            }
        }
        let (eigenvalues, eigenvectors) = hermitian_eigen_deflation(&b);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
        let mut sigma = Vec::with_capacity(k);
        let mut u = ndarray::Array2::<C64>::zeros((m, k));
        let mut vt = ndarray::Array2::<C64>::zeros((k, n));
        for (ki, &i) in idx.iter().take(k).enumerate() {
            let sv = eigenvalues[i].max(0.0).sqrt();
            sigma.push(sv);
            for c in 0..n {
                vt[[ki, c]] = eigenvectors[[c, i]].conj();
            }
            if sv > 1e-15 {
                for r in 0..m {
                    let mut sum = c64_zero();
                    for c in 0..n {
                        sum += mat[[r, c]] * eigenvectors[[c, i]];
                    }
                    u[[r, ki]] = c64_scale(sum, 1.0 / sv);
                }
            }
        }
        (u, sigma, vt)
    }
}

fn hermitian_eigen_deflation(a: &ndarray::Array2<C64>) -> (Vec<f64>, ndarray::Array2<C64>) {
    let n = a.dim().0;
    if n == 1 {
        return (
            vec![a[[0, 0]].re],
            ndarray::Array2::from_elem((1, 1), c64_one()),
        );
    }
    let mut mat = a.clone();
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = ndarray::Array2::<C64>::zeros((n, n));
    // Store previously found eigenvectors for orthogonalization
    let mut prev_vecs: Vec<ndarray::Array1<C64>> = Vec::with_capacity(n);
    for col in 0..n {
        let mut v = ndarray::Array1::<C64>::zeros(n);
        for j in 0..n {
            v[j] = C64::new((j + 1) as f64, 0.3 * (j as f64));
        }
        // Orthogonalize against previously found eigenvectors (Gram-Schmidt)
        for prev in &prev_vecs {
            let overlap: C64 = (0..n).map(|i| prev[i].conj() * v[i]).sum();
            for i in 0..n {
                v[i] -= overlap * prev[i];
            }
        }
        let init_norm: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if init_norm < 1e-15 {
            // Try alternate initial vector
            for j in 0..n {
                v[j] = C64::new(
                    if (j + col) % 2 == 0 { 1.0 } else { -0.5 },
                    if (j + col) % 3 == 0 { 0.7 } else { -0.2 },
                );
            }
            for prev in &prev_vecs {
                let overlap: C64 = (0..n).map(|i| prev[i].conj() * v[i]).sum();
                for i in 0..n {
                    v[i] -= overlap * prev[i];
                }
            }
            let n2: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if n2 < 1e-15 {
                eigenvalues.push(0.0);
                continue;
            }
            for j in 0..n {
                v[j] = c64_scale(v[j], 1.0 / n2);
            }
        } else {
            for j in 0..n {
                v[j] = c64_scale(v[j], 1.0 / init_norm);
            }
        }
        let mut eigenvalue = 0.0;
        for _ in 0..200 {
            let mut w = ndarray::Array1::<C64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    w[i] += mat[[i, j]] * v[j];
                }
            }
            // Re-orthogonalize against previous eigenvectors
            for prev in &prev_vecs {
                let overlap: C64 = (0..n).map(|i| prev[i].conj() * w[i]).sum();
                for i in 0..n {
                    w[i] -= overlap * prev[i];
                }
            }
            let norm_w: f64 = w.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if norm_w < 1e-15 {
                break;
            }
            for i in 0..n {
                v[i] = c64_scale(w[i], 1.0 / norm_w);
            }
            let mut lambda = c64_zero();
            for i in 0..n {
                for j in 0..n {
                    lambda += v[i].conj() * mat[[i, j]] * v[j];
                }
            }
            eigenvalue = lambda.re;
        }
        eigenvalues.push(eigenvalue);
        for i in 0..n {
            eigenvectors[[i, col]] = v[i];
        }
        prev_vecs.push(v.clone());
        for i in 0..n {
            for j in 0..n {
                mat[[i, j]] -= C64::new(eigenvalue, 0.0) * v[i] * v[j].conj();
            }
        }
    }
    (eigenvalues, eigenvectors)
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the CAMPS simulator.
#[derive(Debug, Clone)]
pub struct CampsConfig {
    /// Maximum MPS bond dimension before truncation.
    pub max_bond_dim: usize,
    /// SVD singular value cutoff.
    pub svd_cutoff: f64,
    /// Number of qubits.
    pub num_qubits: usize,
}

impl Default for CampsConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            svd_cutoff: 1e-10,
            num_qubits: 1,
        }
    }
}

impl CampsConfig {
    pub fn with_max_bond_dim(mut self, d: usize) -> Self {
        self.max_bond_dim = d;
        self
    }
    pub fn with_svd_cutoff(mut self, c: f64) -> Self {
        self.svd_cutoff = c;
        self
    }
    pub fn with_num_qubits(mut self, n: usize) -> Self {
        self.num_qubits = n;
        self
    }
}

// ============================================================
// CAMPS STATE
// ============================================================

/// Combined Clifford + MPS state (CAMPS).
#[derive(Clone, Debug)]
pub struct CampsState {
    tableau: CliffordTableau,
    residual: MpsResidual,
    config: CampsConfig,
}

impl CampsState {
    pub fn new(config: CampsConfig) -> Self {
        let n = config.num_qubits;
        assert!(n > 0, "Need at least 1 qubit");
        let tableau = CliffordTableau::new(n);
        let mut residual = MpsResidual::identity(n, 2);
        residual.max_bond_dim = config.max_bond_dim;
        residual.svd_cutoff = config.svd_cutoff;
        CampsState {
            tableau,
            residual,
            config,
        }
    }

    pub fn apply_gate(&mut self, gate_type: &GateKind, qubits: &[usize]) -> Result<(), CampsError> {
        let n = self.config.num_qubits;
        for &q in qubits {
            if q >= n {
                return Err(CampsError::QubitOutOfBounds {
                    qubit: q,
                    num_qubits: n,
                });
            }
        }
        if is_clifford(gate_type) {
            self.apply_clifford(gate_type, qubits)
        } else {
            self.apply_non_clifford(gate_type, qubits)
        }
    }

    pub fn apply_circuit(&mut self, gates: &[(GateKind, Vec<usize>)]) -> Result<(), CampsError> {
        for (gate, qubits) in gates {
            self.apply_gate(gate, qubits)?;
        }
        Ok(())
    }

    pub fn measure_z(&mut self, qubit: usize) -> Result<bool, CampsError> {
        if qubit >= self.config.num_qubits {
            return Err(CampsError::QubitOutOfBounds {
                qubit,
                num_qubits: self.config.num_qubits,
            });
        }
        let (outcome, deterministic) = self.tableau.measure_z(qubit);
        if deterministic {
            Ok(outcome)
        } else {
            let exp_z = self.residual.expectation_z(qubit);
            let prob_0 = (1.0 + exp_z) / 2.0;
            Ok(rand::random::<f64>() >= prob_0)
        }
    }

    pub fn expectation_value_z(&self, qubit: usize) -> Result<f64, CampsError> {
        if qubit >= self.config.num_qubits {
            return Err(CampsError::QubitOutOfBounds {
                qubit,
                num_qubits: self.config.num_qubits,
            });
        }
        let n = self.config.num_qubits;
        if self.is_clifford_only() {
            for i in 0..n {
                if !self.tableau.commutes_with_z(i, qubit) {
                    return Ok(0.0);
                }
            }
            for i in 0..n {
                if self.tableau.z_table[i][qubit] && !self.tableau.x_table[i][qubit] {
                    let mut only_z = true;
                    for j in 0..n {
                        if j != qubit && (self.tableau.x_table[i][j] || self.tableau.z_table[i][j])
                        {
                            only_z = false;
                            break;
                        }
                    }
                    if only_z {
                        return Ok(if self.tableau.phases[i] == 0 {
                            1.0
                        } else {
                            -1.0
                        });
                    }
                }
            }
            Ok(self.residual.expectation_z(qubit))
        } else {
            Ok(self.residual.expectation_z(qubit))
        }
    }

    pub fn mps_bond_dim(&self) -> usize {
        self.residual.max_bond_dim_actual()
    }
    pub fn is_clifford_only(&self) -> bool {
        self.residual.is_identity()
    }
    pub fn tableau(&self) -> &CliffordTableau {
        &self.tableau
    }
    pub fn residual(&self) -> &MpsResidual {
        &self.residual
    }

    fn apply_clifford(&mut self, gate: &GateKind, qubits: &[usize]) -> Result<(), CampsError> {
        match gate {
            GateKind::H => {
                self.tableau.apply_h(qubits[0]);
                Ok(())
            }
            GateKind::S => {
                self.tableau.apply_s(qubits[0]);
                Ok(())
            }
            GateKind::Sdg => {
                self.tableau.apply_sdg(qubits[0]);
                Ok(())
            }
            GateKind::X => {
                self.tableau.apply_x(qubits[0]);
                Ok(())
            }
            GateKind::Y => {
                self.tableau.apply_y(qubits[0]);
                Ok(())
            }
            GateKind::Z => {
                self.tableau.apply_z(qubits[0]);
                Ok(())
            }
            GateKind::Cx => {
                if qubits.len() < 2 {
                    return Err(CampsError::GateFailed("CX requires 2 qubits".into()));
                }
                self.tableau.apply_cx(qubits[0], qubits[1]);
                Ok(())
            }
            GateKind::Cz => {
                if qubits.len() < 2 {
                    return Err(CampsError::GateFailed("CZ requires 2 qubits".into()));
                }
                self.tableau.apply_cz(qubits[0], qubits[1]);
                Ok(())
            }
            GateKind::Swap => {
                if qubits.len() < 2 {
                    return Err(CampsError::GateFailed("SWAP requires 2 qubits".into()));
                }
                self.tableau.apply_swap(qubits[0], qubits[1]);
                Ok(())
            }
            GateKind::Rz(theta) => {
                for _ in 0..clifford_quarter_turns(*theta) {
                    self.tableau.apply_s(qubits[0]);
                }
                Ok(())
            }
            GateKind::Rx(theta) => {
                let turns = clifford_quarter_turns(*theta);
                self.tableau.apply_h(qubits[0]);
                for _ in 0..turns {
                    self.tableau.apply_s(qubits[0]);
                }
                self.tableau.apply_h(qubits[0]);
                Ok(())
            }
            GateKind::Ry(theta) => {
                let turns = clifford_quarter_turns(*theta);
                self.tableau.apply_sdg(qubits[0]);
                self.tableau.apply_h(qubits[0]);
                for _ in 0..turns {
                    self.tableau.apply_s(qubits[0]);
                }
                self.tableau.apply_h(qubits[0]);
                self.tableau.apply_s(qubits[0]);
                Ok(())
            }
            _ => Err(CampsError::GateFailed(format!(
                "Gate {} not handled as Clifford",
                gate
            ))),
        }
    }

    fn apply_non_clifford(&mut self, gate: &GateKind, qubits: &[usize]) -> Result<(), CampsError> {
        match gate {
            GateKind::T => {
                let phase = C64::from_polar(1.0, FRAC_PI_4);
                self.residual
                    .apply_single_qubit(qubits[0], &[[c64_one(), c64_zero()], [c64_zero(), phase]]);
                Ok(())
            }
            GateKind::Tdg => {
                let phase = C64::from_polar(1.0, -FRAC_PI_4);
                self.residual
                    .apply_single_qubit(qubits[0], &[[c64_one(), c64_zero()], [c64_zero(), phase]]);
                Ok(())
            }
            GateKind::Rz(theta) => {
                let half = theta / 2.0;
                self.residual.apply_single_qubit(
                    qubits[0],
                    &[
                        [C64::from_polar(1.0, -half), c64_zero()],
                        [c64_zero(), C64::from_polar(1.0, half)],
                    ],
                );
                Ok(())
            }
            GateKind::Rx(theta) => {
                let half = theta / 2.0;
                let c = C64::new(half.cos(), 0.0);
                let s = C64::new(0.0, -half.sin());
                self.residual
                    .apply_single_qubit(qubits[0], &[[c, s], [s, c]]);
                Ok(())
            }
            GateKind::Ry(theta) => {
                let half = theta / 2.0;
                let c = C64::new(half.cos(), 0.0);
                self.residual.apply_single_qubit(
                    qubits[0],
                    &[
                        [c, C64::new(-half.sin(), 0.0)],
                        [C64::new(half.sin(), 0.0), c],
                    ],
                );
                Ok(())
            }
            _ => Err(CampsError::GateFailed(format!(
                "Unrecognized non-Clifford gate: {}",
                gate
            ))),
        }
    }
}

impl fmt::Display for CampsState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CampsState({} qubits, bond_dim={}, clifford_only={})",
            self.config.num_qubits,
            self.mps_bond_dim(),
            self.is_clifford_only()
        )
    }
}

// ============================================================
// CLIFFORD PREFIX EXTRACTION
// ============================================================

/// Extract the maximal Clifford prefix from a circuit.
pub fn extract_clifford_prefix(
    circuit: &[(GateKind, Vec<usize>)],
) -> (Vec<(GateKind, Vec<usize>)>, Vec<(GateKind, Vec<usize>)>) {
    let mut split_idx = 0;
    for (i, (gate, _)) in circuit.iter().enumerate() {
        if is_clifford(gate) {
            split_idx = i + 1;
        } else {
            break;
        }
    }
    (circuit[..split_idx].to_vec(), circuit[split_idx..].to_vec())
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-6;
    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn test_tableau_new() {
        let tab = CliffordTableau::new(3);
        assert_eq!(tab.num_qubits(), 3);
        assert!(tab.z_table[0][0] && tab.z_table[1][1] && tab.z_table[2][2]);
        assert!(tab.x_table[3][0] && tab.x_table[4][1] && tab.x_table[5][2]);
        for p in &tab.phases {
            assert_eq!(*p, 0);
        }
        assert!(tab.is_identity());
    }

    #[test]
    fn test_tableau_apply_h() {
        let mut tab = CliffordTableau::new(1);
        tab.apply_h(0);
        assert!(tab.x_table[0][0] && !tab.z_table[0][0]);
        tab.apply_h(0);
        assert!(tab.z_table[0][0] && !tab.x_table[0][0]);
    }

    #[test]
    fn test_tableau_apply_s() {
        let mut tab = CliffordTableau::new(1);
        tab.apply_s(0);
        assert!(!tab.x_table[0][0] && tab.z_table[0][0]);
        tab.apply_s(0);
        tab.apply_s(0);
        tab.apply_s(0);
        assert!(tab.z_table[0][0] && !tab.x_table[0][0]);
    }

    #[test]
    fn test_tableau_apply_s_after_h() {
        let mut tab = CliffordTableau::new(1);
        tab.apply_h(0);
        tab.apply_s(0);
        assert!(tab.x_table[0][0] && tab.z_table[0][0]);
    }

    #[test]
    fn test_tableau_apply_cx_bell() {
        let mut tab = CliffordTableau::new(2);
        tab.apply_h(0);
        tab.apply_cx(0, 1);
        assert!(tab.x_table[0][0] && tab.x_table[0][1]);
        assert!(tab.z_table[1][0] && tab.z_table[1][1]);
    }

    #[test]
    fn test_tableau_apply_cz() {
        let mut tab = CliffordTableau::new(2);
        tab.apply_cz(0, 1);
        assert!(tab.z_table[0][0] && !tab.z_table[0][1]);
        assert!(tab.z_table[1][1] && !tab.z_table[1][0]);
    }

    #[test]
    fn test_tableau_measure_z_deterministic() {
        let mut tab = CliffordTableau::new(2);
        let (outcome, det) = tab.measure_z(0);
        assert!(det && !outcome);
    }

    #[test]
    fn test_tableau_measure_z_random_bell() {
        let mut tab = CliffordTableau::new(2);
        tab.apply_h(0);
        tab.apply_cx(0, 1);
        let (_, det) = tab.measure_z(0);
        assert!(!det);
    }

    #[test]
    fn test_tableau_commutes_with_z() {
        let tab = CliffordTableau::new(2);
        assert!(tab.commutes_with_z(0, 0));
        assert!(!tab.commutes_with_z(2, 0));
    }

    #[test]
    fn test_tableau_row_multiply() {
        let mut tab = CliffordTableau::new(2);
        tab.row_multiply(0, 1);
        assert!(tab.z_table[0][0] && tab.z_table[0][1] && tab.phases[0] == 0);
    }

    #[test]
    fn test_tableau_sdg_inverse() {
        let mut tab = CliffordTableau::new(1);
        tab.apply_s(0);
        tab.apply_sdg(0);
        assert!(tab.z_table[0][0] && !tab.x_table[0][0] && tab.phases[0] == 0);
    }

    #[test]
    fn test_mps_identity() {
        let mps = MpsResidual::identity(4, 2);
        assert!(mps.is_identity());
        assert_eq!(mps.total_bond_dim(), 3);
        assert!(approx_eq(mps.norm(), 1.0));
    }

    #[test]
    fn test_mps_apply_single_qubit_x() {
        let mut mps = MpsResidual::identity(2, 2);
        mps.apply_single_qubit(0, &[[c64_zero(), c64_one()], [c64_one(), c64_zero()]]);
        assert!((mps.tensors[0][[0, 1, 0]] - c64_one()).norm() < 1e-10);
        assert!(approx_eq(mps.norm(), 1.0));
    }

    #[test]
    fn test_mps_apply_two_qubit_cnot() {
        let mut mps = MpsResidual::identity(2, 2);
        let mut cnot = [[c64_zero(); 4]; 4];
        cnot[0][0] = c64_one();
        cnot[1][1] = c64_one();
        cnot[2][3] = c64_one();
        cnot[3][2] = c64_one();
        mps.apply_two_qubit(0, 1, &cnot);
        assert!(approx_eq(mps.norm(), 1.0));
    }

    #[test]
    fn test_mps_svd_truncate() {
        let mut mps = MpsResidual::identity(3, 2);
        mps.max_bond_dim = 4;
        let s = 1.0 / 2.0_f64.sqrt();
        mps.apply_single_qubit(
            0,
            &[
                [C64::new(s, 0.0), C64::new(s, 0.0)],
                [C64::new(s, 0.0), C64::new(-s, 0.0)],
            ],
        );
        mps.svd_truncate(0);
        assert!(approx_eq(mps.norm(), 1.0));
    }

    #[test]
    fn test_mps_norm() {
        assert!(approx_eq(MpsResidual::identity(3, 2).norm(), 1.0));
    }

    #[test]
    fn test_mps_expectation_z() {
        let mps = MpsResidual::identity(2, 2);
        assert!(approx_eq(mps.expectation_z(0), 1.0));
        assert!(approx_eq(mps.expectation_z(1), 1.0));
        let mut mps2 = MpsResidual::identity(2, 2);
        mps2.apply_single_qubit(0, &[[c64_zero(), c64_one()], [c64_one(), c64_zero()]]);
        assert!(approx_eq(mps2.expectation_z(0), -1.0));
    }

    #[test]
    fn test_mps_bell_expectations() {
        let mut mps = MpsResidual::identity(2, 2);
        let s = 1.0 / 2.0_f64.sqrt();
        mps.apply_single_qubit(
            0,
            &[
                [C64::new(s, 0.0), C64::new(s, 0.0)],
                [C64::new(s, 0.0), C64::new(-s, 0.0)],
            ],
        );
        let mut cnot = [[c64_zero(); 4]; 4];
        cnot[0][0] = c64_one();
        cnot[1][1] = c64_one();
        cnot[2][3] = c64_one();
        cnot[3][2] = c64_one();
        mps.apply_two_qubit(0, 1, &cnot);
        assert!(mps.expectation_z(0).abs() < 1e-8);
        assert!(mps.expectation_z(1).abs() < 1e-8);
        assert!(approx_eq(mps.norm(), 1.0));
    }

    #[test]
    fn test_camps_new() {
        let state = CampsState::new(CampsConfig::default().with_num_qubits(4));
        assert!(state.is_clifford_only());
        assert_eq!(state.mps_bond_dim(), 1);
    }

    #[test]
    fn test_camps_clifford_only_stays_identity() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(4));
        state.apply_gate(&GateKind::H, &[0]).unwrap();
        state.apply_gate(&GateKind::Cx, &[0, 1]).unwrap();
        state.apply_gate(&GateKind::S, &[2]).unwrap();
        state.apply_gate(&GateKind::Cz, &[1, 2]).unwrap();
        state.apply_gate(&GateKind::X, &[3]).unwrap();
        state.apply_gate(&GateKind::Swap, &[0, 3]).unwrap();
        assert!(state.is_clifford_only());
        assert_eq!(state.mps_bond_dim(), 1);
    }

    #[test]
    fn test_camps_t_on_zero() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(2));
        state.apply_gate(&GateKind::T, &[0]).unwrap();
        assert!(state.is_clifford_only());
    }

    #[test]
    fn test_camps_ry_non_clifford() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(2));
        state.apply_gate(&GateKind::Ry(PI / 3.0), &[0]).unwrap();
        assert!(!state.is_clifford_only());
    }

    #[test]
    fn test_camps_mixed_circuit() {
        let mut state = CampsState::new(
            CampsConfig::default()
                .with_num_qubits(4)
                .with_max_bond_dim(32),
        );
        state.apply_gate(&GateKind::H, &[0]).unwrap();
        state.apply_gate(&GateKind::Cx, &[0, 1]).unwrap();
        assert!(state.is_clifford_only());
        state.apply_gate(&GateKind::Rz(0.123), &[1]).unwrap();
        state.apply_gate(&GateKind::H, &[3]).unwrap();
    }

    #[test]
    fn test_camps_measure_zero() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(1));
        assert!(!state.measure_z(0).unwrap());
    }

    #[test]
    fn test_camps_measure_after_x() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(1));
        state.apply_gate(&GateKind::X, &[0]).unwrap();
        assert!(state.measure_z(0).unwrap());
    }

    #[test]
    fn test_camps_expectation_z() {
        let state = CampsState::new(CampsConfig::default().with_num_qubits(2));
        assert!(approx_eq(state.expectation_value_z(0).unwrap(), 1.0));
        assert!(approx_eq(state.expectation_value_z(1).unwrap(), 1.0));
    }

    #[test]
    fn test_camps_expectation_after_x() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(2));
        state.apply_gate(&GateKind::X, &[0]).unwrap();
        assert!(approx_eq(state.expectation_value_z(0).unwrap(), -1.0));
    }

    #[test]
    fn test_is_clifford_classification() {
        assert!(is_clifford(&GateKind::H));
        assert!(is_clifford(&GateKind::S));
        assert!(is_clifford(&GateKind::Sdg));
        assert!(is_clifford(&GateKind::X));
        assert!(is_clifford(&GateKind::Y));
        assert!(is_clifford(&GateKind::Z));
        assert!(is_clifford(&GateKind::Cx));
        assert!(is_clifford(&GateKind::Cz));
        assert!(is_clifford(&GateKind::Swap));
        assert!(!is_clifford(&GateKind::T));
        assert!(!is_clifford(&GateKind::Tdg));
        assert!(!is_clifford(&GateKind::Rz(0.123)));
        assert!(is_clifford(&GateKind::Rz(0.0)));
        assert!(is_clifford(&GateKind::Rz(FRAC_PI_2)));
        assert!(is_clifford(&GateKind::Rz(PI)));
        assert!(is_clifford(&GateKind::Rx(PI)));
        assert!(is_clifford(&GateKind::Ry(PI)));
    }

    #[test]
    fn test_extract_clifford_prefix_mixed() {
        let circuit = vec![
            (GateKind::H, vec![0]),
            (GateKind::Cx, vec![0, 1]),
            (GateKind::S, vec![1]),
            (GateKind::T, vec![0]),
            (GateKind::H, vec![1]),
        ];
        let (prefix, rem) = extract_clifford_prefix(&circuit);
        assert_eq!(prefix.len(), 3);
        assert_eq!(rem.len(), 2);
        assert_eq!(rem[0].0, GateKind::T);
    }

    #[test]
    fn test_extract_clifford_prefix_all() {
        let (p, r) = extract_clifford_prefix(&[(GateKind::H, vec![0]), (GateKind::Cx, vec![0, 1])]);
        assert_eq!(p.len(), 2);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_extract_clifford_prefix_none() {
        let (p, r) =
            extract_clifford_prefix(&[(GateKind::T, vec![0]), (GateKind::Rz(0.5), vec![1])]);
        assert_eq!(p.len(), 0);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_large_clifford_20q() {
        let n = 20;
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(n));
        for q in 0..n {
            state.apply_gate(&GateKind::H, &[q]).unwrap();
        }
        for q in 0..(n - 1) {
            state.apply_gate(&GateKind::Cx, &[q, q + 1]).unwrap();
        }
        for q in 0..n {
            state.apply_gate(&GateKind::S, &[q]).unwrap();
        }
        assert!(state.is_clifford_only());
        assert_eq!(state.mps_bond_dim(), 1);
    }

    #[test]
    fn test_bond_dim_bounded() {
        let n = 6;
        let mut state = CampsState::new(
            CampsConfig::default()
                .with_num_qubits(n)
                .with_max_bond_dim(16),
        );
        for q in 0..n {
            state.apply_gate(&GateKind::H, &[q]).unwrap();
        }
        for q in 0..n {
            state.apply_gate(&GateKind::T, &[q]).unwrap();
        }
        for q in 0..(n - 1) {
            state.apply_gate(&GateKind::Cx, &[q, q + 1]).unwrap();
        }
        assert!(state.mps_bond_dim() <= 16);
    }

    #[test]
    fn test_apply_circuit() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(3));
        state
            .apply_circuit(&[
                (GateKind::H, vec![0]),
                (GateKind::Cx, vec![0, 1]),
                (GateKind::T, vec![2]),
                (GateKind::S, vec![1]),
            ])
            .unwrap();
    }

    #[test]
    fn test_qubit_out_of_bounds() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(2));
        match state.apply_gate(&GateKind::H, &[5]) {
            Err(CampsError::QubitOutOfBounds {
                qubit: 5,
                num_qubits: 2,
            }) => {}
            other => panic!("Expected QubitOutOfBounds, got {:?}", other),
        }
    }

    #[test]
    fn test_config_builder() {
        let c = CampsConfig::default()
            .with_num_qubits(10)
            .with_max_bond_dim(128)
            .with_svd_cutoff(1e-12);
        assert_eq!(c.num_qubits, 10);
        assert_eq!(c.max_bond_dim, 128);
        assert!((c.svd_cutoff - 1e-12).abs() < 1e-15);
    }

    #[test]
    fn test_display() {
        let state = CampsState::new(CampsConfig::default().with_num_qubits(3));
        let d = format!("{}", state);
        assert!(d.contains("3 qubits") && d.contains("clifford_only=true"));
    }

    #[test]
    fn test_gate_kind_display() {
        assert_eq!(format!("{}", GateKind::H), "H");
        assert_eq!(format!("{}", GateKind::T), "T");
        assert_eq!(format!("{}", GateKind::Cx), "CX");
        assert!(format!("{}", GateKind::Rz(1.234)).contains("Rz"));
    }

    #[test]
    fn test_error_display() {
        let e = CampsError::QubitOutOfBounds {
            qubit: 5,
            num_qubits: 3,
        };
        assert!(format!("{}", e).contains("5") && format!("{}", e).contains("3"));
        let e2: Box<dyn std::error::Error> = Box::new(CampsError::SvdFailed("fail".into()));
        assert!(e2.to_string().contains("fail"));
    }

    #[test]
    fn test_clifford_rz() {
        let mut state = CampsState::new(CampsConfig::default().with_num_qubits(2));
        state.apply_gate(&GateKind::Rz(FRAC_PI_2), &[0]).unwrap();
        assert!(state.is_clifford_only());
    }

    #[test]
    fn test_rx_ry_non_clifford() {
        let mut s1 = CampsState::new(CampsConfig::default().with_num_qubits(2));
        s1.apply_gate(&GateKind::Rx(0.7), &[0]).unwrap();
        assert!(!s1.is_clifford_only());
        let mut s2 = CampsState::new(CampsConfig::default().with_num_qubits(2));
        s2.apply_gate(&GateKind::Ry(1.2), &[1]).unwrap();
        assert!(!s2.is_clifford_only());
    }

    #[test]
    fn test_tableau_display() {
        let d = format!("{}", CliffordTableau::new(2));
        assert!(d.contains("CliffordTableau") && d.contains("2 qubits"));
    }
}
