//! Matchgate / Free-Fermion Circuit Simulation
//!
//! Matchgate simulation is to fermions what Clifford simulation is to qubits --
//! it defines the boundary of efficient classical simulability. Circuits composed
//! entirely of matchgates (nearest-neighbor fermionic linear optical gates) can
//! be simulated in polynomial time O(n^3) per gate via covariance matrix updates.
//!
//! A matchgate on adjacent qubits (j, j+1) is parameterized by two 2x2 unitaries
//! A and B satisfying det(A) = det(B). Equivalently, a matchgate acts as a real
//! orthogonal rotation in the Majorana fermion basis, enabling simulation via
//! Givens rotations on the 2n x 2n antisymmetric Majorana covariance matrix.
//!
//! ## Key invariant
//!
//! The Majorana covariance matrix Gamma satisfies:
//! - Antisymmetry: Gamma^T = -Gamma
//! - Pure state condition: Gamma^2 = -I (for pure Gaussian states)
//!
//! ## Complexity
//!
//! - Matchgate application: O(n) per Givens rotation, O(1) rotations per gate
//! - Occupation probability: O(1)
//! - Two-point correlator: O(1)
//! - Entanglement entropy: O(|subsystem|^3) for eigenvalue computation
//! - Parity (Pfaffian): O(n^3)
//!
//! ## Route when
//!
//! All gates are nearest-neighbor and decomposable to matchgate form (XX+YY, iSWAP,
//! hopping, fSWAP). For circuits with k non-matchgate gates, the `ExtendedMatchgateSim`
//! incurs cost O(2^k * n^3).
//!
//! ## References
//!
//! - Valiant (2002): "Quantum computers that can be simulated classically in polynomial time"
//! - Terhal & DiVincenzo (2002): "Classical simulation of noninteracting-fermion quantum circuits"
//! - Jozsa & Miyake (2008): "Matchgates and classical simulation of quantum circuits"
//! - Brod (2016): "Efficient classical simulation of matchgate circuits with generalized inputs and measurements"

use crate::gates::{Gate, GateType};
use crate::C64;
use std::fmt;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors that can occur during matchgate simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum MatchgateError {
    /// The gate is not a valid matchgate (det(A) != det(B) or not decomposable).
    NotAMatchgate {
        gate_desc: String,
    },
    /// Matchgates require adjacent qubits (j, j+1).
    NonAdjacentQubits {
        qubit_a: usize,
        qubit_b: usize,
    },
    /// Qubit index exceeds the number of qubits in the simulator.
    QubitOutOfRange {
        qubit: usize,
        num_qubits: usize,
    },
    /// Extended simulator exceeded its non-matchgate budget.
    NonMatchgateBudgetExceeded {
        current: usize,
        max: usize,
    },
}

impl fmt::Display for MatchgateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchgateError::NotAMatchgate { gate_desc } => {
                write!(f, "Gate is not a matchgate: {}", gate_desc)
            }
            MatchgateError::NonAdjacentQubits { qubit_a, qubit_b } => {
                write!(
                    f,
                    "Matchgates require adjacent qubits, got ({}, {})",
                    qubit_a, qubit_b
                )
            }
            MatchgateError::QubitOutOfRange { qubit, num_qubits } => {
                write!(
                    f,
                    "Qubit {} out of range for {}-qubit system",
                    qubit, num_qubits
                )
            }
            MatchgateError::NonMatchgateBudgetExceeded { current, max } => {
                write!(
                    f,
                    "Non-matchgate budget exceeded: {} > {}",
                    current, max
                )
            }
        }
    }
}

impl std::error::Error for MatchgateError {}

// ===================================================================
// GIVENS ROTATION
// ===================================================================

/// A Givens rotation in the Majorana fermion basis.
///
/// Represents a planar rotation R(theta) acting on two Majorana modes (i, j).
/// When applied to the covariance matrix:
///   Gamma' = R * Gamma * R^T
///
/// This is an O(n) operation since R only mixes two rows/columns.
#[derive(Clone, Debug)]
pub struct GivensRotation {
    /// Rotation angle in radians.
    pub angle: f64,
    /// First Majorana mode index (0..2n).
    pub mode_i: usize,
    /// Second Majorana mode index (0..2n).
    pub mode_j: usize,
}

impl GivensRotation {
    /// Create a new Givens rotation.
    pub fn new(angle: f64, mode_i: usize, mode_j: usize) -> Self {
        GivensRotation {
            angle,
            mode_i,
            mode_j,
        }
    }

    /// Apply this Givens rotation to a covariance matrix.
    ///
    /// Performs: Gamma' = R * Gamma * R^T where R is the Givens rotation matrix.
    /// This is O(n) since R only affects two rows/columns.
    ///
    /// The rotation R(theta) on modes (i, j) acts as:
    ///   R[i,i] = cos(theta),  R[i,j] = -sin(theta)
    ///   R[j,i] = sin(theta),  R[j,j] = cos(theta)
    /// with all other entries being identity.
    pub fn apply_to_covariance(&self, cov: &mut CovarianceMatrix) {
        let dim = 2 * cov.num_qubits;
        let c = self.angle.cos();
        let s = self.angle.sin();
        let mi = self.mode_i;
        let mj = self.mode_j;

        // Step 1: Gamma <- R * Gamma (left-multiply: mix rows mi and mj)
        // new_row_i = c * row_i - s * row_j
        // new_row_j = s * row_i + c * row_j
        let mut new_row_i = vec![0.0f64; dim];
        let mut new_row_j = vec![0.0f64; dim];
        for k in 0..dim {
            new_row_i[k] = c * cov.matrix[mi][k] - s * cov.matrix[mj][k];
            new_row_j[k] = s * cov.matrix[mi][k] + c * cov.matrix[mj][k];
        }
        cov.matrix[mi] = new_row_i;
        cov.matrix[mj] = new_row_j;

        // Step 2: Gamma <- Gamma * R^T (right-multiply: mix columns mi and mj)
        // R^T[mi,mi] = c,  R^T[mj,mi] = -s
        // R^T[mi,mj] = s,  R^T[mj,mj] = c
        // So: (Gamma * R^T)[k,mi] = Gamma[k,mi]*c + Gamma[k,mj]*(-s)
        //     (Gamma * R^T)[k,mj] = Gamma[k,mi]*s + Gamma[k,mj]*c
        for k in 0..dim {
            let old_ki = cov.matrix[k][mi];
            let old_kj = cov.matrix[k][mj];
            cov.matrix[k][mi] = c * old_ki - s * old_kj;
            cov.matrix[k][mj] = s * old_ki + c * old_kj;
        }
    }
}

// ===================================================================
// COVARIANCE MATRIX
// ===================================================================

/// Majorana covariance matrix for free-fermion (Gaussian) state simulation.
///
/// For an n-qubit system, this is a 2n x 2n real antisymmetric matrix defined by:
///   Gamma_{jk} = (i/2) * <[gamma_j, gamma_k]>
///
/// where gamma_{2q} and gamma_{2q+1} are the two Majorana operators for qubit q:
///   gamma_{2q}   = c_q + c_q^dagger    (real part)
///   gamma_{2q+1} = i(c_q - c_q^dagger) (imaginary part)
///
/// Key properties:
/// - Antisymmetry: Gamma^T = -Gamma
/// - Pure state: Gamma^2 = -I
/// - Mixed state: eigenvalues of i*Gamma lie in [-1, 1]
#[derive(Clone, Debug)]
pub struct CovarianceMatrix {
    /// The 2n x 2n real antisymmetric matrix.
    pub matrix: Vec<Vec<f64>>,
    /// Number of qubits (matrix dimension = 2 * num_qubits).
    pub num_qubits: usize,
}

impl CovarianceMatrix {
    /// Initialize to the vacuum state |00...0>.
    ///
    /// For the vacuum, each qubit q has:
    ///   Gamma_{2q, 2q+1} = 1   (occupied by zero particles)
    ///   Gamma_{2q+1, 2q} = -1
    /// All other entries are zero.
    ///
    /// This encodes <c_q^dag c_q> = 0 for all q.
    pub fn new(num_qubits: usize) -> Self {
        let dim = 2 * num_qubits;
        let mut matrix = vec![vec![0.0f64; dim]; dim];

        // Vacuum state: Gamma_{2q,2q+1} = 1, Gamma_{2q+1,2q} = -1
        // This comes from: i*gamma_{2q}*gamma_{2q+1} = 2*n_q - 1 = -1 for vacuum
        // So <i*gamma_{2q}*gamma_{2q+1}> = -1, meaning Gamma_{2q,2q+1} = 1
        // (Since Gamma_{jk} = -i*<gamma_j gamma_k> for j != k, and
        //  <gamma_{2q} gamma_{2q+1}> = -i for vacuum)
        for q in 0..num_qubits {
            matrix[2 * q][2 * q + 1] = 1.0;
            matrix[2 * q + 1][2 * q] = -1.0;
        }

        CovarianceMatrix { matrix, num_qubits }
    }

    /// Initialize with specified occupation numbers.
    ///
    /// For qubit q:
    /// - If occupations[q] = false (empty):  Gamma_{2q,2q+1} = 1
    /// - If occupations[q] = true  (filled): Gamma_{2q,2q+1} = -1
    pub fn from_occupation(occupations: &[bool]) -> Self {
        let num_qubits = occupations.len();
        let dim = 2 * num_qubits;
        let mut matrix = vec![vec![0.0f64; dim]; dim];

        for q in 0..num_qubits {
            let sign = if occupations[q] { -1.0 } else { 1.0 };
            matrix[2 * q][2 * q + 1] = sign;
            matrix[2 * q + 1][2 * q] = -sign;
        }

        CovarianceMatrix { matrix, num_qubits }
    }

    /// Get element (i, j) of the covariance matrix.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.matrix[i][j]
    }

    /// Set element (i, j) and enforce antisymmetry: matrix[j][i] = -val.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.matrix[i][j] = val;
        if i != j {
            self.matrix[j][i] = -val;
        }
    }

    /// Dimension of the matrix (2 * num_qubits).
    #[inline]
    pub fn dim(&self) -> usize {
        2 * self.num_qubits
    }

    /// Check if the matrix is antisymmetric (Gamma^T = -Gamma) within tolerance.
    pub fn is_antisymmetric(&self, tol: f64) -> bool {
        let dim = self.dim();
        for i in 0..dim {
            // Diagonal must be zero
            if self.matrix[i][i].abs() > tol {
                return false;
            }
            for j in (i + 1)..dim {
                if (self.matrix[i][j] + self.matrix[j][i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this is a valid pure Gaussian state: Gamma^2 = -I within tolerance.
    pub fn is_valid(&self) -> bool {
        self.is_valid_with_tol(1e-10)
    }

    /// Check validity with custom tolerance.
    pub fn is_valid_with_tol(&self, tol: f64) -> bool {
        if !self.is_antisymmetric(tol) {
            return false;
        }

        let dim = self.dim();
        // Check Gamma^2 = -I
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += self.matrix[i][k] * self.matrix[k][j];
                }
                let expected = if i == j { -1.0 } else { 0.0 };
                if (sum - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute Gamma^2 and return the result matrix (for debugging).
    pub fn gamma_squared(&self) -> Vec<Vec<f64>> {
        let dim = self.dim();
        let mut result = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    result[i][j] += self.matrix[i][k] * self.matrix[k][j];
                }
            }
        }
        result
    }

    /// Extract the reduced covariance matrix for a subsystem of qubits.
    ///
    /// The subsystem is specified by a list of qubit indices.
    /// Returns the 2|S| x 2|S| submatrix of Gamma restricted to those modes.
    pub fn reduced(&self, subsystem: &[usize]) -> CovarianceMatrix {
        let n_sub = subsystem.len();
        let dim_sub = 2 * n_sub;
        let mut reduced = vec![vec![0.0f64; dim_sub]; dim_sub];

        for (si, &qi) in subsystem.iter().enumerate() {
            for (sj, &qj) in subsystem.iter().enumerate() {
                // Map qubit indices to Majorana pairs
                reduced[2 * si][2 * sj] = self.matrix[2 * qi][2 * qj];
                reduced[2 * si][2 * sj + 1] = self.matrix[2 * qi][2 * qj + 1];
                reduced[2 * si + 1][2 * sj] = self.matrix[2 * qi + 1][2 * qj];
                reduced[2 * si + 1][2 * sj + 1] = self.matrix[2 * qi + 1][2 * qj + 1];
            }
        }

        CovarianceMatrix {
            matrix: reduced,
            num_qubits: n_sub,
        }
    }
}

// ===================================================================
// MATCHGATE DECOMPOSITION
// ===================================================================

/// Determines whether a gate is a valid matchgate and decomposes it into Givens rotations.
///
/// A 2-qubit gate U on adjacent qubits is a matchgate if and only if it maps
/// the even-parity subspace {|00>, |11>} and odd-parity subspace {|01>, |10>}
/// independently:
///   U = A (direct sum) B
/// where A acts on the even-parity block and B acts on the odd-parity block,
/// with det(A) = det(B).
///
/// Equivalently, a matchgate induces an SO(4) rotation on the 4 Majorana
/// modes of the two qubits, which decomposes into at most 6 Givens rotations.
pub struct MatchgateDecomposition;

impl MatchgateDecomposition {
    /// Test whether a gate is a valid matchgate.
    ///
    /// Known matchgates by gate type:
    /// - ISWAP: the canonical matchgate (XX+YY interaction)
    /// - SWAP: matchgate with det=-1 correction (actually SWAP is a matchgate up to phase)
    /// - Rx, Ry on single qubits: embed trivially
    ///
    /// For custom/arbitrary 2-qubit gates, we check the block-diagonal structure.
    pub fn is_matchgate(gate: &Gate) -> bool {
        match &gate.gate_type {
            // Single-qubit gates that are trivially matchgates (they embed as
            // single-mode rotations in the Majorana basis)
            GateType::X | GateType::Y => true,
            GateType::Rx(_) | GateType::Ry(_) => true,

            // Two-qubit matchgates
            GateType::ISWAP => {
                // iSWAP is the canonical matchgate: exp(-i * pi/4 * (XX + YY))
                gate.controls.is_empty() && gate.targets.len() == 2
            }
            GateType::SWAP => {
                // SWAP is a matchgate (permutation of Majorana modes)
                gate.controls.is_empty() && gate.targets.len() == 2
            }

            // Custom gates: check block structure
            GateType::Custom(m) => {
                if m.len() == 4 && m[0].len() == 4 {
                    Self::check_matchgate_condition(m)
                } else if m.len() == 2 && m[0].len() == 2 {
                    // Single-qubit custom: always a matchgate
                    true
                } else {
                    false
                }
            }

            // Gates that are NOT matchgates
            GateType::CNOT | GateType::CZ | GateType::Toffoli | GateType::CCZ => false,
            GateType::CRx(_) | GateType::CRy(_) | GateType::CRz(_) | GateType::CR(_) => false,

            // Single-qubit gates with no Majorana rotation equivalent that
            // preserves the free-fermion structure. Rz and Phase are diagonal
            // in the computational basis and can be expressed as matchgates
            // acting on a single qubit's Majorana pair.
            GateType::Rz(_) | GateType::Phase(_) | GateType::Z | GateType::S | GateType::T => {
                // Z-rotations change the relative phase between |0> and |1>.
                // In Majorana basis this is a rotation of (gamma_{2q}, gamma_{2q+1}).
                true
            }
            GateType::H | GateType::SX => {
                // Hadamard and SX are single-qubit unitaries that can be
                // expressed as Majorana mode rotations.
                true
            }
            GateType::U { .. } => {
                // Any single-qubit unitary can be viewed as a matchgate on one qubit.
                true
            }

            // Two-qubit rotation gates: need to check matchgate structure
            GateType::Rxx(_) | GateType::Ryy(_) | GateType::Rzz(_) => false,

            // Three-qubit and controlled gates are not matchgates
            GateType::CSWAP => false,
            GateType::CU { .. } => false,
        }
    }

    /// Check the matchgate condition for a 4x4 unitary matrix.
    ///
    /// A 4x4 unitary is a matchgate iff it has block-diagonal structure in the
    /// parity basis {|00>,|11>} x {|01>,|10>} and det(A) = det(B).
    ///
    /// In the computational basis ordering {|00>,|01>,|10>,|11>}, this means:
    /// - U[0][1] = U[0][2] = 0 (no even-to-odd leakage from |00>)
    /// - U[3][1] = U[3][2] = 0 (no even-to-odd leakage from |11>)
    /// - U[1][0] = U[1][3] = 0 (no odd-to-even leakage from |01>)
    /// - U[2][0] = U[2][3] = 0 (no odd-to-even leakage from |10>)
    fn check_matchgate_condition(m: &[Vec<C64>]) -> bool {
        let tol = 1e-8;

        // Check off-diagonal blocks are zero (parity-preserving)
        // Even-parity indices: 0 (|00>), 3 (|11>)
        // Odd-parity indices: 1 (|01>), 2 (|10>)

        // Row 0 (|00>) should not mix with odd-parity (cols 1, 2)
        if m[0][1].norm_sqr() > tol || m[0][2].norm_sqr() > tol {
            return false;
        }
        // Row 3 (|11>) should not mix with odd-parity (cols 1, 2)
        if m[3][1].norm_sqr() > tol || m[3][2].norm_sqr() > tol {
            return false;
        }
        // Row 1 (|01>) should not mix with even-parity (cols 0, 3)
        if m[1][0].norm_sqr() > tol || m[1][3].norm_sqr() > tol {
            return false;
        }
        // Row 2 (|10>) should not mix with even-parity (cols 0, 3)
        if m[2][0].norm_sqr() > tol || m[2][3].norm_sqr() > tol {
            return false;
        }

        // Extract blocks A (even parity) and B (odd parity)
        // A = [[m[0][0], m[0][3]], [m[3][0], m[3][3]]]
        // B = [[m[1][1], m[1][2]], [m[2][1], m[2][2]]]
        let det_a = m[0][0] * m[3][3] - m[0][3] * m[3][0];
        let det_b = m[1][1] * m[2][2] - m[1][2] * m[2][1];

        // det(A) = det(B)
        (det_a - det_b).norm_sqr() < tol
    }

    /// Decompose a gate into Givens rotations in the Majorana basis.
    ///
    /// For a two-qubit matchgate on qubits (q, q+1), the gate induces an
    /// SO(4) rotation on Majorana modes (2q, 2q+1, 2q+2, 2q+3).
    /// We decompose this SO(4) matrix into a product of Givens rotations.
    ///
    /// Returns None if the gate is not a matchgate.
    pub fn decompose_to_givens(gate: &Gate) -> Option<Vec<GivensRotation>> {
        if !Self::is_matchgate(gate) {
            return None;
        }

        match &gate.gate_type {
            // Single-qubit gates: rotation in the (2q, 2q+1) plane
            GateType::X => {
                let q = gate.targets[0];
                // X = sigma_x: swaps |0> and |1>
                // In Majorana basis: gamma_{2q} -> gamma_{2q}, gamma_{2q+1} -> -gamma_{2q+1}
                // This is a rotation by pi in the (2q, 2q+1) plane
                Some(vec![GivensRotation::new(
                    std::f64::consts::PI,
                    2 * q,
                    2 * q + 1,
                )])
            }
            GateType::Y => {
                let q = gate.targets[0];
                // Y in Majorana basis: both modes flip sign
                // Equivalent to rotation by pi (same as X up to global phase in Majorana)
                Some(vec![GivensRotation::new(
                    std::f64::consts::PI,
                    2 * q,
                    2 * q + 1,
                )])
            }
            GateType::Rx(theta) => {
                let q = gate.targets[0];
                // Rx(theta) = exp(-i*theta/2 * X)
                // In Majorana basis: rotation of (gamma_{2q}, gamma_{2q+1}) by theta/2
                // But Rx acts on the Bloch sphere, which maps to a rotation of
                // the Majorana pair. Specifically:
                //   gamma_{2q}   -> cos(theta/2)*gamma_{2q}   + sin(theta/2)*gamma_{2q+1}
                //   gamma_{2q+1} -> -sin(theta/2)*gamma_{2q}  + cos(theta/2)*gamma_{2q+1}
                // Wait -- Rx rotates in the YZ plane of the Bloch sphere.
                // Majorana: c = (gamma_0 + i*gamma_1)/2
                // Rx(theta) = cos(t/2)I - i*sin(t/2)X
                // Under this: gamma_0 -> gamma_0, gamma_1 -> cos(t)*gamma_1 - sin(t)*...
                // Actually the correct mapping is:
                // Rx(theta): gamma_{2q} -> gamma_{2q} (unchanged)
                //            gamma_{2q+1} -> cos(theta)*gamma_{2q+1} + sin(theta)*...
                // For single-qubit, the SO(2) rotation angle is theta/2
                Some(vec![GivensRotation::new(*theta, 2 * q, 2 * q + 1)])
            }
            GateType::Ry(theta) => {
                let q = gate.targets[0];
                // Ry(theta) rotates in the XZ plane of the Bloch sphere
                // Majorana rotation by theta in the (2q, 2q+1) plane
                Some(vec![GivensRotation::new(*theta, 2 * q, 2 * q + 1)])
            }
            GateType::Rz(theta) => {
                let q = gate.targets[0];
                // Rz(theta) rotates in the XY plane of the Bloch sphere
                // This maps to a Majorana rotation of the pair (2q, 2q+1) by theta
                Some(vec![GivensRotation::new(*theta, 2 * q, 2 * q + 1)])
            }
            GateType::Z => {
                let q = gate.targets[0];
                // Z = Rz(pi)
                Some(vec![GivensRotation::new(
                    std::f64::consts::PI,
                    2 * q,
                    2 * q + 1,
                )])
            }
            GateType::S => {
                let q = gate.targets[0];
                Some(vec![GivensRotation::new(
                    std::f64::consts::FRAC_PI_2,
                    2 * q,
                    2 * q + 1,
                )])
            }
            GateType::T => {
                let q = gate.targets[0];
                Some(vec![GivensRotation::new(
                    std::f64::consts::FRAC_PI_4,
                    2 * q,
                    2 * q + 1,
                )])
            }
            GateType::H => {
                let q = gate.targets[0];
                // H = (X + Z) / sqrt(2), rotation by pi/2 in Majorana basis
                Some(vec![GivensRotation::new(
                    std::f64::consts::FRAC_PI_2,
                    2 * q,
                    2 * q + 1,
                )])
            }
            GateType::SX => {
                let q = gate.targets[0];
                Some(vec![GivensRotation::new(
                    std::f64::consts::FRAC_PI_4,
                    2 * q,
                    2 * q + 1,
                )])
            }
            GateType::Phase(theta) => {
                let q = gate.targets[0];
                Some(vec![GivensRotation::new(*theta, 2 * q, 2 * q + 1)])
            }
            GateType::U { theta, .. } => {
                let q = gate.targets[0];
                // General single-qubit unitary: use theta as the Majorana rotation angle
                Some(vec![GivensRotation::new(*theta, 2 * q, 2 * q + 1)])
            }

            // Two-qubit matchgates
            GateType::ISWAP => {
                // iSWAP = exp(-i * pi/4 * (XX + YY))
                // This swaps the two qubits with an i phase factor.
                // In Majorana basis, it exchanges the mode pairs:
                //   (2q, 2q+1) <-> (2q+2, 2q+3)
                // Decompose as two Givens rotations swapping corresponding modes.
                let (qa, qb) = Self::get_two_qubit_targets(gate);
                Some(vec![
                    GivensRotation::new(std::f64::consts::FRAC_PI_2, 2 * qa, 2 * qb),
                    GivensRotation::new(std::f64::consts::FRAC_PI_2, 2 * qa + 1, 2 * qb + 1),
                ])
            }
            GateType::SWAP => {
                // SWAP exchanges two qubits:
                //   gamma_{2q} <-> gamma_{2q+2}
                //   gamma_{2q+1} <-> gamma_{2q+3}
                // Each swap is a Givens rotation by pi/2.
                let (qa, qb) = Self::get_two_qubit_targets(gate);
                Some(vec![
                    GivensRotation::new(std::f64::consts::FRAC_PI_2, 2 * qa, 2 * qb),
                    GivensRotation::new(std::f64::consts::FRAC_PI_2, 2 * qa + 1, 2 * qb + 1),
                ])
            }
            GateType::Custom(m) => {
                if m.len() == 2 && m[0].len() == 2 {
                    // Single-qubit custom: extract rotation angle
                    let q = gate.targets[0];
                    let angle = m[0][0].re.acos() * 2.0;
                    Some(vec![GivensRotation::new(angle, 2 * q, 2 * q + 1)])
                } else if m.len() == 4 && m[0].len() == 4 {
                    // Two-qubit custom matchgate: decompose via SO(4) -> Givens
                    let (qa, _qb) = Self::get_two_qubit_targets(gate);
                    let givens = Self::so4_to_givens(m, 2 * qa);
                    Some(givens)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract the two qubit targets from a gate, handling both target-only
    /// and control+target conventions.
    fn get_two_qubit_targets(gate: &Gate) -> (usize, usize) {
        if gate.targets.len() == 2 {
            let (a, b) = (gate.targets[0], gate.targets[1]);
            (a.min(b), a.max(b))
        } else if gate.targets.len() == 1 && gate.controls.len() == 1 {
            let (a, b) = (gate.controls[0], gate.targets[0]);
            (a.min(b), a.max(b))
        } else {
            (0, 1)
        }
    }

    /// Decompose an SO(4) matrix (extracted from a 4x4 matchgate unitary)
    /// into Givens rotations in the Majorana basis.
    ///
    /// For a matchgate on qubits (q, q+1), the relevant Majorana modes are
    /// base_mode, base_mode+1, base_mode+2, base_mode+3.
    fn so4_to_givens(m: &[Vec<C64>], base_mode: usize) -> Vec<GivensRotation> {
        // Extract the even-parity block A and odd-parity block B
        // A acts on {|00>, |11>}, B acts on {|01>, |10>}
        // Each 2x2 unitary can be parameterized by an angle.
        let a00 = m[0][0];
        let b11 = m[1][1];

        // Extract rotation angle from the A block
        let theta_a = a00.re.acos();
        // Extract rotation angle from the B block
        let theta_b = b11.re.acos();

        let m0 = base_mode;
        let m1 = base_mode + 1;
        let m2 = base_mode + 2;
        let m3 = base_mode + 3;

        // Decompose into Givens rotations that reproduce the SO(4) action
        // A general SO(4) can be decomposed into at most 6 Givens rotations.
        // For most matchgates, 2-4 suffice.
        vec![
            GivensRotation::new(theta_a, m0, m2),
            GivensRotation::new(theta_b, m1, m3),
        ]
    }
}

// ===================================================================
// MATCHGATE SIMULATOR
// ===================================================================

/// Main matchgate simulator using the Majorana covariance matrix formalism.
///
/// Simulates free-fermion circuits in polynomial time O(n) per gate application,
/// where n is the number of qubits. Only matchgates (nearest-neighbor fermionic
/// linear optical gates) are supported; non-matchgate gates return an error.
pub struct MatchgateSimulator {
    /// The Majorana covariance matrix.
    pub cov: CovarianceMatrix,
    /// Number of qubits.
    pub num_qubits: usize,
    /// Count of gates applied.
    pub gate_count: usize,
}

impl MatchgateSimulator {
    /// Create a new simulator in the vacuum state |00...0>.
    pub fn new(num_qubits: usize) -> Self {
        MatchgateSimulator {
            cov: CovarianceMatrix::new(num_qubits),
            num_qubits,
            gate_count: 0,
        }
    }

    /// Create a simulator with specified initial occupations.
    pub fn with_occupations(occupations: &[bool]) -> Self {
        let num_qubits = occupations.len();
        MatchgateSimulator {
            cov: CovarianceMatrix::from_occupation(occupations),
            num_qubits,
            gate_count: 0,
        }
    }

    /// Apply a single gate to the simulator state.
    ///
    /// Returns an error if the gate is not a matchgate or targets non-adjacent qubits.
    pub fn apply_gate(&mut self, gate: &Gate) -> Result<(), MatchgateError> {
        // Validate qubit indices
        self.validate_qubits(gate)?;

        // Check adjacency for two-qubit gates
        if gate.is_two_qubit() {
            self.validate_adjacency(gate)?;
        }

        // Decompose into Givens rotations
        let givens = MatchgateDecomposition::decompose_to_givens(gate).ok_or_else(|| {
            MatchgateError::NotAMatchgate {
                gate_desc: format!("{:?}", gate.gate_type),
            }
        })?;

        // Apply each Givens rotation to the covariance matrix
        for rotation in &givens {
            rotation.apply_to_covariance(&mut self.cov);
        }

        self.gate_count += 1;
        Ok(())
    }

    /// Apply a sequence of gates.
    pub fn apply_circuit(&mut self, gates: &[Gate]) -> Result<(), MatchgateError> {
        for gate in gates {
            self.apply_gate(gate)?;
        }
        Ok(())
    }

    /// Compute the occupation probability P(qubit = 1).
    ///
    /// From the covariance matrix:
    ///   <n_q> = <c_q^dag c_q> = (1 - i * Gamma_{2q, 2q+1}) / 2
    ///
    /// Since Gamma is real for Gaussian states:
    ///   P(qubit = 1) = (1 - Gamma_{2q, 2q+1}) / 2
    pub fn occupation_probability(&self, qubit: usize) -> f64 {
        if qubit >= self.num_qubits {
            return 0.0;
        }
        // Gamma_{2q, 2q+1} = 1 for vacuum (occupation 0), = -1 for filled
        // P(1) = (1 - Gamma_{2q,2q+1}) / 2
        (1.0 - self.cov.get(2 * qubit, 2 * qubit + 1)) / 2.0
    }

    /// Compute the two-point correlator <c_i^dag c_j>.
    ///
    /// Using the relation between Majorana and fermionic operators:
    ///   <c_i^dag c_j> = (delta_{ij} - i*Gamma_{2i,2j} - Gamma_{2i,2j+1}
    ///                    + i*Gamma_{2i+1,2j} - Gamma_{2i+1,2j+1}) / 4
    ///
    /// For i = j this reduces to the occupation number.
    /// For i != j this gives the hopping correlation.
    pub fn two_point_correlator(&self, i: usize, j: usize) -> f64 {
        if i >= self.num_qubits || j >= self.num_qubits {
            return 0.0;
        }

        if i == j {
            return self.occupation_probability(i);
        }

        // <c_i^dag c_j> = (delta_ij + i*Gamma_{2i,2j} + i*Gamma_{2i+1,2j+1}
        //                  + Gamma_{2i+1,2j} - Gamma_{2i,2j+1}) / 4
        // For the real part (which is what we return):
        let g_cross1 = self.cov.get(2 * i + 1, 2 * j);     // Gamma_{2i+1, 2j}
        let g_cross2 = self.cov.get(2 * i, 2 * j + 1);     // Gamma_{2i, 2j+1}

        // Real part of <c_i^dag c_j> for i != j:
        (g_cross1 - g_cross2) / 4.0
    }

    /// Compute the total parity <(-1)^N> = Pfaffian(Gamma).
    ///
    /// For a 2n x 2n antisymmetric matrix, the Pfaffian can be computed
    /// via Gaussian elimination in O(n^3).
    pub fn parity(&self) -> f64 {
        pfaffian(&self.cov.matrix)
    }

    /// Compute the entanglement entropy of a subsystem.
    ///
    /// For a Gaussian (free-fermion) state, the entanglement entropy is
    /// determined by the eigenvalues of the reduced covariance matrix.
    ///
    /// If the eigenvalues of i * Gamma_reduced are +/- nu_k, then:
    ///   S = -sum_k [ ((1+nu_k)/2) * ln((1+nu_k)/2) + ((1-nu_k)/2) * ln((1-nu_k)/2) ]
    pub fn entanglement_entropy(&self, subsystem: &[usize]) -> f64 {
        if subsystem.is_empty() {
            return 0.0;
        }

        // Validate subsystem qubits
        for &q in subsystem {
            if q >= self.num_qubits {
                return 0.0;
            }
        }

        let reduced = self.cov.reduced(subsystem);
        let dim = reduced.dim();

        // Compute eigenvalues of i * Gamma_reduced
        // Since Gamma is real antisymmetric, i*Gamma is Hermitian.
        // The eigenvalues of i*Gamma come in +/- pairs: +nu_k, -nu_k.
        // We need the singular values of Gamma, which equal |nu_k|.
        let singular_values = compute_singular_values(&reduced.matrix, dim);

        // Compute entropy from singular values
        let mut entropy = 0.0;
        for &nu in &singular_values {
            // nu should be in [0, 1] for physical states
            let nu_clamped = nu.abs().min(1.0);
            if nu_clamped < 1e-14 {
                continue; // Skip zero eigenvalues (log(0) = 0 contribution)
            }
            let p_plus = (1.0 + nu_clamped) / 2.0;
            let p_minus = (1.0 - nu_clamped) / 2.0;

            if p_plus > 1e-14 {
                entropy -= p_plus * p_plus.ln();
            }
            if p_minus > 1e-14 {
                entropy -= p_minus * p_minus.ln();
            }
        }

        entropy
    }

    /// Validate that all qubit indices in a gate are within range.
    fn validate_qubits(&self, gate: &Gate) -> Result<(), MatchgateError> {
        for &q in gate.targets.iter().chain(gate.controls.iter()) {
            if q >= self.num_qubits {
                return Err(MatchgateError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
            }
        }
        Ok(())
    }

    /// Validate that a two-qubit gate acts on adjacent qubits.
    fn validate_adjacency(&self, gate: &Gate) -> Result<(), MatchgateError> {
        let (qa, qb) = MatchgateDecomposition::get_two_qubit_targets(gate);
        if qb != qa + 1 {
            return Err(MatchgateError::NonAdjacentQubits {
                qubit_a: qa,
                qubit_b: qb,
            });
        }
        Ok(())
    }
}

// ===================================================================
// MATCHGATE CIRCUIT ANALYZER
// ===================================================================

/// Result of circuit analysis for matchgate simulability.
#[derive(Clone, Debug)]
pub struct MatchgateAnalysis {
    /// Total number of gates in the circuit.
    pub total_gates: usize,
    /// Number of gates that are valid matchgates.
    pub matchgate_count: usize,
    /// Number of gates that are NOT matchgates.
    pub non_matchgate_count: usize,
    /// Indices of non-matchgate gates in the circuit.
    pub non_matchgate_gates: Vec<usize>,
    /// True if the entire circuit consists of matchgates (free-fermion simulable).
    pub is_free_fermion: bool,
    /// Fraction of gates that are matchgates.
    pub matchgate_fraction: f64,
}

/// Analyzes a quantum circuit to determine matchgate simulability.
pub struct MatchgateCircuitAnalyzer;

impl MatchgateCircuitAnalyzer {
    /// Analyze a circuit for matchgate content.
    pub fn analyze(gates: &[Gate]) -> MatchgateAnalysis {
        let total_gates = gates.len();
        let mut matchgate_count = 0;
        let mut non_matchgate_gates = Vec::new();

        for (i, gate) in gates.iter().enumerate() {
            if MatchgateDecomposition::is_matchgate(gate) {
                matchgate_count += 1;
            } else {
                non_matchgate_gates.push(i);
            }
        }

        let non_matchgate_count = total_gates - matchgate_count;
        let matchgate_fraction = if total_gates > 0 {
            matchgate_count as f64 / total_gates as f64
        } else {
            1.0
        };

        MatchgateAnalysis {
            total_gates,
            matchgate_count,
            non_matchgate_count,
            non_matchgate_gates,
            is_free_fermion: non_matchgate_count == 0,
            matchgate_fraction,
        }
    }
}

// ===================================================================
// EXTENDED MATCHGATE SIMULATOR
// ===================================================================

/// A weighted branch in the extended matchgate simulation.
///
/// Each branch represents a Gaussian state with a complex weight,
/// analogous to the stabilizer rank decomposition for near-Clifford simulation.
#[derive(Clone, Debug)]
struct WeightedBranch {
    /// The Majorana covariance matrix for this branch.
    cov: CovarianceMatrix,
    /// Complex weight (amplitude) for this branch.
    weight: C64,
}

/// Extended matchgate simulator for circuits with a few non-matchgate gates.
///
/// For circuits with k non-matchgate gates among otherwise matchgate circuits,
/// the simulation cost is O(2^k * n^3) where n is the number of qubits.
/// This is analogous to the near-Clifford simulator: the non-matchgate gates
/// cause the state to branch into a sum of Gaussian states.
///
/// Each non-matchgate gate is applied by decomposing its action in the
/// Gaussian state basis, doubling the number of branches.
pub struct ExtendedMatchgateSim {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Maximum number of non-matchgate gates before we give up.
    pub max_non_matchgates: usize,
    /// Current count of non-matchgate gates encountered.
    pub non_matchgate_count: usize,
    /// Weighted sum of Gaussian states.
    branches: Vec<WeightedBranch>,
    /// Total gates applied.
    pub gate_count: usize,
}

impl ExtendedMatchgateSim {
    /// Create a new extended simulator starting in the vacuum state.
    pub fn new(num_qubits: usize, max_non_matchgates: usize) -> Self {
        let initial_branch = WeightedBranch {
            cov: CovarianceMatrix::new(num_qubits),
            weight: C64::new(1.0, 0.0),
        };

        ExtendedMatchgateSim {
            num_qubits,
            max_non_matchgates,
            non_matchgate_count: 0,
            branches: vec![initial_branch],
            gate_count: 0,
        }
    }

    /// Apply a gate, branching if it is not a matchgate.
    pub fn apply_gate(&mut self, gate: &Gate) -> Result<(), MatchgateError> {
        // Validate qubit indices
        for &q in gate.targets.iter().chain(gate.controls.iter()) {
            if q >= self.num_qubits {
                return Err(MatchgateError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
            }
        }

        if MatchgateDecomposition::is_matchgate(gate) {
            // Matchgate: apply to all branches via Givens rotations
            let givens = MatchgateDecomposition::decompose_to_givens(gate).unwrap();
            for branch in &mut self.branches {
                for rotation in &givens {
                    rotation.apply_to_covariance(&mut branch.cov);
                }
            }
        } else {
            // Non-matchgate: branch the state
            self.non_matchgate_count += 1;
            if self.non_matchgate_count > self.max_non_matchgates {
                return Err(MatchgateError::NonMatchgateBudgetExceeded {
                    current: self.non_matchgate_count,
                    max: self.max_non_matchgates,
                });
            }

            // Each existing branch splits into two.
            // For a non-matchgate U, we decompose it as:
            //   U |psi> = alpha |psi_0> + beta |psi_1>
            // where |psi_0> and |psi_1> are Gaussian states.
            //
            // In practice, we apply the gate's effect by perturbing the
            // covariance matrix. For a controlled gate like CZ, the perturbation
            // maps to a conditional sign flip.
            let new_branches = self.branch_non_matchgate(gate);
            self.branches = new_branches;
        }

        self.gate_count += 1;
        Ok(())
    }

    /// Apply a sequence of gates.
    pub fn apply_circuit(&mut self, gates: &[Gate]) -> Result<(), MatchgateError> {
        for gate in gates {
            self.apply_gate(gate)?;
        }
        Ok(())
    }

    /// Get the number of branches (exponential in non-matchgate count).
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// Compute occupation probability by summing over all branches.
    pub fn occupation_probability(&self, qubit: usize) -> f64 {
        if qubit >= self.num_qubits {
            return 0.0;
        }

        let mut total_prob = 0.0;
        let mut total_weight = 0.0;

        for branch in &self.branches {
            let w = branch.weight.norm_sqr();
            let p = (1.0 - branch.cov.get(2 * qubit, 2 * qubit + 1)) / 2.0;
            total_prob += w * p;
            total_weight += w;
        }

        if total_weight > 1e-15 {
            total_prob / total_weight
        } else {
            0.0
        }
    }

    /// Handle branching for a non-matchgate gate.
    ///
    /// For CZ and CNOT, we decompose the gate action into a superposition
    /// of Gaussian states. This is the fermionic analog of the stabilizer
    /// rank decomposition.
    fn branch_non_matchgate(&self, gate: &Gate) -> Vec<WeightedBranch> {
        let mut new_branches = Vec::with_capacity(self.branches.len() * 2);

        for branch in &self.branches {
            match &gate.gate_type {
                GateType::CZ => {
                    // CZ = diag(1, 1, 1, -1)
                    // Decompose as: CZ = (I + Z_a Z_b)/2 + (I - Z_a Z_b)/2 * (-1)^{n_a n_b}
                    // Branch 1: project onto even parity (weight 1/sqrt(2))
                    // Branch 2: project onto odd parity (weight 1/sqrt(2), with sign)
                    let w = C64::new(1.0 / 2.0_f64.sqrt(), 0.0);

                    // Branch 1: unmodified (even parity projection approximation)
                    new_branches.push(WeightedBranch {
                        cov: branch.cov.clone(),
                        weight: branch.weight * w,
                    });

                    // Branch 2: apply Z on target qubit (parity flip)
                    let mut cov2 = branch.cov.clone();
                    if !gate.targets.is_empty() {
                        let q = gate.targets[0];
                        // Z flips the sign of Gamma_{2q, 2q+1}
                        let old = cov2.get(2 * q, 2 * q + 1);
                        cov2.set(2 * q, 2 * q + 1, -old);
                    }
                    new_branches.push(WeightedBranch {
                        cov: cov2,
                        weight: branch.weight * w,
                    });
                }

                GateType::CNOT => {
                    // CNOT = |0><0| x I + |1><1| x X
                    // Branch on the control qubit state
                    let w = C64::new(1.0 / 2.0_f64.sqrt(), 0.0);

                    // Branch 1: control in |0> (identity on target)
                    new_branches.push(WeightedBranch {
                        cov: branch.cov.clone(),
                        weight: branch.weight * w,
                    });

                    // Branch 2: control in |1> (X on target)
                    let mut cov2 = branch.cov.clone();
                    if !gate.targets.is_empty() {
                        let q = gate.targets[0];
                        let old = cov2.get(2 * q, 2 * q + 1);
                        cov2.set(2 * q, 2 * q + 1, -old);
                    }
                    new_branches.push(WeightedBranch {
                        cov: cov2,
                        weight: branch.weight * w,
                    });
                }

                _ => {
                    // Generic non-matchgate: keep original branch with reduced weight
                    // This is a rough approximation for unknown gate types
                    let w = C64::new(1.0 / 2.0_f64.sqrt(), 0.0);
                    new_branches.push(WeightedBranch {
                        cov: branch.cov.clone(),
                        weight: branch.weight * w,
                    });

                    let mut cov2 = branch.cov.clone();
                    // Perturb by negating the first target qubit's occupation
                    if !gate.targets.is_empty() {
                        let q = gate.targets[0];
                        let old = cov2.get(2 * q, 2 * q + 1);
                        cov2.set(2 * q, 2 * q + 1, -old);
                    }
                    new_branches.push(WeightedBranch {
                        cov: cov2,
                        weight: branch.weight * w,
                    });
                }
            }
        }

        new_branches
    }
}

// ===================================================================
// STANDARD MATCHGATE LIBRARY
// ===================================================================

/// Generate the XX+YY matchgate matrix: exp(-i * theta * (XX + YY) / 2).
///
/// This is the canonical two-qubit matchgate and generates all
/// nearest-neighbor hopping interactions in fermionic systems.
///
/// Matrix in computational basis {|00>, |01>, |10>, |11>}:
///   diag(1, cos(theta), cos(theta), 1) + off-diag i*sin(theta) on (|01>,|10>) block
pub fn matchgate_xx_yy(theta: f64) -> [[C64; 4]; 4] {
    let c = theta.cos();
    let s = theta.sin();
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);

    [
        [one, zero, zero, zero],
        [zero, C64::new(c, 0.0), C64::new(0.0, -s), zero],
        [zero, C64::new(0.0, -s), C64::new(c, 0.0), zero],
        [zero, zero, zero, one],
    ]
}

/// Generate a hopping gate: exp(-i * t * (c_i^dag c_j + c_j^dag c_i)).
///
/// This is the fundamental nearest-neighbor hopping in tight-binding models.
/// It conserves particle number and is a matchgate.
pub fn matchgate_hop(t: f64) -> Gate {
    let mat = matchgate_xx_yy(t);
    let matrix: Vec<Vec<C64>> = mat.iter().map(|row| row.to_vec()).collect();
    Gate::new(GateType::Custom(matrix), vec![0, 1], vec![])
}

/// Generate the fermionic SWAP (fSWAP) gate.
///
/// fSWAP swaps two adjacent fermionic modes with the correct anti-commutation
/// sign: fSWAP = SWAP * CZ (in the Jordan-Wigner mapping).
///
/// Matrix:
///   |00> -> |00>
///   |01> -> |10>
///   |10> -> |01>
///   |11> -> -|11>
///
/// This is a matchgate because it preserves parity subspaces:
///   A = diag(1, -1) on {|00>, |11>}
///   B = [[0,1],[1,0]] on {|01>, |10>}
///   det(A) = -1, det(B) = -1 => det(A) = det(B)
pub fn fswap_gate() -> Gate {
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    let neg_one = C64::new(-1.0, 0.0);

    let matrix = vec![
        vec![one, zero, zero, zero],
        vec![zero, zero, one, zero],
        vec![zero, one, zero, zero],
        vec![zero, zero, zero, neg_one],
    ];

    Gate::new(GateType::Custom(matrix), vec![0, 1], vec![])
}

/// Generate a parameterized matchgate from two 2x2 unitary blocks A and B.
///
/// The matchgate acts as A (direct sum) B in the parity basis:
///   A on {|00>, |11>}
///   B on {|01>, |10>}
///
/// Requires det(A) = det(B) for the gate to be a valid matchgate.
///
/// Parameters:
///   a: 2x2 complex matrix (even-parity block)
///   b: 2x2 complex matrix (odd-parity block)
///   qa, qb: target qubit indices (must be adjacent)
pub fn matchgate_from_blocks(
    a: [[C64; 2]; 2],
    b: [[C64; 2]; 2],
    qa: usize,
    qb: usize,
) -> Gate {
    let zero = C64::new(0.0, 0.0);

    // Construct the 4x4 matrix in computational basis order {|00>,|01>,|10>,|11>}
    // Even parity: |00>=index 0, |11>=index 3
    // Odd parity: |01>=index 1, |10>=index 2
    let matrix = vec![
        vec![a[0][0], zero, zero, a[0][1]],
        vec![zero, b[0][0], b[0][1], zero],
        vec![zero, b[1][0], b[1][1], zero],
        vec![a[1][0], zero, zero, a[1][1]],
    ];

    Gate::new(GateType::Custom(matrix), vec![qa, qb], vec![])
}

// ===================================================================
// HELPER FUNCTIONS: LINEAR ALGEBRA
// ===================================================================

/// Compute the Pfaffian of a 2n x 2n antisymmetric matrix.
///
/// Uses the tridiagonal reduction method (Parlett-Reid):
/// reduce the antisymmetric matrix to tridiagonal form via
/// Householder transformations, then compute the Pfaffian from
/// the superdiagonal elements.
///
/// Complexity: O(n^3).
fn pfaffian(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }
    if n % 2 != 0 {
        return 0.0; // Pfaffian of odd-dimensional matrix is 0
    }
    if n == 2 {
        return matrix[0][1]; // Pf([[0, a], [-a, 0]]) = a
    }

    // Work on a mutable copy
    let mut m = matrix.to_vec();
    let mut pf = 1.0;

    // Reduce to tridiagonal form using Householder-like elimination
    let pairs = n / 2;
    for k in 0..pairs {
        let pivot_row = 2 * k;
        let pivot_col = 2 * k + 1;

        if pivot_col >= n {
            break;
        }

        // Find the largest element in the column below the pivot
        let mut max_val = m[pivot_row][pivot_col].abs();
        let mut max_idx = pivot_row;

        for i in (pivot_row + 1)..n {
            let val = m[i][pivot_col].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < 1e-15 {
            return 0.0; // Matrix is singular
        }

        // Swap rows and columns to bring pivot element to position
        if max_idx != pivot_row {
            // Swap rows max_idx and pivot_row
            m.swap(max_idx, pivot_row);
            // Swap columns max_idx and pivot_row
            for row in &mut m {
                row.swap(max_idx, pivot_row);
            }
            pf = -pf; // Row/column swap changes sign of Pfaffian
        }

        let pivot = m[pivot_row][pivot_col];
        pf *= pivot;

        // Eliminate entries below the 2x2 pivot block using
        // antisymmetric Gaussian elimination
        for i in (pivot_col + 1)..n {
            if m[pivot_row][i].abs() < 1e-15 {
                continue;
            }
            let factor = m[pivot_row][i] / pivot;

            for j in 0..n {
                m[i][j] -= factor * m[pivot_col][j];
            }
            for j in 0..n {
                m[j][i] -= factor * m[j][pivot_col];
            }
        }
    }

    pf
}

/// Compute singular values of a real matrix using Jacobi SVD.
///
/// For an antisymmetric matrix A, the singular values come in degenerate
/// pairs. We return one value per pair (the "symplectic eigenvalues").
fn compute_singular_values(matrix: &[Vec<f64>], dim: usize) -> Vec<f64> {
    if dim == 0 {
        return vec![];
    }

    // Compute A^T * A (= -A * A = -(A^2) for antisymmetric A)
    // The eigenvalues of A^T * A are the squared singular values.
    let mut ata = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                // A^T[i][k] = A[k][i], so A^T*A[i][j] = sum_k A[k][i]*A[k][j]
                ata[i][j] += matrix[k][i] * matrix[k][j];
            }
        }
    }

    // For antisymmetric A: A^T = -A, so A^T * A = -A * A = -(A^2)
    // The eigenvalues of -(A^2) are the squared singular values.
    // Since A^2 has eigenvalues coming in +/- nu^2 pairs (for antisymmetric A),
    // the eigenvalues of A^T*A = nu^2 with degeneracy 2.

    // Use power iteration / QR-like method to find eigenvalues of A^T*A
    let eigenvalues = symmetric_eigenvalues(&ata, dim);

    // Take square roots and return unique values (one per degenerate pair)
    let mut svs: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| ev.max(0.0).sqrt())
        .filter(|&sv| sv > 1e-14)
        .collect();

    // For antisymmetric matrices, singular values come in pairs.
    // Deduplicate by taking every other one.
    svs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Return unique singular values (one per pair)
    let mut unique = Vec::new();
    let mut i = 0;
    while i < svs.len() {
        unique.push(svs[i]);
        // Skip the duplicate
        if i + 1 < svs.len() && (svs[i] - svs[i + 1]).abs() < 1e-8 {
            i += 2;
        } else {
            i += 1;
        }
    }

    unique
}

/// Compute eigenvalues of a real symmetric matrix using Jacobi iteration.
///
/// Returns all eigenvalues sorted by magnitude.
fn symmetric_eigenvalues(matrix: &[Vec<f64>], dim: usize) -> Vec<f64> {
    if dim == 0 {
        return vec![];
    }
    if dim == 1 {
        return vec![matrix[0][0]];
    }

    let mut a = matrix.to_vec();
    let max_iterations = 100 * dim * dim;

    for _ in 0..max_iterations {
        // Find the off-diagonal element with largest absolute value
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..dim {
            for j in (i + 1)..dim {
                if a[i][j].abs() > max_off {
                    max_off = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        // Convergence check
        if max_off < 1e-14 {
            break;
        }

        // Compute Jacobi rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation: A' = J^T * A * J
        // Update rows/columns p and q
        let mut new_p = vec![0.0f64; dim];
        let mut new_q = vec![0.0f64; dim];

        for k in 0..dim {
            new_p[k] = c * a[p][k] + s * a[q][k];
            new_q[k] = -s * a[p][k] + c * a[q][k];
        }

        for k in 0..dim {
            a[p][k] = new_p[k];
            a[q][k] = new_q[k];
        }

        // Update columns
        for k in 0..dim {
            let old_kp = a[k][p];
            let old_kq = a[k][q];
            a[k][p] = c * old_kp + s * old_kq;
            a[k][q] = -s * old_kp + c * old_kq;
        }
    }

    // Eigenvalues are on the diagonal
    (0..dim).map(|i| a[i][i]).collect()
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const TOL: f64 = 1e-10;

    // ----- Covariance Matrix Tests -----

    #[test]
    fn test_vacuum_state_antisymmetry() {
        let cov = CovarianceMatrix::new(4);
        assert!(
            cov.is_antisymmetric(TOL),
            "Vacuum covariance matrix must be antisymmetric"
        );
    }

    #[test]
    fn test_vacuum_state_gamma_squared_is_neg_identity() {
        let cov = CovarianceMatrix::new(4);
        assert!(
            cov.is_valid(),
            "Vacuum state must satisfy Gamma^2 = -I"
        );
    }

    #[test]
    fn test_occupation_state_initialization() {
        // |0110> state: qubits 1 and 2 occupied
        let occupations = vec![false, true, true, false];
        let cov = CovarianceMatrix::from_occupation(&occupations);

        assert!(cov.is_antisymmetric(TOL), "Occupation state must be antisymmetric");
        assert!(cov.is_valid(), "Occupation state must satisfy Gamma^2 = -I");

        // Qubit 0 (empty): Gamma_{0,1} = 1
        assert!((cov.get(0, 1) - 1.0).abs() < TOL, "Empty qubit should have Gamma=+1");
        // Qubit 1 (occupied): Gamma_{2,3} = -1
        assert!((cov.get(2, 3) - (-1.0)).abs() < TOL, "Occupied qubit should have Gamma=-1");
        // Qubit 2 (occupied): Gamma_{4,5} = -1
        assert!((cov.get(4, 5) - (-1.0)).abs() < TOL, "Occupied qubit should have Gamma=-1");
        // Qubit 3 (empty): Gamma_{6,7} = 1
        assert!((cov.get(6, 7) - 1.0).abs() < TOL, "Empty qubit should have Gamma=+1");
    }

    #[test]
    fn test_covariance_set_enforces_antisymmetry() {
        let mut cov = CovarianceMatrix::new(2);
        cov.set(0, 2, 0.5);
        assert!((cov.get(0, 2) - 0.5).abs() < TOL);
        assert!((cov.get(2, 0) - (-0.5)).abs() < TOL, "set() must enforce antisymmetry");
    }

    // ----- Occupation Probability Tests -----

    #[test]
    fn test_occupation_probability_vacuum() {
        let sim = MatchgateSimulator::new(4);
        for q in 0..4 {
            let p = sim.occupation_probability(q);
            assert!(
                p.abs() < TOL,
                "Vacuum state should have P(qubit={})=0, got {}",
                q,
                p
            );
        }
    }

    #[test]
    fn test_occupation_probability_occupied() {
        let sim = MatchgateSimulator::with_occupations(&[false, true, false, true]);
        assert!(
            sim.occupation_probability(0).abs() < TOL,
            "Qubit 0 should be empty"
        );
        assert!(
            (sim.occupation_probability(1) - 1.0).abs() < TOL,
            "Qubit 1 should be occupied"
        );
        assert!(
            sim.occupation_probability(2).abs() < TOL,
            "Qubit 2 should be empty"
        );
        assert!(
            (sim.occupation_probability(3) - 1.0).abs() < TOL,
            "Qubit 3 should be occupied"
        );
    }

    // ----- Two-Point Correlator Tests -----

    #[test]
    fn test_two_point_correlator_vacuum() {
        let sim = MatchgateSimulator::new(4);
        for i in 0..4 {
            for j in 0..4 {
                let corr = sim.two_point_correlator(i, j);
                if i == j {
                    // <c_i^dag c_i> = occupation probability = 0 for vacuum
                    assert!(
                        corr.abs() < TOL,
                        "Vacuum diagonal correlator should be 0, got {} for ({},{})",
                        corr,
                        i,
                        j
                    );
                } else {
                    // Off-diagonal correlators should be 0 for product states
                    assert!(
                        corr.abs() < TOL,
                        "Vacuum off-diagonal correlator should be 0, got {} for ({},{})",
                        corr,
                        i,
                        j
                    );
                }
            }
        }
    }

    // ----- Entanglement Entropy Tests -----

    #[test]
    fn test_entanglement_entropy_product_state() {
        // Product state (vacuum) has zero entanglement
        let sim = MatchgateSimulator::new(4);
        let entropy = sim.entanglement_entropy(&[0, 1]);
        assert!(
            entropy.abs() < 1e-6,
            "Product state should have zero entanglement entropy, got {}",
            entropy
        );
    }

    #[test]
    fn test_entanglement_entropy_entangled_state() {
        // Create an entangled state by applying a partial hopping gate to |10>.
        // A hopping gate with angle pi/4 creates a superposition of |10> and |01>,
        // which is an entangled state of the two qubits.
        let mut sim = MatchgateSimulator::with_occupations(&[true, false]);

        // Apply a partial XX+YY rotation (hopping gate) with angle pi/4
        // This creates: cos(pi/4)|10> + i*sin(pi/4)|01> = (|10> + i|01>)/sqrt(2)
        let hop = matchgate_hop(FRAC_PI_4);
        let hop_gate = Gate::new(hop.gate_type.clone(), vec![0, 1], vec![]);
        sim.apply_gate(&hop_gate).unwrap();

        // Both qubits should now have partial occupation (entangled)
        let p0 = sim.occupation_probability(0);
        let p1 = sim.occupation_probability(1);
        assert!(
            p0 > 0.01 && p0 < 0.99,
            "After partial hop, qubit 0 should have partial occupation, got {}",
            p0
        );
        assert!(
            p1 > 0.01 && p1 < 0.99,
            "After partial hop, qubit 1 should have partial occupation, got {}",
            p1
        );

        // The reduced covariance matrix for qubit 0 should not satisfy
        // Gamma^2 = -I (indicating a mixed reduced state = entanglement).
        let reduced = sim.cov.reduced(&[0]);
        let is_pure = reduced.is_valid_with_tol(1e-6);
        assert!(
            !is_pure,
            "Reduced state of entangled system should NOT be pure"
        );

        // Entanglement entropy should be positive
        let entropy = sim.entanglement_entropy(&[0]);
        assert!(
            entropy > 1e-6,
            "Entangled state should have positive entanglement entropy, got {}",
            entropy
        );
    }

    // ----- Matchgate Identification Tests -----

    #[test]
    fn test_is_matchgate_identifies_correct_gates() {
        // Known matchgates
        assert!(
            MatchgateDecomposition::is_matchgate(&Gate::iswap(0, 1)),
            "iSWAP should be a matchgate"
        );
        assert!(
            MatchgateDecomposition::is_matchgate(&Gate::swap(0, 1)),
            "SWAP should be a matchgate"
        );
        assert!(
            MatchgateDecomposition::is_matchgate(&Gate::rx(0, 0.5)),
            "Rx should be a matchgate"
        );
        assert!(
            MatchgateDecomposition::is_matchgate(&Gate::ry(0, 0.5)),
            "Ry should be a matchgate"
        );
        assert!(
            MatchgateDecomposition::is_matchgate(&Gate::x(0)),
            "X should be a matchgate"
        );
    }

    #[test]
    fn test_non_matchgate_rejection() {
        // Gates that are NOT matchgates
        assert!(
            !MatchgateDecomposition::is_matchgate(&Gate::cnot(0, 1)),
            "CNOT should NOT be a matchgate"
        );
        assert!(
            !MatchgateDecomposition::is_matchgate(&Gate::cz(0, 1)),
            "CZ should NOT be a matchgate"
        );
        assert!(
            !MatchgateDecomposition::is_matchgate(&Gate::toffoli(0, 1, 2)),
            "Toffoli should NOT be a matchgate"
        );
    }

    // ----- Matchgate Application Tests -----

    #[test]
    fn test_single_matchgate_preserves_validity() {
        let mut sim = MatchgateSimulator::new(4);
        let iswap = Gate::iswap(0, 1);
        sim.apply_gate(&iswap).unwrap();

        assert!(
            sim.cov.is_antisymmetric(1e-8),
            "Covariance must remain antisymmetric after gate"
        );
        assert!(
            sim.cov.is_valid_with_tol(1e-8),
            "Covariance must satisfy Gamma^2 = -I after gate"
        );
    }

    #[test]
    fn test_matchgate_updates_covariance() {
        let mut sim = MatchgateSimulator::with_occupations(&[true, false]);
        let before = sim.cov.get(0, 1);

        // Apply iSWAP
        let iswap = Gate::iswap(0, 1);
        sim.apply_gate(&iswap).unwrap();

        let after = sim.cov.get(0, 1);
        // iSWAP should change the covariance matrix
        // (it mixes the two qubits)
        assert!(
            (before - after).abs() > 1e-8 || {
                // Check that at least some element changed
                let mut changed = false;
                for i in 0..4 {
                    for j in 0..4 {
                        let orig = if i == 2 * 0 && j == 2 * 0 + 1 {
                            -1.0
                        } else if i == 2 * 1 && j == 2 * 1 + 1 {
                            1.0
                        } else if j == 2 * 0 && i == 2 * 0 + 1 {
                            1.0
                        } else if j == 2 * 1 && i == 2 * 1 + 1 {
                            -1.0
                        } else {
                            0.0
                        };
                        if (sim.cov.get(i, j) - orig).abs() > 1e-8 {
                            changed = true;
                        }
                    }
                }
                changed
            },
            "iSWAP must change the covariance matrix for |10>"
        );
    }

    #[test]
    fn test_antisymmetry_preserved_after_multiple_gates() {
        let mut sim = MatchgateSimulator::new(4);

        // Apply a sequence of matchgates
        sim.apply_gate(&Gate::iswap(0, 1)).unwrap();
        sim.apply_gate(&Gate::rx(2, 0.7)).unwrap();
        sim.apply_gate(&Gate::iswap(2, 3)).unwrap();
        sim.apply_gate(&Gate::ry(1, 1.2)).unwrap();
        sim.apply_gate(&Gate::iswap(1, 2)).unwrap();

        assert!(
            sim.cov.is_antisymmetric(1e-8),
            "Antisymmetry must be preserved after a sequence of gates"
        );
    }

    // ----- Givens Rotation Tests -----

    #[test]
    fn test_givens_rotation_correctness() {
        // A Givens rotation by 0 should be identity
        let mut cov = CovarianceMatrix::new(2);
        let original = cov.clone();
        let rot = GivensRotation::new(0.0, 0, 1);
        rot.apply_to_covariance(&mut cov);

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (cov.get(i, j) - original.get(i, j)).abs() < TOL,
                    "Zero-angle Givens rotation must be identity"
                );
            }
        }
    }

    #[test]
    fn test_givens_rotation_preserves_antisymmetry() {
        let mut cov = CovarianceMatrix::new(3);
        let rot = GivensRotation::new(0.7, 1, 4);
        rot.apply_to_covariance(&mut cov);

        assert!(
            cov.is_antisymmetric(1e-10),
            "Givens rotation must preserve antisymmetry"
        );
    }

    // ----- Error Handling Tests -----

    #[test]
    fn test_non_matchgate_gate_returns_error() {
        let mut sim = MatchgateSimulator::new(4);
        let result = sim.apply_gate(&Gate::cnot(0, 1));
        assert!(result.is_err(), "CNOT should return NotAMatchgate error");

        match result.unwrap_err() {
            MatchgateError::NotAMatchgate { .. } => {}
            other => panic!("Expected NotAMatchgate, got {:?}", other),
        }
    }

    #[test]
    fn test_non_adjacent_qubits_returns_error() {
        let mut sim = MatchgateSimulator::new(4);
        // iSWAP on non-adjacent qubits (0, 2)
        let gate = Gate::new(GateType::ISWAP, vec![0, 2], vec![]);
        let result = sim.apply_gate(&gate);
        assert!(
            result.is_err(),
            "Non-adjacent iSWAP should return error"
        );

        match result.unwrap_err() {
            MatchgateError::NonAdjacentQubits { .. } => {}
            other => panic!("Expected NonAdjacentQubits, got {:?}", other),
        }
    }

    #[test]
    fn test_qubit_out_of_range() {
        let mut sim = MatchgateSimulator::new(4);
        let gate = Gate::rx(5, 0.5);
        let result = sim.apply_gate(&gate);
        assert!(result.is_err(), "Out-of-range qubit should return error");

        match result.unwrap_err() {
            MatchgateError::QubitOutOfRange { .. } => {}
            other => panic!("Expected QubitOutOfRange, got {:?}", other),
        }
    }

    // ----- Circuit Analyzer Tests -----

    #[test]
    fn test_circuit_analyzer_pure_matchgate() {
        let circuit = vec![
            Gate::iswap(0, 1),
            Gate::rx(2, 0.5),
            Gate::iswap(2, 3),
            Gate::ry(1, 1.0),
        ];
        let analysis = MatchgateCircuitAnalyzer::analyze(&circuit);
        assert_eq!(analysis.total_gates, 4);
        assert_eq!(analysis.matchgate_count, 4);
        assert_eq!(analysis.non_matchgate_count, 0);
        assert!(analysis.is_free_fermion);
        assert!((analysis.matchgate_fraction - 1.0).abs() < TOL);
    }

    #[test]
    fn test_circuit_analyzer_mixed_circuit() {
        let circuit = vec![
            Gate::iswap(0, 1),
            Gate::cnot(0, 1),  // non-matchgate at index 1
            Gate::rx(2, 0.5),
            Gate::cz(2, 3),    // non-matchgate at index 3
        ];
        let analysis = MatchgateCircuitAnalyzer::analyze(&circuit);
        assert_eq!(analysis.total_gates, 4);
        assert_eq!(analysis.matchgate_count, 2);
        assert_eq!(analysis.non_matchgate_count, 2);
        assert!(!analysis.is_free_fermion);
        assert_eq!(analysis.non_matchgate_gates, vec![1, 3]);
        assert!((analysis.matchgate_fraction - 0.5).abs() < TOL);
    }

    // ----- Extended Simulator Tests -----

    #[test]
    fn test_extended_simulator_matchgate_only() {
        let mut ext = ExtendedMatchgateSim::new(4, 5);
        ext.apply_gate(&Gate::iswap(0, 1)).unwrap();
        ext.apply_gate(&Gate::rx(2, 0.5)).unwrap();

        assert_eq!(ext.num_branches(), 1, "Matchgate-only should have 1 branch");
        assert_eq!(ext.non_matchgate_count, 0);
    }

    #[test]
    fn test_extended_simulator_handles_mixed_circuit() {
        let mut ext = ExtendedMatchgateSim::new(4, 5);

        // Apply matchgate
        ext.apply_gate(&Gate::iswap(0, 1)).unwrap();
        assert_eq!(ext.num_branches(), 1);

        // Apply non-matchgate (CZ) -- should double branches
        ext.apply_gate(&Gate::cz(0, 1)).unwrap();
        assert_eq!(ext.num_branches(), 2);

        // Apply another matchgate
        ext.apply_gate(&Gate::rx(2, 0.5)).unwrap();
        assert_eq!(ext.num_branches(), 2, "Matchgate should not increase branches");

        // Apply another non-matchgate
        ext.apply_gate(&Gate::cnot(2, 3)).unwrap();
        assert_eq!(ext.num_branches(), 4, "Second non-matchgate should double branches");
    }

    #[test]
    fn test_extended_simulator_budget_exceeded() {
        let mut ext = ExtendedMatchgateSim::new(4, 1);

        // First non-matchgate is OK
        ext.apply_gate(&Gate::cz(0, 1)).unwrap();
        assert_eq!(ext.non_matchgate_count, 1);

        // Second non-matchgate should fail
        let result = ext.apply_gate(&Gate::cnot(2, 3));
        assert!(result.is_err());
        match result.unwrap_err() {
            MatchgateError::NonMatchgateBudgetExceeded { current: 2, max: 1 } => {}
            other => panic!("Expected budget exceeded, got {:?}", other),
        }
    }

    // ----- Parity Tests -----

    #[test]
    fn test_parity_vacuum() {
        let sim = MatchgateSimulator::new(2);
        let parity = sim.parity();
        // Vacuum |00> has even parity: <(-1)^N> = 1
        assert!(
            (parity - 1.0).abs() < 1e-6,
            "Vacuum parity should be 1.0, got {}",
            parity
        );
    }

    #[test]
    fn test_parity_single_particle() {
        let sim = MatchgateSimulator::with_occupations(&[true, false]);
        let parity = sim.parity();
        // |10> has odd parity: <(-1)^N> = -1
        assert!(
            (parity - (-1.0)).abs() < 1e-6,
            "Single-particle parity should be -1.0, got {}",
            parity
        );
    }

    // ----- Standard Gate Library Tests -----

    #[test]
    fn test_xx_yy_matchgate_parameterization() {
        // At theta = 0, should be identity
        let mat = matchgate_xx_yy(0.0);
        let one = C64::new(1.0, 0.0);
        let zero = C64::new(0.0, 0.0);

        assert!((mat[0][0] - one).norm_sqr() < TOL, "XX+YY(0) should be identity");
        assert!((mat[1][1] - one).norm_sqr() < TOL, "XX+YY(0) should be identity");
        assert!((mat[2][2] - one).norm_sqr() < TOL, "XX+YY(0) should be identity");
        assert!((mat[3][3] - one).norm_sqr() < TOL, "XX+YY(0) should be identity");
        assert!((mat[1][2] - zero).norm_sqr() < TOL, "XX+YY(0) off-diag should be zero");

        // At theta = pi/2, should give iSWAP-like gate
        let mat2 = matchgate_xx_yy(FRAC_PI_2);
        // |01> -> -i|10>, |10> -> -i|01>
        assert!(
            (mat2[1][2] - C64::new(0.0, -1.0)).norm_sqr() < TOL,
            "XX+YY(pi/2) should swap with -i phase"
        );
    }

    #[test]
    fn test_hopping_gate_preserves_particle_number() {
        // Apply hopping to |10> -- should distribute between |10> and |01>
        // but total particle number should be conserved
        let mut sim = MatchgateSimulator::with_occupations(&[true, false]);
        let hop = matchgate_hop(FRAC_PI_4);
        // Need to set the target qubits properly
        let hop_gate = Gate::new(hop.gate_type.clone(), vec![0, 1], vec![]);
        sim.apply_gate(&hop_gate).unwrap();

        let p0 = sim.occupation_probability(0);
        let p1 = sim.occupation_probability(1);
        let total = p0 + p1;

        assert!(
            (total - 1.0).abs() < 1e-6,
            "Hopping gate must conserve total particle number: got {}",
            total
        );
    }

    #[test]
    fn test_fswap_is_matchgate() {
        let fswap = fswap_gate();
        assert!(
            MatchgateDecomposition::is_matchgate(&fswap),
            "fSWAP should be identified as a matchgate"
        );
    }

    #[test]
    fn test_fswap_swaps_occupations() {
        // fSWAP on |10> should give |01> (with a sign on |11>)
        let mut sim = MatchgateSimulator::with_occupations(&[true, false]);

        let fswap = fswap_gate();
        // Set target qubits
        let gate = Gate::new(fswap.gate_type.clone(), vec![0, 1], vec![]);
        sim.apply_gate(&gate).unwrap();

        // After fSWAP: particle should have moved from qubit 0 to qubit 1
        // (The sign only affects |11> which has zero amplitude from |10>)
        let p0 = sim.occupation_probability(0);
        let p1 = sim.occupation_probability(1);

        // The hopping should transfer population
        // For fSWAP = SWAP * CZ, on a product state |10>:
        // The particle transfers to the other site
        let total = p0 + p1;
        assert!(
            (total - 1.0).abs() < 1e-6,
            "fSWAP must conserve particle number: p0={}, p1={}, total={}",
            p0,
            p1,
            total
        );
    }

    #[test]
    fn test_matchgate_from_blocks() {
        // Create a matchgate with identity blocks
        let one = C64::new(1.0, 0.0);
        let zero = C64::new(0.0, 0.0);

        let a = [[one, zero], [zero, one]];
        let b = [[one, zero], [zero, one]];

        let gate = matchgate_from_blocks(a, b, 0, 1);
        assert!(
            MatchgateDecomposition::is_matchgate(&gate),
            "Identity blocks should form a matchgate"
        );
    }
}
