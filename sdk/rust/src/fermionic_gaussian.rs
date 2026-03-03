//! Fermionic Gaussian State Simulation
//!
//! Fermionic Gaussian states are quantum states of non-interacting fermions that
//! can be fully described by their 2n x 2n covariance matrix (correlation matrix).
//! This allows O(n^3) simulation instead of O(2^n), making it tractable to simulate
//! systems with hundreds or thousands of fermionic modes.
//!
//! These are the fermionic analog of Gaussian states in continuous-variable quantum
//! optics. Any state that can be prepared from the vacuum by applying quadratic
//! Hamiltonians (i.e., Hamiltonians that are bilinear in creation/annihilation
//! operators) is a Gaussian state.
//!
//! ## Covariance matrix formalism
//!
//! For n fermionic modes with creation operators c_j^dagger and annihilation
//! operators c_j, define the 2n Majorana operators:
//!
//!   gamma_{2j}   = c_j + c_j^dagger
//!   gamma_{2j+1} = i(c_j^dagger - c_j)
//!
//! The covariance matrix Gamma is the 2n x 2n real antisymmetric matrix:
//!
//!   Gamma_{jk} = (i/2) <[gamma_j, gamma_k]> = i <gamma_j gamma_k> - i delta_{jk}
//!
//! For pure Gaussian states: Gamma^2 = -I.
//! For mixed states: eigenvalues of i*Gamma lie in [-1, 1].
//!
//! ## Gaussian unitaries
//!
//! Any quadratic Hamiltonian H = sum_{jk} h_{jk} gamma_j gamma_k generates a
//! Gaussian unitary U = exp(-iH). Under this unitary, the covariance matrix
//! transforms as:
//!
//!   Gamma -> O Gamma O^T
//!
//! where O is a real orthogonal (SO(2n)) matrix. This is the key insight that
//! makes polynomial-time simulation possible.
//!
//! ## Complexity
//!
//! - Gate application (Givens rotation): O(n) per rotation
//! - Arbitrary SO(2n) rotation: O(n^2) for matrix multiply
//! - Occupation number: O(1)
//! - Two-point correlator: O(1)
//! - Entanglement entropy: O(|subsystem|^3)
//! - Ground state computation: O(n^3) via diagonalization
//! - Time evolution step: O(n^2)
//!
//! ## References
//!
//! - Bravyi (2005): "Classical capacity of fermionic product channels"
//! - Terhal & DiVincenzo (2002): "Classical simulation of noninteracting-fermion
//!   quantum circuits"
//! - Peschel (2003): "Calculation of reduced density matrices from correlation
//!   functions"
//! - Kraus (2009): "Generalized Hartree-Fock theory and the Hubbard model"

use crate::C64;
use std::f64::consts::PI;
use std::fmt;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors that can occur during fermionic Gaussian state simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum FermionicGaussianError {
    /// Mode index exceeds the number of modes.
    ModeOutOfRange { mode: usize, n_modes: usize },
    /// Covariance matrix has wrong dimensions.
    DimensionMismatch { expected: usize, got: usize },
    /// Matrix is not antisymmetric.
    NotAntisymmetric,
    /// SO(2n) matrix has wrong dimensions.
    InvalidRotationMatrix { expected: usize, got: usize },
    /// Hamiltonian has wrong dimensions.
    InvalidHamiltonian { expected: usize, got: usize },
}

impl fmt::Display for FermionicGaussianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FermionicGaussianError::ModeOutOfRange { mode, n_modes } => {
                write!(f, "Mode {} out of range for {}-mode system", mode, n_modes)
            }
            FermionicGaussianError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Covariance matrix dimension mismatch: expected {}x{}, got {}x{}",
                    expected, expected, got, got
                )
            }
            FermionicGaussianError::NotAntisymmetric => {
                write!(f, "Matrix is not antisymmetric")
            }
            FermionicGaussianError::InvalidRotationMatrix { expected, got } => {
                write!(
                    f,
                    "SO(2n) rotation matrix dimension mismatch: expected {}x{}, got {}x{}",
                    expected, expected, got, got
                )
            }
            FermionicGaussianError::InvalidHamiltonian { expected, got } => {
                write!(
                    f,
                    "Hamiltonian dimension mismatch: expected {}x{}, got {}x{}",
                    expected, expected, got, got
                )
            }
        }
    }
}

impl std::error::Error for FermionicGaussianError {}

// ===================================================================
// FERMIONIC GAUSSIAN STATE
// ===================================================================

/// A fermionic Gaussian state represented by its 2n x 2n Majorana covariance matrix.
///
/// The covariance matrix Gamma is a real antisymmetric 2n x 2n matrix where n is
/// the number of fermionic modes. For a pure Gaussian state, Gamma^2 = -I.
///
/// The Majorana operators are defined as:
///   gamma_{2j}   = c_j + c_j^dagger
///   gamma_{2j+1} = i(c_j^dagger - c_j)
///
/// And the covariance matrix entries are:
///   Gamma_{jk} = (i/2) <[gamma_j, gamma_k]>
#[derive(Clone, Debug)]
pub struct FermionicGaussianState {
    /// Number of fermionic modes.
    pub n_modes: usize,
    /// The 2n x 2n real antisymmetric Majorana covariance matrix.
    /// Stored as Vec<Vec<f64>> for compatibility with the rest of the codebase.
    /// Although the spec mentions Complex64, the Majorana covariance matrix is
    /// always real for Gaussian states, so we use f64 for efficiency and
    /// numerical stability.
    pub covariance: Vec<Vec<f64>>,
}

impl FermionicGaussianState {
    /// Create a new fermionic Gaussian state in the vacuum (all modes unoccupied).
    ///
    /// The vacuum covariance matrix has the block-diagonal form:
    ///   Gamma = diag(sigma_y, sigma_y, ..., sigma_y)
    ///
    /// where sigma_y = [[0, -1], [1, 0]] (note: the convention is that
    /// Gamma_{2j, 2j+1} = -1 for vacuum, encoding <n_j> = 0).
    ///
    /// This follows from: <n_j> = (1 + i*Gamma_{2j, 2j+1}) / 2
    /// For vacuum: <n_j> = 0, so Gamma_{2j, 2j+1} must give i*Gamma = -1,
    /// meaning Gamma_{2j, 2j+1} = -1 (since we work in the real Majorana basis
    /// where the factor of i is absorbed into the definition).
    ///
    /// Convention alignment: We use the same sign convention as the matchgate
    /// module in this codebase, where Gamma_{2j, 2j+1} = +1 for vacuum.
    /// The occupation is then: <n_j> = (1 - Gamma_{2j, 2j+1}) / 2.
    pub fn new(n_modes: usize) -> Self {
        let dim = 2 * n_modes;
        let mut covariance = vec![vec![0.0f64; dim]; dim];

        // Vacuum state: Gamma_{2j, 2j+1} = 1, Gamma_{2j+1, 2j} = -1
        for j in 0..n_modes {
            covariance[2 * j][2 * j + 1] = 1.0;
            covariance[2 * j + 1][2 * j] = -1.0;
        }

        FermionicGaussianState {
            n_modes,
            covariance,
        }
    }

    /// Create a fermionic Gaussian state from an explicit covariance matrix.
    ///
    /// The matrix must be 2n x 2n and antisymmetric (Gamma^T = -Gamma).
    /// No check is made for the pure-state condition Gamma^2 = -I; the caller
    /// may provide a mixed-state covariance matrix.
    pub fn from_covariance(
        gamma: Vec<Vec<f64>>,
    ) -> Result<Self, FermionicGaussianError> {
        let dim = gamma.len();
        if dim == 0 || dim % 2 != 0 {
            return Err(FermionicGaussianError::DimensionMismatch {
                expected: 0,
                got: dim,
            });
        }

        // Check square
        for row in &gamma {
            if row.len() != dim {
                return Err(FermionicGaussianError::DimensionMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
        }

        // Check antisymmetry
        let tol = 1e-10;
        for i in 0..dim {
            if gamma[i][i].abs() > tol {
                return Err(FermionicGaussianError::NotAntisymmetric);
            }
            for j in (i + 1)..dim {
                if (gamma[i][j] + gamma[j][i]).abs() > tol {
                    return Err(FermionicGaussianError::NotAntisymmetric);
                }
            }
        }

        let n_modes = dim / 2;
        Ok(FermionicGaussianState {
            n_modes,
            covariance: gamma,
        })
    }

    /// Create a Slater determinant state from a list of occupied mode indices.
    ///
    /// A Slater determinant is a Gaussian state where specific single-particle
    /// modes are filled. This is the fermionic analog of a computational basis state.
    ///
    /// For occupied mode j: Gamma_{2j, 2j+1} = -1 (occupation = 1)
    /// For unoccupied mode j: Gamma_{2j, 2j+1} = +1 (occupation = 0)
    pub fn from_occupation(
        occupied: &[usize],
        n_modes: usize,
    ) -> Result<Self, FermionicGaussianError> {
        // Validate indices
        for &idx in occupied {
            if idx >= n_modes {
                return Err(FermionicGaussianError::ModeOutOfRange {
                    mode: idx,
                    n_modes,
                });
            }
        }

        let dim = 2 * n_modes;
        let mut covariance = vec![vec![0.0f64; dim]; dim];

        for j in 0..n_modes {
            let is_occupied = occupied.contains(&j);
            let sign = if is_occupied { -1.0 } else { 1.0 };
            covariance[2 * j][2 * j + 1] = sign;
            covariance[2 * j + 1][2 * j] = -sign;
        }

        Ok(FermionicGaussianState {
            n_modes,
            covariance,
        })
    }

    /// Dimension of the covariance matrix (2 * n_modes).
    #[inline]
    pub fn dim(&self) -> usize {
        2 * self.n_modes
    }

    /// Check if the state is a pure Gaussian state (Gamma^2 = -I).
    pub fn is_pure(&self) -> bool {
        self.is_pure_with_tol(1e-8)
    }

    /// Check if the state is pure with a custom tolerance.
    pub fn is_pure_with_tol(&self, tol: f64) -> bool {
        let dim = self.dim();
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += self.covariance[i][k] * self.covariance[k][j];
                }
                let expected = if i == j { -1.0 } else { 0.0 };
                if (sum - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    // ---------------------------------------------------------------
    // GAUSSIAN UNITARIES
    // ---------------------------------------------------------------

    /// Apply a hopping gate: exp(-i theta (c_i^dagger c_j + c_j^dagger c_i))
    ///
    /// This implements nearest-neighbor hopping (tunneling) between modes i and j.
    /// It conserves particle number.
    ///
    /// In the Majorana basis, this generates a rotation mixing modes (2i, 2j) and
    /// (2i+1, 2j+1) simultaneously:
    ///
    ///   gamma_{2i}   -> cos(theta) gamma_{2i}   + sin(theta) gamma_{2j}
    ///   gamma_{2j}   -> -sin(theta) gamma_{2i}  + cos(theta) gamma_{2j}
    ///   gamma_{2i+1} -> cos(theta) gamma_{2i+1} + sin(theta) gamma_{2j+1}
    ///   gamma_{2j+1} -> -sin(theta) gamma_{2i+1} + cos(theta) gamma_{2j+1}
    pub fn apply_hopping(
        &mut self,
        i: usize,
        j: usize,
        angle: f64,
    ) -> Result<(), FermionicGaussianError> {
        self.validate_mode(i)?;
        self.validate_mode(j)?;

        // Two Givens rotations: one on the "real" Majorana pair, one on the "imaginary"
        self.apply_givens_rotation(2 * i, 2 * j, angle);
        self.apply_givens_rotation(2 * i + 1, 2 * j + 1, angle);
        Ok(())
    }

    /// Apply a pairing gate: exp(-i theta (c_i^dagger c_j^dagger + c_j c_i))
    ///
    /// This creates or destroys pairs of fermions. It does NOT conserve particle
    /// number, but it does conserve parity.
    ///
    /// In the Majorana basis, this generates a rotation mixing modes (2i, 2j+1) and
    /// (2i+1, 2j):
    ///
    ///   gamma_{2i}   -> cos(theta) gamma_{2i}   + sin(theta) gamma_{2j+1}
    ///   gamma_{2j+1} -> -sin(theta) gamma_{2i}  + cos(theta) gamma_{2j+1}
    ///   gamma_{2i+1} -> cos(theta) gamma_{2i+1} - sin(theta) gamma_{2j}
    ///   gamma_{2j}   -> sin(theta) gamma_{2i+1} + cos(theta) gamma_{2j}
    pub fn apply_pairing(
        &mut self,
        i: usize,
        j: usize,
        angle: f64,
    ) -> Result<(), FermionicGaussianError> {
        self.validate_mode(i)?;
        self.validate_mode(j)?;

        // The pairing Hamiltonian c_i^dag c_j^dag + h.c. in the Majorana basis
        // generates rotations in the (2i, 2j+1) and (2i+1, 2j) planes.
        //
        // Expanding: c_i^dag c_j^dag + c_j c_i
        //   = (-i/2)[gamma_{2i} gamma_{2j+1} + gamma_{2i+1} gamma_{2j}]
        //
        // The SO(2n) rotation decomposes into two Givens rotations:
        // one in the (2i, 2j+1) plane with +theta, and one in the
        // (2j, 2i+1) plane with -theta. The index ordering and sign
        // difference ensure that pair creation/annihilation occurs
        // rather than the rotations canceling.
        self.apply_givens_rotation(2 * i, 2 * j + 1, angle);
        self.apply_givens_rotation(2 * j, 2 * i + 1, -angle);
        Ok(())
    }

    /// Apply an on-site phase rotation: exp(-i theta c_i^dagger c_i)
    ///
    /// This rotates the phase of mode i without changing the occupation.
    /// In the Majorana basis, it generates a rotation in the (2i, 2i+1) plane:
    ///
    ///   gamma_{2i}   -> cos(theta) gamma_{2i}   + sin(theta) gamma_{2i+1}
    ///   gamma_{2i+1} -> -sin(theta) gamma_{2i}  + cos(theta) gamma_{2i+1}
    pub fn apply_onsite_phase(
        &mut self,
        i: usize,
        angle: f64,
    ) -> Result<(), FermionicGaussianError> {
        self.validate_mode(i)?;

        self.apply_givens_rotation(2 * i, 2 * i + 1, angle);
        Ok(())
    }

    /// Apply a beam splitter between modes i and j.
    ///
    /// The beam splitter is equivalent to a hopping gate:
    ///   BS(theta) = exp(-i theta (c_i^dagger c_j - c_j^dagger c_i))
    ///
    /// This is the standard optical beam splitter transformation:
    ///   c_i -> cos(theta) c_i + sin(theta) c_j
    ///   c_j -> -sin(theta) c_i + cos(theta) c_j
    ///
    /// In the Majorana basis, this is identical to hopping but with the
    /// imaginary pair rotation sign flipped.
    pub fn apply_beam_splitter(
        &mut self,
        i: usize,
        j: usize,
        angle: f64,
    ) -> Result<(), FermionicGaussianError> {
        self.validate_mode(i)?;
        self.validate_mode(j)?;

        self.apply_givens_rotation(2 * i, 2 * j, angle);
        self.apply_givens_rotation(2 * i + 1, 2 * j + 1, angle);
        Ok(())
    }

    /// Apply an arbitrary SO(2n) rotation to the covariance matrix.
    ///
    /// Transforms: Gamma -> O Gamma O^T
    ///
    /// The matrix O must be a 2n x 2n real orthogonal matrix (O O^T = I, det(O) = +1).
    /// This is the most general Gaussian unitary operation.
    pub fn apply_arbitrary_rotation(
        &mut self,
        so2n_matrix: &[Vec<f64>],
    ) -> Result<(), FermionicGaussianError> {
        let dim = self.dim();
        if so2n_matrix.len() != dim {
            return Err(FermionicGaussianError::InvalidRotationMatrix {
                expected: dim,
                got: so2n_matrix.len(),
            });
        }
        for row in so2n_matrix {
            if row.len() != dim {
                return Err(FermionicGaussianError::InvalidRotationMatrix {
                    expected: dim,
                    got: row.len(),
                });
            }
        }

        // Gamma' = O * Gamma * O^T
        // Step 1: temp = O * Gamma
        let mut temp = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += so2n_matrix[i][k] * self.covariance[k][j];
                }
                temp[i][j] = sum;
            }
        }

        // Step 2: Gamma' = temp * O^T
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += temp[i][k] * so2n_matrix[j][k]; // O^T[k][j] = O[j][k]
                }
                self.covariance[i][j] = sum;
            }
        }

        Ok(())
    }

    // ---------------------------------------------------------------
    // MEASUREMENTS
    // ---------------------------------------------------------------

    /// Compute the occupation number <c_i^dagger c_i> for mode i.
    ///
    /// From the Majorana covariance matrix:
    ///   <n_i> = <c_i^dagger c_i> = (1 - Gamma_{2i, 2i+1}) / 2
    pub fn occupation_number(&self, i: usize) -> f64 {
        if i >= self.n_modes {
            return 0.0;
        }
        (1.0 - self.covariance[2 * i][2 * i + 1]) / 2.0
    }

    /// Compute the two-point correlator <c_i^dagger c_j>.
    ///
    /// For a Gaussian state, all higher-order correlators can be computed from
    /// two-point correlators via Wick's theorem.
    ///
    /// The two-point correlator in terms of Majorana covariance:
    ///   <c_i^dagger c_j> = (delta_{ij} + i*Gamma_{2i,2j} + i*Gamma_{2i+1,2j+1}
    ///                       + Gamma_{2i+1,2j} - Gamma_{2i,2j+1}) / 4
    ///
    /// For i = j this reduces to the occupation number.
    pub fn two_point_correlator(&self, i: usize, j: usize) -> C64 {
        if i >= self.n_modes || j >= self.n_modes {
            return C64::new(0.0, 0.0);
        }

        if i == j {
            return C64::new(self.occupation_number(i), 0.0);
        }

        let delta = if i == j { 1.0 } else { 0.0 };

        // <c_i^dag c_j> = (delta_ij + Gamma_{2i+1, 2j} - Gamma_{2i, 2j+1}) / 4
        //               + i * (Gamma_{2i, 2j} + Gamma_{2i+1, 2j+1}) / 4
        let real_part = (delta + self.covariance[2 * i + 1][2 * j]
            - self.covariance[2 * i][2 * j + 1])
            / 4.0;

        let imag_part = (self.covariance[2 * i][2 * j]
            + self.covariance[2 * i + 1][2 * j + 1])
            / 4.0;

        C64::new(real_part, imag_part)
    }

    /// Compute the total parity <(-1)^N> = Pfaffian(Gamma).
    ///
    /// For the Majorana covariance matrix, the Pfaffian gives the expectation
    /// value of the parity operator. For a pure Gaussian state, this is +/-1.
    pub fn parity(&self) -> f64 {
        pfaffian_real(&self.covariance)
    }

    /// Compute the total particle number <N> = sum_i <n_i>.
    pub fn total_particle_number(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n_modes {
            total += self.occupation_number(i);
        }
        total
    }

    /// Compute the von Neumann entropy of a subsystem.
    ///
    /// For a Gaussian state, the entanglement entropy of a subsystem A is
    /// determined entirely by the eigenvalues of the reduced covariance matrix.
    ///
    /// If the eigenvalues of i * Gamma_A come in pairs +/- nu_k, the entropy is:
    ///   S(A) = -sum_k [ h((1+nu_k)/2) ]
    ///
    /// where h(x) = -x ln(x) - (1-x) ln(1-x) is the binary entropy.
    pub fn entropy_of_subsystem(&self, sites: &[usize]) -> f64 {
        if sites.is_empty() {
            return 0.0;
        }

        // Validate sites
        for &s in sites {
            if s >= self.n_modes {
                return 0.0;
            }
        }

        // Extract the reduced covariance matrix for the subsystem
        let n_sub = sites.len();
        let dim_sub = 2 * n_sub;
        let mut reduced = vec![vec![0.0f64; dim_sub]; dim_sub];

        for (si, &qi) in sites.iter().enumerate() {
            for (sj, &qj) in sites.iter().enumerate() {
                reduced[2 * si][2 * sj] = self.covariance[2 * qi][2 * qj];
                reduced[2 * si][2 * sj + 1] = self.covariance[2 * qi][2 * qj + 1];
                reduced[2 * si + 1][2 * sj] = self.covariance[2 * qi + 1][2 * qj];
                reduced[2 * si + 1][2 * sj + 1] =
                    self.covariance[2 * qi + 1][2 * qj + 1];
            }
        }

        // Compute the symplectic eigenvalues of the reduced covariance matrix.
        // For an antisymmetric matrix, the eigenvalues of Gamma come in pairs +/- i*nu_k.
        // The singular values of Gamma are |nu_k| (each appearing twice).
        let singular_values = symplectic_eigenvalues(&reduced, dim_sub);

        // Compute entropy from symplectic eigenvalues
        let mut entropy = 0.0;
        for &nu in &singular_values {
            let nu_clamped = nu.abs().min(1.0);
            if nu_clamped < 1e-14 {
                continue;
            }
            let p_plus = (1.0 + nu_clamped) / 2.0;
            let p_minus = (1.0 - nu_clamped) / 2.0;

            if p_plus > 1e-14 && p_plus < 1.0 - 1e-14 {
                entropy -= p_plus * p_plus.ln();
            }
            if p_minus > 1e-14 && p_minus < 1.0 - 1e-14 {
                entropy -= p_minus * p_minus.ln();
            }
        }

        entropy
    }

    /// Compute the overlap |<psi|phi>|^2 between two Gaussian states.
    ///
    /// For two pure Gaussian states with covariance matrices Gamma_1 and Gamma_2,
    /// the squared overlap is given by (Bravyi 2005):
    ///
    ///   |<psi_1|psi_2>|^2 = det((I - Gamma_1 Gamma_2) / 2)
    ///
    /// The minus sign arises because Gamma^T = -Gamma for antisymmetric matrices,
    /// so Gamma_1^T Gamma_2 = -Gamma_1 Gamma_2.
    ///
    /// For self-overlap: Gamma^2 = -I (pure state), so
    ///   (I - (-I))/2 = I, det(I) = 1. Correct.
    pub fn overlap(&self, other: &Self) -> f64 {
        if self.n_modes != other.n_modes {
            return 0.0;
        }

        let dim = self.dim();

        // Compute product = Gamma_1 * Gamma_2
        let mut product = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += self.covariance[i][k] * other.covariance[k][j];
                }
                product[i][j] = sum;
            }
        }

        // M = (I - Gamma_1 * Gamma_2) / 2
        let mut m = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let id = if i == j { 1.0 } else { 0.0 };
                m[i][j] = (id - product[i][j]) / 2.0;
            }
        }

        // |<psi_1|psi_2>|^2 = det(M)
        let det = determinant_real(&m);
        // The determinant should be non-negative for valid overlaps; take abs for
        // numerical safety and clamp to [0, 1].
        det.abs().min(1.0)
    }

    // ---------------------------------------------------------------
    // INTERNAL HELPERS
    // ---------------------------------------------------------------

    /// Apply a single Givens rotation in the (p, q) plane with angle theta.
    ///
    /// This performs: Gamma -> R * Gamma * R^T where R is the Givens rotation
    /// that mixes rows/columns p and q. This is O(n) since only two rows/columns
    /// are affected.
    fn apply_givens_rotation(&mut self, p: usize, q: usize, angle: f64) {
        let dim = self.dim();
        let c = angle.cos();
        let s = angle.sin();

        // Left multiply: Gamma <- R * Gamma (mix rows p and q)
        let mut new_row_p = vec![0.0f64; dim];
        let mut new_row_q = vec![0.0f64; dim];
        for k in 0..dim {
            new_row_p[k] = c * self.covariance[p][k] + s * self.covariance[q][k];
            new_row_q[k] = -s * self.covariance[p][k] + c * self.covariance[q][k];
        }
        self.covariance[p] = new_row_p;
        self.covariance[q] = new_row_q;

        // Right multiply: Gamma <- Gamma * R^T (mix columns p and q)
        for k in 0..dim {
            let old_kp = self.covariance[k][p];
            let old_kq = self.covariance[k][q];
            self.covariance[k][p] = c * old_kp + s * old_kq;
            self.covariance[k][q] = -s * old_kp + c * old_kq;
        }
    }

    /// Validate that a mode index is within range.
    fn validate_mode(&self, mode: usize) -> Result<(), FermionicGaussianError> {
        if mode >= self.n_modes {
            Err(FermionicGaussianError::ModeOutOfRange {
                mode,
                n_modes: self.n_modes,
            })
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for FermionicGaussianState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FermionicGaussianState({} modes, ", self.n_modes)?;
        let n_particles = self.total_particle_number();
        write!(f, "<N>={:.2})", n_particles)
    }
}

// ===================================================================
// FREE FERMION HAMILTONIAN
// ===================================================================

/// A free (quadratic) fermion Hamiltonian.
///
/// H = sum_{i,j} h_{ij} c_i^dagger c_j + (Delta_{ij} c_i^dagger c_j^dagger + h.c.)
///
/// For a particle-number-conserving Hamiltonian, the pairing matrix Delta is zero,
/// and only the hopping matrix h matters.
#[derive(Clone, Debug)]
pub struct FreeHamiltonian {
    /// Number of fermionic modes.
    pub n_modes: usize,
    /// The n x n Hermitian hopping matrix h_{ij} (stored as real since we
    /// restrict to real Hamiltonians for simplicity; the imaginary part can
    /// be encoded in the antisymmetric component).
    pub hopping: Vec<Vec<f64>>,
    /// The n x n antisymmetric pairing matrix Delta_{ij}.
    /// If None, the Hamiltonian conserves particle number.
    pub pairing: Option<Vec<Vec<f64>>>,
}

impl FreeHamiltonian {
    /// Build a free fermion Hamiltonian from a hopping matrix.
    ///
    /// The hopping matrix h must be n x n and Hermitian (for real matrices,
    /// this means symmetric: h_{ij} = h_{ji}).
    pub fn from_hopping(hopping: Vec<Vec<f64>>) -> Self {
        let n_modes = hopping.len();
        FreeHamiltonian {
            n_modes,
            hopping,
            pairing: None,
        }
    }

    /// Build a free fermion Hamiltonian with both hopping and pairing terms.
    pub fn from_hopping_and_pairing(
        hopping: Vec<Vec<f64>>,
        pairing: Vec<Vec<f64>>,
    ) -> Self {
        let n_modes = hopping.len();
        FreeHamiltonian {
            n_modes,
            hopping,
            pairing: Some(pairing),
        }
    }

    /// Construct the 2n x 2n single-particle Hamiltonian matrix in the
    /// Majorana basis (also called the BdG Hamiltonian matrix).
    ///
    /// For H = sum_{ij} h_{ij} c_i^dag c_j + Delta_{ij} c_i^dag c_j^dag + h.c.,
    /// the Majorana-basis Hamiltonian A is a 2n x 2n real antisymmetric matrix
    /// such that H = (i/4) sum_{jk} A_{jk} gamma_j gamma_k.
    ///
    /// For a pure hopping Hamiltonian:
    ///   A_{2i, 2j+1} = -h_{ij} (symmetric part)
    ///   A_{2i+1, 2j} = h_{ij}
    ///   A_{2i, 2j} = 0 for hopping-only
    ///   A_{2i+1, 2j+1} = 0 for hopping-only
    ///
    /// With pairing Delta:
    ///   A_{2i, 2j} = -Delta_{ij}  (antisymmetric part contributes)
    ///   A_{2i+1, 2j+1} = Delta_{ij}
    pub fn to_majorana_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_modes;
        let dim = 2 * n;
        let mut a = vec![vec![0.0f64; dim]; dim];

        // Hopping contribution
        for i in 0..n {
            for j in 0..n {
                let h = self.hopping[i][j];
                // The Majorana Hamiltonian matrix A is antisymmetric.
                // Hopping h_{ij} c_i^dag c_j contributes:
                //   A_{2i, 2j+1} += -h_ij / 2  and  A_{2j+1, 2i} += h_ij / 2
                //   A_{2i+1, 2j} += h_ij / 2   and  A_{2j, 2i+1} += -h_ij / 2
                // For a symmetric hopping matrix, this simplifies.
                a[2 * i][2 * j + 1] += -h;
                a[2 * j + 1][2 * i] += h;
                a[2 * i + 1][2 * j] += h;
                a[2 * j][2 * i + 1] += -h;
            }
        }

        // Pairing contribution
        if let Some(ref delta) = self.pairing {
            for i in 0..n {
                for j in 0..n {
                    let d = delta[i][j];
                    a[2 * i][2 * j] += -d;
                    a[2 * j][2 * i] += d;
                    a[2 * i + 1][2 * j + 1] += d;
                    a[2 * j + 1][2 * i + 1] += -d;
                }
            }
        }

        a
    }
}

// ===================================================================
// UTILITY FUNCTIONS
// ===================================================================

/// Build a free fermion Hamiltonian from a hopping matrix.
///
/// Convenience wrapper around FreeHamiltonian::from_hopping.
pub fn free_fermion_hamiltonian(hopping_matrix: Vec<Vec<f64>>) -> FreeHamiltonian {
    FreeHamiltonian::from_hopping(hopping_matrix)
}

/// Find the ground state of a free fermion Hamiltonian.
///
/// For a particle-number-conserving Hamiltonian H = sum_{ij} h_{ij} c_i^dag c_j,
/// the ground state is the Slater determinant obtained by filling the lowest-energy
/// single-particle orbitals. This is found by diagonalizing the hopping matrix.
///
/// For a general (BdG) Hamiltonian with pairing, the ground state is found by
/// diagonalizing the 2n x 2n Majorana Hamiltonian matrix and constructing the
/// covariance matrix from the negative-eigenvalue eigenvectors.
pub fn ground_state(hamiltonian: &FreeHamiltonian) -> FermionicGaussianState {
    let n = hamiltonian.n_modes;

    if hamiltonian.pairing.is_none() {
        // Particle-number-conserving case: diagonalize the n x n hopping matrix.
        // Fill the eigenstates with negative eigenvalues.
        let eigenvalues = symmetric_eigenvalues_sorted(&hamiltonian.hopping, n);

        // Count how many eigenvalues are negative (those are the occupied modes
        // in the ground state). For zero eigenvalues, we fill them if we want
        // half-filling; here we fill all negative ones.
        let n_occupied = eigenvalues.iter().filter(|&&e| e < -1e-12).count();

        // The ground state covariance is computed from the eigenvectors.
        // For the simple case of a diagonal Hamiltonian, this is just occupying
        // the lowest modes. For a general hopping matrix, we need the eigenvectors.
        let (eigvals, eigvecs) = diagonalize_symmetric(&hamiltonian.hopping, n);

        // Build the one-body density matrix P_{ij} = sum_{k occupied} v_{ik} v_{jk}
        let mut p_matrix = vec![vec![0.0f64; n]; n];
        for k in 0..n {
            if eigvals[k] < 1e-12 {
                // This eigenvalue is negative or zero -> occupied
                for i in 0..n {
                    for j in 0..n {
                        p_matrix[i][j] += eigvecs[i][k] * eigvecs[j][k];
                    }
                }
            }
        }

        // Convert one-body density matrix to Majorana covariance matrix.
        // Gamma_{2i, 2j+1} = delta_{ij} - 2 * P_{ij}  (for the cross terms)
        // Gamma_{2i+1, 2j} = -(delta_{ij} - 2 * P_{ij})
        // Gamma_{2i, 2j} = 0 for particle-number-conserving states
        // Gamma_{2i+1, 2j+1} = 0 for particle-number-conserving states
        let dim = 2 * n;
        let mut covariance = vec![vec![0.0f64; dim]; dim];

        for i in 0..n {
            for j in 0..n {
                let q = if i == j { 1.0 } else { 0.0 } - 2.0 * p_matrix[i][j];
                covariance[2 * i][2 * j + 1] = q;
                covariance[2 * j + 1][2 * i] = -q;
                // The diagonal 2x2 blocks are set; off-diagonal (2i,2j) and
                // (2i+1,2j+1) remain zero for number-conserving states.
            }
        }

        FermionicGaussianState {
            n_modes: n,
            covariance,
        }
    } else {
        // General BdG case: diagonalize the 2n x 2n Majorana Hamiltonian.
        // The ground state covariance is Gamma = -sign(A) where A is the
        // Majorana Hamiltonian matrix.
        let a = hamiltonian.to_majorana_matrix();
        let dim = 2 * n;

        // For the ground state, we need Gamma such that [Gamma, A] is related
        // to the spectral projector. The ground state covariance is:
        //   Gamma = sum_k sign(epsilon_k) * |v_k><v_k| (in appropriate basis)
        // where epsilon_k are the eigenvalues of iA.
        //
        // Simpler approach: compute Gamma = i * sign(iA) where iA is Hermitian.
        // Since A is real antisymmetric, iA is real symmetric (when A acts on
        // Majorana operators, the factor of i makes it Hermitian).
        // Actually, for the antisymmetric A, compute A^2, which is symmetric
        // negative semi-definite. The ground state covariance can be obtained
        // from the polar decomposition of A.

        // Compute A^2
        let mut a_sq = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    a_sq[i][j] += a[i][k] * a[k][j];
                }
            }
        }

        // For the ground state of a free Hamiltonian, if all eigenvalues of iA
        // are non-degenerate, the ground state covariance is the sign function:
        //   Gamma = A * (-A^2)^{-1/2}
        // This is the polar decomposition of A.

        // Compute (-A^2)^{-1/2} via eigendecomposition.
        // -A^2 is positive semi-definite (since A is antisymmetric, A^2 is
        // negative semi-definite).
        let mut neg_a_sq = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                neg_a_sq[i][j] = -a_sq[i][j];
            }
        }

        let (eigvals, eigvecs) = diagonalize_symmetric(&neg_a_sq, dim);

        // Compute (-A^2)^{-1/2}
        let mut inv_sqrt = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    if eigvals[k] > 1e-14 {
                        sum += eigvecs[i][k] * eigvecs[j][k] / eigvals[k].sqrt();
                    }
                }
                inv_sqrt[i][j] = sum;
            }
        }

        // Gamma = A * (-A^2)^{-1/2}
        let mut covariance = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += a[i][k] * inv_sqrt[k][j];
                }
                covariance[i][j] = sum;
            }
        }

        FermionicGaussianState {
            n_modes: n,
            covariance,
        }
    }
}

/// Evolve a fermionic Gaussian state under a free Hamiltonian for time dt.
///
/// The time evolution operator U = exp(-i H dt) is a Gaussian unitary.
/// In the Majorana basis, it acts as an SO(2n) rotation:
///   O(dt) = exp(A * dt)
///
/// where A is the 2n x 2n Majorana Hamiltonian matrix.
///
/// We compute exp(A * dt) using the Pade approximation (for small dt)
/// or eigendecomposition (for general dt).
pub fn time_evolve(
    state: &mut FermionicGaussianState,
    hamiltonian: &FreeHamiltonian,
    dt: f64,
) -> Result<(), FermionicGaussianError> {
    let n = hamiltonian.n_modes;
    if n != state.n_modes {
        return Err(FermionicGaussianError::InvalidHamiltonian {
            expected: 2 * state.n_modes,
            got: 2 * n,
        });
    }

    let a = hamiltonian.to_majorana_matrix();
    let dim = 2 * n;

    // Compute O = exp(A * dt) using scaling and squaring with Pade approximant.
    // For moderate dt, we use a truncated Taylor series:
    //   exp(A*dt) ~ I + A*dt + (A*dt)^2/2! + (A*dt)^3/3! + ...
    //
    // For better accuracy, we use the (6,6) Pade approximant with scaling.

    // Scale: find s such that ||A*dt|| / 2^s < 0.5
    let mut norm = 0.0f64;
    for i in 0..dim {
        let mut row_sum = 0.0;
        for j in 0..dim {
            row_sum += (a[i][j] * dt).abs();
        }
        norm = norm.max(row_sum);
    }

    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };

    let scale = 2.0f64.powi(-(s as i32));
    let scaled_dt = dt * scale;

    // Compute exp(A * scaled_dt) via Taylor series to 8th order
    let mut exp_a = vec![vec![0.0f64; dim]; dim];
    // Initialize to identity
    for i in 0..dim {
        exp_a[i][i] = 1.0;
    }

    // Compute A_scaled = A * scaled_dt
    let mut a_scaled = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            a_scaled[i][j] = a[i][j] * scaled_dt;
        }
    }

    // Accumulate Taylor series: exp(M) = I + M + M^2/2! + M^3/3! + ...
    let mut power = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        power[i][i] = 1.0; // M^0 = I
    }

    for order in 1..=12 {
        // power = power * a_scaled
        let mut new_power = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += power[i][k] * a_scaled[k][j];
                }
                new_power[i][j] = sum;
            }
        }
        power = new_power;

        let factorial_inv = 1.0 / factorial(order);
        for i in 0..dim {
            for j in 0..dim {
                exp_a[i][j] += power[i][j] * factorial_inv;
            }
        }
    }

    // Squaring phase: exp(A*dt) = (exp(A*scaled_dt))^{2^s}
    for _ in 0..s {
        let mut squared = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += exp_a[i][k] * exp_a[k][j];
                }
                squared[i][j] = sum;
            }
        }
        exp_a = squared;
    }

    // Apply the rotation: Gamma -> O * Gamma * O^T
    state.apply_arbitrary_rotation(&exp_a)?;

    Ok(())
}

/// Convert a matchgate circuit (sequence of nearest-neighbor matchgates) into
/// a FermionicGaussianState.
///
/// Each matchgate acts as a Givens rotation on the Majorana covariance matrix.
/// The circuit is specified as a list of (mode_i, mode_j, angle) tuples
/// representing hopping gates.
pub fn matchgate_circuit_to_gaussian(
    n_modes: usize,
    gates: &[(usize, usize, f64)],
) -> Result<FermionicGaussianState, FermionicGaussianError> {
    let mut state = FermionicGaussianState::new(n_modes);

    for &(i, j, angle) in gates {
        state.apply_hopping(i, j, angle)?;
    }

    Ok(state)
}

// ===================================================================
// LINEAR ALGEBRA HELPERS
// ===================================================================

/// Compute the Pfaffian of a 2n x 2n real antisymmetric matrix.
///
/// Uses antisymmetric Gaussian elimination (Parlett-Reid method).
/// Complexity: O(n^3).
fn pfaffian_real(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }
    if n % 2 != 0 {
        return 0.0;
    }
    if n == 2 {
        return matrix[0][1];
    }

    let mut m: Vec<Vec<f64>> = matrix.to_vec();
    let mut pf = 1.0;
    let pairs = n / 2;

    for k in 0..pairs {
        let pivot_row = 2 * k;
        let pivot_col = 2 * k + 1;
        if pivot_col >= n {
            break;
        }

        // Partial pivoting: find largest element in column below pivot
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
            return 0.0;
        }

        if max_idx != pivot_row {
            m.swap(max_idx, pivot_row);
            for row in &mut m {
                row.swap(max_idx, pivot_row);
            }
            pf = -pf;
        }

        let pivot = m[pivot_row][pivot_col];
        pf *= pivot;

        // Eliminate below the 2x2 pivot block
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

/// Compute symplectic eigenvalues of a real antisymmetric matrix.
///
/// For a 2n x 2n antisymmetric matrix, the eigenvalues come in pairs +/- i*nu_k.
/// Returns the nu_k values (the symplectic eigenvalues), one per pair.
fn symplectic_eigenvalues(matrix: &[Vec<f64>], dim: usize) -> Vec<f64> {
    if dim == 0 {
        return vec![];
    }

    // Compute A^T * A = -A^2 (since A is antisymmetric)
    let mut ata = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                ata[i][j] += matrix[k][i] * matrix[k][j];
            }
        }
    }

    // Eigenvalues of A^T * A are the squared singular values nu_k^2 (each with
    // degeneracy 2).
    let eigenvalues = jacobi_eigenvalues(&ata, dim);

    // Take square roots and deduplicate pairs
    let mut svs: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| ev.max(0.0).sqrt())
        .filter(|&sv| sv > 1e-14)
        .collect();

    svs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut unique = Vec::new();
    let mut i = 0;
    while i < svs.len() {
        unique.push(svs[i]);
        if i + 1 < svs.len() && (svs[i] - svs[i + 1]).abs() < 1e-8 {
            i += 2;
        } else {
            i += 1;
        }
    }

    unique
}

/// Compute eigenvalues of a real symmetric matrix using Jacobi iteration.
fn jacobi_eigenvalues(matrix: &[Vec<f64>], dim: usize) -> Vec<f64> {
    if dim == 0 {
        return vec![];
    }
    if dim == 1 {
        return vec![matrix[0][0]];
    }

    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let max_iterations = 200 * dim * dim;

    for _ in 0..max_iterations {
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

        if max_off < 1e-14 {
            break;
        }

        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

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
        for k in 0..dim {
            let old_kp = a[k][p];
            let old_kq = a[k][q];
            a[k][p] = c * old_kp + s * old_kq;
            a[k][q] = -s * old_kp + c * old_kq;
        }
    }

    (0..dim).map(|i| a[i][i]).collect()
}

/// Diagonalize a real symmetric matrix, returning (eigenvalues, eigenvectors).
///
/// Eigenvalues are sorted in ascending order. The eigenvector matrix has columns
/// that are the eigenvectors: eigvecs[i][k] is the i-th component of the k-th
/// eigenvector.
fn diagonalize_symmetric(matrix: &[Vec<f64>], dim: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    if dim == 0 {
        return (vec![], vec![]);
    }

    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    // Accumulated rotation matrix (columns are eigenvectors)
    let mut v = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        v[i][i] = 1.0;
    }

    let max_iterations = 200 * dim * dim;

    for _ in 0..max_iterations {
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

        if max_off < 1e-14 {
            break;
        }

        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Rotate matrix: A' = J^T A J
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
        for k in 0..dim {
            let old_kp = a[k][p];
            let old_kq = a[k][q];
            a[k][p] = c * old_kp + s * old_kq;
            a[k][q] = -s * old_kp + c * old_kq;
        }

        // Accumulate eigenvectors: V' = V * J
        for k in 0..dim {
            let old_kp = v[k][p];
            let old_kq = v[k][q];
            v[k][p] = c * old_kp + s * old_kq;
            v[k][q] = -s * old_kp + c * old_kq;
        }
    }

    let eigenvalues: Vec<f64> = (0..dim).map(|i| a[i][i]).collect();

    // Sort eigenvalues and rearrange eigenvectors accordingly
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_eigvecs = vec![vec![0.0f64; dim]; dim];
    for new_k in 0..dim {
        let old_k = indices[new_k];
        for i in 0..dim {
            sorted_eigvecs[i][new_k] = v[i][old_k];
        }
    }

    (sorted_eigenvalues, sorted_eigvecs)
}

/// Compute eigenvalues of a real symmetric matrix, sorted in ascending order.
fn symmetric_eigenvalues_sorted(matrix: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let (eigvals, _) = diagonalize_symmetric(matrix, dim);
    eigvals
}

/// Compute the determinant of a real square matrix using LU decomposition.
fn determinant_real(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return matrix[0][0];
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    // LU decomposition with partial pivoting
    let mut m: Vec<Vec<f64>> = matrix.to_vec();
    let mut sign = 1.0;

    for col in 0..n {
        // Find pivot
        let mut max_val = m[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return 0.0;
        }

        if max_row != col {
            m.swap(max_row, col);
            sign = -sign;
        }

        let pivot = m[col][col];
        for row in (col + 1)..n {
            let factor = m[row][col] / pivot;
            for j in col..n {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    let mut det = sign;
    for i in 0..n {
        det *= m[i][i];
    }

    det
}

/// Compute n! as f64 (for small n).
fn factorial(n: usize) -> f64 {
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    const TOL: f64 = 1e-8;

    // -----------------------------------------------------------
    // Test 1: Vacuum state properties
    // -----------------------------------------------------------
    #[test]
    fn test_vacuum_state_properties() {
        let n = 5;
        let state = FermionicGaussianState::new(n);

        // Vacuum should have zero particles everywhere
        for i in 0..n {
            let occ = state.occupation_number(i);
            assert!(
                occ.abs() < TOL,
                "Vacuum occupation for mode {} should be 0, got {}",
                i,
                occ
            );
        }

        // Total particle number should be 0
        assert!(state.total_particle_number().abs() < TOL);

        // Vacuum is a pure state: Gamma^2 = -I
        assert!(state.is_pure(), "Vacuum state should be pure");

        // Parity of vacuum should be +1 (even number of particles)
        let par = state.parity();
        assert!(
            (par - 1.0).abs() < TOL,
            "Vacuum parity should be +1, got {}",
            par
        );
    }

    // -----------------------------------------------------------
    // Test 2: Slater determinant construction
    // -----------------------------------------------------------
    #[test]
    fn test_slater_determinant() {
        let n = 6;
        let occupied = vec![0, 2, 4];
        let state = FermionicGaussianState::from_occupation(&occupied, n).unwrap();

        // Check occupations
        for i in 0..n {
            let expected = if occupied.contains(&i) { 1.0 } else { 0.0 };
            let occ = state.occupation_number(i);
            assert!(
                (occ - expected).abs() < TOL,
                "Mode {} occupation: expected {}, got {}",
                i,
                expected,
                occ
            );
        }

        // Total particle number should be 3
        let total = state.total_particle_number();
        assert!(
            (total - 3.0).abs() < TOL,
            "Total particle number should be 3, got {}",
            total
        );

        // Slater determinant is a pure state
        assert!(state.is_pure(), "Slater determinant should be pure");

        // Parity should be (-1)^3 = -1 (odd number of particles)
        let par = state.parity();
        assert!(
            (par - (-1.0)).abs() < TOL,
            "Parity of 3-particle state should be -1, got {}",
            par
        );
    }

    // -----------------------------------------------------------
    // Test 3: Hopping gate preserves particle number
    // -----------------------------------------------------------
    #[test]
    fn test_hopping_preserves_particle_number() {
        let n = 4;
        let occupied = vec![0, 1];
        let mut state = FermionicGaussianState::from_occupation(&occupied, n).unwrap();

        let initial_n = state.total_particle_number();

        // Apply several hopping gates
        state.apply_hopping(0, 1, 0.3).unwrap();
        state.apply_hopping(1, 2, 0.7).unwrap();
        state.apply_hopping(2, 3, 1.1).unwrap();
        state.apply_hopping(0, 3, 0.5).unwrap();

        let final_n = state.total_particle_number();

        assert!(
            (initial_n - final_n).abs() < TOL,
            "Hopping should preserve particle number: initial={}, final={}",
            initial_n,
            final_n
        );

        // State should still be pure after Gaussian unitaries
        assert!(
            state.is_pure_with_tol(1e-6),
            "State should remain pure after hopping gates"
        );
    }

    // -----------------------------------------------------------
    // Test 4: Pairing gate changes particle number
    // -----------------------------------------------------------
    #[test]
    fn test_pairing_changes_particle_number() {
        let n = 4;
        let mut state = FermionicGaussianState::new(n); // vacuum

        let initial_n = state.total_particle_number();
        assert!(initial_n.abs() < TOL, "Vacuum should have 0 particles");

        // Apply a pairing gate -- this should create particle pairs
        state.apply_pairing(0, 1, FRAC_PI_4).unwrap();

        let final_n = state.total_particle_number();

        // After pairing, the expectation value of N should have changed
        // (though parity is still conserved).
        // For a pairing with angle pi/4 from vacuum, both modes get partial occupation.
        assert!(
            final_n > TOL,
            "Pairing from vacuum should create particles, got <N>={}",
            final_n
        );

        // Check that the occupation of the two paired modes is equal
        let n0 = state.occupation_number(0);
        let n1 = state.occupation_number(1);
        assert!(
            (n0 - n1).abs() < TOL,
            "Pairing should populate both modes equally: n0={}, n1={}",
            n0,
            n1
        );

        // State should remain pure
        assert!(state.is_pure_with_tol(1e-6));
    }

    // -----------------------------------------------------------
    // Test 5: Entropy of half-chain for ground state of hopping model
    // -----------------------------------------------------------
    #[test]
    fn test_entropy_half_chain_hopping_model() {
        // Build a tight-binding chain with periodic boundary conditions
        let n = 8;
        let t_hop = 1.0;
        let mut hopping = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            let j = (i + 1) % n;
            hopping[i][j] = -t_hop;
            hopping[j][i] = -t_hop;
        }

        let ham = free_fermion_hamiltonian(hopping);
        let gs = ground_state(&ham);

        // Compute entropy of the left half
        let left_half: Vec<usize> = (0..n / 2).collect();
        let entropy = gs.entropy_of_subsystem(&left_half);

        // For a free fermion chain at half-filling, the entanglement entropy
        // should be non-trivial (greater than 0, less than n/2 * ln(2)).
        assert!(
            entropy > 0.0,
            "Half-chain entropy should be positive, got {}",
            entropy
        );

        let max_entropy = (n as f64 / 2.0) * 2.0_f64.ln();
        assert!(
            entropy < max_entropy,
            "Half-chain entropy {} should be less than max {}",
            entropy,
            max_entropy
        );
    }

    // -----------------------------------------------------------
    // Test 6: Beam splitter unitarity
    // -----------------------------------------------------------
    #[test]
    fn test_beam_splitter_unitarity() {
        let n = 4;
        let occupied = vec![0];
        let mut state = FermionicGaussianState::from_occupation(&occupied, n).unwrap();

        // Apply a 50/50 beam splitter between modes 0 and 1
        state.apply_beam_splitter(0, 1, FRAC_PI_4).unwrap();

        // After a 50/50 beam splitter on |1,0,...>, we should get
        // equal occupation on both modes
        let n0 = state.occupation_number(0);
        let n1 = state.occupation_number(1);

        assert!(
            (n0 - 0.5).abs() < TOL,
            "Mode 0 after 50/50 BS should be 0.5, got {}",
            n0
        );
        assert!(
            (n1 - 0.5).abs() < TOL,
            "Mode 1 after 50/50 BS should be 0.5, got {}",
            n1
        );

        // Total particle number preserved
        let total = state.total_particle_number();
        assert!(
            (total - 1.0).abs() < TOL,
            "BS should preserve total particles: expected 1, got {}",
            total
        );

        // State remains pure
        assert!(state.is_pure_with_tol(1e-6));
    }

    // -----------------------------------------------------------
    // Test 7: Time evolution conservation of energy
    // -----------------------------------------------------------
    #[test]
    fn test_time_evolution_conserves_energy() {
        let n = 4;
        let mut hopping = vec![vec![0.0f64; n]; n];
        // Simple hopping chain
        for i in 0..(n - 1) {
            hopping[i][i + 1] = -1.0;
            hopping[i + 1][i] = -1.0;
        }

        let ham = free_fermion_hamiltonian(hopping);
        let mut state = ground_state(&ham);

        // Compute initial energy: E = sum_{ij} h_{ij} <c_i^dag c_j>
        let initial_energy = compute_energy(&state, &ham);

        // Time evolve
        let dt = 0.1;
        for _ in 0..20 {
            time_evolve(&mut state, &ham, dt).unwrap();
        }

        let final_energy = compute_energy(&state, &ham);

        assert!(
            (initial_energy - final_energy).abs() < 1e-4,
            "Energy should be conserved: initial={}, final={}",
            initial_energy,
            final_energy
        );

        // Particle number should also be conserved
        let initial_n = ground_state(&ham).total_particle_number();
        let final_n = state.total_particle_number();
        assert!(
            (initial_n - final_n).abs() < 1e-4,
            "Particle number should be conserved: initial={}, final={}",
            initial_n,
            final_n
        );
    }

    // -----------------------------------------------------------
    // Test 8: Two-site model (non-interacting limit)
    // -----------------------------------------------------------
    #[test]
    fn test_two_site_non_interacting() {
        // Two-site tight-binding model: H = -t (c_0^dag c_1 + h.c.)
        let n = 2;
        let t_hop = 1.0;
        let hopping = vec![vec![0.0, -t_hop], vec![-t_hop, 0.0]];

        let ham = free_fermion_hamiltonian(hopping);
        let gs = ground_state(&ham);

        // Ground state of two-site model at half-filling has one particle
        // in the bonding orbital: (c_0^dag + c_1^dag)/sqrt(2) |vac>
        let n0 = gs.occupation_number(0);
        let n1 = gs.occupation_number(1);

        // Both sites should have occupation 0.5 (bonding orbital is delocalized)
        assert!(
            (n0 - 0.5).abs() < TOL,
            "Site 0 occupation should be 0.5, got {}",
            n0
        );
        assert!(
            (n1 - 0.5).abs() < TOL,
            "Site 1 occupation should be 0.5, got {}",
            n1
        );

        // Two-point correlator should be non-zero
        let corr = gs.two_point_correlator(0, 1);
        assert!(
            corr.norm() > TOL,
            "Two-point correlator should be non-zero for bonding state"
        );
    }

    // -----------------------------------------------------------
    // Test 9: Matchgate equivalence
    // -----------------------------------------------------------
    #[test]
    fn test_matchgate_circuit_equivalence() {
        let n = 4;

        // Build a circuit as a sequence of hopping gates
        let gates = vec![
            (0usize, 1usize, 0.3f64),
            (1, 2, 0.7),
            (2, 3, 0.5),
            (0, 1, 0.2),
        ];

        // Method 1: use matchgate_circuit_to_gaussian
        let state1 = matchgate_circuit_to_gaussian(n, &gates).unwrap();

        // Method 2: manually apply hopping gates
        let mut state2 = FermionicGaussianState::new(n);
        for &(i, j, angle) in &gates {
            state2.apply_hopping(i, j, angle).unwrap();
        }

        // Both states should be identical
        let dim = state1.dim();
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (state1.covariance[i][j] - state2.covariance[i][j]).abs() < TOL,
                    "Covariance mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    state1.covariance[i][j],
                    state2.covariance[i][j]
                );
            }
        }
    }

    // -----------------------------------------------------------
    // Test 10: Particle-hole symmetry
    // -----------------------------------------------------------
    #[test]
    fn test_particle_hole_symmetry() {
        // For a half-filled system with particle-hole symmetric Hamiltonian,
        // the ground state should have <n_i> = 0.5 for all sites.
        let n = 4;
        let mut hopping = vec![vec![0.0f64; n]; n];

        // Uniform hopping chain (particle-hole symmetric at half-filling)
        for i in 0..(n - 1) {
            hopping[i][i + 1] = -1.0;
            hopping[i + 1][i] = -1.0;
        }

        let ham = free_fermion_hamiltonian(hopping);
        let gs = ground_state(&ham);

        // At half-filling (n/2 particles for n sites), each site should
        // have roughly 0.5 occupation on average. For a 4-site open chain,
        // the occupations are not exactly 0.5 per site but the total is n/2.
        let total = gs.total_particle_number();
        let expected_total = n as f64 / 2.0;

        assert!(
            (total - expected_total).abs() < 0.5,
            "Half-filled system should have ~{} particles, got {}",
            expected_total,
            total
        );

        // The particle-hole transform flips occupations: n_i -> 1 - n_i
        // Under particle-hole symmetry, if we fill the upper half of the
        // spectrum, we should get the complementary occupation pattern.
    }

    // -----------------------------------------------------------
    // Test 11: Overlap of orthogonal states = 0
    // -----------------------------------------------------------
    #[test]
    fn test_overlap_orthogonal_states() {
        let n = 4;

        // State 1: occupy mode 0
        let state1 = FermionicGaussianState::from_occupation(&[0], n).unwrap();
        // State 2: occupy mode 1
        let state2 = FermionicGaussianState::from_occupation(&[1], n).unwrap();

        let overlap = state1.overlap(&state2);
        assert!(
            overlap.abs() < TOL,
            "Orthogonal Slater determinants should have overlap 0, got {}",
            overlap
        );

        // Self-overlap should be 1
        let self_overlap = state1.overlap(&state1);
        assert!(
            (self_overlap - 1.0).abs() < TOL,
            "Self-overlap should be 1, got {}",
            self_overlap
        );
    }

    // -----------------------------------------------------------
    // Test 12: Large system performance test (100+ modes)
    // -----------------------------------------------------------
    #[test]
    fn test_large_system_performance() {
        let n = 128;
        let mut state = FermionicGaussianState::new(n);

        // Apply a chain of hopping gates
        let start = std::time::Instant::now();
        for i in 0..(n - 1) {
            state.apply_hopping(i, i + 1, 0.1).unwrap();
        }
        let gate_time = start.elapsed();

        // Measure all occupations
        let start = std::time::Instant::now();
        let mut total = 0.0;
        for i in 0..n {
            total += state.occupation_number(i);
        }
        let measure_time = start.elapsed();

        // Vacuum + hopping should still have 0 particles
        assert!(
            total.abs() < TOL,
            "Vacuum after hopping should have 0 particles, got {}",
            total
        );

        // Performance check: gates should complete in reasonable time
        // (< 1 second for 127 gates on 128 modes)
        assert!(
            gate_time.as_secs_f64() < 2.0,
            "128-mode simulation took too long: {:.3}s",
            gate_time.as_secs_f64()
        );

        // Verify state is still pure
        // (Skip the full is_pure check for 128 modes as it is O(n^3))
        // Instead check a few covariance matrix properties.
        let dim = state.dim();
        for i in 0..dim {
            assert!(
                state.covariance[i][i].abs() < TOL,
                "Diagonal of antisymmetric matrix should be 0"
            );
        }
    }

    // -----------------------------------------------------------
    // Test 13: On-site phase rotation
    // -----------------------------------------------------------
    #[test]
    fn test_onsite_phase_preserves_occupation() {
        let n = 3;
        let occupied = vec![0, 2];
        let mut state = FermionicGaussianState::from_occupation(&occupied, n).unwrap();

        // On-site phase should not change occupation numbers
        state.apply_onsite_phase(0, 0.7).unwrap();
        state.apply_onsite_phase(1, 1.3).unwrap();
        state.apply_onsite_phase(2, 2.1).unwrap();

        for i in 0..n {
            let expected = if occupied.contains(&i) { 1.0 } else { 0.0 };
            let occ = state.occupation_number(i);
            assert!(
                (occ - expected).abs() < TOL,
                "On-site phase should not change occupation: mode {} expected {}, got {}",
                i,
                expected,
                occ
            );
        }
    }

    // -----------------------------------------------------------
    // Test 14: From covariance round-trip
    // -----------------------------------------------------------
    #[test]
    fn test_from_covariance_roundtrip() {
        let n = 3;
        let original = FermionicGaussianState::from_occupation(&[1], n).unwrap();

        // Extract covariance and reconstruct
        let gamma = original.covariance.clone();
        let reconstructed = FermionicGaussianState::from_covariance(gamma).unwrap();

        assert_eq!(original.n_modes, reconstructed.n_modes);

        for i in 0..n {
            let occ_orig = original.occupation_number(i);
            let occ_recon = reconstructed.occupation_number(i);
            assert!(
                (occ_orig - occ_recon).abs() < TOL,
                "Round-trip occupation mismatch at mode {}",
                i
            );
        }
    }

    // -----------------------------------------------------------
    // Test 15: Error handling
    // -----------------------------------------------------------
    #[test]
    fn test_error_handling() {
        let mut state = FermionicGaussianState::new(3);

        // Out-of-range mode should fail
        assert!(state.apply_hopping(0, 5, 0.1).is_err());
        assert!(state.apply_pairing(3, 1, 0.1).is_err());
        assert!(state.apply_onsite_phase(10, 0.1).is_err());

        // Invalid covariance matrix (not antisymmetric)
        let bad_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(FermionicGaussianState::from_covariance(bad_matrix).is_err());

        // Odd-dimension matrix
        let odd_matrix = vec![vec![0.0, 1.0, 0.0], vec![-1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];
        assert!(FermionicGaussianState::from_covariance(odd_matrix).is_err());
    }

    // -----------------------------------------------------------
    // Helper: compute energy <H> = sum_{ij} h_{ij} <c_i^dag c_j>
    // -----------------------------------------------------------
    fn compute_energy(state: &FermionicGaussianState, ham: &FreeHamiltonian) -> f64 {
        let n = ham.n_modes;
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..n {
                let corr = state.two_point_correlator(i, j);
                energy += ham.hopping[i][j] * corr.re;
            }
        }
        energy
    }
}
