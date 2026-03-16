//! Quantum Imaginary Time Evolution (QITE) and Variational Quantum Deflation (VQD)
//!
//! Two complementary algorithms for computing ground and excited states of
//! quantum Hamiltonians:
//!
//! - **QITE**: Projects out excited-state components via imaginary-time evolution
//!   |psi(tau+dtau)> = exp(-H*dtau)|psi(tau)> / norm, converging to the ground
//!   state exponentially fast in the spectral gap. Also computes thermal (Gibbs)
//!   states at inverse temperature beta. Based on Motta et al., Nature Physics 16
//!   (2020).
//!
//! - **VQD**: Finds multiple eigenstates by iteratively minimising the cost
//!   E_k(theta) = <psi(theta)|H|psi(theta)> + sum_{j<k} beta_j |<psi(theta)|psi_j>|^2,
//!   which penalises overlap with previously found states. Based on Higgott, Wang,
//!   and Brierley, Quantum 3, 156 (2019).
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::quantum_dynamics::*;
//!
//! // Build 2-qubit transverse-field Ising model
//! let ham = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
//!
//! // Find ground state with QITE
//! let config = QiteConfig::default();
//! let result = qite_ground_state(&ham, &config);
//! assert!(result.converged);
//!
//! // Find ground + first excited state with VQD
//! let vqd_config = VqdConfig::default().num_states(2);
//! let vqd_result = vqd_excited_states(&ham, &VqdAnsatz::HardwareEfficient { num_qubits: 2, depth: 3 }, &vqd_config);
//! assert!(vqd_result.energies[1] >= vqd_result.energies[0] - 1e-6);
//! ```

use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// LOCAL HELPERS
// ============================================================

type C64 = Complex64;

#[inline]
fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

#[inline]
fn c64_zero() -> C64 {
    c64(0.0, 0.0)
}

#[inline]
fn c64_one() -> C64 {
    c64(1.0, 0.0)
}

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from quantum dynamics algorithms.
#[derive(Debug, Clone)]
pub enum QuantumDynamicsError {
    /// Hamiltonian specification is invalid.
    InvalidHamiltonian(String),
    /// Convergence failure.
    ConvergenceFailed(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Singular matrix encountered in linear solve.
    SingularMatrix(String),
}

impl std::fmt::Display for QuantumDynamicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHamiltonian(msg) => write!(f, "InvalidHamiltonian: {}", msg),
            Self::ConvergenceFailed(msg) => write!(f, "ConvergenceFailed: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "DimensionMismatch: expected {expected}, got {got}")
            }
            Self::SingularMatrix(msg) => write!(f, "SingularMatrix: {}", msg),
        }
    }
}

impl std::error::Error for QuantumDynamicsError {}

// ============================================================
// PAULI HAMILTONIAN
// ============================================================

/// A Hamiltonian expressed as a sum of weighted Pauli strings.
///
/// Each term is (coefficient, [(qubit_index, pauli_char)]) where the char is
/// one of 'I', 'X', 'Y', 'Z'. Example: 0.5 * X_0 Z_1 is stored as
/// `(0.5, vec![(0, 'X'), (1, 'Z')])`.
#[derive(Debug, Clone)]
pub struct PauliHamiltonian {
    /// Terms of the Hamiltonian: (coefficient, pauli_string).
    pub terms: Vec<(f64, Vec<(usize, char)>)>,
    /// Number of qubits (cached).
    num_qubits: usize,
}

impl PauliHamiltonian {
    /// Create a Hamiltonian from raw terms.
    pub fn new(terms: Vec<(f64, Vec<(usize, char)>)>) -> Self {
        let num_qubits = terms
            .iter()
            .flat_map(|(_, ops)| ops.iter().map(|&(q, _)| q + 1))
            .max()
            .unwrap_or(0);
        Self { terms, num_qubits }
    }

    /// Number of qubits this Hamiltonian acts on.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of Pauli terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Build the full 2^n x 2^n matrix representation (row-major, flat).
    pub fn to_matrix(&self) -> Vec<C64> {
        let n = self.num_qubits;
        let dim = 1usize << n;
        let mut mat = vec![c64_zero(); dim * dim];

        for (coeff, ops) in &self.terms {
            // Build the Pauli string tensor product as a sparse operator
            // and accumulate into the full matrix.
            add_pauli_term_to_matrix(&mut mat, dim, n, *coeff, ops);
        }
        mat
    }

    /// Compute <state|H|state> by direct statevector manipulation.
    ///
    /// For each Pauli term, applies the Pauli string to a copy of the state
    /// and takes the inner product. This is O(num_terms * 2^n) rather than
    /// O(4^n) for full matrix construction.
    pub fn expectation_value(&self, state: &[C64]) -> f64 {
        let mut energy = 0.0;
        for (coeff, ops) in &self.terms {
            let mut scratch = state.to_vec();
            apply_pauli_string(&mut scratch, self.num_qubits, ops);
            // <state|P|state> = sum_i conj(state[i]) * scratch[i]
            let inner: C64 = state
                .iter()
                .zip(scratch.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            energy += coeff * inner.re;
        }
        energy
    }

    // ----------------------------------------------------------------
    // Constructor helpers for common Hamiltonians
    // ----------------------------------------------------------------

    /// Transverse-field Ising model on a 1D chain with open boundaries:
    ///
    /// H = -J * sum_{i} Z_i Z_{i+1} - h * sum_i X_i
    pub fn transverse_field_ising(n: usize, j: f64, h: f64) -> Self {
        let mut terms = Vec::new();
        // ZZ couplings
        for i in 0..n.saturating_sub(1) {
            terms.push((-j, vec![(i, 'Z'), (i + 1, 'Z')]));
        }
        // Transverse field
        for i in 0..n {
            terms.push((-h, vec![(i, 'X')]));
        }
        Self::new(terms)
    }

    /// Heisenberg XXZ model on a 1D chain with open boundaries:
    ///
    /// H = sum_i [Jxy (X_i X_{i+1} + Y_i Y_{i+1}) + Jz Z_i Z_{i+1}] + h sum_i Z_i
    pub fn heisenberg_xxz(n: usize, jxy: f64, jz: f64, h: f64) -> Self {
        let mut terms = Vec::new();
        for i in 0..n.saturating_sub(1) {
            terms.push((jxy, vec![(i, 'X'), (i + 1, 'X')]));
            terms.push((jxy, vec![(i, 'Y'), (i + 1, 'Y')]));
            terms.push((jz, vec![(i, 'Z'), (i + 1, 'Z')]));
        }
        for i in 0..n {
            terms.push((h, vec![(i, 'Z')]));
        }
        Self::new(terms)
    }

    /// Simple hydrogen-chain-like Hamiltonian (toy model for chemistry tests).
    ///
    /// Uses a minimal encoding: nearest-neighbour XX+YY hopping plus on-site ZZ
    /// interaction and a constant energy offset, roughly mimicking the structure
    /// of a molecular hydrogen chain in minimal basis.
    pub fn hydrogen_chain(n_sites: usize) -> Self {
        let n = 2 * n_sites; // 2 spin-orbitals per site
        let mut terms = Vec::new();
        // Hopping
        for i in 0..n.saturating_sub(1) {
            terms.push((-0.5, vec![(i, 'X'), (i + 1, 'X')]));
            terms.push((-0.5, vec![(i, 'Y'), (i + 1, 'Y')]));
        }
        // On-site interaction
        for s in 0..n_sites {
            terms.push((0.25, vec![(2 * s, 'Z'), (2 * s + 1, 'Z')]));
        }
        // Constant offset (nuclear repulsion analogue)
        // Encoded as identity term (coefficient only, empty Pauli string)
        terms.push((0.7 * n_sites as f64, vec![]));
        Self::new(terms)
    }
}

// ============================================================
// PAULI STRING OPERATIONS
// ============================================================

/// Apply a Pauli string (tensor product of single-qubit Paulis) to a state vector
/// in-place. Each entry in `ops` is (qubit_index, pauli_char).
fn apply_pauli_string(state: &mut [C64], num_qubits: usize, ops: &[(usize, char)]) {
    for &(qubit, pauli) in ops {
        apply_single_pauli(state, num_qubits, qubit, pauli);
    }
}

/// Apply a single Pauli operator to qubit `q` of a state vector.
fn apply_single_pauli(state: &mut [C64], num_qubits: usize, q: usize, pauli: char) {
    let dim = state.len();
    let mask = 1usize << (num_qubits - 1 - q);

    match pauli {
        'I' => {} // identity does nothing
        'X' => {
            for basis in 0..dim {
                if basis & mask == 0 {
                    let partner = basis | mask;
                    state.swap(basis, partner);
                }
            }
        }
        'Y' => {
            // Y = [[0, -i], [i, 0]]
            let neg_i = c64(0.0, -1.0);
            let pos_i = c64(0.0, 1.0);
            for basis in 0..dim {
                if basis & mask == 0 {
                    let partner = basis | mask;
                    let a = state[basis];
                    let b = state[partner];
                    state[basis] = neg_i * b;
                    state[partner] = pos_i * a;
                }
            }
        }
        'Z' => {
            for basis in 0..dim {
                if basis & mask != 0 {
                    state[basis] = -state[basis];
                }
            }
        }
        _ => {} // treat unknown as identity
    }
}

/// Add a single Pauli term (coeff * P) to a dense matrix (row-major flat).
fn add_pauli_term_to_matrix(
    mat: &mut [C64],
    dim: usize,
    num_qubits: usize,
    coeff: f64,
    ops: &[(usize, char)],
) {
    let c = c64(coeff, 0.0);
    for col in 0..dim {
        // Apply the Pauli string to the basis vector |col>
        let mut target_row = col;
        let mut phase = c64_one();

        for &(q, pauli) in ops {
            let bit_pos = num_qubits - 1 - q;
            let bit = (target_row >> bit_pos) & 1;
            match pauli {
                'I' => {}
                'X' => {
                    target_row ^= 1 << bit_pos;
                }
                'Y' => {
                    target_row ^= 1 << bit_pos;
                    if bit == 0 {
                        phase *= c64(0.0, 1.0); // i
                    } else {
                        phase *= c64(0.0, -1.0); // -i
                    }
                }
                'Z' => {
                    if bit == 1 {
                        phase *= c64(-1.0, 0.0);
                    }
                }
                _ => {}
            }
        }
        mat[target_row * dim + col] += c * phase;
    }
}

// ============================================================
// DYNAMICS STATE (self-contained statevector)
// ============================================================

/// Self-contained statevector for quantum dynamics simulations.
///
/// Provides gate operations, Pauli rotations, and inner products without
/// depending on the top-level `QuantumState` type.
#[derive(Debug, Clone)]
pub struct DynamicsState {
    /// Amplitudes in the computational basis.
    pub amplitudes: Vec<C64>,
    /// Number of qubits.
    pub num_qubits: usize,
}

impl DynamicsState {
    /// Create the |0...0> state.
    pub fn zero_state(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut amps = vec![c64_zero(); dim];
        amps[0] = c64_one();
        Self {
            amplitudes: amps,
            num_qubits,
        }
    }

    /// Create the |+...+> state (uniform superposition).
    pub fn plus_state(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let amp = c64(1.0 / (dim as f64).sqrt(), 0.0);
        Self {
            amplitudes: vec![amp; dim],
            num_qubits,
        }
    }

    /// Create a state from a given amplitude vector (normalised).
    pub fn from_amplitudes(amplitudes: Vec<C64>, num_qubits: usize) -> Self {
        assert_eq!(amplitudes.len(), 1 << num_qubits);
        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Dimension of the Hilbert space.
    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    /// Normalise the state in-place.
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq > 1e-30 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for a in &mut self.amplitudes {
                *a *= inv_norm;
            }
        }
    }

    /// Norm of the state vector.
    pub fn norm(&self) -> f64 {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        norm_sq.sqrt()
    }

    /// Inner product <self|other>.
    pub fn inner_product(&self, other: &DynamicsState) -> C64 {
        self.amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }

    /// Overlap probability |<self|other>|^2.
    pub fn overlap(&self, other: &DynamicsState) -> f64 {
        self.inner_product(other).norm_sqr()
    }

    /// Apply a single Pauli rotation exp(-i * angle * P_q) where P is X, Y, or Z.
    pub fn apply_pauli_rotation(&mut self, qubit: usize, pauli: char, angle: f64) {
        let mask = 1usize << (self.num_qubits - 1 - qubit);
        let dim = self.dim();

        match pauli {
            'X' => {
                // exp(-i*angle*X) = cos(angle)*I - i*sin(angle)*X
                let c = c64(angle.cos(), 0.0);
                let s = c64(0.0, -angle.sin());
                for basis in 0..dim {
                    if basis & mask == 0 {
                        let partner = basis | mask;
                        let a = self.amplitudes[basis];
                        let b = self.amplitudes[partner];
                        self.amplitudes[basis] = c * a + s * b;
                        self.amplitudes[partner] = s * a + c * b;
                    }
                }
            }
            'Y' => {
                // exp(-i*angle*Y) = cos(angle)*I - i*sin(angle)*Y
                // Y|0> = i|1>, Y|1> = -i|0>
                // So: exp(-i*a*Y)|0> = cos(a)|0> + sin(a)|1>
                //     exp(-i*a*Y)|1> = -sin(a)|0> + cos(a)|1>
                let c = angle.cos();
                let s = angle.sin();
                for basis in 0..dim {
                    if basis & mask == 0 {
                        let partner = basis | mask;
                        let a = self.amplitudes[basis];
                        let b = self.amplitudes[partner];
                        self.amplitudes[basis] = c64(c, 0.0) * a + c64(s, 0.0) * b;
                        self.amplitudes[partner] = c64(-s, 0.0) * a + c64(c, 0.0) * b;
                    }
                }
            }
            'Z' => {
                // exp(-i*angle*Z) = diag(exp(-i*angle), exp(i*angle))
                let phase_0 = c64(angle.cos(), -angle.sin());
                let phase_1 = c64(angle.cos(), angle.sin());
                for basis in 0..dim {
                    if basis & mask == 0 {
                        self.amplitudes[basis] *= phase_0;
                        self.amplitudes[basis | mask] *= phase_1;
                    }
                }
            }
            _ => {} // identity: nothing
        }
    }

    /// Apply Hadamard gate to qubit q.
    pub fn hadamard(&mut self, q: usize) {
        let mask = 1usize << (self.num_qubits - 1 - q);
        let inv_sqrt2 = c64(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let dim = self.dim();
        for basis in 0..dim {
            if basis & mask == 0 {
                let partner = basis | mask;
                let a = self.amplitudes[basis];
                let b = self.amplitudes[partner];
                self.amplitudes[basis] = inv_sqrt2 * (a + b);
                self.amplitudes[partner] = inv_sqrt2 * (a - b);
            }
        }
    }

    /// Apply CNOT with control `c` and target `t`.
    pub fn cnot(&mut self, c: usize, t: usize) {
        let c_mask = 1usize << (self.num_qubits - 1 - c);
        let t_mask = 1usize << (self.num_qubits - 1 - t);
        let dim = self.dim();
        for basis in 0..dim {
            if (basis & c_mask != 0) && (basis & t_mask == 0) {
                let partner = basis | t_mask;
                self.amplitudes.swap(basis, partner);
            }
        }
    }

    /// Apply Ry(theta) rotation to qubit q.
    pub fn ry(&mut self, q: usize, theta: f64) {
        self.apply_pauli_rotation(q, 'Y', theta / 2.0);
    }

    /// Apply Rz(theta) rotation to qubit q.
    pub fn rz(&mut self, q: usize, theta: f64) {
        self.apply_pauli_rotation(q, 'Z', theta / 2.0);
    }

    /// Apply Rx(theta) rotation to qubit q.
    pub fn rx(&mut self, q: usize, theta: f64) {
        self.apply_pauli_rotation(q, 'X', theta / 2.0);
    }

    /// Apply a dense unitary matrix (flat row-major) to the full state.
    ///
    /// Used for matrix exponential application on small systems.
    pub fn apply_matrix(&mut self, matrix: &[C64]) {
        let dim = self.dim();
        assert_eq!(matrix.len(), dim * dim, "Matrix dimension mismatch");
        let old = self.amplitudes.clone();
        for i in 0..dim {
            let mut sum = c64_zero();
            for j in 0..dim {
                sum += matrix[i * dim + j] * old[j];
            }
            self.amplitudes[i] = sum;
        }
    }
}

// ============================================================
// LINEAR ALGEBRA UTILITIES
// ============================================================

/// Solve A*x = b using LU decomposition with partial pivoting.
///
/// `a` is the NxN matrix in row-major flat form. `b` is the right-hand side.
/// Returns the solution vector x.
fn lu_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, QuantumDynamicsError> {
    // Copy A so we can modify in-place
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let rhs = b.to_vec();

    // LU factorisation with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = 0.0f64;
        let mut max_row = col;
        for row in col..n {
            let val = lu[perm[row] * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            // Singular or nearly singular -- add regularisation
            lu[perm[col] * n + col] += 1e-10;
        }
        perm.swap(col, max_row);

        let pivot = lu[perm[col] * n + col];
        for row in (col + 1)..n {
            let factor = lu[perm[row] * n + col] / pivot;
            lu[perm[row] * n + col] = factor; // store L factor
            for k in (col + 1)..n {
                let val = lu[perm[col] * n + k];
                lu[perm[row] * n + k] -= factor * val;
            }
        }
    }

    // Forward substitution (Ly = Pb)
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[perm[i]];
        for j in 0..i {
            sum -= lu[perm[i] * n + j] * y[j];
        }
        y[i] = sum;
    }

    // Back substitution (Ux = y)
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu[perm[i] * n + j] * x[j];
        }
        let diag = lu[perm[i] * n + i];
        if diag.abs() < 1e-30 {
            return Err(QuantumDynamicsError::SingularMatrix(
                "Zero diagonal in U during LU solve".to_string(),
            ));
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

/// Matrix exponential exp(A) for a small complex matrix using the scaling-and-squaring
/// method with a 13th-order Pade approximant (simplified for matrices up to ~16x16).
///
/// For quantum dynamics we typically need exp(-i*H*dt) where H is Hermitian,
/// but this works for any small matrix.
fn matrix_exponential(matrix: &[C64], dim: usize) -> Vec<C64> {
    // Scaling: find s such that ||A/2^s|| < 1
    let norm = matrix_inf_norm(matrix, dim);
    let s = if norm > 1.0 {
        (norm.log2().ceil() as u32) + 1
    } else {
        0
    };
    let scale = 0.5f64.powi(s as i32);

    // Scale the matrix
    let scaled: Vec<C64> = matrix.iter().map(|&x| x * c64(scale, 0.0)).collect();

    // Pade approximation of order 6: exp(A) ~ P(A) / Q(A)
    // where P(A) = sum_{k=0}^{p} c_k A^k, Q(A) = sum_{k=0}^{p} (-1)^k c_k A^k
    // and c_k = (2p - k)! p! / ((2p)! k! (p-k)!)
    let p = 6;
    let coeffs = pade_coefficients(p);

    // Compute powers of scaled matrix: A^0 = I, A^1, A^2, ... A^p
    let identity = complex_identity(dim);
    let mut powers: Vec<Vec<C64>> = Vec::with_capacity(p + 1);
    powers.push(identity.clone());
    if p >= 1 {
        powers.push(scaled.clone());
    }
    for k in 2..=p {
        let prev = &powers[k - 1];
        let prod = complex_matmul(prev, &scaled, dim);
        powers.push(prod);
    }

    // P = sum c_k * A^k
    let mut p_mat = vec![c64_zero(); dim * dim];
    let mut q_mat = vec![c64_zero(); dim * dim];
    for k in 0..=p {
        let c = c64(coeffs[k], 0.0);
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        for idx in 0..(dim * dim) {
            p_mat[idx] += c * powers[k][idx];
            q_mat[idx] += c64(sign * coeffs[k], 0.0) * powers[k][idx];
        }
    }

    // Solve Q * result = P  =>  result = Q^{-1} P
    let result = complex_matrix_solve(&q_mat, &p_mat, dim);

    // Squaring: result = result^(2^s)
    let mut mat = result;
    for _ in 0..s {
        mat = complex_matmul(&mat, &mat, dim);
    }
    mat
}

/// Pade coefficients c_k for order p.
fn pade_coefficients(p: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0; p + 1];
    coeffs[0] = 1.0;
    for k in 1..=p {
        coeffs[k] = coeffs[k - 1] * ((p + 1 - k) as f64) / ((k * (2 * p + 1 - k)) as f64);
    }
    coeffs
}

/// Infinity norm of a complex matrix.
fn matrix_inf_norm(mat: &[C64], dim: usize) -> f64 {
    let mut max_row_sum = 0.0f64;
    for i in 0..dim {
        let row_sum: f64 = (0..dim).map(|j| mat[i * dim + j].norm()).sum();
        max_row_sum = max_row_sum.max(row_sum);
    }
    max_row_sum
}

/// Complex identity matrix (flat row-major).
fn complex_identity(dim: usize) -> Vec<C64> {
    let mut mat = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        mat[i * dim + i] = c64_one();
    }
    mat
}

/// Complex matrix multiplication C = A * B.
fn complex_matmul(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
    let mut c = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let a_ik = a[i * dim + k];
            if a_ik.norm_sqr() < 1e-30 {
                continue;
            }
            for j in 0..dim {
                c[i * dim + j] += a_ik * b[k * dim + j];
            }
        }
    }
    c
}

/// Solve Q * X = P for matrices (X = Q^{-1} * P) using Gaussian elimination
/// with partial pivoting. Both Q and P are dim x dim, flat row-major.
fn complex_matrix_solve(q: &[C64], p: &[C64], dim: usize) -> Vec<C64> {
    // Augmented matrix [Q | P]
    let mut aug = vec![c64_zero(); dim * 2 * dim];
    for i in 0..dim {
        for j in 0..dim {
            aug[i * 2 * dim + j] = q[i * dim + j];
            aug[i * 2 * dim + dim + j] = p[i * dim + j];
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..dim {
        // Find pivot
        let mut max_val = 0.0f64;
        let mut max_row = col;
        for row in col..dim {
            let val = aug[row * 2 * dim + col].norm();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-20 {
            // Nearly singular: add small regularisation
            aug[col * 2 * dim + col] += c64(1e-12, 0.0);
        }

        // Swap rows
        if max_row != col {
            for k in 0..(2 * dim) {
                let tmp = aug[col * 2 * dim + k];
                aug[col * 2 * dim + k] = aug[max_row * 2 * dim + k];
                aug[max_row * 2 * dim + k] = tmp;
            }
        }

        let pivot = aug[col * 2 * dim + col];
        for row in (col + 1)..dim {
            let factor = aug[row * 2 * dim + col] / pivot;
            for k in col..(2 * dim) {
                let val = aug[col * 2 * dim + k];
                aug[row * 2 * dim + k] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut result = vec![c64_zero(); dim * dim];
    for col_rhs in 0..dim {
        for i in (0..dim).rev() {
            let mut sum = aug[i * 2 * dim + dim + col_rhs];
            for j in (i + 1)..dim {
                sum -= aug[i * 2 * dim + j] * result[j * dim + col_rhs];
            }
            result[i * dim + col_rhs] = sum / aug[i * 2 * dim + i];
        }
    }

    result
}

/// Eigendecomposition of a Hermitian matrix via the Jacobi eigenvalue algorithm.
///
/// Returns (eigenvalues, eigenvectors_column_major). The eigenvectors are stored
/// as columns of a dim x dim matrix in row-major layout, i.e., eigenvector k
/// is at indices [i * dim + k] for i = 0..dim.
fn hermitian_eigendecompose(matrix: &[C64], dim: usize) -> (Vec<f64>, Vec<C64>) {
    // Convert Hermitian matrix to real symmetric by working in real arithmetic
    // (valid when imaginary parts are negligible, or we do a full complex Jacobi).
    //
    // For small quantum systems (2-4 qubits), we use the full matrix approach
    // with iterative diagonalisation.

    // Use power-iteration / QR-like approach via Householder reduction.
    // For small dim, just do direct eigendecomposition via the characteristic
    // polynomial or Jacobi rotations.

    // Practical approach: convert to real and use Jacobi if matrix is real-symmetric,
    // otherwise use a complex Jacobi method.

    // Check if the matrix is real (common for Pauli Hamiltonians with no Y terms
    // or even numbers of Y terms).
    let is_real = matrix.iter().all(|c| c.im.abs() < 1e-12);

    if is_real {
        real_symmetric_eigen(matrix, dim)
    } else {
        complex_hermitian_eigen(matrix, dim)
    }
}

/// Jacobi eigenvalue algorithm for real symmetric matrices.
fn real_symmetric_eigen(matrix: &[C64], dim: usize) -> (Vec<f64>, Vec<C64>) {
    let mut a: Vec<f64> = matrix.iter().map(|c| c.re).collect();
    let mut v = vec![0.0f64; dim * dim]; // eigenvectors
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }

    let max_iter = 100 * dim * dim;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = a[i * dim + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < 1e-14 {
            break;
        }

        // Compute rotation angle
        let app = a[p * dim + p];
        let aqq = a[q * dim + q];
        let apq = a[p * dim + q];

        let theta = if (aqq - app).abs() < 1e-30 {
            PI / 4.0
        } else {
            0.5 * (2.0 * apq / (aqq - app)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Jacobi rotation to A: A' = G^T A G
        // Update columns/rows p and q
        let mut new_col_p = vec![0.0; dim];
        let mut new_col_q = vec![0.0; dim];

        for i in 0..dim {
            new_col_p[i] = cos_t * a[i * dim + p] - sin_t * a[i * dim + q];
            new_col_q[i] = sin_t * a[i * dim + p] + cos_t * a[i * dim + q];
        }
        for i in 0..dim {
            a[i * dim + p] = new_col_p[i];
            a[i * dim + q] = new_col_q[i];
            a[p * dim + i] = new_col_p[i]; // symmetric
            a[q * dim + i] = new_col_q[i];
        }

        // Fix the 2x2 block
        a[p * dim + p] = cos_t * new_col_p[p] - sin_t * new_col_p[q];
        a[q * dim + q] = sin_t * new_col_q[p] + cos_t * new_col_q[q];
        a[p * dim + q] = 0.0;
        a[q * dim + p] = 0.0;

        // Accumulate eigenvectors
        for i in 0..dim {
            let vip = v[i * dim + p];
            let viq = v[i * dim + q];
            v[i * dim + p] = cos_t * vip - sin_t * viq;
            v[i * dim + q] = sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..dim).map(|i| a[i * dim + i]).collect();
    let eigenvectors: Vec<C64> = v.iter().map(|&r| c64(r, 0.0)).collect();

    // Sort by eigenvalue (ascending)
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_vecs = vec![c64_zero(); dim * dim];
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..dim {
            sorted_vecs[row * dim + new_col] = eigenvectors[row * dim + old_col];
        }
    }

    (sorted_vals, sorted_vecs)
}

/// Complex Hermitian eigendecomposition via Jacobi-like rotations.
fn complex_hermitian_eigen(matrix: &[C64], dim: usize) -> (Vec<f64>, Vec<C64>) {
    // For small complex Hermitian matrices, convert to a real symmetric matrix
    // of double dimension: embed C^n as R^{2n}.
    // H = A + iB where A is symmetric, B is anti-symmetric.
    // The real representation is:
    // M = [[ A, -B ],
    //      [ B,  A ]]
    // with eigenvalues appearing in pairs (lambda, lambda).

    let n2 = 2 * dim;
    let mut real_mat = vec![c64_zero(); n2 * n2];

    for i in 0..dim {
        for j in 0..dim {
            let c = matrix[i * dim + j];
            // Top-left: A (real part)
            real_mat[i * n2 + j] = c64(c.re, 0.0);
            // Top-right: -B (negative imaginary part)
            real_mat[i * n2 + (dim + j)] = c64(-c.im, 0.0);
            // Bottom-left: B (imaginary part)
            real_mat[(dim + i) * n2 + j] = c64(c.im, 0.0);
            // Bottom-right: A (real part)
            real_mat[(dim + i) * n2 + (dim + j)] = c64(c.re, 0.0);
        }
    }

    let (all_vals, all_vecs) = real_symmetric_eigen(&real_mat, n2);

    // Extract unique eigenvalues (each appears twice) and reconstruct
    // complex eigenvectors from the real representation.
    let mut eigenvalues = Vec::with_capacity(dim);
    let mut eigenvectors = vec![c64_zero(); dim * dim];
    let mut used = vec![false; n2];

    for k in 0..n2 {
        if used[k] {
            continue;
        }
        if eigenvalues.len() >= dim {
            break;
        }

        let idx = eigenvalues.len();
        eigenvalues.push(all_vals[k]);

        // Reconstruct complex eigenvector: v_complex[i] = v_real[i] + i * v_real[dim + i]
        for i in 0..dim {
            eigenvectors[i * dim + idx] =
                c64(all_vecs[i * n2 + k].re, all_vecs[(dim + i) * n2 + k].re);
        }

        // Mark the paired eigenvalue as used
        for k2 in (k + 1)..n2 {
            if !used[k2] && (all_vals[k2] - all_vals[k]).abs() < 1e-10 {
                used[k2] = true;
                break;
            }
        }
        used[k] = true;
    }

    (eigenvalues, eigenvectors)
}

// ============================================================
// EXACT SOLUTIONS (for validation)
// ============================================================

/// Compute the exact ground state energy and eigenvector via full diagonalisation.
///
/// Returns (ground_energy, ground_state_amplitudes).
pub fn exact_ground_state(
    hamiltonian: &PauliHamiltonian,
) -> (f64, Vec<C64>) {
    let dim = 1usize << hamiltonian.num_qubits();
    let mat = hamiltonian.to_matrix();
    let (eigenvalues, eigenvectors) = hermitian_eigendecompose(&mat, dim);

    let energy = eigenvalues[0];
    let state: Vec<C64> = (0..dim).map(|i| eigenvectors[i * dim]).collect();
    (energy, state)
}

/// Compute the first `num_states` eigenvalues via full diagonalisation.
pub fn exact_eigenvalues(hamiltonian: &PauliHamiltonian, num_states: usize) -> Vec<f64> {
    let dim = 1usize << hamiltonian.num_qubits();
    let mat = hamiltonian.to_matrix();
    let (eigenvalues, _) = hermitian_eigendecompose(&mat, dim);
    eigenvalues.into_iter().take(num_states).collect()
}

/// Compute the first `num_states` eigenpairs via full diagonalisation.
pub fn exact_eigenpairs(
    hamiltonian: &PauliHamiltonian,
    num_states: usize,
) -> (Vec<f64>, Vec<Vec<C64>>) {
    let dim = 1usize << hamiltonian.num_qubits();
    let mat = hamiltonian.to_matrix();
    let (eigenvalues, eigenvectors) = hermitian_eigendecompose(&mat, dim);

    let n = num_states.min(dim);
    let vals: Vec<f64> = eigenvalues[..n].to_vec();
    let vecs: Vec<Vec<C64>> = (0..n)
        .map(|k| (0..dim).map(|i| eigenvectors[i * dim + k]).collect())
        .collect();
    (vals, vecs)
}

// ============================================================
// QITE: QUANTUM IMAGINARY TIME EVOLUTION
// ============================================================

/// Configuration for the QITE algorithm.
#[derive(Debug, Clone)]
pub struct QiteConfig {
    /// Imaginary time step size dtau.
    pub time_step: f64,
    /// Maximum number of imaginary-time steps.
    pub num_steps: usize,
    /// Energy convergence threshold.
    pub convergence_threshold: f64,
    /// Maximum domain size for the Pauli basis expansion (limits the linear
    /// system size). 0 means use all single-qubit Paulis.
    pub max_domain_size: usize,
    /// Optional initial state. If None, starts from |+...+>.
    pub initial_state: Option<Vec<C64>>,
}

impl Default for QiteConfig {
    fn default() -> Self {
        Self {
            time_step: 0.1,
            num_steps: 200,
            convergence_threshold: 1e-8,
            max_domain_size: 0,
            initial_state: None,
        }
    }
}

impl QiteConfig {
    pub fn time_step(mut self, dt: f64) -> Self {
        self.time_step = dt;
        self
    }

    pub fn num_steps(mut self, n: usize) -> Self {
        self.num_steps = n;
        self
    }

    pub fn convergence_threshold(mut self, tol: f64) -> Self {
        self.convergence_threshold = tol;
        self
    }

    pub fn initial_state(mut self, state: Vec<C64>) -> Self {
        self.initial_state = Some(state);
        self
    }
}

/// Result of QITE computation.
#[derive(Debug, Clone)]
pub struct QiteResult {
    /// Energy at each step.
    pub energy_history: Vec<f64>,
    /// Final state vector.
    pub final_state: Vec<C64>,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Number of steps actually taken.
    pub num_steps_taken: usize,
    /// Final energy value.
    pub final_energy: f64,
}

/// Run QITE to find the ground state of a Hamiltonian.
///
/// The algorithm evolves the state in imaginary time:
///   |psi(tau + dtau)> ~ (1 - H*dtau) |psi(tau)> / norm
///
/// Each step, the non-unitary imaginary-time operator is approximated by a
/// unitary generated from a local Pauli basis via the QITE equation:
///   A * x = b
/// where A_{ij} = Re(<psi| sigma_i^dag sigma_j |psi>) and
///       b_i   = -Im(<psi| sigma_i^dag H |psi>)
///
/// For efficiency on small systems (<=4 qubits), we use the direct
/// (exact) imaginary-time approach: apply exp(-H*dtau) and normalise.
pub fn qite_ground_state(
    hamiltonian: &PauliHamiltonian,
    config: &QiteConfig,
) -> QiteResult {
    let n = hamiltonian.num_qubits();
    let dim = 1usize << n;
    let dtau = config.time_step;

    // For small systems, use exact imaginary-time propagation
    // (matrix exponential). For larger systems we would use the
    // QITE unitary approximation.
    if n <= 6 {
        return qite_exact_propagation(hamiltonian, config);
    }

    // QITE unitary approximation for larger systems
    qite_unitary_approx(hamiltonian, config)
}

/// Exact imaginary-time propagation via matrix exponential.
///
/// Computes exp(-H*dtau) at each step and applies it, then normalises.
/// This is exact but limited to small systems.
fn qite_exact_propagation(
    hamiltonian: &PauliHamiltonian,
    config: &QiteConfig,
) -> QiteResult {
    let n = hamiltonian.num_qubits();
    let dim = 1usize << n;
    let dtau = config.time_step;

    // Build the Hamiltonian matrix
    let h_mat = hamiltonian.to_matrix();

    // Construct -H*dtau (real matrix since H is Hermitian and we want exp(-H*dtau))
    let neg_h_dtau: Vec<C64> = h_mat.iter().map(|&c| c * c64(-dtau, 0.0)).collect();

    // Compute propagator U = exp(-H * dtau)
    let propagator = matrix_exponential(&neg_h_dtau, dim);

    // Initial state
    let mut state = if let Some(ref init) = config.initial_state {
        DynamicsState::from_amplitudes(init.clone(), n)
    } else {
        DynamicsState::plus_state(n)
    };

    let mut energy_history = Vec::with_capacity(config.num_steps);
    let initial_energy = hamiltonian.expectation_value(&state.amplitudes);
    energy_history.push(initial_energy);

    let mut converged = false;
    let mut steps_taken = 0;

    for step in 0..config.num_steps {
        // Apply exp(-H * dtau)
        state.apply_matrix(&propagator);
        state.normalize();

        let energy = hamiltonian.expectation_value(&state.amplitudes);
        energy_history.push(energy);
        steps_taken = step + 1;

        // Check convergence
        if step > 0 {
            let de = (energy - energy_history[step]).abs();
            if de < config.convergence_threshold {
                converged = true;
                break;
            }
        }
    }

    let final_energy = *energy_history.last().unwrap();

    QiteResult {
        energy_history,
        final_state: state.amplitudes,
        converged,
        num_steps_taken: steps_taken,
        final_energy,
    }
}

/// QITE via unitary approximation (for systems too large for matrix exponential).
///
/// Each step, we:
/// 1. Build the Pauli basis set {sigma_k} for the unitary approximation.
/// 2. Compute A_{ij} = Re(<psi| sigma_i^dag sigma_j |psi>).
/// 3. Compute b_i = -dtau * Re(<psi| sigma_i H |psi>) (the imaginary-time gradient).
/// 4. Solve A*x = b for the rotation parameters.
/// 5. Apply exp(-i * sum_k x_k sigma_k) to |psi>.
fn qite_unitary_approx(
    hamiltonian: &PauliHamiltonian,
    config: &QiteConfig,
) -> QiteResult {
    let n = hamiltonian.num_qubits();
    let dtau = config.time_step;

    // Build Pauli basis: all single-qubit X, Y, Z operators
    let mut pauli_basis: Vec<(usize, char)> = Vec::new();
    for q in 0..n {
        for &p in &['X', 'Y', 'Z'] {
            pauli_basis.push((q, p));
        }
    }

    // Limit domain size if configured
    let basis_size = if config.max_domain_size > 0 && config.max_domain_size < pauli_basis.len() {
        config.max_domain_size
    } else {
        pauli_basis.len()
    };
    let pauli_basis = &pauli_basis[..basis_size];

    let mut state = if let Some(ref init) = config.initial_state {
        DynamicsState::from_amplitudes(init.clone(), n)
    } else {
        DynamicsState::plus_state(n)
    };

    let mut energy_history = Vec::with_capacity(config.num_steps);
    let initial_energy = hamiltonian.expectation_value(&state.amplitudes);
    energy_history.push(initial_energy);

    let mut converged = false;
    let mut steps_taken = 0;

    for step in 0..config.num_steps {
        let m = basis_size;

        // Build A matrix: A_{ij} = Re(<psi| sigma_i sigma_j |psi>)
        let mut a_mat = vec![0.0; m * m];
        for i in 0..m {
            for j in i..m {
                let mut tmp = state.amplitudes.clone();
                // Apply sigma_j then sigma_i
                apply_single_pauli(&mut tmp, n, pauli_basis[j].0, pauli_basis[j].1);
                apply_single_pauli(&mut tmp, n, pauli_basis[i].0, pauli_basis[i].1);
                let val: C64 = state
                    .amplitudes
                    .iter()
                    .zip(tmp.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                a_mat[i * m + j] = val.re;
                a_mat[j * m + i] = val.re; // symmetric
            }
        }

        // Build b vector: b_i = -dtau * Re(<psi| sigma_i H |psi>)
        let mut b_vec = vec![0.0; m];

        // First compute H|psi>
        let dim = state.dim();
        let h_mat = hamiltonian.to_matrix();
        let mut h_psi = vec![c64_zero(); dim];
        for r in 0..dim {
            for c in 0..dim {
                h_psi[r] += h_mat[r * dim + c] * state.amplitudes[c];
            }
        }

        for i in 0..m {
            let mut tmp = h_psi.clone();
            apply_single_pauli(&mut tmp, n, pauli_basis[i].0, pauli_basis[i].1);
            let val: C64 = state
                .amplitudes
                .iter()
                .zip(tmp.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            // The QITE equation: b_i = -dtau * <psi|sigma_i|dpsi/dtau>
            // where d|psi>/dtau = -(H - E)|psi>, so:
            // b_i = dtau * Re(<psi| sigma_i (H - E) |psi>)
            let e = energy_history.last().unwrap();
            // <psi|sigma_i|psi>
            let mut sigma_psi = state.amplitudes.clone();
            apply_single_pauli(&mut sigma_psi, n, pauli_basis[i].0, pauli_basis[i].1);
            let sigma_exp: C64 = state
                .amplitudes
                .iter()
                .zip(sigma_psi.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            b_vec[i] = -dtau * (val.re - e * sigma_exp.re);
        }

        // Add regularisation to A for numerical stability
        for i in 0..m {
            a_mat[i * m + i] += 1e-8;
        }

        // Solve A * x = b
        let x = match lu_solve(&a_mat, &b_vec, m) {
            Ok(x) => x,
            Err(_) => vec![0.0; m], // fallback: no update
        };

        // Apply unitary exp(-i * sum_k x_k sigma_k) ~ product of individual rotations
        for (k, &(q, p)) in pauli_basis.iter().enumerate() {
            if x[k].abs() > 1e-15 {
                state.apply_pauli_rotation(q, p, x[k]);
            }
        }
        state.normalize();

        let energy = hamiltonian.expectation_value(&state.amplitudes);
        energy_history.push(energy);
        steps_taken = step + 1;

        // Check convergence
        if step > 0 {
            let de = (energy - energy_history[step]).abs();
            if de < config.convergence_threshold {
                converged = true;
                break;
            }
        }
    }

    let final_energy = *energy_history.last().unwrap();

    QiteResult {
        energy_history,
        final_state: state.amplitudes,
        converged,
        num_steps_taken: steps_taken,
        final_energy,
    }
}

/// Compute the thermal (Gibbs) state at inverse temperature beta via imaginary-time
/// evolution of the maximally mixed state.
///
/// rho(beta) = exp(-beta * H) / Z, where Z = Tr[exp(-beta * H)].
///
/// Returns the density matrix as a flat dim x dim vector.
pub fn qite_thermal_state(
    hamiltonian: &PauliHamiltonian,
    beta: f64,
    config: &QiteConfig,
) -> Vec<f64> {
    let n = hamiltonian.num_qubits();
    let dim = 1usize << n;

    // Build the full Hamiltonian matrix
    let h_mat = hamiltonian.to_matrix();

    // Compute exp(-beta * H)
    let neg_beta_h: Vec<C64> = h_mat.iter().map(|&c| c * c64(-beta, 0.0)).collect();
    let rho_unnorm = matrix_exponential(&neg_beta_h, dim);

    // The diagonal of the density matrix gives the populations.
    // Trace = sum of diagonal elements (for normalisation).
    let trace: f64 = (0..dim).map(|i| rho_unnorm[i * dim + i].re).sum();

    // Return the diagonal (populations in the energy basis are the Boltzmann weights).
    // For a full density matrix, we return all dim*dim real parts normalised.
    let mut populations = vec![0.0; dim];
    for i in 0..dim {
        populations[i] = rho_unnorm[i * dim + i].re / trace;
    }

    populations
}

// ============================================================
// VQD: VARIATIONAL QUANTUM DEFLATION
// ============================================================

/// Variational ansatz specification for VQD.
#[derive(Debug, Clone)]
pub enum VqdAnsatz {
    /// Hardware-efficient ansatz with alternating Ry/Rz rotations and CNOT entanglers.
    HardwareEfficient {
        num_qubits: usize,
        depth: usize,
    },
    /// Simplified UCCSD-inspired ansatz (single + double excitations).
    UCCSD {
        num_qubits: usize,
        num_electrons: usize,
    },
}

impl VqdAnsatz {
    /// Number of variational parameters.
    pub fn num_params(&self) -> usize {
        match self {
            VqdAnsatz::HardwareEfficient { num_qubits, depth } => {
                // Each layer: 2 * num_qubits rotations (Ry + Rz) per qubit
                // Plus initial layer of Ry + Rz
                2 * num_qubits * (depth + 1)
            }
            VqdAnsatz::UCCSD {
                num_qubits,
                num_electrons,
            } => {
                let n_occ = *num_electrons;
                let n_virt = num_qubits.saturating_sub(*num_electrons);
                // Single excitations + double excitations
                n_occ * n_virt + n_occ * (n_occ.saturating_sub(1)) / 2 * n_virt * n_virt.saturating_sub(1) / 2
            }
        }
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        match self {
            VqdAnsatz::HardwareEfficient { num_qubits, .. } => *num_qubits,
            VqdAnsatz::UCCSD { num_qubits, .. } => *num_qubits,
        }
    }

    /// Construct and apply the ansatz circuit to a state given parameters.
    pub fn apply(&self, state: &mut DynamicsState, params: &[f64]) {
        match self {
            VqdAnsatz::HardwareEfficient { num_qubits, depth } => {
                let n = *num_qubits;
                let mut idx = 0;

                // Initial rotation layer
                for q in 0..n {
                    state.ry(q, params[idx]);
                    idx += 1;
                    state.rz(q, params[idx]);
                    idx += 1;
                }

                for _layer in 0..*depth {
                    // Entangling layer: linear chain of CNOTs
                    for q in 0..n.saturating_sub(1) {
                        state.cnot(q, q + 1);
                    }
                    // Rotation layer
                    for q in 0..n {
                        state.ry(q, params[idx]);
                        idx += 1;
                        state.rz(q, params[idx]);
                        idx += 1;
                    }
                }
            }
            VqdAnsatz::UCCSD {
                num_qubits,
                num_electrons,
            } => {
                let n = *num_qubits;
                let n_occ = *num_electrons;
                let n_virt = n.saturating_sub(n_occ);
                let mut idx = 0;

                // Start from Hartree-Fock state: |1...1 0...0>
                // (first n_electrons qubits occupied)
                // The state should already be |0...0>, so we flip occupied qubits
                for q in 0..n_occ.min(n) {
                    // X gate to flip |0> to |1>
                    let mask = 1usize << (n - 1 - q);
                    for basis in 0..state.dim() {
                        if basis & mask == 0 {
                            let partner = basis | mask;
                            state.amplitudes.swap(basis, partner);
                        }
                    }
                }

                // Single excitations: Ry rotations between occupied and virtual
                for i in 0..n_occ.min(n) {
                    for a in n_occ..n {
                        if idx >= params.len() {
                            return;
                        }
                        let theta = params[idx];
                        idx += 1;
                        // Givens rotation between orbitals i and a
                        state.ry(i, theta);
                        if a < n {
                            state.cnot(i, a);
                            state.ry(a, theta * 0.5);
                            state.cnot(i, a);
                        }
                    }
                }

                // Double excitations (simplified)
                // Just apply remaining parameters as paired rotations
                while idx < params.len() {
                    let q = idx % n;
                    let theta = params[idx];
                    idx += 1;
                    state.rz(q, theta);
                    if q + 1 < n {
                        state.cnot(q, q + 1);
                    }
                }
            }
        }
    }
}

/// Configuration for the VQD algorithm.
#[derive(Debug, Clone)]
pub struct VqdConfig {
    /// Number of eigenstates to find.
    pub num_states: usize,
    /// Overlap penalty weights. If shorter than num_states - 1, the last value
    /// is reused.
    pub beta_penalties: Vec<f64>,
    /// Maximum optimiser iterations per eigenstate.
    pub max_iterations: usize,
    /// Convergence threshold for the optimiser.
    pub convergence_threshold: f64,
    /// Random seed for parameter initialisation.
    pub seed: u64,
}

impl Default for VqdConfig {
    fn default() -> Self {
        Self {
            num_states: 2,
            beta_penalties: vec![10.0],
            max_iterations: 500,
            convergence_threshold: 1e-7,
            seed: 42,
        }
    }
}

impl VqdConfig {
    pub fn num_states(mut self, n: usize) -> Self {
        self.num_states = n;
        self
    }

    pub fn beta_penalties(mut self, betas: Vec<f64>) -> Self {
        self.beta_penalties = betas;
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn convergence_threshold(mut self, tol: f64) -> Self {
        self.convergence_threshold = tol;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Get the penalty weight for the k-th overlap term.
    fn beta(&self, k: usize) -> f64 {
        if k < self.beta_penalties.len() {
            self.beta_penalties[k]
        } else {
            *self.beta_penalties.last().unwrap_or(&10.0)
        }
    }
}

/// Result of VQD computation.
#[derive(Debug, Clone)]
pub struct VqdResult {
    /// Energies of each found eigenstate (in order).
    pub energies: Vec<f64>,
    /// State vectors of each found eigenstate.
    pub states: Vec<Vec<C64>>,
    /// Optimiser energy history for each eigenstate.
    pub optimizer_history: Vec<Vec<f64>>,
    /// Energy gaps between consecutive eigenstates.
    pub energy_gaps: Vec<f64>,
}

/// Find ground and excited states using VQD.
///
/// For each eigenstate k = 0, 1, ..., num_states-1, minimises:
///   C_k(theta) = <psi(theta)|H|psi(theta)> + sum_{j<k} beta_j |<psi(theta)|psi_j>|^2
///
/// using Nelder-Mead optimisation.
pub fn vqd_excited_states(
    hamiltonian: &PauliHamiltonian,
    ansatz: &VqdAnsatz,
    config: &VqdConfig,
) -> VqdResult {
    let n = ansatz.num_qubits();
    let num_params = ansatz.num_params();

    let mut all_energies = Vec::new();
    let mut all_states: Vec<Vec<C64>> = Vec::new();
    let mut all_histories = Vec::new();

    // Simple PRNG for parameter initialisation (xorshift64)
    let mut rng_state = config.seed;
    let mut rand_f64 = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64) / (u64::MAX as f64) * 2.0 * PI - PI
    };

    for k in 0..config.num_states {
        let previous_states = all_states.clone();

        // Cost function for the k-th state
        let eval = |params: &[f64]| -> f64 {
            let mut state = DynamicsState::zero_state(n);
            ansatz.apply(&mut state, params);

            // Hamiltonian expectation value
            let energy = hamiltonian.expectation_value(&state.amplitudes);

            // Overlap penalties
            let mut penalty = 0.0;
            for (j, prev) in previous_states.iter().enumerate() {
                let prev_state = DynamicsState::from_amplitudes(prev.clone(), n);
                let overlap = state.overlap(&prev_state);
                penalty += config.beta(j) * overlap;
            }

            energy + penalty
        };

        // Initialise parameters with small random values
        let initial_params: Vec<f64> = (0..num_params).map(|_| rand_f64() * 0.1).collect();

        // Run Nelder-Mead optimisation
        let (best_params, history) =
            nelder_mead_optimize(&eval, &initial_params, config.max_iterations, config.convergence_threshold);

        // Evaluate the final state
        let mut final_state = DynamicsState::zero_state(n);
        ansatz.apply(&mut final_state, &best_params);
        let energy = hamiltonian.expectation_value(&final_state.amplitudes);

        all_energies.push(energy);
        all_states.push(final_state.amplitudes);
        all_histories.push(history);
    }

    // Compute energy gaps
    let mut energy_gaps = Vec::new();
    for i in 1..all_energies.len() {
        energy_gaps.push(all_energies[i] - all_energies[i - 1]);
    }

    VqdResult {
        energies: all_energies,
        states: all_states,
        optimizer_history: all_histories,
        energy_gaps,
    }
}

// ============================================================
// NELDER-MEAD OPTIMISER
// ============================================================

/// Nelder-Mead simplex optimiser for derivative-free minimisation.
///
/// Returns (best_parameters, energy_history).
fn nelder_mead_optimize(
    eval: &dyn Fn(&[f64]) -> f64,
    initial: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = initial.len();
    let alpha = 1.0;   // reflection
    let gamma = 2.0;   // expansion
    let rho = 0.5;     // contraction
    let sigma = 0.5;   // shrink

    // Build initial simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.to_vec());
    for i in 0..n {
        let mut vertex = initial.to_vec();
        let step = if vertex[i].abs() > 1e-8 {
            0.05 * vertex[i].abs()
        } else {
            0.00025
        };
        vertex[i] += step;
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| eval(v)).collect();
    let mut history = Vec::new();
    history.push(values.iter().cloned().fold(f64::INFINITY, f64::min));

    for _ in 0..max_iterations {
        // Sort vertices by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best_idx = order[0];
        let worst_idx = order[n];
        let second_worst_idx = order[n - 1];

        history.push(values[best_idx]);

        // Check convergence
        let val_range = (values[worst_idx] - values[best_idx]).abs();
        if val_range < tolerance {
            break;
        }

        // Centroid of all points except the worst
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for d in 0..n {
                centroid[d] += simplex[idx][d];
            }
        }
        for d in 0..n {
            centroid[d] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|d| centroid[d] + alpha * (centroid[d] - simplex[worst_idx][d]))
            .collect();
        let f_reflected = eval(&reflected);

        if f_reflected < values[second_worst_idx] && f_reflected >= values[best_idx] {
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
            continue;
        }

        // Expansion
        if f_reflected < values[best_idx] {
            let expanded: Vec<f64> = (0..n)
                .map(|d| centroid[d] + gamma * (reflected[d] - centroid[d]))
                .collect();
            let f_expanded = eval(&expanded);
            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_reflected;
            }
            continue;
        }

        // Contraction
        let contracted: Vec<f64> = (0..n)
            .map(|d| centroid[d] + rho * (simplex[worst_idx][d] - centroid[d]))
            .collect();
        let f_contracted = eval(&contracted);

        if f_contracted < values[worst_idx] {
            simplex[worst_idx] = contracted;
            values[worst_idx] = f_contracted;
            continue;
        }

        // Shrink
        let best = simplex[best_idx].clone();
        for idx in 0..=n {
            if idx == best_idx {
                continue;
            }
            for d in 0..n {
                simplex[idx][d] = best[d] + sigma * (simplex[idx][d] - best[d]);
            }
            values[idx] = eval(&simplex[idx]);
        }
    }

    // Find the best vertex
    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }

    (simplex[best_idx].clone(), history)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create the 2-qubit Heisenberg model H = XX + YY + ZZ
    fn heisenberg_2q() -> PauliHamiltonian {
        PauliHamiltonian::new(vec![
            (1.0, vec![(0, 'X'), (1, 'X')]),
            (1.0, vec![(0, 'Y'), (1, 'Y')]),
            (1.0, vec![(0, 'Z'), (1, 'Z')]),
        ])
    }

    // -----------------------------------------------------------
    // Hamiltonian tests
    // -----------------------------------------------------------

    #[test]
    fn test_pauli_hamiltonian_num_qubits() {
        let h = PauliHamiltonian::transverse_field_ising(3, 1.0, 0.5);
        assert_eq!(h.num_qubits(), 3);
    }

    #[test]
    fn test_pauli_hamiltonian_num_terms() {
        let h = PauliHamiltonian::transverse_field_ising(3, 1.0, 0.5);
        // 2 ZZ terms + 3 X terms = 5
        assert_eq!(h.num_terms(), 5);
    }

    #[test]
    fn test_transverse_field_ising_matrix_elements() {
        // 2-qubit TFIM: H = -J*Z0Z1 - h*(X0 + X1) with J=1, h=0.5
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
        let mat = h.to_matrix();

        // Matrix dimension
        assert_eq!(mat.len(), 16);

        // Diagonal elements: Z0Z1 eigenvalues are +1 for |00>,|11> and -1 for |01>,|10>
        // <00|H|00> = -J * (+1) = -1.0
        assert!((mat[0 * 4 + 0].re - (-1.0)).abs() < 1e-12);
        // <01|H|01> = -J * (-1) = +1.0
        assert!((mat[1 * 4 + 1].re - 1.0).abs() < 1e-12);
        // <10|H|10> = -J * (-1) = +1.0
        assert!((mat[2 * 4 + 2].re - 1.0).abs() < 1e-12);
        // <11|H|11> = -J * (+1) = -1.0
        assert!((mat[3 * 4 + 3].re - (-1.0)).abs() < 1e-12);

        // Off-diagonal: X contributions
        // <00|X0|10> = 1, <00|X1|01> = 1, each with coefficient -h = -0.5
        assert!((mat[0 * 4 + 2].re - (-0.5)).abs() < 1e-12); // X0: |00> <-> |10>
        assert!((mat[0 * 4 + 1].re - (-0.5)).abs() < 1e-12); // X1: |00> <-> |01>
    }

    #[test]
    fn test_hamiltonian_hermiticity() {
        let h = heisenberg_2q();
        let mat = h.to_matrix();
        let dim = 4;
        for i in 0..dim {
            for j in 0..dim {
                let diff = (mat[i * dim + j] - mat[j * dim + i].conj()).norm();
                assert!(diff < 1e-12, "Matrix not Hermitian at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_expectation_value_matches_matrix() {
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
        let state = DynamicsState::plus_state(2);
        let dim = 4;

        // Via expectation_value method
        let ev_fast = h.expectation_value(&state.amplitudes);

        // Via full matrix multiplication
        let mat = h.to_matrix();
        let mut h_psi = vec![c64_zero(); dim];
        for i in 0..dim {
            for j in 0..dim {
                h_psi[i] += mat[i * dim + j] * state.amplitudes[j];
            }
        }
        let ev_matrix: f64 = state
            .amplitudes
            .iter()
            .zip(h_psi.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum();

        assert!(
            (ev_fast - ev_matrix).abs() < 1e-12,
            "Expectation mismatch: fast={}, matrix={}",
            ev_fast,
            ev_matrix
        );
    }

    #[test]
    fn test_heisenberg_xxz_constructor() {
        let h = PauliHamiltonian::heisenberg_xxz(2, 1.0, 1.0, 0.0);
        // For 2-qubit chain: XX + YY + ZZ (3 coupling terms) + 2 Z_i field terms = 5
        assert_eq!(h.num_terms(), 5);
        assert_eq!(h.num_qubits(), 2);
    }

    #[test]
    fn test_hydrogen_chain_constructor() {
        let h = PauliHamiltonian::hydrogen_chain(2);
        assert_eq!(h.num_qubits(), 4);
        assert!(h.num_terms() > 0);
    }

    #[test]
    fn test_identity_term_constant() {
        // A Hamiltonian with just a constant offset
        let h = PauliHamiltonian::new(vec![(3.14, vec![])]);
        let state = DynamicsState::zero_state(1);
        let ev = h.expectation_value(&state.amplitudes);
        assert!((ev - 3.14).abs() < 1e-12);
    }

    // -----------------------------------------------------------
    // DynamicsState tests
    // -----------------------------------------------------------

    #[test]
    fn test_dynamics_state_zero() {
        let s = DynamicsState::zero_state(2);
        assert!((s.amplitudes[0].re - 1.0).abs() < 1e-12);
        for i in 1..4 {
            assert!(s.amplitudes[i].norm() < 1e-12);
        }
    }

    #[test]
    fn test_dynamics_state_plus() {
        let s = DynamicsState::plus_state(2);
        let expected = 0.25; // |1/sqrt(4)|^2 = 1/4 = 0.25
        for a in &s.amplitudes {
            assert!((a.norm_sqr() - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let s0 = DynamicsState::zero_state(1);
        let mut s1 = DynamicsState::zero_state(1);
        // Flip to |1>
        s1.amplitudes[0] = c64_zero();
        s1.amplitudes[1] = c64_one();

        assert!(s0.inner_product(&s1).norm() < 1e-12);
    }

    #[test]
    fn test_inner_product_self() {
        let s = DynamicsState::plus_state(2);
        let ip = s.inner_product(&s);
        assert!((ip.re - 1.0).abs() < 1e-12);
        assert!(ip.im.abs() < 1e-12);
    }

    #[test]
    fn test_hadamard_gate() {
        let mut s = DynamicsState::zero_state(1);
        s.hadamard(0);
        // Should be |+> = (|0> + |1>) / sqrt(2)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s.amplitudes[0].re - inv_sqrt2).abs() < 1e-12);
        assert!((s.amplitudes[1].re - inv_sqrt2).abs() < 1e-12);
    }

    #[test]
    fn test_cnot_gate() {
        // |10> -> CNOT -> |11>
        let mut s = DynamicsState::zero_state(2);
        s.amplitudes[0] = c64_zero();
        s.amplitudes[2] = c64_one(); // |10>
        s.cnot(0, 1);
        assert!((s.amplitudes[3].norm_sqr() - 1.0).abs() < 1e-12); // |11>
    }

    // -----------------------------------------------------------
    // Matrix exponential tests
    // -----------------------------------------------------------

    #[test]
    fn test_matrix_exp_zero() {
        // exp(0) = I
        let zero_mat = vec![c64_zero(); 4];
        let result = matrix_exponential(&zero_mat, 2);
        assert!((result[0].re - 1.0).abs() < 1e-10);
        assert!(result[1].norm() < 1e-10);
        assert!(result[2].norm() < 1e-10);
        assert!((result[3].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_exp_identity() {
        // exp(I) = e * I
        let id = complex_identity(2);
        let result = matrix_exponential(&id, 2);
        let e = std::f64::consts::E;
        assert!((result[0].re - e).abs() < 1e-8);
        assert!(result[1].norm() < 1e-10);
        assert!(result[2].norm() < 1e-10);
        assert!((result[3].re - e).abs() < 1e-8);
    }

    #[test]
    fn test_matrix_exp_pauli_x_rotation() {
        // exp(-i * theta * X) should give:
        // [[cos(theta), -i*sin(theta)], [-i*sin(theta), cos(theta)]]
        let theta = 0.3;
        let mat = vec![
            c64(0.0, -theta),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, theta),
        ];
        // Wait, -i*theta*X = [[ 0, -i*theta], [-i*theta, 0]]
        let pauli_x_mat = vec![
            c64_zero(),
            c64(0.0, -theta),
            c64(0.0, -theta),
            c64_zero(),
        ];
        let result = matrix_exponential(&pauli_x_mat, 2);

        assert!((result[0].re - theta.cos()).abs() < 1e-8);
        assert!((result[1].im - (-theta.sin())).abs() < 1e-8);
        assert!((result[2].im - (-theta.sin())).abs() < 1e-8);
        assert!((result[3].re - theta.cos()).abs() < 1e-8);
    }

    #[test]
    fn test_matrix_exp_pauli_z_rotation() {
        // exp(-i * theta * Z) = diag(exp(-i*theta), exp(i*theta))
        let theta = 0.7;
        let mat = vec![
            c64(0.0, -theta),
            c64_zero(),
            c64_zero(),
            c64(0.0, theta),
        ];
        let result = matrix_exponential(&mat, 2);

        assert!((result[0].re - theta.cos()).abs() < 1e-8);
        assert!((result[0].im - (-theta.sin())).abs() < 1e-8);
        assert!(result[1].norm() < 1e-10);
        assert!(result[2].norm() < 1e-10);
        assert!((result[3].re - theta.cos()).abs() < 1e-8);
        assert!((result[3].im - theta.sin()).abs() < 1e-8);
    }

    // -----------------------------------------------------------
    // LU solve tests
    // -----------------------------------------------------------

    #[test]
    fn test_lu_solve_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 5.0];
        let x = lu_solve(&a, &b, 2).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_lu_solve_2x2() {
        // 2x + 3y = 8, 5x + 7y = 19  =>  x=1, y=2
        let a = vec![2.0, 3.0, 5.0, 7.0];
        let b = vec![8.0, 19.0];
        let x = lu_solve(&a, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------
    // Exact diagonalisation tests
    // -----------------------------------------------------------

    #[test]
    fn test_exact_ground_state_heisenberg() {
        let h = heisenberg_2q();
        let (energy, _state) = exact_ground_state(&h);
        // The 2-qubit isotropic Heisenberg model H = XX + YY + ZZ has
        // eigenvalues: -3 (singlet) and +1 (triplet, 3-fold degenerate)
        assert!(
            (energy - (-3.0)).abs() < 1e-8,
            "Ground state energy {} != -3.0",
            energy
        );
    }

    #[test]
    fn test_exact_eigenvalues_heisenberg() {
        let h = heisenberg_2q();
        let vals = exact_eigenvalues(&h, 4);
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - (-3.0)).abs() < 1e-8);
        // Three degenerate triplet states at +1
        for i in 1..4 {
            assert!((vals[i] - 1.0).abs() < 1e-8, "Eigenvalue {} = {}", i, vals[i]);
        }
    }

    #[test]
    fn test_exact_eigenvalues_tfim() {
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 1.0);
        let vals = exact_eigenvalues(&h, 4);
        // Eigenvalues should be sorted ascending
        for i in 1..vals.len() {
            assert!(
                vals[i] >= vals[i - 1] - 1e-10,
                "Eigenvalues not sorted: {} > {}",
                vals[i - 1],
                vals[i]
            );
        }
    }

    // -----------------------------------------------------------
    // QITE tests
    // -----------------------------------------------------------

    #[test]
    fn test_qite_converges_to_ground_state_heisenberg() {
        let h = heisenberg_2q();
        // The singlet ground state (|01> - |10>)/sqrt(2) has zero overlap with
        // the symmetric |++> state, so we use an initial state that has nonzero
        // singlet component: (|01> + 0.5|00> + 0.3|10>)/norm
        let mut init = vec![c64(0.5, 0.0), c64(1.0, 0.0), c64(0.3, 0.0), c64(0.1, 0.0)];
        let norm: f64 = init.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for a in &mut init {
            *a /= c64(norm, 0.0);
        }
        let config = QiteConfig::default()
            .time_step(0.05)
            .num_steps(300)
            .initial_state(init);
        let result = qite_ground_state(&h, &config);
        let exact_e = -3.0;
        assert!(
            (result.final_energy - exact_e).abs() < 0.05,
            "QITE energy {} far from exact {}",
            result.final_energy,
            exact_e
        );
    }

    #[test]
    fn test_qite_energy_decreases_monotonically() {
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
        let config = QiteConfig::default().time_step(0.02).num_steps(100);
        let result = qite_ground_state(&h, &config);

        // Energy should generally decrease (allowing small numerical fluctuations)
        for i in 1..result.energy_history.len() {
            let increase = result.energy_history[i] - result.energy_history[i - 1];
            assert!(
                increase < 1e-6,
                "Energy increased at step {}: {} -> {}",
                i,
                result.energy_history[i - 1],
                result.energy_history[i]
            );
        }
    }

    #[test]
    fn test_qite_matches_exact_diag_tfim() {
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
        let config = QiteConfig::default()
            .time_step(0.05)
            .num_steps(400);
        let result = qite_ground_state(&h, &config);
        let (exact_e, _) = exact_ground_state(&h);

        assert!(
            (result.final_energy - exact_e).abs() < 0.05,
            "QITE energy {} vs exact {}",
            result.final_energy,
            exact_e
        );
    }

    #[test]
    fn test_qite_convergence_flag() {
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
        let config = QiteConfig::default()
            .time_step(0.05)
            .num_steps(500)
            .convergence_threshold(1e-6);
        let result = qite_ground_state(&h, &config);
        // With enough steps, should converge
        assert!(result.converged, "QITE did not converge");
    }

    #[test]
    fn test_qite_thermal_state_boltzmann() {
        // For a simple Z Hamiltonian, the thermal state should follow Boltzmann.
        // H = Z_0 with eigenvalues +1 (|0>) and -1 (|1>)
        let h = PauliHamiltonian::new(vec![(1.0, vec![(0, 'Z')])]);
        let beta = 2.0;
        let config = QiteConfig::default();
        let populations = qite_thermal_state(&h, beta, &config);

        // p(|0>) = exp(-beta * 1) / Z, p(|1>) = exp(-beta * (-1)) / Z
        // Z = exp(-2) + exp(2)
        let z = (-beta).exp() + beta.exp();
        let expected_0 = (-beta).exp() / z; // ground state |1> has eigenvalue -1
        let expected_1 = beta.exp() / z;

        // |0> has eigenvalue +1, |1> has eigenvalue -1
        // So populations[0] = exp(-beta*1)/Z, populations[1] = exp(-beta*(-1))/Z
        assert!(
            (populations[0] - expected_0).abs() < 1e-6,
            "p(0)={} vs expected {}",
            populations[0],
            expected_0
        );
        assert!(
            (populations[1] - expected_1).abs() < 1e-6,
            "p(1)={} vs expected {}",
            populations[1],
            expected_1
        );
    }

    #[test]
    fn test_qite_thermal_state_high_temperature() {
        // At very high temperature (beta -> 0), all states equally populated
        let h = heisenberg_2q();
        let populations = qite_thermal_state(&h, 0.001, &QiteConfig::default());
        let expected = 0.25; // 1/4 for 4 states
        for (i, &p) in populations.iter().enumerate() {
            assert!(
                (p - expected).abs() < 0.01,
                "Population {} = {} (expected ~{})",
                i,
                p,
                expected
            );
        }
    }

    #[test]
    fn test_qite_thermal_state_low_temperature() {
        // At very low temperature, only ground state populated.
        // Use a non-degenerate Hamiltonian so the ground state is a single
        // computational basis state: H = Z_0 has ground state |1> with E=-1.
        let h = PauliHamiltonian::new(vec![(1.0, vec![(0, 'Z')])]);
        let populations = qite_thermal_state(&h, 50.0, &QiteConfig::default());
        // At beta=50: p(|0>) = exp(-50)/Z ~ 0, p(|1>) = exp(50)/Z ~ 1
        assert!(
            populations[1] > 0.99,
            "Ground state population {} too low at low temperature",
            populations[1]
        );
    }

    #[test]
    fn test_qite_initial_state_option() {
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
        // Start from |00> instead of |++>
        let init = vec![c64_one(), c64_zero(), c64_zero(), c64_zero()];
        let config = QiteConfig::default()
            .initial_state(init)
            .time_step(0.05)
            .num_steps(300);
        let result = qite_ground_state(&h, &config);
        let (exact_e, _) = exact_ground_state(&h);
        assert!(
            (result.final_energy - exact_e).abs() < 0.1,
            "QITE from |00> energy {} vs exact {}",
            result.final_energy,
            exact_e
        );
    }

    // -----------------------------------------------------------
    // VQD tests
    // -----------------------------------------------------------

    #[test]
    fn test_vqd_ground_state() {
        let h = PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5);
        let ansatz = VqdAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 4,
        };
        let config = VqdConfig::default()
            .num_states(1)
            .max_iterations(800)
            .convergence_threshold(1e-8);
        let result = vqd_excited_states(&h, &ansatz, &config);

        let (exact_e, _) = exact_ground_state(&h);
        assert!(
            (result.energies[0] - exact_e).abs() < 0.3,
            "VQD ground energy {} vs exact {}",
            result.energies[0],
            exact_e
        );
    }

    #[test]
    fn test_vqd_finds_two_states() {
        let h = PauliHamiltonian::new(vec![
            (1.0, vec![(0, 'Z')]),
        ]);
        let ansatz = VqdAnsatz::HardwareEfficient {
            num_qubits: 1,
            depth: 2,
        };
        let config = VqdConfig::default()
            .num_states(2)
            .beta_penalties(vec![20.0])
            .max_iterations(600)
            .convergence_threshold(1e-8)
            .seed(123);
        let result = vqd_excited_states(&h, &ansatz, &config);

        assert_eq!(result.energies.len(), 2);
        // Eigenvalues of Z are -1 and +1
        assert!(
            (result.energies[0] - (-1.0)).abs() < 0.3,
            "Ground energy {} vs -1.0",
            result.energies[0]
        );
        assert!(
            (result.energies[1] - 1.0).abs() < 0.3,
            "Excited energy {} vs 1.0",
            result.energies[1]
        );
    }

    #[test]
    fn test_vqd_excited_state_orthogonal() {
        let h = PauliHamiltonian::new(vec![
            (1.0, vec![(0, 'Z')]),
        ]);
        let ansatz = VqdAnsatz::HardwareEfficient {
            num_qubits: 1,
            depth: 2,
        };
        let config = VqdConfig::default()
            .num_states(2)
            .beta_penalties(vec![20.0])
            .max_iterations(600)
            .seed(123);
        let result = vqd_excited_states(&h, &ansatz, &config);

        let s0 = DynamicsState::from_amplitudes(result.states[0].clone(), 1);
        let s1 = DynamicsState::from_amplitudes(result.states[1].clone(), 1);
        let overlap = s0.overlap(&s1);

        assert!(
            overlap < 0.15,
            "States not orthogonal: overlap = {}",
            overlap
        );
    }

    #[test]
    fn test_vqd_energy_ordering() {
        let h = PauliHamiltonian::new(vec![
            (1.0, vec![(0, 'Z')]),
        ]);
        let ansatz = VqdAnsatz::HardwareEfficient {
            num_qubits: 1,
            depth: 2,
        };
        let config = VqdConfig::default()
            .num_states(2)
            .beta_penalties(vec![20.0])
            .max_iterations(600)
            .seed(123);
        let result = vqd_excited_states(&h, &ansatz, &config);

        // First energy should be less than or close to second
        assert!(
            result.energies[0] <= result.energies[1] + 0.1,
            "Energy ordering violated: {} > {}",
            result.energies[0],
            result.energies[1]
        );
    }

    #[test]
    fn test_vqd_energy_gaps() {
        let h = PauliHamiltonian::new(vec![
            (1.0, vec![(0, 'Z')]),
        ]);
        let ansatz = VqdAnsatz::HardwareEfficient {
            num_qubits: 1,
            depth: 2,
        };
        let config = VqdConfig::default()
            .num_states(2)
            .beta_penalties(vec![20.0])
            .max_iterations(600)
            .seed(123);
        let result = vqd_excited_states(&h, &ansatz, &config);

        assert_eq!(result.energy_gaps.len(), 1);
        // Gap should be close to 2.0 (eigenvalues -1 and +1)
        assert!(
            (result.energy_gaps[0] - 2.0).abs() < 0.6,
            "Energy gap {} vs expected 2.0",
            result.energy_gaps[0]
        );
    }

    #[test]
    fn test_vqd_matches_exact_eigenvalues() {
        // Simple 1-qubit Z Hamiltonian
        let h = PauliHamiltonian::new(vec![(1.0, vec![(0, 'Z')])]);
        let exact = exact_eigenvalues(&h, 2);

        let ansatz = VqdAnsatz::HardwareEfficient {
            num_qubits: 1,
            depth: 2,
        };
        let config = VqdConfig::default()
            .num_states(2)
            .beta_penalties(vec![20.0])
            .max_iterations(600)
            .seed(123);
        let result = vqd_excited_states(&h, &ansatz, &config);

        for (i, (&vqd_e, &exact_e)) in result.energies.iter().zip(exact.iter()).enumerate() {
            assert!(
                (vqd_e - exact_e).abs() < 0.3,
                "State {}: VQD {} vs exact {}",
                i,
                vqd_e,
                exact_e
            );
        }
    }

    #[test]
    fn test_vqd_uccsd_ansatz() {
        let h = PauliHamiltonian::new(vec![(1.0, vec![(0, 'Z')])]);
        let ansatz = VqdAnsatz::UCCSD {
            num_qubits: 2,
            num_electrons: 1,
        };
        let config = VqdConfig::default()
            .num_states(1)
            .max_iterations(400);
        let result = vqd_excited_states(&h, &ansatz, &config);
        // Should find something reasonable
        assert_eq!(result.energies.len(), 1);
    }

    #[test]
    fn test_vqd_ansatz_num_params() {
        let hw = VqdAnsatz::HardwareEfficient {
            num_qubits: 3,
            depth: 2,
        };
        // (depth+1) * 2 * num_qubits = 3 * 2 * 3 = 18
        assert_eq!(hw.num_params(), 18);

        let uccsd = VqdAnsatz::UCCSD {
            num_qubits: 4,
            num_electrons: 2,
        };
        assert!(uccsd.num_params() > 0);
    }

    // -----------------------------------------------------------
    // Integration / cross-validation tests
    // -----------------------------------------------------------

    #[test]
    fn test_qite_and_exact_agree_3_qubit() {
        let h = PauliHamiltonian::transverse_field_ising(3, 1.0, 0.5);
        let config = QiteConfig::default()
            .time_step(0.03)
            .num_steps(500);
        let result = qite_ground_state(&h, &config);
        let (exact_e, _) = exact_ground_state(&h);

        assert!(
            (result.final_energy - exact_e).abs() < 0.1,
            "3-qubit QITE energy {} vs exact {}",
            result.final_energy,
            exact_e
        );
    }

    #[test]
    fn test_qite_state_normalised() {
        let h = heisenberg_2q();
        let config = QiteConfig::default().num_steps(50);
        let result = qite_ground_state(&h, &config);
        let norm_sq: f64 = result.final_state.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "Final state not normalised: norm^2 = {}",
            norm_sq
        );
    }

    #[test]
    fn test_pauli_rotation_consistency() {
        // exp(-i * pi/2 * X)|0> should give -i|1>
        let mut s = DynamicsState::zero_state(1);
        s.apply_pauli_rotation(0, 'X', PI / 2.0);
        // Should be cos(pi/2)|0> - i*sin(pi/2)|1> = -i|1>
        assert!(s.amplitudes[0].norm() < 1e-10);
        assert!((s.amplitudes[1].im - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_exact_eigenpairs_orthogonal() {
        let h = heisenberg_2q();
        let (vals, vecs) = exact_eigenpairs(&h, 4);
        assert_eq!(vals.len(), 4);

        // Check orthogonality of first two eigenvectors
        let s0 = DynamicsState::from_amplitudes(vecs[0].clone(), 2);
        let s1 = DynamicsState::from_amplitudes(vecs[1].clone(), 2);
        let overlap = s0.overlap(&s1);
        assert!(
            overlap < 1e-6,
            "Eigenvectors not orthogonal: overlap = {}",
            overlap
        );
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        // Minimise a simple quadratic f(x) = (x-1)^2 + (y-2)^2
        let eval = |params: &[f64]| -> f64 {
            (params[0] - 1.0).powi(2) + (params[1] - 2.0).powi(2)
        };
        let (best, _history) = nelder_mead_optimize(&eval, &[0.0, 0.0], 1000, 1e-12);
        assert!((best[0] - 1.0).abs() < 1e-4);
        assert!((best[1] - 2.0).abs() < 1e-4);
    }
}
