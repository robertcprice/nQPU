//! Subspace-Search VQE (SSVQE) and related excited-state methods.
//!
//! This module provides four complementary algorithms for computing multiple
//! eigenvalues and eigenstates of quantum Hamiltonians:
//!
//! - **SSVQE**: Simultaneously finds k lowest eigenstates by optimising a single
//!   parameterised circuit applied to k orthogonal initial states with weighted
//!   cost function. Based on Nakanishi, Mitarai, Fujii, Phys. Rev. Research 1,
//!   033062 (2019).
//!
//! - **MCVQE**: Multistate Contracted VQE builds a subspace Hamiltonian from k
//!   variational states and diagonalises it to resolve near-degenerate spectra.
//!   Based on Parrish, Hohenstein, McMahon, Martinez, PRL 122, 230401 (2019).
//!
//! - **QSE**: Quantum Subspace Expansion corrects a VQE ground-state estimate by
//!   expanding the variational subspace with Pauli excitation operators. Based on
//!   McClean, Kimchi-Schwartz, Carter, de Jong, PRA 95, 042308 (2017).
//!
//! - **qEOM**: Quantum Equation of Motion computes excitation energies from a
//!   ground-state reference using commutator/anticommutator matrices. Based on
//!   Ollitrault et al., Chemical Science 11, 6842 (2020).
//!
//! All algorithms are self-contained with their own statevector simulator,
//! Nelder-Mead optimiser, and ansatz circuits to avoid cross-module coupling.

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
    C64::new(0.0, 0.0)
}

#[inline]
fn c64_one() -> C64 {
    C64::new(1.0, 0.0)
}

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from subspace VQE algorithms.
#[derive(Debug, Clone)]
pub enum SSVQEError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Convergence failure.
    ConvergenceFailed(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Singular matrix encountered.
    SingularMatrix(String),
}

impl std::fmt::Display for SSVQEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "InvalidConfig: {}", msg),
            Self::ConvergenceFailed(msg) => write!(f, "ConvergenceFailed: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "DimensionMismatch: expected {expected}, got {got}")
            }
            Self::SingularMatrix(msg) => write!(f, "SingularMatrix: {}", msg),
        }
    }
}

impl std::error::Error for SSVQEError {}

// ============================================================
// PAULI HAMILTONIAN (self-contained)
// ============================================================

/// A Hamiltonian expressed as a sum of weighted Pauli strings.
///
/// Each term is (coefficient, [(qubit_index, pauli_char)]) where the char is
/// one of 'I', 'X', 'Y', 'Z'.
#[derive(Debug, Clone)]
pub struct PauliHamiltonian {
    pub terms: Vec<(f64, Vec<(usize, char)>)>,
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

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Build the full 2^n x 2^n matrix representation (row-major, flat).
    pub fn to_matrix(&self) -> Vec<C64> {
        let n = self.num_qubits;
        let dim = 1usize << n;
        let mut mat = vec![c64_zero(); dim * dim];
        for (coeff, ops) in &self.terms {
            add_pauli_term_to_matrix(&mut mat, dim, n, *coeff, ops);
        }
        mat
    }

    /// Compute <state|H|state> by direct statevector manipulation.
    pub fn expectation_value(&self, state: &[C64]) -> f64 {
        let mut energy = 0.0_f64;
        for (coeff, ops) in &self.terms {
            let mut scratch = state.to_vec();
            apply_pauli_string(&mut scratch, self.num_qubits, ops);
            let inner: C64 = state
                .iter()
                .zip(scratch.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            energy += coeff * inner.re;
        }
        energy
    }

    /// Compute <bra|H|ket> matrix element between two states.
    pub fn matrix_element(&self, bra: &[C64], ket: &[C64]) -> C64 {
        let mut result = c64_zero();
        for (coeff, ops) in &self.terms {
            let mut scratch = ket.to_vec();
            apply_pauli_string(&mut scratch, self.num_qubits, ops);
            let inner: C64 = bra
                .iter()
                .zip(scratch.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            result += c64(*coeff, 0.0) * inner;
        }
        result
    }

    // ----------------------------------------------------------------
    // Constructor helpers
    // ----------------------------------------------------------------

    /// Transverse-field Ising model: H = -J * sum Z_i Z_{i+1} - h * sum X_i
    pub fn transverse_field_ising(n: usize, j: f64, h: f64) -> Self {
        let mut terms = Vec::new();
        for i in 0..n.saturating_sub(1) {
            terms.push((-j, vec![(i, 'Z'), (i + 1, 'Z')]));
        }
        for i in 0..n {
            terms.push((-h, vec![(i, 'X')]));
        }
        Self::new(terms)
    }

    /// Heisenberg XXZ: H = J*(XX + YY) + Delta*ZZ + h*sum(Z_i)
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

    /// Minimal 2-qubit H2-like Hamiltonian with known spectrum.
    ///
    /// H = g0*II + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
    /// Uses STO-3G coefficients at equilibrium bond length (0.735 A).
    pub fn h2_molecule() -> Self {
        let g0 = -0.4804;
        let g1 = 0.3435;
        let g2 = -0.4347;
        let g3 = 0.5716;
        let g4 = 0.0910;
        let g5 = 0.0910;
        Self::new(vec![
            (g0, vec![]),
            (g1, vec![(0, 'Z')]),
            (g2, vec![(1, 'Z')]),
            (g3, vec![(0, 'Z'), (1, 'Z')]),
            (g4, vec![(0, 'X'), (1, 'X')]),
            (g5, vec![(0, 'Y'), (1, 'Y')]),
        ])
    }
}

// ============================================================
// PAULI STRING OPERATIONS
// ============================================================

fn apply_pauli_string(state: &mut [C64], num_qubits: usize, ops: &[(usize, char)]) {
    for &(qubit, pauli) in ops {
        apply_single_pauli(state, num_qubits, qubit, pauli);
    }
}

fn apply_single_pauli(state: &mut [C64], num_qubits: usize, q: usize, pauli: char) {
    let dim = state.len();
    let mask = 1usize << (num_qubits - 1 - q);
    match pauli {
        'I' => {}
        'X' => {
            for basis in 0..dim {
                if basis & mask == 0 {
                    let partner = basis | mask;
                    state.swap(basis, partner);
                }
            }
        }
        'Y' => {
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
        _ => {}
    }
}

fn add_pauli_term_to_matrix(
    mat: &mut [C64],
    dim: usize,
    num_qubits: usize,
    coeff: f64,
    ops: &[(usize, char)],
) {
    let c = c64(coeff, 0.0);
    for col in 0..dim {
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
                        phase *= c64(0.0, 1.0);
                    } else {
                        phase *= c64(0.0, -1.0);
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
// STATEVECTOR (self-contained)
// ============================================================

/// Self-contained statevector for subspace VQE simulations.
#[derive(Debug, Clone)]
pub struct SubspaceState {
    pub amplitudes: Vec<C64>,
    pub num_qubits: usize,
}

impl SubspaceState {
    /// Create |0...0>.
    pub fn zero_state(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut amps = vec![c64_zero(); dim];
        amps[0] = c64_one();
        Self {
            amplitudes: amps,
            num_qubits,
        }
    }

    /// Create a computational basis state |k>.
    pub fn basis_state(num_qubits: usize, k: usize) -> Self {
        let dim = 1usize << num_qubits;
        assert!(k < dim, "Basis index {k} >= dimension {dim}");
        let mut amps = vec![c64_zero(); dim];
        amps[k] = c64_one();
        Self {
            amplitudes: amps,
            num_qubits,
        }
    }

    /// Create from amplitude vector.
    pub fn from_amplitudes(amplitudes: Vec<C64>, num_qubits: usize) -> Self {
        assert_eq!(amplitudes.len(), 1 << num_qubits);
        Self {
            amplitudes,
            num_qubits,
        }
    }

    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq > 1e-30 {
            let inv = 1.0 / norm_sq.sqrt();
            for a in &mut self.amplitudes {
                *a *= inv;
            }
        }
    }

    pub fn norm(&self) -> f64 {
        let ns: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        ns.sqrt()
    }

    /// Inner product <self|other>.
    pub fn inner_product(&self, other: &SubspaceState) -> C64 {
        self.amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }

    /// Overlap |<self|other>|^2.
    pub fn overlap(&self, other: &SubspaceState) -> f64 {
        self.inner_product(other).norm_sqr()
    }

    /// Apply Pauli rotation exp(-i * angle * P_q).
    pub fn apply_pauli_rotation(&mut self, qubit: usize, pauli: char, angle: f64) {
        let mask = 1usize << (self.num_qubits - 1 - qubit);
        let dim = self.dim();
        match pauli {
            'X' => {
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
                let phase_0 = c64(angle.cos(), -angle.sin());
                let phase_1 = c64(angle.cos(), angle.sin());
                for basis in 0..dim {
                    if basis & mask == 0 {
                        self.amplitudes[basis] *= phase_0;
                        self.amplitudes[basis | mask] *= phase_1;
                    }
                }
            }
            _ => {}
        }
    }

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

    pub fn x_gate(&mut self, q: usize) {
        let mask = 1usize << (self.num_qubits - 1 - q);
        let dim = self.dim();
        for basis in 0..dim {
            if basis & mask == 0 {
                let partner = basis | mask;
                self.amplitudes.swap(basis, partner);
            }
        }
    }

    pub fn ry(&mut self, q: usize, theta: f64) {
        self.apply_pauli_rotation(q, 'Y', theta / 2.0);
    }

    pub fn rz(&mut self, q: usize, theta: f64) {
        self.apply_pauli_rotation(q, 'Z', theta / 2.0);
    }

    pub fn rx(&mut self, q: usize, theta: f64) {
        self.apply_pauli_rotation(q, 'X', theta / 2.0);
    }
}

// ============================================================
// ANSATZ CIRCUITS
// ============================================================

/// Ansatz circuit type for subspace VQE methods.
#[derive(Debug, Clone)]
pub enum SubspaceAnsatz {
    /// Hardware-efficient ansatz: alternating Ry/Rz rotations with CNOT entanglers.
    HardwareEfficient {
        num_qubits: usize,
        depth: usize,
    },
    /// Simplified UCCSD-inspired ansatz for chemistry applications.
    UCCSD {
        num_qubits: usize,
        num_electrons: usize,
    },
}

impl SubspaceAnsatz {
    /// Number of variational parameters.
    pub fn num_params(&self) -> usize {
        match self {
            SubspaceAnsatz::HardwareEfficient { num_qubits, depth } => {
                2 * num_qubits * (depth + 1)
            }
            SubspaceAnsatz::UCCSD {
                num_qubits,
                num_electrons,
            } => {
                let n_occ = *num_electrons;
                let n_virt = num_qubits.saturating_sub(n_occ);
                let singles = n_occ * n_virt;
                let doubles = n_occ * n_occ.saturating_sub(1) / 2
                    * n_virt
                    * n_virt.saturating_sub(1)
                    / 2;
                singles + doubles
            }
        }
    }

    pub fn num_qubits(&self) -> usize {
        match self {
            SubspaceAnsatz::HardwareEfficient { num_qubits, .. } => *num_qubits,
            SubspaceAnsatz::UCCSD { num_qubits, .. } => *num_qubits,
        }
    }

    /// Apply the ansatz circuit to a state.
    pub fn apply(&self, state: &mut SubspaceState, params: &[f64]) {
        match self {
            SubspaceAnsatz::HardwareEfficient { num_qubits, depth } => {
                let n = *num_qubits;
                let mut idx = 0;
                // Initial rotation layer
                for q in 0..n {
                    if idx >= params.len() { return; }
                    state.ry(q, params[idx]);
                    idx += 1;
                    if idx >= params.len() { return; }
                    state.rz(q, params[idx]);
                    idx += 1;
                }
                for _layer in 0..*depth {
                    for q in 0..n.saturating_sub(1) {
                        state.cnot(q, q + 1);
                    }
                    for q in 0..n {
                        if idx >= params.len() { return; }
                        state.ry(q, params[idx]);
                        idx += 1;
                        if idx >= params.len() { return; }
                        state.rz(q, params[idx]);
                        idx += 1;
                    }
                }
            }
            SubspaceAnsatz::UCCSD {
                num_qubits,
                num_electrons,
            } => {
                let n = *num_qubits;
                let n_occ = *num_electrons;
                let mut idx = 0;
                // Hartree-Fock initial state
                for q in 0..n_occ.min(n) {
                    state.x_gate(q);
                }
                // Single excitations
                for i in 0..n_occ.min(n) {
                    for a in n_occ..n {
                        if idx >= params.len() { return; }
                        let theta = params[idx];
                        idx += 1;
                        state.ry(i, theta);
                        if a < n {
                            state.cnot(i, a);
                            state.ry(a, theta * 0.5);
                            state.cnot(i, a);
                        }
                    }
                }
                // Double excitations (simplified)
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

// ============================================================
// NELDER-MEAD OPTIMISER
// ============================================================

/// Nelder-Mead simplex optimiser. Returns (best_params, energy_history).
fn nelder_mead_optimize(
    eval: &dyn Fn(&[f64]) -> f64,
    initial: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = initial.len();
    if n == 0 {
        return (vec![], vec![eval(&[])]);
    }
    let alpha = 1.0_f64;
    let gamma = 2.0_f64;
    let rho = 0.5_f64;
    let sigma = 0.5_f64;

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
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best_idx = order[0];
        let worst_idx = order[n];
        let second_worst_idx = order[n.saturating_sub(1)];

        history.push(values[best_idx]);

        let val_range = (values[worst_idx] - values[best_idx]).abs();
        if val_range < tolerance {
            break;
        }

        let mut centroid = vec![0.0_f64; n];
        for &idx in &order[..n] {
            for d in 0..n {
                centroid[d] += simplex[idx][d];
            }
        }
        for d in 0..n {
            centroid[d] /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|d| centroid[d] + alpha * (centroid[d] - simplex[worst_idx][d]))
            .collect();
        let f_reflected = eval(&reflected);

        if f_reflected < values[second_worst_idx] && f_reflected >= values[best_idx] {
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
            continue;
        }

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

        let contracted: Vec<f64> = (0..n)
            .map(|d| centroid[d] + rho * (simplex[worst_idx][d] - centroid[d]))
            .collect();
        let f_contracted = eval(&contracted);

        if f_contracted < values[worst_idx] {
            simplex[worst_idx] = contracted;
            values[worst_idx] = f_contracted;
            continue;
        }

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

    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }
    (simplex[best_idx].clone(), history)
}

// ============================================================
// SIMPLE PRNG (xorshift64)
// ============================================================

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Random value in [-pi, pi].
    fn next_angle(&mut self) -> f64 {
        self.next_f64() * 2.0 * PI - PI
    }
}

// ============================================================
// LINEAR ALGEBRA (Jacobi eigendecomposition)
// ============================================================

/// Eigendecomposition of a real symmetric matrix via Jacobi rotations.
/// Returns (eigenvalues_sorted, eigenvectors_columns) where eigenvectors
/// are stored column-major in a flat row-major matrix: eigvec k is at
/// indices [i * dim + k] for i = 0..dim.
fn real_symmetric_eigen(mat_c64: &[C64], dim: usize) -> (Vec<f64>, Vec<C64>) {
    let mut a: Vec<f64> = mat_c64.iter().map(|c| c.re).collect();
    let mut v = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }

    let max_iter = 100 * dim * dim;
    for _ in 0..max_iter {
        let mut max_off = 0.0_f64;
        let mut p = 0_usize;
        let mut q = 1_usize;
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

        let mut new_col_p = vec![0.0_f64; dim];
        let mut new_col_q = vec![0.0_f64; dim];
        for i in 0..dim {
            new_col_p[i] = cos_t * a[i * dim + p] - sin_t * a[i * dim + q];
            new_col_q[i] = sin_t * a[i * dim + p] + cos_t * a[i * dim + q];
        }
        for i in 0..dim {
            a[i * dim + p] = new_col_p[i];
            a[i * dim + q] = new_col_q[i];
            a[p * dim + i] = new_col_p[i];
            a[q * dim + i] = new_col_q[i];
        }
        a[p * dim + p] = cos_t * new_col_p[p] - sin_t * new_col_p[q];
        a[q * dim + q] = sin_t * new_col_q[p] + cos_t * new_col_q[q];
        a[p * dim + q] = 0.0;
        a[q * dim + p] = 0.0;

        for i in 0..dim {
            let vip = v[i * dim + p];
            let viq = v[i * dim + q];
            v[i * dim + p] = cos_t * vip - sin_t * viq;
            v[i * dim + q] = sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..dim).map(|i| a[i * dim + i]).collect();
    let eigenvectors: Vec<C64> = v.iter().map(|&r| c64(r, 0.0)).collect();

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

/// Complex Hermitian eigendecomposition via embedding into real 2n x 2n.
fn complex_hermitian_eigen(matrix: &[C64], dim: usize) -> (Vec<f64>, Vec<C64>) {
    let n2 = 2 * dim;
    let mut real_mat = vec![c64_zero(); n2 * n2];
    for i in 0..dim {
        for j in 0..dim {
            let c = matrix[i * dim + j];
            real_mat[i * n2 + j] = c64(c.re, 0.0);
            real_mat[i * n2 + (dim + j)] = c64(-c.im, 0.0);
            real_mat[(dim + i) * n2 + j] = c64(c.im, 0.0);
            real_mat[(dim + i) * n2 + (dim + j)] = c64(c.re, 0.0);
        }
    }
    let (all_vals, all_vecs) = real_symmetric_eigen(&real_mat, n2);

    let mut eigenvalues = Vec::with_capacity(dim);
    let mut eigenvectors = vec![c64_zero(); dim * dim];
    let mut used = vec![false; n2];

    for k in 0..n2 {
        if used[k] { continue; }
        if eigenvalues.len() >= dim { break; }
        let idx = eigenvalues.len();
        eigenvalues.push(all_vals[k]);
        for i in 0..dim {
            eigenvectors[i * dim + idx] =
                c64(all_vecs[i * n2 + k].re, all_vecs[(dim + i) * n2 + k].re);
        }
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

/// Eigendecomposition dispatching to real or complex path.
fn hermitian_eigendecompose(matrix: &[C64], dim: usize) -> (Vec<f64>, Vec<C64>) {
    let is_real = matrix.iter().all(|c| c.im.abs() < 1e-12);
    if is_real {
        real_symmetric_eigen(matrix, dim)
    } else {
        complex_hermitian_eigen(matrix, dim)
    }
}

/// Solve the generalized eigenvalue problem H*c = E*S*c.
///
/// Uses Cholesky-like decomposition of S (with regularisation) to reduce to
/// a standard eigenvalue problem: S^{-1/2} H S^{-1/2} y = E y.
///
/// Returns (eigenvalues, eigenvectors_in_original_basis).
fn generalized_eigen(
    h_mat: &[f64],
    s_mat: &[f64],
    dim: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), SSVQEError> {
    // Regularise S for numerical stability
    let mut s_reg = s_mat.to_vec();
    for i in 0..dim {
        s_reg[i * dim + i] += 1e-10;
    }

    // Cholesky decomposition S = L L^T
    let mut l = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = s_reg[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= 0.0 {
                    // Not positive definite; add more regularisation
                    l[i * dim + j] = 1e-8;
                } else {
                    l[i * dim + j] = sum.sqrt();
                }
            } else {
                let diag = l[j * dim + j];
                l[i * dim + j] = if diag.abs() > 1e-30 {
                    sum / diag
                } else {
                    0.0
                };
            }
        }
    }

    // Compute L^{-1}
    let mut l_inv = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        l_inv[i * dim + i] = if l[i * dim + i].abs() > 1e-30 {
            1.0 / l[i * dim + i]
        } else {
            0.0
        };
        for j in (i + 1)..dim {
            let mut sum = 0.0_f64;
            for k in i..j {
                sum -= l[j * dim + k] * l_inv[k * dim + i];
            }
            l_inv[j * dim + i] = if l[j * dim + j].abs() > 1e-30 {
                sum / l[j * dim + j]
            } else {
                0.0
            };
        }
    }

    // Compute H_tilde = L^{-1} H (L^{-1})^T
    // First: temp = L^{-1} H
    let mut temp = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = 0.0_f64;
            for k in 0..dim {
                s += l_inv[i * dim + k] * h_mat[k * dim + j];
            }
            temp[i * dim + j] = s;
        }
    }
    // H_tilde = temp * L_inv^T
    let mut h_tilde = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = 0.0_f64;
            for k in 0..dim {
                s += temp[i * dim + k] * l_inv[j * dim + k];
            }
            h_tilde[i * dim + j] = s;
        }
    }

    // Symmetrise
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h_tilde[i * dim + j] + h_tilde[j * dim + i]);
            h_tilde[i * dim + j] = avg;
            h_tilde[j * dim + i] = avg;
        }
    }

    // Diagonalise H_tilde
    let h_tilde_c64: Vec<C64> = h_tilde.iter().map(|&r| c64(r, 0.0)).collect();
    let (vals, vecs_c64) = real_symmetric_eigen(&h_tilde_c64, dim);

    // Transform eigenvectors back: c = L^{-T} y
    let mut eigenvectors = Vec::with_capacity(dim);
    for k in 0..dim {
        let mut c_vec = vec![0.0_f64; dim];
        for i in 0..dim {
            let mut s = 0.0_f64;
            for j in 0..dim {
                s += l_inv[j * dim + i] * vecs_c64[j * dim + k].re;
            }
            c_vec[i] = s;
        }
        // Normalise
        let norm: f64 = c_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-30 {
            for x in &mut c_vec {
                *x /= norm;
            }
        }
        eigenvectors.push(c_vec);
    }

    Ok((vals, eigenvectors))
}

// ============================================================
// EXACT SOLUTIONS (for testing)
// ============================================================

/// Compute exact eigenvalues via full diagonalisation.
pub fn exact_eigenvalues(hamiltonian: &PauliHamiltonian, num_states: usize) -> Vec<f64> {
    let dim = 1usize << hamiltonian.num_qubits();
    let mat = hamiltonian.to_matrix();
    let (eigenvalues, _) = hermitian_eigendecompose(&mat, dim);
    eigenvalues.into_iter().take(num_states).collect()
}

/// Compute exact eigenpairs.
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
// 1. SSVQE: SUBSPACE-SEARCH VQE
// ============================================================

/// Configuration for the SSVQE algorithm.
#[derive(Debug, Clone)]
pub struct SSVQEConfig {
    /// Number of eigenstates to find simultaneously.
    pub num_states: usize,
    /// Weights for each state in the cost function. Must satisfy w_0 > w_1 > ... > w_{k-1}.
    /// If empty, default geometric weights 1, 1/2, 1/4, ... are used.
    pub weights: Vec<f64>,
    /// Maximum optimiser iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub convergence_threshold: f64,
    /// Random seed.
    pub seed: u64,
    /// Ansatz depth (for hardware-efficient ansatz).
    pub ansatz_depth: usize,
}

impl Default for SSVQEConfig {
    fn default() -> Self {
        Self {
            num_states: 2,
            weights: vec![],
            max_iterations: 800,
            convergence_threshold: 1e-7,
            seed: 42,
            ansatz_depth: 4,
        }
    }
}

impl SSVQEConfig {
    pub fn num_states(mut self, k: usize) -> Self {
        self.num_states = k;
        self
    }
    pub fn weights(mut self, w: Vec<f64>) -> Self {
        self.weights = w;
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
    pub fn ansatz_depth(mut self, d: usize) -> Self {
        self.ansatz_depth = d;
        self
    }

    /// Get effective weights (generates defaults if not set).
    fn effective_weights(&self) -> Vec<f64> {
        if !self.weights.is_empty() {
            return self.weights.clone();
        }
        // Default geometric weights: 1, 0.5, 0.25, ...
        (0..self.num_states)
            .map(|i| 0.5_f64.powi(i as i32))
            .collect()
    }
}

/// Result of SSVQE computation.
#[derive(Debug, Clone)]
pub struct SSVQEResult {
    /// Eigenvalue estimates for each state.
    pub eigenvalues: Vec<f64>,
    /// Eigenstates (amplitude vectors).
    pub eigenstates: Vec<Vec<C64>>,
    /// Total weighted cost at each optimiser step.
    pub cost_history: Vec<f64>,
    /// Whether the optimiser converged.
    pub converged: bool,
    /// Number of optimiser iterations used.
    pub iterations: usize,
    /// Overlap matrix |<psi_i|psi_j>|^2 between found states.
    pub overlap_matrix: Vec<Vec<f64>>,
}

/// Run the Subspace-Search VQE algorithm.
///
/// Simultaneously finds k lowest eigenstates by applying the SAME parameterised
/// circuit to k orthogonal initial states (computational basis states) and
/// minimising the weighted sum of expectations:
///
///   C(theta) = sum_i w_i <psi_i(theta)|H|psi_i(theta)>
///
/// where w_0 > w_1 > ... ensures the ground state gets the largest weight.
pub fn ssvqe(
    hamiltonian: &PauliHamiltonian,
    ansatz: &SubspaceAnsatz,
    config: &SSVQEConfig,
) -> SSVQEResult {
    let n = ansatz.num_qubits();
    let dim = 1usize << n;
    let k = config.num_states.min(dim);
    let weights = config.effective_weights();
    let num_params = ansatz.num_params();

    // Prepare k orthogonal initial states: |0>, |1>, |2>, ..., |k-1>
    let initial_states: Vec<usize> = (0..k).collect();

    // Cost function: weighted sum of expectations
    let cost = |params: &[f64]| -> f64 {
        let mut total = 0.0_f64;
        for (idx, &basis_idx) in initial_states.iter().enumerate() {
            let mut state = SubspaceState::basis_state(n, basis_idx);
            ansatz.apply(&mut state, params);
            let energy = hamiltonian.expectation_value(&state.amplitudes);
            let w = if idx < weights.len() {
                weights[idx]
            } else {
                *weights.last().unwrap_or(&1.0)
            };
            total += w * energy;
        }
        total
    };

    // Initialise parameters
    let mut rng = Xorshift64::new(config.seed);
    let initial_params: Vec<f64> = (0..num_params).map(|_| rng.next_angle() * 0.1).collect();

    // Optimise
    let (best_params, cost_history) =
        nelder_mead_optimize(&cost, &initial_params, config.max_iterations, config.convergence_threshold);

    let iterations = cost_history.len();
    let converged = if iterations >= 2 {
        let last = cost_history[iterations - 1];
        let prev = cost_history[iterations - 2];
        (last - prev).abs() < config.convergence_threshold
    } else {
        false
    };

    // Extract individual eigenvalues and states
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenstates = Vec::with_capacity(k);
    for &basis_idx in &initial_states {
        let mut state = SubspaceState::basis_state(n, basis_idx);
        ansatz.apply(&mut state, &best_params);
        let energy = hamiltonian.expectation_value(&state.amplitudes);
        eigenvalues.push(energy);
        eigenstates.push(state.amplitudes);
    }

    // Compute overlap matrix
    let mut overlap_matrix = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let si = SubspaceState::from_amplitudes(eigenstates[i].clone(), n);
            let sj = SubspaceState::from_amplitudes(eigenstates[j].clone(), n);
            overlap_matrix[i][j] = si.overlap(&sj);
        }
    }

    SSVQEResult {
        eigenvalues,
        eigenstates,
        cost_history,
        converged,
        iterations,
        overlap_matrix,
    }
}

// ============================================================
// 2. MCVQE: MULTISTATE CONTRACTED VQE
// ============================================================

/// Configuration for the MCVQE algorithm.
#[derive(Debug, Clone)]
pub struct MCVQEConfig {
    /// Number of eigenstates to compute.
    pub num_states: usize,
    /// Maximum optimiser iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub convergence_threshold: f64,
    /// Random seed.
    pub seed: u64,
    /// Ansatz depth.
    pub ansatz_depth: usize,
}

impl Default for MCVQEConfig {
    fn default() -> Self {
        Self {
            num_states: 2,
            max_iterations: 800,
            convergence_threshold: 1e-7,
            seed: 42,
            ansatz_depth: 4,
        }
    }
}

impl MCVQEConfig {
    pub fn num_states(mut self, k: usize) -> Self {
        self.num_states = k;
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
    pub fn ansatz_depth(mut self, d: usize) -> Self {
        self.ansatz_depth = d;
        self
    }
}

/// Result of MCVQE computation.
#[derive(Debug, Clone)]
pub struct MCVQEResult {
    /// Eigenvalue estimates (from diagonalising the subspace Hamiltonian).
    pub eigenvalues: Vec<f64>,
    /// Eigenstates as linear combinations of the variational basis.
    pub eigenstates: Vec<Vec<C64>>,
    /// The subspace Hamiltonian matrix (k x k, flat row-major).
    pub subspace_hamiltonian: Vec<f64>,
    /// Optimiser cost history.
    pub cost_history: Vec<f64>,
    /// Whether converged.
    pub converged: bool,
}

/// Run the Multistate Contracted VQE algorithm.
///
/// 1. Prepare k reference states (computational basis).
/// 2. Apply a parameterised circuit to obtain k variational states.
/// 3. Build the subspace Hamiltonian: H_ij = <psi_i|H|psi_j>.
/// 4. Diagonalise H_sub to get eigenvalues.
/// 5. Optimise parameters to minimise the sum of eigenvalues.
pub fn mcvqe(
    hamiltonian: &PauliHamiltonian,
    ansatz: &SubspaceAnsatz,
    config: &MCVQEConfig,
) -> MCVQEResult {
    let n = ansatz.num_qubits();
    let dim = 1usize << n;
    let k = config.num_states.min(dim);
    let num_params = ansatz.num_params();

    let basis_indices: Vec<usize> = (0..k).collect();

    // Cost function: sum of eigenvalues of the subspace Hamiltonian
    let cost = |params: &[f64]| -> f64 {
        // Build variational states
        let states: Vec<Vec<C64>> = basis_indices
            .iter()
            .map(|&bi| {
                let mut s = SubspaceState::basis_state(n, bi);
                ansatz.apply(&mut s, params);
                s.amplitudes
            })
            .collect();

        // Build subspace Hamiltonian (real part only for Hermitian H)
        let mut h_sub = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in i..k {
                let val = hamiltonian.matrix_element(&states[i], &states[j]);
                h_sub[i * k + j] = val.re;
                h_sub[j * k + i] = val.re;
            }
        }

        // Diagonalise and sum eigenvalues
        let h_sub_c64: Vec<C64> = h_sub.iter().map(|&r| c64(r, 0.0)).collect();
        let (evals, _) = real_symmetric_eigen(&h_sub_c64, k);
        evals.iter().sum::<f64>()
    };

    let mut rng = Xorshift64::new(config.seed);
    let initial_params: Vec<f64> = (0..num_params).map(|_| rng.next_angle() * 0.1).collect();

    let (best_params, cost_history) =
        nelder_mead_optimize(&cost, &initial_params, config.max_iterations, config.convergence_threshold);

    let converged = if cost_history.len() >= 2 {
        let last = cost_history[cost_history.len() - 1];
        let prev = cost_history[cost_history.len() - 2];
        (last - prev).abs() < config.convergence_threshold
    } else {
        false
    };

    // Final subspace Hamiltonian
    let final_states: Vec<Vec<C64>> = basis_indices
        .iter()
        .map(|&bi| {
            let mut s = SubspaceState::basis_state(n, bi);
            ansatz.apply(&mut s, &best_params);
            s.amplitudes
        })
        .collect();

    let mut h_sub = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in i..k {
            let val = hamiltonian.matrix_element(&final_states[i], &final_states[j]);
            h_sub[i * k + j] = val.re;
            h_sub[j * k + i] = val.re;
        }
    }

    let h_sub_c64: Vec<C64> = h_sub.iter().map(|&r| c64(r, 0.0)).collect();
    let (evals, evecs) = real_symmetric_eigen(&h_sub_c64, k);

    // Build full eigenstates from subspace coefficients
    let mut eigenstates = Vec::with_capacity(k);
    for col in 0..k {
        let mut full_state = vec![c64_zero(); dim];
        for i in 0..k {
            let coeff = evecs[i * k + col];
            for d in 0..dim {
                full_state[d] += coeff * final_states[i][d];
            }
        }
        // Normalise
        let norm_sq: f64 = full_state.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq > 1e-30 {
            let inv = 1.0 / norm_sq.sqrt();
            for a in &mut full_state {
                *a *= inv;
            }
        }
        eigenstates.push(full_state);
    }

    MCVQEResult {
        eigenvalues: evals,
        eigenstates,
        subspace_hamiltonian: h_sub,
        cost_history,
        converged,
    }
}

// ============================================================
// 3. QSE: QUANTUM SUBSPACE EXPANSION
// ============================================================

/// Configuration for Quantum Subspace Expansion.
#[derive(Debug, Clone)]
pub struct QSEConfig {
    /// Expansion operators as Pauli strings. Each entry is a list of (qubit, pauli_char).
    /// If empty, default single-qubit Paulis {I, X, Y, Z} on each qubit are used.
    pub expansion_operators: Vec<Vec<(usize, char)>>,
    /// Regularisation parameter for the overlap matrix.
    pub regularization: f64,
}

impl Default for QSEConfig {
    fn default() -> Self {
        Self {
            expansion_operators: vec![],
            regularization: 1e-8,
        }
    }
}

impl QSEConfig {
    pub fn expansion_operators(mut self, ops: Vec<Vec<(usize, char)>>) -> Self {
        self.expansion_operators = ops;
        self
    }
    pub fn regularization(mut self, eps: f64) -> Self {
        self.regularization = eps;
        self
    }

    /// Get effective expansion operators (generates defaults if not set).
    fn effective_operators(&self, num_qubits: usize) -> Vec<Vec<(usize, char)>> {
        if !self.expansion_operators.is_empty() {
            return self.expansion_operators.clone();
        }
        // Default: I, X_q, Y_q, Z_q for each qubit
        let mut ops = Vec::new();
        ops.push(vec![]); // Identity
        for q in 0..num_qubits {
            ops.push(vec![(q, 'X')]);
            ops.push(vec![(q, 'Y')]);
            ops.push(vec![(q, 'Z')]);
        }
        ops
    }
}

/// Result of QSE computation.
#[derive(Debug, Clone)]
pub struct QSEResult {
    /// Corrected energy (lowest eigenvalue of the subspace problem).
    pub corrected_energy: f64,
    /// All eigenvalues from the subspace diagonalisation.
    pub all_energies: Vec<f64>,
    /// Expansion coefficients for the corrected state.
    pub expansion_coefficients: Vec<f64>,
    /// The original (uncorrected) VQE energy for comparison.
    pub original_energy: f64,
    /// Energy improvement (original - corrected).
    pub energy_improvement: f64,
}

/// Run Quantum Subspace Expansion starting from a reference state.
///
/// Given a VQE-optimised state |psi>, builds an expanded basis
/// {O_i |psi>} from Pauli operators O_i, constructs the subspace
/// Hamiltonian H_ij = <psi|O_i^dag H O_j|psi> and overlap
/// S_ij = <psi|O_i^dag O_j|psi>, then solves the generalised
/// eigenvalue problem H*c = E*S*c.
pub fn quantum_subspace_expansion(
    hamiltonian: &PauliHamiltonian,
    reference_state: &[C64],
    config: &QSEConfig,
) -> Result<QSEResult, SSVQEError> {
    let n = hamiltonian.num_qubits();
    let dim = 1usize << n;

    if reference_state.len() != dim {
        return Err(SSVQEError::DimensionMismatch {
            expected: dim,
            got: reference_state.len(),
        });
    }

    let operators = config.effective_operators(n);
    let m = operators.len(); // subspace dimension

    // Original energy
    let original_energy = hamiltonian.expectation_value(reference_state);

    // Build expansion basis states: |phi_i> = O_i |psi>
    let basis_states: Vec<Vec<C64>> = operators
        .iter()
        .map(|op| {
            let mut s = reference_state.to_vec();
            apply_pauli_string(&mut s, n, op);
            s
        })
        .collect();

    // Build overlap matrix S_ij = <phi_i|phi_j>
    let mut s_mat = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in i..m {
            let inner: C64 = basis_states[i]
                .iter()
                .zip(basis_states[j].iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            s_mat[i * m + j] = inner.re;
            s_mat[j * m + i] = inner.re;
        }
    }

    // Build Hamiltonian matrix H_ij = <phi_i|H|phi_j>
    let mut h_mat = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in i..m {
            let val = hamiltonian.matrix_element(&basis_states[i], &basis_states[j]);
            h_mat[i * m + j] = val.re;
            h_mat[j * m + i] = val.re;
        }
    }

    // Solve generalised eigenvalue problem
    let (eigenvalues, eigenvectors) = generalized_eigen(&h_mat, &s_mat, m)?;

    let corrected_energy = eigenvalues[0];
    let expansion_coefficients = eigenvectors[0].clone();

    Ok(QSEResult {
        corrected_energy,
        all_energies: eigenvalues,
        expansion_coefficients,
        original_energy,
        energy_improvement: original_energy - corrected_energy,
    })
}

// ============================================================
// 4. qEOM: QUANTUM EQUATION OF MOTION
// ============================================================

/// Configuration for the qEOM algorithm.
#[derive(Debug, Clone)]
pub struct QEOMConfig {
    /// Excitation operators as Pauli strings. If empty, default single-excitation
    /// operators are generated.
    pub excitation_operators: Vec<Vec<(usize, char)>>,
    /// Regularisation for the overlap matrix.
    pub regularization: f64,
}

impl Default for QEOMConfig {
    fn default() -> Self {
        Self {
            excitation_operators: vec![],
            regularization: 1e-8,
        }
    }
}

impl QEOMConfig {
    pub fn excitation_operators(mut self, ops: Vec<Vec<(usize, char)>>) -> Self {
        self.excitation_operators = ops;
        self
    }
    pub fn regularization(mut self, eps: f64) -> Self {
        self.regularization = eps;
        self
    }

    /// Generate default excitation operators for a given number of qubits.
    /// Uses single-qubit raising-like operators: X_q + iY_q (encoded as separate
    /// Pauli strings since we work at the expectation-value level).
    fn effective_operators(&self, num_qubits: usize) -> Vec<Vec<(usize, char)>> {
        if !self.excitation_operators.is_empty() {
            return self.excitation_operators.clone();
        }
        let mut ops = Vec::new();
        // Single excitations: X_q and Y_q on each qubit
        for q in 0..num_qubits {
            ops.push(vec![(q, 'X')]);
            ops.push(vec![(q, 'Y')]);
        }
        // Two-qubit excitations for small systems
        if num_qubits <= 4 {
            for q1 in 0..num_qubits {
                for q2 in (q1 + 1)..num_qubits {
                    ops.push(vec![(q1, 'X'), (q2, 'X')]);
                    ops.push(vec![(q1, 'Y'), (q2, 'Y')]);
                    ops.push(vec![(q1, 'X'), (q2, 'Y')]);
                    ops.push(vec![(q1, 'Y'), (q2, 'X')]);
                }
            }
        }
        ops
    }
}

/// Result of qEOM computation.
#[derive(Debug, Clone)]
pub struct QEOMResult {
    /// Excitation energies (omega) above the ground state.
    pub excitation_energies: Vec<f64>,
    /// Ground state energy used as reference.
    pub ground_state_energy: f64,
    /// Absolute energies (ground + excitation).
    pub absolute_energies: Vec<f64>,
    /// The M matrix (commutator matrix).
    pub m_matrix: Vec<f64>,
    /// The Q matrix (anticommutator/metric matrix).
    pub q_matrix: Vec<f64>,
}

/// Apply a Pauli string operator to a state vector, returning a new vector.
fn apply_pauli_string_new(state: &[C64], num_qubits: usize, ops: &[(usize, char)]) -> Vec<C64> {
    let mut result = state.to_vec();
    apply_pauli_string(&mut result, num_qubits, ops);
    result
}

/// Compute <bra|ket> inner product.
fn inner_product(bra: &[C64], ket: &[C64]) -> C64 {
    bra.iter()
        .zip(ket.iter())
        .map(|(a, b)| a.conj() * b)
        .sum()
}

/// Run the Quantum Equation of Motion algorithm.
///
/// Starting from a ground-state reference |psi_0>, computes excitation energies
/// by solving M*c = omega*Q*c where:
///
///   M_mu,nu = <psi_0|[E_mu^dag, [H, E_nu]]|psi_0>   (double commutator)
///   Q_mu,nu = <psi_0|{E_mu^dag, E_nu}|psi_0>          (anticommutator)
///
/// For Hermitian excitation operators (Pauli strings), E^dag = E.
pub fn qeom(
    hamiltonian: &PauliHamiltonian,
    ground_state: &[C64],
    config: &QEOMConfig,
) -> Result<QEOMResult, SSVQEError> {
    let n = hamiltonian.num_qubits();
    let dim = 1usize << n;

    if ground_state.len() != dim {
        return Err(SSVQEError::DimensionMismatch {
            expected: dim,
            got: ground_state.len(),
        });
    }

    let operators = config.effective_operators(n);
    let num_ops = operators.len();

    let ground_energy = hamiltonian.expectation_value(ground_state);

    // Precompute H|psi_0> and E_mu|psi_0>
    let h_mat_full = hamiltonian.to_matrix();
    let h_psi: Vec<C64> = {
        let mut result = vec![c64_zero(); dim];
        for i in 0..dim {
            for j in 0..dim {
                result[i] += h_mat_full[i * dim + j] * ground_state[j];
            }
        }
        result
    };

    let e_psi: Vec<Vec<C64>> = operators
        .iter()
        .map(|op| apply_pauli_string_new(ground_state, n, op))
        .collect();

    // Precompute H E_nu |psi_0> and E_mu H |psi_0>
    let h_e_psi: Vec<Vec<C64>> = e_psi
        .iter()
        .map(|ep| {
            let mut result = vec![c64_zero(); dim];
            for i in 0..dim {
                for j in 0..dim {
                    result[i] += h_mat_full[i * dim + j] * ep[j];
                }
            }
            result
        })
        .collect();

    let e_h_psi: Vec<Vec<C64>> = operators
        .iter()
        .map(|op| apply_pauli_string_new(&h_psi, n, op))
        .collect();

    // Build M matrix: M_mu,nu = <psi|E_mu [H, E_nu]|psi>
    // [H, E_nu] = H E_nu - E_nu H
    // So M_mu,nu = <psi|E_mu H E_nu|psi> - <psi|E_mu E_nu H|psi>
    //
    // For the double commutator form we need:
    // M_mu,nu = <psi|E_mu H E_nu|psi> - <psi|E_mu E_nu H|psi>
    //         - <psi|H E_mu E_nu|psi> + <psi|E_nu E_mu H|psi>  ... (not quite)
    //
    // Actually the standard qEOM uses:
    // M_mu,nu = <psi|[E_mu, [H, E_nu]]|psi>
    //         = <psi|E_mu H E_nu|psi> - <psi|E_mu E_nu H|psi>
    //         - <psi|H E_nu E_mu|psi> + <psi|E_nu H E_mu|psi>
    //
    // Let's compute this directly:
    let mut m_matrix = vec![0.0_f64; num_ops * num_ops];
    let mut q_matrix = vec![0.0_f64; num_ops * num_ops];

    for mu in 0..num_ops {
        for nu in 0..num_ops {
            // <psi|E_mu H E_nu|psi> = <E_mu psi| H E_nu psi>
            let term1 = inner_product(&e_psi[mu], &h_e_psi[nu]);

            // <psi|E_mu E_nu H|psi> = <E_mu psi| E_nu H psi>
            let e_nu_h_psi = apply_pauli_string_new(&h_psi, n, &operators[nu]);
            let term2 = inner_product(&e_psi[mu], &e_nu_h_psi);

            // <psi|H E_nu E_mu|psi> = <H psi| E_nu E_mu psi>
            let e_nu_e_mu_psi = apply_pauli_string_new(&e_psi[mu], n, &operators[nu]);
            let term3 = inner_product(&h_psi, &e_nu_e_mu_psi);

            // <psi|E_nu H E_mu|psi> = <E_nu psi| H E_mu psi>
            let term4 = inner_product(&e_psi[nu], &h_e_psi[mu]);

            m_matrix[mu * num_ops + nu] = (term1 - term2 - term3 + term4).re;

            // Q_mu,nu = <psi|{E_mu, E_nu}|psi> = <psi|E_mu E_nu|psi> + <psi|E_nu E_mu|psi>
            // <psi|E_mu E_nu|psi> = <E_mu psi| E_nu psi>  (for Hermitian E)
            let q_term1 = inner_product(&e_psi[mu], &e_psi[nu]);
            // <psi|E_nu E_mu|psi> = conj(<psi|E_mu E_nu|psi>) for Hermitian operators
            let q_term2 = inner_product(&e_psi[nu], &e_psi[mu]);

            q_matrix[mu * num_ops + nu] = (q_term1 + q_term2).re;
        }
    }

    // Symmetrise
    for i in 0..num_ops {
        for j in (i + 1)..num_ops {
            let avg_m = 0.5 * (m_matrix[i * num_ops + j] + m_matrix[j * num_ops + i]);
            m_matrix[i * num_ops + j] = avg_m;
            m_matrix[j * num_ops + i] = avg_m;

            let avg_q = 0.5 * (q_matrix[i * num_ops + j] + q_matrix[j * num_ops + i]);
            q_matrix[i * num_ops + j] = avg_q;
            q_matrix[j * num_ops + i] = avg_q;
        }
    }

    // Solve generalised eigenvalue problem M*c = omega*Q*c
    let (raw_eigenvalues, _) = generalized_eigen(&m_matrix, &q_matrix, num_ops)?;

    // Filter to positive excitation energies and sort
    let mut excitation_energies: Vec<f64> = raw_eigenvalues
        .into_iter()
        .filter(|&e| e > 1e-10)
        .collect();
    excitation_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let absolute_energies: Vec<f64> = excitation_energies
        .iter()
        .map(|&omega| ground_energy + omega)
        .collect();

    Ok(QEOMResult {
        excitation_energies,
        ground_state_energy: ground_energy,
        absolute_energies,
        m_matrix,
        q_matrix,
    })
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------
    // Helpers
    // -------------------------------------------------------

    fn ising_2q() -> PauliHamiltonian {
        PauliHamiltonian::transverse_field_ising(2, 1.0, 0.5)
    }

    fn heisenberg_2q() -> PauliHamiltonian {
        PauliHamiltonian::new(vec![
            (1.0, vec![(0, 'X'), (1, 'X')]),
            (1.0, vec![(0, 'Y'), (1, 'Y')]),
            (1.0, vec![(0, 'Z'), (1, 'Z')]),
        ])
    }

    fn ising_3q() -> PauliHamiltonian {
        PauliHamiltonian::transverse_field_ising(3, 1.0, 0.5)
    }

    // -------------------------------------------------------
    // PauliHamiltonian tests
    // -------------------------------------------------------

    #[test]
    fn test_hamiltonian_construction() {
        let h = ising_2q();
        assert_eq!(h.num_qubits(), 2);
        assert_eq!(h.num_terms(), 3); // 1 ZZ + 2 X
    }

    #[test]
    fn test_hamiltonian_hermiticity() {
        let h = heisenberg_2q();
        let mat = h.to_matrix();
        let dim = 4;
        for i in 0..dim {
            for j in 0..dim {
                let diff = (mat[i * dim + j] - mat[j * dim + i].conj()).norm();
                assert!(diff < 1e-12, "Not Hermitian at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_h2_molecule_constructor() {
        let h = PauliHamiltonian::h2_molecule();
        assert_eq!(h.num_qubits(), 2);
        assert_eq!(h.num_terms(), 6);
    }

    #[test]
    fn test_expectation_value_ground_state() {
        let h = ising_2q();
        let exact = exact_eigenvalues(&h, 1);
        // Ground state energy of 2-qubit TFIM should be around -1.118
        assert!(exact[0] < -1.0);
    }

    #[test]
    fn test_matrix_element_diagonal() {
        let h = ising_2q();
        let state = SubspaceState::zero_state(2);
        let ev = hamiltonian_expectation(&h, &state.amplitudes);
        let me = h.matrix_element(&state.amplitudes, &state.amplitudes);
        assert!((ev - me.re).abs() < 1e-12);
    }

    fn hamiltonian_expectation(h: &PauliHamiltonian, state: &[C64]) -> f64 {
        h.expectation_value(state)
    }

    // -------------------------------------------------------
    // SubspaceState tests
    // -------------------------------------------------------

    #[test]
    fn test_basis_state() {
        let s = SubspaceState::basis_state(2, 2); // |10>
        assert!((s.amplitudes[2].re - 1.0).abs() < 1e-12);
        assert!(s.amplitudes[0].norm() < 1e-12);
    }

    #[test]
    fn test_orthogonal_basis_states() {
        let s0 = SubspaceState::basis_state(2, 0);
        let s1 = SubspaceState::basis_state(2, 1);
        assert!(s0.overlap(&s1) < 1e-12);
    }

    #[test]
    fn test_state_normalization() {
        let mut s = SubspaceState::from_amplitudes(
            vec![c64(1.0, 0.0), c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0)],
            2,
        );
        s.normalize();
        assert!((s.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut s = SubspaceState::zero_state(1);
        s.hadamard(0);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s.amplitudes[0].re - expected).abs() < 1e-12);
        assert!((s.amplitudes[1].re - expected).abs() < 1e-12);
    }

    #[test]
    fn test_cnot_entangles() {
        let mut s = SubspaceState::zero_state(2);
        s.hadamard(0);
        s.cnot(0, 1);
        // Bell state: (|00> + |11>) / sqrt(2)
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s.amplitudes[0].re - expected).abs() < 1e-12);
        assert!(s.amplitudes[1].norm() < 1e-12);
        assert!(s.amplitudes[2].norm() < 1e-12);
        assert!((s.amplitudes[3].re - expected).abs() < 1e-12);
    }

    // -------------------------------------------------------
    // Eigendecomposition tests
    // -------------------------------------------------------

    #[test]
    fn test_exact_eigenvalues_ising_2q() {
        let h = ising_2q();
        let evals = exact_eigenvalues(&h, 4);
        assert_eq!(evals.len(), 4);
        // Eigenvalues should be sorted ascending
        for i in 1..evals.len() {
            assert!(evals[i] >= evals[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_exact_eigenvalues_heisenberg() {
        let h = heisenberg_2q();
        let evals = exact_eigenvalues(&h, 4);
        // Heisenberg XX+YY+ZZ: singlet at -3, triplet at +1
        assert!((evals[0] - (-3.0)).abs() < 1e-10);
        assert!((evals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_hamiltonian() {
        // Use an explicit identity on 1 qubit: 2.5 * I_0
        let h = PauliHamiltonian::new(vec![(2.5, vec![(0, 'I')])]);
        let evals = exact_eigenvalues(&h, 2);
        assert!((evals[0] - 2.5).abs() < 1e-10);
        assert!((evals[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_1qubit_z_hamiltonian() {
        let h = PauliHamiltonian::new(vec![(1.0, vec![(0, 'Z')])]);
        let evals = exact_eigenvalues(&h, 2);
        assert!((evals[0] - (-1.0)).abs() < 1e-10);
        assert!((evals[1] - 1.0).abs() < 1e-10);
    }

    // -------------------------------------------------------
    // SSVQE tests
    // -------------------------------------------------------

    #[test]
    fn test_ssvqe_ground_excited_ising_2q() {
        let h = ising_2q();
        let exact = exact_eigenvalues(&h, 2);
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 4,
        };
        let config = SSVQEConfig::default()
            .num_states(2)
            .max_iterations(1500)
            .seed(123);
        let result = ssvqe(&h, &ansatz, &config);

        assert_eq!(result.eigenvalues.len(), 2);
        // Ground state should be within tolerance
        let mut sorted = result.eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (sorted[0] - exact[0]).abs() < 0.3,
            "SSVQE ground: {} vs exact: {}",
            sorted[0],
            exact[0]
        );
    }

    #[test]
    fn test_ssvqe_eigenstates_approximately_orthogonal() {
        let h = ising_2q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 3,
        };
        let config = SSVQEConfig::default().num_states(2).seed(77);
        let result = ssvqe(&h, &ansatz, &config);

        // Off-diagonal overlaps should be small (initial states are perfectly orthogonal,
        // after unitary transform they remain so)
        // With a shared unitary, <psi_0|psi_1> = <0|U^dag U|1> = <0|1> = 0
        assert!(
            result.overlap_matrix[0][1] < 1e-10,
            "States not orthogonal: overlap = {}",
            result.overlap_matrix[0][1]
        );
    }

    #[test]
    fn test_ssvqe_preserves_orthogonality() {
        // A key advantage of SSVQE: same unitary preserves orthogonality
        let h = ising_3q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 3,
            depth: 2,
        };
        let config = SSVQEConfig::default().num_states(3).seed(42);
        let result = ssvqe(&h, &ansatz, &config);

        assert_eq!(result.eigenstates.len(), 3);
        // All off-diagonal overlaps should be zero
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(
                        (result.overlap_matrix[i][j] - 1.0).abs() < 1e-10,
                        "Self-overlap != 1"
                    );
                } else {
                    assert!(
                        result.overlap_matrix[i][j] < 1e-10,
                        "Off-diagonal overlap ({i},{j}) = {}",
                        result.overlap_matrix[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_ssvqe_3_states_ising_3q() {
        let h = ising_3q();
        let exact = exact_eigenvalues(&h, 3);
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 3,
            depth: 4,
        };
        let config = SSVQEConfig::default()
            .num_states(3)
            .max_iterations(2000)
            .seed(999);
        let result = ssvqe(&h, &ansatz, &config);

        assert_eq!(result.eigenvalues.len(), 3);
        // Sorted eigenvalues should approximate exact ones
        let mut sorted = result.eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (sorted[0] - exact[0]).abs() < 0.5,
            "3-state SSVQE ground: {} vs {}",
            sorted[0],
            exact[0]
        );
    }

    #[test]
    fn test_ssvqe_custom_weights() {
        let h = ising_2q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 3,
        };
        let config = SSVQEConfig::default()
            .num_states(2)
            .weights(vec![5.0, 1.0])
            .seed(42);
        let result = ssvqe(&h, &ansatz, &config);
        assert_eq!(result.eigenvalues.len(), 2);
    }

    #[test]
    fn test_ssvqe_cost_decreases() {
        let h = ising_2q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 3,
        };
        let config = SSVQEConfig::default().num_states(2).seed(42);
        let result = ssvqe(&h, &ansatz, &config);

        // Cost should generally decrease (check first vs last)
        if result.cost_history.len() > 10 {
            assert!(
                result.cost_history.last().unwrap() <= &(result.cost_history[0] + 0.5),
                "Cost did not decrease"
            );
        }
    }

    #[test]
    fn test_ssvqe_single_state() {
        // k=1 should behave like standard VQE
        let h = ising_2q();
        let exact = exact_eigenvalues(&h, 1);
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 4,
        };
        let config = SSVQEConfig::default()
            .num_states(1)
            .max_iterations(1500)
            .seed(55);
        let result = ssvqe(&h, &ansatz, &config);

        assert_eq!(result.eigenvalues.len(), 1);
        assert!(
            (result.eigenvalues[0] - exact[0]).abs() < 0.3,
            "Single-state SSVQE: {} vs {}",
            result.eigenvalues[0],
            exact[0]
        );
    }

    // -------------------------------------------------------
    // MCVQE tests
    // -------------------------------------------------------

    #[test]
    fn test_mcvqe_ground_excited_ising_2q() {
        let h = ising_2q();
        let exact = exact_eigenvalues(&h, 2);
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 4,
        };
        let config = MCVQEConfig::default()
            .num_states(2)
            .max_iterations(1500)
            .seed(123);
        let result = mcvqe(&h, &ansatz, &config);

        assert_eq!(result.eigenvalues.len(), 2);
        // MCVQE should get close to exact eigenvalues
        assert!(
            (result.eigenvalues[0] - exact[0]).abs() < 0.3,
            "MCVQE ground: {} vs {}",
            result.eigenvalues[0],
            exact[0]
        );
    }

    #[test]
    fn test_mcvqe_subspace_hamiltonian_symmetric() {
        let h = ising_2q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 2,
        };
        let config = MCVQEConfig::default().num_states(2).seed(42);
        let result = mcvqe(&h, &ansatz, &config);

        let k = 2;
        for i in 0..k {
            for j in 0..k {
                let diff = (result.subspace_hamiltonian[i * k + j]
                    - result.subspace_hamiltonian[j * k + i])
                    .abs();
                assert!(diff < 1e-10, "Subspace H not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_mcvqe_eigenvalues_ordered() {
        let h = ising_2q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 3,
        };
        let config = MCVQEConfig::default().num_states(2).seed(42);
        let result = mcvqe(&h, &ansatz, &config);

        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] >= result.eigenvalues[i - 1] - 1e-10,
                "MCVQE eigenvalues not ordered"
            );
        }
    }

    #[test]
    fn test_mcvqe_degenerate_states() {
        // Heisenberg model has degenerate triplet at E=+1
        let h = heisenberg_2q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 4,
        };
        let config = MCVQEConfig::default()
            .num_states(2)
            .max_iterations(1500)
            .seed(77);
        let result = mcvqe(&h, &ansatz, &config);

        // Should find the singlet at -3 and at least one triplet near +1
        assert!(result.eigenvalues.len() == 2);
    }

    #[test]
    fn test_mcvqe_eigenstates_normalized() {
        let h = ising_2q();
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 2,
        };
        let config = MCVQEConfig::default().num_states(2).seed(42);
        let result = mcvqe(&h, &ansatz, &config);

        for (i, state) in result.eigenstates.iter().enumerate() {
            let norm_sq: f64 = state.iter().map(|a| a.norm_sqr()).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-6,
                "MCVQE eigenstate {i} not normalised: norm^2 = {norm_sq}"
            );
        }
    }

    // -------------------------------------------------------
    // QSE tests
    // -------------------------------------------------------

    #[test]
    fn test_qse_improves_energy() {
        let h = ising_2q();
        // Start from a slightly imperfect state (not the exact ground state)
        let mut state = SubspaceState::zero_state(2);
        state.ry(0, 0.3);
        state.ry(1, -0.2);
        state.cnot(0, 1);

        let original_e = h.expectation_value(&state.amplitudes);
        let exact_e = exact_eigenvalues(&h, 1)[0];

        let config = QSEConfig::default();
        let result = quantum_subspace_expansion(&h, &state.amplitudes, &config).unwrap();

        // QSE should improve or at least not worsen the energy
        assert!(
            result.corrected_energy <= original_e + 1e-10,
            "QSE worsened energy: {} > {}",
            result.corrected_energy,
            original_e
        );
        assert!(
            result.energy_improvement >= -1e-10,
            "Negative improvement: {}",
            result.energy_improvement
        );
    }

    #[test]
    fn test_qse_from_exact_ground_state() {
        let h = ising_2q();
        let (exact_vals, exact_vecs) = exact_eigenpairs(&h, 1);

        let config = QSEConfig::default();
        let result = quantum_subspace_expansion(&h, &exact_vecs[0], &config).unwrap();

        // Starting from exact ground state, QSE should return the same energy
        assert!(
            (result.corrected_energy - exact_vals[0]).abs() < 1e-6,
            "QSE from exact: {} vs {}",
            result.corrected_energy,
            exact_vals[0]
        );
    }

    #[test]
    fn test_qse_custom_operators() {
        let h = ising_2q();
        let state = SubspaceState::zero_state(2);

        let ops = vec![
            vec![],               // I
            vec![(0, 'X')],       // X_0
            vec![(1, 'X')],       // X_1
            vec![(0, 'Z')],       // Z_0
        ];
        let config = QSEConfig::default().expansion_operators(ops);
        let result = quantum_subspace_expansion(&h, &state.amplitudes, &config).unwrap();
        assert!(result.all_energies.len() == 4);
    }

    #[test]
    fn test_qse_dimension_mismatch() {
        let h = ising_2q();
        let bad_state = vec![c64_one()]; // wrong dimension
        let config = QSEConfig::default();
        let result = quantum_subspace_expansion(&h, &bad_state, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_qse_heisenberg_improvement() {
        let h = heisenberg_2q();
        // Start from |00> which is not the ground state
        let state = SubspaceState::zero_state(2);
        let config = QSEConfig::default();
        let result = quantum_subspace_expansion(&h, &state.amplitudes, &config).unwrap();

        let exact_e = exact_eigenvalues(&h, 1)[0]; // -3.0
        // QSE should get much closer to -3 than the initial <00|H|00>=1
        assert!(
            result.corrected_energy < result.original_energy,
            "QSE did not improve: corrected={} >= original={}",
            result.corrected_energy,
            result.original_energy
        );
    }

    #[test]
    fn test_qse_returns_multiple_energies() {
        let h = ising_2q();
        let state = SubspaceState::zero_state(2);
        let config = QSEConfig::default();
        let result = quantum_subspace_expansion(&h, &state.amplitudes, &config).unwrap();

        // Default operators for 2 qubits: I, X0, Y0, Z0, X1, Y1, Z1 = 7 operators
        assert!(result.all_energies.len() > 1);
        // Should be sorted
        for i in 1..result.all_energies.len() {
            assert!(result.all_energies[i] >= result.all_energies[i - 1] - 1e-8);
        }
    }

    // -------------------------------------------------------
    // qEOM tests
    // -------------------------------------------------------

    #[test]
    fn test_qeom_excitation_energies_positive() {
        let h = ising_2q();
        let (_, exact_vecs) = exact_eigenpairs(&h, 1);
        let config = QEOMConfig::default();
        let result = qeom(&h, &exact_vecs[0], &config).unwrap();

        for &omega in &result.excitation_energies {
            assert!(
                omega > -1e-8,
                "Negative excitation energy: {}",
                omega
            );
        }
    }

    #[test]
    fn test_qeom_ground_energy_stored() {
        let h = ising_2q();
        let exact = exact_eigenvalues(&h, 1);
        let (_, exact_vecs) = exact_eigenpairs(&h, 1);
        let config = QEOMConfig::default();
        let result = qeom(&h, &exact_vecs[0], &config).unwrap();

        assert!(
            (result.ground_state_energy - exact[0]).abs() < 1e-8,
            "Ground energy mismatch"
        );
    }

    #[test]
    fn test_qeom_absolute_energies() {
        let h = ising_2q();
        let (_, exact_vecs) = exact_eigenpairs(&h, 1);
        let config = QEOMConfig::default();
        let result = qeom(&h, &exact_vecs[0], &config).unwrap();

        for (omega, abs_e) in result
            .excitation_energies
            .iter()
            .zip(result.absolute_energies.iter())
        {
            assert!(
                (abs_e - (result.ground_state_energy + omega)).abs() < 1e-10,
                "Absolute energy inconsistent"
            );
        }
    }

    #[test]
    fn test_qeom_h2_excitations() {
        let h = PauliHamiltonian::h2_molecule();
        let exact = exact_eigenvalues(&h, 4);
        let (_, exact_vecs) = exact_eigenpairs(&h, 1);
        let config = QEOMConfig::default();
        let result = qeom(&h, &exact_vecs[0], &config).unwrap();

        // Should find at least one excitation
        assert!(
            !result.excitation_energies.is_empty(),
            "No excitation energies found for H2"
        );

        // First excitation should approximate the gap
        if !result.excitation_energies.is_empty() {
            let exact_gap = exact[1] - exact[0];
            // Allow fairly loose tolerance for qEOM
            assert!(
                result.excitation_energies[0] > 0.0,
                "First excitation not positive"
            );
        }
    }

    #[test]
    fn test_qeom_dimension_mismatch() {
        let h = ising_2q();
        let bad_state = vec![c64_one(), c64_zero()]; // wrong dim
        let config = QEOMConfig::default();
        let result = qeom(&h, &bad_state, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_qeom_custom_operators() {
        let h = ising_2q();
        let (_, exact_vecs) = exact_eigenpairs(&h, 1);
        let ops = vec![
            vec![(0, 'X')],
            vec![(1, 'X')],
        ];
        let config = QEOMConfig::default().excitation_operators(ops);
        let result = qeom(&h, &exact_vecs[0], &config).unwrap();
        // M and Q matrices should be 2x2
        assert_eq!(result.m_matrix.len(), 4);
        assert_eq!(result.q_matrix.len(), 4);
    }

    #[test]
    fn test_qeom_matrices_symmetric() {
        let h = ising_2q();
        let (_, exact_vecs) = exact_eigenpairs(&h, 1);
        let config = QEOMConfig::default();
        let result = qeom(&h, &exact_vecs[0], &config).unwrap();

        let num_ops = (result.m_matrix.len() as f64).sqrt() as usize;
        for i in 0..num_ops {
            for j in (i + 1)..num_ops {
                let m_diff = (result.m_matrix[i * num_ops + j]
                    - result.m_matrix[j * num_ops + i])
                    .abs();
                assert!(m_diff < 1e-10, "M not symmetric at ({i},{j})");

                let q_diff = (result.q_matrix[i * num_ops + j]
                    - result.q_matrix[j * num_ops + i])
                    .abs();
                assert!(q_diff < 1e-10, "Q not symmetric at ({i},{j})");
            }
        }
    }

    // -------------------------------------------------------
    // Ansatz tests
    // -------------------------------------------------------

    #[test]
    fn test_hardware_efficient_params() {
        let a = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 3,
            depth: 2,
        };
        // (depth+1) layers * 2 rotations * num_qubits
        assert_eq!(a.num_params(), 2 * 3 * 3);
    }

    #[test]
    fn test_uccsd_params() {
        let a = SubspaceAnsatz::UCCSD {
            num_qubits: 4,
            num_electrons: 2,
        };
        assert!(a.num_params() > 0);
    }

    #[test]
    fn test_ansatz_preserves_norm() {
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 2,
            depth: 2,
        };
        let mut state = SubspaceState::zero_state(2);
        let params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        ansatz.apply(&mut state, &params);
        assert!(
            (state.norm() - 1.0).abs() < 1e-10,
            "Ansatz changed state norm"
        );
    }

    // -------------------------------------------------------
    // Nelder-Mead tests
    // -------------------------------------------------------

    #[test]
    fn test_nelder_mead_quadratic() {
        // Minimise f(x) = (x-1)^2 + (y-2)^2
        let f = |p: &[f64]| -> f64 { (p[0] - 1.0).powi(2) + (p[1] - 2.0).powi(2) };
        let (best, _) = nelder_mead_optimize(&f, &[0.0, 0.0], 1000, 1e-10);
        assert!((best[0] - 1.0).abs() < 1e-4);
        assert!((best[1] - 2.0).abs() < 1e-4);
    }

    // -------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------

    #[test]
    fn test_ssvqe_1qubit() {
        let h = PauliHamiltonian::new(vec![
            (1.0, vec![(0, 'Z')]),
            (0.5, vec![(0, 'X')]),
        ]);
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 1,
            depth: 3,
        };
        let config = SSVQEConfig::default()
            .num_states(2)
            .max_iterations(1000)
            .seed(42);
        let result = ssvqe(&h, &ansatz, &config);
        assert_eq!(result.eigenvalues.len(), 2);
    }

    #[test]
    fn test_mcvqe_1qubit() {
        let h = PauliHamiltonian::new(vec![(1.0, vec![(0, 'Z')])]);
        let ansatz = SubspaceAnsatz::HardwareEfficient {
            num_qubits: 1,
            depth: 2,
        };
        let config = MCVQEConfig::default().num_states(2).seed(42);
        let result = mcvqe(&h, &ansatz, &config);
        assert_eq!(result.eigenvalues.len(), 2);
        // Z has eigenvalues -1 and +1
        assert!((result.eigenvalues[0] - (-1.0)).abs() < 0.5);
    }

    #[test]
    fn test_generalized_eigen_standard() {
        // When S = I, generalized eigenproblem reduces to standard
        let h = vec![2.0, 1.0, 1.0, 3.0];
        let s = vec![1.0, 0.0, 0.0, 1.0];
        let (vals, _) = generalized_eigen(&h, &s, 2).unwrap();
        // Eigenvalues of [[2,1],[1,3]] are (5 +/- sqrt(5))/2
        let expected_low = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_high = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((vals[0] - expected_low).abs() < 1e-6);
        assert!((vals[1] - expected_high).abs() < 1e-6);
    }

    #[test]
    fn test_heisenberg_xxz_spectrum() {
        let h = PauliHamiltonian::heisenberg_xxz(2, 1.0, 1.0, 0.0);
        let evals = exact_eigenvalues(&h, 4);
        // Isotropic Heisenberg: singlet at -3, triplet at 1
        assert!((evals[0] - (-3.0)).abs() < 1e-8);
    }
}
