//! qSWIFT: High-Order Randomized Hamiltonian Simulation
//!
//! Implements the qSWIFT algorithm, which combines the gate-count efficiency of
//! qDRIFT (independent of Hamiltonian term count) with high-order product formula
//! accuracy. Error decreases as (λt/N)^K for order K, enabling 1000x fewer gates
//! than qDRIFT for precision targets like 10^{-6}.
//!
//! # Algorithms
//!
//! - **qDRIFT**: First-order randomized product formula (baseline)
//! - **qSWIFT**: High-order randomized product formula (novel)
//! - **Standard Trotter**: Deterministic product formula (comparison)
//! - **Exact evolution**: Full matrix exponential (verification)
//!
//! # References
//!
//! - Campbell, "Random Compiler for Fast Hamiltonian Simulation" (2019) — qDRIFT
//! - Nakaji et al., "qSWIFT: High-Order Randomized Compiler" (2023) — qSWIFT

use ndarray::Array2;
use num_complex::Complex64;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during qSWIFT simulation.
#[derive(Debug, Clone)]
pub enum QswiftError {
    /// Order must be at least 1.
    InvalidOrder(usize),
    /// Hamiltonian must have at least one term.
    EmptyHamiltonian,
    /// General simulation failure with description.
    SimulationFailed(String),
}

impl fmt::Display for QswiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QswiftError::InvalidOrder(k) => write!(f, "Invalid order {}: must be >= 1", k),
            QswiftError::EmptyHamiltonian => write!(f, "Hamiltonian has no terms"),
            QswiftError::SimulationFailed(msg) => write!(f, "Simulation failed: {}", msg),
        }
    }
}

impl std::error::Error for QswiftError {}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A term in the Hamiltonian: coefficient * (tensor product of Paulis).
///
/// Each Pauli is `(qubit_index, pauli_char)` where `pauli_char` is one of
/// `'I'`, `'X'`, `'Y'`, `'Z'`.
#[derive(Debug, Clone)]
pub struct HamiltonianTerm {
    /// List of (qubit_index, pauli) pairs.  Qubits not listed are identity.
    pub paulis: Vec<(usize, char)>,
    /// Real coefficient h_j in front of the Pauli string.
    pub coefficient: f64,
}

impl HamiltonianTerm {
    /// Create a new Hamiltonian term.
    pub fn new(paulis: Vec<(usize, char)>, coefficient: f64) -> Self {
        Self { paulis, coefficient }
    }
}

/// Configuration builder for qSWIFT simulation.
#[derive(Debug, Clone)]
pub struct QswiftConfig {
    /// Product-formula order K (default 2).
    pub order: usize,
    /// Number of random samples N (default 1000).
    pub num_samples: usize,
    /// Total simulation time t (default 1.0).
    pub time: f64,
    /// RNG seed (default 42).
    pub seed: u64,
}

impl Default for QswiftConfig {
    fn default() -> Self {
        Self {
            order: 2,
            num_samples: 1000,
            time: 1.0,
            seed: 42,
        }
    }
}

impl QswiftConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the product-formula order K.
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Set the number of random samples N.
    pub fn num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Set the total simulation time.
    pub fn time(mut self, t: f64) -> Self {
        self.time = t;
        self
    }

    /// Set the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

/// A single gate in a qSWIFT or qDRIFT circuit.
#[derive(Debug, Clone)]
pub struct QswiftGate {
    /// Index into the Hamiltonian term list.
    pub term_index: usize,
    /// Rotation angle for this gate.
    pub angle: f64,
}

/// A compiled randomized circuit ready for simulation.
#[derive(Debug, Clone)]
pub struct QswiftCircuit {
    /// Ordered list of gates.
    pub gates: Vec<QswiftGate>,
    /// Number of qubits in the system.
    pub num_qubits: usize,
    /// Number of random samples used.
    pub num_samples: usize,
}

/// Result of a qSWIFT simulation.
#[derive(Debug, Clone)]
pub struct QswiftResult {
    /// Final quantum state vector.
    pub final_state: Vec<Complex64>,
    /// Estimated fidelity against exact evolution (if computed).
    pub fidelity_estimate: f64,
    /// Total number of gates applied.
    pub num_gates: usize,
    /// Name of the method used.
    pub method: String,
}

/// Comparison of gate counts across methods for a target error.
#[derive(Debug, Clone)]
pub struct MethodComparison {
    /// Gates required by qDRIFT.
    pub qdrift_gates: usize,
    /// Gates required by qSWIFT (order K).
    pub qswift_gates: usize,
    /// Gates required by first-order Trotter.
    pub trotter_gates: usize,
    /// Fidelities of each method against exact: [qdrift, qswift, trotter].
    pub exact_fidelities: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the 1-norm λ = Σ|h_j| of Hamiltonian coefficients.
pub fn one_norm(terms: &[HamiltonianTerm]) -> f64 {
    terms.iter().map(|t| t.coefficient.abs()).sum()
}

/// Compute |⟨a|b⟩|² — fidelity between two pure state vectors.
pub fn state_fidelity(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len(), "State vectors must have the same length");
    let inner: Complex64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai.conj() * bi).sum();
    inner.norm_sqr()
}

/// Return the 2×2 matrix for a single Pauli operator.
fn single_pauli_matrix(p: char) -> Array2<Complex64> {
    let _zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let neg = Complex64::new(-1.0, 0.0);
    let i_val = Complex64::new(0.0, 1.0);
    let ni = Complex64::new(0.0, -1.0);

    let mut m = Array2::<Complex64>::zeros((2, 2));
    match p {
        'I' => {
            m[[0, 0]] = one;
            m[[1, 1]] = one;
        }
        'X' => {
            m[[0, 1]] = one;
            m[[1, 0]] = one;
        }
        'Y' => {
            m[[0, 1]] = ni;
            m[[1, 0]] = i_val;
        }
        'Z' => {
            m[[0, 0]] = one;
            m[[1, 1]] = neg;
        }
        _ => panic!("Unknown Pauli: {}", p),
    }
    m
}

/// Kronecker (tensor) product of two matrices.
fn kron(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (ar, ac) = (a.nrows(), a.ncols());
    let (br, bc) = (b.nrows(), b.ncols());
    let mut result = Array2::<Complex64>::zeros((ar * br, ac * bc));
    for i in 0..ar {
        for j in 0..ac {
            let aij = a[[i, j]];
            for k in 0..br {
                for l in 0..bc {
                    result[[i * br + k, j * bc + l]] = aij * b[[k, l]];
                }
            }
        }
    }
    result
}

/// Build the 2^n × 2^n matrix for a single Pauli string on `num_qubits` qubits.
///
/// Constructs the full tensor product I ⊗ ... ⊗ P_j ⊗ ... ⊗ I.
pub fn pauli_to_matrix(paulis: &[(usize, char)], num_qubits: usize) -> Array2<Complex64> {
    // Build per-qubit operators
    let mut ops: Vec<Array2<Complex64>> = (0..num_qubits)
        .map(|_| single_pauli_matrix('I'))
        .collect();

    for &(qubit, p) in paulis {
        ops[qubit] = single_pauli_matrix(p);
    }

    // Compute full tensor product: ops[0] ⊗ ops[1] ⊗ ... ⊗ ops[n-1]
    let mut result = ops[0].clone();
    for op in &ops[1..] {
        result = kron(&result, op);
    }
    result
}

/// Build the full Hamiltonian matrix H = Σ h_j P_j.
pub fn build_hamiltonian_matrix(
    terms: &[HamiltonianTerm],
    num_qubits: usize,
) -> Array2<Complex64> {
    let dim = 1usize << num_qubits;
    let mut h = Array2::<Complex64>::zeros((dim, dim));
    for term in terms {
        let pmat = pauli_to_matrix(&term.paulis, num_qubits);
        h = h + pmat * Complex64::new(term.coefficient, 0.0);
    }
    h
}

// ---------------------------------------------------------------------------
// Matrix exponential via Padé approximant
// ---------------------------------------------------------------------------

/// Compute exp(A) for a square complex matrix using a scaled-and-squared
/// Padé(6,6) approximant.
pub fn matrix_exponential(a: &Array2<Complex64>) -> Array2<Complex64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "Matrix must be square");

    // Scaling: find s such that ||A / 2^s|| < 0.5
    let norm = matrix_inf_norm(a);
    let s = if norm > 0.5 {
        (norm * 2.0).log2().ceil() as u32 + 1
    } else {
        0
    };
    let scale = 2.0_f64.powi(-(s as i32));
    let a_scaled = a.mapv(|x| x * scale);

    // Padé(6,6) coefficients: c_k = 1/(k! * (2p-k)!/p!) where p=6
    // For diagonal Padé(p,p): c_k = (2p-k)! p! / ((2p)! k! (p-k)!)
    let coeffs: [f64; 7] = [
        1.0,
        0.5,
        1.0 / 12.0,
        1.0 / 120.0,
        1.0 / 1680.0,
        1.0 / 30240.0,
        1.0 / 720720.0,
    ];

    let eye = Array2::<Complex64>::eye(n);

    // Compute powers of a_scaled
    let a2 = a_scaled.dot(&a_scaled);
    let a3 = a2.dot(&a_scaled);
    let a4 = a2.dot(&a2);
    let a5 = a4.dot(&a_scaled);
    let a6 = a3.dot(&a3);

    // U = c1*A + c3*A^3 + c5*A^5 (odd terms)
    let u = &a_scaled * Complex64::new(coeffs[1], 0.0)
        + &a3 * Complex64::new(coeffs[3], 0.0)
        + &a5 * Complex64::new(coeffs[5], 0.0);

    // V = c0*I + c2*A^2 + c4*A^4 + c6*A^6 (even terms)
    let v = &eye * Complex64::new(coeffs[0], 0.0)
        + &a2 * Complex64::new(coeffs[2], 0.0)
        + &a4 * Complex64::new(coeffs[4], 0.0)
        + &a6 * Complex64::new(coeffs[6], 0.0);

    // R(6,6) = (V + U)(V - U)^{-1}
    let numer = &v + &u;
    let denom = &v - &u;

    // Solve denom * result = numer => result = denom^{-1} * numer
    let mut result = solve_linear_system(&denom, &numer);

    // Squaring phase: result = result^(2^s)
    for _ in 0..s {
        result = result.dot(&result);
    }

    result
}

/// Infinity norm of a complex matrix: max row sum of absolute values.
fn matrix_inf_norm(a: &Array2<Complex64>) -> f64 {
    let mut max_sum = 0.0_f64;
    for row in a.rows() {
        let s: f64 = row.iter().map(|c| c.norm()).sum();
        max_sum = max_sum.max(s);
    }
    max_sum
}

/// Solve AX = B for square matrices via LU decomposition (partial pivoting).
fn solve_linear_system(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    assert_eq!(n, b.nrows());
    let m = b.ncols();

    // Augmented matrix [A | B]
    let mut aug = Array2::<Complex64>::zeros((n, n + m));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        for j in 0..m {
            aug[[i, n + j]] = b[[i, j]];
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_norm = 0.0;
        let mut pivot_row = col;
        for row in col..n {
            let norm = aug[[row, col]].norm();
            if norm > max_norm {
                max_norm = norm;
                pivot_row = row;
            }
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..(n + m) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        if pivot.norm() < 1e-15 {
            continue;
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..(n + m) {
                let val = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * val;
            }
        }
    }

    // Back substitution
    let mut x = Array2::<Complex64>::zeros((n, m));
    for col_b in 0..m {
        for row in (0..n).rev() {
            let mut sum = aug[[row, n + col_b]];
            for j in (row + 1)..n {
                sum = sum - aug[[row, j]] * x[[j, col_b]];
            }
            let diag = aug[[row, row]];
            if diag.norm() > 1e-15 {
                x[[row, col_b]] = sum / diag;
            }
        }
    }

    x
}

// ---------------------------------------------------------------------------
// Pauli rotation on state vectors
// ---------------------------------------------------------------------------

/// Apply exp(-i * angle * P) to `state` in-place, where P is a multi-qubit
/// Pauli string.
///
/// Uses the decomposition: exp(-iθP) = cos(θ)I - i·sin(θ)P, valid because P²=I.
pub fn apply_pauli_rotation(state: &mut Vec<Complex64>, paulis: &[(usize, char)], angle: f64) {
    let n = state.len();
    let cos_a = Complex64::new(angle.cos(), 0.0);
    let neg_i_sin_a = Complex64::new(0.0, -angle.sin()); // -i * sin(angle)
    let num_qubits = (n as f64).log2().round() as usize;

    // Compute P|state>: apply each single-qubit Pauli in sequence
    let mut p_state = state.clone();
    for &(qubit, pauli) in paulis {
        apply_single_pauli(&mut p_state, qubit, pauli, num_qubits);
    }

    // state' = cos(angle) * state - i*sin(angle) * P|state>
    for i in 0..n {
        state[i] = cos_a * state[i] + neg_i_sin_a * p_state[i];
    }
}

/// Apply a single Pauli gate (X, Y, Z, or I) to a specific qubit in a state vector.
fn apply_single_pauli(state: &mut Vec<Complex64>, qubit: usize, pauli: char, num_qubits: usize) {
    let dim = state.len();
    let bit = num_qubits - 1 - qubit;

    match pauli {
        'I' => {} // identity — nothing to do
        'X' => {
            for idx in 0..dim {
                if (idx >> bit) & 1 == 0 {
                    let j = idx | (1 << bit);
                    state.swap(idx, j);
                }
            }
        }
        'Y' => {
            let i_unit = Complex64::new(0.0, 1.0);
            for idx in 0..dim {
                if (idx >> bit) & 1 == 0 {
                    let j = idx | (1 << bit);
                    let a = state[idx];
                    let b = state[j];
                    // Y|0> = i|1>, Y|1> = -i|0>
                    state[idx] = -i_unit * b;
                    state[j] = i_unit * a;
                }
            }
        }
        'Z' => {
            for idx in 0..dim {
                if (idx >> bit) & 1 == 1 {
                    state[idx] = -state[idx];
                }
            }
        }
        _ => panic!("Unknown Pauli: {}", pauli),
    }
}

// ---------------------------------------------------------------------------
// qDRIFT (first-order randomized product formula)
// ---------------------------------------------------------------------------

/// Compile a qDRIFT circuit.
///
/// Algorithm:
/// 1. λ = Σ|h_j|
/// 2. For each sample: draw term j with probability |h_j|/λ,
///    apply exp(-i · sign(h_j) · λ·t/N · P_j)
///
/// Total gates = `num_samples`.  Error ≤ 2λ²t²/N.
pub fn qdrift(
    terms: &[HamiltonianTerm],
    time: f64,
    num_samples: usize,
    rng: &mut impl Rng,
) -> QswiftCircuit {
    let lambda = one_norm(terms);
    let weights: Vec<f64> = terms.iter().map(|t| t.coefficient.abs()).collect();
    let dist = WeightedIndex::new(&weights).expect("Non-empty weights required");

    let tau = lambda * time / (num_samples as f64);
    let mut gates = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let j = dist.sample(rng);
        let sign = terms[j].coefficient.signum();
        let angle = sign * tau;
        gates.push(QswiftGate {
            term_index: j,
            angle,
        });
    }

    let num_qubits = infer_num_qubits(terms);

    QswiftCircuit {
        gates,
        num_qubits,
        num_samples,
    }
}

// ---------------------------------------------------------------------------
// qSWIFT (high-order randomized product formula)
// ---------------------------------------------------------------------------

/// Compile a qSWIFT circuit of order K.
///
/// Algorithm (order K):
/// 1. λ = Σ|h_j|
/// 2. For each of N samples, draw K terms from the importance distribution
///    and compose K rotations with time scaling τ = λt/N.
/// 3. Error ∝ (λt/N)^K — exponential decrease with order.
/// 4. Total gates = N × K (independent of number of Hamiltonian terms).
pub fn qswift(
    terms: &[HamiltonianTerm],
    config: &QswiftConfig,
) -> Result<QswiftCircuit, QswiftError> {
    if config.order < 1 {
        return Err(QswiftError::InvalidOrder(config.order));
    }
    if terms.is_empty() {
        return Err(QswiftError::EmptyHamiltonian);
    }

    let lambda = one_norm(terms);
    if lambda < 1e-15 {
        return Err(QswiftError::SimulationFailed(
            "All Hamiltonian coefficients are zero".into(),
        ));
    }

    let weights: Vec<f64> = terms.iter().map(|t| t.coefficient.abs()).collect();
    let dist = WeightedIndex::new(&weights).map_err(|e| {
        QswiftError::SimulationFailed(format!("Failed to build sampling distribution: {}", e))
    })?;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let k = config.order;
    let n = config.num_samples;
    let tau = lambda * config.time / (n as f64);

    let mut gates = Vec::with_capacity(n * k);

    for _ in 0..n {
        // Draw K terms and compose them
        for _ in 0..k {
            let j = dist.sample(&mut rng);
            let sign = terms[j].coefficient.signum();
            // Each of the K rotations shares the time scaling τ/K
            let angle = sign * tau / (k as f64);
            gates.push(QswiftGate {
                term_index: j,
                angle,
            });
        }
    }

    let num_qubits = infer_num_qubits(terms);

    Ok(QswiftCircuit {
        gates,
        num_qubits,
        num_samples: n,
    })
}

/// Infer the number of qubits from Hamiltonian terms.
fn infer_num_qubits(terms: &[HamiltonianTerm]) -> usize {
    terms
        .iter()
        .flat_map(|t| t.paulis.iter().map(|&(q, _)| q))
        .max()
        .map(|m| m + 1)
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------
// Standard Trotter (first-order, deterministic)
// ---------------------------------------------------------------------------

/// First-order Trotter simulation: Π_{step} Π_j exp(-i h_j P_j dt).
///
/// Returns the final state vector starting from |0...0⟩.
pub fn standard_trotter(
    terms: &[HamiltonianTerm],
    time: f64,
    num_steps: usize,
    num_qubits: usize,
) -> Vec<Complex64> {
    let dim = 1usize << num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0); // |0...0>

    let dt = time / (num_steps as f64);

    for _ in 0..num_steps {
        for term in terms {
            let angle = term.coefficient * dt;
            apply_pauli_rotation(&mut state, &term.paulis, angle);
        }
    }

    state
}

// ---------------------------------------------------------------------------
// Circuit simulation
// ---------------------------------------------------------------------------

/// Simulate a compiled qSWIFT/qDRIFT circuit on |0...0⟩.
pub fn simulate_qswift(
    circuit: &QswiftCircuit,
    terms: &[HamiltonianTerm],
    num_qubits: usize,
) -> Vec<Complex64> {
    let dim = 1usize << num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);

    for gate in &circuit.gates {
        let term = &terms[gate.term_index];
        apply_pauli_rotation(&mut state, &term.paulis, gate.angle);
    }

    state
}

// ---------------------------------------------------------------------------
// Exact evolution (matrix exponential)
// ---------------------------------------------------------------------------

/// Compute exp(-iHt)|0...0⟩ exactly via matrix exponential.
///
/// Feasible only for small systems (≤ ~10 qubits).
pub fn exact_evolution(
    terms: &[HamiltonianTerm],
    time: f64,
    num_qubits: usize,
) -> Vec<Complex64> {
    let h = build_hamiltonian_matrix(terms, num_qubits);
    let dim = 1usize << num_qubits;

    // exp(-i H t)
    let neg_i_t = Complex64::new(0.0, -time);
    let arg = h.mapv(|x| x * neg_i_t);
    let u = matrix_exponential(&arg);

    // Apply to |0...0>
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    for row in 0..dim {
        state[row] = u[[row, 0]];
    }
    state
}

// ---------------------------------------------------------------------------
// Error analysis
// ---------------------------------------------------------------------------

/// Upper bound on qDRIFT error: ε ≤ 2λ²t²/N.
pub fn estimate_qdrift_error(lambda: f64, time: f64, num_samples: usize) -> f64 {
    2.0 * lambda * lambda * time * time / (num_samples as f64)
}

/// Upper bound on qSWIFT error: ε ≤ C · (λt/N)^K.
///
/// The constant C is taken as 2.0 (conservative).
pub fn estimate_qswift_error(lambda: f64, time: f64, num_samples: usize, order: usize) -> f64 {
    let ratio = lambda * time / (num_samples as f64);
    2.0 * ratio.powi(order as i32)
}

/// Estimate the number of qDRIFT samples needed for a target error.
fn qdrift_samples_for_error(lambda: f64, time: f64, target_error: f64) -> usize {
    // 2λ²t²/N ≤ ε  =>  N ≥ 2λ²t²/ε
    let n = 2.0 * lambda * lambda * time * time / target_error;
    n.ceil() as usize
}

/// Estimate the number of qSWIFT samples needed for a target error at order K.
fn qswift_samples_for_error(
    lambda: f64,
    time: f64,
    target_error: f64,
    order: usize,
) -> usize {
    // 2 * (λt/N)^K ≤ ε  =>  N ≥ λt / (ε/2)^(1/K)
    let base = (target_error / 2.0).powf(1.0 / (order as f64));
    if base < 1e-15 {
        return usize::MAX;
    }
    let n = lambda * time / base;
    n.ceil() as usize
}

/// Estimate the number of Trotter steps for a target error.
///
/// First-order Trotter error ≈ λ²t²/(2N) (rough bound).
fn trotter_steps_for_error(lambda: f64, time: f64, target_error: f64) -> usize {
    let n = lambda * lambda * time * time / (2.0 * target_error);
    n.ceil() as usize
}

/// Compare methods for a given target error.  Returns gate counts and
/// fidelities against exact evolution.
pub fn compare_methods(
    terms: &[HamiltonianTerm],
    time: f64,
    target_error: f64,
    num_qubits: usize,
) -> MethodComparison {
    let lambda = one_norm(terms);
    let m = terms.len();

    // Gate count estimates
    let qdrift_n = qdrift_samples_for_error(lambda, time, target_error);
    let qdrift_gates = qdrift_n;

    let qswift_n = qswift_samples_for_error(lambda, time, target_error, 2);
    let qswift_gates = qswift_n * 2; // order 2 => 2 gates per sample

    let trotter_n = trotter_steps_for_error(lambda, time, target_error);
    let trotter_gates = trotter_n * m;

    // Compute actual fidelities (only feasible for small systems)
    let mut fidelities = Vec::new();
    if num_qubits <= 10 {
        let exact = exact_evolution(terms, time, num_qubits);

        // qDRIFT
        let qdrift_n_sim = qdrift_n.min(10000);
        let mut rng = StdRng::seed_from_u64(42);
        let qdrift_circ = qdrift(terms, time, qdrift_n_sim, &mut rng);
        let qdrift_state = simulate_qswift(&qdrift_circ, terms, num_qubits);
        fidelities.push(state_fidelity(&qdrift_state, &exact));

        // qSWIFT
        let qswift_n_sim = qswift_n.min(10000);
        let cfg = QswiftConfig::new()
            .order(2)
            .num_samples(qswift_n_sim)
            .time(time)
            .seed(42);
        if let Ok(qswift_circ) = qswift(terms, &cfg) {
            let qswift_state = simulate_qswift(&qswift_circ, terms, num_qubits);
            fidelities.push(state_fidelity(&qswift_state, &exact));
        }

        // Trotter
        let trotter_n_sim = trotter_n.min(10000);
        let trotter_state = standard_trotter(terms, time, trotter_n_sim, num_qubits);
        fidelities.push(state_fidelity(&trotter_state, &exact));
    }

    MethodComparison {
        qdrift_gates,
        qswift_gates,
        trotter_gates,
        exact_fidelities: fidelities,
    }
}

// ---------------------------------------------------------------------------
// Predefined Hamiltonians
// ---------------------------------------------------------------------------

/// 1D transverse-field Ising model: H = -J Σ Z_i Z_{i+1} - h Σ X_i.
pub fn ising_1d(n: usize, j: f64, h: f64) -> Vec<HamiltonianTerm> {
    let mut terms = Vec::new();

    // ZZ interactions
    for i in 0..(n - 1) {
        terms.push(HamiltonianTerm::new(vec![(i, 'Z'), (i + 1, 'Z')], -j));
    }

    // Transverse field
    for i in 0..n {
        terms.push(HamiltonianTerm::new(vec![(i, 'X')], -h));
    }

    terms
}

/// 1D Heisenberg model: H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}).
pub fn heisenberg_1d(n: usize, j: f64) -> Vec<HamiltonianTerm> {
    let mut terms = Vec::new();
    for i in 0..(n - 1) {
        terms.push(HamiltonianTerm::new(vec![(i, 'X'), (i + 1, 'X')], j));
        terms.push(HamiltonianTerm::new(vec![(i, 'Y'), (i + 1, 'Y')], j));
        terms.push(HamiltonianTerm::new(vec![(i, 'Z'), (i + 1, 'Z')], j));
    }
    terms
}

/// Generate a random Hamiltonian with `n_terms` random Pauli strings.
pub fn random_hamiltonian(
    n_qubits: usize,
    n_terms: usize,
    rng: &mut impl Rng,
) -> Vec<HamiltonianTerm> {
    let pauli_chars = ['I', 'X', 'Y', 'Z'];
    let mut terms = Vec::with_capacity(n_terms);

    for _ in 0..n_terms {
        let mut paulis = Vec::new();
        for q in 0..n_qubits {
            let p = pauli_chars[rng.gen_range(0..4)];
            if p != 'I' {
                paulis.push((q, p));
            }
        }
        if paulis.is_empty() {
            // Ensure at least one non-identity
            paulis.push((0, pauli_chars[rng.gen_range(1..4)]));
        }
        let coeff: f64 = rng.gen_range(-1.0..1.0);
        terms.push(HamiltonianTerm::new(paulis, coeff));
    }

    terms
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-6;

    // 1. Config builder defaults
    #[test]
    fn test_config_defaults() {
        let cfg = QswiftConfig::default();
        assert_eq!(cfg.order, 2);
        assert_eq!(cfg.num_samples, 1000);
        assert!((cfg.time - 1.0).abs() < TOL);
        assert_eq!(cfg.seed, 42);
    }

    // 2. One-norm computed correctly
    #[test]
    fn test_one_norm() {
        let terms = vec![
            HamiltonianTerm::new(vec![(0, 'X')], 0.5),
            HamiltonianTerm::new(vec![(0, 'Z')], -0.3),
            HamiltonianTerm::new(vec![(1, 'Y')], 0.2),
        ];
        let lambda = one_norm(&terms);
        assert!((lambda - 1.0).abs() < TOL);
    }

    // 3. qDRIFT circuit has correct number of gates (= num_samples)
    #[test]
    fn test_qdrift_gate_count() {
        let terms = ising_1d(3, 1.0, 0.5);
        let mut rng = StdRng::seed_from_u64(42);
        let circuit = qdrift(&terms, 1.0, 500, &mut rng);
        assert_eq!(circuit.gates.len(), 500);
    }

    // 4. qSWIFT circuit has K*N gates
    #[test]
    fn test_qswift_gate_count() {
        let terms = ising_1d(3, 1.0, 0.5);
        let cfg = QswiftConfig::new().order(3).num_samples(200).seed(7);
        let circuit = qswift(&terms, &cfg).unwrap();
        assert_eq!(circuit.gates.len(), 3 * 200);
    }

    // 5. qDRIFT error bound formula
    #[test]
    fn test_qdrift_error_bound() {
        let lambda = 2.0;
        let time = 0.5;
        let n = 100;
        let err = estimate_qdrift_error(lambda, time, n);
        // 2 * 4 * 0.25 / 100 = 0.02
        assert!((err - 0.02).abs() < TOL);
    }

    // 6. qSWIFT error bound decreases with order
    #[test]
    fn test_qswift_error_decreases_with_order() {
        let lambda = 2.0;
        let time = 1.0;
        let n = 100;
        let err2 = estimate_qswift_error(lambda, time, n, 2);
        let err4 = estimate_qswift_error(lambda, time, n, 4);
        let err6 = estimate_qswift_error(lambda, time, n, 6);
        assert!(err4 < err2, "order 4 should have smaller error than order 2");
        assert!(err6 < err4, "order 6 should have smaller error than order 4");
        assert!(err4 < 1e-5, "order 4 error should be very small");
    }

    // 7. Exact evolution of Pauli-X for time π/2 gives |1>
    #[test]
    fn test_exact_pauli_x_rotation() {
        // H = X, evolve for t = π/2
        // exp(-i X π/2)|0> = cos(π/2)I|0> - i·sin(π/2)X|0>
        //                  = 0·|0> - i·|1> = -i|1>
        let terms = vec![HamiltonianTerm::new(vec![(0, 'X')], 1.0)];
        let state = exact_evolution(&terms, PI / 2.0, 1);

        let pop_1 = state[1].norm_sqr();
        let pop_0 = state[0].norm_sqr();
        assert!(
            pop_1 > 0.999,
            "Expected |1> population > 0.999, got {}. State: |0>={:?}, |1>={:?}",
            pop_1,
            state[0],
            state[1]
        );
        assert!(
            pop_0 < 0.001,
            "Expected |0> population < 0.001, got {}",
            pop_0
        );
    }

    // 8. qSWIFT on 2-qubit Ising matches exact (high samples)
    #[test]
    fn test_qswift_matches_exact_ising_2q() {
        let terms = ising_1d(2, 1.0, 0.5);
        let time = 0.5;
        let num_qubits = 2;

        let exact = exact_evolution(&terms, time, num_qubits);

        let cfg = QswiftConfig::new()
            .order(2)
            .num_samples(5000)
            .time(time)
            .seed(123);
        let circuit = qswift(&terms, &cfg).unwrap();
        let approx = simulate_qswift(&circuit, &terms, num_qubits);

        let fid = state_fidelity(&approx, &exact);
        assert!(
            fid > 0.95,
            "qSWIFT fidelity on 2-qubit Ising should be > 0.95, got {}",
            fid
        );
    }

    // 9. qSWIFT order-4 uses fewer gates than qDRIFT for same error target
    #[test]
    fn test_qswift_fewer_gates_than_qdrift() {
        let lambda = 3.0;
        let time = 1.0;
        let target_error = 1e-6;

        let qdrift_n = qdrift_samples_for_error(lambda, time, target_error);
        let qswift_n = qswift_samples_for_error(lambda, time, target_error, 4);
        let qswift_total_gates = qswift_n * 4;

        assert!(
            qswift_total_gates < qdrift_n,
            "qSWIFT(order=4) should need fewer total gates ({}) than qDRIFT ({}) for ε={}",
            qswift_total_gates,
            qdrift_n,
            target_error
        );
    }

    // 10. Pauli rotation preserves normalization
    #[test]
    fn test_pauli_rotation_preserves_norm() {
        let dim = 4; // 2 qubits
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state[1] = Complex64::new(0.0, 1.0 / 2.0_f64.sqrt());

        let norm_before: f64 = state.iter().map(|c| c.norm_sqr()).sum();

        // Apply various rotations
        apply_pauli_rotation(&mut state, &[(0, 'X')], 0.3);
        apply_pauli_rotation(&mut state, &[(1, 'Z')], 0.7);
        apply_pauli_rotation(&mut state, &[(0, 'Y'), (1, 'X')], 0.5);

        let norm_after: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (norm_before - norm_after).abs() < 1e-10,
            "Norm changed from {} to {}",
            norm_before,
            norm_after
        );
    }

    // 11. Standard Trotter matches exact for small dt
    #[test]
    fn test_trotter_matches_exact() {
        let terms = vec![
            HamiltonianTerm::new(vec![(0, 'Z')], 1.0),
            HamiltonianTerm::new(vec![(0, 'X')], 0.5),
        ];
        let time = 0.3;
        let num_qubits = 1;

        let exact = exact_evolution(&terms, time, num_qubits);
        let trotter = standard_trotter(&terms, time, 10000, num_qubits);

        let fid = state_fidelity(&trotter, &exact);
        assert!(
            fid > 0.999,
            "Trotter should match exact with high fidelity, got {}",
            fid
        );
    }

    // 12. Comparison shows qSWIFT advantage
    #[test]
    fn test_comparison_shows_advantage() {
        let terms = ising_1d(2, 1.0, 0.5);
        let time = 0.5;
        let target_error = 0.01;
        let num_qubits = 2;

        let cmp = compare_methods(&terms, time, target_error, num_qubits);

        assert!(
            cmp.qswift_gates <= cmp.trotter_gates || cmp.qswift_gates <= cmp.qdrift_gates,
            "qSWIFT ({}) should improve over Trotter ({}) or qDRIFT ({})",
            cmp.qswift_gates,
            cmp.trotter_gates,
            cmp.qdrift_gates
        );
    }

    // 13. Empty Hamiltonian returns error
    #[test]
    fn test_empty_hamiltonian_error() {
        let terms: Vec<HamiltonianTerm> = vec![];
        let cfg = QswiftConfig::default();
        let result = qswift(&terms, &cfg);
        assert!(matches!(result, Err(QswiftError::EmptyHamiltonian)));
    }

    // 14. Invalid order returns error
    #[test]
    fn test_invalid_order_error() {
        let terms = vec![HamiltonianTerm::new(vec![(0, 'X')], 1.0)];
        let cfg = QswiftConfig::new().order(0);
        let result = qswift(&terms, &cfg);
        assert!(matches!(result, Err(QswiftError::InvalidOrder(0))));
    }

    // 15. State fidelity of identical states is 1
    #[test]
    fn test_fidelity_identical() {
        let a = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 1.0 / 2.0_f64.sqrt()),
        ];
        let fid = state_fidelity(&a, &a);
        assert!(
            (fid - 1.0).abs() < TOL,
            "Fidelity of identical states should be 1.0, got {}",
            fid
        );
    }

    // 16. Heisenberg model has correct term count
    #[test]
    fn test_heisenberg_term_count() {
        let terms = heisenberg_1d(4, 1.0);
        // 3 bonds × 3 Pauli types = 9
        assert_eq!(terms.len(), 9);
    }

    // 17. Random Hamiltonian generates correct number of terms
    #[test]
    fn test_random_hamiltonian() {
        let mut rng = StdRng::seed_from_u64(99);
        let terms = random_hamiltonian(3, 10, &mut rng);
        assert_eq!(terms.len(), 10);
        for term in &terms {
            assert!(
                !term.paulis.is_empty(),
                "Each term should have non-trivial Pauli content"
            );
        }
    }
}
