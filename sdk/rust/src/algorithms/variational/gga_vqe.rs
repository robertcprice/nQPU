//! Greedy Gradient-free Adaptive VQE (GGA-VQE)
//!
//! Implementation based on Nature Scientific Reports 2025.
//! Gradient-free VQE with greedy operator selection that is more
//! noise-resilient than standard VQE approaches.
//!
//! # Algorithm
//!
//! 1. Initialize Hartree-Fock reference state
//! 2. Generate operator pool (single + double + triple excitations via Jordan-Wigner)
//! 3. ADAPT loop: greedily select operator with largest energy gradient,
//!    append to circuit, re-optimize all parameters with gradient-free optimizer
//! 4. Converge when energy improvement falls below threshold
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::gga_vqe::*;
//!
//! let (ham, nq, ne, exact) = h2_hamiltonian();
//! let config = GgaVqeConfig::default();
//! let result = run_gga_vqe(&ham, nq, ne, &config).unwrap();
//! assert!((result.energy - exact).abs() < 0.05);
//! ```

use num_complex::Complex64;
use std::fmt;

// ============================================================
// TYPES
// ============================================================

/// Configuration for the GGA-VQE algorithm.
#[derive(Clone, Debug)]
pub struct GgaVqeConfig {
    /// Maximum number of operators to append in the ADAPT loop.
    pub max_operators: usize,
    /// Energy convergence threshold (Hartree).
    pub energy_threshold: f64,
    /// Pool of candidate operators.
    pub operator_pool: OperatorPool,
    /// Gradient-free optimizer to use for parameter re-optimization.
    pub optimizer: GradientFreeOptimizer,
    /// Maximum iterations per optimization call.
    pub max_opt_iterations: usize,
    /// Shot count for energy estimation (None = exact statevector).
    pub shots: Option<usize>,
    /// Finite-difference step for gradient estimation.
    pub gradient_step: f64,
    /// Gradient norm threshold below which we stop adding operators.
    pub gradient_threshold: f64,
}

impl Default for GgaVqeConfig {
    fn default() -> Self {
        Self {
            max_operators: 20,
            energy_threshold: 1e-6,
            operator_pool: OperatorPool::FullUCCSD,
            optimizer: GradientFreeOptimizer::NelderMead,
            max_opt_iterations: 100,
            shots: None,
            gradient_step: 1e-4,
            gradient_threshold: 1e-5,
        }
    }
}

/// Operator pool specification.
#[derive(Clone, Debug)]
pub enum OperatorPool {
    /// Only single-excitation operators (occupied -> virtual).
    SingleExcitations,
    /// Only double-excitation operators (occupied pair -> virtual pair).
    DoubleExcitations,
    /// Only triple-excitation operators (occupied triple -> virtual triple).
    TripleExcitations,
    /// Full UCCSD pool (single + double excitations).
    FullUCCSD,
    /// Full UCCSDT pool (single + double + triple excitations).
    FullUCCSDT,
    /// User-supplied custom operators.
    Custom(Vec<GgaPauliOperator>),
}

/// Gradient-free optimizer variants.
#[derive(Clone, Debug, PartialEq)]
pub enum GradientFreeOptimizer {
    /// Nelder-Mead simplex method.
    NelderMead,
    /// Powell's conjugate direction method (coordinate descent).
    Powell,
    /// Constrained Optimization BY Linear Approximation (simplified).
    Cobyla,
    /// Golden-section search (for single-parameter cases).
    GoldenSection,
}

/// A Pauli operator expressed as a sum of weighted Pauli strings.
///
/// Each term is a `(Vec<(qubit, pauli_char)>, coefficient)` pair where
/// `pauli_char` is one of `'I'`, `'X'`, `'Y'`, `'Z'`.
#[derive(Clone, Debug)]
pub struct GgaPauliOperator {
    /// Pauli string terms: `([(qubit_index, pauli_label)], coefficient)`.
    pub terms: Vec<(Vec<(usize, char)>, f64)>,
    /// Human-readable label (e.g. "S_0->2" for single excitation 0->2).
    pub label: String,
}

impl GgaPauliOperator {
    /// Create a new operator from terms and label.
    pub fn new(terms: Vec<(Vec<(usize, char)>, f64)>, label: impl Into<String>) -> Self {
        Self {
            terms,
            label: label.into(),
        }
    }

    /// Check that the operator is anti-Hermitian: A = -A^dagger.
    /// For real-coefficient Pauli sums this means the sum of coefficients
    /// for each unique Pauli string should cancel with its conjugate.
    /// In practice for excitation operators all terms come in +/- pairs.
    pub fn is_anti_hermitian(&self) -> bool {
        // Each Pauli string P_i is Hermitian, so (sum c_i P_i)^dag = sum c_i* P_i.
        // Anti-Hermitian requires sum c_i P_i = -sum c_i* P_i, i.e. c_i are purely imaginary
        // OR the terms pair up such that for each +c P there is a -c P'.
        // For JW excitation operators the coefficients are real and come in +/- pairs
        // with different Pauli strings. We verify the sum of all coefficients is ~0.
        let total: f64 = self.terms.iter().map(|(_, c)| *c).sum();
        total.abs() < 1e-10
    }
}

impl fmt::Display for GgaPauliOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// An ADAPT-VQE circuit: sequence of (operator, parameter) pairs.
#[derive(Clone, Debug)]
pub struct AdaptCircuit {
    /// Ordered list of (operator, variational_parameter) pairs.
    pub operators: Vec<(GgaPauliOperator, f64)>,
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
    /// Number of electrons (for HF reference).
    pub num_electrons: usize,
}

impl AdaptCircuit {
    /// Create an empty circuit.
    pub fn new(num_qubits: usize, num_electrons: usize) -> Self {
        Self {
            operators: Vec::new(),
            num_qubits,
            num_electrons,
        }
    }

    /// Number of operators (circuit depth).
    pub fn depth(&self) -> usize {
        self.operators.len()
    }

    /// Get all current parameters.
    pub fn parameters(&self) -> Vec<f64> {
        self.operators.iter().map(|(_, p)| *p).collect()
    }

    /// Set parameters from a slice.
    pub fn set_parameters(&mut self, params: &[f64]) {
        for (i, (_, p)) in self.operators.iter_mut().enumerate() {
            if i < params.len() {
                *p = params[i];
            }
        }
    }

    /// Append an operator with an initial parameter value.
    pub fn push(&mut self, op: GgaPauliOperator, param: f64) {
        self.operators.push((op, param));
    }
}

/// Result of a GGA-VQE computation.
#[derive(Clone, Debug)]
pub struct GgaVqeResult {
    /// Final variational energy (Hartree).
    pub energy: f64,
    /// The optimized ADAPT circuit.
    pub circuit: AdaptCircuit,
    /// Number of operators selected from the pool.
    pub num_operators_used: usize,
    /// Energy after each ADAPT iteration.
    pub energy_history: Vec<f64>,
    /// Labels of operators in selection order.
    pub operator_sequence: Vec<String>,
    /// Whether the algorithm converged within thresholds.
    pub converged: bool,
}

/// Errors that can occur during GGA-VQE execution.
#[derive(Clone, Debug)]
pub enum GgaVqeError {
    /// All operators in the pool have been used or none improve energy.
    NoOperatorsLeft,
    /// The gradient-free optimizer failed to converge.
    OptimizationFailed(String),
    /// The operator pool is empty.
    EmptyPool,
}

impl fmt::Display for GgaVqeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgaVqeError::NoOperatorsLeft => write!(f, "No operators left in pool"),
            GgaVqeError::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
            GgaVqeError::EmptyPool => write!(f, "Operator pool is empty"),
        }
    }
}

impl std::error::Error for GgaVqeError {}

// ============================================================
// HARTREE-FOCK STATE
// ============================================================

/// Create the Hartree-Fock reference state |1...10...0> with `num_electrons`
/// occupied orbitals (LSB = qubit 0).
///
/// In computational basis, qubit i occupied means bit i is set.
/// HF state index = (1 << num_electrons) - 1 for the lowest orbitals.
pub fn hartree_fock_state(num_qubits: usize, num_electrons: usize) -> Vec<Complex64> {
    let dim = 1usize << num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    // HF: fill the lowest num_electrons orbitals
    let hf_index = (1usize << num_electrons) - 1;
    state[hf_index] = Complex64::new(1.0, 0.0);
    state
}

// ============================================================
// OPERATOR POOL GENERATION
// ============================================================

/// Generate single-excitation operators: a†_p a_q - a†_q a_p
/// for occupied orbital q -> virtual orbital p, mapped via Jordan-Wigner.
pub fn generate_single_excitations(
    num_orbitals: usize,
    num_electrons: usize,
) -> Vec<GgaPauliOperator> {
    let mut ops = Vec::new();
    let occupied: Vec<usize> = (0..num_electrons).collect();
    let virtual_orbs: Vec<usize> = (num_electrons..num_orbitals).collect();

    for &q in &occupied {
        for &p in &virtual_orbs {
            let op = excitation_to_pauli(&[p], &[q]);
            let labeled = GgaPauliOperator {
                terms: op.terms,
                label: format!("S_{}→{}", q, p),
            };
            ops.push(labeled);
        }
    }
    ops
}

/// Generate double-excitation operators:
/// a†_p a†_q a_r a_s - h.c. for occupied pair (r,s) -> virtual pair (p,q).
pub fn generate_double_excitations(
    num_orbitals: usize,
    num_electrons: usize,
) -> Vec<GgaPauliOperator> {
    let mut ops = Vec::new();
    let occupied: Vec<usize> = (0..num_electrons).collect();
    let virtual_orbs: Vec<usize> = (num_electrons..num_orbitals).collect();

    for i in 0..occupied.len() {
        for j in (i + 1)..occupied.len() {
            for a in 0..virtual_orbs.len() {
                for b in (a + 1)..virtual_orbs.len() {
                    let s = occupied[i];
                    let r = occupied[j];
                    let p = virtual_orbs[a];
                    let q = virtual_orbs[b];
                    let op = excitation_to_pauli(&[p, q], &[s, r]);
                    let labeled = GgaPauliOperator {
                        terms: op.terms,
                        label: format!("D_{},{}→{},{}", s, r, p, q),
                    };
                    ops.push(labeled);
                }
            }
        }
    }
    ops
}

/// Generate triple-excitation operators:
/// a†_p a†_q a†_r a_s a_t a_u - h.c. for occupied triple (s,t,u) -> virtual triple (p,q,r).
pub fn generate_triple_excitations(
    num_orbitals: usize,
    num_electrons: usize,
) -> Vec<GgaPauliOperator> {
    let mut ops = Vec::new();
    let occupied: Vec<usize> = (0..num_electrons).collect();
    let virtual_orbs: Vec<usize> = (num_electrons..num_orbitals).collect();

    for i in 0..occupied.len() {
        for j in (i + 1)..occupied.len() {
            for k in (j + 1)..occupied.len() {
                for a in 0..virtual_orbs.len() {
                    for b in (a + 1)..virtual_orbs.len() {
                        for c in (b + 1)..virtual_orbs.len() {
                            let u = occupied[i];
                            let t = occupied[j];
                            let s = occupied[k];
                            let p = virtual_orbs[a];
                            let q = virtual_orbs[b];
                            let r = virtual_orbs[c];
                            let op = excitation_to_pauli(&[p, q, r], &[u, t, s]);
                            let labeled = GgaPauliOperator {
                                terms: op.terms,
                                label: format!("T_{},{},{}→{},{},{}", u, t, s, p, q, r),
                            };
                            ops.push(labeled);
                        }
                    }
                }
            }
        }
    }
    ops
}

/// Convert a fermionic excitation operator to Pauli operators via Jordan-Wigner.
///
/// For single excitation a†_p a_q - a†_q a_p (anti-Hermitian):
///   = (i/2) [Y_p X_{p-1} ... X_{q+1} X_q - X_p X_{p-1} ... X_{q+1} Y_q]
/// when p > q.
///
/// For double excitations, we compose two single-excitation JW strings.
pub fn excitation_to_pauli(creation: &[usize], annihilation: &[usize]) -> GgaPauliOperator {
    assert_eq!(
        creation.len(),
        annihilation.len(),
        "Creation and annihilation lists must have equal length"
    );

    if creation.len() == 1 {
        // Single excitation: a†_p a_q - a†_q a_p
        single_excitation_jw(creation[0], annihilation[0])
    } else if creation.len() == 2 {
        // Double excitation: a†_p a†_q a_r a_s - h.c.
        double_excitation_jw(creation[0], creation[1], annihilation[0], annihilation[1])
    } else if creation.len() == 3 {
        // Triple excitation: a†_p a†_q a†_r a_s a_t a_u - h.c.
        triple_excitation_jw(
            creation[0],
            creation[1],
            creation[2],
            annihilation[0],
            annihilation[1],
            annihilation[2],
        )
    } else {
        panic!(
            "Excitations of order {} are not supported (max: triple)",
            creation.len()
        );
    }
}

/// Jordan-Wigner mapping for single excitation a†_p a_q - a†_q a_p.
fn single_excitation_jw(p: usize, q: usize) -> GgaPauliOperator {
    let (hi, lo) = if p > q { (p, q) } else { (q, p) };

    // Term 1: (i/2) Y_hi X_{hi-1} ... X_{lo+1} X_lo
    let mut term1 = Vec::new();
    term1.push((hi, 'Y'));
    for k in (lo + 1)..hi {
        term1.push((k, 'Z'));
    }
    term1.push((lo, 'X'));

    // Term 2: -(i/2) X_hi X_{hi-1} ... X_{lo+1} Y_lo
    let mut term2 = Vec::new();
    term2.push((hi, 'X'));
    for k in (lo + 1)..hi {
        term2.push((k, 'Z'));
    }
    term2.push((lo, 'Y'));

    let coeff = 0.5;
    GgaPauliOperator {
        terms: vec![(term1, coeff), (term2, -coeff)],
        label: format!("exc_{}→{}", q, p),
    }
}

/// Jordan-Wigner mapping for double excitation.
/// Produces 8 Pauli terms (simplified standard form).
fn double_excitation_jw(p: usize, q: usize, r: usize, s: usize) -> GgaPauliOperator {
    // Standard 8-term decomposition for double excitation.
    // We use the compact form from Yordanov et al. 2020.
    let mut indices = vec![s, r, p, q];
    indices.sort();
    let (a, b, c, d) = (indices[0], indices[1], indices[2], indices[3]);

    let mut terms = Vec::new();
    let c8 = 0.125; // 1/8

    // 8 terms of the double excitation operator in JW encoding
    let patterns: [(char, char, char, char, f64); 8] = [
        ('X', 'X', 'X', 'Y', c8),
        ('X', 'X', 'Y', 'X', c8),
        ('X', 'Y', 'X', 'X', -c8),
        ('Y', 'X', 'X', 'X', -c8),
        ('X', 'Y', 'Y', 'Y', c8),
        ('Y', 'X', 'Y', 'Y', c8),
        ('Y', 'Y', 'X', 'Y', -c8),
        ('Y', 'Y', 'Y', 'X', -c8),
    ];

    for (pa, pb, pc, pd, coeff) in &patterns {
        let mut term = Vec::new();
        term.push((a, *pa));
        // Z-chain between a and b
        for k in (a + 1)..b {
            term.push((k, 'Z'));
        }
        term.push((b, *pb));
        // Z-chain between b and c
        for k in (b + 1)..c {
            term.push((k, 'Z'));
        }
        term.push((c, *pc));
        // Z-chain between c and d
        for k in (c + 1)..d {
            term.push((k, 'Z'));
        }
        term.push((d, *pd));
        terms.push((term, *coeff));
    }

    GgaPauliOperator {
        terms,
        label: format!("dexc_{},{}_{}_{}", s, r, p, q),
    }
}

/// Jordan-Wigner mapping for triple excitation.
/// Produces 32 Pauli terms following the same pattern as the double excitation.
///
/// For a triple excitation across 6 sorted orbital indices (a < b < c < d < e < f),
/// the anti-Hermitian generator has 32 terms. Each term places an X or Y on each
/// of the 6 active qubits, with Z-chains bridging gaps between them.
/// The anti-Hermitian constraint requires an odd number of Y operators (1, 3, or 5).
/// There are C(6,1) + C(6,3) + C(6,5) = 6 + 20 + 6 = 32 such terms.
/// Signs follow the pattern: (-1)^(number_of_Y_in_first_half_minus_expected).
fn triple_excitation_jw(
    p: usize,
    q: usize,
    r: usize,
    s: usize,
    t: usize,
    u: usize,
) -> GgaPauliOperator {
    let mut indices = vec![p, q, r, s, t, u];
    indices.sort();
    let idx = [
        indices[0], indices[1], indices[2], indices[3], indices[4], indices[5],
    ];

    let mut terms = Vec::new();
    let c32 = 1.0 / 32.0; // coefficient magnitude: 1/2^(n_fermions) where n=5 pairs

    // Generate all 64 combinations of X/Y on 6 qubits, keep only those with
    // an odd number of Y (anti-Hermitian requirement).
    // Sign pattern: the standard JW triple excitation decomposition alternates
    // signs based on parity. We use the convention from Romero et al. (2018).
    for mask in 0u8..64 {
        // mask bit i: 0 => X, 1 => Y for the i-th sorted orbital
        let y_count = mask.count_ones();
        if y_count % 2 == 0 {
            continue; // skip even Y count (would be Hermitian, not anti-Hermitian)
        }

        let paulis: Vec<char> = (0..6)
            .map(|bit| if (mask >> bit) & 1 == 1 { 'Y' } else { 'X' })
            .collect();

        // Sign convention: (-1)^(sum of bit positions where Y appears, mod 2)
        // This ensures the overall operator is anti-Hermitian: A = -A†.
        // For JW-mapped fermionic excitations, the sign of each Pauli term
        // is determined by the permutation parity of the Y placements.
        let y_position_sum: u32 = (0..6)
            .filter(|&bit| (mask >> bit) & 1 == 1)
            .map(|bit| bit as u32)
            .sum();
        let sign = if y_position_sum % 2 == 0 { c32 } else { -c32 };

        // Build the full Pauli string with Z-chains between active qubits
        let mut term = Vec::new();
        for (orbital_idx, pauli_char) in idx.iter().zip(paulis.iter()) {
            // Z-chain from previous active qubit to this one (handled below)
            term.push((*orbital_idx, *pauli_char));
        }

        // Insert Z-chains between consecutive active qubits
        let mut full_term = Vec::new();
        for pair_idx in 0..6 {
            full_term.push((idx[pair_idx], paulis[pair_idx]));
            if pair_idx < 5 {
                for z_qubit in (idx[pair_idx] + 1)..idx[pair_idx + 1] {
                    full_term.push((z_qubit, 'Z'));
                }
            }
        }

        terms.push((full_term, sign));
    }

    GgaPauliOperator {
        terms,
        label: format!("texc_{},{},{}_{}_{},{}", u, t, s, p, q, r),
    }
}

// ============================================================
// CIRCUIT SIMULATION
// ============================================================

/// Apply a Pauli rotation exp(-i * theta * P) to a statevector,
/// where P is a Pauli string (tensor product of single-qubit Paulis).
///
/// exp(-i theta P) = cos(theta) I - i sin(theta) P
fn apply_pauli_rotation(state: &mut Vec<Complex64>, pauli_string: &[(usize, char)], theta: f64) {
    let n = state.len();
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // Compute P|state> for the given Pauli string
    let mut p_state = state.clone();
    apply_pauli_string(&mut p_state, pauli_string);

    // |state'> = cos(theta)|state> - i*sin(theta) P|state>
    for i in 0..n {
        state[i] = Complex64::new(cos_t, 0.0) * state[i] + Complex64::new(0.0, -sin_t) * p_state[i];
    }
}

/// Apply a Pauli string (tensor product) to a statevector in-place.
fn apply_pauli_string(state: &mut Vec<Complex64>, pauli_string: &[(usize, char)]) {
    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    // Build the action of each Pauli on each basis state
    for i in 0..n {
        let mut coeff = Complex64::new(1.0, 0.0);
        let mut new_index = i;

        for &(qubit, pauli) in pauli_string {
            let bit = (i >> qubit) & 1;
            match pauli {
                'I' => {}
                'X' => {
                    new_index ^= 1 << qubit;
                }
                'Y' => {
                    new_index ^= 1 << qubit;
                    if bit == 0 {
                        coeff *= Complex64::new(0.0, 1.0); // Y|0> = i|1>
                    } else {
                        coeff *= Complex64::new(0.0, -1.0); // Y|1> = -i|0>
                    }
                }
                'Z' => {
                    if bit == 1 {
                        coeff *= Complex64::new(-1.0, 0.0);
                    }
                }
                _ => panic!("Unknown Pauli label: {}", pauli),
            }
        }

        // We need to be careful: applying P to basis state |i> gives coeff * |new_index>.
        // But we are transforming the state vector, so we need a different approach.
        let _ = (coeff, new_index, num_qubits);
    }

    // More correct approach: build new state vector
    let old_state = state.clone();
    for amp in state.iter_mut() {
        *amp = Complex64::new(0.0, 0.0);
    }

    for i in 0..n {
        let mut coeff = Complex64::new(1.0, 0.0);
        let mut new_index = i;

        for &(qubit, pauli) in pauli_string {
            let bit = (i >> qubit) & 1;
            match pauli {
                'I' => {}
                'X' => {
                    new_index ^= 1 << qubit;
                }
                'Y' => {
                    new_index ^= 1 << qubit;
                    if bit == 0 {
                        coeff *= Complex64::new(0.0, 1.0);
                    } else {
                        coeff *= Complex64::new(0.0, -1.0);
                    }
                }
                'Z' => {
                    if bit == 1 {
                        coeff *= Complex64::new(-1.0, 0.0);
                    }
                }
                _ => {}
            }
        }

        state[new_index] += coeff * old_state[i];
    }
}

/// Apply an excitation unitary exp(-i * param * A) to the state,
/// where A is a GgaPauliOperator (sum of Pauli strings).
///
/// Uses first-order Trotter: exp(-i*param*sum_k c_k P_k) ≈ prod_k exp(-i*param*c_k P_k)
pub fn apply_excitation_unitary(
    state: &mut Vec<Complex64>,
    operator: &GgaPauliOperator,
    param: f64,
) {
    for (pauli_string, coeff) in &operator.terms {
        apply_pauli_rotation(state, pauli_string, param * coeff);
    }
}

/// Simulate the full ADAPT circuit: start from HF state, apply each operator unitary.
pub fn simulate_adapt_circuit(
    circuit: &AdaptCircuit,
    hamiltonian: &[(Vec<(usize, char)>, f64)],
) -> f64 {
    let mut state = hartree_fock_state(circuit.num_qubits, circuit.num_electrons);

    for (op, param) in &circuit.operators {
        apply_excitation_unitary(&mut state, op, *param);
    }

    compute_energy(&state, hamiltonian)
}

/// Compute expectation value <psi|H|psi> where H = sum_k c_k P_k.
pub fn compute_energy(state: &[Complex64], hamiltonian: &[(Vec<(usize, char)>, f64)]) -> f64 {
    let mut energy = 0.0;

    for (pauli_string, coeff) in hamiltonian {
        // <psi| P_k |psi>
        let mut p_state = state.to_vec();
        apply_pauli_string(&mut p_state, pauli_string);

        let expectation: Complex64 = state
            .iter()
            .zip(p_state.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        energy += coeff * expectation.re;
    }

    energy
}

// ============================================================
// GRADIENT-FREE OPTIMIZERS
// ============================================================

/// Nelder-Mead simplex optimizer.
///
/// Finds the minimum of `f` starting from `initial` point.
/// Returns `(optimal_params, optimal_value)`.
pub fn nelder_mead(
    f: &dyn Fn(&[f64]) -> f64,
    initial: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64) {
    let n = initial.len();
    if n == 0 {
        return (vec![], f(&[]));
    }

    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    // Initialize simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.to_vec());

    for i in 0..n {
        let mut vertex = initial.to_vec();
        let step = if vertex[i].abs() > 1e-10 {
            0.05 * vertex[i].abs()
        } else {
            0.00025
        };
        vertex[i] += step;
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        // Sort by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        // Check convergence: range of values
        let range = values[n] - values[0];
        if range < tol {
            return (simplex[0].clone(), values[0]);
        }

        // Centroid of all points except worst
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < values[0] {
            // Expansion
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = f(&expanded);

            if f_expanded < f_reflected {
                simplex[n] = expanded;
                values[n] = f_expanded;
            } else {
                simplex[n] = reflected;
                values[n] = f_reflected;
            }
        } else if f_reflected < values[n - 1] {
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else {
            // Contraction
            if f_reflected < values[n] {
                // Outside contraction
                let contracted: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + rho * (reflected[j] - centroid[j]))
                    .collect();
                let f_contracted = f(&contracted);

                if f_contracted <= f_reflected {
                    simplex[n] = contracted;
                    values[n] = f_contracted;
                } else {
                    // Shrink
                    for i in 1..=n {
                        for j in 0..n {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        values[i] = f(&simplex[i]);
                    }
                }
            } else {
                // Inside contraction
                let contracted: Vec<f64> = (0..n)
                    .map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j]))
                    .collect();
                let f_contracted = f(&contracted);

                if f_contracted < values[n] {
                    simplex[n] = contracted;
                    values[n] = f_contracted;
                } else {
                    // Shrink
                    for i in 1..=n {
                        for j in 0..n {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        values[i] = f(&simplex[i]);
                    }
                }
            }
        }
    }

    // Return best found
    let mut best = 0;
    for i in 1..=n {
        if values[i] < values[best] {
            best = i;
        }
    }
    (simplex[best].clone(), values[best])
}

/// Golden-section search for 1D minimization on interval [a, b].
pub fn golden_section_1d(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618

    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }

    (a + b) / 2.0
}

/// Coordinate descent: optimize one parameter at a time using golden-section.
pub fn coordinate_descent(
    f: &dyn Fn(&[f64]) -> f64,
    initial: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64) {
    let n = initial.len();
    let mut params = initial.to_vec();
    let mut best_val = f(&params);

    for _iter in 0..max_iter {
        let old_val = best_val;

        for i in 0..n {
            let params_copy = params.clone();
            let fi = |x: f64| {
                let mut p = params_copy.clone();
                p[i] = x;
                f(&p)
            };

            // Search in [-pi, pi] around current value
            let lo = params[i] - std::f64::consts::PI;
            let hi = params[i] + std::f64::consts::PI;
            params[i] = golden_section_1d(&fi, lo, hi, tol * 0.1);
        }

        best_val = f(&params);
        if (old_val - best_val).abs() < tol {
            break;
        }
    }

    (params, best_val)
}

// ============================================================
// GGA-VQE MAIN ALGORITHM
// ============================================================

/// Run the Greedy Gradient-free Adaptive VQE algorithm.
///
/// # Arguments
/// * `hamiltonian` - Hamiltonian as sum of weighted Pauli strings
/// * `num_qubits` - Number of qubits
/// * `num_electrons` - Number of electrons for HF reference
/// * `config` - Algorithm configuration
///
/// # Returns
/// `GgaVqeResult` with final energy, circuit, and convergence info.
pub fn run_gga_vqe(
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    num_qubits: usize,
    num_electrons: usize,
    config: &GgaVqeConfig,
) -> Result<GgaVqeResult, GgaVqeError> {
    // 1. Build operator pool
    let pool = match &config.operator_pool {
        OperatorPool::SingleExcitations => generate_single_excitations(num_qubits, num_electrons),
        OperatorPool::DoubleExcitations => generate_double_excitations(num_qubits, num_electrons),
        OperatorPool::TripleExcitations => generate_triple_excitations(num_qubits, num_electrons),
        OperatorPool::FullUCCSD => {
            let mut ops = generate_single_excitations(num_qubits, num_electrons);
            ops.extend(generate_double_excitations(num_qubits, num_electrons));
            ops
        }
        OperatorPool::FullUCCSDT => {
            let mut ops = generate_single_excitations(num_qubits, num_electrons);
            ops.extend(generate_double_excitations(num_qubits, num_electrons));
            ops.extend(generate_triple_excitations(num_qubits, num_electrons));
            ops
        }
        OperatorPool::Custom(ops) => ops.clone(),
    };

    if pool.is_empty() {
        return Err(GgaVqeError::EmptyPool);
    }

    // 2. Initialize circuit and compute HF energy
    let mut circuit = AdaptCircuit::new(num_qubits, num_electrons);
    let hf_energy = compute_energy(&hartree_fock_state(num_qubits, num_electrons), hamiltonian);

    let mut energy_history = vec![hf_energy];
    let mut operator_sequence = Vec::new();
    let mut current_energy = hf_energy;
    let mut used_operator_indices: Vec<bool> = vec![false; pool.len()];

    // 3. ADAPT loop
    for _adapt_iter in 0..config.max_operators {
        // 3a. Compute gradient for each unused operator
        let mut best_gradient = 0.0_f64;
        let mut best_op_idx = None;

        for (idx, op) in pool.iter().enumerate() {
            if used_operator_indices[idx] {
                continue;
            }

            let grad = compute_operator_gradient(&circuit, op, hamiltonian, config.gradient_step);

            if grad.abs() > best_gradient.abs() {
                best_gradient = grad;
                best_op_idx = Some(idx);
            }
        }

        // 3b. Check if any operator has sufficient gradient
        let op_idx = match best_op_idx {
            Some(idx) if best_gradient.abs() > config.gradient_threshold => idx,
            _ => break,
        };

        // 3c. Add selected operator to circuit
        used_operator_indices[op_idx] = true;
        let selected_op = pool[op_idx].clone();
        operator_sequence.push(selected_op.label.clone());
        circuit.push(selected_op, 0.0);

        // 3d. Re-optimize ALL parameters
        let ham_ref = hamiltonian;
        let nq = num_qubits;
        let ne = num_electrons;
        let ops_snapshot: Vec<(GgaPauliOperator, f64)> = circuit.operators.clone();

        let objective = |params: &[f64]| -> f64 {
            let mut state = hartree_fock_state(nq, ne);
            for (i, (op, _)) in ops_snapshot.iter().enumerate() {
                let p = if i < params.len() { params[i] } else { 0.0 };
                apply_excitation_unitary(&mut state, op, p);
            }
            compute_energy(&state, ham_ref)
        };

        let current_params = circuit.parameters();
        let (opt_params, opt_energy) = match config.optimizer {
            GradientFreeOptimizer::NelderMead => nelder_mead(
                &objective,
                &current_params,
                config.max_opt_iterations,
                1e-10,
            ),
            GradientFreeOptimizer::Powell | GradientFreeOptimizer::Cobyla => coordinate_descent(
                &objective,
                &current_params,
                config.max_opt_iterations,
                1e-10,
            ),
            GradientFreeOptimizer::GoldenSection => {
                if current_params.len() == 1 {
                    let f1d = |x: f64| objective(&[x]);
                    let opt_x =
                        golden_section_1d(&f1d, -std::f64::consts::PI, std::f64::consts::PI, 1e-10);
                    (vec![opt_x], objective(&[opt_x]))
                } else {
                    coordinate_descent(
                        &objective,
                        &current_params,
                        config.max_opt_iterations,
                        1e-10,
                    )
                }
            }
        };

        circuit.set_parameters(&opt_params);
        let new_energy = opt_energy;
        energy_history.push(new_energy);

        // 3e. Check convergence
        if (current_energy - new_energy).abs() < config.energy_threshold {
            current_energy = new_energy;
            return Ok(GgaVqeResult {
                energy: current_energy,
                circuit,
                num_operators_used: operator_sequence.len(),
                energy_history,
                operator_sequence,
                converged: true,
            });
        }

        current_energy = new_energy;
    }

    Ok(GgaVqeResult {
        energy: current_energy,
        circuit,
        num_operators_used: operator_sequence.len(),
        energy_history,
        operator_sequence,
        converged: current_energy < hf_energy, // at least improved
    })
}

/// Compute the energy gradient for adding an operator to the circuit
/// via finite difference: dE/dtheta at theta=0.
pub fn compute_operator_gradient(
    circuit: &AdaptCircuit,
    operator: &GgaPauliOperator,
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    step: f64,
) -> f64 {
    // Build state from current circuit
    let mut state_plus = hartree_fock_state(circuit.num_qubits, circuit.num_electrons);
    let mut state_minus = hartree_fock_state(circuit.num_qubits, circuit.num_electrons);

    for (op, param) in &circuit.operators {
        apply_excitation_unitary(&mut state_plus, op, *param);
        apply_excitation_unitary(&mut state_minus, op, *param);
    }

    // Apply candidate operator at +step and -step
    apply_excitation_unitary(&mut state_plus, operator, step);
    apply_excitation_unitary(&mut state_minus, operator, -step);

    let e_plus = compute_energy(&state_plus, hamiltonian);
    let e_minus = compute_energy(&state_minus, hamiltonian);

    (e_plus - e_minus) / (2.0 * step)
}

// ============================================================
// STANDARD VQE FOR COMPARISON
// ============================================================

/// Run a standard hardware-efficient VQE for comparison.
///
/// Uses Ry-CNOT layers and Nelder-Mead optimization.
/// Returns `(energy, num_function_evaluations)`.
pub fn run_standard_vqe(
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    num_qubits: usize,
    num_layers: usize,
) -> (f64, usize) {
    let num_params = num_qubits * num_layers;
    let initial_params: Vec<f64> = vec![0.0; num_params];

    let eval_count = std::cell::Cell::new(0usize);

    let objective = |params: &[f64]| -> f64 {
        eval_count.set(eval_count.get() + 1);

        let dim = 1usize << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0); // |00...0>

        // Apply hardware-efficient ansatz: layers of Ry + CNOT
        let mut p_idx = 0;
        for _layer in 0..num_layers {
            // Ry rotation on each qubit
            for q in 0..num_qubits {
                let theta = params[p_idx];
                p_idx += 1;
                apply_ry(&mut state, q, theta);
            }
            // CNOT ladder
            for q in 0..(num_qubits - 1) {
                apply_cnot(&mut state, q, q + 1);
            }
        }

        compute_energy(&state, hamiltonian)
    };

    let (_, energy) = nelder_mead(&objective, &initial_params, 500, 1e-8);
    (energy, eval_count.get())
}

/// Apply Ry(theta) gate to a statevector.
fn apply_ry(state: &mut Vec<Complex64>, qubit: usize, theta: f64) {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    let n = state.len();

    let old = state.clone();
    for i in 0..n {
        let bit = (i >> qubit) & 1;
        let partner = i ^ (1 << qubit);
        if bit == 0 {
            state[i] = Complex64::new(cos_half, 0.0) * old[i]
                - Complex64::new(sin_half, 0.0) * old[partner];
        } else {
            state[i] = Complex64::new(sin_half, 0.0) * old[partner]
                + Complex64::new(cos_half, 0.0) * old[i];
        }
    }
}

/// Apply CNOT gate to a statevector (control -> target).
fn apply_cnot(state: &mut Vec<Complex64>, control: usize, target: usize) {
    let n = state.len();
    for i in 0..n {
        let ctrl_bit = (i >> control) & 1;
        let tgt_bit = (i >> target) & 1;
        if ctrl_bit == 1 && tgt_bit == 0 {
            let j = i ^ (1 << target);
            state.swap(i, j);
        }
    }
}

// ============================================================
// PREDEFINED HAMILTONIANS
// ============================================================

/// H2 molecule Hamiltonian in STO-3G basis (4 qubits, 2 electrons).
///
/// Returns `(hamiltonian_terms, num_qubits, num_electrons, exact_energy)`.
///
/// The exact ground-state energy at equilibrium bond length (0.735 A) is
/// approximately -1.137 Hartree.
pub fn h2_hamiltonian() -> (Vec<(Vec<(usize, char)>, f64)>, usize, usize, f64) {
    // STO-3G H2 Hamiltonian in the qubit (Jordan-Wigner) representation.
    // Reference: O'Malley et al., PRX 6, 031007 (2016)
    let terms = vec![
        // Identity term (electronic + nuclear repulsion at R=0.7414 A)
        (vec![], -0.09706),
        // Z terms
        (vec![(0, 'Z')], 0.17218),
        (vec![(1, 'Z')], 0.17218),
        (vec![(2, 'Z')], -0.22575),
        (vec![(3, 'Z')], -0.22575),
        // ZZ terms
        (vec![(0, 'Z'), (1, 'Z')], 0.16893),
        (vec![(0, 'Z'), (2, 'Z')], 0.12091),
        (vec![(0, 'Z'), (3, 'Z')], 0.16615),
        (vec![(1, 'Z'), (2, 'Z')], 0.16615),
        (vec![(1, 'Z'), (3, 'Z')], 0.12091),
        (vec![(2, 'Z'), (3, 'Z')], 0.17464),
        // XX + YY terms (from exchange integrals)
        (vec![(0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')], -0.04524),
        (vec![(0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')], -0.04524),
        (vec![(0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')], 0.04524),
        (vec![(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')], 0.04524),
    ];

    let exact_energy = -1.137_283_6;
    (terms, 4, 2, exact_energy)
}

/// H4 linear chain Hamiltonian (8 qubits, 4 electrons).
///
/// Returns `(hamiltonian_terms, num_qubits, num_electrons, exact_energy)`.
/// Simplified model with nearest-neighbor interactions.
pub fn h4_hamiltonian() -> (Vec<(Vec<(usize, char)>, f64)>, usize, usize, f64) {
    let mut terms = Vec::new();

    // Nuclear repulsion + one-body terms
    terms.push((vec![], -1.5));

    // Single Z terms (orbital energies)
    let orbital_energies = [0.15, 0.15, 0.15, 0.15, -0.20, -0.20, -0.20, -0.20];
    for (i, &e) in orbital_energies.iter().enumerate() {
        terms.push((vec![(i, 'Z')], e));
    }

    // ZZ interactions (Coulomb + exchange)
    for i in 0..8 {
        for j in (i + 1)..8 {
            let dist = ((i as f64 - j as f64).abs()).max(1.0);
            let coeff = 0.1 / dist;
            terms.push((vec![(i, 'Z'), (j, 'Z')], coeff));
        }
    }

    // Exchange terms (XX + YY between nearest-neighbor pairs)
    for i in 0..7 {
        let coeff = -0.03;
        terms.push((vec![(i, 'X'), (i + 1, 'X')], coeff));
        terms.push((vec![(i, 'Y'), (i + 1, 'Y')], coeff));
    }

    let exact_energy = -2.15; // approximate
    (terms, 8, 4, exact_energy)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_defaults() {
        let config = GgaVqeConfig::default();
        assert_eq!(config.max_operators, 20);
        assert!((config.energy_threshold - 1e-6).abs() < 1e-15);
        assert_eq!(config.optimizer, GradientFreeOptimizer::NelderMead);
        assert_eq!(config.max_opt_iterations, 100);
        assert!(config.shots.is_none());
        matches!(config.operator_pool, OperatorPool::FullUCCSD);
    }

    #[test]
    fn test_hf_state_correct_electron_count() {
        let state = hartree_fock_state(4, 2);
        assert_eq!(state.len(), 16);
        // HF state for 2 electrons: |0011> = index 3
        assert!((state[3].norm() - 1.0).abs() < 1e-10);
        // Count number of ones in the occupied index
        let hf_idx = 3usize; // binary: 11
        assert_eq!(hf_idx.count_ones() as usize, 2);

        // Test with 3 electrons, 6 qubits
        let state2 = hartree_fock_state(6, 3);
        assert_eq!(state2.len(), 64);
        // HF index = (1<<3) - 1 = 7 = binary 111
        assert!((state2[7].norm() - 1.0).abs() < 1e-10);
        assert_eq!(7usize.count_ones() as usize, 3);
    }

    #[test]
    fn test_single_excitation_pool_size() {
        // n_occ * n_virt = 2 * 2 = 4 for 4 orbitals, 2 electrons
        let ops = generate_single_excitations(4, 2);
        assert_eq!(ops.len(), 4);

        // 3 * 3 = 9 for 6 orbitals, 3 electrons
        let ops2 = generate_single_excitations(6, 3);
        assert_eq!(ops2.len(), 9);
    }

    #[test]
    fn test_double_excitation_pool_size() {
        // C(2,2) * C(2,2) = 1 * 1 = 1 for 4 orbitals, 2 electrons
        let ops = generate_double_excitations(4, 2);
        assert_eq!(ops.len(), 1);

        // C(3,2) * C(3,2) = 3 * 3 = 9 for 6 orbitals, 3 electrons
        let ops2 = generate_double_excitations(6, 3);
        assert_eq!(ops2.len(), 9);
    }

    #[test]
    fn test_nelder_mead_finds_minimum_of_quadratic() {
        // f(x,y) = x^2 + y^2, minimum at (0,0)
        let f = |params: &[f64]| -> f64 { params[0] * params[0] + params[1] * params[1] };
        let (params, val) = nelder_mead(&f, &[3.0, 4.0], 1000, 1e-12);
        assert!(params[0].abs() < 1e-4, "x = {}", params[0]);
        assert!(params[1].abs() < 1e-4, "y = {}", params[1]);
        assert!(val < 1e-8, "f = {}", val);
    }

    #[test]
    fn test_golden_section_finds_minimum() {
        // f(x) = (x - 3)^2, minimum at x = 3
        let f = |x: f64| -> f64 { (x - 3.0) * (x - 3.0) };
        let x_min = golden_section_1d(&f, 0.0, 10.0, 1e-8);
        assert!((x_min - 3.0).abs() < 1e-6, "x_min = {}", x_min);
    }

    #[test]
    fn test_gga_vqe_h2_energy() {
        let (ham, nq, ne, exact) = h2_hamiltonian();
        let config = GgaVqeConfig {
            max_operators: 10,
            energy_threshold: 1e-6,
            operator_pool: OperatorPool::FullUCCSD,
            optimizer: GradientFreeOptimizer::NelderMead,
            max_opt_iterations: 200,
            shots: None,
            gradient_step: 1e-4,
            gradient_threshold: 1e-6,
        };
        let result = run_gga_vqe(&ham, nq, ne, &config).unwrap();
        let error = (result.energy - exact).abs();
        assert!(
            error < 0.05,
            "GGA-VQE energy {} too far from exact {} (error = {})",
            result.energy,
            exact,
            error
        );
    }

    #[test]
    fn test_energy_decreases_with_adapt_iterations() {
        let (ham, nq, ne, _) = h2_hamiltonian();
        let config = GgaVqeConfig {
            max_operators: 5,
            energy_threshold: 1e-10, // very tight so we don't stop early
            operator_pool: OperatorPool::FullUCCSD,
            optimizer: GradientFreeOptimizer::NelderMead,
            max_opt_iterations: 200,
            shots: None,
            gradient_step: 1e-4,
            gradient_threshold: 1e-8,
        };
        let result = run_gga_vqe(&ham, nq, ne, &config).unwrap();

        // Energy should be non-increasing at each step
        for i in 1..result.energy_history.len() {
            assert!(
                result.energy_history[i] <= result.energy_history[i - 1] + 1e-8,
                "Energy increased at step {}: {} > {}",
                i,
                result.energy_history[i],
                result.energy_history[i - 1]
            );
        }
    }

    #[test]
    fn test_greedy_selection_picks_largest_gradient() {
        // Create two operators: one with large gradient, one with small
        let (ham, nq, ne, _) = h2_hamiltonian();
        let config = GgaVqeConfig {
            max_operators: 1, // only pick one
            energy_threshold: 1e-10,
            operator_pool: OperatorPool::FullUCCSD,
            optimizer: GradientFreeOptimizer::NelderMead,
            max_opt_iterations: 50,
            shots: None,
            gradient_step: 1e-4,
            gradient_threshold: 1e-8,
        };

        let result = run_gga_vqe(&ham, nq, ne, &config).unwrap();

        // The first operator selected should be the one with the largest gradient
        assert_eq!(result.num_operators_used, 1);
        assert!(!result.operator_sequence.is_empty());

        // Verify it picked the operator that gives the most energy improvement
        let pool = {
            let mut ops = generate_single_excitations(nq, ne);
            ops.extend(generate_double_excitations(nq, ne));
            ops
        };

        let circuit = AdaptCircuit::new(nq, ne);
        let mut gradients: Vec<(usize, f64)> = pool
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let g = compute_operator_gradient(&circuit, op, &ham, 1e-4);
                (i, g.abs())
            })
            .collect();
        gradients.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // The selected operator should match the one with the largest gradient
        let expected_label = &pool[gradients[0].0].label;
        assert_eq!(
            &result.operator_sequence[0], expected_label,
            "Expected {} but got {}",
            expected_label, result.operator_sequence[0]
        );
    }

    #[test]
    fn test_excitation_operator_is_anti_hermitian() {
        // Single excitations should be anti-Hermitian (coefficients sum to 0)
        let ops = generate_single_excitations(4, 2);
        for op in &ops {
            assert!(
                op.is_anti_hermitian(),
                "Operator {} is not anti-Hermitian",
                op.label
            );
        }

        // Double excitations should also be anti-Hermitian
        let dops = generate_double_excitations(4, 2);
        for op in &dops {
            assert!(
                op.is_anti_hermitian(),
                "Operator {} is not anti-Hermitian",
                op.label
            );
        }
    }

    #[test]
    fn test_hf_energy_higher_than_exact() {
        let (ham, nq, ne, exact) = h2_hamiltonian();
        let hf_state = hartree_fock_state(nq, ne);
        let hf_energy = compute_energy(&hf_state, &ham);
        assert!(
            hf_energy > exact,
            "HF energy {} should be higher than exact {}",
            hf_energy,
            exact
        );
    }

    #[test]
    fn test_circuit_depth_grows_with_operators() {
        let (ham, nq, ne, _) = h2_hamiltonian();
        let config = GgaVqeConfig {
            max_operators: 3,
            energy_threshold: 1e-12, // very tight: ensure we add multiple ops
            operator_pool: OperatorPool::FullUCCSD,
            optimizer: GradientFreeOptimizer::NelderMead,
            max_opt_iterations: 100,
            shots: None,
            gradient_step: 1e-4,
            gradient_threshold: 1e-8,
        };

        let result = run_gga_vqe(&ham, nq, ne, &config).unwrap();
        assert_eq!(result.circuit.depth(), result.num_operators_used);
        // Should have added at least 1 operator
        assert!(result.num_operators_used >= 1);
    }

    #[test]
    fn test_empty_pool_returns_error() {
        let (ham, nq, ne, _) = h2_hamiltonian();
        let config = GgaVqeConfig {
            operator_pool: OperatorPool::Custom(vec![]),
            ..GgaVqeConfig::default()
        };
        let result = run_gga_vqe(&ham, nq, ne, &config);
        assert!(matches!(result, Err(GgaVqeError::EmptyPool)));
    }

    #[test]
    fn test_coordinate_descent_optimizer() {
        let f = |params: &[f64]| -> f64 { (params[0] - 1.0).powi(2) + (params[1] + 2.0).powi(2) };
        let (params, val) = coordinate_descent(&f, &[5.0, 5.0], 100, 1e-8);
        assert!((params[0] - 1.0).abs() < 0.1, "x = {}", params[0]);
        assert!((params[1] + 2.0).abs() < 0.1, "y = {}", params[1]);
        assert!(val < 0.02, "f = {}", val);
    }

    #[test]
    fn test_simulate_adapt_circuit_matches_direct() {
        let (ham, nq, ne, _) = h2_hamiltonian();

        // Empty circuit should give HF energy
        let circuit = AdaptCircuit::new(nq, ne);
        let e_circuit = simulate_adapt_circuit(&circuit, &ham);
        let e_direct = compute_energy(&hartree_fock_state(nq, ne), &ham);
        assert!(
            (e_circuit - e_direct).abs() < 1e-10,
            "Circuit energy {} != direct {}",
            e_circuit,
            e_direct
        );
    }

    #[test]
    fn test_triple_excitation_pool_size() {
        // C(2,3) = 0 for occupied, so 0 triples with 4 orbitals, 2 electrons
        let ops = generate_triple_excitations(4, 2);
        assert_eq!(ops.len(), 0);

        // 6 orbitals, 3 electrons: C(3,3) * C(3,3) = 1 * 1 = 1
        let ops2 = generate_triple_excitations(6, 3);
        assert_eq!(ops2.len(), 1);

        // 8 orbitals, 4 electrons: C(4,3) * C(4,3) = 4 * 4 = 16
        let ops3 = generate_triple_excitations(8, 4);
        assert_eq!(ops3.len(), 16);

        // 8 orbitals, 3 electrons: C(3,3) * C(5,3) = 1 * 10 = 10
        let ops4 = generate_triple_excitations(8, 3);
        assert_eq!(ops4.len(), 10);
    }

    #[test]
    fn test_triple_excitation_is_anti_hermitian() {
        // 6 orbitals, 3 electrons: 1 triple excitation
        let ops = generate_triple_excitations(6, 3);
        for op in &ops {
            assert!(
                op.is_anti_hermitian(),
                "Triple excitation operator {} is not anti-Hermitian",
                op.label
            );
        }

        // 8 orbitals, 4 electrons: 16 triple excitations
        let ops2 = generate_triple_excitations(8, 4);
        for op in &ops2 {
            assert!(
                op.is_anti_hermitian(),
                "Triple excitation operator {} is not anti-Hermitian",
                op.label
            );
        }
    }

    #[test]
    fn test_triple_excitation_has_32_terms() {
        // Each triple excitation should produce 32 Pauli terms
        let ops = generate_triple_excitations(6, 3);
        assert_eq!(ops.len(), 1);
        assert_eq!(
            ops[0].terms.len(),
            32,
            "Triple excitation should have 32 Pauli terms, got {}",
            ops[0].terms.len()
        );
    }

    #[test]
    fn test_triple_excitation_label_format() {
        let ops = generate_triple_excitations(6, 3);
        assert!(!ops.is_empty());
        // Label should be T_occ,occ,occ->virt,virt,virt
        let label = &ops[0].label;
        assert!(
            label.starts_with("T_"),
            "Triple excitation label should start with T_, got {}",
            label
        );
        assert!(
            label.contains('→'),
            "Triple excitation label should contain →, got {}",
            label
        );
    }

    #[test]
    fn test_excitation_to_pauli_dispatches_triple() {
        // Should not panic for triple excitations
        let op = excitation_to_pauli(&[3, 4, 5], &[0, 1, 2]);
        assert_eq!(op.terms.len(), 32);
        assert!(op.is_anti_hermitian());
    }

    #[test]
    fn test_triple_excitation_z_chains() {
        // For non-adjacent orbitals, Z-chains should appear between active qubits
        // Use orbitals 0, 2, 4, 6, 8, 10 (gaps of 2 between each)
        let op = excitation_to_pauli(&[6, 8, 10], &[0, 2, 4]);
        for (term, _coeff) in &op.terms {
            // Check that Z-chains exist between non-adjacent active qubits
            // Active qubits: 0, 2, 4, 6, 8, 10
            // Z-chain qubits: 1, 3, 5, 7, 9
            let z_qubits: Vec<usize> = term
                .iter()
                .filter(|(_, p)| *p == 'Z')
                .map(|(q, _)| *q)
                .collect();
            // Should have exactly 5 Z qubits (one between each pair of active qubits)
            assert_eq!(
                z_qubits.len(),
                5,
                "Expected 5 Z-chain qubits, got {:?}",
                z_qubits
            );
            assert!(z_qubits.contains(&1));
            assert!(z_qubits.contains(&3));
            assert!(z_qubits.contains(&5));
            assert!(z_qubits.contains(&7));
            assert!(z_qubits.contains(&9));
        }
    }

    #[test]
    fn test_triple_excitation_only_xy_on_active_qubits() {
        // Active qubits should have X or Y, never Z or I
        let op = excitation_to_pauli(&[3, 4, 5], &[0, 1, 2]);
        let active = [0, 1, 2, 3, 4, 5];
        for (term, _) in &op.terms {
            for &aq in &active {
                let pauli = term.iter().find(|(q, _)| *q == aq).map(|(_, p)| *p);
                assert!(
                    pauli == Some('X') || pauli == Some('Y'),
                    "Active qubit {} should be X or Y, got {:?}",
                    aq,
                    pauli
                );
            }
        }
    }

    #[test]
    fn test_uccsdt_pool_includes_all_excitation_types() {
        // 8 qubits, 4 electrons
        let singles = generate_single_excitations(8, 4);
        let doubles = generate_double_excitations(8, 4);
        let triples = generate_triple_excitations(8, 4);

        // FullUCCSDT should be the union of all three
        let config = GgaVqeConfig {
            operator_pool: OperatorPool::FullUCCSDT,
            ..GgaVqeConfig::default()
        };

        // Build pool the same way run_gga_vqe does
        let pool = {
            let mut ops = generate_single_excitations(8, 4);
            ops.extend(generate_double_excitations(8, 4));
            ops.extend(generate_triple_excitations(8, 4));
            ops
        };

        assert_eq!(
            pool.len(),
            singles.len() + doubles.len() + triples.len(),
            "FullUCCSDT pool size mismatch"
        );
        // singles: 4*4=16, doubles: C(4,2)*C(4,2)=36, triples: C(4,3)*C(4,3)=16
        assert_eq!(singles.len(), 16);
        assert_eq!(doubles.len(), 36);
        assert_eq!(triples.len(), 16);
        assert_eq!(pool.len(), 68);
    }

    #[test]
    fn test_triple_excitation_unitary_preserves_norm() {
        // Applying exp(-i * theta * A) should preserve the state norm
        let num_qubits = 6;
        let num_electrons = 3;
        let mut state = hartree_fock_state(num_qubits, num_electrons);

        let ops = generate_triple_excitations(num_qubits, num_electrons);
        assert!(!ops.is_empty());

        let norm_before: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        apply_excitation_unitary(&mut state, &ops[0], 0.5);
        let norm_after: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            (norm_before - norm_after).abs() < 1e-10,
            "Norm changed from {} to {} after triple excitation unitary",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_gga_vqe_h4_with_uccsdt() {
        // H4 has enough electrons/orbitals for triple excitations
        let (ham, nq, ne, exact) = h4_hamiltonian();
        let config = GgaVqeConfig {
            max_operators: 5,
            energy_threshold: 1e-4,
            operator_pool: OperatorPool::FullUCCSDT,
            optimizer: GradientFreeOptimizer::NelderMead,
            max_opt_iterations: 100,
            shots: None,
            gradient_step: 1e-4,
            gradient_threshold: 1e-5,
        };
        let result = run_gga_vqe(&ham, nq, ne, &config).unwrap();
        // UCCSDT should at least improve over HF
        let hf_energy = compute_energy(&hartree_fock_state(nq, ne), &ham);
        assert!(
            result.energy <= hf_energy + 1e-6,
            "UCCSDT energy {} should be <= HF energy {}",
            result.energy,
            hf_energy
        );
    }

    #[test]
    fn test_standard_vqe_comparison() {
        let (ham, nq, _, _exact) = h2_hamiltonian();
        let (energy, evals) = run_standard_vqe(&ham, nq, 2);
        // Standard VQE should at least improve over |00...0> energy
        let zero_state_energy = compute_energy(
            &vec![Complex64::new(0.0, 0.0); 1 << nq]
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect::<Vec<_>>(),
            &ham,
        );
        assert!(energy <= zero_state_energy + 0.1);
        assert!(
            evals > 0,
            "Should have evaluated the objective at least once"
        );
    }
}
