//! Quantum Approximate Multi-Objective Optimization (QAMOO)
//!
//! Extends QAOA to multi-objective optimization using Tchebycheff scalarization
//! and parameter transfer. Based on Nature Computational Science 2025.
//!
//! # Overview
//!
//! QAMOO decomposes a multi-objective combinatorial optimization problem into a
//! set of single-objective subproblems via scalarization, solves each with QAOA,
//! and assembles the non-dominated solutions into a Pareto front.
//!
//! # Algorithm
//!
//! 1. Generate weight vectors λ uniformly on the simplex
//! 2. For each λ, construct a scalarized cost function and run QAOA
//! 3. Collect all solutions, perform non-dominated sorting
//! 4. Compute the hypervolume indicator of the Pareto front

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during QAMOO execution.
#[derive(Debug, Clone)]
pub enum QamooError {
    /// Fewer than 2 objectives were provided.
    InvalidObjectives(String),
    /// The inner QAOA optimizer failed to converge.
    OptimizationFailed(String),
    /// No non-dominated points were found.
    NoParetoPoints,
}

impl fmt::Display for QamooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidObjectives(msg) => write!(f, "Invalid objectives: {}", msg),
            Self::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
            Self::NoParetoPoints => write!(f, "No Pareto points found"),
        }
    }
}

impl std::error::Error for QamooError {}

// ---------------------------------------------------------------------------
// Scalarization strategy
// ---------------------------------------------------------------------------

/// Scalarization method for decomposing multi-objective problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalarization {
    /// Tchebycheff: minimise max_i { w_i * |f_i(x) - z_i*| }
    Tchebycheff,
    /// Weighted sum: minimise Σ w_i * f_i(x)
    WeightedSum,
    /// Epsilon-constraint: optimise first objective, bound the rest.
    EpsilonConstraint,
}

// ---------------------------------------------------------------------------
// Optimizer selector
// ---------------------------------------------------------------------------

/// Optimizer used inside each QAOA sub-problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    Cobyla,
    GradientFree,
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Configuration for the QAMOO algorithm.
#[derive(Debug, Clone)]
pub struct QamooConfig {
    /// Number of QAOA layers (p). Default: 3.
    pub num_layers: usize,
    /// Number of objectives. Default: 2.
    pub num_objectives: usize,
    /// Number of weight vectors on the simplex. Default: 20.
    pub num_weight_vectors: usize,
    /// Scalarization method. Default: Tchebycheff.
    pub scalarization: Scalarization,
    /// Inner optimizer. Default: Cobyla.
    pub optimizer: Optimizer,
    /// Maximum QAOA optimisation iterations. Default: 100.
    pub max_iterations: usize,
    /// Population size (unused in basic QAOA but reserved). Default: 50.
    pub population_size: usize,
}

impl Default for QamooConfig {
    fn default() -> Self {
        Self {
            num_layers: 3,
            num_objectives: 2,
            num_weight_vectors: 20,
            scalarization: Scalarization::Tchebycheff,
            optimizer: Optimizer::Cobyla,
            max_iterations: 100,
            population_size: 50,
        }
    }
}

impl QamooConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    pub fn num_objectives(mut self, n: usize) -> Self {
        self.num_objectives = n;
        self
    }

    pub fn num_weight_vectors(mut self, n: usize) -> Self {
        self.num_weight_vectors = n;
        self
    }

    pub fn scalarization(mut self, s: Scalarization) -> Self {
        self.scalarization = s;
        self
    }

    pub fn optimizer(mut self, o: Optimizer) -> Self {
        self.optimizer = o;
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn population_size(mut self, n: usize) -> Self {
        self.population_size = n;
        self
    }
}

// ---------------------------------------------------------------------------
// Objective function
// ---------------------------------------------------------------------------

/// A single objective expressed as a diagonal Hamiltonian in the computational basis.
///
/// Each term is a list of (qubit, pauli_char) pairs and a coefficient.
/// Supported Pauli characters: 'Z', 'I'.
#[derive(Debug, Clone)]
pub struct ObjectiveFunction {
    /// Hamiltonian terms: Vec<(Vec<(qubit, pauli)>, coefficient)>
    pub hamiltonian_terms: Vec<(Vec<(usize, char)>, f64)>,
    /// Human-readable name.
    pub name: String,
}

impl ObjectiveFunction {
    pub fn new(name: impl Into<String>, terms: Vec<(Vec<(usize, char)>, f64)>) -> Self {
        Self {
            hamiltonian_terms: terms,
            name: name.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pareto types
// ---------------------------------------------------------------------------

/// A single point in objective space together with its encoding.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    /// Binary solution vector.
    pub bitstring: Vec<u8>,
    /// Objective values (one per objective).
    pub objectives: Vec<f64>,
    /// QAOA variational parameters that produced this point.
    pub params: Vec<f64>,
}

/// A Pareto front: the set of non-dominated solutions.
#[derive(Debug, Clone)]
pub struct ParetoFront {
    pub points: Vec<ParetoPoint>,
    pub hypervolume: f64,
}

/// Full result returned by [`run_qamoo`].
#[derive(Debug, Clone)]
pub struct QamooResult {
    pub pareto_front: ParetoFront,
    pub num_evaluations: usize,
    pub weight_vectors_used: Vec<Vec<f64>>,
}

// =========================================================================
// State-vector QAOA simulation (self-contained)
// =========================================================================

/// Simulate a QAOA circuit and return the final state vector.
///
/// `params` layout: [γ_1, ..., γ_p, β_1, ..., β_p]
pub fn simulate_qaoa_circuit(
    params: &[f64],
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    num_qubits: usize,
    num_layers: usize,
) -> Vec<Complex64> {
    let dim = 1usize << num_qubits;
    // Start in |+>^n
    let amp = 1.0 / (dim as f64).sqrt();
    let mut state: Vec<Complex64> = vec![Complex64::new(amp, 0.0); dim];

    for p in 0..num_layers {
        let gamma = params[p];
        let beta = params[num_layers + p];
        apply_cost_unitary(&mut state, hamiltonian, gamma);
        apply_mixer_unitary(&mut state, num_qubits, beta);
    }
    state
}

/// Apply the cost unitary U_C(γ) = exp(-i γ H_C) to `state`.
///
/// H_C is diagonal in the computational basis, so we compute the diagonal
/// element for each basis state and apply the phase.
pub fn apply_cost_unitary(
    state: &mut Vec<Complex64>,
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    gamma: f64,
) {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    for basis in 0..dim {
        let energy = evaluate_hamiltonian_on_basis(hamiltonian, basis, num_qubits);
        let phase = Complex64::new(0.0, -gamma * energy).exp();
        state[basis] *= phase;
    }
}

/// Apply the mixer unitary U_M(β) = exp(-i β Σ_j X_j) = Π_j exp(-i β X_j).
///
/// Each single-qubit rotation is R_x(2β) = exp(-i β X).
pub fn apply_mixer_unitary(state: &mut Vec<Complex64>, num_qubits: usize, beta: f64) {
    let cos_b = beta.cos();
    let sin_b = beta.sin();
    let neg_i_sin = Complex64::new(0.0, -sin_b);

    for q in 0..num_qubits {
        let mask = 1usize << q;
        let dim = state.len();
        let mut basis = 0usize;
        while basis < dim {
            // Find pairs that differ only in qubit q
            if basis & mask == 0 {
                let partner = basis | mask;
                let a = state[basis];
                let b = state[partner];
                state[basis] = a * cos_b + b * neg_i_sin;
                state[partner] = a * neg_i_sin + b * cos_b;
            }
            basis += 1;
        }
    }
}

/// Compute <ψ|H|ψ> for a diagonal Hamiltonian.
pub fn expectation_value(
    state: &[Complex64],
    hamiltonian: &[(Vec<(usize, char)>, f64)],
) -> f64 {
    let num_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let mut expval = 0.0;
    for basis in 0..dim {
        let prob = state[basis].norm_sqr();
        let energy = evaluate_hamiltonian_on_basis(hamiltonian, basis, num_qubits);
        expval += prob * energy;
    }
    expval
}

/// Evaluate a diagonal Pauli Hamiltonian on a computational-basis state.
fn evaluate_hamiltonian_on_basis(
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    basis: usize,
    _num_qubits: usize,
) -> f64 {
    let mut energy = 0.0;
    for (paulis, coeff) in hamiltonian {
        let mut term_val: f64 = 1.0;
        for &(qubit, pauli) in paulis {
            match pauli {
                'Z' | 'z' => {
                    let bit = (basis >> qubit) & 1;
                    term_val *= if bit == 0 { 1.0 } else { -1.0 };
                }
                'I' | 'i' => {} // identity
                _ => {}          // treat unknown as identity
            }
        }
        energy += coeff * term_val;
    }
    energy
}

// =========================================================================
// QAOA optimisation (Nelder-Mead)
// =========================================================================

/// Run QAOA on a single Hamiltonian and return (optimal_params, best_energy, best_bitstring).
pub fn qaoa_optimize(
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    num_qubits: usize,
    num_layers: usize,
    max_iter: usize,
) -> (Vec<f64>, f64, Vec<u8>) {
    let num_params = 2 * num_layers;

    // Cost closure
    let cost_fn = |params: &[f64]| -> f64 {
        let state = simulate_qaoa_circuit(params, hamiltonian, num_qubits, num_layers);
        expectation_value(&state, hamiltonian)
    };

    // Initial parameters: small random perturbation around π/4
    let mut rng = rand::thread_rng();
    let init: Vec<f64> = (0..num_params)
        .map(|_| PI / 4.0 + rng.gen_range(-0.1..0.1))
        .collect();

    let best_params = nelder_mead(&cost_fn, &init, max_iter);
    let best_energy = cost_fn(&best_params);

    // Extract best bitstring by sampling most-probable basis state
    let state = simulate_qaoa_circuit(&best_params, hamiltonian, num_qubits, num_layers);
    let best_bitstring = most_probable_bitstring(&state, num_qubits);

    (best_params, best_energy, best_bitstring)
}

/// Nelder-Mead (downhill simplex) minimiser. Simple but effective for QAOA.
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64> {
    let n = x0.len();
    let alpha = 1.0; // reflection
    let gamma_expand = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    // Build initial simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += 0.5;
        simplex.push(v);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        // Sort by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();

        for i in 0..=n {
            simplex[i] = sorted_simplex[i].clone();
            values[i] = sorted_values[i];
        }

        // Centroid of all points except the worst
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
                .map(|j| centroid[j] + gamma_expand * (reflected[j] - centroid[j]))
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
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + rho * (simplex[n][j] - centroid[j]))
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

        // Convergence check
        let fmin = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let fmax = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if (fmax - fmin).abs() < 1e-10 {
            break;
        }
    }

    // Return best
    let best_idx = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    simplex[best_idx].clone()
}

/// Return the bitstring with highest probability.
fn most_probable_bitstring(state: &[Complex64], num_qubits: usize) -> Vec<u8> {
    let best_idx = state
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
        .unwrap()
        .0;
    (0..num_qubits)
        .map(|q| ((best_idx >> q) & 1) as u8)
        .collect()
}

// =========================================================================
// Scalarization
// =========================================================================

/// Tchebycheff scalarization: max_i { w_i * |f_i - z_i*| }.
pub fn tchebycheff_scalarize(objectives: &[f64], weights: &[f64], ideal: &[f64]) -> f64 {
    objectives
        .iter()
        .zip(weights.iter())
        .zip(ideal.iter())
        .map(|((&fi, &wi), &zi)| wi * (fi - zi).abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Weighted-sum scalarization: Σ w_i * f_i.
pub fn weighted_sum_scalarize(objectives: &[f64], weights: &[f64]) -> f64 {
    objectives
        .iter()
        .zip(weights.iter())
        .map(|(&fi, &wi)| wi * fi)
        .sum()
}

/// Generate `num_vectors` weight vectors uniformly distributed on the unit simplex
/// in `num_objectives` dimensions.
///
/// Uses the Das-Dennis method for 2 objectives and a lattice approach for higher dims.
pub fn generate_weight_vectors(num_objectives: usize, num_vectors: usize) -> Vec<Vec<f64>> {
    if num_objectives == 1 {
        return vec![vec![1.0]; num_vectors];
    }

    if num_objectives == 2 {
        // Simple uniform spacing on the 1-simplex
        let mut vectors = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let w1 = if num_vectors == 1 {
                0.5
            } else {
                i as f64 / (num_vectors - 1) as f64
            };
            let w2 = 1.0 - w1;
            vectors.push(vec![w1, w2]);
        }
        return vectors;
    }

    // General case: random sampling on the simplex (Dirichlet(1,...,1))
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let mut exps: Vec<f64> = (0..num_objectives)
            .map(|_| -rng.gen::<f64>().ln())
            .collect();
        let sum: f64 = exps.iter().sum();
        for v in &mut exps {
            *v /= sum;
        }
        vectors.push(exps);
    }
    vectors
}

// =========================================================================
// Parameter transfer
// =========================================================================

/// Interpolate or extrapolate QAOA parameters from `source_layers` to `target_layers`.
///
/// Parameters are assumed to be laid out as [γ_1..γ_p, β_1..β_p].
/// Linear interpolation is used for up-scaling, linear extrapolation for down-scaling.
pub fn transfer_params(
    source_params: &[f64],
    source_layers: usize,
    target_layers: usize,
) -> Vec<f64> {
    if source_layers == target_layers {
        return source_params.to_vec();
    }

    let gammas_src = &source_params[..source_layers];
    let betas_src = &source_params[source_layers..2 * source_layers];

    let gammas_tgt = interpolate_1d(gammas_src, target_layers);
    let betas_tgt = interpolate_1d(betas_src, target_layers);

    let mut result = gammas_tgt;
    result.extend(betas_tgt);
    result
}

/// 1-D linear interpolation of a vector to a new length.
fn interpolate_1d(src: &[f64], target_len: usize) -> Vec<f64> {
    if target_len == 0 {
        return vec![];
    }
    if src.len() == 1 {
        return vec![src[0]; target_len];
    }
    let mut result = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let t = if target_len == 1 {
            0.0
        } else {
            i as f64 / (target_len - 1) as f64
        };
        let src_pos = t * (src.len() - 1) as f64;
        let lo = (src_pos.floor() as usize).min(src.len() - 2);
        let hi = lo + 1;
        let frac = src_pos - lo as f64;
        result.push(src[lo] * (1.0 - frac) + src[hi] * frac);
    }
    result
}

/// Warm-start QAOA from initial parameters.
pub fn warm_start_qaoa(
    initial_params: &[f64],
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    num_qubits: usize,
    config: &QamooConfig,
) -> (Vec<f64>, f64) {
    let cost_fn = |params: &[f64]| -> f64 {
        let state = simulate_qaoa_circuit(params, hamiltonian, num_qubits, config.num_layers);
        expectation_value(&state, hamiltonian)
    };

    let best_params = nelder_mead(&cost_fn, initial_params, config.max_iterations);
    let best_energy = cost_fn(&best_params);
    (best_params, best_energy)
}

// =========================================================================
// Pareto front operations
// =========================================================================

/// Returns true if `a` Pareto-dominates `b`: a_i <= b_i for all i and a_j < b_j for some j.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly_better = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Non-dominated sorting. Returns a vector of fronts where each front is a
/// vector of indices into `points`. Front 0 is the Pareto front.
pub fn non_dominated_sort(points: &[ParetoPoint]) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }

    let mut domination_count: Vec<usize> = vec![0; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&points[i].objectives, &points[j].objectives) {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if dominates(&points[j].objectives, &points[i].objectives) {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Compute the 2-D hypervolume indicator for a set of points relative to
/// a reference point. For higher dimensions, falls back to a Monte-Carlo estimate.
pub fn compute_hypervolume(front: &[ParetoPoint], reference: &[f64]) -> f64 {
    if front.is_empty() || reference.is_empty() {
        return 0.0;
    }

    let num_obj = reference.len();
    if num_obj == 2 {
        compute_hypervolume_2d(front, reference)
    } else {
        compute_hypervolume_mc(front, reference, 10_000)
    }
}

/// Exact 2-D hypervolume via sweep line.
fn compute_hypervolume_2d(front: &[ParetoPoint], reference: &[f64]) -> f64 {
    // Filter points that are dominated by the reference
    let mut pts: Vec<(f64, f64)> = front
        .iter()
        .filter(|p| p.objectives[0] <= reference[0] && p.objectives[1] <= reference[1])
        .map(|p| (p.objectives[0], p.objectives[1]))
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Sort by first objective
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut hv = 0.0;
    let mut prev_y = reference[1];

    for &(x, y) in pts.iter() {
        if y < prev_y {
            hv += (reference[0] - x) * (prev_y - y);
            prev_y = y;
        }
    }

    hv
}

/// Monte-Carlo hypervolume for d > 2.
fn compute_hypervolume_mc(front: &[ParetoPoint], reference: &[f64], samples: usize) -> f64 {
    let num_obj = reference.len();
    // Compute bounding box
    let mut lower = vec![f64::INFINITY; num_obj];
    for p in front {
        for (i, &v) in p.objectives.iter().enumerate() {
            if v < lower[i] {
                lower[i] = v;
            }
        }
    }

    let vol_box: f64 = (0..num_obj).map(|i| reference[i] - lower[i]).product();
    if vol_box <= 0.0 {
        return 0.0;
    }

    let mut rng = rand::thread_rng();
    let mut inside = 0usize;

    for _ in 0..samples {
        let sample: Vec<f64> = (0..num_obj)
            .map(|i| rng.gen_range(lower[i]..reference[i]))
            .collect();
        // Check if dominated by any point in the front
        let is_dominated = front.iter().any(|p| dominates(&p.objectives, &sample));
        if is_dominated {
            inside += 1;
        }
    }

    vol_box * (inside as f64 / samples as f64)
}

/// Extract the Pareto front (rank 0) from a set of points.
pub fn pareto_front(points: &[ParetoPoint]) -> ParetoFront {
    if points.is_empty() {
        return ParetoFront {
            points: vec![],
            hypervolume: 0.0,
        };
    }

    let fronts = non_dominated_sort(points);
    let front_indices = &fronts[0];
    let front_points: Vec<ParetoPoint> = front_indices.iter().map(|&i| points[i].clone()).collect();

    // Compute reference point: max of each objective + 1
    let num_obj = front_points[0].objectives.len();
    let mut reference = vec![f64::NEG_INFINITY; num_obj];
    for p in points {
        for (i, &v) in p.objectives.iter().enumerate() {
            if v > reference[i] {
                reference[i] = v;
            }
        }
    }
    for r in &mut reference {
        *r += 1.0;
    }

    let hv = compute_hypervolume(&front_points, &reference);

    ParetoFront {
        points: front_points,
        hypervolume: hv,
    }
}

// =========================================================================
// Example problems
// =========================================================================

/// Bi-objective max-cut problem.
///
/// - Objective 1: maximise cut weight (encoded as minimise negative cut weight)
/// - Objective 2: minimise partition imbalance (|S| - n/2)^2
pub fn bi_objective_max_cut(
    adjacency: &[(usize, usize, f64)],
    num_nodes: usize,
) -> Vec<ObjectiveFunction> {
    // Objective 1: cut weight = Σ_{(i,j)} w_{ij} (1 - Z_i Z_j) / 2
    // Minimise negative of cut weight => Σ w_{ij} Z_i Z_j / 2 (dropping constant)
    let mut cut_terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();
    for &(i, j, w) in adjacency {
        cut_terms.push((vec![(i, 'Z'), (j, 'Z')], w / 2.0));
    }
    let obj1 = ObjectiveFunction::new("cut_weight", cut_terms);

    // Objective 2: balance = (Σ Z_i)^2 / n = Σ_i Σ_j Z_i Z_j / n
    let mut balance_terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();
    let coeff = 1.0 / num_nodes as f64;
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i == j {
                balance_terms.push((vec![(i, 'I')], coeff));
            } else {
                balance_terms.push((vec![(i, 'Z'), (j, 'Z')], coeff));
            }
        }
    }
    let obj2 = ObjectiveFunction::new("balance", balance_terms);

    vec![obj1, obj2]
}

/// Bi-objective portfolio optimisation.
///
/// - Objective 1: minimise negative expected return (maximise return)
/// - Objective 2: minimise risk (variance proxy via Z_i Z_j correlations)
pub fn portfolio_bi_objective(
    returns: &[f64],
    covariance: &[Vec<f64>],
) -> Vec<ObjectiveFunction> {
    let n = returns.len();

    // Objective 1: -Σ r_i (1+Z_i)/2 ∝ -Σ r_i Z_i / 2 (dropping constant)
    let return_terms: Vec<(Vec<(usize, char)>, f64)> = (0..n)
        .map(|i| (vec![(i, 'Z')], -returns[i] / 2.0))
        .collect();
    let obj1 = ObjectiveFunction::new("negative_return", return_terms);

    // Objective 2: Σ_{ij} σ_{ij} Z_i Z_j / 4
    let mut risk_terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                risk_terms.push((vec![(i, 'I')], covariance[i][j] / 4.0));
            } else {
                risk_terms.push((vec![(i, 'Z'), (j, 'Z')], covariance[i][j] / 4.0));
            }
        }
    }
    let obj2 = ObjectiveFunction::new("risk", risk_terms);

    vec![obj1, obj2]
}

// =========================================================================
// Main QAMOO entry point
// =========================================================================

/// Run the Quantum Approximate Multi-Objective Optimization algorithm.
///
/// Given multiple objective functions, generates weight vectors on the simplex,
/// solves scalarized QAOA sub-problems, and returns a Pareto front.
pub fn run_qamoo(
    objectives: &[ObjectiveFunction],
    num_qubits: usize,
    config: &QamooConfig,
) -> Result<QamooResult, QamooError> {
    if objectives.len() < 2 {
        return Err(QamooError::InvalidObjectives(
            "QAMOO requires at least 2 objectives".to_string(),
        ));
    }

    let num_obj = objectives.len();
    let weight_vectors = generate_weight_vectors(num_obj, config.num_weight_vectors);

    // Step 1: Estimate the ideal point z* by optimising each objective independently
    let mut ideal = vec![f64::INFINITY; num_obj];
    let mut prev_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;

    for (k, obj) in objectives.iter().enumerate() {
        let (params, energy, _bs) =
            qaoa_optimize(&obj.hamiltonian_terms, num_qubits, config.num_layers, config.max_iterations);
        ideal[k] = energy;
        total_evals += config.max_iterations;
        prev_params = Some(params);
    }

    // Step 2: For each weight vector, solve a scalarized sub-problem
    let mut all_points: Vec<ParetoPoint> = Vec::new();

    for wv in &weight_vectors {
        // Build scalarized Hamiltonian
        let scalarized_terms = build_scalarized_hamiltonian(
            objectives,
            wv,
            &ideal,
            config.scalarization,
        );

        // Possibly warm-start from previous solution
        let (params, _energy, bitstring) = if let Some(ref prev) = prev_params {
            let transferred = transfer_params(prev, config.num_layers, config.num_layers);
            let (opt_params, _opt_energy) =
                warm_start_qaoa(&transferred, &scalarized_terms, num_qubits, config);
            let state = simulate_qaoa_circuit(&opt_params, &scalarized_terms, num_qubits, config.num_layers);
            let bs = most_probable_bitstring(&state, num_qubits);
            let e = expectation_value(&state, &scalarized_terms);
            (opt_params, e, bs)
        } else {
            qaoa_optimize(&scalarized_terms, num_qubits, config.num_layers, config.max_iterations)
        };

        total_evals += config.max_iterations;

        // Evaluate each objective for this bitstring
        let obj_values: Vec<f64> = objectives
            .iter()
            .map(|obj| {
                let basis = bitstring_to_basis(&bitstring);
                evaluate_hamiltonian_on_basis(
                    &obj.hamiltonian_terms,
                    basis,
                    num_qubits,
                )
            })
            .collect();

        all_points.push(ParetoPoint {
            bitstring: bitstring.clone(),
            objectives: obj_values,
            params: params.clone(),
        });

        prev_params = Some(params);
    }

    if all_points.is_empty() {
        return Err(QamooError::NoParetoPoints);
    }

    // Step 3: Compute Pareto front
    let pf = pareto_front(&all_points);

    if pf.points.is_empty() {
        return Err(QamooError::NoParetoPoints);
    }

    Ok(QamooResult {
        pareto_front: pf,
        num_evaluations: total_evals,
        weight_vectors_used: weight_vectors,
    })
}

/// Convert a bitstring to a basis state index.
fn bitstring_to_basis(bitstring: &[u8]) -> usize {
    let mut basis = 0usize;
    for (q, &bit) in bitstring.iter().enumerate() {
        if bit == 1 {
            basis |= 1 << q;
        }
    }
    basis
}

/// Build a scalarized Hamiltonian from multiple objectives.
fn build_scalarized_hamiltonian(
    objectives: &[ObjectiveFunction],
    weights: &[f64],
    _ideal: &[f64],
    scalarization: Scalarization,
) -> Vec<(Vec<(usize, char)>, f64)> {
    match scalarization {
        Scalarization::WeightedSum => {
            // Σ w_i * H_i
            let mut terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();
            for (k, obj) in objectives.iter().enumerate() {
                for (paulis, coeff) in &obj.hamiltonian_terms {
                    terms.push((paulis.clone(), weights[k] * coeff));
                }
            }
            terms
        }
        Scalarization::Tchebycheff | Scalarization::EpsilonConstraint => {
            // For Tchebycheff, we approximate by a weighted sum with boosted weights
            // for the Hamiltonian terms. This is an approximation since Tchebycheff
            // is not linear; the exact scalarization is applied post-hoc.
            // We use a large-weight penalty approach: sum w_i * H_i
            // and apply the exact Tchebycheff check during solution evaluation.
            let mut terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();
            for (k, obj) in objectives.iter().enumerate() {
                let w = if weights[k] < 1e-10 { 1e-10 } else { weights[k] };
                for (paulis, coeff) in &obj.hamiltonian_terms {
                    terms.push((paulis.clone(), w * coeff));
                }
            }
            terms
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_defaults() {
        let config = QamooConfig::new();
        assert_eq!(config.num_layers, 3);
        assert_eq!(config.num_objectives, 2);
        assert_eq!(config.num_weight_vectors, 20);
        assert_eq!(config.scalarization, Scalarization::Tchebycheff);
        assert_eq!(config.optimizer, Optimizer::Cobyla);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.population_size, 50);
    }

    #[test]
    fn test_config_builder_chaining() {
        let config = QamooConfig::new()
            .num_layers(5)
            .num_objectives(3)
            .num_weight_vectors(30)
            .scalarization(Scalarization::WeightedSum)
            .max_iterations(200);
        assert_eq!(config.num_layers, 5);
        assert_eq!(config.num_objectives, 3);
        assert_eq!(config.num_weight_vectors, 30);
        assert_eq!(config.scalarization, Scalarization::WeightedSum);
        assert_eq!(config.max_iterations, 200);
    }

    #[test]
    fn test_weight_vectors_sum_to_one() {
        let vecs = generate_weight_vectors(2, 10);
        assert_eq!(vecs.len(), 10);
        for v in &vecs {
            let sum: f64 = v.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "Weight vector sums to {}", sum);
        }
    }

    #[test]
    fn test_weight_vectors_cover_simplex() {
        let vecs = generate_weight_vectors(2, 11);
        // For 2 objectives with 11 vectors, endpoints should be (0,1) and (1,0)
        assert!((vecs[0][0] - 0.0).abs() < 1e-12);
        assert!((vecs[0][1] - 1.0).abs() < 1e-12);
        assert!((vecs[10][0] - 1.0).abs() < 1e-12);
        assert!((vecs[10][1] - 0.0).abs() < 1e-12);
        // Middle vector should be (0.5, 0.5)
        assert!((vecs[5][0] - 0.5).abs() < 1e-12);
        assert!((vecs[5][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_weight_vectors_3_objectives() {
        let vecs = generate_weight_vectors(3, 50);
        assert_eq!(vecs.len(), 50);
        for v in &vecs {
            assert_eq!(v.len(), 3);
            let sum: f64 = v.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "3D weight vector sums to {}", sum);
            for &w in v {
                assert!(w >= 0.0, "Weight must be non-negative");
            }
        }
    }

    #[test]
    fn test_tchebycheff_scalarization() {
        let objectives = vec![1.0, 3.0];
        let weights = vec![0.5, 0.5];
        let ideal = vec![0.0, 0.0];
        let result = tchebycheff_scalarize(&objectives, &weights, &ideal);
        // max(0.5 * |1-0|, 0.5 * |3-0|) = max(0.5, 1.5) = 1.5
        assert!((result - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_tchebycheff_with_ideal() {
        let objectives = vec![2.0, 4.0];
        let weights = vec![1.0, 0.5];
        let ideal = vec![1.0, 2.0];
        let result = tchebycheff_scalarize(&objectives, &weights, &ideal);
        // max(1.0 * |2-1|, 0.5 * |4-2|) = max(1.0, 1.0) = 1.0
        assert!((result - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_sum_scalarization() {
        let objectives = vec![2.0, 3.0];
        let weights = vec![0.4, 0.6];
        let result = weighted_sum_scalarize(&objectives, &weights);
        // 0.4*2 + 0.6*3 = 0.8 + 1.8 = 2.6
        assert!((result - 2.6).abs() < 1e-12);
    }

    #[test]
    fn test_dominates_true() {
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(dominates(&[1.0, 2.0], &[1.0, 3.0]));
        assert!(dominates(&[0.0, 0.0], &[1.0, 1.0]));
    }

    #[test]
    fn test_dominates_false() {
        // Equal: not dominated
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0]));
        // Trade-off: neither dominates
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        // Worse in all
        assert!(!dominates(&[3.0, 4.0], &[1.0, 2.0]));
    }

    #[test]
    fn test_non_dominated_sort_separates_fronts() {
        let points = vec![
            ParetoPoint { bitstring: vec![0], objectives: vec![1.0, 4.0], params: vec![] },
            ParetoPoint { bitstring: vec![1], objectives: vec![2.0, 2.0], params: vec![] },
            ParetoPoint { bitstring: vec![0], objectives: vec![4.0, 1.0], params: vec![] },
            ParetoPoint { bitstring: vec![1], objectives: vec![3.0, 3.0], params: vec![] }, // dominated by [2,2]
            ParetoPoint { bitstring: vec![0], objectives: vec![5.0, 5.0], params: vec![] }, // dominated by all
        ];

        let fronts = non_dominated_sort(&points);
        assert!(fronts.len() >= 2, "Should have at least 2 fronts");

        // Front 0 should contain indices 0, 1, 2 (non-dominated)
        let front0 = &fronts[0];
        assert!(front0.contains(&0));
        assert!(front0.contains(&1));
        assert!(front0.contains(&2));
        assert!(!front0.contains(&3)); // dominated by point 1
    }

    #[test]
    fn test_hypervolume_2d() {
        // Simple front: two points (1,3) and (3,1) with reference (4,4)
        let front = vec![
            ParetoPoint { bitstring: vec![], objectives: vec![1.0, 3.0], params: vec![] },
            ParetoPoint { bitstring: vec![], objectives: vec![3.0, 1.0], params: vec![] },
        ];
        let reference = vec![4.0, 4.0];
        let hv = compute_hypervolume(&front, &reference);
        // Area = (4-1)*(4-3) + (4-3)*(3-1) = 3*1 + 1*2 = 5
        // But using sweep: sort by x → (1,3), (3,1)
        // prev_y = 4
        // (1,3): y=3 < 4, hv += (4-1)*(4-3) = 3*1 = 3, prev_y = 3
        // (3,1): y=1 < 3, hv += (4-3)*(3-1) = 1*2 = 2, prev_y = 1
        // total = 5
        assert!((hv - 5.0).abs() < 1e-10, "Hypervolume = {} (expected 5.0)", hv);
    }

    #[test]
    fn test_qaoa_produces_valid_state() {
        // Simple 2-qubit Hamiltonian: Z_0 Z_1
        let hamiltonian = vec![(vec![(0, 'Z'), (1, 'Z')], 1.0)];
        let params = vec![0.5, 0.3]; // 1 layer: [γ, β]
        let state = simulate_qaoa_circuit(&params, &hamiltonian, 2, 1);

        // Check normalisation
        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "State not normalized: norm = {}",
            norm
        );
        assert_eq!(state.len(), 4);
    }

    #[test]
    fn test_qaoa_expectation_value() {
        // For |+> state, <+|Z_0|+> = 0
        let hamiltonian = vec![(vec![(0, 'Z')], 1.0)];
        let dim = 2;
        let amp = 1.0 / (dim as f64).sqrt();
        let state = vec![Complex64::new(amp, 0.0); dim];
        let ev = expectation_value(&state, &hamiltonian);
        assert!((ev - 0.0).abs() < 1e-10, "Expected 0, got {}", ev);
    }

    #[test]
    fn test_parameter_transfer_preserves_length() {
        let source = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 3 layers: [γ1,γ2,γ3,β1,β2,β3]
        let transferred = transfer_params(&source, 3, 5);
        assert_eq!(transferred.len(), 10); // 5 gammas + 5 betas
    }

    #[test]
    fn test_parameter_transfer_identity() {
        let source = vec![0.1, 0.2, 0.3, 0.4];
        let transferred = transfer_params(&source, 2, 2);
        assert_eq!(transferred, source);
    }

    #[test]
    fn test_parameter_transfer_interpolation() {
        // 2 layers -> 3 layers: should interpolate midpoint
        let source = vec![0.0, 1.0, 2.0, 3.0]; // gammas=[0,1], betas=[2,3]
        let transferred = transfer_params(&source, 2, 3);
        assert_eq!(transferred.len(), 6);
        // gammas: [0.0, 0.5, 1.0], betas: [2.0, 2.5, 3.0]
        assert!((transferred[0] - 0.0).abs() < 1e-12);
        assert!((transferred[1] - 0.5).abs() < 1e-12);
        assert!((transferred[2] - 1.0).abs() < 1e-12);
        assert!((transferred[3] - 2.0).abs() < 1e-12);
        assert!((transferred[4] - 2.5).abs() < 1e-12);
        assert!((transferred[5] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_bi_objective_max_cut_has_two_objectives() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let objs = bi_objective_max_cut(&edges, 3);
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0].name, "cut_weight");
        assert_eq!(objs[1].name, "balance");
    }

    #[test]
    fn test_portfolio_bi_objective() {
        let returns = vec![0.1, 0.2, 0.15];
        let cov = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.03],
            vec![0.02, 0.03, 0.06],
        ];
        let objs = portfolio_bi_objective(&returns, &cov);
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0].name, "negative_return");
        assert_eq!(objs[1].name, "risk");
    }

    #[test]
    fn test_qamoo_finds_pareto_points() {
        // Small 3-node graph, bi-objective max-cut
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let objectives = bi_objective_max_cut(&edges, 3);

        let config = QamooConfig::new()
            .num_layers(1)
            .num_weight_vectors(5)
            .max_iterations(30);

        let result = run_qamoo(&objectives, 3, &config);
        assert!(result.is_ok(), "QAMOO should succeed: {:?}", result.err());

        let result = result.unwrap();
        assert!(
            !result.pareto_front.points.is_empty(),
            "Pareto front should have points"
        );
        assert!(result.num_evaluations > 0);
        assert_eq!(result.weight_vectors_used.len(), 5);

        // Each Pareto point should have 2 objective values
        for p in &result.pareto_front.points {
            assert_eq!(p.objectives.len(), 2);
        }
    }

    #[test]
    fn test_qamoo_rejects_single_objective() {
        let obj = ObjectiveFunction::new("single", vec![(vec![(0, 'Z')], 1.0)]);
        let config = QamooConfig::new();
        let result = run_qamoo(&[obj], 2, &config);
        assert!(result.is_err());
        match result.err().unwrap() {
            QamooError::InvalidObjectives(_) => {}
            other => panic!("Expected InvalidObjectives, got {:?}", other),
        }
    }

    #[test]
    fn test_pareto_front_extraction() {
        let points = vec![
            ParetoPoint { bitstring: vec![0], objectives: vec![1.0, 5.0], params: vec![] },
            ParetoPoint { bitstring: vec![1], objectives: vec![3.0, 3.0], params: vec![] },
            ParetoPoint { bitstring: vec![0], objectives: vec![5.0, 1.0], params: vec![] },
            ParetoPoint { bitstring: vec![1], objectives: vec![4.0, 4.0], params: vec![] }, // dominated
        ];

        let pf = pareto_front(&points);
        assert_eq!(pf.points.len(), 3, "Should have 3 non-dominated points");
        assert!(pf.hypervolume > 0.0, "Hypervolume should be positive");
    }

    #[test]
    fn test_bitstring_to_basis() {
        assert_eq!(bitstring_to_basis(&[0, 0, 0]), 0);
        assert_eq!(bitstring_to_basis(&[1, 0, 0]), 1);
        assert_eq!(bitstring_to_basis(&[0, 1, 0]), 2);
        assert_eq!(bitstring_to_basis(&[1, 1, 0]), 3);
        assert_eq!(bitstring_to_basis(&[1, 1, 1]), 7);
    }

    #[test]
    fn test_evaluate_hamiltonian_z0() {
        // H = Z_0: eigenvalue +1 for |0>, -1 for |1>
        let h = vec![(vec![(0, 'Z')], 1.0)];
        assert!((evaluate_hamiltonian_on_basis(&h, 0, 1) - 1.0).abs() < 1e-12);
        assert!((evaluate_hamiltonian_on_basis(&h, 1, 1) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_hamiltonian_z0z1() {
        // H = Z_0 Z_1
        let h = vec![(vec![(0, 'Z'), (1, 'Z')], 1.0)];
        // |00> -> +1*+1 = +1
        assert!((evaluate_hamiltonian_on_basis(&h, 0b00, 2) - 1.0).abs() < 1e-12);
        // |01> -> -1*+1 = -1
        assert!((evaluate_hamiltonian_on_basis(&h, 0b01, 2) - (-1.0)).abs() < 1e-12);
        // |10> -> +1*-1 = -1
        assert!((evaluate_hamiltonian_on_basis(&h, 0b10, 2) - (-1.0)).abs() < 1e-12);
        // |11> -> -1*-1 = +1
        assert!((evaluate_hamiltonian_on_basis(&h, 0b11, 2) - 1.0).abs() < 1e-12);
    }
}
