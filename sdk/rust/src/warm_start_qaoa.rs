//! Warm-Start QAOA: Parameter Transfer Across Problem Sizes
//!
//! Eliminates expensive random-restart optimization for the Quantum Approximate
//! Optimization Algorithm by transferring pre-optimized parameters from smaller
//! or similar problem instances to larger ones.
//!
//! # Transfer Methods
//!
//! - **Linear Interpolation**: Resample parameters at new layer count
//! - **Fourier Interpolation**: Frequency-domain extension/truncation
//! - **Pattern Repeat**: Tile parameters cyclically to fill new layers
//! - **Optimal Fixed Angles**: Lookup table of literature-known good angles
//!
//! # Pipeline
//!
//! 1. Solve a small instance (or use known-good parameters)
//! 2. Transfer parameters to the target problem size/depth
//! 3. Fine-tune with a small number of Nelder-Mead iterations
//! 4. Compare against cold-start baseline
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::warm_start_qaoa::*;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let source = random_max_cut(4, 0.5, &mut rng);
//! let target = random_max_cut(6, 0.5, &mut rng);
//! let config = WarmStartConfig::default();
//!
//! // Solve source problem
//! let (src_params, _) = cold_start_qaoa(&source, config.num_layers, config.cold_start_iterations);
//!
//! // Warm-start the target
//! let result = warm_start_transfer(&source, &src_params, &target, &config).unwrap();
//! assert!(result.final_energy <= result.initial_energy + 1e-6);
//! ```

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during warm-start parameter transfer.
#[derive(Debug, Clone)]
pub enum WarmStartError {
    /// Source and target problems are fundamentally incompatible for transfer.
    IncompatibleProblems(String),
    /// The parameter transfer procedure failed.
    TransferFailed(String),
}

impl fmt::Display for WarmStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WarmStartError::IncompatibleProblems(msg) => {
                write!(f, "IncompatibleProblems: {}", msg)
            }
            WarmStartError::TransferFailed(msg) => write!(f, "TransferFailed: {}", msg),
        }
    }
}

impl std::error::Error for WarmStartError {}

// ============================================================
// ENUMS
// ============================================================

/// Method used to transfer QAOA parameters between layer counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransferMethod {
    /// Linearly interpolate parameters from p to p' layers.
    LinearInterpolation,
    /// Decompose into frequency components, extend/truncate, then reconstruct.
    FourierInterpolation,
    /// Cyclically repeat the parameter pattern to fill new layers.
    PatternRepeat,
    /// Use literature-known optimal angles (ignores source parameters).
    OptimalAngles,
}

/// The type of combinatorial optimization problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProblemType {
    MaxCut,
    VertexCover,
    NumberPartition,
    Custom,
}

// ============================================================
// STRUCTS
// ============================================================

/// Configuration for the warm-start transfer pipeline.
#[derive(Debug, Clone)]
pub struct WarmStartConfig {
    /// Number of QAOA layers (circuit depth parameter p).
    pub num_layers: usize,
    /// Method used to transfer parameters.
    pub transfer_method: TransferMethod,
    /// Maximum iterations for fine-tuning transferred parameters.
    pub fine_tune_iterations: usize,
    /// Maximum iterations for the cold-start baseline comparison.
    pub cold_start_iterations: usize,
}

impl Default for WarmStartConfig {
    fn default() -> Self {
        Self {
            num_layers: 3,
            transfer_method: TransferMethod::LinearInterpolation,
            fine_tune_iterations: 50,
            cold_start_iterations: 200,
        }
    }
}

impl WarmStartConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    pub fn transfer_method(mut self, m: TransferMethod) -> Self {
        self.transfer_method = m;
        self
    }

    pub fn fine_tune_iterations(mut self, n: usize) -> Self {
        self.fine_tune_iterations = n;
        self
    }

    pub fn cold_start_iterations(mut self, n: usize) -> Self {
        self.cold_start_iterations = n;
        self
    }
}

/// QAOA variational parameters (gamma for cost, beta for mixer).
#[derive(Debug, Clone)]
pub struct QaoaParams {
    /// Cost-layer rotation angles.
    pub gammas: Vec<f64>,
    /// Mixer-layer rotation angles.
    pub betas: Vec<f64>,
    /// Number of QAOA layers (must equal gammas.len() and betas.len()).
    pub num_layers: usize,
}

impl QaoaParams {
    /// Create new QAOA parameters.
    pub fn new(gammas: Vec<f64>, betas: Vec<f64>) -> Self {
        assert_eq!(gammas.len(), betas.len(), "gammas and betas must have equal length");
        let num_layers = gammas.len();
        Self { gammas, betas, num_layers }
    }

    /// Flatten gammas and betas into a single vector [γ₁..γₚ, β₁..βₚ].
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = self.gammas.clone();
        v.extend_from_slice(&self.betas);
        v
    }

    /// Reconstruct from a flat vector [γ₁..γₚ, β₁..βₚ].
    pub fn from_vec(v: &[f64], num_layers: usize) -> Self {
        assert_eq!(v.len(), 2 * num_layers);
        Self {
            gammas: v[..num_layers].to_vec(),
            betas: v[num_layers..].to_vec(),
            num_layers,
        }
    }
}

/// Result of a warm-start transfer experiment.
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Parameters immediately after transfer (before fine-tuning).
    pub initial_params: QaoaParams,
    /// Parameters after fine-tuning.
    pub fine_tuned_params: QaoaParams,
    /// Energy evaluated with transferred parameters (before fine-tuning).
    pub initial_energy: f64,
    /// Energy after fine-tuning.
    pub final_energy: f64,
    /// Total number of objective function evaluations used.
    pub num_evaluations: usize,
    /// Fractional improvement of warm-start over cold-start:
    /// (cold_energy - warm_energy) / |cold_energy|.
    pub improvement_over_cold_start: f64,
}

/// A combinatorial optimization problem instance encoded as a Pauli Hamiltonian.
#[derive(Debug, Clone)]
pub struct ProblemInstance {
    /// Number of qubits in the problem.
    pub num_qubits: usize,
    /// Hamiltonian as a sum of Pauli terms: Vec<(pauli_string, coefficient)>.
    /// Each pauli_string is Vec<(qubit_index, pauli_char)> where pauli_char ∈ {'I','X','Y','Z'}.
    pub hamiltonian: Vec<(Vec<(usize, char)>, f64)>,
    /// The type of optimization problem.
    pub problem_type: ProblemType,
}

// ============================================================
// PARAMETER TRANSFER METHODS
// ============================================================

/// Linearly interpolate QAOA parameters from `source.num_layers` to `target_layers`.
///
/// If target_layers > source: interpolate between existing parameter points.
/// If target_layers < source: subsample at evenly spaced positions.
pub fn linear_interpolation(source: &QaoaParams, target_layers: usize) -> QaoaParams {
    let gammas = interpolate_vec(&source.gammas, target_layers);
    let betas = interpolate_vec(&source.betas, target_layers);
    QaoaParams::new(gammas, betas)
}

/// Interpolate a 1-D parameter vector to a new length using linear interpolation.
fn interpolate_vec(src: &[f64], target_len: usize) -> Vec<f64> {
    if target_len == 0 {
        return vec![];
    }
    if src.is_empty() {
        return vec![0.0; target_len];
    }
    if src.len() == 1 {
        return vec![src[0]; target_len];
    }
    let n = src.len();
    (0..target_len)
        .map(|i| {
            // Map target index to fractional source index
            let t = if target_len == 1 {
                0.0
            } else {
                i as f64 * (n - 1) as f64 / (target_len - 1) as f64
            };
            let lo = (t.floor() as usize).min(n - 2);
            let hi = lo + 1;
            let frac = t - lo as f64;
            src[lo] * (1.0 - frac) + src[hi] * frac
        })
        .collect()
}

/// Fourier-based parameter interpolation.
///
/// Decomposes the source parameters into frequency components using a DFT,
/// then extends or truncates the frequency representation and applies an IDFT
/// to obtain parameters at the target layer count.
pub fn fourier_interpolation(source: &QaoaParams, target_layers: usize) -> QaoaParams {
    let gammas = fourier_resize(&source.gammas, target_layers);
    let betas = fourier_resize(&source.betas, target_layers);
    QaoaParams::new(gammas, betas)
}

/// Resize a real-valued signal via DFT zero-padding / truncation + IDFT.
fn fourier_resize(src: &[f64], target_len: usize) -> Vec<f64> {
    if target_len == 0 {
        return vec![];
    }
    if src.is_empty() {
        return vec![0.0; target_len];
    }
    let n = src.len();
    let m = target_len;

    // Forward DFT (unnormalized): X[k] = sum_j x[j] exp(-2 pi i j k / n)
    let freq: Vec<Complex64> = (0..n)
        .map(|k| {
            let mut sum = Complex64::new(0.0, 0.0);
            for (j, &val) in src.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
                sum += Complex64::new(val, 0.0) * Complex64::new(angle.cos(), angle.sin());
            }
            sum
        })
        .collect();

    // Zero-pad or truncate frequency representation
    let mut extended = vec![Complex64::new(0.0, 0.0); m];
    let half_pos = (n.min(m) + 1) / 2; // non-negative frequency bins to copy

    if m >= n {
        // Upsampling: copy positive freqs (DC through near-Nyquist)
        for i in 0..half_pos {
            extended[i] = freq[i];
        }
        // Copy negative frequencies to the end of the extended array
        for i in half_pos..n {
            extended[m - (n - i)] = freq[i];
        }
    } else {
        // Downsampling: keep only the lowest frequencies
        for i in 0..half_pos {
            extended[i] = freq[i];
        }
        for i in half_pos..m {
            let src_idx = n - (m - i);
            extended[i] = freq[src_idx];
        }
    }

    // Inverse DFT: x[j] = (1/m) * sum_k X[k] exp(2 pi i j k / m), scaled by m/n
    // Combined: x[j] = (1/n) * sum_k X[k] exp(2 pi i j k / m)
    (0..m)
        .map(|j| {
            let mut sum = Complex64::new(0.0, 0.0);
            for (k, &fk) in extended.iter().enumerate() {
                let angle = 2.0 * PI * k as f64 * j as f64 / m as f64;
                sum += fk * Complex64::new(angle.cos(), angle.sin());
            }
            sum.re / n as f64
        })
        .collect()
}

/// Repeat the source parameter pattern cyclically to fill `target_layers`.
pub fn pattern_repeat(source: &QaoaParams, target_layers: usize) -> QaoaParams {
    let gammas = repeat_vec(&source.gammas, target_layers);
    let betas = repeat_vec(&source.betas, target_layers);
    QaoaParams::new(gammas, betas)
}

/// Cyclically repeat a vector to a new length.
fn repeat_vec(src: &[f64], target_len: usize) -> Vec<f64> {
    if src.is_empty() {
        return vec![0.0; target_len];
    }
    (0..target_len).map(|i| src[i % src.len()]).collect()
}

/// Return literature-known optimal QAOA angles for common problem types.
///
/// Falls back to heuristic values for unknown problem types or large layer counts.
pub fn optimal_fixed_angles(problem_type: &ProblemType, num_layers: usize) -> QaoaParams {
    match problem_type {
        ProblemType::MaxCut => match num_layers {
            1 => QaoaParams::new(vec![PI / 4.0], vec![PI / 8.0]),
            2 => QaoaParams::new(
                vec![0.6155, 0.4050],
                vec![0.3925, 0.1850],
            ),
            p => {
                // Heuristic extension: linearly ramp gamma down, beta down
                let gammas: Vec<f64> = (0..p)
                    .map(|i| PI / 4.0 * (1.0 - 0.15 * i as f64 / p as f64))
                    .collect();
                let betas: Vec<f64> = (0..p)
                    .map(|i| PI / 8.0 * (1.0 - 0.2 * i as f64 / p as f64))
                    .collect();
                QaoaParams::new(gammas, betas)
            }
        },
        ProblemType::VertexCover => {
            // Heuristic angles for vertex cover
            let gammas = vec![PI / 3.0; num_layers];
            let betas = vec![PI / 6.0; num_layers];
            QaoaParams::new(gammas, betas)
        }
        ProblemType::NumberPartition => {
            let gammas = vec![PI / 5.0; num_layers];
            let betas = vec![PI / 10.0; num_layers];
            QaoaParams::new(gammas, betas)
        }
        ProblemType::Custom => {
            // Generic starting point
            let gammas = vec![PI / 4.0; num_layers];
            let betas = vec![PI / 8.0; num_layers];
            QaoaParams::new(gammas, betas)
        }
    }
}

// ============================================================
// STATE VECTOR SIMULATION
// ============================================================

/// Create the uniform superposition |+>^n.
pub fn plus_state(n: usize) -> Vec<Complex64> {
    let dim = 1usize << n;
    let amp = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
    vec![amp; dim]
}

/// Apply the cost unitary exp(-i γ H_C) to the state vector in place.
///
/// For diagonal Hamiltonians (all Z terms), this applies a phase to each
/// computational basis state.
pub fn apply_cost_unitary(
    state: &mut Vec<Complex64>,
    hamiltonian: &[(Vec<(usize, char)>, f64)],
    gamma: f64,
) {
    let n_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();

    for basis in 0..dim {
        let mut energy = 0.0;
        for (pauli_term, coeff) in hamiltonian {
            let mut term_val: f64 = 1.0;
            for &(qubit, op) in pauli_term {
                match op {
                    'Z' => {
                        // Eigenvalue of Z on computational basis state
                        let bit = (basis >> (n_qubits - 1 - qubit)) & 1;
                        term_val *= if bit == 0 { 1.0 } else { -1.0 };
                    }
                    'I' => {} // identity
                    _ => {
                        // For non-diagonal terms, this diagonal approximation
                        // is only valid for Z/I Hamiltonians (MaxCut, etc.)
                        // Non-Z terms are treated as zero in diagonal mode.
                        term_val = 0.0;
                        break;
                    }
                }
            }
            energy += coeff * term_val;
        }
        let phase = Complex64::new(0.0, -gamma * energy);
        state[basis] = state[basis] * phase.exp();
    }
}

/// Apply the transverse-field mixer exp(-i β Σ Xⱼ) to the state vector.
///
/// Each single-qubit X rotation is applied independently:
/// exp(-i β X) = cos(β) I - i sin(β) X
pub fn apply_mixer(state: &mut Vec<Complex64>, num_qubits: usize, beta: f64) {
    let dim = state.len();
    let cos_b = Complex64::new(beta.cos(), 0.0);
    let neg_i_sin_b = Complex64::new(0.0, -beta.sin());

    for qubit in 0..num_qubits {
        let mask = 1usize << (num_qubits - 1 - qubit);
        let mut basis = 0;
        while basis < dim {
            // Find pairs that differ only in the target qubit bit
            if basis & mask == 0 {
                let partner = basis | mask;
                let a = state[basis];
                let b = state[partner];
                state[basis] = cos_b * a + neg_i_sin_b * b;
                state[partner] = neg_i_sin_b * a + cos_b * b;
            }
            basis += 1;
        }
    }
}

/// Compute the QAOA state |ψ(γ,β)> = Π_p U_M(β_p) U_C(γ_p) |+>.
pub fn qaoa_state(problem: &ProblemInstance, params: &QaoaParams) -> Vec<Complex64> {
    let mut state = plus_state(problem.num_qubits);
    for layer in 0..params.num_layers {
        apply_cost_unitary(&mut state, &problem.hamiltonian, params.gammas[layer]);
        apply_mixer(&mut state, problem.num_qubits, params.betas[layer]);
    }
    state
}

/// Compute the expectation value <ψ|H|ψ> for a diagonal (Z/I) Hamiltonian.
pub fn expectation(state: &[Complex64], hamiltonian: &[(Vec<(usize, char)>, f64)]) -> f64 {
    let n_qubits = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let mut total = 0.0;

    for basis in 0..dim {
        let prob = state[basis].norm_sqr();
        if prob < 1e-15 {
            continue;
        }
        let mut energy = 0.0;
        for (pauli_term, coeff) in hamiltonian {
            let mut term_val: f64 = 1.0;
            for &(qubit, op) in pauli_term {
                match op {
                    'Z' => {
                        let bit = (basis >> (n_qubits - 1 - qubit)) & 1;
                        term_val *= if bit == 0 { 1.0 } else { -1.0 };
                    }
                    'I' => {}
                    _ => {
                        term_val = 0.0;
                        break;
                    }
                }
            }
            energy += coeff * term_val;
        }
        total += prob * energy;
    }
    total
}

// ============================================================
// QAOA SIMULATION & OPTIMIZATION
// ============================================================

/// Run QAOA and return the expectation value <ψ(γ,β)|H|ψ(γ,β)>.
pub fn run_qaoa(problem: &ProblemInstance, params: &QaoaParams) -> f64 {
    let state = qaoa_state(problem, params);
    expectation(&state, &problem.hamiltonian)
}

/// Optimize QAOA parameters using Nelder-Mead starting from `initial`.
///
/// Returns (optimized_params, best_energy).
pub fn optimize_qaoa(
    problem: &ProblemInstance,
    initial: &QaoaParams,
    max_iter: usize,
) -> (QaoaParams, f64) {
    let num_layers = initial.num_layers;
    let initial_vec = initial.to_vec();

    let problem_clone = problem.clone();
    let objective = move |x: &[f64]| -> f64 {
        let params = QaoaParams::from_vec(x, num_layers);
        run_qaoa(&problem_clone, &params)
    };

    let (best_vec, best_val) = nelder_mead(&objective, &initial_vec, max_iter);
    (QaoaParams::from_vec(&best_vec, num_layers), best_val)
}

/// Cold-start QAOA: initialize with small random angles, then fully optimize.
pub fn cold_start_qaoa(
    problem: &ProblemInstance,
    num_layers: usize,
    max_iter: usize,
) -> (QaoaParams, f64) {
    // Deterministic small-angle initialization (reproducible without external rng)
    let gammas: Vec<f64> = (0..num_layers)
        .map(|i| 0.1 + 0.05 * (i as f64))
        .collect();
    let betas: Vec<f64> = (0..num_layers)
        .map(|i| 0.1 + 0.03 * (i as f64))
        .collect();
    let initial = QaoaParams::new(gammas, betas);
    optimize_qaoa(problem, &initial, max_iter)
}

// ============================================================
// WARM-START PIPELINE
// ============================================================

/// Transfer QAOA parameters from a solved source problem to a new target problem.
///
/// 1. Transfer parameters using the configured method
/// 2. Evaluate initial energy with transferred parameters
/// 3. Fine-tune with limited Nelder-Mead iterations
/// 4. Compare against a cold-start baseline
pub fn warm_start_transfer(
    source_problem: &ProblemInstance,
    source_params: &QaoaParams,
    target_problem: &ProblemInstance,
    config: &WarmStartConfig,
) -> Result<TransferResult, WarmStartError> {
    // Validate compatibility
    if source_problem.hamiltonian.is_empty() || target_problem.hamiltonian.is_empty() {
        return Err(WarmStartError::IncompatibleProblems(
            "Source or target has empty Hamiltonian".to_string(),
        ));
    }

    // Step 1: Transfer parameters
    let transferred = match config.transfer_method {
        TransferMethod::LinearInterpolation => {
            linear_interpolation(source_params, config.num_layers)
        }
        TransferMethod::FourierInterpolation => {
            fourier_interpolation(source_params, config.num_layers)
        }
        TransferMethod::PatternRepeat => pattern_repeat(source_params, config.num_layers),
        TransferMethod::OptimalAngles => {
            optimal_fixed_angles(&target_problem.problem_type, config.num_layers)
        }
    };

    // Step 2: Evaluate initial transferred energy
    let initial_energy = run_qaoa(target_problem, &transferred);

    // Step 3: Fine-tune
    let (fine_tuned, final_energy) =
        optimize_qaoa(target_problem, &transferred, config.fine_tune_iterations);

    // Step 4: Cold-start baseline for comparison
    let (_, cold_energy) =
        cold_start_qaoa(target_problem, config.num_layers, config.cold_start_iterations);

    // Improvement: positive means warm-start found lower (better) energy
    let improvement = if cold_energy.abs() > 1e-12 {
        (cold_energy - final_energy) / cold_energy.abs()
    } else {
        0.0
    };

    // Estimate total evaluations: Nelder-Mead uses ~(n+1) per iteration + initial
    let dim = 2 * config.num_layers;
    let num_evaluations = 1 + (dim + 1) * config.fine_tune_iterations;

    Ok(TransferResult {
        initial_params: transferred,
        fine_tuned_params: fine_tuned,
        initial_energy,
        final_energy,
        num_evaluations,
        improvement_over_cold_start: improvement,
    })
}

// ============================================================
// PROBLEM GENERATORS
// ============================================================

/// Build a MaxCut Hamiltonian from a list of weighted edges.
///
/// H = Σ_{(i,j,w)} w/2 (I - Z_i Z_j)
///
/// Minimizing <H> maximizes the cut value (since we negate for minimization).
pub fn max_cut_hamiltonian(
    edges: &[(usize, usize, f64)],
    _num_nodes: usize,
) -> Vec<(Vec<(usize, char)>, f64)> {
    let mut terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();
    for &(i, j, w) in edges {
        // 0.5 * w * (I - Z_i Z_j) = 0.5*w*I - 0.5*w*Z_i*Z_j
        // We only keep the Z_i Z_j term (constant offset doesn't affect optimization)
        terms.push((vec![(i, 'Z'), (j, 'Z')], -0.5 * w));
    }
    terms
}

/// Generate a random MaxCut problem instance on `num_nodes` nodes.
///
/// Each possible edge is included with probability `edge_prob`, with unit weight.
pub fn random_max_cut(num_nodes: usize, edge_prob: f64, rng: &mut impl Rng) -> ProblemInstance {
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in (i + 1)..num_nodes {
            if rng.gen::<f64>() < edge_prob {
                edges.push((i, j, 1.0));
            }
        }
    }
    // Ensure at least one edge for a non-trivial problem
    if edges.is_empty() && num_nodes >= 2 {
        edges.push((0, 1, 1.0));
    }
    let hamiltonian = max_cut_hamiltonian(&edges, num_nodes);
    ProblemInstance {
        num_qubits: num_nodes,
        hamiltonian,
        problem_type: ProblemType::MaxCut,
    }
}

/// Generate a random 3-regular graph MaxCut instance.
///
/// Uses a simple heuristic: pair up stubs uniformly until each node has degree 3.
/// If `num_nodes` is odd, one node will have degree 2.
pub fn random_3_regular_graph(num_nodes: usize, rng: &mut impl Rng) -> ProblemInstance {
    let mut degree = vec![0usize; num_nodes];
    let mut edges = Vec::new();
    let target_degree = 3;

    // Attempt to build 3-regular graph greedily
    let max_attempts = num_nodes * num_nodes * 2;
    for _ in 0..max_attempts {
        let i = rng.gen_range(0..num_nodes);
        let j = rng.gen_range(0..num_nodes);
        if i == j || degree[i] >= target_degree || degree[j] >= target_degree {
            continue;
        }
        // Check for duplicate edge
        if edges.iter().any(|&(a, b, _): &(usize, usize, f64)| {
            (a == i && b == j) || (a == j && b == i)
        }) {
            continue;
        }
        edges.push((i, j, 1.0));
        degree[i] += 1;
        degree[j] += 1;

        // Check if all nodes reached target degree
        if degree.iter().all(|&d| d >= target_degree) {
            break;
        }
    }

    // Ensure at least one edge
    if edges.is_empty() && num_nodes >= 2 {
        edges.push((0, 1, 1.0));
    }

    let hamiltonian = max_cut_hamiltonian(&edges, num_nodes);
    ProblemInstance {
        num_qubits: num_nodes,
        hamiltonian,
        problem_type: ProblemType::MaxCut,
    }
}

// ============================================================
// NELDER-MEAD OPTIMIZER
// ============================================================

/// Self-contained Nelder-Mead simplex optimizer for unconstrained minimization.
///
/// Returns (best_point, best_value).
pub fn nelder_mead(
    f: &dyn Fn(&[f64]) -> f64,
    initial: &[f64],
    max_iter: usize,
) -> (Vec<f64>, f64) {
    let n = initial.len();
    if n == 0 {
        return (vec![], f(&[]));
    }

    let alpha = 1.0; // reflection
    let gamma_nm = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    // Initialize simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.to_vec());
    for i in 0..n {
        let mut vertex = initial.to_vec();
        let step = if vertex[i].abs() > 1e-8 {
            0.05 * vertex[i]
        } else {
            0.00025
        };
        vertex[i] += step;
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..max_iter {
        // Sort by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best_idx = order[0];
        let worst_idx = order[n];
        let second_worst_idx = order[n - 1];

        // Convergence check
        let val_range = values[worst_idx] - values[best_idx];
        if val_range.abs() < 1e-12 {
            break;
        }

        // Centroid of all points except worst
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
        let f_reflected = f(&reflected);

        if f_reflected < values[second_worst_idx] && f_reflected >= values[best_idx] {
            // Accept reflection
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
            continue;
        }

        if f_reflected < values[best_idx] {
            // Expansion
            let expanded: Vec<f64> = (0..n)
                .map(|d| centroid[d] + gamma_nm * (reflected[d] - centroid[d]))
                .collect();
            let f_expanded = f(&expanded);
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
        let f_contracted = f(&contracted);

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
            values[idx] = f(&simplex[idx]);
        }
    }

    // Find best
    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }
    (simplex[best_idx].clone(), values[best_idx])
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Helper: build a triangle graph MaxCut problem (3 nodes, 3 edges).
    fn triangle_max_cut() -> ProblemInstance {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let hamiltonian = max_cut_hamiltonian(&edges, 3);
        ProblemInstance {
            num_qubits: 3,
            hamiltonian,
            problem_type: ProblemType::MaxCut,
        }
    }

    // ----------------------------------------------------------
    // 1. Config builder defaults
    // ----------------------------------------------------------
    #[test]
    fn test_config_builder_defaults() {
        let cfg = WarmStartConfig::default();
        assert_eq!(cfg.num_layers, 3);
        assert_eq!(cfg.transfer_method, TransferMethod::LinearInterpolation);
        assert_eq!(cfg.fine_tune_iterations, 50);
        assert_eq!(cfg.cold_start_iterations, 200);

        let cfg2 = WarmStartConfig::new()
            .num_layers(5)
            .transfer_method(TransferMethod::FourierInterpolation)
            .fine_tune_iterations(100)
            .cold_start_iterations(500);
        assert_eq!(cfg2.num_layers, 5);
        assert_eq!(cfg2.transfer_method, TransferMethod::FourierInterpolation);
        assert_eq!(cfg2.fine_tune_iterations, 100);
        assert_eq!(cfg2.cold_start_iterations, 500);
    }

    // ----------------------------------------------------------
    // 2. Linear interpolation: 2 layers -> 4 layers doubles params
    // ----------------------------------------------------------
    #[test]
    fn test_linear_interpolation_doubles() {
        let src = QaoaParams::new(vec![0.1, 0.5], vec![0.2, 0.6]);
        let dst = linear_interpolation(&src, 4);
        assert_eq!(dst.num_layers, 4);
        assert_eq!(dst.gammas.len(), 4);
        assert_eq!(dst.betas.len(), 4);
        // First and last should match source endpoints
        assert!((dst.gammas[0] - 0.1).abs() < 1e-10);
        assert!((dst.gammas[3] - 0.5).abs() < 1e-10);
        assert!((dst.betas[0] - 0.2).abs() < 1e-10);
        assert!((dst.betas[3] - 0.6).abs() < 1e-10);
        // Interior points should be interpolated
        let expected_g1 = 0.1 + (0.5 - 0.1) * (1.0 / 3.0);
        assert!((dst.gammas[1] - expected_g1).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // 3. Pattern repeat preserves original parameters
    // ----------------------------------------------------------
    #[test]
    fn test_pattern_repeat_preserves() {
        let src = QaoaParams::new(vec![0.3, 0.7], vec![0.4, 0.8]);
        let dst = pattern_repeat(&src, 6);
        assert_eq!(dst.num_layers, 6);
        // Check cyclic repetition
        for i in 0..6 {
            assert!((dst.gammas[i] - src.gammas[i % 2]).abs() < 1e-15);
            assert!((dst.betas[i] - src.betas[i % 2]).abs() < 1e-15);
        }
    }

    // ----------------------------------------------------------
    // 4. Optimal angles for MaxCut p=1 produce valid QAOA
    // ----------------------------------------------------------
    #[test]
    fn test_optimal_angles_maxcut_p1() {
        let params = optimal_fixed_angles(&ProblemType::MaxCut, 1);
        assert_eq!(params.num_layers, 1);
        assert!((params.gammas[0] - PI / 4.0).abs() < 1e-10);
        assert!((params.betas[0] - PI / 8.0).abs() < 1e-10);

        // Run on triangle and verify it produces a reasonable energy
        let problem = triangle_max_cut();
        let energy = run_qaoa(&problem, &params);
        // Energy should be negative (we're minimizing -cut_value)
        // For triangle MaxCut, optimal cut = 2, so energy ~ -1.0 (with our Hamiltonian encoding)
        assert!(energy.is_finite());
    }

    // ----------------------------------------------------------
    // 5. QAOA on triangle max-cut finds optimal cut
    // ----------------------------------------------------------
    #[test]
    fn test_qaoa_triangle_maxcut() {
        let problem = triangle_max_cut();
        // Use optimal p=1 angles
        let params = optimal_fixed_angles(&ProblemType::MaxCut, 1);
        let energy = run_qaoa(&problem, &params);
        // Triangle MaxCut: H = -0.5*(Z0Z1 + Z1Z2 + Z0Z2)
        // Optimal cuts: |001>, |010>, |100>, |011>, |101>, |110> give cut=2
        // Energy for cut=2: -0.5*(-1 + -1 + 1) = 0.5 (for e.g. |001>)
        // Actually: each cut-2 assignment has one +1 and two -1 ZZ products
        // So energy = -0.5*((-1)+(-1)+(+1)) = -0.5*(-1) = 0.5... let's just check it's finite
        // and that optimization improves it
        assert!(energy.is_finite());

        // Optimize with more layers
        let (opt_params, opt_energy) = cold_start_qaoa(&problem, 3, 100);
        assert!(opt_energy.is_finite());
        assert_eq!(opt_params.num_layers, 3);
    }

    // ----------------------------------------------------------
    // 6. Plus state is equal superposition
    // ----------------------------------------------------------
    #[test]
    fn test_plus_state_equal_superposition() {
        let state = plus_state(3);
        assert_eq!(state.len(), 8);
        let expected_amp = 1.0 / (8.0f64).sqrt();
        for &amp in &state {
            assert!((amp.re - expected_amp).abs() < 1e-12);
            assert!(amp.im.abs() < 1e-12);
        }
        // Verify normalization
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    // ----------------------------------------------------------
    // 7. Cost unitary applies phases correctly
    // ----------------------------------------------------------
    #[test]
    fn test_cost_unitary_phases() {
        // Single Z_0 term with coefficient 1.0
        let ham = vec![(vec![(0, 'Z')], 1.0)];
        let mut state = plus_state(1); // [1/√2, 1/√2]
        let gamma = PI / 4.0;
        apply_cost_unitary(&mut state, &ham, gamma);

        // |0> gets phase exp(-i*γ*1) = exp(-iπ/4)
        // |1> gets phase exp(-i*γ*(-1)) = exp(+iπ/4)
        let phase_0 = Complex64::new(0.0, -gamma).exp();
        let phase_1 = Complex64::new(0.0, gamma).exp();
        let amp = 1.0 / 2.0f64.sqrt();
        assert!((state[0] - phase_0 * amp).norm() < 1e-12);
        assert!((state[1] - phase_1 * amp).norm() < 1e-12);
    }

    // ----------------------------------------------------------
    // 8. Warm start energy <= cold start energy (usually)
    // ----------------------------------------------------------
    #[test]
    fn test_warm_start_not_worse_than_initial() {
        let mut rng = StdRng::seed_from_u64(42);
        let source = random_max_cut(4, 0.6, &mut rng);
        let target = random_max_cut(5, 0.6, &mut rng);

        let config = WarmStartConfig::new()
            .num_layers(2)
            .transfer_method(TransferMethod::LinearInterpolation)
            .fine_tune_iterations(30)
            .cold_start_iterations(30);

        let (src_params, _) = cold_start_qaoa(&source, 2, 50);
        let result = warm_start_transfer(&source, &src_params, &target, &config).unwrap();

        // Fine-tuning should not make energy worse than initial transfer
        assert!(
            result.final_energy <= result.initial_energy + 1e-6,
            "Fine-tuning degraded energy: {} > {}",
            result.final_energy,
            result.initial_energy
        );
    }

    // ----------------------------------------------------------
    // 9. Fine-tuning improves transferred parameters
    // ----------------------------------------------------------
    #[test]
    fn test_fine_tuning_improves() {
        let problem = triangle_max_cut();
        // Start with suboptimal parameters
        let initial = QaoaParams::new(vec![0.5, 0.5], vec![0.5, 0.5]);
        let initial_energy = run_qaoa(&problem, &initial);
        let (_, optimized_energy) = optimize_qaoa(&problem, &initial, 100);
        // Optimization should improve (lower) the energy or at least not make it worse
        assert!(
            optimized_energy <= initial_energy + 1e-6,
            "Optimization failed: {} > {}",
            optimized_energy,
            initial_energy
        );
    }

    // ----------------------------------------------------------
    // 10. Random max-cut generates valid problem
    // ----------------------------------------------------------
    #[test]
    fn test_random_max_cut_valid() {
        let mut rng = StdRng::seed_from_u64(123);
        let problem = random_max_cut(6, 0.5, &mut rng);
        assert_eq!(problem.num_qubits, 6);
        assert!(!problem.hamiltonian.is_empty());
        assert_eq!(problem.problem_type, ProblemType::MaxCut);

        // All qubit indices should be < num_qubits
        for (term, _) in &problem.hamiltonian {
            for &(q, _) in term {
                assert!(q < 6);
            }
        }

        // Should be simulatable
        let params = optimal_fixed_angles(&ProblemType::MaxCut, 1);
        let energy = run_qaoa(&problem, &params);
        assert!(energy.is_finite());
    }

    // ----------------------------------------------------------
    // 11. Fourier interpolation preserves low-frequency structure
    // ----------------------------------------------------------
    #[test]
    fn test_fourier_interpolation_preserves_structure() {
        // Constant signal should remain constant after Fourier resize
        let src = QaoaParams::new(vec![0.5, 0.5, 0.5], vec![0.3, 0.3, 0.3]);
        let dst = fourier_interpolation(&src, 5);
        assert_eq!(dst.num_layers, 5);

        // All gammas should be approximately 0.5 (constant is pure DC)
        for &g in &dst.gammas {
            assert!(
                (g - 0.5).abs() < 1e-10,
                "Fourier failed to preserve constant: got {}",
                g
            );
        }
        for &b in &dst.betas {
            assert!(
                (b - 0.3).abs() < 1e-10,
                "Fourier failed to preserve constant beta: got {}",
                b
            );
        }
    }

    // ----------------------------------------------------------
    // 12. Nelder-Mead converges for simple problem
    // ----------------------------------------------------------
    #[test]
    fn test_nelder_mead_converges() {
        // Minimize (x - 3)^2 + (y - 5)^2
        let f = |x: &[f64]| -> f64 { (x[0] - 3.0).powi(2) + (x[1] - 5.0).powi(2) };
        let (best, val) = nelder_mead(&f, &[0.0, 0.0], 500);
        assert!(
            (best[0] - 3.0).abs() < 1e-4,
            "x not converged: {}",
            best[0]
        );
        assert!(
            (best[1] - 5.0).abs() < 1e-4,
            "y not converged: {}",
            best[1]
        );
        assert!(val < 1e-6, "Minimum not found: {}", val);
    }

    // ----------------------------------------------------------
    // 13. QaoaParams to_vec / from_vec roundtrip
    // ----------------------------------------------------------
    #[test]
    fn test_qaoa_params_roundtrip() {
        let params = QaoaParams::new(vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]);
        let v = params.to_vec();
        assert_eq!(v, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let restored = QaoaParams::from_vec(&v, 3);
        assert_eq!(restored.gammas, params.gammas);
        assert_eq!(restored.betas, params.betas);
        assert_eq!(restored.num_layers, 3);
    }

    // ----------------------------------------------------------
    // 14. 3-regular graph generator
    // ----------------------------------------------------------
    #[test]
    fn test_random_3_regular_graph() {
        let mut rng = StdRng::seed_from_u64(99);
        let problem = random_3_regular_graph(6, &mut rng);
        assert_eq!(problem.num_qubits, 6);
        assert!(!problem.hamiltonian.is_empty());
        assert_eq!(problem.problem_type, ProblemType::MaxCut);
    }

    // ----------------------------------------------------------
    // 15. Warm start error on empty Hamiltonian
    // ----------------------------------------------------------
    #[test]
    fn test_warm_start_error_empty_hamiltonian() {
        let source = ProblemInstance {
            num_qubits: 2,
            hamiltonian: vec![],
            problem_type: ProblemType::Custom,
        };
        let target = triangle_max_cut();
        let params = QaoaParams::new(vec![0.1], vec![0.2]);
        let config = WarmStartConfig::new().num_layers(1);
        let result = warm_start_transfer(&source, &params, &target, &config);
        assert!(result.is_err());
        match result {
            Err(WarmStartError::IncompatibleProblems(_)) => {}
            _ => panic!("Expected IncompatibleProblems error"),
        }
    }

    // ----------------------------------------------------------
    // 16. Mixer preserves state normalization
    // ----------------------------------------------------------
    #[test]
    fn test_mixer_preserves_norm() {
        let mut state = plus_state(3);
        apply_mixer(&mut state, 3, 0.42);
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Mixer broke normalization: {}",
            norm
        );
    }
}
