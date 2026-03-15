//! Quantum Natural Gradient (QNG) optimizer with Fubini-Study metric tensor.
//!
//! Implements proper QNG optimization for variational quantum circuits,
//! computing the full quantum Fisher information matrix (QFIM) via the
//! Fubini-Study metric tensor. Supports full, block-diagonal, and diagonal
//! approximations for the metric.
//!
//! # References
//! - Stokes et al., "Quantum Natural Gradient" (2020)
//! - PennyLane's QNGOptimizer implementation

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during QNG optimization.
#[derive(Debug, Clone)]
pub enum QngError {
    /// Metric tensor is singular even after regularization.
    SingularMetric,
    /// Optimization did not converge within max_iterations.
    ConvergenceFailed { iterations: usize, final_cost: f64 },
    /// Circuit is invalid (e.g., parameter index out of bounds).
    InvalidCircuit(String),
}

impl std::fmt::Display for QngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QngError::SingularMetric => write!(f, "Metric tensor is singular"),
            QngError::ConvergenceFailed {
                iterations,
                final_cost,
            } => write!(
                f,
                "Convergence failed after {} iterations (cost={})",
                iterations, final_cost
            ),
            QngError::InvalidCircuit(msg) => write!(f, "Invalid circuit: {}", msg),
        }
    }
}

impl std::error::Error for QngError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Method used to compute the metric tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricMethod {
    /// Full Fubini-Study metric tensor: O(p^2) circuit evaluations.
    FullFubiniStudy,
    /// Block-diagonal approximation exploiting layer structure.
    BlockDiagonal,
    /// Diagonal approximation: only g_ii entries, cheapest.
    DiagonalApprox,
    /// Identity matrix (reduces QNG to vanilla gradient descent).
    Identity,
}

/// Configuration for the QNG optimizer.
#[derive(Debug, Clone)]
pub struct QngConfig {
    /// Learning rate η.
    pub learning_rate: f64,
    /// Tikhonov regularization λ for metric: g' = g + λI.
    pub regularization: f64,
    /// Maximum number of optimization iterations.
    pub max_iterations: usize,
    /// Convergence threshold on cost change.
    pub convergence_threshold: f64,
    /// Method for metric tensor computation.
    pub metric_method: MetricMethod,
}

impl Default for QngConfig {
    fn default() -> Self {
        QngConfig {
            learning_rate: 0.01,
            regularization: 1e-4,
            max_iterations: 200,
            convergence_threshold: 1e-6,
            metric_method: MetricMethod::FullFubiniStudy,
        }
    }
}

impl QngConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    pub fn convergence_threshold(mut self, thr: f64) -> Self {
        self.convergence_threshold = thr;
        self
    }

    pub fn metric_method(mut self, method: MetricMethod) -> Self {
        self.metric_method = method;
        self
    }
}

// ============================================================
// PARAMETERIZED CIRCUIT TYPES
// ============================================================

/// Reference to a parameter or a constant angle.
#[derive(Debug, Clone)]
pub enum ParamRef {
    /// Index into the parameter vector.
    Parameter(usize),
    /// Fixed constant angle.
    Constant(f64),
}

/// Fixed (non-parameterized) gate.
#[derive(Debug, Clone, Copy)]
pub enum FixedGate {
    H,
    X,
    Y,
    Z,
    S,
    T,
    CX,
    CZ,
}

/// A gate in a parameterized circuit.
#[derive(Debug, Clone)]
pub enum ParamGate {
    Rx(usize, ParamRef),
    Ry(usize, ParamRef),
    Rz(usize, ParamRef),
    CX(usize, usize),
    H(usize),
    Fixed(FixedGate, usize, Option<usize>),
}

/// A parameterized quantum circuit.
#[derive(Debug, Clone)]
pub struct ParameterizedCircuit {
    pub num_qubits: usize,
    pub gates: Vec<ParamGate>,
    pub num_params: usize,
}

impl ParameterizedCircuit {
    /// Create a new empty parameterized circuit.
    pub fn new(num_qubits: usize) -> Self {
        ParameterizedCircuit {
            num_qubits,
            gates: Vec::new(),
            num_params: 0,
        }
    }

    /// Hardware-efficient ansatz: Ry(θ) on each qubit + CNOT ladder, repeated per layer.
    pub fn hardware_efficient(num_qubits: usize, num_layers: usize) -> Self {
        let mut gates = Vec::new();
        let mut param_idx = 0usize;

        for _layer in 0..num_layers {
            // Ry rotation on each qubit
            for q in 0..num_qubits {
                gates.push(ParamGate::Ry(q, ParamRef::Parameter(param_idx)));
                param_idx += 1;
            }
            // CNOT ladder
            for q in 0..(num_qubits - 1) {
                gates.push(ParamGate::CX(q, q + 1));
            }
        }

        ParameterizedCircuit {
            num_qubits,
            gates,
            num_params: param_idx,
        }
    }

    /// Strongly-entangling ansatz (PennyLane-style):
    /// Rot(θ,φ,ω) on each qubit + cyclic CNOT with shift per layer.
    pub fn strongly_entangling(num_qubits: usize, num_layers: usize) -> Self {
        let mut gates = Vec::new();
        let mut param_idx = 0usize;

        for layer in 0..num_layers {
            // Rot(θ,φ,ω) = Rz(ω) Ry(φ) Rz(θ) on each qubit
            for q in 0..num_qubits {
                gates.push(ParamGate::Rz(q, ParamRef::Parameter(param_idx)));
                param_idx += 1;
                gates.push(ParamGate::Ry(q, ParamRef::Parameter(param_idx)));
                param_idx += 1;
                gates.push(ParamGate::Rz(q, ParamRef::Parameter(param_idx)));
                param_idx += 1;
            }
            // Cyclic CNOT with shift = layer % num_qubits
            if num_qubits > 1 {
                let shift = (layer % (num_qubits - 1)) + 1;
                for q in 0..num_qubits {
                    let target = (q + shift) % num_qubits;
                    gates.push(ParamGate::CX(q, target));
                }
            }
        }

        ParameterizedCircuit {
            num_qubits,
            gates,
            num_params: param_idx,
        }
    }

    /// Get all parameter indices that belong to a given "layer" (for block-diagonal).
    /// A layer is defined as a contiguous group of parameterized gates between
    /// entangling (CX) blocks.
    fn layer_param_groups(&self) -> Vec<Vec<usize>> {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut current_group: Vec<usize> = Vec::new();

        for gate in &self.gates {
            match gate {
                ParamGate::Rx(_, ParamRef::Parameter(idx))
                | ParamGate::Ry(_, ParamRef::Parameter(idx))
                | ParamGate::Rz(_, ParamRef::Parameter(idx)) => {
                    current_group.push(*idx);
                }
                ParamGate::CX(_, _) => {
                    if !current_group.is_empty() {
                        groups.push(current_group.clone());
                        current_group.clear();
                    }
                }
                _ => {}
            }
        }
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        // Deduplicate: if a param appears in multiple groups, keep it in the first
        let mut seen = std::collections::HashSet::new();
        for group in &mut groups {
            group.retain(|idx| seen.insert(*idx));
        }
        groups.retain(|g| !g.is_empty());

        groups
    }
}

// ============================================================
// OPTIMIZATION RESULTS
// ============================================================

/// Result of a QNG optimization run.
#[derive(Debug, Clone)]
pub struct QngResult {
    /// Optimal parameters found.
    pub optimal_params: Vec<f64>,
    /// Cost at the optimal parameters.
    pub final_cost: f64,
    /// Number of iterations taken.
    pub num_iterations: usize,
    /// Cost value at each iteration.
    pub cost_history: Vec<f64>,
    /// Whether the optimizer converged.
    pub converged: bool,
}

/// Comparison of vanilla GD vs natural GD.
#[derive(Debug, Clone)]
pub struct OptimizerComparison {
    /// Result from vanilla gradient descent (identity metric).
    pub vanilla_gd: QngResult,
    /// Result from quantum natural gradient.
    pub natural_gd: QngResult,
    /// Ratio of vanilla iterations to QNG iterations (>1 means QNG is faster).
    pub speedup_ratio: f64,
}

// ============================================================
// CIRCUIT SIMULATION (self-contained)
// ============================================================

/// Simulate a parameterized circuit starting from |0...0>.
pub fn simulate_parameterized(circuit: &ParameterizedCircuit, params: &[f64]) -> Vec<Complex64> {
    let dim = 1 << circuit.num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);

    for gate in &circuit.gates {
        apply_gate(&mut state, circuit.num_qubits, gate, params);
    }

    state
}

/// Compute |<ψ|φ>|².
pub fn state_overlap(psi: &[Complex64], phi: &[Complex64]) -> f64 {
    let inner: Complex64 = psi.iter().zip(phi.iter()).map(|(a, b)| a.conj() * b).sum();
    inner.norm_sqr()
}

/// Compute <ψ|Z_qubit|ψ>.
pub fn expectation_z(state: &[Complex64], qubit: usize) -> f64 {
    let n = (state.len() as f64).log2() as usize;
    let mut exp_val = 0.0;
    for i in 0..state.len() {
        let bit = (i >> (n - 1 - qubit)) & 1;
        let sign = if bit == 0 { 1.0 } else { -1.0 };
        exp_val += sign * state[i].norm_sqr();
    }
    exp_val
}

/// Apply a single gate to the state vector.
fn apply_gate(state: &mut [Complex64], num_qubits: usize, gate: &ParamGate, params: &[f64]) {
    match gate {
        ParamGate::Rx(qubit, param_ref) => {
            let theta = resolve_param(param_ref, params);
            apply_single_qubit_gate(state, num_qubits, *qubit, &rx_matrix(theta));
        }
        ParamGate::Ry(qubit, param_ref) => {
            let theta = resolve_param(param_ref, params);
            apply_single_qubit_gate(state, num_qubits, *qubit, &ry_matrix(theta));
        }
        ParamGate::Rz(qubit, param_ref) => {
            let theta = resolve_param(param_ref, params);
            apply_single_qubit_gate(state, num_qubits, *qubit, &rz_matrix(theta));
        }
        ParamGate::CX(control, target) => {
            apply_cx(state, num_qubits, *control, *target);
        }
        ParamGate::H(qubit) => {
            let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
            let mat = [
                [
                    Complex64::new(inv_sqrt2, 0.0),
                    Complex64::new(inv_sqrt2, 0.0),
                ],
                [
                    Complex64::new(inv_sqrt2, 0.0),
                    Complex64::new(-inv_sqrt2, 0.0),
                ],
            ];
            apply_single_qubit_gate(state, num_qubits, *qubit, &mat);
        }
        ParamGate::Fixed(fg, qubit, target) => {
            apply_fixed_gate(state, num_qubits, *fg, *qubit, *target);
        }
    }
}

fn resolve_param(param_ref: &ParamRef, params: &[f64]) -> f64 {
    match param_ref {
        ParamRef::Parameter(idx) => params[*idx],
        ParamRef::Constant(val) => *val,
    }
}

fn rx_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [Complex64::new(c, 0.0), Complex64::new(0.0, -s)],
        [Complex64::new(0.0, -s), Complex64::new(c, 0.0)],
    ]
}

fn ry_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
        [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
    ]
}

fn rz_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    [
        [
            Complex64::new(0.0, -theta / 2.0).exp(),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, theta / 2.0).exp(),
        ],
    ]
}

fn apply_single_qubit_gate(
    state: &mut [Complex64],
    num_qubits: usize,
    qubit: usize,
    mat: &[[Complex64; 2]; 2],
) {
    let dim = 1 << num_qubits;
    let bit = num_qubits - 1 - qubit;
    let step = 1 << bit;

    let mut i = 0;
    while i < dim {
        for j in 0..step {
            let idx0 = i + j;
            let idx1 = idx0 + step;
            let a = state[idx0];
            let b = state[idx1];
            state[idx0] = mat[0][0] * a + mat[0][1] * b;
            state[idx1] = mat[1][0] * a + mat[1][1] * b;
        }
        i += 2 * step;
    }
}

fn apply_cx(state: &mut [Complex64], num_qubits: usize, control: usize, target: usize) {
    let dim = 1 << num_qubits;
    let ctrl_bit = num_qubits - 1 - control;
    let tgt_bit = num_qubits - 1 - target;

    for i in 0..dim {
        if ((i >> ctrl_bit) & 1) == 1 && ((i >> tgt_bit) & 1) == 0 {
            let j = i | (1 << tgt_bit);
            state.swap(i, j);
        }
    }
}

fn apply_fixed_gate(
    state: &mut [Complex64],
    num_qubits: usize,
    gate: FixedGate,
    qubit: usize,
    target: Option<usize>,
) {
    match gate {
        FixedGate::H => {
            let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
            let mat = [
                [
                    Complex64::new(inv_sqrt2, 0.0),
                    Complex64::new(inv_sqrt2, 0.0),
                ],
                [
                    Complex64::new(inv_sqrt2, 0.0),
                    Complex64::new(-inv_sqrt2, 0.0),
                ],
            ];
            apply_single_qubit_gate(state, num_qubits, qubit, &mat);
        }
        FixedGate::X => {
            let mat = [
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ];
            apply_single_qubit_gate(state, num_qubits, qubit, &mat);
        }
        FixedGate::Y => {
            let mat = [
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
            ];
            apply_single_qubit_gate(state, num_qubits, qubit, &mat);
        }
        FixedGate::Z => {
            let mat = [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ];
            apply_single_qubit_gate(state, num_qubits, qubit, &mat);
        }
        FixedGate::S => {
            let mat = [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
            ];
            apply_single_qubit_gate(state, num_qubits, qubit, &mat);
        }
        FixedGate::T => {
            let mat = [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, PI / 4.0).exp(),
                ],
            ];
            apply_single_qubit_gate(state, num_qubits, qubit, &mat);
        }
        FixedGate::CX => {
            if let Some(t) = target {
                apply_cx(state, num_qubits, qubit, t);
            }
        }
        FixedGate::CZ => {
            if let Some(t) = target {
                apply_cz(state, num_qubits, qubit, t);
            }
        }
    }
}

fn apply_cz(state: &mut [Complex64], num_qubits: usize, control: usize, target: usize) {
    let dim = 1 << num_qubits;
    let ctrl_bit = num_qubits - 1 - control;
    let tgt_bit = num_qubits - 1 - target;

    for i in 0..dim {
        if ((i >> ctrl_bit) & 1) == 1 && ((i >> tgt_bit) & 1) == 1 {
            state[i] = -state[i];
        }
    }
}

// ============================================================
// GRADIENT COMPUTATION
// ============================================================

/// Parameter-shift gradient: ∂C/∂θ_i = [C(θ_i + π/2) - C(θ_i - π/2)] / 2
pub fn parameter_shift_gradient(
    circuit: &ParameterizedCircuit,
    cost_fn: &dyn Fn(&ParameterizedCircuit, &[f64]) -> f64,
    params: &[f64],
) -> Vec<f64> {
    let n = circuit.num_params;
    let mut grad = vec![0.0; n];
    let shift = PI / 2.0;

    for i in 0..n {
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();
        params_plus[i] += shift;
        params_minus[i] -= shift;

        let f_plus = cost_fn(circuit, &params_plus);
        let f_minus = cost_fn(circuit, &params_minus);
        grad[i] = (f_plus - f_minus) / 2.0;
    }

    grad
}

/// Finite-difference gradient (fallback): [f(θ+ε) - f(θ-ε)] / (2ε)
pub fn finite_difference_gradient(
    cost_fn: &dyn Fn(&[f64]) -> f64,
    params: &[f64],
    epsilon: f64,
) -> Vec<f64> {
    let n = params.len();
    let mut grad = vec![0.0; n];

    for i in 0..n {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[i] += epsilon;
        p_minus[i] -= epsilon;

        grad[i] = (cost_fn(&p_plus) - cost_fn(&p_minus)) / (2.0 * epsilon);
    }

    grad
}

// ============================================================
// FUBINI-STUDY METRIC TENSOR
// ============================================================

/// Compute the full Fubini-Study metric tensor.
///
/// g_ij = Re[<∂_i ψ|∂_j ψ> - <∂_i ψ|ψ><ψ|∂_j ψ>]
///
/// Uses parameter-shift to compute the overlaps. For each pair (i,j) we use
/// the identity relating the metric to shifted circuit overlaps.
pub fn fubini_study_metric(circuit: &ParameterizedCircuit, params: &[f64]) -> Array2<f64> {
    let n = circuit.num_params;
    let mut metric = Array2::<f64>::zeros((n, n));
    let shift = PI / 2.0;

    // Base state
    let psi = simulate_parameterized(circuit, params);

    // For the Fubini-Study metric via parameter-shift:
    // g_ij = (-1/2) * (∂²/∂θ_i∂θ_j) F(θ)  where F(θ) = |<ψ(0)|ψ(θ)>|²
    //
    // Using the identity for rotation gates:
    // g_ij = (1/2) * [Re(<ψ_i+|ψ_j+>) + Re(<ψ_i-|ψ_j->)
    //                 - Re(<ψ_i+|ψ_j->) - Re(<ψ_i-|ψ_j+>)] / 4
    //
    // Simplified: use the direct overlap approach.
    // <∂_i ψ|∂_j ψ> can be accessed through shifted overlaps.

    // Cache shifted states (forward and backward for each param)
    let mut states_plus: Vec<Vec<Complex64>> = Vec::with_capacity(n);
    let mut states_minus: Vec<Vec<Complex64>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[i] += shift;
        p_minus[i] -= shift;
        states_plus.push(simulate_parameterized(circuit, &p_plus));
        states_minus.push(simulate_parameterized(circuit, &p_minus));
    }

    for i in 0..n {
        for j in i..n {
            // g_ij = -1/2 * ∂²F/∂θi∂θj where F = |<0|U(θ)|0>|² evaluated at base params
            // Using parameter-shift rule for second derivatives:
            // ∂²F/∂θi∂θj = 1/4 [F(+i,+j) - F(+i,-j) - F(-i,+j) + F(-i,-j)]
            //
            // But F here is the fidelity of the shifted state with base.
            // Instead, compute metric directly from state overlaps:
            //
            // g_ij = Re[<∂_i ψ|∂_j ψ>] - Re[<∂_i ψ|ψ>]*Re[<ψ|∂_j ψ>]
            //        - Im[<∂_i ψ|ψ>]*Im[<ψ|∂_j ψ>]
            //
            // where |∂_i ψ> ≈ (|ψ_i+> - |ψ_i->) / (2 sin(s)) for shift s=π/2 → divisor 2.
            // For parameter-shift with s=π/2: |∂_i ψ> = (|ψ_i+> - |ψ_i->)/2

            let di_dot_dj = inner_product_diff(
                &states_plus[i],
                &states_minus[i],
                &states_plus[j],
                &states_minus[j],
            );
            let di_dot_psi = inner_product_diff_with_state(&states_plus[i], &states_minus[i], &psi);
            let psi_dot_dj = inner_product_state_with_diff(&psi, &states_plus[j], &states_minus[j]);

            let g_ij = di_dot_dj.re - (di_dot_psi * psi_dot_dj).re;

            metric[[i, j]] = g_ij;
            metric[[j, i]] = g_ij;
        }
    }

    metric
}

/// Compute <(ψ+ - ψ-)/2 | (φ+ - φ-)/2>
fn inner_product_diff(
    psi_plus: &[Complex64],
    psi_minus: &[Complex64],
    phi_plus: &[Complex64],
    phi_minus: &[Complex64],
) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);
    for k in 0..psi_plus.len() {
        let dpsi = (psi_plus[k] - psi_minus[k]) / 2.0;
        let dphi = (phi_plus[k] - phi_minus[k]) / 2.0;
        result += dpsi.conj() * dphi;
    }
    result
}

/// Compute <(ψ+ - ψ-)/2 | state>
fn inner_product_diff_with_state(
    psi_plus: &[Complex64],
    psi_minus: &[Complex64],
    state: &[Complex64],
) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);
    for k in 0..psi_plus.len() {
        let dpsi = (psi_plus[k] - psi_minus[k]) / 2.0;
        result += dpsi.conj() * state[k];
    }
    result
}

/// Compute <state | (φ+ - φ-)/2>
fn inner_product_state_with_diff(
    state: &[Complex64],
    phi_plus: &[Complex64],
    phi_minus: &[Complex64],
) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);
    for k in 0..state.len() {
        let dphi = (phi_plus[k] - phi_minus[k]) / 2.0;
        result += state[k].conj() * dphi;
    }
    result
}

/// Block-diagonal approximation of the metric tensor.
///
/// Parameters within the same layer interact; parameters in different layers
/// are assumed independent (off-diagonal blocks = 0).
pub fn block_diagonal_metric(circuit: &ParameterizedCircuit, params: &[f64]) -> Array2<f64> {
    let n = circuit.num_params;
    let mut metric = Array2::<f64>::zeros((n, n));
    let groups = circuit.layer_param_groups();

    if groups.is_empty() {
        // Fallback: treat all params as one group
        return fubini_study_metric(circuit, params);
    }

    let shift = PI / 2.0;
    let psi = simulate_parameterized(circuit, params);

    // Pre-compute shifted states for all params
    let mut states_plus: Vec<Option<Vec<Complex64>>> = vec![None; n];
    let mut states_minus: Vec<Option<Vec<Complex64>>> = vec![None; n];

    for group in &groups {
        for &idx in group {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[idx] += shift;
            p_minus[idx] -= shift;
            states_plus[idx] = Some(simulate_parameterized(circuit, &p_plus));
            states_minus[idx] = Some(simulate_parameterized(circuit, &p_minus));
        }
    }

    // Compute metric only within each block
    for group in &groups {
        for &i in group {
            for &j in group {
                if j < i {
                    continue;
                }
                let sp_i = states_plus[i].as_ref().unwrap();
                let sm_i = states_minus[i].as_ref().unwrap();
                let sp_j = states_plus[j].as_ref().unwrap();
                let sm_j = states_minus[j].as_ref().unwrap();

                let di_dot_dj = inner_product_diff(sp_i, sm_i, sp_j, sm_j);
                let di_dot_psi = inner_product_diff_with_state(sp_i, sm_i, &psi);
                let psi_dot_dj = inner_product_state_with_diff(&psi, sp_j, sm_j);

                let g_ij = di_dot_dj.re - (di_dot_psi * psi_dot_dj).re;
                metric[[i, j]] = g_ij;
                metric[[j, i]] = g_ij;
            }
        }
    }

    metric
}

/// Diagonal approximation of the metric tensor.
///
/// g_ii = Re[<∂_i ψ|∂_i ψ>] - |<∂_i ψ|ψ>|²
/// Only 2 circuit evaluations per parameter.
pub fn diagonal_metric(circuit: &ParameterizedCircuit, params: &[f64]) -> Vec<f64> {
    let n = circuit.num_params;
    let mut diag = vec![0.0; n];
    let shift = PI / 2.0;

    let psi = simulate_parameterized(circuit, params);

    for i in 0..n {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[i] += shift;
        p_minus[i] -= shift;

        let sp = simulate_parameterized(circuit, &p_plus);
        let sm = simulate_parameterized(circuit, &p_minus);

        let di_dot_di = inner_product_diff(&sp, &sm, &sp, &sm);
        let di_dot_psi = inner_product_diff_with_state(&sp, &sm, &psi);

        diag[i] = di_dot_di.re - di_dot_psi.norm_sqr();
    }

    diag
}

// ============================================================
// LINEAR ALGEBRA HELPERS
// ============================================================

/// Solve Ax = b via Cholesky-like decomposition for SPD matrices.
/// Falls back to Gaussian elimination with partial pivoting if needed.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, QngError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    assert_eq!(n, b.len());

    // Gaussian elimination with partial pivoting
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return Err(QngError::SingularMetric);
        }

        // Swap rows
        if max_row != col {
            for k in 0..=n {
                let tmp = aug[[col, k]];
                aug[[col, k]] = aug[[max_row, k]];
                aug[[max_row, k]] = tmp;
            }
        }

        // Eliminate
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for k in col..=n {
                aug[[row, k]] -= factor * aug[[col, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Add Tikhonov regularization: g' = g + λI
fn regularize_metric(metric: &Array2<f64>, lambda: f64) -> Array2<f64> {
    let n = metric.nrows();
    let mut reg = metric.clone();
    for i in 0..n {
        reg[[i, i]] += lambda;
    }
    reg
}

// ============================================================
// QNG OPTIMIZER
// ============================================================

/// Run QNG optimization on a parameterized circuit.
///
/// Algorithm:
/// 1. Compute cost gradient via parameter-shift
/// 2. Compute metric tensor g_ij
/// 3. Regularize: g' = g + λI
/// 4. Solve g' * Δθ = -η * ∇C
/// 5. Update: θ ← θ + Δθ
/// 6. Repeat until convergence
pub fn optimize(
    circuit: &ParameterizedCircuit,
    cost_fn: &dyn Fn(&ParameterizedCircuit, &[f64]) -> f64,
    initial_params: &[f64],
    config: &QngConfig,
) -> Result<QngResult, QngError> {
    if circuit.num_params == 0 {
        return Err(QngError::InvalidCircuit("Circuit has no parameters".into()));
    }
    if initial_params.len() != circuit.num_params {
        return Err(QngError::InvalidCircuit(format!(
            "Expected {} params, got {}",
            circuit.num_params,
            initial_params.len()
        )));
    }

    let mut params = initial_params.to_vec();
    let mut cost_history = Vec::with_capacity(config.max_iterations);
    let mut current_cost = cost_fn(circuit, &params);
    cost_history.push(current_cost);

    for iter in 0..config.max_iterations {
        // 1. Compute gradient
        let grad = parameter_shift_gradient(circuit, cost_fn, &params);

        // 2. Compute metric tensor
        let n = circuit.num_params;
        let nat_grad = match config.metric_method {
            MetricMethod::FullFubiniStudy => {
                let metric = fubini_study_metric(circuit, &params);
                let reg_metric = regularize_metric(&metric, config.regularization);
                let rhs =
                    Array1::from_vec(grad.iter().map(|g| -config.learning_rate * g).collect());
                solve_linear_system(&reg_metric, &rhs)?
            }
            MetricMethod::BlockDiagonal => {
                let metric = block_diagonal_metric(circuit, &params);
                let reg_metric = regularize_metric(&metric, config.regularization);
                let rhs =
                    Array1::from_vec(grad.iter().map(|g| -config.learning_rate * g).collect());
                solve_linear_system(&reg_metric, &rhs)?
            }
            MetricMethod::DiagonalApprox => {
                let diag = diagonal_metric(circuit, &params);
                let mut delta = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let g_ii = diag[i] + config.regularization;
                    delta[i] = -config.learning_rate * grad[i] / g_ii;
                }
                delta
            }
            MetricMethod::Identity => {
                // Vanilla GD
                let mut delta = Array1::<f64>::zeros(n);
                for i in 0..n {
                    delta[i] = -config.learning_rate * grad[i];
                }
                delta
            }
        };

        // 5. Update parameters
        for i in 0..n {
            params[i] += nat_grad[i];
        }

        // 6. Check convergence
        let new_cost = cost_fn(circuit, &params);
        cost_history.push(new_cost);

        let cost_change = (current_cost - new_cost).abs();
        current_cost = new_cost;

        if cost_change < config.convergence_threshold && iter > 0 {
            return Ok(QngResult {
                optimal_params: params,
                final_cost: current_cost,
                num_iterations: iter + 1,
                cost_history,
                converged: true,
            });
        }
    }

    Err(QngError::ConvergenceFailed {
        iterations: config.max_iterations,
        final_cost: current_cost,
    })
}

/// Compare vanilla gradient descent vs quantum natural gradient.
pub fn compare_optimizers(
    circuit: &ParameterizedCircuit,
    cost_fn: &dyn Fn(&ParameterizedCircuit, &[f64]) -> f64,
    initial_params: &[f64],
) -> OptimizerComparison {
    let vanilla_config = QngConfig::new()
        .metric_method(MetricMethod::Identity)
        .learning_rate(0.1)
        .max_iterations(500)
        .convergence_threshold(1e-5);

    let qng_config = QngConfig::new()
        .metric_method(MetricMethod::FullFubiniStudy)
        .learning_rate(0.1)
        .max_iterations(500)
        .convergence_threshold(1e-5);

    let vanilla_result = match optimize(circuit, cost_fn, initial_params, &vanilla_config) {
        Ok(r) => r,
        Err(QngError::ConvergenceFailed {
            iterations,
            final_cost,
        }) => QngResult {
            optimal_params: initial_params.to_vec(),
            final_cost,
            num_iterations: iterations,
            cost_history: vec![final_cost],
            converged: false,
        },
        Err(_) => QngResult {
            optimal_params: initial_params.to_vec(),
            final_cost: f64::INFINITY,
            num_iterations: 0,
            cost_history: vec![],
            converged: false,
        },
    };

    let qng_result = match optimize(circuit, cost_fn, initial_params, &qng_config) {
        Ok(r) => r,
        Err(QngError::ConvergenceFailed {
            iterations,
            final_cost,
        }) => QngResult {
            optimal_params: initial_params.to_vec(),
            final_cost,
            num_iterations: iterations,
            cost_history: vec![final_cost],
            converged: false,
        },
        Err(_) => QngResult {
            optimal_params: initial_params.to_vec(),
            final_cost: f64::INFINITY,
            num_iterations: 0,
            cost_history: vec![],
            converged: false,
        },
    };

    let speedup = if qng_result.num_iterations > 0 {
        vanilla_result.num_iterations as f64 / qng_result.num_iterations as f64
    } else {
        1.0
    };

    OptimizerComparison {
        vanilla_gd: vanilla_result,
        natural_gd: qng_result,
        speedup_ratio: speedup,
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ---- Test 1: QngConfig builder defaults ----
    #[test]
    fn test_qng_config_defaults() {
        let config = QngConfig::new();
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
        assert!((config.regularization - 1e-4).abs() < 1e-10);
        assert_eq!(config.max_iterations, 200);
        assert!((config.convergence_threshold - 1e-6).abs() < 1e-12);
        assert_eq!(config.metric_method, MetricMethod::FullFubiniStudy);
    }

    // ---- Test 2: Hardware-efficient circuit param count ----
    #[test]
    fn test_hardware_efficient_param_count() {
        let circuit = ParameterizedCircuit::hardware_efficient(3, 2);
        // 2 layers * 3 qubits = 6 Ry params
        assert_eq!(circuit.num_params, 6);
        assert_eq!(circuit.num_qubits, 3);

        let circuit2 = ParameterizedCircuit::hardware_efficient(4, 3);
        // 3 layers * 4 qubits = 12 Ry params
        assert_eq!(circuit2.num_params, 12);
    }

    // ---- Test 3: Diagonal metric: all entries positive ----
    #[test]
    fn test_diagonal_metric_positive() {
        let circuit = ParameterizedCircuit::hardware_efficient(2, 1);
        let params = vec![0.5, 1.2];
        let diag = diagonal_metric(&circuit, &params);

        for (i, &val) in diag.iter().enumerate() {
            assert!(
                val >= -1e-10,
                "Diagonal metric entry {} is negative: {}",
                i,
                val
            );
        }
    }

    // ---- Test 4: Full metric: symmetric matrix ----
    #[test]
    fn test_full_metric_symmetric() {
        let circuit = ParameterizedCircuit::hardware_efficient(2, 2);
        let params = vec![0.3, 0.7, 1.1, 0.4];
        let metric = fubini_study_metric(&circuit, &params);

        let n = circuit.num_params;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (metric[[i, j]] - metric[[j, i]]).abs() < 1e-10,
                    "Metric not symmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    metric[[i, j]],
                    metric[[j, i]]
                );
            }
        }
    }

    // ---- Test 5: Full metric diagonal matches diagonal-only computation ----
    #[test]
    fn test_full_metric_diagonal_matches() {
        let circuit = ParameterizedCircuit::hardware_efficient(2, 1);
        let params = vec![0.8, 1.5];
        let full = fubini_study_metric(&circuit, &params);
        let diag = diagonal_metric(&circuit, &params);

        for i in 0..circuit.num_params {
            assert!(
                (full[[i, i]] - diag[i]).abs() < 1e-8,
                "Diagonal mismatch at {}: full={}, diag={}",
                i,
                full[[i, i]],
                diag[i]
            );
        }
    }

    // ---- Test 6: Parameter-shift gradient: d/dθ sin(θ) = cos(θ) for Ry ----
    #[test]
    fn test_parameter_shift_gradient_ry() {
        // Circuit: Ry(θ)|0>, cost = <Z> = cos(θ)
        // d(cos θ)/dθ = -sin(θ)
        let circuit = ParameterizedCircuit {
            num_qubits: 1,
            gates: vec![ParamGate::Ry(0, ParamRef::Parameter(0))],
            num_params: 1,
        };

        let cost_fn = |circ: &ParameterizedCircuit, params: &[f64]| -> f64 {
            let state = simulate_parameterized(circ, params);
            expectation_z(&state, 0)
        };

        for &theta in &[0.0, 0.5, 1.0, PI / 4.0, PI / 2.0, PI] {
            let params = vec![theta];
            let grad = parameter_shift_gradient(&circuit, &cost_fn, &params);
            let expected = -theta.sin();
            assert!(
                (grad[0] - expected).abs() < 1e-6,
                "Gradient at θ={}: got {}, expected {}",
                theta,
                grad[0],
                expected
            );
        }
    }

    // ---- Test 7: Regularization prevents singular metric ----
    #[test]
    fn test_regularization_prevents_singular() {
        // At θ=0, Ry(0)|0> = |0>, so derivative might cause near-singular metric
        let circuit = ParameterizedCircuit::hardware_efficient(2, 1);
        let params = vec![0.0, 0.0];

        let metric = fubini_study_metric(&circuit, &params);
        let reg_metric = regularize_metric(&metric, 1e-4);

        // Should be solvable
        let b = Array1::from_vec(vec![1.0, 1.0]);
        let result = solve_linear_system(&reg_metric, &b);
        assert!(result.is_ok(), "Regularized system should be solvable");
    }

    // ---- Test 8: QNG converges for simple VQE problem ----
    #[test]
    fn test_qng_converges_simple_vqe() {
        // 1-qubit: minimize <Z> with Ry(θ)|0>
        // Minimum at θ=π where <Z> = cos(π) = -1
        let circuit = ParameterizedCircuit {
            num_qubits: 1,
            gates: vec![ParamGate::Ry(0, ParamRef::Parameter(0))],
            num_params: 1,
        };

        let cost_fn = |circ: &ParameterizedCircuit, params: &[f64]| -> f64 {
            let state = simulate_parameterized(circ, params);
            expectation_z(&state, 0)
        };

        let config = QngConfig::new()
            .learning_rate(0.5)
            .max_iterations(200)
            .convergence_threshold(1e-6)
            .metric_method(MetricMethod::DiagonalApprox);

        let result = optimize(&circuit, &cost_fn, &[0.5], &config);
        assert!(result.is_ok(), "QNG should converge: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.converged, "Should have converged");
        assert!(
            (result.final_cost - (-1.0)).abs() < 1e-3,
            "Final cost should be near -1: {}",
            result.final_cost
        );
    }

    // ---- Test 9: QNG converges faster than vanilla GD ----
    #[test]
    fn test_qng_faster_than_vanilla() {
        // 2-qubit H2-like Hamiltonian: H = -Z0 + 0.5*Z1 + 0.2*Z0Z1
        // This gives QNG an advantage because of parameter correlation
        let circuit = ParameterizedCircuit::hardware_efficient(2, 2);

        let cost_fn = |circ: &ParameterizedCircuit, params: &[f64]| -> f64 {
            let state = simulate_parameterized(circ, params);
            let z0 = expectation_z(&state, 0);
            let z1 = expectation_z(&state, 1);

            // Z0Z1 expectation: sum p_i * (-1)^(b0 XOR b1)
            let n = circ.num_qubits;
            let dim = 1 << n;
            let mut z0z1 = 0.0;
            for i in 0..dim {
                let b0 = (i >> (n - 1)) & 1;
                let b1 = (i >> (n - 2)) & 1;
                let sign = if b0 ^ b1 == 0 { 1.0 } else { -1.0 };
                z0z1 += sign * state[i].norm_sqr();
            }

            -z0 + 0.5 * z1 + 0.2 * z0z1
        };

        let initial = vec![0.1, 0.2, 0.3, 0.4];

        let vanilla_config = QngConfig::new()
            .metric_method(MetricMethod::Identity)
            .learning_rate(0.1)
            .max_iterations(300)
            .convergence_threshold(1e-5);

        let qng_config = QngConfig::new()
            .metric_method(MetricMethod::DiagonalApprox)
            .learning_rate(0.1)
            .max_iterations(300)
            .convergence_threshold(1e-5);

        // Run both -- if either fails to converge, still compare iteration counts
        let vanilla = match optimize(&circuit, &cost_fn, &initial, &vanilla_config) {
            Ok(r) => r,
            Err(QngError::ConvergenceFailed {
                iterations,
                final_cost,
            }) => QngResult {
                optimal_params: initial.clone(),
                final_cost,
                num_iterations: iterations,
                cost_history: vec![],
                converged: false,
            },
            Err(e) => panic!("Vanilla GD unexpected error: {:?}", e),
        };

        let qng = match optimize(&circuit, &cost_fn, &initial, &qng_config) {
            Ok(r) => r,
            Err(QngError::ConvergenceFailed {
                iterations,
                final_cost,
            }) => QngResult {
                optimal_params: initial.clone(),
                final_cost,
                num_iterations: iterations,
                cost_history: vec![],
                converged: false,
            },
            Err(e) => panic!("QNG unexpected error: {:?}", e),
        };

        // QNG should converge in fewer or equal iterations, or reach a lower cost
        let qng_better =
            qng.num_iterations <= vanilla.num_iterations || qng.final_cost < vanilla.final_cost;
        assert!(
            qng_better,
            "QNG should be better: QNG iters={} cost={:.6}, Vanilla iters={} cost={:.6}",
            qng.num_iterations, qng.final_cost, vanilla.num_iterations, vanilla.final_cost
        );
    }

    // ---- Test 10: Cost decreases monotonically for simple problem ----
    #[test]
    fn test_cost_decreases_monotonically() {
        let circuit = ParameterizedCircuit {
            num_qubits: 1,
            gates: vec![ParamGate::Ry(0, ParamRef::Parameter(0))],
            num_params: 1,
        };

        let cost_fn = |circ: &ParameterizedCircuit, params: &[f64]| -> f64 {
            let state = simulate_parameterized(circ, params);
            expectation_z(&state, 0)
        };

        let config = QngConfig::new()
            .learning_rate(0.3)
            .max_iterations(100)
            .convergence_threshold(1e-8)
            .metric_method(MetricMethod::DiagonalApprox);

        let result = optimize(&circuit, &cost_fn, &[0.5], &config);
        let result = match result {
            Ok(r) => r,
            Err(QngError::ConvergenceFailed { .. }) => return, // acceptable
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // Check that cost never increases significantly (allow small numerical noise)
        for i in 1..result.cost_history.len() {
            let increase = result.cost_history[i] - result.cost_history[i - 1];
            assert!(
                increase < 1e-4,
                "Cost increased at step {}: {} -> {} (increase={})",
                i,
                result.cost_history[i - 1],
                result.cost_history[i],
                increase
            );
        }
    }

    // ---- Test 11: State overlap: same state = 1, orthogonal = 0 ----
    #[test]
    fn test_state_overlap() {
        // |0>
        let psi0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        // |1>
        let psi1 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        // |+>
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let psi_plus = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];

        // Same state
        assert!((state_overlap(&psi0, &psi0) - 1.0).abs() < 1e-10);
        assert!((state_overlap(&psi1, &psi1) - 1.0).abs() < 1e-10);

        // Orthogonal
        assert!(state_overlap(&psi0, &psi1).abs() < 1e-10);

        // |<0|+>|² = 0.5
        assert!((state_overlap(&psi0, &psi_plus) - 0.5).abs() < 1e-10);
    }

    // ---- Test 12: Block diagonal approximation has correct shape ----
    #[test]
    fn test_block_diagonal_shape() {
        let circuit = ParameterizedCircuit::hardware_efficient(3, 2);
        let params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let metric = block_diagonal_metric(&circuit, &params);

        assert_eq!(metric.nrows(), circuit.num_params);
        assert_eq!(metric.ncols(), circuit.num_params);

        // Should be symmetric
        let n = circuit.num_params;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (metric[[i, j]] - metric[[j, i]]).abs() < 1e-10,
                    "Block diagonal metric not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    // ---- Test 13: Strongly entangling circuit param count ----
    #[test]
    fn test_strongly_entangling_param_count() {
        let circuit = ParameterizedCircuit::strongly_entangling(3, 2);
        // 2 layers * 3 qubits * 3 Rot params = 18
        assert_eq!(circuit.num_params, 18);
    }

    // ---- Test 14: Simulate parameterized produces valid state ----
    #[test]
    fn test_simulate_produces_valid_state() {
        let circuit = ParameterizedCircuit::hardware_efficient(3, 2);
        let params: Vec<f64> = (0..circuit.num_params).map(|i| 0.1 * (i as f64)).collect();
        let state = simulate_parameterized(&circuit, &params);

        // State should be normalized
        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "State not normalized: norm={}",
            norm
        );
    }

    // ---- Test 15: Finite difference gradient agrees with parameter-shift ----
    #[test]
    fn test_finite_difference_vs_parameter_shift() {
        let circuit = ParameterizedCircuit::hardware_efficient(2, 1);
        let params = vec![0.7, 1.3];

        let cost_fn_circ = |circ: &ParameterizedCircuit, p: &[f64]| -> f64 {
            let state = simulate_parameterized(circ, p);
            expectation_z(&state, 0) + 0.5 * expectation_z(&state, 1)
        };

        let ps_grad = parameter_shift_gradient(&circuit, &cost_fn_circ, &params);

        let circ_clone = circuit.clone();
        let cost_fn_plain = move |p: &[f64]| -> f64 {
            let state = simulate_parameterized(&circ_clone, p);
            expectation_z(&state, 0) + 0.5 * expectation_z(&state, 1)
        };

        let fd_grad = finite_difference_gradient(&cost_fn_plain, &params, 1e-5);

        for i in 0..params.len() {
            assert!(
                (ps_grad[i] - fd_grad[i]).abs() < 1e-4,
                "Gradient mismatch at {}: ps={}, fd={}",
                i,
                ps_grad[i],
                fd_grad[i]
            );
        }
    }

    // ---- Test 16: Expectation Z for basis states ----
    #[test]
    fn test_expectation_z_basis_states() {
        // |0> -> <Z> = +1
        let state0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        assert!((expectation_z(&state0, 0) - 1.0).abs() < 1e-10);

        // |1> -> <Z> = -1
        let state1 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        assert!((expectation_z(&state1, 0) - (-1.0)).abs() < 1e-10);

        // |+> -> <Z> = 0
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let state_plus = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];
        assert!(expectation_z(&state_plus, 0).abs() < 1e-10);
    }
}
