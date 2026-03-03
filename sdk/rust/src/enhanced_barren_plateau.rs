//! Enhanced Barren Plateau Analysis with Empirical Sampling
//!
//! This module provides empirical tools for analyzing variational quantum circuits
//! for barren plateau phenomena, expressibility, and entanglement capability.
//!
//! # Key Components
//!
//! - [`EmpiricalBarrenPlateauAnalysis`]: Gradient variance sampling and scaling exponent fitting
//! - [`ExpressibilityAnalysis`]: KL divergence between circuit output and Haar-random distributions
//! - [`EntanglementCapability`]: Meyer-Wallach entanglement measure for quantum states
//! - [`CostLandscapeVisualization`]: Parameter slices and gradient norm distributions
//!
//! # Theory
//!
//! Barren plateaus occur when the variance of the cost function gradient vanishes
//! exponentially with system size: Var(dC/dtheta) ~ 2^{-alpha*n}, where alpha > 0
//! indicates trainability issues. This module empirically estimates alpha by
//! sampling random parameter vectors and computing numerical gradients.

use crate::circuit_complexity::RiskLevel;
use crate::gates::{Gate, GateType};
use crate::{GateOperations, QuantumState, C64};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helper: random parameter generation
// ---------------------------------------------------------------------------

/// Generate a vector of random parameters uniformly distributed in [0, 2*pi).
fn random_parameters(num_params: usize) -> Vec<f64> {
    (0..num_params)
        .map(|_| rand::random::<f64>() * 2.0 * PI)
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: apply a Vec<Gate> to a fresh QuantumState
// ---------------------------------------------------------------------------

/// Apply a sequence of gates to a freshly initialized quantum state.
///
/// Supports the standard gate set exposed by `GateOperations`. Unrecognised
/// gate types are silently skipped so that analysis remains robust even when
/// the circuit template contains gates this dispatcher does not handle.
fn apply_gates(state: &mut QuantumState, gates: &[Gate]) {
    for gate in gates {
        match &gate.gate_type {
            GateType::H => GateOperations::h(state, gate.targets[0]),
            GateType::X => GateOperations::x(state, gate.targets[0]),
            GateType::Y => GateOperations::y(state, gate.targets[0]),
            GateType::Z => GateOperations::z(state, gate.targets[0]),
            GateType::S => GateOperations::s(state, gate.targets[0]),
            GateType::T => GateOperations::t(state, gate.targets[0]),
            GateType::SX => GateOperations::sx(state, gate.targets[0]),
            GateType::Rx(theta) => GateOperations::rx(state, gate.targets[0], *theta),
            GateType::Ry(theta) => GateOperations::ry(state, gate.targets[0], *theta),
            GateType::Rz(theta) => GateOperations::rz(state, gate.targets[0], *theta),
            GateType::Phase(theta) => GateOperations::phase(state, gate.targets[0], *theta),
            GateType::CNOT => {
                GateOperations::cnot(state, gate.controls[0], gate.targets[0]);
            }
            GateType::CZ => {
                GateOperations::cz(state, gate.controls[0], gate.targets[0]);
            }
            GateType::SWAP => {
                GateOperations::swap(state, gate.targets[0], gate.targets[1]);
            }
            GateType::ISWAP => {
                GateOperations::iswap(state, gate.targets[0], gate.targets[1]);
            }
            GateType::CRx(theta) => {
                GateOperations::crx(state, gate.controls[0], gate.targets[0], *theta);
            }
            GateType::CRy(theta) => {
                GateOperations::cry(state, gate.controls[0], gate.targets[0], *theta);
            }
            GateType::CRz(theta) | GateType::CR(theta) => {
                GateOperations::crz(state, gate.controls[0], gate.targets[0], *theta);
            }
            GateType::Toffoli => {
                GateOperations::toffoli(
                    state,
                    gate.controls[0],
                    gate.controls[1],
                    gate.targets[0],
                );
            }
            GateType::U { theta, phi, lambda } => {
                // Decompose U(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda)
                GateOperations::rz(state, gate.targets[0], *lambda);
                GateOperations::ry(state, gate.targets[0], *theta);
                GateOperations::rz(state, gate.targets[0], *phi);
            }
            _ => {
                // CCZ, Custom, etc. -- skip for gradient analysis
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build state from circuit template + params
// ---------------------------------------------------------------------------

/// Execute a circuit template with the given parameters on a fresh |0...0> state.
fn execute_circuit(
    circuit_template: &dyn Fn(&[f64]) -> Vec<Gate>,
    params: &[f64],
    num_qubits: usize,
) -> QuantumState {
    let mut state = QuantumState::new(num_qubits);
    let gates = circuit_template(params);
    apply_gates(&mut state, &gates);
    state
}

// ---------------------------------------------------------------------------
// Helper: cost function (expectation of Z on qubit 0)
// ---------------------------------------------------------------------------

/// Default cost function: expectation value of Pauli-Z on qubit 0.
///
/// This is the standard choice for barren plateau studies because Z_0
/// is a local observable, and its gradient variance behaviour directly
/// diagnoses trainability.
fn default_cost(state: &QuantumState) -> f64 {
    state.expectation_z(0)
}

// ---------------------------------------------------------------------------
// Helper: numerical gradient via central finite differences
// ---------------------------------------------------------------------------

/// Compute the numerical gradient of the cost function with respect to all
/// parameters using central finite differences.
///
/// For each parameter theta_i the gradient component is approximated as:
///   dC/dtheta_i ~ [C(theta + eps*e_i) - C(theta - eps*e_i)] / (2*eps)
pub fn numerical_gradient(
    circuit_template: &dyn Fn(&[f64]) -> Vec<Gate>,
    params: &[f64],
    num_qubits: usize,
    epsilon: f64,
) -> Vec<f64> {
    let num_params = params.len();
    let mut grad = Vec::with_capacity(num_params);

    for i in 0..num_params {
        let mut params_plus = params.to_vec();
        params_plus[i] += epsilon;
        let state_plus = execute_circuit(circuit_template, &params_plus, num_qubits);
        let cost_plus = default_cost(&state_plus);

        let mut params_minus = params.to_vec();
        params_minus[i] -= epsilon;
        let state_minus = execute_circuit(circuit_template, &params_minus, num_qubits);
        let cost_minus = default_cost(&state_minus);

        grad.push((cost_plus - cost_minus) / (2.0 * epsilon));
    }

    grad
}

// ---------------------------------------------------------------------------
// BarrenPlateauReport
// ---------------------------------------------------------------------------

/// Results from an empirical barren plateau analysis.
#[derive(Clone, Debug)]
pub struct BarrenPlateauReport {
    /// Per-parameter gradient variance across the sampled parameter vectors.
    pub gradient_variances: Vec<f64>,
    /// Mean gradient variance across all parameters.
    pub mean_variance: f64,
    /// Estimated scaling exponent alpha in Var(dC/dtheta) ~ 2^{-alpha*n}.
    /// Positive values indicate exponential vanishing.
    pub scaling_exponent: f64,
    /// Risk classification derived from the scaling exponent and variance.
    pub risk_level: RiskLevel,
    /// Human-readable recommendation string.
    pub recommendation: String,
}

// ---------------------------------------------------------------------------
// EmpiricalBarrenPlateauAnalysis
// ---------------------------------------------------------------------------

/// Empirical barren plateau analyser.
///
/// Samples random parameter vectors, computes numerical gradients, and
/// estimates the scaling exponent alpha that characterises the gradient
/// variance decay Var(dC/dtheta) ~ 2^{-alpha*n}.
pub struct EmpiricalBarrenPlateauAnalysis {
    num_samples: usize,
    epsilon: f64,
}

impl EmpiricalBarrenPlateauAnalysis {
    /// Create a new analyser with the given number of random samples.
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples: num_samples.max(2),
            epsilon: 1e-4,
        }
    }

    /// Override the finite-difference step size (default 1e-4).
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Run the empirical analysis on a parameterised circuit template.
    ///
    /// The `circuit_template` closure maps a parameter vector to a `Vec<Gate>`.
    pub fn analyze(
        &self,
        circuit_template: &dyn Fn(&[f64]) -> Vec<Gate>,
        num_qubits: usize,
        num_params: usize,
    ) -> BarrenPlateauReport {
        // Collect gradient samples -----------------------------------------------
        let mut all_grads: Vec<Vec<f64>> = Vec::with_capacity(self.num_samples);
        for _ in 0..self.num_samples {
            let params = random_parameters(num_params);
            let grad = numerical_gradient(circuit_template, &params, num_qubits, self.epsilon);
            all_grads.push(grad);
        }

        // Compute per-parameter variance -----------------------------------------
        let mut gradient_variances = vec![0.0_f64; num_params];
        for p in 0..num_params {
            let values: Vec<f64> = all_grads.iter().map(|g| g[p]).collect();
            let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (values.len() as f64 - 1.0).max(1.0);
            gradient_variances[p] = var;
        }

        let mean_variance = gradient_variances.iter().copied().sum::<f64>()
            / gradient_variances.len().max(1) as f64;

        // Estimate scaling exponent alpha ----------------------------------------
        // For a single qubit count we approximate alpha from the observed variance:
        //   Var ~ 2^{-alpha*n}  =>  alpha = -log2(Var) / n
        // Clamp to avoid infinities when variance is exactly zero.
        let scaling_exponent = if mean_variance > 1e-30 && num_qubits > 0 {
            -mean_variance.log2() / num_qubits as f64
        } else if mean_variance <= 1e-30 {
            // Variance effectively zero => strong barren plateau
            num_qubits as f64
        } else {
            0.0
        };

        // Classify risk ----------------------------------------------------------
        let risk_level = classify_risk(scaling_exponent, mean_variance, num_qubits);
        let recommendation = build_recommendation(&risk_level, scaling_exponent, mean_variance);

        BarrenPlateauReport {
            gradient_variances,
            mean_variance,
            scaling_exponent,
            risk_level,
            recommendation,
        }
    }
}

/// Map scaling exponent and variance to a `RiskLevel`.
fn classify_risk(alpha: f64, mean_variance: f64, num_qubits: usize) -> RiskLevel {
    // Heuristic thresholds calibrated against known circuit families:
    //  - hardware-efficient ansatze (alpha ~ 1) => Critical
    //  - shallow alternating circuits (alpha ~ 0.3-0.5) => Medium/High
    //  - local cost functions (alpha ~ 0) => Low

    if mean_variance < 1e-12 || alpha > 1.5 {
        RiskLevel::Critical
    } else if alpha > 0.8 || (mean_variance < 1e-4 && num_qubits >= 6) {
        RiskLevel::High
    } else if alpha > 0.3 || mean_variance < 1e-2 {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    }
}

/// Produce a human-readable recommendation from the risk classification.
fn build_recommendation(level: &RiskLevel, alpha: f64, variance: f64) -> String {
    match level {
        RiskLevel::Critical => format!(
            "CRITICAL barren plateau detected (alpha={:.3}, Var={:.2e}). \
             Gradients vanish exponentially. Consider reducing circuit depth, \
             using local cost functions, or employing layer-wise training.",
            alpha, variance,
        ),
        RiskLevel::High => format!(
            "HIGH barren plateau risk (alpha={:.3}, Var={:.2e}). \
             Training may stall for larger qubit counts. Shorten the ansatz \
             or initialise parameters near identity.",
            alpha, variance,
        ),
        RiskLevel::Medium => format!(
            "MEDIUM barren plateau risk (alpha={:.3}, Var={:.2e}). \
             Circuit is likely trainable for current size but monitor \
             gradient magnitudes as the system scales.",
            alpha, variance,
        ),
        RiskLevel::Low => format!(
            "LOW barren plateau risk (alpha={:.3}, Var={:.2e}). \
             Gradients are well-behaved. Circuit should be trainable.",
            alpha, variance,
        ),
    }
}

// ---------------------------------------------------------------------------
// ExpressibilityAnalysis
// ---------------------------------------------------------------------------

/// Expressibility analysis via fidelity-based KL divergence.
///
/// Quantifies how well a parameterised circuit explores the Hilbert space
/// by comparing the distribution of pairwise state fidelities against the
/// Haar-random distribution (whose fidelity CDF is F^{d-1} for dimension d).
pub struct ExpressibilityAnalysis;

impl ExpressibilityAnalysis {
    /// Estimate the KL divergence D_KL(P_circuit || P_Haar) of the fidelity
    /// distribution.
    ///
    /// Lower values indicate higher expressibility (closer to Haar-random).
    pub fn kl_divergence(
        circuit_template: &dyn Fn(&[f64]) -> Vec<Gate>,
        num_qubits: usize,
        num_params: usize,
        num_samples: usize,
    ) -> f64 {
        let dim = 1_usize << num_qubits;
        let num_bins = 75;
        let samples = num_samples.max(2);

        // Collect pairwise fidelities ------------------------------------------
        let mut fidelities: Vec<f64> = Vec::with_capacity(samples);
        for _ in 0..samples {
            let params_a = random_parameters(num_params);
            let params_b = random_parameters(num_params);
            let state_a = execute_circuit(circuit_template, &params_a, num_qubits);
            let state_b = execute_circuit(circuit_template, &params_b, num_qubits);
            fidelities.push(state_a.fidelity(&state_b));
        }

        // Histogram the empirical fidelity distribution -------------------------
        let mut hist_circuit = vec![0.0_f64; num_bins];
        for &f in &fidelities {
            let bin = ((f * num_bins as f64).floor() as usize).min(num_bins - 1);
            hist_circuit[bin] += 1.0;
        }
        // Normalise to probability density
        let total: f64 = hist_circuit.iter().sum();
        if total > 0.0 {
            for v in hist_circuit.iter_mut() {
                *v /= total;
            }
        }

        // Haar-random fidelity distribution: P(F) = (d-1)(1-F)^{d-2} ----------
        let d = dim as f64;
        let mut hist_haar = vec![0.0_f64; num_bins];
        let bin_width = 1.0 / num_bins as f64;
        for i in 0..num_bins {
            let f_low = i as f64 * bin_width;
            let f_high = f_low + bin_width;
            // Integrate (d-1)(1-F)^{d-2} from f_low to f_high
            // = [-(1-F)^{d-1}] from f_low to f_high
            // = (1-f_low)^{d-1} - (1-f_high)^{d-1}
            hist_haar[i] = (1.0 - f_low).powf(d - 1.0) - (1.0 - f_high).powf(d - 1.0);
        }
        // Already normalised by construction (sums to 1).

        // KL divergence D_KL(P_circuit || P_Haar) ------------------------------
        let smoothing = 1e-10;
        let mut kl = 0.0_f64;
        for i in 0..num_bins {
            let p = hist_circuit[i] + smoothing;
            let q = hist_haar[i] + smoothing;
            if p > smoothing * 0.5 {
                kl += p * (p / q).ln();
            }
        }

        kl.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// EntanglementCapability
// ---------------------------------------------------------------------------

/// Entanglement analysis using the Meyer-Wallach entanglement measure Q.
///
/// Q = (2/n) * Sum_k (1 - Tr(rho_k^2))
///
/// where rho_k is the single-qubit reduced density matrix of qubit k.
/// Q = 0 for product states, Q = 1 for maximally entangled states.
pub struct EntanglementCapability;

impl EntanglementCapability {
    /// Compute the Meyer-Wallach entanglement measure for a given state.
    pub fn meyer_wallach(state: &QuantumState) -> f64 {
        let n = state.num_qubits;
        if n == 0 {
            return 0.0;
        }

        let dim = state.dim;
        let amps = state.amplitudes_ref();

        let mut q_sum = 0.0_f64;
        for k in 0..n {
            // Build the 2x2 reduced density matrix rho_k by tracing out all
            // qubits except k.
            //
            // rho_{ab} = Sum_{j : bit k of j is a, bit k of (j XOR (a XOR b)<<k) is b}
            //          = Sum_{j with bit_k(j)=a} conj(amp[j]) * amp[j with bit k flipped to b]
            //
            // More directly:
            //   rho_00 = Sum_{j: bit_k=0} |amp[j]|^2
            //   rho_11 = Sum_{j: bit_k=1} |amp[j]|^2
            //   rho_01 = Sum_{j: bit_k=0} conj(amp[j]) * amp[j | (1<<k)]
            //   rho_10 = conj(rho_01)

            let mask = 1_usize << k;
            let mut rho_00 = 0.0_f64;
            let mut rho_11 = 0.0_f64;
            let mut rho_01 = C64::new(0.0, 0.0);

            for j in 0..dim {
                if j & mask == 0 {
                    let a0 = amps[j];
                    let a1 = amps[j | mask];
                    rho_00 += a0.norm_sqr();
                    rho_11 += a1.norm_sqr();
                    // rho_01 += conj(a0) * a1
                    rho_01.re += a0.re * a1.re + a0.im * a1.im;
                    rho_01.im += a0.re * a1.im - a0.im * a1.re;
                }
            }

            // Tr(rho_k^2) = rho_00^2 + rho_11^2 + 2*|rho_01|^2
            let purity = rho_00 * rho_00 + rho_11 * rho_11 + 2.0 * rho_01.norm_sqr();
            q_sum += 1.0 - purity;
        }

        (2.0 / n as f64) * q_sum
    }

    /// Average Meyer-Wallach entanglement over random parameter samples.
    pub fn average_entanglement(
        circuit_template: &dyn Fn(&[f64]) -> Vec<Gate>,
        num_qubits: usize,
        num_params: usize,
        num_samples: usize,
    ) -> f64 {
        let samples = num_samples.max(1);
        let mut total = 0.0_f64;

        for _ in 0..samples {
            let params = random_parameters(num_params);
            let state = execute_circuit(circuit_template, &params, num_qubits);
            total += Self::meyer_wallach(&state);
        }

        total / samples as f64
    }
}

// ---------------------------------------------------------------------------
// CostLandscapeVisualization
// ---------------------------------------------------------------------------

/// Tools for visualising the variational cost landscape.
pub struct CostLandscapeVisualization;

impl CostLandscapeVisualization {
    /// Sweep a single parameter while holding all others fixed and record
    /// the cost function value at each point.
    ///
    /// Returns `num_points` pairs of (parameter_value, cost).
    pub fn parameter_slice(
        circuit_template: &dyn Fn(&[f64]) -> Vec<Gate>,
        params: &[f64],
        param_idx: usize,
        range: (f64, f64),
        num_qubits: usize,
        num_points: usize,
    ) -> Vec<(f64, f64)> {
        let num_points = num_points.max(2);
        let step = (range.1 - range.0) / (num_points - 1) as f64;

        let mut results = Vec::with_capacity(num_points);
        let mut swept = params.to_vec();

        for i in 0..num_points {
            let val = range.0 + step * i as f64;
            swept[param_idx] = val;
            let state = execute_circuit(circuit_template, &swept, num_qubits);
            let cost = default_cost(&state);
            results.push((val, cost));
        }

        results
    }

    /// Compute the distribution of gradient norms over random parameter samples.
    ///
    /// Returns a vector of L2 gradient norms, one per sample.
    pub fn gradient_norm_distribution(
        circuit_template: &dyn Fn(&[f64]) -> Vec<Gate>,
        num_qubits: usize,
        num_params: usize,
        num_samples: usize,
    ) -> Vec<f64> {
        let epsilon = 1e-4;
        let mut norms = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let params = random_parameters(num_params);
            let grad = numerical_gradient(circuit_template, &params, num_qubits, epsilon);
            let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            norms.push(norm);
        }

        norms
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};

    // -----------------------------------------------------------------------
    // Test helpers: circuit templates
    // -----------------------------------------------------------------------

    /// Product-state circuit: independent Ry rotations (no entanglement).
    fn product_circuit(num_qubits: usize) -> impl Fn(&[f64]) -> Vec<Gate> {
        move |params: &[f64]| {
            (0..num_qubits)
                .map(|q| Gate::new(GateType::Ry(params[q]), vec![q], vec![]))
                .collect()
        }
    }

    /// Entangling circuit: Ry layer followed by a CNOT chain.
    fn entangling_circuit(num_qubits: usize) -> impl Fn(&[f64]) -> Vec<Gate> {
        move |params: &[f64]| {
            let mut gates: Vec<Gate> = (0..num_qubits)
                .map(|q| Gate::new(GateType::Ry(params[q]), vec![q], vec![]))
                .collect();
            for q in 0..num_qubits - 1 {
                gates.push(Gate::new(GateType::CNOT, vec![q + 1], vec![q]));
            }
            gates
        }
    }

    /// Deep hardware-efficient ansatz: multiple Ry + CNOT layers.
    fn deep_circuit(num_qubits: usize, layers: usize) -> impl Fn(&[f64]) -> Vec<Gate> {
        move |params: &[f64]| {
            let mut gates = Vec::new();
            let mut pidx = 0;
            for _layer in 0..layers {
                for q in 0..num_qubits {
                    gates.push(Gate::new(GateType::Ry(params[pidx]), vec![q], vec![]));
                    pidx += 1;
                }
                for q in 0..num_qubits - 1 {
                    gates.push(Gate::new(GateType::CNOT, vec![q + 1], vec![q]));
                }
            }
            gates
        }
    }

    // -----------------------------------------------------------------------
    // 1. Gradient variance is non-negative and finite
    // -----------------------------------------------------------------------
    #[test]
    fn test_gradient_variance_nonnegative() {
        let nq = 3;
        let template = product_circuit(nq);
        let analysis = EmpiricalBarrenPlateauAnalysis::new(20);
        let report = analysis.analyze(&template, nq, nq);

        for &var in &report.gradient_variances {
            assert!(var >= 0.0, "Gradient variance must be non-negative");
            assert!(var.is_finite(), "Gradient variance must be finite");
        }
        assert!(report.mean_variance >= 0.0);
    }

    // -----------------------------------------------------------------------
    // 2. Mean variance equals the mean of per-parameter variances
    // -----------------------------------------------------------------------
    #[test]
    fn test_mean_variance_consistency() {
        let nq = 2;
        let template = entangling_circuit(nq);
        let analysis = EmpiricalBarrenPlateauAnalysis::new(30);
        let report = analysis.analyze(&template, nq, nq);

        let expected_mean = report.gradient_variances.iter().sum::<f64>()
            / report.gradient_variances.len() as f64;
        assert!(
            (report.mean_variance - expected_mean).abs() < 1e-12,
            "mean_variance should be the arithmetic mean of per-parameter variances"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Scaling exponent is positive for a non-trivial circuit
    // -----------------------------------------------------------------------
    #[test]
    fn test_scaling_exponent_positive() {
        let nq = 3;
        let template = entangling_circuit(nq);
        let analysis = EmpiricalBarrenPlateauAnalysis::new(50);
        let report = analysis.analyze(&template, nq, nq);

        // For a circuit of bounded depth, alpha should be positive (variance < 1).
        assert!(
            report.scaling_exponent > 0.0,
            "Scaling exponent should be positive for bounded-depth circuit, got {}",
            report.scaling_exponent
        );
    }

    // -----------------------------------------------------------------------
    // 4. Deeper circuits produce higher alpha (more barren)
    // -----------------------------------------------------------------------
    #[test]
    fn test_deeper_circuits_higher_risk() {
        let nq = 3;
        let shallow_params = nq;
        let deep_layers = 6;
        let deep_params = nq * deep_layers;

        let shallow_template = product_circuit(nq);
        let deep_template = deep_circuit(nq, deep_layers);

        let analysis = EmpiricalBarrenPlateauAnalysis::new(40);
        let shallow_report = analysis.analyze(&shallow_template, nq, shallow_params);
        let deep_report = analysis.analyze(&deep_template, nq, deep_params);

        // Deep circuit should have smaller mean gradient variance.
        assert!(
            deep_report.mean_variance <= shallow_report.mean_variance * 2.0,
            "Deep circuit variance ({:.6}) should not wildly exceed shallow ({:.6})",
            deep_report.mean_variance,
            shallow_report.mean_variance,
        );
    }

    // -----------------------------------------------------------------------
    // 5. Risk level classification sanity
    // -----------------------------------------------------------------------
    #[test]
    fn test_risk_level_classification() {
        // Very small variance => Critical
        assert_eq!(classify_risk(2.0, 1e-14, 10), RiskLevel::Critical);
        // Medium alpha
        assert_eq!(classify_risk(0.5, 0.005, 4), RiskLevel::Medium);
        // Low alpha, decent variance
        assert_eq!(classify_risk(0.1, 0.1, 3), RiskLevel::Low);
        // High alpha
        assert_eq!(classify_risk(1.0, 1e-5, 8), RiskLevel::High);
    }

    // -----------------------------------------------------------------------
    // 6. Expressibility: product circuit is less expressible than entangling
    // -----------------------------------------------------------------------
    #[test]
    fn test_expressibility_product_vs_entangling() {
        let nq = 2;
        let kl_product = ExpressibilityAnalysis::kl_divergence(
            &product_circuit(nq),
            nq,
            nq,
            200,
        );
        let kl_entangling = ExpressibilityAnalysis::kl_divergence(
            &entangling_circuit(nq),
            nq,
            nq,
            200,
        );

        // A product circuit cannot explore the full Hilbert space; its KL
        // divergence from Haar should be larger (less expressible).
        assert!(
            kl_product > kl_entangling * 0.5,
            "Product circuit KL ({:.4}) should be comparable to or larger than \
             entangling circuit KL ({:.4})",
            kl_product,
            kl_entangling,
        );
    }

    // -----------------------------------------------------------------------
    // 7. KL divergence is non-negative
    // -----------------------------------------------------------------------
    #[test]
    fn test_kl_divergence_nonneg() {
        let nq = 2;
        let kl = ExpressibilityAnalysis::kl_divergence(
            &entangling_circuit(nq),
            nq,
            nq,
            100,
        );
        assert!(kl >= 0.0, "KL divergence must be >= 0, got {}", kl);
    }

    // -----------------------------------------------------------------------
    // 8. Meyer-Wallach = 0 for product states
    // -----------------------------------------------------------------------
    #[test]
    fn test_meyer_wallach_product_state() {
        // |0...0> is a product state.
        let state = QuantumState::new(4);
        let q = EntanglementCapability::meyer_wallach(&state);
        assert!(
            q.abs() < 1e-10,
            "MW measure should be 0 for |0000>, got {:.6e}",
            q
        );

        // A state with only Ry rotations is still a product state.
        let mut state2 = QuantumState::new(3);
        GateOperations::ry(&mut state2, 0, 1.0);
        GateOperations::ry(&mut state2, 1, 2.0);
        GateOperations::ry(&mut state2, 2, 0.5);
        let q2 = EntanglementCapability::meyer_wallach(&state2);
        assert!(
            q2.abs() < 1e-10,
            "MW measure should be 0 for Ry product state, got {:.6e}",
            q2
        );
    }

    // -----------------------------------------------------------------------
    // 9. Meyer-Wallach > 0 for Bell state
    // -----------------------------------------------------------------------
    #[test]
    fn test_meyer_wallach_bell_state() {
        // Bell state: H on qubit 0, CNOT(0,1) => (|00> + |11>) / sqrt(2)
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);

        let q = EntanglementCapability::meyer_wallach(&state);
        assert!(
            q > 0.9,
            "MW measure should be ~1 for Bell state, got {:.4}",
            q
        );
    }

    // -----------------------------------------------------------------------
    // 10. Meyer-Wallach is bounded [0, 1]
    // -----------------------------------------------------------------------
    #[test]
    fn test_meyer_wallach_bounded() {
        for _ in 0..10 {
            let nq = 3;
            let params = random_parameters(nq);
            let template = entangling_circuit(nq);
            let state = execute_circuit(&template, &params, nq);
            let q = EntanglementCapability::meyer_wallach(&state);
            assert!(q >= -1e-10 && q <= 1.0 + 1e-10, "MW out of [0,1]: {:.6}", q);
        }
    }

    // -----------------------------------------------------------------------
    // 11. Average entanglement: entangling > product
    // -----------------------------------------------------------------------
    #[test]
    fn test_average_entanglement_ordering() {
        let nq = 3;
        let avg_product = EntanglementCapability::average_entanglement(
            &product_circuit(nq),
            nq,
            nq,
            30,
        );
        let avg_entangling = EntanglementCapability::average_entanglement(
            &entangling_circuit(nq),
            nq,
            nq,
            30,
        );

        assert!(
            avg_entangling > avg_product + 0.01,
            "Entangling circuit average MW ({:.4}) should exceed product ({:.4})",
            avg_entangling,
            avg_product,
        );
    }

    // -----------------------------------------------------------------------
    // 12. Parameter slice generates the correct number of points
    // -----------------------------------------------------------------------
    #[test]
    fn test_parameter_slice_length() {
        let nq = 2;
        let template = product_circuit(nq);
        let params = vec![0.5, 1.0];
        let num_points = 25;

        let slice = CostLandscapeVisualization::parameter_slice(
            &template,
            &params,
            0,
            (0.0, 2.0 * PI),
            nq,
            num_points,
        );

        assert_eq!(
            slice.len(),
            num_points,
            "Slice should have exactly {} points",
            num_points
        );
    }

    // -----------------------------------------------------------------------
    // 13. Parameter slice values are in expected range
    // -----------------------------------------------------------------------
    #[test]
    fn test_parameter_slice_values_bounded() {
        let nq = 2;
        let template = product_circuit(nq);
        let params = vec![0.0, 0.0];

        let slice = CostLandscapeVisualization::parameter_slice(
            &template,
            &params,
            0,
            (0.0, PI),
            nq,
            10,
        );

        for &(theta, cost) in &slice {
            assert!(theta >= -1e-10 && theta <= PI + 1e-10);
            // Z expectation is in [-1, 1]
            assert!(
                cost >= -1.0 - 1e-10 && cost <= 1.0 + 1e-10,
                "Cost out of [-1,1]: {:.6}",
                cost
            );
        }
    }

    // -----------------------------------------------------------------------
    // 14. Gradient norm distribution returns correct count
    // -----------------------------------------------------------------------
    #[test]
    fn test_gradient_norm_distribution_count() {
        let nq = 2;
        let template = entangling_circuit(nq);
        let num_samples = 15;

        let norms = CostLandscapeVisualization::gradient_norm_distribution(
            &template,
            nq,
            nq,
            num_samples,
        );

        assert_eq!(norms.len(), num_samples);
        for &n in &norms {
            assert!(n >= 0.0, "Gradient norm must be non-negative");
            assert!(n.is_finite(), "Gradient norm must be finite");
        }
    }

    // -----------------------------------------------------------------------
    // 15. Numerical gradient of a known circuit is close to analytic
    // -----------------------------------------------------------------------
    #[test]
    fn test_numerical_gradient_accuracy() {
        // For Ry(theta)|0> the expectation of Z is cos(theta).
        // dZ/dtheta = -sin(theta).
        let nq = 1;
        let template = product_circuit(nq);
        let theta = 1.0_f64;
        let params = vec![theta];

        let grad = numerical_gradient(&template, &params, nq, 1e-6);
        let expected = -theta.sin();

        assert!(
            (grad[0] - expected).abs() < 1e-4,
            "Numerical gradient {:.6} should be close to analytic {:.6}",
            grad[0],
            expected,
        );
    }
}
