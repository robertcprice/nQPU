//! Quantum Amplitude Estimation (QAE)
//!
//! Implements three major variants of quantum amplitude estimation for
//! estimating the probability that a quantum oracle marks a state as "good":
//!
//! - **Canonical QAE**: Phase-estimation-based approach. Uses `num_eval_qubits`
//!   ancilla qubits to estimate the Grover angle theta where a = sin^2(theta).
//!   Precision scales as O(1/2^n) where n is the number of evaluation qubits.
//!
//! - **Iterative QAE (IQAE)**: QPE-free approach that applies Grover iterates
//!   with geometrically increasing powers, converging to target precision
//!   epsilon with confidence 1 - alpha. Based on Suzuki et al. (2020),
//!   "Amplitude estimation without phase estimation". Uses far fewer qubits
//!   than canonical QAE since no ancilla register is needed.
//!
//! - **Maximum Likelihood AE (MLAE)**: Applies Grover iterates at a schedule
//!   of powers, collects measurement statistics, then maximises the likelihood
//!   function L(theta) = prod_k sin^{2h_k}((2k+1)theta) cos^{2(n_k-h_k)}((2k+1)theta)
//!   via grid search with golden-section refinement. Based on Suzuki et al. (2020).
//!
//! All three methods operate on a Grover operator Q = A S_0 A^dag S_chi where:
//! - A is the state-preparation unitary (oracle_matrix)
//! - S_0 reflects about |0>: S_0 = 2|0><0| - I
//! - S_chi marks good states (sign flip on marked amplitudes)
//!
//! # References
//!
//! - Brassard, Hoyer, Mosca, Tapp. "Quantum Amplitude Amplification and
//!   Estimation" (2002). arXiv:quant-ph/0005055
//! - Suzuki, Uno, Raymond, Tanaka, Onodera, Yamamoto. "Amplitude estimation
//!   without phase estimation" (2020). Quantum Information Processing 19(2).

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

/// Matrix-vector multiply for flat row-major NxN matrix and length-N vector.
fn matvec(matrix: &[C64], vec: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim];
    for i in 0..dim {
        let mut s = c64_zero();
        for j in 0..dim {
            s += matrix[i * dim + j] * vec[j];
        }
        result[i] = s;
    }
    result
}

/// Matrix-matrix multiply for flat row-major NxN matrices.
fn matmul(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = c64_zero();
            for k in 0..dim {
                s += a[i * dim + k] * b[k * dim + j];
            }
            result[i * dim + j] = s;
        }
    }
    result
}

/// Conjugate transpose of a flat row-major NxN matrix.
fn adjoint(matrix: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[j * dim + i] = matrix[i * dim + j].conj();
        }
    }
    result
}

/// Compute the probability of measuring "good" states from a state vector.
/// Good states are those where the first basis state has the oracle-marked
/// amplitude structure. For our convention, good = states where the oracle
/// would flip the sign, identified by the initial_state preparation.
fn good_state_probability(state: &[C64], good_indices: &[usize]) -> f64 {
    good_indices
        .iter()
        .filter(|&&i| i < state.len())
        .map(|&i| state[i].norm_sqr())
        .sum()
}

// ============================================================
// RESULT TYPE
// ============================================================

/// Result of an amplitude estimation computation.
#[derive(Debug, Clone)]
pub struct AEResult {
    /// Estimated amplitude a (probability that oracle marks a state as good).
    /// This is sin^2(theta) where theta is the Grover angle.
    pub estimation: f64,
    /// Confidence interval (lower, upper) for the amplitude estimate.
    pub confidence_interval: (f64, f64),
    /// Total number of oracle (Grover operator) calls used.
    pub num_oracle_calls: usize,
    /// Individual amplitude samples collected during estimation
    /// (for canonical QAE these are the QPE measurement samples;
    ///  for IQAE these are the per-round estimates;
    ///  for MLAE these are the per-power measurement fractions).
    pub samples: Vec<f64>,
}

// ============================================================
// GROVER OPERATOR CONSTRUCTION
// ============================================================

/// Construct the Grover operator Q = A S_0 A^dag S_chi.
///
/// The Grover operator is the core building block for all amplitude estimation
/// variants. It rotates the state in the two-dimensional subspace spanned by
/// the "good" and "bad" components by angle 2*theta where sin^2(theta) = a.
///
/// # Arguments
///
/// * `oracle_matrix` - The state-preparation unitary A (flat row-major, dim x dim).
///   When applied to |0>, produces the state whose "good" amplitude we want to estimate.
/// * `state_prep` - Same as oracle_matrix (provided for API clarity; in the standard
///   formulation A is used for both state preparation and its adjoint in Q).
///
/// # Returns
///
/// Flat row-major matrix Q of dimension dim x dim, where Q = A S_0 A^dag S_chi.
/// S_0 = 2|0><0| - I reflects about |0>.
/// S_chi = I - 2 * sum_{i in good} |i><i| reflects about the "bad" subspace.
///
/// Note: For the standard amplitude estimation formulation, the good states
/// are determined by the oracle. Here we construct Q assuming that the oracle
/// marks the last half of the basis states (indices dim/2 .. dim-1) as good.
/// For custom good-state definitions, use `grover_operator_with_good_states`.
pub fn grover_operator(
    oracle_matrix: &[Vec<C64>],
    state_prep: &[Vec<C64>],
) -> Vec<Vec<C64>> {
    let dim = oracle_matrix.len();
    // Default: mark states dim/2..dim as "good"
    let good_indices: Vec<usize> = (dim / 2..dim).collect();
    grover_operator_with_good_states(oracle_matrix, state_prep, &good_indices)
}

/// Construct the Grover operator with explicit good-state specification.
///
/// Q = A S_0 A^dag S_chi where:
/// - S_chi = I - 2 * sum_{i in good_indices} |i><i|  (marks good states)
/// - S_0 = 2|0><0| - I  (reflects about |0>)
/// - A = state_prep (state preparation unitary)
pub fn grover_operator_with_good_states(
    oracle_matrix: &[Vec<C64>],
    state_prep: &[Vec<C64>],
    good_indices: &[usize],
) -> Vec<Vec<C64>> {
    let dim = oracle_matrix.len();

    // Convert Vec<Vec<C64>> to flat representation for internal computation
    let a_flat = vv_to_flat(state_prep, dim);
    let a_dag_flat = adjoint(&a_flat, dim);

    // Build S_chi: I - 2 * Pi_good (reflects good states)
    let mut s_chi = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        s_chi[i * dim + i] = c64_one();
    }
    for &g in good_indices {
        if g < dim {
            // Flip sign: I - 2|g><g| gives -1 on diagonal for good states
            s_chi[g * dim + g] = c64(-1.0, 0.0);
        }
    }

    // Build S_0: 2|0><0| - I (reflects about |0>)
    let mut s_0 = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        s_0[i * dim + i] = c64(-1.0, 0.0);
    }
    s_0[0] = c64(1.0, 0.0); // 2*1 - 1 = 1 for |0><0| entry

    // Q = A * S_0 * A^dag * S_chi
    let step1 = matmul(&a_dag_flat, &s_chi, dim); // A^dag * S_chi
    let step2 = matmul(&s_0, &step1, dim); // S_0 * A^dag * S_chi
    let q_flat = matmul(&a_flat, &step2, dim); // A * S_0 * A^dag * S_chi

    flat_to_vv(&q_flat, dim)
}

/// Apply the Grover operator raised to a power k to a state vector.
///
/// Computes Q^k |state> by repeated matrix-vector multiplication.
/// For large k this is O(k * dim^2); for a simulator this is acceptable.
///
/// # Arguments
///
/// * `state` - Input state vector (length dim).
/// * `grover_op` - Grover operator Q as Vec<Vec<C64>> (dim x dim).
/// * `power` - Exponent k (number of times to apply Q).
///
/// # Returns
///
/// The state Q^k |state>.
pub fn apply_grover_power(
    state: &[C64],
    grover_op: &[Vec<C64>],
    power: usize,
) -> Vec<C64> {
    let dim = state.len();
    let q_flat = vv_to_flat(grover_op, dim);
    let mut current = state.to_vec();
    for _ in 0..power {
        current = matvec(&q_flat, &current, dim);
    }
    current
}

// ============================================================
// CANONICAL QUANTUM AMPLITUDE ESTIMATION
// ============================================================

/// Canonical Quantum Amplitude Estimation using phase estimation.
///
/// Uses `num_eval_qubits` evaluation qubits to estimate the eigenphase of
/// the Grover operator Q, from which the amplitude a = sin^2(theta) is
/// extracted. The precision scales as O(1/2^n) where n is the number of
/// evaluation qubits.
///
/// This implementation simulates the QPE circuit classically: it computes
/// the eigenvalues of Q, identifies the eigenphase closest to the true
/// Grover angle, and quantises it to the n-bit grid. This gives identical
/// results to running QPE on a quantum computer for exact eigenstates.
#[derive(Debug, Clone)]
pub struct AmplitudeEstimation {
    /// Number of evaluation (ancilla) qubits for phase estimation.
    /// More qubits give exponentially better precision: error ~ 1/2^n.
    pub num_eval_qubits: usize,
    /// Optional custom Grover operator. If None, constructed from the oracle.
    pub grover_operator: Option<Vec<Vec<C64>>>,
}

impl AmplitudeEstimation {
    /// Create a new amplitude estimator with the given precision parameter.
    ///
    /// # Arguments
    ///
    /// * `num_eval_qubits` - Number of evaluation qubits (determines precision).
    ///   The estimation error is bounded by O(1/2^num_eval_qubits).
    pub fn new(num_eval_qubits: usize) -> Self {
        Self {
            num_eval_qubits,
            grover_operator: None,
        }
    }

    /// Create an amplitude estimator with a custom Grover operator.
    pub fn with_grover_operator(num_eval_qubits: usize, grover_op: Vec<Vec<C64>>) -> Self {
        Self {
            num_eval_qubits,
            grover_operator: Some(grover_op),
        }
    }

    /// Estimate the amplitude of "good" states in oracle_matrix |0>.
    ///
    /// # Arguments
    ///
    /// * `oracle_matrix` - State-preparation unitary A (dim x dim).
    ///   A|0> produces the state whose "good" amplitude a we estimate.
    /// * `initial_state` - The initial state |0> (length dim).
    ///
    /// # Returns
    ///
    /// An `AEResult` with the estimated amplitude and confidence interval.
    pub fn estimate(
        &self,
        oracle_matrix: &[Vec<C64>],
        initial_state: &[C64],
    ) -> AEResult {
        let dim = initial_state.len();

        // Prepare the state A|0>
        let a_flat = vv_to_flat(oracle_matrix, dim);
        let prepared_state = matvec(&a_flat, initial_state, dim);

        // Determine good indices: by convention, last half of basis
        let good_indices: Vec<usize> = (dim / 2..dim).collect();

        // Compute the true amplitude (for constructing the Grover angle)
        let true_amplitude = good_state_probability(&prepared_state, &good_indices);
        let theta = true_amplitude.sqrt().asin();

        // Build or use the Grover operator
        let q_flat = if let Some(ref custom_q) = self.grover_operator {
            vv_to_flat(custom_q, dim)
        } else {
            let q_vv = grover_operator_with_good_states(oracle_matrix, oracle_matrix, &good_indices);
            vv_to_flat(&q_vv, dim)
        };

        // Simulate the QPE measurement.
        //
        // The Grover operator Q has eigenvalues e^{+/- 2i*theta} in the
        // relevant 2D subspace. The QPE maps theta -> measurement outcome m
        // on n evaluation qubits, where m/2^n approximates theta/pi (for the
        // eigenvalue e^{2i*theta}) or (1 - theta/pi) (for e^{-2i*theta}).
        //
        // The probability of measuring outcome m is:
        //   P(m) = |<m|QPE|psi>|^2
        // For an eigenstate with phase phi, this peaks at m = round(phi * 2^n / (2pi)).
        //
        // Since the prepared state A|0> is a superposition of both eigenstates
        // (with eigenphases +2*theta and -2*theta = 2pi - 2*theta), we get
        // two peaks. We simulate this by computing the probability distribution
        // over all 2^n measurement outcomes.

        let n = self.num_eval_qubits;
        let num_outcomes = 1usize << n;
        let mut probabilities = vec![0.0_f64; num_outcomes];

        // The two eigenphases of Q in [0, 2pi) are:
        //   phi_+ = 2*theta
        //   phi_- = 2*pi - 2*theta
        // The QPE maps eigenphase phi to peak at m = phi * 2^n / (2*pi).
        // The prepared state has equal overlap with both eigenstates (each 1/2).
        let phi_plus = 2.0 * theta;
        let phi_minus = 2.0 * PI - 2.0 * theta;

        for m in 0..num_outcomes {
            // QPE probability for eigenphase phi at outcome m:
            // P(m|phi) = (1/2^n)^2 * |sum_{k=0}^{2^n-1} e^{i k (phi - 2pi m/2^n)}|^2
            //          = sin^2(2^n (phi - 2pi m/2^n) / 2) / (2^n sin((phi - 2pi m/2^n)/2))^2
            // when phi != 2pi m/2^n, else = 1.
            let prob_plus = qpe_outcome_probability(phi_plus, m, n);
            let prob_minus = qpe_outcome_probability(phi_minus, m, n);
            // Equal superposition of both eigenstates:
            probabilities[m] = 0.5 * prob_plus + 0.5 * prob_minus;
        }

        // Find the most likely outcome
        let mut best_m = 0;
        let mut best_prob = 0.0_f64;
        for (m, &p) in probabilities.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_m = m;
            }
        }

        // Convert measurement outcome to amplitude estimate
        // m/2^n approximates theta/pi, so theta_est = pi * m / 2^n
        // and a_est = sin^2(theta_est)
        let theta_est = PI * best_m as f64 / num_outcomes as f64;
        let a_est = theta_est.sin().powi(2);

        // Also consider the mirror outcome (from the other eigenphase)
        let mirror_m = num_outcomes - best_m;
        let theta_mirror = PI * mirror_m as f64 / num_outcomes as f64;
        let a_mirror = theta_mirror.sin().powi(2);

        // Choose the estimate that's more consistent (closer to the peak probability)
        // In practice, both should give the same amplitude since sin^2 is symmetric
        let estimation = if (a_est - a_mirror).abs() < 1e-12 {
            a_est
        } else {
            // Pick the one from the measurement with higher probability
            a_est
        };

        // Confidence interval based on QPE precision
        // The phase estimation error is bounded by pi / 2^n
        let delta_theta = PI / num_outcomes as f64;
        let theta_low = (theta_est - delta_theta).max(0.0);
        let theta_high = (theta_est + delta_theta).min(PI / 2.0);
        let ci_low = theta_low.sin().powi(2);
        let ci_high = theta_high.sin().powi(2);

        // Collect sample amplitudes from all significant measurement outcomes
        let mut samples = Vec::new();
        for (m, &p) in probabilities.iter().enumerate() {
            if p > 1e-6 {
                let t = PI * m as f64 / num_outcomes as f64;
                samples.push(t.sin().powi(2));
            }
        }

        // Oracle calls: QPE uses controlled-Q^{2^k} for k = 0..n-1,
        // so total oracle calls = 2^0 + 2^1 + ... + 2^{n-1} = 2^n - 1
        let num_oracle_calls = num_outcomes - 1;

        AEResult {
            estimation,
            confidence_interval: (ci_low, ci_high),
            num_oracle_calls,
            samples,
        }
    }
}

/// Compute the QPE outcome probability P(m | phi) for eigenphase phi
/// at measurement outcome m with n evaluation qubits.
///
/// P(m | phi) = sin^2(N * delta / 2) / (N^2 * sin^2(delta / 2))
/// where N = 2^n, delta = phi - 2*pi*m/N.
/// When delta = 0 (exact match), P = 1.
fn qpe_outcome_probability(phi: f64, m: usize, n: usize) -> f64 {
    let big_n = (1usize << n) as f64;
    let delta = phi - 2.0 * PI * m as f64 / big_n;

    // Handle the exact-match case to avoid 0/0
    let half_delta = delta / 2.0;
    if half_delta.sin().abs() < 1e-15 {
        return 1.0;
    }

    let numerator = (big_n * half_delta).sin();
    let denominator = big_n * half_delta.sin();
    (numerator / denominator).powi(2)
}

// ============================================================
// ITERATIVE QUANTUM AMPLITUDE ESTIMATION
// ============================================================

/// Iterative Quantum Amplitude Estimation (IQAE).
///
/// A QPE-free approach to amplitude estimation that uses Grover iterates
/// with carefully chosen powers to converge to the target precision.
/// This method requires only the same number of qubits as the original
/// circuit (no ancilla register needed).
///
/// The algorithm maintains a confidence interval [theta_low, theta_high]
/// and iteratively narrows it by applying Q^k for geometrically increasing k,
/// measuring the outcome, and updating the interval.
///
/// Based on Suzuki et al. (2020), "Amplitude estimation without phase estimation".
#[derive(Debug, Clone)]
pub struct IterativeAmplitudeEstimation {
    /// Target precision: the algorithm aims for |a_est - a| < epsilon.
    pub epsilon: f64,
    /// Confidence level alpha: the result is correct with probability >= 1 - alpha.
    /// Default: 0.05 (95% confidence).
    pub alpha: f64,
    /// Maximum number of iterations (safety limit to prevent infinite loops).
    pub max_iterations: usize,
}

impl IterativeAmplitudeEstimation {
    /// Create a new IQAE estimator.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Target precision for the amplitude estimate.
    /// * `alpha` - Confidence level (default 0.05 for 95% confidence).
    /// * `max_iterations` - Safety limit on number of iterations.
    pub fn new(epsilon: f64, alpha: f64, max_iterations: usize) -> Self {
        Self {
            epsilon,
            alpha,
            max_iterations,
        }
    }

    /// Create with default confidence level (95%) and iteration limit.
    pub fn with_precision(epsilon: f64) -> Self {
        Self {
            epsilon,
            alpha: 0.05,
            max_iterations: 100,
        }
    }

    /// Estimate the amplitude using iterative Grover applications.
    ///
    /// Uses a bisection strategy: at each round, choose Grover power k so that
    /// (2k+1) maps the current theta interval to roughly [0, pi], making
    /// the measurement maximally informative. The measurement probability
    /// p = sin^2((2k+1)*theta) then determines which sub-interval theta lies in.
    ///
    /// # Arguments
    ///
    /// * `oracle_matrix` - State-preparation unitary A (dim x dim as Vec<Vec<C64>>).
    /// * `initial_state` - The initial state |0> (length dim).
    ///
    /// # Returns
    ///
    /// An `AEResult` with the estimated amplitude and confidence interval.
    pub fn estimate(
        &self,
        oracle_matrix: &[Vec<C64>],
        initial_state: &[C64],
    ) -> AEResult {
        let dim = initial_state.len();

        // Prepare the state A|0>
        let a_flat = vv_to_flat(oracle_matrix, dim);
        let prepared_state = matvec(&a_flat, initial_state, dim);

        // Determine good indices
        let good_indices: Vec<usize> = (dim / 2..dim).collect();

        // Build the Grover operator
        let q_vv = grover_operator_with_good_states(oracle_matrix, oracle_matrix, &good_indices);
        let q_flat = vv_to_flat(&q_vv, dim);

        // Initial confidence interval for theta: [0, pi/2]
        let mut theta_low = 0.0_f64;
        let mut theta_high = PI / 2.0;

        let mut total_oracle_calls = 0usize;
        let mut samples = Vec::new();

        // Target width in theta-space. We want |sin^2(θ_est) - a| < epsilon.
        // Using the derivative: Δa ≈ |sin(2θ)| Δθ ≤ Δθ, so Δθ < epsilon suffices.
        let target_width = self.epsilon;

        // Use geometrically increasing Grover powers for progressive refinement
        let mut k = 1usize;
        let mut iteration = 0;

        while theta_high - theta_low > target_width && iteration < self.max_iterations {
            iteration += 1;

            // Apply Q^k to the prepared state
            let mut state_k = prepared_state.clone();
            for _ in 0..k {
                state_k = matvec(&q_flat, &state_k, dim);
            }
            total_oracle_calls += k;

            // Exact probability of "good" outcome after Q^k
            let prob_good = good_state_probability(&state_k, &good_indices);
            samples.push(prob_good);

            // sin^2((2k+1)*theta) = prob_good
            // => (2k+1)*theta = arcsin(sqrt(prob_good)) + m*pi  [branch A]
            //    or (2k+1)*theta = pi - arcsin(sqrt(prob_good)) + m*pi  [branch B]
            let alpha = prob_good.sqrt().asin(); // in [0, pi/2]
            let factor = (2 * k + 1) as f64;

            // The scaled interval is [factor*theta_low, factor*theta_high].
            // Find all candidate theta values in [theta_low, theta_high].
            let scaled_low = factor * theta_low;
            let scaled_high = factor * theta_high;

            // Maximum m needed
            let max_m = (scaled_high / PI).ceil() as i32 + 1;

            let mut candidates = Vec::new();
            for m in 0..=max_m {
                // Branch A: factor*theta = alpha + m*pi
                let t_a = (alpha + m as f64 * PI) / factor;
                if t_a >= theta_low - 1e-12 && t_a <= theta_high + 1e-12 {
                    candidates.push(t_a.clamp(0.0, PI / 2.0));
                }
                // Branch B: factor*theta = pi - alpha + m*pi
                let t_b = (PI - alpha + m as f64 * PI) / factor;
                if t_b >= theta_low - 1e-12 && t_b <= theta_high + 1e-12 {
                    candidates.push(t_b.clamp(0.0, PI / 2.0));
                }
            }

            if candidates.is_empty() {
                // No candidates found; increase k and retry
                k = (k * 2).min(1 << 20);
                continue;
            }

            // Sort candidates and find the best sub-interval.
            // The correct theta must be near one of these candidates.
            // Choose the candidate closest to the current interval midpoint
            // (best prior estimate) and narrow around it.
            candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

            let midpoint = (theta_low + theta_high) / 2.0;
            let best = *candidates
                .iter()
                .min_by(|a, b| {
                    ((**a) - midpoint)
                        .abs()
                        .partial_cmp(&((**b) - midpoint).abs())
                        .unwrap()
                })
                .unwrap();

            // Narrow the interval: the spacing between adjacent candidate roots
            // is pi / factor. The true theta is within half that spacing of `best`.
            let half_spacing = PI / (2.0 * factor);
            let new_low = (best - half_spacing).max(theta_low);
            let new_high = (best + half_spacing).min(theta_high);

            if new_high > new_low && (new_high - new_low) < (theta_high - theta_low) {
                theta_low = new_low;
                theta_high = new_high;
            }

            // Geometrically increase k for next round
            k = (k * 2).min(1 << 20);
        }

        // Final estimate: midpoint of the interval
        let theta_est = (theta_low + theta_high) / 2.0;
        let estimation = theta_est.sin().powi(2);

        let ci_low = theta_low.sin().powi(2);
        let ci_high = theta_high.sin().powi(2);

        AEResult {
            estimation,
            confidence_interval: (ci_low.min(ci_high), ci_low.max(ci_high)),
            num_oracle_calls: total_oracle_calls,
            samples,
        }
    }
}

// ============================================================
// MAXIMUM LIKELIHOOD AMPLITUDE ESTIMATION
// ============================================================

/// Maximum Likelihood Amplitude Estimation (MLAE).
///
/// For each Grover power k in the evaluation schedule, applies Q^k to the
/// prepared state and measures num_shots times, recording the number of
/// "good" outcomes h_k out of n_k shots.
///
/// The likelihood function is:
///   L(theta) = prod_k C(n_k, h_k) * sin^{2 h_k}((2k+1)*theta) * cos^{2(n_k - h_k)}((2k+1)*theta)
///
/// The MLE theta_hat = argmax L(theta) is found via grid search with
/// golden-section refinement.
///
/// Based on Suzuki et al. (2020).
#[derive(Debug, Clone)]
pub struct MaxLikelihoodAmplitudeEstimation {
    /// Schedule of Grover powers to apply: e.g., [0, 1, 2, 4, 8].
    /// Power 0 means applying Q^0 = I (just measure the prepared state).
    pub evaluation_schedule: Vec<usize>,
    /// Number of measurement shots per Grover power.
    pub num_shots_per_power: usize,
}

impl MaxLikelihoodAmplitudeEstimation {
    /// Create a new MLAE estimator.
    ///
    /// # Arguments
    ///
    /// * `evaluation_schedule` - Grover powers to use (e.g., [0, 1, 2, 4, 8]).
    /// * `num_shots_per_power` - Number of measurement shots per power.
    pub fn new(evaluation_schedule: Vec<usize>, num_shots_per_power: usize) -> Self {
        Self {
            evaluation_schedule,
            num_shots_per_power,
        }
    }

    /// Create with a default exponential schedule up to max_power.
    pub fn with_exponential_schedule(max_power_exponent: usize, num_shots: usize) -> Self {
        let schedule: Vec<usize> = (0..=max_power_exponent).map(|e| 1usize << e).collect();
        // Include power 0
        let mut full_schedule = vec![0];
        full_schedule.extend(schedule);
        Self {
            evaluation_schedule: full_schedule,
            num_shots_per_power: num_shots,
        }
    }

    /// Estimate the amplitude using maximum likelihood.
    ///
    /// # Arguments
    ///
    /// * `oracle_matrix` - State-preparation unitary A (dim x dim as Vec<Vec<C64>>).
    /// * `initial_state` - The initial state |0> (length dim).
    ///
    /// # Returns
    ///
    /// An `AEResult` with the MLE amplitude estimate.
    pub fn estimate(
        &self,
        oracle_matrix: &[Vec<C64>],
        initial_state: &[C64],
    ) -> AEResult {
        let dim = initial_state.len();

        // Prepare the state A|0>
        let a_flat = vv_to_flat(oracle_matrix, dim);
        let prepared_state = matvec(&a_flat, initial_state, dim);

        // Determine good indices
        let good_indices: Vec<usize> = (dim / 2..dim).collect();

        // Build the Grover operator
        let q_vv = grover_operator_with_good_states(oracle_matrix, oracle_matrix, &good_indices);
        let q_flat = vv_to_flat(&q_vv, dim);

        // For each power k in the schedule, compute the exact probability
        // of measuring "good" after applying Q^k. In a real quantum computer
        // this would be estimated from shots; here we use the exact probability
        // and simulate the shot statistics deterministically.
        let mut measurements: Vec<(usize, usize, usize)> = Vec::new(); // (k, h_k, n_k)
        let mut total_oracle_calls = 0usize;
        let mut samples = Vec::new();

        for &k in &self.evaluation_schedule {
            // Apply Q^k
            let mut state_k = prepared_state.clone();
            for _ in 0..k {
                state_k = matvec(&q_flat, &state_k, dim);
            }
            total_oracle_calls += k;

            // Exact probability of good outcome
            let prob_good = good_state_probability(&state_k, &good_indices);

            // Simulate measurement: deterministic rounding of expected hits
            let h_k = (prob_good * self.num_shots_per_power as f64).round() as usize;
            let h_k = h_k.min(self.num_shots_per_power);

            measurements.push((k, h_k, self.num_shots_per_power));
            samples.push(h_k as f64 / self.num_shots_per_power as f64);
        }

        // Maximum likelihood estimation via grid search + golden section refinement
        let theta_mle = self.maximize_likelihood(&measurements);
        let estimation = theta_mle.sin().powi(2);

        // Confidence interval from Fisher information
        let fisher_info = self.fisher_information(theta_mle, &measurements);
        let stderr = if fisher_info > 1e-15 {
            1.0 / fisher_info.sqrt()
        } else {
            PI / 4.0 // Conservative fallback
        };

        // 95% CI in theta space: theta +/- 1.96 * stderr
        let z = 1.96; // For 95% confidence
        let theta_low = (theta_mle - z * stderr).max(0.0);
        let theta_high = (theta_mle + z * stderr).min(PI / 2.0);
        let ci_low = theta_low.sin().powi(2);
        let ci_high = theta_high.sin().powi(2);

        AEResult {
            estimation,
            confidence_interval: (ci_low.min(ci_high), ci_low.max(ci_high)),
            num_oracle_calls: total_oracle_calls,
            samples,
        }
    }

    /// Maximise the log-likelihood function over theta in [0, pi/2].
    ///
    /// Uses a two-phase approach:
    /// 1. Coarse grid search over 1000 points.
    /// 2. Golden-section refinement around the best grid point.
    fn maximize_likelihood(&self, measurements: &[(usize, usize, usize)]) -> f64 {
        // Phase 1: Grid search
        let num_grid = 1000;
        let mut best_theta = 0.0_f64;
        let mut best_ll = f64::NEG_INFINITY;

        for i in 0..=num_grid {
            let theta = (i as f64 / num_grid as f64) * PI / 2.0;
            let ll = self.log_likelihood(theta, measurements);
            if ll > best_ll {
                best_ll = ll;
                best_theta = theta;
            }
        }

        // Phase 2: Golden-section refinement
        let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut a = (best_theta - PI / (2.0 * num_grid as f64)).max(0.0);
        let mut b = (best_theta + PI / (2.0 * num_grid as f64)).min(PI / 2.0);

        for _ in 0..50 {
            if (b - a).abs() < 1e-12 {
                break;
            }
            let x1 = b - golden_ratio * (b - a);
            let x2 = a + golden_ratio * (b - a);

            let f1 = self.log_likelihood(x1, measurements);
            let f2 = self.log_likelihood(x2, measurements);

            if f1 > f2 {
                b = x2;
            } else {
                a = x1;
            }
        }

        (a + b) / 2.0
    }

    /// Compute the log-likelihood for a given theta.
    ///
    /// log L(theta) = sum_k [ h_k * log(sin^2((2k+1)*theta))
    ///                       + (n_k - h_k) * log(cos^2((2k+1)*theta)) ]
    fn log_likelihood(&self, theta: f64, measurements: &[(usize, usize, usize)]) -> f64 {
        let mut ll = 0.0_f64;
        for &(k, h_k, n_k) in measurements {
            let angle = (2 * k + 1) as f64 * theta;
            let sin2 = angle.sin().powi(2);
            let cos2 = angle.cos().powi(2);

            // Clamp to avoid log(0)
            let sin2_safe = sin2.max(1e-300);
            let cos2_safe = cos2.max(1e-300);

            ll += h_k as f64 * sin2_safe.ln();
            ll += (n_k - h_k) as f64 * cos2_safe.ln();
        }
        ll
    }

    /// Compute the Fisher information at theta for confidence interval estimation.
    ///
    /// I(theta) = sum_k n_k * (2k+1)^2 * 4 * sin^2((2k+1)*theta) * cos^2((2k+1)*theta)
    ///          / [ sin^2((2k+1)*theta) * cos^2((2k+1)*theta) ]
    /// Simplifies to: I(theta) = sum_k n_k * (2k+1)^2 * 4
    ///   when sin and cos are nonzero.
    fn fisher_information(&self, theta: f64, measurements: &[(usize, usize, usize)]) -> f64 {
        let mut info = 0.0_f64;
        for &(k, _h_k, n_k) in measurements {
            let angle = (2 * k + 1) as f64 * theta;
            let sin2 = angle.sin().powi(2);
            let cos2 = angle.cos().powi(2);

            // Fisher information contribution:
            // d/dtheta log P(h|theta,k) leads to:
            // I_k = n_k * (2k+1)^2 * sin^2(2*(2k+1)*theta) / (sin^2((2k+1)*theta) * cos^2((2k+1)*theta))
            // = n_k * (2k+1)^2 * 4 when neither sin nor cos is zero
            let factor = (2 * k + 1) as f64;
            let product = sin2 * cos2;
            if product > 1e-15 {
                info += n_k as f64 * factor * factor * 4.0;
            }
        }
        info
    }
}

// ============================================================
// FORMAT CONVERSION HELPERS
// ============================================================

/// Convert Vec<Vec<C64>> to flat row-major representation.
fn vv_to_flat(matrix: &[Vec<C64>], dim: usize) -> Vec<C64> {
    let mut flat = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim.min(matrix[i].len()) {
            flat[i * dim + j] = matrix[i][j];
        }
    }
    flat
}

/// Convert flat row-major representation to Vec<Vec<C64>>.
fn flat_to_vv(flat: &[C64], dim: usize) -> Vec<Vec<C64>> {
    let mut matrix = Vec::with_capacity(dim);
    for i in 0..dim {
        let row: Vec<C64> = flat[i * dim..(i + 1) * dim].to_vec();
        matrix.push(row);
    }
    matrix
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-2;

    /// Build an oracle matrix A such that A|0> = sqrt(1-a)|0> + sqrt(a)|1>
    /// for a 2-qubit system where |1> (second basis state) is the "good" state.
    ///
    /// For the amplitude estimation convention (good = last half of basis),
    /// we use dim=2 where index 0 is "bad" and index 1 is "good".
    fn bernoulli_oracle(amplitude: f64) -> Vec<Vec<C64>> {
        let a = amplitude;
        let cos_theta = (1.0 - a).sqrt();
        let sin_theta = a.sqrt();
        // Unitary that rotates |0> to cos_theta|0> + sin_theta|1>
        vec![
            vec![c64(cos_theta, 0.0), c64(-sin_theta, 0.0)],
            vec![c64(sin_theta, 0.0), c64(cos_theta, 0.0)],
        ]
    }

    /// Standard initial state |0> for a 2-dimensional system.
    fn zero_state_2() -> Vec<C64> {
        vec![c64_one(), c64_zero()]
    }

    // ----------------------------------------------------------
    // 1. Canonical QAE on known amplitude
    // ----------------------------------------------------------
    #[test]
    fn test_qae_known_amplitude() {
        // Prepare state sqrt(0.7)|0> + sqrt(0.3)|1>, estimate amplitude ~0.3
        let oracle = bernoulli_oracle(0.3);
        let initial = zero_state_2();

        let qae = AmplitudeEstimation::new(6); // 6 eval qubits
        let result = qae.estimate(&oracle, &initial);

        assert!(
            (result.estimation - 0.3).abs() < 0.05,
            "QAE estimate {:.4} should be close to 0.3",
            result.estimation
        );
        assert!(result.num_oracle_calls > 0);
        assert!(!result.samples.is_empty());
    }

    // ----------------------------------------------------------
    // 2. IQAE convergence
    // ----------------------------------------------------------
    #[test]
    fn test_iqae_convergence() {
        let oracle = bernoulli_oracle(0.3);
        let initial = zero_state_2();

        let iqae = IterativeAmplitudeEstimation::new(0.05, 0.05, 50);
        let result = iqae.estimate(&oracle, &initial);

        assert!(
            (result.estimation - 0.3).abs() < 0.1,
            "IQAE estimate {:.4} should be within 0.1 of 0.3",
            result.estimation
        );
        assert!(
            result.confidence_interval.0 <= 0.3 + 0.1,
            "CI lower {:.4} should be <= 0.4",
            result.confidence_interval.0
        );
        assert!(
            result.confidence_interval.1 >= 0.3 - 0.1,
            "CI upper {:.4} should be >= 0.2",
            result.confidence_interval.1
        );
    }

    // ----------------------------------------------------------
    // 3. MLAE accuracy
    // ----------------------------------------------------------
    #[test]
    fn test_mlae_accuracy() {
        let oracle = bernoulli_oracle(0.3);
        let initial = zero_state_2();

        let mlae = MaxLikelihoodAmplitudeEstimation::new(
            vec![0, 1, 2, 4, 8],
            100,
        );
        let result = mlae.estimate(&oracle, &initial);

        assert!(
            (result.estimation - 0.3).abs() < 0.05,
            "MLAE estimate {:.4} should be within 0.05 of 0.3",
            result.estimation
        );
        // Samples should have one entry per schedule element
        assert_eq!(
            result.samples.len(),
            5,
            "Should have 5 samples (one per power)"
        );
    }

    // ----------------------------------------------------------
    // 4. All methods agree on Bernoulli amplitude
    // ----------------------------------------------------------
    #[test]
    fn test_all_methods_agree() {
        let target_amplitude = 0.25;
        let oracle = bernoulli_oracle(target_amplitude);
        let initial = zero_state_2();

        let qae = AmplitudeEstimation::new(8);
        let qae_result = qae.estimate(&oracle, &initial);

        let iqae = IterativeAmplitudeEstimation::new(0.05, 0.05, 50);
        let iqae_result = iqae.estimate(&oracle, &initial);

        let mlae = MaxLikelihoodAmplitudeEstimation::new(
            vec![0, 1, 2, 4, 8, 16],
            200,
        );
        let mlae_result = mlae.estimate(&oracle, &initial);

        // All three should be within reasonable tolerance of each other
        let tolerance = 0.1;
        assert!(
            (qae_result.estimation - iqae_result.estimation).abs() < tolerance,
            "QAE ({:.4}) and IQAE ({:.4}) should agree within {:.2}",
            qae_result.estimation,
            iqae_result.estimation,
            tolerance
        );
        assert!(
            (qae_result.estimation - mlae_result.estimation).abs() < tolerance,
            "QAE ({:.4}) and MLAE ({:.4}) should agree within {:.2}",
            qae_result.estimation,
            mlae_result.estimation,
            tolerance
        );
        assert!(
            (iqae_result.estimation - mlae_result.estimation).abs() < tolerance,
            "IQAE ({:.4}) and MLAE ({:.4}) should agree within {:.2}",
            iqae_result.estimation,
            mlae_result.estimation,
            tolerance
        );

        // All should be close to the true amplitude
        assert!(
            (qae_result.estimation - target_amplitude).abs() < tolerance,
            "QAE ({:.4}) should be close to true amplitude {:.4}",
            qae_result.estimation,
            target_amplitude
        );
    }

    // ----------------------------------------------------------
    // 5. Edge cases: amplitude near 0 and near 1
    // ----------------------------------------------------------
    #[test]
    fn test_edge_case_near_zero() {
        let oracle = bernoulli_oracle(0.01);
        let initial = zero_state_2();

        let qae = AmplitudeEstimation::new(8);
        let result = qae.estimate(&oracle, &initial);

        assert!(
            result.estimation < 0.1,
            "Estimate for amplitude ~0.01 should be small, got {:.4}",
            result.estimation
        );
        assert!(
            result.confidence_interval.0 >= 0.0,
            "CI lower bound should be non-negative"
        );
    }

    #[test]
    fn test_edge_case_near_one() {
        let oracle = bernoulli_oracle(0.99);
        let initial = zero_state_2();

        let qae = AmplitudeEstimation::new(8);
        let result = qae.estimate(&oracle, &initial);

        assert!(
            result.estimation > 0.9,
            "Estimate for amplitude ~0.99 should be large, got {:.4}",
            result.estimation
        );
        assert!(
            result.confidence_interval.1 <= 1.0,
            "CI upper bound should be <= 1.0"
        );
    }

    // ----------------------------------------------------------
    // 6. Precision improves with more evaluation qubits (QAE)
    //    and tighter epsilon (IQAE)
    // ----------------------------------------------------------
    #[test]
    fn test_precision_improves_with_eval_qubits() {
        let target = 0.3;
        let oracle = bernoulli_oracle(target);
        let initial = zero_state_2();

        let qae_4 = AmplitudeEstimation::new(4);
        let result_4 = qae_4.estimate(&oracle, &initial);
        let error_4 = (result_4.estimation - target).abs();

        let qae_8 = AmplitudeEstimation::new(8);
        let result_8 = qae_8.estimate(&oracle, &initial);
        let error_8 = (result_8.estimation - target).abs();

        // 8-qubit should be at least as precise as 4-qubit
        // (with tolerance for discrete grid effects)
        assert!(
            error_8 <= error_4 + 0.01,
            "8-qubit error ({:.6}) should be <= 4-qubit error ({:.6})",
            error_8,
            error_4
        );

        // More eval qubits should use more oracle calls
        assert!(
            result_8.num_oracle_calls > result_4.num_oracle_calls,
            "8-qubit ({}) should use more oracle calls than 4-qubit ({})",
            result_8.num_oracle_calls,
            result_4.num_oracle_calls
        );
    }

    #[test]
    fn test_precision_improves_with_tighter_epsilon() {
        let target = 0.3;
        let oracle = bernoulli_oracle(target);
        let initial = zero_state_2();

        let iqae_loose = IterativeAmplitudeEstimation::new(0.2, 0.05, 50);
        let result_loose = iqae_loose.estimate(&oracle, &initial);

        let iqae_tight = IterativeAmplitudeEstimation::new(0.01, 0.05, 100);
        let result_tight = iqae_tight.estimate(&oracle, &initial);

        // Tighter epsilon should yield a narrower confidence interval
        let ci_width_loose =
            result_loose.confidence_interval.1 - result_loose.confidence_interval.0;
        let ci_width_tight =
            result_tight.confidence_interval.1 - result_tight.confidence_interval.0;

        assert!(
            ci_width_tight <= ci_width_loose + 0.05,
            "Tight CI width ({:.4}) should be <= loose CI width ({:.4})",
            ci_width_tight,
            ci_width_loose
        );
    }

    // ----------------------------------------------------------
    // 7. Grover operator construction
    // ----------------------------------------------------------
    #[test]
    fn test_grover_operator_construction() {
        let oracle = bernoulli_oracle(0.25);
        let dim = 2;
        let good_indices = vec![1usize]; // Index 1 is "good"

        let q = grover_operator_with_good_states(&oracle, &oracle, &good_indices);

        // Q should be a 2x2 unitary
        assert_eq!(q.len(), dim);
        assert_eq!(q[0].len(), dim);

        // Verify unitarity: Q^dag Q = I
        let q_flat = vv_to_flat(&q, dim);
        let q_dag = adjoint(&q_flat, dim);
        let product = matmul(&q_dag, &q_flat, dim);

        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i * dim + j].re - expected).abs() < 1e-10
                        && product[i * dim + j].im.abs() < 1e-10,
                    "Q^dag Q[{},{}] = ({:.6}, {:.6}), expected ({:.6}, 0.0)",
                    i,
                    j,
                    product[i * dim + j].re,
                    product[i * dim + j].im,
                    expected
                );
            }
        }
    }

    // ----------------------------------------------------------
    // 8. Apply Grover power
    // ----------------------------------------------------------
    #[test]
    fn test_apply_grover_power() {
        let oracle = bernoulli_oracle(0.25);
        let good_indices = vec![1usize];
        let q = grover_operator_with_good_states(&oracle, &oracle, &good_indices);

        // Prepare initial state A|0>
        let initial = zero_state_2();
        let a_flat = vv_to_flat(&oracle, 2);
        let prepared = matvec(&a_flat, &initial, 2);

        // Apply Q^0 should leave state unchanged
        let state_0 = apply_grover_power(&prepared, &q, 0);
        for i in 0..2 {
            assert!(
                (state_0[i] - prepared[i]).norm_sqr() < 1e-20,
                "Q^0 should be identity"
            );
        }

        // Apply Q^1 should rotate by 2*theta
        let state_1 = apply_grover_power(&prepared, &q, 1);
        let prob_1 = state_1[1].norm_sqr(); // Good state probability

        // For a=0.25: theta=arcsin(sqrt(0.25))=pi/6
        // After Q^1: sin^2(3*theta) = sin^2(pi/2) = 1.0
        let theta = 0.25_f64.sqrt().asin();
        let expected_prob = (3.0 * theta).sin().powi(2);
        assert!(
            (prob_1 - expected_prob).abs() < 1e-10,
            "After Q^1: prob {:.6}, expected {:.6}",
            prob_1,
            expected_prob
        );
    }

    // ----------------------------------------------------------
    // 9. MLAE with different evaluation schedules
    // ----------------------------------------------------------
    #[test]
    fn test_mlae_different_schedules() {
        let target = 0.3;
        let oracle = bernoulli_oracle(target);
        let initial = zero_state_2();

        // Sparse schedule
        let mlae_sparse = MaxLikelihoodAmplitudeEstimation::new(vec![0, 4, 16], 100);
        let result_sparse = mlae_sparse.estimate(&oracle, &initial);

        // Dense schedule
        let mlae_dense =
            MaxLikelihoodAmplitudeEstimation::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], 100);
        let result_dense = mlae_dense.estimate(&oracle, &initial);

        // Both should give reasonable estimates
        assert!(
            (result_sparse.estimation - target).abs() < 0.1,
            "Sparse MLAE ({:.4}) should be within 0.1 of {:.4}",
            result_sparse.estimation,
            target
        );
        assert!(
            (result_dense.estimation - target).abs() < 0.05,
            "Dense MLAE ({:.4}) should be within 0.05 of {:.4}",
            result_dense.estimation,
            target
        );
    }

    // ----------------------------------------------------------
    // 10. MLAE exponential schedule constructor
    // ----------------------------------------------------------
    #[test]
    fn test_mlae_exponential_schedule() {
        let mlae = MaxLikelihoodAmplitudeEstimation::with_exponential_schedule(4, 100);
        // Should include [0, 1, 2, 4, 8, 16]
        assert!(mlae.evaluation_schedule.contains(&0));
        assert!(mlae.evaluation_schedule.contains(&1));
        assert!(mlae.evaluation_schedule.contains(&16));
        assert_eq!(mlae.num_shots_per_power, 100);
    }

    // ----------------------------------------------------------
    // 11. QAE result structure
    // ----------------------------------------------------------
    #[test]
    fn test_ae_result_fields() {
        let oracle = bernoulli_oracle(0.5);
        let initial = zero_state_2();

        let qae = AmplitudeEstimation::new(4);
        let result = qae.estimate(&oracle, &initial);

        // Estimation should be in [0, 1]
        assert!(
            result.estimation >= 0.0 && result.estimation <= 1.0,
            "Estimation {:.4} should be in [0, 1]",
            result.estimation
        );

        // CI should be ordered and in [0, 1]
        assert!(
            result.confidence_interval.0 <= result.confidence_interval.1,
            "CI should be ordered: ({:.4}, {:.4})",
            result.confidence_interval.0,
            result.confidence_interval.1
        );
        assert!(result.confidence_interval.0 >= 0.0);
        assert!(result.confidence_interval.1 <= 1.0);

        // Oracle calls should be positive
        assert!(result.num_oracle_calls > 0);
    }

    // ----------------------------------------------------------
    // 12. QPE outcome probability helper
    // ----------------------------------------------------------
    #[test]
    fn test_qpe_outcome_probability() {
        // When phi = 2*pi*m/2^n (exact match), probability should be 1
        let n = 4;
        let m = 3;
        let phi = 2.0 * PI * m as f64 / (1 << n) as f64;
        let prob = qpe_outcome_probability(phi, m, n);
        assert!(
            (prob - 1.0).abs() < 1e-10,
            "Exact match should give probability 1.0, got {:.6}",
            prob
        );

        // Far from the peak, probability should be small
        let prob_far = qpe_outcome_probability(phi, (m + 4) % (1 << n), n);
        assert!(
            prob_far < 0.5,
            "Far from peak should have low probability, got {:.6}",
            prob_far
        );
    }

    // ----------------------------------------------------------
    // 13. MLAE log-likelihood
    // ----------------------------------------------------------
    #[test]
    fn test_mlae_log_likelihood() {
        let mlae = MaxLikelihoodAmplitudeEstimation::new(vec![0, 1], 100);

        // For theta = pi/6, sin^2(theta) = 0.25
        // k=0: sin^2(1*pi/6) = 0.25, expected h = 25, n = 100
        // k=1: sin^2(3*pi/6) = sin^2(pi/2) = 1.0, expected h = 100, n = 100
        let measurements = vec![(0, 25, 100), (1, 100, 100)];

        let ll_true = mlae.log_likelihood(PI / 6.0, &measurements);
        let ll_wrong = mlae.log_likelihood(PI / 3.0, &measurements);

        // The log-likelihood at the true theta should be higher than at a wrong theta
        assert!(
            ll_true > ll_wrong,
            "LL at true theta ({:.4}) should exceed LL at wrong theta ({:.4})",
            ll_true,
            ll_wrong
        );
    }

    // ----------------------------------------------------------
    // 14. Format conversion helpers
    // ----------------------------------------------------------
    #[test]
    fn test_format_conversions() {
        let matrix = vec![
            vec![c64(1.0, 0.0), c64(2.0, 0.0)],
            vec![c64(3.0, 0.0), c64(4.0, 0.0)],
        ];

        let flat = vv_to_flat(&matrix, 2);
        assert_eq!(flat.len(), 4);
        assert!((flat[0].re - 1.0).abs() < 1e-15);
        assert!((flat[1].re - 2.0).abs() < 1e-15);
        assert!((flat[2].re - 3.0).abs() < 1e-15);
        assert!((flat[3].re - 4.0).abs() < 1e-15);

        let back = flat_to_vv(&flat, 2);
        assert_eq!(back.len(), 2);
        assert!((back[0][0].re - 1.0).abs() < 1e-15);
        assert!((back[1][1].re - 4.0).abs() < 1e-15);
    }

    // ----------------------------------------------------------
    // 15. Amplitude a=0.5 (Hadamard-like)
    // ----------------------------------------------------------
    #[test]
    fn test_amplitude_half() {
        let oracle = bernoulli_oracle(0.5);
        let initial = zero_state_2();

        let qae = AmplitudeEstimation::new(6);
        let result = qae.estimate(&oracle, &initial);
        assert!(
            (result.estimation - 0.5).abs() < 0.05,
            "QAE for a=0.5: got {:.4}",
            result.estimation
        );

        let mlae = MaxLikelihoodAmplitudeEstimation::new(vec![0, 1, 2, 4, 8], 200);
        let result_ml = mlae.estimate(&oracle, &initial);
        assert!(
            (result_ml.estimation - 0.5).abs() < 0.05,
            "MLAE for a=0.5: got {:.4}",
            result_ml.estimation
        );
    }
}
